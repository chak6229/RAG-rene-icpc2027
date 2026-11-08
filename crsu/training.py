"""Fixed-epoch prototype training; role checks, no evaluation-driven early stop."""
from dataclasses import dataclass
from typing import Sequence
import numpy as np
import torch

from .model import EvidencePosterior, target_nll


@dataclass(frozen=True)
class EvidenceInput:
    """Only label-free inputs; no IDs/URLs, gold identity, RR labels or split roles."""
    document_features: np.ndarray      # [D,Fd]
    action_evidence: np.ndarray        # [D,A,Fa]
    query_features: np.ndarray         # [Fq]


@dataclass(frozen=True)
class TrainingExample:
    inputs: EvidenceInput
    target_state: int                  # local union position or D for OUTSIDE
    role: str


def collate(inputs: Sequence[EvidenceInput], device='cpu'):
    if not inputs:
        raise ValueError('nonempty query batch required')
    fd = inputs[0].document_features.shape[1]
    fa = inputs[0].action_evidence.shape[2]
    fq = inputs[0].query_features.shape[0]
    d = max(1, max(len(x.document_features) for x in inputs))
    a = max(1, max(x.action_evidence.shape[1] for x in inputs))
    documents = np.zeros((len(inputs), d, fd), np.float32)
    actions = np.zeros((len(inputs), d, a, fa), np.float32)
    queries = np.zeros((len(inputs), fq), np.float32)
    dm = np.zeros((len(inputs), d), bool)
    am = np.zeros((len(inputs), a), bool)
    for i, x in enumerate(inputs):
        di, ai = len(x.document_features), x.action_evidence.shape[1]
        if x.document_features.shape != (di, fd) or x.action_evidence.shape != (di, ai, fa) or x.query_features.shape != (fq,):
            raise ValueError('aligned documents and fixed feature dimensions required')
        if not all(np.isfinite(v).all() for v in (x.document_features, x.action_evidence, x.query_features)):
            raise ValueError('finite input features required')
        documents[i, :di] = x.document_features
        actions[i, :di, :ai] = x.action_evidence
        queries[i] = x.query_features
        dm[i, :di] = True
        am[i, :ai] = True
    return {k: torch.as_tensor(v, device=device) for k, v in {
        'document_features': documents, 'action_evidence': actions, 'query_features': queries,
        'document_mask': dm, 'action_mask': am}.items()}


def fit_fixed_epochs(model: EvidencePosterior, examples: Sequence[TrainingExample], *,
                     epochs=30, batch_size=16, learning_rate=.001, seed=101, device='cpu'):
    """NLL per QUERY, no class reweighting or candidate-as-independent sampling.

    Seed model initialization separately before constructing EvidencePosterior.
    Production preprocessing/split/cache provenance still needs an audited builder.
    Returns training losses only; does not choose checkpoints using calibration.
    """
    if not examples or any(x.role != 'utility_train' for x in examples):
        raise ValueError('only utility_train queries may fit the evidence network')
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError('positive training hyperparameters required')
    for example in examples:
        if not 0 <= example.target_state <= len(example.inputs.document_features):
            raise ValueError('target must address its own union or OUTSIDE')
    rng = np.random.default_rng(seed)
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    history = []
    for _ in range(epochs):
        total = 0.
        order = rng.permutation(len(examples))
        for start in range(0, len(order), batch_size):
            batch = [examples[i] for i in order[start:start+batch_size]]
            values = collate([x.inputs for x in batch], device)
            outside_slot = values['document_features'].shape[1]
            targets = torch.tensor([outside_slot if x.target_state == len(x.inputs.document_features)
                                    else x.target_state for x in batch], dtype=torch.long, device=device)
            optimizer.zero_grad()
            loss = target_nll(model(**values), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach())*len(batch)
        history.append(total/len(examples))
    model.eval()
    return history


@torch.inference_mode()
def predict_probabilities(model: EvidencePosterior, inputs: Sequence[EvidenceInput], device='cpu'):
    """Return unpadded [local docs,OUTSIDE] distributions, aligned with each union."""
    if not inputs:
        return []
    model.to(device).eval()
    values = collate(inputs, device)
    p = model(**values).exp().cpu().numpy().astype(np.float64)
    result = []
    for i, x in enumerate(inputs):
        v = np.r_[p[i, :len(x.document_features)], p[i, -1]]
        result.append(v / v.sum())
    return result
