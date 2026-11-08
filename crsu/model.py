"""Lightweight set evidence network: original query stays the semantic anchor.

No document IDs, gold ranks, RR labels, or split roles enter forward().
Separate Bernoulli OUTSIDE mass from conditional document identity softmax.
"""
import torch
from torch import nn
from torch.nn import functional as F


def _pool(values, mask, dim):
    weights = mask.unsqueeze(-1)
    average = values.masked_fill(~weights, 0).sum(dim) / weights.sum(dim).clamp_min(1)
    maximum = values.masked_fill(~weights, float('-inf')).amax(dim)
    maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    return torch.cat((average, maximum), dim=-1)


class EvidencePosterior(nn.Module):
    """B=batch, D=padded union docs, A=padded available rewrite/fusion actions.

    document_features [B,D,Fd]: original query/code interaction + original scores.
    action_evidence [B,D,A,Fa]: actual action scores/presence/types; no slot ID.
    query_features [B,Fq]: label-free summary, corpus size, constraints, etc.
    masks [B,D], [B,A]. Use D>=1,A>=1 padded zeros even for empty unions/actions.
    Padding is ignored. Document ordering and non-reference action ordering do
    not carry learned positional embeddings. Fully empty union is OUTSIDE=1.
    """
    def __init__(self, document_dim, action_dim, query_dim, hidden=64):
        super().__init__()
        self.document_dim, self.action_dim, self.query_dim = document_dim, action_dim, query_dim
        self.action_encoder = nn.Sequential(nn.Linear(action_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.doc_encoder = nn.Sequential(nn.Linear(document_dim+2*hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.query_encoder = nn.Sequential(nn.Linear(query_dim, hidden), nn.GELU())
        self.doc_head = nn.Sequential(nn.Linear(4*hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.outside_head = nn.Sequential(nn.Linear(3*hidden+1, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, document_features, action_evidence, query_features, document_mask, action_mask):
        b, d, fd = document_features.shape
        if action_evidence.ndim != 4 or action_evidence.shape[:2] != (b, d):
            raise ValueError('action evidence must align with union documents')
        a, fa = action_evidence.shape[2:]
        if d < 1 or a < 1 or fd != self.document_dim or fa != self.action_dim:
            raise ValueError('feature dimensions or padding slots invalid')
        if query_features.shape != (b, self.query_dim):
            raise ValueError('query features mismatch')
        if document_mask.shape != (b, d) or action_mask.shape != (b, a):
            raise ValueError('mask dimensions mismatch')
        if document_mask.dtype != torch.bool or action_mask.dtype != torch.bool:
            raise ValueError('boolean masks required')
        for values in (document_features, action_evidence, query_features):
            if not torch.isfinite(values).all():
                raise ValueError('features must be finite, including zero padding')
        evidence = self.action_encoder(action_evidence)
        mask = action_mask[:, None, :].expand(b, d, a)
        action_context = _pool(evidence, mask, 2)
        h = self.doc_encoder(torch.cat((document_features, action_context), -1))
        global_h = _pool(h, document_mask, 1)
        q = self.query_encoder(query_features)
        context = torch.cat((global_h, q), -1)
        logits = self.doc_head(torch.cat((h, context[:, None].expand(b, d, -1)), -1)).squeeze(-1)
        logits = logits.masked_fill(~document_mask, float('-inf'))
        # Make the intermediate softmax finite for empty unions, then mask it.
        nonempty = document_mask.any(1)
        safe_logits = torch.where(nonempty[:, None], logits, torch.zeros_like(logits))
        log_conditional = F.log_softmax(safe_logits, dim=-1)
        count = document_mask.sum(1, keepdim=True).to(h.dtype).log1p()
        outside_logit = self.outside_head(torch.cat((context, count), -1))
        log_document_mass = F.logsigmoid(-outside_logit)
        log_outside_mass = F.logsigmoid(outside_logit)
        document_logs = (log_conditional+log_document_mass).masked_fill(~document_mask, float('-inf'))
        outside_logs = torch.where(nonempty[:, None], log_outside_mass, torch.zeros_like(log_outside_mass))
        return torch.cat((document_logs, outside_logs), dim=-1)


def target_nll(log_probabilities, targets):
    """Label-side query NLL; target is a valid union slot or the final OUTSIDE slot."""
    if targets.shape != (len(log_probabilities),) or targets.dtype != torch.long:
        raise ValueError('one integer target per query required')
    if torch.any(targets < 0) or torch.any(targets >= log_probabilities.shape[1]):
        raise ValueError('target outside probability support')
    loss = F.nll_loss(log_probabilities, targets)
    if not torch.isfinite(loss):
        raise ValueError('target points to a masked slot or nonfinite prediction')
    return loss
