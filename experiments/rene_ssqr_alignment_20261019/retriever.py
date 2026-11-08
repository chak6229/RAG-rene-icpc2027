"""Published SSQR CodeBERT representation and full in-batch CE objective."""
from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .common import read_json

REPRESENTATION = {
    "schema_version": 1,
    "representation": "roberta_pooler_output",
    "normalization": "none",
    "similarity": "unnormalized_dot_product",
    "objective": "query_to_code_in_batch_cross_entropy",
    "source_repository": "RedSmallPanda/SSQR",
    "source_commit": "7fe5d780740eb4348500658ce6decf188a542d2b",
    "source_path": "Query Reformulation/code/model.py",
}


def pooler_vectors(outputs):
    vectors = getattr(outputs, "pooler_output", None)
    if vectors is None:
        raise ValueError("SSQR retriever requires a pooler output; CLS fallback is forbidden")
    if vectors.ndim != 2:
        raise ValueError("pooler output must be a matrix")
    return vectors


def full_batch_loss(model, code_ids, query_ids, *, pad_token_id=1):
    if code_ids.shape != query_ids.shape or code_ids.ndim != 2:
        raise ValueError("paired inputs must have identical [batch,length] dimensions")
    if len(code_ids) < 2:
        raise ValueError("in-batch training requires at least two code/query pairs")
    ids = torch.cat([code_ids, query_ids], dim=0)
    pooled = pooler_vectors(model(input_ids=ids, attention_mask=ids.ne(pad_token_id)))
    code_vectors, query_vectors = pooled[:len(code_ids)], pooled[len(code_ids):]
    # Use FP32 scores/CE under mixed precision; retain all B x B negatives.
    with torch.autocast(device_type=pooled.device.type, enabled=False):
        scores = query_vectors.float() @ code_vectors.float().T
        labels = torch.arange(len(code_ids), device=scores.device)
        loss = F.cross_entropy(scores, labels)
    return loss, code_vectors, query_vectors


def verify_checkpoint_contract(path: Path):
    marker = path / "ssqr_representation_contract.json"
    if not marker.is_file():
        raise ValueError("checkpoint lacks SSQR training/representation contract")
    actual = read_json(marker)
    if any(actual.get(key) != expected for key, expected in REPRESENTATION.items()):
        raise ValueError("checkpoint was not trained with the contracted SSQR pooler objective")
    return actual


def encode_texts(model, tokenizer, texts, *, batch_size=32, max_length=256, device="cuda", precision="float32"):
    if not texts:
        return np.empty((0, model.config.hidden_size), dtype=np.float32)
    was_training = model.training
    model.eval()
    values = []
    try:
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                inputs = tokenizer(list(texts[start:start + batch_size]), padding="max_length",
                                   truncation=True, max_length=max_length, return_tensors="pt")
                inputs = {key: value.to(device) for key, value in inputs.items()}
                context = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" and precision == "bfloat16" else contextlib.nullcontext()
                with context:
                    output = pooler_vectors(model(**inputs))
                values.append(output.float().cpu().numpy())
    finally:
        model.train(was_training)
    return np.concatenate(values)


def dot_product_scores(query_vectors, code_vectors):
    queries, codes = np.asarray(query_vectors, dtype=np.float32), np.asarray(code_vectors, dtype=np.float32)
    if queries.ndim != 2 or codes.ndim != 2 or queries.shape[1] != codes.shape[1]:
        raise ValueError("query/code vectors must be matrices with a shared dimension")
    return queries @ codes.T
