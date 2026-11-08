"""Separate span generation from raw-logit, full-vocabulary teacher-forcing scoring."""
from __future__ import annotations

import hashlib
import math
import random
from dataclasses import asdict, dataclass

import torch

from .common import ROOT


SCORING_CONTRACT = {
    "entropy": "FP32 softmax/log-softmax over the entire unprocessed model vocabulary",
    "conditioning": "teacher forcing on masked query and generated span",
    "label": "<extra_id_0> {decoded span} <extra_id_1> plus tokenizer EOS",
    "primary_entropy_mask": "all non-padding label positions, including both sentinels and EOS",
    "secondary_entropy_mask": "lexical span tokens only; separate diagnostic field",
    "dropout_during_generation_and_scoring": False,
    "entr_selection": "ascending primary full-vocabulary entropy; position breaks ties",
    "prob_selection": "descending teacher-forced mean label log probability; position breaks ties",
    "prob_status": "declared operationalization; author did not publish a PROB implementation",
    "rand_selection": "uniform positions without replacement from exactly the same cached candidate pool",
    "generated_scores_used_for_entropy": False,
    "identity_padding": False,
    "tokenizer": "T5Tokenizer legacy=True; local t5-base spiece.model; vocabulary checked against saved fast tokenizer",
}


def load_tokenizer(checkpoint):
    from transformers import AutoTokenizer, T5Tokenizer
    local = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    tokenizer = T5Tokenizer(vocab_file=str(ROOT / "models/t5-base/spiece.model"), extra_ids=100, legacy=True)
    if tokenizer.get_vocab() != local.get_vocab():
        raise ValueError("source-family SentencePiece vocabulary differs from the saved checkpoint tokenizer")
    return tokenizer


@dataclass(frozen=True)
class GenerationSettings:
    seed: int = 101
    do_sample: bool = False
    max_new_tokens: int = 19
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    position_policy: str = "source_1_to_n"
    batch_size: int = 8

    def contract(self):
        return {**asdict(self), "num_beams": 1, "num_return_sequences": 1,
                "source_choice": "explicit modern greedy defaults; author generate() options were not recorded",
                "paper_difference": "paper n+1 positions and sampling/m=10 can be requested explicitly; default follows public JSON position loop",
                "same_positions_for_all_selection_policies": True}


def insertion_inputs(query: str, *, position_policy="source_1_to_n"):
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a nonempty string")
    if "<extra_id_" in query:
        raise ValueError("query already contains reserved sentinel text")
    tokens = query.split()
    if position_policy == "source_1_to_n":
        positions = range(1, len(tokens) + 1)
    elif position_policy == "paper_0_to_n":
        positions = range(len(tokens) + 1)
    else:
        raise ValueError("unknown insertion-position policy")
    return [{"position": p, "masked_query": " ".join(tokens[:p] + ["<extra_id_0>"] + tokens[p:])}
            for p in positions]


def full_vocab_statistics(logits, labels, *, lexical_mask=None):
    """Mean conditional token entropy, distinct from generated-token surprisal."""
    if logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise ValueError("expected logits [B,T,V] and labels [B,T]")
    if not torch.isfinite(logits).all():
        raise FloatingPointError("nonfinite raw teacher-forcing logits")
    valid = labels.ne(-100)
    if not valid.any(dim=1).all():
        raise ValueError("each sequence needs at least one valid label")
    if ((labels[valid] < 0) | (labels[valid] >= logits.shape[-1])).any():
        raise ValueError("label outside model vocabulary")
    with torch.autocast(device_type=logits.device.type, enabled=False):
        logp = torch.log_softmax(logits.float(), dim=-1)
        p = logp.exp()
        entropy = -(p * logp).sum(dim=-1)
        target_logp = logp.gather(-1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        maxp = p.amax(dim=-1)
        mask = valid.to(torch.float32)
        counts = mask.sum(dim=1)
        result = {"full_vocab_entropy": (entropy * mask).sum(dim=1) / counts,
                  "label_mean_log_probability": (target_logp * mask).sum(dim=1) / counts,
                  "label_sum_log_probability": (target_logp * mask).sum(dim=1),
                  "mean_max_vocab_probability": (maxp * mask).sum(dim=1) / counts,
                  "label_token_count": counts.to(torch.long)}
        if lexical_mask is not None:
            if lexical_mask.shape != labels.shape or (lexical_mask & ~valid).any():
                raise ValueError("lexical mask must be a subset of valid labels")
            span = lexical_mask.to(torch.float32)
            span_counts = span.sum(dim=1)
            result.update({"span_token_count": span_counts.to(torch.long),
                           "span_full_vocab_entropy": (entropy * span).sum(dim=1) / span_counts.clamp_min(1),
                           "span_mean_log_probability": (target_logp * span).sum(dim=1) / span_counts.clamp_min(1)})
    return result


def _model_device(model):
    return next(model.parameters()).device


def _inputs(tokenizer, masked_queries, device):
    inputs = tokenizer(masked_queries, padding=True, truncation=False, return_tensors="pt")
    if inputs.input_ids.shape[1] > 512:
        raise ValueError("T5 input exceeds declared limit; silent truncation would change insertion positions")
    sentinel_id = tokenizer.convert_tokens_to_ids("<extra_id_0>")
    if not inputs.input_ids.eq(sentinel_id).sum(dim=1).eq(1).all():
        raise ValueError("each masked input must retain exactly one extra_id_0")
    return {key: value.to(device) for key, value in inputs.items()}


def teacher_forced_scores(model, tokenizer, masked_queries, spans, *, batch_size=8):
    if len(masked_queries) != len(spans) or batch_size < 1:
        raise ValueError("paired masked inputs/spans and positive batch size are required")
    device = _model_device(model)
    was_training = model.training
    model.eval()
    scores = []
    sentinel0 = tokenizer.convert_tokens_to_ids("<extra_id_0>")
    sentinel1 = tokenizer.convert_tokens_to_ids("<extra_id_1>")
    excluded = {sentinel0, sentinel1, tokenizer.eos_token_id, tokenizer.pad_token_id}
    try:
        with torch.inference_mode():
            for start in range(0, len(spans), batch_size):
                current = spans[start:start + batch_size]
                if any(not isinstance(span, str) or "<extra_id_" in span for span in current):
                    raise ValueError("spans must be decoded lexical strings, without sentinels")
                inputs = _inputs(tokenizer, masked_queries[start:start + batch_size], device)
                labels_raw = tokenizer([f"<extra_id_0> {span} <extra_id_1>" for span in current],
                                       padding=True, truncation=False, return_tensors="pt")
                labels = labels_raw.input_ids.to(device)
                label_valid = labels_raw.attention_mask.to(device).bool()
                if not (labels.eq(sentinel0).sum(1).eq(1) & labels.eq(sentinel1).sum(1).eq(1)).all():
                    raise ValueError("teacher-forced labels lost wrapper sentinels")
                labels = labels.masked_fill(~label_valid, -100)
                lexical_mask = label_valid.clone()
                for token_id in excluded:
                    lexical_mask &= labels.ne(token_id)
                outputs = model(**inputs, labels=labels, decoder_attention_mask=label_valid)
                metrics = full_vocab_statistics(outputs.logits, labels, lexical_mask=lexical_mask)
                for i in range(len(current)):
                    record = {name: float(value[i].cpu()) for name, value in metrics.items()}
                    for key in ("label_token_count", "span_token_count"):
                        record[key] = int(record[key])
                    if record["span_token_count"] == 0:
                        record["span_full_vocab_entropy"] = None
                        record["span_mean_log_probability"] = None
                    scores.append(record)
    finally:
        model.train(was_training)
    return scores


def generate_candidates(model, tokenizer, query_id, query, settings=GenerationSettings()):
    if settings.batch_size < 1 or settings.max_new_tokens < 1:
        raise ValueError("generation batch size and token budget must be positive")
    records = insertion_inputs(query, position_policy=settings.position_policy)
    device = _model_device(model)
    was_training = model.training
    model.eval()
    # Cache construction order/batch size and seed are declared. Selection never regenerates candidates.
    torch.manual_seed(settings.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(settings.seed)
    try:
        with torch.inference_mode():
            for start in range(0, len(records), settings.batch_size):
                batch = records[start:start + settings.batch_size]
                inputs = _inputs(tokenizer, [r["masked_query"] for r in batch], device)
                kwargs = dict(do_sample=settings.do_sample, num_beams=1, num_return_sequences=1,
                              max_new_tokens=settings.max_new_tokens, output_scores=False,
                              return_dict_in_generate=False, use_cache=True,
                              decoder_start_token_id=tokenizer.pad_token_id,
                              eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
                if settings.do_sample:
                    kwargs.update(temperature=settings.temperature, top_k=settings.top_k, top_p=settings.top_p)
                generated = model.generate(**inputs, **kwargs)
                for record, ids in zip(batch, generated):
                    span = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
                    tokens = query.split()
                    p = record["position"]
                    rewrite = " ".join(tokens[:p] + ([span] if span else []) + tokens[p:])
                    record.update({"query_id": query_id, "span": span, "rewrite": rewrite,
                                   "is_identity": rewrite == " ".join(tokens)})
        metrics = teacher_forced_scores(model, tokenizer, [r["masked_query"] for r in records],
                                       [r["span"] for r in records], batch_size=settings.batch_size)
        for record, metric in zip(records, metrics):
            record.update(metric)
    finally:
        model.train(was_training)
    return records


def select_candidates(records, *, policy, k=3, seed=101):
    if k < 1 or not records:
        raise ValueError("nonempty candidate pool and positive k are required")
    query_ids = {row["query_id"] for row in records}
    positions = [row["position"] for row in records]
    if len(query_ids) != 1 or len(set(positions)) != len(positions):
        raise ValueError("selection requires one query and unique insertion positions")
    if any(not math.isfinite(float(row[field])) for row in records
           for field in ("full_vocab_entropy", "label_mean_log_probability")):
        raise ValueError("candidate scores must be finite")
    if policy == "ENTR":
        ordered = sorted(records, key=lambda r: (r["full_vocab_entropy"], r["position"]))
    elif policy == "PROB":
        ordered = sorted(records, key=lambda r: (-r["label_mean_log_probability"], r["position"]))
    elif policy == "RAND":
        query_id = next(iter(query_ids))
        digest = hashlib.sha256(f"{seed}\0{query_id}".encode("utf-8")).digest()
        ordered = random.Random(int.from_bytes(digest, "big")).sample(sorted(records, key=lambda r: r["position"]), len(records))
    else:
        raise ValueError("unknown selection policy")
    return ordered[:min(k, len(ordered))]
