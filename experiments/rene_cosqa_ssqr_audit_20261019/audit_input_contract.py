"""Check CoSQA field semantics and actual tokenizers, without loading model weights."""
from __future__ import annotations

import collections
import hashlib
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from discover_sources import ROOT, OUT


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**23), b""):
            h.update(chunk)
    return h.hexdigest()


def length_summary(values: list[int]) -> dict:
    ordered = sorted(values)
    return {"count": len(values), "min": min(values), "median": statistics.median(values),
            "p95_nearest_rank": ordered[(95 * len(ordered) + 99) // 100 - 1], "max": max(values)}


def token_lengths(tokenizer, texts: list[str]) -> list[int]:
    values = []
    for start in range(0, len(texts), 256):
        ids = tokenizer(texts[start:start + 256], truncation=False, padding=False,
                        add_special_tokens=True, verbose=False)["input_ids"]
        values.extend(map(len, ids))
    return values


def insertion_check(tokenizer, queries: list[str]) -> dict:
    mask_id = tokenizer.convert_tokens_to_ids("<extra_id_0>")
    all_lengths, source_lengths = [], []
    missing = {"128": 0, "512": 0}
    untruncated_bad = 0
    pending_texts, pending_positions = [], []

    def consume():
        nonlocal untruncated_bad
        if not pending_texts:
            return
        ids = tokenizer(pending_texts, truncation=False, padding=False, verbose=False)["input_ids"]
        for pos, seq in zip(pending_positions, ids):
            all_lengths.append(len(seq))
            if pos > 0:
                source_lengths.append(len(seq))
            untruncated_bad += seq.count(mask_id) != 1
        for limit in (128, 512):
            over = [text for text, seq in zip(pending_texts, ids) if len(seq) > limit]
            if over:
                cropped = tokenizer(over, max_length=limit, truncation=True, padding=False,
                                    verbose=False)["input_ids"]
                missing[str(limit)] += sum(seq.count(mask_id) != 1 for seq in cropped)
        pending_texts.clear()
        pending_positions.clear()

    for query in queries:
        words = query.split()
        for position in range(len(words) + 1):
            pending_texts.append(" ".join(words[:position] + ["<extra_id_0>"] + words[position:]))
            pending_positions.append(position)
            if len(pending_texts) == 256:
                consume()
    consume()
    return {"tokenizer_class": type(tokenizer).__name__, "is_fast": bool(tokenizer.is_fast),
            "sentinel_id": mask_id, "paper_all_boundaries": length_summary(all_lengths),
            "published_source_positions_1_to_n": length_summary(source_lengths),
            "sequences_over_128": sum(n > 128 for n in all_lengths),
            "sequences_over_512": sum(n > 512 for n in all_lengths),
            "untruncated_sentinel_count_not_one": untruncated_bad,
            "sentinel_count_not_one_after_truncation": missing}


def main():
    from transformers import AutoTokenizer, T5Tokenizer

    started = time.monotonic()
    data_path = ROOT / "imports/CodeXGLUE-main/Text-Code/NL-code-search-WebQuery/CoSQA/cosqa-all.json"
    rows = json.loads(data_path.read_text(encoding="utf-8"))
    code_map = json.loads((OUT / "selected_source/coclr/data/search/code_idx_map.txt").read_text(encoding="utf-8"))
    code_versions = collections.defaultdict(set)
    for row in rows:
        assert isinstance(row["doc"], str) and row["doc"].strip()
        assert isinstance(row["code_tokens"], str)
        assert isinstance(row["docstring_tokens"], str)
        code_versions[row["code"]].add(row["code_tokens"])
    assert set(code_versions) == set(code_map)
    assert all(len(versions) == 1 for versions in code_versions.values())
    ordered_codes = sorted(code_map, key=code_map.get)
    code_texts = [next(iter(code_versions[code])) for code in ordered_codes]
    queries = [row["doc"] for row in rows]
    schema = {
        "rows": len(rows), "query_field": "doc", "corpus_id_source": "code_idx_map.txt raw code key -> integer ID",
        "recommended_encoder_text": "existing code_tokens string, passed directly; no join over characters",
        "docstring_tokens_meaning": "paired function documentation, not the web search query",
        "doc_equals_documentation_after_whitespace_normalization": sum(r["doc"].split() == r["docstring_tokens"].split() for r in rows),
        "documentation_nonempty_rows": sum(bool(r["docstring_tokens"].strip()) for r in rows),
        "codes_with_inconsistent_code_tokens": sum(len(v) != 1 for v in code_versions.values()),
        "corpus_raw_codes": len(ordered_codes), "unique_encoder_code_texts": len(set(code_texts)),
        "direct_local_query_text_function_is_compatible": False,
        "incompatibility": "local fallback joins docstring_tokens expecting a list; CoSQA stores code documentation as a string and query in doc",
    }
    t5_path = ROOT / "models/t5-cqc-codenn-1m/best"
    # Our saved checkpoint has tokenizer.json only.  The public source's slow
    # T5Tokenizer needs a SentencePiece model; use the verified local T5 base
    # vocabulary and report that reconstruction explicitly.
    spiece_path = ROOT / "models/t5-base/spiece.model"
    t5_variants = {"published_source_tokenizer_family": T5Tokenizer(vocab_file=str(spiece_path), legacy=True),
                   "current_local_tokenizer": AutoTokenizer.from_pretrained(t5_path, local_files_only=True)}
    token_stats = {}
    for name, tokenizer in t5_variants.items():
        token_stats[name] = insertion_check(tokenizer, queries)
        token_stats[name]["plain_query_lengths"] = length_summary(token_lengths(tokenizer, queries))
        print(json.dumps({"stage": name, **token_stats[name]}, ensure_ascii=False), flush=True)
    cb_path = ROOT / "models/codebert-codexglue-2epoch/checkpoint-3000"
    cb_tokenizer = AutoTokenizer.from_pretrained(cb_path, local_files_only=True)
    code_lengths = token_lengths(cb_tokenizer, code_texts)
    raw_code_lengths = token_lengths(cb_tokenizer, ordered_codes)
    codebert = {"tokenizer_class": type(cb_tokenizer).__name__, "query_lengths": length_summary(token_lengths(cb_tokenizer, queries)),
                "code_tokens_lengths": length_summary(code_lengths), "code_tokens_over_256": sum(n > 256 for n in code_lengths),
                "raw_code_lengths": length_summary(raw_code_lengths), "raw_code_over_256": sum(n > 256 for n in raw_code_lengths)}
    inventory = []
    for directory in (t5_path, cb_path, ROOT / "models/t5-base"):
        for path in sorted(directory.iterdir()):
            if path.is_file() and ("token" in path.name or path.name in {"spiece.model", "vocab.json", "merges.txt"}):
                inventory.append({"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)})
    base_weights = ROOT / "models/codebert-base/pytorch_model.bin"
    base = {"path": base_weights.relative_to(ROOT).as_posix(), "exists": base_weights.exists()}
    if base_weights.exists():
        base.update(bytes=base_weights.stat().st_size, sha256=sha256(base_weights))
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "schema and tokenizer only; no neural model loading/inference, ranks or effects",
              "input_sha256": sha256(data_path), "schema_contract": schema, "t5": token_stats,
              "slow_tokenizer_reconstruction": {"sentencepiece_path": spiece_path.relative_to(ROOT).as_posix(),
                  "sha256": sha256(spiece_path), "legacy": True,
                  "reason": "local best checkpoint has no spiece.model; direct T5Tokenizer.from_pretrained failed; no environment or model files changed",
                  "scope": "published tokenizer class with local base vocabulary, not recovered author tokenizer/checkpoint"},
              "codebert": codebert, "tokenizer_files": inventory, "actual_codebert_base_weights": base,
              "elapsed_seconds": time.monotonic() - started}
    (OUT / "INPUT_CONTRACT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"schema": schema, "codebert": codebert, "elapsed_seconds": report["elapsed_seconds"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
