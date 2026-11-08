#!/usr/bin/env python3
"""检查并按 CODEnn 原始顺序准备 SSQR CQC 训练及固定验证数据。"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
import re
from collections import Counter
from pathlib import Path

import tables


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def corrupt(query: str, rng: random.Random) -> tuple[str, str]:
    words = query.split()
    span_length = max(1, round(0.15 * len(words)))
    start = rng.randrange(len(words) - span_length + 1)
    span = words[start:start + span_length]
    source = " ".join(words[:start] + ["<extra_id_0>"] + words[start + span_length:])
    target = "<extra_id_0> " + " ".join(span) + " <extra_id_1>"
    return source, target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-samples", type=int, default=1_000_000)
    parser.add_argument("--validation-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--sample-count", type=int, default=20)
    args = parser.parse_args()
    if args.train_samples <= 0 or args.validation_samples <= 0:
        parser.error("sample counts must be positive")

    vocabulary = pickle.loads(args.vocab.read_bytes())
    if not all(isinstance(token, str) and isinstance(identifier, int)
               for token, identifier in vocabulary.items()):
        raise TypeError("vocab.desc.pkl is not a token-to-id dictionary")
    id_to_token = {int(identifier): str(token) for token, identifier in vocabulary.items()}
    rejected = Counter()
    train: list[tuple[int, str]] = []
    validation: list[tuple[int, str]] = []
    scanned = 0

    with tables.open_file(str(args.h5), mode="r") as h5:
        node_names = sorted(node._v_name for node in h5.root)
        if not {"indices", "phrases"}.issubset(node_names):
            raise ValueError(f"Unexpected HDF5 nodes: {node_names}")
        indices, phrases = h5.root.indices, h5.root.phrases
        previous_end = 0
        positions_contiguous = True
        for source_index, row in enumerate(indices):
            scanned += 1
            length, position = int(row["length"]), int(row["pos"])
            positions_contiguous &= position == previous_end
            previous_end = position + length
            if length <= 0:
                rejected["zero_length"] += 1
                continue
            identifiers = phrases[position:position + length]
            if any(int(value) not in id_to_token for value in identifiers):
                rejected["id_not_in_vocab"] += 1
                continue
            text = " ".join(id_to_token[int(value)] for value in identifiers).strip()
            if not text:
                rejected["empty_after_decode"] += 1
                continue
            if not re.search(r"[A-Za-z0-9]", text):
                rejected["special_symbols_only"] += 1
                continue
            destination = train if len(train) < args.train_samples else validation
            destination.append((source_index, text))
            if len(validation) >= args.validation_samples:
                break
        structure = {
            "nodes": node_names, "indices_shape": [int(value) for value in indices.shape],
            "indices_dtype": str(indices.dtype),
            "phrases_shape": [int(value) for value in phrases.shape],
            "phrases_dtype": str(phrases.dtype),
            "positions_contiguous_in_scanned_prefix": positions_contiguous,
        }

    if len(train) != args.train_samples or len(validation) != args.validation_samples:
        raise RuntimeError(f"insufficient data: train={len(train)}, validation={len(validation)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.txt"
    validation_path = args.output_dir / "valid.txt"
    fixed_path = args.output_dir / "valid_fixed_cqc.jsonl"
    train_path.write_text("\n".join(text for _, text in train) + "\n", encoding="utf-8")
    validation_path.write_text("\n".join(text for _, text in validation) + "\n", encoding="utf-8")
    rng = random.Random(args.seed + 1_000_003)
    with fixed_path.open("w", encoding="utf-8") as stream:
        for source_index, query in validation:
            source, target = corrupt(query, rng)
            stream.write(json.dumps({"source_index": source_index, "query": query,
                                     "source": source, "target": target},
                                    ensure_ascii=False) + "\n")

    specials = {token: identifier for token, identifier in vocabulary.items()
                if token in {"PAD", "UNK", "BOS", "EOS", "<pad>", "<unk>",
                             "<s>", "</s>", "<PAD>", "<UNK>", "<BOS>", "<EOS>"}}
    manifest = {
        "format_version": 2, "hdf5": structure,
        "vocabulary": {"mapping": "token_to_id", "size": len(vocabulary),
                       "special_tokens": specials,
                       "note": "id 0 is shared by <s> and </s> in the pickle; it does not occur in the selected phrase stream"},
        "selection": "first decodable descriptions in author/source HDF5 order",
        "scanned_source_rows": scanned, "decoded_rows": len(train) + len(validation),
        "discarded_rows": sum(rejected.values()), "discard_reasons": dict(rejected),
        "train": {"samples": len(train), "first_source_index": train[0][0],
                  "last_source_index": train[-1][0]},
        "validation": {"samples": len(validation),
                       "first_source_index": validation[0][0],
                       "last_source_index": validation[-1][0],
                       "fixed_corruption_seed": args.seed + 1_000_003,
                       "masking_unit": "whitespace-delimited natural-language words",
                       "masking": "one contiguous span, round(15% * word_count), minimum 1"},
        "samples": [text for _, text in train[:args.sample_count]],
        "sha256": {"source_h5": sha256(args.h5), "source_vocab": sha256(args.vocab),
                   "train": sha256(train_path), "validation": sha256(validation_path),
                   "fixed_validation_cqc": sha256(fixed_path)},
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
