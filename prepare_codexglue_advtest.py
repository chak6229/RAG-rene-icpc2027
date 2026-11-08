#!/usr/bin/env python3
"""Build CodeXGLUE AdvTest validation/test JSONL files from the official archives."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_urls(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="Extracted dataset directory containing test_code.jsonl.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = {}
    with (args.source / "test_code.jsonl").open("r", encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            records[item["url"]] = item

    args.output.mkdir(parents=True, exist_ok=True)
    offset = 0
    statistics = {}
    for split in ("valid", "test"):
        urls = read_urls(args.source / f"{split}.txt")
        missing = [url for url in urls if url not in records]
        if missing:
            raise RuntimeError(f"{len(missing)} {split} URLs are absent from test_code.jsonl")
        destination = args.output / f"{split}.jsonl"
        with destination.open("w", encoding="utf-8") as stream:
            for index, url in enumerate(urls, offset):
                item = dict(records[url])
                item["idx"] = index
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
        statistics[split] = len(urls)
        offset += len(urls)

    (args.output / "split_manifest.json").write_text(
        json.dumps({"source": str(args.source.resolve()), "counts": statistics},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(statistics))


if __name__ == "__main__":
    main()
