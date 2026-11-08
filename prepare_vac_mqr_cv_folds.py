#!/usr/bin/env python3
"""Prepare repository-grouped, retrieval-stratified VAC-MQR cross-validation folds."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


INNER_PROPORTIONS = {
    "utility_train": .40,
    "lcb_calibration": .15,
    "selection_valid": .125,
    "risk_calibration": .325,
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repository(url: str) -> str:
    match = re.match(r"https?://github\.com/([^/]+/[^/]+)", url)
    return match.group(1).casefold() if match else url.split("#", 1)[0].casefold()


def bins(values: list[float], count: int = 4) -> list[int]:
    edges = np.unique(np.quantile(np.asarray(values, dtype=float), np.linspace(0, 1, count + 1)))
    return np.digitize(values, edges[1:-1], right=True).astype(int).tolist()


def assign_groups(group_rows: dict[str, list[int]], strata: list[tuple],
                  proportions: dict[str, float]) -> dict[str, str]:
    roles = list(proportions)
    total = sum(len(indices) for indices in group_rows.values())
    global_strata = Counter(strata[index] for indices in group_rows.values() for index in indices)
    targets = {role: total * proportions[role] for role in roles}
    target_strata = {role: {key: value * proportions[role] for key, value in global_strata.items()}
                     for role in roles}
    role_counts = Counter()
    role_strata = {role: Counter() for role in roles}
    assignment = {}
    ordered = sorted(group_rows, key=lambda key: (-len(group_rows[key]), key))
    for group in ordered:
        indices = group_rows[group]
        counts = Counter(strata[index] for index in indices)

        def cost(role: str) -> tuple[float, float, str]:
            size_ratio = (role_counts[role] + len(indices)) / max(targets[role], 1.0)
            stratum_cost = sum(
                ((role_strata[role][key] + value) / max(target_strata[role][key], 1.0)) ** 2
                for key, value in counts.items())
            return size_ratio ** 2 + stratum_cost / max(len(counts), 1), size_ratio, role

        chosen = min(roles, key=cost)
        assignment[group] = chosen
        role_counts[chosen] += len(indices)
        role_strata[chosen].update(counts)
    return assignment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.dataset.open(encoding="utf-8") if line.strip()]
    rr_bins = bins([float(row["original_rr_at_100"]) for row in rows])
    length_bins = bins([len(row["query"].split()) for row in rows])
    strata = [(rr_bins[index], length_bins[index],
               bool(re.search(r"[`_.()/:]", row["query"])), bool(row["needs_rewrite"]))
              for index, row in enumerate(rows)]
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[repository(row["url"])].append(index)
    outer = assign_groups(groups, strata, {f"fold_{fold}": .2 for fold in range(5)})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"format": "vac-mqr-repository-grouped-5fold-v1", "folds": {},
               "source_sha256": digest(args.dataset), "test_used": False}
    for fold in range(5):
        outer_role = f"fold_{fold}"
        remaining = {group: indices for group, indices in groups.items()
                     if outer[group] != outer_role}
        inner = assign_groups(remaining, strata, INNER_PROPORTIONS)
        roles = {}
        output = args.output_dir / f"fold_{fold}.jsonl"
        with output.open("w", encoding="utf-8") as stream:
            for index, row in enumerate(rows):
                group = repository(row["url"])
                role = "new_holdout" if outer[group] == outer_role else inner[group]
                roles[role] = roles.get(role, 0) + 1
                stream.write(json.dumps({**row, "split_role": role}, ensure_ascii=False) + "\n")
        role_repositories = {role: {repository(row["url"]) for row in rows
                                    if ("new_holdout" if outer[repository(row["url"])] == outer_role
                                        else inner[repository(row["url"])]) == role}
                             for role in [*INNER_PROPORTIONS, "new_holdout"]}
        overlap = max((len(role_repositories[left] & role_repositories[right])
                       for left in role_repositories for right in role_repositories if left < right),
                      default=0)
        manifest = {"format": "vac-mqr-cv-fold-v1", "fold": fold,
                    "strategy": "repository-grouped with RR/query-length/identifier/positive-gain strata",
                    "roles": roles, "repository_overlap_max": overlap,
                    "source_sha256": digest(args.dataset), "output_sha256": digest(output),
                    "test_used": False}
        manifest_path = args.output_dir / f"fold_{fold}_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        summary["folds"][str(fold)] = {**manifest, "dataset": str(output.resolve()),
                                        "manifest": str(manifest_path.resolve())}
    (args.output_dir / "manifest.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
