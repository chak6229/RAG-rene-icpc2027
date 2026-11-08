"""Inspect consumed-development query pairing; never load model outcomes."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "imports/CodeXGLUE-main/Text-Code/NL-code-search-WebQuery/CoSQA/cosqa-all.json"
QUERIES = ROOT / "results/rene_cosqa_development_20261019/data/queries.jsonl"
GROUPS = ROOT / "results/rene_cosqa_development_20261019/data/groups.json"
OUTPUT = ROOT / "results/rene_core_contribution_design_20261020/FEASIBILITY.json"
PROTECTED = (
    "configs/vac_mqr_gain_model/CURRENT.json",
    "configs/vac_mqr_verification_policy/CURRENT.json",
    "configs/vac_mqr_risk_module/CURRENT.json",
    "configs/uq_candidate_selection/CURRENT.json",
    "results/vac_mqr_v5/STABLE.json",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError("Preserve the previous metadata report; use a new version.")
    before = {name: sha(ROOT / name) for name in PROTECTED}
    source = json.loads(SOURCE.read_text(encoding="utf-8-sig"))
    queries = [json.loads(line) for line in QUERIES.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    groups = json.loads(GROUPS.read_text(encoding="utf-8-sig"))
    lookup = {row["idx"]: row for row in source}
    assert len(lookup) == len(source), "Duplicate source identities"
    assert len({row["query_id"] for row in queries}) == len(queries), "Duplicate D identities"
    assert set(groups) == {row["query_id"] for row in queries}, "D group identities drifted"
    by_code: dict[str, set[str]] = defaultdict(set)
    eligible_groups = set()
    matched = 0
    for query in queries:
        row = lookup[query["query_id"]]
        assert row["doc"] == query["query"], "D input differs from original query"
        assert row["label"] == 1, "D row has no known positive target"
        docstring = row.get("docstring_tokens", "").strip()
        if docstring:
            matched += 1
            by_code[row["code_tokens"]].add(docstring)
            eligible_groups.add(groups[query["query_id"]])
    after = {name: sha(ROOT / name) for name in PROTECTED}
    assert before == after, "Protected configuration changed"
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "role": "metadata_only_design_feasibility_on_consumed_D",
        "query_rows": len(queries),
        "connected_groups": len(set(groups.values())),
        "queries_with_nonempty_target_docstring": matched,
        "queries_without_target_docstring": len(queries) - matched,
        "eligible_distinct_code_strings": len(by_code),
        "eligible_connected_groups": len(eligible_groups),
        "target_code_strings_with_conflicting_docstrings": sum(len(values) != 1 for values in by_code.values()),
        "all_D_query_texts_match_source": True,
        "new_generation": False,
        "new_training": False,
        "new_retrieval": False,
        "method_effects_computed": False,
        "confirmation_outcomes_read": False,
        "sources": [
            {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path)}
            for path in (SOURCE, QUERIES, GROUPS)
        ],
        "protected_files_unchanged": before,
        "limitation": "Existing D metadata only; these pairs are not new independent confirmation data.",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({key: value for key, value in result.items() if key not in {"sources", "protected_files_unchanged"}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
