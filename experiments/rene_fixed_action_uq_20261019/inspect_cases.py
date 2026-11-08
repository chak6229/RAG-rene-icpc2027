"""Post-outcome descriptive influence analysis; no fitting or policy changes."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from experiments.rene_strict_confirmation_20261019.common import (
    load, read_json, sha, utc, write_json,
)

OUT = ROOT / "results/rene_fixed_action_uq_20261019"
DEV = ROOT / "results/rene_strict_confirmation_20261019/development"


def main():
    vectors = load(OUT / "PER_QUERY.pkl")
    items = {e["query_index"]: e for e in load(DEV / "items.pkl")}
    analysis = {}
    for family in ("fusion", "rewrite"):
        analysis[family] = {}
        for upstream in ("no_seed_uq", "seed_uq"):
            analysis[family][upstream] = {}
            for after, before in (("full", "common"), ("support", "common"), ("full", "support")):
                a = vectors[family][upstream][before]["fixed50"]
                b = vectors[family][upstream][after]["fixed50"]
                assert [z["q"] for z in a] == [z["q"] for z in b]
                changed = [(x, y) for x, y in zip(a, b) if x["keep"] != y["keep"]]
                differences = []
                for x, y in changed:
                    assert x["pre_delta"] == y["pre_delta"]
                    effect = y["delta"] - x["delta"]
                    if abs(effect) <= 1e-12:
                        continue
                    q = x["q"]
                    differences.append(dict(
                        query_index=q, repo=items[q]["repo"], query=items[q]["query"],
                        fold=x["fold"], original_rr=x["original_rr"],
                        action_delta_rr=x["pre_delta"], before_keep=x["keep"], after_keep=y["keep"],
                        contrast_rr=effect, contribution_mrr=effect / 512,
                    ))
                differences.sort(key=lambda d: -abs(d["contrast_rr"]))
                total = sum(y["delta"] - x["delta"] for x, y in zip(a, b))
                by_fold = []
                for fold in range(4):
                    values = [y["delta"] - x["delta"] for x, y in zip(a, b) if x["fold"] == fold]
                    assert len(values) == 128
                    by_fold.append(float(np.mean(values)))
                negative_before = sum(max(-x["delta"], 0.) for x in a)
                negative_after = sum(max(-x["delta"], 0.) for x in b)
                positive_before = sum(max(x["delta"], 0.) for x in a)
                positive_after = sum(max(x["delta"], 0.) for x in b)
                record = dict(
                    changed_adoptions=len(changed), rr_changing_cases=len(differences),
                    rr_unchanged_swaps=len(changed) - len(differences), cases=differences,
                    increment_mrr=total / 512,
                    improvement_from_saved_loss=(negative_before - negative_after) / 512,
                    improvement_from_positive_mass=(positive_after - positive_before) / 512,
                    per_fold_increment=by_fold,
                )
                if differences:
                    largest = differences[0]["contrast_rr"]
                    record.update(
                        largest_absolute_case_rr=largest,
                        largest_absolute_case_fraction_of_net=largest / total if total else None,
                        leave_largest_case_out_increment=(total - largest) / 511,
                    )
                analysis[family][upstream][f"{after}_minus_{before}"] = record
    result = dict(
        generated_at_utc=utc(), posthoc=True, policy_changed=False, retraining=False,
        primary_unaltered=True, analysis=analysis,
        script_sha256=sha(Path(__file__)), results_sha256=sha(OUT / "RESULTS.json"),
        per_query_sha256=sha(OUT / "PER_QUERY.pkl"),
        note="Influence descriptions on consumed development data; not additional confirmatory tests.",
    )
    write_json(OUT / "CASE_ANALYSIS.json", result, immutable=True)
    import json
    print(json.dumps(analysis["fusion"]["no_seed_uq"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
