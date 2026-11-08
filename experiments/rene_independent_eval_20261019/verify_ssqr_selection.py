"""Pure-arithmetic check of the SSQR selection contract on the real-weight smoke."""

import json
import os

ROOT = r"."
SMOKE = os.path.join(ROOT, r"results\rene_ssqr_alignment_20261019\T5_REAL_WEIGHT_SMOKE.json")
OUT = os.path.join(ROOT, r"results\rene_independent_eval_20261019\SSQR_SELECTION_CHECK.json")

with open(SMOKE, encoding="utf-8") as handle:
    smoke = json.load(handle)

candidates = smoke["candidates"]
positions = [row["position"] for row in candidates]

entr_expected = [row["position"] for row in sorted(candidates, key=lambda row: row["full_vocab_entropy"])]
prob_expected = [row["position"] for row in sorted(candidates, key=lambda row: -row["label_mean_log_probability"])]
rand_positions = smoke["selected_positions"]["RAND"]

entropy_error = smoke.get("max_independent_fp64_entropy_difference")

result = {
    "schema": "ssqr-selection-check-20261019-v1",
    "positions": positions,
    "entr_expected": entr_expected,
    "entr_actual": smoke["selected_positions"]["ENTR"],
    "entr_ok": entr_expected == smoke["selected_positions"]["ENTR"],
    "prob_expected": prob_expected,
    "prob_actual": smoke["selected_positions"]["PROB"],
    "prob_ok": prob_expected == smoke["selected_positions"]["PROB"],
    "rand_is_permutation": sorted(rand_positions) == sorted(positions),
    "rand_actual": rand_positions,
    "identity_padding": any(row.get("is_identity") for row in candidates),
    "entropy_max_abs_error_vs_fp64": entropy_error,
    "entropy_within_1e5": entropy_error is not None and entropy_error < 1e-5,
}

with open(OUT, "w", encoding="utf-8") as handle:
    json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
    handle.write("\n")

print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
