"""One-time outcome opening after sealed, audited actual decisions."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.metrics import roc_auc_score

from crsu.core import rrf
from experiments.rene_uq_independent_proof_20261019.study import (
    ARMS, FRESH, OUT, TOL, check_protocol, event, heads,
)
from experiments.rene_uq_independent_proof_20261019.inference_stats import (
    block_bootstrap_indices, fixed_half_contrast, intervals, score_difference, verdict,
)
from experiments.rene_strict_confirmation_20261019.common import (
    check_hashes, load, read_json, require_runtime, rr, save, sha, utc, write_json,
)


def method_summary(traces, originals, targets, draws):
    base = np.array([rr(originals[t["query_index"]], targets[t["query_index"]]) for t in traces])
    final = np.array([rr(t["ranking"], targets[t["query_index"]]) for t in traces])
    delta = final - base
    adopted = np.array([t["accepted"] for t in traces], bool)
    assert not np.any(abs(delta[~adopted]) > TOL)
    positive = int((delta > TOL).sum())
    harmful = int((delta < -TOL).sum())
    count = int(adopted.sum())
    result = dict(
        queries=len(traces), original_mrr=float(base.mean()), mrr=float(final.mean()),
        delta_mrr=float(delta.mean()), descriptive_ci95=intervals(delta, draws)["descriptive_ci95"],
        adopted=count, coverage=float(adopted.mean()), positive=positive, harmful=harmful,
        adopted_zero=count-positive-harmful, harmful_rate=harmful/count if count else None,
        positive_gain_sum=float(delta[delta > TOL].sum()),
        negative_loss_sum=float(-delta[delta < -TOL].sum()),
        verified=sum(bool(t["verified"]) for t in traces),
        billed_slots=sum(int(t["billed_slots"]) for t in traces),
        actual_pairs=sum(int(t["actual_pairs"]) for t in traces),
        executed_ce_tokens=sum(int(t["executed_ce_tokens"]) for t in traces),
    )
    return result, delta


def proper_score_vectors(delta, classes):
    delta, classes = np.asarray(delta, float), np.asarray(classes, float)
    assert classes.shape == (len(delta), 3)
    labels = np.where(delta < -TOL, 0, np.where(delta > TOL, 2, 1))
    y = (delta < -TOL).astype(float)
    p = np.clip(classes[:, 0], 1e-12, 1-1e-12)
    return dict(
        harm_brier=(classes[:, 0]-y)**2,
        harm_nll=-(y*np.log(p)+(1-y)*np.log1p(-p)),
        class_nll=-np.log(np.maximum(classes[np.arange(len(delta)), labels], 1e-12)),
        class_brier=np.sum((classes-np.eye(3)[labels])**2, axis=1),
    )


def evaluate():
    if (OUT/"RESULTS.json").exists():
        check_hashes(read_json(OUT/"EVALUATION_SEAL.json")["outputs"])
        print("Existing sealed evaluation verified; no new outcome-driven run", flush=True)
        return
    p = check_protocol()
    audit = read_json(OUT/"PREOUTCOME_AUDIT.json")
    assert audit["passed"] and not audit["fresh_outcomes_opened"]
    assert audit["protocol_sha256"] == sha(FRESH/"PROTOCOL.json")
    seal = read_json(FRESH/"DECISION_SEAL.json")
    assert seal["actual_decisions"] and not seal["outcomes_opened"]
    for file, key in (
        ("label_free_decisions.pkl", "decisions_sha256"),
        ("observable_items.pkl", "observations_sha256"),
        ("ce_scores.pkl", "scores_sha256"),
        ("PLAN_SEAL.json", "plan_seal_sha256"),
        ("PROTOCOL.json", "protocol_sha256"),
    ):
        assert sha(FRESH/file) == seal[key], file
    assert sha(FRESH/"private_outcomes.json") == p["private_outcomes_sha256"]
    if not (OUT/"OUTCOME_OPENING.json").exists():
        write_json(OUT/"OUTCOME_OPENING.json", dict(
            opened_at_utc=utc(), protocol_sha256=sha(FRESH/"PROTOCOL.json"),
            preoutcome_audit_sha256=sha(OUT/"PREOUTCOME_AUDIT.json"),
            decision_seal_sha256=sha(FRESH/"DECISION_SEAL.json"),
            policy_changed_after_opening=False,
        ), immutable=True)
        write_json(OUT/"CONSUMED.json", dict(
            consumed=True, queries=p["query_count"], consumed_at_utc=utc(),
            use="one confirmation of frozen risk-head comparisons",
            never_reuse_as_unseen_confirmation=True,
        ), immutable=True)
        event("outcome_mapping_first_opened")
    else:
        assert read_json(OUT/"OUTCOME_OPENING.json")["protocol_sha256"] == sha(FRESH/"PROTOCOL.json")
        event("same_frozen_evaluation_resumed")
    outcomes = read_json(FRESH/"private_outcomes.json")
    items = load(FRESH/"observable_items.pkl")
    saved = load(FRESH/"label_free_decisions.pkl")
    queries = read_json(FRESH/"queries.json")
    ids = [e["query_index"] for e in items]
    assert ids == [q["query_index"] for q in queries] == [q["query_index"] for q in outcomes]
    assert len(ids) == p["query_count"] == len(set(ids))
    targets = {q["query_index"]: int(q["target_doc_id"]) for q in outcomes}
    originals = {e["query_index"]: e["original_top100"] for e in items}
    byid = {e["query_index"]: e for e in items}
    draws = block_bootstrap_indices(len(ids), p["bootstrap_repeats"])
    methods, deltas = {}, {}
    for name, traces in saved["outputs"].items():
        assert [t["query_index"] for t in traces] == ids
        for t in traces:
            e = byid[t["query_index"]]
            expected = tuple(map(int, e["original_top100"] if t["chosen"] < 0 else
                                 rrf(e["original_top100"], e["candidate_top100"][t["chosen"]])))
            assert tuple(t["ranking"]) == expected
            assert bool(t["accepted"]) == (expected != tuple(e["original_top100"]))
        methods[name], deltas[name] = method_summary(traces, originals, targets, draws)

    primary, secondaries, scores, mechanisms, rows = {}, {}, {}, {}, {}
    index = {q:i for i, q in enumerate(ids)}
    for arm in ARMS:
        arr = saved["inputs"][arm]
        positions = np.array([index[q] for q in arr["q"]], int)
        eligible = np.zeros(len(ids), bool)
        eligible[positions] = True
        pre = deltas[f"{arm}/pre_gate"]
        assert np.array_equal(eligible, [t["accepted"] for t in saved["fixed"][arm]])
        scores[arm] = dict(methods={}, contrasts={})
        score_arrays = {}
        for h in heads.HEADS:
            classes = saved["probabilities"][arm][h]["classes"]
            sv = proper_score_vectors(pre[positions], classes)
            score_arrays[h] = {}
            summary = {k:float(v.mean()) if len(v) else None for k,v in sv.items()}
            yy = pre[positions] < -TOL
            summary["harm_auc"] = float(roc_auc_score(yy, classes[:, 0])) if len(set(yy)) == 2 else None
            summary.update(eligible=int(eligible.sum()), harmful=int(yy.sum()),
                           class_order=["harm", "zero", "positive"])
            scores[arm]["methods"][h] = summary
            for k, v in sv.items():
                expanded = np.zeros(len(ids))
                expanded[positions] = v
                score_arrays[h][k] = expanded
        for before in ("support", "common"):
            label = f"full_minus_{before}"
            scores[arm]["contrasts"][label] = {
                k:score_difference(score_arrays["full"][k]-score_arrays[before][k], eligible, draws)
                for k in score_arrays["full"]
            }
            scores[arm]["contrasts"][label]["interpretation"] = (
                "Negative proper-score differences favor full; descriptive and model-specific, not conditional mutual information.")
            pa, pb = np.ones(len(ids)), np.ones(len(ids))
            pa[positions] = saved["probabilities"][arm][before]["harm"]
            pb[positions] = saved["probabilities"][arm]["full"]["harm"]
            secondary = fixed_half_contrast(pre, eligible, pa, pb, ids, draws)
            observed = deltas[f"{arm}/full/fixed50"]-deltas[f"{arm}/{before}/fixed50"]
            assert abs(secondary["mean"]-float(observed.mean())) < 1e-12
            secondaries[f"{arm}/{label}/fixed50"] = secondary
            values = deltas[f"{arm}/full/threshold075"]-deltas[f"{arm}/{before}/threshold075"]
            contrast = intervals(values, draws)
            contrast.update(verdict=verdict(contrast, p["equivalence_margin"]),
                            confirmatory=arm == p["primary_upstream"])
            bounds = saved["geometry"][arm][label]["threshold075"]
            assert np.all(values >= bounds["per_query_lower"]-1e-12)
            assert np.all(values <= bounds["per_query_upper"]+1e-12)
            contrast["label_free_cohort_bounds"] = {k:bounds[k] for k in ("mean_lower", "mean_upper", "scope")}
            if arm == p["primary_upstream"]:
                primary[label] = contrast
            else:
                secondaries[f"{arm}/{label}/threshold075"] = contrast
            ta = saved["outputs"][f"{arm}/{before}/threshold075"]
            tb = saved["outputs"][f"{arm}/full/threshold075"]
            cases = []
            for i, (a,b) in enumerate(zip(ta,tb)):
                if a["accepted"] == b["accepted"]:
                    continue
                cases.append(dict(
                    query_index=ids[i], repo=queries[i]["repo"], query=queries[i]["query"],
                    before_adopted=bool(a["accepted"]), full_adopted=bool(b["accepted"]),
                    pre_gate_delta=float(pre[i]), delta_contribution_sum=float(values[i]),
                    mrr_contribution=float(values[i]/len(ids)),
                    original_rr=rr(originals[ids[i]], targets[ids[i]]),
                    action_rr=rr(saved["fixed"][arm][i]["ranking"], targets[ids[i]]),
                ))
            cases.sort(key=lambda z:(-abs(z["delta_contribution_sum"]), z["query_index"]))
            mechanisms[f"{arm}/{label}"] = dict(
                adoption_disagreements=len(cases),
                nonzero_outcome_disagreements=sum(abs(c["delta_contribution_sum"])>TOL for c in cases),
                sum_contribution=float(values.sum()), delta_mrr=float(values.mean()),
                top_absolute_contribution_fraction=(
                    abs(cases[0]["delta_contribution_sum"])/sum(abs(c["delta_contribution_sum"]) for c in cases)
                    if any(abs(c["delta_contribution_sum"])>TOL for c in cases) else None),
                cases=cases,
            )
            rows[f"{arm}/{label}"] = values
    stages = {}
    for stage in ("mean", "base", "pre_gate", "common/threshold075", "support/threshold075", "full/threshold075"):
        difference = deltas[f"seed_uq/{stage}"]-deltas[f"no_seed_uq/{stage}"]
        stages[stage] = dict(intervals(difference, draws), confirmatory=False,
                            note="Different upstream actions; not an isolated risk-head effect.")
    oracle = []
    for e in items:
        target = targets[e["query_index"]]
        base = rr(e["original_top100"], target)
        gains = [rr(rrf(e["original_top100"], t), target)-base for t in e["candidate_top100"]]
        oracle.append(max([0.]+gains))
    result = dict(
        completed=True, evaluated_at_utc=utc(), queries=len(items), repositories=len({q["repo"] for q in outcomes}),
        candidates=sum(len(e["x"]) for e in items), corpus_documents=4096,
        baseline_mrr=methods[f"{ARMS[0]}/pre_gate"]["original_mrr"],
        primary_policy=p["primary_policy"], primary_upstream=p["primary_upstream"],
        primary_contrasts=primary, methods=methods, proper_scores=scores,
        secondary_comparisons=secondaries, upstream_stage_contrasts=stages,
        mechanisms=mechanisms,
        oracle=dict(fusion_delta_mrr=float(np.mean(oracle)),
                    queries_with_positive_candidate=int(np.count_nonzero(np.array(oracle)>TOL)),
                    role="label-dependent candidate-pool diagnostic; not a deployable method"),
        power=read_json(OUT/"POWER.json"), ce_physical_cost=read_json(FRESH/"CE_COMPLETE.json"),
        decision_seal_sha256=sha(FRESH/"DECISION_SEAL.json"), protocol_sha256=sha(FRESH/"PROTOCOL.json"),
        old_test_metrics_used=False, risk_certified=False, formal_components_changed=False,
        scope="new downstream query repositories, same corpus and upstream source-train exposure",
    )
    save(OUT/"per_query_outcomes.pkl", dict(ids=ids, delta=deltas, primary_vectors=rows))
    write_json(OUT/"RESULTS.json", result, immutable=True)
    write_json(OUT/"EVALUATION_SEAL.json", dict(
        completed_at_utc=utc(), outputs={str((OUT/file).relative_to(ROOT)):sha(OUT/file) for file in (
            "RESULTS.json", "per_query_outcomes.pkl", "OUTCOME_OPENING.json", "CONSUMED.json")},
    ), immutable=True)
    event("evaluation_completed", results_sha256=sha(OUT/"RESULTS.json"))
    print("PRIMARY " + str(primary), flush=True)
    print("PRIMARY METHODS " + str({h:methods[f"no_seed_uq/{h}/threshold075"] for h in heads.HEADS}), flush=True)


if __name__ == "__main__":
    require_runtime()
    evaluate()
