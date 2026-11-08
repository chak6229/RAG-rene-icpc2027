"""Independent probability and executed-ranking metric recomputation."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.special import logsumexp

from crsu.core import rrf
from experiments.rene_strict_confirmation_20261019.common import (
    check_hashes, load, observable, read_json, reject_gold, sha, utc, write_json,
)
from experiments.rene_fixed_action_uq_20261019 import model as heads

OUT = ROOT / "results/rene_fixed_action_uq_20261019"
DEV = ROOT / "results/rene_strict_confirmation_20261019/development"
TOL = 1e-10


def independent_rr(ranking, target):
    top = [int(d) for d in ranking][:100]
    return 1. / (top.index(int(target)) + 1) if int(target) in top else 0.


def independent_probabilities(model, data):
    n = len(data["q"])
    if model["fallback"]:
        return np.tile(model["class_prior"], (n, 1))
    x = data["common"] if model["head"] == "common" else np.column_stack((data["common"], data["seed"]))
    z = np.column_stack((np.ones(n), ((x - model["mean"]) / model["scale"]) * model["mask"]))
    native = np.minimum(1 - 1e-6, np.maximum(1e-6, data["native"]))
    positive = model["positive_given_nonharm"]
    eta = np.log(np.column_stack((native, (1 - native) * (1 - positive), (1 - native) * positive)))
    effects = z @ model["beta"].T
    eta[:, 0] += effects[:, 0]
    eta[:, 2] += effects[:, 1]
    return np.exp(eta - logsumexp(eta, axis=1)[:, None])


def audit():
    protocol = read_json(OUT / "PROTOCOL.json")
    seal = read_json(OUT / "DECISION_SEAL.json")
    result = read_json(OUT / "RESULTS.json")
    check_hashes(protocol["source_hashes"])
    check_hashes(protocol["input_hashes"])
    check_hashes(protocol["protected_hashes"])
    check_hashes(seal["fold_hashes"])
    assert seal["protocol_sha256"] == sha(OUT / "PROTOCOL.json")
    assert result["decision_seal_sha256"] == sha(OUT / "DECISION_SEAL.json")
    items = load(DEV / "items.pkl")
    byid = {e["query_index"]: e for e in items}
    assert len(items) == len({e["repo"] for e in items}) == 512
    label_checks, maximum_label_error = 0, 0.
    for e in items:
        q = e["query_index"]
        original = independent_rr(e["original_top100"], q)
        assert abs(original - e["original_rr"]) < 1e-12
        for j, ranking in enumerate(e["candidate_top100"]):
            for family, executed in (("rewrite", ranking), ("fusion", rrf(e["original_top100"], ranking))):
                error = abs(independent_rr(executed, q) - original - float(e["y"][family][j]))
                maximum_label_error = max(maximum_label_error, error)
                assert error < 1e-12
                label_checks += 1
    counts = dict(folds=0, true_refits=0, fixed_trace_rows=0, input_rows=0,
                  probability_rows=0, metric_rows=0, adoption_masks=0,
                  gold_rejection_probes=0, poisoned_held_metadata_probes=0)
    maximum_probability_error = 0.
    metric_collections = {}
    primary_deltas = {}
    for family in ("fusion", "rewrite"):
        for upstream in ("no_seed_uq", "seed_uq"):
            collection = {(h, p): [] for h in heads.HEADS for p in heads.POLICIES}
            probability_data = {h: [] for h in heads.HEADS}
            for fold in range(4):
                key = f"{family}_{upstream}_{fold}"
                batch = load(OUT / f"folds/{key}.pkl")
                source = load(DEV / f"folds/{key}.pkl")
                assert batch["source_checkpoint_sha256"] == sha(DEV / f"folds/{key}.pkl")
                assert batch["protocol_sha256"] == sha(OUT / "PROTOCOL.json")
                roles = batch["roles"]
                assert {k: set(v) for k, v in source["model"]["roles"].items()} == {
                    k: set(v) for k, v in roles.items() if k != "held"}
                for role, ids in roles.items():
                    for other, other_ids in roles.items():
                        if role != other:
                            assert not {byid[q]["repo"] for q in ids} & {byid[q]["repo"] for q in other_ids}
                for name in ("gain", "update"):
                    assert set(source["model"][name]["fit_queries"]) <= set(roles[name])
                    assert set(source["model"][name]["fit_repos"]) <= {byid[q]["repo"] for q in roles[name]}
                for role in ("risk", "held"):
                    traces = batch["fixed"][role]
                    assert [t["query_index"] for t in traces] == roles[role]
                    views = [observable(byid[q]) for q in roles[role]]
                    rebuilt = heads.inputs(views, traces)
                    data = batch["inputs"][role]
                    assert rebuilt["q"] == data["q"]
                    for k in ("common", "seed", "native"):
                        np.testing.assert_allclose(rebuilt[k], data[k], atol=1e-12, rtol=0)
                    reject_gold(data)
                    counts["fixed_trace_rows"] += len(traces)
                    counts["input_rows"] += len(data["q"])
                    for head in heads.HEADS:
                        model = batch["models"][head]
                        recomputed = independent_probabilities(model, data)
                        saved = batch["predictions"][role][head]["classes"]
                        maximum_probability_error = max(maximum_probability_error,
                                                        float(np.max(abs(recomputed - saved))) if len(saved) else 0.)
                        np.testing.assert_allclose(recomputed, saved, atol=1e-12, rtol=0)
                        counts["probability_rows"] += len(data["q"])
                train = batch["training"]
                expected_q = [t["query_index"] for t in batch["fixed"]["risk"] if t["accepted"]]
                assert train["q"] == expected_q
                expected_delta = np.array([
                    byid[t["query_index"]]["y"][family][t["chosen"]]
                    for t in batch["fixed"]["risk"] if t["accepted"]
                ])
                np.testing.assert_array_equal(expected_delta, train["delta"])
                for head, model in batch["models"].items():
                    assert model["fit_query_ids"] == expected_q
                    assert set(model["fit_repos"]) <= {byid[q]["repo"] for q in roles["risk"]}
                    assert not set(model["fit_query_ids"]) & set(roles["held"])
                    assert model["fit_labels_sha256"] == hashlib.sha256(expected_delta.tobytes()).hexdigest()
                    np.testing.assert_array_equal(model["mask"], heads.feature_mask(head))
                    if not model["fallback"]:
                        np.testing.assert_array_equal(
                            model["beta"][:, 1:][:, model["mask"] == 0],
                            np.zeros_like(model["beta"][:, 1:][:, model["mask"] == 0]))
                    counts["true_refits"] += 1
                original_prob = independent_probabilities(
                    dict(source["model"]["risk"], head="common"), batch["inputs"]["held"])
                np.testing.assert_allclose(original_prob, batch["predictions"]["held"]["common"]["classes"],
                                           atol=1e-10, rtol=0)
                # Outcome and repo metadata cannot enter inference; the allowed view is unchanged.
                probe_item = byid[roles["held"][0]]
                poisoned = dict(probe_item, original_rr=-100., y={"fusion": [999.]}, repo="poison")
                clean_view, poison_view = observable(probe_item), observable(poisoned)
                probe_trace = batch["fixed"]["held"][0]
                clean_arrays = heads.inputs([clean_view], [probe_trace])
                poison_arrays = heads.inputs([poison_view], [probe_trace])
                for k in ("common", "seed", "native"):
                    np.testing.assert_array_equal(clean_arrays[k], poison_arrays[k])
                counts["poisoned_held_metadata_probes"] += 1
                try:
                    heads.inputs([dict(clean_view, original_rr=.5)], [probe_trace])
                except ValueError:
                    counts["gold_rejection_probes"] += 1
                else:
                    raise AssertionError("Gold input was accepted")
                held = batch["inputs"]["held"]
                tracemap = {t["query_index"]: t for t in batch["fixed"]["held"]}
                for head in heads.HEADS:
                    probability_data[head].extend([
                        (independent_rr(tracemap[q]["ranking"], q) -
                         independent_rr(byid[q]["original_top100"], q), p)
                        for q, p in zip(held["q"], batch["predictions"]["held"][head]["classes"])
                    ])
                    for policy in heads.POLICIES:
                        p = batch["predictions"]["held"][head]["harm"]
                        mask = batch["masks"][head][policy]
                        if policy == "threshold075":
                            np.testing.assert_array_equal(mask, p <= .075)
                        else:
                            fraction = {"fixed25": .25, "fixed50": .5, "fixed75": .75, "all": 1.}[policy]
                            number = int(len(p) * fraction)
                            indices = sorted(range(len(p)), key=lambda i: (float(p[i]), held["q"][i]))[:number]
                            assert set(np.flatnonzero(mask)) == set(indices)
                        counts["adoption_masks"] += 1
                        selected = {q for q, yes in zip(held["q"], mask) if yes}
                        for t in batch["fixed"]["held"]:
                            q = t["query_index"]
                            original = tuple(map(int, byid[q]["original_top100"]))
                            action = tuple(map(int, t["ranking"]))
                            assert t["accepted"] == (original != action)
                            ranking = action if q in selected else original
                            assert q not in selected or (t["accepted"] and t["chosen"] >= 0)
                            delta = independent_rr(ranking, q) - independent_rr(original, q)
                            pre_delta = independent_rr(action, q) - independent_rr(original, q)
                            collection[head, policy].append(
                                dict(q=q, delta=delta, pre=pre_delta, keep=q in selected,
                                     costs=tuple(int(t[k]) for k in heads.COSTS))
                            )
                counts["folds"] += 1
            for (head, policy), rows in collection.items():
                rows.sort(key=lambda r: r["q"])
                expected = result["families"][family][upstream]["methods"][head][policy]
                d = np.array([r["delta"] for r in rows])
                keep = np.array([r["keep"] for r in rows])
                pre = np.array([r["pre"] for r in rows])
                assert len(rows) == len({r["q"] for r in rows}) == 512
                assert expected["accepted"] == int(keep.sum())
                assert expected["harmful"] == int((d < -TOL).sum())
                assert expected["positive"] == int((d > TOL).sum())
                assert expected["zero"] == int(((abs(d) <= TOL) & keep).sum())
                assert abs(expected["delta_mrr"] - float(d.mean())) < 1e-12
                assert abs(expected["negative_rr_mass"] - float(np.maximum(-d, 0).sum())) < 1e-12
                denominator = np.maximum(pre, 0).sum()
                if denominator:
                    assert abs(expected["positive_retention"] - float(np.maximum(d, 0).sum() / denominator)) < 1e-12
                for i, name in enumerate(heads.COSTS):
                    assert expected[name] == sum(r["costs"][i] for r in rows)
                counts["metric_rows"] += 1
            for head in heads.HEADS:
                dp = probability_data[head]
                y = np.array([d < -TOL for d, p in dp], float)
                ph = np.array([p[0] for d, p in dp])
                metric = result["families"][family][upstream]["probability_quality"][head]
                assert abs(metric["harm_brier"] - float(np.mean((ph - y) ** 2))) < 1e-12
                assert metric["harmful"] == int(y.sum())
            metric_collections[family, upstream] = collection
            for policy in ("fixed25", "fixed50", "fixed75", "all"):
                assert len({result["families"][family][upstream]["methods"][h][policy]["accepted"]
                            for h in heads.HEADS}) == 1
            for policy in heads.POLICIES:
                assert len({tuple(result["families"][family][upstream]["methods"][h][policy][k]
                                  for k in heads.COSTS) for h in heads.HEADS}) == 1
                for after, before in (("full", "common"), ("support", "common"), ("full", "support")):
                    delta = np.array([b["delta"] - a["delta"] for a, b in zip(
                        collection[before, policy], collection[after, policy])])
                    stated = result["families"][family][upstream]["contrasts"][policy][f"{after}_minus_{before}"]
                    assert abs(stated["delta_mrr"] - float(delta.mean())) < 1e-12
                    if family == "fusion" and upstream == "no_seed_uq" and policy == "fixed50" and after == "full" and before == "common":
                        primary_deltas = delta
    bootstrap = np.random.default_rng(261019).integers(512, size=(10000, 512), dtype=np.int32)
    interval = np.quantile(primary_deltas[bootstrap].mean(1), [.025, .975])
    np.testing.assert_allclose(interval, result["primary_result"]["descriptive_bootstrap95"], atol=1e-12, rtol=0)
    events = [__import__("json").loads(s) for s in (OUT / "EVENTS.jsonl").read_text("utf8").splitlines()]
    stages = [s["stage"] for s in events]
    assert stages.index("protocol_frozen") < stages.index("all_predictions_and_adoptions_sealed") < stages.index("held_evaluation_started")
    assert stages.count("fold_saved") == 16
    assert counts["true_refits"] == 48
    check_hashes(protocol["source_hashes"])
    check_hashes(protocol["input_hashes"])
    check_hashes(protocol["protected_hashes"])
    verdict = dict(
        passed=True, audited_at_utc=utc(), counts=counts,
        candidate_label_checks=label_checks, maximum_candidate_label_error=maximum_label_error,
        maximum_probability_recompute_error=maximum_probability_error,
        common_head_replay=True, same_actions=True, same_evidence=True, same_cost=True,
        role_disjoint=True, independent_metric_recompute=True, fixed_count_and_threshold=True,
        all_predictions_sealed_before_metric_evaluation=True,
        source_and_cache_hashes_unchanged=True, protected_unchanged=True,
        fresh_1536_read=False, old_test_read=False, neural_model_calls=False,
        risk_certified=False, confirmation=False,
        caveat="conditional fitted-model development diagnostics; no new independent confirmation",
        evidence=dict(protocol=sha(OUT / "PROTOCOL.json"), decision_seal=sha(OUT / "DECISION_SEAL.json"),
                      results=sha(OUT / "RESULTS.json"), per_query=sha(OUT / "PER_QUERY.pkl")),
    )
    write_json(OUT / "AUDIT.json", verdict, immutable=True)
    print("AUDIT PASS " + str(counts), flush=True)


if __name__ == "__main__":
    audit()
