"""Independent Fraction-RR and scalar interval replay of step A."""
from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[2]
BRIDGE = ROOT / "results/rene_ssqr_bridge_20261019"
OUT = ROOT / "results/rene_core_step_a_20261020"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def iterl(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def key(text):
    return hashlib.sha256(text.encode()).hexdigest()


def reciprocal(order, target):
    require(len(order) == len(set(order)) == 100, "Invalid cached rank list")
    found = [rank for rank, doc in enumerate(order, 1) if doc == target]
    return Fraction(1, found[0]) if found else Fraction(0)


def scalar_interval(values, groups, *, seed, repetitions):
    labels = sorted(set(groups))
    members = defaultdict(list)
    for value, group in zip(values, groups):
        members[group].append(value)
    sizes = np.array([len(members[group]) for group in labels])
    sums = np.array([math.fsum(members[group]) for group in labels])
    mean = math.fsum(values) / len(values)
    se = math.sqrt(len(labels) / (len(labels) - 1) * math.fsum(
        (float(total) - count * mean) ** 2 for total, count in zip(sums, sizes)) / len(values) ** 2)
    half = float(t.isf(0.025, len(labels) - 1)) * se
    family_half = float(t.isf(0.05 / 8, len(labels) - 1)) * se
    rng = np.random.default_rng(seed)
    boots = []
    for start in range(0, repetitions, 200):
        draw = rng.integers(0, len(labels), (min(200, repetitions - start), len(labels)))
        boots.extend((sums[draw].sum(1) / sizes[draw].sum(1)).tolist())
    boots.sort()

    def quantile(probability):
        index = (len(boots) - 1) * probability
        low = math.floor(index)
        fraction = index - low
        return boots[low] * (1 - fraction) + boots[min(low + 1, len(boots) - 1)] * fraction

    return {"mean": mean, "standard_error_cluster": se,
            "ci95_cluster_t": [mean - half, mean + half],
            "ci95_group_bootstrap": [quantile(0.025), quantile(0.975)],
            "four_cell_family95_bonferroni_cluster_t": [mean - family_half, mean + family_half],
            "four_cell_family95_bonferroni_group_bootstrap": [quantile(0.05 / 8), quantile(1 - 0.05 / 8)]}


def main():
    require(not (OUT / "AUDIT.json").exists(), "Audit already exists")
    plan = read(OUT / "PROTOCOL.json")
    result = read(OUT / "RESULTS.json")
    require(sha(OUT / "PROTOCOL.json") == result["protocol_sha256"], "Step-A protocol identity changed")
    for item in plan["bound_files"] + plan["protected_files"]:
        require(sha(ROOT / item["path"]) == item["sha256"], f"Protected/frozen drift: {item['path']}")
    bridge = read(BRIDGE / "PROTOCOL.json")
    rows = list(iterl(OUT / "per_query.jsonl"))
    lookup = {(row["domain"], row["retriever"], row["query_id"]): row for row in rows}
    require(len(lookup) == len(rows) == 7076, "Duplicate or missing step-A query ledger")
    largest_rr_error, largest_effect_error, largest_interval_error = 0.0, 0.0, 0.0
    rr_comparisons, effect_comparisons, interval_fields, selections_verified = 0, 0, 0, 0
    for domain in ("source", "cosqa"):
        ids = bridge["samples"][domain]["main_ids"]
        groups = bridge["samples"][domain]["groups"]
        queries = {row["query_id"]: row["query"] for row in iterl(BRIDGE / f"data/{domain}/queries.jsonl")}
        qrel_rows = list(iterl(BRIDGE / f"data/private/{domain}_qrels.jsonl"))
        require(len(qrel_rows) == len(ids) and all(row["relevance"] == 1 for row in qrel_rows), "Audit expects one known positive target per query")
        targets = {row["query_id"]: row["doc_id"] for row in qrel_rows}
        pool = defaultdict(list)
        for row in iterl(BRIDGE / f"candidates/{domain}/baseline.jsonl"):
            pool[row["query_id"]].append(row)
        require(set(ids) == set(queries) == set(pool) == set(targets), "Audit population differs")
        for retriever in ("CodeBERT", "BM25"):
            rankings = {}
            for row in iterl(BRIDGE / f"rankings/{domain}/{retriever}.jsonl"):
                require(row["text_id"] == key(row["text"]) and row["text_id"] not in rankings, "Audit ranking identities differ")
                rankings[row["text_id"]] = row["ranked_doc_ids"]
            for qid in ids:
                candidates = sorted(pool[qid], key=lambda row: row["position"])
                expected = lookup[domain, retriever, qid]
                require(expected["group"] == groups[qid], "Grouping changed")
                entr = [row["position"] for row in sorted(candidates, key=lambda row: (row["full_vocab_entropy"], row["position"]))[:3]]
                prob = [row["position"] for row in sorted(candidates, key=lambda row: (-row["label_mean_log_probability"], row["position"]))[:3]]
                require(entr == expected["ENTR_positions"] and prob == expected["PROB_positions"], "Outcome-independent ordering differs")
                choices = []
                for seed in bridge["rand_seeds"]:
                    seed_bytes = hashlib.sha256(f"{seed}\0{qid}".encode()).digest()
                    order = random.Random(int.from_bytes(seed_bytes, "big")).sample(candidates, len(candidates))
                    choices.append([row["position"] for row in order[:3]])
                selections_verified += 22
                scores = {row["position"]: reciprocal(rankings[key(row["rewrite"])], targets[qid]) for row in candidates}
                base = reciprocal(rankings[key(queries[qid])], targets[qid])
                score_ledger = dict(expected["candidate_RR"])
                require(set(score_ledger) == set(scores), "Candidate RR positions differ")
                for position, value in scores.items():
                    largest_rr_error = max(largest_rr_error, abs(float(value) - score_ledger[position]))
                    rr_comparisons += 1
                ordered = sorted(scores.values())
                k = min(3, len(ordered))
                exact3 = sum((value * Fraction(math.comb(index, k - 1), math.comb(len(ordered), k))
                              for index, value in enumerate(ordered) if index >= k - 1), Fraction())
                values = {
                    "NoRewrite": base, "ENTR_top1": scores[entr[0]], "PROB_top1": scores[prob[0]],
                    "ENTR_best3": max(scores[p] for p in entr), "PROB_best3": max(scores[p] for p in prob),
                    "Uniform_single_exact": sum(scores.values(), Fraction()) / len(scores),
                    "RAND20_best3": sum((max(scores[p] for p in positions) for positions in choices), Fraction()) / 20,
                    "Uniform_best3_exact": exact3,
                }
                require(set(values) == set(expected["values"]), "Metric fields differ")
                for name, value in values.items():
                    largest_rr_error = max(largest_rr_error, abs(float(value) - expected["values"][name]))
                    rr_comparisons += 1
                # Read the frozen algebraic estimand, independent of production effect code.
                effect_values = {name: sum((values[key] * coefficient for key, coefficient in terms.items()), Fraction())
                                 for name, terms in plan["contrast_definitions"].items()}
                require(effect_values["j_ENTR_contract_difference"] == effect_values["a_ENTR_single_minus_uniform"] - effect_values["b_ENTR_best3_minus_RAND20"], "Exact paired contrast definition differs")
                for name, value in effect_values.items():
                    largest_effect_error = max(largest_effect_error, abs(float(value) - expected["contrasts"][name]))
                    effect_comparisons += 1
            print({"fraction_replay_passed": f"{domain}/{retriever}"}, flush=True)
            del rankings
    require(largest_rr_error < 1e-12 and largest_effect_error < 1e-12, "Independent Fraction replay differs")
    for view in result["views"]:
        domain, retriever = view["domain"], view["retriever"]
        ids = bridge["samples"][domain]["main_ids"]
        cell = [lookup[domain, retriever, qid] for qid in ids]
        for name, stats in view["contrasts"].items():
            new = scalar_interval([row["contrasts"][name] for row in cell], [row["group"] for row in cell],
                                  seed=plan["bootstrap"]["seed"], repetitions=plan["bootstrap"]["repetitions"])
            for field in stats:
                if field in new:
                    error = float(np.max(np.abs(np.asarray(new[field]) - np.asarray(stats[field]))))
                    largest_interval_error = max(largest_interval_error, error)
                    interval_fields += 1
        for name, mean in view["mrr"].items():
            largest_interval_error = max(largest_interval_error, abs(math.fsum(row["values"][name] for row in cell) / len(cell) - mean))
        print({"scalar_intervals_replayed": f"{domain}/{retriever}"}, flush=True)
    for item in result["cross_retriever_j_differences"]:
        domain = item["domain"]
        ids = bridge["samples"][domain]["main_ids"]
        values = [lookup[domain, "CodeBERT", q]["contrasts"]["j_ENTR_contract_difference"] - lookup[domain, "BM25", q]["contrasts"]["j_ENTR_contract_difference"] for q in ids]
        new = scalar_interval(values, [bridge["samples"][domain]["groups"][q] for q in ids], seed=plan["bootstrap"]["seed"], repetitions=plan["bootstrap"]["repetitions"])
        for field in ("mean", "standard_error_cluster", "ci95_cluster_t", "ci95_group_bootstrap"):
            largest_interval_error = max(largest_interval_error, float(np.max(np.abs(np.asarray(new[field]) - np.asarray(item[field])))))
            interval_fields += 1
    require(largest_interval_error < 1e-12, "Independent scalar interval replay differs")
    acceptance = {"passed": True, "role": plan["role"], "queries_replayed": len(rows),
                  "selections_recomputed": selections_verified, "Fraction_RR_comparisons": rr_comparisons,
                  "Fraction_effect_comparisons": effect_comparisons, "interval_fields_recomputed": interval_fields,
                  "maximum_RR_error": largest_rr_error, "maximum_effect_error": largest_effect_error,
                  "maximum_interval_or_summary_error": largest_interval_error,
                  "frozen_and_protected_files_verified": len(plan["bound_files"]) + len(plan["protected_files"]),
                  "result_sha256": sha(OUT / "RESULTS.json"), "per_query_sha256": sha(OUT / "per_query.jsonl"),
                  "no_new_confirmation": True, "no_model_inference": True}
    with (OUT / "AUDIT.json").open("x", encoding="utf-8") as handle:
        json.dump(acceptance, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(acceptance, flush=True)


if __name__ == "__main__":
    main()
