"""CPU-only numerical replay of de-identified, cached experimental outcomes."""
import argparse
import csv
import gzip
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import t
from threadpoolctl import threadpool_limits


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(root):
    manifest = read(root / "MANIFEST.json")
    for item in manifest["files"]:
        path = (root / item["path"]).resolve()
        require(root in path.parents, "Unsafe manifest path")
        require(path.is_file(), "Missing file: " + item["path"])
        require(hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"],
                "Checksum mismatch: " + item["path"])
    print("Integrity: {} files PASS".format(len(manifest["files"])), flush=True)


def close(actual, expected, label):
    require(np.allclose(actual, expected, rtol=0, atol=1e-11),
            "Numerical mismatch: " + label + ": " + str(actual) + " / " + str(expected))


def factorial(arm):
    ranks = arm["ranks"]
    require(len(ranks) > 0 and all(isinstance(r, int) and 0 <= r <= 100 for r in ranks),
            "RR@100 ranks must be integers in [0,100]")
    require(isinstance(arm["baseline_rank"], int) and 0 <= arm["baseline_rank"] <= 100,
            "Invalid original-query rank")
    scores = [1.0 / r if r else 0.0 for r in ranks]
    n = len(scores)
    values = {"NoRewrite": 1.0 / arm["baseline_rank"] if arm["baseline_rank"] else 0.0}
    uniform = math.fsum(scores) / n
    ordered = sorted(scores)
    for k in (1, 2, 3):
        effective = min(k, n)
        best = math.fsum(v * math.comb(i, effective - 1) / math.comb(n, effective)
                         for i, v in enumerate(ordered) if i >= effective - 1)
        values["RAND_exact_k{}_mean".format(k)] = uniform
        values["RAND_exact_k{}_max".format(k)] = best
        for policy in ("ENTR", "PROB"):
            positions = arm["orders"][policy]
            require(len(positions) >= min(3, n) and len(set(positions)) == len(positions)
                    and all(isinstance(p, int) and 1 <= p <= n for p in positions),
                    "Invalid selected positions")
            selected = [scores[p - 1] for p in positions[:effective]]
            values["{}_k{}_mean".format(policy, k)] = math.fsum(selected) / effective
            values["{}_k{}_max".format(policy, k)] = max(selected)
    effects = {}
    for policy in ("ENTR", "PROB"):
        for k in (1, 2, 3):
            for aggregation in ("mean", "max"):
                name = "{}_k{}_{}".format(policy, k, aggregation)
                effects["S_" + name] = values[name] - values["RAND_exact_k{}_{}".format(k, aggregation)]
        effects[policy + "_B"] = effects["S_" + policy + "_k1_mean"] - effects["S_" + policy + "_k3_mean"]
        effects[policy + "_G"] = effects["S_" + policy + "_k3_mean"] - effects["S_" + policy + "_k3_max"]
        effects[policy + "_J"] = effects[policy + "_B"] + effects[policy + "_G"]
    for component in ("B", "G", "J"):
        effects["ENTR_minus_PROB_" + component] = effects["ENTR_" + component] - effects["PROB_" + component]
    return {**values, **effects}


def statistics(matrix, groups, config, full):
    x = np.asarray(matrix, dtype=np.float64)
    require(x.ndim == 2 and len(x) == len(groups) and np.isfinite(x).all(), "Invalid metric matrix")
    if config.get("query_order_groups"):
        require(len(set(groups)) == len(groups), "Expected one query per group")
        sums, sizes = x, np.ones(len(x))
        codes = np.arange(len(x))
    else:
        labels = sorted(set(groups))
        mapping = {label: i for i, label in enumerate(labels)}
        codes = np.array([mapping[g] for g in groups])
        sizes = np.bincount(codes).astype(float)
        sums = np.zeros((len(sizes), x.shape[1]))
        np.add.at(sums, codes, x)
    g = len(sizes)
    require(g > 1, "At least two groups required")
    mean = x.mean(0)
    se = np.sqrt(g / (g - 1) * np.square(sums - sizes[:, None] * mean).sum(0) / len(x) ** 2)
    changed = np.zeros((g, x.shape[1]), dtype=bool)
    np.logical_or.at(changed, codes, np.abs(x) > 1e-12)
    family = config["family"]
    critical, adjusted = t.ppf(.975, g - 1), t.ppf(1 - .025 / family, g - 1)
    output = []
    for i in range(x.shape[1]):
        output.append({"mean": float(mean[i]), "se": float(se[i]), "queries": len(x), "groups": g,
                       "groups_with_nonzero": int(changed[:, i].sum()),
                       "t": [float(mean[i] - critical * se[i]), float(mean[i] + critical * se[i])],
                       "family_t": [float(mean[i] - adjusted * se[i]), float(mean[i] + adjusted * se[i])]})
    if full:
        repetitions = config["repetitions"]
        bootstrap = np.empty((repetitions, x.shape[1]))
        rng = np.random.default_rng(config["seed"])
        # Every contrast in a study uses the same group draws, preserving pairing.
        with threadpool_limits(limits=4):
            for start in range(0, repetitions, 16):
                draws = rng.integers(0, g, (min(16, repetitions - start), g))
                counts = np.stack([np.bincount(draw, minlength=g) for draw in draws]).astype(float)
                bootstrap[start:start + len(draws)] = (counts @ sums) / (counts @ sizes)[:, None]
        quantiles = np.quantile(bootstrap, [.025, .975, .025 / family, 1 - .025 / family], axis=0)
        for i, result in enumerate(output):
            result["bootstrap"] = quantiles[:2, i].tolist()
            result["family_bootstrap"] = quantiles[2:, i].tolist()
            lo = min(result["family_t"][0], result["family_bootstrap"][0])
            hi = max(result["family_t"][1], result["family_bootstrap"][1])
            result["direction"] = ("positive" if lo > 0 and result["groups_with_nonzero"] >= 20 else
                                   "negative" if hi < 0 and result["groups_with_nonzero"] >= 20 else "unresolved")
    return output


def factor_rows(records, spec):
    output = []
    for row in records:
        arms = {name: factorial(arm) for name, arm in row["arms"].items()}
        metrics = {arm + "::" + key: value for arm, values in arms.items() for key, value in values.items()}
        for pair, (left, right) in spec.get("pairs", {}).items():
            require(row["arms"][left]["baseline_rank"] == row["arms"][right]["baseline_rank"]
                    or not spec.get("same_baseline", False), "Paired original controls differ")
            metrics.update({pair + "::" + key: arms[left][key] - arms[right][key] for key in arms[left]})
        output.append(metrics)
    return output


def stage_rows(records):
    metrics = []
    base = np.array([row["baseline"] for row in records])
    opportunity = np.array([max(0., row["best3"]["ENTR"] - row["baseline"]) for row in records])
    summaries = {}
    for name in records[0]["policies"]:
        rows = [row["policies"][name] for row in records]
        delta = np.array([row["rr"] for row in rows]) - base
        summaries[name] = {"delta": float(delta.mean()), "mrr": float((base + delta).mean()),
                           "adopted": sum(row["adopted"] for row in rows), "harmed": int((delta < -1e-10).sum()),
                           "improved": int((delta > 1e-10).sum()), "CE_pairs": sum(row["pairs"] for row in rows),
                           "CE_tokens": sum(row["tokens"] for row in rows),
                           "positive_gain": float(np.maximum(delta, 0.).mean()),
                           "loss_magnitude": float(np.maximum(-delta, 0.).mean()),
                           "retention": float(np.maximum(delta, 0.).sum() / opportunity.sum())}
    for row in records:
        current = {name: policy["rr"] - row["baseline"] for name, policy in row["policies"].items()}
        for left, right in (("RIDGE_gain", "NoRewrite"), ("RIDGE_verified", "RIDGE_gain"),
                            ("RIDGE_risk", "RIDGE_verified"), ("RIDGE_risk", "NoRewrite"),
                            ("RIDGE_gain", "HGB_gain_control")):
            current[left + "_minus_" + right] = current[left] - current[right]
        metrics.append(current)
    return metrics, summaries


def run_study(root, spec, full):
    with gzip.open(root / spec["input"], "rt", encoding="ascii") as stream:
        records = [json.loads(line) for line in stream]
    require(len(records) == spec["queries"], "Wrong query count")
    require(len({row["q"] for row in records}) == len(records), "Repeated query identifier")
    groups = [row["g"] for row in records]
    require(len(set(groups)) == spec["groups"], "Wrong group count")
    if spec["kind"] == "factors":
        metrics = factor_rows(records, spec)
        for name, expected in spec["curve_means"].items():
            close(np.mean([row[name] for row in metrics]), expected, spec["id"] + "/" + name)
    elif spec["kind"] == "stages":
        metrics, summaries = stage_rows(records)
        for policy, fields in spec["stage_summaries"].items():
            for key, expected in fields.items():
                close(summaries[policy][key], expected, spec["id"] + "/" + policy + "/" + key)
    elif spec["kind"] == "confirmation":
        metrics = [{"ENTR_minus_RAND20": row["entr_best3"] - row["rand20_best3"]} for row in records]
    else:
        raise ValueError("Unknown study kind")
    names = list(spec["estimates"])
    computed = statistics([[row[name] for name in names] for row in metrics], groups, spec["statistics"], full)
    results = []
    for name, actual in zip(names, computed):
        expected = spec["estimates"][name]
        for field, value in expected.items():
            if not full and field in ("bootstrap", "family_bootstrap", "direction"):
                continue
            if isinstance(value, str):
                require(actual[field] == value, "Direction mismatch: " + spec["id"] + "/" + name)
            else:
                close(actual[field], value, spec["id"] + "/" + name + "/" + field)
        if "family_t" not in expected:
            for field in ("family_t", "family_bootstrap", "direction"):
                actual.pop(field, None)
        if spec["kind"] == "confirmation":
            actual["equivalent_within_005"] = (min(actual["t"][0], actual.get("bootstrap", actual["t"])[0]) > -.005
                                               and max(actual["t"][1], actual.get("bootstrap", actual["t"])[1]) < .005)
            require(actual["equivalent_within_005"], "Primary equivalence rule not reproduced")
        results.append({"study": spec["id"], "contrast": name, **actual})
    print("{}: {}/{} queries/groups, {} contrasts PASS".format(spec["id"], len(records), len(set(groups)), len(names)), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Check means, curves, t intervals and stage accounting; omit bootstrap")
    parser.add_argument("--output", type=Path, default=Path("reproduced"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    verify(root)
    config = read(root / "STUDIES.json")
    started = time.perf_counter()
    results = []
    for study in config["studies"]:
        results.extend(run_study(root, study, not args.quick))
    output = args.output.resolve()
    require((output != root and root not in output.parents) or output == root / "reproduced",
            "Use the default reproduced directory or an output directory outside the package")
    output.mkdir(parents=True, exist_ok=True)
    acceptance = {"passed": True, "full_bootstrap": not args.quick, "studies": len(config["studies"]),
                  "contrasts": len(results), "seconds": time.perf_counter() - started,
                  "scope": "cached rank/metric numerical replay; no generation, retrieval, training or reannotation",
                  "statistics_tolerance": 1e-11, "python": sys.version.split()[0], "numpy": np.__version__}
    (output / "ACCEPTANCE.json").write_text(json.dumps(acceptance, indent=2) + "\n", encoding="utf-8")
    (output / "RESULTS.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    with (output / "RESULTS.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["study", "contrast", "mean", "t", "bootstrap", "direction"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(json.dumps(acceptance), flush=True)


if __name__ == "__main__":
    main()
