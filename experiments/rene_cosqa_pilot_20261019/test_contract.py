"""Meaningful regression tests for the denominator, grouping and action contracts."""
from __future__ import annotations

import copy
import importlib.util
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from .evaluation import clustered_stats, holm, per_query_results, validate_inputs
from .ranking import BM25, lexical_tokens
from .selection import select, selection_rows


def fixture():
    queries = [{"query_id": "q1", "query": "alpha beta gamma delta"}, {"query_id": "q2", "query": "red green blue white"}]
    corpus = [{"doc_id": index, "code_text": str(index)} for index in range(1, 10)]
    protocol = {"query_ids": ["q1", "q2"], "generation": {"position_policy": "source_1_to_n"},
                "candidate_k": 3, "ranking_depth": 3, "metric_k": 3, "rand_seeds": [7, 11]}
    candidates, rankings = [], []
    for query, target in zip(queries, (1, 2)):
        query_id = query["query_id"]
        rankings.append({"query_id": query_id, "policy": "NoRewrite", "position": None, "text": query["query"],
                         "ranked_doc_ids": [9, target, 8], "scores": [3., 2., 1.]})
        for position in (1, 2, 3, 4):
            text = f'{query["query"]} extra{position}'
            candidates.append({"query_id": query_id, "position": position, "rewrite": text,
                               "full_vocab_entropy": float(position), "label_mean_log_probability": -float(position)})
            ranking = [target, 9, 8] if target == 1 or position == 4 else [9, 8, 7]
            rankings.append({"query_id": query_id, "policy": "Candidate", "position": position, "text": text,
                             "ranked_doc_ids": ranking, "scores": [3., 2., 1.]})
    selections = selection_rows(candidates, protocol["query_ids"], protocol)
    return {"queries": queries, "corpus": corpus, "candidates": candidates, "rankings": rankings,
            "selections": selections, "groups": {"q1": "g1", "q2": "g2"}, "protocol": protocol}


class Contracts(unittest.TestCase):
    def setUp(self):
        self.data = fixture()

    def reject(self):
        with self.assertRaises(ValueError):
            validate_inputs(**self.data)

    def test_complete_zero_delta(self):
        rows = per_query_results(**self.data, qrels=[{"query_id": "q1", "doc_id": 1}, {"query_id": "q2", "doc_id": 2}])
        self.assertEqual(len(rows), 2)
        self.assertEqual(sum(row["policies"]["ENTR"]["top1_rr"] - row["NoRewrite"] for row in rows) / 2, 0)

    def test_missing_policy_rejected(self):
        self.data["selections"] = [row for row in self.data["selections"] if not (row["query_id"] == "q2" and row["policy"] == "ENTR")]
        self.reject()

    def test_missing_baseline_rejected(self):
        self.data["rankings"] = [row for row in self.data["rankings"] if not (row["query_id"] == "q2" and row["position"] is None)]
        self.reject()

    def test_missing_query_rejected(self):
        self.data["rankings"] = [row for row in self.data["rankings"] if row["query_id"] != "q2"]
        self.reject()

    def test_best4_rejected(self):
        self.data["selections"][0]["selected"].append({"position": 4, "selected_rank": 4})
        self.reject()

    def test_missing_candidate_rejected(self):
        self.data["candidates"].pop()
        self.reject()

    def test_duplicate_baseline_rejected(self):
        self.data["rankings"].append(copy.deepcopy(self.data["rankings"][0]))
        self.reject()

    def test_duplicate_candidate_rejected(self):
        self.data["candidates"].append(copy.deepcopy(self.data["candidates"][0]))
        self.reject()

    def test_duplicate_selection_rejected(self):
        self.data["selections"].append(copy.deepcopy(self.data["selections"][0]))
        self.reject()

    def test_duplicate_query_rejected(self):
        self.data["queries"].append(copy.deepcopy(self.data["queries"][0]))
        self.reject()

    def test_unknown_doc_rejected(self):
        self.data["rankings"][0]["ranked_doc_ids"][0] = 1000
        self.reject()

    def test_duplicate_doc_rejected(self):
        self.data["rankings"][0]["ranked_doc_ids"][0] = self.data["rankings"][0]["ranked_doc_ids"][1]
        self.reject()

    def test_nonfinite_score_rejected(self):
        self.data["rankings"][0]["scores"][0] = float("nan")
        self.reject()

    def test_tie_order_rejected(self):
        self.data["rankings"][0]["scores"] = [3., 3., 1.]
        self.reject()

    def test_truncated_ranking_rejected(self):
        self.data["rankings"][0]["ranked_doc_ids"].pop()
        self.reject()

    def test_selected_rank_not_file_order(self):
        for row in self.data["selections"]:
            row["selected"].reverse()
        validate_inputs(**self.data)

    def test_duplicate_selected_rank_rejected(self):
        self.data["selections"][0]["selected"][1]["selected_rank"] = 1
        self.reject()

    def test_wrong_position_selection_rejected(self):
        self.data["selections"][0]["selected"][2]["position"] = 4
        self.reject()

    def test_missing_group_rejected(self):
        del self.data["groups"]["q2"]
        self.reject()

    def test_missing_qrels_rejected(self):
        with self.assertRaises(ValueError):
            per_query_results(**self.data, qrels=[{"query_id": "q1", "doc_id": 1}])

    def test_short_query_uses_actual_pool(self):
        data = fixture()
        data["queries"] = [{"query_id": "q1", "query": "alpha"}]
        data["protocol"]["query_ids"] = ["q1"]
        data["groups"] = {"q1": "g1"}
        data["candidates"] = data["candidates"][:1]
        data["rankings"] = data["rankings"][:2]
        data["rankings"][0]["text"] = "alpha"
        data["selections"] = selection_rows(data["candidates"], ["q1"], data["protocol"])
        self.assertTrue(all(len(row["selected"]) == 1 for row in data["selections"]))
        validate_inputs(**data)

    def test_sparse_zero_is_not_equivalence(self):
        stats = clustered_stats([0, 0], ["g1", "g1"], bootstrap=100, seed=7, margin=.005)
        self.assertIsNone(stats["equivalent"])
        self.assertIsNone(stats["p_two_sided_cluster_t"])
        self.assertEqual(stats["groups"], 1)

    def test_clustered_weighted_query_mean(self):
        values = [1., 1., -1., 0.]
        stats = clustered_stats(values, ["a", "a", "b", "c"], bootstrap=500, seed=7, margin=.005)
        expected_se = math.sqrt(3 / 2 * ((2 - .25 * 2) ** 2 + (-1 - .25) ** 2 + (0 - .25) ** 2)) / 4
        self.assertAlmostEqual(stats["mean"], .25)
        self.assertAlmostEqual(stats["cluster_robust_se"], expected_se)

    def test_holm_full_family(self):
        result = holm({"a": .01, "b": .03, "c": .04, "d": 1.})
        self.assertEqual(result, {"a": .04, "b": .09, "c": .09, "d": 1.})

    def test_bm25_hand_calculation_and_query_deduplication(self):
        model = BM25(["alpha alpha", "beta", "alpha beta beta", "gamma"])
        expected = math.log1p((4 - 2 + .5) / (2 + .5)) * 2 * 2.2 / (2 + 1.2 * (.25 + .75 * 2 / 1.75))
        self.assertAlmostEqual(model.score("alpha")[0], expected)
        np.testing.assert_array_equal(model.score("alpha"), model.score("alpha alpha"))
        self.assertEqual(lexical_tokens("loadJSON_file42"), ["load", "json", "file42"])

    def test_generic_loader_rejects_duplicate_queries(self):
        source = Path(__file__).resolve().parents[1] / "rene_ssqr_alignment_20261019/rne_eval.py"
        spec = importlib.util.spec_from_file_location("independent_rne_eval", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "rank.jsonl"
            path.write_text('{"query_id":"q1","doc_ids":[1]}\n{"query_id":"q1","doc_ids":[2]}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                module.load_rankings(str(path))

    def test_generic_loader_rejects_duplicate_docs(self):
        source = Path(__file__).resolve().parents[1] / "rene_ssqr_alignment_20261019/rne_eval.py"
        spec = importlib.util.spec_from_file_location("independent_rne_eval_docs", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "rank.jsonl"
            path.write_text('{"query_id":"q1","doc_ids":[1,1]}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                module.load_rankings(str(path))


if __name__ == "__main__":
    unittest.main(verbosity=2)
