from __future__ import annotations

import itertools
import tempfile
import unittest
from pathlib import Path

from .common import deny_labels
from .metrics import rr, summarize, uniform_max


class ScientificContracts(unittest.TestCase):
    def test_exact_uniform_position_mass(self):
        for values in ([0, .5, .5, 1], [0, 0, 0, 1], [.2], [1, 0], [1, 1, 1]):
            k = min(3, len(values))
            brute = sum(max(items) for items in itertools.combinations(values, k)) / len(list(itertools.combinations(values, k)))
            self.assertAlmostEqual(uniform_max(values), brute, places=15)

    def test_single_candidate_and_best3_are_different_actions(self):
        pool = [0, .5, 1]
        self.assertEqual(pool[0], 0)
        self.assertEqual(max(pool[:3]), 1)
        self.assertAlmostEqual(uniform_max(pool, 1), .5)

    def test_rr_depth_and_document_identity(self):
        self.assertEqual(rr(list(range(100)), 0), 1)
        self.assertEqual(rr(list(range(100)), 99), .01)
        self.assertEqual(rr(list(range(100)), 100), 0)
        with self.assertRaises(ValueError):
            rr([0] * 100, 0)

    def test_grouped_estimate_is_query_weighted(self):
        value = summarize([1, 0, 0], ["a", "b", "b"], repetitions=200)
        self.assertAlmostEqual(value["mean"], 1 / 3)
        self.assertEqual(value["groups"], 2)

    def test_paired_constant_difference_has_zero_width(self):
        value = summarize([.5] * 4, ["a", "b", "c", "d"], repetitions=100)
        self.assertEqual(value["ci95_group_bootstrap"], [.5, .5])
        self.assertEqual(value["ci95_cluster_t"], [.5, .5])

    def test_label_access_prohibited_even_through_pathlib(self):
        with tempfile.TemporaryDirectory() as directory:
            label = Path(directory) / "private" / "qrels.json"
            label.parent.mkdir()
            label.write_text("{}")
            with deny_labels():
                with self.assertRaises(PermissionError):
                    label.read_text()
                with self.assertRaises(PermissionError):
                    open(label).read()
            self.assertEqual(label.read_text(), "{}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
