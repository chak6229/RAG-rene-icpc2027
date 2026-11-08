from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from .t5 import full_vocab_statistics, insertion_inputs, select_candidates


class EntropyTests(unittest.TestCase):
    def test_full_vocabulary_known_distributions(self):
        p = torch.tensor([[[.25, .25, .25, .25], [.7, .1, .1, .1]]])
        scores = full_vocab_statistics(p.log(), torch.tensor([[0, 0]]))
        reference = np.mean([math.log(4), -.7 * math.log(.7) - .3 * math.log(.1)])
        self.assertAlmostEqual(float(scores["full_vocab_entropy"][0]), reference, places=6)
        self.assertAlmostEqual(float(scores["label_mean_log_probability"][0]), (math.log(.25) + math.log(.7)) / 2, places=6)
        # Generated-token surprisal is a different quantity.
        self.assertGreater(abs(reference + float(scores["label_mean_log_probability"][0])), .1)

    def test_padding_is_ignored_and_span_score_is_separate(self):
        logits = torch.tensor([[[2., 0., -1.], [0., 0., 0.], [900., -900., 0.]]])
        first = full_vocab_statistics(logits[:, :2], torch.tensor([[0, 1]]), lexical_mask=torch.tensor([[False, True]]))
        padded = full_vocab_statistics(logits, torch.tensor([[0, 1, -100]]), lexical_mask=torch.tensor([[False, True, False]]))
        for field in first:
            torch.testing.assert_close(first[field], padded[field])
        self.assertAlmostEqual(float(first["span_full_vocab_entropy"][0]), math.log(3), places=6)
        self.assertNotEqual(float(first["full_vocab_entropy"][0]), float(first["span_full_vocab_entropy"][0]))

    def test_entropy_does_not_collapse_to_topk(self):
        logits = torch.zeros(1, 1, 1000)
        result = full_vocab_statistics(logits, torch.tensor([[17]]))
        self.assertAlmostEqual(float(result["full_vocab_entropy"][0]), math.log(1000), places=5)

    def test_positions_and_reserved_tokens(self):
        self.assertEqual([r["position"] for r in insertion_inputs("find list items")], [1, 2, 3])
        self.assertEqual([r["position"] for r in insertion_inputs("find list items", position_policy="paper_0_to_n")], [0, 1, 2, 3])
        with self.assertRaises(ValueError):
            insertion_inputs("find <extra_id_0> item")

    def test_selection_shared_cache_ties_no_padding_and_rand_reproducible(self):
        records = [{"query_id": "q", "position": p, "full_vocab_entropy": e, "label_mean_log_probability": lp}
                   for p, e, lp in [(1, 3., -2.), (2, 1., -5.), (3, 1., -1.), (4, 5., -3.)]]
        self.assertEqual([r["position"] for r in select_candidates(records, policy="ENTR")], [2, 3, 1])
        self.assertEqual([r["position"] for r in select_candidates(records, policy="PROB")], [3, 1, 4])
        rand = select_candidates(records, policy="RAND", k=8)
        self.assertEqual(rand, select_candidates(records, policy="RAND", k=8))
        self.assertEqual(len(rand), 4)
        self.assertEqual({r["position"] for r in rand}, {1, 2, 3, 4})
        self.assertEqual(len(select_candidates(records[:1], policy="ENTR", k=3)), 1)

    def test_invalid_logits_or_masks_rejected(self):
        with self.assertRaises(ValueError):
            full_vocab_statistics(torch.zeros(1, 2, 4), torch.tensor([[-100, -100]]))
        with self.assertRaises(FloatingPointError):
            full_vocab_statistics(torch.full((1, 1, 4), float("nan")), torch.tensor([[0]]))
        with self.assertRaises(ValueError):
            full_vocab_statistics(torch.zeros(1, 1, 4), torch.tensor([[-100]]), lexical_mask=torch.tensor([[True]]))


if __name__ == "__main__":
    unittest.main()
