"""Representation/objective regression checks against the pinned author implementation."""
from __future__ import annotations

import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from transformers import RobertaConfig, RobertaModel

from .common import ROOT
from .retriever import dot_product_scores, full_batch_loss, pooler_vectors, verify_checkpoint_contract


class RetrieverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(751)
        config = RobertaConfig(vocab_size=83, hidden_size=24, num_hidden_layers=2,
                               num_attention_heads=4, intermediate_size=32, max_position_embeddings=66,
                               hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0, pad_token_id=1)
        cls.model = RobertaModel(config).eval()
        path = ROOT / "results/rene_cosqa_ssqr_audit_20261019/selected_source/ssqr/Query Reformulation/code/model.py"
        spec = importlib.util.spec_from_file_location("pinned_author_model", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.author_class = module.Model

    def test_vectors_loss_and_gradients_match_author(self):
        ours = copy.deepcopy(self.model)
        theirs = copy.deepcopy(self.model)
        author = self.author_class(theirs, theirs.config, None, None)
        code = torch.randint(3, 80, (32, 13))
        query = torch.randint(3, 80, (32, 13))
        code[:, -2:] = 1
        query[:, -3:] = 1
        loss, cv, qv = full_batch_loss(ours, code, query)
        ref_loss, ref_cv, ref_qv = author(code, query)
        torch.testing.assert_close(cv, ref_cv, atol=1e-7, rtol=1e-6)
        torch.testing.assert_close(qv, ref_qv, atol=1e-7, rtol=1e-6)
        torch.testing.assert_close(loss, ref_loss, atol=1e-6, rtol=1e-6)
        loss.backward()
        ref_loss.backward()
        self.assertEqual((qv @ cv.T).shape, (32, 32))
        self.assertGreater(float(ours.pooler.dense.weight.grad.norm()), 0)
        for (name, param), (_, reference) in zip(ours.named_parameters(), theirs.named_parameters()):
            if param.grad is not None:
                torch.testing.assert_close(param.grad, reference.grad, atol=1e-7, rtol=1e-4, msg=name)

    def test_dot_product_preserves_norm_information(self):
        score = dot_product_scores([[1, 0]], [[2, 0], [1, 0]])
        np.testing.assert_array_equal(score, [[2, 1]])
        self.assertNotEqual(float(score[0, 0]), float(score[0, 1]))

    def test_no_silent_cls_fallback(self):
        with self.assertRaises(ValueError):
            pooler_vectors(SimpleNamespace(last_hidden_state=torch.ones(2, 4, 8)))
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                verify_checkpoint_contract(Path(folder))

    def test_pairing_and_negative_contract(self):
        with self.assertRaises(ValueError):
            full_batch_loss(self.model, torch.ones(1, 4, dtype=torch.long), torch.ones(1, 4, dtype=torch.long))
        with self.assertRaises(ValueError):
            full_batch_loss(self.model, torch.ones(2, 4, dtype=torch.long), torch.ones(3, 4, dtype=torch.long))


if __name__ == "__main__":
    unittest.main()
