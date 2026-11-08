from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from .common import write_json
from .pipeline import load_trained_retriever, read_public, require_development_roles
from .retriever import REPRESENTATION


class PipelineTests(unittest.TestCase):
    def test_target_metadata_cannot_enter_public_queries(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "queries.jsonl"
            path.write_text(json.dumps({"query_id": "q", "query": "real query", "docstring_tokens": "TARGET"}), encoding="utf-8")
            with self.assertRaises(ValueError):
                read_public(path, kind="queries")

    def test_qrels_cannot_enter_corpus_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "corpus.jsonl"
            path.write_text(json.dumps({"doc_id": 0, "code_text": "pass", "relevance": 1}), encoding="utf-8")
            with self.assertRaises(ValueError):
                read_public(path, kind="corpus")

    def test_cosqa_execution_waits_for_frozen_development_roles(self):
        queries = [{"query_id": "cosqa-train-1", "query": "python list"}]
        with self.assertRaises(ValueError):
            require_development_roles(queries, None)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "roles.json"
            write_json(path, {"status": "FROZEN", "role": "development", "query_ids": ["cosqa-train-2"]})
            with self.assertRaises(ValueError):
                require_development_roles(queries, path)
            write_json(path, {"status": "FROZEN", "role": "development", "query_ids": ["cosqa-train-1"]})
            require_development_roles(queries, path)

    def test_untrained_and_intermediate_weights_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_json(root / "ssqr_representation_contract.json", REPRESENTATION)
            with self.assertRaisesRegex(ValueError, "incomplete"):
                load_trained_retriever(root, "cpu")
            intermediate = root / "checkpoint-787"
            write_json(intermediate / "ssqr_representation_contract.json", REPRESENTATION)
            write_json(root / "training_complete.json", {"step": 15740, "total_steps": 15740})
            with self.assertRaisesRegex(ValueError, "only the completed"):
                load_trained_retriever(intermediate, "cpu")


if __name__ == "__main__":
    unittest.main()
