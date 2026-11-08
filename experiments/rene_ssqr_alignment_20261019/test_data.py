from __future__ import annotations

import unittest

from .data import adapt, query_input, token_text


class DataTests(unittest.TestCase):
    def test_query_documentation_and_labels_are_separated(self):
        row = {"idx": "q1", "doc": "BING USER QUERY", "code": "raw code", "code_tokens": "def f ( ) : pass",
               "docstring_tokens": "TARGET DOCUMENTATION MUST NOT BECOME QUERY", "label": "1", "retrieval_idx": 0}
        queries, corpus, qrels, negatives = adapt([row], {"raw code": 0})
        self.assertEqual(queries, [{"query_id": "q1", "query": "BING USER QUERY"}])
        self.assertEqual(corpus, [{"doc_id": 0, "code_text": "def f ( ) : pass"}])
        self.assertEqual(qrels, [{"query_id": "q1", "doc_id": 0, "relevance": 1}])
        self.assertEqual(negatives, 0)

    def test_string_tokens_not_joined_characterwise(self):
        self.assertEqual(token_text("list remove empty", field="code_tokens"), "list remove empty")
        self.assertEqual(token_text(["list", "remove", "empty"], field="code_tokens"), "list remove empty")

    def test_missing_real_query_fails(self):
        for row in ({"idx": "q1", "docstring_tokens": "target"}, {"idx": "q2", "doc": "  ", "docstring_tokens": "target"}):
            with self.assertRaises(ValueError):
                query_input(row)

    def test_negative_pair_not_no_answer_query(self):
        row = {"idx": "neg", "doc": "user question", "code": "raw", "code_tokens": "pass", "label": 0}
        queries, corpus, qrels, negatives = adapt([row], {"raw": 0})
        self.assertEqual((queries, qrels, negatives), ([], [], 1))
        self.assertEqual(len(corpus), 1)

    def test_conflicting_mapping_and_duplicate_query_fail(self):
        row = {"idx": "q1", "doc": "query", "code": "raw", "code_tokens": "pass", "label": 1, "retrieval_idx": 1}
        with self.assertRaises(ValueError):
            adapt([row], {"raw": 0})
        row["retrieval_idx"] = 0
        with self.assertRaises(ValueError):
            adapt([row, row], {"raw": 0})


if __name__ == "__main__":
    unittest.main()
