from __future__ import annotations

import unittest
from fractions import Fraction

from .analyze import query_bounds, rr_at
from .collect import validate_return


def query():
    return {'query_id': 'q', 'group_id': 'g', 'stratum': 's', 'frame_n': 16, 'sample_n': 16, 'weight': 1,
            'target_doc_ids': [9], 'rankings': {'original': [1, 2, 9], 'x': [2, 1, 9]},
            'actions': {'RIDGE_verified': 'x', 'RIDGE_risk': 'original'},
            'panels': [{'id': 'a', 'doc_ids': [1], 'prior_labels': None},
                       {'id': 'b', 'doc_ids': [2], 'prior_labels': {'A': {'relevance': 'u'}}},
                       {'id': 't', 'doc_ids': [9], 'prior_labels': {'A': {'relevance': '0'}}}]}


class Contracts(unittest.TestCase):
    def test_unknown_shared_assignment_and_fallback(self):
        records = {r['policy']: r for r in query_bounds(query(), 'A', 2, 3)}
        self.assertEqual(records['RIDGE_verified']['expanded_delta_bounds_fraction'], ['-1/2', '1/2'])
        self.assertEqual(records['RIDGE_risk']['expanded_delta_bounds_fraction'], ['0', '0'])

    def test_unknown_shared_identical_ranking(self):
        q = query()
        q['rankings']['x'] = q['rankings']['original']
        result = query_bounds(q, 'A', 2, 3)[0]
        self.assertEqual(result['expanded_delta_bounds_fraction'], ['0', '0'])

    def test_identical_code_ids_share_relevance(self):
        q = query()
        q['rankings']['x'] = [10, 2, 9]
        q['panels'][0]['doc_ids'].append(10)
        result = query_bounds(q, 'A', 2, 3)[0]
        self.assertEqual(result['expanded_delta_bounds_fraction'], ['0', '0'])

    def test_target_text_closure_is_separate_from_original_qrels(self):
        q = query()
        q['rankings']['original'] = [1, 2, 9]
        q['rankings']['x'] = [10, 2, 9]
        q['panels'][0]['prior_labels'] = {'A': {'relevance': '0'}}
        q['panels'][1]['prior_labels'] = {'A': {'relevance': '0'}}
        q['panels'][2]['doc_ids'].append(10)
        result = query_bounds(q, 'A', 2, 3)[0]
        self.assertEqual(result['original_qrels_delta_fraction'], '0')
        self.assertEqual(result['expanded_delta_bounds_fraction'], ['2/3', '2/3'])

    def test_original_qrels_positive_never_removed(self):
        q = query()
        q['panels'][0]['prior_labels'] = {'A': {'relevance': '0'}}
        q['panels'][1]['prior_labels'] = {'A': {'relevance': '0'}}
        result = query_bounds(q, 'A', 2, 3)[0]
        self.assertEqual(result['expanded_delta_bounds_fraction'], ['0', '0'])

    def test_thresholds_and_nonrelevant_absence(self):
        q = query()
        q['panels'][0]['prior_labels'] = {'A': {'relevance': '1'}}
        q['panels'][1]['prior_labels'] = {'A': {'relevance': '0'}}
        strict = query_bounds(q, 'A', 2, 3)[0]
        partial = query_bounds(q, 'A', 1, 3)[0]
        self.assertEqual(strict['expanded_delta_bounds_fraction'], ['0', '0'])
        self.assertEqual(partial['expanded_delta_bounds_fraction'], ['-1/2', '-1/2'])
        self.assertEqual(rr_at([1, 2, 9], {9}, 1), Fraction(0))

    def test_missing_later_fully_judged(self):
        q = query()
        result = query_bounds(q, 'A', 2, 1, {('q', 'a'): {'relevance': '2'}})[0]
        self.assertEqual(result['expanded_delta_bounds_fraction'], ['-1', '0'])
        self.assertEqual(result['missing_panels'], 0)

    def test_return_requires_full_independent_packet(self):
        packet = {'packet_id': 'p', 'cases': [{'case_id': 'c', 'query_id': 'q', 'document_map': {'C1': 'panel'}}]}
        payload = {'schema_version': 1, 'phase': 'relevance_supplement', 'reviewer': 'A', 'packet_id': 'p',
                   'complete': True, 'independence_confirmed': True,
                   'answers': {'c': {'documents': {'C1': {'relevance': 'u', 'note': '需要额外调用语境'}}}}}
        self.assertEqual(len(validate_return(payload, 'A', packet)), 1)
        payload['answers']['c']['documents'] = {}
        with self.assertRaises(ValueError):
            validate_return(payload, 'A', packet)


if __name__ == '__main__':
    unittest.main()
