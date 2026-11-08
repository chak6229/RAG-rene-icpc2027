import unittest

import numpy as np

from .common import UnionFind, fingerprints, lexical, norm, repo
from .precision import factor, grouped, make_plan, power, required_groups, variance


class PreflightTests(unittest.TestCase):
    def test_normalization_preserves_code_case(self):
        self.assertEqual(norm('Ａ  Foo\nBAR'), 'a foo bar')
        self.assertEqual(lexical('Foo(bar)!'), 'foo bar')
        self.assertNotEqual(fingerprints({'code': 'Foo'})['code_ws'], fingerprints({'code': 'foo'})['code_ws'])

    def test_function_identity_across_revision(self):
        a = {'repo': 'A/B', 'path': 'x.py', 'func_name': 'f', 'url': 'old'}
        b = {**a, 'url': 'new'}
        self.assertEqual(fingerprints(a)['function'], fingerprints(b)['function'])
        self.assertEqual(repo({'url': 'https://github.com/A/B/blob/main/a.py'}), 'a/b')

    def test_query_code_separation(self):
        a = fingerprints({'query': 'find foo', 'code': 'return foo'})
        b = fingerprints({'query': 'find bar', 'code': 'return foo'})
        self.assertFalse(a['q_lex'] & b['q_lex'])
        self.assertTrue(a['code_ws'] & b['code_ws'])

    def test_transitive_groups(self):
        uf = UnionFind()
        uf.join('q1', 'code'); uf.join('q2', 'code'); uf.join('q2', 'title')
        self.assertEqual(uf.find('q1'), uf.find('title'))
        self.assertNotEqual(uf.find('q1'), uf.find('unrelated'))

    def test_group_variance_against_equal_clusters(self):
        s, n = grouped([1., 1., 3., 3., 5., 5.], ['a','a','b','b','c','c'])
        self.assertAlmostEqual(variance(s, n), np.var([1.,3.,5.], ddof=1))
        self.assertEqual(factor([2,2,2]), 1.)
        self.assertGreater(factor([1,1,10]), 1.)

    def test_power_required_group_boundary(self):
        n = required_groups(.003)
        self.assertGreaterEqual(power(n,.003), .9)
        self.assertLess(power(n-1,.003), .9)
        self.assertGreater(power(1000,.003), power(100,.003))

    def test_insufficient_groups_no_authorization(self):
        self.assertFalse(make_plan({}, [1,1])['eligible'])

    def test_single_query_variance_cannot_be_hidden_by_group_averaging(self):
        from .precision import CONTRASTS
        dev={'source':{c:{'upper95_bootstrap_variance':.001,
             'matched_one_query_per_group_variance_upper95':.02,
             'all_query_variance_upper95_sensitivity':.015,'size_design_factor':2.,'mean':.01} for c in CONTRASTS}}
        plan=make_plan(dev,[1]*100)
        for v in plan['contrasts'].values():self.assertEqual(v['planning_variance'],.02)


if __name__ == '__main__':
    unittest.main()
