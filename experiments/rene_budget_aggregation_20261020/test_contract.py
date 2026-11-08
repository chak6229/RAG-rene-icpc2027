from __future__ import annotations

import builtins
import io
import itertools
import math
import tempfile
import unittest
from pathlib import Path

from .metrics import clustered_summary,contrasts,exact_random_max,factorial_values,ordered_positions
from .run import deny_outcomes


class ScientificContract(unittest.TestCase):
    def test_random_max_all_budgets_matches_exhaustive_subsets_with_ties(self):
        for n in range(1,10):
            values=[((i*7)%4)/4 for i in range(n)]
            for k in (1,2,3):
                brute=[max(values[i] for i in ix) for ix in itertools.combinations(range(n),min(k,n))]
                self.assertAlmostEqual(exact_random_max(values,k),math.fsum(brute)/len(brute),places=14)

    def test_uniform_subset_mean_is_budget_invariant(self):
        values=[0,.5,.5,1,0]
        for k in (1,2,3):
            means=[math.fsum(values[i] for i in ix)/k for ix in itertools.combinations(range(len(values)),k)]
            self.assertAlmostEqual(math.fsum(means)/len(means),math.fsum(values)/len(values),places=14)

    def test_duplicate_positions_keep_probability_mass(self):
        self.assertAlmostEqual(exact_random_max([0,0,1],1),1/3)
        self.assertAlmostEqual(exact_random_max([0,0,1],2),2/3)
        self.assertNotAlmostEqual(exact_random_max([0,0,1],1),exact_random_max([0,1],1))

    def test_single_candidate_and_first_anchor_identities(self):
        v=factorial_values({1:0,2:1,3:.5},{'ENTR':[2,3,1],'PROB':[3,1,2]},.2,.7)
        for p in ('ENTR','PROB','RAND_exact'):
            self.assertEqual(v[f'{p}_k1_first'],v[f'{p}_k1_mean'])
            self.assertEqual(v[f'{p}_k1_mean'],v[f'{p}_k1_max'])
            self.assertEqual(v[f'{p}_k1_first'],v[f'{p}_k3_first'])
        for p in ('ENTR','PROB'):
            self.assertGreaterEqual(v[f'{p}_k3_max'],v[f'{p}_k2_max'])

    def test_short_pool_caps_without_padding(self):
        v=factorial_values({1:.25},{'ENTR':[1],'PROB':[1]},.5,.25)
        for p in ('ENTR','PROB','RAND_exact'):
            for k in (1,2,3):
                for a in ('first','mean','max'):
                    self.assertEqual(v[f'{p}_k{k}_{a}'],.25)
        with self.assertRaises(ValueError):
            exact_random_max([],3)
        with self.assertRaises(ValueError):
            factorial_values({1:.1},{'ENTR':[1],'PROB':[2]},.1,.1)

    def test_path_and_MC_decompositions_hold_per_query(self):
        for values in ([0,1,.25],[.5,.5,.5],[0,.2,.2,.5,1]):
            scores=dict(enumerate(values,1))
            v=factorial_values(scores,{'ENTR':list(scores),'PROB':list(reversed(scores))},.3,.7)
            fx=contrasts(v)
            for p in ('ENTR','PROB'):
                B=fx[f'{p}_B_budget_mean_1_to_3'];G=fx[f'{p}_G_aggregation_at3_mean_to_max']
                self.assertAlmostEqual(fx[f'{p}_J_exact'],B+G,places=14)
                self.assertAlmostEqual(fx[f'{p}_J_RAND20'],B+G+fx['RAND20_minus_exact3'],places=14)
                self.assertAlmostEqual(G,fx['oracle_lift_RAND_exact_k3']-fx[f'oracle_lift_{p}_k3'],places=14)

    def test_score_order_is_label_free_and_ties_use_original_position(self):
        pool=[dict(position=p,full_vocab_entropy=.2,label_mean_log_probability=-.1) for p in (3,1,2)]
        for policy in ('ENTR','PROB'):
            clean=ordered_positions(pool,policy)
            contaminated=[dict(r,rr=100-r['position'],gain=-r['position']) for r in pool]
            self.assertEqual(clean,[1,2,3])
            self.assertEqual(clean,ordered_positions(contaminated,policy))

    def test_query_weighted_group_SE_differs_from_group_mean_weighting(self):
        values=[.1,.3,-.1,.5,.2,.4];groups=['a','a','b','c','c','c']
        mean=math.fsum(values)/6
        se=math.sqrt(3/2*math.fsum(x*x for x in (.4-2*mean,-.1-mean,1.1-3*mean))/36)
        r=clustered_summary([[v] for v in values],groups,['metric'],repetitions=200,seed=2)['metric']
        self.assertAlmostEqual(r['mean'],mean)
        self.assertAlmostEqual(r['se_cluster'],se)
        self.assertNotAlmostEqual(r['mean'],(.2-.1+1.1/3)/3)

    def test_all_zeros_have_exact_zero_intervals(self):
        name='ENTR_B_budget_mean_1_to_3'
        r=clustered_summary([[0],[0],[0]],['a','b','c'],[name],repetitions=200,seed=7)[name]
        for key in ('ci95_group_bootstrap','ci95_cluster_t','family24_ci95_group_bootstrap','family24_ci95_cluster_t'):
            self.assertEqual(r[key],[0,0])

    def test_outcomes_blocked_for_path_io_and_builtin_open(self):
        with tempfile.TemporaryDirectory(prefix='rene-factors-test-') as folder:
            path=Path(folder)/'private'/'qrels.json'
            path.parent.mkdir();path.write_text('{}',encoding='utf-8')
            with deny_outcomes():
                for opener in (lambda:path.read_text(),lambda:io.open(path),lambda:builtins.open(path)):
                    with self.assertRaises(PermissionError):
                        opener()
            self.assertEqual(path.read_text(),'{}')


if __name__=='__main__':
    unittest.main()
