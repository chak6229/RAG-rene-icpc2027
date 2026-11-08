from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from .analysis import summarize
from .common import OUT, PRIMARY, read
from .inference import deny_labels, stable_top


class ConfirmationContracts(unittest.TestCase):
    def test_top100_boundary_ties_equal_full_stable_sort(self):
        rng=np.random.default_rng(19)
        for n in (3,100,101,500,32768):
            ids=rng.permutation(n)
            for values in (rng.integers(-4,5,n).astype(np.float32),np.zeros(n,np.float32),rng.standard_normal(n).astype(np.float32)):
                self.assertEqual(stable_top(values,ids).tolist(),np.lexsort((ids,-values))[:100].tolist())

    def test_label_guard_prevents_actual_file_reads(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'private'/'qrels.jsonl';p.parent.mkdir();p.write_text('[]',encoding='utf-8')
            with deny_labels():
                with self.assertRaises(PermissionError):p.read_text(encoding='utf-8')
                with self.assertRaises(PermissionError):open(p).read()
                with self.assertRaises(PermissionError):(OUT/'SAMPLE_PLAN.json').read_text(encoding='utf-8')

    def test_shared_paired_bootstrap_and_family_width(self):
        x=np.linspace(-.2,.4,40)
        out=summarize(np.column_stack((x,2*x,x)),['local','base','difference'],['local','base','difference'],repetitions=1000)
        self.assertTrue(np.allclose(out['local']['ci95_bootstrap'],out['difference']['ci95_bootstrap']))
        self.assertTrue(np.allclose(np.array(out['local']['family6_ci_bootstrap'])*2,out['base']['family6_ci_bootstrap']))
        self.assertLess(out['local']['family6_ci_t'][0],out['local']['ci95_t'][0])
        self.assertGreater(out['local']['family6_ci_t'][1],out['local']['ci95_t'][1])

    def test_sparse_or_zero_difference_cannot_establish_equivalence(self):
        for x in (np.zeros(40),np.r_[np.ones(10)*.1,np.zeros(30)]):
            out=summarize(x[:,None],['paired'],['paired'],repetitions=500)['paired']
            self.assertEqual(out['direction'],'unresolved')
            self.assertNotIn('equivalence',out)

    def test_prospective_precision_and_scope(self):
        plan=read(OUT/'PRECISION_PLAN.json');sample=read(OUT/'SAMPLE_PLAN.json')
        self.assertTrue(plan['passed']);self.assertEqual(len(PRIMARY),6)
        self.assertEqual(sample['queries'],plan['planned_queries'])
        self.assertEqual(sample['groups'],sample['queries'])
        self.assertGreaterEqual(sample['corpus_documents'],1.1*sample['queries'])
        for name,v in plan['contrasts'].items():
            if name.startswith('base_minus_local'):self.assertLessEqual(v['simultaneous_CI_halfwidth'],.005)
            else:self.assertGreaterEqual(v['power_at_005'],.95)


if __name__=='__main__':unittest.main()
