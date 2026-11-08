import io
import itertools
import pickle
import unittest
from fractions import Fraction

import numpy as np

from experiments.rene_budget_aggregation_20261020.metrics import exact_random_max,factorial_values,contrasts
from .analysis import rr,summary
from .inference import deny_labels,stable_order
from .common import OUT
from .staqc import DataUnpickler,code_tokens
from . import audit


class ConfirmationTests(unittest.TestCase):
    def test_restricted_data_pickle(self):
        v={42:'title',(42,0):'code'}
        self.assertEqual(DataUnpickler(io.BytesIO(pickle.dumps(v,protocol=2)),encoding='latin1').load(),v)
        with self.assertRaises(pickle.UnpicklingError):
            DataUnpickler(io.BytesIO(pickle.dumps(len))).load()

    def test_code_lexing(self):
        s,f=code_tokens('x = Foo(1) # comment\n')
        self.assertEqual(s,'x = Foo ( 1 )');self.assertFalse(f)
        s,f=code_tokens('x = (')
        self.assertTrue(f);self.assertEqual(s,'x = (')

    def test_random_max_against_enumeration_with_duplicates(self):
        vals=[0.,.5,.5,1.]
        for k in (1,2,3):
            truth=sum(max(c) for c in itertools.combinations(vals,k))/len(list(itertools.combinations(vals,k)))
            self.assertAlmostEqual(exact_random_max(vals,k),truth)

    def test_factor_identity_and_short_pool(self):
        for scores in ({1:1.,2:0.,3:.5},{1:.5}):
            orders={'ENTR':list(scores),'PROB':list(reversed(scores))}
            v=factorial_values(scores,orders,0.,0.)
            c=contrasts(v)
            self.assertAlmostEqual(c['ENTR_J_exact'],c['ENTR_B_budget_mean_1_to_3']+c['ENTR_G_aggregation_at3_mean_to_max'])
            self.assertEqual(v['RAND_exact_k1_mean'],v['RAND_exact_k3_mean'])
            self.assertEqual(v['ENTR_k1_first'],v['ENTR_k3_first'])

    def test_rr_cutoff_and_deterministic_ties(self):
        self.assertEqual(rr(list(range(101)),{100}),0.)
        self.assertEqual(rr([3,1,2],{1}),.5)
        self.assertEqual(stable_order(np.array([1.,2.,2.]),np.array([2,1,0])).tolist(),[2,1,0])

    def test_label_guards_builtin_path_io(self):
        p=OUT/'private/qrels.jsonl'
        with deny_labels():
            for fn in (lambda:open(p),lambda:p.open(),lambda:io.open(p)):
                with self.assertRaises(PermissionError):fn()


if __name__=='__main__':unittest.main()
