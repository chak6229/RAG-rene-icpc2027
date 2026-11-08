"""Tests for the mechanisms that could invalidate this confirmation."""
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

import numpy as np

from experiments.rene_fixed_action_uq_20261019.test_contract import ContractTests
from experiments.rene_uq_independent_proof_20261019 import inference_stats as s
from experiments.rene_uq_independent_proof_20261019.evaluate import method_summary, proper_score_vectors
from experiments.rene_uq_independent_proof_20261019.stages import bounds_for_trace
from experiments.rene_uq_independent_proof_20261019.study import FRESH,OUT,PRIOR,backend
from experiments.rene_strict_confirmation_20261019.common import rr,sha,utc,write_json


class ConfirmationTests(unittest.TestCase):
    def test_backend_writes_are_redirected(self):
        engine=backend()
        self.assertEqual(engine.FRESH,FRESH)
        self.assertEqual(engine.OUT,OUT)
        self.assertNotEqual(engine.FRESH,PRIOR/"fresh")
        with self.assertRaises(RuntimeError):
            engine.inventory()

    def test_label_and_training_reads_are_rejected(self):
        code = (
            "from experiments.rene_uq_independent_proof_20261019.study import deny_outcome_reads;"
            "deny_outcome_reads();"
            "\nfor path in ['x/private_outcomes.json','x/data/codexglue/train.jsonl',"
            "'x/rene_uq_independent_proof_20261019/training/no_seed_uq.pkl']:"
            "\n try: open(path)"
            "\n except PermissionError: pass"
            "\n else: raise AssertionError('unguarded read')"
        )
        value=subprocess.run([sys.executable,"-X","utf8","-c",code],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(value.returncode,0,value.stderr)

    def test_geometry_bounds_are_exact_over_all_target_ids(self):
        original=list(range(100))
        action=[200,3,2,1,0]+list(range(5,100))
        for sign in (-1,0,1):
            values=[sign*(rr(action,t)-rr(original,t)) for t in range(250)]
            lo,hi=bounds_for_trace(original,action,sign)
            self.assertAlmostEqual(lo,min(values))
            self.assertAlmostEqual(hi,max(values))

    def test_identical_decisions_do_not_prove_population_equivalence(self):
        x=np.zeros(32)
        v=s.intervals(x,s.block_bootstrap_indices(32,100))
        self.assertEqual(v["simultaneous_envelope"],[0.,0.])
        self.assertFalse(s.verdict(v)["equivalence_established"])
        self.assertEqual(s.verdict(v)["direction"],"unresolved")

    def test_cluster_interval_preserves_eight_query_dependence(self):
        x=np.repeat([-.1,.1,-.2,.2],8)
        draws=s.block_bootstrap_indices(len(x),10000)
        result=s.intervals(x,draws)
        expected=np.quantile(np.array([-.1,.1,-.2,.2])[draws].mean(1),[.025,.975])
        np.testing.assert_allclose(result["descriptive_ci95"],expected)
        self.assertEqual(result["nonzero_generation_blocks"],4)
        with self.assertRaises(ValueError):
            s.block_bootstrap_indices(9)

    def test_fixed_half_bootstrap_reselects_batch_policy(self):
        n=16
        d=np.array([1.,-.5]+[0.]*14)
        eligible=np.zeros(n,bool);eligible[[0,1,8]]=True
        a=np.ones(n);b=np.ones(n)
        a[[0,1,8]]=[.1,.2,.3];b[[0,1,8]]=[.3,.1,.2]
        draws=np.array([[0,0],[1,1],[0,1],[1,0]],np.int32)
        result=s.fixed_half_contrast(d,eligible,a,b,np.arange(n),draws)
        self.assertAlmostEqual(result["mean"],(-.5-1)/16)
        manual=[]
        for blockdraw in draws:
            repeated=[q for block in blockdraw for q in range(block*8,block*8+8)]
            ix=[q for q in repeated if eligible[q]]
            count=len(ix)//2
            picked_a=sorted(ix,key=lambda q:(a[q],q))[:count]
            picked_b=sorted(ix,key=lambda q:(b[q],q))[:count]
            manual.append((sum(d[q] for q in picked_b)-sum(d[q] for q in picked_a))/n)
        np.testing.assert_allclose(result["descriptive_ci95"],np.quantile(manual,[.025,.975]))
        self.assertTrue(result["resampled_adoption_recomputed"])

    def test_proper_score_orientation_and_no_outcome_qid_shortcut(self):
        delta=np.array([-.5,0.,.5])
        good=np.array([[.8,.1,.1],[.1,.8,.1],[.1,.1,.8]])
        bad=np.full((3,3),1/3)
        for name,value in proper_score_vectors(delta,good).items():
            self.assertLess(value.mean(),proper_score_vectors(delta,bad)[name].mean())
        ids=list(range(2000000,2000008))
        originals={q:[7,8] for q in ids}
        targets={q:8 for q in ids}
        traces=[dict(query_index=q,ranking=[8,7],accepted=True,verified=False,
                     billed_slots=0,actual_pairs=0,executed_ce_tokens=0) for q in ids]
        result,x=method_summary(traces,originals,targets,s.block_bootstrap_indices(8,20))
        self.assertEqual(result["adopted"],8)
        self.assertAlmostEqual(result["delta_mrr"],.5)
        np.testing.assert_array_equal(x,np.full(8,.5))


if __name__ == "__main__":
    suite=unittest.TestSuite([
        unittest.defaultTestLoader.loadTestsFromTestCase(ContractTests),
        unittest.defaultTestLoader.loadTestsFromTestCase(ConfirmationTests),
    ])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if result.wasSuccessful():
        write_json(OUT/"TESTS.json",dict(
            passed=True,tests=result.testsRun,completed_at_utc=utc(),
            test_sources={str(Path(p).relative_to(ROOT)):sha(p) for p in (
                __file__,ROOT/"experiments/rene_fixed_action_uq_20261019/test_contract.py")},
        ),immutable=True)
    sys.exit(0 if result.wasSuccessful() else 1)
