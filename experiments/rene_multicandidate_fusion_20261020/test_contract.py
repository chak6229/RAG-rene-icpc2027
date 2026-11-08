from __future__ import annotations

import unittest

import numpy as np

from .common import fuse, now, unique_text_lists, write
from .metrics import intervals, rr


def row(text, order):
    return {"text": text, "ranked_doc_ids": order, "scores": [100-i for i in range(100)]}


class Contract(unittest.TestCase):
    def test_single_list_reproduces_order(self):
        order=list(range(100,0,-1))
        self.assertEqual(fuse([row("a",order)])[0],order)

    def test_exact_tie_uses_document_id_and_input_order_invariance(self):
        a=row("a",[2,1]+list(range(3,101)))
        b=row("b",[1,2]+list(range(3,101)))
        self.assertEqual(fuse([a,b])[0][:2],[1,2])
        self.assertEqual(fuse([a,b])[0],fuse([b,a])[0])

    def test_same_text_dedup_without_collapsing_different_inputs(self):
        a=row("a",list(range(1,101)))
        b=row("b",list(range(1,101)))
        self.assertEqual(len(unique_text_lists([a,a,b])),2)
        self.assertEqual(fuse([a,a,b])[1]["unique_lists"],2)

    def test_duplicate_text_inconsistent_rank_is_rejected(self):
        a=row("same",list(range(1,101)))
        b=row("same",list(range(100,0,-1)))
        with self.assertRaises(ValueError):
            fuse([a,b])

    def test_fusion_can_exceed_best_single_candidate_rr(self):
        a=row("a",[2,1]+list(range(10,108)))
        b=row("b",[3,1]+list(range(110,208)))
        c=row("c",[4,1]+list(range(210,308)))
        merged=fuse([a,b,c])[0]
        self.assertEqual(rr(merged,{1}),1)
        self.assertEqual(max(rr(r["ranked_doc_ids"],{1}) for r in (a,b,c)),.5)

    def test_original_participation_does_not_guarantee_no_harm(self):
        original=row("original",[1]+list(range(10,109)))
        rewrites=[row(str(j),[2]+list(range(200+100*j,299+100*j))) for j in range(3)]
        merged=fuse([original]+rewrites)[0]
        self.assertLess(rr(merged,{1}),rr(original["ranked_doc_ids"],{1}))

    def test_malformed_rankings_rejected(self):
        with self.assertRaises(ValueError):
            fuse([row("x",[1]*100)])
        with self.assertRaises(ValueError):
            fuse([])

    def test_group_intervals_query_weighted_paired_not_group_mean(self):
        x=np.asarray([[1.,0.],[1.,0.],[1.,0.],[-1.,0.]])
        result=intervals(x,["a","a","a","b"],["effect","paired_zero"],repetitions=300)
        self.assertEqual(result["effect"]["mean"],.5)
        self.assertEqual(result["paired_zero"]["ci95_group_bootstrap"],[0,0])


if __name__ == "__main__":
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contract))
    write("TEST_RESULTS.json",{"passed":result.wasSuccessful(),"tests":result.testsRun,
                               "failures":len(result.failures),"errors":len(result.errors),"created_utc":now()})
    raise SystemExit(0 if result.wasSuccessful() else 1)
