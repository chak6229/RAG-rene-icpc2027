import tempfile
import unittest
from pathlib import Path

import numpy as np

from .common import deny_private_reads, position_seed, require
from .evaluation import selections, evaluate_query, cluster_interval, domain_difference, rr
from .rank import stable_order
from .freeze import UnionFind, choose_one_per_group


class BridgeContractTests(unittest.TestCase):
    def pool(self):
        return [{"query_id":"q","position":p,"full_vocab_entropy":4-p,
                 "span_full_vocab_entropy":None if p==0 else p/10,
                 "label_mean_log_probability":-p} for p in range(5)]

    def test_entropy_and_probability_have_opposite_order(self):
        selected=selections(self.pool(),"q",range(20))
        self.assertEqual(selected["ENTR"],[4,3,2])
        self.assertEqual(selected["PROB"],[0,1,2])

    def test_lexical_empty_placed_last(self):
        self.assertEqual(selections(self.pool(),"q",range(20),mask="lexical")["ENTR"],[1,2,3])

    def test_rand_budget_and_repeat_determinism(self):
        a=selections(self.pool(),"q",range(20))
        b=selections(list(reversed(self.pool())),"q",range(20))
        self.assertEqual(a,b)
        self.assertTrue(all(len(set(x))==3 for x in a["RAND"]))

    def test_duplicate_or_mixed_pool_rejected(self):
        with self.assertRaises(ValueError):
            selections(self.pool()+[self.pool()[0]],"q",range(20))
        pool=self.pool()
        pool[0]["query_id"]="other"
        with self.assertRaises(ValueError):
            selections(pool,"q",range(20))

    def test_fewer_than_k_no_padding(self):
        result=selections(self.pool()[:1],"q",range(20))
        self.assertEqual(result["ENTR"],[0])
        self.assertTrue(all(x==[0] for x in result["RAND"]))

    def test_best3_does_not_implicitly_include_original(self):
        pool=[{"position":0,"rewrite":"a"}]
        chosen={"ENTR":[0],"PROB":[0],"RAND":[[0]]*20}
        lookup={"original":{"ranked_doc_ids":list(range(100))},"a":{"ranked_doc_ids":list(range(1,101))}}
        result=evaluate_query(pool,chosen,lookup,{0},"original",lambda x:x)
        self.assertEqual(result["NoRewrite"],1.)
        self.assertEqual(result["ENTR_best3"],0.)

    def test_incomplete_rr_and_duplicate_rank_rejected(self):
        with self.assertRaises(ValueError):rr([1,2],{1})
        with self.assertRaises(ValueError):rr([1]*100,{1})
        self.assertEqual(rr(list(range(100)),{99}),.01)

    def test_doc_id_breaks_equal_scores(self):
        ids=np.array([8,3,1])
        self.assertEqual(ids[stable_order(np.array([1.,2.,2.]),ids)].tolist(),[1,3,8])

    def test_sampling_seed_is_position_local(self):
        original={p:position_seed("q",p) for p in (1,2,3)}
        with_zero={p:position_seed("q",p) for p in (0,1,2,3)}
        self.assertEqual(original,{p:with_zero[p] for p in original})

    def test_shared_domain_cluster_covariance(self):
        a=domain_difference([1.,2.,3.],["a","b","c"],[1.,2.,3.],["a","b","c"],repetitions=200)
        self.assertEqual(a["mean"],0.)
        self.assertEqual(a["standard_error_cluster"],0.)
        self.assertEqual(a["ci95_group_bootstrap"],[0.,0.])

    def test_query_weighted_cluster_mean(self):
        result=cluster_interval([1.,1.,0.],["a","a","b"],repetitions=200)
        self.assertAlmostEqual(result["mean"],2/3)
        self.assertEqual(result["groups"],2)

    def test_joint_repository_union_and_group_sampling(self):
        uf=UnionFind()
        uf.union("s","repo")
        uf.union("c","repo")
        self.assertEqual(uf.find("s"),uf.find("c"))
        groups={"q1":"a","q2":"a","q3":"b"}
        chosen=choose_one_per_group(list(groups),groups,namespace="test")
        self.assertEqual(len(chosen),2)
        self.assertEqual(len({groups[q] for q in chosen}),2)

    def test_private_read_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"private"/"labels.json"
            path.parent.mkdir()
            path.write_text("{}")
            with deny_private_reads():
                with self.assertRaises(PermissionError):path.read_text()


if __name__=="__main__":
    unittest.main()
