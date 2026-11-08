from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from transformers import AutoTokenizer

from experiments.rene_strict_confirmation_20261019.common import (
    OUT, SEEDS, load, read_json, write_json, observable, reject_gold, assert_disjoint,
    require_runtime, stable_top, rr, sha,
)
from experiments.rene_strict_confirmation_20261019 import features, fresh_data, model
from experiments.rene_strict_confirmation_20261019.development import role_plan
from experiments.rene_strict_confirmation_20261019.evaluate import cp_upper, holm


class Integrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.items = load(OUT / 'development/items.pkl')
        cls.scores = load(OUT / 'development/scores.pkl')
        cls.models = {a: joblib.load(OUT / f'models/rewrite_{a}.joblib') for a in ('no_seed_uq', 'seed_uq')}

    def test_models_are_retrained_with_absent_seed_columns(self):
        no, yes = self.models['no_seed_uq'], self.models['seed_uq']
        self.assertEqual(no['gain']['input_dimension'], 72)
        self.assertEqual(yes['gain']['input_dimension'], 84)
        self.assertEqual(no['update']['input_dimension'], 214)
        self.assertEqual(yes['update']['input_dimension'], 226)
        self.assertFalse(any(34 <= k < 46 for k in no['gain']['columns']))
        for m in no['gain']['members']:
            self.assertEqual(m['mean'].num_feature(), 72)
        for m in no['update']['members']:
            self.assertEqual(m['mean'].num_feature(), 214)
        self.assertEqual(no['role_repositories'], yes['role_repositories'])
        self.assertGreater(sum(m['mean'].feature_importance()[34:46].sum() for m in yes['gain']['members']), 0)
        weak = joblib.load(OUT / 'diagnostic_models/rewrite_base34.joblib')
        self.assertTrue(weak['refitted'])
        self.assertEqual(weak['members'][0]['mean'].num_feature(), 34)

    def test_seed_poisoning_cannot_change_no_seed_full_flow(self):
        views = [observable(e) for e in self.items[:24]]
        poisoned = copy.deepcopy(views)
        for e in poisoned:
            e['seed_x'][:] = np.arange(12)[None] * 10000. + 100000.
        pipeline = self.models['no_seed_uq']
        a, _, _ = model.predict_pipeline(pipeline, views, self.scores[:24])
        b, _, _ = model.predict_pipeline(pipeline, poisoned, self.scores[:24])
        for stage in a:
            for x, y in zip(a[stage], b[stage]):
                self.assertEqual(x['ranking'], y['ranking'])
                self.assertEqual(x['chosen'], y['chosen'])
                self.assertEqual(x['mu'], y['mu'])
                self.assertEqual(x.get('post_harm'), y.get('post_harm'))

    def test_query_identity_is_not_a_hidden_target(self):
        views = [observable(e) for e in self.items[:16]]
        changed = copy.deepcopy(views)
        for i, e in enumerate(changed):
            e['query_index'] = 9000000 + i
            e['strong_evidence']['query_index'] = e['query_index']
        a, _, _ = model.predict_pipeline(self.models['seed_uq'], views, self.scores[:16])
        b, _, _ = model.predict_pipeline(self.models['seed_uq'], changed, self.scores[:16])
        for stage in a:
            for x, y in zip(a[stage], b[stage]):
                self.assertEqual(x['ranking'], y['ranking'])
                self.assertEqual(x['mu'], y['mu'])

    def test_outcome_fields_are_rejected(self):
        for key in ('y', 'target', 'target_doc_id', 'original_rr', 'delta_rr', 'source_idx'):
            e = observable(self.items[0]); e[key] = 0
            with self.assertRaises(ValueError):
                model.predict_gain(self.models['seed_uq']['gain'], [e])

    def test_empty_candidate_query_is_retained(self):
        e = copy.deepcopy(observable(self.items[0]))
        e['x'] = np.empty((0, 34)); e['seed_x'] = np.empty((0, 12))
        e['candidate_top100'] = []; e['candidates'] = []; e['seed_appearances'] = []
        for family in ('rewrite', 'fusion'):
            e['strong_evidence']['x'][family] = np.empty((0, 28))
        output, _, _ = model.predict_pipeline(self.models['seed_uq'], [e], [self.scores[0]])
        for ts in output.values():
            self.assertFalse(ts[0]['accepted'])
            self.assertEqual(ts[0]['ranking'], tuple(e['original_top100']))

    def test_all_roles_and_outer_queries_are_repository_disjoint(self):
        byid = {e['query_index']: e['repo'] for e in self.items}
        folds, final = role_plan(self.items)
        all_held = []
        for roles in folds:
            assert_disjoint({k: [byid[q] for q in ids] for k, ids in roles.items()})
            all_held.extend(roles['held'])
        self.assertEqual(len(set(all_held)), 512)
        assert_disjoint({k: [byid[q] for q in ids] for k, ids in final.items()})
        with self.assertRaises(ValueError):
            assert_disjoint({'gain': ['a'], 'risk': ['a']})

    def test_exclusion_load_failure_does_not_silently_drop_repositories(self):
        with patch.object(fresh_data, 'read_json', side_effect=FileNotFoundError('missing protocol')):
            with self.assertRaises(FileNotFoundError):
                fresh_data.exclusion_inventory()

    def test_actual_four_seed_schema_and_quality_fraction(self):
        manifest = read_json(OUT / 'development/MANIFEST.json')
        self.assertEqual(manifest['eight_seed_feature_parity_max_error'], 0.)
        self.assertEqual(manifest['retained_label_parity_max_error'], 0.)
        self.assertEqual(manifest['maximum_candidates'], 12)
        self.assertTrue(any(np.any(e['seed_x'][:, -1] < .99) for e in self.items))
        self.assertTrue(all(set(a) <= set(SEEDS) for e in self.items for a in e['seed_appearances']))

    def test_empty_seed_groups_are_finite_and_do_not_fake_support(self):
        draws = [dict(seed=s, candidates=[{'text': 'a'}] if s == 101 else [],
                      quality_pass_count=1 if s == 101 else 0, raw_count=2 if s == 101 else 1) for s in SEEDS]
        x = features.sampling_features(dict(draws=draws), np.array([[1., 0.]]),
                                       [np.arange(100)], [[101]])
        self.assertEqual(x[0, 0], .25)
        self.assertEqual(x[0, -1], .125)
        self.assertTrue(np.isfinite(x).all())

    def test_ties_use_actual_ranking_positions(self):
        top = stable_top(np.array([.5, .8, .8]))
        self.assertEqual(list(top), [1, 2, 0])
        self.assertEqual(rr(top, 2), .5)
        self.assertEqual(rr(top, 99), 0.)

    def test_risk_rejection_keeps_incurred_cost_and_has_no_reselection(self):
        pipeline = self.models['seed_uq']
        views = [observable(e) for e in self.items]
        output, _, _ = model.predict_pipeline(pipeline, views, self.scores)
        indices = [i for i, t in enumerate(output['verified']) if t['accepted']]
        self.assertTrue(indices)
        forced = dict(pipeline['risk'], fallback=True, class_prior=np.array([.99, .005, .005]))
        before = [output['verified'][i] for i in indices]
        after = model.apply_risk(forced, [views[i] for i in indices], before)
        for i, a, b in zip(indices, before, after):
            self.assertFalse(b['accepted'])
            self.assertEqual(b['ranking'], tuple(views[i]['original_top100']))
            for key in ('actual_pairs', 'billed_slots', 'executed_ce_tokens', 'verified'):
                self.assertEqual(a[key], b[key])

    def test_actual_ce_token_cost_matches_tokenizer(self):
        from experiments.rene_strict_confirmation_20261019.common import DATA
        tok = AutoTokenizer.from_pretrained(ROOT / 'models/codebert-base', local_files_only=True)
        codes = load(DATA / 'action_code_corpus.pkl')
        e = self.items[0]
        q = tok(e['query'], add_special_tokens=False, truncation=True, max_length=64)['input_ids']
        for d in list(e['original_top100'])[:3]:
            c = tok(codes[int(d)]['code_text'], add_special_tokens=False, truncation=True, max_length=160)['input_ids']
            self.assertEqual(len(tok.build_inputs_with_special_tokens(q, c)),
                             4 + e['ce_query_length'] + e['ce_code_lengths'][int(d)])

    def test_immutable_protocol_and_tamper_rejection(self):
        from experiments.rene_strict_confirmation_20261019.common import check_hashes
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'protocol.json'
            write_json(p, {'a': 1}, immutable=True)
            original = sha(p)
            write_json(p, {'a': 1}, immutable=True)
            with self.assertRaises(ValueError):
                write_json(p, {'a': 2}, immutable=True)
            p.write_text('tampered', encoding='utf8')
            with self.assertRaises(ValueError):
                check_hashes({str(p): original})

    def test_exact_risk_and_multiplicity_edge_cases(self):
        self.assertEqual(cp_upper(0, 0), 1.)
        self.assertAlmostEqual(cp_upper(0, 29), 1 - .05 ** (1 / 29))
        self.assertEqual(holm({'first': .03, 'second': .04}), {'first': .06, 'second': .06})


if __name__ == '__main__':
    require_runtime()
    with threadpool_limits(limits=2):
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(Integrity)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    write_json(OUT / 'development/TESTS.json', dict(passed=result.wasSuccessful(), tests=result.testsRun,
               failures=[str(x[0]) for x in result.failures], errors=[str(x[0]) for x in result.errors],
               source_sha256=sha(__file__)))
    sys.exit(0 if result.wasSuccessful() else 1)
