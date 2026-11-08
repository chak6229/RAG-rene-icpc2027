from __future__ import annotations

from collections import defaultdict

from experiments.rene_budget_aggregation_20261020.metrics import ordered_positions

from .common import COMPONENTS, MODEL, OUT, PACKAGE, PROTECTED, ROOT, SOURCES, now, read, record, require, rows, sha, write, writel
from .statistics import precision_scenario

BRIDGE = ROOT / 'results/rene_ssqr_bridge_20261019'
BUDGET = ROOT / 'results/rene_budget_aggregation_20261020'
FACTOR = ROOT / 'results/rene_factor_confirmation_20261020'
ADAPTER = ROOT / 'results/rene_ssqr_alignment_20261019/cosqa_adapter'


def source_configs():
    return {
        'cosqa': {
            'queries': BRIDGE / 'data/cosqa/queries.jsonl',
            'corpus': ADAPTER / 'public/corpus.jsonl',
            'candidates': BRIDGE / 'candidates/cosqa/baseline.jsonl',
            'old_actions': BUDGET / 'ACTIONS.jsonl',
            'qrels': BRIDGE / 'data/private/cosqa_qrels.jsonl',
            'reference_rows': BUDGET / 'per_query.jsonl',
            'reference_results': BUDGET / 'RESULTS.json',
            'queries_expected': 1490, 'groups_expected': 512, 'documents_expected': 6267,
            'grouping': 'original CoSQA connected group',
        },
        'staqc': {
            'queries': FACTOR / 'public/queries.jsonl',
            'corpus': FACTOR / 'public/corpus.jsonl',
            'candidates': FACTOR / 'candidates.jsonl',
            'old_actions': FACTOR / 'actions.jsonl',
            'groups': FACTOR / 'public/groups.jsonl',
            'qrels': FACTOR / 'private/qrels.jsonl',
            'reference_rows': FACTOR / 'per_query.jsonl',
            'reference_results': FACTOR / 'RESULTS.json',
            'queries_expected': 13312, 'groups_expected': 13312, 'documents_expected': 16384,
            'grouping': 'original StaQC connected group, one query per group',
        },
    }


def reference_rows(domain, cfg):
    keep = {q['query_id'] for q in rows(cfg['queries'])}
    output = {ret: {} for ret in ('CodeBERT', 'BM25')}
    for r in rows(cfg['reference_rows']):
        if r.get('domain', domain) != domain or r['query_id'] not in keep:
            continue
        ret = r['retriever']
        if ret in output:
            require(r['query_id'] not in output[ret], 'Duplicate reference query')
            output[ret][r['query_id']] = r
    require(all(set(records) == keep for records in output.values()), 'Reference coverage mismatch')
    return output


def freeze():
    if (OUT / 'DESIGN.json').exists():
        from .common import verify_design
        return verify_design()
    require((OUT / 'MODEL_PROVENANCE.json').exists(), 'Pinned model download not finished')
    require(read(OUT / 'PREFLIGHT.json')['passed'], 'Model implementation preflight not passed')
    require(read(OUT / 'TEST_RESULTS.json')['passed'], 'Contract tests not passed')
    # Historical source experiments remain immutable. Verify their original seals.
    for path, field in [(BUDGET / 'PROTOCOL.json', 'bound_files'), (FACTOR / 'DESIGN.json', 'bound_files')]:
        old = read(path)
        for item in old[field] + old['protected_files']:
            require(sha(ROOT / item['path']) == item['sha256'], 'Accepted source changed: ' + item['path'])
    configs = source_configs()
    samples, precision = {}, {}
    before = read(OUT / 'BEFORE_INFERENCE_PROTECTED.json')
    for item in before['files']:
        require(sha(ROOT / item['path']) == item['sha256'], 'Original protected state changed before freeze')
    paths = {OUT / 'MODEL_PROVENANCE.json', OUT / 'PREFLIGHT.json', OUT / 'TEST_RESULTS.json',
             OUT / 'BEFORE_INFERENCE_PROTECTED.json', PACKAGE / 'DESIGN.md'}
    paths.update(PACKAGE.glob('*.py'))
    paths.add(ROOT / 'experiments/rene_budget_aggregation_20261020/metrics.py')
    for item in read(OUT / 'MODEL_PROVENANCE.json')['files']:
        paths.add(ROOT / item['path'])
    for domain, cfg in configs.items():
        query_list = list(rows(cfg['queries']))
        corpus = list(rows(cfg['corpus']))
        pools = defaultdict(list)
        for r in rows(cfg['candidates']):
            pools[r['query_id']].append(r)
        ids = [q['query_id'] for q in query_list]
        require(len(ids) == len(set(ids)) == cfg['queries_expected'], 'Query coverage mismatch')
        require(set(ids) == set(pools), 'Candidate/query mismatch')
        require(len(corpus) == cfg['documents_expected'], 'Corpus size mismatch')
        require(len({r['doc_id'] for r in corpus}) == len(corpus), 'Duplicate corpus IDs')
        old_actions = {r['query_id']: r for r in rows(cfg['old_actions']) if r.get('domain', domain) == domain}
        groups = {r['query_id']: r['group_id'] for r in rows(cfg['groups'])} if 'groups' in cfg else {qid: old_actions[qid]['group'] for qid in ids}
        actions = []
        for q in query_list:
            qid = q['query_id']
            pool = pools[qid]
            require(sorted(r['position'] for r in pool) == list(range(1, len(q['query'].split()) + 1)), 'Candidate position coverage')
            orders = {p: ordered_positions(pool, p) for p in ('ENTR', 'PROB')}
            require(orders == old_actions[qid]['orders'], 'Selector changed from accepted experiment')
            actions.append({'query_id': qid, 'group': groups[qid], 'positions': len(pool), 'orders': orders})
        require(len(set(groups.values())) == cfg['groups_expected'], 'Group count mismatch')
        action_path = OUT / f'public/{domain}_actions.jsonl'
        if not action_path.exists():
            writel(action_path, actions)
        else:
            require(list(rows(action_path)) == actions, 'Prepared actions differ')
        paths.add(action_path)
        paths.update(v for v in cfg.values() if hasattr(v, 'relative_to'))
        samples[domain] = {k: v.relative_to(ROOT).as_posix() if hasattr(v, 'relative_to') else v for k, v in cfg.items()}
        samples[domain]['actions'] = action_path.relative_to(ROOT).as_posix()
        samples[domain]['candidate_positions'] = sum(map(len, pools.values()))
        samples[domain]['unique_query_inputs'] = len(set([q['query'] for q in query_list] + [r['rewrite'] for records in pools.values() for r in records]))
        ref = reference_rows(domain, cfg)
        require(all(r['group'] == groups[qid] for records in ref.values() for qid, r in records.items()), 'Reference grouping differs from frozen actions')
        precision[domain] = {}
        for ret in ref:
            precision[domain][ret] = {c: precision_scenario([ref[ret][qid]['contrasts'][c] for qid in ids], [groups[qid] for qid in ids]) for c in COMPONENTS}
    if not (OUT / 'PRECISION_SCENARIOS.json').exists():
        write(OUT / 'PRECISION_SCENARIOS.json', {'created_utc': now(), 'new_UniXcoder_effects_computed': False,
              'fixed_samples_no_effect_driven_expansion': True, 'scenarios': precision})
    paths.add(OUT / 'PRECISION_SCENARIOS.json')
    design = {'created_utc': now(), 'status': 'FROZEN_BEFORE_NEW_STUDY_PREDICTIONS',
        'role': 'prospectively_specified_third_retriever_extension_on_consumed_datasets',
        'new_confirmation': False, 'external_preregistration': False, 'prior_two_retriever_effects_known': True,
        'new_UniXcoder_effects_computed_before_freeze': False,
        'retriever': 'UniXcoder-base author zero-shot checkpoint',
        'model': MODEL.relative_to(ROOT).as_posix(), 'samples': samples,
        'inference': {'mode': '<encoder-only>', 'code_max_length': 256, 'query_max_length': 128,
            'pooling': 'mean of all nonpadding last hidden token states including special tokens',
            'normalization': 'L2 unit vector', 'similarity': 'FP32 dot product of normalized vectors',
            'batch_size': 16, 'precision': 'FP32; TF32 disabled', 'dropout': False,
            'token_prefix': ['<s>', '<encoder-only>', '</s>'], 'token_suffix': ['</s>'],
            'ties': 'score descending, original corpus doc_id ascending', 'depth': 100,
            'recipe_source': 'pinned official downstream-tasks/code-search; code256/nl128',
            'common_input_text_and_corpus': True, 'isolated_architecture_effect': False},
        'selectors': ['ENTR', 'PROB', 'exact uniform without replacement over original positions'],
        'candidate_counts': [1, 2, 3], 'aggregations': ['first structural anchor', 'mean', 'max label-assisted'],
        'primary_components': list(COMPONENTS),
        'primary_family': {'size': 12, 'new_model_B_G_times_2_scenarios': 4,
            'paired_new_minus_each_old_B_G_times_2_scenarios': 8,
            'correction': 'two-sided Bonferroni12; both cluster-t and group bootstrap intervals',
            'nonzero_groups_minimum': 20, 'bootstrap_repetitions': 10000, 'bootstrap_seed': 202610203},
        'secondary': 'pointwise95 descriptive: PROB, ENTR-PROB, all S curves and baseline/candidate RR',
        'sample_planning': 'complete previously fixed samples; no new-model power guarantee; old variance scenarios archived',
        'no_new_training': True, 'no_new_T5_generation': True, 'no_new_H_evaluation': True,
        'not_cross_rewriter_family': True, 'not_full_policy_transfer': True, 'risk_certified': False,
        'bound_files': [record(p) for p in sorted(paths)],
        'protected_files': [record(ROOT / p) for p in PROTECTED]}
    write(OUT / 'DESIGN.json', design)
    write(OUT / 'ACTIONS_SEALED.json', {'created_utc': now(), 'design_sha256': sha(OUT / 'DESIGN.json'),
          'files': [record(ROOT / s['actions']) for s in samples.values()],
          'old_qrels_not_parsed_for_actions': True, 'new_effects_computed': False})
    print({'frozen': True, 'samples': {d: [s['queries_expected'], s['groups_expected'], s['documents_expected']] for d, s in samples.items()}, 'primary_family': 12}, flush=True)
    return design


if __name__ == '__main__':
    freeze()
