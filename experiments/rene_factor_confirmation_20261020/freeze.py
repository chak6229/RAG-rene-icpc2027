from __future__ import annotations

import collections
import ast
import math
import subprocess
import sys

from .common import ROOT, PACKAGE, OUT, SOURCES, PROTECTED, digest, now, read, record, rows, sha, write, writel
from .precision import CONTRASTS, make_plan

T5 = ROOT / 'models/t5-cqc-codenn-1m/best'
CB = ROOT / 'models/codebert-ssqr-pooler-2epoch-20261019'


def prepare():
    if (OUT / 'DESIGN.json').exists() or (OUT / 'OUTCOME_OPENING.json').exists():
        raise FileExistsError('Confirmation already frozen/opened')
    eligibility = read(OUT / 'STAQC_ELIGIBILITY.json')
    precision = read(OUT / 'PRECISION_PLAN.json')
    if not precision['plans']['StaQC_one_question_per_component']['eligible']:
        raise ValueError('New source capacity did not pass prespecified precision gate')
    pairs = list(rows(OUT / 'STAQC_ELIGIBLE_PAIRS.jsonl'))
    grouped = collections.defaultdict(list)
    for r in pairs:
        grouped[r['group_id']].append(r)
    one_per_group = {}
    for g, values in grouped.items():
        one_per_group[g] = min(values, key=lambda r: digest('factor-within-group/' + r['unit_id']))
    # Equal component sampling avoids a few duplicate-title groups dominating capacity.
    unit_plan = make_plan(precision['development'], [1] * len(grouped))
    need = max(r['required_groups_005'] for r in unit_plan['contrasts'].values())
    count = math.ceil((need / .9) / 256) * 256
    if count > len(grouped):
        raise ValueError('No 10% capacity reserve under balanced sampling')
    selected_groups = sorted(grouped, key=lambda g: digest('factor-confirm-query/' + g))[:count]
    selected = [one_per_group[g] for g in selected_groups]
    actual_plan = make_plan(precision['development'], [1] * count)
    if not actual_plan['eligible']:
        raise ValueError('Exact sample fails prospective precision')
    # Corpus is the next power of two >=1.10 times the planned query count;
    # metadata-only sizing preserves room for outcome-blind extra components.
    corpus_size = 2**math.ceil(math.log2(count*1.10))
    if count >= corpus_size or len(grouped) < corpus_size:
        raise ValueError('Cannot form fixed corpus with disjoint components')
    selected_set = set(selected_groups)
    extra = sorted((g for g in grouped if g not in selected_set), key=lambda g: digest('factor-corpus/' + g))[:corpus_size-count]
    corpus_pairs = selected + [one_per_group[g] for g in extra]
    corpus_pairs.sort(key=lambda r: digest('factor-doc-order/' + r['unit_id']))
    doc_ids = {r['unit_id']: i for i, r in enumerate(corpus_pairs)}
    queries = [{'query_id': r['unit_id'], 'query': r['query']} for r in selected]
    corpus = [{'doc_id': doc_ids[r['unit_id']], 'code_text': r['code_text']} for r in corpus_pairs]
    metadata = [{'query_id': r['unit_id'], 'group_id': r['group_id'], 'question_id': r['question_id'],
                 'source_kind': r['kind']} for r in selected]
    qrels = [{'query_id': r['unit_id'], 'relevant_doc_ids': [doc_ids[r['unit_id']]]} for r in selected]
    for name, data in (('public/queries.jsonl', queries), ('public/corpus.jsonl', corpus),
                       ('public/groups.jsonl', metadata), ('private/qrels.jsonl', qrels)):
        writel(OUT / name, data)
    write(OUT / 'SAMPLE_PLAN.json', {'created_utc': now(), 'queries': count, 'groups': count,
          'corpus_documents': corpus_size, 'available_groups': len(grouped),
          'minimum_groups_at_005': need, 'reserve_rule': 'ceil(required/.9) rounded up to256, one question per connected source component',
          'exact_sample_precision': actual_plan, 'query_group_ids': selected_groups,
          'corpus_source_unit_ids': [r['unit_id'] for r in corpus_pairs],
          'sampling': 'SHA256 only, no model outputs, no relevance efficacy or retrieval ranks',
          'estimand_population': 'query-weighted sampled one accepted-single-code question per eligible connected SO component; fixed snippet corpus',
          'known_positive': 'single accepted snippet per question; unjudged other snippets are not human-confirmed nonrelevant',
          'lexical_fallback_code_documents': sum(r['lexical_fallback'] for r in corpus_pairs)})
    return actual_plan


def main():
    for path in PACKAGE.glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
    if not (OUT / 'SAMPLE_PLAN.json').exists():
        prepare()
    if (OUT / 'DESIGN.json').exists():
        raise FileExistsError('Already frozen')
    command = [sys.executable, '-X', 'utf8', '-m', 'unittest',
               'experiments.rene_factor_confirmation_20261020.test_preflight',
               'experiments.rene_factor_confirmation_20261020.test_confirmation', '-v']
    tests = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
    write(OUT / 'TEST_RESULTS.json', {'command': command, 'returncode': tests.returncode,
          'passed': tests.returncode == 0, 'output': tests.stdout + tests.stderr})
    if tests.returncode:
        raise ValueError(tests.stderr)
    sample = read(OUT / 'SAMPLE_PLAN.json')
    paths = set(PACKAGE.glob('*.py')) | {PACKAGE / 'PREFLIGHT_DESIGN.md'}
    paths.update(OUT / n for n in ('LOCAL_ELIGIBILITY.json', 'LOCAL_PRECISION_PLAN.json', 'STAQC_ELIGIBILITY.json',
                 'PRECISION_PLAN.json', 'SAMPLE_PLAN.json', 'TEST_RESULTS.json', 'public/queries.jsonl',
                 'public/corpus.jsonl', 'public/groups.jsonl', 'private/qrels.jsonl'))
    paths.update(p for p in SOURCES.iterdir() if p.is_file())
    if (OUT/'PRE_INFERENCE_REPAIR.json').exists():
        paths.add(OUT/'PRE_INFERENCE_REPAIR.json')
        paths.update(p for p in (OUT/'pre_inference_history').rglob('*') if p.is_file())
    if (OUT/'PRECISION_AMENDMENT_BEFORE_EFFECTS.json').exists():
        paths.add(OUT/'PRECISION_AMENDMENT_BEFORE_EFFECTS.json')
    if not read(OUT/'BATCHING_DEVELOPMENT_CHECK.json')['passed']:
        raise ValueError('Label-blind batch32 fidelity check missing')
    paths.add(OUT/'BATCHING_DEVELOPMENT_CHECK.json')
    paths.update(ROOT / n for n in ('experiments/rene_ssqr_alignment_20261019/t5.py',
                  'experiments/rene_ssqr_alignment_20261019/retriever.py',
                  'experiments/rene_ssqr_alignment_20261019/pipeline.py',
                  'experiments/rene_cosqa_pilot_20261019/ranking.py',
                  'experiments/rene_budget_aggregation_20261020/metrics.py'))
    paths.update(p for p in T5.iterdir() if p.is_file() and p.suffix in ('.json', '.safetensors'))
    paths.update(p for p in CB.iterdir() if p.is_file() and p.suffix in ('.json', '.safetensors', '.txt'))
    paths.add(ROOT / 'models/t5-base/spiece.model')
    design = {'created_utc': now(), 'status': 'FROZEN_BEFORE_NEW_METHOD_PREDICTIONS',
        'evidence_role': 'prospective_one_shot_SSQR_factor_transfer_to_StaQC_single_code_source',
        'external_preregistration': False, 'new_setting': 'author-pinned StaQC Python accepted single-code snippets / question titles',
        'not_CoSQA_confirmation': True, 'new_method_effects_computed_before_freeze': False,
        'raw_source_labels_seen_during_eligibility': True,
        'queries': sample['queries'], 'groups': sample['groups'], 'corpus_documents': sample['corpus_documents'],
        'primary_retriever': 'CodeBERT', 'secondary_retriever': 'BM25',
        'primary_contrasts': list(CONTRASTS), 'primary_family_size': 2, 'family_alpha_two_sided': .05,
        'primary_interpretation': 'positive attenuation established only when simultaneous97.5%cluster-t and group-bootstrap lower limits >0 and >=20groups with nonzero query contrast; unresolved otherwise; negative intervals report reverse direction',
        'family_intervals': 'two co-primary two-sided97.5% intervals (Bonferroni); bootstrap10000 with fixed seed202610202; cluster-t df=G-1',
        'success_scope': 'both positive establishes both pre-specified attenuation effects here; one positive establishes that component only; no CoSQA, unique causal attribution, or new UQ claim',
        'precision': sample['exact_sample_precision'], 'effect_resolution': .005,
        'metric': 'RR@100 of one known accepted code snippet; no NoRewrite in best3',
        'candidate_set': 'one generation per position1..n; keep empty spans and duplicate text positions; min(k,n), no identity padding',
        'generation': {'checkpoint': str(T5.relative_to(ROOT)), 'seed': 101, 'do_sample': False,
                       'max_new_tokens': 19, 'num_beams': 1, 'batch_size_across_positions': 32,
                       'query_block_size':32,'dropout': False, 'precision': 'float32',
                       'runtime_change':'cross-query batching32 independently checked against prior batch8 on consumed source, preserving span/selection within recorded FP32 tolerance'},
        'scoring': 'existing frozen t5.py full vocabulary FP32 teacher forcing, full nonpadding sentinel/span/EOS label; lower entropy ENTR, higher mean logp PROB, position tie',
        'contract': 'k1/2/3 x first/mean/max; exact uniform without replacement position subsets, duplicate position mass retained',
        'B': 'ENTR_k1_mean - RAND_k1_mean - ENTR_k3_mean + RAND_k3_mean',
        'G': 'ENTR_k3_mean - RAND_k3_mean - ENTR_k3_max + RAND_k3_max',
        'secondaries': 'PROB, ENTR-PROB, all absolute utilities/curves and BM25, paired retriever differences; descriptive95%group intervals, no extra confirmatory rejection claims',
        'NoRewrite': 'evaluated separately as a reference; no new policy fitting or risk gate',
        'retrieval': {'max_length':256, 'representation':'roberta_pooler_output', 'normalization':'none',
                      'similarity':'FP32 dot_product', 'TF32':False, 'depth':100,
                      'tie':'score descending, original document ID ascending', 'bm25':{'k1':1.2,'b':.75}},
        'freeze_order':'eligible sources+development-only precision -> sample/protocol/source+model hashes -> candidate/actions seal -> rankings seal -> independent preopening audit -> one outcome opening -> analysis -> independent numerical/model audits',
        'no_optional_stopping': True, 'no_refit_no_seed_search_no_model_promotion': True,
        'risk_certified':False, 'foundation_pretraining_overlap_unknown':True,
        'bound_files':[record(p) for p in sorted(paths)], 'protected_files':[record(ROOT/p) for p in PROTECTED]}
    write(OUT / 'DESIGN.json', design)
    print({'frozen':True,'queries':sample['queries'],'groups':sample['groups'],'corpus':sample['corpus_documents']},flush=True)


if __name__ == '__main__':
    main()
