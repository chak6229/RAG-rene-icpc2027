from __future__ import annotations

import ast
import platform
import subprocess
import sys

from .common import FAMILY, MODEL_PATHS, OLD, OUT, PACKAGE, PRIMARY, REPETITIONS, ROOT, SEED, WEIGHTS, now, read, record, write


def main():
    if (OUT/'DESIGN.json').exists():raise FileExistsError('Design already frozen')
    for path in PACKAGE.glob('*.py'):ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
    if not read(OUT/'BATCHING_DEVELOPMENT_CHECK.json')['passed']:raise ValueError('Both-weight batch check required')
    command=[sys.executable,'-X','utf8','-m','unittest',
       'experiments.rene_weight_confirmation_20261021.test_contract',
       'experiments.rene_t5_weight_factors_20261020.test_contract','-v']
    test=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
    write(OUT/'TEST_RESULTS.json',{'created_utc':now(),'command':command,'passed':test.returncode==0,
          'returncode':test.returncode,'output':test.stdout+test.stderr})
    if test.returncode:raise ValueError(test.stderr)
    sample=read(OUT/'SAMPLE_PLAN.json');plan=read(OUT/'PRECISION_PLAN.json')
    bound=set(PACKAGE.glob('*.py'))|{PACKAGE/'PREFLIGHT_DESIGN.md'}
    bound.update(OUT/n for n in ['ELIGIBILITY.json','EXPOSURE_MANIFEST.json','ELIGIBLE_PAIRS.jsonl','PRECISION_PLAN.json',
        'SAMPLE_PLAN.json','BATCHING_DEVELOPMENT_CHECK.json','TEST_RESULTS.json','public/queries.jsonl','public/corpus.jsonl',
        'public/groups.jsonl','private/qrels.jsonl'])
    for folder in (*MODEL_PATHS.values(),'models/codebert-ssqr-pooler-2epoch-20261019'):
        bound.update(p for p in (ROOT/folder).iterdir() if p.is_file() and p.suffix!='.pt')
    bound.update(ROOT/p for p in [
        'experiments/rene_factor_confirmation_20261020/common.py',
        'experiments/rene_factor_confirmation_20261020/inference.py',
        'experiments/rene_ssqr_alignment_20261019/t5.py',
        'experiments/rene_ssqr_alignment_20261019/retriever.py',
        'experiments/rene_ssqr_alignment_20261019/pipeline.py',
        'experiments/rene_ssqr_alignment_20261019/common.py',
        'experiments/rene_t5_weight_factors_20261020/common.py',
        'experiments/rene_t5_weight_factors_20261020/metrics.py',
        'results/rene_factor_confirmation_20261020/STAQC_ELIGIBILITY.json',
        'results/rene_factor_confirmation_20261020/LOCAL_ELIGIBILITY.json',
        'results/rene_factor_confirmation_20261020/STAQC_ELIGIBLE_PAIRS.jsonl',
        'results/rene_factor_confirmation_20261020/STAQC_PAIR_METADATA.jsonl',
        'results/rene_factor_confirmation_20261020/reference_fingerprints.pkl'])
    bound.update(ROOT/r['path'] for r in plan['development_files'])
    protected={r['path'] for r in read(ROOT/'results/rene_t5_weight_factors_20261020/DESIGN.json')['protected_files']}
    protected.update(r['path'] for r in read(OLD/'DESIGN.json')['protected_files'])
    for folder in ['rene_factor_confirmation_20261020','rene_t5_weight_factors_20261020','rene_third_retriever_20261020',
                   'rene_contribution_strength_20261021','rene_review_completion_20261021']:
        for name in ['FINAL_ACCEPTANCE.json','RESULTS.json','NUMERICAL_AUDIT.json']:
            p=ROOT/'results'/folder/name
            if p.exists():protected.add(p.relative_to(ROOT).as_posix())
    protected.update(p.relative_to(ROOT).as_posix() for p in (ROOT/'results/rene_manuscript_integration_20261021').glob('T*.csv'))
    # Capture source inputs and the whole enumerated historical exposure inventory.
    exposure=read(OUT/'EXPOSURE_MANIFEST.json')
    bound.update(ROOT/r['path'] for r in exposure['files'])
    bound.update(p for p in (OLD/'sources').iterdir() if p.is_file())
    design={'created_utc':now(),'status':'FROZEN_BEFORE_NEW_GENERATION_AND_EFFECTS',
      'role':'prospective_new_component_paired_two_T5_weight_confirmation','external_preregistration':False,
      'queries':sample['queries'],'independent_components':sample['groups'],'corpus_documents':sample['corpus_documents'],
      'weights':MODEL_PATHS,'retriever':'models/codebert-ssqr-pooler-2epoch-20261019',
      'sample':record(OUT/'SAMPLE_PLAN.json'),'precision':record(OUT/'PRECISION_PLAN.json'),
      'generation':{'seed':101,'do_sample':False,'num_beams':1,'max_new_tokens':19,'positions':'1..n',
          'batch_positions':32,'query_block':32,'precision':'float32','dropout':False,'empty_and_duplicate_positions_retained':True},
      'selection':{'ENTR':'ascending full vocabulary teacher-forcing mean entropy, then original position',
          'PROB':'descending teacher-forcing mean log probability, then original position','RAND_exact':'uniform position subsets without replacement'},
      'retrieval':{'representation':'roberta_pooler_output','normalization':'none','similarity':'FP32 dot',
          'TF32':False,'max_length':256,'encoding_batch':32,'scoring_batch':128,'depth':100,'tie':'score descending, doc_id ascending'},
      'estimands':{'S':'selector minus exact uniform random from its own position pool',
          'B':'S(1,mean)-S(3,mean)','G':'S(3,mean)-S(3,best3)','paired':'t5base minus local, same query/corpus/qrels/retriever',
          'mean':'uniform adoption expected RR; candidate count is selection count, not generation cost',
          'best3':'label-assisted maximum, not executable policy','short_pool':'clip k to pool length, no original padding'},
      'primary_family':{'names':PRIMARY,'size':FAMILY,'alpha_two_sided':.05,'correction':'Bonferroni',
          'decision':'both t and percentile bootstrap corrected intervals on same side of zero, >=20 nonzero groups',
          'direction_stability':'all four within-weight ENTR B/G strictly positive under this rule',
          'direct_weight_differences':'two prespecified paired contrasts in the same six-comparison family'},
      'statistics':{'sampling_unit':'connected component, exactly one query each','bootstrap_repetitions':REPETITIONS,
          'bootstrap_seed':SEED,'shared_draws':True,'secondary':'PROB, ENTR-PROB, full k1/2/3 mean/max and J; descriptive95 only'},
      'no_equivalence_test':True,'no_optional_stopping':True,'no_training_tuning_or_seed_search':True,
      'new_effects_computed':False,'risk_certified':False,'foundation_pretraining_overlap_unknown':True,
      'scope':'specified new StaQC condition, CodeBERT, two same-family T5 weight conditions; not isolated training attribution or cross-family universality',
      'environment':{'python':platform.python_version()},'bound_files':[record(p) for p in sorted(bound)],
      'protected_files':[record(ROOT/p) for p in sorted(protected)]}
    write(OUT/'DESIGN.json',design)
    print({'frozen':True,'queries':sample['queries'],'corpus':sample['corpus_documents'],'bound':len(bound),'protected':len(protected)},flush=True)


if __name__=='__main__':main()
