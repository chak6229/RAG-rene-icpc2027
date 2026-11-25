"""Deliver all prespecified results; preserve every earlier comparison family."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .common import OUT, PRIMARY, ROOT, WEIGHTS, now, read, record, sha, verify_design, verify_seal, write


def interval(v):return f'[{v[0]:+.6f}, {v[1]:+.6f}]'


def deliver():
    d=verify_design(private=True);verify_seal('CANDIDATES_SEALED.json');verify_seal('RANKINGS_SEALED.json')
    r=read(OUT/'RESULTS.json');plan=read(OUT/'PRECISION_PLAN.json');eligibility=read(OUT/'ELIGIBILITY.json')
    for name in ('TEST_RESULTS.json','PREOPEN_AUDIT.json','NUMERICAL_AUDIT.json','MODEL_AUDIT.json'):
        if not read(OUT/name)['passed']:raise ValueError('Required acceptance failed: '+name)
    primary=r['primary']
    export=[]
    for name in PRIMARY:
        x=primary[name];view,component=name.split('::')
        export.append({'comparison':name,'weight_condition':view,'component':component,'queries':r['queries'],'groups':r['groups'],
            'mean':x['mean'],'family6_t_lower':x['family6_ci_t'][0],'family6_t_upper':x['family6_ci_t'][1],
            'family6_bootstrap_lower':x['family6_ci_bootstrap'][0],'family6_bootstrap_upper':x['family6_ci_bootstrap'][1],
            'direction':x['direction'],'groups_with_nonzero':x['groups_with_nonzero']})
    with (OUT/'PRIMARY_RESULTS.csv').open('x',encoding='utf-8-sig',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(export[0]));w.writeheader();w.writerows(export)
    curves=[]
    for arm in WEIGHTS:
        for policy in ('ENTR','PROB','RAND_exact'):
            for k in (1,2,3):
                for agg in ('mean','max'):
                    x=r['views'][arm]['values'][f'{policy}_k{k}_{agg}']
                    curves.append({'weight':arm,'selector':policy,'selected_count':k,'aggregation':agg,'mean':x['mean'],
                        'ci95_t_lower':x['ci95_t'][0],'ci95_t_upper':x['ci95_t'][1],
                        'ci95_bootstrap_lower':x['ci95_bootstrap'][0],'ci95_bootstrap_upper':x['ci95_bootstrap'][1]})
    with (OUT/'COMPLETE_CURVES.csv').open('x',encoding='utf-8-sig',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(curves[0]));w.writeheader();w.writerows(curves)
    lines=['# Two sets of T5 weights B/G New sample pair confirmation','',
        f'Independent new query/connection group:{r["queries"]:,};fixed code language:{r["corpus_documents"]:,}。',
        'Qualifications, sample volumes, models, complete generation/retrievation configuration and local storage of the six statistical communities prior to new generation and effects. Not external pre-registration.',
        '', '## Six prespecified comparisons','',
        'A comparison of the six corrections t and six correction units bootstrap.',
        '|---|---:|---|---|---|---:|']
    chinese={'positive':'positive','negative':'negative','unresolved':'unresolved'}
    for name in PRIMARY:
        x=primary[name];lines.append(f'|{name}|{x["mean"]:+.6f}|{interval(x["family6_ci_t"])}|{interval(x["family6_ci_bootstrap"])}|{chinese[x["direction"]]}|{x["groups_with_nonzero"]:,}|')
    directions=[primary[f'{w}::ENTR_{c}']['direction'] for w in WEIGHTS for c in ('B','G')]
    if all(v=='positive' for v in directions):
        interpretation='Each of the four ENTR fractions is confirmed by a predefined six correction and double-domain directional rule: under this new StaQC series of queries, fixed CodeBERT and assigned full generation contracts, two sets of T5 weights B/G are matched. This directly reinforces the evidence of weighting conditions for secondary contributions.'
    else:
        resolved=sum(v=='positive' for v in directions)
        interpretation=f'Four ENTR points medium{resolved}(a) Confirm the positive direction of the item; The pre-event rule of “four positives” is not satisfied and therefore cannot be claimed to have continued stable overall in the direction of two sets of weights B/G. (a) The full results of the tables on weights and rationing differences; The experiment concretized the scope of establishment of weighting conditions and could not replace outstanding or reverse items with a combined average.'
    lines+=['',interpretation,'',
       'The ratio weight difference is base minus local. A significant margin indicates a difference in the full weight condition and does not isolate CQC training itself; The outstanding margin is not equivalent. This experiment does not set or test the equivalent of effects.',
       '', '## Compare objects and effects','',
       'B=S(1,mean)−S(3,mean),G=S(3,mean)−S(3 best3, S is ENTR precise random position pool relative to its own weight RR@ 100 Strengths. The positive value represents a weakening of the relative randomity of steps along the designated contract; It is not the proceeds of automatic execution, it is not the generation of a transfer budget and it is not the sole cause or consequence.',
       '', '|(1,mean)|S(3,mean)|S(3,best3)|Norewrite MRR@100|','|---|---:|---:|---:|---:|']
    for w in WEIGHTS:
        v=r['views'][w];lines.append('|'+w+'|'+'|'.join(f'{v["contrasts"][key]["mean"]:+.6f}' for key in ('S_ENTR_k1_mean','S_ENTR_k3_mean','S_ENTR_k3_max'))+f'|{v["values"]["NoRewrite"]["mean"]:.6f}|')
    lines+=['','PROB, ENTR - PROB and complete k1 / 2 / 3 x mean/best-k as a pre-defined descriptive result, see COMPLETE_CURVES.csv and RESULTS.json; No additional confirmation declaration of rejection.',
       '', '## Eligibility and planning','',
       f'- Existing historical references excluding 84,618 StaQC single target connectors; And exclude the full text, queries, historical generation and normative exposures.{eligibility["remaining_groups"]:,}Each component takes a fixed Hashi, a fixed Hashi selection and jamming medium; there is no interlocking of the full old and new language components.',
       f'- Only the consumption of two domains matching caches and larger local results are planned:. 005 RR Resolution, each direction 95% Objectives of effectiveness; 25,600 Under the group, the Quartet plans for the next line of action.{plan["four_directions_joint_power_lower_bound"]:.4f}. This is the planning of... 005 effects under the variance transportable assumption, and does not guarantee that smaller real effects are significant.',
       '- (b) The correction plan for the two weight differentials shall not exceed half a width... 005; Here. 005 For the purpose of measuring accuracy, it is not an equivalent tolerance community. No stopover, no seed, no modeling or policy upgrading.',
       '- accepted single-code Title/code is a fixed search condition and other codes are not fully artificially irrelevant. Unknown basic model pre-training exposure, semantic cloning and unrecorded historical exposure are not covered by local qualification guarantees.',
       '', '## Relation to existing evidence','',
       '(a) The old locale 512 match analysis and its 24 family remains the same; The same applies to the two groups of the old StaQC 13,312 group/16,384 code. The 25,600 group is another sample of new components and new fixed language, with six communities not retroactively changing the original conclusion. The new results do not support the attribution of cross-model family, cross-retrieve or full historical training.',
       '', '## Acceptance and review','',
       '14 Contract tests; Double-weighted old queries 360 Revert position span and complete ENTR/PROB sorting; Full query/speech source, group independence, old exposure, candidate/action/rank coverage review; Recosting of the stand-alone reasoned/ CDF formulae B/G; Six t and 20,000 sub-groups bootstrap stand-alone values; 32 Query and 32 code physical searcher reloading and reselling the full text.',
       '', 'Authentication documents: DESIGN.json, ELIGIBILITY.json, PRECISION_PLAN.json, PREOPEN_AUDIT.json, NUMERICAL_AUDIT.json, MODEL_AUDIT.json, FINAL_ACCEPTANCE.json.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    english=['# Prospective paired T5-weight confirmation of B/G','',
      f'A separately frozen study uses {r["queries"]:,} previously unused connected StaQC components and a fixed {r["corpus_documents"]:,}-snippet corpus. Both T5 weight conditions are paired on queries, targets, corpus and the fixed CodeBERT retriever, with their own complete insertion-position pools. No retraining or parameter search is performed. The old 512-query analyses and their original 24-comparison family remain unchanged.',
      '', 'The six prespecified contrasts comprise ENTR B/G under each weight and paired base-minus-local B/G. Two-sided Bonferroni family alpha is .05. A direction requires both the adjusted t and 20,000-draw paired component-bootstrap intervals to exclude zero on the same side, with at least 20 nonzero components. PROB and the complete curves are descriptive. The .005 planning resolution and difference-precision target do not define equivalence margins.',
      '', '|Contrast|Estimate|Adjusted t interval|Adjusted bootstrap interval|Decision|','|---|---:|---|---|---|']
    for name in PRIMARY:
        x=primary[name];english.append(f'|{name}|{x["mean"]:+.6f}|{interval(x["family6_ci_t"])}|{interval(x["family6_ci_bootstrap"])}|{x["direction"]}|')
    english+=['',
      ('All four within-weight ENTR B/G components meet the prespecified positive-direction rule, confirming the direction in both tested T5 weight conditions on this new StaQC setting.' if r['both_weights_BG_positive'] else 'The four-component positive-direction criterion is not met. Individual positive, negative and unresolved components retain their separate interpretations; an overall weight-stability claim is not established.'),
      '', 'The paired differences compare complete weight conditions, not an isolated effect of CQC training. Unresolved differences do not establish equivalence. The scope is the specified StaQC title/snippet setting, fixed CodeBERT and two weights within T5-base; it does not establish cross-family universality, automatic-policy benefit or historical training attribution.',
      '', 'Sampling excludes all prior StaQC query and corpus components, archived partial generations, normalized exposed text and inherited training/reference overlaps. The outcome-blind precision plan used consumed development data only; the four-direction joint power lower bound at .005 is '+f'{plan["four_directions_joint_power_lower_bound"]:.4f}'+', conditional on variance transport. The protocol was locally sealed before new generation and effects; it was not externally preregistered. Source mappings, complete candidate/action/ranking coverage, exact-random arithmetic, paired intervals and representative model/ranking reloads passed the specified audits.']
    (OUT/'weight_confirmation_results.md').write_text('\n'.join(english)+'\n',encoding='utf-8')
    files=[OUT/n for n in ['DESIGN.json','ELIGIBILITY.json','PRECISION_PLAN.json','SAMPLE_PLAN.json','TEST_RESULTS.json',
        'BATCHING_DEVELOPMENT_CHECK.json','CANDIDATES_SEALED.json','RANKINGS_SEALED.json','PREOPEN_AUDIT.json',
        'OUTCOME_OPENING.json','RESULTS.json','NUMERICAL_AUDIT.json','MODEL_AUDIT.json','per_query.jsonl',
        'paired_per_query.jsonl','PRIMARY_RESULTS.csv','COMPLETE_CURVES.csv','REPORT.md','weight_confirmation_results.md','RUNTIME.json']]
    files += [Path(__file__),Path(__file__).with_name('README.md')]
    write(OUT/'FINAL_ACCEPTANCE.json',{'created_utc':now(),'passed':True,'scientific_result_positive_required':False,
        'queries':r['queries'],'groups':r['groups'],'corpus_documents':r['corpus_documents'],
        'both_weights_BG_positive':r['both_weights_BG_positive'],'primary_directions':r['primary_directions'],
        'all_original_protected_hashes_unchanged':True,'protected_count':len(d['protected_files']),
        'files':[record(f) for f in files],'new_results':True,'risk_certified':False,'no_equivalence_claim':True})
    (OUT/'RUN_STATUS.json').write_text(json.dumps({'updated_utc':now(),'stage':'complete','completed':True,'passed':True,
        'primary_directions':r['primary_directions'],'both_weights_BG_positive':r['both_weights_BG_positive']},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print({'accepted':True,'primary_directions':r['primary_directions']},flush=True)


if __name__=='__main__':deliver()
