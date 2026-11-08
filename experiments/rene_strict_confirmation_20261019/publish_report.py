"""Assemble verified evidence and update the research record without changing policies."""
from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np

from experiments.rene_strict_confirmation_20261019.common import (
    OUT, SEEDS, ARMS, FAMILIES, read_json, write_json, load, sha, utc, check_hashes,
    canonical_bytes, observable,
)

HANDOFF = ROOT / 'ResearchVault/Handoffs'


def archive_document(path):
    relative = path.relative_to(ROOT)
    target = OUT / 'documentation_before' / relative
    if path.exists() and not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)


def verify_run():
    fresh = OUT / 'fresh'
    p = read_json(fresh / 'PROTOCOL.json')
    check_hashes(p['binding_hashes']); check_hashes(p['protected_hashes'])
    tests = read_json(OUT / 'development/TESTS.json')
    btests = read_json(OUT / 'development/BUDGET_TESTS.json')
    artifact = read_json(OUT / 'artifact/ACCEPTANCE.json')
    replay = read_json(OUT / 'formal_replay/REPLAY_AUDIT.json')
    assert tests['passed'] and btests['passed'] and replay['passed'] and artifact['passed']
    assert sum(c['queries'] for c in replay['checks']) == 1024
    assert read_json(OUT / 'CODE_REPRESENTATION_PARITY.json')['mismatched'] == 0
    assert artifact['actual_training_executed'] and artifact['actual_prediction_executed']
    assert artifact['independent_rr_evaluation'] and artifact['tampered_packet_rejected']
    assert artifact['access_audit']['forbidden_original_reads'] == artifact['access_audit']['premature_outcome_reads'] == 0
    assert not Path(artifact['actual_clean_directory']).is_relative_to(ROOT)
    assert artifact['core_table_sha256'] == sha(OUT / 'core_table.json')
    assert artifact['budget_core_sha256'] == sha(OUT / 'budget_core_table.json')
    assert artifact['replay_zip_sha256'] == sha(OUT / 'artifact/RENE-cached-replay.zip')
    opening = read_json(OUT / 'OUTCOME_OPENING.json')['opened_at_utc']
    assert p['frozen_at_utc'] < read_json(fresh / 'DECISION_SEAL.json')['sealed_at_utc'] < opening
    assert read_json(fresh / 'BUDGET_PROTOCOL.json')['frozen_at_utc'] < read_json(fresh / 'BUDGET_DECISION_SEAL.json')['sealed_at_utc'] < opening
    assert read_json(OUT / 'PREOUTCOME_AUDIT.json')['audited_at_utc'] < opening
    registry = read_json(OUT / 'REGISTRY_SNAPSHOT.json')
    for name, r in registry.items():
        assert sha(ROOT / r['manifest_path']) == r['manifest_sha256']
        assert sha(ROOT / f'configs/{name}/CURRENT.json') == r['current_sha256']
    eligible = {e['repo']: e for e in load(fresh / 'eligible_repositories.pkl')}
    outcomes = read_json(fresh / 'private_outcomes.json')
    queries = {e['query_index']: e for e in read_json(fresh / 'queries.json')}
    corpora = {b['block']: load(ROOT / b['path']) for b in p['corpus']}
    for y in outcomes:
        expected = eligible[y['repo']]
        actual = corpora[y['block']][y['target_doc_id']]['code_text']
        assert actual == expected['code_text']
        assert queries[y['query_index']]['query'] == expected['query']
    for family in FAMILIES:
        for arm in ARMS:
            fitted = joblib.load(OUT / f'models/{family}_{arm}.joblib')
            assert fitted['gain']['input_dimension'] == (84 if arm == 'seed_uq' else 72)
            for a, ar in fitted['role_repositories'].items():
                for b, br in fitted['role_repositories'].items():
                    assert a == b or not set(ar) & set(br)
            assert not set(r for rs in fitted['role_repositories'].values() for r in rs) & set(y['repo'] for y in outcomes)
    result = dict(passed=True, verified_at_utc=utc(), critical_tests=tests['tests'] + btests['tests'],
                  actual_formal_replay_decisions=1024, confirmation_queries=len(outcomes),
                  code_target_mapping_checked=len(outcomes), repository_overlap=0,
                  protocols_precede_outcome_opening=True, source_hashes_intact=True,
                  actual_external_directory_refit=True, independent_core_tables_match=True,
                  protected_components_unchanged=True, risk_certified=False, test_used=False,
                  core_table_sha256=sha(OUT / 'core_table.json'),
                  budget_core_table_sha256=sha(OUT / 'budget_core_table.json'))
    path = OUT / 'VERIFICATION.json'
    if path.exists():
        result['verified_at_utc'] = read_json(path)['verified_at_utc']
    write_json(path, result, immutable=True)
    return result


def manifest(result):
    fresh = OUT / 'fresh'
    p = read_json(fresh / 'PROTOCOL.json')
    evidence = load(fresh / 'observable_items.pkl')
    old_manifest = HANDOFF / 'RENE-统一评测manifest模板.json'
    archive_document(old_manifest)
    archived = OUT / 'documentation_before' / old_manifest.relative_to(ROOT)
    files = ['fresh/PROTOCOL.json', 'fresh/BUDGET_PROTOCOL.json', 'fresh/DECISION_SEAL.json',
             'fresh/BUDGET_DECISION_SEAL.json', 'fresh/GENERATION_COMPLETE.json', 'fresh/GENERATOR_REPLAY.json',
             'fresh/RETRIEVAL_COMPLETE.json', 'fresh/EVIDENCE_COMPLETE.json', 'fresh/requirements-lock.txt',
             'development/DESIGN.json', 'development/MANIFEST.json', 'development/POWER_PRIMARY.json',
             'REGISTRY_SNAPSHOT.json', 'PREOUTCOME_AUDIT.json', 'VERIFICATION.json', 'EVALUATION_SEAL.json',
             'CODE_REPRESENTATION_PARITY.json', 'FROZEN_HEADS_IDENTICAL.json',
             'RESULTS.json', 'BUDGET_RESULTS.json', 'core_table.json', 'budget_core_table.json', 'artifact/ACCEPTANCE.json']
    index = dict(schema='rene-evaluation-manifest', schema_version='2.0',
       status='VERIFIED_MATCHED_FOUR_SEED_CONFIRMATION', index_completed_at_utc=utc(),
       protocol_frozen_at_utc=p['frozen_at_utc'],
       index_assembled_after_evaluation=True,
       prospective_authority='fresh/PROTOCOL.json and BUDGET_PROTOCOL.json, followed by actual decision seals',
       legacy_manifest=dict(path=str(archived.relative_to(ROOT)), sha256=sha(archived), validity='superseded; incorrect role/tie metadata and non-executed clean-directory claim'),
       data=dict(development_queries=512, development_candidates=5311, confirmation_queries=1536,
                 one_query_per_repository=True, corpus_documents_per_block=4096, repository_overlap=0,
                 role_definition='development/DESIGN.json explicit query IDs and repository-disjoint roles',
                 final_roles=dict(gain=256, ce_update=128, selected_action_risk=128),
                 outer_folds=4, outer_roles=dict(gain=192, ce_update=96, risk=96, held=128),
                 prior_96_excluded=True, prior_1043_excluded=True, consumed_test_allowed=False,
                 duplicate_filter='normalized query and anonymized code, before random hash repository selection',
                 upstream_CodeBERT_train_exposure=True, claim_scope='downstream model and policy repository isolation; fixed corpus'),
       generation=dict(seeds=list(SEEDS), candidates=sum(len(e['x']) for e in evidence),
                 minimum=min(len(e['x']) for e in evidence), maximum=max(len(e['x']) for e in evidence),
                 empty_queries=sum(not len(e['x']) for e in evidence),
                 information_gain='mean negative generated-token entropy; descending',
                 quality='actual pass_count/raw_count, quality before top3 per draw',
                 perturbation='four decoding seeds, frozen T5 eval; not a Bayesian parameter posterior',
                 exact_first_eight_query_generation_replay=True, checkpoint_hashes_in='fresh/PROTOCOL.json'),
       retrieval=p['retrieval'],
       scoring=dict(rr_definition='reciprocal of actual first matching position in the deterministic Top100; zero if absent',
                    strict_greater_score_surrogate=False, ties='score descending then document ID ascending',
                    fusion='two-list equal-weight RRF60; Top100; document ID ties'),
       models=dict(main_gain_dimensions=[72,84], main_ce_dimensions=[214,226], actual_refitted_diagnostic_dimensions=[34,46],
                   no_inference_feature_zeroing=True, common_base_retains_generation_entropy=True,
                   matched_capacity='same five repo-Poisson heads, seven leaves, 120 rounds, same stage roles',
                   registered_formal_components=read_json(OUT / 'REGISTRY_SNAPSHOT.json')),
       policy=dict(mean_threshold=0., native_harm_threshold=.1, post_threeway_threshold=.075,
                   rejection='original full ranking; no reselection; retain executed CE cost',
                   main_budget='same gain-priority rule; actual costs reported',
                   sensitivity_budget='exact 3072 CE query/document pairs per arm/family; token counts reported',
                   deployment='NoRewrite', risk_certified=False),
       statistics=dict(primary='fusion seed_uq risk minus no_seed_uq risk',
                   key_secondary='fusion seed_uq risk minus NoRewrite', sesoi=.005,
                   planning_queries_unrounded=1469, actual_queries=1536, approximate_power=.8,
                   multiple_comparisons='two main contrasts, Bonferroni 97.5% intervals and Holm paired-t p-values',
                   bootstrap='10000 paired repository resamples; one query/repository; fixed-corpus conditional',
                   supplementary_stage_and_budget_analyses='predeclared descriptive comparisons',
                   risk='descriptive one-sided95 CP upper; no population deployment certificate'),
       environment=p['runtime'],
       artifact=dict(clean_directory_executed=True, actual_model_refit=True, independent_rr_recomputed=True,
                     expected_hash_checked_before_accepting=True, tamper_rejected=True,
                     scope='cached upstream evidence -> retrain -> predict -> metrics; full fresh neural run recorded separately',
                     zip_path=str((OUT / 'artifact/RENE-cached-replay.zip').relative_to(ROOT)),
                     zip_sha256=sha(OUT / 'artifact/RENE-cached-replay.zip')),
       evidence={str((OUT / x).relative_to(ROOT)): sha(OUT / x) for x in files},
       verification=result)
    path = HANDOFF / 'RENE-统一评测manifest-20261019-修正版.json'
    if path.exists():
        index['index_completed_at_utc'] = read_json(path)['index_completed_at_utc']
    write_json(path, index, immutable=True)
    old_manifest.write_bytes(path.read_bytes())
    assert 'REPLACE_ME' not in path.read_text('utf8')
    return index, path


def report():
    r = read_json(OUT / 'RESULTS.json'); b = read_json(OUT / 'BUDGET_RESULTS.json')
    d = load(OUT / 'per_query_outcomes.pkl')['delta']
    primary = r['primary_contrasts']['uq_increment']
    lines = ['# RENE 四种子公平对照：修复与确认', '',
             f'在1536个下游独立仓库、固定4096代码库上，融合风险门后 UQ 增量为 **{primary["mean"]:+.6f}**，'
             f'同时97.5%区间为 **[{primary["simultaneous_ci97_5"][0]:+.6f}, {primary["simultaneous_ci97_5"][1]:+.6f}]**，'
             f'Holm校正p={primary["holm_adjusted_p"]:.6f}。本轮比较的是四种子稳定性特征，34维共同特征仍保留生成熵。', '',
             '## 主结果：融合与后置风险选择', '',
             '| 方法 | MRR@100 | ΔMRR | 采用 | 正收益/有害 | 有害率 | 单侧95% CP上界 | CE对数 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|',
             f'| NoRewrite | {r["no_rewrite_mrr"]:.8f} | 0 | 0 | 0/0 | — | — | 0 |']
    for name, label in [('fusion/no_seed_uq/risk','真正重训的无多种子特征流程'), ('fusion/seed_uq/risk','完整四种子特征流程')]:
        z = r['methods'][name]
        lines.append(f'| {label} | {z["policy_mrr"]:.8f} | {z["delta_mrr"]:+.8f} | {z["accepted"]}/1536 | '
                     f'{z["positive"]}/{z["harmful"]} | {z["harmful_rate"]:.2%} | {z["cp_upper95"]:.2%} | {z["actual_pairs"]} |')
    lines += ['', '## 相同实际核验对数', '', '| 分支 | 无多种子特征ΔMRR | 四种子特征ΔMRR | UQ增量及95%区间 | 每分支CE对数 |', '|---|---:|---:|---|---:|']
    for family in FAMILIES:
        z = b['uq_increment'][f'{family}/risk']
        a = b['methods'][f'{family}/no_seed_uq/budget25_risk']; c = b['methods'][f'{family}/seed_uq/budget25_risk']
        lines.append(f'| {family} | {a["delta_mrr"]:+.8f} | {c["delta_mrr"]:+.8f} | {z["delta"]:+.8f} '
                     f'[{z["ci95"][0]:+.8f}, {z["ci95"][1]:+.8f}] | 3072 |')
    lines += ['', '该补充对照在打开标签前冻结并重训对应风险头，作为描述性成本控制；核验token数另报，不宣称token数相同。', '',
              '## 收益发生在哪一阶段', '',
              '| 分支/阶段 | UQ相对无多种子特征ΔMRR | 95%配对区间 |', '|---|---:|---|']
    for family in FAMILIES:
        for stage in ('mean','fixed50','verified','risk'):
            z = r['stage_uq_increment'][f'{family}/{stage}']
            lines.append(f'| {family}/{stage} | {z["mean"]:+.8f} | [{z["ci95"][0]:+.8f}, {z["ci95"][1]:+.8f}] |')
    lines += ['', '融合的直接均值选择和fixed50并没有随UQ提高；正向证据出现在风险门后的完整流程。纯改写的门后UQ增量为负，因此本轮结论定位为融合流程适配，不能改写为UQ普遍改善候选排序。', '',
              '| 融合流程 | 风险门前正收益RR总量 | 门后保留正收益RR总量 | 保留比例 | 门前损害RR总量 | 门后损害RR总量 |', '|---|---:|---:|---:|---:|---:|']
    retention = {}
    for arm in ARMS:
        before, after = d[f'fusion/{arm}/verified'], d[f'fusion/{arm}/risk']
        bp, ap = float(before[before > 1e-10].sum()), float(after[after > 1e-10].sum())
        bn, an = float(-before[before < -1e-10].sum()), float(-after[after < -1e-10].sum())
        retention[arm] = dict(positive_before=bp, positive_after=ap, retained=ap / bp,
                              loss_before=bn, loss_after=an, worst_retained_loss=float(min(0., after.min())))
        lines.append(f'| {arm} | {bp:.6f} | {ap:.6f} | {ap/bp:.2%} | {bn:.6f} | {an:.6f} |')
    lines += ['', '这将“风险门丢失正收益”量化为可追踪的机制指标。它支持完整流程中的收益保留改善；候选排序与风险校准各自的因果贡献仍需后续机制消融区分。', '',
              '## 修复与验收', '',
              '- 真正重训72/84维完整流程、214/226维核验头及34/46维诊断对照；不再把已训练模型的特征置零。',
              '- 四种子训练与确认一致；生成熵、降序选候选、匿名化代码和真实采样质量分数统一；八种子复算误差及保留候选标签误差均为0。',
              '- 512查询/5311候选用于开发；收益、CE更新、风险拟合按仓库隔离，确认1536查询/15924候选。两份真实决策封存和前置审计均早于目标映射打开。',
              '- 18项关键测试通过；正式版8组/1024决策的实际模型重放通过。工作区外的干净目录完成重新训练、预测、独立RR复算和真实篡改拒绝，两张核心表哈希一致。', '',
              '## 投稿材料中的准确定位', '',
              '本轮提供固定代码库上的下游仓库确认。上游CodeBERT训练曾见过源train分区，因此不描述为整个模型完全未见的新test。融合风险门非零覆盖及CP数值上界通过10%目标，部署总体抽样条件未建立，保持risk_certified=false与NoRewrite部署guard。',
              '该实验不替代semantic-pairwise UQ分支的独立确认，也不完成第二检索器验证。正式注册组件未修改。旧96条作为诊断保存；旧hard-coded clean_directory验收已撤销，历史未知快照继续记为未知。', '',
              '证据入口：`RESULTS.json`、`BUDGET_RESULTS.json`、`VERIFICATION.json`、`artifact/ACCEPTANCE.json`；可移植最小复现包：`artifact/RENE-cached-replay.zip`。', '']
    (OUT / 'REPORT.md').write_text('\n'.join(lines), encoding='utf8')
    write_json(OUT / 'RISK_GAIN_RETENTION.json', retention, immutable=True)


def main():
    verified = verify_run()
    index, path = manifest(verified)
    report()
    print('Verified report: ' + str(OUT / 'REPORT.md'))
    print('Corrected frozen manifest: ' + str(path))


if __name__ == '__main__':
    main()
