from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
RELEASE=ROOT/'results/rene_evidence_release_20261020'
ENV=ROOT/'results/rene_envelope_completion_20261020'
TABLES=ROOT/'results/rene_evidence_integration_20261020'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(2**23),b''):
            h.update(block)
    return h.hexdigest()


def write(path,value):
    with path.open('x',encoding='utf-8') as handle:
        json.dump(value,handle,ensure_ascii=False,indent=2);handle.write('\n')


def main():
    p=read(ENV/'PROTOCOL.json')
    checks=[]
    for item in p['bound_files']+p['protected_files']:
        assert sha(ROOT/item['path'])==item['sha256'],item['path']
        checks.append(item)
    seal=read(ENV/'RANKINGS_SEALED.json')
    assert sha(ENV/'PROTOCOL.json')==seal['protocol_sha256']
    for item in seal['files']:
        assert sha(ROOT/item['path'])==item['sha256'],item['path']
    acceptance=read(ENV/'FINAL_ACCEPTANCE.json')
    assert acceptance['passed'] and acceptance['scientific_tests']==6
    assert sha(ENV/'AUDIT.json')==acceptance['audit_sha256']
    assert sha(ENV/'RESULTS.json')==acceptance['results_sha256']
    assert sha(ENV/'RANKINGS_SEALED.json')==acceptance['rankings_seal_sha256']
    assert read(TABLES/'AUDIT.json')['passed']
    tables=read(TABLES/'TABLES.json')
    for entry in tables['sources']:
        assert sha(ROOT/entry['path'])==entry['sha256'],entry['path']
    packaging=read(RELEASE/'PACKAGING.json')
    assert sha(Path(packaging['archive']))==packaging['archive_sha256']
    clean=Path(packaging['clean_directory'])
    assert not clean.resolve().is_relative_to(ROOT.resolve())
    clean_acceptance=read(clean/'CLEAN_ACCEPTANCE.json')
    assert clean_acceptance['passed'] and clean_acceptance['isolated_python']
    manifest=read(clean/'PAYLOAD_MANIFEST.json')
    assert sha(clean/'PAYLOAD_MANIFEST.json')==packaging['payload_manifest_sha256']
    for item in manifest['files']:
        assert sha(clean/item['path'])==item['sha256'],item['path']
    for source,destination in [
        (clean/'CLEAN_ACCEPTANCE.json',RELEASE/'CLEAN_ACCEPTANCE.json'),
        (clean/'CLEAN_TABLE_AUDIT.json',RELEASE/'CLEAN_TABLE_AUDIT.json'),
        (clean/'results/rene_envelope_completion_20261020/CACHED_AUDIT.json',RELEASE/'CLEAN_ENVELOPE_AUDIT.json'),
    ]:
        assert not destination.exists()
        shutil.copy2(source,destination)
    manuscript=[]
    for path in sorted((ROOT/'paper/rene2027').glob('*')):
        if path.is_file() and path.suffix in ('.md','.bib'):
            manuscript.append(dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path)))
    result=dict(passed=True,created_utc=datetime.now(timezone.utc).isoformat(),
                full_envelope_audit_passed=True,scientific_tests=6,
                frozen_and_protected_files_verified=len(checks),sealed_ranking_outputs_verified=len(seal['files']),
                full_score_cells=144855715,full_ranking_rows=10138,actual_neural_examples=32,
                integrated_sources=9,integrated_tables=7,table_rows=38,traceable_source_cells=235,
                csv_cells_verified=353,isolated_clean_replay_passed=True,
                payload_files=len(manifest['files']),derived_input_files=len(manifest['derived_inputs']),
                archive_sha256=packaging['archive_sha256'],clean_directory=packaging['clean_directory'],
                clean_acceptance_sha256=sha(RELEASE/'CLEAN_ACCEPTANCE.json'),
                no_new_H_effects=True,original_primary_unchanged=True,
                formal_component_pointers_unchanged=True,new_confirmation=False,risk_certified=False,
                publicly_published=False,manuscript_files=manuscript)
    write(RELEASE/'FINAL_ACCEPTANCE.json',result)
    report='''# 本轮缺项补齐报告

日期：2026-10-20。**两权重可复算证据、主表/贡献整合及增量artifact独立验收均已完成。** 保持“SSQR式候选收益在自动采纳下能实现多少”的主线，原确认和正式组件不变。

## 1. 两权重证据补齐

补齐原来只有摘要的t5-base检索结果，使用原512查询/512关联组匹配子集、原生成候选和冻结CodeBERT。没有重新训练或生成T5。新保存10138条Top100、全语料FP32得分、候选和逐query RR。

|数据与权重|Top1−原查询|best3−原查询|
|---|---:|---:|
|源侧 / 本地CQC|−.03025188|+.02998920|
|源侧 / t5-base|−.01336644|+.03852459|
|CoSQA / 本地CQC|−.02019729|+.06130778|
|CoSQA / t5-base|−.04270304|+.05669980|

八项比较校正后，四格best3正收益成立；Top1有两格校正区间跨0。完整区间见`../rene_envelope_completion_20261020/READOUT.md`。两套权重是同族稳健性诊断，不是跨模型家族或新确认。

CoSQA/base熵单选比精确随机单选高+.03475321，95%[+.01882089,+.05044681]。它同时低于原查询。因此证据支持“信号有池内排序价值，但所测执行策略未建立相对原查询的净优势”，不能以Top1负收益断言熵无信息。

## 2. 论文主表与贡献完成整合

将唯一确认、A同池直接差、B匹配采用头、RRF3/4、全H阶段、两T5权重和旧UQ证据整理成7表38行。独立核对9个来源哈希、235个来源单元、244个语义字段和353个CSV单元；引用键全部可解析。

更新大纲、引言、相关工作、参考文献、结果契约及贡献对应表。六份旧稿已快照并核对哈希；Word任务稿没有覆盖。入口：`../../paper/rene2027/integrated_results.md`、`../../paper/rene2027/contribution_evidence.md`。

**主贡献落实为三项：** 指定SSQR比较的外部检验；同候选/同query下候选预算与机会评价的直接对照；匹配采用、可执行融合及冻结核验/回退的受控经验扩展。A的CoSQA/CodeBERT直接差+.02556783，四项校正区间[+.01416754,+.03661756]，是超出单纯重报Top1/Top3差距的具体证据。PROB相近，预算与max聚合共同改变，不将其包装为熵独有机制。

旧1536全流程UQ增量+.00395239的阳性保留；1024风险头比较的指定等效/未决也保留，各归其数据和估计对象。它们不混写成“UQ全部无效”。

## 3. 独立复核和增量artifact

- 全证据审计：144855715个得分、10138条完整排名重算一致；32个实际模型样例一致；旧摘要最大差3.33e−16；6项科学口径测试通过。
- 本轮59个冻结/保护文件及10个封存排名输出再次哈希通过，包含正式五指针和原确认。
- 约27.1 MB增量包：`RENE-evidence-supplement-20261020.zip`，94个文件、8个可追溯过滤子集。
- 工作区外空目录解压，以隔离Python运行，包内哈希、18680候选RR、16384查询指标、组区间与七表全部通过。验收见`CLEAN_ACCEPTANCE.json`及`FINAL_ACCEPTANCE.json`。

缓存包不含模型和全得分，其验收范围是缓存排名到RR/区间及封存JSON到表格；不冒称重新生成、重训或重跑此前全部H/OOF实验。完整得分/模型核对另由原审计记录。包在本地保存，尚未公开发布。

## 4. 对RENE主线的支撑与剩余任务

本轮关闭了“跨权重摘要不可重放”和“结果尚未对应论文贡献”的缺口。已有证据支持方法特定的外部复现和受控经验扩展；并未建立新UQ算法或风险证书有效。这一定位与当前RENE主线一致，不需要再换主线。

后续重点是完成全文方法/讨论、压缩结果、图表排版、匿名与许可验收、最终投稿包。实现恢复程度按差异表交代，两个权重不能恢复不可得作者checkpoint。发表判断取决于具体新增认识与全文呈现，不能用实验审计通过替代。

同目标来源配对C仅在新增“真实查询来源导致变化”的因果主张时需要；新合格数据确认仅在新增策略泛化或UQ有效主张时启动。当前稿不作这些主张，不为补数量自动重开已经消费的CoSQA或扫描H。原冻结确认继续有效，risk_certified=false。
'''
    with (RELEASE/'本轮补全报告.md').open('x',encoding='utf-8') as handle:
        handle.write(report)
    print(json.dumps({k:v for k,v in result.items() if k!='manuscript_files'},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
