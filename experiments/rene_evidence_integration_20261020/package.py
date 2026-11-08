"""Build a self-contained local cached supplement, never replace prior archives."""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
RELEASE=ROOT/'results/rene_evidence_release_20261020'
STAGE=RELEASE/'payload'
ENV='results/rene_envelope_completion_20261020'
BRIDGE='results/rene_ssqr_bridge_20261019'
INTEGRATION='results/rene_evidence_integration_20261020'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    STAGE.mkdir(parents=True,exist_ok=False)
    transforms=[]

    def copy(relative):
        source,target=ROOT/relative,STAGE/relative
        if target.exists():
            assert sha(source)==sha(target)
            return
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)

    def subset(relative,keep,description):
        source,target=ROOT/relative,STAGE/relative
        source_rows=rows(source)
        selected=[r for r in source_rows if keep(r)]
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('x',encoding='utf-8') as handle:
            for record in selected:
                handle.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
        transforms.append(dict(path=relative,original_sha256=sha(source),derived_sha256=sha(target),
                               original_rows=len(source_rows),derived_rows=len(selected),operation=description))

    for directory in ('experiments/rene_envelope_completion_20261020','experiments/rene_evidence_integration_20261020'):
        for path in sorted((ROOT/directory).glob('*.py')):
            copy(path.relative_to(ROOT).as_posix())
    for path in (ROOT/INTEGRATION).glob('*'):
        if path.is_file() and path.suffix in ('.json','.csv','.md'):
            copy(path.relative_to(ROOT).as_posix())
    bundle=read(ROOT/INTEGRATION/'TABLES.json')
    for item in bundle['sources']:
        copy(item['path'])
    proofs=[
        'results/rene_cosqa_confirmation_20261019/FINAL_ACCEPTANCE.json',
        'results/rene_core_step_a_20261020/AUDIT.json',
        'results/rene_core_step_b_20261020/FINAL_ACCEPTANCE.json',
        'results/rene_multicandidate_fusion_20261020/FINAL_ACCEPTANCE.json',
    ]
    for path in proofs:
        copy(path)
    for name in ('PROTOCOL.json','RANKINGS_SEALED.json','RESULTS.json','AUDIT.json','TEST_RESULTS.json','FINAL_ACCEPTANCE.json','OUTCOME_ACCESS.json','README.md','READOUT.md','per_query.jsonl','candidate_RR.jsonl'):
        copy(ENV+'/'+name)
    protocol=read(ROOT/ENV/'PROTOCOL.json')
    for domain in ('source','cosqa'):
        ids=set(protocol['samples'][domain]['query_ids'])
        subset(BRIDGE+f'/data/{domain}/queries.jsonl',lambda r:r['query_id'] in ids,'keep protocol historical 512 query IDs')
        subset(BRIDGE+f'/data/private/{domain}_qrels.jsonl',lambda r:r['query_id'] in ids,'keep protocol historical 512 query IDs')
        local_path=BRIDGE+'/candidates/source/baseline.jsonl' if domain=='source' else 'results/rene_cosqa_development_20261019/candidates.jsonl'
        subset(local_path,lambda r:r['query_id'] in ids,'keep all candidate positions for the protocol historical 512 query IDs')
        local_candidates=rows(STAGE/local_path)
        queries=rows(STAGE/BRIDGE/f'data/{domain}/queries.jsonl')
        texts={r['rewrite'] for r in local_candidates}|{r['query'] for r in queries}
        text_ids={hashlib.sha256(s.encode()).hexdigest() for s in texts}
        subset(BRIDGE+f'/rankings/{domain}/CodeBERT.jsonl',lambda r:r['text_id'] in text_ids,'keep all original and local-candidate text rankings for the selected queries')
        for name in ('Top100.jsonl','texts.json','MANIFEST.json'):
            copy(ENV+f'/rankings/{domain}/'+name)
        copy('results/rene_ssqr_rewriter_envelope_20261020/candidates/'+domain+'/t5base.jsonl')
        for arm in ('local','t5base'):
            copy('results/rene_ssqr_rewriter_envelope_20261020/'+domain+'_'+arm+'.json')
    for path in [BRIDGE+'/data/source/corpus.jsonl',
                 'results/rene_ssqr_alignment_20261019/cosqa_adapter/public/corpus.jsonl',
                 'results/rene_multicandidate_fusion_20261020/PRIOR_WORK_COMPARISON.md',
                 'results/rene_contribution_fulltext_20261020/REVIEW.json',
                 'results/rene_cosqa_ssqr_audit_20261019/REPORT.md',
                 'results/rene_cosqa_ssqr_audit_20261019/selected_source/coclr/LICENSE',
                 'results/rene_cosqa_ssqr_audit_20261019/selected_source/codesearchnet/LICENSE',
                 'results/rene_cosqa_ssqr_audit_20261019/selected_source/cuda/C-UDA-1.0.md']:
        copy(path)
    for path in (ROOT/'paper/rene2027').glob('*'):
        if path.suffix in ('.md','.bib'):
            copy(path.relative_to(ROOT).as_posix())
    snapshot=ROOT/'paper/rene2027/history/20261020-before-evidence-integration'
    for path in snapshot.iterdir():
        if path.is_file():
            copy(path.relative_to(ROOT).as_posix())
    (STAGE/'requirements-cached.txt').write_text('numpy==2.2.6\nscipy==1.15.3\nthreadpoolctl==3.6.0\n',encoding='utf-8')
    (STAGE/'README.md').write_text('''# RENE cached evidence supplement — 2026-10-20

This local supplement closes the missing two-T5-weight ranking evidence and integrates seven manuscript tables. It is not a public release or a new confirmation set.

## Replay

Use Python 3.10 with `requirements-cached.txt`. From an extracted clean directory run:

    python -I experiments/rene_evidence_integration_20261020/reproduce.py

The entry verifies every bundled file hash, independently recomputes candidate RR, query metrics, grouped intervals and historical summaries from cached rankings, and checks sealed JSON against table/CSV/manuscript cells. It uses no production metric imports. Success creates CLEAN_ACCEPTANCE.json; existing output files are not overwritten, so use a new extraction for each run.

## Scope

The envelope cache contains both complete code corpora, the exact historical 512 queries/groups per domain, their qrels, all candidate positions, and required local/base Top100 rankings. Eight input paths are filtered historical subsets; PAYLOAD_MANIFEST.json records original and derived hashes and row counts. The original protocol/seal remain unchanged as provenance; full-original hash verification requires the original workspace. The portable manifest verifies these derived inputs instead.

The original full audit separately checked 144855715 FP32 scores, 10138 full rankings and 32 actual model examples. Scores, model weights and vectors are omitted here. Cached replay does not repeat these neural checks, generate T5 candidates, retrain models, or reproduce upstream training. The integrated-table audit validates nine sealed result JSON sources; it does not independently rerun all earlier H/OOF experiments. Their prior audit reports are included as provenance.

Source-to-table roles remain distinct: the frozen 7718/1963 primary, full-H 8530/1963 predefined stages, development 1490/512, source 2048/594, matched sensitivity 512/512 per domain, and separate prior 1536/1024 studies. No H outcomes are newly estimated. Two T5-base weight sets are not two model families. No risk certificate is issued.

## Data and permissions

Data are derived from CoSQA/CoCLR and CodeSearchNet/CodeXGLUE. The source/overlap report and original CoCLR, CodeSearchNet and C-UDA notices are included under results/rene_cosqa_ssqr_audit_20261019/. Their terms remain applicable; the supplement does not relicense third-party data. The source report is dated 2026-10-19 and records that historical audit stage, not the later model status. Literature source snapshots and upstream generation/model files referenced in provenance are not all bundled. This archive is prepared and tested locally; public artifact publication, anonymization and the final submission inventory remain separate delivery steps.
''',encoding='utf-8')
    files=[dict(path=p.relative_to(STAGE).as_posix(),sha256=sha(p),bytes=p.stat().st_size)
           for p in sorted(STAGE.rglob('*')) if p.is_file()]
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),kind='local_cached_evidence_supplement',
                  files=files,derived_inputs=transforms,scope='cached RR/statistics replay and sealed-result/table integration',
                  full_models_and_scores_included=False,new_confirmation=False,risk_certified=False)
    (STAGE/'PAYLOAD_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    archive=RELEASE/'RENE-evidence-supplement-20261020.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in sorted(STAGE.rglob('*')):
            if path.is_file():
                z.write(path,path.relative_to(STAGE).as_posix())
    clean=Path(tempfile.mkdtemp(prefix='rene-evidence-clean-20261020-'))
    assert not clean.resolve().is_relative_to(ROOT.resolve())
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            assert (clean/name).resolve().is_relative_to(clean.resolve())
        z.extractall(clean)
    info=dict(created_utc=datetime.now(timezone.utc).isoformat(),archive=str(archive),archive_sha256=sha(archive),
              archive_bytes=archive.stat().st_size,payload_files=len(files),derived_input_files=len(transforms),
              clean_directory=str(clean),clean_outside_workspace=True,
              payload_manifest_sha256=sha(STAGE/'PAYLOAD_MANIFEST.json'),publicly_published=False)
    (RELEASE/'PACKAGING.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(info,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
