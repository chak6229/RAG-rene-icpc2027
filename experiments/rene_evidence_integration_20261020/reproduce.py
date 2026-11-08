"""Portable, isolated cached-evidence replay; no workspace imports."""
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def main():
    root=Path(__file__).resolve().parents[2]
    for path in [
        root/'experiments/rene_evidence_integration_20261020/verify_payload.py',
        root/'experiments/rene_envelope_completion_20261020/audit.py',
        root/'experiments/rene_evidence_integration_20261020/audit.py',
    ]:
        args=['--root',str(root)] if path.name=='audit.py' else []
        if 'rene_envelope_completion_20261020' in str(path):
            args.append('--cached')
        elif path.name=='audit.py':
            args += ['--output',str(root/'CLEAN_TABLE_AUDIT.json')]
        subprocess.run([sys.executable,'-I',str(path),*args],cwd=root,check=True)
    envelope=root/'results/rene_envelope_completion_20261020/CACHED_AUDIT.json'
    tables=root/'CLEAN_TABLE_AUDIT.json'
    result={
        'passed':True,'created_utc':datetime.now(timezone.utc).isoformat(),
        'isolated_python':True,'cached_envelope_audit':json.loads(envelope.read_text(encoding='utf-8')),
        'table_audit':json.loads(tables.read_text(encoding='utf-8')),
        'payload_manifest_sha256':hashlib.sha256((root/'PAYLOAD_MANIFEST.json').read_bytes()).hexdigest(),
        'scope':'cached candidate/ranking/qrel -> independent RR and intervals; sealed JSON -> tables',
        'no_model_or_training_reproduction':True,'new_confirmation':False,'risk_certified':False,
    }
    with (root/'CLEAN_ACCEPTANCE.json').open('x',encoding='utf-8') as handle:
        json.dump(result,handle,ensure_ascii=False,indent=2);handle.write('\n')
    print('CLEAN_ACCEPTANCE PASS',flush=True)


if __name__=='__main__':
    main()
