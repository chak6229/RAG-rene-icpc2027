"""Verify compact-artifact bytes before evaluating any cached outcomes."""
import hashlib
import json
from pathlib import Path


def verify(root):
    ledger=json.loads((root/'PAYLOAD_MANIFEST.json').read_text(encoding='utf-8'))
    for item in ledger['files']:
        path=(root/item['path']).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise AssertionError('Missing or invalid payload path: '+item['path'])
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != item['sha256']:
            raise AssertionError('Payload hash differs: '+item['path'])
    return len(ledger['files'])


if __name__=='__main__':
    root=Path(__file__).resolve().parents[2]
    print(json.dumps({'passed':True,'payload_files':verify(root)}))
