"""Check exact original CODEnn inputs and, optionally, the decoded training/validation files."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def digest(path):
    h=sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-dir',type=Path,required=True)
    p.add_argument('--prepared-dir',type=Path)
    a=p.parse_args()
    spec=json.loads((ROOT/'docs/assets/CODENN_INPUTS.json').read_text(encoding='utf-8'))
    checked=[]
    for row in spec['files']:
        raw=row['role'] in ['source_h5','source_vocab']
        if not raw and a.prepared_dir is None:continue
        target=(a.raw_dir if raw else a.prepared_dir)/Path(row['path']).name
        assert target.stat().st_size==row['bytes'],f'Size mismatch {target.name}'
        assert digest(target)==row['sha256'],f'Hash mismatch {target.name}'
        checked.append(row['role'])
    print(json.dumps({'passed':True,'checked_roles':checked}))

if __name__=='__main__':main()
