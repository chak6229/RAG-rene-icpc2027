"""Validate release attachments and install study models without model execution."""
import argparse
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import shutil
from zipfile import ZipFile

ROOT=Path(__file__).resolve().parents[1]

def digest_stream(stream):
    h=sha256()
    for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--asset-dir',type=Path,required=True)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--check-only',action='store_true')
    a=p.parse_args()
    manifest=json.loads((ROOT/'docs/assets/STUDY_ASSETS.json').read_text(encoding='utf-8'))
    checked=0
    for spec in manifest['assets']:
        archive=a.asset_dir/spec['filename']
        with archive.open('rb') as stream:assert digest_stream(stream)==spec['sha256'],f'Archive hash mismatch {archive.name}'
        expected={x['path']:x for x in spec['files']}
        with ZipFile(archive) as z:
            assert set(z.namelist())==set(expected),'Unexpected archive entries'
            for name,record in expected.items():
                relative=PurePosixPath(name)
                assert not relative.is_absolute() and '..' not in relative.parts and ':' not in name
                with z.open(name) as stream:assert digest_stream(stream)==record['sha256'],f'File hash mismatch {name}'
                checked+=1
                if a.check_only:continue
                target=a.root.joinpath(*relative.parts)
                if target.exists():
                    with target.open('rb') as stream:assert digest_stream(stream)==record['sha256'],f'Existing file differs {name}'
                    continue
                target.parent.mkdir(parents=True,exist_ok=True)
                temp=target.with_name(target.name+'.partial')
                with z.open(name) as source,temp.open('wb') as dest:shutil.copyfileobj(source,dest)
                temp.replace(target)
    print(json.dumps({'passed':True,'archives':len(manifest['assets']),'files':checked,'installed':not a.check_only}))

if __name__=='__main__':main()
