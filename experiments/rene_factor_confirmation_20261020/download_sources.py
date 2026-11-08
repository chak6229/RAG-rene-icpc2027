from __future__ import annotations

import argparse
import base64
import hashlib
import time

import requests

from .common import OUT, SOURCES, now, record, sha, write, read

REPO = 'LittleYUYU/StackOverflow-Question-Code-Dataset'
API = 'https://api.github.com/repos/' + REPO


def get(session, url, **kwargs):
    for attempt in range(3):
        try:
            response = session.get(url, timeout=(20, 55), **kwargs)
            response.raise_for_status()
            return response
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(1 + attempt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', action='store_true')
    ap.add_argument('--remaining', action='store_true')
    args = ap.parse_args()
    SOURCES.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({'User-Agent': 'RENE-new-sample-provenance-audit'})
    pin = SOURCES / 'staqc_PIN.json'
    if not pin.exists():
        metadata = get(session, API).json()
        commit = get(session, API + '/commits/' + metadata['default_branch']).json()
        tree = get(session, API + '/git/trees/' + commit['sha'], params={'recursive': 1}).json()
        write(SOURCES / 'staqc_REPO.json', metadata)
        write(SOURCES / 'staqc_COMMIT.json', commit)
        write(SOURCES / 'staqc_TREE.json', tree)
        write(pin, {'created_utc': now(), 'repository': REPO, 'commit': commit['sha'], 'tree_sha': tree['sha']})
    tree = read(SOURCES / 'staqc_TREE.json')
    desired = ['README.md', 'LICENSE.txt', 'final_collection/python_multi_code_iids.txt']
    if args.data:
        desired += [r['path'] for r in tree['tree'] if r['path'].endswith('.pickle') and
                    '/source/python_how_to_do_it' in r['path'] and
                    any(r['path'].endswith(s) for s in ('qid_to_title.pickle', 'qid_to_code.pickle', 'iid_to_code.pickle'))]
    if args.remaining:
        desired = [p for p in desired if not p.endswith('multiple_iid_to_code.pickle')]
    receipts = []
    for path in desired:
        r = next(r for r in tree['tree'] if r['path'] == path)
        dest = SOURCES / ('staqc_' + path.split('/')[-1])
        if not dest.exists():
            url = API + '/git/blobs/' + r['sha']
            response = get(session, url, headers={'Accept': 'application/vnd.github.raw+json'})
            if response.headers.get('Content-Type', '').startswith('application/json'):
                j = response.json()
                if j.get('encoding') != 'base64':
                    raise ValueError('API omitted blob contents: ' + path)
                content = base64.b64decode(j['content'])
            else:
                content = response.content
            expected = hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest()
            if expected != r['sha'] or len(content) != r['size']:
                raise ValueError('Git blob identity/size mismatch: ' + path)
            if not dest.exists():
                with dest.open('xb') as h:
                    h.write(content)
            elif sha(dest) != hashlib.sha256(content).hexdigest():
                raise ValueError('Concurrent cached blob differs')
        else:
            content = dest.read_bytes()
            if hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest() != r['sha']:
                raise ValueError('Cached Git blob changed')
        receipts.append({**record(dest), 'upstream_path': path, 'git_blob_sha1': r['sha'], 'bytes': len(content)})
        print({'downloaded': path, 'bytes': len(content)}, flush=True)
    name = 'staqc_REMAINING_RECEIPTS.json' if args.remaining else 'staqc_DATA_RECEIPTS.json' if args.data else 'staqc_METADATA_RECEIPTS.json'
    if not (SOURCES / name).exists():
        write(SOURCES / name, {'created_utc': now(), 'pin_sha256': sha(pin), 'files': receipts})


if __name__ == '__main__':
    main()
