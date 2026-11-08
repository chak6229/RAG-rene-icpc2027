"""Archive author instructions and retrieve commit-pinned public weights."""
from __future__ import annotations

import base64
import concurrent.futures
import hashlib
import time

import requests

from .common import MODEL, OUT, SOURCES, now, read, record, require, sha, write

REPO = 'microsoft/CodeBERT'
MODEL_ID = 'microsoft/unixcoder-base'
MODEL_COMMIT = '5604afdc964f6c53782a6813140ade5216b99006'
CODE_COMMIT = 'c0de43d3aaf38e89290f1efb771f8de845e7a489'
ENDPOINT = 'https://hf-mirror.com'


def ranged_download(url, partial, size):
    """Bounded parallel byte ranges; final upstream SHA256 remains mandatory."""
    prefix = partial.with_suffix(partial.suffix + '.prefix')
    if partial.exists() and not prefix.exists():
        partial.rename(prefix)
    offset = prefix.stat().st_size if prefix.exists() else 0
    require(offset <= size, 'Invalid existing download prefix')
    parts = partial.with_suffix(partial.suffix + '.parts')
    parts.mkdir(exist_ok=True)
    ranges = [(start, min(start + 16 * 2**20, size) - 1) for start in range(offset, size, 16 * 2**20)]

    def fetch(interval):
        start, end = interval
        dest = parts / f'{start:010d}-{end:010d}.bin'
        if dest.exists() and dest.stat().st_size == end - start + 1:
            return
        for attempt in range(4):
            try:
                response = requests.get(url, headers={'Range': f'bytes={start}-{end}'}, stream=True, timeout=(20, 60))
                response.raise_for_status()
                require(response.status_code == 206 and response.headers.get('Content-Range') == f'bytes {start}-{end}/{size}', 'Server did not honor exact byte range')
                temp = dest.with_suffix('.partial')
                with temp.open('wb') as h:
                    for block in response.iter_content(2**20):
                        if block:
                            h.write(block)
                require(temp.stat().st_size == end - start + 1, 'Range download incomplete')
                temp.replace(dest)
                print({'download_range_done': [start, end]}, flush=True)
                return
            except (requests.RequestException, ValueError):
                if attempt == 3:
                    raise
                time.sleep(attempt + 1)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(fetch, ranges))
    with partial.open('wb') as target:
        inputs = ([prefix] if offset else []) + [parts / f'{start:010d}-{end:010d}.bin' for start, end in ranges]
        for path in inputs:
            with path.open('rb') as source:
                for block in iter(lambda: source.read(2**23), b''):
                    target.write(block)
    require(partial.stat().st_size == size, 'Assembled download size mismatch')


def get(session, url, **kwargs):
    for attempt in range(3):
        try:
            response = session.get(url, timeout=(20, 50), **kwargs)
            response.raise_for_status()
            return response
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def main():
    SOURCES.mkdir(parents=True, exist_ok=True)
    MODEL.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers['User-Agent'] = 'RENE-frozen-third-retriever-research'
    metadata_path = SOURCES / 'HF_MODEL.json'
    if not metadata_path.exists():
        metadata = get(session, f'{ENDPOINT}/api/models/{MODEL_ID}/revision/{MODEL_COMMIT}?blobs=true').json()
        require(metadata['sha'] == MODEL_COMMIT, 'Unexpected model revision')
        write(metadata_path, metadata)
    metadata = read(metadata_path)
    code_files = ['UniXcoder/README.md', 'UniXcoder/unixcoder.py',
                  'UniXcoder/downstream-tasks/code-search/README.md',
                  'UniXcoder/downstream-tasks/code-search/run.py',
                  'UniXcoder/downstream-tasks/code-search/model.py', 'LICENSE']
    upstream = []
    for path in code_files:
        dest = SOURCES / path.replace('/', '__')
        receipt_path = dest.with_name(dest.name + '.receipt.json')
        if not receipt_path.exists():
            url = f'https://api.github.com/repos/{REPO}/contents/{path}?ref={CODE_COMMIT}'
            data = get(session, url).json()
            content = base64.b64decode(data['content'])
            oid = hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest()
            require(oid == data['sha'], 'Author Git blob mismatch')
            if not dest.exists():
                dest.write_bytes(content)
            require(dest.read_bytes() == content, 'Existing source differs')
            write(receipt_path, {'url': url, 'git_blob_sha1': oid, **record(dest)})
        upstream.extend([record(dest), record(receipt_path)])
    artifacts = []
    for info in metadata['siblings']:
        filename = info['rfilename']
        if filename == '.gitattributes':
            continue
        dest = MODEL / filename
        partial = dest.with_suffix(dest.suffix + '.download')
        expected = info.get('lfs', {}).get('sha256')
        if not dest.exists():
            url = f'{ENDPOINT}/{MODEL_ID}/resolve/{MODEL_COMMIT}/{filename}'
            if expected:
                ranged_download(url, partial, info['size'])
                require(sha(partial) == expected, 'Upstream LFS SHA256 mismatch')
                partial.rename(dest)
                print({'pinned': filename, 'bytes': info['size']}, flush=True)
                artifacts.append({**record(dest), 'bytes': info['size'], 'upstream_git_blob': info['blobId'], 'upstream_lfs_sha256': expected})
                continue
            offset = partial.stat().st_size if partial.exists() else 0
            response = get(session, url, stream=True, headers={'Range': f'bytes={offset}-'} if offset else {})
            append = offset > 0 and response.status_code == 206
            done = offset if append else 0
            last = time.monotonic()
            with partial.open('ab' if append else 'wb') as h:
                for block in response.iter_content(2**20):
                    if block:
                        h.write(block)
                        done += len(block)
                        if time.monotonic() - last > 20:
                            print({'download': filename, 'bytes': done, 'total': info['size']}, flush=True)
                            last = time.monotonic()
            require(partial.stat().st_size == info['size'], 'Model download size mismatch')
            if expected:
                require(sha(partial) == expected, 'Upstream LFS SHA256 mismatch')
            else:
                content = partial.read_bytes()
                require(hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest() == info['blobId'], 'Model Git blob mismatch')
            partial.rename(dest)
        require(dest.stat().st_size == info['size'], 'Pinned file size changed')
        if expected:
            require(sha(dest) == expected, 'Pinned model checksum changed')
        else:
            content = dest.read_bytes()
            require(hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest() == info['blobId'], 'Pinned tokenizer Git blob changed')
        artifacts.append({**record(dest), 'bytes': info['size'], 'upstream_git_blob': info['blobId'], 'upstream_lfs_sha256': expected})
        print({'pinned': filename, 'bytes': info['size']}, flush=True)
    if not (OUT / 'MODEL_PROVENANCE.json').exists():
        write(OUT / 'MODEL_PROVENANCE.json', {'created_utc': now(), 'model_id': MODEL_ID,
            'model_commit': MODEL_COMMIT, 'official_code_commit': CODE_COMMIT, 'download_endpoint': ENDPOINT,
            'transport_note': 'Official Hugging Face direct connection unavailable; mirror bytes verified against pinned Git/LFS identities',
            'official_usage': 'Author explicitly provides zero-shot Python/CoSQA code search with this released checkpoint',
            'model_license': 'Apache-2.0 (pinned HF card)', 'wrapper_license': 'MIT (author source)',
            'training_scope': 'C4/CodeSearchNet; continued pretraining on NL-PL CodeSearchNet pairs; Python supported',
            'exposure_limit': 'No new local training. Pretraining overlap and semantic clones not fully identifiable; this is not unseen-training generalization.',
            'files': artifacts + upstream + [record(metadata_path)]})


if __name__ == '__main__':
    main()
