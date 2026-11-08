"""Create a real isolated replay directory and execute its training and evaluation."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.rene_strict_confirmation_20261019.common import (
    OUT, load, read_json, write_json, sha, utc, require_runtime,
)
from experiments.rene_strict_confirmation_20261019.replay_worker import fingerprint


def main():
    runtime = require_runtime()
    dest = OUT / 'artifact'
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'ACCEPTANCE.json').exists():
        report = read_json(dest / 'ACCEPTANCE.json')
        if not report['passed']:
            raise ValueError('An earlier acceptance failed; preserve and inspect its evidence')
        if sha(dest / 'RENE-cached-replay.zip') != report['replay_zip_sha256']:
            raise ValueError('Accepted replay archive changed')
        print(report, flush=True)
        return
    attempt = dest / 'attempts' / str(time.time_ns())
    attempt.mkdir(parents=True, exist_ok=False)
    clean = Path(tempfile.mkdtemp(prefix='RENE-isolated-replay-')).resolve()
    if any(clean.iterdir()):
        raise ValueError('New replay directory is not empty')
    if clean.is_relative_to(ROOT):
        raise ValueError('Replay must be outside the original workspace')
    sources = list(ROOT.glob('*.py')) + list((ROOT / 'crsu').rglob('*.py'))
    sources += list((ROOT / 'experiments').rglob('*.py'))
    payload = [OUT / 'development/items.pkl', OUT / 'development/scores.pkl',
               OUT / 'development/DESIGN.json', OUT / 'development/POWER_PRIMARY.json',
               OUT / 'fresh/observable_items.pkl', OUT / 'fresh/ce_scores.pkl',
               OUT / 'fresh/private_outcomes.json', OUT / 'fresh/label_free_decisions.pkl',
               OUT / 'fresh/budget_ce_scores.pkl', OUT / 'fresh/budget_label_free_decisions.pkl',
               OUT / 'fresh/BUDGET_PROTOCOL.json', OUT / 'fresh/BUDGET_DECISION_SEAL.json',
               OUT / 'fresh/DECISION_SEAL.json', OUT / 'fresh/PROTOCOL.json', OUT / 'fresh/requirements-lock.txt']
    payload += [Path(__file__).parent / 'README.md']
    payload += list((OUT / 'models').glob('*.joblib')) + list((OUT / 'diagnostic_models').glob('*.joblib'))
    files = {}
    for source in sorted(set(sources + payload)):
        relative = source.relative_to(ROOT)
        target = clean / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        files[relative.as_posix()] = sha(source)
    expected = load(OUT / 'fresh/label_free_decisions.pkl')
    packet = dict(created_at_utc=utc(), original_workspace=str(ROOT), files=files,
                  expected_core_sha256=sha(OUT / 'core_table.json'),
                  expected_decisions_fingerprint=fingerprint(expected['outputs']),
                  expected_budget_fingerprint=fingerprint(load(OUT / 'fresh/budget_label_free_decisions.pkl')['outputs']),
                  expected_budget_core_sha256=sha(OUT / 'budget_core_table.json'),
                  expected_decision_seal_sha256=sha(OUT / 'fresh/DECISION_SEAL.json'),
                  runtime=runtime, cache_retraining_scope=True)
    write_json(clean / 'PACKET_MANIFEST.json', packet, immutable=True)
    write_json(attempt / 'PACKET_MANIFEST.json', packet, immutable=True)
    worker = Path('experiments/rene_strict_confirmation_20261019/replay_worker.py')
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    env.update(PYTHONHASHSEED='0', TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1')
    began = time.perf_counter()
    process = subprocess.run([sys.executable, '-I', str(worker)], cwd=clean, env=env,
                             capture_output=True, text=True, encoding='utf8', errors='replace')
    (attempt / 'clean-replay.stdout.txt').write_text(process.stdout, encoding='utf8')
    (attempt / 'clean-replay.stderr.txt').write_text(process.stderr, encoding='utf8')
    if process.returncode:
        write_json(attempt / 'FAILED_ACCEPTANCE.json', dict(passed=False, directory=str(clean),
                   exit_code=process.returncode, elapsed_seconds=time.perf_counter() - began))
        raise RuntimeError('Actual clean replay failed; see ' + str(attempt / 'clean-replay.stderr.txt'))
    report = read_json(clean / 'WORKER_ACCEPTANCE.json')
    if not report['passed'] or report['core_table_sha256'] != packet['expected_core_sha256']:
        raise ValueError('Worker did not satisfy the prebound core table')
    # A genuine tamper test in another empty directory must fail the packet check.
    tamper = Path(tempfile.mkdtemp(prefix='RENE-tamper-check-')).resolve()
    shutil.copytree(clean, tamper, dirs_exist_ok=True)
    modified = tamper / 'results/rene_strict_confirmation_20261019/development/DESIGN.json'
    modified.write_text(modified.read_text('utf8') + '\n', encoding='utf8')
    rejected = subprocess.run([sys.executable, '-I', str(worker), '--verify-only'], cwd=tamper, env=env,
                              capture_output=True, text=True, encoding='utf8', errors='replace')
    if rejected.returncode == 0 or 'Packet hash mismatch' not in rejected.stderr:
        raise ValueError('Altered clean packet was not rejected')
    for name in ('WORKER_ACCEPTANCE.json', 'REPLAY_DECISION_SEAL.json', 'recomputed_core.json', 'recomputed_budget_core.json'):
        shutil.copyfile(clean / name, dest / name)
    shutil.copyfile(attempt / 'PACKET_MANIFEST.json', dest / 'PACKET_MANIFEST.json')
    for name in ('clean-replay.stdout.txt', 'clean-replay.stderr.txt'):
        shutil.copyfile(attempt / name, dest / name)
    archive = dest / 'RENE-cached-replay.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        z.write(clean / 'PACKET_MANIFEST.json', 'PACKET_MANIFEST.json')
        for relative in sorted(files):
            z.write(clean / relative, relative)
    report.update(clean_directory=True, actual_clean_directory=str(clean),
                  clean_directory_initially_empty=True, subprocess_exit_code=process.returncode,
                  isolated_python=True, tampered_packet_rejected=True, test_used=False,
                  runtime=runtime, replay_zip_sha256=sha(archive),
                  packet_manifest_sha256=sha(dest / 'PACKET_MANIFEST.json'),
                  acceptance_attempt=str(attempt.relative_to(ROOT)),
                  elapsed_seconds=time.perf_counter() - began)
    write_json(dest / 'ACCEPTANCE.json', report, immutable=True)
    print(report, flush=True)


if __name__ == '__main__':
    main()
