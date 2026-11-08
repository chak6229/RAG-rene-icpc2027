from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.rene_strict_confirmation_20261019.code_representation import anonymized_code
from experiments.rene_strict_confirmation_20261019.common import DATA, OUT, load, write_json, sha


def main():
    corpus = load(DATA / 'action_code_corpus.pkl')
    by = {int(e['source_index']): e for e in corpus}
    checked, different = 0, []
    for i, line in enumerate((ROOT / 'data/codexglue/training/train.jsonl').open(encoding='utf8')):
        if i not in by:
            continue
        source = json.loads(line)
        actual = anonymized_code(source.get('code_tokens') or source.get('function_tokens'))
        if actual != by[i]['code_text']:
            different.append(i)
        checked += 1
        if checked == len(corpus):
            break
    report = dict(checked=checked, mismatched=len(different), different_source_indices=different,
                  explicit_builtin_names=True, source_sha256=sha(Path(__file__).parent / 'code_representation.py'),
                  historical_corpus_sha256=sha(DATA / 'action_code_corpus.pkl'), test_used=False)
    write_json(OUT / 'CODE_REPRESENTATION_PARITY.json', report, immutable=True)
    if checked != 4096 or different:
        raise ValueError('Historical representation is not reconstructed exactly: ' + str(report))
    print(report, flush=True)


if __name__ == '__main__':
    main()
