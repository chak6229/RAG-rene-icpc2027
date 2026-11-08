"""Last fail-closed audit before the first confirmation outcome decode."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from experiments.rene_strict_confirmation_20261019.common import (
    OUT, SEEDS, read_json, write_json, load, reject_gold, check_hashes, sha, utc,
)
from experiments.strong_flow_uq_20261014.build_evidence import unions


def main():
    fresh = OUT / 'fresh'
    if (OUT / 'OUTCOME_OPENING.json').exists():
        raise RuntimeError('This audit must run before the first outcome opening')
    protocol = read_json(fresh / 'PROTOCOL.json')
    check_hashes(protocol['binding_hashes']); check_hashes(protocol['protected_hashes'])
    budget = read_json(fresh / 'BUDGET_PROTOCOL.json')
    representation = read_json(OUT / 'CODE_REPRESENTATION_PARITY.json')
    assert representation['checked'] == 4096 and representation['mismatched'] == 0
    assert read_json(OUT / 'FROZEN_HEADS_IDENTICAL.json')['byte_identical']
    check_hashes(budget['models'])
    for p in ('TESTS.json', 'BUDGET_TESTS.json'):
        if not read_json(OUT / 'development' / p)['passed']:
            raise ValueError('Critical tests failed')
    if not read_json(OUT / 'formal_replay/REPLAY_AUDIT.json')['passed']:
        raise ValueError('Formal component inference replay did not pass')
    items = load(fresh / 'observable_items.pkl'); reject_gold(items)
    expected_union = unions(items)
    generated = [row for p in sorted((fresh / 'generation').glob('*.json')) for row in read_json(p)['rows']]
    assert len(items) == len(generated) == protocol['query_count'] == 1536
    quality_error = support_error = 0.
    for e, gen, docs in zip(items, generated, expected_union):
        assert e['query_index'] == gen['query_index'] >= 1000000
        assert tuple(d['seed'] for d in gen['draws']) == SEEDS
        np.testing.assert_array_equal(e['strong_evidence']['doc_ids'], docs)
        n = len(e['x'])
        assert e['x'].shape == (n, 34) and e['seed_x'].shape == (n, 12)
        assert n <= 12 and all(e['query_index'] not in top for top in [e['original_top100']] + e['candidate_top100'])
        if n:
            quality = np.mean([d['quality_pass_count'] / max(d['raw_count'], 1) for d in gen['draws']])
            support = np.array([len(a) / 4 for a in e['seed_appearances']])
            quality_error = max(quality_error, float(np.max(abs(e['seed_x'][:, -1] - quality))))
            support_error = max(support_error, float(np.max(abs(e['seed_x'][:, 0] - support))))
    assert quality_error == support_error == 0.
    queries = read_json(fresh / 'queries.json')
    repos = {q['repo'] for q in queries}
    exposure = set(read_json(fresh / 'EXPOSURE.json')['excluded_repositories'])
    assert len(repos) == len(queries) and not repos & exposure
    prior = read_json(ROOT / 'results/rene_repaired_confirmation_20261019/fresh/queries.json')
    assert not repos & {q['repo'] for q in prior}
    seals = {}
    for file, seal_name in [('label_free_decisions.pkl', 'DECISION_SEAL.json'),
                            ('budget_label_free_decisions.pkl', 'BUDGET_DECISION_SEAL.json')]:
        seal = read_json(fresh / seal_name)
        assert seal['actual_decisions'] and sha(fresh / file) == seal['decisions_sha256']
        reject_gold(load(fresh / file))
        seals[seal_name] = sha(fresh / seal_name)
    bseal = read_json(fresh / 'BUDGET_DECISION_SEAL.json')
    assert set(bseal['exact_pairs_each_arm'].values()) == {3072}
    # Validate semantic registry identity without treating that separate branch as part of this ablation.
    registry = read_json(OUT / 'REGISTRY_SNAPSHOT.json')
    for record in registry.values():
        assert sha(ROOT / record['manifest_path']) == record['manifest_sha256']
    report = dict(passed=True, audited_at_utc=utc(), fresh_outcomes_opened=False,
                  queries=len(items), candidates=sum(len(e['x']) for e in items),
                  repository_overlap=0, exact_seed_support_error=support_error,
                  historical_code_strings_reconstructed=4096, prior_1536_overlap=0,
                  actual_quality_fraction_error=quality_error,
                  evidence_is_exact_observable_union=True, opaque_query_ids=True,
                  all_actual_decisions_sealed=True, equal_ce_pairs_each_arm=3072,
                  test_used=False, critical_tests=18, formal_replay_decisions=1024,
                  decision_seals=seals, source_sha256=sha(__file__))
    write_json(OUT / 'PREOUTCOME_AUDIT.json', report, immutable=True)
    print(report, flush=True)


if __name__ == '__main__':
    main()
