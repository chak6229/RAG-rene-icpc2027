"""Bind power to the two hypotheses named in DESIGN, before fresh outcomes."""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scipy.stats import norm

from experiments.rene_strict_confirmation_20261019.common import OUT, read_json, write_json, sha, utc


def main():
    if (OUT / 'fresh/PROTOCOL.json').exists():
        raise RuntimeError('Power amendments are forbidden after confirmation freeze')
    original = read_json(OUT / 'development/POWER.json')
    design = read_json(OUT / 'development/DESIGN.json')
    records = [r for r in original['observations'] if r['family'] == 'fusion' and r['stage'] == 'risk']
    assert {r['contrast'] for r in records} == {'uq_increment', 'versus_no_rewrite'}
    cfg = design['power']
    sd = max(cfg['sd_floor'], max(r['sd_upper80'] for r in records))
    z = float(norm.ppf(1 - cfg['familywise_alpha'] / cfg['key_contrasts'] / 2) + norm.ppf(cfg['power']))
    raw = math.ceil(z * z * sd * sd / cfg['sesoi'] ** 2)
    n = math.ceil(max(raw, cfg['minimum_queries']) / cfg['rounding']) * cfg['rounding']
    value = dict(amended_at_utc=utc(), original_design_sha256=sha(OUT / 'development/DESIGN.json'),
                 original_all_stage_power_sha256=sha(OUT / 'development/POWER.json'),
                 explanation='The original maximum also powered high-variance ungated rewrite diagnostics. '
                   'This amendment powers exactly the previously named fusion post-risk primary and key secondary. '
                   'All other stage/family comparisons remain descriptive; no hypothesis or threshold changed.',
                 original_all_stage_queries=original['confirmation_queries'],
                 planning_sd=sd, z_sum=z, target_effect=cfg['sesoi'], target_power=cfg['power'],
                 required_unrounded=raw, confirmation_queries=n, observations=records,
                 sensitivity_n={str(s): math.ceil(z * z * sd * sd / s ** 2) for s in cfg['sensitivity_sesoi']},
                 fresh_outcomes_opened=False, fresh_protocol_exists=False,
                 key_contrasts=2, familywise_alpha=.05, model_unseen_claim=False)
    path = OUT / 'development/POWER_PRIMARY.json'
    if path.exists():
        value['amended_at_utc'] = read_json(path)['amended_at_utc']
    write_json(path, value, immutable=True)
    print(value, flush=True)


if __name__ == '__main__':
    main()
