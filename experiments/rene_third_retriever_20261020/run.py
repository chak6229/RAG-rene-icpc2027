from __future__ import annotations

import time
import traceback

from .common import OUT, now, read, require, write


def main():
    started = time.perf_counter()
    try:
        from .download import main as download
        from .preflight import main as preflight
        from .test_contract import main as test
        from .prepare import freeze
        from .inference import rank
        from .audit import before_effects, after_effects
        from .analysis import main as analyze
        from .report import main as report
        if not (OUT / 'MODEL_PROVENANCE.json').exists():
            download()
        preflight()
        test()
        freeze()
        rank()
        if not (OUT / 'PREOPEN_AUDIT.json').exists():
            before_effects()
        if not (OUT / 'RESULTS.json').exists():
            analyze()
        if not (OUT / 'FINAL_ACCEPTANCE.json').exists():
            after_effects()
        if not (OUT / 'DELIVERY_ACCEPTANCE.json').exists():
            report()
        require(read(OUT / 'DELIVERY_ACCEPTANCE.json')['passed'], 'Delivery not accepted')
        write(OUT / 'RUN_STATUS.json', {'updated_utc': now(), 'completed': True, 'passed': True,
              'seconds_this_run': time.perf_counter() - started, 'new_confirmation': False,
              'risk_certified': False}, immutable=False)
    except Exception as exc:
        write(OUT / 'RUN_STATUS.json', {'updated_utc': now(), 'completed': False, 'passed': False,
              'error': str(exc), 'seconds_this_run': time.perf_counter() - started}, immutable=False)
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
