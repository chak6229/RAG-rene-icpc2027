from __future__ import annotations

import time
import traceback

from .common import OUT, now, write, read
from .freeze import prepare
from .generate import generate
from .rank import rank
from .audit import before_effects, after_effects
from .analyze import analyze


def main():
    started=time.perf_counter()
    try:
        prepare()
        generate()
        rank()
        if not (OUT/"PRE_EFFECT_AUDIT.json").exists():before_effects()
        analyze()
        if not (OUT/"FINAL_ACCEPTANCE.json").exists():after_effects()
        write(OUT/"RUN_STATUS.json",{"completed":True,"passed":True,"updated_utc":now(),
              "seconds":time.perf_counter()-started,"role":"post_confirmation_supplementary_diagnostic"},immutable=False)
        print({"completed":True,"passed":True,"seconds":round(time.perf_counter()-started,1)},flush=True)
    except Exception as exc:
        write(OUT/"RUN_STATUS.json",{"completed":False,"passed":False,"updated_utc":now(),"error":str(exc)},immutable=False)
        traceback.print_exc()
        raise


if __name__=="__main__":main()
