from __future__ import annotations

import subprocess
import sys

from .common import OUT, ROOT, require, status


def main():
    prefix = "experiments.rene_cosqa_confirmation_20261019."
    jobs = []
    if not (OUT / "DESIGN.json").exists():
        jobs.append(("freeze", []))
    if not (OUT / "POLICY_SEALED.json").exists():
        jobs.append(("training", []))
    if not (OUT / "DECISIONS_SEALED.json").exists():
        jobs.append(("inference", []))
    if not (OUT / "PREOUTCOME_AUDIT.json").exists():
        jobs.append(("audit", ["pre"]))
    if not (OUT / "OUTCOME_OPENING.json").exists():
        jobs.append(("evaluation", []))
    else:
        require((OUT / "PRIMARY_RESULT.json").exists(), "previous interrupted H opening; inspect, never silently rerun/retune")
    if not (OUT / "FINAL_ACCEPTANCE.json").exists():
        jobs.append(("audit", ["post"]))
    if not (OUT / "REPORT_RECEIPT.json").exists():
        jobs.append(("report", []))
    for name, args in jobs:
        status(name)
        print({"starting": name, "arguments": args}, flush=True)
        result = subprocess.run([sys.executable, "-X", "utf8", "-m", prefix + name, *args], cwd=ROOT)
        if result.returncode:
            status("failed", failed_stage=name, exit_code=result.returncode)
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
