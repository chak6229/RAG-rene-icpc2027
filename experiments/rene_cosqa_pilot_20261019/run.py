"""Ordered pilot execution: freeze -> generate/rank -> audit -> open -> replay -> power."""
from __future__ import annotations

import subprocess
import sys

from .common import OUT, ROOT, now, read, write


def main():
    commands = []
    if not (OUT / "PROTOCOL.json").exists():
        commands.append(["experiments.rene_cosqa_pilot_20261019.freeze"])
    commands += [["experiments.rene_cosqa_pilot_20261019.ranking", stage] for stage in ("generate", "CodeBERT", "BM25", "seal")]
    commands += [["experiments.rene_cosqa_pilot_20261019.audit", "pre"]]
    commands += [["experiments.rene_cosqa_pilot_20261019.evaluation", "--retriever", name] for name in ("CodeBERT", "BM25")]
    commands += [["experiments.rene_cosqa_pilot_20261019.audit", "post"], ["experiments.rene_cosqa_pilot_20261019.analyze"]]
    for module in commands:
        result_path = OUT / "RUN_STATUS.json"
        write(result_path, {"updated_utc": now(), "stage": module, "status": "running"})
        result = subprocess.run([sys.executable, "-X", "utf8", "-m", *module], cwd=ROOT)
        if result.returncode:
            write(result_path, {"updated_utc": now(), "stage": module, "status": "failed", "exit_code": result.returncode,
                                "confirmation_started": False})
            raise SystemExit(result.returncode)
    write(OUT / "RUN_STATUS.json", {"updated_utc": now(), "status": "complete", "confirmation_started": False})


if __name__ == "__main__":
    main()
