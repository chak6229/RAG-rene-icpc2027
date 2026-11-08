from __future__ import annotations

import subprocess
import sys

from .common import OUT, ROOT, now, write


def main():
    package = "experiments.rene_cosqa_development_20261019."
    commands = []
    if not (OUT / "PROTOCOL.json").exists():
        commands.append([package + "freeze"])
    commands += [[package + "ranking", stage] for stage in
                 ("generate", "CodeBERT", "BM25", "features-CodeBERT", "features-BM25", "seal")]
    commands.append([package + "audit", "pre"])
    commands += [[package + "evaluation", "rq1", r] for r in ("CodeBERT", "BM25")]
    commands += [[package + "gain", r] for r in ("CodeBERT", "BM25")]
    commands += [[package + "evaluation", "rq2", r] for r in ("CodeBERT", "BM25")]
    commands += [[package + "audit", "post"], [package + "analyze"]]
    for module in commands:
        write(OUT / "RUN_STATUS.json", {"updated_utc": now(), "status": "running", "stage": module, "confirmation_started": False})
        result = subprocess.run([sys.executable, "-X", "utf8", "-m", *module], cwd=ROOT)
        if result.returncode:
            write(OUT / "RUN_STATUS.json", {"updated_utc": now(), "status": "failed", "stage": module,
                                           "exit_code": result.returncode, "confirmation_started": False})
            raise SystemExit(result.returncode)
    write(OUT / "RUN_STATUS.json", {"updated_utc": now(), "status": "complete", "confirmation_started": False})


if __name__ == "__main__":
    main()
