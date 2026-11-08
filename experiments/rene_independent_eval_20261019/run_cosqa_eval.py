"""Strict CoSQA v2 entry point, bound to public/private exports and frozen pilot.

Usage: python run_cosqa_eval.py --retriever CodeBERT (or BM25).
Expected queries, groups, action rankings and selections come from the frozen run.
Incomplete inputs are rejected instead of changing the denominator.
The v1 source is preserved in results/rene_cosqa_pilot_20261019/legacy_evaluator_snapshot/.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.rene_cosqa_pilot_20261019.evaluation import main

if __name__ == "__main__":
    main()
