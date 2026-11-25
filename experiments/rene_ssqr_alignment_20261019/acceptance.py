"""Acceptance checks for the implementation and, once available, the completed new retriever."""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from .common import MODEL_OUTPUT, OUT, ROOT, read_json, sha256, write_json
from .data import write_jsonl
from .pipeline import generate, rank
from .retriever import REPRESENTATION, encode_texts, full_batch_loss, verify_checkpoint_contract


def now():
    return datetime.now(timezone.utc).isoformat()


def file_contract(path):
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)}


def check_protected():
    source = read_json(ROOT / "results/rene_cosqa_ssqr_audit_20261019/LOCAL_CHECKPOINTS.json")
    checks = []
    for name, expected in source["protected_snapshot"].items():
        actual = sha256(ROOT / name)
        checks.append({"path": name, "expected": expected, "actual": actual, "passed": actual == expected})
    for item in source["weights"]:
        if item.get("exists") is False:
            continue
        actual = sha256(ROOT / item["path"])
        checks.append({"path": item["path"], "expected": item["sha256"], "actual": actual,
                       "passed": actual == item["sha256"]})
    if not all(row["passed"] for row in checks):
        raise AssertionError("a previously protected artifact changed")
    return checks


def verify_implementation():
    tests = read_json(OUT / "TEST_RESULTS.json")
    if not tests["passed"] or tests["tests_run"] != 19:
        raise AssertionError("implementation test suite failed or is incomplete")
    for row in tests["source_files"]:
        if sha256(ROOT / row["path"]) != row["sha256"]:
            raise AssertionError("tested implementation changed after the recorded test run")
    adapter = read_json(OUT / "cosqa_adapter/MANIFEST.json")
    if adapter["adapter_sha256"] != sha256(Path(__file__).with_name("data.py")):
        raise AssertionError("CoSQA adapter evidence is stale")
    for row in adapter["files"] + adapter["sources"]:
        if sha256(ROOT / row["path"]) != row["sha256"]:
            raise AssertionError("CoSQA adapter output or source changed")
    smoke = read_json(OUT / "T5_REAL_WEIGHT_SMOKE.json")
    if not smoke["passed"] or smoke["vocab_size"] != 32128 or smoke["cosqa_queries_used"] != 0:
        raise AssertionError("real T5 smoke did not pass")
    if smoke["scorer_sha256"] != sha256(Path(__file__).with_name("t5.py")):
        raise AssertionError("T5 scorer changed after real-weight verification")
    protected = check_protected()
    synthetic = OUT / "synthetic"
    queries = [{"query_id": "synthetic-alignment-q1", "query": "python sort dictionary keys"}]
    corpus = [{"doc_id": 0, "code_text": "def ordered_keys ( mapping ) : return sorted ( mapping . keys ( ) )"},
              {"doc_id": 1, "code_text": "def remove_empty ( items ) : return [ x for x in items if x ]"},
              {"doc_id": 2, "code_text": "def read_file ( path ) : return open ( path ) . read ( )"}]
    write_jsonl(synthetic / "queries.jsonl", queries)
    write_jsonl(synthetic / "corpus.jsonl", corpus)
    generated = generate(SimpleNamespace(queries=synthetic / "queries.jsonl", roles=None,
                         t5=ROOT / "models/t5-cqc-codenn-1m/best", device="cpu", seed=101, sample=False,
                         max_new_tokens=19, positions="source_1_to_n", batch_size=4,
                         output=synthetic / "candidates.jsonl"))
    result = {"passed": True, "created_utc": now(), "tests_run": tests["tests_run"],
              "data_adapter": "PASS", "full_vocabulary_entropy": "PASS", "pooler_objective_vs_author": "PASS",
              "full_training_completed": False, "synthetic_generation": generated,
              "protected_files_and_old_weights": protected,
              "effect_evaluation_on_cosqa": False, "old_scheduler_correction":
              "Frozen audit's constant-scheduler description was wrong for the full local CodeBERT trainer; it already used linear 10% warmup. The CLS training objective still differed."}
    write_json(OUT / "IMPLEMENTATION_VERIFICATION.json", result)
    return result


def check_real_checkpoint(path):
    path = path.resolve()
    contract = verify_checkpoint_contract(path)
    if any(contract[key] != expected for key, expected in REPRESENTATION.items()):
        raise AssertionError("checkpoint representation differs")
    model = AutoModel.from_pretrained(path, local_files_only=True).cpu().eval()
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    codes = ["def keys ( obj ) : return list ( obj . keys ( ) )", "def sort_values ( seq ) : return sorted ( seq )",
             "def read ( name ) : return open ( name ) . read ( )"]
    queries = ["python dictionary keys", "sort a list of numbers", "read file contents"]
    code_ids = tokenizer(codes, max_length=256, padding="max_length", truncation=True, return_tensors="pt").input_ids
    query_ids = tokenizer(queries, max_length=256, padding="max_length", truncation=True, return_tensors="pt").input_ids
    source = ROOT / "results/rene_cosqa_ssqr_audit_20261019/selected_source/ssqr/Query Reformulation/code/model.py"
    spec = importlib.util.spec_from_file_location("author_pooler_acceptance", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    author = module.Model(model, model.config, tokenizer, None)
    with torch.inference_mode():
        loss, cv, qv = full_batch_loss(model, code_ids, query_ids)
        ref_loss, ref_cv, ref_qv = author(code_ids, query_ids)
    torch.testing.assert_close(cv, ref_cv, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(qv, ref_qv, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(loss, ref_loss, atol=5e-5, rtol=1e-6)
    first = encode_texts(model, tokenizer, queries, device="cpu", batch_size=3)
    # A fresh load validates actual saved/reloaded weights, not just the in-memory training graph.
    reloaded = AutoModel.from_pretrained(path, local_files_only=True).cpu().eval()
    second = encode_texts(reloaded, tokenizer, queries, device="cpu", batch_size=3)
    np.testing.assert_array_equal(first, second)
    if not np.isfinite(first).all():
        raise AssertionError("nonfinite saved checkpoint vectors")
    return {"passed": True, "checkpoint": path.relative_to(ROOT).as_posix(),
            "weights_sha256": sha256(path / "model.safetensors"), "synthetic_only": True,
            "author_vector_max_difference": float((cv - ref_cv).abs().max()),
            "author_loss_difference": abs(float(loss) - float(ref_loss)),
            "reload_max_difference": float(np.max(np.abs(first - second))),
            "query_vector_norms": np.linalg.norm(first, axis=1).astype(float).tolist(), "normalized": False}


def report(stage, details):
    training = read_json(OUT / "TRAINING_STATUS.json")
    complete = stage == "complete"
    checkpoint_line = (f"Final weight: `{details['training']['final_checkpoint']}`；SHA-256 `{details['training']['final_weights_sha256']}`。"
                       if complete else f"Current training status:{training['step']}/{training['total_steps']}Step; the actual status is updated in `TRAINING_STATUS.json`.")
    text = f"""# SSQR Three alignments\n\nUpdate:{now()}. Status:{'Full training and final acceptance clearance' if complete else 'Accomplish acceptance pass and complete searcher retrain is in progress'}。\n\n| Project Achievement and Acceptance Status\n|---|---|---|\n| CoSQA Enter interface | True query takes only doc; code_tokens Strings keep words; Official ID; public Query/code separated from private qrels; 20,604 Matching, 10,020 Querying, 6,267 Conformity of Codes\n|T5 negative entropy rating | 32,128 full word arrays of original teacher-forcing logits; Including sentry/EOS, exclusion padding; ENTR / PROB / RAND shared caches; Real weight reconciliation\n| CodeBERT Training and reasoning | Retraining from base pooler, unincorporated point accumulation, query→ code full batch CE; 32 Matching, 256 length, two rounds 15,740 steps, linear10%warmup, weight_decay = 0| {'Completed' if complete else 'Code validated; complete training ongoing'} |\n\n{checkpoint_line}\n\n19 Contract test passed. Real T5 lot/bar Max deviation 2.8610e-6, independent FP64 entropy max variance 3.4758e-6. The main entropy contains all padding tag positions; span-only Entropy is a separate column, which cannot be mixed.\n\n## Revert Settings\n\n- Default Follow Open JSON Source 1 n Position, visible modern greed settings; Also supports the paper 0...n position. The publicly available author does not record the full decode option and cannot claim that the original decoder status has been restored.\n- PROB is defined as the probability of teacher-forced average labels log; Author is not publicly available PROB source code, this operational definition needs to be frozen before the new experiment. The random position is not put back, the selection from the same cache is not sufficient and the original sentence is not filled.\n- T5 Use local best92000 weights; Source family T5Tokenizer legacy = True and base SentencePiece, word by word check local fast tokenizer.\n- CodeBERT Sovereignty reverted to the scheduled final two rounds checkpoint; Save only best selected by CodeXGLUE valid. Modern PyTorch, BF16 and activated check points are operating environment differences; Negative reservation in full. The last batch that is insufficient 32 achieves the reservation by source.\n- The new weights are kept separately in models/codebert-ssqr-pooler-2epoch-20261019; Old weight and five places CURRENT/STABLE are checked to be the same.\n\n## One correction from the old review\n\nThe previous freeze review incorrectly wrote the local complete CodeBERT learning rate schedule as constant; The original local script is already in use linear+10%warmup. What needs to be corrected is CLS / Normalized training target. The evidence of the old freeze is retained and has been further corrected in this report.\n\n## Evidence of acceptance and next steps\n\n- IMPLEMENTATION_VERIFICATION.json、TEST_RESULTS.json、T5_REAL_WEIGHT_SMOKE.json、TRAINING_BENCHMARK.json。\n- {'FINAL_ACCEPTANCE.json: The actual final weight passes the entire chain ranking for author vectors/losses, independent reloading, synthesis input.' if complete else '(b) Implementation of the acceptance and acceptance of author consistency, independent reloading and synthesis input chain ranking after final training on actual weights; The training was not complete before it was passed.'}\n- CoSQA Method effects have not been calculated. The next step is to freeze the development/confirmation of clusters, set-up and statistical comparisons in accordance with the established plan, before developing test runs and efficiency estimates; This task does not generate new 4/8 seed candidates. risk_certified = false, NoRewrite Maintain.\n"""
    (OUT / "REPORT.md").write_text(text, encoding="utf-8")
    handoff = ROOT / 'ResearchVault/Handoffs/ RENE - SSQR Three completed repairs -20261019.md'
    handoff.write_text(text + '\nRestore entrance: `experiments/rene_ssqr_alignment_20261019/`, result `results/rene_ssqr_alignment_20261019/`.\n', encoding="utf-8")


def finish():
    started = time.perf_counter()
    implementation = read_json(OUT / "IMPLEMENTATION_VERIFICATION.json")
    if not implementation["passed"]:
        raise AssertionError("implementation acceptance is missing")
    tested = read_json(OUT / "TEST_RESULTS.json")
    for item in tested["source_files"]:
        if sha256(ROOT / item["path"]) != item["sha256"]:
            raise AssertionError("implementation changed after testing")
    training = read_json(MODEL_OUTPUT / "training_complete.json")
    contract = read_json(MODEL_OUTPUT / "TRAINING_CONTRACT.json")
    if training["step"] != 15740 or training["total_steps"] != 15740 or training["epochs"] != 2:
        raise AssertionError("two full training epochs have not finished")
    for key, filename in (("trainer_sha256", "train_pooler.py"), ("retriever_sha256", "retriever.py")):
        if contract[key] != sha256(Path(__file__).with_name(filename)):
            raise AssertionError("training code changed after contract freezing")
    final_path = ROOT / training["final_checkpoint"]
    if sha256(final_path / "model.safetensors") != training["final_weights_sha256"]:
        raise AssertionError("final checkpoint hash differs from training completion")
    protected = check_protected()
    actual = check_real_checkpoint(MODEL_OUTPUT)
    synthetic = OUT / "synthetic"
    ranking = rank(SimpleNamespace(queries=synthetic / "queries.jsonl", roles=None,
                   corpus=synthetic / "corpus.jsonl", candidates=synthetic / "candidates.jsonl",
                   retriever=MODEL_OUTPUT, device="cpu", seed=101, top_k=3,
                   output=synthetic / "rankings.jsonl"))
    if ranking["records"] != 10 or ranking["qrels_read"] or ranking["metrics_computed"]:
        raise AssertionError("synthetic full pipeline did not produce original + 3x3 policy actions")
    result = {"passed": True, "stage": "completed_two_epoch_training_and_acceptance", "created_utc": now(),
              "implementation_verification_sha256": sha256(OUT / "IMPLEMENTATION_VERIFICATION.json"),
              "training": training, "final_weight_smoke": actual, "synthetic_ranking": ranking,
              "protected_files_and_old_weights": protected, "cosqa_effects_evaluated": False,
              "risk_certified": False, "elapsed_acceptance_seconds": time.perf_counter() - started,
              "final_weights": file_contract(MODEL_OUTPUT / "model.safetensors"),
              "best_weights": file_contract(MODEL_OUTPUT / "best/model.safetensors")}
    write_json(OUT / "FINAL_ACCEPTANCE.json", result, immutable=True)
    report("complete", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("implementation", "checkpoint", "complete", "wait"))
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.mode == "implementation":
        result = verify_implementation()
        report("implementation", result)
    elif args.mode == "checkpoint":
        if args.checkpoint is None:
            parser.error("checkpoint mode requires --checkpoint")
        result = check_real_checkpoint(args.checkpoint)
        write_json(OUT / "INTERMEDIATE_CHECKPOINT_SMOKE.json", result)
    else:
        if args.mode == "wait":
            write_json(OUT / "ACCEPTANCE_STATUS.json", {"status": "waiting_for_training", "started_utc": now()})
            deadline = time.monotonic() + 8 * 3600
            while not (MODEL_OUTPUT / "training_complete.json").is_file():
                if time.monotonic() > deadline:
                    raise TimeoutError("training did not finish within eight hours; acceptance is pending")
                time.sleep(30)
        result = finish()
        write_json(OUT / "ACCEPTANCE_STATUS.json", {"status": "passed", "completed_utc": now()})
    print(json.dumps({"passed": result["passed"], "mode": args.mode}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        write_json(OUT / "ACCEPTANCE_STATUS.json", {"status": "failed", "error": repr(error), "created_utc": now()})
        raise
