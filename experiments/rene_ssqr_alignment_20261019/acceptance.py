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
    checkpoint_line = (f"最终权重：`{details['training']['final_checkpoint']}`；SHA-256 `{details['training']['final_weights_sha256']}`。"
                       if complete else f"当前训练状态：{training['step']}/{training['total_steps']}步；实际最新状态见 `TRAINING_STATUS.json`。")
    text = f"""# SSQR 三项实现对齐\n\n更新：{now()}。状态：{'完整训练与最终验收通过' if complete else '实现验收通过，完整检索器重训进行中'}。\n\n| 项目 | 实现和验收 | 状态 |\n|---|---|---|\n| CoSQA 输入接口 | 真实查询只取 doc；code_tokens 字符串保持词项；官方 ID；public 查询/代码与 private qrels 分开；20,604配对、10,020正查询、6,267代码一致 | 完成 |\n| T5 负熵位置评分 | 原始 teacher-forcing logits 的32,128全词表熵；包括哨兵/EOS、排除 padding；ENTR/PROB/RAND共享缓存；实际权重数值核对 | 完成 |\n| CodeBERT 训练与推理 | 从base重新训练pooler，无归一化点积，query→code完整批内CE；32配对、256长度、两轮15,740步、linear10%warmup、weight_decay=0 | {'完成' if complete else '代码已验证；完整训练进行中'} |\n\n{checkpoint_line}\n\n19项契约测试通过。真实T5批量/单条最大差2.8610e-6，独立FP64熵最大差3.4758e-6。主熵包含全部非padding标签位置；span-only熵单列，不能混用。\n\n## 复现设置\n\n- 默认沿公开JSON源码1…n位置、显式现代贪心设置；另支持论文0…n位置。公开作者未记录完整解码选项，不能声称恢复了原解码状态。\n- PROB定义为teacher-forced平均标签log概率；作者未公开PROB源码，此操作定义需在新实验前冻结。随机位置不放回，从同一缓存选择，候选不足不补原句。\n- T5使用本地best92000权重；源家族T5Tokenizer legacy=True和base SentencePiece，词表逐项核对本地fast tokenizer。\n- CodeBERT主权重为预定最终两轮checkpoint；另存只按CodeXGLUE valid选出的best。现代PyTorch、BF16和激活检查点是运行环境差异；完整批内负例保留。最后不足32的批次沿源实现保留。\n- 新权重单独保存在models/codebert-ssqr-pooler-2epoch-20261019；旧权重和五处CURRENT/STABLE保护项均核对不变。\n\n## 旧审查的一处纠正\n\n此前冻结审查将本地完整CodeBERT的学习率调度写为constant有误；原完整本地脚本已经使用linear+10%warmup。需要纠正的是CLS/归一化训练目标。旧冻结证据保留，本报告追加纠正。\n\n## 验收证据与下一步\n\n- IMPLEMENTATION_VERIFICATION.json、TEST_RESULTS.json、T5_REAL_WEIGHT_SMOKE.json、TRAINING_BENCHMARK.json。\n- {'FINAL_ACCEPTANCE.json：实际最终权重对作者向量/损失、独立重载、合成输入全链路排名全部通过。' if complete else '最终训练结束后执行实际权重的作者一致性、独立重载和合成输入全链路排名验收；通过前不称完整训练完成。'}\n- CoSQA方法效果尚未计算。下一步按既定计划冻结开发/确认分组、设置与统计比较，再开展开发试跑与功效估计；本任务不生成新的4/8seed候选。risk_certified=false、NoRewrite保持。\n"""
    (OUT / "REPORT.md").write_text(text, encoding="utf-8")
    handoff = ROOT / "ResearchVault/Handoffs/RENE-SSQR三项实现修复-20261019.md"
    handoff.write_text(text + "\n恢复入口：`experiments/rene_ssqr_alignment_20261019/`，结果 `results/rene_ssqr_alignment_20261019/`。\n", encoding="utf-8")


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
    local = datetime.now().strftime("%Y-%m-%d")
    daily = ROOT / f"ResearchVault/Logs/{local}.md"
    entry = (f"\n\n## SSQR三项实现完整重训验收（{datetime.now().strftime('%H:%M')}）\n\n"
             f"- 独立pooler/raw-dot/完整批内CE检索器两轮15740步已完成；最终权重SHA256 {training['final_weights_sha256']}。\n"
             "- 最终权重与作者向量/损失一致性、独立重载和合成输入端到端排名验收PASS；两项输入/熵修复已验证，旧权重与五保护项不变。\n"
             "- 结果results/rene_ssqr_alignment_20261019/FINAL_ACCEPTANCE.json；未计算CoSQA效果，risk_certified=false、NoRewrite。下一按原计划冻结新数据角色/协议后开发试跑。\n")
    with daily.open("a", encoding="utf-8") as handle:
        handle.write(entry)
    agent = ROOT / f"ResearchVault/Logs/Agent/{local}-codex.md"
    with agent.open("a", encoding="utf-8") as handle:
        handle.write("\n\n## SSQR对齐后台验收完成\n\n- acceptance入口确认最终两轮权重、作者实现一致性、重载与合成排名全部通过；未开展CoSQA效果评测。\n- 恢复Handoffs/RENE-SSQR三项实现修复-20261019.md及results/rene_ssqr_alignment_20261019/FINAL_ACCEPTANCE.json；后续仍先冻结角色与比较。\n")
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
