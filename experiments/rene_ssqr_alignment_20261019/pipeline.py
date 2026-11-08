"""Label-free generation/ranking entry point for the separately aligned SSQR branch."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer

from .common import MODEL_OUTPUT, ROOT, read_json, sha256, write_json
from .data import write_jsonl
from .retriever import dot_product_scores, encode_texts, verify_checkpoint_contract
from .t5 import GenerationSettings, SCORING_CONTRACT, generate_candidates, load_tokenizer, select_candidates


def read_public(path, *, kind):
    fields = {"queries": {"query_id", "query"}, "corpus": {"doc_id", "code_text"}}[kind]
    id_field = "query_id" if kind == "queries" else "doc_id"
    text_field = "query" if kind == "queries" else "code_text"
    with Path(path).open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows or any(set(row) != fields for row in rows):
        raise ValueError(f"{kind} must use the public, label-free schema {sorted(fields)}")
    if any(not isinstance(row[text_field], str) or not row[text_field].strip() for row in rows):
        raise ValueError("input text must be nonempty")
    if kind == "queries" and any(not isinstance(row[id_field], str) for row in rows):
        raise ValueError("query IDs must be strings")
    if kind == "corpus" and any(type(row[id_field]) is not int or row[id_field] < 0 for row in rows):
        raise ValueError("corpus IDs must be nonnegative integers")
    if len({row[id_field] for row in rows}) != len(rows):
        raise ValueError("public input IDs are not unique")
    return rows


def require_development_roles(queries, role_file):
    # Synthetic checks need no role assignment. CoSQA method runs wait for a real frozen D/H allocation.
    cosqa_ids = {row["query_id"] for row in queries if row["query_id"].startswith("cosqa-")}
    if not cosqa_ids:
        return
    if role_file is None:
        raise ValueError("CoSQA method generation requires a frozen development role manifest")
    roles = read_json(Path(role_file))
    if roles.get("status") != "FROZEN" or roles.get("role") != "development":
        raise ValueError("only the frozen development role may run through this development entry point")
    allowed = set(roles.get("query_ids", []))
    if not cosqa_ids.issubset(allowed):
        raise ValueError("query input contains CoSQA IDs outside frozen development")


def generate(args):
    queries = read_public(args.queries, kind="queries")
    require_development_roles(queries, args.roles)
    tokenizer = load_tokenizer(args.t5)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.t5, local_files_only=True).to(args.device).eval()
    settings = GenerationSettings(seed=args.seed, do_sample=args.sample, max_new_tokens=args.max_new_tokens,
                                  position_policy=args.positions, batch_size=args.batch_size)
    rows = []
    for query in queries:
        rows.extend(generate_candidates(model, tokenizer, query["query_id"], query["query"], settings))
    write_jsonl(args.output, rows)
    contract = {"query_file": str(args.queries.resolve()), "query_file_sha256": sha256(args.queries),
                "candidate_file_sha256": sha256(args.output), "query_count": len(queries), "candidate_count": len(rows),
                "t5_weights_sha256": sha256(args.t5 / "model.safetensors"),
                "sentencepiece_sha256": sha256(ROOT / "models/t5-base/spiece.model"),
                "generation": settings.contract(), "scoring": SCORING_CONTRACT,
                "scorer_sha256": sha256(Path(__file__).with_name("t5.py")),
                "qrels_read": False, "retrieval_effects_computed": False,
                "roles_sha256": sha256(args.roles) if args.roles else None}
    write_json(args.output.with_suffix(".manifest.json"), contract)
    return contract


def load_trained_retriever(path, device):
    contract = verify_checkpoint_contract(path)
    training_root = path if (path / "training_complete.json").is_file() else path.parent
    complete = training_root / "training_complete.json"
    if not complete.is_file():
        raise ValueError("aligned CodeBERT training is incomplete; intermediate weights cannot pass as final")
    if path != training_root and path.name not in ("checkpoint-15740", "best"):
        raise ValueError("only the completed final checkpoint or source-validation best may be loaded")
    status = read_json(complete)
    if status.get("step") != status.get("total_steps") or status.get("step") != 15740:
        raise ValueError("aligned checkpoint has not completed the contracted two epochs")
    if read_json(training_root / "TRAINING_CONTRACT.json") != contract:
        raise ValueError("checkpoint representation/training contract differs from completed run")
    if path.name == "best" and read_json(path / "best_checkpoint.json")["step"] != status["best_step"]:
        raise ValueError("best checkpoint was not selected by the completed source-validation run")
    if path == training_root or path.name == "checkpoint-15740":
        if sha256(path / "model.safetensors") != status["final_weights_sha256"]:
            raise ValueError("final aligned CodeBERT weights changed")
    model = AutoModel.from_pretrained(path, local_files_only=True).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    return model, tokenizer


def rank(args):
    queries = read_public(args.queries, kind="queries")
    require_development_roles(queries, args.roles)
    corpus = read_public(args.corpus, kind="corpus")
    candidate_manifest = read_json(args.candidates.with_suffix(".manifest.json"))
    if candidate_manifest["query_file_sha256"] != sha256(args.queries):
        raise ValueError("candidate cache belongs to different query inputs")
    if candidate_manifest["candidate_file_sha256"] != sha256(args.candidates):
        raise ValueError("candidate cache was modified after scoring")
    with args.candidates.open(encoding="utf-8") as handle:
        candidates = [json.loads(line) for line in handle if line.strip()]
    grouped = {query["query_id"]: [] for query in queries}
    for row in candidates:
        if row["query_id"] not in grouped:
            raise ValueError("candidate cache contains unknown query ID")
        grouped[row["query_id"]].append(row)
    model, tokenizer = load_trained_retriever(args.retriever, args.device)
    code_vectors = encode_texts(model, tokenizer, [row["code_text"] for row in corpus], device=args.device)
    records = []
    doc_ids = np.array([row["doc_id"] for row in corpus], dtype=np.int64)
    for query in queries:
        actions = [{"policy": "NoRewrite", "position": None, "text": query["query"]}]
        for policy in ("ENTR", "PROB", "RAND"):
            actions.extend({"policy": policy, "position": row["position"], "text": row["rewrite"]}
                           for row in select_candidates(grouped[query["query_id"]], policy=policy, seed=args.seed))
        vectors = encode_texts(model, tokenizer, [a["text"] for a in actions], device=args.device)
        scores = dot_product_scores(vectors, code_vectors)
        for action, values in zip(actions, scores):
            order = np.lexsort((doc_ids, -values))[:min(args.top_k, len(corpus))]
            records.append({"query_id": query["query_id"], **action, "ranked_doc_ids": doc_ids[order].tolist(),
                            "scores": values[order].astype(float).tolist()})
    write_jsonl(args.output, records)
    manifest = {"status": "label_free_ranking_only", "qrels_read": False, "metrics_computed": False,
                "query_file_sha256": sha256(args.queries), "corpus_file_sha256": sha256(args.corpus),
                "candidate_file_sha256": sha256(args.candidates), "ranking_file_sha256": sha256(args.output),
                "retriever": str(args.retriever.resolve()), "retriever_weights_sha256": sha256(args.retriever / "model.safetensors"),
                "inference_precision": "float32", "representation": "pooler_output", "normalization": "none",
                "similarity": "raw_dot_product", "tie_rule": "score_desc_doc_id_asc", "records": len(records)}
    write_json(args.output.with_suffix(".manifest.json"), manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--t5", type=Path, default=ROOT / "models/t5-cqc-codenn-1m/best")
    gen.add_argument("--positions", choices=("source_1_to_n", "paper_0_to_n"), default="source_1_to_n")
    gen.add_argument("--max-new-tokens", type=int, default=19)
    gen.add_argument("--sample", action="store_true")
    gen.add_argument("--batch-size", type=int, default=8)
    ret = sub.add_parser("rank")
    ret.add_argument("--corpus", type=Path, required=True)
    ret.add_argument("--candidates", type=Path, required=True)
    ret.add_argument("--retriever", type=Path, default=MODEL_OUTPUT)
    ret.add_argument("--top-k", type=int, default=100)
    for command in (gen, ret):
        command.add_argument("--queries", type=Path, required=True)
        command.add_argument("--roles", type=Path)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
        command.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.mode == "rank" and args.top_k < 1:
        parser.error("top-k must be positive")
    result = generate(args) if args.mode == "generate" else rank(args)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
