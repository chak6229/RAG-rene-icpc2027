from __future__ import annotations

import gc
import json
import time

from .common import (
    OUT, DEVELOPMENT, T5, now, read, readl, write, writel, sha, record, require,
    verify_frozen, deny_private_reads, position_seed, PROFILES, pools_for,
)
from .evaluation import selections


def generate_positions(model, tokenizer, query_id, query, *, positions, sample, max_new_tokens):
    """Per-position sampling prevents the position-zero ablation changing common draws."""
    import torch
    from experiments.rene_ssqr_alignment_20261019.t5 import insertion_inputs, teacher_forced_scores, _inputs
    rows = [row for row in insertion_inputs(query, position_policy="paper_0_to_n") if row["position"] in positions]
    tokens = query.split()
    for row in rows:
        seed = position_seed(query_id, row["position"])
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        inputs = _inputs(tokenizer, [row["masked_query"]], next(model.parameters()).device)
        kwargs = {"do_sample": sample, "num_beams": 1, "num_return_sequences": 1,
                  "max_new_tokens": max_new_tokens, "use_cache": True,
                  "decoder_start_token_id": tokenizer.pad_token_id,
                  "eos_token_id": tokenizer.eos_token_id, "pad_token_id": tokenizer.pad_token_id}
        if sample:
            kwargs.update(temperature=1., top_k=0, top_p=1.)
        with torch.inference_mode():
            generated = model.generate(**inputs, **kwargs)[0]
        span = tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
        p = row["position"]
        rewrite = " ".join(tokens[:p] + ([span] if span else []) + tokens[p:])
        row.update({"query_id": query_id, "span": span, "rewrite": rewrite,
                    "is_identity": rewrite == " ".join(tokens), "position_rng_seed": seed,
                    "generated_ids": generated.cpu().tolist()})
    metrics = teacher_forced_scores(model, tokenizer, [row["masked_query"] for row in rows],
                                   [row["span"] for row in rows], batch_size=8)
    for row, metric in zip(rows, metrics):
        row.update(metric)
    return rows


def generate():
    protocol = verify_frozen()
    if (OUT / "CANDIDATES_SEALED.json").exists():
        from .common import verify_seal
        return verify_seal("CANDIDATES_SEALED.json")
    import torch
    from transformers import AutoModelForSeq2SeqLM
    from experiments.rene_ssqr_alignment_20261019.t5 import GenerationSettings, generate_candidates, load_tokenizer
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    queries = {
        "source": readl(OUT / "data/source/queries.jsonl"),
        "cosqa": readl(DEVELOPMENT / "data/queries.jsonl"),
    }
    started = time.perf_counter()
    tokenizer = load_tokenizer(T5)
    model = AutoModelForSeq2SeqLM.from_pretrained(T5, local_files_only=True).float().cuda().eval()
    paths = []
    with deny_private_reads():
        for domain in ("source", "cosqa"):
            jobs = ["baseline"] if domain == "source" else []
            jobs += ["greedy20_position0", "greedy10", "sample10"]
            sensitivity = set(protocol["samples"][domain]["sensitivity_ids"])
            for job in jobs:
                path = OUT / f"candidates/{domain}/{job}.jsonl"
                paths.append(path)
                selected_queries = [q for q in queries[domain] if job == "baseline" or q["query_id"] in sensitivity]
                if path.exists():
                    existing = readl(path)
                    require({r["query_id"] for r in existing} == {q["query_id"] for q in selected_queries}, "completed candidate bank missing queries")
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                partial = path.with_suffix(".partial.jsonl")
                completed = readl(partial) if partial.exists() else []
                completed_ids = {r["query_id"] for r in completed}
                require(completed_ids <= {q["query_id"] for q in selected_queries}, "partial cache has foreign query IDs")
                stage_started = time.perf_counter()
                with partial.open("a", encoding="utf-8") as handle:
                    for index, query in enumerate(selected_queries):
                        qid, text = query["query_id"], query["query"]
                        if qid in completed_ids:
                            continue
                        if job == "baseline":
                            rows = generate_candidates(model, tokenizer, qid, text, GenerationSettings())
                        elif job == "greedy20_position0":
                            rows = generate_positions(model, tokenizer, qid, text, positions={0}, sample=False, max_new_tokens=19)
                        elif job == "greedy10":
                            rows = generate_candidates(model, tokenizer, qid, text,
                                    GenerationSettings(max_new_tokens=9, position_policy="paper_0_to_n"))
                        else:
                            rows = generate_positions(model, tokenizer, qid, text,
                                    positions=set(range(len(text.split()) + 1)), sample=True, max_new_tokens=9)
                        handle.write("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows))
                        handle.flush()
                        if (index+1) % 64 == 0 or index+1 == len(selected_queries):
                            status = {"stage": "generation", "domain": domain, "bank": job,
                                      "queries_done": index+1, "queries_total": len(selected_queries),
                                      "stage_seconds": round(time.perf_counter()-stage_started,1)}
                            print(status, flush=True)
                            write(OUT / "RUN_STATUS.json", {**status, "updated_utc": now(), "completed": False}, immutable=False)
                partial.rename(path)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    selected_rows = []
    for domain in ("source", "cosqa"):
        for scope, profiles in (("main", ["greedy20_source"]), ("sensitivity", list(PROFILES))):
            for profile in profiles:
                pools = pools_for(domain, profile, sensitivity=scope == "sensitivity")
                for mask in ("full", "lexical"):
                    for qid, pool in pools.items():
                        selected_rows.append({"domain": domain, "scope": scope, "profile": profile, "mask": mask,
                                              "query_id": qid, **selections(pool, qid, protocol["rand_seeds"], mask=mask)})
    writel(OUT / "selections.jsonl", selected_rows)
    seal = {"created_utc": now(), "protocol_sha256": sha(OUT / "PROTOCOL.json"),
            "files": [record(p) for p in paths + [OUT / "selections.jsonl"]],
            "qrels_read": False, "effects_computed": False,
            "seconds": time.perf_counter()-started,
            "candidate_counts": {str(p.relative_to(OUT)): len(readl(p)) for p in paths},
            "common_position_reuse": True, "sampling_draws_per_position": 1}
    write(OUT / "CANDIDATES_SEALED.json", seal)
    return seal


if __name__ == "__main__":
    generate()
