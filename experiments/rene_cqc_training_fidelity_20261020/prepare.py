from __future__ import annotations

import base64
import json
import random
from collections import Counter

from .common import ROOT, PACKAGE, OUT, STORE, BASE, BRIDGE, ARM_SPECS, PROTECTED, now, read, rows, record, require, write, writel


def main():
    require(not (OUT / 'TRAIN_PROTOCOL.json').exists(), 'Training protocol already exists')
    import torch
    from transformers import AutoTokenizer
    from train_t5_cqc import corrupt
    raw = (ROOT / 'data/codenn_cqc/train.txt').read_text(encoding='utf-8').splitlines()
    require(len(raw) == 1000000, 'Original corpus size')
    indices = random.Random(20261020).sample(range(len(raw)), 32768)
    rng = random.Random(20261021)
    training = []
    for index in indices:
        source, target = corrupt(raw[index], rng)
        training.append(dict(source_index=index, query=raw[index], source=source, target=target))
    validation_all = rows(ROOT / 'data/codenn_cqc/valid_fixed_cqc.jsonl')
    vi = random.Random(20261022).sample(range(len(validation_all)), 1024)
    validation = [dict(validation_index=i, source=validation_all[i]['source'], target=validation_all[i]['target']) for i in vi]
    writel(OUT / 'data/train.jsonl', training)
    writel(OUT / 'data/validation.jsonl', validation)
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    reports = {}
    for name, dataset in [('training', training), ('validation', validation)]:
        source_ids = tokenizer([r['source'] for r in dataset], truncation=False)['input_ids']
        target_ids = tokenizer([r['target'] for r in dataset], truncation=False)['input_ids']
        sentinel = tokenizer.convert_tokens_to_ids('<extra_id_0>')
        reports[name] = dict(examples=len(dataset), source_max=max(map(len, source_ids)), target_max=max(map(len, target_ids)),
            source_gt128=sum(len(x)>128 for x in source_ids), target_gt128=sum(len(x)>128 for x in target_ids),
            source_gt512=sum(len(x)>512 for x in source_ids), target_gt512=sum(len(x)>512 for x in target_ids),
            encoder_tokens=sum(map(len,source_ids)), target_tokens=sum(map(len,target_ids)),
            missing_source_sentinel_at128=sum(sentinel not in x[:128] for x in source_ids))
    original = read(ROOT / 'results/rene_cosqa_ssqr_audit_20261019/sources/ssqr_blob_3e158a9e5e31b92b06c395437518d131f57b2be0.json')
    public = base64.b64decode(original['content']).decode('utf-8')
    with (OUT / 'PUBLIC_PRETRAIN.py').open('x', encoding='utf-8') as stream:
        stream.write(public)
    require('t_total = 100000' in public and 'set_seed(42)' in public, 'Public hard-coded settings')
    bridge = read(BRIDGE / 'PROTOCOL.json')
    samples = {}
    for domain in ('source', 'cosqa'):
        ids = bridge['samples'][domain]['sensitivity_ids']
        require(len(ids) == 512, 'Use existing fixed 512-query sample')
        allq = rows(BRIDGE / f'data/{domain}/queries.jsonl')
        lookup = {r['query_id']:r for r in allq}
        writel(OUT / f'data/{domain}_queries.jsonl', [dict(query_id=qid, query=lookup[qid]['query']) for qid in ids])
        groups = bridge['samples'][domain]['groups']
        samples[domain] = dict(query_ids=ids, groups={qid:groups[qid] for qid in ids}, queries=512, unique_groups=len({groups[qid] for qid in ids}))
    write(OUT / 'INPUT_AUDIT.json', dict(created_utc=now(), lengths=reports, selected_raw_rows=32768,
        unique_raw_texts=len({r['query'] for r in training}), validation_rows=1024,
        length_only_should_be_identical=reports['training']['source_gt128']==reports['training']['target_gt128']==0,
        shared_raw_data_and_corruptions=True, no_retrieval_labels_read=True,
        public_schedule_horizon=100000, historical_training_updates=93750, limited_budget_updates=1024,
        timing_benchmark=dict(discarded_weights=True, updates=8, mean_last6_update_seconds=.558809500032415, measured_before_design=True)))
    bound = [PACKAGE / f for f in ('__init__.py', 'common.py', 'prepare.py', 'train.py', 'DESIGN.md', 'test_train.py')]
    bound += [ROOT / 'train_t5_cqc.py', ROOT / 'data/codenn_cqc/train.txt', ROOT / 'data/codenn_cqc/manifest.json',
        ROOT / 'data/codenn_cqc/valid_fixed_cqc.jsonl', OUT / 'data/train.jsonl', OUT / 'data/validation.jsonl',
        OUT / 'INPUT_AUDIT.json', OUT / 'PUBLIC_PRETRAIN.py', OUT / 'TRAIN_TESTS.json', BRIDGE / 'PROTOCOL.json']
    bound += [OUT / f'data/{d}_queries.jsonl' for d in samples]
    bound += [p for p in BASE.iterdir() if p.is_file() and p.name in ('model.safetensors','config.json','tokenizer.json','spiece.model','generation_config.json')]
    STORE.mkdir(parents=True, exist_ok=True)
    write(OUT / 'TRAIN_PROTOCOL.json', dict(created_utc=now(), role='budget_matched_post_hoc_training_configuration_diagnostic',
        status='FROZEN_BEFORE_TRAINING', new_confirmation=False, risk_certified=False,
        common=dict(model=str(BASE.relative_to(ROOT)), initialization='same original t5-base; no continuation from adapted model',
            seed=101, raw_data_examples=32768, epochs=1, updates=1024, microbatch=8, gradient_accumulation=4,
            effective_batch=32, learning_rate=.001, clip_grad_norm=1., precision='BF16 compute / FP32 weights',
            data_order='fixed shared bank; no arm-specific shuffle/corruption', final_checkpoint_only=True,
            validation_examples=1024, no_early_stopping=True, optimizer_state_not_used_for_model_selection=True),
        arms=ARM_SPECS, samples=samples, model_store=STORE.as_posix(),
        evaluation=dict(retriever='frozen CodeBERT pooler', generation='positions1..n, greedy19 new tokens, no dropout',
            views=['own_generated_pool','fixed_historical_local_pool_rescored'],
            policies=['ENTR','PROB','RAND_exact'], budgets=[1,3], depth=100,
            primary_family='7 non-L arms x 2 domains x {change in ENTR-best3 minus RAND_exact-best3, change in ENTR-top1 minus NoRewrite}',
            family_size=28, bootstrap=10000, seed=20261020, conditional_on_training_seeds=True),
        historical_budget_not_recreated=True, no_H_evaluation=True,
        bound_files=[record(p) for p in bound], protected_files=[record(ROOT/p) for p in PROTECTED],
        environment=dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(0))))
    print(json.dumps(dict(stage='training_frozen', length_audit=reports, arms=list(ARM_SPECS)), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
