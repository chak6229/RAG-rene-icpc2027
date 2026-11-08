"""Repository-isolated action replay and joint query/code witness evidence.

Produces real T5 rewrites and frozen CodeBERT Top100 labels on new train
queries, and four matched semantic CE controls.  Semantic CE query repositories
are disjoint from action query repositories and from every valid repository.
Only code tokens are encoded.  No test split is read or modified.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import keyword
import math
from pathlib import Path
import pickle
import random
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer
from build_rewrite_utility_dataset import code_text, encode
from build_crsu_pilot import stable_indices
from crsu.core import RankingStates, rrf
from prepare_vac_mqr_cv_folds import repository
from run_bm25_acquisition_pilot import BM25
from vac_mqr import CandidateQualityValidator
import run_t5_seed_uq_pilot as old

HERE = Path(__file__).resolve().parent
OUT = ROOT/'results/uq_pilot_20261014/action_witness_uq/data'
TRAIN = ROOT/'data/codexglue/training/train.jsonl'
VALID = ROOT/'data/codexglue/advtest/valid.jsonl'
T5 = ROOT/'models/t5-cqc-codenn-1m/best'
RETRIEVER = ROOT/'models/codebert-codexglue-2epoch/checkpoint-3000'
CE_BASE = ROOT/'models/codebert-base'
STABLE = ROOT/'results/vac_mqr_v5/STABLE.json'
CURRENT = ROOT/'configs/uq_candidate_selection/CURRENT.json'
VALID_ITEMS = ROOT/'results/uq_pilot_20261011/t5_seed_consistency/evidence.pkl'
VALID_ROWS = ROOT/'results/uq_pilot_20261011/t5_seed_consistency/sampled_rows.pkl'
VALID_CODES = ROOT/'results/crsu_development/valid_vector_cache/code_vectors.npy'
SEED = 141414
PROJECTION_SEED = 141402
SEEDS = tuple(old.SEEDS)
CORPUS = 4096
ACTION_QUERIES = 512
ARMS = ('small_bm25_1024', 'small_dense_1024', 'large_bm25_4096', 'large_dense_4096')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()


def read(path):
    return [json.loads(line) for line in Path(path).open(encoding='utf8') if line.strip()]


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf8')


def save_pickle(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    with temp.open('wb') as stream: pickle.dump(value, stream)
    temp.replace(path)


def load_pickle(path):
    return pickle.load(Path(path).open('rb'))


def status(stage, **values):
    dump(OUT/'STATUS.json', dict(stage=stage, utc=datetime.now(timezone.utc).isoformat(),
        test_used=False, **values))


def anonymized_code(tokens):
    """Same token-only synthetic name obfuscation as the previous CE pilot."""
    declared = {tokens[i+1] for i, t in enumerate(tokens[:-1]) if t == 'def'}
    keep = set(keyword.kwlist) | {'self', 'cls', 'True', 'False', 'None'} | set(dir(__builtins__))
    mapping = {}; result = []
    for i, t in enumerate(tokens):
        previous = tokens[i-1] if i else ''; following = tokens[i+1] if i+1 < len(tokens) else ''
        if previous == '.': result.append(t)
        elif t in declared: result.append('Func')
        elif t.isidentifier() and t not in keep and following != '(':
            if t not in mapping: mapping[t] = f'arg_{len(mapping)}'
            result.append(mapping[t])
        else: result.append(t)
    return ' '.join(result)


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT/'GPU_PROTOCOL.json'
    if path.exists():
        p = json.loads(path.read_text('utf8'))
        for group in ('source_hashes', 'input_hashes', 'model_hashes'):
            for name, digest in p[group].items(): assert sha(name) == digest, name
        assert sha(STABLE) == p['stable_hash'] and sha(CURRENT) == p['current_hash']
        return p
    valid = read(VALID); excluded = {repository(r['url']) for r in valid}
    pools = {'action': [], 'semantic': []}; eligible = {'action': 0, 'semantic': 0}
    rng = {k: np.random.default_rng(SEED+(k == 'semantic')) for k in pools}
    for source_index, line in enumerate(TRAIN.open(encoding='utf8')):
        row = json.loads(line); repo = repository(row['url'])
        if repo in excluded: continue
        query = ' '.join(row['docstring_tokens'])
        if not query.strip() or len(query.split()) > 64: continue
        group = 'action' if int(hashlib.sha256(('action-witness/'+repo).encode()).hexdigest()[:8], 16)%2 == 0 else 'semantic'
        token_values = row.get('code_tokens') or row.get('function_tokens')
        assert token_values, 'code-only inputs required'
        saved = dict(source_index=source_index, source_idx=int(row.get('idx', source_index)),
            query=query, url=row['url'], repo=repo, code_text=anonymized_code(token_values))
        eligible[group] += 1
        if len(pools[group]) < CORPUS: pools[group].append(saved)
        else:
            j = int(rng[group].integers(eligible[group]))
            if j < CORPUS: pools[group][j] = saved
    assert all(len(v) == CORPUS for v in pools.values())
    for group, values in pools.items():
        values.sort(key=lambda r: r['source_index'])
        for i, row in enumerate(values): row['doc_id'] = i; row['query_index'] = i
        save_pickle(OUT/f'{group}_code_corpus.pkl', values)
    action_repos = {r['repo'] for r in pools['action']}
    semantic_repos = {r['repo'] for r in pools['semantic']}
    assert not action_repos & semantic_repos and not (action_repos | semantic_repos) & excluded
    # Round-robin repositories, so 512 distinct semantic problems are not a
    # random row sample dominated by a few large repositories.
    by_repo = defaultdict(list)
    for row in pools['action']: by_repo[row['repo']].append(row)
    reps = sorted(by_repo, key=lambda r: hashlib.sha256(('query-select/'+r).encode()).hexdigest())
    for r in reps:
        by_repo[r].sort(key=lambda row: hashlib.sha256(('query/'+row['url']).encode()).hexdigest())
    actions = []; turn = 0
    while len(actions) < ACTION_QUERIES:
        for repo in reps:
            if turn < len(by_repo[repo]): actions.append(by_repo[repo][turn])
            if len(actions) == ACTION_QUERIES: break
        turn += 1
    actions.sort(key=lambda r: r['query_index'])
    save_pickle(OUT/'action_rows.pkl', actions)
    small_ids = sorted(np.random.default_rng(SEED+3).choice(CORPUS, 1024, replace=False).tolist())
    save_pickle(OUT/'semantic_small_ids.pkl', small_ids)
    held = load_pickle(VALID_ITEMS); rows = load_pickle(VALID_ROWS)
    assert [e['query_index'] for e in held] == [r['query_index'] for r in rows]
    held = [dict(e, query=r['query'], url=r['url'], x46=np.c_[e['x'], e['seed_x']]) for e, r in zip(held, rows)]
    save_pickle(OUT/'valid_items.pkl', held)
    valid_code_values = [code_text(r) for r in valid]
    save_pickle(OUT/'valid_code_corpus.pkl', valid_code_values)
    base_names = list(old.BASE_FEATURES); seed_names = list(old.SEED_FEATURES)
    assert len(base_names) == 34 and len(seed_names) == 12
    dump(OUT/'names.json', dict(base_names=base_names, seed_names=seed_names, names46=base_names+seed_names,
        x_dimension=34, seed_x_dimension=12, cls_projection_dimension=32))
    projection = np.random.default_rng(PROJECTION_SEED).normal(0, 1/np.sqrt(32), (768, 32)).astype('float32')
    np.save(OUT/'cls_projection.npy', projection)
    sources = [Path(__file__), ROOT/'run_t5_seed_uq_pilot.py', ROOT/'vac_mqr.py',
        ROOT/'build_rewrite_utility_dataset.py', ROOT/'build_crsu_pilot.py',
        ROOT/'build_enhanced_retrieval_feedback_dataset.py', ROOT/'crsu/core.py',
        ROOT/'prepare_vac_mqr_cv_folds.py', ROOT/'run_bm25_acquisition_pilot.py']
    inputs = [TRAIN, VALID, VALID_ITEMS, VALID_ROWS, VALID_CODES,
        ROOT/'results/uq_pilot_20261011/t5_seed_consistency/query_vectors.npy',
        ROOT/'results/uq_pilot_20261011/t5_seed_consistency/candidate_vectors.npy',
        OUT/'action_code_corpus.pkl', OUT/'semantic_code_corpus.pkl', OUT/'action_rows.pkl',
        OUT/'semantic_small_ids.pkl', OUT/'valid_items.pkl', OUT/'valid_code_corpus.pkl',
        OUT/'names.json', OUT/'cls_projection.npy']
    assets = [p for model in (T5, RETRIEVER, CE_BASE) for p in model.iterdir()
              if p.is_file() and p.suffix in ('.safetensors', '.json', '.txt', '.model')]
    p = dict(frozen_utc=datetime.now(timezone.utc).isoformat(), test_used=False, exploratory=True,
        source_hashes={str(p): sha(p) for p in sources}, input_hashes={str(p): sha(p) for p in inputs},
        model_hashes={str(p): sha(p) for p in assets}, stable_hash=sha(STABLE), current_hash=sha(CURRENT),
        eligible_rows=eligible, action_corpus_size=CORPUS, semantic_corpus_size=CORPUS,
        action_queries=len(actions), action_query_ids=[r['query_index'] for r in actions],
        action_query_urls=[r['url'] for r in actions], action_query_repos=sorted({r['repo'] for r in actions}),
        action_corpus_repos=sorted(action_repos), semantic_query_ids=list(range(CORPUS)),
        semantic_query_urls=[r['url'] for r in pools['semantic']], semantic_repos=sorted(semantic_repos),
        semantic_small_ids=small_ids, all_valid_excluded_repos=sorted(excluded),
        repository_split='sha256(action-witness/repo) parity; independent reservoir4096/group; action512 round-robin repositories, sorted localdoc position',
        target_ids='action query_index is actual action corpus doc position; source_idx is separate; semantic local IDs have separate namespace',
        code='Only code/function_tokens, token name-obfuscation copied from old CE; synthetic train augmentation differs from exact AdvTest anonymization',
        generation=dict(seeds=list(SEEDS), chunk_queries=8, batch=32, all_whitespace_boundaries=True,
            do_sample=True, top_k=50, top_p=1., temperature=1., num_beams=1, max_input_tokens=512,
            max_new_tokens=12, precision='bfloat16', quality_before_top3=True, seed_schedule='seed+chunk_start'),
        retrieval=dict(model=str(RETRIEVER), corpus='separate4096 action code-only corpus, not valid9604 corpus',
            max_query_tokens=256, max_code_tokens=256, batch=48, stable_score_docid_ties=True,
            labels='actual replacement Top100 RR and equal-weight original+rewrite RRF(offset60) RR; target training labels only'),
        CE=dict(arms=list(ARMS), initialization=str(CE_BASE), seed=SEED,
            epochs=1, query_counts=[1024,4096], pair_counts=[4096,16384],
            positive='benchmark matched code; non-target hard negatives not proven truly irrelevant',
            hard_negatives='2 BM25 or actual checkpoint3000 dense Top100 non-target within semantic4096 + random1',
            same_small_query_ids=True, same_model_initialization=True, lr=2e-5, weight_decay=.01,
            microbatch=8, gradient_accumulation=4, effective_batch=32, query_tokens=64, code_tokens=160,
            no_valid_early_stopping=True, action_and_all_valid_repository_disjoint=True),
        witnesses=dict(candidate_budget='score all predefined actual candidate witnesses once; CPU selects shortlist and globally limits query verification budget',
            selection='Top100 signed RR displacement:4 largest positive+4 largest negative by magnitude, plus originalTop2/actionTop2; query union; no gold or outcome enters selection',
            max_per_action_predefined=12, score='CE logit plus fixed CLS randomprojection32, uncalibrated',
            projection_seed=PROJECTION_SEED, namespace='split(action|valid),query_index,doc_id'),
        no_hyperparameter_search=True, no_case_rules=True, STABLE_and_CURRENT_not_changed=True)
    dump(path, p)
    for source in sources:
        dest = OUT/'frozen_source'/source.relative_to(ROOT); dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(source.read_bytes())
    status('prepared', action_repos=len(p['action_query_repos']), semantic_repos=len(semantic_repos))
    print(f'Frozen action512/{len(p["action_query_repos"])}repo, semantic4096/{len(semantic_repos)}repo; disjoint allvalid', flush=True)
    return p


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def dense_inputs(protocol):
    paths = [OUT/'action_code_vectors.npy', OUT/'semantic_code_vectors.npy', OUT/'semantic_query_vectors.npy']
    if all(p.exists() for p in paths): return
    start = time.perf_counter(); model = AutoModel.from_pretrained(RETRIEVER).to('cuda').eval()
    tokenizer = AutoTokenizer.from_pretrained(RETRIEVER)
    for group in ('semantic','action'):
        corpus = load_pickle(OUT/f'{group}_code_corpus.pkl')
        kinds = [('code',[r['code_text'] for r in corpus])]
        if group == 'semantic': kinds.append(('query',[r['query'] for r in corpus]))
        for kind, texts in kinds:
            path = OUT/f'{group}_{kind}_vectors.npy'
            if path.exists(): continue
            values = encode(model, tokenizer, texts, 48, 256, 'cuda'); np.save(path, values)
    del model; torch.cuda.empty_cache()
    status('dense_corpora_ready', elapsed_seconds=time.perf_counter()-start)


def semantic_pairs(protocol):
    corpus = load_pickle(OUT/'semantic_code_corpus.pkl'); small = load_pickle(OUT/'semantic_small_ids.pkl')
    index = BM25([r['code_text'] for r in corpus])
    qvectors = np.load(OUT/'semantic_query_vectors.npy'); codes = np.load(OUT/'semantic_code_vectors.npy')
    for arm in ARMS:
        path = OUT/f'ce_pairs_{arm}.pkl'
        if path.exists(): continue
        ids = small if arm.startswith('small') else list(range(CORPUS))
        pairs = []
        for qi in ids:
            if '_bm25_' in arm: hard, _ = index.retrieve(corpus[qi]['query'])
            else: hard = stable_indices(codes@qvectors[qi], 100)
            negative = [int(i) for i in hard if int(i) != qi][:2]
            assert len(negative) == 2
            rng = np.random.default_rng(SEED+qi*71+5); ri = int(rng.integers(CORPUS))
            while ri == qi or ri in negative: ri = int(rng.integers(CORPUS))
            negative.append(ri)
            pairs.append((corpus[qi]['query'], corpus[qi]['code_text'], 1, qi, qi))
            pairs.extend((corpus[qi]['query'], corpus[d]['code_text'], 0, qi, d) for d in negative)
        save_pickle(path, pairs)
        dump(OUT/f'ce_pairs_{arm}.json', dict(queries=len(ids), pairs=len(pairs), query_ids=ids,
            corpus_hash=sha(OUT/'semantic_code_corpus.pkl'), pair_hash=sha(path),
            hard_source='BM25' if '_bm25_' in arm else 'actual frozen checkpoint3000 Top100',
            labels='matched benchmark target vs non-target, not proven semantic irrelevance'))


class CE(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.encoder = AutoModel.from_pretrained(CE_BASE)
        self.drop = torch.nn.Dropout(.1); self.head = torch.nn.Linear(self.encoder.config.hidden_size,1)
    def forward(self, return_cls=False, **kw):
        cls = self.encoder(**kw).last_hidden_state[:,0]
        logits = self.head(self.drop(cls)).squeeze(-1)
        return (logits,cls) if return_cls else logits


def ce_tokens(tokenizer, queries, codes):
    qa = tokenizer(queries,add_special_tokens=False,truncation=True,max_length=64)['input_ids']
    ca = tokenizer(codes,add_special_tokens=False,truncation=True,max_length=160)['input_ids']
    ids = [tokenizer.build_inputs_with_special_tokens(q,c) for q,c in zip(qa,ca)]
    return tokenizer.pad({'input_ids':ids},padding=True,return_tensors='pt').to('cuda')


def train_ce(protocol, arm):
    path = OUT/f'ce_{arm}.pt'
    if path.exists(): return
    start = time.perf_counter(); seed_all(SEED)
    pairs = load_pickle(OUT/f'ce_pairs_{arm}.pkl'); tokenizer = AutoTokenizer.from_pretrained(CE_BASE)
    model = CE().to('cuda').train(); optimizer = torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=.01)
    order = np.random.default_rng(SEED).permutation(len(pairs)); losses=[]; optimizer.zero_grad(set_to_none=True)
    for step,begin in enumerate(range(0,len(order),8)):
        batch=[pairs[int(i)] for i in order[begin:begin+8]]
        tokens=ce_tokens(tokenizer,[r[0] for r in batch],[r[1] for r in batch])
        y=torch.tensor([r[2] for r in batch],device='cuda',dtype=torch.float32)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits=model(**tokens); loss=torch.nn.functional.binary_cross_entropy_with_logits(logits.float(),y)
        if not torch.isfinite(loss): raise FloatingPointError('CE loss')
        (loss/4).backward(); losses.append(float(loss.detach()))
        if (step+1)%4==0:
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm): raise FloatingPointError('CE gradient')
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
        if (step+1)%256==0: print(f'CE {arm} train{step+1}/{len(order)//8} loss{np.mean(losses[-256:]):.4f}',flush=True)
    torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},path)
    dump(OUT/f'ce_{arm}_train_status.json',dict(completed=True,arm=arm,steps=len(order)//8,updates=len(order)//32,
        elapsed_seconds=time.perf_counter()-start,loss_blocks=[float(np.mean(losses[i:i+128])) for i in range(0,len(losses),128)],
        checkpoint_hash=sha(path), pairs_hash=sha(OUT/f'ce_pairs_{arm}.pkl'), test_used=False))
    del model,optimizer; torch.cuda.empty_cache()
    print(f'CE {arm} done {time.perf_counter()-start:.1f}s',flush=True)


def generate_chunk(model, tokenizer, current, chunk_start, seeds=SEEDS):
    """Reusable frozen-model helper; rows need query/query_index/url only."""
    validator = CandidateQualityValidator(); records=[]
    for local,row in enumerate(current):
        tokens=row['query'].split()
        for position in range(len(tokens)+1):
            records.append((local,position,' '.join(tokens[:position]+['<extra_id_0>']+tokens[position:])))
    output=[dict(query_index=r['query_index'],query=r['query'],url=r['url'],repo=r['repo'],draws=[]) for r in current]
    excluded={tokenizer.pad_token_id,tokenizer.eos_token_id}
    for seed in seeds:
        torch.manual_seed(seed+chunk_start); torch.cuda.manual_seed_all(seed+chunk_start)
        candidates=[[] for _ in current]
        for start in range(0,len(records),32):
            batch=records[start:start+32]
            encoded=tokenizer([r[2] for r in batch],padding=True,truncation=True,max_length=512,return_tensors='pt').to('cuda')
            with torch.inference_mode():
                generated=model.generate(**encoded,do_sample=True,top_k=50,top_p=1.,temperature=1.,num_beams=1,
                    num_return_sequences=1,max_new_tokens=12,return_dict_in_generate=True,output_scores=True)
            decoded=tokenizer.batch_decode(generated.sequences,skip_special_tokens=False); ids=generated.sequences[:,1:]
            terms=[[] for _ in batch]
            for step,logits in enumerate(generated.scores):
                probabilities=torch.softmax(logits.float(),-1)
                ent=(probabilities*torch.log(probabilities.clamp_min(1e-12))).sum(-1).cpu().tolist()
                for j,token in enumerate(ids[:,step].tolist()):
                    if token not in excluded: terms[j].append(ent[j])
            for j,(local,position,_) in enumerate(batch):
                match=re.search(r'<extra_id_0>\s*(.*?)\s*(?:<extra_id_1>|</s>|$)',decoded[j])
                span=re.sub(r'<[^>]+>','',match.group(1)).strip() if match else ''
                if not span or not terms[j]: continue
                tokens=current[local]['query'].split(); text=' '.join(tokens[:position]+[span]+tokens[position:])
                if text==current[local]['query']: continue
                ig=float(np.mean(terms[j])); assert math.isfinite(ig)
                candidates[local].append(dict(text=text,position=position,span=span,information_gain=ig,seed=seed))
        for local,raw in enumerate(candidates):
            raw.sort(key=lambda c:-c['information_gain']); seen=set(); accepted=[]; reasons=Counter()
            for c in raw:
                quality=validator.validate(current[local]['query'],c['text'],seen)
                if not quality.accepted: reasons.update(quality.reasons); continue
                accepted.append(dict(c,quality_pass=True,constraint_preservation=quality.constraint_preservation,rank=len(accepted)+1))
            output[local]['draws'].append(dict(seed=seed,candidates=accepted[:3],raw_count=len(raw),quality_pass_count=len(accepted),
                quality_reasons=dict(reasons),evaluated_positions=len(current[local]['query'].split())+1))
    return output


def generate_actions(protocol):
    rows=load_pickle(OUT/'action_rows.pkl'); directory=OUT/'generation_chunks'; directory.mkdir(exist_ok=True)
    if (OUT/'generation.jsonl').exists(): return
    seed_all(SEED); tokenizer=AutoTokenizer.from_pretrained(T5)
    model=AutoModelForSeq2SeqLM.from_pretrained(T5,torch_dtype=torch.bfloat16).to('cuda').eval(); start=time.perf_counter()
    for begin in range(0,len(rows),8):
        path=directory/f'{begin:04d}.pkl'
        if path.exists(): continue
        records=generate_chunk(model,tokenizer,rows[begin:begin+8],begin)
        save_pickle(path,records); status('generation',completed=begin+len(records),total=len(rows),elapsed_session_seconds=time.perf_counter()-start)
        print(f'T5 action {begin+len(records)}/{len(rows)} {time.perf_counter()-start:.1f}s',flush=True)
    generated=sum([load_pickle(directory/f'{begin:04d}.pkl') for begin in range(0,len(rows),8)],[])
    assert [r['query_index'] for r in generated]==[r['query_index'] for r in rows]
    with (OUT/'generation.jsonl').open('w',encoding='utf8') as stream:
        for row in generated: stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    del model;torch.cuda.empty_cache();status('generation_complete',queries=len(rows),elapsed_seconds=time.perf_counter()-start)


def retrieve_actions(protocol):
    if (OUT/'action_items.pkl').exists(): return
    rows=read(OUT/'generation.jsonl'); corpus=load_pickle(OUT/'action_code_corpus.pkl')
    candidates=[]; appearances=[];texts=[]
    for row in rows:
        lookup={}; unique=[]; seeds=[]
        for draw in row['draws']:
            for c in draw['candidates']:
                key=CandidateQualityValidator.canonical(c['text'])
                if key not in lookup: lookup[key]=len(unique);unique.append(dict(c));seeds.append(set())
                seeds[lookup[key]].add(draw['seed'])
        candidates.append(unique);appearances.append(seeds);texts.extend(c['text'] for c in unique)
    model=AutoModel.from_pretrained(RETRIEVER).to('cuda').eval();tokenizer=AutoTokenizer.from_pretrained(RETRIEVER)
    qvectors=encode(model,tokenizer,[r['query'] for r in rows],48,256,'cuda')
    cvectors=encode(model,tokenizer,texts,48,256,'cuda')
    del model;torch.cuda.empty_cache();np.save(OUT/'action_query_vectors.npy',qvectors);np.save(OUT/'action_candidate_vectors.npy',cvectors)
    codes=np.load(OUT/'action_code_vectors.npy');items=[];offset=0;label_checks=0
    for row,cs,apps,qv in zip(rows,candidates,appearances,qvectors):
        qi=row['query_index'];assert corpus[qi]['url']==row['url']
        original=codes@qv;otop=stable_indices(original,100);rank=np.flatnonzero(otop==qi);rr0=1/(int(rank[0])+1) if len(rank) else 0.
        n=len(cs);vectors=cvectors[offset:offset+n];offset+=n;scores=vectors@codes.T;tops=[stable_indices(s,100) for s in scores]
        base_x=np.asarray([old.base_features(row,c,qv,cv,original,s,otop,top) for c,cv,s,top in zip(cs,vectors,scores,tops)]).reshape(n,len(old.BASE_FEATURES))
        seed_x=old.sampling_features(row,vectors,tops,apps); yy={}
        for family in ('rewrite','fusion'):
            actions={str(j):top if family=='rewrite' else rrf(otop,top) for j,top in enumerate(tops)}
            states=RankingStates.from_rankings(otop,actions,100);y=states.gains[states.target_index(qi)]
            yy[family]=y.copy(); label_checks+=len(y)
        items.append(dict(query_index=qi,source_idx=corpus[qi]['source_idx'],query=row['query'],url=row['url'],repo=row['repo'],
            role='independent_action_training',x=base_x,seed_x=seed_x,x46=np.c_[base_x,seed_x],original_rr=rr0,y=yy,
            candidates=cs,seed_appearances=[sorted(s) for s in apps],original_top100=otop,candidate_top100=tops,
            pools=dict(seed101=np.array([101 in s for s in apps]),multi8=np.ones(n,bool))))
    assert offset==len(cvectors);save_pickle(OUT/'action_items.pkl',items)
    # Copy the frozen historical valid vectors, never regenerate a ranking or
    # gold-dependent score during a later selector comparison.
    oldq=np.load(ROOT/'results/uq_pilot_20261011/t5_seed_consistency/query_vectors.npy')
    oldc=np.load(ROOT/'results/uq_pilot_20261011/t5_seed_consistency/candidate_vectors.npy')
    np.save(OUT/'valid_query_vectors.npy',oldq);np.save(OUT/'valid_candidate_vectors.npy',oldc)
    dump(OUT/'action_retrieval_status.json',dict(queries=len(items),candidates=len(cvectors),actual_label_checks=label_checks,
        original_mrr=float(np.mean([e['original_rr'] for e in items])),test_used=False,
        positive_pool_queries={f:sum(bool(np.any(e['y'][f]>1e-10)) for e in items) for f in ('rewrite','fusion')}))
    print(f'Action retrieval {len(items)}query/{len(cvectors)}candidate labelchecks{label_checks}',flush=True)


def action_witness(original, action):
    """Gold-free: rank displacement using all observable Top100 positions."""
    orank={int(d):i+1 for i,d in enumerate(original)}; arank={int(d):i+1 for i,d in enumerate(action)}
    union=sorted(set(orank)|set(arank));shift={d:(1/arank[d] if d in arank else 0)-(1/orank[d] if d in orank else 0) for d in union}
    positive=sorted([d for d in union if shift[d]>1e-12],key=lambda d:(-shift[d],d))[:4]
    negative=sorted([d for d in union if shift[d]<-1e-12],key=lambda d:(shift[d],d))[:4]
    ids=sorted(set(positive+negative+list(map(int,original[:2]))+list(map(int,action[:2]))))
    return dict(doc_ids=np.array(ids,int),displacement=np.array([shift[d] for d in ids]),
        original_rank=np.array([orank.get(d,0) for d in ids]),action_rank=np.array([arank.get(d,0) for d in ids]))


def build_witnesses(split):
    path=OUT/f'witness_map_{split}.pkl'
    if path.exists(): return load_pickle(path)
    items=load_pickle(OUT/f'{split}_items.pkl');out=[]
    for e in items:
        actions={};union=set(map(int,e['original_top100'][:2]))
        for family in ('rewrite','fusion'):
            values=[]
            for j,top in enumerate(e['candidate_top100']):
                actual=top if family=='rewrite' else rrf(e['original_top100'],top)
                witness=action_witness(e['original_top100'],actual);union.update(map(int,witness['doc_ids']))
                values.append(dict(candidate_index=j,**witness))
            actions[family]=values
        out.append(dict(query_index=e['query_index'],original_top2=np.asarray(e['original_top100'][:2]),
            actions=actions,doc_ids=np.array(sorted(union),int)))
    save_pickle(path,out);return out


def score_ce(protocol, arm, split):
    path=OUT/f'ce_{arm}_scores_{split}.pkl'
    if path.exists(): return
    items=load_pickle(OUT/f'{split}_items.pkl');witnesses=build_witnesses(split)
    if split=='action': codes=[r['code_text'] for r in load_pickle(OUT/'action_code_corpus.pkl')]
    else: codes=load_pickle(OUT/'valid_code_corpus.pkl')
    queries=[];documents=[];lengths=[]
    for e,w in zip(items,witnesses):
        assert e['query_index']==w['query_index'];lengths.append(len(w['doc_ids']))
        queries.extend([e['query']]*len(w['doc_ids']));documents.extend(codes[int(d)] for d in w['doc_ids'])
    tokenizer=AutoTokenizer.from_pretrained(CE_BASE);seed_all(SEED);model=CE().to('cuda').eval()
    model.load_state_dict(torch.load(OUT/f'ce_{arm}.pt',map_location='cpu',weights_only=True))
    projection=torch.tensor(np.load(OUT/'cls_projection.npy'),device='cuda');logits=[];embeddings=[];start=time.perf_counter()
    with torch.inference_mode():
        for begin in range(0,len(queries),32):
            tokens=ce_tokens(tokenizer,queries[begin:begin+32],documents[begin:begin+32])
            with torch.autocast('cuda',dtype=torch.bfloat16): scores,cls=model(return_cls=True,**tokens)
            logits.extend(scores.float().cpu().tolist());embeddings.append((cls.float()@projection).cpu().numpy())
            if (begin//32+1)%256==0:print(f'CE {arm} score{split} {begin+32}/{len(queries)}',flush=True)
    embeddings=np.concatenate(embeddings);offsets=np.r_[0,np.cumsum(lengths)]
    result=[dict(query_index=e['query_index'],doc_ids=w['doc_ids'].copy(),logits=np.array(logits[offsets[i]:offsets[i+1]]),
        cls_projection=embeddings[offsets[i]:offsets[i+1]].copy()) for i,(e,w) in enumerate(zip(items,witnesses))]
    assert all(np.isfinite(z['logits']).all() and np.isfinite(z['cls_projection']).all() for z in result)
    save_pickle(path,result);dump(OUT/f'ce_{arm}_scores_{split}.json',dict(completed=True,queries=len(items),unique_query_doc_pairs=len(queries),
        elapsed_seconds=time.perf_counter()-start,labels_used=False,checkpoint_hash=sha(OUT/f'ce_{arm}.pt'),
        score_hash=sha(path),witness_hash=sha(OUT/f'witness_map_{split}.pkl'),projection_hash=sha(OUT/'cls_projection.npy')))
    del model;torch.cuda.empty_cache();print(f'CE {arm} scored{split} {len(queries)}pair {time.perf_counter()-start:.1f}s',flush=True)


def final_manifest(protocol):
    names=['GPU_PROTOCOL.json','names.json','action_code_corpus.pkl','semantic_code_corpus.pkl','action_rows.pkl',
        'semantic_small_ids.pkl','valid_items.pkl','action_items.pkl','generation.jsonl','cls_projection.npy',
        'action_query_vectors.npy','action_candidate_vectors.npy','valid_query_vectors.npy','valid_candidate_vectors.npy',
        'witness_map_action.pkl','witness_map_valid.pkl','action_retrieval_status.json']
    names.extend(f'ce_{arm}.pt' for arm in ARMS)
    names.extend(f'ce_{arm}_scores_{split}.pkl' for arm in ARMS for split in ('action','valid'))
    assert all((OUT/n).exists() for n in names)
    assert sha(STABLE)==protocol['stable_hash'] and sha(CURRENT)==protocol['current_hash']
    dump(OUT/'FINAL_MANIFEST.json',dict(completed=True,test_used=False,stable_unchanged=True,current_unchanged=True,
        outputs={str(OUT/n):sha(OUT/n) for n in names}))
    status('complete',arms=list(ARMS),action_queries=ACTION_QUERIES)


def main():
    sys.stdout.reconfigure(encoding='utf8');torch.set_num_threads(2)
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('prepare','semantic','generation','retrieval','score_action','all'),default='all')
    args=parser.parse_args();protocol=prepare()
    if args.stage=='prepare':return
    with threadpool_limits(limits=2):
        if args.stage in ('semantic','all'):
            dense_inputs(protocol);semantic_pairs(protocol);build_witnesses('valid')
            for arm in ARMS:train_ce(protocol,arm);score_ce(protocol,arm,'valid')
        if args.stage in ('generation','all'):generate_actions(protocol)
        if args.stage in ('retrieval','all'):dense_inputs(protocol);retrieve_actions(protocol)
        if args.stage in ('score_action','all'):
            build_witnesses('action')
            for arm in ARMS:score_ce(protocol,arm,'action')
        if args.stage=='all':final_manifest(protocol)


if __name__=='__main__':main()
