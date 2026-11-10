"""Frozen cross-encoder training components used by the complete-H study.

The delivered study uses large_dense_4096. These functions preserve the frozen
training operations; corpus preparation requires lawful upstream data and the
documented semantic-corpus/pair manifests. No earlier selection study is included.
"""
from __future__ import annotations
import argparse, hashlib, json, pickle, random, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from experiments.rene_cosqa_pilot_20261019.ranking import BM25
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'models/ssqr-verifier'
CE_BASE=ROOT/'models/codebert-base'
SEED=141414
CORPUS=4096
ARMS=('large_dense_4096',)


def stable_indices(scores, k=100):
    scores=np.asarray(scores,float)
    return np.lexsort((np.arange(len(scores)), -scores))[:min(k,len(scores))]

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf8')


def save_pickle(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    with temp.open('wb') as stream: pickle.dump(value, stream)
    temp.replace(path)


def load_pickle(path):
    return pickle.load(Path(path).open('rb'))


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


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

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',type=Path,required=True,help='Frozen semantic corpus, vectors and pair files')
    args=parser.parse_args()
    OUT=args.data_dir.resolve()
    OUT.mkdir(parents=True,exist_ok=True)
    semantic_pairs({})
    train_ce({},'large_dense_4096')
