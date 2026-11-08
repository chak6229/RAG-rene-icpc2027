"""Auxiliary full-training-bank truncation exposure, without retrieval labels."""
import json
import os
import random
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path

os.environ.setdefault('TOKENIZERS_PARALLELISM','false')


def main():
    from transformers import AutoTokenizer
    root=Path(__file__).resolve().parents[1]
    output=root/'results/rene_cqc_training_fidelity_20261020/FULL_BANK_LENGTH_DIAGNOSTIC.json'
    if output.exists():
        print('Full-bank diagnostic already exists; preserve original.');return
    tokenizer=AutoTokenizer.from_pretrained(root/'models/t5-base',local_files_only=True)
    raw=(root/'data/codenn_cqc/train.txt').read_text(encoding='utf-8').splitlines()
    rng=random.Random(101);source_hist=Counter();target_hist=Counter()
    for offset in range(0,len(raw),8192):
        pairs=[]
        for query in raw[offset:offset+8192]:
            words=query.split();length=max(1,round(.15*len(words)));start=rng.randrange(len(words)-length+1)
            pairs.append((' '.join(words[:start]+['<extra_id_0>']+words[start+length:]),
                '<extra_id_0> '+' '.join(words[start:start+length])+' <extra_id_1>'))
        sources=tokenizer([a for a,b in pairs],truncation=False)['input_ids']
        targets=tokenizer([b for a,b in pairs],truncation=False)['input_ids']
        source_hist.update(map(len,sources));target_hist.update(map(len,targets))
    value=dict(created_utc=datetime.now(timezone.utc).isoformat(),training_rows=len(raw),
        purpose='Auxiliary truncation exposure only, not a new training arm or outcome evaluation',mask_seed=101,
        masks='one fixed mask per row in source order, not all historical shuffled epoch masks',
        source_max=max(source_hist),target_max=max(target_hist),
        source_gt128=sum(n for l,n in source_hist.items() if l>128),target_gt128=sum(n for l,n in target_hist.items() if l>128),
        source_gt512=sum(n for l,n in source_hist.items() if l>512),target_gt512=sum(n for l,n in target_hist.items() if l>512),
        source_length_histogram=dict(sorted(source_hist.items())),target_length_histogram=dict(sorted(target_hist.items())),
        retrieval_outcomes_read=False)
    with output.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2)


if __name__=='__main__':main()
