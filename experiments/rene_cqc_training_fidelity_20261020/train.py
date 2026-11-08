from __future__ import annotations

import gc
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
from .common import ROOT, OUT, BASE, STORE, ARM_SPECS, lr_at, now, rows, write, record, sha, require, verify


def optimizer_for(model, spec):
    import torch
    from transformers import Adafactor
    if spec['optimizer'] == 'AdamW':
        return torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=spec['weight_decay'])
    if spec['optimizer'] == 'Adafactor':
        return Adafactor(model.parameters(), lr=.001, weight_decay=spec['weight_decay'], scale_parameter=False, relative_step=False)
    raise ValueError(spec['optimizer'])


def encode(tokenizer, batch, limit, device='cuda'):
    inputs = tokenizer([r['source'] for r in batch], padding=True, truncation=True, max_length=limit, return_tensors='pt').to(device)
    labels = tokenizer([r['target'] for r in batch], padding=True, truncation=True, max_length=limit, return_tensors='pt').input_ids.to(device)
    labels[labels == tokenizer.pad_token_id] = -100
    return inputs, labels


def validate(model, tokenizer, bank):
    import torch
    model.eval()
    total=0.;tokens=0
    with torch.inference_mode():
        for offset in range(0,len(bank),8):
            inputs,labels=encode(tokenizer,bank[offset:offset+8],512)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                loss=model(**inputs,labels=labels).loss
            n=int(labels.ne(-100).sum())
            total+=float(loss)*n;tokens+=n
    model.train()
    return dict(token_nll=total/tokens, tokens=tokens, examples=len(bank), cap=512)


def train_arm(name):
    p=verify(); spec=p['arms'][name]
    destination=STORE/name
    completed=OUT/f'training/{name}.json'
    if completed.exists():
        result=json.loads(completed.read_text(encoding='utf-8'))
        require(result['steps']==1024 and sha(destination/'model.safetensors')==result['weights_sha256'], 'Existing completed weights')
        return result
    import torch
    import numpy as np
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True, warn_only=True)
    tokenizer=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
    bank=rows(OUT/'data/train.jsonl'); validation=rows(OUT/'data/validation.jsonl')
    model=AutoModelForSeq2SeqLM.from_pretrained(BASE,local_files_only=True).float().cuda().train()
    model.config.use_cache=False
    random.seed(spec['seed']);np.random.seed(spec['seed']);torch.manual_seed(spec['seed']);torch.cuda.manual_seed_all(spec['seed'])
    optimizer=optimizer_for(model,spec)
    start=0;validation_records=[]; resumed=False
    latest=destination/'resume.pt'
    if latest.exists():
        state=torch.load(latest,map_location='cpu',weights_only=False)
        require(state['protocol_sha256']==sha(OUT/'TRAIN_PROTOCOL.json'), 'Resume protocol')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        for values in optimizer.state.values():
            for key,value in values.items():
                if isinstance(value,torch.Tensor):values[key]=value.cuda()
        start=state['step'];validation_records=state['validation_records']
        torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng']); resumed=True
        del state
    destination.mkdir(parents=True,exist_ok=True)
    print(json.dumps(dict(stage='train_start',arm=name,start_step=start,spec=spec)),flush=True)
    if start==0:
        # Keep validation from consuming the training-dropout stream.
        validation_records.append(dict(step=0,**validate(model,tokenizer,validation)))
    started=time.perf_counter(); recent=[]; step_rows=[]
    logpath=OUT/f'training/{name}_steps.jsonl';logpath.parent.mkdir(parents=True,exist_ok=True)
    with logpath.open('a' if resumed else 'x',encoding='utf-8') as log:
        for step in range(start,1024):
            tick=time.perf_counter(); optimizer.zero_grad(set_to_none=True)
            current_lr=lr_at(spec,step)
            for group in optimizer.param_groups: group['lr']=current_lr
            micro_losses=[]
            for micro in range(4):
                offset=step*32+micro*8
                inputs,labels=encode(tokenizer,bank[offset:offset+8],spec['max_length'])
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    loss=model(**inputs,labels=labels).loss
                require(bool(torch.isfinite(loss)), 'Nonfinite training loss')
                (loss/4).backward();micro_losses.append(float(loss.detach()))
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            require(bool(torch.isfinite(norm)), 'Nonfinite gradients')
            optimizer.step(); torch.cuda.synchronize()
            row=dict(step=step+1,bank_start=step*32,bank_stop=(step+1)*32,learning_rate=current_lr,
                mean_microbatch_loss=sum(micro_losses)/4,gradient_norm=float(norm),seconds=time.perf_counter()-tick)
            log.write(json.dumps(row)+'\n');log.flush();recent.append(row['mean_microbatch_loss'])
            if (step+1)%64==0:
                print(json.dumps(dict(stage='training',arm=name,step=step+1,total=1024,loss64=sum(recent[-64:])/64,
                    lr_used=current_lr,seconds=round(time.perf_counter()-started,1))),flush=True)
                (OUT/'RUN_STATUS.json').write_text(json.dumps(dict(stage='training',arm=name,step=step+1,total=1024,
                    updated_utc=now(),completed=False)),encoding='utf-8')
            if (step+1)%256==0:
                if step+1==1024: validation_records.append(dict(step=1024,**validate(model,tokenizer,validation)))
                temporary=destination/'resume.tmp.pt'
                torch.save(dict(step=step+1,protocol_sha256=sha(OUT/'TRAIN_PROTOCOL.json'),model=model.state_dict(),
                    optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                    validation_records=validation_records),temporary)
                temporary.replace(latest)
    model.config.use_cache=True
    model.save_pretrained(destination,safe_serialization=True);tokenizer.save_pretrained(destination)
    report=dict(created_utc=now(),arm=name,spec=spec,steps=1024,examples_seen=32768,seed=spec['seed'],
        checkpoint_selection='fixed final update',validation=validation_records,seconds_this_process=time.perf_counter()-started,
        resumed=resumed,training_bank_sha256=sha(OUT/'data/train.jsonl'),weights_sha256=sha(destination/'model.safetensors'),
        files=[record(x) for x in sorted(destination.iterdir()) if x.is_file() and x.name!='resume.pt'],
        train_steps_file=record(logpath),no_retrieval_labels_read=True,protocol_sha256=sha(OUT/'TRAIN_PROTOCOL.json'))
    write(completed,report)
    # Recovery checkpoint is redundant after immutable final weights/receipt; only this dedicated file is removed.
    latest.unlink()
    del model,optimizer;gc.collect();torch.cuda.empty_cache()
    print(json.dumps(dict(stage='train_complete',arm=name,validation=validation_records[-1],seconds=round(report['seconds_this_process'],1))),flush=True)
    return report


def main():
    for name in ARM_SPECS:
        train_arm(name)
    verify()
    write(OUT/'TRAINING_COMPLETE.json',dict(created_utc=now(),arms=list(ARM_SPECS),all_updates=1024,
        files=[record(OUT/f'training/{a}.json') for a in ARM_SPECS], no_outcome_evaluation=True))


if __name__=='__main__':main()
