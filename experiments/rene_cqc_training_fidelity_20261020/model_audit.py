"""Independent actual-model checks for new training configurations."""
from __future__ import annotations

import gc
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel,AutoModelForSeq2SeqLM,AutoTokenizer,T5Tokenizer

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/rene_cqc_training_fidelity_20261020'
STORE=Path('external_inputs/cqc_training_model_store')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def rows(path):
    with Path(path).open(encoding='utf-8-sig') as h:return [json.loads(line) for line in h if line.strip()]


def require(ok,message):
    if not ok:raise AssertionError(message)


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    p=read(OUT/'TRAIN_PROTOCOL.json');cases=[];cbcases=[]
    tokenizer=T5Tokenizer(vocab_file=str(ROOT/'models/t5-base/spiece.model'),extra_ids=100,legacy=True)
    for arm in p['arms']:
        model=AutoModelForSeq2SeqLM.from_pretrained(STORE/arm,local_files_only=True).float().cuda().eval()
        for domain in ('source','cosqa'):
            for view in ('own','fixed'):
                bank=rows(OUT/f'candidates/{arm}/{domain}_{view}.jsonl')
                qids=p['samples'][domain]['query_ids'][:1]
                qid=qids[0];subset=[r for r in bank if r['query_id']==qid][:8]
                sources=[r['masked_query'] for r in subset]
                labels=tokenizer([f"<extra_id_0> {r['span']} <extra_id_1>" for r in subset],padding=True,return_tensors='pt')
                inputs=tokenizer(sources,padding=True,return_tensors='pt').to('cuda')
                raw=labels.input_ids.cuda();valid=labels.attention_mask.cuda().bool();masked=raw.masked_fill(~valid,-100)
                with torch.inference_mode():
                    logits=model(**inputs,labels=masked,decoder_attention_mask=valid).logits.double()
                    logz=torch.logsumexp(logits,-1,keepdim=True)
                    probs=torch.exp(logits-logz)
                    entropy=(probs*(logz-logits)).sum(-1)
                    logp=(logits-logz).gather(-1,raw.unsqueeze(-1)).squeeze(-1)
                    h=(entropy*valid).sum(1)/valid.sum(1);lp=(logp*valid).sum(1)/valid.sum(1)
                    if view=='own':
                        generated=model.generate(**inputs,do_sample=False,num_beams=1,num_return_sequences=1,max_new_tokens=19,use_cache=True,
                            decoder_start_token_id=0,eos_token_id=1,pad_token_id=0)
                for i,r in enumerate(subset):
                    herror=abs(float(h[i])-r['full_vocab_entropy']);perror=abs(float(lp[i])-r['label_mean_log_probability'])
                    require(herror<3e-6 and perror<3e-6,'Independent FP64 teacher-forcing scores')
                    if view=='own':
                        decoded=tokenizer.decode(generated[i],skip_special_tokens=True,clean_up_tokenization_spaces=True).strip()
                        require(decoded==r['span'],'Actual greedy regeneration differs')
                    cases.append(dict(arm=arm,domain=domain,view=view,query_id=qid,position=r['position'],entropy_abs_error=herror,logprob_abs_error=perror,
                        actual_regeneration_checked=view=='own'))
        del model;gc.collect();torch.cuda.empty_cache()
        print(dict(stage='actual_t5_audit',arm=arm,cases=len(cases)),flush=True)
    checkpoint=ROOT/'models/codebert-ssqr-pooler-2epoch-20261019'
    model=AutoModel.from_pretrained(checkpoint,local_files_only=True).float().cuda().eval()
    tokenizer=AutoTokenizer.from_pretrained(checkpoint,local_files_only=True)
    for domain in ('source','cosqa'):
        texts=read(OUT/f'rankings/{domain}/new_texts.json');cached=np.load(OUT/f'rankings/{domain}/new_vectors.npy')
        # Same declared historical batch boundaries for reproducible FP32 comparisons.
        for start in [0,max(0,(len(texts)//2//16)*16)]:
            selected=texts[start:start+16]
            if not selected:continue
            encoded=tokenizer(selected,padding='max_length',max_length=256,truncation=True,return_tensors='pt').to('cuda')
            with torch.inference_mode():actual=model(**encoded).pooler_output.float().cpu().numpy()
            error=float(np.max(np.abs(actual-cached[start:start+len(selected)])))
            require(error<5e-6,'Independent CodeBERT forward vectors')
            cbcases.append(dict(domain=domain,start=start,texts=len(selected),max_abs_vector_error=error))
    result=dict(passed=True,created_utc=datetime.now(timezone.utc).isoformat(),independent_no_production_imports=True,
        T5_cases=cases,CodeBERT_cases=cbcases,max_entropy_abs_error=max(c['entropy_abs_error'] for c in cases),
        max_logprob_abs_error=max(c['logprob_abs_error'] for c in cases),new_confirmation=False)
    with (OUT/'MODEL_AUDIT.json').open('x',encoding='utf-8') as h:json.dump(result,h,ensure_ascii=False,indent=2)
    print(dict(stage='actual_models_audit_passed',T5_cases=len(cases),CodeBERT_batches=len(cbcases)),flush=True)


if __name__=='__main__':main()
