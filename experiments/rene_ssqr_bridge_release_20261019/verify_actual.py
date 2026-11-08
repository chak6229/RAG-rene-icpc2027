"""Portable real-weight T5 generation / FP64 teacher-forcing entropy check."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b""):h.update(block)
    return h.hexdigest()


def main():
    import torch
    from transformers import AutoModelForSeq2SeqLM,AutoTokenizer,T5Tokenizer
    parser=argparse.ArgumentParser()
    parser.add_argument("--t5",type=Path,required=True)
    parser.add_argument("--spiece",type=Path,required=True)
    parser.add_argument("--codebert",type=Path)
    parser.add_argument("--bundle",type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument("--output",type=Path,default=Path("ACTUAL_AUDIT.json"))
    args=parser.parse_args()
    root=args.bundle.resolve()
    cases=json.loads((root/"actual_cases.json").read_text(encoding="utf-8"))
    if digest(args.t5/"model.safetensors")!=cases["t5_sha256"]:raise ValueError("wrong T5 weights")
    if digest(args.spiece)!=cases["spiece_sha256"]:raise ValueError("wrong SentencePiece vocabulary")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    tokenizer=T5Tokenizer(vocab_file=str(args.spiece),extra_ids=100,legacy=True)
    reference_tokenizer=AutoTokenizer.from_pretrained(args.t5,local_files_only=True)
    if tokenizer.get_vocab()!=reference_tokenizer.get_vocab():raise ValueError("checkpoint vocabulary differs")
    model=AutoModelForSeq2SeqLM.from_pretrained(args.t5,local_files_only=True).float().cuda().eval()
    max_error=0.
    for row in cases["cases"]:
        inputs=tokenizer([row["masked_query"]],return_tensors="pt")
        inputs={key:value.cuda() for key,value in inputs.items()}
        seed=int.from_bytes(hashlib.sha256(f'101\0{row["query_id"]}\0{row["position"]}'.encode()).digest()[:8],"big")%(2**31-1)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        with torch.inference_mode():
            generated=model.generate(**inputs,do_sample=True,num_beams=1,num_return_sequences=1,max_new_tokens=9,
                temperature=1.,top_k=0,top_p=1.,use_cache=True,decoder_start_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.pad_token_id)[0]
        if generated.cpu().tolist()!=row["generated_ids"]:raise ValueError("actual sampled generation differs")
        labels=tokenizer(f'<extra_id_0> {row["span"]} <extra_id_1>',return_tensors="pt").input_ids.cuda()
        with torch.inference_mode():
            logits=model(**inputs,labels=labels).logits[0].double().cpu()
        logp=logits-torch.logsumexp(logits,dim=-1,keepdim=True)
        entropy=float(-(logp.exp()*logp).sum(-1).mean())
        max_error=max(max_error,abs(entropy-row["full_vocab_entropy"]))
    if max_error>=2e-5:raise ValueError("independent FP64 full-vocabulary entropy differs")
    result={"passed":True,"actual_cases":len(cases["cases"]),"sampled_generation_ids_identical":True,
            "fp64_entropy_max_error":max_error,"scope":"real frozen T5 inference; not full retraining"}
    if args.codebert is not None:
        import gc
        from transformers import AutoModel
        reference=cases["codebert"]
        if digest(args.codebert/"model.safetensors")!=reference["sha256"]:
            raise ValueError("wrong CodeBERT weights")
        del model
        gc.collect()
        torch.cuda.empty_cache()
        encoder=AutoModel.from_pretrained(args.codebert,local_files_only=True).float().cuda().eval()
        encoder_tokenizer=AutoTokenizer.from_pretrained(args.codebert,local_files_only=True)
        vector_error=0.
        for row in reference["cases"]:
            batch=encoder_tokenizer([row["text"]],padding="max_length",truncation=True,
                                    max_length=256,return_tensors="pt")
            with torch.inference_mode():
                vector=encoder(**{key:value.cuda() for key,value in batch.items()}).pooler_output[0].float().cpu()
            vector_error=max(vector_error,float((vector-torch.tensor(row["vector"])).abs().max()))
        if vector_error>=5e-5:raise ValueError("independent actual CodeBERT vectors differ")
        result["codebert"]={"passed":True,"actual_cases":len(reference["cases"]),
                            "pooler_vector_max_error":vector_error,
                            "scope":"eight query and six code forwards; full-corpus ranking audit is separate"}
        result["scope"]="real frozen T5 and CodeBERT inference; not full retraining"
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result),flush=True)


if __name__=="__main__":main()
