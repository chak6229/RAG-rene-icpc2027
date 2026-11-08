"""Build and train a small valid-only CRSU rewrite pilot.

This is deliberately a separate artifact. It recomputes actual rankings from the
frozen CodeBERT checkpoint, audits existing Delta-RR labels, then trains the
query-level identity posterior. It never uses a gold document to form features.
"""
from __future__ import annotations
import argparse, hashlib, json, pickle, random
from pathlib import Path
import numpy as np
import torch
from scipy import stats

from build_rewrite_utility_dataset import code_text, encode, reciprocal_rank
from prepare_vac_mqr_cv_folds import repository
from crsu import RankingStates, summarize, rrf
from crsu.model import EvidencePosterior
from crsu.training import EvidenceInput, TrainingExample, fit_fixed_epochs, predict_probabilities


def read_jsonl(path): return [json.loads(s) for s in Path(path).open(encoding='utf-8') if s.strip()]
def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()


def stable_indices(scores, k=100):
    scores=np.asarray(scores,float)
    return np.lexsort((np.arange(len(scores)), -scores))[:min(k,len(scores))]


def action_features(doc_ids, scores, query_vec, candidate_vec):
    top=stable_indices(scores,100); position={int(x):i+1 for i,x in enumerate(top)}
    qcos=float(query_vec@candidate_vec)
    out=np.zeros((len(doc_ids),6),np.float32)
    for i,doc in enumerate(doc_ids):
        rank=position.get(int(doc),0)
        out[i]=[float(scores[doc]), rank/100., float(rank>0), qcos,
                float(scores[top[0]]), float(scores[doc]-scores[top[0]])]
    return out, tuple(int(x) for x in top)


def build_query(row, valid, code_vectors, query_vec, candidate_vectors, offset, audit, projection, *, metric_policy='legacy_audit'):
    qi=int(row['query_index']); original_scores=code_vectors@query_vec
    original_top=stable_indices(original_scores,100)
    candidates=[c for c in row['candidates'] if c.get('quality_pass')]
    candidate_scores=[]; action_lists={}; action_arrays=[]
    for j,c in enumerate(candidates):
        scores=code_vectors@candidate_vectors[offset+j]; candidate_scores.append(scores)
        arr,ranking=action_features(np.arange(len(valid)),scores,query_vec,candidate_vectors[offset+j])
        action_arrays.append(arr); action_lists[f'rewrite_{j+1}']=ranking
    states=RankingStates.from_rankings(original_top,action_lists,cutoff=100)
    fusion_lists={f'fusion_{j+1}':rrf(original_top, ranking) for j,ranking in enumerate(action_lists.values())}
    fusion_states=RankingStates.from_rankings(original_top,fusion_lists,cutoff=100)
    # Feature rows follow the exact sorted document union, while corpus IDs remain
    # metadata outside the network input.
    doc_ids=np.asarray(states.document_ids,dtype=int)
    original_rank={int(x):i+1 for i,x in enumerate(original_top)}
    anchor=query_vec@projection; doc_projected=code_vectors[doc_ids]@projection
    geometry=np.zeros((len(doc_ids),5),np.float32)
    for i,doc in enumerate(doc_ids):
        r=original_rank.get(int(doc),0)
        geometry[i]=[float(original_scores[doc]),r/100.,float(r>0),
                         float(original_scores[original_top[0]]),float(original_scores[doc]-original_scores[original_top[0]])]
    doc_features=np.concatenate((doc_projected*anchor,abs(doc_projected-anchor),geometry),axis=1).astype(np.float32)
    # Align action arrays from corpus ordinal to union order.
    action_evidence=np.zeros((len(doc_ids),len(candidates),6),np.float32)
    for a,(name,arr) in enumerate(zip(action_lists,action_arrays)):
        action_evidence[:,a]=arr[doc_ids]
    qfeatures=np.asarray([float(original_scores[original_top[0]]),
                          float(original_scores[original_top[0]]-original_scores[original_top[1]]),
                          float(original_scores.std()),np.log1p(len(doc_ids))],np.float32)
    target=states.target_index(qi)
    # Existing labels are joined only here, after all unlabeled features/rankings
    # are built. Audit every quality-passed candidate, not just the target.
    if metric_policy not in ('legacy_audit','actual_list_v2'):raise ValueError('unknown metric policy')
    competition_original_rr=reciprocal_rank(original_scores,qi,100)
    actual_original_rr=float(states.reciprocal_ranks[target,0])
    original_rr=competition_original_rr if metric_policy=='legacy_audit' else actual_original_rr
    if not np.isclose(competition_original_rr,actual_original_rr,atol=1e-12,rtol=0):
        audit.setdefault('competition_list_tie_differences',[]).append({'query_index':qi,'action':'original',
            'competition_rr':competition_original_rr,'actual_list_rr':actual_original_rr})
    if not np.isclose(original_rr,float(row['original_rr_at_100']),atol=1e-12,rtol=0):
        audit['original_mismatches']+=1
    deltas={}
    for j,c in enumerate(candidates):
        competition_replacement=reciprocal_rank(candidate_scores[j],qi,100)
        actual_replacement=float(states.reciprocal_ranks[target,j+1])
        replacement=competition_replacement if metric_policy=='legacy_audit' else actual_replacement
        if not np.isclose(competition_replacement,actual_replacement,atol=1e-12,rtol=0):
            audit.setdefault('competition_list_tie_differences',[]).append({'query_index':qi,'action':f'rewrite_{j+1}',
                'competition_rr':competition_replacement,'actual_list_rr':actual_replacement})
        deltas[f'rewrite_{j+1}']=float(replacement-original_rr)
        cached=float(c['replacement_rr_at_100']-row['original_rr_at_100'])
        if not np.isclose(replacement,float(c['replacement_rr_at_100']),atol=1e-12,rtol=0):
            audit['replacement_mismatches']+=1
        audit['checked_candidates']+=1
    states.audit_labels(qi,original_rr,deltas)
    # Fusion is audited separately against its frozen labels; identity training
    # still uses the simpler rewrite-only action family in this first pilot.
    fusion_deltas={name:float(fusion_states.gains[fusion_states.target_index(qi),j])
                   for j,name in enumerate(fusion_lists)}
    for j,c in enumerate(candidates):
        actual_fused_rr=float(fusion_states.reciprocal_ranks[fusion_states.target_index(qi),j+1])
        audit['fusion_rr_checks']=audit.get('fusion_rr_checks',0)+1
        audit['fusion_delta_checks']=audit.get('fusion_delta_checks',0)+1
        if not np.isclose(actual_fused_rr,float(c['rr_at_100']),atol=1e-12,rtol=0):
            audit['fusion_rr_mismatches']=audit.get('fusion_rr_mismatches',0)+1
        if not np.isclose(fusion_deltas[f'fusion_{j+1}'],float(c['rr_at_100']-row['original_rr_at_100']),atol=1e-12,rtol=0):
            audit['fusion_delta_mismatches']=audit.get('fusion_delta_mismatches',0)+1
    for name,d in deltas.items():
        c=candidates[int(name.rsplit('_',1)[1])-1]
        audit['delta_checks']+=1
        if not np.isclose(d,float(c['replacement_rr_at_100']-row['original_rr_at_100']),atol=1e-12,rtol=0):
            audit['delta_mismatches']+=1
    return dict(query_index=qi,role=row['split_role'],repo=repository(row['url']),
                states=states, inputs=EvidenceInput(doc_features,action_evidence,qfeatures),
                target=target, observed=np.asarray(list(deltas.values()),float),
                action_names=list(deltas), original_rr=float(original_rr), deltas=deltas,
                candidate_count=len(candidates),candidate_source_ranks=[int(c['rank']) for c in candidates],
                fusion_states=fusion_states,fusion_deltas=fusion_deltas)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--max-train',type=int,default=160); ap.add_argument('--max-eval',type=int,default=64); ap.add_argument('--audit-only',action='store_true'); args=ap.parse_args()
    out=args.output; out.mkdir(parents=True,exist_ok=True); rng=np.random.default_rng(1101); torch.manual_seed(1101); torch.set_num_threads(2)
    source=Path('results/vac_mqr_v5/enhanced_cv_folds/fold_0.jsonl'); valid_path=Path('data/codexglue/advtest/valid.jsonl'); model_path=Path('models/codebert-codexglue-2epoch/checkpoint-3000')
    rows=read_jsonl(source); valid=read_jsonl(valid_path)
    if any(r.get('split_role')=='test' for r in rows): raise RuntimeError('test rows forbidden')
    pools={role:[r for r in rows if r['split_role']==role] for role in ('utility_train','selection_valid','new_holdout')}
    def sample(pool, n):
        ix=np.sort(rng.choice(len(pool),min(n,len(pool)),replace=False))
        return [pool[i] for i in ix]
    train_rows=sample(pools['utility_train'],args.max_train); eval_rows=sample(pools['new_holdout'],args.max_eval)
    assert not {repository(r['url']) for r in train_rows} & {repository(r['url']) for r in eval_rows}
    # Encode only valid corpus and sampled query/candidate texts. The corpus is
    # the fixed closed retrieval set; labels are never sent to encode().
    from transformers import AutoTokenizer,AutoModel
    device='cuda' if torch.cuda.is_available() else 'cpu'; tokenizer=AutoTokenizer.from_pretrained(model_path); encoder=AutoModel.from_pretrained(model_path).to(device).eval()
    cache=Path('results/crsu_development/valid_vector_cache');cache.mkdir(parents=True,exist_ok=True)
    weight_files=sorted(list(model_path.glob('*.bin'))+list(model_path.glob('*.safetensors')))
    cache_key=dict(valid_sha256=sha256(valid_path),weights={str(p):sha256(p) for p in weight_files},
                   tokenizer={str(p):sha256(p) for p in model_path.glob('*') if p.is_file() and p.suffix in ('.json','.txt')},
                   max_length=256,batch_size=64,encoding='CLS normalized; BF16 when supported')
    key_file=cache/'manifest.json';vector_file=cache/'code_vectors.npy'
    with torch.inference_mode():
        if key_file.exists() and vector_file.exists() and json.loads(key_file.read_text())==cache_key:
            code_vectors=np.load(vector_file)
            if code_vectors.shape[0]!=len(valid) or not np.isfinite(code_vectors).all():raise RuntimeError('invalid cached corpus vectors')
            print('valid-only code vector cache loaded',flush=True)
        else:
            print(f'encoding valid-only corpus: {len(valid)} documents',flush=True)
            code_vectors=encode(encoder,tokenizer,[code_text(r) for r in valid],64,256,device)
            np.save(vector_file,code_vectors);key_file.write_text(json.dumps(cache_key,indent=2),encoding='utf-8')
        all_rows=train_rows+eval_rows
        qvecs=encode(encoder,tokenizer,[r['query'] for r in all_rows],64,256,device)
        ctexts=[c['text'] for r in all_rows for c in r['candidates'] if c.get('quality_pass')]
        cvecs=encode(encoder,tokenizer,ctexts,64,256,device)
    if any(valid[int(r['query_index'])]['url']!=r['url'] for r in all_rows):raise RuntimeError('query/corpus alignment mismatch')
    projection=np.random.default_rng(777).normal(0,1/np.sqrt(32),(code_vectors.shape[1],32)).astype('float32')
    audit={'test_used':False,'source':str(source),'valid':str(valid_path),'checked_queries':0,'checked_candidates':0,'delta_checks':0,'original_mismatches':0,'replacement_mismatches':0,'delta_mismatches':0,
        'fusion_rr_checks':0,'fusion_delta_checks':0,'fusion_rr_mismatches':0,'fusion_delta_mismatches':0,'ranking_errors':[]}
    examples=[]; meta=[]; offset=0
    for i,row in enumerate(all_rows):
        try: item=build_query(row,valid,code_vectors,qvecs[i],cvecs,offset,audit,projection)
        except Exception as exc:
            audit['ranking_errors'].append({'query_index':int(row['query_index']),'error':repr(exc)})
            (out/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
            raise
        offset += item['candidate_count']; audit['checked_queries']+=1; meta.append(item)
        if i<len(train_rows): examples.append(TrainingExample(item['inputs'],item['target'],'utility_train'))
    (out/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    with (out/'items.pkl').open('wb') as f: pickle.dump(meta,f)
    with (out/'rankings.jsonl').open('w',encoding='utf-8') as f:
        for item in meta:
            f.write(json.dumps({'query_index':item['query_index'],'role':item['role'],
                'document_ids':item['states'].document_ids,'action_names':item['states'].action_names,
                'ranks_including_outside':item['states'].ranks.tolist(),'candidate_source_ranks':item['candidate_source_ranks'],
                'fusion_document_ids':item['fusion_states'].document_ids,
                'fusion_action_names':item['fusion_states'].action_names,
                'fusion_ranks_including_outside':item['fusion_states'].ranks.tolist()})+'\n')
    with (out/'labels.jsonl').open('w',encoding='utf-8') as f:
        for item in meta:
            f.write(json.dumps({'query_index':item['query_index'],'role':item['role'],
                'target_slot':item['target'],'original_rr':item['original_rr'],'deltas':item['deltas'],
                'fusion_deltas':item['fusion_deltas']})+'\n')
    np.save(out/'projection.npy',projection)
    (out/'sample_manifest.json').write_text(json.dumps({'train_query_ids':[r['query_index'] for r in train_rows],
        'evaluation_query_ids':[r['query_index'] for r in eval_rows],'seed':1101,'train_role':'utility_train',
        'evaluation_role':'new_holdout','repository_overlap':0,'test_used':False},indent=2),encoding='utf-8')
    if any(audit[k] for k in ('original_mismatches','replacement_mismatches','delta_mismatches','fusion_rr_mismatches','fusion_delta_mismatches')):
        raise RuntimeError('historical labels mismatch; audit saved and training blocked')
    if args.audit_only:
        print(json.dumps(audit,indent=2)); return
    # Fit normalization only on training-role evidence. Preserve actual rank
    # geometry separately in RankingStates; these statistics never see gold.
    feature_groups=('document_features','action_evidence','query_features'); normalization={}
    for name in feature_groups:
        values=[getattr(x.inputs,name) for x in examples]
        dimension=values[0].shape[-1]
        matrix=np.concatenate([v.reshape(-1,dimension) for v in values if v.size],axis=0)
        mean=matrix.mean(0);std=matrix.std(0);std[std<1e-6]=1
        normalization[name]=(mean,std)
    def normalized(x):
        return EvidenceInput(*[((getattr(x,name)-normalization[name][0])/normalization[name][1]).astype(np.float32) for name in feature_groups])
    examples=[TrainingExample(normalized(x.inputs),x.target_state,x.role) for x in examples]
    model=EvidencePosterior(69,6,4,hidden=64)
    print(f'identity training: {len(examples)} queries / 30 fixed epochs',flush=True)
    losses=fit_fixed_epochs(model,examples,epochs=30,batch_size=16,learning_rate=.001,seed=1101,device=device)
    eval_items=meta[len(train_rows):]; probabilities=[]
    for start in range(0,len(eval_items),16):
        probabilities.extend(predict_probabilities(model,[normalized(x['inputs']) for x in eval_items[start:start+16]],device))
    nll=[]; brier=[]; rows_out=[]
    for item,p in zip(eval_items,probabilities):
        target=item['target']; nll.append(float(-np.log(max(p[target],1e-15))))
        y=np.zeros(len(p)); y[target]=1.; brier.append(float(np.sum((p-y)**2)))
        dist=summarize(item['states'],p); actual=item['observed']; crps=dist.crps(actual)
        rows_out.append({'query_index':item['query_index'],'union_size':len(item['states'].document_ids),
            'target_in_union':target<len(item['states'].document_ids),'target_probability':float(p[target]),
            'identity_top1_correct':int(np.argmax(p))==target,
            'conditional_identity_top1_correct':int(np.argmax(p[:-1]))==target if target<len(p)-1 else None,
            'outside_probability':float(p[-1]),'nll':nll[-1],'brier':brier[-1],
            'predicted_mu':dist.mu.tolist(),'predicted_scale':dist.scale.tolist(),'predicted_harm':dist.harm.tolist(),
            'predicted_expected_loss':dist.expected_loss.tolist(),'observed_gain':actual.tolist(),
            'crps':crps.tolist(),'candidate_count':item['candidate_count']})
    train_tail=(sum(x['target']==len(x['states'].document_ids) for x in meta[:len(train_rows)])+.5)/(len(train_rows)+1)
    baseline_nll=[];baseline_brier=[];baseline_crps=[];baseline_mae=[];baseline_tail=[];baseline_top1=[]
    for item in eval_items:
        cosine=item['inputs'].document_features[:,-5].astype(np.float64);weights=np.exp((cosine-cosine.max())/.1);weights/=weights.sum()
        prior=np.r_[(1-train_tail)*weights,train_tail];target=item['target']
        baseline_nll.append(float(-np.log(prior[target])))
        target_vector=np.zeros(len(prior));target_vector[target]=1
        baseline_brier.append(float(np.sum((prior-target_vector)**2)))
        base_distribution=summarize(item['states'],prior)
        baseline_crps.extend(base_distribution.crps(item['observed']).tolist())
        baseline_mae.extend(abs(base_distribution.mu-item['observed']).tolist())
        baseline_tail.append((train_tail-int(target==len(prior)-1))**2)
        baseline_top1.append(int(np.argmax(prior))==target)
    dump={'protocol':'crsu-rewrite-pilot-v1','development_only':True,'test_used':False,'source_sha256':sha256(source),'valid_sha256':sha256(valid_path),'model_checkpoint':str(model_path),'encoding_provenance':cache_key,'train_queries':len(examples),'eval_queries':len(eval_items),'ranking_audit':audit,'training_loss':losses,'identity_nll_mean':float(np.mean(nll)),'identity_brier_mean':float(np.mean(brier)),'target_in_union_rate':float(np.mean([x['target_in_union'] for x in rows_out])),
          'fixed_dense_identity_baseline':{'temperature':.1,'train_estimated_tail':train_tail,'identity_nll':float(np.mean(baseline_nll)),'identity_brier':float(np.mean(baseline_brier)),
            'gain_mae':float(np.mean(baseline_mae)),'marginal_crps':float(np.mean(baseline_crps)),
            'tail_brier':float(np.mean(baseline_tail)),'identity_top1_accuracy':float(np.mean(baseline_top1))},'results':rows_out}
    mu=np.concatenate([np.asarray(x['predicted_mu']) for x in rows_out]);scale=np.concatenate([np.asarray(x['predicted_scale']) for x in rows_out]);y=np.concatenate([np.asarray(x['observed_gain']) for x in rows_out])
    rho=stats.spearmanr(scale,abs(mu-y)).statistic
    dump['gain_diagnostics']={'candidate_count':len(y),'mae':float(np.mean(abs(mu-y))),
        'zero_mean_baseline_mae':float(np.mean(abs(y))),
        'marginal_crps':float(np.mean(np.concatenate([np.asarray(x['crps']) for x in rows_out]))),
        'error_scale_spearman':float(rho) if np.isfinite(rho) else None,
        'identity_top1_accuracy':float(np.mean([x['identity_top1_correct'] for x in rows_out])),
        'conditional_in_union_top1_accuracy':float(np.mean([x['conditional_identity_top1_correct'] for x in rows_out if x['target_in_union']])),
        'tail_brier':float(np.mean([(x['outside_probability']-int(not x['target_in_union']))**2 for x in rows_out]))}
    (out/'summary.json').write_text(json.dumps(dump,ensure_ascii=False,indent=2),encoding='utf-8')
    with (out/'items.pkl').open('wb') as f: pickle.dump(meta,f)
    torch.save({'state_dict':model.state_dict(),'document_dim':69,'action_dim':6,'query_dim':4,'hidden':64,'normalization':normalization},out/'model.pt')
    with (out/'evaluation_probabilities.pkl').open('wb') as f:pickle.dump(probabilities,f)
    (out/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    scripts=[Path(__file__),*Path('crsu').glob('*.py')]
    (out/'reproducibility.json').write_text(json.dumps({'scripts_sha256':{str(p):sha256(p) for p in scripts},
        'environment':{'torch':torch.__version__,'numpy':np.__version__,'device':device},
        'train':{'seed':1101,'hidden':64,'epochs':30,'batch_size':16,'lr':.001},
        'projection':{'seed':777,'dimension':32},'normalization_fit_role':'utility_train',
        'ranking_policy':'score descending, corpus ordinal ascending; RRF offset=60',
        'actual_sample_size':{'train':len(examples),'evaluation':len(eval_items)},'test_used':False},indent=2),encoding='utf-8')
    print(json.dumps({'status':'completed','train_queries':len(examples),'eval_queries':len(eval_items),'identity_nll_mean':dump['identity_nll_mean'],'target_in_union_rate':dump['target_in_union_rate'],'audit':audit},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
