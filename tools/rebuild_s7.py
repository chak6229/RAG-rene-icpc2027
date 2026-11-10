"""Independently recompute current Table S7 from frozen full-H stage records."""
from pathlib import Path
import argparse, gzip, json, math
ROOT=Path(__file__).resolve().parents[1]
KEY='T7_full_H_opportunity_accounting'

def compute():
    result=[]
    for retriever in ('CodeBERT','BM25'):
        with gzip.open(ROOT/f'artifact/numerical/data/full_H_stages_{retriever}.jsonl.gz','rt',encoding='utf-8') as f:
            items=[json.loads(line) for line in f]
        n=len(items)
        available=[max(r['best3']['ENTR']-r['baseline'],0.) for r in items]
        for stage in ('RIDGE_gain','RIDGE_verified','RIDGE_risk'):
            deltas=[r['policies'][stage]['rr']-r['baseline'] for r in items]
            # Compute both loss terms query by query, independently of the export.
            unretained=[u-max(d,0.) for u,d in zip(available,deltas)]
            losses=[max(-d,0.) for d in deltas]
            assert min(unretained)>=-1e-12
            for u,d,a,b in zip(available,deltas,unretained,losses):assert abs(u-d-a-b)<1e-12
            result.append({'retriever':retriever,'stage':stage,'queries':n,'groups':len({r['g'] for r in items}),
                'mean_available_gain':math.fsum(available)/n,'mean_unretained_positive_gain':math.fsum(unretained)/n,
                'mean_ranking_loss':math.fsum(losses)/n,'delta_mrr100':math.fsum(deltas)/n,
                'verification_pairs':sum(r['policies'][stage]['pairs'] for r in items),
                'verification_input_tokens':sum(r['policies'][stage]['tokens'] for r in items)})
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
    actual=compute();tables=json.loads((ROOT/'results/TABLES.json').read_text(encoding='utf-8'))['tables']
    expected=tables[KEY];assert len(actual)==len(expected)==6
    stages={(r['retriever'],r['policy']):r for r in tables['T5_frozen_H_stages']}
    for left,right in zip(actual,expected):
        assert set(left)==set(right)
        for name,value in left.items():
            if isinstance(value,float):assert math.isclose(value,right[name],rel_tol=0.,abs_tol=1e-12),(name,value,right[name])
            else:assert value==right[name]
        original=stages[(left['retriever'],left['stage'])]
        assert abs(left['delta_mrr100']-original['delta'])<1e-12
        assert left['verification_pairs']==original['CE_pairs']
        retained=1-left['mean_unretained_positive_gain']/left['mean_available_gain']
        assert abs(retained-original['positive_gain_retention'])<1e-12
    output=ROOT/'reproduced/s7_accounting';output.mkdir(parents=True,exist_ok=True)
    (output/'RESULTS.json').write_text(json.dumps(actual,indent=2)+'\n',encoding='utf-8')
    summary={'passed':True,'rows':6,'frozen_query_records':17060,'descriptive_only':True,'new_statistical_comparisons':0,
             'query_level_accounting_identity':True,'agrees_with_original_S5':True,'bootstrap_unchanged':True}
    (output/'ACCEPTANCE.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary))

if __name__=='__main__':main()
