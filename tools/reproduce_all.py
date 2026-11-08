"""Verify and replay all self-contained numerical and human review evidence."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]

def main():
    start=time.perf_counter()
    tasks=[['tools/verify_bundle.py'],['-I','artifact/numerical/reproduce.py','--output',str(ROOT/'reproduced/numerical')],
           ['human_review/reproduce.py','--output',str(ROOT/'reproduced/human_review')],['human_review/verify_human_reviews.py']]
    for args in tasks: subprocess.run([sys.executable,*args],cwd=ROOT,check=True)
    numerical=json.loads((ROOT/'reproduced/numerical/ACCEPTANCE.json').read_text(encoding='utf-8'))
    human=json.loads((ROOT/'reproduced/human_review/ACCEPTANCE.json').read_text(encoding='utf-8'))
    if not numerical['passed'] or not numerical['full_bootstrap'] or not human['passed']: raise SystemExit('Incomplete replay')
    result={'passed':True,'numerical_studies':numerical['studies'],'numerical_contrasts':numerical['contrasts'],
            'full_bootstrap':True,'human_review':'human review','human_query_bounds':human['recomputed_exact_fraction_bounds'],
            'human_sampling_combinations':human['recomputed_sampling_envelopes'],'elapsed_seconds':time.perf_counter()-start}
    (ROOT/'reproduced/ACCEPTANCE.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
