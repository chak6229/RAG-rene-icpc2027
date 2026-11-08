"""Verify the paper-free review repository without model or data downloads."""
import ast
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]

def rows(p):
    return [json.loads(x) for x in p.read_text(encoding='utf-8-sig').splitlines() if x.strip()]

def main():
    hashes=json.loads((ROOT/'SHA256SUMS.json').read_text(encoding='utf-8'))
    errors=[]
    py=0
    for rel,expected in hashes.items():
        path=ROOT/rel
        if not path.is_file(): errors.append(f'Missing {rel}'); continue
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected: errors.append(f'Hash mismatch {rel}')
        if path.stat().st_size>=100*1024*1024: errors.append(f'Exceeds GitHub regular-file limit: {rel}')
        if (path.suffix.lower() in ('.docx','.doc','.tex','.bib','.zip') and rel != 'supplementary/SSQR_Supplementary_S1-S25_EN.docx') or path.name in ('manuscript.md','main.pdf') or 'paper' in Path(rel).parts:
            errors.append(f'Unexpected paper/archive file {rel}')
        if path.suffix=='.py':
            py+=1
            try: ast.parse(path.read_text(encoding='utf-8-sig'),filename=rel)
            except SyntaxError as e: errors.append(f'Python syntax: {rel}: {e}')
        if path.suffix in ('.md','.json','.jsonl','.py','.csv','.txt','.ps1','.sh'):
            content=path.read_text(encoding='utf-8-sig')
            if re.search(r'[A-Za-z]:[/\\]+Users[/\\]+\d{3,}',content): errors.append(f'Workstation account path {rel}')
    initial=rows(ROOT/'human_review/data/original_annotations.jsonl')
    supplement=rows(ROOT/'human_review/data/supplement_annotations.jsonl')
    bounds=rows(ROOT/'human_review/data/weighted_query_bounds.jsonl')
    expected=Counter({('A','query_clarity'):128,('B','query_clarity'):128,('A','rewrite_fidelity'):383,('B','rewrite_fidelity'):383,('A','code_relevance'):327,('B','code_relevance'):327})
    if Counter((r['reviewer'],r['item_type']) for r in initial)!=expected: errors.append('Initial annotation counts')
    if Counter(r['reviewer'] for r in supplement)!={'A':344,'B':344}: errors.append('Supplement annotation counts')
    if len(bounds)!=6144: errors.append('Reference relevance bounds count')
    for folder in (ROOT/'human_review/data',ROOT/'human_review/reference'):
        for p in folder.glob('*'):
            text=p.read_text(encoding='utf-8')
            if re.search(r'"(?:started_at|exported_at|duration|elapsed|process_time|timestamp|created_utc)"\s*:',text): errors.append(f'Review process-time field: {p.name}')
    if errors: raise SystemExit('\n'.join(errors))
    subprocess.run([sys.executable,str(ROOT/'tools/check_tables.py')],cwd=ROOT,check=True)
    print(json.dumps({'passed':True,'hashed_files':len(hashes),'python_sources_parsed':py,
                      'paper_files':0,'max_file_bytes':max((ROOT/rel).stat().st_size for rel in hashes),
                      'initial_annotations':len(initial),'supplement_annotations':len(supplement),'weighted_query_bounds':len(bounds)},indent=2))

if __name__=='__main__': main()
