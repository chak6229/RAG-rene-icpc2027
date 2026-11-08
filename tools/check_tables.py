"""Check delivered table exports against their accepted machine-readable source."""
import ast
import csv
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def cell(text, value):
    if value is None: return text in ('','None','null')
    if isinstance(value,str): return text==value
    try: parsed=json.loads(text)
    except (ValueError,TypeError):
        try: parsed=ast.literal_eval(text)
        except (ValueError,SyntaxError): parsed=text
    if isinstance(value,float): return isinstance(parsed,(int,float)) and math.isclose(parsed,value,rel_tol=0,abs_tol=1e-12)
    return parsed==value

def main():
    data=json.loads((ROOT/'results/TABLES.json').read_text(encoding='utf-8-sig'))
    total=0
    for key,expected in data['tables'].items():
        path=ROOT/'results/tables'/f'{key}.csv'
        with path.open(encoding='utf-8-sig',newline='') as f: actual=list(csv.DictReader(f))
        if len(actual)!=len(expected): raise ValueError(f'Table length: {key}')
        for i,(left,right) in enumerate(zip(actual,expected)):
            if not set(right).issubset(left): raise ValueError(f'Columns: {key} row {i}')
            if any(left[k] not in ('','None','null') for k in set(left)-set(right)):
                raise ValueError(f'Unexpected populated column: {key} row {i}')
            for column,value in right.items():
                if not cell(left[column],value): raise ValueError(f'Cell mismatch: {key} row {i} column {column}')
        total+=len(expected)
    if len(data['tables'])!=15 or total!=176: raise ValueError('Expected 15 tables / 176 rows')
    print(json.dumps({'passed':True,'tables':15,'rows':176,'scope':'export integrity against accepted TABLES.json'}))

if __name__=='__main__': main()
