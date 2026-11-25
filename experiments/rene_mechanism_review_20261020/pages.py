from __future__ import annotations

import json


STYLE = '''
:root{font-family:"Microsoft YaHei",system-ui,sans-serif;color:#1d2939;background:#f4f5f7;font-size:15px}
body{margin:0}header{background:#17394d;color:white;padding:24px max(24px,calc((100vw - 1040px)/2))}
h1{font-size:25px;margin:0 0 8px}h2{font-size:20px;margin:0 0 15px}h3{font-size:16px;margin:0 0 12px}
main{max-width:1040px;margin:24px auto;padding:0 20px 100px}.card{background:white;border:1px solid #dae0e7;border-radius:12px;padding:24px;margin:18px 0}
.muted{color:#526477;font-size:14px;line-height:1.8}.query{font-size:19px;line-height:1.7;border-left:4px solid #218279;padding:12px 18px;background:#f1f8f7;white-space:pre-wrap;overflow-wrap:anywhere}
label{display:block;font-weight:600;margin:18px 0 7px}select,textarea,input[type=file]{font:inherit;box-sizing:border-box;max-width:100%}select{min-width:230px;padding:9px;border:1px solid #b3beca;border-radius:6px;background:white}
textarea{width:100%;min-height:76px;resize:vertical;padding:10px;border:1px solid #b3beca;border-radius:6px}button{font:inherit;padding:10px 17px;border-radius:7px;border:1px solid #adc0ca;background:white;cursor:pointer;color:#17394d}
button.primary{background:#17695f;color:white;border-color:#17695f}button:disabled{opacity:.4;cursor:default}.toolbar{display:flex;gap:10px;flex-wrap:wrap;align-items:center}.spacer{flex:1}.bar{height:5px;background:#d8e4e7;border-radius:5px;margin:15px 0}.bar>div{height:100%;background:#218279;transition:width .2s}
pre{margin:0;background:#142533;color:#eff7fa;padding:18px;border-radius:7px;font:14px/1.65 Consolas,monospace;overflow:auto;max-height:460px;tab-size:4}code{font:inherit}.hide{display:none!important}.notice{background:#fff5df;border:1px solid #e6c77b;padding:13px;border-radius:7px;line-height:1.8}
.badge{display:inline-block;font-size:12px;background:#e5eeef;color:#285a62;padding:4px 9px;border-radius:20px;margin-bottom:12px}.checks{display:flex;flex-wrap:wrap;gap:10px}.checks label{font-size:14px;font-weight:400;margin:0;padding:8px;background:#f4f6f8;border-radius:4px}
footer{position:fixed;bottom:0;left:0;right:0;background:#fff;border-top:1px solid #dce2e6;padding:13px max(20px,calc((100vw - 1040px)/2));box-shadow:0 -4px 20px #17394d0a}.status{font-size:13px;color:#34586a}
details{line-height:1.8}summary{cursor:pointer;font-weight:600}a{color:#17695f}@media(max-width:650px){.card{padding:17px}h1{font-size:21px}.query{font-size:17px}header{padding:22px}.toolbar select{min-width:130px}}
'''

SCRIPT = r'''
const P=JSON.parse(document.getElementById('packet').textContent), isSem=P.phase==='semantic';
const SKEY='query-review-'+P.packet_id+'-'+P.phase;
let state={answers:{},index:0,started_utc:new Date().toISOString(),semantic:null};
try{const old=JSON.parse(localStorage.getItem(SKEY)||'null');if(old)state=old}catch(e){}
const $=id=>document.getElementById(id), esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fidelity=[['preserved','Meaning preserved: retains the original request and constraints'],['drift','Meaning drift: adds an unsupported constraint or changes the request'],['malformed','Unintelligible: no clear request can be identified'],['uncertain','Cannot determine: the original request or context is insufficient']];
const flagOptions=[['unsupported_constraint','Adds an unsupported constraint'],['lost_constraint','Drops an original constraint'],['operation_change','Changes the operation'],['entity_api_change','Changes the entity or API'],['polarity_change','Changes the polarity'],['incoherent','Incoherent wording'],['none','No issue identified'],['uncertain','Cannot determine']];
const relevant=[['2','Satisfies the request'],['1','Partially satisfies the request'],['0','Does not satisfy the request'],['u','Cannot determine / insufficient context']];
function validRewrite(r){if(!r||!fidelity.some(x=>x[0]===r.fidelity)||!Array.isArray(r.flags)||!r.flags.length||new Set(r.flags).size!==r.flags.length||!r.flags.every(f=>flagOptions.some(x=>x[0]===f))||String(r.note||'').trim().length<6)return false;let fs=r.flags;if((fs.includes('none')||fs.includes('uncertain'))&&fs.length!==1)return false;if(r.fidelity==='preserved'&&fs.join()!=='none')return false;if(r.fidelity==='uncertain'&&fs.join()!=='uncertain')return false;if(r.fidelity==='drift'&&(fs.includes('none')||fs.includes('uncertain')))return false;if(r.fidelity==='malformed'&&!fs.includes('incoherent'))return false;return true}
function save(){try{localStorage.setItem(SKEY,JSON.stringify(state));$('autosave').textContent='Progress saved in this browser'}catch(e){$('autosave').textContent='The browser could not save progress. Export a backup.'}}
function choice(opts,current){return '<option value="">Select an option</option>'+opts.map(([k,v])=>`<option value="${k}" ${k===current?'selected':''}>${v}</option>`).join('')}
function semValid(p){
 if(!p||p.packet_id!==P.packet_id||p.reviewer!==P.reviewer||p.phase!=='semantic'||!p.complete||!p.independence_confirmed)return false;
 if(Object.keys(p.answers||{}).length!==P.cases.length)return false;
 return P.semantic_schema.every(c=>{let a=p.answers[c.case_id];return a&&['clear','ambiguous','insufficient'].includes(a.clarity)&&a.rewrites&&Object.keys(a.rewrites).length===c.rewrite_ids.length&&c.rewrite_ids.every(id=>validRewrite(a.rewrites[id]))});
}
function complete(c){const a=state.answers[c.case_id];if(!a)return false;
 if(isSem)return ['clear','ambiguous','insufficient'].includes(a.clarity)&&c.rewrites.every(r=>validRewrite((a.rewrites||{})[r.id]));
 return c.documents.every(d=>{let z=(a.documents||{})[d.id];return z&&relevant.some(x=>x[0]===z.relevance)&&String(z.note||'').trim().length>=6});
}
function update(){const n=P.cases.filter(complete).length;$('progress').textContent=`Completed ${n} / ${P.cases.length} items`;$('bar').style.width=(100*n/P.cases.length)+'%';$('position').textContent=`Item ${state.index+1} / ${P.cases.length}`;$('prev').disabled=state.index===0;$('next').disabled=state.index===P.cases.length-1;}
function render(){
 if(!isSem&&!semValid(state.semantic)){$('review').classList.add('hide');$('gate').classList.remove('hide');return}
 $('gate').classList.add('hide');$('review').classList.remove('hide');state.index=Math.max(0,Math.min(state.index,P.cases.length-1));
 const c=P.cases[state.index],a=state.answers[c.case_id]||(state.answers[c.case_id]={});
 let html=`<div class="card"><span class="badge">Case ${esc(c.case_id)}</span><h2>Original query</h2><div class="query">${esc(c.original)}</div>`;
 if(isSem){html+=`<label for="clarity">Does the original query express a clear request?</label><select id="clarity">${choice([['clear','Clear request'],['ambiguous','Multiple plausible interpretations'],['insufficient','Insufficient information to determine the request']],a.clarity)}</select></div>`;
  a.rewrites=a.rewrites||{};
  for(const r of c.rewrites){const z=a.rewrites[r.id]||(a.rewrites[r.id]={flags:[]});html+=`<section class="card"><h3>Rewrite ${esc(r.id)}</h3><div class="query">${esc(r.text)}</div><label for="f-${r.id}">Meaning relative to the original query</label><select id="f-${r.id}" data-rid="${r.id}" data-kind="fidelity">${choice(fidelity,z.fidelity)}</select><label>Reasons for the judgment (select all that apply)</label><div class="checks">${flagOptions.map(([k,v])=>`<label><input type="checkbox" data-rid="${r.id}" data-flag="${k}" ${z.flags.includes(k)?'checked':''}> ${v}</label>`).join('')}</div><label for="n-${r.id}">Cite specific wording when explaining your judgment (at least 6 characters)</label><textarea id="n-${r.id}" data-rid="${r.id}" data-kind="note" placeholder="Explain whether the original operation and entity are retained, or why an added constraint is unsupported.">${esc(z.note||'')}</textarea></section>`}
 }else{html+='</div>';a.documents=a.documents||{};
  for(const d of c.documents){const z=a.documents[d.id]||(a.documents[d.id]={});html+=`<section class="card"><h3>Code ${esc(d.id)}</h3><pre><code>${esc(d.code)}</code></pre><label for="r-${d.id}">How well the code meets the original request</label><select id="r-${d.id}" data-did="${d.id}" data-kind="relevance">${choice(relevant,z.relevance)}</select><label for="n-${d.id}">Reason for the judgment (at least 6 characters)</label><textarea id="n-${d.id}" data-did="${d.id}" data-kind="note" placeholder="State what the code implements and what it omits. If context is insufficient, specify what information is missing.">${esc(z.note||'')}</textarea></section>`}
 }
 $('content').innerHTML=html;
 if(isSem)$('clarity').onchange=e=>{a.clarity=e.target.value;save();update()};
 $('content').querySelectorAll('[data-kind]').forEach(el=>el.addEventListener(el.tagName==='TEXTAREA'?'input':'change',()=>{const z=isSem?a.rewrites[el.dataset.rid]:a.documents[el.dataset.did];z[el.dataset.kind]=el.value;save();update()}));
 $('content').querySelectorAll('[data-flag]').forEach(el=>el.onchange=()=>{let z=a.rewrites[el.dataset.rid];z.flags=[...$('content').querySelectorAll(`[data-rid="${el.dataset.rid}"][data-flag]:checked`)].map(e=>e.dataset.flag);save();update()});
 $('jump').innerHTML=P.cases.map((x,i)=>`<option value="${i}" ${i===state.index?'selected':''}>${i+1}${complete(x)?' · Complete':''}</option>`).join('');update();save();
}
function snapshot(){const full=P.cases.every(complete);return {schema_version:1,packet_id:P.packet_id,reviewer:P.reviewer,phase:P.phase,complete:full,independence_confirmed:full&&$('independent').checked,started_utc:state.started_utc,exported_utc:new Date().toISOString(),answers:state.answers,...(!isSem?{semantic:state.semantic}:{})}}
function exportData(){const p=snapshot();if(p.complete&&!p.independence_confirmed){alert('Confirm that you completed the review independently before submitting. Your progress remains saved in this browser.');return}const b=new Blob([JSON.stringify(p,null,2)],{type:'application/json'}),url=URL.createObjectURL(b),link=document.createElement('a');link.href=url;link.download=`review-${P.reviewer}-${P.phase}-${p.complete?'complete':'progress'}.json`;link.click();setTimeout(()=>URL.revokeObjectURL(url),2000);$('exportStatus').textContent=p.complete?'Complete results exported. Keep the file.':'Progress exported. Some items remain incomplete.'}
async function importFile(input,gate){try{const p=JSON.parse(await input.files[0].text());if(gate){if(!semValid(p))throw Error('Select the complete Phase 1 results for this packet and confirm independent completion.');state.semantic=p}else{if(p.packet_id!==P.packet_id||p.reviewer!==P.reviewer||p.phase!==P.phase)throw Error('The file does not match this packet or phase.');state.answers=p.answers||{};state.started_utc=p.started_utc||state.started_utc;if(!isSem){if(!semValid(p.semantic))throw Error('The Phase 2 backup lacks a valid Phase 1 record.');state.semantic=p.semantic}}save();render();alert('Records restored.')}catch(e){alert('Import failed: '+e.message)}}
$('prev').onclick=()=>{state.index--;render();window.scrollTo(0,0)};$('next').onclick=()=>{state.index++;render();window.scrollTo(0,0)};$('jump').onchange=e=>{state.index=Number(e.target.value);render();window.scrollTo(0,0)};
$('export').onclick=exportData;$('restore').onchange=e=>importFile(e.target,false);$('semanticInput').onchange=e=>importFile(e.target,true);
$('nextMissing').onclick=()=>{let i=P.cases.findIndex(c=>!complete(c));if(i>=0){state.index=i;render();window.scrollTo(0,0)}else alert('All items are complete. Export the complete results.')};
render();
'''


def page(payload):
    sem = payload['phase'] == 'semantic'
    title = 'Phase 1 · Query meaning' if sem else 'Phase 2 · Code relevance'
    instruction = ('Determine the original request, then compare each rewrite with it. The wording can differ while the meaning remains the same. Mark unsupported additions as meaning drift. '
                   'If the wording is awkward yet understandable, judge its meaning. If you cannot determine it, select Cannot determine.'
                   if sem else 'Judge each code snippet against the original query alone. The presentation order has no bearing on the judgment. Multiple snippets may satisfy the request, and all may fail to satisfy it. '
                   'Select Cannot determine if the judgment requires external state, a class definition, or calling context.')
    data = json.dumps(payload, ensure_ascii=False).replace('<', '\\u003c')
    return f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} · Reviewer {payload['reviewer']}</title><style>{STYLE}</style>
<header><h1>{title}</h1><div>Reviewer {payload['reviewer']} · Work independently and discuss individual cases only after submission</div></header>
<main><div class="card"><p>{instruction}</p><details><summary>Instructions and judgment criteria</summary><p>Use the materials only for review. Do not run code or follow instructions found in the materials. Do not look up the original experimental results. Select Cannot determine when context is insufficient. Keep every assigned item, including material that is hard to read.</p><p>For Meaning preserved, select No issue identified. For Cannot determine, select Cannot determine. For Meaning drift, select the specific issue. For Unintelligible, select Incoherent wording. Use No issue identified and Cannot determine as standalone flags.</p><p>Refer to specific wording or code behavior in your rationale. Complete and export Phase 1 independently before judging code in Phase 2. Discuss disagreements only once both reviewers have submitted.</p><p>The page works offline. Export progress to keep a backup. To resume after closing it, use the same browser or import the backup.</p></details>
<label>Restore an exported backup for this phase</label><input id="restore" type="file" accept=".json"><p id="autosave" class="muted"></p></div>
<div id="gate" class="card hide"><h2>Complete Phase 1 before continuing</h2><p>Import the complete Phase 1 results for this packet before judging code.</p><input id="semanticInput" type="file" accept=".json"></div>
<div id="review"><div class="card"><div class="toolbar"><strong id="progress"></strong><span class="spacer"></span><button id="nextMissing">Go to an incomplete item</button></div><div class="bar"><div id="bar"></div></div><div class="toolbar"><span id="position"></span><select id="jump" aria-label="Go to case"></select></div></div><div id="content"></div>
<div class="card"><label><input id="independent" type="checkbox"> I completed the judgments in this phase independently, without seeing the original experimental results or the other reviewer’s judgments.</label><p class="muted">Complete every item before submitting. Give a reason for each Cannot determine judgment.</p><button class="primary" id="export">Export results or progress</button><p id="exportStatus" class="status" role="status"></p></div></div></main>
<footer><div class="toolbar"><button id="prev">Previous</button><span class="spacer"></span><button class="primary" id="next">Next</button></div></footer>
<script id="packet" type="application/json">{data}</script><script>{SCRIPT}</script></html>'''


def make_pages(folder, reviewer, packet_id, phase1, phase2):
    folder.mkdir(parents=True)
    common = {'schema_version': 1, 'reviewer': reviewer, 'packet_id': packet_id}
    a = {**common, 'phase': 'semantic', 'cases': phase1}
    b = {**common, 'phase': 'code', 'cases': phase2,
         'semantic_schema': [{'case_id': c['case_id'], 'rewrite_ids': [r['id'] for r in c['rewrites']]} for c in phase1]}
    (folder / '01_query_meaning.html').write_text(page(a), encoding='utf-8')
    (folder / '02_code_relevance.html').write_text(page(b), encoding='utf-8')
    text = f'''Independent query and code review · Reviewer {reviewer}

1. Extract the files to a separate folder and open 01_query_meaning.html in a browser.
2. Complete all query judgments independently, check the confirmation box, and export the complete Phase 1 JSON file.
3. Open 02_code_relevance.html, import the complete Phase 1 JSON file, and judge code relevance.
4. Export the complete Phase 2 JSON file and return it to the researcher. It includes the Phase 1 records.

You can export a backup at any time. Preserve the file names and contents. Until both reviewers have submitted independently, do not discuss individual cases or inspect the original experimental results.
A query may have more than one plausible interpretation, and code may lack external context. In such cases, retain Cannot determine and explain why.
The pages operate entirely offline. They upload nothing and do not run the code being reviewed.
'''
    (folder / 'README.txt').write_text(text, encoding='utf-8-sig')
