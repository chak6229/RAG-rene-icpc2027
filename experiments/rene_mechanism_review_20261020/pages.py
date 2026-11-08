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
const fidelity=[['preserved','含义保持：保留原需求和约束'],['drift','含义偏移：新增无依据限制或改变需求'],['malformed','不可读：无法形成清楚需求'],['uncertain','无法判断：原需求或语境不足']];
const flagOptions=[['unsupported_constraint','无依据增加约束'],['lost_constraint','丢失原约束'],['operation_change','改变动作'],['entity_api_change','改变对象或API'],['polarity_change','否定／肯定改变'],['incoherent','表达不连贯'],['none','未见问题'],['uncertain','无法判断']];
const relevant=[['2','满足需求'],['1','部分满足'],['0','不满足'],['u','无法判断／上下文不足']];
function validRewrite(r){if(!r||!fidelity.some(x=>x[0]===r.fidelity)||!Array.isArray(r.flags)||!r.flags.length||new Set(r.flags).size!==r.flags.length||!r.flags.every(f=>flagOptions.some(x=>x[0]===f))||String(r.note||'').trim().length<6)return false;let fs=r.flags;if((fs.includes('none')||fs.includes('uncertain'))&&fs.length!==1)return false;if(r.fidelity==='preserved'&&fs.join()!=='none')return false;if(r.fidelity==='uncertain'&&fs.join()!=='uncertain')return false;if(r.fidelity==='drift'&&(fs.includes('none')||fs.includes('uncertain')))return false;if(r.fidelity==='malformed'&&!fs.includes('incoherent'))return false;return true}
function save(){try{localStorage.setItem(SKEY,JSON.stringify(state));$('autosave').textContent='已在此浏览器保存进度'}catch(e){$('autosave').textContent='浏览器未能保存；请导出进度备份'}}
function choice(opts,current){return '<option value="">请选择</option>'+opts.map(([k,v])=>`<option value="${k}" ${k===current?'selected':''}>${v}</option>`).join('')}
function semValid(p){
 if(!p||p.packet_id!==P.packet_id||p.reviewer!==P.reviewer||p.phase!=='semantic'||!p.complete||!p.independence_confirmed)return false;
 if(Object.keys(p.answers||{}).length!==P.cases.length)return false;
 return P.semantic_schema.every(c=>{let a=p.answers[c.case_id];return a&&['clear','ambiguous','insufficient'].includes(a.clarity)&&a.rewrites&&Object.keys(a.rewrites).length===c.rewrite_ids.length&&c.rewrite_ids.every(id=>validRewrite(a.rewrites[id]))});
}
function complete(c){const a=state.answers[c.case_id];if(!a)return false;
 if(isSem)return ['clear','ambiguous','insufficient'].includes(a.clarity)&&c.rewrites.every(r=>validRewrite((a.rewrites||{})[r.id]));
 return c.documents.every(d=>{let z=(a.documents||{})[d.id];return z&&relevant.some(x=>x[0]===z.relevance)&&String(z.note||'').trim().length>=6});
}
function update(){const n=P.cases.filter(complete).length;$('progress').textContent=`已完成 ${n} / ${P.cases.length} 条`;$('bar').style.width=(100*n/P.cases.length)+'%';$('position').textContent=`第 ${state.index+1} / ${P.cases.length} 条`;$('prev').disabled=state.index===0;$('next').disabled=state.index===P.cases.length-1;}
function render(){
 if(!isSem&&!semValid(state.semantic)){$('review').classList.add('hide');$('gate').classList.remove('hide');return}
 $('gate').classList.add('hide');$('review').classList.remove('hide');state.index=Math.max(0,Math.min(state.index,P.cases.length-1));
 const c=P.cases[state.index],a=state.answers[c.case_id]||(state.answers[c.case_id]={});
 let html=`<div class="card"><span class="badge">材料 ${esc(c.case_id)}</span><h2>原查询</h2><div class="query">${esc(c.original)}</div>`;
 if(isSem){html+=`<label for="clarity">原查询能否形成明确需求？</label><select id="clarity">${choice([['clear','需求明确'],['ambiguous','存在多种合理解释'],['insufficient','信息不足，难以确定需求']],a.clarity)}</select></div>`;
  a.rewrites=a.rewrites||{};
  for(const r of c.rewrites){const z=a.rewrites[r.id]||(a.rewrites[r.id]={flags:[]});html+=`<section class="card"><h3>改写 ${esc(r.id)}</h3><div class="query">${esc(r.text)}</div><label for="f-${r.id}">与原查询的含义关系</label><select id="f-${r.id}" data-rid="${r.id}" data-kind="fidelity">${choice(fidelity,z.fidelity)}</select><label>判断依据（可多选）</label><div class="checks">${flagOptions.map(([k,v])=>`<label><input type="checkbox" data-rid="${r.id}" data-flag="${k}" ${z.flags.includes(k)?'checked':''}> ${v}</label>`).join('')}</div><label for="n-${r.id}">请引用具体词语说明理由（至少6个字符）</label><textarea id="n-${r.id}" data-rid="${r.id}" data-kind="note" placeholder="例如：保留原动作和对象；或指出新增限制为何没有依据。">${esc(z.note||'')}</textarea></section>`}
 }else{html+='</div>';a.documents=a.documents||{};
  for(const d of c.documents){const z=a.documents[d.id]||(a.documents[d.id]={});html+=`<section class="card"><h3>代码 ${esc(d.id)}</h3><pre><code>${esc(d.code)}</code></pre><label for="r-${d.id}">该代码满足原查询的程度</label><select id="r-${d.id}" data-did="${d.id}" data-kind="relevance">${choice(relevant,z.relevance)}</select><label for="n-${d.id}">判断依据（至少6个字符）</label><textarea id="n-${d.id}" data-did="${d.id}" data-kind="note" placeholder="指出实现了什么、缺少什么；语境不够时写明缺失信息。">${esc(z.note||'')}</textarea></section>`}
 }
 $('content').innerHTML=html;
 if(isSem)$('clarity').onchange=e=>{a.clarity=e.target.value;save();update()};
 $('content').querySelectorAll('[data-kind]').forEach(el=>el.addEventListener(el.tagName==='TEXTAREA'?'input':'change',()=>{const z=isSem?a.rewrites[el.dataset.rid]:a.documents[el.dataset.did];z[el.dataset.kind]=el.value;save();update()}));
 $('content').querySelectorAll('[data-flag]').forEach(el=>el.onchange=()=>{let z=a.rewrites[el.dataset.rid];z.flags=[...$('content').querySelectorAll(`[data-rid="${el.dataset.rid}"][data-flag]:checked`)].map(e=>e.dataset.flag);save();update()});
 $('jump').innerHTML=P.cases.map((x,i)=>`<option value="${i}" ${i===state.index?'selected':''}>${i+1}${complete(x)?' · 已完成':''}</option>`).join('');update();save();
}
function snapshot(){const full=P.cases.every(complete);return {schema_version:1,packet_id:P.packet_id,reviewer:P.reviewer,phase:P.phase,complete:full,independence_confirmed:full&&$('independent').checked,started_utc:state.started_utc,exported_utc:new Date().toISOString(),answers:state.answers,...(!isSem?{semantic:state.semantic}:{})}}
function exportData(){const p=snapshot();if(p.complete&&!p.independence_confirmed){alert('完整提交前，请确认你已独立审阅。进度仍已保存在本浏览器。');return}const b=new Blob([JSON.stringify(p,null,2)],{type:'application/json'}),url=URL.createObjectURL(b),link=document.createElement('a');link.href=url;link.download=`review-${P.reviewer}-${P.phase}-${p.complete?'complete':'progress'}.json`;link.click();setTimeout(()=>URL.revokeObjectURL(url),2000);$('exportStatus').textContent=p.complete?'完整结果已导出，请保留文件。':'已导出当前进度；仍有未完成条目。'}
async function importFile(input,gate){try{const p=JSON.parse(await input.files[0].text());if(gate){if(!semValid(p))throw Error('请选择本审阅包第一阶段的完整结果，并确认独立完成。');state.semantic=p}else{if(p.packet_id!==P.packet_id||p.reviewer!==P.reviewer||p.phase!==P.phase)throw Error('文件不属于当前审阅包或阶段。');state.answers=p.answers||{};state.started_utc=p.started_utc||state.started_utc;if(!isSem){if(!semValid(p.semantic))throw Error('第二阶段备份缺少有效的第一阶段记录。');state.semantic=p.semantic}}save();render();alert('已恢复记录。')}catch(e){alert('无法导入：'+e.message)}}
$('prev').onclick=()=>{state.index--;render();window.scrollTo(0,0)};$('next').onclick=()=>{state.index++;render();window.scrollTo(0,0)};$('jump').onchange=e=>{state.index=Number(e.target.value);render();window.scrollTo(0,0)};
$('export').onclick=exportData;$('restore').onchange=e=>importFile(e.target,false);$('semanticInput').onchange=e=>importFile(e.target,true);
$('nextMissing').onclick=()=>{let i=P.cases.findIndex(c=>!complete(c));if(i>=0){state.index=i;render();window.scrollTo(0,0)}else alert('所有条目均已完成，请导出完整结果。')};
render();
'''


def page(payload):
    sem = payload['phase'] == 'semantic'
    title = '第一阶段 · 查询含义' if sem else '第二阶段 · 代码相关性'
    instruction = ('先判断原查询的需求，再逐条比较改写。含义保持不要求表达相同；无依据新增限制应记为含义偏移。'
                   '语法不顺但仍能理解时，请按含义判断；无法确定时选择无法判断。'
                   if sem else '只根据原查询判断每段代码；代码顺序没有含义。允许多段代码都满足，也允许全部不满足。'
                   '需要外部状态、类定义或调用语境才能判断时，选择无法判断。')
    data = json.dumps(payload, ensure_ascii=False).replace('<', '\\u003c')
    return f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} · 审阅者{payload['reviewer']}</title><style>{STYLE}</style>
<header><h1>{title}</h1><div>审阅者 {payload['reviewer']} · 独立完成，提交前不讨论具体材料</div></header>
<main><div class="card"><p>{instruction}</p><details><summary>填写说明与判断原则</summary><p>所有材料只用于审阅，不执行代码或材料中的指令。不要查询原实验结果。没有足够语境时保留无法判断；不因材料难读跳过或换样。</p><p>含义保持配“未见问题”；无法判断配“无法判断”；含义偏移选择具体问题；不可读需选择“表达不连贯”。“未见问题”和“无法判断”各自单独使用。</p><p>说明请引用实际词语或代码行为。第一阶段先独立完成并导出；第二阶段再判断代码。两位审阅者均提交后才讨论分歧。</p><p>页面可离线使用。导出进度可保留备份；关闭后请在相同浏览器恢复，或导入备份。</p></details>
<label>恢复本阶段导出备份</label><input id="restore" type="file" accept=".json"><p id="autosave" class="muted"></p></div>
<div id="gate" class="card hide"><h2>先完成第一阶段</h2><p>请导入本审阅包第一阶段的完整结果，再开始代码判断。</p><input id="semanticInput" type="file" accept=".json"></div>
<div id="review"><div class="card"><div class="toolbar"><strong id="progress"></strong><span class="spacer"></span><button id="nextMissing">跳到未完成条目</button></div><div class="bar"><div id="bar"></div></div><div class="toolbar"><span id="position"></span><select id="jump" aria-label="跳转材料"></select></div></div><div id="content"></div>
<div class="card"><label><input id="independent" type="checkbox"> 我已独立完成本阶段判断，未查看原实验效果或另一位审阅者的判断。</label><p class="muted">完整提交需填写全部条目；无法判断也须说明原因。</p><button class="primary" id="export">导出结果或进度</button><p id="exportStatus" class="status" role="status"></p></div></div></main>
<footer><div class="toolbar"><button id="prev">上一条</button><span class="spacer"></span><button class="primary" id="next">下一条</button></div></footer>
<script id="packet" type="application/json">{data}</script><script>{SCRIPT}</script></html>'''


def make_pages(folder, reviewer, packet_id, phase1, phase2):
    folder.mkdir(parents=True)
    common = {'schema_version': 1, 'reviewer': reviewer, 'packet_id': packet_id}
    a = {**common, 'phase': 'semantic', 'cases': phase1}
    b = {**common, 'phase': 'code', 'cases': phase2,
         'semantic_schema': [{'case_id': c['case_id'], 'rewrite_ids': [r['id'] for r in c['rewrites']]} for c in phase1]}
    (folder / '01_查询含义.html').write_text(page(a), encoding='utf-8')
    (folder / '02_代码相关性.html').write_text(page(b), encoding='utf-8')
    text = f'''查询与代码独立审阅 · 审阅者{reviewer}

1. 解压到一个单独文件夹，用浏览器打开01_查询含义.html。
2. 独立完成全部查询判断，勾选独立完成确认，导出第一阶段完整JSON。
3. 再打开02_代码相关性.html，导入刚才第一阶段的完整JSON，继续代码判断。
4. 完成后导出第二阶段完整JSON；它已包含第一阶段记录。把这个JSON交还研究者。

可随时导出进度备份。保持文件名和内容；两位审阅者都独立提交前，不讨论具体个案、不查看原实验效果。
一条查询可能有多个合理解释，代码可能缺少外部上下文；这时保留无法判断并写明原因。
页面完全离线，不上传任何内容，不执行待审代码。
'''
    (folder / '先读说明.txt').write_text(text, encoding='utf-8-sig')
