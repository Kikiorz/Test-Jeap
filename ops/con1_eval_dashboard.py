"""Read-only full-Plus progress view; no model imports or GPU work."""
from collections import defaultdict
import json
import os
from pathlib import Path
import time

ROOT = Path('/workspace/artifacts/research_reports/con/con1_full_plus_10k_15k_20260909')


def read_json(path, default):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def snapshot(root=ROOT):
    status = read_json(root / 'status.json', {})
    alive = False
    if isinstance(status.get('pid'), int):
        try:
            command = Path(f"/proc/{status['pid']}/cmdline").read_bytes()
            alive = b'run_con1_full_plus.py' in command
        except OSError:
            pass
    manifest = read_json(root / 'panels.json', {})
    expected_categories = defaultdict(int)
    expected_suites = {}
    for suite, rows in manifest.get('final', {}).items():
        expected_suites[suite] = len(rows)
        for row in rows:
            expected_categories[row['category']] += 1
    result = dict(server_time=time.time(), status=status, process_alive=alive,
                  expected_categories=dict(expected_categories), expected_suites=expected_suites, models={})
    for model in ('10k', '15k'):
        folder = root / model
        progress = read_json(folder / 'progress.json', {})
        summary = read_json(folder / 'summary.json', {})
        data = summary or progress
        groups = data.get('groups', {})
        categories, suites = {}, {}
        for key, group in groups.items():
            suite, category = key.split('/', 1)
            for bucket, name in ((categories, category), (suites, suite)):
                row = bucket.setdefault(name, dict(count=0, success=0, failure=0, error=0))
                for field in row:
                    row[field] += group.get(field, 0)
        count = data.get('completed_episodes', data.get('completed', 0))
        success = data.get('successes', 0)
        result['models'][model] = dict(completed=count, successes=success,
            failures=sum(g.get('failure', 0) for g in groups.values()),
            errors=sum(g.get('error', 0) for g in groups.values()),
            success_rate=success / count if count else None, expected=10030,
            complete=bool(summary.get('complete')), updated=progress.get('unix_time'),
            categories=categories, suites=suites, workers=progress.get('workers', []),
            slots_per_gpu=progress.get('slots_per_gpu', [1]*4))
    result['paired'] = read_json(root / 'paired_15k_vs_10k.json', None)
    return result


PAGE = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Con1 · LIBERO-Plus 全量评测</title>
<style>body{font:16px system-ui;background:#f4f6f8;color:#172b3a;max-width:1100px;margin:30px auto;padding:20px}h1{font-size:26px}section{background:white;padding:22px;border-radius:12px;margin:20px 0}table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:11px 8px;border-bottom:1px solid #e3e8ed}small,.muted{color:#617282}.bad{color:#b32222}progress{width:100%;height:15px}a{color:#1869ab}.scroll{overflow-x:auto}</style>
<h1>Con1 · LIBERO-Plus 全量评测</h1>
<p>10k → 15k，四卡多环境动态队列，每批 32 条，每卡共享一份常驻模型；每模型 10,030 条，共 20,060 回合。仅 Plus，训练保持停止。</p>
<p class="muted">待领取任务按 L5 → L4 → L3 → L2 → L1 → 未标难度分配；同一难度内优先 Camera / Robot / Noise / Layout。已领取批次继续执行，完成顺序不严格按难度。同一清单、种子基数 743、每任务 1 回合；每 10 秒刷新。</p>
<p id="state">正在读取…</p><p id="refresh" class="muted"></p>
<section><h2>总体进度</h2><div class="scroll"><table><thead><tr><th>模型</th><th>完成</th><th>成功</th><th>失败</th><th>错误</th><th>成功率</th><th>状态</th></tr></thead><tbody id="models"></tbody></table></div><p id="total"></p><progress id="bar" max="20060" value="0"></progress></section>
<section><h2>按干扰类别</h2><p class="muted">每格：成功 / 已完成（成功率）；未完成时两模型样本集合可能不同，不能直接视为配对差异。</p><div class="scroll"><table><thead><tr><th>类别</th><th>每模型总数</th><th>10k</th><th>15k</th></tr></thead><tbody id="categories"></tbody></table></div></section>
<section><h2>GPU 当前任务</h2><div class="scroll"><table><thead><tr><th>GPU</th><th>模型 / 套件</th><th>类别</th><th>本批条数</th></tr></thead><tbody id="workers"></tbody></table></div></section>
<section><h2>任务套件汇总（不再固定分卡）</h2><div class="scroll"><table><thead><tr><th>套件</th><th>每模型总数</th><th>10k</th><th>15k</th></tr></thead><tbody id="suites"></tbody></table></div></section>
<section><h2>最终配对比较</h2><p id="paired">两轮全部完成后生成 15k 相对 10k 的改善/退化统计。</p></section>
<p class="muted">全量集包含此前用于检查点选择的任务，不是完全独立的留出集。错误单独计数，成功率分母包含已完成的错误回合；存在错误时不标记完整完成。</p>
<p><a href="/training">历史训练监控</a> · <a href="/api/full-plus">原始实时 JSON</a></p>
<script>
const names={'Background Textures':'其他 · 背景纹理','Robot Initial States':'Robot · 机器人初始状态','Camera Viewpoints':'Camera · 相机视角','Language Instructions':'其他 · 语言指令','Sensor Noise':'Noise · 传感器噪声','Objects Layout':'Layout · 物体布局','Light Conditions':'其他 · 光照'};
const pct=(s,n)=>n?(100*s/n).toFixed(2)+'%':'—';
const cell=g=>g&&g.count?`${g.success} / ${g.count}（${pct(g.success,g.count)}）${g.error?'；错误 '+g.error:''}`:'等待';
const stamp=t=>t?new Date(t*1000).toLocaleString('zh-CN',{timeZone:'Asia/Shanghai',hour12:false})+' 北京时间':'尚无';
function row(target,values){const tr=document.createElement('tr');for(const v of values){const td=document.createElement('td');td.textContent=v;tr.appendChild(td)}document.getElementById(target).appendChild(tr)}
async function refresh(){try{
 const response=await fetch('/api/full-plus',{cache:'no-store'});if(!response.ok)throw Error('HTTP '+response.status);const d=await response.json();
 const s=d.status;const labels={running:'运行中',starting:'正在启动',model_complete:'模型测试完成，准备下一阶段',complete:'全部完成',error:'异常停止'};
 document.getElementById('state').textContent=`调度：${labels[s.state]||'等待启动'}；当前：${s.model||'—'}；进程：${d.process_alive?'在线':'已退出/未启动'}${s.error?'；'+s.error:''}`;
 document.getElementById('state').className=s.state==='error'?'bad':'';
 document.getElementById('refresh').textContent='页面读取：'+stamp(d.server_time)+'；调度心跳：'+stamp(s.unix_time);
 for(const id of ['models','categories','suites','workers'])document.getElementById(id).replaceChildren();
 let total=0;for(const label of ['10k','15k']){const m=d.models[label];total+=m.completed;row('models',[label,m.completed+' / '+m.expected,m.successes,m.failures,m.errors,pct(m.successes,m.completed),m.complete?'完成':s.model===label&&d.process_alive?'进行中':'未完成 / 排队']);}
 document.getElementById('total').textContent=`合计 ${total} / 20060（${pct(total,20060)}）`;
 document.getElementById('bar').value=total;
 const categoryOrder=['Camera Viewpoints','Robot Initial States','Sensor Noise','Objects Layout','Background Textures','Language Instructions','Light Conditions'];
 for(const key of categoryOrder)row('categories',[names[key],d.expected_categories[key]||'—',cell(d.models['10k'].categories[key]),cell(d.models['15k'].categories[key])]);
 for(let gpu=0;gpu<4;gpu++){const model=d.models[s.model];const active=(model?.workers||[]).filter(w=>w.gpu===gpu);const limit=model?.slots_per_gpu?.[gpu]||1;if(!active.length)row('workers',[gpu+'（并发上限 '+limit+'）','空闲 / 启动 / 已结束','—','—']);for(const w of active)row('workers',[gpu+' / 环境 '+(w.slot??0)+'（上限 '+limit+'）',s.model+' / '+w.suite,names[w.category],w.tasks]);}
 const order=['libero_10','libero_goal','libero_object','libero_spatial'];for(const k of order)row('suites',[k,d.expected_suites[k]||'—',cell(d.models['10k'].suites[k]),cell(d.models['15k'].suites[k])]);
 if(d.paired){const p=d.paired.groups.all;document.getElementById('paired').textContent=`15k 相对 10k：改善 ${p.gain}，退化 ${p.regression}，两者成功 ${p.both_success}，两者失败 ${p.both_failure}，错误 ${p.error}；成功率差 ${(100*p.success_rate_delta).toFixed(2)} 个百分点。`;}
 }catch(e){document.getElementById('state').textContent='刷新失败（下次自动重试）：'+e;document.getElementById('state').className='bad';}finally{setTimeout(refresh,10000)}}refresh();
</script></html>'''
