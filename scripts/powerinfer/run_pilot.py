#!/usr/bin/env python3
"""Small comparison: two engines, two lengths, three held-out prompts each."""
import json
import math
import statistics as st
import subprocess
import sys
from experiment import BASE, ROOT, digest, write
from summarize_comparison import read

SYSTEMS={
    'atsinfer':['--ngl','99','--config',str(BASE/'layout16384-f32/static.json')],
    'powerinfer':['--vram','18'],
}

def main():
    reference=BASE/'calibration-f32-logits-16384/correctness.json'
    if not json.loads(reference.read_text())['passed']: raise ValueError('Dense correctness gate failed')
    manifest=read(BASE/'inputs/manifest.jsonl')
    warmups=[next(r for r in manifest if r['scenario']==s and r['split']=='calibration') for s in ('L','S')]
    selected={s:[r for r in manifest if r['scenario']==s and r['split']=='test'][:3] for s in ('S','L')}
    requests=warmups+[r for pair in zip(selected['S'],selected['L']) for r in pair]
    inputs=BASE/'inputs/pilot.jsonl'
    content=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in requests)
    if inputs.exists() and inputs.read_text()!=content: raise ValueError('Pilot input already exists with different content')
    inputs.write_text(content)
    files=[inputs,reference,BASE/'layout16384-f32/static.json',BASE/'assets/llama-13b-relu.powerinfer.gguf.generated.gpuidx',
        ROOT/'build-release/bin/llama-powerinfer-compare',ROOT/'build-release/comparison/powerinfer-compare',
        ROOT/'scripts/powerinfer/experiment.py',ROOT/'scripts/powerinfer/run_pilot.py']
    files+=list((ROOT/'build-release/bin').glob('*.so'))
    frozen={str(p.relative_to(ROOT)):digest(p) for p in files}
    protocol=BASE/'pilot-protocol.json'
    if protocol.exists():
        if json.loads(protocol.read_text())['artifacts']!=frozen: raise ValueError('Pilot artifacts changed')
    else:
        write(protocol,{'scope':'pilot, not final paper statistics','systems':SYSTEMS,
            'prompts_per_scenario':3,'rounds':1,'formal_requests':12,'warmups_per_system':2,
            'context':2560,'batch':512,'threads':6,'gpu_cap_bytes':20<<30,
            'artifacts':frozen,'model_sha256':json.loads((BASE/'environment.json').read_text())['model_sha256']})
        with (BASE/'pilot-source.patch').open('w') as f:
            subprocess.run(['git','diff','--binary'],cwd=ROOT,stdout=f,check=True)
    all_rows=[]; resources={}
    for system,args in SYSTEMS.items():
        tag='pilot-'+system; dest=BASE/tag
        if not dest.exists():
            subprocess.run([sys.executable,str(ROOT/'scripts/powerinfer/experiment.py'),'--tag',tag,
                '--engine',system,'--input',str(inputs),*args],cwd=ROOT,check=True)
        run=json.loads((dest/'run.json').read_text())
        if run['exit_code'] or not run['complete'] or not run['within_cap'] or run['completed_records']!=8: raise ValueError(f'Incomplete pilot: {system}')
        resources[system]=run
        rows=[r for r in read(dest/'results.jsonl') if r['type']=='perf' and r['split']=='test']
        if len(rows)!=6 or {r['id'] for r in rows}!={r['id'] for s in selected.values() for r in s}: raise ValueError('Pilot IDs do not match')
        for r in rows:
            if r['system']!=system or r['n_gen']!=128 or len(r['output_tokens'])!=128: raise ValueError('Invalid output count/system')
            if r['n_prompt']!=(128 if r['scenario']=='S' else 2048): raise ValueError('Invalid prompt length')
            times=r['token_monotonic_ns']
            if len(times)!=128 or not all(b>a for a,b in zip(times,times[1:])): raise ValueError('Invalid timestamps')
            if not math.isclose(r['request_s'],(times[-1]-r['start_monotonic_ns'])*1e-9,abs_tol=1e-8): raise ValueError('Request timing mismatch')
            if not math.isclose(r['request_s'],r['ttft_s']+127*r['tpot_s'],abs_tol=1e-8): raise ValueError('Timing identity mismatch')
        all_rows+=rows
        for path,sha in frozen.items():
            if digest(ROOT/path)!=sha: raise ValueError(f'Frozen artifact changed: {path}')
    stats=[]
    for scenario in ('S','L'):
        for system in SYSTEMS:
            rows=[r for r in all_rows if r['scenario']==scenario and r['system']==system]
            stats.append({'scenario':scenario,'system':system,'n':len(rows),
                **{k:{'mean':st.mean(r[k] for r in rows),'sd':st.stdev(r[k] for r in rows)} for k in ('ttft_s','tpot_s','decode_tps','request_s')}})
    write(BASE/'pilot-summary.json',{'scope':'3 prompts per scenario, one run per engine; performance pilot only',
        'requests':12,'statistics':stats,'resources':resources})
    lines=['# ATSInfer 与 PowerInfer：小样本性能对比','',
        '每个场景 3 条相同的留出文本，每个系统运行一次；2 条校准文本预热并排除，共 12 条测量请求。表中为三个 prompt 的算术均值。',
        'ReluLLaMA-13B F16，RTX 4090，6 CPU 线程，context 2560，chunk 512，公共进程显存上限 20 GiB。ATSInfer 为 static+async；PowerInfer 保留原生预测稀疏。','',
        '| 输入/输出 | 系统 | TTFT (s) | TPOT (ms) | Decode (token/s) | 总延迟 (s) | GPU 峰值 (GiB) |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in stats:
        lines.append(f"| {'128' if r['scenario']=='S' else '2048'}/128 | {r['system']} | {r['ttft_s']['mean']:.2f} | {r['tpot_s']['mean']*1000:.1f} | {r['decode_tps']['mean']:.2f} | {r['request_s']['mean']:.2f} | {resources[r['system']]['gpu_observed_peak_bytes']/(1<<30):.2f} |")
    lines+=['','这是用于观察系统取舍的小样本结果，没有重复轮次，不能作为稳定的最终论文统计。',
        '两套系统生成的 continuation 可能不同；性能表没有证明质量等价。模型适配独立校验与正式任务质量评估是不同问题。',
        '逐请求结果、计时、输出 token、命令和显存日志位于 `pilot-atsinfer/` 与 `pilot-powerinfer/`。','']
    (BASE/'pilot-summary.md').write_text('\n'.join(lines))
    print('\n'.join(lines),flush=True)

if __name__=='__main__': main()
