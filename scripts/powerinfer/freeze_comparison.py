#!/usr/bin/env python3
"""Freeze validated execution artifacts; refuse incomplete calibration."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from experiment import BASE, ROOT, ORIGINAL, digest, write

def main():
    target=BASE/'protocol.json'
    if target.exists(): raise ValueError('Protocol already frozen; inspect before replacing it')
    checks=['calibration-layer-ngl29-f32-logits/correctness.json','calibration-f32-logits-16384/correctness.json']
    for path in checks:
        if not json.loads((BASE/path).read_text())['passed']: raise ValueError('Correctness gate failed')
    if not json.loads((BASE/'inputs/verification.json').read_text())['passed']: raise ValueError('Input verification failed')
    diagnostic=json.loads((BASE/'quality-path-diagnostic.json').read_text())
    if len(diagnostic)!=3 or any(r['needs_decode_quality_followup'] for r in diagnostic): raise ValueError('Quality-path diagnostic needs followup')
    calibration={}
    for system,tag in [('layer','calibration-final-layer'),('atsinfer','calibration-final-atsinfer-16384'),('powerinfer','calibration-powerinfer18-v2')]:
        r=json.loads((BASE/tag/'run.json').read_text())
        if r['exit_code'] or not r['complete'] or r['completed_records']!=4 or r['gpu_observed_peak_bytes']>(20<<30)-(256<<20): raise ValueError(f'Incomplete calibration: {tag}')
        calibration[system]={'directory':tag,**r}
    index=BASE/'assets/llama-13b-relu.powerinfer.gguf.generated.gpuidx'
    for system in ('layer','atsinfer','powerinfer'):
        suffix='-16384' if system=='atsinfer' else ''
        r=json.loads((BASE/('calibration-quality-resource-'+system+suffix)/'run.json').read_text())
        if r['exit_code'] or not r['complete'] or r['gpu_observed_peak_bytes']>(20<<30)-(256<<20): raise ValueError('Quality workspace exceeds GPU budget or margin')
    if digest(index)!=calibration['powerinfer']['index_after_sha256']: raise ValueError('PowerInfer index changed since complete calibration')
    snapshot=BASE/'frozen-source'; snapshot.mkdir()
    with (snapshot/'tracked.patch').open('w') as f: subprocess.run(['git','diff','--binary'],stdout=f,check=True,cwd=ROOT)
    with (snapshot/'powerinfer.patch').open('w') as f: subprocess.run(['git','-C','/home/lush/PowerInfer','diff','--binary'],stdout=f,check=True)
    for folder in ['scripts/powerinfer','tools/powerinfer-compare']:
        shutil.copytree(ROOT/folder,snapshot/folder,ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(ROOT/'build-release/CMakeCache.txt',snapshot/'CMakeCache.txt')
    files=[ROOT/'build-release/bin/llama-powerinfer-compare',ROOT/'build-release/comparison/powerinfer-compare',
        BASE/'layout16384-f32/static.json',BASE/'profile-f32.json',index,BASE/'inputs/verification.json',
        ROOT/'scripts/powerinfer/experiment.py',ROOT/'scripts/powerinfer/run_suite.py',ROOT/'scripts/powerinfer/summarize_comparison.py']
    files+=list((ROOT/'build-release/bin').glob('*.so'))+list((BASE/'inputs').glob('*.jsonl'))
    files+=[BASE/path for path in checks]
    model_stat=ORIGINAL.stat()
    protocol={'version':1,'frozen_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'gpu_cap_bytes':20<<30,'calibration_margin_bytes':256<<20,'context':2560,'batch':512,'threads':6,'affinity':[0,1,2,3,4,5],
        'kv_dtype':'f16','flash_attention':False,'sampling':'greedy, no penalties, ignore EOS, 128 outputs',
        'scenarios':{'S':[128,128],'L':[2048,128]},'prompts_per_scenario':10,'rounds':3,
        'timing':'TTFT=t1-t0; TPOT=(t128-t1)/127; request=t128-t0; preloaded engine; clear KV before t0',
        'dense_precision':'F16 weights/KV; F16 MUL_MAT requests FP32 accumulation; F16 OUT_PROD uses FP32 accumulation',
        'powerinfer_precision':'native unmodified kernels','powerinfer_sparse_threshold':0.0,
        'model_sha256':json.loads((BASE/'environment.json').read_text())['model_sha256'],
        'model_stat':{'size':model_stat.st_size,'mtime_ns':model_stat.st_mtime_ns},
        'runtime_overrides':{k:v for k,v in os.environ.items() if k.startswith(('LLAMA_','GGML_','CUDA_'))},
        'quality':{'dataset':'WikiText-2 raw-v1 test, fixed subset','windows':32,'tokens_per_window':1024,'scored_tokens':32736,'batch':512},
        'systems':{'layer':['--ngl','29'],'atsinfer':['--ngl','99','--config',str(BASE/'layout16384-f32/static.json')],'powerinfer':['--vram','18']},
        'correctness_reports':[str((BASE/path).relative_to(ROOT)) for path in checks],
        'artifacts':{str(p.relative_to(ROOT)):digest(p) for p in files},'calibration':calibration}
    write(target,protocol); print('Frozen:',target)

if __name__=='__main__': main()
