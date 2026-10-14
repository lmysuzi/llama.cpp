#!/usr/bin/env python3
"""Run the frozen 3-round matrix, then the fixed quality subset, serially."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from experiment import BASE, ROOT, ORIGINAL, digest

def main():
    p=argparse.ArgumentParser(); p.add_argument('--freeze',type=Path,required=True)
    p.add_argument('--prefix',default='formal'); p.add_argument('--resume',action='store_true')
    a=p.parse_args(); frozen=json.loads(a.freeze.read_text())
    for path,sha in frozen['artifacts'].items():
        if digest(ROOT/path)!=sha: raise ValueError(f'Frozen artifact changed: {path}')
    for path in frozen['correctness_reports']:
        if not json.loads((ROOT/path).read_text())['passed']: raise ValueError(f'Correctness gate failed: {path}')
    stat=ORIGINAL.stat()
    if frozen['model_stat']!={'size':stat.st_size,'mtime_ns':stat.st_mtime_ns}: raise ValueError('Model file changed since freeze')
    overrides={k:v for k,v in os.environ.items() if k.startswith(('LLAMA_','GGML_','CUDA_'))}
    if overrides!=frozen['runtime_overrides']: raise ValueError('Runtime environment overrides changed')
    orders=[['layer','atsinfer','powerinfer'],['atsinfer','powerinfer','layer'],['powerinfer','layer','atsinfer']]
    jobs=[]
    for round_id,systems in enumerate(orders):
        for system in systems:
            jobs.append((f'{a.prefix}-r{round_id}-{system}',system,'perf',f'round-{round_id}.jsonl',round_id))
    for system in orders[0]: jobs.append((f'{a.prefix}-quality-{system}',system,'quality','quality.jsonl',0))
    journal=BASE/(a.prefix+'-suite.jsonl')
    for tag,system,mode,inputs,round_id in jobs:
        command=[sys.executable,str(ROOT/'scripts/powerinfer/experiment.py'),'--tag',tag,'--engine',system,
            '--mode',mode,'--input',str(BASE/'inputs'/inputs),'--round',str(round_id),*frozen['systems'][system]]
        dest=BASE/tag
        if dest.exists():
            if not a.resume: raise ValueError(f'Output already exists: {dest}')
            summary=json.loads((dest/'run.json').read_text())
            if summary['exit_code'] or not summary['complete'] or not summary['within_cap']: raise ValueError(f'Cannot silently skip failed run: {dest}')
            previous=json.loads((dest/'command.json').read_text())
            if previous['input_sha256']!=digest(BASE/'inputs'/inputs): raise ValueError('Resume input changed')
            print('Verified completed run, skipping',tag,flush=True); continue
        with journal.open('a') as f: f.write(json.dumps({'event':'start','tag':tag,'time':time.time(),'argv':command})+'\n')
        rc=subprocess.call(command,cwd=ROOT)
        with journal.open('a') as f: f.write(json.dumps({'event':'finish','tag':tag,'time':time.time(),'exit_code':rc})+'\n')
        if rc: return rc
        index='logs/powerinfer-comparison/assets/llama-13b-relu.powerinfer.gguf.generated.gpuidx'
        if digest(ROOT/index)!=frozen['artifacts'][index]: raise ValueError('Native GPU index changed during formal run')
    return subprocess.call([sys.executable,str(ROOT/'scripts/powerinfer/summarize_comparison.py'),'--prefix',a.prefix],cwd=ROOT)

if __name__=='__main__': sys.exit(main())
