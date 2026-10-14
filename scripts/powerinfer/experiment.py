#!/usr/bin/env python3
"""Run one native engine, preserving commands, telemetry, outputs and failures."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'logs/powerinfer-comparison'
ORIGINAL=Path('/home/lush/Models/ReluLLaMA-13B-PowerInfer-GGUF/llama-13b-relu.powerinfer.gguf')

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()

def write(path,obj): Path(path).write_text(json.dumps(obj,indent=2)+'\n')

def assets():
    dest=BASE/'assets'; dest.mkdir(parents=True,exist_ok=True)
    model=dest/ORIGINAL.name
    if not model.exists(): model.symlink_to(ORIGINAL)
    for name in ['activation','config.json']:
        target=dest/name
        if not target.exists(): target.symlink_to(ORIGINAL.parent/name)
    idx=Path(str(model)+'.generated.gpuidx')
    if not idx.exists(): shutil.copy2(str(ORIGINAL)+'.generated.gpuidx',idx)
    return model

def env():
    e=os.environ.copy()
    e['PATH']='/home/lush/miniconda3/bin:'+e.get('PATH','')
    e['PYTHONPATH']='/home/lush/PowerInfer/gguf-py:/home/lush/PowerInfer/powerinfer-py'
    e['OMP_NUM_THREADS']='6'; e['OPENBLAS_NUM_THREADS']='6'; e['MKL_NUM_THREADS']='6'
    return e

def run(a):
    import pynvml as nv
    nv.nvmlInit(); handle=nv.nvmlDeviceGetHandleByIndex(0)
    dest=BASE/a.tag
    dest.mkdir(parents=True,exist_ok=False)
    model=assets()
    binary=ROOT/('build-release/comparison/powerinfer-compare' if a.engine=='powerinfer' else 'build-release/bin/llama-powerinfer-compare')
    command=[str(binary),'--model',str(model),'--mode',a.mode,'--system',a.engine,
        '--input',str(a.input.resolve()),'--output',str(dest/'results.jsonl'),
        '--ngl',str(a.ngl),'--vram',str(a.vram),'--ctx','2560','--batch',str(a.batch),
        '--threads','6','--round',str(a.round),'--gen',str(a.gen)]
    if a.engine=='atsinfer':
        if a.config is None: raise ValueError('ATSInfer needs --config')
        command+=['--pipo','--config',str(a.config.resolve())]
        shutil.copy2(a.config,dest/'config.json')
    if a.reset_index: command+=['--reset-index']
    e=env()
    sparse_env={k:v for k,v in e.items() if k.startswith('LLAMA_SPARSE') or k.startswith('GGML_CUDA')}
    index=Path(str(model)+'.generated.gpuidx')
    write(dest/'command.json',{'argv':command,'affinity':[0,1,2,3,4,5],
        'environment':{'PATH':e['PATH'],'PYTHONPATH':e['PYTHONPATH'],**sparse_env},
        'binary_sha256':digest(binary),'input_sha256':digest(a.input),
        'shared_libraries_sha256':{str(p):digest(p) for p in (ROOT/'build-release/bin').glob('*.so')}
            if a.engine!='powerinfer' else {},
        'index_before_sha256':digest(index) if index.exists() else None,
        'gpu_name':nv.nvmlDeviceGetName(handle),'driver':nv.nvmlSystemGetDriverVersion(),
        'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'start_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    # Keep the observer off the six inference cores.
    os.sched_setaffinity(0,{6,7})
    def bind(): os.sched_setaffinity(0,{0,1,2,3,4,5})
    peak=0; rss_peak=0; samples=0
    with (dest/'stdout.log').open('w') as out,(dest/'stderr.log').open('w') as err,(dest/'telemetry.csv').open('w') as tf:
        p=subprocess.Popen(command,stdout=out,stderr=err,env=e,cwd=ROOT,preexec_fn=bind)
        writer=csv.DictWriter(tf,fieldnames=['monotonic_ns','pid','gpu_bytes','device_bytes','other_gpu_bytes','rss_kib','temperature_c','sm_clock_mhz','mem_clock_mhz','gpu_util'])
        writer.writeheader()
        print(f'{a.tag}: started PID {p.pid}',flush=True)
        heartbeat=time.monotonic()
        while p.poll() is None:
            start=time.monotonic()
            usage={}
            for getter in [nv.nvmlDeviceGetComputeRunningProcesses,nv.nvmlDeviceGetGraphicsRunningProcesses]:
                try:
                    for proc in getter(handle):
                        value=proc.usedGpuMemory
                        if isinstance(value,int) and value<1<<60: usage[proc.pid]=max(usage.get(proc.pid,0),value)
                except nv.NVMLError: pass
            gpu=usage.get(p.pid,0); peak=max(peak,gpu)
            rss=0
            try:
                for line in Path(f'/proc/{p.pid}/status').read_text().splitlines():
                    if line.startswith('VmRSS:'): rss=int(line.split()[1])
            except FileNotFoundError: pass
            rss_peak=max(rss_peak,rss)
            total=nv.nvmlDeviceGetMemoryInfo(handle).used
            writer.writerow({'monotonic_ns':time.monotonic_ns(),'pid':p.pid,'gpu_bytes':gpu,'device_bytes':total,
                'other_gpu_bytes':total-gpu,'rss_kib':rss,'temperature_c':nv.nvmlDeviceGetTemperature(handle,nv.NVML_TEMPERATURE_GPU),
                'sm_clock_mhz':nv.nvmlDeviceGetClockInfo(handle,nv.NVML_CLOCK_SM),
                'mem_clock_mhz':nv.nvmlDeviceGetClockInfo(handle,nv.NVML_CLOCK_MEM),
                'gpu_util':nv.nvmlDeviceGetUtilizationRates(handle).gpu})
            samples+=1
            if time.monotonic()-heartbeat>30:
                tf.flush(); heartbeat=time.monotonic()
                results=dest/'results.jsonl'
                count=sum(1 for _ in results.open())-1 if results.exists() else 0
                print(f'{a.tag}: {max(0,count)} records, GPU {gpu/(1<<30):.2f} GiB, peak {peak/(1<<30):.2f} GiB',flush=True)
            time.sleep(max(0,.02-(time.monotonic()-start)))
        rc=p.wait()
    rows=[json.loads(line) for line in (dest/'results.jsonl').open()] if (dest/'results.jsonl').exists() else []
    complete=bool(rows and rows[-1].get('type')=='complete')
    summary={'exit_code':rc,'complete':complete,'gpu_observed_peak_bytes':peak,'rss_observed_peak_kib':rss_peak,
        'cap_bytes':20<<30,'within_cap':peak<=20<<30,'telemetry_samples':samples,
        'index_after_sha256':digest(index) if index.exists() else None,
        'completed_records':sum(r.get('type') in ('perf','quality','logits') for r in rows)}
    write(dest/'run.json',summary)
    print(json.dumps(summary),flush=True)
    nv.nvmlShutdown()
    return 0 if rc==0 and complete and summary['within_cap'] else 1

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--tag',required=True)
    p.add_argument('--engine',choices=['layer','atsinfer','powerinfer'],required=True)
    p.add_argument('--mode',choices=['perf','quality','logits'],default='perf')
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--config',type=Path)
    p.add_argument('--ngl',type=int,default=0)
    p.add_argument('--vram',type=float,default=18)
    p.add_argument('--gen',type=int,default=128)
    p.add_argument('--batch',type=int,default=512)
    p.add_argument('--round',type=int,default=0)
    p.add_argument('--reset-index',action='store_true')
    sys.exit(run(p.parse_args()))
