#!/usr/bin/env python3
"""Finish the fixed calibration checks, without using formal prompts for tuning."""
import json
from pathlib import Path
import subprocess
import sys
from experiment import ROOT, BASE, digest
from check_logits import check

SYSTEM_ARGS={
    'layer':['--ngl','29'],
    'atsinfer':['--ngl','99','--config',str(BASE/'layout16384-f32/static.json')],
    'powerinfer':['--vram','18'],
}

def job(tag,system,mode,inputs,batch=512):
    if system=='atsinfer': tag+='-16384'
    dest=BASE/tag
    if not dest.exists():
        cmd=[sys.executable,str(ROOT/'scripts/powerinfer/experiment.py'),'--tag',tag,'--engine',system,
            '--mode',mode,'--input',str(BASE/'inputs'/inputs),'--batch',str(batch),*SYSTEM_ARGS[system]]
        subprocess.run(cmd,cwd=ROOT,check=True)
    result=json.loads((dest/'run.json').read_text())
    if result['exit_code'] or not result['complete'] or not result['within_cap']:
        raise ValueError(f'Calibration failed: {tag}')
    if result['gpu_observed_peak_bytes']>(20<<30)-(256<<20):
        raise ValueError(f'Insufficient GPU margin: {tag}')
    return dest

def main():
    # All-logits quality evaluation has a larger output workspace than generation.
    # Use calibration text, never a formal WikiText window, to test that capacity.
    with (BASE/'inputs/calibration.jsonl').open() as f:
        calibration=[json.loads(s) for s in f]
    probe=next(r for r in calibration if r['scenario']=='L')
    probe={**probe,'id':'quality-resource-calibration','tokens':probe['tokens'][:1024]}
    (BASE/'inputs/quality-resource.jsonl').write_text(json.dumps(probe)+'\n')
    for system in SYSTEM_ARGS:
        job('calibration-quality-resource-'+system,system,'quality','quality-resource.jsonl')
    for system,tag in [('layer','calibration-layer-ngl29-f32-logits'),('atsinfer','calibration-f32-logits')]:
        dest=job(tag,system,'logits','calibration.jsonl')
        report=check(BASE/'reference/reference.json',dest/'results.jsonl')
        (dest/'correctness.json').write_text(json.dumps(report,indent=2)+'\n')
        if not report['passed']: raise ValueError(f'Final {system} configuration failed independent reference')
    for system in ('layer','atsinfer'):
        job('calibration-final-'+system,system,'perf','calibration.jsonl')
    diagnostics=[]
    for system in SYSTEM_ARGS:
        by_batch={}
        for batch in (512,1):
            dest=job(f'calibration-quality-{system}-b{batch}',system,'quality','quality-check.jsonl',batch)
            rows=[json.loads(s) for s in (dest/'results.jsonl').open()]
            by_batch[batch]={r['id']:r for r in rows if r['type']=='quality'}
        if set(by_batch[1])!=set(by_batch[512]): raise ValueError('Quality diagnostic ID mismatch')
        differences=[]
        for ident,r in by_batch[512].items():
            other=by_batch[1][ident]
            if r['scored_tokens']!=31 or other['scored_tokens']!=31: raise ValueError('Invalid diagnostic token counts')
            differences.append({'id':ident,'nll_per_token_batch512':r['nll_sum']/31,
                'nll_per_token_batch1':other['nll_sum']/31,
                'absolute_difference':abs(r['nll_sum']-other['nll_sum'])/31})
        diagnostics.append({'system':system,'requests':differences,
            'needs_decode_quality_followup':any(r['absolute_difference']>0.01 for r in differences)})
    (BASE/'quality-path-diagnostic.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    print(json.dumps(diagnostics,indent=2),flush=True)
    if any(r['needs_decode_quality_followup'] for r in diagnostics):
        raise ValueError('Batch-vs-decode quality difference needs investigation before freeze')
    print('All remaining calibration gates passed.',flush=True)

if __name__=='__main__': main()
