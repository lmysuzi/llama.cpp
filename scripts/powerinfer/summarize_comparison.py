#!/usr/bin/env python3
"""Validate the complete protocol before computing paired comparison statistics."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics as st

SYSTEMS=('layer','atsinfer','powerinfer')
METRICS=('ttft_s','tpot_s','decode_tps','request_s')

def read(path):
    with Path(path).open() as f: return [json.loads(s) for s in f]

def summarize(base, prefix):
    expected={r['id']:r for r in read(base/'inputs/manifest.jsonl') if r['split']=='test'}
    if len(expected)!=20 or any(sum(r['scenario']==s for r in expected.values())!=10 for s in ('S','L')):
        raise ValueError('Expected ten independent test prompts per scenario')
    records={}; resources=[]; quality={}
    for system in SYSTEMS:
        for round_id in range(3):
            dest=base/f'{prefix}-r{round_id}-{system}'
            run=json.loads((dest/'run.json').read_text())
            if run['exit_code'] or not run['complete'] or not run['within_cap']: raise ValueError(f'Failed run: {dest}')
            rows=read(dest/'results.jsonl')
            samples=[r for r in rows if r.get('type')=='perf' and r['split']=='test']
            if len(samples)!=len(expected) or {r['id'] for r in samples}!=set(expected): raise ValueError('Incomplete prompt matrix')
            for r in samples:
                if r['system']!=system or r['round']!=round_id or r['status']!='ok': raise ValueError('Run identity/status mismatch')
                if r['n_gen']!=128 or len(r['output_tokens'])!=128 or r['n_prompt']!=len(expected[r['id']]['tokens']): raise ValueError('Incorrect token counts')
                times=r['token_monotonic_ns']
                if len(times)!=128 or not all(b>a for a,b in zip(times,times[1:])): raise ValueError('Invalid token timestamps')
                if not math.isclose(r['request_s'],r['ttft_s']+127*r['tpot_s'],abs_tol=1e-8): raise ValueError('Timing identity failed')
                measured={'ttft_s':(times[0]-r['start_monotonic_ns'])*1e-9,
                    'request_s':(times[-1]-r['start_monotonic_ns'])*1e-9,
                    'tpot_s':(times[-1]-times[0])*1e-9/127}
                if any(not math.isclose(r[k],v,abs_tol=1e-8) for k,v in measured.items()): raise ValueError('Timestamp-derived timing mismatch')
                if not math.isclose(r['decode_tps'],1/r['tpot_s'],rel_tol=1e-8): raise ValueError('Decode rate mismatch')
                for k in METRICS:
                    if not math.isfinite(r[k]) or r[k]<=0: raise ValueError('Invalid timing')
                records[system,round_id,r['id']]=r
            metadata=next(r for r in rows if r['type']=='metadata')
            first_formal=min(r['start_monotonic_ns'] for r in samples)
            phase_peaks={'initialization':0,'warmup':0,'measurement':0}; other_min=math.inf; other_max=0
            with (dest/'telemetry.csv').open() as f:
                for t in csv.DictReader(f):
                    when=int(t['monotonic_ns']); gpu=int(t['gpu_bytes'])
                    phase='initialization' if when<metadata['start_monotonic_ns'] else ('warmup' if when<first_formal else 'measurement')
                    phase_peaks[phase]=max(phase_peaks[phase],gpu)
                    other_min=min(other_min,int(t['other_gpu_bytes'])); other_max=max(other_max,int(t['other_gpu_bytes']))
            resources.append({'system':system,'round':round_id,**run,'phase_peak_bytes':phase_peaks,
                'other_gpu_bytes_min':other_min,'other_gpu_bytes_max':other_max})
        qdir=base/f'{prefix}-quality-{system}'
        qrun=json.loads((qdir/'run.json').read_text())
        if qrun['exit_code'] or not qrun['complete'] or not qrun['within_cap']: raise ValueError('Failed quality run')
        qrows=[r for r in read(qdir/'results.jsonl') if r.get('type')=='quality']
        if any(r['system']!=system or r['status']!='ok' for r in qrows): raise ValueError('Quality identity/status mismatch')
        qexpected={r['id'] for r in read(base/'inputs/quality.jsonl')}
        if len(qrows)!=32 or {r['id'] for r in qrows}!=qexpected or any(r['scored_tokens']!=1023 for r in qrows): raise ValueError('Incomplete quality set')
        if any(not math.isfinite(r['nll_sum']) or r['nll_sum']<0 for r in qrows): raise ValueError('Invalid NLL')
        nll=sum(r['nll_sum'] for r in qrows); count=sum(r['scored_tokens'] for r in qrows)
        quality[system]={'nll_sum':nll,'scored_tokens':count,'ppl':math.exp(nll/count),'gpu_peak_bytes':qrun['gpu_observed_peak_bytes']}
    aggregated=[]; paired=[]
    means={}
    for system in SYSTEMS:
        for ident in expected:
            means[system,ident]={k:st.mean(records[system,r,ident][k] for r in range(3)) for k in METRICS}
        for scenario in ('S','L'):
            ids=[i for i,r in expected.items() if r['scenario']==scenario]
            row={'system':system,'scenario':scenario,'prompts':len(ids),'rounds':3}
            for k in METRICS:
                values=[means[system,i][k] for i in ids]
                row[k]={'mean':st.mean(values),'prompt_sd':st.stdev(values),
                    'round_means':[st.mean(records[system,r,i][k] for i in ids) for r in range(3)]}
            aggregated.append(row)
    for first,second in [('atsinfer','layer'),('powerinfer','layer'),('atsinfer','powerinfer')]:
        for scenario in ('S','L'):
            ids=[i for i,r in expected.items() if r['scenario']==scenario]
            paired.append({'numerator_latency_system':second,'denominator_latency_system':first,'scenario':scenario,
                **{k:math.exp(st.mean(math.log(means[second,i][k]/means[first,i][k]) for i in ids)) for k in ('ttft_s','tpot_s','request_s')}})
    for system in SYSTEMS: quality[system]['delta_ppl_vs_layer']=quality[system]['ppl']-quality['layer']['ppl']
    return {'formal_requests':len(records),'statistics':aggregated,'paired_latency_speedups':paired,'quality':quality,'resources':resources}

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--base',type=Path,default=Path('logs/powerinfer-comparison'))
    p.add_argument('--prefix',default='formal'); p.add_argument('--out',type=Path)
    a=p.parse_args(); result=summarize(a.base,a.prefix)
    out=a.out or a.base/(a.prefix+'-summary.json'); out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='resources'},indent=2))
