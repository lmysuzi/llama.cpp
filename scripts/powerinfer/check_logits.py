#!/usr/bin/env python3
"""Compare the dense adapter against independently computed reference logits.

Acceptance bounds are fixed before inspecting the adapter's results. Top-1 is
reported diagnostically because near ties need not survive F16 rounding.
"""
import argparse
import json
from pathlib import Path
import numpy as np

RELATIVE_L2_LIMIT = 0.01
MAX_ABS_LIMIT = 0.25

def check(reference, results):
    refs = {r['id']: r for r in json.loads(Path(reference).read_text())}
    rows = [json.loads(s) for s in Path(results).open()]
    records = [r for r in rows if r.get('type') == 'logits']
    if set(refs) != {r['id'] for r in records} or len(records) != len(refs):
        raise ValueError('Missing or duplicate reference/request IDs')
    output = []
    for r in records:
        ref = refs[r['id']]
        if ref['shape'] != r['shape']: raise ValueError('Shape mismatch')
        x = np.fromfile(ref['logits_file'], dtype='<f4').reshape(ref['shape']).astype(np.float64)
        y = np.fromfile(r['logits_file'], dtype='<f4').reshape(r['shape']).astype(np.float64)
        delta = y-x
        relative = np.linalg.norm(delta, axis=1)/np.linalg.norm(x, axis=1)
        absolute = np.max(np.abs(delta), axis=1)
        agreement = np.argmax(x,axis=1)==np.argmax(y,axis=1)
        passed = bool(np.isfinite(y).all() and relative.max() <= RELATIVE_L2_LIMIT and absolute.max() <= MAX_ABS_LIMIT)
        output.append({'id':r['id'], 'max_row_relative_l2':float(relative.max()),
            'max_abs':float(absolute.max()), 'top1_agreement':float(agreement.mean()),
            'row_relative_l2':relative.tolist(), 'row_max_abs':absolute.tolist(), 'passed':passed})
    return {'limits':{'relative_l2':RELATIVE_L2_LIMIT,'max_abs':MAX_ABS_LIMIT},
        'passed':all(r['passed'] for r in output),'requests':output}

if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--reference',required=True)
    p.add_argument('--results',required=True); p.add_argument('--out',required=True)
    a=p.parse_args(); result=check(a.reference,a.results)
    Path(a.out).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2)); raise SystemExit(0 if result['passed'] else 1)
