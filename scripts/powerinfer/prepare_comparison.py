#!/usr/bin/env python3
"""Fetch pinned public datasets; construct deterministic real-token manifests."""
import argparse
import hashlib
import json
import random
import subprocess
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEED = 20260916
MODEL = Path('/home/lush/Models/ReluLLaMA-13B-PowerInfer-GGUF/llama-13b-relu.powerinfer.gguf')

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''): h.update(block)
    return h.hexdigest()

def json_write(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')

def jsonl_write(path, rows):
    with open(path, 'w') as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False) + '\n')

def fetch(out):
    out.mkdir(parents=True, exist_ok=True)
    specifications = [
        ('sharegpt', 'anon8231489123/ShareGPT_Vicuna_unfiltered', 'ShareGPT_V3_unfiltered_cleaned_split.json'),
        ('wikitext', 'Salesforce/wikitext', None),
    ]
    sources = {}
    for name, repo, filename in specifications:
        record = out / (name + '.source.json')
        if record.exists():
            entry = json.loads(record.read_text())
            if Path(entry['file']).exists() and sha256(entry['file']) == entry['sha256']:
                sources[name] = entry; continue
        url = 'https://huggingface.co/api/datasets/' + repo
        with urllib.request.urlopen(url, timeout=60) as r: meta = json.load(r)
        revision = meta['sha']
        if filename is None:
            matches = [s['rfilename'] for s in meta['siblings'] if s['rfilename'].startswith('wikitext-2-raw-v1/test-') and s['rfilename'].endswith('.parquet')]
            if len(matches) != 1: raise RuntimeError(f'Unexpected test shard list: {matches}')
            filename = matches[0]
        url = f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{filename}'
        target = out / Path(filename).name
        temporary = target.with_suffix(target.suffix + '.partial')
        print('Downloading', url, flush=True)
        with urllib.request.urlopen(url, timeout=120) as r, temporary.open('wb') as f:
            while block := r.read(4 << 20): f.write(block)
        temporary.replace(target)
        sources[name] = {'repo': repo, 'revision': revision, 'url': url, 'file': str(target.resolve()), 'sha256': sha256(target)}
        json_write(record, sources[name])
    json_write(out / 'sources.json', sources)

class Tokenizer:
    def __init__(self, binary, model, log):
        self.log = open(log, 'w')
        self.proc = subprocess.Popen([str(binary), '--mode', 'tokenize', '--model', str(model)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True)
    def __call__(self, text):
        self.proc.stdin.write(json.dumps({'text': text}) + '\n'); self.proc.stdin.flush()
        while True:
            line = self.proc.stdout.readline()
            if not line: raise RuntimeError('Tokenizer exited; see log')
            if line.startswith('{'):
                j = json.loads(line); return j['tokens'], j['bos']
    def close(self):
        self.proc.stdin.close(); rc = self.proc.wait(); self.log.close()
        if rc: raise RuntimeError(f'Tokenizer failed: {rc}')

def prepare(data, out, model, local_sharegpt=None):
    import pyarrow.parquet as pq
    out.mkdir(parents=True, exist_ok=True)
    sources = json.loads((data / 'sources.json').read_text())
    sharegpt = Path(local_sharegpt or sources['sharegpt']['file'])
    source = sources['sharegpt'] if not local_sharegpt else {'file': str(sharegpt.resolve()), 'sha256': sha256(sharegpt), 'origin': 'user-supplied'}
    rows = json.loads(sharegpt.read_text())
    candidates = []
    for index, row in enumerate(rows):
        first = next((c for c in row.get('conversations', []) if c.get('from') in ('human','user')), None)
        if first and first.get('value'): candidates.append((str(row.get('id', index)), first['value']))
    # Deduplicate identities before partitioning; IDs in split snapshots may repeat.
    unique = {}; [unique.setdefault(r[0], r) for r in candidates]
    candidates = list(unique.values()); random.Random(SEED).shuffle(candidates)
    tokenizer = Tokenizer(ROOT / 'build-release/bin/llama-powerinfer-compare', model, out / 'tokenize.log')
    cache = {}; selected = []; used = set()
    try:
        for scenario, size in [('L',2048),('S',128)]:
            group = []
            for ident, text in candidates:
                if ident in used: continue
                if ident not in cache: cache[ident] = tokenizer(text)
                tokens, bos = cache[ident]
                if len(tokens) < size-1: continue
                # Continuations are only for teacher-forced correctness diagnostics.
                continuation = tokens[size-1:size+15]
                if len(continuation)<16: continuation=(tokens[:16])
                item = {'id':f'{scenario}-{len(group):02d}', 'source_id':ident, 'scenario':scenario,
                    'split':'calibration' if len(group)<2 else 'test', 'text':text,
                    'tokens':[bos]+tokens[:size-1], 'continuation':continuation,
                    'tokens_sha256':hashlib.sha256(json.dumps([bos]+tokens[:size-1],separators=(',',':')).encode()).hexdigest()}
                group.append(item); used.add(ident)
                if len(group)==12: break
            if len(group)!=12: raise RuntimeError(f'Not enough {scenario} candidates')
            selected.extend(group)
        quality_text = '\n\n'.join(pq.read_table(sources['wikitext']['file'])['text'].to_pylist())
        tokens,bos=tokenizer(quality_text)
    finally: tokenizer.close()
    jsonl_write(out/'manifest.jsonl',selected)
    calibration=[r for r in selected if r['split']=='calibration']
    jsonl_write(out/'calibration.jsonl',calibration)
    for round_id in range(3):
        rng=random.Random(SEED+round_id)
        groups=[[r for r in selected if r['split']=='test' and r['scenario']==s] for s in ['S','L']]
        for g in groups: rng.shuffle(g)
        requests=[r for pair in zip(*groups) for r in pair]
        # Warmups cover both shapes; all four calibration requests are excluded from statistics.
        jsonl_write(out/f'round-{round_id}.jsonl',calibration+requests)
    blocks=random.Random(SEED).sample(range(len(tokens)//1023),32)
    quality=[{'id':f'wiki-{i:04d}','block':i,'tokens':[bos]+tokens[i*1023:(i+1)*1023], 'split':'test','scenario':'quality'} for i in blocks]
    jsonl_write(out/'quality.jsonl',quality)
    json_write(out/'data_manifest.json',{'seed':SEED,'sharegpt':source,'wikitext':sources['wikitext'],
        'quality_blocks':blocks,'scored_tokens':32*1023,'tokenizer_model':str(model),
        'files':{p.name:sha256(p) for p in out.glob('*.jsonl')},'original_paper_snapshot_verified':False})
    print('Prepared 4 calibration + 20 formal prompts and 32 quality windows',flush=True)

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['fetch','prepare'])
    p.add_argument('--data',type=Path,default=ROOT/'logs/powerinfer-comparison/datasets')
    p.add_argument('--out',type=Path,default=ROOT/'logs/powerinfer-comparison/inputs')
    p.add_argument('--model',type=Path,default=MODEL)
    p.add_argument('--sharegpt')
    a=p.parse_args()
    if a.action=='fetch': fetch(a.data)
    else: prepare(a.data,a.out,a.model,a.sharegpt)
