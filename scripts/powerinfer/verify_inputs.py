#!/usr/bin/env python3
"""Verify the canonical token manifest and diagnose legacy tokenizer differences."""
import json
from pathlib import Path
from prepare_comparison import ROOT, MODEL, Tokenizer, jsonl_write, json_write

base=ROOT/'logs/powerinfer-comparison'
old=json.loads((base/'powerinfer-vocab.json').read_text())
new=json.loads((base/'dense-vocab.json').read_text())
if old!=new: raise ValueError('Vocabularies differ')
rows=[json.loads(s) for s in (base/'inputs/manifest.jsonl').open()]
tokenizer=Tokenizer(ROOT/'build-release/comparison/powerinfer-compare',MODEL,base/'legacy-tokenize.log')
differences=[]
try:
    for r in rows:
        tokens,bos=tokenizer(r['text'])
        if bos!=r['tokens'][0] or any(t<0 or t>=old['n_vocab'] for t in r['tokens']): raise ValueError('Invalid canonical token IDs')
        if [bos]+tokens[:len(r['tokens'])-1]!=r['tokens']:
            padded,_=tokenizer(' '+r['text'])
            differences.append({'id':r['id'],'legacy_ids_prefix':tokens[:16],
                'canonical_ids_prefix':r['tokens'][1:17],
                'matches_with_explicit_leading_space':[bos]+padded[:len(r['tokens'])-1]==r['tokens']})
finally: tokenizer.close()
if len({r['source_id'] for r in rows})!=24: raise ValueError('Duplicate conversations')
if len({r['id'] for r in rows})!=24: raise ValueError('Duplicate request IDs')
quality=[json.loads(s) for s in (base/'inputs/quality.jsonl').open()]
if len({r['block'] for r in quality})!=32 or any(len(r['tokens'])!=1024 for r in quality): raise ValueError('Bad quality windows')
diagnostics=[]
for r in rows:
    if r['scenario']=='S' and r['split']=='calibration':
        diagnostics.append({**r,'tokens':r['tokens'][:32],'id':r['id']+'-quality-check'})
jsonl_write(base/'inputs/quality-check.jsonl',diagnostics)
json_write(base/'inputs/verification.json',{'passed':True,'vocabulary_size':old['n_vocab'],
    'canonical_prompts':len(rows),'legacy_tokenizer_differences':differences,'quality_windows':len(quality),
    'input_policy':'Tokenize once with current llama tokenizer (SentencePiece leading-space normalization), add BOS once; feed identical canonical IDs to every runner.',
    'quality_check_scored_tokens':62,'batch_path_mean_nll_difference_diagnostic_limit':0.01})
print(f'PASS: identical vocabulary, 24 canonical prompts, 32 quality windows; {len(differences)} legacy normalization differences recorded')
