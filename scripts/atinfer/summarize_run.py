#!/usr/bin/env python3
"""Summarize llama-bench JSON outputs from logs/run-2026-09-13 into a markdown table."""
import json, sys, glob, os

LOGDIR = "logs/run-2026-09-13"

rows = []
for path in sorted(glob.glob(os.path.join(LOGDIR, "*.json"))):
    name = os.path.basename(path).replace(".json", "")
    try:
        data = json.load(open(path))
    except Exception as e:
        print(f"skip {name}: {e}", file=sys.stderr)
        continue
    for t in data:
        # llama-bench json fields: n_gpu_layers, n_gen, n_prompt, avg_tok_per_s etc.
        rows.append({
            "file": name,
            "ngl": t.get("n_gpu_layers"),
            "pipo": t.get("pipo", t.get("enable_pipo")),
            "do": t.get("do", t.get("enable_decode_offload")),
            "pp": t.get("n_prompt"),
            "ngen": t.get("n_gen"),
            "tps": t.get("avg_tokens_per_second"),
        })

print("| run | ngl | pipo | -do | n_prompt | n_gen | tokens/s |")
print("|---|---|---|---|---|---|---|")
for r in rows:
    print(f"| {r['file']} | {r['ngl']} | {r['pipo']} | {r['do']} | {r['pp']} | {r['ngen']} | {r['tps']:.2f} |" if r['tps'] else
          f"| {r['file']} | {r['ngl']} | {r['pipo']} | {r['do']} | {r['pp']} | {r['ngen']} | n/a |")
