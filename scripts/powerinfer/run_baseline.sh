#!/usr/bin/env bash

set -euo pipefail

MODEL=/home/lush/Models/ReluLLaMA-13B-PowerInfer-GGUF/llama-13b-relu.powerinfer.gguf
POWERINFER_BIN=/home/lush/PowerInfer/build-sm89/bin
OUT=/home/lush/llama.cpp/logs/powerinfer
VRAM_GB=18
THREADS=6

test -f "$MODEL"
test -x "$POWERINFER_BIN/llama-bench"
mkdir -p "$OUT"

export PYTHONPATH=/home/lush/PowerInfer/gguf-py

"$POWERINFER_BIN/llama-bench" \
  -m "$MODEL" --vram-budget "$VRAM_GB" -ngl 0 \
  -o md -r 3 -t "$THREADS" -b 512 -p 512,2048 -n 0 \
  > "$OUT/powerinfer_vram18_prefill.md" \
  2> "$OUT/powerinfer_vram18_prefill.err"

"$POWERINFER_BIN/llama-bench" \
  -m "$MODEL" --vram-budget "$VRAM_GB" -ngl 0 \
  -o md -r 3 -t "$THREADS" -b 512 -p 0 -n 128,512 \
  > "$OUT/powerinfer_vram18_decode.md" \
  2> "$OUT/powerinfer_vram18_decode.err"

cat "$OUT/powerinfer_vram18_prefill.md"
cat "$OUT/powerinfer_vram18_decode.md"
