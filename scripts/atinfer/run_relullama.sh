#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL="${RELULLAMA_MODEL:-/home/lush/Models/ReluLLaMA-13B-PowerInfer-GGUF/llama-13b-relu.powerinfer.gguf}"
CONFIG="${RELULLAMA_ATSINFER_CONFIG:-$ROOT_DIR/examples/pipo-alg/alg_cfg/relullama-13b-4090.json}"
MODE="${1:-smoke}"
THREADS="${RELULLAMA_THREADS:-6}"

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "Missing file: $1" >&2
        exit 1
    fi
}

require_executable() {
    if [[ ! -x "$1" ]]; then
        echo "Missing executable: $1" >&2
        echo "Build the required targets first; see RELULLAMA_ATSINFER.md." >&2
        exit 1
    fi
}

check_cuda() {
    if ! nvidia-smi >/dev/null 2>&1; then
        echo "CUDA is not available: nvidia-smi cannot communicate with the driver." >&2
        echo "Restore the NVIDIA driver, then rerun this command." >&2
        exit 2
    fi
}

require_file "$MODEL"
require_file "$CONFIG"
check_cuda

case "$MODE" in
    smoke)
        BIN="$ROOT_DIR/build-release/bin/llama-simple"
        require_executable "$BIN"
        exec "$BIN" -m "$MODEL" -pipo -config "$CONFIG" -do 0 \
            -t "$THREADS" -ubatch 512 -n 16 \
            "Explain why the sky is blue in one sentence."
        ;;
    decode)
        BIN="$ROOT_DIR/build-release/bin/llama-bench"
        require_executable "$BIN"
        exec "$BIN" -m "$MODEL" -o md -r 3 -t "$THREADS" \
            -b 512 -ub 512 -p 0 -n 128,512 -d 2048 \
            --pipo 1 -config "$CONFIG" -do 0
        ;;
    prefill)
        BIN="$ROOT_DIR/build-release/bin/llama-bench"
        require_executable "$BIN"
        exec "$BIN" -m "$MODEL" -o md -r 3 -t "$THREADS" \
            -b 512 -ub 512 -p 512,1024,2048 -n 0 -d 2048 \
            --pipo 1 -config "$CONFIG" -do 0
        ;;
    profile)
        BIN="$ROOT_DIR/build-release/bin/llama-pipo-perf"
        require_executable "$BIN"
        OUT="${RELULLAMA_PROFILE:-$ROOT_DIR/logs/relullama-atsinfer/pipo_profile.json}"
        exec "$BIN" -m "$MODEL" -c "$OUT" -t "$THREADS"
        ;;
    *)
        echo "Usage: $0 {smoke|decode|prefill|profile}" >&2
        exit 1
        ;;
esac
