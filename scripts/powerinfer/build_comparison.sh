#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PI=${POWERINFER_ROOT:-/home/lush/PowerInfer}
CUDA=${CUDA_ROOT:-/usr/local/cuda}
cmake -S "$ROOT" -B "$ROOT/build-release" -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build "$ROOT/build-release" -j8 --target llama-powerinfer-compare llama-pipo-perf llama-pipo-override test-relullama-outprod
mkdir -p "$ROOT/build-release/comparison"
c++ -O3 -DNDEBUG -std=c++17 -DPOWERINFER_LEGACY \
    -I"$PI" -I"$ROOT/vendor" "$ROOT/tools/powerinfer-compare/compare.cpp" \
    "$PI/build-sm89/libllama.a" \
    -L"$CUDA/targets/x86_64-linux/lib" -Wl,-rpath,"$CUDA/targets/x86_64-linux/lib" \
    -lcublas -lcublasLt -lcudart -lcudadevrt -lcudart_static -lrt -lpthread -ldl \
    -o "$ROOT/build-release/comparison/powerinfer-compare"
