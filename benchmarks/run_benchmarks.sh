#!/usr/bin/env bash
set -e

# thu muc goc cua du an
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_DIR="${ROOT_DIR}/build/bin"
mkdir -p "${BIN_DIR}"

echo ">>> [1/2] Bien dich & Chay Benchmark Neural Network MLP (784 -> 128 -> 64 -> 10)..."
gcc -O3 -mavx2 -mfma -Wall -Wextra \
    -I"${ROOT_DIR}/runtime/include" \
    "${ROOT_DIR}/runtime/src/vision_runtime.c" \
    "${ROOT_DIR}/benchmarks/bench_mlp.c" \
    -o "${BIN_DIR}/bench_mlp" -lm
"${BIN_DIR}/bench_mlp"

echo ">>> [2/2] Bien dich & Chay Benchmark Conv2D..."
gcc -O3 -mavx2 -mfma -Wall -Wextra \
    -I"${ROOT_DIR}/runtime/include" \
    "${ROOT_DIR}/runtime/src/vision_runtime.c" \
    "${ROOT_DIR}/benchmarks/bench_conv.c" \
    -o "${BIN_DIR}/bench_conv" -lm
"${BIN_DIR}/bench_conv"
