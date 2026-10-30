#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
BIN_DIR="${BUILD_DIR}/bin"

if [[ ! -x "${BIN_DIR}/bench_mlp" || ! -x "${BIN_DIR}/bench_conv" ]]; then
  echo "Building benchmark binaries..."
  cmake -B "${BUILD_DIR}" -S "${ROOT_DIR}" -DCMAKE_BUILD_TYPE=Release
  cmake --build "${BUILD_DIR}" --target bench_mlp bench_conv -j"$(nproc)"
fi

"${BIN_DIR}/bench_mlp" "$@"
echo ""
"${BIN_DIR}/bench_conv" "$@"
