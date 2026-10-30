#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"

echo "============================================================"
echo "  Bien Dich Du An MLIR_Compiler"
echo "============================================================"

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# tim kiem thu muc cmake cua mlir neu co trong he thong
MLIR_CMAKE_PATH=""
for p in /usr/lib/llvm-18/lib/cmake/mlir /usr/lib/llvm-19/lib/cmake/mlir /usr/local/lib/cmake/mlir; do
  if [ -d "$p" ]; then
    MLIR_CMAKE_PATH="-DMLIR_DIR=$p"
    break
  fi
done

echo "[*] Chay CMake..."
cmake .. ${MLIR_CMAKE_PATH} -DCMAKE_BUILD_TYPE=Release

echo "[*] Tien hanh bien dich..."
make -j"$(nproc)"

echo "[+] Hoan tat! Cac file binary nam tai: ${BUILD_DIR}/bin"
