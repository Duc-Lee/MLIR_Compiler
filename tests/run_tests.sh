#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPT_BIN="${ROOT_DIR}/build/bin/vision-opt"
FILECHECK_BIN="/home/duc-le/.local/llvm-18/usr/lib/llvm-18/bin/FileCheck"

if [ ! -f "${OPT_BIN}" ]; then
  echo "[!] vision-opt chua duoc build tai ${OPT_BIN}"
  echo "    Vui long chay: make vision-opt"
  exit 1
fi

if [ ! -f "${FILECHECK_BIN}" ]; then
  FILECHECK_BIN=$(command -v FileCheck || true)
fi

echo "=== Chay Test Suite Cho Vision MLIR Passes ==="
for test_file in "${ROOT_DIR}"/tests/*.mlir; do
  test_name=$(basename "${test_file}")
  echo -n "[TEST] ${test_name} ... "
  
  if grep -q "RUN:" "${test_file}"; then
    run_cmd=$(grep "RUN:" "${test_file}" | sed "s|// RUN: ||; s|vision-opt|${OPT_BIN}|; s|FileCheck|${FILECHECK_BIN}|; s|%s|${test_file}|g")
    if eval "${run_cmd}" > /dev/null 2>&1; then
      echo "PASSED"
    else
      echo "FAILED (command: ${run_cmd})"
    fi
  else
    echo "SKIPPED (no RUN line)"
  fi
done
