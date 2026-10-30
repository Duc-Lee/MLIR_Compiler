#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPT_BIN="${ROOT_DIR}/build/bin/vision-opt"

if [[ ! -x "${OPT_BIN}" ]]; then
  echo "error: compiler binary not found at ${OPT_BIN}" >&2
  echo "hint: run ./scripts/build.sh first" >&2
  exit 1
fi

FILECHECK_BIN=""
CANDIDATE_PATHS=(
  "$(command -v FileCheck 2>/dev/null || true)"
  "$(command -v FileCheck-18 2>/dev/null || true)"
  "$(command -v FileCheck-19 2>/dev/null || true)"
  "${HOME}/.local/llvm-18/usr/lib/llvm-18/bin/FileCheck"
  "${HOME}/.local/llvm-19/usr/lib/llvm-19/bin/FileCheck"
  "/usr/lib/llvm-18/bin/FileCheck"
  "/usr/lib/llvm-19/bin/FileCheck"
)

for cand in "${CANDIDATE_PATHS[@]}"; do
  if [[ -n "${cand}" && -x "${cand}" ]]; then
    FILECHECK_BIN="${cand}"
    break
  fi
done

if [[ -z "${FILECHECK_BIN}" ]]; then
  echo "error: FileCheck utility not found" >&2
  exit 1
fi

echo "Testing MLIR compiler passes (${OPT_BIN##*/}):"

TOTAL=0
PASSED=0
FAILED=0

for test_file in "${ROOT_DIR}"/tests/*.mlir; do
  test_name=$(basename "${test_file}")
  TOTAL=$((TOTAL + 1))

  if grep -q "RUN:" "${test_file}"; then
    run_cmd=$(grep "RUN:" "${test_file}" | head -n 1 | sed \
      -e "s|// RUN: ||" \
      -e "s|vision-opt|${OPT_BIN}|g" \
      -e "s|FileCheck|${FILECHECK_BIN}|g" \
      -e "s|%s|${test_file}|g")

    if eval "${run_cmd}" > /dev/null 2>&1; then
      printf "  %-32s pass\n" "${test_name}"
      PASSED=$((PASSED + 1))
    else
      printf "  %-32s FAIL\n" "${test_name}"
      echo "    cmd: ${run_cmd}" >&2
      FAILED=$((FAILED + 1))
    fi
  else
    printf "  %-32s skip (no RUN line)\n" "${test_name}"
  fi
done

echo "Result: ${PASSED}/${TOTAL} passed, ${FAILED} failed"

if [[ ${FAILED} -gt 0 ]]; then
  exit 1
fi
