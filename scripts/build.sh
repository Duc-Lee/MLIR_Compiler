#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
BUILD_TYPE="Release"
CLEAN_BUILD=false
JOBS="$(nproc 2>/dev/null || echo 4)"

print_usage() {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  -c, --clean         Wipe build directory before configuring"
  echo "  -d, --debug         Build with Debug configuration (default: Release)"
  echo "  -j, --jobs <N>      Parallel compile jobs (default: ${JOBS})"
  echo "  -h, --help          Show this help message"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--clean)
      CLEAN_BUILD=true
      shift
      ;;
    -d|--debug)
      BUILD_TYPE="Debug"
      shift
      ;;
    -j|--jobs)
      JOBS="$2"
      shift 2
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      print_usage
      exit 1
      ;;
  esac
done

if [[ "${CLEAN_BUILD}" == true && -d "${BUILD_DIR}" ]]; then
  echo "Cleaning build directory: ${BUILD_DIR}"
  rm -rf "${BUILD_DIR}"
fi

mkdir -p "${BUILD_DIR}"

GENERATOR="Unix Makefiles"
if command -v ninja >/dev/null 2>&1; then
  GENERATOR="Ninja"
fi

echo "Configuring build (${BUILD_TYPE}, generator: ${GENERATOR})..."
cmake -B "${BUILD_DIR}" -S "${ROOT_DIR}" \
  -G "${GENERATOR}" \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

echo "Compiling with ${JOBS} jobs..."
cmake --build "${BUILD_DIR}" -- -j"${JOBS}"

echo "Build complete. Binaries available in: ${BUILD_DIR}/bin"
