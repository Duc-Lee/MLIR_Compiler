#!/usr/bin/env bash
set -eo pipefail

if [[ -f /etc/os-release ]]; then
  . /etc/os-release
  echo "Host OS: ${PRETTY_NAME:-Linux}"
fi

echo "Toolchain verification:"
REQUIRED_TOOLS=(cmake gcc g++ clang-18 mlir-opt-18 FileCheck-18 python3)
MISSING_TOOLS=()

for tool in "${REQUIRED_TOOLS[@]}"; do
  if command -v "${tool}" >/dev/null 2>&1; then
    printf "  %-16s : found (%s)\n" "${tool}" "$(command -v "${tool}")"
  else
    printf "  %-16s : missing\n" "${tool}"
    MISSING_TOOLS+=("${tool}")
  fi
done

if [[ ${#MISSING_TOOLS[@]} -gt 0 ]]; then
  echo ""
  echo "To install missing system packages:"
  echo "  sudo apt update"
  echo "  sudo apt install -y llvm-18-dev libmlir-18-dev mlir-18-tools clang-18 build-essential cmake ninja-build"
fi

echo ""
echo "Python environment:"
if python3 -c "import torch, numpy" >/dev/null 2>&1; then
  echo "  PyTorch $(python3 -c 'import torch; print(torch.__version__)') & NumPy $(python3 -c 'import numpy; print(numpy.__version__)') found"
else
  echo "  missing torch or numpy"
  echo "  install via: pip install torch numpy"
fi

if python3 -c "import torch_mlir" >/dev/null 2>&1; then
  echo "  torch-mlir $(python3 -c 'import torch_mlir; print(getattr(torch_mlir, \"__version__\", \"found\"))')"
else
  echo "  torch-mlir : not installed (optional)"
  echo "  install via: pip install --pre torch-mlir -f https://github.com/llvm/torch-mlir-release/releases/expanded_assets/dev-wheels"
fi
