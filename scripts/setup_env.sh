#!/usr/bin/env bash
set -e

echo "============================================================"
echo "  Thiet Lap Moi Truong Cho MLIR Vision Compiler Pipeline"
echo "============================================================"

# kiem tra he dieu hanh
if [ -f /etc/os-release ]; then
  . /etc/os-release
  echo "[+] Phat hien OS: $PRETTY_NAME"
fi

echo ""
echo "[1] Cai dat cac goi LLVM/MLIR 18 tren Ubuntu/Debian:"
echo "    sudo apt update"
echo "    sudo apt install -y llvm-18-dev libmlir-18-dev mlir-18-tools clang-18 ninja-build build-essential cmake"
echo ""

# kiem tra cac cong cu da co
echo "[2] Kiem tra cac cong cu hien tai tren may:"
for tool in cmake gcc clang-18 mlir-opt-18 python3; do
  if command -v "$tool" >/dev/null 2>&1; then
    echo "    - $tool: DA CO ($(command -v "$tool"))"
  else
    echo "    - $tool: CHUA CO"
  fi
done

echo ""
echo "[3] Cai dat thu vien Python cho frontend (PyTorch & torch-mlir):"
echo "    python3 -m venv .venv"
echo "    source .venv/bin/activate"
echo "    pip install --upgrade pip"
echo "    pip install torch torchvision"
echo "    pip install --pre torch-mlir -f https://llvm.github.io/torch-mlir/package-index/"
echo ""
echo "[+] Hoan tat huong dan thiet lap moi truong."
