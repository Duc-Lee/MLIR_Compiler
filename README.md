# MLIR Deep Learning Compiler Pipeline

An end-to-end MLIR-based optimizing compiler pipeline for deep learning inference on x86-64 CPUs. The compiler transforms high-level neural network graphs (`Linalg` / `TOSA`) through custom middle-end optimization passes down to vectorized LLVM IR.

**Target Workload**: 3-Layer Neural Network (MNIST Multi-Layer Perceptron: `784 -> 128 -> 64 -> 10`) and Conv2D kernels.

---

## Features

* **End-to-End Lowering**: Directly lowers PyTorch models and MLIR graphs to executable LLVM IR without intermediate ONNX dependencies.
* **GEMM Epilogue Fusion**: Fuses matrix multiplication with bias addition and activation functions (`MatMul + Bias + ReLU`) into a single pass, eliminating intermediate buffer allocations.
* **Cache-Aware Loop Tiling**: Decomposes matrix iteration spaces into tiles fitting CPU L1/L2 data cache capacity using MLIR's `TilingInterface` and `SCF`.
* **AVX2 SIMD Vectorization**: Lowers Linalg operations to the Vector dialect to emit 256-bit FMA vector instructions (`vfmadd213ps`).

---

## Requirements

* Linux (Ubuntu 20.04 / 22.04 / 24.04 or WSL2)
* CMake >= 3.20
* GCC or Clang with C++17 support
* LLVM / MLIR 18 (`llvm-18-dev`, `libmlir-18-dev`, `mlir-18-tools`)
* Python >= 3.10 with PyTorch and NumPy

### Python Setup & `torch-mlir` Installation

The compiler uses PyTorch for model definition, graph tracing via `torch.fx`, and numeric verification. If you also need `torch-mlir` for dialect compilation and frontend lowering:

```bash
# 1. Create and activate a Python virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# 2. Install PyTorch and NumPy
pip install --upgrade pip
pip install torch numpy

# 3. Install torch-mlir (official LLVM release wheels, supports Python 3.10 - 3.12)
pip install --pre torch-mlir \
  --extra-index-url https://download.pytorch.org/whl/nightly/cpu \
  -f https://github.com/llvm/torch-mlir-release/releases/expanded_assets/dev-wheels
```

> **Note**: 
> - If you run Ubuntu 24.04 (PEP 668), use the virtual environment above or specify `--break-system-packages` if installing globally.
> - The old index (`llvm.github.io/torch-mlir/package-index`) is deprecated and only supports up to Python 3.11. Modern builds (including Python 3.12) are hosted directly at [llvm/torch-mlir-release](https://github.com/llvm/torch-mlir-release).

---

## Build

### Automated Build

```bash
./scripts/build.sh
```

Flags supported:
- `--clean`: Wipe build cache before compiling.
- `--debug`: Compile with `-DCMAKE_BUILD_TYPE=Debug`.
- `-j <N>`: Set build parallelism.

### Manual CMake Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

Outputs generated in `build/bin/`:
| Target | Description |
| :--- | :--- |
| `vision-opt` | Custom optimizer driver (`--vision-fusion`, `--vision-tiling`, `--vision-vectorize`, `--vision-pipeline`) |
| `vision-translate` | Translation utility emitting LLVM IR from MLIR modules |
| `bench_mlp` | C benchmark harness evaluating Naive, Fused, Tiled, and AVX2 GEMM kernels |
| `bench_conv` | C benchmark harness evaluating Conv2D AVX2 SIMD acceleration |

---

## Workflow & Verification

### 1. Regression Test Suite
Run the MLIR FileCheck regression suite:
```bash
./tests/run_tests.sh
```

### 2. End-to-End Pipeline & Numeric Verification
Export PyTorch weights, trace model to MLIR, lower through `vision-opt`, and verify output against PyTorch ground truth:
```bash
# Multi-Layer Perceptron (MNIST: 784 -> 128 -> 64 -> 10)
python3 models/compile_and_run_mlp.py

# Conv2D + BatchNorm + ReLU Kernel
python3 models/compile_and_run_model.py
```

### 3. Benchmarks & Profiling
Measure execution latency and speedup across optimization passes:
```bash
bash benchmarks/run_benchmarks.sh
```

### 4. Running Passes Standalone (`vision-opt`)
```bash
# Operator Fusion (GEMM + Bias + ReLU)
./build/bin/vision-opt --vision-fusion models/mlp.mlir

# Cache-aware Loop Tiling
./build/bin/vision-opt --vision-tiling models/mlp.mlir

# Complete lowering down to LLVM dialect
./build/bin/vision-opt --vision-pipeline models/mlp.mlir -o build/mlp_lowered.mlir
```

---

## Optimization Passes

1. **Operator Fusion (`--vision-fusion`)**:
   Applies rewrite patterns on high-level dialects to combine elementwise activations and bias additions directly with matrix operations. Avoids round-tripping intermediate tensors through memory.

2. **Loop Tiling (`--vision-tiling`)**:
   Applies loop blocking via `scf.for` over iteration spaces $(M, N, K)$. Configured to fit working sets within standard 32 KB L1 data caches to minimize cache misses.

3. **SIMD Vectorization (`--vision-vectorize`)**:
   Vectorizes iteration spaces down to the Vector dialect targeting 256-bit AVX2 vector registers (8 x float32), generating hardware-accelerated FMA instructions.

---

## Evaluation Results

Measured on x86-64 CPU (Batch Size = 64, 500 iterations):

| Optimization Stage | Latency | Speedup | Latency / Sample |
| :--- | :---: | :---: | :---: |
| 1. Unoptimized (Separate Ops + Dynamic Alloc) | 0.363 ms | 1.00x | 5.7 µs |
| 2. Pass 1: Operator Fusion | 0.297 ms | 1.22x | 4.6 µs |
| 3. Pass 2: Cache-Aware Loop Tiling | 0.403 ms | 0.90x | 6.3 µs |
| 4. Pass 3: AVX2 SIMD Vectorization | 0.298 ms | 1.22x | 4.7 µs |

* **Ground Truth Verification**: Verified against PyTorch Eager reference outputs (`Max Absolute Error = 4.76e-07`, Predicted Class = 7).
