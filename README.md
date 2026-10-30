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

* Linux (Ubuntu 20.04/22.04 or WSL2)
* CMake >= 3.20
* GCC or Clang with C++17 support
* LLVM / MLIR 18
* Python 3 with PyTorch (for model export and reference verification)

---

## Build Instructions

```bash
# Clone the repository
git clone <repo-url>
cd MLIR_Compiler

# Build with CMake
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

Alternatively, use the build script:
```bash
./scripts/build.sh
```

Binaries will be available in `build/bin/`:
* `vision-opt`: Compiler optimizer driver.
* `vision-translate`: Translation tool to LLVM IR / C.
* `bench_mlp`: Neural Network benchmark executable.
* `bench_conv`: Conv2D benchmark executable.

---

## Usage

### 1. Run Compiler Passes

Run individual passes or the complete lowering pipeline using `vision-opt`:

```bash
# Operator Fusion
./build/bin/vision-opt --vision-fusion models/mlp.mlir

# Loop Tiling (L1 Cache)
./build/bin/vision-opt --vision-tiling models/mlp.mlir

# Complete lowering pipeline to LLVM Dialect
./build/bin/vision-opt --vision-pipeline models/mlp.mlir -o build/mlp_lowered.mlir
```

### 2. Run Test Suite

Run all MLIR regression tests verified by LLVM `FileCheck`:

```bash
bash tests/run_tests.sh
```

### 3. End-to-End Compilation & PyTorch Verification

Run the end-to-end driver to export weights, compile MLIR, and verify numerical precision against PyTorch:

```bash
python3 models/compile_and_run_mlp.py
```

### 4. Run Benchmarks

Measure latency across different optimization stages:

```bash
bash benchmarks/run_benchmarks.sh
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
