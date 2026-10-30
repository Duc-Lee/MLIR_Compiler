#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace mlir::vision {

/// Tiling dimensions for 2D convolution and GEMM operators.
struct TilingConfig {
  int64_t tileM = 32;
  int64_t tileN = 32;
  int64_t tileK = 64;
  bool enableVectorization = true;
};

/// Target processor architecture / Instruction Set Architecture.
enum class TargetISA {
  Generic,
  X86_AVX2,
  X86_AVX512,
  ARM_NEON
};

/// Hardware parameters for backend code generation and vectorization.
struct TargetHardwareConfig {
  TargetISA isa = TargetISA::X86_AVX2;
  int64_t vectorWidthBits = 256;      // 256-bit registers for AVX2 (8 float32 elements)
  int64_t l1DataCacheBytes = 32768;   // Typical 32 KB L1 data cache
  int64_t l2CacheBytes = 262144;      // 256 KB L2 cache
};

} // namespace mlir::vision
