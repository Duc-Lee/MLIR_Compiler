#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace mlir::vision {

// cau hinh tiling cho cac lop tinh toan conv2d va gemm
struct TilingConfig {
  int64_t tileM = 32;
  int64_t tileN = 32;
  int64_t tileK = 64;
  bool enableVectorization = true;
};

// tap lenh muc tieu
enum class TargetISA {
  Generic,
  X86_AVX2,
  X86_AVX512,
  ARM_NEON
};

// thong so vector SIMD cho backend
struct TargetHardwareConfig {
  TargetISA isa = TargetISA::X86_AVX2;
  int64_t vectorWidthBits = 256;      // 256-bit cho AVX2 (8 float32)
  int64_t l1DataCacheBytes = 32768;   // 32KB L1 data cache pho bien
  int64_t l2CacheBytes = 262144;      // 256KB-512KB L2
};

} // namespace mlir::vision
