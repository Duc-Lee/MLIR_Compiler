#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * MLIR standard 4D strided memref descriptor.
 * Layout matches MLIR's default memref<NxCxHxWxf32> ABI.
 */
typedef struct {
  float *allocated;
  float *aligned;
  int64_t offset;
  int64_t sizes[4];
  int64_t strides[4];
} StridedMemRefType4D;

/**
 * MLIR standard 1D strided memref descriptor.
 * Layout matches MLIR's default memref<Nxf32> ABI.
 */
typedef struct {
  float *allocated;
  float *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
} StridedMemRefType1D;

/**
 * Allocates a 64-byte aligned 4D memref suitable for AVX2/AVX-512 SIMD operations.
 */
StridedMemRefType4D vision_alloc_memref_4d(int64_t n, int64_t c, int64_t h, int64_t w);

/**
 * Frees memory backing a 4D memref descriptor.
 */
void vision_free_memref_4d(StridedMemRefType4D *memref);

/**
 * Allocates a 64-byte aligned 1D memref.
 */
StridedMemRefType1D vision_alloc_memref_1d(int64_t n);

/**
 * Frees memory backing a 1D memref descriptor.
 */
void vision_free_memref_1d(StridedMemRefType1D *memref);

/**
 * Initializes a float buffer with uniform random values in [-1.0, 1.0].
 */
void vision_init_tensor_random(float *data, size_t count);

/**
 * Fills a float buffer with a constant scalar value.
 */
void vision_init_tensor_constant(float *data, size_t count, float val);

/**
 * High-resolution wall-clock timer (monotonic).
 */
void vision_timer_start(void);
double vision_timer_stop_ms(void);

/**
 * Evicts data from CPU caches (L1/L2/L3) to measure cold-cache latency.
 */
void vision_flush_cpu_cache(void);

/**
 * Verifies numerical tolerance between actual and expected buffers.
 * Returns true if max absolute error <= epsilon.
 */
bool vision_verify_results(const float *actual, const float *expected,
                           size_t count, float epsilon);

/**
 * Resolves a model or asset path across candidate working directories.
 * Checks: 1) rel_path, 2) ../rel_path, 3) ../../rel_path, 4) $MLIR_PROJECT_ROOT/rel_path.
 * Returns true if an accessible file was found.
 */
bool vision_resolve_path(const char *rel_path, char *out_buf, size_t buf_size);

#ifdef __cplusplus
}
#endif
