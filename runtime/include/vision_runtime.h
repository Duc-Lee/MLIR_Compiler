#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  float *allocated;
  float *aligned;
  int64_t offset;
  int64_t sizes[4];
  int64_t strides[4];
} StridedMemRefType4D;

// cau truc memref 1D (bias / 1D tensor)
typedef struct {
  float *allocated;
  float *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
} StridedMemRefType1D;

// cap phat memref 4D duoc can le 64 bytes (cho AVX2/AVX512)
StridedMemRefType4D vision_alloc_memref_4d(int64_t n, int64_t c, int64_t h, int64_t w);

// giai phong memref 4D
void vision_free_memref_4d(StridedMemRefType4D *memref);

// khoi tao tensor voi gia tri ngau nhien hoac hang so
void vision_init_tensor_random(float *data, size_t count);
void vision_init_tensor_constant(float *data, size_t count, float val);

// do thoi gian thuc thi tinh bang mili giay
void vision_timer_start(void);
double vision_timer_stop_ms(void);

// xoa sach L1/L2/L3 cache de do chinh xac cold cache latency
void vision_flush_cpu_cache(void);

// kiem tra do lech giua ket qua compiler va ground truth
bool vision_verify_results(const float *actual, const float *expected,
                           size_t count, float epsilon);

#ifdef __cplusplus
}
#endif
