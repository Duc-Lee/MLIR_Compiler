#define _DEFAULT_SOURCE
#define _POSIX_C_SOURCE 200809L
#include "vision_runtime.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static struct timespec g_start_time;

StridedMemRefType4D vision_alloc_memref_4d(int64_t n, int64_t c, int64_t h, int64_t w) {
  StridedMemRefType4D ref;
  size_t total_elements = (size_t)(n * c * h * w);
  size_t bytes = total_elements * sizeof(float);

  // can le 64 bytes cho AVX2 / AVX-512
  void *ptr = NULL;
  if (posix_memalign(&ptr, 64, bytes) != 0) {
    fprintf(stderr, "Loi cap phat bo nho aligned 64 bytes\n");
    exit(1);
  }

  ref.allocated = (float *)ptr;
  ref.aligned = (float *)ptr;
  ref.offset = 0;

  ref.sizes[0] = n;
  ref.sizes[1] = c;
  ref.sizes[2] = h;
  ref.sizes[3] = w;

  // row-major strides (NCHW)
  ref.strides[3] = 1;
  ref.strides[2] = w;
  ref.strides[1] = h * w;
  ref.strides[0] = c * h * w;

  return ref;
}

void vision_free_memref_4d(StridedMemRefType4D *memref) {
  if (memref && memref->allocated) {
    free(memref->allocated);
    memref->allocated = NULL;
    memref->aligned = NULL;
  }
}

void vision_init_tensor_random(float *data, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    data[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
  }
}

void vision_init_tensor_constant(float *data, size_t count, float val) {
  for (size_t i = 0; i < count; ++i) {
    data[i] = val;
  }
}

void vision_timer_start(void) {
  clock_gettime(CLOCK_MONOTONIC, &g_start_time);
}

double vision_timer_stop_ms(void) {
  struct timespec end_time;
  clock_gettime(CLOCK_MONOTONIC, &end_time);
  double elapsed = (double)(end_time.tv_sec - g_start_time.tv_sec) * 1000.0 +
                   (double)(end_time.tv_nsec - g_start_time.tv_nsec) / 1000000.0;
  return elapsed;
}

void vision_flush_cpu_cache(void) {
  // tao vung nho 64MB lon hon L3 cache de day toan bo du lieu ra khoi cache
  const size_t flush_size = 64 * 1024 * 1024;
  volatile char *buf = (volatile char *)malloc(flush_size);
  if (!buf)
    return;
  for (size_t i = 0; i < flush_size; i += 64) {
    buf[i] = (char)i;
  }
  free((void *)buf);
}

bool vision_verify_results(const float *actual, const float *expected,
                           size_t count, float epsilon) {
  size_t error_count = 0;
  float max_diff = 0.0f;

  for (size_t i = 0; i < count; ++i) {
    float diff = fabsf(actual[i] - expected[i]);
    if (diff > max_diff)
      max_diff = diff;
    if (diff > epsilon) {
      if (error_count < 5) {
        printf("Loi tai vi tri [%zu]: actual=%.5f, expected=%.5f, diff=%.5f\n",
               i, actual[i], expected[i], diff);
      }
      error_count++;
    }
  }

  if (error_count > 0) {
    printf("Xac minh that bai: %zu/%zu phan tu sai lech (max diff = %.6f)\n",
           error_count, count, max_diff);
    return false;
  }

  printf("Xac minh thanh cong: %zu phan tu khop (max diff = %.6f <= epsilon %.6f)\n",
         count, max_diff, epsilon);
  return true;
}
