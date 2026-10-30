#define _DEFAULT_SOURCE
#define _POSIX_C_SOURCE 200809L
#include "vision_runtime.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static struct timespec g_start_time;

StridedMemRefType4D vision_alloc_memref_4d(int64_t n, int64_t c, int64_t h, int64_t w) {
  StridedMemRefType4D ref;
  size_t total_elements = (size_t)(n * c * h * w);
  size_t bytes = total_elements * sizeof(float);

  void *ptr = NULL;
  if (posix_memalign(&ptr, 64, bytes) != 0) {
    fprintf(stderr, "error: failed to allocate 64-byte aligned memory (%zu bytes)\n", bytes);
    exit(EXIT_FAILURE);
  }

  ref.allocated = (float *)ptr;
  ref.aligned = (float *)ptr;
  ref.offset = 0;

  ref.sizes[0] = n;
  ref.sizes[1] = c;
  ref.sizes[2] = h;
  ref.sizes[3] = w;

  // Row-major strides (NCHW)
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

StridedMemRefType1D vision_alloc_memref_1d(int64_t n) {
  StridedMemRefType1D ref;
  size_t bytes = (size_t)n * sizeof(float);

  void *ptr = NULL;
  if (posix_memalign(&ptr, 64, bytes) != 0) {
    fprintf(stderr, "error: failed to allocate 64-byte aligned memory (%zu bytes)\n", bytes);
    exit(EXIT_FAILURE);
  }

  ref.allocated = (float *)ptr;
  ref.aligned = (float *)ptr;
  ref.offset = 0;
  ref.sizes[0] = n;
  ref.strides[0] = 1;

  return ref;
}

void vision_free_memref_1d(StridedMemRefType1D *memref) {
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
  return (double)(end_time.tv_sec - g_start_time.tv_sec) * 1000.0 +
         (double)(end_time.tv_nsec - g_start_time.tv_nsec) / 1000000.0;
}

void vision_flush_cpu_cache(void) {
  const size_t flush_size = 64 * 1024 * 1024; // 64 MB buffer to clear L3
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
        fprintf(stderr, "verify: mismatch at idx %zu: actual=%.5f, expected=%.5f, diff=%.5f\n",
                i, actual[i], expected[i], diff);
      }
      error_count++;
    }
  }

  if (error_count > 0) {
    fprintf(stderr, "verify: %zu/%zu elements differ (max diff=%.6f, tol=%.6f)\n",
            error_count, count, max_diff, epsilon);
    return false;
  }

  return true;
}

bool vision_resolve_path(const char *rel_path, char *out_buf, size_t buf_size) {
  if (!rel_path || !out_buf || buf_size == 0)
    return false;

  const char *candidates[4];
  char env_candidate[512] = {0};

  candidates[0] = rel_path;

  char candidate_up1[512];
  snprintf(candidate_up1, sizeof(candidate_up1), "../%s", rel_path);
  candidates[1] = candidate_up1;

  char candidate_up2[512];
  snprintf(candidate_up2, sizeof(candidate_up2), "../../%s", rel_path);
  candidates[2] = candidate_up2;

  const char *project_root = getenv("MLIR_PROJECT_ROOT");
  if (project_root) {
    snprintf(env_candidate, sizeof(env_candidate), "%s/%s", project_root, rel_path);
    candidates[3] = env_candidate;
  } else {
    candidates[3] = NULL;
  }

  for (int i = 0; i < 4; ++i) {
    if (candidates[i] && access(candidates[i], R_OK) == 0) {
      strncpy(out_buf, candidates[i], buf_size - 1);
      out_buf[buf_size - 1] = '\0';
      return true;
    }
  }

  return false;
}
