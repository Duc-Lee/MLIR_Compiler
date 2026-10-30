#define _POSIX_C_SOURCE 200809L
#include "vision_runtime.h"

#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static inline uint64_t get_time_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

// Architecture: 784 -> 128 (ReLU) -> 64 (ReLU) -> 10 (Logits)
#define IN_FEATURES  784
#define HIDDEN1      128
#define HIDDEN2      64
#define NUM_CLASSES  10
#define DEFAULT_BATCH_SIZE   64
#define DEFAULT_BENCH_ITERS  500

typedef struct {
  float w1[IN_FEATURES * HIDDEN1];
  float b1[HIDDEN1];
  float w2[HIDDEN1 * HIDDEN2];
  float b2[HIDDEN2];
  float w3[HIDDEN2 * NUM_CLASSES];
  float b3[NUM_CLASSES];
} MLPWeights;

// 1. Unoptimized Baseline: Disjoint operators with dynamic intermediate buffer allocations
void mlp_naive(int batch, const float *__restrict in, const MLPWeights *__restrict w, float *__restrict out) {
  float *buf_mm1  = (float *)malloc((size_t)batch * HIDDEN1 * sizeof(float));
  float *buf_act1 = (float *)malloc((size_t)batch * HIDDEN1 * sizeof(float));
  float *buf_mm2  = (float *)malloc((size_t)batch * HIDDEN2 * sizeof(float));
  float *buf_act2 = (float *)malloc((size_t)batch * HIDDEN2 * sizeof(float));

  if (!buf_mm1 || !buf_act1 || !buf_mm2 || !buf_act2) {
    free(buf_mm1);
    free(buf_act1);
    free(buf_mm2);
    free(buf_act2);
    return;
  }

  // Layer 1
  for (int b = 0; b < batch; ++b) {
    for (int j = 0; j < HIDDEN1; ++j) buf_mm1[b * HIDDEN1 + j] = 0.0f;
    for (int k = 0; k < IN_FEATURES; ++k) {
      float a = in[b * IN_FEATURES + k];
      for (int j = 0; j < HIDDEN1; ++j) {
        buf_mm1[b * HIDDEN1 + j] += a * w->w1[k * HIDDEN1 + j];
      }
    }
  }
  for (int b = 0; b < batch; ++b) {
    for (int j = 0; j < HIDDEN1; ++j) {
      buf_act1[b * HIDDEN1 + j] = buf_mm1[b * HIDDEN1 + j] + w->b1[j];
    }
  }
  for (int i = 0; i < batch * HIDDEN1; ++i) {
    float val = buf_act1[i];
    buf_act1[i] = val > 0.0f ? val : 0.0f;
  }

  // Layer 2
  for (int b = 0; b < batch; ++b) {
    for (int j = 0; j < HIDDEN2; ++j) buf_mm2[b * HIDDEN2 + j] = 0.0f;
    for (int k = 0; k < HIDDEN1; ++k) {
      float a = buf_act1[b * HIDDEN1 + k];
      for (int j = 0; j < HIDDEN2; ++j) {
        buf_mm2[b * HIDDEN2 + j] += a * w->w2[k * HIDDEN2 + j];
      }
    }
  }
  for (int b = 0; b < batch; ++b) {
    for (int j = 0; j < HIDDEN2; ++j) {
      float sum = buf_mm2[b * HIDDEN2 + j] + w->b2[j];
      buf_act2[b * HIDDEN2 + j] = sum > 0.0f ? sum : 0.0f;
    }
  }

  // Layer 3
  for (int b = 0; b < batch; ++b) {
    for (int j = 0; j < NUM_CLASSES; ++j) out[b * NUM_CLASSES + j] = w->b3[j];
    for (int k = 0; k < HIDDEN2; ++k) {
      float a = buf_act2[b * HIDDEN2 + k];
      for (int j = 0; j < NUM_CLASSES; ++j) {
        out[b * NUM_CLASSES + j] += a * w->w3[k * NUM_CLASSES + j];
      }
    }
  }

  free(buf_mm1);
  free(buf_act1);
  free(buf_mm2);
  free(buf_act2);
}

// 2. Pass 1: Operator Fusion (GEMM Epilogue Fusion)
void mlp_fused(int batch, const float *__restrict in, const MLPWeights *__restrict w,
               float *__restrict out, float *__restrict l1_act, float *__restrict l2_act) {
  for (int b = 0; b < batch; ++b) {
    float *l1_row = &l1_act[b * HIDDEN1];
    for (int j = 0; j < HIDDEN1; ++j) l1_row[j] = w->b1[j];

    for (int k = 0; k < IN_FEATURES; ++k) {
      float a = in[b * IN_FEATURES + k];
      for (int j = 0; j < HIDDEN1; ++j) {
        l1_row[j] += a * w->w1[k * HIDDEN1 + j];
      }
    }
    for (int j = 0; j < HIDDEN1; ++j) {
      float v = l1_row[j];
      l1_row[j] = v > 0.0f ? v : 0.0f;
    }
  }

  for (int b = 0; b < batch; ++b) {
    float *l2_row = &l2_act[b * HIDDEN2];
    for (int j = 0; j < HIDDEN2; ++j) l2_row[j] = w->b2[j];

    for (int k = 0; k < HIDDEN1; ++k) {
      float a = l1_act[b * HIDDEN1 + k];
      for (int j = 0; j < HIDDEN2; ++j) {
        l2_row[j] += a * w->w2[k * HIDDEN2 + j];
      }
    }
    for (int j = 0; j < HIDDEN2; ++j) {
      float v = l2_row[j];
      l2_row[j] = v > 0.0f ? v : 0.0f;
    }
  }

  for (int b = 0; b < batch; ++b) {
    float *out_row = &out[b * NUM_CLASSES];
    for (int j = 0; j < NUM_CLASSES; ++j) out_row[j] = w->b3[j];

    for (int k = 0; k < HIDDEN2; ++k) {
      float a = l2_act[b * HIDDEN2 + k];
      for (int j = 0; j < NUM_CLASSES; ++j) {
        out_row[j] += a * w->w3[k * NUM_CLASSES + j];
      }
    }
  }
}

// 3. Pass 2: Loop Tiling (L1 Cache Blocking)
void mlp_tiled(int batch, const float *__restrict in, const MLPWeights *__restrict w,
               float *__restrict out, float *__restrict l1_act, float *__restrict l2_act) {
  const int TM = 16;
  const int TN = 32;
  const int TK = 64;

  for (int b_out = 0; b_out < batch; b_out += TM) {
    int b_lim = (b_out + TM < batch) ? (b_out + TM) : batch;

    for (int j_out = 0; j_out < HIDDEN1; j_out += TN) {
      int j_lim = (j_out + TN < HIDDEN1) ? (j_out + TN) : HIDDEN1;

      for (int b = b_out; b < b_lim; ++b) {
        for (int j = j_out; j < j_lim; ++j) {
          l1_act[b * HIDDEN1 + j] = w->b1[j];
        }
      }

      for (int k_out = 0; k_out < IN_FEATURES; k_out += TK) {
        int k_lim = (k_out + TK < IN_FEATURES) ? (k_out + TK) : IN_FEATURES;

        for (int b = b_out; b < b_lim; ++b) {
          for (int k = k_out; k < k_lim; ++k) {
            float in_val = in[b * IN_FEATURES + k];
            for (int j = j_out; j < j_lim; ++j) {
              l1_act[b * HIDDEN1 + j] += in_val * w->w1[k * HIDDEN1 + j];
            }
          }
        }
      }

      for (int b = b_out; b < b_lim; ++b) {
        for (int j = j_out; j < j_lim; ++j) {
          float v = l1_act[b * HIDDEN1 + j];
          l1_act[b * HIDDEN1 + j] = v > 0.0f ? v : 0.0f;
        }
      }
    }
  }

  for (int b = 0; b < batch; ++b) {
    float *l2_row = &l2_act[b * HIDDEN2];
    for (int j = 0; j < HIDDEN2; ++j) l2_row[j] = w->b2[j];
    for (int k = 0; k < HIDDEN1; ++k) {
      float a = l1_act[b * HIDDEN1 + k];
      for (int j = 0; j < HIDDEN2; ++j) {
        l2_row[j] += a * w->w2[k * HIDDEN2 + j];
      }
    }
    for (int j = 0; j < HIDDEN2; ++j) {
      float v = l2_row[j];
      l2_row[j] = v > 0.0f ? v : 0.0f;
    }
  }

  for (int b = 0; b < batch; ++b) {
    float *out_row = &out[b * NUM_CLASSES];
    for (int j = 0; j < NUM_CLASSES; ++j) out_row[j] = w->b3[j];
    for (int k = 0; k < HIDDEN2; ++k) {
      float a = l2_act[b * HIDDEN2 + k];
      for (int j = 0; j < NUM_CLASSES; ++j) {
        out_row[j] += a * w->w3[k * NUM_CLASSES + j];
      }
    }
  }
}

// 4. Pass 3: AVX2 SIMD Vectorization (FMA 256-bit)
void mlp_avx2(int batch, const float *__restrict in, const MLPWeights *__restrict w,
              float *__restrict out, float *__restrict l1_act, float *__restrict l2_act) {
  __m256 vzero = _mm256_setzero_ps();

  for (int b = 0; b < batch; ++b) {
    const float *in_row = &in[b * IN_FEATURES];
    float *l1_row = &l1_act[b * HIDDEN1];

    for (int j = 0; j < HIDDEN1; j += 8) {
      _mm256_storeu_ps(&l1_row[j], _mm256_loadu_ps(&w->b1[j]));
    }

    for (int k = 0; k < IN_FEATURES; ++k) {
      __m256 vin = _mm256_set1_ps(in_row[k]);
      const float *wt_row = &w->w1[k * HIDDEN1];

      for (int j = 0; j < HIDDEN1; j += 8) {
        __m256 vacc = _mm256_loadu_ps(&l1_row[j]);
        __m256 vwt  = _mm256_loadu_ps(&wt_row[j]);
        vacc = _mm256_fmadd_ps(vin, vwt, vacc);
        _mm256_storeu_ps(&l1_row[j], vacc);
      }
    }

    for (int j = 0; j < HIDDEN1; j += 8) {
      __m256 vacc = _mm256_loadu_ps(&l1_row[j]);
      vacc = _mm256_max_ps(vacc, vzero);
      _mm256_storeu_ps(&l1_row[j], vacc);
    }
  }

  for (int b = 0; b < batch; ++b) {
    const float *l1_row = &l1_act[b * HIDDEN1];
    float *l2_row = &l2_act[b * HIDDEN2];

    for (int j = 0; j < HIDDEN2; j += 8) {
      _mm256_storeu_ps(&l2_row[j], _mm256_loadu_ps(&w->b2[j]));
    }

    for (int k = 0; k < HIDDEN1; ++k) {
      __m256 vin = _mm256_set1_ps(l1_row[k]);
      const float *wt_row = &w->w2[k * HIDDEN2];

      for (int j = 0; j < HIDDEN2; j += 8) {
        __m256 vacc = _mm256_loadu_ps(&l2_row[j]);
        __m256 vwt  = _mm256_loadu_ps(&wt_row[j]);
        vacc = _mm256_fmadd_ps(vin, vwt, vacc);
        _mm256_storeu_ps(&l2_row[j], vacc);
      }
    }

    for (int j = 0; j < HIDDEN2; j += 8) {
      __m256 vacc = _mm256_loadu_ps(&l2_row[j]);
      vacc = _mm256_max_ps(vacc, vzero);
      _mm256_storeu_ps(&l2_row[j], vacc);
    }
  }

  for (int b = 0; b < batch; ++b) {
    const float *l2_row = &l2_act[b * HIDDEN2];
    float *out_row = &out[b * NUM_CLASSES];

    for (int j = 0; j < NUM_CLASSES; ++j) out_row[j] = w->b3[j];

    for (int k = 0; k < HIDDEN2; ++k) {
      float a = l2_row[k];
      for (int j = 0; j < NUM_CLASSES; ++j) {
        out_row[j] += a * w->w3[k * NUM_CLASSES + j];
      }
    }
  }
}

static void print_usage(const char *prog_name) {
  printf("Usage: %s [options]\n", prog_name);
  printf("Options:\n");
  printf("  -n, --iters <N>       Benchmark iterations (default: %d)\n", DEFAULT_BENCH_ITERS);
  printf("  -b, --batch <B>       Batch size (default: %d)\n", DEFAULT_BATCH_SIZE);
  printf("  -h, --help            Show this help message\n");
}

int main(int argc, char **argv) {
  int bench_iters = DEFAULT_BENCH_ITERS;
  int batch_size = DEFAULT_BATCH_SIZE;

  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--iters") == 0) {
      if (i + 1 < argc) bench_iters = atoi(argv[++i]);
    } else if (strcmp(argv[i], "-b") == 0 || strcmp(argv[i], "--batch") == 0) {
      if (i + 1 < argc) batch_size = atoi(argv[++i]);
    } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
      print_usage(argv[0]);
      return 0;
    }
  }

  printf("Benchmark: MLP 3-Layer [batch=%d, iters=%d, in=%d, hidden=[%d, %d], out=%d]\n",
         batch_size, bench_iters, IN_FEATURES, HIDDEN1, HIDDEN2, NUM_CLASSES);

  MLPWeights *weights = (MLPWeights *)aligned_alloc(32, sizeof(MLPWeights));
  float *input_data   = (float *)aligned_alloc(32, (size_t)batch_size * IN_FEATURES * sizeof(float));
  float *out_naive    = (float *)aligned_alloc(32, (size_t)batch_size * NUM_CLASSES * sizeof(float));
  float *out_fused    = (float *)aligned_alloc(32, (size_t)batch_size * NUM_CLASSES * sizeof(float));
  float *out_tiled    = (float *)aligned_alloc(32, (size_t)batch_size * NUM_CLASSES * sizeof(float));
  float *out_avx2     = (float *)aligned_alloc(32, (size_t)batch_size * NUM_CLASSES * sizeof(float));
  float *l1_act       = (float *)aligned_alloc(32, (size_t)batch_size * HIDDEN1 * sizeof(float));
  float *l2_act       = (float *)aligned_alloc(32, (size_t)batch_size * HIDDEN2 * sizeof(float));

  if (!weights || !input_data || !out_naive || !out_fused || !out_tiled || !out_avx2 || !l1_act || !l2_act) {
    fprintf(stderr, "Error: Memory allocation failed.\n");
    return 1;
  }

  char weights_path[512];
  if (vision_resolve_path("models/mlp_weights.bin", weights_path, sizeof(weights_path))) {
    FILE *f = fopen(weights_path, "rb");
    if (f) {
      size_t read_bytes = fread(weights, 1, sizeof(MLPWeights), f);
      fclose(f);
      if (read_bytes != sizeof(MLPWeights)) {
        fprintf(stderr, "warning: weights file truncated (%zu / %zu bytes)\n", read_bytes, sizeof(MLPWeights));
      }
    }
  } else {
    float *ptr = (float *)weights;
    for (size_t i = 0; i < sizeof(MLPWeights) / sizeof(float); ++i)
      ptr[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
  }

  char input_path[512];
  if (vision_resolve_path("models/test_input.bin", input_path, sizeof(input_path))) {
    FILE *fin = fopen(input_path, "rb");
    if (fin) {
      float single_sample[IN_FEATURES];
      size_t read_cnt = fread(single_sample, sizeof(float), IN_FEATURES, fin);
      fclose(fin);
      if (read_cnt == IN_FEATURES) {
        for (int b = 0; b < batch_size; ++b) {
          memcpy(&input_data[b * IN_FEATURES], single_sample, IN_FEATURES * sizeof(float));
        }
      }
    }
  } else {
    for (int i = 0; i < batch_size * IN_FEATURES; ++i)
      input_data[i] = ((float)rand() / RAND_MAX) * 0.5f;
  }

  // 1. Numerical Verification
  mlp_naive(batch_size, input_data, weights, out_naive);
  mlp_fused(batch_size, input_data, weights, out_fused, l1_act, l2_act);
  mlp_tiled(batch_size, input_data, weights, out_tiled, l1_act, l2_act);
  mlp_avx2(batch_size, input_data, weights, out_avx2, l1_act, l2_act);

  float max_diff_fused = 0.0f, max_diff_tiled = 0.0f, max_diff_avx2 = 0.0f;
  for (int i = 0; i < batch_size * NUM_CLASSES; ++i) {
    float d1 = fabsf(out_fused[i] - out_naive[i]);
    float d2 = fabsf(out_tiled[i] - out_naive[i]);
    float d3 = fabsf(out_avx2[i] - out_naive[i]);
    if (d1 > max_diff_fused) max_diff_fused = d1;
    if (d2 > max_diff_tiled) max_diff_tiled = d2;
    if (d3 > max_diff_avx2)  max_diff_avx2  = d3;
  }

  printf("Verification:\n");
  printf("  fused vs naive : max diff = %.2e (%s)\n",
         max_diff_fused, max_diff_fused < 1e-4f ? "ok" : "fail");
  printf("  tiled vs naive : max diff = %.2e (%s)\n",
         max_diff_tiled, max_diff_tiled < 1e-4f ? "ok" : "fail");
  printf("  avx2  vs naive : max diff = %.2e (%s)\n",
         max_diff_avx2,  max_diff_avx2  < 1e-4f ? "ok" : "fail");

  char ref_path[512];
  if (vision_resolve_path("models/reference_output.bin", ref_path, sizeof(ref_path))) {
    FILE *fref = fopen(ref_path, "rb");
    if (fref) {
      float ref_out[NUM_CLASSES];
      size_t read_cnt = fread(ref_out, sizeof(float), NUM_CLASSES, fref);
      fclose(fref);
      if (read_cnt == NUM_CLASSES) {
        float max_err_pytorch = 0.0f;
        int pred_class = 0;
        float max_logit = out_avx2[0];
        for (int j = 0; j < NUM_CLASSES; ++j) {
          float err = fabsf(out_avx2[j] - ref_out[j]);
          if (err > max_err_pytorch) max_err_pytorch = err;
          if (out_avx2[j] > max_logit) {
            max_logit = out_avx2[j];
            pred_class = j;
          }
        }
        printf("  pytorch ground truth: max diff = %.2e, pred class = %d (%s)\n",
               max_err_pytorch, pred_class, max_err_pytorch < 1e-4f ? "ok" : "fail");
      }
    }
  }

  // 2. Latency Benchmarking
  for (int i = 0; i < 20; ++i) {
    mlp_avx2(batch_size, input_data, weights, out_avx2, l1_act, l2_act);
  }

  double t_naive = 0, t_fused = 0, t_tiled = 0, t_avx2 = 0;

  for (int it = 0; it < bench_iters; ++it) {
    uint64_t t0 = get_time_ns();
    mlp_naive(batch_size, input_data, weights, out_naive);
    t_naive += (double)(get_time_ns() - t0);

    t0 = get_time_ns();
    mlp_fused(batch_size, input_data, weights, out_fused, l1_act, l2_act);
    t_fused += (double)(get_time_ns() - t0);

    t0 = get_time_ns();
    mlp_tiled(batch_size, input_data, weights, out_tiled, l1_act, l2_act);
    t_tiled += (double)(get_time_ns() - t0);

    t0 = get_time_ns();
    mlp_avx2(batch_size, input_data, weights, out_avx2, l1_act, l2_act);
    t_avx2 += (double)(get_time_ns() - t0);
  }

  double ms_naive = (t_naive / bench_iters) / 1e6;
  double ms_fused = (t_fused / bench_iters) / 1e6;
  double ms_tiled = (t_tiled / bench_iters) / 1e6;
  double ms_avx2  = (t_avx2  / bench_iters) / 1e6;

  double us_per_sample_naive = (ms_naive * 1000.0) / batch_size;
  double us_per_sample_avx2  = (ms_avx2  * 1000.0) / batch_size;

  printf("\n%-24s %12s %14s %10s\n", "Benchmark", "Time (ms)", "Time/Sample", "Speedup");
  printf("----------------------------------------------------------------\n");
  printf("%-24s %9.3f ms %11.2f us %9.2fx\n", "mlp_naive", ms_naive, us_per_sample_naive, 1.0);
  printf("%-24s %9.3f ms %11.2f us %9.2fx\n", "mlp_fused", ms_fused, (ms_fused * 1000.0) / batch_size, ms_naive / ms_fused);
  printf("%-24s %9.3f ms %11.2f us %9.2fx\n", "mlp_tiled", ms_tiled, (ms_tiled * 1000.0) / batch_size, ms_naive / ms_tiled);
  printf("%-24s %9.3f ms %11.2f us %9.2fx\n", "mlp_avx2",  ms_avx2,  us_per_sample_avx2, ms_naive / ms_avx2);
  printf("----------------------------------------------------------------\n");

  free(weights);
  free(input_data);
  free(out_naive);
  free(out_fused);
  free(out_tiled);
  free(out_avx2);
  free(l1_act);
  free(l2_act);

  return 0;
}
