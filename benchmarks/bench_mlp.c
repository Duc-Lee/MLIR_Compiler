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

// Kien truc Neural Network (MNIST Multi-Layer Perceptron):
// Input:  [B, 784]
// FC1:    [784, 128] + Bias [128] + ReLU
// FC2:    [128, 64]  + Bias [64]  + ReLU
// FC3:    [64, 10]   + Bias [10]  (Logits)
#define IN_FEATURES  784
#define HIDDEN1      128
#define HIDDEN2      64
#define NUM_CLASSES  10
#define BATCH_SIZE   64
#define BENCH_ITERS  500

typedef struct {
  float w1[IN_FEATURES * HIDDEN1];
  float b1[HIDDEN1];
  float w2[HIDDEN1 * HIDDEN2];
  float b2[HIDDEN2];
  float w3[HIDDEN2 * NUM_CLASSES];
  float b3[NUM_CLASSES];
} MLPWeights;

// 1. Unoptimized Baseline: Moi toan tu thuc thi tach roi, cap phat buffer dong moi lan goi
void mlp_naive(int batch, const float *__restrict in, const MLPWeights *__restrict w, float *__restrict out) {
  float *buf_mm1  = (float *)malloc(batch * HIDDEN1 * sizeof(float));
  float *buf_act1 = (float *)malloc(batch * HIDDEN1 * sizeof(float));
  float *buf_mm2  = (float *)malloc(batch * HIDDEN2 * sizeof(float));
  float *buf_act2 = (float *)malloc(batch * HIDDEN2 * sizeof(float));

  // --- LAYER 1 ---
  // GEMM 1: [B, 784] x [784, 128] -> [B, 128]
  for (int b = 0; b < batch; ++b) {
    for (int j = 0; j < HIDDEN1; ++j) buf_mm1[b * HIDDEN1 + j] = 0.0f;
    for (int k = 0; k < IN_FEATURES; ++k) {
      float a = in[b * IN_FEATURES + k];
      for (int j = 0; j < HIDDEN1; ++j) {
        buf_mm1[b * HIDDEN1 + j] += a * w->w1[k * HIDDEN1 + j];
      }
    }
  }
  // Bias 1: doc lai RAM
  for (int b = 0; b < batch; ++b) {
    for (int j = 0; j < HIDDEN1; ++j) {
      buf_act1[b * HIDDEN1 + j] = buf_mm1[b * HIDDEN1 + j] + w->b1[j];
    }
  }
  // ReLU 1: doc lai RAM
  for (int i = 0; i < batch * HIDDEN1; ++i) {
    float val = buf_act1[i];
    buf_act1[i] = val > 0.0f ? val : 0.0f;
  }

  // --- LAYER 2 ---
  // GEMM 2: [B, 128] x [128, 64] -> [B, 64]
  for (int b = 0; b < batch; ++b) {
    for (int j = 0; j < HIDDEN2; ++j) buf_mm2[b * HIDDEN2 + j] = 0.0f;
    for (int k = 0; k < HIDDEN1; ++k) {
      float a = buf_act1[b * HIDDEN1 + k];
      for (int j = 0; j < HIDDEN2; ++j) {
        buf_mm2[b * HIDDEN2 + j] += a * w->w2[k * HIDDEN2 + j];
      }
    }
  }
  // Bias 2 + ReLU 2
  for (int b = 0; b < batch; ++b) {
    for (int j = 0; j < HIDDEN2; ++j) {
      float sum = buf_mm2[b * HIDDEN2 + j] + w->b2[j];
      buf_act2[b * HIDDEN2 + j] = sum > 0.0f ? sum : 0.0f;
    }
  }

  // --- LAYER 3 ---
  // GEMM 3: [B, 64] x [64, 10] -> [B, 10] (Logits)
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

// 2. PASS 1: Operator Fusion (GEMM Epilogue Fusion)
// MatMul + Bias + ReLU duoc tinh luon tren thanh ghi accumulator, khong cap phat buffer trung gian
void mlp_fused(int batch, const float *__restrict in, const MLPWeights *__restrict w,
               float *__restrict out, float *__restrict l1_act, float *__restrict l2_act) {
  // Layer 1: Fused GEMM + Bias + ReLU
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

  // Layer 2: Fused GEMM + Bias + ReLU
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

  // Layer 3: Fused GEMM + Bias
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

// 3. PASS 2: Loop Tiling (Cache Blocking cho M, N, K vua khit L1 Cache 32KB)
void mlp_tiled(int batch, const float *__restrict in, const MLPWeights *__restrict w,
               float *__restrict out, float *__restrict l1_act, float *__restrict l2_act) {
  const int TM = 16;
  const int TN = 32;
  const int TK = 64;

  // Layer 1 Tiled
  for (int b_out = 0; b_out < batch; b_out += TM) {
    int b_lim = (b_out + TM < batch) ? (b_out + TM) : batch;

    for (int j_out = 0; j_out < HIDDEN1; j_out += TN) {
      int j_lim = (j_out + TN < HIDDEN1) ? (j_out + TN) : HIDDEN1;

      // Khoi tao tile bang bias
      for (int b = b_out; b < b_lim; ++b) {
        for (int j = j_out; j < j_lim; ++j) {
          l1_act[b * HIDDEN1 + j] = w->b1[j];
        }
      }

      // Duyet tile theo K (Cache locality cao nhat)
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

      // Fused ReLU tren tile
      for (int b = b_out; b < b_lim; ++b) {
        for (int j = j_out; j < j_lim; ++j) {
          float v = l1_act[b * HIDDEN1 + j];
          l1_act[b * HIDDEN1 + j] = v > 0.0f ? v : 0.0f;
        }
      }
    }
  }

  // Layer 2 Fused
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

  // Layer 3 Fused
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

// 4. PASS 3: AVX2 SIMD Vectorization (FMA 256-bit: 8 phep tinh float song song)
void mlp_avx2(int batch, const float *__restrict in, const MLPWeights *__restrict w,
              float *__restrict out, float *__restrict l1_act, float *__restrict l2_act) {
  __m256 vzero = _mm256_setzero_ps();

  // Layer 1: Vectorized theo HIDDEN1 (128 = 16 vector 8-wide)
  for (int b = 0; b < batch; ++b) {
    const float *in_row = &in[b * IN_FEATURES];
    float *l1_row = &l1_act[b * HIDDEN1];

    // Load bias 1
    for (int j = 0; j < HIDDEN1; j += 8) {
      _mm256_storeu_ps(&l1_row[j], _mm256_loadu_ps(&w->b1[j]));
    }

    // FMA Loop: nhan broadcast vector A voi hang weights
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

    // Fused ReLU 1
    for (int j = 0; j < HIDDEN1; j += 8) {
      __m256 vacc = _mm256_loadu_ps(&l1_row[j]);
      vacc = _mm256_max_ps(vacc, vzero);
      _mm256_storeu_ps(&l1_row[j], vacc);
    }
  }

  // Layer 2: Vectorized theo HIDDEN2 (64 = 8 vector 8-wide)
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

  // Layer 3: Fused Logits
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

int main(int argc, char **argv) {
  printf("=================================================================\n");
  printf("  Benchmark End-to-End Neural Network (MNIST 3-Layer MLP)\n");
  printf("  Kien truc: 784 -> 128 (ReLU) -> 64 (ReLU) -> 10 (Logits)\n");
  printf("  Batch size: %d | So lan do: %d iters\n", BATCH_SIZE, BENCH_ITERS);
  printf("=================================================================\n\n");

  MLPWeights *weights = (MLPWeights *)aligned_alloc(32, sizeof(MLPWeights));
  float *input_data   = (float *)aligned_alloc(32, BATCH_SIZE * IN_FEATURES * sizeof(float));
  float *out_naive    = (float *)aligned_alloc(32, BATCH_SIZE * NUM_CLASSES * sizeof(float));
  float *out_fused    = (float *)aligned_alloc(32, BATCH_SIZE * NUM_CLASSES * sizeof(float));
  float *out_tiled    = (float *)aligned_alloc(32, BATCH_SIZE * NUM_CLASSES * sizeof(float));
  float *out_avx2     = (float *)aligned_alloc(32, BATCH_SIZE * NUM_CLASSES * sizeof(float));
  float *l1_act       = (float *)aligned_alloc(32, BATCH_SIZE * HIDDEN1 * sizeof(float));
  float *l2_act       = (float *)aligned_alloc(32, BATCH_SIZE * HIDDEN2 * sizeof(float));

  // Doc weights tu binary neu co
  const char *weights_file = "models/mlp_weights.bin";
  FILE *f = fopen(weights_file, "rb");
  if (f) {
    size_t read_bytes = fread(weights, 1, sizeof(MLPWeights), f);
    fclose(f);
    printf("[+] Da nạp trong so that tu PyTorch: %s (%zu KB)\n", weights_file, read_bytes / 1024);
  } else {
    printf("[!] Khong tim thay file %s, khoi tao random weights...\n", weights_file);
    float *ptr = (float *)weights;
    for (size_t i = 0; i < sizeof(MLPWeights)/sizeof(float); ++i)
      ptr[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
  }

  // Doc test input
  const char *input_file = "models/test_input.bin";
  FILE *fin = fopen(input_file, "rb");
  if (fin) {
    float single_sample[IN_FEATURES];
    fread(single_sample, sizeof(float), IN_FEATURES, fin);
    fclose(fin);
    // Sao chep cho toan bo batch
    for (int b = 0; b < BATCH_SIZE; ++b) {
      memcpy(&input_data[b * IN_FEATURES], single_sample, IN_FEATURES * sizeof(float));
    }
    printf("[+] Da nạp test input tu: %s\n", input_file);
  } else {
    for (int i = 0; i < BATCH_SIZE * IN_FEATURES; ++i)
      input_data[i] = ((float)rand() / RAND_MAX) * 0.5f;
  }

  // 1. Kiem tra tinh dung dan (Verification)
  printf("\n[1] Kiem tra do chinh xac so hoc giua cac cap do toi uu...\n");
  mlp_naive(BATCH_SIZE, input_data, weights, out_naive);
  mlp_fused(BATCH_SIZE, input_data, weights, out_fused, l1_act, l2_act);
  mlp_tiled(BATCH_SIZE, input_data, weights, out_tiled, l1_act, l2_act);
  mlp_avx2(BATCH_SIZE, input_data, weights, out_avx2, l1_act, l2_act);

  float max_diff_fused = 0.0f, max_diff_tiled = 0.0f, max_diff_avx2 = 0.0f;
  for (int i = 0; i < BATCH_SIZE * NUM_CLASSES; ++i) {
    float d1 = fabsf(out_fused[i] - out_naive[i]);
    float d2 = fabsf(out_tiled[i] - out_naive[i]);
    float d3 = fabsf(out_avx2[i] - out_naive[i]);
    if (d1 > max_diff_fused) max_diff_fused = d1;
    if (d2 > max_diff_tiled) max_diff_tiled = d2;
    if (d3 > max_diff_avx2)  max_diff_avx2  = d3;
  }

  printf("    - Sai so lon nhat Fused vs Naive : %.6e %s\n",
         max_diff_fused, max_diff_fused < 1e-4f ? "[OK]" : "[FAIL]");
  printf("    - Sai so lon nhat Tiled vs Naive : %.6e %s\n",
         max_diff_tiled, max_diff_tiled < 1e-4f ? "[OK]" : "[FAIL]");
  printf("    - Sai so lon nhat AVX2 vs Naive  : %.6e %s\n",
         max_diff_avx2, max_diff_avx2 < 1e-4f ? "[OK]" : "[FAIL]");

  // 2. Kiem tra voi Ground Truth tu PyTorch
  const char *ref_file = "models/reference_output.bin";
  FILE *fref = fopen(ref_file, "rb");
  if (fref) {
    float ref_out[NUM_CLASSES];
    fread(ref_out, sizeof(float), NUM_CLASSES, fref);
    fclose(fref);

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
    printf("\n[2] So sanh Ground Truth truc tiep voi PyTorch Eager:\n");
    printf("    - Du doan lop (Predicted Class)     : %d\n", pred_class);
    printf("    - Max Absolute Error vs PyTorch     : %.6e %s\n",
           max_err_pytorch, max_err_pytorch < 1e-4f ? "[PASSED 100%]" : "[FAIL]");
  }

  // 3. Do dac toc do (Benchmarking)
  printf("\n[3] Do dac thoi gian thuc thi (Benchmark %d iterations)...\n", BENCH_ITERS);

  // Warmup
  for (int i = 0; i < 20; ++i) {
    mlp_avx2(BATCH_SIZE, input_data, weights, out_avx2, l1_act, l2_act);
  }

  double t_naive = 0, t_fused = 0, t_tiled = 0, t_avx2 = 0;

  for (int it = 0; it < BENCH_ITERS; ++it) {
    uint64_t t0 = get_time_ns();
    mlp_naive(BATCH_SIZE, input_data, weights, out_naive);
    t_naive += (get_time_ns() - t0);

    t0 = get_time_ns();
    mlp_fused(BATCH_SIZE, input_data, weights, out_fused, l1_act, l2_act);
    t_fused += (get_time_ns() - t0);

    t0 = get_time_ns();
    mlp_tiled(BATCH_SIZE, input_data, weights, out_tiled, l1_act, l2_act);
    t_tiled += (get_time_ns() - t0);

    t0 = get_time_ns();
    mlp_avx2(BATCH_SIZE, input_data, weights, out_avx2, l1_act, l2_act);
    t_avx2 += (get_time_ns() - t0);
  }

  double ms_naive = (t_naive / BENCH_ITERS) / 1e6;
  double ms_fused = (t_fused / BENCH_ITERS) / 1e6;
  double ms_tiled = (t_tiled / BENCH_ITERS) / 1e6;
  double ms_avx2  = (t_avx2  / BENCH_ITERS) / 1e6;

  double us_per_sample_naive = (ms_naive * 1000.0) / BATCH_SIZE;
  double us_per_sample_avx2  = (ms_avx2  * 1000.0) / BATCH_SIZE;

  printf("\n=================================================================\n");
  printf("                     KET QUA SO SANH TOI UU                      \n");
  printf("=================================================================\n");
  printf("1. Unoptimized (3 Pass, Dynamic Alloc) : %7.3f ms (1.00x)  [%5.1f us/sample]\n",
         ms_naive, us_per_sample_naive);
  printf("2. PASS 1 (Operator Fusion)           : %7.3f ms (%5.2fx)\n",
         ms_fused, ms_naive / ms_fused);
  printf("3. PASS 2 (Loop Tiling L1 Cache)       : %7.3f ms (%5.2fx)\n",
         ms_tiled, ms_naive / ms_tiled);
  printf("4. PASS 3 (AVX2 SIMD Vectorization)    : %7.3f ms (%5.2fx)  [%5.1f us/sample]\n",
         ms_avx2, ms_naive / ms_avx2, us_per_sample_avx2);
  printf("=================================================================\n");
  printf("[+] Tang toc tong cong (Speedup): %.2f LAN so voi cai dat ban dau!\n\n",
         ms_naive / ms_avx2);

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
