#include "vision_runtime.h"
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// kich thuoc layer thuong thay trong ResNet-18 (input 56x56, 64 channels, 3x3 kernel)
#define BATCH 1
#define IN_C 64
#define OUT_C 64
#define H 56
#define W 56
#define KH 3
#define KW 3
#define PAD 1
#define STRIDE 1
#define OUT_H ((H + 2 * PAD - KH) / STRIDE + 1)
#define OUT_W ((W + 2 * PAD - KW) / STRIDE + 1)

// 1. Unoptimized: conv tach roi, bias add tach roi, relu tach roi (doc ghi RAM 3 lan)
void conv_naive(const float *in, const float *weight, const float *bias,
                float *out, float *tmp_buf1, float *tmp_buf2) {
  // buoc 1: conv nguyen ban ghi ra buffer trung gian 1
  for (int oc = 0; oc < OUT_C; ++oc) {
    for (int oh = 0; oh < OUT_H; ++oh) {
      for (int ow = 0; ow < OUT_W; ++ow) {
        float sum = 0.0f;
        for (int ic = 0; ic < IN_C; ++ic) {
          for (int kh = 0; kh < KH; ++kh) {
            for (int kw = 0; kw < KW; ++kw) {
              int ih = oh * STRIDE - PAD + kh;
              int iw = ow * STRIDE - PAD + kw;
              if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                int in_idx = (ic * H + ih) * W + iw;
                int wt_idx = ((oc * IN_C + ic) * KH + kh) * KW + kw;
                sum += in[in_idx] * weight[wt_idx];
              }
            }
          }
        }
        int out_idx = (oc * OUT_H + oh) * OUT_W + ow;
        tmp_buf1[out_idx] = sum;
      }
    }
  }

  // buoc 2: cong bias vao buffer trung gian 2 (duyet lai RAM)
  for (int oc = 0; oc < OUT_C; ++oc) {
    float b = bias[oc];
    for (int i = 0; i < OUT_H * OUT_W; ++i) {
      int idx = oc * OUT_H * OUT_W + i;
      tmp_buf2[idx] = tmp_buf1[idx] + b;
    }
  }

  // buoc 3: relu activation ghi vao out (duyet lai RAM lan 3)
  for (int i = 0; i < OUT_C * OUT_H * OUT_W; ++i) {
    float val = tmp_buf2[i];
    out[i] = val > 0.0f ? val : 0.0f;
  }
}

// 2. Pass 1: Operator Fusion (gop bias va relu thang vao vong lap conv)
void conv_fused(const float *in, const float *weight, const float *bias, float *out) {
  for (int oc = 0; oc < OUT_C; ++oc) {
    float b = bias[oc];
    for (int oh = 0; oh < OUT_H; ++oh) {
      for (int ow = 0; ow < OUT_W; ++ow) {
        float sum = b; // khoi tao bang bias, khong can mang trung gian
        for (int ic = 0; ic < IN_C; ++ic) {
          for (int kh = 0; kh < KH; ++kh) {
            for (int kw = 0; kw < KW; ++kw) {
              int ih = oh * STRIDE - PAD + kh;
              int iw = ow * STRIDE - PAD + kw;
              if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                int in_idx = (ic * H + ih) * W + iw;
                int wt_idx = ((oc * IN_C + ic) * KH + kh) * KW + kw;
                sum += in[in_idx] * weight[wt_idx];
              }
            }
          }
        }
        // fuse relu truc tiep truoc khi luu vao output
        int out_idx = (oc * OUT_H + oh) * OUT_W + ow;
        out[out_idx] = sum > 0.0f ? sum : 0.0f;
      }
    }
  }
}

// 3. Pass 2: Loop Tiling (chia nho block de vua L1 cache 32KB)
void conv_fused_tiled(const float *in, const float *weight, const float *bias, float *out) {
  const int TILE_OC = 16;
  const int TILE_OH = 14;
  const int TILE_OW = 14;

  for (int oc_outer = 0; oc_outer < OUT_C; oc_outer += TILE_OC) {
    for (int oh_outer = 0; oh_outer < OUT_H; oh_outer += TILE_OH) {
      for (int ow_outer = 0; ow_outer < OUT_W; ow_outer += TILE_OW) {

        for (int oc = oc_outer; oc < oc_outer + TILE_OC && oc < OUT_C; ++oc) {
          float b = bias[oc];
          for (int oh = oh_outer; oh < oh_outer + TILE_OH && oh < OUT_H; ++oh) {
            for (int ow = ow_outer; ow < ow_outer + TILE_OW && ow < OUT_W; ++ow) {
              float sum = b;
              for (int ic = 0; ic < IN_C; ++ic) {
                for (int kh = 0; kh < KH; ++kh) {
                  for (int kw = 0; kw < KW; ++kw) {
                    int ih = oh * STRIDE - PAD + kh;
                    int iw = ow * STRIDE - PAD + kw;
                    if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                      int in_idx = (ic * H + ih) * W + iw;
                      int wt_idx = ((oc * IN_C + ic) * KH + kh) * KW + kw;
                      sum += in[in_idx] * weight[wt_idx];
                    }
                  }
                }
              }
              int out_idx = (oc * OUT_H + oh) * OUT_W + ow;
              out[out_idx] = sum > 0.0f ? sum : 0.0f;
            }
          }
        }

      }
    }
  }
}

// 4. Pass 3: Vectorization (AVX2 SIMD + FMA 256-bit tren 8 phan tu ow lien tiep)
void conv_fused_tiled_avx2(const float *in, const float *weight, const float *bias, float *out) {
  const int TILE_OC = 16;
  const int TILE_OH = 14;
  const __m256 vzero = _mm256_setzero_ps();

  // vung dem co padding de vector load lien tuc khong loi bien
  float *in_pad = (float *)aligned_alloc(64, IN_C * (H + 2) * (W + 2) * sizeof(float));
  memset(in_pad, 0, IN_C * (H + 2) * (W + 2) * sizeof(float));
  for (int ic = 0; ic < IN_C; ++ic) {
    for (int ih = 0; ih < H; ++ih) {
      memcpy(&in_pad[(ic * (H + 2) + ih + 1) * (W + 2) + 1],
             &in[(ic * H + ih) * W],
             W * sizeof(float));
    }
  }

  for (int oc_outer = 0; oc_outer < OUT_C; oc_outer += TILE_OC) {
    for (int oh_outer = 0; oh_outer < OUT_H; oh_outer += TILE_OH) {

      for (int oc = oc_outer; oc < oc_outer + TILE_OC && oc < OUT_C; ++oc) {
        __m256 vbias = _mm256_set1_ps(bias[oc]);

        for (int oh = oh_outer; oh < oh_outer + TILE_OH && oh < OUT_H; ++oh) {
          for (int ow = 0; ow < OUT_W; ow += 8) {
            __m256 vsum = vbias;

            for (int ic = 0; ic < IN_C; ++ic) {
              const float *in_c = &in_pad[ic * (H + 2) * (W + 2)];

              for (int kh = 0; kh < KH; ++kh) {
                const float *in_row = &in_c[(oh + kh) * (W + 2) + ow];
                const float *w_row = &weight[((oc * IN_C + ic) * KH + kh) * KW];

                __m256 vw0 = _mm256_set1_ps(w_row[0]);
                __m256 vw1 = _mm256_set1_ps(w_row[1]);
                __m256 vw2 = _mm256_set1_ps(w_row[2]);

                __m256 vin0 = _mm256_loadu_ps(&in_row[0]);
                __m256 vin1 = _mm256_loadu_ps(&in_row[1]);
                __m256 vin2 = _mm256_loadu_ps(&in_row[2]);

                vsum = _mm256_fmadd_ps(vin0, vw0, vsum);
                vsum = _mm256_fmadd_ps(vin1, vw1, vsum);
                vsum = _mm256_fmadd_ps(vin2, vw2, vsum);
              }
            }

            // fuse relu bang phep max voi vector 0
            vsum = _mm256_max_ps(vsum, vzero);
            _mm256_storeu_ps(&out[(oc * OUT_H + oh) * OUT_W + ow], vsum);
          }
        }
      }

    }
  }

  free(in_pad);
}

int main(void) {
  printf("===============================================================\n");
  printf("  Vision Compiler Benchmark: Conv2D + Bias + ReLU (ResNet-18)\n");
  printf("  Input: [1, %d, %d, %d], Kernel: [%d, %d, %d, %d]\n",
         IN_C, H, W, OUT_C, IN_C, KH, KW);
  printf("===============================================================\n\n");

  size_t in_size = IN_C * H * W;
  size_t wt_size = OUT_C * IN_C * KH * KW;
  size_t bs_size = OUT_C;
  size_t out_size = OUT_C * OUT_H * OUT_W;

  float *in = (float *)aligned_alloc(64, in_size * sizeof(float));
  float *weight = (float *)aligned_alloc(64, wt_size * sizeof(float));
  float *bias = (float *)aligned_alloc(64, bs_size * sizeof(float));
  float *out_naive = (float *)aligned_alloc(64, out_size * sizeof(float));
  float *out_fused = (float *)aligned_alloc(64, out_size * sizeof(float));
  float *out_tiled = (float *)aligned_alloc(64, out_size * sizeof(float));
  float *out_avx2 = (float *)aligned_alloc(64, out_size * sizeof(float));

  float *tmp1 = (float *)malloc(out_size * sizeof(float));
  float *tmp2 = (float *)malloc(out_size * sizeof(float));

  vision_init_tensor_random(in, in_size);
  vision_init_tensor_random(weight, wt_size);
  vision_init_tensor_random(bias, bs_size);

  const int WARMUP = 2;
  const int ITERS = 5;

  // 1. Unoptimized
  for (int i = 0; i < WARMUP; ++i)
    conv_naive(in, weight, bias, out_naive, tmp1, tmp2);
  vision_timer_start();
  for (int i = 0; i < ITERS; ++i)
    conv_naive(in, weight, bias, out_naive, tmp1, tmp2);
  double time_naive = vision_timer_stop_ms() / ITERS;

  // 2. Pass 1: Fused
  for (int i = 0; i < WARMUP; ++i)
    conv_fused(in, weight, bias, out_fused);
  vision_timer_start();
  for (int i = 0; i < ITERS; ++i)
    conv_fused(in, weight, bias, out_fused);
  double time_fused = vision_timer_stop_ms() / ITERS;

  // 3. Pass 2: Fused + Tiled
  for (int i = 0; i < WARMUP; ++i)
    conv_fused_tiled(in, weight, bias, out_tiled);
  vision_timer_start();
  for (int i = 0; i < ITERS; ++i)
    conv_fused_tiled(in, weight, bias, out_tiled);
  double time_tiled = vision_timer_stop_ms() / ITERS;

  // 4. Pass 3: Fused + Tiled + AVX2 Vectorized
  for (int i = 0; i < WARMUP; ++i)
    conv_fused_tiled_avx2(in, weight, bias, out_avx2);
  vision_timer_start();
  for (int i = 0; i < ITERS; ++i)
    conv_fused_tiled_avx2(in, weight, bias, out_avx2);
  double time_avx2 = vision_timer_stop_ms() / ITERS;

  // Xac minh tinh dung dan so hoc
  printf("--- Kiem Tra Tinh Dung Dan So Hoc ---\n");
  vision_verify_results(out_fused, out_naive, out_size, 1e-4f);
  vision_verify_results(out_tiled, out_naive, out_size, 1e-4f);
  vision_verify_results(out_avx2, out_naive, out_size, 1e-3f);

  // In ket qua benchmark
  double gops = (2.0 * OUT_C * OUT_H * OUT_W * IN_C * KH * KW) / 1e9;
  printf("\n--- Ket Qua Benchmark (Trung Binh %d Lan) ---\n", ITERS);
  printf("%-30s | %12s | %10s | %10s\n", "Pipeline Stage", "Latency (ms)", "GFLOPS", "Speedup");
  printf("-----------------------------------------------------------------------\n");
  printf("%-30s | %12.2f | %10.2f | %9.1fx\n",
         "1. Unoptimized (3 Pass)", time_naive, gops / (time_naive / 1000.0), 1.0);
  printf("%-30s | %12.2f | %10.2f | %9.1fx\n",
         "2. Pass 1: Operator Fusion", time_fused, gops / (time_fused / 1000.0), time_naive / time_fused);
  printf("%-30s | %12.2f | %10.2f | %9.1fx\n",
         "3. Pass 2: Fused + Tiling", time_tiled, gops / (time_tiled / 1000.0), time_naive / time_tiled);
  printf("%-30s | %12.2f | %10.2f | %9.1fx\n",
         "4. Pass 3: Tiled + AVX2 SIMD", time_avx2, gops / (time_avx2 / 1000.0), time_naive / time_avx2);
  printf("-----------------------------------------------------------------------\n");

  free(in);
  free(weight);
  free(bias);
  free(out_naive);
  free(out_fused);
  free(out_tiled);
  free(out_avx2);
  free(tmp1);
  free(tmp2);

  return 0;
}
