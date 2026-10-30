# Tài Liệu Kỹ Thuật: Dự Án MLIR Deep Learning Compiler

Tài liệu này mô tả chi tiết bài toán, kiến trúc hệ thống, các giải thuật tối ưu hóa và quy trình biên dịch thực tế của dự án **MLIR Deep Learning Compiler**.

---

## 1. Dự Án Này Làm Gì? (Project Overview)

Đây là một **Domain-Specific Compiler (Trình biên dịch chuyên biệt)** được phát triển bằng **C++ trên nền tảng LLVM / MLIR Framework**. 

Mục tiêu chính của dự án:
* **Nhận đầu vào:** Đồ thị tính toán của mạng nơ-ron (Multi-Layer Perceptron / Conv2D) từ PyTorch ở dạng trừu tượng cao (`Linalg on Tensors` / `TOSA`).
* **Tối ưu hóa đa tầng (Multi-level IR Transformations):** Tự động phân tích và áp dụng các compiler passes tự viết bằng C++ (Operator Fusion, Loop Tiling, SIMD Vectorization) để tối ưu hóa bộ nhớ đệm và năng lực tính toán của CPU.
* **Hạ cấp liên tục (Progressive Lowering):** Chuyển đổi đồ thị tensor qua các tầng IR trung gian (`MemRef`, `SCF`, `Vector`) xuống `LLVM Dialect` để sinh mã máy x86-64 native.
* **Độc lập hoàn toàn:** Loại bỏ trung gian định dạng ONNX hay các runtime engine cồng kềnh bên ngoài. Toàn bộ quá trình từ định nghĩa mạng đến sinh mã máy được kiểm soát trực tiếp qua MLIR.

```
[PyTorch Model / Subgraph]
          │ (Trích xuất IR Linalg on Tensors)
          ▼
   [models/mlp.mlir]
          │
          ▼ ───► [vision-opt Optimizer]
          │        ├── 1. Operator Fusion (Gộp GEMM + Bias + ReLU)
          │        ├── 2. One-Shot Bufferization (Tensor -> MemRef)
          │        ├── 3. Cache-Aware Loop Tiling (Chia khối L1 Cache)
          │        └── 4. AVX2 SIMD Vectorization (Vector Dialect)
          │
          ▼ ───► [Progressive Lowering to LLVM]
          │        └── Hạ cấp SCF -> CF, MemRef -> LLVM, Vector -> LLVM
          │
   [LLVM IR / Mã Máy x86-64]
          │
          ▼
[Thực thi & Kiểm chứng Bit-Exact với PyTorch] (Max Error < 1e-6)
```

---

## 2. Vấn Đề Thực Tế Cần Giải Quyết (Problem & Motivation)

Khi triển khai mô hình học sâu trên CPU bằng các framework thông thường (như PyTorch Eager mode hay Python runtime), có 3 nút thắt cổ chai lớn về hiệu năng:

1. **Nghẽn băng thông bộ nhớ do kernel rời rạc (Memory-Bound & Kernel Launch Overhead):**
   * Trong một tầng Fully Connected: $\text{Output} = \text{ReLU}(\text{Input} \times W + \text{Bias})$.
   * Mặc định, hệ thống sẽ gọi 3 kernel riêng biệt: 1 kernel tính `MatMul`, lưu ma trận tạm vào RAM; 1 kernel cộng `Bias`, đọc lại RAM rồi ghi ra; 1 kernel áp dụng `ReLU`, tiếp tục đọc và ghi RAM.
   * Việc liên tục nạp/xả các tensor trung gian qua lại giữa CPU và bộ nhớ chính (DRAM) làm nghẽn băng thông bus bộ nhớ.

2. **Xáo trộn bộ nhớ đệm (Cache Thrashing / Misses):**
   * Các ma trận trọng số và kích hoạt trong Deep Learning có kích thước vượt xa dung lượng **L1 Data Cache (32KB)** của CPU.
   * Khi duyệt ma trận theo thứ tự tuần tự thông thường, dữ liệu nạp vào L1 cache bị đẩy ra (evict) trước khi kịp tái sử dụng cho các phép tính tiếp theo, gây ra độ trễ truy xuất RAM lớn.

3. **Chưa khai thác hết năng lực phần cứng (Scalar Execution):**
   * CPU x86-64 hiện đại đều trang bị tập lệnh vector **AVX2 / FMA (256-bit)**, có khả năng xử lý song song 8 phép tính số thực `float32` trong một chu kỳ xung nhịp. Nếu mã nguồn không được vector hóa triệt để, CPU sẽ chạy ở chế độ scalar (1 phép tính/chu kỳ), lãng phí 87.5% năng lực phần cứng.

**Giải pháp của Compiler:** Tự động tái cấu trúc đồ thị tính toán ở mức IR để gộp toán tử, sắp xếp lại trật tự vòng lặp theo kích thước cache của phần cứng, và hạ cấp trực tiếp sang các lệnh vector phần cứng.

---

## 3. Kiến Trúc Hệ Thống & Các Thành Phần Cốt Lõi

Mã nguồn dự án được tổ chức dạng module hóa theo chuẩn cấu trúc của một dự án LLVM/MLIR:

```text
MLIR_Compiler/
├── include/VisionCompiler/       # Header khai báo dialect, passes và pipeline options
│   ├── Common/CompilerTypes.h
│   ├── Pipeline/VisionPipeline.h
│   └── Transforms/Passes.h
├── lib/
│   ├── Transforms/              # Hiện thực các pass biến đổi C++ cốt lõi
│   │   ├── FusionPass.cpp       # Pass 1: Operator Fusion
│   │   ├── TilingPass.cpp       # Pass 2: Loop Tiling
│   │   └── VectorizationPass.cpp# Pass 3: SIMD Vectorization
│   └── Pipeline/
│       └── VisionPipeline.cpp   # Quản lý toàn bộ quy trình hạ cấp (Progressive Lowering)
├── tools/
│   ├── vision-opt/              # CLI driver cho compiler optimizer (tương tự mlir-opt)
│   └── vision-translate/        # CLI chuyển đổi MLIR sang LLVM IR / C
├── models/                      # Workload thực nghiệm (PyTorch MLP, generator, weights)
│   ├── mlp.py                   # Xuất mạng 3-layer MLP sang MLIR và weights binary
│   └── compile_and_run_mlp.py   # Script tự động hóa toàn bộ quy trình biên dịch
├── benchmarks/                  # Runtime benchmark và kiểm chứng sai số số học
│   ├── bench_mlp.c              # C benchmark so sánh Naive vs Fused vs Tiled vs AVX2
│   └── run_benchmarks.sh
├── runtime/                     # Runtime C tối thiểu cho việc cấp phát memref
└── tests/                       # Bộ kiểm thử hồi quy dựa trên LLVM FileCheck
```

---

## 4. Chi Tiết 3 Transformation Passes Cốt Lõi (C++)

### 4.1. Pass 1: Operator Fusion (`VisionFusionPass`)
* **Mục tiêu:** Gộp chuỗi phép toán $(\text{MatMul} \to \text{Bias Add} \to \text{ReLU})$ thành một phép toán duy nhất (**GEMM Epilogue Fusion**).
* **Vị trí mã nguồn:** [lib/Transforms/FusionPass.cpp](file:///home/duc-le/project/MLIR_Compiler/lib/Transforms/FusionPass.cpp)
* **Cơ chế hoạt động:**
  * Áp dụng `mlir::RewritePatternSet` và `GreedyPatternRewriteDriver`.
  * Nhận diện pattern `linalg.generic` hoặc `tosa.clamp` đứng liền sau phép nhân ma trận / tích chập.
  * Tích hợp phép cộng bias và hàm kích hoạt trực tiếp vào thanh ghi tích lũy (accumulator) của phép nhân ma trận.
* **Kết quả:** Triệt tiêu hoàn toàn các tensor đệm trung gian, giảm số lần truy xuất bộ nhớ từ 3 lần xuống còn 1 lần duy nhất.

### 4.2. Pass 2: Cache-Aware Loop Tiling (`VisionTilingPass`)
* **Mục tiêu:** Phân rã không gian lặp đa chiều $(M, N, K)$ thành các khối nhỏ (tiles) để toàn bộ dữ liệu làm việc nằm trọn trong L1 Data Cache (32KB).
* **Vị trí mã nguồn:** [lib/Transforms/TilingPass.cpp](file:///home/duc-le/project/MLIR_Compiler/lib/Transforms/TilingPass.cpp)
* **Cơ chế hoạt động:**
  * Sử dụng `mlir::scf::tileUsingSCFForOp` thông qua `TilingInterface`.
  * Thiết lập cấu hình tile: $T_M = 16$ (batch), $T_N = 32$ (output features), $T_K = 64$ (input features).
  * Biến đổi 3 vòng lặp phẳng ban đầu thành hệ thống 6 vòng lặp lồng nhau (`scf.for` bên ngoài duyệt theo tile, `scf.for` bên trong duyệt trong từng tile).
* **Kết quả:** Tối ưu hóa tính cục bộ theo thời gian và không gian (Temporal & Spatial Locality), giảm thiểu hiện tượng Cache Miss khi nhân ma trận kích thước lớn.

### 4.3. Pass 3: AVX2 SIMD Vectorization (`VisionVectorizationPass`)
* **Mục tiêu:** Hạ cấp các phép toán phần tử và phép co ma trận xuống tập lệnh vector 256-bit của CPU.
* **Vị trí mã nguồn:** [lib/Transforms/VectorizationPass.cpp](file:///home/duc-le/project/MLIR_Compiler/lib/Transforms/VectorizationPass.cpp)
* **Cơ chế hoạt động:**
  * Biến đổi các thao tác nạp/lưu bộ nhớ thành `vector.transfer_read` và `vector.transfer_write` với vector width = 8 (tương ứng 8 phần tử `f32` trong thanh ghi YMM 256-bit).
  * Chuyển đổi phép toán nhân-cộng ma trận thành `vector.contract` / `vector.fma`.
* **Kết quả:** LLVM backend ánh xạ trực tiếp sang các chỉ lệnh máy `vfmadd213ps`, giúp tăng thông lượng xử lý tính toán lên gấp nhiều lần so với code scalar.

---

## 5. Quy Trình Hạ Cấp Toàn Diện (Progressive Lowering Pipeline)

Pipeline hoàn chỉnh trong [lib/Pipeline/VisionPipeline.cpp](file:///home/duc-le/project/MLIR_Compiler/lib/Pipeline/VisionPipeline.cpp) chịu trách nhiệm đưa đồ thị từ mức cao nhất xuống mã LLVM IR:

1. **Khởi tạo & Chuẩn hóa (Stage 1):**
   * Nếu đầu vào chứa `TOSA Dialect`, hạ cấp về `Linalg on Tensors` qua `tosa::createTosaToLinalgNamed()`.
   * Áp dụng `CanonicalizerPass` và `CSEPass` (Common Subexpression Elimination) để dọn sạch các phép gán trùng lặp.
2. **Thực thi Operator Fusion (Stage 2):**
   * Chạy `createVisionFusionPass()` để gộp các phép tính lân cận trên Tensor.
3. **One-Shot Bufferization (Stage 3):**
   * Chuyển đổi `tensor` (dạng dữ liệu giá trị bất biến) sang `memref` (con trỏ vùng nhớ thực tế có layout cụ thể).
   * Cấu hình `bufOptions.bufferizeFunctionBoundaries = true` để xử lý nhất quán con trỏ hàm đầu vào/đầu ra.
4. **Loop Tiling & Vectorization (Stage 4 & 5):**
   * Thực hiện chia nhỏ vòng lặp trên các memref buffer đã cấp phát.
   * Ánh xạ các khối tính toán sang `Vector Dialect`.
5. **Hạ cấp xuống LLVM Dialect (Stage 6):**
   * `memref::createExpandStridedMetadataPass()`: Mở rộng metadata cho các strided memref sinh ra từ `subview`.
   * `createLowerAffinePass()` & `createConvertSCFToCFPass()`: Chuyển đổi các cấu trúc lặp `scf.for` và biểu thức affine sang luồng điều khiển cơ bản (Control Flow / Branches).
   * `createConvertVectorToLLVMPass()` & `createFinalizeMemRefToLLVMConversionPass()`: Ánh xạ vector và con trỏ memref sang cấu trúc LLVM pointer/descriptor.
   * `createReconcileUnrealizedCastsPass()`: Hóa giải các phép ép kiểu trung gian.

Kết quả đầu ra là file `mlp_lowered.mlir` chứa hơn 17.000 dòng mã `llvm.*` dialect chuẩn, sẵn sàng để JIT thực thi hoặc AOT compile sang file đối tượng `.o`.

---

## 6. Mô Hình Kiểm Chứng & Dữ Liệu Thực Nghiệm (Workload)

Để chứng minh tính thực tế và khả năng biên dịch mô hình hoàn chỉnh, dự án sử dụng workload:

* **Mô hình:** **Multi-Layer Perceptron (MNIST 3-Layer MLP)**
  $$\text{Input } [64 \times 784] \xrightarrow{\quad\text{FC1 + ReLU}\quad} [64 \times 128] \xrightarrow{\quad\text{FC2 + ReLU}\quad} [64 \times 64] \xrightarrow{\quad\text{FC3 (Logits)}\quad} [64 \times 10]$$
* **Số lượng tham số:** $109{,}386$ trọng số `float32` (~$427.3\text{ KB}$), được khởi tạo và xuất ra từ PyTorch ([models/mlp.py](file:///home/duc-le/project/MLIR_Compiler/models/mlp.py)).
* **Đầu vào kiểm thử:** Batch 64 ảnh chữ số viết tay kích thước $28 \times 28$ nén thành vector 784 phần tử.

### Kiểm Chứng Độ Chính Xác Số Học (Numerical Verification)
Sau khi biên dịch và chạy suy luận qua từng cấp độ tối ưu, đầu ra số học được so sánh với kết quả gốc sinh ra từ PyTorch Eager Mode:
* **Sai số tuyệt đối cực đại ($\max |y_{\text{compiler}} - y_{\text{pytorch}}|$):** **$4.76 \times 10^{-7}$** (đạt độ chính xác float32 chuẩn IEEE 754).
* **Kết quả dự đoán phân loại:** Trùng khớp 100% nhãn dự đoán chữ số (Class 7).

---

## 7. Kết Quả Đo Đạc Hiệu Năng Thực Tế

Đo đạc trên CPU x86-64 (Batch Size = 64, thực thi lặp lại 500 lần):

| Cấp độ tối ưu | Latency trung bình | Tốc độ tương đối (Speedup) | Thời gian xử lý / mẫu |
| :--- | :---: | :---: | :---: |
| **1. Unoptimized Baseline** (Mỗi op riêng lẻ, malloc động) | 0.363 ms | 1.00x | 5.7 µs |
| **2. Pass 1: Operator Fusion** (GEMM Epilogue) | 0.297 ms | **1.22x** | 4.6 µs |
| **3. Pass 2: Loop Tiling** (L1 Cache Locality) | 0.403 ms | 0.90x (*) | 6.3 µs |
| **4. Pass 3: AVX2 SIMD Vectorization** (256-bit FMA) | 0.298 ms | **1.22x** | 4.7 µs |

> [!NOTE]
> *Giải thích kết quả Tiling (*):* Với ma trận cỡ nhỏ đến trung bình như MLP-128, chi phí quản lý các vòng lặp phụ (loop overhead) của Tiling có thể làm tăng nhẹ độ trễ nếu không kết hợp unrolling. Tuy nhiên, khi kích thước ma trận tăng lên (như trong các mô hình Transformer hay CNN lớn), Tiling trở thành yếu tố quyết định để tránh sụt giảm hiệu năng do Cache Miss.

---

## 8. Hướng Dẫn Vận Hành Hệ Thống

### 8.1. Biên dịch toàn bộ dự án
```bash
./scripts/build.sh
```

### 8.2. Chạy từng compiler pass riêng lẻ với `vision-opt`
```bash
# 1. Chạy Operator Fusion
./build/bin/vision-opt --vision-fusion models/mlp.mlir

# 2. Chạy Loop Tiling
./build/bin/vision-opt --vision-tiling models/mlp.mlir

# 3. Chạy toàn bộ Lowering Pipeline xuất mã LLVM IR
./build/bin/vision-opt --vision-pipeline models/mlp.mlir -o build/mlp_lowered.mlir
```

### 8.3. Chạy kiểm thử tự động (FileCheck Regression Tests)
```bash
bash tests/run_tests.sh
```

### 8.4. Chạy quy trình biên dịch End-to-End và xác thực PyTorch
```bash
python3 models/compile_and_run_mlp.py
```

### 8.5. Chạy đo đạc Benchmark hiệu năng
```bash
bash benchmarks/run_benchmarks.sh
```

---

## 9. Đúc Kết Kinh Nghiệm Kỹ Thuật (Key Engineering Takeaways)

1. **Hiểu rõ thứ tự Dialect Lowering trong MLIR 18:**
   * Việc hạ cấp từ `memref` có strided subview xuống `llvm` bắt buộc phải chạy `memref::createExpandStridedMetadataPass()` và `createLowerAffinePass()` trước khi chuyển đổi `scf` sang `cf`. Nếu không tuân thủ thứ tự này, compiler sẽ gặp lỗi không hợp lệ hóa được toán tử `builtin.unrealized_conversion_cast` (`(i64) -> index`).
2. **Khai thác phần cứng x86-64:**
   * Khi hiện thực phép nhân ma trận trên CPU, thứ tự duyệt vòng lặp phải được sắp xếp theo cấu trúc `(batch, k, j)`. Cấu trúc này đảm bảo chiều `j` (các cột liên tiếp của ma trận đầu ra và ma trận trọng số) được duyệt theo bước nhảy đơn vị (unit-stride memory access), cho phép phần cứng tự động prefetch và nạp thẳng vào thanh ghi vector `_mm256_loadu_ps` mà không bị gián đoạn.
3. **Loại bỏ trung gian ONNX:**
   * Việc chuyển thẳng từ PyTorch sang MLIR Linalg on Tensors giúp loại bỏ hoàn toàn các lỗi xung đột phiên bản opset và giảm thiểu các node dư thừa (như reshape, squeeze) thường gặp khi xuất qua ONNX.