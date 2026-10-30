import os
import struct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Mo hinh Multi-Layer Perceptron (MLP) 3 lop phan loai chu so (MNIST architecture).
- Input:  [1, 784] (anh 28x28 da trai phang)
- Layer 1: Linear(784, 128) + ReLU
- Layer 2: Linear(128, 64)  + ReLU
- Layer 3: Linear(64, 10)   (Logits 10 lop chu so 0-9)

Script nay:
1. Khoi tao mo hinh PyTorch chuan voi seed co dinh (deterministic)
2. Thuc hien forward pass mau de lay Ground Truth (Dap an chuan)
3. Xuat weights ra file binary gon nhe (chi ~437 KB)
4. Sinh file MLIR bieu dien toan ven 100% kien truc mang (models/mlp.mlir)
"""


class MNISTMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128, bias=True)
        self.fc2 = nn.Linear(128, 64, bias=True)
        self.fc3 = nn.Linear(64, 10, bias=True)

        self._init_weights()

    def _init_weights(self):
        # Khoi tao Kaiming Normal chuan cho ReLU
        torch.manual_seed(42)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.constant_(self.fc1.bias, 0.1)
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.constant_(self.fc2.bias, 0.1)
        nn.init.xavier_normal_(self.fc3.weight)
        nn.init.constant_(self.fc3.bias, 0.0)

    def forward(self, x):
        # Layer 1: GEMM + Bias + ReLU
        x = F.relu(self.fc1(x))
        # Layer 2: GEMM + Bias + ReLU
        x = F.relu(self.fc2(x))
        # Layer 3: GEMM + Bias (Logits)
        x = self.fc3(x)
        return x

def generate_mlp_mlir(model, output_path):
    """
    Sinh ma MLIR chuan (Linalg on tensors) bieu dien toan ven ca 3 lop cua mang MLP.
    Moi lop gom: linalg.matmul (GEMM) + bias add + relu clamp.
    """
    mlir_template = """// =========================================================================
//  Multi-Layer Perceptron (MLP) MNIST Digit Classifier (3-Layer Neural Net)
//  Input:  [1, 784]
//  Layer 1: Linear(784 -> 128) + ReLU
//  Layer 2: Linear(128 -> 64)  + ReLU
//  Layer 3: Linear(64 -> 10)   (Logits)
// =========================================================================

module {
  func.func @mlp_forward(
      %input: tensor<1x784xf32>,
      %w1: tensor<784x128xf32>,
      %b1: tensor<1x128xf32>,
      %w2: tensor<128x64xf32>,
      %b2: tensor<1x64xf32>,
      %w3: tensor<64x10xf32>,
      %b3: tensor<1x10xf32>) -> tensor<1x10xf32> {

    %c0 = arith.constant 0.0 : f32

    // -------------------------------------------------------------
    // LAYER 1: MatMul (1x784 * 784x128 -> 1x128) + Bias + ReLU
    // -------------------------------------------------------------
    %init_l1 = tensor.empty() : tensor<1x128xf32>
    %fill_l1 = linalg.fill ins(%c0 : f32) outs(%init_l1 : tensor<1x128xf32>) -> tensor<1x128xf32>
    %mm1 = linalg.matmul
      ins(%input, %w1 : tensor<1x784xf32>, tensor<784x128xf32>)
      outs(%fill_l1 : tensor<1x128xf32>) -> tensor<1x128xf32>

    // Bias Add & ReLU
    %l1_out = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%mm1, %b1 : tensor<1x128xf32>, tensor<1x128xf32>)
      outs(%init_l1 : tensor<1x128xf32>) {
    ^bb0(%in_val: f32, %bias_val: f32, %out_val: f32):
      %sum = arith.addf %in_val, %bias_val : f32
      %relu = arith.maximumf %sum, %c0 : f32
      linalg.yield %relu : f32
    } -> tensor<1x128xf32>

    // -------------------------------------------------------------
    // LAYER 2: MatMul (1x128 * 128x64 -> 1x64) + Bias + ReLU
    // -------------------------------------------------------------
    %init_l2 = tensor.empty() : tensor<1x64xf32>
    %fill_l2 = linalg.fill ins(%c0 : f32) outs(%init_l2 : tensor<1x64xf32>) -> tensor<1x64xf32>
    %mm2 = linalg.matmul
      ins(%l1_out, %w2 : tensor<1x128xf32>, tensor<128x64xf32>)
      outs(%fill_l2 : tensor<1x64xf32>) -> tensor<1x64xf32>

    // Bias Add & ReLU
    %l2_out = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%mm2, %b2 : tensor<1x64xf32>, tensor<1x64xf32>)
      outs(%init_l2 : tensor<1x64xf32>) {
    ^bb0(%in_val: f32, %bias_val: f32, %out_val: f32):
      %sum = arith.addf %in_val, %bias_val : f32
      %relu = arith.maximumf %sum, %c0 : f32
      linalg.yield %relu : f32
    } -> tensor<1x64xf32>

    // -------------------------------------------------------------
    // LAYER 3: MatMul (1x64 * 64x10 -> 1x10) + Bias (Logits)
    // -------------------------------------------------------------
    %init_l3 = tensor.empty() : tensor<1x10xf32>
    %fill_l3 = linalg.fill ins(%c0 : f32) outs(%init_l3 : tensor<1x10xf32>) -> tensor<1x10xf32>
    %mm3 = linalg.matmul
      ins(%l2_out, %w3 : tensor<1x64xf32>, tensor<64x10xf32>)
      outs(%fill_l3 : tensor<1x10xf32>) -> tensor<1x10xf32>

    // Bias Add (Linear Output Logits)
    %l3_out = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%mm3, %b3 : tensor<1x10xf32>, tensor<1x10xf32>)
      outs(%init_l3 : tensor<1x10xf32>) {
    ^bb0(%in_val: f32, %bias_val: f32, %out_val: f32):
      %sum = arith.addf %in_val, %bias_val : f32
      linalg.yield %sum : f32
    } -> tensor<1x10xf32>

    return %l3_out : tensor<1x10xf32>
  }
}
"""
    with open(output_path, "w") as f:
        f.write(mlir_template.strip() + "\n")
    print(f"[+] Da tao file bieu dien MLIR: {output_path}")

def export_weights_and_ground_truth():
    models_dir = os.path.dirname(os.path.abspath(__file__))
    model = MNISTMLP().eval()

    # Tao dummy input mau (vector 784 dac trung)
    np.random.seed(123)
    dummy_input_np = np.random.randn(1, 784).astype(np.float32) * 0.5
    dummy_input = torch.from_numpy(dummy_input_np)

    with torch.no_grad():
        output = model(dummy_input)
        probs = F.softmax(output, dim=-1).numpy()[0]
        pred_class = int(np.argmax(probs))

    print("=================================================================")
    print("  Khoi Tao Mo Hinh Neural Network MLP (MNIST Classifier)")
    print("=================================================================")
    print(f"[*] Kien truc: 784 -> 128 (ReLU) -> 64 (ReLU) -> 10 (Logits)")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[*] Tong so tham so: {total_params:,} (~{total_params * 4 / 1024:.1f} KB)")
    print(f"[*] PyTorch Ground Truth Reference:")
    print(f"    - Du doan lop (Predicted Class): {pred_class} (Confidence: {probs[pred_class]*100:.2f}%)")
    print(f"    - Raw Logits: {np.round(output.numpy()[0], 4).tolist()}")

    # 1. Luu weights ra file binary nho gon
    weights_bin = os.path.join(models_dir, "mlp_weights.bin")
    with open(weights_bin, "wb") as f:
        # Luu theo thu tu: w1, b1, w2, b2, w3, b3
        # Chu y: PyTorch Linear weight co shape [out, in], can transpose sang [in, out] cho MatMul
        f.write(model.fc1.weight.t().detach().numpy().tobytes())
        f.write(model.fc1.bias.detach().numpy().tobytes())
        f.write(model.fc2.weight.t().detach().numpy().tobytes())
        f.write(model.fc2.bias.detach().numpy().tobytes())
        f.write(model.fc3.weight.t().detach().numpy().tobytes())
        f.write(model.fc3.bias.detach().numpy().tobytes())
    print(f"[+] Da luu binary weights: {weights_bin} ({os.path.getsize(weights_bin)/1024:.1f} KB)")

    # 2. Luu test input va reference output ra numpy / binary de runtime kiem chung
    input_bin = os.path.join(models_dir, "test_input.bin")
    dummy_input_np.tofile(input_bin)
    ref_output_bin = os.path.join(models_dir, "reference_output.bin")
    output.numpy().tofile(ref_output_bin)
    print(f"[+] Da luu test input & reference output de kiem chung do chinh xac")

    # 3. Sinh file MLIR bieu dien toan bo mo hinh
    mlir_path = os.path.join(models_dir, "mlp.mlir")
    generate_mlp_mlir(model, mlir_path)

if __name__ == "__main__":
    export_weights_and_ground_truth()
