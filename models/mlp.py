import os
import struct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx as fx

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



class PyTorchToMLIRExporter:
    """
    Compiler Frontend: Trace do thi tinh toan cua PyTorch nn.Module (su dung torch.fx)
    va tu dong sinh ma MLIR (Linalg on Tensors) tuong ung.
    """
    def __init__(self, model: nn.Module, example_input: torch.Tensor):
        self.model = model
        self.example_input = example_input
        self.traced = fx.symbolic_trace(model)

    def emit_mlir(self) -> str:
        lines = []
        lines.append("// =========================================================================")
        lines.append("//  Auto-generated MLIR (Linalg Dialect) from PyTorch via torch.fx graph tracer")
        lines.append("// =========================================================================")
        lines.append("")
        lines.append("module {")

        # Trich xuat danh sach cac lop Linear va tham so tu do thi
        layers = []
        linear_count = 0
        for node in self.traced.graph.nodes:
            if node.op == "call_module":
                submod = getattr(self.model, node.target)
                if isinstance(submod, nn.Linear):
                    linear_count += 1
                    layers.append((linear_count, node.target, submod))

        # Xay dung chu ky ham @mlp_forward
        in_shape = "x".join(map(str, self.example_input.shape)) + "xf32"
        args = [f"      %input: tensor<{in_shape}>"]
        for idx, name, submod in layers:
            in_f = submod.in_features
            out_f = submod.out_features
            args.append(f"      %w{idx}: tensor<{in_f}x{out_f}xf32>")
            if submod.bias is not None:
                args.append(f"      %b{idx}: tensor<1x{out_f}xf32>")

        out_shape = f"1x{layers[-1][2].out_features}xf32"
        func_sig = "  func.func @mlp_forward(\n" + ",\n".join(args) + f") -> tensor<{out_shape}> {{\n"
        lines.append(func_sig)
        lines.append("    %c0 = arith.constant 0.000000e+00 : f32\n")

        env = {}
        layer_idx = 0
        last_out_feat = self.example_input.shape[-1]

        for node in self.traced.graph.nodes:
            if node.op == "placeholder":
                env[node.name] = "%input"
            elif node.op == "call_module":
                submod = getattr(self.model, node.target)
                if isinstance(submod, nn.Linear):
                    layer_idx += 1
                    in_feat = submod.in_features
                    out_feat = submod.out_features
                    last_out_feat = out_feat
                    in_val = env[node.args[0].name]
                    w_arg = f"%w{layer_idx}"
                    b_arg = f"%b{layer_idx}"

                    init_t = f"%init_l{layer_idx}"
                    fill_t = f"%fill_l{layer_idx}"
                    mm_t = f"%mm{layer_idx}"

                    lines.append(f"    // Layer {layer_idx}: {node.target} (Linear {in_feat} -> {out_feat})")
                    lines.append(f"    {init_t} = tensor.empty() : tensor<1x{out_feat}xf32>")
                    lines.append(f"    {fill_t} = linalg.fill ins(%c0 : f32) outs({init_t} : tensor<1x{out_feat}xf32>) -> tensor<1x{out_feat}xf32>")
                    lines.append(f"    {mm_t} = linalg.matmul")
                    lines.append(f"      ins({in_val}, {w_arg} : tensor<1x{in_feat}xf32>, tensor<{in_feat}x{out_feat}xf32>)")
                    lines.append(f"      outs({fill_t} : tensor<1x{out_feat}xf32>) -> tensor<1x{out_feat}xf32>")

                    if submod.bias is not None:
                        bias_t = f"%bias_l{layer_idx}"
                        lines.append(f"    // Bias Add for Layer {layer_idx}")
                        lines.append(f"    {bias_t} = linalg.generic {{")
                        lines.append("      indexing_maps = [")
                        lines.append("        affine_map<(d0, d1) -> (d0, d1)>,")
                        lines.append("        affine_map<(d0, d1) -> (d0, d1)>,")
                        lines.append("        affine_map<(d0, d1) -> (d0, d1)>")
                        lines.append("      ],")
                        lines.append("      iterator_types = [\"parallel\", \"parallel\"]")
                        lines.append(f"    }} ins({mm_t}, {b_arg} : tensor<1x{out_feat}xf32>, tensor<1x{out_feat}xf32>)")
                        lines.append(f"      outs({init_t} : tensor<1x{out_feat}xf32>) {{")
                        lines.append("    ^bb0(%in_val: f32, %bias_val: f32, %out_val: f32):")
                        lines.append("      %sum = arith.addf %in_val, %bias_val : f32")
                        lines.append("      linalg.yield %sum : f32")
                        lines.append(f"    }} -> tensor<1x{out_feat}xf32>\n")
                        env[node.name] = bias_t
                    else:
                        env[node.name] = mm_t

            elif node.op == "call_function":
                if node.target in (F.relu, torch.relu):
                    in_val = env[node.args[0].name]
                    relu_t = f"%relu_{node.name}"
                    lines.append(f"    // Activation: ReLU ({node.name})")
                    lines.append(f"    {relu_t} = linalg.generic {{")
                    lines.append("      indexing_maps = [")
                    lines.append("        affine_map<(d0, d1) -> (d0, d1)>,")
                    lines.append("        affine_map<(d0, d1) -> (d0, d1)>")
                    lines.append("      ],")
                    lines.append("      iterator_types = [\"parallel\", \"parallel\"]")
                    lines.append(f"    }} ins({in_val} : tensor<1x{last_out_feat}xf32>)")
                    lines.append(f"      outs(%init_l{layer_idx} : tensor<1x{last_out_feat}xf32>) {{")
                    lines.append("    ^bb0(%in_val: f32, %out_val: f32):")
                    lines.append("      %act = arith.maximumf %in_val, %c0 : f32")
                    lines.append("      linalg.yield %act : f32")
                    lines.append(f"    }} -> tensor<1x{last_out_feat}xf32>\n")
                    env[node.name] = relu_t

            elif node.op == "output":
                final_val = env[node.args[0].name]
                lines.append(f"    return {final_val} : tensor<{out_shape}>")

        lines.append("  }")
        lines.append("}\n")
        return "\n".join(lines)

def generate_mlp_mlir(model, output_path, example_input=None):
    """
    Su dung PyTorchToMLIRExporter de trace do thi PyTorch va tu dong sinh file MLIR.
    """
    if example_input is None:
        example_input = torch.randn(1, 784)
    exporter = PyTorchToMLIRExporter(model, example_input)
    mlir_code = exporter.emit_mlir()
    with open(output_path, "w") as f:
        f.write(mlir_code)
    num_nodes = len(exporter.traced.graph.nodes)
    print(f"[+] Da trace do thi PyTorch ({num_nodes} nodes) va tu dong sinh file MLIR: {output_path}")


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
