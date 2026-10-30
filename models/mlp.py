from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Configurable Multi-Layer Perceptron (MLP).

    Args:
        in_features: Dimensionality of input vector (e.g. 784 for flattened 28x28).
        hidden_dims: Sizes of intermediate hidden layers.
        num_classes: Number of output logits (e.g. 10 for MNIST classes).
        bias: Whether to include bias terms in linear layers.
        seed: Random seed for deterministic parameter initialization.
    """

    def __init__(
        self,
        in_features: int = 784,
        hidden_dims: Sequence[int] = (128, 64),
        num_classes: int = 10,
        bias: bool = True,
        seed: Optional[int] = 42,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.hidden_dims = list(hidden_dims)
        self.num_classes = num_classes
        dims = [in_features] + self.hidden_dims + [num_classes]
        self.layers = nn.ModuleList([
            nn.Linear(dims[i], dims[i + 1], bias=bias)
            for i in range(len(dims) - 1)
        ])
        self.reset_parameters(seed=seed)

    @property
    def fc1(self) -> nn.Linear:
        """Alias for backward compatibility with 3-layer architecture."""
        return self.layers[0]

    @property
    def fc2(self) -> nn.Linear:
        """Alias for backward compatibility with 3-layer architecture."""
        return self.layers[1]

    @property
    def fc3(self) -> nn.Linear:
        """Alias for backward compatibility with 3-layer architecture."""
        return self.layers[2]

    def reset_parameters(self, seed: Optional[int] = 42) -> None:
        """Initializes weights deterministically (Kaiming for hidden, Xavier for classifier)."""
        if seed is not None:
            torch.manual_seed(seed)
        num_layers = len(self.layers)
        for idx, layer in enumerate(self.layers):
            if idx < num_layers - 1:
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.1)
            else:
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with ReLU activation on all hidden layers."""
        num_layers = len(self.layers)
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx < num_layers - 1:
                x = F.relu(x)
        return x


class MNISTMLP(MLP):
    """Standard 3-layer MLP for MNIST digit classification (784 -> 128 -> 64 -> 10)."""
    def __init__(
        self,
        in_features: int = 784,
        hidden_dims: Sequence[int] = (128, 64),
        num_classes: int = 10,
        **kwargs,
    ) -> None:
        super().__init__(
            in_features=in_features,
            hidden_dims=hidden_dims,
            num_classes=num_classes,
            **kwargs,
        )


class PyTorchToMLIRExporter:
    """Traces a PyTorch nn.Module using torch.fx and emits MLIR (Linalg on Tensors)."""

    def __init__(
        self,
        model: nn.Module,
        example_input: torch.Tensor,
        func_name: str = "mlp_forward",
    ) -> None:
        self.model = model
        self.example_input = example_input
        self.func_name = func_name
        self.traced = fx.symbolic_trace(model)

    @staticmethod
    def _format_tensor_type(*dims: int | str, dtype: str = "f32") -> str:
        return f"tensor<{'x'.join(map(str, dims))}x{dtype}>"

    @staticmethod
    def _emit_matmul(
        layer_idx: int, in_val: str, w_arg: str, in_t: str, w_t: str, out_t: str
    ) -> str:
        return (
            f"    // Layer {layer_idx} Linear\n"
            f"    %init_l{layer_idx} = tensor.empty() : {out_t}\n"
            f"    %fill_l{layer_idx} = linalg.fill ins(%c0 : f32) outs(%init_l{layer_idx} : {out_t}) -> {out_t}\n"
            f"    %mm{layer_idx} = linalg.matmul\n"
            f"      ins({in_val}, {w_arg} : {in_t}, {w_t})\n"
            f"      outs(%fill_l{layer_idx} : {out_t}) -> {out_t}"
        )

    @staticmethod
    def _emit_bias_add(
        layer_idx: int, in_val: str, b_arg: str, out_t: str, b_t: str
    ) -> str:
        return (
            f"    // Bias Add for Layer {layer_idx}\n"
            f"    %bias_l{layer_idx} = linalg.generic {{\n"
            f"      indexing_maps = [\n"
            f"        affine_map<(d0, d1) -> (d0, d1)>,\n"
            f"        affine_map<(d0, d1) -> (d0, d1)>,\n"
            f"        affine_map<(d0, d1) -> (d0, d1)>\n"
            f"      ],\n"
            f"      iterator_types = [\"parallel\", \"parallel\"]\n"
            f"    }} ins({in_val}, {b_arg} : {out_t}, {b_t})\n"
            f"      outs(%init_l{layer_idx} : {out_t}) {{\n"
            f"    ^bb0(%in_val: f32, %bias_val: f32, %out_val: f32):\n"
            f"      %sum = arith.addf %in_val, %bias_val : f32\n"
            f"      linalg.yield %sum : f32\n"
            f"    }} -> {out_t}"
        )

    @staticmethod
    def _emit_relu(
        node_name: str, in_val: str, init_val: str, act_t: str
    ) -> str:
        return (
            f"    // Activation: ReLU ({node_name})\n"
            f"    %relu_{node_name} = linalg.generic {{\n"
            f"      indexing_maps = [\n"
            f"        affine_map<(d0, d1) -> (d0, d1)>,\n"
            f"        affine_map<(d0, d1) -> (d0, d1)>\n"
            f"      ],\n"
            f"      iterator_types = [\"parallel\", \"parallel\"]\n"
            f"    }} ins({in_val} : {act_t})\n"
            f"      outs({init_val} : {act_t}) {{\n"
            f"    ^bb0(%in_val: f32, %out_val: f32):\n"
            f"      %act = arith.maximumf %in_val, %c0 : f32\n"
            f"      linalg.yield %act : f32\n"
            f"    }} -> {act_t}"
        )

    def emit_mlir(self) -> str:
        """Generates MLIR Linalg code representing the traced computation graph."""
        batch_size = self.example_input.shape[0]

        linear_layers: List[Tuple[int, str, nn.Linear]] = [
            (i + 1, node.target, self.model.get_submodule(node.target))
            for i, node in enumerate(
                n for n in self.traced.graph.nodes
                if n.op == "call_module" and isinstance(self.model.get_submodule(n.target), nn.Linear)
            )
        ]
        if not linear_layers:
            raise ValueError("No linear layers found in model graph to export.")

        in_type = self._format_tensor_type(*self.example_input.shape)
        func_params: List[str] = [f"      %input: {in_type}"]
        for idx, _, submod in linear_layers:
            func_params.append(
                f"      %w{idx}: {self._format_tensor_type(submod.in_features, submod.out_features)}"
            )
            if submod.bias is not None:
                func_params.append(
                    f"      %b{idx}: {self._format_tensor_type(1, submod.out_features)}"
                )

        out_features = linear_layers[-1][2].out_features
        out_type = self._format_tensor_type(batch_size, out_features)

        env: Dict[str, str] = {}
        ops: List[str] = ["    %c0 = arith.constant 0.000000e+00 : f32"]
        layer_idx = 0
        curr_features = self.example_input.shape[-1]
        final_val = "%input"

        for node in self.traced.graph.nodes:
            if node.op == "placeholder":
                env[node.name] = "%input"

            elif node.op == "call_module":
                submod = self.model.get_submodule(node.target)
                if isinstance(submod, nn.Linear):
                    layer_idx += 1
                    in_f, out_f = submod.in_features, submod.out_features
                    curr_features = out_f

                    in_val = env[node.args[0].name]
                    out_t = self._format_tensor_type(batch_size, out_f)
                    in_t = self._format_tensor_type(batch_size, in_f)
                    w_t = self._format_tensor_type(in_f, out_f)

                    ops.append(self._emit_matmul(layer_idx, in_val, f"%w{layer_idx}", in_t, w_t, out_t))
                    if submod.bias is not None:
                        b_t = self._format_tensor_type(1, out_f)
                        ops.append(self._emit_bias_add(layer_idx, f"%mm{layer_idx}", f"%b{layer_idx}", out_t, b_t))
                        env[node.name] = f"%bias_l{layer_idx}"
                    else:
                        env[node.name] = f"%mm{layer_idx}"

                elif isinstance(submod, nn.ReLU):
                    in_val = env[node.args[0].name]
                    act_t = self._format_tensor_type(batch_size, curr_features)
                    ops.append(self._emit_relu(node.name, in_val, f"%init_l{layer_idx}", act_t))
                    env[node.name] = f"%relu_{node.name}"

            elif node.op == "call_function" and node.target in (F.relu, torch.relu):
                in_val = env[node.args[0].name]
                act_t = self._format_tensor_type(batch_size, curr_features)
                ops.append(self._emit_relu(node.name, in_val, f"%init_l{layer_idx}", act_t))
                env[node.name] = f"%relu_{node.name}"

            elif node.op == "output":
                final_val = env[node.args[0].name]

        params_decl = ",\n".join(func_params)
        body_content = "\n\n".join(ops)

        return (
            f"module {{\n"
            f"  func.func @{self.func_name}(\n"
            f"{params_decl}) -> {out_type} {{\n"
            f"{body_content}\n\n"
            f"    return {final_val} : {out_type}\n"
            f"  }}\n"
            f"}}\n"
        )


def export_weights(model: nn.Module, output_path: str | Path) -> int:
    """Exports linear weights and biases in binary format expected by C runtime.

    Weight matrices are transposed from PyTorch [out_features, in_features]
    to row-major [in_features, out_features] for standard MatMul alignment.
    """
    path = Path(output_path)
    total_bytes = 0
    with path.open("wb") as f:
        for module in model.modules():
            if isinstance(module, nn.Linear):
                w_bytes = module.weight.detach().t().contiguous().numpy().tobytes()
                f.write(w_bytes)
                total_bytes += len(w_bytes)

                if module.bias is not None:
                    b_bytes = module.bias.detach().contiguous().numpy().tobytes()
                    f.write(b_bytes)
                    total_bytes += len(b_bytes)
    return total_bytes


def export_test_data(
    model: nn.Module,
    input_path: str | Path,
    output_path: str | Path,
    seed: int = 123,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates synthetic input data and computes PyTorch reference ground truth."""
    in_features = getattr(model, "in_features", 784)
    np.random.seed(seed)
    dummy_input_np = np.random.randn(1, in_features).astype(np.float32) * 0.5
    dummy_input = torch.from_numpy(dummy_input_np)
    model.eval()
    with torch.no_grad():
        reference_output = model(dummy_input).numpy()
    Path(input_path).write_bytes(dummy_input_np.tobytes())
    Path(output_path).write_bytes(reference_output.tobytes())
    return dummy_input_np, reference_output


def generate_mlp_mlir(
    model: nn.Module,
    output_path: str | Path,
    example_input: Optional[torch.Tensor] = None,
) -> None:
    """Traces the PyTorch model and writes the MLIR Linalg code to file."""
    if example_input is None:
        in_features = getattr(model, "in_features", 784)
        example_input = torch.randn(1, in_features)

    exporter = PyTorchToMLIRExporter(model, example_input)
    mlir_code = exporter.emit_mlir()
    Path(output_path).write_text(mlir_code, encoding="utf-8")


def export_weights_and_ground_truth(output_dir: Optional[str | Path] = None) -> None:
    """Primary pipeline export: serializes weights, test data, and MLIR file."""
    base_dir = Path(output_dir) if output_dir else Path(__file__).resolve().parent
    model = MNISTMLP().eval()

    # 1. Weights
    weights_path = base_dir / "mlp_weights.bin"
    n_bytes = export_weights(model, weights_path)
    total_params = sum(p.numel() for p in model.parameters())

    # 2. Input and Ground Truth Output
    input_path = base_dir / "test_input.bin"
    output_path = base_dir / "reference_output.bin"
    _, ref_out = export_test_data(model, input_path, output_path)
    pred_class = int(np.argmax(ref_out[0]))

    # 3. MLIR Code
    mlir_path = base_dir / "mlp.mlir"
    generate_mlp_mlir(model, mlir_path)

    print(f"Exported MLP model ({model.in_features} -> {' -> '.join(map(str, model.hidden_dims))} -> {model.num_classes}, {total_params:,} params):")
    print(f"  weights      : {weights_path} ({n_bytes / 1024:.1f} KB)")
    print(f"  ground truth : {output_path} (class={pred_class})")
    print(f"  mlir dialect : {mlir_path}")

def main() -> None:
    parser = argparse.ArgumentParser(description="PyTorch MLP model definition and MLIR exporter.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory to save generated artifacts.",
    )
    parser.add_argument(
        "--in-features",
        type=int,
        default=784,
        help="Input feature dimension.",
    )
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[128, 64],
        help="Hidden layer dimensions.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=10,
        help="Number of output classification classes.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for weight initialization.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = MLP(
        in_features=args.in_features,
        hidden_dims=args.hidden_dims,
        num_classes=args.num_classes,
        seed=args.seed,
    ).eval()

    weights_path = args.output_dir / "mlp_weights.bin"
    n_bytes = export_weights(model, weights_path)
    total_params = sum(p.numel() for p in model.parameters())

    input_path = args.output_dir / "test_input.bin"
    output_path = args.output_dir / "reference_output.bin"
    _, ref_out = export_test_data(model, input_path, output_path)
    pred_class = int(np.argmax(ref_out[0]))

    mlir_path = args.output_dir / "mlp.mlir"
    generate_mlp_mlir(model, mlir_path)

    print(f"Exported MLP model to {args.output_dir} ({total_params:,} params, {n_bytes / 1024:.1f} KB, class={pred_class})")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        export_weights_and_ground_truth()
