import torch
import torch.nn as nn
import torch_mlir
import sys

def check_dependencies():
    missing = []
    for pkg in ["torch", "torchvision", "torch_mlir"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[!] Thieu cac thu vien: {', '.join(missing)}")
        print("    Cai dat bang: pip install torch torchvision torch-mlir")
        return False
    return True

def export_resnet_block(output_path="models/resnet_block_exported.mlir"):
    class BasicBlock(nn.Module):
        def __init__(self, in_planes=64, planes=64):
            super().__init__()
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)

        def forward(self, x):
            identity = x
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += identity
            out = self.relu(out)
            return out

    model = BasicBlock(64, 64).eval()
    dummy_input = torch.randn(1, 64, 56, 56)

    # ha cap sang tosa dialect thong qua torch-mlir
    print(f"[*] Dang bien dich model PyTorch sang TOSA MLIR...")
    module = torch_mlir.compile(
        model,
        dummy_input,
        output_type=torch_mlir.OutputType.TOSA,
        use_tracing=True
    )

    with open(output_path, "w") as f:
        f.write(str(module))
    print(f"[+] Da xuat thanh cong ra: {output_path}")

if __name__ == "__main__":
    if check_dependencies():
        export_resnet_block()
    else:
        print("[i] Vui long dung file MLIR mau da tao san tai models/resnet_block.mlir")
