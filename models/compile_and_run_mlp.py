"""
Bien dich va xac minh toan dien mo hinh Neural Network (MNIST 3-Layer MLP)
qua MLIR Compiler Pipeline.
"""

import os
import subprocess
import sys
import numpy as np

def main():
    print("===================================================================")
    print("  Bien Dich Mo Hinh Neural Network MLP Qua MLIR Compiler Pipeline")
    print("  Kien truc: 784 -> 128 (ReLU) -> 64 (ReLU) -> 10 (Logits)")
    print("===================================================================")

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    opt_bin = os.path.join(root_dir, "build", "bin", "vision-opt")
    mlp_py = os.path.join(root_dir, "models", "mlp.py")
    mlp_mlir = os.path.join(root_dir, "models", "mlp.mlir")
    bench_bin = os.path.join(root_dir, "build", "bin", "bench_mlp")

    # Buoc 1: Sinh weights va file MLIR tu PyTorch
    print("\n[1] Khoi tao PyTorch Model va sinh Ground Truth Reference...")
    subprocess.run([sys.executable, mlp_py], check=True)

    if not os.path.exists(opt_bin):
        print(f"[!] Chua tim thay binary compiler: {opt_bin}")
        print("    Vui long chay: ./scripts/build.sh")
        return

    # Buoc 2: Chay Pass 1 - Operator Fusion
    print("\n[2] Thuc thi PASS 1: Operator Fusion (GEMM Epilogue Fusion)...")
    cmd_pass1 = [opt_bin, "--vision-fusion", mlp_mlir]
    res1 = subprocess.run(cmd_pass1, capture_output=True, text=True)
    if res1.returncode == 0:
        print("[+] PASS 1 (Fusion) thanh cong! Fused MatMul + Bias + ReLU.")
    else:
        print(f"[-] PASS 1 that bai: {res1.stderr}")

    # Buoc 3: Chay Pass 2 - Loop Tiling
    print("\n[3] Thuc thi PASS 2: Loop Tiling (L1 Cache Locality M, N, K)...")
    cmd_pass2 = [opt_bin, "--vision-tiling", mlp_mlir]
    res2 = subprocess.run(cmd_pass2, capture_output=True, text=True)
    if res2.returncode == 0:
        print("[+] PASS 2 (Tiling) thanh cong! Sinh ra cac scf.for blocks chia nho ma tran.")
    else:
        print(f"[-] PASS 2 that bai: {res2.stderr}")

    # Buoc 4: Chay End-to-End Lowering Pipeline
    print("\n[4] Thuc thi End-to-End Pipeline (--vision-pipeline -> LLVM Dialect)...")
    cmd_full = [opt_bin, "--vision-pipeline", mlp_mlir]
    res_full = subprocess.run(cmd_full, capture_output=True, text=True)
    if res_full.returncode == 0:
        print("[+] Toan bo 3-Layer Neural Network da duoc ha cap thanh cong xuong LLVM Dialect!")
        output_ir_path = os.path.join(root_dir, "build", "mlp_lowered.mlir")
        with open(output_ir_path, "w") as f:
            f.write(res_full.stdout)
        num_lines = len(res_full.stdout.splitlines())
        print(f"[+] Da xuat LLVM IR hoan chinh ({num_lines:,} dong) vao: {output_ir_path}")
    else:
        print(f"[-] Ha cap pipeline that bai: {res_full.stderr.strip()}")

    # Buoc 5: Chay benchmark do toc do va verify so hoc
    print("\n[5] Chay Benchmark kiem tra do chinh xac so hoc va do latency...")
    if os.path.exists(bench_bin):
        subprocess.run([bench_bin])
    else:
        print(f"[!] Chua tim thay benchmark binary: {bench_bin}")

if __name__ == "__main__":
    main()
