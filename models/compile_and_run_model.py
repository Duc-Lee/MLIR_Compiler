import os
import subprocess
import sys
import torch

def main():
    print("===================================================================")
    print("  Bien Dich Mo Hinh Deep Learning Qua MLIR Vision Compiler Pipeline")
    print("===================================================================")

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    opt_bin = os.path.join(root_dir, "build", "bin", "vision-opt")
    model_mlir = os.path.join(root_dir, "models", "conv_bn_relu.mlir")
    resnet_mlir = os.path.join(root_dir, "models", "resnet_block.mlir")

    if not os.path.exists(opt_bin):
        print(f"[!] Chua tim thay binary compiler: {opt_bin}")
        print("    Vui long chay: ./scripts/build.sh")
        return

    print(f"\n[1] File IR mo hinh dau vao:")
    print(f"    - Subgraph Conv2D+Bias+ReLU: {model_mlir}")
    print(f"    - ResNet-18 Residual Block : {resnet_mlir}")

    # Buoc 1: Chay Pass 1 - Operator Fusion
    print("\n[2] Thuc thi PASS 1: Operator Fusion...")
    cmd_pass1 = [opt_bin, "--vision-fusion", model_mlir]
    res1 = subprocess.run(cmd_pass1, capture_output=True, text=True)
    if res1.returncode == 0:
        print("[+] PASS 1 (Fusion) thanh cong!")
    else:
        print(f"[-] PASS 1 that bai: {res1.stderr}")

    # Buoc 2: Chay Pass 2 - Loop Tiling
    print("\n[3] Thuc thi PASS 2: Loop Tiling (L1 Cache Locality)...")
    cmd_pass2 = [opt_bin, "--vision-tiling", os.path.join(root_dir, "tests", "test_tiling.mlir")]
    res2 = subprocess.run(cmd_pass2, capture_output=True, text=True)
    if res2.returncode == 0:
        print("[+] PASS 2 (Tiling) thanh cong! Sinh ra cac scf.for loop chia khoi.")
    else:
        print(f"[-] PASS 2 that bai: {res2.stderr}")

    # Buoc 3: Chay Pass 3 - Vectorization
    print("\n[4] Thuc thi PASS 3: Vectorization (AVX2 / SIMD)...")
    cmd_pass3 = [opt_bin, "--vision-vectorize", os.path.join(root_dir, "tests", "test_vectorize.mlir")]
    res3 = subprocess.run(cmd_pass3, capture_output=True, text=True)
    if res3.returncode == 0:
        print("[+] PASS 3 (Vectorization) thanh cong! Sinh ra vector.transfer va arith.")
    else:
        print(f"[-] PASS 3 that bai: {res3.stderr}")

    # Buoc 4: Chay End-to-End Pipeline
    print("\n[5] Thuc thi End-to-End Lowering Pipeline (--vision-pipeline)...")
    cmd_full = [opt_bin, "--vision-pipeline", model_mlir]
    res_full = subprocess.run(cmd_full, capture_output=True, text=True)
    if res_full.returncode == 0:
        print("[+] Toan bo pipeline ha cap hoan tat!")
        output_ir_path = os.path.join(root_dir, "build", "lowered_model.mlir")
        with open(output_ir_path, "w") as f:
            f.write(res_full.stdout)
        print(f"[+] Da luu LLVM IR sau ha cap vao: {output_ir_path}")
    else:
        print(f"[i] Chi tiet ha cap: {res_full.stderr.strip() or 'OK'}")

    # Buoc 5: Chay benchmark do toc do thuc te tren CPU
    print("\n[6] Kiem tra toc do thuc thi sau toi uu (AVX2 Kernel vs PyTorch Baseline)...")
    bench_bin = os.path.join(root_dir, "build", "bin", "bench_conv")
    if os.path.exists(bench_bin):
        subprocess.run([bench_bin])

if __name__ == "__main__":
    main()
