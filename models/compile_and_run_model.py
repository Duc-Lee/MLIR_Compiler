#!/usr/bin/env python3
"""End-to-end compilation and execution driver for Conv2D / Vision models."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str) -> subprocess.CompletedProcess[str]:
    """Executes a command and returns the completed process."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as err:
        print(f"error: {description} failed (exit code {err.returncode})", file=sys.stderr)
        if err.stderr:
            print(err.stderr.strip(), file=sys.stderr)
        sys.exit(err.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile and run Conv2D vision models through MLIR pipeline.")
    parser.add_argument("--skip-bench", action="store_true", help="Skip running C benchmark binary.")
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent.parent
    opt_bin = root_dir / "build" / "bin" / "vision-opt"
    bench_bin = root_dir / "build" / "bin" / "bench_conv"
    model_mlir = root_dir / "models" / "conv_bn_relu.mlir"
    tiling_mlir = root_dir / "tests" / "test_tiling.mlir"
    vectorize_mlir = root_dir / "tests" / "test_vectorize.mlir"
    build_dir = root_dir / "build"
    lowered_mlir = build_dir / "lowered_model.mlir"

    if not opt_bin.exists():
        print(f"error: compiler binary not found: {opt_bin}", file=sys.stderr)
        print("hint: run ./scripts/build.sh to build the optimizer", file=sys.stderr)
        sys.exit(1)

    print(f"Compiling vision passes on {model_mlir.name}:")
    run_command([str(opt_bin), "--vision-fusion", str(model_mlir)], "vision-fusion")
    print("  pass: vision-fusion (ok)")

    run_command([str(opt_bin), "--vision-tiling", str(tiling_mlir)], "vision-tiling")
    print("  pass: vision-tiling (ok)")

    run_command([str(opt_bin), "--vision-vectorize", str(vectorize_mlir)], "vision-vectorize")
    print("  pass: vision-vectorize (ok)")

    res_full = run_command([str(opt_bin), "--vision-pipeline", str(model_mlir)], "vision-pipeline")
    build_dir.mkdir(parents=True, exist_ok=True)
    lowered_mlir.write_text(res_full.stdout, encoding="utf-8")
    num_lines = len(res_full.stdout.splitlines())
    print(f"  pass: vision-pipeline -> {lowered_mlir.name} ({num_lines:,} lines)")

    if not args.skip_bench:
        print()
        if bench_bin.exists():
            subprocess.run([str(bench_bin)], check=True)
        else:
            print(f"warning: benchmark binary not found: {bench_bin}", file=sys.stderr)


if __name__ == "__main__":
    main()
