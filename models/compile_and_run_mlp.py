#!/usr/bin/env python3
"""End-to-end compilation and execution driver for the PyTorch MLP model."""

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
    parser = argparse.ArgumentParser(description="Compile and run MLP model through MLIR pipeline.")
    parser.add_argument("--skip-bench", action="store_true", help="Skip running C benchmark binary.")
    parser.add_argument("--skip-export", action="store_true", help="Skip PyTorch model and weight export.")
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent.parent
    opt_bin = root_dir / "build" / "bin" / "vision-opt"
    bench_bin = root_dir / "build" / "bin" / "bench_mlp"
    mlp_py = root_dir / "models" / "mlp.py"
    mlp_mlir = root_dir / "models" / "mlp.mlir"
    build_dir = root_dir / "build"
    lowered_mlir = build_dir / "mlp_lowered.mlir"

    if not args.skip_export:
        subprocess.run([sys.executable, str(mlp_py)], check=True)

    if not opt_bin.exists():
        print(f"error: compiler binary not found: {opt_bin}", file=sys.stderr)
        print("hint: run ./scripts/build.sh to build the optimizer", file=sys.stderr)
        sys.exit(1)

    print(f"Compiling {mlp_mlir.name} through vision-opt:")
    run_command([str(opt_bin), "--vision-fusion", str(mlp_mlir)], "vision-fusion")
    print("  pass: vision-fusion (ok)")

    run_command([str(opt_bin), "--vision-tiling", str(mlp_mlir)], "vision-tiling")
    print("  pass: vision-tiling (ok)")

    res_full = run_command([str(opt_bin), "--vision-pipeline", str(mlp_mlir)], "vision-pipeline")
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
