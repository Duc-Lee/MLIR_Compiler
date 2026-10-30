from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple


def render_ascii_chart(title: str, results: List[Tuple[str, float, float]]) -> None:
    """Renders an aligned ASCII horizontal bar chart."""
    if not results:
        return

    max_lat = max(lat for _, lat, _ in results)
    bar_width = 28
    print(f"{title}:")
    for name, lat, speedup in results:
        ratio = lat / max_lat if max_lat > 0 else 0
        filled = int(round(ratio * bar_width))
        bar = "█" * filled + "░" * (bar_width - filled)
        print(f"  {name:<14} | {bar} {lat:>8.3f} ms ({speedup:>5.2f}x)")
    print()


def parse_benchmark_output(text: str) -> List[Tuple[str, float, float]]:
    """Extracts benchmark kernel names, latency (ms), and speedup factor from tabular output."""
    results: List[Tuple[str, float, float]] = []
    pattern = re.compile(
        r"^\s*([a-zA-Z0-9_]+)\s+([0-9.]+)\s+ms\s+.*?\s+([0-9.]+)x",
        re.MULTILINE,
    )
    for match in pattern.finditer(text):
        name, lat_str, sp_str = match.groups()
        results.append((name, float(lat_str), float(sp_str)))
    return results


def run_benchmark(bin_path: Path, iters: Optional[int] = None) -> str:
    """Executes benchmark binary and returns stdout."""
    cmd = [str(bin_path)]
    if iters:
        cmd.extend(["--iters", str(iters)])
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return res.stdout


def main() -> None:
    parser = argparse.ArgumentParser(description="Render benchmark latency and speedup charts.")
    parser.add_argument(
        "--target",
        choices=["mlp", "conv", "all"],
        default="all",
        help="Benchmark target to profile (default: all).",
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="Path to an existing benchmark log file to parse.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=None,
        help="Iteration count for benchmark execution.",
    )
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent.parent
    bin_dir = root_dir / "build" / "bin"

    if args.file:
        if not args.file.exists():
            print(f"error: file not found: {args.file}", file=sys.stderr)
            sys.exit(1)
        content = args.file.read_text(encoding="utf-8")
        parsed = parse_benchmark_output(content)
        render_ascii_chart(f"Results from {args.file.name}", parsed)
        return

    targets = ["mlp", "conv"] if args.target == "all" else [args.target]

    for tgt in targets:
        bin_path = bin_dir / f"bench_{tgt}"
        if not bin_path.exists():
            print(f"error: benchmark binary not found: {bin_path}", file=sys.stderr)
            continue

        try:
            output = run_benchmark(bin_path, iters=args.iters)
            parsed = parse_benchmark_output(output)
            title = "MLP 3-Layer Latency" if tgt == "mlp" else "ResNet-18 Conv2D Latency"
            render_ascii_chart(title, parsed)
        except subprocess.CalledProcessError as err:
            print(f"error: {bin_path.name} failed: {err.stderr.strip()}", file=sys.stderr)


if __name__ == "__main__":
    main()
