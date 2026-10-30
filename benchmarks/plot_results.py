import sys

def draw_chart(stages, latencies, speedups):
    max_lat = max(latencies)
    print("\n" + "=" * 65)
    print("           BIEU DO LATENCY QUA TUNG PASS TOI UU (ms)")
    print("=" * 65)
    for stage, lat, sp in zip(stages, latencies, speedups):
        bar_len = int((lat / max_lat) * 35)
        bar = "█" * bar_len
        print(f"{stage:<28} | {bar:<35} {lat:>6.2f} ms ({sp:>4.1f}x)")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    stages = [
        "1. Unoptimized (3 Pass)",
        "2. Pass 1: Operator Fusion",
        "3. Pass 2: Fused + Tiling",
        "4. Pass 3: Tiled + AVX2 SIMD"
    ]
    latencies = [100.05, 99.68, 99.38, 12.39]
    speedups = [1.0, 1.0, 1.0, 8.1]
    draw_chart(stages, latencies, speedups)
