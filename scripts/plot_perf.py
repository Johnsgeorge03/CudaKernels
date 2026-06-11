#!/usr/bin/env python3
"""Run the kernel test binaries and plot their benchmark results.

Usage:
    python scripts/plot_perf.py [--build-dir build] [--out perf.png]
                                [--peak-bw <GB/s>] [--peak-flops <GFLOP/s>]

Finds the test executables in the build directory (build/Release/ on
Windows multi-config generators, build/ on Linux), runs them, and parses
the perf lines they print, e.g.

    matmul naive                  13.9502 ms/iter     153.9 GFLOP/s
    dotprod                        1.0275 ms/iter     130.6 GB/s

Memory-bound kernels (GB/s) and compute-bound kernels (GFLOP/s) are drawn
on separate charts. Pass --peak-bw / --peak-flops (from your GPU's spec
sheet) to draw the hardware ceiling on each chart.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to file; no display needed
import matplotlib.pyplot as plt

PERF_RE = re.compile(
    r"^(?P<name>.+?)\s{2,}(?P<ms>[\d.]+)\s+ms/iter"
    r"(?:\s+(?P<gbps>[\d.]+)\s+GB/s)?"
    r"(?:\s+(?P<gflops>[\d.]+)\s+GFLOP/s)?\s*$"
)


def find_test_binaries(build_dir: Path) -> list[Path]:
    """Locate test executables for both single- and multi-config generators."""
    found = sorted(build_dir.glob("Release/test_*.exe"))
    found += sorted(
        p for p in build_dir.glob("test_*") if p.is_file() and p.suffix == ""
    )
    return found


def run_and_parse(exe: Path) -> list[dict]:
    """Run one test binary and return its parsed perf rows."""
    proc = subprocess.run([str(exe)], capture_output=True, text=True)
    print(proc.stdout, end="")

    if proc.returncode != 0 or "FAILED" in proc.stdout:
        sys.exit(f"error: {exe.name} reported a failure; not plotting bad numbers")

    rows = []
    for line in proc.stdout.splitlines():
        m = PERF_RE.match(line)
        if m:
            rows.append(
                {
                    "name": m.group("name").strip(),
                    "ms": float(m.group("ms")),
                    "gbps": float(m.group("gbps")) if m.group("gbps") else None,
                    "gflops": float(m.group("gflops")) if m.group("gflops") else None,
                }
            )
    return rows


def draw_bars(ax, rows, key, unit, peak, peak_label):
    names = [r["name"] for r in rows]
    values = [r[key] for r in rows]

    bars = ax.barh(names, values, color="#3b7dd8")
    ax.bar_label(bars, fmt="%.1f", padding=3)
    ax.invert_yaxis()  # first kernel on top
    ax.set_xlabel(unit)
    ax.spines[["top", "right"]].set_visible(False)

    if peak:
        ax.axvline(peak, color="#d83b3b", linestyle="--", linewidth=1.5)
        ax.text(
            peak, -0.45, f" {peak_label}: {peak:g} {unit}",
            color="#d83b3b", ha="right", va="bottom", fontsize=9, rotation=0,
        )
        ax.set_xlim(0, peak * 1.08)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--build-dir", type=Path, default=Path("build"))
    parser.add_argument("--out", type=Path, default=Path("perf.png"))
    parser.add_argument(
        "--peak-bw", type=float, default=None,
        help="GPU peak DRAM bandwidth in GB/s (drawn as the ceiling)",
    )
    parser.add_argument(
        "--peak-flops", type=float, default=None,
        help="GPU peak fp32 throughput in GFLOP/s (drawn as the ceiling)",
    )
    args = parser.parse_args()

    exes = find_test_binaries(args.build_dir)
    if not exes:
        sys.exit(
            f"error: no test binaries under {args.build_dir}/ — build first:\n"
            "  cmake --build build --config Release"
        )

    rows = []
    for exe in exes:
        rows += run_and_parse(exe)

    compute = [r for r in rows if r["gflops"] is not None]
    memory = [r for r in rows if r["gbps"] is not None]
    if not compute and not memory:
        sys.exit("error: no perf lines parsed from test output")

    panels = [p for p in (
        (compute, "gflops", "GFLOP/s", args.peak_flops, "fp32 peak"),
        (memory, "gbps", "GB/s", args.peak_bw, "DRAM peak"),
    ) if p[0]]

    fig, axes = plt.subplots(
        len(panels), 1,
        figsize=(8, 1.2 + 1.1 * sum(len(p[0]) for p in panels)),
        squeeze=False,
    )
    for ax, (panel_rows, key, unit, peak, peak_label) in zip(axes.flat, panels):
        draw_bars(ax, panel_rows, key, unit, peak, peak_label)

    fig.suptitle("Kernel throughput (median-timed)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
