#!/usr/bin/env python3
"""Create a multi-panel PNG of primal/dual area histograms for all K datasets.

Scans data/q*/rvec_*.dat and matches each to Triangle_*.dat with the same K.
Each panel overlays:
- Fundamental triangle area histogram
- Dual cell area histogram (from circumcenter dual)

Example:
  python Playground/make_all_k_histograms.py \
      --repo-root /workspaces/QFEgeometry \
      --output Playground/all_k_area_histograms.png
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_simulation import (
    circumcenters_from_triangles,
    dual_polygons_and_areas,
    load_triangles,
    load_vertices_rvec,
    triangle_areas,
)


K_RE = re.compile(r"_(\d+)\.dat$")


def extract_k(path: Path) -> int:
    m = K_RE.search(path.name)
    if not m:
        raise ValueError(f"Could not extract K from filename: {path}")
    return int(m.group(1))


def discover_pairs(repo_root: Path) -> list[tuple[str, int, Path, Path]]:
    pairs: list[tuple[str, int, Path, Path]] = []
    for qdir in sorted((repo_root / "data").glob("q*")):
        if not qdir.is_dir():
            continue
        qname = qdir.name
        rvec_files = sorted(qdir.glob("rvec_*.dat"), key=lambda p: extract_k(p))
        for rvec in rvec_files:
            k = extract_k(rvec)
            tri = qdir / f"Triangle_{k}.dat"
            if tri.exists():
                pairs.append((qname, k, rvec, tri))
    return pairs
def compute_areas(rvec_file: Path, tri_file: Path, include_boundary_dual: bool = True) -> tuple[np.ndarray, np.ndarray]:
    vertices = load_vertices_rvec(rvec_file)
    triangles = load_triangles(tri_file, len(vertices))

    primal = triangle_areas(vertices, triangles)
    centers = circumcenters_from_triangles(vertices, triangles)
    _, dual = dual_polygons_and_areas(
        vertices,
        triangles,
        centers,
        include_boundary=include_boundary_dual,
    )
    return primal, dual


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build multi-panel area histograms for every K dataset")
    p.add_argument("--repo-root", type=Path, default=Path.cwd(), help="Repository root (default: cwd)")
    p.add_argument("--output", type=Path, default=Path("Playground/all_k_area_histograms.png"), help="Output path (PNG or PDF)")
    p.add_argument("--bins", type=int, default=32, help="Histogram bins per panel")
    p.add_argument("--cols", type=int, default=5, help="Number of panel columns")
    p.add_argument("--qfilter", type=str, default=None, help="Only include this q directory (e.g. q5)")
    p.add_argument(
        "--exclude-boundary-dual",
        action="store_true",
        help="Exclude boundary dual cells from dual-area histograms",
    )
    return p


def safe_hist(ax, data: np.ndarray, bins: int, **kwargs) -> None:
    data = np.asarray(data, dtype=float)
    if data.size == 0:
        return
    dmin = float(np.min(data))
    dmax = float(np.max(data))
    if not np.isfinite(dmin) or not np.isfinite(dmax):
        return
    if abs(dmax - dmin) < 1e-12:
        width = max(abs(dmin) * 1e-3, 1e-6)
        hist_range = (dmin - width, dmax + width)
        ax.hist(data, bins=1, range=hist_range, **kwargs)
        return
    ax.hist(data, bins=bins, **kwargs)


def main() -> int:
    args = build_parser().parse_args()

    repo_root = args.repo_root.resolve()
    pairs = discover_pairs(repo_root)
    if args.qfilter:
        pairs = [(q, k, r, t) for q, k, r, t in pairs if q == args.qfilter]
    if not pairs:
        raise SystemExit("No matched rvec_K/Triangle_K dataset pairs found under data/q*/")

    n = len(pairs)
    cols = max(1, args.cols)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4.1 * cols, 2.8 * rows), squeeze=False)
    axes_flat = axes.ravel()

    for i, (qname, k, rvec, tri) in enumerate(pairs):
        ax = axes_flat[i]
        primal, dual = compute_areas(rvec, tri, include_boundary_dual=not args.exclude_boundary_dual)

        safe_hist(ax, primal, bins=args.bins, alpha=0.65, color="#4e79a7", edgecolor="none", label="triangles")
        if dual.size > 0:
            safe_hist(ax, dual, bins=args.bins, alpha=0.55, color="#e15759", edgecolor="none", label="dual")
        else:
            ax.text(0.5, 0.82, "no interior dual cells", transform=ax.transAxes, ha="center", fontsize=8)

        ax.set_title(f"{qname}, K={k}")
        ax.set_xlabel("Area")
        ax.set_ylabel("Count")
        ax.tick_params(axis="both", labelsize=8)

        if i == 0:
            ax.legend(fontsize=8)

    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle(f"Primal and Dual Area Histograms for All K Datasets (n={n})", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    out = args.output
    if not out.is_absolute():
        out = repo_root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
