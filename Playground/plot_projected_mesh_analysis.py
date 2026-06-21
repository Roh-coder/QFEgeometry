#!/usr/bin/env python3
"""Run the spherical-mesh analysis pipeline on the naively-projected (non-optimised) mesh."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import spherical_dual as sd
from evaluate_k_series_terms import make_projected_patch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q", type=int, default=5, help="Icosahedral subdivision q (default 5)")
    parser.add_argument("--k", type=int, default=64, help="Refinement level K (default 64)")
    parser.add_argument("--panel", type=Path,
                        default=Path("Playground/q5proj_k64_mesh_analysis.png"),
                        help="Output path for combined 4-panel PNG")
    parser.add_argument("--bins", type=int, default=40, help="Histogram bin count (default 40)")
    parser.add_argument("--cmap", default="viridis", help="Colormap for dual heatmap")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    K = args.k
    q = args.q

    tri_path = Path(f"data/q{q}/Triangle_{K}.dat")
    if not tri_path.exists():
        raise FileNotFoundError(f"Triangle connectivity file not found: {tri_path}")

    print(f"Building projected patch  q={q}  K={K} …")
    patch_verts = make_projected_patch(K, radius=float(K))
    n_patch = len(patch_verts)
    print(f"  {n_patch} patch vertices")

    patch_tris = sd.load_triangles(tri_path, n_patch)
    print(f"  {len(patch_tris)} patch triangles")

    print("Tiling across all 20 icosahedral faces …")
    verts, tris, seam_mask = sd.build_full_sphere(patch_verts, patch_tris)
    print(f"  Full sphere: {len(verts)} vertices, {len(tris)} triangles")

    centre, radius = sd.fit_sphere(verts)
    verts = sd.project_to_sphere(verts, centre, radius)
    print(f"  Sphere radius: {radius:.6f}")

    cc = sd.circumcenters(verts, tris)
    cc_sph = sd.project_to_sphere(cc, centre, radius)

    polys, dareas, pidx, _ = sd.build_dual(verts, tris, cc_sph, include_boundary=True)
    tareas = sd.primal_areas(verts, tris)
    print(f"  Primal area: min={tareas.min():.6f}  max={tareas.max():.6f}  sum={tareas.sum():.6f}")
    print(f"  Dual area:   min={dareas.min():.6f}  max={dareas.max():.6f}  sum={dareas.sum():.6f}")

    k_label = f"K={K} projected"
    fig = sd.plot_four_panel(
        verts, tris, tareas,
        polys, dareas,
        centre, radius,
        bins=args.bins,
        cmap=args.cmap,
        title=f"Spherical mesh analysis – {k_label}",
    )

    fig.savefig(args.panel, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {args.panel}")


if __name__ == "__main__":
    main()
