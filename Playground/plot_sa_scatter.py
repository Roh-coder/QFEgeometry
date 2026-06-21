#!/usr/bin/env python3
"""Plot the (s, a) scatter for a spherical triangle dataset."""

from __future__ import annotations

import argparse
import math
from fractions import Fraction
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.tri as mtri
import numpy as np
from matplotlib.collections import LineCollection

import spherical_dual as sd
from evaluate_k_series_terms import DatasetSpec, kij_samples, load_geometry


def boundary_segments(x_vals: np.ndarray, y_vals: np.ndarray) -> np.ndarray:
    """Return boundary edge segments of the convex hull of the triangulated point set."""
    triang = mtri.Triangulation(x_vals, y_vals)
    edge_counts: dict[tuple[int, int], int] = {}
    for tri in triang.triangles:
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        for a_idx, b_idx in ((i0, i1), (i1, i2), (i2, i0)):
            edge = (a_idx, b_idx) if a_idx < b_idx else (b_idx, a_idx)
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    segs = [
        [(float(x_vals[a]), float(y_vals[a])), (float(x_vals[b]), float(y_vals[b]))]
        for (a, b), cnt in edge_counts.items() if cnt == 1
    ]
    return np.array(segs, dtype=float)


def _format_pi_fraction(value: float, _pos: float) -> str:
    """Render a numeric tick as a compact rational multiple of pi."""
    if abs(value) < 1e-12:
        return r"$0$"

    frac = Fraction(value / math.pi).limit_denominator(36)
    num = frac.numerator
    den = frac.denominator
    sign = "-" if num < 0 else ""
    num = abs(num)

    if den == 1:
        if num == 1:
            return rf"${sign}\pi$"
        return rf"${sign}{num}\pi$"

    if num == 1:
        return rf"${sign}\pi/{den}$"
    return rf"${sign}{num}\pi/{den}$"


def project_plot_coords(
    s_vals: np.ndarray,
    a_vals: np.ndarray,
    frame: str,
) -> tuple[np.ndarray, np.ndarray, str, str, str]:
    if frame == "raw":
        return (
            s_vals,
            a_vals,
            r"$s = \delta_i + \delta_j$ (rad)",
            r"$a = \delta_i - \delta_j$ (rad)",
            "Triangle (s, a) scatter",
        )

    if frame == "symmetry_delta3":
        return (
            a_vals / np.sqrt(3.0),
            -s_vals,
            r"$a / \sqrt{3}$ (rad)",
            r"$\delta_3 = -s$ (rad)",
            r"Triangle symmetry-frame scatter $(a / \sqrt{3}, \delta_3)$",
        )

    if frame == "theta":
        theta3_vals = (math.pi / 3.0) - s_vals
        return (
            a_vals / np.sqrt(3.0),
            theta3_vals,
            r"$(\theta_i - \theta_j) / \sqrt{3}$ (rad)",
            r"$\theta_3$ (rad)",
            r"Triangle angle-frame scatter $((\theta_i-\theta_j)/\sqrt{3}, \theta_3)$",
        )

    # In the delta1 + delta2 + delta3 = 0 plane, (s, a / sqrt(3)) is an
    # orthonormal coordinate choice up to an overall scale, so cyclic relabeling
    # acts as an exact 120-degree rotation instead of an anisotropic shear.
    return (
        a_vals / np.sqrt(3.0),
        s_vals,
        r"$a / \sqrt{3}$ (rad)",
        r"$s = \delta_i + \delta_j$ (rad)",
        r"Triangle symmetry-frame scatter $(a / \sqrt{3}, s)$",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="refined:q5:64",
        help="Dataset spec such as refined:q5:64 or projected:q5:64",
    )
    parser.add_argument(
        "--frame",
        choices=("symmetry", "symmetry_delta3", "theta", "raw"),
        default="symmetry",
        help="Plot either the raw (s, a) coordinates or the symmetry-adapted frame.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Playground/q5_k64_sa_scatter_rad.png"),
        help="Output PNG path",
    )
    parser.add_argument(
        "--outline-dataset",
        default=None,
        help="Optional second dataset spec whose domain boundary is overlaid on the scatter.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = DatasetSpec.parse(args.dataset)
    verts, tris = load_geometry(spec)
    angles = sd.primal_triangle_angles(verts, tris)
    s_vals, a_vals, _ = kij_samples(angles)
    x_vals, y_vals, x_label, y_label, title = project_plot_coords(s_vals, a_vals, args.frame)

    fig, ax = plt.subplots(figsize=(7.5, 6.8))
    ax.scatter(x_vals, y_vals, color="#2a6fbb", s=10, alpha=0.8, linewidths=0)
    ax.axhline(0.0, color="0.45", lw=0.8, alpha=0.6)
    ax.axvline(0.0, color="0.45", lw=0.8, alpha=0.6)

    x_min = float(np.min(x_vals))
    x_max = float(np.max(x_vals))
    y_min = float(np.min(y_vals))
    y_max = float(np.max(y_vals))
    span = max(x_max - x_min, y_max - y_min, 1e-6)
    pad = 0.08 * span
    ax.set_xlim(x_min - pad, x_max + pad)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_aspect("equal", adjustable="box")

    if args.outline_dataset:
        ospec = DatasetSpec.parse(args.outline_dataset)
        overts, otris = load_geometry(ospec)
        oangles = sd.primal_triangle_angles(overts, otris)
        os_vals, oa_vals, _ = kij_samples(oangles)
        ox_vals, oy_vals, *_ = project_plot_coords(os_vals, oa_vals, args.frame)
        segs = boundary_segments(ox_vals, oy_vals)
        if len(segs):
            ax.add_collection(LineCollection(segs, colors="white", linewidths=1.8, alpha=0.95, zorder=3))
            ax.add_collection(LineCollection(segs, colors="crimson", linewidths=0.9, alpha=0.85, zorder=4))

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"{title} ({spec.label})")
    ax.grid(True, alpha=0.2)

    if args.frame == "theta":
        tick_step = math.pi / 36.0
        formatter = mticker.FuncFormatter(_format_pi_fraction)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(base=tick_step))
        ax.yaxis.set_major_locator(mticker.MultipleLocator(base=tick_step))
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)

    fig.tight_layout()
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {args.output}")
    print(f"frame: {args.frame}")
    print(f"x range: {x_min:.12e} .. {x_max:.12e}")
    print(f"y range: {y_min:.12e} .. {y_max:.12e}")


if __name__ == "__main__":
    main()