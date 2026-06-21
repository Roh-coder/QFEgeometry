#!/usr/bin/env python3
"""Plot the k(theta)=arcsinh(cot(theta)) manifold over angle-frame scatter points."""

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
from evaluate_k_series_terms import DatasetSpec, load_geometry


def _format_pi_fraction(value: float, _pos: float) -> str:
    if abs(value) < 1e-12:
        return "0"

    frac = Fraction(value / math.pi).limit_denominator(36)
    num = frac.numerator
    den = frac.denominator
    sign = "-" if num < 0 else ""
    num = abs(num)

    if den == 1:
        if num == 1:
            return f"{sign}pi"
        return f"{sign}{num}pi"

    if num == 1:
        return f"{sign}pi/{den}"
    return f"{sign}{num}pi/{den}"


def theta_frame_samples(angles: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (x, y, k) for all cyclic choices: x=(theta_i-theta_j)/sqrt(3), y=theta_k."""
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    k_parts: list[np.ndarray] = []

    for i_idx, j_idx, k_idx in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        theta_i = angles[:, i_idx]
        theta_j = angles[:, j_idx]
        theta_k = angles[:, k_idx]

        x_parts.append((theta_i - theta_j) / np.sqrt(3.0))
        y_parts.append(theta_k)
        # User-requested definition: k = 1/2 arcsinh(cot(theta))
        k_parts.append(0.5 * np.arcsinh(1.0 / np.tan(theta_k)))

    return np.concatenate(x_parts), np.concatenate(y_parts), np.concatenate(k_parts)


def boundary_segments_from_triangulation(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    triangles: np.ndarray,
) -> np.ndarray:
    """Return line segments for outer boundary edges of a 2D triangulation."""
    edge_counts: dict[tuple[int, int], int] = {}
    for tri in triangles:
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        for a_idx, b_idx in ((i0, i1), (i1, i2), (i2, i0)):
            edge = (a_idx, b_idx) if a_idx < b_idx else (b_idx, a_idx)
            edge_counts[edge] = edge_counts.get(edge, 0) + 1

    segments: list[list[tuple[float, float]]] = []
    for (a_idx, b_idx), count in edge_counts.items():
        if count == 1:
            segments.append(
                [
                    (float(x_vals[a_idx]), float(y_vals[a_idx])),
                    (float(x_vals[b_idx]), float(y_vals[b_idx])),
                ]
            )

    return np.array(segments, dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="refined:q5:64",
        help="Dataset spec such as refined:q5:64 or projected:q5:64",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Playground/q5_k64_theta_k_manifold.png"),
        help="Output PNG path",
    )
    parser.add_argument(
        "--heatmap-output",
        type=Path,
        default=Path("Playground/q5_k64_theta_k_heatmap.png"),
        help="Optional 2D rectangular heatmap output PNG path (set empty to skip)",
    )
    parser.add_argument(
        "--x-min",
        type=float,
        default=-5.0 * math.pi / 24.0,
        help="Heatmap x-min in radians (default -5pi/24)",
    )
    parser.add_argument(
        "--x-max",
        type=float,
        default=5.0 * math.pi / 24.0,
        help="Heatmap x-max in radians (default 5pi/24)",
    )
    parser.add_argument(
        "--theta3-min",
        type=float,
        default=math.pi / 12.0,
        help="Heatmap theta_3 min in radians (default pi/12)",
    )
    parser.add_argument(
        "--theta3-max",
        type=float,
        default=math.pi / 2.0,
        help="Heatmap theta_3 max in radians (default pi/2)",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=420,
        help="Heatmap grid resolution per axis (default 420)",
    )
    parser.add_argument(
        "--elev",
        type=float,
        default=28.0,
        help="3D view elevation angle in degrees",
    )
    parser.add_argument(
        "--azim",
        type=float,
        default=-62.0,
        help="3D view azimuth angle in degrees",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = DatasetSpec.parse(args.dataset)

    verts, tris = load_geometry(spec)
    angles = sd.primal_triangle_angles(verts, tris)
    x_vals, y_vals, k_vals = theta_frame_samples(angles)

    triang = mtri.Triangulation(x_vals, y_vals)
    domain_outline = boundary_segments_from_triangulation(x_vals, y_vals, triang.triangles)

    fig = plt.figure(figsize=(9.0, 7.2))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_trisurf(
        triang,
        k_vals,
        cmap="viridis",
        linewidth=0.1,
        antialiased=True,
        alpha=0.95,
    )

    tick_step = math.pi / 36.0
    pi_formatter = mticker.FuncFormatter(_format_pi_fraction)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(base=tick_step))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(base=tick_step))
    ax.xaxis.set_major_formatter(pi_formatter)
    ax.yaxis.set_major_formatter(pi_formatter)

    ax.set_xlabel(r"$(\theta_i-\theta_j)/\sqrt{3}$ (rad)", labelpad=10)
    ax.set_ylabel(r"$\theta_3$ (rad)", labelpad=10)
    ax.set_zlabel(r"$k=\frac{1}{2}\operatorname{arcsinh}(\cot(\theta_3))$", labelpad=10)
    ax.set_title(rf"k-manifold over angle frame ({spec.label})")
    ax.view_init(elev=args.elev, azim=args.azim)

    cbar = fig.colorbar(surf, ax=ax, shrink=0.7, pad=0.08)
    cbar.set_label(r"$k=\frac{1}{2}\operatorname{arcsinh}(\cot(\theta_3))$")

    fig.tight_layout()
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)

    heatmap_path = str(args.heatmap_output).strip()
    if heatmap_path:
        if args.x_max <= args.x_min:
            raise ValueError("--x-max must be greater than --x-min")
        if args.theta3_max <= args.theta3_min:
            raise ValueError("--theta3-max must be greater than --theta3-min")
        if args.grid_size < 2:
            raise ValueError("--grid-size must be at least 2")

        fig2, ax2 = plt.subplots(figsize=(8.0, 6.8))
        x_grid = np.linspace(args.x_min, args.x_max, args.grid_size)
        theta3_grid = np.linspace(args.theta3_min, args.theta3_max, args.grid_size)
        x_mesh, theta3_mesh = np.meshgrid(x_grid, theta3_grid)
        k_mesh = 0.5 * np.arcsinh(1.0 / np.tan(theta3_mesh))

        heat = ax2.pcolormesh(
            x_mesh,
            theta3_mesh,
            k_mesh,
            shading="auto",
            cmap="viridis",
        )

        tick_step = math.pi / 36.0
        pi_formatter = mticker.FuncFormatter(_format_pi_fraction)
        ax2.xaxis.set_major_locator(mticker.MultipleLocator(base=tick_step))
        ax2.yaxis.set_major_locator(mticker.MultipleLocator(base=tick_step))
        ax2.xaxis.set_major_formatter(pi_formatter)
        ax2.yaxis.set_major_formatter(pi_formatter)
        ax2.set_xlim(args.x_min, args.x_max)
        ax2.set_ylim(args.theta3_min, args.theta3_max)
        ax2.set_aspect("equal", adjustable="box")

        ax2.set_xlabel(r"$(\theta_i-\theta_j)/\sqrt{3}$ (rad)")
        ax2.set_ylabel(r"$\theta_3$ (rad)")
        ax2.set_title(rf"k-heatmap on rectangular domain ({spec.label})")
        ax2.grid(True, alpha=0.18)

        if len(domain_outline) > 0:
            outline = LineCollection(domain_outline, colors="white", linewidths=1.6, alpha=0.95)
            ax2.add_collection(outline)
            outline_dark = LineCollection(domain_outline, colors="black", linewidths=0.6, alpha=0.55)
            ax2.add_collection(outline_dark)

        cbar2 = fig2.colorbar(heat, ax=ax2, pad=0.02)
        cbar2.set_label(r"$k=\frac{1}{2}\operatorname{arcsinh}(\cot(\theta_3))$")

        fig2.tight_layout()
        fig2.savefig(Path(heatmap_path), dpi=180, bbox_inches="tight")
        plt.close(fig2)

    print(f"Saved {args.output}")
    if heatmap_path:
        print(f"Saved {heatmap_path}")
    print(f"dataset: {spec.label}")
    print(f"samples: {len(x_vals)}")
    print(f"k range: {float(np.min(k_vals)):.12e} .. {float(np.max(k_vals)):.12e}")


if __name__ == "__main__":
    main()
