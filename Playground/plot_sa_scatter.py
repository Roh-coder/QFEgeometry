#!/usr/bin/env python3
"""Plot the (s, a) scatter for a spherical triangle dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import spherical_dual as sd
from evaluate_k_series_terms import DatasetSpec, kij_samples, load_geometry


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

    # In the delta1 + delta2 + delta3 = 0 plane, (s, a / sqrt(3)) is an
    # orthonormal coordinate choice up to an overall scale, so cyclic relabeling
    # acts as an exact 120-degree rotation instead of an anisotropic shear.
    return (
        s_vals,
        a_vals / np.sqrt(3.0),
        r"$s = \delta_i + \delta_j$ (rad)",
        r"$a / \sqrt{3}$ (rad)",
        r"Triangle symmetry-frame scatter $(s, a / \sqrt{3})$",
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
        choices=("symmetry", "raw"),
        default="symmetry",
        help="Plot either the raw (s, a) coordinates or the symmetry-adapted frame.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Playground/q5_k64_sa_scatter_rad.png"),
        help="Output PNG path",
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

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"{title} ({spec.label})")
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {args.output}")
    print(f"frame: {args.frame}")
    print(f"x range: {x_min:.12e} .. {x_max:.12e}")
    print(f"y range: {y_min:.12e} .. {y_max:.12e}")


if __name__ == "__main__":
    main()