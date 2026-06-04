#!/usr/bin/env python3
"""Plot the (s, a) scatter for a spherical triangle dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import spherical_dual as sd
from evaluate_k_series_terms import DatasetSpec, kij_samples, load_geometry


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

    fig, ax = plt.subplots(figsize=(7.5, 6.8))
    ax.scatter(s_vals, a_vals, color="#2a6fbb", s=10, alpha=0.8, linewidths=0)
    ax.axhline(0.0, color="0.45", lw=0.8, alpha=0.6)
    ax.axvline(0.0, color="0.45", lw=0.8, alpha=0.6)

    x_min = float(np.min(s_vals))
    x_max = float(np.max(s_vals))
    y_min = float(np.min(a_vals))
    y_max = float(np.max(a_vals))
    span = max(x_max - x_min, y_max - y_min, 1e-6)
    pad = 0.08 * span
    ax.set_xlim(x_min - pad, x_max + pad)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel(r"$s = \delta_i + \delta_j$ (rad)")
    ax.set_ylabel(r"$a = \delta_i - \delta_j$ (rad)")
    ax.set_title(f"Triangle (s, a) scatter ({spec.label})")
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {args.output}")
    print(f"s range: {x_min:.12e} .. {x_max:.12e}")
    print(f"a range: {y_min:.12e} .. {y_max:.12e}")


if __name__ == "__main__":
    main()