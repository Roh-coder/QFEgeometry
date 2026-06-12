#!/usr/bin/env python3
"""Plot exact coupling-ratio scatter in (k1/k3, k2/k3) space."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import spherical_dual as sd
from evaluate_k_series_terms import DatasetSpec, exact_k, load_geometry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="refined:q5:64",
        help="Dataset spec such as refined:q5:64 or projected:q5:64",
    )
    parser.add_argument(
        "--orbit",
        choices=("cyclic", "single"),
        default="cyclic",
        help="Plot only (k1/k3, k2/k3) or include all cyclic relabelings in the same chart.",
    )
    parser.add_argument(
        "--scale",
        choices=("linear", "loglog"),
        default="linear",
        help="Use ordinary linear axes or a log-log parameter-space plot.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional PNG path; defaults to Playground/q*_k*_kratio_scatter_<mode>_<orbit>[_loglog].png",
    )
    return parser.parse_args()


def ratio_samples(couplings: np.ndarray, orbit: str) -> tuple[np.ndarray, np.ndarray]:
    if orbit == "single":
        denom = couplings[:, 2]
        safe = np.where(np.abs(denom) < 1e-14, np.nan, denom)
        x_vals = couplings[:, 0] / safe
        y_vals = couplings[:, 1] / safe
    else:
        x_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []
        for i_idx, j_idx, k_idx in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            denom = couplings[:, k_idx]
            safe = np.where(np.abs(denom) < 1e-14, np.nan, denom)
            x_parts.append(couplings[:, i_idx] / safe)
            y_parts.append(couplings[:, j_idx] / safe)
        x_vals = np.concatenate(x_parts)
        y_vals = np.concatenate(y_parts)

    mask = np.isfinite(x_vals) & np.isfinite(y_vals) & (x_vals > 0.0) & (y_vals > 0.0)
    return x_vals[mask], y_vals[mask]


def default_output_path(spec: DatasetSpec, orbit: str, scale: str) -> Path:
    suffix = "_loglog" if scale == "loglog" else ""
    return Path(f"Playground/q{spec.q}_k{spec.k}_kratio_scatter_{spec.mode}_{orbit}{suffix}.png")


def set_plot_limits(ax: plt.Axes, x_vals: np.ndarray, y_vals: np.ndarray, scale: str) -> tuple[float, float, float, float]:
    x_min = float(np.min(x_vals))
    x_max = float(np.max(x_vals))
    y_min = float(np.min(y_vals))
    y_max = float(np.max(y_vals))

    if scale == "loglog":
        x_factor = float(np.exp(0.08 * np.log(x_max / x_min))) if x_max > x_min else 1.08
        y_factor = float(np.exp(0.08 * np.log(y_max / y_min))) if y_max > y_min else 1.08
        ax.set_xlim(x_min / x_factor, x_max * x_factor)
        ax.set_ylim(y_min / y_factor, y_max * y_factor)
    else:
        span = max(x_max - x_min, y_max - y_min, 1e-6)
        pad = 0.08 * span
        ax.set_xlim(x_min - pad, x_max + pad)
        ax.set_ylim(y_min - pad, y_max + pad)

    return x_min, x_max, y_min, y_max


def main() -> None:
    args = parse_args()
    spec = DatasetSpec.parse(args.dataset)
    output_path = args.output or default_output_path(spec, args.orbit, args.scale)

    verts, tris = load_geometry(spec)
    angles = sd.primal_triangle_angles(verts, tris)
    couplings = exact_k(angles)
    x_vals, y_vals = ratio_samples(couplings, args.orbit)

    fig, ax = plt.subplots(figsize=(7.5, 6.8))
    ax.scatter(x_vals, y_vals, color="#2a6fbb", s=10, alpha=0.8, linewidths=0)

    if args.scale == "loglog":
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.axhline(1.0, color="0.45", lw=0.8, alpha=0.6)
    ax.axvline(1.0, color="0.45", lw=0.8, alpha=0.6)
    diag_lo = min(float(np.min(x_vals)), float(np.min(y_vals)))
    diag_hi = max(float(np.max(x_vals)), float(np.max(y_vals)))
    ax.plot([diag_lo, diag_hi], [diag_lo, diag_hi], color="0.55", lw=0.8, alpha=0.35)

    x_min, x_max, y_min, y_max = set_plot_limits(ax, x_vals, y_vals, args.scale)
    ax.set_aspect("equal", adjustable="box")

    orbit_label = "cyclic orbit" if args.orbit == "cyclic" else "single labeling"
    scale_label = "log-log " if args.scale == "loglog" else ""
    ax.set_xlabel(r"$k_1 / k_3$")
    ax.set_ylabel(r"$k_2 / k_3$")
    ax.set_title(f"Exact {scale_label}coupling-ratio scatter ({orbit_label}) ({spec.label})")
    ax.grid(True, which="both", alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {output_path}")
    print(f"orbit: {args.orbit}")
    print(f"scale: {args.scale}")
    print(f"x range: {x_min:.12e} .. {x_max:.12e}")
    print(f"y range: {y_min:.12e} .. {y_max:.12e}")


if __name__ == "__main__":
    main()