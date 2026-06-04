#!/usr/bin/env python3
"""Evaluate truncation orders for the symmetry-respecting K_ij(s) expansion."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import sympy as sp

sys.path.append(str(Path(__file__).resolve().parent))
import spherical_dual as sd


@dataclass(frozen=True)
class DatasetSpec:
    mode: str
    q: int
    k: int

    @property
    def label(self) -> str:
        return f"q{self.q} {self.mode} K={self.k}"

    @classmethod
    def parse(cls, text: str) -> "DatasetSpec":
        parts = text.split(":")
        if len(parts) != 3:
            raise ValueError(f"Invalid dataset spec '{text}'. Use refined:q5:64 or projected:q5:64")
        mode, qpart, kpart = parts
        if mode not in {"refined", "projected"}:
            raise ValueError(f"Unsupported mode '{mode}' in dataset spec '{text}'")
        if not qpart.startswith("q"):
            raise ValueError(f"Dataset q value must look like q5 in '{text}'")
        return cls(mode=mode, q=int(qpart[1:]), k=int(kpart))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Dataset spec: refined:q5:4 or projected:q5:64. Repeatable.",
    )
    parser.add_argument(
        "--target",
        type=float,
        action="append",
        default=None,
        help="Absolute max-error target. Repeatable. Defaults to 1e-2, 1e-4, 1e-6.",
    )
    parser.add_argument(
        "--max-order",
        type=int,
        default=12,
        help="Largest Taylor order to test (default 12).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print max and RMS error for every tested order.",
    )
    return parser.parse_args()


def series_coeffs(max_order: int) -> np.ndarray:
    delta = sp.symbols("delta")
    expr = sp.asinh(sp.cot(sp.pi / 3 + delta)) / 2
    series = sp.series(expr, delta, 0, max_order + 1).removeO().expand()
    return np.array(
        [float(sp.N(series.coeff(delta, order), 50)) for order in range(max_order + 1)],
        dtype=float,
    )


def make_projected_patch(k: int, radius: float) -> np.ndarray:
    """Build an unoptimized face patch by barycentric subdivision + radial projection."""
    ico_verts, ico_faces = sd._standard_icosahedron()
    a_pt, b_pt, c_pt = ico_verts[ico_faces[0]]
    points = []
    for y_idx in range(k + 1):
        for x_idx in range(k + 1):
            if x_idx + y_idx > k:
                continue
            w_a = 1.0 - (x_idx + y_idx) / k
            w_b = x_idx / k
            w_c = y_idx / k
            point = w_a * a_pt + w_b * b_pt + w_c * c_pt
            point /= np.linalg.norm(point)
            points.append(radius * point)
    return np.array(points, dtype=float)


def full_sphere_from_patch(patch_verts: np.ndarray, tri_path: Path) -> tuple[np.ndarray, np.ndarray]:
    patch_tris = sd.load_triangles(tri_path, len(patch_verts))
    verts, tris, _ = sd.build_full_sphere(patch_verts, patch_tris)
    centre, radius = sd.fit_sphere(verts)
    verts = sd.project_to_sphere(verts, centre, radius)
    return verts, tris


def load_geometry(spec: DatasetSpec) -> tuple[np.ndarray, np.ndarray]:
    tri_path = Path(f"data/q{spec.q}/Triangle_{spec.k}.dat")
    if spec.mode == "refined":
        rvec_path = Path(f"data/q{spec.q}/rvec_{spec.k}.dat")
        patch_verts = sd.load_vertices(rvec_path)
    else:
        patch_verts = make_projected_patch(spec.k, radius=float(spec.k))
    return full_sphere_from_patch(patch_verts, tri_path)


def exact_k(alpha: np.ndarray) -> np.ndarray:
    return 0.5 * np.arcsinh(1.0 / np.tan(alpha))


def trunc_k(delta: np.ndarray, coeffs: np.ndarray, order: int) -> np.ndarray:
    approx = np.full_like(delta, coeffs[0], dtype=float)
    power = np.ones_like(delta, dtype=float)
    for idx in range(1, order + 1):
        power *= delta
        approx += coeffs[idx] * power
    return approx


def kij_samples(angles: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build K_ij samples using all cyclic choices of (i,j,k).

    For each cyclic relabeling, define:
      s = delta_i + delta_j = -delta_k
      a = delta_i - delta_j
      K_ij = K(alpha_k)

    K_ij depends only on s; a is returned only as a diagnostic size scale.
    """
    delta = angles - math.pi / 3.0
    exact = exact_k(angles)

    s_parts: list[np.ndarray] = []
    a_parts: list[np.ndarray] = []
    kij_parts: list[np.ndarray] = []

    for i_idx, j_idx, k_idx in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        delta_i = delta[:, i_idx]
        delta_j = delta[:, j_idx]
        s_parts.append(delta_i + delta_j)
        a_parts.append(delta_i - delta_j)
        kij_parts.append(exact[:, k_idx])

    return (
        np.concatenate(s_parts),
        np.concatenate(a_parts),
        np.concatenate(kij_parts),
    )


def evaluate_dataset(
    spec: DatasetSpec,
    coeffs: np.ndarray,
    max_order: int,
    targets: list[float],
    verbose: bool,
) -> None:
    verts, tris = load_geometry(spec)
    angles = sd.primal_triangle_angles(verts, tris)
    s_vals, a_vals, exact_kij = kij_samples(angles)

    print(spec.label)
    print(f"  max|s| = {np.max(np.abs(s_vals)):.12e}")
    print(f"  max|a| = {np.max(np.abs(a_vals)):.12e}")

    rows: list[tuple[int, float, float]] = []
    for order in range(max_order + 1):
        kij_approx = trunc_k(-s_vals, coeffs, order)
        kij_err = np.abs(exact_kij - kij_approx)
        kij_max = float(np.max(kij_err))
        kij_rms = float(np.sqrt(np.mean(kij_err * kij_err)))
        rows.append((order, kij_max, kij_rms))
        if verbose:
            print(f"  order {order:2d}: K_ij max = {kij_max:.12e}, rms = {kij_rms:.12e}")

    for target in targets:
        kij_order = None
        for order, kij_max, _ in rows:
            if kij_order is None and kij_max < target:
                kij_order = order
        kij_terms = None if kij_order is None else kij_order + 1
        print(f"  target {target:.0e}: K_ij min order = {kij_order}, min terms = {kij_terms}")

    print()


def main() -> None:
    args = parse_args()
    datasets = args.dataset or ["refined:q5:4", "refined:q5:64", "projected:q5:64"]
    targets = args.target or [1e-2, 1e-4, 1e-6]
    specs = [DatasetSpec.parse(text) for text in datasets]
    coeffs = series_coeffs(args.max_order)
    for spec in specs:
        evaluate_dataset(spec, coeffs, args.max_order, targets, args.verbose)


if __name__ == "__main__":
    main()