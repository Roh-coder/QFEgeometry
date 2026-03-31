#!/usr/bin/env python3
"""Replicate core functionality of Mathematica/PlotSimulation.nb.

Features:
- Plot triangulated surfaces from rvec_*.dat + Triangle_*.dat files.
- Plot polygon soup from rows of flattened triangle vertices (data.txt).
- Optionally overlay circumcenters from circum.txt.
- Generate and plot a dual mesh built from triangle circumcenters.

Examples:
  python Playground/plot_simulation.py --preset q5 --show
  python Playground/plot_simulation.py --preset q3 --output q3_mesh.png
    python Playground/plot_simulation.py --preset q5 --dual --output q5_dual.png
  python Playground/plot_simulation.py --data-file Mathematica/data.txt \
      --circum-file Mathematica/circum.txt --show
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def _load_table(path: Path, dtype=float) -> np.ndarray:
    data = np.loadtxt(path, dtype=dtype)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def load_vertices_rvec(path: Path) -> np.ndarray:
    """Load xyz from columns 2,3,4 in Mathematica 1-based indexing.

    Mathematica used: data[[All, {2,3,4}]]
    Python equivalent: data[:, 1:4]
    """
    data = _load_table(path, dtype=float)
    if data.shape[1] < 4:
        raise ValueError(f"Expected at least 4 columns in {path}, got {data.shape[1]}")
    return data[:, 1:4]


def load_triangles(path: Path, n_vertices: int) -> np.ndarray:
    """Load triangle indices and normalize to 0-based indexing for Python."""
    tri = _load_table(path, dtype=int)
    if tri.shape[1] < 3:
        raise ValueError(f"Expected at least 3 columns in {path}, got {tri.shape[1]}")
    tri = tri[:, :3].astype(int)

    min_idx = int(tri.min())
    max_idx = int(tri.max())

    # Notebook adds +1 because Mathematica is 1-based.
    # Here we keep Python 0-based indices and auto-normalize if needed.
    if min_idx >= 1 and max_idx <= n_vertices:
        tri = tri - 1
    elif min_idx < 0 or max_idx >= n_vertices:
        raise ValueError(
            f"Triangle indices out of range after normalization: min={min_idx}, max={max_idx}, n={n_vertices}"
        )

    return tri


def polygons_from_flat_rows(path: Path) -> np.ndarray:
    """Load rows where each row is 9 floats => 3 xyz points => one triangle."""
    data = _load_table(path, dtype=float)
    if data.shape[1] % 3 != 0:
        raise ValueError(f"Expected columns to be multiple of 3 in {path}, got {data.shape[1]}")
    if data.shape[1] != 9:
        # Keep generic, but PlotSimulation.nb expects triangle rows.
        pts_per_row = data.shape[1] // 3
        if pts_per_row != 3:
            raise ValueError(f"Expected exactly 9 columns (triangle row) in {path}, got {data.shape[1]}")

    return data.reshape(-1, 3, 3)


def circumcenters_from_triangles(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Compute circumcenters for each triangle in 3D."""
    centers = np.zeros((len(triangles), 3), dtype=float)
    for i, tri in enumerate(triangles):
        a, b, c = vertices[tri]
        ab = b - a
        ac = c - a
        n = np.cross(ab, ac)
        denom = 2.0 * np.dot(n, n)
        if denom <= 1e-16:
            # Degenerate triangle: fall back to centroid.
            centers[i] = (a + b + c) / 3.0
            continue

        term1 = np.cross(np.cross(ab, ac), ab) * np.dot(ac, ac)
        term2 = np.cross(ac, np.cross(ab, ac)) * np.dot(ab, ab)
        centers[i] = a + (term1 + term2) / denom

    return centers


def triangle_areas(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Compute area of each triangle in a mesh."""
    a = vertices[triangles[:, 0]]
    b = vertices[triangles[:, 1]]
    c = vertices[triangles[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def polygon_area_3d(points: np.ndarray) -> float:
    """Area of a near-planar 3D polygon via projection to a fitted 2D plane."""
    if len(points) < 3:
        return 0.0

    center = points.mean(axis=0)
    rel = points - center

    # Best-fit plane from SVD.
    _, _, vh = np.linalg.svd(rel, full_matrices=False)
    e0 = vh[0]
    e1 = vh[1]

    x = rel @ e0
    y = rel @ e1

    # Shoelace formula in projected 2D coordinates.
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    area2 = np.sum(x * y_next - y * x_next)
    return float(0.5 * abs(area2))


def order_points_around_vertex(points: np.ndarray, vertex: np.ndarray) -> np.ndarray:
    """Sort points cyclically around a primal vertex using local tangent basis."""
    if len(points) <= 2:
        return points

    n = np.asarray(vertex, dtype=float)
    n_norm = np.linalg.norm(n)
    if n_norm <= 1e-14:
        # Fallback normal from points if vertex is near origin.
        ref = points[0]
        n = ref / (np.linalg.norm(ref) + 1e-14)
    else:
        n = n / n_norm

    v0 = points[0] - vertex
    v0 = v0 - n * np.dot(v0, n)
    v0_norm = np.linalg.norm(v0)
    if v0_norm <= 1e-14:
        v0 = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(v0, n)) > 0.9:
            v0 = np.array([0.0, 1.0, 0.0])
        v0 = v0 - n * np.dot(v0, n)
        v0 = v0 / (np.linalg.norm(v0) + 1e-14)
    else:
        v0 = v0 / v0_norm

    v1 = np.cross(n, v0)
    v1 = v1 / (np.linalg.norm(v1) + 1e-14)

    rel = points - vertex
    x = rel @ v0
    y = rel @ v1
    angles = np.arctan2(y, x)
    order = np.argsort(angles)
    return points[order]


def dual_polygons_and_areas(
    vertices: np.ndarray,
    triangles: np.ndarray,
    centers: np.ndarray,
    include_boundary: bool = True,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Build dual polygons around each primal vertex and compute their areas."""
    vertex_to_faces: list[list[int]] = [[] for _ in range(len(vertices))]
    edge_to_faces: dict[tuple[int, int], list[int]] = {}

    for fi, tri in enumerate(triangles):
        i, j, k = int(tri[0]), int(tri[1]), int(tri[2])
        for vi in (i, j, k):
            vertex_to_faces[vi].append(fi)

        for u, v in ((i, j), (j, k), (k, i)):
            key = (u, v) if u < v else (v, u)
            edge_to_faces.setdefault(key, []).append(fi)

    boundary_vertices = np.zeros(len(vertices), dtype=bool)
    for (u, v), faces in edge_to_faces.items():
        if len(faces) == 1:
            boundary_vertices[u] = True
            boundary_vertices[v] = True

    dual_polys: list[np.ndarray] = []
    dual_areas: list[float] = []

    for vi, incident_faces in enumerate(vertex_to_faces):
        if not include_boundary and boundary_vertices[vi]:
            continue

        # Need at least 3 circumcenters to define a 2D dual cell.
        if len(incident_faces) < 3:
            continue

        pts = centers[np.asarray(incident_faces, dtype=int)]
        ordered = order_points_around_vertex(pts, vertices[vi])
        dual_polys.append(ordered)
        dual_areas.append(polygon_area_3d(ordered))

    return dual_polys, np.asarray(dual_areas, dtype=float)


def dual_edges_from_triangles(triangles: np.ndarray) -> np.ndarray:
    """Return dual edges by connecting circumcenters of adjacent triangles."""
    edge_to_faces: dict[tuple[int, int], list[int]] = {}

    for fi, (i, j, k) in enumerate(triangles):
        primal_edges = ((i, j), (j, k), (k, i))
        for u, v in primal_edges:
            key = (u, v) if u < v else (v, u)
            edge_to_faces.setdefault(key, []).append(fi)

    dual_edges = []
    for faces in edge_to_faces.values():
        if len(faces) == 2:
            dual_edges.append((faces[0], faces[1]))

    return np.asarray(dual_edges, dtype=int)


def boundary_vertices_from_triangles(triangles: np.ndarray, n_vertices: int) -> np.ndarray:
    """Return boolean mask of boundary vertices in the primal mesh."""
    edge_to_faces: dict[tuple[int, int], int] = {}
    for i, j, k in triangles:
        for u, v in ((i, j), (j, k), (k, i)):
            key = (u, v) if u < v else (v, u)
            edge_to_faces[key] = edge_to_faces.get(key, 0) + 1

    mask = np.zeros(n_vertices, dtype=bool)
    for (u, v), count in edge_to_faces.items():
        if count == 1:
            mask[u] = True
            mask[v] = True
    return mask


def vertices_and_triangles_from_polygons(polygons: np.ndarray, decimals: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct unique vertices and triangle indices from triangle polygons."""
    key_to_idx: dict[tuple[float, float, float], int] = {}
    verts: list[np.ndarray] = []
    tris: list[list[int]] = []

    for poly in polygons:
        tri_idx = []
        for p in poly:
            key = tuple(np.round(p, decimals=decimals))
            if key not in key_to_idx:
                key_to_idx[key] = len(verts)
                verts.append(np.asarray(p, dtype=float))
            tri_idx.append(key_to_idx[key])
        tris.append(tri_idx)

    return np.asarray(verts, dtype=float), np.asarray(tris, dtype=int)


def project_points_to_sphere(points: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    """Radially project points to a sphere with given center and radius."""
    rel = points - center
    norms = np.linalg.norm(rel, axis=1)
    safe = np.where(norms > 1e-14, norms, 1.0)
    return center + rel / safe[:, None] * radius


def fit_sphere_from_points(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares sphere fit: returns center and radius."""
    p = np.asarray(points, dtype=float)
    if len(p) < 4:
        c = p.mean(axis=0)
        r = float(np.mean(np.linalg.norm(p - c, axis=1)))
        return c, r

    # Solve x^2+y^2+z^2 + Ax + By + Cz + D = 0 in least squares.
    a = np.column_stack([p[:, 0], p[:, 1], p[:, 2], np.ones(len(p))])
    b = -(p[:, 0] ** 2 + p[:, 1] ** 2 + p[:, 2] ** 2)
    coeff, *_ = np.linalg.lstsq(a, b, rcond=None)
    A, B, C, D = coeff
    center = np.array([-A / 2.0, -B / 2.0, -C / 2.0], dtype=float)
    r2 = float(np.dot(center, center) - D)
    radius = float(np.sqrt(max(r2, 0.0)))
    return center, radius


def set_equal_axes(ax, pts: np.ndarray) -> None:
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    centers = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0)
    if radius == 0:
        radius = 1.0

    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def plot_triangle_mesh(vertices: np.ndarray, triangles: np.ndarray, title: str = "Triangle mesh"):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    polys = vertices[triangles]
    coll = Poly3DCollection(polys, alpha=0.85, edgecolor="k", linewidth=0.25)
    coll.set_facecolor((0.38, 0.66, 0.87, 0.9))
    ax.add_collection3d(coll)

    set_equal_axes(ax, vertices)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    return fig, ax


def plot_dual_mesh(
    vertices: np.ndarray,
    triangles: np.ndarray,
    dual_only: bool = False,
    highlight_boundary_vertices: bool = False,
    spherical_dual: bool = False,
    show_sphere: bool = False,
    dual_face_heatmap: bool = False,
    exclude_boundary_dual_faces: bool = False,
    dual_cmap: str = "viridis",
    title: str = "Dual mesh from circumcenters",
):
    centers = circumcenters_from_triangles(vertices, triangles)
    dual_edges = dual_edges_from_triangles(triangles)

    sphere_center, sphere_radius = fit_sphere_from_points(vertices)
    if spherical_dual:
        centers = project_points_to_sphere(centers, sphere_center, sphere_radius)

    dual_polys, dual_areas = dual_polygons_and_areas(
        vertices,
        triangles,
        centers,
        include_boundary=not exclude_boundary_dual_faces,
    )

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    if not dual_only:
        polys = vertices[triangles]
        coll = Poly3DCollection(polys, alpha=0.25, edgecolor="gray", linewidth=0.2)
        coll.set_facecolor((0.72, 0.80, 0.90, 0.22))
        ax.add_collection3d(coll)

    if dual_face_heatmap and len(dual_polys) > 0:
        area_min = float(np.min(dual_areas))
        area_max = float(np.max(dual_areas))
        if abs(area_max - area_min) < 1e-14:
            area_max = area_min + 1e-14
        norm = colors.Normalize(vmin=area_min, vmax=area_max)
        cmap = plt.get_cmap(dual_cmap)

        face_colors = cmap(norm(dual_areas))
        dcoll = Poly3DCollection(dual_polys, linewidth=0.35, edgecolor=(0.15, 0.15, 0.15, 0.75), alpha=0.92)
        dcoll.set_facecolor(face_colors)
        ax.add_collection3d(dcoll)

        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array(dual_areas)
        cbar = fig.colorbar(mappable, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("Dual face area")

    if show_sphere:
        u = np.linspace(0, 2 * np.pi, 64)
        v = np.linspace(0, np.pi, 32)
        x = sphere_center[0] + sphere_radius * np.outer(np.cos(u), np.sin(v))
        y = sphere_center[1] + sphere_radius * np.outer(np.sin(u), np.sin(v))
        z = sphere_center[2] + sphere_radius * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_surface(x, y, z, color="#9ecae1", alpha=0.12, linewidth=0, antialiased=True)

    if not dual_face_heatmap:
        for i, j in dual_edges:
            seg = centers[[i, j]]
            ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color="crimson", linewidth=0.8)

        ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c="crimson", s=8, depthshade=True)

    if highlight_boundary_vertices:
        bmask = boundary_vertices_from_triangles(triangles, len(vertices))
        bverts = vertices[bmask]
        if spherical_dual and len(bverts) > 0:
            bverts = project_points_to_sphere(bverts, sphere_center, sphere_radius)
        if len(bverts) > 0:
            ax.scatter(
                bverts[:, 0],
                bverts[:, 1],
                bverts[:, 2],
                c="#ffbf00",
                s=22,
                edgecolors="black",
                linewidths=0.3,
                depthshade=True,
                label="boundary vertices",
            )
            ax.legend(loc="best")

    pts = centers if dual_only else np.vstack([vertices, centers])
    set_equal_axes(ax, pts)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    return fig, ax


def plot_area_histograms(
    primal_areas: np.ndarray,
    dual_areas: np.ndarray,
    bins: int = 40,
    title: str = "Area Histograms",
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    axes[0].hist(primal_areas, bins=bins, color="#4e79a7", edgecolor="black", alpha=0.85)
    axes[0].set_title("Fundamental Triangle Areas")
    axes[0].set_xlabel("Area")
    axes[0].set_ylabel("Count")

    axes[1].hist(dual_areas, bins=bins, color="#e15759", edgecolor="black", alpha=0.85)
    axes[1].set_title("Dual Cell Areas")
    axes[1].set_xlabel("Area")
    axes[1].set_ylabel("Count")

    fig.suptitle(title)
    fig.tight_layout()
    return fig, axes


def plot_polygons_with_circumcenters(
    polygons: np.ndarray,
    circumcenters: Optional[np.ndarray] = None,
    title: str = "Polygons + circumcenters",
):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    coll = Poly3DCollection(polygons, alpha=0.78, edgecolor="k", linewidth=0.25)
    coll.set_facecolor((0.50, 0.65, 0.85, 0.85))
    ax.add_collection3d(coll)

    if circumcenters is not None and circumcenters.size > 0:
        cc = np.asarray(circumcenters, dtype=float)
        ax.scatter(cc[:, 0], cc[:, 1], cc[:, 2], c="crimson", s=8, depthshade=True, label="circumcenters")
        ax.legend(loc="best")

    pts = polygons.reshape(-1, 3)
    if circumcenters is not None and circumcenters.size > 0:
        pts = np.vstack([pts, circumcenters])

    set_equal_axes(ax, pts)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    return fig, ax


def resolve_repo_root(script_path: Path) -> Path:
    # Playground/plot_simulation.py -> repo root is parent of Playground
    return script_path.resolve().parent.parent


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Replicate PlotSimulation.nb visualizations")
    p.add_argument("--preset", choices=["q3", "q5"], help="Load default q3/q5 rvec+triangle files")
    p.add_argument("--rvec-file", type=Path, help="Path to rvec_*.dat")
    p.add_argument("--triangle-file", type=Path, help="Path to Triangle_*.dat")
    p.add_argument("--data-file", type=Path, help="Path to flattened triangle rows file (e.g., Mathematica/data.txt)")
    p.add_argument("--circum-file", type=Path, help="Path to circumcenter file (e.g., Mathematica/circum.txt)")
    p.add_argument("--dual", action="store_true", help="Plot dual mesh from triangle circumcenters")
    p.add_argument("--dual-only", action="store_true", help="When --dual is set, hide the primal mesh")
    p.add_argument(
        "--spherical-dual",
        action="store_true",
        help="Project dual circumcenters radially onto a sphere fitted from mesh data",
    )
    p.add_argument(
        "--show-sphere",
        action="store_true",
        help="Render the fitted sphere (useful with --spherical-dual)",
    )
    p.add_argument(
        "--highlight-boundary-vertices",
        action="store_true",
        help="Highlight primal boundary vertices on dual mesh plots",
    )
    p.add_argument(
        "--dual-face-heatmap",
        action="store_true",
        help="Color dual faces by area and display a colorbar",
    )
    p.add_argument(
        "--exclude-boundary-dual-faces",
        action="store_true",
        help="Exclude boundary dual faces from heatmap/dual-face rendering",
    )
    p.add_argument("--dual-cmap", default="viridis", help="Matplotlib colormap for dual-face heatmap")
    p.add_argument("--output", type=Path, help="Output image path (PNG recommended)")
    p.add_argument("--area-hist", action="store_true", help="Produce histograms of primal and dual cell areas")
    p.add_argument("--hist-output", type=Path, help="Histogram output image path")
    p.add_argument("--hist-bins", type=int, default=40, help="Number of bins for area histograms")
    p.add_argument(
        "--exclude-boundary-dual",
        action="store_true",
        help="Exclude boundary dual cells from dual-area histogram",
    )
    p.add_argument("--show", action="store_true", help="Show interactive figure")
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    repo_root = resolve_repo_root(Path(__file__))

    fig = None
    hist_fig = None
    vertices = None
    triangles = None

    if args.preset:
        if args.preset == "q5":
            rvec = repo_root / "data" / "q5" / "rvec_10.dat"
            tri = repo_root / "data" / "q5" / "Triangle_10.dat"
        else:
            rvec = repo_root / "data" / "q3" / "rvec_8.dat"
            tri = repo_root / "data" / "q3" / "Triangle_8.dat"

        vertices = load_vertices_rvec(rvec)
        triangles = load_triangles(tri, len(vertices))
        if args.dual:
            fig, _ = plot_dual_mesh(
                vertices,
                triangles,
                dual_only=args.dual_only,
                highlight_boundary_vertices=args.highlight_boundary_vertices,
                spherical_dual=args.spherical_dual,
                show_sphere=args.show_sphere,
                dual_face_heatmap=args.dual_face_heatmap,
                exclude_boundary_dual_faces=args.exclude_boundary_dual_faces,
                dual_cmap=args.dual_cmap,
                title=f"{args.preset.upper()} dual mesh from circumcenters",
            )
        else:
            fig, _ = plot_triangle_mesh(vertices, triangles, title=f"{args.preset.upper()} triangulated surface")

    elif args.rvec_file and args.triangle_file:
        vertices = load_vertices_rvec(args.rvec_file)
        triangles = load_triangles(args.triangle_file, len(vertices))
        if args.dual:
            fig, _ = plot_dual_mesh(
                vertices,
                triangles,
                dual_only=args.dual_only,
                highlight_boundary_vertices=args.highlight_boundary_vertices,
                spherical_dual=args.spherical_dual,
                show_sphere=args.show_sphere,
                dual_face_heatmap=args.dual_face_heatmap,
                exclude_boundary_dual_faces=args.exclude_boundary_dual_faces,
                dual_cmap=args.dual_cmap,
            )
        else:
            fig, _ = plot_triangle_mesh(vertices, triangles, title="Triangulated surface")

    elif args.data_file:
        polygons = polygons_from_flat_rows(args.data_file)
        circum = _load_table(args.circum_file, dtype=float) if args.circum_file else None
        vertices, triangles = vertices_and_triangles_from_polygons(polygons)
        if args.dual:
            fig, _ = plot_dual_mesh(
                vertices,
                triangles,
                dual_only=args.dual_only,
                highlight_boundary_vertices=args.highlight_boundary_vertices,
                spherical_dual=args.spherical_dual,
                show_sphere=args.show_sphere,
                dual_face_heatmap=args.dual_face_heatmap,
                exclude_boundary_dual_faces=args.exclude_boundary_dual_faces,
                dual_cmap=args.dual_cmap,
            )
        else:
            fig, _ = plot_polygons_with_circumcenters(polygons, circumcenters=circum)

    else:
        parser.error(
            "Provide one mode: --preset q3|q5, or --rvec-file + --triangle-file, or --data-file (optional --circum-file)."
        )

    if args.area_hist:
        if vertices is None or triangles is None:
            parser.error("Could not build mesh for area histogram.")
        primal = triangle_areas(vertices, triangles)
        centers = circumcenters_from_triangles(vertices, triangles)
        _, dual = dual_polygons_and_areas(
            vertices,
            triangles,
            centers,
            include_boundary=not args.exclude_boundary_dual,
        )
        hist_fig, _ = plot_area_histograms(primal, dual, bins=max(args.hist_bins, 5))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=180, bbox_inches="tight")
        print(f"Saved: {args.output}")

    if hist_fig is not None:
        hist_out = args.hist_output
        if hist_out is None:
            if args.output:
                hist_out = args.output.with_name(args.output.stem + "_area_hist" + args.output.suffix)
            else:
                hist_out = Path("Playground/area_hist.png")
        hist_out.parent.mkdir(parents=True, exist_ok=True)
        hist_fig.savefig(hist_out, dpi=180, bbox_inches="tight")
        print(f"Saved: {hist_out}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)
        if hist_fig is not None:
            plt.close(hist_fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
