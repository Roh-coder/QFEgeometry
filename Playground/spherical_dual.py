#!/usr/bin/env python3
"""
Spherical dual mesh pipeline.

Steps (mirroring PlotSimulation.nb):
  1. Load the single-face patch from rvec_K.dat + Triangle_K.dat
     (rvec stores one icosahedral face: (K+1)(K+2)/2 optimised vertices)
  2. Tile the patch across all 20 faces of the icosahedron by applying the
     corresponding rotation to each face, merging shared edge/corner vertices
     → this builds the FULL closed spherical mesh
  3. Fit a sphere to the primal vertices (least-squares) and project every
     vertex exactly onto it
  4. Compute the circumcenter of every spherical-mesh triangle
  5. Project those circumcenters onto the same sphere → dual-mesh vertices
  6. Build dual polygons: for each primal vertex v, collect the projected
     circumcenters of all triangles that share v, sort cyclically, close the polygon
  7. Compute each dual-face area (SVD best-fit plane + shoelace)
  8. Render the spherical dual mesh as a 3-D heatmap (face colour = area)
  9. Plot a histogram of all dual-face areas and primal triangle areas

Usage
-----
  python Playground/spherical_dual.py --rvec data/q5/rvec_8.dat \
         --tri data/q5/Triangle_8.dat \
         --heatmap Playground/q5_k8_heatmap.png \
         --hist    Playground/q5_k8_hist.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ---------------------------------------------------------------------------
# 1.  Data loading
# ---------------------------------------------------------------------------

def load_vertices(path: Path) -> np.ndarray:
    """Return (N,3) float array from rvec file – columns 1,2,3 (0-based)."""
    data = np.loadtxt(path)          # each row: index x y z [...]
    return data[:, 1:4].astype(float)


def load_triangles(path: Path, n_verts: int) -> np.ndarray:
    """Return (F,3) int array of 0-based triangle indices."""
    tri = np.loadtxt(path, dtype=int)[:, :3]
    # Auto-detect 1-based indexing and convert to 0-based
    if tri.min() >= 1 and tri.max() <= n_verts:
        tri -= 1
    if tri.min() < 0 or tri.max() >= n_verts:
        raise ValueError(f"Triangle indices out of range: min={tri.min()}, max={tri.max()}, n={n_verts}")
    return tri


# ---------------------------------------------------------------------------
# 1b.  Patch → full sphere tiling
# ---------------------------------------------------------------------------

def _standard_icosahedron() -> tuple[np.ndarray, np.ndarray]:
    """Return (12,3) unit-sphere vertices and (20,3) face indices (outward CCW)."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    r = np.sqrt(1.0 + phi ** 2)
    # 12 vertices
    v = np.array([
        [ 0,  1,  phi], [ 0, -1,  phi], [ 0,  1, -phi], [ 0, -1, -phi],
        [ 1,  phi,  0], [-1,  phi,  0], [ 1, -phi,  0], [-1, -phi,  0],
        [ phi,  0,  1], [ phi,  0, -1], [-phi,  0,  1], [-phi,  0, -1],
    ], dtype=float) / r

    # 20 faces (CCW when viewed from outside) – standard icosahedral triangulation
    f = np.array([
        [ 0,  1,  8], [ 0,  8,  4], [ 0,  4,  5], [ 0,  5, 10], [ 0, 10,  1],
        [ 1, 10,  7], [ 1,  7,  6], [ 1,  6,  8], [ 8,  6,  9], [ 8,  9,  4],
        [ 4,  9,  2], [ 4,  2,  5], [ 5,  2, 11], [ 5, 11, 10], [10, 11,  7],
        [ 3,  9,  6], [ 3,  6,  7], [ 3,  7, 11], [ 3, 11,  2], [ 3,  2,  9],
    ], dtype=int)
    return v, f


def _rotation_from_triangle_to_triangle(
    src: np.ndarray, dst: np.ndarray
) -> np.ndarray:
    """
    Return 3×3 rotation R (det +1) such that R @ src[i] ≈ dst[i] for i=0,1,2.
    src and dst are (3,3) arrays whose rows are unit-sphere points.
    Tries all 3 cyclic permutations of dst rows until det > 0.
    """
    P = src.T          # shape (3,3): column i = src[i]
    for perm in [(0,1,2), (1,2,0), (2,0,1)]:
        Q = dst[list(perm), :].T
        R = Q @ np.linalg.inv(P)
        if np.linalg.det(R) > 0.5:   # should be ≈ +1
            return R
    # fallback: use SVD to find nearest proper rotation
    U, _, Vt = np.linalg.svd(dst.T @ src)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def _corner_indices(K: int) -> tuple[int, int, int]:
    """
    Return patch vertex indices for the 3 icosahedral face corners.
    The C++ scan (x=0..K, y=0..K-x) assigns site indices sequentially.
     corner A = (x=0, y=0)  → index 0
     corner B = (x=K, y=0)  → index K
     corner C = (x=0, y=K)  → last site in the scan
    """
    k1 = K + 1
    idx = 0
    for xy in range(k1 * k1):
        x = xy % k1
        y = xy // k1
        if (x + y) == 0:
            iA = idx
        if x == K and y == 0:
            iB = idx
        if x == 0 and y == K:
            iC = idx
        if (x + y) <= K:
            idx += 1
    return iA, iB, iC


def build_full_sphere(
    patch_verts: np.ndarray,
    patch_tris: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Tile a single icosahedral face patch across all 20 icosahedral faces to
    produce the full closed spherical mesh.

    patch_verts : (N, 3) optimised vertex positions for one face
    patch_tris  : (F, 3) triangle connectivity (0-based, referring to patch)
    returns     : (all_verts, all_tris, seam_mask)
                  seam_mask is a bool array of length len(all_verts); True means
                  the vertex sits on an icosahedral face boundary (shared between
                  two or more patches).
    """
    n_patch = len(patch_verts)
    K = int(round((-3 + np.sqrt(9 + 8 * (n_patch - 1))) / 2))
    assert (K + 1) * (K + 2) // 2 == n_patch, f"Unexpected patch size {n_patch} for K={K}"

    # Corner indices within the patch
    iA, iB, iC = _corner_indices(K)
    src_corners = patch_verts[[iA, iB, iC]]
    radius = float(np.mean(np.linalg.norm(patch_verts, axis=1)))
    src_unit = src_corners / radius   # normalise to unit sphere

    # Standard icosahedron faces
    ico_verts, ico_faces = _standard_icosahedron()

    coord_map: dict[tuple, int] = {}   # rounded coord → global index
    all_verts: list[np.ndarray] = []
    all_tris:  list[np.ndarray] = []
    seam_set: set[int] = set()         # global indices of seam vertices
    prec = 6   # decimal places for coordinate matching

    for fi, face_idx in enumerate(ico_faces):
        dst_unit = ico_verts[face_idx]  # shape (3,3), rows = corners

        # Rotation R maps patch (unit sphere) → this ico face (unit sphere)
        R = _rotation_from_triangle_to_triangle(src_unit, dst_unit)

        # Transform all patch vertices
        rotated = (R @ patch_verts.T).T    # shape (N_patch, 3)

        # Register vertices, merging duplicates
        local_to_global: dict[int, int] = {}
        for li, pos in enumerate(rotated):
            key = tuple(np.round(pos, prec))
            if key not in coord_map:
                coord_map[key] = len(all_verts)
                all_verts.append(pos)
            else:
                # Vertex already registered from a previous face → seam vertex
                seam_set.add(coord_map[key])
            local_to_global[li] = coord_map[key]

        # Remap triangles to global indices
        for tri in patch_tris:
            all_tris.append([local_to_global[int(v)] for v in tri])

    all_verts_arr = np.array(all_verts, dtype=float)
    seam_mask = np.zeros(len(all_verts_arr), dtype=bool)
    for idx in seam_set:
        seam_mask[idx] = True

    return all_verts_arr, np.array(all_tris, dtype=int), seam_mask


# ---------------------------------------------------------------------------
# 2.  Sphere fitting & projection
# ---------------------------------------------------------------------------

def fit_sphere(pts: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares sphere fit.  Returns (centre_xyz, radius)."""
    p = pts.astype(float)
    # Linearise x²+y²+z² + Ax+By+Cz+D = 0
    A = np.column_stack([p, np.ones(len(p))])
    b = -(p[:, 0]**2 + p[:, 1]**2 + p[:, 2]**2)
    coeff, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, cz = -coeff[0]/2, -coeff[1]/2, -coeff[2]/2
    centre = np.array([cx, cy, cz])
    r = float(np.sqrt(max(cx**2 + cy**2 + cz**2 - coeff[3], 0.0)))
    return centre, r


def project_to_sphere(pts: np.ndarray, centre: np.ndarray, r: float) -> np.ndarray:
    """Radially project pts onto the sphere."""
    v = pts - centre
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.where(norms < 1e-14, 1.0, norms)
    return centre + v / norms * r


# ---------------------------------------------------------------------------
# 3.  Circumcenters
# ---------------------------------------------------------------------------

def circumcenters(verts: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """3-D circumcenter of every triangle (centroid fallback for degenerates)."""
    a = verts[tris[:, 0]]
    b = verts[tris[:, 1]]
    c = verts[tris[:, 2]]
    ab = b - a
    ac = c - a
    n  = np.cross(ab, ac)                          # (F,3) face normals
    denom = 2.0 * np.einsum('fi,fi->f', n, n)      # (F,)

    # Vectorised circumcenter formula
    term1 = (np.einsum('fi,fi->f', ac, ac)[:, None] *
             np.cross(n, ab))
    term2 = (np.einsum('fi,fi->f', ab, ab)[:, None] *
             np.cross(ac, n))
    # denom is zero for degenerate triangles → fall back to centroid
    safe = np.where(denom > 1e-16, denom, 1.0)[:, None]
    cc   = a + (term1 + term2) / safe
    # replace degenerate with centroid
    degenerate = (denom <= 1e-16)
    cc[degenerate] = ((a + b + c) / 3.0)[degenerate]
    return cc


# ---------------------------------------------------------------------------
# 4.  Dual mesh construction
# ---------------------------------------------------------------------------

def _sort_cyclic(pts: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """Sort 3-D pts cyclically around 'centre' (which lies on the sphere)."""
    if len(pts) <= 2:
        return pts
    # Use centre as the "normal" direction (outward normal on unit sphere)
    n = centre / (np.linalg.norm(centre) + 1e-14)
    # Build a local 2-D frame
    ref = pts[0] - centre
    ref = ref - n * np.dot(ref, n)
    ref_norm = np.linalg.norm(ref)
    if ref_norm < 1e-14:
        # fall back: pick any direction not parallel to n
        ref = np.array([1., 0., 0.])
        if abs(np.dot(ref, n)) > 0.9:
            ref = np.array([0., 1., 0.])
        ref = ref - n * np.dot(ref, n)
        ref_norm = np.linalg.norm(ref)
    e0 = ref / ref_norm
    e1 = np.cross(n, e0)
    e1 /= np.linalg.norm(e1) + 1e-14
    rel = pts - centre
    x   = rel @ e0
    y   = rel @ e1
    return pts[np.argsort(np.arctan2(y, x))]


def _polygon_area(pts: np.ndarray) -> float:
    """Area of a near-planar 3-D polygon (SVD plane + shoelace)."""
    if len(pts) < 3:
        return 0.0
    c   = pts.mean(axis=0)
    rel = pts - c
    _, _, vh = np.linalg.svd(rel, full_matrices=False)
    x   = rel @ vh[0]
    y   = rel @ vh[1]
    x2  = np.roll(x, -1)
    y2  = np.roll(y, -1)
    return float(0.5 * abs((x * y2 - y * x2).sum()))


def build_dual(verts: np.ndarray,
               tris: np.ndarray,
               dual_verts: np.ndarray,
               include_boundary: bool = True
               ) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """
    Build dual polygons from projected circumcenters.

    Returns
    -------
    polys         : list of (K_i, 3) arrays – one per primal vertex
    areas         : (N,) float array
    primal_idx    : (N,) int array – which primal vertex each dual cell belongs to
    boundary_mask : (len(polys),) bool array – True if this dual cell is at a boundary vertex
    """
    n_verts = len(verts)

    # adjacency: vertex → list of incident triangle indices
    v2f: list[list[int]] = [[] for _ in range(n_verts)]
    edge_count: dict[tuple[int, int], int] = {}
    for fi, (i, j, k) in enumerate(tris):
        v2f[i].append(fi)
        v2f[j].append(fi)
        v2f[k].append(fi)
        for u, v in ((i, j), (j, k), (k, i)):
            key = (min(u, v), max(u, v))
            edge_count[key] = edge_count.get(key, 0) + 1

    # boundary vertex mask
    boundary = np.zeros(n_verts, dtype=bool)
    for (u, v), cnt in edge_count.items():
        if cnt == 1:
            boundary[u] = True
            boundary[v] = True

    polys         : list[np.ndarray] = []
    areas         : list[float]      = []
    primal_idx    : list[int]        = []
    boundary_list : list[bool]       = []

    for vi, faces in enumerate(v2f):
        if not include_boundary and boundary[vi]:
            continue
        if len(faces) < 3:
            continue
        pts     = dual_verts[np.array(faces, dtype=int)]
        ordered = _sort_cyclic(pts, verts[vi])
        polys.append(ordered)
        areas.append(_polygon_area(ordered))
        primal_idx.append(vi)
        boundary_list.append(bool(boundary[vi]))

    return polys, np.array(areas, dtype=float), np.array(primal_idx, dtype=int), np.array(boundary_list, dtype=bool)


# ---------------------------------------------------------------------------
# 5.  Primal triangle areas
# ---------------------------------------------------------------------------

def primal_areas(verts: np.ndarray, tris: np.ndarray) -> np.ndarray:
    a = verts[tris[:, 0]]
    b = verts[tris[:, 1]]
    c = verts[tris[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def primal_triangle_angles(verts: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Return interior triangle angles (F,3) in radians."""
    a = verts[tris[:, 0]]
    b = verts[tris[:, 1]]
    c = verts[tris[:, 2]]

    ab = b - a
    ac = c - a
    ba = a - b
    bc = c - b
    ca = a - c
    cb = b - c

    def _angle(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        den = np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)
        den = np.where(den < 1e-14, 1.0, den)
        cosang = np.einsum("ij,ij->i", u, v) / den
        return np.arccos(np.clip(cosang, -1.0, 1.0))

    ang_a = _angle(ab, ac)
    ang_b = _angle(ba, bc)
    ang_c = _angle(ca, cb)
    return np.column_stack([ang_a, ang_b, ang_c])


# ---------------------------------------------------------------------------
# 6.  Plotting helpers
# ---------------------------------------------------------------------------

def _set_equal_axes(ax, pts: np.ndarray) -> None:
    lo, hi = pts.min(0), pts.max(0)
    mid    = (lo + hi) / 2
    r      = max((hi - lo).max() / 2, 1e-6)
    ax.set_xlim(mid[0] - r, mid[0] + r)
    ax.set_ylim(mid[1] - r, mid[1] + r)
    ax.set_zlim(mid[2] - r, mid[2] + r)


def plot_spherical_mesh(verts: np.ndarray, tris: np.ndarray,
                        areas: np.ndarray | None = None,
                        cmap: str = "plasma",
                        title: str = "Spherical primal mesh") -> plt.Figure:
    fig = plt.figure(figsize=(9, 8))
    ax  = fig.add_subplot(111, projection="3d")
    polys = verts[tris]
    coll  = Poly3DCollection(polys, alpha=0.92, edgecolor=(0.1, 0.1, 0.1, 0.5), linewidth=0.2)
    if areas is not None and len(areas) == len(tris):
        a_min, a_max = areas.min(), areas.max()
        if abs(a_max - a_min) < 1e-14:
            a_max = a_min + 1e-14
        norm  = mcolors.Normalize(vmin=a_min, vmax=a_max)
        cm    = plt.get_cmap(cmap)
        coll.set_facecolor(cm(norm(areas)))
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cm)
        sm.set_array(areas)
        fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.04, label="Triangle area")
    else:
        coll.set_facecolor((0.38, 0.66, 0.87, 0.8))
    ax.add_collection3d(coll)
    _set_equal_axes(ax, verts)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(title)
    return fig


def plot_dual_heatmap(polys: list[np.ndarray],
                      areas: np.ndarray,
                      sphere_centre: np.ndarray,
                      sphere_radius: float,
                      show_sphere: bool = True,
                      cmap: str = "viridis",
                      title: str = "Spherical dual mesh – face area") -> plt.Figure:
    fig = plt.figure(figsize=(9, 8))
    ax  = fig.add_subplot(111, projection="3d")

    if show_sphere:
        u = np.linspace(0, 2*np.pi, 64)
        v = np.linspace(0, np.pi,   32)
        xs = sphere_centre[0] + sphere_radius * np.outer(np.cos(u), np.sin(v))
        ys = sphere_centre[1] + sphere_radius * np.outer(np.sin(u), np.sin(v))
        zs = sphere_centre[2] + sphere_radius * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_surface(xs, ys, zs, color="#b8d4e8", alpha=0.08,
                        linewidth=0, antialiased=True)

    a_min, a_max = areas.min(), areas.max()
    if abs(a_max - a_min) < 1e-14:
        a_max = a_min + 1e-14
    norm   = mcolors.Normalize(vmin=a_min, vmax=a_max)
    cm     = plt.get_cmap(cmap)
    fcolors = cm(norm(areas))

    dcoll = Poly3DCollection(polys,
                             edgecolor=(0.1, 0.1, 0.1, 0.6),
                             linewidth=0.3, alpha=0.95)
    dcoll.set_facecolor(fcolors)
    ax.add_collection3d(dcoll)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cm)
    sm.set_array(areas)
    fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.04, label="Dual face area")

    all_pts = np.vstack(polys)
    _set_equal_axes(ax, all_pts)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(title)
    return fig


def plot_histograms(tri_areas: np.ndarray,
                    dual_areas: np.ndarray,
                    bins: int = 40,
                    title: str = "Area histograms") -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.hist(tri_areas, bins=bins, color="#4e79a7", edgecolor="black", alpha=0.85)
    ax1.set_title("Primal triangle areas")
    ax1.set_xlabel("Area")
    ax1.set_ylabel("Count")

    ax2.hist(dual_areas, bins=bins, color="#e15759", edgecolor="black", alpha=0.85)
    ax2.set_title("Dual face areas")
    ax2.set_xlabel("Area")
    ax2.set_ylabel("Count")

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_triangle_angle_scatter(angles_rad: np.ndarray,
                                use_degrees: bool = False,
                                centered_at_equilateral: bool = False,
                                title: str = "Triangle angle scatter") -> plt.Figure:
    """
    2D scatter of two triangle angles (x, y) with the third angle in colour.
    Angles are sorted per triangle so each point is permutation-invariant.
    """
    ang = np.sort(angles_rad, axis=1)
    if use_degrees:
        unit = "deg"
        conv = 180.0 / np.pi
        limit = 180.0
    else:
        unit = "rad"
        conv = 1.0
        limit = np.pi

    if centered_at_equilateral:
        ang = ang - (np.pi / 3.0)

    a1 = ang[:, 0] * conv
    a2 = ang[:, 1] * conv
    a3 = ang[:, 2] * conv

    fig, ax = plt.subplots(figsize=(7.5, 6.8))
    if centered_at_equilateral:
        ax.scatter(a1, a2, color="#2a6fbb", s=18, alpha=0.9, linewidths=0)
    else:
        sc = ax.scatter(a1, a2, c=a3, cmap="viridis", s=16, alpha=0.85, linewidths=0)

    if centered_at_equilateral:
        # In delta variables, delta_1 + delta_2 + delta_3 = 0.
        ax.axhline(0.0, color="0.4", lw=0.8, alpha=0.6)
        ax.axvline(0.0, color="0.4", lw=0.8, alpha=0.6)

        # Use data-driven limits so the delta cloud is not visually tiny.
        xy = np.concatenate([a1, a2])
        data_min = float(np.min(xy))
        data_max = float(np.max(xy))
        span = max(data_max - data_min, 1e-6)
        pad = max(0.35 * span, 1.0 if use_degrees else 0.02)
        lo = data_min - pad
        hi = data_max + pad
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f"$\\delta_1=\\alpha_1-\\pi/3$ ({unit})")
        ax.set_ylabel(f"$\\delta_2=\\alpha_2-\\pi/3$ ({unit})")
    else:
        boundary = np.linspace(0.0, limit, 400)
        ax.plot(boundary, limit - boundary, "k--", lw=1.0, alpha=0.8)
        ax.set_xlim(0.0, limit)
        ax.set_ylim(0.0, limit)
        ax.set_xlabel(f"$\\alpha_1$ ({unit})")
        ax.set_ylabel(f"$\\alpha_2$ ({unit})")
        cbar_label = f"$\\alpha_3$ ({unit})"

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)

    if not centered_at_equilateral:
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(cbar_label)

    fig.tight_layout()
    return fig


def plot_four_panel(
    verts: np.ndarray,
    tris: np.ndarray,
    tareas: np.ndarray,
    polys: list[np.ndarray],
    dareas: np.ndarray,
    sphere_centre: np.ndarray,
    sphere_radius: float,
    bins: int = 40,
    cmap: str = "viridis",
    title: str = "",
    dual_boundary_mask: np.ndarray | None = None,
    tri_boundary_mask: np.ndarray | None = None,
) -> plt.Figure:
    """3-row panel: primal/dual heatmaps | histograms | bin tables.

    If dual_boundary_mask / tri_boundary_mask are provided (bool arrays),
    boundary elements are rendered in gray and excluded from the colour scale
    and histograms (interior-only mode).
    """
    from matplotlib.gridspec import GridSpec

    # Interior-only mode: restrict histograms/tables to non-boundary elements
    _tareas_plot = tareas if tri_boundary_mask is None else tareas[~tri_boundary_mask]
    _dareas_plot = dareas if dual_boundary_mask is None else dareas[~dual_boundary_mask]

    # Pre-compute non-zero histogram bins for the table row
    p_counts, p_edges = np.histogram(_tareas_plot, bins=bins)
    d_counts, d_edges = np.histogram(_dareas_plot, bins=bins)
    p_mask = p_counts > 0
    d_mask = d_counts > 0
    p_table_data = [
        [str(i),
         f"{p_edges[i]:.6e}", f"{p_edges[i+1]:.6e}",
         f"{0.5*(p_edges[i]+p_edges[i+1]):.6e}",
         str(int(p_counts[i]))]
        for i in np.where(p_mask)[0]
    ]
    d_table_data = [
        [str(i),
         f"{d_edges[i]:.6e}", f"{d_edges[i+1]:.6e}",
         f"{0.5*(d_edges[i]+d_edges[i+1]):.6e}",
         str(int(d_counts[i]))]
        for i in np.where(d_mask)[0]
    ]
    col_labels = ["bin", "low", "high", "centre", "count"]

    n_table_rows = max(len(p_table_data), len(d_table_data))
    # Estimate height of table row (~0.40 inches per row, min 3 inches)
    table_h = max(3.0, n_table_rows * 0.40)

    fig = plt.figure(figsize=(18, 14 + table_h))
    gs  = GridSpec(3, 2, figure=fig,
                   height_ratios=[1, 0.8, table_h / 14],
                   hspace=0.40, wspace=0.3)

    # ── top-left: primal mesh coloured by triangle area ──────────────────
    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    if tri_boundary_mask is not None:
        interior_t = ~tri_boundary_mask
    else:
        interior_t = np.ones(len(tareas), dtype=bool)
    polys_p = verts[tris[interior_t]]
    _tareas_int = tareas[interior_t]
    a_min, a_max = _tareas_int.min(), _tareas_int.max()
    if abs(a_max - a_min) < 1e-14:
        a_max = a_min + 1e-14
    norm_p = mcolors.Normalize(vmin=a_min, vmax=a_max)
    cm_p   = plt.get_cmap("plasma")
    coll_p = Poly3DCollection(polys_p, alpha=0.92,
                              edgecolor=(0.1, 0.1, 0.1, 0.4), linewidth=0.2)
    coll_p.set_facecolor(cm_p(norm_p(_tareas_int)))
    ax1.add_collection3d(coll_p)
    _set_equal_axes(ax1, verts)
    ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("z")
    ax1.set_title("Primal mesh – triangle area")
    sm_p = plt.cm.ScalarMappable(norm=norm_p, cmap=cm_p)
    sm_p.set_array(_tareas_int)
    fig.colorbar(sm_p, ax=ax1, fraction=0.025, pad=0.04, label="Triangle area")

    # ── top-right: dual mesh coloured by face area ────────────────────────
    ax2 = fig.add_subplot(gs[0, 1], projection="3d")
    u = np.linspace(0, 2*np.pi, 48)
    v = np.linspace(0, np.pi,   24)
    xs = sphere_centre[0] + sphere_radius * np.outer(np.cos(u), np.sin(v))
    ys = sphere_centre[1] + sphere_radius * np.outer(np.sin(u), np.sin(v))
    zs = sphere_centre[2] + sphere_radius * np.outer(np.ones_like(u), np.cos(v))
    ax2.plot_surface(xs, ys, zs, color="#b8d4e8", alpha=0.06,
                     linewidth=0, antialiased=True)
    if dual_boundary_mask is not None:
        interior_d = ~dual_boundary_mask
    else:
        interior_d = np.ones(len(dareas), dtype=bool)
    _polys_int = [p for p, m in zip(polys, interior_d) if m]
    _dareas_int = dareas[interior_d]
    d_min, d_max = _dareas_int.min(), _dareas_int.max()
    if abs(d_max - d_min) < 1e-14:
        d_max = d_min + 1e-14
    norm_d = mcolors.Normalize(vmin=d_min, vmax=d_max)
    cm_d   = plt.get_cmap(cmap)
    dcoll  = Poly3DCollection(_polys_int, edgecolor=(0.1, 0.1, 0.1, 0.5),
                              linewidth=0.25, alpha=0.95)
    dcoll.set_facecolor(cm_d(norm_d(_dareas_int)))
    ax2.add_collection3d(dcoll)
    _set_equal_axes(ax2, np.vstack(_polys_int) if _polys_int else np.vstack(polys))
    ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_zlabel("z")
    ax2.set_title("Dual mesh – face area")
    sm_d = plt.cm.ScalarMappable(norm=norm_d, cmap=cm_d)
    sm_d.set_array(_dareas_int)
    fig.colorbar(sm_d, ax=ax2, fraction=0.025, pad=0.04, label="Dual face area")

    # ── bottom-left: primal triangle area histogram ───────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    hist_title_suffix = " (interior only)" if tri_boundary_mask is not None else ""
    ax3.hist(_tareas_plot, bins=bins, color="#4e79a7", edgecolor="black", alpha=0.85)
    ax3.set_title("Primal triangle area histogram" + hist_title_suffix)
    ax3.set_xlabel("Area")
    ax3.set_ylabel("Count")

    # ── bottom-right: dual face area histogram ────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(_dareas_plot, bins=bins, color="#e15759", edgecolor="black", alpha=0.85)
    ax4.set_title("Dual face area histogram" + hist_title_suffix)
    ax4.set_xlabel("Area")
    ax4.set_ylabel("Count")

    # ── table row: non-zero bin tables ────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.axis("off")
    ax5.set_title("Primal – non-zero bins", fontsize=9, pad=4)
    if p_table_data:
        t5 = ax5.table(cellText=p_table_data, colLabels=col_labels,
                       loc="upper center", cellLoc="center")
        t5.auto_set_font_size(False)
        t5.set_fontsize(11)
        t5.scale(1, 1.8)
        t5.auto_set_column_width(list(range(len(col_labels))))

    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis("off")
    ax6.set_title("Dual – non-zero bins", fontsize=9, pad=4)
    if d_table_data:
        t6 = ax6.table(cellText=d_table_data, colLabels=col_labels,
                       loc="upper center", cellLoc="center")
        t6.auto_set_font_size(False)
        t6.set_fontsize(11)
        t6.scale(1, 1.8)
        t6.auto_set_column_width(list(range(len(col_labels))))

    if title:
        fig.suptitle(title, fontsize=14, y=1.01)
    return fig


# ---------------------------------------------------------------------------
# 7.  Main
# ---------------------------------------------------------------------------

def _run_one_k(rvec_path: Path, tri_path: Path, args) -> "plt.Figure | None":
    """Run the full pipeline for a single K and return the 4-panel figure."""
    import numpy as _np

    print(f"\n{'='*60}")
    print(f"Loading patch vertices from {rvec_path} …")
    patch_verts = load_vertices(rvec_path)
    n_patch = len(patch_verts)
    K = int(round((-3 + np.sqrt(9 + 8 * (n_patch - 1))) / 2))
    print(f"  {n_patch} vertices  →  refinement level K = {K}")

    print(f"Loading patch triangles from {tri_path} …")
    patch_tris = load_triangles(tri_path, n_patch)
    print(f"  {len(patch_tris)} triangles  (one icosahedral face = K² = {K**2})")

    print("Tiling patch across all 20 icosahedral faces …")
    verts, tris, seam_mask = build_full_sphere(patch_verts, patch_tris)
    print(f"  Full sphere: {len(verts)} vertices  (expected {10*K*K+2}), "
          f"{len(tris)} triangles  (expected {20*K*K})")

    centre, radius = fit_sphere(verts)
    print(f"Fitted sphere: centre={centre.round(4)}, radius={radius:.6f}")
    verts = project_to_sphere(verts, centre, radius)

    cc = circumcenters(verts, tris)
    cc_sph = project_to_sphere(cc, centre, radius)

    polys, dareas, pidx, dual_bnd = build_dual(
        verts, tris, cc_sph,
        include_boundary=not args.exclude_boundary,
    )
    print(f"Built {len(polys)} dual polygons  "
          f"(area min={dareas.min():.5f}, max={dareas.max():.5f})")

    tareas = primal_areas(verts, tris)

    k_label = f"K={K}"
    # Triangle seam mask: any vertex is a seam vertex
    tri_bnd = seam_mask[tris].any(axis=1)
    # dual_bnd already uses seam_mask via build_dual's boundary detection;
    # replace with seam-based mask for primal_idx
    dual_seam = seam_mask[pidx]

    _tri_bnd_arg  = tri_bnd   if getattr(args, 'interior_only', False) else None
    _dual_bnd_arg = dual_seam if getattr(args, 'interior_only', False) else None

    fig = plot_four_panel(
        verts, tris, tareas,
        polys, dareas,
        centre, radius,
        bins=args.bins,
        cmap=args.cmap,
        title=f"Spherical mesh analysis – {k_label}",
        dual_boundary_mask=_dual_bnd_arg,
        tri_boundary_mask=_tri_bnd_arg,
    )
    return fig, tareas, dareas, verts, tris, polys, pidx, centre, radius, K, k_label


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rvec",  type=Path, default=None,
                   help="rvec_K.dat – vertex positions (index x y z ...)")
    p.add_argument("--tri",   type=Path, default=None,
                   help="Triangle_K.dat – triangle vertex indices")
    p.add_argument("--datadir", type=Path, default=None,
                   help="Directory containing rvec_*.dat / Triangle_*.dat for all K (batch mode)")
    p.add_argument("--pdf",     type=Path, default=None,
                   help="Output multi-page PDF (one page per K, batch mode)")
    p.add_argument("--panel",   type=Path, default=None,
                   help="Output path for a combined 4-panel PNG (single-K)")
    p.add_argument("--heatmap", type=Path, default=None,
                   help="Output path for the dual-mesh heatmap PNG (optional, separate)")
    p.add_argument("--hist",    type=Path, default=None,
                   help="Output path for the area histograms PNG (optional, separate)")
    p.add_argument("--angles",  type=Path, default=None,
                   help="Output path for triangle-angle scatter PNG (optional, separate)")
    p.add_argument("--angles-degrees", action="store_true",
                   help="Render triangle-angle scatter in degrees (default radians)")
    p.add_argument("--angles-delta", action="store_true",
                   help="Plot centered variables delta_i = theta_i - pi/3")
    p.add_argument("--primal",  type=Path, default=None,
                   help="Output path for the primal mesh PNG (optional, separate)")
    p.add_argument("--bins", type=int, default=40,
                   help="Histogram bin count (default 40)")
    p.add_argument("--cmap", default="viridis",
                   help="Colormap for the dual-face heatmap")
    p.add_argument("--exclude-boundary", action="store_true",
                   help="Exclude boundary dual cells (open partial polygons)")
    p.add_argument("--interior-only", action="store_true",
                   help="Colour only interior (non-boundary) primal and dual cells; boundary shown in gray")
    p.add_argument("--no-sphere", action="store_true",
                   help="Do not render the background sphere wireframe")
    p.add_argument("--dat",     type=Path, default=None,
                   help="Output path for a .dat file with histogram tables")
    p.add_argument("--show", action="store_true",
                   help="Show interactive windows in addition to saving")
    args = p.parse_args()

    # ------------------------------------------------------------------
    # Batch PDF mode
    # ------------------------------------------------------------------
    if args.pdf is not None:
        from matplotlib.backends.backend_pdf import PdfPages

        # Determine data directory
        if args.datadir is not None:
            datadir = args.datadir
        elif args.rvec is not None:
            datadir = args.rvec.parent
        else:
            p.error("Provide --datadir (or --rvec) so the script knows where to find rvec_*.dat files.")

        rvec_files = sorted(datadir.glob("rvec_*.dat"),
                            key=lambda f: int(f.stem.split("_")[1]))
        if not rvec_files:
            p.error(f"No rvec_*.dat files found in {datadir}")

        print(f"Found {len(rvec_files)} K values: "
              f"{[int(f.stem.split('_')[1]) for f in rvec_files]}")
        print(f"Writing PDF → {args.pdf}")

        with PdfPages(args.pdf) as pdf:
            for rvec_path in rvec_files:
                k_str = rvec_path.stem.split("_")[1]
                tri_path = datadir / f"Triangle_{k_str}.dat"
                if not tri_path.exists():
                    print(f"  WARNING: {tri_path} not found, skipping K={k_str}")
                    continue
                result = _run_one_k(rvec_path, tri_path, args)
                fig = result[0]
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                print(f"  Added K={k_str} to PDF")

        print(f"\nDone. PDF saved → {args.pdf}")
        return

    # ------------------------------------------------------------------
    # Single-K mode (original behaviour)
    # ------------------------------------------------------------------
    if args.rvec is None or args.tri is None:
        p.error("Provide --rvec and --tri for single-K mode, or --pdf (+ --datadir) for batch mode.")

    print(f"Loading patch vertices from {args.rvec} …")
    patch_verts = load_vertices(args.rvec)
    n_patch = len(patch_verts)
    K = int(round((-3 + np.sqrt(9 + 8 * (n_patch - 1))) / 2))
    print(f"  {n_patch} vertices  →  refinement level K = {K}")

    print(f"Loading patch triangles from {args.tri} …")
    patch_tris = load_triangles(args.tri, n_patch)
    print(f"  {len(patch_tris)} triangles  (one icosahedral face = K² = {K**2})")

    print("Tiling patch across all 20 icosahedral faces …")
    verts, tris, seam_mask = build_full_sphere(patch_verts, patch_tris)
    print(f"  Full sphere: {len(verts)} vertices  (expected {10*K*K+2}), "
          f"{len(tris)} triangles  (expected {20*K*K})")

    centre, radius = fit_sphere(verts)
    print(f"Fitted sphere: centre={centre.round(4)}, radius={radius:.6f}")
    verts = project_to_sphere(verts, centre, radius)
    print("  Snapped all vertices to sphere "
          f"(residual max={np.abs(np.linalg.norm(verts - centre, axis=1) - radius).max():.2e})")

    cc = circumcenters(verts, tris)
    print(f"Computed {len(cc)} circumcenters")
    cc_sph = project_to_sphere(cc, centre, radius)
    print("Projected circumcenters onto sphere")

    polys, dareas, pidx, dual_bnd = build_dual(
        verts, tris, cc_sph,
        include_boundary=not args.exclude_boundary,
    )
    print(f"Built {len(polys)} dual polygons")
    print(f"  min area  = {dareas.min():.6f}")
    print(f"  mean area = {dareas.mean():.6f}")
    print(f"  median    = {float(np.median(dareas)):.6f}")
    print(f"  max area  = {dareas.max():.6f}")
    print(f"  sum       = {dareas.sum():.6f}  (sphere surface area = {4*np.pi*radius**2:.6f})")

    tareas = primal_areas(verts, tris)
    print(f"Primal triangle areas: sum={tareas.sum():.6f}")

    k_label = args.rvec.stem.replace("rvec_", "K=")

    # Seam-based interior masks
    tri_bnd  = seam_mask[tris].any(axis=1)
    dual_seam = seam_mask[pidx]

    _tri_bnd_arg  = tri_bnd   if args.interior_only else None
    _dual_bnd_arg = dual_seam if args.interior_only else None

    # 4-panel combined figure
    if args.panel is not None or args.show:
        fig_panel = plot_four_panel(
            verts, tris, tareas,
            polys, dareas,
            centre, radius,
            bins=args.bins,
            cmap=args.cmap,
            title=f"Spherical mesh analysis – {k_label}",
            dual_boundary_mask=_dual_bnd_arg,
            tri_boundary_mask=_tri_bnd_arg,
        )
        if args.panel is not None:
            fig_panel.savefig(args.panel, dpi=150, bbox_inches="tight")
            print(f"Saved 4-panel figure → {args.panel}")
        if not args.show:
            plt.close(fig_panel)

    # Optional separate figures
    if args.primal is not None:
        fig_p = plot_spherical_mesh(verts, tris, areas=tareas,
                                     cmap="plasma",
                                     title=f"Spherical primal mesh – triangle area ({k_label})")
        fig_p.savefig(args.primal, dpi=150, bbox_inches="tight")
        print(f"Saved primal mesh → {args.primal}")
        if not args.show:
            plt.close(fig_p)

    if args.heatmap is not None:
        fig_h = plot_dual_heatmap(
            polys, dareas, centre, radius,
            show_sphere=not args.no_sphere,
            cmap=args.cmap,
            title=f"Spherical dual mesh – face area ({k_label})",
        )
        fig_h.savefig(args.heatmap, dpi=150, bbox_inches="tight")
        print(f"Saved heatmap → {args.heatmap}")
        if not args.show:
            plt.close(fig_h)

    if args.hist is not None:
        fig_hist = plot_histograms(
            tareas, dareas, bins=args.bins,
            title=f"Area histograms ({k_label})",
        )
        fig_hist.savefig(args.hist, dpi=150, bbox_inches="tight")
        print(f"Saved histograms → {args.hist}")
        if not args.show:
            plt.close(fig_hist)

    if args.angles is not None:
        angles = primal_triangle_angles(verts, tris)
        fig_angles = plot_triangle_angle_scatter(
            angles,
            use_degrees=args.angles_degrees,
            centered_at_equilateral=args.angles_delta,
            title=(f"Triangle delta-angle scatter ({k_label})"
                   if args.angles_delta else f"Triangle angle scatter ({k_label})"),
        )
        fig_angles.savefig(args.angles, dpi=180, bbox_inches="tight")
        print(f"Saved triangle-angle scatter → {args.angles}")
        if not args.show:
            plt.close(fig_angles)

    if args.show:
        plt.show()

    # ------------------------------------------------------------------
    # Write histogram .dat file
    # ------------------------------------------------------------------
    if args.dat is not None:
        n_bins = args.bins
        p_counts, p_edges = np.histogram(tareas, bins=n_bins)
        d_counts, d_edges = np.histogram(dareas, bins=n_bins)

        with open(args.dat, "w") as fh:
            fh.write(f"# Spherical mesh area histograms – {k_label}\n")
            fh.write(f"# Generated from {args.rvec.name} + {args.tri.name}\n")
            fh.write(f"# Full sphere: {len(verts)} vertices, {len(tris)} triangles\n")
            fh.write(f"# Bins: {n_bins}\n")
            fh.write("#\n")
            fh.write("# PRIMAL TRIANGLE AREA HISTOGRAM\n")
            fh.write("# columns: bin_index  bin_low  bin_high  bin_centre  count\n")
            for i, (lo, hi, cnt) in enumerate(zip(p_edges[:-1], p_edges[1:], p_counts)):
                if cnt == 0:
                    continue
                fh.write(f"{i:6d}  {lo:.15e}  {hi:.15e}  {0.5*(lo+hi):.15e}  {cnt:6d}\n")
            fh.write("#\n")
            fh.write("# DUAL CELL AREA HISTOGRAM\n")
            fh.write("# columns: bin_index  bin_low  bin_high  bin_centre  count\n")
            for i, (lo, hi, cnt) in enumerate(zip(d_edges[:-1], d_edges[1:], d_counts)):
                if cnt == 0:
                    continue
                fh.write(f"{i:6d}  {lo:.15e}  {hi:.15e}  {0.5*(lo+hi):.15e}  {cnt:6d}\n")

        print(f"Saved histogram data → {args.dat}")


if __name__ == "__main__":
    main()
