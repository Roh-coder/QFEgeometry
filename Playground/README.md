# Playground

Python scripts for visualising and analysing the QFE spherical triangulation data.

---

## Scripts

### `plot_simulation.py`

Replicates the core functionality of `Mathematica/PlotSimulation.nb` in Python.

**Features**
- Load a triangulated surface from `rvec_K.dat` + `Triangle_K.dat`.
- Optionally load polygon soup from `Mathematica/data.txt` and overlay circumcenters from `circum.txt`.
- Build a dual mesh from triangle circumcenters (projected onto a fitted sphere).
- Render 3-D primal and dual meshes, coloured by face area.
- Export separate primal mesh, dual heatmap, and area histogram PNGs.

**Key functions** (importable by other scripts):
| Function | Purpose |
|---|---|
| `load_vertices_rvec` | Load `rvec_K.dat` → `(N,3)` vertex positions |
| `load_triangles` | Load `Triangle_K.dat` → 0-based `(F,3)` index array |
| `triangle_areas` | Compute per-triangle areas |
| `circumcenters_from_triangles` | 3-D circumcenter of every triangle |
| `dual_polygons_and_areas` | Build dual polygons and compute their areas |

---

### `spherical_dual.py`

Full spherical dual mesh pipeline.  Mirrors the Mathematica notebook workflow end-to-end, operating on a single icosahedral face patch and tiling it to a closed sphere.

**Pipeline steps**
1. Load the single-face patch (`rvec_K.dat`, `Triangle_K.dat`).  One face has `(K+1)(K+2)/2` optimised vertices.
2. Tile the patch across all 20 icosahedral faces by applying the appropriate rotation to each face, merging shared edge/corner vertices → full closed sphere mesh.  The tiling also records which vertices are *seam vertices* (shared between two or more patches).
3. Least-squares sphere fit + project every vertex exactly onto it.
4. Compute triangle circumcenters; project onto the same sphere → dual-mesh vertices.
5. For each primal vertex, collect circumcenters of its incident triangles, sort cyclically, and close the polygon.
6. Compute dual-face area via SVD best-fit plane + shoelace formula.
7. Render a 4-panel figure: primal heatmap | dual heatmap | area histograms | non-zero bin tables.

**CLI flags**
| Flag | Effect |
|---|---|
| `--rvec` | Single-K mode: path to `rvec_K.dat` |
| `--tri` | Single-K mode: path to `Triangle_K.dat` |
| `--panel` | Output path for 4-panel PNG |
| `--heatmap` | Output path for dual heatmap PNG (optional) |
| `--hist` | Output path for area histogram PNG (optional) |
| `--primal` | Output path for primal mesh PNG (optional) |
| `--dat` | Output path for histogram bin table `.dat` file |
| `--datadir` | Batch mode: directory of `rvec_*.dat` / `Triangle_*.dat` |
| `--pdf` | Batch mode: output multi-page PDF (one page per K) |
| `--bins` | Histogram bin count (default 40) |
| `--cmap` | Colormap for dual heatmap (default `viridis`) |
| `--exclude-boundary` | Exclude dual cells at boundary vertices |
| `--interior-only` | Colour and histogram **only interior** (non-seam) cells; seam/boundary polygons are omitted entirely |
| `--show` | Open interactive matplotlib windows |

**Examples**

```bash
# Single K panel
python Playground/spherical_dual.py \
    --rvec data/q5/rvec_16.dat \
    --tri  data/q5/Triangle_16.dat \
    --panel Playground/q5_k16_panel.png

# Interior-only panel (exclude icosahedral face seam vertices)
python Playground/spherical_dual.py \
    --rvec data/q5/rvec_16.dat \
    --tri  data/q5/Triangle_16.dat \
    --interior-only \
    --panel Playground/q5_k16_interior_panel.png

# Batch PDF (all K in a directory)
python Playground/spherical_dual.py \
    --datadir data/q5 \
    --pdf Playground/q5_all_k_panels.pdf
```

---

### `make_all_k_histograms.py`

Scans `data/q*/rvec_*.dat` and matches each file to its `Triangle_K.dat` counterpart.  Produces a single multi-panel figure with one panel per `(q, K)` pair, where each panel overlays the primal triangle area histogram and the dual cell area histogram.

**CLI flags**
| Flag | Effect |
|---|---|
| `--repo-root` | Repository root (default: `cwd`) |
| `--output` | Output path – PNG or PDF (default `Playground/all_k_area_histograms.png`) |
| `--bins` | Bins per histogram panel (default 32) |
| `--cols` | Panel columns (default 5) |
| `--qfilter` | Restrict to one `q` directory, e.g. `q5` |
| `--exclude-boundary-dual` | Exclude boundary dual cells from dual histograms |

**Example**

```bash
python Playground/make_all_k_histograms.py \
    --repo-root /workspaces/QFEgeometry \
    --qfilter q5 \
    --output Playground/q5_all_k.pdf \
    --cols 5
```

---

## Outputs

### Summary PDFs / multi-panel figures

| File | Contents |
|---|---|
| `q5_all_k.pdf` | Overlaid primal+dual area histograms for every K in `data/q5` |
| `q5_all_k_panels.pdf` | 4-panel figure (meshes + histograms + bin tables) assembled from individual K PNGs — one page per K |
| `all_k_area_histograms.png` | Same as `q5_all_k.pdf` but includes q3, q4, q5 in a single PNG grid |
| `all_k_area_histograms_no_boundary_dual.png` | Same grid with boundary dual cells excluded |

### Per-K 4-panel PNGs (`q5`)

One file per refinement level K.  Each panel shows:
- **Top-left** — primal triangulated sphere, coloured by triangle area
- **Top-right** — dual mesh, coloured by dual-cell area
- **Bottom-left** — primal triangle area histogram
- **Bottom-right** — dual face area histogram
- **Tables** — non-zero histogram bins for both primal and dual

| File | K |
|---|---|
| `q5_k4_panel.png` | 4 |
| `q5_k8_panel.png` | 8 |
| `q5_k10_panel.png` | 10 |
| `q5_k12_panel.png` | 12 |
| `q5_k16_panel.png` | 16 |
| `q5_k24_panel.png` | 24 |
| `q5_k32_panel.png` | 32 |
| `q5_k36_panel.png` | 36 |
| `q5_k48_panel.png` | 48 |
| `q5_k52_panel.png` | 52 |
| `q5_k64_panel.png` | 64 |
| `q5_k72_panel.png` | 72 |
| `q5_k84_panel.png` | 84 |
| `q5_k96_panel.png` | 96 |
| `q5_k128_panel.png` | 128 |

### Interior-only panels

| File | Description |
|---|---|
| `q5_k16_interior_panel.png` | K=16 panel with icosahedral face seam vertices excluded — only interior primal triangles and dual cells are coloured and histogrammed |

### Miscellaneous earlier outputs

| File | Description |
|---|---|
| `q5_k8_sph_primal.png` | Primal mesh heatmap, K=8 |
| `q5_k8_sph_heatmap.png` | Dual mesh heatmap, K=8 |
| `q5_k8_sph_hist.png` | Area histograms, K=8 |
| `q5_k8_areas.dat` | Histogram bin table for K=8 |
| `q5_k32_sph_primal.png` | Primal mesh heatmap, K=32 |
| `q5_k32_sph_heatmap.png` | Dual mesh heatmap, K=32 |
| `q5_k32_sph_hist.png` | Area histograms, K=32 |
| `q5_k32_sphere_dual_area_hist.png` | Full-sphere dual area histogram, K=32 |
| `q5_k32_sphere_dual_face_area_heatmap.png` | Full-sphere dual face heatmap, K=32 |
| `q5_mesh.png` | Plain primal mesh render |
| `q5_dual_mesh.png` | Plain dual mesh render |
| `q5_k16_dual_boundary_vertices.png` | Boundary vertex visualisation, K=16 |
| `q5_k16_spherical_dual_boundary_vertices.png` | Spherical dual boundary vertices, K=16 |
| `data_area_hist.png` | Area histogram from `Mathematica/data.txt` |
| `data_circum.png` | Circumcenter overlay from `Mathematica/data.txt` |
| `data_dual_mesh.png` | Dual mesh from `Mathematica/data.txt` |
| `data_sphere_dual_area_hist.png` | Spherical dual area histogram from `data.txt` |
| `data_sphere_dual_boundary_vertices.png` | Boundary vertices from `data.txt` |
| `data_sphere_dual_face_area_heatmap.png` | Spherical dual heatmap from `data.txt` |
| `full_sphere_dual_area_hist.png` | Full-sphere dual area histogram (early exploration) |
| `full_sphere_dual_face_area_heatmap.png` | Full-sphere dual heatmap (early exploration) |

---

## Interior-only mode — technical notes

The `--interior-only` flag in `spherical_dual.py` works as follows:

1. **Seam detection** — during `build_full_sphere`, any vertex that is registered from more than one icosahedral face patch is marked as a *seam vertex*.  These are vertices on the edges and corners of the icosahedron.
2. **Primal filter** — triangles where *any* vertex is a seam vertex are omitted from rendering and histograms.
3. **Dual filter** — dual cells whose primal vertex is a seam vertex are omitted from rendering and histograms.

This isolates the geometry strictly inside each icosahedral face, removing the stitching artefacts along face boundaries.
