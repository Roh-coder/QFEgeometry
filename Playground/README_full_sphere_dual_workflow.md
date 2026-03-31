# Full-Sphere Dual Mesh Workflow (From Triangle Data)

This workflow builds a dual mesh on a full sphere directly from triangle data, then colors dual faces by area and makes a dual-area histogram.

## Goal

Given triangle soup data (rows of 9 numbers = 3 xyz vertices), do the following:

1. Build the full primal sphere mesh from triangles in data.
2. Compute circumcenters of all primal triangular faces.
3. Fit a sphere from the primal vertices.
4. Radially project circumcenters onto the fitted sphere.
5. Build the dual mesh from those projected circumcenters.
6. Compute dual face areas.
7. Color dual faces by area (heatmap).
8. Plot histogram of dual face areas.

## Input Used

- `Mathematica/data.txt`

## Sanity Check (Full Sphere)

A closed sphere must have no boundary edges.

- Reconstruct unique vertices/triangles from `data.txt`.
- Count edge incidences.
- Verify all edges are shared by exactly 2 triangles.

For current data:

- vertices: `162`
- triangles: `320`
- boundary edges: `0`
- z-range: `[-1, 1]`

This confirms a full sphere mesh.

## Commands

### 1. Full-sphere dual face heatmap

```bash
python Playground/plot_simulation.py \
  --data-file Mathematica/data.txt \
  --dual --dual-only \
  --spherical-dual --show-sphere \
  --dual-face-heatmap \
  --output Playground/full_sphere_dual_face_area_heatmap.png
```

### 2. Full-sphere dual area histogram

```bash
python - <<'PY'
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from Playground.plot_simulation import (
    polygons_from_flat_rows,
    vertices_and_triangles_from_polygons,
    circumcenters_from_triangles,
    fit_sphere_from_points,
    project_points_to_sphere,
    dual_polygons_and_areas,
)

polys = polygons_from_flat_rows(Path('Mathematica/data.txt'))
vertices, triangles = vertices_and_triangles_from_polygons(polys)
centers = circumcenters_from_triangles(vertices, triangles)
center, radius = fit_sphere_from_points(vertices)
centers_sph = project_points_to_sphere(centers, center, radius)
_, dual_areas = dual_polygons_and_areas(vertices, triangles, centers_sph, include_boundary=True)

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(dual_areas, bins=40, color='#e15759', edgecolor='black', alpha=0.9)
ax.set_title('Full Sphere Dual Face Area Histogram')
ax.set_xlabel('Dual face area')
ax.set_ylabel('Count')
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig('Playground/full_sphere_dual_area_hist.png', dpi=180, bbox_inches='tight')
plt.close(fig)
print('Saved: Playground/full_sphere_dual_area_hist.png')
print('stats:', len(dual_areas), float(dual_areas.min()), float(np.median(dual_areas)), float(dual_areas.max()))
PY
```

## Outputs

- `Playground/full_sphere_dual_face_area_heatmap.png`
- `Playground/full_sphere_dual_area_hist.png`

## Notes

- Sphere is fit from the input vertices (least-squares), not assumed.
- Dual vertices are spherical projections of face circumcenters.
- For a closed sphere, boundary dual artifacts should not appear.
