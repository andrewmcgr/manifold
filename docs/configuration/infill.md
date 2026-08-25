# Infill Patterns & Region Generation

Manifold provides distinct infill generators for interior sparse cavities (`sparse_infill_pattern`) and solid exposure layers (`solid_infill_pattern`).

---

## Infill Configuration Keys

```json
{
  "sparse_infill_pattern": "Cubic",
  "solid_infill_pattern": "AllWalls",
  "infill_density": 0.20,
  "infill_line_width": 0.40
}
```

| Key in JSON | Options | Default | Description |
|---|---|---|---|
| `sparse_infill_pattern` | `None`, `AllWalls`, `Concentric`, `Cubic` | `Cubic` | Infill pattern used across interior sparse regions. |
| `solid_infill_pattern` | `None`, `AllWalls`, `Concentric`, `Cubic` | `AllWalls` | Infill pattern used across 100% density solid exposure regions. |
| `infill_density` | Float (`0.0`..`1.0`) | `0.20` | Volume density fraction for sparse infill. |
| `infill_line_width` | `f64` | `0.40` | Extrusion line width for infill scanlines. |

---

## Pattern Descriptions

### 1. `Cubic` (3D Periodic Cubic Lattice Infill)
- Generates a **self-supporting 3D cubic/octahedral truss lattice** rotated such that its space diagonals align with the build axis ($45^\circ$ inclination to horizontal).
- **Non-Planar Reconstruction**: For each layer, intersecting scanlines shift across $Z$ and reproject onto the non-planar layer surface via Newton-style order field gradient refinement (`refine_point_onto_order_field`).
- **Continuous Boustrophedon Chaining**: Consecutive scanlines are grouped into connected components and chained into continuous alternating zigzag paths, eliminating 85% of travels and retractions.

### 2. `AllWalls` (Offset Wall Ring Infill)
- Repeatedly offsets the perimeter boundary inward by `wall_line_width` until the entire interior is packed with concentric perimeter loops.
- Provides high torsional rigidity and dense surface bonding.

### 3. `Concentric` (Concentric Ring Infill)
- Inward polygon offsetting scaled to the configured `infill_density`.

---

## Solid Exposure Boundary Computation

Solid top and bottom layers are computed via **2D polygon difference and union operations** (`polygon2d` in `i_overlay`):
1. Unsupported downward-facing exposures are detected: $\text{ExposedBelow}_k = \text{Layer}_k \setminus \text{Layer}_{k-1}$.
2. Upward-facing exposures are detected: $\text{ExposedAbove}_k = \text{Layer}_k \setminus \text{Layer}_{k+1}$.
3. Exposures propagate through $N_{\text{bottom}}$ and $N_{\text{top}}$ layers to guarantee watertight solid skins.
4. Sliver loops with area $< 0.25 \times d_{\text{nozzle}}^2$ are automatically filtered to prevent boolean memory explosion.
