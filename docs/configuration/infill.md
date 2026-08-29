# Infill Patterns & Region Generation

Manifold provides distinct infill generators for interior sparse cavities (`sparse_infill_pattern`) and solid exposure layers (`solid_infill_pattern`).

---

## Infill Configuration Keys

```json
{
  "sparse_infill_pattern": "Gyroid",
  "solid_infill_pattern": "AllWalls",
  "infill_density": 0.20,
  "infill_line_width": 0.40,
  "infill_angle_deg": 45.0
}
```

| Key in JSON | Options | Default | Description |
|---|---|---|---|
| `sparse_infill_pattern` | `Monotonic`, `Concentric`, `AllWalls`, `Cubic`, `Gyroid`, `SchwarzD`, `SchwarzP`, `None` | `Cubic` | Infill pattern used across interior sparse regions. |
| `solid_infill_pattern` | `Monotonic`, `Concentric`, `AllWalls`, `Cubic`, `Gyroid`, `SchwarzD`, `SchwarzP`, `None` | `AllWalls` | Infill pattern used across 100% density solid exposure regions. |
| `infill_density` | Float (`0.0`..`1.0`) | `0.20` | Volume density fraction for sparse infill. |
| `infill_line_width` | `f64` | `0.40` | Extrusion line width for infill scanlines. |
| `infill_angle_deg` | `f64` | `45.0` | Base scanline rotation angle for rectilinear/monotonic fills. |

---

## Pattern Descriptions

### 1. `Gyroid` (Triply Periodic Minimal Surface — TPMS)
- Generates a **continuous, self-supporting 3D minimal surface lattice** with isotropic stiffness and high fluid permeability:
  $$\sin\left(\frac{2\pi x}{L}\right) \cos\left(\frac{2\pi y}{L}\right) + \sin\left(\frac{2\pi y}{L}\right) \cos\left(\frac{2\pi z}{L}\right) + \sin\left(\frac{2\pi z}{L}\right) \cos\left(\frac{2\pi x}{L}\right) = 0$$
- **Non-Planar Adaptation**: Slice contours are evaluated by sampling the TPMS scalar field directly along the non-planar order field isosurfaces, guaranteeing continuous surface continuity across non-planar layers without slicing shear.
- **WGPU Compute Acceleration**: Evaluated with optional GPU-accelerated raymarching on NanoVDB grids.

### 2. `SchwarzD` (Schwarz Diamond TPMS Infill)
- A high-strength diamond minimal surface truss structure:
  $$\sin x \sin y \sin z + \sin x \cos y \cos z + \cos x \sin y \cos z + \cos x \cos y \sin z = 0$$
- Exceptional shear resistance and energy absorption under multi-axial load.

### 3. `SchwarzP` (Schwarz Primitive TPMS Infill)
- A highly open cubic minimal surface structure:
  $$\cos x + \cos y + \cos z = 0$$
- Creates orthogonal intersecting cylindrical channels ideal for lightweight parts and permeable filters.

### 4. `Cubic` (3D Periodic Cubic Lattice Infill)
- Generates a **self-supporting 3D cubic/octahedral truss lattice** rotated such that its space diagonals align with the build axis ($45^\circ$ inclination to horizontal).
- **Non-Planar Reconstruction**: For each layer, intersecting scanlines shift across $Z$ and reproject onto the non-planar layer surface via Newton-style order field gradient refinement (`refine_point_onto_order_field`).

### 5. `Monotonic` (Boustrophedon Rectilinear Scanlines)
- Continuous zig-zag scanlines alternating by $\pm\theta$ between adjacent layers.
- Chained with serpentine turnaround loops to minimize retractions.

### 6. `AllWalls` (Offset Wall Ring Infill)
- Repeatedly offsets the perimeter boundary inward by `wall_line_width` until the entire interior is packed with concentric perimeter loops.
- Always generates 100% fill density regardless of `infill_density`, making it ideal for solid skins and structural perimeters.

### 7. `Concentric` (Concentric Ring Infill)
- Inward polygon offsetting spaced according to the configured `infill_density`.

---

## Travel Optimization: Serpentine Bridges & Polyline Chaining

To minimize non-extruding travel moves and surface oozing, Manifold's infill planner groups scanline segments into connected components:
- Neighboring scanline endpoints are joined via **serpentine turnaround bridges** along the perimeter boundary when travel distance is within tolerance.
- Disconnected polylines are chained using nearest-neighbor TSP heuristics, eliminating over 80% of travel moves compared to naive scanline emission.

---

## Solid Exposure Boundary Computation

Solid top and bottom layers are computed via **2D polygon difference and union operations** (`polygon2d` in `i_overlay`):
1. Unsupported downward-facing exposures are detected: $\text{ExposedBelow}_k = \text{Layer}_k \setminus \text{Layer}_{k-1}$.
2. Upward-facing exposures are detected: $\text{ExposedAbove}_k = \text{Layer}_k \setminus \text{Layer}_{k+1}$.
3. Exposures propagate through $N_{\text{bottom}}$ and $N_{\text{top}}$ layers to guarantee watertight solid skins.
4. Sliver loops with area $< 0.25 \times d_{\text{nozzle}}^2$ are automatically filtered to prevent boolean memory explosion.
