# Manifold Slicer — Application Capabilities

**Manifold Slicer** (`Manifold2`) is a next-generation volumetric, non-planar, multi-axis additive manufacturing slicer built on a hybrid **C++20 (OpenCASCADE + OpenVDB)** geometry/field kernel and a **Rust (Bevy + egui)** interactive 3D frontend, toolpath engine, and Klipper G-code compiler.

Unlike conventional 2.5D planar slicers that intersect triangle meshes with flat horizontal $Z$ planes, Manifold Slicer represents parts as volumetric scalar and vector fields in **bed coordinates**, computes collision-free curved arrival-time manifolds $\tau(\mathbf{x})$, and generates true 3D conformal toolpaths with physics-based non-Newtonian polymer rheology compensation.

---

## 1. Target Hardware & Multi-Axis Kinematics

Manifold Slicer is architected from the ground up so that standard 3-axis planar assumptions ($\text{nozzle} = +Z$, $\text{gravity} = -Z$, constant layer thickness) are **never** hardcoded:

1. **6-Axis Robotic Arm Cells with FDM Toolheads**:
   - Positions and orients the nozzle in arbitrary 3D directions across space and time.
   - Queries the local right-handed orthonormal **`NozzleFrame`** $(\hat{\mathbf{x}}_{\text{nozzle}}, \hat{\mathbf{y}}_{\text{nozzle}}, \hat{\mathbf{z}}_{\text{nozzle}})$ at every toolpath vertex, where $\hat{\mathbf{z}}_{\text{nozzle}}$ points opposite to nozzle discharge, $\hat{\mathbf{x}}_{\text{nozzle}}$ points along toolhead travel, and $\hat{\mathbf{y}}_{\text{nozzle}} = \hat{\mathbf{z}}_{\text{nozzle}} \times \hat{\mathbf{x}}_{\text{nozzle}}$.
2. **5-Axis Motion Systems with Tilting Build Plates**:
   - Models bed tilt (`Rx`, `Ry`) relative to the world frame so gravitational acceleration $\hat{\mathbf{g}}_{\text{bed}} = R_{\text{bed}}^{-1}(0, 0, -1)$ rotates dynamically in bed coordinates.
   - All overhang angles, bridge spans, wave overhangs, and support projections query $\hat{\mathbf{g}}_{\text{bed}}$ rather than inspecting raw $Z$ components.
3. **3-Axis CoreXY, Cartesian & Circular Delta Printers**:
   - Supports non-planar conformal slicing within the physical clearance cone of standard 3-axis toolheads (`has_rotational_axes = false` or bounded `max_angle_from_bed_normal_deg`).
   - Supports both Rectangular (front-left or center G-code origin) and Circular (center G-code origin) build plates.
4. **Multi-Tool Extrusion Systems**:
   - Maintains a multi-tool registry (`T0`, `T1`, …) where each tool carries its own nozzle orifice diameter, target hotend temperature, rotational axis capability, and piecewise-linear revolved XZ clearance profile (`clearance_profile_xz`) modeling the nozzle tip land, brass cone, heater block, and cooling shroud.

---

## 2. Native CAD Ingestion & Multi-Plate Project Ecosystem

- **Direct STEP B-Rep CAD Import (`.step` / `.stp`)**:
  - Powered by **OpenCASCADE (OCCT)** (`STEPControl_Reader` + `BRepMesh_IncrementalMesh`).
  - Preserves exact topological `TopoDS_Face` boundaries as selectable surface patches (`SurfacePatch`) so you can click individual analytical CAD faces to assign conformal or orthogonal layer orientation hints.
- **Mesh Import with Automatic Patch Segmentation (`.stl` / `.3mf`)**:
  - Imports ASCII and binary STL files as well as 3MF mesh objects, automatically clustering contiguous coplanar triangles into selectable surface patches and computing 3D convex hulls for instant **Place on Face** orientation.
- **Multi-Plate `.3mf` Project Compatibility**:
  - Native multi-plate workspace (`Plate 1`, `Plate 2`, …) with per-plate object assignment, plate duplication, object cloning, and Archimedean spiral auto-arrangement.
  - Imports multi-plate `.3mf` project archives from **OrcaSlicer**, **Bambu Studio**, and **PrusaSlicer** (parsing `Metadata/model_settings.config` and `3D/3dmodel.model` build items/components) as well as native Manifold `.3mf` projects (`/Metadata/manifold2_config.toml` and `/Metadata/manifold2_project.toml`).

---

## 3. Volumetric OpenVDB Engine & Constrained Eikonal Slicing

Manifold Slicer converts build-plate geometry into sparse volumetric grids using **OpenVDB**:

- **Dual-Grid Multi-Resolution Architecture**:
  - Eliminates the $\mathcal{O}(L^3 / \Delta x_{\text{fine}}^3)$ memory and compute bottleneck on large build volumes by combining a **coarse full-interior core grid** ($\Delta x_{\text{coarse}} = R \cdot \Delta x_{\text{fine}}$, $64\times$ fewer interior voxels at $R=4$) with a **fine boundary-shell narrow-band grid** ($\Delta x_{\text{fine}}$) across the outer skin ($-T_{\text{shell}} \le \phi_{\text{sdf}}(\mathbf{x}) \le 0$).
  - Automatically scales $\Delta x_{\text{fine}} = \text{clamp}(0.375\,d_{\text{nozzle}} + 0.50\,h_{\text{target}},\, 0.08,\, 2.00)$ from the active nozzle diameter and target layer height.
- **Four Coupled Volumetric Fields**:
  1. **Signed Distance Field (`"sdf"`, $\phi_{\text{sdf}}(\mathbf{x})$)**: Exact Euclidean distance to the part boundary ($\phi_{\text{sdf}} < 0$ inside, $\phi_{\text{sdf}} = 0$ on the surface).
  2. **Gravity-Aware Overhang Field (`"overhang"`, $o(\mathbf{x})$)**: Evaluates $o(\mathbf{x}) = -\nabla\phi_{\text{sdf}}(\mathbf{x}) \cdot \hat{\mathbf{g}}_{\text{bed}}$ throughout the shell and interior.
  3. **Coupled 5/6-Axis Tool Orientation Field ($\hat{\mathbf{z}}_{\text{nozzle}}(\mathbf{x})$)**: Smoothly steers the local nozzle axis within `max_angle_from_bed_normal_deg` to support steep overhangs and respect user orientation hints.
  4. **Constrained Eikonal Order Field (`"eikonal"`, $\tau(\mathbf{x})$)**: Solves $\|\nabla\tau(\mathbf{x})\| = 1/F(\mathbf{x})$ subject to causal buildability from the build plate (and optional support contact seeds) and hard toolhead clearance-cone slope clamping ($\angle(\nabla\tau, \hat{\mathbf{z}}_{\text{nozzle}}) \le \theta_{\max}$).
- **Automatic Overhang Conformal vs. Orthogonal Surface Bias**:
  - Automatically steers boundary layer normals toward **orthogonal** ($\nabla\tau \perp \hat{\mathbf{n}}_{\text{surf}}$) on walls and moderate overhangs (default $-30^\circ$ supported to $+60^\circ$ overhang) so layers tilt to support overhanging faces at $90^\circ$, and smoothly transitions via $C^1$ cubic Hermite blending to **conformal** ($\nabla\tau \parallel \pm\hat{\mathbf{n}}_{\text{surf}}$) on shallow domes/roofs ($< -30^\circ$) to eliminate stair-stepping and on steep ceilings ($> +60^\circ$).
- **Interactive Surface & 3D Point Orientation Hints**:
  - Users can click any B-Rep face or planar mesh patch to request **Conformal Layers** (`Normal ⊥ Surface`) or **Orthogonal Layers** (`Normal ∥ Surface`), or drop **3D Point Hints** with Wendland $C^2$ radial falloff inside bulky regions.
- **Reeb-Graph Multi-Branch Bidirectional Layer Scheduling**:
  - Detects topological splits, saddles, and independent local maxima in the arrival-time field $\tau(\mathbf{x})$, scheduling non-planar isosurfaces bidirectionally so both bottom-up and top-down branch constraints satisfy $[h_{\min}, h_{\text{target}}, h_{\max}]$.

---

## 4. 3D Conformal Shells, Arachne Variable Widths & TPMS Infill

- **True 3D Euclidean Shell Thickness**:
  - Instead of counting a fixed number of vertical "top" and "bottom" planar layers, Manifold Slicer defines the dense outer shell via the 3D signed distance field: $-T_{\text{shell}} \le \phi_{\text{sdf}}(\mathbf{x}) \le 0$. Roofs, floors, domes, and vertical walls all receive a true uniform 3D skin thickness.
- **Conformal Mesh Subdivision & Geodesic Perimeter Insetting**:
  - Subdivides coarse interior triangles near the shell boundary using fine-grid SDF interpolation and extracts smooth outer walls ($\phi_{\text{sdf}} = -\frac{1}{2}w_0$) and concentric inner walls Directly on the curved 2-manifold layer mesh.
- **Arachne-Style Medial-Axis Variable Bead Sizing & Gap Fill**:
  - Dynamically expands or compresses wall and infill extrusion widths within configurable `(target_mm, min_mm, max_mm)` bounds along thin wedges and extracts medial-axis `GapFill` traces for residual pockets without leaving pinhole voids.
- **Volumetric 3D TPMS & Space-Diagonal Core Infill**:
  - Intersects curved non-planar layer surfaces directly with analytical 3D lattices inside the core interior ($\phi_{\text{sdf}}(\mathbf{x}) < -T_{\text{shell}}$):
    - **Gyroid TPMS**: $\sin X\cos Y + \sin Y\cos Z + \sin Z\cos X = 0$
    - **Schwarz-D (Diamond) TPMS**
    - **Schwarz-P (Primitive) TPMS**
    - **Cubic 3D Space-Diagonal Truss**
    - **Monotonic Rectilinear**, **Concentric**, and **All Walls (100% Concentric)**

---

## 5. Gravity-Aware Bridging, Wave Overhangs & Causal Supports

- **Two-Pier Circular Bridging (`Bridge`)**:
  - Detects unsupported high-overhang spans anchored at both ends within `max_bridge_length_mm` and emits straight, tensioned free-air circular extrusions ($A_{\text{bridge}} = \pi (d_{\text{nozzle}}/2)^2$) with dedicated bridge speed, flow multiplier, and cooling fan duty cycle.
- **Inside-Out Wave Overhangs (`WaveOverhang`)**:
  - For one-sided cantilevers or curved overhangs that lack a second bridge pier, Manifold Slicer orders passes **strictly inside-out** from the supported boundary, maintaining a precise lateral overlap fraction (`wave_overhang_overlap_fraction`) against the preceding bead even as the 3D centerline curves in space.
- **Causal Support Generation & Eikonal Seeding (`Support` & `SupportInterface`)**:
  - Identifies floating islands, downward-hanging local minima, and overhangs exceeding the toolhead clearance cone.
  - Projects gravity-aligned support columns along $+\hat{\mathbf{g}}_{\text{bed}}$ to the build plate or lower part surfaces while respecting `support_z_gap_mm` and `support_xy_gap_mm`.
  - Optionally seeds the Eikonal arrival-time field $\tau(\mathbf{x})$ at support contact roofs so the part slices causally upward from the support interface, and supports a **Support Only Unreachable & Bridges** mode to maximize support-free non-planar printing.

---

## 6. Physics-Based Polymer Rheology & Klipper G-code Generation

Manifold Slicer includes a physics-driven G-code compiler tailored for **Klipper** and modern multi-axis motion controllers:

1. **Chained Lookahead Velocity Planner with Stepper Back-EMF Dynamics**:
   - Computes exact entry, cruise, and exit velocities across every 3D polyline vertex using Klipper's `square_corner_velocity` ($v_{\text{scv}}$), `minimum_cruise_ratio`, per-feature accelerations, hotend volumetric flow ceiling ($v \le Q_{\max} / (w \cdot h)$), and high-speed stepper motor back-EMF torque roll-off ($a_{\text{eff}}(v) = a_0(1 - v / v_{\max,\text{stepper}})$).
2. **Shear-Thinning Non-Newtonian Pressure Advance ($k(Q)$)**:
   - Models polymer shear-thinning across volumetric flow rates via power-law interpolation between $(Q_{\text{low}}, k_{\text{low}})$ and $(Q_{\text{high}}, k_{\text{high}})$.
   - Supports both deadband-compacted firmware `SET_PRESSURE_ADVANCE ADVANCE=...` emission and **Slicer-Side Trapezoidal PA Subdivision** (splitting segments into entry-accel, cruise, and exit-decel sub-segments with pre-compensated $E$ values).
3. **Viscoelastic Die-Swell Flow & Geometric Inset Compensation ($B(Q)$)**:
   - Compensates for radial extrudate swell exiting the nozzle orifice as a function of shear rate, scaling flow by $1/B(Q)$ and insetting outer walls along $-\hat{\mathbf{y}}_{\text{nozzle}}$ so high-speed perimeters remain dimensionally accurate.
4. **Klipper SCV Corner Overlap & Centripetal Shortcut Compensation**:
   - Deducts both the geometric inner-corner diamond overlap volume and the centripetal corner-rounding shortcut volume induced by Klipper's `square_corner_velocity` on segments entering and exiting sharp turns.
5. **First-Order Melt-Chamber Transient Pressure Lag**:
   - Tracks internal hotend melt-zone pressure state $P_{\text{state}}$ across variable-width and variable-height non-planar segments, preventing over-extrusion swells when transitioning from high-flow infill to thin conformal walls.
6. **Flat-Nozzle Annular Slope Clearance**:
   - Elevates the nozzle tip along $\hat{\mathbf{z}}_{\text{nozzle}}$ on tilted outer walls by $\Delta z = R_{\text{flat}}\sin\alpha\,(1 - \cos\alpha)$ so the brass nozzle flat never gouges the uphill side of a conformal layer.
7. **3D Helical Scarf Joints & Viscoelastic Ooze-Relaxation Retractions**:
   - Replaces abrupt perimeter seams with a 3D helical height-and-flow ramp along $\hat{\mathbf{z}}_{\text{nozzle}}$, pairs retractions with forward/backward wipes and local-frame clearance hops, and scales unretract prime length using an exponential thermal ooze relaxation model $1 - e^{-t_{\text{travel}} / \tau_{\text{ooze}}}$.
8. **Klipper `[exclude_object]` & Embedded Moonraker 3D Thumbnails**:
   - Emits `EXCLUDE_OBJECT_DEFINE` convex-hull polygons and embeds software-rasterized `32x32` and `300x300` isometric 3D toolpath PNG thumbnails directly in the `.gcode` header.

---

## 7. Headless CLI & Automated Profile Management

The `manifold-frontend` binary provides a full headless Command-Line Interface (see [`docs/cli.md`](file:///Users/amcgregor/work/manifold2/docs/cli.md)):
- **Headless Slicing (`manifold-frontend slice`)**: Batch-slice `.step`, `.stl`, and single- or multi-plate `.3mf` files to `.gcode` (and optionally export arranged `.3mf` projects) with layered `.json`/`.toml` profiles, named presets, and `-s key=value` overrides.
- **Profile & Preset Verbs (`manifold-frontend profile`)**: `init`, `show`, `get`, `set`, `diff`, `keys`, `list`, `save-preset`, `delete-preset`, and `validate` for reproducible CI/CD slicing pipelines and rapid profile tuning.
