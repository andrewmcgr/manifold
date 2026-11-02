# Manifold Slicer — Core Mathematical & Geometric Algorithms

This document details the mathematical formulations, volumetric partial differential equations, computational geometry pipelines, and non-Newtonian polymer rheology models implemented across Manifold Slicer's **C++20 backend** (`backend/src/slicer_backend.cpp`) and **Rust frontend** (`frontend/src/toolpath.rs`, `frontend/src/gcode.rs`, `frontend/src/hull.rs`, `frontend/src/arrangement.rs`).

---

## 1. Multi-Axis Kinematic Frames & Clearance Envelopes

### 1.1 Right-Handed Orthonormal Nozzle Frame (`NozzleFrame`)
Manifold Slicer never assumes the nozzle axis is $+Z_{\text{bed}}$. At any bed-space position $\mathbf{x} \in \mathbb{R}^3$ and toolhead travel vector $\mathbf{t} \in \mathbb{R}^3$, the local orthonormal nozzle frame $(\hat{\mathbf{x}}_{\text{nozzle}}, \hat{\mathbf{y}}_{\text{nozzle}}, \hat{\mathbf{z}}_{\text{nozzle}})$ is constructed by Gram-Schmidt orthogonalization:

$$\hat{\mathbf{z}}_{\text{nozzle}} = \frac{\mathbf{a}_{\text{nozzle}}}{\|\mathbf{a}_{\text{nozzle}}\|}, \qquad \hat{\mathbf{x}}_{\text{nozzle}} = \frac{\mathbf{t} - (\mathbf{t} \cdot \hat{\mathbf{z}}_{\text{nozzle}})\,\hat{\mathbf{z}}_{\text{nozzle}}}{\left\|\mathbf{t} - (\mathbf{t} \cdot \hat{\mathbf{z}}_{\text{nozzle}})\,\hat{\mathbf{z}}_{\text{nozzle}}\right\|}, \qquad \hat{\mathbf{y}}_{\text{nozzle}} = \hat{\mathbf{z}}_{\text{nozzle}} \times \hat{\mathbf{x}}_{\text{nozzle}}$$

where $\hat{\mathbf{z}}_{\text{nozzle}}$ points **opposite to the nozzle discharge direction** (up along the nozzle barrel) and $\hat{\mathbf{x}}_{\text{nozzle}} \times \hat{\mathbf{y}}_{\text{nozzle}} = \hat{\mathbf{z}}_{\text{nozzle}}$. Any bed-space point $\mathbf{p}_{\text{bed}}$ transforms into local nozzle coordinates centered at the nozzle tip $\mathbf{p}_{\text{tip}}$ via:

$$\mathbf{p}_{\text{local}} = \begin{bmatrix} (\mathbf{p}_{\text{bed}} - \mathbf{p}_{\text{tip}}) \cdot \hat{\mathbf{x}}_{\text{nozzle}} \\ (\mathbf{p}_{\text{bed}} - \mathbf{p}_{\text{tip}}) \cdot \hat{\mathbf{y}}_{\text{nozzle}} \\ (\mathbf{p}_{\text{bed}} - \mathbf{p}_{\text{tip}}) \cdot \hat{\mathbf{z}}_{\text{nozzle}} \end{bmatrix}$$

### 1.2 Bed-Coordinate Gravity Vector ($\hat{\mathbf{g}}_{\text{bed}}$)
For a 5-axis tilting build plate with Euler tilt angles $(R_x, R_y)$ relative to the world frame, the build-plate rotation matrix is $R_{\text{bed}} = R_x(\theta_x)\,R_y(\theta_y)$. World gravity $\hat{\mathbf{g}}_{\text{world}} = (0, 0, -1)^T$ expressed in bed coordinates is:

$$\hat{\mathbf{g}}_{\text{bed}} = R_{\text{bed}}^{-1} \begin{bmatrix} 0 \\ 0 \\ -1 \end{bmatrix}$$

All overhang fields, bridge checks, and support column projections query $\hat{\mathbf{g}}_{\text{bed}}$ rather than reading the raw $Z$ coordinate.

### 1.3 Piecewise-Linear Nozzle Clearance Envelope
Each toolhead defines a revolved clearance profile $\{(r_i, z_i)\}_{i=0}^{m-1}$ in the local nozzle frame ($z_i \ge 0$ along $\hat{\mathbf{z}}_{\text{nozzle}}$, $r_i \ge 0$ perpendicular to $\hat{\mathbf{z}}_{\text{nozzle}}$). The maximum collision-free surface slope angle $\theta_{\text{clear}}$ relative to the plane orthogonal to $\hat{\mathbf{z}}_{\text{nozzle}}$ is bounded by the tightest secant cone from the nozzle tip flat $(r_0, 0)$:

$$\theta_{\text{clear}} = \min_{i \ge 1} \arctan\!\left(\frac{z_i}{r_i - r_0}\right), \qquad \theta_{\max} = \min\!\left(\theta_{\text{user\_max}},\; \theta_{\text{clear}} - 2^\circ\right)$$

---

## 2. CAD B-Rep Ingestion, Surface Patch Segmentation & Convex Hulls

### 2.1 Exact B-Rep Triangulation & Coplanar Mesh Clustering
1. **STEP B-Rep Import**: OpenCASCADE (`STEPControl_Reader`) loads exact parametric surfaces and tessellates them via `BRepMesh_IncrementalMesh` (linear deflection $0.05\text{ mm}$, angular deflection $0.25\text{ rad}$). Each topological `TopoDS_Face` is mapped to a selectable `SurfacePatch` with area-weighted unit normal $\hat{\mathbf{n}}_{\text{patch}}$ and centroid.
2. **Triangle Mesh Patch Clustering**: For `.stl` and `.3mf` meshes lacking B-Rep topology, Manifold Slicer constructs a triangle adjacency graph via quantized half-edges and runs breadth-first flood fill across adjacent triangles whose normals satisfy $\hat{\mathbf{n}}_a \cdot \hat{\mathbf{n}}_b \ge \cos(12^\circ)$, grouping contiguous planar/gentle regions into selectable `SurfacePatch` primitives.

### 2.2 3D Quickhull & Coplanar Support Facets (`Place on Face`)
To orient arbitrary parts flush onto the build plate (`frontend/src/hull.rs`):
1. An incremental 3D **Quickhull** algorithm builds the outer convex polyhedron from an initial extremal tetrahedron, iteratively partitioning outside vertices across visible horizon ridges.
2. Coplanar adjacent hull triangles ($\hat{\mathbf{n}}_i \cdot \hat{\mathbf{n}}_j \ge \cos(1.5^\circ)$) are merged into candidate planar `HullFace` polygons ranked by surface area.
3. Selecting face $k$ with outward normal $\hat{\mathbf{n}}_k$ applies the shortest-arc quaternion rotation $q_{\text{align}}$ rotating $q_{\text{current}}\,\hat{\mathbf{n}}_k \to (0, 0, -1)^T$, then translates the part so $\min Z = 0$.

### 2.3 2D Convex-Hull Footprint & Archimedean Spiral Packing
1. Projecting all transformed part vertices onto the build plate $(X, Y)$ and running **Andrew's Monotone Chain** yields the 2D convex hull $\mathcal{H}_{2\text{D}}$ (`hull.rs`), used both for Klipper's `EXCLUDE_OBJECT_DEFINE` polygon and for fast Separating Axis Theorem (SAT) collision checks.
2. During **Auto-Arrangement** (`arrangement.rs`), parts are sorted in descending order of footprint area and placed sequentially along an outward **Archimedean spiral** centered at the bed origin $(0, 0)$:

$$r(\theta) = \frac{p_{\text{radial}}}{2\pi}\,\theta, \qquad \mathbf{x}(\theta) = \bigl(r(\theta)\cos\theta,\; r(\theta)\sin\theta\bigr)$$

finding the earliest angle $\theta_k = k\,\Delta\theta$ where the part's hull stays inside the build plate minus `bed_edge_margin_mm` and maintains $\ge \text{part\_clearance\_mm}$ Euclidean distance from all previously placed hulls.

---

## 3. Dual-Grid OpenVDB Volumetric Fields & Constrained Eikonal Solver

### 3.1 Automatic Fine Voxel Scaling & Dual-Grid Multi-Resolution
The fine OpenVDB voxel size $\Delta x_{\text{fine}}$ scales automatically with the active tool's nozzle diameter $d_{\text{nozzle}}$ and target layer height $h_{\text{target}}$:

$$\Delta x_{\text{fine}} = \text{clamp}\!\left(0.375\,d_{\text{nozzle}} + 0.50\,h_{\text{target}},\; 0.08\text{ mm},\; 2.00\text{ mm}\right)$$

When **Dual-Grid Multi-Resolution** (`eikonal.enable_multi_resolution_grid`) is enabled with ratio $R = \text{coarse\_grid\_ratio} \in [2, 8]$ (default $R = 4$):
- **Fine Narrow-Band Shell Grid ($\Delta x_{\text{fine}}$)**: Activated across the outer boundary shell $-T_{\text{shell}} - 2\Delta x_{\text{fine}} \le \phi_{\text{sdf}}(\mathbf{x}) \le +w_{\text{nb}}\Delta x_{\text{fine}}$, capturing sharp CAD edges, thin walls, and surface orientation hints at full resolution.
- **Coarse Full-Interior Core Grid ($\Delta x_{\text{coarse}} = R\,\Delta x_{\text{fine}}$)**: Flood-filled across the entire deep interior ($\phi_{\text{sdf}}(\mathbf{x}) < 0$), reducing interior voxel count and global Eikonal Fast-Sweeping cost by $R^3 = 64\times$.

### 3.2 Signed Distance Field ($\phi_{\text{sdf}}$) & Gravity-Aware Overhang Field ($o$)
1. **Signed Distance Field (`"sdf"`)**: Constructed via `openvdb::tools::meshToVolume`, with negative inside distances ($\phi_{\text{sdf}}(\mathbf{x}) < 0$) and positive outside distances ($\phi_{\text{sdf}}(\mathbf{x}) > 0$). Deep interior background tiles are converted into explicit active interior voxels via morphological erosion so gradients are defined everywhere inside the part.
2. **Gravity-Aware Overhang Field (`"overhang"`)**: At every interior voxel $\mathbf{x}$, the outward unit surface normal is $\hat{\mathbf{n}}_{\text{surf}}(\mathbf{x}) = \nabla\phi_{\text{sdf}}(\mathbf{x}) / \|\nabla\phi_{\text{sdf}}(\mathbf{x})\|$. Querying the bed-coordinate gravity direction $\hat{\mathbf{g}}_{\text{bed}}(\mathbf{x})$ yields the scalar overhang field:

$$o(\mathbf{x}) = -\hat{\mathbf{n}}_{\text{surf}}(\mathbf{x}) \cdot \hat{\mathbf{g}}_{\text{bed}}(\mathbf{x}) \in [-1, +1]$$

The signed overhang angle past vertical is $\theta_{\text{ovh}}(\mathbf{x}) = \arcsin\bigl(\hat{\mathbf{n}}_{\text{surf}}(\mathbf{x}) \cdot \hat{\mathbf{g}}_{\text{bed}}(\mathbf{x})\bigr) \in [-90^\circ, +90^\circ]$, where:
- $\theta_{\text{ovh}} = -90^\circ$: horizontal upward-facing roof ($\hat{\mathbf{n}}_{\text{surf}} = -\hat{\mathbf{g}}_{\text{bed}}$)
- $\theta_{\text{ovh}} = 0^\circ$: vertical wall ($\hat{\mathbf{n}}_{\text{surf}} \perp \hat{\mathbf{g}}_{\text{bed}}$)
- $\theta_{\text{ovh}} > 0^\circ$: downward-facing overhang ($\hat{\mathbf{n}}_{\text{surf}} \cdot \hat{\mathbf{g}}_{\text{bed}} > 0$)

### 3.3 Target Director Field $\hat{\mathbf{d}}(\mathbf{x})$: Overhang Bias & Orientation Hints
Before solving the Eikonal PDE, Manifold Slicer constructs a smooth target layer-normal director field $\hat{\mathbf{d}}(\mathbf{x})$ across the volume:

1. **Baseline Gravity / Nozzle Axis**: Initializes $\mathbf{d}_0(\mathbf{x}) = -\hat{\mathbf{g}}_{\text{bed}}(\mathbf{x})$ (anti-gravity upward build direction).
2. **Automatic Overhang Conformal vs. Orthogonal Surface Bias**:
   Within the boundary shell ($-\phi_{\text{sdf}}(\mathbf{x}) \le D_{\text{bias}}$), let $\hat{\mathbf{n}} = \hat{\mathbf{n}}_{\text{surf}}(\mathbf{x})$ and let $\hat{\mathbf{t}}_{\perp}$ be the upward unit vector **tangent** to the part surface (so layers with normal $\hat{\mathbf{t}}_{\perp}$ meet the surface orthogonally at $90^\circ$):

   $$\hat{\mathbf{t}}_{\perp} = \frac{\mathbf{d}_0 - (\mathbf{d}_0 \cdot \hat{\mathbf{n}})\,\hat{\mathbf{n}}}{\left\|\mathbf{d}_0 - (\mathbf{d}_0 \cdot \hat{\mathbf{n}})\,\hat{\mathbf{n}}\right\|}$$

   Given the supported transition angle $\theta_{\min} = -\theta_{\text{supported}}$ (default $-30^\circ$), overhang transition angle $\theta_{\max} = +\theta_{\text{overhang}}$ (default $+60^\circ$), and transition width $\Delta\theta$ (default $15^\circ$), two $C^1$ cubic Hermite `smoothstep` weights $S_1(\theta_{\text{ovh}})$ and $S_2(\theta_{\text{ovh}})$ blend smoothly between three physical regimes:
   - **Shallow Supported Roofs ($\theta_{\text{ovh}} < \theta_{\min}$)**: Conformal upward ($\hat{\mathbf{d}}_{\text{surf}} \to +\hat{\mathbf{n}}$) to eliminate stair-stepping on domes.
   - **Walls & Moderate Overhangs ($\theta_{\min} \le \theta_{\text{ovh}} \le \theta_{\max}$)**: Orthogonal to surface ($\hat{\mathbf{d}}_{\text{surf}} \to \hat{\mathbf{t}}_{\perp}$) so layers tilt to support overhangs at $90^\circ$.
   - **Steep Overhangs & Ceilings ($\theta_{\text{ovh}} > \theta_{\max}$)**: Conformal inward ($\hat{\mathbf{d}}_{\text{surf}} \to -\hat{\mathbf{n}}$).

3. **User B-Rep Surface Hints & 3D Point Hints**:
   - **Surface Patch Hints**: For each hinted patch $p$ within Euclidean distance $d_p(\mathbf{x}) \le D_p$, applies a cubic falloff weight $w_p(\mathbf{x}) = w_{0,p}\,(1 - d_p / D_p)^2$ steering $\hat{\mathbf{d}}(\mathbf{x})$ either parallel to $\pm\hat{\mathbf{n}}_p$ (`NormalOrthogonalToSurface`) or projected onto the plane tangent to $\hat{\mathbf{n}}_p$ (`NormalTangentToSurface`).
   - **3D Point Hints**: For each 3D point hint $(\mathbf{p}_q, \hat{\mathbf{u}}_q, R_q, w_q)$ with normalized radial distance $r = \|\mathbf{x} - \mathbf{p}_q\| / R_q < 1$, applies the compactly supported **Wendland $C^2$ radial basis kernel**:

   $$\psi(r) = (1 - r)_+^4\,(4r + 1)$$

4. **Laplacian Director Regularization & 5/6-Axis Tool Orientation Field**:
   The blended director field $\hat{\mathbf{d}}(\mathbf{x})$ is smoothed over `solver_iterations` passes with weight `curvature_regularization`. For 5/6-axis tools (`has_rotational_axes = true`), the spatially varying tool axis $\hat{\mathbf{z}}_{\text{nozzle}}(\mathbf{x})$ rotates toward $\hat{\mathbf{d}}(\mathbf{x})$ clamped within `max_angle_from_bed_normal_deg` of $+Z_{\text{bed}}$.

### 3.4 Constrained Anisotropic Eikonal Solver ($\tau(\mathbf{x})$)
The non-planar arrival-time / order field $\tau(\mathbf{x}) \ge 0$ satisfies the constrained Eikonal problem:

$$\|\nabla\tau(\mathbf{x})\| = \frac{1}{F(\mathbf{x})}, \qquad \text{subject to} \quad \frac{\nabla\tau(\mathbf{x})}{\|\nabla\tau(\mathbf{x})\|} \cdot \hat{\mathbf{z}}_{\text{nozzle}}(\mathbf{x}) \ge \cos\theta_{\max}$$

1. **Boundary Seeding**: Voxels contacting the build plate ($z \le z_{\min} + 1.5\Delta x$) are locked at $\tau(\mathbf{x}) = 0$. When `seed_order_field_from_supports` is enabled, unreachable overhang minima supported by support columns are seeded at their causal anti-gravity height $\tau_{\text{seed}} = (\mathbf{x} - \mathbf{x}_{\min}) \cdot (-\hat{\mathbf{g}}_{\text{bed}})$.
2. **8-Octant Gauss-Seidel Fast Sweeping & Gradient-Cone Clamping**:
   In each iteration, the solver sweeps through all 8 octant index orderings $(\pm i, \pm j, \pm k)$, solving the 3D Godunov upwind quadratic update from smaller neighbor values $(\tau_x, \tau_y, \tau_z)$ modulated by the anisotropic alignment speed $F(\mathbf{x}) = \max\bigl(0.25,\; \hat{\mathbf{g}}_{\text{upwind}} \cdot \hat{\mathbf{d}}(\mathbf{x})\bigr)$, and projects $\nabla\tau$ onto the allowable toolhead clearance cone $\cos\theta_{\max}$ so no isosurface ever exceeds `max_nonplanar_slope_deg` or collides with the toolhead XZ clearance profile.
3. **Coarse-to-Fine Prolongation**: On Dual-Grid runs, the converged coarse arrival-time field $\tau_{\text{coarse}}$ is tricubically interpolated onto the fine shell narrow band and relaxed with fine-grid Gauss-Seidel passes so $\tau(\mathbf{x})$ is $C^1$-continuous across the shell-core interface.

---

## 4. Reeb-Graph Multi-Branch Layer Scheduling & Conformal Mesh Subdivision

### 4.1 Bidirectional Multi-Branch Isovalue Selection
Because $\tau(\mathbf{x})$ has speed $F(\mathbf{x}) \approx 1$, the physical normal thickness between two consecutive isosurfaces $\tau_k$ and $\tau_{k+1}$ is:

$$h(\mathbf{x}) = \frac{\tau_{k+1} - \tau_k}{\|\nabla\tau(\mathbf{x})\|}$$

Manifold Slicer builds a discrete **Reeb graph** of connected components across level sets of $\tau(\mathbf{x})$ to track topological splits (e.g., a part branching into multiple towers) and independent local maxima:
- Starting from $\tau_0 = h_{\text{first}}$, each branch steps upward by $\Delta\tau_k = h_{\text{target}} \cdot \overline{\|\nabla\tau\|}_{\text{branch}}$, clamped so that local thickness satisfies $h_{\min} \le h(\mathbf{x}) \le h_{\max}$.
- Near each branch's terminal peak $\tau_{\max,b}$, the remaining interval is redistributed bidirectionally so the final cap layer hits `top_layer_mm` cleanly without leaving an ultra-thin sliver layer.

### 4.2 Isosurface Extraction & Conformal Shell Mesh Subdivision
For each scheduled isovalue $\tau_k$:
1. An initial non-planar layer surface $\mathcal{M}_k$ is extracted from the interior level set $\psi_k(\mathbf{x}) = \tau(\mathbf{x}) - \tau_k = 0$ clipped to the part interior $\phi_{\text{sdf}}(\mathbf{x}) \le 0$ via `openvdb::tools::volumeToMesh`.
2. **Conformal Boundary Shell Mesh Subdivision**: Every triangle in $\mathcal{M}_k$ that overlaps or approaches the fine outer shell ($\min_v \phi_{\text{sdf}}(v) \ge -T_{\text{shell}} - 2\Delta x_{\text{fine}}$) is recursively split ($1 \to 4$ mid-edge refinement) down to edge length $\le \Delta x_{\text{fine}}$, and its new vertices are projected onto the exact $\tau(\mathbf{x}) = \tau_k$ manifold while sampling the **fine-grid** signed distance $\phi_{\text{sdf}}^{\text{fine}}(\mathbf{x})$ and overhang $o^{\text{fine}}(\mathbf{x})$. This guarantees razor-sharp perimeters while keeping the deep core mesh lightweight.

---

## 5. 3D Conformal Toolpaths, Arachne Variable Widths & TPMS Infill

### 5.1 True 3D SDF Shell vs. Core Partitioning
On each non-planar layer mesh $\mathcal{M}_k$, every vertex $\mathbf{v}_i$ carries its exact 3D Euclidean signed distance $\phi_{\text{sdf}}(\mathbf{v}_i) \le 0$ and inward depth $d(\mathbf{v}_i) = -\phi_{\text{sdf}}(\mathbf{v}_i) \ge 0$:
- **Conformal 3D Shell**: $0 \le d(\mathbf{x}) \le T_{\text{shell}}$
- **Interior Sparse Core**: $d(\mathbf{x}) > T_{\text{shell}}$

Because $d(\mathbf{x})$ is the 3D Euclidean distance to the nearest part surface in any direction, horizontal roofs, floors, sloped domes, and vertical walls all receive an identical 3D shell thickness $T_{\text{shell}}$.

### 5.2 Geodesic Isocontouring & Arachne Medial-Axis Variable Bead Widths
1. **Perimeter Extraction**: Outer walls are extracted by marching-triangles isocontouring of $d(\mathbf{x}) = \frac{1}{2}w_{\text{outer}}$ across $\mathcal{M}_k$, followed by $N_{\text{inner}}$ concentric inner walls at inward geodesic/SDF offsets $d_m = w_{\text{outer}} + (m - \frac{1}{2})w_{\text{inner}}$.
2. **Medial-Axis Variable Width (`Arachne`)**: In thin walls or acute wedges where opposite boundaries approach each other with local ridge thickness $D_{\text{ridge}}(\mathbf{x}) = 2\,\max_{v \in \mathcal{N}(\mathbf{x})} d(v)$, the local bead width $w(\mathbf{x})$ is dynamically adapted within $[w_{\min}, w_{\max}]$:

$$w(\mathbf{x}) = \text{clamp}\!\left(\frac{D_{\text{ridge}}(\mathbf{x})}{2\,N_{\text{walls}}},\; w_{\min},\; w_{\max}\right)$$

3. **Medial-Axis Gap Fill**: Residual shell pockets narrower than $2\,w_{\text{solid,target}}$ around local maxima of $d(\mathbf{x})$ are extracted as single-pass variable-width `GapFill` traces.

### 5.3 Analytical 3D TPMS & Space-Diagonal Core Infill
Inside the core region ($d(\mathbf{x}) > T_{\text{shell}}$), Manifold Slicer evaluates analytical 3D scalar fields directly at the 3D coordinates $\mathbf{x} = (x, y, z)$ of the curved layer mesh (rotated in XY by `infill_angle_deg`) with spatial wavevector $k = 2\pi / L(\rho)$ scaled by `infill_density_percent` $\rho$:
- **Gyroid TPMS**:
  $$f_{\text{gyroid}}(\mathbf{x}) = \sin(kx)\cos(ky) + \sin(ky)\cos(kz) + \sin(kz)\cos(kx) = 0$$
- **Schwarz-D (Diamond) TPMS**:
  $$f_{\text{diamond}}(\mathbf{x}) = \sin(kx)\sin(ky)\sin(kz) + \sin(kx)\cos(ky)\cos(kz) + \cos(kx)\sin(ky)\cos(kz) + \cos(kx)\cos(ky)\sin(kz) = 0$$
- **Schwarz-P (Primitive) TPMS**:
  $$f_{\text{primitive}}(\mathbf{x}) = \cos(kx) + \cos(ky) + \cos(kz) = 0$$

Extracting the $f(\mathbf{x}) = 0$ contour directly on each curved triangle mesh $\mathcal{M}_k$ produces continuous, smoothly curving 3D minimal-surface infill ribs that never suffer from planar staircase misalignment.

### 5.4 Gravity-Aware Bridging, Inside-Out Wave Overhangs & Causal Supports
1. **Unsupported Overhang Identification**: An extrusion vertex $\mathbf{x}$ with overhang $o(\mathbf{x}) > \sin(\theta_{\text{ovh\_thresh}})$ is checked against the spatial hash of the preceding layer ($k - 1$) along $+\hat{\mathbf{g}}_{\text{bed}}$. If no supporting extrusion or support interface lies within $1.75\,h_{\text{layer}}$, it is flagged as an unsupported overhang.
2. **Circular Bridging vs. Wave Overhangs**:
   - If a contiguous unsupported segment is anchored at both ends on supported regions with 3D span length $L_{\text{span}} \le L_{\text{max\_bridge}}$, it is replaced by a straight two-pier **`Bridge`** path extruded with a free-air circular cross-section $A_{\text{bridge}} = \pi (d_{\text{nozzle}}/2)^2$.
   - Otherwise, it is classified as **`WaveOverhang`** and re-ordered **strictly inside-out** (from deepest supported contour outward to the outer perimeter), shifting its 3D centerline inward along the local layer tangent plane so it overlaps the adjacent inner bead by $\Delta w_{\text{ov}} = \alpha_{\text{ov}}\,w_{\text{std}}$, with compensated cross-sectional area $A_{\text{wave}} = (w - \Delta w_{\text{ov}})\,h$.
3. **Causal Support Columns**: Downward-hanging local minima of $\tau(\mathbf{x})$ and overhangs exceeding the Eikonal slope cone are projected along $+\hat{\mathbf{g}}_{\text{bed}}$ onto a 2D anti-gravity support grid, generating coarse `Support` base rasters (`support_base_spacing_mm`) and dense orthogonal `SupportInterface` roofs (`support_interface_spacing_mm`) offset by `support_z_gap_mm` and `support_xy_gap_mm`.

---

## 6. Non-Newtonian Polymer Rheology & Klipper Lookahead G-code Compilation

### 6.1 Chained Lookahead Velocity Planner & Stepper Back-EMF Dynamics
Across each continuous extrusion chain $\mathbf{p}_0, \mathbf{p}_1, \dots, \mathbf{p}_M$, the G-code planner (`frontend/src/gcode.rs`) computes realistic entry, cruise, and exit velocities $(v_{\text{entry},i}, v_{\text{cruise},i}, v_{\text{exit},i})$:
1. **Volumetric Flow Speed Ceiling**: For segment $i$ with width $w_i$ and layer height $h_i$, the cross-sectional area is $A_i = w_i h_i$ (or $\pi d_{\text{nozzle}}^2 / 4$ for bridges), capping segment speed to:
   $$v_{\max,i} = \min\!\left(v_{\text{feature}},\; \frac{Q_{\max}}{A_i}\right)$$
2. **Klipper Centripetal Junction Velocity**: At vertex $i$ with deflection angle $\theta_i = \arccos(\hat{\mathbf{t}}_{i-1} \cdot \hat{\mathbf{t}}_i)$ and Klipper `square_corner_velocity` $v_{\text{scv}}$, the maximum instantaneous corner velocity is:
   $$v_{\text{junction},i} = \frac{v_{\text{scv}}}{\sqrt{\sin(\theta_i / 2) / \sin(\pi / 4)}}$$
3. **Stepper Motor Back-EMF Torque Roll-Off**: When `use_stepper_dynamics = true`, available stepper acceleration decays linearly with velocity toward the pull-out speed $v_{\max,\text{stepper}}$:
   $$\frac{dv}{dt} = a_0\!\left(1 - \frac{v}{v_{\max,\text{stepper}}}\right)$$
   Integrating $v\,dv / a_{\text{eff}}(v) = ds$ yields exact analytic acceleration/deceleration distance bounds used in the backward (deceleration) and forward (acceleration) lookahead passes, followed by `minimum_cruise_ratio` peak clamping.

### 6.2 Power-Law Shear-Thinning Pressure Advance $k(Q)$ & Die Swell $B(Q)$
Molten thermoplastics are shear-thinning viscoelastic fluids (power-law index $n < 1$). Given calibrated low-flow and high-flow anchor points $(Q_{\text{low}}, Q_{\text{high}})$:
- **Shear-Thinning Pressure Advance**:
  $$k(Q) = k_{\text{low}} \left(\frac{\text{clamp}(Q, Q_{\text{low}}, Q_{\text{high}})}{Q_{\text{low}}}\right)^{\frac{\ln(k_{\text{high}} / k_{\text{low}})}{\ln(Q_{\text{high}} / Q_{\text{low}})}}$$
  Emitted either via deadband-compacted `SET_PRESSURE_ADVANCE ADVANCE=...` commands or via **Slicer-Side Trapezoidal Subdivision** (splitting each segment into acceleration, cruise, and deceleration sub-moves with $\Delta E_{\text{pa}} = k(Q)\,\Delta Q / A_{\text{fil}}$).
- **Viscoelastic Die Swell & Outer-Wall Insetting**:
  Interpolating the die-swell ratio $B(Q) \in [B_{\text{low}}, B_{\text{high}}]$ between $Q_{\text{low}}$ and $Q_{\text{high}}$ scales volumetric flow by $1 / B(Q)$ and insets `OuterWall` vertices inward along $-\hat{\mathbf{y}}_{\text{nozzle}}$ by:
  $$\Delta y_{\text{swell}}(Q) = \frac{1}{2}\,w\,\bigl(B(Q) - 1\bigr)$$

### 6.3 Corner Overlap & Klipper SCV Shortcut Deduction
At a corner with turn angle $\theta \in (5^\circ, 165^\circ)$, two physical effects cause corner bulging if uncompensated:
1. **Inner-Corner Diamond Overlap**: Two rectangular beads of width $w$ and height $h$ meeting at angle $\theta$ overlap geometrically by volume:
   $$V_{\text{diamond}} = \frac{1}{4}\,w^2 h \tan\!\left(\frac{\theta}{2}\right)$$
2. **Klipper SCV Centripetal Corner Rounding**: Klipper rounds the corner along an arc of effective radius $R_{\text{eff}} = v_{\text{junction}}^2 / a$, shortening the toolhead's true path length relative to the sharp vertex by:
   $$\Delta L_{\text{scv}} = 2\,R_{\text{eff}}\!\left(\tan\frac{\theta}{2} - \frac{\theta}{2}\right)$$

Manifold Slicer deducts `corner_flow_comp_ratio` $\times \bigl(V_{\text{diamond}} + \Delta L_{\text{scv}}\,w h\bigr)$ symmetrically from the segments immediately entering and exiting the corner.

### 6.4 Melt-Chamber Transient Pressure Lag, Flat-Nozzle Slope Clearance & 3D Scarf Seams
1. **First-Order Melt-Chamber Pressure Lag**: Tracks normalized nozzle pressure $P_{\text{state}}(t)$ with relaxation time $\tau_p \approx k(Q)$ across segments:
   $$P_{\text{state},i} = P_{\text{target},i} + \bigl(P_{\text{state},i-1} - P_{\text{target},i}\bigr)\,e^{-\Delta t_i / \tau_p}$$
   When stepping down from high-flow infill to a slow wall ($P_{\text{state},i-1} > P_{\text{target},i}$), segment flow is throttled by $(P_{\text{target}} / P_{\text{state}})^{\gamma} \ge m_{\min}$ while stored melt-zone pressure bleeds off.
2. **Flat-Nozzle Annular Slope Clearance**: On a conformal layer tilted by angle $\alpha = \arccos(\hat{\mathbf{n}}_{\text{layer}} \cdot \hat{\mathbf{z}}_{\text{nozzle}})$ relative to the nozzle axis, the nozzle tip is elevated along $+\hat{\mathbf{z}}_{\text{nozzle}}$ by:
   $$\Delta z_{\text{flat}}(\alpha) = R_{\text{flat}}\,\sin\alpha\,(1 - \cos\alpha)$$
   preventing the brass nozzle flat land ($R_{\text{flat}} = d_{\text{nozzle}}$) from scraping the uphill slope.
3. **3D Helical Scarf Joints**: On closed perimeter loops, the first $L_{\text{scarf}}$ millimeters ramp height along $\hat{\mathbf{z}}_{\text{nozzle}}$ and volumetric flow linearly from $r_0 \in [0.02, 0.50]$ up to $1.0$, while the final $L_{\text{scarf}}$ millimeters ramp height and flow from $(1 - r_0)$ down to $0$, forming a flush lap joint with constant total wall height.
