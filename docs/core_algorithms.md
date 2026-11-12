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

### 1.3 Nozzle Clearance Envelopes (`PiecewiseXZ`, `ParametricCone`, `RectilinearCone`)
Each toolhead defines a 3D collision clearance envelope in the local nozzle frame ($\hat{\mathbf{x}}_{\text{nozzle}}, \hat{\mathbf{y}}_{\text{nozzle}}, \hat{\mathbf{z}}_{\text{nozzle}}$) with axial height $z \ge 0$ measured along $\hat{\mathbf{z}}_{\text{nozzle}}$:
1. **`PiecewiseXZ`**: Multi-point revolved profile $\{(r_i, z_i)\}_{i=0}^{m-1}$ ($r_i \ge 0$ perpendicular to $\hat{\mathbf{z}}_{\text{nozzle}}$). The maximum collision-free surface slope angle $\theta_{\text{clear}}$ relative to the plane orthogonal to $\hat{\mathbf{z}}_{\text{nozzle}}$ is bounded by the tightest secant cone from the nozzle tip flat $(r_0, 0)$:
   $$\theta_{\text{clear}} = \min_{i \ge 1} \arctan\!\left(\frac{z_i}{r_i - r_0}\right), \qquad \theta_{\max} = \min\!\left(\theta_{\text{user\_max}},\; \theta_{\text{clear}} - 2^\circ\right)$$
2. **`ParametricCone`**: Straight-sided circular cone defined by tip radius $r_0$, cone half-angle $\theta_{\text{cone}}$ from $\hat{\mathbf{z}}_{\text{nozzle}}$, and axial height $H_{\text{cone}}$, with radius $r(z) = r_0 + z\tan\theta_{\text{cone}}$ and $\theta_{\text{clear}} = 90^\circ - \theta_{\text{cone}}$.
3. **`RectilinearCone`**: Anisotropic 4-sided rectangular pyramid defined by tip half-widths $(w_{x0}, w_{y0})$ along $(\hat{\mathbf{x}}_{\text{nozzle}}, \hat{\mathbf{y}}_{\text{nozzle}})$ and independent half-angles $(\theta_X, \theta_Y)$ in the local $XZ$ and $YZ$ planes, yielding directional half-widths $w_x(z) = w_{x0} + z\tan\theta_X$, $w_y(z) = w_{y0} + z\tan\theta_Y$ and directional slope bounds $(\theta_{\text{clear},X}, \theta_{\text{clear},Y}) = (90^\circ - \theta_X,\, 90^\circ - \theta_Y)$.

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

### 3.4 Constrained Anisotropic Eikonal Solver ($\tau(\mathbf{x})$) & Height-Only Planar Mode
Manifold Slicer supports two order-field formulations (`eikonal.order_field_mode`):
- **`HeightOnlyPlanar`**: Evaluates the linear height field $\tau(\mathbf{x}) = \max\bigl(0,\, (\mathbf{x} - \mathbf{x}_{\min,\text{bed}}) \cdot \hat{\mathbf{z}}_{\text{nozzle}}(\mathbf{x})\bigr)$ along the queried local nozzle axis with $0^\circ$ tool tilt, producing flat planar layers perpendicular to $\hat{\mathbf{z}}_{\text{nozzle}}$ while retaining 3D SDF shell classification, variable layer heights, bridging, and gravity-aware support detection.
- **`ConstrainedNonPlanar` (Default)**: The non-planar arrival-time / order field $\tau(\mathbf{x}) \ge 0$ satisfies the constrained Eikonal problem:

$$\|\nabla\tau(\mathbf{x})\| = \frac{1}{F(\mathbf{x})}, \qquad \text{subject to} \quad \frac{\nabla\tau(\mathbf{x})}{\|\nabla\tau(\mathbf{x})\|} \cdot \hat{\mathbf{z}}_{\text{nozzle}}(\mathbf{x}) \ge \cos\theta_{\max}$$

1. **Boundary Seeding**: Voxels contacting the build plate ($z \le z_{\min} + 1.5\Delta x$) are locked at $\tau(\mathbf{x}) = 0$. When `seed_order_field_from_supports` is enabled, unreachable overhang minima supported by support columns are seeded at their causal anti-gravity height $\tau_{\text{seed}} = (\mathbf{x} - \mathbf{x}_{\min}) \cdot (-\hat{\mathbf{g}}_{\text{bed}})$.
2. **8-Octant Gauss-Seidel Fast Sweeping & Gradient-Cone Clamping**:
   In each iteration, the solver sweeps through all 8 octant index orderings $(\pm i, \pm j, \pm k)$, solving the 3D Godunov upwind quadratic update from smaller neighbor values $(\tau_x, \tau_y, \tau_z)$ modulated by the anisotropic alignment speed $F(\mathbf{x}) = \max\bigl(0.25,\; \hat{\mathbf{g}}_{\text{upwind}} \cdot \hat{\mathbf{d}}(\mathbf{x})\bigr)$, and projects $\nabla\tau$ onto the allowable toolhead clearance cone $\cos\theta_{\max}$ (or elliptical cone $(\theta_{\max,X}, \theta_{\max,Y})$ for `RectilinearCone`) so no isosurface ever exceeds `max_nonplanar_slope_deg` or collides with the toolhead clearance envelope.
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

### 4.3 Conformal SDF Surface Skin Slicing & 3D Tool-Clearance Collision Prevention
On sloped upward-facing decks, roofs, and domes where the order field $\tau(\mathbf{x})$ arrives at an oblique angle ($\nabla\tau \not\parallel \nabla\phi_{\text{sdf}}$), standard order-field isosurfaces terminate in stair-steps across the exterior surface. Manifold Slicer eliminates these stair-steps via **Conformal SDF Surface Skin Slicing**:

1. **Candidate Surface Selection (Automatic & Manual Hints)**:
   - **Manual `ConformalSurfaceSkin` (`Skin`) Hints**: Any B-Rep face or mesh patch tagged with `SurfaceHintMode::ConformalSurfaceSkin` is assigned its custom `skin_layer_count` ($N_p \in [1, 12]$), or explicitly excluded from automatic detection when `skin_layer_count = 0`.
   - **Automatic Obliqueness Detection (`enable_auto_conformal_skin`)**: Upward-facing exit voxels near $\phi_{\text{sdf}}(\mathbf{x}) = 0$ where $\hat{\mathbf{n}}_{\text{sdf}} \cdot \hat{\mathbf{z}}_{\text{nozzle}} \ge \cos(\theta_{\text{max,slope}})$ and the obliqueness angle between $\nabla\tau$ and $\hat{\mathbf{n}}_{\text{sdf}} = \nabla\phi_{\text{sdf}}$ exceeds `conformal_skin_min_oblique_deg` are selected as candidate skin voxels with $N_p = \text{conformal\_skin\_layers}$.
2. **Two-Stage 3D Tool-Clearance Collision Pruning**:
   - Because conformal skin layers are printed *after* the underlying eroded core of patch $P$ completes at $\tau_{\text{completion}}(P) = \max_{\mathbf{x} \in P} \tau(\mathbf{x})$, reaching down onto the lower edge of a recessed or obstructed sloped deck (such as the sloped bow deck of a 3DBenchy flanked by raised hull bulwarks and cabin walls) could cause the nozzle tip, heater block, or fan shroud to collide with surrounding geometry.
   - For any candidate skin point $\mathbf{x}$ with inward erosion depth $D_{\text{erode}} = N_p \cdot h_{\text{top}}$, Manifold Slicer places the virtual nozzle tip at the deepest skin layer $\mathbf{p}_{\text{tip}} = \mathbf{x} - D_{\text{erode}}\,\hat{\mathbf{n}}_{\text{sdf}}(\mathbf{x})$, queries the local orthonormal `NozzleFrame` $(\hat{\mathbf{x}}_{\text{nozzle}}, \hat{\mathbf{y}}_{\text{nozzle}}, \hat{\mathbf{z}}_{\text{nozzle}})$, and sweeps the active tool's 3D clearance envelope (`PiecewiseXZ`, `ParametricCone`, or anisotropic `RectilinearCone`) upward along $+\hat{\mathbf{z}}_{\text{nozzle}}$ up to `maxClearanceHeight`:
     - **Stage 1 (Pre-Cluster Protrusion Stand-Off)**: Rejects any voxel where the 3D tool clearance envelope intersects solid geometry rising above the local tangent plane $(\mathbf{y} - \mathbf{x})\cdot\hat{\mathbf{n}}_{\text{sdf}}(\mathbf{x}) > \epsilon_{\text{curv}}$.
     - **Coplanar Step-Bounded 26-Connected Clustering**: Groups surviving voxels into contiguous patches while refusing to merge across sharp normal creases ($\hat{\mathbf{n}}_a \cdot \hat{\mathbf{n}}_b < \cos 22^\circ$) or vertical step discontinuities ($|(\mathbf{x}_b - \mathbf{x}_a)\cdot\hat{\mathbf{n}}| > 1.25\,\Delta x_{\text{fine}}$), preventing a lower deck and an upper roof from merging into a single patch.
     - **Stage 2 (Iterative Causal $\tau_{\text{completion}}(P)$ Pruning)**: Computes $\tau_{\text{completion}}(P) = \max_{\mathbf{x} \in P} \tau(\mathbf{x})$ for each cluster $P$ and prunes any voxel whose 3D tool clearance cone intersects **any** part voxel $\mathbf{y}$ that has already been printed at or before $\tau_{\text{completion}}(P)$ ($\tau(\mathbf{y}) \le \tau_{\text{completion}}(P) + \epsilon_\tau$). Iterates up to 4 passes as $\tau_{\text{completion}}(P)$ decreases when high-$\tau$ obstructed edges are trimmed.
   - **Coarse Brick Acceleration & Multi-Core Execution**: Both 3D clearance sweeps use a flat dense 3D coarse brick acceleration grid (`brickMinTau`, $4^3$ fine voxels per brick) storing the minimum $\tau$ inside each brick to skip empty exterior air and unprinted upper bricks in $O(1)$ array lookups without touching the OpenVDB `RootNode` map, parallelized across CPU cores via `tbb::parallel_for`.
3. **Core SDF Erosion & Conformal Skin Layer Extraction**:
   - The underlying core SDF is eroded strictly within the collision-free patch footprint via a non-dilating narrow-band field:
     $$\phi_{\text{core}}(\mathbf{x}) = \phi_{\text{sdf}}(\mathbf{x}) + D_{\text{erode}}(\mathbf{x})$$
     causing standard order-field layers to terminate cleanly beneath the skin at $\phi_{\text{core}}(\mathbf{x}) = 0$.
   - For each patch $P$ and skin layer $m \in \{1, \dots, N_p\}$, Manifold Slicer extracts the conformal SDF offset isosurface:
     $$\phi_{\text{sdf}}(\mathbf{x}) = -\bigl(N_p - m + 0.15\bigr)\,h_{\text{top}}$$
     trimmed to patch $P$'s tangent footprint and scheduled causally at $\tau_{\text{layer}} = \tau_{\text{completion}}(P) + m \cdot 10^{-3}$ with `is_conformal_skin_layer = true` so the toolpath generator fills the entire skin surface with dense concentric walls and solid infill.

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
4. **Sub-Skin Boundary Anchor Contours & Rib-Anchored Solid Infill Over Sparse Infill**:
   - When an isolated solid roof, sloped deck, or internal floor begins over sparse core infill on layer $k$, its internal boundary ($\phi_{\text{sdf}} = -T_{\text{shell}}$) hangs over empty sparse infill cells rather than resting on a lower perimeter wall. Without anchoring, infill turnaround loops or concentric rings would drop into the cavities between sparse infill ribs and curl upward.
   - **1-Layer Lookahead Sub-Skin Boundary Anchor Contours**: While generating layer $k-1$, Manifold Slicer projects layer $k$'s $\phi_{\text{sdf}}^{\text{next}}(\mathbf{x}) = -T_{\text{shell}}$ contour onto layer $k-1$. Any segment that lies inside layer $k-1$'s sparse core ($\phi_{\text{sdf}}^{\text{curr}}(\mathbf{x}) < -T_{\text{shell}} - 0.15\,w$) is emitted on layer $k-1$ as a continuous `SolidInfill` anchor perimeter bridging directly across the sparse infill ribs to form a solid landing shelf for layer $k$.
   - **Automatic Over-Sparse `Concentric` Promotion & Rib Extension**: Any solid infill region on layer $k$ whose centroid lies over sparse core on layer $k-1$ automatically promotes `Concentric` solid infill to rib-anchored `Monotonic` scanlines, and extends each scanline endpoint that terminates on the internal $\phi_{\text{sdf}} = -T_{\text{shell}}$ boundary outward into the sparse core (up to $1.35\times$ the sparse infill pitch) until it intersects and welds onto an actual layer $k-1$ sparse infill rib or perimeter.

### 5.3 Analytical 3D TPMS & Space-Diagonal Core Infill
Inside the core region ($d(\mathbf{x}) > T_{\text{shell}}$), Manifold Slicer evaluates analytical 3D scalar fields directly at the 3D coordinates $\mathbf{x} = (x, y, z)$ of the curved layer mesh (rotated in XY by `infill_angle_deg`) with spatial wavevector $k = 2\pi / L(\rho)$ scaled by `infill_density_percent` $\rho$:
- **Gyroid TPMS**:
  $$f_{\text{gyroid}}(\mathbf{x}) = \sin(kx)\cos(ky) + \sin(ky)\cos(kz) + \sin(kz)\cos(kx) = 0$$
- **Schwarz-D (Diamond) TPMS**:
  $$f_{\text{diamond}}(\mathbf{x}) = \sin(kx)\sin(ky)\sin(kz) + \sin(kx)\cos(ky)\cos(kz) + \cos(kx)\sin(ky)\cos(kz) + \cos(kx)\cos(ky)\sin(kz) = 0$$
- **Schwarz-P (Primitive) TPMS**:
  $$f_{\text{primitive}}(\mathbf{x}) = \cos(kx) + \cos(ky) + \cos(kz) = 0$$

Extracting the $f(\mathbf{x}) = 0$ contour directly on each curved triangle mesh $\mathcal{M}_k$ produces continuous, smoothly curving 3D minimal-surface infill ribs that never suffer from planar staircase misalignment. To prevent infill scanlines or turnaround connectors from cutting across concave hull recesses (such as a Benchy bow or multi-resolution coarse-to-fine shell boundaries), boundary contours are projected onto the exact fine-grid $\phi_{\text{sdf}} = 0$ zero-crossing, valley-spanning mesh edges are split when their midpoint SDF exceeds the endpoint SDFs, and infill turnaround connectors are rejected whenever their 3D midpoint falls outside the core shell boundary.

### 5.4 Gravity-Aware Bridging, Inside-Out Wave Overhangs & Causal Supports
1. **Unsupported Overhang Identification**: An extrusion vertex $\mathbf{x}$ with overhang $o(\mathbf{x}) > \sin(\theta_{\text{ovh\_thresh}})$ is checked against the spatial hash of the preceding layer ($k - 1$) along $+\hat{\mathbf{g}}_{\text{bed}}$. If no supporting extrusion or support interface lies within $1.75\,h_{\text{layer}}$, it is flagged as an unsupported overhang.
2. **Circular Bridging vs. Wave Overhangs**:
   - If a contiguous unsupported segment is anchored at both ends on supported regions with 3D span length $L_{\text{span}} \le L_{\text{max\_bridge}}$, it is replaced by a straight two-pier **`Bridge`** path extruded with a free-air circular cross-section $A_{\text{bridge}} = \pi (d_{\text{nozzle}}/2)^2$.
   - Otherwise, it is classified as **`WaveOverhang`** and re-ordered **strictly inside-out** (from deepest supported contour outward to the outer perimeter), shifting its 3D centerline inward along the local layer tangent plane so it overlaps the adjacent inner bead by $\Delta w_{\text{ov}} = \alpha_{\text{ov}}\,w_{\text{std}}$, with compensated cross-sectional area $A_{\text{wave}} = (w - \Delta w_{\text{ov}})\,h$.
3. **Causal Support Columns & Obstacle-Avoiding Non-Planar Tree Supports (`ColumnarGrid` & `NonPlanarTree`)**:
   - Downward-hanging local minima of $\tau(\mathbf{x})$ and overhangs exceeding the Eikonal slope cone are collected into support contact masks.
   - In **`ColumnarGrid`** mode, support volumes are projected along $+\hat{\mathbf{g}}_{\text{bed}}$ onto a 2D anti-gravity support grid, generating coarse `Support` base rasters (`support_base_spacing_mm`) and dense orthogonal `SupportInterface` roofs (`support_interface_spacing_mm`, where `0.0 mm` yields a 100% solid zero-gap soluble roof) offset by `support_z_gap_mm` and `support_xy_gap_mm`.
   - In **`NonPlanarTree`** mode, overhang contact tips (`tree_tip_diameter_mm`) propagate step-by-step downward along $+\hat{\mathbf{g}}_{\text{bed}}$, steering laterally outward along $+\nabla\phi_{\text{sdf}}(\mathbf{x})$ within `tree_branch_angle_deg` to route around model geometry (and anchoring exclusively on the build plate when `support_on_bed_only = true`), coalescing into tapered hollow trunks (`tree_trunk_max_diameter_mm`, `tree_branch_wall_count`, defaulting to `1` perimeter wall) anchored on closed concentric bed adhesion base pads (`support_bed_interface_layers`). On 5/6-axis hardware (`has_rotational_axes`), the nozzle axis $\hat{\mathbf{z}}_{\text{nozzle}}$ tilts to align with the 3D branch centerline tangent so leaning tree branches print self-supportingly. Supports can also be assigned to a dedicated toolhead via `support_tool_index`.

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

Manifold Slicer deducts `corner_flow_comp_ratio` $\times \bigl(V_{\text{diamond}} + \Delta L_{\text{scv}}\,w h\bigr)$ symmetrically from the segments immediately entering and exiting the corner. On short segments ($L_{\text{seg}} \le 2.5\,w$), the deduction is scaled by $\min\bigl(1,\, L_{\text{seg}} / (2.5\,w)\bigr)$ and clamped to at most $35\%$ of the segment's nominal volume so rapid micro-zigzags or finely tessellated curved contours never suffer flow starvation.

### 6.4 Melt-Chamber Transient Pressure Lag, Flat-Nozzle Slope Clearance, 3D Scarf Seams & Error-Diffused Precision
1. **First-Order Melt-Chamber Pressure Lag**: Tracks normalized nozzle pressure $P_{\text{state}}(t)$ with relaxation time $\tau_{\text{melt}} = \text{transient\_pressure\_tau\_s}$ across segments:
   $$P_{\text{state},i} = P_{\text{target},i} + \bigl(P_{\text{state},i-1} - P_{\text{target},i}\bigr)\,e^{-\Delta t_i / \tau_{\text{melt}}}$$
   When stepping down from high-flow infill to a slow wall ($P_{\text{state},i-1} > P_{\text{target},i}$), segment flow is throttled by $(P_{\text{target}} / P_{\text{state}})^{\gamma} \ge m_{\min}$ while stored melt-zone pressure bleeds off.
2. **Flat-Nozzle Annular Slope Clearance**: On a conformal layer tilted by angle $\alpha = \arccos(\hat{\mathbf{n}}_{\text{layer}} \cdot \hat{\mathbf{z}}_{\text{nozzle}})$ relative to the nozzle axis, the nozzle tip is elevated along $+\hat{\mathbf{z}}_{\text{nozzle}}$ by:
   $$\Delta z_{\text{flat}}(\alpha) = R_{\text{flat}}\,\sin\alpha\,(1 - \cos\alpha)$$
   preventing the brass nozzle flat land ($R_{\text{flat}} = d_{\text{nozzle}}$) from scraping the uphill slope.
3. **3D Helical Scarf Joints**: On closed perimeter loops, the first $L_{\text{scarf}}$ millimeters ramp height along $\hat{\mathbf{z}}_{\text{nozzle}}$ and volumetric flow linearly from $r_0 \in [0.02, 0.50]$ up to $1.0$, while the final $L_{\text{scarf}}$ millimeters ramp height and flow from $(1 - r_0)$ down to $0$, forming a flush lap joint with constant total wall height.
4. **High-Precision Coordinate Emission & E-Axis Rounding Error Diffusion**:
   Because curved non-planar meshes and conformal scarf joints contain dense sub-millimeter micro-segments ($0.05\text{--}0.20\text{ mm}$), 3-decimal coordinate quantization can introduce up to $5^\circ$ of artificial tangent jitter, while truncating relative `M83` `E` increments to 4 decimal places can lose $1\text{--}5\%$ of total extrusion volume. Manifold Slicer emits `X`/`Y`/`Z` coordinates at **4 decimal places** ($0.1\,\mu\text{m}$ resolution), `E` increments at **5 decimal places** ($0.01\,\mu\text{m}$ resolution), and diffuses rounding residuals across consecutive `G1` moves via an accumulator (`e_rounding_residual_mm`) so cumulative extrusion error stays $< 10^{-5}\text{ mm}$.

### 6.5 Gravity-Aligned Height Maps, Monotonic Layer Z, 3D Toolhead Clearance Clamping & Obstacle-Aware Z-Hop
To guarantee collision-free non-planar G-code on parts with steep elevation swings, cantilevers, or narrow clip prongs:
1. **Gravity-Aligned Printed Height Map (`PrintedHeightMap`)**:
   Constructs a right-handed orthonormal basis $(\hat{\mathbf{e}}_1, \hat{\mathbf{e}}_2, \hat{\mathbf{u}}_{\text{bed}})$ aligned with anti-gravity $\hat{\mathbf{u}}_{\text{bed}} = -\hat{\mathbf{g}}_{\text{bed}}$ and tracks both a fine `0.5 mm` height grid (`prev_layer_max_h`, `curr_max_h`) and a `2.0 mm` coarse clearance grid (`curr_coarse_max_h`).
2. **Low-to-High Elevation-Banded Chain Ordering**:
   When a non-planar layer spans $\Delta h > 0.45\text{ mm}$ along $\hat{\mathbf{u}}_{\text{bed}}$, toolpath chains within each priority tier are partitioned into ascending `0.60 mm` elevation bands so valleys are always printed before neighboring peaks.
3. **Monotonic Layer Z & 3D Toolhead Clearance Cone Clamping**:
   After seam/wipe generation and segment subdivision, every vertex and `0.35 mm` interior segment sample $\mathbf{p}$ is checked against `prev_layer_max_h` (so no extrusion ever plows below the preceding layer's solidified top surface) and against `curr_coarse_max_h` within the active tool's maximum clearance radius $R_{\text{clear}}$:
   $$h_{\min,\text{clear}}(\mathbf{p}) = \max_{\|\mathbf{q}_{2\text{D}} - \mathbf{p}_{2\text{D}}\| \le R_{\text{clear}}} \bigl(h_{\text{printed}}(\mathbf{q}) - Z_{\text{clear}}(\|\mathbf{q}_{2\text{D}} - \mathbf{p}_{2\text{D}}\|)\bigr)$$
   where $Z_{\text{clear}}(r)$ is evaluated from the active tool's `PiecewiseXZ`, `ParametricCone`, or `RectilinearCone` envelope along $\hat{\mathbf{z}}_{\text{nozzle}}$. Post-seam `Wipe` moves are additionally clamped above `curr_max_h` and the chain's own deposited peak so wipes never dive back into newly extruded ridges.
4. **Obstacle-Aware Non-Planar Z-Hop**:
   Every retracted travel move from $\mathbf{p}_{\text{start}}$ to $\mathbf{p}_{\text{end}}$ sweeps a `2.0 mm` corridor across `PrintedHeightMap` to find the highest printed obstacle $h_{\text{obs}}$ along the travel path, lifting along $\hat{\mathbf{z}}_{\text{nozzle}}$ to at least $\max(h_{\text{start}}, h_{\text{end}}, h_{\text{obs}}) + \text{z\_hop\_mm}$.

### 6.6 Automatic First-Layer Brim Generation (`BedFootprintProfile`)
Before emitting Layer 0, Manifold Slicer analyzes the first-layer footprint $(\hat{\mathbf{e}}_1, \hat{\mathbf{e}}_2)$ and total build height $H_{\text{part}}$ along $\hat{\mathbf{u}}_{\text{bed}}$:
1. Computes the global height-to-base lever-arm ratio $\Lambda_{\text{global}} = H_{\text{part}} / W_{\text{base,min}}$, where $W_{\text{base,min}}$ is the narrowest horizontal span of the Layer 0 footprint.
2. When `brim_width_mm > 0.0` or `enable_auto_brim = true` with $\Lambda_{\text{global}} > 2.2$, rasterizes all Layer 0 part and support chains onto a `0.40 mm` 2D grid, runs an exterior breadth-first flood fill from the grid border so internal part holes are never filled, computes the exact 2D Euclidean distance field $d_{2\text{D}}(u, v)$ to the exterior boundary, and extracts outward concentric Marching-Squares brim loops at offsets $d_k = \text{brim\_gap\_mm} + (k + \frac{1}{2})\,w_{\text{brim}}$ printed from outside in before the first-layer walls.

### 6.7 Automatic Lever-Arm & Curled-Corner Speed/Acceleration Governor
Tall parts with narrow bases and cantilevered overhangs experience amplified bending moments $\tau = F_{\text{drag}}\,h$ from viscous nozzle drag and contact with upward-curled corners. When `enable_lever_arm_slowdown = true`, each extruding segment $i$ at height $h_i$ above the bed is evaluated for:
1. **Bed-Detachment & Prong-Bending Lever Ratios**:
   $$\Lambda_{\text{bed},i} = \frac{h_i + 1.35\,d_{\text{cant},i}}{W_{\text{base,min}}}, \qquad \Lambda_{\text{prong},i} = \frac{0.42\,h_i}{W_{\text{layer,min}}(h_i)}, \qquad \Lambda_{\text{eff},i} = \max(\Lambda_{\text{bed},i},\, \Lambda_{\text{prong},i})$$
   where $d_{\text{cant},i}$ is the horizontal cantilever distance outside the Layer 0 footprint hull and $W_{\text{layer,min}}(h_i)$ is the local layer cross-section width. Above $\Lambda_{\text{ref}} = \text{lever\_arm\_reference\_ratio}$, the baseline lever factor scales as:
   $$f_{\text{lever},i} = \text{clamp}\!\left(\frac{1}{\sqrt{1 + 0.85\,(\Lambda_{\text{eff},i} / \Lambda_{\text{ref}} - 1)^2}},\; f_{\min},\; 1\right)$$
2. **Window-Smoothed Curled-Corner Risk ($\mathcal{C}_{\text{curl},i} \in [0, 1]$)**:
   Combines horizontal corner turning angle $\theta_{xy} = \arccos(\hat{\mathbf{t}}_{\text{prev},xy} \cdot \hat{\mathbf{t}}_{\text{curr},xy})$, non-planar climb slope $|\Delta h / \Delta s|$, and overhang/bridge proximity, smoothed over a `3.0 mm` neighborhood window so speed drops *before* the nozzle reaches a curled corner and stays low until past the apex:
   $$f_{\text{total},i} = \text{clamp}\!\left(f_{\text{lever},i} \cdot \bigl(1 - \mathcal{C}_{\text{curl},i}\,(1 - f_{\text{cornerFloor}})\bigr),\; 0.60\,f_{\min},\; 1\right)$$
   Both segment target speed and segment acceleration are scaled by $f_{\text{total},i}$ prior to lookahead velocity planning.

### 6.8 Dynamic Volumetric & Airflow-Residence Cooling Fan Matching
When `enable_dynamic_fan_speed = true` (after `disable_fan_first_layers`), Manifold Slicer replaces static layer-wide fan speeds with a physics-based per-segment PWM schedule (`compute_layer_fan_pwm_schedule`) bounded by `[min_fan_speed_percent, max_fan_speed_percent]`:
1. **Instantaneous Volumetric Flow Demand**:
   $$u_{\text{flow},i} = \text{clamp}\!\left(\frac{Q_i}{0.75\,Q_{\max}},\; 0,\; 1\right), \qquad Q_i = A_i\,v_{\text{avg},i}$$
2. **Cooling Airflow Footprint Residence & Local Heat Density ($R_{\text{duct}} = 12.0\text{ mm}$)**:
   Because a part-cooling fan duct blows across a finite patch of radius $R_{\text{duct}} \approx 12\text{ mm}$ around the nozzle tip, a toolpath that zigzags or loops tightly inside a `12 mm` zone deposits concentrated thermal energy into the same patch of polymer while remaining inside the cooling airflow. For each segment $i$, the planner integrates the total extruded volume $V_{\text{duct},i}$ and residence time $\tau_{\text{duct},i}$ of all consecutive moves remaining within $R_{\text{duct}}$ of $\mathbf{p}_i$:
   $$Q_{\text{duct},i} = \frac{V_{\text{duct},i}}{\tau_{\text{duct},i}}, \qquad u_{\text{duct},i} = \text{clamp}\!\left(\frac{Q_{\text{duct},i}}{0.60\,Q_{\max}} + 0.35\,\text{clamp}\!\left(\frac{3.0\,\text{mm}}{D_{\text{span},i}} - 1,\; 0,\; 1\right),\; 0,\; 1\right)$$
3. **Layer Return Time & Curled-Corner / Overhang Boost**:
   When total layer duration $T_{\text{layer}} < T_{\text{target}} = \text{target\_layer\_cooling\_time\_s}$, a layer-time boost $u_{\text{layer}} = \text{clamp}\bigl((T_{\text{target}} - T_{\text{layer}}) / T_{\text{target}},\, 0,\, 1\bigr)$ ramps cooling upward. Acute corners and non-planar overhangs add $u_{\text{curl},i} \in [0, 1]$ so newly deposited corners freeze immediately before upward thermal curling can occur.
4. **Rotor Spin-Up Lead Compensation & PWM Hysteresis**:
   Combining the demand terms yields $u_i = \text{clamp}\bigl(0.35\,u_{\text{flow},i} + 0.30\,u_{\text{duct},i} + 0.35\,\max(u_{\text{layer}}, u_{\text{curl},i}),\, 0,\, 1\bigr)$. Because DC blower fans require $\sim 0.35\text{ s}$ to spin up, each segment's target fan duty cycle takes the maximum over a `0.35 s` forward lookahead window (`M106` commands are emitted *before* entering an overhang, bridge, or high-flow burst) and applies a `7 / 255` (`~2.7%`) PWM deadband to avoid G-code chatter.
