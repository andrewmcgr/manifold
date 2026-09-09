# Order Field PDE Solvers: Design Considerations & Anisotropic FSM Architecture

## 1. Executive Summary & Problem Statement

Non-planar 3D printing requires generating an scalar deposition order field $\phi(\mathbf{x}): \Omega \subset \mathbb{R}^3 \to \mathbb{R}$ such that the extracted level sets (isosurfaces $\phi(\mathbf{x}) = \text{const}$) define smooth, collision-free, printable layer sheets.

Currently, Manifold evaluates non-planar order fields using a **Narrow-Band Fast Marching Method (FMM)** on a 3D Cartesian grid (`manifold-fidget::eikonal::EikonalOrderField`), augmented with post-hoc bilateral fixed-point conformal blending, exterior geodesic clamping, slope-limit relaxation, and vertical monotonicity heuristics.

In practice, two core challenges have emerged:
1. **Top and bottom surface conformity is brittle**: The current blending heuristics between the bulk FMM field and surface-offset distance fields ($A \mp d_{\text{side}}$) frequently fight the underlying bulk field, creating boundary artifacts, gradient cliffs, or unprintable layer folds.
2. **Vertical column monotonicity enforcement does not produce desirable geometry**: Enforcing strict $\partial \phi / \partial z \ge c > 0$ along vertical 1D columns distorts curved wavefronts and fails to capture natural non-planar topologies around reentrant features.

### Objective
We require direct, first-principles control over the **tangency and orthogonality** of order field isosurfaces relative to CAD surface boundaries. Rather than stitching disconnected distance fields together with heuristic weights, the order field should emerge from a **continuously steered partial differential equation (PDE)** driven by an **anisotropic velocity / metric tensor field** $\mathbf{D}(\mathbf{x})$:
- Level sets can be guided into near-tangency ($\nabla \phi \parallel \mathbf{n}$) along cosmetic top surfaces.
- Level sets can be guided into near-orthogonality ($\nabla \phi \perp \mathbf{n}$) along vertical walls or overhang boundaries.
- The new solver must coexist alongside the existing FMM solver as a dedicated, independent engine to allow side-by-side visual and physical comparison.

---

## 2. Why Fast Marching Method (FMM) Breaks Down Under Anisotropy

The isotropic Eikonal equation is:
$$\|\nabla \phi(\mathbf{x})\| = \frac{1}{f(\mathbf{x})}$$
In the isotropic case:
- The **characteristics** (rays along which information propagates, $\dot{\mathbf{x}}$) are strictly collinear with the gradient $\nabla \phi$ (the wavefront normal).
- The numerical domain of dependence is strictly upwind in the direction of $-\nabla \phi$.
- Because characteristics align with coordinate rays, Dijkstra-like sorting in a priority queue (FMM) preserves causality: once a grid node attains the minimum tentative value in the active narrow-band heap, its value is guaranteed to be optimal and can be frozen permanently.

### The Anisotropic Breakdown
When propagation speed varies with direction, the medium is characterized by a symmetric positive-definite metric tensor $\mathbf{M}(\mathbf{x}) = \mathbf{D}(\mathbf{x})^{-1}$:
$$\nabla \phi^T \mathbf{D}(\mathbf{x}) \nabla \phi = 1 \quad \iff \quad \|\nabla \phi\|_{\mathbf{D}} = 1$$

Under this equation:
1. **Ray Direction vs. Wavefront Normal Divergence**:
   The group velocity (characteristic ray direction) is:
   $$\mathbf{v}_g = \dot{\mathbf{x}} = \mathbf{D}(\mathbf{x}) \nabla \phi$$
   while the phase velocity (wavefront normal) is $\mathbf{n}_\phi = \frac{\nabla \phi}{\|\nabla \phi\|}$. Because $\mathbf{D}(\mathbf{x})$ has distinct eigenvalues, **the characteristic ray $\mathbf{v}_g$ is not parallel to $\nabla \phi$**. Information travels along oblique trajectories relative to the wavefront normal.
2. **Causality Violation in FMM**:
   A scalar priority queue sorts nodes purely by arrival order $\phi$. Under strong anisotropy, a characteristic ray can propagate into a cell from an angle that does not coincide with the smallest neighboring node values along Cartesian axes. Freezing a node based on priority-queue order causes FMM to freeze nodes prematurely, before the true upwind characteristic arrives.
3. **The Ordered Upwind Method (OUM) Bottleneck**:
   Sethian and Vladimirsky developed the *Ordered Upwind Method* (OUM) to restore causality by searching an expanded neighborhood stencil of radius proportional to the anisotropy ratio $\kappa = \lambda_{\max} / \lambda_{\min}$. However, OUM's computational complexity scales as $\mathcal{O}(\kappa^{d-1} N \log N)$, its stencil search logic is branching-heavy and cache-unfriendly, and it resists GPU parallelization.

---

## 3. Comparative Analysis of Solver Formulations

| Property | Fast Marching Method (FMM) | Fast Sweeping Method (FSM) | Fast Iterative Method (FIM) | Anisotropic Laplace / Potential | Vector Heat Method |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **PDE Class** | Hyperbolic non-linear | Hyperbolic non-linear | Hyperbolic non-linear | Elliptic linear | Parabolic $\to$ Elliptic |
| **Governing Equation** | $\|\nabla \phi\| = 1/f$ | $\nabla \phi^T \mathbf{D} \nabla \phi = 1$ | $\nabla \phi^T \mathbf{D} \nabla \phi = 1$ | $\nabla \cdot (\mathbf{D} \nabla \phi) = 0$ | $(I - t \nabla \cdot \mathbf{D} \nabla)u = \delta$; $\nabla^2 \phi = \nabla \cdot X$ |
| **Anisotropic Support** | ❌ Fails causality | ✅ Robust & exact | ✅ Robust & exact | ✅ Native | ✅ Native |
| **Data Structure** | Priority Queue (`BinaryHeap`) | Contiguous 3D Arrays | Active Block Lists | Sparse CSR Matrix | Sparse CSR Matrix |
| **Hardware Fit** | Serial CPU only | CPU SIMD / Rayon | GPU Compute (`wgpu`) | CPU/GPU Sparse Solvers | CPU/GPU Sparse Solvers |
| **Shock / Crease Behavior** | Forms shocks (ridges) | Forms shocks (ridges) | Forms shocks (ridges) | **Zero shocks ($C^\infty$ smooth)** | Mildly smoothed shocks |
| **Boundary Control** | Dirichlet seed only | Dirichlet seed only | Dirichlet seed only | Exact Dirichlet & Neumann | Dirichlet seed only |
| **Memory Footprint** | Low ($1\times$ grid + heap) | Low ($1\times$ grid) | Low ($1\times$ grid + queue) | High (sparse matrix + vectors) | High (sparse matrix + vectors) |

### Key Insight
- **For direct metric distance solving with anisotropic tensors**: The **Fast Sweeping Method (FSM)** is the optimal hyperbolic choice. It does not suffer from causality sorting issues because it sweeps all 8 coordinate octants systematically, resolving all characteristic directions in a small number of passes ($4 \dots 8$).
- **For shock-free, perfectly smooth non-planar layer surfaces**: The **Anisotropic Laplace Equation** is the optimal elliptic choice because it eliminates medial axis creases entirely by physical principle.

---

## 4. Steering Surface Tangency & Orthogonality via Tensors

To control the orientation of the level sets $\phi(\mathbf{x}) = c$ near a CAD surface with outward unit normal $\mathbf{n} \in \mathbb{R}^3$, we construct a local symmetric positive-definite conductivity tensor $\mathbf{D}(\mathbf{x}) \in \mathbb{R}^{3 \times 3}$.

Let $\mathbf{P}_n = \mathbf{n}\mathbf{n}^T$ be the projection operator onto the surface normal, and $\mathbf{P}_t = \mathbf{I} - \mathbf{n}\mathbf{n}^T$ be the projection operator onto the tangent plane. The local metric tensor is parameterized by a normal eigenvalue $\lambda_n$ and tangential eigenvalue $\lambda_t$:
$$\mathbf{D}(\mathbf{x}) = \lambda_n \mathbf{n}\mathbf{n}^T + \lambda_t (\mathbf{I} - \mathbf{n}\mathbf{n}^T)$$

```
        Near-Tangency (Layers parallel to surface)            Near-Orthogonality (Layers perpendicular to surface)
                     λ_n >> λ_t                                                  λ_t >> λ_n
                     
              Surface Normal n                                            Surface Normal n
                     ▲                                                           ▲
                     │ (Fast propagation)                                        │ (Slow propagation)
          ═══════════╪═══════════ CAD Surface                         ═══════════╪═══════════ CAD Surface
                     │                                                           │
          ───────────┼─────────── Isosurface φ = c+Δφ                            │    │    │
          ─────────────────────── Isosurface φ = c                               │    │    │ Isosurfaces φ = const
                                                                                 │    │    │ (Run perpendicular into surface)
```

### Case 1: Near-Tangency (Conforming Skin)
- We desire the order gradient $\nabla \phi$ to align with $\mathbf{n}$, so level sets lie parallel to the outer skin.
- Set $\lambda_n \gg \lambda_t$ (e.g. $\lambda_n = 4.0, \lambda_t = 0.25$).
- Fast propagation along $\mathbf{n}$ rapidly advances the phase front outward perpendicular to the surface, causing the level sets to drape smoothly along the contour.

### Case 2: Near-Orthogonality (Upright Intersections)
- We desire the level sets to strike the surface at right angles ($90^\circ$), ensuring standard layer bead cross-sections against vertical walls.
- Set $\lambda_t \gg \lambda_n$ (e.g. $\lambda_t = 4.0, \lambda_n = 0.5$).
- The front propagates rapidly along the surface tangent, flattening the level sets so their normals point along the surface rather than into it.

### Spatial Blending of Tensor Seeds
Given a set of surface seed regions $S_k$ with preferred normals $\mathbf{n}_k$ and target tensor configurations $\mathbf{D}_k$:
1. At each grid node $\mathbf{x}$, measure distance $d_k(\mathbf{x})$ to the boundary region.
2. Compute spatial weights $w_k(\mathbf{x}) = \exp\left(-\frac{d_k(\mathbf{x})^2}{2 \sigma^2}\right)$, where $\sigma$ corresponds to the configured skin depth.
3. Blend with the isotropic background metric $\mathbf{I}$:
   $$\mathbf{D}(\mathbf{x}) = (1 - W(\mathbf{x}))\,\mathbf{I} + \sum_{k} w_k(\mathbf{x})\,\mathbf{D}_k, \quad W(\mathbf{x}) = \min\left(1.0, \sum_k w_k(\mathbf{x})\right)$$

---

## 5. Architectural Implementation Plan: Anisotropic FSM (`AnisotropicFsmOrderField`)

To allow rigorous benchmarking against the baseline `EikonalOrderField`, the Fast Sweeping solver will be implemented as a completely separate, standalone order field in `crates/manifold-fidget/src/fsm.rs`.

### Phase 1: Local Anisotropic Simplex Quadratic Solver (`fsm_simplex.rs`)
- **Location**: `crates/manifold-fidget/src/fsm_simplex.rs`
- **Objective**: Given a grid node $\mathbf{x}_0$ and known neighbor values at $\mathbf{x}_1, \mathbf{x}_2, \mathbf{x}_3$ within a Cartesian simplex, solve the local anisotropic Eikonal update:
  $$\min_{\boldsymbol{\lambda} \ge 0, \sum \lambda_i = 1} \left( \sum_{i=1}^3 \lambda_i \phi_i + \sqrt{\Delta \mathbf{x}(\boldsymbol{\lambda})^T \mathbf{M}(\mathbf{x}_0) \Delta \mathbf{x}(\boldsymbol{\lambda})} \right)$$
  where $\mathbf{M} = \mathbf{D}^{-1}$.
- **Methodology**:
  - Implement Legendre-Fenchel dual formulation or direct projection onto the 1D, 2D, and 3D simplex faces.
  - Test for upwind causality: if the optimal characteristic ray lands outside the simplex face, project $\boldsymbol{\lambda}$ onto the sub-faces (edges and vertices) to guarantee monotonicity.
- **Unit Verification**:
  - Analytical verification against tilted planar waves and rotated ellipsoidal distance metrics.

### Phase 2: 3D Multi-Directional Gauss-Seidel Sweeping Engine (`fsm.rs`)
- **Location**: `crates/manifold-fidget/src/fsm.rs`
- **Data Structure**:
  ```rust
  pub struct AnisotropicFsmOrderField {
      min_corner: DVec3,
      dims: [usize; 3],
      h: f64,
      /// Node distances ϕ(x, y, z)
      distances: Vec<f64>,
      /// Precomputed symmetric metric tensor M(x) = D(x)^-1 (6 floats per node)
      metric_tensors: Vec<[f32; 6]>,
      /// Gradients ∇ϕ for Hermite continuous interpolation
      gradients: Vec<DVec3>,
  }
  ```
- **Sweeping Order**:
  Execute 8 alternating Gauss-Seidel sweeps over the volume grid:
  1. $x: 0 \to N_x-1, \quad y: 0 \to N_y-1, \quad z: 0 \to N_z-1$
  2. $x: N_x-1 \to 0, \quad y: 0 \to N_y-1, \quad z: 0 \to N_z-1$
  3. $x: 0 \to N_x-1, \quad y: N_y-1 \to 0, \quad z: 0 \to N_z-1$
  4. $x: N_x-1 \to 0, \quad y: N_y-1 \to 0, \quad z: 0 \to N_z-1$
  5. $x: 0 \to N_x-1, \quad y: 0 \to N_y-1, \quad z: N_z-1 \to 0$
  6. $x: N_x-1 \to 0, \quad y: 0 \to N_y-1, \quad z: N_z-1 \to 0$
  7. $x: 0 \to N_x-1, \quad y: N_y-1 \to 0, \quad z: N_z-1 \to 0$
  8. $x: N_x-1 \to 0, \quad y: N_y-1 \to 0, \quad z: N_z-1 \to 0$
- **Occupancy & Boundary Conditions**:
  - Non-solid voxels (`!is_solid(p)`) remain at `f64::INFINITY`.
  - Bed seed voxels are pinned at $\phi = 0.0$.
  - Terminate when $\max |\phi^{(m+1)} - \phi^{(m)}| < \epsilon_{\text{tol}}$ or after a fixed iteration cap (e.g. 6 sweeps).

### Phase 3: Parallelization Strategy
- **Diagonal Hyperplane Wavefront Sweeping**:
  - In a standard Gauss-Seidel sweep, updates at $(i, j, k)$ depend only on $(i-1, j, k)$, $(i, j-1, k)$, and $(i, j, k-1)$.
  - Consequently, all nodes satisfying $i + j + k = c$ (a diagonal plane through the grid) are mutually independent.
  - We can parallelize across the slice $i + j + k = c$ using Rayon:
    ```rust
    for plane_sum in 0..max_sum {
        let nodes = get_diagonal_plane_indices(plane_sum);
        nodes.par_iter().for_each(|&(x, y, z)| {
            update_node(x, y, z);
        });
    }
    ```
  - This provides lock-free, race-free parallelism while preserving exact Gauss-Seidel convergence.

### Phase 4: Integration with Slicer Pipeline & Settings
- **Configuration**:
  Add `OrderFieldType::AnisotropicFsm` alongside `OrderFieldType::ConformalEikonal` in `manifold-core`.
- **Side-by-Side Evaluation**:
  - Slicing engine can instantiate either `EikonalOrderField` or `AnisotropicFsmOrderField`.
  - Expose selector in GUI settings panel under "Non-Planar Slicing".
  - Compare contour smoothness, layer thickness uniformity, and absence of crease defects across real benchmark prints (`pug_v4_l_sop_85mm.stl`).
