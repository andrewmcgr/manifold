# Order Fields & Toolhead Clearance Envelopes

Unlike planar slicers that slice models with flat horizontal planes ($Z = k \cdot h$), Manifold evaluates a 3D scalar order field $\Phi(x, y, z)$. Slicing extracts non-planar cross-sections along the level-sets $\Phi(x, y, z) = T_k$.

---

## Order Field Types (`order_field`)

```json
{
  "order_field": "Eikonal"
}
```

### 1. `Height` (Planar Slicing)
- Standard horizontal planar slicing where $\Phi(x, y, z) = z$.
- Used for planar benchmarking or traditional flat layer printing.

### 2. `Conical` (Conical Non-Planar Slicing)
- Conical slicing surface inclined by an apex angle $\theta$ radiating outward from a central build axis.
- Configurable via `order_field_apex` $(X, Y, Z)$ and `order_field_slope`.

### 3. `Eikonal` (Geodesic Distance Field Slicing)
- Propagates a geodesic distance field $\Phi(x, y, z)$ monotonically upward from the print bed through the 3D solid interior using a narrow-band **Fast Marching Method (FMM)** on an auto-sized voxel grid.
- Layers naturally deform to follow the part's external geometry, producing smooth non-planar curved surfaces with superior strength along load lines.

---

## Geodesic Surface Order Lower Bound (`eikonal_surface_order_weight`)

```json
{
  "eikonal_surface_order_weight": 1.0
}
```

To eliminate interior local minima near curved skins, Manifold precomputes a **Surface Eikonal** arrival grid $\Phi_{\text{surface}}$ across the triangle boundary in parallel using Rayon. During 3D volume propagation, this surface arrival time is applied as a lower bound via a $C^2$-continuous smooth-max function:

$$\Phi_{\text{effective}} = \text{smooth\_max}\left(\Phi_{\text{bulk}}, w \cdot \Phi_{\text{surface}}, r_{\text{smooth}}\right)$$

with smoothing radius $r_{\text{smooth}} = 2\sqrt{2} \cdot h_{\text{grid}}$, guaranteeing $C^2$ smoothness and eliminating sharp derivative creases.

---

## Bilateral Conformal Surface Slicing

Manifold supports dual-sided conformal slicing, warping internal isosurfaces to match both upward-facing roofs and downward-facing overhangs:

```json
{
  "eikonal_conform_top_surfaces": true,
  "eikonal_conformal_max_angle_deg": 45.0,
  "eikonal_conform_bottom_surfaces": true,
  "eikonal_conformal_bottom_max_angle_deg": 30.0,
  "eikonal_conformal_skin_depth_mm": 1.2,
  "eikonal_enforce_monotonic_growth": true
}
```

### 1. Top Surface Conforming (`eikonal_conform_top_surfaces`)
- Warps isosurfaces parallel to upward-facing exterior surfaces within the configured detach angle ($\theta \le \theta_{\text{top}}$).
- Blend weights transition smoothly into the bulk interior using a $C^2$ quintic smootherstep function:
  $$S(t) = 6t^5 - 15t^4 + 10t^3$$

### 2. Bottom Surface Conforming (`eikonal_conform_bottom_surfaces`)
- For downward-facing overhang surfaces ($\theta \le \theta_{\text{bottom}}$), Manifold constructs printable per-node surface anchors:
  $$\Phi_{\text{anchor}} = S + d$$
  where $S$ is the marched bulk order at the surface foot point and $d$ is the inward normal distance.
- Ensures sloped undersides advance as printable wavefronts outward from supported perimeters with uniform normal thickness, preventing mid-air floating loops.

### 3. Bilateral Fixed-Point Iteration & Convergence
- Top and bottom conformal constraints are solved simultaneously using a symmetric reverse Lipschitz relaxation pass followed by bilateral upward-downward fixed-point iteration until convergence.

### 4. Vertical Monotonicity (`eikonal_enforce_monotonic_growth`)
- Enforces strictly increasing layer order along vertical columns:
  $$\frac{\partial\Phi}{\partial z} \ge 0.15$$
  preventing downward stalls and guaranteeing monotonic bottom-to-top (+Z) build progression.

---

## Toolhead Clearance Envelope (`eikonal_slope_profile`)

To ensure that the non-planar nozzle does not collide with already-printed material on steep peaks, `Machine::eikonal_slope_profile` specifies the physical toolhead clearance envelope as a series of $(X, Z)$ coordinate points:

$$X = \text{radial distance from nozzle center (mm)}, \quad Z = \text{clearance height above nozzle tip (mm)}$$

```json
{
  "machine": {
    "eikonal_slope_profile": [
      [0.0, 0.0],
      [15.0, 5.0],
      [35.0, 20.0],
      [50.0, 40.0]
    ]
  }
}
```

### Euclidean Slope Profile Relaxation

During Eikonal field generation, Manifold applies a **3D Euclidean Lipschitz Relaxation** pass across all grid nodes (including open air gaps between disconnected model features):

$$|\Phi(A) - \Phi(B)| \le \Delta r \cdot \tan\left(\theta_{\text{max}}(Z)\right)$$

This mathematical constraint guarantees that no feature anywhere on the bed can advance ahead of an adjacent feature by more than the gantry's physical clearance angle, preventing toolhead collisions.

---

## Trajectory Slope Volumetric Compensation

When printing along sloped 3D trajectories, toolpath segments elongate in 3D space. Manifold dynamically scales the required extrusion volume by the trajectory slope cosine:

$$\Delta E = \frac{A_{\text{bead}} \cdot L_{\text{3D}} \cdot \cos(\theta_{\text{slope}})}{A_{\text{filament}}}$$

where $\cos(\theta_{\text{slope}}) = \frac{\sqrt{\Delta x^2 + \Delta y^2}}{L_{\text{3D}}}$, maintaining constant horizontal layer thickness and preventing over-extrusion on steep non-planar inclines.
