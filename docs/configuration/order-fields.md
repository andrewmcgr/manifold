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
- Useful for cylindrical, vase-mode, or rotational parts.

### 3. `Eikonal` (Geodesic Distance Field Slicing)
- Propagates a geodesic distance field $\Phi(x, y, z)$ monotonically upward from the print bed through the 3D solid interior using a narrow-band **Fast Marching Method (FMM)** on an auto-sized voxel grid.
- Layers naturally deform to follow the part's external geometry, producing smooth non-planar curved surfaces with superior strength along load lines.

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

## Conformal Top Surfaces (`eikonal_conform_top_surfaces`)

```json
{
  "eikonal_conform_top_surfaces": true
}
```

When enabled, a secondary downward geodesic march measures distance from upward-facing exterior surfaces ($d_{\text{top}}$). Within the top skin region, order values smoothly warp to lie parallel to exterior roofs and decks without creating multi-tier inversions.
