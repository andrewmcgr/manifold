# End-of-Print Pressure Bleed Wipe and Clearance Move Design

**Status:** Approved Design
**Date:** 2025-05-13
**Target Crates:** `crates/manifold-core`, `crates/manifold-gui`

---

## 1. Overview & Goals

At the end of a 3D print, the nozzle remains pressurized with molten polymer. If the toolhead stops and lifts immediately or waits for `end_gcode` at the final extrusion coordinate, this residual pressure bleeds out onto the finished print, causing a prominent blob, stringing, or surface scar on the topmost feature.

This feature introduces an automated end-of-print pressure-bleed wipe and non-colliding clearance sequence to Manifold:

- **Reverse Wipe along Final Bead:** Moves backwards along the final extruded bead vector with $\Delta E = 0$, allowing residual melt zone pressure to bleed down to ambient.
- **Fluid Model Pressure Sizing:** Derives the wipe distance dynamically from the fluid dynamics model (volumetric flow rate and dynamic Pressure Advance) so that residual pressure is reduced as low as reasonably achievable.
- **Clean Meniscus Shear Step:** Steps cleanly outward using the CAD surface normal and isosurface normal to sever the molten meniscus in shear rather than tension.
- **Arrangement-Aware Clearance Move:** Travels horizontally outside the printed object footprint by a safe distance derived from machine arrangement clearance ($\frac{1}{2} \times \text{arrangement\_clearance}$), ensuring non-collision with adjacent parts on auto-arranged beds.
- **Z-Lift with 3D Build Volume Clamping:** Lifts the nozzle vertically above the final layer while strictly clamping against the machine's maximum build volume in $X$, $Y$, and $Z$ ($Z \le Z_{\text{max}}$).
- **Clean Retraction & G-code Handover:** Triggers the final full retraction before clearance travel, leaving the nozzle empty and fully clear of the print before executing user `end_gcode`.

---

## 2. Mathematical Formulation & Physical Model

### 2.1. Residual Pressure & Wipe Distance Calculation ($L_{\text{wipe}}$)

Under the dynamic fluid model (see `RETRACTION_AND_PA.md`), the pressurized volume stored in the elastic melt zone at a path stop is:
$$V_{\text{residual}} = C_{\text{PA}} \cdot Q = C_{\text{PA}} \cdot (A_{\text{bead}} \cdot v)$$
where:

- $Q$ is volumetric flow rate ($\text{mm}^3/\text{s}$).
- $C_{\text{PA}}$ is the dynamic pressure advance coefficient ($\text{s}$), evaluated at the final bead's flow rate, nozzle temperature, and part-cooling fan speed.
- $A_{\text{bead}} = W \cdot H$ is the cross-sectional area of the final line ($\text{mm}^2$).
- $v$ is the terminal toolhead velocity ($\text{mm}/\text{s}$).

The distance along the reverse bead trajectory required to decompress this volume is:
$$L_{\text{calc}} = C_{\text{PA}} \cdot v + L_{\text{static\_wipe}}$$
where $L_{\text{static\_wipe}} = 1.0 \times \text{nozzle\_diameter}$ (e.g. 0.4 mm).

To ensure safety across all geometries:
$$L_{\text{wipe}} = \text{clamp}(L_{\text{calc}},\, L_{\text{min}},\, L_{\text{max}})$$

- $L_{\text{min}} = 0.5 \times \text{nozzle\_diameter}$ (e.g. 0.2 mm).
- $L_{\text{max}} = \min(5.0\text{ mm},\, \text{length of final extruded segment})$.
- If the final segment is shorter than $L_{\text{calc}}$, $L_{\text{wipe}}$ clamps to the segment length so the nozzle never reverses past the segment's starting vertex into air.
- If fluid dynamics is disabled, $C_{\text{PA}}$ falls back to `config.pressure_advance.unwrap_or(0.0)`.

### 2.2. Reverse Trajectory

Let the final extruded segment run from $P_{\text{prev}}$ to $P_{\text{final}}$:
$$\vec{u}_{\text{reverse}} = \frac{P_{\text{prev}} - P_{\text{final}}}{\|P_{\text{prev}} - P_{\text{final}}\|}$$
$$P_{\text{wipe}} = P_{\text{final}} + \vec{u}_{\text{reverse}} \cdot L_{\text{wipe}}$$
Extrusion during this move is strictly $\Delta E = 0$.

---

## 3. Clearance Trajectory & Multi-Axis Clamping

### 3.1. Outward Meniscus Shear Vector

From $P_{\text{wipe}}$, the nozzle makes a short outward departure step along the combined CAD surface normal $\vec{n}_{\text{cad}}$ and isosurface normal $\vec{n}_{\text{iso}}$:
$$\vec{d}_{\text{shear}} = (\vec{n}_{\text{cad}} \cdot 1.5 + \vec{n}_{\text{iso}} \cdot 0.5).\text{normalize}()$$
$$P_{\text{shear}} = P_{\text{wipe}} + \vec{d}_{\text{shear}} \cdot (1.5 \times \text{nozzle\_diameter})$$

### 3.2. Target Clearance Point ($P_{\text{clear}}$)

The toolhead then moves outward into open air:

1. **XY Distance from Object:**
   $$D_{\text{clear}} = \frac{1}{2} \times \text{machine.arrangement\_clearance}()$$
   Defaulting to $5.0\text{ mm}$ (half of the default $10.0\text{ mm}$ inter-object spacing).
2. **Vertical Lift ($\Delta Z$):**
   Defaults to $2.0\text{ mm}$ (`config.end_of_print_clearance_z_lift`).
3. **Build Volume Clamping ($X, Y, Z$):**
   Let the machine's AABB bounding box be $[(X_{\text{min}}, Y_{\text{min}}, Z_{\text{min}}), (X_{\text{max}}, Y_{\text{max}}, Z_{\text{max}})]$.
   With a safety boundary margin $\epsilon = 0.5\text{ mm}$:
   $$P_{\text{clear}} = \left( \text{clamp}(P_{\text{shear}, x} + d_{\text{shear}, x} \cdot D_{\text{clear}},\, X_{\text{min}} + \epsilon,\, X_{\text{max}} - \epsilon),\, \text{clamp}(P_{\text{shear}, y} + d_{\text{shear}, y} \cdot D_{\text{clear}},\, Y_{\text{min}} + \epsilon,\, Y_{\text{max}} - \epsilon),\, \min(P_{\text{shear}, z} + \Delta Z,\, Z_{\text{max}} - \epsilon) \right)$$
   If $P_{\text{shear}, z} \ge Z_{\text{max}} - \epsilon$, the vertical lift $\Delta Z$ is suppressed to $0.0$, performing an XY-only exit to avoid exceeding the mechanical Z axis limit.

### 3.3. Obstacle Collision Avoidance

Before finalizing the move to $P_{\text{clear}}$, the chord is checked via `travel_chord_is_blocked`:

- If non-planar peaks or neighboring tall objects obstruct the horizontal path to $P_{\text{clear}}$, the toolhead lifts to $\min(\text{max\_layer\_z} + \text{clearance},\, Z_{\text{max}} - \epsilon)$ before moving horizontally.

---

## 4. Pipeline Architecture & Implementation

### 4.1. Data Flow in `manifold-core`

1. `plan_toolpaths_with_progress` in `crates/manifold-core/src/lib.rs` completes layer toolpaths.
2. Calls `toolpath::append_end_of_print_wipe_and_clearance(&mut paths, objects, machine, config)`:
   - Finds the final extruding `Path` and its final extruding segment.
   - Computes $L_{\text{wipe}}$, $P_{\text{wipe}}$, $P_{\text{shear}}$, and $P_{\text{clear}}$.
   - Appends:
     - `Segment { kind: MoveKind::Wipe, .. }` from $P_{\text{final}}$ to $P_{\text{wipe}}$ with $\Delta E = 0$.
     - `Segment { kind: MoveKind::Travel, .. }` from $P_{\text{wipe}}$ to $P_{\text{shear}}$ with $\Delta E = 0$.
     - `Segment { kind: MoveKind::Travel, .. }` from $P_{\text{shear}}$ to $P_{\text{clear}}$ with $\Delta E = 0$.
3. `toolpath::validate_within_bounds(&paths, &workspace.machine.build_volume)` validates that all generated waypoints lie inside the build volume.

### 4.2. G-code Emission in `crates/manifold-core/src/gcode.rs`

1. Slicer emits toolpaths sequentially.
2. The final extruding segment finishes.
3. The reverse wipe segment is emitted as `G1 X.. Y.. Z.. F{wipe_speed}` with no `E` parameter.
4. Transitioning into the clearance `Travel` segment triggers the standard retraction logic:
   - Emits `G10` (firmware retraction) or explicit `G1 E-{len} F{speed}`.
   - Sets `retracted = true`.
5. Emits travel move `G0` / `G1` to $P_{\text{clear}}$ with `F{travel_speed}`.
6. When the toolpath loop completes:
   - Because `retracted` is already `true`, the loop tail avoids emitting a redundant second retraction.
   - `config.end_gcode` is emitted with the nozzle safely clear of the printed model and within build volume limits.

---

## 5. Configuration Schema in `SlicerConfig`

In `crates/manifold-core/src/lib.rs`:

```rust
pub struct SlicerConfig {
    // ...
    /// Whether to perform an automated pressure-bleed wipe and clearance move at the end of the print.
    /// Defaults to true.
    #[serde(default = "default_true")]
    pub end_of_print_wipe_enabled: bool,

    /// Optional manual override for the end wipe distance in mm.
    /// When None, dynamically derived from the fluid pressure advance model.
    #[serde(default)]
    pub end_of_print_wipe_distance: Option<f64>,

    /// Vertical Z lift in mm applied during the final clearance move. Defaults to 2.0 mm.
    #[serde(default)]
    pub end_of_print_clearance_z_lift: Option<f64>,
}
```

---

## 6. Verification & Test Plan

1. **Fluid Dynamics Sizing Tests (`crates/manifold-core/src/toolpath.rs`)**:
   - Assert higher junction speed and higher PA yield longer $L_{\text{wipe}}$.
   - Assert $L_{\text{wipe}}$ is clamped when the final bead length is very short.
2. **Clearance Geometry & Build Volume Boundary Tests**:
   - Single object: assert clearance moves out by $\frac{1}{2}\times\text{arrangement\_clearance}$ and applies $+2.0\text{ mm}$ Z lift.
   - Max Z boundary: place an object reaching $Z_{\text{max}} - 0.5\text{ mm}$; assert clearance point does not exceed $Z_{\text{max}}$ and passes `validate_within_bounds`.
   - Max XY boundary: place an object at $X_{\text{max}}$; assert $P_{\text{clear}}$ clamps to $X_{\text{max}} - 0.5\text{ mm}$ and does not error out of bounds.
3. **G-code Emission Sequence Tests (`crates/manifold-core/src/gcode.rs`)**:
   - Verify sequence: final extrusion $\to$ unextruded reverse wipe $\to$ retraction $\to$ clearance travel $\to$ no duplicate retraction $\to$ `end_gcode`.
4. **GUI Integration Tests**:
   - Verify settings UI exposes the toggle and serializes in `Profile`.
