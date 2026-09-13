# Pressure Advance and Flow Compensation Guide

This guide explains Manifold's pressure advance and flow compensation pipeline, why multiple compensations can interact to cause overextrusion or underextrusion, and how to calibrate each feature systematically.

---

## 1. Overview of the Compensation Pipeline

Manifold includes four distinct flow and pressure compensation systems in `crates/manifold-core`:

1. **Firmware / Slicer Pressure Advance (PA)**
   - **Firmware PA (`pressure_advance`):** Emits `SET_PRESSURE_ADVANCE ADVANCE=<val>` for Klipper to manage on the MCU stepper planner.
   - **Slicer PA (`enable_slicer_pressure_advance`):** Discretizes extruding moves into small chords and injects positional $E$ advance along acceleration/deceleration trapezoids in G-code, disabling firmware PA.
2. **Transient Pressure Compensation (`enable_transient_pressure_compensation`)**
   - Models viscous pressure buildup and exponential relaxation in the melt zone over time.
   - Dampens nominal extruded volume down to a minimum multiplier (`transient_pressure_min_multiplier`) during rapid accelerations and short strokes.
3. **Corner Flow Compensation (`enable_corner_flow_compensation`)**
   - Analyzes toolpath direction changes and deducts overlapping volume deposited on the inside radius of corners and square-corner velocity arcs.
4. **Viscoelastic Die Swell Compensation (`swell_ratio_low`, `swell_ratio_high`)**
   - Accounts for polymer relaxation after exiting the nozzle orifice.
   - Scales extruded volume inversely by $1 / B(Q, T)$, where $B$ is the swell ratio.

---

## 2. Why Compensations Can Fight Each Other

When multiple compensation systems are active at the same time, their flow adjustments multiply geometrically:

$$\text{Final Flow Multiplier} = M_{\text{transient}} \times M_{\text{corner}} \times M_{\text{swell}}$$

For example, on a sharp corner during rapid acceleration:

- Corner flow compensation deducts $20\text{--}35\%$ of volume ($M_{\text{corner}} \approx 0.65\text{--}0.80$).
- Transient pressure compensation reduces volume down to $M_{\text{min}}$ ($0.80$).
- Viscoelastic die swell reduces volume by $1 / B$ ($\approx 0.85\text{--}0.95$).

Combined, the effective flow can drop below $50\%$ of nominal volume, causing severe underextrusion, gaps in perimeters, and weak infill bonding. Conversely, when the nozzle unretracts or transitions into steady cruise speed, accumulated pressure advance can surge, producing blobs and overextrusion.

---

## 3. Extrudate Die Swell Demystified

### The Physical Phenomenon

As molten polymer is forced through the narrow nozzle capillary under high shear stress, long-chain polymer molecules are stretched. The moment the melt exits the orifice into ambient air, these elastic stresses relax, causing the extrudate to expand in cross-section ($B = D_{\text{extrudate}} / D_{\text{orifice}} > 1.0$).

### Why Swell Compensation Causes Confusion

In FDM printing, the extrudate is not unconstrained in all directions; it is flattened between the nozzle flat and the previous layer at fixed layer height $h$. Because the vertical dimension is physically constrained, die swell primarily pushes material **laterally in $XY$** (increasing effective line width) rather than increasing volumetric density.

If the slicer reduces volumetric flow by $1 / B(Q, T)$ to counteract swell, but the printer's baseline extrusion multiplier was already tuned at standard print speeds, the slicer will starve the hotend at high flow rates, causing matte, weak, or underextruded infill and perimeters.

**Recommendation:** Keep `swell_ratio_low` and `swell_ratio_high` at `1.0` (disabled/neutral) unless you are performing specialized dimensional calibration for free-air overhang beads.

---

## 4. Slicer PA vs. Firmware PA

You should not use both Slicer PA and Firmware PA simultaneously:

- **Use Firmware PA (Recommended for Klipper):**
  - Smooth stepper-level execution on the MCU without inflating G-code size.
  - Set `config.pressure_advance` to your tuned value (e.g. `0.035`).
  - Keep `enable_slicer_pressure_advance = false`.
- **Use Slicer PA:**
  - Designed for firmwares lacking native pressure advance or for non-planar geometries where isosurface curvature introduces continuous non-linear acceleration that standard linear MCU planners cannot anticipate.

---

## 5. Step-by-Step Calibration Procedure

To achieve predictable, consistent extrusion, calibrate each layer in isolation:

### Step 1: Baseline Flow (Zero Compensations)

1. In your profile, disable secondary compensations:
   - `enable_slicer_pressure_advance: false`
   - `enable_transient_pressure_compensation: false`
   - `enable_corner_flow_compensation: false`
   - `swell_ratio_low: 1.0`, `swell_ratio_high: 1.0`
2. Print a single-wall calibration cube at steady perimeter speed (e.g. 60 mm/s).
3. Measure wall thickness with calipers and tune `extrusion_multiplier` until a 0.40 mm nominal wall measures exactly 0.40 mm.

### Step 2: Calibrate Firmware Pressure Advance

1. Run a standard Klipper pressure advance tuning tower or line test.
2. Enter the tuned coefficient into `pressure_advance` (e.g. `0.025` to `0.040`).
3. Verify that corners are square and seam blobs are minimal at your primary print speed.

### Step 3: Calibrate Dynamic Fluid Dynamics (Optional)

If your printer runs across a wide flow range (e.g. 30 mm/s outer walls up to 250 mm/s infill):

1. Measure PA at low flow ($Q_1 \approx 5\text{ mm}^3/\text{s}$, e.g. outer wall) $\to C_{\text{PA\_low}}$.
2. Measure PA at high flow ($Q_2 \approx 20\text{--}30\text{ mm}^3/\text{s}$, e.g. rapid infill) $\to C_{\text{PA\_high}}$.
3. Configure `pa_calibration_low` and `pa_calibration_high` under `fluid_dynamics`.

### Step 4: Corner Flow Compensation (Fine-Tuning Only)

If outside corners still bulge slightly due to deceleration down to `square_corner_velocity`:

1. Enable `enable_corner_flow_compensation: true`.
2. Start with `corner_flow_compensation_ratio: 0.5` to `0.8`.
3. Avoid values greater than `1.0` (values above `1.0` deduct excessive volume and cause corner holes).

### Step 5: Transient Pressure Compensation (High-Speed Toolheads Only)

If printing with high accelerations ($>5,000\text{ mm}/\text{s}^2$) where rapid direction changes cause brief overextrusion at the start of sharp infill strokes:

1. Enable `enable_transient_pressure_compensation: true`.
2. Set `transient_pressure_min_multiplier` to a conservative value (`0.85` to `0.90`).
3. Set `transient_pressure_beta` to `1.0`.

---

## 6. Summary of Settings

| Setting | Purpose | When to Enable | Recommended Value |
| --- | --- | --- | --- |
| `pressure_advance` | Primary dynamic pressure advance coefficient. | Always on Klipper / Voron. | 0.02 – 0.05 s |
| `enable_slicer_pressure_advance` | Discretizes moves into chords with advance in G-code. | Non-planar layers or firmware without native PA. | False (use firmware PA) |
| `enable_corner_flow_compensation` | Deducts volume on inside of sharp corners. | Corners bulge despite tuned PA. | Ratio: 0.5 – 0.8 |
| `corner_flow_compensation_ratio` | Scales corner volume deduction. | When corner compensation is enabled. | 0.5 – 0.8 (avoid > 1.0) |
| `enable_transient_pressure_compensation` | Dampens flow on rapid accelerations and short moves. | High-speed, high-accel machines. | Min multiplier: 0.85 – 0.90 |
| `transient_pressure_min_multiplier` | Floor for transient flow reduction. | When transient compensation is enabled. | 0.85 – 0.90 |
| `transient_pressure_beta` | Nonlinearity exponent for transient pressure decay. | Tuning transient decay sensitivity. | 1.0 |
| `swell_ratio_low` / `high` | Models extrudate die swell expansion. | Experimental / free-air extrusions. | 1.0 (disabled) |
