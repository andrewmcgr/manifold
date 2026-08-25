# Retraction, Scarf Joints & Thermodynamic Fluid Dynamics

Manifold provides advanced extrusion control including perimeter wipe moves, non-planar scarf joint seams, and a unified thermodynamic & non-Newtonian fluid dynamics engine.

---

## Retraction & Seam Parameters

```json
{
  "retraction_length": 0.25,
  "retraction_speed": 1500.0,
  "unretract_speed": 1500.0,
  "unretract_extra_length": 0.0,
  "use_firmware_retraction": false,
  "scarf_joint_enabled": true,
  "scarf_joint_length": 3.0,
  "wipe_enabled": true,
  "wipe_distance": 1.0,
  "pre_retract_taper_distance": 1.0,
  "z_hop_enabled": true,
  "z_hop_height": 0.40
}
```

- **Non-Planar Scarf Joint Seams (`scarf_joint_enabled`)**: Over the configured `scarf_joint_length` ($3.0\text{ mm}$), the lead-in segment ramps extrusion flow from $0 \to 100\%$. After traversing the closed loop, the nozzle continues for an overlapping $3.0\text{ mm}$ tail ramping flow from $100\% \to 0$, eliminating vertical seam lines.
- **Wipe Moves (`wipe_enabled`)**: Appends an unextruded nozzle move along the perimeter/interior before travel lifts to break the capillary droplet bridge.
- **Two-Line-Width Travel Clearance**: Travel moves automatically enforce $2 \times \text{line\_width}$ ($0.8\text{ mm}$) clearance from printed material, departing and arriving along surface normals into open air.

---

## Unified Thermodynamic & Non-Newtonian Fluid Model (`fluid_dynamics`)

```json
{
  "fluid_dynamics": {
    "pa_calibration_low": [0.045, 2.0],
    "pa_calibration_high": [0.030, 15.0],
    "heater_block_temp_c": 240.0,
    "reference_temp_c": 240.0,
    "max_fan_temp_drop_c": 8.0,
    "ooze_time_constant_ref_s": 1.5,
    "ooze_max_length_ref_mm": 0.30,
    "static_retraction_mm": 0.25,
    "max_retraction_mm": 1.5,
    "pa_deadband": 0.10
  }
}
```

### 1. 2-Point Power-Law Calibration (Shear Thinning)
Non-Newtonian polymer melt viscosity drops under higher shear strain rates:

$$\alpha = -\frac{\ln(C_{\text{PA\_high}}) - \ln(C_{\text{PA\_low}})}{\ln(Q_{\text{high}}) - \ln(Q_{\text{low}})}, \quad C_{\text{PA}}(Q) = C_{\text{PA\_zero}} \cdot Q^{-\alpha}$$

### 2. Forced Convective Tip Cooling & Arrhenius Viscosity
Cooling fan air stream drops effective polymer tip temperature:

$$T_{\text{effective}} = T_{\text{block}} - \Delta T_{\text{max\_fan}} \cdot F^{0.6}, \quad C_{\text{PA\_dynamic}} = C_{\text{PA}}(Q) \cdot e^{-0.02 \cdot (T_{\text{effective}} - T_{\text{ref}})}$$

### 3. Adaptive Junction-Velocity Retraction ($L_{\text{retract}}$)
Elastic melt-zone pressure is relieved using filament feed velocity:

$$L_{\text{retract}} = C_{\text{PA}} \cdot \left(\frac{Q_{\text{exit}}}{A_{\text{filament}}}\right) + L_{\text{static}}$$

### 4. Time-Dependent Thermal Ooze Recovery ($L_{\text{unretract}}$)
Over travel duration $t_{\text{travel}}$, re-primes oozed polymer:

$$L_{\text{unretract}} = L_{\text{retract}} + L_{\text{max\_ooze}} \cdot \left(1 - e^{-t_{\text{travel}} / \tau}\right)$$
