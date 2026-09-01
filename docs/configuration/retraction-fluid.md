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
  "scarf_joint_length": 8.0,
  "scarf_joint_steps": 9,
  "scarf_joint_start_height_fraction": 0.10,
  "scarf_joint_flow_ratio": 0.90,
  "wipe_enabled": true,
  "wipe_distance": 1.0,
  "pre_retract_taper_distance": 1.0,
  "min_travel_for_retract": 1.5,
  "z_hop_enabled": true,
  "z_hop_height": 0.40,
  "travel_order_optimization_enabled": true,
  "travel_collision_avoidance_enabled": true,
  "z_travel_penalty": 8.0
}
```

- **Non-Planar Scarf Joint Seams (`scarf_joint_enabled`)**: Over the configured `scarf_joint_length` ($8.0\text{ mm}$), the lead-in segment ramps extrusion flow and slice-normal layer height from `scarf_joint_start_height_fraction` ($10\%$) to $100\%$ across `scarf_joint_steps` ($9$ discrete steps), forming a bottom wedge offset along the negative slice normal. After traversing the closed loop, the nozzle continues for an overlapping $8.0\text{ mm}$ tail at nominal surface height, ramping flow from $90\% \to 0\%$ (top wedge). The combined volume is scaled by `scarf_joint_flow_ratio` (default $0.90$, i.e. $90\%$ combined volume) to prevent seam bulging on curved or non-planar perimeters.
- **Minimum Travel for Retract (`min_travel_for_retract`)**: Minimum travel distance ($1.5\text{ mm}$ default) required to trigger a retraction move. When scarf joints are enabled, the scarf joint length ($8.0\text{ mm}$) is automatically added to this threshold ($1.5 + 8.0 = 9.5\text{ mm}$), preventing unnecessary retract/unretract cycles and filament grinding when the nozzle transitions directly from the end of one perimeter scarf joint to the start of an adjacent wall seam.
- **Wipe Moves (`wipe_enabled`)**: Appends an unextruded nozzle move along the perimeter/interior before travel lifts to break the capillary droplet bridge.
- **Travel Collision Avoidance (`travel_collision_avoidance_enabled`)**: Detects travel chords that cross solid material and plans collision-free 3D detour paths in open air using parallel A* search.
- **Order-Aware Temporal Geometry**: Collision checking is strictly time-aware: a point $p$ only obstructs travel if it is inside the solid mesh (`MeshSdf` value $< \text{clearance}$) *and* its scalar order-field value satisfies $\Phi(p) \le T_{\text{current}} + \epsilon$ (material that has already been deposited on the bed). Future unprinted geometry ($\Phi(p) > T_{\text{current}}$) is recognized as open physical air, allowing direct straight-line travel without artificial detours around objects that do not yet exist.
- **Two-Line-Width Travel Clearance**: Travel moves automatically enforce $2 \times \text{line width}$ ($0.8\text{ mm}$) clearance from printed material, departing and arriving along surface normals into open air.
- **Minimum Bed Clearance Floor**: Travel waypoints in open air maintain $Z \ge \min(\text{start}.z, \text{end}.z)$ with an absolute floor of $0.5 \times \text{first\_layer\_height}$, preventing the toolhead from diving into the build plate or dragging across textured bed clips.
- **Kinematic Z-Travel Cost Penalty (`z_travel_penalty`)**: Travel order optimization and A* obstacle routing calculate movement cost using the anisotropic kinematic metric:
  $$\text{cost}(a, b) = \sqrt{(b_x - a_x)^2 + (b_y - a_y)^2 + (\text{penalty}_z \cdot (b_z - a_z))^2}$$
  where $\text{penalty}_z$ dynamically incorporates the printer's Z vs. XY axis speed ratio ($v_{xy} / v_z$) and acceleration ratio ($\sqrt{a_{xy} / a_z}$) from `machine.axis_limits`.

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
    "pa_deadband": 0.10,
    "swell_ratio_low": 1.05,
    "swell_ratio_high": 1.20
  }
}
```

### 1. 2-Point Power-Law Calibration (Shear Thinning)
Non-Newtonian polymer melt viscosity drops under higher shear strain rates:

$$\alpha = -\frac{\ln(C_{\text{PA,high}}) - \ln(C_{\text{PA,low}})}{\ln(Q_{\text{high}}) - \ln(Q_{\text{low}})}, \quad C_{\text{PA}}(Q) = C_{\text{PA,zero}} \cdot Q^{-\alpha}$$

### 2. Forced Convective Tip Cooling & Arrhenius Viscosity
Cooling fan air stream drops effective polymer tip temperature:

$$T_{\text{effective}} = T_{\text{block}} - \Delta T_{\text{max,fan}} \cdot F^{0.6}, \quad C_{\text{PA,dynamic}} = C_{\text{PA}}(Q) \cdot e^{-0.02 \cdot (T_{\text{effective}} - T_{\text{ref}})}$$

### 3. Viscoelastic Extrudate Swell Compensation
High-viscosity polymer melts store elastic strain energy in the nozzle melt chamber, resulting in die swell (transverse bead expansion upon exit). Manifold interpolates shear-dependent swell ratios:

$$B(Q) = B_{\text{low}} + (B_{\text{high}} - B_{\text{low}}) \cdot \left(\frac{Q - Q_{\text{low}}}{Q_{\text{high}} - Q_{\text{low}}}\right)$$

and scales nominal extrusion rate to maintain precise dimensional tolerances across variable-speed non-planar passes.

### 4. Adaptive Junction-Velocity Retraction ($L_{\text{retract}}$)
Elastic melt-zone pressure is relieved using filament feed velocity:

$$L_{\text{retract}} = C_{\text{PA}} \cdot \left(\frac{Q_{\text{exit}}}{A_{\text{filament}}}\right) + L_{\text{static}}$$

### 5. Time-Dependent Thermal Ooze Recovery ($L_{\text{unretract}}$)
Over travel duration $t_{\text{travel}}$, re-primes oozed polymer:

$$L_{\text{unretract}} = L_{\text{retract}} + L_{\text{max,ooze}} \cdot \left(1 - e^{-t_{\text{travel}} / \tau}\right)$$

with relaxation time constant $\tau$ tunable down to $50\text{ ms}$ ($0.05\text{ s}$) for fast direct-drive extruders.
