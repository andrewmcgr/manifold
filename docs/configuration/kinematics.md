# Kinematics, Speeds & Stepper Dynamics

Manifold provides a multi-phase kinematic motion planning and lookahead engine with support for standard speed ceilings, Klipper Square Corner Velocity (SCV), per-axis physical limits, and stepper motor torque roll-off ODE modeling.

---

## Speeds & Feedrates (mm/min internally, mm/s in UI/CLI)

```json
{
  "print_speed": 9300.0,
  "travel_speed": 12000.0,
  "outer_wall_speed": 6600.0,
  "inner_wall_speed": 9300.0,
  "infill_speed": 9300.0,
  "solid_infill_speed": 7800.0,
  "bridge_speed": 3000.0,
  "first_layer_print_speed": 3720.0,
  "max_volumetric_speed": 24.0,
  "speed_deadband_percent": 10.0
}
```

- **Volumetric Melt Rate Clamping (`max_volumetric_speed`)**: For every move, the maximum feedrate is dynamically capped by the hotend's volumetric melt limit:
  $$v \le \frac{Q_{\text{max}}}{A_{\text{bead}}}$$
- **Speed Deadband (`speed_deadband_percent`)**: Emits `F{speed}` feedrates only when commanded speed changes by $\ge 10\%$, eliminating redundant micro-feedrate outputs.

---

## Accelerations (mm/s²)

```json
{
  "default_acceleration": 5000.0,
  "outer_wall_acceleration": 2500.0,
  "inner_wall_acceleration": 5000.0,
  "infill_acceleration": 7000.0,
  "solid_infill_acceleration": 5000.0,
  "bridge_acceleration": 3000.0,
  "travel_acceleration": 10000.0,
  "first_layer_acceleration": 2000.0,
  "acceleration_deadband_percent": 20.0,
  "square_corner_velocity": 5.0
}
```

- **Klipper Square Corner Velocity (SCV)**: Emitted in G-code header via `SET_VELOCITY_LIMIT SQUARE_CORNER_VELOCITY=5.0`. Evaluates junction speeds using Klipper's cornering model:
  $$v_{\text{corner}} = v_{\text{scv}} \cdot \cot\left(\frac{\theta}{2}\right)$$
- **Acceleration Deadband**: Emits `SET_VELOCITY_LIMIT ACCEL={target_accel:.0}` only when acceleration changes by $\ge 20\%$ or when move kind changes.

---

## Stepper Dynamics Model (`use_stepper_dynamics`)

Under physical stepper motor physics, motor torque rolls off linearly with rotational speed due to back-EMF and coil inductance. Available acceleration follows the ODE:

$$\frac{dv}{dt} = a(v) = a_0 \cdot \max\left(0, 1 - \frac{v}{v_{\text{max}}}\right)$$

```json
{
  "machine": {
    "use_stepper_dynamics": true,
    "zero_speed_acceleration": 20000.0,
    "max_available_speed": 1000.0,
    "acceleration_limit": 10000.0,
    "speed_limit": 750.0
  }
}
```

- **$a_0$ (`zero_speed_acceleration`)**: Peak holding acceleration at zero velocity (default $20,000\text{ mm/s}^2$).
- **$v_{\text{max}}$ (`max_available_speed`)**: Theoretical speed where motor torque drops to zero (default $1,000\text{ mm/s}$).
- **Kinematic Potential Integration**: Velocity lookahead integrates the closed-form potential $F(v) = v_{\text{max}}^2 [(1 - v/v_{\text{max}}) - \ln(1 - v/v_{\text{max}})]$ via Newton-Raphson iteration, giving exact motor acceleration profiles.

---

## Per-Axis Kinematics & 3D Vector Projection (`axis_limits`)

Non-planar 3D toolpaths involve coordinated simultaneous movement across X, Y, and Z axes. Because Z-axis leadscrews or belt reductions typically possess lower maximum speeds and higher inertia than XY gantries, Manifold supports independent per-axis kinematic limits:

```json
{
  "machine": {
    "axis_limits": {
      "X": {
        "speed_limit": 500.0,
        "acceleration_limit": 15000.0
      },
      "Y": {
        "speed_limit": 500.0,
        "acceleration_limit": 15000.0
      },
      "Z": {
        "speed_limit": 40.0,
        "acceleration_limit": 1500.0,
        "use_stepper_dynamics": true,
        "zero_speed_acceleration": 3000.0,
        "max_available_speed": 60.0
      }
    }
  }
}
```

### Dynamic 3D Move Throttling

For any 3D move along unit direction vector $\vec{u} = (\hat{u}_x, \hat{u}_y, \hat{u}_z)$, the permissible 3D path velocity $v_{\text{path}}$ is dynamically clamped against each active axis limit:

$$v_{\text{path}} \le \min \left( v_{\text{global}}, \frac{v_{\text{limit}, X}}{|\hat{u}_x|}, \frac{v_{\text{limit}, Y}}{|\hat{u}_y|}, \frac{v_{\text{limit}, Z}}{|\hat{u}_z|} \right)$$

This guarantees that steep non-planar vertical climbs automatically throttle toolhead speed so that the Z axis never exceeds its safe feedrate or torque envelope.

---

## Kinematic Travel Move & Detour Planning

Travel moves in 3D non-planar space are planned using machine axis speeds and accelerations to minimize actual elapsed print time:

1. **Relative Velocity & Acceleration Scaling**:
   The travel planner calculates the physical kinematic penalty factor $\text{penalty}_z$:
   $$\text{speed\_ratio} = \frac{v_{xy}}{v_z}, \quad \text{accel\_ratio} = \sqrt{\frac{a_{xy}}{a_z}}$$
   $$\text{penalty}_z = \max\left(\text{config.z\_travel\_penalty}, \text{speed\_ratio}, \text{accel\_ratio}\right)$$

2. **Kinematic Travel Order Optimization (`optimize_travel_order`)**:
   Nearest-neighbor path selection uses an anisotropic distance metric:
   $$\text{cost}(a, b) = \sqrt{(b_x - a_x)^2 + (b_y - a_y)^2 + (\text{penalty}_z \cdot (b_z - a_z))^2}$$
   preventing excessive vertical jumps between separate features or scanlines.

3. **Anisotropic A\* Collision Detour Routing (`route_around_obstruction`)**:
   When a travel chord passes through solid material, A\* obstacle routing uses the same consistent kinematic metric for step costs and heuristic estimation, naturally favoring fast planar detours around obstacles over slow vertical climbing.
