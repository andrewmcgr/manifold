# Kinematics, Speeds & Stepper Dynamics

Manifold provides a multi-phase kinematic motion planning and lookahead engine with support for standard speed ceilings, Klipper Square Corner Velocity (SCV), and physical stepper motor torque roll-off ODE modeling.

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
  "speed_deadband_percent": 10.0
}
```

- **Volumetric Melt Rate Clamping**: For every move, the maximum feedrate is dynamically capped by the hotend's maximum volumetric melt rate limit:
  $$v \le \frac{Q_{\text{max}}}{A_{\text{bead}}}$$
- **Speed Deadband**: Emits `F{speed}` feedrates only when commanded speed changes by $\ge 10\%$, eliminating redundant micro-feedrate outputs.

---

## Accelerations (mm/s²)

```json
{
  "outer_wall_acceleration": 2500.0,
  "inner_wall_acceleration": 4000.0,
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
