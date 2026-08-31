//! Kinematics, motion modeling, and extrusion rate control.
//!
//! Provides pluggable motion models ([`MotionModel`]), stepper dynamic torque
//! roll-off modeling, per-move-type speed/acceleration limits, and volumetric flow
//! constraints.

use crate::toolpath::MoveKind;
use glam::DVec3;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Motion axes for multi-axis machines.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Axis {
    X,
    Y,
    Z,
}

impl std::fmt::Display for Axis {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Axis::X => write!(f, "X"),
            Axis::Y => write!(f, "Y"),
            Axis::Z => write!(f, "Z"),
        }
    }
}

/// Limits and dynamic motor model parameters for an individual motion axis.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct AxisLimits {
    /// Maximum velocity limit along this axis (mm/s).
    #[serde(default)]
    pub speed_limit: Option<f64>,
    /// Maximum acceleration limit along this axis (mm/s²).
    #[serde(default)]
    pub acceleration_limit: Option<f64>,
    /// Whether this axis uses a dedicated stepper dynamics roll-off model.
    #[serde(default)]
    pub use_stepper_dynamics: bool,
    /// Acceleration at zero velocity for this axis (a0, mm/s²).
    #[serde(default)]
    pub zero_speed_acceleration: Option<f64>,
    /// Maximum velocity where motor torque reaches zero for this axis (v_max, mm/s).
    #[serde(default)]
    pub max_available_speed: Option<f64>,
}

impl AxisLimits {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Available motor acceleration for this axis at component velocity `v_axis_mm_s`.
    #[must_use]
    pub fn motor_acceleration_at_speed(
        &self,
        v_axis_mm_s: f64,
        fallback_a0: f64,
        fallback_vmax: f64,
    ) -> f64 {
        let a0 = self.zero_speed_acceleration.unwrap_or(fallback_a0);
        let vmax = self.max_available_speed.unwrap_or(fallback_vmax);
        if vmax <= 1e-6 {
            return 0.0;
        }
        let factor = (1.0 - (v_axis_mm_s / vmax).clamp(0.0, 1.0)).max(0.0);
        a0 * factor
    }
}

/// Pluggable interface for motion kinematics and acceleration constraints.
pub trait MotionModel: Send + Sync {
    /// Maximum target feedrate for a move kind (in mm/min).
    fn max_feedrate(&self, kind: MoveKind, is_first_layer: bool) -> f64;

    /// Available acceleration at current speed `v_mm_s` (in mm/s²).
    fn available_acceleration(&self, kind: MoveKind, is_first_layer: bool, v_mm_s: f64) -> f64;

    /// Direction-constrained maximum feedrate (in mm/min) along unit direction `dir`.
    fn max_directional_feedrate(&self, kind: MoveKind, is_first_layer: bool, _dir: DVec3) -> f64 {
        self.max_feedrate(kind, is_first_layer)
    }

    /// Direction-constrained available acceleration (in mm/s²) at speed `v_mm_s` along unit direction `dir`.
    fn available_directional_acceleration(
        &self,
        kind: MoveKind,
        is_first_layer: bool,
        v_mm_s: f64,
        _dir: DVec3,
    ) -> f64 {
        self.available_acceleration(kind, is_first_layer, v_mm_s)
    }

    /// Calculate the maximum reachable speed (in mm/s) over `distance_mm` starting from `v_entry_mm_s`.
    fn max_reachable_speed(
        &self,
        kind: MoveKind,
        is_first_layer: bool,
        v_entry_mm_s: f64,
        distance_mm: f64,
    ) -> f64 {
        let accel = self.available_acceleration(kind, is_first_layer, v_entry_mm_s);
        let max_v = self.max_feedrate(kind, is_first_layer) / 60.0;
        let reachable = (v_entry_mm_s * v_entry_mm_s + 2.0 * accel * distance_mm.max(0.0))
            .max(0.0)
            .sqrt();
        reachable.min(max_v)
    }

    /// Calculate the directional maximum reachable speed (in mm/s) over `distance_mm` starting from `v_entry_mm_s` along `dir`.
    fn max_directional_reachable_speed(
        &self,
        kind: MoveKind,
        is_first_layer: bool,
        v_entry_mm_s: f64,
        distance_mm: f64,
        dir: DVec3,
    ) -> f64 {
        let accel =
            self.available_directional_acceleration(kind, is_first_layer, v_entry_mm_s, dir);
        let max_v = self.max_directional_feedrate(kind, is_first_layer, dir) / 60.0;
        let reachable = (v_entry_mm_s * v_entry_mm_s + 2.0 * accel * distance_mm.max(0.0))
            .max(0.0)
            .sqrt();
        reachable.min(max_v)
    }

    /// Calculate the time (in seconds) required to traverse `distance_mm` from `v_entry_mm_s` to `v_exit_mm_s`.
    fn move_duration(
        &self,
        kind: MoveKind,
        is_first_layer: bool,
        distance_mm: f64,
        v_entry_mm_s: f64,
        v_exit_mm_s: f64,
    ) -> f64 {
        self.directional_move_duration(
            kind,
            is_first_layer,
            distance_mm,
            v_entry_mm_s,
            v_exit_mm_s,
            DVec3::ZERO,
        )
    }

    /// Calculate the time (in seconds) required to traverse `distance_mm` from `v_entry_mm_s` to `v_exit_mm_s` along `dir`.
    fn directional_move_duration(
        &self,
        kind: MoveKind,
        is_first_layer: bool,
        distance_mm: f64,
        v_entry_mm_s: f64,
        v_exit_mm_s: f64,
        dir: DVec3,
    ) -> f64 {
        let avg_v = ((v_entry_mm_s + v_exit_mm_s) * 0.5).max(1.0);
        let accel = self.available_directional_acceleration(kind, is_first_layer, avg_v, dir);
        let max_v = self.max_directional_feedrate(kind, is_first_layer, dir) / 60.0;
        let v_peak =
            ((v_entry_mm_s * v_entry_mm_s + v_exit_mm_s * v_exit_mm_s + 2.0 * accel * distance_mm)
                * 0.5)
                .max(0.0)
                .sqrt()
                .min(max_v);

        if v_peak > v_entry_mm_s && v_peak > v_exit_mm_s {
            let t_acc = (v_peak - v_entry_mm_s) / accel.max(1.0);
            let t_dec = (v_peak - v_exit_mm_s) / accel.max(1.0);
            let d_acc = (v_entry_mm_s + v_peak) * 0.5 * t_acc;
            let d_dec = (v_peak + v_exit_mm_s) * 0.5 * t_dec;
            let d_cruise = (distance_mm - d_acc - d_dec).max(0.0);
            let t_cruise = d_cruise / v_peak.max(1.0);
            t_acc + t_dec + t_cruise
        } else {
            distance_mm / avg_v
        }
    }
}

/// Standard motion model using per-feature-type constant acceleration and speed limits.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StandardMotionModel {
    pub outer_wall_speed: f64,
    pub inner_wall_speed: f64,
    pub infill_speed: f64,
    pub solid_infill_speed: f64,
    pub bridge_speed: f64,
    pub travel_speed: f64,
    pub first_layer_speed: f64,

    pub default_acceleration: f64,
    pub outer_wall_acceleration: f64,
    pub inner_wall_acceleration: f64,
    pub infill_acceleration: f64,
    pub travel_acceleration: f64,
    pub first_layer_acceleration: f64,

    #[serde(default)]
    pub axis_limits: HashMap<Axis, AxisLimits>,
}

impl Default for StandardMotionModel {
    fn default() -> Self {
        Self {
            outer_wall_speed: 6000.0,   // 100 mm/s
            inner_wall_speed: 9000.0,   // 150 mm/s
            infill_speed: 12000.0,      // 200 mm/s
            solid_infill_speed: 7200.0, // 120 mm/s
            bridge_speed: 3600.0,       // 60 mm/s
            travel_speed: 18000.0,      // 300 mm/s
            first_layer_speed: 1800.0,  // 30 mm/s

            default_acceleration: 5000.0,
            outer_wall_acceleration: 2500.0,
            inner_wall_acceleration: 5000.0,
            infill_acceleration: 7000.0,
            travel_acceleration: 10000.0,
            first_layer_acceleration: 2000.0,
            axis_limits: HashMap::new(),
        }
    }
}

impl MotionModel for StandardMotionModel {
    fn max_feedrate(&self, kind: MoveKind, is_first_layer: bool) -> f64 {
        let nominal = match kind {
            MoveKind::WallOuter => self.outer_wall_speed,
            MoveKind::WallInner => self.inner_wall_speed,
            MoveKind::Infill => self.infill_speed,
            MoveKind::TopSurface => self.solid_infill_speed,
            MoveKind::Bridge | MoveKind::Overhang => self.bridge_speed,
            MoveKind::Travel => self.travel_speed,
        };
        if is_first_layer && kind != MoveKind::Travel {
            nominal.min(self.first_layer_speed)
        } else {
            nominal
        }
    }

    fn available_acceleration(&self, kind: MoveKind, is_first_layer: bool, _v_mm_s: f64) -> f64 {
        if is_first_layer && kind != MoveKind::Travel {
            return self.first_layer_acceleration;
        }
        match kind {
            MoveKind::WallOuter => self.outer_wall_acceleration,
            MoveKind::WallInner => self.inner_wall_acceleration,
            MoveKind::Infill | MoveKind::TopSurface => self.infill_acceleration,
            MoveKind::Bridge | MoveKind::Overhang => self.outer_wall_acceleration,
            MoveKind::Travel => self.travel_acceleration,
        }
    }

    fn max_directional_feedrate(&self, kind: MoveKind, is_first_layer: bool, dir: DVec3) -> f64 {
        let mut max_speed = self.max_feedrate(kind, is_first_layer) / 60.0;
        let comps = [
            (Axis::X, dir.x.abs()),
            (Axis::Y, dir.y.abs()),
            (Axis::Z, dir.z.abs()),
        ];
        for (axis, comp) in comps {
            if comp > 1e-6 {
                if let Some(limits) = self.axis_limits.get(&axis) {
                    if let Some(spd) = limits.speed_limit {
                        max_speed = max_speed.min(spd / comp);
                    }
                }
            }
        }
        max_speed * 60.0
    }

    fn available_directional_acceleration(
        &self,
        kind: MoveKind,
        is_first_layer: bool,
        v_mm_s: f64,
        dir: DVec3,
    ) -> f64 {
        let mut accel = self.available_acceleration(kind, is_first_layer, v_mm_s);
        let comps = [
            (Axis::X, dir.x.abs()),
            (Axis::Y, dir.y.abs()),
            (Axis::Z, dir.z.abs()),
        ];
        for (axis, comp) in comps {
            if comp > 1e-6 {
                if let Some(limits) = self.axis_limits.get(&axis) {
                    if let Some(a_lim) = limits.acceleration_limit {
                        accel = accel.min(a_lim / comp);
                    }
                    if limits.use_stepper_dynamics {
                        let motor_a =
                            limits.motor_acceleration_at_speed(v_mm_s * comp, 20000.0, 1000.0);
                        accel = accel.min(motor_a / comp);
                    }
                }
            }
        }
        accel.max(10.0)
    }
}

/// Stepper motor dynamic performance model.
///
/// Models the physical torque/back-EMF roll-off curve of stepper motors:
/// - Maximum available acceleration `zero_speed_accel` (mm/s²) at zero velocity ($a_0$, default 20,000 mm/s²).
/// - Maximum attainable velocity `max_available_speed` (mm/s) where torque/acceleration drops to zero ($v_{\text{max}}$, default 1,000 mm/s).
/// - Linearly interpolates available acceleration as:
///   $$a(v) = a_0 \cdot \max\left(0, 1 - \frac{v}{v_{\text{max}}}\right)$$
/// - Bounds move acceleration by $\min(a_{\text{limit}}, a(v), a_{\text{kind}})$, where $a_{\text{limit}}$ defaults to $50\% \times a_0$.
/// - Bounds move speed by $\min(v_{\text{limit}}, v_{\text{kind}}, v_{\text{volumetric}})$, where $v_{\text{limit}}$ defaults to $75\% \times v_{\text{max}}$.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StepperDynamicModel {
    pub standard_model: StandardMotionModel,
    /// Maximum acceleration at zero velocity (mm/s²).
    pub zero_speed_accel: f64,
    /// Maximum velocity at zero acceleration (mm/s, due to back-EMF / torque limit).
    pub max_available_speed: f64,
    /// Hard upper bound on acceleration (mm/s²).
    pub acceleration_limit: f64,
    /// Hard upper bound on speed (mm/s).
    pub speed_limit: f64,
    /// Per-axis kinematic overrides and dedicated stepper models.
    #[serde(default)]
    pub axis_limits: HashMap<Axis, AxisLimits>,
}

impl StepperDynamicModel {
    #[must_use]
    pub fn new(
        standard_model: StandardMotionModel,
        zero_speed_accel: f64,
        max_available_speed: f64,
        acceleration_limit: f64,
        speed_limit: f64,
    ) -> Self {
        Self {
            standard_model,
            zero_speed_accel,
            max_available_speed,
            acceleration_limit,
            speed_limit,
            axis_limits: HashMap::new(),
        }
    }

    /// Available motor acceleration at linear speed `v_mm_s`.
    /// Linearly rolls off from `zero_speed_accel` down to 0 at `max_available_speed`.
    #[must_use]
    pub fn motor_acceleration_at_speed(&self, v_mm_s: f64) -> f64 {
        if self.max_available_speed <= 1e-6 {
            return 0.0;
        }
        let factor = (1.0 - (v_mm_s / self.max_available_speed).clamp(0.0, 1.0)).max(0.0);
        self.zero_speed_accel * factor
    }
}

impl Default for StepperDynamicModel {
    fn default() -> Self {
        let zero_speed_accel = 20000.0;
        let max_available_speed = 1000.0;
        Self {
            standard_model: StandardMotionModel::default(),
            zero_speed_accel,
            max_available_speed,
            acceleration_limit: zero_speed_accel * 0.5, // 10,000 mm/s²
            speed_limit: max_available_speed * 0.75,    // 750 mm/s
            axis_limits: HashMap::new(),
        }
    }
}

impl MotionModel for StepperDynamicModel {
    fn max_feedrate(&self, kind: MoveKind, is_first_layer: bool) -> f64 {
        let std_max = self.standard_model.max_feedrate(kind, is_first_layer);
        let hard_limit = if is_first_layer && kind != MoveKind::Travel {
            self.standard_model.first_layer_speed
        } else {
            self.speed_limit * 60.0
        };
        std_max.min(hard_limit)
    }

    fn available_acceleration(&self, kind: MoveKind, is_first_layer: bool, v_mm_s: f64) -> f64 {
        let std_accel = self
            .standard_model
            .available_acceleration(kind, is_first_layer, v_mm_s);
        let motor_accel = self.motor_acceleration_at_speed(v_mm_s);
        let dynamic_accel = motor_accel.min(self.acceleration_limit);
        std_accel.min(dynamic_accel).max(100.0)
    }

    fn max_directional_feedrate(&self, kind: MoveKind, is_first_layer: bool, dir: DVec3) -> f64 {
        let mut max_speed = self.max_feedrate(kind, is_first_layer) / 60.0;
        let comps = [
            (Axis::X, dir.x.abs()),
            (Axis::Y, dir.y.abs()),
            (Axis::Z, dir.z.abs()),
        ];
        for (axis, comp) in comps {
            if comp > 1e-6 {
                if let Some(limits) = self.axis_limits.get(&axis) {
                    if let Some(spd) = limits.speed_limit {
                        max_speed = max_speed.min(spd / comp);
                    }
                }
            }
        }
        max_speed * 60.0
    }

    fn available_directional_acceleration(
        &self,
        kind: MoveKind,
        is_first_layer: bool,
        v_mm_s: f64,
        dir: DVec3,
    ) -> f64 {
        let mut accel = self.available_acceleration(kind, is_first_layer, v_mm_s);
        let comps = [
            (Axis::X, dir.x.abs()),
            (Axis::Y, dir.y.abs()),
            (Axis::Z, dir.z.abs()),
        ];
        for (axis, comp) in comps {
            if comp > 1e-6 {
                if let Some(limits) = self.axis_limits.get(&axis) {
                    if let Some(a_lim) = limits.acceleration_limit {
                        accel = accel.min(a_lim / comp);
                    }
                    if limits.use_stepper_dynamics {
                        let motor_a = limits.motor_acceleration_at_speed(
                            v_mm_s * comp,
                            self.zero_speed_accel,
                            self.max_available_speed,
                        );
                        accel = accel.min(motor_a / comp);
                    }
                }
            }
        }
        accel.max(10.0)
    }

    fn max_reachable_speed(
        &self,
        kind: MoveKind,
        is_first_layer: bool,
        v_entry_mm_s: f64,
        distance_mm: f64,
    ) -> f64 {
        self.max_directional_reachable_speed(
            kind,
            is_first_layer,
            v_entry_mm_s,
            distance_mm,
            DVec3::ZERO,
        )
    }

    fn max_directional_reachable_speed(
        &self,
        kind: MoveKind,
        is_first_layer: bool,
        v_entry_mm_s: f64,
        distance_mm: f64,
        dir: DVec3,
    ) -> f64 {
        let max_v = self.max_directional_feedrate(kind, is_first_layer, dir) / 60.0;
        let mut v_stepper = stepper_max_reachable_velocity(
            v_entry_mm_s,
            distance_mm,
            self.zero_speed_accel,
            self.max_available_speed,
        );
        let comps = [
            (Axis::X, dir.x.abs()),
            (Axis::Y, dir.y.abs()),
            (Axis::Z, dir.z.abs()),
        ];
        for (axis, comp) in comps {
            if comp > 1e-6 {
                if let Some(limits) = self.axis_limits.get(&axis) {
                    if limits.use_stepper_dynamics {
                        let a0 = limits
                            .zero_speed_acceleration
                            .unwrap_or(self.zero_speed_accel);
                        let vmax = limits
                            .max_available_speed
                            .unwrap_or(self.max_available_speed);
                        let v_axis_entry = v_entry_mm_s * comp;
                        let d_axis = distance_mm * comp;
                        let v_axis_exit =
                            stepper_max_reachable_velocity(v_axis_entry, d_axis, a0, vmax);
                        v_stepper = v_stepper.min(v_axis_exit / comp);
                    }
                }
            }
        }
        let v_std = self.standard_model.max_directional_reachable_speed(
            kind,
            is_first_layer,
            v_entry_mm_s,
            distance_mm,
            dir,
        );
        v_stepper.min(v_std).min(max_v)
    }
}

/// Potential function for stepper equation of motion:
/// $$F(v) = v_{\text{max}}^2 \left[\left(1 - \frac{v}{v_{\text{max}}}\right) - \ln\left(1 - \frac{v}{v_{\text{max}}}\right)\right]$$
fn stepper_kinetic_potential(v: f64, v_max: f64) -> f64 {
    let u = (1.0 - (v / v_max).clamp(0.0, 0.999999)).max(1e-6);
    v_max * v_max * (u - u.ln())
}

/// Calculates maximum attainable exit velocity (mm/s) over distance `d` (mm) starting at `v_i` (mm/s),
/// under stepper dynamic torque roll-off: a(v) = a_0 * (1 - v / v_max).
#[must_use]
pub fn stepper_max_reachable_velocity(
    v_i: f64,
    distance: f64,
    a_max_zero_v: f64,
    v_max_zero_a: f64,
) -> f64 {
    if distance <= 1e-6 || a_max_zero_v <= 1.0 {
        return v_i;
    }
    let v_start = v_i.clamp(0.0, v_max_zero_a * 0.999);
    let target_potential =
        stepper_kinetic_potential(v_start, v_max_zero_a) + a_max_zero_v * distance;

    // Standard constant-accel estimate as initial guess
    let mut v = (v_start * v_start + 2.0 * a_max_zero_v * distance)
        .sqrt()
        .clamp(v_start + 1e-3, v_max_zero_a * 0.999);

    // Newton-Raphson to solve for v where stepper_kinetic_potential(v) == target_potential
    for _ in 0..8 {
        let u = (1.0 - (v / v_max_zero_a).clamp(0.0, 0.999999)).max(1e-6);
        let f_val = v_max_zero_a * v_max_zero_a * (u - u.ln()) - target_potential;
        let f_prime = v / u; // derivative dF/dv = v / (1 - v/v_max)
        if f_prime.abs() < 1e-6 {
            break;
        }
        let delta = f_val / f_prime;
        v = (v - delta).clamp(v_start, v_max_zero_a * 0.9999);
        if delta.abs() < 1e-3 {
            break;
        }
    }
    v
}

/// Computes the maximum junction/cornering velocity (in mm/s) according to Klipper's
/// Square Corner Velocity (SCV) model.
///
/// Given incoming unit vector `dir_in`, outgoing unit vector `dir_out`, square corner
/// velocity `scv` (mm/s), and acceleration `accel` (mm/s²).
#[must_use]
pub fn klipper_corner_velocity(
    dir_in: DVec3,
    dir_out: DVec3,
    square_corner_velocity: f64,
    accel: f64,
) -> f64 {
    let cos_theta = dir_in.dot(dir_out).clamp(-1.0, 1.0);
    if cos_theta >= 0.999999 {
        return 10000.0; // Collinear / straight move
    }
    if cos_theta <= -0.999999 {
        return 0.0; // 180° full reversal
    }

    // Klipper square corner velocity formula:
    // sin(theta/2) = sqrt((1 - cos_theta) / 2)
    // cos(theta/2) = sqrt((1 + cos_theta) / 2)
    let sin_half = ((1.0 - cos_theta) * 0.5).max(0.0).sqrt();
    let cos_half = ((1.0 + cos_theta) * 0.5).max(0.0).sqrt();

    let scv_limit = if sin_half > 1e-6 {
        square_corner_velocity * (cos_half / sin_half)
    } else {
        10000.0
    };

    // Centripetal acceleration limit over 0.04mm junction deviation
    let junction_deviation = 0.04;
    let centripetal_limit = if sin_half < 0.999 {
        ((accel * junction_deviation * sin_half) / (1.0 - sin_half))
            .max(0.0)
            .sqrt()
    } else {
        0.0
    };

    scv_limit.min(centripetal_limit).max(0.0)
}

/// Kinematic motion profile for a single move segment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlannedMotionProfile {
    pub entry_speed: f64,  // mm/min
    pub cruise_speed: f64, // mm/min
    pub exit_speed: f64,   // mm/min
    pub duration_seconds: f64,
}

/// Plans time-optimal velocity profiles along a polyline path using forward and backward
/// acceleration passes constrained by Klipper SCV and stepper torque limits.
#[must_use]
pub fn plan_path_velocities(
    points: &[DVec3],
    segments: &[crate::toolpath::Segment],
    model: &dyn MotionModel,
    is_first_layer: bool,
    square_corner_velocity_mm_s: f64,
) -> Vec<PlannedMotionProfile> {
    let n = segments.len();
    if n == 0 || points.len() < 2 {
        return Vec::new();
    }

    let mut nominal_speeds = Vec::with_capacity(n);
    let mut distances = Vec::with_capacity(n);
    let mut directions = Vec::with_capacity(n);

    for (i, seg) in segments.iter().enumerate() {
        let p0 = points[i];
        let p1 = points[(i + 1) % points.len()];
        let diff = p1 - p0;
        let d = diff.length();
        distances.push(d);
        let dir = if d > 1e-6 { diff / d } else { DVec3::ZERO };
        directions.push(dir);
        let max_feed = model.max_directional_feedrate(seg.kind, is_first_layer, dir);
        nominal_speeds.push((seg.speed.min(max_feed)) / 60.0); // mm/s
    }

    // 1. Compute junction speed limits between consecutive segments
    let mut junction_speeds = vec![0.0; n + 1];
    // Start of path starts from 0 (or low entry speed)
    junction_speeds[0] = 0.0;
    // End of path stops at 0 (before next travel / retract)
    junction_speeds[n] = 0.0;

    for i in 0..n.saturating_sub(1) {
        let d0 = directions[i];
        let d1 = directions[i + 1];
        let accel = model.available_directional_acceleration(
            segments[i].kind,
            is_first_layer,
            nominal_speeds[i],
            d0,
        );
        let corner_v = klipper_corner_velocity(d0, d1, square_corner_velocity_mm_s, accel);
        junction_speeds[i + 1] = corner_v.min(nominal_speeds[i]).min(nominal_speeds[i + 1]);
    }

    // 2. Forward pass: acceleration from entry speed
    let mut entry_speeds = vec![0.0; n];
    let mut exit_speeds = vec![0.0; n];

    for i in 0..n {
        let v_in = junction_speeds[i].min(nominal_speeds[i]);
        entry_speeds[i] = v_in;
        let v_reachable = model.max_directional_reachable_speed(
            segments[i].kind,
            is_first_layer,
            v_in,
            distances[i],
            directions[i],
        );
        exit_speeds[i] = v_reachable
            .min(nominal_speeds[i])
            .min(junction_speeds[i + 1]);
        junction_speeds[i + 1] = exit_speeds[i];
    }

    // 3. Backward pass: deceleration to junction limits
    for i in (0..n).rev() {
        let v_out = junction_speeds[i + 1];
        exit_speeds[i] = exit_speeds[i].min(v_out);
        let accel = model.available_directional_acceleration(
            segments[i].kind,
            is_first_layer,
            exit_speeds[i],
            directions[i],
        );
        let max_v_in = (exit_speeds[i] * exit_speeds[i] + 2.0 * accel * distances[i])
            .max(0.0)
            .sqrt();
        entry_speeds[i] = entry_speeds[i].min(max_v_in);
        junction_speeds[i] = entry_speeds[i];
    }

    // 4. Construct motion profiles
    let mut profiles = Vec::with_capacity(n);
    for i in 0..n {
        let v_entry = entry_speeds[i];
        let v_exit = exit_speeds[i];
        let v_cruise = nominal_speeds[i].min(v_entry.max(v_exit));
        let duration = model.directional_move_duration(
            segments[i].kind,
            is_first_layer,
            distances[i],
            v_entry,
            v_exit,
            directions[i],
        );
        profiles.push(PlannedMotionProfile {
            entry_speed: v_entry * 60.0,
            cruise_speed: v_cruise * 60.0,
            exit_speed: v_exit * 60.0,
            duration_seconds: duration,
        });
    }

    profiles
}

/// Computes maximum allowable linear feedrate (in mm/min) constrained by a volumetric flow limit.
///
/// If `max_volumetric_speed_mm3_s` is provided and $> 0$, clamps velocity such that:
/// $$v \le \frac{Q_{\text{max}}}{A_{\text{bead}}}$$
#[must_use]
pub fn clamp_feedrate_by_volumetric_limit(
    nominal_feedrate_mm_min: f64,
    bead_area_mm2: f64,
    max_volumetric_speed_mm3_s: Option<f64>,
) -> f64 {
    let Some(max_q) = max_volumetric_speed_mm3_s else {
        return nominal_feedrate_mm_min;
    };
    if max_q <= 0.0 || bead_area_mm2 <= 1e-6 {
        return nominal_feedrate_mm_min;
    }
    let max_v_mm_s = max_q / bead_area_mm2;
    let max_v_mm_min = max_v_mm_s * 60.0;
    nominal_feedrate_mm_min.min(max_v_mm_min)
}

/// Tapers extrusion rate and feedrate across the final `taper_distance_mm` of an extrusion run
/// (preceding a retraction / travel move) to bleed excess melt-zone pressure.
///
/// Uses the pressure-advance aware bleed model:
/// - As distance to the path end drops below `taper_distance_mm`, the segment's `extrusion_rate`
///   smoothly tapers from 1.0 down to `min_rate` (default 0.20 = 20%).
/// - Adjusts `segment.extrusion_length` proportionally: `extrusion_length *= effective_rate`.
pub fn apply_pre_retract_taper(
    points: &mut Vec<DVec3>,
    segments: &mut Vec<crate::toolpath::Segment>,
    taper_distance_mm: f64,
    min_rate: f64,
) {
    if taper_distance_mm <= 1e-4 || segments.is_empty() {
        return;
    }

    // Identify the last extruding segment index
    let mut last_extruding_idx = None;
    for (i, seg) in segments.iter().enumerate().rev() {
        if seg.kind != MoveKind::Travel {
            last_extruding_idx = Some(i);
            break;
        }
    }
    let Some(last_idx) = last_extruding_idx else {
        return;
    };

    let total_extruding_len: f64 = (0..=last_idx)
        .filter_map(|i| {
            if segments[i].kind != MoveKind::Travel {
                let p0 = points[i];
                let p1 = points[(i + 1) % points.len()];
                Some((p1 - p0).length())
            } else {
                None
            }
        })
        .sum();
    if total_extruding_len < taper_distance_mm * 1.5 {
        return;
    }

    let p_start = points[last_idx];
    let p_end = points[(last_idx + 1) % points.len()];
    let last_seg_len = (p_end - p_start).length();

    if last_seg_len > taper_distance_mm + 0.1 {
        // Split last segment into untapered lead-in + tapered tail
        let split_ratio = (last_seg_len - taper_distance_mm) / last_seg_len;
        let p_split = p_start.lerp(p_end, split_ratio);

        let orig_seg = segments[last_idx];
        let mut lead_seg = orig_seg;
        let mut tail_seg = orig_seg;

        lead_seg.extrusion_length = orig_seg.extrusion_length * split_ratio;
        tail_seg.extrusion_length = orig_seg.extrusion_length * (1.0 - split_ratio);

        let avg_tail_rate = (1.0 + min_rate) * 0.5;
        tail_seg.extrusion_rate *= avg_tail_rate;
        tail_seg.extrusion_length *= avg_tail_rate;

        points.insert(last_idx + 1, p_split);
        segments[last_idx] = lead_seg;
        segments.insert(last_idx + 1, tail_seg);
        return;
    }

    let mut seg_lengths = Vec::new();
    for i in 0..=last_idx {
        let p0 = points[i];
        let p1 = points[(i + 1) % points.len()];
        seg_lengths.push((p1 - p0).length());
    }

    let mut dist_from_end = 0.0;
    for i in (0..=last_idx).rev() {
        let seg_len = seg_lengths[i];
        if segments[i].kind == MoveKind::Travel {
            break;
        }
        let seg_mid_dist = dist_from_end + seg_len * 0.5;
        if seg_mid_dist < taper_distance_mm {
            let t = (seg_mid_dist / taper_distance_mm).clamp(0.0, 1.0);
            let taper_factor = min_rate + (1.0 - min_rate) * t;
            segments[i].extrusion_rate *= taper_factor;
            segments[i].extrusion_length *= taper_factor;
        }
        dist_from_end += seg_len;
        if dist_from_end >= taper_distance_mm {
            break;
        }
    }
}

/// Inserts an unextruded wipe segment at the end of closed perimeter wall loops
/// to wipe the nozzle tip along the loop before lifting for travel / retracting.
pub fn apply_wipe_moves(
    points: &mut Vec<DVec3>,
    segments: &mut Vec<crate::toolpath::Segment>,
    wipe_distance_mm: f64,
) {
    if wipe_distance_mm <= 1e-4 || points.len() < 3 || segments.is_empty() {
        return;
    }
    // Only apply to paths that start and finish with extrusion (e.g. wall loops)
    let is_extruding_loop = segments.first().is_some_and(|s| s.kind != MoveKind::Travel)
        && segments.last().is_some_and(|s| s.kind != MoveKind::Travel);
    if !is_extruding_loop {
        return;
    }

    // Direction along the first segment of the loop (p0 -> p1)
    let p0 = points[0];
    let p1 = points[1];
    let d = (p1 - p0).length();
    if d <= 1e-4 {
        return;
    }
    let wipe_dir = (p1 - p0) / d;
    let actual_wipe_len = wipe_distance_mm.min(d);
    let p_wipe = p0 + wipe_dir * actual_wipe_len;

    let last_seg = *segments.last().unwrap();
    let wipe_seg = crate::toolpath::Segment {
        kind: MoveKind::Travel,
        extrusion_rate: 0.0,
        extrusion_length: 0.0,
        speed: last_seg.speed,
        order: last_seg.order,
        support_fraction: last_seg.support_fraction,
        line_width: 0.0,
        is_scarf: false,
    };

    points.push(p_wipe);
    segments.push(wipe_seg);
}

/// Applies non-planar scarf joint seam ramping to a closed perimeter wall loop:
/// - Subdivides the scarf region into `steps` discrete segments over length `scarf_length_mm`.
/// - Ramps extrusion flow and slice-normal height from `start_height_fraction` (e.g. 10%) -> 100%
///   over the initial lead-in ramp, offset along `-layer_normal` to create the bottom wedge.
/// - Overlaps the start of the loop by continuing past the start point at nominal layer height,
///   ramping extrusion flow from `(1.0 - start_height_fraction)` -> 0% (lead-out top wedge).
/// - The sum of the complementary flow ramps is exactly 100% nominal bead everywhere across the joint,
///   eliminating vertical seam lines on perimeters without localized overextrusion.
pub fn apply_scarf_joint(
    points: &mut Vec<DVec3>,
    segments: &mut Vec<crate::toolpath::Segment>,
    scarf_length_mm: f64,
    steps: usize,
    start_height_fraction: f64,
    layer_height: f64,
    order_field: Option<&dyn manifold_fidget::order::OrderField>,
) {
    if scarf_length_mm <= 1e-4 || steps == 0 || points.len() < 3 || segments.is_empty() {
        return;
    }
    // Only apply to closed extruding loops (points.len() == segments.len())
    if points.len() != segments.len() {
        return;
    }
    let is_wall_loop = segments
        .first()
        .is_some_and(|s| s.kind == MoveKind::WallOuter || s.kind == MoveKind::WallInner);
    if !is_wall_loop {
        return;
    }

    let n = points.len();
    let orig_points = points.clone();
    let orig_segments = segments.clone();

    let mut seg_lens = Vec::with_capacity(n);
    let mut cum_dist = Vec::with_capacity(n + 1);
    cum_dist.push(0.0);
    let mut total_len = 0.0;

    for i in 0..n {
        let p0 = orig_points[i];
        let p1 = orig_points[(i + 1) % n];
        let l = (p1 - p0).length();
        seg_lens.push(l);
        total_len += l;
        cum_dist.push(total_len);
    }

    if total_len <= 1e-4 {
        return;
    }

    let effective_scarf_len = scarf_length_mm.min(0.40 * total_len);
    if effective_scarf_len <= 1e-3 {
        return;
    }

    let k_steps = steps.max(1);
    let delta_s = effective_scarf_len / (k_steps as f64);
    let h_start = start_height_fraction.clamp(0.0, 0.95);

    let sample_at_distance = |d: f64| -> (DVec3, crate::toolpath::Segment, f64) {
        let d = d.clamp(0.0, total_len);
        let mut seg_idx = 0;
        for i in 0..n {
            if d <= cum_dist[i + 1] || i == n - 1 {
                seg_idx = i;
                break;
            }
        }
        let seg_len = seg_lens[seg_idx];
        let seg_start_d = cum_dist[seg_idx];
        let u = if seg_len > 1e-9 {
            ((d - seg_start_d) / seg_len).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let p_start = orig_points[seg_idx];
        let p_end = orig_points[(seg_idx + 1) % n];
        let p = p_start.lerp(p_end, u);
        let seg = orig_segments[seg_idx];
        let e_per_mm = if seg_len > 1e-9 {
            seg.extrusion_length / seg_len
        } else {
            0.0
        };
        (p, seg, e_per_mm)
    };

    // Do not apply scarf joints on loops that contain unsupported overhang segments
    if orig_segments.iter().any(|s| {
        s.support_fraction < 0.8 || s.kind == MoveKind::Overhang || s.kind == MoveKind::Bridge
    }) {
        return;
    }

    let slice_normal = |p: DVec3| -> DVec3 {
        if let Some(field) = order_field {
            if let Some(grad) = crate::order_field::numeric_gradient(field, p) {
                let len = grad.length();
                if len > 1e-6 {
                    let mut norm = grad / len;
                    if norm.z < 0.0 {
                        norm = -norm;
                    }
                    return norm;
                }
            }
        }
        DVec3::Z
    };

    let mut new_points = Vec::with_capacity(n + 2 * k_steps + 2);
    let mut new_segments = Vec::with_capacity(n + 2 * k_steps + 2);

    // 1. Lead-in ramp (k = 0..k_steps)
    let mut lead_in_pts = Vec::with_capacity(k_steps + 1);
    for k in 0..=k_steps {
        let s = (k as f64) * delta_s;
        let tau = (k as f64) / (k_steps as f64);
        let h_frac = h_start + (1.0 - h_start) * tau;
        let (base_p, seg, _) = sample_at_distance(s);
        let offset = if seg.support_fraction >= 0.7
            && seg.kind != MoveKind::Overhang
            && seg.kind != MoveKind::Bridge
        {
            let norm = slice_normal(base_p);
            -((1.0 - h_frac) * layer_height) * norm
        } else {
            DVec3::ZERO
        };
        lead_in_pts.push(base_p + offset);
    }

    for (k, &pt) in lead_in_pts.iter().enumerate().take(k_steps) {
        let t = (k as f64 + 0.5) / (k_steps as f64);
        let flow_frac = h_start + (1.0 - h_start) * t;
        let s_mid = (k as f64 + 0.5) * delta_s;
        let (_, mut seg, e_per_mm) = sample_at_distance(s_mid);
        seg.extrusion_rate *= flow_frac;
        seg.extrusion_length = e_per_mm * delta_s * flow_frac;
        seg.is_scarf = true;
        new_points.push(pt);
        new_segments.push(seg);
    }

    // 2. Main loop body (from s = effective_scarf_len to s = total_len)
    let p_scarf_end = lead_in_pts[k_steps];
    new_points.push(p_scarf_end);

    // Find the segment spanning effective_scarf_len
    let mut span_idx = 0;
    for i in 0..n {
        if effective_scarf_len <= cum_dist[i + 1] || i == n - 1 {
            span_idx = i;
            break;
        }
    }

    // Remainder of the spanning segment (from effective_scarf_len to cum_dist[span_idx + 1])
    let rem_len = cum_dist[span_idx + 1] - effective_scarf_len;
    if rem_len > 1e-6 {
        let mut seg = orig_segments[span_idx];
        let e_per_mm = if seg_lens[span_idx] > 1e-9 {
            seg.extrusion_length / seg_lens[span_idx]
        } else {
            0.0
        };
        seg.extrusion_length = e_per_mm * rem_len;
        new_segments.push(seg);
        new_points.push(orig_points[(span_idx + 1) % n]);
    }

    // Subsequent full segments up to total_len (which ends at orig_points[0])
    for i in (span_idx + 1)..n {
        let seg = orig_segments[i];
        new_segments.push(seg);
        new_points.push(orig_points[(i + 1) % n]);
    }

    // 3. Lead-out overlap ramp (k = 0..k_steps)
    let mut lead_out_pts = Vec::with_capacity(k_steps + 1);
    for k in 0..=k_steps {
        let s = (k as f64) * delta_s;
        let (base_p, _, _) = sample_at_distance(s);
        lead_out_pts.push(base_p);
    }

    for k in 0..k_steps {
        let t = (k as f64 + 0.5) / (k_steps as f64);
        let flow_frac = (1.0 - h_start) * (1.0 - t);
        let s_mid = (k as f64 + 0.5) * delta_s;
        let (_, mut seg, e_per_mm) = sample_at_distance(s_mid);
        seg.extrusion_rate *= flow_frac;
        seg.extrusion_length = e_per_mm * delta_s * flow_frac;
        seg.is_scarf = false;
        new_segments.push(seg);
        new_points.push(lead_out_pts[k + 1]);
    }

    *points = new_points;
    *segments = new_segments;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_motion_model_applies_per_kind_speeds_and_accelerations() {
        let model = StandardMotionModel {
            outer_wall_speed: 3000.0,
            inner_wall_speed: 6000.0,
            outer_wall_acceleration: 2000.0,
            inner_wall_acceleration: 4000.0,
            ..StandardMotionModel::default()
        };

        assert_eq!(model.max_feedrate(MoveKind::WallOuter, false), 3000.0);
        assert_eq!(model.max_feedrate(MoveKind::WallInner, false), 6000.0);
        assert_eq!(
            model.available_acceleration(MoveKind::WallOuter, false, 50.0),
            2000.0
        );
        assert_eq!(
            model.available_acceleration(MoveKind::WallInner, false, 50.0),
            4000.0
        );
    }

    #[test]
    fn stepper_dynamic_model_interpolates_acceleration_with_velocity() {
        let model = StepperDynamicModel {
            zero_speed_accel: 10000.0,
            max_available_speed: 500.0,
            acceleration_limit: 8000.0,
            speed_limit: 400.0,
            standard_model: StandardMotionModel {
                outer_wall_acceleration: 20000.0,
                outer_wall_speed: 600.0 * 60.0,
                ..StandardMotionModel::default()
            },
            axis_limits: HashMap::new(),
        };

        // At v = 0, available motor accel is 10,000, clamped by acceleration_limit (8,000)
        let a_0 = model.available_acceleration(MoveKind::WallOuter, false, 0.0);
        assert!((a_0 - 8000.0).abs() < 1e-3);

        // At v = 250 mm/s (half max speed), motor accel is 5000, which is below limit (8000) -> 5000
        let a_half = model.available_acceleration(MoveKind::WallOuter, false, 250.0);
        assert!((a_half - 5000.0).abs() < 1e-3);

        // At v = 500 mm/s (max speed), acceleration clamped to minimum floor (100)
        let a_max = model.available_acceleration(MoveKind::WallOuter, false, 500.0);
        assert!((a_max - 100.0).abs() < 1e-3);

        // Max feedrate is clamped by speed_limit (400 mm/s = 24,000 mm/min)
        assert_eq!(model.max_feedrate(MoveKind::WallOuter, false), 24000.0);
    }

    #[test]
    fn first_layer_speed_acts_as_speed_limit_for_first_layer_extrusions() {
        let model = StandardMotionModel {
            outer_wall_speed: 6000.0,  // 100 mm/s
            inner_wall_speed: 9000.0,  // 150 mm/s
            bridge_speed: 1200.0,      // 20 mm/s
            travel_speed: 18000.0,     // 300 mm/s
            first_layer_speed: 1800.0, // 30 mm/s limit
            ..StandardMotionModel::default()
        };

        // On first layer: moves faster than first_layer_speed are clamped down to 1800
        assert_eq!(model.max_feedrate(MoveKind::WallOuter, true), 1800.0);
        assert_eq!(model.max_feedrate(MoveKind::WallInner, true), 1800.0);

        // Moves already slower than first_layer_speed keep their lower speed
        assert_eq!(model.max_feedrate(MoveKind::Bridge, true), 1200.0);

        // Travel moves are not clamped by first_layer_speed
        assert_eq!(model.max_feedrate(MoveKind::Travel, true), 18000.0);
    }

    #[test]
    fn stepper_dynamic_model_respects_first_layer_speed_limit() {
        let model = StepperDynamicModel {
            zero_speed_accel: 20000.0,
            max_available_speed: 1000.0,
            acceleration_limit: 10000.0,
            speed_limit: 750.0, // Global speed limit: 750 mm/s = 45,000 mm/min
            standard_model: StandardMotionModel {
                outer_wall_speed: 6000.0,  // 100 mm/s
                inner_wall_speed: 9000.0,  // 150 mm/s
                travel_speed: 18000.0,     // 300 mm/s
                first_layer_speed: 1800.0, // 30 mm/s limit
                first_layer_acceleration: 2000.0,
                ..StandardMotionModel::default()
            },
            axis_limits: HashMap::new(),
        };

        // Normal layer: outer wall runs at 6000 (below global 750mm/s limit)
        assert_eq!(model.max_feedrate(MoveKind::WallOuter, false), 6000.0);

        // First layer: extrusions are capped by first_layer_speed (1800 mm/min = 30 mm/s)
        assert_eq!(model.max_feedrate(MoveKind::WallOuter, true), 1800.0);
        assert_eq!(model.max_feedrate(MoveKind::WallInner, true), 1800.0);

        // First layer: lookahead max reachable speed is strictly capped by first layer limit (30 mm/s)
        let reachable = model.max_reachable_speed(MoveKind::WallOuter, true, 0.0, 100.0);
        assert!((reachable - 30.0).abs() < 1e-4);

        // First layer: available acceleration uses first_layer_acceleration (2000), bounded by motor curve and accel limit
        let accel = model.available_acceleration(MoveKind::WallOuter, true, 30.0);
        assert_eq!(accel, 2000.0);
    }

    #[test]
    fn volumetric_limit_caps_linear_feedrate_when_bead_is_thick() {
        let nominal_speed = 6000.0; // 100 mm/s
        let bead_area = 0.40 * 0.20; // 0.08 mm²
        let max_volumetric_speed = 4.0; // 4.0 mm³/s => max linear v = 4.0 / 0.08 = 50 mm/s = 3000 mm/min

        let clamped = clamp_feedrate_by_volumetric_limit(
            nominal_speed,
            bead_area,
            Some(max_volumetric_speed),
        );
        assert!((clamped - 3000.0).abs() < 1e-3);
    }

    #[test]
    fn klipper_corner_velocity_calculates_exact_right_angle_scv() {
        let d_in = DVec3::new(1.0, 0.0, 0.0);
        let d_out = DVec3::new(0.0, 1.0, 0.0); // 90° right angle turn
        let scv = 5.0; // 5 mm/s
        let accel = 5000.0;

        let v_corner = klipper_corner_velocity(d_in, d_out, scv, accel);
        // At 90°, Klipper SCV evaluates to scv (5.0 mm/s)
        assert!(
            (v_corner - 5.0).abs() < 1e-2,
            "90° corner velocity should match SCV: {v_corner}"
        );

        // Collinear straight line -> high corner velocity
        let v_straight = klipper_corner_velocity(d_in, d_in, scv, accel);
        assert!(v_straight > 1000.0);

        // 180° reversal -> 0 corner velocity
        let v_reverse = klipper_corner_velocity(d_in, -d_in, scv, accel);
        assert_eq!(v_reverse, 0.0);
    }

    #[test]
    fn stepper_max_reachable_velocity_converges_and_scales_with_distance() {
        let v_0 = 0.0;
        let a_0 = 10000.0;
        let v_max = 500.0;

        let v_short = stepper_max_reachable_velocity(v_0, 1.0, a_0, v_max);
        let v_long = stepper_max_reachable_velocity(v_0, 50.0, a_0, v_max);

        assert!(v_short > 0.0);
        assert!(v_long > v_short);
        assert!(v_long < v_max);
    }

    #[test]
    fn plan_path_velocities_ramps_acceleration_and_deceleration_around_sharp_corners() {
        use crate::toolpath::Segment;

        let points = vec![
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(50.0, 0.0, 0.0),
            DVec3::new(50.0, 50.0, 0.0), // 90° turn
            DVec3::new(0.0, 50.0, 0.0),  // 90° turn
        ];
        let segments = vec![
            Segment {
                kind: MoveKind::WallOuter,
                speed: 6000.0, // 100 mm/s
                ..Segment::default()
            },
            Segment {
                kind: MoveKind::WallOuter,
                speed: 6000.0,
                ..Segment::default()
            },
            Segment {
                kind: MoveKind::WallOuter,
                speed: 6000.0,
                ..Segment::default()
            },
        ];
        let model = StandardMotionModel::default();
        let profiles = plan_path_velocities(&points, &segments, &model, false, 5.0);

        assert_eq!(profiles.len(), 3);
        // First segment starts from entry 0.0
        assert_eq!(profiles[0].entry_speed, 0.0);
        // Exit speed at 90° corner is bounded by Klipper SCV (5 mm/s = 300 mm/min)
        assert!((profiles[0].exit_speed - 300.0).abs() < 10.0);
        // Last segment finishes at exit 0.0
        assert_eq!(profiles[2].exit_speed, 0.0);
    }

    #[test]
    fn apply_pre_retract_taper_reduces_tail_extrusion_rate() {
        use crate::toolpath::Segment;

        let mut points = vec![DVec3::new(0.0, 0.0, 0.0), DVec3::new(10.0, 0.0, 0.0)];
        let mut segments = vec![Segment {
            kind: MoveKind::WallOuter,
            extrusion_length: 5.0,
            extrusion_rate: 1.0,
            ..Segment::default()
        }];

        apply_pre_retract_taper(&mut points, &mut segments, 2.0, 0.2);

        // Long segment (10mm) should be split at 8.0mm into lead-in and 2.0mm tail
        assert_eq!(points.len(), 3);
        assert_eq!(segments.len(), 2);
        assert!((points[1].x - 8.0).abs() < 1e-4);

        // Lead-in (80% length) has 4.0mm extrusion
        assert!((segments[0].extrusion_length - 4.0).abs() < 1e-4);
        // Tapered tail (20% length with average 0.6x flow) has 1.0 * 0.6 = 0.6mm extrusion
        assert!((segments[1].extrusion_length - 0.6).abs() < 1e-4);
        assert!((segments[1].extrusion_rate - 0.6).abs() < 1e-4);
    }

    #[test]
    fn apply_wipe_moves_appends_unextruded_wipe_segment() {
        use crate::toolpath::Segment;

        let mut points = vec![
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(10.0, 0.0, 0.0),
            DVec3::new(10.0, 10.0, 0.0),
            DVec3::new(0.0, 10.0, 0.0),
        ];
        let mut segments = vec![
            Segment {
                kind: MoveKind::WallOuter,
                extrusion_length: 5.0,
                ..Segment::default()
            },
            Segment {
                kind: MoveKind::WallOuter,
                extrusion_length: 5.0,
                ..Segment::default()
            },
            Segment {
                kind: MoveKind::WallOuter,
                extrusion_length: 5.0,
                ..Segment::default()
            },
            Segment {
                kind: MoveKind::WallOuter,
                extrusion_length: 5.0,
                ..Segment::default()
            },
        ];

        apply_wipe_moves(&mut points, &mut segments, 2.0);

        // 1 extra point and 1 extra unextruded travel segment
        assert_eq!(points.len(), 5);
        assert_eq!(segments.len(), 5);
        assert_eq!(segments[4].kind, MoveKind::Travel);
        assert_eq!(segments[4].extrusion_length, 0.0);
        // Wipe vector extends 2.0mm along p0->p1 (X=2.0)
        assert!((points[4].x - 2.0).abs() < 1e-4);
    }

    #[test]
    fn apply_scarf_joint_creates_overlapping_ramps_on_closed_wall_loop() {
        use crate::toolpath::Segment;

        let mut points = vec![
            DVec3::new(0.0, 0.0, 1.0),
            DVec3::new(20.0, 0.0, 1.0),
            DVec3::new(20.0, 20.0, 1.0),
            DVec3::new(0.0, 20.0, 1.0),
        ];
        let mut segments = vec![
            Segment {
                kind: MoveKind::WallOuter,
                extrusion_length: 10.0,
                extrusion_rate: 1.0,
                support_fraction: 1.0,
                ..Segment::default()
            },
            Segment {
                kind: MoveKind::WallOuter,
                extrusion_length: 10.0,
                extrusion_rate: 1.0,
                support_fraction: 1.0,
                ..Segment::default()
            },
            Segment {
                kind: MoveKind::WallOuter,
                extrusion_length: 10.0,
                extrusion_rate: 1.0,
                support_fraction: 1.0,
                ..Segment::default()
            },
            Segment {
                kind: MoveKind::WallOuter,
                extrusion_length: 10.0,
                extrusion_rate: 1.0,
                support_fraction: 1.0,
                ..Segment::default()
            },
        ];

        // 8.0mm scarf joint with 9 steps, 10% start height, 0.2mm layer height on planar order field
        apply_scarf_joint(&mut points, &mut segments, 8.0, 9, 0.10, 0.2, None);

        // 9 lead-in segments + 1 body segment + 3 remaining full segments + 9 lead-out segments = 22 segments
        // Waypoints: 22 + 1 = 23 points
        assert_eq!(segments.len(), 22);
        assert_eq!(points.len(), 23);

        // Lead-in start (point 0) is lowered by -0.90 * 0.2mm = -0.18mm in Z (normal = DVec3::Z)
        assert!((points[0].z - (1.0 - 0.18)).abs() < 1e-4);
        assert_eq!(points[0].x, 0.0);

        // Lead-in end (point 9) reaches nominal height (Z = 1.0) and distance 8.0mm
        assert!((points[9].z - 1.0).abs() < 1e-4);
        assert!((points[9].x - 8.0).abs() < 1e-4);

        // First lead-in segment has flow factor starting near 10% (midpoint at t = 0.5/9 ≈ 0.0556 -> 0.10 + 0.90*0.0556 ≈ 0.15)
        assert!(segments[0].extrusion_rate < 0.20);
        assert!(segments[0].extrusion_rate > 0.10);

        // Last lead-in segment has flow factor near 100% (midpoint at t = 8.5/9 ≈ 0.9444 -> 0.10 + 0.90*0.9444 ≈ 0.95)
        assert!(segments[8].extrusion_rate > 0.90);

        // Lead-out overlap starts at nominal height (Z = 1.0) from x = 0.0 to x = 8.0
        let lead_out_start_idx = 9 + 4; // after 9 lead-in + 4 body moves = idx 13
        assert_eq!(points[lead_out_start_idx].x, 0.0);
        assert_eq!(points[lead_out_start_idx].z, 1.0);

        // Volume conservation: Across the 8mm overlap, total lead-in E + total lead-out E equals exact nominal E (4.0mm of 10.0E / 20mm = 4.0E)
        let mut lead_in_e = 0.0;
        for seg in segments.iter().take(9) {
            lead_in_e += seg.extrusion_length;
        }
        let mut lead_out_e = 0.0;
        for seg in segments.iter().take(22).skip(13) {
            lead_out_e += seg.extrusion_length;
        }
        let total_scarf_e = lead_in_e + lead_out_e;
        let expected_nominal_8mm_e = 10.0 * (8.0 / 20.0); // 4.0 mm filament
        assert!(
            (total_scarf_e - expected_nominal_8mm_e).abs() < 1e-4,
            "Total scarf extrusion {total_scarf_e} must equal exact nominal volume {expected_nominal_8mm_e}"
        );
    }

    #[test]
    fn per_axis_limits_constrain_directional_moves() {
        let mut model = StandardMotionModel::default();
        // Global travel speed: 18000 mm/min = 300 mm/s
        // Global travel accel: 10000 mm/s²

        // Configure Z axis with low limits (e.g. leadscrew Z)
        let mut z_limits = AxisLimits::new();
        z_limits.speed_limit = Some(30.0); // 30 mm/s = 1800 mm/min
        z_limits.acceleration_limit = Some(1500.0); // 1500 mm/s²
        model.axis_limits.insert(Axis::Z, z_limits);

        // Pure horizontal XY move: should use full global speed & accel
        let xy_dir = DVec3::new(1.0, 0.0, 0.0);
        let xy_feed = model.max_directional_feedrate(MoveKind::Travel, false, xy_dir);
        let xy_accel =
            model.available_directional_acceleration(MoveKind::Travel, false, 200.0, xy_dir);
        assert_eq!(xy_feed, 18000.0);
        assert_eq!(xy_accel, 10000.0);

        // Pure vertical Z move: should be capped by Z axis limits
        let z_dir = DVec3::new(0.0, 0.0, 1.0);
        let z_feed = model.max_directional_feedrate(MoveKind::Travel, false, z_dir);
        let z_accel =
            model.available_directional_acceleration(MoveKind::Travel, false, 20.0, z_dir);
        assert_eq!(z_feed, 1800.0); // 30 mm/s * 60
        assert_eq!(z_accel, 1500.0);

        // 45-degree climbing move (equal XY and Z components)
        let climb_dir = DVec3::new(1.0, 0.0, 1.0).normalize(); // comp_z = 1 / sqrt(2) ~ 0.7071
        let climb_feed = model.max_directional_feedrate(MoveKind::Travel, false, climb_dir);
        let climb_accel =
            model.available_directional_acceleration(MoveKind::Travel, false, 20.0, climb_dir);
        // Linear speed along vector is capped such that v * comp_z <= 30 => v <= 30 * sqrt(2) ~ 42.42 mm/s = 2545.5 mm/min
        assert!((climb_feed - (30.0 * std::f64::consts::SQRT_2 * 60.0)).abs() < 1e-2);
        assert!((climb_accel - (1500.0 * std::f64::consts::SQRT_2)).abs() < 1e-2);
    }

    #[test]
    fn per_axis_stepper_dynamics_rolls_off_individual_axis_acceleration() {
        let mut model = StepperDynamicModel::default();
        let mut z_limits = AxisLimits::new();
        z_limits.use_stepper_dynamics = true;
        z_limits.zero_speed_acceleration = Some(2000.0);
        z_limits.max_available_speed = Some(50.0); // 50 mm/s max Z speed
        model.axis_limits.insert(Axis::Z, z_limits);

        let z_dir = DVec3::new(0.0, 0.0, 1.0);
        // At 0 speed, full 2000 mm/s² available
        let a_0 = model.available_directional_acceleration(MoveKind::Travel, false, 0.0, z_dir);
        assert_eq!(a_0, 2000.0);

        // At 25 mm/s (50% max speed), 50% torque = 1000 mm/s² available
        let a_half = model.available_directional_acceleration(MoveKind::Travel, false, 25.0, z_dir);
        assert!((a_half - 1000.0).abs() < 1e-2);
    }
}
