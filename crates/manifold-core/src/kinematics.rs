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
            MoveKind::Bridge | MoveKind::Overhang | MoveKind::DebugExcluded => self.bridge_speed,
            MoveKind::Travel | MoveKind::Wipe => self.travel_speed,
        };
        if is_first_layer && kind != MoveKind::Travel && kind != MoveKind::Wipe {
            nominal.min(self.first_layer_speed)
        } else {
            nominal
        }
    }

    fn available_acceleration(&self, kind: MoveKind, is_first_layer: bool, _v_mm_s: f64) -> f64 {
        if is_first_layer && kind != MoveKind::Travel && kind != MoveKind::Wipe {
            return self.first_layer_acceleration;
        }
        match kind {
            MoveKind::WallOuter => self.outer_wall_acceleration,
            MoveKind::WallInner => self.inner_wall_acceleration,
            MoveKind::Infill | MoveKind::TopSurface => self.infill_acceleration,
            MoveKind::Bridge | MoveKind::Overhang | MoveKind::DebugExcluded => {
                self.outer_wall_acceleration
            }
            MoveKind::Travel | MoveKind::Wipe => self.travel_acceleration,
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
    let _ = accel;
    let cos_theta = dir_in.dot(dir_out).clamp(-1.0, 1.0);
    if cos_theta >= 0.999999 {
        return 10000.0; // Collinear / straight move
    }
    if cos_theta <= -0.999999 {
        return 0.0; // 180° full reversal
    }

    // Klipper's real junction-deviation formula (`toolhead.py`'s
    // `Move.calc_junction`) is built on `junction_cos_theta =
    // -(dir_in . dir_out)` -- the NEGATED dot product, since Klipper
    // measures theta as the *turn angle* (0 for a straight line, 180°
    // for a full reversal), not the angle between the two direction
    // vectors as position vectors (which is the other way around: 0 for
    // a full reversal, 180° for a straight line). Substituting the
    // negation through Klipper's own
    // `sin_theta_d2 = sqrt(max(0, 0.5*(1 - junction_cos_theta)))` /
    // `cos_theta_d2 = sqrt(max(0, 0.5*(1 + junction_cos_theta)))` gives
    // the two lines below in terms of the plain (non-negated) dot
    // product `cos_theta` used here. Getting this sign wrong (as a
    // prior version of this function did, computing `sin_half`/`cos_half`
    // directly from `cos_theta` with no negation) inverts the entire
    // curve: velocity would *increase* with sharper turns and *decrease*
    // toward a straight line, the opposite of correct physics, and was
    // capping nearly-collinear (large-radius, finely-subdivided curved
    // wall) moves to a couple mm/s instead of near-nominal speed.
    let sin_theta_d2 = ((1.0 + cos_theta) * 0.5).max(0.0).sqrt();
    let cos_theta_d2 = ((1.0 - cos_theta) * 0.5).max(0.0).sqrt();
    if cos_theta_d2 <= 1e-9 || sin_theta_d2 >= 1.0 - 1e-9 {
        return 0.0;
    }

    // Klipper computes `junction_deviation = scv^2 * (sqrt(2) - 1) /
    // accel` once (`_calc_junction_deviation`), then
    // `move_jd_v2 = R_jd * junction_deviation * accel` where
    // `R_jd = sin_theta_d2 / (1 - sin_theta_d2)`. `accel` cancels
    // algebraically (`junction_deviation * accel = scv^2 * (sqrt(2) -
    // 1)`), leaving corner velocity a pure function of angle and `scv`
    // -- consistent with "square corner velocity" being defined
    // independent of acceleration (verified: at exactly 90°, this
    // reduces to exactly `scv`, matching Klipper's own defining
    // property and this function's existing unit test).
    //
    // Klipper also applies a second, move-length-and-accel-dependent
    // "centripetal" clamp per adjacent move (`move_centripetal_v2`).
    // That term is intentionally not replicated here, consistent with
    // this function's existing single-`accel`, no-move-length
    // simplification: distance-based reachable-speed limiting is
    // separately handled by `max_directional_reachable_speed` in the
    // caller's forward/backward passes.
    let r_jd = sin_theta_d2 / (1.0 - sin_theta_d2);
    let move_jd_v2 =
        r_jd * square_corner_velocity * square_corner_velocity * (2.0_f64.sqrt() - 1.0);
    move_jd_v2.max(0.0).sqrt()
}

/// Kinematic motion profile for a single move segment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlannedMotionProfile {
    pub entry_speed: f64,     // mm/min
    pub cruise_speed: f64,    // mm/min
    pub exit_speed: f64,      // mm/min
    pub accel_distance: f64,  // mm
    pub cruise_distance: f64, // mm
    pub decel_distance: f64,  // mm
    pub duration_seconds: f64,
}

impl PlannedMotionProfile {
    /// Evaluates the instantaneous speed (mm/s) at distance `s` (mm) from move start along total distance `total_d`.
    #[must_use]
    pub fn speed_at_distance(&self, s: f64, total_d: f64) -> f64 {
        let s = s.clamp(0.0, total_d.max(0.0));
        let v_entry = self.entry_speed / 60.0;
        let v_cruise = self.cruise_speed / 60.0;
        let v_exit = self.exit_speed / 60.0;

        if s <= self.accel_distance && self.accel_distance > 1e-6 {
            let t = (s / self.accel_distance).clamp(0.0, 1.0);
            (v_entry * v_entry + t * (v_cruise * v_cruise - v_entry * v_entry))
                .max(0.0)
                .sqrt()
        } else if s <= self.accel_distance + self.cruise_distance {
            v_cruise
        } else {
            let decel_s = s - (self.accel_distance + self.cruise_distance);
            if self.decel_distance > 1e-6 {
                let t = (decel_s / self.decel_distance).clamp(0.0, 1.0);
                (v_cruise * v_cruise + t * (v_exit * v_exit - v_cruise * v_cruise))
                    .max(0.0)
                    .sqrt()
            } else {
                v_exit
            }
        }
    }
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
    minimum_cruise_ratio: f64,
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
    let cruise_ratio = minimum_cruise_ratio.clamp(0.0, 0.999);
    for i in 0..n {
        let v_entry = entry_speeds[i];
        let v_exit = exit_speeds[i];
        let d = distances[i];
        let dir = directions[i];
        let accel = model
            .available_directional_acceleration(segments[i].kind, is_first_layer, v_entry, dir)
            .max(10.0);

        let v_cruise = if d > 1e-6 {
            let max_ramp_dist = (1.0 - cruise_ratio) * d;
            let v_cruise_sq =
                (2.0 * accel * max_ramp_dist + v_entry * v_entry + v_exit * v_exit) * 0.5;
            let v_cruise_limit = v_cruise_sq.max(0.0).sqrt();
            nominal_speeds[i]
                .min(v_cruise_limit)
                .max(v_entry)
                .max(v_exit)
        } else {
            nominal_speeds[i].min(v_entry.max(v_exit))
        };

        let mut d_accel = if accel > 1e-6 && v_cruise > v_entry {
            ((v_cruise * v_cruise - v_entry * v_entry) / (2.0 * accel)).max(0.0)
        } else {
            0.0
        };

        let mut d_decel = if accel > 1e-6 && v_cruise > v_exit {
            ((v_cruise * v_cruise - v_exit * v_exit) / (2.0 * accel)).max(0.0)
        } else {
            0.0
        };

        if d_accel + d_decel > d && d > 1e-6 {
            let scale = d / (d_accel + d_decel);
            d_accel *= scale;
            d_decel *= scale;
        }

        let d_cruise = (d - d_accel - d_decel).max(0.0);

        let duration = model.directional_move_duration(
            segments[i].kind,
            is_first_layer,
            d,
            v_entry,
            v_exit,
            dir,
        );
        profiles.push(PlannedMotionProfile {
            entry_speed: v_entry * 60.0,
            cruise_speed: v_cruise * 60.0,
            exit_speed: v_exit * 60.0,
            accel_distance: d_accel,
            cruise_distance: d_cruise,
            decel_distance: d_decel,
            duration_seconds: duration,
        });
    }

    profiles
}

/// Groups `paths` into runs of same-tool, same-layer paths connected by
/// either an exact touch or a short ("no retraction would fire") travel
/// gap, and plans velocities across each run as one continuous polyline,
/// rather than calling [`plan_path_velocities`] separately per `Path`
/// object and forcing a phantom stop-to-zero at every one of those object
/// boundaries. A non-planar slice can produce thousands of `Path` objects
/// (one per wall loop, one per infill/TPMS/tangent-fill run, one per
/// island); treating each as an isolated deceleration-to-zero-then-
/// reacceleration event -- something Klipper's real lookahead planner
/// never does for a short, non-retracting hop between them -- was
/// inflating `estimated_time_seconds` far above actual measured print
/// time, and (since the exact same profiles bake pressure-advance E
/// values and feed transient/corner flow compensation) was feeding those
/// compensation systems a fictitious accel/decel cycle that never
/// actually happens on the real machine.
///
/// `max_bridge_gap_mm` is normally `config.effective_min_travel_for_retract()`:
/// a gap up to (but not exceeding) that distance will not trigger a real
/// retraction in `gcode::emit`, so the toolhead's XY motion genuinely can
/// (and, on real hardware with cross-move lookahead, typically does)
/// continue through it -- the gap is bridged with a synthetic zero-
/// extrusion `MoveKind::Travel` connector segment (at the motion model's
/// travel feedrate) purely for velocity-continuity planning, not added to
/// any real `Path`. A gap beyond `max_bridge_gap_mm` triggers a genuine
/// retraction in the emitted Gcode -- a real stop of XY motion -- so that
/// boundary is correctly left isolated (entry/exit junction velocity 0).
///
/// Closed loops (`segments.len() == points.len()`) are represented, purely
/// for this planning pass, in the same shape as open paths by
/// materializing their wraparound closing point as an explicit extra point
/// (`segments.len() + 1 == points.len()`, `points[n] == points[0]`) --
/// this re-indexing is lossless: it's exactly the point the closed loop's
/// own final (`n - 1`-th) segment already connects to via `% points.len()`
/// wraparound, just spelled out explicitly so the loop can be spliced into
/// a combined run like any open path. The original `Path` objects (with
/// their real wraparound-indexed `points`/`segments`) are untouched --
/// only this function's own working copies are re-shaped.
///
/// Returns one `Vec<PlannedMotionProfile>` per input path, in the same
/// order, so callers can index into the result exactly as if they had
/// called `plan_path_velocities` once per path.
#[must_use]
pub fn plan_chained_path_velocities(
    paths: &[crate::toolpath::Path],
    model: &dyn MotionModel,
    first_layer_flags: &[bool],
    square_corner_velocity_mm_s: f64,
    minimum_cruise_ratio: f64,
    max_bridge_gap_mm: f64,
) -> Vec<Vec<PlannedMotionProfile>> {
    const CHAIN_EPS: f64 = 1e-4;
    let n = paths.len();
    debug_assert_eq!(first_layer_flags.len(), n);

    // Every path (open or closed) is given an "opened" point list for
    // planning purposes; see doc comment above.
    let opened_points: Vec<Vec<DVec3>> = paths
        .iter()
        .map(|p| {
            if !p.points.is_empty() && p.segments.len() == p.points.len() {
                let mut pts = p.points.clone();
                pts.push(p.points[0]);
                pts
            } else {
                p.points.clone()
            }
        })
        .collect();

    let mut result: Vec<Vec<PlannedMotionProfile>> = Vec::with_capacity(n);
    let mut i = 0;
    while i < n {
        if paths[i].segments.is_empty() || opened_points[i].len() < 2 {
            result.push(Vec::new());
            i += 1;
            continue;
        }

        // Extend the run [i..=j], recording the bridging gap distance
        // (0 for an exact touch) consumed between each adjacent pair.
        let mut j = i;
        let mut gaps: Vec<f64> = Vec::new();
        while j + 1 < n {
            if paths[j + 1].segments.is_empty() || opened_points[j + 1].len() < 2 {
                break;
            }
            if paths[j].tool != paths[j + 1].tool
                || first_layer_flags[j] != first_layer_flags[j + 1]
            {
                break;
            }
            let a = *opened_points[j].last().unwrap();
            let b = opened_points[j + 1][0];
            let gap = a.distance(b);
            if gap > max_bridge_gap_mm {
                break;
            }
            gaps.push(gap);
            j += 1;
        }

        if j == i {
            let profiles = plan_path_velocities(
                &opened_points[i],
                &paths[i].segments,
                model,
                first_layer_flags[i],
                square_corner_velocity_mm_s,
                minimum_cruise_ratio,
            );
            result.push(profiles);
            i += 1;
            continue;
        }

        let travel_speed_mm_min = model.max_feedrate(MoveKind::Travel, first_layer_flags[i]);
        let mut points: Vec<DVec3> = opened_points[i].clone();
        let mut segments: Vec<crate::toolpath::Segment> = paths[i].segments.clone();
        // Parallel to `lengths`: whether that chunk of the combined run
        // corresponds to a real input path (must be pushed to `result`) or
        // a synthetic bridging connector (must be discarded).
        let mut lengths: Vec<usize> = vec![paths[i].segments.len()];
        let mut is_real: Vec<bool> = vec![true];
        for (k, path_idx) in ((i + 1)..=j).enumerate() {
            let gap = gaps[k];
            if gap > CHAIN_EPS {
                // Bridge the gap with a synthetic zero-extrusion travel
                // connector: push the next unit's first point (the
                // connector's target) and a matching Travel segment.
                points.push(opened_points[path_idx][0]);
                segments.push(crate::toolpath::Segment {
                    kind: MoveKind::Travel,
                    speed: travel_speed_mm_min,
                    order: paths[path_idx]
                        .segments
                        .first()
                        .map(|s| s.order)
                        .unwrap_or(0.0),
                    ..crate::toolpath::Segment::default()
                });
                lengths.push(1);
                is_real.push(false);
            }
            points.extend(opened_points[path_idx].iter().skip(1).copied());
            segments.extend(paths[path_idx].segments.iter().copied());
            lengths.push(paths[path_idx].segments.len());
            is_real.push(true);
        }

        let combined_profiles = plan_path_velocities(
            &points,
            &segments,
            model,
            first_layer_flags[i],
            square_corner_velocity_mm_s,
            minimum_cruise_ratio,
        );

        let mut offset = 0;
        for (len, real) in lengths.iter().zip(is_real.iter()) {
            let chunk = combined_profiles[offset..offset + len].to_vec();
            if *real {
                result.push(chunk);
            }
            offset += len;
        }

        i = j + 1;
    }

    result
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

/// Leaves an unextruded coasting gap of length `seam_gap_mm` at the end of closed perimeter
/// wall loops preceding loop closure back to the start point, bleeding residual nozzle pressure
/// into the gap to eliminate seam blobs/zits.
pub fn apply_seam_gap(
    points: &mut Vec<DVec3>,
    segments: &mut Vec<crate::toolpath::Segment>,
    seam_gap_mm: f64,
) {
    if seam_gap_mm <= 1e-4 || points.len() < 3 || segments.is_empty() {
        return;
    }
    // Only apply to closed wall loops
    let is_wall_loop = segments
        .first()
        .is_some_and(|s| s.kind == MoveKind::WallOuter || s.kind == MoveKind::WallInner);
    if !is_wall_loop {
        return;
    }

    // Find the last extruding segment index
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

    let p_start = points[last_idx];
    let p_end = points[(last_idx + 1) % points.len()];
    let last_seg_len = (p_end - p_start).length();

    if last_seg_len > seam_gap_mm + 0.05 {
        // Split last segment into extruding lead-in + unextruded coasting tail
        let split_ratio = (last_seg_len - seam_gap_mm) / last_seg_len;
        let p_split = p_start.lerp(p_end, split_ratio);

        let orig_seg = segments[last_idx];
        let mut lead_seg = orig_seg;
        let mut coast_seg = orig_seg;

        lead_seg.extrusion_length = orig_seg.extrusion_length * split_ratio;
        coast_seg.extrusion_length = 0.0;
        coast_seg.extrusion_rate = 0.0;

        points.insert(last_idx + 1, p_split);
        segments[last_idx] = lead_seg;
        segments.insert(last_idx + 1, coast_seg);
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
        if dist_from_end + seg_len <= seam_gap_mm {
            segments[i].extrusion_length = 0.0;
            segments[i].extrusion_rate = 0.0;
            dist_from_end += seg_len;
        } else {
            let needed = seam_gap_mm - dist_from_end;
            if needed > 1e-4 && seg_len > needed + 0.05 {
                let p_s = points[i];
                let p_e = points[(i + 1) % points.len()];
                let split_ratio = (seg_len - needed) / seg_len;
                let p_split = p_s.lerp(p_e, split_ratio);

                let orig_seg = segments[i];
                let mut lead_seg = orig_seg;
                let mut coast_seg = orig_seg;

                lead_seg.extrusion_length = orig_seg.extrusion_length * split_ratio;
                coast_seg.extrusion_length = 0.0;
                coast_seg.extrusion_rate = 0.0;

                points.insert(i + 1, p_split);
                segments[i] = lead_seg;
                segments.insert(i + 1, coast_seg);
            } else {
                segments[i].extrusion_length = 0.0;
                segments[i].extrusion_rate = 0.0;
            }
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
        id: 0,
        island: last_seg.island,
        channel_width: last_seg.channel_width,
        flow_breakdown: None,
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
#[allow(clippy::too_many_arguments)]
pub fn apply_scarf_joint(
    points: &mut Vec<DVec3>,
    segments: &mut Vec<crate::toolpath::Segment>,
    scarf_length_mm: f64,
    steps: usize,
    start_height_fraction: f64,
    scarf_flow_ratio: f64,
    layer_height: f64,
    order_field: Option<&dyn manifold_fidget::order::OrderField>,
    fluid_engine: Option<&crate::fluid_dynamics::FluidDynamicsEngine>,
    compensation_mode: crate::SlopeCompensationMode,
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

    if total_len <= 1e-4 || total_len < 2.5 * scarf_length_mm {
        return;
    }

    let effective_scarf_len = scarf_length_mm.min(0.40 * total_len);
    if effective_scarf_len <= 1e-3 {
        return;
    }

    let k_steps = steps.max(1);
    let delta_s = effective_scarf_len / (k_steps as f64);
    if delta_s < 0.20 {
        return;
    }
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
        let offset = if compensation_mode == crate::SlopeCompensationMode::GeometricOffset
            && seg.support_fraction >= 0.7
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

    let calc_swell_correction = |seg: &crate::toolpath::Segment, flow_frac: f64| -> f64 {
        if let Some(engine) = fluid_engine {
            let bead_area =
                crate::extrusion::bead_cross_section_area(seg.line_width.max(0.2), layer_height);
            let nom_q = ((seg.speed / 60.0) * bead_area).max(0.01);
            let nom_swell = engine.swell_volume_multiplier(nom_q, 0.0);
            let ramp_q = (nom_q * flow_frac).max(0.01);
            let ramp_swell = engine.swell_volume_multiplier(ramp_q, 0.0);
            (ramp_swell / nom_swell.max(1.0)).clamp(0.60, 1.0)
        } else {
            1.0
        }
    };

    for (k, &pt) in lead_in_pts.iter().enumerate().take(k_steps) {
        let t = (k as f64 + 0.5) / (k_steps as f64);
        let flow_frac = h_start + (1.0 - h_start) * t;
        let s_mid = (k as f64 + 0.5) * delta_s;
        let (_, mut seg, e_per_mm) = sample_at_distance(s_mid);
        let swell_corr = calc_swell_correction(&seg, flow_frac);

        let pt_next = lead_in_pts[k + 1];
        let diff = pt_next - pt;
        let d_3d = diff.length();
        let unit_dir = if d_3d > 1e-6 {
            diff / d_3d
        } else {
            DVec3::ZERO
        };
        let wedge_climb = unit_dir.dot(crate::slicing::BUILD_DIRECTION);
        let downhill_corr = if wedge_climb < -0.05 {
            (1.0 - 0.15 * (-wedge_climb).clamp(0.0, 1.0)).clamp(0.70, 1.0)
        } else {
            1.0
        };

        let flow_mult = flow_frac * scarf_flow_ratio.clamp(0.10, 2.0);
        seg.extrusion_rate *= flow_mult;
        seg.extrusion_length = e_per_mm * delta_s * flow_mult * swell_corr * downhill_corr;
        // This ramp's actual extrusion_length was just recomputed with
        // scarf-specific flow_mult/swell_corr/downhill_corr factors that
        // have no home in FlowBreakdown's schema -- the breakdown Copy'd
        // forward from `seg` (the original wall segment sampled at this
        // point) no longer multiplies out to this segment's real
        // extrusion_length, so null it rather than show stale numbers.
        seg.flow_breakdown = None;
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
        let swell_corr = calc_swell_correction(&seg, flow_frac);

        let pt_curr = lead_out_pts[k];
        let pt_next = lead_out_pts[k + 1];
        let diff = pt_next - pt_curr;
        let d_3d = diff.length();
        let unit_dir = if d_3d > 1e-6 {
            diff / d_3d
        } else {
            DVec3::ZERO
        };
        let climb = unit_dir.dot(crate::slicing::BUILD_DIRECTION);
        let downhill_corr = if climb < -0.05 {
            (1.0 - 0.15 * (-climb).clamp(0.0, 1.0)).clamp(0.70, 1.0)
        } else {
            1.0
        };

        let flow_mult = flow_frac * scarf_flow_ratio.clamp(0.10, 2.0);
        seg.extrusion_rate *= flow_mult;
        seg.extrusion_length = e_per_mm * delta_s * flow_mult * swell_corr * downhill_corr;
        // See the identical comment on the lead-in ramp above: this
        // segment's extrusion_length was just recomputed with scarf-
        // specific factors FlowBreakdown can't represent, so null the
        // stale copied-forward breakdown.
        seg.flow_breakdown = None;
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

    /// Regression test for a sign-convention bug: an earlier version of
    /// `klipper_corner_velocity` computed its half-angle terms straight
    /// from `dir_in.dot(dir_out)` with no negation, which is backwards
    /// relative to Klipper's real `junction_cos_theta = -(dir_in .
    /// dir_out)` convention. That inverted the entire curve -- corner
    /// velocity *increased* with a sharper turn and *decreased* toward a
    /// straight line, the opposite of correct physics -- capping nearly-
    /// collinear moves (the common case along a finely-subdivided curved
    /// wall) to a couple mm/s instead of near-nominal speed, which was a
    /// dominant contributor to `estimated_time_seconds` being roughly
    /// 2.5x real measured print time for detailed non-planar prints.
    #[test]
    fn klipper_corner_velocity_decreases_monotonically_as_turn_angle_increases() {
        let scv = 6.0;
        let accel = 5000.0;
        let d_in = DVec3::new(1.0, 0.0, 0.0);

        let mut prev_v = f64::INFINITY;
        for deg in [1.0f64, 5.0, 15.0, 30.0, 60.0, 90.0, 120.0, 150.0, 179.0] {
            let rad: f64 = deg.to_radians();
            let d_out = DVec3::new(rad.cos(), rad.sin(), 0.0);
            let v = klipper_corner_velocity(d_in, d_out, scv, accel);
            assert!(
                v <= prev_v + 1e-6,
                "corner velocity must not increase as the turn sharpens: \
                 at {deg}deg got v={v}, but a smaller angle already gave {prev_v}"
            );
            prev_v = v;
        }

        // A gentle 1-degree deviation (typical of a finely-subdivided
        // smooth curve) must permit substantially faster cornering than
        // the configured square_corner_velocity (the speed for a 90-degree
        // turn) -- not less than it, which is what the sign bug produced.
        let d_out_1deg = {
            let rad = 1.0_f64.to_radians();
            DVec3::new(rad.cos(), rad.sin(), 0.0)
        };
        let v_1deg = klipper_corner_velocity(d_in, d_out_1deg, scv, accel);
        assert!(
            v_1deg > scv * 5.0,
            "a 1-degree near-straight deviation should permit much faster \
             cornering than the 90-degree square_corner_velocity ({scv}mm/s), got {v_1deg}"
        );
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
        let profiles = plan_path_velocities(&points, &segments, &model, false, 5.0, 0.5);

        assert_eq!(profiles.len(), 3);
        // First segment starts from entry 0.0
        assert_eq!(profiles[0].entry_speed, 0.0);
        // Exit speed at 90° corner is bounded by Klipper SCV (5 mm/s = 300 mm/min)
        assert!((profiles[0].exit_speed - 300.0).abs() < 10.0);
        // Last segment finishes at exit 0.0
        // Last segment finishes at exit 0.0
        assert_eq!(profiles[2].exit_speed, 0.0);
    }

    #[test]
    fn minimum_cruise_ratio_caps_peak_velocity_and_reserves_cruise_distance() {
        use crate::toolpath::Segment;

        let points = vec![DVec3::new(0.0, 0.0, 0.0), DVec3::new(0.6, 0.0, 0.0)];
        let segments = vec![Segment {
            kind: MoveKind::WallOuter,
            speed: 60000.0, // very high requested speed (1000 mm/s)
            ..Segment::default()
        }];
        let model = StandardMotionModel::default();

        // With minimum_cruise_ratio = 0.5, at least 50% of the 0.6mm move (0.3mm) must be cruise
        let profiles = plan_path_velocities(&points, &segments, &model, false, 5.0, 0.5);
        assert_eq!(profiles.len(), 1);
        let p = profiles[0];
        assert!(
            p.cruise_distance >= 0.29,
            "expected cruise >= 0.3, got {}",
            p.cruise_distance
        );
        assert!(p.accel_distance + p.decel_distance <= 0.31);

        // With minimum_cruise_ratio = 0.0, the move can use the entire 0.6mm for ramp
        let profiles_zero = plan_path_velocities(&points, &segments, &model, false, 5.0, 0.0);
        assert_eq!(profiles_zero.len(), 1);
        let p_zero = profiles_zero[0];
        assert!(p_zero.cruise_speed > p.cruise_speed);
        assert!(
            p_zero.accel_distance + p_zero.decel_distance > p.accel_distance + p.decel_distance
        );
    }

    #[test]
    fn plan_chained_path_velocities_treats_connected_open_paths_as_one_continuous_run() {
        use crate::toolpath::Segment;

        // Two open paths, same tool, same first-layer flag, where path 2 begins
        // exactly where path 1 ends: a chained infill run.
        let path1 = crate::toolpath::Path {
            points: vec![DVec3::new(0.0, 0.0, 0.0), DVec3::new(50.0, 0.0, 0.0)],
            segments: vec![Segment {
                kind: MoveKind::Infill,
                speed: 12000.0,
                ..Segment::default()
            }],
            tool: crate::ids::ToolId(0),
            object: crate::ids::ObjectId::default(),
        };
        let path2 = crate::toolpath::Path {
            points: vec![DVec3::new(50.0, 0.0, 0.0), DVec3::new(100.0, 0.0, 0.0)],
            segments: vec![Segment {
                kind: MoveKind::Infill,
                speed: 12000.0,
                ..Segment::default()
            }],
            tool: crate::ids::ToolId(0),
            object: crate::ids::ObjectId::default(),
        };
        let paths = vec![path1, path2];
        let model = StandardMotionModel::default();

        let chained = plan_chained_path_velocities(&paths, &model, &[false, false], 5.0, 0.5, 1.5);
        assert_eq!(chained.len(), 2);
        assert_eq!(chained[0].len(), 1);
        assert_eq!(chained[1].len(), 1);

        // Chained: path1's exit speed matches path2's entry speed (no forced
        // deceleration to zero at the shared boundary point).
        assert!(
            chained[0][0].exit_speed > 1.0,
            "expected non-zero exit speed when chained, got {}",
            chained[0][0].exit_speed
        );
        assert!((chained[0][0].exit_speed - chained[1][0].entry_speed).abs() < 1e-6);

        // Isolated (unchained) planning of the same two paths independently
        // forces both to decelerate to zero at their own boundaries.
        let isolated1 = plan_path_velocities(
            &paths[0].points,
            &paths[0].segments,
            &model,
            false,
            5.0,
            0.5,
        );
        assert_eq!(isolated1[0].exit_speed, 0.0);
    }

    #[test]
    fn plan_chained_path_velocities_does_not_chain_across_tool_boundaries() {
        use crate::toolpath::Segment;

        // Different tool: must not chain even though spatially contiguous.
        let path1 = crate::toolpath::Path {
            points: vec![DVec3::new(0.0, 0.0, 0.0), DVec3::new(50.0, 0.0, 0.0)],
            segments: vec![Segment {
                kind: MoveKind::Infill,
                speed: 12000.0,
                ..Segment::default()
            }],
            tool: crate::ids::ToolId(0),
            object: crate::ids::ObjectId::default(),
        };
        let path2 = crate::toolpath::Path {
            points: vec![DVec3::new(50.0, 0.0, 0.0), DVec3::new(100.0, 0.0, 0.0)],
            segments: vec![Segment {
                kind: MoveKind::Infill,
                speed: 12000.0,
                ..Segment::default()
            }],
            tool: crate::ids::ToolId(1),
            object: crate::ids::ObjectId::default(),
        };
        let model = StandardMotionModel::default();
        let chained =
            plan_chained_path_velocities(&[path1, path2], &model, &[false, false], 5.0, 0.5, 1.5);
        assert_eq!(chained[0][0].exit_speed, 0.0);
    }

    /// A closed wall loop (`points.len() == segments.len()`) immediately
    /// followed by a spatially-contiguous open path (touching exactly, zero
    /// gap) must now chain through: the loop is given an "opened"
    /// representation (see `plan_chained_path_velocities`'s doc comment)
    /// so its own exit speed is no longer artificially forced to zero.
    #[test]
    fn plan_chained_path_velocities_chains_a_closed_loop_into_a_contiguous_open_follow_up() {
        use crate::toolpath::Segment;

        let closed = crate::toolpath::Path {
            points: vec![
                DVec3::new(0.0, 0.0, 0.0),
                DVec3::new(10.0, 0.0, 0.0),
                DVec3::new(10.0, 10.0, 0.0),
            ],
            segments: vec![
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
                Segment {
                    kind: MoveKind::WallOuter,
                    speed: 6000.0,
                    ..Segment::default()
                },
            ],
            tool: crate::ids::ToolId(0),
            object: crate::ids::ObjectId::default(),
        };
        let open_follow_up = crate::toolpath::Path {
            points: vec![DVec3::new(0.0, 0.0, 0.0), DVec3::new(5.0, 0.0, 0.0)],
            segments: vec![Segment {
                kind: MoveKind::Infill,
                speed: 12000.0,
                ..Segment::default()
            }],
            tool: crate::ids::ToolId(0),
            object: crate::ids::ObjectId::default(),
        };
        let model = StandardMotionModel::default();
        let chained_closed = plan_chained_path_velocities(
            &[closed, open_follow_up],
            &model,
            &[false, false],
            5.0,
            0.5,
            1.5,
        );
        assert!(
            chained_closed[0].last().unwrap().exit_speed > 1.0,
            "expected the closed loop's exit speed to carry through into the \
             contiguous follow-up path, got {}",
            chained_closed[0].last().unwrap().exit_speed
        );
        assert!(
            (chained_closed[0].last().unwrap().exit_speed - chained_closed[1][0].entry_speed).abs()
                < 1e-6
        );
    }

    /// A short (sub-retraction-threshold) travel gap between two closed
    /// loops of the same tool/layer must be bridged with a synthetic
    /// connector rather than forcing both to decelerate to zero.
    #[test]
    fn plan_chained_path_velocities_bridges_a_short_gap_between_two_closed_loops() {
        use crate::toolpath::Segment;

        let make_loop = |origin: DVec3| crate::toolpath::Path {
            points: vec![
                origin,
                origin + DVec3::new(10.0, 0.0, 0.0),
                origin + DVec3::new(10.0, 10.0, 0.0),
            ],
            segments: vec![
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
                Segment {
                    kind: MoveKind::WallOuter,
                    speed: 6000.0,
                    ..Segment::default()
                },
            ],
            tool: crate::ids::ToolId(0),
            object: crate::ids::ObjectId::default(),
        };
        // Loop B's start is 1.0mm from loop A's end (its own start, since it's
        // closed) -- well under a 1.5mm max_bridge_gap_mm.
        let loop_a = make_loop(DVec3::new(0.0, 0.0, 0.0));
        let loop_b = make_loop(DVec3::new(1.0, 0.0, 0.0));
        let model = StandardMotionModel::default();
        let chained =
            plan_chained_path_velocities(&[loop_a, loop_b], &model, &[false, false], 5.0, 0.5, 1.5);
        assert_eq!(chained[0].len(), 3);
        assert_eq!(chained[1].len(), 3);
        assert!(
            chained[0].last().unwrap().exit_speed > 1.0,
            "expected a bridged short gap to carry nonzero speed through, got {}",
            chained[0].last().unwrap().exit_speed
        );
    }

    /// A gap larger than `max_bridge_gap_mm` (i.e. long enough that
    /// `gcode::emit` will actually retract across it) must still isolate --
    /// a real stop of XY motion is correctly modeled as such.
    #[test]
    fn plan_chained_path_velocities_does_not_bridge_a_gap_beyond_max_bridge_gap_mm() {
        use crate::toolpath::Segment;

        let make_loop = |origin: DVec3| crate::toolpath::Path {
            points: vec![
                origin,
                origin + DVec3::new(10.0, 0.0, 0.0),
                origin + DVec3::new(10.0, 10.0, 0.0),
            ],
            segments: vec![
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
                Segment {
                    kind: MoveKind::WallOuter,
                    speed: 6000.0,
                    ..Segment::default()
                },
            ],
            tool: crate::ids::ToolId(0),
            object: crate::ids::ObjectId::default(),
        };
        // Loop B's start is 20mm from loop A's end -- far beyond a 1.5mm
        // max_bridge_gap_mm, so a real retraction would fire here.
        let loop_a = make_loop(DVec3::new(0.0, 0.0, 0.0));
        let loop_b = make_loop(DVec3::new(20.0, 0.0, 0.0));
        let model = StandardMotionModel::default();
        let chained =
            plan_chained_path_velocities(&[loop_a, loop_b], &model, &[false, false], 5.0, 0.5, 1.5);
        assert_eq!(chained[0].last().unwrap().exit_speed, 0.0);
        assert_eq!(chained[1][0].entry_speed, 0.0);
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
    fn apply_seam_gap_zeroes_tail_extrusion() {
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

        apply_seam_gap(&mut points, &mut segments, 1.0);

        // Last segment (10mm) should be split into a 9mm extruding move + 1mm unextruded coast move
        assert_eq!(points.len(), 5);
        assert_eq!(segments.len(), 5);
        let coast_seg = segments[4];
        assert_eq!(coast_seg.extrusion_length, 0.0);
        assert_eq!(coast_seg.extrusion_rate, 0.0);
        // Split point should be 1mm before the end (0, 1, 0)
        assert!((points[4].y - 1.0).abs() < 1e-4);
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

        // 8.0mm scarf joint with 9 steps, 10% start height, 100% flow ratio, 0.2mm layer height on planar order field
        apply_scarf_joint(
            &mut points,
            &mut segments,
            8.0,
            9,
            0.10,
            1.00,
            0.2,
            None,
            None,
            crate::SlopeCompensationMode::GeometricOffset,
        );

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

        // Apply seam gap (1.0mm) to the scarf joint path:
        // The tail of the scarf overlap must be coasted with zero extrusion.
        apply_seam_gap(&mut points, &mut segments, 1.0);
        let last_seg = segments.last().unwrap();
        assert_eq!(last_seg.extrusion_length, 0.0);
        assert_eq!(last_seg.extrusion_rate, 0.0);
    }

    #[test]
    fn apply_scarf_joint_nulls_stale_flow_breakdown_on_ramp_segments_but_preserves_it_on_body() {
        use crate::toolpath::{FlowBreakdown, Segment};

        // Every original wall segment carries a populated flow_breakdown --
        // regression test for a bug where apply_scarf_joint's lead-in/lead-
        // out ramp segments (which recompute extrusion_length with scarf-
        // specific flow_mult/swell_corr/downhill_corr factors that don't
        // exist as FlowBreakdown fields) copied the ORIGINAL segment's
        // breakdown forward unchanged via Segment's Copy semantics --
        // showing numbers that no longer multiplied out to the ramp
        // segment's real extrusion_length.
        let original_breakdown = FlowBreakdown {
            slope_cosine: 0.95,
            first_layer_mult: 1.0,
            directional_flow_mult: 1.0,
            swell_mult: 1.02,
            corner_flow_mult: 1.0,
            transient_pressure_mult: 1.0,
        };
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
                flow_breakdown: Some(original_breakdown),
                ..Segment::default()
            };
            4
        ];

        apply_scarf_joint(
            &mut points,
            &mut segments,
            8.0,
            9,
            0.10,
            1.00,
            0.2,
            None,
            None,
            crate::SlopeCompensationMode::GeometricOffset,
        );

        // Same layout as the sibling test above: 9 lead-in + 1 body + 3
        // remaining full segments + 9 lead-out = 22 segments.
        assert_eq!(segments.len(), 22);

        for seg in segments.iter().take(9) {
            assert_eq!(
                seg.flow_breakdown, None,
                "lead-in ramp segment's stale flow_breakdown must be nulled, not copied forward"
            );
        }
        for seg in segments.iter().skip(13).take(9) {
            assert_eq!(
                seg.flow_breakdown, None,
                "lead-out ramp segment's stale flow_breakdown must be nulled, not copied forward"
            );
        }

        // The remainder-of-spanning-segment and subsequent full body
        // segments (indices 9..13) only prorate length -- no new
        // multiplier factor is introduced, so the original breakdown
        // still accurately describes them and must be preserved.
        for seg in segments.iter().take(13).skip(9) {
            assert_eq!(
                seg.flow_breakdown,
                Some(original_breakdown),
                "body segments must keep their real, still-accurate flow_breakdown"
            );
        }
    }

    #[test]
    fn apply_scarf_joint_with_fluid_engine_applies_low_flow_swell_correction() {
        use crate::fluid_dynamics::{FluidDynamicsConfig, FluidDynamicsEngine};
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
                speed: 6000.0, // 100 mm/s
                extrusion_length: 10.0,
                extrusion_rate: 1.0,
                support_fraction: 1.0,
                line_width: 0.4,
                ..Segment::default()
            },
            Segment {
                kind: MoveKind::WallOuter,
                speed: 6000.0,
                extrusion_length: 10.0,
                extrusion_rate: 1.0,
                support_fraction: 1.0,
                line_width: 0.4,
                ..Segment::default()
            },
            Segment {
                kind: MoveKind::WallOuter,
                speed: 6000.0,
                extrusion_length: 10.0,
                extrusion_rate: 1.0,
                support_fraction: 1.0,
                line_width: 0.4,
                ..Segment::default()
            },
            Segment {
                kind: MoveKind::WallOuter,
                speed: 6000.0,
                extrusion_length: 10.0,
                extrusion_rate: 1.0,
                support_fraction: 1.0,
                line_width: 0.4,
                ..Segment::default()
            },
        ];

        let fluid_cfg = FluidDynamicsConfig {
            swell_ratio_low: Some(1.01),
            swell_ratio_high: Some(1.15),
            ..FluidDynamicsConfig::default()
        };
        let engine = FluidDynamicsEngine::new(fluid_cfg);

        apply_scarf_joint(
            &mut points,
            &mut segments,
            8.0,
            9,
            0.10,
            1.00,
            0.2,
            None,
            Some(&engine),
            crate::SlopeCompensationMode::GeometricOffset,
        );

        // At low flow (step 0), extrusion length should be reduced by the swell correction factor (ramp_swell / nom_swell < 1.0)
        let uncorrected_step0_e = (10.0 / 20.0) * (8.0 / 9.0) * (0.10 + 0.90 * (0.5 / 9.0));
        assert!(
            segments[0].extrusion_length < uncorrected_step0_e,
            "Step 0 extrusion {} must be strictly less than uncorrected {}",
            segments[0].extrusion_length,
            uncorrected_step0_e
        );
    }

    #[test]
    fn apply_scarf_joint_volumetric_mode_leaves_points_at_nominal_centerline() {
        let mut points = vec![
            DVec3::new(0.0, 0.0, 5.0),
            DVec3::new(10.0, 0.0, 5.0),
            DVec3::new(10.0, 10.0, 5.0),
            DVec3::new(0.0, 10.0, 5.0),
        ];
        let mut segments = vec![
            crate::toolpath::Segment {
                island: 0,
                kind: MoveKind::WallOuter,
                speed: 60.0,
                extrusion_rate: 1.0,
                support_fraction: 1.0,
                order: 5.0,
                extrusion_length: 1.0,
                line_width: 0.4,
                is_scarf: false,
                id: 0,
                channel_width: f64::INFINITY,
                flow_breakdown: None,
            };
            4
        ];

        apply_scarf_joint(
            &mut points,
            &mut segments,
            8.0,
            9,
            0.10,
            1.00,
            0.2,
            None,
            None,
            crate::SlopeCompensationMode::VolumetricModulation,
        );

        // In volumetric modulation mode, Z positions must remain strictly at 5.0 (no normal wedging)
        for p in &points {
            assert_eq!(p.z, 5.0);
        }
        // Segments are still partitioned and tagged as scarf
        assert!(segments.iter().take(9).all(|s| s.is_scarf));
    }

    #[test]
    fn apply_scarf_joint_scales_total_volume_with_flow_ratio() {
        let mut points = vec![
            DVec3::new(0.0, 0.0, 1.0),
            DVec3::new(20.0, 0.0, 1.0),
            DVec3::new(20.0, 20.0, 1.0),
            DVec3::new(0.0, 20.0, 1.0),
        ];
        let mut segments = vec![
            crate::toolpath::Segment {
                island: 0,
                kind: MoveKind::WallOuter,
                speed: 60.0,
                extrusion_rate: 1.0,
                support_fraction: 1.0,
                order: 1.0,
                extrusion_length: 10.0,
                line_width: 0.4,
                is_scarf: false,
                id: 0,
                channel_width: f64::INFINITY,
                flow_breakdown: None,
            };
            4
        ];

        apply_scarf_joint(
            &mut points,
            &mut segments,
            8.0,
            9,
            0.10,
            0.90, // 90% default flow ratio
            0.2,
            None,
            None,
            crate::SlopeCompensationMode::GeometricOffset,
        );

        let mut lead_in_e = 0.0;
        for seg in segments.iter().take(9) {
            lead_in_e += seg.extrusion_length;
        }
        let mut lead_out_e = 0.0;
        for seg in segments.iter().take(22).skip(13) {
            lead_out_e += seg.extrusion_length;
        }
        let total_scarf_e = lead_in_e + lead_out_e;
        let expected_nominal_8mm_e = 10.0 * (8.0 / 20.0) * 0.90; // 3.6 mm filament
        assert!(
            (total_scarf_e - expected_nominal_8mm_e).abs() < 1e-4,
            "Total scarf extrusion {total_scarf_e} must equal scaled volume {expected_nominal_8mm_e}"
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
