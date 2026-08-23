//! Kinematics, motion modeling, and extrusion rate control.
//!
//! Provides pluggable motion models ([`MotionModel`]), stepper dynamic torque
//! roll-off modeling, per-move-type speed/acceleration limits, and volumetric flow
//! constraints.

use crate::toolpath::MoveKind;
use serde::{Deserialize, Serialize};

/// Pluggable interface for motion kinematics and acceleration constraints.
pub trait MotionModel: Send + Sync {
    /// Maximum target feedrate for a move kind (in mm/min).
    fn max_feedrate(&self, kind: MoveKind, is_first_layer: bool) -> f64;

    /// Available acceleration at current speed `v_mm_s` (in mm/s²).
    fn available_acceleration(&self, kind: MoveKind, is_first_layer: bool, v_mm_s: f64) -> f64;
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
        }
    }
}

impl MotionModel for StandardMotionModel {
    fn max_feedrate(&self, kind: MoveKind, is_first_layer: bool) -> f64 {
        if is_first_layer && kind != MoveKind::Travel {
            return self.first_layer_speed;
        }
        match kind {
            MoveKind::WallOuter => self.outer_wall_speed,
            MoveKind::WallInner => self.inner_wall_speed,
            MoveKind::Infill => self.infill_speed,
            MoveKind::TopSurface => self.solid_infill_speed,
            MoveKind::Bridge | MoveKind::Overhang => self.bridge_speed,
            MoveKind::Travel => self.travel_speed,
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
}

/// Stepper motor dynamic performance model.
///
/// Models the physical torque/back-EMF roll-off curve of stepper motors:
/// - Maximum available acceleration `a_max_zero_v` (mm/s²) at zero velocity ($a_0$).
/// - Maximum attainable velocity `v_max_zero_a` (mm/s) where torque/acceleration drops to zero ($v_{\text{max}}$).
/// - Linearly interpolates available acceleration as:
///   $$a(v) = a_0 \cdot \max\left(0, 1 - \frac{v}{v_{\text{max}}}\right)$$
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StepperDynamicModel {
    pub standard_model: StandardMotionModel,
    /// Maximum acceleration at zero velocity (mm/s²).
    pub a_max_zero_v: f64,
    /// Maximum velocity at zero acceleration (mm/s, due to back-EMF / torque limit).
    pub v_max_zero_a: f64,
}

impl Default for StepperDynamicModel {
    fn default() -> Self {
        Self {
            standard_model: StandardMotionModel::default(),
            a_max_zero_v: 15000.0,
            v_max_zero_a: 500.0, // 500 mm/s (30,000 mm/min)
        }
    }
}

impl MotionModel for StepperDynamicModel {
    fn max_feedrate(&self, kind: MoveKind, is_first_layer: bool) -> f64 {
        let std_max = self.standard_model.max_feedrate(kind, is_first_layer);
        let stepper_max = self.v_max_zero_a * 60.0;
        std_max.min(stepper_max)
    }

    fn available_acceleration(&self, kind: MoveKind, is_first_layer: bool, v_mm_s: f64) -> f64 {
        let std_accel = self
            .standard_model
            .available_acceleration(kind, is_first_layer, v_mm_s);
        let v_clamped = v_mm_s.clamp(0.0, self.v_max_zero_a);
        let factor = (1.0 - (v_clamped / self.v_max_zero_a).clamp(0.0, 1.0)).max(0.0);
        let stepper_accel = self.a_max_zero_v * factor;
        std_accel.min(stepper_accel).max(100.0)
    }
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
            a_max_zero_v: 10000.0,
            v_max_zero_a: 500.0,
            standard_model: StandardMotionModel {
                outer_wall_acceleration: 20000.0,
                ..StandardMotionModel::default()
            },
        };

        // At v = 0, full acceleration (10000)
        let a_0 = model.available_acceleration(MoveKind::WallOuter, false, 0.0);
        assert!((a_0 - 10000.0).abs() < 1e-3);

        // At v = 250 mm/s (half speed), half acceleration (5000)
        let a_half = model.available_acceleration(MoveKind::WallOuter, false, 250.0);
        assert!((a_half - 5000.0).abs() < 1e-3);

        // At v = 500 mm/s (max speed), acceleration clamped to minimum floor (100)
        let a_max = model.available_acceleration(MoveKind::WallOuter, false, 500.0);
        assert!((a_max - 100.0).abs() < 1e-3);
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
}
