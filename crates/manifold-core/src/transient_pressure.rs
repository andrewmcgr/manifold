//! Time-based dynamic residual pressure flow compensation.
//!
//! Models the hotend melt zone as a first-order differential system (RC circuit analog)
//! using the Pressure Advance constant $K = C_{\text{PA}}$ (in seconds).
//!
//! Across consecutive toolpath segments, internal nozzle pressure $P(t)$ accumulates
//! during rapid short strokes where segment execution time $t_{\text{move}} \ll K$.
//! When average pressure $P_{\text{average}}$ exceeds target flow $Q_{\text{target}}$,
//! the commanded volume is scaled down ($V_{\text{compensated}} = V_{\text{nominal}} \cdot M$)
//! using the residual nozzle pressure to supply the difference, preventing overextrusion
//! bulges on dense infill zig-zags and small gaps.
//!
//! See `TIME_BASED_PA.md` and `RETRACTION_AND_PA.md`.

use crate::fluid_dynamics::FluidDynamicsEngine;
use crate::kinematics::plan_path_velocities;
use crate::machine::Machine;
use crate::toolpath::{MoveKind, Path};
use crate::SlicerConfig;
use glam::DVec3;

/// Evaluates the normalized integral $\frac{1 - e^{-x}}{x}$ with a Taylor series
/// expansion for small $x$ to avoid floating-point cancellation.
///
/// $$\frac{1 - e^{-x}}{x} = 1 - \frac{x}{2!} + \frac{x^2}{3!} - \frac{x^3}{4!} + \mathcal{O}(x^4)$$
#[inline]
#[must_use]
pub fn integrated_exp_decay_ratio(x: f64) -> f64 {
    if x.abs() < 1e-4 {
        1.0 - (x * 0.5) + (x * x / 6.0) - (x * x * x / 24.0)
    } else {
        (1.0 - (-x).exp()) / x
    }
}

/// Stateful tracker modeling hotend melt zone pressure $P(t)$ across sequential segments.
#[derive(Debug, Clone, PartialEq)]
pub struct PressureTracker {
    /// Current internal nozzle pressure in equivalent steady-state flow units ($\text{mm}^3/\text{s}$).
    pub current_pressure: f64,
    /// Minimum allowed flow multiplier $M_{\text{min}} \in [0.1, 1.0]$ to prevent total starvation.
    pub min_multiplier: f64,
    /// Sensitivity exponent $\beta$ scaling the flow compensation ratio $(Q_{\text{target}} / P_{\text{average}})^\beta$.
    pub beta: f64,
    /// Physical pressure floor ($\text{mm}^3/\text{s}$) to prevent unbounded negative values during retractions.
    pub pressure_floor: f64,
}

impl Default for PressureTracker {
    fn default() -> Self {
        Self {
            current_pressure: 0.0,
            min_multiplier: 0.75,
            beta: 1.0,
            pressure_floor: 0.0,
        }
    }
}

impl PressureTracker {
    /// Creates a new `PressureTracker` with the given minimum compensation multiplier $M_{\text{min}}$
    /// and sensitivity exponent $\beta$.
    #[must_use]
    pub fn new(min_multiplier: f64, beta: f64) -> Self {
        Self {
            current_pressure: 0.0,
            min_multiplier: min_multiplier.clamp(0.1, 1.0),
            beta: beta.clamp(0.05, 5.0),
            pressure_floor: 0.0,
        }
    }

    /// Resets the internal pressure state to zero (e.g., at tool changes or unretract boundaries).
    pub fn reset(&mut self) {
        self.current_pressure = 0.0;
    }

    /// Processes an extruding segment of nominal volume $V_{\text{nominal}}$ ($\text{mm}^3$)
    /// and duration $t_{\text{move}}$ (seconds) under Pressure Advance time constant $K$ (seconds).
    ///
    /// Returns `(v_compensated, multiplier)` where:
    /// - `multiplier`: $M \in [M_{\text{min}}, 1.0]$
    /// - `v_compensated`: $V_{\text{nominal}} \cdot M$
    pub fn process_extrusion(&mut self, v_nominal: f64, t_move: f64, k_pa: f64) -> (f64, f64) {
        if t_move <= 1e-9 || v_nominal <= 0.0 {
            return (v_nominal, 1.0);
        }

        let q_target = v_nominal / t_move;

        // If PA elasticity is zero or negligible, hotend responds instantaneously.
        if k_pa <= 1e-6 {
            self.current_pressure = q_target;
            return (v_nominal, 1.0);
        }

        let x = (t_move / k_pa).max(0.0);
        let decay_ratio = integrated_exp_decay_ratio(x);
        let p_start = self.current_pressure;

        // P_average = Q_target + (P_start - Q_target) * ((1 - e^(-x)) / x)
        let p_average = q_target + (p_start - q_target) * decay_ratio;

        // If average pressure exceeds target flow, nozzle is pre-pressurized from previous moves.
        let multiplier = if p_average > q_target && p_average > 1e-6 {
            let ratio = q_target / p_average;
            ratio.powf(self.beta).clamp(self.min_multiplier, 1.0)
        } else {
            1.0
        };

        let v_compensated = v_nominal * multiplier;
        let q_compensated = v_compensated / t_move;

        // Update state with actual compensated flow rate:
        // P_end = Q_compensated + (P_start - Q_compensated) * e^(-x)
        let exp_neg_x = (-x).exp();
        let p_end = q_compensated + (p_start - q_compensated) * exp_neg_x;
        self.current_pressure = p_end.max(self.pressure_floor);

        (v_compensated, multiplier)
    }

    /// Decays residual melt pressure exponentially toward zero during a travel move of duration $t_{\text{move}}$.
    pub fn process_travel(&mut self, t_move: f64, k_pa: f64) {
        if t_move <= 1e-9 || self.current_pressure <= 1e-6 {
            return;
        }

        if k_pa <= 1e-6 {
            self.current_pressure = 0.0;
            return;
        }

        let x = (t_move / k_pa).max(0.0);
        self.current_pressure = (self.current_pressure * (-x).exp()).max(0.0);
    }

    /// Models melt pressure relief during a retraction move of negative volume $V_{\text{nominal}}$ (< 0.0).
    pub fn process_retraction(&mut self, v_nominal: f64, t_move: f64, k_pa: f64) {
        if t_move <= 1e-9 {
            return;
        }

        let q_retract = v_nominal / t_move;
        if k_pa <= 1e-6 {
            self.current_pressure = self.pressure_floor;
            return;
        }

        let x = (t_move / k_pa).max(0.0);
        let p_end = q_retract + (self.current_pressure - q_retract) * (-x).exp();
        self.current_pressure = p_end.max(self.pressure_floor);
    }
}

/// Applies time-based transient nozzle pressure flow compensation across all toolpaths.
///
/// Modulates each segment's `extrusion_length` based on residual nozzle pressure
/// accumulated across previous moves, preventing overextrusion on rapid short-stroke infill.
pub fn apply_transient_flow_compensation(
    paths: &mut [Path],
    config: &SlicerConfig,
    machine: Option<&Machine>,
) {
    if !config.enable_transient_pressure_compensation || paths.is_empty() {
        return;
    }

    let min_mult = config.transient_pressure_min_multiplier();
    let beta = config.transient_pressure_beta();
    let mut tracker = PressureTracker::new(min_mult, beta);

    let motion_model = config.resolved_motion_model(machine);
    let filament_area =
        std::f64::consts::PI * 0.25 * config.filament_diameter * config.filament_diameter;
    let static_pa = config.pressure_advance.unwrap_or(0.0);

    let min_order = paths
        .iter()
        .filter_map(|path| path.segments.first())
        .map(|segment| segment.order)
        .fold(f64::INFINITY, f64::min);

    let mut current_tool = None;
    let mut last_pos: Option<DVec3> = None;
    let mut fluid_engine: Option<FluidDynamicsEngine> = None;

    for path in paths.iter_mut() {
        if path.segments.is_empty() || path.points.is_empty() {
            continue;
        }

        // Handle tool changes
        if current_tool != Some(path.tool) {
            current_tool = Some(path.tool);
            tracker.reset();
            let tool_temp = machine
                .and_then(|m| m.tools.iter().find(|t| t.id == path.tool))
                .map(crate::tool::Tool::nozzle_temperature);
            fluid_engine = config.fluid_dynamics_engine(tool_temp);
        }

        let path_order = path.segments.first().map(|s| s.order).unwrap_or(0.0);
        let is_first_layer = (path_order - min_order).abs() < 1e-4;

        // Kinematic profile gives exact duration_seconds for each segment
        let profiles = plan_path_velocities(
            &path.points,
            &path.segments,
            motion_model.as_ref(),
            is_first_layer,
            config.square_corner_velocity(),
            config.minimum_cruise_ratio(),
        );

        if profiles.len() != path.segments.len() {
            continue;
        }

        // Inter-path positioning travel move
        if let Some(prev) = last_pos {
            let start_pt = path.points[0];
            let travel_dist = (start_pt - prev).length();
            if travel_dist > 1e-6 {
                let travel_feed =
                    motion_model.max_feedrate(MoveKind::Travel, is_first_layer) / 60.0;
                let travel_time = travel_dist / travel_feed.max(1.0);
                let k_pa = if let Some(ref engine) = fluid_engine {
                    engine.dynamic_pressure_advance(1.0, 0.0)
                } else {
                    static_pa
                };
                tracker.process_travel(travel_time, k_pa);
            }
        }

        // Process segments along the path
        for (i, segment) in path.segments.iter_mut().enumerate() {
            let t_move = profiles[i].duration_seconds;
            let is_extruding = segment.extrusion_length > 0.0
                && segment.kind != MoveKind::Travel
                && segment.kind != MoveKind::DebugExcluded;

            if is_extruding && t_move > 1e-9 {
                let v_nominal = segment.extrusion_length * filament_area; // mm³
                let q_nominal = v_nominal / t_move; // mm³/s

                let k_pa = if let Some(ref engine) = fluid_engine {
                    engine.dynamic_pressure_advance(q_nominal, 0.0)
                } else {
                    static_pa
                };

                let (v_compensated, multiplier) =
                    tracker.process_extrusion(v_nominal, t_move, k_pa);
                segment.extrusion_length = v_compensated / filament_area;
                segment.extrusion_rate *= multiplier;
            } else if segment.kind == MoveKind::Travel || segment.extrusion_length <= 0.0 {
                let k_pa = if let Some(ref engine) = fluid_engine {
                    engine.dynamic_pressure_advance(1.0, 0.0)
                } else {
                    static_pa
                };
                tracker.process_travel(t_move, k_pa);
            }
        }

        last_pos = path.points.last().copied();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integrated_exp_decay_ratio_taylor_series_matches_analytical() {
        // Test normal range where direct analytical formula is accurate
        let x_norm: f64 = 0.5;
        let direct = (1.0 - (-x_norm).exp()) / x_norm;
        let approx = integrated_exp_decay_ratio(x_norm);
        assert!((direct - approx).abs() < 1e-9);

        // Test near-zero range where Taylor series prevents floating point cancellation
        let x_tiny = 1e-6;
        let result = integrated_exp_decay_ratio(x_tiny);
        let expected = 1.0 - (x_tiny * 0.5);
        assert!((result - expected).abs() < 1e-9);

        // Zero limit is exactly 1.0
        assert!((integrated_exp_decay_ratio(0.0) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn single_move_at_steady_state_preserves_full_flow() {
        let mut tracker = PressureTracker::new(0.75, 1.0);
        let k_pa = 0.04; // 40ms
        let q_target = 10.0; // 10 mm³/s
        let t_move = 0.10; // 100ms
        let v_nominal = q_target * t_move;

        // Start from steady state pressure
        tracker.current_pressure = q_target;

        let (v_compensated, mult) = tracker.process_extrusion(v_nominal, t_move, k_pa);
        assert!((mult - 1.0).abs() < 1e-5);
        assert!((v_compensated - v_nominal).abs() < 1e-5);
        assert!((tracker.current_pressure - q_target).abs() < 1e-5);
    }

    #[test]
    fn rapid_short_infill_accumulates_pressure_and_throttles_flow() {
        let mut tracker = PressureTracker::new(0.75, 1.0);
        let k_pa = 0.05; // 50ms
        let q_target = 15.0; // 15 mm³/s
        let t_move = 0.005; // 5ms per short zig-zag stroke (e.g. 0.5mm at 100mm/s)
        let v_nominal = q_target * t_move;

        // Start from high residual pressure from a preceding long fast travel/infill move
        tracker.current_pressure = 25.0;

        let mut multipliers = Vec::new();
        for _ in 0..10 {
            let (_, mult) = tracker.process_extrusion(v_nominal, t_move, k_pa);
            multipliers.push(mult);
        }

        // First move with high residual pressure should be throttled
        assert!(multipliers[0] < 1.0);
        // Multiplier must respect safety bound M_min
        for &m in &multipliers {
            assert!(m >= 0.75);
            assert!(m <= 1.0);
        }
        // Pressure should progressively bleed down toward q_target
        assert!(tracker.current_pressure < 25.0);
    }

    #[test]
    fn travel_move_decays_residual_pressure_exponentially() {
        let mut tracker = PressureTracker::new(0.75, 1.0);
        tracker.current_pressure = 20.0;
        let k_pa = 0.04;
        let t_travel = 0.08; // 2 time constants -> e^(-2) ≈ 0.1353

        tracker.process_travel(t_travel, k_pa);
        let expected = 20.0 * (-2.0_f64).exp();
        assert!((tracker.current_pressure - expected).abs() < 1e-4);

        // Long travel should bleed pressure to near zero
        tracker.process_travel(0.50, k_pa);
        assert!(tracker.current_pressure < 1e-3);
    }

    #[test]
    fn retraction_relieves_pressure_and_clamps_to_floor() {
        let mut tracker = PressureTracker::new(0.75, 1.0);
        tracker.current_pressure = 10.0;
        let k_pa = 0.04;
        let v_retract = -2.0; // -2 mm³ retraction
        let t_retract = 0.05;

        tracker.process_retraction(v_retract, t_retract, k_pa);
        assert!(tracker.current_pressure >= tracker.pressure_floor);
    }

    #[test]
    fn beta_sensitivity_scales_compensation_nonlinearly() {
        let k_pa = 0.04;
        let q_target = 10.0;
        let t_move = 0.01;
        let v_nominal = q_target * t_move;

        // Baseline beta = 1.0
        let mut tracker_base = PressureTracker::new(0.50, 1.0);
        tracker_base.current_pressure = 20.0;
        let (_, mult_base) = tracker_base.process_extrusion(v_nominal, t_move, k_pa);

        // Sublinear beta = 0.5 (less aggressive reduction)
        let mut tracker_mild = PressureTracker::new(0.50, 0.5);
        tracker_mild.current_pressure = 20.0;
        let (_, mult_mild) = tracker_mild.process_extrusion(v_nominal, t_move, k_pa);

        // Superlinear beta = 2.0 (more aggressive reduction)
        let mut tracker_sharp = PressureTracker::new(0.50, 2.0);
        tracker_sharp.current_pressure = 20.0;
        let (_, mult_sharp) = tracker_sharp.process_extrusion(v_nominal, t_move, k_pa);

        assert!(mult_sharp < mult_base);
        assert!(mult_base < mult_mild);
    }
}
