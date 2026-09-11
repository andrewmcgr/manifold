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
    /// Running counter of consecutive short moves ($t_{\text{move}} < 0.25 \times K_{\text{PA}}$).
    pub consecutive_short_moves: usize,
    /// Last seen steady/normal extrusion flow rate ($\text{mm}^3/\text{s}$), used to maintain
    /// a physical pressure floor across pre-retract tapers and prevent false depressurization.
    pub last_steady_flow: f64,
}

impl Default for PressureTracker {
    fn default() -> Self {
        Self {
            current_pressure: 0.0,
            min_multiplier: 0.75,
            beta: 1.0,
            pressure_floor: 0.0,
            consecutive_short_moves: 0,
            last_steady_flow: 0.0,
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
            consecutive_short_moves: 0,
            last_steady_flow: 0.0,
        }
    }

    /// Resets the internal pressure state to zero (e.g., at tool changes or unretract boundaries).
    pub fn reset(&mut self) {
        self.current_pressure = 0.0;
        self.consecutive_short_moves = 0;
        self.last_steady_flow = 0.0;
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
            self.consecutive_short_moves = 0;
            self.last_steady_flow = q_target;
            return (v_nominal, 1.0);
        }

        let short_threshold = 0.25 * k_pa;
        let is_short_move = t_move < short_threshold;

        if is_short_move {
            self.consecutive_short_moves = self.consecutive_short_moves.saturating_add(1);

            // On rapid short moves (t_move << K_PA), continuous time-step integration degenerates
            // because (1 - e^(-x))/x -> 1.0, collapsing P_avg to P_start.
            // When corners/strokes clip entry speeds, P_start is low, falsely giving M = 1.0.
            // In reality, high-frequency direction reversals build up hotend backpressure.
            // An adaptive temporal decay proxy ramps down M toward M_min based on consecutive short moves:
            let ramp = (self.consecutive_short_moves as f64 / 4.0).min(1.0);
            let short_mult =
                (1.0 - (1.0 - self.min_multiplier) * ramp).clamp(self.min_multiplier, 1.0);

            // If continuous tracker already had a lower multiplier due to high residual pressure, respect it
            let p_start = self.current_pressure;
            let x = (t_move / k_pa).max(0.0);
            let decay_ratio = integrated_exp_decay_ratio(x);
            let p_average = q_target + (p_start - q_target) * decay_ratio;
            let continuous_mult = if p_average > q_target && p_average > 1e-6 {
                let ratio = q_target / p_average;
                ratio.powf(self.beta).clamp(self.min_multiplier, 1.0)
            } else {
                1.0
            };

            let multiplier = short_mult.min(continuous_mult);
            let v_compensated = v_nominal * multiplier;
            let q_compensated = v_compensated / t_move;

            // Pressure accumulation on short moves:
            // High frequency pulses pump pressure toward q_target rather than letting it collapse
            let exp_neg_x = (-x).exp();
            let p_end = q_compensated + (p_start - q_compensated) * exp_neg_x;
            self.current_pressure = p_end.max(self.pressure_floor);

            (v_compensated, multiplier)
        } else {
            // Normal move: decay the short move counter and update last steady flow
            self.consecutive_short_moves = 0;
            self.last_steady_flow = q_target;

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
    }

    /// Decays residual melt pressure exponentially toward zero during a travel move of duration $t_{\text{move}}$.
    pub fn process_travel(&mut self, t_move: f64, k_pa: f64) {
        self.consecutive_short_moves = 0;
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

                // Enforce physical pressure floor across pre-retract tapers:
                // Pre-retract tapers slash extrusion_length down to ~20% over the last few millimeters.
                // In physical hotends, viscous resistance prevents internal melt pressure from cratering to zero.
                // If a segment's extrusion rate has been tapered below steady state, maintain a physical
                // residual pressure floor (at least 50% of the pre-taper steady flow) so that the subsequent
                // travel move and long move don't assume the nozzle is empty and cause a start-of-move bump.
                if segment.extrusion_rate < 0.90 && tracker.last_steady_flow > 1e-6 {
                    let taper_pressure_floor = tracker.last_steady_flow * 0.50;
                    if tracker.current_pressure < taper_pressure_floor {
                        tracker.current_pressure = taper_pressure_floor;
                    }
                }
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
    use crate::toolpath::Segment;

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

    #[test]
    fn rapid_micro_move_zigzags_actively_trigger_multiplier_approaching_min() {
        let mut tracker = PressureTracker::new(0.60, 1.0);
        let k_pa = 0.04; // 40ms
        let q_target = 12.0; // 12 mm³/s
                             // Ultra short micro-moves: 2ms per segment (t_move = 0.002 << 0.25 * 0.04 = 0.010)
        let t_move = 0.002;
        let v_nominal = q_target * t_move;

        // Even starting from zero residual pressure:
        tracker.current_pressure = 0.0;

        let mut multipliers = Vec::new();
        for _ in 0..12 {
            let (_, mult) = tracker.process_extrusion(v_nominal, t_move, k_pa);
            multipliers.push(mult);
        }

        // As consecutive micro-moves repeat, multiplier must ramp down towards M_min (0.60)
        let final_mult = *multipliers.last().unwrap();
        assert!(
            final_mult <= 0.65,
            "expected rapid micro-move zigzags to scale down toward M_min (0.60), got {final_mult}"
        );
        assert!(final_mult >= 0.60);
    }

    #[test]
    fn long_move_after_travel_maintains_clean_full_multiplier() {
        let mut tracker = PressureTracker::new(0.70, 1.0);
        let k_pa = 0.04;
        let q_target = 10.0;

        // Simulate preceding extrusion
        tracker.process_extrusion(q_target * 0.1, 0.1, k_pa);
        assert!(tracker.current_pressure > 0.0);

        // Preceding travel move of moderate duration (e.g. 150ms)
        tracker.process_travel(0.15, k_pa);

        // Long move beginning: e.g. 200ms move at nominal target flow
        let t_long = 0.20;
        let v_long = q_target * t_long;
        let (v_comp, mult) = tracker.process_extrusion(v_long, t_long, k_pa);

        // Post-travel long move must maintain clean M = 1.0 without over-throttling or over-extrusion pulse
        assert!(
            (mult - 1.0).abs() < 1e-4,
            "expected clean M = 1.0 on long move following travel, got {mult}"
        );
        assert!((v_comp - v_long).abs() < 1e-4);
    }

    #[test]
    fn pre_retract_taper_maintains_residual_pressure_floor_and_prevents_bump() {
        let config = SlicerConfig {
            enable_transient_pressure_compensation: true,
            pre_retract_taper_distance: Some(3.0),
            pressure_advance: Some(0.04),
            transient_pressure_min_multiplier: Some(0.70),
            path_simplify_enabled: false,
            ..SlicerConfig::default()
        };

        // Path 1 ends with tapered segments down to 0.2x rate
        let p0 = DVec3::new(0.0, 0.0, 0.2);
        let p1 = DVec3::new(20.0, 0.0, 0.2);
        let p2 = DVec3::new(23.0, 0.0, 0.2);
        let path1 = Path {
            points: vec![p0, p1, p2],
            segments: vec![
                Segment {
                    kind: MoveKind::WallOuter,
                    extrusion_length: 1.0,
                    extrusion_rate: 1.0,
                    speed: 3000.0,
                    ..Segment::default()
                },
                Segment {
                    kind: MoveKind::WallOuter,
                    extrusion_length: 0.03, // tapered tail
                    extrusion_rate: 0.20,
                    speed: 3000.0,
                    ..Segment::default()
                },
            ],
            tool: crate::ids::ToolId(0),
        };

        // Path 2 begins at (50, 50, 0.2) after a travel move, with a long extrusion
        let p3 = DVec3::new(50.0, 50.0, 0.2);
        let p4 = DVec3::new(80.0, 50.0, 0.2);
        let path2 = Path {
            points: vec![p3, p4],
            segments: vec![Segment {
                kind: MoveKind::WallOuter,
                extrusion_length: 1.5,
                extrusion_rate: 1.0,
                speed: 3000.0,
                ..Segment::default()
            }],
            tool: crate::ids::ToolId(0),
        };

        let mut paths = vec![path1, path2];
        apply_transient_flow_compensation(&mut paths, &config, None);

        // Path 2's initial long segment must receive a clean M = 1.0 (extrusion_rate == 1.0)
        assert!(
            (paths[1].segments[0].extrusion_rate - 1.0).abs() < 1e-4,
            "expected path 2 start segment to have extrusion_rate 1.0, got {}",
            paths[1].segments[0].extrusion_rate
        );
    }
}
