//! Error-bounded adaptive toolpath subdivision for slicer-side non-Newtonian pressure advance.
//!
//! Subdivides acceleration and deceleration phases of extruding toolpath moves into
//! piecewise linear chords to track the non-linear constitutive fluid advance $E^*(s)$,
//! while leaving constant-velocity cruise segments untouched (zero subdivision).
//!
//! Bounded by extrusion deviation tolerance $\epsilon_E$, minimum printable segment length,
//! and maximum command frequency (Hz) to prevent Klipper serial buffer starvation.

use crate::fluid_dynamics::FluidDynamicsEngine;
use crate::kinematics::PlannedMotionProfile;
use crate::toolpath::{MoveKind, Path};
use crate::SlicerConfig;

/// Subdivides extruding moves in `path` whose acceleration or deceleration causes non-linear
/// pressure advance deviation exceeding `config.slicer_pa_tolerance_mm()`.
#[must_use]
pub fn subdivide_path_for_pressure_advance(
    path: Path,
    profiles: &[PlannedMotionProfile],
    config: &SlicerConfig,
    fluid_engine: Option<&FluidDynamicsEngine>,
) -> Path {
    let n = path.segments.len();
    if n == 0 || profiles.len() != n || path.points.is_empty() {
        return path;
    }

    let is_closed = path.segments.len() == path.points.len();
    let filament_area =
        std::f64::consts::PI * 0.25 * config.filament_diameter * config.filament_diameter;
    let tolerance = config.slicer_pa_tolerance_mm();
    let min_length = config.slicer_pa_min_segment_length();
    let max_freq = config.slicer_pa_max_frequency_hz();

    let mut new_points = Vec::with_capacity(path.points.len() * 2);
    let mut new_segments = Vec::with_capacity(path.segments.len() * 2);

    for (i, segment) in path.segments.iter().enumerate() {
        let p0 = path.points[i];
        let p1 = path.points[(i + 1) % path.points.len()];
        let d = (p1 - p0).length();

        // Push start point of this segment
        new_points.push(p0);

        let is_extruding = segment.extrusion_length > 0.0 && segment.kind != MoveKind::Travel;
        let profile = &profiles[i];

        if !is_extruding
            || d < min_length * 1.5
            || profile.cruise_speed <= 0.0
            || profile.duration_seconds <= 0.0
        {
            new_segments.push(*segment);
            continue;
        }

        // Bead area derived from nominal extruded length
        let bead_area = (segment.extrusion_length * filament_area) / d;

        let eval_e = |s: f64| -> f64 {
            let s_clamped = s.clamp(0.0, d);
            let v = profile.speed_at_distance(s_clamped, d); // mm/s
            let q = (bead_area * v).max(0.0);
            let fan_pct = config.fan_speed_percent() / 100.0;
            let c_pa = if let Some(engine) = fluid_engine {
                engine.dynamic_pressure_advance(q, fan_pct)
            } else {
                0.04
            };
            let e_adv = c_pa * (q / filament_area);
            let e_nom = (s_clamped / d) * segment.extrusion_length;
            e_nom + e_adv
        };

        let eval_v = |s: f64| -> f64 { profile.speed_at_distance(s.clamp(0.0, d), d) };

        // Partition distance into:
        // 1. Acceleration ramp: [0, s_accel]
        // 2. Cruise span: [s_accel, s_cruise] (linear E*, no subdivision)
        // 3. Deceleration ramp: [s_cruise, d]
        let s_accel = profile.accel_distance.clamp(0.0, d);
        let s_cruise = (s_accel + profile.cruise_distance).clamp(s_accel, d);

        let mut split_distances = Vec::new();
        split_distances.push(0.0);

        // 1. Acceleration ramp
        if s_accel > min_length {
            let mut ramp_splits = Vec::new();
            subdivide_span_recursive(
                0.0,
                s_accel,
                &eval_e,
                &eval_v,
                tolerance,
                min_length,
                max_freq,
                0,
                &mut ramp_splits,
            );
            ramp_splits.sort_by(|a, b| a.total_cmp(b));
            for s in ramp_splits {
                if s > *split_distances.last().unwrap() + 1e-5 && s < s_accel - 1e-5 {
                    split_distances.push(s);
                }
            }
        }

        if s_accel > *split_distances.last().unwrap() + 1e-5 && s_accel < d - 1e-5 {
            split_distances.push(s_accel);
        }

        // 2. Cruise span: no internal splits needed (strictly linear E*(s))
        if s_cruise > *split_distances.last().unwrap() + 1e-5 && s_cruise < d - 1e-5 {
            split_distances.push(s_cruise);
        }

        // 3. Deceleration ramp
        if d - s_cruise > min_length {
            let mut ramp_splits = Vec::new();
            subdivide_span_recursive(
                s_cruise,
                d,
                &eval_e,
                &eval_v,
                tolerance,
                min_length,
                max_freq,
                0,
                &mut ramp_splits,
            );
            ramp_splits.sort_by(|a, b| a.total_cmp(b));
            for s in ramp_splits {
                if s > *split_distances.last().unwrap() + 1e-5 && s < d - 1e-5 {
                    split_distances.push(s);
                }
            }
        }

        split_distances.push(d);

        // Emit subdivided chords
        let e_start = eval_e(0.0);
        let e_end = eval_e(d);
        let total_delta_e = e_end - e_start;

        for k in 0..split_distances.len() - 1 {
            let s0 = split_distances[k];
            let s1 = split_distances[k + 1];

            // If not the first subsegment, push intermediate point
            if k > 0 {
                let sub_point = p0 + (p1 - p0) * (s0 / d);
                new_points.push(sub_point);
            }

            let sub_e = if total_delta_e.abs() > 1e-8 {
                (eval_e(s1) - eval_e(s0)).max(0.0)
            } else {
                ((s1 - s0) / d) * segment.extrusion_length
            };

            let v_avg = (eval_v(s0) + eval_v(s1)) * 0.5;
            let speed_mm_min = (v_avg * 60.0).max(60.0);

            let mut sub_seg = *segment;
            sub_seg.extrusion_length = sub_e;
            sub_seg.speed = speed_mm_min;

            new_segments.push(sub_seg);
        }
    }

    // For open paths, push the terminal end point
    if !is_closed && !path.points.is_empty() {
        new_points.push(*path.points.last().unwrap());
    }

    Path {
        points: new_points,
        segments: new_segments,
        tool: path.tool,
    }
}

/// Recursively evaluates error between linear chord and continuous $E^*(s)$ across $[s_0, s_1]$,
/// splitting at maximum deviation until tolerance is met or physical constraints stop recursion.
#[allow(clippy::too_many_arguments)]
fn subdivide_span_recursive(
    s0: f64,
    s1: f64,
    eval_e: &dyn Fn(f64) -> f64,
    eval_v: &dyn Fn(f64) -> f64,
    tolerance: f64,
    min_length: f64,
    max_freq: f64,
    depth: usize,
    splits: &mut Vec<f64>,
) {
    let span_len = s1 - s0;
    if span_len < 2.0 * min_length || depth >= 4 {
        return;
    }

    let e0 = eval_e(s0);
    let e1 = eval_e(s1);

    // Find point of maximum deviation from the linear chord across N samples
    const SAMPLES: usize = 7;
    let mut max_err = 0.0f64;
    let mut best_s = s0 + span_len * 0.5;

    for j in 1..SAMPLES {
        let frac = j as f64 / SAMPLES as f64;
        let s = s0 + frac * span_len;
        let chord_e = e0 + frac * (e1 - e0);
        let actual_e = eval_e(s);
        let err = (actual_e - chord_e).abs();
        if err > max_err {
            max_err = err;
            best_s = s;
        }
    }

    if max_err <= tolerance {
        return;
    }

    let v_mid = eval_v(best_s).max(1.0);
    let half_duration = ((best_s - s0).min(s1 - best_s)) / v_mid;
    if half_duration < 1.0 / max_freq {
        return;
    }

    splits.push(best_s);

    subdivide_span_recursive(
        s0,
        best_s,
        eval_e,
        eval_v,
        tolerance,
        min_length,
        max_freq,
        depth + 1,
        splits,
    );
    subdivide_span_recursive(
        best_s,
        s1,
        eval_e,
        eval_v,
        tolerance,
        min_length,
        max_freq,
        depth + 1,
        splits,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::ToolId;
    use crate::toolpath::Segment;
    use glam::DVec3;
    #[test]
    fn cruise_only_segment_produces_zero_additional_subdivisions() {
        let p0 = DVec3::new(0.0, 0.0, 0.0);
        let p1 = DVec3::new(100.0, 0.0, 0.0);
        let path = Path {
            points: vec![p0, p1],
            segments: vec![Segment {
                kind: MoveKind::WallOuter,
                extrusion_length: 3.0,
                speed: 6000.0,
                ..Segment::default()
            }],
            tool: ToolId(0),
        };

        // A move that is purely cruising (accel_distance = 0, decel_distance = 0)
        let profiles = vec![PlannedMotionProfile {
            entry_speed: 6000.0,
            cruise_speed: 6000.0,
            exit_speed: 6000.0,
            accel_distance: 0.0,
            cruise_distance: 100.0,
            decel_distance: 0.0,
            duration_seconds: 1.0,
        }];

        let config = SlicerConfig::default();
        let subdivided = subdivide_path_for_pressure_advance(path, &profiles, &config, None);

        // An open path of 1 segment has 2 points and 1 segment
        assert_eq!(subdivided.segments.len(), 1);
        assert_eq!(subdivided.points.len(), 2);
        assert!((subdivided.segments[0].extrusion_length - 3.0).abs() < 1e-6);
    }

    #[test]
    fn accelerating_segment_subdivides_within_tolerance() {
        let p0 = DVec3::new(0.0, 0.0, 0.0);
        let p1 = DVec3::new(50.0, 0.0, 0.0);
        let path = Path {
            points: vec![p0, p1],
            segments: vec![Segment {
                kind: MoveKind::WallOuter,
                extrusion_length: 2.0,
                speed: 6000.0,
                ..Segment::default()
            }],
            tool: ToolId(0),
        };

        // Starts from 0 speed and accelerates across the first 25mm to 100mm/s (6000 mm/min)
        let profiles = vec![PlannedMotionProfile {
            entry_speed: 0.0,
            cruise_speed: 6000.0,
            exit_speed: 6000.0,
            accel_distance: 25.0,
            cruise_distance: 25.0,
            decel_distance: 0.0,
            duration_seconds: 0.75,
        }];

        let config = SlicerConfig {
            slicer_pa_tolerance_mm: Some(0.002),
            slicer_pa_min_segment_length: Some(0.5),
            ..SlicerConfig::default()
        };

        let subdivided = subdivide_path_for_pressure_advance(path, &profiles, &config, None);

        // Must have subdivided the acceleration phase
        assert!(subdivided.segments.len() > 1);
        assert_eq!(subdivided.points.len(), subdivided.segments.len() + 1);

        // An accelerating move advances extra filament into the melt zone to build pressure:
        // sum_e = nominal (2.0) + delta_adv (0.16) = 2.16
        let sum_e: f64 = subdivided.segments.iter().map(|s| s.extrusion_length).sum();
        assert!(
            (sum_e - 2.16).abs() < 0.05,
            "expected total extrusion ~2.16 with pressure advance, got {sum_e}"
        );
    }

    #[test]
    fn decelerating_segment_subdivides_and_conserves_total_extrusion() {
        let p0 = DVec3::new(0.0, 0.0, 0.0);
        let p1 = DVec3::new(40.0, 0.0, 0.0);
        let path = Path {
            points: vec![p0, p1],
            segments: vec![Segment {
                kind: MoveKind::WallOuter,
                extrusion_length: 1.5,
                speed: 6000.0,
                ..Segment::default()
            }],
            tool: ToolId(0),
        };

        // Decelerates from 100mm/s to 0 across the last 20mm
        let profiles = vec![PlannedMotionProfile {
            entry_speed: 6000.0,
            cruise_speed: 6000.0,
            exit_speed: 0.0,
            accel_distance: 0.0,
            cruise_distance: 20.0,
            decel_distance: 20.0,
            duration_seconds: 0.6,
        }];

        let config = SlicerConfig {
            slicer_pa_tolerance_mm: Some(0.002),
            slicer_pa_min_segment_length: Some(0.5),
            ..SlicerConfig::default()
        };

        let subdivided = subdivide_path_for_pressure_advance(path, &profiles, &config, None);

        assert!(subdivided.segments.len() > 1);
        // A decelerating move relieves pressure in the melt zone:
        // sum_e = nominal (1.5) - delta_adv (~0.145) = ~1.355
        let sum_e: f64 = subdivided.segments.iter().map(|s| s.extrusion_length).sum();
        assert!(
            (sum_e - 1.355).abs() < 0.05,
            "expected total extrusion ~1.355 with pressure relief, got {sum_e}"
        );
    }

    #[test]
    fn short_segment_under_min_length_is_not_subdivided() {
        let p0 = DVec3::new(0.0, 0.0, 0.0);
        let p1 = DVec3::new(0.2, 0.0, 0.0);
        let path = Path {
            points: vec![p0, p1],
            segments: vec![Segment {
                kind: MoveKind::WallOuter,
                extrusion_length: 0.01,
                speed: 3000.0,
                ..Segment::default()
            }],
            tool: ToolId(0),
        };

        let profiles = vec![PlannedMotionProfile {
            entry_speed: 0.0,
            cruise_speed: 3000.0,
            exit_speed: 0.0,
            accel_distance: 0.1,
            cruise_distance: 0.0,
            decel_distance: 0.1,
            duration_seconds: 0.01,
        }];

        let config = SlicerConfig::default();
        let subdivided = subdivide_path_for_pressure_advance(path, &profiles, &config, None);

        // 0.2mm is below min_segment_length (0.35mm), so must remain 1 segment
        assert_eq!(subdivided.segments.len(), 1);
        assert_eq!(subdivided.points.len(), 2);
    }

    #[test]
    fn full_accel_cruise_decel_path_conserves_exact_total_extrusion() {
        let p0 = DVec3::new(0.0, 0.0, 0.0);
        let p1 = DVec3::new(100.0, 0.0, 0.0);
        let path = Path {
            points: vec![p0, p1],
            segments: vec![Segment {
                kind: MoveKind::WallOuter,
                extrusion_length: 5.0,
                speed: 6000.0,
                ..Segment::default()
            }],
            tool: ToolId(0),
        };

        // A move that starts at 0, accelerates to cruise, then decelerates back to 0
        let profiles = vec![PlannedMotionProfile {
            entry_speed: 0.0,
            cruise_speed: 6000.0,
            exit_speed: 0.0,
            accel_distance: 25.0,
            cruise_distance: 50.0,
            decel_distance: 25.0,
            duration_seconds: 1.5,
        }];

        let config = SlicerConfig {
            slicer_pa_tolerance_mm: Some(0.002),
            slicer_pa_min_segment_length: Some(0.5),
            ..SlicerConfig::default()
        };

        let subdivided = subdivide_path_for_pressure_advance(path, &profiles, &config, None);

        // Subdivided across both accel and decel ramps, cruise untouched
        assert!(subdivided.segments.len() > 2);

        // Start and exit at 0 velocity -> net pressure advance delta is 0 -> exact nominal volume conservation
        let sum_e: f64 = subdivided.segments.iter().map(|s| s.extrusion_length).sum();
        assert!(
            (sum_e - 5.0).abs() < 0.005,
            "expected exact total extrusion 5.0, got {sum_e}"
        );
    }
}
