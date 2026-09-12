//! Slicing and toolpath print statistics (print time, filament volume, and mass).

use crate::{
    kinematics::plan_chained_path_velocities,
    toolpath::{MoveKind, Path},
    SlicerConfig,
};
use serde::{Deserialize, Serialize};

/// Summary statistics for a sliced program.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PrintStatistics {
    /// Total estimated print execution time in seconds.
    pub estimated_time_seconds: f64,
    /// Total linear filament consumed in meters.
    pub filament_length_meters: f64,
    /// Total filament volume consumed in cubic centimeters (cm³).
    pub filament_volume_cm3: f64,
    /// Total filament mass consumed in grams (g).
    pub filament_weight_grams: f64,
    /// Total number of layers in the print.
    pub total_layers: usize,
    /// Total number of toolpath paths.
    pub total_paths: usize,
    /// Total number of extruding moves.
    pub total_extruding_moves: usize,
    /// Total number of travel moves.
    pub total_travel_moves: usize,
}

impl PrintStatistics {
    /// Formats the estimated print time into a human-readable string (e.g., "1h 23m 45s" or "12m 34s").
    #[must_use]
    pub fn formatted_time(&self) -> String {
        let total_secs = self.estimated_time_seconds.round() as u64;
        let hours = total_secs / 3600;
        let minutes = (total_secs % 3600) / 60;
        let seconds = total_secs % 60;

        if hours > 0 {
            format!("{hours}h {minutes}m {seconds}s")
        } else if minutes > 0 {
            format!("{minutes}m {seconds}s")
        } else {
            format!("{seconds}s")
        }
    }
}

/// Computes print statistics from planned toolpaths, kinematics, and slicer configuration.
#[must_use]
pub fn compute_print_statistics(
    paths: &[Path],
    config: &SlicerConfig,
    material_density_g_cm3: Option<f64>,
) -> PrintStatistics {
    compute_print_statistics_with_machine(paths, config, None, material_density_g_cm3)
}

/// Computes print statistics from planned toolpaths, kinematics, and slicer configuration,
/// taking into account the active machine motion model (e.g. stepper motor dynamics).
#[must_use]
pub fn compute_print_statistics_with_machine(
    paths: &[Path],
    config: &SlicerConfig,
    machine: Option<&crate::machine::Machine>,
    material_density_g_cm3: Option<f64>,
) -> PrintStatistics {
    if paths.is_empty() {
        return PrintStatistics {
            estimated_time_seconds: 0.0,
            filament_length_meters: 0.0,
            filament_volume_cm3: 0.0,
            filament_weight_grams: 0.0,
            total_layers: 0,
            total_paths: 0,
            total_extruding_moves: 0,
            total_travel_moves: 0,
        };
    }

    let model = config.resolved_motion_model(machine);
    let min_order = paths
        .iter()
        .filter_map(|p| p.segments.first())
        .map(|s| s.order)
        .fold(f64::INFINITY, f64::min);

    let mut total_time = 0.0;
    let mut total_extrusion_mm = 0.0;
    let mut total_extruding_moves = 0;
    let mut total_travel_moves = 0;
    let mut seen_orders: Vec<f64> = Vec::new();

    let density = material_density_g_cm3.unwrap_or_else(|| config.filament_density());

    // Mirrors `gcode::emit`'s per-tool `retracted` flag and
    // `transient_pressure::apply_transient_flow_compensation`'s equivalent
    // tracking: retraction/unretraction are E-only moves with no XY
    // displacement, so they're otherwise invisible to any duration computed
    // from travel distance alone. Charging their configured duration here
    // keeps `estimated_time_seconds` consistent with the actual Gcode
    // `gcode::emit` writes. Uses each travel move's own distance against
    // `effective_min_travel_for_retract()` rather than summing the full
    // contiguous travel run gcode.rs considers -- an approximation that
    // only differs for multi-segment travel runs, which are rare.
    let mut current_tool: Option<crate::ids::ToolId> = None;
    let mut retracted = true;
    let retract_time = config.retraction_duration_seconds();
    let unretract_time = config.unretraction_duration_seconds();
    let min_retract_travel = config.effective_min_travel_for_retract();

    // Plans velocities across contiguous same-tool open-path runs as one
    // continuous polyline (see `plan_chained_path_velocities`), rather than
    // per-`Path` in isolation, so `estimated_time_seconds` doesn't inflate
    // print time with a phantom stop-to-zero at every one of the (often
    // thousands of) `Path` object boundaries a non-planar slice produces.
    let first_layer_flags: Vec<bool> = paths
        .iter()
        .map(|path| {
            let path_order = path.segments.first().map(|s| s.order).unwrap_or(0.0);
            (path_order - min_order).abs() < 1e-4
        })
        .collect();
    let all_profiles = plan_chained_path_velocities(
        paths,
        &*model,
        &first_layer_flags,
        config.square_corner_velocity(),
        config.minimum_cruise_ratio(),
    );

    for (path, profiles) in paths.iter().zip(all_profiles.iter()) {
        if current_tool != Some(path.tool) {
            current_tool = Some(path.tool);
            retracted = true;
        }

        let path_order = path.segments.first().map(|s| s.order).unwrap_or(0.0);
        if !seen_orders.iter().any(|&o| (o - path_order).abs() < 1e-4) {
            seen_orders.push(path_order);
        }

        for (i, seg) in path.segments.iter().enumerate() {
            let is_travel = seg.kind == MoveKind::Travel;
            if is_travel {
                total_travel_moves += 1;
            } else {
                total_extruding_moves += 1;
                total_extrusion_mm += seg.extrusion_length;
            }

            if is_travel && !retracted {
                let p0 = path.points.get(i).copied().unwrap_or(glam::DVec3::ZERO);
                let p1 = path.points.get(i + 1).copied().unwrap_or(p0);
                if p0.distance(p1) > min_retract_travel {
                    total_time += retract_time;
                    retracted = true;
                }
            } else if !is_travel && retracted {
                total_time += unretract_time;
                retracted = false;
            }

            if let Some(profile) = profiles.get(i) {
                total_time += profile.duration_seconds;
            } else {
                // Fallback constant-velocity time estimate
                let p0 = path.points.get(i).copied().unwrap_or(glam::DVec3::ZERO);
                let p1 = path
                    .points
                    .get((i + 1) % path.points.len())
                    .copied()
                    .unwrap_or(glam::DVec3::ZERO);
                let dist = (p1 - p0).length();
                let v = (seg.speed / 60.0).max(1.0);
                total_time += dist / v;
            }
        }
    }

    let filament_length_meters = total_extrusion_mm / 1000.0;
    let filament_radius_mm = config.filament_diameter * 0.5;
    let filament_area_mm2 = std::f64::consts::PI * filament_radius_mm * filament_radius_mm;
    let filament_volume_mm3 = total_extrusion_mm * filament_area_mm2;
    let filament_volume_cm3 = filament_volume_mm3 / 1000.0;
    let filament_weight_grams = filament_volume_cm3 * density;

    PrintStatistics {
        estimated_time_seconds: total_time,
        filament_length_meters,
        filament_volume_cm3,
        filament_weight_grams,
        total_layers: seen_orders.len(),
        total_paths: paths.len(),
        total_extruding_moves,
        total_travel_moves,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::toolpath::Segment;
    use glam::DVec3;

    #[test]
    fn compute_print_statistics_calculates_time_and_filament_weight() {
        let paths = vec![Path {
            points: vec![DVec3::new(0.0, 0.0, 0.0), DVec3::new(100.0, 0.0, 0.0)],
            segments: vec![Segment {
                kind: MoveKind::WallOuter,
                speed: 6000.0,          // 100 mm/s
                extrusion_length: 50.0, // 50 mm of filament
                ..Segment::default()
            }],
            tool: crate::ids::ToolId(0),
            object: crate::ids::ObjectId::default(),
        }];

        let config = SlicerConfig {
            filament_diameter: 1.75,
            ..SlicerConfig::default()
        };

        let stats = compute_print_statistics(&paths, &config, Some(1.24));

        assert!(stats.estimated_time_seconds > 0.0);
        assert!((stats.filament_length_meters - 0.05).abs() < 1e-4);
        assert!(stats.filament_volume_cm3 > 0.0);
        assert!(stats.filament_weight_grams > 0.0);
        assert_eq!(stats.total_extruding_moves, 1);
        assert_eq!(stats.total_travel_moves, 0);
    }

    #[test]
    fn print_statistics_formats_hours_minutes_seconds() {
        let stats = PrintStatistics {
            estimated_time_seconds: 3723.0, // 1h 2m 3s
            filament_length_meters: 10.0,
            filament_volume_cm3: 5.0,
            filament_weight_grams: 6.2,
            total_layers: 100,
            total_paths: 200,
            total_extruding_moves: 1000,
            total_travel_moves: 200,
        };

        assert_eq!(stats.formatted_time(), "1h 2m 3s");
    }

    #[test]
    fn stepper_dynamics_affects_estimated_print_time() {
        use crate::bounds::BoundingVolume;
        use crate::machine::Machine;

        let paths = vec![Path {
            points: vec![DVec3::new(0.0, 0.0, 0.0), DVec3::new(200.0, 0.0, 0.0)],
            segments: vec![Segment {
                kind: MoveKind::WallOuter,
                speed: 60000.0, // 1000 mm/s
                extrusion_length: 10.0,
                ..Segment::default()
            }],
            tool: crate::ids::ToolId(0),
            object: crate::ids::ObjectId::default(),
        }];
        let config = SlicerConfig::default();

        let mut machine = Machine::new(
            BoundingVolume::Aabb {
                min: DVec3::ZERO,
                max: DVec3::new(300.0, 300.0, 300.0),
            },
            Vec::new(),
        );

        let stats_std =
            compute_print_statistics_with_machine(&paths, &config, Some(&machine), None);

        // Turn on stepper dynamics with severe speed and acceleration limits
        machine.use_stepper_dynamics = true;
        machine.zero_speed_acceleration = Some(1000.0);
        machine.max_available_speed = Some(200.0);
        machine.acceleration_limit = Some(500.0);
        machine.speed_limit = Some(100.0);

        let stats_dynamic =
            compute_print_statistics_with_machine(&paths, &config, Some(&machine), None);

        assert!(
            stats_dynamic.estimated_time_seconds > stats_std.estimated_time_seconds,
            "stepper dynamic limits should increase print time: {} vs {}",
            stats_dynamic.estimated_time_seconds,
            stats_std.estimated_time_seconds
        );
    }
}
