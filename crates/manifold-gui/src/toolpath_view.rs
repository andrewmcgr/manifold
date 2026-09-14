//! Toolpath preview geometry: converts `manifold_core::toolpath::Path`
//! data into line-list vertices colored by `MoveKind` (Phase 13, see
//! ROADMAP.md). Pure geometry builders — no GPU/wgpu types here, kept
//! separate from `render.rs`'s GPU upload/pipeline concerns, mirroring
//! `scene.rs`'s existing separation.
//!

use manifold_core::toolpath::{MoveKind, Path};

/// One line segment instance for the unlit toolpath line shader: start/end
/// positions + RGBA color + the source segment's `order` value (carried
/// per-instance so the scrub filter can operate either CPU-side, before
/// upload, or shader-side, via a uniform threshold against this field).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Default)]
pub struct ToolpathLineInstance {
    pub start: [f32; 3],
    pub end: [f32; 3],
    pub color: [f32; 4],
    pub order: f32,
    pub width: f32,
    pub height: f32,
    pub _pad: [f32; 3],
}

impl ToolpathLineInstance {
    pub fn new(
        start: glam::DVec3,
        end: glam::DVec3,
        color: [f32; 4],
        order: f64,
        width: f64,
        height: f64,
    ) -> Self {
        Self {
            start: start.as_vec3().to_array(),
            end: end.as_vec3().to_array(),
            color,
            order: order as f32,
            width: width as f32,
            height: height as f32,
            _pad: [0.0; 3],
        }
    }
}

/// Computes physical cross-sectional dimensions `(width, height)` in millimeters for `segment`.
pub fn segment_bead_dimensions(
    segment: &manifold_core::toolpath::Segment,
    start: glam::DVec3,
    end: glam::DVec3,
    config: &manifold_core::SlicerConfig,
) -> (f64, f64) {
    if segment.kind == MoveKind::Travel || segment.extrusion_length <= 0.0 {
        return (0.0, 0.0);
    }

    let dist = start.distance(end);
    let nominal_width = if segment.line_width > 1e-4 {
        segment.line_width
    } else {
        manifold_core::extrusion::line_width_for_kind(segment.kind, config)
    };
    let nominal_height = config.layer_height;

    if dist < 1e-6 {
        return (nominal_width, nominal_height);
    }

    // Actual volume deposited per unit path length:
    let filament_area =
        manifold_core::extrusion::filament_cross_section_area(config.filament_diameter);
    let actual_bead_area = (segment.extrusion_length / dist) * filament_area;
    let nominal_bead_area =
        manifold_core::extrusion::bead_cross_section_area(nominal_width, nominal_height);

    if nominal_bead_area > 1e-6 && actual_bead_area > 1e-6 {
        let volumetric_ratio = actual_bead_area / nominal_bead_area;
        let scale = volumetric_ratio.sqrt();
        let width = (nominal_width * scale).clamp(0.05, 3.0);
        let height = (nominal_height * scale).clamp(0.02, 2.0);
        (width, height)
    } else {
        (nominal_width, nominal_height)
    }
}

pub const COLOR_WALL_OUTER: [f32; 4] = [0.9, 0.9, 0.9, 1.0];
pub const COLOR_WALL_INNER: [f32; 4] = [0.6, 0.8, 1.0, 1.0];
pub const COLOR_INFILL: [f32; 4] = [0.95, 0.65, 0.15, 1.0];
pub const COLOR_BRIDGE: [f32; 4] = [0.9, 0.2, 0.75, 1.0];
pub const COLOR_OVERHANG: [f32; 4] = [0.9, 0.15, 0.15, 1.0];
pub const COLOR_TOP_SURFACE: [f32; 4] = [0.2, 0.85, 0.4, 1.0];
pub const COLOR_SCARF_JOINT: [f32; 4] = [1.0, 0.80, 0.25, 1.0];
pub const COLOR_DEBUG_EXCLUDED: [f32; 4] = [1.0, 0.1, 0.9, 1.0];
pub const COLOR_TRAVEL: [f32; 4] = [0.35, 0.55, 0.75, 0.65];

/// Identifiers for toggling individual line types on and off in the viewport.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum LineTypeKey {
    WallOuter,
    WallInner,
    Infill,
    Bridge,
    Overhang,
    TopSurface,
    ScarfJoint,
    Travel,
    DebugExcluded,
}

/// Available data view color-coding modes for the toolpath preview.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum ToolpathDataView {
    #[default]
    LineType,
    Speed,
    ActualSpeed,
    FlowRate,
    Acceleration,
    ActualAcceleration,
    TravelDurations,
    /// `FlowBreakdown::first_layer_mult` -- see `SlicerConfig::first_layer_extrusion_multiplier`.
    FirstLayerMultiplier,
    /// `FlowBreakdown::directional_flow_mult` -- see `SlicerConfig::slope_compensation_mode`.
    DirectionalFlowMultiplier,
    /// `FlowBreakdown::swell_mult` -- see `fluid_dynamics::FluidDynamicsEngine`.
    SwellMultiplier,
    /// `FlowBreakdown::corner_flow_mult` -- see `corner_flow::apply_corner_flow_compensation`.
    CornerFlowMultiplier,
    /// `FlowBreakdown::transient_pressure_mult` -- see `transient_pressure::apply_transient_flow_compensation`.
    TransientPressureMultiplier,
    /// `FlowBreakdown::slope_cosine` -- combined surface-inclination/trajectory-climb factor.
    SlopeCosine,
    /// Normalized position of each segment's flow rate between the two-point fluid-dynamics
    /// PA calibration plateaus (`FluidDynamicsConfig::pa_calibration_low`/`_high`): 0.0 = pinned
    /// at or below the low-flow plateau, 1.0 = pinned at or above the high-flow plateau,
    /// log-space interpolated in between.
    PressureAdvanceBlend,
    /// Real dynamic pressure-advance value (seconds) from the configured
    /// `fluid_dynamics::FluidDynamicsEngine::dynamic_pressure_advance`,
    /// evaluated at each segment's own flow rate with a fixed 0.0 (no-
    /// cooling) fan fraction -- per-segment fan speed isn't tracked, so
    /// this zeroes only the fan's contribution rather than guessing a
    /// cooling state. This is NOT full temperature neutrality: the
    /// nozzle temperature is still `SlicerConfig::default_nozzle_
    /// temperature()` (no per-tool live temperature is available here
    /// -- `segment_scalar_value_with_profile` only receives the bare
    /// `Segment`, not its parent `Path`/tool), which can genuinely
    /// differ from the calibration's own `reference_temp_c` (e.g. a
    /// material calibrated at one temperature but sliced at another),
    /// producing a real, non-cosmetic temperature-correction term in
    /// that case. Falls back to the static
    /// `SlicerConfig::pressure_advance` when no fluid-dynamics model is
    /// configured, matching `transient_pressure`'s own fallback.
    PressureAdvanceValue,
}

impl ToolpathDataView {
    pub fn name(&self) -> &'static str {
        match self {
            Self::LineType => "Line Type",
            Self::Speed => "Speed",
            Self::ActualSpeed => "Actual Speed",
            Self::FlowRate => "Flow Rate",
            Self::Acceleration => "Acceleration",
            Self::ActualAcceleration => "Actual Acceleration",
            Self::TravelDurations => "Travel Durations",
            Self::FirstLayerMultiplier => "First Layer Multiplier",
            Self::DirectionalFlowMultiplier => "Directional Flow Multiplier",
            Self::SwellMultiplier => "Extrudate Swell Multiplier",
            Self::CornerFlowMultiplier => "Corner Flow Multiplier",
            Self::TransientPressureMultiplier => "Transient Pressure Multiplier",
            Self::SlopeCosine => "Slope Cosine",
            Self::PressureAdvanceBlend => "PA Blend Zone",
            Self::PressureAdvanceValue => "Pressure Advance",
        }
    }

    pub fn unit(&self) -> &'static str {
        match self {
            Self::LineType | Self::PressureAdvanceBlend => "",
            Self::Speed => "mm/s",
            Self::ActualSpeed => "mm/s",
            Self::FlowRate => "mm³/s",
            Self::Acceleration => "mm/s²",
            Self::ActualAcceleration => "mm/s²",
            Self::TravelDurations => "s",
            Self::FirstLayerMultiplier
            | Self::DirectionalFlowMultiplier
            | Self::SwellMultiplier
            | Self::CornerFlowMultiplier
            | Self::TransientPressureMultiplier
            | Self::SlopeCosine => "x",
            Self::PressureAdvanceValue => "s",
        }
    }
}

/// Smooth 5-stop color gradient mapping (Blue -> Cyan -> Green -> Yellow -> Red) for normalized `t` in [0, 1].
pub fn scalar_to_color(t: f64) -> [f32; 4] {
    let t = t.clamp(0.0, 1.0) as f32;
    let (c0, c1, local_t) = if t < 0.25 {
        ([0.15f32, 0.20, 0.85], [0.00f32, 0.75, 0.90], t / 0.25)
    } else if t < 0.50 {
        (
            [0.00f32, 0.75, 0.90],
            [0.10f32, 0.85, 0.25],
            (t - 0.25) / 0.25,
        )
    } else if t < 0.75 {
        (
            [0.10f32, 0.85, 0.25],
            [0.98f32, 0.85, 0.10],
            (t - 0.50) / 0.25,
        )
    } else {
        (
            [0.98f32, 0.85, 0.10],
            [0.95f32, 0.15, 0.15],
            (t - 0.75) / 0.25,
        )
    };

    [
        c0[0] + (c1[0] - c0[0]) * local_t,
        c0[1] + (c1[1] - c0[1]) * local_t,
        c0[2] + (c1[2] - c0[2]) * local_t,
        1.0,
    ]
}

/// Volumetric flow rate (mm³/s) for a single segment, `0.0` for travel/wipe or zero-length moves.
fn segment_flow_rate_mm3_s(
    segment: &manifold_core::toolpath::Segment,
    diff: glam::DVec3,
    config: &manifold_core::SlicerConfig,
) -> f64 {
    if segment.kind == MoveKind::Travel || segment.extrusion_length <= 0.0 {
        return 0.0;
    }
    let length = diff.length();
    if length < 1e-6 {
        return 0.0;
    }
    let speed_mm_s = segment.speed / 60.0;
    if speed_mm_s <= 1e-6 {
        return 0.0;
    }
    let duration = length / speed_mm_s;
    let fil_radius = config.filament_diameter * 0.5;
    let fil_area = std::f64::consts::PI * fil_radius * fil_radius;
    let vol = segment.extrusion_length * fil_area;
    vol / duration
}

/// Computes the scalar value for `segment` under `data_view`, optionally using a precomputed motion profile.
pub fn segment_scalar_value_with_profile(
    segment: &manifold_core::toolpath::Segment,
    start: glam::DVec3,
    end: glam::DVec3,
    data_view: ToolpathDataView,
    config: &manifold_core::SlicerConfig,
    machine: Option<&manifold_core::machine::Machine>,
    profile: Option<&manifold_core::kinematics::PlannedMotionProfile>,
) -> f64 {
    let diff = end - start;
    let dir = if diff.length() > 1e-6 {
        diff.normalize()
    } else {
        glam::DVec3::ZERO
    };

    match data_view {
        ToolpathDataView::LineType => 0.0,
        ToolpathDataView::Speed => {
            let model = config.resolved_motion_model(machine);
            let is_first_layer = (segment.order - config.first_layer_height()).abs() < 1e-4
                || segment.order <= config.first_layer_height();
            let max_feed = model.max_directional_feedrate(segment.kind, is_first_layer, dir);
            segment.speed.min(max_feed) / 60.0
        }
        ToolpathDataView::ActualSpeed => {
            if let Some(prof) = profile {
                prof.cruise_speed / 60.0
            } else {
                let model = config.resolved_motion_model(machine);
                let is_first_layer = (segment.order - config.first_layer_height()).abs() < 1e-4
                    || segment.order <= config.first_layer_height();
                let max_feed = model.max_directional_feedrate(segment.kind, is_first_layer, dir);
                segment.speed.min(max_feed) / 60.0
            }
        }
        ToolpathDataView::FlowRate => segment_flow_rate_mm3_s(segment, diff, config),
        ToolpathDataView::Acceleration => {
            let model = config.resolved_motion_model(machine);
            let is_first_layer = (segment.order - config.first_layer_height()).abs() < 1e-4
                || segment.order <= config.first_layer_height();
            model.available_directional_acceleration(
                segment.kind,
                is_first_layer,
                segment.speed / 60.0,
                dir,
            )
        }
        ToolpathDataView::ActualAcceleration => {
            let model = config.resolved_motion_model(machine);
            let is_first_layer = (segment.order - config.first_layer_height()).abs() < 1e-4
                || segment.order <= config.first_layer_height();
            let v = if let Some(prof) = profile {
                prof.cruise_speed / 60.0
            } else {
                segment.speed / 60.0
            };
            model.available_directional_acceleration(segment.kind, is_first_layer, v, dir)
        }
        ToolpathDataView::TravelDurations => {
            if segment.kind != MoveKind::Travel {
                return 0.0;
            }
            let length = diff.length();
            let speed_mm_s = (segment.speed / 60.0).max(1e-3);
            length / speed_mm_s
        }
        ToolpathDataView::FirstLayerMultiplier => {
            segment.flow_breakdown.map_or(1.0, |fb| fb.first_layer_mult)
        }
        ToolpathDataView::DirectionalFlowMultiplier => segment
            .flow_breakdown
            .map_or(1.0, |fb| fb.directional_flow_mult),
        ToolpathDataView::SwellMultiplier => segment.flow_breakdown.map_or(1.0, |fb| fb.swell_mult),
        ToolpathDataView::CornerFlowMultiplier => {
            segment.flow_breakdown.map_or(1.0, |fb| fb.corner_flow_mult)
        }
        ToolpathDataView::TransientPressureMultiplier => segment
            .flow_breakdown
            .map_or(1.0, |fb| fb.transient_pressure_mult),
        ToolpathDataView::SlopeCosine => segment.flow_breakdown.map_or(1.0, |fb| fb.slope_cosine),
        ToolpathDataView::PressureAdvanceBlend => {
            let q = segment_flow_rate_mm3_s(segment, diff, config);
            if q <= 1e-6 {
                return 0.0;
            }
            let fd = config.fluid_dynamics.unwrap_or_default();
            let q_low = fd.pa_calibration_low.1.max(0.1);
            let q_high = fd.pa_calibration_high.1.max(q_low + 0.1);
            ((q.ln() - q_low.ln()) / (q_high.ln() - q_low.ln())).clamp(0.0, 1.0)
        }
        ToolpathDataView::PressureAdvanceValue => {
            let q = segment_flow_rate_mm3_s(segment, diff, config);
            if q <= 1e-6 {
                return 0.0;
            }
            match config.fluid_dynamics_engine(None) {
                Some(engine) => engine.dynamic_pressure_advance(q, 0.0),
                None => config.pressure_advance.unwrap_or(0.0),
            }
        }
    }
}

/// Computes the scalar value for `segment` under `data_view`.
pub fn segment_scalar_value(
    segment: &manifold_core::toolpath::Segment,
    start: glam::DVec3,
    end: glam::DVec3,
    data_view: ToolpathDataView,
    config: &manifold_core::SlicerConfig,
    machine: Option<&manifold_core::machine::Machine>,
) -> f64 {
    segment_scalar_value_with_profile(segment, start, end, data_view, config, machine, None)
}

fn plan_path_motion_profiles(
    path: &Path,
    config: &manifold_core::SlicerConfig,
    machine: Option<&manifold_core::machine::Machine>,
) -> Vec<manifold_core::kinematics::PlannedMotionProfile> {
    let motion_model = config.resolved_motion_model(machine);
    let is_first_layer = path.segments.first().is_some_and(|s| {
        (s.order - config.first_layer_height()).abs() < 1e-4
            || s.order <= config.first_layer_height()
    });
    let scv = config.square_corner_velocity.unwrap_or(5.0);
    let cruise_ratio = machine
        .and_then(|m| m.minimum_cruise_ratio)
        .unwrap_or_else(|| config.minimum_cruise_ratio());
    manifold_core::kinematics::plan_path_velocities(
        &path.points,
        &path.segments,
        &*motion_model,
        is_first_layer,
        scv,
        cruise_ratio,
    )
}

/// The `(min, max)` scalar value range across all segments in `paths` for `data_view`.
/// Returns `None` for [`ToolpathDataView::LineType`] or when no valid segments exist.
pub fn data_view_range(
    paths: &[Path],
    data_view: ToolpathDataView,
    config: &manifold_core::SlicerConfig,
    machine: Option<&manifold_core::machine::Machine>,
) -> Option<(f64, f64)> {
    if data_view == ToolpathDataView::LineType {
        return None;
    }
    if data_view == ToolpathDataView::PressureAdvanceBlend {
        return Some((0.0, 1.0));
    }
    let mut min_val = f64::INFINITY;
    let mut max_val = f64::NEG_INFINITY;
    let mut count = 0;

    let mut prev_endpoint: Option<glam::DVec3> = None;

    for path in paths {
        let n = path.points.len();
        if n == 0 {
            continue;
        }
        let path_start = path.points[0];
        let path_order = path.segments.first().map(|s| s.order).unwrap_or(0.0);

        if let Some(prev_end) = prev_endpoint {
            if prev_end.distance(path_start) > 1e-6
                && data_view == ToolpathDataView::TravelDurations
            {
                let travel_segment = manifold_core::toolpath::Segment {
                    kind: MoveKind::Travel,
                    speed: config.travel_speed,
                    extrusion_rate: 0.0,
                    support_fraction: 0.0,
                    order: path_order,
                    extrusion_length: 0.0,
                    line_width: 0.0,
                    is_scarf: false,
                    id: 0,
                    island: 0,
                    channel_width: f64::INFINITY,
                    flow_breakdown: None,
                };
                let val = segment_scalar_value(
                    &travel_segment,
                    prev_end,
                    path_start,
                    data_view,
                    config,
                    machine,
                );
                if val > 1e-6 {
                    min_val = min_val.min(val);
                    max_val = max_val.max(val);
                    count += 1;
                }
            }
        }

        let profiles = if matches!(
            data_view,
            ToolpathDataView::ActualSpeed | ToolpathDataView::ActualAcceleration
        ) {
            plan_path_motion_profiles(path, config, machine)
        } else {
            Vec::new()
        };

        for (i, segment) in path.segments.iter().enumerate() {
            if data_view == ToolpathDataView::TravelDurations {
                if segment.kind != MoveKind::Travel {
                    continue;
                }
                let start = path.points[i];
                let end = path.points[(i + 1) % n];
                let val = segment_scalar_value(segment, start, end, data_view, config, machine);
                if val > 1e-6 {
                    min_val = min_val.min(val);
                    max_val = max_val.max(val);
                    count += 1;
                }
                continue;
            }

            // Travel moves are rendered in a dedicated travel color (COLOR_TRAVEL)
            // and should not skew the scalar gradient range of extrusion features.
            if segment.kind == MoveKind::Travel {
                continue;
            }
            let start = path.points[i];
            let end = path.points[(i + 1) % n];
            let val = segment_scalar_value_with_profile(
                segment,
                start,
                end,
                data_view,
                config,
                machine,
                profiles.get(i),
            );
            // Ignore near-zero non-extruding artifacts when computing flow rate ranges
            if data_view == ToolpathDataView::FlowRate && val <= 1e-3 {
                continue;
            }
            min_val = min_val.min(val);
            max_val = max_val.max(val);
            count += 1;
        }

        if path.segments.len() == n {
            prev_endpoint = Some(path.points[0]);
        } else {
            prev_endpoint = path.points.last().copied();
        }
    }

    if count > 0 && min_val.is_finite() && max_val.is_finite() {
        Some((min_val, max_val))
    } else {
        None
    }
}

/// An entry in a toolpath viewport legend (color badge + descriptive label).
pub struct LegendEntry {
    pub key: LineTypeKey,
    pub label: &'static str,
    pub color: [f32; 4],
}

/// The set of color-coded legend entries for the [`ToolpathDataView::LineType`] view.
pub fn line_type_legend() -> &'static [LegendEntry] {
    &[
        LegendEntry {
            key: LineTypeKey::WallOuter,
            label: "Outer Wall",
            color: COLOR_WALL_OUTER,
        },
        LegendEntry {
            key: LineTypeKey::WallInner,
            label: "Inner Wall",
            color: COLOR_WALL_INNER,
        },
        LegendEntry {
            key: LineTypeKey::Infill,
            label: "Infill",
            color: COLOR_INFILL,
        },
        LegendEntry {
            key: LineTypeKey::Bridge,
            label: "Bridge",
            color: COLOR_BRIDGE,
        },
        LegendEntry {
            key: LineTypeKey::Overhang,
            label: "Overhang",
            color: COLOR_OVERHANG,
        },
        LegendEntry {
            key: LineTypeKey::TopSurface,
            label: "Top Surface",
            color: COLOR_TOP_SURFACE,
        },
        LegendEntry {
            key: LineTypeKey::ScarfJoint,
            label: "Scarf Joint",
            color: COLOR_SCARF_JOINT,
        },
        LegendEntry {
            key: LineTypeKey::Travel,
            label: "Travel",
            color: COLOR_TRAVEL,
        },
        LegendEntry {
            key: LineTypeKey::DebugExcluded,
            label: "Debug / Excluded",
            color: COLOR_DEBUG_EXCLUDED,
        },
    ]
}

/// Fixed `MoveKind` -> RGBA color palette used for toolpath preview
/// rendering, with support for distinct scarf joint seam highlighting.
pub fn palette_color(kind: MoveKind, is_scarf: bool) -> [f32; 4] {
    if is_scarf {
        COLOR_SCARF_JOINT
    } else {
        match kind {
            MoveKind::WallOuter => COLOR_WALL_OUTER,
            MoveKind::WallInner => COLOR_WALL_INNER,
            MoveKind::Infill => COLOR_INFILL,
            MoveKind::Bridge => COLOR_BRIDGE,
            MoveKind::Overhang => COLOR_OVERHANG,
            MoveKind::TopSurface => COLOR_TOP_SURFACE,
            MoveKind::Travel | MoveKind::Wipe => COLOR_TRAVEL,
            MoveKind::DebugExcluded => COLOR_DEBUG_EXCLUDED,
        }
    }
}

/// Build a line-instance buffer from a set of planned toolpaths: one
/// line instance per `Segment` (`points[i] -> points[(i + 1) % points.len()]`),
/// colored by `segment.kind` via [`palette_color`], with `segment.order` carried
/// on each instance.
#[allow(dead_code)]
pub fn build_toolpath_lines(
    paths: &[Path],
    scrub_order: f64,
    data_view: ToolpathDataView,
    config: &manifold_core::SlicerConfig,
    machine: Option<&manifold_core::machine::Machine>,
) -> Vec<ToolpathLineInstance> {
    build_toolpath_lines_filtered(
        paths,
        scrub_order,
        data_view,
        config,
        machine,
        &std::collections::HashSet::new(),
    )
}

/// Build a line-instance buffer from a set of planned toolpaths with line type filtering.
pub fn build_toolpath_lines_filtered(
    paths: &[Path],
    scrub_order: f64,
    data_view: ToolpathDataView,
    config: &manifold_core::SlicerConfig,
    machine: Option<&manifold_core::machine::Machine>,
    hidden_line_types: &std::collections::HashSet<LineTypeKey>,
) -> Vec<ToolpathLineInstance> {
    let scalar_range = data_view_range(paths, data_view, config, machine);
    let mut instances = Vec::new();
    let mut prev_endpoint: Option<glam::DVec3> = None;

    for path in paths {
        let count = path.points.len();
        if count == 0 {
            continue;
        }

        let path_start = path.points[0];
        let path_order = path.segments.first().map(|s| s.order).unwrap_or(0.0);

        // Inter-path travel move from previous path end to this path start
        if let Some(prev_end) = prev_endpoint {
            if prev_end.distance(path_start) > 1e-6
                && path_order <= scrub_order
                && !hidden_line_types.contains(&LineTypeKey::Travel)
            {
                let travel_segment = manifold_core::toolpath::Segment {
                    kind: MoveKind::Travel,
                    speed: config.travel_speed,
                    extrusion_rate: 0.0,
                    support_fraction: 0.0,
                    order: path_order,
                    extrusion_length: 0.0,
                    line_width: 0.0,
                    is_scarf: false,
                    id: 0,
                    island: 0,
                    channel_width: f64::INFINITY,
                    flow_breakdown: None,
                };
                let color = match data_view {
                    ToolpathDataView::LineType
                    | ToolpathDataView::FlowRate
                    | ToolpathDataView::Speed
                    | ToolpathDataView::ActualSpeed
                    | ToolpathDataView::Acceleration
                    | ToolpathDataView::ActualAcceleration
                    | ToolpathDataView::FirstLayerMultiplier
                    | ToolpathDataView::DirectionalFlowMultiplier
                    | ToolpathDataView::SwellMultiplier
                    | ToolpathDataView::CornerFlowMultiplier
                    | ToolpathDataView::TransientPressureMultiplier
                    | ToolpathDataView::PressureAdvanceBlend
                    | ToolpathDataView::PressureAdvanceValue
                    | ToolpathDataView::SlopeCosine => COLOR_TRAVEL,
                    _ => {
                        let val = segment_scalar_value(
                            &travel_segment,
                            prev_end,
                            path_start,
                            data_view,
                            config,
                            machine,
                        );
                        let t = if let Some((min, max)) = scalar_range {
                            if max > min + 1e-4 {
                                ((val - min) / (max - min)).clamp(0.0, 1.0)
                            } else {
                                0.5
                            }
                        } else {
                            0.5
                        };
                        scalar_to_color(t)
                    }
                };
                instances.push(ToolpathLineInstance::new(
                    prev_end, path_start, color, path_order, 0.0, 0.0,
                ));
            }
        }

        let profiles = if matches!(
            data_view,
            ToolpathDataView::ActualSpeed | ToolpathDataView::ActualAcceleration
        ) {
            plan_path_motion_profiles(path, config, machine)
        } else {
            Vec::new()
        };

        for (index, segment) in path.segments.iter().enumerate() {
            if segment.order > scrub_order {
                continue;
            }

            let key = if segment.is_scarf {
                LineTypeKey::ScarfJoint
            } else {
                match segment.kind {
                    MoveKind::WallOuter => LineTypeKey::WallOuter,
                    MoveKind::WallInner => LineTypeKey::WallInner,
                    MoveKind::Infill => LineTypeKey::Infill,
                    MoveKind::Bridge => LineTypeKey::Bridge,
                    MoveKind::Overhang => LineTypeKey::Overhang,
                    MoveKind::TopSurface => LineTypeKey::TopSurface,
                    MoveKind::Travel | MoveKind::Wipe => LineTypeKey::Travel,
                    MoveKind::DebugExcluded => LineTypeKey::DebugExcluded,
                }
            };
            if hidden_line_types.contains(&key) {
                continue;
            }

            let start = path.points[index];
            let end = path.points[(index + 1) % count];

            // In TravelDurations mode, show only travel moves
            if data_view == ToolpathDataView::TravelDurations {
                if segment.kind != MoveKind::Travel {
                    continue;
                }
                let val = segment_scalar_value(segment, start, end, data_view, config, machine);
                let t = if let Some((min, max)) = scalar_range {
                    if max > min + 1e-4 {
                        ((val - min) / (max - min)).clamp(0.0, 1.0)
                    } else {
                        0.5
                    }
                } else {
                    0.5
                };
                let color = scalar_to_color(t);
                instances.push(ToolpathLineInstance::new(
                    start,
                    end,
                    color,
                    segment.order,
                    0.0,
                    0.0,
                ));
                continue;
            }

            let color = match data_view {
                ToolpathDataView::LineType => palette_color(segment.kind, segment.is_scarf),
                ToolpathDataView::FlowRate
                | ToolpathDataView::Speed
                | ToolpathDataView::ActualSpeed
                | ToolpathDataView::Acceleration
                | ToolpathDataView::ActualAcceleration
                | ToolpathDataView::FirstLayerMultiplier
                | ToolpathDataView::DirectionalFlowMultiplier
                | ToolpathDataView::SwellMultiplier
                | ToolpathDataView::CornerFlowMultiplier
                | ToolpathDataView::TransientPressureMultiplier
                | ToolpathDataView::SlopeCosine
                | ToolpathDataView::PressureAdvanceBlend
                | ToolpathDataView::PressureAdvanceValue
                    if segment.kind == MoveKind::Travel =>
                {
                    COLOR_TRAVEL
                }
                _ => {
                    let val = segment_scalar_value_with_profile(
                        segment,
                        start,
                        end,
                        data_view,
                        config,
                        machine,
                        profiles.get(index),
                    );
                    let t = if let Some((min, max)) = scalar_range {
                        if max > min + 1e-4 {
                            ((val - min) / (max - min)).clamp(0.0, 1.0)
                        } else {
                            0.5
                        }
                    } else {
                        0.5
                    };
                    scalar_to_color(t)
                }
            };

            let (w, h) = segment_bead_dimensions(segment, start, end, config);
            instances.push(ToolpathLineInstance::new(
                start,
                end,
                color,
                segment.order,
                w,
                h,
            ));
        }

        if path.segments.len() == count {
            prev_endpoint = Some(path.points[0]);
        } else {
            prev_endpoint = path.points.last().copied();
        }
    }
    instances
}

/// The `(min, max)` `order` value across all segments in `paths`, used to
/// size the scrub slider's range. `None` if `paths` contains no segments
/// at all.
pub fn order_range(paths: &[Path]) -> Option<(f64, f64)> {
    let mut range: Option<(f64, f64)> = None;
    for path in paths {
        for segment in &path.segments {
            range = Some(match range {
                None => (segment.order, segment.order),
                Some((min, max)) => (min.min(segment.order), max.max(segment.order)),
            });
        }
    }
    range
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::DVec3;
    use manifold_core::ids::ToolId;
    use manifold_core::toolpath::Segment;

    fn path_with_kinds(kinds: &[MoveKind], order: f64) -> Path {
        let points: Vec<DVec3> = (0..kinds.len())
            .map(|i| DVec3::new(i as f64, 0.0, 0.0))
            .collect();
        let segments = kinds
            .iter()
            .map(|&kind| Segment {
                island: 0,
                kind,
                speed: 60.0,
                extrusion_rate: 1.0,
                support_fraction: 0.0,
                order,
                extrusion_length: 0.0,
                line_width: 0.4,
                is_scarf: false,
                id: 0,
                channel_width: f64::INFINITY,
                flow_breakdown: None,
            })
            .collect();
        Path {
            points,
            segments,
            tool: ToolId(0),
            object: manifold_core::ids::ObjectId::default(),
        }
    }

    fn build_lines_default(paths: &[Path], scrub: f64) -> Vec<ToolpathLineInstance> {
        build_toolpath_lines(
            paths,
            scrub,
            ToolpathDataView::LineType,
            &manifold_core::SlicerConfig::default(),
            None,
        )
    }

    #[test]
    fn instance_count_matches_one_per_segment() {
        let path = path_with_kinds(
            &[MoveKind::WallOuter, MoveKind::WallInner, MoveKind::Infill],
            0.0,
        );
        let instances = build_lines_default(&[path], f64::INFINITY);
        assert_eq!(instances.len(), 3);
    }

    #[test]
    fn instance_count_sums_across_multiple_paths() {
        let path_a = path_with_kinds(&[MoveKind::WallOuter, MoveKind::WallOuter], 0.0);
        let path_b = path_with_kinds(
            &[MoveKind::Infill, MoveKind::Bridge, MoveKind::Overhang],
            0.0,
        );
        let instances = build_lines_default(&[path_a, path_b], f64::INFINITY);
        assert_eq!(instances.len(), 2 + 3);
    }

    #[test]
    fn segment_endpoints_wrap_around_the_closing_edge() {
        let path = path_with_kinds(&[MoveKind::WallOuter, MoveKind::WallOuter], 0.0);
        let instances = build_lines_default(&[path], f64::INFINITY);
        // Second segment closes the loop: points[1] -> points[0].
        assert_eq!(instances[1].start, [1.0, 0.0, 0.0]);
        assert_eq!(instances[1].end, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn open_path_does_not_wrap_around_to_closing_edge() {
        // Open polyline: 3 points, 2 segments (N points -> N - 1 segments).
        let points = vec![
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(2.0, 1.0, 0.0),
        ];
        let segments = vec![
            manifold_core::toolpath::Segment {
                kind: MoveKind::Infill,
                speed: 60.0,
                extrusion_rate: 1.0,
                support_fraction: 0.0,
                order: 0.0,
                extrusion_length: 0.0,
                line_width: 0.4,
                is_scarf: false,
                id: 0,
                island: 0,
                channel_width: f64::INFINITY,
                flow_breakdown: None,
            },
            manifold_core::toolpath::Segment {
                kind: MoveKind::Infill,
                speed: 60.0,
                extrusion_rate: 1.0,
                support_fraction: 0.0,
                order: 0.0,
                extrusion_length: 0.0,
                line_width: 0.4,
                is_scarf: false,
                id: 0,
                island: 0,
                channel_width: f64::INFINITY,
                flow_breakdown: None,
            },
        ];
        let path = Path {
            points,
            segments,
            tool: manifold_core::ids::ToolId::default(),
            object: manifold_core::ids::ObjectId::default(),
        };
        let instances = build_lines_default(&[path], f64::INFINITY);
        assert_eq!(instances.len(), 2);
        assert_eq!(instances[0].start, [0.0, 0.0, 0.0]);
        assert_eq!(instances[0].end, [1.0, 0.0, 0.0]);
        assert_eq!(instances[1].start, [1.0, 0.0, 0.0]);
        assert_eq!(instances[1].end, [2.0, 1.0, 0.0]);
        // Crucially: no closing move from [2.0, 1.0, 0.0] back to [0.0, 0.0, 0.0].
    }

    #[test]
    fn color_mapping_matches_palette_per_move_kind() {
        let cases = [
            (MoveKind::WallOuter, COLOR_WALL_OUTER),
            (MoveKind::WallInner, COLOR_WALL_INNER),
            (MoveKind::Infill, COLOR_INFILL),
            (MoveKind::Bridge, COLOR_BRIDGE),
            (MoveKind::Overhang, COLOR_OVERHANG),
            (MoveKind::Travel, COLOR_TRAVEL),
        ];
        for (kind, expected_color) in cases {
            let path = path_with_kinds(&[kind], 0.0);
            let instances = build_lines_default(&[path], f64::INFINITY);
            assert_eq!(instances[0].color, expected_color);
        }
    }

    #[test]
    fn order_is_propagated_to_instance() {
        let path = path_with_kinds(&[MoveKind::WallOuter, MoveKind::Infill], 0.75);
        let instances = build_lines_default(&[path], f64::INFINITY);
        assert!(instances.iter().all(|inst| inst.order == 0.75));
    }

    #[test]
    fn distinct_paths_can_carry_distinct_order_values() {
        let path_a = path_with_kinds(&[MoveKind::WallOuter], 0.0);
        let path_b = path_with_kinds(&[MoveKind::WallOuter], 1.0);
        let instances = build_lines_default(&[path_a, path_b], f64::INFINITY);
        assert_eq!(instances[0].order, 0.0);
        assert_eq!(instances[1].order, 1.0);
    }

    #[test]
    fn inter_path_travel_moves_are_emitted_between_disconnected_paths() {
        let path_a = Path {
            points: vec![DVec3::new(0.0, 0.0, 0.0), DVec3::new(10.0, 0.0, 0.0)],
            segments: vec![
                Segment {
                    island: 0,
                    kind: MoveKind::WallOuter,
                    speed: 60.0,
                    extrusion_rate: 1.0,
                    support_fraction: 0.0,
                    order: 0.0,
                    extrusion_length: 1.0,
                    line_width: 0.4,
                    is_scarf: false,
                    id: 0,
                    channel_width: f64::INFINITY,
                    flow_breakdown: None,
                },
                Segment {
                    island: 0,
                    kind: MoveKind::WallOuter,
                    speed: 60.0,
                    extrusion_rate: 1.0,
                    support_fraction: 0.0,
                    order: 0.0,
                    extrusion_length: 1.0,
                    line_width: 0.4,
                    is_scarf: false,
                    id: 0,
                    channel_width: f64::INFINITY,
                    flow_breakdown: None,
                },
            ],
            tool: ToolId(0),
            object: manifold_core::ids::ObjectId::default(),
        };
        let path_b = Path {
            points: vec![DVec3::new(20.0, 20.0, 0.2), DVec3::new(30.0, 20.0, 0.2)],
            segments: vec![
                Segment {
                    island: 0,
                    kind: MoveKind::WallOuter,
                    speed: 60.0,
                    extrusion_rate: 1.0,
                    support_fraction: 0.0,
                    order: 0.2,
                    extrusion_length: 1.0,
                    line_width: 0.4,
                    is_scarf: false,
                    id: 0,
                    channel_width: f64::INFINITY,
                    flow_breakdown: None,
                },
                Segment {
                    island: 0,
                    kind: MoveKind::WallOuter,
                    speed: 60.0,
                    extrusion_rate: 1.0,
                    support_fraction: 0.0,
                    order: 0.2,
                    extrusion_length: 1.0,
                    line_width: 0.4,
                    is_scarf: false,
                    id: 0,
                    channel_width: f64::INFINITY,
                    flow_breakdown: None,
                },
            ],
            tool: ToolId(0),
            object: manifold_core::ids::ObjectId::default(),
        };
        let instances = build_lines_default(&[path_a, path_b], f64::INFINITY);
        // path_a: 2 segments (ends at [0,0,0])
        // travel: [0,0,0] -> [20,20,0.2] (1 segment)
        // path_b: 2 segments
        assert_eq!(instances.len(), 2 + 1 + 2);
        assert_eq!(instances[2].start, [0.0, 0.0, 0.0]);
        assert_eq!(instances[2].end, [20.0, 20.0, 0.2]);
        assert_eq!(instances[2].color, COLOR_TRAVEL);
        assert_eq!(instances[2].order, 0.2);
    }

    #[test]
    fn empty_paths_produce_no_instances() {
        assert!(build_lines_default(&[], f64::INFINITY).is_empty());
    }

    #[test]
    fn scrub_order_excludes_segments_above_cutoff() {
        let path_a = path_with_kinds(&[MoveKind::WallOuter], 0.0);
        let path_b = path_with_kinds(&[MoveKind::Infill], 1.0);
        let instances = build_lines_default(&[path_a, path_b], 0.0);
        // Only path_a's segment (order 0.0) survives the <= 0.0 cutoff.
        assert_eq!(instances.len(), 1);
        assert_eq!(instances[0].order, 0.0);
    }

    #[test]
    fn scrub_order_is_inclusive_of_the_cutoff_value() {
        let path = path_with_kinds(&[MoveKind::WallOuter], 0.5);
        let instances = build_lines_default(&[path], 0.5);
        assert_eq!(instances.len(), 1);
    }

    #[test]
    fn order_range_spans_min_and_max_across_paths() {
        let path_a = path_with_kinds(&[MoveKind::WallOuter], 0.0);
        let path_b = path_with_kinds(&[MoveKind::Infill, MoveKind::Bridge], 2.0);
        let (min, max) = order_range(&[path_a, path_b]).unwrap();
        assert_eq!(min, 0.0);
        assert_eq!(max, 2.0);
    }

    #[test]
    fn order_range_is_none_for_empty_paths() {
        assert!(order_range(&[]).is_none());
    }

    #[test]
    fn line_type_legend_covers_all_line_types_and_has_valid_colors() {
        let legend = line_type_legend();
        assert_eq!(legend.len(), 9);
        for entry in legend {
            assert!(!entry.label.is_empty());
            assert!(entry.color[3] > 0.0);
        }
    }

    #[test]
    fn scarf_joint_segments_are_colored_with_scarf_palette() {
        let mut seg = Segment {
            island: 0,
            kind: MoveKind::WallOuter,
            speed: 60.0,
            extrusion_rate: 1.0,
            support_fraction: 1.0,
            order: 1.0,
            extrusion_length: 0.5,
            line_width: 0.4,
            is_scarf: true,
            id: 0,
            channel_width: f64::INFINITY,
            flow_breakdown: None,
        };
        assert_eq!(palette_color(seg.kind, seg.is_scarf), COLOR_SCARF_JOINT);
        seg.is_scarf = false;
        assert_eq!(palette_color(seg.kind, seg.is_scarf), COLOR_WALL_OUTER);
    }

    #[test]
    fn build_toolpath_lines_filtered_skips_hidden_line_types() {
        let path_a = path_with_kinds(&[MoveKind::WallOuter], 0.0);
        let path_b = path_with_kinds(&[MoveKind::Infill], 1.0);
        let mut hidden = std::collections::HashSet::new();
        hidden.insert(LineTypeKey::Travel);

        let instances = build_toolpath_lines_filtered(
            &[path_a, path_b],
            f64::INFINITY,
            ToolpathDataView::LineType,
            &manifold_core::SlicerConfig::default(),
            None,
            &hidden,
        );
        // Without travel moves, only path_a (1) + path_b (1) = 2 instances (inter-path travel omitted)
        assert_eq!(instances.len(), 2);
    }

    #[test]
    fn scalar_gradient_colors_scale_with_speed_and_accel() {
        let config = manifold_core::SlicerConfig::default();
        let points = vec![
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(10.0, 0.0, 0.0),
            DVec3::new(20.0, 0.0, 0.0),
        ];
        let segments = vec![
            Segment {
                island: 0,
                kind: MoveKind::WallOuter,
                speed: 1200.0, // 20 mm/s (low)
                extrusion_rate: 1.0,
                support_fraction: 0.0,
                order: 1.0,
                extrusion_length: 0.3,
                line_width: 0.4,
                is_scarf: false,
                id: 0,
                channel_width: f64::INFINITY,
                flow_breakdown: None,
            },
            Segment {
                island: 0,
                kind: MoveKind::Infill,
                speed: 12000.0, // 200 mm/s (high)
                extrusion_rate: 1.0,
                support_fraction: 0.0,
                order: 1.0,
                extrusion_length: 0.3,
                line_width: 0.4,
                is_scarf: false,
                id: 0,
                channel_width: f64::INFINITY,
                flow_breakdown: None,
            },
            Segment {
                island: 0,
                kind: MoveKind::Travel,
                speed: 18000.0, // 300 mm/s (travel)
                extrusion_rate: 0.0,
                support_fraction: 0.0,
                order: 1.0,
                extrusion_length: 0.0,
                line_width: 0.0,
                is_scarf: false,
                id: 0,
                channel_width: f64::INFINITY,
                flow_breakdown: None,
            },
        ];
        let path = Path {
            points,
            segments,
            tool: ToolId(0),
            object: manifold_core::ids::ObjectId::default(),
        };

        // Speed view: low speed gets blue (c[2] > c[0]), high speed gets red (c[0] > c[2]), travel gets COLOR_TRAVEL
        let speed_lines = build_toolpath_lines(
            std::slice::from_ref(&path),
            f64::INFINITY,
            ToolpathDataView::Speed,
            &config,
            None,
        );
        assert_eq!(speed_lines.len(), 3);
        assert!(speed_lines[0].color[2] > speed_lines[0].color[0]); // Blue > Red for low
        assert!(speed_lines[1].color[0] > speed_lines[1].color[2]); // Red > Blue for high
        assert_eq!(speed_lines[2].color, COLOR_TRAVEL); // Travel move gets COLOR_TRAVEL

        // Flow rate view: travel move gets grey COLOR_TRAVEL, extrusions get scalar colors
        let flow_lines = build_toolpath_lines(
            std::slice::from_ref(&path),
            f64::INFINITY,
            ToolpathDataView::FlowRate,
            &config,
            None,
        );
        assert_eq!(flow_lines.len(), 3);
        assert_eq!(flow_lines[2].color, COLOR_TRAVEL);

        // Acceleration view
        let accel_lines = build_toolpath_lines(
            std::slice::from_ref(&path),
            f64::INFINITY,
            ToolpathDataView::Acceleration,
            &config,
            None,
        );
        assert_eq!(accel_lines.len(), 3);

        // Actual Speed view: plans acceleration ramp and returns valid line colors
        let actual_speed_lines = build_toolpath_lines(
            std::slice::from_ref(&path),
            f64::INFINITY,
            ToolpathDataView::ActualSpeed,
            &config,
            None,
        );
        assert_eq!(actual_speed_lines.len(), 3);
        assert_eq!(actual_speed_lines[2].color, COLOR_TRAVEL);

        // Actual Acceleration view
        let actual_accel_lines = build_toolpath_lines(
            std::slice::from_ref(&path),
            f64::INFINITY,
            ToolpathDataView::ActualAcceleration,
            &config,
            None,
        );
        assert_eq!(actual_accel_lines.len(), 3);
        assert_eq!(actual_accel_lines[2].color, COLOR_TRAVEL);

        // Travel Durations view: only the Travel move is emitted, non-travel extrusion moves are filtered out
        let travel_lines = build_toolpath_lines(
            std::slice::from_ref(&path),
            f64::INFINITY,
            ToolpathDataView::TravelDurations,
            &config,
            None,
        );
        assert_eq!(travel_lines.len(), 1);
        assert_eq!(travel_lines[0].start, [20.0, 0.0, 0.0]);
        assert_eq!(travel_lines[0].end, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn segment_bead_dimensions_scales_with_extrusion_volume_ratio() {
        let config = manifold_core::SlicerConfig {
            wall_line_width: 0.4,
            layer_height: 0.2,
            filament_diameter: 1.75,
            ..manifold_core::SlicerConfig::default()
        };

        let start = DVec3::new(0.0, 0.0, 0.0);
        let end = DVec3::new(10.0, 0.0, 0.0);

        // Travel move: dimensions must be 0.0
        let travel_seg = Segment {
            kind: MoveKind::Travel,
            extrusion_length: 0.0,
            line_width: 0.0,
            ..Segment::default()
        };
        let (w_travel, h_travel) = segment_bead_dimensions(&travel_seg, start, end, &config);
        assert_eq!(w_travel, 0.0);
        assert_eq!(h_travel, 0.0);

        // Standard 100% extrusion move
        let nominal_bead_area = manifold_core::extrusion::bead_cross_section_area(0.4, 0.2);
        let filament_area = manifold_core::extrusion::filament_cross_section_area(1.75);
        let nominal_e = (10.0 * nominal_bead_area) / filament_area;

        let full_seg = Segment {
            kind: MoveKind::WallOuter,
            extrusion_length: nominal_e,
            line_width: 0.4,
            ..Segment::default()
        };
        let (w_full, h_full) = segment_bead_dimensions(&full_seg, start, end, &config);
        assert!((w_full - 0.4).abs() < 1e-3);
        assert!((h_full - 0.2).abs() < 1e-3);

        // Scarf joint 10% low-flow move
        let low_flow_seg = Segment {
            kind: MoveKind::WallOuter,
            extrusion_length: nominal_e * 0.10,
            line_width: 0.4,
            is_scarf: true,
            ..Segment::default()
        };
        let (w_low, h_low) = segment_bead_dimensions(&low_flow_seg, start, end, &config);
        assert!(w_low < w_full);
        assert!(h_low < h_full);
        assert!((w_low - 0.4 * 0.10_f64.sqrt()).abs() < 1e-3);
    }

    #[test]
    fn multiplier_data_views_read_the_matching_flow_breakdown_field() {
        let config = manifold_core::SlicerConfig::default();
        let start = DVec3::new(0.0, 0.0, 0.0);
        let end = DVec3::new(10.0, 0.0, 0.0);
        let breakdown = manifold_core::toolpath::FlowBreakdown {
            slope_cosine: 0.81,
            first_layer_mult: 1.15,
            directional_flow_mult: 0.92,
            swell_mult: 1.08,
            corner_flow_mult: 0.77,
            transient_pressure_mult: 0.65,
        };
        let seg = Segment {
            kind: MoveKind::WallOuter,
            extrusion_length: 1.0,
            flow_breakdown: Some(breakdown),
            ..Segment::default()
        };

        let cases = [
            (ToolpathDataView::SlopeCosine, breakdown.slope_cosine),
            (
                ToolpathDataView::FirstLayerMultiplier,
                breakdown.first_layer_mult,
            ),
            (
                ToolpathDataView::DirectionalFlowMultiplier,
                breakdown.directional_flow_mult,
            ),
            (ToolpathDataView::SwellMultiplier, breakdown.swell_mult),
            (
                ToolpathDataView::CornerFlowMultiplier,
                breakdown.corner_flow_mult,
            ),
            (
                ToolpathDataView::TransientPressureMultiplier,
                breakdown.transient_pressure_mult,
            ),
        ];
        for (view, expected) in cases {
            let val = segment_scalar_value(&seg, start, end, view, &config, None);
            assert!(
                (val - expected).abs() < 1e-9,
                "{:?} expected {expected}, got {val}",
                view
            );
        }

        // A segment with no flow_breakdown reports the neutral 1.0 for every
        // multiplier view -- not 0.0 -- since these are ratios, and a
        // missing/unpopulated breakdown means "no adjustment applied".
        let unpopulated = Segment {
            kind: MoveKind::Travel,
            flow_breakdown: None,
            ..Segment::default()
        };
        for (view, _) in cases {
            let val = segment_scalar_value(&unpopulated, start, end, view, &config, None);
            assert!(
                (val - 1.0).abs() < 1e-9,
                "{:?} with no flow_breakdown should default to the neutral 1.0, got {val}",
                view
            );
        }
    }

    #[test]
    fn pressure_advance_blend_view_maps_flow_rate_to_calibration_plateaus() {
        let fluid_dynamics = manifold_core::fluid_dynamics::FluidDynamicsConfig {
            pa_calibration_low: (0.045, 2.0),
            pa_calibration_high: (0.030, 8.0),
            ..manifold_core::fluid_dynamics::FluidDynamicsConfig::default()
        };
        let config = manifold_core::SlicerConfig {
            filament_diameter: 2.0,
            fluid_dynamics: Some(fluid_dynamics),
            ..manifold_core::SlicerConfig::default()
        };
        let start = DVec3::new(0.0, 0.0, 0.0);
        let end = DVec3::new(10.0, 0.0, 0.0);

        // q = extrusion_length * fil_area / duration, with fil_area = pi (diameter 2.0)
        // and duration = 10.0 s (length 10mm at speed 60mm/min = 1mm/s).
        let seg_for_flow = |q: f64| Segment {
            kind: MoveKind::WallOuter,
            extrusion_length: q * 10.0 / std::f64::consts::PI,
            speed: 60.0,
            ..Segment::default()
        };

        let cases = [
            (1.0, 0.0),  // below q_low: pinned at the low plateau
            (2.0, 0.0),  // exactly q_low
            (4.0, 0.5),  // geometric mean of q_low/q_high: exact midpoint
            (8.0, 1.0),  // exactly q_high
            (16.0, 1.0), // above q_high: pinned at the high plateau
        ];
        for (q, expected) in cases {
            let seg = seg_for_flow(q);
            let val = segment_scalar_value(
                &seg,
                start,
                end,
                ToolpathDataView::PressureAdvanceBlend,
                &config,
                None,
            );
            assert!(
                (val - expected).abs() < 1e-6,
                "q={q} expected blend {expected}, got {val}"
            );
        }

        // Travel moves and zero-extrusion segments carry no flow-rate signal
        // and must report the low plateau, not stray into the gradient.
        let travel = Segment {
            kind: MoveKind::Travel,
            extrusion_length: 0.0,
            speed: 60.0,
            ..Segment::default()
        };
        let val = segment_scalar_value(
            &travel,
            start,
            end,
            ToolpathDataView::PressureAdvanceBlend,
            &config,
            None,
        );
        assert_eq!(val, 0.0);
    }

    #[test]
    fn pressure_advance_value_view_matches_the_real_calibrated_model() {
        let fluid_dynamics = manifold_core::fluid_dynamics::FluidDynamicsConfig {
            pa_calibration_low: (0.045, 2.0),
            pa_calibration_high: (0.030, 8.0),
            ..manifold_core::fluid_dynamics::FluidDynamicsConfig::default()
        };
        let config = manifold_core::SlicerConfig {
            filament_diameter: 2.0,
            fluid_dynamics: Some(fluid_dynamics),
            ..manifold_core::SlicerConfig::default()
        };
        let start = DVec3::new(0.0, 0.0, 0.0);
        let end = DVec3::new(10.0, 0.0, 0.0);

        // Same flow-rate construction as the PA-blend test above: q =
        // extrusion_length * fil_area / duration.
        let seg_for_flow = |q: f64| Segment {
            kind: MoveKind::WallOuter,
            extrusion_length: q * 10.0 / std::f64::consts::PI,
            speed: 60.0,
            ..Segment::default()
        };
        let pa_value_at = |q: f64| {
            segment_scalar_value(
                &seg_for_flow(q),
                start,
                end,
                ToolpathDataView::PressureAdvanceValue,
                &config,
                None,
            )
        };

        // The two-point calibration curve is constructed to pass through
        // both calibration points exactly -- default_nozzle_temperature()
        // (240.0C) matches FluidDynamicsConfig::default()'s reference_temp_c
        // (240.0C), so the temperature-correction term is a no-op here and
        // this is an exact, not approximate, check.
        assert!(
            (pa_value_at(2.0) - 0.045).abs() < 1e-9,
            "PA at q_low must equal the calibrated low-flow value exactly, got {}",
            pa_value_at(2.0)
        );
        assert!(
            (pa_value_at(8.0) - 0.030).abs() < 1e-9,
            "PA at q_high must equal the calibrated high-flow value exactly, got {}",
            pa_value_at(8.0)
        );

        // Shear-thinning: PA strictly decreases as flow rate increases, so
        // an intermediate flow rate must land strictly between the two
        // calibration points, not just clamp to one end.
        let pa_mid = pa_value_at(4.0);
        assert!(
            pa_mid > 0.030 && pa_mid < 0.045,
            "PA at an intermediate flow rate must lie strictly between the calibration points, got {pa_mid}"
        );

        // Travel moves carry no flow-rate signal and must report 0.0, not
        // extrapolate a spurious PA value from a zero flow rate.
        let travel = Segment {
            kind: MoveKind::Travel,
            extrusion_length: 0.0,
            speed: 60.0,
            ..Segment::default()
        };
        assert_eq!(
            segment_scalar_value(
                &travel,
                start,
                end,
                ToolpathDataView::PressureAdvanceValue,
                &config,
                None,
            ),
            0.0
        );

        // With no fluid-dynamics model configured, fall back to the static
        // pressure_advance config value -- matching transient_pressure's
        // own fallback for the identical case.
        let static_config = manifold_core::SlicerConfig {
            filament_diameter: 2.0,
            fluid_dynamics: None,
            pressure_advance: Some(0.035),
            ..manifold_core::SlicerConfig::default()
        };
        let val = segment_scalar_value(
            &seg_for_flow(4.0),
            start,
            end,
            ToolpathDataView::PressureAdvanceValue,
            &static_config,
            None,
        );
        assert_eq!(val, 0.035);
    }
}
