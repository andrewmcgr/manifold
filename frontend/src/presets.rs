use std::path::PathBuf;

use bevy::prelude::*;
use serde::{Deserialize, Serialize};

use crate::config::{
    BedShape, BrickLayerMode, ClearancePointXZ, ClearanceProfileMode, FiveAxisKinematicsMode,
    GcodeDialect, LayerHeightConfig, LineWidthConfig, MoonrakerNetworkConfig, OrderFieldMode,
    PrinterNetworkProtocol, SlicerConfig, SolidInfillPattern, SparseInfillPattern, SupportStyle,
    ToolConfig, WallPrintOrder,
};

/// Top-level tab in the left workspace panel (`Project Setup` vs. `Configuration`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum LeftPanelTab {
    #[default]
    ProjectSetup,
    Configuration,
}

impl LeftPanelTab {
    pub fn label(self) -> &'static str {
        match self {
            Self::ProjectSetup => "Project Setup",
            Self::Configuration => "Configuration & Presets",
        }
    }
}

/// The four saved preset categories in Manifold2.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PresetCategory {
    #[default]
    Printer,
    Tool,
    Filament,
    Process,
}

impl PresetCategory {
    pub const ALL: [Self; 4] = [Self::Printer, Self::Tool, Self::Filament, Self::Process];

    pub fn label(self) -> &'static str {
        match self {
            Self::Printer => "Printer",
            Self::Tool => "Tool",
            Self::Filament => "Filament",
            Self::Process => "Process",
        }
    }

    pub fn subdir_name(self) -> &'static str {
        match self {
            Self::Printer => "printers",
            Self::Tool => "tools",
            Self::Filament => "filaments",
            Self::Process => "processes",
        }
    }
}

fn approx_eq(a: f32, b: f32) -> bool {
    (a - b).abs() <= 1.0e-4
}

fn approx_eq_vec3(a: [f32; 3], b: [f32; 3]) -> bool {
    approx_eq(a[0], b[0]) && approx_eq(a[1], b[1]) && approx_eq(a[2], b[2])
}

fn approx_eq_line_width(a: LineWidthConfig, b: LineWidthConfig) -> bool {
    approx_eq(a.target_mm, b.target_mm)
        && approx_eq(a.min_mm, b.min_mm)
        && approx_eq(a.max_mm, b.max_mm)
}

fn approx_eq_layer_height(a: LayerHeightConfig, b: LayerHeightConfig) -> bool {
    approx_eq(a.target_mm, b.target_mm)
        && approx_eq(a.min_mm, b.min_mm)
        && approx_eq(a.max_mm, b.max_mm)
        && approx_eq(a.first_layer_mm, b.first_layer_mm)
        && approx_eq(a.top_layer_mm, b.top_layer_mm)
}

// ============================================================================
// 1. PrinterPreset — Machine Hardware, Kinematics, Motion Limits & G-code Macros
// ============================================================================

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PrinterPreset {
    pub name: String,
    #[serde(skip)]
    pub is_builtin: bool,

    // Build Volume & Bed Coordinate Origin
    pub bed_shape: BedShape,
    pub width_x: f32,
    pub depth_y: f32,
    pub height_z: f32,
    pub gcode_origin_at_bed_center: bool,

    // Multi-Axis Kinematics in Bed Coordinates (Rule 1 & Rule 2)
    pub kinematics_mode: FiveAxisKinematicsMode,
    pub nozzle_axis_bed: [f32; 3],
    pub gravity_direction_bed: [f32; 3],
    pub bed_tilt_rx_deg: f32,
    pub bed_tilt_ry_deg: f32,
    pub a_min_deg: f32,
    pub a_max_deg: f32,
    pub invert_a: bool,
    pub max_a_speed_deg_s: f32,
    pub a_accel_deg_s2: f32,
    pub b_min_deg: f32,
    pub b_max_deg: f32,
    pub c_continuous: bool,
    pub invert_b: bool,
    pub invert_c: bool,
    pub c_zero_offset_deg: f32,
    pub max_b_speed_deg_s: f32,
    pub max_c_speed_deg_s: f32,
    pub b_accel_deg_s2: f32,
    pub c_accel_deg_s2: f32,
    pub orientation_smoothing_window_mm: f32,
    pub bed_tilts_with_b: bool,
    pub use_inverse_time_feedrate_g93: bool,

    // Machine Motion Envelope & Stepper Dynamics
    pub travel_speed_mms: f32,
    pub z_hop_speed_mms: f32,
    pub outer_wall_accel_mms2: f32,
    pub inner_wall_accel_mms2: f32,
    pub infill_accel_mms2: f32,
    pub travel_accel_mms2: f32,
    pub square_corner_velocity_mms: f32,
    pub minimum_cruise_ratio: f32,
    pub use_stepper_dynamics: bool,
    pub max_stepper_speed_mms: f32,
    pub z_travel_penalty: f32,

    // Firmware Features & Start/End G-code Templates
    pub gcode_dialect: GcodeDialect,
    pub enable_dynamic_pa: bool,
    pub enable_slicer_pa_subdivision: bool,
    pub enable_corner_flow_comp: bool,
    pub corner_flow_comp_ratio: f32,
    pub emit_exclude_object: bool,
    pub emit_thumbnails: bool,
    pub start_gcode_template: String,
    pub end_gcode_template: String,
    /// Optional default extruder/toolhead count when selecting this printer preset (e.g. `Some(4)` or `Some(8)` for Bondtech INDX).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_tool_count: Option<usize>,

    // Network Printing (Moonraker / Klipper)
    pub network: MoonrakerNetworkConfig,
}

impl Default for PrinterPreset {
    fn default() -> Self {
        Self::from_config(
            "256³ CoreXY (Standard Klipper)",
            &SlicerConfig::default(),
            false,
        )
    }
}

impl PrinterPreset {
    pub fn from_config(name: impl Into<String>, cfg: &SlicerConfig, is_builtin: bool) -> Self {
        Self {
            name: name.into(),
            is_builtin,
            bed_shape: cfg.bed.shape,
            width_x: cfg.bed.width_x,
            depth_y: cfg.bed.depth_y,
            height_z: cfg.bed.height_z,
            gcode_origin_at_bed_center: cfg.gcode.gcode_origin_at_bed_center,
            kinematics_mode: cfg.machine.kinematics_mode,
            nozzle_axis_bed: cfg.machine.nozzle_axis_bed,
            gravity_direction_bed: cfg.machine.gravity_direction_bed,
            bed_tilt_rx_deg: cfg.machine.bed_tilt_rx_deg,
            bed_tilt_ry_deg: cfg.machine.bed_tilt_ry_deg,
            a_min_deg: cfg.machine.a_min_deg,
            a_max_deg: cfg.machine.a_max_deg,
            invert_a: cfg.machine.invert_a,
            max_a_speed_deg_s: cfg.machine.max_a_speed_deg_s,
            a_accel_deg_s2: cfg.machine.a_accel_deg_s2,
            b_min_deg: cfg.machine.b_min_deg,
            b_max_deg: cfg.machine.b_max_deg,
            c_continuous: cfg.machine.c_continuous,
            invert_b: cfg.machine.invert_b,
            invert_c: cfg.machine.invert_c,
            c_zero_offset_deg: cfg.machine.c_zero_offset_deg,
            max_b_speed_deg_s: cfg.machine.max_b_speed_deg_s,
            max_c_speed_deg_s: cfg.machine.max_c_speed_deg_s,
            b_accel_deg_s2: cfg.machine.b_accel_deg_s2,
            c_accel_deg_s2: cfg.machine.c_accel_deg_s2,
            orientation_smoothing_window_mm: cfg.machine.orientation_smoothing_window_mm,
            bed_tilts_with_b: cfg.machine.bed_tilts_with_b,
            use_inverse_time_feedrate_g93: cfg.machine.use_inverse_time_feedrate_g93,
            travel_speed_mms: cfg.gcode.travel_speed_mms,
            z_hop_speed_mms: cfg.gcode.z_hop_speed_mms,
            outer_wall_accel_mms2: cfg.gcode.outer_wall_accel_mms2,
            inner_wall_accel_mms2: cfg.gcode.inner_wall_accel_mms2,
            infill_accel_mms2: cfg.gcode.infill_accel_mms2,
            travel_accel_mms2: cfg.gcode.travel_accel_mms2,
            square_corner_velocity_mms: cfg.gcode.square_corner_velocity_mms,
            minimum_cruise_ratio: cfg.gcode.minimum_cruise_ratio,
            use_stepper_dynamics: cfg.gcode.use_stepper_dynamics,
            max_stepper_speed_mms: cfg.gcode.max_stepper_speed_mms,
            z_travel_penalty: cfg.gcode.z_travel_penalty,
            gcode_dialect: cfg.gcode.gcode_dialect,
            enable_dynamic_pa: cfg.gcode.enable_dynamic_pa,
            enable_slicer_pa_subdivision: cfg.gcode.enable_slicer_pa_subdivision,
            enable_corner_flow_comp: cfg.gcode.enable_corner_flow_comp,
            corner_flow_comp_ratio: cfg.gcode.corner_flow_comp_ratio,
            emit_exclude_object: cfg.gcode.emit_exclude_object,
            emit_thumbnails: cfg.gcode.emit_thumbnails,
            start_gcode_template: cfg.gcode.start_gcode_template.clone(),
            end_gcode_template: cfg.gcode.end_gcode_template.clone(),
            default_tool_count: None,
            network: cfg.network.clone(),
        }
    }

    pub fn apply_to_config(&self, cfg: &mut SlicerConfig) {
        cfg.printer_preset_name = self.name.clone();
        cfg.bed.shape = self.bed_shape;
        cfg.bed.width_x = self.width_x;
        cfg.bed.depth_y = self.depth_y;
        cfg.bed.height_z = self.height_z;
        cfg.gcode.gcode_origin_at_bed_center = self.gcode_origin_at_bed_center;
        cfg.machine.kinematics_mode = self.kinematics_mode;
        cfg.machine.nozzle_axis_bed = self.nozzle_axis_bed;
        cfg.machine.gravity_direction_bed = self.gravity_direction_bed;
        cfg.machine.bed_tilt_rx_deg = self.bed_tilt_rx_deg;
        cfg.machine.bed_tilt_ry_deg = self.bed_tilt_ry_deg;
        cfg.machine.a_min_deg = self.a_min_deg;
        cfg.machine.a_max_deg = self.a_max_deg;
        cfg.machine.invert_a = self.invert_a;
        cfg.machine.max_a_speed_deg_s = self.max_a_speed_deg_s;
        cfg.machine.a_accel_deg_s2 = self.a_accel_deg_s2;
        cfg.machine.b_min_deg = self.b_min_deg;
        cfg.machine.b_max_deg = self.b_max_deg;
        cfg.machine.c_continuous = self.c_continuous;
        cfg.machine.invert_b = self.invert_b;
        cfg.machine.invert_c = self.invert_c;
        cfg.machine.c_zero_offset_deg = self.c_zero_offset_deg;
        cfg.machine.max_b_speed_deg_s = self.max_b_speed_deg_s;
        cfg.machine.max_c_speed_deg_s = self.max_c_speed_deg_s;
        cfg.machine.b_accel_deg_s2 = self.b_accel_deg_s2;
        cfg.machine.c_accel_deg_s2 = self.c_accel_deg_s2;
        cfg.machine.orientation_smoothing_window_mm = self.orientation_smoothing_window_mm;
        cfg.machine.bed_tilts_with_b = self.bed_tilts_with_b;
        cfg.machine.use_inverse_time_feedrate_g93 = self.use_inverse_time_feedrate_g93;
        cfg.gcode.travel_speed_mms = self.travel_speed_mms;
        cfg.gcode.z_hop_speed_mms = self.z_hop_speed_mms;
        cfg.gcode.outer_wall_accel_mms2 = self.outer_wall_accel_mms2;
        cfg.gcode.inner_wall_accel_mms2 = self.inner_wall_accel_mms2;
        cfg.gcode.infill_accel_mms2 = self.infill_accel_mms2;
        cfg.gcode.travel_accel_mms2 = self.travel_accel_mms2;
        cfg.gcode.square_corner_velocity_mms = self.square_corner_velocity_mms;
        cfg.gcode.minimum_cruise_ratio = self.minimum_cruise_ratio;
        cfg.gcode.use_stepper_dynamics = self.use_stepper_dynamics;
        cfg.gcode.max_stepper_speed_mms = self.max_stepper_speed_mms;
        cfg.gcode.z_travel_penalty = self.z_travel_penalty;
        cfg.gcode.gcode_dialect = self.gcode_dialect;
        cfg.gcode.enable_dynamic_pa = self.enable_dynamic_pa;
        cfg.gcode.enable_slicer_pa_subdivision = self.enable_slicer_pa_subdivision;
        cfg.gcode.enable_corner_flow_comp = self.enable_corner_flow_comp;
        cfg.gcode.corner_flow_comp_ratio = self.corner_flow_comp_ratio;
        cfg.gcode.emit_exclude_object = self.emit_exclude_object;
        cfg.gcode.emit_thumbnails = self.emit_thumbnails;
        cfg.gcode.start_gcode_template = self.start_gcode_template.clone();
        cfg.gcode.end_gcode_template = self.end_gcode_template.clone();
        if let Some(target_tools) = self.default_tool_count {
            let target_tools = target_tools.clamp(1, 16);
            let template_tool = cfg.active_tool().cloned().unwrap_or_default();
            if target_tools > 1
                && cfg.tools.len() == 2
                && cfg.tools[1].name == "T1 - 0.6mm High-Flow"
            {
                cfg.tools.truncate(1);
            }
            if cfg.tools.len() > target_tools {
                cfg.tools.truncate(target_tools);
            }
            while cfg.tools.len() < target_tools {
                let idx = cfg.tools.len();
                let mut t = template_tool.clone();
                t.id = idx as u32;
                let (hex, srgba) = crate::config::DEFAULT_FILAMENT_SLOT_PALETTE
                    [idx % crate::config::DEFAULT_FILAMENT_SLOT_PALETTE.len()];
                t.filament_color_hex = hex.to_string();
                t.filament_color_srgba = srgba;
                let suffix = if self.gcode_dialect == GcodeDialect::BambuLab {
                    "AMS"
                } else {
                    "INDX"
                };
                t.name = format!("T{idx} - {:.1}mm {suffix}", t.nozzle_diameter_mm);
                cfg.tools.push(t);
            }
            if cfg.active_tool_index >= cfg.tools.len() {
                cfg.active_tool_index = 0;
            }
        }
        cfg.network = self.network.clone();
        cfg.machine.sanitize();
        cfg.gcode.sanitize();
        cfg.network.sanitize();
    }

    pub fn matches_config(&self, cfg: &SlicerConfig) -> bool {
        self.bed_shape == cfg.bed.shape
            && approx_eq(self.width_x, cfg.bed.width_x)
            && approx_eq(self.depth_y, cfg.bed.depth_y)
            && approx_eq(self.height_z, cfg.bed.height_z)
            && self.gcode_origin_at_bed_center == cfg.gcode.gcode_origin_at_bed_center
            && self.kinematics_mode == cfg.machine.kinematics_mode
            && approx_eq_vec3(self.nozzle_axis_bed, cfg.machine.nozzle_axis_bed)
            && approx_eq_vec3(self.gravity_direction_bed, cfg.machine.gravity_direction_bed)
            && approx_eq(self.bed_tilt_rx_deg, cfg.machine.bed_tilt_rx_deg)
            && approx_eq(self.bed_tilt_ry_deg, cfg.machine.bed_tilt_ry_deg)
            && approx_eq(self.a_min_deg, cfg.machine.a_min_deg)
            && approx_eq(self.a_max_deg, cfg.machine.a_max_deg)
            && self.invert_a == cfg.machine.invert_a
            && approx_eq(self.max_a_speed_deg_s, cfg.machine.max_a_speed_deg_s)
            && approx_eq(self.a_accel_deg_s2, cfg.machine.a_accel_deg_s2)
            && approx_eq(self.b_min_deg, cfg.machine.b_min_deg)
            && approx_eq(self.b_max_deg, cfg.machine.b_max_deg)
            && self.c_continuous == cfg.machine.c_continuous
            && self.invert_b == cfg.machine.invert_b
            && self.invert_c == cfg.machine.invert_c
            && approx_eq(self.c_zero_offset_deg, cfg.machine.c_zero_offset_deg)
            && approx_eq(self.max_b_speed_deg_s, cfg.machine.max_b_speed_deg_s)
            && approx_eq(self.max_c_speed_deg_s, cfg.machine.max_c_speed_deg_s)
            && approx_eq(self.b_accel_deg_s2, cfg.machine.b_accel_deg_s2)
            && approx_eq(self.c_accel_deg_s2, cfg.machine.c_accel_deg_s2)
            && approx_eq(
                self.orientation_smoothing_window_mm,
                cfg.machine.orientation_smoothing_window_mm,
            )
            && self.bed_tilts_with_b == cfg.machine.bed_tilts_with_b
            && self.use_inverse_time_feedrate_g93 == cfg.machine.use_inverse_time_feedrate_g93
            && approx_eq(self.travel_speed_mms, cfg.gcode.travel_speed_mms)
            && approx_eq(self.z_hop_speed_mms, cfg.gcode.z_hop_speed_mms)
            && approx_eq(self.outer_wall_accel_mms2, cfg.gcode.outer_wall_accel_mms2)
            && approx_eq(self.inner_wall_accel_mms2, cfg.gcode.inner_wall_accel_mms2)
            && approx_eq(self.infill_accel_mms2, cfg.gcode.infill_accel_mms2)
            && approx_eq(self.travel_accel_mms2, cfg.gcode.travel_accel_mms2)
            && approx_eq(
                self.square_corner_velocity_mms,
                cfg.gcode.square_corner_velocity_mms,
            )
            && approx_eq(self.minimum_cruise_ratio, cfg.gcode.minimum_cruise_ratio)
            && self.use_stepper_dynamics == cfg.gcode.use_stepper_dynamics
            && approx_eq(self.max_stepper_speed_mms, cfg.gcode.max_stepper_speed_mms)
            && approx_eq(self.z_travel_penalty, cfg.gcode.z_travel_penalty)
            && self.gcode_dialect == cfg.gcode.gcode_dialect
            && self.enable_dynamic_pa == cfg.gcode.enable_dynamic_pa
            && self.enable_slicer_pa_subdivision == cfg.gcode.enable_slicer_pa_subdivision
            && self.enable_corner_flow_comp == cfg.gcode.enable_corner_flow_comp
            && approx_eq(self.corner_flow_comp_ratio, cfg.gcode.corner_flow_comp_ratio)
            && self.emit_exclude_object == cfg.gcode.emit_exclude_object
            && self.emit_thumbnails == cfg.gcode.emit_thumbnails
            && self.start_gcode_template == cfg.gcode.start_gcode_template
            && self.end_gcode_template == cfg.gcode.end_gcode_template
            && self.network.protocol == cfg.network.protocol
            && self.network.printer_uri == cfg.network.printer_uri
            && self.network.web_ui_uri == cfg.network.web_ui_uri
            && self.network.username == cfg.network.username
            && self.network.password == cfg.network.password
            && self.network.api_key == cfg.network.api_key
            && self.network.ca_cert_pem_path == cfg.network.ca_cert_pem_path
            && self.network.client_cert_pem_path == cfg.network.client_cert_pem_path
            && self.network.client_key_pem_path == cfg.network.client_key_pem_path
            && self.network.enable_status_polling == cfg.network.enable_status_polling
            && approx_eq(self.network.poll_interval_s, cfg.network.poll_interval_s)
    }
}

// ============================================================================
// 2. ToolPreset — Nozzle Geometry, 5/6-Axis Rotational Limits, XZ Clearance & Extruder
// ============================================================================

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ToolPreset {
    pub name: String,
    #[serde(skip)]
    pub is_builtin: bool,

    // Nozzle & Multi-Axis Orientation Capabilities
    pub nozzle_diameter_mm: f32,
    pub has_rotational_axes: bool,
    pub max_angle_from_bed_normal_deg: f32,
    pub enable_flat_nozzle_comp: bool,
    pub clearance_mode: ClearanceProfileMode,
    pub clearance_profile_xz: Vec<ClearancePointXZ>,
    pub cone_tip_radius_mm: f32,
    pub cone_half_angle_deg: f32,
    pub cone_height_mm: f32,
    pub rect_tip_half_width_x_mm: f32,
    pub rect_tip_half_width_y_mm: f32,
    pub rect_half_angle_x_deg: f32,
    pub rect_half_angle_y_deg: f32,
    pub rect_height_mm: f32,

    // Extruder Hotend Capacity & Retraction / Wipe Baseline
    pub max_volumetric_flow_mm3s: f32,
    pub retract_length_mm: f32,
    pub retract_speed_mms: f32,
    pub min_travel_for_retract_mm: f32,
    pub z_hop_mm: f32,
    pub wipe_length_mm: f32,
    pub pre_retract_taper_mm: f32,
    pub tool_change_time_s: f32,
    pub tool_change_hop_before: bool,
    pub tool_change_approach_after: bool,
}

impl Default for ToolPreset {
    fn default() -> Self {
        Self::from_config(
            "0.4mm Standard 5/6-Axis Conical",
            &SlicerConfig::default(),
            false,
        )
    }
}

impl ToolPreset {
    pub fn from_config(name: impl Into<String>, cfg: &SlicerConfig, is_builtin: bool) -> Self {
        let tool = cfg.active_tool().cloned().unwrap_or_default();
        Self {
            name: name.into(),
            is_builtin,
            nozzle_diameter_mm: tool.nozzle_diameter_mm,
            has_rotational_axes: tool.has_rotational_axes,
            max_angle_from_bed_normal_deg: tool.max_angle_from_bed_normal_deg,
            enable_flat_nozzle_comp: cfg.gcode.enable_flat_nozzle_comp,
            clearance_mode: tool.clearance_mode,
            clearance_profile_xz: tool.clearance_profile_xz,
            cone_tip_radius_mm: tool.cone_tip_radius_mm,
            cone_half_angle_deg: tool.cone_half_angle_deg,
            cone_height_mm: tool.cone_height_mm,
            rect_tip_half_width_x_mm: tool.rect_tip_half_width_x_mm,
            rect_tip_half_width_y_mm: tool.rect_tip_half_width_y_mm,
            rect_half_angle_x_deg: tool.rect_half_angle_x_deg,
            rect_half_angle_y_deg: tool.rect_half_angle_y_deg,
            rect_height_mm: tool.rect_height_mm,
            max_volumetric_flow_mm3s: cfg.gcode.max_volumetric_flow_mm3s,
            retract_length_mm: cfg.gcode.retract_length_mm,
            retract_speed_mms: cfg.gcode.retract_speed_mms,
            min_travel_for_retract_mm: cfg.gcode.min_travel_for_retract_mm,
            z_hop_mm: cfg.gcode.z_hop_mm,
            wipe_length_mm: cfg.gcode.wipe_length_mm,
            pre_retract_taper_mm: cfg.gcode.pre_retract_taper_mm,
            tool_change_time_s: cfg.gcode.tool_change_time_s,
            tool_change_hop_before: cfg.gcode.tool_change_hop_before,
            tool_change_approach_after: cfg.gcode.tool_change_approach_after,
        }
    }

    pub fn apply_to_config(&self, cfg: &mut SlicerConfig) {
        cfg.tool_preset_name = self.name.clone();
        if cfg.tools.is_empty() {
            cfg.tools.push(ToolConfig::default());
            cfg.active_tool_index = 0;
        }
        let idx = cfg.active_tool_index.min(cfg.tools.len() - 1);
        if let Some(tool) = cfg.tools.get_mut(idx) {
            tool.name = format!("T{} - {}", tool.id, self.name);
            tool.nozzle_diameter_mm = self.nozzle_diameter_mm;
            tool.has_rotational_axes = self.has_rotational_axes;
            tool.max_angle_from_bed_normal_deg = self.max_angle_from_bed_normal_deg;
            tool.clearance_mode = self.clearance_mode;
            tool.clearance_profile_xz = self.clearance_profile_xz.clone();
            tool.cone_tip_radius_mm = self.cone_tip_radius_mm;
            tool.cone_half_angle_deg = self.cone_half_angle_deg;
            tool.cone_height_mm = self.cone_height_mm;
            tool.rect_tip_half_width_x_mm = self.rect_tip_half_width_x_mm;
            tool.rect_tip_half_width_y_mm = self.rect_tip_half_width_y_mm;
            tool.rect_half_angle_x_deg = self.rect_half_angle_x_deg;
            tool.rect_half_angle_y_deg = self.rect_half_angle_y_deg;
            tool.rect_height_mm = self.rect_height_mm;
        }
        cfg.gcode.enable_flat_nozzle_comp = self.enable_flat_nozzle_comp;
        cfg.gcode.max_volumetric_flow_mm3s = self.max_volumetric_flow_mm3s;
        cfg.gcode.retract_length_mm = self.retract_length_mm;
        cfg.gcode.retract_speed_mms = self.retract_speed_mms;
        cfg.gcode.min_travel_for_retract_mm = self.min_travel_for_retract_mm;
        cfg.gcode.z_hop_mm = self.z_hop_mm;
        cfg.gcode.wipe_length_mm = self.wipe_length_mm;
        cfg.gcode.pre_retract_taper_mm = self.pre_retract_taper_mm;
        cfg.gcode.tool_change_time_s = self.tool_change_time_s;
        cfg.gcode.tool_change_hop_before = self.tool_change_hop_before;
        cfg.gcode.tool_change_approach_after = self.tool_change_approach_after;
        cfg.sync_auto_voxel_size();
        cfg.gcode.sanitize();
    }

    pub fn matches_config(&self, cfg: &SlicerConfig) -> bool {
        let Some(tool) = cfg.active_tool() else {
            return false;
        };
        if self.clearance_mode != tool.clearance_mode {
            return false;
        }
        let clearance_matches = match self.clearance_mode {
            ClearanceProfileMode::PiecewiseXZ => {
                self.clearance_profile_xz.len() == tool.clearance_profile_xz.len()
                    && self
                        .clearance_profile_xz
                        .iter()
                        .zip(tool.clearance_profile_xz.iter())
                        .all(|(a, b)| {
                            approx_eq(a.x_radius_mm, b.x_radius_mm)
                                && approx_eq(a.z_height_mm, b.z_height_mm)
                        })
            }
            ClearanceProfileMode::ParametricCone => {
                approx_eq(self.cone_tip_radius_mm, tool.cone_tip_radius_mm)
                    && approx_eq(self.cone_half_angle_deg, tool.cone_half_angle_deg)
                    && approx_eq(self.cone_height_mm, tool.cone_height_mm)
            }
            ClearanceProfileMode::RectilinearCone => {
                approx_eq(self.rect_tip_half_width_x_mm, tool.rect_tip_half_width_x_mm)
                    && approx_eq(self.rect_tip_half_width_y_mm, tool.rect_tip_half_width_y_mm)
                    && approx_eq(self.rect_half_angle_x_deg, tool.rect_half_angle_x_deg)
                    && approx_eq(self.rect_half_angle_y_deg, tool.rect_half_angle_y_deg)
                    && approx_eq(self.rect_height_mm, tool.rect_height_mm)
            }
        };

        clearance_matches
            && approx_eq(self.nozzle_diameter_mm, tool.nozzle_diameter_mm)
            && self.has_rotational_axes == tool.has_rotational_axes
            && approx_eq(
                self.max_angle_from_bed_normal_deg,
                tool.max_angle_from_bed_normal_deg,
            )
            && self.enable_flat_nozzle_comp == cfg.gcode.enable_flat_nozzle_comp
            && approx_eq(
                self.max_volumetric_flow_mm3s,
                cfg.gcode.max_volumetric_flow_mm3s,
            )
            && approx_eq(self.retract_length_mm, cfg.gcode.retract_length_mm)
            && approx_eq(self.retract_speed_mms, cfg.gcode.retract_speed_mms)
            && approx_eq(
                self.min_travel_for_retract_mm,
                cfg.gcode.min_travel_for_retract_mm,
            )
            && approx_eq(self.z_hop_mm, cfg.gcode.z_hop_mm)
            && approx_eq(self.wipe_length_mm, cfg.gcode.wipe_length_mm)
            && approx_eq(self.pre_retract_taper_mm, cfg.gcode.pre_retract_taper_mm)
            && approx_eq(self.tool_change_time_s, cfg.gcode.tool_change_time_s)
            && self.tool_change_hop_before == cfg.gcode.tool_change_hop_before
            && self.tool_change_approach_after == cfg.gcode.tool_change_approach_after
    }
}

// ============================================================================
// 3. FilamentPreset — Material, Thermal, Cooling & Non-Newtonian Rheology
// ============================================================================

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FilamentPreset {
    pub name: String,
    #[serde(skip)]
    pub is_builtin: bool,

    // Material Identity & Physical Feedstock Properties
    pub filament_type: String,
    pub filament_diameter_mm: f32,
    pub filament_density_g_cm3: f32,
    /// Optional explicit filament colour (`#RRGGBB`); empty on colour-neutral generic material presets.
    pub filament_color_hex: String,
    /// Optional explicit Kubelka-Munk filament Transmission Distance (`TD` in mm, `None` = Auto TD).
    pub transmission_distance_mm: Option<f32>,

    // Temperatures
    pub nozzle_temperature_celsius: f32,
    pub bed_temperature_celsius: f32,

    // Flow Multipliers
    pub extrusion_multiplier: f32,
    pub first_layer_flow_multiplier: f32,
    pub bridge_flow_multiplier: f32,
    pub wave_overhang_flow_multiplier: f32,

    // Part Cooling Fan Schedule
    pub fan_speed_percent: f32,
    pub first_layer_fan_speed_percent: f32,
    pub disable_fan_first_layers: u32,
    pub bridge_fan_speed_percent: f32,
    pub wave_overhang_fan_speed_percent: f32,
    pub enable_dynamic_fan_speed: bool,
    pub min_fan_speed_percent: f32,
    pub max_fan_speed_percent: f32,
    pub target_layer_cooling_time_s: f32,

    // Non-Newtonian Rheology, Pressure Advance, Die Swell & Ooze Relaxation
    pub pa_low_flow_s: f32,
    pub pa_high_flow_s: f32,
    pub pa_ref_low_flow_mm3s: f32,
    pub pa_ref_high_flow_mm3s: f32,
    pub swell_low: f32,
    pub swell_high: f32,
    pub enable_transient_pressure_comp: bool,
    pub transient_pressure_exponent: f32,
    pub transient_min_flow_multiplier: f32,
    pub transient_pressure_tau_s: f32,
    pub ooze_relaxation_extra_prime_mm: f32,
    pub ooze_relaxation_tau_s: f32,
}

impl Default for FilamentPreset {
    fn default() -> Self {
        Self::from_config("Generic PLA", &SlicerConfig::default(), false)
    }
}

impl FilamentPreset {
    pub fn from_config(name: impl Into<String>, cfg: &SlicerConfig, is_builtin: bool) -> Self {
        let active_tool = cfg.active_tool();
        let nozzle_temp = active_tool.map(|t| t.temperature_celsius).unwrap_or(215.0);
        let filament_color_hex = if is_builtin {
            String::new()
        } else {
            active_tool
                .map(|t| t.filament_color_hex.clone())
                .unwrap_or_default()
        };
        let transmission_distance_mm = if is_builtin {
            None
        } else {
            active_tool.and_then(|t| t.transmission_distance_mm)
        };
        Self {
            name: name.into(),
            is_builtin,
            filament_type: cfg.gcode.filament_type.clone(),
            filament_diameter_mm: cfg.gcode.filament_diameter_mm,
            filament_density_g_cm3: cfg.gcode.filament_density_g_cm3,
            filament_color_hex,
            transmission_distance_mm,
            nozzle_temperature_celsius: nozzle_temp,
            bed_temperature_celsius: cfg.gcode.bed_temperature_celsius,
            extrusion_multiplier: cfg.gcode.extrusion_multiplier,
            first_layer_flow_multiplier: cfg.gcode.first_layer_flow_multiplier,
            bridge_flow_multiplier: cfg.gcode.bridge_flow_multiplier,
            wave_overhang_flow_multiplier: cfg.gcode.wave_overhang_flow_multiplier,
            fan_speed_percent: cfg.gcode.fan_speed_percent,
            first_layer_fan_speed_percent: cfg.gcode.first_layer_fan_speed_percent,
            disable_fan_first_layers: cfg.gcode.disable_fan_first_layers,
            bridge_fan_speed_percent: cfg.gcode.bridge_fan_speed_percent,
            wave_overhang_fan_speed_percent: cfg.gcode.wave_overhang_fan_speed_percent,
            enable_dynamic_fan_speed: cfg.gcode.enable_dynamic_fan_speed,
            min_fan_speed_percent: cfg.gcode.min_fan_speed_percent,
            max_fan_speed_percent: cfg.gcode.max_fan_speed_percent,
            target_layer_cooling_time_s: cfg.gcode.target_layer_cooling_time_s,
            pa_low_flow_s: cfg.gcode.pa_low_flow_s,
            pa_high_flow_s: cfg.gcode.pa_high_flow_s,
            pa_ref_low_flow_mm3s: cfg.gcode.pa_ref_low_flow_mm3s,
            pa_ref_high_flow_mm3s: cfg.gcode.pa_ref_high_flow_mm3s,
            swell_low: cfg.gcode.swell_low,
            swell_high: cfg.gcode.swell_high,
            enable_transient_pressure_comp: cfg.gcode.enable_transient_pressure_comp,
            transient_pressure_exponent: cfg.gcode.transient_pressure_exponent,
            transient_min_flow_multiplier: cfg.gcode.transient_min_flow_multiplier,
            transient_pressure_tau_s: cfg.gcode.transient_pressure_tau_s,
            ooze_relaxation_extra_prime_mm: cfg.gcode.ooze_relaxation_extra_prime_mm,
            ooze_relaxation_tau_s: cfg.gcode.ooze_relaxation_tau_s,
        }
    }

    pub fn apply_to_tool(&self, tool: &mut ToolConfig) {
        tool.filament_preset_name = self.name.clone();
        tool.filament_type = self.filament_type.clone();
        if !self.filament_color_hex.trim().is_empty() {
            tool.set_color_from_encoding(&self.filament_color_hex);
        }
        tool.transmission_distance_mm = self.transmission_distance_mm;
        tool.temperature_celsius = self.nozzle_temperature_celsius;
        tool.filament_density_g_cm3 = self.filament_density_g_cm3;
        tool.extrusion_multiplier = self.extrusion_multiplier;
        tool.fan_speed_percent = self.fan_speed_percent;
        tool.min_fan_speed_percent = self.min_fan_speed_percent;
        tool.max_fan_speed_percent = self.max_fan_speed_percent;
        tool.bridge_fan_speed_percent = self.bridge_fan_speed_percent;
        tool.wave_overhang_fan_speed_percent = self.wave_overhang_fan_speed_percent;
        tool.pa_low_flow_s = self.pa_low_flow_s;
        tool.pa_high_flow_s = self.pa_high_flow_s;
        tool.swell_low = self.swell_low;
        tool.swell_high = self.swell_high;
    }

    pub fn apply_to_config(&self, cfg: &mut SlicerConfig) {
        cfg.filament_preset_name = self.name.clone();
        if let Some(tool) = cfg.tools.get_mut(cfg.active_tool_index) {
            self.apply_to_tool(tool);
        } else if let Some(tool) = cfg.tools.first_mut() {
            self.apply_to_tool(tool);
        }
        cfg.gcode.filament_type = self.filament_type.clone();
        cfg.gcode.filament_diameter_mm = self.filament_diameter_mm;
        cfg.gcode.filament_density_g_cm3 = self.filament_density_g_cm3;
        cfg.gcode.bed_temperature_celsius = self.bed_temperature_celsius;
        cfg.gcode.extrusion_multiplier = self.extrusion_multiplier;
        cfg.gcode.first_layer_flow_multiplier = self.first_layer_flow_multiplier;
        cfg.gcode.bridge_flow_multiplier = self.bridge_flow_multiplier;
        cfg.gcode.wave_overhang_flow_multiplier = self.wave_overhang_flow_multiplier;
        cfg.gcode.fan_speed_percent = self.fan_speed_percent;
        cfg.gcode.first_layer_fan_speed_percent = self.first_layer_fan_speed_percent;
        cfg.gcode.disable_fan_first_layers = self.disable_fan_first_layers;
        cfg.gcode.bridge_fan_speed_percent = self.bridge_fan_speed_percent;
        cfg.gcode.wave_overhang_fan_speed_percent = self.wave_overhang_fan_speed_percent;
        cfg.gcode.enable_dynamic_fan_speed = self.enable_dynamic_fan_speed;
        cfg.gcode.min_fan_speed_percent = self.min_fan_speed_percent;
        cfg.gcode.max_fan_speed_percent = self.max_fan_speed_percent;
        cfg.gcode.target_layer_cooling_time_s = self.target_layer_cooling_time_s;
        cfg.gcode.pa_low_flow_s = self.pa_low_flow_s;
        cfg.gcode.pa_high_flow_s = self.pa_high_flow_s;
        cfg.gcode.pa_ref_low_flow_mm3s = self.pa_ref_low_flow_mm3s;
        cfg.gcode.pa_ref_high_flow_mm3s = self.pa_ref_high_flow_mm3s;
        cfg.gcode.swell_low = self.swell_low;
        cfg.gcode.swell_high = self.swell_high;
        cfg.gcode.enable_transient_pressure_comp = self.enable_transient_pressure_comp;
        cfg.gcode.transient_pressure_exponent = self.transient_pressure_exponent;
        cfg.gcode.transient_min_flow_multiplier = self.transient_min_flow_multiplier;
        cfg.gcode.transient_pressure_tau_s = self.transient_pressure_tau_s;
        cfg.gcode.ooze_relaxation_extra_prime_mm = self.ooze_relaxation_extra_prime_mm;
        cfg.gcode.ooze_relaxation_tau_s = self.ooze_relaxation_tau_s;
        cfg.gcode.sanitize();
    }

    pub fn matches_config(&self, cfg: &SlicerConfig) -> bool {
        let active_tool = cfg.active_tool();
        let nozzle_temp = active_tool.map(|t| t.temperature_celsius).unwrap_or(215.0);
        let color_ok = if self.filament_color_hex.trim().is_empty() {
            true
        } else {
            active_tool
                .map(|t| t.filament_color_hex.eq_ignore_ascii_case(self.filament_color_hex.trim()))
                .unwrap_or(false)
        };
        let td_ok = match (
            self.transmission_distance_mm,
            active_tool.and_then(|t| t.transmission_distance_mm),
        ) {
            (None, None) => true,
            (Some(a), Some(b)) => approx_eq(a, b),
            _ => false,
        };
        color_ok
            && td_ok
            && self.filament_type == cfg.gcode.filament_type
            && approx_eq(self.filament_diameter_mm, cfg.gcode.filament_diameter_mm)
            && approx_eq(self.filament_density_g_cm3, cfg.gcode.filament_density_g_cm3)
            && approx_eq(self.nozzle_temperature_celsius, nozzle_temp)
            && approx_eq(self.bed_temperature_celsius, cfg.gcode.bed_temperature_celsius)
            && approx_eq(self.extrusion_multiplier, cfg.gcode.extrusion_multiplier)
            && approx_eq(
                self.first_layer_flow_multiplier,
                cfg.gcode.first_layer_flow_multiplier,
            )
            && approx_eq(self.bridge_flow_multiplier, cfg.gcode.bridge_flow_multiplier)
            && approx_eq(
                self.wave_overhang_flow_multiplier,
                cfg.gcode.wave_overhang_flow_multiplier,
            )
            && approx_eq(self.fan_speed_percent, cfg.gcode.fan_speed_percent)
            && approx_eq(
                self.first_layer_fan_speed_percent,
                cfg.gcode.first_layer_fan_speed_percent,
            )
            && self.disable_fan_first_layers == cfg.gcode.disable_fan_first_layers
            && approx_eq(
                self.bridge_fan_speed_percent,
                cfg.gcode.bridge_fan_speed_percent,
            )
            && approx_eq(
                self.wave_overhang_fan_speed_percent,
                cfg.gcode.wave_overhang_fan_speed_percent,
            )
            && self.enable_dynamic_fan_speed == cfg.gcode.enable_dynamic_fan_speed
            && approx_eq(self.min_fan_speed_percent, cfg.gcode.min_fan_speed_percent)
            && approx_eq(self.max_fan_speed_percent, cfg.gcode.max_fan_speed_percent)
            && approx_eq(
                self.target_layer_cooling_time_s,
                cfg.gcode.target_layer_cooling_time_s,
            )
            && approx_eq(self.pa_low_flow_s, cfg.gcode.pa_low_flow_s)
            && approx_eq(self.pa_high_flow_s, cfg.gcode.pa_high_flow_s)
            && approx_eq(self.pa_ref_low_flow_mm3s, cfg.gcode.pa_ref_low_flow_mm3s)
            && approx_eq(self.pa_ref_high_flow_mm3s, cfg.gcode.pa_ref_high_flow_mm3s)
            && approx_eq(self.swell_low, cfg.gcode.swell_low)
            && approx_eq(self.swell_high, cfg.gcode.swell_high)
            && self.enable_transient_pressure_comp == cfg.gcode.enable_transient_pressure_comp
            && approx_eq(
                self.transient_pressure_exponent,
                cfg.gcode.transient_pressure_exponent,
            )
            && approx_eq(
                self.transient_min_flow_multiplier,
                cfg.gcode.transient_min_flow_multiplier,
            )
            && approx_eq(
                self.transient_pressure_tau_s,
                cfg.gcode.transient_pressure_tau_s,
            )
            && approx_eq(
                self.ooze_relaxation_extra_prime_mm,
                cfg.gcode.ooze_relaxation_extra_prime_mm,
            )
            && approx_eq(self.ooze_relaxation_tau_s, cfg.gcode.ooze_relaxation_tau_s)
    }
}

// ============================================================================
// 4. ProcessPreset — Eikonal Solver, Layer Height, Shells, Infill, Supports, Seams & Speeds
// ============================================================================

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ProcessPreset {
    pub name: String,
    #[serde(skip)]
    pub is_builtin: bool,

    // Volumetric Eikonal Solver & Overhang Surface Bias
    pub order_field_mode: OrderFieldMode,
    pub auto_voxel_size: bool,
    pub voxel_size_mm: f32,
    pub enable_multi_resolution_grid: bool,
    pub coarse_grid_ratio: u32,
    pub narrow_band_width_voxels: u32,
    pub curvature_regularization: f32,
    pub solver_iterations: u32,
    pub default_surface_hint_depth_mm: f32,
    pub default_point_hint_radius_mm: f32,
    pub default_hint_weight: f32,
    pub enable_overhang_orientation_bias: bool,
    pub overhang_bias_weight: f32,
    pub orthogonal_supported_angle_deg: f32,
    pub orthogonal_overhang_angle_deg: f32,
    pub overhang_bias_transition_deg: f32,

    // Variable Layer Height
    pub layer_height: LayerHeightConfig,
    pub max_nonplanar_slope_deg: f32,

    // Shells, Line Widths & Infill
    pub shell_thickness_mm: f32,
    pub inner_wall_count: u32,
    #[serde(default)]
    pub brick_layer_mode: BrickLayerMode,
    pub one_perimeter_on_conformal_patches: bool,
    pub enable_top_surface_lofting: bool,
    pub orient_outer_surfaces_to_normal: bool,
    pub outer_surface_parallel_angle_deg: f32,
    pub wall_print_order: WallPrintOrder,
    pub outer_wall_width: LineWidthConfig,
    pub inner_wall_width: LineWidthConfig,
    pub solid_infill_width: LineWidthConfig,
    pub solid_infill_pattern: SolidInfillPattern,
    pub sparse_infill_width: LineWidthConfig,
    pub sparse_infill_pattern: SparseInfillPattern,
    #[serde(default)]
    pub infill_filament_mode: crate::config::InfillFilamentMode,
    pub infill_density_percent: f32,
    pub infill_angle_deg: f32,
    pub min_extrusion_length_mm: f32,

    // Overhangs (Bridging & Wave Overhangs)
    pub overhang_angle_threshold_deg: f32,
    pub max_bridge_length_mm: f32,
    pub wave_overhang_overlap_fraction: f32,

    // Supports (Unreachable Order-Field Regions & Long Bridges)
    pub enable_support_generation: bool,
    pub support_tool_index: Option<usize>,
    #[serde(default)]
    pub support_interface_tool_index: Option<usize>,
    pub support_style: SupportStyle,
    pub support_on_bed_only: bool,
    pub seed_order_field_from_supports: bool,
    pub support_only_unreachable_and_bridges: bool,
    pub support_z_gap_mm: f32,
    pub support_xy_gap_mm: f32,
    pub support_interface_layers: u32,
    pub support_bed_interface_layers: u32,
    pub support_base_spacing_mm: f32,
    pub support_interface_spacing_mm: f32,
    pub support_line_width_mm: f32,
    pub tree_branch_angle_deg: f32,
    pub tree_tip_diameter_mm: f32,
    pub tree_trunk_max_diameter_mm: f32,
    pub tree_branch_wall_count: u32,

    // 3D Conformal Scarf Seam & First-Layer Brim
    pub enable_scarf_joint: bool,
    pub scarf_length_mm: f32,
    pub scarf_start_height_ratio: f32,
    pub seam_gap_mm: f32,
    pub enable_auto_brim: bool,
    pub brim_width_mm: f32,
    pub brim_gap_mm: f32,

    // Feature Print Speeds (mm/s) & Lever-Arm / Curled-Corner Speed Governor
    pub first_layer_speed_mms: f32,
    pub outer_wall_speed_mms: f32,
    pub inner_wall_speed_mms: f32,
    pub solid_infill_speed_mms: f32,
    pub sparse_infill_speed_mms: f32,
    pub gap_fill_speed_mms: f32,
    pub bridge_speed_mms: f32,
    pub wave_overhang_speed_mms: f32,
    pub support_speed_mms: f32,
    pub support_interface_speed_mms: f32,
    pub enable_lever_arm_slowdown: bool,
    pub lever_arm_reference_ratio: f32,
    pub lever_arm_min_speed_fraction: f32,

    // Multi-Filament CMYKW & Arbitrary-Filament Optical Colour Mixing
    pub color_mixing: crate::color_optics::ColorMixingConfig,
}

impl Default for ProcessPreset {
    fn default() -> Self {
        Self::from_config(
            "0.20mm Standard Non-Planar",
            &SlicerConfig::default(),
            false,
        )
    }
}

impl ProcessPreset {
    pub fn from_config(name: impl Into<String>, cfg: &SlicerConfig, is_builtin: bool) -> Self {
        Self {
            name: name.into(),
            is_builtin,
            order_field_mode: cfg.eikonal.order_field_mode,
            auto_voxel_size: cfg.eikonal.auto_voxel_size,
            voxel_size_mm: cfg.eikonal.voxel_size_mm,
            enable_multi_resolution_grid: cfg.eikonal.enable_multi_resolution_grid,
            coarse_grid_ratio: cfg.eikonal.coarse_grid_ratio,
            narrow_band_width_voxels: cfg.eikonal.narrow_band_width_voxels,
            curvature_regularization: cfg.eikonal.curvature_regularization,
            solver_iterations: cfg.eikonal.solver_iterations,
            default_surface_hint_depth_mm: cfg.eikonal.default_surface_hint_depth_mm,
            default_point_hint_radius_mm: cfg.eikonal.default_point_hint_radius_mm,
            default_hint_weight: cfg.eikonal.default_hint_weight,
            enable_overhang_orientation_bias: cfg.eikonal.enable_overhang_orientation_bias,
            overhang_bias_weight: cfg.eikonal.overhang_bias_weight,
            orthogonal_supported_angle_deg: cfg.eikonal.orthogonal_supported_angle_deg,
            orthogonal_overhang_angle_deg: cfg.eikonal.orthogonal_overhang_angle_deg,
            overhang_bias_transition_deg: cfg.eikonal.overhang_bias_transition_deg,
            layer_height: cfg.slicing.layer_height,
            max_nonplanar_slope_deg: cfg.slicing.max_nonplanar_slope_deg,
            shell_thickness_mm: cfg.slicing.shell_thickness_mm,
            inner_wall_count: cfg.slicing.inner_wall_count,
            brick_layer_mode: cfg.slicing.brick_layer_mode,
            one_perimeter_on_conformal_patches: cfg.slicing.one_perimeter_on_conformal_patches,
            enable_top_surface_lofting: cfg.slicing.enable_top_surface_lofting,
            orient_outer_surfaces_to_normal: cfg.slicing.orient_outer_surfaces_to_normal,
            outer_surface_parallel_angle_deg: cfg.slicing.outer_surface_parallel_angle_deg,
            wall_print_order: cfg.gcode.wall_print_order,
            outer_wall_width: cfg.slicing.outer_wall_width,
            inner_wall_width: cfg.slicing.inner_wall_width,
            solid_infill_width: cfg.slicing.solid_infill_width,
            solid_infill_pattern: cfg.slicing.solid_infill_pattern,
            sparse_infill_width: cfg.slicing.sparse_infill_width,
            sparse_infill_pattern: cfg.slicing.sparse_infill_pattern,
            infill_filament_mode: cfg.slicing.infill_filament_mode,
            infill_density_percent: cfg.slicing.infill_density_percent,
            infill_angle_deg: cfg.slicing.infill_angle_deg,
            min_extrusion_length_mm: cfg.slicing.min_extrusion_length_mm,
            overhang_angle_threshold_deg: cfg.slicing.overhang_angle_threshold_deg,
            max_bridge_length_mm: cfg.slicing.max_bridge_length_mm,
            wave_overhang_overlap_fraction: cfg.slicing.wave_overhang_overlap_fraction,
            enable_support_generation: cfg.slicing.enable_support_generation,
            support_tool_index: cfg.slicing.support_tool_index,
            support_interface_tool_index: cfg.slicing.support_interface_tool_index,
            support_style: cfg.slicing.support_style,
            support_on_bed_only: cfg.slicing.support_on_bed_only,
            seed_order_field_from_supports: cfg.slicing.seed_order_field_from_supports,
            support_only_unreachable_and_bridges: cfg.slicing.support_only_unreachable_and_bridges,
            support_z_gap_mm: cfg.slicing.support_z_gap_mm,
            support_xy_gap_mm: cfg.slicing.support_xy_gap_mm,
            support_interface_layers: cfg.slicing.support_interface_layers,
            support_bed_interface_layers: cfg.slicing.support_bed_interface_layers,
            support_base_spacing_mm: cfg.slicing.support_base_spacing_mm,
            support_interface_spacing_mm: cfg.slicing.support_interface_spacing_mm,
            support_line_width_mm: cfg.slicing.support_line_width_mm,
            tree_branch_angle_deg: cfg.slicing.tree_branch_angle_deg,
            tree_tip_diameter_mm: cfg.slicing.tree_tip_diameter_mm,
            tree_trunk_max_diameter_mm: cfg.slicing.tree_trunk_max_diameter_mm,
            tree_branch_wall_count: cfg.slicing.tree_branch_wall_count,
            enable_scarf_joint: cfg.gcode.enable_scarf_joint,
            scarf_length_mm: cfg.gcode.scarf_length_mm,
            scarf_start_height_ratio: cfg.gcode.scarf_start_height_ratio,
            seam_gap_mm: cfg.gcode.seam_gap_mm,
            enable_auto_brim: cfg.gcode.enable_auto_brim,
            brim_width_mm: cfg.gcode.brim_width_mm,
            brim_gap_mm: cfg.gcode.brim_gap_mm,
            first_layer_speed_mms: cfg.gcode.first_layer_speed_mms,
            outer_wall_speed_mms: cfg.gcode.outer_wall_speed_mms,
            inner_wall_speed_mms: cfg.gcode.inner_wall_speed_mms,
            solid_infill_speed_mms: cfg.gcode.solid_infill_speed_mms,
            sparse_infill_speed_mms: cfg.gcode.sparse_infill_speed_mms,
            gap_fill_speed_mms: cfg.gcode.gap_fill_speed_mms,
            bridge_speed_mms: cfg.gcode.bridge_speed_mms,
            wave_overhang_speed_mms: cfg.gcode.wave_overhang_speed_mms,
            support_speed_mms: cfg.gcode.support_speed_mms,
            support_interface_speed_mms: cfg.gcode.support_interface_speed_mms,
            enable_lever_arm_slowdown: cfg.gcode.enable_lever_arm_slowdown,
            lever_arm_reference_ratio: cfg.gcode.lever_arm_reference_ratio,
            lever_arm_min_speed_fraction: cfg.gcode.lever_arm_min_speed_fraction,
            color_mixing: cfg.slicing.color_mixing.clone(),
        }
    }

    pub fn apply_to_config(&self, cfg: &mut SlicerConfig) {
        cfg.process_preset_name = self.name.clone();
        cfg.eikonal.order_field_mode = self.order_field_mode;
        cfg.eikonal.auto_voxel_size = self.auto_voxel_size;
        cfg.eikonal.voxel_size_mm = self.voxel_size_mm;
        cfg.eikonal.enable_multi_resolution_grid = self.enable_multi_resolution_grid;
        cfg.eikonal.coarse_grid_ratio = self.coarse_grid_ratio;
        cfg.eikonal.narrow_band_width_voxels = self.narrow_band_width_voxels;
        cfg.eikonal.curvature_regularization = self.curvature_regularization;
        cfg.eikonal.solver_iterations = self.solver_iterations;
        cfg.eikonal.default_surface_hint_depth_mm = self.default_surface_hint_depth_mm;
        cfg.eikonal.default_point_hint_radius_mm = self.default_point_hint_radius_mm;
        cfg.eikonal.default_hint_weight = self.default_hint_weight;
        cfg.eikonal.enable_overhang_orientation_bias = self.enable_overhang_orientation_bias;
        cfg.eikonal.overhang_bias_weight = self.overhang_bias_weight;
        cfg.eikonal.orthogonal_supported_angle_deg = self.orthogonal_supported_angle_deg;
        cfg.eikonal.orthogonal_overhang_angle_deg = self.orthogonal_overhang_angle_deg;
        cfg.eikonal.overhang_bias_transition_deg = self.overhang_bias_transition_deg;

        cfg.slicing.layer_height = self.layer_height;
        cfg.slicing.max_nonplanar_slope_deg = self.max_nonplanar_slope_deg;
        cfg.slicing.shell_thickness_mm = self.shell_thickness_mm;
        cfg.slicing.inner_wall_count = self.inner_wall_count;
        cfg.slicing.brick_layer_mode = self.brick_layer_mode;
        cfg.slicing.one_perimeter_on_conformal_patches = self.one_perimeter_on_conformal_patches;
        cfg.slicing.enable_top_surface_lofting = self.enable_top_surface_lofting;
        cfg.slicing.orient_outer_surfaces_to_normal = self.orient_outer_surfaces_to_normal;
        cfg.slicing.outer_surface_parallel_angle_deg = self.outer_surface_parallel_angle_deg;
        cfg.gcode.wall_print_order = self.wall_print_order;
        cfg.slicing.outer_wall_width = self.outer_wall_width;
        cfg.slicing.inner_wall_width = self.inner_wall_width;
        cfg.slicing.solid_infill_width = self.solid_infill_width;
        cfg.slicing.solid_infill_pattern = self.solid_infill_pattern;
        cfg.slicing.sparse_infill_width = self.sparse_infill_width;
        cfg.slicing.sparse_infill_pattern = self.sparse_infill_pattern;
        cfg.slicing.infill_filament_mode = self.infill_filament_mode;
        cfg.slicing.infill_density_percent = self.infill_density_percent;
        cfg.slicing.infill_angle_deg = self.infill_angle_deg;
        cfg.slicing.min_extrusion_length_mm = self.min_extrusion_length_mm;
        cfg.slicing.overhang_angle_threshold_deg = self.overhang_angle_threshold_deg;
        cfg.slicing.max_bridge_length_mm = self.max_bridge_length_mm;
        cfg.slicing.wave_overhang_overlap_fraction = self.wave_overhang_overlap_fraction;
        cfg.slicing.enable_support_generation = self.enable_support_generation;
        cfg.slicing.support_tool_index = self.support_tool_index;
        cfg.slicing.support_interface_tool_index = self.support_interface_tool_index;
        cfg.slicing.support_style = self.support_style;
        cfg.slicing.support_on_bed_only = self.support_on_bed_only;
        cfg.slicing.seed_order_field_from_supports = self.seed_order_field_from_supports;
        cfg.slicing.support_only_unreachable_and_bridges =
            self.support_only_unreachable_and_bridges;
        cfg.slicing.support_z_gap_mm = self.support_z_gap_mm;
        cfg.slicing.support_xy_gap_mm = self.support_xy_gap_mm;
        cfg.slicing.support_interface_layers = self.support_interface_layers;
        cfg.slicing.support_bed_interface_layers = self.support_bed_interface_layers;
        cfg.slicing.support_base_spacing_mm = self.support_base_spacing_mm;
        cfg.slicing.support_interface_spacing_mm = self.support_interface_spacing_mm;
        cfg.slicing.support_line_width_mm = self.support_line_width_mm;
        cfg.slicing.tree_branch_angle_deg = self.tree_branch_angle_deg;
        cfg.slicing.tree_tip_diameter_mm = self.tree_tip_diameter_mm;
        cfg.slicing.tree_trunk_max_diameter_mm = self.tree_trunk_max_diameter_mm;
        cfg.slicing.tree_branch_wall_count = self.tree_branch_wall_count;
        cfg.slicing.color_mixing = self.color_mixing.clone();

        cfg.gcode.enable_scarf_joint = self.enable_scarf_joint;
        cfg.gcode.scarf_length_mm = self.scarf_length_mm;
        cfg.gcode.scarf_start_height_ratio = self.scarf_start_height_ratio;
        cfg.gcode.seam_gap_mm = self.seam_gap_mm;
        cfg.gcode.enable_auto_brim = self.enable_auto_brim;
        cfg.gcode.brim_width_mm = self.brim_width_mm;
        cfg.gcode.brim_gap_mm = self.brim_gap_mm;
        cfg.gcode.first_layer_speed_mms = self.first_layer_speed_mms;
        cfg.gcode.outer_wall_speed_mms = self.outer_wall_speed_mms;
        cfg.gcode.inner_wall_speed_mms = self.inner_wall_speed_mms;
        cfg.gcode.solid_infill_speed_mms = self.solid_infill_speed_mms;
        cfg.gcode.sparse_infill_speed_mms = self.sparse_infill_speed_mms;
        cfg.gcode.gap_fill_speed_mms = self.gap_fill_speed_mms;
        cfg.gcode.bridge_speed_mms = self.bridge_speed_mms;
        cfg.gcode.wave_overhang_speed_mms = self.wave_overhang_speed_mms;
        cfg.gcode.support_speed_mms = self.support_speed_mms;
        cfg.gcode.support_interface_speed_mms = self.support_interface_speed_mms;
        cfg.gcode.enable_lever_arm_slowdown = self.enable_lever_arm_slowdown;
        cfg.gcode.lever_arm_reference_ratio = self.lever_arm_reference_ratio;
        cfg.gcode.lever_arm_min_speed_fraction = self.lever_arm_min_speed_fraction;

        cfg.eikonal.sanitize();
        cfg.slicing.sanitize();
        cfg.gcode.sanitize();
        cfg.sync_auto_voxel_size();
    }

    pub fn matches_config(&self, cfg: &SlicerConfig) -> bool {
        let voxel_ok = if self.auto_voxel_size && cfg.eikonal.auto_voxel_size {
            true
        } else {
            approx_eq(self.voxel_size_mm, cfg.eikonal.voxel_size_mm)
        };
        self.order_field_mode == cfg.eikonal.order_field_mode
            && self.auto_voxel_size == cfg.eikonal.auto_voxel_size
            && voxel_ok
            && self.enable_multi_resolution_grid == cfg.eikonal.enable_multi_resolution_grid
            && self.coarse_grid_ratio == cfg.eikonal.coarse_grid_ratio
            && self.narrow_band_width_voxels == cfg.eikonal.narrow_band_width_voxels
            && approx_eq(
                self.curvature_regularization,
                cfg.eikonal.curvature_regularization,
            )
            && self.solver_iterations == cfg.eikonal.solver_iterations
            && approx_eq(
                self.default_surface_hint_depth_mm,
                cfg.eikonal.default_surface_hint_depth_mm,
            )
            && approx_eq(
                self.default_point_hint_radius_mm,
                cfg.eikonal.default_point_hint_radius_mm,
            )
            && approx_eq(self.default_hint_weight, cfg.eikonal.default_hint_weight)
            && self.enable_overhang_orientation_bias
                == cfg.eikonal.enable_overhang_orientation_bias
            && approx_eq(self.overhang_bias_weight, cfg.eikonal.overhang_bias_weight)
            && approx_eq(
                self.orthogonal_supported_angle_deg,
                cfg.eikonal.orthogonal_supported_angle_deg,
            )
            && approx_eq(
                self.orthogonal_overhang_angle_deg,
                cfg.eikonal.orthogonal_overhang_angle_deg,
            )
            && approx_eq(
                self.overhang_bias_transition_deg,
                cfg.eikonal.overhang_bias_transition_deg,
            )
            && approx_eq_layer_height(self.layer_height, cfg.slicing.layer_height)
            && approx_eq(
                self.max_nonplanar_slope_deg,
                cfg.slicing.max_nonplanar_slope_deg,
            )
            && approx_eq(self.shell_thickness_mm, cfg.slicing.shell_thickness_mm)
            && self.inner_wall_count == cfg.slicing.inner_wall_count
            && self.brick_layer_mode == cfg.slicing.brick_layer_mode
            && self.one_perimeter_on_conformal_patches
                == cfg.slicing.one_perimeter_on_conformal_patches
            && self.enable_top_surface_lofting == cfg.slicing.enable_top_surface_lofting
            && self.orient_outer_surfaces_to_normal
                == cfg.slicing.orient_outer_surfaces_to_normal
            && approx_eq(
                self.outer_surface_parallel_angle_deg,
                cfg.slicing.outer_surface_parallel_angle_deg,
            )
            && self.wall_print_order == cfg.gcode.wall_print_order
            && approx_eq_line_width(self.outer_wall_width, cfg.slicing.outer_wall_width)
            && approx_eq_line_width(self.inner_wall_width, cfg.slicing.inner_wall_width)
            && approx_eq_line_width(self.solid_infill_width, cfg.slicing.solid_infill_width)
            && self.solid_infill_pattern == cfg.slicing.solid_infill_pattern
            && approx_eq_line_width(self.sparse_infill_width, cfg.slicing.sparse_infill_width)
            && self.sparse_infill_pattern == cfg.slicing.sparse_infill_pattern
            && self.infill_filament_mode == cfg.slicing.infill_filament_mode
            && approx_eq(
                self.infill_density_percent,
                cfg.slicing.infill_density_percent,
            )
            && approx_eq(self.infill_angle_deg, cfg.slicing.infill_angle_deg)
            && approx_eq(
                self.min_extrusion_length_mm,
                cfg.slicing.min_extrusion_length_mm,
            )
            && approx_eq(
                self.overhang_angle_threshold_deg,
                cfg.slicing.overhang_angle_threshold_deg,
            )
            && approx_eq(self.max_bridge_length_mm, cfg.slicing.max_bridge_length_mm)
            && approx_eq(
                self.wave_overhang_overlap_fraction,
                cfg.slicing.wave_overhang_overlap_fraction,
            )
            && self.enable_support_generation == cfg.slicing.enable_support_generation
            && self.support_tool_index == cfg.slicing.support_tool_index
            && self.support_interface_tool_index == cfg.slicing.support_interface_tool_index
            && self.support_style == cfg.slicing.support_style
            && self.support_on_bed_only == cfg.slicing.support_on_bed_only
            && self.seed_order_field_from_supports == cfg.slicing.seed_order_field_from_supports
            && self.support_only_unreachable_and_bridges
                == cfg.slicing.support_only_unreachable_and_bridges
            && approx_eq(self.support_z_gap_mm, cfg.slicing.support_z_gap_mm)
            && approx_eq(self.support_xy_gap_mm, cfg.slicing.support_xy_gap_mm)
            && self.support_interface_layers == cfg.slicing.support_interface_layers
            && self.support_bed_interface_layers == cfg.slicing.support_bed_interface_layers
            && approx_eq(
                self.support_base_spacing_mm,
                cfg.slicing.support_base_spacing_mm,
            )
            && approx_eq(
                self.support_interface_spacing_mm,
                cfg.slicing.support_interface_spacing_mm,
            )
            && approx_eq(
                self.support_line_width_mm,
                cfg.slicing.support_line_width_mm,
            )
            && approx_eq(
                self.tree_branch_angle_deg,
                cfg.slicing.tree_branch_angle_deg,
            )
            && approx_eq(
                self.tree_tip_diameter_mm,
                cfg.slicing.tree_tip_diameter_mm,
            )
            && approx_eq(
                self.tree_trunk_max_diameter_mm,
                cfg.slicing.tree_trunk_max_diameter_mm,
            )
            && self.tree_branch_wall_count == cfg.slicing.tree_branch_wall_count
            && self.enable_scarf_joint == cfg.gcode.enable_scarf_joint
            && approx_eq(self.scarf_length_mm, cfg.gcode.scarf_length_mm)
            && approx_eq(
                self.scarf_start_height_ratio,
                cfg.gcode.scarf_start_height_ratio,
            )
            && approx_eq(self.seam_gap_mm, cfg.gcode.seam_gap_mm)
            && approx_eq(
                self.first_layer_speed_mms,
                cfg.gcode.first_layer_speed_mms,
            )
            && approx_eq(self.outer_wall_speed_mms, cfg.gcode.outer_wall_speed_mms)
            && approx_eq(self.inner_wall_speed_mms, cfg.gcode.inner_wall_speed_mms)
            && approx_eq(
                self.solid_infill_speed_mms,
                cfg.gcode.solid_infill_speed_mms,
            )
            && approx_eq(
                self.sparse_infill_speed_mms,
                cfg.gcode.sparse_infill_speed_mms,
            )
            && approx_eq(self.gap_fill_speed_mms, cfg.gcode.gap_fill_speed_mms)
            && approx_eq(self.bridge_speed_mms, cfg.gcode.bridge_speed_mms)
            && approx_eq(
                self.wave_overhang_speed_mms,
                cfg.gcode.wave_overhang_speed_mms,
            )
            && approx_eq(self.support_speed_mms, cfg.gcode.support_speed_mms)
            && approx_eq(
                self.support_interface_speed_mms,
                cfg.gcode.support_interface_speed_mms,
            )
            && self.enable_auto_brim == cfg.gcode.enable_auto_brim
            && approx_eq(self.brim_width_mm, cfg.gcode.brim_width_mm)
            && approx_eq(self.brim_gap_mm, cfg.gcode.brim_gap_mm)
            && self.enable_lever_arm_slowdown == cfg.gcode.enable_lever_arm_slowdown
            && approx_eq(
                self.lever_arm_reference_ratio,
                cfg.gcode.lever_arm_reference_ratio,
            )
            && approx_eq(
                self.lever_arm_min_speed_fraction,
                cfg.gcode.lever_arm_min_speed_fraction,
            )
            && self.color_mixing == cfg.slicing.color_mixing
    }
}

// ============================================================================
// 5. PresetLibrary Resource & Disk Persistence
// ============================================================================

#[derive(Resource, Clone, Debug)]
pub struct PresetLibrary {
    pub printers: Vec<PrinterPreset>,
    pub tools: Vec<ToolPreset>,
    pub filaments: Vec<FilamentPreset>,
    pub processes: Vec<ProcessPreset>,

    pub selected_printer: String,
    pub selected_tool: String,
    pub selected_filament: String,
    pub selected_process: String,

    pub left_tab: LeftPanelTab,
    pub config_subtab: PresetCategory,
    pub show_left_panel: bool,
    pub show_view_panel: bool,
    pub save_as_open_for: Option<PresetCategory>,
    pub save_as_name_buf: String,
}

impl Default for PresetLibrary {
    fn default() -> Self {
        let mut lib = Self {
            printers: Self::builtin_printers(),
            tools: Self::builtin_tools(),
            filaments: Self::builtin_filaments(),
            processes: Self::builtin_processes(),
            selected_printer: "256³ CoreXY (Standard Klipper)".to_string(),
            selected_tool: "0.4mm Standard 5/6-Axis Conical".to_string(),
            selected_filament: "Generic PLA".to_string(),
            selected_process: "0.20mm Standard Non-Planar".to_string(),
            left_tab: LeftPanelTab::ProjectSetup,
            config_subtab: PresetCategory::Process,
            show_left_panel: true,
            show_view_panel: true,
            save_as_open_for: None,
            save_as_name_buf: String::new(),
        };
        lib.load_user_presets_from_disk();
        lib
    }
}

impl PresetLibrary {
    pub fn builtin_printers() -> Vec<PrinterPreset> {
        let base_cfg = SlicerConfig::default();
        let p_corexy =
            PrinterPreset::from_config("256³ CoreXY (Standard Klipper)", &base_cfg, true);

        let make_voron = |name: &str, wx: f32, dy: f32, hz: f32| -> PrinterPreset {
            let mut p = PrinterPreset::from_config(name, &base_cfg, true);
            p.bed_shape = BedShape::Rectangular;
            p.width_x = wx;
            p.depth_y = dy;
            p.height_z = hz;
            p.gcode_origin_at_bed_center = false;
            p.gcode_dialect = GcodeDialect::Klipper;
            p.network.protocol = PrinterNetworkProtocol::Moonraker;
            p.network.username.clear();
            p
        };

        let p_v0 = make_voron("Voron 0", 120.0, 120.0, 120.0);
        let p_trident_250 = make_voron("Voron Trident 250", 250.0, 250.0, 250.0);
        let p_trident_300 = make_voron("Voron Trident 300", 300.0, 300.0, 250.0);
        let p_trident_350 = make_voron("Voron Trident 350", 350.0, 350.0, 250.0);
        let p_v24_250 = make_voron("Voron 2.4 250", 250.0, 250.0, 220.0);
        let p_v24_300 = make_voron("Voron 2.4 300", 300.0, 300.0, 280.0);
        let p_v24_350 = make_voron("Voron 2.4 350", 350.0, 350.0, 330.0);

        let make_prusa_core_one = |name: &str,
                                   model_tag: &str,
                                   wx: f32,
                                   dy: f32,
                                   hz: f32,
                                   park_x: f32,
                                   park_y: f32,
                                   tool_count: Option<usize>|
         -> PrinterPreset {
            let mut p = PrinterPreset::from_config(name, &base_cfg, true);
            p.bed_shape = BedShape::Rectangular;
            p.width_x = wx;
            p.depth_y = dy;
            p.height_z = hz;
            p.gcode_origin_at_bed_center = false;
            p.gcode_dialect = GcodeDialect::Prusa;
            p.network.protocol = PrinterNetworkProtocol::PrusaLink;
            p.network.username = "maker".to_string();
            p.emit_exclude_object = true;
            p.emit_thumbnails = true;
            p.default_tool_count = tool_count;
            p.travel_speed_mms = 350.0;
            p.z_hop_speed_mms = 15.0;
            p.outer_wall_accel_mms2 = 4000.0;
            p.inner_wall_accel_mms2 = 6000.0;
            p.infill_accel_mms2 = 10000.0;
            p.travel_accel_mms2 = 10000.0;
            p.square_corner_velocity_mms = 9.0;

            let is_indx = tool_count.unwrap_or(1) > 1;
            p.start_gcode_template = if is_indx {
                format!(
                    concat!(
                        "; --- Manifold2 Prusa {name} (Buddy / Bondtech INDX) Print Start ---\n",
                        "M862.3 P \"{model_tag}\" ; Printer model check\n",
                        "M862.1 P{{nozzle_diameter}} ; Nozzle diameter check\n",
                        "G90 ; Absolute positioning\n",
                        "M83 ; Relative extrusion\n",
                        "M140 S{{bed_temp}} ; Set bed temperature\n",
                        "M104 S170 T{{tool_id}} ; No-ooze probing preheat on initial INDX tool\n",
                        "G28 ; Home all axes\n",
                        "T{{tool_id}} ; Pick up initial INDX toolhead\n",
                        "M190 S{{bed_temp}} ; Wait for bed temperature\n",
                        "M555 X{{bbox_min_x}} Y{{bbox_min_y}} W{{bbox_width_x}} H{{bbox_depth_y}} ; Set print area\n",
                        "G29 ; Print-area mesh bed leveling\n",
                        "M109 S{{nozzle_temp}} T{{tool_id}} ; Heat active INDX induction nozzle\n",
                        "G92 E0 ; Reset extruder"
                    ),
                    name = name,
                    model_tag = model_tag
                )
            } else {
                format!(
                    concat!(
                        "; --- Manifold2 Prusa {name} (Buddy) Print Start ---\n",
                        "M862.3 P \"{model_tag}\" ; Printer model check\n",
                        "M862.1 P{{nozzle_diameter}} ; Nozzle diameter check\n",
                        "G90 ; Absolute positioning\n",
                        "M83 ; Relative extrusion\n",
                        "M140 S{{bed_temp}} ; Set bed temperature\n",
                        "M104 S170 ; No-ooze probing preheat\n",
                        "G28 ; Home all axes\n",
                        "M190 S{{bed_temp}} ; Wait for bed temperature\n",
                        "M555 X{{bbox_min_x}} Y{{bbox_min_y}} W{{bbox_width_x}} H{{bbox_depth_y}} ; Set print area\n",
                        "G29 ; Print-area mesh bed leveling\n",
                        "M109 S{{nozzle_temp}} T{{tool_id}} ; Wait for nozzle print temperature\n",
                        "G92 E0 ; Reset extruder"
                    ),
                    name = name,
                    model_tag = model_tag
                )
            };

            let turn_off_tools = if let Some(n) = tool_count.filter(|&n| n > 1) {
                let mut cmds = String::new();
                for tid in 0..n {
                    cmds.push_str(&format!("M104 S0 T{tid} ; Turn off INDX tool T{tid}\n"));
                }
                cmds
            } else {
                "M104 S0 ; Turn off nozzle heater\n".to_string()
            };
            p.end_gcode_template = format!(
                concat!(
                    "; --- Manifold2 Prusa {name} Print End ---\n",
                    "M400 ; Wait for moves to finish\n",
                    "{turn_off_tools}",
                    "M140 S0 ; Turn off heated bed\n",
                    "M107 ; Turn off part cooling fan\n",
                    "G1 X{park_x:.0} Y{park_y:.0} F10200 ; Park carriage\n",
                    "M84 X Y E ; Disable XY & E steppers"
                ),
                name = name,
                turn_off_tools = turn_off_tools,
                park_x = park_x,
                park_y = park_y
            );
            p
        };

        let p_core_one = make_prusa_core_one(
            "Prusa CORE One",
            "COREONE",
            250.0,
            220.0,
            270.0,
            242.0,
            211.0,
            Some(1),
        );
        let p_core_one_l = make_prusa_core_one(
            "Prusa CORE One L",
            "COREONEL",
            300.0,
            300.0,
            330.0,
            290.0,
            290.0,
            Some(1),
        );
        let p_core_one_indx_4 = make_prusa_core_one(
            "Prusa CORE One INDX (4-Tool)",
            "COREONE",
            248.0,
            205.0,
            270.0,
            240.0,
            200.0,
            Some(4),
        );
        let p_core_one_indx_8 = make_prusa_core_one(
            "Prusa CORE One INDX (8-Tool)",
            "COREONE",
            248.0,
            205.0,
            270.0,
            240.0,
            200.0,
            Some(8),
        );
        let p_core_one_l_indx_4 = make_prusa_core_one(
            "Prusa CORE One L INDX (4-Tool)",
            "COREONEL",
            298.0,
            275.0,
            330.0,
            290.0,
            270.0,
            Some(4),
        );
        let p_core_one_l_indx_8 = make_prusa_core_one(
            "Prusa CORE One L INDX (8-Tool)",
            "COREONEL",
            298.0,
            275.0,
            330.0,
            290.0,
            270.0,
            Some(8),
        );

        let mut p_5axis =
            PrinterPreset::from_config("5-Axis Tilting Build Plate (250×210)", &base_cfg, true);
        p_5axis.width_x = 250.0;
        p_5axis.depth_y = 210.0;
        p_5axis.height_z = 220.0;
        p_5axis.gcode_origin_at_bed_center = true;
        p_5axis.kinematics_mode = FiveAxisKinematicsMode::FiveAxisTiltingBedAB;
        p_5axis.a_min_deg = -45.0;
        p_5axis.a_max_deg = 45.0;
        p_5axis.b_min_deg = -45.0;
        p_5axis.b_max_deg = 45.0;
        p_5axis.max_a_speed_deg_s = 60.0;
        p_5axis.max_b_speed_deg_s = 60.0;
        p_5axis.a_accel_deg_s2 = 300.0;
        p_5axis.b_accel_deg_s2 = 300.0;
        p_5axis.bed_tilts_with_b = true;
        p_5axis.outer_wall_accel_mms2 = 2000.0;
        p_5axis.inner_wall_accel_mms2 = 3500.0;
        p_5axis.infill_accel_mms2 = 5000.0;
        p_5axis.travel_accel_mms2 = 7500.0;
        p_5axis.start_gcode_template = concat!(
            "; --- Manifold2 5-Axis Tilting Bed (XYZ + AB) Print Start ---\n",
            "; Print Bounds: X[{bbox_min_x}..{bbox_max_x}] Y[{bbox_min_y}..{bbox_max_y}] Z[{bbox_min_z}..{bbox_max_z}]\n",
            "G90 ; Absolute positioning\n",
            "M83 ; Relative extrusion\n",
            "M140 S{bed_temp} ; Set bed temperature\n",
            "M104 S{nozzle_temp} T{tool_id} ; Set nozzle temperature\n",
            "G28 ; Home all axes (X Y Z A B)\n",
            "T{tool_id} ; Select active tool\n",
            "M190 S{bed_temp} ; Wait for bed temperature\n",
            "M109 S{nozzle_temp} T{tool_id} ; Wait for nozzle temperature\n",
            "G1 X0 Y0 Z15 A0 B0 F3000 ; Move to neutral level-bed 5-axis posture\n",
            "G92 E0 ; Reset extruder"
        )
        .to_string();
        p_5axis.end_gcode_template = concat!(
            "; --- Manifold2 5-Axis Tilting Bed (XYZ + AB) Print End ---\n",
            "M400 ; Wait for moves to finish\n",
            "G1 A0 B0 F1800 ; Level tilting bed axes\n",
            "M104 S0 ; Turn off hotend\n",
            "M140 S0 ; Turn off bed\n",
            "M107 ; Turn off part cooling fan\n",
            "G28 X Y ; Park toolhead\n",
            "M84 ; Disable steppers"
        )
        .to_string();

        let mut p_trident_5axis_ab = PrinterPreset::from_config(
            "Voron Trident 5-Axis Tilting Bed (300³ A/B)",
            &base_cfg,
            true,
        );
        p_trident_5axis_ab.width_x = 300.0;
        p_trident_5axis_ab.depth_y = 300.0;
        p_trident_5axis_ab.height_z = 250.0;
        p_trident_5axis_ab.gcode_origin_at_bed_center = true;
        p_trident_5axis_ab.gcode_dialect = GcodeDialect::Klipper;
        p_trident_5axis_ab.kinematics_mode = FiveAxisKinematicsMode::FiveAxisTiltingBedAB;
        p_trident_5axis_ab.a_min_deg = -45.0;
        p_trident_5axis_ab.a_max_deg = 45.0;
        p_trident_5axis_ab.b_min_deg = -45.0;
        p_trident_5axis_ab.b_max_deg = 45.0;
        p_trident_5axis_ab.max_a_speed_deg_s = 60.0;
        p_trident_5axis_ab.max_b_speed_deg_s = 60.0;
        p_trident_5axis_ab.a_accel_deg_s2 = 300.0;
        p_trident_5axis_ab.b_accel_deg_s2 = 300.0;
        p_trident_5axis_ab.orientation_smoothing_window_mm = 2.5;
        p_trident_5axis_ab.bed_tilts_with_b = true;
        p_trident_5axis_ab.outer_wall_accel_mms2 = 2500.0;
        p_trident_5axis_ab.inner_wall_accel_mms2 = 4000.0;
        p_trident_5axis_ab.infill_accel_mms2 = 6000.0;
        p_trident_5axis_ab.travel_accel_mms2 = 9000.0;
        p_trident_5axis_ab.start_gcode_template = p_5axis.start_gcode_template.clone();
        p_trident_5axis_ab.end_gcode_template = p_5axis.end_gcode_template.clone();

        let mut p_rrf_5axis_bc =
            PrinterPreset::from_config("5-Axis Circular Bed (RRF B/C)", &base_cfg, true);
        p_rrf_5axis_bc.bed_shape = BedShape::Circular;
        p_rrf_5axis_bc.width_x = 240.0;
        p_rrf_5axis_bc.depth_y = 240.0;
        p_rrf_5axis_bc.height_z = 220.0;
        p_rrf_5axis_bc.gcode_origin_at_bed_center = true;
        p_rrf_5axis_bc.gcode_dialect = GcodeDialect::RepRapFirmware;
        p_rrf_5axis_bc.kinematics_mode = FiveAxisKinematicsMode::FiveAxisBC;
        p_rrf_5axis_bc.b_min_deg = 0.0;
        p_rrf_5axis_bc.b_max_deg = 60.0;
        p_rrf_5axis_bc.c_continuous = true;
        p_rrf_5axis_bc.max_b_speed_deg_s = 60.0;
        p_rrf_5axis_bc.max_c_speed_deg_s = 120.0;
        p_rrf_5axis_bc.b_accel_deg_s2 = 300.0;
        p_rrf_5axis_bc.c_accel_deg_s2 = 600.0;
        p_rrf_5axis_bc.orientation_smoothing_window_mm = 2.5;
        p_rrf_5axis_bc.bed_tilts_with_b = true;
        p_rrf_5axis_bc.outer_wall_accel_mms2 = 2000.0;
        p_rrf_5axis_bc.inner_wall_accel_mms2 = 3500.0;
        p_rrf_5axis_bc.infill_accel_mms2 = 5000.0;
        p_rrf_5axis_bc.travel_accel_mms2 = 7500.0;
        p_rrf_5axis_bc.start_gcode_template = concat!(
            "; --- Manifold2 5-Axis RRF (XYZ + BC) Print Start ---\n",
            "; Print Bounds: X[{bbox_min_x}..{bbox_max_x}] Y[{bbox_min_y}..{bbox_max_y}] Z[{bbox_min_z}..{bbox_max_z}]\n",
            "G90 ; Absolute positioning\n",
            "M83 ; Relative extrusion\n",
            "M140 S{bed_temp} ; Set bed temperature\n",
            "M568 P{tool_id} S{nozzle_temp} R{nozzle_temp} A2 ; Set RRF tool temperature\n",
            "G28 ; Home all axes (X Y Z B C)\n",
            "T{tool_id} ; Select active tool\n",
            "M190 S{bed_temp} ; Wait for bed temperature\n",
            "M116 P{tool_id} ; Wait for tool temperature\n",
            "G1 X0 Y0 Z15 B0 C0 F3000 ; Move to neutral 5-axis posture above bed center\n",
            "G92 E0 ; Reset extruder"
        )
        .to_string();
        p_rrf_5axis_bc.end_gcode_template = concat!(
            "; --- Manifold2 5-Axis RRF (XYZ + BC) Print End ---\n",
            "M400 ; Wait for moves to finish\n",
            "G1 B0 C0 F1800 ; Level rotary axes\n",
            "M568 P{tool_id} S0 R0 A0 ; Turn off tool heater\n",
            "M140 S0 ; Turn off heated bed\n",
            "M106 S0 ; Turn off part cooling fan\n",
            "G28 X Y ; Park linear axes\n",
            "M84 ; Disable steppers"
        )
        .to_string();

        let mut p_6axis =
            PrinterPreset::from_config("6-Axis Robot Arm Cell (350³)", &base_cfg, true);
        p_6axis.width_x = 350.0;
        p_6axis.depth_y = 350.0;
        p_6axis.height_z = 350.0;
        p_6axis.gcode_origin_at_bed_center = true;
        p_6axis.travel_speed_mms = 180.0;
        p_6axis.outer_wall_accel_mms2 = 1500.0;
        p_6axis.inner_wall_accel_mms2 = 2500.0;
        p_6axis.infill_accel_mms2 = 3500.0;
        p_6axis.travel_accel_mms2 = 5000.0;

        let make_bambu = |name: &str,
                          wx: f32,
                          dy: f32,
                          hz: f32,
                          park_x: f32,
                          park_y: f32,
                          tool_count: Option<usize>|
         -> PrinterPreset {
            let mut p = PrinterPreset::from_config(name, &base_cfg, true);
            p.bed_shape = BedShape::Rectangular;
            p.width_x = wx;
            p.depth_y = dy;
            p.height_z = hz;
            p.gcode_origin_at_bed_center = false;
            p.gcode_dialect = GcodeDialect::BambuLab;
            p.network.protocol = PrinterNetworkProtocol::BambuLan;
            p.network.username = "bblp".to_string();
            p.emit_exclude_object = true;
            p.emit_thumbnails = true;
            p.default_tool_count = tool_count;
            p.travel_speed_mms = 400.0;
            p.z_hop_speed_mms = 15.0;
            p.outer_wall_accel_mms2 = 5000.0;
            p.inner_wall_accel_mms2 = 8000.0;
            p.infill_accel_mms2 = 10000.0;
            p.travel_accel_mms2 = 10000.0;
            p.square_corner_velocity_mms = 9.0;

            let use_ams_swap = tool_count.unwrap_or(1) > 1;
            let ams_init = if use_ams_swap {
                "M620 M\nM620 S{tool_id}A\nT{tool_id}\nM109 S{nozzle_temp}\nM621 S{tool_id}A ; Load initial AMS filament slot\n"
            } else {
                "T{tool_id}\n"
            };
            p.start_gcode_template = format!(
                concat!(
                    "; --- Manifold2 {name} Print Start ---\n",
                    "M17 ; Enable steppers\n",
                    "G90 ; Absolute positioning\n",
                    "M83 ; Relative extrusion\n",
                    "M140 S{{bed_temp}} ; Set bed temperature\n",
                    "M104 S150 ; Preheat nozzle for bed probing\n",
                    "G28 ; Home all axes\n",
                    "M190 S{{bed_temp}} ; Wait for bed temperature\n",
                    "G29.1 Z0.0 ; Apply bed mesh offset\n",
                    "G29 A X{{bbox_min_x}} Y{{bbox_min_y}} I{{bbox_width_x}} J{{bbox_depth_y}} ; Print-area mesh bed leveling\n",
                    "{ams_init}",
                    "M109 S{{nozzle_temp}} ; Wait for nozzle print temperature\n",
                    "G92 E0 ; Reset extruder"
                ),
                name = name,
                ams_init = ams_init
            );
            p.end_gcode_template = format!(
                concat!(
                    "; --- Manifold2 {name} Print End ---\n",
                    "M400 ; Wait for moves to finish\n",
                    "M104 S0 ; Turn off hotend\n",
                    "M140 S0 ; Turn off heated bed\n",
                    "M106 P1 S0 ; Turn off part cooling fan\n",
                    "M106 P2 S0 ; Turn off auxiliary cooling fan\n",
                    "M106 P3 S0 ; Turn off chamber fan\n",
                    "G1 X{park_x:.0} Y{park_y:.0} F12000 ; Park toolhead\n",
                    "M84 ; Disable steppers"
                ),
                name = name,
                park_x = park_x,
                park_y = park_y
            );
            p
        };

        let p_bambu_x1c = make_bambu("Bambu Lab X1 Carbon", 256.0, 256.0, 256.0, 65.0, 250.0, Some(1));
        let p_bambu_x1c_ams = make_bambu(
            "Bambu Lab X1 Carbon (AMS 4-Slot)",
            256.0,
            256.0,
            256.0,
            65.0,
            250.0,
            Some(4),
        );
        let p_bambu_p1s = make_bambu("Bambu Lab P1S / P1P", 256.0, 256.0, 256.0, 65.0, 250.0, Some(1));
        let p_bambu_p1s_ams = make_bambu(
            "Bambu Lab P1S (AMS 4-Slot)",
            256.0,
            256.0,
            256.0,
            65.0,
            250.0,
            Some(4),
        );
        let p_bambu_a1 = make_bambu("Bambu Lab A1", 256.0, 256.0, 256.0, 0.0, 250.0, Some(1));
        let p_bambu_a1_ams = make_bambu(
            "Bambu Lab A1 (AMS Lite 4-Slot)",
            256.0,
            256.0,
            256.0,
            0.0,
            250.0,
            Some(4),
        );
        let p_bambu_a1_mini = make_bambu("Bambu Lab A1 Mini", 180.0, 180.0, 180.0, 0.0, 176.0, Some(1));

        vec![
            p_corexy,
            p_v0,
            p_trident_250,
            p_trident_300,
            p_trident_350,
            p_v24_250,
            p_v24_300,
            p_v24_350,
            p_core_one,
            p_core_one_l,
            p_core_one_indx_4,
            p_core_one_indx_8,
            p_core_one_l_indx_4,
            p_core_one_l_indx_8,
            p_bambu_x1c,
            p_bambu_x1c_ams,
            p_bambu_p1s,
            p_bambu_p1s_ams,
            p_bambu_a1,
            p_bambu_a1_ams,
            p_bambu_a1_mini,
            p_5axis,
            p_trident_5axis_ab,
            p_rrf_5axis_bc,
            p_6axis,
        ]
    }

    pub fn builtin_tools() -> Vec<ToolPreset> {
        let base_cfg = SlicerConfig::default();
        let t_04 =
            ToolPreset::from_config("0.4mm Standard 5/6-Axis Conical", &base_cfg, true);

        let mut t_06 = ToolPreset::from_config("0.6mm High-Flow", &base_cfg, true);
        t_06.nozzle_diameter_mm = 0.60;
        t_06.max_volumetric_flow_mm3s = 28.0;
        t_06.clearance_profile_xz = vec![
            ClearancePointXZ::new(0.75, 0.0),
            ClearancePointXZ::new(2.8, 2.0),
            ClearancePointXZ::new(9.5, 6.0),
            ClearancePointXZ::new(22.0, 18.0),
            ClearancePointXZ::new(28.0, 45.0),
        ];

        let mut t_025 =
            ToolPreset::from_config("0.25mm Precision Extended-Tip", &base_cfg, true);
        t_025.nozzle_diameter_mm = 0.25;
        t_025.max_angle_from_bed_normal_deg = 60.0;
        t_025.max_volumetric_flow_mm3s = 8.0;
        t_025.clearance_profile_xz = vec![
            ClearancePointXZ::new(0.35, 0.0),
            ClearancePointXZ::new(1.8, 3.0),
            ClearancePointXZ::new(6.5, 9.0),
            ClearancePointXZ::new(18.0, 22.0),
            ClearancePointXZ::new(26.0, 48.0),
        ];

        let mut t_rect =
            ToolPreset::from_config("0.4mm Rectilinear Shroud Cone (XY)", &base_cfg, true);
        t_rect.clearance_mode = ClearanceProfileMode::RectilinearCone;
        t_rect.rect_tip_half_width_x_mm = 0.60;
        t_rect.rect_tip_half_width_y_mm = 0.60;
        t_rect.rect_half_angle_x_deg = 38.0;
        t_rect.rect_half_angle_y_deg = 24.0;
        t_rect.rect_height_mm = 45.0;

        let mut t_nextruder =
            ToolPreset::from_config("0.4mm Prusa Nextruder High-Flow", &base_cfg, true);
        t_nextruder.nozzle_diameter_mm = 0.40;
        t_nextruder.has_rotational_axes = false;
        t_nextruder.max_angle_from_bed_normal_deg = 45.0;
        t_nextruder.max_volumetric_flow_mm3s = 26.0;
        t_nextruder.retract_length_mm = 0.70;
        t_nextruder.retract_speed_mms = 35.0;
        t_nextruder.z_hop_mm = 0.20;
        t_nextruder.clearance_profile_xz = vec![
            ClearancePointXZ::new(0.50, 0.0),
            ClearancePointXZ::new(3.5, 1.5),
            ClearancePointXZ::new(11.0, 3.0),
            ClearancePointXZ::new(24.0, 9.0),
            ClearancePointXZ::new(34.0, 18.0),
            ClearancePointXZ::new(38.0, 45.0),
        ];

        let mut t_nextruder_std =
            ToolPreset::from_config("0.4mm Prusa Nextruder Standard", &base_cfg, true);
        t_nextruder_std.nozzle_diameter_mm = 0.40;
        t_nextruder_std.has_rotational_axes = false;
        t_nextruder_std.max_angle_from_bed_normal_deg = 45.0;
        t_nextruder_std.max_volumetric_flow_mm3s = 16.0;
        t_nextruder_std.retract_length_mm = 0.70;
        t_nextruder_std.retract_speed_mms = 35.0;
        t_nextruder_std.z_hop_mm = 0.20;
        t_nextruder_std.clearance_profile_xz = t_nextruder.clearance_profile_xz.clone();

        let mut t_nextruder_06 =
            ToolPreset::from_config("0.6mm Prusa Nextruder High-Flow", &base_cfg, true);
        t_nextruder_06.nozzle_diameter_mm = 0.60;
        t_nextruder_06.has_rotational_axes = false;
        t_nextruder_06.max_angle_from_bed_normal_deg = 45.0;
        t_nextruder_06.max_volumetric_flow_mm3s = 32.0;
        t_nextruder_06.retract_length_mm = 0.80;
        t_nextruder_06.retract_speed_mms = 35.0;
        t_nextruder_06.z_hop_mm = 0.25;
        t_nextruder_06.clearance_profile_xz = vec![
            ClearancePointXZ::new(0.75, 0.0),
            ClearancePointXZ::new(3.5, 1.5),
            ClearancePointXZ::new(11.0, 3.0),
            ClearancePointXZ::new(24.0, 9.0),
            ClearancePointXZ::new(34.0, 18.0),
            ClearancePointXZ::new(38.0, 45.0),
        ];

        let mut t_indx =
            ToolPreset::from_config("0.4mm Bondtech INDX Toolchanger", &base_cfg, true);
        t_indx.nozzle_diameter_mm = 0.40;
        t_indx.has_rotational_axes = false;
        t_indx.max_angle_from_bed_normal_deg = 45.0;
        t_indx.max_volumetric_flow_mm3s = 24.0;
        t_indx.retract_length_mm = 0.60;
        t_indx.retract_speed_mms = 40.0;
        t_indx.z_hop_mm = 0.20;
        t_indx.tool_change_time_s = 12.0;
        t_indx.tool_change_hop_before = true;
        t_indx.tool_change_approach_after = true;
        t_indx.clearance_profile_xz = vec![
            ClearancePointXZ::new(0.50, 0.0),
            ClearancePointXZ::new(3.0, 2.0),
            ClearancePointXZ::new(8.5, 5.0),
            ClearancePointXZ::new(17.5, 11.0),
            ClearancePointXZ::new(30.0, 22.0),
            ClearancePointXZ::new(35.0, 45.0),
        ];

        let mut t_indx_06 =
            ToolPreset::from_config("0.6mm Bondtech INDX Toolchanger", &base_cfg, true);
        t_indx_06.nozzle_diameter_mm = 0.60;
        t_indx_06.has_rotational_axes = false;
        t_indx_06.max_angle_from_bed_normal_deg = 45.0;
        t_indx_06.max_volumetric_flow_mm3s = 30.0;
        t_indx_06.retract_length_mm = 0.70;
        t_indx_06.retract_speed_mms = 40.0;
        t_indx_06.z_hop_mm = 0.25;
        t_indx_06.tool_change_time_s = 12.0;
        t_indx_06.tool_change_hop_before = true;
        t_indx_06.tool_change_approach_after = true;
        t_indx_06.clearance_profile_xz = vec![
            ClearancePointXZ::new(0.75, 0.0),
            ClearancePointXZ::new(3.0, 2.0),
            ClearancePointXZ::new(8.5, 5.0),
            ClearancePointXZ::new(17.5, 11.0),
            ClearancePointXZ::new(30.0, 22.0),
            ClearancePointXZ::new(35.0, 45.0),
        ];

        let mut t_bambu_x1_p1 =
            ToolPreset::from_config("0.4mm Bambu Lab X1/P1 Hardened Steel", &base_cfg, true);
        t_bambu_x1_p1.nozzle_diameter_mm = 0.40;
        t_bambu_x1_p1.has_rotational_axes = false;
        t_bambu_x1_p1.max_volumetric_flow_mm3s = 22.0;
        t_bambu_x1_p1.retract_length_mm = 0.80;
        t_bambu_x1_p1.retract_speed_mms = 30.0;
        t_bambu_x1_p1.z_hop_mm = 0.40;
        t_bambu_x1_p1.tool_change_time_s = 85.0;
        t_bambu_x1_p1.clearance_profile_xz = vec![
            ClearancePointXZ::new(0.55, 0.0),
            ClearancePointXZ::new(3.2, 2.2),
            ClearancePointXZ::new(14.5, 4.5),
            ClearancePointXZ::new(28.0, 14.0),
            ClearancePointXZ::new(42.0, 45.0),
        ];

        let mut t_bambu_a1 =
            ToolPreset::from_config("0.4mm Bambu Lab A1 Quick-Swap", &base_cfg, true);
        t_bambu_a1.nozzle_diameter_mm = 0.40;
        t_bambu_a1.has_rotational_axes = false;
        t_bambu_a1.max_volumetric_flow_mm3s = 22.0;
        t_bambu_a1.retract_length_mm = 0.80;
        t_bambu_a1.retract_speed_mms = 30.0;
        t_bambu_a1.z_hop_mm = 0.40;
        t_bambu_a1.tool_change_time_s = 45.0;
        t_bambu_a1.clearance_profile_xz = vec![
            ClearancePointXZ::new(0.55, 0.0),
            ClearancePointXZ::new(3.0, 2.5),
            ClearancePointXZ::new(13.0, 5.0),
            ClearancePointXZ::new(26.0, 15.0),
            ClearancePointXZ::new(38.0, 45.0),
        ];

        // --- Voron Toolhead Presets (Measured from official VoronDesign & Armchair Heavy Industries CAD/STLs) ---

        // 1a. Voron Stealthburner (Standard HF hotend: 1.5 mm nozzle protrusion below duct tips,
        //     65.1 mm main cowling width -> 32.6 mm radius, 71.1 mm rear carriage width -> 35.5 mm radius)
        let mut t_sb_std =
            ToolPreset::from_config("0.4mm Voron Stealthburner (Standard)", &base_cfg, true);
        t_sb_std.nozzle_diameter_mm = 0.40;
        t_sb_std.has_rotational_axes = false;
        t_sb_std.max_angle_from_bed_normal_deg = 45.0;
        t_sb_std.max_volumetric_flow_mm3s = 24.0;
        t_sb_std.retract_length_mm = 0.75;
        t_sb_std.retract_speed_mms = 40.0;
        t_sb_std.z_hop_mm = 0.40;
        t_sb_std.clearance_profile_xz = vec![
            ClearancePointXZ::new(0.50, 0.0),
            ClearancePointXZ::new(4.0, 1.5),
            ClearancePointXZ::new(10.5, 2.0),
            ClearancePointXZ::new(24.0, 6.0),
            ClearancePointXZ::new(32.6, 12.0),
            ClearancePointXZ::new(35.5, 45.0),
        ];

        // 1b. Voron Stealthburner (+8.5mm MZE / UHF: shifts shroud/ducts up by +8.5 mm relative to nozzle tip)
        let mut t_sb_mze = ToolPreset::from_config(
            "0.4mm Voron Stealthburner (+8.5mm MZE / UHF)",
            &base_cfg,
            true,
        );
        t_sb_mze.nozzle_diameter_mm = 0.40;
        t_sb_mze.has_rotational_axes = false;
        t_sb_mze.max_angle_from_bed_normal_deg = 45.0;
        t_sb_mze.max_volumetric_flow_mm3s = 35.0;
        t_sb_mze.retract_length_mm = 0.90;
        t_sb_mze.retract_speed_mms = 45.0;
        t_sb_mze.z_hop_mm = 0.40;
        t_sb_mze.clearance_profile_xz = vec![
            ClearancePointXZ::new(0.50, 0.0),
            ClearancePointXZ::new(3.5, 2.0),
            ClearancePointXZ::new(6.0, 8.5),
            ClearancePointXZ::new(10.5, 10.5),
            ClearancePointXZ::new(24.0, 14.5),
            ClearancePointXZ::new(32.6, 20.5),
            ClearancePointXZ::new(35.5, 53.5),
        ];

        // 2a. Voron DownDraft (Standard / Dual 4010 Low-Mount: 61.0 mm shroud width -> 30.5 mm radius;
        //     Dual 5015 variant reaches 34.0 mm at z >= 15 mm)
        let mut t_dd_std = ToolPreset::from_config(
            "0.4mm Voron DownDraft (Standard / Low-Mount)",
            &base_cfg,
            true,
        );
        t_dd_std.nozzle_diameter_mm = 0.40;
        t_dd_std.has_rotational_axes = false;
        t_dd_std.max_angle_from_bed_normal_deg = 45.0;
        t_dd_std.max_volumetric_flow_mm3s = 26.0;
        t_dd_std.retract_length_mm = 0.75;
        t_dd_std.retract_speed_mms = 45.0;
        t_dd_std.z_hop_mm = 0.40;
        t_dd_std.clearance_profile_xz = vec![
            ClearancePointXZ::new(0.50, 0.0),
            ClearancePointXZ::new(4.0, 1.5),
            ClearancePointXZ::new(9.5, 2.2),
            ClearancePointXZ::new(22.0, 6.5),
            ClearancePointXZ::new(30.5, 10.5),
            ClearancePointXZ::new(34.0, 43.0),
        ];

        // 2b. Voron DownDraft (+8.5mm MZE / UHF High-Mount: Note that DownDraft uses modular Low/Middle/High
        //     front mounts to keep the lower duct tips at constant height relative to the nozzle tip, while
        //     giving more vertical heater-block clearance inside the throat!)
        let mut t_dd_mze = ToolPreset::from_config(
            "0.4mm Voron DownDraft (+8.5mm MZE / UHF)",
            &base_cfg,
            true,
        );
        t_dd_mze.nozzle_diameter_mm = 0.40;
        t_dd_mze.has_rotational_axes = false;
        t_dd_mze.max_angle_from_bed_normal_deg = 45.0;
        t_dd_mze.max_volumetric_flow_mm3s = 38.0;
        t_dd_mze.retract_length_mm = 0.90;
        t_dd_mze.retract_speed_mms = 45.0;
        t_dd_mze.z_hop_mm = 0.40;
        t_dd_mze.clearance_profile_xz = vec![
            ClearancePointXZ::new(0.50, 0.0),
            ClearancePointXZ::new(3.5, 2.0),
            ClearancePointXZ::new(6.0, 8.5),
            ClearancePointXZ::new(9.5, 10.7),
            ClearancePointXZ::new(22.0, 15.0),
            ClearancePointXZ::new(30.5, 19.0),
            ClearancePointXZ::new(34.0, 51.5),
        ];

        // 3a. Voron Xol Toolhead (Standard HF: SLiMXolPAP / 4010 ducts 71.68 mm wide -> 35.8 mm radius,
        //     upper fan mount 78.64 mm wide -> 39.3 mm radius)
        let mut t_xol_std =
            ToolPreset::from_config("0.4mm Voron Xol (Standard)", &base_cfg, true);
        t_xol_std.nozzle_diameter_mm = 0.40;
        t_xol_std.has_rotational_axes = false;
        t_xol_std.max_angle_from_bed_normal_deg = 45.0;
        t_xol_std.max_volumetric_flow_mm3s = 28.0;
        t_xol_std.retract_length_mm = 0.80;
        t_xol_std.retract_speed_mms = 45.0;
        t_xol_std.z_hop_mm = 0.40;
        t_xol_std.clearance_profile_xz = vec![
            ClearancePointXZ::new(0.50, 0.0),
            ClearancePointXZ::new(4.0, 1.5),
            ClearancePointXZ::new(9.0, 5.5),
            ClearancePointXZ::new(35.8, 9.0),
            ClearancePointXZ::new(39.3, 44.0),
        ];

        // 3b. Voron Xol Toolhead (+8.5mm MZE / UHF: with free-hanging MZE or UHF nozzle drop)
        let mut t_xol_mze =
            ToolPreset::from_config("0.4mm Voron Xol (+8.5mm MZE / UHF)", &base_cfg, true);
        t_xol_mze.nozzle_diameter_mm = 0.40;
        t_xol_mze.has_rotational_axes = false;
        t_xol_mze.max_angle_from_bed_normal_deg = 45.0;
        t_xol_mze.max_volumetric_flow_mm3s = 38.0;
        t_xol_mze.retract_length_mm = 0.90;
        t_xol_mze.retract_speed_mms = 45.0;
        t_xol_mze.z_hop_mm = 0.40;
        t_xol_mze.clearance_profile_xz = vec![
            ClearancePointXZ::new(0.50, 0.0),
            ClearancePointXZ::new(3.5, 2.0),
            ClearancePointXZ::new(6.0, 8.5),
            ClearancePointXZ::new(9.0, 14.0),
            ClearancePointXZ::new(35.8, 17.5),
            ClearancePointXZ::new(39.3, 52.5),
        ];

        // 4a. Voron A4T Toolhead (Standard HF: 1.4 mm tip clearance, 64 mm lower duct span -> 32.0 mm radius
        //     at z = 9.0 mm, 88.58 mm outer cowling width -> 44.3 mm radius at z = 40..53.8 mm)
        let mut t_a4t_std =
            ToolPreset::from_config("0.4mm Voron A4T (Standard)", &base_cfg, true);
        t_a4t_std.nozzle_diameter_mm = 0.40;
        t_a4t_std.has_rotational_axes = false;
        t_a4t_std.max_angle_from_bed_normal_deg = 45.0;
        t_a4t_std.max_volumetric_flow_mm3s = 28.0;
        t_a4t_std.retract_length_mm = 0.80;
        t_a4t_std.retract_speed_mms = 45.0;
        t_a4t_std.z_hop_mm = 0.40;
        t_a4t_std.clearance_profile_xz = vec![
            ClearancePointXZ::new(0.50, 0.0),
            ClearancePointXZ::new(4.0, 1.4),
            ClearancePointXZ::new(9.0, 6.0),
            ClearancePointXZ::new(32.0, 9.0),
            ClearancePointXZ::new(44.3, 40.0),
        ];

        // 4b. Voron A4T Toolhead (+8.5mm MZE / UHF: standard A4T cowling with +8.5 mm MZE drop, or UHF cowling)
        let mut t_a4t_mze =
            ToolPreset::from_config("0.4mm Voron A4T (+8.5mm MZE / UHF)", &base_cfg, true);
        t_a4t_mze.nozzle_diameter_mm = 0.40;
        t_a4t_mze.has_rotational_axes = false;
        t_a4t_mze.max_angle_from_bed_normal_deg = 45.0;
        t_a4t_mze.max_volumetric_flow_mm3s = 38.0;
        t_a4t_mze.retract_length_mm = 0.90;
        t_a4t_mze.retract_speed_mms = 45.0;
        t_a4t_mze.z_hop_mm = 0.40;
        t_a4t_mze.clearance_profile_xz = vec![
            ClearancePointXZ::new(0.50, 0.0),
            ClearancePointXZ::new(3.5, 2.0),
            ClearancePointXZ::new(6.0, 8.5),
            ClearancePointXZ::new(9.0, 14.5),
            ClearancePointXZ::new(32.0, 17.5),
            ClearancePointXZ::new(44.3, 48.5),
        ];

        vec![
            t_04,
            t_06,
            t_025,
            t_rect,
            t_nextruder_std,
            t_nextruder,
            t_nextruder_06,
            t_indx,
            t_indx_06,
            t_bambu_x1_p1,
            t_bambu_a1,
            t_sb_std,
            t_sb_mze,
            t_dd_std,
            t_dd_mze,
            t_xol_std,
            t_xol_mze,
            t_a4t_std,
            t_a4t_mze,
        ]
    }

    pub fn builtin_filaments() -> Vec<FilamentPreset> {
        let base_cfg = SlicerConfig::default();
        let f_pla = FilamentPreset::from_config("Generic PLA", &base_cfg, true);

        let mut f_plaplus = FilamentPreset::from_config("High-Flow PLA+", &base_cfg, true);
        f_plaplus.nozzle_temperature_celsius = 225.0;
        f_plaplus.bed_temperature_celsius = 60.0;
        f_plaplus.pa_low_flow_s = 0.036;
        f_plaplus.pa_high_flow_s = 0.020;

        let mut f_petg = FilamentPreset::from_config("Generic PETG", &base_cfg, true);
        f_petg.filament_type = "PETG".to_string();
        f_petg.filament_density_g_cm3 = 1.27;
        f_petg.nozzle_temperature_celsius = 240.0;
        f_petg.bed_temperature_celsius = 75.0;
        f_petg.fan_speed_percent = 40.0;
        f_petg.min_fan_speed_percent = 20.0;
        f_petg.max_fan_speed_percent = 80.0;
        f_petg.bridge_fan_speed_percent = 80.0;
        f_petg.wave_overhang_fan_speed_percent = 70.0;
        f_petg.pa_low_flow_s = 0.055;
        f_petg.pa_high_flow_s = 0.032;

        let mut f_abs = FilamentPreset::from_config("Generic ABS", &base_cfg, true);
        f_abs.filament_type = "ABS".to_string();
        f_abs.filament_density_g_cm3 = 1.04;
        f_abs.nozzle_temperature_celsius = 255.0;
        f_abs.bed_temperature_celsius = 100.0;
        f_abs.fan_speed_percent = 20.0;
        f_abs.min_fan_speed_percent = 10.0;
        f_abs.max_fan_speed_percent = 50.0;
        f_abs.bridge_fan_speed_percent = 50.0;
        f_abs.wave_overhang_fan_speed_percent = 45.0;
        f_abs.pa_low_flow_s = 0.038;
        f_abs.pa_high_flow_s = 0.022;

        let mut f_asa = FilamentPreset::from_config("Generic ASA", &base_cfg, true);
        f_asa.filament_type = "ASA".to_string();
        f_asa.filament_density_g_cm3 = 1.07;
        f_asa.nozzle_temperature_celsius = 260.0;
        f_asa.bed_temperature_celsius = 105.0;
        f_asa.fan_speed_percent = 20.0;
        f_asa.min_fan_speed_percent = 10.0;
        f_asa.max_fan_speed_percent = 50.0;
        f_asa.bridge_fan_speed_percent = 50.0;
        f_asa.wave_overhang_fan_speed_percent = 45.0;

        let mut f_tpu = FilamentPreset::from_config("TPU 95A", &base_cfg, true);
        f_tpu.filament_type = "TPU".to_string();
        f_tpu.filament_density_g_cm3 = 1.21;
        f_tpu.nozzle_temperature_celsius = 225.0;
        f_tpu.bed_temperature_celsius = 50.0;
        f_tpu.fan_speed_percent = 50.0;
        f_tpu.min_fan_speed_percent = 25.0;
        f_tpu.max_fan_speed_percent = 70.0;
        f_tpu.pa_low_flow_s = 0.120;
        f_tpu.pa_high_flow_s = 0.085;

        let mut out = vec![f_pla, f_plaplus, f_petg, f_abs, f_asa, f_tpu];
        for entry in crate::color_optics::bambu_pla_cmykw_palette() {
            let mut fp = FilamentPreset::from_config(entry.name, &base_cfg, true);
            fp.filament_type = entry.filament_type.to_string();
            fp.filament_color_hex = entry.color_hex.to_string();
            fp.transmission_distance_mm = Some(entry.transmission_distance_mm);
            out.push(fp);
        }
        out
    }

    pub fn builtin_processes() -> Vec<ProcessPreset> {
        let base_cfg = SlicerConfig::default();
        let pr_std =
            ProcessPreset::from_config("0.20mm Standard Non-Planar", &base_cfg, true);

        let mut pr_fine =
            ProcessPreset::from_config("0.12mm Fine Conformal", &base_cfg, true);
        pr_fine.layer_height = LayerHeightConfig {
            target_mm: 0.12,
            min_mm: 0.05,
            max_mm: 0.20,
            first_layer_mm: 0.16,
            top_layer_mm: 0.12,
        };
        pr_fine.outer_wall_speed_mms = 50.0;
        pr_fine.inner_wall_speed_mms = 80.0;
        pr_fine.support_z_gap_mm = 0.15;

        let mut pr_draft =
            ProcessPreset::from_config("0.28mm Structural Draft", &base_cfg, true);
        pr_draft.layer_height = LayerHeightConfig {
            target_mm: 0.28,
            min_mm: 0.12,
            max_mm: 0.40,
            first_layer_mm: 0.24,
            top_layer_mm: 0.24,
        };
        pr_draft.shell_thickness_mm = 2.00;
        pr_draft.inner_wall_count = 3;
        pr_draft.infill_density_percent = 30.0;
        pr_draft.sparse_infill_pattern = SparseInfillPattern::Cubic;

        let mut pr_planar = ProcessPreset::from_config(
            "0.20mm Standard Planar (Height-Only)",
            &base_cfg,
            true,
        );
        pr_planar.order_field_mode = OrderFieldMode::HeightOnlyPlanar;

        let mut pr_tree = ProcessPreset::from_config(
            "0.20mm Non-Planar Tree Support (Bed Only)",
            &base_cfg,
            true,
        );
        pr_tree.support_style = SupportStyle::NonPlanarTree;
        pr_tree.support_on_bed_only = true;

        let mut pr_cmykw = ProcessPreset::from_config(
            "0.16mm CMYKW Full-Colour Optical Mixing",
            &base_cfg,
            true,
        );
        pr_cmykw.layer_height = LayerHeightConfig {
            target_mm: 0.16,
            min_mm: 0.08,
            max_mm: 0.24,
            first_layer_mm: 0.20,
            top_layer_mm: 0.14,
        };
        pr_cmykw.shell_thickness_mm = 1.60;
        pr_cmykw.inner_wall_count = 2;
        pr_cmykw.color_mixing.enabled = true;
        pr_cmykw.color_mixing.optical_shell_layers = 3;

        vec![pr_std, pr_fine, pr_draft, pr_planar, pr_tree, pr_cmykw]
    }

    pub fn presets_root_dir() -> Option<PathBuf> {
        let home = std::env::var_os("HOME").map(PathBuf::from)?;
        Some(home.join(".config").join("manifold2").join("presets"))
    }

    pub fn sanitize_filename_slug(name: &str) -> String {
        let slug: String = name
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        let trimmed = slug.trim_matches('_');
        if trimmed.is_empty() {
            "custom_preset".to_string()
        } else {
            trimmed.to_string()
        }
    }

    pub fn load_user_presets_from_disk(&mut self) {
        let Some(root) = Self::presets_root_dir() else {
            return;
        };

        let load_dir = |subdir: &str| -> Vec<PathBuf> {
            let dir = root.join(subdir);
            let Ok(entries) = std::fs::read_dir(dir) else {
                return Vec::new();
            };
            let mut paths: Vec<PathBuf> = entries
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("toml"))
                .collect();
            paths.sort();
            paths
        };

        for path in load_dir(PresetCategory::Printer.subdir_name()) {
            if let Ok(text) = std::fs::read_to_string(&path) {
                if let Ok(mut p) = toml::from_str::<PrinterPreset>(&text) {
                    p.is_builtin = false;
                    let has_explicit_protocol = toml::from_str::<toml::Value>(&text)
                        .ok()
                        .and_then(|v| v.get("network").and_then(|n| n.get("protocol")).cloned())
                        .is_some();
                    if !has_explicit_protocol {
                        p.network.protocol = PrinterNetworkProtocol::default_for_printer(
                            p.gcode_dialect,
                            &p.name,
                        );
                        if p.network.protocol == PrinterNetworkProtocol::PrusaLink
                            && p.network.username.trim().is_empty()
                        {
                            p.network.username = "maker".to_string();
                        }
                    }
                    if !p.name.trim().is_empty() {
                        self.printers
                            .retain(|existing| existing.is_builtin || existing.name != p.name);
                        self.printers.push(p);
                    }
                }
            }
        }
        for path in load_dir(PresetCategory::Tool.subdir_name()) {
            if let Ok(text) = std::fs::read_to_string(&path) {
                if let Ok(mut p) = toml::from_str::<ToolPreset>(&text) {
                    p.is_builtin = false;
                    if !p.name.trim().is_empty() {
                        self.tools
                            .retain(|existing| existing.is_builtin || existing.name != p.name);
                        self.tools.push(p);
                    }
                }
            }
        }
        for path in load_dir(PresetCategory::Filament.subdir_name()) {
            if let Ok(text) = std::fs::read_to_string(&path) {
                if let Ok(mut p) = toml::from_str::<FilamentPreset>(&text) {
                    p.is_builtin = false;
                    if !p.name.trim().is_empty() {
                        self.filaments
                            .retain(|existing| existing.is_builtin || existing.name != p.name);
                        self.filaments.push(p);
                    }
                }
            }
        }
        for path in load_dir(PresetCategory::Process.subdir_name()) {
            if let Ok(text) = std::fs::read_to_string(&path) {
                if let Ok(mut p) = toml::from_str::<ProcessPreset>(&text) {
                    p.is_builtin = false;
                    if !p.name.trim().is_empty() {
                        self.processes
                            .retain(|existing| existing.is_builtin || existing.name != p.name);
                        self.processes.push(p);
                    }
                }
            }
        }
    }

    pub fn selected_name(&self, category: PresetCategory) -> &str {
        match category {
            PresetCategory::Printer => &self.selected_printer,
            PresetCategory::Tool => &self.selected_tool,
            PresetCategory::Filament => &self.selected_filament,
            PresetCategory::Process => &self.selected_process,
        }
    }

    pub fn preset_names(&self, category: PresetCategory) -> Vec<(String, bool)> {
        match category {
            PresetCategory::Printer => self
                .printers
                .iter()
                .map(|p| (p.name.clone(), p.is_builtin))
                .collect(),
            PresetCategory::Tool => self
                .tools
                .iter()
                .map(|p| (p.name.clone(), p.is_builtin))
                .collect(),
            PresetCategory::Filament => self
                .filaments
                .iter()
                .map(|p| (p.name.clone(), p.is_builtin))
                .collect(),
            PresetCategory::Process => self
                .processes
                .iter()
                .map(|p| (p.name.clone(), p.is_builtin))
                .collect(),
        }
    }

    pub fn is_selected_builtin(&self, category: PresetCategory) -> bool {
        match category {
            PresetCategory::Printer => self
                .printers
                .iter()
                .find(|p| p.name == self.selected_printer)
                .map(|p| p.is_builtin)
                .unwrap_or(true),
            PresetCategory::Tool => self
                .tools
                .iter()
                .find(|p| p.name == self.selected_tool)
                .map(|p| p.is_builtin)
                .unwrap_or(true),
            PresetCategory::Filament => self
                .filaments
                .iter()
                .find(|p| p.name == self.selected_filament)
                .map(|p| p.is_builtin)
                .unwrap_or(true),
            PresetCategory::Process => self
                .processes
                .iter()
                .find(|p| p.name == self.selected_process)
                .map(|p| p.is_builtin)
                .unwrap_or(true),
        }
    }

    pub fn is_category_modified(&self, category: PresetCategory, cfg: &SlicerConfig) -> bool {
        match category {
            PresetCategory::Printer => self
                .printers
                .iter()
                .find(|p| p.name == self.selected_printer)
                .map(|p| !p.matches_config(cfg))
                .unwrap_or(true),
            PresetCategory::Tool => self
                .tools
                .iter()
                .find(|p| p.name == self.selected_tool)
                .map(|p| !p.matches_config(cfg))
                .unwrap_or(true),
            PresetCategory::Filament => self
                .filaments
                .iter()
                .find(|p| p.name == self.selected_filament)
                .map(|p| !p.matches_config(cfg))
                .unwrap_or(true),
            PresetCategory::Process => self
                .processes
                .iter()
                .find(|p| p.name == self.selected_process)
                .map(|p| !p.matches_config(cfg))
                .unwrap_or(true),
        }
    }

    pub fn select_and_apply(
        &mut self,
        category: PresetCategory,
        name: &str,
        cfg: &mut SlicerConfig,
    ) -> bool {
        match category {
            PresetCategory::Printer => {
                if let Some(p) = self.printers.iter().find(|p| p.name == name) {
                    p.apply_to_config(cfg);
                    self.selected_printer = p.name.clone();
                    return true;
                }
            }
            PresetCategory::Tool => {
                if let Some(p) = self.tools.iter().find(|p| p.name == name) {
                    p.apply_to_config(cfg);
                    self.selected_tool = p.name.clone();
                    return true;
                }
            }
            PresetCategory::Filament => {
                if let Some(p) = self.filaments.iter().find(|p| p.name == name) {
                    p.apply_to_config(cfg);
                    self.selected_filament = p.name.clone();
                    return true;
                }
            }
            PresetCategory::Process => {
                if let Some(p) = self.processes.iter().find(|p| p.name == name) {
                    p.apply_to_config(cfg);
                    self.selected_process = p.name.clone();
                    return true;
                }
            }
        }
        false
    }

    pub fn revert_selected(&mut self, category: PresetCategory, cfg: &mut SlicerConfig) -> bool {
        let name = self.selected_name(category).to_string();
        self.select_and_apply(category, &name, cfg)
    }

    pub fn save_user_preset(
        &mut self,
        category: PresetCategory,
        raw_name: &str,
        cfg: &mut SlicerConfig,
    ) -> Result<String, String> {
        let mut name = raw_name.trim().to_string();
        if name.is_empty() {
            return Err("Preset name cannot be empty.".to_string());
        }
        // Prevent overwriting a built-in preset name directly
        let collides_builtin = match category {
            PresetCategory::Printer => self.printers.iter().any(|p| p.is_builtin && p.name == name),
            PresetCategory::Tool => self.tools.iter().any(|p| p.is_builtin && p.name == name),
            PresetCategory::Filament => self
                .filaments
                .iter()
                .any(|p| p.is_builtin && p.name == name),
            PresetCategory::Process => self
                .processes
                .iter()
                .any(|p| p.is_builtin && p.name == name),
        };
        if collides_builtin {
            name = format!("{name} (Custom)");
        }

        let toml_text = match category {
            PresetCategory::Printer => {
                let preset = PrinterPreset::from_config(name.clone(), cfg, false);
                let s = toml::to_string_pretty(&preset).map_err(|e| e.to_string())?;
                if let Some(existing) = self
                    .printers
                    .iter_mut()
                    .find(|p| !p.is_builtin && p.name == name)
                {
                    *existing = preset;
                } else {
                    self.printers.push(preset);
                }
                self.selected_printer = name.clone();
                cfg.printer_preset_name = name.clone();
                s
            }
            PresetCategory::Tool => {
                let preset = ToolPreset::from_config(name.clone(), cfg, false);
                let s = toml::to_string_pretty(&preset).map_err(|e| e.to_string())?;
                if let Some(existing) = self
                    .tools
                    .iter_mut()
                    .find(|p| !p.is_builtin && p.name == name)
                {
                    *existing = preset;
                } else {
                    self.tools.push(preset);
                }
                self.selected_tool = name.clone();
                cfg.tool_preset_name = name.clone();
                s
            }
            PresetCategory::Filament => {
                let preset = FilamentPreset::from_config(name.clone(), cfg, false);
                let s = toml::to_string_pretty(&preset).map_err(|e| e.to_string())?;
                if let Some(existing) = self
                    .filaments
                    .iter_mut()
                    .find(|p| !p.is_builtin && p.name == name)
                {
                    *existing = preset;
                } else {
                    self.filaments.push(preset);
                }
                self.selected_filament = name.clone();
                cfg.filament_preset_name = name.clone();
                s
            }
            PresetCategory::Process => {
                let preset = ProcessPreset::from_config(name.clone(), cfg, false);
                let s = toml::to_string_pretty(&preset).map_err(|e| e.to_string())?;
                if let Some(existing) = self
                    .processes
                    .iter_mut()
                    .find(|p| !p.is_builtin && p.name == name)
                {
                    *existing = preset;
                } else {
                    self.processes.push(preset);
                }
                self.selected_process = name.clone();
                cfg.process_preset_name = name.clone();
                s
            }
        };

        if let Some(root) = Self::presets_root_dir() {
            let dir = root.join(category.subdir_name());
            let _ = std::fs::create_dir_all(&dir);
            let file_path = dir.join(format!("{}.toml", Self::sanitize_filename_slug(&name)));
            let _ = std::fs::write(file_path, toml_text);
        }

        Ok(name)
    }

    pub fn delete_selected_user_preset(
        &mut self,
        category: PresetCategory,
        cfg: &mut SlicerConfig,
    ) -> Result<String, String> {
        if self.is_selected_builtin(category) {
            return Err("Built-in factory presets cannot be deleted.".to_string());
        }
        let removed_name = self.selected_name(category).to_string();
        match category {
            PresetCategory::Printer => {
                self.printers
                    .retain(|p| p.is_builtin || p.name != removed_name);
                if let Some(first) = self.printers.first().cloned() {
                    self.selected_printer = first.name.clone();
                    first.apply_to_config(cfg);
                }
            }
            PresetCategory::Tool => {
                self.tools
                    .retain(|p| p.is_builtin || p.name != removed_name);
                if let Some(first) = self.tools.first().cloned() {
                    self.selected_tool = first.name.clone();
                    first.apply_to_config(cfg);
                }
            }
            PresetCategory::Filament => {
                self.filaments
                    .retain(|p| p.is_builtin || p.name != removed_name);
                if let Some(first) = self.filaments.first().cloned() {
                    self.selected_filament = first.name.clone();
                    first.apply_to_config(cfg);
                }
            }
            PresetCategory::Process => {
                self.processes
                    .retain(|p| p.is_builtin || p.name != removed_name);
                if let Some(first) = self.processes.first().cloned() {
                    self.selected_process = first.name.clone();
                    first.apply_to_config(cfg);
                }
            }
        }

        if let Some(root) = Self::presets_root_dir() {
            let file_path = root
                .join(category.subdir_name())
                .join(format!("{}.toml", Self::sanitize_filename_slug(&removed_name)));
            let _ = std::fs::remove_file(file_path);
        }

        Ok(removed_name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factory_presets_match_default_config_and_detect_modifications() {
        let mut cfg = SlicerConfig::default();
        let mut lib = PresetLibrary {
            printers: PresetLibrary::builtin_printers(),
            tools: PresetLibrary::builtin_tools(),
            filaments: PresetLibrary::builtin_filaments(),
            processes: PresetLibrary::builtin_processes(),
            selected_printer: "256³ CoreXY (Standard Klipper)".to_string(),
            selected_tool: "0.4mm Standard 5/6-Axis Conical".to_string(),
            selected_filament: "Generic PLA".to_string(),
            selected_process: "0.20mm Standard Non-Planar".to_string(),
            left_tab: LeftPanelTab::ProjectSetup,
            config_subtab: PresetCategory::Process,
            show_left_panel: true,
            show_view_panel: true,
            save_as_open_for: None,
            save_as_name_buf: String::new(),
        };

        // Initially, all 4 default factory presets match SlicerConfig::default() with 0 modifications
        for cat in PresetCategory::ALL {
            assert!(
                !lib.is_category_modified(cat, &cfg),
                "Expected default {cat:?} preset to match SlicerConfig::default()"
            );
        }

        // Modifying a printer field marks only Printer as modified
        cfg.bed.width_x = 310.0;
        assert!(lib.is_category_modified(PresetCategory::Printer, &cfg));
        assert!(!lib.is_category_modified(PresetCategory::Tool, &cfg));
        assert!(!lib.is_category_modified(PresetCategory::Filament, &cfg));
        assert!(!lib.is_category_modified(PresetCategory::Process, &cfg));

        // Reverting Printer restores the factory value
        assert!(lib.revert_selected(PresetCategory::Printer, &mut cfg));
        assert!((cfg.bed.width_x - 256.0).abs() < 1.0e-4);
        assert!(!lib.is_category_modified(PresetCategory::Printer, &cfg));

        // Dynamic PA emission & SCV Corner Flow Compensation belong to Printer, not Filament
        cfg.gcode.enable_dynamic_pa = false;
        cfg.gcode.enable_corner_flow_comp = false;
        assert!(lib.is_category_modified(PresetCategory::Printer, &cfg));
        assert!(!lib.is_category_modified(PresetCategory::Filament, &cfg));
        assert!(lib.revert_selected(PresetCategory::Printer, &mut cfg));
        assert!(cfg.gcode.enable_dynamic_pa);
        assert!(cfg.gcode.enable_corner_flow_comp);
        assert!(!lib.is_category_modified(PresetCategory::Printer, &cfg));

        // Switching to a built-in Filament preset updates temperatures, density, and cooling
        assert!(lib.select_and_apply(PresetCategory::Filament, "Generic PETG", &mut cfg));
        assert_eq!(cfg.gcode.filament_type, "PETG");
        assert!((cfg.gcode.bed_temperature_celsius - 75.0).abs() < 1.0e-4);
        assert!(!lib.is_category_modified(PresetCategory::Filament, &cfg));

        // Switching to a built-in Tool preset updates nozzle diameter and auto voxel size
        assert!(lib.select_and_apply(PresetCategory::Tool, "0.6mm High-Flow", &mut cfg));
        assert!((cfg.active_tool().unwrap().nozzle_diameter_mm - 0.60).abs() < 1.0e-4);
        assert!(!lib.is_category_modified(PresetCategory::Tool, &cfg));

        // Switching to a built-in Process preset updates layer heights
        assert!(lib.select_and_apply(PresetCategory::Process, "0.12mm Fine Conformal", &mut cfg));
        assert!((cfg.slicing.layer_height.target_mm - 0.12).abs() < 1.0e-4);
        assert!(!lib.is_category_modified(PresetCategory::Process, &cfg));

        // Switching to the Prusa CORE One & INDX printer presets applies Prusa dialect, PrusaLink protocol, build volumes, and tool counts
        let prusa_expected: [(&str, f32, f32, f32, usize); 6] = [
            ("Prusa CORE One", 250.0, 220.0, 270.0, 1),
            ("Prusa CORE One L", 300.0, 300.0, 330.0, 1),
            ("Prusa CORE One INDX (4-Tool)", 248.0, 205.0, 270.0, 4),
            ("Prusa CORE One INDX (8-Tool)", 248.0, 205.0, 270.0, 8),
            ("Prusa CORE One L INDX (4-Tool)", 298.0, 275.0, 330.0, 4),
            ("Prusa CORE One L INDX (8-Tool)", 298.0, 275.0, 330.0, 8),
        ];
        for (name, wx, dy, hz, expected_tools) in prusa_expected {
            assert!(
                lib.select_and_apply(PresetCategory::Printer, name, &mut cfg),
                "Missing built-in printer preset {name}"
            );
            assert_eq!(cfg.bed.shape, BedShape::Rectangular);
            assert_eq!(cfg.gcode.gcode_dialect, GcodeDialect::Prusa);
            assert_eq!(cfg.network.protocol, PrinterNetworkProtocol::PrusaLink);
            assert_eq!(cfg.network.username, "maker");
            assert!((cfg.bed.width_x - wx).abs() < 1.0e-4);
            assert!((cfg.bed.depth_y - dy).abs() < 1.0e-4);
            assert!((cfg.bed.height_z - hz).abs() < 1.0e-4);
            assert!(!cfg.gcode.gcode_origin_at_bed_center);
            assert!(cfg.gcode.start_gcode_template.contains("M555"));
            assert!(cfg.gcode.start_gcode_template.contains("G29"));
            assert_eq!(cfg.tools.len(), expected_tools);
            assert!(!lib.is_category_modified(PresetCategory::Printer, &cfg));
        }

        // Switching from Prusa back to any Voron preset restores Klipper dialect and Moonraker network protocol
        let voron_expected = [
            ("Voron 0", 120.0, 120.0, 120.0),
            ("Voron Trident 250", 250.0, 250.0, 250.0),
            ("Voron Trident 300", 300.0, 300.0, 250.0),
            ("Voron Trident 350", 350.0, 350.0, 250.0),
            ("Voron 2.4 250", 250.0, 250.0, 220.0),
            ("Voron 2.4 300", 300.0, 300.0, 280.0),
            ("Voron 2.4 350", 350.0, 350.0, 330.0),
        ];
        for (name, wx, dy, hz) in voron_expected {
            assert!(
                lib.select_and_apply(PresetCategory::Printer, name, &mut cfg),
                "Missing built-in printer preset {name}"
            );
            assert_eq!(cfg.bed.shape, BedShape::Rectangular);
            assert_eq!(cfg.gcode.gcode_dialect, GcodeDialect::Klipper);
            assert_eq!(cfg.network.protocol, PrinterNetworkProtocol::Moonraker);
            assert!((cfg.bed.width_x - wx).abs() < 1.0e-4);
            assert!((cfg.bed.depth_y - dy).abs() < 1.0e-4);
            assert!((cfg.bed.height_z - hz).abs() < 1.0e-4);
            assert!(!cfg.gcode.gcode_origin_at_bed_center);
            assert!(!lib.is_category_modified(PresetCategory::Printer, &cfg));
        }

        // Verify all 5 Prusa Nextruder & Bondtech INDX toolhead presets and their XZ clearance envelopes
        let prusa_tools = [
            ("0.4mm Prusa Nextruder Standard", 0.40, 16.0),
            ("0.4mm Prusa Nextruder High-Flow", 0.40, 26.0),
            ("0.6mm Prusa Nextruder High-Flow", 0.60, 32.0),
            ("0.4mm Bondtech INDX Toolchanger", 0.40, 24.0),
            ("0.6mm Bondtech INDX Toolchanger", 0.60, 30.0),
        ];
        for (t_name, expected_diam, expected_flow) in prusa_tools {
            assert!(
                lib.select_and_apply(PresetCategory::Tool, t_name, &mut cfg),
                "Missing built-in Prusa tool preset {t_name}"
            );
            assert!((cfg.active_tool().unwrap().nozzle_diameter_mm - expected_diam).abs() < 1.0e-4);
            assert!((cfg.gcode.max_volumetric_flow_mm3s - expected_flow).abs() < 1.0e-4);
            let prof = &cfg.active_tool().unwrap().clearance_profile_xz;
            assert!(prof.len() >= 6);
            assert!(!lib.is_category_modified(PresetCategory::Tool, &cfg));
        }

        // Switching to Bambu Lab printer presets applies BambuLab dialect, BambuLan protocol, build volumes, and AMS tool counts
        let bambu_expected: [(&str, f32, f32, f32, usize); 7] = [
            ("Bambu Lab X1 Carbon", 256.0, 256.0, 256.0, 1),
            ("Bambu Lab X1 Carbon (AMS 4-Slot)", 256.0, 256.0, 256.0, 4),
            ("Bambu Lab P1S / P1P", 256.0, 256.0, 256.0, 1),
            ("Bambu Lab P1S (AMS 4-Slot)", 256.0, 256.0, 256.0, 4),
            ("Bambu Lab A1", 256.0, 256.0, 256.0, 1),
            ("Bambu Lab A1 (AMS Lite 4-Slot)", 256.0, 256.0, 256.0, 4),
            ("Bambu Lab A1 Mini", 180.0, 180.0, 180.0, 1),
        ];
        for (name, wx, dy, hz, expected_tools) in bambu_expected {
            assert!(
                lib.select_and_apply(PresetCategory::Printer, name, &mut cfg),
                "Missing built-in printer preset {name}"
            );
            assert_eq!(cfg.bed.shape, BedShape::Rectangular);
            assert_eq!(cfg.gcode.gcode_dialect, GcodeDialect::BambuLab);
            assert_eq!(cfg.network.protocol, PrinterNetworkProtocol::BambuLan);
            assert_eq!(cfg.network.username, "bblp");
            assert!((cfg.bed.width_x - wx).abs() < 1.0e-4);
            assert!((cfg.bed.depth_y - dy).abs() < 1.0e-4);
            assert!((cfg.bed.height_z - hz).abs() < 1.0e-4);
            assert_eq!(cfg.tools.len(), expected_tools);
            assert!(!lib.is_category_modified(PresetCategory::Printer, &cfg));
        }

        assert!(lib.select_and_apply(
            PresetCategory::Tool,
            "0.4mm Bambu Lab X1/P1 Hardened Steel",
            &mut cfg
        ));
        assert!(!lib.is_category_modified(PresetCategory::Tool, &cfg));

        // Verify all 8 Voron toolhead presets (Stealthburner, DownDraft, Xol, A4T with & without +8.5mm MZE)
        let voron_tools = [
            ("0.4mm Voron Stealthburner (Standard)", 24.0),
            ("0.4mm Voron Stealthburner (+8.5mm MZE / UHF)", 35.0),
            ("0.4mm Voron DownDraft (Standard / Low-Mount)", 26.0),
            ("0.4mm Voron DownDraft (+8.5mm MZE / UHF)", 38.0),
            ("0.4mm Voron Xol (Standard)", 28.0),
            ("0.4mm Voron Xol (+8.5mm MZE / UHF)", 38.0),
            ("0.4mm Voron A4T (Standard)", 28.0),
            ("0.4mm Voron A4T (+8.5mm MZE / UHF)", 38.0),
        ];
        for (t_name, expected_flow) in voron_tools {
            assert!(
                lib.select_and_apply(PresetCategory::Tool, t_name, &mut cfg),
                "Missing built-in Voron tool preset {t_name}"
            );
            assert!((cfg.gcode.max_volumetric_flow_mm3s - expected_flow).abs() < 1.0e-4);
            let prof = &cfg.active_tool().unwrap().clearance_profile_xz;
            assert!(prof.len() >= 5);
            for w in prof.windows(2) {
                assert!(
                    w[1].z_height_mm >= w[0].z_height_mm,
                    "Clearance profile Z must be monotonically non-decreasing in {t_name}"
                );
            }
            assert!(!lib.is_category_modified(PresetCategory::Tool, &cfg));
        }
    }

    #[test]
    fn test_preset_toml_roundtrip_all_categories() {
        let cfg = SlicerConfig::default();

        let printer = PrinterPreset::from_config("Test Printer", &cfg, false);
        let printer_toml = toml::to_string_pretty(&printer).unwrap();
        let printer_rt: PrinterPreset = toml::from_str(&printer_toml).unwrap();
        assert_eq!(printer, printer_rt);

        let tool = ToolPreset::from_config("Test Tool", &cfg, false);
        let tool_toml = toml::to_string_pretty(&tool).unwrap();
        let tool_rt: ToolPreset = toml::from_str(&tool_toml).unwrap();
        assert_eq!(tool, tool_rt);

        let filament = FilamentPreset::from_config("Test Filament", &cfg, false);
        let filament_toml = toml::to_string_pretty(&filament).unwrap();
        let filament_rt: FilamentPreset = toml::from_str(&filament_toml).unwrap();
        assert_eq!(filament, filament_rt);

        let process = ProcessPreset::from_config("Test Process", &cfg, false);
        let process_toml = toml::to_string_pretty(&process).unwrap();
        let process_rt: ProcessPreset = toml::from_str(&process_toml).unwrap();
        assert_eq!(process, process_rt);
    }
}
