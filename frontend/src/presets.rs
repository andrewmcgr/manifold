use std::path::PathBuf;

use bevy::prelude::*;
use serde::{Deserialize, Serialize};

use crate::config::{
    BedShape, ClearancePointXZ, ClearanceProfileMode, LayerHeightConfig, LineWidthConfig,
    OrderFieldMode, SlicerConfig, SolidInfillPattern, SparseInfillPattern, SupportStyle,
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
    pub nozzle_axis_bed: [f32; 3],
    pub gravity_direction_bed: [f32; 3],
    pub bed_tilt_rx_deg: f32,
    pub bed_tilt_ry_deg: f32,

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
    pub emit_exclude_object: bool,
    pub emit_thumbnails: bool,
    pub start_gcode_template: String,
    pub end_gcode_template: String,
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
            nozzle_axis_bed: cfg.machine.nozzle_axis_bed,
            gravity_direction_bed: cfg.machine.gravity_direction_bed,
            bed_tilt_rx_deg: cfg.machine.bed_tilt_rx_deg,
            bed_tilt_ry_deg: cfg.machine.bed_tilt_ry_deg,
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
            emit_exclude_object: cfg.gcode.emit_exclude_object,
            emit_thumbnails: cfg.gcode.emit_thumbnails,
            start_gcode_template: cfg.gcode.start_gcode_template.clone(),
            end_gcode_template: cfg.gcode.end_gcode_template.clone(),
        }
    }

    pub fn apply_to_config(&self, cfg: &mut SlicerConfig) {
        cfg.printer_preset_name = self.name.clone();
        cfg.bed.shape = self.bed_shape;
        cfg.bed.width_x = self.width_x;
        cfg.bed.depth_y = self.depth_y;
        cfg.bed.height_z = self.height_z;
        cfg.gcode.gcode_origin_at_bed_center = self.gcode_origin_at_bed_center;
        cfg.machine.nozzle_axis_bed = self.nozzle_axis_bed;
        cfg.machine.gravity_direction_bed = self.gravity_direction_bed;
        cfg.machine.bed_tilt_rx_deg = self.bed_tilt_rx_deg;
        cfg.machine.bed_tilt_ry_deg = self.bed_tilt_ry_deg;
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
        cfg.gcode.emit_exclude_object = self.emit_exclude_object;
        cfg.gcode.emit_thumbnails = self.emit_thumbnails;
        cfg.gcode.start_gcode_template = self.start_gcode_template.clone();
        cfg.gcode.end_gcode_template = self.end_gcode_template.clone();
        cfg.gcode.sanitize();
    }

    pub fn matches_config(&self, cfg: &SlicerConfig) -> bool {
        self.bed_shape == cfg.bed.shape
            && approx_eq(self.width_x, cfg.bed.width_x)
            && approx_eq(self.depth_y, cfg.bed.depth_y)
            && approx_eq(self.height_z, cfg.bed.height_z)
            && self.gcode_origin_at_bed_center == cfg.gcode.gcode_origin_at_bed_center
            && approx_eq_vec3(self.nozzle_axis_bed, cfg.machine.nozzle_axis_bed)
            && approx_eq_vec3(self.gravity_direction_bed, cfg.machine.gravity_direction_bed)
            && approx_eq(self.bed_tilt_rx_deg, cfg.machine.bed_tilt_rx_deg)
            && approx_eq(self.bed_tilt_ry_deg, cfg.machine.bed_tilt_ry_deg)
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
            && self.emit_exclude_object == cfg.gcode.emit_exclude_object
            && self.emit_thumbnails == cfg.gcode.emit_thumbnails
            && self.start_gcode_template == cfg.gcode.start_gcode_template
            && self.end_gcode_template == cfg.gcode.end_gcode_template
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

    // Non-Newtonian Rheology, Dynamic Pressure Advance, Die Swell & Ooze Relaxation
    pub enable_dynamic_pa: bool,
    pub enable_slicer_pa_subdivision: bool,
    pub pa_low_flow_s: f32,
    pub pa_high_flow_s: f32,
    pub pa_ref_low_flow_mm3s: f32,
    pub pa_ref_high_flow_mm3s: f32,
    pub swell_low: f32,
    pub swell_high: f32,
    pub enable_corner_flow_comp: bool,
    pub corner_flow_comp_ratio: f32,
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
        let nozzle_temp = cfg
            .active_tool()
            .map(|t| t.temperature_celsius)
            .unwrap_or(215.0);
        Self {
            name: name.into(),
            is_builtin,
            filament_type: cfg.gcode.filament_type.clone(),
            filament_diameter_mm: cfg.gcode.filament_diameter_mm,
            filament_density_g_cm3: cfg.gcode.filament_density_g_cm3,
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
            enable_dynamic_pa: cfg.gcode.enable_dynamic_pa,
            enable_slicer_pa_subdivision: cfg.gcode.enable_slicer_pa_subdivision,
            pa_low_flow_s: cfg.gcode.pa_low_flow_s,
            pa_high_flow_s: cfg.gcode.pa_high_flow_s,
            pa_ref_low_flow_mm3s: cfg.gcode.pa_ref_low_flow_mm3s,
            pa_ref_high_flow_mm3s: cfg.gcode.pa_ref_high_flow_mm3s,
            swell_low: cfg.gcode.swell_low,
            swell_high: cfg.gcode.swell_high,
            enable_corner_flow_comp: cfg.gcode.enable_corner_flow_comp,
            corner_flow_comp_ratio: cfg.gcode.corner_flow_comp_ratio,
            enable_transient_pressure_comp: cfg.gcode.enable_transient_pressure_comp,
            transient_pressure_exponent: cfg.gcode.transient_pressure_exponent,
            transient_min_flow_multiplier: cfg.gcode.transient_min_flow_multiplier,
            transient_pressure_tau_s: cfg.gcode.transient_pressure_tau_s,
            ooze_relaxation_extra_prime_mm: cfg.gcode.ooze_relaxation_extra_prime_mm,
            ooze_relaxation_tau_s: cfg.gcode.ooze_relaxation_tau_s,
        }
    }

    pub fn apply_to_config(&self, cfg: &mut SlicerConfig) {
        cfg.filament_preset_name = self.name.clone();
        if let Some(tool) = cfg.tools.get_mut(cfg.active_tool_index) {
            tool.temperature_celsius = self.nozzle_temperature_celsius;
        } else if let Some(tool) = cfg.tools.first_mut() {
            tool.temperature_celsius = self.nozzle_temperature_celsius;
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
        cfg.gcode.enable_dynamic_pa = self.enable_dynamic_pa;
        cfg.gcode.enable_slicer_pa_subdivision = self.enable_slicer_pa_subdivision;
        cfg.gcode.pa_low_flow_s = self.pa_low_flow_s;
        cfg.gcode.pa_high_flow_s = self.pa_high_flow_s;
        cfg.gcode.pa_ref_low_flow_mm3s = self.pa_ref_low_flow_mm3s;
        cfg.gcode.pa_ref_high_flow_mm3s = self.pa_ref_high_flow_mm3s;
        cfg.gcode.swell_low = self.swell_low;
        cfg.gcode.swell_high = self.swell_high;
        cfg.gcode.enable_corner_flow_comp = self.enable_corner_flow_comp;
        cfg.gcode.corner_flow_comp_ratio = self.corner_flow_comp_ratio;
        cfg.gcode.enable_transient_pressure_comp = self.enable_transient_pressure_comp;
        cfg.gcode.transient_pressure_exponent = self.transient_pressure_exponent;
        cfg.gcode.transient_min_flow_multiplier = self.transient_min_flow_multiplier;
        cfg.gcode.transient_pressure_tau_s = self.transient_pressure_tau_s;
        cfg.gcode.ooze_relaxation_extra_prime_mm = self.ooze_relaxation_extra_prime_mm;
        cfg.gcode.ooze_relaxation_tau_s = self.ooze_relaxation_tau_s;
        cfg.gcode.sanitize();
    }

    pub fn matches_config(&self, cfg: &SlicerConfig) -> bool {
        let nozzle_temp = cfg
            .active_tool()
            .map(|t| t.temperature_celsius)
            .unwrap_or(215.0);
        self.filament_type == cfg.gcode.filament_type
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
            && self.enable_dynamic_pa == cfg.gcode.enable_dynamic_pa
            && self.enable_slicer_pa_subdivision == cfg.gcode.enable_slicer_pa_subdivision
            && approx_eq(self.pa_low_flow_s, cfg.gcode.pa_low_flow_s)
            && approx_eq(self.pa_high_flow_s, cfg.gcode.pa_high_flow_s)
            && approx_eq(self.pa_ref_low_flow_mm3s, cfg.gcode.pa_ref_low_flow_mm3s)
            && approx_eq(self.pa_ref_high_flow_mm3s, cfg.gcode.pa_ref_high_flow_mm3s)
            && approx_eq(self.swell_low, cfg.gcode.swell_low)
            && approx_eq(self.swell_high, cfg.gcode.swell_high)
            && self.enable_corner_flow_comp == cfg.gcode.enable_corner_flow_comp
            && approx_eq(self.corner_flow_comp_ratio, cfg.gcode.corner_flow_comp_ratio)
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
    pub wall_print_order: WallPrintOrder,
    pub outer_wall_width: LineWidthConfig,
    pub inner_wall_width: LineWidthConfig,
    pub solid_infill_width: LineWidthConfig,
    pub solid_infill_pattern: SolidInfillPattern,
    pub sparse_infill_width: LineWidthConfig,
    pub sparse_infill_pattern: SparseInfillPattern,
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

    // 3D Conformal Scarf Seam
    pub enable_scarf_joint: bool,
    pub scarf_length_mm: f32,
    pub scarf_start_height_ratio: f32,
    pub seam_gap_mm: f32,

    // Feature Print Speeds (mm/s)
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
            wall_print_order: cfg.gcode.wall_print_order,
            outer_wall_width: cfg.slicing.outer_wall_width,
            inner_wall_width: cfg.slicing.inner_wall_width,
            solid_infill_width: cfg.slicing.solid_infill_width,
            solid_infill_pattern: cfg.slicing.solid_infill_pattern,
            sparse_infill_width: cfg.slicing.sparse_infill_width,
            sparse_infill_pattern: cfg.slicing.sparse_infill_pattern,
            infill_density_percent: cfg.slicing.infill_density_percent,
            infill_angle_deg: cfg.slicing.infill_angle_deg,
            min_extrusion_length_mm: cfg.slicing.min_extrusion_length_mm,
            overhang_angle_threshold_deg: cfg.slicing.overhang_angle_threshold_deg,
            max_bridge_length_mm: cfg.slicing.max_bridge_length_mm,
            wave_overhang_overlap_fraction: cfg.slicing.wave_overhang_overlap_fraction,
            enable_support_generation: cfg.slicing.enable_support_generation,
            support_tool_index: cfg.slicing.support_tool_index,
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
        cfg.gcode.wall_print_order = self.wall_print_order;
        cfg.slicing.outer_wall_width = self.outer_wall_width;
        cfg.slicing.inner_wall_width = self.inner_wall_width;
        cfg.slicing.solid_infill_width = self.solid_infill_width;
        cfg.slicing.solid_infill_pattern = self.solid_infill_pattern;
        cfg.slicing.sparse_infill_width = self.sparse_infill_width;
        cfg.slicing.sparse_infill_pattern = self.sparse_infill_pattern;
        cfg.slicing.infill_density_percent = self.infill_density_percent;
        cfg.slicing.infill_angle_deg = self.infill_angle_deg;
        cfg.slicing.min_extrusion_length_mm = self.min_extrusion_length_mm;
        cfg.slicing.overhang_angle_threshold_deg = self.overhang_angle_threshold_deg;
        cfg.slicing.max_bridge_length_mm = self.max_bridge_length_mm;
        cfg.slicing.wave_overhang_overlap_fraction = self.wave_overhang_overlap_fraction;
        cfg.slicing.enable_support_generation = self.enable_support_generation;
        cfg.slicing.support_tool_index = self.support_tool_index;
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

        cfg.gcode.enable_scarf_joint = self.enable_scarf_joint;
        cfg.gcode.scarf_length_mm = self.scarf_length_mm;
        cfg.gcode.scarf_start_height_ratio = self.scarf_start_height_ratio;
        cfg.gcode.seam_gap_mm = self.seam_gap_mm;
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
            && self.wall_print_order == cfg.gcode.wall_print_order
            && approx_eq_line_width(self.outer_wall_width, cfg.slicing.outer_wall_width)
            && approx_eq_line_width(self.inner_wall_width, cfg.slicing.inner_wall_width)
            && approx_eq_line_width(self.solid_infill_width, cfg.slicing.solid_infill_width)
            && self.solid_infill_pattern == cfg.slicing.solid_infill_pattern
            && approx_eq_line_width(self.sparse_infill_width, cfg.slicing.sparse_infill_width)
            && self.sparse_infill_pattern == cfg.slicing.sparse_infill_pattern
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

        let mut p_5axis =
            PrinterPreset::from_config("5-Axis Tilting Build Plate (250×210)", &base_cfg, true);
        p_5axis.width_x = 250.0;
        p_5axis.depth_y = 210.0;
        p_5axis.height_z = 220.0;
        p_5axis.gcode_origin_at_bed_center = true;
        p_5axis.outer_wall_accel_mms2 = 2000.0;
        p_5axis.inner_wall_accel_mms2 = 3500.0;
        p_5axis.infill_accel_mms2 = 5000.0;
        p_5axis.travel_accel_mms2 = 7500.0;

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

        vec![p_corexy, p_5axis, p_6axis]
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

        vec![t_04, t_06, t_025, t_rect]
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
        f_asa.bridge_fan_speed_percent = 50.0;
        f_asa.wave_overhang_fan_speed_percent = 45.0;

        let mut f_tpu = FilamentPreset::from_config("TPU 95A", &base_cfg, true);
        f_tpu.filament_type = "TPU".to_string();
        f_tpu.filament_density_g_cm3 = 1.21;
        f_tpu.nozzle_temperature_celsius = 225.0;
        f_tpu.bed_temperature_celsius = 50.0;
        f_tpu.fan_speed_percent = 50.0;
        f_tpu.pa_low_flow_s = 0.120;
        f_tpu.pa_high_flow_s = 0.085;

        vec![f_pla, f_plaplus, f_petg, f_abs, f_asa, f_tpu]
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

        vec![pr_std, pr_fine, pr_draft, pr_planar, pr_tree]
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
