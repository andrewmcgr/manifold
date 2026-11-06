use bevy::prelude::*;
use serde::{Deserialize, Serialize};

use crate::config::{NozzleFrame, SlicerConfig};
use crate::gcode::{
    compute_exclude_object_polygon_2d, expand_gcode_template, format_exclude_object_define,
    format_moonraker_estimated_time, render_toolpath_preview_thumbnails, sanitize_exclude_object_name,
    PlannedGcodeResult, PlannedLayerGcode, PlannedMoveKind, PlannedSegment,
};

/// Top-level calibration utility category.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CalibrationUtilityKind {
    /// Orca-style surface quality coupon array for Extrusion Multiplier (Pass 1 Coarse, Pass 2 Fine)
    /// and Pass 3 High-Flow Viscoelastic Die-Swell (`gcode.swell_high`).
    #[default]
    ExtrusionMultiplier,
    /// Ellis-style 90-degree sharp-corner chevron pattern for Pressure Advance
    /// (Single-Band or Dual-Band Low+High Shear-Thinning Flow).
    PressureAdvance,
    /// Constant-velocity cross-section step-down & short-move staccato striation pattern
    /// for Melt-Zone Transient Pressure Compensation (`transient_pressure_exponent` γ and `transient_pressure_tau_s` τ_melt).
    TransientPressure,
}

impl CalibrationUtilityKind {
    pub const ALL: [Self; 3] = [
        Self::ExtrusionMultiplier,
        Self::PressureAdvance,
        Self::TransientPressure,
    ];

    pub const fn label(self) -> &'static str {
        match self {
            Self::ExtrusionMultiplier => "Extrusion Multiplier & Die Swell (Orca Surface Coupons)",
            Self::PressureAdvance => "Pressure Advance (Ellis 90° Chevron Pattern)",
            Self::TransientPressure => "Transient Melt-Zone Pressure (Flow-Step Striations)",
        }
    }

    pub const fn short_label(self) -> &'static str {
        match self {
            Self::ExtrusionMultiplier => "EM & Die Swell",
            Self::PressureAdvance => "Pressure Advance (PA)",
            Self::TransientPressure => "Transient Pressure",
        }
    }
}

/// Pass selection for Orca-style Extrusion Multiplier & Die-Swell calibration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum EmCalibrationPass {
    /// Pass 1 (Coarse): Sweeps relative EM % (default `-10%` to `+10%` in `2.5%` steps, 9 coupons).
    #[default]
    Pass1Coarse,
    /// Pass 2 (Fine): Sweeps relative EM % (default `-4%` to `+4%` in `1.0%` steps, 9 coupons).
    Pass2Fine,
    /// Pass 3 (High-Flow Die Swell): Sweeps `gcode.swell_high` (default `1.00` to `1.18` in `0.03` steps, 7 coupons)
    /// while printing the isolated top monotonic patch at `pa_ref_high_flow_mm3s`.
    Pass3HighFlowDieSwell,
}

impl EmCalibrationPass {
    pub const ALL: [Self; 3] = [
        Self::Pass1Coarse,
        Self::Pass2Fine,
        Self::Pass3HighFlowDieSwell,
    ];

    pub const fn label(self) -> &'static str {
        match self {
            Self::Pass1Coarse => "Pass 1: Coarse EM (-10% to +10%, step 2.5%)",
            Self::Pass2Fine => "Pass 2: Fine EM (-4% to +4%, step 1.0%)",
            Self::Pass3HighFlowDieSwell => "Pass 3: High-Flow Die Swell (swell_high @ Q_high)",
        }
    }

    pub const fn short_label(self) -> &'static str {
        match self {
            Self::Pass1Coarse => "Pass 1 (Coarse EM)",
            Self::Pass2Fine => "Pass 2 (Fine EM)",
            Self::Pass3HighFlowDieSwell => "Pass 3 (High-Flow Swell)",
        }
    }
}

/// Mode selection for Ellis-style Pressure Advance chevron calibration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PaCalibrationMode {
    /// Dual-Band Shear-Thinning Mode: Prints two Ellis chevron banks side-by-side in one job:
    /// Bank A at `pa_ref_low_flow_mm3s` (for `pa_low_flow_s`) and Bank B at `pa_ref_high_flow_mm3s` (for `pa_high_flow_s`).
    #[default]
    DualBandShearThinning,
    /// Classic Single-Band Mode: Prints one Ellis chevron bank at user-configured `fast_speed_mms`.
    SingleBand,
}

impl PaCalibrationMode {
    pub const ALL: [Self; 2] = [Self::DualBandShearThinning, Self::SingleBand];

    pub const fn label(self) -> &'static str {
        match self {
            Self::DualBandShearThinning => "Dual-Band Shear-Thinning (k_low @ Q_low + k_high @ Q_high)",
            Self::SingleBand => "Single-Band Classic Ellis Pattern",
        }
    }
}

/// Pass selection for Transient Melt-Zone Pressure Compensation calibration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum TransientCalibrationPass {
    /// Pass 1: Sweeps `transient_pressure_exponent` γ (default `0.00` to `0.70`, step `0.10`) at fixed `transient_pressure_tau_s`.
    #[default]
    Pass1ExponentGamma,
    /// Pass 2: Sweeps `transient_pressure_tau_s` τ_melt (default `0.05 s` to `0.65 s`, step `0.10 s`) at fixed `transient_pressure_exponent`.
    Pass2RelaxationTau,
}

impl TransientCalibrationPass {
    pub const ALL: [Self; 2] = [Self::Pass1ExponentGamma, Self::Pass2RelaxationTau];

    pub const fn label(self) -> &'static str {
        match self {
            Self::Pass1ExponentGamma => "Pass 1: Sweep Exponent γ (0.00 .. 0.70) @ Fixed τ_melt",
            Self::Pass2RelaxationTau => "Pass 2: Sweep Relaxation Time τ_melt (0.05s .. 0.65s) @ Fixed γ",
        }
    }
}

/// Parameters for Orca-style Extrusion Multiplier & High-Flow Die-Swell coupon generation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EmCalibrationSpec {
    pub pass: EmCalibrationPass,
    /// Optional index into `SlicerConfig::tools` (`None` = use `config.active_tool_index`).
    #[serde(default)]
    pub tool_index: Option<usize>,
    /// Baseline extrusion multiplier around which Pass 1 and Pass 2 step percentages are evaluated.
    pub base_extrusion_multiplier: f32,
    /// Starting relative variation (%) for Pass 1 / Pass 2 (e.g. `-10.0` or `-4.0`).
    pub start_delta_pct: f32,
    /// Ending relative variation (%) for Pass 1 / Pass 2 (e.g. `+10.0` or `+4.0`).
    pub end_delta_pct: f32,
    /// Step increment (%) for Pass 1 / Pass 2 (e.g. `2.5` or `1.0`).
    pub step_delta_pct: f32,
    /// Starting `swell_high` ratio for Pass 3 (`>= 1.00`, default `1.00`).
    pub start_swell_high: f32,
    /// Ending `swell_high` ratio for Pass 3 (default `1.18`).
    pub end_swell_high: f32,
    /// Step increment for `swell_high` in Pass 3 (default `0.03`).
    pub step_swell_high: f32,
    /// Coupon body width along local X (`travel_axis`, mm).
    pub coupon_width_mm: f32,
    /// Coupon body depth along local Y (`lateral_axis`, mm).
    pub coupon_depth_mm: f32,
    /// Number of dense monotonic top test layers per coupon (`>= 3`).
    pub top_test_layers: u32,
    /// Neutralize die-swell scaling during Pass 1 & Pass 2 so baseline EM is isolated at reference speed.
    pub isolate_die_swell_in_em_passes: bool,
    /// User-selected winning step value in the UI calculator (`%` for Pass 1/2, `swell_high` for Pass 3).
    pub selected_winner_value: f32,
}

impl Default for EmCalibrationSpec {
    fn default() -> Self {
        Self {
            pass: EmCalibrationPass::Pass1Coarse,
            tool_index: None,
            base_extrusion_multiplier: 1.00,
            start_delta_pct: -10.0,
            end_delta_pct: 10.0,
            step_delta_pct: 2.5,
            start_swell_high: 1.00,
            end_swell_high: 1.18,
            step_swell_high: 0.03,
            coupon_width_mm: 26.0,
            coupon_depth_mm: 18.0,
            top_test_layers: 4,
            isolate_die_swell_in_em_passes: true,
            selected_winner_value: 0.0,
        }
    }
}

impl EmCalibrationSpec {
    pub fn apply_pass_defaults(&mut self, pass: EmCalibrationPass, current_em: f32) {
        self.pass = pass;
        self.base_extrusion_multiplier = current_em.clamp(0.4, 1.8);
        match pass {
            EmCalibrationPass::Pass1Coarse => {
                self.start_delta_pct = -10.0;
                self.end_delta_pct = 10.0;
                self.step_delta_pct = 2.5;
                self.selected_winner_value = 0.0;
            }
            EmCalibrationPass::Pass2Fine => {
                self.start_delta_pct = -4.0;
                self.end_delta_pct = 4.0;
                self.step_delta_pct = 1.0;
                self.selected_winner_value = 0.0;
            }
            EmCalibrationPass::Pass3HighFlowDieSwell => {
                self.start_swell_high = 1.00;
                self.end_swell_high = 1.18;
                self.step_swell_high = 0.03;
                self.selected_winner_value = 1.09;
            }
        }
    }

    /// Computes the new `extrusion_multiplier` from the selected Pass 1 / Pass 2 winning percentage delta.
    pub fn compute_updated_em(&self) -> f32 {
        (self.base_extrusion_multiplier * (1.0 + self.selected_winner_value / 100.0))
            .clamp(0.20, 2.00)
    }
}

/// Parameters for Ellis-style 90-degree chevron Pressure Advance calibration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PaCalibrationSpec {
    pub mode: PaCalibrationMode,
    /// Optional index into `SlicerConfig::tools` (`None` = use `config.active_tool_index`).
    #[serde(default)]
    pub tool_index: Option<usize>,
    /// Starting Pressure Advance value $k_{\text{start}}$ (seconds).
    pub pa_start_s: f32,
    /// Ending Pressure Advance value $k_{\text{end}}$ (seconds).
    pub pa_end_s: f32,
    /// Step increment $\Delta k$ (seconds) per chevron.
    pub pa_step_s: f32,
    /// Slow corner/entry speed $v_{\text{slow}}$ (mm/s).
    pub slow_speed_mms: f32,
    /// Fast burst speed $v_{\text{fast}}$ (mm/s) used in `SingleBand` mode.
    pub fast_speed_mms: f32,
    /// Test acceleration (`SET_VELOCITY_LIMIT ACCEL=...`, $\text{mm/s}^2$).
    pub accel_mms2: f32,
    /// Number of stacked chevron test layers (`2..=6`).
    pub test_layer_count: u32,
    /// Prints a 1-layer adhesion ribbon underneath the speed-transition vertices and 90° apexes.
    pub include_adhesion_ribbons: bool,
    /// Selected winning `pa_low_flow_s` in the UI calculator.
    pub winner_pa_low_s: f32,
    /// Selected winning `pa_high_flow_s` in the UI calculator.
    pub winner_pa_high_s: f32,
}

impl Default for PaCalibrationSpec {
    fn default() -> Self {
        Self {
            mode: PaCalibrationMode::DualBandShearThinning,
            tool_index: None,
            pa_start_s: 0.005,
            pa_end_s: 0.065,
            pa_step_s: 0.005,
            slow_speed_mms: 20.0,
            fast_speed_mms: 140.0,
            accel_mms2: 4500.0,
            test_layer_count: 4,
            include_adhesion_ribbons: false,
            winner_pa_low_s: 0.040,
            winner_pa_high_s: 0.024,
        }
    }
}

/// Parameters for Constant-Speed Flow-Step Transient Melt-Zone Pressure calibration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TransientCalibrationSpec {
    pub pass: TransientCalibrationPass,
    /// Optional index into `SlicerConfig::tools` (`None` = use `config.active_tool_index`).
    #[serde(default)]
    pub tool_index: Option<usize>,
    /// Fixed or baseline exponent $\gamma$ (`transient_pressure_exponent`).
    pub base_exponent_gamma: f32,
    /// Starting exponent $\gamma$ for Pass 1 (`0.00` = uncompensated control rib).
    pub start_gamma: f32,
    /// Ending exponent $\gamma$ for Pass 1.
    pub end_gamma: f32,
    /// Step increment for $\gamma$ in Pass 1.
    pub step_gamma: f32,
    /// Fixed or baseline relaxation time constant $\tau_{\text{melt}}$ (`transient_pressure_tau_s`).
    pub base_tau_s: f32,
    /// Starting $\tau_{\text{melt}}$ (s) for Pass 2.
    pub start_tau_s: f32,
    /// Ending $\tau_{\text{melt}}$ (s) for Pass 2.
    pub end_tau_s: f32,
    /// Step increment for $\tau_{\text{melt}}$ (s) in Pass 2.
    pub step_tau_s: f32,
    /// Minimum flow multiplier clamp (`transient_min_flow_multiplier`) during step-down.
    pub min_flow_multiplier: f32,
    /// Constant linear toolhead speed (mm/s) held across the high-to-low cross-section step-down
    /// so firmware PA (`SET_PRESSURE_ADVANCE`) is 100% inactive ($\Delta v = 0$).
    pub constant_test_speed_mms: f32,
    /// Number of stacked rib layers (`3..=8`).
    pub rib_layer_count: u32,
    /// Selected winning parameter value ($\gamma$ in Pass 1, $\tau_{\text{melt}}$ in Pass 2).
    pub selected_winner_value: f32,
}

impl Default for TransientCalibrationSpec {
    fn default() -> Self {
        Self {
            pass: TransientCalibrationPass::Pass1ExponentGamma,
            tool_index: None,
            base_exponent_gamma: 0.35,
            start_gamma: 0.00,
            end_gamma: 0.70,
            step_gamma: 0.10,
            base_tau_s: 0.25,
            start_tau_s: 0.05,
            end_tau_s: 0.65,
            step_tau_s: 0.10,
            min_flow_multiplier: 0.72,
            constant_test_speed_mms: 75.0,
            rib_layer_count: 5,
            selected_winner_value: 0.35,
        }
    }
}

/// Resource holding the interactive state of the Calibration Utilities panel in the GUI.
#[derive(Resource, Clone, Debug)]
pub struct CalibrationUiState {
    pub show_window: bool,
    pub active_kind: CalibrationUtilityKind,
    /// Optional override index into `SlicerConfig::tools` (`None` = follow `config.active_tool_index`).
    pub selected_tool_index: Option<usize>,
    pub em: EmCalibrationSpec,
    pub pa: PaCalibrationSpec,
    pub transient: TransientCalibrationSpec,
    pub last_generated_filename_stem: String,
    pub last_summary_message: String,
}

impl Default for CalibrationUiState {
    fn default() -> Self {
        Self {
            show_window: false,
            active_kind: CalibrationUtilityKind::ExtrusionMultiplier,
            selected_tool_index: None,
            em: EmCalibrationSpec::default(),
            pa: PaCalibrationSpec::default(),
            transient: TransientCalibrationSpec::default(),
            last_generated_filename_stem: String::new(),
            last_summary_message: String::new(),
        }
    }
}

/// Orthonormal coordinate helper anchored at the build-plate center that maps local nozzle-frame
/// coordinates `(u_travel, v_lateral, h_nozzle)` into both scene coordinates and printer G-code
/// bed coordinates while strictly obeying `AGENTS.md` Rule 1 & Rule 2.
#[derive(Clone, Debug)]
struct CalibrationBasis {
    frame: NozzleFrame,
    up_bed: Vec3,
    scene_to_gcode_offset: Vec3,
    tool_id: u32,
    tool_name: String,
    nozzle_diam_mm: f32,
    nozzle_temp_c: f32,
    filament_area_mm2: f32,
}

impl CalibrationBasis {
    fn from_config_with_tool(config: &SlicerConfig, tool_index_override: Option<usize>) -> Self {
        let gravity_bed = config.query_gravity_bed(Vec3::ZERO);
        let up_bed = (-gravity_bed).try_normalize().unwrap_or(Vec3::Z);
        let base_frame = config.query_nozzle_frame(Vec3::ZERO, Vec3::X);

        let selected_tool = match tool_index_override {
            Some(idx) if !config.tools.is_empty() => {
                config.tools.get(idx.min(config.tools.len() - 1))
            }
            _ => config.active_tool(),
        };
        let tool_id = selected_tool.map(|t| t.id).unwrap_or(0);
        let tool_name = selected_tool
            .map(|t| t.name.clone())
            .unwrap_or_else(|| format!("T{tool_id}"));
        let nozzle_diam_mm = selected_tool
            .map(|t| t.nozzle_diameter_mm)
            .unwrap_or(0.40)
            .max(0.15);
        let nozzle_temp_c = selected_tool
            .map(|t| t.temperature_celsius)
            .unwrap_or(215.0);
        let d_fil = config.gcode.filament_diameter_mm.clamp(1.0, 3.5);
        let filament_area_mm2 = std::f32::consts::FRAC_PI_4 * d_fil * d_fil;

        Self {
            frame: base_frame,
            up_bed,
            scene_to_gcode_offset: config.scene_to_gcode_offset(),
            tool_id,
            tool_name,
            nozzle_diam_mm,
            nozzle_temp_c,
            filament_area_mm2,
        }
    }

    #[inline]
    fn to_scene(&self, u: f32, v: f32, h: f32) -> Vec3 {
        self.frame.travel_axis * u + self.frame.lateral_axis * v + self.frame.nozzle_axis * h
    }

    #[inline]
    fn to_bed(&self, u: f32, v: f32, h: f32) -> Vec3 {
        self.to_scene(u, v, h) + self.scene_to_gcode_offset
    }

    #[inline]
    fn query_nozzle_axis_for_segment(&self, p0_bed: Vec3, p1_bed: Vec3, config: &SlicerConfig) -> Vec3 {
        let travel = (p1_bed - p0_bed).try_normalize().unwrap_or(self.frame.travel_axis);
        config
            .query_nozzle_frame(0.5 * (p0_bed + p1_bed), travel)
            .nozzle_axis
    }
}

/// Returns single-stroke vector polylines in normalized `[0..1] x [0..1]` glyph coordinates
/// for crisp single-bead 3D printing of numerical calibration labels and headers.
fn glyph_strokes(ch: char) -> &'static [&'static [[f32; 2]]] {
    match ch.to_ascii_uppercase() {
        '0' => &[&[[0.1, 0.0], [0.9, 0.0], [0.9, 1.0], [0.1, 1.0], [0.1, 0.0]]],
        '1' => &[&[[0.25, 0.8], [0.5, 1.0], [0.5, 0.0]], &[[0.2, 0.0], [0.8, 0.0]]],
        '2' => &[&[
            [0.1, 0.85],
            [0.3, 1.0],
            [0.8, 1.0],
            [0.9, 0.8],
            [0.9, 0.6],
            [0.1, 0.0],
            [0.9, 0.0],
        ]],
        '3' => &[
            &[[0.1, 1.0], [0.9, 1.0], [0.9, 0.0], [0.1, 0.0]],
            &[[0.25, 0.5], [0.9, 0.5]],
        ],
        '4' => &[
            &[[0.15, 1.0], [0.15, 0.45], [0.9, 0.45]],
            &[[0.75, 1.0], [0.75, 0.0]],
        ],
        '5' => &[&[
            [0.9, 1.0],
            [0.1, 1.0],
            [0.1, 0.52],
            [0.85, 0.52],
            [0.85, 0.0],
            [0.1, 0.0],
        ]],
        '6' => &[&[
            [0.85, 1.0],
            [0.1, 1.0],
            [0.1, 0.0],
            [0.9, 0.0],
            [0.9, 0.5],
            [0.1, 0.5],
        ]],
        '7' => &[&[[0.1, 1.0], [0.9, 1.0], [0.4, 0.0]]],
        '8' => &[
            &[[0.1, 0.0], [0.9, 0.0], [0.9, 1.0], [0.1, 1.0], [0.1, 0.0]],
            &[[0.1, 0.5], [0.9, 0.5]],
        ],
        '9' => &[&[
            [0.15, 0.0],
            [0.9, 0.0],
            [0.9, 1.0],
            [0.1, 1.0],
            [0.1, 0.5],
            [0.9, 0.5],
        ]],
        '-' => &[&[[0.15, 0.5], [0.85, 0.5]]],
        '+' => &[&[[0.15, 0.5], [0.85, 0.5]], &[[0.5, 0.18], [0.5, 0.82]]],
        '=' => &[&[[0.15, 0.35], [0.85, 0.35]], &[[0.15, 0.65], [0.85, 0.65]]],
        '/' => &[&[[0.15, 0.0], [0.85, 1.0]]],
        '.' => &[&[[0.4, 0.0], [0.6, 0.0], [0.6, 0.15], [0.4, 0.15], [0.4, 0.0]]],
        '%' => &[
            &[[0.15, 0.0], [0.85, 1.0]],
            &[[0.15, 0.75], [0.35, 0.75], [0.35, 0.95], [0.15, 0.95], [0.15, 0.75]],
            &[[0.65, 0.05], [0.85, 0.05], [0.85, 0.25], [0.65, 0.25], [0.65, 0.05]],
        ],
        'A' => &[
            &[[0.1, 0.0], [0.5, 1.0], [0.9, 0.0]],
            &[[0.25, 0.4], [0.75, 0.4]],
        ],
        'B' => &[
            &[[0.15, 0.0], [0.15, 1.0], [0.8, 1.0], [0.88, 0.75], [0.8, 0.5], [0.15, 0.5]],
            &[[0.8, 0.5], [0.9, 0.25], [0.8, 0.0], [0.15, 0.0]],
        ],
        'E' => &[
            &[[0.85, 1.0], [0.15, 1.0], [0.15, 0.0], [0.85, 0.0]],
            &[[0.15, 0.5], [0.7, 0.5]],
        ],
        'F' => &[
            &[[0.15, 0.0], [0.15, 1.0], [0.85, 1.0]],
            &[[0.15, 0.5], [0.7, 0.5]],
        ],
        'G' => &[&[
            [0.85, 0.85],
            [0.65, 1.0],
            [0.15, 1.0],
            [0.15, 0.0],
            [0.85, 0.0],
            [0.85, 0.45],
            [0.5, 0.45],
        ]],
        'H' => &[
            &[[0.15, 0.0], [0.15, 1.0]],
            &[[0.85, 0.0], [0.85, 1.0]],
            &[[0.15, 0.5], [0.85, 0.5]],
        ],
        'I' => &[
            &[[0.25, 1.0], [0.75, 1.0]],
            &[[0.5, 1.0], [0.5, 0.0]],
            &[[0.25, 0.0], [0.75, 0.0]],
        ],
        'K' => &[
            &[[0.15, 0.0], [0.15, 1.0]],
            &[[0.85, 1.0], [0.15, 0.5], [0.85, 0.0]],
        ],
        'L' => &[&[[0.15, 1.0], [0.15, 0.0], [0.85, 0.0]]],
        'M' => &[&[[0.1, 0.0], [0.1, 1.0], [0.5, 0.4], [0.9, 1.0], [0.9, 0.0]]],
        'O' => &[&[[0.15, 0.0], [0.85, 0.0], [0.85, 1.0], [0.15, 1.0], [0.15, 0.0]]],
        'Q' => &[
            &[[0.15, 0.0], [0.85, 0.0], [0.85, 1.0], [0.15, 1.0], [0.15, 0.0]],
            &[[0.55, 0.30], [0.92, 0.0]],
        ],
        'P' => &[&[
            [0.15, 0.0],
            [0.15, 1.0],
            [0.85, 1.0],
            [0.85, 0.5],
            [0.15, 0.5],
        ]],
        'R' => &[
            &[[0.15, 0.0], [0.15, 1.0], [0.85, 1.0], [0.85, 0.5], [0.15, 0.5]],
            &[[0.5, 0.5], [0.85, 0.0]],
        ],
        'S' => &[&[
            [0.85, 1.0],
            [0.15, 1.0],
            [0.15, 0.5],
            [0.85, 0.5],
            [0.85, 0.0],
            [0.15, 0.0],
        ]],
        'T' => &[&[[0.1, 1.0], [0.9, 1.0]], &[[0.5, 1.0], [0.5, 0.0]]],
        'U' => &[&[[0.15, 1.0], [0.15, 0.0], [0.85, 0.0], [0.85, 1.0]]],
        'W' => &[&[[0.1, 1.0], [0.3, 0.0], [0.5, 0.6], [0.7, 0.0], [0.9, 1.0]]],
        'X' => &[&[[0.15, 0.0], [0.85, 1.0]], &[[0.15, 1.0], [0.85, 0.0]]],
        _ => &[],
    }
}

/// Converts a text string into 2D `(u, v)` stroke polylines starting at `(u0, v0)` with character
/// width `char_w` and height `char_h`.
fn build_vector_text_polylines(
    text: &str,
    u0: f32,
    v0: f32,
    char_w: f32,
    char_h: f32,
    spacing: f32,
) -> Vec<Vec<[f32; 2]>> {
    let mut polylines = Vec::new();
    let mut cursor_u = u0;
    for ch in text.chars() {
        if ch == ' ' {
            cursor_u += char_w * 0.7;
            continue;
        }
        let w_eff = if ch == '.' { char_w * 0.45 } else { char_w };
        for stroke in glyph_strokes(ch) {
            let mut poly = Vec::with_capacity(stroke.len());
            for &[gx, gy] in *stroke {
                poly.push([cursor_u + gx * w_eff, v0 + gy * char_h]);
            }
            if poly.len() >= 2 {
                polylines.push(poly);
            }
        }
        cursor_u += w_eff + spacing;
    }
    polylines
}

/// Helper builder that accumulates extruding polylines and travel/Z-hop moves on a layer
/// in the local `CalibrationBasis`.
struct LayerSegmentBuilder<'a> {
    basis: &'a CalibrationBasis,
    config: &'a SlicerConfig,
    segments: Vec<PlannedSegment>,
    curr_pos_bed: Option<Vec3>,
}

impl<'a> LayerSegmentBuilder<'a> {
    fn new(basis: &'a CalibrationBasis, config: &'a SlicerConfig, start_pos_bed: Option<Vec3>) -> Self {
        Self {
            basis,
            config,
            segments: Vec::new(),
            curr_pos_bed: start_pos_bed,
        }
    }

    fn travel_to(&mut self, target_u: f32, target_v: f32, target_h: f32, force_zhop: bool) {
        let target_bed = self.basis.to_bed(target_u, target_v, target_h);
        let Some(prev_bed) = self.curr_pos_bed else {
            self.curr_pos_bed = Some(target_bed);
            return;
        };
        let dist = (target_bed - prev_bed).length();
        if dist <= 1.0e-3 {
            return;
        }
        let gcfg = &self.config.gcode;
        let needs_retract = dist > gcfg.min_travel_for_retract_mm || force_zhop;
        let use_zhop = gcfg.z_hop_mm > 1.0e-3 && (force_zhop || dist > 3.5);
        let retract_mm = if needs_retract {
            gcfg.retract_length_mm
        } else {
            0.0
        };

        let z_axis = self.basis.frame.nozzle_axis;
        if use_zhop {
            let hop = gcfg.z_hop_mm;
            let p_lift = prev_bed + z_axis * hop;
            let p_above = target_bed + z_axis * hop;
            let p_prev_scene = prev_bed - self.basis.scene_to_gcode_offset;
            let p_lift_scene = p_lift - self.basis.scene_to_gcode_offset;
            let p_above_scene = p_above - self.basis.scene_to_gcode_offset;
            let p_target_scene = target_bed - self.basis.scene_to_gcode_offset;

            let dt_hop = (hop / gcfg.z_hop_speed_mms.max(1.0)).max(1.0e-4);
            self.segments.push(PlannedSegment {
                p0_local: p_prev_scene,
                p1_local: p_lift_scene,
                p0_bed: prev_bed,
                p1_bed: p_lift,
                nozzle_local_0: z_axis,
                nozzle_local_1: z_axis,
                nozzle_bed_0: z_axis,
                nozzle_bed_1: z_axis,
                kind: PlannedMoveKind::ZHop,
                tool_id: self.basis.tool_id,
                wall_index: 0,
                width_mm: self.basis.nozzle_diam_mm * 0.35,
                height_mm: hop,
                length_mm: hop,
                nominal_speed_mms: gcfg.z_hop_speed_mms,
                entry_speed_mms: 0.0,
                cruise_speed_mms: gcfg.z_hop_speed_mms,
                exit_speed_mms: 0.0,
                accel_mms2: gcfg.travel_accel_mms2,
                duration_s: dt_hop,
                delta_e_mm: 0.0,
                volumetric_flow_mm3s: 0.0,
                extrusion_multiplier: 0.0,
                pressure_advance_s: 0.0,
                melt_pressure_mm3s: 0.0,
                retract_before_mm: retract_mm,
                unretract_after_mm: 0.0,
            });

            let mid_len = (p_above - p_lift).length().max(1.0e-4);
            self.segments.push(PlannedSegment {
                p0_local: p_lift_scene,
                p1_local: p_above_scene,
                p0_bed: p_lift,
                p1_bed: p_above,
                nozzle_local_0: z_axis,
                nozzle_local_1: z_axis,
                nozzle_bed_0: z_axis,
                nozzle_bed_1: z_axis,
                kind: PlannedMoveKind::Travel,
                tool_id: self.basis.tool_id,
                wall_index: 0,
                width_mm: self.basis.nozzle_diam_mm * 0.35,
                height_mm: 0.15,
                length_mm: mid_len,
                nominal_speed_mms: gcfg.travel_speed_mms,
                entry_speed_mms: 0.0,
                cruise_speed_mms: gcfg.travel_speed_mms,
                exit_speed_mms: 0.0,
                accel_mms2: gcfg.travel_accel_mms2,
                duration_s: (mid_len / gcfg.travel_speed_mms.max(1.0)).max(1.0e-4),
                delta_e_mm: 0.0,
                volumetric_flow_mm3s: 0.0,
                extrusion_multiplier: 0.0,
                pressure_advance_s: 0.0,
                melt_pressure_mm3s: 0.0,
                retract_before_mm: 0.0,
                unretract_after_mm: 0.0,
            });

            self.segments.push(PlannedSegment {
                p0_local: p_above_scene,
                p1_local: p_target_scene,
                p0_bed: p_above,
                p1_bed: target_bed,
                nozzle_local_0: z_axis,
                nozzle_local_1: z_axis,
                nozzle_bed_0: z_axis,
                nozzle_bed_1: z_axis,
                kind: PlannedMoveKind::ZHop,
                tool_id: self.basis.tool_id,
                wall_index: 0,
                width_mm: self.basis.nozzle_diam_mm * 0.35,
                height_mm: hop,
                length_mm: hop,
                nominal_speed_mms: gcfg.z_hop_speed_mms,
                entry_speed_mms: 0.0,
                cruise_speed_mms: gcfg.z_hop_speed_mms,
                exit_speed_mms: 0.0,
                accel_mms2: gcfg.travel_accel_mms2,
                duration_s: dt_hop,
                delta_e_mm: 0.0,
                volumetric_flow_mm3s: 0.0,
                extrusion_multiplier: 0.0,
                pressure_advance_s: 0.0,
                melt_pressure_mm3s: 0.0,
                retract_before_mm: 0.0,
                unretract_after_mm: retract_mm,
            });
        } else {
            let p_prev_scene = prev_bed - self.basis.scene_to_gcode_offset;
            let p_target_scene = target_bed - self.basis.scene_to_gcode_offset;
            self.segments.push(PlannedSegment {
                p0_local: p_prev_scene,
                p1_local: p_target_scene,
                p0_bed: prev_bed,
                p1_bed: target_bed,
                nozzle_local_0: z_axis,
                nozzle_local_1: z_axis,
                nozzle_bed_0: z_axis,
                nozzle_bed_1: z_axis,
                kind: PlannedMoveKind::Travel,
                tool_id: self.basis.tool_id,
                wall_index: 0,
                width_mm: self.basis.nozzle_diam_mm * 0.35,
                height_mm: 0.15,
                length_mm: dist,
                nominal_speed_mms: gcfg.travel_speed_mms,
                entry_speed_mms: 0.0,
                cruise_speed_mms: gcfg.travel_speed_mms,
                exit_speed_mms: 0.0,
                accel_mms2: gcfg.travel_accel_mms2,
                duration_s: (dist / gcfg.travel_speed_mms.max(1.0)).max(1.0e-4),
                delta_e_mm: 0.0,
                volumetric_flow_mm3s: 0.0,
                extrusion_multiplier: 0.0,
                pressure_advance_s: 0.0,
                melt_pressure_mm3s: 0.0,
                retract_before_mm: retract_mm,
                unretract_after_mm: retract_mm,
            });
        }
        self.curr_pos_bed = Some(target_bed);
    }

    #[allow(clippy::too_many_arguments)]
    fn add_extruding_segment(
        &mut self,
        u0: f32,
        v0: f32,
        h0: f32,
        u1: f32,
        v1: f32,
        h1: f32,
        width_mm: f32,
        height_mm: f32,
        speed_mms: f32,
        accel_mms2: f32,
        flow_mult: f32,
        pa_s: f32,
        melt_pressure_mm3s: f32,
        kind: PlannedMoveKind,
        wall_index: u32,
        is_first_layer: bool,
    ) {
        let p0_bed = self.basis.to_bed(u0, v0, h0);
        let p1_bed = self.basis.to_bed(u1, v1, h1);
        let len = (p1_bed - p0_bed).length();
        if len <= 1.0e-5 {
            return;
        }
        if self
            .curr_pos_bed
            .is_none_or(|c| (c - p0_bed).length() > 0.08)
        {
            self.travel_to(u0, v0, h0, false);
        }

        let w_eff = width_mm.max(height_mm);
        let a_bead = if is_first_layer {
            w_eff * height_mm
        } else {
            height_mm * (w_eff - height_mm) + std::f32::consts::FRAC_PI_4 * height_mm * height_mm
        };
        let delta_e = ((len * a_bead) / self.basis.filament_area_mm2) * flow_mult;
        let dt = (len / speed_mms.max(1.0)).max(1.0e-4);
        let q_actual = (delta_e * self.basis.filament_area_mm2) / dt;
        let z_nozzle = self
            .basis
            .query_nozzle_axis_for_segment(p0_bed, p1_bed, self.config);

        self.segments.push(PlannedSegment {
            p0_local: p0_bed - self.basis.scene_to_gcode_offset,
            p1_local: p1_bed - self.basis.scene_to_gcode_offset,
            p0_bed,
            p1_bed,
            nozzle_local_0: z_nozzle,
            nozzle_local_1: z_nozzle,
            nozzle_bed_0: z_nozzle,
            nozzle_bed_1: z_nozzle,
            kind,
            tool_id: self.basis.tool_id,
            wall_index,
            width_mm,
            height_mm,
            length_mm: len,
            nominal_speed_mms: speed_mms,
            entry_speed_mms: speed_mms,
            cruise_speed_mms: speed_mms,
            exit_speed_mms: speed_mms,
            accel_mms2,
            duration_s: dt,
            delta_e_mm: delta_e,
            volumetric_flow_mm3s: q_actual,
            extrusion_multiplier: flow_mult,
            pressure_advance_s: pa_s,
            melt_pressure_mm3s: if melt_pressure_mm3s > 0.0 {
                melt_pressure_mm3s
            } else {
                q_actual
            },
            retract_before_mm: 0.0,
            unretract_after_mm: 0.0,
        });
        self.curr_pos_bed = Some(p1_bed);
    }

    #[allow(clippy::too_many_arguments)]
    fn add_polyline_2d(
        &mut self,
        pts_uv: &[[f32; 2]],
        h: f32,
        width_mm: f32,
        height_mm: f32,
        speed_mms: f32,
        accel_mms2: f32,
        flow_mult: f32,
        pa_s: f32,
        kind: PlannedMoveKind,
        wall_index: u32,
        is_first_layer: bool,
    ) {
        if pts_uv.len() < 2 {
            return;
        }
        self.travel_to(pts_uv[0][0], pts_uv[0][1], h, false);
        for pair in pts_uv.windows(2) {
            self.add_extruding_segment(
                pair[0][0],
                pair[0][1],
                h,
                pair[1][0],
                pair[1][1],
                h,
                width_mm,
                height_mm,
                speed_mms,
                accel_mms2,
                flow_mult,
                pa_s,
                0.0,
                kind,
                wall_index,
                is_first_layer,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn add_rect_loop(
        &mut self,
        u_min: f32,
        v_min: f32,
        u_max: f32,
        v_max: f32,
        h: f32,
        width_mm: f32,
        height_mm: f32,
        speed_mms: f32,
        accel_mms2: f32,
        flow_mult: f32,
        pa_s: f32,
        kind: PlannedMoveKind,
        wall_index: u32,
        is_first_layer: bool,
    ) {
        let pts = [
            [u_min, v_min],
            [u_max, v_min],
            [u_max, v_max],
            [u_min, v_max],
            [u_min, v_min],
        ];
        self.add_polyline_2d(
            &pts,
            h,
            width_mm,
            height_mm,
            speed_mms,
            accel_mms2,
            flow_mult,
            pa_s,
            kind,
            wall_index,
            is_first_layer,
        );
    }

    /// Fills an axis-aligned rectangle `[u_min..u_max] x [v_min..v_max]` with a continuous
    /// serpentine path of parallel scanlines at angle `+45°` (`dir_parity == 0`) or `-45°`
    /// (`dir_parity == 1`), extruding the perimeter turnarounds so scanlines anchor solidly
    /// into the surrounding wall without non-extruding travel breaks.
    #[allow(clippy::too_many_arguments)]
    fn add_monotonic_rect_infill(
        &mut self,
        u_min: f32,
        v_min: f32,
        u_max: f32,
        v_max: f32,
        h: f32,
        spacing_mm: f32,
        dir_parity: usize,
        width_mm: f32,
        height_mm: f32,
        speed_mms: f32,
        accel_mms2: f32,
        flow_mult: f32,
        pa_s: f32,
        kind: PlannedMoveKind,
        is_first_layer: bool,
    ) {
        if u_max <= u_min + 0.2 || v_max <= v_min + 0.2 {
            return;
        }
        let step_c = spacing_mm.max(0.15) * std::f32::consts::SQRT_2;
        let mut serpentine_pts: Vec<[f32; 2]> = Vec::new();
        if dir_parity.is_multiple_of(2) {
            // Scanlines along u - v = c (+45 deg), ordered monotonically by increasing c
            let c_min = u_min - v_max;
            let c_max = u_max - v_min;
            let count = ((c_max - c_min) / step_c).floor() as usize;
            serpentine_pts.reserve((count + 1) * 2);
            for i in 0..=count {
                let c = c_min + (i as f32 + 0.5) * step_c;
                let u_a = (v_min + c).max(u_min);
                let v_a = (u_a - c).clamp(v_min, v_max);
                let u_b = (v_max + c).min(u_max);
                let v_b = (u_b - c).clamp(v_min, v_max);
                if (u_b - u_a).hypot(v_b - v_a) > 0.25 {
                    let (p0, p1) = if i.is_multiple_of(2) {
                        ([u_a, v_a], [u_b, v_b])
                    } else {
                        ([u_b, v_b], [u_a, v_a])
                    };
                    serpentine_pts.push(p0);
                    serpentine_pts.push(p1);
                }
            }
        } else {
            // Scanlines along u + v = c (-45 deg), ordered monotonically by increasing c
            let c_min = u_min + v_min;
            let c_max = u_max + v_max;
            let count = ((c_max - c_min) / step_c).floor() as usize;
            serpentine_pts.reserve((count + 1) * 2);
            for i in 0..=count {
                let c = c_min + (i as f32 + 0.5) * step_c;
                let u_a = (c - v_max).max(u_min);
                let v_a = (c - u_a).clamp(v_min, v_max);
                let u_b = (c - v_min).min(u_max);
                let v_b = (c - u_b).clamp(v_min, v_max);
                if (u_b - u_a).hypot(v_b - v_a) > 0.25 {
                    let (p0, p1) = if i.is_multiple_of(2) {
                        ([u_a, v_a], [u_b, v_b])
                    } else {
                        ([u_b, v_b], [u_a, v_a])
                    };
                    serpentine_pts.push(p0);
                    serpentine_pts.push(p1);
                }
            }
        }
        if serpentine_pts.len() >= 2 {
            self.add_polyline_2d(
                &serpentine_pts,
                h,
                width_mm,
                height_mm,
                speed_mms,
                accel_mms2,
                flow_mult,
                pa_s,
                kind,
                0,
                is_first_layer,
            );
        }
    }
}

/// Compiles a sequence of `PlannedLayerGcode` layers into a complete `PlannedGcodeResult`
/// with Moonraker thumbnails, `EXCLUDE_OBJECT_DEFINE`, start/end G-code macros, and range statistics.
fn finalize_calibration_gcode_result(
    job_name: &str,
    header_summary_comment: &str,
    planned_layers: Vec<PlannedLayerGcode>,
    basis: &CalibrationBasis,
    config: &SlicerConfig,
    force_emit_pa_every_change: bool,
) -> PlannedGcodeResult {
    let mut gcfg = config.gcode.clone();
    gcfg.sanitize();

    let mut bbox_min = Vec3::splat(f32::MAX);
    let mut bbox_max = Vec3::splat(f32::MIN);
    for layer in &planned_layers {
        for seg in &layer.segments {
            if seg.kind.is_extruding() && seg.delta_e_mm > 1.0e-6 {
                bbox_min = bbox_min.min(seg.p0_bed).min(seg.p1_bed);
                bbox_max = bbox_max.max(seg.p0_bed).max(seg.p1_bed);
            }
        }
    }
    if !bbox_min.is_finite() || bbox_min.x > bbox_max.x {
        bbox_min = Vec3::ZERO;
        bbox_max = Vec3::ZERO;
    }

    let mut min_speed = f32::MAX;
    let mut max_speed = 0.0f32;
    let mut min_accel = f32::MAX;
    let mut max_accel = 0.0f32;
    let mut min_flow = f32::MAX;
    let mut max_flow = 0.0f32;
    let mut min_mult = f32::MAX;
    let mut max_mult = 0.0f32;
    let mut min_pa = f32::MAX;
    let mut max_pa = 0.0f32;
    let mut min_press = f32::MAX;
    let mut max_press = 0.0f32;
    let mut min_w = f32::MAX;
    let mut max_w = 0.0f32;
    let mut min_h = f32::MAX;
    let mut max_h = 0.0f32;

    let mut total_time_s = 0.0f32;
    let mut total_filament_mm = 0.0f32;
    let mut extruding_move_count = 0usize;
    let mut travel_move_count = 0usize;
    let mut retract_count = 0usize;

    let exclude_object_name = sanitize_exclude_object_name(job_name, 1);
    let exclude_object_center_bed = [
        0.5 * (bbox_min.x + bbox_max.x),
        0.5 * (bbox_min.y + bbox_max.y),
    ];
    let exclude_object_polygon_bed =
        compute_exclude_object_polygon_2d(&planned_layers, bbox_min, bbox_max);

    let mut gcode = String::with_capacity(64 * 1024);
    gcode.push_str("; generated by PrusaSlicer 2.8.0-Manifold2 on 2026-11-06 at 00:00:00 UTC\n");
    gcode.push_str(&format!("; Manifold2 Calibration Utility: {job_name}\n"));
    gcode.push_str(&format!(
        "; Calibration Tool: T{} ({}, {:.2}mm nozzle, {:.0}°C)\n",
        basis.tool_id, basis.tool_name, basis.nozzle_diam_mm, basis.nozzle_temp_c
    ));
    for line in header_summary_comment.lines() {
        gcode.push_str(&format!("; {line}\n"));
    }
    if gcfg.emit_thumbnails && !planned_layers.is_empty() {
        gcode.push_str(&render_toolpath_preview_thumbnails(
            &planned_layers,
            bbox_min,
            bbox_max,
            basis.up_bed,
        ));
    }
    if gcfg.emit_exclude_object {
        gcode.push_str(&format_exclude_object_define(
            &exclude_object_name,
            exclude_object_center_bed,
            &exclude_object_polygon_bed,
        ));
        gcode.push('\n');
    }

    let start_expanded = expand_gcode_template(
        &gcfg.start_gcode_template,
        basis.tool_id,
        basis.nozzle_temp_c,
        gcfg.bed_temperature_celsius,
        basis.nozzle_diam_mm,
        bbox_min,
        bbox_max,
        planned_layers.len(),
    );
    gcode.push_str(&start_expanded);
    if !start_expanded.ends_with('\n') {
        gcode.push('\n');
    }
    if !gcfg.start_gcode_template.contains("{tool_id}") {
        gcode.push_str(&format!(
            "M104 S{:.0} T{} ; Set calibration tool temperature\nT{} ; Select calibration tool ({})\nM109 S{:.0} T{} ; Wait for calibration tool temperature\n",
            basis.nozzle_temp_c,
            basis.tool_id,
            basis.tool_id,
            basis.tool_name,
            basis.nozzle_temp_c,
            basis.tool_id,
        ));
    }
    gcode.push_str("M83 ; Ensure relative extrusion\n");

    let mut emitted_accel = -1.0f32;
    let mut emitted_feed_mm_min = -1.0f32;
    let mut emitted_pa = -1.0f32;
    let mut emitted_fan_pwm: Option<u8> = None;
    let mut last_emitted_kind: Option<PlannedMoveKind> = None;
    let mut active_retract_mm = 0.0f32;

    for (layer_ord, layer) in planned_layers.iter().enumerate() {
        if layer.segments.is_empty() {
            continue;
        }
        let mut layer_max_z = 0.0f32;
        let mut layer_sum_h = 0.0f32;
        let mut layer_ext_cnt = 0usize;
        for s in &layer.segments {
            if s.kind.is_extruding() && s.delta_e_mm > 1.0e-6 {
                layer_max_z = layer_max_z.max(s.p0_bed.z).max(s.p1_bed.z);
                layer_sum_h += s.height_mm;
                layer_ext_cnt += 1;
            }
        }
        let layer_mean_h = if layer_ext_cnt > 0 {
            layer_sum_h / (layer_ext_cnt as f32)
        } else {
            config.slicing.layer_height.target_mm
        };

        gcode.push_str(&format!(
            "\n; --- LAYER {} (Iso {:.3}) ---\n;LAYER_CHANGE\n;Z:{:.3}\n;HEIGHT:{:.3}\n",
            layer.layer_index, layer.isovalue, layer_max_z, layer_mean_h
        ));
        if gcfg.emit_exclude_object {
            gcode.push_str(&format!("EXCLUDE_OBJECT_START NAME={exclude_object_name}\n"));
        }

        let layer_base_fan_pct = if (layer_ord as u32) < gcfg.disable_fan_first_layers {
            gcfg.first_layer_fan_speed_percent.clamp(0.0, 100.0)
        } else {
            gcfg.fan_speed_percent.clamp(0.0, 100.0)
        };
        let layer_base_fan_pwm = (layer_base_fan_pct * 2.55).round().clamp(0.0, 255.0) as u8;
        if emitted_fan_pwm != Some(layer_base_fan_pwm) {
            if layer_base_fan_pwm == 0 {
                gcode.push_str("M107 ; Part cooling fan off\n");
            } else {
                gcode.push_str(&format!(
                    "M106 S{layer_base_fan_pwm} ; Part cooling fan {:.0}%\n",
                    layer_base_fan_pct
                ));
            }
            emitted_fan_pwm = Some(layer_base_fan_pwm);
        }

        for seg in &layer.segments {
            total_time_s += seg.duration_s;

            if seg.retract_before_mm > 1.0e-4 && active_retract_mm <= 1.0e-4 {
                let r_len = seg.retract_before_mm;
                let r_feed = (gcfg.retract_speed_mms * 60.0).max(60.0);
                gcode.push_str(&format!("G1 E-{r_len:.5} F{r_feed:.0} ; Retract\n"));
                active_retract_mm = r_len;
                emitted_feed_mm_min = r_feed;
                retract_count += 1;
                total_time_s += r_len / gcfg.retract_speed_mms.max(1.0);
            }

            if seg.kind.is_extruding() && seg.delta_e_mm > 1.0e-6 && active_retract_mm > 1.0e-4 {
                let u_len = active_retract_mm;
                let u_feed = (gcfg.retract_speed_mms * 60.0).max(60.0);
                gcode.push_str(&format!("G1 E{u_len:.5} F{u_feed:.0} ; Unretract\n"));
                active_retract_mm = 0.0;
                emitted_feed_mm_min = u_feed;
                total_time_s += u_len / gcfg.retract_speed_mms.max(1.0);
            }

            let kind_changed = last_emitted_kind != Some(seg.kind);
            if kind_changed && seg.kind.is_extruding() {
                gcode.push_str(&format!("; TYPE:{}\n", seg.kind.label()));
            }
            if kind_changed
                || emitted_accel <= 0.0
                || ((seg.accel_mms2 - emitted_accel).abs() / emitted_accel.max(1.0)) >= 0.05
            {
                gcode.push_str(&format!(
                    "SET_VELOCITY_LIMIT ACCEL={:.0}\n",
                    seg.accel_mms2
                ));
                emitted_accel = seg.accel_mms2;
            }

            if seg.kind.is_extruding() && seg.delta_e_mm > 1.0e-6 {
                extruding_move_count += 1;
                total_filament_mm += seg.delta_e_mm;

                min_speed = min_speed.min(seg.cruise_speed_mms);
                max_speed = max_speed.max(seg.cruise_speed_mms);
                min_accel = min_accel.min(seg.accel_mms2);
                max_accel = max_accel.max(seg.accel_mms2);
                min_flow = min_flow.min(seg.volumetric_flow_mm3s);
                max_flow = max_flow.max(seg.volumetric_flow_mm3s);
                min_mult = min_mult.min(seg.extrusion_multiplier);
                max_mult = max_mult.max(seg.extrusion_multiplier);
                min_pa = min_pa.min(seg.pressure_advance_s);
                max_pa = max_pa.max(seg.pressure_advance_s);
                min_press = min_press.min(seg.melt_pressure_mm3s);
                max_press = max_press.max(seg.melt_pressure_mm3s);
                min_w = min_w.min(seg.width_mm);
                max_w = max_w.max(seg.width_mm);
                min_h = min_h.min(seg.height_mm);
                max_h = max_h.max(seg.height_mm);

                let pa_tol = if force_emit_pa_every_change {
                    1.0e-4
                } else {
                    0.002
                };
                if emitted_pa < 0.0 || (seg.pressure_advance_s - emitted_pa).abs() >= pa_tol {
                    gcode.push_str(&format!(
                        "SET_PRESSURE_ADVANCE ADVANCE={:.4}\n",
                        seg.pressure_advance_s
                    ));
                    emitted_pa = seg.pressure_advance_s;
                }

                let feed_mm_min = (seg.cruise_speed_mms * 60.0).max(30.0);
                if kind_changed
                    || emitted_feed_mm_min <= 0.0
                    || (feed_mm_min - emitted_feed_mm_min).abs() >= 6.0
                {
                    gcode.push_str(&format!(
                        "G1 X{:.3} Y{:.3} Z{:.3} E{:.5} F{:.0}\n",
                        seg.p1_bed.x, seg.p1_bed.y, seg.p1_bed.z, seg.delta_e_mm, feed_mm_min
                    ));
                    emitted_feed_mm_min = feed_mm_min;
                } else {
                    gcode.push_str(&format!(
                        "G1 X{:.3} Y{:.3} Z{:.3} E{:.5}\n",
                        seg.p1_bed.x, seg.p1_bed.y, seg.p1_bed.z, seg.delta_e_mm
                    ));
                }
            } else {
                travel_move_count += 1;
                let feed_mm_min = (seg.cruise_speed_mms * 60.0).max(60.0);
                gcode.push_str(&format!(
                    "G0 X{:.3} Y{:.3} Z{:.3} F{:.0}\n",
                    seg.p1_bed.x, seg.p1_bed.y, seg.p1_bed.z, feed_mm_min
                ));
                emitted_feed_mm_min = feed_mm_min;
            }
            last_emitted_kind = Some(seg.kind);
        }

        if gcfg.emit_exclude_object {
            gcode.push_str(&format!("EXCLUDE_OBJECT_END NAME={exclude_object_name}\n"));
        }
    }

    let end_expanded = expand_gcode_template(
        &gcfg.end_gcode_template,
        basis.tool_id,
        basis.nozzle_temp_c,
        gcfg.bed_temperature_celsius,
        basis.nozzle_diam_mm,
        bbox_min,
        bbox_max,
        planned_layers.len(),
    );
    gcode.push('\n');
    gcode.push_str(&end_expanded);
    if !end_expanded.ends_with('\n') {
        gcode.push('\n');
    }

    if min_speed > max_speed {
        min_speed = 10.0;
        max_speed = 100.0;
    }
    if min_accel > max_accel {
        min_accel = 1000.0;
        max_accel = 5000.0;
    }
    if min_flow > max_flow {
        min_flow = 0.0;
        max_flow = 15.0;
    }
    if min_mult > max_mult {
        min_mult = 0.8;
        max_mult = 1.1;
    }
    if min_pa > max_pa {
        min_pa = 0.0;
        max_pa = 0.06;
    }
    if min_press > max_press {
        min_press = 0.0;
        max_press = 15.0;
    }
    if min_w > max_w {
        min_w = 0.35;
        max_w = 0.55;
    }
    if min_h > max_h {
        min_h = 0.12;
        max_h = 0.28;
    }

    let total_filament_volume_mm3 = total_filament_mm * basis.filament_area_mm2;
    let density_g_mm3 = (gcfg.filament_density_g_cm3 * 0.001).max(0.0001);
    let total_filament_grams = total_filament_volume_mm3 * density_g_mm3;
    let time_str = format_moonraker_estimated_time(total_time_s);

    gcode.push_str(&format!(
        concat!(
            "\n; --- Moonraker / PrusaSlicer Metadata Footer ---\n",
            "; filament used [mm] = {:.2}\n",
            "; filament used [cm3] = {:.2}\n",
            "; filament used [g] = {:.2}\n",
            "; total filament used [g] = {:.2}\n",
            "; estimated printing time (normal mode) = {}\n",
            "; first_layer_height = {:.3}\n",
            "; layer_height = {:.3}\n",
            "; max_print_height = {:.3}\n",
            "; total layers count = {}\n",
            "; nozzle_diameter = {:.2}\n",
            "; filament_diameter = {:.2}\n",
            "; filament_type = {}\n",
            "; filament_settings_id = \"Manifold2 Calibration\"\n",
            "; first_layer_bed_temperature = {:.0}\n",
            "; first_layer_temperature = {:.0}\n"
        ),
        total_filament_mm,
        total_filament_volume_mm3 / 1000.0,
        total_filament_grams,
        total_filament_grams,
        time_str,
        config.slicing.layer_height.first_layer_mm,
        config.slicing.layer_height.target_mm,
        bbox_max.z.max(0.0),
        planned_layers.len(),
        basis.nozzle_diam_mm,
        gcfg.filament_diameter_mm,
        gcfg.filament_type.trim(),
        gcfg.bed_temperature_celsius,
        basis.nozzle_temp_c,
    ));

    PlannedGcodeResult {
        layers: planned_layers,
        gcode_text: gcode,
        total_time_s,
        total_filament_mm,
        total_filament_volume_mm3,
        total_filament_grams,
        bbox_min_bed: bbox_min,
        bbox_max_bed: bbox_max,
        exclude_object_name,
        exclude_object_center_bed,
        exclude_object_polygon_bed,
        extruding_move_count,
        travel_move_count,
        retract_count,
        min_speed_mms: min_speed,
        max_speed_mms: max_speed,
        min_accel_mms2: min_accel,
        max_accel_mms2: max_accel,
        min_flow_mm3s: min_flow,
        max_flow_mm3s: max_flow,
        min_flow_mult: min_mult,
        max_flow_mult: max_mult,
        min_pa_s: min_pa,
        max_pa_s: max_pa,
        min_pressure_mm3s: min_press,
        max_pressure_mm3s: max_press,
        min_width_mm: min_w,
        max_width_mm: max_w,
        min_height_mm: min_h,
        max_height_mm: max_h,
    }
}

// ============================================================================
// 1. Extrusion Multiplier & Die-Swell Calibration Generator (Orca Coupons)
// ============================================================================

#[derive(Clone, Debug)]
struct EmCouponEntry {
    label: String,
    effective_flow_mult: f32,
    top_speed_mms: f32,
}

/// Generates an Orca-style Surface Quality Coupon array for:
/// - `Pass1Coarse`: Coarse EM sweep (`-10%..+10%`)
/// - `Pass2Fine`: Fine EM sweep (`-4%..+4%`)
/// - `Pass3HighFlowDieSwell`: High-flow viscoelastic die-swell sweep (`swell_high` at `Q_high`)
pub fn generate_em_calibration_gcode(
    spec: &EmCalibrationSpec,
    config: &SlicerConfig,
) -> PlannedGcodeResult {
    let basis = CalibrationBasis::from_config_with_tool(config, spec.tool_index);
    let d_noz = basis.nozzle_diam_mm;
    let h_first = config.slicing.layer_height.first_layer_mm.clamp(0.12, 0.35);
    let h_layer = config.slicing.layer_height.target_mm.clamp(0.08, 0.32);
    let line_w = config.slicing.solid_infill_width.target_mm.max(d_noz);

    let mut coupons: Vec<EmCouponEntry> = Vec::new();
    let base_em = spec.base_extrusion_multiplier.clamp(0.4, 1.8);

    match spec.pass {
        EmCalibrationPass::Pass1Coarse | EmCalibrationPass::Pass2Fine => {
            let step = spec.step_delta_pct.abs().max(0.25);
            let lo = spec.start_delta_pct.min(spec.end_delta_pct);
            let hi = spec.start_delta_pct.max(spec.end_delta_pct);
            let count = (((hi - lo) / step).round() as usize).clamp(1, 20);
            let top_spd = config.gcode.solid_infill_speed_mms.clamp(35.0, 140.0);
            let swell_factor = if spec.isolate_die_swell_in_em_passes {
                1.0
            } else {
                1.0 / config.gcode.swell_low.max(1.0)
            };
            for i in 0..=count {
                let pct = (lo + (i as f32) * step).clamp(lo, hi);
                let label = if pct.fract().abs() < 0.05 {
                    if pct > 0.0 {
                        format!("+{pct:.0}")
                    } else {
                        format!("{pct:.0}")
                    }
                } else if pct > 0.0 {
                    format!("+{pct:.1}")
                } else {
                    format!("{pct:.1}")
                };
                let mult = base_em * (1.0 + pct / 100.0) * swell_factor;
                coupons.push(EmCouponEntry {
                    label,
                    effective_flow_mult: mult.clamp(0.3, 2.0),
                    top_speed_mms: top_spd,
                });
            }
        }
        EmCalibrationPass::Pass3HighFlowDieSwell => {
            let step = spec.step_swell_high.abs().max(0.005);
            let lo = spec.start_swell_high.min(spec.end_swell_high).max(1.0);
            let hi = spec.start_swell_high.max(spec.end_swell_high).max(lo + 0.01);
            let count = (((hi - lo) / step).round() as usize).clamp(1, 15);
            let w_eff = line_w.max(h_layer);
            let a_bead =
                h_layer * (w_eff - h_layer) + std::f32::consts::FRAC_PI_4 * h_layer * h_layer;
            let q_high = config.gcode.pa_ref_high_flow_mm3s.max(8.0);
            let high_flow_spd = (q_high / a_bead.max(0.02)).clamp(80.0, 320.0);
            for i in 0..=count {
                let swell_hi = (lo + (i as f32) * step).clamp(lo, hi);
                let recip_mult = 1.0 / swell_hi;
                coupons.push(EmCouponEntry {
                    label: format!("{swell_hi:.2}"),
                    effective_flow_mult: (base_em * recip_mult).clamp(0.4, 1.8),
                    top_speed_mms: high_flow_spd,
                });
            }
        }
    }

    let n_coupons = coupons.len();
    let cols = if n_coupons <= 4 {
        2
    } else if n_coupons <= 9 {
        3
    } else {
        4
    };
    let rows = n_coupons.div_ceil(cols);

    let body_w = spec.coupon_width_mm.clamp(18.0, 45.0);
    let body_d = spec.coupon_depth_mm.clamp(14.0, 35.0);
    let tab_w = 9.0f32;
    let tab_d = (body_d * 0.68).max(10.0);
    let tile_total_w = body_w + tab_w;
    let gap_u = 6.0f32;
    let gap_v = 6.0f32;

    let grid_w = (cols as f32) * tile_total_w + ((cols.saturating_sub(1)) as f32) * gap_u;
    let grid_d = (rows as f32) * body_d + ((rows.saturating_sub(1)) as f32) * gap_v;
    let origin_u = -0.5 * grid_w;
    let origin_v = -0.5 * grid_d;

    let num_buffer_layers = 2u32;
    let num_top_layers = spec.top_test_layers.clamp(3, 8);
    // Layer 0 = base, Layers 1..=2 = buffer, Layers 3..=3+top-1 = monotonic top test, Final Layer = embossed tab label
    let total_layers = 1 + num_buffer_layers + num_top_layers + 1;

    let pa_base = config.gcode.pa_low_flow_s;
    let mut planned_layers = Vec::with_capacity(total_layers as usize);
    let mut prev_layer_end_bed: Option<Vec3> = None;

    for layer_idx in 0..total_layers {
        let is_first = layer_idx == 0;
        let is_buffer = layer_idx >= 1 && layer_idx <= num_buffer_layers;
        let is_label_layer = layer_idx + 1 == total_layers;
        let is_top_test = !is_first && !is_buffer && !is_label_layer;

        let z_h = if is_first {
            h_first
        } else {
            h_first + (layer_idx as f32) * h_layer
        };
        let cur_h = if is_first { h_first } else { h_layer };

        let mut builder = LayerSegmentBuilder::new(&basis, config, prev_layer_end_bed);

        for (c_idx, coupon) in coupons.iter().enumerate() {
            let col = c_idx % cols;
            let row = c_idx / cols;
            let u0 = origin_u + (col as f32) * (tile_total_w + gap_u);
            let v0 = origin_v + (row as f32) * (body_d + gap_v);

            let tab_u_min = u0;
            let tab_u_max = u0 + tab_w;
            let tab_v_min = v0 + 0.5 * (body_d - tab_d);
            let tab_v_max = tab_v_min + tab_d;

            let body_u_min = u0 + tab_w;
            let body_u_max = body_u_min + body_w;
            let body_v_min = v0;
            let body_v_max = v0 + body_d;

            if is_label_layer {
                // Emboss crisp single-bead vector text on the side tab
                let label_w = (tab_w - 2.0) / (coupon.label.len().max(2) as f32 * 1.25);
                let label_h = (tab_d * 0.48).clamp(3.2, 5.2);
                let text_u = tab_u_min + 1.0;
                let text_v = tab_v_min + 0.5 * (tab_d - label_h);
                let strokes = build_vector_text_polylines(
                    &coupon.label,
                    text_u,
                    text_v,
                    label_w.clamp(1.3, 2.4),
                    label_h,
                    0.45,
                );
                for stroke in &strokes {
                    builder.add_polyline_2d(
                        stroke,
                        z_h,
                        d_noz * 0.95,
                        cur_h,
                        config.gcode.first_layer_speed_mms.min(40.0),
                        config.gcode.outer_wall_accel_mms2,
                        base_em,
                        pa_base,
                        PlannedMoveKind::OuterWall,
                        0,
                        false,
                    );
                }
                continue;
            }

            let base_mult = if is_first {
                base_em * config.gcode.first_layer_flow_multiplier
            } else if is_buffer {
                base_em
            } else {
                coupon.effective_flow_mult
            };
            let wall_spd = if is_first {
                config.gcode.first_layer_speed_mms
            } else {
                config.gcode.outer_wall_speed_mms
            };
            let infill_spd = if is_first {
                config.gcode.first_layer_speed_mms
            } else if is_top_test {
                coupon.top_speed_mms
            } else {
                config.gcode.solid_infill_speed_mms
            };

            // 1. Tab + Body Outer Perimeter
            builder.add_rect_loop(
                tab_u_min,
                tab_v_min,
                tab_u_max,
                tab_v_max,
                z_h,
                line_w,
                cur_h,
                wall_spd,
                config.gcode.outer_wall_accel_mms2,
                base_mult,
                pa_base,
                PlannedMoveKind::OuterWall,
                0,
                is_first,
            );
            builder.add_rect_loop(
                body_u_min,
                body_v_min,
                body_u_max,
                body_v_max,
                z_h,
                line_w,
                cur_h,
                wall_spd,
                config.gcode.outer_wall_accel_mms2,
                base_mult,
                pa_base,
                PlannedMoveKind::OuterWall,
                0,
                is_first,
            );
            // Inner wall loop around main body
            builder.add_rect_loop(
                body_u_min + line_w,
                body_v_min + line_w,
                body_u_max - line_w,
                body_v_max - line_w,
                z_h,
                line_w,
                cur_h,
                wall_spd,
                config.gcode.inner_wall_accel_mms2,
                base_mult,
                pa_base,
                PlannedMoveKind::InnerWall,
                1,
                is_first,
            );

            // Stadium cross-section solid infill pitch:
            // On upper layers, `a_bead = cur_h * (line_w - (1 - pi/4) * cur_h)`, so 100% space-filling
            // centerline pitch at EM=1.0 is `line_w - (1 - pi/4) * cur_h`.
            let solid_spacing = if is_first {
                line_w
            } else {
                (line_w - (1.0 - std::f32::consts::FRAC_PI_4) * cur_h).max(0.5 * line_w)
            };

            // Solid fill inside the tab so the embossed label has a flat roof.
            // Outer tab wall centerline is at `tab_u_min` (inner bead edge at `+0.50 * line_w`);
            // placing the serpentine turnaround boundary at `+0.55 * line_w` overlaps the wall bead
            // by ~45% of line width and eliminates perimeter sawtooth gaps.
            builder.add_monotonic_rect_infill(
                tab_u_min + 0.55 * line_w,
                tab_v_min + 0.55 * line_w,
                tab_u_max - 0.55 * line_w,
                tab_v_max - 0.55 * line_w,
                z_h,
                if is_buffer {
                    line_w * 2.2
                } else {
                    solid_spacing
                },
                layer_idx as usize,
                line_w,
                cur_h,
                wall_spd,
                config.gcode.infill_accel_mms2,
                base_em,
                pa_base,
                PlannedMoveKind::SolidInfill,
                is_first,
            );

            if is_first {
                // Solid first-layer foundation across the coupon body (inner wall centerline at `+1.0 * line_w`)
                builder.add_monotonic_rect_infill(
                    body_u_min + 1.55 * line_w,
                    body_v_min + 1.55 * line_w,
                    body_u_max - 1.55 * line_w,
                    body_v_max - 1.55 * line_w,
                    z_h,
                    solid_spacing,
                    layer_idx as usize,
                    line_w,
                    cur_h,
                    infill_spd,
                    config.gcode.infill_accel_mms2,
                    base_mult,
                    pa_base,
                    PlannedMoveKind::SolidInfill,
                    true,
                );
            } else if is_buffer {
                // Sparse damping buffer lattice (40% density, anchored into inner wall at `+1.55 * line_w`)
                builder.add_monotonic_rect_infill(
                    body_u_min + 1.55 * line_w,
                    body_v_min + 1.55 * line_w,
                    body_u_max - 1.55 * line_w,
                    body_v_max - 1.55 * line_w,
                    z_h,
                    line_w * 2.4,
                    layer_idx as usize,
                    line_w,
                    cur_h,
                    infill_spd,
                    config.gcode.infill_accel_mms2,
                    base_mult,
                    pa_base,
                    PlannedMoveKind::SparseInfill,
                    false,
                );
            } else {
                // Orca-Style Moat-Isolated Inner Inspection Patch:
                // Leave a 0.45 mm relief moat inside the outer shell, print a dedicated inner
                // inspection frame loop, and fill the central patch with monotonic solid scanlines!
                let moat = 0.45f32;
                let patch_u_min = body_u_min + 2.0 * line_w + moat;
                let patch_v_min = body_v_min + 2.0 * line_w + moat;
                let patch_u_max = body_u_max - 2.0 * line_w - moat;
                let patch_v_max = body_v_max - 2.0 * line_w - moat;

                builder.add_rect_loop(
                    patch_u_min,
                    patch_v_min,
                    patch_u_max,
                    patch_v_max,
                    z_h,
                    line_w,
                    cur_h,
                    infill_spd,
                    config.gcode.inner_wall_accel_mms2,
                    coupon.effective_flow_mult,
                    pa_base,
                    PlannedMoveKind::InnerWall,
                    2,
                    false,
                );
                builder.add_monotonic_rect_infill(
                    patch_u_min + 0.55 * line_w,
                    patch_v_min + 0.55 * line_w,
                    patch_u_max - 0.55 * line_w,
                    patch_v_max - 0.55 * line_w,
                    z_h,
                    solid_spacing,
                    layer_idx as usize,
                    line_w,
                    cur_h,
                    infill_spd,
                    config.gcode.infill_accel_mms2,
                    coupon.effective_flow_mult,
                    pa_base,
                    PlannedMoveKind::SolidInfill,
                    false,
                );
            }
        }

        prev_layer_end_bed = builder.curr_pos_bed;
        planned_layers.push(PlannedLayerGcode {
            layer_index: layer_idx,
            branch_id: 0,
            isovalue: z_h,
            segments: builder.segments,
        });
    }

    let summary = format!(
        "Pass: {}\nBase EM: {:.3} | Coupons: {} ({} .. {})",
        spec.pass.label(),
        base_em,
        coupons.len(),
        coupons.first().map(|c| c.label.as_str()).unwrap_or(""),
        coupons.last().map(|c| c.label.as_str()).unwrap_or(""),
    );
    finalize_calibration_gcode_result(
        "Calibration_EM_Coupons",
        &summary,
        planned_layers,
        &basis,
        config,
        false,
    )
}

// ============================================================================
// 2. Pressure Advance Calibration Generator (Ellis 90° Chevron Pattern)
// ============================================================================

/// Generates Ellis's 90° sharp-corner chevron Pressure Advance pattern in bed coordinates:
/// - `SingleBand`: Single chevron array at `spec.fast_speed_mms`.
/// - `DualBandShearThinning`: Two side-by-side chevron banks testing `Q_low` (`pa_low_flow_s`)
///   and `Q_high` (`pa_high_flow_s`) in a single print.
pub fn generate_pa_calibration_gcode(
    spec: &PaCalibrationSpec,
    config: &SlicerConfig,
) -> PlannedGcodeResult {
    let basis = CalibrationBasis::from_config_with_tool(config, spec.tool_index);
    let d_noz = basis.nozzle_diam_mm;
    let h_first = config.slicing.layer_height.first_layer_mm.clamp(0.15, 0.32);
    let h_layer = config.slicing.layer_height.target_mm.clamp(0.10, 0.30);

    let pa_lo = spec.pa_start_s.min(spec.pa_end_s).max(0.0);
    let pa_hi = spec.pa_end_s.max(spec.pa_start_s).max(pa_lo + 0.001);
    let pa_step = spec.pa_step_s.abs().max(0.001);
    let step_count = (((pa_hi - pa_lo) / pa_step).round() as usize).clamp(1, 30);
    let pa_values: Vec<f32> = (0..=step_count)
        .map(|i| (pa_lo + (i as f32) * pa_step).min(pa_hi))
        .collect();

    // Determine banks: (bank_title, v_offset, width_mm, v_fast_mms, actual_q_mm3s)
    let base_em = config.gcode.extrusion_multiplier;
    let std_w = (d_noz * 1.12).max(0.42);
    let std_area =
        h_layer * (std_w - h_layer) + std::f32::consts::FRAC_PI_4 * h_layer * h_layer;

    let banks: Vec<(&str, f32, f32, f32, f32)> = match spec.mode {
        PaCalibrationMode::SingleBand => {
            let v_fast = spec.fast_speed_mms.clamp(40.0, 350.0);
            let q_actual = std_area * v_fast * base_em;
            vec![("PA", 0.0, std_w, v_fast, q_actual)]
        }
        PaCalibrationMode::DualBandShearThinning => {
            let q_low = config.gcode.pa_ref_low_flow_mm3s.max(2.0);
            let v_fast_low = (q_low / (std_area * base_em).max(0.02)).clamp(35.0, 120.0);
            let q_low_actual = std_area * v_fast_low * base_em;

            let hi_w = (d_noz * 1.40).max(0.54);
            let hi_area =
                h_layer * (hi_w - h_layer) + std::f32::consts::FRAC_PI_4 * h_layer * h_layer;
            let q_high = config.gcode.pa_ref_high_flow_mm3s.max(q_low + 4.0);
            let v_fast_high = (q_high / (hi_area * base_em).max(0.02)).clamp(100.0, 300.0);
            let q_high_actual = hi_area * v_fast_high * base_em;

            vec![
                ("LOW Q", -34.0, std_w, v_fast_low, q_low_actual),
                ("HIGH Q", 34.0, hi_w, v_fast_high, q_high_actual),
            ]
        }
    };

    let chevron_spacing_u = 5.0f32;
    let leg_lead_mm = 2.5f32; // Short slow lead-in/out right at the frame wall
    let leg_fast_mm = 29.5f32; // Continuous fast run all the way into and out of the 90° apex!
    let leg_total = leg_lead_mm + leg_fast_mm; // 32 mm along 45-deg leg
    let cos45 = std::f32::consts::FRAC_1_SQRT_2;
    let half_span_v = leg_total * cos45; // ~22.6 mm per half-chevron
    let apex_du = leg_total * cos45;

    let array_u_span = ((pa_values.len().saturating_sub(1)) as f32) * chevron_spacing_u + apex_du;
    let start_u_base = -0.5 * array_u_span;

    let v_slow = spec.slow_speed_mms.clamp(5.0, 60.0);
    let test_accel = spec.accel_mms2.clamp(500.0, 30000.0);
    let num_test_layers = spec.test_layer_count.clamp(2, 8);
    let total_layers = 1 + num_test_layers;
    let frame_pa = pa_values[pa_values.len() / 2];

    let mut planned_layers = Vec::with_capacity(total_layers as usize);
    let mut prev_layer_end_bed: Option<Vec3> = None;

    for layer_idx in 0..total_layers {
        let is_first = layer_idx == 0;
        let z_h = if is_first {
            h_first
        } else {
            h_first + (layer_idx as f32) * h_layer
        };
        let cur_h = if is_first { h_first } else { h_layer };
        let layer_flow = if is_first {
            base_em * config.gcode.first_layer_flow_multiplier
        } else {
            base_em
        };

        let mut builder = LayerSegmentBuilder::new(&basis, config, prev_layer_end_bed);

        for &(bank_label, bank_v_center, bank_w, bank_v_fast, bank_q_mm3s) in &banks {
            let v_bot = bank_v_center - half_span_v;
            let v_top = bank_v_center + half_span_v;
            let frame_u_min = start_u_base - 4.0;
            let frame_u_max = start_u_base + array_u_span + 4.0;
            let frame_spd = if is_first {
                config.gcode.first_layer_speed_mms
            } else {
                config.gcode.outer_wall_speed_mms
            };
            let frame_accel = if is_first {
                config.gcode.outer_wall_accel_mms2
            } else {
                test_accel
            };

            // 1. Two-wall Anchor Frame around the Ellis Chevron Bank on EVERY layer.
            // Inner frame wall centerline sits directly at `v_bot` and `v_top` so chevron starts
            // and ends fuse solidly into the frame wall on every layer.
            builder.add_rect_loop(
                frame_u_min,
                v_bot - bank_w,
                frame_u_max,
                v_top + bank_w,
                z_h,
                bank_w,
                cur_h,
                frame_spd,
                frame_accel,
                layer_flow,
                frame_pa,
                PlannedMoveKind::OuterWall,
                0,
                is_first,
            );
            builder.add_rect_loop(
                frame_u_min + bank_w,
                v_bot,
                frame_u_max - bank_w,
                v_top,
                z_h,
                bank_w,
                cur_h,
                frame_spd,
                frame_accel,
                layer_flow,
                frame_pa,
                PlannedMoveKind::InnerWall,
                1,
                is_first,
            );

            if is_first {
                // 2. Optional slim 1-layer apex stabilizer (off by default so the 90° corners sit
                // directly against the high-contrast build plate).
                if spec.include_adhesion_ribbons {
                    builder.add_monotonic_rect_infill(
                        frame_u_min + 1.55 * bank_w,
                        bank_v_center - 0.65,
                        frame_u_max - 1.55 * bank_w,
                        bank_v_center + 0.65,
                        z_h,
                        bank_w,
                        0,
                        bank_w,
                        cur_h,
                        config.gcode.first_layer_speed_mms,
                        config.gcode.infill_accel_mms2,
                        layer_flow,
                        frame_pa,
                        PlannedMoveKind::SupportInterface,
                        true,
                    );
                }

                // 3. Bank Identification Label ("LOW Q" / "HIGH Q" + volumetric flow Q + test speed)
                // printed on the left of the frame
                let bank_strokes = build_vector_text_polylines(
                    bank_label,
                    frame_u_min - 16.5,
                    bank_v_center + 3.4,
                    1.55,
                    2.8,
                    0.38,
                );
                let q_label = format!("{bank_q_mm3s:.1}MM3/S");
                let q_strokes = build_vector_text_polylines(
                    &q_label,
                    frame_u_min - 16.5,
                    bank_v_center - 0.6,
                    1.35,
                    2.5,
                    0.32,
                );
                let spd_label = format!("{bank_v_fast:.0}MM/S");
                let spd_strokes = build_vector_text_polylines(
                    &spd_label,
                    frame_u_min - 16.5,
                    bank_v_center - 4.6,
                    1.35,
                    2.5,
                    0.32,
                );
                for stroke in bank_strokes
                    .iter()
                    .chain(q_strokes.iter())
                    .chain(spd_strokes.iter())
                {
                    builder.add_polyline_2d(
                        stroke,
                        z_h,
                        d_noz * 0.9,
                        cur_h,
                        35.0,
                        config.gcode.outer_wall_accel_mms2,
                        base_em,
                        frame_pa,
                        PlannedMoveKind::OuterWall,
                        0,
                        true,
                    );
                }

                // 4. Graduated bottom & top ruler tick marks + numerical PA labels
                for (i, &pa_val) in pa_values.iter().enumerate() {
                    let u_start = start_u_base + (i as f32) * chevron_spacing_u;
                    let is_major = i % 2 == 0;
                    let bot_tick_len = if is_major { 2.4 } else { 1.3 };
                    // Bottom ruler tick
                    builder.add_polyline_2d(
                        &[
                            [u_start, v_bot - bank_w],
                            [u_start, v_bot - bank_w - bot_tick_len],
                        ],
                        z_h,
                        d_noz * 0.9,
                        cur_h,
                        35.0,
                        config.gcode.outer_wall_accel_mms2,
                        base_em,
                        frame_pa,
                        PlannedMoveKind::OuterWall,
                        0,
                        true,
                    );
                    // Top alignment tick
                    builder.add_polyline_2d(
                        &[
                            [u_start, v_top + bank_w],
                            [u_start, v_top + bank_w + if is_major { 1.8 } else { 1.0 }],
                        ],
                        z_h,
                        d_noz * 0.9,
                        cur_h,
                        35.0,
                        config.gcode.outer_wall_accel_mms2,
                        base_em,
                        frame_pa,
                        PlannedMoveKind::OuterWall,
                        0,
                        true,
                    );
                    if is_major {
                        let txt = format!("{pa_val:.3}");
                        let trimmed = txt.trim_end_matches('0').trim_end_matches('.');
                        let label = if trimmed.is_empty() { "0.0" } else { trimmed };
                        let strokes = build_vector_text_polylines(
                            label,
                            u_start - 2.2,
                            v_bot - bank_w - 5.8,
                            1.35,
                            2.8,
                            0.35,
                        );
                        for stroke in &strokes {
                            builder.add_polyline_2d(
                                stroke,
                                z_h,
                                d_noz * 0.9,
                                cur_h,
                                35.0,
                                config.gcode.outer_wall_accel_mms2,
                                base_em,
                                frame_pa,
                                PlannedMoveKind::OuterWall,
                                0,
                                true,
                            );
                        }
                    }
                }
            }

            // 5. Print Ellis 90° Chevrons on ALL layers (Layer 0 prints the adhered foundation
            // track directly on the build plate; upper layers run the high-speed corner test!).
            let s_lead = leg_lead_mm * cos45;
            let s_total = leg_total * cos45;
            let anchor_ext = 0.35 * bank_w;

            for (i, &pa_val) in pa_values.iter().enumerate() {
                let u0 = start_u_base + (i as f32) * chevron_spacing_u;

                // Start & end points extend `0.35 * bank_w` past the inner frame wall (`v_bot` / `v_top`)
                // so every chevron is anchored into the 2-wall frame at both ends.
                let p_start = [u0 - anchor_ext, v_bot - anchor_ext];
                let p_lead_in = [u0 + s_lead, v_bot + s_lead];
                let p_apex = [u0 + s_total, bank_v_center];
                let p_lead_out = [u0 + s_lead, v_top - s_lead];
                let p_end = [u0 - anchor_ext, v_top + anchor_ext];

                let (spd_lead, spd_fast, seg_accel) = if is_first {
                    let s_first = config.gcode.first_layer_speed_mms;
                    (s_first, s_first, config.gcode.outer_wall_accel_mms2)
                } else {
                    (v_slow, bank_v_fast, test_accel)
                };

                builder.travel_to(p_start[0], p_start[1], z_h, false);
                for (a, b, spd) in [
                    (p_start, p_lead_in, spd_lead),
                    (p_lead_in, p_apex, spd_fast),
                    (p_apex, p_lead_out, spd_fast),
                    (p_lead_out, p_end, spd_lead),
                ] {
                    builder.add_extruding_segment(
                        a[0],
                        a[1],
                        z_h,
                        b[0],
                        b[1],
                        z_h,
                        bank_w,
                        cur_h,
                        spd,
                        seg_accel,
                        layer_flow,
                        pa_val,
                        0.0,
                        PlannedMoveKind::OuterWall,
                        0,
                        is_first,
                    );
                }
            }
        }

        prev_layer_end_bed = builder.curr_pos_bed;
        planned_layers.push(PlannedLayerGcode {
            layer_index: layer_idx,
            branch_id: 0,
            isovalue: z_h,
            segments: builder.segments,
        });
    }

    let banks_summary: String = banks
        .iter()
        .map(|&(lbl, _, w, spd, q)| format!("{lbl}: {q:.1} mm³/s ({spd:.0} mm/s @ {w:.2} mm width)"))
        .collect::<Vec<_>>()
        .join(" | ");
    let summary = format!(
        "Mode: {}\nBanks: {}\nPA Range: {:.4}s .. {:.4}s (step {:.4}s, {} chevrons)\nSlow Speed: {:.0} mm/s | Accel: {:.0} mm/s²",
        spec.mode.label(),
        banks_summary,
        pa_lo,
        pa_hi,
        pa_step,
        pa_values.len(),
        v_slow,
        test_accel,
    );
    finalize_calibration_gcode_result(
        "Calibration_PA_Ellis_Pattern",
        &summary,
        planned_layers,
        &basis,
        config,
        true,
    )
}

// ============================================================================
// 3. Transient Melt-Zone Pressure Calibration Generator
//    ("Constant-Speed Flow-Step & Staccato Striation Pattern")
// ============================================================================

/// Generates the Constant-Speed Cross-Section Step-Down Striation Pattern for calibrating
/// `transient_pressure_exponent` ($\gamma$) and `transient_pressure_tau_s` ($\tau_{\text{melt}}$).
///
/// Because linear toolhead speed $v_{\text{test}}$ is held strictly constant across the $X = 0$
/// cross-section step-down ($\Delta v = 0$), Klipper firmware PA exerts zero effect, isolating
/// the viscoelastic melt-chamber pressure discharge cleanly.
pub fn generate_transient_pressure_calibration_gcode(
    spec: &TransientCalibrationSpec,
    config: &SlicerConfig,
) -> PlannedGcodeResult {
    let basis = CalibrationBasis::from_config_with_tool(config, spec.tool_index);
    let d_noz = basis.nozzle_diam_mm;
    let h_first = config.slicing.layer_height.first_layer_mm.clamp(0.16, 0.30);
    let h_nom = config.slicing.layer_height.target_mm.clamp(0.12, 0.28);

    // Candidate (label, gamma, tau_s) ribs
    let mut ribs: Vec<(String, f32, f32)> = Vec::new();
    match spec.pass {
        TransientCalibrationPass::Pass1ExponentGamma => {
            let lo = spec.start_gamma.min(spec.end_gamma).max(0.0);
            let hi = spec.end_gamma.max(spec.start_gamma).max(lo + 0.05);
            let step = spec.step_gamma.abs().max(0.02);
            let count = (((hi - lo) / step).round() as usize).clamp(1, 16);
            let tau = spec.base_tau_s.clamp(0.02, 3.0);
            for i in 0..=count {
                let g = (lo + (i as f32) * step).min(hi);
                ribs.push((format!("{g:.2}"), g, tau));
            }
        }
        TransientCalibrationPass::Pass2RelaxationTau => {
            let lo = spec.start_tau_s.min(spec.end_tau_s).max(0.01);
            let hi = spec.end_tau_s.max(spec.start_tau_s).max(lo + 0.02);
            let step = spec.step_tau_s.abs().max(0.02);
            let count = (((hi - lo) / step).round() as usize).clamp(1, 16);
            let gamma = spec.base_exponent_gamma.clamp(0.05, 1.0);
            for i in 0..=count {
                let tau = (lo + (i as f32) * step).min(hi);
                ribs.push((format!("{tau:.2}S"), gamma, tau));
            }
        }
    }

    let rib_spacing_v = 7.0f32;
    let total_v_span = ((ribs.len().saturating_sub(1)) as f32) * rib_spacing_v;
    let start_v = -0.5 * total_v_span;

    // Zone 1: u in [-32.0 .. 0.0] (High-Flow Melt-Chamber Charge: wide bead `w_high` at constant Z and constant speed)
    // Zone 2: u in [0.0 .. +32.0] (Constant-Speed Step-Down to thin precision wall `w_low`, subdivided every 1.0 mm)
    // Zone 3: u in [+32.0 .. +48.0] (Short-Move Arachne Staccato Pulses testing short_move_streak & min_mult)
    let u_charge_start = -32.0f32;
    let u_step_transition = 0.0f32;
    let u_wall_end = 32.0f32;
    let u_burst_end = 48.0f32;

    // Step volumetric flow by ~3.4x across u = 0 purely via extrusion width (`w_high` -> `w_low`)
    // at strictly constant layer height (`ΔZ = 0`) and constant linear speed (`Δv = 0`).
    // Keeping `ΔZ = 0` prevents Klipper from braking XY velocity for Z-axis velocity/accel limits
    // and prevents the nozzle tip from plowing down a Z step at u = 0!
    let w_high = (d_noz * 2.80).clamp(0.95, 1.80);
    let w_low = (d_noz * 0.92).clamp(0.32, 0.65);

    let v_test = spec.constant_test_speed_mms.clamp(35.0, 180.0);
    let test_accel = config.gcode.outer_wall_accel_mms2.max(3000.0);
    let pa_const = config.gcode.pa_low_flow_s;
    let base_em = config.gcode.extrusion_multiplier;
    let min_mult = spec.min_flow_multiplier.clamp(0.40, 1.0);
    // Allow Pass 1 / Pass 2 sweeps to express the full `gamma in [0.0..0.7]` range without saturating early at 0.72
    let sweep_clamp_floor = min_mult.min(0.25);

    let rib_layers = spec.rib_layer_count.clamp(3, 8);
    let total_layers = 1 + rib_layers;
    let mut planned_layers = Vec::with_capacity(total_layers as usize);
    let mut prev_layer_end_bed: Option<Vec3> = None;

    let pad_v = 4.5f32;
    let frame_w = (d_noz * 1.15).max(0.42);
    let frame_u_min = u_charge_start;
    let frame_u_max = u_burst_end;
    let frame_v_min = start_v - pad_v;
    let frame_v_max = start_v + total_v_span + pad_v;

    for layer_idx in 0..total_layers {
        let is_first = layer_idx == 0;
        let z_h = if is_first {
            h_first
        } else {
            h_first + (layer_idx as f32) * h_nom
        };
        let cur_h = if is_first { h_first } else { h_nom };
        let layer_flow = if is_first {
            base_em * config.gcode.first_layer_flow_multiplier
        } else {
            base_em
        };
        let frame_spd = if is_first {
            config.gcode.first_layer_speed_mms
        } else {
            config.gcode.outer_wall_speed_mms
        };

        let mut builder = LayerSegmentBuilder::new(&basis, config, prev_layer_end_bed);

        // 1. Two-wall Anchor Frame around the Rib Array on EVERY layer so rib starts (`u_charge_start`)
        //    and ends (`u_burst_end`) fuse solidly into the frame walls.
        builder.add_rect_loop(
            frame_u_min - frame_w,
            frame_v_min - frame_w,
            frame_u_max + frame_w,
            frame_v_max + frame_w,
            z_h,
            frame_w,
            cur_h,
            frame_spd,
            config.gcode.outer_wall_accel_mms2,
            layer_flow,
            pa_const,
            PlannedMoveKind::OuterWall,
            0,
            is_first,
        );
        builder.add_rect_loop(
            frame_u_min,
            frame_v_min,
            frame_u_max,
            frame_v_max,
            z_h,
            frame_w,
            cur_h,
            frame_spd,
            config.gcode.inner_wall_accel_mms2,
            layer_flow,
            pa_const,
            PlannedMoveKind::InnerWall,
            1,
            is_first,
        );

        if is_first {
            // 2. Left-hand vector labels for each rib (outside the anchor frame)
            for (r_idx, (label, _, _)) in ribs.iter().enumerate() {
                let v_rib = start_v + (r_idx as f32) * rib_spacing_v;
                let strokes = build_vector_text_polylines(
                    label,
                    frame_u_min - frame_w - 11.5,
                    v_rib - 1.4,
                    1.55,
                    2.8,
                    0.40,
                );
                for stroke in &strokes {
                    builder.add_polyline_2d(
                        stroke,
                        z_h,
                        d_noz * 0.9,
                        cur_h,
                        35.0,
                        config.gcode.outer_wall_accel_mms2,
                        base_em,
                        pa_const,
                        PlannedMoveKind::OuterWall,
                        0,
                        true,
                    );
                }
            }

            // 3. Non-intersecting Fiducial Tick Marks at u = 0.0 mm (Step-Down), u = +12.0 mm (Decay Window),
            //    and u = +32.0 mm (Staccato Zone). Placed outside the frame and in the mid-gaps BETWEEN ribs
            //    so the nozzle never crosses over a perpendicular bump while printing a rib!
            for (u_tick, tick_label) in [
                (u_step_transition, "0"),
                (12.0f32, "12"),
                (u_wall_end, "32"),
            ] {
                // Bottom & top outer frame ticks
                builder.add_polyline_2d(
                    &[
                        [u_tick, frame_v_min - frame_w],
                        [u_tick, frame_v_min - frame_w - 2.4],
                    ],
                    z_h,
                    d_noz * 0.95,
                    cur_h,
                    35.0,
                    config.gcode.outer_wall_accel_mms2,
                    layer_flow,
                    pa_const,
                    PlannedMoveKind::OuterWall,
                    0,
                    true,
                );
                builder.add_polyline_2d(
                    &[
                        [u_tick, frame_v_max + frame_w],
                        [u_tick, frame_v_max + frame_w + 2.4],
                    ],
                    z_h,
                    d_noz * 0.95,
                    cur_h,
                    35.0,
                    config.gcode.outer_wall_accel_mms2,
                    layer_flow,
                    pa_const,
                    PlannedMoveKind::OuterWall,
                    0,
                    true,
                );
                let lbl_strokes = build_vector_text_polylines(
                    tick_label,
                    u_tick - 1.4,
                    frame_v_min - frame_w - 6.0,
                    1.4,
                    2.6,
                    0.35,
                );
                for stroke in &lbl_strokes {
                    builder.add_polyline_2d(
                        stroke,
                        z_h,
                        d_noz * 0.9,
                        cur_h,
                        35.0,
                        config.gcode.outer_wall_accel_mms2,
                        base_em,
                        pa_const,
                        PlannedMoveKind::OuterWall,
                        0,
                        true,
                    );
                }

                // Short inter-rib fiducial ticks in the open space halfway between adjacent ribs
                for r_idx in 0..ribs.len().saturating_sub(1) {
                    let v_mid = start_v + (r_idx as f32 + 0.5) * rib_spacing_v;
                    builder.add_polyline_2d(
                        &[[u_tick, v_mid - 1.1], [u_tick, v_mid + 1.1]],
                        z_h,
                        d_noz * 0.95,
                        cur_h,
                        config.gcode.first_layer_speed_mms,
                        config.gcode.outer_wall_accel_mms2,
                        layer_flow,
                        pa_const,
                        PlannedMoveKind::InnerWall,
                        1,
                        true,
                    );
                }
            }
        }

        // 4. Print the Ribs on ALL layers (Layer 0 lays down the adhered foundation track directly
        //    on the build plate; Layers 1..=rib_layers run the constant-velocity step-down test).
        let anchor_ext = 0.35 * frame_w;
        for (r_idx, &(ref _label, gamma, tau_s)) in ribs.iter().enumerate() {
            let v_rib = start_v + (r_idx as f32) * rib_spacing_v;

            if is_first {
                let first_spd = config.gcode.first_layer_speed_mms;
                let first_acc = config.gcode.outer_wall_accel_mms2;
                builder.travel_to(u_charge_start - anchor_ext, v_rib, z_h, true);
                builder.add_extruding_segment(
                    u_charge_start - anchor_ext,
                    v_rib,
                    z_h,
                    u_step_transition,
                    v_rib,
                    z_h,
                    w_high,
                    cur_h,
                    first_spd,
                    first_acc,
                    layer_flow,
                    pa_const,
                    0.0,
                    PlannedMoveKind::SolidInfill,
                    0,
                    true,
                );
                builder.add_extruding_segment(
                    u_step_transition,
                    v_rib,
                    z_h,
                    u_burst_end + anchor_ext,
                    v_rib,
                    z_h,
                    w_low,
                    cur_h,
                    first_spd,
                    first_acc,
                    layer_flow,
                    pa_const,
                    0.0,
                    PlannedMoveKind::OuterWall,
                    0,
                    true,
                );
                continue;
            }

            // Test Rib Layers (Layers 1..=rib_layers):
            // Simulate first-order RC melt-zone pressure state across Zone 1 -> Zone 2 -> Zone 3
            // at strictly constant Z (`z_h`) and constant linear speed (`v_test`).
            let mut p_state_mm3s = 0.0f32;
            let mut short_streak = 0u32;

            let mut step_melt =
                |w: f32, h: f32, seg_len: f32, allow_short_streak: bool| -> (f32, f32) {
                    let w_eff = w.max(h);
                    let a_bead = h * (w_eff - h) + std::f32::consts::FRAC_PI_4 * h * h;
                    let dt = (seg_len / v_test.max(1.0)).max(1.0e-4);
                    let q_target = a_bead * v_test * base_em;
                    let k = tau_s.max(0.005);
                    let x = (dt / k).clamp(0.0, 50.0);
                    let i_x = if x.abs() < 1.0e-3 {
                        1.0 - 0.5 * x
                    } else {
                        (1.0 - (-x).exp()) / x
                    };
                    let p_avg = q_target + (p_state_mm3s - q_target) * i_x;

                    let mut m_trans = 1.0f32;
                    if gamma > 1.0e-4 && q_target > 1.0e-3 {
                        if p_avg > q_target {
                            let ratio = (q_target / p_avg).clamp(0.05, 1.0);
                            m_trans = ratio.powf(gamma).clamp(sweep_clamp_floor, 1.0);
                        }
                        if allow_short_streak && dt < 0.25 * k {
                            short_streak = (short_streak + 1).min(8);
                            let st = (short_streak as f32 / 4.0).clamp(0.0, 1.0);
                            let m_short = 1.0 - (1.0 - min_mult) * 0.5 * st;
                            m_trans = m_trans.min(m_short);
                        } else if !allow_short_streak {
                            short_streak = 0;
                        }
                    }

                    // Physical melt-chamber relaxation tracks the step from `p_state_mm3s` toward `q_target`
                    // with time constant `tau_s`, so the exponential recovery over `u in [0..+32]` directly
                    // reflects `(gamma, tau_s)`.
                    p_state_mm3s = (q_target + (p_state_mm3s - q_target) * (-x).exp()).max(0.0);
                    (base_em * m_trans, p_avg.max(0.0))
                };

            builder.travel_to(u_charge_start - anchor_ext, v_rib, z_h, true);

            // Zone 1: High-Flow Melt-Chamber Charge (u in [-32..0], 4 segments of 8 mm)
            let charge_steps = 4usize;
            let du_charge =
                (u_step_transition - (u_charge_start - anchor_ext)) / (charge_steps as f32);
            for s in 0..charge_steps {
                let u0 = (u_charge_start - anchor_ext) + (s as f32) * du_charge;
                let u1 = u0 + du_charge;
                let (mult, p_avg) = step_melt(w_high, cur_h, du_charge, false);
                builder.add_extruding_segment(
                    u0,
                    v_rib,
                    z_h,
                    u1,
                    v_rib,
                    z_h,
                    w_high,
                    cur_h,
                    v_test,
                    test_accel,
                    mult,
                    pa_const,
                    p_avg,
                    PlannedMoveKind::SolidInfill,
                    0,
                    false,
                );
            }

            // Zone 2: Constant-Velocity Cross-Section Step-Down (u in [0..+32], 32 segments of 1.0 mm)
            // Pure exponential melt-pressure relaxation (`allow_short_streak = false`) at constant Z (`z_h`)!
            let wall_steps = 32usize;
            let du_wall = (u_wall_end - u_step_transition) / (wall_steps as f32);
            for s in 0..wall_steps {
                let u0 = u_step_transition + (s as f32) * du_wall;
                let u1 = u0 + du_wall;
                let (mult, p_avg) = step_melt(w_low, cur_h, du_wall, false);
                builder.add_extruding_segment(
                    u0,
                    v_rib,
                    z_h,
                    u1,
                    v_rib,
                    z_h,
                    w_low,
                    cur_h,
                    v_test,
                    test_accel,
                    mult,
                    pa_const,
                    p_avg,
                    PlannedMoveKind::OuterWall,
                    0,
                    false,
                );
            }

            // Zone 3: Short-Move Arachne Staccato Pulses (u in [+32..+48], 16 segments of 1.0 mm)
            let burst_steps = 16usize;
            let du_burst = ((u_burst_end + anchor_ext) - u_wall_end) / (burst_steps as f32);
            for s in 0..burst_steps {
                let u0 = u_wall_end + (s as f32) * du_burst;
                let u1 = u0 + du_burst;
                let w_pulse = if s % 2 == 0 {
                    w_high * 0.85
                } else {
                    w_low
                };
                let (mult, p_avg) = step_melt(w_pulse, cur_h, du_burst, true);
                builder.add_extruding_segment(
                    u0,
                    v_rib,
                    z_h,
                    u1,
                    v_rib,
                    z_h,
                    w_pulse,
                    cur_h,
                    v_test,
                    test_accel,
                    mult,
                    pa_const,
                    p_avg,
                    PlannedMoveKind::GapFill,
                    0,
                    false,
                );
            }
        }

        prev_layer_end_bed = builder.curr_pos_bed;
        planned_layers.push(PlannedLayerGcode {
            layer_index: layer_idx,
            branch_id: 0,
            isovalue: z_h,
            segments: builder.segments,
        });
    }

    let summary = format!(
        "Pass: {}\nConstant Speed: {:.0} mm/s (Δv = 0 across X=0 transition -> Firmware PA isolated)\nRibs: {} ({} .. {}) | Min Flow Mult: {:.2}×",
        spec.pass.label(),
        v_test,
        ribs.len(),
        ribs.first().map(|r| r.0.as_str()).unwrap_or(""),
        ribs.last().map(|r| r.0.as_str()).unwrap_or(""),
        min_mult,
    );
    finalize_calibration_gcode_result(
        "Calibration_Transient_Melt_Pressure",
        &summary,
        planned_layers,
        &basis,
        config,
        false,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_em_calibration_passes_1_2_3_generate_valid_gcode_and_respect_tilted_kinematics() {
        let mut cfg = SlicerConfig::default();
        // Configure a 5-axis tilted build plate & non-vertical nozzle frame (AGENTS.md Rule 1 & Rule 2)
        cfg.machine.bed_tilt_rx_deg = 20.0;
        cfg.machine.update_gravity_from_bed_tilt();
        cfg.machine.nozzle_axis_bed = [0.0, -0.34202015, 0.9396926];

        for pass in EmCalibrationPass::ALL {
            let mut spec = EmCalibrationSpec::default();
            spec.apply_pass_defaults(pass, 0.98);
            let res = generate_em_calibration_gcode(&spec, &cfg);
            assert!(res.layers.len() >= 6);
            assert!(res.extruding_move_count > 100);
            assert!(res.max_flow_mult > res.min_flow_mult + 0.04);
            assert!(res.gcode_text.contains("EXCLUDE_OBJECT_DEFINE"));

            // Verify every extruding segment's nozzle axis matches the queried non-vertical nozzle frame
            let expected_nozzle = Vec3::from_array(cfg.machine.nozzle_axis_bed).normalize();
            for seg in &res.layers[3].segments {
                if seg.kind.is_extruding() {
                    assert!(
                        seg.nozzle_bed_0.dot(expected_nozzle) > 0.999,
                        "Calibration segment nozzle axis must follow queried NozzleFrame, got {:?}",
                        seg.nozzle_bed_0
                    );
                }
            }
        }
    }

    #[test]
    fn test_ellis_pa_single_and_dual_band_calibration_generation() {
        let cfg = SlicerConfig::default();
        for mode in PaCalibrationMode::ALL {
            let spec = PaCalibrationSpec {
                mode,
                pa_start_s: 0.010,
                pa_end_s: 0.050,
                pa_step_s: 0.010,
                ..Default::default()
            };
            let res = generate_pa_calibration_gcode(&spec, &cfg);
            assert_eq!(res.layers.len(), (1 + spec.test_layer_count) as usize);
            assert!(res.gcode_text.contains("SET_PRESSURE_ADVANCE ADVANCE=0.0100"));
            assert!(res.gcode_text.contains("SET_PRESSURE_ADVANCE ADVANCE=0.0500"));
            assert!((res.min_pa_s - 0.010).abs() < 1.0e-3);
            assert!((res.max_pa_s - 0.050).abs() < 1.0e-3);
        }
    }

    #[test]
    fn test_transient_pressure_calibration_isolates_step_down_and_throttles_flow() {
        let cfg = SlicerConfig::default();
        for pass in TransientCalibrationPass::ALL {
            let spec = TransientCalibrationSpec {
                pass,
                ..Default::default()
            };
            let res = generate_transient_pressure_calibration_gcode(&spec, &cfg);
            assert_eq!(res.layers.len(), (1 + spec.rib_layer_count) as usize);
            let min_expected_span = match pass {
                TransientCalibrationPass::Pass1ExponentGamma => 0.45,
                TransientCalibrationPass::Pass2RelaxationTau => 0.25,
            };
            assert!(
                res.max_flow_mult > res.min_flow_mult + min_expected_span,
                "Expected transient pressure step-down to throttle flow multiplier across full gamma range without early saturation: min={}, max={}",
                res.min_flow_mult,
                res.max_flow_mult
            );
            assert!(
                res.max_pressure_mm3s > 2.2 * res.min_pressure_mm3s.max(0.5),
                "Expected tracked melt-zone pressure to span high-flow charge to low-flow wall"
            );
            // Verify every extruding segment on a test layer stays at strictly constant Z (ΔZ = 0)
            for seg in &res.layers[2].segments {
                if seg.kind.is_extruding() {
                    assert!(
                        (seg.p0_bed.z - seg.p1_bed.z).abs() < 1.0e-5,
                        "Expected ΔZ = 0 along transient pressure test rib, got z0={} z1={}",
                        seg.p0_bed.z,
                        seg.p1_bed.z
                    );
                }
            }
        }
    }

    #[test]
    fn test_calibration_tool_selection_uses_selected_tool_id_nozzle_and_temperature() {
        let cfg = SlicerConfig::default();
        // Default config has T0 (0.40 mm, 215°C) at index 0 and T1 (0.60 mm, 230°C) at index 1.
        let pa_spec = PaCalibrationSpec {
            tool_index: Some(1),
            pa_start_s: 0.010,
            pa_end_s: 0.030,
            pa_step_s: 0.010,
            ..Default::default()
        };
        let res = generate_pa_calibration_gcode(&pa_spec, &cfg);
        assert!(res.gcode_text.contains("T1 ; Select active tool"));
        assert!(res.gcode_text.contains("M109 S230 T1"));
        assert!(res.gcode_text.contains("; nozzle_diameter = 0.60"));
        assert!(res
            .layers
            .iter()
            .flat_map(|l| &l.segments)
            .all(|s| s.tool_id == 1));
    }
}
