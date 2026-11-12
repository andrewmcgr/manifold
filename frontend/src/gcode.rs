use std::collections::HashMap;

use bevy::prelude::*;

use crate::bridge::ffi::ObjectToolpathsResult;
use crate::config::{
    FiveAxisKinematicsMode, GcodeDialect, GcodeProcessConfig, MachineKinematicsConfig,
    SlicerConfig, ToolConfig, WallPrintOrder,
};

/// Formats a firmware-specific dynamic Pressure Advance command for `dialect`.
#[inline]
pub fn format_pressure_advance_cmd(dialect: GcodeDialect, tool_id: u32, pa_s: f32) -> String {
    match dialect {
        GcodeDialect::Klipper => format!("SET_PRESSURE_ADVANCE ADVANCE={pa_s:.5}\n"),
        GcodeDialect::RepRapFirmware => format!("M572 D{tool_id} S{pa_s:.4}\n"),
    }
}

/// Formats a firmware-specific acceleration limit command for `dialect`.
#[inline]
pub fn format_accel_limit_cmd(
    dialect: GcodeDialect,
    print_accel_mms2: f32,
    travel_accel_mms2: f32,
) -> String {
    match dialect {
        GcodeDialect::Klipper => format!("SET_VELOCITY_LIMIT ACCEL={print_accel_mms2:.0}\n"),
        GcodeDialect::RepRapFirmware => {
            format!("M204 P{print_accel_mms2:.0} T{travel_accel_mms2:.0}\n")
        }
    }
}

/// Classification of each planned G-code segment for kinematics, emission, and 3D visualization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum PlannedMoveKind {
    OuterWall = 0,
    InnerWall = 1,
    SolidInfill = 2,
    GapFill = 3,
    SparseInfill = 4,
    Bridge = 5,
    WaveOverhang = 6,
    Support = 7,
    SupportInterface = 8,
    ScarfSeam = 9,
    Travel = 10,
    ZHop = 11,
    Wipe = 12,
}

impl PlannedMoveKind {
    pub fn from_feature_kind(feature_kind: u8) -> Self {
        match feature_kind {
            0 => Self::OuterWall,
            1 => Self::InnerWall,
            2 => Self::SolidInfill,
            3 => Self::GapFill,
            4 => Self::SparseInfill,
            5 => Self::Bridge,
            6 => Self::WaveOverhang,
            7 => Self::Support,
            8 => Self::SupportInterface,
            _ => Self::SolidInfill,
        }
    }

    pub fn is_extruding(self) -> bool {
        matches!(
            self,
            Self::OuterWall
                | Self::InnerWall
                | Self::SolidInfill
                | Self::GapFill
                | Self::SparseInfill
                | Self::Bridge
                | Self::WaveOverhang
                | Self::Support
                | Self::SupportInterface
                | Self::ScarfSeam
        )
    }

    pub fn is_support(self) -> bool {
        matches!(self, Self::Support | Self::SupportInterface)
    }

    pub fn is_wall(self) -> bool {
        matches!(
            self,
            Self::OuterWall | Self::InnerWall | Self::WaveOverhang | Self::ScarfSeam
        )
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::OuterWall => "Outer Wall",
            Self::InnerWall => "Inner Wall",
            Self::SolidInfill => "Solid Infill",
            Self::GapFill => "Gap Fill",
            Self::SparseInfill => "Sparse Infill",
            Self::Bridge => "Bridge",
            Self::WaveOverhang => "Wave Overhang",
            Self::Support => "Support Base",
            Self::SupportInterface => "Support Interface",
            Self::ScarfSeam => "Scarf Seam",
            Self::Travel => "Travel",
            Self::ZHop => "Z-Hop",
            Self::Wipe => "Wipe",
        }
    }
}

/// A single directed segment $\mathbf{p}_0 \to \mathbf{p}_1$ after geometric compensation,
/// volumetric flow calculation, kinematic lookahead, and transient pressure tracking.
#[derive(Clone, Debug)]
pub struct PlannedSegment {
    /// Start and end positions in object-local coordinates (for 3D viewport entity alignment).
    pub p0_local: Vec3,
    pub p1_local: Vec3,
    /// Start and end positions in bed coordinates (mm, emitted in G-code).
    pub p0_bed: Vec3,
    pub p1_bed: Vec3,
    /// Local nozzle axis $\hat{\mathbf{z}}_{\text{nozzle}}$ in object-local coordinates at endpoints.
    pub nozzle_local_0: Vec3,
    pub nozzle_local_1: Vec3,
    /// Local nozzle axis $\hat{\mathbf{z}}_{\text{nozzle}}$ in bed coordinates at endpoints.
    pub nozzle_bed_0: Vec3,
    pub nozzle_bed_1: Vec3,
    /// Start and end 5-axis polar tilt `B` and azimuthal rotation `C` angles (degrees).
    pub b0_deg: f32,
    pub c0_deg: f32,
    pub b1_deg: f32,
    pub c1_deg: f32,
    pub kind: PlannedMoveKind,
    /// Extruder tool ID (`T{tool_id}`, e.g. `0` for `T0`, `1` for `T1`) assigned to this segment.
    pub tool_id: u32,
    pub wall_index: u32,
    pub width_mm: f32,
    pub height_mm: f32,
    pub length_mm: f32,
    /// Target speed before lookahead deceleration (mm/s).
    pub nominal_speed_mms: f32,
    /// Planned entry speed at `p0` (mm/s).
    pub entry_speed_mms: f32,
    /// Planned peak/cruise speed inside the segment (mm/s).
    pub cruise_speed_mms: f32,
    /// Planned exit speed at `p1` (mm/s).
    pub exit_speed_mms: f32,
    /// Segment acceleration limit $a$ ($\text{mm/s}^2$).
    pub accel_mms2: f32,
    /// Exact segment execution time $t_{\text{move}}$ (s).
    pub duration_s: f32,
    /// Relative filament extrusion length $\Delta E$ (mm of raw filament).
    pub delta_e_mm: f32,
    /// Actual volumetric extrusion rate $Q_{\text{actual}} = \frac{\Delta E \cdot A_{\text{filament}}}{t_{\text{move}}}$ ($\text{mm}^3/\text{s}$).
    pub volumetric_flow_mm3s: f32,
    /// Composite dimensionless flow multiplier relative to nominal stadium cross-section.
    pub extrusion_multiplier: f32,
    /// Dynamic non-Newtonian Pressure Advance $C_{\text{PA}}(Q, F)$ (s).
    pub pressure_advance_s: f32,
    /// Tracked average melt-zone pressure $P_{\text{avg}}$ ($\text{mm}^3/\text{s}$ equivalent).
    pub melt_pressure_mm3s: f32,
    /// Non-zero if a retraction occurs immediately before this move (mm).
    pub retract_before_mm: f32,
    /// Non-zero if an unretraction occurs immediately after this travel/hop sequence (mm).
    pub unretract_after_mm: f32,
}

/// Planned segments for a single non-planar slicing layer $S_k$.
#[derive(Clone, Debug)]
pub struct PlannedLayerGcode {
    pub layer_index: u32,
    pub branch_id: u32,
    pub isovalue: f32,
    pub segments: Vec<PlannedSegment>,
}

/// Complete planned G-code output and summary statistics for 3D visualization and file export.
#[derive(Clone, Debug)]
pub struct PlannedGcodeResult {
    pub layers: Vec<PlannedLayerGcode>,
    pub gcode_text: String,
    pub total_time_s: f32,
    pub total_filament_mm: f32,
    pub total_filament_volume_mm3: f32,
    pub total_filament_grams: f32,
    pub bbox_min_bed: Vec3,
    pub bbox_max_bed: Vec3,
    pub exclude_object_name: String,
    pub exclude_object_center_bed: [f32; 2],
    pub exclude_object_polygon_bed: Vec<[f32; 2]>,
    pub extruding_move_count: usize,
    pub travel_move_count: usize,
    pub retract_count: usize,

    // Range statistics across segments for 3D colormap normalization
    pub min_speed_mms: f32,
    pub max_speed_mms: f32,
    pub min_accel_mms2: f32,
    pub max_accel_mms2: f32,
    pub min_flow_mm3s: f32,
    pub max_flow_mm3s: f32,
    pub min_flow_mult: f32,
    pub max_flow_mult: f32,
    pub min_pa_s: f32,
    pub max_pa_s: f32,
    pub min_pressure_mm3s: f32,
    pub max_pressure_mm3s: f32,
    pub min_width_mm: f32,
    pub max_width_mm: f32,
    pub min_height_mm: f32,
    pub max_height_mm: f32,
}

#[derive(Clone, Debug)]
struct WorkingVertex {
    pos_bed: Vec3,
    nozzle_bed: Vec3,
    width_mm: f32,
    height_mm: f32,
    flow_rate_mult: f32,
    kind_override: Option<PlannedMoveKind>,
}

#[derive(Clone, Debug)]
struct WorkingChain {
    pts: Vec<WorkingVertex>,
    base_kind: PlannedMoveKind,
    wall_index: u32,
    is_closed: bool,
}

/// Non-Newtonian rheology & viscoelastic die-swell helper corresponding to `FluidDynamicsEngine`.
struct FluidDynamicsModel {
    pa_zero: f32,
    pa_alpha: f32,
    pa_max: f32,
    q_low: f32,
    swell_k: f32,
    swell_m: f32,
    delta_t: f32,
}

impl FluidDynamicsModel {
    fn from_config(cfg: &GcodeProcessConfig, tool_temp_c: f32) -> Self {
        let f_frac = (cfg.fan_speed_percent / 100.0).clamp(0.0, 1.0);
        // Forced-convection tip temperature drop
        let t_eff = tool_temp_c - 3.5 * f_frac.powf(0.6);
        let delta_t = t_eff - tool_temp_c;

        let q_low = cfg.pa_ref_low_flow_mm3s.max(0.2);
        let q_high = cfg.pa_ref_high_flow_mm3s.max(q_low + 1.0);
        let pa_low = cfg.pa_low_flow_s.max(1.0e-4);
        let pa_high = cfg.pa_high_flow_s.max(1.0e-4);

        let ln_q_span = (q_high / q_low).ln().max(1.0e-4);
        let pa_alpha = (-(pa_high / pa_low).ln() / ln_q_span).clamp(0.0, 1.0);
        let pa_zero = pa_low / q_low.powf(-pa_alpha);
        let pa_max = 1.5 * pa_low.max(pa_high);

        let b_low = (cfg.swell_low - 1.0).max(1.0e-4);
        let b_high = (cfg.swell_high - 1.0).max(b_low);
        let swell_m = ((b_high / b_low).ln() / ln_q_span).clamp(0.05, 1.0);
        let swell_k = b_low / q_low.powf(swell_m);

        Self {
            pa_zero,
            pa_alpha,
            pa_max,
            q_low,
            swell_k,
            swell_m,
            delta_t,
        }
    }

    /// Evaluates viscoelastic die swell ratio $B(Q, T) \in [1.0, 2.0]$.
    fn die_swell_ratio(&self, q_mm3s: f32) -> f32 {
        if q_mm3s <= 1.0e-4 {
            return 1.0;
        }
        let temp_factor = (-0.015 * self.delta_t).exp();
        (1.0 + self.swell_k * q_mm3s.powf(self.swell_m) * temp_factor).clamp(1.0, 2.0)
    }

    /// Volumetric flow multiplier $M_{\text{swell}} = 1 / B(Q, T)$ preventing die-swell over-packing.
    fn swell_volume_multiplier(&self, q_mm3s: f32) -> f32 {
        1.0 / self.die_swell_ratio(q_mm3s)
    }

    /// Two-point power-law shear-thinning Pressure Advance $C_{\text{PA}}(Q, F)$ (seconds).
    fn pressure_advance_s(&self, q_mm3s: f32, enabled: bool, fallback_pa_s: f32) -> f32 {
        if !enabled {
            return fallback_pa_s.max(0.0);
        }
        let q_eff = q_mm3s.max(self.q_low);
        let temp_factor = (-0.02 * self.delta_t).exp();
        (self.pa_zero * q_eff.powf(-self.pa_alpha) * temp_factor).clamp(0.0, self.pa_max)
    }

    /// Adaptive retraction length $L_{\text{retract}} = \text{clamp}(C_{\text{PA}} v_{\text{fil}} + L_{\text{static}}, 0.5 L_{\text{static}}, L_{\max})$.
    fn adaptive_retract_length_mm(
        &self,
        q_last_mm3s: f32,
        pa_s: f32,
        filament_area_mm2: f32,
        static_retract_mm: f32,
    ) -> f32 {
        if static_retract_mm <= 1.0e-4 {
            return 0.0;
        }
        let v_fil = q_last_mm3s / filament_area_mm2.max(0.1);
        let max_retract = (2.2 * static_retract_mm).max(static_retract_mm + 0.6);
        (pa_s * v_fil + 0.75 * static_retract_mm).clamp(0.5 * static_retract_mm, max_retract)
    }

    /// Thermal ooze / die-swell relaxation extra prime after a travel move of duration $t_{\text{travel}}$.
    fn unretract_length_mm(
        &self,
        retract_len_mm: f32,
        travel_time_s: f32,
        cfg: &GcodeProcessConfig,
    ) -> f32 {
        if retract_len_mm <= 1.0e-4 {
            return 0.0;
        }
        let tau = (cfg.ooze_relaxation_tau_s * (-0.03 * self.delta_t).exp()).max(0.02);
        let l_max = cfg.ooze_relaxation_extra_prime_mm * (1.0 + 0.02 * self.delta_t);
        let extra = (l_max * (1.0 - (-travel_time_s.max(0.0) / tau).exp())).clamp(-0.8, 0.8);
        (retract_len_mm + extra).max(0.0)
    }
}

/// First-order RC low-pass melt-zone pressure tracker (`PressureTracker` in Section 3.4).
struct MeltPressureTracker {
    pressure_mm3s: f32,
    last_steady_q_mm3s: f32,
    short_move_streak: u32,
}

impl MeltPressureTracker {
    fn new() -> Self {
        Self {
            pressure_mm3s: 0.0,
            last_steady_q_mm3s: 0.0,
            short_move_streak: 0,
        }
    }

    /// Normalized integral decay ratio $I(x) = \frac{1 - e^{-x}}{x}$ with Taylor series for $|x| < 10^{-3}$.
    #[inline]
    fn integral_decay_ratio(x: f32) -> f32 {
        if x.abs() < 1.0e-3 {
            1.0 - 0.5 * x + (x * x) / 6.0 - (x * x * x) / 24.0
        } else {
            (1.0 - (-x).exp()) / x
        }
    }

    /// Advances the melt-zone state across an extruding segment and returns `(flow_multiplier, p_avg_mm3s)`.
    pub(crate) fn step_extrusion(
        &mut self,
        q_target_mm3s: f32,
        duration_s: f32,
        tau_pa_s: f32,
        rate_mult: f32,
        cfg: &GcodeProcessConfig,
    ) -> (f32, f32) {
        let k = cfg.transient_pressure_tau_s.max(tau_pa_s).max(0.005);
        let dt = duration_s.max(1.0e-4);
        let x = (dt / k).clamp(0.0, 50.0);
        let i_x = Self::integral_decay_ratio(x);
        let p_avg = q_target_mm3s + (self.pressure_mm3s - q_target_mm3s) * i_x;

        let mut mult = 1.0f32;
        if cfg.enable_transient_pressure_comp && q_target_mm3s > 1.0e-3 && p_avg > q_target_mm3s {
            // Apply only a gentle viscous excess-pressure trim (`>= 0.95`) so slicer-side
            // flow scaling never starves corners or fights Klipper firmware Pressure Advance,
            // and never penalizes smooth curved walls tessellated into short chords.
            let ratio = (q_target_mm3s / p_avg).clamp(0.05, 1.0);
            let min_m = cfg.transient_min_flow_multiplier.clamp(0.95, 1.0);
            mult = ratio
                .powf((cfg.transient_pressure_exponent * 0.35).clamp(0.05, 0.5))
                .clamp(min_m, 1.0);
        }
        self.short_move_streak = 0;

        let q_comp = mult * q_target_mm3s;
        let mut p_end = q_comp + (self.pressure_mm3s - q_comp) * (-x).exp();
        if rate_mult >= 0.90 {
            self.last_steady_q_mm3s = q_comp;
        } else {
            // Viscous melt retention floor during pre-retract taper
            p_end = p_end.max(0.50 * self.last_steady_q_mm3s);
        }
        self.pressure_mm3s = p_end.max(0.0);
        (mult, p_avg.max(0.0))
    }

    /// Decays melt-zone pressure during non-extruding travel moves.
    pub(crate) fn step_travel(&mut self, duration_s: f32, tau_pa_s: f32) -> f32 {
        let k = tau_pa_s.max(0.005);
        let dt = duration_s.max(0.0);
        let x = (dt / k).clamp(0.0, 50.0);
        let p_avg = self.pressure_mm3s * Self::integral_decay_ratio(x);
        self.pressure_mm3s *= (-x).exp();
        self.short_move_streak = 0;
        p_avg
    }
}

/// 3D Ramer-Douglas-Peucker simplification for open chains (`!is_closed`).
fn rdp_simplify_open(pts: &[WorkingVertex], tol_sq: f32) -> Vec<WorkingVertex> {
    if pts.len() <= 2 {
        return pts.to_vec();
    }
    let mut keep = vec![false; pts.len()];
    keep[0] = true;
    keep[pts.len() - 1] = true;

    let mut stack = vec![(0usize, pts.len() - 1)];
    while let Some((i0, i1)) = stack.pop() {
        if i1 <= i0 + 1 {
            continue;
        }
        let a = pts[i0].pos_bed;
        let b = pts[i1].pos_bed;
        let ab = b - a;
        let ab_len_sq = ab.length_squared();

        let mut max_d_sq = -1.0f32;
        let mut split_idx = i0;
        for (i, pt) in pts.iter().enumerate().take(i1).skip(i0 + 1) {
            let p = pt.pos_bed;
            let d_sq = if ab_len_sq > 1.0e-10 {
                let t = ((p - a).dot(ab) / ab_len_sq).clamp(0.0, 1.0);
                (p - (a + ab * t)).length_squared()
            } else {
                (p - a).length_squared()
            };
            // Preserve vertices where width, layer height, or nozzle orientation changes noticeably
            let width_diff = (pt.width_mm - 0.5 * (pts[i0].width_mm + pts[i1].width_mm)).abs();
            let height_diff = (pt.height_mm - 0.5 * (pts[i0].height_mm + pts[i1].height_mm)).abs();
            if d_sq > max_d_sq && width_diff <= 0.025 && height_diff <= 0.010 {
                max_d_sq = d_sq;
                split_idx = i;
            } else if width_diff > 0.025 || height_diff > 0.010 {
                max_d_sq = tol_sq + 1.0;
                split_idx = i;
                break;
            }
        }

        if max_d_sq > tol_sq && split_idx > i0 && split_idx < i1 {
            keep[split_idx] = true;
            stack.push((i0, split_idx));
            stack.push((split_idx, i1));
        }
    }

    pts.iter()
        .enumerate()
        .filter_map(|(i, v)| if keep[i] { Some(v.clone()) } else { None })
        .collect()
}

/// Simplifies a `WorkingChain` with 3D RDP:
/// - Closed loops split at their farthest vertex pair, simplify both halves, and rejoin.
/// - Open paths simplify directly with pinned start/end points (never wrapping `N-1` to `0`).
fn simplify_working_chain(chain: &mut WorkingChain, tol_mm: f32) {
    if !chain.base_kind.is_wall() || chain.pts.len() < 4 || tol_mm <= 1.0e-5 {
        return;
    }
    let tol_sq = tol_mm * tol_mm;
    if chain.is_closed {
        let n = chain.pts.len();
        let mut best_j = n / 2;
        let mut best_d_sq = -1.0f32;
        let p0 = chain.pts[0].pos_bed;
        for j in 1..n {
            let d_sq = (chain.pts[j].pos_bed - p0).length_squared();
            if d_sq > best_d_sq {
                best_d_sq = d_sq;
                best_j = j;
            }
        }
        if best_j > 0 && best_j + 1 < n {
            let half1 = rdp_simplify_open(&chain.pts[0..=best_j], tol_sq);
            let mut tail = chain.pts[best_j..].to_vec();
            tail.push(chain.pts[0].clone());
            let half2 = rdp_simplify_open(&tail, tol_sq);
            let mut merged = half1;
            if half2.len() > 2 {
                merged.extend_from_slice(&half2[1..half2.len() - 1]);
            }
            if merged.len() >= 3 {
                chain.pts = merged;
            }
        }
    } else {
        chain.pts = rdp_simplify_open(&chain.pts, tol_sq);
    }
}

/// Smooths per-vertex nozzle orientation vectors (`nozzle_bed`) along a `WorkingChain` using
/// a triangular arc-length kernel over `window_mm`, and clamps the polar tilt from `up_bed`
/// to `max_tilt_deg`. Prevents mesh-facet normal noise from causing 5-axis `B`/`C` jitter.
fn smooth_chain_nozzle_orientations(
    chain: &mut WorkingChain,
    window_mm: f32,
    max_tilt_deg: f32,
    up_bed: Vec3,
) {
    let n = chain.pts.len();
    if n == 0 {
        return;
    }
    let up = up_bed.try_normalize().unwrap_or(Vec3::Z);
    let max_tilt_rad = max_tilt_deg.clamp(0.0, 120.0).to_radians();

    let clamp_to_cone = |raw_dir: Vec3| -> Vec3 {
        let dir = raw_dir.try_normalize().unwrap_or(up);
        let cos_theta = dir.dot(up).clamp(-1.0, 1.0);
        let theta = cos_theta.acos();
        if theta <= max_tilt_rad {
            dir
        } else if max_tilt_rad <= 1.0e-5 {
            up
        } else {
            let perp = (dir - up * cos_theta).try_normalize().unwrap_or(Vec3::X);
            (up * max_tilt_rad.cos() + perp * max_tilt_rad.sin())
                .try_normalize()
                .unwrap_or(up)
        }
    };

    if n < 3 || window_mm <= 1.0e-3 {
        for pt in &mut chain.pts {
            pt.nozzle_bed = clamp_to_cone(pt.nozzle_bed);
        }
        return;
    }

    let is_closed = chain.is_closed;
    let mut smoothed = Vec::with_capacity(n);
    for i in 0..n {
        let mut accum = chain.pts[i].nozzle_bed.try_normalize().unwrap_or(up);

        // Walk backward along the chain up to `window_mm`
        let mut dist_back = 0.0f32;
        let mut curr_idx = i;
        for _ in 0..n.saturating_sub(1) {
            let prev_idx = if curr_idx > 0 {
                curr_idx - 1
            } else if is_closed {
                n - 1
            } else {
                break;
            };
            let step_len = (chain.pts[curr_idx].pos_bed - chain.pts[prev_idx].pos_bed).length();
            dist_back += step_len;
            if dist_back >= window_mm {
                break;
            }
            let w = (1.0 - dist_back / window_mm).max(0.0);
            accum += chain.pts[prev_idx].nozzle_bed.try_normalize().unwrap_or(up) * w;
            curr_idx = prev_idx;
        }

        // Walk forward along the chain up to `window_mm`
        let mut dist_fwd = 0.0f32;
        curr_idx = i;
        for _ in 0..n.saturating_sub(1) {
            let next_idx = if curr_idx + 1 < n {
                curr_idx + 1
            } else if is_closed {
                0
            } else {
                break;
            };
            let step_len = (chain.pts[next_idx].pos_bed - chain.pts[curr_idx].pos_bed).length();
            dist_fwd += step_len;
            if dist_fwd >= window_mm {
                break;
            }
            let w = (1.0 - dist_fwd / window_mm).max(0.0);
            accum += chain.pts[next_idx].nozzle_bed.try_normalize().unwrap_or(up) * w;
            curr_idx = next_idx;
        }

        smoothed.push(clamp_to_cone(accum));
    }

    for (pt, dir) in chain.pts.iter_mut().zip(smoothed.into_iter()) {
        pt.nozzle_bed = dir;
    }
}

/// Applies flat-nozzle slope elevation along each vertex's local nozzle axis $\hat{\mathbf{z}}_{\text{nozzle}}$
/// (respecting `AGENTS.md` Rule 1 & Rule 2) and inward outer-wall centerline pinning + viscoelastic swell offset.
fn compensate_wall_geometry(
    chain: &mut WorkingChain,
    nozzle_diam_mm: f32,
    nominal_outer_w: f32,
    up_bed: Vec3,
    fluid: &FluidDynamicsModel,
    nominal_speed_mms: f32,
    enable_flat_nozzle: bool,
) {
    let n = chain.pts.len();
    if n < 2 {
        return;
    }

    let r_flat = nozzle_diam_mm; // Flat annular land radius R_flat = 0.5 * (2 * D_nozzle)
    let _ = nominal_outer_w;
    let is_outer = chain.base_kind == PlannedMoveKind::OuterWall;
    let is_closed = chain.is_closed;

    let max_step = 4usize.min(n - 1);
    let mut shifts = vec![Vec3::ZERO; n];
    for i in 0..n {
        let curr = &chain.pts[i];
        // Compute a smoothed tangent over at least `0.35 mm` chord span (wrap around ONLY when `is_closed == true`)
        // so short triangle crossings never cause normal jitter or fold adjacent segments.
        let mut p_prev = curr.pos_bed;
        for step in 1..=max_step {
            let idx = if i >= step {
                i - step
            } else if is_closed {
                (i + n - (step % n)) % n
            } else {
                0
            };
            p_prev = chain.pts[idx].pos_bed;
            if (curr.pos_bed - p_prev).length_squared() >= 0.35 * 0.35 || (!is_closed && idx == 0) {
                break;
            }
        }
        let mut p_next = curr.pos_bed;
        for step in 1..=max_step {
            let idx = if i + step < n {
                i + step
            } else if is_closed {
                (i + step) % n
            } else {
                n - 1
            };
            p_next = chain.pts[idx].pos_bed;
            if (p_next - curr.pos_bed).length_squared() >= 0.35 * 0.35
                || (!is_closed && idx == n - 1)
            {
                break;
            }
        }

        let tangent = (p_next - p_prev).try_normalize().unwrap_or(Vec3::X);
        let z_nozzle = curr.nozzle_bed.try_normalize().unwrap_or(up_bed);
        let mut delta = Vec3::ZERO;

        // 1. Flat-nozzle slope clearance along local `z_nozzle`
        if enable_flat_nozzle && is_outer {
            let cos_alpha = z_nozzle.dot(up_bed).abs().clamp(0.0, 1.0);
            let sin_alpha = (1.0 - cos_alpha * cos_alpha).max(0.0).sqrt();
            let dz_slope = (r_flat * sin_alpha * (1.0 - cos_alpha)).min(0.45 * curr.height_mm);
            delta += z_nozzle * dz_slope;
        }

        // 2. Viscoelastic die-swell inset (note: `path_generator.cpp` already extracts `OuterWall`
        //    at `d_outer = 0.5 * local_outer_width`, so only die-swell inset is applied here).
        if is_outer {
            let inward_dir = z_nozzle.cross(tangent).try_normalize().unwrap_or(Vec3::Y);
            let est_q = curr.width_mm * curr.height_mm * nominal_speed_mms;
            let swell_ratio = fluid.die_swell_ratio(est_q);
            let swell_inset = 0.5 * (swell_ratio - 1.0) * curr.height_mm;
            delta += inward_dir * swell_inset;
        }

        shifts[i] = delta;
    }

    for (pt, shift) in chain.pts.iter_mut().zip(shifts.into_iter()) {
        pt.pos_bed += shift;
    }
}

/// Rotates a closed loop's seam index to the vertex closest to `from_pos_bed` (weighted slightly
/// toward sharper corners to hide the seam), or reverses an open path if its end is closer.
fn orient_chain_for_minimal_travel(chain: &mut WorkingChain, from_pos_bed: Vec3, z_penalty: f32) {
    let weighted_dist_sq = |a: Vec3, b: Vec3| -> f32 {
        let d = b - a;
        d.x * d.x + d.y * d.y + (z_penalty * d.z) * (z_penalty * d.z)
    };

    if chain.is_closed && chain.pts.len() >= 3 {
        let n = chain.pts.len();
        let mut best_idx = 0usize;
        let mut best_score = f32::MAX;
        for i in 0..n {
            let p_prev = chain.pts[(i + n - 1) % n].pos_bed;
            let p_curr = chain.pts[i].pos_bed;
            let p_next = chain.pts[(i + 1) % n].pos_bed;
            let d_in = (p_curr - p_prev).try_normalize().unwrap_or(Vec3::X);
            let d_out = (p_next - p_curr).try_normalize().unwrap_or(Vec3::X);
            // Prefer sharper corners (smaller dot product) when travel distances are similar
            let corner_bonus = 1.0 - 0.15 * (1.0 - d_in.dot(d_out)).clamp(0.0, 2.0);
            let score = weighted_dist_sq(from_pos_bed, p_curr) * corner_bonus;
            if score < best_score {
                best_score = score;
                best_idx = i;
            }
        }
        if best_idx > 0 {
            chain.pts.rotate_left(best_idx);
        }
    } else if !chain.is_closed && chain.pts.len() >= 2 {
        // Open path (open wall or infill): NEVER rotate indices; only check forward vs reverse!
        let d_front = weighted_dist_sq(from_pos_bed, chain.pts.first().unwrap().pos_bed);
        let d_back = weighted_dist_sq(from_pos_bed, chain.pts.last().unwrap().pos_bed);
        if d_back < d_front {
            chain.pts.reverse();
        }
    }
}

/// Orthonormal frame in bed coordinates with `up_bed = -gravity_bed` as the vertical axis.
/// All elevation, heightmap, brim, and lever-arm calculations operate in this frame so
/// gravity-aware and nozzle-frame invariants hold on tilted-bed and 6-axis machines.
#[derive(Clone, Copy, Debug)]
struct BedUpBasis {
    tangent_u: Vec3,
    tangent_v: Vec3,
    up_bed: Vec3,
}

impl BedUpBasis {
    fn from_up_bed(up_bed: Vec3) -> Self {
        let up = up_bed.try_normalize().unwrap_or(Vec3::Z);
        let ref_x = if up.dot(Vec3::X).abs() < 0.9 {
            Vec3::X
        } else {
            Vec3::Y
        };
        let tangent_u = (ref_x - up * ref_x.dot(up)).try_normalize().unwrap_or(Vec3::X);
        let tangent_v = up.cross(tangent_u).try_normalize().unwrap_or(Vec3::Y);
        Self {
            tangent_u,
            tangent_v,
            up_bed: up,
        }
    }

    #[inline]
    fn project_uvh(&self, p_bed: Vec3) -> (f32, f32, f32) {
        (
            p_bed.dot(self.tangent_u),
            p_bed.dot(self.tangent_v),
            p_bed.dot(self.up_bed),
        )
    }

    #[inline]
    fn unproject_uvh(&self, u: f32, v: f32, h: f32) -> Vec3 {
        self.tangent_u * u + self.tangent_v * v + self.up_bed * h
    }
}

/// 2D spatial heightmap of already-printed material in the `BedUpBasis` `(u, v)` plane.
/// Tracks both completed-layer heights (`prev_layer_max_h`) and all-printed-so-far heights
/// (`curr_max_h`) at `1.0 mm` cell resolution so travel corridors, non-planar Z-clamping,
/// and A4T tool clearance cone checks run in $O(1)$ per sample.
struct PrintedHeightMap {
    basis: BedUpBasis,
    cell_size_mm: f32,
    prev_layer_max_h: HashMap<(i32, i32), f32>,
    curr_max_h: HashMap<(i32, i32), f32>,
    coarse_cell_size_mm: f32,
    /// `(cx, cy) -> (max_h, u_at_max, v_at_max)` from completed previous layers
    prev_coarse_max_h: HashMap<(i32, i32), (f32, f32, f32)>,
    /// `(cx, cy) -> (max_h, u_at_max, v_at_max)` from all extrusions emitted so far
    curr_coarse_max_h: HashMap<(i32, i32), (f32, f32, f32)>,
}

impl PrintedHeightMap {
    fn new(basis: BedUpBasis) -> Self {
        Self {
            basis,
            cell_size_mm: 1.0,
            prev_layer_max_h: HashMap::new(),
            curr_max_h: HashMap::new(),
            coarse_cell_size_mm: 2.0,
            prev_coarse_max_h: HashMap::new(),
            curr_coarse_max_h: HashMap::new(),
        }
    }

    #[inline]
    fn cell_coords(&self, u: f32, v: f32) -> (i32, i32) {
        (
            (u / self.cell_size_mm).floor() as i32,
            (v / self.cell_size_mm).floor() as i32,
        )
    }

    #[inline]
    fn coarse_coords(&self, u: f32, v: f32) -> (i32, i32) {
        (
            (u / self.coarse_cell_size_mm).floor() as i32,
            (v / self.coarse_cell_size_mm).floor() as i32,
        )
    }

    /// Commits the current layer's accumulated heights into `prev_layer_max_h` at the end of a layer.
    fn finish_layer(&mut self) {
        for (&k, &h) in &self.curr_max_h {
            let entry = self.prev_layer_max_h.entry(k).or_insert(h);
            if h > *entry {
                *entry = h;
            }
        }
        for (&ck, &val) in &self.curr_coarse_max_h {
            let entry = self.prev_coarse_max_h.entry(ck).or_insert(val);
            if val.0 > entry.0 {
                *entry = val;
            }
        }
    }

    /// Records an extruding segment `(p0_bed -> p1_bed)` into `curr_max_h` and `curr_coarse_max_h`.
    fn record_extrusion_segment(&mut self, p0_bed: Vec3, p1_bed: Vec3) {
        let (u0, v0, h0) = self.basis.project_uvh(p0_bed);
        let (u1, v1, h1) = self.basis.project_uvh(p1_bed);
        let dist_uv = ((u1 - u0) * (u1 - u0) + (v1 - v0) * (v1 - v0)).sqrt();
        let steps = ((dist_uv / 0.45).ceil() as usize).max(1);
        for s in 0..=steps {
            let t = (s as f32) / (steps as f32);
            let su = u0 + t * (u1 - u0);
            let sv = v0 + t * (v1 - v0);
            let sh = h0 + t * (h1 - h0);
            let key = self.cell_coords(su, sv);
            let entry = self.curr_max_h.entry(key).or_insert(sh);
            if sh > *entry {
                *entry = sh;
            }
            let ckey = self.coarse_coords(su, sv);
            let centry = self.curr_coarse_max_h.entry(ckey).or_insert((sh, su, sv));
            if sh > centry.0 {
                *centry = (sh, su, sv);
            }
        }
    }

    /// Clamps a finalized sequence of `WorkingVertex` points on `layer_ord > 0` so no extruding vertex:
    /// 1. Dips more than `0.15 mm` below already-printed material in the same `1.0 mm` `(u, v)` cell
    ///    from previous layers (eliminating direct Z-plows where support/infill/wall isovalues lag).
    /// 2. Violates the active tool's local-frame radial clearance cone (`ToolConfig::max_allowed_obstacle_z_at_radius`)
    ///    against higher already-printed material in the surrounding neighborhood.
    fn clamp_vertices_monotonic_and_clearance(
        &self,
        pts: &mut [WorkingVertex],
        is_closed: bool,
        layer_ord: usize,
        tool: Option<&ToolConfig>,
    ) {
        if layer_ord == 0 || pts.is_empty() {
            return;
        }
        let n = pts.len();
        let mut required_lift = vec![0.0f32; n];

        let query_point_floor = |u: f32, v: f32, h: f32, z_noz: Vec3| -> f32 {
            let mut min_h_req = h;
            let (ix, iy) = self.cell_coords(u, v);
            if let Some(&prev_h) = self.prev_layer_max_h.get(&(ix, iy)) {
                min_h_req = min_h_req.max(prev_h - 0.08);
            }
            if let Some(&curr_h) = self.curr_max_h.get(&(ix, iy)) {
                min_h_req = min_h_req.max(curr_h - 0.10);
            }
            if let Some(tcfg) = tool {
                let (cx, cy) = self.coarse_coords(u, v);
                let cos_tilt = z_noz.dot(self.basis.up_bed).abs().clamp(0.5, 1.0);
                for dcx in -10..=10 {
                    for dcy in -10..=10 {
                        if let Some(&(obs_h, obs_u, obs_v)) =
                            self.curr_coarse_max_h.get(&(cx + dcx, cy + dcy))
                        {
                            if obs_h <= min_h_req + 0.50 {
                                continue;
                            }
                            let dr = ((obs_u - u) * (obs_u - u) + (obs_v - v) * (obs_v - v)).sqrt();
                            let r_eff = ((dr - 0.12).max(0.0)) * cos_tilt;
                            let max_dz = tcfg.max_allowed_obstacle_z_at_radius(r_eff);
                            let req_h = obs_h - (max_dz - 0.08).max(0.50);
                            if req_h > min_h_req {
                                min_h_req = req_h;
                            }
                        }
                    }
                }
            }
            min_h_req
        };

        for (i, vtx) in pts.iter().enumerate() {
            let (u, v, h) = self.basis.project_uvh(vtx.pos_bed);
            let z_noz = vtx.nozzle_bed.try_normalize().unwrap_or(self.basis.up_bed);
            let min_h_req = query_point_floor(u, v, h, z_noz);
            if min_h_req > h {
                required_lift[i] = required_lift[i].max((min_h_req - h).min(5.0));
            }
        }

        // Also check along each segment `[i .. i+1]` every 0.35 mm so intermediate cells crossed
        // by a 0.6..0.8 mm segment lift both segment endpoints above any interior cell obstacle!
        for i in 0..n.saturating_sub(1) {
            let (u0, v0, h0) = self.basis.project_uvh(pts[i].pos_bed);
            let (u1, v1, h1) = self.basis.project_uvh(pts[i + 1].pos_bed);
            let dist_uv = ((u1 - u0) * (u1 - u0) + (v1 - v0) * (v1 - v0)).sqrt();
            if dist_uv <= 0.20 {
                continue;
            }
            let z_noz = (pts[i].nozzle_bed + pts[i + 1].nozzle_bed)
                .try_normalize()
                .unwrap_or(self.basis.up_bed);
            let steps = ((dist_uv / 0.35).ceil() as usize).max(2);
            let mut seg_extra_lift = 0.0f32;
            for s in 1..steps {
                let t = (s as f32) / (steps as f32);
                let su = u0 + t * (u1 - u0);
                let sv = v0 + t * (v1 - v0);
                let sh = h0 + t * (h1 - h0);
                let req_h = query_point_floor(su, sv, sh, z_noz);
                if req_h > sh {
                    seg_extra_lift = seg_extra_lift.max((req_h - sh).min(5.0));
                }
            }
            if seg_extra_lift > 1.0e-4 {
                required_lift[i] = required_lift[i].max(seg_extra_lift);
                required_lift[i + 1] = required_lift[i + 1].max(seg_extra_lift);
            }
        }

        if required_lift.iter().all(|&dl| dl <= 1.0e-4) {
            return;
        }

        // Smooth `required_lift` along the vertex sequence with a 3-pass max-preserving ramp
        let mut smoothed = required_lift.clone();
        for _ in 0..3 {
            let prev = smoothed.clone();
            for i in 0..n {
                let l = if i > 0 {
                    prev[i - 1]
                } else if is_closed {
                    prev[n - 1]
                } else {
                    prev[i]
                };
                let r = if i + 1 < n {
                    prev[i + 1]
                } else if is_closed {
                    prev[0]
                } else {
                    prev[i]
                };
                let avg = 0.25 * l + 0.50 * prev[i] + 0.25 * r;
                smoothed[i] = required_lift[i].max(avg);
            }
        }

        for (i, vtx) in pts.iter_mut().enumerate() {
            if smoothed[i] > 1.0e-4 {
                vtx.pos_bed += self.basis.up_bed * smoothed[i];
            }
        }

        // Ensure trailing non-extruding `Wipe` vertices never dip below the peak of the
        // extruding vertices they wipe across (or below any cell in `curr_max_h`), including
        // cells newly covered by the preceding extrusions of THIS same chain!
        let mut last_ext_h: Option<f32> = None;
        let mut chain_cell_max_h: HashMap<(i32, i32), f32> = HashMap::new();
        for i in 0..pts.len() {
            let (u, v, h) = self.basis.project_uvh(pts[i].pos_bed);
            let is_wipe = pts[i].kind_override == Some(PlannedMoveKind::Wipe)
                || pts[i].flow_rate_mult <= 1.0e-4;
            if is_wipe {
                let mut req_h = last_ext_h.unwrap_or(h);
                let (u_prev, v_prev, _) = if i > 0 {
                    self.basis.project_uvh(pts[i - 1].pos_bed)
                } else {
                    (u, v, h)
                };
                let dist_uv = ((u - u_prev) * (u - u_prev) + (v - v_prev) * (v - v_prev)).sqrt();
                let steps = ((dist_uv / 0.30).ceil() as usize).max(1);
                for s in 0..=steps {
                    let t = (s as f32) / (steps as f32);
                    let su = u_prev + t * (u - u_prev);
                    let sv = v_prev + t * (v - v_prev);
                    let cell = self.cell_coords(su, sv);
                    if let Some(&ch) = self.curr_max_h.get(&cell) {
                        req_h = req_h.max(ch);
                    }
                    if let Some(&ch) = chain_cell_max_h.get(&cell) {
                        req_h = req_h.max(ch);
                    }
                }
                if req_h > h {
                    pts[i].pos_bed += self.basis.up_bed * (req_h - h);
                    // Also lift the wipe start vertex `pts[i - 1]` if `req_h` exceeds its height
                    if i > 0 {
                        let (_, _, h_prev) = self.basis.project_uvh(pts[i - 1].pos_bed);
                        if req_h > h_prev {
                            pts[i - 1].pos_bed += self.basis.up_bed * (req_h - h_prev);
                        }
                    }
                }
                last_ext_h = Some(req_h.max(h));
            } else {
                if i > 0 {
                    let (u0, v0, h0) = self.basis.project_uvh(pts[i - 1].pos_bed);
                    let dist_uv = ((u - u0) * (u - u0) + (v - v0) * (v - v0)).sqrt();
                    let steps = ((dist_uv / 0.35).ceil() as usize).max(1);
                    for s in 0..=steps {
                        let t = (s as f32) / (steps as f32);
                        let su = u0 + t * (u - u0);
                        let sv = v0 + t * (v - v0);
                        let sh = h0 + t * (h - h0);
                        let cell = self.cell_coords(su, sv);
                        let entry = chain_cell_max_h.entry(cell).or_insert(sh);
                        if sh > *entry {
                            *entry = sh;
                        }
                    }
                } else {
                    let cell = self.cell_coords(u, v);
                    chain_cell_max_h.insert(cell, h);
                }
                last_ext_h = Some(h);
            }
        }
    }

    /// Queries the maximum already-printed elevation `h` along the 2D travel corridor `(p0_bed -> p1_bed)`
    /// (including a 1-cell lateral safety margin for the nozzle tip).
    fn max_corridor_obstacle_h(&self, p0_bed: Vec3, p1_bed: Vec3) -> Option<f32> {
        let (u0, v0, _) = self.basis.project_uvh(p0_bed);
        let (u1, v1, _) = self.basis.project_uvh(p1_bed);
        let dist_uv = ((u1 - u0) * (u1 - u0) + (v1 - v0) * (v1 - v0)).sqrt();
        if dist_uv <= 1.0e-4 {
            return None;
        }
        let steps = ((dist_uv / 0.35).ceil() as usize).max(1);
        let mut max_obs_h: Option<f32> = None;

        for s in 0..=steps {
            let t = (s as f32) / (steps as f32);
            let su = u0 + t * (u1 - u0);
            let sv = v0 + t * (v1 - v0);
            let (ix, iy) = self.cell_coords(su, sv);
            for dx in -1..=1 {
                for dy in -1..=1 {
                    if let Some(&ch) = self.curr_max_h.get(&(ix + dx, iy + dy)) {
                        max_obs_h = Some(max_obs_h.map_or(ch, |m| m.max(ch)));
                    }
                }
            }
        }
        max_obs_h
    }
}

/// Orders a single elevation band of chains according to `Supports -> Walls -> WaveOverhangs -> Bridges -> Infill`.
fn order_chain_band(
    mut chains: Vec<WorkingChain>,
    wall_order: WallPrintOrder,
    start_pos_bed: Vec3,
    z_penalty: f32,
) -> (Vec<WorkingChain>, Vec3) {
    if chains.is_empty() {
        return (chains, start_pos_bed);
    }

    let mut supports: Vec<WorkingChain> = Vec::new();
    let mut walls: Vec<WorkingChain> = Vec::new();
    let mut wave_overhangs: Vec<WorkingChain> = Vec::new();
    let mut bridges: Vec<WorkingChain> = Vec::new();
    let mut infills: Vec<WorkingChain> = Vec::new();
    for ch in chains.drain(..) {
        match ch.base_kind {
            PlannedMoveKind::Support | PlannedMoveKind::SupportInterface => supports.push(ch),
            PlannedMoveKind::WaveOverhang => wave_overhangs.push(ch),
            PlannedMoveKind::Bridge => bridges.push(ch),
            k if k.is_wall() => walls.push(ch),
            _ => infills.push(ch),
        }
    }

    // Sort supported walls according to `WallPrintOrder`
    let max_wall_idx = walls.iter().map(|w| w.wall_index).max().unwrap_or(0);
    let wall_rank = |w_idx: u32| -> u32 {
        match wall_order {
            WallPrintOrder::OutsideIn => w_idx,
            WallPrintOrder::InnerOuterInner => {
                // [n..=2, 0, 1]: print inner walls >= 2 from inside out, then outer wall 0, then backing wall 1
                if w_idx >= 2 {
                    max_wall_idx - w_idx
                } else if w_idx == 0 {
                    max_wall_idx
                } else {
                    max_wall_idx + 1
                }
            }
        }
    };
    walls.sort_by_key(|w| wall_rank(w.wall_index));

    let weighted_dist_sq = |a: Vec3, b: Vec3| -> f32 {
        let d = b - a;
        d.x * d.x + d.y * d.y + (z_penalty * d.z) * (z_penalty * d.z)
    };

    let mut ordered: Vec<WorkingChain> = Vec::with_capacity(
        supports.len() + walls.len() + wave_overhangs.len() + bridges.len() + infills.len(),
    );
    let mut curr_pos = start_pos_bed;

    // 0. Print Support Base and Support Interface chains first on each band so overhanging
    //    walls, wave overhangs, and bridges have their support foundation in place.
    let mut used_supports = vec![false; supports.len()];
    for _ in 0..supports.len() {
        let mut best_i = 0usize;
        let mut best_d = f32::MAX;
        for (i, sp) in supports.iter().enumerate() {
            if used_supports[i] || sp.pts.is_empty() {
                continue;
            }
            let d = weighted_dist_sq(curr_pos, sp.pts.first().unwrap().pos_bed)
                .min(weighted_dist_sq(curr_pos, sp.pts.last().unwrap().pos_bed));
            if d < best_d {
                best_d = d;
                best_i = i;
            }
        }
        used_supports[best_i] = true;
        let mut chosen = supports[best_i].clone();
        orient_chain_for_minimal_travel(&mut chosen, curr_pos, z_penalty);
        curr_pos = chosen.pts.last().unwrap().pos_bed;
        ordered.push(chosen);
    }

    // 1. Group supported walls by rank and greedily pick nearest within each rank
    let mut used_walls = vec![false; walls.len()];
    for _ in 0..walls.len() {
        let min_remaining_rank = walls
            .iter()
            .enumerate()
            .filter(|(i, _)| !used_walls[*i])
            .map(|(_, w)| wall_rank(w.wall_index))
            .min()
            .unwrap_or(0);

        let mut best_i = 0usize;
        let mut best_d = f32::MAX;
        for (i, w) in walls.iter().enumerate() {
            if used_walls[i] || wall_rank(w.wall_index) != min_remaining_rank {
                continue;
            }
            let d = if w.is_closed {
                w.pts
                    .iter()
                    .map(|pt| weighted_dist_sq(curr_pos, pt.pos_bed))
                    .fold(f32::MAX, f32::min)
            } else {
                weighted_dist_sq(curr_pos, w.pts.first().unwrap().pos_bed)
                    .min(weighted_dist_sq(curr_pos, w.pts.last().unwrap().pos_bed))
            };
            if d < best_d {
                best_d = d;
                best_i = i;
            }
        }
        used_walls[best_i] = true;
        let mut chosen = walls[best_i].clone();
        orient_chain_for_minimal_travel(&mut chosen, curr_pos, z_penalty);
        curr_pos = if chosen.is_closed {
            chosen.pts.first().unwrap().pos_bed
        } else {
            chosen.pts.last().unwrap().pos_bed
        };
        ordered.push(chosen);
    }

    // 2. Order Wave Overhangs strictly inside-out (`wall_index = 1, 2, ...`) so each pass
    //    prints in continuous contact with the preceding supported or wave extrusion.
    wave_overhangs.sort_by_key(|w| w.wall_index);
    let mut used_waves = vec![false; wave_overhangs.len()];
    for _ in 0..wave_overhangs.len() {
        let min_wave_idx = wave_overhangs
            .iter()
            .enumerate()
            .filter(|(i, _)| !used_waves[*i])
            .map(|(_, w)| w.wall_index)
            .min()
            .unwrap_or(0);

        let mut best_i = 0usize;
        let mut best_d = f32::MAX;
        for (i, w) in wave_overhangs.iter().enumerate() {
            if used_waves[i] || w.wall_index != min_wave_idx || w.pts.is_empty() {
                continue;
            }
            let d = if w.is_closed {
                w.pts
                    .iter()
                    .map(|pt| weighted_dist_sq(curr_pos, pt.pos_bed))
                    .fold(f32::MAX, f32::min)
            } else {
                weighted_dist_sq(curr_pos, w.pts.first().unwrap().pos_bed)
                    .min(weighted_dist_sq(curr_pos, w.pts.last().unwrap().pos_bed))
            };
            if d < best_d {
                best_d = d;
                best_i = i;
            }
        }
        used_waves[best_i] = true;
        let mut chosen = wave_overhangs[best_i].clone();
        orient_chain_for_minimal_travel(&mut chosen, curr_pos, z_penalty);
        curr_pos = if chosen.is_closed {
            chosen.pts.first().unwrap().pos_bed
        } else {
            chosen.pts.last().unwrap().pos_bed
        };
        ordered.push(chosen);
    }

    // 3. Order straight-line circular Bridge chains by nearest neighbor
    let mut used_bridges = vec![false; bridges.len()];
    for _ in 0..bridges.len() {
        let mut best_i = 0usize;
        let mut best_d = f32::MAX;
        for (i, br) in bridges.iter().enumerate() {
            if used_bridges[i] || br.pts.is_empty() {
                continue;
            }
            let d = weighted_dist_sq(curr_pos, br.pts.first().unwrap().pos_bed)
                .min(weighted_dist_sq(curr_pos, br.pts.last().unwrap().pos_bed));
            if d < best_d {
                best_d = d;
                best_i = i;
            }
        }
        used_bridges[best_i] = true;
        let mut chosen = bridges[best_i].clone();
        orient_chain_for_minimal_travel(&mut chosen, curr_pos, z_penalty);
        curr_pos = chosen.pts.last().unwrap().pos_bed;
        ordered.push(chosen);
    }

    // 4. Greedily order remaining infill & gap-fill chains
    let mut used_infill = vec![false; infills.len()];
    for _ in 0..infills.len() {
        let mut best_i = 0usize;
        let mut best_d = f32::MAX;
        for (i, inf) in infills.iter().enumerate() {
            if used_infill[i] || inf.pts.is_empty() {
                continue;
            }
            let d = if inf.is_closed {
                weighted_dist_sq(curr_pos, inf.pts.first().unwrap().pos_bed)
            } else {
                weighted_dist_sq(curr_pos, inf.pts.first().unwrap().pos_bed)
                    .min(weighted_dist_sq(curr_pos, inf.pts.last().unwrap().pos_bed))
            };
            if d < best_d {
                best_d = d;
                best_i = i;
            }
        }
        used_infill[best_i] = true;
        let mut chosen = infills[best_i].clone();
        orient_chain_for_minimal_travel(&mut chosen, curr_pos, z_penalty);
        curr_pos = if chosen.is_closed {
            chosen.pts.first().unwrap().pos_bed
        } else {
            chosen.pts.last().unwrap().pos_bed
        };
        ordered.push(chosen);
    }

    (ordered, curr_pos)
}

/// Orders chains on a layer according to low-to-high elevation bands along `up_bed`
/// (preventing high-Z ridges from being printed before adjacent low-Z valleys on the same
/// non-planar layer), and within each band according to `WallPrintOrder` (`InnerOuterInner`
/// or `OutsideIn`) followed by greedy Z-penalized nearest-neighbor traversal.
fn order_layer_chains(
    chains: Vec<WorkingChain>,
    wall_order: WallPrintOrder,
    start_pos_bed: Vec3,
    z_penalty: f32,
    up_bed: Vec3,
) -> Vec<WorkingChain> {
    if chains.is_empty() {
        return chains;
    }

    let up = up_bed.try_normalize().unwrap_or(Vec3::Z);
    let chain_mean_h = |ch: &WorkingChain| -> f32 {
        if ch.pts.is_empty() {
            return 0.0;
        }
        let sum: f32 = ch.pts.iter().map(|p| p.pos_bed.dot(up)).sum();
        sum / (ch.pts.len() as f32)
    };

    let mut min_h = f32::MAX;
    let mut max_h = f32::MIN;
    let mut mean_hs = Vec::with_capacity(chains.len());
    for ch in &chains {
        let mh = chain_mean_h(ch);
        min_h = min_h.min(mh);
        max_h = max_h.max(mh);
        mean_hs.push(mh);
    }

    // When a non-planar layer spans > 0.45 mm in elevation across chains, partition chains
    // into ascending 0.60 mm elevation bands so low-Z valleys are always printed before
    // adjacent high-Z peaks on the same layer.
    let h_span = max_h - min_h;
    if h_span <= 0.45 {
        return order_chain_band(chains, wall_order, start_pos_bed, z_penalty).0;
    }

    let band_step_mm = 0.60f32;
    let num_bands = ((h_span / band_step_mm).ceil() as usize).clamp(2, 64);
    let mut bands: Vec<Vec<WorkingChain>> = vec![Vec::new(); num_bands];
    for (ch, mh) in chains.into_iter().zip(mean_hs.into_iter()) {
        let b_idx = (((mh - min_h) / band_step_mm).floor() as usize).min(num_bands - 1);
        bands[b_idx].push(ch);
    }

    let mut ordered = Vec::new();
    let mut curr_pos = start_pos_bed;
    for band in bands {
        if band.is_empty() {
            continue;
        }
        let (mut band_ordered, next_pos) =
            order_chain_band(band, wall_order, curr_pos, z_penalty);
        curr_pos = next_pos;
        ordered.append(&mut band_ordered);
    }
    ordered
}

/// Samples along a closed loop's perimeter at arc-length distance `s_target` from `pts[0]`.
fn sample_closed_loop_at_arc_length(pts: &[WorkingVertex], s_target: f32) -> WorkingVertex {
    let n = pts.len();
    let mut accum = 0.0f32;
    for i in 0..n {
        let a = &pts[i];
        let b = &pts[(i + 1) % n];
        let seg_len = (b.pos_bed - a.pos_bed).length();
        if accum + seg_len >= s_target && seg_len > 1.0e-6 {
            let t = ((s_target - accum) / seg_len).clamp(0.0, 1.0);
            return WorkingVertex {
                pos_bed: a.pos_bed.lerp(b.pos_bed, t),
                nozzle_bed: a.nozzle_bed.lerp(b.nozzle_bed, t).normalize_or_zero(),
                width_mm: a.width_mm + t * (b.width_mm - a.width_mm),
                height_mm: a.height_mm + t * (b.height_mm - a.height_mm),
                flow_rate_mult: 1.0,
                kind_override: None,
            };
        }
        accum += seg_len;
    }
    pts[0].clone()
}

/// Converts a `WorkingChain` into an explicit open sequence of `WorkingVertex` points
/// (`segments.len() == pts.len() - 1`) by applying:
/// - **Closed loops (`is_closed == true`)**: 3D Scarf Joint Seam Ramping (`apply_scarf_joint`) or
///   explicit loop closure + seam gap (`apply_seam_gap`) + forward wipe (`apply_wipe_moves`).
/// - **Open paths (`is_closed == false`)**: Pre-retract taper (`apply_pre_retract_taper`) +
///   **backward** wipe retracing the end of the path so the nozzle never wipes outside the boundary!
fn finalize_chain_seam_and_wipe(chain: &WorkingChain, cfg: &GcodeProcessConfig) -> Vec<WorkingVertex> {
    if chain.pts.len() < 2 {
        return Vec::new();
    }

    let mut total_len = 0.0f32;
    for w in chain.pts.windows(2) {
        total_len += (w[1].pos_bed - w[0].pos_bed).length();
    }
    if chain.is_closed {
        total_len += (chain.pts.first().unwrap().pos_bed - chain.pts.last().unwrap().pos_bed).length();
    }

    // 1. Closed outer wall loop with 3D Scarf Joint Seam Ramping (strictly gated on `is_closed`,
    //    `OuterWall`, and perimeter >= 20 mm so internal walls and small pillars are not plowed)
    if chain.is_closed
        && chain.base_kind == PlannedMoveKind::OuterWall
        && cfg.enable_scarf_joint
        && total_len >= (4.0 * cfg.scarf_length_mm).max(20.0)
    {
        let l_scarf = cfg.scarf_length_mm.min(0.35 * total_len).max(1.0);
        let k_steps = ((l_scarf / 0.40).ceil() as usize).clamp(4, 32);
        let ds = l_scarf / (k_steps as f32);
        let h_start = cfg.scarf_start_height_ratio.clamp(0.05, 0.50);

        let mut out: Vec<WorkingVertex> = Vec::with_capacity(chain.pts.len() + 2 * k_steps + 4);

        // Lead-in bottom wedge (s in [0, l_scarf]): displaced downward along -nozzle_bed
        for k in 0..=k_steps {
            let tau = (k as f32) / (k_steps as f32);
            let h_frac = h_start + (1.0 - h_start) * tau;
            let mut v = sample_closed_loop_at_arc_length(&chain.pts, (k as f32) * ds);
            let z_axis = v.nozzle_bed.try_normalize().unwrap_or(Vec3::Z);
            v.pos_bed -= z_axis * ((1.0 - h_frac) * v.height_mm);
            v.flow_rate_mult = h_frac;
            v.kind_override = Some(PlannedMoveKind::ScarfSeam);
            out.push(v);
        }

        // Main loop body (s in (l_scarf, total_len])
        let n = chain.pts.len();
        let mut accum = 0.0f32;
        for i in 0..n {
            let a = &chain.pts[i];
            let b = &chain.pts[(i + 1) % n];
            let seg_len = (b.pos_bed - a.pos_bed).length();
            if accum + seg_len > l_scarf + 1.0e-4 {
                let mut v = b.clone();
                v.flow_rate_mult = 1.0;
                v.kind_override = None;
                out.push(v);
            }
            accum += seg_len;
        }

        // Lead-out top overlap wedge (s in (0, l_scarf]): rides on nominal trajectory with complementary flow
        for k in 1..=k_steps {
            let tau = (k as f32) / (k_steps as f32);
            let f_out = ((1.0 - h_start) * (1.0 - tau)).max(0.04);
            let mut v = sample_closed_loop_at_arc_length(&chain.pts, (k as f32) * ds);
            v.flow_rate_mult = f_out;
            v.kind_override = Some(PlannedMoveKind::ScarfSeam);
            out.push(v);
        }

        // Forward wipe along closed loop (never descending below the scarf lead-out end)
        if cfg.wipe_length_mm > 1.0e-3 {
            let p_end = out.last().unwrap().pos_bed;
            let n_end = out
                .last()
                .unwrap()
                .nozzle_bed
                .try_normalize()
                .unwrap_or(Vec3::Z);
            let mut w_v = sample_closed_loop_at_arc_length(&chain.pts, l_scarf + cfg.wipe_length_mm);
            let down_comp = (w_v.pos_bed - p_end).dot(n_end).min(0.0);
            w_v.pos_bed -= n_end * down_comp;
            w_v.flow_rate_mult = 0.0;
            w_v.kind_override = Some(PlannedMoveKind::Wipe);
            out.push(w_v);
        }

        return out;
    }

    // 2. Closed loop without scarf joint: explicit closure + optional seam gap + forward wipe
    if chain.is_closed {
        let mut out = chain.pts.clone();
        let mut close_pt = chain.pts[0].clone();
        if cfg.seam_gap_mm > 1.0e-3 && total_len > 4.0 * cfg.seam_gap_mm {
            let last_pos = out.last().unwrap().pos_bed;
            let close_vec = close_pt.pos_bed - last_pos;
            let close_len = close_vec.length();
            if close_len > cfg.seam_gap_mm * 1.5 {
                let pre_gap_t = ((close_len - cfg.seam_gap_mm) / close_len).clamp(0.1, 0.95);
                let mut pre_gap = close_pt.clone();
                pre_gap.pos_bed = last_pos.lerp(close_pt.pos_bed, pre_gap_t);
                out.push(pre_gap);
                close_pt.flow_rate_mult = 0.0;
                close_pt.kind_override = Some(PlannedMoveKind::Wipe);
            }
        }
        out.push(close_pt);

        // Forward wipe along p0 -> p1 on closed loop (never descending below close_pt)
        if cfg.wipe_length_mm > 1.0e-3 && chain.pts.len() >= 2 {
            let p_close = out.last().unwrap().pos_bed;
            let n_close = out
                .last()
                .unwrap()
                .nozzle_bed
                .try_normalize()
                .unwrap_or(Vec3::Z);
            let mut w_v = sample_closed_loop_at_arc_length(&chain.pts, cfg.wipe_length_mm);
            let down_comp = (w_v.pos_bed - p_close).dot(n_close).min(0.0);
            w_v.pos_bed -= n_close * down_comp;
            w_v.flow_rate_mult = 0.0;
            w_v.kind_override = Some(PlannedMoveKind::Wipe);
            out.push(w_v);
        }
        return out;
    }

    // 3. Open path (`!chain.is_closed`, including open non-planar walls and infill):
    //    Apply optional pre-retract taper near the tail, and wipe BACKWARD along the tail
    //    so the nozzle never protrudes past the open endpoint!
    let mut out = chain.pts.clone();
    if !chain.base_kind.is_wall()
        && cfg.pre_retract_taper_mm > 1.0e-3
        && total_len >= 2.0 * cfg.pre_retract_taper_mm
    {
        let n = out.len();
        let last_idx = n - 1;
        out[last_idx].flow_rate_mult = 0.90;
    }

    if cfg.wipe_length_mm > 1.0e-3 && chain.base_kind.is_wall() && out.len() >= 2 {
        let p_end = out[out.len() - 1].pos_bed;
        let p_prev = out[out.len() - 2].pos_bed;
        let n_end = out[out.len() - 1]
            .nozzle_bed
            .try_normalize()
            .unwrap_or(Vec3::Z);
        let seg_vec = p_prev - p_end; // BACKWARD along the last segment!
        // Keep the backward wipe level (or slightly climbing) along `n_end` so retracing a steep
        // non-planar ramp never descends into the newly extruded peak at `p_end`.
        let down_comp = seg_vec.dot(n_end).min(0.0);
        let level_vec = seg_vec - n_end * down_comp;
        let seg_len = level_vec.length();
        if seg_len > 1.0e-4 {
            let wipe_d = cfg.wipe_length_mm.min(0.85 * seg_len);
            let mut wipe_v = out[out.len() - 1].clone();
            wipe_v.pos_bed = p_end + (level_vec / seg_len) * wipe_d;
            wipe_v.flow_rate_mult = 0.0;
            wipe_v.kind_override = Some(PlannedMoveKind::Wipe);
            out.push(wipe_v);
        }
    }

    out
}

/// Subdivides long non-planar segments (> 2.5 mm, or > 0.6 mm when nozzle axis rotates > ~2°)
/// where nozzle orientation or Z changes, ensuring downstream 5-axis `(B, C)` IK, flow, and
/// inclination math accurately track curved conformal surfaces.
fn subdivide_long_segments(pts: Vec<WorkingVertex>) -> Vec<WorkingVertex> {
    if pts.len() < 2 {
        return pts;
    }
    let mut out: Vec<WorkingVertex> = Vec::with_capacity(pts.len() + 8);
    out.push(pts[0].clone());

    for w in pts.windows(2) {
        let a = &w[0];
        let b = &w[1];
        let len = (b.pos_bed - a.pos_bed).length();
        let n_dot = a.nozzle_bed.dot(b.nozzle_bed).clamp(-1.0, 1.0);
        let h_diff = (b.height_mm - a.height_mm).abs();
        let angle_deg = n_dot.acos().to_degrees();

        let needs_subdiv = (len > 2.5 && (len > 6.0 || n_dot < 0.998 || h_diff > 0.02))
            || (len > 0.6 && angle_deg > 2.0);
        if needs_subdiv {
            let steps_len = (len / 2.5).ceil() as usize;
            let steps_ang = (angle_deg / 2.0).ceil() as usize;
            let steps = steps_len.max(steps_ang).clamp(2, 16);
            for s in 1..steps {
                let t = (s as f32) / (steps as f32);
                out.push(WorkingVertex {
                    pos_bed: a.pos_bed.lerp(b.pos_bed, t),
                    nozzle_bed: a.nozzle_bed.lerp(b.nozzle_bed, t).normalize_or_zero(),
                    width_mm: a.width_mm + t * (b.width_mm - a.width_mm),
                    height_mm: a.height_mm + t * (b.height_mm - a.height_mm),
                    flow_rate_mult: a.flow_rate_mult + t * (b.flow_rate_mult - a.flow_rate_mult),
                    kind_override: b.kind_override,
                });
            }
        }
        out.push(b.clone());
    }
    out
}

/// Assigns continuous 5-axis `(B, C)` rotary postures (`b0_deg`, `c0_deg`, `b1_deg`, `c1_deg`)
/// to each `PlannedSegment` and couples rotary speed/acceleration limits into `nominal_speed_mms`
/// and `accel_mms2` when `FiveAxisKinematicsMode::FiveAxisBC` is active.
///
/// Includes **zenith azimuth lookahead pre-positioning**: when traveling at $B \approx 0^\circ$
/// before a tilted extrusion path, `C` rotates to the upcoming path's entry azimuth during the
/// non-extruding `Travel` move so `C` is already aligned when the nozzle touches down.
pub(crate) fn assign_five_axis_bc_postures_and_rotary_limits(
    segments: &mut [PlannedSegment],
    machine: &MachineKinematicsConfig,
    max_tool_tilt_deg: f32,
    curr_b_deg: &mut f32,
    curr_c_deg: &mut f32,
) {
    let n = segments.len();
    if n == 0 {
        return;
    }
    let is_five_axis = machine.kinematics_mode == FiveAxisKinematicsMode::FiveAxisBC;
    let max_b_spd = machine.max_b_speed_deg_s.max(1.0);
    let max_c_spd = machine.max_c_speed_deg_s.max(1.0);
    let max_b_acc = machine.b_accel_deg_s2.max(10.0);
    let max_c_acc = machine.c_accel_deg_s2.max(10.0);

    for i in 0..n {
        let mut solved_by_lookahead = false;
        // Zenith azimuth lookahead pre-positioning during non-extruding Travel moves:
        // If we are on a Travel move and the nozzle is vertical (B ~ 0), look ahead to the
        // first tilted extruding segment and pre-align C during the Travel move.
        if segments[i].kind == PlannedMoveKind::Travel {
            let (b_start_check, _, _) = machine.nozzle_axis_to_bc_deg(
                segments[i].nozzle_bed_0,
                *curr_b_deg,
                *curr_c_deg,
                max_tool_tilt_deg,
            );
            if b_start_check.abs() < 0.25 {
                for future in &segments[(i + 1)..n] {
                    if future.kind.is_extruding() {
                        let (b_fut0, c_fut0, _) = machine.nozzle_axis_to_bc_deg(
                            future.nozzle_bed_0,
                            *curr_b_deg,
                            *curr_c_deg,
                            max_tool_tilt_deg,
                        );
                        if b_fut0.abs() >= 0.25 {
                            let b0 = *curr_b_deg;
                            let c0 = *curr_c_deg;
                            let (b1, _, _) = machine.nozzle_axis_to_bc_deg(
                                segments[i].nozzle_bed_1,
                                b0,
                                c_fut0,
                                max_tool_tilt_deg,
                            );
                            let c1 = c_fut0;
                            segments[i].b0_deg = b0;
                            segments[i].c0_deg = c0;
                            segments[i].b1_deg = b1;
                            segments[i].c1_deg = c1;
                            *curr_b_deg = b1;
                            *curr_c_deg = c1;
                            solved_by_lookahead = true;
                            break;
                        }
                        let (b_fut1, c_fut1, _) = machine.nozzle_axis_to_bc_deg(
                            future.nozzle_bed_1,
                            *curr_b_deg,
                            *curr_c_deg,
                            max_tool_tilt_deg,
                        );
                        if b_fut1.abs() >= 0.25 {
                            let b0 = *curr_b_deg;
                            let c0 = *curr_c_deg;
                            let (b1, _, _) = machine.nozzle_axis_to_bc_deg(
                                segments[i].nozzle_bed_1,
                                b0,
                                c_fut1,
                                max_tool_tilt_deg,
                            );
                            let c1 = c_fut1;
                            segments[i].b0_deg = b0;
                            segments[i].c0_deg = c0;
                            segments[i].b1_deg = b1;
                            segments[i].c1_deg = c1;
                            *curr_b_deg = b1;
                            *curr_c_deg = c1;
                            solved_by_lookahead = true;
                            break;
                        }
                    }
                }
            }
        }

        // Standard endpoint IK if not already populated by zenith lookahead above
        if !solved_by_lookahead {
            let b0 = *curr_b_deg;
            let c0 = *curr_c_deg;
            let (b1, c1, _) = machine.nozzle_axis_to_bc_deg(
                segments[i].nozzle_bed_1,
                b0,
                c0,
                max_tool_tilt_deg,
            );
            segments[i].b0_deg = b0;
            segments[i].c0_deg = c0;
            segments[i].b1_deg = b1;
            segments[i].c1_deg = c1;
            *curr_b_deg = b1;
            *curr_c_deg = c1;
        }

        if is_five_axis {
            let seg = &mut segments[i];
            let db = (seg.b1_deg - seg.b0_deg).abs();
            let dc = (seg.c1_deg - seg.c0_deg).abs();
            if db > 1.0e-4 || dc > 1.0e-4 {
                let t_rot_spd = (db / max_b_spd).max(dc / max_c_spd);
                if t_rot_spd > 1.0e-5 {
                    let v_rot_cap = (seg.length_mm / t_rot_spd).max(0.5);
                    seg.nominal_speed_mms = seg.nominal_speed_mms.min(v_rot_cap);
                    seg.cruise_speed_mms = seg.cruise_speed_mms.min(seg.nominal_speed_mms);
                    seg.duration_s = seg
                        .duration_s
                        .max(t_rot_spd)
                        .max((seg.length_mm / seg.nominal_speed_mms.max(0.5)).max(1.0e-4));
                }
                if db > 0.05 {
                    let a_b_lin = (max_b_acc * (seg.length_mm / db)).max(50.0);
                    seg.accel_mms2 = seg.accel_mms2.min(a_b_lin);
                }
                if dc > 0.05 {
                    let a_c_lin = (max_c_acc * (seg.length_mm / dc)).max(50.0);
                    seg.accel_mms2 = seg.accel_mms2.min(a_c_lin);
                }
            }
        }
    }
}

/// Computes Klipper's Square Corner Velocity junction speed $v_{\text{corner}}$ ($\text{mm/s}$)
/// and effective corner-rounding shortcut arc radius $R_{\text{eff}}$ ($\text{mm}$) without
/// blowing up as deflection angle $\alpha \to 0$.
pub fn compute_klipper_corner_kinematics(
    d_in: Vec3,
    d_out: Vec3,
    scv_mms: f32,
    accel_mms2: f32,
    v_in_mms: f32,
    v_out_mms: f32,
    len_in_mm: f32,
    len_out_mm: f32,
) -> (f32, f32) {
    let v_limit = v_in_mms.min(v_out_mms).max(0.0);
    if v_limit <= 1.0e-4 {
        return (0.0, 0.0);
    }

    let cos_alpha = d_in.dot(d_out).clamp(-1.0, 1.0);
    // Nearly collinear (\alpha < ~1.1 deg): full cruise speed and zero corner shortcut radius
    if cos_alpha > 0.9998 {
        return (v_limit, 0.0);
    }
    // Hairpin 180-deg reversal: full stop at vertex
    if cos_alpha < -0.9998 {
        return (0.0, 0.0);
    }

    // Half-angle trig for deflection angle \alpha \in (0, \pi):
    // \cos(\alpha/2) = \sqrt{(1 + \cos\alpha)/2}, \sin(\alpha/2) = \sqrt{(1 - \cos\alpha)/2}
    let cos_half = (0.5 * (1.0 + cos_alpha)).max(0.0).sqrt();
    let sin_half = (0.5 * (1.0 - cos_alpha)).max(1.0e-6).sqrt();

    // At 90-deg turn (\cos\alpha = 0), \cos(\alpha/2) = \sin(\alpha/2) = 1/\sqrt{2},
    // so r_jd = (1/\sqrt{2}) / (1 - 1/\sqrt{2}) = 1 / (\sqrt{2} - 1), giving v_corner = scv_mms!
    let denom = (1.0 - cos_half).max(1.0e-5);
    let r_jd = cos_half / denom;
    let sqrt2_minus_1 = std::f32::consts::SQRT_2 - 1.0;
    let v_jd = scv_mms * (r_jd * sqrt2_minus_1).max(0.0).sqrt();
    let v_corner = v_jd.min(v_limit);

    // Physical junction deviation \delta_jd = SCV^2 (\sqrt{2} - 1) / a
    let delta_jd = (scv_mms * scv_mms * sqrt2_minus_1) / accel_mms2.max(10.0);
    let speed_ratio_sq = if v_jd > 1.0e-4 {
        (v_corner / v_jd).clamp(0.0, 1.0).powi(2)
    } else {
        0.0
    };
    let tan_half = (sin_half / cos_half.max(0.05)).min(4.0);
    let max_geom_r = (0.40 * len_in_mm.min(len_out_mm)) / tan_half.max(0.1);
    let r_jd_capped = r_jd.min(1.0 / sqrt2_minus_1);
    let r_eff = (delta_jd * r_jd_capped * speed_ratio_sq)
        .min(max_geom_r)
        .max(0.0);

    (v_corner, r_eff)
}

/// Solves the Stepper Motor Torque / Back-EMF Roll-Off ODE (`Section 3.3.2`) for forward
/// acceleration $a(v) = a_0 (1 - v / v_{\max})$ over distance $d$, safely clamped below $0.98 v_{\max}$.
pub fn stepper_max_reachable_velocity(
    v_entry_mms: f32,
    accel_0_mms2: f32,
    dist_mm: f32,
    v_max_stepper_mms: f32,
    use_stepper_dynamics: bool,
) -> f32 {
    if dist_mm <= 1.0e-6 || accel_0_mms2 <= 1.0e-4 {
        return v_entry_mms;
    }
    let v_kin = (v_entry_mms * v_entry_mms + 2.0 * accel_0_mms2 * dist_mm).sqrt();
    if !use_stepper_dynamics || v_max_stepper_mms <= 10.0 {
        return v_kin;
    }

    let v_cap = 0.98 * v_max_stepper_mms;
    if v_entry_mms >= v_cap {
        return v_entry_mms;
    }

    let potential = |v: f32| -> f32 {
        let u = (1.0 - v / v_max_stepper_mms).clamp(0.02, 1.0);
        v_max_stepper_mms * v_max_stepper_mms * (u - u.ln())
    };

    let target_f = potential(v_entry_mms) + accel_0_mms2 * dist_mm;
    if target_f >= potential(v_cap) {
        return v_cap.min(v_kin);
    }

    let mut v = v_kin.min(v_cap);
    for _ in 0..8 {
        let f_v = potential(v) - target_f;
        if f_v.abs() < 1.0e-3 {
            break;
        }
        let df_dv = v.max(1.0) / (1.0 - v / v_max_stepper_mms).clamp(0.02, 1.0);
        let lo = v_entry_mms.min(v_cap);
        v = (v - f_v / df_dv).clamp(lo, v_cap);
    }
    v.min(v_kin)
}

/// Runs forward/backward chained velocity lookahead and enforces `minimum_cruise_ratio`
/// without ever clamping `cruise_speed_mms` below `max(entry_speed_mms, exit_speed_mms)`.
fn plan_velocity_lookahead(segments: &mut [PlannedSegment], cfg: &GcodeProcessConfig) {
    let n = segments.len();
    if n == 0 {
        return;
    }

    // 1. Initialize junction boundary speeds from Klipper SCV corner kinematics
    let mut junction_max = vec![0.0f32; n + 1];
    junction_max[0] = 0.0;
    junction_max[n] = 0.0;

    for i in 1..n {
        let prev = &segments[i - 1];
        let curr = &segments[i];
        // Full stop at Z-hop vertical transitions or retraction boundaries
        if prev.kind == PlannedMoveKind::ZHop
            || curr.kind == PlannedMoveKind::ZHop
            || curr.retract_before_mm > 0.0
            || prev.unretract_after_mm > 0.0
        {
            junction_max[i] = 0.0;
            continue;
        }
        let d_in = (prev.p1_bed - prev.p0_bed).try_normalize().unwrap_or(Vec3::X);
        let d_out = (curr.p1_bed - curr.p0_bed).try_normalize().unwrap_or(Vec3::X);
        let accel = prev.accel_mms2.min(curr.accel_mms2);
        let (v_corner, _) = compute_klipper_corner_kinematics(
            d_in,
            d_out,
            cfg.square_corner_velocity_mms,
            accel,
            prev.nominal_speed_mms,
            curr.nominal_speed_mms,
            prev.length_mm,
            curr.length_mm,
        );
        junction_max[i] = v_corner;
    }

    // 2. Forward acceleration pass (with optional stepper torque roll-off)
    for i in 0..n {
        let v_in = junction_max[i].min(segments[i].nominal_speed_mms);
        junction_max[i] = v_in;
        let v_reach = stepper_max_reachable_velocity(
            v_in,
            segments[i].accel_mms2,
            segments[i].length_mm,
            cfg.max_stepper_speed_mms,
            cfg.use_stepper_dynamics,
        );
        junction_max[i + 1] = junction_max[i + 1]
            .min(v_reach)
            .min(segments[i].nominal_speed_mms);
    }

    // 3. Backward braking pass (full braking deceleration `a_0`)
    for i in (0..n).rev() {
        let v_out = junction_max[i + 1];
        let v_max_entry =
            (v_out * v_out + 2.0 * segments[i].accel_mms2 * segments[i].length_mm).sqrt();
        if v_max_entry < junction_max[i] {
            junction_max[i] = v_max_entry;
        }
    }

    // 4. Finalize per-segment `(entry_speed_mms, cruise_speed_mms, exit_speed_mms, duration_s)`
    let rho = cfg.minimum_cruise_ratio.clamp(0.0, 0.95);
    for i in 0..n {
        let seg = &mut segments[i];
        let v0 = junction_max[i].min(seg.nominal_speed_mms);
        let v1 = junction_max[i + 1].min(seg.nominal_speed_mms);
        let a = seg.accel_mms2.max(10.0);
        let d = seg.length_mm.max(1.0e-5);

        // Peak speed limited by minimum_cruise_ratio, strictly clamped >= max(v0, v1)
        // so ramp distances s_accel and s_decel are NEVER negative!
        let v_floor = v0.max(v1);
        let v_ceil = seg.nominal_speed_mms.max(v_floor);
        let v_mcr = (0.5 * (v0 * v0 + v1 * v1) + a * (1.0 - rho) * d).sqrt();
        let v_cruise = v_mcr.clamp(v_floor, v_ceil).max(1.0);

        let s_accel = ((v_cruise * v_cruise - v0 * v0) / (2.0 * a)).max(0.0);
        let s_decel = ((v_cruise * v_cruise - v1 * v1) / (2.0 * a)).max(0.0);
        let s_cruise = (d - s_accel - s_decel).max(0.0);

        let t_accel = (v_cruise - v0).max(0.0) / a;
        let t_decel = (v_cruise - v1).max(0.0) / a;
        let t_cruise = s_cruise / v_cruise.max(1.0);

        seg.entry_speed_mms = v0;
        seg.cruise_speed_mms = v_cruise;
        seg.exit_speed_mms = v1;
        seg.duration_s = (t_accel + t_cruise + t_decel).max(1.0e-4);
    }
}

/// Optional slicer-side Pressure Advance subdivision (`Section 3.6`): splits acceleration and
/// deceleration ramps into sub-segments when `enable_slicer_pa_subdivision` is enabled.
fn subdivide_segments_for_slicer_pa(
    segments: Vec<PlannedSegment>,
    filament_area_mm2: f32,
    fluid: &FluidDynamicsModel,
    cfg: &GcodeProcessConfig,
) -> Vec<PlannedSegment> {
    if !cfg.enable_slicer_pa_subdivision || !cfg.enable_dynamic_pa {
        return segments;
    }

    let mut out = Vec::with_capacity(segments.len());
    for seg in segments {
        if !seg.kind.is_extruding()
            || seg.delta_e_mm <= 1.0e-5
            || seg.length_mm < 3.0
            || (seg.cruise_speed_mms - seg.entry_speed_mms.min(seg.exit_speed_mms)) < 40.0
        {
            out.push(seg);
            continue;
        }

        let n_sub = ((seg.length_mm / 1.5).ceil() as usize).clamp(2, 3);
        let inv_n = 1.0 / (n_sub as f32);
        for k in 0..n_sub {
            let t0 = (k as f32) * inv_n;
            let t1 = ((k + 1) as f32) * inv_n;
            let v0 = seg.entry_speed_mms + t0 * (seg.exit_speed_mms - seg.entry_speed_mms);
            let v1 = seg.entry_speed_mms + t1 * (seg.exit_speed_mms - seg.entry_speed_mms);
            let v_mid = (0.5 * (v0 + v1)).max(seg.cruise_speed_mms * 0.75);
            let sub_len = seg.length_mm * inv_n;
            let sub_dt = (sub_len / v_mid.max(1.0)).max(1.0e-4);
            let base_de = seg.delta_e_mm * inv_n;
            let q_sub = (base_de * filament_area_mm2) / sub_dt;
            let pa_sub = fluid.pressure_advance_s(q_sub, true, cfg.pa_low_flow_s);

            let mut sub = seg.clone();
            sub.p0_local = seg.p0_local.lerp(seg.p1_local, t0);
            sub.p1_local = seg.p0_local.lerp(seg.p1_local, t1);
            sub.p0_bed = seg.p0_bed.lerp(seg.p1_bed, t0);
            sub.p1_bed = seg.p0_bed.lerp(seg.p1_bed, t1);
            sub.b0_deg = seg.b0_deg + t0 * (seg.b1_deg - seg.b0_deg);
            sub.b1_deg = seg.b0_deg + t1 * (seg.b1_deg - seg.b0_deg);
            sub.c0_deg = seg.c0_deg + t0 * (seg.c1_deg - seg.c0_deg);
            sub.c1_deg = seg.c0_deg + t1 * (seg.c1_deg - seg.c0_deg);
            sub.length_mm = sub_len;
            sub.entry_speed_mms = v0;
            sub.cruise_speed_mms = seg.cruise_speed_mms;
            sub.exit_speed_mms = v1;
            sub.duration_s = sub_dt;
            sub.delta_e_mm = base_de;
            sub.volumetric_flow_mm3s = q_sub;
            sub.pressure_advance_s = pa_sub;
            if k > 0 {
                sub.retract_before_mm = 0.0;
            }
            if k + 1 < n_sub {
                sub.unretract_after_mm = 0.0;
            }
            out.push(sub);
        }
    }
    out
}

/// Expands template placeholders (`{nozzle_temp}`, `{bed_temp}`, `{tool_id}`, `{bbox_min_x}`, etc.)
/// in `start_gcode_template` and `end_gcode_template`.
pub fn expand_gcode_template(
    template: &str,
    tool_id: u32,
    nozzle_temp_c: f32,
    bed_temp_c: f32,
    nozzle_diam_mm: f32,
    bbox_min: Vec3,
    bbox_max: Vec3,
    layer_count: usize,
) -> String {
    let center = 0.5 * (bbox_min + bbox_max);
    template
        .replace("{tool_id}", &format!("{tool_id}"))
        .replace("{nozzle_temp}", &format!("{nozzle_temp_c:.0}"))
        .replace("{bed_temp}", &format!("{bed_temp_c:.0}"))
        .replace("{nozzle_diameter}", &format!("{nozzle_diam_mm:.2}"))
        .replace("{bbox_min_x}", &format!("{:.2}", bbox_min.x))
        .replace("{bbox_max_x}", &format!("{:.2}", bbox_max.x))
        .replace("{bbox_min_y}", &format!("{:.2}", bbox_min.y))
        .replace("{bbox_max_y}", &format!("{:.2}", bbox_max.y))
        .replace("{bbox_min_z}", &format!("{:.2}", bbox_min.z))
        .replace("{bbox_max_z}", &format!("{:.2}", bbox_max.z))
        .replace("{bbox_center_x}", &format!("{:.2}", center.x))
        .replace("{bbox_center_y}", &format!("{:.2}", center.y))
        .replace("{layer_count}", &format!("{layer_count}"))
}

/// Sanitizes an object name into a valid Klipper `EXCLUDE_OBJECT` identifier (`[A-Za-z0-9_.-]`,
/// no whitespace or special characters).
pub fn sanitize_exclude_object_name(raw_name: &str, object_id: u64) -> String {
    let stem = std::path::Path::new(raw_name)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(raw_name)
        .trim();

    let mut sanitized = String::with_capacity(stem.len() + 8);
    let mut prev_underscore = false;
    for ch in stem.chars() {
        if ch.is_ascii_alphanumeric() || ch == '.' || ch == '-' {
            sanitized.push(ch);
            prev_underscore = false;
        } else if !prev_underscore && !sanitized.is_empty() {
            sanitized.push('_');
            prev_underscore = true;
        }
    }
    let trimmed = sanitized.trim_matches('_');
    if trimmed.is_empty() {
        format!("Object_{object_id}")
    } else if object_id > 1 && !trimmed.ends_with(&format!("_{object_id}")) {
        format!("{trimmed}_{object_id}")
    } else {
        trimmed.to_string()
    }
}

/// Builds a compact, closed 2D convex hull polygon `[[x0,y0],[x1,y1],...,[x0,y0]]`
/// (with `<= 24` vertices and `>= 1.0 mm` vertex spacing) from a set of 2D XY points
/// and fallback bounding-box corners, matching Klipper's `EXCLUDE_OBJECT_DEFINE POLYGON=...`.
pub fn build_exclude_object_hull_polygon_2d(
    mut pts: Vec<[f32; 2]>,
    bbox_min_xy: [f32; 2],
    bbox_max_xy: [f32; 2],
) -> Vec<[f32; 2]> {
    #[inline]
    fn cross_2d(o: [f32; 2], a: [f32; 2], b: [f32; 2]) -> f32 {
        (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    }

    pts.sort_by(|a, b| {
        a[0].partial_cmp(&b[0])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a[1].partial_cmp(&b[1]).unwrap_or(std::cmp::Ordering::Equal))
    });
    pts.dedup_by(|a, b| (a[0] - b[0]).abs() < 0.05 && (a[1] - b[1]).abs() < 0.05);

    let mut hull: Vec<[f32; 2]> = Vec::new();
    if pts.len() >= 3 {
        let mut lower: Vec<[f32; 2]> = Vec::with_capacity(pts.len());
        for &p in &pts {
            while lower.len() >= 2 && cross_2d(lower[lower.len() - 2], lower[lower.len() - 1], p) <= 0.0 {
                lower.pop();
            }
            lower.push(p);
        }
        let mut upper: Vec<[f32; 2]> = Vec::with_capacity(pts.len());
        for &p in pts.iter().rev() {
            while upper.len() >= 2 && cross_2d(upper[upper.len() - 2], upper[upper.len() - 1], p) <= 0.0 {
                upper.pop();
            }
            upper.push(p);
        }
        lower.pop();
        upper.pop();
        hull = lower;
        hull.extend(upper);
    }

    // Fallback to the 2D bounding-box rectangle if hull is degenerate (< 3 vertices)
    if hull.len() < 3 {
        let x0 = bbox_min_xy[0];
        let y0 = bbox_min_xy[1];
        let x1 = bbox_max_xy[0].max(x0 + 0.1);
        let y1 = bbox_max_xy[1].max(y0 + 0.1);
        return vec![[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]];
    }

    // Prune closely-spaced vertices (< 1.0 mm) and limit vertex count to <= 24 so the
    // `EXCLUDE_OBJECT_DEFINE` line remains compact for Klipper & Moonraker
    let mut simplified: Vec<[f32; 2]> = Vec::with_capacity(hull.len() + 1);
    for p in hull {
        if let Some(&last) = simplified.last() {
            let dx = p[0] - last[0];
            let dy = p[1] - last[1];
            if dx * dx + dy * dy < 1.0 && simplified.len() >= 3 {
                continue;
            }
        }
        simplified.push(p);
    }
    if simplified.len() > 24 {
        let n = simplified.len();
        simplified = (0..24).map(|k| simplified[(k * n) / 24]).collect();
    }
    if let Some(&first) = simplified.first() {
        if simplified
            .last()
            .is_none_or(|last| (last[0] - first[0]).abs() > 1.0e-4 || (last[1] - first[1]).abs() > 1.0e-4)
        {
            simplified.push(first);
        }
    }
    simplified
}

/// Computes a compact 2D convex hull polygon `[[x0,y0],[x1,y1],...,[x0,y0]]` in G-code bed
/// coordinates (`mm`) for Klipper's `EXCLUDE_OBJECT_DEFINE POLYGON=...` parameter.
pub fn compute_exclude_object_polygon_2d(
    layers: &[PlannedLayerGcode],
    bbox_min: Vec3,
    bbox_max: Vec3,
) -> Vec<[f32; 2]> {
    // Collect quantized 2D points (0.05 mm grid) from extruding moves to build the convex hull
    let mut pts: Vec<[f32; 2]> = Vec::new();
    for layer in layers {
        for seg in &layer.segments {
            if seg.kind.is_extruding() && seg.delta_e_mm > 1.0e-6 {
                pts.push([seg.p0_bed.x, seg.p0_bed.y]);
                pts.push([seg.p1_bed.x, seg.p1_bed.y]);
            }
        }
    }
    build_exclude_object_hull_polygon_2d(
        pts,
        [bbox_min.x, bbox_min.y],
        [bbox_max.x, bbox_max.y],
    )
}

/// Formats a Klipper `EXCLUDE_OBJECT_DEFINE` G-code line with zero internal spaces inside
/// `CENTER=` and `POLYGON=` so Klipper's whitespace-splitting command parser handles it cleanly.
pub fn format_exclude_object_define(name: &str, center: [f32; 2], polygon: &[[f32; 2]]) -> String {
    let mut poly_json = String::with_capacity(polygon.len() * 18 + 4);
    poly_json.push('[');
    for (i, pt) in polygon.iter().enumerate() {
        if i > 0 {
            poly_json.push(',');
        }
        poly_json.push_str(&format!("[{:.4},{:.4}]", pt[0], pt[1]));
    }
    poly_json.push(']');
    format!(
        "EXCLUDE_OBJECT_DEFINE NAME={} CENTER={:.4},{:.4} POLYGON={}",
        name, center[0], center[1], poly_json
    )
}

/// Standard RFC 4648 Base64 encoder for embedding PNG thumbnails in G-code comments.
fn encode_base64(bytes: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;

        out.push(TABLE[((triple >> 18) & 0x3F) as usize] as char);
        out.push(TABLE[((triple >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            out.push(TABLE[((triple >> 6) & 0x3F) as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(TABLE[(triple & 0x3F) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}

/// Encodes an RGBA8 image buffer (`width * height * 4` bytes) into PNG format in memory.
fn encode_rgba_to_png(width: u32, height: u32, rgba: &[u8]) -> Vec<u8> {
    let mut png_bytes = Vec::with_capacity((width * height) as usize + 256);
    {
        let mut encoder = png::Encoder::new(&mut png_bytes, width, height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.set_compression(png::Compression::Fast);
        if let Ok(mut writer) = encoder.write_header() {
            let _ = writer.write_image_data(rgba);
        }
    }
    png_bytes
}

/// Formats a single PNG thumbnail block (`; thumbnail begin WxH B64_LEN` ... `; thumbnail end`)
/// adhering to Moonraker's `BaseSlicer.parse_thumbnails` specification in `metadata.py`.
fn format_thumbnail_gcode_block(width: u32, height: u32, rgba: &[u8]) -> String {
    let png_bytes = encode_rgba_to_png(width, height, rgba);
    let b64 = encode_base64(&png_bytes);
    let mut out = String::with_capacity(b64.len() + (b64.len() / 78 + 4) * 4 + 64);
    out.push_str(";\n");
    out.push_str(&format!(
        "; thumbnail begin {}x{} {}\n",
        width,
        height,
        b64.len()
    ));
    for chunk in b64.as_bytes().chunks(78) {
        out.push_str("; ");
        if let Ok(s) = std::str::from_utf8(chunk) {
            out.push_str(s);
        }
        out.push('\n');
    }
    out.push_str("; thumbnail end\n");
    out.push_str(";\n");
    out
}

/// Renders an isometric 3D Z-buffered shaded preview of the planned non-planar toolpaths into
/// `300x300` and downsampled `32x32` RGBA buffers, and returns the formatted Moonraker thumbnail blocks.
pub fn render_toolpath_preview_thumbnails(
    layers: &[PlannedLayerGcode],
    bbox_min: Vec3,
    bbox_max: Vec3,
    up_bed: Vec3,
) -> String {
    const HI_W: usize = 300;
    const HI_H: usize = 300;
    const LO_W: usize = 32;
    const LO_H: usize = 32;

    let mut rgba_hi = vec![0u8; HI_W * HI_H * 4];
    let mut z_buf = vec![f32::INFINITY; HI_W * HI_H];

    // Orthonormal isometric camera basis aligned with `up_bed` (respecting AGENTS.md Rule 2)
    let world_up = up_bed.try_normalize().unwrap_or(Vec3::Z);
    let ref_x = if world_up.dot(Vec3::X).abs() < 0.9 {
        Vec3::X
    } else {
        Vec3::Y
    };
    let bed_tangent_x = (ref_x - world_up * ref_x.dot(world_up))
        .try_normalize()
        .unwrap_or(Vec3::X);
    let bed_tangent_y = world_up.cross(bed_tangent_x).try_normalize().unwrap_or(Vec3::Y);

    // Isometric 3/4 view direction looking toward object center
    let yaw = -38.0f32.to_radians();
    let pitch = -28.0f32.to_radians();
    let horiz_dir = bed_tangent_x * yaw.cos() + bed_tangent_y * yaw.sin();
    let view_forward = (horiz_dir * pitch.cos() + world_up * pitch.sin())
        .try_normalize()
        .unwrap_or(Vec3::ONE.normalize());
    let cam_right = view_forward
        .cross(world_up)
        .try_normalize()
        .unwrap_or(bed_tangent_x);
    let cam_up = cam_right
        .cross(view_forward)
        .try_normalize()
        .unwrap_or(world_up);

    let center = 0.5 * (bbox_min + bbox_max);
    let ext = (bbox_max - bbox_min).max(Vec3::splat(1.0));
    let half_ext = 0.5 * ext;

    // Project all 8 AABB corners onto `(cam_right, cam_up)` to auto-frame the object
    let mut proj_min_x = f32::INFINITY;
    let mut proj_max_x = f32::NEG_INFINITY;
    let mut proj_min_y = f32::INFINITY;
    let mut proj_max_y = f32::NEG_INFINITY;

    for sx in [-1.0f32, 1.0] {
        for sy in [-1.0f32, 1.0] {
            for sz in [-1.0f32, 1.0] {
                let offset = Vec3::new(sx * half_ext.x, sy * half_ext.y, sz * half_ext.z);
                let px = offset.dot(cam_right);
                let py = offset.dot(cam_up);
                proj_min_x = proj_min_x.min(px);
                proj_max_x = proj_max_x.max(px);
                proj_min_y = proj_min_y.min(py);
                proj_max_y = proj_max_y.max(py);
            }
        }
    }

    let span_x = (proj_max_x - proj_min_x).max(1.0);
    let span_y = (proj_max_y - proj_min_y).max(1.0);
    let scale = (0.84 * (HI_W as f32)) / span_x.max(span_y);
    let proj_cx = 0.5 * (proj_min_x + proj_max_x);
    let proj_cy = 0.5 * (proj_min_y + proj_max_y);

    let project_pt = |p: Vec3| -> (f32, f32, f32) {
        let rel = p - center;
        let x_cam = rel.dot(cam_right) - proj_cx;
        let y_cam = rel.dot(cam_up) - proj_cy;
        let z_cam = rel.dot(view_forward);
        let sx = 0.5 * (HI_W as f32) + x_cam * scale;
        let sy = 0.5 * (HI_H as f32) - y_cam * scale;
        (sx, sy, z_cam)
    };

    let light_dir = (-0.45 * view_forward + 0.50 * cam_right + 0.74 * cam_up)
        .try_normalize()
        .unwrap_or(world_up);
    let z_span = (bbox_max.dot(world_up) - bbox_min.dot(world_up)).max(1.0);
    let z_base = bbox_min.dot(world_up);

    // Count total extruding segments and stride if very dense so thumbnail rendering stays < 3 ms
    let total_ext: usize = layers
        .iter()
        .map(|l| {
            l.segments
                .iter()
                .filter(|s| s.kind.is_extruding() && s.delta_e_mm > 1.0e-6)
                .count()
        })
        .sum();
    let seg_stride = (total_ext / 25_000).max(1);

    let mut seg_counter = 0usize;
    for (layer_idx, layer) in layers.iter().enumerate() {
        let is_boundary_layer = layer_idx == 0 || layer_idx + 1 == layers.len();
        for seg in &layer.segments {
            if !seg.kind.is_extruding() || seg.delta_e_mm <= 1.0e-6 {
                continue;
            }
            seg_counter += 1;
            if seg_stride > 1
                && !is_boundary_layer
                && seg.kind != PlannedMoveKind::OuterWall
                && (seg_counter % seg_stride != 0)
            {
                continue;
            }

            let (x0, y0, z0) = project_pt(seg.p0_bed);
            let (x1, y1, z1) = project_pt(seg.p1_bed);
            let r_px = (0.55 * seg.width_mm * scale).clamp(1.25, 4.2);

            let min_px = ((x0.min(x1) - r_px - 0.5).floor() as i32).max(0) as usize;
            let max_px = ((x0.max(x1) + r_px + 0.5).ceil() as i32).min(HI_W as i32 - 1) as usize;
            let min_py = ((y0.min(y1) - r_px - 0.5).floor() as i32).max(0) as usize;
            let max_py = ((y0.max(y1) + r_px + 0.5).ceil() as i32).min(HI_H as i32 - 1) as usize;
            if min_px > max_px || min_py > max_py {
                continue;
            }

            let base_rgb: [f32; 3] = match seg.kind {
                PlannedMoveKind::OuterWall => [0.98, 0.52, 0.16],
                PlannedMoveKind::ScarfSeam => [0.92, 0.42, 0.24],
                PlannedMoveKind::InnerWall => [0.32, 0.78, 0.44],
                PlannedMoveKind::SolidInfill | PlannedMoveKind::GapFill => [0.24, 0.76, 0.88],
                PlannedMoveKind::SparseInfill => [0.92, 0.74, 0.22],
                PlannedMoveKind::Bridge => [0.20, 0.56, 0.98],
                PlannedMoveKind::WaveOverhang => [0.88, 0.32, 0.86],
                PlannedMoveKind::Support => [0.18, 0.82, 0.72],
                PlannedMoveKind::SupportInterface => [0.10, 0.94, 0.88],
                _ => [0.85, 0.50, 0.18],
            };

            let n3d = (seg.nozzle_bed_0 + seg.nozzle_bed_1)
                .try_normalize()
                .unwrap_or(world_up);
            let h_u = (((0.5 * (seg.p0_bed + seg.p1_bed)).dot(world_up) - z_base) / z_span)
                .clamp(0.0, 1.0);
            let height_boost = 0.82 + 0.24 * h_u;
            let surf_diffuse = n3d.dot(light_dir).max(0.0);

            let dx = x1 - x0;
            let dy = y1 - y0;
            let len_sq = dx * dx + dy * dy;

            for py in min_py..=max_py {
                let fy = py as f32 + 0.5;
                for px in min_px..=max_px {
                    let fx = px as f32 + 0.5;
                    let t = if len_sq > 1.0e-6 {
                        (((fx - x0) * dx + (fy - y0) * dy) / len_sq).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    let cx = x0 + t * dx;
                    let cy = y0 + t * dy;
                    let dist_sq = (fx - cx) * (fx - cx) + (fy - cy) * (fy - cy);
                    let max_r = r_px + 0.45;
                    if dist_sq > max_r * max_r {
                        continue;
                    }
                    let dist = dist_sq.sqrt();
                    let u_rad = (dist / r_px).clamp(0.0, 1.0);
                    let cyl_z = (1.0 - u_rad * u_rad).max(0.0).sqrt();
                    let z_depth = z0 + t * (z1 - z0) - cyl_z * (0.5 * seg.height_mm);

                    let idx = py * HI_W + px;
                    if z_depth <= z_buf[idx] + 0.08 {
                        let shade = (0.36 + 0.44 * cyl_z + 0.26 * surf_diffuse) * height_boost;
                        let spec = cyl_z.powi(6) * 0.14;
                        let r = ((base_rgb[0] * shade + spec).clamp(0.0, 1.0) * 255.0) as u8;
                        let g = ((base_rgb[1] * shade + spec).clamp(0.0, 1.0) * 255.0) as u8;
                        let b = ((base_rgb[2] * shade + spec).clamp(0.0, 1.0) * 255.0) as u8;
                        let alpha = if dist <= r_px - 0.35 {
                            255u8
                        } else {
                            (((max_r - dist) / 0.80).clamp(0.0, 1.0) * 255.0) as u8
                        };

                        if alpha >= 200 || rgba_hi[idx * 4 + 3] == 0 {
                            if alpha >= 200 {
                                z_buf[idx] = z_depth;
                            }
                            rgba_hi[idx * 4] = r;
                            rgba_hi[idx * 4 + 1] = g;
                            rgba_hi[idx * 4 + 2] = b;
                            rgba_hi[idx * 4 + 3] = rgba_hi[idx * 4 + 3].max(alpha);
                        }
                    }
                }
            }
        }
    }

    // Box-filter downsample 300x300 -> 32x32 with premultiplied alpha weighting
    let mut rgba_lo = vec![0u8; LO_W * LO_H * 4];
    for ly in 0..LO_H {
        let y_start = (ly * HI_H) / LO_H;
        let y_end = (((ly + 1) * HI_H) / LO_H).max(y_start + 1);
        for lx in 0..LO_W {
            let x_start = (lx * HI_W) / LO_W;
            let x_end = (((lx + 1) * HI_W) / LO_W).max(x_start + 1);

            let mut sum_r = 0u32;
            let mut sum_g = 0u32;
            let mut sum_b = 0u32;
            let mut sum_a = 0u32;
            let mut count = 0u32;

            for hy in y_start..y_end {
                for hx in x_start..x_end {
                    let p = (hy * HI_W + hx) * 4;
                    let a = rgba_hi[p + 3] as u32;
                    sum_r += (rgba_hi[p] as u32) * a;
                    sum_g += (rgba_hi[p + 1] as u32) * a;
                    sum_b += (rgba_hi[p + 2] as u32) * a;
                    sum_a += a;
                    count += 1;
                }
            }

            let dst = (ly * LO_W + lx) * 4;
            if sum_a > 0 && count > 0 {
                rgba_lo[dst] = (sum_r / sum_a).min(255) as u8;
                rgba_lo[dst + 1] = (sum_g / sum_a).min(255) as u8;
                rgba_lo[dst + 2] = (sum_b / sum_a).min(255) as u8;
                rgba_lo[dst + 3] = (sum_a / count).min(255) as u8;
            }
        }
    }

    let mut blocks = format_thumbnail_gcode_block(LO_W as u32, LO_H as u32, &rgba_lo);
    blocks.push_str(&format_thumbnail_gcode_block(HI_W as u32, HI_H as u32, &rgba_hi));
    blocks
}

/// Formats `total_time_s` into the human-readable duration string expected by Moonraker's
/// `PrusaSlicer._parse_time_to_seconds` (`r"(\d+)d"`, `r"(\d+)h"`, `r"(\d+)m"`, `r"(\d+)s"`).
pub fn format_moonraker_estimated_time(total_time_s: f32) -> String {
    let total_secs = total_time_s.max(1.0).round() as u64;
    let days = total_secs / 86_400;
    let hours = (total_secs % 86_400) / 3_600;
    let mins = (total_secs % 3_600) / 60;
    let secs = total_secs % 60;

    if days > 0 {
        format!("{days}d {hours}h {mins}m {secs}s")
    } else if hours > 0 {
        format!("{hours}h {mins}m {secs}s")
    } else if mins > 0 {
        format!("{mins}m {secs}s")
    } else {
        format!("{secs}s")
    }
}

/// Formats the Moonraker-compatible `PrusaSlicer` metadata footer block (`metadata.py`).
#[allow(clippy::too_many_arguments)]
fn format_moonraker_metadata_footer(
    total_time_s: f32,
    total_filament_mm: f32,
    total_filament_volume_mm3: f32,
    total_filament_grams: f32,
    first_layer_height_mm: f32,
    target_layer_height_mm: f32,
    max_print_height_mm: f32,
    layer_count: usize,
    nozzle_diam_mm: f32,
    filament_diam_mm: f32,
    filament_type: &str,
    bed_temp_c: f32,
    nozzle_temp_c: f32,
) -> String {
    let total_filament_cm3 = total_filament_volume_mm3 / 1000.0;
    let time_str = format_moonraker_estimated_time(total_time_s);
    let clean_filament_type = if filament_type.trim().is_empty() {
        "PLA"
    } else {
        filament_type.trim()
    };
    format!(
        concat!(
            "\n; --- Moonraker / PrusaSlicer Metadata Footer ---\n",
            "; filament used [mm] = {:.2}\n",
            "; filament used [cm3] = {:.2}\n",
            "; filament used [g] = {:.2}\n",
            "; total filament used [g] = {:.2}\n",
            "; estimated printing time (normal mode) = {}\n",
            "; first_layer_height = {:.4}\n",
            "; layer_height = {:.4}\n",
            "; max_print_height = {:.4}\n",
            "; total layers count = {}\n",
            "; nozzle_diameter = {:.2}\n",
            "; filament_diameter = {:.2}\n",
            "; filament_type = {}\n",
            "; filament_settings_id = \"Manifold2 Non-Planar\"\n",
            "; first_layer_bed_temperature = {:.0}\n",
            "; first_layer_temperature = {:.0}\n"
        ),
        total_filament_mm,
        total_filament_cm3,
        total_filament_grams,
        total_filament_grams,
        time_str,
        first_layer_height_mm,
        target_layer_height_mm,
        max_print_height_mm.max(0.0),
        layer_count,
        nozzle_diam_mm,
        filament_diam_mm,
        clean_filament_type,
        bed_temp_c,
        nozzle_temp_c,
    )
}

/// First-layer bed contact footprint and overall part height profile used for automatic
/// brim sizing and lever-arm / cantilevered bending slowdown calculations.
struct BedFootprintProfile {
    basis: BedUpBasis,
    h_base_mm: f32,
    h_top_mm: f32,
    w_base_min_mm: f32,
    part_base_pts_uv: Vec<(f32, f32)>,
}

impl BedFootprintProfile {
    fn from_raw_toolpaths(
        raw_toolpaths: &ObjectToolpathsResult,
        obj_world_tf: Transform,
        bed_origin_offset: Vec3,
        basis: BedUpBasis,
        layer0_chains: &[WorkingChain],
    ) -> Self {
        let mut h_min = f32::MAX;
        let mut h_max = f32::MIN;
        for layer in &raw_toolpaths.layers {
            for path in &layer.paths {
                for pt in &path.points {
                    let p_bed =
                        obj_world_tf.transform_point(Vec3::new(pt.x, pt.y, pt.z)) + bed_origin_offset;
                    let (_, _, h) = basis.project_uvh(p_bed);
                    h_min = h_min.min(h);
                    h_max = h_max.max(h);
                }
            }
        }
        if !h_min.is_finite() || h_min > h_max {
            h_min = 0.20;
            h_max = 0.20;
        }

        let mut part_pts_uv: Vec<(f32, f32)> = Vec::new();
        let mut all_pts_uv: Vec<(f32, f32)> = Vec::new();
        let mut u_min_part = f32::MAX;
        let mut u_max_part = f32::MIN;
        let mut v_min_part = f32::MAX;
        let mut v_max_part = f32::MIN;
        let mut u_min_all = f32::MAX;
        let mut u_max_all = f32::MIN;
        let mut v_min_all = f32::MAX;
        let mut v_max_all = f32::MIN;

        for ch in layer0_chains {
            let is_supp = ch.base_kind.is_support();
            for vtx in &ch.pts {
                let (u, v, _) = basis.project_uvh(vtx.pos_bed);
                u_min_all = u_min_all.min(u);
                u_max_all = u_max_all.max(u);
                v_min_all = v_min_all.min(v);
                v_max_all = v_max_all.max(v);
                all_pts_uv.push((u, v));
                if !is_supp {
                    u_min_part = u_min_part.min(u);
                    u_max_part = u_max_part.max(u);
                    v_min_part = v_min_part.min(v);
                    v_max_part = v_max_all.max(v);
                    part_pts_uv.push((u, v));
                }
            }
        }

        let (base_pts, w_min) = if part_pts_uv.len() >= 3 {
            let wu = (u_max_part - u_min_part).max(2.0);
            let wv = (v_max_part - v_min_part).max(2.0);
            // Subsample to <= 160 points for fast cantilevered distance queries
            let stride = (part_pts_uv.len() / 160).max(1);
            let sub: Vec<(f32, f32)> = part_pts_uv.into_iter().step_by(stride).collect();
            (sub, wu.min(wv))
        } else if !all_pts_uv.is_empty() {
            let wu = (u_max_all - u_min_all).max(2.0);
            let wv = (v_max_all - v_min_all).max(2.0);
            let stride = (all_pts_uv.len() / 160).max(1);
            let sub: Vec<(f32, f32)> = all_pts_uv.into_iter().step_by(stride).collect();
            (sub, wu.min(wv))
        } else {
            (vec![(0.0, 0.0)], 20.0)
        };

        Self {
            basis,
            h_base_mm: h_min,
            h_top_mm: h_max,
            w_base_min_mm: w_min.max(2.0),
            part_base_pts_uv: base_pts,
        }
    }

    /// Global height-to-narrowest-base lever ratio $\Lambda_{\text{global}} = (H_{\text{top}} - H_{\text{base}}) / W_{\text{base,min}}$.
    fn global_lever_ratio(&self) -> f32 {
        ((self.h_top_mm - self.h_base_mm).max(0.0)) / self.w_base_min_mm.max(2.0)
    }

    /// Lateral cantilevered overhang distance from `(u, v)` to the nearest Layer 0 part contact point.
    fn lateral_cantilever_dist_mm(&self, u: f32, v: f32) -> f32 {
        let mut best_d2 = f32::MAX;
        for &(bu, bv) in &self.part_base_pts_uv {
            let d2 = (u - bu) * (u - bu) + (v - bv) * (v - bv);
            if d2 < best_d2 {
                best_d2 = d2;
            }
        }
        if best_d2 < f32::MAX {
            best_d2.sqrt()
        } else {
            0.0
        }
    }
}

/// Generates outward concentric first-layer brim loops around Layer 0 extrusions when either
/// `gcfg.brim_width_mm > 0.0` or `gcfg.enable_auto_brim` detects a tall / narrow-base part
/// ($\Lambda_{\text{global}} > 2.2$ and `total_layers >= 5`).
fn generate_first_layer_brim_chains(
    layer0_chains: &[WorkingChain],
    footprint: &BedFootprintProfile,
    total_layers: usize,
    gcfg: &GcodeProcessConfig,
    line_width_mm: f32,
    first_layer_height_mm: f32,
) -> Vec<WorkingChain> {
    if layer0_chains.is_empty() {
        return Vec::new();
    }

    let eff_brim_width_mm = if gcfg.brim_width_mm > 0.05 {
        gcfg.brim_width_mm
    } else if gcfg.enable_auto_brim && total_layers >= 5 {
        let ratio = footprint.global_lever_ratio();
        if ratio > 2.2 {
            (1.6 * (ratio - 2.0)).clamp(2.4, 8.0)
        } else {
            0.0
        }
    } else {
        0.0
    };

    let w_line = line_width_mm.max(0.25);
    let num_loops = (eff_brim_width_mm / w_line).round() as usize;
    if num_loops == 0 {
        return Vec::new();
    }

    // Collect sampled 2D segments `(u0, v0, u1, v1, h_avg)` from Layer 0 chains
    let mut u_min = f32::MAX;
    let mut u_max = f32::MIN;
    let mut v_min = f32::MAX;
    let mut v_max = f32::MIN;
    let mut h_sum = 0.0f32;
    let mut h_cnt = 0usize;
    let mut segs_uv: Vec<(f32, f32, f32, f32)> = Vec::new();

    for ch in layer0_chains {
        if ch.pts.len() < 2 {
            continue;
        }
        for w in ch.pts.windows(2) {
            let (u0, v0, h0) = footprint.basis.project_uvh(w[0].pos_bed);
            let (u1, v1, h1) = footprint.basis.project_uvh(w[1].pos_bed);
            u_min = u_min.min(u0).min(u1);
            u_max = u_max.max(u0).max(u1);
            v_min = v_min.min(v0).min(v1);
            v_max = v_max.max(v0).max(v1);
            h_sum += 0.5 * (h0 + h1);
            h_cnt += 1;
            segs_uv.push((u0, v0, u1, v1));
        }
        if ch.is_closed && ch.pts.len() >= 3 {
            let last = ch.pts.last().unwrap();
            let first = ch.pts.first().unwrap();
            let (u0, v0, _) = footprint.basis.project_uvh(last.pos_bed);
            let (u1, v1, _) = footprint.basis.project_uvh(first.pos_bed);
            segs_uv.push((u0, v0, u1, v1));
        }
    }
    if segs_uv.is_empty() || u_min > u_max {
        return Vec::new();
    }

    let h_brim = if h_cnt > 0 {
        h_sum / (h_cnt as f32)
    } else {
        footprint.h_base_mm
    };
    let max_offset = gcfg.brim_gap_mm + (num_loops as f32 + 1.5) * w_line;
    let step_mm = 0.40f32;
    let pad_mm = max_offset + 2.0 * step_mm;

    let origin_u = u_min - pad_mm;
    let origin_v = v_min - pad_mm;
    let nx = (((u_max - u_min + 2.0 * pad_mm) / step_mm).ceil() as usize).clamp(8, 700);
    let ny = (((v_max - v_min + 2.0 * pad_mm) / step_mm).ceil() as usize).clamp(8, 700);

    // Compute unsigned distance field on `(nx + 1) x (ny + 1)` grid by rasterizing each segment's
    // padded bounding box `[min - max_offset .. max + max_offset]`
    let gw = nx + 1;
    let gh = ny + 1;
    let mut dist_grid = vec![max_offset + 2.0; gw * gh];
    let r_pad = max_offset + step_mm;

    for &(u0, v0, u1, v1) in &segs_uv {
        let su_min = u0.min(u1) - r_pad;
        let su_max = u0.max(u1) + r_pad;
        let sv_min = v0.min(v1) - r_pad;
        let sv_max = v0.max(v1) + r_pad;

        let ix0 = (((su_min - origin_u) / step_mm).floor() as isize).max(0) as usize;
        let ix1 = (((su_max - origin_u) / step_mm).ceil() as isize).min(nx as isize) as usize;
        let iy0 = (((sv_min - origin_v) / step_mm).floor() as isize).max(0) as usize;
        let iy1 = (((sv_max - origin_v) / step_mm).ceil() as isize).min(ny as isize) as usize;

        let du = u1 - u0;
        let dv = v1 - v0;
        let len_sq = du * du + dv * dv;

        for iy in iy0..=iy1 {
            let gv = origin_v + (iy as f32) * step_mm;
            let row_off = iy * gw;
            for ix in ix0..=ix1 {
                let gu = origin_u + (ix as f32) * step_mm;
                let t = if len_sq > 1.0e-8 {
                    (((gu - u0) * du + (gv - v0) * dv) / len_sq).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let cu = u0 + t * du;
                let cv = v0 + t * dv;
                let d = ((gu - cu) * (gu - cu) + (gv - cv) * (gv - cv)).sqrt();
                let cell = &mut dist_grid[row_off + ix];
                if d < *cell {
                    *cell = d;
                }
            }
        }
    }

    // Flood-fill exterior cells (`dist > 0.55 * w_line`) starting from the grid border `(0, 0)`
    // so internal holes/cavities inside the part footprint are excluded from outward brim loops.
    let mut is_exterior = vec![false; gw * gh];
    let mut queue: Vec<(usize, usize)> = Vec::with_capacity(gw * gh / 4);
    let inner_barrier = (0.45 * w_line).max(0.22);
    for ix in 0..gw {
        for &iy in &[0usize, gh - 1] {
            let idx = iy * gw + ix;
            if !is_exterior[idx] && dist_grid[idx] > inner_barrier {
                is_exterior[idx] = true;
                queue.push((ix, iy));
            }
        }
    }
    for iy in 0..gh {
        for &ix in &[0usize, gw - 1] {
            let idx = iy * gw + ix;
            if !is_exterior[idx] && dist_grid[idx] > inner_barrier {
                is_exterior[idx] = true;
                queue.push((ix, iy));
            }
        }
    }
    let mut head = 0usize;
    while head < queue.len() {
        let (cx, cy) = queue[head];
        head += 1;
        let neighbors = [
            (cx.wrapping_sub(1), cy),
            (cx + 1, cy),
            (cx, cy.wrapping_sub(1)),
            (cx, cy + 1),
        ];
        for (nx_i, ny_i) in neighbors {
            if nx_i < gw && ny_i < gh {
                let nidx = ny_i * gw + nx_i;
                if !is_exterior[nidx] && dist_grid[nidx] > inner_barrier {
                    is_exterior[nidx] = true;
                    queue.push((nx_i, ny_i));
                }
            }
        }
    }
    // Set interior cells to 0.0 so Marching Squares only extracts outward exterior contours
    for idx in 0..(gw * gh) {
        if !is_exterior[idx] {
            dist_grid[idx] = 0.0;
        }
    }

    let mut brim_chains: Vec<WorkingChain> = Vec::new();
    // Emit from outside in (`loop_i = num_loops - 1` down to `0`) so the nozzle primes on the
    // outermost loop and progressively locks each inner loop against the part base.
    for loop_i in (0..num_loops).rev() {
        let iso = gcfg.brim_gap_mm + ((loop_i as f32) + 0.5) * w_line + 0.5 * w_line;
        let loops_uv = extract_marching_squares_loops(&dist_grid, gw, gh, origin_u, origin_v, step_mm, iso);
        for loop_pts in loops_uv {
            if loop_pts.len() < 6 {
                continue;
            }
            let pts: Vec<WorkingVertex> = loop_pts
                .into_iter()
                .map(|(u, v)| WorkingVertex {
                    pos_bed: footprint.basis.unproject_uvh(u, v, h_brim),
                    nozzle_bed: footprint.basis.up_bed,
                    width_mm: w_line,
                    height_mm: first_layer_height_mm.max(0.08),
                    flow_rate_mult: 1.0,
                    kind_override: None,
                })
                .collect();
            let mut wc = WorkingChain {
                pts,
                base_kind: PlannedMoveKind::OuterWall,
                wall_index: 0,
                is_closed: true,
            };
            simplify_working_chain(&mut wc, 0.05 * w_line);
            if wc.pts.len() >= 4 {
                brim_chains.push(wc);
            }
        }
    }

    brim_chains
}

/// Extracts closed 2D iso-contours from a scalar grid using Marching Squares + segment stitching.
fn extract_marching_squares_loops(
    grid: &[f32],
    gw: usize,
    gh: usize,
    origin_u: f32,
    origin_v: f32,
    step_mm: f32,
    iso: f32,
) -> Vec<Vec<(f32, f32)>> {
    let mut raw_segs: Vec<((f32, f32), (f32, f32))> = Vec::new();
    let interp = |a: f32, b: f32| -> f32 {
        let denom = b - a;
        if denom.abs() > 1.0e-6 {
            ((iso - a) / denom).clamp(0.0, 1.0)
        } else {
            0.5
        }
    };

    for iy in 0..(gh.saturating_sub(1)) {
        let v0 = origin_v + (iy as f32) * step_mm;
        let v1 = v0 + step_mm;
        for ix in 0..(gw.saturating_sub(1)) {
            let u0 = origin_u + (ix as f32) * step_mm;
            let u1 = u0 + step_mm;

            let f00 = grid[iy * gw + ix];
            let f10 = grid[iy * gw + ix + 1];
            let f11 = grid[(iy + 1) * gw + ix + 1];
            let f01 = grid[(iy + 1) * gw + ix];

            let mut mask = 0u8;
            if f00 >= iso {
                mask |= 1;
            }
            if f10 >= iso {
                mask |= 2;
            }
            if f11 >= iso {
                mask |= 4;
            }
            if f01 >= iso {
                mask |= 8;
            }
            if mask == 0 || mask == 15 {
                continue;
            }

            let e_bottom = (u0 + interp(f00, f10) * step_mm, v0);
            let e_right = (u1, v0 + interp(f10, f11) * step_mm);
            let e_top = (u0 + interp(f01, f11) * step_mm, v1);
            let e_left = (u0, v0 + interp(f00, f01) * step_mm);

            match mask {
                1 | 14 => raw_segs.push((e_left, e_bottom)),
                2 | 13 => raw_segs.push((e_bottom, e_right)),
                3 | 12 => raw_segs.push((e_left, e_right)),
                4 | 11 => raw_segs.push((e_right, e_top)),
                5 => {
                    raw_segs.push((e_left, e_top));
                    raw_segs.push((e_bottom, e_right));
                }
                6 | 9 => raw_segs.push((e_bottom, e_top)),
                7 | 8 => raw_segs.push((e_left, e_top)),
                10 => {
                    raw_segs.push((e_left, e_bottom));
                    raw_segs.push((e_right, e_top));
                }
                _ => {}
            }
        }
    }

    if raw_segs.is_empty() {
        return Vec::new();
    }

    let mut used = vec![false; raw_segs.len()];
    let mut loops: Vec<Vec<(f32, f32)>> = Vec::new();
    let tol_sq = (0.15 * step_mm) * (0.15 * step_mm);

    for start_i in 0..raw_segs.len() {
        if used[start_i] {
            continue;
        }
        used[start_i] = true;
        let mut chain = vec![raw_segs[start_i].0, raw_segs[start_i].1];

        loop {
            let tail = *chain.last().unwrap();
            let mut found_next = None;
            for (j, &(p0, p1)) in raw_segs.iter().enumerate() {
                if used[j] {
                    continue;
                }
                let d0 = (p0.0 - tail.0) * (p0.0 - tail.0) + (p0.1 - tail.1) * (p0.1 - tail.1);
                if d0 <= tol_sq {
                    found_next = Some((j, p1));
                    break;
                }
                let d1 = (p1.0 - tail.0) * (p1.0 - tail.0) + (p1.1 - tail.1) * (p1.1 - tail.1);
                if d1 <= tol_sq {
                    found_next = Some((j, p0));
                    break;
                }
            }
            if let Some((j, next_pt)) = found_next {
                used[j] = true;
                chain.push(next_pt);
            } else {
                break;
            }
        }

        if chain.len() >= 6 {
            let head = chain[0];
            let tail = *chain.last().unwrap();
            let close_d2 = (head.0 - tail.0) * (head.0 - tail.0) + (head.1 - tail.1) * (head.1 - tail.1);
            if close_d2 <= 4.0 * step_mm * step_mm {
                chain.pop();
                loops.push(chain);
            }
        }
    }

    loops
}

/// Applies automatic lever-arm and curled-corner speed/acceleration slowdown on upper layers
/// (`layer_ord > 0` when `gcfg.enable_lever_arm_slowdown` is enabled).
/// Throttles speeds and accelerations where:
/// 1. The global bed torque lever ratio $\Lambda_{\text{bed}} = (h + 1.35 \, d_{\text{cant}}) / W_{\text{base,min}}$
///    or local slender-prong ratio exceeds `lever_arm_reference_ratio`.
/// 2. Acute corners, overhangs, or steep non-planar climbs are prone to upward thermal corner curl.
fn apply_lever_arm_and_curl_slowdown(
    layer_segments: &mut [PlannedSegment],
    layer_ord: usize,
    footprint: &BedFootprintProfile,
    gcfg: &GcodeProcessConfig,
) {
    if !gcfg.enable_lever_arm_slowdown || layer_ord == 0 || layer_segments.is_empty() {
        return;
    }

    // 1. Measure current layer's part-extrusion horizontal span to detect slender cantilevered prongs
    let mut u_min_layer = f32::MAX;
    let mut u_max_layer = f32::MIN;
    let mut v_min_layer = f32::MAX;
    let mut v_max_layer = f32::MIN;
    let mut part_ext_cnt = 0usize;

    for seg in layer_segments.iter() {
        if seg.kind.is_extruding() && !seg.kind.is_support() && seg.delta_e_mm > 1.0e-6 {
            let (u, v, _) = footprint.basis.project_uvh(0.5 * (seg.p0_bed + seg.p1_bed));
            u_min_layer = u_min_layer.min(u);
            u_max_layer = u_max_layer.max(u);
            v_min_layer = v_min_layer.min(v);
            v_max_layer = v_max_layer.max(v);
            part_ext_cnt += 1;
        }
    }
    let w_layer_min = if part_ext_cnt > 0 && u_min_layer <= u_max_layer {
        (u_max_layer - u_min_layer)
            .min(v_max_layer - v_min_layer)
            .max(2.5)
    } else {
        footprint.w_base_min_mm
    };

    // 2. Compute per-segment raw corner/overhang/climb curl risk `curl_raw[i] \in [0, 1]`
    let n = layer_segments.len();
    let mut curl_raw = vec![0.0f32; n];
    for i in 0..n {
        let seg = &layer_segments[i];
        if !seg.kind.is_extruding() || seg.delta_e_mm <= 1.0e-6 {
            continue;
        }
        let d_curr = (seg.p1_bed - seg.p0_bed)
            .try_normalize()
            .unwrap_or(footprint.basis.tangent_u);
        let z_noz = (seg.nozzle_bed_0 + seg.nozzle_bed_1)
            .try_normalize()
            .unwrap_or(footprint.basis.up_bed);

        // Turning angle deflection at segment entry and exit
        let mut max_turn_sin = 0.0f32;
        if i > 0 && layer_segments[i - 1].kind.is_extruding() {
            let d_prev = (layer_segments[i - 1].p1_bed - layer_segments[i - 1].p0_bed)
                .try_normalize()
                .unwrap_or(d_curr);
            let cos_a = d_prev.dot(d_curr).clamp(-1.0, 1.0);
            let sin_half = (0.5 * (1.0 - cos_a)).max(0.0).sqrt();
            max_turn_sin = max_turn_sin.max(sin_half);
        }
        if i + 1 < n && layer_segments[i + 1].kind.is_extruding() {
            let d_next = (layer_segments[i + 1].p1_bed - layer_segments[i + 1].p0_bed)
                .try_normalize()
                .unwrap_or(d_curr);
            let cos_a = d_curr.dot(d_next).clamp(-1.0, 1.0);
            let sin_half = (0.5 * (1.0 - cos_a)).max(0.0).sqrt();
            max_turn_sin = max_turn_sin.max(sin_half);
        }

        let r_corner = ((max_turn_sin - 0.18) / 0.55).clamp(0.0, 1.0);
        let climb_sin = d_curr.dot(footprint.basis.up_bed).abs();
        let tilt_sin = (1.0 - z_noz.dot(footprint.basis.up_bed).abs().clamp(0.0, 1.0)).sqrt();
        let r_overhang = match seg.kind {
            PlannedMoveKind::WaveOverhang | PlannedMoveKind::Bridge => 0.85,
            PlannedMoveKind::OuterWall => (1.5 * climb_sin + 0.9 * tilt_sin).clamp(0.0, 1.0),
            _ => (1.1 * climb_sin + 0.6 * tilt_sin).clamp(0.0, 1.0),
        };
        curl_raw[i] = (0.65 * r_corner + 0.55 * r_overhang).clamp(0.0, 1.0);
    }

    // 3. Smooth `curl_raw` across a +/- 3.0 mm arc-length neighborhood along contiguous extrusions
    let mut curl_smooth = curl_raw.clone();
    for i in 0..n {
        if !layer_segments[i].kind.is_extruding() {
            continue;
        }
        let mut max_c = curl_raw[i];
        let mut dist_back = 0.0f32;
        let mut j = i;
        while j > 0 && dist_back < 3.0 {
            j -= 1;
            if !layer_segments[j].kind.is_extruding() {
                break;
            }
            dist_back += layer_segments[j].length_mm;
            let w = (1.0 - dist_back / 3.5).clamp(0.0, 1.0);
            max_c = max_c.max(curl_raw[j] * w);
        }
        let mut dist_fwd = 0.0f32;
        let mut k = i + 1;
        while k < n && dist_fwd < 3.0 {
            if !layer_segments[k].kind.is_extruding() {
                break;
            }
            dist_fwd += layer_segments[k].length_mm;
            let w = (1.0 - dist_fwd / 3.5).clamp(0.0, 1.0);
            max_c = max_c.max(curl_raw[k] * w);
            k += 1;
        }
        curl_smooth[i] = max_c;
    }

    let ref_ratio = gcfg.lever_arm_reference_ratio.max(0.5);
    let min_frac = gcfg.lever_arm_min_speed_fraction.clamp(0.10, 1.0);

    for (i, seg) in layer_segments.iter_mut().enumerate() {
        if !seg.kind.is_extruding() || seg.delta_e_mm <= 1.0e-6 {
            continue;
        }
        let mid_bed = 0.5 * (seg.p0_bed + seg.p1_bed);
        let (u, v, h) = footprint.basis.project_uvh(mid_bed);
        let h_rel = (h - footprint.h_base_mm).max(0.0);
        if h_rel < 2.0 {
            continue;
        }

        let d_cant = footprint.lateral_cantilever_dist_mm(u, v);
        let lambda_bed = (h_rel + 1.35 * d_cant) / footprint.w_base_min_mm.max(2.0);
        let lambda_prong = (0.45 * h_rel) / w_layer_min.max(2.5);
        let lambda_eff = lambda_bed.max(lambda_prong);

        let excess_lever = ((lambda_eff - ref_ratio) / ref_ratio).max(0.0);
        let c_curl = curl_smooth[i];
        if excess_lever <= 1.0e-4 && (lambda_eff < 0.65 * ref_ratio || c_curl <= 0.15) {
            continue;
        }

        // Base lever slowdown + extra deceleration when traversing curled-corner risk zones at height
        let height_gate = (lambda_eff / ref_ratio).clamp(0.25, 2.0);
        let denom = 1.0 + 0.42 * excess_lever + 0.70 * c_curl * height_gate;
        let speed_scale = (1.0 / denom).clamp(min_frac, 1.0);
        if speed_scale >= 0.995 {
            continue;
        }

        // Scale linear speed and acceleration (even more aggressively on acceleration to cap reaction torque)
        let accel_scale = (speed_scale * (1.0 - 0.30 * c_curl))
            .clamp((min_frac * 0.65).max(0.12), 1.0);

        seg.nominal_speed_mms = (seg.nominal_speed_mms * speed_scale).max(5.0);
        seg.entry_speed_mms = (seg.entry_speed_mms * speed_scale).max(5.0);
        seg.cruise_speed_mms = (seg.cruise_speed_mms * speed_scale).max(5.0);
        seg.exit_speed_mms = (seg.exit_speed_mms * speed_scale).max(5.0);
        seg.accel_mms2 = (seg.accel_mms2 * accel_scale).max(250.0);
        seg.duration_s = (seg.length_mm / seg.cruise_speed_mms.max(1.0)).max(1.0e-4);
    }
}

/// Computes a dynamic per-segment part cooling fan PWM schedule (`0..=255`) for `layer`
/// matching fan airflow to:
/// 1. Instantaneous volumetric extrusion rate $Q_i$ (`mm^3/s`),
/// 2. Cumulative extruded polymer volume $V_{\text{duct}}$ and cooling airflow residence time
///    $t_{\text{res}}$ within the toolhead's `12.0 mm` cooling duct footprint,
/// 3. Total layer return time $t_{\text{layer}}$ relative to `target_layer_cooling_time_s`, and
/// 4. Local corner/overhang curl risk.
fn compute_layer_fan_pwm_schedule(
    layer: &PlannedLayerGcode,
    layer_ord: usize,
    up_bed: Vec3,
    gcfg: &GcodeProcessConfig,
) -> Vec<u8> {
    let n = layer.segments.len();
    if n == 0 {
        return Vec::new();
    }

    let is_initial_layer = (layer_ord as u32) < gcfg.disable_fan_first_layers;
    let base_fan_pct = if is_initial_layer {
        gcfg.first_layer_fan_speed_percent.clamp(0.0, 100.0)
    } else {
        gcfg.fan_speed_percent.clamp(0.0, 100.0)
    };
    let base_pwm = (base_fan_pct * 2.55).round().clamp(0.0, 255.0) as u8;

    // On initial bed-adhesion layers or when dynamic fan modulation is disabled, preserve
    // standard static/feature-override fan behavior.
    if is_initial_layer || !gcfg.enable_dynamic_fan_speed {
        let mut sched = vec![base_pwm; n];
        let mut last_pwm = base_pwm;
        for (i, seg) in layer.segments.iter().enumerate() {
            if seg.kind.is_extruding() && seg.delta_e_mm > 1.0e-6 {
                let pct = match seg.kind {
                    PlannedMoveKind::Bridge => gcfg.bridge_fan_speed_percent.clamp(0.0, 100.0),
                    PlannedMoveKind::WaveOverhang => {
                        gcfg.wave_overhang_fan_speed_percent.clamp(0.0, 100.0)
                    }
                    _ => base_fan_pct,
                };
                last_pwm = (pct * 2.55).round().clamp(0.0, 255.0) as u8;
            }
            sched[i] = last_pwm;
        }
        return sched;
    }

    let min_pct = gcfg.min_fan_speed_percent.clamp(0.0, 100.0);
    let max_pct = gcfg.max_fan_speed_percent.clamp(min_pct, 100.0);
    let q_ref = (0.65 * gcfg.max_volumetric_flow_mm3s).max(4.0);

    // Total layer time for layer-return thermal boost
    let layer_time_s: f32 = layer
        .segments
        .iter()
        .map(|s| s.duration_s)
        .sum::<f32>()
        .max(0.05);
    let u_layer_time = if layer_time_s < gcfg.target_layer_cooling_time_s {
        ((gcfg.target_layer_cooling_time_s - layer_time_s)
            / gcfg.target_layer_cooling_time_s.max(1.0))
        .clamp(0.0, 1.0)
    } else {
        0.0
    };

    // Cooling duct effective airflow radius around the nozzle tip (12.0 mm)
    let r_duct_mm = 12.0f32;
    let r_duct_sq = r_duct_mm * r_duct_mm;
    let filament_area_mm2 =
        std::f32::consts::FRAC_PI_4 * gcfg.filament_diameter_mm * gcfg.filament_diameter_mm;

    let mut raw_pct = vec![min_pct; n];
    for i in 0..n {
        let seg = &layer.segments[i];
        if !seg.kind.is_extruding() || seg.delta_e_mm <= 1.0e-6 {
            raw_pct[i] = if i > 0 { raw_pct[i - 1] } else { min_pct };
            continue;
        }

        // Dedicated feature overrides for free-air Bridges and Wave Overhangs
        if seg.kind == PlannedMoveKind::Bridge {
            raw_pct[i] = gcfg.bridge_fan_speed_percent.clamp(min_pct, 100.0);
            continue;
        }
        if seg.kind == PlannedMoveKind::WaveOverhang {
            raw_pct[i] = gcfg.wave_overhang_fan_speed_percent.clamp(min_pct, 100.0);
            continue;
        }

        // 1. Instantaneous volumetric extrusion rate factor
        let u_vol = (seg.volumetric_flow_mm3s / q_ref).clamp(0.0, 1.0);

        // 2. Cooling airflow residence time `t_res` and deposited melt volume `v_duct_mm3`
        //    within the 12.0 mm cooling duct footprint around `mid_i`
        let mid_i = 0.5 * (seg.p0_bed + seg.p1_bed);
        let mut t_res_s = seg.duration_s;
        let mut v_duct_mm3 = seg.delta_e_mm * filament_area_mm2;
        let mut dist_linear = 0.0f32;

        let mut j = i;
        while j > 0 && dist_linear < 36.0 {
            j -= 1;
            let sj = &layer.segments[j];
            dist_linear += sj.length_mm;
            let mid_j = 0.5 * (sj.p0_bed + sj.p1_bed);
            if (mid_j - mid_i).length_squared() <= r_duct_sq {
                t_res_s += sj.duration_s;
                if sj.kind.is_extruding() && sj.delta_e_mm > 0.0 {
                    v_duct_mm3 += sj.delta_e_mm * filament_area_mm2;
                }
            }
        }
        dist_linear = 0.0;
        let mut k = i + 1;
        while k < n && dist_linear < 36.0 {
            let sk = &layer.segments[k];
            dist_linear += sk.length_mm;
            let mid_k = 0.5 * (sk.p0_bed + sk.p1_bed);
            if (mid_k - mid_i).length_squared() <= r_duct_sq {
                t_res_s += sk.duration_s;
                if sk.kind.is_extruding() && sk.delta_e_mm > 0.0 {
                    v_duct_mm3 += sk.delta_e_mm * filament_area_mm2;
                }
            }
            k += 1;
        }

        // Thermal load inside the cooling duct footprint:
        // - High extruded volume per unit residence time (`q_duct = v_duct_mm3 / t_res_s`) needs high airflow.
        // - Short airflow residence time (`t_res_s < 1.2 s`, fast sweeping moves where the nozzle leaves
        //   the region before the air jet can quench the bead) needs higher fan velocity.
        // - Long residence time with low volume (`t_res_s > 1.8 s` on a tiny slow feature) still needs
        //   steady heat removal to prevent heat soak under the hot nozzle block.
        let q_duct = v_duct_mm3 / t_res_s.max(0.05);
        let u_duct_q = (q_duct / q_ref).clamp(0.0, 1.0);
        let u_fast_sweep = (0.45 / t_res_s.max(0.15)).clamp(0.0, 1.0);
        let u_nozzle_rad_soak = ((t_res_s - 0.8) / 2.2).clamp(0.0, 1.0)
            * (v_duct_mm3 / 2.5).clamp(0.25, 1.0);
        let u_residence = u_duct_q.max(0.65 * u_fast_sweep).max(0.80 * u_nozzle_rad_soak);

        // 3. Local corner / overhang / non-planar climb curl demand
        let seg_dir = (seg.p1_bed - seg.p0_bed).try_normalize().unwrap_or(Vec3::X);
        let z_noz = (seg.nozzle_bed_0 + seg.nozzle_bed_1)
            .try_normalize()
            .unwrap_or(up_bed);
        let climb_sin = seg_dir.dot(up_bed).abs();
        let tilt_sin = (1.0 - z_noz.dot(up_bed).abs().clamp(0.0, 1.0)).sqrt();
        let u_curl = (1.4 * climb_sin + 0.9 * tilt_sin).clamp(0.0, 1.0);

        let demand = (0.42 * u_vol + 0.38 * u_residence + 0.20 * u_curl)
            .max(0.85 * u_layer_time)
            .max(0.75 * u_curl)
            .clamp(0.0, 1.0);

        raw_pct[i] = (min_pct + (max_pct - min_pct) * demand).clamp(min_pct, max_pct);
    }

    // 4. Apply a 0.35 s forward lookahead pre-spin window (accounting for blower rotor inertia)
    //    followed by a 0.50 s temporal smoothing window and 3% PWM hysteresis deadband.
    let mut lead_pct = raw_pct.clone();
    for i in 0..n {
        let mut max_upcoming = raw_pct[i];
        let mut t_ahead = 0.0f32;
        let mut k = i + 1;
        while k < n && t_ahead < 0.35 {
            t_ahead += layer.segments[k].duration_s;
            max_upcoming = max_upcoming.max(raw_pct[k]);
            k += 1;
        }
        lead_pct[i] = 0.35 * raw_pct[i] + 0.65 * max_upcoming;
    }

    let mut pwm_schedule = vec![base_pwm; n];
    let mut active_pwm = (lead_pct[0] * 2.55).round().clamp(0.0, 255.0) as u8;
    for i in 0..n {
        let mut w_sum = 0.0f32;
        let mut p_sum = 0.0f32;
        let mut t_b = 0.0f32;
        let mut j = i;
        loop {
            let dt = layer.segments[j].duration_s.max(0.005);
            w_sum += dt;
            p_sum += dt * lead_pct[j];
            if j == 0 || t_b >= 0.25 {
                break;
            }
            j -= 1;
            t_b += dt;
        }
        let mut t_f = 0.0f32;
        let mut k = i + 1;
        while k < n && t_f < 0.25 {
            let dt = layer.segments[k].duration_s.max(0.005);
            w_sum += dt;
            p_sum += dt * lead_pct[k];
            t_f += dt;
            k += 1;
        }
        let smoothed_pct = if w_sum > 0.0 {
            p_sum / w_sum
        } else {
            lead_pct[i]
        };
        let target_pwm = (smoothed_pct * 2.55).round().clamp(0.0, 255.0) as u8;
        // Preserve exact PWM for explicit Bridge/WaveOverhang features; otherwise use 7-unit (~2.7%) deadband
        let is_special = matches!(
            layer.segments[i].kind,
            PlannedMoveKind::Bridge | PlannedMoveKind::WaveOverhang
        );
        if is_special || target_pwm.abs_diff(active_pwm) >= 7 {
            active_pwm = target_pwm;
        }
        pwm_schedule[i] = active_pwm;
    }

    pwm_schedule
}

/// Executes the complete Toolpath-to-Gcode planning & emission pipeline on `raw_toolpaths`.
pub fn plan_and_emit_gcode(
    raw_toolpaths: &ObjectToolpathsResult,
    obj_world_tf: Transform,
    config: &SlicerConfig,
) -> PlannedGcodeResult {
    plan_and_emit_gcode_for_object("Object_1", 1, raw_toolpaths, obj_world_tf, config)
}

/// Executes the complete Toolpath-to-Gcode planning & emission pipeline for a named object.
pub fn plan_and_emit_gcode_for_object(
    object_name: &str,
    object_id: u64,
    raw_toolpaths: &ObjectToolpathsResult,
    obj_world_tf: Transform,
    config: &SlicerConfig,
) -> PlannedGcodeResult {
    let mut gcfg = config.gcode.clone();
    gcfg.sanitize();
    let mut mcfg = config.machine.clone();
    mcfg.sanitize();

    let active_tool = config.active_tool();
    let tool_id = active_tool.map(|t| t.id).unwrap_or(0);
    let nozzle_diam_mm = active_tool
        .map(|t| t.nozzle_diameter_mm)
        .unwrap_or(0.40)
        .max(0.10);
    let nozzle_temp_c = active_tool
        .map(|t| t.temperature_celsius)
        .unwrap_or(215.0);
    let is_five_axis_bc = mcfg.kinematics_mode == FiveAxisKinematicsMode::FiveAxisBC;
    let max_bs_bound = mcfg.b_min_deg.abs().max(mcfg.b_max_deg.abs());
    let max_tool_tilt_deg = active_tool
        .map(|t| {
            if t.has_rotational_axes {
                t.max_angle_from_bed_normal_deg
            } else if is_five_axis_bc {
                max_bs_bound
            } else {
                0.0
            }
        })
        .unwrap_or(if is_five_axis_bc { max_bs_bound } else { 45.0 });

    let support_tool = config.support_tool();
    let support_tool_id = support_tool.map(|t| t.id).unwrap_or(tool_id);
    let support_nozzle_diam_mm = support_tool
        .map(|t| t.nozzle_diameter_mm)
        .unwrap_or(nozzle_diam_mm)
        .max(0.10);
    let support_nozzle_temp_c = support_tool
        .map(|t| t.temperature_celsius)
        .unwrap_or(nozzle_temp_c);
    let support_max_tool_tilt_deg = support_tool
        .map(|t| {
            if t.has_rotational_axes {
                t.max_angle_from_bed_normal_deg
            } else if is_five_axis_bc {
                max_bs_bound
            } else {
                0.0
            }
        })
        .unwrap_or(max_tool_tilt_deg);

    // Query gravity direction in bed coordinates (AGENTS.md Rule 2: never hardcode -Z!)
    let gravity_bed = config.query_gravity_bed(Vec3::ZERO);
    let up_bed = (-gravity_bed).try_normalize().unwrap_or(Vec3::Z);
    let bed_basis = BedUpBasis::from_up_bed(up_bed);
    let mut height_map = PrintedHeightMap::new(bed_basis);
    let mut footprint_profile: Option<BedFootprintProfile> = None;

    // Viewport scene places (0, 0) at the center of the build plate (`[-width_x/2 .. +width_x/2]`),
    // whereas rectangular printers expect G-code (0, 0) at the front-left corner (`[0 .. width_x]`).
    let bed_origin_offset = config.scene_to_gcode_offset();

    let inv_world_mat = obj_world_tf.compute_matrix().inverse();
    let to_local_pos =
        |p_bed: Vec3| -> Vec3 { inv_world_mat.transform_point3(p_bed - bed_origin_offset) };
    let to_local_dir = |d_bed: Vec3| -> Vec3 {
        (obj_world_tf.rotation.inverse() * d_bed)
            .try_normalize()
            .unwrap_or(Vec3::Z)
    };

    let filament_area_mm2 =
        std::f32::consts::FRAC_PI_4 * gcfg.filament_diameter_mm * gcfg.filament_diameter_mm;
    let fluid = FluidDynamicsModel::from_config(&gcfg, nozzle_temp_c);
    let support_fluid = FluidDynamicsModel::from_config(&gcfg, support_nozzle_temp_c);
    let min_retract_travel_mm = gcfg.effective_min_travel_for_retract();

    let mut planned_layers: Vec<PlannedLayerGcode> = Vec::with_capacity(raw_toolpaths.layers.len());
    let mut curr_nozzle_bed: Option<Vec3> = None;
    let mut curr_nozzle_axis_bed = Vec3::from_array(config.machine.nozzle_axis_bed)
        .try_normalize()
        .unwrap_or(up_bed);
    let mut curr_b_deg = 0.0f32;
    let mut curr_c_deg = mcfg.c_zero_offset_deg;
    let mut curr_planned_tool_id = tool_id;
    let mut curr_planned_is_support = false;

    let mut bbox_min = Vec3::splat(f32::MAX);
    let mut bbox_max = Vec3::splat(f32::MIN);

    for (layer_ord, raw_layer) in raw_toolpaths.layers.iter().enumerate() {
        let is_first_layer = layer_ord == 0;

        // 1. Convert raw FFI paths into bed-space `WorkingChain`s, simplify walls, and compensate nozzle geometry
        let mut chains: Vec<WorkingChain> = Vec::with_capacity(raw_layer.paths.len());
        for raw_path in &raw_layer.paths {
            if raw_path.points.len() < 2 {
                continue;
            }
            let base_kind = PlannedMoveKind::from_feature_kind(raw_path.feature_kind);
            let pts: Vec<WorkingVertex> = raw_path
                .points
                .iter()
                .map(|pt| {
                    let p_local = Vec3::new(pt.x, pt.y, pt.z);
                    let n_local = Vec3::new(pt.nx, pt.ny, pt.nz)
                        .try_normalize()
                        .unwrap_or(Vec3::Z);
                    let p_bed = obj_world_tf.transform_point(p_local) + bed_origin_offset;
                    let n_bed = (obj_world_tf.rotation * n_local)
                        .try_normalize()
                        .unwrap_or(up_bed);
                    WorkingVertex {
                        pos_bed: p_bed,
                        nozzle_bed: n_bed,
                        width_mm: pt.width_mm.max(0.08),
                        height_mm: pt.height_mm.max(0.02),
                        flow_rate_mult: 1.0,
                        kind_override: None,
                    }
                })
                .collect();

            let mut wc = WorkingChain {
                pts,
                base_kind,
                wall_index: raw_path.wall_index,
                is_closed: raw_path.is_closed,
            };
            let chain_nozzle_diam = if base_kind.is_support() {
                support_nozzle_diam_mm
            } else {
                nozzle_diam_mm
            };
            let chain_max_tilt_deg = if base_kind.is_support() {
                support_max_tool_tilt_deg
            } else {
                max_tool_tilt_deg
            };
            let chain_fluid = if base_kind.is_support() {
                &support_fluid
            } else {
                &fluid
            };
            simplify_working_chain(&mut wc, 0.04 * chain_nozzle_diam);
            smooth_chain_nozzle_orientations(
                &mut wc,
                mcfg.orientation_smoothing_window_mm,
                chain_max_tilt_deg,
                up_bed,
            );
            let nom_spd = if is_first_layer {
                gcfg.first_layer_speed_mms
            } else {
                gcfg.outer_wall_speed_mms
            };
            compensate_wall_geometry(
                &mut wc,
                chain_nozzle_diam,
                config.slicing.outer_wall_width.target_mm,
                up_bed,
                chain_fluid,
                nom_spd,
                gcfg.enable_flat_nozzle_comp,
            );
            chains.push(wc);
        }

            let start_ref = curr_nozzle_bed.unwrap_or(bed_origin_offset);
        let mut ordered_chains = order_layer_chains(
            chains,
            gcfg.wall_print_order,
            start_ref,
            gcfg.z_travel_penalty,
            up_bed,
        );

        // On Layer 0, compute the bed contact footprint and prepend outward concentric brim loops
        // if explicit brim or auto-brim is active.
        if is_first_layer {
            let fp = BedFootprintProfile::from_raw_toolpaths(
                raw_toolpaths,
                obj_world_tf,
                bed_origin_offset,
                bed_basis,
                &ordered_chains,
            );
            let mut brim_chains = generate_first_layer_brim_chains(
                &ordered_chains,
                &fp,
                raw_toolpaths.layers.len(),
                &gcfg,
                config.slicing.outer_wall_width.target_mm,
                config.slicing.layer_height.first_layer_mm,
            );
            if !brim_chains.is_empty() {
                brim_chains.append(&mut ordered_chains);
                ordered_chains = brim_chains;
            }
            footprint_profile = Some(fp);
        }

        let mut layer_segments: Vec<PlannedSegment> = Vec::new();

        for chain in &mut ordered_chains {
            let chain_is_support = chain.base_kind.is_support();
            let chain_tool_cfg = if chain_is_support {
                support_tool
            } else {
                active_tool
            };
            let chain_tool_id = if chain_is_support {
                support_tool_id
            } else {
                tool_id
            };
            let chain_nozzle_diam = if chain_is_support {
                support_nozzle_diam_mm
            } else {
                nozzle_diam_mm
            };
            let chain_fluid = if chain_is_support {
                &support_fluid
            } else {
                &fluid
            };

            let mut vertices = if chain.base_kind == PlannedMoveKind::Bridge {
                // Keep circular Bridge extrusions as pure un-subdivided straight lines
                chain.pts.clone()
            } else {
                subdivide_long_segments(finalize_chain_seam_and_wipe(chain, &gcfg))
            };
            if vertices.len() < 2 {
                continue;
            }
            // Enforce monotonic layer elevation and local-frame toolhead clearance cone against
            // all already-printed material AFTER `finalize_chain_seam_and_wipe` so scarf-seam
            // lead-ins and backward wipes never dip below already-printed cells!
            height_map.clamp_vertices_monotonic_and_clearance(
                &mut vertices,
                chain.is_closed,
                layer_ord,
                chain_tool_cfg,
            );

            let chain_start_bed = vertices[0].pos_bed;
            let chain_start_nozzle = vertices[0].nozzle_bed;

            // Emit inter-path travel + obstacle-aware Z-hop along local nozzle axis (`z_nozzle`) / `up_bed`.
            // Always Z-hop when crossing between support and object extrusions, when traveling
            // between separate support structures, when switching tools, OR whenever the straight-line
            // travel corridor passes within 0.20 mm of already-printed material!
            if let Some(prev_bed) = curr_nozzle_bed {
                let travel_vec = chain_start_bed - prev_bed;
                let travel_dist = travel_vec.length();
                let tool_changed = chain_tool_id != curr_planned_tool_id;
                let support_boundary_crossing = chain_is_support != curr_planned_is_support;
                if travel_dist > 1.0e-3 || tool_changed {
                    let (_, _, h_prev) = bed_basis.project_uvh(prev_bed);
                    let (_, _, h_next) = bed_basis.project_uvh(chain_start_bed);
                    let corridor_obs_h = height_map.max_corridor_obstacle_h(prev_bed, chain_start_bed);
                    let corridor_blocked = corridor_obs_h
                        .is_some_and(|obs_h| obs_h > h_prev.min(h_next) - 0.15);

                    let needs_retract = travel_dist > min_retract_travel_mm
                        || tool_changed
                        || support_boundary_crossing
                        || corridor_blocked;
                    let is_infill_to_infill = matches!(
                        chain.base_kind,
                        PlannedMoveKind::SolidInfill | PlannedMoveKind::SparseInfill
                    ) && layer_segments.last().is_some_and(|s| {
                        matches!(
                            s.kind,
                            PlannedMoveKind::SolidInfill | PlannedMoveKind::SparseInfill
                        )
                    });
                    let use_z_hop = corridor_blocked
                        || (gcfg.z_hop_mm > 1.0e-3
                            && ((needs_retract && !is_infill_to_infill)
                                || tool_changed
                                || support_boundary_crossing));

                    if use_z_hop {
                        let hop_margin = gcfg.z_hop_mm.max(0.30);
                        // When clearing an obstacle or performing a Z-hop across non-planar terrain,
                        // lift and drop strictly along `up_bed` when `corridor_blocked` so the Z-hop
                        // lift/drop legs stay within their start/end `(u, v)` columns and never tilt
                        // laterally into adjacent steeper walls!
                        let lift_dir = if corridor_blocked {
                            up_bed
                        } else {
                            curr_nozzle_axis_bed.try_normalize().unwrap_or(up_bed)
                        };
                        let drop_dir = if corridor_blocked {
                            up_bed
                        } else {
                            chain_start_nozzle.try_normalize().unwrap_or(up_bed)
                        };

                        let mut h_peak = h_prev
                            .max(h_next)
                            .max(corridor_obs_h.unwrap_or(h_prev.max(h_next)));
                        let lift_cos = lift_dir.dot(up_bed).max(0.35);
                        let drop_cos = drop_dir.dot(up_bed).max(0.35);

                        let mut lift_dist = ((h_peak + hop_margin - h_prev).max(hop_margin) / lift_cos).max(hop_margin);
                        let mut drop_dist = ((h_peak + hop_margin - h_next).max(hop_margin) / drop_cos).max(hop_margin);
                        let mut p_lift = prev_bed + lift_dir * lift_dist;
                        let mut p_above = chain_start_bed + drop_dir * drop_dist;

                        if let Some(elev_obs_h) = height_map.max_corridor_obstacle_h(p_lift, p_above) {
                            if elev_obs_h > h_peak {
                                h_peak = elev_obs_h;
                                let h_clear = h_peak + hop_margin;
                                lift_dist = ((h_clear - h_prev).max(hop_margin) / lift_cos).max(hop_margin);
                                drop_dist = ((h_clear - h_next).max(hop_margin) / drop_cos).max(hop_margin);
                                p_lift = prev_bed + lift_dir * lift_dist;
                                p_above = chain_start_bed + drop_dir * drop_dist;
                            }
                        }

                        // 1. Z-Hop Lift (carries `retract_before_mm` marker)
                        layer_segments.push(PlannedSegment {
                            p0_local: to_local_pos(prev_bed),
                            p1_local: to_local_pos(p_lift),
                            p0_bed: prev_bed,
                            p1_bed: p_lift,
                            nozzle_local_0: to_local_dir(curr_nozzle_axis_bed),
                            nozzle_local_1: to_local_dir(curr_nozzle_axis_bed),
                            nozzle_bed_0: curr_nozzle_axis_bed,
                            nozzle_bed_1: curr_nozzle_axis_bed,
                            b0_deg: 0.0,
                            c0_deg: 0.0,
                            b1_deg: 0.0,
                            c1_deg: 0.0,
                            kind: PlannedMoveKind::ZHop,
                            tool_id: curr_planned_tool_id,
                            wall_index: 0,
                            width_mm: chain_nozzle_diam * 0.35,
                            height_mm: lift_dist,
                            length_mm: lift_dist,
                            nominal_speed_mms: gcfg.z_hop_speed_mms,
                            entry_speed_mms: 0.0,
                            cruise_speed_mms: gcfg.z_hop_speed_mms,
                            exit_speed_mms: 0.0,
                            accel_mms2: gcfg.travel_accel_mms2,
                            duration_s: (lift_dist / gcfg.z_hop_speed_mms.max(1.0)).max(1.0e-4),
                            delta_e_mm: 0.0,
                            volumetric_flow_mm3s: 0.0,
                            extrusion_multiplier: 0.0,
                            pressure_advance_s: 0.0,
                            melt_pressure_mm3s: 0.0,
                            retract_before_mm: gcfg.retract_length_mm,
                            unretract_after_mm: 0.0,
                        });

                        // 2. Elevated Travel Move
                        let mid_len = (p_above - p_lift).length().max(1.0e-4);
                        layer_segments.push(PlannedSegment {
                            p0_local: to_local_pos(p_lift),
                            p1_local: to_local_pos(p_above),
                            p0_bed: p_lift,
                            p1_bed: p_above,
                            nozzle_local_0: to_local_dir(curr_nozzle_axis_bed),
                            nozzle_local_1: to_local_dir(chain_start_nozzle),
                            nozzle_bed_0: curr_nozzle_axis_bed,
                            nozzle_bed_1: chain_start_nozzle,
                            b0_deg: 0.0,
                            c0_deg: 0.0,
                            b1_deg: 0.0,
                            c1_deg: 0.0,
                            kind: PlannedMoveKind::Travel,
                            tool_id: chain_tool_id,
                            wall_index: 0,
                            width_mm: chain_nozzle_diam * 0.35,
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

                        // 3. Z-Hop Drop (carries `unretract_after_mm` marker)
                        layer_segments.push(PlannedSegment {
                            p0_local: to_local_pos(p_above),
                            p1_local: to_local_pos(chain_start_bed),
                            p0_bed: p_above,
                            p1_bed: chain_start_bed,
                            nozzle_local_0: to_local_dir(chain_start_nozzle),
                            nozzle_local_1: to_local_dir(chain_start_nozzle),
                            nozzle_bed_0: chain_start_nozzle,
                            nozzle_bed_1: chain_start_nozzle,
                            b0_deg: 0.0,
                            c0_deg: 0.0,
                            b1_deg: 0.0,
                            c1_deg: 0.0,
                            kind: PlannedMoveKind::ZHop,
                            tool_id: chain_tool_id,
                            wall_index: 0,
                            width_mm: chain_nozzle_diam * 0.35,
                            height_mm: drop_dist,
                            length_mm: drop_dist,
                            nominal_speed_mms: gcfg.z_hop_speed_mms,
                            entry_speed_mms: 0.0,
                            cruise_speed_mms: gcfg.z_hop_speed_mms,
                            exit_speed_mms: 0.0,
                            accel_mms2: gcfg.travel_accel_mms2,
                            duration_s: (drop_dist / gcfg.z_hop_speed_mms.max(1.0)).max(1.0e-4),
                            delta_e_mm: 0.0,
                            volumetric_flow_mm3s: 0.0,
                            extrusion_multiplier: 0.0,
                            pressure_advance_s: 0.0,
                            melt_pressure_mm3s: 0.0,
                            retract_before_mm: 0.0,
                            unretract_after_mm: gcfg.retract_length_mm,
                        });
                    } else {
                        let r_len = if needs_retract {
                            gcfg.retract_length_mm
                        } else {
                            0.0
                        };
                        layer_segments.push(PlannedSegment {
                            p0_local: to_local_pos(prev_bed),
                            p1_local: to_local_pos(chain_start_bed),
                            p0_bed: prev_bed,
                            p1_bed: chain_start_bed,
                            nozzle_local_0: to_local_dir(curr_nozzle_axis_bed),
                            nozzle_local_1: to_local_dir(chain_start_nozzle),
                            nozzle_bed_0: curr_nozzle_axis_bed,
                            nozzle_bed_1: chain_start_nozzle,
                            b0_deg: 0.0,
                            c0_deg: 0.0,
                            b1_deg: 0.0,
                            c1_deg: 0.0,
                            kind: PlannedMoveKind::Travel,
                            tool_id: chain_tool_id,
                            wall_index: 0,
                            width_mm: chain_nozzle_diam * 0.35,
                            height_mm: 0.15,
                            length_mm: travel_dist.max(1.0e-4),
                            nominal_speed_mms: gcfg.travel_speed_mms,
                            entry_speed_mms: 0.0,
                            cruise_speed_mms: gcfg.travel_speed_mms,
                            exit_speed_mms: 0.0,
                            accel_mms2: gcfg.travel_accel_mms2,
                            duration_s: (travel_dist / gcfg.travel_speed_mms.max(1.0)).max(1.0e-4),
                            delta_e_mm: 0.0,
                            volumetric_flow_mm3s: 0.0,
                            extrusion_multiplier: 0.0,
                            pressure_advance_s: 0.0,
                            melt_pressure_mm3s: 0.0,
                            retract_before_mm: r_len,
                            unretract_after_mm: r_len,
                        });
                    }
                }
            }

            // 2. Emit segments of this chain with base extrusion & volumetric speed capping
            let chain_seg_start = layer_segments.len();
            let chain_arc_len: f32 = vertices
                .windows(2)
                .map(|w| (w[1].pos_bed - w[0].pos_bed).length())
                .sum();
            // Smoothly slow down small closed wall perimeters (< 18 mm, e.g. chimney, pillars, small holes)
            // down to 50% speed at <= 6 mm perimeter, matching OrcaSlicer small_perimeter_speed behavior.
            let small_perim_speed_mult =
                if chain.is_closed && chain.base_kind.is_wall() && chain_arc_len < 18.0 {
                    0.50 + 0.50 * ((chain_arc_len - 6.0) / 12.0).clamp(0.0, 1.0)
                } else {
                    1.0
                };

            for w in vertices.windows(2) {
                let a = &w[0];
                let b = &w[1];
                let seg_vec = b.pos_bed - a.pos_bed;
                let seg_len = seg_vec.length();
                if seg_len <= 1.0e-5 {
                    continue;
                }
                let seg_dir = seg_vec / seg_len;
                let z_nozzle = (a.nozzle_bed + b.nozzle_bed)
                    .try_normalize()
                    .unwrap_or(up_bed);

                let kind = b.kind_override.unwrap_or(chain.base_kind);
                let width = 0.5 * (a.width_mm + b.width_mm);
                let height = 0.5 * (a.height_mm + b.height_mm);
                let rate_mult = b.flow_rate_mult;

                let (base_spd, base_accel) = match kind {
                    PlannedMoveKind::OuterWall | PlannedMoveKind::ScarfSeam => (
                        gcfg.outer_wall_speed_mms * small_perim_speed_mult,
                        gcfg.outer_wall_accel_mms2,
                    ),
                    PlannedMoveKind::InnerWall => (
                        gcfg.inner_wall_speed_mms * small_perim_speed_mult,
                        gcfg.inner_wall_accel_mms2,
                    ),
                    PlannedMoveKind::SolidInfill => {
                        (gcfg.solid_infill_speed_mms, gcfg.infill_accel_mms2)
                    }
                    PlannedMoveKind::GapFill => (gcfg.gap_fill_speed_mms, gcfg.inner_wall_accel_mms2),
                    PlannedMoveKind::SparseInfill => {
                        (gcfg.sparse_infill_speed_mms, gcfg.infill_accel_mms2)
                    }
                    PlannedMoveKind::Bridge => {
                        (gcfg.bridge_speed_mms, gcfg.outer_wall_accel_mms2)
                    }
                    PlannedMoveKind::WaveOverhang => {
                        (gcfg.wave_overhang_speed_mms, gcfg.outer_wall_accel_mms2)
                    }
                    PlannedMoveKind::Support => {
                        (gcfg.support_speed_mms, gcfg.infill_accel_mms2)
                    }
                    PlannedMoveKind::SupportInterface => {
                        (gcfg.support_interface_speed_mms, gcfg.inner_wall_accel_mms2)
                    }
                    PlannedMoveKind::Wipe | PlannedMoveKind::Travel => {
                        (gcfg.travel_speed_mms * 0.6, gcfg.travel_accel_mms2)
                    }
                    PlannedMoveKind::ZHop => (gcfg.z_hop_speed_mms, gcfg.travel_accel_mms2),
                };

                let target_spd = if is_first_layer && kind.is_extruding() {
                    base_spd.min(gcfg.first_layer_speed_mms)
                } else {
                    base_spd
                };

                if kind.is_extruding() && rate_mult > 1.0e-4 {
                    bbox_min = bbox_min.min(a.pos_bed).min(b.pos_bed);
                    bbox_max = bbox_max.max(a.pos_bed).max(b.pos_bed);

                    // Bead cross-section area A_bead:
                    // - Bridges: free-air circular cross-section A_bridge = (\pi / 4) * width^2
                    // - Wave Overhangs: stadium cross-section reduced by the lateral overlap
                    //   \delta_overlap = \eta_wave * w_standard with the adjacent extrusion
                    // - Standard extrusions: stadium (upper layers) or rectangular (first layer)
                    let a_bead = if kind == PlannedMoveKind::Bridge {
                        std::f32::consts::FRAC_PI_4 * width * width
                    } else {
                        let w_eff = width.max(height);
                        let a_stadium = height * (w_eff - height)
                            + std::f32::consts::FRAC_PI_4 * height * height;
                        let a_rect = w_eff * height;
                        let base_area = if is_first_layer { a_rect } else { a_stadium };
                        if kind == PlannedMoveKind::WaveOverhang {
                            let std_w = config.slicing.outer_wall_width.target_mm.max(0.10);
                            let overlap_frac = config
                                .slicing
                                .wave_overhang_overlap_fraction
                                .clamp(0.01, 0.35);
                            let overlap_mm = overlap_frac * std_w;
                            let net_width_ratio =
                                ((width - overlap_mm) / width.max(0.05)).clamp(0.50, 1.0);
                            base_area * net_width_ratio
                        } else {
                            base_area
                        }
                    };

                    // Surface inclination & trajectory climb cosine along local `z_nozzle` and `up_bed`
                    let c_slope = if kind == PlannedMoveKind::Bridge {
                        // Circular free-air bridge extrusions maintain full circular cross-section
                        // at any accessible 3D angle without substrate cosine compression.
                        1.0
                    } else {
                        let c_surf = if is_first_layer {
                            1.0
                        } else {
                            z_nozzle.dot(up_bed).abs().clamp(0.25, 1.0)
                        };
                        let climb_nozzle = seg_dir.dot(z_nozzle);
                        let c_traj = (1.0 - climb_nozzle * climb_nozzle).max(0.15).sqrt();
                        c_surf.min(c_traj).clamp(0.25, 1.0)
                    };

                    // Keep uphill and downhill sides of non-planar loops symmetric (`m_dir = 1.0`)
                    // since `c_slope` already accounts for the nozzle-frame geometric projection.
                    let m_dir = 1.0f32;

                    // Dedicated feature flow multiplier for Bridges and Wave Overhangs (in addition
                    // to global extrusion_multiplier, die swell, corner, and PA/transient multipliers)
                    let m_feature = match kind {
                        PlannedMoveKind::Bridge => gcfg.bridge_flow_multiplier,
                        PlannedMoveKind::WaveOverhang => gcfg.wave_overhang_flow_multiplier,
                        _ => 1.0,
                    };

                    // Volumetric speed limit v_clamped <= Q_max / A_bead
                    let v_vol_max = gcfg.max_volumetric_flow_mm3s / a_bead.max(1.0e-4);
                    let v_clamped = target_spd.min(v_vol_max).max(5.0);
                    let q_est = a_bead * v_clamped * rate_mult * m_feature;
                    let m_swell = chain_fluid.swell_volume_multiplier(q_est);
                    let m_first = if is_first_layer {
                        gcfg.first_layer_flow_multiplier
                    } else {
                        1.0
                    };

                    let composite_mult = rate_mult
                        * gcfg.extrusion_multiplier
                        * m_feature
                        * m_first
                        * c_slope
                        * m_dir
                        * m_swell;
                    let delta_e = ((seg_len * a_bead) / filament_area_mm2) * composite_mult;

                    layer_segments.push(PlannedSegment {
                        p0_local: to_local_pos(a.pos_bed),
                        p1_local: to_local_pos(b.pos_bed),
                        p0_bed: a.pos_bed,
                        p1_bed: b.pos_bed,
                        nozzle_local_0: to_local_dir(a.nozzle_bed),
                        nozzle_local_1: to_local_dir(b.nozzle_bed),
                        nozzle_bed_0: a.nozzle_bed,
                        nozzle_bed_1: b.nozzle_bed,
                        b0_deg: 0.0,
                        c0_deg: 0.0,
                        b1_deg: 0.0,
                        c1_deg: 0.0,
                        kind,
                        tool_id: chain_tool_id,
                        wall_index: chain.wall_index,
                        width_mm: width,
                        height_mm: height,
                        length_mm: seg_len,
                        nominal_speed_mms: v_clamped,
                        entry_speed_mms: v_clamped,
                        cruise_speed_mms: v_clamped,
                        exit_speed_mms: v_clamped,
                        accel_mms2: base_accel,
                        duration_s: (seg_len / v_clamped).max(1.0e-4),
                        delta_e_mm: delta_e,
                        volumetric_flow_mm3s: q_est,
                        extrusion_multiplier: composite_mult,
                        pressure_advance_s: chain_fluid.pressure_advance_s(
                            q_est,
                            gcfg.enable_dynamic_pa,
                            gcfg.pa_low_flow_s,
                        ),
                        melt_pressure_mm3s: q_est,
                        retract_before_mm: 0.0,
                        unretract_after_mm: 0.0,
                    });
                } else {
                    layer_segments.push(PlannedSegment {
                        p0_local: to_local_pos(a.pos_bed),
                        p1_local: to_local_pos(b.pos_bed),
                        p0_bed: a.pos_bed,
                        p1_bed: b.pos_bed,
                        nozzle_local_0: to_local_dir(a.nozzle_bed),
                        nozzle_local_1: to_local_dir(b.nozzle_bed),
                        nozzle_bed_0: a.nozzle_bed,
                        nozzle_bed_1: b.nozzle_bed,
                        b0_deg: 0.0,
                        c0_deg: 0.0,
                        b1_deg: 0.0,
                        c1_deg: 0.0,
                        kind: PlannedMoveKind::Wipe,
                        tool_id: chain_tool_id,
                        wall_index: chain.wall_index,
                        width_mm: width * 0.5,
                        height_mm: height * 0.5,
                        length_mm: seg_len,
                        nominal_speed_mms: target_spd,
                        entry_speed_mms: target_spd,
                        cruise_speed_mms: target_spd,
                        exit_speed_mms: target_spd,
                        accel_mms2: base_accel,
                        duration_s: (seg_len / target_spd.max(1.0)).max(1.0e-4),
                        delta_e_mm: 0.0,
                        volumetric_flow_mm3s: 0.0,
                        extrusion_multiplier: 0.0,
                        pressure_advance_s: 0.0,
                        melt_pressure_mm3s: 0.0,
                        retract_before_mm: 0.0,
                        unretract_after_mm: 0.0,
                    });
                }
            }

            // 3. Apply Corner Overlap & Klipper SCV Shortcut Arc Flow Compensation
            //    Strictly on interior junctions of this chain (`chain_seg_start .. layer_segments.len()`)
            //    at genuine macroscopic corners (\alpha >= 25 deg), NEVER on smooth tessellated curves!
            if gcfg.enable_corner_flow_comp && layer_segments.len() >= chain_seg_start + 2 {
                let end_idx = layer_segments.len();
                let mut deductions_mm3 = vec![0.0f32; end_idx - chain_seg_start];

                for k in (chain_seg_start + 1)..end_idx {
                    let s_in = &layer_segments[k - 1];
                    let s_out = &layer_segments[k];
                    if !s_in.kind.is_extruding()
                        || !s_out.kind.is_extruding()
                        || s_in.delta_e_mm <= 0.0
                        || s_out.delta_e_mm <= 0.0
                    {
                        continue;
                    }

                    let w_avg = 0.5 * (s_in.width_mm + s_out.width_mm);
                    let h_avg = 0.5 * (s_in.height_mm + s_out.height_mm);
                    if s_in.length_mm < 0.5 * w_avg || s_out.length_mm < 0.5 * w_avg {
                        continue;
                    }

                    let z_surf = s_in
                        .nozzle_bed_1
                        .try_normalize()
                        .unwrap_or(up_bed);
                    let raw_in = (s_in.p1_bed - s_in.p0_bed).try_normalize().unwrap_or(Vec3::X);
                    let raw_out = (s_out.p1_bed - s_out.p0_bed).try_normalize().unwrap_or(Vec3::X);

                    // Project incoming/outgoing directions onto local layer tangent plane
                    let u_in = (raw_in - z_surf * raw_in.dot(z_surf))
                        .try_normalize()
                        .unwrap_or(raw_in);
                    let u_out = (raw_out - z_surf * raw_out.dot(z_surf))
                        .try_normalize()
                        .unwrap_or(raw_out);

                    let cos_alpha = u_in.dot(u_out).clamp(-1.0, 1.0);
                    // Ignore shallow curve tessellation facets (\alpha < ~25 deg, cos(25 deg) = 0.9063)
                    if cos_alpha >= 0.9063 {
                        continue;
                    }

                    let alpha = cos_alpha.acos();
                    let sin_half = (0.5 * (1.0 - cos_alpha)).max(0.0).sqrt();
                    let cos_half = (0.5 * (1.0 + cos_alpha)).max(0.125).sqrt();
                    let tan_half = (sin_half / cos_half).min(2.5);

                    // Geometric inner-corner diamond overlap volume
                    let dv_geom = 0.25 * w_avg * w_avg * h_avg * tan_half;

                    // Kinematic Klipper SCV arc shortcut volume
                    let (_, r_eff) = compute_klipper_corner_kinematics(
                        u_in,
                        u_out,
                        gcfg.square_corner_velocity_mms,
                        s_in.accel_mms2.min(s_out.accel_mms2),
                        s_in.nominal_speed_mms,
                        s_out.nominal_speed_mms,
                        s_in.length_mm,
                        s_out.length_mm,
                    );
                    let dl_kin = (r_eff * (2.0 * tan_half - alpha)).max(0.0);
                    let dv_kin = dl_kin * w_avg * h_avg;

                    let half_excess =
                        0.5 * gcfg.corner_flow_comp_ratio.clamp(0.0, 1.0) * (dv_geom + dv_kin);
                    deductions_mm3[k - 1 - chain_seg_start] += half_excess;
                    deductions_mm3[k - chain_seg_start] += half_excess;
                }

                for (rel_i, &ded_v) in deductions_mm3.iter().enumerate() {
                    if ded_v <= 1.0e-6 {
                        continue;
                    }
                    let seg = &mut layer_segments[chain_seg_start + rel_i];
                    let nom_vol = seg.delta_e_mm * filament_area_mm2;
                    let max_ded = 0.15 * nom_vol;
                    let actual_ded = ded_v.min(max_ded);
                    if nom_vol > 1.0e-5 {
                        let factor = ((nom_vol - actual_ded) / nom_vol).clamp(0.85, 1.0);
                        seg.delta_e_mm *= factor;
                        seg.extrusion_multiplier *= factor;
                    }
                }
            }

            // Record all extruding segments of this chain into `height_map` so subsequent chains
            // on this same layer (and future layers) avoid colliding with or plowing through them.
            for seg in &layer_segments[chain_seg_start..] {
                if seg.kind.is_extruding() && seg.delta_e_mm > 1.0e-6 {
                    height_map.record_extrusion_segment(seg.p0_bed, seg.p1_bed);
                }
            }

            if let Some(last_v) = vertices.last() {
                curr_nozzle_bed = Some(last_v.pos_bed);
                curr_nozzle_axis_bed = last_v.nozzle_bed;
                curr_planned_tool_id = chain_tool_id;
                curr_planned_is_support = chain_is_support;
            }
        }

        height_map.finish_layer();

        // 4a. Apply automatic Lever-Arm & Curled-Corner Speed/Acceleration Governor before velocity lookahead
        if let Some(ref fp) = footprint_profile {
            apply_lever_arm_and_curl_slowdown(&mut layer_segments, layer_ord, fp, &gcfg);
        }

        // 4b. Solve 5-Axis B/C postures, rotary velocity/accel throttling, and Chained Velocity Lookahead
        assign_five_axis_bc_postures_and_rotary_limits(
            &mut layer_segments,
            &mcfg,
            max_tool_tilt_deg,
            &mut curr_b_deg,
            &mut curr_c_deg,
        );
        plan_velocity_lookahead(&mut layer_segments, &gcfg);

        planned_layers.push(PlannedLayerGcode {
            layer_index: raw_layer.layer_index,
            branch_id: raw_layer.branch_id,
            isovalue: raw_layer.isovalue,
            segments: layer_segments,
        });
    }

    // 5. End-of-Print Reverse-Wipe & Outward Clearance Lift (Section 2.5)
    if let Some(last_layer) = planned_layers.last_mut() {
        if let Some(last_ext) = last_layer
            .segments
            .iter()
            .rev()
            .find(|s| s.kind.is_extruding() && s.delta_e_mm > 0.0)
            .cloned()
        {
            let back_dir = (last_ext.p0_bed - last_ext.p1_bed)
                .try_normalize()
                .unwrap_or(-Vec3::X);
            let max_wipe_len = last_ext.length_mm.min(4.0).max(0.01);
            let min_wipe_len = (0.5 * nozzle_diam_mm).min(max_wipe_len);
            let wipe_len = (last_ext.pressure_advance_s * last_ext.exit_speed_mms + nozzle_diam_mm)
                .clamp(min_wipe_len, max_wipe_len);
            let p_wipe = last_ext.p1_bed + back_dir * wipe_len;
            let p_clear = p_wipe + last_ext.nozzle_bed_1.try_normalize().unwrap_or(up_bed) * 2.0;

            last_layer.segments.push(PlannedSegment {
                p0_local: to_local_pos(last_ext.p1_bed),
                p1_local: to_local_pos(p_wipe),
                p0_bed: last_ext.p1_bed,
                p1_bed: p_wipe,
                nozzle_local_0: last_ext.nozzle_local_1,
                nozzle_local_1: last_ext.nozzle_local_1,
                nozzle_bed_0: last_ext.nozzle_bed_1,
                nozzle_bed_1: last_ext.nozzle_bed_1,
                b0_deg: last_ext.b1_deg,
                c0_deg: last_ext.c1_deg,
                b1_deg: last_ext.b1_deg,
                c1_deg: last_ext.c1_deg,
                kind: PlannedMoveKind::Wipe,
                tool_id: last_ext.tool_id,
                wall_index: 0,
                width_mm: nozzle_diam_mm * 0.4,
                height_mm: 0.15,
                length_mm: wipe_len,
                nominal_speed_mms: gcfg.travel_speed_mms * 0.5,
                entry_speed_mms: gcfg.square_corner_velocity_mms,
                cruise_speed_mms: gcfg.travel_speed_mms * 0.5,
                exit_speed_mms: 0.0,
                accel_mms2: gcfg.travel_accel_mms2,
                duration_s: (wipe_len / (gcfg.travel_speed_mms * 0.5).max(1.0)).max(1.0e-4),
                delta_e_mm: 0.0,
                volumetric_flow_mm3s: 0.0,
                extrusion_multiplier: 0.0,
                pressure_advance_s: 0.0,
                melt_pressure_mm3s: 0.0,
                retract_before_mm: 0.0,
                unretract_after_mm: 0.0,
            });

            last_layer.segments.push(PlannedSegment {
                p0_local: to_local_pos(p_wipe),
                p1_local: to_local_pos(p_clear),
                p0_bed: p_wipe,
                p1_bed: p_clear,
                nozzle_local_0: last_ext.nozzle_local_1,
                nozzle_local_1: last_ext.nozzle_local_1,
                nozzle_bed_0: last_ext.nozzle_bed_1,
                nozzle_bed_1: last_ext.nozzle_bed_1,
                b0_deg: last_ext.b1_deg,
                c0_deg: last_ext.c1_deg,
                b1_deg: last_ext.b1_deg,
                c1_deg: last_ext.c1_deg,
                kind: PlannedMoveKind::ZHop,
                tool_id: last_ext.tool_id,
                wall_index: 0,
                width_mm: nozzle_diam_mm * 0.35,
                height_mm: 2.0,
                length_mm: 2.0,
                nominal_speed_mms: gcfg.z_hop_speed_mms,
                entry_speed_mms: 0.0,
                cruise_speed_mms: gcfg.z_hop_speed_mms,
                exit_speed_mms: 0.0,
                accel_mms2: gcfg.travel_accel_mms2,
                duration_s: (2.0 / gcfg.z_hop_speed_mms.max(1.0)).max(1.0e-4),
                delta_e_mm: 0.0,
                volumetric_flow_mm3s: 0.0,
                extrusion_multiplier: 0.0,
                pressure_advance_s: 0.0,
                melt_pressure_mm3s: 0.0,
                retract_before_mm: gcfg.retract_length_mm,
                unretract_after_mm: 0.0,
            });
        }
    }

    // 6. Global Melt-Zone Transient Pressure Pass & Slicer-Side PA Subdivision
    let mut pressure_tracker = MeltPressureTracker::new();
    let mut last_q_mm3s = 0.0f32;
    let mut last_pa_s = gcfg.pa_low_flow_s;

    for layer in &mut planned_layers {
        for seg in &mut layer.segments {
            let seg_fluid = if seg.kind.is_support() {
                &support_fluid
            } else {
                &fluid
            };
            if seg.retract_before_mm > 0.0 {
                seg.retract_before_mm = seg_fluid.adaptive_retract_length_mm(
                    last_q_mm3s,
                    last_pa_s,
                    filament_area_mm2,
                    gcfg.retract_length_mm,
                );
            }
            if seg.kind.is_extruding() && seg.delta_e_mm > 0.0 {
                let q_nom = (seg.delta_e_mm * filament_area_mm2) / seg.duration_s.max(1.0e-4);
                let pa_s = seg_fluid.pressure_advance_s(
                    q_nom,
                    gcfg.enable_dynamic_pa,
                    gcfg.pa_low_flow_s,
                );
                let (m_trans, p_avg) = pressure_tracker.step_extrusion(
                    q_nom,
                    seg.duration_s,
                    pa_s,
                    seg.extrusion_multiplier,
                    &gcfg,
                );
                seg.delta_e_mm *= m_trans;
                seg.extrusion_multiplier *= m_trans;
                seg.volumetric_flow_mm3s =
                    (seg.delta_e_mm * filament_area_mm2) / seg.duration_s.max(1.0e-4);
                seg.pressure_advance_s = seg_fluid.pressure_advance_s(
                    seg.volumetric_flow_mm3s,
                    gcfg.enable_dynamic_pa,
                    gcfg.pa_low_flow_s,
                );
                seg.melt_pressure_mm3s = p_avg;
                last_q_mm3s = seg.volumetric_flow_mm3s;
                last_pa_s = seg.pressure_advance_s;
            } else {
                let p_avg = pressure_tracker.step_travel(
                    seg.duration_s,
                    gcfg.transient_pressure_tau_s.max(last_pa_s),
                );
                seg.melt_pressure_mm3s = p_avg;
                seg.pressure_advance_s = last_pa_s;
            }
        }

        if gcfg.enable_slicer_pa_subdivision {
            let taken = std::mem::take(&mut layer.segments);
            layer.segments =
                subdivide_segments_for_slicer_pa(taken, filament_area_mm2, &fluid, &gcfg);
        }
    }

    if !bbox_min.is_finite() || bbox_min.x > bbox_max.x {
        bbox_min = Vec3::ZERO;
        bbox_max = Vec3::ZERO;
    }

    // 7. Compute Range Statistics & Format Klipper G-code String
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

    let exclude_object_name = sanitize_exclude_object_name(object_name, object_id);
    let exclude_object_center_bed = [
        0.5 * (bbox_min.x + bbox_max.x),
        0.5 * (bbox_min.y + bbox_max.y),
    ];
    let exclude_object_polygon_bed =
        compute_exclude_object_polygon_2d(&planned_layers, bbox_min, bbox_max);

    let mut gcode = String::with_capacity(64 * 1024);
    // Moonraker's `metadata.py` (`PrusaSlicer.check_identity`) requires a header comment
    // matching `r"PrusaSlicer\s(.*)\son"` (or `OrcaSlicer\s(.*)\son`) to enable full metadata
    // and embedded PNG thumbnail extraction.
    gcode.push_str("; generated by PrusaSlicer 2.8.0-Manifold2 on 2026-10-20 at 00:00:00 UTC\n");
    if gcfg.emit_thumbnails && !planned_layers.is_empty() {
        gcode.push_str(&render_toolpath_preview_thumbnails(
            &planned_layers,
            bbox_min,
            bbox_max,
            up_bed,
        ));
    }
    if gcfg.emit_exclude_object {
        match gcfg.gcode_dialect {
            GcodeDialect::Klipper => {
                gcode.push_str(&format_exclude_object_define(
                    &exclude_object_name,
                    exclude_object_center_bed,
                    &exclude_object_polygon_bed,
                ));
                gcode.push('\n');
            }
            GcodeDialect::RepRapFirmware => {
                gcode.push_str("M486 T1 ; RepRapFirmware object cancellation count\n");
            }
        }
    }
    let start_expanded = expand_gcode_template(
        &gcfg.start_gcode_template,
        tool_id,
        nozzle_temp_c,
        gcfg.bed_temperature_celsius,
        nozzle_diam_mm,
        bbox_min,
        bbox_max,
        planned_layers.len(),
    );
    gcode.push_str(&start_expanded);
    if !start_expanded.ends_with('\n') {
        gcode.push('\n');
    }
    gcode.push_str("M83 ; Ensure relative extrusion\n");
    if support_tool_id != tool_id {
        let has_support_segs = planned_layers.iter().any(|l| {
            l.segments
                .iter()
                .any(|s| s.kind.is_support() && s.delta_e_mm > 1.0e-6)
        });
        if has_support_segs {
            match gcfg.gcode_dialect {
                GcodeDialect::Klipper => {
                    gcode.push_str(&format!(
                        "M104 S{support_nozzle_temp_c:.0} T{support_tool_id} ; Preheat support tool T{support_tool_id}\n"
                    ));
                }
                GcodeDialect::RepRapFirmware => {
                    gcode.push_str(&format!(
                        "M568 P{support_tool_id} S{support_nozzle_temp_c:.0} R{support_nozzle_temp_c:.0} A2 ; Preheat support tool T{support_tool_id}\n"
                    ));
                }
            }
        }
    }
    let init_pa_s = gcfg.pa_low_flow_s.max(0.0);
    gcode.push_str(&format!(
        "{} ; Initial Pressure Advance\n",
        format_pressure_advance_cmd(gcfg.gcode_dialect, tool_id, init_pa_s)
    ));

    let is_five_axis = mcfg.kinematics_mode == FiveAxisKinematicsMode::FiveAxisBC;
    let use_g93 = is_five_axis && mcfg.use_inverse_time_feedrate_g93;
    let mut in_g93_mode = false;
    let mut emitted_accel = -1.0f32;
    let mut emitted_feed_mm_min = -1.0f32;
    let mut emitted_pa = init_pa_s;
    let mut emitted_fan_pwm: Option<u8> = None;
    let mut last_emitted_kind: Option<PlannedMoveKind> = None;
    let mut emitted_tool_id = tool_id;
    let mut active_retract_mm = 0.0f32;
    let mut accum_travel_time_s = 0.0f32;
    // Track relative E-axis (`M83`) output rounding error in `f64` (error diffusion)
    // so cumulative filament extrusion across thousands of short non-planar/scarf segments
    // never drifts from the exact analytical volume integral.
    let mut e_rounding_residual_mm = 0.0f64;

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
            "\n; --- LAYER {} (Branch {}, Iso {:.4}) ---\n;LAYER_CHANGE\n;Z:{:.4}\n;HEIGHT:{:.4}\n",
            layer.layer_index, layer.branch_id, layer.isovalue, layer_max_z, layer_mean_h
        ));
        if gcfg.emit_exclude_object {
            match gcfg.gcode_dialect {
                GcodeDialect::Klipper => {
                    gcode.push_str(&format!("EXCLUDE_OBJECT_START NAME={exclude_object_name}\n"));
                }
                GcodeDialect::RepRapFirmware => {
                    let obj_idx = object_id.saturating_sub(1);
                    gcode.push_str(&format!("M486 S{obj_idx} A\"{exclude_object_name}\"\n"));
                }
            }
        }

        // Compute per-segment part cooling fan PWM schedule (`0..=255`) for this layer
        let fan_pwm_schedule =
            compute_layer_fan_pwm_schedule(layer, layer_ord, up_bed, &gcfg);
        if let Some(&first_pwm) = fan_pwm_schedule.first() {
            if emitted_fan_pwm != Some(first_pwm) {
                if first_pwm == 0 {
                    gcode.push_str("M107 ; Part cooling fan off\n");
                } else {
                    let pct = (first_pwm as f32) / 2.55;
                    gcode.push_str(&format!(
                        "M106 S{first_pwm} ; Part cooling fan {pct:.0}%\n"
                    ));
                }
                emitted_fan_pwm = Some(first_pwm);
            }
        }

        for (seg_idx, seg) in layer.segments.iter().enumerate() {
            total_time_s += seg.duration_s;

            // Emit dynamic or feature-override cooling fan update when scheduled PWM changes
            if seg.kind.is_extruding() && seg.delta_e_mm > 1.0e-6 {
                let seg_fan_pwm = fan_pwm_schedule.get(seg_idx).copied().unwrap_or(0);
                if emitted_fan_pwm != Some(seg_fan_pwm) {
                    if seg_fan_pwm == 0 {
                        gcode.push_str("M107 ; Part cooling fan off\n");
                    } else {
                        let seg_fan_pct = (seg_fan_pwm as f32) / 2.55;
                        gcode.push_str(&format!(
                            "M106 S{seg_fan_pwm} ; {} cooling fan {seg_fan_pct:.0}%\n",
                            seg.kind.label()
                        ));
                    }
                    emitted_fan_pwm = Some(seg_fan_pwm);
                }
            }

            // Emit retraction before non-extruding travel if scheduled
            if seg.retract_before_mm > 1.0e-4 && active_retract_mm <= 1.0e-4 {
                if in_g93_mode {
                    gcode.push_str("G94 ; Standard mm/min feedrate for retract\n");
                    in_g93_mode = false;
                }
                let r_len = seg.retract_before_mm;
                let r_feed = (gcfg.retract_speed_mms * 60.0).max(60.0);
                gcode.push_str(&format!("G1 E-{r_len:.6} F{r_feed:.1} ; Retract\n"));
                active_retract_mm = r_len;
                accum_travel_time_s = 0.0;
                emitted_feed_mm_min = r_feed;
                retract_count += 1;
                total_time_s += r_len / gcfg.retract_speed_mms.max(1.0);
            }

            // Emit toolchange (`T{id}`) before resuming extrusion if the segment uses a different tool
            if seg.kind.is_extruding() && seg.delta_e_mm > 1.0e-6 && seg.tool_id != emitted_tool_id {
                if active_retract_mm <= 1.0e-4 && gcfg.retract_length_mm > 1.0e-4 {
                    if in_g93_mode {
                        gcode.push_str("G94 ; Standard mm/min feedrate for retract\n");
                        in_g93_mode = false;
                    }
                    let r_len = gcfg.retract_length_mm;
                    let r_feed = (gcfg.retract_speed_mms * 60.0).max(60.0);
                    gcode.push_str(&format!(
                        "G1 E-{r_len:.6} F{r_feed:.1} ; Retract before toolchange\n"
                    ));
                    active_retract_mm = r_len;
                    accum_travel_time_s = 0.0;
                    retract_count += 1;
                    total_time_s += r_len / gcfg.retract_speed_mms.max(1.0);
                }
                let target_temp_c = if seg.tool_id == support_tool_id {
                    support_nozzle_temp_c
                } else {
                    nozzle_temp_c
                };
                match gcfg.gcode_dialect {
                    GcodeDialect::Klipper => {
                        gcode.push_str(&format!(
                            "M104 S{target_temp_c:.0} T{}\nT{} ; Switch tool ({})\n",
                            seg.tool_id,
                            seg.tool_id,
                            seg.kind.label()
                        ));
                    }
                    GcodeDialect::RepRapFirmware => {
                        gcode.push_str(&format!(
                            "M568 P{} S{target_temp_c:.0} R{target_temp_c:.0} A2\nT{} ; Switch tool ({})\n",
                            seg.tool_id,
                            seg.tool_id,
                            seg.kind.label()
                        ));
                    }
                }
                emitted_tool_id = seg.tool_id;
                emitted_pa = -1.0;
                emitted_feed_mm_min = -1.0;
                e_rounding_residual_mm = 0.0;
            }

            if !seg.kind.is_extruding() || seg.delta_e_mm <= 1.0e-6 {
                accum_travel_time_s += seg.duration_s;
            } else if active_retract_mm > 1.0e-4 {
                if in_g93_mode {
                    gcode.push_str("G94 ; Standard mm/min feedrate for unretract\n");
                    in_g93_mode = false;
                }
                // Unretract before resuming extrusion (with thermal ooze / swell relaxation)
                let seg_fluid = if seg.kind.is_support() {
                    &support_fluid
                } else {
                    &fluid
                };
                let u_len =
                    seg_fluid.unretract_length_mm(active_retract_mm, accum_travel_time_s, &gcfg);
                let u_feed = (gcfg.retract_speed_mms * 60.0).max(60.0);
                gcode.push_str(&format!("G1 E{u_len:.6} F{u_feed:.1} ; Unretract\n"));
                active_retract_mm = 0.0;
                accum_travel_time_s = 0.0;
                emitted_feed_mm_min = u_feed;
                total_time_s += u_len / gcfg.retract_speed_mms.max(1.0);
            }

            // Deadband-compacted `SET_VELOCITY_LIMIT ACCEL=...` / `M204 P... T...`
            let kind_changed = last_emitted_kind != Some(seg.kind);
            if kind_changed && seg.kind.is_extruding() {
                gcode.push_str(&format!("; TYPE:{}\n", seg.kind.label()));
            }
            if kind_changed
                || emitted_accel <= 0.0
                || ((seg.accel_mms2 - emitted_accel).abs() / emitted_accel.max(1.0)) >= 0.15
            {
                gcode.push_str(&format_accel_limit_cmd(
                    gcfg.gcode_dialect,
                    seg.accel_mms2,
                    gcfg.travel_accel_mms2,
                ));
                gcode.push('\n');
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

                // Emit firmware PA update after toolchange or when dynamic PA shifts outside 8% deadband
                if emitted_pa < 0.0
                    || (gcfg.enable_dynamic_pa
                        && (seg.pressure_advance_s - emitted_pa).abs()
                            / emitted_pa.max(0.005)
                            >= 0.08)
                {
                    gcode.push_str(&format_pressure_advance_cmd(
                        gcfg.gcode_dialect,
                        emitted_tool_id,
                        seg.pressure_advance_s,
                    ));
                    gcode.push('\n');
                    emitted_pa = seg.pressure_advance_s;
                }

                // Error-diffusion quantization of relative extrusion `E` at 6 decimal places (`1e-6 mm`)
                let e_desired = (seg.delta_e_mm as f64) + e_rounding_residual_mm;
                let e_emitted = ((e_desired * 1_000_000.0).round() * 1.0e-6).max(0.0);
                e_rounding_residual_mm = e_desired - e_emitted;

                if use_g93 {
                    if !in_g93_mode {
                        gcode.push_str("G93 ; Inverse-time feedrate mode (1/min)\n");
                        in_g93_mode = true;
                    }
                    let inv_time_f = (60.0 / seg.duration_s.max(1.0e-4)).clamp(0.1, 600_000.0);
                    gcode.push_str(&format!(
                        "G1 X{:.4} Y{:.4} Z{:.4} B{:.3} C{:.3} E{:.6} F{:.2}\n",
                        seg.p1_bed.x,
                        seg.p1_bed.y,
                        seg.p1_bed.z,
                        seg.b1_deg,
                        seg.c1_deg,
                        e_emitted,
                        inv_time_f
                    ));
                    emitted_feed_mm_min = -1.0;
                } else {
                    if in_g93_mode {
                        gcode.push_str("G94 ; Standard mm/min feedrate mode\n");
                        in_g93_mode = false;
                    }
                    // Emit the feature's nominal target speed (`seg.nominal_speed_mms * 60.0`) so the
                    // native motion planner handles corner deceleration/acceleration smoothly without
                    // facet-by-facet feedrate stuttering.
                    let feed_mm_min = (seg.nominal_speed_mms * 60.0).max(30.0);
                    let emit_f = kind_changed
                        || emitted_feed_mm_min <= 0.0
                        || ((feed_mm_min - emitted_feed_mm_min).abs() / emitted_feed_mm_min.max(1.0))
                            >= 0.05;

                    if is_five_axis {
                        if emit_f {
                            gcode.push_str(&format!(
                                "G1 X{:.4} Y{:.4} Z{:.4} B{:.3} C{:.3} E{:.6} F{:.1}\n",
                                seg.p1_bed.x,
                                seg.p1_bed.y,
                                seg.p1_bed.z,
                                seg.b1_deg,
                                seg.c1_deg,
                                e_emitted,
                                feed_mm_min
                            ));
                            emitted_feed_mm_min = feed_mm_min;
                        } else {
                            gcode.push_str(&format!(
                                "G1 X{:.4} Y{:.4} Z{:.4} B{:.3} C{:.3} E{:.6}\n",
                                seg.p1_bed.x,
                                seg.p1_bed.y,
                                seg.p1_bed.z,
                                seg.b1_deg,
                                seg.c1_deg,
                                e_emitted
                            ));
                        }
                    } else if emit_f {
                        gcode.push_str(&format!(
                            "G1 X{:.4} Y{:.4} Z{:.4} E{:.6} F{:.1}\n",
                            seg.p1_bed.x, seg.p1_bed.y, seg.p1_bed.z, e_emitted, feed_mm_min
                        ));
                        emitted_feed_mm_min = feed_mm_min;
                    } else {
                        gcode.push_str(&format!(
                            "G1 X{:.4} Y{:.4} Z{:.4} E{:.6}\n",
                            seg.p1_bed.x, seg.p1_bed.y, seg.p1_bed.z, e_emitted
                        ));
                    }
                }
            } else {
                travel_move_count += 1;
                if use_g93 {
                    if !in_g93_mode {
                        gcode.push_str("G93 ; Inverse-time feedrate mode (1/min)\n");
                        in_g93_mode = true;
                    }
                    let inv_time_f = (60.0 / seg.duration_s.max(1.0e-4)).clamp(0.1, 600_000.0);
                    gcode.push_str(&format!(
                        "G0 X{:.4} Y{:.4} Z{:.4} B{:.3} C{:.3} F{:.2}\n",
                        seg.p1_bed.x,
                        seg.p1_bed.y,
                        seg.p1_bed.z,
                        seg.b1_deg,
                        seg.c1_deg,
                        inv_time_f
                    ));
                    emitted_feed_mm_min = -1.0;
                } else {
                    if in_g93_mode {
                        gcode.push_str("G94 ; Standard mm/min feedrate mode\n");
                        in_g93_mode = false;
                    }
                    let feed_mm_min = (seg.nominal_speed_mms * 60.0).max(60.0);
                    let emit_f = kind_changed
                        || emitted_feed_mm_min <= 0.0
                        || ((feed_mm_min - emitted_feed_mm_min).abs() / emitted_feed_mm_min.max(1.0))
                            >= 0.05;

                    if is_five_axis {
                        if emit_f {
                            gcode.push_str(&format!(
                                "G0 X{:.4} Y{:.4} Z{:.4} B{:.3} C{:.3} F{:.1}\n",
                                seg.p1_bed.x,
                                seg.p1_bed.y,
                                seg.p1_bed.z,
                                seg.b1_deg,
                                seg.c1_deg,
                                feed_mm_min
                            ));
                            emitted_feed_mm_min = feed_mm_min;
                        } else {
                            gcode.push_str(&format!(
                                "G0 X{:.4} Y{:.4} Z{:.4} B{:.3} C{:.3}\n",
                                seg.p1_bed.x,
                                seg.p1_bed.y,
                                seg.p1_bed.z,
                                seg.b1_deg,
                                seg.c1_deg
                            ));
                        }
                    } else if emit_f {
                        gcode.push_str(&format!(
                            "G0 X{:.4} Y{:.4} Z{:.4} F{:.1}\n",
                            seg.p1_bed.x, seg.p1_bed.y, seg.p1_bed.z, feed_mm_min
                        ));
                        emitted_feed_mm_min = feed_mm_min;
                    } else {
                        gcode.push_str(&format!(
                            "G0 X{:.4} Y{:.4} Z{:.4}\n",
                            seg.p1_bed.x, seg.p1_bed.y, seg.p1_bed.z
                        ));
                    }
                }
            }

            last_emitted_kind = Some(seg.kind);
        }

        if gcfg.emit_exclude_object {
            match gcfg.gcode_dialect {
                GcodeDialect::Klipper => {
                    gcode.push_str(&format!("EXCLUDE_OBJECT_END NAME={exclude_object_name}\n"));
                }
                GcodeDialect::RepRapFirmware => {
                    gcode.push_str("M486 S-1\n");
                }
            }
        }
    }

    if in_g93_mode {
        gcode.push_str("G94 ; Restore standard mm/min feedrate mode\n");
    }

    gcode.push('\n');
    let end_expanded = expand_gcode_template(
        &gcfg.end_gcode_template,
        tool_id,
        nozzle_temp_c,
        gcfg.bed_temperature_celsius,
        nozzle_diam_mm,
        bbox_min,
        bbox_max,
        planned_layers.len(),
    );
    gcode.push_str(&end_expanded);
    if !end_expanded.ends_with('\n') {
        gcode.push('\n');
    }

    if min_speed > max_speed {
        min_speed = 0.0;
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
        max_mult = 1.0;
    }
    if min_pa > max_pa {
        min_pa = 0.02;
        max_pa = 0.05;
    }
    if min_press > max_press {
        min_press = 0.0;
        max_press = 15.0;
    }
    if min_w > max_w {
        min_w = 0.28;
        max_w = 0.56;
    }
    if min_h > max_h {
        min_h = 0.10;
        max_h = 0.30;
    }

    let total_filament_volume_mm3 = total_filament_mm * filament_area_mm2;
    let density_g_mm3 = (gcfg.filament_density_g_cm3 * 0.001).max(0.0001);
    let total_filament_grams = total_filament_volume_mm3 * density_g_mm3;

    gcode.push_str(&format_moonraker_metadata_footer(
        total_time_s,
        total_filament_mm,
        total_filament_volume_mm3,
        total_filament_grams,
        config.slicing.layer_height.first_layer_mm,
        config.slicing.layer_height.target_mm,
        bbox_max.z,
        planned_layers.len(),
        nozzle_diam_mm,
        gcfg.filament_diameter_mm,
        &gcfg.filament_type,
        gcfg.bed_temperature_celsius,
        nozzle_temp_c,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bridge::ffi::{ExtrusionPath, LayerToolpaths, ToolpathPoint};

    fn make_pt(x: f32, y: f32, z: f32, w: f32, h: f32) -> ToolpathPoint {
        ToolpathPoint {
            x,
            y,
            z,
            nx: 0.0,
            ny: 0.0,
            nz: 1.0,
            width_mm: w,
            height_mm: h,
        }
    }

    #[test]
    fn test_klipper_corner_kinematics_and_minimum_cruise_ratio_bounds() {
        // 1. At a 90-degree square corner, v_corner must equal SCV (5.0 mm/s)
        let (v_90, r_90) = compute_klipper_corner_kinematics(
            Vec3::X,
            Vec3::Y,
            5.0,
            3000.0,
            100.0,
            100.0,
            10.0,
            10.0,
        );
        assert!(
            (v_90 - 5.0).abs() < 1.0e-3,
            "Expected 90-deg corner velocity == SCV (5.0 mm/s), got {v_90}"
        );
        assert!(r_90 > 0.0 && r_90 < 1.0);

        // 2. At a shallow 2-degree deflection, v_corner must clamp to v_limit (100.0)
        //    and r_eff must stay small (no blow-up from dividing by sin(alpha/2)!)
        let shallow_dir = Vec3::new(2.0f32.to_radians().cos(), 2.0f32.to_radians().sin(), 0.0);
        let (v_shallow, r_shallow) = compute_klipper_corner_kinematics(
            Vec3::X,
            shallow_dir,
            5.0,
            3000.0,
            100.0,
            100.0,
            5.0,
            5.0,
        );
        assert!((v_shallow - 100.0).abs() < 1.0e-3);
        assert!(
            r_shallow < 2.0,
            "Expected shallow corner effective radius not to blow up, got {r_shallow}"
        );

        // 3. Stepper torque roll-off must stay below v_max_stepper and never NaN near v_max
        let v_reach = stepper_max_reachable_velocity(395.0, 5000.0, 20.0, 400.0, true);
        assert!(v_reach.is_finite() && v_reach >= 395.0 && v_reach <= 400.0);
    }

    #[test]
    fn test_open_vs_closed_wall_planning_and_gcode_emission() {
        // Construct a synthetic layer with:
        // - 1 closed outer wall loop (square 20x20 mm)
        // - 1 open inner wall polyline (3-sided U-channel that is NOT closed)
        // - 1 open sparse infill polyline
        let closed_outer = ExtrusionPath {
            points: vec![
                make_pt(-10.0, -10.0, 0.2, 0.42, 0.20),
                make_pt(10.0, -10.0, 0.2, 0.42, 0.20),
                make_pt(10.0, 10.0, 0.2, 0.42, 0.20),
                make_pt(-10.0, 10.0, 0.2, 0.42, 0.20),
            ],
            feature_kind: 0,
            wall_index: 0,
            is_closed: true,
            total_length_mm: 80.0,
            mean_width_mm: 0.42,
        };

        let open_inner = ExtrusionPath {
            points: vec![
                make_pt(-8.0, -8.0, 0.2, 0.42, 0.20),
                make_pt(8.0, -8.0, 0.2, 0.42, 0.20),
                make_pt(8.0, 8.0, 0.2, 0.42, 0.20),
            ],
            feature_kind: 1,
            wall_index: 1,
            is_closed: false,
            total_length_mm: 32.0,
            mean_width_mm: 0.42,
        };

        let open_sparse = ExtrusionPath {
            points: vec![
                make_pt(-5.0, -5.0, 0.2, 0.45, 0.20),
                make_pt(5.0, 5.0, 0.2, 0.45, 0.20),
                make_pt(5.06, 5.06, 0.2, 0.45, 0.20), // Very short 0.085mm final segment (< 0.5 * D_nozzle)
            ],
            feature_kind: 4,
            wall_index: 0,
            is_closed: false,
            total_length_mm: 14.14,
            mean_width_mm: 0.45,
        };

        let raw = ObjectToolpathsResult {
            valid: true,
            layers: vec![LayerToolpaths {
                layer_index: 0,
                branch_id: 0,
                isovalue: 0.2,
                paths: vec![closed_outer, open_inner, open_sparse],
            }],
            total_path_count: 3,
            outer_wall_count: 1,
            inner_wall_count: 1,
            solid_infill_count: 0,
            sparse_infill_count: 1,
            bridge_count: 0,
            wave_overhang_count: 0,
            support_count: 0,
            support_interface_count: 0,
        };

        let cfg = SlicerConfig::default();
        let res = plan_and_emit_gcode_for_object(
            "My Part (v1).stl",
            1,
            &raw,
            Transform::IDENTITY,
            &cfg,
        );

        assert!(res.extruding_move_count > 0);
        assert!(res.total_filament_mm > 0.0);
        assert!(res.total_time_s > 0.0);
        assert_eq!(res.exclude_object_name, "My_Part_v1");
        assert!(res
            .gcode_text
            .starts_with("; generated by PrusaSlicer 2.8.0-Manifold2 on "));
        assert!(res.gcode_text.contains("; thumbnail begin 32x32 "));
        assert!(res.gcode_text.contains("; thumbnail begin 300x300 "));
        assert!(res.gcode_text.contains("; iVBORw0KGgo")); // PNG magic header in Base64
        assert!(res.gcode_text.contains("; thumbnail end"));
        assert!(res
            .gcode_text
            .contains("EXCLUDE_OBJECT_DEFINE NAME=My_Part_v1 CENTER="));
        assert!(res.gcode_text.contains("POLYGON=[["));
        assert!(res
            .gcode_text
            .contains("EXCLUDE_OBJECT_START NAME=My_Part_v1\n"));
        assert!(res
            .gcode_text
            .contains("EXCLUDE_OBJECT_END NAME=My_Part_v1\n"));
        assert!(res.gcode_text.contains("; filament used [mm] = "));
        assert!(res.gcode_text.contains("; filament used [cm3] = "));
        assert!(res.gcode_text.contains("; filament used [g] = "));
        assert!(res.gcode_text.contains("; total filament used [g] = "));
        assert!(res
            .gcode_text
            .contains("; estimated printing time (normal mode) = "));
        assert!(res.gcode_text.contains("T0"));
        assert!(res.gcode_text.contains("M140 S60"));
        assert!(res.gcode_text.contains("M104 S215 T0"));
        assert!(res.gcode_text.contains("M83"));
        assert!(res.gcode_text.contains("SET_PRESSURE_ADVANCE ADVANCE="));
        assert!(res.gcode_text.contains("SET_VELOCITY_LIMIT ACCEL="));

        // Verify that the exported G-code bounding box is translated to the front-left bed origin
        // (`[0..256, 0..256]`, centered around `(128, 128)` for the default 256x256 mm rectangular bed)
        assert!(
            res.bbox_min_bed.x > 115.0 && res.bbox_max_bed.x < 141.0,
            "Expected bed X bounds centered around 128 mm, got [{}..{}]",
            res.bbox_min_bed.x,
            res.bbox_max_bed.x
        );
        assert!(
            res.bbox_min_bed.y > 115.0 && res.bbox_max_bed.y < 141.0,
            "Expected bed Y bounds centered around 128 mm, got [{}..{}]",
            res.bbox_min_bed.y,
            res.bbox_max_bed.y
        );

        // Verify that the open inner wall was NOT closed with a phantom diagonal segment
        // from (8, 8) back to (-8, -8), and that its wipe move did not protrude past X=8 or Y=8!
        for seg in &res.layers[0].segments {
            if seg.kind == PlannedMoveKind::InnerWall {
                let is_phantom_closing = (seg.p0_local.x - 8.0).abs() < 0.5
                    && (seg.p0_local.y - 8.0).abs() < 0.5
                    && (seg.p1_local.x - (-8.0)).abs() < 0.5
                    && (seg.p1_local.y - (-8.0)).abs() < 0.5;
                assert!(
                    !is_phantom_closing,
                    "Open inner wall must never emit a phantom closing segment!"
                );
            }
            // Verify cruise_speed_mms >= max(entry_speed_mms, exit_speed_mms) on every segment
            assert!(
                seg.cruise_speed_mms + 1.0e-3 >= seg.entry_speed_mms.max(seg.exit_speed_mms),
                "Segment cruise speed {} fell below entry/exit ({}, {})!",
                seg.cruise_speed_mms,
                seg.entry_speed_mms,
                seg.exit_speed_mms
            );
        }
    }

    #[test]
    fn test_thin_vs_thick_layer_height_scales_extrusion_volume_and_flow_rate() {
        let make_square_layer = |layer_index: u32, z: f32, h: f32| -> LayerToolpaths {
            LayerToolpaths {
                layer_index,
                branch_id: 0,
                isovalue: z,
                paths: vec![ExtrusionPath {
                    feature_kind: 0,
                    wall_index: 0,
                    is_closed: false,
                    total_length_mm: 40.0,
                    mean_width_mm: 0.42,
                    points: vec![
                        ToolpathPoint {
                            x: -20.0,
                            y: 0.0,
                            z,
                            nx: 0.0,
                            ny: 0.0,
                            nz: 1.0,
                            width_mm: 0.42,
                            height_mm: h,
                        },
                        ToolpathPoint {
                            x: 0.0,
                            y: 0.0,
                            z,
                            nx: 0.0,
                            ny: 0.0,
                            nz: 1.0,
                            width_mm: 0.42,
                            height_mm: h,
                        },
                        ToolpathPoint {
                            x: 20.0,
                            y: 0.0,
                            z,
                            nx: 0.0,
                            ny: 0.0,
                            nz: 1.0,
                            width_mm: 0.42,
                            height_mm: h,
                        },
                    ],
                }],
            }
        };

        let raw = ObjectToolpathsResult {
            valid: true,
            total_path_count: 3,
            outer_wall_count: 3,
            inner_wall_count: 0,
            solid_infill_count: 0,
            sparse_infill_count: 0,
            bridge_count: 0,
            wave_overhang_count: 0,
            support_count: 0,
            support_interface_count: 0,
            layers: vec![
                make_square_layer(0, 0.20, 0.20),
                make_square_layer(1, 0.28, 0.08), // thin upper layer (0.08 mm)
                make_square_layer(2, 0.56, 0.28), // thick upper layer (0.28 mm)
            ],
        };

        let cfg = SlicerConfig::default();
        let res = plan_and_emit_gcode_for_object("HeightScaleTest", 1, &raw, Transform::IDENTITY, &cfg);
        assert_eq!(res.layers.len(), 3);

        let thin_segs: Vec<_> = res.layers[1]
            .segments
            .iter()
            .filter(|s| s.kind.is_extruding() && s.delta_e_mm > 0.0)
            .collect();
        let thick_segs: Vec<_> = res.layers[2]
            .segments
            .iter()
            .filter(|s| s.kind.is_extruding() && s.delta_e_mm > 0.0)
            .collect();
        assert!(!thin_segs.is_empty() && !thick_segs.is_empty());

        let thin_h = thin_segs.iter().map(|s| s.height_mm).sum::<f32>() / (thin_segs.len() as f32);
        let thick_h = thick_segs.iter().map(|s| s.height_mm).sum::<f32>() / (thick_segs.len() as f32);
        assert!((thin_h - 0.08).abs() < 1.0e-3);
        assert!((thick_h - 0.28).abs() < 1.0e-3);

        let thin_e_per_mm: f32 = thin_segs.iter().map(|s| s.delta_e_mm).sum::<f32>()
            / thin_segs.iter().map(|s| s.length_mm).sum::<f32>();
        let thick_e_per_mm: f32 = thick_segs.iter().map(|s| s.delta_e_mm).sum::<f32>()
            / thick_segs.iter().map(|s| s.length_mm).sum::<f32>();
        let thin_max_q = thin_segs
            .iter()
            .map(|s| s.volumetric_flow_mm3s)
            .fold(0.0f32, f32::max);
        let thick_max_q = thick_segs
            .iter()
            .map(|s| s.volumetric_flow_mm3s)
            .fold(0.0f32, f32::max);

        assert!(
            thick_e_per_mm > 2.5 * thin_e_per_mm,
            "Expected thick layer E/mm ({thick_e_per_mm:.4}) > 2.5x thin layer E/mm ({thin_e_per_mm:.4})"
        );
        assert!(
            thick_max_q > 2.5 * thin_max_q,
            "Expected thick layer flow rate ({thick_max_q:.2} mm^3/s) > 2.5x thin layer flow rate ({thin_max_q:.2} mm^3/s)"
        );
    }

    #[test]
    fn test_bridge_and_wave_overhang_planning_order_volume_speed_and_cooling() {
        // Construct an upper layer (layer_index = 2) containing:
        // - 1 supported inner wall (feature_kind = 1, wall_index = 1)
        // - 2 wave overhang passes (feature_kind = 6, passed out-of-order with wall_index = 2 then 1)
        // - 1 3D inclined straight-line circular bridge (feature_kind = 5, 20mm span in 3D)
        let supported_inner = ExtrusionPath {
            feature_kind: 1,
            wall_index: 1,
            is_closed: false,
            total_length_mm: 20.0,
            mean_width_mm: 0.45,
            points: vec![
                make_pt(-10.0, 0.0, 2.0, 0.45, 0.20),
                make_pt(10.0, 0.0, 2.0, 0.45, 0.20),
            ],
        };
        let wave_outer_pass = ExtrusionPath {
            feature_kind: 6,
            wall_index: 2, // outer wave pass (must be re-ordered AFTER wall_index = 1)
            is_closed: false,
            total_length_mm: 20.0,
            mean_width_mm: 0.45,
            points: vec![
                make_pt(-10.0, 0.828, 2.0, 0.45, 0.20),
                make_pt(10.0, 0.828, 2.0, 0.45, 0.20),
            ],
        };
        let wave_inner_pass = ExtrusionPath {
            feature_kind: 6,
            wall_index: 1, // inner wave pass (must print first among wave overhangs: inside-out!)
            is_closed: false,
            total_length_mm: 20.0,
            mean_width_mm: 0.45,
            points: vec![
                make_pt(-10.0, 0.414, 2.0, 0.45, 0.20),
                make_pt(10.0, 0.414, 2.0, 0.45, 0.20),
            ],
        };
        // Straight 3D bridge climbing from Z=2.0 to Z=4.0 across X=[-10..10]
        let bridge_path = ExtrusionPath {
            feature_kind: 5,
            wall_index: 0,
            is_closed: false,
            total_length_mm: (20.0f32 * 20.0 + 2.0 * 2.0).sqrt(),
            mean_width_mm: 0.40,
            points: vec![
                make_pt(-10.0, 5.0, 2.0, 0.40, 0.40),
                make_pt(10.0, 5.0, 4.0, 0.40, 0.40),
            ],
        };

        let raw = ObjectToolpathsResult {
            valid: true,
            total_path_count: 4,
            outer_wall_count: 0,
            inner_wall_count: 1,
            solid_infill_count: 0,
            sparse_infill_count: 0,
            bridge_count: 1,
            wave_overhang_count: 2,
            support_count: 0,
            support_interface_count: 0,
            layers: vec![
                LayerToolpaths {
                    layer_index: 0,
                    branch_id: 0,
                    isovalue: 0.2,
                    paths: vec![supported_inner.clone()],
                },
                LayerToolpaths {
                    layer_index: 1,
                    branch_id: 0,
                    isovalue: 2.0,
                    paths: vec![
                        wave_outer_pass,
                        bridge_path,
                        supported_inner,
                        wave_inner_pass,
                    ],
                },
            ],
        };

        let mut cfg = SlicerConfig::default();
        cfg.slicing.wave_overhang_overlap_fraction = 0.08;
        cfg.gcode.inner_wall_speed_mms = 80.0;
        cfg.gcode.bridge_speed_mms = 28.0;
        cfg.gcode.wave_overhang_speed_mms = 22.0;
        cfg.gcode.bridge_flow_multiplier = 0.90;
        cfg.gcode.wave_overhang_flow_multiplier = 1.05;
        cfg.gcode.fan_speed_percent = 50.0;
        cfg.gcode.bridge_fan_speed_percent = 100.0;
        cfg.gcode.wave_overhang_fan_speed_percent = 90.0;

        let res = plan_and_emit_gcode_for_object(
            "BridgeAndWaveTest",
            1,
            &raw,
            Transform::IDENTITY,
            &cfg,
        );
        let segs = &res.layers[1].segments;

        // 1. Verify inside-out ordering: Supported InnerWall -> WaveOverhang(wall_index=1) -> WaveOverhang(wall_index=2) -> Bridge
        let first_inner_idx = segs
            .iter()
            .position(|s| s.kind == PlannedMoveKind::InnerWall)
            .expect("Expected InnerWall segment");
        let wave_segs: Vec<(usize, &PlannedSegment)> = segs
            .iter()
            .enumerate()
            .filter(|(_, s)| s.kind == PlannedMoveKind::WaveOverhang)
            .collect();
        let bridge_segs: Vec<(usize, &PlannedSegment)> = segs
            .iter()
            .enumerate()
            .filter(|(_, s)| s.kind == PlannedMoveKind::Bridge)
            .collect();
        assert!(!wave_segs.is_empty(), "Expected WaveOverhang segments");
        assert_eq!(
            bridge_segs.len(),
            1,
            "Bridge must remain a single un-subdivided straight-line 3D segment"
        );
        assert!(
            first_inner_idx < wave_segs[0].0,
            "Supported walls must print before WaveOverhangs"
        );
        // First wave segment must be at Y = 0.414 (wall_index = 1), last at Y = 0.828 (wall_index = 2)
        assert!(
            (wave_segs.first().unwrap().1.p0_local.y - 0.414).abs() < 1.0e-3,
            "Inside-out WaveOverhang ordering violated: first wave Y={}",
            wave_segs.first().unwrap().1.p0_local.y
        );
        assert!(
            (wave_segs.last().unwrap().1.p0_local.y - 0.828).abs() < 1.0e-3,
            "Inside-out WaveOverhang ordering violated: last wave Y={}",
            wave_segs.last().unwrap().1.p0_local.y
        );

        // 2. Verify Bridge circular cross-section & 3D straight-line geometry & volume
        let b_seg = bridge_segs[0].1;
        assert!((b_seg.width_mm - 0.40).abs() < 1.0e-4);
        assert!((b_seg.height_mm - 0.40).abs() < 1.0e-4);
        assert!((b_seg.p0_local.z - 2.0).abs() < 1.0e-4 && (b_seg.p1_local.z - 4.0).abs() < 1.0e-4);
        assert!(b_seg.cruise_speed_mms <= 28.0 + 1.0e-3);

        let expected_circle_area = std::f32::consts::FRAC_PI_4 * 0.40 * 0.40;
        let fil_area = std::f32::consts::FRAC_PI_4 * cfg.gcode.filament_diameter_mm.powi(2);
        let expected_bridge_e =
            (expected_circle_area * b_seg.length_mm * b_seg.extrusion_multiplier) / fil_area;
        assert!(
            (b_seg.delta_e_mm - expected_bridge_e).abs() < 1.0e-3,
            "Bridge delta_e_mm ({}) did not match circular area volume ({expected_bridge_e})",
            b_seg.delta_e_mm
        );

        // 3. Verify WaveOverhang overlap-adjusted volume and speed
        let w_seg = wave_segs[0].1;
        assert!(w_seg.cruise_speed_mms <= 22.0 + 1.0e-3);
        let std_area =
            0.20 * (0.45 - 0.20) + std::f32::consts::FRAC_PI_4 * 0.20 * 0.20;
        let overlap_mm = 0.08 * cfg.slicing.outer_wall_width.target_mm;
        let expected_wave_area = std_area * ((0.45 - overlap_mm) / 0.45);
        let expected_wave_e =
            (expected_wave_area * w_seg.length_mm * w_seg.extrusion_multiplier) / fil_area;
        assert!(
            (w_seg.delta_e_mm - expected_wave_e).abs() < 1.0e-3,
            "WaveOverhang delta_e_mm ({}) did not match overlap-reduced volume ({expected_wave_e})",
            w_seg.delta_e_mm
        );

        // 4. Verify G-code emits dedicated fan speeds for Bridge (100% -> S255) and WaveOverhang (90% -> S230)
        assert!(
            res.gcode_text.contains("M106 S255"),
            "Expected M106 S255 for 100% Bridge fan cooling"
        );
        assert!(
            res.gcode_text.contains("M106 S230"),
            "Expected M106 S230 for 90% WaveOverhang fan cooling"
        );
    }

    #[test]
    fn test_gcode_extra_decimal_precision_and_e_rounding_error_compensation() {
        // Construct a fine non-planar perimeter loop with 360 short segments (~0.35 mm each)
        // and scarf-joint ramping where per-segment dE is very small (~0.0005 to ~0.008 mm)
        // and systematic rounding without error diffusion would accumulate noticeable drift.
        let n_pts = 360usize;
        let radius = 20.0f32;
        let mut pts = Vec::with_capacity(n_pts);
        for i in 0..n_pts {
            let theta = (i as f32) * std::f32::consts::TAU / (n_pts as f32);
            let z = 0.25 + 0.04 * (3.0 * theta).sin();
            pts.push(make_pt(
                radius * theta.cos(),
                radius * theta.sin(),
                z,
                0.36,
                0.12,
            ));
        }
        let fine_loop = ExtrusionPath {
            points: pts,
            feature_kind: 0,
            wall_index: 0,
            is_closed: true,
            total_length_mm: std::f32::consts::TAU * radius,
            mean_width_mm: 0.36,
        };
        let raw = ObjectToolpathsResult {
            valid: true,
            total_path_count: 1,
            outer_wall_count: 1,
            inner_wall_count: 0,
            solid_infill_count: 0,
            sparse_infill_count: 0,
            bridge_count: 0,
            wave_overhang_count: 0,
            support_count: 0,
            support_interface_count: 0,
            layers: vec![LayerToolpaths {
                layer_index: 0,
                branch_id: 0,
                isovalue: 0.25,
                paths: vec![fine_loop],
            }],
        };

        let cfg = SlicerConfig::default();
        let res = plan_and_emit_gcode_for_object(
            "PrecisionAndRoundingTest",
            1,
            &raw,
            Transform::IDENTITY,
            &cfg,
        );

        let exact_sum_e: f64 = res.layers[0]
            .segments
            .iter()
            .filter(|s| s.kind.is_extruding() && s.delta_e_mm > 1.0e-6)
            .map(|s| s.delta_e_mm as f64)
            .sum();

        let mut emitted_sum_e = 0.0f64;
        let mut checked_coord_decimals = false;
        for line in res.gcode_text.lines() {
            if line.starts_with("G1 X") {
                for token in line.split_whitespace() {
                    if let Some(x_str) = token.strip_prefix('X') {
                        let dot_pos = x_str.find('.').expect("X coordinate must have decimal");
                        assert_eq!(
                            x_str.len() - dot_pos - 1,
                            4,
                            "Expected 4 decimal places on X coordinate, got {token}"
                        );
                        checked_coord_decimals = true;
                    } else if let Some(e_str) = token.strip_prefix('E') {
                        let dot_pos = e_str.find('.').expect("E value must have decimal");
                        assert_eq!(
                            e_str.len() - dot_pos - 1,
                            6,
                            "Expected 6 decimal places on E axis, got {token}"
                        );
                        let e_val: f64 = e_str.parse().expect("Valid float E");
                        emitted_sum_e += e_val;
                    }
                }
            }
        }
        assert!(checked_coord_decimals);
        let total_rounding_drift_mm = (emitted_sum_e - exact_sum_e).abs();
        assert!(
            total_rounding_drift_mm <= 0.55e-6,
            "Expected cumulative E rounding drift <= 0.55e-6 mm with error diffusion, got {total_rounding_drift_mm:.9} mm (emitted={emitted_sum_e:.8}, exact={exact_sum_e:.8})"
        );
    }

    #[test]
    fn test_five_axis_rrf_bc_planning_zenith_lookahead_rotary_throttling_and_g93() {
        // Construct a 2-path layer on a circular bed:
        // - Path 0: flat planar path at the center with vertical nozzle (0, 0, 1) -> B = 0
        // - Path 1: non-planar tilted cone wall where the nozzle tilts to 30 deg at azimuth 90 deg (+Y)
        //           and then sweeps azimuth around a circle.
        let flat_path = ExtrusionPath {
            feature_kind: 1,
            wall_index: 2, // InnerOuterInner prints wall_index >= 2 first, then outer wall 0
            is_closed: false,
            total_length_mm: 10.0,
            mean_width_mm: 0.42,
            points: vec![
                ToolpathPoint {
                    x: -5.0,
                    y: -10.0,
                    z: 0.5,
                    nx: 0.0,
                    ny: 0.0,
                    nz: 1.0,
                    width_mm: 0.42,
                    height_mm: 0.20,
                },
                ToolpathPoint {
                    x: 5.0,
                    y: -10.0,
                    z: 0.5,
                    nx: 0.0,
                    ny: 0.0,
                    nz: 1.0,
                    width_mm: 0.42,
                    height_mm: 0.20,
                },
            ],
        };

        let tilt_rad = 30.0f32.to_radians();
        let mut cone_pts = Vec::new();
        for i in 0..=16 {
            let az = std::f32::consts::FRAC_PI_2 + (i as f32) * std::f32::consts::TAU / 16.0;
            let r = 20.0f32;
            cone_pts.push(ToolpathPoint {
                x: r * az.cos(),
                y: r * az.sin(),
                z: 2.0,
                nx: tilt_rad.sin() * az.cos(),
                ny: tilt_rad.sin() * az.sin(),
                nz: tilt_rad.cos(),
                width_mm: 0.42,
                height_mm: 0.20,
            });
        }
        let cone_path = ExtrusionPath {
            feature_kind: 0,
            wall_index: 0,
            is_closed: true,
            total_length_mm: std::f32::consts::TAU * 20.0,
            mean_width_mm: 0.42,
            points: cone_pts,
        };

        let raw = ObjectToolpathsResult {
            valid: true,
            total_path_count: 2,
            outer_wall_count: 1,
            inner_wall_count: 1,
            solid_infill_count: 0,
            sparse_infill_count: 0,
            bridge_count: 0,
            wave_overhang_count: 0,
            support_count: 0,
            support_interface_count: 0,
            layers: vec![LayerToolpaths {
                layer_index: 0,
                branch_id: 0,
                isovalue: 0.5,
                paths: vec![flat_path, cone_path],
            }],
        };

        let mut cfg = SlicerConfig::default();
        cfg.bed.shape = crate::config::BedShape::Circular;
        cfg.bed.width_x = 220.0;
        cfg.bed.depth_y = 220.0;
        cfg.gcode.gcode_origin_at_bed_center = true;
        cfg.gcode.gcode_dialect = GcodeDialect::RepRapFirmware;
        cfg.gcode.z_hop_mm = 0.0; // Direct travel between flat_path and cone_path to test Travel C lookahead
        cfg.gcode.outer_wall_speed_mms = 120.0;
        cfg.machine.kinematics_mode = FiveAxisKinematicsMode::FiveAxisBC;
        cfg.machine.b_min_deg = 0.0;
        cfg.machine.b_max_deg = 65.0;
        cfg.machine.c_continuous = true;
        cfg.machine.max_b_speed_deg_s = 30.0;
        cfg.machine.max_c_speed_deg_s = 45.0; // Deliberately moderate C speed to verify linear speed throttling
        cfg.machine.orientation_smoothing_window_mm = 1.5;

        let res = plan_and_emit_gcode_for_object(
            "FiveAxisRrfCone",
            1,
            &raw,
            Transform::IDENTITY,
            &cfg,
        );

        let segs = &res.layers[0].segments;
        // 1. Flat segments at zenith (B ~ 0) must keep B = 0
        let first_inner = segs
            .iter()
            .find(|s| s.kind == PlannedMoveKind::InnerWall)
            .expect("Expected flat InnerWall segment");
        assert!(first_inner.b1_deg.abs() < 1.0e-3);

        // 2. Travel move from flat InnerWall to tilted OuterWall must pre-position C toward the upcoming tilted azimuth (~ -67.5 deg at the nearest point on the cone)
        let travel_to_cone = segs
            .iter()
            .find(|s| s.kind == PlannedMoveKind::Travel && s.b1_deg > 10.0)
            .expect("Expected Travel segment transitioning to tilted cone");
        assert!(
            (travel_to_cone.c1_deg - (-67.5)).abs() < 20.0,
            "Expected Travel move to pre-orient C near -67.5 deg, got C={}",
            travel_to_cone.c1_deg
        );

        // 3. OuterWall segments on the cone must have B ~ 30 deg, continuous C > 300 deg by the end of the loop,
        //    and nominal_speed_mms throttled below 120 mm/s by max_c_speed_deg_s (45 deg/s).
        let outer_segs: Vec<&PlannedSegment> = segs
            .iter()
            .filter(|s| s.kind == PlannedMoveKind::OuterWall)
            .collect();
        assert!(!outer_segs.is_empty());
        let max_c_seen = outer_segs
            .iter()
            .map(|s| s.c1_deg)
            .fold(f32::NEG_INFINITY, f32::max);
        let min_c_seen = outer_segs
            .iter()
            .map(|s| s.c1_deg)
            .fold(f32::INFINITY, f32::min);
        assert!(
            (max_c_seen - min_c_seen) > 300.0,
            "Expected continuous C sweep > 300 deg around closed cone loop, got min={min_c_seen}, max={max_c_seen}"
        );
        assert!(
            outer_segs.iter().any(|s| s.nominal_speed_mms < 100.0),
            "Expected rotary C speed limit (45 deg/s) to throttle linear nominal_speed_mms below 120 mm/s"
        );

        // 4. Verify RRF dialect commands and B/C axis coordinates in emitted G-code
        assert!(
            res.gcode_text.contains("M486 T1"),
            "Expected RRF M486 T1 object count header"
        );
        assert!(
            res.gcode_text.contains("M486 S0 A\"FiveAxisRrfCone\""),
            "Expected RRF M486 S0 object start command"
        );
        assert!(
            res.gcode_text.contains("M572 D0 S"),
            "Expected RRF M572 D0 S... pressure advance command"
        );
        assert!(
            res.gcode_text.contains("M204 P"),
            "Expected RRF M204 P... T... acceleration command"
        );
        assert!(
            !res.gcode_text.contains("SET_PRESSURE_ADVANCE")
                && !res.gcode_text.contains("SET_VELOCITY_LIMIT"),
            "RRF dialect must not emit Klipper SET_PRESSURE_ADVANCE or SET_VELOCITY_LIMIT macros"
        );
        assert!(
            res.gcode_text
                .lines()
                .any(|l| l.starts_with("G1 X") && l.contains(" B") && l.contains(" C") && l.contains(" E")),
            "Expected 5-axis G1 X... Y... Z... B... C... E... lines"
        );

        // 5. Verify G93 Inverse-Time Feedrate mode
        cfg.machine.use_inverse_time_feedrate_g93 = true;
        let res_g93 = plan_and_emit_gcode_for_object(
            "FiveAxisRrfConeG93",
            1,
            &raw,
            Transform::IDENTITY,
            &cfg,
        );
        assert!(
            res_g93
                .gcode_text
                .contains("G93 ; Inverse-time feedrate mode (1/min)"),
            "Expected G93 mode switch when use_inverse_time_feedrate_g93 = true"
        );
        assert!(
            res_g93
                .gcode_text
                .contains("G94 ; Restore standard mm/min feedrate mode"),
            "Expected G94 restore at end of print"
        );
    }

    #[test]
    fn test_obstacle_aware_zhop_low_to_high_ordering_auto_brim_lever_slowdown_and_dynamic_fan() {
        // Construct a tall narrow-base 6-layer non-planar part (base 8x8 mm, top Z = 32 mm -> lever ratio = 4.0 > 2.2):
        // - Layer 0: 8x8 mm square at Z=0.20 (triggers auto-brim)
        // - Layer 1: two chains at different Z heights (a high ridge at X=0, Z=3.5 and a low valley at X=6, Z=0.5)
        //            plus a third chain at X=-6, Z=0.5 so travel between X=-6 and X=+6 crosses the X=0 ridge!
        // - Layers 2..5: tall square loops climbing up to Z=32.0 mm with sharp 90-deg corners and varying segment speeds.
        let make_square_loop = |half_w: f32, z: f32, w: f32, h: f32| -> ExtrusionPath {
            ExtrusionPath {
                feature_kind: 0,
                wall_index: 0,
                is_closed: true,
                total_length_mm: 8.0 * half_w,
                mean_width_mm: w,
                points: vec![
                    make_pt(-half_w, -half_w, z, w, h),
                    make_pt(half_w, -half_w, z, w, h),
                    make_pt(half_w, half_w, z, w, h),
                    make_pt(-half_w, half_w, z, w, h),
                ],
            }
        };

        // On Layer 1, place the high ridge FIRST in the input `paths` vector to verify
        // `order_layer_chains` re-orders low-Z valleys before high-Z peaks, and also put
        // a tall obstacle on Layer 0 at X=0, Z=2.2 mm so travel across X=0 on Layer 1 must Z-hop over it.
        let obstacle_ridge_l0 = ExtrusionPath {
            feature_kind: 1,
            wall_index: 1,
            is_closed: false,
            total_length_mm: 8.0,
            mean_width_mm: 0.42,
            points: vec![
                make_pt(0.0, -4.0, 2.2, 0.42, 0.20),
                make_pt(0.0, 4.0, 2.2, 0.42, 0.20),
            ],
        };
        let high_peak_l1 = ExtrusionPath {
            feature_kind: 0,
            wall_index: 0,
            is_closed: false,
            total_length_mm: 8.0,
            mean_width_mm: 0.42,
            points: vec![
                make_pt(0.0, -4.0, 3.4, 0.42, 0.20),
                make_pt(0.0, 4.0, 3.4, 0.42, 0.20),
            ],
        };
        let left_valley_l1 = ExtrusionPath {
            feature_kind: 0,
            wall_index: 0,
            is_closed: false,
            total_length_mm: 8.0,
            mean_width_mm: 0.42,
            points: vec![
                make_pt(-3.5, -4.0, 0.6, 0.42, 0.20),
                make_pt(-3.5, 4.0, 0.6, 0.42, 0.20),
            ],
        };
        let right_valley_l1 = ExtrusionPath {
            feature_kind: 0,
            wall_index: 0,
            is_closed: false,
            total_length_mm: 8.0,
            mean_width_mm: 0.42,
            points: vec![
                make_pt(3.5, -4.0, 0.6, 0.42, 0.20),
                make_pt(3.5, 4.0, 0.6, 0.42, 0.20),
            ],
        };

        let raw = ObjectToolpathsResult {
            valid: true,
            total_path_count: 9,
            outer_wall_count: 8,
            inner_wall_count: 1,
            solid_infill_count: 0,
            sparse_infill_count: 0,
            bridge_count: 0,
            wave_overhang_count: 0,
            support_count: 0,
            support_interface_count: 0,
            layers: vec![
                LayerToolpaths {
                    layer_index: 0,
                    branch_id: 0,
                    isovalue: 0.2,
                    paths: vec![make_square_loop(4.0, 0.20, 0.42, 0.20), obstacle_ridge_l0],
                },
                LayerToolpaths {
                    layer_index: 1,
                    branch_id: 0,
                    isovalue: 1.0,
                    paths: vec![high_peak_l1, left_valley_l1, right_valley_l1],
                },
                LayerToolpaths {
                    layer_index: 2,
                    branch_id: 0,
                    isovalue: 8.0,
                    paths: vec![make_square_loop(4.0, 8.0, 0.42, 0.20)],
                },
                LayerToolpaths {
                    layer_index: 3,
                    branch_id: 0,
                    isovalue: 16.0,
                    paths: vec![make_square_loop(4.0, 16.0, 0.42, 0.20)],
                },
                LayerToolpaths {
                    layer_index: 4,
                    branch_id: 0,
                    isovalue: 24.0,
                    paths: vec![make_square_loop(4.0, 24.0, 0.42, 0.20)],
                },
                LayerToolpaths {
                    layer_index: 5,
                    branch_id: 0,
                    isovalue: 32.0,
                    paths: vec![
                        make_square_loop(4.0, 32.0, 0.42, 0.20),
                        ExtrusionPath {
                            feature_kind: 4, // SparseInfill (higher speed/flow than OuterWall)
                            wall_index: 0,
                            is_closed: false,
                            total_length_mm: 24.0,
                            mean_width_mm: 0.48,
                            points: vec![
                                make_pt(-3.5, -3.5, 32.0, 0.48, 0.28),
                                make_pt(3.5, 3.5, 32.0, 0.48, 0.28),
                                make_pt(-3.5, 3.5, 32.0, 0.48, 0.28),
                                make_pt(3.5, -3.5, 32.0, 0.48, 0.28),
                            ],
                        },
                    ],
                },
            ],
        };

        let mut cfg = SlicerConfig::default();
        cfg.gcode.outer_wall_speed_mms = 120.0;
        cfg.gcode.sparse_infill_speed_mms = 200.0;
        cfg.gcode.enable_auto_brim = true;
        cfg.gcode.brim_width_mm = 0.0; // Auto-brim should activate because lever_ratio = 32 / 8 = 4.0 > 2.2
        cfg.gcode.enable_lever_arm_slowdown = true;
        cfg.gcode.lever_arm_reference_ratio = 1.5;
        cfg.gcode.lever_arm_min_speed_fraction = 0.30;
        cfg.gcode.enable_dynamic_fan_speed = true;
        cfg.gcode.min_fan_speed_percent = 25.0;
        cfg.gcode.max_fan_speed_percent = 100.0;

        let res = plan_and_emit_gcode_for_object(
            "TallCantileverTest",
            1,
            &raw,
            Transform::IDENTITY,
            &cfg,
        );

        // 1. Verify Auto-Brim expanded Layer 0 footprint outward beyond the 8x8 mm square (`[-4..+4]` -> `> 6.0 mm` half-span)
        let l0_ext_min_x = res.layers[0]
            .segments
            .iter()
            .filter(|s| s.kind.is_extruding())
            .map(|s| s.p0_local.x.min(s.p1_local.x))
            .fold(f32::MAX, f32::min);
        assert!(
            l0_ext_min_x < -6.0,
            "Expected auto-brim on Layer 0 to extend outward past X=-6.0 mm, got min X={l0_ext_min_x}"
        );

        // 2. Verify Low-to-High intra-layer chain ordering on Layer 1:
        //    Both Z=0.6 valleys (`X = +/-3.5`) must print BEFORE the Z=3.4 peak (`X = 0.0`)!
        let l1_ext: Vec<&PlannedSegment> = res.layers[1]
            .segments
            .iter()
            .filter(|s| s.kind.is_extruding() && s.delta_e_mm > 1.0e-5)
            .collect();
        assert!(l1_ext.len() >= 3);
        assert!(
            l1_ext.first().unwrap().p0_local.z < 1.0,
            "Expected lowest-elevation valley chain (Z=0.6) to print first on Layer 1, got Z={}",
            l1_ext.first().unwrap().p0_local.z
        );
        assert!(
            l1_ext.last().unwrap().p1_local.z > 3.0,
            "Expected highest-elevation peak chain (Z=3.4) to print last on Layer 1, got Z={}",
            l1_ext.last().unwrap().p1_local.z
        );

        // 3. Verify Obstacle-Aware Z-Hop on Layer 1:
        //    Traveling from `left_valley_l1` (Z=0.6, X=-3.5) to `right_valley_l1` (Z=0.6, X=+3.5)
        //    crosses the Layer 0 ridge at `X=0, Z=2.2`, so the travel segment must lift above Z=2.45 mm!
        let l1_high_travel = res.layers[1]
            .segments
            .iter()
            .find(|s| s.kind == PlannedMoveKind::Travel && s.p0_local.x * s.p1_local.x < 0.0)
            .expect("Expected cross-ridge Travel segment on Layer 1");
        assert!(
            l1_high_travel.p0_local.z >= 2.45 && l1_high_travel.p1_local.z >= 2.45,
            "Expected obstacle-aware Z-hop travel across Z=2.2 ridge to clear Z>=2.45 mm, got ({:.2} -> {:.2})",
            l1_high_travel.p0_local.z,
            l1_high_travel.p1_local.z
        );

        // 4. Verify Lever-Arm & Curled-Corner Slowdown progressively throttles nominal speed & accel on upper layers:
        let l2_max_wall_spd = res.layers[2]
            .segments
            .iter()
            .filter(|s| s.kind == PlannedMoveKind::OuterWall)
            .map(|s| s.nominal_speed_mms)
            .fold(0.0f32, f32::max);
        let l5_max_wall_spd = res.layers[5]
            .segments
            .iter()
            .filter(|s| s.kind == PlannedMoveKind::OuterWall)
            .map(|s| s.nominal_speed_mms)
            .fold(0.0f32, f32::max);
        assert!(
            l5_max_wall_spd < l2_max_wall_spd && l5_max_wall_spd < 90.0,
            "Expected top layer (Z=32mm) outer wall speed ({l5_max_wall_spd:.1} mm/s) to be throttled below Layer 2 ({l2_max_wall_spd:.1} mm/s) and < 90 mm/s"
        );

        // 5. Verify Dynamic Cooling Fan emits multiple distinct M106 S... values across the print
        let mut m106_values: Vec<u8> = Vec::new();
        for line in res.gcode_text.lines() {
            if let Some(rest) = line.strip_prefix("M106 S") {
                if let Some(num_str) = rest.split_whitespace().next() {
                    if let Ok(pwm) = num_str.parse::<u8>() {
                        if !m106_values.contains(&pwm) {
                            m106_values.push(pwm);
                        }
                    }
                }
            }
        }
        assert!(
            m106_values.len() >= 2,
            "Expected dynamic cooling fan to emit at least 2 distinct M106 PWM levels, got {m106_values:?}"
        );
    }
}
