use bevy::prelude::*;

use crate::bridge::ffi::ObjectToolpathsResult;
use crate::config::{GcodeProcessConfig, SlicerConfig, WallPrintOrder};

/// Classification of each planned G-code segment for kinematics, emission, and 3D visualization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum PlannedMoveKind {
    OuterWall = 0,
    InnerWall = 1,
    SolidInfill = 2,
    GapFill = 3,
    SparseInfill = 4,
    ScarfSeam = 5,
    Travel = 6,
    ZHop = 7,
    Wipe = 8,
}

impl PlannedMoveKind {
    pub fn from_feature_kind(feature_kind: u8) -> Self {
        match feature_kind {
            0 => Self::OuterWall,
            1 => Self::InnerWall,
            2 => Self::SolidInfill,
            3 => Self::GapFill,
            4 => Self::SparseInfill,
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
                | Self::ScarfSeam
        )
    }

    pub fn is_wall(self) -> bool {
        matches!(self, Self::OuterWall | Self::InnerWall | Self::ScarfSeam)
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::OuterWall => "Outer Wall",
            Self::InnerWall => "Inner Wall",
            Self::SolidInfill => "Solid Infill",
            Self::GapFill => "Gap Fill",
            Self::SparseInfill => "Sparse Infill",
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
    pub kind: PlannedMoveKind,
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
    fn step_extrusion(
        &mut self,
        q_target_mm3s: f32,
        duration_s: f32,
        tau_pa_s: f32,
        rate_mult: f32,
        cfg: &GcodeProcessConfig,
    ) -> (f32, f32) {
        let k = tau_pa_s.max(0.005);
        let dt = duration_s.max(1.0e-4);
        let x = (dt / k).clamp(0.0, 50.0);
        let i_x = Self::integral_decay_ratio(x);
        let p_avg = q_target_mm3s + (self.pressure_mm3s - q_target_mm3s) * i_x;

        let mut mult = 1.0f32;
        if cfg.enable_transient_pressure_comp && q_target_mm3s > 1.0e-3 {
            if p_avg > q_target_mm3s {
                let ratio = (q_target_mm3s / p_avg).clamp(0.05, 1.0);
                let min_m = cfg.transient_min_flow_multiplier.clamp(0.1, 1.0);
                mult = ratio
                    .powf(cfg.transient_pressure_exponent.clamp(0.05, 1.0))
                    .clamp(min_m, 1.0);
            }
            if dt < 0.25 * k {
                self.short_move_streak = (self.short_move_streak + 1).min(8);
                let streak_t = (self.short_move_streak as f32 / 4.0).clamp(0.0, 1.0);
                let m_short = 1.0 - (1.0 - cfg.transient_min_flow_multiplier) * 0.5 * streak_t;
                mult = mult.min(m_short);
            } else {
                self.short_move_streak = 0;
            }
        }

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
    fn step_travel(&mut self, duration_s: f32, tau_pa_s: f32) -> f32 {
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
            // Preserve vertices where width or nozzle orientation changes noticeably
            let width_diff = (pt.width_mm - 0.5 * (pts[i0].width_mm + pts[i1].width_mm)).abs();
            if d_sq > max_d_sq && width_diff <= 0.025 {
                max_d_sq = d_sq;
                split_idx = i;
            } else if width_diff > 0.025 {
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
    let is_outer = chain.base_kind == PlannedMoveKind::OuterWall;
    let is_closed = chain.is_closed;

    let mut shifts = vec![Vec3::ZERO; n];
    for i in 0..n {
        let curr = &chain.pts[i];
        // Safe tangent calculation: wrap around ONLY when `is_closed == true`!
        let p_prev = if i > 0 {
            chain.pts[i - 1].pos_bed
        } else if is_closed {
            chain.pts[n - 1].pos_bed
        } else {
            curr.pos_bed
        };
        let p_next = if i + 1 < n {
            chain.pts[i + 1].pos_bed
        } else if is_closed {
            chain.pts[0].pos_bed
        } else {
            curr.pos_bed
        };

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

        // 2. Outer wall centerline pinning for variable bead width + viscoelastic die swell inset
        if is_outer {
            // Right-handed inward normal in the layer tangent plane (left of travel tangent)
            let inward_dir = z_nozzle.cross(tangent).try_normalize().unwrap_or(Vec3::Y);
            let max_pin = (0.4 * nominal_outer_w).max(0.0);
            let width_pin_shift =
                (0.5 * (curr.width_mm - nominal_outer_w)).clamp(-max_pin, max_pin);
            let est_q = curr.width_mm * curr.height_mm * nominal_speed_mms;
            let swell_ratio = fluid.die_swell_ratio(est_q);
            let swell_inset = 0.5 * (swell_ratio - 1.0) * curr.height_mm;
            delta += inward_dir * (width_pin_shift + swell_inset);
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

/// Orders chains on a layer according to `WallPrintOrder` (`InnerOuterInner` or `OutsideIn`)
/// followed by greedy Z-penalized nearest-neighbor traversal of remaining open/closed paths.
fn order_layer_chains(
    mut chains: Vec<WorkingChain>,
    wall_order: WallPrintOrder,
    start_pos_bed: Vec3,
    z_penalty: f32,
) -> Vec<WorkingChain> {
    if chains.is_empty() {
        return chains;
    }

    let mut walls: Vec<WorkingChain> = Vec::new();
    let mut infills: Vec<WorkingChain> = Vec::new();
    for ch in chains.drain(..) {
        if ch.base_kind.is_wall() {
            walls.push(ch);
        } else {
            infills.push(ch);
        }
    }

    // Sort walls according to `WallPrintOrder`
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

    let mut ordered: Vec<WorkingChain> = Vec::with_capacity(walls.len() + infills.len());
    let mut curr_pos = start_pos_bed;

    // Group walls by rank and greedily pick nearest within each rank
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

    // Greedily order remaining infill & gap-fill chains
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

    // 1. Closed wall loop with 3D Scarf Joint Seam Ramping (strictly gated on `is_closed`)
    if chain.is_closed
        && chain.base_kind.is_wall()
        && cfg.enable_scarf_joint
        && total_len >= 2.5 * cfg.scarf_length_mm
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

        // Forward wipe along closed loop
        if cfg.wipe_length_mm > 1.0e-3 {
            let mut w_v = sample_closed_loop_at_arc_length(&chain.pts, l_scarf + cfg.wipe_length_mm);
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

        // Forward wipe along p0 -> p1 on closed loop
        if cfg.wipe_length_mm > 1.0e-3 && chain.pts.len() >= 2 {
            let mut w_v = sample_closed_loop_at_arc_length(&chain.pts, cfg.wipe_length_mm);
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
    if cfg.pre_retract_taper_mm > 1.0e-3 && total_len >= 2.0 * cfg.pre_retract_taper_mm {
        let n = out.len();
        let last_idx = n - 1;
        out[last_idx].flow_rate_mult = 0.65;
    }

    if cfg.wipe_length_mm > 1.0e-3 && chain.base_kind.is_wall() && out.len() >= 2 {
        let p_end = out[out.len() - 1].pos_bed;
        let p_prev = out[out.len() - 2].pos_bed;
        let seg_vec = p_prev - p_end; // BACKWARD along the last segment!
        let seg_len = seg_vec.length();
        if seg_len > 1.0e-4 {
            let wipe_d = cfg.wipe_length_mm.min(0.85 * seg_len);
            let mut wipe_v = out[out.len() - 1].clone();
            wipe_v.pos_bed = p_end + (seg_vec / seg_len) * wipe_d;
            wipe_v.flow_rate_mult = 0.0;
            wipe_v.kind_override = Some(PlannedMoveKind::Wipe);
            out.push(wipe_v);
        }
    }

    out
}

/// Subdivides long non-planar segments (> 2.5 mm) where nozzle orientation or Z changes,
/// ensuring downstream flow and inclination math accurately tracks curved surfaces.
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
        let n_dot = a.nozzle_bed.dot(b.nozzle_bed);
        let h_diff = (b.height_mm - a.height_mm).abs();

        if len > 2.5 && (len > 6.0 || n_dot < 0.998 || h_diff > 0.02) {
            let steps = ((len / 2.5).ceil() as usize).clamp(2, 16);
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
    if !cfg.enable_slicer_pa_subdivision {
        return segments;
    }

    let mut out = Vec::with_capacity(segments.len() * 2);
    for seg in segments {
        if !seg.kind.is_extruding()
            || seg.delta_e_mm <= 1.0e-5
            || seg.length_mm < 1.2
            || (seg.cruise_speed_mms - seg.entry_speed_mms.min(seg.exit_speed_mms)) < 8.0
        {
            out.push(seg);
            continue;
        }

        let n_sub = ((seg.length_mm / 0.8).ceil() as usize).clamp(2, 6);
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
            sub.length_mm = sub_len;
            sub.entry_speed_mms = v0;
            sub.cruise_speed_mms = v_mid;
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

/// Computes a compact 2D convex hull polygon `[[x0,y0],[x1,y1],...,[x0,y0]]` in G-code bed
/// coordinates (`mm`) for Klipper's `EXCLUDE_OBJECT_DEFINE POLYGON=...` parameter.
pub fn compute_exclude_object_polygon_2d(
    layers: &[PlannedLayerGcode],
    bbox_min: Vec3,
    bbox_max: Vec3,
) -> Vec<[f32; 2]> {
    #[inline]
    fn cross_2d(o: [f32; 2], a: [f32; 2], b: [f32; 2]) -> f32 {
        (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    }

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
        let x0 = bbox_min.x;
        let y0 = bbox_min.y;
        let x1 = bbox_max.x.max(x0 + 0.1);
        let y1 = bbox_max.y.max(y0 + 0.1);
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

/// Formats a Klipper `EXCLUDE_OBJECT_DEFINE` G-code line with zero internal spaces inside
/// `CENTER=` and `POLYGON=` so Klipper's whitespace-splitting command parser handles it cleanly.
pub fn format_exclude_object_define(name: &str, center: [f32; 2], polygon: &[[f32; 2]]) -> String {
    let mut poly_json = String::with_capacity(polygon.len() * 18 + 4);
    poly_json.push('[');
    for (i, pt) in polygon.iter().enumerate() {
        if i > 0 {
            poly_json.push(',');
        }
        poly_json.push_str(&format!("[{:.3},{:.3}]", pt[0], pt[1]));
    }
    poly_json.push(']');
    format!(
        "EXCLUDE_OBJECT_DEFINE NAME={} CENTER={:.3},{:.3} POLYGON={}",
        name, center[0], center[1], poly_json
    )
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

    let active_tool = config.active_tool();
    let tool_id = active_tool.map(|t| t.id).unwrap_or(0);
    let nozzle_diam_mm = active_tool
        .map(|t| t.nozzle_diameter_mm)
        .unwrap_or(0.40)
        .max(0.10);
    let nozzle_temp_c = active_tool
        .map(|t| t.temperature_celsius)
        .unwrap_or(215.0);

    // Query gravity direction in bed coordinates (AGENTS.md Rule 2: never hardcode -Z!)
    let gravity_bed = config.query_gravity_bed(Vec3::ZERO);
    let up_bed = (-gravity_bed).try_normalize().unwrap_or(Vec3::Z);

    // Viewport scene places (0, 0) at the center of the build plate (`[-width_x/2 .. +width_x/2]`),
    // whereas rectangular printers expect G-code (0, 0) at the front-left corner (`[0 .. width_x]`).
    let bed_origin_offset = if gcfg.gcode_origin_at_bed_center
        || config.bed.shape == crate::config::BedShape::Circular
    {
        Vec3::ZERO
    } else {
        Vec3::new(0.5 * config.bed.width_x, 0.5 * config.bed.depth_y, 0.0)
    };

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
    let simplify_tol_mm = 0.04 * nozzle_diam_mm;
    let min_retract_travel_mm = gcfg.effective_min_travel_for_retract();

    let mut planned_layers: Vec<PlannedLayerGcode> = Vec::with_capacity(raw_toolpaths.layers.len());
    let mut curr_nozzle_bed: Option<Vec3> = None;
    let mut curr_nozzle_axis_bed = Vec3::from_array(config.machine.nozzle_axis_bed)
        .try_normalize()
        .unwrap_or(up_bed);

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
            simplify_working_chain(&mut wc, simplify_tol_mm);
            let nom_spd = if is_first_layer {
                gcfg.first_layer_speed_mms
            } else {
                gcfg.outer_wall_speed_mms
            };
            compensate_wall_geometry(
                &mut wc,
                nozzle_diam_mm,
                config.slicing.outer_wall_width.target_mm,
                up_bed,
                &fluid,
                nom_spd,
                gcfg.enable_flat_nozzle_comp,
            );
            chains.push(wc);
        }

        let start_ref = curr_nozzle_bed.unwrap_or(bed_origin_offset);
        let ordered_chains = order_layer_chains(
            chains,
            gcfg.wall_print_order,
            start_ref,
            gcfg.z_travel_penalty,
        );

        let mut layer_segments: Vec<PlannedSegment> = Vec::new();

        for chain in &ordered_chains {
            let vertices = subdivide_long_segments(finalize_chain_seam_and_wipe(chain, &gcfg));
            if vertices.len() < 2 {
                continue;
            }

            let chain_start_bed = vertices[0].pos_bed;
            let chain_start_nozzle = vertices[0].nozzle_bed;

            // Emit inter-path travel + conditional Z-hop along local nozzle axis (`z_nozzle`)
            if let Some(prev_bed) = curr_nozzle_bed {
                let travel_vec = chain_start_bed - prev_bed;
                let travel_dist = travel_vec.length();
                if travel_dist > 1.0e-3 {
                    let needs_retract = travel_dist > min_retract_travel_mm;
                    let is_infill_to_infill = matches!(
                        chain.base_kind,
                        PlannedMoveKind::SolidInfill | PlannedMoveKind::SparseInfill
                    ) && layer_segments.last().is_some_and(|s| {
                        matches!(
                            s.kind,
                            PlannedMoveKind::SolidInfill | PlannedMoveKind::SparseInfill
                        )
                    });
                    let use_z_hop =
                        needs_retract && gcfg.z_hop_mm > 1.0e-3 && !is_infill_to_infill;

                    if use_z_hop {
                        let lift_dir = curr_nozzle_axis_bed.try_normalize().unwrap_or(up_bed);
                        let drop_dir = chain_start_nozzle.try_normalize().unwrap_or(up_bed);
                        let p_lift = prev_bed + lift_dir * gcfg.z_hop_mm;
                        let p_above = chain_start_bed + drop_dir * gcfg.z_hop_mm;

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
                            kind: PlannedMoveKind::ZHop,
                            wall_index: 0,
                            width_mm: nozzle_diam_mm * 0.35,
                            height_mm: gcfg.z_hop_mm,
                            length_mm: gcfg.z_hop_mm,
                            nominal_speed_mms: gcfg.z_hop_speed_mms,
                            entry_speed_mms: 0.0,
                            cruise_speed_mms: gcfg.z_hop_speed_mms,
                            exit_speed_mms: 0.0,
                            accel_mms2: gcfg.travel_accel_mms2,
                            duration_s: (gcfg.z_hop_mm / gcfg.z_hop_speed_mms.max(1.0)).max(1.0e-4),
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
                            kind: PlannedMoveKind::Travel,
                            wall_index: 0,
                            width_mm: nozzle_diam_mm * 0.35,
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
                            kind: PlannedMoveKind::ZHop,
                            wall_index: 0,
                            width_mm: nozzle_diam_mm * 0.35,
                            height_mm: gcfg.z_hop_mm,
                            length_mm: gcfg.z_hop_mm,
                            nominal_speed_mms: gcfg.z_hop_speed_mms,
                            entry_speed_mms: 0.0,
                            cruise_speed_mms: gcfg.z_hop_speed_mms,
                            exit_speed_mms: 0.0,
                            accel_mms2: gcfg.travel_accel_mms2,
                            duration_s: (gcfg.z_hop_mm / gcfg.z_hop_speed_mms.max(1.0)).max(1.0e-4),
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
                            kind: PlannedMoveKind::Travel,
                            wall_index: 0,
                            width_mm: nozzle_diam_mm * 0.35,
                            height_mm: 0.15,
                            length_mm: travel_dist,
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
                    PlannedMoveKind::OuterWall | PlannedMoveKind::ScarfSeam => {
                        (gcfg.outer_wall_speed_mms, gcfg.outer_wall_accel_mms2)
                    }
                    PlannedMoveKind::InnerWall => {
                        (gcfg.inner_wall_speed_mms, gcfg.inner_wall_accel_mms2)
                    }
                    PlannedMoveKind::SolidInfill => {
                        (gcfg.solid_infill_speed_mms, gcfg.infill_accel_mms2)
                    }
                    PlannedMoveKind::GapFill => (gcfg.gap_fill_speed_mms, gcfg.inner_wall_accel_mms2),
                    PlannedMoveKind::SparseInfill => {
                        (gcfg.sparse_infill_speed_mms, gcfg.infill_accel_mms2)
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

                    // Bead cross-section area A_bead (blending rectangular first layer & stadium upper layers)
                    let w_eff = width.max(height);
                    let a_stadium =
                        height * (w_eff - height) + std::f32::consts::FRAC_PI_4 * height * height;
                    let a_rect = w_eff * height;
                    let a_bead = if is_first_layer { a_rect } else { a_stadium };

                    // Surface inclination & trajectory climb cosine along local `z_nozzle` and `up_bed`
                    let c_surf = if is_first_layer {
                        1.0
                    } else {
                        z_nozzle.dot(up_bed).abs().clamp(0.25, 1.0)
                    };
                    let climb_nozzle = seg_dir.dot(z_nozzle);
                    let c_traj = (1.0 - climb_nozzle * climb_nozzle).max(0.15).sqrt();
                    let c_slope = c_surf.min(c_traj).clamp(0.25, 1.0);

                    // Directional downhill squeeze multiplier
                    let s_climb = seg_dir.dot(up_bed);
                    let m_dir = if s_climb < 0.0 {
                        (1.0 - 0.12 * (-s_climb).clamp(0.0, 1.0)).max(0.75)
                    } else {
                        1.0
                    };

                    // Volumetric speed limit v_clamped <= Q_max / A_bead
                    let v_vol_max = gcfg.max_volumetric_flow_mm3s / a_bead.max(1.0e-4);
                    let v_clamped = target_spd.min(v_vol_max).max(5.0);
                    let q_est = a_bead * v_clamped * rate_mult;
                    let m_swell = fluid.swell_volume_multiplier(q_est);
                    let m_first = if is_first_layer {
                        gcfg.first_layer_flow_multiplier
                    } else {
                        1.0
                    };

                    let composite_mult =
                        rate_mult * gcfg.extrusion_multiplier * m_first * c_slope * m_dir * m_swell;
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
                        kind,
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
                        pressure_advance_s: fluid.pressure_advance_s(
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
                        kind: PlannedMoveKind::Wipe,
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
            //    Strictly on interior junctions of this chain (`chain_seg_start .. layer_segments.len()`),
            //    NEVER wrapping around open paths!
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
                    if cos_alpha >= 0.9995 {
                        continue;
                    }

                    let alpha = cos_alpha.acos();
                    let sin_half = (0.5 * (1.0 - cos_alpha)).max(0.0).sqrt();
                    let cos_half = (0.5 * (1.0 + cos_alpha)).max(0.125).sqrt();
                    let tan_half = (sin_half / cos_half).min(4.0);

                    let w_avg = 0.5 * (s_in.width_mm + s_out.width_mm);
                    let h_avg = 0.5 * (s_in.height_mm + s_out.height_mm);

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
                    let max_ded = 0.35 * nom_vol;
                    let actual_ded = ded_v.min(max_ded);
                    if nom_vol > 1.0e-5 {
                        let factor = ((nom_vol - actual_ded) / nom_vol).clamp(0.65, 1.0);
                        seg.delta_e_mm *= factor;
                        seg.extrusion_multiplier *= factor;
                    }
                }
            }

            if let Some(last_v) = vertices.last() {
                curr_nozzle_bed = Some(last_v.pos_bed);
                curr_nozzle_axis_bed = last_v.nozzle_bed;
            }
        }

        // 4. Chained Cross-Path Velocity Lookahead on this layer
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
                kind: PlannedMoveKind::Wipe,
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
                kind: PlannedMoveKind::ZHop,
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
            if seg.retract_before_mm > 0.0 {
                seg.retract_before_mm = fluid.adaptive_retract_length_mm(
                    last_q_mm3s,
                    last_pa_s,
                    filament_area_mm2,
                    gcfg.retract_length_mm,
                );
            }
            if seg.kind.is_extruding() && seg.delta_e_mm > 0.0 {
                let q_nom = (seg.delta_e_mm * filament_area_mm2) / seg.duration_s.max(1.0e-4);
                let pa_s = fluid.pressure_advance_s(q_nom, gcfg.enable_dynamic_pa, gcfg.pa_low_flow_s);
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
                seg.pressure_advance_s = fluid.pressure_advance_s(
                    seg.volumetric_flow_mm3s,
                    gcfg.enable_dynamic_pa,
                    gcfg.pa_low_flow_s,
                );
                seg.melt_pressure_mm3s = p_avg;
                last_q_mm3s = seg.volumetric_flow_mm3s;
                last_pa_s = seg.pressure_advance_s;
            } else {
                let p_avg = pressure_tracker.step_travel(seg.duration_s, last_pa_s);
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
    if gcfg.enable_slicer_pa_subdivision {
        gcode.push_str("SET_PRESSURE_ADVANCE ADVANCE=0.0000 ; Slicer-side PA active\n");
    }

    let mut emitted_accel = -1.0f32;
    let mut emitted_feed_mm_min = -1.0f32;
    let mut emitted_pa = -1.0f32;
    let mut last_emitted_kind: Option<PlannedMoveKind> = None;
    let mut active_retract_mm = 0.0f32;
    let mut accum_travel_time_s = 0.0f32;

    for layer in &planned_layers {
        if layer.segments.is_empty() {
            continue;
        }
        gcode.push_str(&format!(
            "\n; --- LAYER {} (Branch {}, Iso {:.3}) ---\n",
            layer.layer_index, layer.branch_id, layer.isovalue
        ));
        if gcfg.emit_exclude_object {
            gcode.push_str(&format!("EXCLUDE_OBJECT_START NAME={exclude_object_name}\n"));
        }

        for seg in &layer.segments {
            total_time_s += seg.duration_s;

            // Emit retraction before non-extruding travel if scheduled
            if seg.retract_before_mm > 1.0e-4 && active_retract_mm <= 1.0e-4 {
                let r_len = seg.retract_before_mm;
                let r_feed = (gcfg.retract_speed_mms * 60.0).max(60.0);
                gcode.push_str(&format!("G1 E-{r_len:.5} F{r_feed:.0} ; Retract\n"));
                active_retract_mm = r_len;
                accum_travel_time_s = 0.0;
                emitted_feed_mm_min = r_feed;
                retract_count += 1;
                total_time_s += r_len / gcfg.retract_speed_mms.max(1.0);
            }

            if !seg.kind.is_extruding() || seg.delta_e_mm <= 1.0e-6 {
                accum_travel_time_s += seg.duration_s;
            } else if active_retract_mm > 1.0e-4 {
                // Unretract before resuming extrusion (with thermal ooze / swell relaxation)
                let u_len = fluid.unretract_length_mm(active_retract_mm, accum_travel_time_s, &gcfg);
                let u_feed = (gcfg.retract_speed_mms * 60.0).max(60.0);
                gcode.push_str(&format!("G1 E{u_len:.5} F{u_feed:.0} ; Unretract\n"));
                active_retract_mm = 0.0;
                accum_travel_time_s = 0.0;
                emitted_feed_mm_min = u_feed;
                total_time_s += u_len / gcfg.retract_speed_mms.max(1.0);
            }

            // Deadband-compacted `SET_VELOCITY_LIMIT ACCEL=...`
            let kind_changed = last_emitted_kind != Some(seg.kind);
            if kind_changed
                || emitted_accel <= 0.0
                || ((seg.accel_mms2 - emitted_accel).abs() / emitted_accel.max(1.0)) >= 0.15
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

                // Emit dynamic firmware PA update when outside 8% deadband
                if !gcfg.enable_slicer_pa_subdivision
                    && gcfg.enable_dynamic_pa
                    && (emitted_pa < 0.0
                        || (seg.pressure_advance_s - emitted_pa).abs()
                            / emitted_pa.max(0.005)
                            >= 0.08)
                {
                    gcode.push_str(&format!(
                        "SET_PRESSURE_ADVANCE ADVANCE={:.4}\n",
                        seg.pressure_advance_s
                    ));
                    emitted_pa = seg.pressure_advance_s;
                }

                let feed_mm_min = (seg.cruise_speed_mms * 60.0).max(30.0);
                let emit_f = kind_changed
                    || emitted_feed_mm_min <= 0.0
                    || ((feed_mm_min - emitted_feed_mm_min).abs() / emitted_feed_mm_min.max(1.0))
                        >= 0.05;

                if emit_f {
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
                let emit_f = kind_changed
                    || emitted_feed_mm_min <= 0.0
                    || ((feed_mm_min - emitted_feed_mm_min).abs() / emitted_feed_mm_min.max(1.0))
                        >= 0.05;

                if emit_f {
                    gcode.push_str(&format!(
                        "G0 X{:.3} Y{:.3} Z{:.3} F{:.0}\n",
                        seg.p1_bed.x, seg.p1_bed.y, seg.p1_bed.z, feed_mm_min
                    ));
                    emitted_feed_mm_min = feed_mm_min;
                } else {
                    gcode.push_str(&format!(
                        "G0 X{:.3} Y{:.3} Z{:.3}\n",
                        seg.p1_bed.x, seg.p1_bed.y, seg.p1_bed.z
                    ));
                }
            }

            last_emitted_kind = Some(seg.kind);
        }

        if gcfg.emit_exclude_object {
            gcode.push_str(&format!("EXCLUDE_OBJECT_END NAME={exclude_object_name}\n"));
        }
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
    // Standard PLA density ~1.24 g/cm³ = 0.00124 g/mm³
    let total_filament_grams = total_filament_volume_mm3 * 0.00124;

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
            .starts_with("EXCLUDE_OBJECT_DEFINE NAME=My_Part_v1 CENTER="));
        assert!(res.gcode_text.contains("POLYGON=[["));
        assert!(res
            .gcode_text
            .contains("EXCLUDE_OBJECT_START NAME=My_Part_v1\n"));
        assert!(res
            .gcode_text
            .contains("EXCLUDE_OBJECT_END NAME=My_Part_v1\n"));
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
}
