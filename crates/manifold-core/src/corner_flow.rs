//! Kinematic and geometric corner overlap flow compensation.
//!
//! When two extruded tracks of line width $w$ and layer height $h$ meet at a corner
//! with in-surface turning angle $\alpha \in (0, \pi)$ (interior angle $\theta = \pi - \alpha$):
//!
//! 1. **Geometric Inner-Corner Overlap**:
//!    The inner boundaries of the two tracks intersect, creating a triangular/rhombus
//!    overlap footprint of area:
//!    $$\Delta A_{\text{overlap}} = \frac{w^2}{4} \cot\left(\frac{\theta}{2}\right) = \frac{w^2}{4} \tan\left(\frac{\alpha}{2}\right)$$
//!    This deposits redundant volume on the inside of the turn:
//!    $$\Delta V_{\text{geom}} = \frac{w^2 h}{4} \tan\left(\frac{\alpha}{2}\right)$$
//!
//! 2. **Klipper SCV Shortcut Arc & Kinematic Path Shortening**:
//!    Under Klipper's Square Corner Velocity (SCV) model, the toolhead corners at
//!    velocity $v_{\text{corner}} = \text{klipper\_corner\_velocity}(\hat{\mathbf{d}}_1, \hat{\mathbf{d}}_2, \text{scv}, a)$.
//!    Instead of tracking the programmed sharp corner, the toolhead rounds the vertex
//!    along a smooth trajectory of effective radius:
//!    $$R_{\text{eff}} = \frac{v_{\text{corner}}^2}{a \sin(\alpha/2)}$$
//!    shortcutting distance $s_{\text{tangent}} = R_{\text{eff}} \tan(\alpha/2)$ on each leg.
//!    The actual nozzle path length along the rounding arc is $L_{\text{arc}} = R_{\text{eff}} \cdot \alpha$,
//!    yielding kinematic path shortening:
//!    $$\Delta L_{\text{kinematic}} = 2 s_{\text{tangent}} - R_{\text{eff}} \alpha = R_{\text{eff}} \left(2 \tan\left(\frac{\alpha}{2}\right) - \alpha\right)$$
//!    Commanding full nominal extrusion over the straight distance deposits redundant volume:
//!    $$\Delta V_{\text{kinematic}} = \Delta L_{\text{kinematic}} \cdot A_{\text{bead}}$$
//!
//! In non-planar slicing, to prevent straight lines draping over curved hills or arches
//! from being falsely detected as corners, directions $\hat{\mathbf{d}}_1$ and $\hat{\mathbf{d}}_2$
//! are projected onto the local layer tangent plane orthogonal to the layer surface normal $\hat{\mathbf{n}}$
//! before evaluating the in-surface turning angle $\alpha$.

use glam::DVec3;
use manifold_fidget::order::OrderField;

use crate::{
    kinematics::klipper_corner_velocity,
    machine::Machine,
    toolpath::{MoveKind, Path},
    SlicerConfig,
};

/// Calculated excess volume breakdown at a corner junction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CornerExcess {
    /// Geometric inner-corner overlap volume ($\text{mm}^3$).
    pub v_geom: f64,
    /// Kinematic SCV path-shortening volume ($\text{mm}^3$).
    pub v_kinematic: f64,
    /// Total excess volume ($\text{mm}^3$).
    pub v_total: f64,
    /// In-surface turning angle $\alpha$ (radians).
    pub turning_angle: f64,
    /// Effective rounding arc radius $R_{\text{eff}}$ (mm).
    pub radius_eff: f64,
}

/// Evaluates the excess volume at a corner junction between incoming segment
/// (`p_prev -> p_corner`) and outgoing segment (`p_corner -> p_next`).
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn calculate_corner_excess(
    p_prev: DVec3,
    p_corner: DVec3,
    p_next: DVec3,
    line_width: f64,
    layer_height: f64,
    order_field: Option<&dyn OrderField>,
    scv: f64,
    accel: f64,
) -> Option<CornerExcess> {
    let v_in = p_corner - p_prev;
    let v_out = p_next - p_corner;
    let d_in = v_in.length();
    let d_out = v_out.length();
    if d_in < 1e-6 || d_out < 1e-6 {
        return None;
    }
    let dir_in = v_in / d_in;
    let dir_out = v_out / d_out;

    // Evaluate local layer surface normal
    let surface_normal = if let Some(field) = order_field {
        crate::order_field::numeric_gradient(field, p_corner)
            .and_then(|g| g.try_normalize())
            .unwrap_or(DVec3::Z)
    } else {
        DVec3::Z
    };

    // Project directions onto the local layer tangent plane
    let t_in = dir_in - surface_normal * dir_in.dot(surface_normal);
    let t_out = dir_out - surface_normal * dir_out.dot(surface_normal);

    let (u_in, u_out) = match (t_in.try_normalize(), t_out.try_normalize()) {
        (Some(u1), Some(u2)) => (u1, u2),
        _ => {
            // Fallback to XY projection if tangent projection degenerates
            let xy_in = DVec3::new(dir_in.x, dir_in.y, 0.0);
            let xy_out = DVec3::new(dir_out.x, dir_out.y, 0.0);
            match (xy_in.try_normalize(), xy_out.try_normalize()) {
                (Some(u1), Some(u2)) => (u1, u2),
                _ => return None,
            }
        }
    };

    let cos_alpha = u_in.dot(u_out).clamp(-1.0, 1.0);

    // If turning angle is negligible (< ~1.15°), skip corner compensation
    if cos_alpha >= 0.9998 {
        return None;
    }

    let sin_half = ((1.0 - cos_alpha) * 0.5).max(0.0).sqrt();
    let cos_half = ((1.0 + cos_alpha) * 0.5).max(0.0).sqrt();
    let turning_angle = sin_half.atan2(cos_half) * 2.0; // in (0, pi]

    // tan(alpha/2) with safety ceiling to avoid infinity on sharp 180° reversals
    let tan_half = (sin_half / cos_half.max(0.125)).min(4.0);

    let w = line_width.max(1e-4);
    let h = layer_height.max(1e-4);

    // 1. Geometric inner-corner overlap volume: ΔV_geom = (w² h / 4) * tan(α/2)
    let v_geom = (w * w * h * 0.25) * tan_half;

    // 2. Kinematic SCV path-shortening volume
    let v_corner = klipper_corner_velocity(dir_in, dir_out, scv, accel);
    let (v_kinematic, radius_eff) = if v_corner > 0.05 && accel > 1.0 && sin_half > 1e-4 {
        let unconstrained_r = (v_corner * v_corner) / (accel * sin_half);
        // Tangent distance along each leg cannot exceed 40% of either segment
        let max_tangent = 0.40 * d_in.min(d_out);
        let max_r = if tan_half > 1e-4 {
            max_tangent / tan_half
        } else {
            0.0
        };
        let r_eff = unconstrained_r.min(max_r);

        // Path shortening: ΔL = r_eff * (2 * tan(α/2) - α)
        let delta_l = (r_eff * (2.0 * tan_half - turning_angle)).max(0.0);
        let bead_area = w * h;
        (delta_l * bead_area, r_eff)
    } else {
        (0.0, 0.0)
    };

    let v_total = v_geom + v_kinematic;

    Some(CornerExcess {
        v_geom,
        v_kinematic,
        v_total,
        turning_angle,
        radius_eff,
    })
}

/// Applies corner overlap and kinematic SCV flow compensation across all toolpath paths.
///
/// Modulates each segment's `extrusion_length` and `extrusion_rate` to deduct redundant
/// volume deposited on the inside of corners and along shortcutting SCV arcs.
pub fn apply_corner_flow_compensation(
    paths: &mut [Path],
    order_field: Option<&dyn OrderField>,
    config: &SlicerConfig,
    machine: Option<&Machine>,
) {
    if !config.enable_corner_flow_compensation || paths.is_empty() {
        return;
    }

    let compensation_ratio = config.corner_flow_compensation_ratio();
    if compensation_ratio <= 1e-4 {
        return;
    }

    let motion_model = config.resolved_motion_model(machine);
    let scv = config.square_corner_velocity();
    let filament_area =
        std::f64::consts::PI * 0.25 * config.filament_diameter * config.filament_diameter;
    let nominal_h = config.layer_height;

    // Safety limit: a segment cannot lose more than 35% of its volume to corner deductions
    const MAX_CORNER_DEDUCTION_FRACTION: f64 = 0.35;

    for path in paths {
        let n_pts = path.points.len();
        let n_segs = path.segments.len();
        if n_pts < 3 || n_segs < 2 {
            continue;
        }

        let is_closed = n_segs == n_pts;
        let is_first_layer = path
            .segments
            .first()
            .is_some_and(|s| s.order <= config.first_layer_height() + 1e-4);
        let layer_h = if is_first_layer {
            config.first_layer_height()
        } else {
            nominal_h
        };

        // Track deductions per segment index
        let mut deductions = vec![0.0; n_segs];

        // Process all corner junctions
        let junction_count = if is_closed { n_pts } else { n_pts - 2 };

        for j in 0..junction_count {
            let (idx_prev, idx_corner, idx_next, seg_in_idx, seg_out_idx) = if is_closed {
                let prev = (j + n_pts - 1) % n_pts;
                let corner = j;
                let next = (j + 1) % n_pts;
                (prev, corner, next, prev, corner)
            } else {
                let corner = j + 1;
                (corner - 1, corner, corner + 1, corner - 1, corner)
            };

            let seg_in = &path.segments[seg_in_idx];
            let seg_out = &path.segments[seg_out_idx];

            // Only compensate junctions between two extruding moves
            let is_extruding = |kind: MoveKind| -> bool {
                kind != MoveKind::Travel && kind != MoveKind::DebugExcluded
            };

            if !is_extruding(seg_in.kind) || !is_extruding(seg_out.kind) {
                continue;
            }
            if seg_in.extrusion_length <= 1e-6 || seg_out.extrusion_length <= 1e-6 {
                continue;
            }

            let p_prev = path.points[idx_prev];
            let p_corner = path.points[idx_corner];
            let p_next = path.points[idx_next];

            let line_w = (seg_in.line_width + seg_out.line_width) * 0.5;
            let dir_in = (p_corner - p_prev).try_normalize().unwrap_or(DVec3::ZERO);
            let accel = motion_model.available_directional_acceleration(
                seg_in.kind,
                is_first_layer,
                seg_in.speed / 60.0,
                dir_in,
            );

            if let Some(excess) = calculate_corner_excess(
                p_prev,
                p_corner,
                p_next,
                line_w,
                layer_h,
                order_field,
                scv,
                accel,
            ) {
                let v_excess = excess.v_total * compensation_ratio;
                // Symmetrically split deduction between incoming and outgoing segments
                let half_v = v_excess * 0.5;
                deductions[seg_in_idx] += half_v;
                deductions[seg_out_idx] += half_v;
            }
        }

        // Apply deductions to segments
        for (i, seg) in path.segments.iter_mut().enumerate() {
            let deduct_v = deductions[i];
            if deduct_v <= 1e-8 {
                continue;
            }

            let p0 = path.points[i];
            let p1 = path.points[(i + 1) % n_pts];
            let seg_d = p0.distance(p1);
            let nominal_v = seg_d * seg.line_width * layer_h;

            let max_deduct = nominal_v * MAX_CORNER_DEDUCTION_FRACTION;
            let clamped_deduct_v = deduct_v.min(max_deduct);

            let delta_filament_length = clamped_deduct_v / filament_area;
            let old_e = seg.extrusion_length;
            let new_e = (old_e - delta_filament_length).max(0.0);

            if old_e > 1e-6 {
                seg.extrusion_rate *= new_e / old_e;
            }
            seg.extrusion_length = new_e;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn straight_line_has_zero_corner_excess() {
        let p_prev = DVec3::new(0.0, 0.0, 0.0);
        let p_corner = DVec3::new(10.0, 0.0, 0.0);
        let p_next = DVec3::new(20.0, 0.0, 0.0);

        let excess =
            calculate_corner_excess(p_prev, p_corner, p_next, 0.40, 0.20, None, 5.0, 5000.0);
        assert!(excess.is_none());
    }

    #[test]
    fn right_angle_corner_evaluates_expected_geometric_overlap() {
        let p_prev = DVec3::new(0.0, 0.0, 0.0);
        let p_corner = DVec3::new(10.0, 0.0, 0.0);
        let p_next = DVec3::new(10.0, 10.0, 0.0);

        let excess =
            calculate_corner_excess(p_prev, p_corner, p_next, 0.40, 0.20, None, 0.0, 5000.0)
                .expect("90-degree corner must have excess");

        // For 90° turn: alpha = pi/2, tan(alpha/2) = 1.0
        // Delta V_geom = w^2 * h / 4 * 1.0 = 0.16 * 0.20 / 4 = 0.008 mm³
        assert!((excess.v_geom - 0.008).abs() < 1e-5);
        assert_eq!(excess.v_kinematic, 0.0); // SCV = 0 disables kinematic shortening
        assert!((excess.v_total - 0.008).abs() < 1e-5);
    }

    #[test]
    fn scv_kinematics_adds_path_shortening_to_corner_excess() {
        let p_prev = DVec3::new(0.0, 0.0, 0.0);
        let p_corner = DVec3::new(10.0, 0.0, 0.0);
        let p_next = DVec3::new(10.0, 10.0, 0.0);

        let excess_static =
            calculate_corner_excess(p_prev, p_corner, p_next, 0.40, 0.20, None, 0.0, 5000.0)
                .unwrap();

        let excess_dynamic =
            calculate_corner_excess(p_prev, p_corner, p_next, 0.40, 0.20, None, 5.0, 5000.0)
                .unwrap();

        assert!(excess_dynamic.v_kinematic > 0.0);
        assert!(excess_dynamic.v_total > excess_static.v_total);
    }

    #[test]
    fn straight_line_over_sloped_surface_has_zero_in_surface_corner_excess() {
        // A path climbing over a 45-degree sloped ridge:
        // Chord 1 climbs at +45°, Chord 2 descends at -45°
        let p_prev = DVec3::new(0.0, 0.0, 0.0);
        let p_corner = DVec3::new(10.0, 0.0, 10.0);
        let p_next = DVec3::new(20.0, 0.0, 0.0);

        // Ridge apex order surface normal tilts with the ridge (pointing up in XZ plane)
        // Normal at apex: DVec3::new(0.0, 0.0, 1.0)
        // Notice in XY projection (and transverse to normal), path travels straight along X (no lateral turn)!
        let excess =
            calculate_corner_excess(p_prev, p_corner, p_next, 0.40, 0.20, None, 5.0, 5000.0);

        // In XY / lateral projection, direction does not turn (u_in = (1,0,0), u_out = (1,0,0))
        assert!(excess.is_none());
    }

    #[test]
    fn apply_corner_flow_compensation_deducts_volume_from_rectangle_loop() {
        let path = Path {
            points: vec![
                DVec3::new(0.0, 0.0, 0.0),
                DVec3::new(10.0, 0.0, 0.0),
                DVec3::new(10.0, 10.0, 0.0),
                DVec3::new(0.0, 10.0, 0.0),
            ],
            segments: vec![
                crate::toolpath::Segment {
                    kind: MoveKind::WallOuter,
                    speed: 60.0 * 50.0,
                    extrusion_rate: 1.0,
                    support_fraction: 1.0,
                    order: 0.20,
                    extrusion_length: 0.50,
                    line_width: 0.40,
                    is_scarf: false,
                    id: 0,
                    island: 0,
                    channel_width: f64::INFINITY,
                };
                4
            ],
            tool: crate::ids::ToolId(0),
        };

        let config = SlicerConfig::default();
        let initial_e = path.segments[0].extrusion_length;

        apply_corner_flow_compensation(&mut [path.clone()], None, &config, None);

        // Segment extrusion length should be reduced due to the four 90-degree corners
        let mut paths = vec![path];
        apply_corner_flow_compensation(&mut paths, None, &config, None);

        for seg in &paths[0].segments {
            assert!(seg.extrusion_length < initial_e);
            assert!(seg.extrusion_length > 0.0);
        }
    }
}
