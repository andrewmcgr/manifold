//! Gap-fill toolpath generation for complete shell fill on sloped non-planar layers.
//!
//! On non-planar surfaces, the in-surface spacing between adjacent wall passes
//! expands according to the local contact angle:
//!
//! $$\Delta s = \frac{\text{nominal\_width}}{\|\hat{n}_{\text{CAD}} \times \hat{n}_{\text{order}}\|}$$
//!
//! When $\Delta s > w_{\max}$ ($1.6 \times \text{nozzle\_diameter}$), a single inner wall bead
//! cannot span the distance without dragging or sagging. This module computes intermediate
//! gap-fill toolpaths along the 3D in-surface centerline between adjacent wall passes (and between
//! the innermost wall and the infill boundary), halving the required bead width to
//! $\Delta s / 2 \in [w_{\min}, w_{\max}]$ and achieving 100% solid shell fill.

use crate::ids::ToolId;
use crate::slicing::{Layer, WallLoop};
use crate::toolpath::{speed_for_kind, MoveKind, Path, Segment};
use crate::SlicerConfig;
use glam::DVec3;
use manifold_fidget::ScalarField;

/// Computed gap-fill point along the in-surface centerline.
#[derive(Debug, Clone, Copy)]
pub struct GapPoint {
    pub point: DVec3,
    pub line_width: f64,
}

/// Context bundle for planning gap fill on a layer.
#[derive(Clone, Copy)]
pub struct GapFillContext<'a> {
    pub config: &'a SlicerConfig,
    pub canonical_tangent_footprint: &'a [Vec<[f64; 2]>],
    pub basis1: DVec3,
    pub basis2: DVec3,
    pub origin: DVec3,
    pub tool: ToolId,
}

/// Evaluates the 3D centerline gap-fill point between a wall loop vertex `p` and its neighbor.
///
/// `inward`:
/// - `false`: towards the outer wall pass ($+\hat{u}$)
/// - `true`: towards the interior / infill boundary ($-\hat{u}$)
#[must_use]
pub fn compute_gap_point(
    p: DVec3,
    target_ds: f64,
    n_cad: DVec3,
    n_order: DVec3,
    inward: bool,
    config: &SlicerConfig,
    layer: &Layer,
) -> Option<GapPoint> {
    let dot = n_cad.dot(n_order);
    let u_vec = n_cad - n_order * dot;
    let u_len = u_vec.length();
    if u_len <= 1e-4 || !u_len.is_finite() {
        return None;
    }

    let u_hat = u_vec / u_len;
    let dir = if inward { -u_hat } else { u_hat };
    let step = 0.5 * target_ds;
    // Cap step: gap fill should never jump further than 1.5x nominal line width
    if step > 1.5 * config.wall_line_width {
        return None;
    }
    let p_raw = p + dir * step;

    // Refine point onto the order field isosurface
    let p_refined = crate::order_field::refine_point_onto_order_field(
        p_raw,
        layer.order,
        config.layer_height,
        layer.order_field.as_ref(),
    )
    .unwrap_or(p_raw);

    // Verify point and chord are inside or on the solid CAD model
    if let Some(sdf) = layer.mesh_sdf.as_deref() {
        if sdf.sample(p_refined).value > 0.05 {
            return None;
        }
        let p_mid = (p + p_refined) * 0.5;
        if sdf.sample(p_mid).value > 0.05 {
            return None;
        }
    }

    let line_width = (target_ds * 0.5).clamp(config.min_bead_width(), config.max_bead_width());
    Some(GapPoint {
        point: p_refined,
        line_width,
    })
}

/// Extracts contiguous qualifying index runs from a boolean array.
///
/// Returns a list of `(indices, is_closed_chain)` tuples:
/// - If `is_closed` is true and all elements qualify, returns a single closed chain.
/// - Otherwise, returns all contiguous runs with $\ge 2$ points (wrapping around if `is_closed`).
#[must_use]
pub fn extract_gap_chains(qualifying: &[bool], is_closed: bool) -> Vec<(Vec<usize>, bool)> {
    let n = qualifying.len();
    if n < 2 {
        return Vec::new();
    }

    let all_true = qualifying.iter().all(|&q| q);
    if all_true && is_closed {
        return vec![((0..n).collect(), true)];
    }

    let mut chains = Vec::new();

    if is_closed {
        // Find the first false to break the cycle cleanly
        let start_search = qualifying.iter().position(|&q| !q).unwrap_or(0);
        let mut current_run = Vec::new();

        for step in 0..n {
            let idx = (start_search + 1 + step) % n;
            if qualifying[idx] {
                current_run.push(idx);
            } else if !current_run.is_empty() {
                if current_run.len() >= 2 {
                    chains.push((std::mem::take(&mut current_run), false));
                } else {
                    current_run.clear();
                }
            }
        }
        if current_run.len() >= 2 {
            chains.push((current_run, false));
        }
    } else {
        let mut current_run = Vec::new();
        for (idx, &q) in qualifying.iter().enumerate() {
            if q {
                current_run.push(idx);
            } else if !current_run.is_empty() {
                if current_run.len() >= 2 {
                    chains.push((std::mem::take(&mut current_run), false));
                } else {
                    current_run.clear();
                }
            }
        }
        if current_run.len() >= 2 {
            chains.push((current_run, false));
        }
    }

    chains
}

/// Plans gap-fill paths for a wall loop where in-surface spacing $\Delta s > w_{\max}$.
///
/// Returns:
/// - `updated_line_widths`: Line widths for the original wall loop (halved where gap fill is present).
/// - `gap_fill_paths`: Generated centerline gap fill toolpaths.
#[must_use]
pub fn plan_gap_fill_for_wall(
    wall_loop: &WallLoop,
    layer: &Layer,
    ctx: &GapFillContext<'_>,
) -> (Vec<f64>, Vec<Path>) {
    let point_count = wall_loop.points.len();
    if point_count < 2 {
        return (wall_loop.line_widths.clone(), Vec::new());
    }

    let config = ctx.config;
    let min_w = config.min_bead_width();
    let max_w = config.max_bead_width();
    let sdf = match layer.mesh_sdf.as_deref() {
        Some(s) => s,
        None => return (wall_loop.line_widths.clone(), Vec::new()),
    };

    let eps = 0.02;
    let mut target_ds_list = Vec::with_capacity(point_count);
    let mut gap_points_outer: Vec<Option<GapPoint>> = vec![None; point_count];
    let mut gap_points_inner: Vec<Option<GapPoint>> = vec![None; point_count];

    let max_w_on_island = layer
        .loops
        .iter()
        .filter(|l| l.island == wall_loop.island)
        .map(|l| l.wall_index)
        .max()
        .unwrap_or(0);
    let is_innermost =
        wall_loop.wall_index >= max_w_on_island || wall_loop.wall_index + 1 >= config.wall_count();

    for i in 0..point_count {
        let p = wall_loop.points[i];
        let p_2d = [
            (p - ctx.origin).dot(ctx.basis1),
            (p - ctx.origin).dot(ctx.basis2),
        ];

        // Skip points inside tangent surface footprints (already filled with wave fill)
        if !ctx.canonical_tangent_footprint.is_empty()
            && crate::polygon2d::contains_point(ctx.canonical_tangent_footprint, p_2d)
        {
            target_ds_list.push(config.wall_line_width);
            continue;
        }

        let dx = sdf.sample(p + DVec3::X * eps).value - sdf.sample(p - DVec3::X * eps).value;
        let dy = sdf.sample(p + DVec3::Y * eps).value - sdf.sample(p - DVec3::Y * eps).value;
        let dz = sdf.sample(p + DVec3::Z * eps).value - sdf.sample(p - DVec3::Z * eps).value;
        let g_cad = DVec3::new(dx, dy, dz);
        let g_len = g_cad.length();

        let odx = layer.order_field.order(p + DVec3::X * eps)
            - layer.order_field.order(p - DVec3::X * eps);
        let ody = layer.order_field.order(p + DVec3::Y * eps)
            - layer.order_field.order(p - DVec3::Y * eps);
        let odz = layer.order_field.order(p + DVec3::Z * eps)
            - layer.order_field.order(p - DVec3::Z * eps);
        let g_order = DVec3::new(odx, ody, odz);
        let o_len = g_order.length();

        if g_len > 1e-6 && o_len > 1e-6 {
            let n_cad = g_cad / g_len;
            let n_order = g_order / o_len;
            let cross_len = n_cad.cross(n_order).length();

            if cross_len > 1e-4 && cross_len.is_finite() {
                let ds = config.wall_line_width / cross_len;
                target_ds_list.push(ds);

                if ds > max_w {
                    // Gap fill outward (between Wall w and Wall w - 1)
                    if wall_loop.wall_index > 0 {
                        gap_points_outer[i] =
                            compute_gap_point(p, ds, n_cad, n_order, false, config, layer);
                    }

                    // Gap fill inward (between innermost wall and infill boundary)
                    if is_innermost && !layer.infill_boundary.is_empty() {
                        gap_points_inner[i] =
                            compute_gap_point(p, ds, n_cad, n_order, true, config, layer);
                    }
                }
                continue;
            }
        }
        target_ds_list.push(config.wall_line_width);
    }

    let mut updated_line_widths = Vec::with_capacity(point_count);
    let mut gap_fill_paths = Vec::new();

    // Determine updated wall line widths for the primary wall
    for i in 0..point_count {
        let ds = target_ds_list[i];
        let has_gap_fill = gap_points_outer[i].is_some() || gap_points_inner[i].is_some();
        let w = if wall_loop.wall_index > 0 {
            if has_gap_fill {
                (ds * 0.5).clamp(min_w, max_w)
            } else {
                ds.clamp(min_w, max_w)
            }
        } else {
            config.wall_line_width
        };
        updated_line_widths.push(w);
    }

    // Build toolpaths for outward and inward gap fill
    for (gap_points, _is_inward) in [(gap_points_outer, false), (gap_points_inner, true)] {
        let qualifying: Vec<bool> = gap_points.iter().map(|gp| gp.is_some()).collect();
        let chains = extract_gap_chains(&qualifying, !wall_loop.is_open);

        let min_path_len = config.nozzle_diameter * 0.75;

        for (chain, is_closed_chain) in chains {
            let num_pts = chain.len();
            if num_pts < 2 {
                continue;
            }

            let pts: Vec<DVec3> = chain
                .iter()
                .map(|&idx| gap_points[idx].unwrap().point)
                .collect();

            // Discard chains with any excessive jump between consecutive vertices
            let max_edge_len = 3.0 * config.wall_line_width;
            let has_void_jump = (0..num_pts.saturating_sub(1))
                .any(|j| pts[j].distance(pts[j + 1]) > max_edge_len)
                || (is_closed_chain && pts[num_pts - 1].distance(pts[0]) > max_edge_len);
            if has_void_jump {
                continue;
            }

            // Check total path length
            let mut total_len = 0.0;
            for j in 0..num_pts.saturating_sub(1) {
                total_len += pts[j].distance(pts[j + 1]);
            }
            if is_closed_chain {
                total_len += pts[num_pts - 1].distance(pts[0]);
            }

            if total_len < min_path_len {
                continue;
            }

            let seg_count = if is_closed_chain {
                num_pts
            } else {
                num_pts - 1
            };
            let segments: Vec<Segment> = (0..seg_count)
                .map(|j| {
                    let idx = chain[j];
                    let line_width = gap_points[idx].unwrap().line_width;
                    Segment {
                        kind: MoveKind::WallInner,
                        speed: speed_for_kind(MoveKind::WallInner, config),
                        extrusion_rate: 1.0,
                        support_fraction: 0.0,
                        order: layer.order,
                        extrusion_length: 0.0,
                        line_width,
                        is_scarf: false,
                        id: 0,
                        island: wall_loop.island,
                        channel_width: f64::INFINITY,
                    }
                })
                .collect();

            gap_fill_paths.push(Path {
                points: pts,
                segments,
                tool: ctx.tool,
            });
        }
    }

    (updated_line_widths, gap_fill_paths)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_gap_chains_empty_returns_empty() {
        assert!(extract_gap_chains(&[], true).is_empty());
        assert!(extract_gap_chains(&[true], true).is_empty());
    }

    #[test]
    fn extract_gap_chains_all_true_closed_returns_closed_chain() {
        let q = vec![true, true, true, true];
        let chains = extract_gap_chains(&q, true);
        assert_eq!(chains.len(), 1);
        assert_eq!(chains[0].0, vec![0, 1, 2, 3]);
        assert!(chains[0].1); // is_closed
    }

    #[test]
    fn extract_gap_chains_all_true_open_returns_open_chain() {
        let q = vec![true, true, true, true];
        let chains = extract_gap_chains(&q, false);
        assert_eq!(chains.len(), 1);
        assert_eq!(chains[0].0, vec![0, 1, 2, 3]);
        assert!(!chains[0].1); // is_closed == false
    }

    #[test]
    fn extract_gap_chains_wraparound_closed_joins_into_single_chain() {
        // [true, false, true, true] with wrap: 2 -> 3 -> 0 is contiguous!
        let q = vec![true, false, true, true];
        let chains = extract_gap_chains(&q, true);
        assert_eq!(chains.len(), 1);
        assert_eq!(chains[0].0, vec![2, 3, 0]);
        assert!(!chains[0].1);
    }

    #[test]
    fn extract_gap_chains_filters_single_isolated_true() {
        let q = vec![false, true, false, true, true];
        let chains = extract_gap_chains(&q, false);
        assert_eq!(chains.len(), 1);
        assert_eq!(chains[0].0, vec![3, 4]);
    }
}
