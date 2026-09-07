//! Straight bridge path generation for unsupported spans connecting preexisting material.
//!
//! A bridge is an extrusion over empty space that contacts preexisting material at both ends.
//! Bridges must be straight lines (no turning or curling in mid-air), running between
//! supported anchors with dedicated bridge feedrate, fan cooling, and line width.

use glam::DVec3;
use rayon::prelude::*;

use crate::ids::ToolId;
use crate::order_field;
use crate::polygon2d;
use crate::slicing::Layer;
use crate::toolpath::{MoveKind, Path, Segment};
use crate::wave_overhang::{group_loops_into_polygon_shapes, LineSegment2D, PolygonShape2D};
use crate::SlicerConfig;

/// Result of bridge path planning across all layers.
#[derive(Clone, Debug, Default)]
pub struct BridgePlan {
    /// Generated bridge toolpaths per layer.
    pub paths_by_layer: Vec<Vec<Path>>,
    /// 2D bridge footprints per layer (in layer's order plane basis).
    pub bridge_footprints_by_layer: Vec<Vec<Vec<[f64; 2]>>>,
}

/// Identifies unsupported spans that contact preexisting material at both ends
/// and generates straight bridging toolpaths across them.
#[must_use]
pub fn plan_bridges(layers: &[Layer], config: &SlicerConfig, tool: ToolId) -> BridgePlan {
    if layers.len() < 2 {
        return BridgePlan {
            paths_by_layer: vec![Vec::new(); layers.len()],
            bridge_footprints_by_layer: vec![Vec::new(); layers.len()],
        };
    }

    let (axis, apex, _slope) = order_field::resolve_axis_apex_slope(config.order_field, config);
    let (basis1, basis2) = manifold_fidget::contour::plane_basis(axis);
    let origin = apex;

    let min_bridge_area = 0.25 * config.nozzle_diameter * config.nozzle_diameter;
    let max_along = order_field::max_along_for(config);
    let bridge_speed = config.bridge_speed();
    let bridge_line_width = config.infill_line_width.max(0.1);

    // Determine whether layer index `k` increases with physical height (Z)
    let z_at = |l: &Layer| -> f64 {
        let mut sum_z = 0.0;
        let mut count = 0usize;
        for pts in &l.infill_boundary {
            for p in pts {
                sum_z += p.z;
                count += 1;
            }
        }
        if count == 0 {
            for wall in &l.loops {
                for p in &wall.points {
                    sum_z += p.z;
                    count += 1;
                }
            }
        }
        if count > 0 {
            sum_z / count as f64
        } else {
            0.0
        }
    };

    let first_pos = layers
        .iter()
        .find(|l| !l.infill_boundary.is_empty() || !l.loops.is_empty());
    let last_pos = layers
        .iter()
        .rfind(|l| !l.infill_boundary.is_empty() || !l.loops.is_empty());
    let z_increases = match (first_pos, last_pos) {
        (Some(f), Some(l)) if f.index != l.index => z_at(l) >= z_at(f),
        _ => true,
    };

    // Compute 2D outer wall boundaries for all layers in parallel
    let boundaries_2d: Vec<Vec<Vec<[f64; 2]>>> = layers
        .par_iter()
        .map(|layer| {
            let wall0_loops: Vec<Vec<DVec3>> = layer
                .loops
                .iter()
                .filter(|w| w.wall_index == 0)
                .map(|w| w.points.clone())
                .collect();
            let raw_2d = if wall0_loops.is_empty() {
                polygon2d::to_2d(&layer.infill_boundary, basis1, basis2, origin)
            } else {
                polygon2d::to_2d(&wall0_loops, basis1, basis2, origin)
            };
            polygon2d::canonicalize(&raw_2d)
        })
        .collect();

    let (paths_result, footprints_result): (Vec<_>, Vec<_>) = (0..layers.len())
        .into_par_iter()
        .map(|k| {
            let prev_idx = if z_increases {
                if k == 0 {
                    None
                } else {
                    Some(k - 1)
                }
            } else if k + 1 < layers.len() {
                Some(k + 1)
            } else {
                None
            };

            let Some(prev_k) = prev_idx else {
                return (Vec::new(), Vec::new());
            };

            let cur_b = &boundaries_2d[k];
            let prev_b = &boundaries_2d[prev_k];

            if cur_b.is_empty() || prev_b.is_empty() {
                return (Vec::new(), Vec::new());
            }

            // Unsupported overhang region: cur_layer \ prev_layer
            let raw_overhang = polygon2d::difference(cur_b, prev_b);
            let overhang_filtered = polygon2d::filter_min_area(&raw_overhang, min_bridge_area);

            if overhang_filtered.is_empty() {
                return (Vec::new(), Vec::new());
            }

            let shapes = group_loops_into_polygon_shapes(&overhang_filtered);
            let mut layer_bridge_paths_2d = Vec::new();
            let mut layer_bridge_footprints = Vec::new();

            let mut references: Vec<Vec<DVec3>> = layers[k]
                .loops
                .iter()
                .filter(|w| w.wall_index == 0)
                .map(|w| w.points.clone())
                .collect();
            if references.is_empty() {
                references = layers[k].infill_boundary.clone();
            }

            let search_dist = (config.nozzle_diameter * 1.25).max(0.4);

            for shape in &shapes {
                let n = shape.outer.len();
                if n < 3 {
                    continue;
                }

                // Identify contact segments with prev_b and cluster into anchor groups
                let mut contact_runs: Vec<Vec<usize>> = Vec::new();
                let mut current_run = Vec::new();
                let mut in_contact = false;

                for i in 0..n {
                    let p0 = shape.outer[i];
                    let p1 = shape.outer[(i + 1) % n];
                    let mid = [(p0[0] + p1[0]) * 0.5, (p0[1] + p1[1]) * 0.5];
                    let near = polygon2d_contains_or_near(mid, prev_b, search_dist);
                    if near {
                        current_run.push(i);
                        in_contact = true;
                    } else if in_contact {
                        contact_runs.push(current_run.clone());
                        current_run.clear();
                        in_contact = false;
                    }
                }
                if !current_run.is_empty() {
                    contact_runs.push(current_run);
                }
                // Handle cyclic wrap-around at index 0
                if contact_runs.len() > 1
                    && contact_runs.first().unwrap().contains(&0)
                    && contact_runs.last().unwrap().contains(&(n - 1))
                {
                    let last = contact_runs.pop().unwrap();
                    contact_runs[0].extend(last);
                }

                // A bridge requires contact with preexisting material at BOTH ends (>= 2 separated anchor groups)
                if contact_runs.len() < 2 {
                    continue;
                }

                // 3D Solid Mesh Validation
                if let Some(sdf) = &layers[k].mesh_sdf {
                    let step = (n / 8).max(1);
                    let sample_pts: Vec<[f64; 2]> = (0..n)
                        .step_by(step)
                        .take(8)
                        .map(|i| shape.outer[i])
                        .collect();
                    let reconstructed = order_field::reconstruct_on_order_field_near(
                        vec![sample_pts],
                        &references,
                        basis1,
                        basis2,
                        axis,
                        apex,
                        layers[k].order,
                        max_along,
                        layers[k].order_field.as_ref(),
                    );
                    if let Some(pts_3d) = reconstructed.first() {
                        if !pts_3d.is_empty() {
                            let in_solid_count = pts_3d
                                .iter()
                                .filter(|p| {
                                    manifold_fidget::ScalarField::sample(sdf.as_ref(), **p).value
                                        <= 0.35
                                })
                                .count();
                            if in_solid_count == 0 {
                                continue;
                            }
                        }
                    }
                }

                // Generate straight bridge lines across shape between anchor groups
                let lines_2d = generate_straight_bridge_paths_2d(
                    shape,
                    &contact_runs,
                    prev_b,
                    bridge_line_width,
                    config.nozzle_diameter,
                );

                if !lines_2d.is_empty() {
                    layer_bridge_paths_2d.extend(lines_2d);
                    layer_bridge_footprints.push(shape.outer.clone());
                }
            }

            if layer_bridge_paths_2d.is_empty() {
                return (Vec::new(), Vec::new());
            }

            // Reconstruct 2D straight bridge segments to 3D
            let lines_3d = order_field::reconstruct_on_order_field_near(
                layer_bridge_paths_2d,
                &references,
                basis1,
                basis2,
                axis,
                apex,
                layers[k].order,
                max_along,
                layers[k].order_field.as_ref(),
            );

            let mut paths = Vec::new();
            let mut reverse = false;

            for line in lines_3d {
                if line.len() < 2 {
                    continue;
                }
                let pts = if reverse {
                    vec![line[1], line[0]]
                } else {
                    vec![line[0], line[1]]
                };
                reverse = !reverse;

                let segments = vec![Segment {
                    kind: MoveKind::Bridge,
                    speed: bridge_speed,
                    extrusion_rate: 1.0,
                    support_fraction: 0.0,
                    order: layers[k].order,
                    extrusion_length: 0.0,
                    line_width: bridge_line_width,
                    is_scarf: false,
                    id: 0,
                    island: 0,
                    channel_width: f64::INFINITY,
                }];

                paths.push(Path {
                    points: pts,
                    segments,
                    tool,
                });
            }

            (paths, layer_bridge_footprints)
        })
        .unzip();

    BridgePlan {
        paths_by_layer: paths_result,
        bridge_footprints_by_layer: footprints_result,
    }
}

/// Generates straight bridging lines across an unsupported polygon shape
/// connecting at least two separated contact anchor groups.
#[must_use]
pub fn generate_straight_bridge_paths_2d(
    shape: &PolygonShape2D,
    contact_runs: &[Vec<usize>],
    prev_b: &[Vec<[f64; 2]>],
    spacing: f64,
    nozzle_diameter: f64,
) -> Vec<Vec<[f64; 2]>> {
    if contact_runs.len() < 2 || shape.outer.len() < 3 {
        return Vec::new();
    }

    let n = shape.outer.len();

    // Compute centroids of the two largest anchor groups
    let mut sorted_runs = contact_runs.to_vec();
    sorted_runs.sort_by_key(|r| std::cmp::Reverse(r.len()));

    let run_centroid = |indices: &[usize]| -> [f64; 2] {
        let mut sum_u = 0.0;
        let mut sum_v = 0.0;
        let mut count = 0.0;
        for &idx in indices {
            let p = shape.outer[idx];
            sum_u += p[0];
            sum_v += p[1];
            count += 1.0;
        }
        if count > 0.0 {
            [sum_u / count, sum_v / count]
        } else {
            [0.0, 0.0]
        }
    };

    let c0 = run_centroid(&sorted_runs[0]);
    let c1 = run_centroid(&sorted_runs[1]);

    let span_v = [c1[0] - c0[0], c1[1] - c0[1]];
    let span_len = span_v[0].hypot(span_v[1]);
    if span_len < 1e-4 {
        return Vec::new();
    }

    // Direction vector of the bridge span
    let dir = [span_v[0] / span_len, span_v[1] / span_len];
    // Normal vector perpendicular to bridge lines
    let normal = [-dir[1], dir[0]];

    // Find lateral extent across the normal direction
    let mut min_w = f64::INFINITY;
    let mut max_w = f64::NEG_INFINITY;
    for &p in &shape.outer {
        let w = p[0] * normal[0] + p[1] * normal[1];
        min_w = min_w.min(w);
        max_w = max_w.max(w);
    }

    let spacing = spacing.max(0.1);
    let mut lines_2d = Vec::new();
    let anchor_margin = nozzle_diameter * 0.75;

    let mut w = min_w + spacing * 0.5;
    while w <= max_w - spacing * 0.25 {
        // Intersect line { p | p . normal = w } with shape's outer boundary edges
        let mut t_values = Vec::new();

        for i in 0..n {
            let p0 = shape.outer[i];
            let p1 = shape.outer[(i + 1) % n];

            let w0 = p0[0] * normal[0] + p0[1] * normal[1];
            let w1 = p1[0] * normal[0] + p1[1] * normal[1];

            if (w0 <= w && w1 > w) || (w1 <= w && w0 > w) {
                let frac = (w - w0) / (w1 - w0);
                let hit_u = p0[0] + frac * (p1[0] - p0[0]);
                let hit_v = p0[1] + frac * (p1[1] - p0[1]);
                let t = hit_u * dir[0] + hit_v * dir[1];
                t_values.push(t);
            }
        }

        t_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Group into segments inside the polygon
        for chunk in t_values.chunks_exact(2) {
            let t_start = chunk[0];
            let t_end = chunk[1];
            if t_end - t_start < nozzle_diameter * 0.75 {
                continue;
            }

            let mid_t = (t_start + t_end) * 0.5;
            let mid_u = w * normal[0] + mid_t * dir[0];
            let mid_v = w * normal[1] + mid_t * dir[1];

            if !shape.contains_point([mid_u, mid_v]) {
                continue;
            }

            // Extend endpoints into preexisting material for solid anchor contact
            let p_start = [
                w * normal[0] + (t_start - anchor_margin) * dir[0],
                w * normal[1] + (t_start - anchor_margin) * dir[1],
            ];
            let p_end = [
                w * normal[0] + (t_end + anchor_margin) * dir[0],
                w * normal[1] + (t_end + anchor_margin) * dir[1],
            ];

            // Verify that endpoints actually land on or near preexisting material (prev_b)
            let start_supported =
                polygon2d_contains_or_near(p_start, prev_b, nozzle_diameter * 1.5);
            let end_supported = polygon2d_contains_or_near(p_end, prev_b, nozzle_diameter * 1.5);

            if start_supported && end_supported {
                lines_2d.push(vec![p_start, p_end]);
            }
        }

        w += spacing;
    }

    lines_2d
}

fn polygon2d_contains_or_near(pt: [f64; 2], loops: &[Vec<[f64; 2]>], eps: f64) -> bool {
    let eps_sq = eps * eps;
    for loop_ in loops {
        if point_in_single_loop(pt, loop_) {
            return true;
        }
        let n = loop_.len();
        for i in 0..n {
            let seg = LineSegment2D {
                p0: loop_[i],
                p1: loop_[(i + 1) % n],
            };
            if seg.dist_sq_to_point(pt) <= eps_sq {
                return true;
            }
        }
    }
    false
}

fn point_in_single_loop(pt: [f64; 2], loop_: &[[f64; 2]]) -> bool {
    if loop_.len() < 3 {
        return false;
    }
    let [x, y] = pt;
    let mut inside = false;
    let mut j = loop_.len() - 1;
    for i in 0..loop_.len() {
        let [xi, yi] = loop_[i];
        let [xj, yj] = loop_[j];
        let intersect = ((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
        if intersect {
            inside = !inside;
        }
        j = i;
    }
    inside
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn straight_bridge_generates_parallel_straight_lines() {
        let outer = vec![[0.0, 0.0], [10.0, 0.0], [10.0, 5.0], [0.0, 5.0]];
        let shape = PolygonShape2D {
            outer,
            holes: Vec::new(),
        };
        // Anchors at x = 0 (left) and x = 10 (right)
        let contact_runs = vec![vec![3], vec![1]];
        let prev_b = vec![
            vec![[-2.0, 0.0], [0.1, 0.0], [0.1, 5.0], [-2.0, 5.0]],
            vec![[9.9, 0.0], [12.0, 0.0], [12.0, 5.0], [9.9, 5.0]],
        ];
        let lines = generate_straight_bridge_paths_2d(&shape, &contact_runs, &prev_b, 1.0, 0.4);
        assert!(!lines.is_empty(), "should generate straight bridge lines");
        for line in &lines {
            assert_eq!(
                line.len(),
                2,
                "bridge line must have exactly 2 points (straight)"
            );
            assert!(line[1][0] > line[0][0], "span must run across the gap");
        }
    }
}
