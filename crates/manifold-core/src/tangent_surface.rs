//! Tangent Surface planning: categorizes surfaces where an inner wall or infill
//! of an adjacent isosurface would be exposed, and generates complete wave fill
//! with longest-segment midpoint seeding.

use crate::ids::ToolId;
use crate::order_field;
use crate::polygon2d;
use crate::slicing::Layer;
use crate::toolpath::{MoveKind, Path, Segment};
use crate::wave_overhang::{
    generate_wave_overhang_paths_2d, group_loops_into_polygon_shapes, LineSegment2D,
};
use crate::SlicerConfig;
use glam::DVec3;
use manifold_fidget::ScalarField;
use rayon::prelude::*;

/// Orientation of a tangent surface relative to the build progression.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TangentOrientation {
    /// Surface exposed to empty air below (where layer below stepped inward or did not exist).
    /// Printed with wave overhang settings (MoveKind::Overhang, wave_overhang_speed).
    Downward,
    /// Surface exposed to air above (where layer above steps inward).
    /// Printed with outer wall settings (MoveKind::WallOuter, outer_wall_speed).
    Upward,
}

/// Result of tangent surface planning containing categorized wave fill paths
/// and 2D tangent surface footprints per layer.
#[derive(Clone, Debug, Default)]
pub struct TangentSurfacePlan {
    pub paths_by_layer: Vec<Vec<Path>>,
    pub downward_footprints_by_layer: Vec<Vec<Vec<[f64; 2]>>>,
    pub upward_footprints_by_layer: Vec<Vec<Vec<[f64; 2]>>>,
    pub footprints_by_layer: Vec<Vec<Vec<[f64; 2]>>>,
}

/// Identifies tangent surfaces across layers and generates wave fill toolpaths.
#[must_use]
pub fn plan_tangent_surfaces(
    layers: &[Layer],
    config: &SlicerConfig,
    tool: ToolId,
) -> TangentSurfacePlan {
    if layers.len() < 2 {
        return TangentSurfacePlan {
            paths_by_layer: vec![Vec::new(); layers.len()],
            downward_footprints_by_layer: vec![Vec::new(); layers.len()],
            upward_footprints_by_layer: vec![Vec::new(); layers.len()],
            footprints_by_layer: vec![Vec::new(); layers.len()],
        };
    }

    let (axis, apex, _slope) = order_field::resolve_axis_apex_slope(config.order_field, config);
    let (basis1, basis2) = manifold_fidget::contour::plane_basis(axis);
    let origin = apex;

    let wavelength = (config.nozzle_diameter - config.wave_overhang_overlap()).max(0.10);
    let min_tangent_area = 0.25 * config.nozzle_diameter * config.nozzle_diameter;
    let max_along = order_field::max_along_for(config);

    let z_increases = crate::slicing::layer_z_increases(layers);
    let outer_2d = crate::slicing::layers_outer_boundaries_2d(layers, basis1, basis2, origin);

    let results: Vec<_> = (0..layers.len())
        .into_par_iter()
        .map(|k| {
            let (prev_idx, next_idx) = if z_increases {
                (
                    if k > 0 { Some(k - 1) } else { None },
                    if k + 1 < layers.len() {
                        Some(k + 1)
                    } else {
                        None
                    },
                )
            } else {
                (
                    if k + 1 < layers.len() {
                        Some(k + 1)
                    } else {
                        None
                    },
                    if k > 0 { Some(k - 1) } else { None },
                )
            };

            let cur_b = &outer_2d[k];
            if cur_b.is_empty() {
                return (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            }

            let mut layer_paths = Vec::new();
            let mut layer_downward_footprints = Vec::new();
            let mut layer_upward_footprints = Vec::new();
            let mut layer_all_footprints = Vec::new();

            let references: Vec<Vec<DVec3>> = {
                let w0: Vec<Vec<DVec3>> = layers[k]
                    .loops
                    .iter()
                    .filter(|w| w.wall_index == 0)
                    .map(|w| w.points.clone())
                    .collect();
                if w0.is_empty() {
                    layers[k].infill_boundary.clone()
                } else {
                    w0
                }
            };

            // Categorize tangent surfaces: Downward vs Upward
            let categories = [
                (TangentOrientation::Downward, prev_idx),
                (TangentOrientation::Upward, next_idx),
            ];

            for (orientation, adjacent_idx) in categories {
                let Some(adj_k) = adjacent_idx else {
                    continue;
                };
                let adj_b = &outer_2d[adj_k];
                if adj_b.is_empty() {
                    continue;
                }

                // Tangent surface region: area where adjacent layer stepped inward, exposing this surface
                let raw_tangent = polygon2d::difference(cur_b, adj_b);
                let tangent_filtered = polygon2d::filter_min_area(&raw_tangent, min_tangent_area);
                if tangent_filtered.is_empty() {
                    continue;
                }

                let shapes = group_loops_into_polygon_shapes(&tangent_filtered);
                let contact_search_dist = (config.nozzle_diameter * 0.50).max(0.10);
                let contact_tol_sq = contact_search_dist * contact_search_dist;

                for shape in shapes {
                    let n = shape.outer.len();
                    if n < 3 {
                        continue;
                    }

                    // 1. Identify contact segments bordering the reference adjacent layer
                    // across all boundary loops (outer loop and any interior hole loops),
                    // and select the LONGEST available contact segment to avoid filling from corners.
                    let mut candidate_segments = Vec::new();
                    let mut all_loops = vec![&shape.outer];
                    all_loops.extend(shape.holes.iter());

                    for loop_ in all_loops {
                        let ln = loop_.len();
                        for i in 0..ln {
                            let p0 = loop_[i];
                            let p1 = loop_[(i + 1) % ln];
                            let mid = [(p0[0] + p1[0]) * 0.5, (p0[1] + p1[1]) * 0.5];

                            // Test contact with adj_b
                            let mut min_d_sq = f64::INFINITY;
                            for l in adj_b {
                                let l_len = l.len();
                                for j in 0..l_len {
                                    let s = LineSegment2D {
                                        p0: l[j],
                                        p1: l[(j + 1) % l_len],
                                    };
                                    min_d_sq = min_d_sq.min(s.dist_sq_to_point(mid));
                                }
                            }

                            if min_d_sq <= contact_tol_sq {
                                let len = (p1[0] - p0[0]).hypot(p1[1] - p0[1]);
                                candidate_segments.push((p0, p1, len));
                            }
                        }
                    }

                    // Seed placement: placed at the midpoint of the longest available segment
                    let seed_segments = if let Some(&(p0, p1, len)) = candidate_segments
                        .iter()
                        .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
                    {
                        let mid = [(p0[0] + p1[0]) * 0.5, (p0[1] + p1[1]) * 0.5];
                        let dir = if len > 1e-9 {
                            [(p1[0] - p0[0]) / len, (p1[1] - p0[1]) / len]
                        } else {
                            [1.0, 0.0]
                        };
                        let half_w = (config.nozzle_diameter * 0.5).min(len * 0.5);
                        vec![LineSegment2D {
                            p0: [mid[0] - dir[0] * half_w, mid[1] - dir[1] * half_w],
                            p1: [mid[0] + dir[0] * half_w, mid[1] + dir[1] * half_w],
                        }]
                    } else {
                        // Unconnected island without contact: skip to prevent mid-air floating fills
                        continue;
                    };

                    // 2. Generate wave fill paths in 2D
                    let wave_polylines_2d =
                        generate_wave_overhang_paths_2d(&shape, &seed_segments, wavelength, config);
                    if wave_polylines_2d.is_empty() {
                        continue;
                    }

                    // 3. Reconstruct wave fill polylines to 3D
                    let wave_polylines_3d = order_field::reconstruct_on_order_field_near(
                        wave_polylines_2d,
                        &references,
                        basis1,
                        basis2,
                        axis,
                        apex,
                        layers[k].order,
                        max_along,
                        layers[k].order_field.as_ref(),
                    );

                    // Determine true physical surface orientation from 3D surface normal:
                    // If the surface normal points downward (nz < -0.15), it is an unsupported overhang.
                    // If the surface normal points upward (nz >= -0.15), it is an upward-facing tangent surface.
                    let mut sum_nz = 0.0;
                    let mut count_n = 0;
                    if let Some(sdf) = &layers[k].mesh_sdf {
                        let eps = 0.02;
                        for p in &wave_polylines_3d {
                            for pt in p {
                                let dx = sdf.sample(*pt + glam::DVec3::X * eps).value
                                    - sdf.sample(*pt - glam::DVec3::X * eps).value;
                                let dy = sdf.sample(*pt + glam::DVec3::Y * eps).value
                                    - sdf.sample(*pt - glam::DVec3::Y * eps).value;
                                let dz = sdf.sample(*pt + glam::DVec3::Z * eps).value
                                    - sdf.sample(*pt - glam::DVec3::Z * eps).value;
                                let len = (dx * dx + dy * dy + dz * dz).sqrt();
                                if len > 1e-6 {
                                    sum_nz += dz / len;
                                    count_n += 1;
                                }
                            }
                        }
                    }
                    let avg_nz = if count_n > 0 {
                        sum_nz / count_n as f64
                    } else {
                        match orientation {
                            TangentOrientation::Downward => -1.0,
                            TangentOrientation::Upward => 1.0,
                        }
                    };
                    let true_orientation = if avg_nz < -0.15 {
                        TangentOrientation::Downward
                    } else {
                        TangentOrientation::Upward
                    };

                    let (kind, line_width) = match true_orientation {
                        TangentOrientation::Downward => {
                            (MoveKind::Overhang, config.nozzle_diameter)
                        }
                        TangentOrientation::Upward => (MoveKind::WallOuter, config.wall_line_width),
                    };
                    let speed = crate::toolpath::speed_for_kind(kind, config);

                    let mut reverse = false;
                    let mut generated_any_path = false;
                    for poly in wave_polylines_3d {
                        if poly.len() < 2 {
                            continue;
                        }
                        generated_any_path = true;
                        let pts = if reverse {
                            poly.into_iter().rev().collect()
                        } else {
                            poly
                        };
                        reverse = !reverse;

                        let seg_count = pts.len() - 1;
                        let segments: Vec<Segment> = (0..seg_count)
                            .map(|_| Segment {
                                kind,
                                speed,
                                extrusion_rate: 1.0,
                                support_fraction: 0.0,
                                order: layers[k].order,
                                extrusion_length: 0.0,
                                line_width,
                                is_scarf: false,
                                id: 0,
                                island: 0,
                                channel_width: f64::INFINITY,
                            })
                            .collect();

                        layer_paths.push(Path {
                            points: pts,
                            segments,
                            tool,
                        });
                    }

                    if generated_any_path {
                        match true_orientation {
                            TangentOrientation::Downward => {
                                layer_downward_footprints.push(shape.outer.clone());
                            }
                            TangentOrientation::Upward => {
                                layer_upward_footprints.push(shape.outer.clone());
                            }
                        }
                        layer_all_footprints.push(shape.outer);
                    }
                }
            }

            (
                layer_paths,
                layer_downward_footprints,
                layer_upward_footprints,
                layer_all_footprints,
            )
        })
        .collect();

    let mut paths_by_layer = Vec::with_capacity(layers.len());
    let mut downward_footprints_by_layer = Vec::with_capacity(layers.len());
    let mut upward_footprints_by_layer = Vec::with_capacity(layers.len());
    let mut footprints_by_layer = Vec::with_capacity(layers.len());

    for (p, df, uf, af) in results {
        paths_by_layer.push(p);
        downward_footprints_by_layer.push(df);
        upward_footprints_by_layer.push(uf);
        footprints_by_layer.push(af);
    }

    TangentSurfacePlan {
        paths_by_layer,
        downward_footprints_by_layer,
        upward_footprints_by_layer,
        footprints_by_layer,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::ObjectId;
    use crate::slicing::WallLoop;
    use std::sync::Arc;

    fn make_test_layer(index: usize, order: f64, size: f64) -> Layer {
        let half = size * 0.5;
        let pts = vec![
            DVec3::new(-half, -half, order),
            DVec3::new(half, -half, order),
            DVec3::new(half, half, order),
            DVec3::new(-half, half, order),
        ];
        let n = pts.len();
        let loop_ = WallLoop {
            is_open: false,
            wall_index: 0,
            island: 0,
            unsupported: vec![false; n],
            top_surface: vec![false; n],
            arc_fraction: vec![0.0; n],
            line_widths: vec![0.4; n],
            channel_width: vec![f64::INFINITY; n],
            points: pts,
        };
        Layer {
            index,
            object: ObjectId(0),
            order,
            loops: vec![loop_],
            infill_boundary: Vec::new(),
            solid_fill_boundary: Vec::new(),
            order_field: Arc::new(manifold_fidget::order::HeightOrderField::new(DVec3::Z)),
            mesh_sdf: None,
        }
    }

    #[test]
    fn plan_tangent_surfaces_categorizes_downward_and_upward() {
        let config = SlicerConfig {
            nozzle_diameter: 0.4,
            wall_line_width: 0.4,
            print_speed: 3000.0,
            travel_speed: 9000.0,
            ..SlicerConfig::default()
        };

        // Layer 0: 10x10 at Z=0.2
        // Layer 1: 20x20 at Z=0.4 (expands beyond layer 0 -> downward tangent surface)
        // Layer 2: 10x10 at Z=0.6 (layer 1 shrinks to layer 2 -> upward tangent surface on layer 1)
        let layers = vec![
            make_test_layer(0, 0.2, 10.0),
            make_test_layer(1, 0.4, 20.0),
            make_test_layer(2, 0.6, 10.0),
        ];

        let plan = plan_tangent_surfaces(&layers, &config, ToolId(0));
        assert_eq!(plan.paths_by_layer.len(), 3);

        // Layer 1 should have both downward (Overhang) and upward (WallOuter) tangent paths
        let l1_paths = &plan.paths_by_layer[1];
        assert!(
            !l1_paths.is_empty(),
            "Layer 1 should have tangent surface paths"
        );

        let has_overhang = l1_paths
            .iter()
            .any(|p| p.segments.iter().any(|s| s.kind == MoveKind::Overhang));
        let has_wall_outer = l1_paths
            .iter()
            .any(|p| p.segments.iter().any(|s| s.kind == MoveKind::WallOuter));

        assert!(
            has_overhang,
            "Downward tangent surface should use MoveKind::Overhang"
        );
        assert!(
            has_wall_outer,
            "Upward tangent surface should use MoveKind::WallOuter"
        );
    }
}
