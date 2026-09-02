//! Wave-inspired path-planning for support-free horizontal overhangs in FDM.
//!
//! Implements Laterally Supported Overhang (LaSO) path generation based on
//! Huygens-principle wave propagation (Andersons, Sanchez, Vaneker 2024).
//!
//! Overhang regions with insufficient underlying layer support are filled with
//! continuous wavefront toolpaths that propagate outward from the supported
//! contact interface (seed curve $W_0$). Wavefronts bend naturally around
//! corners via diffraction, avoiding short-arc heat accumulation and sagging.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use glam::DVec3;
use manifold_fidget::ScalarField;
use rayon::prelude::*;

use crate::ids::ToolId;
use crate::order_field;
use crate::polygon2d;
use crate::slicing::Layer;
use crate::toolpath::{MoveKind, Path, Segment};
use crate::SlicerConfig;

/// State node for 2D Fast Marching min-heap.
#[derive(Copy, Clone, PartialEq)]
struct GridState {
    dist: f64,
    x: usize,
    y: usize,
}

impl Eq for GridState {}

impl Ord for GridState {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse for min-heap
        other
            .dist
            .partial_cmp(&self.dist)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for GridState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
enum CellTag {
    Wall,
    Far,
    Trial,
    Alive,
}

/// A 2D line segment `(p0, p1)`.
#[derive(Copy, Clone, Debug)]
pub struct LineSegment2D {
    pub p0: [f64; 2],
    pub p1: [f64; 2],
}

impl LineSegment2D {
    #[must_use]
    pub fn dist_sq_to_point(&self, p: [f64; 2]) -> f64 {
        let vx = self.p1[0] - self.p0[0];
        let vy = self.p1[1] - self.p0[1];
        let wx = p[0] - self.p0[0];
        let wy = p[1] - self.p0[1];

        let c1 = wx * vx + wy * vy;
        if c1 <= 0.0 {
            return (p[0] - self.p0[0]).powi(2) + (p[1] - self.p0[1]).powi(2);
        }
        let c2 = vx * vx + vy * vy;
        if c2 <= c1 {
            return (p[0] - self.p1[0]).powi(2) + (p[1] - self.p1[1]).powi(2);
        }
        let b = c1 / c2;
        let pb = [self.p0[0] + b * vx, self.p0[1] + b * vy];
        (p[0] - pb[0]).powi(2) + (p[1] - pb[1]).powi(2)
    }
}

/// A 2D polygon shape with an outer boundary and zero or more interior hole loops.
#[derive(Clone, Debug)]
pub struct PolygonShape2D {
    pub outer: Vec<[f64; 2]>,
    pub holes: Vec<Vec<[f64; 2]>>,
}

impl PolygonShape2D {
    /// Tests whether `pt` lies strictly inside the polygon's outer boundary and outside all its holes.
    #[must_use]
    pub fn contains_point(&self, pt: [f64; 2]) -> bool {
        if !point_in_single_loop(pt, &self.outer) {
            return false;
        }
        for hole in &self.holes {
            if point_in_single_loop(pt, hole) {
                return false;
            }
        }
        true
    }
}

/// Groups canonicalized 2D loops (outer CCW, holes CW) into distinct polygon shapes
/// where each outer loop owns its corresponding interior hole loops.
#[must_use]
pub fn group_loops_into_polygon_shapes(loops2d: &[Vec<[f64; 2]>]) -> Vec<PolygonShape2D> {
    let canonical = polygon2d::canonicalize(loops2d);
    let mut outers = Vec::new();
    let mut holes = Vec::new();

    for loop_ in canonical {
        if polygon2d::signed_area(&loop_) > 0.0 {
            outers.push(loop_);
        } else {
            holes.push(loop_);
        }
    }

    let mut shapes: Vec<PolygonShape2D> = outers
        .into_iter()
        .map(|outer| PolygonShape2D {
            outer,
            holes: Vec::new(),
        })
        .collect();

    for hole in holes {
        if hole.is_empty() {
            continue;
        }
        let sample_pt = hole[0];
        for shape in &mut shapes {
            if point_in_single_loop(sample_pt, &shape.outer) {
                shape.holes.push(hole);
                break;
            }
        }
    }

    shapes
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

/// Generates wave overhang toolpaths in 2D for an unsupported overhang polygon
/// shape (outer boundary minus holes) using 2D Huygens wavefront propagation from contact seed curves.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn generate_wave_overhang_paths_2d(
    shape: &PolygonShape2D,
    seed_segments_2d: &[LineSegment2D],
    wavelength: f64,
    config: &SlicerConfig,
) -> Vec<Vec<[f64; 2]>> {
    if shape.outer.is_empty() {
        return Vec::new();
    }

    // Compute bounding box of the outer loop
    let mut min_u = f64::INFINITY;
    let mut max_u = f64::NEG_INFINITY;
    let mut min_v = f64::INFINITY;
    let mut max_v = f64::NEG_INFINITY;

    for &[u, v] in &shape.outer {
        min_u = min_u.min(u);
        max_u = max_u.max(u);
        min_v = min_v.min(v);
        max_v = max_v.max(v);
    }

    if !min_u.is_finite() || max_u <= min_u || max_v <= min_v {
        return Vec::new();
    }

    // Grid resolution: wavelength / 4
    let grid_step = (wavelength / 4.0).clamp(0.04, 0.15);
    let pad = wavelength * 1.5;
    let u_start = min_u - pad;
    let u_end = max_u + pad;
    let v_start = min_v - pad;
    let v_end = max_v + pad;

    let nx = ((u_end - u_start) / grid_step).ceil() as usize + 1;
    let ny = ((v_end - v_start) / grid_step).ceil() as usize + 1;

    if nx < 3 || ny < 3 || nx > 2000 || ny > 2000 {
        return Vec::new();
    }

    let mut dist = vec![vec![f64::INFINITY; ny]; nx];
    let mut tag = vec![vec![CellTag::Far; ny]; nx];
    let mut heap = BinaryHeap::new();

    // 1. Initialize cells: Wall vs Inside, and compute seed distances
    let mut has_seed_points = false;
    for (i, col) in tag.iter_mut().enumerate().take(nx) {
        let u = u_start + i as f64 * grid_step;
        for (j, cell_tag) in col.iter_mut().enumerate().take(ny) {
            let v = v_start + j as f64 * grid_step;
            let pt = [u, v];

            if !shape.contains_point(pt) {
                *cell_tag = CellTag::Wall;
                continue;
            }

            // Inside overhang polygon: find distance to seed curves
            if !seed_segments_2d.is_empty() {
                let mut min_dist_sq = f64::INFINITY;
                for seg in seed_segments_2d {
                    let d_sq = seg.dist_sq_to_point(pt);
                    if d_sq < min_dist_sq {
                        min_dist_sq = d_sq;
                    }
                }
                let seed_dist = min_dist_sq.sqrt();
                if seed_dist <= grid_step * 1.0 {
                    dist[i][j] = seed_dist;
                    *cell_tag = CellTag::Alive;
                    heap.push(GridState {
                        dist: seed_dist,
                        x: i,
                        y: j,
                    });
                    has_seed_points = true;
                }
            }
        }
    }

    // Fallback if no seed points landed close enough (e.g. narrow contact)
    if !has_seed_points {
        let u_mid = (min_u + max_u) * 0.5;
        let v_mid = (min_v + max_v) * 0.5;
        let mut best_i = nx / 2;
        let mut best_j = ny / 2;
        let mut min_mid_dist = f64::INFINITY;

        for (i, col) in tag.iter().enumerate().take(nx) {
            let u = u_start + i as f64 * grid_step;
            for (j, cell_tag) in col.iter().enumerate().take(ny) {
                if *cell_tag != CellTag::Wall {
                    let v = v_start + j as f64 * grid_step;
                    let d = (u - u_mid).hypot(v - v_mid);
                    if d < min_mid_dist {
                        min_mid_dist = d;
                        best_i = i;
                        best_j = j;
                    }
                }
            }
        }

        dist[best_i][best_j] = 0.0;
        tag[best_i][best_j] = CellTag::Alive;
        heap.push(GridState {
            dist: 0.0,
            x: best_i,
            y: best_j,
        });
    }

    // 2. Fast Marching Method in 2D
    while let Some(GridState {
        dist: d_cur,
        x: cx,
        y: cy,
    }) = heap.pop()
    {
        if d_cur > dist[cx][cy] + 1e-9 {
            continue;
        }
        tag[cx][cy] = CellTag::Alive;

        // Check 4 neighbors
        let neighbors = [
            (cx.wrapping_sub(1), cy),
            (cx + 1, cy),
            (cx, cy.wrapping_sub(1)),
            (cx, cy + 1),
        ];

        for &(nx_idx, ny_idx) in &neighbors {
            if nx_idx >= nx || ny_idx >= ny || tag[nx_idx][ny_idx] == CellTag::Wall {
                continue;
            }

            // Find min distance among Alive neighbors in X and Y directions
            let mut ux = f64::INFINITY;
            if nx_idx > 0 && tag[nx_idx - 1][ny_idx] == CellTag::Alive {
                ux = ux.min(dist[nx_idx - 1][ny_idx]);
            }
            if nx_idx + 1 < nx && tag[nx_idx + 1][ny_idx] == CellTag::Alive {
                ux = ux.min(dist[nx_idx + 1][ny_idx]);
            }

            let mut uy = f64::INFINITY;
            if ny_idx > 0 && tag[nx_idx][ny_idx - 1] == CellTag::Alive {
                uy = uy.min(dist[nx_idx][ny_idx - 1]);
            }
            if ny_idx + 1 < ny && tag[nx_idx][ny_idx + 1] == CellTag::Alive {
                uy = uy.min(dist[nx_idx][ny_idx + 1]);
            }

            let new_d = if ux.is_finite() && uy.is_finite() {
                if (ux - uy).abs() < grid_step {
                    let s = ux + uy;
                    let disc = 2.0 * grid_step * grid_step - (ux - uy).powi(2);
                    (s + disc.max(0.0).sqrt()) * 0.5
                } else {
                    ux.min(uy) + grid_step
                }
            } else if ux.is_finite() {
                ux + grid_step
            } else if uy.is_finite() {
                uy + grid_step
            } else {
                continue;
            };

            if new_d < dist[nx_idx][ny_idx] {
                dist[nx_idx][ny_idx] = new_d;
                tag[nx_idx][ny_idx] = CellTag::Trial;
                heap.push(GridState {
                    dist: new_d,
                    x: nx_idx,
                    y: ny_idx,
                });
            }
        }
    }

    // 3. Extract Wavefront Isocontours at k * wavelength
    let mut max_dist = 0.0f64;
    for i in 0..nx {
        for j in 0..ny {
            if tag[i][j] == CellTag::Alive && dist[i][j].is_finite() {
                max_dist = max_dist.max(dist[i][j]);
            }
        }
    }

    if max_dist < wavelength * 0.3 {
        return Vec::new();
    }

    let mut wavefront_polylines_2d: Vec<Vec<[f64; 2]>> = Vec::new();
    let mut target_distances = Vec::new();
    if max_dist < wavelength * 1.1 {
        target_distances.push(max_dist * 0.5);
    } else {
        let mut d = wavelength;
        while d <= max_dist + wavelength * 0.2 {
            target_distances.push(d);
            d += wavelength;
        }
    }

    for target_d in target_distances {
        let isocontour_segments = extract_marching_squares_segments(
            &dist, &tag, nx, ny, u_start, v_start, grid_step, target_d,
        );
        let polylines = stitch_segments_into_polylines(isocontour_segments, grid_step * 1.5);
        for poly in polylines {
            if poly.len() >= 2 {
                let simplified = simplify_polyline_collinear(&poly, 0.015);
                if polyline_length(&simplified) >= config.nozzle_diameter * 0.75 {
                    wavefront_polylines_2d.push(simplified);
                }
            }
        }
    }

    wavefront_polylines_2d
}

/// Marching squares segment extraction for a single level set.
#[allow(clippy::too_many_arguments)]
fn extract_marching_squares_segments(
    dist: &[Vec<f64>],
    tag: &[Vec<CellTag>],
    nx: usize,
    ny: usize,
    u_start: f64,
    v_start: f64,
    step: f64,
    iso: f64,
) -> Vec<([f64; 2], [f64; 2])> {
    let mut segments = Vec::new();

    for i in 0..nx - 1 {
        let u0 = u_start + i as f64 * step;
        let u1 = u0 + step;
        for j in 0..ny - 1 {
            let v0 = v_start + j as f64 * step;
            let v1 = v0 + step;

            // All 4 corners must be Alive and valid
            if tag[i][j] != CellTag::Alive
                || tag[i + 1][j] != CellTag::Alive
                || tag[i + 1][j + 1] != CellTag::Alive
                || tag[i][j + 1] != CellTag::Alive
            {
                continue;
            }

            let d00 = dist[i][j];
            let d10 = dist[i + 1][j];
            let d11 = dist[i + 1][j + 1];
            let d01 = dist[i][j + 1];

            let mut mask = 0;
            if d00 >= iso {
                mask |= 1;
            }
            if d10 >= iso {
                mask |= 2;
            }
            if d11 >= iso {
                mask |= 4;
            }
            if d01 >= iso {
                mask |= 8;
            }

            if mask == 0 || mask == 15 {
                continue;
            }

            let edge_bottom = || {
                let t = (iso - d00) / (d10 - d00);
                [u0 + t * step, v0]
            };
            let edge_right = || {
                let t = (iso - d10) / (d11 - d10);
                [u1, v0 + t * step]
            };
            let edge_top = || {
                let t = (iso - d01) / (d11 - d01);
                [u0 + t * step, v1]
            };
            let edge_left = || {
                let t = (iso - d00) / (d01 - d00);
                [u0, v0 + t * step]
            };

            match mask {
                1 | 14 => segments.push((edge_left(), edge_bottom())),
                2 | 13 => segments.push((edge_bottom(), edge_right())),
                3 | 12 => segments.push((edge_left(), edge_right())),
                4 | 11 => segments.push((edge_right(), edge_top())),
                5 => {
                    segments.push((edge_left(), edge_top()));
                    segments.push((edge_bottom(), edge_right()));
                }
                6 | 9 => segments.push((edge_bottom(), edge_top())),
                7 | 8 => segments.push((edge_left(), edge_top())),
                10 => {
                    segments.push((edge_left(), edge_bottom()));
                    segments.push((edge_top(), edge_right()));
                }
                _ => {}
            }
        }
    }

    segments
}

/// Stitches disconnected line segments into continuous polyline chains.
fn stitch_segments_into_polylines(
    mut segments: Vec<([f64; 2], [f64; 2])>,
    tolerance: f64,
) -> Vec<Vec<[f64; 2]>> {
    let tol_sq = tolerance * tolerance;
    let mut polylines: Vec<Vec<[f64; 2]>> = Vec::new();

    while let Some((p0, p1)) = segments.pop() {
        let mut chain = vec![p0, p1];

        // Extend forward
        let mut extended = true;
        while extended {
            extended = false;
            let tip = *chain.last().unwrap();
            for i in (0..segments.len()).rev() {
                let (s0, s1) = segments[i];
                if (tip[0] - s0[0]).powi(2) + (tip[1] - s0[1]).powi(2) <= tol_sq {
                    chain.push(s1);
                    segments.swap_remove(i);
                    extended = true;
                    break;
                } else if (tip[0] - s1[0]).powi(2) + (tip[1] - s1[1]).powi(2) <= tol_sq {
                    chain.push(s0);
                    segments.swap_remove(i);
                    extended = true;
                    break;
                }
            }
        }

        // Extend backward
        let mut extended_back = true;
        while extended_back {
            extended_back = false;
            let base = chain[0];
            for i in (0..segments.len()).rev() {
                let (s0, s1) = segments[i];
                if (base[0] - s1[0]).powi(2) + (base[1] - s1[1]).powi(2) <= tol_sq {
                    chain.insert(0, s0);
                    segments.swap_remove(i);
                    extended_back = true;
                    break;
                } else if (base[0] - s0[0]).powi(2) + (base[1] - s0[1]).powi(2) <= tol_sq {
                    chain.insert(0, s1);
                    segments.swap_remove(i);
                    extended_back = true;
                    break;
                }
            }
        }

        polylines.push(chain);
    }

    polylines
}

fn polyline_length(pts: &[[f64; 2]]) -> f64 {
    let mut len = 0.0;
    for i in 0..pts.len().saturating_sub(1) {
        len += (pts[i + 1][0] - pts[i][0]).hypot(pts[i + 1][1] - pts[i][1]);
    }
    len
}

fn simplify_polyline_collinear(pts: &[[f64; 2]], eps: f64) -> Vec<[f64; 2]> {
    if pts.len() <= 2 {
        return pts.to_vec();
    }
    let mut out = vec![pts[0]];
    for i in 1..pts.len() - 1 {
        let prev = *out.last().unwrap();
        let cur = pts[i];
        let next = pts[i + 1];

        let vx = next[0] - prev[0];
        let vy = next[1] - prev[1];
        let len = vx.hypot(vy);
        if len > 1e-9 {
            let cross = ((cur[0] - prev[0]) * vy - (cur[1] - prev[1]) * vx).abs() / len;
            if cross >= eps {
                out.push(cur);
            }
        } else {
            out.push(cur);
        }
    }
    out.push(*pts.last().unwrap());
    out
}

/// Result of wave overhang path planning, containing both wave fill paths
/// and per-wall-point overhang classification tags.
#[derive(Clone, Debug, Default)]
pub struct WaveOverhangPlan {
    pub paths_by_layer: Vec<Vec<Path>>,
    pub wall_overhang_tags_by_layer: Vec<Vec<Vec<bool>>>,
}

/// Detects unsupported overhang regions across layers and generates
/// wave overhang toolpaths for each layer, along with wall contour overhang tags.
#[must_use]
pub fn plan_wave_overhangs(
    layers: &[Layer],
    config: &SlicerConfig,
    tool: ToolId,
) -> WaveOverhangPlan {
    if !config.wave_overhangs_enabled() || layers.len() < 2 {
        return WaveOverhangPlan {
            paths_by_layer: vec![Vec::new(); layers.len()],
            wall_overhang_tags_by_layer: vec![Vec::new(); layers.len()],
        };
    }

    let (axis, apex, _slope) = order_field::resolve_axis_apex_slope(config.order_field, config);
    let (basis1, basis2) = manifold_fidget::contour::plane_basis(axis);
    let origin = apex;

    let wavelength = (config.nozzle_diameter - config.wave_overhang_overlap()).max(0.10);
    let min_overhang_area = 0.25 * config.nozzle_diameter * config.nozzle_diameter;
    let max_along = order_field::max_along_for(config);
    let speed = config.wave_overhang_speed();

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

    let (wall_tags_result, paths_result): (Vec<_>, Vec<_>) = (0..layers.len())
        .into_par_iter()
        .map(|k| {
            // Find the layer physically underneath layer k
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

            // Tag wall loops for layer k: points not supported by previous layer are Overhang
            let mut layer_wall_tags = Vec::new();
            let support_dist = config.nozzle_diameter * 0.6;
            let support_dist_sq = support_dist * support_dist;
            let bed_threshold = config.first_layer_height() * 0.75;
            for wall in &layers[k].loops {
                let mut tags = vec![false; wall.points.len()];
                if let Some(prev_k) = prev_idx {
                    let prev_layer = &layers[prev_k];
                    let prev_b = &boundaries_2d[prev_k];
                    for (i, p) in wall.points.iter().enumerate() {
                        if p.z <= bed_threshold {
                            continue;
                        }

                        let p_2d = [(p - origin).dot(basis1), (p - origin).dot(basis2)];
                        let in_prev_2d = !prev_b.is_empty()
                            && polygon2d_contains_or_near(p_2d, prev_b, support_dist);

                        if !in_prev_2d {
                            let mut supported_3d = false;
                            'search: for prev_wall in &prev_layer.loops {
                                for prev_p in &prev_wall.points {
                                    if p.distance_squared(*prev_p) <= support_dist_sq {
                                        supported_3d = true;
                                        break 'search;
                                    }
                                }
                            }
                            if !supported_3d {
                                let solid_underneath =
                                    layers[k].mesh_sdf.as_ref().is_some_and(|sdf| {
                                        let probe_p =
                                            *p - DVec3::Z * (config.nozzle_diameter * 0.75);
                                        sdf.sample(probe_p).value <= config.nozzle_diameter * 0.5
                                    });
                                if !solid_underneath {
                                    tags[i] = true;
                                }
                            }
                        }
                    }
                }
                layer_wall_tags.push(tags);
            }

            let Some(prev_k) = prev_idx else {
                return (layer_wall_tags, Vec::new());
            };

            let cur_b = &boundaries_2d[k];
            let prev_b = &boundaries_2d[prev_k];

            if cur_b.is_empty() || prev_b.is_empty() {
                return (layer_wall_tags, Vec::new());
            }

            // Unsupported overhang region: cur_layer \ prev_layer
            let raw_overhang = polygon2d::difference(cur_b, prev_b);
            let overhang_filtered = polygon2d::filter_min_area(&raw_overhang, min_overhang_area);

            if overhang_filtered.is_empty() {
                return (layer_wall_tags, Vec::new());
            }

            // Group loops into outer boundaries and holes
            let shapes = group_loops_into_polygon_shapes(&overhang_filtered);

            let mut layer_polylines_2d = Vec::new();

            // 3D references for height refinement on this layer
            let mut references: Vec<Vec<DVec3>> = layers[k]
                .loops
                .iter()
                .filter(|w| w.wall_index == 0)
                .map(|w| w.points.clone())
                .collect();
            if references.is_empty() {
                references = layers[k].infill_boundary.clone();
            }

            for shape in &shapes {
                // 3D Solid Mesh Validation: Ensure the overhang shape is actually part of the solid model
                // and not an empty internal hole void.
                if let Some(sdf) = &layers[k].mesh_sdf {
                    let mut c_u = 0.0;
                    let mut c_v = 0.0;
                    for &[u, v] in &shape.outer {
                        c_u += u;
                        c_v += v;
                    }
                    let len = shape.outer.len().max(1) as f64;
                    let c_u = c_u / len;
                    let c_v = c_v / len;
                    if let Some(p_3d) = order_field::reconstruct_point_on_order_field(
                        apex + basis1 * c_u + basis2 * c_v,
                        axis,
                        layers[k].order,
                        max_along,
                        layers[k].order_field.as_ref(),
                    ) {
                        if sdf.sample(p_3d).value > 0.0 {
                            // In open air or inside a hole void - do not generate wave overhang in holes!
                            continue;
                        }
                    }
                }

                // Extract seed contact segments: edges of outer boundary bordering prev_b or other loops in cur_b
                let mut seed_segments = Vec::new();
                let n = shape.outer.len();
                let search_dist = (config.nozzle_diameter * 1.25).max(0.4);
                for i in 0..n {
                    let p0 = shape.outer[i];
                    let p1 = shape.outer[(i + 1) % n];
                    let mid = [(p0[0] + p1[0]) * 0.5, (p0[1] + p1[1]) * 0.5];

                    if polygon2d_contains_or_near(mid, prev_b, search_dist) {
                        seed_segments.push(LineSegment2D { p0, p1 });
                    }
                }

                // If no contact with prev_b, check for contact with other loops in current layer's boundary cur_b
                if seed_segments.is_empty() && cur_b.len() > 1 {
                    for i in 0..n {
                        let p0 = shape.outer[i];
                        let p1 = shape.outer[(i + 1) % n];
                        let mid = [(p0[0] + p1[0]) * 0.5, (p0[1] + p1[1]) * 0.5];

                        for other_loop in cur_b {
                            let other_shape = [other_loop.clone()];
                            if !point_in_single_loop(mid, other_loop)
                                && polygon2d_contains_or_near(mid, &other_shape, search_dist)
                            {
                                seed_segments.push(LineSegment2D { p0, p1 });
                                break;
                            }
                        }
                    }
                }

                let polylines =
                    generate_wave_overhang_paths_2d(shape, &seed_segments, wavelength, config);
                layer_polylines_2d.extend(polylines);
            }

            if layer_polylines_2d.is_empty() {
                return (layer_wall_tags, Vec::new());
            }

            // Reconstruct 2D wave polylines to 3D on the layer's order field
            let polylines_3d = order_field::reconstruct_on_order_field_near(
                layer_polylines_2d,
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

            for poly in polylines_3d {
                let pts_3d: Vec<DVec3> = if reverse {
                    poly.into_iter().rev().collect()
                } else {
                    poly
                };
                reverse = !reverse;

                if pts_3d.len() >= 2 {
                    let segment_count = pts_3d.len() - 1;
                    let segments: Vec<Segment> = (0..segment_count)
                        .map(|_| Segment {
                            kind: MoveKind::Overhang,
                            speed,
                            extrusion_rate: 1.0,
                            support_fraction: 0.0,
                            order: layers[k].order,
                            extrusion_length: 0.0,
                            line_width: config.wall_line_width,
                            is_scarf: false,
                            id: 0,
                        })
                        .collect();

                    paths.push(Path {
                        points: pts_3d,
                        segments,
                        tool,
                    });
                }
            }

            (layer_wall_tags, paths)
        })
        .unzip();

    WaveOverhangPlan {
        paths_by_layer: paths_result,
        wall_overhang_tags_by_layer: wall_tags_result,
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::ObjectId;
    use crate::slicing::WallLoop;
    use manifold_fidget::order::HeightOrderField;
    use std::sync::Arc;

    fn mock_layer(order: f64, loop_points: Vec<DVec3>) -> Layer {
        Layer {
            index: 0,
            object: ObjectId(0),
            order,
            loops: vec![WallLoop {
                wall_index: 0,
                points: loop_points.clone(),
                unsupported: vec![false; loop_points.len()],
                top_surface: vec![false; loop_points.len()],
                arc_fraction: vec![0.0; loop_points.len()],
                line_widths: vec![0.4; loop_points.len()],
            }],
            infill_boundary: vec![loop_points],
            solid_fill_boundary: Vec::new(),
            mesh_sdf: None,
            order_field: Arc::new(HeightOrderField::new(DVec3::Z)),
        }
    }

    #[test]
    fn wave_overhang_propagates_wavefronts_across_cantilever_rectangle() {
        let config = SlicerConfig {
            nozzle_diameter: 0.4,
            layer_height: 0.2,
            ..SlicerConfig::default()
        };

        let overhang_shape = PolygonShape2D {
            outer: vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
            holes: Vec::new(),
        };
        let seed = vec![LineSegment2D {
            p0: [0.0, 0.0],
            p1: [0.0, 10.0],
        }];

        let polylines = generate_wave_overhang_paths_2d(&overhang_shape, &seed, 0.35, &config);

        assert!(!polylines.is_empty(), "should generate wave paths");
        assert!(
            polylines.len() >= 15,
            "expected at least 15 wavefront passes across 10mm, got {}",
            polylines.len()
        );
    }

    #[test]
    fn wave_overhang_diffracts_around_concave_corner() {
        let config = SlicerConfig {
            nozzle_diameter: 0.4,
            layer_height: 0.2,
            ..SlicerConfig::default()
        };

        // L-shaped concave polygon: [0,0]->[10,0]->[10,5]->[5,5]->[5,10]->[0,10]
        let l_shape = PolygonShape2D {
            outer: vec![
                [0.0, 0.0],
                [10.0, 0.0],
                [10.0, 5.0],
                [5.0, 5.0],
                [5.0, 10.0],
                [0.0, 10.0],
            ],
            holes: Vec::new(),
        };

        let seed = vec![LineSegment2D {
            p0: [0.0, 0.0],
            p1: [10.0, 0.0],
        }];

        let polylines = generate_wave_overhang_paths_2d(&l_shape, &seed, 0.35, &config);

        assert!(
            !polylines.is_empty(),
            "should generate wave paths for L-shape"
        );

        let max_y_reached = polylines
            .iter()
            .flatten()
            .map(|pt| pt[1])
            .fold(0.0f64, f64::max);
        assert!(
            max_y_reached > 7.0,
            "waves should diffract around corner up to y > 7.0, reached {max_y_reached}"
        );
    }

    #[test]
    fn plan_wave_overhangs_detects_cantilever_between_layers() {
        let config = SlicerConfig {
            nozzle_diameter: 0.4,
            layer_height: 0.2,
            wave_overhangs_enabled: true,
            ..SlicerConfig::default()
        };

        let l0 = mock_layer(
            0.0,
            vec![
                DVec3::new(0.0, 0.0, 0.0),
                DVec3::new(10.0, 0.0, 0.0),
                DVec3::new(10.0, 10.0, 0.0),
                DVec3::new(0.0, 10.0, 0.0),
            ],
        );

        let mut l1 = mock_layer(
            0.2,
            vec![
                DVec3::new(0.0, 0.0, 0.2),
                DVec3::new(20.0, 0.0, 0.2),
                DVec3::new(20.0, 10.0, 0.2),
                DVec3::new(0.0, 10.0, 0.2),
            ],
        );
        l1.index = 1;

        let layers = vec![l0, l1];
        let overhang_plan = plan_wave_overhangs(&layers, &config, ToolId(0));

        assert_eq!(overhang_plan.paths_by_layer.len(), 2);
        assert!(
            overhang_plan.paths_by_layer[0].is_empty(),
            "layer 0 has no overhang"
        );
        assert!(
            !overhang_plan.paths_by_layer[1].is_empty(),
            "layer 1 should have wave overhang paths"
        );

        let min_x = overhang_plan.paths_by_layer[1]
            .iter()
            .flat_map(|p| &p.points)
            .map(|pt| pt.x)
            .fold(f64::INFINITY, f64::min);
        let max_x = overhang_plan.paths_by_layer[1]
            .iter()
            .flat_map(|p| &p.points)
            .map(|pt| pt.x)
            .fold(f64::NEG_INFINITY, f64::max);

        assert!(min_x >= 9.5, "overhang should start near x=10, got {min_x}");
        assert!(max_x <= 20.1, "overhang should end at x=20, got {max_x}");
    }

    #[test]
    fn plan_wave_overhangs_tags_contour_boundary_overhang_segments() {
        let config = SlicerConfig {
            nozzle_diameter: 0.4,
            layer_height: 0.2,
            wave_overhangs_enabled: true,
            ..SlicerConfig::default()
        };

        // Layer 0: 10x10 square (x: 0..10, y: 0..10)
        let l0 = mock_layer(
            0.0,
            vec![
                DVec3::new(0.0, 0.0, 0.0),
                DVec3::new(10.0, 0.0, 0.0),
                DVec3::new(10.0, 10.0, 0.0),
                DVec3::new(0.0, 10.0, 0.0),
            ],
        );

        // Layer 1: 20x10 rectangle (x: 0..20, y: 0..10)
        // Vertices at x=0, x=10, x=20
        let mut l1 = mock_layer(
            0.2,
            vec![
                DVec3::new(0.0, 0.0, 0.2),
                DVec3::new(10.0, 0.0, 0.2),
                DVec3::new(20.0, 0.0, 0.2),
                DVec3::new(20.0, 10.0, 0.2),
                DVec3::new(10.0, 10.0, 0.2),
                DVec3::new(0.0, 10.0, 0.2),
            ],
        );
        l1.index = 1;

        let layers = vec![l0, l1];
        let plan = plan_wave_overhangs(&layers, &config, ToolId(0));

        assert_eq!(plan.wall_overhang_tags_by_layer.len(), 2);
        let tags_l1 = &plan.wall_overhang_tags_by_layer[1][0];

        // Points at x <= 10 (supported) should be false, points at x > 10 (overhang) should be true
        assert!(!tags_l1[0], "p(0, 0) is supported");
        assert!(!tags_l1[1], "p(10, 0) is supported");
        assert!(tags_l1[2], "p(20, 0) is unsupported overhang");
        assert!(tags_l1[3], "p(20, 10) is unsupported overhang");
        assert!(!tags_l1[4], "p(10, 10) is supported");
        assert!(!tags_l1[5], "p(0, 10) is supported");
    }
}
