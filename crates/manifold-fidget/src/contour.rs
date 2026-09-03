//! Planar marching-squares contour extraction: given a [`ScalarField`]
//! sampled on a plane (reusing [`crate::slice::sample_plane`]'s
//! [`SliceGrid`](crate::slice::SliceGrid)/[`grid_point`](crate::slice::grid_point)),
//! extract the field's zero-crossing (or arbitrary `iso`) contours as closed
//! polylines in world space.
//!
//! This is the 2D analogue of [`crate::marching_cubes::extract_isosurface`]
//! (3D marching cubes -> triangle soup); here the classic 2D
//! marching-squares case table produces line segments, which are then
//! stitched into closed loops (see [`extract_contours`]'s doc for the
//! stitching approach).

use std::collections::{HashMap, HashSet};

use glam::DVec3;

use crate::order::OrderField;
use crate::slice::{grid_point, sample_plane};
use crate::ScalarField;

/// Quantization scale used to key segment endpoints for loop-stitching (see
/// [`extract_contours`]): endpoints from adjacent marching-squares cells
/// are computed from the same shared-edge corner values, but the shared
/// edge is walked in opposite corner order by each of its two cells (e.g.
/// one cell's top edge is another's bottom edge, with endpoints swapped),
/// so floating-point non-associativity can produce a tiny (sub-ULP-times-a
/// bit) difference between the two computed endpoints. This tolerance is
/// loose enough to treat those as the same point while still being far
/// tighter than any of this module's geometric assertions.
const STITCH_EPSILON: f64 = 1e-6;

/// Extracts the `field(p) == iso` contours of `field` over the plane defined
/// by `origin`/`basis1`/`basis2` (same parameterization as
/// [`crate::slice::sample_plane`]), as closed polylines in world space.
///
/// `resolution_u`/`resolution_v` are the number of grid *samples* along
/// `basis1`/`basis2` (must both be `>= 2`, else there are no cells to run
/// marching squares over and an empty `Vec` is returned).
///
/// Returns `Vec<Vec<DVec3>>`: each inner `Vec` is one closed loop, listed in
/// segment-walk order, with the first point *not* repeated at the end
/// (i.e. the loop's implicit closing edge is `last -> first`).
///
/// ## Algorithm
///
/// 1. Sample `field` on the plane via [`sample_plane`] (reusing the same
///    grid-sampling logic as the slice heatmap visualizer).
/// 2. Run the standard 2D marching-squares case table over each of the
///    `(resolution_u - 1) * (resolution_v - 1)` cells between adjacent grid
///    samples, linearly interpolating the crossing point along each cut
///    cell edge. Saddle cases (all-corners-alternate) are resolved by the
///    average-of-corners heuristic (compare the 4 corner values' mean
///    against `iso`).
/// 3. Stitch the resulting line segments into closed loops by matching
///    each segment's end point to the next segment's start point (segments
///    from adjacent cells sharing an edge produce identical endpoints,
///    since both are interpolated from the same pair of corner values).
///    Any segment chain that does not close back on itself (e.g. a contour
///    clipped by the plane's finite extent) is still emitted as an open
///    "loop" best-effort, since the common case (a closed solid fully
///    inside the sampled plane extent) always closes.
/// 4. Orientations are canonicalized by nesting depth (see
///    [`canonicalize_orientation`]): a loop directly enclosing the sampled
///    solid (even containment depth, e.g. an outer boundary) winds
///    counter-clockwise in the `(basis1, basis2)` plane, and a loop
///    bounding a hole (odd depth) winds clockwise — needed because
///    per-cell marching-squares segments aren't stitched with a
///    consistent direction (see [`stitch_loops`]), so without this pass a
///    hole boundary can come back with the *same* winding as its
///    enclosing outer boundary, which downstream nonzero-fill-rule
///    polygon booleans (e.g. `i_overlay`) would then treat as solid
///    material rather than a hole.
#[allow(clippy::too_many_arguments)]
pub fn extract_contours<F: ScalarField + ?Sized>(
    field: &F,
    origin: DVec3,
    basis1: DVec3,
    basis2: DVec3,
    width: f64,
    height: f64,
    resolution_u: usize,
    resolution_v: usize,
    iso: f64,
) -> Vec<Vec<DVec3>> {
    if resolution_u < 2 || resolution_v < 2 {
        return Vec::new();
    }

    let grid = sample_plane(
        field,
        origin,
        basis1,
        basis2,
        width,
        height,
        resolution_u,
        resolution_v,
    );

    let position = |col: usize, row: usize| -> DVec3 {
        grid_point(
            origin,
            basis1,
            basis2,
            width,
            height,
            resolution_u,
            resolution_v,
            col,
            row,
        )
    };
    let value = |col: usize, row: usize| -> f64 { grid.values[row * grid.width + col] as f64 };

    // Edge indices around a cell: 0 = bottom (bl-br), 1 = right (br-tr),
    // 2 = top (tr-tl), 3 = left (tl-bl).
    let mut segments: Vec<(DVec3, DVec3)> = Vec::new();

    for row in 0..resolution_v - 1 {
        for col in 0..resolution_u - 1 {
            let v_bl = value(col, row);
            let v_br = value(col + 1, row);
            let v_tr = value(col + 1, row + 1);
            let v_tl = value(col, row + 1);

            let mut case_index = 0usize;
            if v_bl < iso {
                case_index |= 1;
            }
            if v_br < iso {
                case_index |= 2;
            }
            if v_tr < iso {
                case_index |= 4;
            }
            if v_tl < iso {
                case_index |= 8;
            }

            if case_index == 0 || case_index == 15 {
                continue;
            }

            let p_bl = position(col, row);
            let p_br = position(col + 1, row);
            let p_tr = position(col + 1, row + 1);
            let p_tl = position(col, row + 1);

            // Lazily-interpolated crossing point per edge (only computed
            // for edges an active case actually references).
            let edge_point = |edge: usize| -> DVec3 {
                match edge {
                    0 => lerp_crossing(p_bl, v_bl, p_br, v_br, iso),
                    1 => lerp_crossing(p_br, v_br, p_tr, v_tr, iso),
                    2 => lerp_crossing(p_tr, v_tr, p_tl, v_tl, iso),
                    3 => lerp_crossing(p_tl, v_tl, p_bl, v_bl, iso),
                    _ => unreachable!("marching-squares cells only have 4 edges"),
                }
            };

            for &(e0, e1) in case_segments(case_index, v_bl, v_br, v_tr, v_tl, iso) {
                segments.push((edge_point(e0), edge_point(e1)));
            }
        }
    }

    canonicalize_orientation(stitch_loops(segments), basis1, basis2)
}

/// Linearly interpolates the position where the field crosses `iso` along
/// the edge from `(p0, v0)` to `(p1, v1)`.
fn lerp_crossing(p0: DVec3, v0: f64, p1: DVec3, v1: f64, iso: f64) -> DVec3 {
    if !v0.is_finite() || !v1.is_finite() {
        // One endpoint's field value is non-finite (e.g. an
        // `EikonalOrderField` query landing outside the FMM front's
        // reached/occupied region, where `order` returns `f64::INFINITY`
        // by design — see `EikonalOrderField::order`). The two callers of
        // this function only invoke it once they've already detected a
        // sign change across `iso`, so there is a real crossing somewhere
        // on this edge, but `(iso - v0) / (v1 - v0)` here would be an
        // `inf/inf`- or `-inf/-inf`-shaped NaN, not a usable `t`. Fall back
        // to the midpoint, the same "no better information" default the
        // near-zero-denominator branch below already uses.
        return p0.lerp(p1, 0.5);
    }
    let denom = v1 - v0;
    let t = if denom.abs() <= f64::EPSILON {
        0.5
    } else {
        ((iso - v0) / denom).clamp(0.0, 1.0)
    };
    p0.lerp(p1, t)
}

/// Returns the edge-pair segments for a marching-squares cell configuration,
/// given its 4 corner values (needed only to resolve the two ambiguous
/// saddle cases, 5 and 10, via the average-of-corners heuristic).
///
/// Edge indices: 0 = bottom, 1 = right, 2 = top, 3 = left (see
/// [`extract_contours`]).
fn case_segments(
    case_index: usize,
    v_bl: f64,
    v_br: f64,
    v_tr: f64,
    v_tl: f64,
    iso: f64,
) -> &'static [(usize, usize)] {
    match case_index {
        1 => &[(3, 0)],
        2 => &[(0, 1)],
        3 => &[(3, 1)],
        4 => &[(1, 2)],
        5 => {
            // Saddle: bl and tr inside, br and tl outside.
            let average = (v_bl + v_br + v_tr + v_tl) / 4.0;
            if average < iso {
                &[(3, 2), (0, 1)]
            } else {
                &[(3, 0), (1, 2)]
            }
        }
        6 => &[(0, 2)],
        7 => &[(3, 2)],
        8 => &[(2, 3)],
        9 => &[(0, 2)],
        10 => {
            // Saddle: br and tl inside, bl and tr outside.
            let average = (v_bl + v_br + v_tr + v_tl) / 4.0;
            if average < iso {
                &[(0, 1), (2, 3)]
            } else {
                &[(0, 3), (1, 2)]
            }
        }
        11 => &[(1, 2)],
        12 => &[(1, 3)],
        13 => &[(0, 1)],
        14 => &[(3, 0)],
        _ => &[],
    }
}

/// Quantizes a point to an integer key (scaled by [`STITCH_EPSILON`]) for
/// exact-match endpoint stitching.
fn point_key(p: DVec3) -> (i64, i64, i64) {
    let scale = 1.0 / STITCH_EPSILON;
    (
        (p.x * scale).round() as i64,
        (p.y * scale).round() as i64,
        (p.z * scale).round() as i64,
    )
}

/// Widened matching tolerance used only as a repair pass when the exact
/// quantized-key stitch (`STITCH_EPSILON`, exact-match-only) fails to find
/// a partner for the current chain endpoint. Floating-point drift between
/// independently-lerped points on a shared edge (e.g. from
/// `canonicalize_orientation_per_loop_basis`'s per-loop basis fitting, or
/// the marching-squares saddle-case average-of-corners heuristic) can leave
/// two endpoints that are geometrically the same point but a few ULPs
/// apart -- far tighter than any real, distinct topological gap on typical
/// mesh scales, so widening this far does not risk bridging a genuine
/// stitching gap.
const STITCH_REPAIR_TOLERANCE: f64 = 4.0;

/// Minimum point count for a stitched chain to be a plausible closed
/// contour: a real closed polygon needs at least 3 distinct points/segments
/// to bound any area (a triangle).
const MIN_LOOP_POINTS: usize = 3;

/// Finds the nearest not-yet-used segment with an endpoint within
/// [`STITCH_REPAIR_TOLERANCE`] of `point`, used only after the exact-match
/// lookup in [`stitch_loops`] fails.
fn find_nearest_unused_endpoint(
    segments: &[(DVec3, DVec3)],
    used: &[bool],
    point: DVec3,
) -> Option<usize> {
    let mut best: Option<(usize, f64)> = None;
    for (i, &(a, b)) in segments.iter().enumerate() {
        if used[i] {
            continue;
        }
        let d = a.distance(point).min(b.distance(point));
        if d <= STITCH_REPAIR_TOLERANCE && best.is_none_or(|(_, best_d)| d < best_d) {
            best = Some((i, d));
        }
    }
    best.map(|(i, _)| i)
}

/// Stitches unordered line segments (from [`extract_contours`]'s
/// marching-squares pass) into closed polylines.
///
/// Segments are treated as undirected: a marching-squares case's segment
/// orientation is not guaranteed to be consistent between adjacent cells
/// (e.g. one cell's edge walk can produce a segment in the opposite
/// direction from its neighbor's matching segment), so stitching matches
/// on *either* endpoint of the next candidate segment, not just its start.
///
/// If the exact-match stitch cannot find a partner for the current chain
/// endpoint, a widened-tolerance repair pass (see
/// [`STITCH_REPAIR_TOLERANCE`]) is tried before giving up. Chains that
/// still cannot be closed -- or that close with too few points to bound
/// any area (see [`MIN_LOOP_POINTS`]) -- are dropped rather than emitted
/// as printable geometry: a degenerate 1-3 point "loop" reaching
/// `toolpath::plan` as a `WallLoop` produces a nonsensical G-code path,
/// which is worse than an omission that is at least visible via the
/// `tracing::warn!` below.
fn stitch_loops(segments: Vec<(DVec3, DVec3)>) -> Vec<Vec<DVec3>> {
    // Deduplicate undirected segments and drop degenerate zero-length segments:
    // duplicate edges (e.g. from adjacent triangles sharing an On-On plateau edge
    // in `extract_order_contours_on_mesh`) create false degree-3+ vertices that
    // cause loop-tracing to self-intersect or terminate prematurely.
    let mut seen = HashSet::new();
    let mut segments: Vec<(DVec3, DVec3)> = segments
        .into_iter()
        .filter(|&(a, b)| {
            let ka = point_key(a);
            let kb = point_key(b);
            if ka == kb {
                return false;
            }
            let seg_key = if ka <= kb { (ka, kb) } else { (kb, ka) };
            seen.insert(seg_key)
        })
        .collect();

    // Prune dangling degree-1 endpoints (open-ended hair/spurs with no exact or
    // repair-tolerance partner): any segment with an unpartnered endpoint cannot belong
    // to a closed cycle, and walking into a dangling spur causes loop-stitching to fail
    // to close and falsely mark entire valid cycles as dead starts.
    loop {
        let mut by_point: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
        for (i, &(a, b)) in segments.iter().enumerate() {
            by_point.entry(point_key(a)).or_default().push(i);
            by_point.entry(point_key(b)).or_default().push(i);
        }
        let before_len = segments.len();
        let has_partner = |idx: usize, pt: DVec3, segs: &[(DVec3, DVec3)]| -> bool {
            if let Some(cand) = by_point.get(&point_key(pt)) {
                if cand.iter().any(|&i| i != idx) {
                    return true;
                }
            }
            segs.iter().enumerate().any(|(i, &(a, b))| {
                i != idx
                    && (a.distance(pt) <= STITCH_REPAIR_TOLERANCE
                        || b.distance(pt) <= STITCH_REPAIR_TOLERANCE)
            })
        };
        let keep_mask: Vec<bool> = segments
            .iter()
            .enumerate()
            .map(|(i, &(a, b))| has_partner(i, a, &segments) && has_partner(i, b, &segments))
            .collect();
        let mut idx = 0;
        segments.retain(|_| {
            let keep = keep_mask[idx];
            idx += 1;
            keep
        });
        if segments.len() == before_len {
            break;
        }
    }

    stitch_loops_with_debug(segments).0
}

/// Stitches unordered line segments into closed polylines and returns both
/// `(closed_loops, unclosed_debug_fragments)`.
pub fn stitch_loops_with_debug(
    segments: Vec<(DVec3, DVec3)>,
) -> (Vec<Vec<DVec3>>, Vec<Vec<DVec3>>) {
    // Deduplicate undirected segments and drop degenerate zero-length segments:
    // duplicate edges (e.g. from adjacent triangles sharing an On-On plateau edge
    // in `extract_order_contours_on_mesh`) create false degree-3+ vertices that
    // cause loop-tracing to self-intersect or terminate prematurely.
    let mut seen = HashSet::new();
    let segments: Vec<(DVec3, DVec3)> = segments
        .into_iter()
        .filter(|&(a, b)| {
            let ka = point_key(a);
            let kb = point_key(b);
            if ka == kb {
                return false;
            }
            let seg_key = if ka <= kb { (ka, kb) } else { (kb, ka) };
            seen.insert(seg_key)
        })
        .collect();

    // Map from a (quantized) endpoint key to the indices of segments
    // touching that point (a segment appears under both of its endpoints'
    // keys, unless they collide, e.g. a degenerate zero-length segment).
    let mut by_point: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
    for (i, &(a, b)) in segments.iter().enumerate() {
        by_point.entry(point_key(a)).or_default().push(i);
        by_point.entry(point_key(b)).or_default().push(i);
    }
    let mut used = vec![false; segments.len()];
    let mut dead_start = vec![false; segments.len()];

    let mut loops = Vec::new();
    let mut debug_unclosed = Vec::new();
    let mut dropped_fragments = 0usize;
    let mut dropped_points = 0usize;

    for start_idx in 0..segments.len() {
        if used[start_idx] || dead_start[start_idx] {
            continue;
        }
        let mut trail = vec![start_idx];
        used[start_idx] = true;
        let (first_point, mut current_point) = segments[start_idx];
        let mut loop_points = vec![first_point];
        let mut closed = false;

        loop {
            if let Some(pos) = loop_points
                .iter()
                .position(|&p| point_key(p) == point_key(current_point))
            {
                let cycle: Vec<DVec3> = loop_points[pos..].to_vec();
                if cycle.len() >= MIN_LOOP_POINTS {
                    loops.push(cycle);
                    closed = true;
                }
                break;
            }

            let next = by_point
                .get(&point_key(current_point))
                .and_then(|candidates| candidates.iter().copied().find(|&i| !used[i]))
                .or_else(|| find_nearest_unused_endpoint(&segments, &used, current_point));

            match next {
                Some(next_idx) => {
                    used[next_idx] = true;
                    trail.push(next_idx);
                    let (a, b) = segments[next_idx];
                    let other = if a.distance(current_point) <= b.distance(current_point) {
                        b
                    } else {
                        a
                    };
                    loop_points.push(current_point);
                    current_point = other;
                }
                None => {
                    if let Some(pos) = loop_points
                        .iter()
                        .position(|&p| p.distance(current_point) <= STITCH_REPAIR_TOLERANCE)
                    {
                        let cycle: Vec<DVec3> = loop_points[pos..].to_vec();
                        if cycle.len() >= MIN_LOOP_POINTS {
                            loops.push(cycle);
                            closed = true;
                        }
                    } else {
                        loop_points.push(current_point);
                    }
                    break;
                }
            }
        }

        if !closed {
            for &idx in &trail {
                used[idx] = false;
            }
            dead_start[start_idx] = true;
            dropped_fragments += 1;
            dropped_points += loop_points.len();
        }
    }

    // Second pass: any remaining unused segments that could not form closed loops
    // are collected once as unclosed debug chains (each segment visited exactly once).
    for start_idx in 0..segments.len() {
        if used[start_idx] {
            continue;
        }
        used[start_idx] = true;
        let (first_point, mut current_point) = segments[start_idx];
        let mut chain = vec![first_point];
        loop {
            let next = by_point
                .get(&point_key(current_point))
                .and_then(|candidates| candidates.iter().copied().find(|&i| !used[i]));
            match next {
                Some(next_idx) => {
                    used[next_idx] = true;
                    let (a, b) = segments[next_idx];
                    let other = if a.distance(current_point) <= b.distance(current_point) {
                        b
                    } else {
                        a
                    };
                    chain.push(current_point);
                    current_point = other;
                }
                None => {
                    chain.push(current_point);
                    break;
                }
            }
        }
        if chain.len() >= 2 {
            debug_unclosed.push(chain);
        }
    }

    if dropped_fragments > 0 {
        tracing::warn!(
            dropped_fragments,
            dropped_points,
            "stitch_loops: preserved {dropped_fragments} open/degenerate contour \
             fragment(s) ({dropped_points} points total) as debug paths",
        );
    }

    (loops, debug_unclosed)
}

/// Fixes up loop winding so nesting parity determines orientation: a loop
/// at even containment depth (not inside any other loop, or inside an
/// even number of them) winds counter-clockwise in the `(basis1, basis2)`
/// plane, and a loop at odd depth (bounding a hole one level in, or three
/// levels in, etc.) winds clockwise. Needed because [`stitch_loops`]
/// doesn't guarantee a consistent per-loop direction (see its docs), so
/// raw output can have a hole boundary sharing its enclosing outer
/// boundary's winding — which a nonzero-fill-rule polygon boolean (e.g.
/// `i_overlay`) would then read as solid material, not a hole.
///
/// Containment is decided by testing each loop's first point against
/// every other loop with a standard even-odd ray cast; `O(n^2)` in loop
/// count, which is fine for the small per-layer loop counts this is
/// applied to.
fn canonicalize_orientation(
    loops: Vec<Vec<DVec3>>,
    basis1: DVec3,
    basis2: DVec3,
) -> Vec<Vec<DVec3>> {
    if loops.len() <= 1 {
        return loops;
    }

    let projected: Vec<Vec<(f64, f64)>> = loops
        .iter()
        .map(|l| l.iter().map(|p| (p.dot(basis1), p.dot(basis2))).collect())
        .collect();

    let depths: Vec<usize> = (0..loops.len())
        .map(|i| {
            if projected[i].is_empty() {
                return 0;
            }
            let test_point = projected[i][0];
            (0..loops.len())
                .filter(|&j| j != i && point_in_polygon(test_point, &projected[j]))
                .count()
        })
        .collect();

    loops
        .into_iter()
        .zip(projected)
        .zip(depths)
        .map(|((loop_points, poly2d), depth)| {
            let want_ccw = depth % 2 == 0;
            let is_ccw = signed_area(&poly2d) > 0.0;
            if want_ccw == is_ccw {
                loop_points
            } else {
                loop_points.into_iter().rev().collect()
            }
        })
        .collect()
}

/// Signed area (shoelace formula) of a closed 2D polygon; positive for
/// counter-clockwise winding, negative for clockwise. `< 3` points yields
/// `0.0`.
fn signed_area(points: &[(f64, f64)]) -> f64 {
    let n = points.len();
    if n < 3 {
        return 0.0;
    }
    let mut area = 0.0;
    for i in 0..n {
        let (x0, y0) = points[i];
        let (x1, y1) = points[(i + 1) % n];
        area += x0 * y1 - x1 * y0;
    }
    area * 0.5
}

/// Standard even-odd ray-casting point-in-polygon test (casts along `+u`
/// from `point`, counting crossings of `polygon`'s edges). `< 3` points
/// yields `false`.
fn point_in_polygon(point: (f64, f64), polygon: &[(f64, f64)]) -> bool {
    let n = polygon.len();
    if n < 3 {
        return false;
    }
    let (px, py) = point;
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = polygon[i];
        let (xj, yj) = polygon[j];
        if (yi > py) != (yj > py) && px < (xj - xi) * (py - yi) / (yj - yi) + xi {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Estimates a closed loop's centroid (simple vertex average — sufficient
/// as a local-basis origin, not intended to be the area centroid) and a
/// best-fit normal via Newell's method (the standard robust way to get a
/// consistent normal direction for a possibly-non-planar polygon: it's the
/// exact face normal when the loop *is* planar, and a reasonable
/// least-squares-ish approximation when it's only nearly planar, which is
/// the case this is built for — a curved-order-field contour loop is
/// exactly planar only for [`crate::order::HeightOrderField`]-style fields,
/// but stays *locally* near-planar for the concentric/nested loops produced
/// by fields like [`crate::order::ConicalOrderField`], since such loops are
/// isosurface cross-sections at a single order value sharing a common
/// apex/axis). Returns `(centroid, normal)`; `normal` is `DVec3::Z` as an
/// arbitrary fallback for degenerate input (`< 3` points, or a normal too
/// small to normalize, e.g. a collinear/zero-area loop).
///
/// Exposed (rather than kept private) so callers that need a per-loop
/// best-fit local 2D projection/basis for a loop that may not lie in any
/// single shared plane (e.g. `manifold-core::polygon2d`'s inward-offset
/// step, which needs a per-loop local basis for curved-order-field contour
/// loops instead of one shared `(basis1, basis2)` plane per layer) can
/// combine this with [`plane_basis`] — `plane_basis(normal)` after calling
/// this — the exact same pairing [`canonicalize_orientation_per_loop_basis`]
/// uses internally, instead of duplicating this centroid/normal-estimation
/// logic.
pub fn loop_centroid_and_normal(loop_points: &[DVec3]) -> (DVec3, DVec3) {
    let n = loop_points.len();
    if n < 3 {
        let centroid = loop_points.iter().copied().sum::<DVec3>() / (n.max(1) as f64);
        return (centroid, DVec3::Z);
    }

    let centroid = loop_points.iter().copied().sum::<DVec3>() / n as f64;

    let mut normal = DVec3::ZERO;
    for i in 0..n {
        let current = loop_points[i];
        let next = loop_points[(i + 1) % n];
        normal.x += (current.y - next.y) * (current.z + next.z);
        normal.y += (current.z - next.z) * (current.x + next.x);
        normal.z += (current.x - next.x) * (current.y + next.y);
    }

    let normal = if normal.length_squared() > f64::EPSILON {
        normal.normalize()
    } else {
        DVec3::Z
    };

    (centroid, normal)
}

/// 3D-aware analogue of [`canonicalize_orientation`] for loops that may not
/// share a single common plane (e.g. contours from a curved
/// [`crate::order::OrderField`] like [`crate::order::ConicalOrderField`],
/// extracted by [`extract_order_contours_on_mesh`]).
///
/// Rather than projecting every loop into one shared `(basis1, basis2)`
/// plane (which [`canonicalize_orientation`] does, and which is only valid
/// when every loop in the layer is coplanar), each loop gets its *own*
/// best-fit local origin/basis (see [`loop_centroid_and_normal`] +
/// [`plane_basis`]) — with the Newell normal's sign first aligned against
/// `reference` (flipped when `normal.dot(reference) < 0`). That alignment
/// is what anchors the result to a *global* winding convention: Newell's
/// normal follows the loop's (arbitrary, stitch-order-dependent) point
/// order, so "CCW in the loop's own raw frame" is a tautology — without an
/// external reference every loop would simply keep/flip relative to its
/// own arbitrary orientation, and outer loops vs holes could come out with
/// the *same* effective winding when later projected into a shared plane
/// (which erases holes from any downstream NonZero-fill polygon boolean —
/// exactly what happened on hole-through-the-base layers where the plateau
/// edge path stitches the outer loop in the opposite direction).
/// `reference` should be the direction downstream consumers project along
/// (e.g. the build direction for slicing's shared-plane 2D offsets).
///
/// Nesting depth is decided by projecting a test point
/// from loop `i` into loop `j`'s *own* local basis before running
/// [`point_in_polygon`] against loop `j`'s own points (also projected into
/// that same local basis) — i.e. every containment test is done in
/// whichever loop's frame is being tested against, not in one global
/// frame. This is exact for the concentric-nested-loops case this is built
/// for (loops from the same isosurface at the same order value, nested
/// around a common axis/apex — each individually near-planar even though
/// the whole layer is not one shared plane), and degrades gracefully (not
/// necessarily correctly) for a genuinely non-nested, wildly non-planar
/// arrangement, which the order fields this codebase implements do not
/// produce.
fn canonicalize_orientation_per_loop_basis(
    loops: Vec<Vec<DVec3>>,
    reference: DVec3,
) -> Vec<Vec<DVec3>> {
    if loops.is_empty() {
        return loops;
    }

    // Each loop's own (centroid, basis1, basis2) frame — normal sign
    // aligned with `reference` so "positive signed area" means the same
    // global winding for every loop — and its own points projected into
    // that frame.
    let frames: Vec<(DVec3, DVec3, DVec3)> = loops
        .iter()
        .map(|l| {
            let (centroid, normal) = loop_centroid_and_normal(l);
            let aligned = if normal.dot(reference) < 0.0 {
                -normal
            } else {
                normal
            };
            let (basis1, basis2) = plane_basis(aligned);
            (centroid, basis1, basis2)
        })
        .collect();

    let project = |p: DVec3, frame: &(DVec3, DVec3, DVec3)| -> (f64, f64) {
        let (centroid, basis1, basis2) = *frame;
        let d = p - centroid;
        (d.dot(basis1), d.dot(basis2))
    };

    let self_projected: Vec<Vec<(f64, f64)>> = loops
        .iter()
        .zip(&frames)
        .map(|(l, frame)| l.iter().map(|&p| project(p, frame)).collect())
        .collect();

    let depths: Vec<usize> = (0..loops.len())
        .map(|i| {
            if loops[i].is_empty() {
                return 0;
            }
            let test_point = loops[i][0];
            (0..loops.len())
                .filter(|&j| {
                    j != i && point_in_polygon(project(test_point, &frames[j]), &self_projected[j])
                })
                .count()
        })
        .collect();

    loops
        .into_iter()
        .zip(self_projected)
        .zip(depths)
        .map(|((loop_points, poly2d), depth)| {
            let want_ccw = depth % 2 == 0;
            let is_ccw = signed_area(&poly2d) > 0.0;
            if want_ccw == is_ccw {
                loop_points
            } else {
                loop_points.into_iter().rev().collect()
            }
        })
        .collect()
}

/// Extracts the contour of `field` at the plane `{ p : p.dot(direction) ==
/// order_value }`, deriving the plane's origin and in-plane basis generally
/// from `direction` (never hardcoded to a fixed axis), so a future
/// non-height order field (e.g. an angle/Eikonal-driven field, see
/// `NON_PLANAR_SLICING.md`) does not force a signature rewrite here.
///
/// `direction` must be a unit vector (not renormalized defensively, to
/// match `HeightOrderField`'s documented precondition in `order.rs`).
///
/// `width`/`height`/`resolution_u`/`resolution_v` parameterize the sampled
/// extent and grid density of the plane, same as [`extract_contours`].
#[allow(clippy::too_many_arguments)]
pub fn extract_contours_at_order<F: ScalarField + ?Sized>(
    field: &F,
    direction: DVec3,
    order_value: f64,
    width: f64,
    height: f64,
    resolution_u: usize,
    resolution_v: usize,
) -> Vec<Vec<DVec3>> {
    let origin = direction * order_value;
    let (basis1, basis2) = plane_basis(direction);
    extract_contours(
        field,
        origin,
        basis1,
        basis2,
        width,
        height,
        resolution_u,
        resolution_v,
        0.0,
    )
}

/// Extracts polylines (open or closed) at `order_field(p) == order_value`
/// from a triangle soup (e.g. [`crate::marching_cubes::extract_isosurface`]'s
/// output), by walking each triangle's edges for an `order_value` crossing
/// and stitching the resulting segments the same way [`extract_contours`]
/// stitches its marching-squares segments.
///
/// Unlike [`extract_contours_at_order`], this does not assume the contour
/// lies in a single plane: `order_field` may be a curved (e.g.
/// [`crate::order::ConicalOrderField`]) order field, so winding/nesting is
/// canonicalized via [`canonicalize_orientation_per_loop_basis`] rather
/// than [`canonicalize_orientation`] — each loop gets its own best-fit
/// local 2D projection (see that function's docs) instead of assuming one
/// shared plane basis for the whole layer, which is exact for the
/// concentric-nested-around-a-common-axis loops this codebase's order
/// fields (`HeightOrderField`, `ConicalOrderField`) actually produce.
/// `reference` anchors every loop's winding to a global convention (see
/// that function's docs — pass the direction downstream consumers project
/// along, e.g. the slicer's build direction): outer loops come back CCW
/// and holes CW when viewed along `reference`.
///
/// `triangle_positions` is a flat slice where every 3 consecutive entries
/// form one triangle (matching
/// [`crate::marching_cubes::Vertex::position`](crate::marching_cubes::Vertex)'s
/// layout — pass `.iter().map(|v| v.position).collect::<Vec<_>>()` when
/// using `extract_isosurface`'s output directly).
///
/// A vertex within a small relative epsilon of `order_value` is treated as
/// lying *on* the contour rather than strictly above or below it. This
/// matters beyond the single-point "exact match" case: an
/// [`crate::eikonal::EikonalOrderField`] seeded over a broad flat region
/// (see
/// [`crate::eikonal::EikonalOrderField::new_with_occupancy_and_seed_region`])
/// marches as a perfectly planar front through unobstructed flat geometry
/// (e.g. a flat plate-contact base), so an entire connected patch of the
/// wall isosurface can sit almost exactly at `order_value` at once — not
/// just an isolated vertex. A strict `(v0-order_value)*(v1-order_value) <
/// 0.0` sign test never fires across such a patch (nothing is strictly on
/// the other side), which used to silently drop the whole plateau's
/// boundary and could zero out an entire layer's contour. Instead:
///
/// - An edge between two "on" vertices contributes no crossing point
///   itself (it lies flat on the contour).
/// - An edge between an "on" vertex and a strictly-above/below vertex
///   contributes the "on" vertex's own position as the crossing point (no
///   interpolation needed — it's already at the target value). A triangle
///   with exactly two "on" vertices and one strictly-above/below vertex
///   therefore yields those two "on" vertices as its crossing pair — the
///   plateau's own edge, which is exactly the boundary the layer's contour
///   needs there.
/// - A triangle with all three vertices "on" is fully interior to the
///   plateau and contributes nothing (matches the prior degenerate-case
///   handling, just generalized from exact equality to a small epsilon).
/// - An edge between two vertices strictly on the same side (both above or
///   both below) contributes nothing, as before.
///
/// Duplicate crossing points from the same "on" vertex touching two edges
/// of a triangle are deduplicated before deciding whether the triangle
/// yields a usable segment.
pub fn extract_order_contours_on_mesh<O: OrderField + ?Sized>(
    triangle_positions: &[DVec3],
    order_field: &O,
    order_value: f64,
    reference: DVec3,
) -> Vec<Vec<DVec3>> {
    let orders: Vec<f64> = triangle_positions
        .iter()
        .map(|&p| order_field.order(p))
        .collect();
    extract_order_contours_on_mesh_with_orders(triangle_positions, &orders, order_value, reference)
}

pub fn extract_order_contours_on_mesh_with_debug(
    triangle_positions: &[DVec3],
    triangle_orders: &[f64],
    order_value: f64,
    reference: DVec3,
) -> (Vec<Vec<DVec3>>, Vec<Vec<DVec3>>) {
    let mut segments: Vec<(DVec3, DVec3)> = Vec::new();

    // Deterministic infinitesimal perturbation (Simulation of Simplicity).
    // Offsets the slicing isovalue by 1e-5 to prevent flat horizontal plateaus
    // from fracturing into dozens of degenerate micro-triangles due to floating-point noise.
    let target_order = order_value + 1e-5 * order_value.abs().max(1.0);

    for (tri, v_chunk) in triangle_positions
        .chunks_exact(3)
        .zip(triangle_orders.chunks_exact(3))
    {
        let p = [tri[0], tri[1], tri[2]];
        let v = [v_chunk[0], v_chunk[1], v_chunk[2]];

        if !v[0].is_finite() || !v[1].is_finite() || !v[2].is_finite() {
            continue;
        }

        let above = [
            v[0] > target_order,
            v[1] > target_order,
            v[2] > target_order,
        ];

        if (above[0] == above[1]) && (above[1] == above[2]) {
            continue;
        }

        let mut crossing_points: Vec<DVec3> = Vec::with_capacity(2);
        for &(i0, i1) in &[(0, 1), (1, 2), (2, 0)] {
            if above[i0] != above[i1] {
                let p0 = p[i0];
                let p1 = p[i1];
                let v0 = v[i0];
                let v1 = v[i1];
                let edge_len = p0.distance(p1);
                if edge_len > 1e-6 {
                    crossing_points.push(lerp_crossing(p0, v0, p1, v1, target_order));
                }
            }
        }

        if crossing_points.len() == 2 {
            let ka = point_key(crossing_points[0]);
            let kb = point_key(crossing_points[1]);
            if ka != kb {
                segments.push((crossing_points[0], crossing_points[1]));
            }
        }
    }

    let (closed, debug_unclosed) = stitch_loops_with_debug(segments);
    (
        canonicalize_orientation_per_loop_basis(closed, reference),
        debug_unclosed,
    )
}

/// Extracts polylines at `order_value` from `triangle_positions` with precomputed per-vertex `triangle_orders`.
pub fn extract_order_contours_on_mesh_with_orders(
    triangle_positions: &[DVec3],
    triangle_orders: &[f64],
    order_value: f64,
    reference: DVec3,
) -> Vec<Vec<DVec3>> {
    extract_order_contours_on_mesh_with_debug(
        triangle_positions,
        triangle_orders,
        order_value,
        reference,
    )
    .0
}

/// Builds an orthonormal in-plane basis (`basis1`, `basis2`) perpendicular
/// to `direction`, without hardcoding any particular world axis as "up" —
/// picks whichever of world X/Z is least parallel to `direction` as a seed
/// to avoid the degenerate cross product when `direction` is itself close
/// to that axis.
///
/// Exposed (rather than kept private) so callers that need to size a
/// sampling extent to fit *only* the in-plane footprint of some bounding
/// volume (e.g. `manifold-core::slicing::slice_mesh`, which must not let
/// a mesh's extent along `direction` inflate its per-layer contour grid
/// extent) can project onto the same basis [`extract_contours_at_order`]
/// samples against, instead of duplicating this seed-vector logic.
pub fn plane_basis(direction: DVec3) -> (DVec3, DVec3) {
    let seed = if direction.x.abs() < 0.9 {
        DVec3::X
    } else {
        DVec3::Z
    };
    let basis1 = direction.cross(seed).normalize();
    let basis2 = direction.cross(basis1).normalize();
    (basis1, basis2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{sphere_tree, TreeField};
    use std::f64::consts::PI;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    fn loop_perimeter(loop_points: &[DVec3]) -> f64 {
        let n = loop_points.len();
        (0..n)
            .map(|i| loop_points[i].distance(loop_points[(i + 1) % n]))
            .sum()
    }

    fn loop_max_radius_xy(loop_points: &[DVec3]) -> f64 {
        loop_points
            .iter()
            .map(|p| (p.x * p.x + p.y * p.y).sqrt())
            .fold(0.0, f64::max)
    }

    #[test]
    fn equatorial_slice_of_sphere_is_one_loop_with_expected_perimeter() {
        // Equatorial (Z=0) slice of a unit sphere is a unit circle:
        // perimeter = 2*pi*r, and every point's XY-radius should be ~1.
        let radius = 1.0;
        let field = TreeField::new(sphere_tree(radius));
        let loops = extract_contours(
            &field,
            DVec3::ZERO,
            DVec3::X,
            DVec3::Y,
            3.0,
            3.0,
            60,
            60,
            0.0,
        );

        assert_eq!(loops.len(), 1, "expected exactly one contour loop");
        let loop_points = &loops[0];
        assert!(loop_points.len() > 10);

        let perimeter = loop_perimeter(loop_points);
        let expected_perimeter = 2.0 * PI * radius;
        assert!(
            approx_eq(perimeter, expected_perimeter, 0.1),
            "expected perimeter ~{expected_perimeter}, got {perimeter}"
        );

        for p in loop_points {
            let r = (p.x * p.x + p.y * p.y).sqrt();
            assert!(
                approx_eq(r, radius, 0.05),
                "expected radius ~{radius}, got {r} at {p:?}"
            );
        }
    }

    /// A 2D annulus (ring) `ScalarField`: negative inside the ring band
    /// (between `inner_radius` and `outer_radius`), positive elsewhere —
    /// same sign convention as `MeshSdf` (negative = inside solid). Used
    /// to test that a hole's contour loop comes back with the opposite
    /// winding from its enclosing outer boundary.
    struct AnnulusField {
        inner_radius: f64,
        outer_radius: f64,
    }

    impl ScalarField for AnnulusField {
        fn sample(&self, p: DVec3) -> crate::FieldSample {
            let r = (p.x * p.x + p.y * p.y).sqrt();
            let outer = r - self.outer_radius;
            let inner = self.inner_radius - r;
            let value = outer.max(inner);
            let gradient = if r > f64::EPSILON {
                DVec3::new(p.x / r, p.y / r, 0.0) * if outer > inner { 1.0 } else { -1.0 }
            } else {
                DVec3::X
            };
            crate::FieldSample { value, gradient }
        }
    }

    #[test]
    fn annulus_hole_loop_winds_opposite_to_its_enclosing_outer_loop() {
        let field = AnnulusField {
            inner_radius: 0.5,
            outer_radius: 1.0,
        };
        let loops = extract_contours(
            &field,
            DVec3::ZERO,
            DVec3::X,
            DVec3::Y,
            3.0,
            3.0,
            80,
            80,
            0.0,
        );

        assert_eq!(loops.len(), 2, "expected an outer loop and a hole loop");

        let signed_area_3d = |points: &[DVec3]| -> f64 {
            let n = points.len();
            let mut area = 0.0;
            for i in 0..n {
                let a = points[i];
                let b = points[(i + 1) % n];
                area += a.x * b.y - b.x * a.y;
            }
            area * 0.5
        };

        let areas: Vec<f64> = loops.iter().map(|l| signed_area_3d(l)).collect();
        let outer_idx = areas
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
            .unwrap()
            .0;
        let hole_idx = 1 - outer_idx;

        assert!(
            areas[outer_idx].signum() != areas[hole_idx].signum(),
            "expected outer loop ({}) and hole loop ({}) to wind oppositely",
            areas[outer_idx],
            areas[hole_idx]
        );
    }

    #[test]
    fn off_center_slice_of_sphere_has_smaller_radius() {
        // A slice at Z = 0.5 through a unit sphere is a circle of radius
        // sqrt(1 - 0.5^2) ~= 0.866.
        let radius = 1.0;
        let z = 0.5;
        let field = TreeField::new(sphere_tree(radius));
        let loops = extract_contours(
            &field,
            DVec3::new(0.0, 0.0, z),
            DVec3::X,
            DVec3::Y,
            3.0,
            3.0,
            60,
            60,
            0.0,
        );

        assert_eq!(loops.len(), 1);
        let expected_radius = (radius * radius - z * z).sqrt();
        let max_radius = loop_max_radius_xy(&loops[0]);
        assert!(
            approx_eq(max_radius, expected_radius, 0.05),
            "expected radius ~{expected_radius}, got {max_radius}"
        );
    }

    #[test]
    fn slice_beyond_sphere_extent_has_no_contours() {
        let radius = 1.0;
        let field = TreeField::new(sphere_tree(radius));
        let loops = extract_contours(
            &field,
            DVec3::new(0.0, 0.0, 2.0),
            DVec3::X,
            DVec3::Y,
            3.0,
            3.0,
            20,
            20,
            0.0,
        );
        assert!(loops.is_empty());
    }

    #[test]
    fn extract_contours_at_order_matches_direct_plane_extraction_for_height_direction() {
        // For direction = +Z, extract_contours_at_order at order_value = z
        // should match extract_contours on the Z=z plane (same loop count,
        // same approximate radius), even though the in-plane basis vectors
        // may differ (rotated) from (X, Y).
        let radius = 1.0;
        let z = 0.3;
        let field = TreeField::new(sphere_tree(radius));

        let loops = extract_contours_at_order(&field, DVec3::Z, z, 3.0, 3.0, 60, 60);
        assert_eq!(loops.len(), 1);

        let expected_radius = (radius * radius - z * z).sqrt();
        let perimeter = loop_perimeter(&loops[0]);
        let expected_perimeter = 2.0 * PI * expected_radius;
        assert!(
            approx_eq(perimeter, expected_perimeter, 0.15),
            "expected perimeter ~{expected_perimeter}, got {perimeter}"
        );

        // All loop points should lie on the Z=z plane.
        for p in &loops[0] {
            assert!(approx_eq(p.z, z, 1e-6));
        }
    }

    #[test]
    fn cube_cross_section_is_a_square_with_expected_perimeter() {
        use crate::mesh_sdf::MeshSdf;

        // Unit cube spanning [0,1]^3 (reusing the cube_mesh fixture pattern
        // from mesh_sdf.rs), sliced at Z=0.5: a mid-height horizontal
        // cross-section is the unit square [0,1]x[0,1], perimeter 4.0.
        let vertices = vec![
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(1.0, 1.0, 0.0),
            DVec3::new(0.0, 1.0, 0.0),
            DVec3::new(0.0, 0.0, 1.0),
            DVec3::new(1.0, 0.0, 1.0),
            DVec3::new(1.0, 1.0, 1.0),
            DVec3::new(0.0, 1.0, 1.0),
        ];
        let faces = vec![
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [3, 7, 6],
            [3, 6, 2],
            [0, 4, 7],
            [0, 7, 3],
            [1, 2, 6],
            [1, 6, 5],
        ];
        let sdf = MeshSdf::new(vertices, faces);

        let loops = extract_contours(
            &sdf,
            DVec3::new(0.5, 0.5, 0.5),
            DVec3::X,
            DVec3::Y,
            2.0,
            2.0,
            80,
            80,
            0.0,
        );

        assert_eq!(loops.len(), 1, "expected exactly one contour loop");
        let perimeter = loop_perimeter(&loops[0]);
        assert!(
            approx_eq(perimeter, 4.0, 0.1),
            "expected perimeter ~4.0, got {perimeter}"
        );
    }

    #[test]
    fn extract_order_contours_on_mesh_finds_a_loop_on_a_sphere_isosurface_for_height_order() {
        use crate::marching_cubes::extract_isosurface;
        use crate::order::HeightOrderField;

        let radius = 1.0;
        let field = TreeField::new(sphere_tree(radius));
        let resolution = 40;
        let bound = radius * 1.5;
        let min = DVec3::splat(-bound);
        let max = DVec3::splat(bound);
        let vertices = extract_isosurface(&field, min, max, resolution, 0.0);
        assert!(!vertices.is_empty());
        let triangle_positions: Vec<DVec3> = vertices.iter().map(|v| v.position).collect();

        let z = 0.3;
        let order_field = HeightOrderField::new(DVec3::Z);
        let loops = extract_order_contours_on_mesh(&triangle_positions, &order_field, z, DVec3::Z);

        assert!(
            !loops.is_empty(),
            "expected at least one loop at height {z} on the sphere isosurface"
        );

        let expected_radius = (radius * radius - z * z).sqrt();
        let cell_size = 2.0 * bound / resolution as f64;
        let tolerance = 3.0 * cell_size;
        for loop_points in &loops {
            for p in loop_points {
                assert!(
                    approx_eq(p.z, z, tolerance),
                    "loop point {p:?} should lie near z = {z}"
                );
                let xy_radius = (p.x * p.x + p.y * p.y).sqrt();
                assert!(
                    approx_eq(xy_radius, expected_radius, tolerance),
                    "loop point {p:?} should have xy-radius ~{expected_radius}, got {xy_radius}"
                );
            }
        }
    }

    #[test]
    fn extract_order_contours_on_mesh_finds_a_loop_on_a_sphere_isosurface_for_conical_order() {
        use crate::marching_cubes::extract_isosurface;
        use crate::order::ConicalOrderField;

        let radius = 1.0;
        let field = TreeField::new(sphere_tree(radius));
        let resolution = 40;
        let bound = radius * 1.5;
        let min = DVec3::splat(-bound);
        let max = DVec3::splat(bound);
        let vertices = extract_isosurface(&field, min, max, resolution, 0.0);
        assert!(!vertices.is_empty());
        let triangle_positions: Vec<DVec3> = vertices.iter().map(|v| v.position).collect();

        // A cone apexed at the origin, opening along +Z: curved order field,
        // so the resulting contour on the sphere is not a flat plane slice.
        let order_field = ConicalOrderField::new(DVec3::ZERO, DVec3::Z, 0.3);
        let order_value = 0.5;
        let loops = extract_order_contours_on_mesh(
            &triangle_positions,
            &order_field,
            order_value,
            DVec3::Z,
        );

        assert!(
            !loops.is_empty(),
            "expected at least one loop for the conical order field on the sphere isosurface"
        );

        // All loop points should be (approximately) on the sphere surface
        // and hold the order field roughly constant at order_value.
        let cell_size = 2.0 * bound / resolution as f64;
        let tolerance = 3.0 * cell_size;
        for loop_points in &loops {
            for p in loop_points {
                assert!(
                    approx_eq(p.length(), radius, tolerance),
                    "loop point {p:?} should lie near the sphere surface (radius {radius})"
                );
                let order = order_field.order(*p);
                assert!(
                    approx_eq(order, order_value, tolerance),
                    "loop point {p:?} should have order ~{order_value}, got {order}"
                );
            }
        }
    }

    #[test]
    fn extract_order_contours_on_mesh_produces_finite_points_when_an_edge_endpoint_order_is_infinite(
    ) {
        // Regression test: an `EikonalOrderField` returns `f64::INFINITY` for
        // points the FMM front never reached (see `EikonalOrderField::order`).
        // A triangle edge with one finite and one infinite order value can
        // still register as an `order_value` crossing (the sign-change check
        // in `extract_order_contours_on_mesh` treats "infinite and above
        // order_value" as a valid sign), but naively lerping
        // `(order_value - v0) / (v1 - v0)` across that infinite jump used to
        // produce a NaN `t` (inf/inf or -inf/-inf) and thus a NaN crossing
        // point. `lerp_crossing` now falls back to the edge midpoint whenever
        // either endpoint is non-finite.
        struct StepOrderField;
        impl OrderField for StepOrderField {
            fn order(&self, p: DVec3) -> f64 {
                if p.x < 0.5 {
                    0.0
                } else {
                    f64::INFINITY
                }
            }
        }

        let p0 = DVec3::new(0.0, 0.0, 0.0);
        let p1 = DVec3::new(1.0, 0.0, 0.0);
        let p2 = DVec3::new(0.0, 1.0, 0.0);
        let triangle_positions = [p0, p1, p2];

        let loops =
            extract_order_contours_on_mesh(&triangle_positions, &StepOrderField, 0.5, DVec3::Z);

        for loop_points in &loops {
            for p in loop_points {
                assert!(
                    p.is_finite(),
                    "crossing point {p:?} should be finite, not NaN/inf"
                );
            }
        }
    }

    /// A hollow-shell `ScalarField`: negative between `inner_radius` and
    /// `outer_radius` (a spherical shell), positive elsewhere — same sign
    /// convention as [`AnnulusField`] but isotropic in 3D (uses the full
    /// `|p|` radius, not just an XY radius), so its isosurface is two
    /// concentric sphere surfaces rather than one flat ring.
    struct HollowSphereField {
        inner_radius: f64,
        outer_radius: f64,
    }

    impl ScalarField for HollowSphereField {
        fn sample(&self, p: DVec3) -> crate::FieldSample {
            let r = p.length();
            let outer = r - self.outer_radius;
            let inner = self.inner_radius - r;
            let value = outer.max(inner);
            let gradient = if r > f64::EPSILON {
                p.normalize() * if outer > inner { 1.0 } else { -1.0 }
            } else {
                DVec3::X
            };
            crate::FieldSample { value, gradient }
        }
    }

    #[test]
    fn conical_order_field_hole_loop_winds_opposite_to_its_enclosing_outer_loop() {
        use crate::marching_cubes::extract_isosurface;
        use crate::order::ConicalOrderField;

        // A hollow shell (solid between radius 0.5 and 1.0) sliced by a
        // cone apexed at the origin, opening along +Z: the cone crosses
        // both the outer and inner sphere surfaces, producing two loops on
        // the same isosurface mesh that are *not* coplanar (this is the
        // curved-order-field analogue of `annulus_hole_loop_winds_
        // opposite_to_its_enclosing_outer_loop`, which uses a single flat
        // plane instead).
        let outer_radius = 1.0;
        let inner_radius = 0.5;
        let field = HollowSphereField {
            inner_radius,
            outer_radius,
        };
        let resolution = 64;
        let bound = outer_radius * 1.5;
        let min = DVec3::splat(-bound);
        let max = DVec3::splat(bound);
        let vertices = extract_isosurface(&field, min, max, resolution, 0.0);
        assert!(!vertices.is_empty());
        let triangle_positions: Vec<DVec3> = vertices.iter().map(|v| v.position).collect();

        let order_field = ConicalOrderField::new(DVec3::ZERO, DVec3::Z, 0.3);
        let order_value = 0.2;
        let loops = extract_order_contours_on_mesh(
            &triangle_positions,
            &order_field,
            order_value,
            DVec3::Z,
        );

        assert_eq!(
            loops.len(),
            2,
            "expected an outer loop (on the outer sphere) and a hole loop (on the inner sphere)"
        );

        let signed_area_3d = |points: &[DVec3]| -> f64 {
            let n = points.len();
            let mut area = 0.0;
            for i in 0..n {
                let a = points[i];
                let b = points[(i + 1) % n];
                area += a.x * b.y - b.x * a.y;
            }
            area * 0.5
        };

        let areas: Vec<f64> = loops.iter().map(|l| signed_area_3d(l)).collect();
        let outer_idx = areas
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
            .unwrap()
            .0;
        let hole_idx = 1 - outer_idx;

        assert!(
            areas[outer_idx].signum() != areas[hole_idx].signum(),
            "expected outer loop ({}) and hole loop ({}) to wind oppositely",
            areas[outer_idx],
            areas[hole_idx]
        );

        // Sanity check: every point on the outer loop should be near the
        // outer sphere's surface, every point on the hole loop near the
        // inner sphere's surface.
        let cell_size = 2.0 * bound / resolution as f64;
        let tolerance = 3.0 * cell_size;
        for &p in &loops[outer_idx] {
            assert!(
                approx_eq(p.length(), outer_radius, tolerance),
                "outer loop point {p:?} should lie near the outer sphere (radius {outer_radius})"
            );
        }
        for &p in &loops[hole_idx] {
            assert!(
                approx_eq(p.length(), inner_radius, tolerance),
                "hole loop point {p:?} should lie near the inner sphere (radius {inner_radius})"
            );
        }
    }

    #[test]
    fn stitch_loops_closes_a_well_formed_square() {
        let segments = vec![
            (DVec3::new(0.0, 0.0, 0.0), DVec3::new(1.0, 0.0, 0.0)),
            (DVec3::new(1.0, 0.0, 0.0), DVec3::new(1.0, 1.0, 0.0)),
            (DVec3::new(1.0, 1.0, 0.0), DVec3::new(0.0, 1.0, 0.0)),
            (DVec3::new(0.0, 1.0, 0.0), DVec3::new(0.0, 0.0, 0.0)),
        ];
        let loops = stitch_loops(segments);
        assert_eq!(loops.len(), 1, "expected exactly one closed loop");
        assert!(loops[0].len() >= MIN_LOOP_POINTS);
    }

    #[test]
    fn stitch_loops_drops_an_unmatched_open_fragment_instead_of_emitting_it() {
        // A well-formed closed triangle, plus one isolated dangling segment
        // whose endpoints don't match anything else (e.g. clipped by plane
        // extent, or a genuine stitching gap) -- the dangling segment must
        // not be emitted as a printable "loop".
        let segments = vec![
            (DVec3::new(0.0, 0.0, 0.0), DVec3::new(1.0, 0.0, 0.0)),
            (DVec3::new(1.0, 0.0, 0.0), DVec3::new(0.5, 1.0, 0.0)),
            (DVec3::new(0.5, 1.0, 0.0), DVec3::new(0.0, 0.0, 0.0)),
            (DVec3::new(10.0, 10.0, 0.0), DVec3::new(11.0, 10.0, 0.0)),
        ];
        let loops = stitch_loops(segments);
        assert_eq!(
            loops.len(),
            1,
            "expected only the well-formed triangle, dangling fragment dropped"
        );
        assert!(loops[0].len() >= MIN_LOOP_POINTS);
        for p in &loops[0] {
            assert!(
                p.x < 5.0 && p.y < 5.0,
                "unexpected point from dangling fragment: {p:?}"
            );
        }
    }

    #[test]
    fn stitch_loops_drops_a_too_few_point_closed_fragment() {
        // Two segments that happen to close on each other (a degenerate
        // 2-point "loop") -- geometrically not a valid closed polygon, so
        // it must be dropped rather than kept as a `MIN_LOOP_POINTS`-violating
        // fragment.
        let segments = vec![
            (DVec3::new(0.0, 0.0, 0.0), DVec3::new(1.0, 0.0, 0.0)),
            (DVec3::new(1.0, 0.0, 0.0), DVec3::new(0.0, 0.0, 0.0)),
        ];
        let loops = stitch_loops(segments);
        assert!(
            loops.is_empty(),
            "expected the degenerate 2-point closed fragment to be dropped, got {loops:?}"
        );
    }

    #[test]
    fn stitch_loops_repairs_a_gap_within_the_widened_tolerance() {
        // The second segment's start point is off by slightly more than
        // `STITCH_EPSILON` from the first segment's end point (simulating
        // floating-point drift between independently-lerped points on a
        // shared edge), but well within `STITCH_REPAIR_TOLERANCE` -- the
        // widened repair pass should still close this into one loop.
        let drift = STITCH_REPAIR_TOLERANCE / 10.0;
        let segments = vec![
            (DVec3::new(0.0, 0.0, 0.0), DVec3::new(1.0, 0.0, 0.0)),
            (DVec3::new(1.0 + drift, 0.0, 0.0), DVec3::new(0.5, 1.0, 0.0)),
            (DVec3::new(0.5, 1.0, 0.0), DVec3::new(0.0, 0.0, 0.0)),
        ];
        let loops = stitch_loops(segments);
        assert_eq!(
            loops.len(),
            1,
            "expected the small-drift gap to be repaired into one closed loop"
        );
    }

    #[test]
    fn stitch_loops_deduplicates_identical_segments_and_closes_loop() {
        // A square where one edge is duplicated (e.g. emitted by two adjacent
        // triangles sharing an on-contour edge in `extract_order_contours_on_mesh`).
        // Without deduplication, the duplicate edge creates degree-3 vertices and
        // causes the loop to terminate prematurely or fail to close.
        let segments = vec![
            (DVec3::new(0.0, 0.0, 0.0), DVec3::new(1.0, 0.0, 0.0)),
            // Duplicate of edge (0,0) -> (1,0) in reverse order:
            (DVec3::new(1.0, 0.0, 0.0), DVec3::new(0.0, 0.0, 0.0)),
            (DVec3::new(1.0, 0.0, 0.0), DVec3::new(1.0, 1.0, 0.0)),
            (DVec3::new(1.0, 1.0, 0.0), DVec3::new(0.0, 1.0, 0.0)),
            (DVec3::new(0.0, 1.0, 0.0), DVec3::new(0.0, 0.0, 0.0)),
        ];
        let loops = stitch_loops(segments);
        assert_eq!(
            loops.len(),
            1,
            "expected duplicate edge to be deduplicated and close the square"
        );
        assert_eq!(loops[0].len(), 4);
    }

    #[test]
    fn stitch_loops_prunes_dangling_spurs_and_closes_attached_cycle() {
        // A triangle loop with a dangling 2-segment tail attached at (1, 0).
        // Starting at the dangling tail could greedily consume edges and hit a dead end,
        // but spur pruning and backtracking ensure the true closed triangle is extracted.
        let segments = vec![
            (DVec3::new(2.0, 0.0, 0.0), DVec3::new(3.0, 0.0, 0.0)),
            (DVec3::new(1.0, 0.0, 0.0), DVec3::new(2.0, 0.0, 0.0)),
            (DVec3::new(0.0, 0.0, 0.0), DVec3::new(1.0, 0.0, 0.0)),
            (DVec3::new(1.0, 0.0, 0.0), DVec3::new(0.5, 1.0, 0.0)),
            (DVec3::new(0.5, 1.0, 0.0), DVec3::new(0.0, 0.0, 0.0)),
        ];
        let loops = stitch_loops(segments);
        assert_eq!(
            loops.len(),
            1,
            "expected the closed triangle to be preserved and spur pruned"
        );
        assert_eq!(loops[0].len(), 3);
    }
}
