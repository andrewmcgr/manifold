//! Toolpath planning: layers -> ordered extrusion moves.

use crate::infill::{self, InfillRegion};
use crate::{
    bounds::BoundingVolume, extrusion, ids::ToolId, object::Object, slicing::Layer, tool::Tool,
    Error, Result, SlicerConfig,
};
use glam::DVec3;
use manifold_fidget::mesh_sdf::MeshSdf;
use manifold_fidget::ScalarField;
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// Signed-distance tolerance (mm) used by [`retain_contained_paths`] to
/// decide whether a path point is "outside the solid": a point sampled via
/// [`ScalarField::sample`] with a value beyond this threshold is genuinely
/// outside real mesh material, not just float noise from a point that's
/// meant to sit essentially on the surface. Deliberately far above f64
/// rounding noise but far below any real printable feature (the smallest
/// legitimate inward offset is on the order of a nozzle radius, i.e.
/// several hundredths of a mm at minimum), so this cannot mask an actual
/// containment failure while still tolerating numerical jitter.
const CONTAINMENT_EPSILON: f64 = 1e-6;

/// Drops any non-[`MoveKind::Travel`] path in `paths` that isn't fully
/// contained in the real solid, using `mesh_sdf` (built directly from the
/// mesh -- see [`Layer::mesh_sdf`]) as ground truth rather than trusting
/// the 2D loop/boundary geometry `paths` were generated from.
///
/// This exists as a final safety net: wall/infill loop geometry is derived
/// from contour extraction and polygon boolean ops on
/// `infill_boundary`/`solid_fill_boundary`, which have (rarely) produced
/// loops that don't correspond to real solid material -- e.g. infill
/// printed inside a hole that isn't actually part of the object. Rather
/// than only trying to prevent every possible source of that class of bug
/// upstream, every extruding path is re-checked here against the mesh
/// itself and dropped wholesale if any of its points land outside the
/// solid (see [`CONTAINMENT_EPSILON`]) -- a partially-valid path is
/// dropped entirely rather than clipped, since splitting it would risk
/// producing a spurious partial loop/travel move that's arguably worse
/// than simply omitting the whole (already-wrong) path.
///
/// No-op (returns `paths` unchanged) when `mesh_sdf` is `None` -- a
/// synthetic/test [`Layer`] has no ground truth to check against, so
/// containment is treated as unknown rather than enforced.
fn retain_contained_paths(
    paths: Vec<Path>,
    mesh_sdf: Option<&Arc<MeshSdf>>,
    order: f64,
) -> Vec<Path> {
    let Some(mesh_sdf) = mesh_sdf else {
        return paths;
    };

    let total = paths.len();
    let mut dropped_paths = 0usize;
    let mut dropped_points = 0usize;
    let retained: Vec<Path> = paths
        .into_iter()
        .filter(|path| {
            if path
                .segments
                .iter()
                .all(|segment| segment.kind == MoveKind::Travel)
            {
                return true;
            }
            let contained = path
                .points
                .iter()
                .all(|&p| mesh_sdf.sample(p).value <= CONTAINMENT_EPSILON);
            if !contained {
                dropped_paths += 1;
                dropped_points += path.points.len();
            }
            contained
        })
        .collect();

    if dropped_paths > 0 {
        tracing::warn!(
            layer.order = order,
            dropped_paths,
            dropped_points,
            total_paths = total,
            "dropped extruding path(s) not contained in the solid mesh"
        );
    }

    retained
}

/// Applies configurable Z-hop (lift-before-travel / lower-after-arrival) to
/// every [`Path`] in `paths`, when `config.z_hop_enabled`. No-op (returns
/// `paths` unchanged, not even reallocated) when disabled -- the default --
/// so existing behavior/output is completely unaffected unless a caller
/// opts in. See `.tmp/tasks/slicer-fix-backlog/scoping-phase-d.md` for the
/// full design rationale.
///
/// For each maximal run of consecutive [`MoveKind::Travel`] segments within
/// a `Path` (i.e. `segments[run_start..=run_end]` all `Travel`), inserts:
/// - a lift point immediately after the run's departure point (same XY,
///   `Z + config.z_hop_height`);
/// - every original travel point strictly inside the run, raised by the
///   same `Z + config.z_hop_height` (so lateral travel happens entirely at
///   hop height, not just at the endpoints);
/// - a drop point immediately before the run's arrival point (arrival's
///   XY, still at `Z + config.z_hop_height`), followed by the unmodified
///   arrival point itself (its original, real Z) to lower back down.
///
/// All inserted points/segments are tagged [`MoveKind::Travel`] with
/// `extrusion_length: 0.0`, matching how `gcode::emit` already renders
/// existing travel points -- `emit` requires no changes for this feature
/// (see scoping doc §3). Works uniformly for both closed loops
/// (`segments.len() == points.len()`, with an unused-by-`emit` closing
/// segment -- see that doc comment) and open paths (e.g. infill
/// zig-zags, `segments.len() == points.len() - 1`, no closing edge): any
/// closing segment present is left untouched at the end, preserving
/// whichever parallel-array shape the path already had.
fn insert_z_hops(paths: Vec<Path>, config: &SlicerConfig) -> Vec<Path> {
    if !config.z_hop_enabled {
        return paths;
    }
    paths
        .into_iter()
        .map(|path| insert_z_hops_into_path(path, config.z_hop_height))
        .collect()
}

/// Rebuilds a single `path`'s `points`/`segments` with Z-hop lift/drop
/// geometry inserted around every maximal run of consecutive
/// [`MoveKind::Travel`] segments -- except a run that both departs from
/// and arrives at an [`MoveKind::Infill`] segment, which is left as a
/// plain, un-hopped travel move. A single infill fill (e.g. one
/// `MonotonicInfill::generate` call's boustrophedon zigzag, or multiple
/// islands scanned together within one region) already emits its own
/// internal travel jumps between scan-line segments as one continuous
/// `Path`; those jumps stay at the same printed Z and don't need to clear
/// already-extruded geometry the way a travel between *different* move
/// kinds (e.g. wall-to-infill, or path-to-path) might, so hopping between
/// them is pure wasted motion. See [`insert_z_hops`]'s doc comment for the
/// exact point sequence when a hop *is* inserted.
fn insert_z_hops_into_path(path: Path, hop_height: f64) -> Path {
    let Path {
        points,
        segments,
        tool,
    } = path;
    let point_count = points.len();
    // Fewer than 2 points means no edges at all -- nothing to hop around.
    if point_count < 2 {
        return Path {
            points,
            segments,
            tool,
        };
    }

    let mut new_points = Vec::with_capacity(point_count);
    let mut new_segments = Vec::with_capacity(point_count);

    // Walk edges `e` in `0..point_count - 1` (`segments[e]` describes
    // `points[e] -> points[e + 1]`); the closing edge
    // `segments[point_count - 1]` (`points[point_count - 1] -> points[0]`)
    // is handled separately below, unmodified (see this function's caller's
    // doc comment).
    let mut e = 0usize;
    while e < point_count - 1 {
        new_points.push(points[e]);
        if segments[e].kind == MoveKind::Travel {
            let run_start = e;
            let mut run_end = e;
            while run_end + 1 < point_count - 1 && segments[run_end + 1].kind == MoveKind::Travel {
                run_end += 1;
            }

            // A run entirely surrounded by Infill segments (the move
            // arriving at the departure point, and the move leaving the
            // arrival point, both `MoveKind::Infill`) is an internal jump
            // within the same infill patch -- skip the hop entirely and
            // fall through to copying the run's original points/segments
            // unmodified, exactly as the non-Travel branch below does. A
            // run at either end of the whole path (no bounding segment on
            // that side) is conservatively treated as needing a hop, since
            // there's no infill segment to confirm it's an internal jump.
            let departs_infill = run_start > 0 && segments[run_start - 1].kind == MoveKind::Infill;
            let arrives_infill =
                run_end + 1 < point_count - 1 && segments[run_end + 1].kind == MoveKind::Infill;
            if departs_infill && arrives_infill {
                for k in run_start..run_end {
                    new_points.push(points[k + 1]);
                }
                for segment in segments.iter().take(run_end + 1).skip(run_start) {
                    new_segments.push(*segment);
                }
                e = run_end + 1;
                continue;
            }

            let departure = points[run_start];
            let arrival = points[run_end + 1];

            // Lift straight up from the departure point.
            new_points.push(DVec3::new(
                departure.x,
                departure.y,
                departure.z + hop_height,
            ));
            new_segments.push(Segment {
                kind: MoveKind::Travel,
                extrusion_length: 0.0,
                ..segments[run_start]
            });

            // Lateral travel at hop height through every original travel
            // point strictly inside the run.
            for k in (run_start + 1)..=run_end {
                let p = points[k];
                new_points.push(DVec3::new(p.x, p.y, p.z + hop_height));
                new_segments.push(Segment {
                    kind: MoveKind::Travel,
                    extrusion_length: 0.0,
                    ..segments[k - 1]
                });
            }

            // Final lateral move to the arrival XY, still at hop height.
            new_points.push(DVec3::new(arrival.x, arrival.y, arrival.z + hop_height));
            new_segments.push(Segment {
                kind: MoveKind::Travel,
                extrusion_length: 0.0,
                ..segments[run_end]
            });

            // Drop straight down onto the real (unmodified) arrival point,
            // pushed on the next loop iteration (or after the loop, if the
            // arrival point is the path's last point).
            new_segments.push(Segment {
                kind: MoveKind::Travel,
                extrusion_length: 0.0,
                ..segments[run_end]
            });

            e = run_end + 1;
        } else {
            new_segments.push(segments[e]);
            e += 1;
        }
    }
    // Push the final point unchanged. Only append the closing segment if
    // one exists: closed loops carry `segments.len() == point_count` (see
    // `Path`'s doc comment), but open paths (e.g. infill zig-zags, see
    // `infill::MonotonicInfill::generate`) carry only `point_count - 1`
    // segments and have no closing edge to preserve.
    new_points.push(points[point_count - 1]);
    if segments.len() == point_count {
        new_segments.push(segments[point_count - 1]);
    }

    Path {
        points: new_points,
        segments: new_segments,
        tool,
    }
}

/// Applies Ramer-Douglas-Peucker (RDP) perpendicular-distance
/// simplification to wall-loop paths (`MoveKind::WallOuter` /
/// `MoveKind::WallInner`), reducing point count from pathological point
/// density (e.g. curved-order-field contour extraction such as Eikonal,
/// which can produce long "staircase" runs of near-collinear points) while
/// staying within `config.path_simplify_tolerance` (mm) of the original
/// geometry. No-op (returns `paths` unchanged, not even reallocated) when
/// `config.path_simplify_enabled` is `false` -- mirrors [`insert_z_hops`]'s
/// handling of `z_hop_enabled: false`.
///
/// Only wall-loop paths are simplified: infill paths are deliberately
/// spaced for density guarantees (see `infill` generators), so simplifying
/// them is explicitly out of scope for v1 -- future work. Every other path
/// (infill, travel-only, etc.) is passed through completely untouched. A
/// path's kind is read from its first segment, since `plan_with_progress`
/// currently tags every segment of a given wall loop with the same
/// `MoveKind` uniformly.
///
/// Does not recompute `Segment::extrusion_length` itself -- that is left
/// to the existing downstream extrusion-length pass in
/// `plan_with_progress`, which runs after this pass on the (possibly
/// simplified) segment geometry.
/// Compensates wall-loop contours for the nozzle's flat tip land (see
/// `SlicerConfig::nozzle_flat_diameter`) when the local nozzle axis tilts
/// away from the layer's own "vertical" for non-planar printing.
///
/// The nozzle axis at a point `p` is modeled as the local surface normal
/// of the layer's order-field isosurface: `-normalize(grad order(p))`
/// (see `order_field::numeric_gradient`), which reduces to world-up for a
/// flat `Height` field and tilts to follow the surface for `Conical`/
/// `Eikonal`. The flat land (radius `flat_radius`) lies in the plane
/// perpendicular to that axis, centered on the nozzle-center path -- for a
/// perfectly flat/untilted layer this touches the true contour
/// symmetrically all around and needs no correction (that's what
/// `SlicerConfig::wall_offset` already handles for the isotropic case).
///
/// Correction is only needed where the axis itself is turning *along the
/// direction of travel*, i.e. where the surface has curvature in the
/// travel direction: the flat's trailing edge (the point flat_radius
/// behind the center, against the just-extruded material) can end up off
/// the true contour by the amount the surface tilts over that
/// flat-radius span. For each point this is estimated by probing the
/// nozzle axis a `flat_radius` step to either side of the point along the
/// *radial* (meridian/cross-section) direction -- perpendicular to both
/// the nozzle axis and the loop's own tangent (travel) direction:
///
/// - `radial_dir = normalize(axis.cross(tangent))`. This deliberately
///   excludes the tangent direction itself: walking around an
///   axisymmetric loop (e.g. a cone's constant-slope wall) rotates the
///   axis vector purely as an artifact of going around the loop, without
///   the surface's actual cross-sectional slope changing at all -- an
///   earlier version of this function derived the shift direction from
///   `axis_ahead - axis_behind` (probing along tangent) and it collapsed
///   onto the tangent direction exactly for a cone, shifting points along
///   their own contour instead of across it. Probing along `radial_dir`
///   instead measures the surface's real slope profile (how the tilt
///   changes moving toward/away from the object's core), which is the
///   only direction the flat's footprint can actually mismatch the
///   surface across.
/// - `theta` = angle between the two radial-probe axes (the tilt swept
///   over one flat-radius span in the radial direction).
/// - the shift magnitude is `flat_radius * sin(theta / 2.0)`, the lateral
///   displacement of a chord of length `flat_radius` swept through half
///   that turn angle, applied along `radial_dir`.
///
/// Before applying, each point checks *which* radial side actually
/// descends toward already-printed material by comparing `field.order`
/// at each radial probe against `layer_order + layer_height` (the order
/// value of the layer printed just before this one -- order decreases as
/// printing proceeds, see `slicing::BUILD_DIRECTION`): only the side
/// that's actually closer to already-solid material gets compensated.
/// This is the "climbing" exemption: when neither probe is meaningfully
/// closer to solid than the other (or the point is tilting away from
/// solid on both sides, e.g. an overhang-like climb), there's no
/// already-printed surface for the flat to (mis)contact, so the point is
/// left unshifted.
///
/// Degenerate cases (missing/zero gradient, near-zero curvature, a loop
/// too short to have neighbors, or `axis`/`tangent` too close to parallel
/// to define a radial direction) fall through as a no-op for that point
/// rather than injecting noise -- this is a best-effort geometric
/// approximation, not an exact physical simulation.
fn compensate_flat_nozzle(paths: Vec<Path>, layer: &Layer, config: &SlicerConfig) -> Vec<Path> {
    let flat_radius = config.nozzle_flat_diameter() / 2.0;
    if flat_radius <= f64::EPSILON {
        return paths;
    }
    let field = layer.order_field.as_ref();

    paths
        .into_iter()
        .map(|mut path| {
            let is_wall_loop = path.segments.first().is_some_and(|segment| {
                matches!(segment.kind, MoveKind::WallOuter | MoveKind::WallInner)
            });
            if is_wall_loop && path.points.len() >= 3 {
                path.points = compensate_wall_loop_points(
                    &path.points,
                    field,
                    flat_radius,
                    config.layer_height,
                    layer.order,
                );
            }
            path
        })
        .collect()
}

/// Per-point worker behind [`compensate_flat_nozzle`]; see that function's
/// doc for the geometric model. `points` is a closed loop (wraps like
/// `Path`'s own contract).
fn compensate_wall_loop_points(
    points: &[DVec3],
    field: &dyn manifold_fidget::order::OrderField,
    flat_radius: f64,
    layer_height: f64,
    layer_order: f64,
) -> Vec<DVec3> {
    let len = points.len();
    // Order value of the layer printed just before this one -- order
    // decreases as printing proceeds (see `slicing::BUILD_DIRECTION`), so
    // the already-solidified prior layer sits at a *higher* order value.
    let prev_layer_order = layer_order + layer_height;

    points
        .iter()
        .enumerate()
        .map(|(i, &p)| {
            let Some(axis) = crate::order_field::numeric_gradient(field, p).map(|g| -g.normalize())
            else {
                return p;
            };

            let prev_point = points[(i + len - 1) % len];
            let next_point = points[(i + 1) % len];
            let tangent = next_point - prev_point;
            if tangent.length_squared() < 1e-12 {
                return p;
            }
            let tangent = tangent.normalize();

            // Perpendicular to both the nozzle axis and the direction of
            // travel: the meridian/cross-section direction, not the
            // tangent -- see this function's doc for why tangent itself
            // is the wrong probing direction.
            let radial = axis.cross(tangent);
            let radial_len = radial.length();
            if radial_len < 1e-9 {
                return p;
            }
            let radial_dir = radial / radial_len;

            let probe_a = p + radial_dir * flat_radius;
            let probe_b = p - radial_dir * flat_radius;
            let (Some(axis_a), Some(axis_b)) = (
                crate::order_field::numeric_gradient(field, probe_a).map(|g| -g.normalize()),
                crate::order_field::numeric_gradient(field, probe_b).map(|g| -g.normalize()),
            ) else {
                return p;
            };

            let theta = axis_a.dot(axis_b).clamp(-1.0, 1.0).acos();
            if theta < 1e-6 {
                return p;
            }

            let order_a = field.order(probe_a);
            let order_b = field.order(probe_b);
            if !order_a.is_finite() || !order_b.is_finite() {
                return p;
            }
            let score_a = -(order_a - prev_layer_order).abs();
            let score_b = -(order_b - prev_layer_order).abs();

            let shift_mag = flat_radius * (theta / 2.0).sin();
            if score_b > score_a {
                // Side `b` (`-radial_dir`) is the one closer to
                // already-printed material: pull the center back toward
                // `a` so the flat's trailing edge on the `b` side lands
                // on the original contour.
                p + radial_dir * shift_mag
            } else if score_a > score_b {
                p - radial_dir * shift_mag
            } else {
                p
            }
        })
        .collect()
}

fn simplify_paths(paths: Vec<Path>, config: &SlicerConfig) -> Vec<Path> {
    if !config.path_simplify_enabled {
        return paths;
    }
    paths
        .into_iter()
        .map(|path| {
            let is_wall_loop = path.segments.first().is_some_and(|segment| {
                matches!(segment.kind, MoveKind::WallOuter | MoveKind::WallInner)
            });
            if is_wall_loop {
                simplify_path(path, config.path_simplify_tolerance)
            } else {
                path
            }
        })
        .collect()
}

/// Simplifies a single wall-loop `path` via RDP, dispatching to the
/// closed-loop-aware or open-path variant based on the parallel-array
/// invariant (see [`Path`]'s doc comment). Degenerate inputs (fewer than 3
/// points, or a non-positive `tolerance`) are returned unchanged rather
/// than risking a panic or a meaningless simplification.
fn simplify_path(path: Path, tolerance: f64) -> Path {
    let Path {
        points,
        segments,
        tool,
    } = path;
    let point_count = points.len();
    if point_count < 3 || tolerance <= 0.0 {
        return Path {
            points,
            segments,
            tool,
        };
    }
    if segments.len() == point_count {
        simplify_closed_path(points, segments, tolerance, tool)
    } else {
        simplify_open_path(points, segments, tolerance, tool)
    }
}

/// RDP-simplifies an open path (`segments.len() == points.len() - 1`, no
/// closing edge): classic Douglas-Peucker over the single chain
/// `0..points.len()`, always keeping the first and last point.
fn simplify_open_path(
    points: Vec<DVec3>,
    segments: Vec<Segment>,
    tolerance: f64,
    tool: ToolId,
) -> Path {
    let point_count = points.len();
    let chain: Vec<usize> = (0..point_count).collect();
    let mut keep = vec![false; point_count];
    keep[0] = true;
    keep[point_count - 1] = true;
    rdp_mark(&points, &chain, tolerance, &mut keep);

    let kept_indices: Vec<usize> = chain.into_iter().filter(|&i| keep[i]).collect();
    let new_points = kept_indices.iter().map(|&i| points[i]).collect();
    // Every kept point except the last keeps its own original outgoing
    // segment verbatim (no interpolation/averaging across dropped points).
    let new_segments = kept_indices[..kept_indices.len() - 1]
        .iter()
        .map(|&i| segments[i])
        .collect();

    Path {
        points: new_points,
        segments: new_segments,
        tool,
    }
}

/// RDP-simplifies a closed loop (`segments.len() == points.len()`):
/// classic Douglas-Peucker is defined on an open polyline, so the loop is
/// split into two open chains at its two most mutually distant points (a
/// standard technique for closed-loop RDP), each chain is simplified
/// independently, and the surviving points are rejoined into a single
/// closed loop, preserving the parallel-array invariant.
fn simplify_closed_path(
    points: Vec<DVec3>,
    segments: Vec<Segment>,
    tolerance: f64,
    tool: ToolId,
) -> Path {
    let point_count = points.len();
    let (a, b) = farthest_pair(&points);
    if a == b {
        // All points coincide (zero-length/degenerate loop) -- nothing
        // meaningful to simplify.
        return Path {
            points,
            segments,
            tool,
        };
    }

    let chain_ab = forward_chain(a, b, point_count);
    let chain_ba = forward_chain(b, a, point_count);
    let mut keep = vec![false; point_count];
    keep[a] = true;
    keep[b] = true;
    rdp_mark(&points, &chain_ab, tolerance, &mut keep);
    rdp_mark(&points, &chain_ba, tolerance, &mut keep);

    // Rebuild in the loop's original index/orientation order (rather than
    // starting at `a`, which the two chains above are anchored to) so a
    // loop that survives simplification unchanged is *actually* returned
    // unchanged -- same starting point and winding -- not just
    // point-for-point equal under rotation.
    let final_indices: Vec<usize> = (0..point_count).filter(|&i| keep[i]).collect();

    // Each surviving point keeps its own original outgoing segment
    // verbatim -- e.g. `[p0, p1, p2, p3]` simplifying to `[p0, p3]` keeps
    // `segments[0]` (p0's original outgoing edge) for the new `p0`, not
    // some blend of `segments[0..3]`.
    let new_points = final_indices.iter().map(|&i| points[i]).collect();
    let new_segments = final_indices.iter().map(|&i| segments[i]).collect();

    Path {
        points: new_points,
        segments: new_segments,
        tool,
    }
}

/// Walks the cyclic index range `start..=end` (inclusive of both ends,
/// wrapping modulo `len`), used to carve a closed loop's point indices
/// into one of the two open chains RDP needs.
fn forward_chain(start: usize, end: usize, len: usize) -> Vec<usize> {
    let mut chain = Vec::new();
    let mut i = start;
    loop {
        chain.push(i);
        if i == end {
            break;
        }
        i = (i + 1) % len;
    }
    chain
}

/// Returns the pair of point indices with the greatest Euclidean distance
/// apart, used to pick the closed-loop split points for RDP. O(n^2); fine
/// for the point counts wall loops carry in practice, but a candidate for
/// optimization if extremely dense input loops ever make this pass show up
/// in profiling.
fn farthest_pair(points: &[DVec3]) -> (usize, usize) {
    let len = points.len();
    let mut best = (0usize, (len - 1).min(1));
    let mut best_dist_sq = -1.0f64;
    for i in 0..len {
        for j in (i + 1)..len {
            let dist_sq = points[i].distance_squared(points[j]);
            if dist_sq > best_dist_sq {
                best_dist_sq = dist_sq;
                best = (i, j);
            }
        }
    }
    best
}

/// Recursively marks (in `keep`, indexed by global point index) which
/// points along `chain` survive RDP simplification against `tolerance`
/// (mm): finds the point in `chain`'s interior farthest (perpendicular
/// distance) from the line through its endpoints; if that distance exceeds
/// `tolerance`, keeps that point and recurses on both halves, otherwise
/// drops the entire interior. `chain`'s first and last points are assumed
/// already marked kept by the caller.
fn rdp_mark(points: &[DVec3], chain: &[usize], tolerance: f64, keep: &mut [bool]) {
    if chain.len() < 3 {
        return;
    }
    let first = chain[0];
    let last = chain[chain.len() - 1];
    let mut max_dist = 0.0f64;
    let mut max_pos = 0usize;
    for (pos, &idx) in chain.iter().enumerate().take(chain.len() - 1).skip(1) {
        let dist = perpendicular_distance(points[idx], points[first], points[last]);
        if dist > max_dist {
            max_dist = dist;
            max_pos = pos;
        }
    }
    if max_dist > tolerance {
        keep[chain[max_pos]] = true;
        rdp_mark(points, &chain[..=max_pos], tolerance, keep);
        rdp_mark(points, &chain[max_pos..], tolerance, keep);
    }
}

/// Perpendicular distance from `p` to the infinite line through `a` and
/// `b` (classic Douglas-Peucker uses the line, not the segment). Falls
/// back to plain point-to-point distance when `a`/`b` coincide, rather
/// than dividing by (near-)zero.
fn perpendicular_distance(p: DVec3, a: DVec3, b: DVec3) -> f64 {
    let ab = b - a;
    let ab_len_sq = ab.length_squared();
    if ab_len_sq < f64::EPSILON {
        return p.distance(a);
    }
    let ap = p - a;
    ap.cross(ab).length() / ab_len_sq.sqrt()
}

/// Validates that every point of every planned path in `paths` lies within
/// `build_volume`, returning [`Error::MoveOutOfBounds`] naming the first
/// offending point found (in `paths` order) if not.
///
/// This is a last-resort safety net, checked once after planning
/// completes (see `crate::plan_toolpaths_with_progress`) rather than
/// silently dropped/clipped like [`retain_contained_paths`]: a move
/// outside the machine's physical build volume means either a genuine
/// geometry/config problem (e.g. object placement, or a slicing-pipeline
/// bug producing a wild point far from the object -- see the Eikonal
/// order field's seed-region/contour-plateau fixes this guards against a
/// regression of) or a real out-of-range command that would otherwise
/// only be discovered by the printer firmware refusing the move at print
/// time. Either way, failing fast here with a clear error is preferable
/// to reaching Gcode.
pub fn validate_within_bounds(paths: &[Path], build_volume: &BoundingVolume) -> Result<()> {
    for path in paths {
        for &point in &path.points {
            if !build_volume.contains(point) {
                return Err(Error::MoveOutOfBounds { point });
            }
        }
    }
    Ok(())
}

/// Chooses the Gcode feedrate (`Segment::speed`, mm/min) for a segment of
/// the given `kind`, from `config`. [`MoveKind::Travel`] uses
/// `config.travel_speed`; every extruding kind (`WallOuter`/`WallInner`/
/// `Infill`/`Bridge`/`Overhang`) uses `config.print_speed` -- there is no
/// finer-grained per-extruding-kind speed yet (e.g. a separate bridge
/// speed), so all of them share one "print speed" until that becomes
/// configurable.
#[must_use]
pub fn speed_for_kind(kind: MoveKind, config: &SlicerConfig) -> f64 {
    match kind {
        MoveKind::Travel => config.travel_speed,
        MoveKind::WallOuter
        | MoveKind::WallInner
        | MoveKind::Infill
        | MoveKind::Bridge
        | MoveKind::Overhang => config.print_speed,
    }
}

/// Classification of a single toolpath segment (the move from one point to
/// the next along a [`Path`]). [`plan`] derives `WallOuter`/`WallInner`
/// from each loop's wall index (see `slicing::WallLoop`); real infill/
/// support/bridge/overhang *detection* is still future work.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MoveKind {
    #[default]
    WallOuter,
    WallInner,
    Infill,
    Bridge,
    Overhang,
    Travel,
}

/// Per-segment motion metadata for one `points[i] -> points[i+1]` edge of a
/// [`Path]` (including the closing edge of a closed loop).
#[derive(Debug, Clone, Copy, Default)]
pub struct Segment {
    pub kind: MoveKind,
    pub speed: f64,
    pub extrusion_rate: f64,
    pub support_fraction: f64,
    /// The order-field value (see `manifold_fidget::order`) whose
    /// isosurface produced this segment's source [`Layer`]. Stored
    /// per-segment (rather than per-`Path`/per-`Layer`) so it can vary
    /// once non-planar order fields exist.
    pub order: f64,
    /// Linear filament feed length (mm) to extrude for this segment --
    /// the Gcode `E`-axis delta `gcode::emit` accumulates into a running
    /// total. `0.0` for `MoveKind::Travel`. Computed by `plan` from the
    /// segment's geometric length, its `kind`'s configured line width,
    /// `SlicerConfig::layer_height`/`filament_diameter` (see
    /// `crate::extrusion`), `extrusion_rate`, and the printing tool's
    /// `Tool::extrusion_multiplier`.
    pub extrusion_length: f64,
}

/// A single continuous toolpath (e.g. one perimeter or infill pass).
///
/// Per-segment metadata is carried in a sibling `segments` vector: for a
/// closed loop of N `points`, there are N segments — `segments[i]`
/// describes the move `points[i] -> points[(i + 1) % points.len()]`, with
/// the last segment being the closing edge back to `points[0]`. This keeps
/// `points`/`segments` as parallel `Vec`s (`segments.len() == points.len()`)
/// rather than pairing them in a single `Vec<(DVec3, Segment)>`, so callers
/// that only need geometry (e.g. bounding-box/preview code) can read
/// `points` without also touching `segments`.
#[derive(Debug, Clone, Default)]
pub struct Path {
    pub points: Vec<DVec3>,
    pub segments: Vec<Segment>,
    /// The tool this path is printed with — looked up from the layer's
    /// source object's `Tool` assignment. Lets [`crate::gcode::emit`]
    /// insert tool-change Gcode between paths assigned to different
    /// tools.
    pub tool: ToolId,
}

/// Plan toolpaths for a set of layers, tagging each planned path with the
/// tool assigned to its source object.
///
/// Emits one [`Path`] per contour loop in each [`Layer`] (a layer with no
/// loops contributes no paths), classifying each as [`MoveKind::WallOuter`]
/// or [`MoveKind::WallInner`] from its source `WallLoop::wall_index`, then
/// appends [`MoveKind::Infill`] [`Path`]s per layer (if any) generated by
/// `config.infill_pattern` (see `infill` module): one pass over the
/// layer's sparse fillable area (`InfillRegion::from_layer`, which is
/// `Layer::infill_boundary` minus `Layer::solid_fill_boundary`) at
/// `config.infill_density`, plus (when non-empty) a second pass with the
/// same generator at full density (`1.0`) over
/// `Layer::solid_fill_boundary` itself — there is no separate solid-fill
/// pattern or `MoveKind`; both passes are tagged `MoveKind::Infill`. As a
/// final safety net, every wall/infill path is then re-validated against
/// the layer's real mesh containment query (`Layer::mesh_sdf`, when
/// present) and dropped wholesale if any of its points fall outside the
/// solid -- see [`retain_contained_paths`]. Real path planning beyond
/// this (travel-move ordering/optimization across paths, non-planar
/// toolpath deformation) is future work.
/// Layers are planned in parallel across all available cores (via
/// `rayon`): each layer only reads the shared, immutable `layers`/`objects`
/// slices and produces its own independent `Vec<Path>` (the expensive part
/// per layer — `InfillRegion::from_layer`'s polygon boolean ops and the
/// scanline infill generator itself — has no cross-layer dependency to
/// serialize on). Output order matches input layer order regardless of
/// completion order, since `rayon`'s indexed `map`/`collect` preserves it.
///
/// Every segment's `Segment::extrusion_length` is finalized in a second
/// pass over the assembled paths, once all of a layer's wall/infill paths
/// exist (see `crate::extrusion`): using each segment's geometric length
/// (`points[i] -> points[(i + 1) % points.len()]`, the same wrap-around
/// contract documented on [`Path`], which works uniformly for both closed
/// wall loops and open infill polylines), its `kind`'s configured line
/// width, `config.layer_height`/`config.filament_diameter`, its
/// `extrusion_rate`, and `tools`' matching `Tool::extrusion_multiplier`
/// (looked up by the layer's object's assigned `ToolId`; defaults to
/// `1.0` if `tools` has no matching entry, so callers that don't model
/// machine tools at all — most existing tests — keep working unchanged).
///
/// # Errors
///
/// Returns [`crate::Error::InvalidMesh`] if a layer references an object
/// id not present in `objects`.
pub fn plan(
    layers: &[Layer],
    objects: &[Object],
    tools: &[Tool],
    config: &SlicerConfig,
) -> Result<Vec<Path>> {
    plan_with_progress(layers, objects, tools, config, &mut |_| {})
}

/// Same as [`plan`], but calls `on_progress` with a `0.0..=1.0` fraction of
/// how many of `layers` have finished planning so far.
///
/// Layers are planned in parallel (see [`plan`]'s docs), so completions can
/// arrive from any worker thread in any order; `on_progress` is called once
/// per completed layer, in completion order (not necessarily layer order),
/// serialized behind a `Mutex` so callers don't need to worry about
/// concurrent invocations. Reaches `1.0` once every layer is planned, right
/// before the final extrusion-length pass (which is comparatively fast and
/// not separately reported).
///
/// # Errors
///
/// Returns [`crate::Error::InvalidMesh`] if a layer references an object id
/// not present in `objects`.
pub fn plan_with_progress(
    layers: &[Layer],
    objects: &[Object],
    tools: &[Tool],
    config: &SlicerConfig,
    on_progress: &mut (dyn FnMut(f64) + Send),
) -> Result<Vec<Path>> {
    let generator = infill::generator_for(config.infill_pattern);
    let filament_area = extrusion::filament_cross_section_area(config.filament_diameter);
    let total_layers = layers.len().max(1) as f64;
    let completed = AtomicUsize::new(0);
    let on_progress = Mutex::new(on_progress);
    let per_layer: Vec<Vec<Path>> = layers
        .par_iter()
        .map(|layer| -> Result<Vec<Path>> {
            let object = objects
                .iter()
                .find(|object| object.id == layer.object)
                .ok_or_else(|| {
                    crate::Error::InvalidMesh(format!(
                        "layer references unknown object {}",
                        layer.object
                    ))
                })?;

            let mut paths = Vec::new();
            for wall_loop in &layer.loops {
                // Placeholder metadata: real support/bridge/overhang
                // classification and speed/extrusion-rate planning is future
                // work (see toolpath-metadata-phase12 subtask 03). Wall
                // classification (outer vs. inner) is derived from the loop's
                // wall index; fixed sane defaults are used for the rest since
                // they aren't yet meaningfully configurable.
                let kind = if wall_loop.wall_index == 0 {
                    MoveKind::WallOuter
                } else {
                    MoveKind::WallInner
                };
                let segments = wall_loop
                    .points
                    .iter()
                    .map(|_| Segment {
                        kind,
                        speed: speed_for_kind(kind, config),
                        extrusion_rate: 1.0,
                        support_fraction: 0.0,
                        order: layer.order,
                        extrusion_length: 0.0,
                    })
                    .collect();
                paths.push(Path {
                    points: wall_loop.points.clone(),
                    segments,
                    tool: object.tool,
                });
            }

            let region = InfillRegion::from_layer(layer, config);
            for mut infill_path in generator.generate(
                &region,
                config,
                layer,
                &object.transform,
                config.infill_density,
            ) {
                infill_path.tool = object.tool;
                paths.push(infill_path);
            }

            if !layer.solid_fill_boundary.is_empty() {
                let solid_region = InfillRegion {
                    loops: layer.solid_fill_boundary.clone(),
                };
                for mut infill_path in
                    generator.generate(&solid_region, config, layer, &object.transform, 1.0)
                {
                    infill_path.tool = object.tool;
                    paths.push(infill_path);
                }
            }

            let paths = retain_contained_paths(paths, layer.mesh_sdf.as_ref(), layer.order);
            let paths = compensate_flat_nozzle(paths, layer, config);
            let paths = simplify_paths(paths, config);
            let mut paths = insert_z_hops(paths, config);

            let extrusion_multiplier = tools
                .iter()
                .find(|tool| tool.id == object.tool)
                .map_or(1.0, |tool| tool.extrusion_multiplier);
            for path in &mut paths {
                let point_count = path.points.len();
                if point_count == 0 {
                    continue;
                }
                for (i, segment) in path.segments.iter_mut().enumerate() {
                    if segment.kind == MoveKind::Travel {
                        segment.extrusion_length = 0.0;
                        continue;
                    }
                    let distance = path.points[i].distance(path.points[(i + 1) % point_count]);
                    let line_width = extrusion::line_width_for_kind(segment.kind, config);
                    let bead_area =
                        extrusion::bead_cross_section_area(line_width, config.layer_height);
                    segment.extrusion_length =
                        extrusion::segment_extrusion_length(distance, bead_area, filament_area)
                            * segment.extrusion_rate
                            * extrusion_multiplier;
                }
            }

            let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
            if let Ok(mut on_progress) = on_progress.lock() {
                on_progress(done as f64 / total_layers);
            }

            Ok(paths)
        })
        .collect::<Result<Vec<Vec<Path>>>>()?;

    Ok(per_layer.into_iter().flatten().collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ids::ObjectId, mesh::Mesh, slicing::WallLoop};
    use manifold_fidget::order::HeightOrderField;
    use std::sync::Arc;

    use crate::slicing::BUILD_DIRECTION;

    fn path_with_points(points: Vec<DVec3>) -> Path {
        let segments = points
            .iter()
            .map(|_| Segment {
                kind: MoveKind::WallOuter,
                speed: 60.0,
                extrusion_rate: 1.0,
                support_fraction: 0.0,
                order: 0.0,
                extrusion_length: 0.0,
            })
            .collect();
        Path {
            points,
            segments,
            tool: ToolId(0),
        }
    }

    #[test]
    fn validate_within_bounds_accepts_paths_entirely_inside_the_build_volume() {
        let build_volume = BoundingVolume::Aabb {
            min: DVec3::ZERO,
            max: DVec3::new(200.0, 200.0, 200.0),
        };
        let paths = vec![path_with_points(vec![
            DVec3::new(10.0, 10.0, 0.2),
            DVec3::new(20.0, 10.0, 0.2),
        ])];

        assert!(validate_within_bounds(&paths, &build_volume).is_ok());
    }

    #[test]
    fn validate_within_bounds_rejects_a_point_outside_the_build_volume() {
        let build_volume = BoundingVolume::Aabb {
            min: DVec3::ZERO,
            max: DVec3::new(200.0, 200.0, 200.0),
        };
        let offending = DVec3::new(-0.036, 30.0, 0.46);
        let paths = vec![path_with_points(vec![
            DVec3::new(10.0, 10.0, 0.2),
            offending,
        ])];

        let err = validate_within_bounds(&paths, &build_volume).unwrap_err();
        match err {
            Error::MoveOutOfBounds { point } => assert_eq!(point, offending),
            other => panic!("expected MoveOutOfBounds, got {other:?}"),
        }
    }

    #[test]
    fn validate_within_bounds_ignores_travel_only_paths_that_are_still_out_of_bounds() {
        // Travel moves aren't checked by `retain_contained_paths` against the
        // solid mesh, but `validate_within_bounds` is a machine-envelope
        // check, not a solid-containment check -- a travel move outside the
        // build volume is just as much a real problem for the machine, so
        // it must still be caught.
        let build_volume = BoundingVolume::Aabb {
            min: DVec3::ZERO,
            max: DVec3::new(200.0, 200.0, 200.0),
        };
        let mut path = path_with_points(vec![
            DVec3::new(10.0, 10.0, 0.2),
            DVec3::new(-5.0, 10.0, 0.2),
        ]);
        for segment in &mut path.segments {
            segment.kind = MoveKind::Travel;
        }

        assert!(validate_within_bounds(&[path], &build_volume).is_err());
    }

    #[test]
    fn plan_tags_paths_with_objects_assigned_tool() {
        let objects = vec![
            Object::new(ObjectId(0), Mesh::default(), ToolId(0)),
            Object::new(ObjectId(1), Mesh::default(), ToolId(2)),
        ];
        let loop_a = vec![
            DVec3::ZERO,
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(0.0, 1.0, 0.0),
        ];
        let loop_b = vec![
            DVec3::ZERO,
            DVec3::new(2.0, 0.0, 0.0),
            DVec3::new(0.0, 2.0, 0.0),
        ];
        let layers = vec![
            Layer {
                index: 0,
                object: ObjectId(1),
                order: 0.0,
                loops: vec![WallLoop {
                    wall_index: 0,
                    points: loop_a.clone(),
                }],
                infill_boundary: Vec::new(),
                solid_fill_boundary: Vec::new(),
                mesh_sdf: None,
                order_field: Arc::new(HeightOrderField::new(BUILD_DIRECTION)),
            },
            Layer {
                index: 0,
                object: ObjectId(0),
                order: 0.0,
                loops: vec![WallLoop {
                    wall_index: 0,
                    points: loop_b.clone(),
                }],
                infill_boundary: Vec::new(),
                solid_fill_boundary: Vec::new(),
                mesh_sdf: None,
                order_field: Arc::new(HeightOrderField::new(BUILD_DIRECTION)),
            },
        ];

        let paths = plan(&layers, &objects, &[], &SlicerConfig::default()).unwrap();

        // No `infill_boundary` is set on either layer, so `plan` emits no
        // infill paths here — every emitted path is a wall path.
        let wall_paths: Vec<_> = paths
            .iter()
            .filter(|p| {
                p.segments
                    .iter()
                    .all(|segment| segment.kind == MoveKind::WallOuter)
            })
            .collect();
        assert_eq!(wall_paths.len(), 2);
        assert_eq!(wall_paths[0].tool, ToolId(2));
        assert_eq!(wall_paths[0].points, loop_a);
        assert_eq!(wall_paths[0].segments.len(), wall_paths[0].points.len());
        assert!(wall_paths[0]
            .segments
            .iter()
            .all(|segment| segment.kind == MoveKind::WallOuter));
        assert!(wall_paths[0]
            .segments
            .iter()
            .all(|segment| segment.order == 0.0));
        assert!(wall_paths[0]
            .segments
            .iter()
            .all(|segment| segment.support_fraction == 0.0));
        assert_eq!(wall_paths[1].tool, ToolId(0));
        assert_eq!(wall_paths[1].points, loop_b);
        assert_eq!(wall_paths[1].segments.len(), wall_paths[1].points.len());
        assert!(wall_paths[1]
            .segments
            .iter()
            .all(|segment| segment.kind == MoveKind::WallOuter));
        assert!(wall_paths[1]
            .segments
            .iter()
            .all(|segment| segment.order == 0.0));
        assert!(wall_paths[1]
            .segments
            .iter()
            .all(|segment| segment.support_fraction == 0.0));
    }

    #[test]
    fn plan_stamps_segment_order_from_the_source_layer() {
        let objects = vec![Object::new(ObjectId(0), Mesh::default(), ToolId(0))];
        let layers = vec![Layer {
            index: 0,
            object: ObjectId(0),
            order: 0.75,
            loops: vec![WallLoop {
                wall_index: 0,
                points: vec![
                    DVec3::ZERO,
                    DVec3::new(1.0, 0.0, 0.0),
                    DVec3::new(0.0, 1.0, 0.0),
                ],
            }],
            infill_boundary: Vec::new(),
            solid_fill_boundary: Vec::new(),
            mesh_sdf: None,
            order_field: Arc::new(HeightOrderField::new(BUILD_DIRECTION)),
        }];

        let paths = plan(&layers, &objects, &[], &SlicerConfig::default()).unwrap();

        // No `infill_boundary` is set, so `plan` emits no infill path
        // here — every emitted path is a wall path.
        let wall_paths: Vec<_> = paths
            .iter()
            .filter(|p| {
                p.segments
                    .iter()
                    .all(|segment| segment.kind == MoveKind::WallOuter)
            })
            .collect();
        assert_eq!(wall_paths.len(), 1);
        assert!(wall_paths[0]
            .segments
            .iter()
            .all(|segment| segment.order == 0.75));
    }

    #[test]
    fn speed_for_kind_uses_travel_speed_for_travel_and_print_speed_for_extruding_kinds() {
        let config = SlicerConfig {
            travel_speed: 9000.0,
            print_speed: 3000.0,
            ..SlicerConfig::default()
        };

        assert_eq!(speed_for_kind(MoveKind::Travel, &config), 9000.0);
        assert_eq!(speed_for_kind(MoveKind::WallOuter, &config), 3000.0);
        assert_eq!(speed_for_kind(MoveKind::WallInner, &config), 3000.0);
        assert_eq!(speed_for_kind(MoveKind::Infill, &config), 3000.0);
        assert_eq!(speed_for_kind(MoveKind::Bridge, &config), 3000.0);
        assert_eq!(speed_for_kind(MoveKind::Overhang, &config), 3000.0);
    }

    #[test]
    fn plan_assigns_wall_segments_the_configured_print_speed_not_a_hardcoded_value() {
        let objects = vec![Object::new(ObjectId(0), Mesh::default(), ToolId(0))];
        let layers = vec![Layer {
            index: 0,
            object: ObjectId(0),
            order: 0.0,
            loops: vec![WallLoop {
                wall_index: 0,
                points: vec![
                    DVec3::ZERO,
                    DVec3::new(1.0, 0.0, 0.0),
                    DVec3::new(0.0, 1.0, 0.0),
                ],
            }],
            infill_boundary: Vec::new(),
            solid_fill_boundary: Vec::new(),
            mesh_sdf: None,
            order_field: Arc::new(HeightOrderField::new(BUILD_DIRECTION)),
        }];
        let config = SlicerConfig {
            print_speed: 1234.0,
            ..SlicerConfig::default()
        };

        let paths = plan(&layers, &objects, &[], &config).unwrap();

        assert!(paths
            .iter()
            .flat_map(|p| p.segments.iter())
            .all(|segment| segment.speed == 1234.0));
    }

    #[test]
    fn plan_emits_no_paths_for_layer_with_no_loops() {
        let objects = vec![Object::new(ObjectId(0), Mesh::default(), ToolId(0))];
        let layers = vec![Layer {
            index: 0,
            object: ObjectId(0),
            order: 0.0,
            loops: Vec::new(),
            infill_boundary: Vec::new(),
            solid_fill_boundary: Vec::new(),
            mesh_sdf: None,
            order_field: Arc::new(HeightOrderField::new(BUILD_DIRECTION)),
        }];

        let paths = plan(&layers, &objects, &[], &SlicerConfig::default()).unwrap();

        assert!(paths.is_empty());
    }

    #[test]
    fn plan_generates_extra_infill_pass_for_solid_fill_boundary() {
        let objects = vec![Object::new(ObjectId(0), Mesh::default(), ToolId(0))];
        let square = vec![vec![
            DVec3::new(-5.0, -5.0, 0.0),
            DVec3::new(5.0, -5.0, 0.0),
            DVec3::new(5.0, 5.0, 0.0),
            DVec3::new(-5.0, 5.0, 0.0),
        ]];
        let solid_square = vec![vec![
            DVec3::new(1.0, 1.0, 0.0),
            DVec3::new(5.0, 1.0, 0.0),
            DVec3::new(5.0, 5.0, 0.0),
            DVec3::new(1.0, 5.0, 0.0),
        ]];
        let layer_no_solid = Layer {
            index: 0,
            object: ObjectId(0),
            order: 0.0,
            loops: Vec::new(),
            infill_boundary: square.clone(),
            solid_fill_boundary: Vec::new(),
            mesh_sdf: None,
            order_field: Arc::new(HeightOrderField::new(BUILD_DIRECTION)),
        };
        let layer_with_solid = Layer {
            solid_fill_boundary: solid_square,
            ..layer_no_solid.clone()
        };

        let cfg = SlicerConfig {
            infill_line_width: 0.5,
            infill_angle_deg: 0.0,
            ..SlicerConfig::default()
        };

        let paths_no_solid = plan(&[layer_no_solid], &objects, &[], &cfg).unwrap();
        let paths_with_solid = plan(&[layer_with_solid], &objects, &[], &cfg).unwrap();

        let infill_paths = |paths: &[Path]| -> usize {
            paths
                .iter()
                .filter(|p| p.segments.iter().any(|s| s.kind == MoveKind::Infill))
                .count()
        };

        // A solid_fill_boundary adds a second infill pass (sparse region +
        // solid region), both generated by the same `config.infill_pattern`
        // generator and tagged `MoveKind::Infill` — no new pattern/`MoveKind`.
        assert_eq!(infill_paths(&paths_no_solid), 1);
        assert_eq!(infill_paths(&paths_with_solid), 2);
    }

    #[test]
    fn plan_solid_fill_boundary_prints_at_full_density_even_when_infill_density_is_zero() {
        // Reuses the same square/solid-square shapes as
        // `plan_generates_extra_infill_pass_for_solid_fill_boundary`, but
        // with `infill_density: 0.0` -- the sparse pass over the region
        // outside `solid_fill_boundary` must vanish entirely, while the
        // solid pass over `solid_fill_boundary` itself must still print
        // (always full density, regardless of `config.infill_density`).
        let objects = vec![Object::new(ObjectId(0), Mesh::default(), ToolId(0))];
        let square = vec![vec![
            DVec3::new(-5.0, -5.0, 0.0),
            DVec3::new(5.0, -5.0, 0.0),
            DVec3::new(5.0, 5.0, 0.0),
            DVec3::new(-5.0, 5.0, 0.0),
        ]];
        let solid_square = vec![vec![
            DVec3::new(1.0, 1.0, 0.0),
            DVec3::new(5.0, 1.0, 0.0),
            DVec3::new(5.0, 5.0, 0.0),
            DVec3::new(1.0, 5.0, 0.0),
        ]];
        let layer = Layer {
            index: 0,
            object: ObjectId(0),
            order: 0.0,
            loops: Vec::new(),
            infill_boundary: square,
            solid_fill_boundary: solid_square,
            mesh_sdf: None,
            order_field: Arc::new(HeightOrderField::new(BUILD_DIRECTION)),
        };

        let cfg = SlicerConfig {
            infill_line_width: 0.5,
            infill_angle_deg: 0.0,
            infill_density: 0.0,
            ..SlicerConfig::default()
        };

        let paths = plan(&[layer], &objects, &[], &cfg).unwrap();

        let infill_paths: Vec<&Path> = paths
            .iter()
            .filter(|p| p.segments.iter().any(|s| s.kind == MoveKind::Infill))
            .collect();

        assert_eq!(
            infill_paths.len(),
            1,
            "expected exactly one infill pass (solid only) when infill_density is 0.0"
        );
    }

    #[test]
    fn plan_emits_one_path_per_loop_in_a_layer() {
        let objects = vec![Object::new(ObjectId(0), Mesh::default(), ToolId(0))];
        let layers = vec![Layer {
            index: 0,
            object: ObjectId(0),
            order: 0.0,
            loops: vec![
                WallLoop {
                    wall_index: 0,
                    points: vec![DVec3::ZERO, DVec3::new(1.0, 0.0, 0.0)],
                },
                WallLoop {
                    wall_index: 1,
                    points: vec![DVec3::new(2.0, 0.0, 0.0), DVec3::new(3.0, 0.0, 0.0)],
                },
            ],
            infill_boundary: Vec::new(),
            solid_fill_boundary: Vec::new(),
            mesh_sdf: None,
            order_field: Arc::new(HeightOrderField::new(BUILD_DIRECTION)),
        }];

        let paths = plan(&layers, &objects, &[], &SlicerConfig::default()).unwrap();

        // No `infill_boundary` is set, so `plan` emits no infill path
        // here — every emitted path is a wall path.
        let wall_paths: Vec<_> = paths
            .iter()
            .filter(|p| {
                p.segments.iter().all(|segment| {
                    matches!(segment.kind, MoveKind::WallOuter | MoveKind::WallInner)
                })
            })
            .collect();
        assert_eq!(wall_paths.len(), 2);
    }

    #[test]
    fn plan_classifies_nonzero_wall_index_as_wall_inner() {
        let objects = vec![Object::new(ObjectId(0), Mesh::default(), ToolId(0))];
        let layers = vec![Layer {
            index: 0,
            object: ObjectId(0),
            order: 0.0,
            loops: vec![
                WallLoop {
                    wall_index: 0,
                    points: vec![DVec3::ZERO, DVec3::new(1.0, 0.0, 0.0)],
                },
                WallLoop {
                    wall_index: 1,
                    points: vec![DVec3::new(2.0, 0.0, 0.0), DVec3::new(3.0, 0.0, 0.0)],
                },
                WallLoop {
                    wall_index: 2,
                    points: vec![DVec3::new(4.0, 0.0, 0.0), DVec3::new(5.0, 0.0, 0.0)],
                },
            ],
            infill_boundary: Vec::new(),
            solid_fill_boundary: Vec::new(),
            mesh_sdf: None,
            order_field: Arc::new(HeightOrderField::new(BUILD_DIRECTION)),
        }];

        let paths = plan(&layers, &objects, &[], &SlicerConfig::default()).unwrap();

        // No `infill_boundary` is set, so `plan` emits no infill path
        // here — every emitted path is a wall path.
        let wall_paths: Vec<_> = paths
            .iter()
            .filter(|p| {
                p.segments.iter().all(|segment| {
                    matches!(segment.kind, MoveKind::WallOuter | MoveKind::WallInner)
                })
            })
            .collect();
        assert_eq!(wall_paths.len(), 3);
        assert!(wall_paths[0]
            .segments
            .iter()
            .all(|segment| segment.kind == MoveKind::WallOuter));
        assert!(wall_paths[1]
            .segments
            .iter()
            .all(|segment| segment.kind == MoveKind::WallInner));
        assert!(wall_paths[2]
            .segments
            .iter()
            .all(|segment| segment.kind == MoveKind::WallInner));
    }

    #[test]
    fn insert_z_hops_inserts_lift_and_drop_points_around_a_travel_run_when_enabled() {
        // p0 -(WallOuter)-> p1 -(Travel)-> p2 -(WallOuter)-> p3, closed by an
        // (unused-by-emit) closing edge p3 -> p0.
        let p0 = DVec3::new(0.0, 0.0, 0.0);
        let p1 = DVec3::new(1.0, 0.0, 0.0);
        let p2 = DVec3::new(5.0, 0.0, 0.0);
        let p3 = DVec3::new(6.0, 0.0, 0.0);
        let wall_segment = Segment {
            kind: MoveKind::WallOuter,
            speed: 60.0,
            extrusion_rate: 1.0,
            support_fraction: 0.0,
            order: 0.0,
            extrusion_length: 1.23,
        };
        let travel_segment = Segment {
            kind: MoveKind::Travel,
            speed: 150.0,
            extrusion_rate: 1.0,
            support_fraction: 0.0,
            order: 0.0,
            extrusion_length: 0.0,
        };
        let path = Path {
            points: vec![p0, p1, p2, p3],
            segments: vec![wall_segment, travel_segment, wall_segment, wall_segment],
            tool: ToolId(0),
        };

        let config = SlicerConfig {
            z_hop_enabled: true,
            z_hop_height: 0.4,
            ..SlicerConfig::default()
        };

        let hopped = insert_z_hops(vec![path], &config);
        assert_eq!(hopped.len(), 1);
        let hopped = &hopped[0];

        // Exact point sequence: departure (p1) unchanged, a lift point at
        // p1's XY raised by the hop height, a drop point at p2's XY still
        // raised by the hop height, then the real arrival point (p2) at its
        // original Z -- per the scoping doc's §2 point sequence.
        assert_eq!(
            hopped.points,
            vec![
                p0,
                p1,
                DVec3::new(1.0, 0.0, 0.4),
                DVec3::new(5.0, 0.0, 0.4),
                p2,
                p3,
            ]
        );
        assert_eq!(
            hopped.segments.iter().map(|s| s.kind).collect::<Vec<_>>(),
            vec![
                MoveKind::WallOuter,
                MoveKind::Travel,
                MoveKind::Travel,
                MoveKind::Travel,
                MoveKind::WallOuter,
                MoveKind::WallOuter,
            ]
        );
        // Every inserted hop segment carries zero extrusion, matching
        // ordinary `Travel` segments.
        for kind_and_segment in hopped.segments.iter().zip(hopped.points.iter()) {
            let (segment, _) = kind_and_segment;
            if segment.kind == MoveKind::Travel {
                assert_eq!(segment.extrusion_length, 0.0);
            }
        }
        // Parallel-array invariant preserved.
        assert_eq!(hopped.points.len(), hopped.segments.len());
    }

    #[test]
    fn insert_z_hops_skips_the_hop_when_a_travel_run_is_bounded_by_infill_on_both_sides() {
        // p0 -(Infill)-> p1 -(Travel)-> p2 -(Infill)-> p3: an internal jump
        // within one infill patch's boustrophedon zigzag -- both the
        // departing and arriving edges are Infill, so this travel run
        // should be left completely unmodified (no lift/drop points), even
        // though z-hop is enabled.
        let p0 = DVec3::new(0.0, 0.0, 0.0);
        let p1 = DVec3::new(1.0, 0.0, 0.0);
        let p2 = DVec3::new(2.0, 0.0, 0.0);
        let p3 = DVec3::new(3.0, 0.0, 0.0);
        let infill_segment = Segment {
            kind: MoveKind::Infill,
            speed: 60.0,
            extrusion_rate: 1.0,
            support_fraction: 0.0,
            order: 0.0,
            extrusion_length: 1.0,
        };
        let travel_segment = Segment {
            kind: MoveKind::Travel,
            speed: 150.0,
            extrusion_rate: 1.0,
            support_fraction: 0.0,
            order: 0.0,
            extrusion_length: 0.0,
        };
        let path = Path {
            points: vec![p0, p1, p2, p3],
            segments: vec![infill_segment, travel_segment, infill_segment],
            tool: ToolId(0),
        };

        let config = SlicerConfig {
            z_hop_enabled: true,
            z_hop_height: 0.4,
            ..SlicerConfig::default()
        };

        let hopped = insert_z_hops(vec![path.clone()], &config);
        assert_eq!(hopped.len(), 1);
        assert_eq!(hopped[0].points, path.points);
        assert_eq!(
            hopped[0]
                .segments
                .iter()
                .map(|s| s.kind)
                .collect::<Vec<_>>(),
            vec![MoveKind::Infill, MoveKind::Travel, MoveKind::Infill]
        );
    }

    #[test]
    fn insert_z_hops_still_hops_a_travel_run_at_the_edge_of_the_path_even_next_to_infill() {
        // A travel run at the very *start* of the path (no preceding
        // segment at all) followed by Infill: there's no bounding segment
        // on the departure side to confirm this is an internal infill
        // jump, so it's conservatively still hopped.
        let p0 = DVec3::new(0.0, 0.0, 0.0);
        let p1 = DVec3::new(5.0, 0.0, 0.0);
        let p2 = DVec3::new(6.0, 0.0, 0.0);
        let travel_segment = Segment {
            kind: MoveKind::Travel,
            speed: 150.0,
            extrusion_rate: 1.0,
            support_fraction: 0.0,
            order: 0.0,
            extrusion_length: 0.0,
        };
        let infill_segment = Segment {
            kind: MoveKind::Infill,
            speed: 60.0,
            extrusion_rate: 1.0,
            support_fraction: 0.0,
            order: 0.0,
            extrusion_length: 1.0,
        };
        let path = Path {
            points: vec![p0, p1, p2],
            segments: vec![travel_segment, infill_segment],
            tool: ToolId(0),
        };

        let config = SlicerConfig {
            z_hop_enabled: true,
            z_hop_height: 0.4,
            ..SlicerConfig::default()
        };

        let hopped = insert_z_hops(vec![path], &config);
        assert_eq!(hopped.len(), 1);
        // Lift + drop points inserted -> more points than the original 3.
        assert!(hopped[0].points.len() > 3);
    }

    #[test]
    fn insert_z_hops_handles_open_paths_with_no_closing_segment() {
        // Open (infill-style) path: `segments.len() == points.len() - 1`,
        // no closing edge -- see `infill::MonotonicInfill::generate`'s doc
        // comment. Regression test for a panic where the tail-append step
        // assumed a closing segment always existed.
        let p0 = DVec3::new(0.0, 0.0, 0.0);
        let p1 = DVec3::new(1.0, 0.0, 0.0);
        let p2 = DVec3::new(5.0, 0.0, 0.0);
        let infill_segment = Segment {
            kind: MoveKind::Infill,
            speed: 60.0,
            extrusion_rate: 1.0,
            support_fraction: 0.0,
            order: 0.0,
            extrusion_length: 1.23,
        };
        let travel_segment = Segment {
            kind: MoveKind::Travel,
            speed: 150.0,
            extrusion_rate: 1.0,
            support_fraction: 0.0,
            order: 0.0,
            extrusion_length: 0.0,
        };
        // Only 2 segments for 3 points: no closing edge.
        let path = Path {
            points: vec![p0, p1, p2],
            segments: vec![travel_segment, infill_segment],
            tool: ToolId(0),
        };

        let config = SlicerConfig {
            z_hop_enabled: true,
            z_hop_height: 0.4,
            ..SlicerConfig::default()
        };

        let hopped = insert_z_hops(vec![path], &config);
        assert_eq!(hopped.len(), 1);
        let hopped = &hopped[0];

        assert_eq!(
            hopped.points,
            vec![
                p0,
                DVec3::new(0.0, 0.0, 0.4),
                DVec3::new(1.0, 0.0, 0.4),
                p1,
                p2
            ]
        );
        assert_eq!(
            hopped.segments.iter().map(|s| s.kind).collect::<Vec<_>>(),
            vec![
                MoveKind::Travel,
                MoveKind::Travel,
                MoveKind::Travel,
                MoveKind::Infill,
            ]
        );
        // Open path: no closing segment appended, so `segments.len() ==
        // points.len() - 1` is preserved.
        assert_eq!(hopped.segments.len(), hopped.points.len() - 1);
    }

    #[test]
    fn insert_z_hops_is_a_no_op_when_disabled() {
        let p0 = DVec3::new(0.0, 0.0, 0.0);
        let p1 = DVec3::new(1.0, 0.0, 0.0);
        let p2 = DVec3::new(5.0, 0.0, 0.0);
        let wall_segment = Segment {
            kind: MoveKind::WallOuter,
            speed: 60.0,
            extrusion_rate: 1.0,
            support_fraction: 0.0,
            order: 0.0,
            extrusion_length: 1.23,
        };
        let travel_segment = Segment {
            kind: MoveKind::Travel,
            speed: 150.0,
            extrusion_rate: 1.0,
            support_fraction: 0.0,
            order: 0.0,
            extrusion_length: 0.0,
        };
        let path = Path {
            points: vec![p0, p1, p2],
            segments: vec![wall_segment, travel_segment, wall_segment],
            tool: ToolId(0),
        };

        let config = SlicerConfig {
            z_hop_enabled: false,
            z_hop_height: 0.4,
            ..SlicerConfig::default()
        };

        let result = insert_z_hops(vec![path.clone()], &config);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].points, path.points);
        assert_eq!(
            result[0]
                .segments
                .iter()
                .map(|s| s.kind)
                .collect::<Vec<_>>(),
            path.segments.iter().map(|s| s.kind).collect::<Vec<_>>()
        );
    }

    #[test]
    fn plan_rejects_layer_with_unknown_object() {
        let objects = vec![Object::new(ObjectId(0), Mesh::default(), ToolId(0))];
        let layers = vec![Layer {
            index: 0,
            object: ObjectId(99),
            order: 0.0,
            loops: Vec::new(),
            infill_boundary: Vec::new(),
            solid_fill_boundary: Vec::new(),
            mesh_sdf: None,
            order_field: Arc::new(HeightOrderField::new(BUILD_DIRECTION)),
        }];

        let err = plan(&layers, &objects, &[], &SlicerConfig::default()).unwrap_err();
        assert!(matches!(err, crate::Error::InvalidMesh(_)));
    }

    /// Shortest distance from `p` to the segment `a -> b` (clamped, unlike
    /// [`perpendicular_distance`] which measures against the infinite
    /// line) -- used to bound how far an RDP-dropped point can end up from
    /// the simplified polyline that replaces it.
    fn point_to_segment_distance(p: DVec3, a: DVec3, b: DVec3) -> f64 {
        let ab = b - a;
        let len_sq = ab.length_squared();
        if len_sq < f64::EPSILON {
            return p.distance(a);
        }
        let t = ((p - a).dot(ab) / len_sq).clamp(0.0, 1.0);
        p.distance(a + ab * t)
    }

    /// Builds a closed 'staircase' loop: one long near-collinear run of
    /// `staircase_points` points along y = 0 (each nudged by a tiny
    /// sub-tolerance jitter, mimicking the near-collinear point runs
    /// curved-order-field contour extraction produces), closed off by two
    /// short legs up to (span, span, 0) and back to (0, span, 0) so the
    /// loop has a real 2D extent (and thus a well-defined farthest pair for
    /// closed-loop RDP splitting) rather than degenerating to a line.
    fn staircase_loop(staircase_points: usize, span: f64, jitter: f64) -> Path {
        let mut points = Vec::with_capacity(staircase_points + 2);
        for i in 0..staircase_points {
            let t = i as f64 / (staircase_points - 1) as f64;
            let x = t * span;
            let y = if i % 2 == 0 { 0.0 } else { jitter };
            points.push(DVec3::new(x, y, 0.0));
        }
        points.push(DVec3::new(span, span, 0.0));
        points.push(DVec3::new(0.0, span, 0.0));
        path_with_points(points)
    }

    #[test]
    fn simplify_paths_reduces_a_staircase_loop_while_staying_within_tolerance() {
        let tolerance = 0.05;
        let staircase_points = 100;
        let path = staircase_loop(staircase_points, 10.0, 0.01);
        let original_points = path.points.clone();
        let original_point_count = original_points.len();
        assert_eq!(path.segments.len(), original_point_count);

        let config = SlicerConfig {
            path_simplify_enabled: true,
            path_simplify_tolerance: tolerance,
            ..SlicerConfig::default()
        };
        let simplified = simplify_paths(vec![path], &config);
        assert_eq!(simplified.len(), 1);
        let simplified = &simplified[0];

        // Meaningfully fewer points: the 100-point near-collinear run
        // should collapse to a small handful.
        assert!(
            simplified.points.len() < original_point_count / 2,
            "expected meaningful reduction, got {} of {} points",
            simplified.points.len(),
            original_point_count
        );

        // Parallel-array invariant preserved (still a closed loop).
        assert_eq!(simplified.segments.len(), simplified.points.len());

        // Every original point (dropped or kept) lies within `tolerance`
        // of the simplified polyline.
        let simplified_point_count = simplified.points.len();
        for &original_point in &original_points {
            let min_dist = (0..simplified_point_count)
                .map(|i| {
                    let a = simplified.points[i];
                    let b = simplified.points[(i + 1) % simplified_point_count];
                    point_to_segment_distance(original_point, a, b)
                })
                .fold(f64::INFINITY, f64::min);
            assert!(
                min_dist <= tolerance + 1e-9,
                "point {original_point:?} is {min_dist} from the simplified polyline, exceeding tolerance {tolerance}"
            );
        }
    }

    #[test]
    fn simplify_paths_is_a_no_op_when_disabled() {
        let path = staircase_loop(50, 10.0, 0.01);
        let original = path.clone();

        let config = SlicerConfig {
            path_simplify_enabled: false,
            path_simplify_tolerance: 0.05,
            ..SlicerConfig::default()
        };
        let result = simplify_paths(vec![path], &config);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].points, original.points);
        assert_eq!(
            result[0]
                .segments
                .iter()
                .map(|s| s.kind)
                .collect::<Vec<_>>(),
            original.segments.iter().map(|s| s.kind).collect::<Vec<_>>()
        );
    }

    #[test]
    fn simplify_paths_leaves_an_already_minimal_loop_unchanged_with_near_zero_tolerance() {
        // Minimal 4-point square loop -- nothing to simplify even in
        // principle (every chain between the two farthest-apart corners is
        // just a single edge, too short to have an interior candidate).
        let points = vec![
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(1.0, 1.0, 0.0),
            DVec3::new(0.0, 1.0, 0.0),
        ];
        let path = path_with_points(points.clone());

        let config = SlicerConfig {
            path_simplify_enabled: true,
            path_simplify_tolerance: 1e-9,
            ..SlicerConfig::default()
        };
        let result = simplify_paths(vec![path], &config);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].points, points);
        assert_eq!(result[0].segments.len(), points.len());
    }

    #[test]
    fn simplify_paths_leaves_infill_paths_completely_untouched() {
        // Same near-collinear staircase shape as the wall-loop test, but
        // tagged `MoveKind::Infill` -- infill simplification is explicitly
        // out of scope for v1, so this must pass through unchanged.
        let mut path = staircase_loop(50, 10.0, 0.01);
        for segment in &mut path.segments {
            segment.kind = MoveKind::Infill;
        }
        let original = path.clone();

        let config = SlicerConfig {
            path_simplify_enabled: true,
            path_simplify_tolerance: 0.05,
            ..SlicerConfig::default()
        };
        let result = simplify_paths(vec![path], &config);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].points, original.points);
        assert_eq!(
            result[0]
                .segments
                .iter()
                .map(|s| s.kind)
                .collect::<Vec<_>>(),
            original.segments.iter().map(|s| s.kind).collect::<Vec<_>>()
        );
    }

    #[test]
    fn simplify_path_preserves_the_parallel_array_invariant_for_closed_and_open_paths() {
        let closed = staircase_loop(40, 10.0, 0.01);
        assert_eq!(closed.segments.len(), closed.points.len());
        let simplified_closed = simplify_path(closed, 0.05);
        assert_eq!(
            simplified_closed.segments.len(),
            simplified_closed.points.len()
        );

        // Open (non-closed-loop) wall-style path: segments.len() ==
        // points.len() - 1, no closing edge -- exercises
        // `simplify_open_path` directly via `simplify_path`.
        let mut open = staircase_loop(40, 10.0, 0.01);
        open.segments.pop();
        assert_eq!(open.segments.len(), open.points.len() - 1);
        let simplified_open = simplify_path(open, 0.05);
        assert_eq!(
            simplified_open.segments.len(),
            simplified_open.points.len() - 1
        );
    }

    #[test]
    fn plan_extrusion_length_reflects_simplified_segment_distances_not_pre_simplify_distances() {
        // A wall loop with a long near-collinear staircase run -- with
        // simplification enabled and a tolerance big enough to collapse
        // it, the surviving segments span a different (larger) distance
        // than any individual pre-simplify segment did, so
        // `Segment::extrusion_length` must reflect the *post-simplify*
        // geometry, not the original per-point-pair distances.
        let objects = vec![Object::new(ObjectId(0), Mesh::default(), ToolId(0))];
        let staircase_points = 60;
        let span = 10.0;
        let jitter = 0.01;
        let mut loop_points = Vec::with_capacity(staircase_points + 2);
        for i in 0..staircase_points {
            let t = i as f64 / (staircase_points - 1) as f64;
            let x = t * span;
            let y = if i % 2 == 0 { 0.0 } else { jitter };
            loop_points.push(DVec3::new(x, y, 0.0));
        }
        loop_points.push(DVec3::new(span, span, 0.0));
        loop_points.push(DVec3::new(0.0, span, 0.0));

        let layers = vec![Layer {
            index: 0,
            object: ObjectId(0),
            order: 0.0,
            loops: vec![WallLoop {
                wall_index: 0,
                points: loop_points,
            }],
            infill_boundary: Vec::new(),
            solid_fill_boundary: Vec::new(),
            mesh_sdf: None,
            order_field: Arc::new(HeightOrderField::new(BUILD_DIRECTION)),
        }];

        let config = SlicerConfig {
            path_simplify_enabled: true,
            path_simplify_tolerance: 0.05,
            ..SlicerConfig::default()
        };

        let paths = plan(&layers, &objects, &[], &config).unwrap();
        let wall_paths: Vec<_> = paths
            .iter()
            .filter(|p| {
                p.segments
                    .iter()
                    .all(|segment| segment.kind == MoveKind::WallOuter)
            })
            .collect();
        assert_eq!(wall_paths.len(), 1);
        let wall_path = wall_paths[0];

        // Simplification actually happened (fewer points than the
        // original 62-point loop) -- otherwise this test would not be
        // exercising post-simplify extrusion lengths at all.
        assert!(wall_path.points.len() < staircase_points + 2);

        let filament_area = extrusion::filament_cross_section_area(config.filament_diameter);
        let line_width = extrusion::line_width_for_kind(MoveKind::WallOuter, &config);
        let bead_area = extrusion::bead_cross_section_area(line_width, config.layer_height);
        let point_count = wall_path.points.len();
        for (i, segment) in wall_path.segments.iter().enumerate() {
            let distance = wall_path.points[i].distance(wall_path.points[(i + 1) % point_count]);
            let expected = extrusion::segment_extrusion_length(distance, bead_area, filament_area)
                * segment.extrusion_rate;
            assert!(
                (segment.extrusion_length - expected).abs() < 1e-9,
                "segment {i}: extrusion_length {} did not match post-simplify distance-derived value {expected}",
                segment.extrusion_length
            );
        }
    }
}
