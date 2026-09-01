//! Toolpath planning: layers -> ordered extrusion moves.

use crate::infill::{self, InfillRegion};
use crate::{
    bounds::BoundingVolume, extrusion, ids::ToolId, object::Object, slicing::Layer, tool::Tool,
    Error, Result, SlicerConfig,
};
use glam::DVec3;
use manifold_fidget::mesh_sdf::MeshSdf;
use manifold_fidget::order::OrderField;
use manifold_fidget::ScalarField;
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// Signed-distance threshold (mm) beyond which a single path point counts
/// as "outside the solid" for [`retain_contained_paths`]'s outside-point
/// fraction rule: half a default nozzle diameter. Wall points produced by
/// order-field reprojection near level-set topology changes (hole/bore
/// junctions) legitimately wander a couple tenths of a millimetre off the
/// exact surface, and inter-layer stitch points are deliberately allowed
/// up to one bead radius outside (see `slicing::chord_stays_in_solid`) —
/// neither must count as a containment violation.
const CONTAINMENT_POINT_SLACK: f64 = 0.2;

/// Fraction of a path's points that may sit beyond
/// [`CONTAINMENT_POINT_SLACK`] before the whole path is treated as bogus
/// geometry rather than a real path with local reprojection excursions. A
/// genuine wall loop has thousands of points with at most a handful of
/// outliers; the spurious fragment loops contour extraction shatters off
/// near topology changes are small and mostly-outside.
const CONTAINMENT_OUTSIDE_FRACTION: f64 = 0.2;

/// Drops any non-[`MoveKind::Travel`] path in `paths` that isn't contained
/// in the real solid, using `mesh_sdf` (built directly from the mesh --
/// see [`Layer::mesh_sdf`]) as ground truth rather than trusting the 2D
/// loop/boundary geometry `paths` were generated from.
///
/// This exists as a final safety net: wall/infill loop geometry is derived
/// from contour extraction and polygon boolean ops on
/// `infill_boundary`/`solid_fill_boundary`, which have (rarely) produced
/// loops that don't correspond to real solid material -- e.g. infill
/// printed inside a hole that isn't actually part of the object, or the
/// small fragment loops contour extraction shatters off near level-set
/// topology changes (a side hole meeting a bore), which stray millimetres
/// outside the mesh.
///
/// The check is deliberately graded rather than exact: real wall loops
/// near those same topology changes carry a few reprojection outliers up
/// to a couple tenths of a millimetre outside the surface, and dropping a
/// thousands-of-points wall loop for one such point visibly removes whole
/// walls from the print (a far worse defect than the excursion itself). A
/// path is dropped only when it is *grossly* wrong: some point further
/// outside than `gross_tolerance` (one nozzle diameter -- more than a
/// whole bead hanging in air), or more than
/// [`CONTAINMENT_OUTSIDE_FRACTION`] of its points beyond
/// [`CONTAINMENT_POINT_SLACK`]. A partially-valid path is dropped entirely
/// rather than clipped, since splitting it would risk producing a spurious
/// partial loop/travel move that's arguably worse than simply omitting the
/// whole (already-wrong) path.
///
/// No-op (returns `paths` unchanged) when `mesh_sdf` is `None` -- a
/// synthetic/test [`Layer`] has no ground truth to check against, so
/// containment is treated as unknown rather than enforced.
fn retain_contained_paths(
    paths: Vec<Path>,
    mesh_sdf: Option<&Arc<MeshSdf>>,
    order: f64,
    gross_tolerance: f64,
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
            let mut max_distance = f64::NEG_INFINITY;
            let mut outside_points = 0usize;
            for &p in &path.points {
                let d = mesh_sdf.sample(p).value;
                max_distance = max_distance.max(d);
                if d > CONTAINMENT_POINT_SLACK {
                    outside_points += 1;
                }
            }
            let outside_fraction = outside_points as f64 / path.points.len().max(1) as f64;
            let contained =
                max_distance <= gross_tolerance && outside_fraction <= CONTAINMENT_OUTSIDE_FRACTION;
            if !contained {
                dropped_paths += 1;
                dropped_points += path.points.len();
                tracing::debug!(
                    layer.order = order,
                    points = path.points.len(),
                    max_distance,
                    outside_fraction,
                    "dropping uncontained path"
                );
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
        .into_par_iter()
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
/// `SlicerConfig::nozzle_flat_diameter`) when the local surface normal
/// tilts away from the physical nozzle axis (`slicing::NOZZLE_DIRECTION`)
/// for non-planar printing.
///
/// The build surface's normal at a point `p` is modeled from the layer's
/// order-field isosurface: `-normalize(grad order(p))` (see
/// `order_field::numeric_gradient`), which reduces to world-up for a flat
/// `Height` field and tilts to follow the surface for `Conical`/
/// `Eikonal`. The flat land (radius `flat_radius`) lies in the plane
/// perpendicular to the *nozzle's* axis, centered on the nozzle-center
/// path -- for a perfectly flat/untilted layer (surface normal parallel
/// to the nozzle axis) this touches the true contour symmetrically all
/// around and needs no correction (that's what `SlicerConfig::wall_offset`
/// already handles for the isotropic case).
///
/// Correction is only needed where the surface normal leans away from
/// the nozzle axis: the flat's trailing edge (in the direction the
/// surface tilts) can end up off the true contour. For each point this
/// is estimated by projecting the (world-fixed) nozzle axis onto the
/// tangent plane at `p`:
///
/// - `shift_dir = normalize(nozzle_dir - normal * nozzle_dir.dot(normal))`
///   -- the component of the nozzle axis lying in `p`'s tangent plane,
///   i.e. the direction the nozzle axis leans away from this point's own
///   surface normal. This is deliberately *not* derived from the loop's
///   own tangent/travel direction: walking around an axisymmetric loop
///   (e.g. a cone's constant-slope wall) rotates the surface normal
///   purely as an artifact of going around the loop, without the
///   surface's actual cross-sectional slope (relative to the fixed
///   nozzle axis) changing at all. Since `nozzle_dir` is the same fixed
///   vector everywhere, its tangent-plane projection varies only with
///   the local normal, not with position along the loop, so it
///   naturally lands on the meridian/cross-section direction for such
///   shapes without needing to reason about travel direction at all.
/// - probing the surface normal a `flat_radius` step to either side of
///   the point along `shift_dir` gives `theta`, the angle swept between
///   the two probe normals over one flat-radius span.
/// - the shift magnitude is `flat_radius * sin(theta / 2.0)`, the lateral
///   displacement of a chord of length `flat_radius` swept through half
///   that turn angle, applied along `shift_dir`.
///
/// Before applying, each point checks *which* side actually descends
/// toward already-printed material by comparing `field.order` at each
/// probe against `layer_order + layer_height` (the order value of the
/// layer printed just before this one -- order decreases as printing
/// proceeds, see `slicing::BUILD_DIRECTION`): only the side that's
/// actually closer to already-solid material gets compensated. This is
/// the "climbing" exemption: when neither probe is meaningfully closer to
/// solid than the other (or the point is tilting away from solid on both
/// sides, e.g. an overhang-like climb), there's no already-printed
/// surface for the flat to (mis)contact, so the point is left unshifted.
///
/// Degenerate cases (missing/zero gradient, the nozzle axis parallel to
/// the surface normal so no tangent-plane component exists, or near-zero
/// curvature between the two probes) fall through as a no-op for that
/// point rather than injecting noise -- this is a best-effort geometric
/// approximation, not an exact physical simulation.
/// Classifies how a bead extruded at `p` is supported, returning
/// `(support_fraction, bed_fraction)` for
/// [`extrusion::blended_bead_cross_section_area`].
///
/// "Below" for a non-planar layer is along the order field's local
/// gradient, not world-down: previously printed material lies at *lower*
/// order values (layers are emitted in increasing `Layer::order` — see
/// `slicing::slice_mesh_with_progress`'s order walk and
/// `stitch_wall_gaps`' previous-layer convention), so the probe point is
/// one layer height *against* the normalized gradient:
/// `q = p - layer_height * normalize(grad order(p))`.
///
/// - **Bed contact** (`bed_fraction`): the build plate sits at `bed_z`
///   (the print's lowest deposited point — the "rests on floor"
///   convention). When the probe point dips below the plate, the bead is
///   squished against it: `clamp((bed_z - q.z) / layer_height, 0, 1)`,
///   i.e. `1.0` when `q` is a full layer below the plate (a true first
///   layer directly on the bed) fading to `0.0` at the plate itself (a
///   second layer sitting on the first).
/// - **Material support** (`support_fraction`): sample the mesh SDF at
///   `q`. Inside the mesh (`sdf <= 0`) *and* scheduled earlier than the
///   bead (`order(q) <= bead_order - 0.5 * layer_height`) an earlier
///   layer deposited material there — fully supported. Fraction fades
///   linearly to zero by one nozzle radius outside: `clamp(1 - sdf(q) /
///   nozzle_radius, 0, 1)`, giving overhang perimeters a smooth
///   stadium->circle flow ramp instead of a binary jump. Mesh-solid
///   material the order field schedules *later* than the bead is air at
///   deposition time and counts as no support at all — without the order
///   gate, conformal order fields (e.g. bottom-surface conforming) would
///   report full support for beads bridging not-yet-printed solid. (The
///   mesh SDF is a proxy for "printed material": exact for walls/solid
///   regions; sparse-infill interiors read as supported, which matches
///   the traditional slicer treatment of infill-on-infill.)
///
/// Degenerate cases fall back to fully-supported stadium flow (today's
/// uniform model) rather than fabricating a bridge: missing/zero
/// gradient uses `slicing::BUILD_DIRECTION` as the gradient direction
/// (exact for `Height` fields), and a missing mesh SDF returns
/// `support_fraction = 1.0`.
fn support_fractions_at(
    p: DVec3,
    bead_order: f64,
    field: &dyn manifold_fidget::order::OrderField,
    mesh_sdf: Option<&manifold_fidget::mesh_sdf::MeshSdf>,
    bed_z: f64,
    config: &SlicerConfig,
) -> (f64, f64) {
    let layer_height = config.layer_height.abs().max(f64::EPSILON);
    let gradient_dir = crate::order_field::numeric_gradient(field, p)
        .filter(|g| g.length_squared() > 1e-12 && g.is_finite())
        .map_or(crate::slicing::BUILD_DIRECTION, |g| g.normalize());
    let probe = p - layer_height * gradient_dir;

    let bed_fraction = ((bed_z - probe.z) / layer_height).clamp(0.0, 1.0);

    let support_fraction = match mesh_sdf {
        Some(sdf) => {
            let nozzle_radius = (config.nozzle_diameter / 2.0).max(f64::EPSILON);
            let distance = sdf.sample(probe).value;
            let fraction = (1.0 - distance / nozzle_radius).clamp(0.0, 1.0);
            if fraction > 0.0 {
                // Order gate: mesh-solid material only supports this bead
                // if the order field schedules it *earlier*. Solid-but-
                // later material is air at deposition time (conformal
                // fields can invert deposition order relative to the mesh).
                let probe_order = field.order(probe);
                if probe_order.is_finite() && probe_order <= bead_order - 0.5 * layer_height {
                    fraction
                } else {
                    0.0
                }
            } else {
                fraction
            }
        }
        None => 1.0,
    };

    (support_fraction, bed_fraction)
}

fn compensate_flat_nozzle(
    paths: Vec<Path>,
    layer: &Layer,
    config: &SlicerConfig,
    tools: &[Tool],
) -> Vec<Path> {
    if config.slope_compensation_mode() == crate::SlopeCompensationMode::VolumetricModulation {
        return paths;
    }
    let field = layer.order_field.as_ref();

    paths
        .into_iter()
        .map(|mut path| {
            let flat_radius = tools.iter().find(|t| t.id == path.tool).map_or_else(
                || config.nozzle_flat_diameter() / 2.0,
                |t| t.nozzle_flat_diameter() / 2.0,
            );
            if flat_radius <= f64::EPSILON {
                return path;
            }
            let is_outer_wall = path
                .segments
                .first()
                .is_some_and(|segment| matches!(segment.kind, MoveKind::WallOuter));
            if is_outer_wall && path.points.len() >= 3 {
                path.points = compensate_wall_loop_points(
                    &path.points,
                    field,
                    flat_radius,
                    config.layer_height,
                    layer.order,
                    0.5 * config.first_layer_height(),
                );
            }
            path
        })
        .collect()
}

/// Per-point worker behind [`compensate_flat_nozzle`].
///
/// On sloped non-planar layers, a flat nozzle tip (with land radius `flat_radius`)
/// tilts relative to the surface normal, causing its lowest outer perimeter edge
/// to drop toward previously deposited material.
///
/// Rather than shifting the toolpath centerline laterally in X/Y (which introduces
/// dimensional asymmetry and wall bulging), this pass elevates the nozzle along
/// the vertical axis (+Z) by `flat_radius * sin(alpha) * (1.0 - cos(alpha))`
/// where `alpha` is the surface slope angle. This ensures the lowest edge of the
/// rigid flat land clears the underlying sloped surface without plowing, while
/// preserving the true X/Y contour centerline.
fn compensate_wall_loop_points(
    points: &[DVec3],
    field: &dyn manifold_fidget::order::OrderField,
    flat_radius: f64,
    layer_height: f64,
    _layer_order: f64,
    min_extrusion_z: f64,
) -> Vec<DVec3> {
    let n_pts = points.len();
    points
        .iter()
        .enumerate()
        .map(|(i, &p)| {
            let Some(normal) = crate::order_field::numeric_gradient(field, p)
                .and_then(|g| g.try_normalize().map(|n| -n))
            else {
                return p;
            };

            // Calculate tilt angle alpha between the surface normal and the vertical nozzle axis.
            let cos_alpha = normal
                .dot(crate::slicing::NOZZLE_DIRECTION)
                .abs()
                .clamp(0.0, 1.0);
            let sin_alpha = (1.0 - cos_alpha * cos_alpha).sqrt();

            // Lowest outer edge clearance adjustment for planar slope:
            let z_slope_clearance = if sin_alpha >= 1e-4 {
                flat_radius * sin_alpha * (1.0 - cos_alpha)
            } else {
                0.0
            };

            // Transverse concave clearance adjustment (V-grooves / valleys):
            let p_prev = points[(i + n_pts - 1) % n_pts];
            let p_next = points[(i + 1) % n_pts];
            let tangent = (p_next - p_prev).try_normalize().unwrap_or(DVec3::X);
            let u_perp = tangent.cross(normal).try_normalize().unwrap_or(DVec3::ZERO);

            let z_concave_clearance = if u_perp.length_squared() > 0.5 {
                let n_plus = crate::order_field::numeric_gradient(field, p + u_perp * flat_radius)
                    .and_then(|g| g.try_normalize().map(|n| -n));
                let n_minus = crate::order_field::numeric_gradient(field, p - u_perp * flat_radius)
                    .and_then(|g| g.try_normalize().map(|n| -n));
                if let (Some(np), Some(nm)) = (n_plus, n_minus) {
                    let sin_transverse = ((np - nm).dot(u_perp) * 0.5).clamp(0.0, 1.0);
                    let cos_transverse = (1.0 - sin_transverse * sin_transverse).sqrt();
                    let flank_rise = flat_radius * sin_transverse;
                    (flank_rise * (1.0 - cos_transverse)).min(0.60 * layer_height)
                } else {
                    0.0
                }
            } else {
                0.0
            };

            let z_clearance = z_slope_clearance + z_concave_clearance;
            if z_clearance < 1e-5 {
                return p;
            }

            let mut elevated = p + crate::slicing::NOZZLE_DIRECTION * z_clearance;

            let p_proj = p.dot(crate::slicing::BUILD_DIRECTION);
            let elevated_proj = elevated.dot(crate::slicing::BUILD_DIRECTION);

            if p_proj >= min_extrusion_z && elevated_proj < min_extrusion_z {
                elevated += crate::slicing::BUILD_DIRECTION * (min_extrusion_z - elevated_proj);
            } else if elevated_proj < 0.0 {
                elevated += crate::slicing::BUILD_DIRECTION * (-elevated_proj);
            }
            elevated
        })
        .collect()
}

/// Greedily reorders `paths` to reduce travel-move distance between them,
/// controlled by `config.travel_order_optimization_enabled` (no-op,
/// `paths` unchanged, when `false`).
///
/// Without this pass, `paths` are emitted in whatever order they were
/// generated in (walls, then sparse infill, then solid fill, each in
/// generation order) with no regard for where the nozzle physically ends
/// up between them -- `gcode::emit` always starts a path with a plain
/// `G0` from the previous path's last point, however far away that is.
/// For patterns like scanline infill this routinely produces long travel
/// moves that jump across the whole layer to print one short line, then
/// jump straight back.
///
/// Uses a simple greedy nearest-neighbor heuristic (not an optimal
/// TSP solve -- that's overkill for a per-layer path list and would cost
/// far more than it saves): the first path in `paths` is kept as the
/// fixed starting anchor (its own point order/direction is never
/// touched, and there's no prior-layer nozzle position available here --
/// layers are planned independently in parallel, see [`plan_with_progress`]'s
/// docs), then each subsequent step picks whichever *remaining* path has
/// an entry point closest to the current position and appends it,
/// updating the current position to that path's exit point.
///
/// A path with no closing segment (`segments.len() + 1 == points.len()`,
/// i.e. an open path such as an infill scan-line pass -- see [`Path`]'s
/// doc comment on the parallel-array convention) may also be considered
/// *reversed* (entering from its last point, exiting from its first) if
/// that orientation is closer -- reversal is `points.reverse()` +
/// `segments.reverse()`, which is exactly self-inverse for this
/// convention (segment `i` describes `points[i] -> points[i + 1]`, so
/// reversing both arrays turns segment `i` into the same edge walked
/// backward at index `len - 2 - i`, preserving every segment's
/// kind/speed/extrusion_length -- only the direction of travel along it
/// changes). Closed loops (walls) are never reversed or start-rotated:
/// their `points[0]` is meaningful (indexed by the upstream wall-gap
/// stitching/arc-length-correspondence passes), so only their position in
/// the overall path order is changed, never their internal orientation.
///
/// This is an O(n²) scan over the remaining paths at each step, which is
/// fine for the tens-to-low-hundreds of paths typical of a single layer;
/// it does not attempt any *routing* around obstacles (see ROADMAP.md's
/// open item on travel collision avoidance) -- only which path to visit
/// next and which end to enter it from.
fn optimize_travel_order(
    mut paths: Vec<Path>,
    config: &SlicerConfig,
    z_travel_penalty: f64,
) -> Vec<Path> {
    if !config.travel_order_optimization_enabled || paths.len() <= 1 {
        return paths;
    }

    let mut ordered = Vec::with_capacity(paths.len());
    ordered.push(paths.remove(0));
    let mut current = ordered[0].points.last().copied().unwrap_or(DVec3::ZERO);

    let z_scale = z_travel_penalty.max(1.0);
    let kinematic_cost = |a: DVec3, b: DVec3| -> f64 {
        let dx = b.x - a.x;
        let dy = b.y - a.y;
        let dz = (b.z - a.z) * z_scale;
        (dx * dx + dy * dy + dz * dz).sqrt()
    };

    while !paths.is_empty() {
        let mut best_idx = 0;
        let mut best_reverse = false;
        let mut best_cost = f64::INFINITY;

        for (idx, path) in paths.iter().enumerate() {
            let Some(&start) = path.points.first() else {
                continue;
            };
            let forward_cost = kinematic_cost(current, start);
            if forward_cost < best_cost {
                best_cost = forward_cost;
                best_idx = idx;
                best_reverse = false;
            }

            let is_open = path.segments.len() + 1 == path.points.len();
            if is_open {
                if let Some(&end) = path.points.last() {
                    let reverse_cost = kinematic_cost(current, end);
                    if reverse_cost < best_cost {
                        best_cost = reverse_cost;
                        best_idx = idx;
                        best_reverse = true;
                    }
                }
            }
        }

        let mut next = paths.remove(best_idx);
        if best_reverse {
            next = reverse_open_path(next);
        }
        current = next.points.last().copied().unwrap_or(current);
        ordered.push(next);
    }

    ordered
}

/// Node budget for [`route_around_obstruction`]'s local grid search,
/// mirroring `slicing`'s Eikonal grid node budget's role of bounding
/// memory/compute for a dense grid, just scoped to the small local region
/// around one blocked travel chord rather than a whole mesh.
const MAX_TRAVEL_GRID_NODES: usize = 8_000;

/// Returns whether the straight travel chord `a -> b` crosses solid material or violates
/// `clearance` from existing printed material at any sampled point.
///
/// Uses sphere tracing (SDF ray marching) leveraging the Lipschitz-1 bound of signed distance
/// fields to safely advance across open air in large steps rather than fixed dense sampling,
/// speeding up clearance verification by 50x-100x. Near the endpoints `a` and `b`, the required
/// clearance ramps from 0 at the contact boundary up to `clearance`.
fn travel_chord_is_blocked(
    mesh_sdf: &MeshSdf,
    order_field: Option<&dyn OrderField>,
    current_order: f64,
    a: DVec3,
    b: DVec3,
    clearance: f64,
) -> bool {
    let distance = a.distance(b);
    if distance <= f64::EPSILON {
        return false;
    }
    let step = (clearance * 0.5).max(0.1);
    let samples = ((distance / step).ceil() as usize).clamp(4, 64);
    let order_epsilon = 1e-4;
    (0..=samples).any(|s| {
        let t = s as f64 / samples as f64;
        let p = a.lerp(b, t);
        let dist_from_start = t * distance;
        let dist_from_end = (1.0 - t) * distance;
        let required_clearance = clearance.min(dist_from_start).min(dist_from_end);
        let sample = mesh_sdf.sample(p);
        if sample.value < required_clearance {
            if let Some(field) = order_field {
                let p_order = field.order(p);
                if p_order.is_finite() && p_order > current_order + order_epsilon {
                    return false;
                }
            }
            true
        } else {
            false
        }
    })
}

/// Searches a bounded local grid around the straight chord `start -> end`
/// for a route that avoids solid material (queried via `mesh_sdf`),
/// using A* search over a 26-connected neighborhood.
///
/// The search region is `start`/`end`'s bounding box expanded by a margin
/// so a genuine detour around an obstruction has room to be found. `cell_size`
/// is coarsened (grown) as needed so the dense grid's total node count
/// never exceeds [`MAX_TRAVEL_GRID_NODES`].
///
/// Any accepted step with a nonzero Z component is charged `z_penalty`
/// (relative to a horizontal step of the same length), biasing the
/// search toward horizontal detours while still allowing a genuinely
/// necessary 3D diagonal route when it is cheaper than any
/// horizontal-plus-vertical alternative.
///
/// Returns `None` if no route reaches `end`'s grid cell (e.g. it is fully
/// enclosed by solid material within this local region) -- the caller
/// falls back to the plain straight chord in that case.
#[allow(clippy::too_many_arguments)]
fn route_around_obstruction(
    mesh_sdf: &MeshSdf,
    order_field: Option<&dyn OrderField>,
    current_order: f64,
    _slope_profile: &manifold_fidget::slope_profile::SlopeProfile,
    start: DVec3,
    end: DVec3,
    _cell_size: f64,
    z_penalty: f64,
    clearance: f64,
    min_travel_z: f64,
) -> Option<Vec<DVec3>> {
    let start_sample = mesh_sdf.sample(start);
    let start_normal = start_sample.gradient.try_normalize();
    let start_clear = if let Some(n) = start_normal {
        let p = start + n * clearance;
        let p = DVec3::new(p.x, p.y, p.z.max(min_travel_z));
        if mesh_sdf.sample(p).value >= start_sample.value {
            p
        } else {
            DVec3::new(start.x, start.y, start.z.max(min_travel_z))
        }
    } else {
        DVec3::new(start.x, start.y, start.z.max(min_travel_z))
    };

    let end_sample = mesh_sdf.sample(end);
    let end_normal = end_sample.gradient.try_normalize();
    let end_clear = if let Some(n) = end_normal {
        let p = end + n * clearance;
        let p = DVec3::new(p.x, p.y, p.z.max(min_travel_z));
        if mesh_sdf.sample(p).value >= end_sample.value {
            p
        } else {
            DVec3::new(end.x, end.y, end.z.max(min_travel_z))
        }
    } else {
        DVec3::new(end.x, end.y, end.z.max(min_travel_z))
    };

    let search_start = start_clear;
    let search_end = end_clear;

    let base_cell = (clearance * 0.5).max(0.4);
    let chord_dist = start.distance(end);
    let margin = (chord_dist * 0.75).max(clearance * 6.0).max(10.0);
    // Never allow the search grid to extend below the travel Z floor
    // (min(start.z, end.z) with minimum bed clearance) to prevent nozzle collisions
    // with the build plate, bed clips, or texture ridges.
    let min = DVec3::new(
        start.x.min(end.x).min(search_start.x).min(search_end.x) - margin,
        start.y.min(end.y).min(search_start.y).min(search_end.y) - margin,
        (start.z.min(end.z).min(search_start.z).min(search_end.z) - margin).max(min_travel_z),
    );
    let max = DVec3::new(
        start.x.max(end.x).max(search_start.x).max(search_end.x) + margin,
        start.y.max(end.y).max(search_start.y).max(search_end.y) + margin,
        start.z.max(end.z).max(search_start.z).max(search_end.z) + margin,
    );
    let extent = max - min;

    let dims_for = |cell: f64| -> [usize; 3] {
        [
            ((extent.x / cell).ceil() as usize + 1).max(2),
            ((extent.y / cell).ceil() as usize + 1).max(2),
            ((extent.z / cell).ceil() as usize + 1).max(2),
        ]
    };
    let mut cell = base_cell;
    let mut dims = dims_for(cell);
    while dims[0] * dims[1] * dims[2] > MAX_TRAVEL_GRID_NODES {
        cell *= 1.25;
        dims = dims_for(cell);
    }

    let index_of = |p: DVec3| -> [usize; 3] {
        [
            (((p.x - min.x) / cell).round() as isize).clamp(0, dims[0] as isize - 1) as usize,
            (((p.y - min.y) / cell).round() as isize).clamp(0, dims[1] as isize - 1) as usize,
            (((p.z - min.z) / cell).round() as isize).clamp(0, dims[2] as isize - 1) as usize,
        ]
    };
    let point_of = |idx: [usize; 3]| -> DVec3 {
        DVec3::new(
            min.x + idx[0] as f64 * cell,
            min.y + idx[1] as f64 * cell,
            min.z + idx[2] as f64 * cell,
        )
    };
    let flat = |idx: [usize; 3]| -> usize { (idx[2] * dims[1] + idx[1]) * dims[0] + idx[0] };
    let coords_of = |flat_idx: usize| -> [usize; 3] {
        [
            flat_idx % dims[0],
            (flat_idx / dims[0]) % dims[1],
            flat_idx / (dims[0] * dims[1]),
        ]
    };

    let start_idx = index_of(search_start);
    let end_idx = index_of(search_end);
    if start_idx == end_idx {
        let direct_start = index_of(start);
        let direct_end = index_of(end);
        if direct_start == direct_end {
            let lift_z = start.z.max(end.z) + clearance * 2.0;
            let mid = (start + end) * 0.5;
            let lift_pt = DVec3::new(mid.x, mid.y, lift_z);
            if !travel_chord_is_blocked(
                mesh_sdf,
                order_field,
                current_order,
                start,
                lift_pt,
                clearance * 0.5,
            ) && !travel_chord_is_blocked(
                mesh_sdf,
                order_field,
                current_order,
                lift_pt,
                end,
                clearance * 0.5,
            ) {
                return Some(vec![start, lift_pt, end]);
            }
            return None;
        }
    }

    let total = dims[0] * dims[1] * dims[2];
    let start_flat = flat(start_idx);
    let end_flat = flat(end_idx);

    #[derive(Copy, Clone, PartialEq)]
    struct HeapEntry {
        f_score: f64,
        cost: f64,
        idx: usize,
    }
    impl Eq for HeapEntry {}
    impl Ord for HeapEntry {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            other
                .f_score
                .partial_cmp(&self.f_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        }
    }
    impl PartialOrd for HeapEntry {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }

    let heuristic = |p: DVec3| -> f64 {
        let d = search_end - p;
        (d.x * d.x + d.y * d.y + (d.z * z_penalty).powi(2)).sqrt()
    };

    let mut best_cost = vec![f64::INFINITY; total];
    let mut came_from: Vec<Option<usize>> = vec![None; total];
    let mut clearance_memo: Vec<u8> = vec![0; total];
    best_cost[start_flat] = 0.0;
    let mut heap = std::collections::BinaryHeap::new();
    heap.push(HeapEntry {
        f_score: heuristic(search_start),
        cost: 0.0,
        idx: start_flat,
    });

    let mut neighbor_offsets: Vec<(isize, isize, isize)> = Vec::with_capacity(26);
    for dx in -1..=1 {
        for dy in -1..=1 {
            for dz in -1..=1 {
                if dx == 0 && dy == 0 && dz == 0 {
                    continue;
                }
                neighbor_offsets.push((dx, dy, dz));
            }
        }
    }

    while let Some(HeapEntry { cost, idx, .. }) = heap.pop() {
        if idx == end_flat {
            break;
        }
        if cost > best_cost[idx] {
            continue;
        }
        let cur = coords_of(idx);
        let cur_point = point_of(cur);
        for &(dx, dy, dz) in &neighbor_offsets {
            let nx = cur[0] as isize + dx;
            let ny = cur[1] as isize + dy;
            let nz = cur[2] as isize + dz;
            if nx < 0
                || ny < 0
                || nz < 0
                || nx >= dims[0] as isize
                || ny >= dims[1] as isize
                || nz >= dims[2] as isize
            {
                continue;
            }
            let neighbor = [nx as usize, ny as usize, nz as usize];
            let neighbor_flat = flat(neighbor);
            let neighbor_point = point_of(neighbor);

            let is_clear = if neighbor_flat == end_flat || neighbor_flat == start_flat {
                true
            } else {
                let status = clearance_memo[neighbor_flat];
                if status == 0 {
                    let sample = mesh_sdf.sample(neighbor_point);
                    let clear = if sample.value >= clearance {
                        true
                    } else if let Some(field) = order_field {
                        let p_order = field.order(neighbor_point);
                        p_order.is_finite() && p_order > current_order + 1e-4
                    } else {
                        false
                    };
                    clearance_memo[neighbor_flat] = if clear { 2 } else { 1 };
                    clear
                } else {
                    status == 2
                }
            };

            if !is_clear {
                continue;
            }

            let step = neighbor_point - cur_point;
            let step_cost =
                (step.x * step.x + step.y * step.y + (step.z * z_penalty).powi(2)).sqrt();
            let next_cost = cost + step_cost;

            if next_cost < best_cost[neighbor_flat] {
                best_cost[neighbor_flat] = next_cost;
                came_from[neighbor_flat] = Some(idx);
                let f_score = next_cost + heuristic(neighbor_point);
                heap.push(HeapEntry {
                    f_score,
                    cost: next_cost,
                    idx: neighbor_flat,
                });
            }
        }
    }

    if !best_cost[end_flat].is_finite() {
        return None;
    }

    let mut path_indices = vec![end_flat];
    let mut cur = end_flat;
    while cur != start_flat {
        let Some(prev) = came_from[cur] else {
            break;
        };
        path_indices.push(prev);
        cur = prev;
    }
    path_indices.reverse();

    let mut waypoints: Vec<DVec3> = vec![start];
    if start_clear.distance(start) > 1e-4 {
        waypoints.push(start_clear);
    }
    for &idx in &path_indices {
        let p = point_of(coords_of(idx));
        if waypoints.last().is_none_or(|last| last.distance(p) > 1e-4) {
            waypoints.push(p);
        }
    }
    if end_clear.distance(end) > 1e-4
        && waypoints
            .last()
            .is_none_or(|last| last.distance(end_clear) > 1e-4)
    {
        waypoints.push(end_clear);
    }
    if waypoints
        .last()
        .is_none_or(|last| last.distance(end) > 1e-4)
    {
        waypoints.push(end);
    }

    if waypoints.len() > 2 {
        let last_idx = waypoints.len() - 1;
        for pt in &mut waypoints[1..last_idx] {
            pt.z = pt.z.max(min_travel_z);
        }
    }

    Some(waypoints)
}

/// Routes travel moves whose straight-line chord would cross solid
/// material around it -- see ROADMAP.md's former "travel collision
/// avoidance" open item.
///
/// Runs after [`optimize_travel_order`] (so it operates on the final path
/// order) and before [`insert_z_hops`] (Z-hop still applies on top of any
/// inserted routing waypoints). For each pair of consecutive paths,
/// checks whether the straight travel chord connecting the first path's
/// last point to the second path's first point stays clear of solid
/// material (via [`travel_chord_is_blocked`] against `layer_mesh_sdf`);
/// if it does not, searches a bounded local grid (via
/// [`route_around_obstruction`], gated by `slope_profile`'s per-height
/// climb limit) for a path around the obstruction and inserts it as a new
/// all-[`MoveKind::Travel`] [`Path`] between the two, tagged with the
/// preceding path's tool. Falls back to leaving the plain straight chord
/// in place (today's behavior, i.e. no-op) when
/// `config.travel_collision_avoidance_enabled` is `false`, when
/// `layer_mesh_sdf` is `None` (e.g. a synthetic/test [`Layer`] with no
/// real mesh), when no obstruction is detected, or when the search finds
/// no feasible route.
fn route_travel_moves(
    paths: Vec<Path>,
    layer_mesh_sdf: Option<&MeshSdf>,
    order_field: Option<&dyn OrderField>,
    slope_profile: &manifold_fidget::slope_profile::SlopeProfile,
    config: &SlicerConfig,
    z_penalty: f64,
) -> Vec<Path> {
    if !config.travel_collision_avoidance_enabled || paths.len() < 2 {
        return paths;
    }
    let Some(mesh_sdf) = layer_mesh_sdf else {
        return paths;
    };

    let clearance = 2.0
        * config
            .wall_line_width
            .abs()
            .max(config.nozzle_diameter.abs());
    let cell_size = config
        .layer_height
        .abs()
        .min(config.nozzle_diameter.abs())
        .max(f64::EPSILON)
        / 2.0;

    let pairs: Vec<(usize, DVec3, DVec3, crate::ids::ToolId, f64)> = (0..paths.len() - 1)
        .filter_map(|i| {
            let a = paths[i].points.last().copied()?;
            let b = paths[i + 1].points.first().copied()?;
            if a.distance(b) <= f64::EPSILON {
                return None;
            }
            let tool = paths[i].tool;
            let order = paths[i].segments.first().map(|s| s.order).unwrap_or(0.0);
            Some((i, a, b, tool, order))
        })
        .collect();

    if pairs.is_empty() {
        return paths;
    }

    let min_bed_clearance = 0.5 * config.first_layer_height();
    let detours: Vec<(usize, Path)> = pairs
        .into_par_iter()
        .filter_map(|(i, a, b, tool, order)| {
            if !travel_chord_is_blocked(mesh_sdf, order_field, order, a, b, clearance) {
                return None;
            }
            let min_travel_z = a.z.min(b.z).max(min_bed_clearance);
            let waypoints = route_around_obstruction(
                mesh_sdf,
                order_field,
                order,
                slope_profile,
                a,
                b,
                cell_size,
                z_penalty,
                clearance,
                min_travel_z,
            )?;
            if waypoints.len() < 2 {
                return None;
            }
            let segment_count = waypoints.len() - 1;
            let segments = (0..segment_count)
                .map(|_| Segment {
                    kind: MoveKind::Travel,
                    speed: speed_for_kind(MoveKind::Travel, config),
                    extrusion_rate: 0.0,
                    support_fraction: 0.0,
                    order,
                    extrusion_length: 0.0,
                    line_width: 0.0,
                    is_scarf: false,
                })
                .collect();
            Some((
                i,
                Path {
                    points: waypoints,
                    segments,
                    tool,
                },
            ))
        })
        .collect();

    if detours.is_empty() {
        return paths;
    }

    let mut detour_map = std::collections::HashMap::with_capacity(detours.len());
    for (i, detour) in detours {
        detour_map.insert(i, detour);
    }

    let mut routed = Vec::with_capacity(paths.len() + detour_map.len());
    for (i, path) in paths.into_iter().enumerate() {
        routed.push(path);
        if let Some(detour) = detour_map.remove(&i) {
            routed.push(detour);
        }
    }

    routed
}

/// Reverses an open [`Path`]'s traversal direction in place: `points` and
/// `segments` both reversed. Self-inverse and metadata-preserving -- see
/// [`optimize_travel_order`]'s doc comment for why this works for the
/// `segments[i]` describes `points[i] -> points[i + 1]` convention. Only
/// valid for open paths (`segments.len() + 1 == points.len()`); callers
/// must not use this on closed loops, where `points[0]` carries meaning
/// from upstream passes.
fn reverse_open_path(mut path: Path) -> Path {
    path.points.reverse();
    path.segments.reverse();
    path
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
                tracing::error!(
                    "Planned move at [{:.6}, {:.6}, {:.6}] lies outside the machine's build volume: {:?}",
                    point.x,
                    point.y,
                    point.z,
                    build_volume
                );
                return Err(Error::MoveOutOfBounds { point });
            }
        }
    }
    Ok(())
}

/// Chooses the Gcode feedrate (`Segment::speed`, mm/min) for a segment of
/// the given `kind`, from `config` via its [`crate::kinematics::MotionModel`].
#[must_use]
pub fn speed_for_kind(kind: MoveKind, config: &SlicerConfig) -> f64 {
    use crate::kinematics::MotionModel;
    config.motion_model().max_feedrate(kind, false)
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
    /// A wall-0 point with solid mesh material directly beneath it (real
    /// support, per [`WallLoop::top_surface`]/[`WallLoop::unsupported`])
    /// but nothing solid directly above it one nozzle-diameter along
    /// `-BUILD_DIRECTION` -- i.e. the last printed point before open air
    /// going forward, the roof of the part rather than an unsupported
    /// gap. Distinct from `Overhang` (nothing solid *beneath*): a point
    /// can be `TopSurface` while still fully supported from below, and
    /// this classification never overrides a genuine `Overhang` -- see
    /// `plan`'s segment-classification loop.
    TopSurface,
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
    /// Dynamic physical line width (mm) of this segment's deposited bead.
    pub line_width: f64,
    /// Whether this segment is part of a non-planar scarf joint seam ramp.
    pub is_scarf: bool,
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
/// loops contributes no paths), classifying each segment as
/// [`MoveKind::WallOuter`] or [`MoveKind::WallInner`] from its source
/// `WallLoop::wall_index`, except that a segment whose *destination* point
/// is `WallLoop::unsupported == true` (see `slicing::WallLoop`, populated by
/// the inter-layer wall-gap stitching pass) is classified
/// [`MoveKind::Overhang`] instead, then
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
/// this (non-planar toolpath deformation) is future work. Travel-move
/// ordering IS optimized -- see [`optimize_travel_order`] -- but travel
/// *routing*/collision avoidance around already-printed geometry is not
/// yet implemented (travel moves are still straight lines between their
/// endpoints).
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
    plan_with_progress(
        layers,
        objects,
        tools,
        config,
        None,
        &manifold_fidget::slope_profile::SlopeProfile::new(Vec::new()),
        &mut |_| {},
    )
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
    machine: Option<&crate::machine::Machine>,
    slope_profile: &manifold_fidget::slope_profile::SlopeProfile,
    on_progress: &mut (dyn FnMut(f64) + Send),
) -> Result<Vec<Path>> {
    let sparse_generator = infill::generator_for(config.sparse_infill_pattern());
    let solid_generator = infill::generator_for(config.solid_infill_pattern());
    let filament_area = extrusion::filament_cross_section_area(config.filament_diameter);
    // Build-plate height: the lowest wall-loop point across the whole
    // print rests on the bed (the "rests on floor" convention shared with
    // `object::center_on_bed` and the Eikonal seeding — the plate is at
    // the part's minimum Z, not necessarily world z=0). Used by
    // `support_fractions_at` to detect beads squished directly against
    // the plate. `INFINITY` when there are no loops at all, which makes
    // every bed test come back false.
    let bed_z = layers
        .iter()
        .flat_map(|layer| &layer.loops)
        .flat_map(|wall| &wall.points)
        .map(|p| p.z)
        .fold(f64::INFINITY, f64::min);
    let order_min = layers
        .iter()
        .map(|layer| layer.order)
        .fold(f64::INFINITY, f64::min);
    let total_layers = layers.len().max(1) as f64;
    let completed = AtomicUsize::new(0);
    let on_progress = Mutex::new(on_progress);
    let z_travel_penalty = config.resolved_z_travel_penalty(machine);

    let wave_overhang_plan = if config.wave_overhangs_enabled() {
        let default_tool = objects.first().map(|o| o.tool).unwrap_or(ToolId(0));
        crate::wave_overhang::plan_wave_overhangs(layers, config, default_tool)
    } else {
        crate::wave_overhang::WaveOverhangPlan::default()
    };

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

            let is_layer_0 = layer.index == 0 || (layer.order - order_min).abs() < 1e-6;
            let mut paths = Vec::new();
            for (w_idx, wall_loop) in layer.loops.iter().enumerate() {
                // Placeholder metadata: real support/bridge/overhang
                // classification and speed/extrusion-rate planning is future
                // work (see toolpath-metadata-phase12 subtask 03). Wall
                // classification (outer vs. inner) is derived from the loop's
                // wall index; fixed sane defaults are used for the rest since
                // they aren't yet meaningfully configurable.
                let base_kind = if wall_loop.wall_index == 0 {
                    MoveKind::WallOuter
                } else {
                    MoveKind::WallInner
                };
                let point_count = wall_loop.points.len();
                let segments = (0..point_count)
                    .map(|i| {
                        // Segment `i` is the move `points[i] -> points[(i + 1) %
                        // point_count]` (see `Path`'s doc comment on the
                        // `points`/`segments` parallel-array convention, which
                        // also applies here to `WallLoop::points`/`unsupported`).
                        // We classify a segment as `Overhang` when its
                        // *destination* point is `unsupported == true`, rather
                        // than either endpoint: a stitched or wave-overhang
                        // point is unsupported because there's no order-field/mesh
                        // surface directly beneath it, and that lack of support
                        // applies to the bead being laid down as the nozzle
                        // arrives at (i.e. extrudes into) that point.
                        let dest = (i + 1) % point_count.max(1);
                        let is_unsupported =
                            wall_loop.unsupported.get(dest).copied().unwrap_or(false)
                                || wave_overhang_plan
                                    .wall_overhang_tags_by_layer
                                    .get(layer.index)
                                    .and_then(|l| l.get(w_idx))
                                    .and_then(|w| w.get(dest))
                                    .copied()
                                    .unwrap_or(false);
                        let kind = if is_unsupported {
                            MoveKind::Overhang
                        } else if wall_loop.top_surface.get(dest).copied().unwrap_or(false) {
                            MoveKind::TopSurface
                        } else {
                            base_kind
                        };
                        let line_w = wall_loop
                            .line_widths
                            .get(dest)
                            .copied()
                            .unwrap_or(config.wall_line_width);
                        Segment {
                            kind,
                            speed: speed_for_kind(kind, config),
                            extrusion_rate: 1.0,
                            support_fraction: 0.0,
                            order: layer.order,
                            extrusion_length: 0.0,
                            line_width: line_w,
                            is_scarf: false,
                        }
                    })
                    .collect();
                paths.push(Path {
                    points: wall_loop.points.clone(),
                    segments,
                    tool: object.tool,
                });
            }

            let region = InfillRegion::from_layer(layer, config);
            for mut infill_path in sparse_generator.generate(
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
                    solid_generator.generate(&solid_region, config, layer, &object.transform, 1.0)
                {
                    infill_path.tool = object.tool;
                    paths.push(infill_path);
                }
            }

            if let Some(wave_paths) = wave_overhang_plan.paths_by_layer.get(layer.index) {
                for mut wp in wave_paths.clone() {
                    wp.tool = object.tool;
                    paths.push(wp);
                }
            }

            let paths = retain_contained_paths(
                paths,
                layer.mesh_sdf.as_ref(),
                layer.order,
                config.nozzle_diameter,
            );
            let paths = compensate_flat_nozzle(paths, layer, config, tools);
            let paths = simplify_paths(paths, config);
            let paths = optimize_travel_order(paths, config, z_travel_penalty);
            let paths = route_travel_moves(
                paths,
                layer.mesh_sdf.as_deref(),
                Some(layer.order_field.as_ref()),
                slope_profile,
                config,
                z_travel_penalty,
            );
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
                    let start = path.points[i];
                    let end = path.points[(i + 1) % point_count];
                    let diff = end - start;
                    let distance = diff.length();
                    let unit_dir = if distance > 1e-6 {
                        diff / distance
                    } else {
                        DVec3::ZERO
                    };
                    let nozzle_parallel_comp = unit_dir.dot(crate::slicing::NOZZLE_DIRECTION);
                    let climb_slope = unit_dir.dot(crate::slicing::BUILD_DIRECTION);

                    // Trajectory slope cosine compensation:
                    // A nozzle oriented along NOZZLE_DIRECTION sweeps a volume proportional to its
                    // orthogonal cross-section (distance * cos(beta)).
                    // On sloped moves, 3D arc length L_3D is inflated by 1 / cos(beta) = sec(beta),
                    // which causes severe overextrusion if uncompensated.
                    // Multiplying by cos(beta) = sqrt(1 - (dir . NOZZLE_DIRECTION)^2) preserves exact
                    // volumetric consistency across all slope angles.
                    let slope_cosine = (1.0 - nozzle_parallel_comp * nozzle_parallel_comp)
                        .max(0.0)
                        .sqrt()
                        .clamp(0.05, 1.0);
                    let effective_distance = distance * slope_cosine;

                    let line_width = if segment.line_width > 1e-4 {
                        segment.line_width
                    } else {
                        extrusion::line_width_for_kind(segment.kind, config)
                    };
                    let (support_fraction, bed_fraction) = support_fractions_at(
                        (start + end) * 0.5,
                        segment.order,
                        layer.order_field.as_ref(),
                        layer.mesh_sdf.as_deref(),
                        bed_z,
                        config,
                    );
                    // Stored for downstream consumers (flow visualization,
                    // future speed planning): the effective "how supported is
                    // this bead" figure actually used for its flow, with bed
                    // contact counting as full support.
                    segment.support_fraction = support_fraction.max(bed_fraction);
                    let is_first_layer =
                        bed_fraction > 0.0 || (layer.order - order_min).abs() < 1e-6;
                    let effective_line_width = if is_first_layer {
                        config.first_layer_line_width()
                    } else {
                        line_width
                    };
                    let effective_layer_height = if is_first_layer {
                        config.first_layer_height()
                    } else {
                        config.layer_height
                    };
                    let first_layer_mult = if is_first_layer {
                        config.first_layer_extrusion_multiplier()
                    } else {
                        1.0
                    };
                    let is_overhang = segment.kind == MoveKind::Overhang;
                    let p_prev = path.points[(i + point_count - 1) % point_count];
                    let r_curvature = extrusion::in_plane_radius_of_curvature(p_prev, start, end);
                    let raw_bead_area = if is_overhang {
                        let track_w = config.nozzle_diameter - config.wave_overhang_overlap();
                        track_w * effective_layer_height * config.wave_overhang_flow()
                    } else {
                        extrusion::blended_bead_cross_section_area(
                            effective_line_width,
                            effective_layer_height,
                            config.nozzle_diameter,
                            support_fraction,
                            bed_fraction,
                        )
                    };
                    let bead_area = if config.curvature_compensation_enabled() && !is_overhang {
                        let r_curv_area = extrusion::curvature_compensated_bead_area(
                            raw_bead_area,
                            effective_line_width,
                            r_curvature,
                        );
                        // Transverse concavity / V-groove compensation:
                        let mid_pt = (start + end) * 0.5;
                        if let Some(normal) =
                            crate::order_field::numeric_gradient(layer.order_field.as_ref(), mid_pt)
                                .and_then(|g| g.try_normalize().map(|n| -n))
                        {
                            let u_perp = unit_dir
                                .cross(normal)
                                .try_normalize()
                                .unwrap_or(DVec3::ZERO);
                            if u_perp.length_squared() > 0.5 {
                                let r_flat = config.nozzle_flat_diameter() * 0.5;
                                let n_plus = crate::order_field::numeric_gradient(
                                    layer.order_field.as_ref(),
                                    mid_pt + u_perp * r_flat,
                                )
                                .and_then(|g| g.try_normalize().map(|n| -n));
                                let n_minus = crate::order_field::numeric_gradient(
                                    layer.order_field.as_ref(),
                                    mid_pt - u_perp * r_flat,
                                )
                                .and_then(|g| g.try_normalize().map(|n| -n));
                                if let (Some(np), Some(nm)) = (n_plus, n_minus) {
                                    let sin_transverse =
                                        ((np - nm).dot(u_perp) * 0.5).clamp(0.0, 1.0);
                                    let flank_rise = r_flat * sin_transverse;
                                    extrusion::concavity_compensated_bead_area(
                                        r_curv_area,
                                        effective_layer_height,
                                        flank_rise,
                                        sin_transverse,
                                    )
                                } else {
                                    r_curv_area
                                }
                            } else {
                                r_curv_area
                            }
                        } else {
                            r_curv_area
                        }
                    } else {
                        raw_bead_area
                    };

                    // Directional slope flow compensation:
                    let directional_flow_mult = match config.slope_compensation_mode() {
                        crate::SlopeCompensationMode::GeometricOffset => {
                            if climb_slope >= 0.0 {
                                1.0
                            } else {
                                1.0 - 0.12 * (-climb_slope).clamp(0.0, 1.0)
                            }
                        }
                        crate::SlopeCompensationMode::VolumetricModulation => {
                            let flat_diam = config.nozzle_flat_diameter();
                            let bead_w = line_width.max(1e-3);
                            let squeeze_ratio = (flat_diam / (2.0 * bead_w)).clamp(0.5, 2.5);
                            if climb_slope >= 0.0 {
                                1.0 + 0.03 * climb_slope.clamp(0.0, 1.0)
                            } else {
                                let descent_sin = (-climb_slope).clamp(0.0, 1.0);
                                (1.0 - (0.15 * squeeze_ratio * descent_sin)).max(0.60)
                            }
                        }
                    };

                    let fluid_engine = config.fluid_dynamics_engine();
                    let motion_model = config.resolved_motion_model(machine);
                    let nominal_speed = if is_overhang {
                        config.wave_overhang_speed()
                    } else {
                        motion_model.max_directional_feedrate(
                            segment.kind,
                            is_first_layer,
                            unit_dir,
                        )
                    };
                    let clamped_speed = crate::kinematics::clamp_feedrate_by_volumetric_limit(
                        nominal_speed,
                        bead_area,
                        config.max_volumetric_speed,
                    );
                    segment.speed = clamped_speed;

                    let swell_mult = if let Some(ref engine) = fluid_engine {
                        let flow_q = ((clamped_speed / 60.0) * bead_area).max(0.01);
                        engine.swell_volume_multiplier(flow_q, 0.0)
                    } else {
                        1.0
                    };

                    segment.extrusion_length = extrusion::segment_extrusion_length(
                        effective_distance,
                        bead_area,
                        filament_area,
                    ) * segment.extrusion_rate
                        * extrusion_multiplier
                        * first_layer_mult
                        * directional_flow_mult
                        * swell_mult;
                }
            }

            if config.scarf_joint_enabled {
                let scarf_len = config.scarf_joint_length();
                let scarf_steps = config.scarf_joint_steps();
                let scarf_start_h = config.scarf_joint_start_height_fraction();
                let scarf_flow = config.scarf_joint_flow_ratio();
                let layer_h = if is_layer_0 {
                    config.first_layer_height()
                } else {
                    config.layer_height
                };
                let fluid_engine = config.fluid_dynamics_engine();
                for path in &mut paths {
                    crate::kinematics::apply_scarf_joint(
                        &mut path.points,
                        &mut path.segments,
                        scarf_len,
                        scarf_steps,
                        scarf_start_h,
                        scarf_flow,
                        layer_h,
                        Some(layer.order_field.as_ref()),
                        fluid_engine.as_ref(),
                        config.slope_compensation_mode(),
                    );
                }
            }

            if let Some(taper_dist) = config.pre_retract_taper_distance {
                if taper_dist > 0.0 {
                    for path in &mut paths {
                        // Skip pre-retract taper on paths that already have a scarf joint seam
                        if config.scarf_joint_enabled && path.segments.iter().any(|s| s.is_scarf) {
                            continue;
                        }
                        crate::kinematics::apply_pre_retract_taper(
                            &mut path.points,
                            &mut path.segments,
                            taper_dist,
                            0.20,
                        );
                    }
                }
            }

            if config.wipe_enabled {
                let wipe_dist = config.wipe_distance();
                for path in &mut paths {
                    crate::kinematics::apply_wipe_moves(
                        &mut path.points,
                        &mut path.segments,
                        wipe_dist,
                    );
                }
            }

            // Drop unprintable micro-paths whose total extruding length is negligible (< 0.5 * nozzle_diameter or total E < 0.0005 mm)
            let min_extruding_distance = config.nozzle_diameter * 0.5;
            let mut paths: Vec<Path> = paths
                .into_iter()
                .filter(|path| {
                    let n = path.points.len();
                    if n < 2 {
                        return false;
                    }
                    let mut total_d = 0.0;
                    let mut total_e = 0.0;
                    for (i, segment) in path.segments.iter().enumerate() {
                        if segment.kind != MoveKind::Travel {
                            let p0 = path.points[i];
                            let p1 = path.points[(i + 1) % n];
                            total_d += p0.distance(p1);
                            total_e += segment.extrusion_length;
                        }
                    }
                    total_d >= min_extruding_distance && total_e >= 0.0005
                })
                .collect();

            // Ensure no planned points dip below the build bed floor (Z = 0.0)
            for path in &mut paths {
                for pt in &mut path.points {
                    let p_z = pt.dot(crate::slicing::BUILD_DIRECTION);
                    if p_z < 0.0 {
                        *pt += crate::slicing::BUILD_DIRECTION * (-p_z);
                    }
                }
            }

            let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
            if let Ok(mut on_progress) = on_progress.lock() {
                on_progress((done as f64 / total_layers) * 0.9);
            }

            Ok(paths)
        })
        .collect::<Result<Vec<Vec<Path>>>>()?;

    let all_paths = defer_unsupported_paths(per_layer, layers, config);
    let global_mesh_sdf = layers.iter().find_map(|l| l.mesh_sdf.as_deref());
    let global_order_field = layers.first().map(|l| l.order_field.as_ref());
    let all_paths = route_travel_moves(
        all_paths,
        global_mesh_sdf,
        global_order_field,
        slope_profile,
        config,
        z_travel_penalty,
    );
    let mut all_paths = insert_z_hops(all_paths, config);

    // Ensure no planned points dip below the build bed floor (Z = 0.0)
    for path in &mut all_paths {
        for pt in &mut path.points {
            let p_z = pt.dot(crate::slicing::BUILD_DIRECTION);
            if p_z < 0.0 {
                *pt += crate::slicing::BUILD_DIRECTION * (-p_z);
            }
        }
    }

    if let Ok(mut on_progress) = on_progress.lock() {
        on_progress(1.0);
    }

    Ok(all_paths)
}

/// Support-aware emission deferral: reorders the flattened per-layer path
/// list so that a path planned at layer order `O` but resting mostly on
/// mesh-solid whose order-field value is *later* than `O` (e.g. infill
/// over an Eikonal tunnel interior that the fast-march reaches from the
/// far side) is emitted only *after* the layer group containing its
/// supporting material, instead of being extruded into free air.
///
/// Detection mirrors `verification::floating_loops`' order-aware probe:
/// each extruding segment midpoint is stepped one layer height against
/// the local order-field gradient; the landing point is classified as
/// bed / already-printed solid (`probe order <= own order - eps`) /
/// future solid (`probe order > own order + eps`) / open air. A path is
/// deferred only when (a) at least [`DEFER_MIN_UNSUPPORTED_FRACTION`] of
/// its extruded length is not on bed or earlier-order solid AND (b) at
/// least [`DEFER_MIN_FUTURE_FRACTION`] of that unsupported length lands
/// on *future* solid — genuine open-air overhangs (nothing beneath in
/// the model at all) are left alone for the overhang/bridging passes.
///
/// The deferral target is the maximum supporting-solid order among the
/// future-solid probes, capped at [`DEFER_MAX_ORDER_SPAN`] layer heights
/// past the path's own layer (deferring further risks nozzle collisions
/// with much-taller surrounding geometry; such paths are left in place).
/// The reorder is a stable sort on emission order, so non-deferred
/// paths keep their exact original sequence (including per-layer travel
/// optimization), and a deferred path lands just after the layer group
/// it now rests on. Segment `order` stamps are left untouched — they
/// describe the geometry's layer, not emission time.
fn defer_unsupported_paths(
    per_layer: Vec<Vec<Path>>,
    layers: &[Layer],
    config: &SlicerConfig,
) -> Vec<Path> {
    const DEFER_MIN_UNSUPPORTED_FRACTION: f64 = 0.7;
    const DEFER_MIN_FUTURE_FRACTION: f64 = 0.5;
    const DEFER_MAX_ORDER_SPAN: f64 = 40.0;

    let layer_height = config.layer_height.abs().max(f64::EPSILON);
    let order_epsilon = 0.5 * layer_height;
    let sdf_tolerance = 0.5 * config.nozzle_diameter;
    // Sorted layer orders, for snapping a raw field-sample requirement up
    // to the discrete layer group that actually prints that material.
    let mut layer_orders: Vec<f64> = layers.iter().map(|l| l.order).collect();
    layer_orders.sort_by(f64::total_cmp);
    let bed_z = per_layer
        .iter()
        .flatten()
        .flat_map(|p| p.points.iter())
        .map(|p| p.z)
        .fold(f64::INFINITY, f64::min);

    // (emission_order, original_index) keys; stable sort preserves the
    // existing sequence wherever emission orders tie.
    let mut keyed: Vec<(f64, usize, Path)> = Vec::new();
    let mut flat_index = 0usize;
    for (layer, paths) in layers.iter().zip(per_layer) {
        let field = layer.order_field.as_ref();
        let sdf = layer.mesh_sdf.as_deref();
        for path in paths {
            let n = path.points.len();
            let mut total = 0.0;
            let mut unsupported = 0.0;
            let mut future = 0.0;
            let mut required_order = f64::NEG_INFINITY;
            for (i, segment) in path.segments.iter().enumerate() {
                if segment.extrusion_length <= 0.0 || n == 0 {
                    continue;
                }
                let a = path.points[i];
                let b = path.points[(i + 1) % n];
                let length = (b - a).length();
                if length <= f64::EPSILON {
                    continue;
                }
                total += length;
                let midpoint = (a + b) * 0.5;
                let (gradient_dir, gradient_len) =
                    match crate::order_field::numeric_gradient(field, midpoint)
                        .filter(|g| g.length_squared() > 1e-12 && g.is_finite())
                    {
                        Some(g) => (g / g.length(), g.length()),
                        None => (crate::slicing::BUILD_DIRECTION, 1.0),
                    };
                let step = (layer_height / gradient_len).clamp(layer_height, 4.0 * layer_height);
                let probe = midpoint - step * gradient_dir;
                if probe.z <= bed_z + 0.25 * layer_height {
                    continue; // resting on the bed
                }
                let inside_solid = sdf.is_some_and(|sdf| sdf.sample(probe).value <= sdf_tolerance);
                if inside_solid {
                    let probe_order = field.order(probe);
                    if probe_order.is_finite() {
                        if probe_order <= segment.order - order_epsilon {
                            continue; // supported by already-printed solid
                        }
                        // Solid beneath, but printed at the same order or
                        // later (same-band or genuinely future material) —
                        // deferrable either way: emitting after the layer
                        // group that owns `probe_order` makes it printed.
                        future += length;
                        required_order = required_order.max(probe_order);
                    }
                }
                unsupported += length;
            }

            let mut emission_order = layer.order;
            if total > f64::EPSILON
                && unsupported / total >= DEFER_MIN_UNSUPPORTED_FRACTION
                && future / unsupported.max(f64::EPSILON) >= DEFER_MIN_FUTURE_FRACTION
                && required_order.is_finite()
                && required_order - layer.order <= DEFER_MAX_ORDER_SPAN * layer_height
            {
                // Snap the raw field sample up to the discrete layer order
                // whose group prints the supporting material, then nudge
                // past it so ties resolve to "after".
                let group_order = layer_orders
                    .iter()
                    .copied()
                    .find(|&o| o >= required_order - 1e-9)
                    .unwrap_or(required_order);
                emission_order = group_order.max(layer.order) + 1e-9;
            }
            keyed.push((emission_order, flat_index, path));
            flat_index += 1;
        }
    }

    keyed.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    keyed.into_iter().map(|(_, _, path)| path).collect()
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
                line_width: 0.4,
                is_scarf: false,
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
                    ..Default::default()
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
                    ..Default::default()
                }],
                infill_boundary: Vec::new(),
                solid_fill_boundary: Vec::new(),
                mesh_sdf: None,
                order_field: Arc::new(HeightOrderField::new(BUILD_DIRECTION)),
            },
        ];

        let config = SlicerConfig {
            scarf_joint_enabled: false,
            ..SlicerConfig::default()
        };
        let paths = plan(&layers, &objects, &[], &config).unwrap();

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
        // Layers built without a `mesh_sdf` fall back to fully-supported
        // flow (see `support_fractions_at`), so the extrusion pass stamps
        // `support_fraction = 1.0` on every extruding segment.
        assert!(wall_paths[0]
            .segments
            .iter()
            .all(|segment| segment.support_fraction == 1.0));
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
            .all(|segment| segment.support_fraction == 1.0));
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
                ..Default::default()
            }],
            infill_boundary: Vec::new(),
            solid_fill_boundary: Vec::new(),
            mesh_sdf: None,
            order_field: Arc::new(HeightOrderField::new(BUILD_DIRECTION)),
        }];

        let config = SlicerConfig {
            scarf_joint_enabled: false,
            ..SlicerConfig::default()
        };
        let paths = plan(&layers, &objects, &[], &config).unwrap();

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
        assert_eq!(speed_for_kind(MoveKind::WallOuter, &config), 1800.0); // 60% of print_speed
        assert_eq!(speed_for_kind(MoveKind::WallInner, &config), 3000.0);
        assert_eq!(speed_for_kind(MoveKind::Infill, &config), 3000.0);
        assert_eq!(speed_for_kind(MoveKind::Bridge, &config), 1500.0); // 50% of print_speed
        assert_eq!(speed_for_kind(MoveKind::Overhang, &config), 1500.0);

        let explicit_config = SlicerConfig {
            travel_speed: 9000.0,
            print_speed: 3000.0,
            outer_wall_speed: Some(2400.0),
            bridge_speed: Some(1200.0),
            ..SlicerConfig::default()
        };
        assert_eq!(
            speed_for_kind(MoveKind::WallOuter, &explicit_config),
            2400.0
        );
        assert_eq!(speed_for_kind(MoveKind::Bridge, &explicit_config), 1200.0);
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
                ..Default::default()
            }],
            infill_boundary: Vec::new(),
            solid_fill_boundary: Vec::new(),
            mesh_sdf: None,
            order_field: Arc::new(HeightOrderField::new(BUILD_DIRECTION)),
        }];
        let config = SlicerConfig {
            print_speed: 1234.0,
            outer_wall_speed: Some(1234.0),
            first_layer_print_speed: Some(1234.0),
            ..SlicerConfig::default()
        };

        let paths = plan(&layers, &objects, &[], &config).unwrap();

        assert!(paths
            .iter()
            .flat_map(|p| p.segments.iter())
            .all(|segment| segment.speed == 1234.0));
    }

    #[test]
    fn plan_applies_first_layer_speed_and_extrusion_multiplier_to_first_layer_only() {
        let objects = vec![Object::new(ObjectId(0), Mesh::default(), ToolId(0))];
        let layer0 = Layer {
            index: 0,
            object: ObjectId(0),
            order: 0.25,
            loops: vec![WallLoop {
                wall_index: 0,
                points: vec![
                    DVec3::new(0.0, 0.0, 0.25),
                    DVec3::new(10.0, 0.0, 0.25),
                    DVec3::new(10.0, 10.0, 0.25),
                ],
                ..Default::default()
            }],
            infill_boundary: Vec::new(),
            solid_fill_boundary: Vec::new(),
            mesh_sdf: None,
            order_field: Arc::new(HeightOrderField::new(BUILD_DIRECTION)),
        };
        let layer1 = Layer {
            index: 1,
            object: ObjectId(0),
            order: 0.50,
            loops: vec![WallLoop {
                wall_index: 0,
                points: vec![
                    DVec3::new(0.0, 0.0, 0.50),
                    DVec3::new(10.0, 0.0, 0.50),
                    DVec3::new(10.0, 10.0, 0.50),
                ],
                ..Default::default()
            }],
            infill_boundary: Vec::new(),
            solid_fill_boundary: Vec::new(),
            mesh_sdf: None,
            order_field: Arc::new(HeightOrderField::new(BUILD_DIRECTION)),
        };
        let config = SlicerConfig {
            layer_height: 0.20,
            first_layer_height: Some(0.25),
            print_speed: 3000.0,
            outer_wall_speed: Some(3000.0),
            first_layer_print_speed: Some(1200.0),
            first_layer_extrusion_multiplier: Some(1.2),
            ..SlicerConfig::default()
        };

        let paths = plan(&[layer0, layer1], &objects, &[], &config).unwrap();
        let p0 = &paths[0];
        let p1 = &paths[1];

        // Layer 0 segments must use first_layer_print_speed (1200.0)
        assert!(p0.segments.iter().all(|s| s.speed == 1200.0));
        // Layer 1 segments must use standard print_speed (3000.0)
        assert!(p1.segments.iter().all(|s| s.speed == 3000.0));

        // Layer 0 has larger bead area (0.25 vs 0.20) and 1.2x multiplier,
        // so its extrusion length must be strictly larger for equal distance segments
        let e0: f64 = p0.segments.iter().map(|s| s.extrusion_length).sum();
        let e1: f64 = p1.segments.iter().map(|s| s.extrusion_length).sum();
        assert!(
            e0 > e1 * 1.3,
            "layer 0 extrusion ({e0}) should be significantly higher than layer 1 ({e1})"
        );
    }

    #[test]
    fn plan_applies_pre_retract_taper_when_configured() {
        let objects = vec![Object::new(ObjectId(0), Mesh::default(), ToolId(0))];
        let layer = Layer {
            index: 0,
            object: ObjectId(0),
            order: 0.20,
            loops: vec![WallLoop {
                wall_index: 0,
                points: vec![DVec3::new(0.0, 0.0, 0.20), DVec3::new(20.0, 0.0, 0.20)],
                ..Default::default()
            }],
            infill_boundary: Vec::new(),
            solid_fill_boundary: Vec::new(),
            mesh_sdf: None,
            order_field: Arc::new(HeightOrderField::new(BUILD_DIRECTION)),
        };
        let config_taper = SlicerConfig {
            pre_retract_taper_distance: Some(3.0),
            path_simplify_enabled: false,
            ..SlicerConfig::default()
        };

        let paths = plan(&[layer], &objects, &[], &config_taper).unwrap();
        let path = &paths[0];

        // The closing 20mm segment is split into a 17mm lead-in and 3mm tapered tail (3 segments total)
        assert_eq!(path.segments.len(), 3);
        // The tail segment's extrusion rate is tapered (0.6x average)
        assert!((path.segments[2].extrusion_rate - 0.6).abs() < 1e-3);
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
            sparse_infill_pattern: Some(infill::InfillPatternKind::Monotonic),
            solid_infill_pattern: Some(infill::InfillPatternKind::Monotonic),
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
            sparse_infill_pattern: Some(infill::InfillPatternKind::Monotonic),
            solid_infill_pattern: Some(infill::InfillPatternKind::Monotonic),
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
                    unsupported: vec![false, false],
                    top_surface: Vec::new(),
                    line_widths: Vec::new(),
                    arc_fraction: vec![0.0, 0.5],
                },
                WallLoop {
                    wall_index: 1,
                    points: vec![DVec3::new(2.0, 0.0, 0.0), DVec3::new(3.0, 0.0, 0.0)],
                    unsupported: vec![false, false],
                    top_surface: Vec::new(),
                    line_widths: Vec::new(),
                    arc_fraction: vec![0.0, 0.5],
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
                    unsupported: vec![false, false],
                    top_surface: Vec::new(),
                    line_widths: Vec::new(),
                    arc_fraction: vec![0.0, 0.5],
                },
                WallLoop {
                    wall_index: 1,
                    points: vec![DVec3::new(2.0, 0.0, 0.0), DVec3::new(3.0, 0.0, 0.0)],
                    unsupported: vec![false, false],
                    top_surface: Vec::new(),
                    line_widths: Vec::new(),
                    arc_fraction: vec![0.0, 0.5],
                },
                WallLoop {
                    wall_index: 2,
                    points: vec![DVec3::new(4.0, 0.0, 0.0), DVec3::new(5.0, 0.0, 0.0)],
                    unsupported: vec![false, false],
                    top_surface: Vec::new(),
                    line_widths: Vec::new(),
                    arc_fraction: vec![0.0, 0.5],
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
    fn plan_classifies_segments_landing_on_unsupported_points_as_overhang() {
        let objects = vec![Object::new(ObjectId(0), Mesh::default(), ToolId(0))];
        let layers = vec![Layer {
            index: 0,
            object: ObjectId(0),
            order: 0.0,
            loops: vec![
                // wall_index 0: points[2] is a stitched/unsupported point.
                // Segment 1 (points[1] -> points[2]) lands on it, so it
                // should be `Overhang`; every other segment (including the
                // wrap-around segment 3: points[3] -> points[0], whose
                // destination is supported) should remain `WallOuter`.
                WallLoop {
                    wall_index: 0,
                    points: vec![
                        DVec3::ZERO,
                        DVec3::new(1.0, 0.0, 0.0),
                        DVec3::new(1.0, 1.0, 0.0),
                        DVec3::new(0.0, 1.0, 0.0),
                    ],
                    unsupported: vec![false, false, true, false],
                    top_surface: Vec::new(),
                    line_widths: Vec::new(),
                    arc_fraction: vec![0.0, 0.25, 0.5, 0.75],
                },
                // wall_index 1: no unsupported points -- must be unaffected
                // (no regression to plain `WallInner` classification).
                WallLoop {
                    wall_index: 1,
                    points: vec![DVec3::new(4.0, 0.0, 0.0), DVec3::new(5.0, 0.0, 0.0)],
                    unsupported: vec![false, false],
                    top_surface: Vec::new(),
                    line_widths: Vec::new(),
                    arc_fraction: vec![0.0, 0.5],
                },
            ],
            infill_boundary: Vec::new(),
            solid_fill_boundary: Vec::new(),
            mesh_sdf: None,
            order_field: Arc::new(HeightOrderField::new(BUILD_DIRECTION)),
        }];

        let config = SlicerConfig {
            scarf_joint_enabled: false,
            ..SlicerConfig::default()
        };
        let paths = plan(&layers, &objects, &[], &config).unwrap();

        let wall_paths: Vec<_> = paths
            .iter()
            .filter(|p| {
                p.segments.iter().all(|segment| {
                    matches!(
                        segment.kind,
                        MoveKind::WallOuter | MoveKind::WallInner | MoveKind::Overhang
                    )
                })
            })
            .collect();
        assert_eq!(wall_paths.len(), 2);

        let outer = &wall_paths[0];
        assert_eq!(outer.segments.len(), 4);
        assert_eq!(outer.segments[0].kind, MoveKind::WallOuter);
        assert_eq!(outer.segments[1].kind, MoveKind::Overhang);
        assert_eq!(outer.segments[2].kind, MoveKind::WallOuter);
        assert_eq!(outer.segments[3].kind, MoveKind::WallOuter);

        let inner = &wall_paths[1];
        assert!(inner
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
            line_width: 0.4,
            is_scarf: false,
        };
        let travel_segment = Segment {
            kind: MoveKind::Travel,
            speed: 150.0,
            extrusion_rate: 1.0,
            support_fraction: 0.0,
            order: 0.0,
            extrusion_length: 0.0,
            line_width: 0.0,
            is_scarf: false,
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
            line_width: 0.4,
            is_scarf: false,
        };
        let travel_segment = Segment {
            kind: MoveKind::Travel,
            speed: 150.0,
            extrusion_rate: 1.0,
            support_fraction: 0.0,
            order: 0.0,
            extrusion_length: 0.0,
            line_width: 0.0,
            is_scarf: false,
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
            line_width: 0.0,
            is_scarf: false,
        };
        let infill_segment = Segment {
            kind: MoveKind::Infill,
            speed: 60.0,
            extrusion_rate: 1.0,
            support_fraction: 0.0,
            order: 0.0,
            extrusion_length: 1.0,
            line_width: 0.4,
            is_scarf: false,
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
            line_width: 0.4,
            is_scarf: false,
        };
        let travel_segment = Segment {
            kind: MoveKind::Travel,
            speed: 150.0,
            extrusion_rate: 1.0,
            support_fraction: 0.0,
            order: 0.0,
            extrusion_length: 0.0,
            line_width: 0.0,
            is_scarf: false,
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
            line_width: 0.4,
            is_scarf: false,
        };
        let travel_segment = Segment {
            kind: MoveKind::Travel,
            speed: 150.0,
            extrusion_rate: 1.0,
            support_fraction: 0.0,
            order: 0.0,
            extrusion_length: 0.0,
            line_width: 0.0,
            is_scarf: false,
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
    fn open_path(points: Vec<DVec3>, kind: MoveKind) -> Path {
        let segments = (0..points.len().saturating_sub(1))
            .map(|_| Segment {
                kind,
                speed: 60.0,
                extrusion_rate: 1.0,
                support_fraction: 0.0,
                order: 0.0,
                extrusion_length: 0.0,
                line_width: 0.4,
                is_scarf: false,
            })
            .collect();
        Path {
            points,
            segments,
            tool: ToolId::default(),
        }
    }

    fn closed_path(points: Vec<DVec3>, kind: MoveKind) -> Path {
        let segments = points
            .iter()
            .map(|_| Segment {
                kind,
                speed: 60.0,
                extrusion_rate: 1.0,
                support_fraction: 0.0,
                order: 0.0,
                extrusion_length: 0.0,
                line_width: 0.4,
                is_scarf: false,
            })
            .collect();
        Path {
            points,
            segments,
            tool: ToolId::default(),
        }
    }

    #[test]
    fn optimize_travel_order_is_a_no_op_when_disabled() {
        let paths = vec![
            open_path(
                vec![DVec3::new(0.0, 0.0, 0.0), DVec3::new(1.0, 0.0, 0.0)],
                MoveKind::Infill,
            ),
            open_path(
                vec![DVec3::new(100.0, 0.0, 0.0), DVec3::new(101.0, 0.0, 0.0)],
                MoveKind::Infill,
            ),
        ];
        let config = SlicerConfig {
            travel_order_optimization_enabled: false,
            ..SlicerConfig::default()
        };
        let original_starts: Vec<DVec3> = paths.iter().map(|p| p.points[0]).collect();
        let result = optimize_travel_order(paths, &config, config.z_travel_penalty);
        let result_starts: Vec<DVec3> = result.iter().map(|p| p.points[0]).collect();
        assert_eq!(result_starts, original_starts);
    }

    #[test]
    fn optimize_travel_order_reorders_paths_to_minimize_total_travel_distance() {
        // Three short open (infill-style) lines laid out so that
        // generation order (near, far, near) would force a long jump out
        // and back if left unreordered. Anchor stays first (its own
        // fixed start), but the remaining two must be visited in
        // nearest-first order: [0,1] segment, then [2,3] (right next to
        // it), leaving the far-away [100,101] segment for last.
        let anchor = open_path(
            vec![DVec3::new(0.0, 0.0, 0.0), DVec3::new(1.0, 0.0, 0.0)],
            MoveKind::Infill,
        );
        let far = open_path(
            vec![DVec3::new(100.0, 0.0, 0.0), DVec3::new(101.0, 0.0, 0.0)],
            MoveKind::Infill,
        );
        let near = open_path(
            vec![DVec3::new(2.0, 0.0, 0.0), DVec3::new(3.0, 0.0, 0.0)],
            MoveKind::Infill,
        );
        let config = SlicerConfig::default();
        let result =
            optimize_travel_order(vec![anchor, far, near], &config, config.z_travel_penalty);

        assert_eq!(result.len(), 3);
        assert_eq!(result[0].points[0], DVec3::new(0.0, 0.0, 0.0));
        assert_eq!(result[1].points[0], DVec3::new(2.0, 0.0, 0.0));
        assert_eq!(result[2].points[0], DVec3::new(100.0, 0.0, 0.0));
    }

    #[test]
    fn optimize_travel_order_reverses_an_open_path_when_its_far_end_is_closer() {
        // The remaining open path's *far* endpoint (10.0) is much closer
        // to the anchor's exit point (1.0) than its *near* endpoint
        // (9.0..10.0 span placed backwards) -- the optimizer should enter
        // it from that closer end, i.e. reverse it.
        let anchor = open_path(
            vec![DVec3::new(0.0, 0.0, 0.0), DVec3::new(1.0, 0.0, 0.0)],
            MoveKind::Infill,
        );
        let candidate = open_path(
            vec![DVec3::new(9.0, 0.0, 0.0), DVec3::new(1.2, 0.0, 0.0)],
            MoveKind::Infill,
        );
        let config = SlicerConfig::default();
        let result =
            optimize_travel_order(vec![anchor, candidate], &config, config.z_travel_penalty);

        assert_eq!(result.len(), 2);
        // Reversed: now starts at 1.2 (close to anchor's exit at 1.0) and
        // ends at 9.0.
        assert_eq!(result[1].points[0], DVec3::new(1.2, 0.0, 0.0));
        assert_eq!(result[1].points[1], DVec3::new(9.0, 0.0, 0.0));
    }

    #[test]
    fn optimize_travel_order_never_reverses_a_closed_wall_loop() {
        // Closed loops (segments.len() == points.len()) must keep their
        // own points[0] -- only their position in the overall order may
        // change, never their internal start point/direction.
        let anchor = open_path(
            vec![DVec3::new(0.0, 0.0, 0.0), DVec3::new(1.0, 0.0, 0.0)],
            MoveKind::Infill,
        );
        let wall = closed_path(
            vec![
                DVec3::new(9.0, 0.0, 0.0),
                DVec3::new(9.0, 1.0, 0.0),
                DVec3::new(1.1, 1.0, 0.0),
                DVec3::new(1.1, 0.0, 0.0),
            ],
            MoveKind::WallOuter,
        );
        let original_points = wall.points.clone();
        let config = SlicerConfig::default();
        let result = optimize_travel_order(vec![anchor, wall], &config, config.z_travel_penalty);

        assert_eq!(result.len(), 2);
        assert_eq!(result[1].points, original_points);
    }

    #[test]
    fn optimize_travel_order_penalizes_z_motion_over_xy_motion() {
        // Anchor ends at (1.0, 0.0, 0.0).
        // Candidate A is at (5.0, 0.0, 0.0) -> XY distance 4.0, delta Z 0.0
        // Candidate B is at (1.0, 0.0, 1.0) -> XY distance 0.0, delta Z 1.0
        // With z_penalty = 8.0, B has cost 8.0 > A's cost 4.0, so A is chosen first.
        let anchor = open_path(
            vec![DVec3::new(0.0, 0.0, 0.0), DVec3::new(1.0, 0.0, 0.0)],
            MoveKind::Infill,
        );
        let path_xy_far = open_path(
            vec![DVec3::new(5.0, 0.0, 0.0), DVec3::new(6.0, 0.0, 0.0)],
            MoveKind::Infill,
        );
        let path_z_near = open_path(
            vec![DVec3::new(1.0, 0.0, 1.0), DVec3::new(2.0, 0.0, 1.0)],
            MoveKind::Infill,
        );
        let config = SlicerConfig {
            z_travel_penalty: 8.0,
            ..SlicerConfig::default()
        };
        let result = optimize_travel_order(
            vec![anchor, path_z_near, path_xy_far],
            &config,
            config.z_travel_penalty,
        );
        assert_eq!(result.len(), 3);
        // path_xy_far (at z=0) selected before path_z_near (at z=1) due to z_penalty
        assert_eq!(result[1].points[0], DVec3::new(5.0, 0.0, 0.0));
        // path_z_near entered at closer reversed end (2.0, 0.0, 1.0)
        assert_eq!(result[2].points[0], DVec3::new(2.0, 0.0, 1.0));
    }

    #[test]
    fn reverse_open_path_preserves_segment_metadata_for_the_same_physical_edges() {
        let path = open_path(
            vec![
                DVec3::new(0.0, 0.0, 0.0),
                DVec3::new(1.0, 0.0, 0.0),
                DVec3::new(3.0, 0.0, 0.0),
            ],
            MoveKind::Infill,
        );
        let reversed = reverse_open_path(path);
        assert_eq!(
            reversed.points,
            vec![
                DVec3::new(3.0, 0.0, 0.0),
                DVec3::new(1.0, 0.0, 0.0),
                DVec3::new(0.0, 0.0, 0.0),
            ]
        );
        assert_eq!(reversed.segments.len(), 2);
    }

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
                ..Default::default()
            }],
            infill_boundary: Vec::new(),
            solid_fill_boundary: Vec::new(),
            mesh_sdf: None,
            order_field: Arc::new(HeightOrderField::new(BUILD_DIRECTION)),
        }];

        let config = SlicerConfig {
            path_simplify_enabled: true,
            path_simplify_tolerance: 0.05,
            scarf_joint_enabled: false,
            curvature_compensation_enabled: Some(false),
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
        let line_width = config.first_layer_line_width();
        let bead_area = extrusion::blended_bead_cross_section_area(
            line_width,
            config.first_layer_height(),
            config.nozzle_diameter,
            1.0,
            1.0,
        );
        let point_count = wall_path.points.len();
        for (i, segment) in wall_path.segments.iter().enumerate() {
            let distance = wall_path.points[i].distance(wall_path.points[(i + 1) % point_count]);
            let expected = extrusion::segment_extrusion_length(distance, bead_area, filament_area)
                * segment.extrusion_rate
                * config.first_layer_extrusion_multiplier();
            assert!(
                (segment.extrusion_length - expected).abs() < 1e-9,
                "segment {i}: extrusion_length {} did not match post-simplify distance-derived value {expected}",
                segment.extrusion_length
            );
        }
    }

    /// Unit cube spanning [0,1]^3, as a ready-to-use `MeshSdf` (same fixture
    /// pattern as `slicing::tests::cube_mesh`).
    fn cube_sdf_fixture() -> MeshSdf {
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
            [0, 3, 2], // -Z
            [4, 5, 6],
            [4, 6, 7], // +Z
            [0, 1, 5],
            [0, 5, 4], // -Y
            [3, 7, 6],
            [3, 6, 2], // +Y
            [0, 4, 7],
            [0, 7, 3], // -X
            [1, 2, 6],
            [1, 6, 5], // +X
        ];
        MeshSdf::new(vertices, faces)
    }

    #[test]
    fn travel_chord_is_blocked_detects_a_chord_through_solid_material() {
        let sdf = cube_sdf_fixture();
        let clearance = 0.05;

        // Straight through the cube along x at y = z = 0.5.
        assert!(travel_chord_is_blocked(
            &sdf,
            None,
            0.0,
            DVec3::new(-1.0, 0.5, 0.5),
            DVec3::new(2.0, 0.5, 0.5),
            clearance,
        ));

        // Well clear of the cube.
        assert!(!travel_chord_is_blocked(
            &sdf,
            None,
            0.0,
            DVec3::new(-1.0, 5.0, 5.0),
            DVec3::new(2.0, 5.0, 5.0),
            clearance,
        ));
    }

    #[test]
    fn travel_chord_is_blocked_respects_order_field_for_future_geometry() {
        let sdf = cube_sdf_fixture();
        let field = HeightOrderField::new(DVec3::Z);
        let clearance = 0.05;

        // Path crossing through the cube at z = 0.8:
        // When current_order = 0.2 (layer height 0.2), the cube material at z = 0.8
        // is in the future (unprinted), so it should NOT be considered blocked.
        assert!(!travel_chord_is_blocked(
            &sdf,
            Some(&field),
            0.2,
            DVec3::new(-1.0, 0.5, 0.8),
            DVec3::new(2.0, 0.5, 0.8),
            clearance,
        ));

        // When current_order = 0.9 (layer height 0.9), the cube material at z = 0.8
        // has already been printed, so it MUST be considered blocked.
        assert!(travel_chord_is_blocked(
            &sdf,
            Some(&field),
            0.9,
            DVec3::new(-1.0, 0.5, 0.8),
            DVec3::new(2.0, 0.5, 0.8),
            clearance,
        ));
    }

    #[test]
    fn route_travel_moves_replaces_a_blocked_travel_with_a_routed_path_around_solid_material() {
        let sdf = Arc::new(cube_sdf_fixture());

        let a = open_path(
            vec![DVec3::new(-2.0, 0.5, 0.5), DVec3::new(-0.5, 0.5, 0.5)],
            MoveKind::Infill,
        );
        let b = open_path(
            vec![DVec3::new(1.5, 0.5, 0.5), DVec3::new(3.0, 0.5, 0.5)],
            MoveKind::Infill,
        );
        let config = SlicerConfig::default();
        let slope_profile = manifold_fidget::slope_profile::SlopeProfile::new(Vec::new());

        let routed = route_travel_moves(
            vec![a, b],
            Some(&sdf),
            None,
            &slope_profile,
            &config,
            config.z_travel_penalty,
        );

        assert_eq!(
            routed.len(),
            3,
            "expected an inserted routing path between the two blocked paths"
        );
        let detour = &routed[1];
        assert!(detour
            .segments
            .iter()
            .all(|segment| segment.kind == MoveKind::Travel));

        // Every step along the routed detour must stay clear of the solid
        // cube (within the same clearance used to plan it).
        let clearance = 2.0 * config.wall_line_width;
        for pair in detour.points.windows(2) {
            assert!(!travel_chord_is_blocked(
                &sdf, None, 0.0, pair[0], pair[1], clearance
            ));
        }
    }

    #[test]
    fn travel_move_departs_and_arrives_maintaining_clearance() {
        let sdf = Arc::new(cube_sdf_fixture());

        let a = open_path(
            vec![DVec3::new(-2.0, 0.5, 0.5), DVec3::new(0.0, 0.5, 0.5)],
            MoveKind::Infill,
        );
        let b = open_path(
            vec![DVec3::new(1.0, 0.5, 0.5), DVec3::new(3.0, 0.5, 0.5)],
            MoveKind::Infill,
        );
        let config = SlicerConfig::default();
        let slope_profile = manifold_fidget::slope_profile::SlopeProfile::new(Vec::new());

        let routed = route_travel_moves(
            vec![a, b],
            Some(&sdf),
            None,
            &slope_profile,
            &config,
            config.z_travel_penalty,
        );
        assert_eq!(routed.len(), 3);
        let detour = &routed[1];

        assert!(detour.points.len() >= 4);
        let p_dep = detour.points[1];
        assert!(
            sdf.sample(p_dep).value >= 0.5,
            "departure waypoint must be in open air"
        );
        let p_arr = detour.points[detour.points.len() - 2];
        assert!(
            sdf.sample(p_arr).value >= 0.5,
            "arrival waypoint must be in open air"
        );
    }

    #[test]
    fn route_travel_moves_with_steep_slope_profile_still_routes_around_solid() {
        let sdf = Arc::new(cube_sdf_fixture());

        let a = open_path(
            vec![DVec3::new(-2.0, 0.5, 0.5), DVec3::new(-0.5, 0.5, 0.5)],
            MoveKind::Infill,
        );
        let b = open_path(
            vec![DVec3::new(1.5, 0.5, 0.5), DVec3::new(3.0, 0.5, 0.5)],
            MoveKind::Infill,
        );
        let config = SlicerConfig::default();
        // Restrictive slope profile simulating nozzle clearance envelope
        let slope_profile =
            manifold_fidget::slope_profile::SlopeProfile::new(vec![(3.8, 1.6), (32.0, 11.0)]);

        let routed = route_travel_moves(
            vec![a, b],
            Some(&sdf),
            None,
            &slope_profile,
            &config,
            config.z_travel_penalty,
        );
        assert_eq!(
            routed.len(),
            3,
            "expected route_travel_moves to find a detour even with slope profile"
        );
        let detour = &routed[1];
        let clearance = 2.0 * config.wall_line_width;
        for pair in detour.points.windows(2) {
            assert!(!travel_chord_is_blocked(
                &sdf, None, 0.0, pair[0], pair[1], clearance
            ));
        }
    }

    #[test]
    fn route_travel_moves_clamps_all_waypoints_at_or_above_bed_floor() {
        let sdf = Arc::new(cube_sdf_fixture());

        // Chords along bottom bed (z = 0.0) where outward normals point downward
        let a = open_path(
            vec![DVec3::new(-2.0, 0.5, 0.0), DVec3::new(-0.5, 0.5, 0.0)],
            MoveKind::Infill,
        );
        let b = open_path(
            vec![DVec3::new(1.5, 0.5, 0.0), DVec3::new(3.0, 0.5, 0.0)],
            MoveKind::Infill,
        );
        let config = SlicerConfig::default();
        let slope_profile = manifold_fidget::slope_profile::SlopeProfile::new(Vec::new());

        let routed = route_travel_moves(
            vec![a, b],
            Some(&sdf),
            None,
            &slope_profile,
            &config,
            config.z_travel_penalty,
        );
        assert_eq!(routed.len(), 3);
        let detour = &routed[1];
        for pt in &detour.points {
            assert!(
                pt.z >= 0.0,
                "travel waypoint must never dip below bed floor z=0: {pt:?}"
            );
        }
        for pt in &detour.points[1..detour.points.len() - 1] {
            assert!(
                pt.z >= 0.5 * config.first_layer_height(),
                "intermediate travel transit waypoint must maintain clearance above bed floor: {pt:?}"
            );
        }
    }

    #[test]
    fn route_travel_moves_is_a_no_op_when_disabled() {
        let sdf = Arc::new(cube_sdf_fixture());
        let a = open_path(
            vec![DVec3::new(-2.0, 0.5, 0.5), DVec3::new(-0.5, 0.5, 0.5)],
            MoveKind::Infill,
        );
        let b = open_path(
            vec![DVec3::new(1.5, 0.5, 0.5), DVec3::new(3.0, 0.5, 0.5)],
            MoveKind::Infill,
        );
        let config = SlicerConfig {
            travel_collision_avoidance_enabled: false,
            ..SlicerConfig::default()
        };
        let slope_profile = manifold_fidget::slope_profile::SlopeProfile::new(Vec::new());

        let routed = route_travel_moves(
            vec![a, b],
            Some(&sdf),
            None,
            &slope_profile,
            &config,
            config.z_travel_penalty,
        );
        assert_eq!(routed.len(), 2, "disabled pass must leave paths untouched");
    }

    #[test]
    fn compensate_flat_nozzle_elevates_on_concave_v_groove() {
        // Conical field creates a concave cone where the apex is below and walls slope outward/upward:
        let field =
            manifold_fidget::order::ConicalOrderField::new(DVec3::ZERO, BUILD_DIRECTION, 0.5);
        let points = vec![
            DVec3::new(1.0, 0.0, 1.0),
            DVec3::new(0.0, 1.0, 1.0),
            DVec3::new(-1.0, 0.0, 1.0),
            DVec3::new(0.0, -1.0, 1.0),
        ];
        let flat_radius = 0.5;
        let layer_height = 0.2;
        let min_extrusion_z = 0.1;
        let compensated = compensate_wall_loop_points(
            &points,
            &field,
            flat_radius,
            layer_height,
            1.0,
            min_extrusion_z,
        );
        for (orig, comp) in points.iter().zip(compensated.iter()) {
            assert!(
                comp.z >= orig.z,
                "compensated point must be elevated on concave geometry: comp.z={}, orig.z={}",
                comp.z,
                orig.z
            );
        }
    }

    #[test]
    fn compensate_flat_nozzle_volumetric_mode_leaves_paths_untouched() {
        let field: Arc<dyn manifold_fidget::order::OrderField> =
            Arc::new(HeightOrderField::new(BUILD_DIRECTION));
        let obj_id = ObjectId(0);
        let layer = Layer {
            object: obj_id,
            index: 1,
            order: 1.0,
            loops: Vec::new(),
            infill_boundary: Vec::new(),
            solid_fill_boundary: Vec::new(),
            order_field: Arc::clone(&field),
            mesh_sdf: None,
        };
        let config = SlicerConfig {
            slope_compensation_mode: Some(crate::SlopeCompensationMode::VolumetricModulation),
            ..SlicerConfig::default()
        };
        let tools = vec![Tool::new(ToolId(0), 0.4)];
        let orig_points = vec![
            DVec3::new(0.0, 0.0, 1.0),
            DVec3::new(10.0, 0.0, 1.0),
            DVec3::new(10.0, 10.0, 1.0),
        ];
        let paths = vec![Path {
            tool: ToolId(0),
            points: orig_points.clone(),
            segments: vec![
                Segment {
                    kind: MoveKind::WallOuter,
                    ..Segment::default()
                };
                3
            ],
        }];

        let result = compensate_flat_nozzle(paths, &layer, &config, &tools);
        assert_eq!(result[0].points, orig_points);
    }

    #[test]
    fn compensate_flat_nozzle_clamps_at_bed_floor() {
        let field = HeightOrderField::new(BUILD_DIRECTION);
        let points = vec![
            DVec3::new(0.0, 0.0, 0.2),
            DVec3::new(1.0, 0.0, 0.2),
            DVec3::new(1.0, 1.0, 0.2),
            DVec3::new(0.0, 1.0, 0.2),
        ];
        let min_extrusion_z = 0.1;
        let compensated =
            compensate_wall_loop_points(&points, &field, 0.4, 0.2, 0.2, min_extrusion_z);
        for pt in compensated {
            assert!(
                pt.z >= min_extrusion_z,
                "compensated point must not dip below min extrusion z: {pt:?}"
            );
        }
    }

    #[test]
    fn compensate_flat_nozzle_preserves_xy_centerline_without_distortion() {
        let field =
            manifold_fidget::order::ConicalOrderField::new(DVec3::ZERO, BUILD_DIRECTION, 0.5);
        let points = vec![
            DVec3::new(2.0, 0.0, 1.0),
            DVec3::new(0.0, 2.0, 1.0),
            DVec3::new(-2.0, 0.0, 1.0),
            DVec3::new(0.0, -2.0, 1.0),
        ];
        let compensated = compensate_wall_loop_points(&points, &field, 0.4, 0.2, 1.0, 0.0);
        for (orig, comp) in points.iter().zip(compensated.iter()) {
            assert_eq!(orig.x, comp.x, "X coordinate must remain on true contour");
            assert_eq!(orig.y, comp.y, "Y coordinate must remain on true contour");
            assert!(
                comp.z >= orig.z,
                "Z coordinate must be elevated for clearance on slope"
            );
        }
    }

    #[test]
    fn plan_applies_slope_cosine_volumetric_compensation_to_sloped_moves() {
        let obj_id = ObjectId(1);
        let object = Object::new(obj_id, Mesh::default(), ToolId(0));
        let field: Arc<dyn manifold_fidget::order::OrderField> =
            Arc::new(HeightOrderField::new(BUILD_DIRECTION));

        let p0 = DVec3::new(0.0, 0.0, 1.0);
        let p1 = DVec3::new(10.0, 0.0, 11.0);
        let p2 = DVec3::new(10.0, 10.0, 11.0);
        let p3 = DVec3::new(0.0, 10.0, 1.0);

        let layer0 = Layer {
            object: obj_id,
            index: 0,
            order: 0.2,
            loops: vec![WallLoop {
                points: vec![
                    DVec3::new(0.0, 0.0, 0.2),
                    DVec3::new(10.0, 0.0, 0.2),
                    DVec3::new(10.0, 10.0, 0.2),
                    DVec3::new(0.0, 10.0, 0.2),
                ],
                wall_index: 0,
                top_surface: vec![false; 4],
                line_widths: Vec::new(),
                arc_fraction: vec![0.0; 4],
                unsupported: vec![false; 4],
            }],
            infill_boundary: Vec::new(),
            solid_fill_boundary: Vec::new(),
            order_field: Arc::clone(&field),
            mesh_sdf: None,
        };

        let layer = Layer {
            object: obj_id,
            index: 5,
            order: 5.0,
            loops: vec![WallLoop {
                points: vec![p0, p1, p2, p3],
                wall_index: 0,
                top_surface: vec![false; 4],
                line_widths: Vec::new(),
                arc_fraction: vec![0.0; 4],
                unsupported: vec![false; 4],
            }],
            infill_boundary: Vec::new(),
            solid_fill_boundary: Vec::new(),
            order_field: Arc::clone(&field),
            mesh_sdf: None,
        };

        let config = SlicerConfig {
            scarf_joint_enabled: false,
            path_simplify_enabled: false,
            travel_order_optimization_enabled: false,
            travel_collision_avoidance_enabled: false,
            curvature_compensation_enabled: Some(false),
            ..SlicerConfig::default()
        };
        let tools = vec![Tool::new(ToolId(0), 0.4)];

        let planned = plan(&[layer0, layer], &[object], &tools, &config).unwrap();
        assert_eq!(planned.len(), 2);
        let path = &planned[1];
        assert_eq!(path.segments.len(), 4);

        let bead_area =
            extrusion::bead_cross_section_area(config.wall_line_width, config.layer_height);
        let fil_area = extrusion::filament_cross_section_area(config.filament_diameter);
        let expected_flat_e = extrusion::segment_extrusion_length(10.0, bead_area, fil_area);

        let expected_climbing_e = expected_flat_e;

        let actual_e = path.segments[0].extrusion_length;
        assert!(
            (actual_e - expected_climbing_e).abs() < 1e-4,
            "Actual extrusion length ({actual_e}) should match horizontal projection ({expected_climbing_e})"
        );
    }

    #[test]
    fn plan_clamps_segment_speed_directionally_against_axis_limits() {
        use crate::kinematics::{Axis, AxisLimits};
        use crate::machine::Machine;
        use crate::tool::Tool;

        let obj_id = ObjectId(1);
        let object = Object::new(obj_id, Mesh::default(), ToolId(0));
        let field: Arc<dyn manifold_fidget::order::OrderField> =
            Arc::new(HeightOrderField::new(BUILD_DIRECTION));

        let layer0 = Layer {
            object: obj_id,
            index: 0,
            order: 0.0,
            loops: vec![WallLoop {
                points: vec![
                    DVec3::new(0.0, 0.0, 0.0),
                    DVec3::new(10.0, 0.0, 0.0),
                    DVec3::new(10.0, 10.0, 0.0),
                    DVec3::new(0.0, 10.0, 0.0),
                ],
                wall_index: 0,
                top_surface: vec![false; 4],
                line_widths: Vec::new(),
                arc_fraction: vec![0.0; 4],
                unsupported: vec![false; 4],
            }],
            infill_boundary: Vec::new(),
            solid_fill_boundary: Vec::new(),
            order_field: Arc::clone(&field),
            mesh_sdf: None,
        };

        let layer = Layer {
            object: obj_id,
            index: 1,
            order: 1.0,
            loops: vec![WallLoop {
                points: vec![
                    DVec3::new(0.0, 0.0, 0.0),
                    DVec3::new(10.0, 0.0, 10.0),
                    DVec3::new(10.0, 10.0, 10.0),
                    DVec3::new(0.0, 10.0, 0.0),
                ],
                wall_index: 0,
                top_surface: vec![false; 4],
                line_widths: Vec::new(),
                arc_fraction: vec![0.0; 4],
                unsupported: vec![false; 4],
            }],
            infill_boundary: Vec::new(),
            solid_fill_boundary: Vec::new(),
            order_field: Arc::clone(&field),
            mesh_sdf: None,
        };

        // Standard outer wall speed: 6000 mm/min (100 mm/s)
        // Machine has Z max speed = 20 mm/s (1200 mm/min)
        let mut machine = Machine::default();
        machine.set_axis_limits(
            Axis::Z,
            AxisLimits {
                speed_limit: Some(20.0),
                ..AxisLimits::default()
            },
        );

        let config = SlicerConfig {
            scarf_joint_enabled: false,
            path_simplify_enabled: false,
            travel_order_optimization_enabled: false,
            travel_collision_avoidance_enabled: false,
            ..SlicerConfig::default()
        };

        let tools = vec![Tool::new(ToolId(0), 0.4)];
        let paths = plan_with_progress(
            &[layer0, layer],
            &[object],
            &tools,
            &config,
            Some(&machine),
            &manifold_fidget::slope_profile::SlopeProfile::new(Vec::new()),
            &mut |_| {},
        )
        .unwrap();
        assert_eq!(paths.len(), 2);
        let path = &paths[1];

        let mut sloped_count = 0;
        for (i, seg) in path.segments.iter().enumerate() {
            let start = path.points[i];
            let end = path.points[(i + 1) % path.points.len()];
            let diff = end - start;
            let d = diff.length();
            if d > 1e-4 {
                let dir = diff / d;
                if dir.z.abs() > 0.01 {
                    sloped_count += 1;
                    let expected_speed = (20.0 / dir.z.abs()) * 60.0;
                    assert!(
                        (seg.speed - expected_speed).abs() < 1.0,
                        "Sloped segment speed ({}) must match expected ({})",
                        seg.speed,
                        expected_speed
                    );
                }
            }
        }
        assert!(sloped_count > 0, "Expected at least one sloped segment");
    }
}
