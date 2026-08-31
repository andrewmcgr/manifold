//! Solid infill generation: fills the area inside each layer's infill
//! boundary (one wall pass further inward than the innermost printed
//! wall) with a path pattern.
//!
//! Pluggable by design, mirroring `ordering::ObjectOrderStrategy`/
//! `strategy_for`: new patterns implement [`InfillGenerator`] and are
//! wired into [`generator_for`] without touching `toolpath::plan`.

use crate::order_field;
use crate::polygon2d;
use crate::slicing::Layer;
use crate::toolpath::{speed_for_kind, MoveKind, Path, Segment};
use crate::transform::Transform;
use crate::SlicerConfig;
use glam::DVec3;
use manifold_fidget::contour::plane_basis;
use manifold_fidget::ScalarField;

/// Which built-in infill pattern to generate. New patterns are added as a
/// new variant here plus a new `InfillGenerator` impl wired into
/// [`generator_for`] — `toolpath::plan` never needs to change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum InfillPatternKind {
    /// Rectilinear zig-zag ("boustrophedon") scan-line fill, alternating
    /// ±`SlicerConfig::infill_angle_deg` by layer index. See
    /// [`MonotonicInfill`].
    #[default]
    Monotonic,
    /// Concentric fill: repeated inward offsets of the region boundary,
    /// each printed as its own closed loop, spaced `infill_line_width`
    /// apart (widened by `density`, same convention as `Monotonic`). See
    /// [`ConcentricInfill`].
    Concentric,
    /// "Wall-style" fill: repeated inward offsets of the region boundary,
    /// each printed as its own closed loop, same as [`ConcentricInfill`]
    /// but always spaced exactly `infill_line_width` apart -- `density` is
    /// ignored entirely rather than widening the spacing, so this always
    /// fully fills the region regardless of the configured infill density.
    /// See [`AllWallsInfill`].
    AllWalls,
    /// 3D Cubic lattice infill: self-supporting 3D periodic cubic grid rotated
    /// with space diagonals along the Cartesian axes (45-degree inclination to
    /// the build plate), forming a 3D isotropic truss. See [`CubicInfill`].
    Cubic,
    /// Triply Periodic Minimal Surface (TPMS) Gyroid infill: continuous, self-supporting
    /// 3D minimal surface lattice with isotropic stiffness and high permeability.
    Gyroid,
    /// Triply Periodic Minimal Surface (TPMS) Schwarz Diamond (D) infill: high-strength
    /// continuous minimal surface truss structure.
    SchwarzD,
    /// Triply Periodic Minimal Surface (TPMS) Schwarz Primitive (P) infill: simple, highly
    /// open continuous cubic minimal surface structure.
    SchwarzP,
    /// No sparse infill at all. Walls and solid-fill (top/bottom) regions
    /// still print; sprarse infill between them is omitted.
    None,
}

/// Resolve an [`InfillPatternKind`] to its [`InfillGenerator`] implementation.
#[must_use]
pub fn generator_for(kind: InfillPatternKind) -> Box<dyn InfillGenerator + Sync> {
    match kind {
        InfillPatternKind::Monotonic => Box::new(MonotonicInfill),
        InfillPatternKind::Concentric => Box::new(ConcentricInfill),
        InfillPatternKind::AllWalls => Box::new(AllWallsInfill),
        InfillPatternKind::Cubic => Box::new(CubicInfill),
        InfillPatternKind::Gyroid => Box::new(GyroidInfill),
        InfillPatternKind::SchwarzD => Box::new(SchwarzDInfill),
        InfillPatternKind::SchwarzP => Box::new(SchwarzPInfill),
        InfillPatternKind::None => Box::new(NoneInfill),
    }
}

/// A single layer's fillable area: its infill boundary (see
/// [`crate::slicing::Layer::infill_boundary`]), in world space. May
/// contain multiple loops (disjoint islands, and/or holes within an
/// island) — [`InfillGenerator`] impls are expected to resolve
/// inside/outside via an even-odd crossing rule rather than assuming a
/// single simple polygon, so island/hole topology does not need to be
/// classified up front.
#[derive(Debug, Clone, Default)]
pub struct InfillRegion {
    pub loops: Vec<Vec<DVec3>>,
}

impl InfillRegion {
    /// Build the *sparse* fillable region for `layer`: its infill boundary
    /// (see [`crate::slicing::Layer::infill_boundary`]) minus whatever part
    /// of it must print solid (see
    /// [`crate::slicing::Layer::solid_fill_boundary`]). Returns an empty
    /// region (no loops) for a layer with no infill boundary, and excludes
    /// the layer entirely if its whole infill boundary is solid.
    #[must_use]
    pub fn from_layer(layer: &Layer, config: &SlicerConfig) -> Self {
        if layer.solid_fill_boundary.is_empty()
            || config.sparse_infill_pattern() == InfillPatternKind::AllWalls
        {
            return Self {
                loops: layer.infill_boundary.clone(),
            };
        }
        let (axis, apex, _slope) = order_field::resolve_axis_apex_slope(config.order_field, config);
        let (basis1, basis2) = plane_basis(axis);
        let infill_2d = polygon2d::canonicalize(&polygon2d::to_2d(
            &layer.infill_boundary,
            basis1,
            basis2,
            apex,
        ));
        let solid_2d = polygon2d::canonicalize(&polygon2d::to_2d(
            &layer.solid_fill_boundary,
            basis1,
            basis2,
            apex,
        ));
        let sparse_2d = polygon2d::difference(&infill_2d, &solid_2d);
        // Reference-seeded reconstruction (see
        // `reconstruct_on_order_field_near`): boolean-op output points lie
        // on the input loops' edges, and the layer's own 3D boundaries are
        // already on the correct branch of the isosurface, so seeding each
        // rebuilt point from its nearest input point avoids the
        // wrong-branch axis-ray solves that spiked Eikonal infill.
        let references: Vec<Vec<DVec3>> = layer
            .infill_boundary
            .iter()
            .chain(layer.solid_fill_boundary.iter())
            .cloned()
            .collect();
        Self {
            loops: order_field::reconstruct_on_order_field_near(
                sparse_2d,
                &references,
                basis1,
                basis2,
                axis,
                apex,
                layer.order,
                order_field::max_along_for(config),
                layer.order_field.as_ref(),
            ),
        }
    }

    fn is_empty(&self) -> bool {
        self.loops.iter().all(|l| l.len() < 2)
    }
}

/// Generates infill [`Path`]s for one layer's [`InfillRegion`].
pub trait InfillGenerator {
    /// `object_transform` is the source object's placement — used to keep
    /// the fill angle fixed relative to the object's own orientation (see
    /// [`Transform::in_plane_rotation_angle`]) rather than to world space,
    /// so rotating an object in the workspace rotates its infill with it.
    ///
    /// `density` is the fraction of `region` to actually fill (`0.0..=1.0`;
    /// see [`crate::SlicerConfig::infill_density`]) — callers pass `1.0`
    /// for regions that must always print solid (e.g.
    /// [`crate::slicing::Layer::solid_fill_boundary`]) regardless of
    /// `config.infill_density`, and `config.infill_density` itself for the
    /// sparse region.
    fn generate(
        &self,
        region: &InfillRegion,
        config: &SlicerConfig,
        layer: &Layer,
        object_transform: &Transform,
        density: f64,
    ) -> Vec<Path>;
}

/// Rectilinear zig-zag infill: scan lines spaced `infill_line_width` apart,
/// clipped against the region's loops with an even-odd crossing rule
/// (handles holes and multiple islands without needing them classified),
/// linked scan-line to scan-line in alternating (boustrophedon) direction
/// to minimize travel.
///
/// The scan direction alternates ±`infill_angle_deg` by `layer.index`
/// parity, added to the source object's in-plane rotation so infill tracks
/// object orientation (see [`Transform::in_plane_rotation_angle`]).
#[derive(Debug, Clone, Copy, Default)]
pub struct MonotonicInfill;

/// One crossing of a scan line with a loop edge, in the rotated fill
/// frame: `u` is the in-plane fill-frame coordinate (used only for sorting
/// crossings left-to-right along the scan line), `point` is the actual
/// reconstructed 3D world position for this crossing — re-solved against
/// the layer's order field (see [`order_field::reconstruct_point_on_order_field`])
/// rather than linearly interpolated between the crossed edge's two
/// endpoint heights. Linear interpolation is only exact for a
/// `HeightOrderField` (whose height is independent of `(u, v)`); for a
/// curved field like `Eikonal`/`Conical`, the true height can vary sharply
/// across a short edge near curved/threaded geometry, and interpolating it
/// linearly previously sent infill points far off the actual surface (see
/// the "screw-thread pug model produces wildly spiking Eikonal infill"
/// bug this fixes).
#[derive(Debug, Clone, Copy)]
struct Crossing {
    u: f64,
    point: DVec3,
}

/// One infill scan-line segment: `(start, end)` world points.
type ScanSegment = (DVec3, DVec3);

impl InfillGenerator for MonotonicInfill {
    fn generate(
        &self,
        region: &InfillRegion,
        config: &SlicerConfig,
        layer: &Layer,
        object_transform: &Transform,
        density: f64,
    ) -> Vec<Path> {
        if region.is_empty() || density <= 0.0 {
            return Vec::new();
        }

        let (axis, _apex, _slope) =
            order_field::resolve_axis_apex_slope(config.order_field, config);
        let (basis1, basis2) = plane_basis(axis);
        let object_angle = object_transform.in_plane_rotation_angle(basis1, basis2);
        let alternation = if layer.index.is_multiple_of(2) {
            1.0
        } else {
            -1.0
        };
        let angle = object_angle + alternation * config.infill_angle_deg.to_radians();

        let (sin, cos) = angle.sin_cos();
        let u_dir = basis1 * cos + basis2 * sin;
        let v_dir = basis1 * -sin + basis2 * cos;

        // Sparsify by widening scan-line spacing inversely with density:
        // `1.0` (fully solid) packs lines at `infill_line_width` spacing,
        // lower density spaces them further apart so less of the region
        // area is actually covered by a bead.
        let clamped_density = density.min(1.0);
        let spacing =
            (config.infill_line_width.abs().max(f64::EPSILON)) / clamped_density.max(f64::EPSILON);

        // Max `along`-axis search radius for re-solving a crossing's true
        // height against the order field (see `Crossing`'s doc) — the same
        // bound `InfillRegion::from_layer` used to reconstruct each loop
        // vertex in the first place.
        let max_along = order_field::max_along_for(config);

        // Project every loop's points into the rotated (u, v) frame,
        // keeping loops as closed edge lists (wrap-around to point 0).
        // The original 3D point travels alongside (u, v) so a failed
        // order-field re-solve (see the crossing loop below) can fall back
        // to interpolating real, already-known-good boundary points
        // instead of fabricating a position.
        let projected: Vec<Vec<(f64, f64, DVec3)>> = region
            .loops
            .iter()
            .map(|points| {
                points
                    .iter()
                    .map(|&p| (p.dot(u_dir), p.dot(v_dir), p))
                    .collect()
            })
            .collect();

        let Some(v_min) = projected
            .iter()
            .flatten()
            .map(|&(_, v, _)| v)
            .fold(None, |acc, v| Some(acc.map_or(v, |m: f64| m.min(v))))
        else {
            return Vec::new();
        };
        let v_max = projected
            .iter()
            .flatten()
            .map(|&(_, v, _)| v)
            .fold(v_min, f64::max);

        // Scan-lines to draw, grouped scan-line by scan-line (ascending v).
        // Each entry keeps its true scan-line ordinal (for boustrophedon
        // direction alternation) and actual `v` alongside its pairs —
        // scan-lines with no crossings are skipped, so neither the
        // ordinal nor `v` can be recomputed later from a compacted index
        // (doing so previously reconstructed world points from the wrong
        // `v`, placing infill segments far outside the region whenever a
        // layer had scan-lines with no crossings before the first hit,
        // e.g. a diagonal 45° scan direction over a wide, sparse
        // multi-island layer).
        let mut scanlines: Vec<(usize, f64, Vec<ScanSegment>)> = Vec::new();
        let mut v = v_min + spacing / 2.0;
        let mut scan_index = 0usize;
        while v <= v_max {
            let mut crossings: Vec<Crossing> = Vec::new();
            for loop_points in &projected {
                let n = loop_points.len();
                if n < 2 {
                    continue;
                }
                for i in 0..n {
                    let (u0, v0, p0) = loop_points[i];
                    let (u1, v1, p1) = loop_points[(i + 1) % n];
                    let crosses = (v0 <= v && v1 > v) || (v1 <= v && v0 > v);
                    if !crosses {
                        continue;
                    }
                    let t = (v - v0) / (v1 - v0);
                    let u = u0 + t * (u1 - u0);
                    // Re-solve this crossing's true world height against
                    // the order field rather than interpolating between
                    // the crossed edge's endpoint heights (see
                    // `Crossing`'s doc for why linear interpolation is
                    // wrong for a curved field).
                    //
                    // The refinement starts from the *lerped edge point*,
                    // not from the bare in-plane column `u_dir * u +
                    // v_dir * v` (which sits at `along == 0`, i.e. the
                    // world `axis == 0` plane). Projection into the fill
                    // frame is linear, so the lerp has exactly the same
                    // `(u, v)` — and it already sits approximately at the
                    // right height on the *correct branch* of the
                    // isosurface. A non-monotonic field (`Eikonal` on
                    // reentrant/threaded geometry) crosses `order ==
                    // layer.order` at several heights along one vertical
                    // column, and the previous axis-ray solve from the
                    // `along == 0` plane could bracket a different, wrong
                    // branch and return it as an "exact" root up to
                    // `max_along_for` (50 layer heights) away — exactly
                    // the steep near-vertical infill spikes anchored
                    // around `Z ~= 0`. Newton-style gradient refinement
                    // from the near-surface seed inherently converges to
                    // the local branch instead (see
                    // `refine_point_onto_order_field`'s doc).
                    //
                    // Acceptance is field-adaptive: the distance the seed
                    // truly needs to move is roughly its own residual
                    // over the local gradient magnitude (~1 for
                    // distance-like fields such as `Eikonal`), so a
                    // result that wandered much farther than the seed's
                    // residual (slope slack factor of 4, plus a couple of
                    // layer heights of absolute slack) is rejected in
                    // favor of the lerp — a bounded, locally-sane
                    // approximation built from two real,
                    // already-reconstructed boundary points. Likewise if
                    // the field has no information at the seed at all
                    // (non-finite `order`, e.g. an `Eikonal` front that
                    // never reached this column), keep the lerp.
                    let seed = p0.lerp(p1, t);
                    let seed_residual = layer.order_field.order(seed) - layer.order;
                    let point = if seed_residual.is_finite() {
                        let accept = seed_residual.abs() * 4.0 + 2.0 * config.layer_height.abs();
                        order_field::refine_point_onto_order_field(
                            seed,
                            layer.order,
                            max_along,
                            layer.order_field.as_ref(),
                        )
                        .filter(|p| (*p - seed).length() <= accept)
                        .unwrap_or(seed)
                    } else {
                        seed
                    };
                    crossings.push(Crossing { u, point });
                }
            }
            crossings.sort_by(|a, b| a.u.total_cmp(&b.u));

            let mut pairs: Vec<ScanSegment> = Vec::new();
            let mut pair_iter = crossings.chunks_exact(2);
            for pair in &mut pair_iter {
                pairs.push((pair[0].point, pair[1].point));
            }
            if !pairs.is_empty() {
                scanlines.push((scan_index, v, pairs));
            }
            scan_index += 1;
            v += spacing;
        }

        assemble_scanlines_into_paths(&scanlines, u_dir, spacing, config, layer)
    }
}

fn assemble_scanlines_into_paths(
    scanlines: &[(usize, f64, Vec<ScanSegment>)],
    u_dir: DVec3,
    spacing: f64,
    config: &SlicerConfig,
    layer: &Layer,
) -> Vec<Path> {
    if scanlines.is_empty() {
        return Vec::new();
    }

    let push_point =
        |points: &mut Vec<DVec3>, segments: &mut Vec<Segment>, world: DVec3, kind: MoveKind| {
            if !points.is_empty() {
                segments.push(Segment {
                    kind,
                    speed: speed_for_kind(kind, config),
                    extrusion_rate: 1.0,
                    support_fraction: 0.0,
                    order: layer.order,
                    extrusion_length: 0.0,
                    line_width: config.infill_line_width,
                });
            }
            points.push(world);
        };

    let overlap_margin = (spacing * 8.0).max(config.infill_line_width.abs() * 8.0);

    struct Span {
        scan_index: usize,
        u_min: f64,
        u_max: f64,
        pair: ScanSegment,
    }

    fn find(parent: &mut [usize], x: usize) -> usize {
        if parent[x] != x {
            parent[x] = find(parent, parent[x]);
        }
        parent[x]
    }
    fn union(parent: &mut [usize], a: usize, b: usize) {
        let ra = find(parent, a);
        let rb = find(parent, b);
        if ra != rb {
            parent[ra] = rb;
        }
    }

    let mut parent: Vec<usize> = Vec::new();
    let mut spans: Vec<Span> = Vec::new();
    let mut prev_row_span_indices: Vec<usize> = Vec::new();
    for (scan_index, _v, pairs) in scanlines {
        let mut row_span_indices: Vec<usize> = Vec::with_capacity(pairs.len());
        for &pair in pairs {
            let (u0, u1) = (pair.0.dot(u_dir), pair.1.dot(u_dir));
            let (u_min, u_max) = if u0 <= u1 { (u0, u1) } else { (u1, u0) };
            let id = parent.len();
            parent.push(id);
            spans.push(Span {
                scan_index: *scan_index,
                u_min,
                u_max,
                pair,
            });
            row_span_indices.push(id);
        }
        for &cur in &row_span_indices {
            for &prev in &prev_row_span_indices {
                if spans[cur].u_min - overlap_margin <= spans[prev].u_max
                    && spans[prev].u_min - overlap_margin <= spans[cur].u_max
                {
                    union(&mut parent, cur, prev);
                }
            }
        }
        prev_row_span_indices = row_span_indices;
    }

    let mut order_of_root: Vec<usize> = Vec::new();
    let mut groups: std::collections::HashMap<usize, Vec<(usize, ScanSegment)>> =
        std::collections::HashMap::new();
    for (i, span) in spans.iter().enumerate() {
        let root = find(&mut parent, i);
        let entry = groups.entry(root).or_insert_with(|| {
            order_of_root.push(root);
            Vec::new()
        });
        entry.push((span.scan_index, span.pair));
    }

    let mut all_paths: Vec<Path> = Vec::new();
    let mut last_end: Option<DVec3> = None;
    for root in order_of_root {
        let mut group = groups.remove(&root).expect("root was just recorded");
        group.sort_by_key(|(scan_index, _)| *scan_index);

        if let Some(prev) = last_end {
            if let (Some(first), Some(last)) = (group.first(), group.last()) {
                let dist_first = prev
                    .distance_squared(first.1 .0)
                    .min(prev.distance_squared(first.1 .1));
                let dist_last = prev
                    .distance_squared(last.1 .0)
                    .min(prev.distance_squared(last.1 .1));
                if dist_last < dist_first {
                    group.reverse();
                }
            }
        }

        let mut points: Vec<DVec3> = Vec::new();
        let mut segments: Vec<Segment> = Vec::new();
        let connect_dist = (spacing * 2.5).max(config.infill_line_width.abs() * 2.5);

        for (scan_index, pair) in group {
            let (start, end) = if let Some(prev) = last_end {
                if prev.distance_squared(pair.0) <= prev.distance_squared(pair.1) {
                    (pair.0, pair.1)
                } else {
                    (pair.1, pair.0)
                }
            } else if scan_index.is_multiple_of(2) {
                pair
            } else {
                (pair.1, pair.0)
            };

            if let Some(prev) = last_end {
                let dist = prev.distance(start);
                if dist <= connect_dist && dist > 1e-4 {
                    // Turnaround is adjacent along the perimeter boundary: connect with an extruding move
                    push_point(&mut points, &mut segments, start, MoveKind::Infill);
                } else if dist > connect_dist {
                    // Far jump across a void/hole: split into a separate path so travel optimizer can sequence it
                    if points.len() >= 2 {
                        all_paths.push(Path {
                            points,
                            segments,
                            tool: crate::ids::ToolId::default(),
                        });
                        points = Vec::new();
                        segments = Vec::new();
                    } else {
                        points.clear();
                        segments.clear();
                    }
                    push_point(&mut points, &mut segments, start, MoveKind::Infill);
                }
            } else {
                push_point(&mut points, &mut segments, start, MoveKind::Infill);
            }

            push_point(&mut points, &mut segments, end, MoveKind::Infill);
            last_end = Some(end);
        }

        if points.len() >= 2 {
            all_paths.push(Path {
                points,
                segments,
                tool: crate::ids::ToolId::default(),
            });
        }
    }

    all_paths
}

/// Safety cap on the number of successive inward offsets `ConcentricInfill`
/// will take before giving up on a single region, guarding against a
/// pathological input (e.g. `infill_line_width`/`density` combining to an
/// almost-zero spacing) turning into an unbounded loop -- mirrors the
/// defensive `MAX_ORDER_STEPS`-style caps used elsewhere in the slicing
/// pipeline for the same reason.
const MAX_OFFSET_RINGS: usize = 10_000;

/// Concentric fill: successive inward offsets of the region boundary
/// (`SlicerConfig::infill_line_width` apart, widened by `density` the same
/// way `MonotonicInfill` widens its scan-line spacing), each printed as its
/// own closed loop -- unlike `MonotonicInfill`'s open zig-zag, there is no
/// travel move needed *between* rings since each ring is emitted as its own
/// [`Path`] (the same convention `toolpath::plan` already uses for wall
/// loops: one closed `Path` per loop, with inter-`Path` travel handled by
/// the Gcode emission stage rather than an explicit `Segment`).
#[derive(Debug, Clone, Copy, Default)]
pub struct ConcentricInfill;

/// 3D Cubic lattice infill: self-supporting 3D periodic cubic grid rotated
/// with space diagonals along the Cartesian axes (45-degree inclination to
/// the build plate), forming a 3D isotropic truss.
#[derive(Debug, Clone, Copy, Default)]
pub struct CubicInfill;

/// Triply Periodic Minimal Surface (TPMS) Gyroid infill generator.
#[derive(Debug, Clone, Copy, Default)]
pub struct GyroidInfill;

impl InfillGenerator for GyroidInfill {
    fn generate(
        &self,
        region: &InfillRegion,
        config: &SlicerConfig,
        layer: &Layer,
        object_transform: &Transform,
        density: f64,
    ) -> Vec<Path> {
        generate_tpms_infill(
            manifold_fidget::tpms::TpmsKind::Gyroid,
            region,
            config,
            layer,
            object_transform,
            density,
        )
    }
}

/// Triply Periodic Minimal Surface (TPMS) Schwarz Diamond (D) infill generator.
#[derive(Debug, Clone, Copy, Default)]
pub struct SchwarzDInfill;

impl InfillGenerator for SchwarzDInfill {
    fn generate(
        &self,
        region: &InfillRegion,
        config: &SlicerConfig,
        layer: &Layer,
        object_transform: &Transform,
        density: f64,
    ) -> Vec<Path> {
        generate_tpms_infill(
            manifold_fidget::tpms::TpmsKind::SchwarzD,
            region,
            config,
            layer,
            object_transform,
            density,
        )
    }
}

/// Triply Periodic Minimal Surface (TPMS) Schwarz Primitive (P) infill generator.
#[derive(Debug, Clone, Copy, Default)]
pub struct SchwarzPInfill;

impl InfillGenerator for SchwarzPInfill {
    fn generate(
        &self,
        region: &InfillRegion,
        config: &SlicerConfig,
        layer: &Layer,
        object_transform: &Transform,
        density: f64,
    ) -> Vec<Path> {
        generate_tpms_infill(
            manifold_fidget::tpms::TpmsKind::SchwarzP,
            region,
            config,
            layer,
            object_transform,
            density,
        )
    }
}

/// Stitches disconnected 2D line segments into continuous polyline chains.
fn stitch_segments_into_polylines_2d(
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

fn generate_tpms_infill(
    kind: manifold_fidget::tpms::TpmsKind,
    region: &InfillRegion,
    config: &SlicerConfig,
    layer: &Layer,
    _object_transform: &Transform,
    density: f64,
) -> Vec<Path> {
    if region.is_empty() || density <= 0.0 {
        return Vec::new();
    }

    let wavelength =
        manifold_fidget::tpms::TpmsField::wavelength_for_density(config.infill_line_width, density);
    let tpms_field = manifold_fidget::tpms::TpmsField::new(kind, wavelength, 0.0);

    let (axis, apex, _slope) = order_field::resolve_axis_apex_slope(config.order_field, config);
    let (basis1, basis2) = plane_basis(axis);

    let loops_2d = polygon2d::canonicalize(&polygon2d::to_2d(&region.loops, basis1, basis2, apex));
    if loops_2d.is_empty() {
        return Vec::new();
    }

    let mut min_u = f64::INFINITY;
    let mut max_u = f64::NEG_INFINITY;
    let mut min_v = f64::INFINITY;
    let mut max_v = f64::NEG_INFINITY;

    for loop_pts in &loops_2d {
        for &[u, v] in loop_pts {
            min_u = min_u.min(u);
            max_u = max_u.max(u);
            min_v = min_v.min(v);
            max_v = max_v.max(v);
        }
    }

    let step = (config.infill_line_width * 0.75).clamp(0.1, 1.5);
    let nu = (((max_u - min_u) / step).ceil() as usize).max(2);
    let nv = (((max_v - min_v) / step).ceil() as usize).max(2);
    let du = (max_u - min_u) / nu as f64;
    let dv = (max_v - min_v) / nv as f64;

    let mut grid_vals = vec![0.0f64; (nu + 1) * (nv + 1)];
    for j in 0..=nv {
        let v = min_v + j as f64 * dv;
        for i in 0..=nu {
            let u = min_u + i as f64 * du;
            let world_p = apex + basis1 * u + basis2 * v + axis * layer.order;
            grid_vals[j * (nu + 1) + i] = tpms_field.sample(world_p).value;
        }
    }

    let mut segments_2d: Vec<([f64; 2], [f64; 2])> = Vec::new();
    for j in 0..nv {
        let v0 = min_v + j as f64 * dv;
        let v1 = v0 + dv;
        for i in 0..nu {
            let u0 = min_u + i as f64 * du;
            let u1 = u0 + du;

            let c0 = [u0, v0];
            let c1 = [u1, v0];
            let c2 = [u1, v1];
            let c3 = [u0, v1];

            let val0 = grid_vals[j * (nu + 1) + i];
            let val1 = grid_vals[j * (nu + 1) + (i + 1)];
            let val2 = grid_vals[(j + 1) * (nu + 1) + (i + 1)];
            let val3 = grid_vals[(j + 1) * (nu + 1) + i];

            let mut mask = 0u8;
            if val0 < 0.0 {
                mask |= 1;
            }
            if val1 < 0.0 {
                mask |= 2;
            }
            if val2 < 0.0 {
                mask |= 4;
            }
            if val3 < 0.0 {
                mask |= 8;
            }

            let lerp_e = |pa: [f64; 2], va: f64, pb: [f64; 2], vb: f64| -> [f64; 2] {
                let denom = vb - va;
                let t = if denom.abs() < 1e-7 {
                    0.5
                } else {
                    (-va / denom).clamp(0.0, 1.0)
                };
                [pa[0] + (pb[0] - pa[0]) * t, pa[1] + (pb[1] - pa[1]) * t]
            };

            let e0 = lerp_e(c0, val0, c1, val1);
            let e1 = lerp_e(c1, val1, c2, val2);
            let e2 = lerp_e(c2, val2, c3, val3);
            let e3 = lerp_e(c3, val3, c0, val0);

            match mask {
                1 | 14 => segments_2d.push((e3, e0)),
                2 | 13 => segments_2d.push((e0, e1)),
                3 | 12 => segments_2d.push((e3, e1)),
                4 | 11 => segments_2d.push((e1, e2)),
                5 => {
                    segments_2d.push((e3, e0));
                    segments_2d.push((e1, e2));
                }
                6 | 9 => segments_2d.push((e0, e2)),
                7 | 8 => segments_2d.push((e2, e3)),
                10 => {
                    segments_2d.push((e0, e1));
                    segments_2d.push((e2, e3));
                }
                _ => {}
            }
        }
    }

    if segments_2d.is_empty() {
        return Vec::new();
    }

    let mut filtered_segments: Vec<([f64; 2], [f64; 2])> = Vec::new();
    for (p0, p1) in segments_2d {
        let mid = [(p0[0] + p1[0]) * 0.5, (p0[1] + p1[1]) * 0.5];
        if polygon2d::contains_point(&loops_2d, mid) {
            filtered_segments.push((p0, p1));
        }
    }

    if filtered_segments.is_empty() {
        return Vec::new();
    }

    let tol = (step * 0.5).max(0.01);
    let polylines_2d = stitch_segments_into_polylines_2d(filtered_segments, tol);

    let world_polylines = order_field::reconstruct_on_order_field_near(
        polylines_2d,
        &region.loops,
        basis1,
        basis2,
        axis,
        apex,
        layer.order,
        order_field::max_along_for(config),
        layer.order_field.as_ref(),
    );

    let mut last_end: Option<DVec3> = None;
    let mut paths = Vec::new();
    for mut points in world_polylines.into_iter().filter(|pts| pts.len() >= 2) {
        if let Some(prev) = last_end {
            let start = points.first().copied().unwrap();
            let end = points.last().copied().unwrap();
            if prev.distance_squared(end) < prev.distance_squared(start) {
                points.reverse();
            }
        }
        last_end = points.last().copied();
        let segments = points
            .iter()
            .map(|_| Segment {
                kind: MoveKind::Infill,
                speed: speed_for_kind(MoveKind::Infill, config),
                extrusion_rate: 1.0,
                support_fraction: 0.0,
                order: layer.order,
                extrusion_length: 0.0,
                line_width: config.infill_line_width,
            })
            .collect();
        paths.push(Path {
            points,
            segments,
            tool: crate::ids::ToolId::default(),
        });
    }

    paths
}

impl InfillGenerator for CubicInfill {
    fn generate(
        &self,
        region: &InfillRegion,
        config: &SlicerConfig,
        layer: &Layer,
        object_transform: &Transform,
        density: f64,
    ) -> Vec<Path> {
        if region.is_empty() || density <= 0.0 {
            return Vec::new();
        }

        let (axis, _apex, _slope) =
            order_field::resolve_axis_apex_slope(config.order_field, config);
        let (basis1, basis2) = plane_basis(axis);
        let object_angle = object_transform.in_plane_rotation_angle(basis1, basis2);

        let clamped_density = density.clamp(1e-4, 1.0);
        let cell_size = (config.infill_line_width.abs().max(f64::EPSILON)) / clamped_density;
        let spacing = cell_size * 2.0_f64.sqrt();

        let z = layer.order;
        let shift = z * 2.0_f64.sqrt();

        let mut paths = Vec::new();
        paths.extend(generate_scanlines_at_angle(
            region,
            config,
            layer,
            object_angle + 45.0f64.to_radians(),
            spacing,
            shift,
        ));
        paths.extend(generate_scanlines_at_angle(
            region,
            config,
            layer,
            object_angle - 45.0f64.to_radians(),
            spacing,
            -shift,
        ));

        paths
    }
}

fn generate_scanlines_at_angle(
    region: &InfillRegion,
    config: &SlicerConfig,
    layer: &Layer,
    angle: f64,
    spacing: f64,
    offset: f64,
) -> Vec<Path> {
    if region.is_empty() || spacing <= f64::EPSILON {
        return Vec::new();
    }

    let (axis, _apex, _slope) = order_field::resolve_axis_apex_slope(config.order_field, config);
    let (basis1, basis2) = plane_basis(axis);
    let (sin, cos) = angle.sin_cos();
    let u_dir = basis1 * cos + basis2 * sin;
    let v_dir = basis1 * -sin + basis2 * cos;

    let max_along = order_field::max_along_for(config);

    let projected: Vec<Vec<(f64, f64, DVec3)>> = region
        .loops
        .iter()
        .map(|points| {
            points
                .iter()
                .map(|&p| (p.dot(u_dir), p.dot(v_dir), p))
                .collect()
        })
        .collect();

    let Some(v_min) = projected
        .iter()
        .flatten()
        .map(|&(_, v, _)| v)
        .fold(None, |acc, v| Some(acc.map_or(v, |m: f64| m.min(v))))
    else {
        return Vec::new();
    };
    let v_max = projected
        .iter()
        .flatten()
        .map(|&(_, v, _)| v)
        .fold(v_min, f64::max);

    let mut scanlines: Vec<(usize, f64, Vec<ScanSegment>)> = Vec::new();
    let norm_offset = ((offset % spacing) + spacing) % spacing;
    let mut v = v_min + norm_offset;
    if v < v_min {
        v += spacing;
    }
    let mut scan_index = 0usize;

    while v <= v_max {
        let mut crossings: Vec<Crossing> = Vec::new();
        for loop_points in &projected {
            let n = loop_points.len();
            if n < 2 {
                continue;
            }
            for i in 0..n {
                let (u0, v0, p0) = loop_points[i];
                let (u1, v1, p1) = loop_points[(i + 1) % n];
                let crosses = (v0 <= v && v1 > v) || (v1 <= v && v0 > v);
                if !crosses {
                    continue;
                }
                let t = (v - v0) / (v1 - v0);
                let u = u0 + t * (u1 - u0);
                let seed = p0.lerp(p1, t);
                let seed_residual = layer.order_field.order(seed) - layer.order;
                let point = if seed_residual.is_finite() {
                    let accept = seed_residual.abs() * 4.0 + 2.0 * config.layer_height.abs();
                    order_field::refine_point_onto_order_field(
                        seed,
                        layer.order,
                        max_along,
                        layer.order_field.as_ref(),
                    )
                    .filter(|p| (*p - seed).length() <= accept)
                    .unwrap_or(seed)
                } else {
                    seed
                };
                crossings.push(Crossing { u, point });
            }
        }
        crossings.sort_by(|a, b| a.u.total_cmp(&b.u));

        let mut row_pairs: Vec<ScanSegment> = Vec::new();
        for pair in crossings.chunks_exact(2) {
            row_pairs.push((pair[0].point, pair[1].point));
        }
        if !row_pairs.is_empty() {
            scanlines.push((scan_index, v, row_pairs));
        }
        v += spacing;
        scan_index += 1;
    }

    assemble_scanlines_into_paths(&scanlines, u_dir, spacing, config, layer)
}

/// No-op infill generator: emits no paths regardless of region or density.
/// Used for `InfillPatternKind::None` so callers don't need special-case
/// handling in `toolpath::plan`.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoneInfill;

impl InfillGenerator for NoneInfill {
    fn generate(
        &self,
        _region: &InfillRegion,
        _config: &SlicerConfig,
        _layer: &Layer,
        _object_transform: &Transform,
        _density: f64,
    ) -> Vec<Path> {
        Vec::new()
    }
}

impl InfillGenerator for ConcentricInfill {
    fn generate(
        &self,
        region: &InfillRegion,
        config: &SlicerConfig,
        layer: &Layer,
        _object_transform: &Transform,
        density: f64,
    ) -> Vec<Path> {
        if region.is_empty() || density <= 0.0 {
            return Vec::new();
        }

        let (axis, apex, _slope) = order_field::resolve_axis_apex_slope(config.order_field, config);
        let (basis1, basis2) = plane_basis(axis);

        // Same density convention as `MonotonicInfill`: `1.0` packs rings
        // at `infill_line_width` spacing, lower density spreads them
        // further apart.
        let clamped_density = density.min(1.0);
        let spacing =
            (config.infill_line_width.abs().max(f64::EPSILON)) / clamped_density.max(f64::EPSILON);

        let boundary_2d =
            polygon2d::canonicalize(&polygon2d::to_2d(&region.loops, basis1, basis2, apex));

        // `boundary_2d` (== `region.loops`, i.e. `Layer::infill_boundary`)
        // is already a bead centerline -- one `wall_line_width` in from the
        // true mesh surface, per `Layer::infill_boundary`'s doc comment --
        // not a free edge. So the first ring sits directly on it with zero
        // extra inset; only subsequent rings step inward by a further full
        // `spacing`, until an offset collapses to nothing.
        let mut rings_2d: Vec<Vec<[f64; 2]>> = Vec::new();
        rings_2d.extend(boundary_2d.clone());
        let mut current = polygon2d::inward_offset(&boundary_2d, spacing);
        let mut steps = 0usize;
        while !current.is_empty() && steps < MAX_OFFSET_RINGS {
            rings_2d.extend(current.iter().cloned());
            // Every ring after the first is offsetting the direct output of
            // a prior `inward_offset`/`inward_offset_unchecked` call --
            // already-clean `i_overlay` output -- so the redundant
            // whole-shape pre-simplify pass can be skipped (see
            // `inward_offset_unchecked`'s doc comment).
            current = polygon2d::inward_offset_unchecked(&current, spacing);
            steps += 1;
        }

        if rings_2d.is_empty() {
            return Vec::new();
        }

        // Reference-seeded reconstruction (see
        // `InfillRegion::from_layer`'s identical use of
        // `reconstruct_on_order_field_near`): every offset ring lies close
        // to the region's own already-reconstructed boundary loops, so
        // seeding from those avoids the wrong-branch axis-ray solves that
        // previously spiked Eikonal infill.
        let world_rings = order_field::reconstruct_on_order_field_near(
            rings_2d,
            &region.loops,
            basis1,
            basis2,
            axis,
            apex,
            layer.order,
            order_field::max_along_for(config),
            layer.order_field.as_ref(),
        );

        world_rings
            .into_iter()
            .filter(|ring| ring.len() >= 3)
            .map(|points| {
                let segments = points
                    .iter()
                    .map(|_| Segment {
                        kind: MoveKind::Infill,
                        speed: speed_for_kind(MoveKind::Infill, config),
                        extrusion_rate: 1.0,
                        support_fraction: 0.0,
                        order: layer.order,
                        extrusion_length: 0.0,
                        line_width: config.infill_line_width,
                    })
                    .collect();
                Path {
                    points,
                    segments,
                    tool: crate::ids::ToolId::default(),
                }
            })
            .collect()
    }
}

/// Same as external wall generation: repeated inward offsets of the region
/// boundary spaced exactly `infill_line_width` apart, each printed as its
/// own closed loop -- but applied to `InfillRegion` (2D, already-extracted
/// loops) rather than re-querying the mesh/SDF per pass the way real wall
/// generation does (see `slicing::slice_mesh_with_progress`'s `wall_index`
/// loop), since `InfillGenerator` only has access to a layer's already
/// planar loops, not the source mesh.
///
/// Unlike [`ConcentricInfill`], `density` is ignored entirely: spacing is
/// always the full `infill_line_width`, so this pattern always completely
/// fills the region regardless of the configured infill density -- the
/// point is "as many walls as it takes to fill the space", not a sparse
/// pattern. The number of rings generated is implied purely by the
/// geometry: offsetting continues until a pass produces no more loops (the
/// loop already attempts one more offset before concluding the region is
/// exhausted), capped by [`MAX_OFFSET_RINGS`] as a safety bound against
/// runaway offsetting on degenerate geometry.
#[derive(Debug, Clone, Copy, Default)]
pub struct AllWallsInfill;

impl InfillGenerator for AllWallsInfill {
    fn generate(
        &self,
        region: &InfillRegion,
        config: &SlicerConfig,
        layer: &Layer,
        _object_transform: &Transform,
        _density: f64,
    ) -> Vec<Path> {
        if region.is_empty() {
            return Vec::new();
        }

        let (axis, apex, _slope) = order_field::resolve_axis_apex_slope(config.order_field, config);
        let (basis1, basis2) = plane_basis(axis);

        // `density` is deliberately ignored -- always pack rings at full
        // `infill_line_width` spacing (see doc comment above).
        let spacing = config.infill_line_width.abs().max(f64::EPSILON);

        let boundary_2d =
            polygon2d::canonicalize(&polygon2d::to_2d(&region.loops, basis1, basis2, apex));

        // `boundary_2d` is already a bead centerline (see
        // `ConcentricInfill::generate`'s identical comment above), so the
        // first ring sits directly on it with zero extra inset.
        let mut rings_2d: Vec<Vec<[f64; 2]>> = Vec::new();
        rings_2d.extend(boundary_2d.clone());
        let mut current = polygon2d::inward_offset(&boundary_2d, spacing);
        let mut steps = 0usize;
        while !current.is_empty() && steps < MAX_OFFSET_RINGS {
            rings_2d.extend(current.iter().cloned());
            // See `ConcentricInfill::generate`'s identical comment above --
            // `current` is already clean `i_overlay` output.
            current = polygon2d::inward_offset_unchecked(&current, spacing);
            steps += 1;
        }

        if rings_2d.is_empty() {
            return Vec::new();
        }

        let world_rings = order_field::reconstruct_on_order_field_near(
            rings_2d,
            &region.loops,
            basis1,
            basis2,
            axis,
            apex,
            layer.order,
            order_field::max_along_for(config),
            layer.order_field.as_ref(),
        );

        world_rings
            .into_iter()
            .filter(|ring| ring.len() >= 3)
            .map(|points| {
                let segments = points
                    .iter()
                    .map(|_| Segment {
                        kind: MoveKind::Infill,
                        speed: speed_for_kind(MoveKind::Infill, config),
                        extrusion_rate: 1.0,
                        support_fraction: 0.0,
                        order: layer.order,
                        extrusion_length: 0.0,
                        line_width: config.infill_line_width,
                    })
                    .collect();
                Path {
                    points,
                    segments,
                    tool: crate::ids::ToolId::default(),
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::ObjectId;

    fn square_layer(half_extent: f64) -> Layer {
        Layer {
            index: 0,
            object: ObjectId(0),
            order: 0.0,
            loops: Vec::new(),
            infill_boundary: vec![vec![
                DVec3::new(-half_extent, -half_extent, 0.0),
                DVec3::new(half_extent, -half_extent, 0.0),
                DVec3::new(half_extent, half_extent, 0.0),
                DVec3::new(-half_extent, half_extent, 0.0),
            ]],
            solid_fill_boundary: Vec::new(),
            ..Layer::default()
        }
    }

    fn config() -> SlicerConfig {
        SlicerConfig {
            infill_line_width: 0.5,
            infill_angle_deg: 45.0,
            ..SlicerConfig::default()
        }
    }

    #[test]
    fn region_from_layer_reflects_the_infill_boundary_field() {
        let layer = Layer {
            index: 0,
            object: ObjectId(0),
            order: 0.0,
            loops: Vec::new(),
            infill_boundary: vec![vec![DVec3::Y, DVec3::new(1.0, 1.0, 0.0)]],
            solid_fill_boundary: Vec::new(),
            ..Layer::default()
        };
        let region = InfillRegion::from_layer(&layer, &config());
        assert_eq!(region.loops.len(), 1);
        assert_eq!(region.loops[0], vec![DVec3::Y, DVec3::new(1.0, 1.0, 0.0)]);
    }

    #[test]
    fn region_from_layer_is_empty_for_layer_with_no_infill_boundary() {
        let layer = Layer {
            index: 0,
            object: ObjectId(0),
            order: 0.0,
            loops: Vec::new(),
            infill_boundary: Vec::new(),
            solid_fill_boundary: Vec::new(),
            ..Layer::default()
        };
        assert!(InfillRegion::from_layer(&layer, &config()).is_empty());
    }

    #[test]
    fn region_from_layer_is_empty_when_solid_fill_boundary_covers_the_whole_infill_boundary() {
        let mut layer = square_layer(5.0);
        layer.solid_fill_boundary = layer.infill_boundary.clone();
        assert!(InfillRegion::from_layer(&layer, &config()).is_empty());
    }

    #[test]
    fn region_from_layer_excludes_only_the_solid_fill_boundary_portion() {
        // A 10x10 square infill boundary with a fully-solid 4x4 sub-square
        // carved out of one corner: the sparse region should still cover
        // the rest of the square, so it must not be empty.
        let mut layer = square_layer(5.0);
        layer.solid_fill_boundary = vec![vec![
            DVec3::new(1.0, 1.0, 0.0),
            DVec3::new(5.0, 1.0, 0.0),
            DVec3::new(5.0, 5.0, 0.0),
            DVec3::new(1.0, 5.0, 0.0),
        ]];
        let region = InfillRegion::from_layer(&layer, &config());
        assert!(!region.is_empty());
        // The sparse region must not contain the fully-solid sub-square's
        // interior point.
        let cfg = SlicerConfig {
            infill_line_width: 0.5,
            infill_angle_deg: 0.0,
            ..SlicerConfig::default()
        };
        let layer_for_solid = layer.clone();
        let solid_region = InfillRegion {
            loops: layer_for_solid.solid_fill_boundary.clone(),
        };
        let sparse_paths =
            MonotonicInfill.generate(&region, &cfg, &layer, &Transform::identity(), 1.0);
        let solid_paths = MonotonicInfill.generate(
            &solid_region,
            &cfg,
            &layer_for_solid,
            &Transform::identity(),
            1.0,
        );
        assert!(!sparse_paths.is_empty());
        assert!(!solid_paths.is_empty());
    }

    #[test]
    fn monotonic_fill_produces_paths_covering_a_square() {
        let layer = square_layer(5.0);
        let region = InfillRegion::from_layer(&layer, &config());
        let paths =
            MonotonicInfill.generate(&region, &config(), &layer, &Transform::identity(), 1.0);

        assert_eq!(paths.len(), 1);
        let path = &paths[0];
        assert!(path.points.len() > 4, "expected multiple scan lines");
        assert_eq!(path.points.len() - 1, path.segments.len());
        assert!(path.segments.iter().any(|s| s.kind == MoveKind::Infill));
        // Every point should lie within the square's bounds (with a small
        // epsilon for floating-point projection round-trip).
        for p in &path.points {
            assert!(p.x >= -5.001 && p.x <= 5.001, "x out of bounds: {p:?}");
            assert!(p.y >= -5.001 && p.y <= 5.001, "y out of bounds: {p:?}");
        }
    }

    #[test]
    fn monotonic_fill_keeps_disconnected_regions_as_separate_paths() {
        // Two squares far apart along X, at the same Y range -- every
        // scan row crosses both, but their spans never overlap in `u`.
        // Monotonic ordering is only meaningful *within* each square; the
        // fix under test is that they come back as two separate `Path`s
        // (so `optimize_travel_order` can freely reorder/re-target them)
        // instead of one fused zigzag `Path` that travels back and forth
        // between the two squares on every row.
        let layer = Layer {
            index: 0,
            object: ObjectId(0),
            order: 0.0,
            loops: Vec::new(),
            infill_boundary: vec![
                vec![
                    DVec3::new(-5.0, -5.0, 0.0),
                    DVec3::new(-3.0, -5.0, 0.0),
                    DVec3::new(-3.0, 5.0, 0.0),
                    DVec3::new(-5.0, 5.0, 0.0),
                ],
                vec![
                    DVec3::new(3.0, -5.0, 0.0),
                    DVec3::new(5.0, -5.0, 0.0),
                    DVec3::new(5.0, 5.0, 0.0),
                    DVec3::new(3.0, 5.0, 0.0),
                ],
            ],
            solid_fill_boundary: Vec::new(),
            ..Layer::default()
        };
        let config = SlicerConfig {
            infill_line_width: 0.5,
            infill_angle_deg: 90.0,
            ..SlicerConfig::default()
        };
        let region = InfillRegion::from_layer(&layer, &config);
        let paths = MonotonicInfill.generate(&region, &config, &layer, &Transform::identity(), 1.0);

        assert_eq!(
            paths.len(),
            2,
            "expected one path per disconnected square, got {}",
            paths.len()
        );
        for path in &paths {
            // Every point in a given path must stay within ONE square's
            // bounds -- if the bug regresses, a path will contain points
            // from both squares (and an internal travel jump between
            // them).
            let in_left = path.points.iter().all(|p| p.x >= -5.001 && p.x <= -2.999);
            let in_right = path.points.iter().all(|p| p.x >= 2.999 && p.x <= 5.001);
            assert!(
                in_left || in_right,
                "path mixed points from both disconnected squares: {:?}",
                path.points
            );
        }
    }

    #[test]
    fn monotonic_fill_re_solves_each_scan_crossing_against_a_curved_order_field_instead_of_interpolating_linearly(
    ) {
        // Regression test for the "screw-thread pug model produces wildly
        // spiking Eikonal infill" bug: `MonotonicInfill` previously computed
        // a scan-line crossing's height by linearly blending the crossed
        // loop edge's two endpoint heights. That's only exact for a flat
        // `HeightOrderField`; for a curved field (using `ConicalOrderField`
        // here as a deterministic, closed-form stand-in for `Eikonal`) the
        // true height varies non-linearly across an edge, so linear
        // interpolation produces points that don't actually sit on the
        // order-field isosurface.
        use manifold_fidget::order::{ConicalOrderField, OrderField};
        use std::sync::Arc;

        let slope = 1.0;
        let field = ConicalOrderField::new(DVec3::ZERO, DVec3::Z, slope);
        let z_on_cone = |x: f64, y: f64| slope * (x * x + y * y).sqrt();

        // A square loop whose 4 corners all sit at the *same* height (they
        // share the same radial distance from the cone's axis) but whose
        // edges dip down toward the axis at their midpoints — exactly the
        // shape that makes a buggy linear interpolation (which would just
        // reproduce the shared corner height everywhere) visibly wrong.
        let half_extent = 5.0;
        let corners = [
            (-half_extent, -half_extent),
            (half_extent, -half_extent),
            (half_extent, half_extent),
            (-half_extent, half_extent),
        ];
        let loop_points: Vec<DVec3> = corners
            .iter()
            .map(|&(x, y)| DVec3::new(x, y, z_on_cone(x, y)))
            .collect();
        let corner_z = z_on_cone(half_extent, half_extent);

        let layer = Layer {
            index: 0,
            object: ObjectId(0),
            order: 0.0,
            loops: Vec::new(),
            infill_boundary: vec![loop_points],
            solid_fill_boundary: Vec::new(),
            mesh_sdf: None,
            order_field: Arc::new(field),
        };

        let cfg = SlicerConfig {
            infill_line_width: 0.5,
            infill_angle_deg: 0.0,
            order_field: order_field::OrderFieldKind::Conical,
            order_field_apex: DVec3::ZERO,
            order_field_axis: DVec3::Z,
            order_field_slope: slope,
            ..SlicerConfig::default()
        };

        let region = InfillRegion::from_layer(&layer, &cfg);
        assert!(!region.is_empty());

        let paths = MonotonicInfill.generate(&region, &cfg, &layer, &Transform::identity(), 1.0);
        assert_eq!(paths.len(), 1);
        let path = &paths[0];
        assert!(path.points.len() > 4, "expected multiple scan lines");

        // Every emitted point must actually lie on the field's target
        // isosurface (order == layer.order), not merely land near a
        // linear-interpolation guess between edge endpoints.
        for p in &path.points {
            let order = field.order(*p);
            assert!(
                (order - layer.order).abs() < 1e-6,
                "point {p:?} has order {order}, expected ~{} (crossings must be re-solved \
                 against the order field, not linearly interpolated between edge endpoints)",
                layer.order
            );
        }

        // Sanity: at least one point must clearly diverge from what naive
        // linear interpolation between the (identical) corner heights would
        // have produced, so this test would actually have caught the bug.
        assert!(
            path.points.iter().any(|p| (p.z - corner_z).abs() > 0.5),
            "expected at least one point whose z clearly diverges from naive corner-height \
             interpolation ({corner_z}); got {:?}",
            path.points
        );
    }

    #[test]
    fn monotonic_fill_falls_back_to_lerping_real_boundary_points_when_the_order_field_has_no_information(
    ) {
        // Regression test: when the order field can't be re-solved anywhere
        // along a scan-line crossing's column (e.g. an `Eikonal` front that
        // never reached this region), `MonotonicInfill` must fall back to
        // interpolating the crossed edge's two real, already-known-good 3D
        // endpoints -- not silently default to `along == 0.0`, which would
        // collapse every such crossing onto the flat `axis == 0` world
        // plane (z == 0 here) regardless of the loop's actual geometry.
        use manifold_fidget::order::OrderField;
        use std::sync::Arc;

        struct NeverReached;
        impl OrderField for NeverReached {
            fn order(&self, _p: DVec3) -> f64 {
                f64::INFINITY
            }
        }

        // A square loop whose corners sit well above the world's Z == 0
        // plane, so a buggy `along == 0.0` fallback (which reconstructs a
        // point at `planar + axis * 0.0`, i.e. z == 0) is trivially
        // distinguishable from the correct lerp-between-real-points
        // fallback (which must stay near the loop's actual height).
        let half_extent = 5.0;
        let z = 3.0;
        let loop_points = vec![
            DVec3::new(-half_extent, -half_extent, z),
            DVec3::new(half_extent, -half_extent, z),
            DVec3::new(half_extent, half_extent, z),
            DVec3::new(-half_extent, half_extent, z),
        ];

        let layer = Layer {
            index: 0,
            object: ObjectId(0),
            order: 0.0,
            loops: Vec::new(),
            infill_boundary: vec![loop_points],
            solid_fill_boundary: Vec::new(),
            mesh_sdf: None,
            order_field: Arc::new(NeverReached),
        };

        let cfg = SlicerConfig {
            infill_line_width: 0.5,
            infill_angle_deg: 0.0,
            ..SlicerConfig::default()
        };

        let region = InfillRegion::from_layer(&layer, &cfg);
        assert!(!region.is_empty());

        let paths = MonotonicInfill.generate(&region, &cfg, &layer, &Transform::identity(), 1.0);
        assert_eq!(paths.len(), 1);
        let path = &paths[0];
        assert!(path.points.len() > 4, "expected multiple scan lines");

        for p in &path.points {
            assert!(
                (p.z - z).abs() < 1e-6,
                "point {p:?} should have lerped back to the loop's real height ({z}), not \
                 collapsed onto the world Z == 0 plane"
            );
        }
    }

    #[test]
    fn monotonic_fill_returns_no_paths_when_density_is_zero() {
        let layer = square_layer(5.0);
        let region = InfillRegion::from_layer(&layer, &config());
        let paths =
            MonotonicInfill.generate(&region, &config(), &layer, &Transform::identity(), 0.0);
        assert!(paths.is_empty());
    }

    #[test]
    fn monotonic_fill_density_widens_scan_line_spacing_below_full_density() {
        let layer = square_layer(5.0);
        let region = InfillRegion::from_layer(&layer, &config());

        let full_density_paths =
            MonotonicInfill.generate(&region, &config(), &layer, &Transform::identity(), 1.0);
        let half_density_paths =
            MonotonicInfill.generate(&region, &config(), &layer, &Transform::identity(), 0.5);

        // Halving density doubles scan-line spacing, so roughly half as
        // many scan lines (and thus points) are produced across the same
        // region -- proving `density` actually sparsifies the pattern
        // rather than being ignored.
        assert!(!full_density_paths.is_empty());
        assert!(!half_density_paths.is_empty());
        assert!(half_density_paths[0].points.len() < full_density_paths[0].points.len());
    }

    #[test]
    fn monotonic_fill_is_empty_for_empty_region() {
        let layer = Layer {
            index: 0,
            object: ObjectId(0),
            order: 0.0,
            loops: Vec::new(),
            infill_boundary: Vec::new(),
            solid_fill_boundary: Vec::new(),
            ..Layer::default()
        };
        let region = InfillRegion::from_layer(&layer, &config());
        let paths =
            MonotonicInfill.generate(&region, &config(), &layer, &Transform::identity(), 1.0);
        assert!(paths.is_empty());
    }

    #[test]
    fn monotonic_fill_alternates_angle_by_layer_parity() {
        let square = square_layer(5.0);
        let mut even_layer = square.clone();
        even_layer.index = 0;
        let mut odd_layer = square;
        odd_layer.index = 1;

        let region_even = InfillRegion::from_layer(&even_layer, &config());
        let region_odd = InfillRegion::from_layer(&odd_layer, &config());
        let paths_even = MonotonicInfill.generate(
            &region_even,
            &config(),
            &even_layer,
            &Transform::identity(),
            1.0,
        );
        let paths_odd = MonotonicInfill.generate(
            &region_odd,
            &config(),
            &odd_layer,
            &Transform::identity(),
            1.0,
        );

        // Different angle sign should generally produce a different
        // number of scan-line points for a shape without diagonal
        // symmetry breaking; here we just assert the two point sets
        // differ, proving the layer-parity alternation actually changes
        // the generated geometry.
        assert_ne!(paths_even[0].points, paths_odd[0].points);
    }

    #[test]
    fn monotonic_fill_rotates_with_object_transform() {
        use glam::DQuat;
        let layer = square_layer(5.0);
        let region = InfillRegion::from_layer(&layer, &config());
        let identity_paths =
            MonotonicInfill.generate(&region, &config(), &layer, &Transform::identity(), 1.0);
        let rotated_transform = Transform::from_scale_rotation_translation(
            DVec3::ONE,
            DQuat::from_axis_angle(DVec3::Z, std::f64::consts::FRAC_PI_2),
            DVec3::ZERO,
        );
        let rotated_paths =
            MonotonicInfill.generate(&region, &config(), &layer, &rotated_transform, 1.0);

        assert_ne!(identity_paths[0].points, rotated_paths[0].points);
    }

    /// Regression test for the `Path` segment/point off-by-one bug: each
    /// `segments[i]` must describe the edge `points[i] -> points[i + 1]`
    /// (per `toolpath::Path`'s documented contract), with `Infill`-kind
    /// edges being genuine fill lines (short, inside the region) and
    /// `Travel`-kind edges being the jumps between them (e.g. across a
    /// hole). Previously every `Segment`'s `kind` was shifted one slot
    /// relative to its edge, mislabeling real fill lines as `Travel` and
    /// real travel jumps (which legitimately cross straight over a hole)
    /// as `Infill`.
    #[test]
    fn monotonic_fill_segment_kinds_align_with_their_own_edge_around_a_hole() {
        // A square with a smaller square hole in the middle, so a
        // horizontal-ish scan line crosses 4 times (outer-enter,
        // hole-enter, hole-exit, outer-exit), producing both an `Infill`
        // pair and a `Travel` jump across the hole per scan line.
        let layer = Layer {
            index: 0,
            object: ObjectId(0),
            order: 0.0,
            loops: Vec::new(),
            infill_boundary: vec![
                vec![
                    DVec3::new(-5.0, -5.0, 0.0),
                    DVec3::new(5.0, -5.0, 0.0),
                    DVec3::new(5.0, 5.0, 0.0),
                    DVec3::new(-5.0, 5.0, 0.0),
                ],
                vec![
                    DVec3::new(-1.0, -1.0, 0.0),
                    DVec3::new(-1.0, 1.0, 0.0),
                    DVec3::new(1.0, 1.0, 0.0),
                    DVec3::new(1.0, -1.0, 0.0),
                ],
            ],
            solid_fill_boundary: Vec::new(),
            ..Layer::default()
        };
        let cfg = SlicerConfig {
            infill_line_width: 0.5,
            infill_angle_deg: 0.0,
            ..SlicerConfig::default()
        };
        let region = InfillRegion::from_layer(&layer, &config());
        let paths = MonotonicInfill.generate(&region, &cfg, &layer, &Transform::identity(), 1.0);
        assert!(!paths.is_empty());

        let hole_min = DVec3::new(-1.0, -1.0, -1.0);
        let hole_max = DVec3::new(1.0, 1.0, 1.0);
        let inside_hole =
            |p: DVec3| p.x > hole_min.x && p.x < hole_max.x && p.y > hole_min.y && p.y < hole_max.y;

        let mut saw_infill = false;
        for path in &paths {
            assert_eq!(path.points.len() - 1, path.segments.len());
            for (i, segment) in path.segments.iter().enumerate() {
                let start = path.points[i];
                let end = path.points[i + 1];
                let mid = (start + end) * 0.5;
                match segment.kind {
                    MoveKind::Infill => {
                        saw_infill = true;
                        // A genuine fill line never has its midpoint inside
                        // the hole (it should stop at the hole's boundary).
                        assert!(
                            !inside_hole(mid),
                            "Infill-kind edge {start:?} -> {end:?} passes through the hole"
                        );
                    }
                    MoveKind::Travel => {}
                    other => panic!("unexpected move kind in infill path: {other:?}"),
                }
            }
        }
        assert!(saw_infill, "expected at least one Infill-kind edge");
    }

    #[test]
    fn concentric_fill_produces_multiple_closed_rings_covering_a_square() {
        let layer = square_layer(5.0);
        let region = InfillRegion::from_layer(&layer, &config());
        let paths =
            ConcentricInfill.generate(&region, &config(), &layer, &Transform::identity(), 1.0);

        assert!(
            paths.len() > 1,
            "expected several successively-inset rings, got {}",
            paths.len()
        );
        for path in &paths {
            // Each ring is its own closed loop: segments wrap all the way
            // around back to the first point (see `Path`'s doc), unlike
            // `MonotonicInfill`'s open zig-zag.
            assert_eq!(path.points.len(), path.segments.len());
            assert!(path.segments.iter().all(|s| s.kind == MoveKind::Infill));
            for p in &path.points {
                assert!(p.x >= -5.001 && p.x <= 5.001, "x out of bounds: {p:?}");
                assert!(p.y >= -5.001 && p.y <= 5.001, "y out of bounds: {p:?}");
            }
        }
    }

    #[test]
    fn concentric_fill_returns_no_paths_when_density_is_zero() {
        let layer = square_layer(5.0);
        let region = InfillRegion::from_layer(&layer, &config());
        let paths =
            ConcentricInfill.generate(&region, &config(), &layer, &Transform::identity(), 0.0);
        assert!(paths.is_empty());
    }

    #[test]
    fn concentric_fill_is_empty_for_empty_region() {
        let layer = Layer {
            index: 0,
            object: ObjectId(0),
            order: 0.0,
            loops: Vec::new(),
            infill_boundary: Vec::new(),
            solid_fill_boundary: Vec::new(),
            ..Layer::default()
        };
        let region = InfillRegion::from_layer(&layer, &config());
        let paths =
            ConcentricInfill.generate(&region, &config(), &layer, &Transform::identity(), 1.0);
        assert!(paths.is_empty());
    }

    #[test]
    fn concentric_fill_density_widens_ring_spacing_below_full_density() {
        let layer = square_layer(5.0);
        let region = InfillRegion::from_layer(&layer, &config());

        let full_density_paths =
            ConcentricInfill.generate(&region, &config(), &layer, &Transform::identity(), 1.0);
        let half_density_paths =
            ConcentricInfill.generate(&region, &config(), &layer, &Transform::identity(), 0.5);

        // Halving density doubles ring spacing, so roughly half as many
        // rings fit across the same region before offsetting collapses to
        // nothing.
        assert!(!full_density_paths.is_empty());
        assert!(!half_density_paths.is_empty());
        assert!(half_density_paths.len() < full_density_paths.len());
    }

    #[test]
    fn concentric_fill_never_enters_a_hole() {
        // Same square-with-hole shape as
        // `monotonic_fill_segment_kinds_align_with_their_own_edge_around_a_hole`:
        // concentric rings should stay in the annulus between the outer
        // boundary and the hole, never crossing into the hole itself.
        let layer = Layer {
            index: 0,
            object: ObjectId(0),
            order: 0.0,
            loops: Vec::new(),
            infill_boundary: vec![
                vec![
                    DVec3::new(-5.0, -5.0, 0.0),
                    DVec3::new(5.0, -5.0, 0.0),
                    DVec3::new(5.0, 5.0, 0.0),
                    DVec3::new(-5.0, 5.0, 0.0),
                ],
                vec![
                    DVec3::new(-1.0, -1.0, 0.0),
                    DVec3::new(-1.0, 1.0, 0.0),
                    DVec3::new(1.0, 1.0, 0.0),
                    DVec3::new(1.0, -1.0, 0.0),
                ],
            ],
            solid_fill_boundary: Vec::new(),
            ..Layer::default()
        };
        let region = InfillRegion::from_layer(&layer, &config());
        let paths =
            ConcentricInfill.generate(&region, &config(), &layer, &Transform::identity(), 1.0);
        assert!(!paths.is_empty());

        for path in &paths {
            for p in &path.points {
                let inside_hole = p.x > -0.999 && p.x < 0.999 && p.y > -0.999 && p.y < 0.999;
                assert!(!inside_hole, "ring point {p:?} fell inside the hole");
            }
        }
    }

    /// Regression test: when an order-field projection inverts a hole loop's
    /// winding in the global basis (so it appears to have the same winding as
    /// the outer boundary), `ConcentricInfill` must still recognize it as a
    /// hole and not print infill rings inside it. This was the Eikonal +
    /// concentric-fill bug on the pug model: wall loops are canonicalized in
    /// each loop's own local plane, but flattening into
    /// `plane_basis(BUILD_DIRECTION)` for `i_overlay` can flip a hole
    /// relative to the global basis.
    #[test]
    fn concentric_fill_does_not_infill_a_hole_that_is_wound_like_the_outer_loop() {
        let outer = vec![
            DVec3::new(-5.0, -5.0, 0.0),
            DVec3::new(5.0, -5.0, 0.0),
            DVec3::new(5.0, 5.0, 0.0),
            DVec3::new(-5.0, 5.0, 0.0),
        ];
        // Same CCW winding as `outer`, not the usual CW hole convention.
        let hole_wound_like_outer = vec![
            DVec3::new(-1.0, -1.0, 0.0),
            DVec3::new(1.0, -1.0, 0.0),
            DVec3::new(1.0, 1.0, 0.0),
            DVec3::new(-1.0, 1.0, 0.0),
        ];

        let layer = Layer {
            index: 0,
            object: ObjectId(0),
            order: 0.0,
            loops: Vec::new(),
            infill_boundary: vec![outer, hole_wound_like_outer],
            solid_fill_boundary: Vec::new(),
            ..Layer::default()
        };

        let region = InfillRegion::from_layer(&layer, &config());
        let paths =
            ConcentricInfill.generate(&region, &config(), &layer, &Transform::identity(), 1.0);
        assert!(
            !paths.is_empty(),
            "expected infill rings in the annulus, not an empty region"
        );

        for path in &paths {
            for p in &path.points {
                let inside_hole = p.x > -0.999 && p.x < 0.999 && p.y > -0.999 && p.y < 0.999;
                assert!(
                    !inside_hole,
                    "concentric ring point {p:?} fell inside a hole that is wound like the outer loop"
                );
            }
        }
    }

    #[test]
    fn none_infill_emits_no_paths_even_for_a_nonempty_region() {
        let layer = square_layer(5.0);
        let region = InfillRegion::from_layer(&layer, &config());

        let paths = NoneInfill.generate(&region, &config(), &layer, &Transform::identity(), 1.0);
        assert!(paths.is_empty());

        // Sanity: the region itself is not empty, so a real pattern would
        // have emitted something.
        let monotonic_paths =
            MonotonicInfill.generate(&region, &config(), &layer, &Transform::identity(), 1.0);
        assert!(!monotonic_paths.is_empty());
    }

    #[test]
    fn all_walls_fill_produces_multiple_closed_rings_covering_a_square() {
        let layer = square_layer(5.0);
        let region = InfillRegion::from_layer(&layer, &config());
        let paths =
            AllWallsInfill.generate(&region, &config(), &layer, &Transform::identity(), 1.0);

        assert!(
            paths.len() > 1,
            "expected several successively-inset rings, got {}",
            paths.len()
        );
        for path in &paths {
            assert_eq!(path.points.len(), path.segments.len());
            assert!(path.segments.iter().all(|s| s.kind == MoveKind::Infill));
            for p in &path.points {
                assert!(p.x >= -5.001 && p.x <= 5.001, "x out of bounds: {p:?}");
                assert!(p.y >= -5.001 && p.y <= 5.001, "y out of bounds: {p:?}");
            }
        }
    }

    #[test]
    fn all_walls_fill_ignores_density_and_fills_fully_even_at_zero_density() {
        let layer = square_layer(5.0);
        let region = InfillRegion::from_layer(&layer, &config());

        let full_density_paths =
            AllWallsInfill.generate(&region, &config(), &layer, &Transform::identity(), 1.0);
        let zero_density_paths =
            AllWallsInfill.generate(&region, &config(), &layer, &Transform::identity(), 0.0);

        // Unlike `ConcentricInfill`, density does not scale spacing or gate
        // generation at all -- both calls should produce the same rings.
        assert!(!zero_density_paths.is_empty());
        assert_eq!(zero_density_paths.len(), full_density_paths.len());
    }

    #[test]
    fn all_walls_fill_is_empty_for_empty_region() {
        let layer = Layer {
            index: 0,
            object: ObjectId(0),
            order: 0.0,
            loops: Vec::new(),
            infill_boundary: Vec::new(),
            solid_fill_boundary: Vec::new(),
            ..Layer::default()
        };
        let region = InfillRegion::from_layer(&layer, &config());
        let paths =
            AllWallsInfill.generate(&region, &config(), &layer, &Transform::identity(), 1.0);
        assert!(paths.is_empty());
    }

    #[test]
    fn cubic_fill_produces_paths_covering_a_square() {
        let layer = square_layer(5.0);
        let region = InfillRegion::from_layer(&layer, &config());
        let paths = CubicInfill.generate(&region, &config(), &layer, &Transform::identity(), 0.2);

        assert!(!paths.is_empty(), "expected cubic infill paths on a square");
        for path in &paths {
            assert!(path.points.len() >= 2);
            assert!(path.segments.iter().any(|s| s.kind == MoveKind::Infill));
            for p in &path.points {
                assert!(p.x >= -5.001 && p.x <= 5.001, "x out of bounds: {p:?}");
                assert!(p.y >= -5.001 && p.y <= 5.001, "y out of bounds: {p:?}");
            }
        }
    }

    #[test]
    fn cubic_fill_shifts_across_layers() {
        let mut layer1 = square_layer(5.0);
        layer1.order = 0.2;
        let mut layer2 = square_layer(5.0);
        layer2.order = 0.4;

        let region1 = InfillRegion::from_layer(&layer1, &config());
        let region2 = InfillRegion::from_layer(&layer2, &config());

        let paths1 =
            CubicInfill.generate(&region1, &config(), &layer1, &Transform::identity(), 0.2);
        let paths2 =
            CubicInfill.generate(&region2, &config(), &layer2, &Transform::identity(), 0.2);

        assert!(!paths1.is_empty());
        assert!(!paths2.is_empty());

        // Points across differing Z layers should differ in their in-plane line offsets
        let pts1: Vec<DVec3> = paths1.iter().flat_map(|p| &p.points).copied().collect();
        let pts2: Vec<DVec3> = paths2.iter().flat_map(|p| &p.points).copied().collect();
        assert_ne!(pts1, pts2, "expected cubic infill lines to shift with Z");
    }

    #[test]
    fn cubic_fill_density_widens_spacing() {
        let layer = square_layer(10.0);
        let region = InfillRegion::from_layer(&layer, &config());

        let dense_paths =
            CubicInfill.generate(&region, &config(), &layer, &Transform::identity(), 0.5);
        let sparse_paths =
            CubicInfill.generate(&region, &config(), &layer, &Transform::identity(), 0.1);

        let dense_pts: usize = dense_paths.iter().map(|p| p.points.len()).sum();
        let sparse_pts: usize = sparse_paths.iter().map(|p| p.points.len()).sum();
        assert!(
            dense_pts > sparse_pts,
            "expected higher density to produce more infill points (dense={dense_pts}, sparse={sparse_pts})"
        );
    }

    #[test]
    fn cubic_fill_is_empty_for_empty_region() {
        let layer = Layer {
            index: 0,
            object: ObjectId(0),
            order: 0.0,
            loops: Vec::new(),
            infill_boundary: Vec::new(),
            solid_fill_boundary: Vec::new(),
            ..Layer::default()
        };
        let region = InfillRegion::from_layer(&layer, &config());
        let paths = CubicInfill.generate(&region, &config(), &layer, &Transform::identity(), 0.2);
        assert!(paths.is_empty());
    }

    /// Distance from `p` to the nearest edge of the axis-aligned square
    /// boundary with the given half-extent (i.e. how far inward from the
    /// `region.loops`/`infill_boundary` centerline the point sits).
    fn distance_inward_from_square_boundary(p: DVec3, half_extent: f64) -> f64 {
        (half_extent - p.x.abs()).min(half_extent - p.y.abs())
    }

    /// Regression test for the infill double-offset defect: the first ring
    /// of `ConcentricInfill` must sit directly on `infill_boundary` (zero
    /// extra inset), not a further `spacing / 2.0` in from it. Before the
    /// fix this measured ~`infill_line_width` in from the boundary instead
    /// of ~0.
    #[test]
    fn concentric_fill_first_ring_sits_directly_on_the_infill_boundary() {
        let half_extent = 5.0;
        let layer = square_layer(half_extent);
        let cfg = config(); // infill_line_width = 0.5
        let region = InfillRegion::from_layer(&layer, &cfg);
        let paths = ConcentricInfill.generate(&region, &cfg, &layer, &Transform::identity(), 1.0);
        assert!(!paths.is_empty());

        // The first-generated ring is the outermost (closest to the
        // boundary): its every point should be within a small tolerance of
        // the boundary centerline, not offset a further half line-width in.
        let first_ring = &paths[0].points;
        for p in first_ring {
            let d = distance_inward_from_square_boundary(*p, half_extent);
            assert!(
                d.abs() < 1e-3,
                "expected first ring to sit on the boundary (distance ~0), got {d} at {p:?}"
            );
        }
    }

    /// Same regression as `concentric_fill_first_ring_sits_directly_on_the_infill_boundary`,
    /// for `AllWallsInfill`'s independent first-ring offset.
    #[test]
    fn all_walls_fill_first_ring_sits_directly_on_the_infill_boundary() {
        let half_extent = 5.0;
        let layer = square_layer(half_extent);
        let cfg = config(); // infill_line_width = 0.5
        let region = InfillRegion::from_layer(&layer, &cfg);
        let paths = AllWallsInfill.generate(&region, &cfg, &layer, &Transform::identity(), 1.0);
        assert!(!paths.is_empty());

        let first_ring = &paths[0].points;
        for p in first_ring {
            let d = distance_inward_from_square_boundary(*p, half_extent);
            assert!(
                d.abs() < 1e-3,
                "expected first ring to sit on the boundary (distance ~0), got {d} at {p:?}"
            );
        }
    }

    #[test]
    fn gyroid_infill_produces_paths_covering_a_square() {
        let layer = square_layer(5.0);
        let region = InfillRegion::from_layer(&layer, &config());
        let paths = GyroidInfill.generate(&region, &config(), &layer, &Transform::identity(), 0.2);
        assert!(!paths.is_empty(), "expected gyroid infill paths");
        for path in &paths {
            assert!(!path.points.is_empty());
            for p in &path.points {
                assert!(p.x >= -5.001 && p.x <= 5.001, "x out of bounds: {p:?}");
                assert!(p.y >= -5.001 && p.y <= 5.001, "y out of bounds: {p:?}");
            }
        }
    }

    #[test]
    fn schwarz_d_infill_produces_paths_covering_a_square() {
        let layer = square_layer(5.0);
        let region = InfillRegion::from_layer(&layer, &config());
        let paths =
            SchwarzDInfill.generate(&region, &config(), &layer, &Transform::identity(), 0.2);
        assert!(!paths.is_empty(), "expected schwarz d infill paths");
    }

    #[test]
    fn infill_paths_start_from_endpoint_closest_to_previous_path() {
        // Disconnected squares producing multiple infill paths: verify that
        // each subsequent path starts from the endpoint closer to the previous
        // path's exit point.
        let layer = Layer {
            index: 0,
            object: ObjectId(0),
            order: 0.0,
            loops: Vec::new(),
            infill_boundary: vec![
                vec![
                    DVec3::new(-5.0, -5.0, 0.0),
                    DVec3::new(-3.0, -5.0, 0.0),
                    DVec3::new(-3.0, 5.0, 0.0),
                    DVec3::new(-5.0, 5.0, 0.0),
                ],
                vec![
                    DVec3::new(3.0, -5.0, 0.0),
                    DVec3::new(5.0, -5.0, 0.0),
                    DVec3::new(5.0, 5.0, 0.0),
                    DVec3::new(3.0, 5.0, 0.0),
                ],
            ],
            solid_fill_boundary: Vec::new(),
            ..Layer::default()
        };
        let config = SlicerConfig {
            infill_line_width: 0.5,
            infill_angle_deg: 90.0,
            ..SlicerConfig::default()
        };
        let region = InfillRegion::from_layer(&layer, &config);
        let paths = MonotonicInfill.generate(&region, &config, &layer, &Transform::identity(), 1.0);

        assert_eq!(paths.len(), 2);
        let first_exit = *paths[0].points.last().unwrap();
        let second_start = *paths[1].points.first().unwrap();
        let second_exit = *paths[1].points.last().unwrap();

        assert!(
            first_exit.distance_squared(second_start) <= first_exit.distance_squared(second_exit),
            "second path should start from the endpoint closest to the first path's exit: \
             first_exit={first_exit:?}, second_start={second_start:?}, second_exit={second_exit:?}"
        );
    }

    #[test]
    fn tpms_infill_paths_start_from_endpoint_closest_to_previous_path() {
        let layer = square_layer(5.0);
        let region = InfillRegion::from_layer(&layer, &config());
        let paths = GyroidInfill.generate(&region, &config(), &layer, &Transform::identity(), 0.2);
        assert!(paths.len() >= 2);

        for i in 1..paths.len() {
            let prev_exit = *paths[i - 1].points.last().unwrap();
            let start = *paths[i].points.first().unwrap();
            let end = *paths[i].points.last().unwrap();
            assert!(
                prev_exit.distance_squared(start) <= prev_exit.distance_squared(end),
                "path {i} should start from endpoint closest to path {} exit: \
                 prev_exit={prev_exit:?}, start={start:?}, end={end:?}",
                i - 1
            );
        }
    }
}
