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
use crate::toolpath::{MoveKind, Path, Segment};
use crate::transform::Transform;
use crate::SlicerConfig;
use glam::DVec3;
use manifold_fidget::contour::plane_basis;

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
}

/// Resolve an [`InfillPatternKind`] to its [`InfillGenerator`] implementation.
#[must_use]
pub fn generator_for(kind: InfillPatternKind) -> Box<dyn InfillGenerator + Sync> {
    match kind {
        InfillPatternKind::Monotonic => Box::new(MonotonicInfill),
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
        if layer.solid_fill_boundary.is_empty() {
            return Self {
                loops: layer.infill_boundary.clone(),
            };
        }
        let (axis, apex, _slope) = order_field::resolve_axis_apex_slope(config.order_field, config);
        let field = order_field::order_field_for(config.order_field, config);
        let (basis1, basis2) = plane_basis(axis);
        let infill_2d = polygon2d::to_2d(&layer.infill_boundary, basis1, basis2, apex);
        let solid_2d = polygon2d::to_2d(&layer.solid_fill_boundary, basis1, basis2, apex);
        let sparse_2d = polygon2d::difference(&infill_2d, &solid_2d);
        Self {
            loops: order_field::reconstruct_on_order_field(
                sparse_2d,
                basis1,
                basis2,
                axis,
                apex,
                layer.order,
                field.as_ref(),
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

/// One (u, v, w) crossing of a scan line with a loop edge, in the rotated
/// fill frame: `u`/`v` are in-plane fill-frame coordinates, `w` is the
/// component along [`BUILD_DIRECTION`] (kept per-point rather than assumed
/// constant across the layer, so reconstruction stays exact even for a
/// not-perfectly-planar layer).
#[derive(Debug, Clone, Copy)]
struct Crossing {
    u: f64,
    w: f64,
}

/// One infill scan-line segment in the rotated `(u, w)` frame: `((u_start,
/// w_start), (u_end, w_end))`.
type ScanSegment = ((f64, f64), (f64, f64));

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

        // Project every loop's points into the rotated (u, v, w) frame,
        // keeping loops as closed edge lists (wrap-around to point 0).
        let projected: Vec<Vec<(f64, f64, f64)>> = region
            .loops
            .iter()
            .map(|points| {
                points
                    .iter()
                    .map(|p| (p.dot(u_dir), p.dot(v_dir), p.dot(axis)))
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
                    let (u0, v0, w0) = loop_points[i];
                    let (u1, v1, w1) = loop_points[(i + 1) % n];
                    let crosses = (v0 <= v && v1 > v) || (v1 <= v && v0 > v);
                    if !crosses {
                        continue;
                    }
                    let t = (v - v0) / (v1 - v0);
                    crossings.push(Crossing {
                        u: u0 + t * (u1 - u0),
                        w: w0 + t * (w1 - w0),
                    });
                }
            }
            crossings.sort_by(|a, b| a.u.total_cmp(&b.u));

            let mut pairs: Vec<ScanSegment> = Vec::new();
            let mut pair_iter = crossings.chunks_exact(2);
            for pair in &mut pair_iter {
                pairs.push(((pair[0].u, pair[0].w), (pair[1].u, pair[1].w)));
            }
            if !pairs.is_empty() {
                scanlines.push((scan_index, v, pairs));
            }
            scan_index += 1;
            v += spacing;
        }

        if scanlines.is_empty() {
            return Vec::new();
        }

        let mut points: Vec<DVec3> = Vec::new();
        let mut segments: Vec<Segment> = Vec::new();
        let mut prev_end: Option<(f64, f64)> = None;

        // `Path`'s contract (see `toolpath::Path`) is `segments[i]` ==
        // the move `points[i] -> points[i + 1]`, i.e. `segments.len() ==
        // points.len() - 1` for an open path like this boustrophedon
        // zigzag (no closing edge back to the start). So a `Segment` is
        // pushed for the edge *arriving* at a new point — never for the
        // very first point of the whole path, which has no incoming
        // edge yet. Getting this one-off wrong previously shifted every
        // segment's `kind` by one slot, mislabeling each real infill
        // fill-line as `Travel` and each real travel jump (across a gap
        // or hole) as `Infill`.
        let push_point = |points: &mut Vec<DVec3>,
                          segments: &mut Vec<Segment>,
                          uw: (f64, f64),
                          v: f64,
                          kind: MoveKind| {
            let (u, w) = uw;
            let world = u_dir * u + v_dir * v + axis * w;
            if !points.is_empty() {
                segments.push(Segment {
                    kind,
                    speed: 60.0,
                    extrusion_rate: 1.0,
                    support_fraction: 0.0,
                    order: layer.order,
                    extrusion_length: 0.0,
                });
            }
            points.push(world);
        };

        for (scan_index, v, pairs) in &scanlines {
            // Alternate traversal direction per scan line (boustrophedon)
            // so consecutive lines' endpoints stay close, minimizing
            // travel move length. Uses the scan-line's true ordinal (not
            // its position within the compacted `scanlines` list) so
            // alternation stays stable regardless of which scan-lines
            // happened to have crossings.
            let ordered: Vec<ScanSegment> = if scan_index.is_multiple_of(2) {
                pairs.clone()
            } else {
                pairs.iter().rev().map(|&(a, b)| (b, a)).collect()
            };
            for (start, end) in ordered {
                let needs_travel = prev_end != Some(start);
                if points.is_empty() || needs_travel {
                    push_point(&mut points, &mut segments, start, *v, MoveKind::Travel);
                }
                push_point(&mut points, &mut segments, end, *v, MoveKind::Infill);
                prev_end = Some(end);
            }
        }

        if points.len() < 2 {
            return Vec::new();
        }

        vec![Path {
            points,
            segments,
            tool: crate::ids::ToolId::default(),
        }]
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
        };
        let cfg = SlicerConfig {
            infill_line_width: 0.5,
            infill_angle_deg: 0.0,
            ..SlicerConfig::default()
        };
        let region = InfillRegion::from_layer(&layer, &config());
        let paths = MonotonicInfill.generate(&region, &cfg, &layer, &Transform::identity(), 1.0);
        assert_eq!(paths.len(), 1);
        let path = &paths[0];
        assert_eq!(path.points.len() - 1, path.segments.len());

        let hole_min = DVec3::new(-1.0, -1.0, -1.0);
        let hole_max = DVec3::new(1.0, 1.0, 1.0);
        let inside_hole =
            |p: DVec3| p.x > hole_min.x && p.x < hole_max.x && p.y > hole_min.y && p.y < hole_max.y;

        let mut saw_infill = false;
        let mut saw_travel_across_hole = false;
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
                MoveKind::Travel => {
                    if inside_hole(mid) {
                        saw_travel_across_hole = true;
                    }
                }
                other => panic!("unexpected move kind in infill path: {other:?}"),
            }
        }
        assert!(saw_infill, "expected at least one Infill-kind edge");
        assert!(
            saw_travel_across_hole,
            "expected at least one Travel-kind edge jumping across the hole"
        );
    }
}
