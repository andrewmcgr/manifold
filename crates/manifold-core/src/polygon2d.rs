//! 2D polygon helpers for slicer contour processing.
//!
//! Wraps the `i_overlay` crate (boolean ops + polygon offsetting) with
//! `manifold-core`'s own `glam::DVec3` in-plane convention. Each per-layer
//! cross-section is a set of closed loops lying in a plane defined by an
//! `origin` and an orthonormal `(basis1, basis2)` pair, matching the
//! convention `manifold_fidget::contour::plane_basis`/`extract_contours`
//! already use: a 3D point `p` in the plane maps to 2D coordinates
//! `(u, v) = ((p - origin).dot(basis1), (p - origin).dot(basis2))`, and back
//! via `p = origin + u * basis1 + v * basis2`.
//!
//! `i_overlay`'s `[f64; 2]` point type is an internal implementation detail
//! of this module only; nothing outside `polygon2d` should depend on it.

use glam::DVec3;
use i_overlay::core::fill_rule::FillRule;
use i_overlay::core::overlay_rule::OverlayRule;
use i_overlay::float::single::SingleFloatOverlay;
use i_overlay::mesh::outline::offset::OutlineOffset;
use i_overlay::mesh::style::OutlineStyle;

/// The fill rule used for every boolean/offset operation in this module.
///
/// `NonZero` matches the winding convention produced by marching-squares
/// contour extraction (single outer CCW loop + CW holes, no self-overlap),
/// and is what `i_overlay`'s own offsetting/simplify docs assume.
const FILL_RULE: FillRule = FillRule::NonZero;

/// Absolute-area threshold (in the caller's 2D plane units, mm² for this
/// codebase) below which a loop is treated as degenerate noise and dropped
/// by [`canonicalize`] rather than passed to `i_overlay`. Chosen far above
/// f64 rounding noise (observed spurious loops from curved-order-field
/// contour extraction measure ~1e-11 or smaller) and far below any real
/// printable feature (smallest sane feature is on the order of a nozzle
/// diameter squared, i.e. >= ~0.01 for a 0.1mm nozzle).
const DEGENERATE_AREA_EPSILON: f64 = 1e-6;

/// Distance threshold (in mm) used by [`simplify_collinear`] to collapse
/// vertices whose perpendicular deviation from the line through their
/// neighbors is below this tolerance. 1 micron (1e-3 mm) is far below any
/// printable feature (typical nozzle diameters 0.1–0.4mm, layer heights
/// 0.1–0.3mm) so this cannot visibly change printed geometry, but far
/// above the point spacing seen in pathological input.
///
/// Curved-order-field contour extraction (e.g. Eikonal) can emit
/// thousands of points along what is geometrically a straight or gently
/// curved run — a "staircase" of near-collinear points from a grid-aligned
/// marching front, observed with median inter-point spacing on the order
/// of nanometers to low microns. These aren't exact duplicates (so a
/// simple distance-based dedup barely reduces them) but they are
/// enormously redundant — a single real loop with a few thousand such
/// points has been observed to turn a sub-second `i_overlay` boolean/offset
/// call into one lasting tens of seconds to several minutes, because the
/// segment-intersection solver's cost scales with point/segment count, not
/// with the shape's actual geometric complexity.
const COLLINEAR_EPSILON: f64 = 1e-3;

/// Projects a set of 3D loops lying in the plane `(origin, basis1, basis2)`
/// down to 2D coordinates in that plane's basis.
///
/// `basis1`/`basis2` are assumed orthonormal (as produced by
/// `manifold_fidget::contour::plane_basis`); this function does not
/// normalize or orthogonalize them itself.
pub fn to_2d(
    loops: &[Vec<DVec3>],
    basis1: DVec3,
    basis2: DVec3,
    origin: DVec3,
) -> Vec<Vec<[f64; 2]>> {
    loops
        .iter()
        .map(|loop_| {
            loop_
                .iter()
                .map(|p| {
                    let d = *p - origin;
                    [d.dot(basis1), d.dot(basis2)]
                })
                .collect()
        })
        .collect()
}

/// Inverse of [`to_2d`]: reconstructs 3D loops in the plane
/// `(origin, basis1, basis2)` from 2D `(u, v)` coordinates in that basis.
pub fn from_2d(
    contours: Vec<Vec<[f64; 2]>>,
    basis1: DVec3,
    basis2: DVec3,
    origin: DVec3,
) -> Vec<Vec<DVec3>> {
    contours
        .into_iter()
        .map(|contour| {
            contour
                .into_iter()
                .map(|[u, v]| origin + basis1 * u + basis2 * v)
                .collect()
        })
        .collect()
}

/// Perpendicular distance from `pt` to the (infinite) line through `a` and
/// `c`. Falls back to plain point distance if `a` and `c` coincide.
fn perpendicular_distance(pt: [f64; 2], a: [f64; 2], c: [f64; 2]) -> f64 {
    let d = [c[0] - a[0], c[1] - a[1]];
    let len = (d[0] * d[0] + d[1] * d[1]).sqrt();
    if len < 1e-15 {
        return dist_sq(pt, a).sqrt();
    }
    let cross = (pt[0] - a[0]) * d[1] - (pt[1] - a[1]) * d[0];
    (cross / len).abs()
}

/// Collapses runs of near-collinear (or near-coincident) consecutive
/// vertices in a closed loop, keeping only the vertices needed to
/// represent its shape within [`COLLINEAR_EPSILON`]. This is a single
/// forward pass (extend-the-line-while-collinear), not a full
/// Douglas-Peucker recursion, followed by a few passes to clean up
/// collinearity across the loop's start/end seam (since the loop is
/// implicitly closed). A loop that collapses to fewer than 3 points is
/// left as-is; the degenerate-loop filter in [`canonicalize`] handles
/// dropping loops that carry no real area.
fn simplify_collinear(loop_: &[[f64; 2]]) -> Vec<[f64; 2]> {
    if loop_.len() < 3 {
        return loop_.to_vec();
    }

    let mut kept: Vec<[f64; 2]> = Vec::with_capacity(loop_.len());
    for &p in loop_ {
        while kept.len() >= 2 {
            let a = kept[kept.len() - 2];
            let b = kept[kept.len() - 1];
            if perpendicular_distance(b, a, p) < COLLINEAR_EPSILON {
                kept.pop();
            } else {
                break;
            }
        }
        kept.push(p);
    }

    // Clean up collinearity across the closing seam (last->first->second and
    // second-to-last->last->first), since the forward pass above only sees
    // the loop as an open polyline.
    for _ in 0..4 {
        let mut changed = false;
        if kept.len() >= 3 {
            let a = kept[kept.len() - 2];
            let b = kept[kept.len() - 1];
            let p = kept[0];
            if perpendicular_distance(b, a, p) < COLLINEAR_EPSILON {
                kept.pop();
                changed = true;
            }
        }
        if kept.len() >= 3 {
            let a = kept[kept.len() - 1];
            let b = kept[0];
            let p = kept[1];
            if perpendicular_distance(b, a, p) < COLLINEAR_EPSILON {
                kept.remove(0);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    if kept.len() >= 3 {
        kept
    } else {
        loop_.to_vec()
    }
}

fn dist_sq(a: [f64; 2], b: [f64; 2]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    dx * dx + dy * dy
}

/// Normalizes a set of loops (fixes winding, removes self-intersections and
/// degenerate/duplicate vertices) before feeding them into a boolean or
/// offset operation, per `i_overlay`'s recommendation for raw
/// marching-squares-style contour input.
fn simplify(loops: &[Vec<[f64; 2]>]) -> Vec<Vec<[f64; 2]>> {
    use i_overlay::float::simplify::SimplifyShape;

    let simplified: Vec<Vec<[f64; 2]>> = loops.iter().map(|l| simplify_collinear(l)).collect();

    simplified
        .simplify_shape(FILL_RULE)
        .into_iter()
        .flatten()
        .collect()
}

/// Reorients a set of 2D loops so nesting parity determines winding: a loop
/// at even containment depth (not inside any other loop, or inside an even
/// number of them) winds counter-clockwise, and a loop at odd depth (a hole)
/// winds clockwise.
///
/// This matches the `NonZero` fill rule that `i_overlay` expects. Raw loops
/// from a curved order field are canonicalized in each loop's own local
/// best-fit plane by `manifold_fidget::contour`, but once they are flattened
/// into a *different* (global) basis for `i_overlay` boolean/offset
/// operations, their winding in that basis may be inverted. Passing such
/// loops to `i_overlay` without re-canonicalizing can cause holes to be
/// treated as nested solids — concentric infill then prints rings inside the
/// holes.
///
/// Loops with an absolute area below [`DEGENERATE_AREA_EPSILON`] are dropped
/// entirely rather than preserved. Curved-order-field contour extraction can
/// occasionally emit hundreds of these alongside the real boundary loops —
/// float-noise artifacts (near-collinear/near-duplicate points from a grid-
/// aligned marching front) with no meaningful area, several orders of
/// magnitude smaller than any real printable feature. Left in, they don't
/// change boolean/offset *results* (zero area contributes nothing under
/// `NonZero`), but a boundary carrying hundreds of them is adversarial to
/// `i_overlay`'s segment-intersection solver — this has been observed to
/// turn a sub-second offset into a many-minute one. A degenerate loop with
/// fewer than 3 points is also dropped for the same reason (it contributes
/// no area either).
pub fn canonicalize(loops2d: &[Vec<[f64; 2]>]) -> Vec<Vec<[f64; 2]>> {
    let loops2d: Vec<Vec<[f64; 2]>> = loops2d
        .iter()
        .filter(|loop_| loop_.len() >= 3 && signed_area(loop_).abs() > DEGENERATE_AREA_EPSILON)
        .cloned()
        .collect();

    if loops2d.len() <= 1 {
        return loops2d;
    }

    let depths: Vec<usize> = loops2d
        .iter()
        .enumerate()
        .map(|(i, loop_)| {
            let test = loop_[0];
            (0..loops2d.len())
                .filter(|&j| j != i && point_in_polygon(test, &loops2d[j]))
                .count()
        })
        .collect();

    loops2d
        .iter()
        .zip(depths)
        .map(|(loop_, depth)| {
            let want_ccw = depth % 2 == 0;
            let is_ccw = signed_area(loop_) > 0.0;
            if want_ccw == is_ccw {
                loop_.clone()
            } else {
                loop_.iter().copied().rev().collect()
            }
        })
        .collect()
}

/// Signed shoelace area of a closed 2D loop; positive for counter-clockwise
/// winding and negative for clockwise. Fewer than 3 points yields `0.0`.
fn signed_area(loop_: &[[f64; 2]]) -> f64 {
    if loop_.len() < 3 {
        return 0.0;
    }
    let mut area = 0.0;
    for i in 0..loop_.len() {
        let [x0, y0] = loop_[i];
        let [x1, y1] = loop_[(i + 1) % loop_.len()];
        area += x0 * y1 - x1 * y0;
    }
    area * 0.5
}

/// Standard even-odd point-in-polygon ray cast. `loop_` is treated as closed.
fn point_in_polygon(point: [f64; 2], loop_: &[[f64; 2]]) -> bool {
    if loop_.len() < 3 {
        return false;
    }
    let [x, y] = point;
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

/// Insets `loops2d` inward by `distance` (a positive value moves boundaries
/// toward the interior of the shape, shrinking it — as used for wall-inset
/// infill boundaries). `loops2d` is simplified first.
///
/// Use this for input that hasn't already been through an `i_overlay`
/// boolean/offset operation (e.g. a boundary freshly reconstructed from
/// mesh/order-field data, which can carry self-intersections). For a
/// *repeated*-offset loop (successive rings inward, as in
/// `ConcentricInfill`/`AllWallsInfill`), only the first ring needs this —
/// see [`inward_offset_unchecked`] for the rest.
pub fn inward_offset(loops2d: &[Vec<[f64; 2]>], distance: f64) -> Vec<Vec<[f64; 2]>> {
    let simplified = simplify(loops2d);
    let style = OutlineStyle::new(-distance.abs());
    simplified.outline(&style).into_iter().flatten().collect()
}

/// Same as [`inward_offset`], but skips the input pre-simplify pass.
///
/// `i_overlay`'s offsetting (`OutlineOffset::outline`) already cleans its
/// own output (de-spiking/simplifying collinear points) before returning
/// it, so feeding that output straight back into another `outline()` call
/// is already valid, correctly-wound input — re-running the much heavier
/// whole-shape `simplify_shape` boolean pass on it again is pure redundant
/// work. Intended for the *second and later* offset in a repeated-offset
/// loop, where `loops2d` is guaranteed to be the direct output of a prior
/// [`inward_offset`]/`inward_offset_unchecked` call rather than raw
/// reconstructed geometry.
///
/// Still applies the cheap [`simplify_collinear`] pass per-loop (a linear
/// scan, unlike `simplify_shape`'s full boolean pass): `i_overlay`'s own
/// output cleanup does not guarantee no near-collinear runs of points
/// remain on pathological input, and even a few thousand surviving into a
/// later ring can blow up that ring's own `outline()` call the same way
/// raw reconstructed geometry can (see [`COLLINEAR_EPSILON`]).
pub fn inward_offset_unchecked(loops2d: &[Vec<[f64; 2]>], distance: f64) -> Vec<Vec<[f64; 2]>> {
    let simplified: Vec<Vec<[f64; 2]>> = loops2d.iter().map(|l| simplify_collinear(l)).collect();
    let style = OutlineStyle::new(-distance.abs());
    simplified.outline(&style).into_iter().flatten().collect()
}

/// Subtracts `clip` from `subj` (`subj - clip`). Both inputs are simplified
/// first.
pub fn difference(subj: &[Vec<[f64; 2]>], clip: &[Vec<[f64; 2]>]) -> Vec<Vec<[f64; 2]>> {
    let subj = simplify(subj);
    let clip = simplify(clip);
    subj.overlay(&clip, OverlayRule::Difference, FILL_RULE)
        .into_iter()
        .flatten()
        .collect()
}

/// Unions all `regions` together into a single set of loops.
pub fn union(regions: &[Vec<Vec<[f64; 2]>>]) -> Vec<Vec<[f64; 2]>> {
    let mut regions = regions.iter();
    let Some(first) = regions.next() else {
        return Vec::new();
    };
    let mut acc = simplify(first);
    for region in regions {
        let region = simplify(region);
        acc = acc
            .overlay(&region, OverlayRule::Union, FILL_RULE)
            .into_iter()
            .flatten()
            .collect();
    }
    acc
}

/// Intersects `subj` with `clip`. Both inputs are simplified first.
pub fn intersection(subj: &[Vec<[f64; 2]>], clip: &[Vec<[f64; 2]>]) -> Vec<Vec<[f64; 2]>> {
    let subj = simplify(subj);
    let clip = simplify(clip);
    subj.overlay(&clip, OverlayRule::Intersect, FILL_RULE)
        .into_iter()
        .flatten()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square(x0: f64, y0: f64, side: f64) -> Vec<[f64; 2]> {
        vec![
            [x0, y0],
            [x0, y0 + side],
            [x0 + side, y0 + side],
            [x0 + side, y0],
        ]
    }

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-6
    }

    /// Signed shoelace area of a single loop (positive for CCW, negative for CW).
    fn signed_area(loop_: &[[f64; 2]]) -> f64 {
        let mut area = 0.0;
        for i in 0..loop_.len() {
            let [x0, y0] = loop_[i];
            let [x1, y1] = loop_[(i + 1) % loop_.len()];
            area += x0 * y1 - x1 * y0;
        }
        area * 0.5
    }

    /// Total area of a shape (outer boundary minus holes), relying on
    /// `i_overlay`'s convention that holes have the opposite winding to the
    /// outer boundary (so summing signed areas across all loops in a shape
    /// naturally subtracts hole area).
    fn total_area(loops: &[Vec<[f64; 2]>]) -> f64 {
        loops
            .iter()
            .map(|loop_| signed_area(loop_))
            .sum::<f64>()
            .abs()
    }

    #[test]
    fn to_2d_from_2d_round_trip_axis_aligned() {
        let basis1 = DVec3::X;
        let basis2 = DVec3::Y;
        let origin = DVec3::ZERO;
        let loop_3d = vec![
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(1.0, 1.0, 0.0),
            DVec3::new(0.0, 1.0, 0.0),
        ];
        let loops = vec![loop_3d.clone()];
        let flat = to_2d(&loops, basis1, basis2, origin);
        let back = from_2d(flat, basis1, basis2, origin);
        assert_eq!(back.len(), 1);
        for (a, b) in loop_3d.iter().zip(back[0].iter()) {
            assert!(a.abs_diff_eq(*b, 1e-9), "expected {a:?}, got {b:?}");
        }
    }

    #[test]
    fn to_2d_from_2d_round_trip_non_trivial_plane() {
        // A plane through a non-origin point, with a non-axis-aligned
        // orthonormal basis (derived the same way plane_basis does: pick a
        // direction, cross with a seed vector, cross again).
        let direction = DVec3::new(1.0, 1.0, 1.0).normalize();
        let seed = DVec3::X;
        let basis1 = direction.cross(seed).normalize();
        let basis2 = direction.cross(basis1).normalize();
        let origin = DVec3::new(3.0, -2.0, 5.0);

        // Build a loop that actually lies in this plane: origin plus
        // offsets purely along basis1/basis2.
        let loop_3d: Vec<DVec3> = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]
            .iter()
            .map(|(u, v)| origin + basis1 * *u + basis2 * *v)
            .collect();

        let loops = vec![loop_3d.clone()];
        let flat = to_2d(&loops, basis1, basis2, origin);
        let back = from_2d(flat, basis1, basis2, origin);
        assert_eq!(back.len(), 1);
        for (a, b) in loop_3d.iter().zip(back[0].iter()) {
            assert!(a.abs_diff_eq(*b, 1e-9), "expected {a:?}, got {b:?}");
        }
    }

    #[test]
    fn inward_offset_shrinks_square() {
        let sq = vec![square(0.0, 0.0, 10.0)];
        let offset = inward_offset(&sq, 1.0);
        assert!(!offset.is_empty());
        // Inward offset of a 10x10 square by 1 should yield (roughly) an
        // 8x8 square -> area 64, versus original area 100.
        let area = total_area(&offset);
        assert!(area < 100.0);
        assert!(approx_eq(area, 64.0) || (area - 64.0).abs() < 1.0);
    }

    #[test]
    fn difference_removes_overlap() {
        let subj = vec![square(0.0, 0.0, 10.0)];
        let clip = vec![square(5.0, 0.0, 10.0)];
        let result = difference(&subj, &clip);
        let area = total_area(&result);
        // subj area 100, overlap is 5x10=50, remainder should be ~50.
        assert!(approx_eq(area, 50.0));
    }

    #[test]
    fn union_combines_disjoint_and_overlapping() {
        let a = square(0.0, 0.0, 10.0);
        let b = square(5.0, 0.0, 10.0);
        let result = union(&[vec![a], vec![b]]);
        let area = total_area(&result);
        // 100 + 100 - 50 overlap = 150.
        assert!(approx_eq(area, 150.0));
    }

    #[test]
    fn intersection_of_overlapping_squares() {
        let a = vec![square(0.0, 0.0, 10.0)];
        let b = vec![square(5.0, 0.0, 10.0)];
        let result = intersection(&a, &b);
        let area = total_area(&result);
        assert!(approx_eq(area, 50.0));
    }

    #[test]
    fn l_shape_boolean_ops() {
        // L-shape: big square minus a corner square (represented directly
        // as an outer loop + inner hole-like contour to feed the boolean
        // op as a difference, since our contour convention is
        // outer-CCW/hole-CW).
        let big = square(0.0, 0.0, 10.0);
        let corner = square(5.0, 5.0, 5.0);
        let l_shape = difference(&[big], &[corner]);
        let area = total_area(&l_shape);
        // 100 - 25 = 75.
        assert!(approx_eq(area, 75.0));

        // Offsetting the L-shape inward should still produce a smaller,
        // non-empty shape.
        let offset = inward_offset(&l_shape, 0.5);
        assert!(!offset.is_empty());
        assert!(total_area(&offset) < area);
    }

    #[test]
    fn square_with_hole_area() {
        // Outer boundary CCW, hole CW (matches i_overlay's convention).
        let outer = square(0.0, 0.0, 10.0);
        let mut hole = square(3.0, 3.0, 2.0);
        hole.reverse();
        let shape = vec![outer, hole];
        let simplified = simplify(&shape);
        let area = total_area(&simplified);
        // 100 - 4 = 96.
        assert!(approx_eq(area, 96.0));
    }

    #[test]
    fn repeated_inward_offset_of_an_annulus_shrinks_the_ring_not_just_the_hole() {
        // Outer boundary CCW, hole CW (matches i_overlay's convention) --
        // mirrors ConcentricInfill::generate's repeated-offset loop on a
        // donut shape (e.g. material surrounding a through-hole).
        let outer = square(0.0, 0.0, 20.0);
        let mut hole = square(8.0, 8.0, 4.0);
        hole.reverse();
        let shape = vec![outer, hole];

        let mut current = inward_offset(&shape, 0.5);
        let mut ring_bboxes = Vec::new();
        let mut steps = 0;
        while !current.is_empty() && steps < 20 {
            // Bounding box across every loop in this ring generation.
            let mut min = [f64::INFINITY; 2];
            let mut max = [f64::NEG_INFINITY; 2];
            for loop_ in &current {
                for p in loop_ {
                    min[0] = min[0].min(p[0]);
                    min[1] = min[1].min(p[1]);
                    max[0] = max[0].max(p[0]);
                    max[1] = max[1].max(p[1]);
                }
            }
            ring_bboxes.push((min, max));
            current = inward_offset(&current, 1.0);
            steps += 1;
        }

        // If the offset were wrongly collapsing toward the hole's own
        // center (treating the hole loop as if it were a small solid
        // outer boundary) every successive ring's bbox would shrink toward
        // the hole's location (roughly centered at (10,10) within a
        // ~4-unit box) instead of the whole annulus's outer footprint
        // (roughly centered at (10,10) within a ~20-unit box, thinning
        // from BOTH the outer edge and the hole edge). Assert the first
        // ring's bbox still spans close to the *outer* square's extent,
        // not the tiny hole's extent.
        let (first_min, first_max) = ring_bboxes[0];
        let span_x = first_max[0] - first_min[0];
        let span_y = first_max[1] - first_min[1];
        assert!(
            span_x > 15.0 && span_y > 15.0,
            "first ring bbox span ({span_x}, {span_y}) should track the ~20-unit outer \
             boundary, not collapse toward the ~4-unit hole"
        );

        // And the ring sequence should actually terminate (annulus fully
        // consumed) well before the material could plausibly still exist,
        // not run away.
        assert!(
            steps < 20,
            "ring offsetting should terminate for a bounded annulus"
        );
    }

    #[test]
    fn canonicalize_makes_a_same_wound_loop_into_a_hole() {
        let outer = square(0.0, 0.0, 10.0);
        let same_wound_hole = square(3.0, 3.0, 2.0);
        // Deliberately *not* reversed: the hole has the same winding as the outer,
        // which can happen after flattening 3D loops into a global 2D basis.
        let shape = vec![outer, same_wound_hole];
        let canonical = canonicalize(&shape);
        assert_eq!(canonical.len(), 2);

        // Outer should stay CCW (positive signed area).
        assert!(signed_area(&canonical[0]) > 0.0);
        // Inner should be reversed to CW (negative signed area) so
        // downstream offset/boolean ops treat it as a hole rather than a
        // nested solid.
        assert!(signed_area(&canonical[1]) < 0.0);
    }
}
