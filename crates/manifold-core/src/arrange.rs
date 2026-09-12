//! Auto-arrangement of objects on the print bed.
//!
//! Greedy spiral placement: objects are placed tallest-first outward along
//! an expanding Archimedean spiral centered on the bed, each accepted at the
//! first candidate position whose clearance-inflated convex footprint
//! doesn't overlap any already-placed object and fits within the bed
//! bounds. This is not a globally optimal packing (that's bin-packing-hard
//! and not worth the cost here), but it's cheap — O(n * spiral_samples) —
//! and naturally arranges the tallest, most failure-prone objects nearest
//! the bed center, where thermal/adhesion conditions are typically best.

use glam::DVec2;

use crate::polygon2d;

/// One object to place: its convex footprint (in the object's own current
/// XY position) and its build height, which determines placement priority
/// (tallest first, placed closest to the bed center).
#[derive(Debug, Clone)]
pub struct ArrangeItem {
    pub footprint: Vec<DVec2>,
    pub height: f64,
}

/// Resolved arrangement: a per-item XY translation to apply to move that
/// item's footprint to its new, non-overlapping position, in the same
/// order as the input `items` slice.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Placement {
    pub translation: DVec2,
}

/// Angular step (radians) between candidate points on a given spiral ring;
/// smaller values search more thoroughly at the cost of more candidates.
const SPIRAL_ANGLE_STEP: f64 = std::f64::consts::PI / 16.0;

/// Radial step (mm) between successive spiral rings, expressed as a
/// fraction of the largest item's clearance-inflated bounding radius so the
/// search makes reasonably sized jumps regardless of part scale.
const SPIRAL_RADIUS_STEP_FRACTION: f64 = 0.25;

/// An item after footprint preparation: recentered on its own centroid,
/// inflated outward by half the clearance, with a precomputed bounding
/// radius (for spiral step sizing and cheap AABB rejection).
struct Prepared {
    original_index: usize,
    centered_inflated: Vec<DVec2>,
    bounding_radius: f64,
    height: f64,
}

/// Arranges `items` on a rectangular bed `[bed_min, bed_max]`, inflating
/// each footprint outward by `clearance / 2` (so any two placed objects end
/// up at least `clearance` apart) and returning one [`Placement`] per input
/// item, in input order.
///
/// Items are placed tallest-first (ties broken by input order) via a
/// greedy expanding-spiral search centered on the bed, so taller objects
/// -- typically the most likely to warp, shift, or fail mid-print -- end
/// up closest to the bed center. If an item's footprint doesn't fit inside
/// the bed at all (its own bounding radius alone exceeds what the bed can
/// hold), it degrades gracefully to a best-effort center placement rather
/// than failing the whole arrangement.
#[must_use]
pub fn arrange(items: &[ArrangeItem], clearance: f64, bed_min: DVec2, bed_max: DVec2) -> Vec<Placement> {
    let n = items.len();
    if n == 0 {
        return Vec::new();
    }

    let bed_center = (bed_min + bed_max) * 0.5;
    let half_gap = (clearance * 0.5).max(0.0);

    let mut prepared: Vec<Prepared> = items
        .iter()
        .enumerate()
        .map(|(original_index, item)| prepare_item(original_index, item, half_gap))
        .collect();

    // Tallest first; stable sort so equal-height objects keep input order.
    prepared.sort_by(|a, b| b.height.partial_cmp(&a.height).unwrap_or(std::cmp::Ordering::Equal));

    let radius_step = prepared
        .iter()
        .map(|p| p.bounding_radius)
        .fold(0.0_f64, f64::max)
        .max(1.0)
        * SPIRAL_RADIUS_STEP_FRACTION;
    let max_spiral_radius = (bed_max - bed_min).length() * 2.0 + radius_step;

    let mut placed: Vec<Vec<DVec2>> = Vec::with_capacity(n);
    let mut translations = vec![DVec2::ZERO; n];

    for item in &prepared {
        let candidate = find_placement(item, &placed, bed_center, bed_min, bed_max, radius_step, max_spiral_radius)
            .unwrap_or(bed_center);

        let placed_shape: Vec<DVec2> = item.centered_inflated.iter().map(|p| *p + candidate).collect();
        placed.push(placed_shape);
        translations[item.original_index] = candidate;
    }

    translations.into_iter().map(|translation| Placement { translation }).collect()
}

/// Recenters `item.footprint` on its own centroid and inflates it outward
/// by `half_gap`, falling back to the un-inflated centered loop if the
/// offset degenerates (e.g. a near-zero-area input footprint).
fn prepare_item(original_index: usize, item: &ArrangeItem, half_gap: f64) -> Prepared {
    let centroid = centroid_of(&item.footprint);
    let centered: Vec<[f64; 2]> = item.footprint.iter().map(|p| [p.x - centroid.x, p.y - centroid.y]).collect();

    let loop_2d = if half_gap > 0.0 {
        polygon2d::outward_offset(std::slice::from_ref(&centered), half_gap)
            .into_iter()
            .next()
            .unwrap_or(centered)
    } else {
        centered
    };

    let bounding_radius = loop_2d
        .iter()
        .map(|p| (p[0] * p[0] + p[1] * p[1]).sqrt())
        .fold(0.0_f64, f64::max)
        .max(1e-6);
    let centered_inflated: Vec<DVec2> = loop_2d.iter().map(|p| DVec2::new(p[0], p[1])).collect();

    Prepared {
        original_index,
        centered_inflated,
        bounding_radius,
        height: item.height,
    }
}

/// Searches an expanding spiral of candidate center points for the first
/// one where `item`'s inflated footprint (translated there) fits inside
/// the bed and doesn't overlap any shape in `placed`.
fn find_placement(
    item: &Prepared,
    placed: &[Vec<DVec2>],
    bed_center: DVec2,
    bed_min: DVec2,
    bed_max: DVec2,
    radius_step: f64,
    max_spiral_radius: f64,
) -> Option<DVec2> {
    let try_center = |center: DVec2| -> bool {
        let shape: Vec<DVec2> = item.centered_inflated.iter().map(|p| *p + center).collect();
        fits_in_bed(&shape, bed_min, bed_max) && placed.iter().all(|other| !convex_polygons_overlap(&shape, other))
    };

    if try_center(bed_center) {
        return Some(bed_center);
    }

    let mut radius = radius_step;
    while radius <= max_spiral_radius {
        let steps = ((2.0 * std::f64::consts::PI * radius / (radius_step.max(1e-6))).max(1.0) as usize)
            .max((2.0 * std::f64::consts::PI / SPIRAL_ANGLE_STEP) as usize);
        for i in 0..steps {
            let angle = i as f64 * (2.0 * std::f64::consts::PI / steps as f64);
            let candidate = bed_center + DVec2::new(angle.cos(), angle.sin()) * radius;
            if try_center(candidate) {
                return Some(candidate);
            }
        }
        radius += radius_step;
    }

    None
}

/// Whether every vertex of `shape` lies within `[bed_min, bed_max]`.
fn fits_in_bed(shape: &[DVec2], bed_min: DVec2, bed_max: DVec2) -> bool {
    shape
        .iter()
        .all(|p| p.x >= bed_min.x && p.x <= bed_max.x && p.y >= bed_min.y && p.y <= bed_max.y)
}

/// Centroid (arithmetic mean of vertices) of a polygon loop. Adequate for
/// convex hull footprints, where vertex density is roughly uniform relative
/// to shape extent; not the true area centroid, but close enough to center
/// a bounding-radius spiral search.
pub(crate) fn centroid_of(points: &[DVec2]) -> DVec2 {
    if points.is_empty() {
        return DVec2::ZERO;
    }
    points.iter().fold(DVec2::ZERO, |acc, p| acc + *p) / points.len() as f64
}

/// Separating Axis Theorem overlap test for two convex polygons: they
/// overlap unless some edge normal (from either polygon) separates their
/// vertex projections. Cheap and exact for convex inputs -- no boolean
/// library call needed for a simple yes/no overlap test.
fn convex_polygons_overlap(a: &[DVec2], b: &[DVec2]) -> bool {
    if a.len() < 3 || b.len() < 3 {
        return false;
    }
    !has_separating_axis(a, b) && !has_separating_axis(b, a)
}

/// Whether any edge normal of `poly` separates the projected vertex ranges
/// of `poly` and `other`.
fn has_separating_axis(poly: &[DVec2], other: &[DVec2]) -> bool {
    let n = poly.len();
    for i in 0..n {
        let p0 = poly[i];
        let p1 = poly[(i + 1) % n];
        let edge = p1 - p0;
        let axis = DVec2::new(-edge.y, edge.x);
        if axis.length_squared() < 1e-15 {
            continue;
        }

        let (min_a, max_a) = project(poly, axis);
        let (min_b, max_b) = project(other, axis);
        if max_a < min_b || max_b < min_a {
            return true;
        }
    }
    false
}

/// Projects every vertex of `poly` onto `axis`, returning `(min, max)`.
fn project(poly: &[DVec2], axis: DVec2) -> (f64, f64) {
    poly.iter()
        .map(|p| p.dot(axis))
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), d| (lo.min(d), hi.max(d)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square(half: f64) -> Vec<DVec2> {
        vec![
            DVec2::new(-half, -half),
            DVec2::new(half, -half),
            DVec2::new(half, half),
            DVec2::new(-half, half),
        ]
    }

    #[test]
    fn single_item_is_centered() {
        let items = vec![ArrangeItem {
            footprint: square(5.0),
            height: 10.0,
        }];
        let placements = arrange(&items, 10.0, DVec2::new(-100.0, -100.0), DVec2::new(100.0, 100.0));
        assert_eq!(placements.len(), 1);
        assert!(placements[0].translation.length() < 1e-6);
    }

    #[test]
    fn two_items_end_up_non_overlapping() {
        let items = vec![
            ArrangeItem {
                footprint: square(10.0),
                height: 20.0,
            },
            ArrangeItem {
                footprint: square(10.0),
                height: 20.0,
            },
        ];
        let placements = arrange(&items, 4.0, DVec2::new(-100.0, -100.0), DVec2::new(100.0, 100.0));
        assert_eq!(placements.len(), 2);

        let half = 10.0 + 2.0; // inflated by clearance/2 = 2.0
        let shape_a: Vec<DVec2> = square(half).iter().map(|p| *p + placements[0].translation).collect();
        let shape_b: Vec<DVec2> = square(half).iter().map(|p| *p + placements[1].translation).collect();
        assert!(!convex_polygons_overlap(&shape_a, &shape_b));
    }

    #[test]
    fn tallest_item_is_placed_closest_to_bed_center() {
        let items = vec![
            ArrangeItem {
                footprint: square(5.0),
                height: 5.0,
            },
            ArrangeItem {
                footprint: square(5.0),
                height: 50.0,
            },
            ArrangeItem {
                footprint: square(5.0),
                height: 15.0,
            },
        ];
        let placements = arrange(&items, 5.0, DVec2::new(-100.0, -100.0), DVec2::new(100.0, 100.0));
        // Item 1 (height 50, tallest) should land exactly at the bed center
        // since it's placed first with nothing else on the bed yet.
        assert!(placements[1].translation.length() < 1e-6);
    }

    #[test]
    fn all_items_fit_within_bed_bounds() {
        let items = vec![
            ArrangeItem {
                footprint: square(5.0),
                height: 10.0,
            },
            ArrangeItem {
                footprint: square(5.0),
                height: 8.0,
            },
            ArrangeItem {
                footprint: square(5.0),
                height: 12.0,
            },
            ArrangeItem {
                footprint: square(5.0),
                height: 6.0,
            },
        ];
        let bed_min = DVec2::new(-50.0, -50.0);
        let bed_max = DVec2::new(50.0, 50.0);
        let placements = arrange(&items, 5.0, bed_min, bed_max);
        for (i, placement) in placements.iter().enumerate() {
            let half = 5.0 + 2.5;
            let shape: Vec<DVec2> = square(half).iter().map(|p| *p + placement.translation).collect();
            assert!(fits_in_bed(&shape, bed_min, bed_max), "item {i} escaped the bed");
        }
    }

    #[test]
    fn convex_polygons_overlap_detects_separated_squares() {
        let a = square(5.0);
        let b: Vec<DVec2> = square(5.0).iter().map(|p| *p + DVec2::new(20.0, 0.0)).collect();
        assert!(!convex_polygons_overlap(&a, &b));
    }

    #[test]
    fn convex_polygons_overlap_detects_overlapping_squares() {
        let a = square(5.0);
        let b: Vec<DVec2> = square(5.0).iter().map(|p| *p + DVec2::new(3.0, 0.0)).collect();
        assert!(convex_polygons_overlap(&a, &b));
    }
}
