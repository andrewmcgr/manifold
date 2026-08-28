//! Physical-plausibility verification of planned toolpaths.
//!
//! [`find_unsupported_patches`] is an *order-aware* unsupported-bead
//! detector, built as a regression net for non-planar order-field bugs
//! (e.g. bottom-surface conformal Eikonal fields ordering layers so that
//! beads land on material that has not been printed yet).
//!
//! Why `Segment::support_fraction` is not enough: that value (see
//! `toolpath::support_fractions_at`) uses the mesh SDF as a proxy for
//! "previously printed material" — inside the mesh counts as supported.
//! That proxy is exactly what a curved order field can break: the probe
//! point one layer height below a bead can lie *inside* the mesh but at a
//! **higher** order value, meaning the material there is scheduled to
//! print *later*. Such a bead is extruded over air even though the SDF
//! says solid. This detector therefore requires both:
//!
//! 1. the probe point is inside the mesh (`sdf <= tolerance`), **and**
//! 2. the probe point's order value is *lower* than the bead's own order
//!    (already printed when this bead is deposited),
//!
//! or, alternatively, bed contact. Everything else is unsupported.
//!
//! Isolated unsupported points are normal (reprojection noise, legitimate
//! short bridging chords), so points are clustered into spatial patches
//! and only patches whose total extruded length exceeds a bridgeable
//! threshold are reported.

use std::collections::HashMap;

use glam::DVec3;
use manifold_fidget::ScalarField;

use crate::slicing::Layer;
use crate::toolpath::{MoveKind, Path};
use crate::SlicerConfig;

/// A contiguous cluster of unsupported extruded beads.
#[derive(Debug, Clone)]
pub struct UnsupportedPatch {
    /// Mean order value of the patch's segments (which "layer" it is on).
    pub order: f64,
    /// Total extruded bead length (mm) in the patch.
    pub total_length_mm: f64,
    /// Length-weighted centroid of the patch's segment midpoints.
    pub centroid: DVec3,
    /// Number of segments in the patch.
    pub segment_count: usize,
    /// Length (mm) of the single longest segment in the patch — large
    /// values indicate long chords flown over air (e.g. a travel move
    /// miscategorized as an extruding kind) rather than dense bead
    /// coverage of an unsupported region.
    pub max_segment_length_mm: f64,
    /// Extruded length (mm) per [`MoveKind`], largest first — which kinds
    /// of move the unsupported beads were planned as.
    pub length_by_kind: Vec<(MoveKind, f64)>,
    /// Axis-aligned lower bound of the patch's segment midpoints.
    pub min: DVec3,
    /// Axis-aligned upper bound of the patch's segment midpoints.
    pub max: DVec3,
}

/// Tuning knobs for [`find_unsupported_patches`].
#[derive(Debug, Clone, Copy)]
pub struct UnsupportedPatchOptions {
    /// Minimum total extruded length (mm) for a cluster to be reported
    /// as a patch. Filters isolated noise points and short legitimate
    /// bridging chords.
    pub min_patch_length_mm: f64,
    /// Linking radius (mm) for clustering unsupported midpoints; two
    /// unsupported segments closer than this belong to the same patch.
    pub link_radius_mm: f64,
    /// When `true`, also report planner-acknowledged unsupported beads
    /// (`Overhang`/`Bridge` kinds and segments with near-zero stamped
    /// `support_fraction`). Useful for debugging: shows *everything*
    /// physically unsupported, not just what the planner wrongly
    /// believes is supported.
    pub include_acknowledged: bool,
}

impl UnsupportedPatchOptions {
    /// Defaults resolved against a config's nozzle diameter: 3 mm minimum
    /// patch length, `2 * nozzle_diameter` linking radius.
    #[must_use]
    pub fn for_config(config: &SlicerConfig) -> Self {
        Self {
            min_patch_length_mm: 3.0,
            link_radius_mm: 2.0 * config.nozzle_diameter,
            include_acknowledged: false,
        }
    }
}

/// One unsupported extruded segment midpoint, pre-clustering.
struct UnsupportedPoint {
    midpoint: DVec3,
    length: f64,
    order: f64,
    kind: MoveKind,
}

/// Scans every extruding `WallOuter`/`WallInner`/`Infill`/`TopSurface`
/// segment in `paths`, probes one layer height against the local order
/// gradient, and clusters beads that are neither bed-supported nor
/// resting on already-printed material into [`UnsupportedPatch`]es,
/// sorted largest first.
///
/// `MoveKind::Overhang`/`MoveKind::Bridge` segments are deliberately
/// excluded: those are *known* unsupported and printed with adapted
/// flow/anchoring — this detector hunts for beads the planner believes
/// are supported but physically are not. For the same reason, segments
/// whose stamped `Segment::support_fraction` is already near zero are
/// skipped: the planner acknowledged the lack of support and blended
/// bridge-style flow (e.g. conformal underside skins advance as strips
/// anchored *laterally* to the previous strip — unsupported along the
/// order gradient by design). `MoveKind::Travel` extrudes nothing and is
/// skipped via the `extrusion_length` guard.
///
/// Layers are matched to segments by `Segment::order` bit-pattern (the
/// same convention `toolpath::plan` uses to stamp segments), falling back
/// to the nearest layer order when no exact match exists.
#[must_use]
pub fn find_unsupported_patches(
    paths: &[Path],
    layers: &[Layer],
    config: &SlicerConfig,
    options: &UnsupportedPatchOptions,
) -> Vec<UnsupportedPatch> {
    if paths.is_empty() || layers.is_empty() {
        return Vec::new();
    }

    let layer_by_order: HashMap<u64, &Layer> =
        layers.iter().map(|l| (l.order.to_bits(), l)).collect();
    let nearest_layer = |order: f64| -> &Layer {
        layer_by_order
            .get(&order.to_bits())
            .copied()
            .unwrap_or_else(|| {
                layers
                    .iter()
                    .min_by(|a, b| (a.order - order).abs().total_cmp(&(b.order - order).abs()))
                    .expect("layers is non-empty")
            })
    };

    let layer_height = config.layer_height.abs().max(f64::EPSILON);
    // "Rests on floor" convention (see `gcode`): the bed sits at the
    // lowest deposited point.
    let bed_z = paths
        .iter()
        .flat_map(|p| p.points.iter())
        .map(|p| p.z)
        .fold(f64::INFINITY, f64::min);

    // Solid-containment tolerance for the probe point: a bead resting on
    // the previous layer probes right at the surface of the deposited
    // shell, so allow a modest positive SDF band.
    let sdf_tolerance = 0.5 * config.nozzle_diameter;
    // A probe must be at least half a layer "older" (lower order) than
    // the bead itself to count as already printed.
    let order_epsilon = 0.5 * layer_height;
    // Planner-acknowledged unsupported beads (bridge-style blended flow)
    // are not "believed supported" — skip them like `Overhang`/`Bridge`.
    let acknowledged_support = 0.25;

    let mut unsupported: Vec<UnsupportedPoint> = Vec::new();

    for path in paths {
        for (i, segment) in path.segments.iter().enumerate() {
            let acknowledged_kind = matches!(segment.kind, MoveKind::Overhang | MoveKind::Bridge);
            let relevant_kind = acknowledged_kind
                || matches!(
                    segment.kind,
                    MoveKind::WallOuter
                        | MoveKind::WallInner
                        | MoveKind::Infill
                        | MoveKind::TopSurface
                );
            let acknowledged = acknowledged_kind || segment.support_fraction < acknowledged_support;
            if !relevant_kind
                || segment.extrusion_length <= 0.0
                || (acknowledged && !options.include_acknowledged)
            {
                continue;
            }
            let a = path.points[i];
            let b = path.points[(i + 1) % path.points.len()];
            let midpoint = (a + b) * 0.5;
            let length = (b - a).length();
            if length <= f64::EPSILON {
                continue;
            }

            let layer = nearest_layer(segment.order);
            let field = layer.order_field.as_ref();
            // Step against the gradient far enough to descend one full
            // layer in *order space*: where the field is conformally
            // blended/relaxed its gradient magnitude deviates from 1, so a
            // fixed geometric step could stay within the current layer's
            // order band and read as "not yet printed". Scale the step by
            // 1/|grad| (clamped to avoid runaway probes in flat spots).
            let (gradient_dir, gradient_len) =
                match crate::order_field::numeric_gradient(field, midpoint)
                    .filter(|g| g.length_squared() > 1e-12 && g.is_finite())
                {
                    Some(g) => (g / g.length(), g.length()),
                    None => (crate::slicing::BUILD_DIRECTION, 1.0),
                };
            let step = (layer_height / gradient_len).clamp(layer_height, 4.0 * layer_height);
            let probe = midpoint - step * gradient_dir;

            // Bed contact: probing at/below the plate means the bead is
            // squished onto the bed — supported.
            if probe.z <= bed_z + 0.25 * layer_height {
                continue;
            }

            // Already-printed material: inside the deposited shell AND
            // ordered earlier than this bead.
            let inside_solid = layer
                .mesh_sdf
                .as_ref()
                .is_some_and(|sdf| sdf.as_ref().sample(probe).value <= sdf_tolerance);
            if inside_solid {
                let probe_order = field.order(probe);
                if probe_order.is_finite() && probe_order <= segment.order - order_epsilon {
                    continue;
                }
            }

            unsupported.push(UnsupportedPoint {
                midpoint,
                length,
                order: segment.order,
                kind: segment.kind,
            });
        }
    }

    let mut patches: Vec<UnsupportedPatch> = cluster_patches(&unsupported, options.link_radius_mm)
        .into_iter()
        .filter(|p| p.total_length_mm >= options.min_patch_length_mm)
        .collect();
    patches.sort_by(|a, b| b.total_length_mm.total_cmp(&a.total_length_mm));
    patches
}

/// Groups unsupported points into connected clusters: two points closer
/// than `link_radius` (Euclidean, 3D) belong to the same patch. Uses a
/// spatial hash grid (cell size = `link_radius`) so only the 27
/// neighboring cells are checked per point, with path-compressed
/// union-find for the connectivity.
fn cluster_patches(points: &[UnsupportedPoint], link_radius: f64) -> Vec<UnsupportedPatch> {
    if points.is_empty() {
        return Vec::new();
    }
    let link_radius = link_radius.max(f64::EPSILON);
    let cell = |p: DVec3| -> (i64, i64, i64) {
        (
            (p.x / link_radius).floor() as i64,
            (p.y / link_radius).floor() as i64,
            (p.z / link_radius).floor() as i64,
        )
    };

    let mut grid: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
    for (i, p) in points.iter().enumerate() {
        grid.entry(cell(p.midpoint)).or_default().push(i);
    }

    let mut parent: Vec<usize> = (0..points.len()).collect();
    fn find(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }
    let link_radius_sq = link_radius * link_radius;
    for (i, p) in points.iter().enumerate() {
        let (cx, cy, cz) = cell(p.midpoint);
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let Some(neighbors) = grid.get(&(cx + dx, cy + dy, cz + dz)) else {
                        continue;
                    };
                    for &j in neighbors {
                        if j <= i {
                            continue;
                        }
                        if points[j].midpoint.distance_squared(p.midpoint) <= link_radius_sq {
                            let (ri, rj) = (find(&mut parent, i), find(&mut parent, j));
                            if ri != rj {
                                parent[rj] = ri;
                            }
                        }
                    }
                }
            }
        }
    }

    let mut clusters: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..points.len() {
        let root = find(&mut parent, i);
        clusters.entry(root).or_default().push(i);
    }

    clusters
        .into_values()
        .map(|members| {
            let mut total_length = 0.0;
            let mut max_segment_length = 0.0f64;
            let mut by_kind: HashMap<u8, (MoveKind, f64)> = HashMap::new();
            let mut weighted_centroid = DVec3::ZERO;
            let mut order_sum = 0.0;
            let mut min = DVec3::splat(f64::INFINITY);
            let mut max = DVec3::splat(f64::NEG_INFINITY);
            for &i in &members {
                let p = &points[i];
                total_length += p.length;
                max_segment_length = max_segment_length.max(p.length);
                by_kind.entry(p.kind as u8).or_insert((p.kind, 0.0)).1 += p.length;
                weighted_centroid += p.midpoint * p.length;
                order_sum += p.order;
                min = min.min(p.midpoint);
                max = max.max(p.midpoint);
            }
            UnsupportedPatch {
                order: order_sum / members.len() as f64,
                total_length_mm: total_length,
                centroid: weighted_centroid / total_length.max(f64::EPSILON),
                segment_count: members.len(),
                max_segment_length_mm: max_segment_length,
                length_by_kind: {
                    let mut kinds: Vec<(MoveKind, f64)> = by_kind.into_values().collect();
                    kinds.sort_by(|a, b| b.1.total_cmp(&a.1));
                    kinds
                },
                min,
                max,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(x: f64, y: f64, z: f64, length: f64) -> UnsupportedPoint {
        UnsupportedPoint {
            midpoint: DVec3::new(x, y, z),
            length,
            order: z,
            kind: MoveKind::WallOuter,
        }
    }

    #[test]
    fn clustering_merges_points_within_link_radius_and_splits_distant_groups() {
        // Two groups 10mm apart, points within each group 0.5mm apart.
        let points = vec![
            point(0.0, 0.0, 1.0, 1.0),
            point(0.5, 0.0, 1.0, 1.0),
            point(1.0, 0.0, 1.0, 1.0),
            point(10.0, 0.0, 1.0, 2.0),
            point(10.5, 0.0, 1.0, 2.0),
        ];
        let mut patches = cluster_patches(&points, 0.8);
        patches.sort_by(|a, b| a.centroid.x.total_cmp(&b.centroid.x));
        assert_eq!(patches.len(), 2);
        assert_eq!(patches[0].segment_count, 3);
        assert!((patches[0].total_length_mm - 3.0).abs() < 1e-12);
        assert_eq!(patches[1].segment_count, 2);
        assert!((patches[1].total_length_mm - 4.0).abs() < 1e-12);
    }

    #[test]
    fn clustering_is_transitive_across_chained_points() {
        // A chain where consecutive points are within the radius but the
        // ends are far apart must still form one patch.
        let points: Vec<_> = (0..20)
            .map(|i| point(i as f64 * 0.7, 0.0, 1.0, 0.5))
            .collect();
        let patches = cluster_patches(&points, 0.8);
        assert_eq!(patches.len(), 1);
        assert_eq!(patches[0].segment_count, 20);
    }
}

/// One closed toolpath loop that floats almost entirely in mid-air.
#[derive(Debug, Clone)]
pub struct FloatingLoop {
    /// Mean order value of the loop's extruding segments.
    pub order: f64,
    /// Total extruded length (mm) of the loop.
    pub total_length_mm: f64,
    /// Length-weighted centroid of the loop's segment midpoints.
    pub centroid: DVec3,
    /// Fraction (0..=1, length-weighted) of the loop's extruded length
    /// whose downward-order probe found neither bed nor earlier-order
    /// solid material.
    pub unsupported_fraction: f64,
    /// Extruded length (mm) per [`MoveKind`], largest first.
    pub length_by_kind: Vec<(MoveKind, f64)>,
    /// Up to a few unsupported segment midpoints with their probe
    /// diagnostics: (midpoint, probe point, sdf value at probe, order at
    /// probe) — for manual debugging of why the loop counts as floating.
    pub probe_samples: Vec<(DVec3, DVec3, f64, f64)>,
}

/// Finds closed loops that materialize almost entirely in mid-air —
/// contour *islands* with no connection to already-printed material —
/// regardless of stamped kind or `support_fraction` (a fully floating
/// loop is unprintable even when the planner blended bridge flow into
/// it; bridge flow needs anchors at both ends).
///
/// Reports loops with at least 90% (length-weighted) unsupported
/// extruded length and more than 2 mm total extrusion.
#[must_use]
pub fn floating_loops(
    paths: &[Path],
    layers: &[Layer],
    config: &SlicerConfig,
) -> Vec<FloatingLoop> {
    if paths.is_empty() || layers.is_empty() {
        return Vec::new();
    }

    let layer_by_order: HashMap<u64, &Layer> =
        layers.iter().map(|l| (l.order.to_bits(), l)).collect();
    let nearest_layer = |order: f64| -> &Layer {
        layer_by_order
            .get(&order.to_bits())
            .copied()
            .unwrap_or_else(|| {
                layers
                    .iter()
                    .min_by(|a, b| (a.order - order).abs().total_cmp(&(b.order - order).abs()))
                    .expect("layers is non-empty")
            })
    };

    let layer_height = config.layer_height.abs().max(f64::EPSILON);
    let bed_z = paths
        .iter()
        .flat_map(|p| p.points.iter())
        .map(|p| p.z)
        .fold(f64::INFINITY, f64::min);
    let sdf_tolerance = 0.5 * config.nozzle_diameter;
    let order_epsilon = 0.5 * layer_height;

    let mut loops = Vec::new();
    for path in paths {
        let mut total = 0.0;
        let mut unsupported = 0.0;
        let mut centroid = DVec3::ZERO;
        let mut order_sum = 0.0;
        let mut by_kind: Vec<(MoveKind, f64)> = Vec::new();
        let mut probe_samples: Vec<(DVec3, DVec3, f64, f64)> = Vec::new();
        for (i, segment) in path.segments.iter().enumerate() {
            if segment.extrusion_length <= 0.0 {
                continue;
            }
            let a = path.points[i];
            let b = path.points[(i + 1) % path.points.len()];
            let length = (b - a).length();
            if length <= f64::EPSILON {
                continue;
            }
            let midpoint = (a + b) * 0.5;
            total += length;
            centroid += midpoint * length;
            order_sum += segment.order * length;
            match by_kind.iter_mut().find(|(k, _)| *k == segment.kind) {
                Some((_, l)) => *l += length,
                None => by_kind.push((segment.kind, length)),
            }

            let layer = nearest_layer(segment.order);
            let field = layer.order_field.as_ref();
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
                continue;
            }
            let inside_solid = layer
                .mesh_sdf
                .as_ref()
                .is_some_and(|sdf| sdf.as_ref().sample(probe).value <= sdf_tolerance);
            if inside_solid {
                let probe_order = field.order(probe);
                if probe_order.is_finite() && probe_order <= segment.order - order_epsilon {
                    continue;
                }
            }
            unsupported += length;
            if probe_samples.len() < 4 {
                let sdf_value = layer
                    .mesh_sdf
                    .as_ref()
                    .map_or(f64::NAN, |sdf| sdf.as_ref().sample(probe).value);
                probe_samples.push((midpoint, probe, sdf_value, field.order(probe)));
            }
        }

        if total > 2.0 && unsupported / total >= 0.9 {
            let mut length_by_kind = by_kind;
            length_by_kind.sort_by(|a, b| b.1.total_cmp(&a.1));
            loops.push(FloatingLoop {
                order: order_sum / total,
                total_length_mm: total,
                centroid: centroid / total,
                unsupported_fraction: unsupported / total,
                length_by_kind,
                probe_samples,
            });
        }
    }
    loops.sort_by(|a, b| a.order.total_cmp(&b.order));
    loops
}
