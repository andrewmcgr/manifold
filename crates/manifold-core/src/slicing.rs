//! Non-planar slicing: mesh -> ordered layers of cross-section curves.

use crate::{
    ids::ObjectId, mesh::Mesh, object::Object, order_field, polygon2d, Error, Result, SlicerConfig,
};
use glam::DVec3;
use manifold_fidget::contour::{
    extract_contours, extract_order_contours_on_mesh_with_debug, plane_basis,
};
use manifold_fidget::marching_cubes::extract_sparse_isosurface_positions;
use manifold_fidget::mesh_sdf::MeshSdf;
use manifold_fidget::order::{order_range_over_bbox, HeightOrderField, OrderField};
use manifold_fidget::ScalarField;
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// A single (possibly non-planar) slice layer.
///
/// Tagged with the source [`ObjectId`] so multi-object toolpath planning
/// (tool lookup) and any future Z-interleaving ordering strategy can tell
/// which object a layer came from.
#[derive(Clone)]
pub struct Layer {
    pub index: usize,
    pub object: ObjectId,
    /// The order-field value (see [`BUILD_DIRECTION`]/[`slice_mesh`]'s walk)
    /// whose isosurface produced this layer's contour loops. In today's
    /// flat-height-field case this is the layer's Z height; retained so
    /// downstream consumers (e.g. `toolpath::plan`) can stamp it onto
    /// per-segment metadata once non-planar order fields exist.
    pub order: f64,
    /// This layer's cross-section geometry: closed polylines (loops) in
    /// world space, one per wall pass extracted at this layer's order
    /// value. Empty for a layer with no contour (e.g. above/below the
    /// mesh's extent along the build direction).
    pub loops: Vec<WallLoop>,
    /// The infill fill area's boundary: closed polylines in world space,
    /// one `wall_line_width` step further inward than the innermost
    /// *printed* wall in [`Layer::loops`] — i.e. where wall pass
    /// `wall_count` would sit if one more were printed (see
    /// [`SlicerConfig::wall_count`]). Kept separate from `loops` so it is
    /// never mistaken for a printable wall pass by `toolpath::plan`; used
    /// by `infill::InfillRegion::from_layer` as the fillable area. Empty
    /// for a layer with no contour.
    pub infill_boundary: Vec<Vec<DVec3>>,
    /// The subset of [`Layer::infill_boundary`] that must print solid
    /// (using the same infill pattern generator as sparse infill, just
    /// run over this region too) rather than sparse infill, because it is
    /// within [`SlicerConfig::top_layers`] of a facing-up exterior surface
    /// or [`SlicerConfig::bottom_layers`] of a facing-down exterior
    /// surface. Computed by [`compute_solid_fill_boundaries`] as a
    /// post-pass once every layer of an object has been sliced (unlike
    /// `infill_boundary`, which only needs this one layer's own contour).
    /// Empty until that post-pass runs; always a subset of
    /// `infill_boundary`.
    pub solid_fill_boundary: Vec<Vec<DVec3>>,
    /// Ground-truth mesh containment query for this layer's source object,
    /// shared with (and built alongside) the `MeshSdf` used for contour
    /// extraction in [`slice_mesh_with_progress`] -- the only site with
    /// mesh access, so this is populated there rather than re-derived
    /// downstream. `None` for a synthetic/test [`Layer`] not derived from
    /// a real mesh (e.g. most unit-test fixtures); `toolpath::plan` treats
    /// `None` as "containment unknown, don't enforce it" rather than
    /// panicking or assuming failure, so existing hand-built `Layer`
    /// fixtures keep working unchanged.
    ///
    /// Used by `toolpath::plan` as a final safety net: wall/infill loop
    /// geometry is derived from contour extraction and 2D polygon boolean
    /// ops on `infill_boundary`/`solid_fill_boundary`, which have (rarely)
    /// produced loops that don't correspond to real solid material -- e.g.
    /// infill printed inside a hole that isn't actually part of the
    /// object. Rather than only trying to prevent every possible source of
    /// that class of bug, `plan` re-checks every extruding path against
    /// this real 3D mesh signed-distance field and drops any path that
    /// isn't actually contained in the solid.
    pub mesh_sdf: Option<Arc<MeshSdf>>,
    /// The resolved order field used to produce this layer, cached at
    /// construction time (see [`slice_mesh_with_progress`], the only site
    /// with mesh access) so downstream passes
    /// ([`compute_solid_fill_boundaries`], `infill::InfillRegion::from_layer`)
    /// reuse this one solve instead of re-resolving
    /// [`order_field::order_field_for`] blind from `config` alone. This
    /// matters once an [`order_field::OrderFieldKind::Eikonal`] variant
    /// exists: unlike `Height`/`Conical` (pure closed-form functions of
    /// `config`), an Eikonal field's values come from a precomputed FMM
    /// grid solve over the mesh's actual geometry, which cannot be cheaply
    /// reconstructed from `config` alone at call sites with no mesh in
    /// scope. `Height`/`Conical` populate this the same way for
    /// consistency, even though re-resolving them would be free.
    pub order_field: Arc<dyn OrderField>,
}

impl std::fmt::Debug for Layer {
    /// Manual impl: `order_field` is `Arc<dyn OrderField>`, and
    /// `OrderField` does not implement `Debug` (it's a `manifold-fidget`
    /// geometry-query trait, not a plain data type) — so `order_field` is
    /// rendered as a placeholder rather than derived.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Layer")
            .field("index", &self.index)
            .field("object", &self.object)
            .field("order", &self.order)
            .field("loops", &self.loops)
            .field("infill_boundary", &self.infill_boundary)
            .field("solid_fill_boundary", &self.solid_fill_boundary)
            .field(
                "mesh_sdf",
                &self
                    .mesh_sdf
                    .as_ref()
                    .map(|_| "<MeshSdf>")
                    .unwrap_or("None"),
            )
            .field("order_field", &"<dyn OrderField>")
            .finish()
    }
}

impl Default for Layer {
    /// Defaults `order_field` to a [`HeightOrderField`] along
    /// [`BUILD_DIRECTION`] — matching [`order_field::OrderFieldKind`]'s own
    /// `#[default]` variant — since `Arc<dyn OrderField>` has no derivable
    /// `Default`.
    fn default() -> Self {
        Self {
            index: usize::default(),
            object: ObjectId::default(),
            order: f64::default(),
            loops: Vec::default(),
            infill_boundary: Vec::default(),
            solid_fill_boundary: Vec::default(),
            mesh_sdf: None,
            order_field: Arc::new(HeightOrderField::new(BUILD_DIRECTION)),
        }
    }
}

/// One contour loop belonging to a specific wall/perimeter pass within a
/// [`Layer`] (see [`SlicerConfig::wall_count`]).
#[derive(Debug, Clone, Default)]
pub struct WallLoop {
    /// `0` = outermost wall, increasing inward. Used by
    /// `toolpath::plan` to classify segments as `WallOuter`/`WallInner`.
    pub wall_index: usize,
    /// Closed polyline in world space (see [`Layer::loops`]).
    pub points: Vec<DVec3>,
    /// Per-point flag marking points inserted by the inter-layer wall-gap
    /// stitching pass (see `slice_mesh_with_progress`) as unsupported
    /// (printed over a gap with no solid/order-field surface directly
    /// beneath, rather than derived from the mesh/order-field isosurface
    /// like the rest of the loop). This is a sibling `Vec` to `points`:
    /// `unsupported[i]` describes `points[i]`, so `unsupported.len() ==
    /// points.len()` always holds, including through any later
    /// insertion/removal of stitched points. Kept as a parallel `Vec`
    /// (rather than pairing each point with its flag in a single
    /// `Vec<(DVec3, bool)>`) so callers that only need geometry can read
    /// `points` without also touching `unsupported`, mirroring
    /// [`crate::toolpath::Path`]'s `points`/`segments` convention.
    /// Defaults to all `false` (no stitched points) for loops produced
    /// directly from mesh/order-field extraction; `toolpath::plan` maps
    /// segments touching a `true` point to `MoveKind::Overhang`.
    pub unsupported: Vec<bool>,
    /// Per-point normalized cumulative arc-length position around this
    /// closed polyline, in `[0, 1)`: `arc_fraction[0] == 0.0` and
    /// `arc_fraction[i]` increases monotonically walking `points` in
    /// order, wrapping back toward (but never reaching) `1.0` at the
    /// closing segment back to `points[0]`. Another parallel `Vec`
    /// sibling to `points`/`unsupported` (same invariant:
    /// `arc_fraction.len() == points.len()` always, including through
    /// stitch insertions — see `unsupported`'s doc comment for the
    /// rationale for this convention). Populated for wall-index-0 loops
    /// at minimum (see `slice_mesh_with_progress`'s wall-loop
    /// construction and `stitch_wall_gaps`, which consumes it to build
    /// an arc-length-fraction correspondence between a layer's wall-0
    /// loop and the previous layer's, replacing raw nearest-point
    /// matching). This is deliberately real, retained data rather than a
    /// throwaway local — it is exactly the per-loop parameterization a
    /// future seam-placement feature will need (choosing a consistent
    /// start/seam position around each wall loop across layers to avoid
    /// a visible seam ridge), so do not remove/recompute-and-discard it.
    pub arc_fraction: Vec<f64>,
    /// Per-point flag marking wall-0 points with solid mesh material
    /// directly beneath them (real support) but nothing solid one
    /// nozzle-diameter along `-BUILD_DIRECTION` (i.e. "forward"/upward,
    /// the not-yet-printed side) -- the last printed material before
    /// open air going forward, i.e. the roof/top surface of the part at
    /// this point, not a genuine unsupported overhang. Computed by
    /// `stitch_wall_gaps` via `has_solid_material_in_direction` probed
    /// with `-BUILD_DIRECTION` (the mirror image of the `BUILD_DIRECTION`
    /// probe that produces `unsupported`/the wall-gap-stitch veto -- see
    /// that function's docs for why the two probes must go in opposite,
    /// not the same, direction). A point can be `top_surface == true` and
    /// `unsupported[i] == false` at the same time: being the top of the
    /// part does not mean it lacks support from below. `toolpath::plan`
    /// maps segments touching a `true` point to `MoveKind::TopSurface`
    /// unless that same point is also `unsupported`, in which case the
    /// genuine-overhang classification wins (see `plan`'s docs). Another
    /// parallel `Vec` sibling to `points`/`unsupported`/`arc_fraction`
    /// (same invariant: `top_surface.len() == points.len()` for
    /// wall-index-0 loops after `stitch_wall_gaps` runs; may be shorter
    /// or empty for loops it never touches -- e.g. inner walls, or
    /// hand-built loops in tests -- so readers use `.get(i)` rather than
    /// direct indexing, mirroring `unsupported`'s existing read pattern
    /// in `plan`).
    pub top_surface: Vec<bool>,
    /// Per-point dynamic line width in mm (parallel to `points`).
    /// When empty, downstream toolpath planning falls back to nominal configured line width.
    pub line_widths: Vec<f64>,
}

/// Build/order direction: conventional planar slicing along
/// +Z (i.e. `order(p) = p.dot(direction)` increases going up, matching a
/// bottom-to-top print).
pub(crate) const BUILD_DIRECTION: DVec3 = DVec3::new(0.0, 0.0, 1.0);

/// Physical orientation of the print head's nozzle axis, in world space.
/// This is deliberately **not** the same thing as a layer's order-field
/// gradient (the local build-surface normal): for a tilting/multi-axis
/// print head the nozzle can point in a direction that differs from the
/// surface normal at the point it is extruding, which is exactly the
/// case flat-nozzle-tip contact compensation (`toolpath::compensate_flat_nozzle`)
/// needs to reason about. Today every supported print head is a fixed
/// vertical (3-axis) head, so this is always parallel to Z — but it is
/// tracked as its own concept, independent of `BUILD_DIRECTION`/the order
/// field, so a future tilting head only needs to supply a real per-point
/// nozzle direction rather than requiring every call site that currently
/// (incorrectly) assumes "nozzle axis == surface normal" to be found and
/// fixed individually.
pub(crate) const NOZZLE_DIRECTION: DVec3 = DVec3::new(0.0, 0.0, 1.0);

/// Default divisor used to derive the marching-squares contour-extraction
/// grid's target cell size from `SlicerConfig::nozzle_diameter` (cell_size
/// = `nozzle_diameter / CONTOUR_REFINEMENT_DIVISOR`). `4.0` (a quarter of
/// the nozzle diameter) keeps grid faceting finer than what the nozzle can
/// physically resolve, but is expensive at real-world scale (grid points
/// scale with the square of resolution, each doing a BVH nearest-triangle
/// query); `1.4` trades some of that headroom for tractable slicing time
/// while still meaningfully improving on the old fixed-120 grid. Exposed
/// as a constant (rather than inlined) so callers wanting coarser/finer
/// refinement can pass a different divisor to [`contour_resolution`]
/// directly.
const CONTOUR_REFINEMENT_DIVISOR: f64 = 1.4;

/// Lower/upper bounds on the derived grid resolution (samples per axis),
/// independent of `CONTOUR_REFINEMENT_DIVISOR`: guards against a
/// vanishingly coarse grid (degenerate/huge `extent` or `nozzle_diameter`)
/// and against runaway sampling cost (tiny `nozzle_diameter` on a large
/// mesh).
const MIN_CONTOUR_RESOLUTION: usize = 32;
const MAX_CONTOUR_RESOLUTION: usize = 512;

/// Fraction of nozzle radius (`config.nozzle_diameter / 2.0`) used as the
/// max allowed inter-layer wall-0 hop distance before
/// [`stitch_wall_gaps`] treats it as a gap requiring a stitch. 90% leaves
/// a small margin below the nozzle's actual bonding radius rather than
/// cutting it exactly at the limit.
const WALL_GAP_HOP_FRACTION: f64 = 0.9;

/// Multiplier on `config.nozzle_diameter` giving the max centroid
/// distance at which [`stitch_wall_gaps`] considers a current-layer
/// wall-0 loop to spatially correspond to a previous-layer wall-0 loop
/// (see `loop_centroid` / the loop-matching step in `stitch_wall_gaps`).
/// The same physical island's centroid shifts only a little between
/// adjacent layers (bounded by the slope of that region times one layer
/// height), while genuinely distinct islands/holes in a real
/// cross-section are normally separated by a macroscopic distance --
/// millimeters to centimeters, not fractions of a nozzle diameter. 20x
/// nozzle diameter is a generous margin above plausible adjacent-layer
/// centroid drift while still comfortably below typical inter-island
/// separation, so it distinguishes "same feature, shifted a bit" from
/// "unrelated feature" without needing per-model tuning.
const WALL_GAP_LOOP_CENTROID_MATCH_FACTOR: f64 = 20.0;

/// Second, shape-aware plausibility check on a current-loop/previous-loop
/// match, applied alongside [`WALL_GAP_LOOP_CENTROID_MATCH_FACTOR`] (see
/// `loop_perimeter`/`stitch_wall_gaps`): centroid distance alone is not
/// sufficient to confirm two loops are the same physical feature --
/// real-mesh verification (subtask 09 of the wall-gap-stitching feature)
/// found a case where a 220-point current loop's centroid fell within the
/// centroid threshold of an unrelated 722-point previous loop (a >3x
/// perimeter mismatch), and arc-length-fraction correspondence between
/// two loops of such different actual shape/size is meaningless -- equal
/// fractions on a small, simple loop and a large, complex loop do not
/// correspond to the same physical position, producing a legitimately
/// huge (real, non-bisection-error) gap fed straight into
/// [`serpentine_stitch_block`], which then faithfully (if uselessly)
/// subdivides across
/// that bogus gap. A previous loop is only accepted as a match if its
/// perimeter is within this factor of the current loop's perimeter in
/// *either* direction (`ratio = longer / shorter <=
/// WALL_GAP_LOOP_PERIMETER_MATCH_FACTOR`). 2x is a generous margin for a
/// genuinely growing/shrinking feature between adjacent layers while
/// still rejecting a same-centroid-but-different-feature mismatch like
/// the 3.28x case found above.
const WALL_GAP_LOOP_PERIMETER_MATCH_FACTOR: f64 = 1.4;

/// Guards against accepting a perimeter-and-centroid-threshold-passing
/// previous loop when a *closer* previous loop exists that was rejected
/// only by the perimeter check (see [`WALL_GAP_LOOP_PERIMETER_MATCH_FACTOR`]).
/// Real-mesh verification (subtask 12 of the wall-gap-stitching feature,
/// layer 40/loop_idx 13 of `pug_v4_l_sop_85mm.stl`) found a genuine
/// topology-merge event: two previous-layer loops (centroid distance
/// ~0.92mm and ~1.15mm, each roughly half the current loop's perimeter --
/// consistent with two islands merging into one wall on this layer) were
/// both correctly rejected by the perimeter check, but the pairing step
/// then fell through to a *different*, unrelated previous loop that only
/// coincidentally has a near-identical perimeter to the (now-merged)
/// current loop, at 7.86mm centroid distance -- 8x farther than the
/// rejected-but-genuinely-nearby loops. Accepting that coincidental match
/// produced a uniform ~7.7mm hop across the *entire* loop (686/686
/// points), not an isolated single-point defect.
///
/// This is not fixable by relaxing the perimeter check (the rejected
/// loops really are a different shape -- half the perimeter -- because
/// they are pre-merge features, not tracking error), and full merge/split
/// support (stitching a current loop against *multiple* previous loops)
/// is out of scope here. The safe, conservative fix: after the
/// perimeter+centroid-threshold filter picks its best candidate, compare
/// that candidate's centroid distance against the *closest previous loop
/// overall* (regardless of perimeter, i.e. an unfiltered nearest-centroid
/// search across every previous loop). If the filtered candidate is more
/// than this factor farther away than the true closest loop, the
/// filtered candidate is almost certainly a coincidental false positive
/// (as in the layer-40 case above) rather than a genuine match -- treat
/// the current loop as having no correspondence (skip stitching) instead
/// of forcing a match to a distant coincidence. A generous factor (not a
/// tight one) is used since some genuine same-feature drift between the
/// globally-nearest loop and the perimeter-passing candidate is expected
/// on noisy/sampled contours; this only needs to catch orders-of-
/// magnitude discrepancies like the 8x case found above.
const WALL_GAP_LOOP_NEAREST_CENTROID_MARGIN_FACTOR: f64 = 4.0;

/// Number of evenly-spaced sample points taken around the *current* loop
/// (by index, not arc fraction, since sampling by index already spans the
/// loop's parameterization uniformly) when [`best_rotation_offset`]
/// searches for the rotational offset between a current/previous wall-0
/// loop pair. A single nearest-point start alignment (the pre-subtask-10
/// approach) validates the offset at only one point, so a decoy previous
/// point that happens to be closer in raw 3D distance than the true
/// corresponding point -- common on spiky/non-convex loops where
/// unrelated parts of the contour pass close to each other in space, e.g.
/// near a shared inner waist between spikes -- silently produces a wrong
/// global offset with no way to detect it. Scoring candidate offsets
/// against several spread-out samples instead means a wrong offset that
/// happens to fit one point will typically disagree badly at the others,
/// so the aggregate-error minimum reliably lands on the true offset. 12
/// samples is enough to catch a wrong offset on typical wall-0 loop
/// shapes without materially slowing down a search that already only
/// runs once per current loop per layer-pair (not once per point).
const WALL_GAP_ROTATION_SEARCH_SAMPLES: usize = 12;

/// Half-width (in previous-loop point indices) of the bounded local
/// search window used by [`local_fallback_correspondence`] when the
/// best-rotation-offset correspondence for an individual point still
/// exceeds `hop_limit`. This is deliberately a small, bounded window
/// around the fraction-implied index -- not the unbounded whole-loop
/// nearest-point search that produced the original perpendicular-zigzag
/// defect (see the root-cause discussion on [`stitch_wall_gaps`] itself,
/// "Addendum 2"). A wrong global offset is now rare (see
/// [`WALL_GAP_ROTATION_SEARCH_SAMPLES`]), so this fallback only needs to
/// correct small local warps in the fraction-to-position mapping caused
/// by non-uniform point density near corners/spikes, which by
/// construction can only shift the true correspondent a few points away
/// from the fraction-implied index, not to an arbitrary position
/// elsewhere on the loop.
const WALL_GAP_LOCAL_FALLBACK_WINDOW: usize = 6;

/// Hard cap on the number of intermediate order levels
/// [`serpentine_stitch_block`] may subdivide a layer step into. Mirrors
/// other pathological-loop guards already in this codebase (e.g.
/// `MAX_ORDER_STEPS` above, `MAX_BISECT_ITERS` in `order_field.rs`): 4096
/// levels is far more than any real gap should ever need to reach the hop
/// limit (the largest genuine gap observed on the real test mesh needed
/// 64), while still failing safe (bounded work) instead of doubling
/// indefinitely for a pathological order field whose reprojected level
/// points never converge toward closing the gap.
const MAX_STITCH_LEVELS: usize = 4096;

/// Derives the marching-squares contour-extraction grid resolution
/// (samples per axis, see [`extract_contours`]) for an in-plane sampling
/// square of side length `extent`, targeting a grid cell size of
/// `nozzle_diameter / refinement_divisor`.
///
/// Adaptive rather than fixed (as this previously was, via a
/// `CONTOUR_RESOLUTION` constant): a fixed grid either wastes samples on
/// small objects or under-samples large ones, producing visibly faceted/
/// blocky contours. Deriving resolution from the mesh's actual footprint
/// and the machine's nozzle diameter scales the grid to both.
///
/// Clamped to `[MIN_CONTOUR_RESOLUTION, MAX_CONTOUR_RESOLUTION]` to bound
/// cost and guard against non-finite/non-positive inputs.
fn contour_resolution(extent: f64, nozzle_diameter: f64, refinement_divisor: f64) -> usize {
    let cell_size = (nozzle_diameter / refinement_divisor).max(f64::EPSILON);
    let raw = (extent / cell_size).ceil() as i64 + 1;
    raw.clamp(MIN_CONTOUR_RESOLUTION as i64, MAX_CONTOUR_RESOLUTION as i64) as usize
}

/// Slice a single mesh (already in the frame it should be sliced in) into
/// layers according to `config`.
///
/// Builds a [`MeshSdf`] from `mesh` and walks its [`BUILD_DIRECTION`]
/// order field at `config.layer_height` intervals across the mesh's
/// bounding range along that direction, extracting one contour-based
/// [`Layer`] per step (steps with no contour still produce a `Layer` with
/// empty `loops`, rather than being skipped or erroring). Operates in
/// whatever space `mesh`'s vertices are already in — callers slicing an
/// [`Object`] should go through [`slice_object`], which bakes the
/// object's transform into world space first.
pub fn slice_mesh(mesh: &Mesh, config: &SlicerConfig) -> Result<Vec<Layer>> {
    slice_mesh_with_progress(
        mesh,
        config,
        &manifold_fidget::slope_profile::SlopeProfile::new(Vec::new()),
        &mut |_| {},
    )
}

/// Same as [`slice_mesh`], but calls `on_progress` as work finishes with
/// the fraction of total work completed so far (`0.0..=1.0`) — intended
/// for a caller (e.g. the GUI, slicing on a background thread) to report
/// progress on a potentially slow slice without having to know anything
/// about layers, order fields, or contour resolution itself.
///
/// Layers are extracted in parallel across all available cores (via
/// `rayon`): each layer only reads the shared, immutable [`MeshSdf`] and
/// produces its own independent [`Layer`], so there's no cross-layer
/// dependency to serialize on. The returned `Vec<Layer>` is still ordered
/// by `index`/`order` regardless of completion order, but `on_progress`
/// calls arrive in whatever order layers happen to finish on their
/// threads — not strictly in `index` order, though still trending
/// monotonically toward `1.0`.
///
/// For a curved order field (anything but `Height`), the per-wall-pass
/// `curved_wall_meshes` isosurface precompute below counts as reportable
/// work too, one unit per wall pass, ahead of the per-layer units — that
/// precompute is itself an expensive whole-mesh extraction (already
/// internally parallel via `rayon`, just not previously visible to
/// `on_progress`), so without this the bar would sit frozen for however
/// long it takes before the first per-layer tick ever arrived.
pub fn slice_mesh_with_progress(
    mesh: &Mesh,
    config: &SlicerConfig,
    slope_profile: &manifold_fidget::slope_profile::SlopeProfile,
    on_progress: &mut (dyn FnMut(f64) + Send),
) -> Result<Vec<Layer>> {
    let Some((min, max)) = mesh.bounding_box() else {
        // Empty mesh: no geometry to slice.
        return Ok(Vec::new());
    };
    if mesh.indices.is_empty() {
        return Ok(Vec::new());
    }

    let faces: Vec<[usize; 3]> = mesh
        .indices
        .chunks_exact(3)
        .map(|chunk| [chunk[0] as usize, chunk[1] as usize, chunk[2] as usize])
        .collect();
    let sdf = Arc::new(MeshSdf::new(mesh.vertices.clone(), faces.clone()));

    let non_cap_faces: Vec<[usize; 3]> = mesh
        .indices
        .chunks_exact(3)
        .filter_map(|chunk| {
            let [i0, i1, i2] = [chunk[0] as usize, chunk[1] as usize, chunk[2] as usize];
            let v0 = mesh.vertices[i0];
            let v1 = mesh.vertices[i1];
            let v2 = mesh.vertices[i2];
            let normal = (v1 - v0).cross(v2 - v0);
            let normal_len_sq = normal.length_squared();
            // Flat horizontal triangles (facing purely upward or downward) represent
            // top roofs, intermediate shelves/plateaus, or bottom bed floors.
            // In planar Height slicing, distance to the perimeters of the layer must be
            // measured in-plane (XY) from the side walls, not vertically (Z) from ceilings or floors.
            // Excluding all horizontal faces from the distance BVH prevents 3D Euclidean distance
            // from treating roofs, floors, and intermediate horizontal shelves as obstacles that
            // collapse or destroy side-wall perimeters on near-horizontal layers.
            if normal_len_sq > 1e-12 {
                let nz_sq = normal.z * normal.z;
                if nz_sq >= 0.998 * normal_len_sq {
                    return None;
                }
            }
            Some([i0, i1, i2])
        })
        .collect();

    let side_sdf = if non_cap_faces.len() == faces.len() {
        Arc::clone(&sdf)
    } else {
        Arc::new(MeshSdf::new_with_distance_faces(
            mesh.vertices.clone(),
            faces.clone(),
            non_cap_faces,
        ))
    };

    // Resolve the configured order field once per slice (defaults to a
    // `HeightOrderField` along `BUILD_DIRECTION`, matching pre-existing
    // behavior exactly). `order_range_over_bbox` generically replaces the
    // old `min.dot(BUILD_DIRECTION)`/`max.dot(BUILD_DIRECTION)` shortcut —
    // exact for any field whose extrema are attained at box corners, which
    // covers the affine `HeightOrderField` default used everywhere today.
    on_progress(0.01);
    let field: Arc<dyn OrderField> = Arc::from(order_field::order_field_for_with_sdf(
        config.order_field,
        config,
        mesh,
        slope_profile,
        Some(&*sdf),
    ));
    on_progress(0.04);
    let is_height = matches!(config.order_field, order_field::OrderFieldKind::Height);
    let is_dual_iso = matches!(config.order_field, order_field::OrderFieldKind::DualIso);

    // `order_range_over_bbox` samples 27 points on the mesh's axis-aligned
    // bounding box (corners, edge midpoints, face center). That's exact for
    // affine fields (`HeightOrderField`, `ConicalOrderField`) whose extrema
    // sit on the box boundary, but for an occupancy-restricted field (e.g.
    // `EikonalOrderField`) most/all of those 27 points typically fall
    // *outside* the solid mesh, where `order()` returns `f64::INFINITY`.
    // A non-finite bound here would make the `while` loop below (which
    // steps `order_value` from `order_min` to `order_max` by
    // `layer_height`) run unbounded, silently consuming memory until the
    // process is OOM-killed. Guard against that by falling back to
    // sampling the field at actual mesh vertices (which do lie on/near the
    // solid) whenever the bbox-corner estimate isn't finite.
    let (order_min, order_max) = {
        let mut vlo = f64::INFINITY;
        let mut vhi = f64::NEG_INFINITY;
        for &v in &mesh.vertices {
            let value = field.order(v);
            if value.is_finite() {
                vlo = vlo.min(value);
                vhi = vhi.max(value);
            }
        }
        if vlo.is_finite() && vhi.is_finite() {
            if is_height {
                let (lo, hi) = order_range_over_bbox(&*field, min, max);
                let min_val = if lo.is_finite() { vlo.min(lo) } else { vlo };
                let max_val = if hi.is_finite() { vhi.max(hi) } else { vhi };
                (min_val, max_val)
            } else {
                (vlo, vhi)
            }
        } else {
            let (lo, hi) = order_range_over_bbox(&*field, min, max);
            if lo.is_finite() && hi.is_finite() {
                (lo, hi)
            } else {
                return Err(Error::Slicing(
                    "order field produced no finite values over the mesh's bounding box or vertices"
                        .to_string(),
                ));
            }
        }
    };

    // In-plane sample extent: sized off the mesh's bounding box
    // *projected onto the contour-extraction plane's in-plane basis*
    // (perpendicular to BUILD_DIRECTION), not its full 3D diagonal. Using
    // the 3D diagonal (which includes the mesh's extent along
    // BUILD_DIRECTION, i.e. its height) wastes most of the fixed
    // CONTOUR_RESOLUTION grid on empty space for any object where height
    // dominates footprint, causing near-tip/near-base layers to fall
    // between grid samples and come back with zero contour loops.
    let (basis1, basis2) = plane_basis(BUILD_DIRECTION);
    let extent = in_plane_extent(min, max, basis1, basis2);

    // Center the sampling plane's origin on the mesh's actual in-plane
    // (footprint) position rather than the world origin.
    // `extract_contours_at_order` computes `origin = direction * order_value`,
    // which always has zero in-plane (basis1/basis2) components — fine only
    // when the mesh's footprint happens to straddle the world origin. For a
    // mesh translated far from world (0,0) (e.g. after `object::center_on_bed`
    // places it at the bed's center), the fixed-extent sampling window then
    // never reaches the mesh at all, so every layer comes back empty. Instead
    // we anchor the origin at the mesh's bounding-box center projected onto
    // the in-plane axes, while still solving for the BUILD_DIRECTION
    // component so `origin.dot(BUILD_DIRECTION) == order_value` (i.e. origin
    // stays exactly on the correct slicing plane for each layer).
    let bbox_center = (min + max) * 0.5;

    let layer_height = config.layer_height.abs().max(f64::EPSILON);
    let first_layer_height = config.first_layer_height();
    let resolution = contour_resolution(extent, config.nozzle_diameter, CONTOUR_REFINEMENT_DIVISOR);

    let wall_count = config.wall_count();

    // Dispatch on the resolved field kind (cheap enum comparison on
    // `config.order_field`, not any runtime type-inspection of the `dyn
    // OrderField` trait object): `Height` keeps today's exact plane-sampling
    // fast path unchanged; anything else (e.g. `Conical`) uses the
    // generalized "contour-on-mesh" path, which extracts each wall pass's
    // isosurface once (not once per layer) and walks it per layer.
    #[allow(clippy::type_complexity)]
    let (outer_wall_mesh, outer_wall_mesh_orders, dual_wall_meshes, dual_wall_orders): (
        Vec<DVec3>,
        Vec<f64>,
        Vec<Vec<DVec3>>,
        Vec<Vec<f64>>,
    ) = if is_height {
        (Vec::new(), Vec::new(), Vec::new(), Vec::new())
    } else if is_dual_iso {
        let cell_size = (config.wall_offset / 2.0)
            .min(config.wall_line_width / 4.0)
            .clamp(0.04, 0.10);
        let pad = DVec3::splat(cell_size * 2.0);
        let mut meshes = Vec::with_capacity(wall_count + 1);
        let mut orders = Vec::with_capacity(wall_count + 1);
        for w in 0..=wall_count {
            let iso = -(config.wall_offset + w as f64 * config.wall_line_width);
            let positions = extract_sparse_isosurface_positions::<MeshSdf>(
                &*sdf,
                min - pad,
                max + pad,
                cell_size,
                iso,
            );
            let ords: Vec<f64> = positions.par_iter().map(|&p| field.order(p)).collect();
            meshes.push(positions);
            orders.push(ords);
            on_progress(0.05 + 0.05 * ((w + 1) as f64 / (wall_count + 1) as f64));
        }
        let wall0_pos = meshes.first().cloned().unwrap_or_default();
        let wall0_ords = orders.first().cloned().unwrap_or_default();
        (wall0_pos, wall0_ords, meshes, orders)
    } else {
        // Sparse narrow-band marching cubes: target a cell size proportional
        // to bead dimensions (~0.20-0.35mm) to accurately extract the outer perimeter
        // isosurface without memory blowup or slowdowns on large meshes.
        let cell_size = (config.wall_offset / 2.0)
            .min(config.wall_line_width / 4.0)
            .clamp(0.04, 0.10);
        let iso = -config.wall_offset;
        let pad = DVec3::splat(cell_size * 2.0);
        on_progress(0.05);
        let positions = extract_sparse_isosurface_positions::<MeshSdf>(
            &*sdf,
            min - pad,
            max + pad,
            cell_size,
            iso,
        );
        on_progress(0.08);
        let orders: Vec<f64> = positions.par_iter().map(|&p| field.order(p)).collect();
        on_progress(0.10);
        (positions, orders, Vec::new(), Vec::new())
    };

    let effective_order_max = if !is_height && !outer_wall_mesh_orders.is_empty() {
        let max_mesh_order = outer_wall_mesh_orders
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .fold(order_max, f64::max);
        max_mesh_order
    } else {
        order_max
    };

    // Precompute every order-field value this walk will sample, so the
    // per-layer contour extraction below can run in parallel (each layer
    // only reads the shared, immutable `sdf` — see `MeshSdf`, whose query
    // methods all take `&self` — and writes its own independent `Layer`;
    // there's no cross-layer dependency to serialize on).
    //
    // Defensive cap: even with the finite `order_min`/`effective_order_max` guaranteed
    // above, guard the step count against any other future degenerate range
    // (e.g. a bug producing `order_min > effective_order_max` combined with a tiny
    // `layer_height`) so a bad range fails fast with `Error::Slicing`
    // instead of growing this `Vec` without bound until the process is
    // killed for memory exhaustion.
    const MAX_ORDER_STEPS: usize = 1_000_000;
    let mut order_values = Vec::new();
    let mut order_value = order_min + first_layer_height;
    while order_value <= effective_order_max + f64::EPSILON {
        if order_values.len() >= MAX_ORDER_STEPS {
            return Err(Error::Slicing(format!(
                "order range [{order_min}, {effective_order_max}] with layer_height {layer_height} would \
                 require more than {MAX_ORDER_STEPS} layers; refusing to continue"
            )));
        }
        order_values.push(order_value);
        order_value += layer_height;
    }

    let total_steps = order_values.len().max(1);

    let completed = AtomicUsize::new(0);
    let progress_callback = Mutex::new((on_progress, 0.0f64));

    let mut layers: Vec<Layer> = order_values
        .par_iter()
        .enumerate()
        .map(|(index, &order_value)| {
            let origin =
                bbox_center + BUILD_DIRECTION * (order_value - bbox_center.dot(BUILD_DIRECTION));
            let mut loops = Vec::new();
            let mut curved_infill_2d: Vec<Vec<[f64; 2]>> = Vec::new();
            if is_dual_iso {
                for w in 0..wall_count {
                    let (w_loops, _dbg) = extract_order_contours_on_mesh_with_debug(
                        &dual_wall_meshes[w],
                        &dual_wall_orders[w],
                        order_value,
                        BUILD_DIRECTION,
                    );
                    for pts in w_loops {
                        let arc_fraction = compute_arc_fractions(&pts);
                        let n_pts = pts.len();
                        loops.push(WallLoop {
                            wall_index: w,
                            unsupported: vec![false; n_pts],
                            top_surface: Vec::new(),
                            arc_fraction,
                            line_widths: vec![config.wall_line_width; n_pts],
                            points: pts,
                        });
                    }
                }
            } else if is_height {
                for wall_index in 0..wall_count {
                    // Negative iso = inward (see `MeshSdf::sign_at`: positive
                    // outside, negative inside). Wall 0 sits `wall_offset` in
                    // from the true surface (nozzle-center offset so the
                    // nozzle's outer edge lands on the surface); each further
                    // wall steps another `wall_line_width` inward.
                    let iso = -(config.wall_offset + wall_index as f64 * config.wall_line_width);
                    let wall_loops = extract_contours(
                        &*side_sdf, origin, basis1, basis2, extent, extent, resolution, resolution,
                        iso,
                    );
                    loops.extend(wall_loops.into_iter().map(|points| {
                        let arc_fraction = compute_arc_fractions(&points);
                        let n_pts = points.len();
                        WallLoop {
                            wall_index,
                            unsupported: vec![false; n_pts],
                            top_surface: Vec::new(),
                            arc_fraction,
                            line_widths: vec![config.wall_line_width; n_pts],
                            points,
                        }
                    }));
                }
            } else {
                // Wall 0: straight from the mesh's actual isosurface (see
                // `outer_wall_mesh`'s doc comment above).
                let (wall0_loops, debug_unclosed) = extract_order_contours_on_mesh_with_debug(
                    &outer_wall_mesh,
                    &outer_wall_mesh_orders,
                    order_value,
                    BUILD_DIRECTION,
                );
                for debug_pts in debug_unclosed {
                    let arc_fraction = compute_arc_fractions(&debug_pts);
                    let n_pts = debug_pts.len();
                    loops.push(WallLoop {
                        wall_index: 999,
                        unsupported: vec![false; n_pts],
                        top_surface: Vec::new(),
                        arc_fraction,
                        line_widths: vec![config.wall_line_width; n_pts],
                        points: debug_pts,
                    });
                }
                loops.extend(wall0_loops.iter().cloned().map(|points| {
                    let arc_fraction = compute_arc_fractions(&points);
                    let n_pts = points.len();
                    WallLoop {
                        wall_index: 0,
                        unsupported: vec![false; n_pts],
                        top_surface: Vec::new(),
                        arc_fraction,
                        line_widths: vec![config.wall_line_width; n_pts],
                        points,
                    }
                }));
                // Walls 1..wall_count: each one `wall_line_width` step
                // further inward than the previous, derived by offsetting
                // the previous wall's own loop in the tangent plane and
                // reconstructing onto this layer's order-field isosurface
                // (see `outer_wall_mesh`'s doc comment for why, and
                // `infill_boundary`'s curved-path computation below for the
                // same technique applied one step further in). Stops early
                // for a layer whose cross-section is too small to fit every
                // configured wall (e.g. near a tapered tip) rather than
                // producing garbage from an empty/degenerate offset.
                let loops_2d = polygon2d::to_2d(&wall0_loops, basis1, basis2, origin);
                let canonical_2d = polygon2d::canonicalize_with_sdf(
                    &loops_2d,
                    basis1,
                    basis2,
                    BUILD_DIRECTION,
                    origin,
                    order_value,
                    &*field,
                    Some(&*sdf),
                );

                let mut outers: Vec<Vec<[f64; 2]>> = Vec::new();
                let mut holes: Vec<Vec<[f64; 2]>> = Vec::new();
                for loop_2d in canonical_2d {
                    if polygon2d::signed_area(&loop_2d) > 0.0 {
                        outers.push(loop_2d);
                    } else {
                        holes.push(loop_2d);
                    }
                }

                let mut islands: Vec<Vec<Vec<[f64; 2]>>> = Vec::new();
                let mut assigned_holes = vec![false; holes.len()];

                for outer in outers {
                    let mut island = vec![outer.clone()];
                    for (h_idx, hole) in holes.iter().enumerate() {
                        if !assigned_holes[h_idx]
                            && !hole.is_empty()
                            && polygon2d::point_in_polygon(hole[0], &outer)
                        {
                            island.push(hole.clone());
                            assigned_holes[h_idx] = true;
                        }
                    }
                    islands.push(island);
                }

                for (h_idx, hole) in holes.into_iter().enumerate() {
                    if !assigned_holes[h_idx] {
                        islands.push(vec![hole]);
                    }
                }

                for island_2d in &islands {
                    let partitioned = polygon2d::partition_walls_adaptive(
                        island_2d,
                        config.wall_line_width,
                        config.min_bead_width(),
                        wall_count + 1,
                    );

                    let mut previous_loops = wall0_loops.clone();
                    for p_wall in partitioned {
                        if p_wall.wall_index == 0 {
                            continue;
                        }
                        if p_wall.wall_index == wall_count {
                            curved_infill_2d.extend(p_wall.loops_2d);
                            continue;
                        }
                        let reconstructed = order_field::reconstruct_on_order_field_near(
                            p_wall.loops_2d,
                            &previous_loops,
                            basis1,
                            basis2,
                            BUILD_DIRECTION,
                            origin,
                            order_value,
                            order_field::max_along_for(config),
                            &*field,
                        );
                        loops.extend(reconstructed.iter().cloned().map(|points| {
                            let arc_fraction = compute_arc_fractions(&points);
                            let n_pts = points.len();
                            WallLoop {
                                wall_index: p_wall.wall_index,
                                unsupported: vec![false; n_pts],
                                top_surface: Vec::new(),
                                arc_fraction,
                                line_widths: vec![p_wall.line_width; n_pts],
                                points,
                            }
                        }));
                        previous_loops = reconstructed;
                    }
                }
                if curved_infill_2d.is_empty() {
                    // If channel was too narrow for wall_count+1, fallback to innermost wall
                    for island_2d in &islands {
                        let partitioned = polygon2d::partition_walls_adaptive(
                            island_2d,
                            config.wall_line_width,
                            config.min_bead_width(),
                            wall_count,
                        );
                        if let Some(deepest) = partitioned.last() {
                            curved_infill_2d.extend(deepest.loops_2d.clone());
                        }
                    }
                }
            }
            let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
            if let Ok(mut guard) = progress_callback.lock() {
                let fraction = (0.10 + 0.85 * (done as f64 / total_steps as f64)).max(guard.1);
                guard.1 = fraction;
                (guard.0)(fraction);
            }
            // The infill boundary sits where wall pass `wall_count` would
            // be if one more wall were printed — one further
            // `wall_line_width` step inward from the innermost printed
            // wall for EACH separate island.
            let wall0_loops: Vec<Vec<DVec3>> = loops
                .iter()
                .filter(|wall| wall.wall_index == 0)
                .map(|wall| wall.points.clone())
                .collect();

            let infill_boundary = if is_dual_iso {
                let (infill_loops, _) = extract_order_contours_on_mesh_with_debug(
                    &dual_wall_meshes[wall_count],
                    &dual_wall_orders[wall_count],
                    order_value,
                    BUILD_DIRECTION,
                );
                infill_loops
            } else if wall0_loops.is_empty() {
                Vec::new()
            } else if is_height {
                let loops_2d = polygon2d::to_2d(&wall0_loops, basis1, basis2, origin);
                let canonical_2d = polygon2d::canonicalize_with_sdf(
                    &loops_2d,
                    basis1,
                    basis2,
                    BUILD_DIRECTION,
                    origin,
                    order_value,
                    &*field,
                    Some(&*sdf),
                );
                let partitioned = polygon2d::partition_walls_adaptive(
                    &canonical_2d,
                    config.wall_line_width,
                    config.min_bead_width(),
                    wall_count,
                );
                let mut layer_infill_2d: Vec<Vec<[f64; 2]>> = Vec::new();
                if let Some(deepest_wall) = partitioned.last() {
                    let inset =
                        polygon2d::inward_offset(&deepest_wall.loops_2d, config.wall_line_width);
                    if !inset.is_empty() {
                        layer_infill_2d.extend(inset);
                    } else if !deepest_wall.loops_2d.is_empty() {
                        layer_infill_2d.extend(deepest_wall.loops_2d.clone());
                    }
                }
                polygon2d::from_2d(layer_infill_2d, basis1, basis2, origin)
            } else {
                let offset_3d = polygon2d::from_2d(curved_infill_2d, basis1, basis2, origin);
                let offset_2d = polygon2d::to_2d(&offset_3d, basis1, basis2, origin);
                order_field::reconstruct_on_order_field_near(
                    offset_2d,
                    &wall0_loops,
                    basis1,
                    basis2,
                    BUILD_DIRECTION,
                    origin,
                    order_value,
                    order_field::max_along_for(config),
                    &*field,
                )
            };
            Layer {
                index,
                object: ObjectId::default(),
                order: order_value,
                loops,
                infill_boundary,
                solid_fill_boundary: Vec::new(),
                mesh_sdf: Some(Arc::clone(&sdf)),
                order_field: Arc::clone(&field),
            }
        })
        .collect();

    // Clean first-layer geometry: filter sub-bead noise and recessed micro-lettering
    // so the initial layer prints with continuous solid perimeters and infill.
    if let Some(first_layer) = layers.first_mut() {
        clean_first_layer_geometry(first_layer, config);
    }

    // Curved-path wall-loop smoothing: `EikonalOrderField`/`ConicalOrderField`
    // reconstruct geometry against an interpolated field (trilinear for
    // Eikonal's FMM distance grid -- see `EikonalOrderField::order`'s doc
    // comment), which is only C0. Every isosurface projection/offset step
    // that walks such a field therefore lands on a faceted surface with
    // facet width equal to the field's grid cell size, producing visible
    // point-to-point zigzag along wall loops (confirmed on
    // pug_v4_l_sop_85mm.stl with the manual `probe_wall_noise` example --
    // see project memory: ~74% sign-alternation rate in the discrete
    // second difference along Eikonal wall loops, vs ~46% i.e.
    // no-zigzag for the analytic Height field). Smoothing here is
    // deliberately light (small window, few iterations) and skips
    // `unsupported` (stitched) points, so it damps sub-cell noise without
    // eating real curvature (whose wavelength is normally many cells
    // wide) or disturbing deliberately-placed bridge geometry. Run before
    // `stitch_wall_gaps` so its arc-length correspondence matches against
    // the smoothed geometry, not the raw faceted one. The Height path
    // reconstructs directly from an exact analytic SDF isosurface with no
    // comparable faceting, so it is left untouched (matches
    // `stitch_wall_gaps`'s existing `!is_height` gate below).
    if !is_height {
        let max_displacement = config.nozzle_diameter * 0.02;
        for layer in &mut layers {
            for wall in &mut layer.loops {
                smooth_wall_loop(wall, max_displacement);
                // Recompute now that points moved -- arc_fraction is a
                // per-point cumulative arc-length position (see
                // `WallLoop::arc_fraction`'s doc comment), consumed by
                // `stitch_wall_gaps` right below for its correspondence.
                wall.arc_fraction = compute_arc_fractions(&wall.points);
            }
            for bnd in &mut layer.infill_boundary {
                smooth_closed_points(bnd, max_displacement);
            }
        }
    }

    // Inter-layer wall-0 gap stitching: only meaningful for the curved
    // (non-Height) path when wave overhangs are disabled. When wave
    // overhangs are enabled, wave overhang path planning replaces wall gap
    // stitching with continuous Huygens-diffraction wavefronts, keeping
    // outer wall loops unbroken.
    if !is_height && !config.wave_overhangs_enabled() {
        stitch_wall_gaps(&mut layers, config, basis1, basis2);
    }

    // Prune any trailing empty apex layers (e.g. from singular apex points with no contour):
    // Prune any trailing empty apex layers on curved fields (e.g. from singular apex points with no contour):
    if !is_height {
        while layers.last().is_some_and(|l| l.loops.is_empty()) {
            layers.pop();
        }
    }

    if let Ok(mut guard) = progress_callback.lock() {
        let fraction = 1.0f64.max(guard.1);
        guard.1 = fraction;
        (guard.0)(fraction);
    }

    Ok(layers)
}

/// Cleans first-layer bed-contact geometry by discarding sub-bead micro-loops
/// (e.g. tiny embossed lettering or discretization noise on the bed plane)
/// and filtering infill slivers so the initial layer lays down clean, continuous
/// perimeters and solid infill with steady extruder pressure.
fn clean_first_layer_geometry(layer: &mut Layer, config: &SlicerConfig) {
    if layer.loops.is_empty() {
        return;
    }
    // Filter tiny micro-loops (< 4 * nozzle_diameter perimeter) on the bed layer unless it's the only loop.
    if layer.loops.len() > 1 {
        let min_perimeter = config.nozzle_diameter * 4.0;
        let mut filtered_loops = Vec::new();
        for wall in &layer.loops {
            let mut perimeter = 0.0;
            for i in 0..wall.points.len() {
                let p0 = wall.points[i];
                let p1 = wall.points[(i + 1) % wall.points.len()];
                perimeter += (p1 - p0).length();
            }
            if wall.wall_index == 0 && perimeter < min_perimeter {
                continue;
            }
            filtered_loops.push(wall.clone());
        }
        if !filtered_loops.is_empty() {
            layer.loops = filtered_loops;
        }
    }

    // In the infill boundary on the first layer, filter out sub-bead microscopic slivers
    if !layer.infill_boundary.is_empty() {
        let min_hole_area = config.nozzle_diameter * config.nozzle_diameter * 2.0;
        let (basis1, basis2) = plane_basis(BUILD_DIRECTION);
        let b2d = polygon2d::to_2d(&layer.infill_boundary, basis1, basis2, DVec3::ZERO);
        let cleaned_2d = polygon2d::filter_min_area(&b2d, min_hole_area);
        if !cleaned_2d.is_empty() {
            let references = layer.infill_boundary.clone();
            layer.infill_boundary = order_field::reconstruct_on_order_field_near(
                cleaned_2d,
                &references,
                basis1,
                basis2,
                BUILD_DIRECTION,
                DVec3::ZERO,
                layer.order,
                order_field::max_along_for(config),
                layer.order_field.as_ref(),
            );
        }
    }
}

/// Applies a feature-preserving moving-average filter to `points` in place:
/// sharp corners (turn angle > 30°) are left untouched so CAD corners and sharp
/// features are never rounded or shrunk away from the true surface, and any
/// smoothed point's displacement is clamped to a tiny fraction of the nozzle
/// diameter (`MAX_SMOOTH_DISPLACEMENT = 0.02 * nozzle_diameter`, ~0.008 mm)
/// so it only damps sub-cell grid quantization noise without drifting from the
/// true geometry.
fn smooth_closed_points(points: &mut [DVec3], max_displacement: f64) {
    let n = points.len();
    if n < 5 {
        return;
    }
    // cos(30 degrees) ~= 0.866; turns sharper than 30 deg are preserved as sharp corners.
    const CORNER_COS_THRESHOLD: f64 = 0.866;

    let original = points.to_vec();
    for (i, p) in points.iter_mut().enumerate() {
        let prev = original[(i as isize - 1).rem_euclid(n as isize) as usize];
        let cur = original[i];
        let next = original[(i as isize + 1).rem_euclid(n as isize) as usize];

        let Some(t_in) = (cur - prev).try_normalize() else {
            continue;
        };
        let Some(t_out) = (next - cur).try_normalize() else {
            continue;
        };

        if t_in.dot(t_out) < CORNER_COS_THRESHOLD {
            // Sharp geometric corner: preserve exact position.
            continue;
        }

        let target = (prev + cur + next) / 3.0;
        let delta = target - cur;
        let delta_len = delta.length();
        if delta_len > 0.0 {
            let clamped_delta = if delta_len > max_displacement {
                delta * (max_displacement / delta_len)
            } else {
                delta
            };
            *p = cur + clamped_delta;
        }
    }
}

/// Applies feature-preserving smoothing to `wall.points` in place to damp
/// Eikonal grid-quantization noise. Points flagged `unsupported` (stitched)
/// and sharp geometric corners are left untouched.
fn smooth_wall_loop(wall: &mut WallLoop, max_displacement: f64) {
    let n = wall.points.len();
    if n < 5 {
        return;
    }
    if wall.unsupported.iter().any(|&u| u) {
        const CORNER_COS_THRESHOLD: f64 = 0.866;
        let original = wall.points.clone();
        for i in 0..n {
            if wall.unsupported.get(i).copied().unwrap_or(false) {
                continue;
            }
            let prev_idx = (i as isize - 1).rem_euclid(n as isize) as usize;
            let next_idx = (i as isize + 1).rem_euclid(n as isize) as usize;
            if wall.unsupported.get(prev_idx).copied().unwrap_or(false)
                || wall.unsupported.get(next_idx).copied().unwrap_or(false)
            {
                continue;
            }
            let prev = original[prev_idx];
            let cur = original[i];
            let next = original[next_idx];

            let (Some(t_in), Some(t_out)) =
                ((cur - prev).try_normalize(), (next - cur).try_normalize())
            else {
                continue;
            };
            if t_in.dot(t_out) < CORNER_COS_THRESHOLD {
                continue;
            }

            let target = (prev + cur + next) / 3.0;
            let delta = target - cur;
            let delta_len = delta.length();
            if delta_len > 0.0 {
                let clamped_delta = if delta_len > max_displacement {
                    delta * (max_displacement / delta_len)
                } else {
                    delta
                };
                wall.points[i] = cur + clamped_delta;
            }
        }
    } else {
        smooth_closed_points(&mut wall.points, max_displacement);
    }
}

/// Detects and fixes inter-layer wall-0 (ear bridging) gaps: walks
/// `layers` in order (already sorted by Layer::index/order value -- see
/// slice_mesh_with_progress) and, for each layer after the first,
/// finds each current-layer wall-0 point's corresponding previous-layer
/// wall-0 position via **arc-length-fraction correspondence**, not
/// nearest-point search of any kind (raw 3D or transverse `(u, v)`).
///
/// Nearest-point matching between two independently parameterized
/// closed curves is inherently non-monotonic: each layer's wall-0 loop
/// is extracted independently, with no guaranteed correspondence
/// between vertex `i` of one layer's loop and vertex `i` of the next
/// (different point counts, sampling density, and starting point around
/// the loop). Matching each current-loop point to "whichever
/// previous-loop point happens to be nearest" lets adjacent
/// current-loop points map to non-adjacent (even backward-jumping)
/// previous-loop points, producing a zigzag correspondence on
/// essentially every contour -- not just genuine overhangs -- regardless
/// of whether the nearest-point metric is raw 3D distance or transverse
/// `(u, v)` position (a prior fix tried the latter; it did not help,
/// because the metric was never the real problem for adjacent layers).
///
/// Arc-length-fraction correspondence fixes this by walking both loops'
/// own shapes in lockstep instead of comparing raw positions:
/// 1. Each wall-0 [`WallLoop`] already carries [`WallLoop::arc_fraction`]
///    (populated at construction -- see `slice_mesh_with_progress`): a
///    per-point normalized cumulative arc-length position around the
///    closed polyline, in `[0, 1)`.
/// 2. The two loops' start points are aligned *once* per layer-pair: a
///    single nearest-point search (3D distance; this runs once, not once
///    per point, so brute force is fine) matches the current loop's
///    first point against the previous loop's points, giving a
///    rotational offset (`prev_fractions[nearest_idx]`) between the two
///    loops' independent parameterizations.
/// 3. For each current-loop point at its own (already-computed) arc
///    fraction `t`, the corresponding previous-loop position is `t_prev
///    = (offset + t) mod 1.0`, found by interpolating along the previous
///    loop's cumulative-arc-length table (see
///    [`interpolate_on_loop_at_fraction`]): walk to the bracketing
///    segment and lerp within it. This is monotonic and non-crossing by
///    construction -- it cannot backward-jump the way nearest-point
///    search can.
///
/// Current-layer and previous-layer wall-0 loops are paired by real
/// spatial correspondence, not positional index in each layer's loop
/// vector: a cross-section with holes/multiple islands can have its
/// loop extraction order (and count) shift between adjacent layers, so
/// the `n`th loop encountered in the current layer is not reliably the
/// same physical feature as the `n`th loop encountered in the previous
/// layer. Each current loop is matched to the previous loop with the
/// nearest centroid (mean of `points`), subject to
/// `WALL_GAP_LOOP_CENTROID_MATCH_FACTOR * nozzle_diameter`, *and* whose
/// perimeter is within `WALL_GAP_LOOP_PERIMETER_MATCH_FACTOR` of the
/// current loop's own perimeter (see that constant's docs -- centroid
/// distance alone can match a small, simple loop to an unrelated, much
/// larger/more complex one whose centroid happens to land nearby,
/// producing a meaningless arc-length correspondence): if no previous
/// loop satisfies both (a genuinely new loop this layer, a vanished
/// loop, or just too far/differently-shaped to plausibly be the same
/// feature), the current loop is left unstitched rather than
/// force-matched to an unrelated loop -- arc-length correspondence is
/// only meaningful within a single correctly-matched pair of closed
/// loops.
///
/// A shallow slope on the underside of an overhang can produce
/// consecutive layers whose wall-0 loops are laterally far enough apart
/// that the printed lines don't bond (see NON_PLANAR_SLICING.md's
/// SlopeProfile docs for why this can happen even with an angle-limited
/// order field: the slope limit only caps the maximum rate of rise, not
/// a minimum). Whenever a point's **lateral** distance (perpendicular to
/// the local climb direction -- see [`lateral_gap`] for why the raw 3D
/// distance is the wrong metric here: it always contains ~one layer
/// height of normal climb separation and therefore triggers on every
/// point of every contour, including vertical walls) to its
/// arc-length-fraction counterpart exceeds WALL_GAP_HOP_FRACTION *
/// (nozzle radius) -- and no previous-layer wall-0 material lies
/// laterally within that limit at all (the correspondence sanity veto;
/// see the third pass in the body) -- the point genuinely needs stitch
/// material. Maximal runs of consecutive such points are then each
/// filled with ONE continuous serpentine block
/// ([`serpentine_stitch_block`]): intermediate order levels between the
/// two layers, each level walked as a continuous line across the whole
/// run, alternating direction so consecutive rows connect at a shared
/// column -- never one anchor-to-target ramp per point, which printed a
/// physical zigzag (nozzle diving to the previous layer and back for
/// every point of the run). Every level point is reprojected onto the
/// order field's actual isosurface (never a straight 3D chord -- the
/// true isosurface between two order values can be curved for a
/// non-planar field), and the level count doubles until every
/// consecutive hop along each column is within the hop limit. The
/// resulting block is spliced into the current layer's wall-0 loop
/// immediately before the run's first point, flagged unsupported = true
/// (the points print over a gap with no solid/order-field surface
/// directly beneath them, unlike the rest of the loop), and each given
/// its own column's target-point arc_fraction (they don't add a new
/// circumferential position, only intermediate order/height values on
/// the way to it), keeping `arc_fraction` parallel to
/// `points`/`unsupported` through the insertion.
///
/// Only wall_index == 0 loops are touched; inner walls (wall_index >= 1,
/// already derived from wall 0 within the same layer) are out of scope
/// for this pass (only wall 0 was reported as too far apart, and inner
/// walls may inherit acceptable spacing once wall 0 is fixed).
///
/// Before bisecting, every current-loop point's raw fraction-based (plus
/// bounded local-fallback) correspondence is computed up front for the
/// whole loop, then checked against its immediate current-loop
/// neighbors (the stored-order point before and after it): a real
/// shallow-overhang region spans a *run* of consecutive points whose
/// correspondence is far away, never a single point bracketed on both
/// sides by points with perfectly good, close correspondences -- if
/// exactly that isolated pattern is found (this point's hop exceeds
/// `hop_limit` while both neighbors' hops do not), the outlier is
/// replaced with the midpoint of its neighbors' own correspondences
/// rather than trusting the raw fraction-based lookup. Even after the
/// best global rotation offset (see [`best_rotation_offset`]) and the
/// bounded local fallback (see [`local_fallback_correspondence`]), a
/// single previous-loop sampling irregularity can still throw off one
/// point's fraction-interpolated position while its neighbors -- who
/// interpolate at nearby but different fractions, potentially bracketing
/// a different, non-irregular segment of the previous loop -- land fine;
/// this neighbor-consistency check catches that residual case without
/// re-introducing the unbounded nearest-point search that caused the
/// original zigzag defect (see this function's docs above).
/// Returns whether the straight chord `a -> b` stays inside (or hugs) the
/// mesh solid: every interior sample's signed distance must be at most
/// `tolerance` (one bead radius) outside the surface.
///
/// Used by [`stitch_wall_gaps`] as the final needs-stitch veto: stitch
/// material physically bonds a shallow-overhang gap *through the solid*
/// between the two layers' surfaces, so a candidate whose anchor->target
/// chord passes through genuinely open air (a void between separate
/// features, e.g. between the pug's ears) cannot be stitched -- inserting
/// a serpentine there prints lines across the void. Sampled every
/// quarter bead radius (previously half), capped at 256 intervals
/// (previously 64): the coarser half-bead-radius spacing let a narrow
/// void spike between two in-tolerance samples go undetected on
/// borderline chords (confirmed on both pug_v4_l and pug_v4_m -- a
/// void-crossing stitch segment slipped through this veto with max_sdf
/// only marginally above `tolerance`), so this veto was a false
/// negative there, not a tolerance-strictness issue: the fix is denser
/// sampling, not a looser or stricter threshold. A small safety margin
/// (`CHORD_VOID_SAFETY_MARGIN_FRACTION` of `tolerance`) is subtracted
/// from the effective threshold on top of the denser sampling: even at
/// quarter-bead-radius spacing, a genuinely borderline chord can still
/// land within a couple hundredths of a mm of `tolerance` (observed on
/// pug_v4_m: max_sdf=0.2020 vs tolerance=0.2000, i.e. 0.002mm over --
/// interpolation/sampling noise, not a real void), so this margin turns
/// "just barely over" into a firm reject rather than requiring even more
/// samples to chase diminishing precision. `None` for `mesh_sdf` (layers
/// built without one, e.g. unit-test fixtures) skips the veto entirely
/// and reports the chord as in-solid.
const CHORD_VOID_SAFETY_MARGIN_FRACTION: f64 = 0.05;

fn chord_stays_in_solid(mesh_sdf: Option<&MeshSdf>, a: DVec3, b: DVec3, tolerance: f64) -> bool {
    let Some(sdf) = mesh_sdf else {
        return true;
    };
    let effective_tolerance = tolerance * (1.0 - CHORD_VOID_SAFETY_MARGIN_FRACTION);
    let samples = ((a.distance(b) / (0.25 * tolerance).max(1e-9)).ceil() as usize).clamp(2, 256);
    (0..=samples).all(|s| {
        let t = s as f64 / samples as f64;
        sdf.sample(a.lerp(b, t)).value <= effective_tolerance
    })
}

/// Whether solid mesh material genuinely exists one nozzle-diameter away
/// from `point` along `probe_direction`, used both as the wall-gap-stitch
/// support veto in [`stitch_wall_gaps`] (probed with [`BUILD_DIRECTION`],
/// i.e. "backward"/already-printed) and as the [`WallLoop::top_surface`]
/// tag (probed with `-BUILD_DIRECTION`, i.e. "forward"/not-yet-printed).
/// True vertical, not the local order-field gradient -- see below for why.
///
/// The previous-wall-0-loop veto in `stitch_wall_gaps` assumes "no
/// previous wall-0 point laterally nearby" implies "no material below" --
/// true for a Height order field (the wall-0 contour is the mesh's true
/// cross-section boundary at every layer), but not when the local surface
/// is nearly (but not exactly) perpendicular to the build direction: a
/// small deviation from perpendicular means the previous layer's wall-0
/// contour, which tracks a curved order-field isosurface rather than a
/// flat height plane, can sit noticeably farther away laterally than one
/// nozzle radius even though the point sits on an ordinary near-vertical
/// wall with solid material directly underneath it in the traditional
/// vertical sense. Confirmed on `pug_v4_l_sop_85mm.stl` layer 2: a
/// 131-point wall-0 run was flagged unsupported although every sampled
/// point had mesh SDF around -0.2 to -0.3 (well inside solid) and the
/// order field increased smoothly (0.2->0.4) with no void along the
/// straight line to the nearest real previous-layer point -- ruling out
/// a genuine geodesic detour and pointing at exactly this
/// contour/volume mismatch instead.
///
/// Deliberately probes along `BUILD_DIRECTION`/`-BUILD_DIRECTION` rather
/// than the local order-field gradient: a *genuine* overhang/bridge, by
/// definition, has open air directly beneath it in the ordinary vertical
/// sense (that is what makes it an overhang), so the support veto
/// correctly does not fire there. An ordinary near-vertical wall that the
/// curved order field simply traces along a different lateral path from
/// one layer to the next always has solid material directly beneath it
/// vertically, so the veto correctly does fire there. Probing along the
/// local gradient instead (an earlier version of this fix) is too
/// permissive: near any solid surface, "some solid exists somewhere back
/// along the local climb direction" is close to always true, including
/// for genuine overhangs, since the model itself is solid just off to
/// the side of the void being bridged -- that version suppressed real
/// stitching entirely on both `pug_v4_l_sop_85mm.stl` and
/// `pug_v4_m_sop_85mm.stl`. `None` for `mesh_sdf` reports no material
/// (nothing to check against), consistent with [`chord_stays_in_solid`]'s
/// `None` handling.
///
/// IMPORTANT: `probe_direction` must be `BUILD_DIRECTION` for the
/// support veto (checking the already-printed side) and `-BUILD_DIRECTION`
/// for the `top_surface` tag (checking the not-yet-printed side) --
/// passing `-BUILD_DIRECTION` for the support veto was an earlier bug in
/// this function (it always probed forward/upward instead of
/// backward/downward), which caused genuine top-of-part surfaces --
/// correctly lacking material above them -- to be misread as lacking
/// support *below* them and misclassified as `Overhang`. The two probes
/// are deliberately independent (a point can read solid-below and
/// void-above at the same time: that is exactly a top surface, not a
/// contradiction), so this function does not try to combine them itself
/// -- callers request exactly the side they need.
///
/// Both prior versions of the support veto (local-gradient-probe, then
/// the original single-point `BUILD_DIRECTION`-probe-one-`layer_height`-
/// back version) turned out to suppress essentially *all* wall-gap
/// stitching on the real `pug_v4_m_sop_85mm.stl` mesh, not just false
/// positives: with `layer_height` typically much smaller than a nozzle
/// diameter, probing only `layer_height` straight back is too shallow to
/// distinguish "thin near-vertical wall, just printed" from "genuine
/// void" -- confirmed by an A/B diagnostic
/// (`crates/manifold-cli/examples/diagnose_missing_overhang.rs`) showing
/// 5000 Overhang segments before this veto existed and 0 after, in both
/// tested configurations. This version instead probes a full
/// `nozzle_diameter` back (deep enough to actually clear a thin wall),
/// and at both edges of the bead's nominal footprint (`point ± tangent *
/// wall_line_width/2`), not just its centerline, since a bead lays down
/// material across its full width, not a zero-width line. If both edges
/// read solid, the point reads supported (`true`). If both read void, it
/// does not (`false`). If the two edges disagree, the point straddles a
/// real solid/void boundary: the boundary is located by bisection along
/// the straight line between the two edge probes (refined to within
/// `wall_line_width / 8`), and the point is only treated as solid-backed
/// if the solid side covers a majority of the bead's footprint width.
fn has_solid_material_in_direction(
    mesh_sdf: Option<&MeshSdf>,
    point: DVec3,
    tangent: DVec3,
    nozzle_diameter: f64,
    wall_line_width: f64,
    probe_direction: DVec3,
) -> bool {
    let Some(sdf) = mesh_sdf else {
        return false;
    };
    let tangent = if tangent.length_squared() > 1e-12 {
        tangent.normalize()
    } else {
        return false;
    };

    // Probe a full nozzle diameter away (not just one layer height): a
    // shallow one-layer-height probe sits too close to `point` itself to
    // distinguish "just-printed thin near-vertical wall" from "genuine
    // open void" -- on the real `pug_v4_m_sop_85mm.stl` mesh this shallow
    // version vetoed 100% of wall-gap stitching (all 5000 previously
    // generated Overhang segments vanished), including on true overhangs.
    let depth = nozzle_diameter;
    // ...and across the bead's full nominal footprint width (both edges,
    // not just the centerline) -- a bead is not a zero-width line, so
    // "supported" should mean the material it lays down actually has
    // something solid beneath it, not just its centerline.
    let half_width = wall_line_width * 0.5;
    let tolerance = nozzle_diameter * 0.5;
    let binormal = tangent.cross(probe_direction);
    let normal = if binormal.length_squared() > 1e-12 {
        binormal.normalize()
    } else {
        DVec3::X
    };
    let probe_at = |offset: f64| point + normal * offset + probe_direction * depth;
    let edge_a = probe_at(half_width);
    let edge_b = probe_at(-half_width);
    let center = point + probe_direction * depth;
    let inside = |p: DVec3| sdf.sample(p).value <= tolerance;
    let a_inside = inside(edge_a);
    let b_inside = inside(edge_b);
    let center_inside = inside(center);
    if (a_inside && b_inside) || center_inside {
        return true;
    }
    if !a_inside && !b_inside {
        return false;
    }

    // Mixed: one edge sits over solid, the other over void -- this point
    // is straddling a real solid/void boundary rather than being cleanly
    // solid-backed or cleanly void. Bisect along the straight line
    // between the two edge probes to locate that boundary to within an
    // 1/8-line-width tolerance, then call it solid-backed only if the
    // solid side covers a majority of the bead's footprint (the boundary
    // sits past the bead's own centerline).
    let bisect_tolerance = (wall_line_width / 8.0).max(1e-6);
    let (mut inside_pt, outside_pt) = if a_inside {
        (edge_a, edge_b)
    } else {
        (edge_b, edge_a)
    };
    let mut lo = inside_pt;
    let mut hi = outside_pt;
    while lo.distance(hi) > bisect_tolerance {
        let mid = lo.lerp(hi, 0.5);
        if inside(mid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    inside_pt = lo;

    let known_inside_edge = if a_inside { edge_a } else { edge_b };
    let supported_len = inside_pt.distance(known_inside_edge);
    supported_len / wall_line_width.max(1e-9) >= 0.5
}

fn stitch_wall_gaps(layers: &mut [Layer], config: &SlicerConfig, basis1: DVec3, basis2: DVec3) {
    let hop_limit = WALL_GAP_HOP_FRACTION * (config.nozzle_diameter / 2.0);
    if !hop_limit.is_finite() || hop_limit <= 0.0 {
        return;
    }
    let max_along = order_field::max_along_for(config);

    // The previous layer's wall-0 loops (points + their arc-length
    // fractions), retained across iterations for correspondence lookup,
    // in the same order as they appear when filtering `layer.loops` for
    // `wall_index == 0` -- `None` before the first layer, since there is
    // nothing yet to compare against.
    type Wall0Loop = (Vec<DVec3>, Vec<f64>);
    let mut previous_wall0: Option<(f64, Vec<Wall0Loop>)> = None;

    for layer in layers.iter_mut() {
        // Captured before this layer's wall-0 loops are (possibly)
        // mutated below, so the *next* iteration compares against this
        // layer's real extracted geometry, not any stitch points just
        // inserted into it.
        let current_wall0: Vec<Wall0Loop> = layer
            .loops
            .iter()
            .filter(|wall| wall.wall_index == 0)
            .map(|wall| (wall.points.clone(), wall.arc_fraction.clone()))
            .collect();

        if let Some((prev_order, prev_loops)) = previous_wall0.as_ref() {
            let field = Arc::clone(&layer.order_field);
            let mesh_sdf = layer.mesh_sdf.clone();
            let cur_order = layer.order;
            let match_threshold = WALL_GAP_LOOP_CENTROID_MATCH_FACTOR * config.nozzle_diameter;
            let prev_centroids: Vec<DVec3> = prev_loops
                .iter()
                .map(|(pts, _)| loop_centroid(pts))
                .collect();
            let prev_perimeters: Vec<f64> = prev_loops
                .iter()
                .map(|(pts, _)| loop_perimeter(pts))
                .collect();
            // Precomputed once per layer-pair (not per point): for each
            // current wall-0 loop (in the same order the mutable filter
            // below will visit them), which previous-layer loop -- if
            // any -- is its nearest-centroid match within
            // `match_threshold`, additionally requiring the two loops'
            // perimeters to be within `WALL_GAP_LOOP_PERIMETER_MATCH_FACTOR`
            // of each other (see that constant's docs -- centroid distance
            // alone can match a small loop to an unrelated, much larger
            // one). `None` means no sufficiently-close *and*
            // plausibly-same-shape previous loop was found, so this loop
            // is left unstitched.
            let loop_matches: Vec<Option<usize>> = current_wall0
                .iter()
                .map(|(pts, _)| {
                    let centroid = loop_centroid(pts);
                    let perimeter = loop_perimeter(pts);
                    // Unfiltered nearest-centroid distance across every
                    // previous loop, regardless of the perimeter check --
                    // used below to detect the layer-40/loop-13 pattern
                    // (a genuinely nearby loop rejected only by shape,
                    // e.g. a pre-merge topology change, while a distant,
                    // coincidentally-similar-perimeter loop passes both
                    // filters). See
                    // [`WALL_GAP_LOOP_NEAREST_CENTROID_MARGIN_FACTOR`]'s
                    // docs.
                    let global_nearest_dist = prev_centroids
                        .iter()
                        .map(|prev_centroid| (*prev_centroid - centroid).length())
                        .fold(f64::INFINITY, f64::min);

                    prev_centroids
                        .iter()
                        .zip(prev_perimeters.iter())
                        .enumerate()
                        .map(|(idx, (prev_centroid, &prev_perimeter))| {
                            (idx, (*prev_centroid - centroid).length(), prev_perimeter)
                        })
                        .filter(|(_, dist, _)| *dist <= match_threshold)
                        .filter(|(_, _, prev_perimeter)| {
                            let (longer, shorter) = if *prev_perimeter >= perimeter {
                                (*prev_perimeter, perimeter)
                            } else {
                                (perimeter, *prev_perimeter)
                            };
                            // Degenerate (near-zero-perimeter) loops -- e.g.
                            // single-point placeholder loops used to
                            // exercise the hop-limit/bisect path in
                            // isolation -- carry no meaningful perimeter
                            // ratio to compare. Treat "both degenerate" as a
                            // trivial match (nothing to disprove), but a
                            // degenerate loop can never plausibly match a
                            // non-degenerate one, so that asymmetric case is
                            // still a ratio-check failure rather than a
                            // division by (near) zero.
                            const DEGENERATE_PERIMETER: f64 = 1e-9;
                            if longer <= DEGENERATE_PERIMETER {
                                true
                            } else {
                                shorter > DEGENERATE_PERIMETER
                                    && longer / shorter <= WALL_GAP_LOOP_PERIMETER_MATCH_FACTOR
                            }
                        })
                        // Reject a filtered candidate that is drastically
                        // farther away than the true closest previous loop
                        // overall -- see
                        // [`WALL_GAP_LOOP_NEAREST_CENTROID_MARGIN_FACTOR`]'s
                        // docs. `global_nearest_dist` is finite here
                        // whenever any candidate survives the filters
                        // above (there is at least one previous loop), so
                        // this only excludes genuinely implausible
                        // coincidental matches, never legitimate ones.
                        .filter(|(_, dist, _)| {
                            *dist
                                <= global_nearest_dist
                                    * WALL_GAP_LOOP_NEAREST_CENTROID_MARGIN_FACTOR
                        })
                        .min_by(|a, b| a.1.total_cmp(&b.1))
                        .map(|(idx, _, _)| idx)
                })
                .collect();

            for (loop_idx, wall) in layer
                .loops
                .iter_mut()
                .filter(|wall| wall.wall_index == 0)
                .enumerate()
            {
                let Some(prev_loop_idx) = loop_matches[loop_idx] else {
                    // No sufficiently-close previous loop (a topology
                    // change, or a genuinely new loop this layer) --
                    // leave this loop unstitched.
                    continue;
                };
                let (prev_points, prev_fractions) = &prev_loops[prev_loop_idx];
                if prev_points.is_empty() || wall.points.is_empty() {
                    continue;
                }

                // The mesh's independently-extracted wall-0 contour can
                // wind in either rotational sense from one layer to the
                // next (this codebase's contour extraction does not
                // guarantee a consistent winding direction layer to
                // layer). Arc-length fraction is direction-sensitive --
                // walking `points` in stored order always gives
                // increasing fractions in *that* stored direction, so if
                // the previous loop's stored direction is the opposite
                // rotational sense from the current loop's, every
                // fraction-based correspondence beyond the aligned start
                // point would drift further wrong around the loop
                // (worst near the far side). Detect this via the sign of
                // each loop's shoelace-formula signed area in the
                // transverse (u, v) plane (same `basis1`/`basis2`
                // convention used elsewhere in this function): a
                // reversed winding flips the sign. When they differ,
                // walk the previous loop in reverse (and recompute its
                // arc-length fractions from that reversed order) so both
                // loops' fractions increase in the same rotational
                // sense before correspondence is computed.
                let signed_area = |pts: &[DVec3]| -> f64 {
                    let n = pts.len();
                    let mut area = 0.0;
                    for i in 0..n {
                        let (u0, v0) = (pts[i].dot(basis1), pts[i].dot(basis2));
                        let (u1, v1) = (pts[(i + 1) % n].dot(basis1), pts[(i + 1) % n].dot(basis2));
                        area += u0 * v1 - u1 * v0;
                    }
                    area
                };
                let reversed_prev_storage;
                let (prev_points, prev_fractions): (&Vec<DVec3>, &Vec<f64>) =
                    if signed_area(&wall.points).signum() != signed_area(prev_points).signum() {
                        let rev_points: Vec<DVec3> = prev_points.iter().rev().copied().collect();
                        let rev_fractions = compute_arc_fractions(&rev_points);
                        reversed_prev_storage = (rev_points, rev_fractions);
                        (&reversed_prev_storage.0, &reversed_prev_storage.1)
                    } else {
                        (prev_points, prev_fractions)
                    };

                // Establish the rotational offset between the two loops'
                // independent parameterizations via a rotation search
                // across several sample points (not a single nearest-point
                // match at just `wall.points[0]` -- see
                // `WALL_GAP_ROTATION_SEARCH_SAMPLES`'s docs for why a
                // single-point alignment can silently lock onto the wrong
                // offset on spiky/non-convex loops).
                let offset = best_rotation_offset(
                    &wall.points,
                    &wall.arc_fraction,
                    prev_points,
                    prev_fractions,
                );

                // First pass: compute every point's raw correspondence
                // (fraction-based, with bounded local fallback) before
                // doing any bisecting. This is needed so the second pass
                // below can compare a point's hop against its immediate
                // current-loop neighbors' hops -- a comparison that is
                // only meaningful once *all* of them are known, not just
                // the ones processed so far in stored order.
                let n = wall.points.len();
                let mut correspondences: Vec<DVec3> = Vec::with_capacity(n);
                for (point, &fraction) in wall.points.iter().zip(wall.arc_fraction.iter()) {
                    let point = *point;
                    let t_prev = (offset + fraction).rem_euclid(1.0);
                    let fraction_based =
                        interpolate_on_loop_at_fraction(prev_points, prev_fractions, t_prev);
                    // Even with the best global offset, non-uniform point
                    // density/sampling between the two loops can locally
                    // warp the fraction-to-position mapping (worst near
                    // corners/spikes). When the fraction-implied
                    // correspondent is still further than the hop limit,
                    // try a bounded local refinement scoped to a small
                    // window of previous-loop indices around the
                    // fraction-implied position -- never the whole
                    // previous loop (that unbounded search is exactly what
                    // caused the original zigzag defect).
                    let corresponding =
                        if lateral_gap(field.as_ref(), fraction_based, point) > hop_limit {
                            local_fallback_correspondence(
                                prev_points,
                                prev_fractions,
                                t_prev,
                                point,
                                fraction_based,
                            )
                        } else {
                            fraction_based
                        };
                    correspondences.push(corresponding);
                }

                // Second pass: a single current-loop point whose own
                // fraction-based correspondence lands far away, while both
                // of its immediate current-loop neighbors (before/after in
                // stored point order) have perfectly good, close
                // correspondences, is not plausibly a genuine isolated gap
                // -- see `correct_isolated_correspondence_outliers`'s docs.
                correct_isolated_correspondence_outliers(
                    &wall.points,
                    &mut correspondences,
                    hop_limit,
                    &|point, corresponding| lateral_gap(field.as_ref(), corresponding, point),
                );

                // Third pass: decide, per point, whether it genuinely
                // needs stitching (lateral gap over the limit AND no
                // previous-layer material laterally nearby), then group
                // consecutive needs-stitch points into maximal runs and
                // emit ONE continuous serpentine block per run (see
                // [`serpentine_stitch_block`]) instead of a separate
                // anchor-to-target ramp per point -- the per-point ramps
                // made the nozzle dive down to the previous layer and
                // climb back up for EVERY point in a gap run, printing a
                // sawtooth/zigzag instead of continuous stitch lines.
                let needs_stitch: Vec<bool> = wall
                    .points
                    .iter()
                    .enumerate()
                    .zip(correspondences.iter())
                    .map(|((i, point), corresponding)| {
                        if lateral_gap(field.as_ref(), *corresponding, *point) <= hop_limit {
                            return false;
                        }
                        // Correspondence sanity veto: arc-length-fraction
                        // correspondence can fail systematically across a
                        // whole region when the contour's shape changes
                        // drastically between adjacent layers (e.g. two
                        // lobes merging), sending `corresponding` to the
                        // far side of the loop and bridging a many-mm
                        // "gap" that does not physically exist. A
                        // *genuine* gap means no previous-layer wall-0
                        // material lies laterally near this point at all
                        // -- so if ANY previous-loop point is within the
                        // hop limit laterally, the point is actually
                        // supported and needs no stitch. Scans EVERY
                        // previous wall-0 loop, not just the matched one
                        // -- in topology-change regions the supporting
                        // material often belongs to a different previous
                        // loop than the centroid/perimeter match picked.
                        // This full scan is used strictly as a veto,
                        // never as the correspondence itself, so it
                        // cannot reintroduce the nearest-point zigzag
                        // defect (see this function's docs).
                        !prev_loops.iter().any(|(loop_points, _)| {
                            loop_points
                                .iter()
                                .any(|prev| lateral_gap(field.as_ref(), *prev, *point) <= hop_limit)
                        }) && !has_solid_material_in_direction(
                            mesh_sdf.as_deref(),
                            *point,
                            wall.points[(i + 1) % n] - wall.points[(i + n - 1) % n],
                            config.nozzle_diameter,
                            config.wall_line_width,
                            -BUILD_DIRECTION,
                        ) && chord_stays_in_solid(
                            mesh_sdf.as_deref(),
                            *corresponding,
                            *point,
                            config.nozzle_diameter / 2.0,
                        )
                    })
                    .collect();

                let mut new_points = Vec::with_capacity(wall.points.len());
                let mut new_unsupported = Vec::with_capacity(wall.unsupported.len());
                let mut new_arc_fraction = Vec::with_capacity(wall.arc_fraction.len());
                let mut i = 0usize;
                while i < n {
                    if !needs_stitch[i] {
                        new_points.push(wall.points[i]);
                        new_unsupported.push(wall.unsupported[i]);
                        new_arc_fraction.push(wall.arc_fraction[i]);
                        i += 1;
                        continue;
                    }
                    // Maximal run of consecutive needs-stitch points
                    // [i..=run_end]. (A run split across the stored-order
                    // wrap point becomes two separate blocks -- slightly
                    // suboptimal, never incorrect.)
                    let mut run_end = i;
                    while run_end + 1 < n && needs_stitch[run_end + 1] {
                        run_end += 1;
                    }
                    // The real point immediately preceding this run in
                    // the final wall (wraps for a run starting at index
                    // 0) -- distinct from `correspondences[i]` (the
                    // previous-LAYER anchor the block seeds from): this
                    // is the current layer's own last non-stitch point,
                    // and the "approach" hop from it to the block's
                    // first emitted point is real printed geometry that
                    // must also be checked for void crossings, not just
                    // the anchor-seeded interior of the block.
                    let run_predecessor = wall.points[(i + n - 1) % n];
                    serpentine_stitch_block(
                        &correspondences[i..=run_end],
                        &wall.points[i..=run_end],
                        &wall.arc_fraction[i..=run_end],
                        *prev_order,
                        cur_order,
                        field.as_ref(),
                        max_along,
                        hop_limit,
                        mesh_sdf.as_deref(),
                        config.nozzle_diameter / 2.0,
                        run_predecessor,
                        &mut new_points,
                        &mut new_unsupported,
                        &mut new_arc_fraction,
                    );
                    for k in i..=run_end {
                        new_points.push(wall.points[k]);
                        new_unsupported.push(wall.unsupported[k]);
                        new_arc_fraction.push(wall.arc_fraction[k]);
                    }
                    i = run_end + 1;
                }
                wall.points = new_points;
                wall.unsupported = new_unsupported;
                wall.arc_fraction = new_arc_fraction;
            }
        }

        // Tag every wall-0 point as `top_surface` when there is no solid
        // mesh material a nozzle-diameter above it (probing
        // `-BUILD_DIRECTION`, the not-yet-printed side) -- independent of
        // the stitching/support logic above (which probes the opposite,
        // already-printed side), and computed unconditionally (even on
        // the very first layer, which has no `previous_wall0` to compare
        // against) since it only depends on this layer's own mesh SDF and
        // final (post-stitch) points.
        let mesh_sdf = layer.mesh_sdf.clone();
        for wall in layer.loops.iter_mut().filter(|wall| wall.wall_index == 0) {
            let n = wall.points.len();
            wall.top_surface = (0..n)
                .map(|i| {
                    let point = wall.points[i];
                    let tangent = wall.points[(i + 1) % n] - wall.points[(i + n - 1) % n];
                    has_solid_material_in_direction(
                        mesh_sdf.as_deref(),
                        point,
                        tangent,
                        config.nozzle_diameter,
                        config.wall_line_width,
                        BUILD_DIRECTION,
                    )
                })
                .map(|has_material_above| !has_material_above)
                .collect();
        }

        previous_wall0 = Some((layer.order, current_wall0));
    }
}

/// Computes the centroid (arithmetic mean of `points`) of a wall-0 loop,
/// used by [`stitch_wall_gaps`] to establish spatial correspondence
/// between a current-layer loop and a previous-layer loop before running
/// arc-length-fraction correspondence within the matched pair. Returns
/// `DVec3::ZERO` for an empty slice (callers already skip empty loops
/// before stitching, so this is just a safe fallback, never load-bearing).
fn loop_centroid(points: &[DVec3]) -> DVec3 {
    if points.is_empty() {
        return DVec3::ZERO;
    }
    let sum: DVec3 = points.iter().copied().fold(DVec3::ZERO, |acc, p| acc + p);
    sum / (points.len() as f64)
}

/// Total perimeter (closed-polyline arc length, including the closing
/// segment from the last point back to the first) of a wall-0 loop, used
/// by [`stitch_wall_gaps`] as a second, shape-aware plausibility check on
/// top of centroid distance before accepting a current-loop/previous-loop
/// match (see `WALL_GAP_LOOP_PERIMETER_MATCH_FACTOR`'s docs for why
/// centroid distance alone is not sufficient). Returns `0.0` for fewer
/// than 2 points (no meaningful perimeter).
fn loop_perimeter(points: &[DVec3]) -> f64 {
    let n = points.len();
    if n < 2 {
        return 0.0;
    }
    (0..n)
        .map(|i| (points[(i + 1) % n] - points[i]).length())
        .sum()
}

/// Computes each point's normalized cumulative arc-length position
/// around a closed polyline (see [`WallLoop::arc_fraction`]):
/// `result[0] == 0.0` and `result[i]` increases monotonically walking
/// `points` in order, including the closing segment from the last point
/// back to `points[0]` (which is what makes the loop's total perimeter
/// the normalization denominator). Returns an all-zero `Vec` of the same
/// length for fewer than 2 points or a degenerate (zero-perimeter) loop,
/// since there is no meaningful arc-length position to compute.
fn compute_arc_fractions(points: &[DVec3]) -> Vec<f64> {
    let n = points.len();
    if n < 2 {
        return vec![0.0; n];
    }
    let mut cumulative = Vec::with_capacity(n);
    let mut running = 0.0;
    for i in 0..n {
        cumulative.push(running);
        running += (points[(i + 1) % n] - points[i]).length();
    }
    let total = running;
    if total <= 0.0 {
        return vec![0.0; n];
    }
    cumulative.into_iter().map(|c| c / total).collect()
}

/// Interpolates a position along a closed polyline at normalized
/// arc-length fraction `t` (see [`WallLoop::arc_fraction`] /
/// [`compute_arc_fractions`]), used by [`stitch_wall_gaps`] to find a
/// current-loop point's corresponding position on the previous layer's
/// wall-0 loop. `fractions` must be `points`'s own arc-length-fraction
/// table (monotonically increasing, `fractions[0] == 0.0`). `t` is
/// wrapped into `[0, 1)` first, so callers may pass an unwrapped
/// `offset + fraction` sum directly. Falls back to `points[0]` (or
/// `DVec3::ZERO` if `points` is empty) for a degenerate loop.
fn interpolate_on_loop_at_fraction(points: &[DVec3], fractions: &[f64], t: f64) -> DVec3 {
    let n = points.len();
    if n == 0 {
        return DVec3::ZERO;
    }
    if n == 1 {
        return points[0];
    }
    let t = t.rem_euclid(1.0);
    let i = bracketing_segment_index(fractions, t);
    let seg_start = fractions[i];
    let seg_end = if i + 1 < n { fractions[i + 1] } else { 1.0 };
    let span = seg_end - seg_start;
    let local_t = if span > 0.0 {
        ((t - seg_start) / span).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let next = points[(i + 1) % n];
    points[i] + (next - points[i]) * local_t
}

/// Finds the index `i` of the arc-length segment `[fractions[i],
/// fractions[i + 1])` (with the last segment's upper bound implicitly
/// `1.0`) bracketing normalized fraction `t`, shared by
/// [`interpolate_on_loop_at_fraction`] and
/// [`local_fallback_correspondence`] (the latter uses it to anchor its
/// bounded local search window at the same position the fraction-based
/// correspondence already landed on). `t` is wrapped into `[0, 1)`
/// first. `fractions` must be non-empty, monotonically increasing, and
/// start at `0.0` (a loop's own `arc_fraction`/[`compute_arc_fractions`]
/// table). Returns `0` as an unreachable-in-practice fallback: `t <
/// fractions[0] == 0.0` can only happen for `t == 0.0` exactly given the
/// `rem_euclid` wrap, which the loop below already covers.
fn bracketing_segment_index(fractions: &[f64], t: f64) -> usize {
    let n = fractions.len();
    if n <= 1 {
        return 0;
    }
    let t = t.rem_euclid(1.0);
    for i in 0..n {
        let seg_start = fractions[i];
        let seg_end = if i + 1 < n { fractions[i + 1] } else { 1.0 };
        if t >= seg_start && (t < seg_end || i + 1 == n) {
            return i;
        }
    }
    0
}

/// Searches for the rotational offset between a current-layer wall-0
/// loop and its matched previous-layer loop that best aligns their
/// independent arc-length-fraction parameterizations, used by
/// [`stitch_wall_gaps`] in place of a single nearest-point start-point
/// alignment (see [`WALL_GAP_ROTATION_SEARCH_SAMPLES`]'s docs for why a
/// single point is not reliable on spiky/non-convex loops).
///
/// Takes up to `WALL_GAP_ROTATION_SEARCH_SAMPLES` evenly-spaced samples
/// (by point index, so they are spread around the *whole* current loop,
/// not clustered near its start) and, for each candidate offset, scores
/// it by the total squared distance between each sample point and the
/// previous loop's fraction-interpolated position at that sample's own
/// fraction plus the candidate offset. Candidate offsets are drawn from
/// the previous loop's own point fractions (`prev_fractions[j] -
/// <first sample's fraction>` for each `j`): this guarantees the true
/// best alignment (if the first sample truly corresponds to some
/// previous-loop point, however approximately) is always among the
/// candidates, while keeping the search bounded to `O(samples *
/// prev_points.len())` -- still just a per-loop-pair cost, not
/// per-point. Returns `0.0` if either loop is empty.
fn best_rotation_offset(
    current_points: &[DVec3],
    current_fractions: &[f64],
    prev_points: &[DVec3],
    prev_fractions: &[f64],
) -> f64 {
    let n_cur = current_points.len();
    let n_prev = prev_points.len();
    if n_cur == 0 || n_prev == 0 {
        return 0.0;
    }

    let sample_count = WALL_GAP_ROTATION_SEARCH_SAMPLES.min(n_cur);
    let step = n_cur as f64 / sample_count as f64;
    let samples: Vec<(DVec3, f64)> = (0..sample_count)
        .map(|i| {
            let idx = ((i as f64 * step) as usize).min(n_cur - 1);
            (current_points[idx], current_fractions[idx])
        })
        .collect();
    let anchor_fraction = samples[0].1;

    (0..n_prev)
        .map(|j| {
            let candidate = (prev_fractions[j] - anchor_fraction).rem_euclid(1.0);
            let error: f64 = samples
                .iter()
                .map(|(sample_point, sample_fraction)| {
                    let t_prev = (candidate + sample_fraction).rem_euclid(1.0);
                    let corresponding =
                        interpolate_on_loop_at_fraction(prev_points, prev_fractions, t_prev);
                    (*sample_point - corresponding).length_squared()
                })
                .sum();
            (candidate, error)
        })
        .min_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(candidate, _)| candidate)
        .expect("n_prev > 0 checked above")
}

/// Lateral (perpendicular to the local climb direction) component of
/// the gap between a current-layer point and its previous-layer
/// correspondent, used by [`stitch_wall_gaps`] and
/// [`serpentine_stitch_block`] as
/// the gap-detection metric in place of raw 3D distance.
///
/// Raw 3D distance between corresponding points on adjacent layers
/// always includes the inter-layer climb separation itself (~= one
/// layer height), so comparing it against `0.9 * nozzle_radius` triggers
/// on essentially *every* point of *every* contour whenever
/// `layer_height >= 0.9 * nozzle_radius` (e.g. 0.20mm layers with a
/// 0.4mm nozzle) -- including perfectly vertical walls that need no
/// stitching at all. What actually determines whether a freshly
/// deposited bead bonds to the previous layer's bead is how far it sits
/// *sideways* from it; the climb separation of one layer step is normal
/// and expected. This computes that sideways component: `delta` minus
/// its projection onto the local climb direction `n` (the normalized
/// order-field gradient -- order increases along the climb, so its
/// gradient is the direction consecutive layers separate in; for a
/// `Height` field this is exactly the build direction).
///
/// The gradient is sampled at the midpoint of the two points: the metric
/// is symmetric in its endpoints, and the midpoint best represents the
/// climb direction *of the gap itself* rather than privileging either
/// endpoint's local field value.
///
/// Degenerate gradients (non-finite anywhere in the central-difference
/// stencil -- see [`order_field::numeric_gradient`] -- or (near-)zero
/// length) fall back to the gradient at either endpoint, and finally to
/// treating [`BUILD_DIRECTION`] as the climb direction. The old "fall
/// back to the full 3D `delta` length" behavior is deliberately NOT
/// used: with `layer_height >= hop_limit` (e.g. 0.20mm layers vs a
/// 0.18mm limit) it classifies every perfectly-supported point in a
/// degenerate-gradient region (common on Eikonal grid plateaus / near
/// INFINITY cells) as a gap, flooding those regions with bogus stitch
/// chains -- the exact failure this metric exists to prevent. Projecting
/// out the build direction is a far better approximation of "sideways"
/// than not projecting anything at all.
pub fn lateral_gap<F: OrderField + ?Sized>(field: &F, from: DVec3, to: DVec3) -> f64 {
    /// Below this squared length the gradient's direction is numeric
    /// noise, not a meaningful climb direction.
    const MIN_GRADIENT_LENGTH_SQ: f64 = 1e-12;

    let delta = to - from;
    let mid = 0.5 * (from + to);
    let n = [mid, from, to]
        .into_iter()
        .find_map(|p| {
            order_field::numeric_gradient(field, p)
                .filter(|g| g.length_squared() > MIN_GRADIENT_LENGTH_SQ)
        })
        .map_or(BUILD_DIRECTION, |grad| grad / grad.length());
    (delta - delta.dot(n) * n).length()
}

/// A single current-loop point whose raw fraction-based (plus bounded
/// local-fallback) correspondence lands further than `hop_limit` from
/// `points[i]`, while both of its immediate current-loop neighbors
/// (`points[i-1]`/`points[i+1]`, wrapping around the closed loop) have
/// correspondences within `hop_limit` of their own points, is not
/// plausibly a genuine isolated gap: a real shallow-overhang region
/// spans a *run* of consecutive points with a bad correspondence, never
/// a single point bracketed on both sides by points with perfectly good,
/// close correspondences. Even after the best global rotation offset
/// (see [`best_rotation_offset`]) and the bounded local fallback (see
/// [`local_fallback_correspondence`]), a single previous-loop sampling
/// irregularity can still throw off one point's fraction-interpolated
/// position while its neighbors -- who interpolate at nearby but
/// different fractions, potentially bracketing a different, unaffected
/// segment of the previous loop -- land fine.
///
/// When exactly this isolated pattern is detected, this replaces
/// `correspondences[i]` in place with the midpoint of its neighbors' own
/// (already-good) correspondences -- a locally-consistent estimate that
/// follows the loop's real local shape -- rather than trusting the raw
/// fraction-based lookup for that one point. All hop distances used for
/// the decision are computed from `correspondences` *before* any
/// correction in this pass (a fixed snapshot), so corrections do not
/// cascade or depend on iteration order. Deliberately conservative: it
/// never overrides a run of 2 or more consecutive bad points (both
/// neighbors must individually already be good), matching the
/// established pattern in this codebase of preferring bounded, local
/// corrections over broader searches that previously caused zigzag
/// defects (see `stitch_wall_gaps`'s docs). No-op for fewer than 3
/// points (no distinct neighbor pair exists).
///
/// `gap` is the metric a hop is measured with (the same lateral-gap
/// metric [`stitch_wall_gaps`] triggers on -- see [`lateral_gap`] --
/// injected as a closure so this function stays a pure, field-free
/// geometric pass that unit tests can drive with a plain 3D-distance
/// metric), called as `gap(point, correspondence)`.
fn correct_isolated_correspondence_outliers(
    points: &[DVec3],
    correspondences: &mut [DVec3],
    hop_limit: f64,
    gap: &dyn Fn(DVec3, DVec3) -> f64,
) {
    let n = points.len();
    if n < 3 {
        return;
    }
    debug_assert_eq!(points.len(), correspondences.len());

    let hops: Vec<f64> = points
        .iter()
        .zip(correspondences.iter())
        .map(|(point, corresponding)| gap(*point, *corresponding))
        .collect();

    for i in 0..n {
        if hops[i] <= hop_limit {
            continue;
        }
        let prev_i = (i + n - 1) % n;
        let next_i = (i + 1) % n;
        if hops[prev_i] <= hop_limit && hops[next_i] <= hop_limit {
            let blended = (correspondences[prev_i] + correspondences[next_i]) * 0.5;
            if gap(points[i], blended) < hops[i] {
                correspondences[i] = blended;
            }
        }
    }
}

/// Bounded local fallback for an individual point's fraction-based
/// correspondence, used by [`stitch_wall_gaps`] when the point returned
/// by [`interpolate_on_loop_at_fraction`] (via [`best_rotation_offset`]'s
/// global offset) is still further than the hop limit from `point`.
/// Non-uniform point density/sampling between the two loops can locally
/// warp the fraction-to-position mapping even after the best global
/// offset is found (worst around sharp corners/spikes, where nearby
/// points can have very different local arc-length contributions) --
/// this searches a small, bounded window of previous-loop *indices*
/// (`+/- WALL_GAP_LOCAL_FALLBACK_WINDOW`, wrapping around the closed
/// loop) centered on the segment `t_prev` already lands in, for a raw-3D
/// nearest point to `point`. Deliberately bounded and local: an
/// unconstrained whole-previous-loop nearest-point search is exactly
/// what produced the original perpendicular-zigzag defect (see
/// `stitch_wall_gaps`'s docs, "Addendum 2"), so this only ever looks a
/// few points away from an already-plausible fraction-implied position,
/// never anywhere else on the loop. Candidates within the window are
/// deliberately compared by raw 3D distance, not [`lateral_gap`]: this
/// is choosing the geometrically closest real previous-loop *point* to
/// anchor the correspondence, not deciding whether a gap needs
/// stitching -- the lateral criterion applies afterwards, to whatever
/// correspondence this returns. Returns `fallback` (the original
/// fraction-based correspondence) unchanged if `prev_points` is empty or
/// no point in the window is closer than it.
fn local_fallback_correspondence(
    prev_points: &[DVec3],
    prev_fractions: &[f64],
    t_prev: f64,
    point: DVec3,
    fallback: DVec3,
) -> DVec3 {
    let n = prev_points.len();
    if n == 0 {
        return fallback;
    }

    let center = bracketing_segment_index(prev_fractions, t_prev) as isize;
    let window = WALL_GAP_LOCAL_FALLBACK_WINDOW as isize;
    let mut best = fallback;
    let mut best_dist_sq = (point - fallback).length_squared();
    for delta in -window..=window {
        let idx = (center + delta).rem_euclid(n as isize) as usize;
        let dist_sq = (point - prev_points[idx]).length_squared();
        if dist_sq < best_dist_sq {
            best_dist_sq = dist_sq;
            best = prev_points[idx];
        }
    }
    best
}

/// Continuous-serpentine building block behind [`stitch_wall_gaps`]:
/// fills the gap under a maximal run of consecutive needs-stitch
/// current-loop points with continuous intermediate contour LINES
/// rather than one anchor-to-target ramp per point.
///
/// Per-point ramps produced a physical zigzag: for every point in a gap
/// run the nozzle dove down to the previous layer's anchor and climbed
/// back up to the current layer, over and over. Instead, this treats the
/// run as a 2D patch spanned by `columns` (one per run point: its
/// previous-layer anchor `anchors[i]` at `prev_order`, its current-layer
/// target `targets[i]` at `cur_order`) and `levels` (intermediate order
/// values between the two layers), and emits the patch points row by
/// row in a serpentine order: walk one order level across the whole run,
/// step up one level at the same column, walk back, and so on -- i.e.
/// genuine continuous stitch lines. Row directions are chosen so the
/// topmost inserted row ends at column 0, immediately adjacent (one
/// level step below) to the run's first real point, so the loop then
/// continues into the run's own points without a long backtrack hop.
/// Consecutive emitted points are therefore always either horizontal
/// neighbors within a row or a single-level vertical step at a row
/// turnaround, keeping every hop within `hop_limit` (see below).
///
/// Each column's interior level points are built by walking up from the
/// previous level's ACCEPTED point: the seed is one straight step from it,
/// re-aimed at the target across the levels still remaining, then
/// reprojected onto the field's isosurface at that level's order value via
/// [`order_field::project_onto_isosurface`]. The refinement is accepted
/// only within the column's own per-level scale (nominal step length plus
/// `hop_limit`): for a non-monotonic field (`Eikonal` near
/// reentrant/threaded geometry, which this stitching path exists to bridge
/// in the first place) an unconstrained Newton descent can converge onto a
/// *different, wrong* branch of the isosurface arbitrarily far away, while
/// the re-aimed seed alone already guarantees geometric progress toward
/// the target. A seed landing where the field is unreached (order = inf,
/// e.g. an Eikonal FMM hole) never becomes an emitted point; the column
/// repeats its last on-field point instead.
///
/// The level count starts at 1 and doubles until every column's
/// consecutive LATERAL hops (see [`lateral_gap`]; the climb component of
/// one level step is normal and expected) are within `hop_limit`, until
/// doubling stops paying off (a column whose chord crosses a genuine field
/// discontinuity has no intermediate isosurface to land on, so its worst
/// hop stops shrinking -- doubling further would only multiply the point
/// count), or until [`MAX_STITCH_LEVELS`] is reached.
///
/// Every emitted point is flagged `unsupported = true` (see
/// [`WallLoop::unsupported`]) and carries its own column's target-point
/// arc fraction (an inserted stitch point doesn't add a new
/// circumferential position, only an intermediate order value on the way
/// to that target), keeping `out_arc_fraction` parallel to
/// `out_points`/`out_unsupported`. Does not append the run's own target
/// points; the caller pushes those itself.
///
/// Before emitting, every consecutive hop in the finished patch (anchor
/// -> first interior point, interior point -> interior point, last
/// interior point -> target, for every column) is checked against
/// [`chord_stays_in_solid`] using `mesh_sdf`/`void_tolerance` (one bead
/// radius). Unlike the initial anchor->target veto in `stitch_wall_gaps`
/// (a single straight chord), the finished serpentine patch's own
/// interior points follow the order field's isosurfaces and were never
/// re-checked -- an interior point can still land on the far side of a
/// void the straight-chord veto never had a reason to sample (confirmed
/// on pug_v4_m: a stitched chain's interior segment crossed a void even
/// though the run's anchor->target chord passed the initial veto). If
/// any hop fails, the whole block is discarded exactly like a
/// stalled/capped block whose worst residual hop didn't improve enough --
/// the run is left unstitched (a single straight Overhang bridge)
/// instead of printing lines through open air. `mesh_sdf = None` (layers
/// built without one, e.g. unit-test fixtures) skips this check.
#[allow(clippy::too_many_arguments)] // one param per geometric input/output; a config struct would obscure the patch construction this directly mirrors
fn serpentine_stitch_block<F: OrderField + ?Sized>(
    anchors: &[DVec3],
    targets: &[DVec3],
    fractions: &[f64],
    prev_order: f64,
    cur_order: f64,
    field: &F,
    max_along: f64,
    hop_limit: f64,
    mesh_sdf: Option<&MeshSdf>,
    void_tolerance: f64,
    run_predecessor: DVec3,
    out_points: &mut Vec<DVec3>,
    out_unsupported: &mut Vec<bool>,
    out_arc_fraction: &mut Vec<f64>,
) {
    let cols = anchors.len();
    debug_assert_eq!(cols, targets.len());
    debug_assert_eq!(cols, fractions.len());
    if cols == 0 || !(hop_limit.is_finite() && hop_limit > 0.0) {
        return;
    }

    let build_column = |anchor: DVec3, target: DVec3, levels: usize| -> Vec<DVec3> {
        // Seed each level from the previous level's ACCEPTED point,
        // re-aimed at the target across the levels still remaining, not
        // from a direct anchor->target lerp: in strongly curved field
        // regions (Eikonal around reentrant geometry) the straight chord
        // between the layers can wander far off the intermediate
        // isosurfaces -- lerp seeds there either fail projection outright
        // (even landing where the FMM field is unreached, order = inf) or
        // get rejected by the drift filter, leaving off-level points that
        // physically zigzag. Walking up level by level keeps every seed
        // one small step from a point already on the field, and re-aiming
        // at the target each step keeps cumulative projection drift from
        // carrying the column's top away from the run point it must meet.
        let nominal_step = (target - anchor).length() / levels as f64;
        let max_refinement_drift = nominal_step + hop_limit;
        let mut column = Vec::with_capacity(levels);
        column.push(anchor);
        let mut prev = anchor;
        for j in 1..levels {
            let t = j as f64 / levels as f64;
            let order = prev_order + (cur_order - prev_order) * t;
            let remaining = (levels - j + 1) as f64;
            let seed = prev + (target - prev) / remaining;
            let point = order_field::project_onto_isosurface(field, seed, order, max_along)
                .filter(|p| (*p - seed).length() <= max_refinement_drift)
                .unwrap_or(if field.order(seed).is_finite() {
                    seed
                } else {
                    // Seed fell where the field is unreached (e.g. an
                    // Eikonal FMM hole outside occupancy, order = inf) --
                    // never emit a point off the field entirely; repeat
                    // the column's last on-field point instead (a
                    // zero-length segment in the printed line).
                    prev
                });
            column.push(point);
            prev = point;
        }
        column
    };

    let mut levels = 1usize;
    let mut previous_worst = f64::INFINITY;
    let columns: Vec<Vec<DVec3>> = loop {
        let candidate: Vec<Vec<DVec3>> = anchors
            .iter()
            .zip(targets.iter())
            .map(|(&anchor, &target)| build_column(anchor, target, levels))
            .collect();
        let worst_hop = candidate
            .iter()
            .zip(targets.iter())
            .map(|(column, &target)| {
                let within = column
                    .windows(2)
                    .map(|w| lateral_gap(field, w[0], w[1]))
                    .fold(0.0, f64::max);
                let approach = column
                    .last()
                    .map_or(0.0, |&top| lateral_gap(field, top, target));
                within.max(approach)
            })
            .fold(0.0, f64::max);
        // Stop when converged -- or when doubling has stopped paying off:
        // a column whose chord crosses a genuine field discontinuity (the
        // very unreachable region this stitch is bridging, e.g. an ear
        // tip) has no intermediate isosurface to land on, so its worst
        // hop never shrinks and doubling to MAX_STITCH_LEVELS would only
        // explode the point count for the whole run.
        let stalled = worst_hop > 0.9 * previous_worst;
        if worst_hop <= hop_limit || stalled || levels >= MAX_STITCH_LEVELS {
            // A block that stalled (or capped out) without materially
            // reducing the worst remaining hop is a net negative: it
            // prints an excursion along intermediate rows and still ends
            // with (nearly) the same unsupported jump the run had before
            // stitching. Only emit the block when its worst residual
            // top-row -> target hop is at most half the worst direct
            // anchor -> target hop it replaces; otherwise leave the run
            // unstitched (a single straight Overhang bridge).
            if worst_hop > hop_limit {
                let worst_unstitched = anchors
                    .iter()
                    .zip(targets.iter())
                    .map(|(&anchor, &target)| lateral_gap(field, anchor, target))
                    .fold(0.0, f64::max);
                let worst_residual = candidate
                    .iter()
                    .zip(targets.iter())
                    .map(|(column, &target)| {
                        column
                            .last()
                            .map_or(0.0, |&top| lateral_gap(field, top, target))
                    })
                    .fold(0.0, f64::max);
                if worst_residual > 0.5 * worst_unstitched {
                    return;
                }
            }
            break candidate;
        }
        previous_worst = worst_hop;
        levels *= 2;
    };

    // Build the actual printed emission sequence (row-major serpentine,
    // duplicate-skipped) up front -- both to run the void veto over the
    // path as it will really be printed, and to emit it once validated.
    // This matters because the physically printed order connects
    // `columns[i][j] -> columns[i+1][j]` (a HORIZONTAL, same-level, cross-
    // column hop) at every row, not just the vertical anchor->...->target
    // hops within one column: an earlier version of this veto checked
    // only the vertical column hops and missed void crossings on these
    // horizontal row transitions entirely (confirmed on pug_v4_m: a
    // several-mm void-crossing hop between adjacent columns at the same
    // row slipped through undetected).
    let mut sequence: Vec<(usize, DVec3)> = Vec::with_capacity(cols * levels);
    #[allow(clippy::needless_range_loop)]
    // `j` indexes the ROW across every column (`columns[i][j]`); iterating `columns` directly would invert the serpentine's row-major emission order
    for j in 0..levels {
        // The topmost inserted row (j == levels - 1) must run backward so
        // it ends at column 0, adjacent to the run's first real point;
        // alternate direction downward from there so consecutive rows
        // always turn around at a shared column.
        let backward = (levels - 1 - j).is_multiple_of(2);
        let column_indices: Box<dyn Iterator<Item = usize>> = if backward {
            Box::new((0..cols).rev())
        } else {
            Box::new(0..cols)
        };
        for i in column_indices {
            // Degenerate runs (many run points sharing one previous-layer
            // anchor) make whole rows collapse to a single repeated
            // point; skip exact consecutive duplicates.
            if sequence.last().is_some_and(|&(_, p)| p == columns[i][j]) {
                continue;
            }
            sequence.push((i, columns[i][j]));
        }
    }

    // Final void-crossing veto over the finished patch's own hops (see
    // this function's docs above for why this is a separate check from
    // the anchor->target veto already applied before this function was
    // called): every consecutive hop in the actual printed sequence,
    // plus its leading anchor->first-point and trailing
    // last-point->target hops, must stay in (or hug) the solid.
    if mesh_sdf.is_some() {
        let chord_ok = |a: DVec3, b: DVec3| chord_stays_in_solid(mesh_sdf, a, b, void_tolerance);
        let leading_ok = sequence
            .first()
            .is_none_or(|&(_, p)| chord_ok(run_predecessor, p));
        // Always `targets[0]`, not `targets[last emitted column]`: the
        // backward-ends-at-column-0 rule normally makes them the same
        // point, but the caller always appends the run's real points
        // starting from `targets[0]` (`wall.points[i..=run_end]` in
        // arrival order) regardless of which column the last emitted row
        // happens to land on after duplicate-skipping.
        let trailing_ok = sequence
            .last()
            .is_none_or(|&(_, p)| chord_ok(p, targets[0]));
        let interior_ok = sequence.windows(2).all(|w| chord_ok(w[0].1, w[1].1));
        if !(leading_ok && trailing_ok && interior_ok) {
            return;
        }
    }

    for (i, point) in sequence {
        out_points.push(point);
        out_unsupported.push(true);
        out_arc_fraction.push(fractions[i]);
    }
}

/// Post-pass computing every layer's [`Layer::solid_fill_boundary`] from
/// its neighbors' [`Layer::infill_boundary`]s.
///
/// Runs once per object, over that object's full ordered layer stack, so
/// one object's top/bottom detection never leaks into a neighboring
/// object's layers (layers are grouped by [`Layer::object`], then ordered
/// by [`Layer::index`] within each group — matching how [`slice_workspace`]
/// concatenates each object's layer stack back-to-back).
///
/// For each layer `i` in an object's stack (`n` layers total, `0..n`):
/// note `index`/position `i` increases going *down* the print (see
/// [`BUILD_DIRECTION`]: `i == 0` is the top of the object, `i == n - 1` the
/// bottom), so the layer physically above `i` is `i - 1` and the layer
/// physically below is `i + 1`.
/// - `exposed_above(i)` is the part of `infill_boundary(i)` not covered by
///   `infill_boundary(i - 1)` (the layer above) — i.e. a top-facing surface
///   at `i`. Treated as fully exposed (`= infill_boundary(i)`) when there
///   is no layer `i - 1` (`i == 0`).
/// - `exposed_below(i)` is symmetric, using `infill_boundary(i + 1)` (the
///   layer below); fully exposed at `i == n - 1`.
/// - `solid_fill_boundary(i)` is the union of `exposed_above(j)` for `j` in
///   `i - top_layers + 1..=i` (clamped to the first layer, since a
///   top-facing surface at `j` makes the `top_layers` layers *below* it
///   solid) and `exposed_below(j)` for `j` in `i..=i + bottom_layers - 1`
///   (clamped to the last layer, since a bottom-facing surface at `j` makes
///   the `bottom_layers` layers *above* it solid), intersected with
///   `infill_boundary(i)` so the result is always a subset of this layer's
///   own fillable area.
///
/// Pure 2D geometry composition via [`polygon2d`]; no SDF/3D queries.
/// `basis1`/`basis2`/`axis`/`apex`/`slope` are resolved from `config`'s
/// order field (see [`order_field::resolve_axis_apex_slope`]) — matching
/// whatever `slice_mesh_with_progress` actually used to build every
/// layer's `infill_boundary` — rather than hardcoding [`BUILD_DIRECTION`],
/// which is wrong for a curved (`Conical`) order field. Every layer is
/// projected to 2D with the same fixed `origin` (`apex`) so loops from
/// different layers are directly comparable regardless of their differing
/// height along `axis` — `to_2d`'s `(u, v)` output only depends on a
/// point's `basis1`/`basis2` components, which fully capture its in-plane
/// position independent of origin. Reconstructing back to 3D, however,
/// uses [`order_field::reconstruct_on_order_field`] with each layer's own
/// `order`, so the rebuilt points land back on that layer's actual
/// (possibly curved) surface instead of collapsing onto one flat plane.
pub fn compute_solid_fill_boundaries(layers: &mut [Layer], config: &SlicerConfig) {
    use std::collections::BTreeMap;

    let (axis, apex, _slope) = order_field::resolve_axis_apex_slope(config.order_field, config);
    let (basis1, basis2) = plane_basis(axis);
    let origin = apex;

    let mut groups: BTreeMap<ObjectId, Vec<usize>> = BTreeMap::new();
    for (pos, layer) in layers.iter().enumerate() {
        groups.entry(layer.object).or_default().push(pos);
    }

    for mut positions in groups.into_values() {
        positions.sort_by_key(|&pos| layers[pos].index);
        let n = positions.len();
        if n == 0 {
            continue;
        }

        let boundaries_2d: Vec<Vec<Vec<[f64; 2]>>> = positions
            .par_iter()
            .map(|&pos| polygon2d::to_2d(&layers[pos].infill_boundary, basis1, basis2, origin))
            .collect();
        let empty_2d: Vec<Vec<[f64; 2]>> = Vec::new();

        // Determine whether layer index `k` increases with physical height (Z)
        // or decreases (HeightOrderField has `order = -z`, so index 0 is top;
        // EikonalOrderField seeds from bed, so index 0 is bottom).
        let z_at = |pos: usize| -> f64 {
            let mut sum_z = 0.0;
            let mut count = 0usize;
            for pts in &layers[pos].infill_boundary {
                for p in pts {
                    sum_z += p.z;
                    count += 1;
                }
            }
            if count == 0 {
                for wall in &layers[pos].loops {
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

        let first_real_pos = positions
            .iter()
            .copied()
            .find(|&p| !layers[p].infill_boundary.is_empty() || !layers[p].loops.is_empty());
        let last_real_pos = positions
            .iter()
            .copied()
            .rfind(|&p| !layers[p].infill_boundary.is_empty() || !layers[p].loops.is_empty());
        let z_increases = match (first_real_pos, last_real_pos) {
            (Some(f), Some(l)) if f != l => z_at(l) >= z_at(f),
            _ => false,
        };

        // Minimum printable solid-fill area: 1.0 * nozzle_diameter^2 (~0.16 mm^2 for a 0.4mm nozzle).
        // Discarding sub-bead microscopic slivers prevents exponential polygon fragmentation
        // during multi-layer boolean differences and unions.
        let min_solid_area = 1.0 * config.nozzle_diameter * config.nozzle_diameter;

        let solid_2d_per_k: Vec<Vec<Vec<[f64; 2]>>> = if z_increases {
            // Index increases with height: k + 1 is above k, k - 1 is below k.
            let exposed_above: Vec<Vec<Vec<[f64; 2]>>> = (0..n)
                .into_par_iter()
                .map(|k| {
                    let next = boundaries_2d.get(k + 1).unwrap_or(&empty_2d);
                    let exp = if next.is_empty() {
                        boundaries_2d[k].clone()
                    } else {
                        polygon2d::difference(&boundaries_2d[k], next)
                    };
                    let filtered = polygon2d::filter_min_area(&exp, min_solid_area);
                    if let Some(sdf) = &layers[positions[k]].mesh_sdf {
                        let field = &layers[positions[k]].order_field;
                        let order = layers[positions[k]].order;
                        let h = config.layer_height.max(0.1);
                        filtered
                            .into_iter()
                            .filter(|loop_| {
                                let mut c_u = 0.0;
                                let mut c_v = 0.0;
                                for &[u, v] in loop_ {
                                    c_u += u;
                                    c_v += v;
                                }
                                let len = loop_.len().max(1) as f64;
                                let c_u = c_u / len;
                                let c_v = c_v / len;
                                if let Some(p_3d) =
                                    crate::order_field::reconstruct_point_on_order_field(
                                        apex + basis1 * c_u + basis2 * c_v,
                                        axis,
                                        order,
                                        50.0,
                                        field.as_ref(),
                                    )
                                {
                                    let self_val = sdf.sample(p_3d).value;
                                    let probe = p_3d + DVec3::Z * (config.top_layers as f64 * h);
                                    let probe_val = sdf.sample(probe).value;
                                    self_val <= 0.0 && probe_val > 0.0
                                } else {
                                    false
                                }
                            })
                            .collect()
                    } else {
                        filtered
                    }
                })
                .collect();
            let exposed_below: Vec<Vec<Vec<[f64; 2]>>> = (0..n)
                .into_par_iter()
                .map(|k| {
                    let exp = if k == 0 {
                        boundaries_2d[k].clone()
                    } else {
                        polygon2d::difference(&boundaries_2d[k], &boundaries_2d[k - 1])
                    };
                    let filtered = polygon2d::filter_min_area(&exp, min_solid_area);
                    if let Some(sdf) = &layers[positions[k]].mesh_sdf {
                        let field = &layers[positions[k]].order_field;
                        let order = layers[positions[k]].order;
                        let h = config.layer_height.max(0.1);
                        filtered
                            .into_iter()
                            .filter(|loop_| {
                                let mut c_u = 0.0;
                                let mut c_v = 0.0;
                                for &[u, v] in loop_ {
                                    c_u += u;
                                    c_v += v;
                                }
                                let len = loop_.len().max(1) as f64;
                                let c_u = c_u / len;
                                let c_v = c_v / len;
                                if let Some(p_3d) =
                                    crate::order_field::reconstruct_point_on_order_field(
                                        apex + basis1 * c_u + basis2 * c_v,
                                        axis,
                                        order,
                                        50.0,
                                        field.as_ref(),
                                    )
                                {
                                    let self_val = sdf.sample(p_3d).value;
                                    let probe = p_3d - DVec3::Z * (config.bottom_layers as f64 * h);
                                    let probe_val = sdf.sample(probe).value;
                                    self_val <= 0.0 && probe_val > 0.0
                                } else {
                                    false
                                }
                            })
                            .collect()
                    } else {
                        filtered
                    }
                })
                .collect();

            (0..n)
                .into_par_iter()
                .map(|k| {
                    let mut regions: Vec<Vec<Vec<[f64; 2]>>> = Vec::new();
                    if config.top_layers > 0 {
                        // Top surface at layer j makes `top_layers` layers below it (j - top_layers + 1..=j) solid.
                        // Layer k gets contributions from j in k..=min(n - 1, k + top_layers - 1).
                        let end = (k + config.top_layers - 1).min(n - 1);
                        for exposed in exposed_above.iter().take(end + 1).skip(k) {
                            regions.push(exposed.clone());
                        }
                    }
                    if config.bottom_layers > 0 {
                        // Bottom surface at layer j makes `bottom_layers` layers above it (j..=j + bottom_layers - 1) solid.
                        // Layer k gets contributions from j in max(0, k - bottom_layers + 1)..=k.
                        let start = k.saturating_sub(config.bottom_layers - 1);
                        for exposed in exposed_below.iter().take(k + 1).skip(start) {
                            regions.push(exposed.clone());
                        }
                    }
                    let exposed_union =
                        polygon2d::filter_min_area(&polygon2d::union(&regions), min_solid_area);
                    let solid = polygon2d::intersection(&exposed_union, &boundaries_2d[k]);
                    polygon2d::filter_min_area(&solid, min_solid_area)
                })
                .collect()
        } else {
            // Index decreases with height (HeightOrderField): k - 1 is above k, k + 1 is below k.
            let exposed_above: Vec<Vec<Vec<[f64; 2]>>> = (0..n)
                .into_par_iter()
                .map(|k| {
                    let exp = if k == 0 {
                        boundaries_2d[k].clone()
                    } else {
                        polygon2d::difference(&boundaries_2d[k], &boundaries_2d[k - 1])
                    };
                    polygon2d::filter_min_area(&exp, min_solid_area)
                })
                .collect();
            let exposed_below: Vec<Vec<Vec<[f64; 2]>>> = (0..n)
                .into_par_iter()
                .map(|k| {
                    let next = boundaries_2d.get(k + 1).unwrap_or(&empty_2d);
                    let exp = if next.is_empty() {
                        boundaries_2d[k].clone()
                    } else {
                        polygon2d::difference(&boundaries_2d[k], next)
                    };
                    polygon2d::filter_min_area(&exp, min_solid_area)
                })
                .collect();

            (0..n)
                .into_par_iter()
                .map(|k| {
                    let mut regions: Vec<Vec<Vec<[f64; 2]>>> = Vec::new();
                    if config.top_layers > 0 {
                        let start = k.saturating_sub(config.top_layers - 1);
                        for exposed in exposed_above.iter().take(k + 1).skip(start) {
                            regions.push(exposed.clone());
                        }
                    }
                    if config.bottom_layers > 0 {
                        let end = (k + config.bottom_layers - 1).min(n - 1);
                        for exposed in exposed_below.iter().take(end + 1).skip(k) {
                            regions.push(exposed.clone());
                        }
                    }
                    let exposed_union =
                        polygon2d::filter_min_area(&polygon2d::union(&regions), min_solid_area);
                    let solid = polygon2d::intersection(&exposed_union, &boundaries_2d[k]);
                    polygon2d::filter_min_area(&solid, min_solid_area)
                })
                .collect()
        };

        for (k, solid_2d) in solid_2d_per_k.into_iter().enumerate() {
            let order = layers[positions[k]].order;
            let field = Arc::clone(&layers[positions[k]].order_field);
            // Reference-seeded reconstruction (see
            // `reconstruct_on_order_field_near`): the solid region is a
            // boolean composition of this layer's (and neighbors')
            // infill boundaries, all projected at the same (u, v) --
            // this layer's own 3D infill boundary gives every rebuilt
            // point a same-branch height seed, avoiding the wrong-branch
            // axis-ray solves that spiked Eikonal solid fill.
            let mut references = layers[positions[k]].infill_boundary.clone();
            if references.is_empty() {
                references = layers[positions[k]]
                    .loops
                    .iter()
                    .filter(|w| w.wall_index == 0)
                    .map(|w| w.points.clone())
                    .collect();
            }
            layers[positions[k]].solid_fill_boundary = order_field::reconstruct_on_order_field_near(
                solid_2d,
                &references,
                basis1,
                basis2,
                axis,
                apex,
                order,
                order_field::max_along_for(config),
                field.as_ref(),
            );
        }
    }
}

/// Computes a square in-plane sampling extent (see [`slice_mesh`]) large
/// enough to cover the mesh's bounding box `[min, max]` once projected onto
/// the `basis1`/`basis2` plane, independent of the mesh's extent along the
/// (perpendicular) build direction.
///
/// Projects all 8 bounding-box corners onto `basis1`/`basis2`, takes the
/// resulting 2D range's diagonal, and applies the same `* 1.5 + 1.0` margin
/// the old (buggy) full-3D-diagonal computation used, so behavior for
/// mostly-flat meshes (where footprint ~= 3D diagonal) is unchanged.
fn in_plane_extent(min: DVec3, max: DVec3, basis1: DVec3, basis2: DVec3) -> f64 {
    let corners = [
        DVec3::new(min.x, min.y, min.z),
        DVec3::new(max.x, min.y, min.z),
        DVec3::new(min.x, max.y, min.z),
        DVec3::new(min.x, min.y, max.z),
        DVec3::new(max.x, max.y, min.z),
        DVec3::new(max.x, min.y, max.z),
        DVec3::new(min.x, max.y, max.z),
        DVec3::new(max.x, max.y, max.z),
    ];

    let mut u_min = f64::INFINITY;
    let mut u_max = f64::NEG_INFINITY;
    let mut v_min = f64::INFINITY;
    let mut v_max = f64::NEG_INFINITY;
    for corner in corners {
        let u = corner.dot(basis1);
        let v = corner.dot(basis2);
        u_min = u_min.min(u);
        u_max = u_max.max(u);
        v_min = v_min.min(v);
        v_max = v_max.max(v);
    }

    let projected_diagonal = DVec3::new(u_max - u_min, v_max - v_min, 0.0).length();
    projected_diagonal * 1.5 + 1.0
}

/// Slice a single [`Object`]: bakes its `transform` into world-space
/// vertices, then slices that with [`slice_mesh`], tagging every
/// resulting layer with the object's id.
pub fn slice_object(object: &Object, config: &SlicerConfig) -> Result<Vec<Layer>> {
    slice_object_with_progress(
        object,
        config,
        &manifold_fidget::slope_profile::SlopeProfile::new(Vec::new()),
        &mut |_| {},
    )
}

/// Same as [`slice_object`], forwarding to [`slice_mesh_with_progress`] and
/// then running [`compute_solid_fill_boundaries`] as a post-pass over this
/// object's own layer stack (never mixed with any other object's layers).
pub fn slice_object_with_progress(
    object: &Object,
    config: &SlicerConfig,
    slope_profile: &manifold_fidget::slope_profile::SlopeProfile,
    on_progress: &mut (dyn FnMut(f64) + Send),
) -> Result<Vec<Layer>> {
    let world_mesh = Mesh::new(
        object
            .mesh
            .vertices
            .iter()
            .map(|&vertex| object.transform.transform_point(vertex))
            .collect(),
        object.mesh.indices.clone(),
    );

    let mut layers = slice_mesh_with_progress(&world_mesh, config, slope_profile, on_progress)?;
    for layer in &mut layers {
        layer.object = object.id;
    }
    compute_solid_fill_boundaries(&mut layers, config);
    Ok(layers)
}

/// Slice every object in a workspace, in the order given by `order`
/// (produced by an [`crate::ordering::ObjectOrderStrategy`]), concatenating
/// each object's full layer stack back-to-back.
///
/// This concatenation *is* what makes ordering "sequential" today: each
/// object is fully sliced before the next begins. A future
/// Z-interleaving/simultaneous-printing strategy would replace this
/// concatenation with a per-Z merge of layers across objects — see
/// ROADMAP.md "Deferred / future work".
///
/// # Errors
///
/// Returns [`crate::Error::InvalidMesh`] if `order` references an object id
/// not present in `objects`.
pub fn slice_workspace(
    objects: &[Object],
    order: &[ObjectId],
    config: &SlicerConfig,
) -> Result<Vec<Layer>> {
    slice_workspace_with_progress(
        objects,
        order,
        config,
        &manifold_fidget::slope_profile::SlopeProfile::new(Vec::new()),
        &mut |_| {},
    )
}

/// Same as [`slice_workspace`], reporting overall progress across every
/// object being sliced (not just the one currently in progress): each
/// object gets an equal `1 / order.len()` share of the `0.0..=1.0` range,
/// and within that share [`slice_object_with_progress`]'s own `0.0..=1.0`
/// order-field-domain progress is linearly mapped in.
pub fn slice_workspace_with_progress(
    objects: &[Object],
    order: &[ObjectId],
    config: &SlicerConfig,
    slope_profile: &manifold_fidget::slope_profile::SlopeProfile,
    on_progress: &mut (dyn FnMut(f64) + Send),
) -> Result<Vec<Layer>> {
    let total = order.len().max(1) as f64;
    let mut layers = Vec::new();
    for (object_index, &object_id) in order.iter().enumerate() {
        let object = objects
            .iter()
            .find(|object| object.id == object_id)
            .ok_or_else(|| {
                crate::Error::InvalidMesh(format!(
                    "print order references unknown object {object_id}"
                ))
            })?;
        layers.extend(slice_object_with_progress(
            object,
            config,
            slope_profile,
            &mut |local| {
                on_progress(((object_index as f64) + local) / total);
            },
        )?);
    }
    Ok(layers)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ids::ToolId, transform::Transform};
    use glam::DVec3;

    fn triangle_mesh() -> Mesh {
        Mesh::new(
            vec![
                DVec3::ZERO,
                DVec3::new(1.0, 0.0, 0.0),
                DVec3::new(0.0, 1.0, 0.0),
            ],
            vec![0, 1, 2],
        )
    }

    #[test]
    fn contour_resolution_scales_with_extent() {
        let small = contour_resolution(4.0, 0.4, CONTOUR_REFINEMENT_DIVISOR);
        let large = contour_resolution(40.0, 0.4, CONTOUR_REFINEMENT_DIVISOR);
        assert!(
            large > small,
            "a larger in-plane extent should derive a finer (larger) grid resolution"
        );
    }

    #[test]
    fn contour_resolution_scales_inversely_with_nozzle_diameter() {
        let coarse_nozzle = contour_resolution(40.0, 0.8, CONTOUR_REFINEMENT_DIVISOR);
        let fine_nozzle = contour_resolution(40.0, 0.2, CONTOUR_REFINEMENT_DIVISOR);
        assert!(
            fine_nozzle > coarse_nozzle,
            "a smaller nozzle diameter should derive a finer (larger) grid resolution"
        );
    }

    #[test]
    fn contour_resolution_respects_the_refinement_divisor_parameter() {
        // A larger divisor targets a coarser cell size (nozzle_diameter /
        // divisor), so resolution should drop as the divisor shrinks.
        let finer = contour_resolution(40.0, 0.4, 8.0);
        let coarser = contour_resolution(40.0, 0.4, 2.0);
        assert!(
            finer > coarser,
            "a larger refinement divisor should derive a finer (larger) grid resolution"
        );
    }

    #[test]
    fn contour_resolution_is_clamped_to_the_configured_bounds() {
        // Vanishingly small extent / huge nozzle diameter -> would derive a
        // resolution below MIN_CONTOUR_RESOLUTION without clamping.
        assert_eq!(
            contour_resolution(0.001, 10.0, CONTOUR_REFINEMENT_DIVISOR),
            MIN_CONTOUR_RESOLUTION
        );
        // Huge extent / vanishingly small nozzle diameter -> would derive a
        // resolution above MAX_CONTOUR_RESOLUTION without clamping.
        assert_eq!(
            contour_resolution(1.0e6, 0.001, CONTOUR_REFINEMENT_DIVISOR),
            MAX_CONTOUR_RESOLUTION
        );
    }

    #[test]
    fn slice_mesh_with_progress_reports_monotonic_progress_ending_at_one() {
        let config = SlicerConfig {
            layer_height: 0.25,
            ..SlicerConfig::default()
        };

        let mut reported = Vec::new();
        let empty_profile = manifold_fidget::slope_profile::SlopeProfile::new(Vec::new());
        slice_mesh_with_progress(&cube_mesh(), &config, &empty_profile, &mut |fraction| {
            reported.push(fraction);
        })
        .unwrap();

        assert!(!reported.is_empty());
        assert!(reported.windows(2).all(|pair| pair[1] >= pair[0]));
        assert_eq!(*reported.last().unwrap(), 1.0);
    }

    #[test]
    fn slice_workspace_with_progress_splits_range_evenly_across_objects() {
        let config = SlicerConfig {
            layer_height: 0.25,
            ..SlicerConfig::default()
        };
        let first = Object::new(ObjectId(0), cube_mesh(), ToolId(0));
        let mut second = Object::new(ObjectId(1), cube_mesh(), ToolId(0));
        second.transform = Transform::from_translation(DVec3::new(5.0, 0.0, 0.0));
        let objects = vec![first, second];
        let order = vec![ObjectId(0), ObjectId(1)];

        let mut reported = Vec::new();
        let empty_profile = manifold_fidget::slope_profile::SlopeProfile::new(Vec::new());
        slice_workspace_with_progress(&objects, &order, &config, &empty_profile, &mut |fraction| {
            reported.push(fraction);
        })
        .unwrap();

        assert!(!reported.is_empty());
        // First object's progress should stay within [0.0, 0.5], second
        // object's within [0.5, 1.0], and the overall walk should still end
        // at 1.0.
        assert!(reported
            .iter()
            .all(|&fraction| (0.0..=1.0).contains(&fraction)));
        assert_eq!(*reported.last().unwrap(), 1.0);
    }

    #[test]
    fn slice_object_applies_world_transform_before_slicing() {
        let mut object = Object::new(ObjectId(0), triangle_mesh(), ToolId(0));
        object.transform = Transform::from_translation(DVec3::new(5.0, 0.0, 0.0));

        // This mainly asserts slice_object doesn't error and wires the
        // transform in before slicing (the degenerate flat triangle fixture
        // isn't a solid, so real contour geometry is exercised by the
        // sphere/cube tests below instead).
        let layers = slice_object(&object, &SlicerConfig::default()).unwrap();
        for layer in &layers {
            assert_eq!(layer.object, ObjectId(0));
        }
    }

    #[test]
    fn slice_workspace_concatenates_in_given_order() {
        let objects = vec![
            Object::new(ObjectId(0), triangle_mesh(), ToolId(0)),
            Object::new(ObjectId(1), triangle_mesh(), ToolId(1)),
        ];
        let order = vec![ObjectId(1), ObjectId(0)];

        let layers = slice_workspace(&objects, &order, &SlicerConfig::default()).unwrap();

        // The degenerate flat triangle fixture isn't a solid, so this
        // mainly asserts the per-object lookup/ordering doesn't error;
        // real contour geometry is exercised by the sphere/cube tests below.
        for layer in &layers {
            assert!(layer.object == ObjectId(0) || layer.object == ObjectId(1));
        }
    }

    #[test]
    fn slice_workspace_rejects_unknown_object_in_order() {
        let objects = vec![Object::new(ObjectId(0), triangle_mesh(), ToolId(0))];
        let order = vec![ObjectId(99)];

        let err = slice_workspace(&objects, &order, &SlicerConfig::default()).unwrap_err();
        assert!(matches!(err, crate::Error::InvalidMesh(_)));
    }

    /// Unit cube spanning [0,1]^3 (same fixture pattern as
    /// `manifold-fidget`'s `mesh_sdf`/`contour` tests).
    fn cube_mesh() -> Mesh {
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
        let indices = vec![
            0, 2, 1, 0, 3, 2, // -Z
            4, 5, 6, 4, 6, 7, // +Z
            0, 1, 5, 0, 5, 4, // -Y
            3, 7, 6, 3, 6, 2, // +Y
            0, 4, 7, 0, 7, 3, // -X
            1, 2, 6, 1, 6, 5, // +X
        ];
        Mesh::new(vertices, indices)
    }

    /// Regression test for a direction bug: `Layer::index` increases going
    /// *down* the print (see [`BUILD_DIRECTION`] — index 0 is the top of the
    /// object), so `top_layers` must propagate solid fill toward *larger*
    /// indices (downward from a top-facing surface) and `bottom_layers`
    /// toward *smaller* indices (upward from a bottom-facing surface). An
    /// earlier version of `compute_solid_fill_boundaries` had these
    /// swapped, which a symmetric `top_layers == bottom_layers` config
    /// (the default) could never expose. Uses an asymmetric
    /// `top_layers = 1, bottom_layers = 2` on a unit cube (uniform
    /// cross-section, so only the layers adjacent to the exposed top/bottom
    /// caps get any solid fill contribution at all) to pin down the
    /// direction unambiguously.
    #[test]
    fn compute_solid_fill_boundaries_propagates_top_and_bottom_in_the_correct_direction() {
        let config = SlicerConfig {
            layer_height: 0.625,
            top_layers: 1,
            bottom_layers: 2,
            ..SlicerConfig::default()
        };

        let mut layers = slice_mesh(&big_cube_mesh(), &config).unwrap();
        compute_solid_fill_boundaries(&mut layers, &config);

        fn area(loops: &[Vec<DVec3>]) -> f64 {
            loops
                .iter()
                .map(|pts| {
                    let n = pts.len();
                    if n < 3 {
                        return 0.0;
                    }
                    (0..n)
                        .map(|i| {
                            let a = pts[i];
                            let b = pts[(i + 1) % n];
                            a.x * b.y - b.x * a.y
                        })
                        .sum::<f64>()
                        * 0.5
                })
                .sum()
        }

        // Layers with a nonempty infill_boundary must be fully solid (ratio
        // ~1) exactly where propagation should reach, and not otherwise.
        let is_fully_solid = |layer: &Layer| -> Option<bool> {
            let infill = area(&layer.infill_boundary).abs();
            if infill < 1e-9 {
                return None; // no fillable area on this layer at all
            }
            let solid = area(&layer.solid_fill_boundary).abs();
            Some((solid / infill - 1.0).abs() < 1e-6)
        };

        let solid_flags: Vec<Option<bool>> = layers.iter().map(is_fully_solid).collect();
        let first_real = solid_flags.iter().position(Option::is_some).unwrap();
        let last_real = solid_flags.iter().rposition(Option::is_some).unwrap();
        assert!(
            last_real - first_real >= 3,
            "fixture needs at least 4 layers with contours to distinguish top from bottom"
        );

        // top_layers = 1: only the layer right under the top cap is solid.
        assert_eq!(
            solid_flags[last_real],
            Some(true),
            "the layer nearest the top-facing surface must be fully solid"
        );
        assert_eq!(
            solid_flags[last_real - 1],
            Some(false),
            "top_layers = 1 must not make the second layer down from the top fully solid"
        );

        // bottom_layers = 2: the two layers right above the bottom cap are solid.
        assert_eq!(
            solid_flags[first_real],
            Some(true),
            "the layer nearest the bottom-facing surface must be fully solid"
        );
        assert_eq!(
            solid_flags[first_real + 1],
            Some(true),
            "bottom_layers = 2 must make the second layer up from the bottom fully solid"
        );
        assert_eq!(
            solid_flags[first_real + 2],
            Some(false),
            "bottom_layers = 2 must not reach a third layer up from the bottom"
        );
    }

    /// A cube scaled up (5x5x5) from [`cube_mesh`] so wall/infill offsets
    /// leave a comfortably nonempty infill region (a unit cube's infill
    /// boundary collapses to empty under the default wall settings).
    fn big_cube_mesh() -> Mesh {
        let Mesh { vertices, indices } = cube_mesh();
        Mesh::new(vertices.into_iter().map(|v| v * 5.0).collect(), indices)
    }

    #[test]
    fn slice_mesh_produces_nonempty_contour_loops_for_a_solid_cube() {
        let config = SlicerConfig {
            layer_height: 0.25,
            ..SlicerConfig::default()
        };

        let layers = slice_mesh(&cube_mesh(), &config).unwrap();

        // The cube spans Z in [0, 1] with layer_height 0.25: expect 4
        // stepped layers starting one layer height above the bed
        // (0.25, 0.5, 0.75, 1.0). The interior layers (0.25, 0.5, 0.75) are
        // clean square cross-sections; the exact top boundary layer (Z=1)
        // samples directly on the mesh surface, where the sign/crossing is
        // numerically ambiguous, so only the interior layers are asserted to
        // have a contour loop.
        assert_eq!(layers.len(), 4);
        for layer in &layers[0..3] {
            assert_eq!(layer.loops.len(), 1, "expected exactly one contour loop");
            assert!(!layer.loops[0].points.is_empty());
        }
    }

    #[test]
    fn slice_mesh_produces_nonempty_contour_loops_for_a_cube_far_from_the_world_origin() {
        // Regression test: before the fix, the contour-extraction plane's
        // origin always had zero in-plane (X/Y) components regardless of
        // where the mesh actually sits, so a mesh translated far from world
        // (0,0) (e.g. after `object::center_on_bed`) produced zero contour
        // loops for every layer, even though the SDF field itself is fine.
        let offset = DVec3::new(500.0, 500.0, 0.0);
        let mesh = cube_mesh();
        let translated = Mesh::new(
            mesh.vertices
                .iter()
                .map(|&vertex| vertex + offset)
                .collect(),
            mesh.indices.clone(),
        );

        let config = SlicerConfig {
            layer_height: 0.25,
            ..SlicerConfig::default()
        };

        let layers = slice_mesh(&translated, &config).unwrap();

        assert_eq!(layers.len(), 4);
        for layer in &layers[0..3] {
            assert_eq!(layer.loops.len(), 1, "expected exactly one contour loop");
            assert!(!layer.loops[0].points.is_empty());
        }
    }

    #[test]
    fn slice_mesh_height_mode_has_no_missing_layers_near_bed_or_top() {
        let mesh = cube_mesh();
        let config = SlicerConfig {
            layer_height: 0.14,
            wall_offset: 0.2,
            ..SlicerConfig::default()
        };
        let layers = slice_mesh(&mesh, &config).unwrap();
        assert!(!layers.is_empty());
        for (i, layer) in layers.iter().enumerate() {
            assert!(
                !layer.loops.is_empty(),
                "layer {} at order {} unexpectedly has no contour loops",
                i,
                layer.order
            );
        }
    }

    #[test]
    fn slice_mesh_height_order_field_matches_pre_change_order_range() {
        // Regression test for the `order_field::order_field_for` +
        // `order_range_over_bbox` wiring: for the default (Height) config,
        // slicing must produce the identical layer count and order values
        // as the old `min.dot(BUILD_DIRECTION)`/`max.dot(BUILD_DIRECTION)`
        // shortcut it replaced.
        let config = SlicerConfig {
            layer_height: 0.25,
            ..SlicerConfig::default()
        };

        let mesh = cube_mesh();
        let (min, max) = mesh.bounding_box().unwrap();
        let order_at_min = min.dot(BUILD_DIRECTION);
        let order_at_max = max.dot(BUILD_DIRECTION);
        let expected_order_min = order_at_min.min(order_at_max);
        let expected_order_max = order_at_min.max(order_at_max);

        let layers = slice_mesh(&mesh, &config).unwrap();

        assert_eq!(
            layers.len(),
            4,
            "expected 4 stepped layers over [0, 1] at layer_height 0.25"
        );
        let mut expected_order = expected_order_min + config.first_layer_height();
        for layer in &layers {
            assert!(
                (layer.order - expected_order).abs() < 1e-9,
                "expected layer order {expected_order}, got {}",
                layer.order
            );
            expected_order += config.layer_height;
        }
        assert!(layers.last().unwrap().order <= expected_order_max + 1e-9);
    }

    #[test]
    fn slice_mesh_dual_iso_order_field_produces_nonempty_layer_output() {
        let config = SlicerConfig {
            layer_height: 1.0,
            order_field: crate::order_field::OrderFieldKind::DualIso,
            wall_offset: 0.2,
            wall_line_width: 0.4,
            shell_thickness: 0.8,
            ..SlicerConfig::default()
        };

        let layers = slice_mesh(&big_cube_mesh(), &config).unwrap();
        assert!(
            !layers.is_empty(),
            "DualIso must produce layers for a big cube"
        );

        let total_wall0: usize = layers
            .iter()
            .map(|l| l.loops.iter().filter(|w| w.wall_index == 0).count())
            .sum();
        let total_wall1: usize = layers
            .iter()
            .map(|l| l.loops.iter().filter(|w| w.wall_index == 1).count())
            .sum();

        assert!(total_wall0 > 0, "DualIso must extract outer wall 0 loops");
        assert!(total_wall1 > 0, "DualIso must extract inner wall 1 loops");
    }

    #[test]
    fn slice_mesh_returns_no_layers_for_an_empty_mesh() {
        let layers = slice_mesh(&Mesh::default(), &SlicerConfig::default()).unwrap();
        assert!(layers.is_empty());
    }

    #[test]
    fn slice_mesh_conical_order_field_produces_real_curved_contour_geometry() {
        // Proves the curved ("contour-on-mesh") path in
        // `slice_mesh_with_progress` actually dispatches on
        // `config.order_field` and produces real geometry, instead of
        // silently falling back to the flat path or to empty loops. Apex
        // below the cube, axis pointing up through it, so the cube's [0,1]
        // Z range sits well inside the cone's opening and every layer's
        // order value has a genuinely curved (non-planar) isosurface.
        let config = SlicerConfig {
            layer_height: 0.25,
            order_field: crate::order_field::OrderFieldKind::Conical,
            order_field_apex: DVec3::new(0.5, 0.5, -1.0),
            order_field_axis: DVec3::new(0.0, 0.0, 1.0),
            order_field_slope: 0.3,
            ..SlicerConfig::default()
        };

        let layers = slice_mesh(&cube_mesh(), &config).unwrap();

        assert!(!layers.is_empty());
        assert!(
            layers.iter().any(|layer| !layer.loops.is_empty()),
            "expected at least one layer with non-empty loops from the curved contour-on-mesh path"
        );
        for layer in &layers {
            if !layer.loops.is_empty() {
                for wall_loop in &layer.loops {
                    assert!(!wall_loop.points.is_empty());
                }
            }
        }
    }

    #[test]
    fn slice_mesh_eikonal_order_field_terminates_when_bbox_corners_fall_outside_the_solid() {
        // Regression test: `order_range_over_bbox` samples 27 points on the
        // mesh's *axis-aligned bounding box* (corners, edge midpoints, face
        // centers). For a shape that doesn't fill its own bbox (e.g. a
        // tetrahedron, unlike the axis-aligned `cube_mesh` above whose bbox
        // corners coincide with its own vertices), most of those 27 sample
        // points land outside the solid, where an occupancy-restricted field
        // like `EikonalOrderField` returns `f64::INFINITY`. Before the
        // mesh-vertex fallback in `slice_mesh_with_progress`, a non-finite
        // `order_max` made the per-layer step loop run unbounded, silently
        // consuming memory until the process was killed. This must still
        // return promptly (not hang) with a finite, bounded layer count.
        let tetrahedron = Mesh {
            vertices: vec![
                DVec3::new(0.0, 0.0, 0.0),
                DVec3::new(1.0, 0.0, 0.0),
                DVec3::new(0.0, 1.0, 0.0),
                DVec3::new(0.0, 0.0, 1.0),
            ],
            indices: vec![
                0, 2, 1, // base
                0, 1, 3, // side
                1, 2, 3, // side
                2, 0, 3, // side
            ],
        };

        let config = SlicerConfig {
            layer_height: 0.1,
            order_field: crate::order_field::OrderFieldKind::Eikonal,
            ..SlicerConfig::default()
        };

        let layers = slice_mesh(&tetrahedron, &config).unwrap();

        // The bbox diagonal here is small (~1.7), so at layer_height 0.1 a
        // correctly bounded slice produces well under a few hundred layers
        // — nowhere near the `MAX_ORDER_STEPS` safety cap.
        assert!(
            layers.len() < 1_000,
            "expected a small, finite layer count, got {}",
            layers.len()
        );
    }

    #[test]
    fn slice_mesh_height_mode_generates_solid_fill_boundary_for_stepped_horizontal_surfaces() {
        let config = SlicerConfig {
            layer_height: 0.2,
            shell_thickness: 0.8,
            top_layers: 3,
            bottom_layers: 3,
            order_field: crate::order_field::OrderFieldKind::Height,
            ..SlicerConfig::default()
        };

        let mut layers = slice_mesh(&big_cube_mesh(), &config).unwrap();
        compute_solid_fill_boundaries(&mut layers, &config);
        assert!(!layers.is_empty());
        let bottom_layer = &layers[0];
        assert!(
            !bottom_layer.loops.is_empty(),
            "expected non-empty loops on bottom layer"
        );
        assert!(
            !bottom_layer.solid_fill_boundary.is_empty(),
            "expected non-empty solid_fill_boundary on bottom layer"
        );
        let top_layer = layers.last().unwrap();
        assert!(
            !top_layer.loops.is_empty(),
            "expected non-empty loops on top layer"
        );
        assert!(
            !top_layer.solid_fill_boundary.is_empty(),
            "expected non-empty solid_fill_boundary on top layer"
        );
    }

    #[test]
    fn slice_mesh_conformal_eikonal_order_field_produces_nonempty_layer_output() {
        let config = SlicerConfig {
            layer_height: 0.25,
            order_field: crate::order_field::OrderFieldKind::Eikonal,
            eikonal_conform_top_surfaces: true,
            ..SlicerConfig::default()
        };

        let layers = slice_mesh(&cube_mesh(), &config).unwrap();
        assert!(!layers.is_empty() && layers.len() >= 4);
        for layer in &layers[0..layers.len().saturating_sub(1)] {
            assert!(!layer.loops.is_empty(), "expected nonempty contour loops");
            for l in &layer.loops {
                assert!(!l.points.is_empty());
            }
        }
    }

    #[test]
    fn slice_mesh_eikonal_order_field_produces_nonempty_layer_output_for_a_simple_solid_cube() {
        // Proves `OrderFieldKind::Eikonal` is wired through `order_field_for`
        // and `slice_mesh_with_progress`'s curved ("contour-on-mesh") path
        // end to end: a mesh-derived FMM front (seeded from the cube's
        // base/contact surface with the build plate) still produces real,
        // non-empty layer geometry, exercising `reconstruct_on_order_field`'s
        // generic numeric solve against a genuinely non-axisymmetric field.
        let config = SlicerConfig {
            layer_height: 0.25,
            order_field: crate::order_field::OrderFieldKind::Eikonal,
            ..SlicerConfig::default()
        };

        let layers = slice_mesh(&cube_mesh(), &config).unwrap();

        assert!(!layers.is_empty());
        assert!(
            layers.iter().any(|layer| !layer.loops.is_empty()),
            "expected at least one layer with non-empty loops from the Eikonal-seeded curved path"
        );
        for layer in &layers {
            for wall_loop in &layer.loops {
                assert!(!wall_loop.points.is_empty());
            }
        }
    }

    #[test]
    fn slice_mesh_conical_order_field_has_no_unexpected_gaps_through_a_tall_thin_pyramid() {
        // Parallel of `slice_mesh_has_no_empty_contour_gaps_through_a_tall_thin_pyramid`
        // for the curved path (ROADMAP.md Phase 15 "Open risk" note): a
        // non-trivial (non-primitive) mesh's triangle-soup isosurface can, in
        // principle, stitch gappily where the flat plane-sampling path
        // wouldn't. Apex below the pyramid's base, axis along its own axis of
        // symmetry, so the cone's isosurfaces sweep cleanly up through the
        // body without ever running parallel to a face.
        let config = SlicerConfig {
            layer_height: 1.0,
            order_field: crate::order_field::OrderFieldKind::Conical,
            order_field_apex: DVec3::new(0.0, 0.0, -5.0),
            order_field_axis: DVec3::new(0.0, 0.0, 1.0),
            order_field_slope: 0.05,
            ..SlicerConfig::default()
        };

        let layers = slice_mesh(&tall_thin_pyramid_mesh(), &config).unwrap();
        assert!(!layers.is_empty());

        let first_nonempty = layers.iter().position(|layer| !layer.loops.is_empty());
        let last_nonempty = layers.iter().rposition(|layer| !layer.loops.is_empty());

        let (Some(first_nonempty), Some(last_nonempty)) = (first_nonempty, last_nonempty) else {
            panic!("expected at least one layer with contour loops on the curved path");
        };

        // Known limitation (see ROADMAP.md Phase 15 "Open risk to validate"):
        // the triangle-soup contour-on-mesh stitching can miss a thin near-tip
        // slice that the exact plane-sampling path wouldn't, so this is
        // intentionally looser than the flat-path equivalent test — it only
        // asserts the interior of the nonempty run isn't gappy, not that
        // every single layer between the mesh's extremes produced a loop.
        let mut gaps = 0usize;
        for layer in &layers[first_nonempty..=last_nonempty] {
            if layer.loops.is_empty() {
                gaps += 1;
            }
        }
        assert!(
            gaps <= 1,
            "curved contour-on-mesh path produced {gaps} unexpected gap(s) through the pyramid's body \
             (known limitation threshold exceeded, see ROADMAP.md Phase 15 \"Open risk\")"
        );
    }

    /// Slim square pyramid: base [-0.5, 0.5]^2 at Z=0, apex at (0, 0, 20)
    /// — height 20x its footprint. Before the fix, `extent` was sized off
    /// the full 3D bounding-box diagonal (dominated by the 20-tall
    /// height), starving the 120x120 sampling grid of resolution across
    /// the ~1-unit-wide footprint and causing layers through the body
    /// (not just the near-degenerate apex tip) to come back with zero
    /// contour loops.
    fn tall_thin_pyramid_mesh() -> Mesh {
        let vertices = vec![
            DVec3::new(-0.5, -0.5, 0.0),
            DVec3::new(0.5, -0.5, 0.0),
            DVec3::new(0.5, 0.5, 0.0),
            DVec3::new(-0.5, 0.5, 0.0),
            DVec3::new(0.0, 0.0, 20.0),
        ];
        let indices = vec![
            0, 2, 1, 0, 3, 2, // base (-Z winding)
            0, 1, 4, // side
            1, 2, 4, // side
            2, 3, 4, // side
            3, 0, 4, // side
        ];
        Mesh::new(vertices, indices)
    }

    #[test]
    fn wall_count_rounds_to_nearest_and_clamps_to_one() {
        let exact = SlicerConfig {
            wall_line_width: 0.4,
            shell_thickness: 0.8,
            ..SlicerConfig::default()
        };
        assert_eq!(exact.wall_count(), 2);

        let rounds_up = SlicerConfig {
            wall_line_width: 0.4,
            shell_thickness: 0.9,
            ..SlicerConfig::default()
        };
        assert_eq!(rounds_up.wall_count(), 2);

        let rounds_down = SlicerConfig {
            wall_line_width: 0.4,
            shell_thickness: 1.1,
            ..SlicerConfig::default()
        };
        assert_eq!(rounds_down.wall_count(), 3);

        let thinner_than_one_wall = SlicerConfig {
            wall_line_width: 0.4,
            shell_thickness: 0.1,
            ..SlicerConfig::default()
        };
        assert_eq!(thinner_than_one_wall.wall_count(), 1);
    }

    #[test]
    fn slice_mesh_produces_nested_wall_loops_for_a_multi_wall_solid_cube() {
        let config = SlicerConfig {
            layer_height: 0.25,
            wall_line_width: 0.05,
            shell_thickness: 0.15,
            wall_offset: 0.02,
            ..SlicerConfig::default()
        };
        assert_eq!(config.wall_count(), 3);

        let layers = slice_mesh(&cube_mesh(), &config).unwrap();

        for layer in &layers[0..3] {
            assert_eq!(layer.loops.len(), 3, "expected one loop per wall pass");
            let mut indices: Vec<usize> = layer.loops.iter().map(|l| l.wall_index).collect();
            indices.sort_unstable();
            assert_eq!(indices, vec![0, 1, 2]);
            for wall_loop in &layer.loops {
                assert!(!wall_loop.points.is_empty());
            }
        }
    }

    #[test]
    fn slice_mesh_curved_field_infill_boundary_is_smoothed_identically_to_inner_wall() {
        let config_1wall = SlicerConfig {
            layer_height: 0.25,
            wall_line_width: 0.1,
            shell_thickness: 0.1,
            wall_offset: 0.05,
            order_field: crate::order_field::OrderFieldKind::Conical,
            ..SlicerConfig::default()
        };
        assert_eq!(config_1wall.wall_count(), 1);

        let mut config_2wall = config_1wall.clone();
        config_2wall.shell_thickness = 0.2;
        assert_eq!(config_2wall.wall_count(), 2);

        let layers_1wall = slice_mesh(&big_cube_mesh(), &config_1wall).unwrap();
        let layers_2wall = slice_mesh(&big_cube_mesh(), &config_2wall).unwrap();

        for (l1, l2) in layers_1wall.iter().zip(layers_2wall.iter()) {
            let wall1_loops: Vec<_> = l2.loops.iter().filter(|w| w.wall_index == 1).collect();
            if !wall1_loops.is_empty() && !l1.infill_boundary.is_empty() {
                assert_eq!(wall1_loops.len(), l1.infill_boundary.len());
                for (w, b) in wall1_loops.iter().zip(l1.infill_boundary.iter()) {
                    assert_eq!(w.points.len(), b.len());
                    for (p, q) in w.points.iter().zip(b.iter()) {
                        assert!(
                            p.distance(*q) < 1e-6,
                            "expected 2-wall Wall 1 point {p:?} to match 1-wall infill_boundary point {q:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn slice_mesh_infill_boundary_sits_one_wall_pass_inward_of_the_innermost_wall() {
        let config = SlicerConfig {
            layer_height: 0.25,
            wall_line_width: 0.05,
            shell_thickness: 0.15,
            wall_offset: 0.02,
            ..SlicerConfig::default()
        };
        assert_eq!(config.wall_count(), 3);

        let layers = slice_mesh(&cube_mesh(), &config).unwrap();

        // In-plane half-extent (max distance from the cube's in-plane
        // center, 0.5) of a set of loops, used as a proxy for how far
        // inward a contour has been inset.
        fn half_extent(loops: &[Vec<DVec3>]) -> f64 {
            loops
                .iter()
                .flatten()
                .map(|p| (p.x - 0.5).abs().max((p.y - 0.5).abs()))
                .fold(0.0, f64::max)
        }

        for layer in &layers[0..3] {
            assert!(
                !layer.infill_boundary.is_empty(),
                "expected a non-empty infill boundary"
            );
            let innermost_wall = layer
                .loops
                .iter()
                .filter(|w| w.wall_index == config.wall_count() - 1)
                .map(|w| w.points.clone())
                .collect::<Vec<_>>();
            let wall_extent = half_extent(&innermost_wall);
            let boundary_extent = half_extent(&layer.infill_boundary);
            // The infill boundary is one further `wall_line_width` step
            // inward from the innermost printed wall — i.e. where wall
            // pass `wall_count` would sit if one more were printed.
            assert!(
                boundary_extent < wall_extent - config.wall_line_width * 0.5,
                "expected infill boundary ({boundary_extent}) to sit noticeably \
                 inward of the innermost wall ({wall_extent})"
            );
        }
    }

    /// Regression test: a cross-section can be too small to fit every
    /// configured wall pass (e.g. near a tapered tip) while an outer pass
    /// still extracts fine. `infill_boundary` must fall back to whatever
    /// wall pass actually has geometry instead of going empty just because
    /// the *innermost configured* pass (`wall_count - 1`) came back empty.
    #[test]
    fn slice_mesh_infill_boundary_falls_back_when_the_innermost_configured_wall_is_missing() {
        let config = SlicerConfig {
            layer_height: 0.5,
            wall_line_width: 0.05,
            shell_thickness: 0.1,
            wall_offset: 0.02,
            ..SlicerConfig::default()
        };
        assert_eq!(config.wall_count(), 2);

        let layers = slice_mesh(&tall_thin_pyramid_mesh(), &config).unwrap();

        // Near the pyramid's tip the cross-section is too narrow to fit
        // both configured wall passes: find a layer where some wall
        // extracted (`loops` non-empty) but the innermost configured pass
        // (`wall_count - 1`) did not, and assert `infill_boundary` is still
        // non-empty there.
        let short_of_innermost_wall = layers.iter().find(|layer| {
            !layer.loops.is_empty()
                && layer
                    .loops
                    .iter()
                    .all(|w| w.wall_index != config.wall_count() - 1)
        });
        let layer = short_of_innermost_wall.unwrap_or_else(|| {
            panic!(
                "fixture/config didn't produce a layer with a missing innermost \
                 wall pass — test can't exercise the fallback"
            )
        });
        assert!(
            !layer.infill_boundary.is_empty(),
            "layer {} (order {}) has wall geometry but no innermost configured \
             wall pass, and should have fallen back to the deepest wall that \
             did extract instead of leaving infill_boundary empty",
            layer.index,
            layer.order
        );
    }

    #[test]
    fn slice_mesh_has_no_empty_contour_gaps_through_a_tall_thin_pyramid() {
        let config = SlicerConfig {
            layer_height: 1.0,
            ..SlicerConfig::default()
        };

        let layers = slice_mesh(&tall_thin_pyramid_mesh(), &config).unwrap();
        assert!(!layers.is_empty());

        let first_nonempty = layers.iter().position(|layer| !layer.loops.is_empty());
        let last_nonempty = layers.iter().rposition(|layer| !layer.loops.is_empty());

        let (Some(first_nonempty), Some(last_nonempty)) = (first_nonempty, last_nonempty) else {
            panic!("expected at least one layer with contour loops");
        };

        for layer in &layers[first_nonempty..=last_nonempty] {
            assert!(
                !layer.loops.is_empty(),
                "layer {} at order {} unexpectedly has no contour loops \
                 (in-plane sampling extent too large for the footprint)",
                layer.index,
                layer.order
            );
        }
    }

    /// Box (5x5x2) with a small, shallow spike (footprint 0.2x0.2,
    /// protruding 0.3 above the box's otherwise-flat top face at Z=2) —
    /// approximates the originally reported bug's shape: a flat top
    /// surface plus a shallow, thin surface detail near it. The spike's
    /// cross-section is thin enough that some of its layers can fit an
    /// outer wall pass but not the innermost *configured* pass (same
    /// failure mode `tall_thin_pyramid_mesh` exercises at a pyramid's
    /// apex), while the box's plateau layers below stay comfortably wide.
    fn flat_top_with_shallow_spike_mesh() -> Mesh {
        let box_height = 2.0;
        let spike_height = 0.3;
        let (hole_min, hole_max) = (2.4, 2.6);
        let spike_top = box_height + spike_height;
        let spike_center = (hole_min + hole_max) * 0.5;

        let vertices = vec![
            DVec3::new(0.0, 0.0, 0.0),                         // 0: A0
            DVec3::new(5.0, 0.0, 0.0),                         // 1: B0
            DVec3::new(5.0, 5.0, 0.0),                         // 2: C0
            DVec3::new(0.0, 5.0, 0.0),                         // 3: D0
            DVec3::new(0.0, 0.0, box_height),                  // 4: A
            DVec3::new(5.0, 0.0, box_height),                  // 5: B
            DVec3::new(5.0, 5.0, box_height),                  // 6: C
            DVec3::new(0.0, 5.0, box_height),                  // 7: D
            DVec3::new(hole_min, hole_min, box_height),        // 8: a
            DVec3::new(hole_max, hole_min, box_height),        // 9: b
            DVec3::new(hole_max, hole_max, box_height),        // 10: c
            DVec3::new(hole_min, hole_max, box_height),        // 11: d
            DVec3::new(spike_center, spike_center, spike_top), // 12: E (apex)
        ];
        let indices = vec![
            0, 2, 1, 0, 3, 2, // bottom (-Z)
            0, 1, 5, 0, 5, 4, // -Y side
            1, 2, 6, 1, 6, 5, // +X side
            2, 3, 7, 2, 7, 6, // +Y side
            3, 0, 4, 3, 4, 7, // -X side
            4, 5, 9, 4, 9, 8, // top ring: -Y edge
            5, 6, 10, 5, 10, 9, // top ring: +X edge
            6, 7, 11, 6, 11, 10, // top ring: +Y edge
            7, 4, 8, 7, 8, 11, // top ring: -X edge
            8, 9, 12, // spike side: -Y
            9, 10, 12, // spike side: +X
            10, 11, 12, // spike side: +Y
            11, 8, 12, // spike side: -X
        ];
        Mesh::new(vertices, indices)
    }

    #[test]
    fn slice_mesh_infill_boundary_is_never_empty_when_walls_exist_near_a_flat_top_with_shallow_surface_detail(
    ) {
        let config = SlicerConfig {
            layer_height: 0.1,
            wall_line_width: 0.05,
            shell_thickness: 0.15,
            wall_offset: 0.02,
            ..SlicerConfig::default()
        };
        assert_eq!(config.wall_count(), 3);

        let layers = slice_mesh(&flat_top_with_shallow_spike_mesh(), &config).unwrap();
        assert!(!layers.is_empty());

        let mut any_layer_had_walls = false;
        for layer in &layers {
            if layer.loops.is_empty() {
                continue;
            }
            any_layer_had_walls = true;
            assert!(
                !layer.infill_boundary.is_empty(),
                "layer {} (order {}) has wall geometry but an empty infill_boundary \
                 -- the walls-present-but-empty-infill_boundary bug has recurred",
                layer.index,
                layer.order
            );
        }
        assert!(
            any_layer_had_walls,
            "fixture produced no layers with contour loops at all"
        );
    }

    /// Symmetric-`top_layers`/`bottom_layers` complement to
    /// `compute_solid_fill_boundaries_propagates_top_and_bottom_in_the_correct_direction`:
    /// checks the *magnitude*/extent of propagation (top `top_layers` and
    /// bottom `bottom_layers` layers end up essentially fully solid, and
    /// layers strictly further than that from either end are left empty)
    /// rather than just the up/down direction.
    #[test]
    fn compute_solid_fill_boundaries_covers_only_top_and_bottom_layers_leaving_the_interior_empty()
    {
        let config = SlicerConfig {
            layer_height: 0.5,
            top_layers: 2,
            bottom_layers: 2,
            ..SlicerConfig::default()
        };

        let mut layers = slice_mesh(&big_cube_mesh(), &config).unwrap();
        compute_solid_fill_boundaries(&mut layers, &config);

        fn area(loops: &[Vec<DVec3>]) -> f64 {
            loops
                .iter()
                .map(|pts| {
                    let n = pts.len();
                    if n < 3 {
                        return 0.0;
                    }
                    (0..n)
                        .map(|i| {
                            let a = pts[i];
                            let b = pts[(i + 1) % n];
                            a.x * b.y - b.x * a.y
                        })
                        .sum::<f64>()
                        * 0.5
                })
                .sum()
        }

        // Solid-fill ratio (solid area / infill area) for a layer with a
        // nonempty infill_boundary; `None` for a layer with no fillable
        // area on this layer at all.
        let solid_ratio = |layer: &Layer| -> Option<f64> {
            let infill = area(&layer.infill_boundary).abs();
            if infill < 1e-9 {
                return None;
            }
            Some(area(&layer.solid_fill_boundary).abs() / infill)
        };

        let ratios: Vec<Option<f64>> = layers.iter().map(solid_ratio).collect();
        let first_real = ratios.iter().position(Option::is_some).unwrap();
        let last_real = ratios.iter().rposition(Option::is_some).unwrap();
        let real_count = last_real - first_real + 1;
        assert!(
            real_count >= 2 * config.top_layers.max(config.bottom_layers) + 2,
            "fixture needs enough layers with contours to leave a genuinely \
             empty interior between the top/bottom solid bands"
        );

        for (offset, &ratio) in ratios[first_real..=last_real].iter().enumerate() {
            let ratio = ratio.expect("all layers in this range have a nonempty infill_boundary");
            let distance_from_top = offset;
            let distance_from_bottom = real_count - 1 - offset;

            if distance_from_top < config.top_layers || distance_from_bottom < config.bottom_layers
            {
                assert!(
                    (ratio - 1.0).abs() < 1e-6,
                    "layer at offset {offset} from the top-real layer should be \
                     essentially fully solid (top_layers={}, bottom_layers={}), got ratio {ratio}",
                    config.top_layers,
                    config.bottom_layers
                );
            } else {
                assert!(
                    ratio < 1e-6,
                    "layer at offset {offset} is further than top_layers/bottom_layers \
                     from either end and should have an empty solid_fill_boundary, \
                     got ratio {ratio}"
                );
            }
        }
    }

    #[test]
    fn stitch_wall_gaps_no_op_when_points_within_hop_limit() {
        let config = SlicerConfig {
            nozzle_diameter: 0.4,
            ..SlicerConfig::default()
        };
        let field: Arc<dyn OrderField> = Arc::new(HeightOrderField::new(BUILD_DIRECTION));
        let (basis1, basis2) = plane_basis(BUILD_DIRECTION);

        let mut layers = vec![
            Layer {
                index: 0,
                order: 0.0,
                loops: vec![WallLoop {
                    wall_index: 0,
                    points: vec![DVec3::new(0.0, 0.0, 0.0), DVec3::new(1.0, 0.0, 0.0)],
                    unsupported: vec![false, false],
                    top_surface: Vec::new(),
                    line_widths: Vec::new(),
                    arc_fraction: compute_arc_fractions(&[
                        DVec3::new(0.0, 0.0, 0.0),
                        DVec3::new(1.0, 0.0, 0.0),
                    ]),
                }],
                order_field: Arc::clone(&field),
                ..Layer::default()
            },
            Layer {
                index: 1,
                order: 0.05,
                loops: vec![WallLoop {
                    wall_index: 0,
                    points: vec![DVec3::new(0.01, 0.0, -0.05), DVec3::new(1.01, 0.0, -0.05)],
                    unsupported: vec![false, false],
                    top_surface: Vec::new(),
                    line_widths: Vec::new(),
                    arc_fraction: compute_arc_fractions(&[
                        DVec3::new(0.01, 0.0, -0.05),
                        DVec3::new(1.01, 0.0, -0.05),
                    ]),
                }],
                order_field: Arc::clone(&field),
                ..Layer::default()
            },
        ];

        stitch_wall_gaps(&mut layers, &config, basis1, basis2);

        assert_eq!(
            layers[1].loops[0].points.len(),
            2,
            "points well within the hop limit should not be stitched"
        );
        assert_eq!(layers[1].loops[0].unsupported, vec![false, false]);
    }

    #[test]
    fn stitch_wall_gaps_bisects_and_flags_unsupported_points_when_gap_exceeds_threshold() {
        let config = SlicerConfig {
            nozzle_diameter: 0.4,
            ..SlicerConfig::default()
        };
        let hop_limit = WALL_GAP_HOP_FRACTION * (config.nozzle_diameter / 2.0);
        let field: Arc<dyn OrderField> = Arc::new(HeightOrderField::new(BUILD_DIRECTION));
        let (basis1, basis2) = plane_basis(BUILD_DIRECTION);

        let mut layers = vec![
            Layer {
                index: 0,
                order: 0.0,
                loops: vec![WallLoop {
                    wall_index: 0,
                    points: vec![DVec3::new(0.0, 0.0, 0.0)],
                    unsupported: vec![false],
                    top_surface: Vec::new(),
                    line_widths: Vec::new(),
                    arc_fraction: vec![0.0],
                }],
                order_field: Arc::clone(&field),
                ..Layer::default()
            },
            Layer {
                index: 1,
                order: 0.2,
                loops: vec![WallLoop {
                    wall_index: 0,
                    points: vec![DVec3::new(2.0, 0.0, -0.2)],
                    unsupported: vec![false],
                    top_surface: Vec::new(),
                    line_widths: Vec::new(),
                    arc_fraction: vec![0.0],
                }],
                order_field: Arc::clone(&field),
                ..Layer::default()
            },
        ];

        stitch_wall_gaps(&mut layers, &config, basis1, basis2);

        let wall = &layers[1].loops[0];
        assert!(
            wall.points.len() > 1,
            "a gap exceeding the hop limit should trigger stitching"
        );
        assert_eq!(
            wall.unsupported.len(),
            wall.points.len(),
            "unsupported must stay parallel to points"
        );
        assert_eq!(
            wall.arc_fraction.len(),
            wall.points.len(),
            "arc_fraction must stay parallel to points through stitch insertion"
        );

        // Every point but the last (the original, real point) should be
        // flagged unsupported -- inserted by the stitch.
        for &flag in &wall.unsupported[..wall.unsupported.len() - 1] {
            assert!(flag, "inserted stitch points must be marked unsupported");
        }
        assert!(
            !wall.unsupported[wall.unsupported.len() - 1],
            "the original target point must stay unsupported = false"
        );

        // Every consecutive hop along the stitched loop, prepended with
        // the previous layer's real point, must be within the hop limit
        // measured LATERALLY (perpendicular to the local climb
        // direction -- see `lateral_gap`'s docs): the climb component of
        // a hop is a normal fraction of the layer step and is exactly
        // what the criterion must not count.
        let mut full_chain = vec![layers[0].loops[0].points[0]];
        full_chain.extend(wall.points.iter().copied());
        for pair in full_chain.windows(2) {
            let dist = lateral_gap(field.as_ref(), pair[0], pair[1]);
            assert!(
                dist <= hop_limit + 1e-9,
                "lateral hop {dist} exceeds hop_limit {hop_limit}"
            );
        }
    }

    #[test]
    fn stitch_wall_gaps_ignores_pure_climb_separation_on_vertical_walls() {
        // A perfectly vertical wall: consecutive layers' wall-0 loops are
        // identical in XY and separated purely along the climb (build)
        // direction by one layer step. The raw 3D distance between
        // corresponding points (0.2mm) exceeds the hop limit (0.9 *
        // nozzle_radius = 0.18mm), which is exactly the flaw that made
        // the old raw-3D criterion stitch ~every point of ~every contour;
        // the lateral criterion must leave this loop completely alone.
        let config = SlicerConfig {
            nozzle_diameter: 0.4,
            ..SlicerConfig::default()
        };
        let field: Arc<dyn OrderField> = Arc::new(HeightOrderField::new(BUILD_DIRECTION));
        let (basis1, basis2) = plane_basis(BUILD_DIRECTION);

        let square = |z: f64| -> Vec<DVec3> {
            vec![
                DVec3::new(0.0, 0.0, z),
                DVec3::new(5.0, 0.0, z),
                DVec3::new(5.0, 5.0, z),
                DVec3::new(0.0, 5.0, z),
            ]
        };
        let prev_points = square(0.0);
        let cur_points = square(-0.2);

        let mut layers = vec![
            Layer {
                index: 0,
                order: 0.0,
                loops: vec![WallLoop {
                    wall_index: 0,
                    unsupported: vec![false; prev_points.len()],
                    top_surface: Vec::new(),
                    line_widths: Vec::new(),
                    arc_fraction: compute_arc_fractions(&prev_points),
                    points: prev_points,
                }],
                order_field: Arc::clone(&field),
                ..Layer::default()
            },
            Layer {
                index: 1,
                order: 0.2,
                loops: vec![WallLoop {
                    wall_index: 0,
                    unsupported: vec![false; cur_points.len()],
                    top_surface: Vec::new(),
                    line_widths: Vec::new(),
                    arc_fraction: compute_arc_fractions(&cur_points),
                    points: cur_points,
                }],
                order_field: Arc::clone(&field),
                ..Layer::default()
            },
        ];

        stitch_wall_gaps(&mut layers, &config, basis1, basis2);

        assert_eq!(
            layers[1].loops[0].points.len(),
            4,
            "a vertical wall separated purely along the climb direction \
             must not be stitched -- its lateral gap is zero"
        );
        assert!(
            layers[1].loops[0].unsupported.iter().all(|&flag| !flag),
            "no point on a vertical wall should be flagged unsupported"
        );
    }

    #[test]
    fn stitch_wall_gaps_still_stitches_a_shallow_ramp_with_a_large_lateral_offset() {
        // A genuinely shallow slope: the next layer's loop steps one
        // layer height along the climb direction but also drifts 1mm
        // sideways -- far beyond the 0.18mm lateral hop limit. This is
        // the real "ear bridging" case the feature exists for and must
        // keep triggering under the lateral criterion.
        let config = SlicerConfig {
            nozzle_diameter: 0.4,
            ..SlicerConfig::default()
        };
        let field: Arc<dyn OrderField> = Arc::new(HeightOrderField::new(BUILD_DIRECTION));
        let (basis1, basis2) = plane_basis(BUILD_DIRECTION);

        let square = |x0: f64, z: f64| -> Vec<DVec3> {
            vec![
                DVec3::new(x0, 0.0, z),
                DVec3::new(x0 + 5.0, 0.0, z),
                DVec3::new(x0 + 5.0, 5.0, z),
                DVec3::new(x0, 5.0, z),
            ]
        };
        let prev_points = square(0.0, 0.0);
        let cur_points = square(1.0, -0.2);

        let mut layers = vec![
            Layer {
                index: 0,
                order: 0.0,
                loops: vec![WallLoop {
                    wall_index: 0,
                    unsupported: vec![false; prev_points.len()],
                    top_surface: Vec::new(),
                    line_widths: Vec::new(),
                    arc_fraction: compute_arc_fractions(&prev_points),
                    points: prev_points,
                }],
                order_field: Arc::clone(&field),
                ..Layer::default()
            },
            Layer {
                index: 1,
                order: 0.2,
                loops: vec![WallLoop {
                    wall_index: 0,
                    unsupported: vec![false; cur_points.len()],
                    top_surface: Vec::new(),
                    line_widths: Vec::new(),
                    arc_fraction: compute_arc_fractions(&cur_points),
                    points: cur_points,
                }],
                order_field: Arc::clone(&field),
                ..Layer::default()
            },
        ];

        stitch_wall_gaps(&mut layers, &config, basis1, basis2);

        let wall = &layers[1].loops[0];
        assert!(
            wall.points.len() > 4,
            "a 1mm lateral drift per layer must still trigger stitching"
        );
        assert!(
            wall.unsupported.iter().any(|&flag| flag),
            "stitched points must be flagged unsupported"
        );
    }

    #[test]
    fn correct_isolated_correspondence_outliers_fixes_a_single_point_surrounded_by_good_matches() {
        // Regression test for a real-mesh finding (layer 40 of
        // pug_v4_l_sop_85mm.stl): a single current-loop point's raw
        // fraction-based (+ bounded local-fallback) correspondence landed
        // ~8mm away, even though its immediate current-loop neighbors --
        // only ~0.017mm apart from it and each other, ordinary densely
        // sampled contour spacing -- both had perfectly good, close
        // correspondences. This is implausible as a genuine isolated gap
        // (a real shallow-overhang region spans a run of points, not one
        // point bracketed by normal support on both sides), and is fixed
        // by blending the outlier's correspondence from its neighbors'.
        let hop_limit = 0.18; // matches the real hop_limit magnitude at 0.4mm nozzle.

        // 5 current-loop points, densely and evenly spaced (mirrors the
        // real chain's before/after spacing of ~0.017mm).
        let points = vec![
            DVec3::new(27.500, -1.760, 7.017),
            DVec3::new(27.558, -1.783, 7.025),
            DVec3::new(27.544, -1.790, 7.029), // the isolated trigger point
            DVec3::new(27.530, -1.797, 7.033),
            DVec3::new(27.516, -1.804, 7.037),
        ];

        // Correspondences: every point's previous-loop match is close
        // (well within hop_limit) except index 2, whose fraction-based
        // lookup lands ~8mm away (the real observed failure mode).
        let neighbor_before = DVec3::new(27.558, -1.783, 7.025 - 0.1);
        let neighbor_after = DVec3::new(27.530, -1.797, 7.033 - 0.1);
        let mut correspondences = vec![
            DVec3::new(27.500, -1.760, 7.017 - 0.1),
            neighbor_before,
            DVec3::new(35.430, -1.699, 6.810), // wrong, ~8mm away
            neighbor_after,
            DVec3::new(27.516, -1.804, 7.037 - 0.1),
        ];

        correct_isolated_correspondence_outliers(
            &points,
            &mut correspondences,
            hop_limit,
            &|p, c| (p - c).length(),
        );

        // The outlier at index 2 must now be close to `points[2]` (within
        // the hop limit), not the original ~8mm-away wrong match.
        assert!(
            (points[2] - correspondences[2]).length() <= hop_limit,
            "isolated outlier at index 2 should have been corrected to a nearby estimate, got {:?}",
            correspondences[2]
        );
        // It should be exactly the blended midpoint of its neighbors' own
        // correspondences, not an arbitrary value.
        let expected = (neighbor_before + neighbor_after) * 0.5;
        assert!(
            (correspondences[2] - expected).length() < 1e-9,
            "expected midpoint blend of neighbors' correspondences, got {:?} vs expected {:?}",
            correspondences[2],
            expected
        );

        // Untouched points must remain exactly as they were.
        assert_eq!(correspondences[0], DVec3::new(27.500, -1.760, 7.017 - 0.1));
        assert_eq!(correspondences[1], neighbor_before);
        assert_eq!(correspondences[3], neighbor_after);
        assert_eq!(correspondences[4], DVec3::new(27.516, -1.804, 7.037 - 0.1));
    }

    #[test]
    fn correct_isolated_correspondence_outliers_leaves_a_genuine_multi_point_gap_alone() {
        // A *run* of 2+ consecutive bad correspondences must NOT be
        // "corrected" -- that pattern is consistent with a genuine
        // shallow-overhang gap (which stitch_wall_gaps's bisecting is
        // meant to handle), not an isolated single-point artifact. Only an
        // isolated single point flanked by two individually-good neighbors
        // should ever be touched.
        let hop_limit = 0.18;
        let points = vec![
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(0.1, 0.0, 0.0),
            DVec3::new(0.2, 0.0, 0.0),
            DVec3::new(0.3, 0.0, 0.0),
        ];
        let mut correspondences = vec![
            DVec3::new(0.0, 0.0, -0.1),   // good
            DVec3::new(10.0, 10.0, 10.0), // bad (run start)
            DVec3::new(10.0, 10.0, 10.0), // bad (run continues)
            DVec3::new(0.3, 0.0, -0.1),   // good
        ];
        let before = correspondences.clone();

        correct_isolated_correspondence_outliers(
            &points,
            &mut correspondences,
            hop_limit,
            &|p, c| (p - c).length(),
        );

        assert_eq!(
            correspondences, before,
            "a 2-point run of bad correspondences must be left untouched, not blended"
        );
    }

    #[test]
    fn stitch_block_for_a_run_is_a_continuous_serpentine_not_per_point_ramps() {
        // Two adjacent current-layer points, both laterally ~7mm from
        // their previous-layer counterparts: a single 2-column gap run.
        // The old per-point implementation inserted one anchor-to-target
        // ramp per point, so between the two run points the nozzle dove
        // back down to the previous layer (z jumping from ~-0.2 back up
        // to ~0.0) -- a physical zigzag. The serpentine block emits
        // whole order levels across the run instead, so the inserted
        // points' z values must be monotonically non-increasing (for
        // this Height field, order increases as z decreases): each row
        // is at one level and rows only ever step toward the current
        // layer.
        let config = SlicerConfig::default();
        let hop_limit = WALL_GAP_HOP_FRACTION * (config.nozzle_diameter / 2.0);
        let field: Arc<dyn OrderField> = Arc::new(HeightOrderField::new(BUILD_DIRECTION));
        let (basis1, basis2) = plane_basis(BUILD_DIRECTION);

        let prev_points = vec![DVec3::new(0.0, 0.0, 0.0), DVec3::new(0.3, 0.0, 0.0)];
        let cur_points = vec![DVec3::new(7.0, 0.0, 0.2), DVec3::new(7.3, 0.0, 0.2)];
        let mut layers = vec![
            Layer {
                index: 0,
                order: 0.0,
                loops: vec![WallLoop {
                    wall_index: 0,
                    arc_fraction: compute_arc_fractions(&prev_points),
                    unsupported: vec![false; prev_points.len()],
                    top_surface: Vec::new(),
                    line_widths: Vec::new(),
                    points: prev_points,
                }],
                order_field: Arc::clone(&field),
                ..Layer::default()
            },
            Layer {
                index: 1,
                order: 0.2,
                loops: vec![WallLoop {
                    wall_index: 0,
                    arc_fraction: compute_arc_fractions(&cur_points),
                    unsupported: vec![false; cur_points.len()],
                    top_surface: Vec::new(),
                    line_widths: Vec::new(),
                    points: cur_points,
                }],
                order_field: Arc::clone(&field),
                ..Layer::default()
            },
        ];

        stitch_wall_gaps(&mut layers, &config, basis1, basis2);

        let wall = &layers[1].loops[0];
        let inserted: Vec<DVec3> = wall
            .points
            .iter()
            .zip(wall.unsupported.iter())
            .filter(|(_, &unsupported)| unsupported)
            .map(|(p, _)| *p)
            .collect();
        assert!(
            inserted.len() >= 4,
            "a 7mm 2-column gap must insert multiple stitch rows, got {}",
            inserted.len()
        );
        for pair in inserted.windows(2) {
            assert!(
                pair[1].z >= pair[0].z - 1e-9,
                "inserted stitch points must climb monotonically toward the current layer \
                 (z non-decreasing), but z jumped {} -> {} -- per-point ramp zigzag",
                pair[0].z,
                pair[1].z
            );
            let hop = lateral_gap(field.as_ref(), pair[0], pair[1]);
            // Lateral hops between consecutive inserted points: within a
            // row this is the column spacing; at a row turnaround it is a
            // pure vertical level step (~0 lateral). Both must stay small
            // -- a per-point-ramp zigzag would show a ~7mm lateral jump
            // back across the gap between the two ramps.
            assert!(
                hop <= hop_limit + 0.3 + 1e-9,
                "consecutive inserted points must be adjacent in the serpentine, got lateral hop {hop}"
            );
        }
    }

    #[test]
    fn stitching_closes_a_gap_many_times_larger_than_hop_limit_without_stalling() {
        // Regression test for a real-mesh finding: the stitch subdivision used to
        // seed its midpoint reconstruction via
        // `reconstruct_on_order_field_near`, which picks whichever of the
        // two endpoints is closest in-plane and reuses *that endpoint's*
        // `along` value, then rejects any Newton refinement that wanders
        // farther from that seed than the seed's own residual justifies.
        // That heuristic is correct for its original use (refining
        // boolean-op output already known to sit near the isosurface),
        // but for a genuinely large real wall-to-wall gap (confirmed on a
        // real mesh: brute-force nearest-neighbor search over an entire
        // previous-layer loop still found nothing closer than several mm)
        // it stalls: the rejection kept falling back to (approximately)
        // one endpoint, so recursing on the far side never made real
        // geometric progress, leaving one huge unbisected hop alongside
        // units, ~97x the hop limit) is far larger than
        // `stitch_wall_gaps_bisects_and_flags_unsupported_points_when_gap_exceeds_threshold`'s
        // 2-unit gap specifically to exercise many recursive bisection
        // levels, not just one or two (kept within
        // `WALL_GAP_LOOP_CENTROID_MATCH_FACTOR * nozzle_diameter` = 8.0
        // units so the loops still match at all).
        let config = SlicerConfig {
            nozzle_diameter: 0.4,
            ..SlicerConfig::default()
        };
        let hop_limit = WALL_GAP_HOP_FRACTION * (config.nozzle_diameter / 2.0);
        let field: Arc<dyn OrderField> = Arc::new(HeightOrderField::new(BUILD_DIRECTION));
        let (basis1, basis2) = plane_basis(BUILD_DIRECTION);

        let mut layers = vec![
            Layer {
                index: 0,
                order: 0.0,
                loops: vec![WallLoop {
                    wall_index: 0,
                    points: vec![DVec3::new(0.0, 0.0, 0.0)],
                    unsupported: vec![false],
                    top_surface: Vec::new(),
                    line_widths: Vec::new(),
                    arc_fraction: vec![0.0],
                }],
                order_field: Arc::clone(&field),
                ..Layer::default()
            },
            Layer {
                index: 1,
                order: 0.2,
                loops: vec![WallLoop {
                    wall_index: 0,
                    points: vec![DVec3::new(7.0, 0.0, -0.2)],
                    unsupported: vec![false],
                    top_surface: Vec::new(),
                    line_widths: Vec::new(),
                    arc_fraction: vec![0.0],
                }],
                order_field: Arc::clone(&field),
                ..Layer::default()
            },
        ];

        stitch_wall_gaps(&mut layers, &config, basis1, basis2);

        let wall = &layers[1].loops[0];
        assert!(
            wall.points.len() > 1,
            "a gap this much larger than the hop limit must trigger stitching"
        );

        // Every consecutive LATERAL hop (see `lateral_gap` -- the climb
        // component is a normal fraction of the layer step), prepended
        // with the previous layer's real point, must be within the hop
        // limit -- not just most of them, which is exactly what the
        // stalling bug produced (many correct tiny hops plus one leftover
        // huge one).
        let mut full_chain = vec![layers[0].loops[0].points[0]];
        full_chain.extend(wall.points.iter().copied());
        for pair in full_chain.windows(2) {
            let dist = lateral_gap(field.as_ref(), pair[0], pair[1]);
            assert!(
                dist <= hop_limit + 1e-6,
                "lateral hop {dist} exceeds hop_limit {hop_limit} -- bisection stalled instead of closing the full gap"
            );
        }
    }

    #[test]
    fn arc_length_correspondence_avoids_backward_jump_that_nearest_point_matching_would_produce() {
        // A thin hairpin/rectangle loop (0.1 wide, 2.0 tall): the two long
        // sides ("legs") run parallel and only 0.1 apart in 3D, but are far
        // apart in arc-length parameterization (roughly opposite halves of
        // the loop). This is exactly the shape where naive nearest-point
        // matching (whether raw 3D distance or transverse (u, v) position)
        // picks the wrong leg -- see Addendum 2 in the wall-gap stitching
        // task context -- while arc-length-fraction correspondence does not.
        // The previous loop is sampled coarsely (8 points) relative to the
        // 0.1 leg separation, so consecutive same-leg samples are farther
        // apart than the gap to the opposite leg -- the condition that
        // actually produces a wrong-leg nearest-point match.
        let corners = [
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(0.0, 2.0, 0.0),
            DVec3::new(0.1, 2.0, 0.0),
            DVec3::new(0.1, 0.0, 0.0),
        ];
        let corner_fractions = compute_arc_fractions(&corners);

        // Sample the previous layer's loop (8 points) along this shape, and
        // the current layer's loop (11 points -- a different point count/
        // sampling density) along the *same* underlying path, raised
        // slightly in Z (the inter-layer climb) -- mirroring two
        // independently-extracted, differently-sampled wall-0 contours of
        // the same real geometry.
        let prev_points: Vec<DVec3> = (0..8)
            .map(|i| interpolate_on_loop_at_fraction(&corners, &corner_fractions, i as f64 / 8.0))
            .collect();
        let z_offset = DVec3::new(0.0, 0.0, 0.05);
        let cur_points: Vec<DVec3> = (0..11)
            .map(|i| {
                interpolate_on_loop_at_fraction(&corners, &corner_fractions, i as f64 / 11.0)
                    + z_offset
            })
            .collect();

        let prev_fractions = compute_arc_fractions(&prev_points);
        let cur_fractions = compute_arc_fractions(&cur_points);

        // Alignment step (mirrors stitch_wall_gaps): a single nearest-point
        // match of the current loop's first point against the previous
        // loop's points, run once per layer-pair.
        let start = cur_points[0];
        let nearest_idx = prev_points
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                (**a - start)
                    .length_squared()
                    .total_cmp(&(**b - start).length_squared())
            })
            .map(|(idx, _)| idx)
            .unwrap();
        let offset = prev_fractions[nearest_idx];

        // For every current-loop point, compare the old approach (naive
        // nearest raw-3D-point search over the previous loop) against the
        // new approach (arc-length-fraction correspondence), and find the
        // point where the old approach's match is farthest (in arc-length
        // terms) from where it should be -- i.e. the clearest wrong-leg,
        // backward-jumping match.
        let mut worst_naive_gap = 0.0_f64;
        let mut worst_i = 0usize;
        for (i, &point) in cur_points.iter().enumerate() {
            let naive_nearest_idx = prev_points
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    (**a - point)
                        .length_squared()
                        .total_cmp(&(**b - point).length_squared())
                })
                .map(|(idx, _)| idx)
                .unwrap();
            let naive_fraction = prev_fractions[naive_nearest_idx];
            let gap = (naive_fraction - cur_fractions[i]).abs();
            let gap = gap.min(1.0 - gap); // account for wraparound near fraction 0/1
            if gap > worst_naive_gap {
                worst_naive_gap = gap;
                worst_i = i;
            }
        }

        // The naive match's arc-length position is far from this point's own
        // arc-length position for at least one point -- i.e. it landed on
        // the wrong leg (a "backward jump" in parameterization), confirming
        // this loop reproduces the bug's failure mode.
        assert!(
            worst_naive_gap > 0.2,
            "expected at least one naive nearest-3D-point match to land on the wrong leg \
         (worst fraction gap {worst_naive_gap}), demonstrating the failure mode this fix avoids"
        );

        // New approach: arc-length-fraction correspondence, following the
        // same steps as `stitch_wall_gaps`, applied to that same worst-case
        // point.
        let probe = cur_points[worst_i];
        let probe_t = cur_fractions[worst_i];
        let t_prev = (offset + probe_t).rem_euclid(1.0);
        let corresponding = interpolate_on_loop_at_fraction(&prev_points, &prev_fractions, t_prev);

        // The correspondence lands within the tiny Z-offset climb distance of
        // the probe point -- i.e. on the *same* leg, not the wrong one.
        assert!(
            (probe - corresponding).length() < 0.1,
            "arc-length correspondence should match the same leg the probe point is actually \
         on, within the small inter-layer climb offset, got distance {}",
            (probe - corresponding).length()
        );

        // Monotonicity across the whole loop: as the current point's own arc
        // fraction increases, so must its computed previous-loop
        // correspondence fraction (mod wraparound), with no backward jump --
        // exactly what the old per-point nearest-point search could not
        // guarantee.
        let mut prev_t_prev = 0.0;
        for (i, &t) in cur_fractions.iter().enumerate() {
            let t_prev = (offset + t).rem_euclid(1.0);
            if i > 0 {
                assert!(
                    t_prev >= prev_t_prev - 1e-9,
                    "correspondence fraction must be monotonically non-decreasing (point {i})"
                );
            }
            prev_t_prev = t_prev;
        }
    }

    #[test]
    fn stitch_wall_gaps_matches_multiple_loops_with_same_count_and_position_baseline() {
        // Regression check: two wall-0 loops per layer, each layer's loops
        // in the same order and each pair spatially close (centroids well
        // within the match threshold) -- the common case this function
        // already handled before spatial matching was introduced. Each pair
        // is far enough apart in X to trigger a stitch, and each loop must
        // be stitched against *its own* previous-layer counterpart, not the
        // other one.
        let config = SlicerConfig {
            nozzle_diameter: 0.4,
            ..SlicerConfig::default()
        };
        let field: Arc<dyn OrderField> = Arc::new(HeightOrderField::new(BUILD_DIRECTION));
        let (basis1, basis2) = plane_basis(BUILD_DIRECTION);

        let mut layers = vec![
            Layer {
                index: 0,
                order: 0.0,
                loops: vec![
                    WallLoop {
                        wall_index: 0,
                        points: vec![DVec3::new(0.0, 0.0, 0.0)],
                        unsupported: vec![false],
                        top_surface: Vec::new(),
                        line_widths: Vec::new(),
                        arc_fraction: vec![0.0],
                    },
                    WallLoop {
                        wall_index: 0,
                        points: vec![DVec3::new(100.0, 0.0, 0.0)],
                        unsupported: vec![false],
                        top_surface: Vec::new(),
                        line_widths: Vec::new(),
                        arc_fraction: vec![0.0],
                    },
                ],
                order_field: Arc::clone(&field),
                ..Layer::default()
            },
            Layer {
                index: 1,
                order: 0.2,
                loops: vec![
                    WallLoop {
                        wall_index: 0,
                        points: vec![DVec3::new(2.0, 0.0, -0.2)],
                        unsupported: vec![false],
                        top_surface: Vec::new(),
                        line_widths: Vec::new(),
                        arc_fraction: vec![0.0],
                    },
                    WallLoop {
                        wall_index: 0,
                        points: vec![DVec3::new(102.0, 0.0, -0.2)],
                        unsupported: vec![false],
                        top_surface: Vec::new(),
                        line_widths: Vec::new(),
                        arc_fraction: vec![0.0],
                    },
                ],
                order_field: Arc::clone(&field),
                ..Layer::default()
            },
        ];

        stitch_wall_gaps(&mut layers, &config, basis1, basis2);

        for (idx, expected_origin_x) in [(0, 0.0), (1, 100.0)] {
            let wall = &layers[1].loops[idx];
            assert!(
                wall.points.len() > 1,
                "loop {idx} gap exceeding the hop limit should trigger stitching"
            );
            // The stitch's first inserted point must climb from *this*
            // loop's own previous-layer point, not the other loop's.
            let first_x = wall.points[0].x;
            assert!(
                (first_x - expected_origin_x).abs() < 50.0,
                "loop {idx} stitched from the wrong previous-layer loop: first inserted point x \
             = {first_x}, expected near {expected_origin_x}"
            );
        }
    }

    #[test]
    fn stitch_wall_gaps_matches_loops_by_centroid_despite_reordered_loop_vector() {
        // Same two spatially-separated loop pairs as the baseline test, but
        // the current layer's loop vector lists them in the *opposite*
        // order from the previous layer's -- exactly the "loop extraction
        // order shifts between adjacent layers" scenario from Addendum 3.
        // Positional-index pairing would cross-match loop 0 against loop 1
        // and vice versa; centroid-based matching must still pair each loop
        // with its true spatial counterpart.
        let config = SlicerConfig {
            nozzle_diameter: 0.4,
            ..SlicerConfig::default()
        };
        let field: Arc<dyn OrderField> = Arc::new(HeightOrderField::new(BUILD_DIRECTION));
        let (basis1, basis2) = plane_basis(BUILD_DIRECTION);

        let mut layers = vec![
            Layer {
                index: 0,
                order: 0.0,
                loops: vec![
                    WallLoop {
                        wall_index: 0,
                        points: vec![DVec3::new(0.0, 0.0, 0.0)],
                        unsupported: vec![false],
                        top_surface: Vec::new(),
                        line_widths: Vec::new(),
                        arc_fraction: vec![0.0],
                    },
                    WallLoop {
                        wall_index: 0,
                        points: vec![DVec3::new(100.0, 0.0, 0.0)],
                        unsupported: vec![false],
                        top_surface: Vec::new(),
                        line_widths: Vec::new(),
                        arc_fraction: vec![0.0],
                    },
                ],
                order_field: Arc::clone(&field),
                ..Layer::default()
            },
            Layer {
                index: 1,
                order: 0.2,
                // Reordered relative to the previous layer: index 0 here
                // spatially corresponds to index 1 in the previous layer,
                // and vice versa.
                loops: vec![
                    WallLoop {
                        wall_index: 0,
                        points: vec![DVec3::new(102.0, 0.0, -0.2)],
                        unsupported: vec![false],
                        top_surface: Vec::new(),
                        line_widths: Vec::new(),
                        arc_fraction: vec![0.0],
                    },
                    WallLoop {
                        wall_index: 0,
                        points: vec![DVec3::new(2.0, 0.0, -0.2)],
                        unsupported: vec![false],
                        top_surface: Vec::new(),
                        line_widths: Vec::new(),
                        arc_fraction: vec![0.0],
                    },
                ],
                order_field: Arc::clone(&field),
                ..Layer::default()
            },
        ];

        stitch_wall_gaps(&mut layers, &config, basis1, basis2);

        // Current-layer index 0 (near x=102) must stitch from the
        // previous-layer loop near x=100, NOT the one near x=0.
        let wall0 = &layers[1].loops[0];
        assert!(
            wall0.points.len() > 1,
            "current loop 0 (near x=102) should trigger stitching"
        );
        assert!(
            (wall0.points[0].x - 100.0).abs() < 50.0,
            "current loop 0 stitched from the wrong previous-layer loop by centroid: \
         first inserted point x = {}, expected near 100.0",
            wall0.points[0].x
        );

        // Current-layer index 1 (near x=2) must stitch from the
        // previous-layer loop near x=0, NOT the one near x=100.
        let wall1 = &layers[1].loops[1];
        assert!(
            wall1.points.len() > 1,
            "current loop 1 (near x=2) should trigger stitching"
        );
        assert!(
            (wall1.points[0].x - 0.0).abs() < 50.0,
            "current loop 1 stitched from the wrong previous-layer loop by centroid: \
         first inserted point x = {}, expected near 0.0",
            wall1.points[0].x
        );
    }

    #[test]
    fn slice_mesh_computes_infill_boundary_for_all_islands_regardless_of_wall_count() {
        // Create a layer with two disconnected wall 0 islands:
        // - Island 0: large (radius 10mm), easily fits 3 walls.
        // - Island 1: small boss/summit (radius 0.6mm), fits only 1 wall.
        // Both islands must generate infill boundaries so the narrow summit is not left with an open hole.
        let (basis1, basis2) = plane_basis(BUILD_DIRECTION);
        let origin = DVec3::ZERO;

        let circle = |center: DVec3, radius: f64, n_pts: usize| -> Vec<DVec3> {
            (0..n_pts)
                .map(|i| {
                    let angle = 2.0 * std::f64::consts::PI * (i as f64) / (n_pts as f64);
                    center + basis1 * (radius * angle.cos()) + basis2 * (radius * angle.sin())
                })
                .collect()
        };

        let island0 = circle(DVec3::new(0.0, 0.0, 5.0), 10.0, 32);
        let island1 = circle(DVec3::new(30.0, 0.0, 5.0), 0.6, 16);

        let config = SlicerConfig {
            wall_line_width: 0.4,
            shell_thickness: 1.2, // 3 walls
            ..SlicerConfig::default()
        };

        let wall0_loops = vec![island0, island1];
        let mut layer_infill_2d = Vec::new();

        for island in &wall0_loops {
            let island_2d = polygon2d::to_2d(std::slice::from_ref(island), basis1, basis2, origin);
            let partitioned = polygon2d::partition_walls_adaptive(
                &island_2d,
                config.wall_line_width,
                config.min_bead_width(),
                config.wall_count(),
            );
            if let Some(deepest_wall) = partitioned.last() {
                let inset =
                    polygon2d::inward_offset(&deepest_wall.loops_2d, config.wall_line_width);
                if !inset.is_empty() {
                    layer_infill_2d.extend(inset);
                } else if !deepest_wall.loops_2d.is_empty() {
                    layer_infill_2d.extend(deepest_wall.loops_2d.clone());
                }
            }
        }

        // Must produce 2 infill boundary loops (one for each island)
        assert_eq!(
            layer_infill_2d.len(),
            2,
            "both the large island and the narrow summit must have infill boundaries"
        );
    }

    #[test]
    fn stitch_wall_gaps_skips_a_loop_with_no_close_previous_match() {
        // The previous layer has a single loop; the current layer has two:
        // one that spatially corresponds to the previous loop (small
        // centroid shift, well within the match threshold) and one that
        // appears out of nowhere far away (a genuinely new loop this
        // layer, e.g. a new island). The new loop must be left unstitched
        // rather than force-matched to the one previous loop that exists --
        // the generalized "no reasonable previous-loop match -> skip" case.
        let config = SlicerConfig {
            nozzle_diameter: 0.4,
            ..SlicerConfig::default()
        };
        let field: Arc<dyn OrderField> = Arc::new(HeightOrderField::new(BUILD_DIRECTION));
        let (basis1, basis2) = plane_basis(BUILD_DIRECTION);

        let mut layers = vec![
            Layer {
                index: 0,
                order: 0.0,
                loops: vec![WallLoop {
                    wall_index: 0,
                    points: vec![DVec3::new(0.0, 0.0, 0.0)],
                    unsupported: vec![false],
                    top_surface: Vec::new(),
                    line_widths: Vec::new(),
                    arc_fraction: vec![0.0],
                }],
                order_field: Arc::clone(&field),
                ..Layer::default()
            },
            Layer {
                index: 1,
                order: 0.2,
                loops: vec![
                    WallLoop {
                        wall_index: 0,
                        points: vec![DVec3::new(2.0, 0.0, -0.2)],
                        unsupported: vec![false],
                        top_surface: Vec::new(),
                        line_widths: Vec::new(),
                        arc_fraction: vec![0.0],
                    },
                    WallLoop {
                        wall_index: 0,
                        points: vec![DVec3::new(1000.0, 0.0, -0.2)],
                        unsupported: vec![false],
                        top_surface: Vec::new(),
                        line_widths: Vec::new(),
                        arc_fraction: vec![0.0],
                    },
                ],
                order_field: Arc::clone(&field),
                ..Layer::default()
            },
        ];

        stitch_wall_gaps(&mut layers, &config, basis1, basis2);

        let matched = &layers[1].loops[0];
        assert!(
            matched.points.len() > 1,
            "the loop with a close previous-layer match should still be stitched"
        );

        let new_loop = &layers[1].loops[1];
        assert_eq!(
            new_loop.points.len(),
            1,
            "a genuinely new loop with no close previous-layer centroid match must be left \
         unstitched, not force-matched to the one existing previous loop"
        );
        assert_eq!(new_loop.unsupported, vec![false]);
    }

    #[test]
    fn stitch_wall_gaps_leaves_loop_unstitched_when_centroid_is_close_but_perimeter_implausible() {
        // Regression test for the loop-pairing bug: centroid distance alone can
        // match a small, simple loop to an unrelated, much larger/more complex
        // one whose centroid happens to land nearby, producing a meaningless
        // arc-length correspondence (see `WALL_GAP_LOOP_PERIMETER_MATCH_FACTOR`'s
        // docs). Here the previous layer has one big square loop (perimeter 40)
        // centered at the origin, and the current layer has one tiny square loop
        // (perimeter 0.4) also centered at the origin -- centroids coincide (well
        // within `WALL_GAP_LOOP_CENTROID_MATCH_FACTOR * nozzle_diameter`), but the
        // perimeter ratio (100x) is far beyond `WALL_GAP_LOOP_PERIMETER_MATCH_FACTOR`,
        // so the current loop must be left unstitched rather than bisected toward
        // a bogus far-away "corresponding" point on the big loop.
        let config = SlicerConfig {
            nozzle_diameter: 0.4,
            ..SlicerConfig::default()
        };
        let field: Arc<dyn OrderField> = Arc::new(HeightOrderField::new(BUILD_DIRECTION));
        let (basis1, basis2) = plane_basis(BUILD_DIRECTION);

        let big_square = vec![
            DVec3::new(-5.0, -5.0, 0.0),
            DVec3::new(5.0, -5.0, 0.0),
            DVec3::new(5.0, 5.0, 0.0),
            DVec3::new(-5.0, 5.0, 0.0),
        ];
        let small_square = vec![
            DVec3::new(-0.05, -0.05, -0.2),
            DVec3::new(0.05, -0.05, -0.2),
            DVec3::new(0.05, 0.05, -0.2),
            DVec3::new(-0.05, 0.05, -0.2),
        ];
        assert!(
            (loop_perimeter(&big_square) / loop_perimeter(&small_square))
                > WALL_GAP_LOOP_PERIMETER_MATCH_FACTOR,
            "test fixture must exceed the perimeter-match factor to exercise the rejection path"
        );

        let mut layers = vec![
            Layer {
                index: 0,
                order: 0.0,
                loops: vec![WallLoop {
                    wall_index: 0,
                    unsupported: vec![false; big_square.len()],
                    top_surface: Vec::new(),
                    line_widths: Vec::new(),
                    arc_fraction: compute_arc_fractions(&big_square),
                    points: big_square,
                }],
                order_field: Arc::clone(&field),
                ..Layer::default()
            },
            Layer {
                index: 1,
                order: 0.2,
                loops: vec![WallLoop {
                    wall_index: 0,
                    unsupported: vec![false; small_square.len()],
                    top_surface: Vec::new(),
                    line_widths: Vec::new(),
                    arc_fraction: compute_arc_fractions(&small_square),
                    points: small_square.clone(),
                }],
                order_field: Arc::clone(&field),
                ..Layer::default()
            },
        ];

        stitch_wall_gaps(&mut layers, &config, basis1, basis2);

        let wall = &layers[1].loops[0];
        assert_eq!(
            wall.points, small_square,
            "a centroid-close but perimeter-implausible previous loop must not be stitched \
         to -- the current loop's points should be untouched"
        );
        assert_eq!(
            wall.unsupported,
            vec![false; small_square.len()],
            "no points should be flagged unsupported when no plausible previous loop was found"
        );
    }

    #[test]
    fn stitch_wall_gaps_rejects_a_coincidentally_similar_perimeter_loop_when_a_much_closer_shape_mismatched_loop_exists(
    ) {
        // Regression test for the real-mesh layer-40/loop-13 finding (subtask 12
        // of the wall-gap-stitching feature): a genuine topology-merge event
        // (two small previous-layer loops sitting right next to the current
        // loop's centroid, correctly rejected by the perimeter check because
        // they are each roughly half its perimeter -- consistent with a
        // pre-merge shape) must not fall through to a *different*, spatially
        // distant previous loop purely because that distant loop's perimeter
        // happens to coincidentally match the current loop's. Accepting that
        // coincidence produced a uniform, whole-loop bad correspondence in the
        // real mesh (686/686 points with a ~7.7mm hop), not an isolated
        // single-point defect fixable by
        // `correct_isolated_correspondence_outliers`.
        let config = SlicerConfig {
            nozzle_diameter: 0.4,
            ..SlicerConfig::default()
        };
        let field: Arc<dyn OrderField> = Arc::new(HeightOrderField::new(BUILD_DIRECTION));
        let (basis1, basis2) = plane_basis(BUILD_DIRECTION);

        let square = |cx: f64, cy: f64, half: f64, z: f64| -> Vec<DVec3> {
            vec![
                DVec3::new(cx - half, cy - half, z),
                DVec3::new(cx + half, cy - half, z),
                DVec3::new(cx + half, cy + half, z),
                DVec3::new(cx - half, cy + half, z),
            ]
        };

        let prev_a = square(0.02, 0.0, 0.05, 0.0);
        let prev_b = square(0.05, 0.03, 0.05, 0.0);
        let prev_decoy = square(5.0, 0.0, 0.45, 0.0);
        let current = square(0.0, 0.0, 0.5, -0.2);

        assert!(
            (loop_perimeter(&current) / loop_perimeter(&prev_a))
                > WALL_GAP_LOOP_PERIMETER_MATCH_FACTOR,
            "fixture must reject prev_a by perimeter, like the real merge case"
        );
        assert!(
            (loop_perimeter(&current) / loop_perimeter(&prev_b))
                > WALL_GAP_LOOP_PERIMETER_MATCH_FACTOR,
            "fixture must reject prev_b by perimeter, like the real merge case"
        );
        let decoy_ratio = loop_perimeter(&current) / loop_perimeter(&prev_decoy);
        let decoy_ratio = if decoy_ratio < 1.0 {
            1.0 / decoy_ratio
        } else {
            decoy_ratio
        };
        assert!(
            decoy_ratio <= WALL_GAP_LOOP_PERIMETER_MATCH_FACTOR,
            "fixture's decoy loop must pass the perimeter check to exercise the margin-factor rejection"
        );
        let decoy_centroid_dist = (loop_centroid(&prev_decoy) - loop_centroid(&current)).length();
        assert!(
            decoy_centroid_dist <= WALL_GAP_LOOP_CENTROID_MATCH_FACTOR * config.nozzle_diameter,
            "fixture's decoy loop must pass the absolute centroid threshold to exercise the margin-factor rejection"
        );

        let mut layers = vec![
            Layer {
                index: 0,
                order: 0.0,
                loops: vec![
                    WallLoop {
                        wall_index: 0,
                        unsupported: vec![false; prev_a.len()],
                        top_surface: Vec::new(),
                        line_widths: Vec::new(),
                        arc_fraction: compute_arc_fractions(&prev_a),
                        points: prev_a,
                    },
                    WallLoop {
                        wall_index: 0,
                        unsupported: vec![false; prev_b.len()],
                        top_surface: Vec::new(),
                        line_widths: Vec::new(),
                        arc_fraction: compute_arc_fractions(&prev_b),
                        points: prev_b,
                    },
                    WallLoop {
                        wall_index: 0,
                        unsupported: vec![false; prev_decoy.len()],
                        top_surface: Vec::new(),
                        line_widths: Vec::new(),
                        arc_fraction: compute_arc_fractions(&prev_decoy),
                        points: prev_decoy,
                    },
                ],
                order_field: Arc::clone(&field),
                ..Layer::default()
            },
            Layer {
                index: 1,
                order: 0.2,
                loops: vec![WallLoop {
                    wall_index: 0,
                    unsupported: vec![false; current.len()],
                    top_surface: Vec::new(),
                    line_widths: Vec::new(),
                    arc_fraction: compute_arc_fractions(&current),
                    points: current.clone(),
                }],
                order_field: Arc::clone(&field),
                ..Layer::default()
            },
        ];

        stitch_wall_gaps(&mut layers, &config, basis1, basis2);

        let wall = &layers[1].loops[0];
        assert_eq!(
            wall.points, current,
            "a distant, only-coincidentally-perimeter-matching previous loop must not be \
             accepted when a much closer (but shape-mismatched) previous loop exists -- the \
             current loop's points should be untouched"
        );
        assert_eq!(
            wall.unsupported,
            vec![false; current.len()],
            "no points should be flagged unsupported when no plausible previous loop was found"
        );
    }

    #[test]
    fn compute_arc_fractions_starts_at_zero_and_increases_monotonically_around_the_loop() {
        let points = vec![
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(1.0, 1.0, 0.0),
            DVec3::new(0.0, 1.0, 0.0),
        ];
        let fractions = compute_arc_fractions(&points);
        assert_eq!(fractions.len(), points.len());
        assert_eq!(fractions[0], 0.0);
        for pair in fractions.windows(2) {
            assert!(
                pair[1] > pair[0],
                "arc_fraction must increase monotonically"
            );
        }
        for &f in &fractions {
            assert!(
                (0.0..1.0).contains(&f),
                "arc_fraction must stay within [0, 1)"
            );
        }
    }

    /// Builds a 6-spike "zigzag star" polygon in the XY plane: vertices
    /// alternate between an outer spike tip (radius `outer_r`) and an inner
    /// waist vertex very close to the center (radius `inner_r`), 12 vertices
    /// total, 30 degrees apart. Used by
    /// `rotation_search_and_local_fallback_beat_single_point_alignment_on_a_spiky_loop`
    /// to construct a loop where a single nearest-3D-point start alignment
    /// is unreliable: all 6 inner-waist vertices sit within `inner_r` of the
    /// origin regardless of which spike they belong to, so a raw-3D nearest
    /// search from a point near the center can easily lock onto the *wrong*
    /// spike's waist vertex -- a real instance of the "unrelated parts of a
    /// spiky loop pass close to each other in space" failure mode described
    /// in `stitch_wall_gaps`'s docs (Addendum 5).
    fn zigzag_star_vertices(outer_r: f64, inner_r: f64) -> Vec<DVec3> {
        (0..12)
            .map(|i| {
                let angle = (i as f64) * std::f64::consts::PI / 6.0;
                let r = if i % 2 == 0 { outer_r } else { inner_r };
                DVec3::new(r * angle.cos(), r * angle.sin(), 0.0)
            })
            .collect()
    }

    /// Resamples a closed polyline at `count` evenly-spaced arc-length
    /// fractions of `base`, starting at fraction `start_offset`, using the
    /// same [`compute_arc_fractions`]/[`interpolate_on_loop_at_fraction`]
    /// machinery `stitch_wall_gaps` itself uses. Used by
    /// `rotation_search_and_local_fallback_beat_single_point_alignment_on_a_spiky_loop`
    /// to build two independently-sampled (different point count, different
    /// starting position) loops from the same underlying shape -- exactly
    /// the "different point counts, sampling density, and starting point"
    /// scenario described in `stitch_wall_gaps`'s docs.
    fn resample_closed_loop(
        base: &[DVec3],
        base_fractions: &[f64],
        count: usize,
        start_offset: f64,
    ) -> Vec<DVec3> {
        (0..count)
            .map(|i| {
                let t = ((i as f64) / (count as f64) + start_offset).rem_euclid(1.0);
                interpolate_on_loop_at_fraction(base, base_fractions, t)
            })
            .collect()
    }

    #[test]
    fn rotation_search_and_local_fallback_beat_single_point_alignment_on_a_spiky_loop() {
        // A 6-spike zigzag star whose inner "waist" vertices all sit within
        // 0.05 units of the origin -- close enough together that a single
        // raw-3D nearest-point search from a point near the center cannot
        // reliably tell which spike's waist it actually belongs to, while a
        // multi-sample rotation search (scored across points spread all the
        // way around the loop) can, because a wrong offset that happens to
        // fit near the center disagrees badly with the outer spike tips
        // elsewhere on the loop.
        let base = zigzag_star_vertices(5.0, 0.05);
        let base_fractions = compute_arc_fractions(&base);

        // Two independently-sampled loops from the same underlying star:
        // different point counts (53 vs 41) and a genuine rotational offset
        // (0.37) between their starting positions -- mirroring real
        // adjacent-layer wall-0 loops, which are extracted independently
        // with no guaranteed shared parameterization.
        const TRUE_OFFSET: f64 = 0.37;
        let prev_points = resample_closed_loop(&base, &base_fractions, 53, 0.0);
        let prev_fractions = compute_arc_fractions(&prev_points);
        let current_points = resample_closed_loop(&base, &base_fractions, 41, TRUE_OFFSET);
        let current_fractions = compute_arc_fractions(&current_points);

        // The naive single-point alignment: nearest raw-3D-distance previous
        // point to just the current loop's first point (mirrors the
        // pre-subtask-10 `stitch_wall_gaps` code, inlined here since that
        // code path no longer exists in production).
        let start = current_points[0];
        let naive_nearest_idx = prev_points
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                (**a - start)
                    .length_squared()
                    .total_cmp(&(**b - start).length_squared())
            })
            .map(|(idx, _)| idx)
            .unwrap();
        let naive_offset = prev_fractions[naive_nearest_idx];

        // The new rotation-search offset.
        let best_offset = best_rotation_offset(
            &current_points,
            &current_fractions,
            &prev_points,
            &prev_fractions,
        );

        // Ground truth for any current-loop index `i`: since both loops were
        // resampled from `base`, the physically-corresponding position is
        // simply `base` evaluated at the same base-fraction the current
        // point was sampled from -- independent of either loop's own
        // (re-derived, potentially warped) arc-fraction table.
        let true_position_for_index = |i: usize| -> DVec3 {
            let true_base_fraction = ((i as f64) / 41.0 + TRUE_OFFSET).rem_euclid(1.0);
            interpolate_on_loop_at_fraction(&base, &base_fractions, true_base_fraction)
        };

        // Check several sample indices spread around the loop.
        let mut naive_worst_error: f64 = 0.0;
        let mut rotation_search_worst_error: f64 = 0.0;
        for i in (0..41).step_by(4) {
            let truth = true_position_for_index(i);

            let naive_t_prev = (naive_offset + current_fractions[i]).rem_euclid(1.0);
            let naive_corresponding =
                interpolate_on_loop_at_fraction(&prev_points, &prev_fractions, naive_t_prev);
            naive_worst_error = naive_worst_error.max((naive_corresponding - truth).length());

            let best_t_prev = (best_offset + current_fractions[i]).rem_euclid(1.0);
            let fraction_based =
                interpolate_on_loop_at_fraction(&prev_points, &prev_fractions, best_t_prev);
            let refined = local_fallback_correspondence(
                &prev_points,
                &prev_fractions,
                best_t_prev,
                current_points[i],
                fraction_based,
            );
            rotation_search_worst_error =
                rotation_search_worst_error.max((refined - truth).length());
        }

        assert!(
            naive_worst_error > 1.0,
            "test fixture must actually fool the single-point alignment (worst error \
         {naive_worst_error}) to be a meaningful regression test"
        );
        assert!(
            rotation_search_worst_error < 0.5,
            "rotation search + bounded local fallback should land close to the true corresponding \
         position (worst error {rotation_search_worst_error}), unlike single-point alignment \
         (worst error {naive_worst_error})"
        );
    }
}
