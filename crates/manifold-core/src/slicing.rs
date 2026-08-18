//! Non-planar slicing: mesh -> ordered layers of cross-section curves.

use crate::{
    ids::ObjectId, mesh::Mesh, object::Object, order_field, polygon2d, Error, Result, SlicerConfig,
};
use glam::DVec3;
use manifold_fidget::contour::{extract_contours, extract_order_contours_on_mesh, plane_basis};
use manifold_fidget::marching_cubes::extract_isosurface;
use manifold_fidget::mesh_sdf::MeshSdf;
use manifold_fidget::order::{order_range_over_bbox, HeightOrderField, OrderField};
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
}

/// Build/order direction for this MVP: conventional planar slicing along
/// -Z (i.e. `order(p) = p.dot(direction)` decreases going up, matching a
/// bottom-to-top print). Hardcoded per this task's scope — see
/// `NON_PLANAR_SLICING.md` for the follow-up angle-driven order field that
/// will make this configurable.
pub(crate) const BUILD_DIRECTION: DVec3 = DVec3::new(0.0, 0.0, -1.0);

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
    slice_mesh_with_progress(mesh, config, &mut |_| {})
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
    let sdf = Arc::new(MeshSdf::new(mesh.vertices.clone(), faces));

    // Resolve the configured order field once per slice (defaults to a
    // `HeightOrderField` along `BUILD_DIRECTION`, matching pre-existing
    // behavior exactly). `order_range_over_bbox` generically replaces the
    // old `min.dot(BUILD_DIRECTION)`/`max.dot(BUILD_DIRECTION)` shortcut —
    // exact for any field whose extrema are attained at box corners, which
    // covers the affine `HeightOrderField` default used everywhere today.
    let field: Arc<dyn OrderField> = Arc::from(order_field::order_field_for(
        config.order_field,
        config,
        mesh,
    ));
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
        let (lo, hi) = order_range_over_bbox(&*field, min, max);
        if lo.is_finite() && hi.is_finite() {
            (lo, hi)
        } else {
            let (mut vlo, mut vhi) = (f64::INFINITY, f64::NEG_INFINITY);
            for &v in &mesh.vertices {
                let value = field.order(v);
                if value.is_finite() {
                    vlo = vlo.min(value);
                    vhi = vhi.max(value);
                }
            }
            if vlo.is_finite() && vhi.is_finite() {
                (vlo, vhi)
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
    let resolution = contour_resolution(extent, config.nozzle_diameter, CONTOUR_REFINEMENT_DIVISOR);

    // Precompute every order-field value this walk will sample, so the
    // per-layer contour extraction below can run in parallel (each layer
    // only reads the shared, immutable `sdf` — see `MeshSdf`, whose query
    // methods all take `&self` — and writes its own independent `Layer`;
    // there's no cross-layer dependency to serialize on).
    //
    // Defensive cap: even with the finite `order_min`/`order_max` guaranteed
    // above, guard the step count against any other future degenerate range
    // (e.g. a bug producing `order_min > order_max` combined with a tiny
    // `layer_height`) so a bad range fails fast with `Error::Slicing`
    // instead of growing this `Vec` without bound until the process is
    // killed for memory exhaustion.
    const MAX_ORDER_STEPS: usize = 1_000_000;
    let mut order_values = Vec::new();
    let mut order_value = order_min;
    while order_value <= order_max {
        if order_values.len() >= MAX_ORDER_STEPS {
            return Err(Error::Slicing(format!(
                "order range [{order_min}, {order_max}] with layer_height {layer_height} would \
                 require more than {MAX_ORDER_STEPS} layers; refusing to continue"
            )));
        }
        order_values.push(order_value);
        order_value += layer_height;
    }

    let total_steps = order_values.len().max(1);

    let wall_count = config.wall_count();

    // Dispatch on the resolved field kind (cheap enum comparison on
    // `config.order_field`, not any runtime type-inspection of the `dyn
    // OrderField` trait object): `Height` keeps today's exact plane-sampling
    // fast path unchanged; anything else (e.g. `Conical`) uses the
    // generalized "contour-on-mesh" path, which extracts each wall pass's
    // isosurface once (not once per layer) and walks it per layer.
    let is_height = matches!(config.order_field, order_field::OrderFieldKind::Height);

    // Total units of reportable work: for `Height` this is just the
    // per-layer loop (unchanged from before). For a curved field, the
    // `curved_wall_meshes` precompute below is itself an expensive
    // whole-mesh isosurface extraction per wall pass — done once, before
    // the per-layer loop, previously with *no* progress reporting at all,
    // so `on_progress` sat frozen at its last value for however long that
    // took (seconds, on a real mesh) even though the extraction itself is
    // internally parallel. Counting each wall pass as one more reportable
    // unit alongside the `total_steps` layers keeps the bar moving through
    // that phase instead of appearing stalled.
    let total_units = if is_height {
        total_steps
    } else {
        wall_count + total_steps
    };
    let completed = AtomicUsize::new(0);
    // `on_progress` is `&mut dyn FnMut`, not `Sync`, so serialize calls into
    // it behind a `Mutex` — contention is negligible since each holder only
    // reports one f64 and releases immediately, and the expensive work
    // (`extract_contours`/`extract_order_contours_on_mesh`/
    // `extract_isosurface`) happens before the lock is taken.
    let progress_callback = Mutex::new(on_progress);

    // Curved path only: one triangle soup per wall pass, shared across every
    // layer's `extract_order_contours_on_mesh` call below. Computed up front
    // (outside the per-layer parallel loop) since it does not depend on the
    // layer's `order_value` — only on the wall pass's `iso`. Each wall's
    // isosurface extraction reports one `total_units` step as it finishes,
    // so `on_progress` keeps advancing through this whole-mesh precompute
    // instead of only starting once the per-layer loop below begins.
    let curved_wall_meshes: Vec<Vec<DVec3>> = if is_height {
        Vec::new()
    } else {
        let full_diagonal = (max - min).length();
        let iso_resolution = contour_resolution(
            full_diagonal,
            config.nozzle_diameter,
            CONTOUR_REFINEMENT_DIVISOR,
        );
        (0..wall_count)
            .into_par_iter()
            .map(|wall_index| {
                let iso = -(config.wall_offset + wall_index as f64 * config.wall_line_width);
                let vertices = extract_isosurface::<MeshSdf>(&*sdf, min, max, iso_resolution, iso);
                // `extract_order_contours_on_mesh` walks a flat position soup
                // (see its doc comment), decoupled from `marching_cubes`'s
                // `Vertex` (position + normal) — no `OrderField` dependency
                // is added to `marching_cubes` itself.
                let positions: Vec<DVec3> = vertices.into_iter().map(|v| v.position).collect();
                let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
                if let Ok(mut callback) = progress_callback.lock() {
                    callback(done as f64 / total_units as f64);
                }
                positions
            })
            .collect()
    };

    let layers: Vec<Layer> = order_values
        .par_iter()
        .enumerate()
        .map(|(index, &order_value)| {
            let origin =
                bbox_center + BUILD_DIRECTION * (order_value - bbox_center.dot(BUILD_DIRECTION));
            let mut loops = Vec::new();
            if is_height {
                for wall_index in 0..wall_count {
                    // Negative iso = inward (see `MeshSdf::sign_at`: positive
                    // outside, negative inside). Wall 0 sits `wall_offset` in
                    // from the true surface (nozzle-center offset so the
                    // nozzle's outer edge lands on the surface); each further
                    // wall steps another `wall_line_width` inward.
                    let iso = -(config.wall_offset + wall_index as f64 * config.wall_line_width);
                    let wall_loops = extract_contours(
                        &*sdf, origin, basis1, basis2, extent, extent, resolution, resolution, iso,
                    );
                    loops.extend(
                        wall_loops
                            .into_iter()
                            .map(|points| WallLoop { wall_index, points }),
                    );
                }
            } else {
                for (wall_index, triangle_positions) in curved_wall_meshes.iter().enumerate() {
                    let wall_loops =
                        extract_order_contours_on_mesh(triangle_positions, &*field, order_value);
                    loops.extend(
                        wall_loops
                            .into_iter()
                            .map(|points| WallLoop { wall_index, points }),
                    );
                }
            }
            let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
            if let Ok(mut callback) = progress_callback.lock() {
                callback(done as f64 / total_units as f64);
            }
            // The infill boundary sits where wall pass `wall_count` would
            // be if one more wall were printed — one further
            // `wall_line_width` step inward from the innermost printed
            // wall. Kept out of `loops` so `toolpath::plan` never treats it
            // as a printable wall.
            //
            // The innermost *configured* pass (`wall_count - 1`) may have no
            // geometry at all: a cross-section can be too small to fit every
            // nested wall the configured `shell_thickness` calls for (e.g.
            // near a tapered tip), while an outer pass still does. Using a
            // hardcoded `wall_count - 1` in that case would silently zero
            // out `infill_boundary` (and therefore `solid_fill_boundary`)
            // even though the outer wall(s) that did extract still bound a
            // real interior area. So walk backward from `wall_count - 1` to
            // find the deepest wall pass that actually produced loops, and
            // offset inward from there instead — the offset step still
            // conceptually represents "one more `wall_line_width` inward
            // from whatever the innermost printed wall turned out to be".
            //
            // Computed as a 2D inward offset of that wall loop (rather than
            // a further 3D SDF isosurface probe at `boundary_iso`): the SDF
            // probe could come back empty even when the innermost wall
            // itself is non-empty (e.g. thin features, or contour-extraction
            // grid resolution missing the slightly-smaller isosurface),
            // silently dropping infill on otherwise fillable layers.
            // Offsetting the already-extracted wall loop in 2D instead
            // guarantees a boundary whenever any wall exists.
            //
            // Curved path (non-Height order field): uses the same shared
            // `plane_basis(BUILD_DIRECTION)` basis as the Height path below
            // (rather than a separate per-loop best-fit local basis) so the
            // inward-offset step is consistent across all order field
            // kinds. Each layer's loops are only *locally* near-planar
            // (concentric around a common apex/axis — see
            // `manifold_fidget::contour`'s module docs), so this is a
            // best-effort approximation, not a single unified
            // curved-boundary computation: exact correctness for
            // arbitrarily curved solid-fill boundaries is an explicitly
            // soft requirement for this phase, not a hard blocker — see
            // ROADMAP.md Phase 15's "explicitly out of scope" section (no
            // toolpath-level physical-printability guarantees this phase).
            let innermost_wall_loops: Vec<Vec<DVec3>> = (0..wall_count)
                .rev()
                .map(|wall_index| {
                    loops
                        .iter()
                        .filter(|wall| wall.wall_index == wall_index)
                        .map(|wall| wall.points.clone())
                        .collect::<Vec<_>>()
                })
                .find(|wall_loops| !wall_loops.is_empty())
                .unwrap_or_default();
            let infill_boundary = if innermost_wall_loops.is_empty() {
                Vec::new()
            } else if is_height {
                let loops_2d = polygon2d::to_2d(&innermost_wall_loops, basis1, basis2, origin);
                let offset_2d = polygon2d::inward_offset(&loops_2d, config.wall_line_width);
                polygon2d::from_2d(offset_2d, basis1, basis2, origin)
            } else {
                // Uses the same global build-direction basis
                // (`basis1`/`basis2` from `plane_basis(BUILD_DIRECTION)`) as
                // the Height path above, applied uniformly across all
                // innermost wall loops rather than a separate per-loop
                // best-fit local basis — even though each loop is only
                // *locally* near-planar (concentric around a common
                // apex/axis, see `manifold_fidget::contour`'s module docs),
                // a per-loop local basis let small/noisy loops yield a bad
                // plane normal and send offset points well off the actual
                // isosurface. Any 2D-projection distortion the shared
                // global basis introduces on steeply-curved loops is
                // corrected immediately below by reprojecting each offset
                // point back onto the isosurface.
                let loops_2d = polygon2d::to_2d(&innermost_wall_loops, basis1, basis2, origin);
                let offset_2d_flat = polygon2d::inward_offset(&loops_2d, config.wall_line_width);
                let offset_3d = polygon2d::from_2d(offset_2d_flat, basis1, basis2, origin);
                // The shared-basis 2D offset above is only an
                // approximation for curved layers: projecting non-planar
                // loops into one flat basis before offsetting can send
                // offset points somewhat off the layer's actual isosurface
                // (even below the build plate), and those points then
                // poison every downstream consumer that uses
                // `infill_boundary` as a same-branch height reference
                // (`InfillRegion::from_layer`, `compute_solid_fill_boundaries`,
                // infill crossing re-solves). Refine each offset point back
                // onto the isosurface, seeded from the nearest
                // innermost-wall point -- known-good geometry straight from
                // contour extraction on the mesh (see
                // `reconstruct_on_order_field_near`).
                let offset_2d = polygon2d::to_2d(&offset_3d, basis1, basis2, origin);
                order_field::reconstruct_on_order_field_near(
                    offset_2d,
                    &innermost_wall_loops,
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

    Ok(layers)
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

        // Index increases going *down* (see `BUILD_DIRECTION`: index 0 is
        // the top of the object, index `n - 1` the bottom), so the layer
        // physically above `k` is `k - 1` and the layer physically below is
        // `k + 1`.
        let exposed_above: Vec<Vec<Vec<[f64; 2]>>> = (0..n)
            .into_par_iter()
            .map(|k| {
                if k == 0 {
                    boundaries_2d[k].clone()
                } else {
                    polygon2d::difference(&boundaries_2d[k], &boundaries_2d[k - 1])
                }
            })
            .collect();
        let exposed_below: Vec<Vec<Vec<[f64; 2]>>> = (0..n)
            .into_par_iter()
            .map(|k| {
                let next = boundaries_2d.get(k + 1).unwrap_or(&empty_2d);
                if next.is_empty() {
                    boundaries_2d[k].clone()
                } else {
                    polygon2d::difference(&boundaries_2d[k], next)
                }
            })
            .collect();

        let solid_2d_per_k: Vec<Vec<Vec<[f64; 2]>>> = (0..n)
            .into_par_iter()
            .map(|k| {
                let mut regions: Vec<Vec<Vec<[f64; 2]>>> = Vec::new();
                if config.top_layers > 0 {
                    // A top-facing surface at layer `j` makes `top_layers` worth
                    // of layers *below* it (physically below = larger index,
                    // i.e. `j..=j+top_layers-1`) solid. So layer `k` picks up
                    // contributions from `exposed_above(j)` for `j` in
                    // `k-top_layers+1..=k` (backward from `k`, since index
                    // increases going down).
                    let start = k.saturating_sub(config.top_layers - 1);
                    for exposed in exposed_above.iter().take(k + 1).skip(start) {
                        regions.push(exposed.clone());
                    }
                }
                if config.bottom_layers > 0 {
                    // Symmetric: a bottom-facing surface at `j` makes
                    // `bottom_layers` worth of layers *above* it (physically
                    // above = smaller index, i.e. `j-bottom_layers+1..=j`)
                    // solid, so layer `k` picks up `exposed_below(j)` for `j` in
                    // `k..=k+bottom_layers-1` (forward from `k`).
                    let end = (k + config.bottom_layers - 1).min(n - 1);
                    for exposed in exposed_below.iter().take(end + 1).skip(k) {
                        regions.push(exposed.clone());
                    }
                }
                let exposed_union = polygon2d::union(&regions);
                polygon2d::intersection(&exposed_union, &boundaries_2d[k])
            })
            .collect();

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
            let references = layers[positions[k]].infill_boundary.clone();
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
    slice_object_with_progress(object, config, &mut |_| {})
}

/// Same as [`slice_object`], forwarding to [`slice_mesh_with_progress`] and
/// then running [`compute_solid_fill_boundaries`] as a post-pass over this
/// object's own layer stack (never mixed with any other object's layers).
pub fn slice_object_with_progress(
    object: &Object,
    config: &SlicerConfig,
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

    let mut layers = slice_mesh_with_progress(&world_mesh, config, on_progress)?;
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
    slice_workspace_with_progress(objects, order, config, &mut |_| {})
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
        layers.extend(slice_object_with_progress(object, config, &mut |local| {
            on_progress(((object_index as f64) + local) / total);
        })?);
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
        slice_mesh_with_progress(&cube_mesh(), &config, &mut |fraction| {
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
        slice_workspace_with_progress(&objects, &order, &config, &mut |fraction| {
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
            solid_flags[first_real],
            Some(true),
            "the layer nearest the top-facing surface must be fully solid"
        );
        assert_eq!(
            solid_flags[first_real + 1],
            Some(false),
            "top_layers = 1 must not make the second layer down from the top fully solid"
        );

        // bottom_layers = 2: the two layers right above the bottom cap are solid.
        assert_eq!(
            solid_flags[last_real],
            Some(true),
            "the layer nearest the bottom-facing surface must be fully solid"
        );
        assert_eq!(
            solid_flags[last_real - 1],
            Some(true),
            "bottom_layers = 2 must make the second layer up from the bottom fully solid"
        );
        assert_eq!(
            solid_flags[last_real - 2],
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

        // The cube spans Z in [0, 1] with layer_height 0.25: expect 5
        // stepped layers (0.0, 0.25, 0.5, 0.75, 1.0). The interior layers
        // (0.25, 0.5, 0.75) are clean square cross-sections; the exact
        // boundary layers (Z=0, Z=1) sample directly on the mesh surface,
        // where the sign/crossing is numerically ambiguous, so only the
        // interior layers are asserted to have a contour loop.
        assert_eq!(layers.len(), 5);
        for layer in &layers[1..4] {
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

        assert_eq!(layers.len(), 5);
        for layer in &layers[1..4] {
            assert_eq!(layer.loops.len(), 1, "expected exactly one contour loop");
            assert!(!layer.loops[0].points.is_empty());
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
            5,
            "expected 5 stepped layers over [0, 1] at layer_height 0.25"
        );
        let mut expected_order = expected_order_min;
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

        for layer in &layers[1..4] {
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

        for layer in &layers[1..4] {
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
}
