//! Pluggable slicing order fields.
//!
//! Slicing walks isosurfaces of a scalar "order field" over 3D space (see
//! `manifold_fidget::order::OrderField`) in increasing order to produce
//! layers. Historically this was hardcoded to a flat height field along
//! `slicing::BUILD_DIRECTION`. This module makes the choice **pluggable**,
//! mirroring `ordering.rs`'s `ObjectOrderingKind`/`strategy_for` pattern:
//! `SlicerConfig::order_field` selects an [`OrderFieldKind`], resolved to a
//! concrete `manifold_fidget::order::OrderField` via [`order_field_for`].
//! Adding a new field later means adding an enum variant, a
//! `manifold-fidget` type implementing the trait, and one match arm here.

use glam::DVec3;
use manifold_fidget::eikonal::EikonalOrderField;
use manifold_fidget::mesh_sdf::MeshSdf;
use manifold_fidget::order::{ConicalOrderField, HeightOrderField, OrderField};

// Re-exported so downstream diagnostics (e.g. the manifold-cli
// verification examples) can name the trait when calling helpers like
// `slicing::lateral_gap` without depending on manifold-fidget directly.
pub use manifold_fidget::order::OrderField as OrderFieldTrait;
use manifold_fidget::ScalarField;

use manifold_fidget::height_along::ConstantAxisHeight;

use crate::{mesh::Mesh, slicing::BUILD_DIRECTION, SlicerConfig};

/// Selects which [`OrderField`] `slice_mesh`/`slice_mesh_with_progress` use.
/// Persisted on [`crate::SlicerConfig`] like any other slicing parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum OrderFieldKind {
    /// Flat planar slicing along `slicing::BUILD_DIRECTION` — today's
    /// exact behavior, kept as the default (zero-risk, no breaking change
    /// to existing profiles/configs). See
    /// `manifold_fidget::order::HeightOrderField`.
    #[default]
    Height,
    /// Curved, cone-shaped slicing around a configurable apex/axis/slope.
    /// See `manifold_fidget::order::ConicalOrderField`.
    Conical,
    /// Front-propagation slicing: a grid-based Fast Marching Method (FMM)
    /// solve (see `manifold_fidget::eikonal::EikonalOrderField`) seeded from
    /// the mesh's actual base/contact surface with the build plate, marching
    /// outward with uniform speed. Unlike `Height`/`Conical` (pure
    /// closed-form functions of `config` alone), this field's values depend
    /// on the mesh's actual geometry, so it can only be resolved where a
    /// mesh is in scope (see [`order_field_for`]) -- downstream passes with
    /// no mesh in scope reuse the field cached on `slicing::Layer` instead
    /// of re-resolving this variant.
    Eikonal,
}

/// Resolve a config-level [`OrderFieldKind`] to a concrete
/// `manifold_fidget::order::OrderField`, using `config`'s `apex`/`axis`/
/// `slope` fields for the [`OrderFieldKind::Conical`] case, and `mesh` for
/// the [`OrderFieldKind::Eikonal`] case (its front is seeded from `mesh`'s
/// actual base/contact surface with the build plate -- see
/// [`eikonal_field_for`]).
///
/// `mesh` is only actually used for `Eikonal`; `Height`/`Conical` ignore it.
/// This is only called from `slicing::slice_mesh_with_progress`, the one
/// call site with mesh access -- downstream passes
/// (`slicing::compute_solid_fill_boundaries`, `infill::InfillRegion::from_layer`)
/// have no mesh in scope and instead reuse the field cached on
/// `slicing::Layer` (see [`Layer::order_field`](crate::slicing::Layer::order_field)).
pub fn order_field_for(
    kind: OrderFieldKind,
    config: &SlicerConfig,
    mesh: &Mesh,
    slope_profile: &manifold_fidget::slope_profile::SlopeProfile,
) -> Box<dyn OrderField> {
    order_field_for_with_sdf(kind, config, mesh, slope_profile, None)
}

/// Same as [`order_field_for`], but allows passing an already-constructed
/// [`MeshSdf`] to avoid redundant BVH and spatial-partitioning construction.
pub fn order_field_for_with_sdf(
    kind: OrderFieldKind,
    config: &SlicerConfig,
    mesh: &Mesh,
    slope_profile: &manifold_fidget::slope_profile::SlopeProfile,
    sdf: Option<&MeshSdf>,
) -> Box<dyn OrderField> {
    match kind {
        OrderFieldKind::Height => Box::new(HeightOrderField::new(BUILD_DIRECTION)),
        OrderFieldKind::Conical => Box::new(ConicalOrderField::new(
            config.order_field_apex,
            config.order_field_axis,
            config.order_field_slope,
        )),
        OrderFieldKind::Eikonal => Box::new(eikonal_field_for(config, mesh, slope_profile, sdf)),
    }
}

/// Builds the [`OrderFieldKind::Eikonal`] field for `mesh`: seeds the FMM
/// front from `mesh`'s base/contact region with the build plate (every
/// solid-classified grid node at or within a small tolerance of the mesh's
/// minimum Z -- the "rests on floor" convention already used by
/// `object::center_on_bed` and `manifold-gui`'s `scene::build_bed_quad`,
/// both of which treat `min.z` as the build plate's height), and derives
/// grid resolution from `mesh`'s bounding box and the finer of
/// `config.layer_height`/`config.nozzle_diameter` (reusing those existing
/// resolution-driving config fields rather than adding a new one). Using
/// `layer_height` alone (0.2mm by default) produced a grid too coarse to
/// resolve fine surface detail (e.g. helical/screw grooves) -- trilinear
/// interpolation of an under-resolved distance field visibly facets the
/// reconstructed contours -- so the cell size is halved again on top of the
/// finer of the two dimensions for extra headroom. That resolution-driven
/// cell size is then passed through [`clamp_cell_size_to_node_budget`],
/// which coarsens it (grows it) as needed so the dense grid's total node
/// count never exceeds [`MAX_EIKONAL_GRID_NODES`] -- without this, a large
/// part sliced at a fine layer height/nozzle diameter (e.g. an 85mm-tall
/// part at the 0.2mm default layer height, cell size 0.05mm) requests a
/// `~1700^3` node grid, whose `Vec<f64>` distance buffer alone is tens of
/// gigabytes and OOMs the process; the budget trades local resolution for
/// a bounded, predictable memory footprint on large parts.
///
/// Seeding is by *region*
/// ([`manifold_fidget::eikonal::EikonalOrderField::new_with_occupancy_and_seed_region`]),
/// not by mesh vertex position: every solid grid node within a small
/// tolerance (half the grid's own cell size) of the mesh's minimum Z is
/// frozen at distance `0.0` directly,
/// rather than only the mesh's own vertices near that height. Seeding from
/// vertices alone only seeds wherever those vertices happen to sit -- a
/// flat base face triangulated with vertices solely along its silhouette
/// (a quad base commonly has only 4 corner vertices) would seed just the
/// footprint's *boundary* outline, leaving its interior to march in from
/// that boundary and report a small but nonzero order there instead of the
/// `0.0` a point already resting flat on the plate deserves -- the base
/// layer would then read as an eroded/rounded version of the true
/// footprint. Region seeding fills the interior of each connected
/// component of the contact footprint uniformly, at the grid's own
/// resolution, independent of the mesh's triangulation density.
///
/// The march is occupancy-aware: a grid node counts as solid when a fresh
/// `MeshSdf` built from `mesh` reports it inside the mesh, or within
/// `cell_size` of the surface (a tolerance covering the discretization gap
/// between the grid and the true boundary -- without it, nodes exactly on
/// a thin wall's surface could be spuriously excluded, breaking the march
/// right where it matters most). This keeps the front from taking a
/// straight-line shortcut through open air (e.g. across a gap between two
/// close walls, or up the outside of a wall toward a nearby build-plate
/// seed) -- without it, `order` reduces to plain Euclidean distance and
/// can produce isosurfaces that climb outside the object along a path no
/// real toolpath could follow.
///
/// An empty mesh (no vertices, so no bounding box) is a documented
/// best-effort fallback -- a degenerate unit-box field with no seeds, whose
/// `order()` always returns `f64::INFINITY` (see
/// [`manifold_fidget::eikonal::EikonalOrderField`]'s own documented
/// behavior for an empty seed set) -- rather than a panic. A mesh with no
/// triangles (nothing to classify solid/void against) falls back to
/// treating every grid node as solid, so the region seeding and march are
/// unconstrained (equivalent to plain Euclidean distance from the contact
/// band) rather than leaving the whole grid unreachable.
///
/// After the FMM march, `config.eikonal_slope_profile` is converted to a
/// `manifold_fidget::slope_profile::SlopeProfile` and applied via a
/// relaxation pass (see
/// `EikonalOrderField::new_with_occupancy_and_seed_region_and_slope_limit`),
/// measuring height with a `ConstantAxisHeight` along the same
/// `BUILD_DIRECTION`/`min`-anchored convention `is_seed_region` already
/// uses. An empty/default slope profile is documented as unconstrained, so
/// this is a no-op when no profile is configured.
fn eikonal_field_for(
    config: &SlicerConfig,
    mesh: &Mesh,
    slope_profile: &manifold_fidget::slope_profile::SlopeProfile,
    existing_sdf: Option<&MeshSdf>,
) -> EikonalOrderField {
    let Some((min, max)) = mesh.bounding_box() else {
        return EikonalOrderField::new(DVec3::ZERO, DVec3::ONE, &[], 1.0);
    };

    let layer_height = config.layer_height.abs().max(f64::EPSILON);
    let nozzle_diameter = config.nozzle_diameter.abs().max(f64::EPSILON);
    // `/ 4.0` rather than the originally-tried `/ 2.0`: `EikonalOrderField`
    // interpolates its FMM distance grid trilinearly (C0 only -- see
    // `EikonalOrderField::order`'s doc comment), so every isosurface
    // projection/reconstruction that walks this field (wall extraction,
    // `reconstruct_on_order_field_near`, wall-gap stitching, ...) lands on
    // a faceted surface with facet width == this cell size. Halving it
    // again roughly halves that faceting amplitude (measured on
    // pug_v4_l_sop_85mm.stl with the manual `probe_wall_noise` example --
    // see project memory). `clamp_cell_size_to_node_budget` below still
    // coarsens this back up for large parts rather than blowing the node
    // budget, so this is a quality/perf trade only for parts small enough
    // to afford the finer grid.
    let requested_cell_size = layer_height.min(nozzle_diameter) / 4.0;
    let cell_size = clamp_cell_size_to_node_budget(max - min, requested_cell_size);
    // Contact-region tolerance: every solid grid node within *half a grid
    // cell* of the mesh's minimum Z counts as "touching" the build plate --
    // generous enough to include a slightly-faceted/non-flat mesh base
    // (tessellation noise at that scale) without pulling in nodes from well
    // above the true contact surface.
    //
    // This used to be a full `layer_height`, which froze an entire
    // layer-height-thick *slab* of grid nodes to distance `0.0` instead of
    // a thin band hugging the true contact plane. Since the FMM front then
    // marches outward from the *far* edge of that frozen slab (at
    // `min.z + layer_height`, not `min.z`), every order value effectively
    // reported the distance from one layer height *above* the true base --
    // shifting the entire reconstructed field (and therefore every printed
    // layer's Z) up by a full `layer_height` relative to the `Height`
    // order field's convention, so the first Gcode move landed at
    // `2 * layer_height` instead of `layer_height`. Shrinking the tolerance
    // to a fraction of the grid's own cell size keeps the frozen band thin
    // enough that its far edge sits within one grid cell of `min.z`,
    // matching `Height`'s convention that order `0` corresponds to the
    // mesh's contact surface itself.
    let seed_tolerance = cell_size / 2.0;
    let is_seed_region = |p: DVec3| p.z <= min.z + seed_tolerance;
    let height_along = ConstantAxisHeight::new(BUILD_DIRECTION, min);

    let faces: Vec<[usize; 3]> = mesh
        .indices
        .chunks_exact(3)
        .map(|chunk| [chunk[0] as usize, chunk[1] as usize, chunk[2] as usize])
        .collect();
    if faces.is_empty() {
        // No triangles to classify solid/void against: fall back to an
        // unconstrained march (every node solid) rather than treating
        // every node as void, which would leave the whole grid
        // unreachable.
        let is_solid = |_p: DVec3| true;
        return EikonalOrderField::new_with_occupancy_and_seed_region_and_slope_limit(
            min,
            max,
            cell_size,
            &is_solid,
            &is_seed_region,
            Some(slope_profile),
            Some(&height_along),
        );
    }
    let owned_sdf;
    let sdf = match existing_sdf {
        Some(sdf) => sdf,
        None => {
            owned_sdf = MeshSdf::new(mesh.vertices.clone(), faces.clone());
            &owned_sdf
        }
    };

    let conform_top = config.eikonal_conform_top_surfaces;
    let conform_bottom = config.eikonal_conform_bottom_surfaces;
    let skin_depth = config.eikonal_conformal_skin_depth_mm();
    let top_detach_deg = config.eikonal_conformal_max_angle_deg();
    let bottom_detach_deg = config.eikonal_conformal_bottom_max_angle_deg();

    let side_faces = |upward: bool, detach_deg: f64| -> Vec<[usize; 3]> {
        let cos_limit = detach_deg.to_radians().cos();
        mesh.indices
            .chunks_exact(3)
            .filter_map(|chunk| {
                let [i0, i1, i2] = [chunk[0] as usize, chunk[1] as usize, chunk[2] as usize];
                let v0 = mesh.vertices[i0];
                let v1 = mesh.vertices[i1];
                let v2 = mesh.vertices[i2];
                let normal = (v1 - v0).cross(v2 - v0);
                let normal_len = normal.length();
                if normal_len <= 1e-9 {
                    return None;
                }
                let cos_up = normal.dot(BUILD_DIRECTION) / normal_len;
                let facing = if upward {
                    cos_up >= cos_limit
                } else {
                    let on_bed = v0.z <= min.z + seed_tolerance
                        && v1.z <= min.z + seed_tolerance
                        && v2.z <= min.z + seed_tolerance;
                    cos_up <= -cos_limit && !on_bed
                };
                facing.then_some([i0, i1, i2])
            })
            .collect()
    };

    let build_side_sdf = |faces: Vec<[usize; 3]>| -> Option<MeshSdf> {
        (!faces.is_empty()).then(|| MeshSdf::new(mesh.vertices.clone(), faces))
    };
    let top_sdf = conform_top
        .then(|| build_side_sdf(side_faces(true, top_detach_deg)))
        .flatten();
    let bottom_sdf = conform_bottom
        .then(|| build_side_sdf(side_faces(false, bottom_detach_deg)))
        .flatten();

    let is_top_seed = |p: DVec3| {
        top_sdf
            .as_ref()
            .is_some_and(|s| s.sample(p).value.abs() <= cell_size)
    };
    let is_bottom_seed = |p: DVec3| {
        bottom_sdf
            .as_ref()
            .is_some_and(|s| s.sample(p).value.abs() <= cell_size)
    };

    let surface_weight = config.eikonal_surface_order_weight();
    let surface_times = (surface_weight > 0.0).then(|| {
        manifold_fidget::surface_eikonal::solve_surface_eikonal(
            &mesh.vertices,
            &faces,
            is_seed_region,
        )
    });
    let surface_min_order = |p: DVec3| -> Option<f64> {
        let surface_times = surface_times.as_ref()?;
        let sample = sdf.sample(p);
        if sample.value.abs() <= skin_depth {
            if let Some((face_idx, closest, _)) = sdf.nearest(p) {
                if face_idx < faces.len() {
                    let [i0, i1, i2] = faces[face_idx];
                    let t = manifold_fidget::surface_eikonal::interpolate_barycentric(
                        mesh.vertices[i0],
                        mesh.vertices[i1],
                        mesh.vertices[i2],
                        surface_times[i0],
                        surface_times[i1],
                        surface_times[i2],
                        closest,
                    );
                    if t.is_finite() {
                        return Some(t * surface_weight);
                    }
                }
            }
        }
        None
    };

    let options = manifold_fidget::eikonal::ConformalSurfaceOptions {
        is_top_seed_region: top_sdf
            .is_some()
            .then_some(&is_top_seed as &(dyn Fn(DVec3) -> bool + Sync)),
        is_bottom_seed_region: bottom_sdf
            .is_some()
            .then_some(&is_bottom_seed as &(dyn Fn(DVec3) -> bool + Sync)),
        skin_depth_mm: skin_depth,
        top_detach_angle_deg: top_detach_deg,
        bottom_detach_angle_deg: bottom_detach_deg,
        detach_feather_mm: 2.0 * config.wall_line_width,
        target_lipschitz_constant: 1.0,
        surface_min_order: (surface_weight > 0.0)
            .then_some(&surface_min_order as &(dyn Fn(DVec3) -> Option<f64> + Sync)),
    };
    let is_solid = |p: DVec3| sdf.sample(p).value <= cell_size;
    EikonalOrderField::new_conformal_with_occupancy_and_seed_regions_and_slope_limit(
        min,
        max,
        cell_size,
        &is_solid,
        &is_seed_region,
        &options,
        Some(slope_profile),
        Some(&height_along),
    )
}

/// Hard cap on the dense Eikonal grid's total node count (`dims[0] *
/// dims[1] * dims[2]`). At 8 bytes/node this bounds the `distances` buffer
/// alone to ~8MB, but the real driver of this number is
/// `march_from_seeds`'s narrow-band bookkeeping, not the distance buffer:
/// its heap can hold up to ~6 (one per face-neighbor direction) stale
/// entries per node before they are popped and discarded, and each
/// `HeapEntry` is ~32 bytes, so worst case the march's `BinaryHeap` alone
/// can reach roughly `6 * MAX_EIKONAL_GRID_NODES * 32` bytes -- at
/// 8,000,000 nodes that is close to 1.5GB on top of the `occupied`/frozen
/// bookkeeping and the mesh's own BVH, which is what produced the reported
/// OOM even after the grid's distance buffer itself was bounded. Chosen
/// empirically as "enough resolution to matter, not enough to OOM a real
/// part" -- not derived from a hard per-machine memory budget. Revisit if
/// this proves too coarse in practice.
const MAX_EIKONAL_GRID_NODES: f64 = 1_000_000.0;

/// Coarsens (grows) `requested_cell_size` as needed so that a dense grid
/// covering `extent` at that cell size would not exceed
/// [`MAX_EIKONAL_GRID_NODES`] total nodes -- see `eikonal_field_for`'s docs
/// for why this guard exists (unbounded resolution on a large part is an
/// OOM waiting to happen, since the grid is a dense `Vec<f64>`, not a
/// sparse/adaptive structure). Never shrinks the requested cell size: a
/// small part that would need few nodes at the requested resolution is
/// returned unchanged. Non-finite/non-positive `requested_cell_size` is
/// returned unchanged too -- `EikonalOrderField::new`/`new_with_occupancy`
/// already document a resolution fallback for that case, so this helper
/// only clamps well-formed requests.
fn clamp_cell_size_to_node_budget(extent: DVec3, requested_cell_size: f64) -> f64 {
    if !requested_cell_size.is_finite() || requested_cell_size <= 0.0 {
        return requested_cell_size;
    }
    // +1 mirrors `grid_node_count`'s "at least one node per axis, plus one
    // for the trailing edge" so this estimate matches the grid actually
    // built downstream.
    let estimated_nodes = |cell_size: f64| -> f64 {
        (extent.x.max(0.0) / cell_size + 1.0).ceil()
            * (extent.y.max(0.0) / cell_size + 1.0).ceil()
            * (extent.z.max(0.0) / cell_size + 1.0).ceil()
    };
    let nodes = estimated_nodes(requested_cell_size);
    if !nodes.is_finite() || nodes <= MAX_EIKONAL_GRID_NODES {
        return requested_cell_size;
    }
    // Node count scales as ~1/cell_size^3 for a fixed extent, so scale the
    // cell size by the cube root of the overshoot ratio to bring the
    // estimate back within budget. Each axis's node count is a `ceil`, so a
    // single cube-root scale can still overshoot the budget by up to one
    // node per axis; a few correction passes (each nudging the scale up by
    // the residual overshoot ratio) converge past that rounding slack
    // without needing a closed-form inverse of `ceil`.
    let mut cell_size = requested_cell_size * (nodes / MAX_EIKONAL_GRID_NODES).cbrt();
    for _ in 0..8 {
        let nodes = estimated_nodes(cell_size);
        if !nodes.is_finite() || nodes <= MAX_EIKONAL_GRID_NODES {
            break;
        }
        cell_size *= (nodes / MAX_EIKONAL_GRID_NODES).cbrt();
    }
    cell_size
}

/// Multiplier applied to `config.layer_height` to bound
/// [`reconstruct_on_order_field`]'s numeric solve (see
/// [`max_along_for`]) — generous enough to cover legitimate tall
/// reconstructions for well-behaved (monotonic) fields, while still
/// containing a non-monotonic field's (e.g. `Eikonal`'s) failure mode to a
/// bounded region instead of an unbounded runaway.
const MAX_ALONG_LAYER_HEIGHTS: f64 = 50.0;

/// Bound passed as `reconstruct_on_order_field`'s `max_along` for a given
/// `config`: `MAX_ALONG_LAYER_HEIGHTS` layer heights either side of zero.
pub fn max_along_for(config: &SlicerConfig) -> f64 {
    config.layer_height.abs().max(f64::EPSILON) * MAX_ALONG_LAYER_HEIGHTS
}

/// Resolves `kind`'s effective `(axis, apex, slope)` for order-field-aware
/// plane-basis/reconstruction math (see [`reconstruct_on_order_field`]).
/// `Height` is treated as a degenerate cone — `apex` at the origin, `slope`
/// `0.0` — along [`BUILD_DIRECTION`] specifically (not
/// `config.order_field_axis`, which is documented as inert unless
/// `order_field` is `Conical`), so this always matches whatever
/// `slicing::slice_mesh` actually used to produce `layer.order` in the
/// first place.
pub fn resolve_axis_apex_slope(kind: OrderFieldKind, config: &SlicerConfig) -> (DVec3, DVec3, f64) {
    match kind {
        OrderFieldKind::Height => (BUILD_DIRECTION, DVec3::ZERO, 0.0),
        OrderFieldKind::Conical => (
            config.order_field_axis,
            config.order_field_apex,
            config.order_field_slope,
        ),
        // Eikonal has no single global axis/apex (its front is a
        // mesh-derived FMM grid, not axisymmetric) -- treated the same as
        // `Height` here (degenerate cone along `BUILD_DIRECTION`) purely as
        // a projection-plane choice for `compute_solid_fill_boundaries`'s 2D
        // flattening; `reconstruct_on_order_field`'s numeric solve against
        // the *actual* cached `EikonalOrderField` (not this closed-form
        // triple) is what makes the reconstructed geometry correct
        // regardless of this choice.
        OrderFieldKind::Eikonal => (BUILD_DIRECTION, DVec3::ZERO, 0.0),
    }
}

/// Order-field-aware inverse of [`crate::polygon2d::from_2d`]: reconstructs
/// each `(u, v)` point's true position along `axis` by numerically solving
/// `field.order(apex + basis1*u + basis2*v + axis*along) == target_order`
/// for `along`, via bracket expansion + bisection on `field` itself.
///
/// This works for *any* [`OrderField`], not just axisymmetric ones: it
/// replaces the previous closed-form cone inversion
/// (`along = order + slope * radial`), which only held for
/// [`ConicalOrderField`]/[`HeightOrderField`] and had no meaning for a
/// non-axisymmetric field such as a future Eikonal/front-propagation field
/// (see `NON_PLANAR_SLICING.md`) — rather than adding a second closed-form
/// special case, every field now goes through the same numeric solve, so
/// adding another field kind later needs no change here at all.
///
/// Relies on the documented well-posedness precondition that `field.order`
/// is monotonic along `axis` at any fixed in-plane position (true of
/// `Height`/`Conical` by construction, and required of any future field
/// for slicing to make sense in the first place — see
/// `NON_PLANAR_SLICING.md`'s well-posedness discussion), but **not**
/// necessarily *increasing*: `Height`'s `order(p) == p.dot(axis)` moves in
/// the same direction as `along`, while `Eikonal`'s `order(p)` is an
/// unsigned front-propagation distance that *decreases* as `along`
/// increases along `axis == BUILD_DIRECTION` (larger `along` means smaller
/// `p.z`, i.e. closer to the seeded build-plate surface). Because of this,
/// [`solve_along`] does not assume the sign of `target_order` maps
/// directly onto `along` — it probes `field`'s *local* direction of travel
/// near `along == 0` and mirrors `target_order` accordingly before
/// centering the search there, so the bracket starts near the
/// physically-plausible root for either sign convention instead of
/// hardcoding one. This also does not hold exactly outside a small
/// neighborhood for non-monotonic fields (e.g. `Eikonal` near
/// holes/overhangs, where there can be more than one `along` solving the
/// equation at a given `(u, v)`) — the probe only needs to get the right
/// *side*, not the exact value; bisection finds the precise root within
/// whatever bracket that side yields. Bracket expansion is additionally
/// bounded to `max_along` either side of the initial guess (see
/// [`solve_along`]) rather than doubling toward `f64` extremes — this
/// keeps a non-monotonic field's failure to a locally-bounded,
/// still-imperfect reconstruction instead of flinging points arbitrarily
/// far along `axis`.
///
/// `basis1`/`basis2` must be the orthonormal in-plane basis perpendicular
/// to `axis` (e.g. from `manifold_fidget::contour::plane_basis(axis)`),
/// matching whatever produced `contours`' `(u, v)` coordinates via
/// `polygon2d::to_2d(.., basis1, basis2, apex)`.
#[allow(clippy::too_many_arguments)] // one param per geometric input; a config struct would obscure the plane-basis/reconstruction math this directly mirrors
pub fn reconstruct_on_order_field<F: OrderField + ?Sized>(
    contours: Vec<Vec<[f64; 2]>>,
    basis1: DVec3,
    basis2: DVec3,
    axis: DVec3,
    apex: DVec3,
    target_order: f64,
    max_along: f64,
    field: &F,
) -> Vec<Vec<DVec3>> {
    contours
        .into_iter()
        .map(|contour| {
            contour
                .into_iter()
                .map(|[u, v]| {
                    let planar = apex + basis1 * u + basis2 * v;
                    // Contour points come from real isosurface extraction
                    // on the mesh (see `contour::extract_order_contours_on_mesh`),
                    // so `reconstruct_point_on_order_field` (which now tries
                    // both an axis-only search and, if that fails to
                    // bracket -- e.g. a steep isosurface -- a full 3D
                    // gradient projection) is expected to actually succeed.
                    // Falling back to `planar` itself here would mean the
                    // field has literally no information anywhere near a
                    // point already known to sit on its own isosurface,
                    // which should not happen in practice; kept only as a
                    // best-effort last resort, not a proof obligation.
                    reconstruct_point_on_order_field(planar, axis, target_order, max_along, field)
                        .unwrap_or(planar)
                })
                .collect()
        })
        .collect()
}

/// Like [`reconstruct_on_order_field`], but seeds every point's height
/// from the nearest (in `(u, v)`) point of `references` -- 3D loops already
/// known to lie on (or very near) the layer's `target_order` isosurface,
/// e.g. the layer's own `infill_boundary` before a 2D boolean op -- and
/// then refines with [`project_onto_isosurface`]'s local Newton descent
/// instead of an axis-ray bracket search from the `along == 0` plane.
///
/// This exists because [`reconstruct_on_order_field`]'s axis-ray solve is
/// launched from the world `axis == 0` plane with its initial bracket
/// centered at `along ~= +-target_order`: for a *non-monotonic* field
/// (`Eikonal` on reentrant/threaded geometry) the same vertical column can
/// cross `order == target_order` at several heights, and that search can
/// bracket and "exactly" solve a *different branch* many layer heights
/// from the true one -- producing steep near-vertical spikes in
/// infill/solid boundaries reconstructed after 2D boolean ops. Boolean-op
/// output points always lie on (or at intersections of) input edges, so
/// the nearest input point's height is a locally-correct same-branch seed;
/// Newton descent from there inherently converges to the local branch.
///
/// Refinement acceptance is field-adaptive: a refined point that wandered
/// farther from its seed than the seed's own |residual| justifies (slope
/// slack factor 4, plus a small absolute slack) is rejected in favor of
/// the seed itself. Empty `references` falls back to
/// [`reconstruct_on_order_field`] wholesale.
#[allow(clippy::too_many_arguments)] // one param per geometric input; see reconstruct_on_order_field
pub fn reconstruct_on_order_field_near<F: OrderField + ?Sized>(
    contours: Vec<Vec<[f64; 2]>>,
    references: &[Vec<DVec3>],
    basis1: DVec3,
    basis2: DVec3,
    axis: DVec3,
    apex: DVec3,
    target_order: f64,
    max_along: f64,
    field: &F,
) -> Vec<Vec<DVec3>> {
    // Project references once into (u, v, along) triples.
    let refs: Vec<(f64, f64, f64)> = references
        .iter()
        .flatten()
        .map(|&p| {
            let rel = p - apex;
            (rel.dot(basis1), rel.dot(basis2), rel.dot(axis))
        })
        .collect();
    if refs.is_empty() {
        return reconstruct_on_order_field(
            contours,
            basis1,
            basis2,
            axis,
            apex,
            target_order,
            max_along,
            field,
        );
    }

    contours
        .into_iter()
        .map(|contour| {
            contour
                .into_iter()
                .map(|[u, v]| {
                    let nearest_along = refs
                        .iter()
                        .min_by(|a, b| {
                            let da = (a.0 - u).powi(2) + (a.1 - v).powi(2);
                            let db = (b.0 - u).powi(2) + (b.1 - v).powi(2);
                            da.total_cmp(&db)
                        })
                        .map(|&(_, _, along)| along)
                        .unwrap_or(0.0);
                    let seed = apex + basis1 * u + basis2 * v + axis * nearest_along;
                    let seed_residual = field.order(seed) - target_order;
                    if !seed_residual.is_finite() {
                        return seed;
                    }
                    // Small absolute slack on top of the residual-derived
                    // bound: `max_along` is 50 layer heights (see
                    // `max_along_for`), so 0.04x of it is ~2 layer heights.
                    let accept = seed_residual.abs() * 4.0 + max_along * 0.04;
                    project_onto_isosurface(field, seed, target_order, max_along)
                        .filter(|p| (*p - seed).length() <= accept)
                        .unwrap_or(seed)
                })
                .collect()
        })
        .collect()
}

/// Single-point building block behind [`reconstruct_on_order_field`]: given
/// a transverse (perpendicular-to-`axis`) reference position `planar`,
/// solves for the `axis`-offset that lands on `field`'s `target_order`
/// isosurface and returns the resulting 3D world point, or `None` if no
/// finite `field.order` sample was ever observed anywhere along the search
/// (see [`solve_along`]'s doc) -- i.e. `planar`'s whole column is outside
/// the region `field` has any information about at all (e.g. a straight
/// ray from a synthetic, not-necessarily-on-the-mesh `(u, v)` location,
/// such as an infill scan-line/loop-edge crossing, that passes entirely
/// through empty space next to reentrant/threaded geometry an `Eikonal`
/// front never reached). Callers must not treat `None` as "assume
/// `along == 0`" -- that previously produced exactly the flat, badly-wrong
/// spike-plane bug this return type exists to prevent; see this function's
/// callers for their fallback strategy when reconstruction fails.
///
/// `planar` need not be `apex + basis1 * u + basis2 * v` specifically — any
/// point with the desired transverse `(u, v)` location works, regardless of
/// its own `axis` component, since [`solve_along`] searches `along` freely
/// in either direction from `planar` and only the final `planar + axis *
/// along` (not `planar` itself) needs to land on the isosurface. This lets
/// callers reconstruct at points derived purely from an orthonormal
/// `(u_dir, v_dir, axis)` frame (e.g. a rotated infill-scan frame) without
/// re-deriving `apex`'s original `basis1`/`basis2` coordinates — see
/// `infill::MonotonicInfill::generate`'s per-scan-crossing reconstruction,
/// which needs this exact single-point form: unlike a loop's own vertices
/// (already reconstructed once by `reconstruct_on_order_field`), a
/// scan-line/loop-edge *crossing* is a new `(u, v)` location that does not
/// coincide with any already-reconstructed vertex, so its true `axis`
/// height must be re-solved from the field rather than linearly
/// interpolated between the edge's two endpoint heights — linear
/// interpolation is only exact for a `HeightOrderField` (`w` independent of
/// `(u, v)`), not for a curved field like `Eikonal`/`Conical` where `w` can
/// vary sharply across a short edge near curved/threaded geometry.
pub(crate) fn reconstruct_point_on_order_field<F: OrderField + ?Sized>(
    planar: DVec3,
    axis: DVec3,
    target_order: f64,
    max_along: f64,
    field: &F,
) -> Option<DVec3> {
    match solve_along(field, planar, axis, target_order, max_along)? {
        SolveAlong::Exact(along) => Some(planar + axis * along),
        SolveAlong::ClosestObserved(along) => {
            // The axis-only ray search never actually bracketed the
            // target (see `solve_along`'s doc) -- this is typically a
            // *steep* isosurface, where `order` changes fast in-plane but
            // slowly along `axis`, so the true nearest isosurface point
            // needs a lateral shift the axis-only search can never make
            // no matter how far it travels along `axis` alone. Try a full
            // 3D gradient projection from the same starting point first;
            // only fall back to the cruder axis-only closest-observed
            // sample if that also fails to converge.
            project_onto_isosurface(field, planar, target_order, max_along)
                .or(Some(planar + axis * along))
        }
    }
}

/// Refines an already-approximately-on-the-isosurface `seed` onto `field`'s
/// exact `target_order` isosurface via [`project_onto_isosurface`]'s
/// Newton-style 3D gradient descent (each step clamped to `max_step`).
///
/// This exists for callers that already hold a locally-sane starting point
/// on the *correct branch* of the isosurface (e.g. an infill scan-line
/// crossing lerped between two real, already-reconstructed boundary
/// points) and only need the residual curvature error removed. Routing
/// such a seed through [`reconstruct_point_on_order_field`] instead is
/// actively dangerous for a non-monotonic field (`Eikonal` on
/// reentrant/threaded geometry): its `solve_along` centers the bracketing
/// search on `along ~= +-target` -- correct for a bare in-plane `planar`
/// at `along == 0`, but far away from a seed that already sits at the
/// right height -- so it can bracket and "exactly" solve a *different
/// branch* of the isosurface many layer-heights away. Gradient descent
/// from a near-surface seed, by contrast, inherently converges to the
/// local branch.
pub(crate) fn refine_point_onto_order_field<F: OrderField + ?Sized>(
    seed: DVec3,
    target_order: f64,
    max_step: f64,
    field: &F,
) -> Option<DVec3> {
    project_onto_isosurface(field, seed, target_order, max_step)
}

/// Solves `field.order(planar + axis * along) == target` for `along`
/// (see [`reconstruct_on_order_field`]'s preconditions on `field`).
///
/// Different [`OrderField`]s use different sign conventions for how
/// `order` relates to `along`: `HeightOrderField::order(p) == p.dot(axis)`
/// moves in the *same* direction as `along`, while
/// `EikonalOrderField::order(p)` is an unsigned distance that moves in the
/// *opposite* direction along `axis == BUILD_DIRECTION` (see
/// [`reconstruct_on_order_field`]'s doc). To center the search on the
/// physically-plausible root regardless of which convention `field` uses,
/// this first probes `residual` at a small `+epsilon`/`-epsilon` pair
/// around `along == 0.0` and picks whichever of `target`/`-target` moves
/// `along` in the direction the probe indicates `order` actually
/// increases — falling back to `+target` if both probe samples are
/// non-finite (e.g. a point the field's front never reached) since that
/// matches every currently-known field's convention.
///
/// From that initial guess, expands a bracket outward by doubling steps
/// until the target sign is bracketed, clamped to `max_along` either side
/// of the initial guess (see [`reconstruct_on_order_field`]'s doc on why
/// this bound exists — it protects against a non-monotonic field, e.g.
/// `Eikonal` near holes/overhangs, sending the doubling search toward
/// `f64` extremes), then bisects (`MAX_BISECT_ITERS` iterations —
/// comfortably enough for `f64` precision on any bracket reached by the
/// doubling above).
///
/// If a sign change is never actually bracketed within `max_along` (e.g.
/// a straight ray along `axis` from `planar` never crosses `target` at
/// all — common for a non-monotonic field like `Eikonal` on reentrant or
/// threaded geometry, where the true isosurface for this `(u, v)` column
/// simply isn't reachable by a vertical search), this falls back to a
/// bounded, evenly-spaced scan of the whole `[min_bound, max_bound]`
/// range and returns the *closest-to-target* sample found there (rather
/// than just the handful of widely-spaced points visited by the
/// exponential doubling above, which can skip straight over the region
/// where the field's true closest approach to `target` actually lives),
/// as long as at least one *finite* sample was actually observed
/// somewhere along the ray. Previously this fell through to the last
/// bracket's midpoint, which for a failed search sits right at the
/// `max_along` bound — up to 50 layer heights away — producing exactly
/// the multi-mm vertical spikes seen on screw-thread geometry with an
/// `Eikonal` order field; a full bounded scan is a far tighter,
/// caller-visible "best effort" position for a pathological/misbehaving
/// field than either "as far as the search was allowed to wander" or a
/// handful of sparse doubling-search samples.
///
/// If **no** finite sample was ever observed anywhere along the ray
/// (`field.order` returns non-finite — e.g. `Eikonal`'s
/// front-never-reached `f64::INFINITY` — at every probed `along`,
/// including `along == 0.0` itself), this returns `None` rather than
/// falling back to `along == 0.0`. Returning `0.0` here previously
/// silently collapsed every such column onto the `axis == 0` plane (world
/// `Z == 0` for the common `axis == BUILD_DIRECTION` case) — a
/// caller-invisible wrong answer that reads as a real point, producing the
/// flat "spike floor" seen when many infill scan-line crossings from
/// reentrant/threaded geometry all land in a region the field's front
/// never reached at all. `None` forces callers to make an explicit choice
/// for a column with *zero* information instead of silently fabricating
/// one.
fn solve_along<F: OrderField + ?Sized>(
    field: &F,
    planar: DVec3,
    axis: DVec3,
    target: f64,
    max_along: f64,
) -> Option<SolveAlong> {
    const TOLERANCE: f64 = 1e-9;
    const MAX_BISECT_ITERS: u32 = 64;

    let max_along = max_along.abs();
    let residual = |along: f64| field.order(planar + axis * along) - target;

    // Best (lowest |residual|) sample seen anywhere during the search --
    // the fallback returned if bracketing never finds a real sign change
    // (see this function's doc). `found_any_finite` distinguishes "found
    // some real information, just no exact bracket" (still worth using
    // `best_along`) from "zero finite samples anywhere" (must return
    // `None` -- see this function's doc).
    let mut best_along = 0.0_f64;
    let mut best_residual = f64::INFINITY;
    let mut found_any_finite = false;
    let mut consider = |along: f64, r: f64| {
        if r.is_finite() {
            found_any_finite = true;
            if r.abs() < best_residual {
                best_residual = r.abs();
                best_along = along;
            }
        }
    };

    // Probe the field's local direction of travel near `along == 0.0` to
    // pick the correct sign convention (see this function's doc) before
    // committing to an initial guess.
    let probe = (max_along * 1e-3).max(f64::EPSILON);
    let f_pos = residual(probe);
    let f_neg = residual(-probe);
    consider(probe, f_pos);
    consider(-probe, f_neg);
    let initial_guess = if f_pos.is_finite() && f_neg.is_finite() && f_pos < f_neg {
        -target
    } else {
        target
    };
    let min_bound = initial_guess - max_along;
    let max_bound = initial_guess + max_along;

    let mut lo = initial_guess;
    let mut hi = initial_guess;
    let mut f_lo = residual(lo);
    consider(lo, f_lo);
    if f_lo.abs() <= TOLERANCE {
        return Some(SolveAlong::Exact(lo));
    }

    let mut step = 1.0_f64.min(max_along.max(f64::EPSILON));
    let mut bracketed = false;
    if f_lo < 0.0 {
        // Need a larger `along`; grow `hi` until the residual turns
        // non-negative, never exceeding `max_bound`.
        loop {
            hi = (hi + step).min(max_bound);
            let f_hi = residual(hi);
            consider(hi, f_hi);
            if f_hi >= 0.0 {
                bracketed = true;
                break;
            }
            if hi >= max_bound {
                break;
            }
            step *= 2.0;
        }
    } else {
        // Need a smaller `along`; shrink `lo` until the residual turns
        // non-positive, never going below `min_bound`.
        loop {
            lo = (lo - step).max(min_bound);
            let f_lo_candidate = residual(lo);
            consider(lo, f_lo_candidate);
            if f_lo_candidate <= 0.0 {
                bracketed = true;
                break;
            }
            if lo <= min_bound {
                break;
            }
            step *= 2.0;
        }
    }

    if !bracketed {
        // No sign change was ever found within `max_along` -- the true
        // isosurface isn't reachable by a straight ray along `axis` from
        // `planar` at all (common for a non-monotonic field like
        // `Eikonal` on reentrant/threaded geometry). The exponential
        // doubling above only visited a handful of widely-spaced samples
        // (each step roughly doubling the previous one), so "closest
        // observed so far" from *those* alone can still be a poor,
        // spurious guess -- e.g. it might have jumped straight from a
        // sample near `along == 0` to one near `max_bound` without ever
        // probing the region in between, which is exactly where the
        // field's true closest approach to `target` usually lives. Do one
        // bounded, evenly-spaced scan across the whole reachable range to
        // find the actual closest-to-target sample before giving up --
        // this only runs for the rare pathological columns that fail to
        // bracket at all, so the extra field evaluations are not paid on
        // the common path.
        const FALLBACK_SCAN_STEPS: u32 = 256;
        let span = max_bound - min_bound;
        if span.is_finite() && span > 0.0 {
            for i in 0..=FALLBACK_SCAN_STEPS {
                let along = min_bound + span * (f64::from(i) / f64::from(FALLBACK_SCAN_STEPS));
                consider(along, residual(along));
            }
        }
        // Closest-to-target sample actually observed, not the outer edge
        // of the failed bracket (see this function's doc) -- unless
        // *nothing* finite was ever observed, in which case there is
        // nothing to fall back to at all.
        return found_any_finite.then_some(SolveAlong::ClosestObserved(best_along));
    }

    f_lo = residual(lo);
    for _ in 0..MAX_BISECT_ITERS {
        let mid = 0.5 * (lo + hi);
        let f_mid = residual(mid);
        if f_mid.abs() <= TOLERANCE {
            return Some(SolveAlong::Exact(mid));
        }
        if (f_mid < 0.0) == (f_lo < 0.0) {
            lo = mid;
            f_lo = f_mid;
        } else {
            hi = mid;
        }
    }
    Some(SolveAlong::Exact(
        (0.5 * (lo + hi)).clamp(min_bound, max_bound),
    ))
}

/// Distinguishes a genuinely-bracketed-and-bisected root from
/// [`solve_along`]'s no-bracket "closest observed sample" fallback, so
/// [`reconstruct_point_on_order_field`] can try a fuller 3D projection
/// (see [`project_onto_isosurface`]) before accepting the cruder
/// axis-only fallback.
enum SolveAlong {
    /// A real sign change was bracketed and refined by bisection; `along`
    /// lands on the isosurface (within [`solve_along`]'s `TOLERANCE`).
    Exact(f64),
    /// No sign change was ever bracketed; `along` is merely the
    /// closest-to-target sample observed during the search, not an actual
    /// root.
    ClosestObserved(f64),
}

/// Projects `start` onto `field`'s `target` isosurface by moving freely in
/// full 3D (not constrained to a single `axis`), for cases where
/// [`solve_along`]'s axis-only ray search never brackets the target at
/// all. That axis-only search is exact and cheap for the common case, but
/// fundamentally cannot succeed for a *steep* isosurface (`order` changes
/// fast in-plane, slowly along `axis`) where the true nearest isosurface
/// point requires a lateral shift, not just a different `along` -- exactly
/// the geometry that produced `Eikonal`-driven infill points collapsing
/// onto the flat `axis == 0` plane near steep threaded/reentrant
/// features.
///
/// Also used directly (rather than through
/// [`reconstruct_on_order_field_near`]'s single-nearest-reference seeding)
/// by [`crate::slicing::serpentine_stitch_block`] to subdivide a genuinely large
/// real wall-to-wall gap: that caller already has *two* known-correct
/// points (the previous and current loop's own vertices) bracketing the
/// target order, so it seeds from their true 3D midpoint instead of
/// picking one endpoint's `along` at a new in-plane location -- avoiding
/// both the plain axis-ray solve's risk of locking onto the wrong branch
/// (that solve is centered on the world `axis == 0` plane, with no
/// knowledge of which branch is correct) and `reconstruct_on_order_field_near`'s
/// small-residual-only acceptance filter (built for boolean-op output
/// already known to sit *near* the isosurface, which wrongly rejects a
/// large-but-real displacement as noise).
///
/// Uses a central-difference numeric gradient (`field` exposes only scalar
/// `order`, no analytic gradient) and Newton-style steps `p -= grad *
/// (residual / grad.length_squared())`, each step clamped to `max_step` so
/// a near-zero or wildly large local gradient can't produce a nonsensical
/// jump. Returns `None` if `field` is already non-finite at `start`
/// (nothing to descend from), if the gradient ever degenerates (zero or
/// non-finite -- no local direction of improvement), if a step lands
/// somewhere `field` is non-finite (stepped outside the field's known
/// region), or if the residual never converges within `MAX_ITERS` steps.
pub(crate) fn project_onto_isosurface<F: OrderField + ?Sized>(
    field: &F,
    start: DVec3,
    target: f64,
    max_step: f64,
) -> Option<DVec3> {
    const TOLERANCE: f64 = 1e-6;
    const MAX_ITERS: u32 = 64;
    let max_step = max_step.abs().max(f64::EPSILON);

    let mut p = start;
    let mut value = field.order(p);
    if !value.is_finite() {
        return None;
    }

    for _ in 0..MAX_ITERS {
        let residual = value - target;
        if residual.abs() <= TOLERANCE {
            return Some(p);
        }

        let grad = numeric_gradient(field, p)?;
        let grad_len_sq = grad.length_squared();
        if !grad_len_sq.is_finite() || grad_len_sq < 1e-12 {
            return None;
        }

        let mut step = grad * (residual / grad_len_sq);
        let step_len = step.length();
        if step_len > max_step {
            step *= max_step / step_len;
        }

        let next = p - step;
        let next_value = field.order(next);
        if !next_value.is_finite() {
            return None;
        }
        p = next;
        value = next_value;
    }

    None
}

/// Central-difference numeric gradient of `field.order` at `p` (`field`
/// exposes only a scalar `order`, no analytic gradient). Shared by
/// [`project_onto_isosurface`] and `toolpath::compensate_flat_nozzle`,
/// which both need a local surface-normal-like direction from an
/// arbitrary [`OrderField`].
///
/// Returns `None` if `field.order` is non-finite at any of the six sample
/// points (e.g. `p` is outside the region the field has information about
/// at all) -- callers must not treat that as "assume zero gradient", which
/// would silently fabricate a direction.
pub fn numeric_gradient<F: OrderField + ?Sized>(field: &F, p: DVec3) -> Option<DVec3> {
    const GRAD_EPS: f64 = 1e-4;

    let sample = |offset: DVec3| -> Option<f64> {
        let plus = field.order(p + offset);
        let minus = field.order(p - offset);
        if plus.is_finite() && minus.is_finite() {
            Some((plus - minus) / (2.0 * GRAD_EPS))
        } else {
            None
        }
    };

    Some(DVec3::new(
        sample(DVec3::X * GRAD_EPS)?,
        sample(DVec3::Y * GRAD_EPS)?,
        sample(DVec3::Z * GRAD_EPS)?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::DVec3;

    #[test]
    fn default_config_resolves_to_height_field_matching_build_direction() {
        let config = SlicerConfig::default();
        let mesh = crate::mesh::Mesh::default();
        let field = order_field_for(
            config.order_field,
            &config,
            &mesh,
            &manifold_fidget::slope_profile::SlopeProfile::new(Vec::new()),
        );

        let p = DVec3::new(1.0, 2.0, 3.0);
        let expected = HeightOrderField::new(BUILD_DIRECTION).order(p);
        assert_eq!(field.order(p), expected);
    }

    #[test]
    fn conical_kind_resolves_to_conical_field_with_configured_params() {
        let mut config = SlicerConfig {
            order_field: OrderFieldKind::Conical,
            ..SlicerConfig::default()
        };
        config.order_field_apex = DVec3::new(1.0, 2.0, 3.0);
        config.order_field_axis = DVec3::new(0.0, 0.0, 1.0);
        config.order_field_slope = 0.5;

        let mesh = crate::mesh::Mesh::default();
        let field = order_field_for(
            config.order_field,
            &config,
            &mesh,
            &manifold_fidget::slope_profile::SlopeProfile::new(Vec::new()),
        );

        let p = DVec3::new(4.0, 6.0, 10.0);
        let expected = ConicalOrderField::new(
            config.order_field_apex,
            config.order_field_axis,
            config.order_field_slope,
        )
        .order(p);
        assert_eq!(field.order(p), expected);
    }

    #[test]
    fn reconstruct_on_order_field_matches_closed_form_for_height_field() {
        let axis = BUILD_DIRECTION;
        let apex = DVec3::ZERO;
        let field = HeightOrderField::new(axis);
        let (basis1, basis2) = manifold_fidget::contour::plane_basis(axis);

        let target_order = 5.0;
        let contours = vec![vec![[1.0, -2.0], [3.0, 4.0]]];
        let reconstructed = reconstruct_on_order_field(
            contours,
            basis1,
            basis2,
            axis,
            apex,
            target_order,
            1000.0,
            &field,
        );

        for p in &reconstructed[0] {
            assert!((field.order(*p) - target_order).abs() < 1e-6);
        }
    }

    #[test]
    fn reconstruct_on_order_field_matches_closed_form_for_conical_field() {
        let axis = DVec3::new(0.0, 0.0, 1.0);
        let apex = DVec3::new(1.0, 2.0, 3.0);
        let slope = 0.4;
        let field = ConicalOrderField::new(apex, axis, slope);
        let (basis1, basis2) = manifold_fidget::contour::plane_basis(axis);

        let target_order = -2.0;
        let contours = vec![vec![[2.0, 0.0], [0.0, 5.0], [-3.0, -1.0]]];
        let reconstructed = reconstruct_on_order_field(
            contours.clone(),
            basis1,
            basis2,
            axis,
            apex,
            target_order,
            1000.0,
            &field,
        );

        for (p, [u, v]) in reconstructed[0].iter().zip(&contours[0]) {
            assert!((field.order(*p) - target_order).abs() < 1e-6);
            let radial = (u * u + v * v).sqrt();
            let expected_along = target_order + slope * radial;
            let expected = apex + basis1 * u + basis2 * v + axis * expected_along;
            assert!((*p - expected).length() < 1e-6);
        }
    }

    #[test]
    fn solve_along_falls_back_to_the_closest_observed_sample_when_no_root_is_bracketed() {
        // Regression test for the "screw-thread pug model produces wildly
        // spiking Eikonal infill" bug: when a straight ray along `axis`
        // from `planar` never actually crosses `target` (common for a
        // non-monotonic field like `Eikonal` on reentrant/threaded
        // geometry), `solve_along` must fall back to the closest-to-target
        // sample it actually observed, not the outer edge of the failed
        // bracket search (which previously produced points up to
        // `max_along` away -- a multi-mm spike).
        struct NeverReachesTarget;
        impl OrderField for NeverReachesTarget {
            fn order(&self, p: DVec3) -> f64 {
                // Symmetric "bowl" in `along` with its minimum (5.0) right
                // at `planar` itself: never reaches the target (0.0)
                // anywhere along the ray, so bracketing must fail no
                // matter how far the search doubles outward.
                p.z * p.z + 5.0
            }
        }

        let field = NeverReachesTarget;
        let axis = DVec3::Z;
        let planar = DVec3::new(3.0, -2.0, 0.0);
        let max_along = 10.0;

        let point = reconstruct_point_on_order_field(planar, axis, 0.0, max_along, &field)
            .expect("a finite sample was observed everywhere along this field, so reconstruction must succeed");

        assert!(
            (point - planar).length() < 1e-6,
            "expected the closest-observed fallback right at `planar` (along == 0.0), got {point:?} \
             (drifted {} units away)",
            (point - planar).length()
        );
    }

    #[test]
    fn solve_along_fallback_finds_a_narrow_near_root_the_doubling_probes_skip_over() {
        // Regression test for "infill Z heights are still spiky/wrong even
        // after the `None` fix": the exponential-doubling search only
        // visits a handful of widely-spaced `along` samples (roughly
        // 1, 3, 7, 15, ... units out from the initial guess here). A field
        // whose true closest approach to `target` sits in a narrow region
        // *between* two of those samples was previously invisible to the
        // no-bracket fallback, which only remembered the best of the
        // sparse doubling samples themselves -- e.g. it would report the
        // residual at `along == 3` or `along == 7` (tied, both far from
        // the root) instead of the true near-root minimum at `along == 5`
        // sitting right between them. The bounded dense scan added to the
        // no-bracket fallback must find that narrow near-root region
        // instead.
        struct NarrowBumpBetweenDoublingSamples;
        impl OrderField for NarrowBumpBetweenDoublingSamples {
            fn order(&self, p: DVec3) -> f64 {
                // Always negative (so `residual = order - target` never
                // changes sign and bracketing can never succeed), with its
                // magnitude minimized in a narrow region right at
                // `p.z == 5.0` -- a point the doubling search's sample
                // sequence (1, 3, 7, 15, ...) straddles but never lands on.
                -(p.z - 5.0).abs() - 0.001
            }
        }

        let field = NarrowBumpBetweenDoublingSamples;
        let axis = DVec3::Z;
        let planar = DVec3::new(0.0, 0.0, 0.0);
        let target = 0.0;
        let max_along = 50.0;

        let point = reconstruct_point_on_order_field(planar, axis, target, max_along, &field)
            .expect("a finite sample was observed everywhere along this field");

        assert!(
            (point.z - 5.0).abs() < 0.5,
            "expected the dense fallback scan to land near the true closest-approach at z == 5.0, \
             got {point:?} -- a sparse-doubling-only fallback would have landed near z == 3.0 or \
             z == 7.0 instead"
        );
    }

    #[test]
    fn solve_along_returns_none_when_no_finite_sample_is_ever_observed() {
        // Regression test for the "infill scan-line crossings snap to a
        // flat Z == 0 plane" bug: when *every* probed `along` (including
        // `along == 0.0` itself) yields a non-finite `field.order` (e.g.
        // an `EikonalOrderField` whose front never reached this column at
        // all), `solve_along`/`reconstruct_point_on_order_field` must
        // return `None` -- there is zero information to reconstruct a
        // position from -- rather than silently defaulting to
        // `along == 0.0`, which previously collapsed every such column
        // onto the `axis == 0` world plane (a caller-invisible wrong
        // answer masquerading as a real point).
        struct NeverReached;
        impl OrderField for NeverReached {
            fn order(&self, _p: DVec3) -> f64 {
                f64::INFINITY
            }
        }

        let field = NeverReached;
        let axis = DVec3::Z;
        let planar = DVec3::new(1.0, 2.0, 0.0);
        let max_along = 10.0;

        assert!(reconstruct_point_on_order_field(planar, axis, 0.0, max_along, &field).is_none());
    }

    #[test]
    fn reconstruct_projects_laterally_onto_a_steep_isosurface_the_axis_only_search_cannot_reach() {
        // Regression test for "infill Z heights are still spiky/wrong even
        // after the no-bracket dense-scan fix, specifically where the
        // Eikonal isosurface is very steep": an axis-only ray search (fixed
        // in-plane position, varying only `along` the build direction) can
        // never reach the isosurface at all when `order` barely depends on
        // `along` but depends steeply on the in-plane position -- the
        // isosurface is nearly *parallel* to `axis`, so no amount of
        // marching along `axis` alone changes the residual. This field is
        // the extreme case: `order` doesn't depend on `along` (== world Z
        // here) whatsoever, only on `x`, so `solve_along`'s axis-only search
        // must fail to bracket no matter how far it searches -- exactly the
        // steep-isosurface failure mode `project_onto_isosurface`'s lateral
        // 3D movement exists to recover from.
        struct SteepWallAlongX;
        impl OrderField for SteepWallAlongX {
            fn order(&self, p: DVec3) -> f64 {
                1000.0 * p.x
            }
        }

        let field = SteepWallAlongX;
        let axis = DVec3::Z;
        let planar = DVec3::new(5.0, 0.0, 0.0);
        let target_order = 0.0;
        let max_along = 10.0;

        let point = reconstruct_point_on_order_field(planar, axis, target_order, max_along, &field)
            .expect("a full 3D projection should recover this steep isosurface");

        assert!(
            (field.order(point) - target_order).abs() < 1e-6,
            "point {point:?} does not actually lie on the target isosurface (order == \
             {}, expected ~{target_order})",
            field.order(point)
        );
        assert!(
            point.x.abs() < 0.5,
            "expected the lateral projection to move x toward 0 (the true isosurface location), \
             but it stayed near the starting x == 5.0: {point:?}"
        );
    }

    #[test]
    fn clamp_cell_size_to_node_budget_leaves_small_requests_unchanged() {
        // A tiny part at a normal resolution is well under budget, so the
        // requested cell size should come back exactly as given.
        let extent = DVec3::splat(5.0);
        let requested = 0.1;
        assert_eq!(clamp_cell_size_to_node_budget(extent, requested), requested);
    }

    #[test]
    fn clamp_cell_size_to_node_budget_coarsens_a_large_part_within_budget() {
        // Mirrors the reported OOM: an 85mm-tall part at the default
        // resolution-driven cell size (0.05mm) would request a ~1700^3
        // node grid. The clamped cell size must bring the estimated node
        // count back within `MAX_EIKONAL_GRID_NODES`.
        let extent = DVec3::splat(85.0);
        let requested = 0.05;
        let clamped = clamp_cell_size_to_node_budget(extent, requested);
        assert!(clamped > requested);

        let dims_per_axis = (extent.x / clamped + 1.0).ceil();
        let nodes = dims_per_axis.powi(3);
        assert!(nodes <= MAX_EIKONAL_GRID_NODES);
    }

    #[test]
    fn clamp_cell_size_to_node_budget_passes_through_non_finite_or_non_positive_input() {
        let extent = DVec3::splat(85.0);
        assert_eq!(clamp_cell_size_to_node_budget(extent, 0.0), 0.0);
        assert_eq!(clamp_cell_size_to_node_budget(extent, -1.0), -1.0);
        assert!(clamp_cell_size_to_node_budget(extent, f64::NAN).is_nan());
        assert_eq!(
            clamp_cell_size_to_node_budget(extent, f64::INFINITY),
            f64::INFINITY
        );
    }
}
