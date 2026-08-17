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
use manifold_fidget::ScalarField;

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
) -> Box<dyn OrderField> {
    match kind {
        OrderFieldKind::Height => Box::new(HeightOrderField::new(BUILD_DIRECTION)),
        OrderFieldKind::Conical => Box::new(ConicalOrderField::new(
            config.order_field_apex,
            config.order_field_axis,
            config.order_field_slope,
        )),
        OrderFieldKind::Eikonal => Box::new(eikonal_field_for(config, mesh)),
    }
}

/// Builds the [`OrderFieldKind::Eikonal`] field for `mesh`: seeds the FMM
/// front from `mesh`'s base/contact surface with the build plate (vertices
/// at or within a small tolerance of the mesh's minimum Z -- the "rests on
/// floor" convention already used by `object::center_on_bed` and
/// `manifold-gui`'s `scene::build_bed_quad`, both of which treat `min.z` as
/// the build plate's height), and derives grid resolution from `mesh`'s
/// bounding box and the finer of `config.layer_height`/`config.nozzle_diameter`
/// (reusing those existing resolution-driving config fields rather than
/// adding a new one). Using `layer_height` alone (0.2mm by default) produced
/// a grid too coarse to resolve fine surface detail (e.g. helical/screw
/// grooves) -- trilinear interpolation of an under-resolved distance field
/// visibly facets the reconstructed contours -- so the cell size is halved
/// again on top of the finer of the two dimensions for extra headroom. That
/// resolution-driven cell size is then passed through
/// [`clamp_cell_size_to_node_budget`], which coarsens it (grows it) as
/// needed so the dense grid's total node count never exceeds
/// [`MAX_EIKONAL_GRID_NODES`] -- without this, a large part sliced at a
/// fine layer height/nozzle diameter (e.g. an 85mm-tall part at the 0.2mm
/// default layer height, cell size 0.05mm) requests a `~1700^3` node grid,
/// whose `Vec<f64>` distance buffer alone is tens of gigabytes and OOMs the
/// process; the budget trades local resolution for a bounded, predictable
/// memory footprint on large parts.
///
/// The march is occupancy-aware (see
/// [`manifold_fidget::eikonal::EikonalOrderField::new_with_occupancy`]):
/// a grid node counts as solid when a fresh `MeshSdf` built from `mesh`
/// reports it inside the mesh, or within `cell_size` of the surface (a
/// tolerance covering the discretization gap between the grid and the true
/// boundary -- without it, nodes exactly on a thin wall's surface could be
/// spuriously excluded, breaking the march right where it matters most).
/// This keeps the front from taking a straight-line shortcut through open
/// air (e.g. across a gap between two close walls, or up the outside of a
/// wall toward a nearby build-plate seed) -- without it, `order` reduces to
/// plain Euclidean distance and can produce isosurfaces that climb outside
/// the object along a path no real toolpath could follow.
///
/// An empty mesh (no vertices, so no bounding box) is a documented
/// best-effort fallback -- a degenerate unit-box field with no seeds, whose
/// `order()` always returns `f64::INFINITY` (see
/// [`manifold_fidget::eikonal::EikonalOrderField`]'s own documented
/// behavior for an empty seed set) -- rather than a panic.
fn eikonal_field_for(config: &SlicerConfig, mesh: &Mesh) -> EikonalOrderField {
    let Some((min, max)) = mesh.bounding_box() else {
        return EikonalOrderField::new(DVec3::ZERO, DVec3::ONE, &[], 1.0);
    };

    let layer_height = config.layer_height.abs().max(f64::EPSILON);
    let nozzle_diameter = config.nozzle_diameter.abs().max(f64::EPSILON);
    let requested_cell_size = layer_height.min(nozzle_diameter) / 2.0;
    let cell_size = clamp_cell_size_to_node_budget(max - min, requested_cell_size);
    // Contact-surface tolerance: within one layer height of the mesh's
    // minimum Z counts as "touching" the build plate, generous enough to
    // include a slightly-faceted/non-flat mesh base without pulling in
    // vertices from well above the true contact surface.
    let seeds: Vec<DVec3> = mesh
        .vertices
        .iter()
        .copied()
        .filter(|v| v.z <= min.z + layer_height)
        .collect();

    let faces: Vec<[usize; 3]> = mesh
        .indices
        .chunks_exact(3)
        .map(|chunk| [chunk[0] as usize, chunk[1] as usize, chunk[2] as usize])
        .collect();
    if faces.is_empty() {
        // No triangles to classify solid/void against: fall back to the
        // unconstrained march rather than treating every node as void
        // (which would leave the whole grid unreachable).
        return EikonalOrderField::new(min, max, &seeds, cell_size);
    }
    let sdf = MeshSdf::new(mesh.vertices.clone(), faces);
    let is_solid = |p: DVec3| sdf.sample(p).value <= cell_size;

    EikonalOrderField::new_with_occupancy(min, max, &seeds, cell_size, &is_solid)
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
                    reconstruct_point_on_order_field(planar, axis, target_order, max_along, field)
                })
                .collect()
        })
        .collect()
}

/// Single-point building block behind [`reconstruct_on_order_field`]: given
/// a transverse (perpendicular-to-`axis`) reference position `planar`,
/// solves for the `axis`-offset that lands on `field`'s `target_order`
/// isosurface and returns the resulting 3D world point.
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
) -> DVec3 {
    let along = solve_along(field, planar, axis, target_order, max_along);
    planar + axis * along
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
/// simply isn't reachable by a vertical search), this returns the
/// *closest-to-target* sample observed anywhere during the search
/// (including the initial probes) rather than the outer edge of the
/// failed bracket. Previously this fell through to the last bracket's
/// midpoint, which for a failed search sits right at the `max_along`
/// bound — up to 50 layer heights away — producing exactly the multi-mm
/// vertical spikes seen on screw-thread geometry with an `Eikonal` order
/// field; "closest real sample we actually saw" is a far tighter,
/// caller-visible "best effort" position for a pathological/misbehaving
/// field than "as far as the search was allowed to wander."
fn solve_along<F: OrderField + ?Sized>(
    field: &F,
    planar: DVec3,
    axis: DVec3,
    target: f64,
    max_along: f64,
) -> f64 {
    const TOLERANCE: f64 = 1e-9;
    const MAX_BISECT_ITERS: u32 = 64;

    let max_along = max_along.abs();
    let residual = |along: f64| field.order(planar + axis * along) - target;

    // Best (lowest |residual|) sample seen anywhere during the search --
    // the fallback returned if bracketing never finds a real sign change
    // (see this function's doc).
    let mut best_along = 0.0_f64;
    let mut best_residual = residual(0.0).abs();
    let mut consider = |along: f64, r: f64| {
        if r.is_finite() && r.abs() < best_residual {
            best_residual = r.abs();
            best_along = along;
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
        return lo;
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
        // `Eikonal` on reentrant/threaded geometry). Fall back to the
        // closest-to-target sample actually observed, not the outer edge
        // of the failed bracket (see this function's doc).
        return best_along;
    }

    f_lo = residual(lo);
    for _ in 0..MAX_BISECT_ITERS {
        let mid = 0.5 * (lo + hi);
        let f_mid = residual(mid);
        if f_mid.abs() <= TOLERANCE {
            return mid;
        }
        if (f_mid < 0.0) == (f_lo < 0.0) {
            lo = mid;
            f_lo = f_mid;
        } else {
            hi = mid;
        }
    }
    (0.5 * (lo + hi)).clamp(min_bound, max_bound)
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::DVec3;

    #[test]
    fn default_config_resolves_to_height_field_matching_build_direction() {
        let config = SlicerConfig::default();
        let mesh = crate::mesh::Mesh::default();
        let field = order_field_for(config.order_field, &config, &mesh);

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
        let field = order_field_for(config.order_field, &config, &mesh);

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

        let point = reconstruct_point_on_order_field(planar, axis, 0.0, max_along, &field);

        assert!(
            (point - planar).length() < 1e-6,
            "expected the closest-observed fallback right at `planar` (along == 0.0), got {point:?} \
             (drifted {} units away)",
            (point - planar).length()
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
