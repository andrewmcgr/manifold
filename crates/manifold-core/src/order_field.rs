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
use manifold_fidget::order::{ConicalOrderField, HeightOrderField, OrderField};

use crate::{slicing::BUILD_DIRECTION, SlicerConfig};

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
}

/// Resolve a config-level [`OrderFieldKind`] to a concrete
/// `manifold_fidget::order::OrderField`, using `config`'s `apex`/`axis`/
/// `slope` fields for the [`OrderFieldKind::Conical`] case.
pub fn order_field_for(kind: OrderFieldKind, config: &SlicerConfig) -> Box<dyn OrderField> {
    match kind {
        OrderFieldKind::Height => Box::new(HeightOrderField::new(BUILD_DIRECTION)),
        OrderFieldKind::Conical => Box::new(ConicalOrderField::new(
            config.order_field_apex,
            config.order_field_axis,
            config.order_field_slope,
        )),
    }
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
/// is strictly monotonically increasing along `axis` at any fixed in-plane
/// position (true of `Height`/`Conical` by construction, and required of
/// any future field for slicing to make sense in the first place — see
/// `NON_PLANAR_SLICING.md`'s well-posedness discussion). Bracket expansion
/// assumes this monotonicity holds arbitrarily far from the initial guess;
/// it is capped (see [`solve_along`]) rather than looping forever if that
/// assumption is violated.
///
/// `basis1`/`basis2` must be the orthonormal in-plane basis perpendicular
/// to `axis` (e.g. from `manifold_fidget::contour::plane_basis(axis)`),
/// matching whatever produced `contours`' `(u, v)` coordinates via
/// `polygon2d::to_2d(.., basis1, basis2, apex)`.
pub fn reconstruct_on_order_field<F: OrderField + ?Sized>(
    contours: Vec<Vec<[f64; 2]>>,
    basis1: DVec3,
    basis2: DVec3,
    axis: DVec3,
    apex: DVec3,
    target_order: f64,
    field: &F,
) -> Vec<Vec<DVec3>> {
    contours
        .into_iter()
        .map(|contour| {
            contour
                .into_iter()
                .map(|[u, v]| {
                    let planar = apex + basis1 * u + basis2 * v;
                    let along = solve_along(field, planar, axis, target_order);
                    planar + axis * along
                })
                .collect()
        })
        .collect()
}

/// Solves `field.order(planar + axis * along) == target` for `along`,
/// assuming `field.order` is strictly increasing along `axis` from
/// `planar` (see [`reconstruct_on_order_field`]'s preconditions).
///
/// Expands a bracket `[lo, hi]` outward from `along == 0.0` by doubling
/// steps until the target sign is bracketed (capped at `MAX_EXPAND_STEPS`
/// doublings), then bisects (`MAX_BISECT_ITERS` iterations — comfortably
/// enough for `f64` precision on any bracket reached by the doubling
/// above). Returns the midpoint of the last bracket if the search never
/// finds a sign change or the residual never reaches tolerance, rather
/// than panicking — a caller-visible "best effort" position for a
/// pathological/misbehaving field instead of a hard failure.
fn solve_along<F: OrderField + ?Sized>(field: &F, planar: DVec3, axis: DVec3, target: f64) -> f64 {
    const TOLERANCE: f64 = 1e-9;
    const MAX_EXPAND_STEPS: u32 = 64;
    const MAX_BISECT_ITERS: u32 = 64;

    let residual = |along: f64| field.order(planar + axis * along) - target;

    let mut lo = 0.0_f64;
    let mut hi = 0.0_f64;
    let mut f_lo = residual(lo);
    if f_lo.abs() <= TOLERANCE {
        return lo;
    }

    let mut step = 1.0_f64;
    if f_lo < 0.0 {
        // Need a larger `along`; grow `hi` until the residual turns
        // non-negative.
        loop {
            hi += step;
            let f_hi = residual(hi);
            if f_hi >= 0.0 || step >= f64::from(MAX_EXPAND_STEPS).exp2() {
                break;
            }
            step *= 2.0;
        }
    } else {
        // Need a smaller `along`; shrink `lo` until the residual turns
        // non-positive.
        loop {
            lo -= step;
            let f_lo_candidate = residual(lo);
            if f_lo_candidate <= 0.0 || step >= f64::from(MAX_EXPAND_STEPS).exp2() {
                break;
            }
            step *= 2.0;
        }
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
    0.5 * (lo + hi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::DVec3;

    #[test]
    fn default_config_resolves_to_height_field_matching_build_direction() {
        let config = SlicerConfig::default();
        let field = order_field_for(config.order_field, &config);

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

        let field = order_field_for(config.order_field, &config);

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
        let reconstructed =
            reconstruct_on_order_field(contours, basis1, basis2, axis, apex, target_order, &field);

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
}
