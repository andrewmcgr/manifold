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
/// each `(u, v)` point's true height along `axis` by solving
/// `order(p) = along - slope * radial` for `along`, where
/// `radial = sqrt(u*u + v*v)` is `(u, v)`'s distance from the axis line
/// (exact when `origin` is `apex` and `basis1`/`basis2` are the orthonormal
/// perpendicular-to-`axis` basis from `plane_basis(axis)` — see
/// `ConicalOrderField`'s doc comment for the same `radial` identity).
/// `slope == 0.0` (the `Height` case, see [`resolve_axis_apex_slope`])
/// degenerates `along` to exactly `order` for every point, matching
/// `polygon2d::from_2d`'s flat reconstruction exactly.
///
/// This must be used instead of `polygon2d::from_2d` whenever `origin` is
/// a genuinely curved field's apex — `from_2d` assumes one flat height for
/// every point, which is wrong for `Conical` wherever `radial` varies
/// across the loop (i.e. essentially always, since a cone's radius varies
/// continuously along its surface).
pub fn reconstruct_on_order_field(
    contours: Vec<Vec<[f64; 2]>>,
    basis1: DVec3,
    basis2: DVec3,
    axis: DVec3,
    apex: DVec3,
    order: f64,
    slope: f64,
) -> Vec<Vec<DVec3>> {
    contours
        .into_iter()
        .map(|contour| {
            contour
                .into_iter()
                .map(|[u, v]| {
                    let radial = (u * u + v * v).sqrt();
                    let along = order + slope * radial;
                    apex + basis1 * u + basis2 * v + axis * along
                })
                .collect()
        })
        .collect()
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
}
