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
