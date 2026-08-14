//! The `order` field primitive: a scalar field over 3D space whose
//! isosurfaces define the slicing sequence (see `NON_PLANAR_SLICING.md`).
//!
//! This module reuses the [`ScalarField`](crate::ScalarField)-style shape
//! (`order(p) -> f64`) rather than inventing a parallel trait — an order
//! field is conceptually just a scalar field whose isosurfaces are walked
//! in increasing order to produce slice layers, so no gradient is required
//! (unlike [`ScalarField::sample`](crate::ScalarField::sample), which also
//! returns a gradient for the angle-field use case).

use glam::DVec3;

/// A scalar field over 3D space whose isosurfaces (`order(p) == c` for a
/// sequence of increasing `c`) define the slicing order — the simplest
/// instance, [`HeightOrderField`], reduces this to conventional planar
/// slicing along a build direction.
pub trait OrderField {
    /// Evaluates the order field at `p`.
    fn order(&self, p: DVec3) -> f64;
}

/// The simplest [`OrderField`]: a plain height field along `direction`,
/// `order(p) = p.dot(direction)`. Its isosurfaces are flat planes
/// perpendicular to `direction`, i.e. this reduces to conventional planar
/// slicing.
///
/// `direction` is expected to be a unit vector; this is a documented
/// precondition, not defensively enforced (`order` does not normalize
/// `direction` internally), so a non-unit-length `direction` will produce
/// values scaled by its magnitude rather than a panic or silent
/// normalization.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeightOrderField {
    pub direction: DVec3,
}

impl HeightOrderField {
    /// Builds a [`HeightOrderField`] from `direction`, which must be a unit
    /// vector (see the type-level precondition note).
    pub fn new(direction: DVec3) -> Self {
        Self { direction }
    }
}

impl OrderField for HeightOrderField {
    fn order(&self, p: DVec3) -> f64 {
        p.dot(self.direction)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn order_is_monotonic_along_direction() {
        let field = HeightOrderField::new(DVec3::new(0.0, 0.0, 1.0));

        let samples: Vec<f64> = [-2.0, -1.0, 0.0, 0.5, 1.0, 3.0]
            .iter()
            .map(|&z| field.order(DVec3::new(0.0, 0.0, z)))
            .collect();

        for window in samples.windows(2) {
            assert!(
                window[1] > window[0],
                "expected strictly increasing order values"
            );
        }
    }

    #[test]
    fn order_matches_dot_product_for_non_axis_direction() {
        let direction = DVec3::new(1.0, 1.0, 1.0).normalize();
        let field = HeightOrderField::new(direction);

        let p = DVec3::new(2.0, 3.0, 4.0);
        assert!(approx_eq(field.order(p), p.dot(direction), 1e-9));
    }

    #[test]
    fn displacement_orthogonal_to_direction_does_not_change_value() {
        let direction = DVec3::new(0.0, 0.0, 1.0);
        let field = HeightOrderField::new(direction);

        let base = DVec3::new(0.0, 0.0, 2.0);
        let base_order = field.order(base);

        // Displacements purely in X/Y (orthogonal to +Z) must not change
        // the order value.
        for offset in [
            DVec3::new(5.0, 0.0, 0.0),
            DVec3::new(0.0, -3.0, 0.0),
            DVec3::new(2.5, -7.0, 0.0),
        ] {
            let displaced = field.order(base + offset);
            assert!(approx_eq(displaced, base_order, 1e-9));
        }
    }

    #[test]
    fn displacement_orthogonal_to_arbitrary_direction_does_not_change_value() {
        let direction = DVec3::new(1.0, 1.0, 0.0).normalize();
        let field = HeightOrderField::new(direction);

        let base = DVec3::new(1.0, -1.0, 3.0);
        let base_order = field.order(base);

        // (1, -1, 0) is orthogonal to (1, 1, 0); Z displacement is also
        // orthogonal to a direction with zero Z component.
        for offset in [
            DVec3::new(4.0, -4.0, 0.0),
            DVec3::new(0.0, 0.0, 10.0),
            DVec3::new(2.0, -2.0, -6.0),
        ] {
            let displaced = field.order(base + offset);
            assert!(approx_eq(displaced, base_order, 1e-9));
        }
    }
}
