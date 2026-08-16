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

/// A genuinely curved [`OrderField`]: isosurfaces are cones nested around
/// `axis`, apex at `apex`, opening in the `+axis` direction.
/// `order(p) = along - slope * radial`, where `along = (p - apex).dot(axis)`
/// is the height along `axis` and `radial = ((p - apex) - along * axis).length()`
/// is the distance from `p` to the axis line. Isosurfaces `order(p) == c` are
/// cones: as `radial` grows, `along` must grow proportionally (by `1 /
/// slope`) to hold `order` constant, so this is not a plane for any nonzero
/// `slope` — unlike [`HeightOrderField`], whose isosurfaces are always flat.
///
/// `slope` controls how steeply the cone opens: `slope == 0.0` degenerates
/// exactly to a [`HeightOrderField`] along `axis` (flat isosurfaces); larger
/// `slope` makes the cone narrower (steeper walls) for a given height range.
///
/// `axis` is expected to be a unit vector, same documented (not defensively
/// enforced) precondition as [`HeightOrderField::direction`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConicalOrderField {
    pub apex: DVec3,
    pub axis: DVec3,
    pub slope: f64,
}

impl ConicalOrderField {
    /// Builds a [`ConicalOrderField`] with apex at `apex`, opening along
    /// `axis` (must be a unit vector), with cone steepness `slope`.
    pub fn new(apex: DVec3, axis: DVec3, slope: f64) -> Self {
        Self { apex, axis, slope }
    }
}

impl OrderField for ConicalOrderField {
    fn order(&self, p: DVec3) -> f64 {
        let offset = p - self.apex;
        let along = offset.dot(self.axis);
        let radial = (offset - along * self.axis).length();
        along - self.slope * radial
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

    #[test]
    fn conical_order_field_degenerates_to_height_order_field_when_slope_is_zero() {
        let axis = DVec3::new(0.0, 0.0, 1.0);
        let cone = ConicalOrderField::new(DVec3::ZERO, axis, 0.0);
        let height = HeightOrderField::new(axis);

        for p in [
            DVec3::new(0.0, 0.0, 3.0),
            DVec3::new(5.0, -2.0, 1.0),
            DVec3::new(-1.0, 4.0, -2.0),
        ] {
            assert!(approx_eq(cone.order(p), height.order(p), 1e-9));
        }
    }

    #[test]
    fn conical_order_field_isosurface_is_not_flat() {
        // With nonzero slope, points at the same `along` height but
        // different radial distance from the axis must get different
        // order values -- i.e. the isosurface is not a flat plane
        // perpendicular to `axis` (unlike `HeightOrderField`).
        let axis = DVec3::new(0.0, 0.0, 1.0);
        let field = ConicalOrderField::new(DVec3::ZERO, axis, 1.0);

        let on_axis = field.order(DVec3::new(0.0, 0.0, 5.0));
        let off_axis = field.order(DVec3::new(3.0, 0.0, 5.0));

        assert!(
            (on_axis - off_axis).abs() > 1e-9,
            "expected radial displacement to change order value for nonzero slope"
        );
    }

    #[test]
    fn conical_order_field_isosurface_holds_order_constant_along_the_cone() {
        // For a fixed `order` value `c`, points satisfying
        // `along = c + slope * radial` for varying `radial` all lie on the
        // same isosurface -- i.e. actually trace out the cone.
        let axis = DVec3::new(0.0, 0.0, 1.0);
        let slope = 0.5;
        let field = ConicalOrderField::new(DVec3::ZERO, axis, slope);
        let target = 2.0;

        for radial in [0.0, 1.0, 4.0, 10.0] {
            let along = target + slope * radial;
            let p = DVec3::new(radial, 0.0, along);
            assert!(approx_eq(field.order(p), target, 1e-9));
        }
    }

    #[test]
    fn conical_order_field_is_monotonic_along_axis_at_fixed_radius() {
        let axis = DVec3::new(0.0, 1.0, 0.0);
        let field = ConicalOrderField::new(DVec3::new(1.0, 0.0, 0.0), axis, 0.3);

        let samples: Vec<f64> = [-2.0, -1.0, 0.0, 0.5, 1.0, 3.0]
            .iter()
            .map(|&t| field.order(DVec3::new(1.0, t, 2.0)))
            .collect();

        for window in samples.windows(2) {
            assert!(
                window[1] > window[0],
                "expected strictly increasing order values along axis at fixed radius"
            );
        }
    }

    #[test]
    fn conical_order_field_respects_apex_offset() {
        let apex = DVec3::new(1.0, 2.0, 3.0);
        let axis = DVec3::new(0.0, 0.0, 1.0);
        let field = ConicalOrderField::new(apex, axis, 0.0);

        // With slope 0.0 this reduces to a height field measured relative
        // to `apex`, i.e. `order(p) = (p - apex).dot(axis)`.
        let p = DVec3::new(1.0, 2.0, 7.0);
        assert!(approx_eq(field.order(p), 4.0, 1e-9));
    }
}
