//! SDF/angle-field backend for Manifold, originally prototyped as a
//! research spike around the [`fidget`] crate (see `NON_PLANAR_SLICING.md`)
//! and now depended on by `manifold-gui` for isosurface/slice visualization
//! (see `MESH_SDF_VISUALIZATION.md`).
//!
//! This crate is still independent of `manifold-core`'s domain logic (no
//! slicing/toolpath/Gcode code here) — it provides scalar-field primitives
//! (`fidget`-`Tree`-backed fields and mesh-derived SDFs) and the
//! angle-field primitive `angle(p) = angle_between(normalize(grad f(p)), v)`
//! that both the slicing research and the GUI visualization tooling build
//! on.

use fidget::{context::Tree, shape::EzShape, types::Grad, vm::VmShape};
use glam::DVec3;
pub mod contour;
pub mod geometry;
pub mod marching_cubes;
pub mod mesh_sdf;
pub mod order;
pub mod slice;

/// Builds a `fidget::context::Tree` for a sphere of the given `radius`
/// centered at the origin: `f(p) = |p| - radius`.
pub fn sphere_tree(radius: f64) -> Tree {
    let x = Tree::x();
    let y = Tree::y();
    let z = Tree::z();
    (x.square() + y.square() + z.square()).sqrt() - radius
}

/// Value and gradient of a scalar field at a single point, as produced by a
/// `fidget` grad-slice evaluation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FieldSample {
    pub value: f64,
    pub gradient: DVec3,
}

/// A scalar field over 3D space that can be sampled for value and gradient
/// at a point, e.g. `f(p) = |p| - radius` for a sphere SDF.
pub trait ScalarField {
    /// Samples the field at `p`, returning its value and gradient.
    fn sample(&self, p: DVec3) -> FieldSample;
}

/// A [`ScalarField`] backed by a `fidget::context::Tree` expression.
pub struct TreeField {
    tree: Tree,
}

impl TreeField {
    /// Wraps `tree` as a [`ScalarField`].
    pub fn new(tree: Tree) -> Self {
        Self { tree }
    }
}

impl ScalarField for TreeField {
    fn sample(&self, p: DVec3) -> FieldSample {
        let shape = VmShape::from(self.tree.clone());
        let mut eval = VmShape::new_grad_slice_eval();
        let tape = shape.ez_grad_slice_tape();
        let out = eval
            .eval(
                &tape,
                &[Grad::new(p.x as f32, 1.0, 0.0, 0.0)],
                &[Grad::new(p.y as f32, 0.0, 1.0, 0.0)],
                &[Grad::new(p.z as f32, 0.0, 0.0, 1.0)],
            )
            .expect("grad-slice evaluation on a single point should not fail");
        let g = out[0];
        FieldSample {
            value: g.v as f64,
            gradient: DVec3::new(g.dx as f64, g.dy as f64, g.dz as f64),
        }
    }
}

/// Evaluates `tree` at `p`, returning its value and gradient (`grad f(p)`).
///
/// This is the core capability the spike doc's angle-field primitive needs:
/// `angle(p) = angle_between(normalize(grad f(p)), v)`.
///
/// Thin wrapper over [`TreeField`]/[`ScalarField::sample`]; kept as a free
/// function for existing callers.
pub fn evaluate(tree: &Tree, p: DVec3) -> FieldSample {
    TreeField::new(tree.clone()).sample(p)
}

/// The angle-field primitive from `NON_PLANAR_SLICING.md`: the angle (in
/// radians) between the surface normal at `p` (`normalize(grad f(p))`) and a
/// reference vector `v` (e.g. the gravity direction).
///
/// Returns `None` if the gradient at `p` is degenerate (zero-length, e.g.
/// far from the surface for some field shapes) and cannot be normalized.
pub fn angle_field(tree: &Tree, p: DVec3, v: DVec3) -> Option<f64> {
    let sample = evaluate(tree, p);
    if sample.gradient.length_squared() <= f64::EPSILON {
        return None;
    }
    let normal = sample.gradient.normalize();
    Some(normal.angle_between(v))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_2, PI};

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn sphere_value_matches_distance_to_surface() {
        let tree = sphere_tree(1.0);

        // On the surface: distance should be ~0.
        let on_surface = evaluate(&tree, DVec3::new(1.0, 0.0, 0.0));
        assert!(approx_eq(on_surface.value, 0.0, 1e-4));

        // Outside, along +X: distance should be ~1 (2 - radius 1).
        let outside = evaluate(&tree, DVec3::new(2.0, 0.0, 0.0));
        assert!(approx_eq(outside.value, 1.0, 1e-4));

        // Inside, at the center: distance should be ~-1 (0 - radius 1).
        let inside = evaluate(&tree, DVec3::new(0.0, 0.0, 0.0));
        assert!(approx_eq(inside.value, -1.0, 1e-4));
    }

    #[test]
    fn sphere_gradient_is_outward_normal() {
        let tree = sphere_tree(1.0);

        // At (1, 0, 0) the outward normal of a sphere centered at the
        // origin is exactly +X.
        let sample = evaluate(&tree, DVec3::new(1.0, 0.0, 0.0));
        let normal = sample.gradient.normalize();
        assert!(approx_eq(normal.x, 1.0, 1e-4));
        assert!(approx_eq(normal.y, 0.0, 1e-4));
        assert!(approx_eq(normal.z, 0.0, 1e-4));
    }

    #[test]
    fn angle_field_identifies_top_as_facing_up() {
        let tree = sphere_tree(1.0);
        let down = DVec3::new(0.0, 0.0, -1.0);

        // Top of the sphere: normal points straight up (+Z), so the angle
        // between the normal and "down" is pi (180 degrees) — this point
        // faces directly away from gravity, i.e. it is the most
        // overhang-free point on the sphere.
        let top_angle = angle_field(&tree, DVec3::new(0.0, 0.0, 1.0), down).unwrap();
        assert!(approx_eq(top_angle, PI, 1e-3));
    }

    #[test]
    fn angle_field_identifies_bottom_as_worst_overhang() {
        let tree = sphere_tree(1.0);
        let down = DVec3::new(0.0, 0.0, -1.0);

        // Bottom of the sphere: normal points straight down, same
        // direction as `v` -> angle ~0. In the overhang framing from
        // NON_PLANAR_SLICING.md, angle near 0 (surface normal aligned
        // with the print/gravity direction) marks the most
        // overhang-prone point.
        let bottom_angle = angle_field(&tree, DVec3::new(0.0, 0.0, -1.0), down).unwrap();
        assert!(approx_eq(bottom_angle, 0.0, 1e-3));
    }

    #[test]
    fn tree_field_sample_matches_free_function_evaluate() {
        let tree = sphere_tree(1.0);
        let field = TreeField::new(tree.clone());

        let p = DVec3::new(1.0, 0.0, 0.0);
        let via_trait = field.sample(p);
        let via_free_fn = evaluate(&tree, p);
        assert_eq!(via_trait, via_free_fn);
    }

    #[test]
    fn tree_field_gradient_is_outward_normal() {
        let tree = sphere_tree(1.0);
        let field = TreeField::new(tree);

        let sample = field.sample(DVec3::new(0.0, 1.0, 0.0));
        let normal = sample.gradient.normalize();
        assert!(approx_eq(normal.x, 0.0, 1e-4));
        assert!(approx_eq(normal.y, 1.0, 1e-4));
        assert!(approx_eq(normal.z, 0.0, 1e-4));
    }

    #[test]
    fn angle_field_equator_is_perpendicular_to_gravity() {
        let tree = sphere_tree(1.0);
        let down = DVec3::new(0.0, 0.0, -1.0);

        // Equator: normal is horizontal, perpendicular to "down" -> pi/2.
        let equator_angle = angle_field(&tree, DVec3::new(1.0, 0.0, 0.0), down).unwrap();
        assert!(approx_eq(equator_angle, FRAC_PI_2, 1e-3));
    }
}
