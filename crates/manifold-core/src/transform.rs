//! Position + orientation + scale transforms shared by [`crate::tool::Tool`]
//! mounting, [`crate::object::Object`] placement, and
//! [`crate::machine::Machine`] substrate orientation.
//!
//! Modeled as a full affine transform (not a Z-offset scalar or a fixed
//! Z=0 plane) so the deferred multi-axis (tool-tilting / substrate
//! reorientation) work does not force a schema refactor later — see
//! ROADMAP.md.

use glam::{DAffine3, DQuat, DVec3};

/// A position + orientation + scale transform, in millimeters.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Transform(pub DAffine3);

impl Transform {
    /// The identity transform (no translation, rotation, or scale).
    pub fn identity() -> Self {
        Self(DAffine3::IDENTITY)
    }

    /// A transform with only a translation applied.
    pub fn from_translation(translation: DVec3) -> Self {
        Self(DAffine3::from_translation(translation))
    }

    /// Construct from independent scale, rotation, and translation
    /// components.
    pub fn from_scale_rotation_translation(
        scale: DVec3,
        rotation: DQuat,
        translation: DVec3,
    ) -> Self {
        Self(DAffine3::from_scale_rotation_translation(
            scale,
            rotation,
            translation,
        ))
    }

    /// Transform a point from local space into the parent frame.
    pub fn transform_point(&self, point: DVec3) -> DVec3 {
        self.0.transform_point3(point)
    }

    /// Compose this transform with an additional world-space translation
    /// applied afterwards — i.e. `result.transform_point(p) ==
    /// self.transform_point(p) + offset`. Used to re-center already-placed
    /// objects (e.g. a 3MF assembly's build-item transforms) without
    /// disturbing their relative arrangement.
    pub fn then_translate(&self, offset: DVec3) -> Self {
        Self(DAffine3::from_translation(offset) * self.0)
    }

    /// The angle (radians) this transform's rotation carries `basis1`
    /// through, measured within the plane spanned by the orthonormal pair
    /// `basis1`/`basis2` (e.g. `manifold_fidget::contour::plane_basis`'s
    /// output).
    ///
    /// Used by `infill` so fill-line angle tracks object orientation:
    /// rotating an object about the slicing build axis rotates its infill
    /// lines by the same amount, without infill needing to un-rotate
    /// already-world-space wall-loop geometry. Ignores translation
    /// (irrelevant to a direction) and any rotation component that tilts
    /// `basis1` out of the `basis1`/`basis2` plane (out of scope for the
    /// current planar-slicing MVP — see ROADMAP.md's deferred non-planar
    /// order field work) and normalizes away scale.
    pub fn in_plane_rotation_angle(&self, basis1: DVec3, basis2: DVec3) -> f64 {
        let rotated = self.0.transform_vector3(basis1).normalize_or_zero();
        rotated.dot(basis2).atan2(rotated.dot(basis1))
    }
}

impl Default for Transform {
    fn default() -> Self {
        Self::identity()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_transform_leaves_point_unchanged() {
        let point = DVec3::new(1.0, 2.0, 3.0);
        assert_eq!(Transform::identity().transform_point(point), point);
    }

    #[test]
    fn translation_transform_offsets_point() {
        let transform = Transform::from_translation(DVec3::new(1.0, 0.0, 0.0));
        assert_eq!(
            transform.transform_point(DVec3::ZERO),
            DVec3::new(1.0, 0.0, 0.0)
        );
    }

    #[test]
    fn then_translate_offsets_result_in_world_space() {
        let transform = Transform::from_translation(DVec3::new(1.0, 0.0, 0.0))
            .then_translate(DVec3::new(0.0, 2.0, 0.0));
        assert_eq!(
            transform.transform_point(DVec3::ZERO),
            DVec3::new(1.0, 2.0, 0.0)
        );
    }

    #[test]
    fn in_plane_rotation_angle_is_zero_for_identity() {
        let transform = Transform::identity();
        let angle = transform.in_plane_rotation_angle(DVec3::X, DVec3::Y);
        assert!(angle.abs() < 1e-9, "expected ~0, got {angle}");
    }

    #[test]
    fn in_plane_rotation_angle_matches_a_quarter_turn_about_the_plane_normal() {
        let transform = Transform::from_scale_rotation_translation(
            DVec3::ONE,
            DQuat::from_axis_angle(DVec3::Z, std::f64::consts::FRAC_PI_2),
            DVec3::ZERO,
        );
        let angle = transform.in_plane_rotation_angle(DVec3::X, DVec3::Y);
        assert!(
            (angle - std::f64::consts::FRAC_PI_2).abs() < 1e-9,
            "expected ~pi/2, got {angle}"
        );
    }

    #[test]
    fn in_plane_rotation_angle_ignores_scale() {
        let transform = Transform::from_scale_rotation_translation(
            DVec3::new(3.0, 5.0, 1.0),
            DQuat::from_axis_angle(DVec3::Z, std::f64::consts::FRAC_PI_4),
            DVec3::new(10.0, -4.0, 0.0),
        );
        let angle = transform.in_plane_rotation_angle(DVec3::X, DVec3::Y);
        assert!(
            (angle - std::f64::consts::FRAC_PI_4).abs() < 1e-9,
            "expected ~pi/4, got {angle}"
        );
    }
}
