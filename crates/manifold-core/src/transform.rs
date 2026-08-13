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
}
