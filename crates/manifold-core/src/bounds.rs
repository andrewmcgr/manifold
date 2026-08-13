//! Bounding volumes used for build-volume checks and (future) collision
//! avoidance between simultaneously-printing tools.
//!
//! TODO(roadmap): Deferred work (see ROADMAP.md) — multi-object collision
//! avoidance will check a tool's [`BoundingVolume`] (via
//! [`crate::tool::Tool::collision_envelope`]) against already-printed
//! objects to safely order/interleave simultaneous multi-object printing.
//! Not implemented yet.

use glam::DVec3;

/// A simple bounding volume, expressed in the local frame of whatever it
/// is attached to (e.g. a tool's mount transform, or a machine's
/// substrate transform).
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind")]
pub enum BoundingVolume {
    /// Axis-aligned box, given by opposite corners.
    Aabb { min: DVec3, max: DVec3 },
    /// Sphere given by a center and radius.
    Sphere { center: DVec3, radius: f64 },
}

impl BoundingVolume {
    /// Whether `point` lies within this volume.
    pub fn contains(&self, point: DVec3) -> bool {
        match *self {
            BoundingVolume::Aabb { min, max } => point.cmpge(min).all() && point.cmple(max).all(),
            BoundingVolume::Sphere { center, radius } => center.distance(point) <= radius,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aabb_contains_point_inside_bounds() {
        let volume = BoundingVolume::Aabb {
            min: DVec3::new(-1.0, -1.0, -1.0),
            max: DVec3::new(1.0, 1.0, 1.0),
        };
        assert!(volume.contains(DVec3::ZERO));
        assert!(!volume.contains(DVec3::new(2.0, 0.0, 0.0)));
    }

    #[test]
    fn sphere_contains_point_within_radius() {
        let volume = BoundingVolume::Sphere {
            center: DVec3::ZERO,
            radius: 1.0,
        };
        assert!(volume.contains(DVec3::new(0.5, 0.0, 0.0)));
        assert!(!volume.contains(DVec3::new(1.5, 0.0, 0.0)));
    }
}
