//! Mesh representation and loading.
//!
//! Format-specific loaders live in their own modules: [`crate::threemf`]
//! (via `lib3mf`) for 3MF, since it natively models multiple build items
//! with transforms and materials and so populates `Object`s (Phase 0)
//! directly rather than a bare `Mesh`; [`crate::stl`] (via `stl_io`) for
//! STL, which only ever describes a single triangle [`Mesh`].

use glam::DVec3;

/// A triangle mesh in model space (millimeters).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Mesh {
    pub vertices: Vec<DVec3>,
    /// Triangle vertex indices, three per triangle.
    pub indices: Vec<u32>,
}

impl Mesh {
    pub fn new(vertices: Vec<DVec3>, indices: Vec<u32>) -> Self {
        Self { vertices, indices }
    }

    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// The axis-aligned bounding box (min, max corners) enclosing every
    /// vertex, in local mesh space. `None` for an empty mesh.
    pub fn bounding_box(&self) -> Option<(DVec3, DVec3)> {
        let mut vertices = self.vertices.iter();
        let first = *vertices.next()?;
        let (min, max) = vertices.fold((first, first), |(min, max), &vertex| {
            (min.min(vertex), max.max(vertex))
        });
        Some((min, max))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounding_box_is_none_for_empty_mesh() {
        assert_eq!(Mesh::default().bounding_box(), None);
    }

    #[test]
    fn bounding_box_encloses_all_vertices() {
        let mesh = Mesh::new(
            vec![
                DVec3::new(-1.0, 2.0, 0.0),
                DVec3::new(3.0, -2.0, 1.0),
                DVec3::new(0.0, 0.0, -5.0),
            ],
            vec![0, 1, 2],
        );
        assert_eq!(
            mesh.bounding_box(),
            Some((DVec3::new(-1.0, -2.0, -5.0), DVec3::new(3.0, 2.0, 1.0)))
        );
    }
}
