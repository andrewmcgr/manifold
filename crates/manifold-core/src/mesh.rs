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
}
