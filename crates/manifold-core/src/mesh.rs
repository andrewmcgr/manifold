//! Mesh representation and loading.
//!
//! 3MF loading lives in [`crate::threemf`] (via `lib3mf`), since 3MF
//! natively models multiple build items with transforms and materials and
//! so populates `Object`s (Phase 0) directly rather than a bare `Mesh`.
//!
//! TODO(roadmap): Phase 1 (see ROADMAP.md) — add STL loading via `stl_io`.

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
