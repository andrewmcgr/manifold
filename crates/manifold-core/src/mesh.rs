//! Mesh representation and loading.
//!
//! TODO(roadmap): Phase 1 (see ROADMAP.md) — add STL loading via `stl_io`
//! and 3MF loading via `lib3mf` (pending disambiguation from the unrelated
//! `lib3mf-core` crate family; see ROADMAP.md). The 3MF loader should
//! populate `Object`s (Phase 0) directly, since 3MF natively models
//! multiple build items with transforms and materials.

use glam::DVec3;

/// A triangle mesh in model space (millimeters).
#[derive(Debug, Clone, Default)]
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
