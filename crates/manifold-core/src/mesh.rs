//! Mesh representation and loading.

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
