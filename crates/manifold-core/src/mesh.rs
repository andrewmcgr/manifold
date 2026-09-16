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

/// Triangle indices of `mesh` excluding downward-facing bed-contact
/// triangles resting on the build plate (any vertex at `z <= min_z + 0.02`)
/// -- the same "keep every top ceiling, roof, and side wall; drop only the
/// literal bed floor" exclusion `slicing::slice_mesh_with_progress` already
/// applies when building its own bed-exclusion SDF for wall/infill
/// boundary extraction, extracted here so both that use and
/// `order_field::order_field_for_with_sdf`'s directional top-surface
/// distance can share the same filter logic instead of duplicating it.
/// Returns the full unfiltered face list when the mesh has no bed-contact
/// faces at all (i.e. `min_z` isn't actually touched by any downward face
/// -- a floating or non-flat-bottomed mesh), matching the "no exclusion
/// needed" case callers already handle by reusing their original SDF.
pub(crate) fn non_bed_floor_faces(mesh: &Mesh, min_z: f64) -> Vec<[usize; 3]> {
    mesh.indices
        .chunks_exact(3)
        .filter_map(|chunk| {
            let [i0, i1, i2] = [chunk[0] as usize, chunk[1] as usize, chunk[2] as usize];
            let v0 = mesh.vertices[i0];
            let v1 = mesh.vertices[i1];
            let v2 = mesh.vertices[i2];
            let normal = (v1 - v0).cross(v2 - v0);
            let normal_len_sq = normal.length_squared();
            if normal_len_sq > 1e-12
                && normal.z < 0.0
                && (v0.z <= min_z + 0.02 || v1.z <= min_z + 0.02 || v2.z <= min_z + 0.02)
            {
                let nz_sq = normal.z * normal.z;
                if nz_sq >= 0.998 * normal_len_sq {
                    return None;
                }
            }
            Some([i0, i1, i2])
        })
        .collect()
}

#[cfg(test)]
mod bed_floor_face_tests {
    use super::*;

    #[test]
    fn non_bed_floor_faces_excludes_only_the_downward_bed_contact_cap() {
        // A unit cube: bottom cap at z=0 (must be excluded), everything
        // else (top cap, 4 side walls) must survive.
        let vertices = vec![
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(1.0, 1.0, 0.0),
            DVec3::new(0.0, 1.0, 0.0),
            DVec3::new(0.0, 0.0, 1.0),
            DVec3::new(1.0, 0.0, 1.0),
            DVec3::new(1.0, 1.0, 1.0),
            DVec3::new(0.0, 1.0, 1.0),
        ];
        let indices = vec![
            0, 2, 1, 0, 3, 2, // -Z (bed contact, downward)
            4, 5, 6, 4, 6, 7, // +Z (top, upward)
            0, 1, 5, 0, 5, 4, // -Y
            3, 7, 6, 3, 6, 2, // +Y
            0, 4, 7, 0, 7, 3, // -X
            1, 2, 6, 1, 6, 5, // +X
        ];
        let mesh = Mesh::new(vertices, indices);
        let faces = non_bed_floor_faces(&mesh, 0.0);
        // 12 total triangles minus the 2 bottom-cap triangles = 10.
        assert_eq!(
            faces.len(),
            10,
            "expected exactly the 2 bottom-cap triangles excluded"
        );
    }
}

/// Returns the 8 corner vertices of the axis-aligned bounding box defined by `min` and `max`.
#[must_use]
pub fn bounding_box_corners(min: DVec3, max: DVec3) -> [DVec3; 8] {
    [
        DVec3::new(min.x, min.y, min.z),
        DVec3::new(max.x, min.y, min.z),
        DVec3::new(min.x, max.y, min.z),
        DVec3::new(max.x, max.y, min.z),
        DVec3::new(min.x, min.y, max.z),
        DVec3::new(max.x, min.y, max.z),
        DVec3::new(min.x, max.y, max.z),
        DVec3::new(max.x, max.y, max.z),
    ]
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
