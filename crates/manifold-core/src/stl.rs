//! STL (STereoLithography) mesh loading, via the `stl_io` crate.
//!
//! TODO(roadmap): Phase 1 (see ROADMAP.md) — unlike 3MF, STL has no
//! concept of build items, transforms, or materials: a file is just one
//! triangle mesh. Wrapping the result into a [`crate::object::Object`]
//! (assigning an id/tool/transform) is left to the caller.

use std::io::{Read, Seek};

use glam::DVec3;

use crate::{error::Result, mesh::Mesh};

/// Load a single triangle mesh from an STL file (binary or ASCII —
/// `stl_io` auto-detects the format).
///
/// # Errors
///
/// Returns [`crate::error::Error::Io`] if the reader fails, or if the STL
/// content is malformed. `stl_io` does not distinguish I/O failures from
/// parse failures: both surface as `std::io::Error` (typically
/// `ErrorKind::InvalidData` for malformed content), so both map to the
/// same `Error::Io` variant here.
pub fn load_stl<R: Read + Seek>(mut reader: R) -> Result<Mesh> {
    let indexed = stl_io::read_stl(&mut reader)?;

    let vertices = indexed
        .vertices
        .iter()
        .map(|v| DVec3::new(v.0[0] as f64, v.0[1] as f64, v.0[2] as f64))
        .collect();

    let mut indices = Vec::with_capacity(indexed.faces.len() * 3);
    for face in &indexed.faces {
        for vertex_index in face.vertices {
            indices.push(vertex_index as u32);
        }
    }

    Ok(Mesh::new(vertices, indices))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn loads_a_single_triangle_from_ascii_stl() {
        let ascii = b"solid triangle
            facet normal 0 0 1
                outer loop
                    vertex 0 0 0
                    vertex 10 0 0
                    vertex 5 10 0
                endloop
            endfacet
            endsolid triangle";

        let mesh = load_stl(Cursor::new(ascii.to_vec())).expect("load should succeed");

        assert_eq!(mesh.vertices.len(), 3);
        assert_eq!(mesh.indices.len(), 3);
        assert_eq!(mesh.vertices[1], DVec3::new(10.0, 0.0, 0.0));
    }

    #[test]
    fn deduplicates_shared_vertices_across_triangles() {
        // Two triangles sharing an edge (0,0,0)-(10,0,0): six vertex
        // references but only four distinct positions.
        let triangles = [
            stl_io::Triangle {
                normal: stl_io::Normal::new([0.0, 0.0, 1.0]),
                vertices: [
                    stl_io::Vertex::new([0.0, 0.0, 0.0]),
                    stl_io::Vertex::new([10.0, 0.0, 0.0]),
                    stl_io::Vertex::new([5.0, 10.0, 0.0]),
                ],
            },
            stl_io::Triangle {
                normal: stl_io::Normal::new([0.0, 0.0, -1.0]),
                vertices: [
                    stl_io::Vertex::new([10.0, 0.0, 0.0]),
                    stl_io::Vertex::new([0.0, 0.0, 0.0]),
                    stl_io::Vertex::new([5.0, -10.0, 0.0]),
                ],
            },
        ];

        let mut buffer = Vec::new();
        stl_io::write_stl(&mut buffer, triangles.iter()).expect("stl should serialize");

        let mesh = load_stl(Cursor::new(buffer)).expect("load should succeed");

        assert_eq!(mesh.vertices.len(), 4);
        assert_eq!(mesh.indices.len(), 6);
    }

    #[test]
    fn rejects_malformed_stl_content() {
        let garbage = b"this is not a valid stl file at all".to_vec();
        let err = load_stl(Cursor::new(garbage)).unwrap_err();
        assert!(matches!(err, crate::error::Error::Io(_)));
    }
}
