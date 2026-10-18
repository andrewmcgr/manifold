use std::fs::File;
use std::io::{BufReader, Read, Seek};
use std::path::Path;

use crate::bridge::ffi;
use crate::io::{build_loaded_object_from_part, LoadedObjectData};

/// Parses a binary or ASCII `.stl` file from disk using `stl_io` and converts it
/// into a C++ backend mesh object (`ffi::register_triangle_mesh`).
pub fn load_stl_file(path: &Path) -> Result<LoadedObjectData, String> {
    let file = File::open(path).map_err(|e| format!("Failed to open STL file: {e}"))?;
    let mut reader = BufReader::new(file);
    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("stl_model");
    load_stl_from_reader(&mut reader, name, &path.to_string_lossy())
}

/// Parses an STL stream (`Read + Seek`) via `stl_io::read_stl` and registers the
/// resulting indexed triangle mesh in the C++ backend.
pub fn load_stl_from_reader<R: Read + Seek>(
    reader: &mut R,
    name: &str,
    source_label: &str,
) -> Result<LoadedObjectData, String> {
    let indexed_mesh =
        stl_io::read_stl(reader).map_err(|e| format!("STL parse error in {source_label}: {e}"))?;

    if indexed_mesh.vertices.is_empty() || indexed_mesh.faces.is_empty() {
        return Err(format!("STL mesh '{name}' contains no triangles"));
    }

    let mut positions = Vec::with_capacity(indexed_mesh.vertices.len() * 3);
    for v in &indexed_mesh.vertices {
        positions.push(v[0]);
        positions.push(v[1]);
        positions.push(v[2]);
    }

    let mut indices = Vec::with_capacity(indexed_mesh.faces.len() * 3);
    for face in &indexed_mesh.faces {
        indices.push(face.vertices[0] as u32);
        indices.push(face.vertices[1] as u32);
        indices.push(face.vertices[2] as u32);
    }

    let part_info = ffi::register_triangle_mesh(name, &positions, &indices);
    build_loaded_object_from_part(part_info, source_label)
        .ok_or_else(|| format!("Backend failed to register STL mesh '{name}'"))
}
