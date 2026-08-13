//! 3MF (3D Manufacturing Format) loading, via the `lib3mf` crate.
//!
//! TODO(roadmap): Phase 1 (see ROADMAP.md) — this covers the common case
//! (one or more mesh objects referenced directly by build items, with an
//! optional per-item transform). It does not yet flatten `Object`
//! assemblies (`components`) into world-space geometry, and it does not
//! extract materials into [`crate::material::Material`]/[`crate::ids::MaterialId`].
//! Both are tracked as follow-up work.

use std::io::{Read, Seek};

use glam::{DMat3, DVec3};

use crate::{
    error::{Error, Result},
    ids::{ObjectId, ToolId},
    mesh::Mesh,
    object::Object,
    transform::Transform,
};

/// Load every buildable object from a 3MF file, assigning them all to
/// `tool`.
///
/// Each 3MF build item becomes one [`Object`]: its referenced mesh is
/// converted (with unit-to-millimeter scaling applied to vertex
/// coordinates) and its optional transform is decoded into a
/// [`Transform`]. Object IDs are assigned sequentially starting from 0,
/// in build-item order.
///
/// # Errors
///
/// Returns [`Error::ThreeMf`] if the underlying file cannot be parsed,
/// and [`Error::InvalidMesh`] if a build item references an unknown
/// object, an object with no mesh (e.g. an assembly of components), or
/// the model declares an unsupported unit.
pub fn load_3mf<R: Read + Seek>(reader: R, tool: ToolId) -> Result<Vec<Object>> {
    let model = lib3mf::Model::from_reader(reader)?;
    let unit_scale = unit_to_millimeter_scale(&model.unit)?;

    model
        .build
        .items
        .iter()
        .enumerate()
        .map(|(index, item)| {
            let source = model
                .resources
                .objects
                .iter()
                .find(|object| object.id == item.objectid)
                .ok_or_else(|| {
                    Error::InvalidMesh(format!(
                        "3mf build item references unknown object id {}",
                        item.objectid
                    ))
                })?;

            let source_mesh = source.mesh.as_ref().ok_or_else(|| {
                Error::InvalidMesh(format!(
                    "3mf object {} has no mesh (assemblies/components are not yet supported)",
                    source.id
                ))
            })?;

            let mesh = convert_mesh(source_mesh, unit_scale);
            let transform = item
                .transform
                .map(matrix_to_transform)
                .unwrap_or_else(Transform::identity);

            let mut object = Object::new(ObjectId(index as u32), mesh, tool);
            object.transform = transform;
            Ok(object)
        })
        .collect()
}

/// Convert a `lib3mf` mesh into a [`Mesh`], scaling vertex coordinates
/// from the model's declared unit into millimeters.
fn convert_mesh(mesh: &lib3mf::Mesh, unit_scale: f64) -> Mesh {
    let vertices = mesh
        .vertices
        .iter()
        .map(|v| DVec3::new(v.x, v.y, v.z) * unit_scale)
        .collect();

    let mut indices = Vec::with_capacity(mesh.triangles.len() * 3);
    for triangle in &mesh.triangles {
        indices.push(triangle.v1 as u32);
        indices.push(triangle.v2 as u32);
        indices.push(triangle.v3 as u32);
    }

    Mesh::new(vertices, indices)
}

/// Decode a 3MF build-item transform (a row-major 4x3 affine matrix: three
/// rows of the 3x3 linear part followed by a translation row) into a
/// [`Transform`].
fn matrix_to_transform(m: [f64; 12]) -> Transform {
    let mat3 = DMat3::from_cols(
        DVec3::new(m[0], m[3], m[6]),
        DVec3::new(m[1], m[4], m[7]),
        DVec3::new(m[2], m[5], m[8]),
    );
    let translation = DVec3::new(m[9], m[10], m[11]);
    Transform(glam::DAffine3::from_mat3_translation(mat3, translation))
}

/// Resolve a 3MF unit name to the scale factor that converts it to
/// millimeters.
fn unit_to_millimeter_scale(unit: &str) -> Result<f64> {
    match unit {
        "micron" => Ok(0.001),
        "millimeter" => Ok(1.0),
        "centimeter" => Ok(10.0),
        "inch" => Ok(25.4),
        "foot" => Ok(304.8),
        "meter" => Ok(1000.0),
        other => Err(Error::InvalidMesh(format!("unsupported 3mf unit: {other}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn write_model(model: lib3mf::Model) -> Vec<u8> {
        let buffer = Cursor::new(Vec::new());
        let buffer = model.to_writer(buffer).expect("model should serialize");
        buffer.into_inner()
    }

    fn triangle_mesh() -> lib3mf::Mesh {
        let mut mesh = lib3mf::Mesh::new();
        mesh.vertices.push(lib3mf::Vertex::new(0.0, 0.0, 0.0));
        mesh.vertices.push(lib3mf::Vertex::new(10.0, 0.0, 0.0));
        mesh.vertices.push(lib3mf::Vertex::new(5.0, 10.0, 0.0));
        mesh.triangles.push(lib3mf::Triangle::new(0, 1, 2));
        mesh
    }

    #[test]
    fn loads_a_single_object_at_identity() {
        let mut model = lib3mf::Model::new();
        model.unit = "millimeter".to_string();

        let mut object = lib3mf::Object::new(1);
        object.mesh = Some(triangle_mesh());
        model.resources.objects.push(object);
        model.build.items.push(lib3mf::BuildItem::new(1));

        let bytes = write_model(model);
        let objects = load_3mf(Cursor::new(bytes), ToolId(0)).expect("load should succeed");

        assert_eq!(objects.len(), 1);
        let loaded = &objects[0];
        assert_eq!(loaded.mesh.vertices.len(), 3);
        assert_eq!(loaded.mesh.indices, vec![0, 1, 2]);
        assert_eq!(loaded.transform, Transform::identity());
        assert_eq!(loaded.tool, ToolId(0));
        assert_eq!(loaded.mesh.vertices[1], DVec3::new(10.0, 0.0, 0.0));
    }

    #[test]
    fn applies_build_item_transform() {
        let mut model = lib3mf::Model::new();
        model.unit = "millimeter".to_string();

        let mut object = lib3mf::Object::new(1);
        object.mesh = Some(triangle_mesh());
        model.resources.objects.push(object);

        let mut item = lib3mf::BuildItem::new(1);
        // Row-major 4x3: identity 3x3 rotation/scale, translation (1, 2, 3).
        item.transform = Some([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 3.0]);
        model.build.items.push(item);

        let bytes = write_model(model);
        let objects = load_3mf(Cursor::new(bytes), ToolId(0)).expect("load should succeed");

        let loaded = &objects[0];
        assert_eq!(
            loaded.transform.transform_point(DVec3::ZERO),
            DVec3::new(1.0, 2.0, 3.0)
        );
    }

    #[test]
    fn converts_units_to_millimeters() {
        let mut model = lib3mf::Model::new();
        model.unit = "centimeter".to_string();

        let mut object = lib3mf::Object::new(1);
        object.mesh = Some(triangle_mesh());
        model.resources.objects.push(object);
        model.build.items.push(lib3mf::BuildItem::new(1));

        let bytes = write_model(model);
        let objects = load_3mf(Cursor::new(bytes), ToolId(0)).expect("load should succeed");

        assert_eq!(objects[0].mesh.vertices[1], DVec3::new(100.0, 0.0, 0.0));
    }

    #[test]
    fn rejects_build_item_referencing_unknown_object() {
        let mut model = lib3mf::Model::new();
        model.unit = "millimeter".to_string();
        model.build.items.push(lib3mf::BuildItem::new(42));

        let bytes = write_model(model);
        let err = load_3mf(Cursor::new(bytes), ToolId(0)).unwrap_err();
        // lib3mf itself validates build-item object references while
        // parsing, so this may surface as `Error::ThreeMf` (caught by
        // lib3mf) rather than our own `Error::InvalidMesh` check in
        // `load_3mf` (kept as defensive code for lenient parser configs).
        assert!(matches!(err, Error::InvalidMesh(_) | Error::ThreeMf(_)));
    }

    #[test]
    fn rejects_object_without_mesh() {
        let mut model = lib3mf::Model::new();
        model.unit = "millimeter".to_string();

        // An object with no mesh (e.g. a components-only assembly).
        let object = lib3mf::Object::new(1);
        model.resources.objects.push(object);
        model.build.items.push(lib3mf::BuildItem::new(1));

        let bytes = write_model(model);
        let err = load_3mf(Cursor::new(bytes), ToolId(0)).unwrap_err();
        assert!(matches!(err, Error::InvalidMesh(_)));
    }
}
