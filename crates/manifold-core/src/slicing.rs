//! Non-planar slicing: mesh -> ordered layers of cross-section curves.

use crate::{ids::ObjectId, mesh::Mesh, object::Object, Result, SlicerConfig};

/// A single (possibly non-planar) slice layer.
///
/// Tagged with the source [`ObjectId`] so multi-object toolpath planning
/// (tool lookup) and any future Z-interleaving ordering strategy can tell
/// which object a layer came from.
#[derive(Debug, Clone, Default)]
pub struct Layer {
    pub index: usize,
    pub object: ObjectId,
}

/// Slice a single mesh (already in the frame it should be sliced in) into
/// layers according to `config`.
///
/// Placeholder implementation: real non-planar slicing logic lives here.
/// Operates in whatever space `mesh`'s vertices are already in — callers
/// slicing an [`Object`] should go through [`slice_object`], which bakes
/// the object's transform into world space first.
pub fn slice_mesh(_mesh: &Mesh, _config: &SlicerConfig) -> Result<Vec<Layer>> {
    Ok(Vec::new())
}

/// Slice a single [`Object`]: bakes its `transform` into world-space
/// vertices, then slices that with [`slice_mesh`], tagging every
/// resulting layer with the object's id.
pub fn slice_object(object: &Object, config: &SlicerConfig) -> Result<Vec<Layer>> {
    let world_mesh = Mesh::new(
        object
            .mesh
            .vertices
            .iter()
            .map(|&vertex| object.transform.transform_point(vertex))
            .collect(),
        object.mesh.indices.clone(),
    );

    let mut layers = slice_mesh(&world_mesh, config)?;
    for layer in &mut layers {
        layer.object = object.id;
    }
    Ok(layers)
}

/// Slice every object in a workspace, in the order given by `order`
/// (produced by an [`crate::ordering::ObjectOrderStrategy`]), concatenating
/// each object's full layer stack back-to-back.
///
/// This concatenation *is* what makes ordering "sequential" today: each
/// object is fully sliced before the next begins. A future
/// Z-interleaving/simultaneous-printing strategy would replace this
/// concatenation with a per-Z merge of layers across objects — see
/// ROADMAP.md "Deferred / future work".
///
/// # Errors
///
/// Returns [`crate::Error::InvalidMesh`] if `order` references an object id
/// not present in `objects`.
pub fn slice_workspace(
    objects: &[Object],
    order: &[ObjectId],
    config: &SlicerConfig,
) -> Result<Vec<Layer>> {
    let mut layers = Vec::new();
    for &object_id in order {
        let object = objects
            .iter()
            .find(|object| object.id == object_id)
            .ok_or_else(|| {
                crate::Error::InvalidMesh(format!(
                    "print order references unknown object {object_id}"
                ))
            })?;
        layers.extend(slice_object(object, config)?);
    }
    Ok(layers)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ids::ToolId, transform::Transform};
    use glam::DVec3;

    fn triangle_mesh() -> Mesh {
        Mesh::new(
            vec![
                DVec3::ZERO,
                DVec3::new(1.0, 0.0, 0.0),
                DVec3::new(0.0, 1.0, 0.0),
            ],
            vec![0, 1, 2],
        )
    }

    #[test]
    fn slice_object_applies_world_transform_before_slicing() {
        let mut object = Object::new(ObjectId(0), triangle_mesh(), ToolId(0));
        object.transform = Transform::from_translation(DVec3::new(5.0, 0.0, 0.0));

        // Placeholder slice_mesh always returns no layers today; this
        // asserts slice_object doesn't error and wires the transform in
        // (exercised further once slice_mesh has real geometry logic).
        assert!(slice_object(&object, &SlicerConfig::default()).is_ok());
    }

    #[test]
    fn slice_workspace_concatenates_in_given_order() {
        let objects = vec![
            Object::new(ObjectId(0), triangle_mesh(), ToolId(0)),
            Object::new(ObjectId(1), triangle_mesh(), ToolId(1)),
        ];
        let order = vec![ObjectId(1), ObjectId(0)];

        let layers = slice_workspace(&objects, &order, &SlicerConfig::default()).unwrap();

        // Placeholder slice_mesh produces zero layers per object today, so
        // this mainly asserts the lookup/ordering doesn't error.
        assert!(layers.is_empty());
    }

    #[test]
    fn slice_workspace_rejects_unknown_object_in_order() {
        let objects = vec![Object::new(ObjectId(0), triangle_mesh(), ToolId(0))];
        let order = vec![ObjectId(99)];

        let err = slice_workspace(&objects, &order, &SlicerConfig::default()).unwrap_err();
        assert!(matches!(err, crate::Error::InvalidMesh(_)));
    }
}
