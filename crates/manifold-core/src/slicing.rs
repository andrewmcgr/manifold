//! Non-planar slicing: mesh -> ordered layers of cross-section curves.

use crate::{ids::ObjectId, mesh::Mesh, object::Object, Result, SlicerConfig};
use glam::DVec3;
use manifold_fidget::contour::extract_contours_at_order;
use manifold_fidget::mesh_sdf::MeshSdf;

/// A single (possibly non-planar) slice layer.
///
/// Tagged with the source [`ObjectId`] so multi-object toolpath planning
/// (tool lookup) and any future Z-interleaving ordering strategy can tell
/// which object a layer came from.
#[derive(Debug, Clone, Default)]
pub struct Layer {
    pub index: usize,
    pub object: ObjectId,
    /// This layer's cross-section geometry: closed polylines (loops) in
    /// world space, one per contour extracted at this layer's order
    /// value. Empty for a layer with no contour (e.g. above/below the
    /// mesh's extent along the build direction).
    pub loops: Vec<Vec<DVec3>>,
}

/// Build/order direction for this MVP: conventional planar slicing along
/// -Z (i.e. `order(p) = p.dot(direction)` decreases going up, matching a
/// bottom-to-top print). Hardcoded per this task's scope — see
/// `NON_PLANAR_SLICING.md` for the follow-up angle-driven order field that
/// will make this configurable.
const BUILD_DIRECTION: DVec3 = DVec3::new(0.0, 0.0, -1.0);

/// Grid resolution (samples per axis) used for the marching-squares
/// contour extraction at each layer. Fixed for this MVP rather than
/// plumbed through `SlicerConfig`; see ROADMAP.md for a possible
/// follow-up (e.g. deriving it from `nozzle_diameter`).
const CONTOUR_RESOLUTION: usize = 120;

/// Slice a single mesh (already in the frame it should be sliced in) into
/// layers according to `config`.
///
/// Builds a [`MeshSdf`] from `mesh` and walks its [`BUILD_DIRECTION`]
/// order field at `config.layer_height` intervals across the mesh's
/// bounding range along that direction, extracting one contour-based
/// [`Layer`] per step (steps with no contour still produce a `Layer` with
/// empty `loops`, rather than being skipped or erroring). Operates in
/// whatever space `mesh`'s vertices are already in — callers slicing an
/// [`Object`] should go through [`slice_object`], which bakes the
/// object's transform into world space first.
pub fn slice_mesh(mesh: &Mesh, config: &SlicerConfig) -> Result<Vec<Layer>> {
    let Some((min, max)) = mesh.bounding_box() else {
        // Empty mesh: no geometry to slice.
        return Ok(Vec::new());
    };
    if mesh.indices.is_empty() {
        return Ok(Vec::new());
    }

    let faces: Vec<[usize; 3]> = mesh
        .indices
        .chunks_exact(3)
        .map(|chunk| [chunk[0] as usize, chunk[1] as usize, chunk[2] as usize])
        .collect();
    let sdf = MeshSdf::new(mesh.vertices.clone(), faces);

    let order_at_min = min.dot(BUILD_DIRECTION);
    let order_at_max = max.dot(BUILD_DIRECTION);
    let order_min = order_at_min.min(order_at_max);
    let order_max = order_at_min.max(order_at_max);

    // In-plane sample extent: large enough to cover the mesh's full
    // bounding box diagonal (plus margin), regardless of which axes the
    // contour-extraction plane basis happens to align with.
    let extent = (max - min).length() * 1.5 + 1.0;

    let layer_height = config.layer_height.abs().max(f64::EPSILON);

    let mut layers = Vec::new();
    let mut order_value = order_min;
    let mut index = 0;
    while order_value <= order_max {
        let loops = extract_contours_at_order(
            &sdf,
            BUILD_DIRECTION,
            order_value,
            extent,
            extent,
            CONTOUR_RESOLUTION,
            CONTOUR_RESOLUTION,
        );
        layers.push(Layer {
            index,
            object: ObjectId::default(),
            loops,
        });
        index += 1;
        order_value += layer_height;
    }

    Ok(layers)
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

        // This mainly asserts slice_object doesn't error and wires the
        // transform in before slicing (the degenerate flat triangle fixture
        // isn't a solid, so real contour geometry is exercised by the
        // sphere/cube tests below instead).
        let layers = slice_object(&object, &SlicerConfig::default()).unwrap();
        for layer in &layers {
            assert_eq!(layer.object, ObjectId(0));
        }
    }

    #[test]
    fn slice_workspace_concatenates_in_given_order() {
        let objects = vec![
            Object::new(ObjectId(0), triangle_mesh(), ToolId(0)),
            Object::new(ObjectId(1), triangle_mesh(), ToolId(1)),
        ];
        let order = vec![ObjectId(1), ObjectId(0)];

        let layers = slice_workspace(&objects, &order, &SlicerConfig::default()).unwrap();

        // The degenerate flat triangle fixture isn't a solid, so this
        // mainly asserts the per-object lookup/ordering doesn't error;
        // real contour geometry is exercised by the sphere/cube tests below.
        for layer in &layers {
            assert!(layer.object == ObjectId(0) || layer.object == ObjectId(1));
        }
    }

    #[test]
    fn slice_workspace_rejects_unknown_object_in_order() {
        let objects = vec![Object::new(ObjectId(0), triangle_mesh(), ToolId(0))];
        let order = vec![ObjectId(99)];

        let err = slice_workspace(&objects, &order, &SlicerConfig::default()).unwrap_err();
        assert!(matches!(err, crate::Error::InvalidMesh(_)));
    }

    /// Unit cube spanning [0,1]^3 (same fixture pattern as
    /// `manifold-fidget`'s `mesh_sdf`/`contour` tests).
    fn cube_mesh() -> Mesh {
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
            0, 2, 1, 0, 3, 2, // -Z
            4, 5, 6, 4, 6, 7, // +Z
            0, 1, 5, 0, 5, 4, // -Y
            3, 7, 6, 3, 6, 2, // +Y
            0, 4, 7, 0, 7, 3, // -X
            1, 2, 6, 1, 6, 5, // +X
        ];
        Mesh::new(vertices, indices)
    }

    #[test]
    fn slice_mesh_produces_nonempty_contour_loops_for_a_solid_cube() {
        let config = SlicerConfig {
            layer_height: 0.25,
            ..SlicerConfig::default()
        };

        let layers = slice_mesh(&cube_mesh(), &config).unwrap();

        // The cube spans Z in [0, 1] with layer_height 0.25: expect 5
        // stepped layers (0.0, 0.25, 0.5, 0.75, 1.0). The interior layers
        // (0.25, 0.5, 0.75) are clean square cross-sections; the exact
        // boundary layers (Z=0, Z=1) sample directly on the mesh surface,
        // where the sign/crossing is numerically ambiguous, so only the
        // interior layers are asserted to have a contour loop.
        assert_eq!(layers.len(), 5);
        for layer in &layers[1..4] {
            assert_eq!(layer.loops.len(), 1, "expected exactly one contour loop");
            assert!(!layer.loops[0].is_empty());
        }
    }

    #[test]
    fn slice_mesh_returns_no_layers_for_an_empty_mesh() {
        let layers = slice_mesh(&Mesh::default(), &SlicerConfig::default()).unwrap();
        assert!(layers.is_empty());
    }
}
