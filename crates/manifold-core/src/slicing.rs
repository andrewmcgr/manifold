//! Non-planar slicing: mesh -> ordered layers of cross-section curves.

use crate::{ids::ObjectId, mesh::Mesh, object::Object, Result, SlicerConfig};
use glam::DVec3;
use manifold_fidget::contour::{extract_contours, plane_basis};
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
    /// The order-field value (see [`BUILD_DIRECTION`]/[`slice_mesh`]'s walk)
    /// whose isosurface produced this layer's contour loops. In today's
    /// flat-height-field case this is the layer's Z height; retained so
    /// downstream consumers (e.g. `toolpath::plan`) can stamp it onto
    /// per-segment metadata once non-planar order fields exist.
    pub order: f64,
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

/// Default divisor used to derive the marching-squares contour-extraction
/// grid's target cell size from `SlicerConfig::nozzle_diameter` (cell_size
/// = `nozzle_diameter / CONTOUR_REFINEMENT_DIVISOR`). `4.0` (a quarter of
/// the nozzle diameter) keeps grid faceting finer than what the nozzle can
/// physically resolve, but is expensive at real-world scale (grid points
/// scale with the square of resolution, each doing a BVH nearest-triangle
/// query); `1.4` trades some of that headroom for tractable slicing time
/// while still meaningfully improving on the old fixed-120 grid. Exposed
/// as a constant (rather than inlined) so callers wanting coarser/finer
/// refinement can pass a different divisor to [`contour_resolution`]
/// directly.
const CONTOUR_REFINEMENT_DIVISOR: f64 = 1.4;

/// Lower/upper bounds on the derived grid resolution (samples per axis),
/// independent of `CONTOUR_REFINEMENT_DIVISOR`: guards against a
/// vanishingly coarse grid (degenerate/huge `extent` or `nozzle_diameter`)
/// and against runaway sampling cost (tiny `nozzle_diameter` on a large
/// mesh).
const MIN_CONTOUR_RESOLUTION: usize = 32;
const MAX_CONTOUR_RESOLUTION: usize = 512;

/// Derives the marching-squares contour-extraction grid resolution
/// (samples per axis, see [`extract_contours`]) for an in-plane sampling
/// square of side length `extent`, targeting a grid cell size of
/// `nozzle_diameter / refinement_divisor`.
///
/// Adaptive rather than fixed (as this previously was, via a
/// `CONTOUR_RESOLUTION` constant): a fixed grid either wastes samples on
/// small objects or under-samples large ones, producing visibly faceted/
/// blocky contours. Deriving resolution from the mesh's actual footprint
/// and the machine's nozzle diameter scales the grid to both.
///
/// Clamped to `[MIN_CONTOUR_RESOLUTION, MAX_CONTOUR_RESOLUTION]` to bound
/// cost and guard against non-finite/non-positive inputs.
fn contour_resolution(extent: f64, nozzle_diameter: f64, refinement_divisor: f64) -> usize {
    let cell_size = (nozzle_diameter / refinement_divisor).max(f64::EPSILON);
    let raw = (extent / cell_size).ceil() as i64 + 1;
    raw.clamp(MIN_CONTOUR_RESOLUTION as i64, MAX_CONTOUR_RESOLUTION as i64) as usize
}

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

    // In-plane sample extent: sized off the mesh's bounding box
    // *projected onto the contour-extraction plane's in-plane basis*
    // (perpendicular to BUILD_DIRECTION), not its full 3D diagonal. Using
    // the 3D diagonal (which includes the mesh's extent along
    // BUILD_DIRECTION, i.e. its height) wastes most of the fixed
    // CONTOUR_RESOLUTION grid on empty space for any object where height
    // dominates footprint, causing near-tip/near-base layers to fall
    // between grid samples and come back with zero contour loops.
    let (basis1, basis2) = plane_basis(BUILD_DIRECTION);
    let extent = in_plane_extent(min, max, basis1, basis2);

    // Center the sampling plane's origin on the mesh's actual in-plane
    // (footprint) position rather than the world origin.
    // `extract_contours_at_order` computes `origin = direction * order_value`,
    // which always has zero in-plane (basis1/basis2) components — fine only
    // when the mesh's footprint happens to straddle the world origin. For a
    // mesh translated far from world (0,0) (e.g. after `object::center_on_bed`
    // places it at the bed's center), the fixed-extent sampling window then
    // never reaches the mesh at all, so every layer comes back empty. Instead
    // we anchor the origin at the mesh's bounding-box center projected onto
    // the in-plane axes, while still solving for the BUILD_DIRECTION
    // component so `origin.dot(BUILD_DIRECTION) == order_value` (i.e. origin
    // stays exactly on the correct slicing plane for each layer).
    let bbox_center = (min + max) * 0.5;

    let layer_height = config.layer_height.abs().max(f64::EPSILON);
    let resolution = contour_resolution(extent, config.nozzle_diameter, CONTOUR_REFINEMENT_DIVISOR);

    let mut layers = Vec::new();
    let mut order_value = order_min;
    let mut index = 0;
    while order_value <= order_max {
        let origin =
            bbox_center + BUILD_DIRECTION * (order_value - bbox_center.dot(BUILD_DIRECTION));
        let loops = extract_contours(
            &sdf, origin, basis1, basis2, extent, extent, resolution, resolution, 0.0,
        );
        layers.push(Layer {
            index,
            object: ObjectId::default(),
            order: order_value,
            loops,
        });
        index += 1;
        order_value += layer_height;
    }

    Ok(layers)
}

/// Computes a square in-plane sampling extent (see [`slice_mesh`]) large
/// enough to cover the mesh's bounding box `[min, max]` once projected onto
/// the `basis1`/`basis2` plane, independent of the mesh's extent along the
/// (perpendicular) build direction.
///
/// Projects all 8 bounding-box corners onto `basis1`/`basis2`, takes the
/// resulting 2D range's diagonal, and applies the same `* 1.5 + 1.0` margin
/// the old (buggy) full-3D-diagonal computation used, so behavior for
/// mostly-flat meshes (where footprint ~= 3D diagonal) is unchanged.
fn in_plane_extent(min: DVec3, max: DVec3, basis1: DVec3, basis2: DVec3) -> f64 {
    let corners = [
        DVec3::new(min.x, min.y, min.z),
        DVec3::new(max.x, min.y, min.z),
        DVec3::new(min.x, max.y, min.z),
        DVec3::new(min.x, min.y, max.z),
        DVec3::new(max.x, max.y, min.z),
        DVec3::new(max.x, min.y, max.z),
        DVec3::new(min.x, max.y, max.z),
        DVec3::new(max.x, max.y, max.z),
    ];

    let mut u_min = f64::INFINITY;
    let mut u_max = f64::NEG_INFINITY;
    let mut v_min = f64::INFINITY;
    let mut v_max = f64::NEG_INFINITY;
    for corner in corners {
        let u = corner.dot(basis1);
        let v = corner.dot(basis2);
        u_min = u_min.min(u);
        u_max = u_max.max(u);
        v_min = v_min.min(v);
        v_max = v_max.max(v);
    }

    let projected_diagonal = DVec3::new(u_max - u_min, v_max - v_min, 0.0).length();
    projected_diagonal * 1.5 + 1.0
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
    fn contour_resolution_scales_with_extent() {
        let small = contour_resolution(4.0, 0.4, CONTOUR_REFINEMENT_DIVISOR);
        let large = contour_resolution(40.0, 0.4, CONTOUR_REFINEMENT_DIVISOR);
        assert!(
            large > small,
            "a larger in-plane extent should derive a finer (larger) grid resolution"
        );
    }

    #[test]
    fn contour_resolution_scales_inversely_with_nozzle_diameter() {
        let coarse_nozzle = contour_resolution(40.0, 0.8, CONTOUR_REFINEMENT_DIVISOR);
        let fine_nozzle = contour_resolution(40.0, 0.2, CONTOUR_REFINEMENT_DIVISOR);
        assert!(
            fine_nozzle > coarse_nozzle,
            "a smaller nozzle diameter should derive a finer (larger) grid resolution"
        );
    }

    #[test]
    fn contour_resolution_respects_the_refinement_divisor_parameter() {
        // A larger divisor targets a coarser cell size (nozzle_diameter /
        // divisor), so resolution should drop as the divisor shrinks.
        let finer = contour_resolution(40.0, 0.4, 8.0);
        let coarser = contour_resolution(40.0, 0.4, 2.0);
        assert!(
            finer > coarser,
            "a larger refinement divisor should derive a finer (larger) grid resolution"
        );
    }

    #[test]
    fn contour_resolution_is_clamped_to_the_configured_bounds() {
        // Vanishingly small extent / huge nozzle diameter -> would derive a
        // resolution below MIN_CONTOUR_RESOLUTION without clamping.
        assert_eq!(
            contour_resolution(0.001, 10.0, CONTOUR_REFINEMENT_DIVISOR),
            MIN_CONTOUR_RESOLUTION
        );
        // Huge extent / vanishingly small nozzle diameter -> would derive a
        // resolution above MAX_CONTOUR_RESOLUTION without clamping.
        assert_eq!(
            contour_resolution(1.0e6, 0.001, CONTOUR_REFINEMENT_DIVISOR),
            MAX_CONTOUR_RESOLUTION
        );
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
    fn slice_mesh_produces_nonempty_contour_loops_for_a_cube_far_from_the_world_origin() {
        // Regression test: before the fix, the contour-extraction plane's
        // origin always had zero in-plane (X/Y) components regardless of
        // where the mesh actually sits, so a mesh translated far from world
        // (0,0) (e.g. after `object::center_on_bed`) produced zero contour
        // loops for every layer, even though the SDF field itself is fine.
        let offset = DVec3::new(500.0, 500.0, 0.0);
        let mesh = cube_mesh();
        let translated = Mesh::new(
            mesh.vertices
                .iter()
                .map(|&vertex| vertex + offset)
                .collect(),
            mesh.indices.clone(),
        );

        let config = SlicerConfig {
            layer_height: 0.25,
            ..SlicerConfig::default()
        };

        let layers = slice_mesh(&translated, &config).unwrap();

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

    /// Slim square pyramid: base [-0.5, 0.5]^2 at Z=0, apex at (0, 0, 20)
    /// — height 20x its footprint. Before the fix, `extent` was sized off
    /// the full 3D bounding-box diagonal (dominated by the 20-tall
    /// height), starving the 120x120 sampling grid of resolution across
    /// the ~1-unit-wide footprint and causing layers through the body
    /// (not just the near-degenerate apex tip) to come back with zero
    /// contour loops.
    fn tall_thin_pyramid_mesh() -> Mesh {
        let vertices = vec![
            DVec3::new(-0.5, -0.5, 0.0),
            DVec3::new(0.5, -0.5, 0.0),
            DVec3::new(0.5, 0.5, 0.0),
            DVec3::new(-0.5, 0.5, 0.0),
            DVec3::new(0.0, 0.0, 20.0),
        ];
        let indices = vec![
            0, 2, 1, 0, 3, 2, // base (-Z winding)
            0, 1, 4, // side
            1, 2, 4, // side
            2, 3, 4, // side
            3, 0, 4, // side
        ];
        Mesh::new(vertices, indices)
    }

    #[test]
    fn slice_mesh_has_no_empty_contour_gaps_through_a_tall_thin_pyramid() {
        let config = SlicerConfig {
            layer_height: 1.0,
            ..SlicerConfig::default()
        };

        let layers = slice_mesh(&tall_thin_pyramid_mesh(), &config).unwrap();
        assert!(!layers.is_empty());

        let first_nonempty = layers.iter().position(|layer| !layer.loops.is_empty());
        let last_nonempty = layers.iter().rposition(|layer| !layer.loops.is_empty());

        let (Some(first_nonempty), Some(last_nonempty)) = (first_nonempty, last_nonempty) else {
            panic!("expected at least one layer with contour loops");
        };

        for layer in &layers[first_nonempty..=last_nonempty] {
            assert!(
                !layer.loops.is_empty(),
                "layer {} at order {} unexpectedly has no contour loops \
                 (in-plane sampling extent too large for the footprint)",
                layer.index,
                layer.order
            );
        }
    }
}
