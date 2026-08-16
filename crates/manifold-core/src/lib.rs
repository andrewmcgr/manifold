//! Manifold slicing engine.
//!
//! `manifold-core` turns a mesh into non-planar toolpaths and emits Gcode.
//! It has no UI or CLI dependencies so it can run headless (e.g. embedded
//! in a service) or be driven by the `manifold-cli` front-end.

pub mod bounds;
pub mod error;
pub mod gcode;
pub mod ids;
pub mod infill;
pub mod machine;
pub mod material;
pub mod mesh;
pub mod object;
pub mod order_field;
pub mod ordering;
pub mod polygon2d;
pub mod slicing;
pub mod stl;
pub mod threemf;
pub mod tool;
pub mod toolpath;
pub mod transform;
pub mod workspace;

pub use error::{Error, Result};
pub use workspace::Workspace;

/// Slicer configuration shared across the pipeline.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SlicerConfig {
    pub layer_height: f64,
    pub nozzle_diameter: f64,
    /// Strategy used to decide the order objects are printed in. See
    /// `ordering` module and ROADMAP.md open decision #2.
    pub object_ordering: ordering::ObjectOrderingKind,
    /// Nozzle-center line width for a single wall/perimeter pass, in mm.
    /// Also used as the spacing between successive wall passes.
    pub wall_line_width: f64,
    /// Total desired shell thickness (mm), i.e. how far inward the walls
    /// should extend from the outer surface. The actual number of wall
    /// passes is derived (see [`SlicerConfig::wall_count`]) by rounding
    /// `shell_thickness / wall_line_width` to the nearest whole wall,
    /// clamped to a minimum of one.
    pub shell_thickness: f64,
    /// Inset of the outermost wall's nozzle-center path from the true
    /// mesh surface, in mm. Defaults to half the nozzle diameter so the
    /// nozzle's outer edge (not its center) lands on the surface.
    pub wall_offset: f64,
    /// Which infill pattern to generate inside each layer's innermost
    /// wall loop(s). See `infill::InfillPatternKind`.
    pub infill_pattern: infill::InfillPatternKind,
    /// Nozzle-center line width for infill passes, in mm. Also the scan-
    /// line spacing for the `Monotonic` pattern. Defaults to
    /// `nozzle_diameter`.
    pub infill_line_width: f64,
    /// Base infill scan-line angle (degrees), before per-layer
    /// alternation. `infill::MonotonicInfill` alternates ±this angle by
    /// layer index (even layers add it, odd layers subtract it),
    /// measured relative to the source object's own orientation (see
    /// `Transform::in_plane_rotation_angle`) so rotating an object
    /// rotates its infill with it. Defaults to `45.0`.
    pub infill_angle_deg: f64,
    /// Number of fully solid layers to generate at the top of each
    /// object (adjacent to any facing-up exterior surface), replacing
    /// what would otherwise be sparse infill with a solid fill pattern.
    /// Defaults to `3`.
    pub top_layers: usize,
    /// Number of fully solid layers to generate at the bottom of each
    /// object (adjacent to any facing-down exterior surface), replacing
    /// what would otherwise be sparse infill with a solid fill pattern.
    /// Defaults to `3`.
    pub bottom_layers: usize,
    /// Which order field slicing walks isosurfaces of. See
    /// `order_field::OrderFieldKind`. Defaults to `Height`, i.e. today's
    /// exact flat planar slicing along `slicing::BUILD_DIRECTION`.
    pub order_field: order_field::OrderFieldKind,
    /// Apex point of the cone used when `order_field` is
    /// `OrderFieldKind::Conical`. Inert (unused) otherwise. Defaults to
    /// the origin.
    pub order_field_apex: glam::DVec3,
    /// Axis the cone opens along when `order_field` is
    /// `OrderFieldKind::Conical`. Inert (unused) otherwise. Defaults to
    /// `slicing::BUILD_DIRECTION`, matching the `Height` field's
    /// direction so switching kinds is a smooth transition.
    pub order_field_axis: glam::DVec3,
    /// Cone steepness used when `order_field` is `OrderFieldKind::Conical`
    /// (`0.0` degenerates to a flat height field). Inert (unused)
    /// otherwise. Defaults to `0.0`.
    pub order_field_slope: f64,
}

impl Default for SlicerConfig {
    fn default() -> Self {
        let nozzle_diameter = 0.4;
        let wall_line_width = nozzle_diameter;
        Self {
            layer_height: 0.2,
            nozzle_diameter,
            object_ordering: ordering::ObjectOrderingKind::default(),
            wall_line_width,
            shell_thickness: wall_line_width,
            wall_offset: nozzle_diameter / 2.0,
            infill_pattern: infill::InfillPatternKind::default(),
            infill_line_width: nozzle_diameter,
            infill_angle_deg: 45.0,
            top_layers: 3,
            bottom_layers: 3,
            order_field: order_field::OrderFieldKind::default(),
            order_field_apex: glam::DVec3::ZERO,
            order_field_axis: slicing::BUILD_DIRECTION,
            order_field_slope: 0.0,
        }
    }
}

impl SlicerConfig {
    /// Number of wall/perimeter passes derived from `shell_thickness` /
    /// `wall_line_width`, rounded to the nearest whole wall and clamped to
    /// a minimum of one (a `shell_thickness` smaller than `wall_line_width`
    /// still gets a single outer wall, never zero).
    #[must_use]
    pub fn wall_count(&self) -> usize {
        let line_width = self.wall_line_width.abs().max(f64::EPSILON);
        (self.shell_thickness / line_width).round().max(1.0) as usize
    }
}

/// Run the full pipeline: workspace -> order objects -> slice -> plan
/// toolpaths -> emit Gcode.
///
/// # Errors
///
/// Returns [`Error::InvalidMesh`] if `workspace` has no objects, or
/// whatever error the slicing/toolpath stages produce.
pub fn slice_to_gcode(workspace: &Workspace) -> Result<String> {
    let paths = plan_toolpaths(workspace)?;
    Ok(gcode::emit(&paths, &workspace.config))
}

/// Run the pipeline up to (and including) toolpath planning, stopping short
/// of Gcode emission: workspace -> order objects -> slice -> plan toolpaths.
///
/// Exposes the intermediate `Vec<toolpath::Path>` for consumers (e.g. the
/// GUI's 3D preview) that need the planned geometry without re-parsing
/// emitted Gcode text. [`slice_to_gcode`] builds on top of this and emits
/// Gcode from the same planned paths.
///
/// # Errors
///
/// Returns [`Error::InvalidMesh`] if `workspace` has no objects, or
/// whatever error the ordering/slicing/toolpath stages produce.
pub fn plan_toolpaths(workspace: &Workspace) -> Result<Vec<toolpath::Path>> {
    plan_toolpaths_with_progress(workspace, &mut |_| {})
}

/// Same as [`plan_toolpaths`], but calls `on_progress` with a `0.0..=1.0`
/// fraction of how far through the order-field domain slicing currently
/// is (see [`slicing::slice_workspace_with_progress`]), so a caller
/// running this on a background thread (slicing can be slow) can show
/// live progress without needing to know anything about layers or order
/// fields. Toolpath planning itself is comparatively fast and is not
/// separately reported — `on_progress` reaches `1.0` once slicing
/// finishes, before toolpath planning runs.
///
/// # Errors
///
/// Returns [`Error::InvalidMesh`] if `workspace` has no objects, or
/// whatever error the ordering/slicing/toolpath stages produce.
pub fn plan_toolpaths_with_progress(
    workspace: &Workspace,
    on_progress: &mut (dyn FnMut(f64) + Send),
) -> Result<Vec<toolpath::Path>> {
    if workspace.objects.is_empty() {
        return Err(Error::InvalidMesh("workspace has no objects".to_string()));
    }

    let strategy = ordering::strategy_for(workspace.config.object_ordering);
    let order = strategy.order(&workspace.objects)?;

    let layers = slicing::slice_workspace_with_progress(
        &workspace.objects,
        &order,
        &workspace.config,
        on_progress,
    )?;
    toolpath::plan(&layers, &workspace.objects, &workspace.config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_sane() {
        let cfg = SlicerConfig::default();
        assert!(cfg.layer_height > 0.0);
        assert!(cfg.nozzle_diameter > 0.0);
        assert!(cfg.wall_line_width > 0.0);
        assert!(cfg.shell_thickness > 0.0);
        assert!(cfg.wall_offset > 0.0);
        assert_eq!(cfg.wall_count(), 1);
        assert!(cfg.infill_line_width > 0.0);
        assert!(cfg.infill_angle_deg > 0.0);
    }

    #[test]
    fn slice_to_gcode_rejects_empty_workspace() {
        let machine = crate::machine::Machine::new(
            crate::bounds::BoundingVolume::Sphere {
                center: glam::DVec3::ZERO,
                radius: 1.0,
            },
            Vec::new(),
        );
        let workspace = Workspace::new(Vec::new(), machine, SlicerConfig::default());

        let err = slice_to_gcode(&workspace).unwrap_err();
        assert!(matches!(err, Error::InvalidMesh(_)));
    }

    #[test]
    fn slice_to_gcode_slices_first_object() {
        let machine = crate::machine::Machine::new(
            crate::bounds::BoundingVolume::Sphere {
                center: glam::DVec3::ZERO,
                radius: 1.0,
            },
            Vec::new(),
        );
        let object = crate::object::Object::new(
            crate::ids::ObjectId(0),
            mesh::Mesh::default(),
            crate::ids::ToolId(0),
        );
        let workspace = Workspace::new(vec![object], machine, SlicerConfig::default());

        assert!(slice_to_gcode(&workspace).is_ok());
    }

    /// Unit cube spanning [0,1]^3 — same fixture pattern as
    /// `slicing.rs`'s `cube_mesh` (and `manifold-fidget`'s
    /// `mesh_sdf`/`contour` tests).
    fn cube_mesh() -> mesh::Mesh {
        let vertices = vec![
            glam::DVec3::new(0.0, 0.0, 0.0),
            glam::DVec3::new(1.0, 0.0, 0.0),
            glam::DVec3::new(1.0, 1.0, 0.0),
            glam::DVec3::new(0.0, 1.0, 0.0),
            glam::DVec3::new(0.0, 0.0, 1.0),
            glam::DVec3::new(1.0, 0.0, 1.0),
            glam::DVec3::new(1.0, 1.0, 1.0),
            glam::DVec3::new(0.0, 1.0, 1.0),
        ];
        let indices = vec![
            0, 2, 1, 0, 3, 2, // -Z
            4, 5, 6, 4, 6, 7, // +Z
            0, 1, 5, 0, 5, 4, // -Y
            3, 7, 6, 3, 6, 2, // +Y
            0, 4, 7, 0, 7, 3, // -X
            1, 2, 6, 1, 6, 5, // +X
        ];
        mesh::Mesh::new(vertices, indices)
    }

    /// End-to-end: slicing a real solid (unit cube) through
    /// `slice_to_gcode` must produce non-trivial Gcode with real
    /// extrusion moves — not just the placeholder header. This is the
    /// first test exercising the full pipeline (slicing -> toolpath ->
    /// gcode) with a mesh that actually has contour geometry, so it
    /// would meaningfully fail if the pipeline regressed to producing
    /// empty layers/paths again.
    #[test]
    fn slice_to_gcode_produces_real_extrusion_moves_for_a_solid_cube() {
        // Arrange: a unit cube, one tool, and a layer height that steps
        // evenly across the cube's Z extent [0, 1] (5 layers: 0.0, 0.25,
        // 0.5, 0.75, 1.0 — matching slicing.rs's equivalent fixture/test).
        let machine = crate::machine::Machine::new(
            crate::bounds::BoundingVolume::Aabb {
                min: glam::DVec3::new(-10.0, -10.0, -10.0),
                max: glam::DVec3::new(10.0, 10.0, 10.0),
            },
            Vec::new(),
        );
        let object =
            crate::object::Object::new(crate::ids::ObjectId(0), cube_mesh(), crate::ids::ToolId(0));
        let config = SlicerConfig {
            layer_height: 0.25,
            ..SlicerConfig::default()
        };
        let workspace = Workspace::new(vec![object], machine, config);

        // Act.
        let gcode = slice_to_gcode(&workspace).unwrap();

        // Assert: exactly one tool-select line (single object/tool), and
        // at least 3 travel-to-loop-start moves ("G0 X") — one per
        // interior layer's single wall contour loop (the two exact-
        // boundary layers at Z=0 and Z=1 sample directly on the mesh
        // surface and produce no contour — see slicing.rs's
        // `slice_mesh_produces_nonempty_contour_loops_for_a_solid_cube`).
        // Infill (see `infill` module) appends further "G0 X" travel
        // moves between scan-line segments, so the count is a lower bound
        // rather than an exact match now that infill is generated by
        // default; a regressed/placeholder pipeline (empty layers or
        // empty paths) would still produce fewer than 3, so this remains
        // a meaningful assertion.
        assert_eq!(gcode.matches("T0\n").count(), 1);
        let path_starts = gcode.matches("G0 X").count();
        assert!(
            path_starts >= 3,
            "expected at least one travel move per interior contour loop (3 layers with geometry), got {path_starts}"
        );
        // Each contour loop has more than one vertex, so real extrusion
        // ("G1") moves must follow each path's initial travel move.
        let extrusion_moves = gcode.matches("G1 X").count();
        assert!(
            extrusion_moves > 0,
            "expected non-zero G1 extrusion moves, got gcode:\n{gcode}"
        );
    }

    #[test]
    fn slice_to_gcode_handles_multiple_objects_and_tools() {
        let machine = crate::machine::Machine::new(
            crate::bounds::BoundingVolume::Sphere {
                center: glam::DVec3::ZERO,
                radius: 1.0,
            },
            Vec::new(),
        );
        let objects = vec![
            crate::object::Object::new(
                crate::ids::ObjectId(0),
                mesh::Mesh::default(),
                crate::ids::ToolId(0),
            ),
            crate::object::Object::new(
                crate::ids::ObjectId(1),
                mesh::Mesh::default(),
                crate::ids::ToolId(1),
            ),
        ];
        let workspace = Workspace::new(objects, machine, SlicerConfig::default());

        assert!(slice_to_gcode(&workspace).is_ok());
    }

    /// `plan_toolpaths` is the shared helper `slice_to_gcode` now builds on
    /// top of — verify it returns non-empty paths for a real solid, and
    /// that `slice_to_gcode`'s Gcode output is unaffected by routing
    /// through it (still produces the same real extrusion moves as before
    /// the refactor).
    #[test]
    fn plan_toolpaths_returns_paths_and_slice_to_gcode_output_is_unaffected() {
        let machine = crate::machine::Machine::new(
            crate::bounds::BoundingVolume::Aabb {
                min: glam::DVec3::new(-10.0, -10.0, -10.0),
                max: glam::DVec3::new(10.0, 10.0, 10.0),
            },
            Vec::new(),
        );
        let object =
            crate::object::Object::new(crate::ids::ObjectId(0), cube_mesh(), crate::ids::ToolId(0));
        let config = SlicerConfig {
            layer_height: 0.25,
            ..SlicerConfig::default()
        };
        let workspace = Workspace::new(vec![object], machine, config);

        let paths = plan_toolpaths(&workspace).unwrap();
        assert!(!paths.is_empty(), "expected non-empty planned toolpaths");

        let gcode = slice_to_gcode(&workspace).unwrap();
        assert_eq!(gcode.matches("T0\n").count(), 1);
        assert!(gcode.matches("G0 X").count() >= 3);
        assert!(gcode.matches("G1 X").count() > 0);
    }
}
