use bytemuck::{Pod, Zeroable};

#[cxx::bridge(namespace = "manifold")]
pub mod ffi {
    /// Shared C++/Rust vertex representation laid out contiguously in memory
    /// (`#[repr(C)]` guaranteed by `cxx` for shared structs) so that slices of
    /// `RenderVertex` can be cast directly to raw GPU upload bytes via `bytemuck`.
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct RenderVertex {
        pub x: f32,
        pub y: f32,
        pub z: f32,
        pub nx: f32,
        pub ny: f32,
        pub nz: f32,
    }

    /// Metadata returned by the C++ backend for each decomposed solid/part
    /// in a STEP assembly or registered STL/3MF triangle mesh.
    #[derive(Clone, Debug, Default, PartialEq)]
    pub struct DecomposedPartInfo {
        pub object_id: u64,
        pub name: String,
        pub origin_x: f32,
        pub origin_y: f32,
        pub origin_z: f32,
        pub has_brep: bool,
    }

    /// Summary statistics for a named `openvdb::FloatGrid` scalar field on an object.
    #[derive(Clone, Debug, Default, PartialEq)]
    pub struct FieldStats {
        pub valid: bool,
        pub field_name: String,
        pub active_voxel_count: u64,
        pub min_value: f32,
        pub max_value: f32,
        pub background_value: f32,
        pub voxel_size: f32,
    }

    /// Triangle mesh extracted from any scalar `openvdb::FloatGrid` via `openvdb::tools::volumeToMesh`.
    /// For non-SDF fields, triangles straddling the object surface ($\phi_{\text{sdf}} = 0$) are
    /// split along the exact intersection contour so the interior ($\phi_{\text{sdf}} \le 0$)
    /// and exterior ($\phi_{\text{sdf}} > 0$) portions can be independently styled or clipped.
    #[derive(Clone, Debug, Default, PartialEq)]
    pub struct IsoSurfaceMesh {
        pub vertices: Vec<RenderVertex>,
        pub vertex_sdf: Vec<f32>,
        pub indices: Vec<u32>,
        pub interior_indices: Vec<u32>,
        pub exterior_indices: Vec<u32>,
        pub intersection_contour: Vec<f32>,
    }

    /// 2D uniform grid sample of any scalar `openvdb::FloatGrid` across an axis-aligned
    /// cross-section plane (`axis`: 0 = YZ/X-slice, 1 = XZ/Y-slice, 2 = XY/Z-slice).
    #[derive(Clone, Debug, Default, PartialEq)]
    pub struct FieldSlice2D {
        pub valid: bool,
        pub width: u32,
        pub height: u32,
        pub origin: [f32; 3],
        pub u_vec: [f32; 3],
        pub v_vec: [f32; 3],
        pub normal: [f32; 3],
        pub plane_coord: f32,
        pub min_value: f32,
        pub max_value: f32,
        pub background_value: f32,
        pub values: Vec<f32>,
        pub sdf_values: Vec<f32>,
    }

    /// Orthonormal right-handed basis of the nozzle's local coordinate frame in bed coordinates:
    /// - `nozzle_axis`: unit vector pointing opposite to the nozzle discharge axis (local +Z).
    /// - `travel_axis`: unit vector pointing in the direction of travel (local +X).
    /// - `lateral_axis`: unit vector from the right-hand rule (`nozzle_axis × travel_axis`, local +Y).
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct NozzleBasis {
        pub nozzle_axis: [f32; 3],
        pub travel_axis: [f32; 3],
        pub lateral_axis: [f32; 3],
    }

    /// A single `(X, Z)` pair in a tool's revolved clearance profile in the nozzle's local frame:
    /// `x_radius_mm` is the radial distance from the nozzle center and `z_height_mm` is the Z
    /// distance above the nozzle tip along `nozzle_axis`.
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct ClearanceXZ {
        pub x_radius_mm: f32,
        pub z_height_mm: f32,
    }

    /// Machine extrusion tool definition shared across the C++/Rust boundary.
    #[derive(Clone, Debug, Default, PartialEq)]
    pub struct ToolSpec {
        pub id: u32,
        pub name: String,
        pub nozzle_diameter_mm: f32,
        pub temperature_celsius: f32,
        pub has_rotational_axes: bool,
        pub max_angle_from_bed_normal_deg: f32,
        /// `0` = `PiecewiseXZ`, `1` = `ParametricCone`, `2` = `RectilinearCone`
        pub clearance_mode: u8,
        pub clearance_profile_xz: Vec<ClearanceXZ>,
        pub cone_tip_radius_mm: f32,
        pub cone_half_angle_deg: f32,
        pub cone_height_mm: f32,
        pub rect_tip_half_width_x_mm: f32,
        pub rect_tip_half_width_y_mm: f32,
        pub rect_half_angle_x_deg: f32,
        pub rect_half_angle_y_deg: f32,
        pub rect_height_mm: f32,
    }

    /// Variable layer height specification (`target_mm`, `min_mm`, `max_mm`, `first_layer_mm`, `top_layer_mm`)
    /// for non-planar slicing.
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct LayerHeightSpec {
        pub target_mm: f32,
        pub min_mm: f32,
        pub max_mm: f32,
        pub first_layer_mm: f32,
        pub top_layer_mm: f32,
    }

    /// Metadata and extracted geometry for a single selected slicing isosurface layer.
    #[derive(Clone, Debug, Default, PartialEq)]
    pub struct SelectedLayerInfo {
        pub layer_index: u32,
        pub branch_id: u32,
        pub parent_branch_id: i32,
        pub branch_layer_index: u32,
        pub branch_layer_count: u32,
        pub isovalue: f32,
        pub mean_thickness_mm: f32,
        pub min_thickness_mm: f32,
        pub max_thickness_mm: f32,
        pub is_first_layer: bool,
        pub is_top_layer: bool,
        pub vertices: Vec<RenderVertex>,
        pub indices: Vec<u32>,
        pub contour_segments: Vec<f32>,
    }

    /// Result of bidirectional branch-aware slicing isosurface selection on an object's Eikonal field.
    #[derive(Clone, Debug, Default, PartialEq)]
    pub struct SelectedIsosurfacesResult {
        pub valid: bool,
        pub branch_count: u32,
        pub layers: Vec<SelectedLayerInfo>,
    }

    /// Spatial/surface orientation hint (`kind`: 0 = Tangency, 1 = Orthogonality)
    /// guiding the iterative constrained Eikonal & tool-orientation solver.
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct SpatialOrientationHint {
        pub kind: u8,
        pub position_bed: [f32; 3],
        pub direction_bed: [f32; 3],
        pub influence_radius_mm: f32,
        pub weight: f32,
        pub apply_on_surface_only: bool,
    }

    /// Overhang-driven conformal vs. orthogonal Eikonal surface orientation bias parameters
    /// shared across the C++/Rust boundary.
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct OverhangBiasSpec {
        pub enabled: bool,
        /// Overall soft blend weight in `[0.0, 1.0]` for the overhang-driven surface orientation bias.
        pub weight: f32,
        /// Supported surface angle threshold from vertical (degrees, e.g. `30.0` for 30° supported,
        /// corresponding to signed overhang angle `-30°`). Surfaces between `-supported_angle_deg`
        /// and `+overhang_angle_deg` bias toward orthogonal ($\nabla \tau \perp \hat{\mathbf{n}}_{\text{surf}}$);
        /// shallower supported surfaces ($< -\text{supported\_angle\_deg}$) bias toward conformal.
        pub orthogonal_supported_angle_deg: f32,
        /// Overhang surface angle threshold past vertical (degrees, e.g. `60.0` for 60° overhang,
        /// corresponding to signed overhang angle `+60°`). Steeper overhangs ($> +\text{overhang\_angle\_deg}$)
        /// bias toward conformal ($\nabla \tau \parallel -\hat{\mathbf{n}}_{\text{surf}}$).
        pub orthogonal_overhang_angle_deg: f32,
        /// Smooth cubic Hermite (`smoothstep`) transition width (degrees, e.g. `15.0`) around
        /// `-orthogonal_supported_angle_deg` and `+orthogonal_overhang_angle_deg`.
        pub transition_width_deg: f32,
    }

    /// Lock-free progress snapshot for background OpenVDB & Eikonal field computation.
    #[derive(Clone, Debug, Default, PartialEq)]
    pub struct FieldComputeProgress {
        pub running: bool,
        pub fraction: f32,
        pub stage: String,
    }

    /// Configurable variable extrusion line width `(target_mm, min_mm, max_mm)` shared across FFI.
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct LineWidthSpec {
        pub target_mm: f32,
        pub min_mm: f32,
        pub max_mm: f32,
    }

    /// Support detection, order-field seeding, and support toolpath generation parameters shared across FFI.
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct SupportSpec {
        pub enabled: bool,
        pub seed_order_field: bool,
        /// When true, only generate support for regions that are actually causally unreachable by
        /// the constrained order field (floating islands / downward-hanging local minima) and
        /// two-pier bridges whose span exceeds `max_bridge_length_mm` (leaving reachable overhangs
        /// to non-planar conformal layers and inside-out wave overhangs).
        pub only_unreachable_and_long_bridges: bool,
        /// Maximum unsupported two-pier bridge length (mm); if `<= 0.0`, falls back to `ShellPathSpec::max_bridge_length_mm`.
        pub max_bridge_length_mm: f32,
        pub z_gap_mm: f32,
        pub xy_gap_mm: f32,
        pub interface_layers: u32,
        pub bed_interface_layers: u32,
        pub base_spacing_mm: f32,
        pub interface_spacing_mm: f32,
        pub line_width_mm: f32,
    }

    /// Shell, overhang (bridging & wave overhangs), and sparse infill path generation parameters shared across FFI.
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct ShellPathSpec {
        pub shell_thickness_mm: f32,
        pub inner_wall_count: u32,
        pub outer_wall_width: LineWidthSpec,
        pub inner_wall_width: LineWidthSpec,
        pub solid_infill_width: LineWidthSpec,
        /// `0 = Monotonic`, `1 = Concentric`
        pub solid_infill_pattern: u8,
        /// Fractional sparse infill density in `[0.0, 1.0]`.
        pub sparse_infill_density: f32,
        pub sparse_infill_width: LineWidthSpec,
        /// `0 = Monotonic`, `1 = Cubic`, `2 = Gyroid`, `3 = SchwarzD`, `4 = SchwarzP`, `5 = Concentric`, `6 = AllWalls`
        pub sparse_infill_pattern: u8,
        pub infill_angle_deg: f32,
        pub nozzle_diameter_mm: f32,
        pub min_extrusion_length_mm: f32,
        /// Overhang angle threshold past vertical (degrees, e.g. `45.0`) above which unsupported extrusions
        /// use straight-line circular bridging or inside-out wave overhangs.
        pub overhang_angle_threshold_deg: f32,
        /// Maximum unsupported straight-line bridge span length (mm).
        pub max_bridge_length_mm: f32,
        /// Lateral overlap fraction of standard line width (`0.08` = 8%) for inside-out wave overhang extrusions.
        pub wave_overhang_overlap_fraction: f32,
    }

    /// A single 3D vertex along a continuous non-planar extrusion toolpath.
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct ToolpathPoint {
        pub x: f32,
        pub y: f32,
        pub z: f32,
        /// Local unit `nozzle_axis` $\hat{\mathbf{z}}_{\text{nozzle}}$ pointing opposite to nozzle discharge.
        pub nx: f32,
        pub ny: f32,
        pub nz: f32,
        /// Local extruded line width (mm) along $\hat{\mathbf{y}}_{\text{nozzle}}$.
        pub width_mm: f32,
        /// Local layer thickness (mm) along $\hat{\mathbf{z}}_{\text{nozzle}}$.
        pub height_mm: f32,
    }

    /// A continuous open or closed extrusion path on a non-planar slicing isosurface.
    #[derive(Clone, Debug, Default, PartialEq)]
    pub struct ExtrusionPath {
        /// `0 = OuterWall`, `1 = InnerWall`, `2 = SolidInfill`, `3 = GapFill`, `4 = SparseInfill`, `5 = Bridge`, `6 = WaveOverhang`, `7 = Support`, `8 = SupportInterface`
        pub feature_kind: u8,
        /// `0` for outer wall, `1..=inner_wall_count` for inner walls, inside-out pass index for wave overhangs, pass index for infill/gap fill/bridges/supports.
        pub wall_index: u32,
        pub is_closed: bool,
        pub total_length_mm: f32,
        pub mean_width_mm: f32,
        pub points: Vec<ToolpathPoint>,
    }

    /// All shell and sparse infill toolpaths generated on a single selected slicing isosurface layer.
    #[derive(Clone, Debug, Default, PartialEq)]
    pub struct LayerToolpaths {
        pub layer_index: u32,
        pub branch_id: u32,
        pub isovalue: f32,
        pub paths: Vec<ExtrusionPath>,
    }

    /// Complete toolpath generation result for an object across all selected slicing isosurfaces.
    #[derive(Clone, Debug, Default, PartialEq)]
    pub struct ObjectToolpathsResult {
        pub valid: bool,
        pub total_path_count: u32,
        pub outer_wall_count: u32,
        pub inner_wall_count: u32,
        pub solid_infill_count: u32,
        pub sparse_infill_count: u32,
        pub bridge_count: u32,
        pub wave_overhang_count: u32,
        pub support_count: u32,
        pub support_interface_count: u32,
        pub layers: Vec<LayerToolpaths>,
    }

    unsafe extern "C++" {
        include!("slicer_backend.h");

        /// Legacy single-model STEP loader.
        pub fn load_cad_model(path: &str) -> bool;

        /// Retrieves the tessellated vertex array (`x, y, z, nx, ny, nz`) of the last loaded object.
        pub fn get_tessellated_vertices() -> Vec<RenderVertex>;

        /// Retrieves the tessellated triangle index buffer (`u32`) of the last loaded object.
        pub fn get_tessellated_indices() -> Vec<u32>;

        /// Loads a STEP file and decomposes assemblies into independent solid/shell objects.
        pub fn load_step_assembly(path: &str) -> Vec<DecomposedPartInfo>;

        /// Registers an indexed triangle mesh (from STL or 3MF) in the C++ backend,
        /// computing vertex normals and centering local coordinates (without computing
        /// volumetric scalar fields on initial load).
        pub fn register_triangle_mesh(
            name: &str,
            positions: &[f32],
            indices: &[u32],
        ) -> DecomposedPartInfo;

        /// Retrieves the centered local-space vertex buffer for `object_id`.
        pub fn get_object_vertices(object_id: u64) -> Vec<RenderVertex>;

        /// Retrieves the triangle index buffer for `object_id`.
        pub fn get_object_indices(object_id: u64) -> Vec<u32>;

        /// Retrieves the per-triangle STEP B-Rep `TopoDS_Face` index for `object_id`
        /// (empty for pure triangle meshes).
        pub fn get_object_triangle_face_ids(object_id: u64) -> Vec<u32>;

        /// Retrieves the solved 3D unit tool axis (`[ax, ay, az]` per vertex in local object
        /// coordinates, length `3 * vertex_count`) sampled on the object's surface mesh from
        /// `toolAxisGrid`, or an empty `Vec` if `toolAxisGrid` has not been computed yet.
        pub fn get_object_vertex_tool_axes(object_id: u64) -> Vec<f32>;

        /// Retrieves the per-vertex `"support_needed"` scalar (`[0.0, 1.0]`, length `vertex_count`)
        /// sampled on the object's surface mesh, indicating regions unreachable by the constrained
        /// Eikonal order field without support.
        pub fn get_object_vertex_support_needed(object_id: u64) -> Vec<f32>;

        /// Removes `object_id` from the C++ backend without disturbing other objects.
        pub fn remove_cad_object(object_id: u64) -> bool;

        /// Clones `source_object_id` in the C++ backend (including B-Rep shape, mesh, and
        /// any computed OpenVDB fields) and returns the new `object_id` (or `0` on failure).
        pub fn clone_cad_object(source_object_id: u64, clone_name: &str) -> u64;

        /// Exports the original OCCT B-Rep `TopoDS_Shape` for `object_id` to a STEP file.
        pub fn export_object_step_to_path(object_id: u64, path: &str) -> bool;

        /// Lists all scalar `openvdb::FloatGrid` field names available on `object_id`
        /// (e.g., `"sdf"`, `"eikonal"`, `"overhang"`, `"tool_tilt_deg"`).
        pub fn list_object_fields(object_id: u64) -> Vec<String>;

        /// Queries voxel count, min/max scalar range, and voxel size for a named field on `object_id`.
        pub fn get_object_field_stats(object_id: u64, field_name: &str) -> FieldStats;

        /// Recomputes the OpenVDB SDF and derived scalar fields for `object_id` at the
        /// specified `voxel_size` (mm) and `narrow_band_voxels` half-width.
        pub fn recompute_object_fields(
            object_id: u64,
            voxel_size: f32,
            narrow_band_voxels: f32,
        ) -> bool;

        /// Queries the current lock-free progress of background field computation.
        pub fn get_field_compute_progress() -> FieldComputeProgress;

        /// Extracts a 3D polygonal isosurface mesh at `isovalue` from any named scalar field
        /// on `object_id` using `openvdb::tools::volumeToMesh`.
        pub fn extract_field_isosurface(
            object_id: u64,
            field_name: &str,
            isovalue: f32,
            adaptivity: f32,
        ) -> IsoSurfaceMesh;

        /// Samples a 2D cross-section slice of any named scalar field on `object_id`
        /// along `axis` (0 = X, 1 = Y, 2 = Z) at `normalized_pos` in `[0.0, 1.0]`.
        pub fn sample_field_slice(
            object_id: u64,
            field_name: &str,
            axis: u8,
            normalized_pos: f32,
            resolution: u32,
        ) -> FieldSlice2D;

        /// Queries the right-handed orthonormal local frame of the nozzle at `position_bed`
        /// moving along `travel_dir_bed` in bed coordinates.
        pub fn query_nozzle_frame(
            position_bed: [f32; 3],
            travel_dir_bed: [f32; 3],
        ) -> NozzleBasis;

        /// Queries the unit gravity direction vector in bed coordinates at `position_bed`.
        pub fn query_gravity_direction(position_bed: [f32; 3]) -> [f32; 3];

        /// Sets the active nozzle axis orientation (pointing opposite to nozzle discharge) in bed coordinates.
        pub fn set_nozzle_axis_bed(axis_bed: [f32; 3]);

        /// Sets the active gravity direction vector in bed coordinates (e.g. when a 5-axis machine tilts the bed).
        pub fn set_gravity_direction_bed(gravity_bed: [f32; 3]);

        /// Synchronizes the machine's tool list and active tool index into the C++ backend.
        pub fn set_machine_tools(tools: &[ToolSpec], active_tool_index: u32);

        /// Retrieves the machine's configured tool list from the C++ backend.
        pub fn get_machine_tools() -> Vec<ToolSpec>;

        /// Retrieves the currently active tool index in the C++ backend.
        pub fn get_active_tool_index() -> u32;

        /// Sets the variable slice layer height (`target_mm`, `min_mm`, `max_mm`, `first_layer_mm`, `top_layer_mm`) in the C++ backend.
        pub fn set_slice_layer_height(layer_height: LayerHeightSpec);

        /// Retrieves the current variable slice layer height from the C++ backend.
        pub fn get_slice_layer_height() -> LayerHeightSpec;

        /// Sets the number of outer constraint & tool-orientation relaxation iterations for the Eikonal solver.
        pub fn set_eikonal_solver_iterations(iterations: u32);

        /// Retrieves the number of outer constraint & tool-orientation relaxation iterations.
        pub fn get_eikonal_solver_iterations() -> u32;

        /// Configures dual-grid multi-resolution volumetric field evaluation (coarse full-interior core
        /// + fine boundary-shell narrow band).
        pub fn set_multi_resolution_grid_config(enabled: bool, coarse_ratio: u32);

        /// Sets spatial/surface tangency and orthogonality hints for the Eikonal solver.
        pub fn set_spatial_orientation_hints(hints: &[SpatialOrientationHint]);

        /// Sets the overhang-driven conformal/orthogonal surface orientation bias parameters for the Eikonal solver.
        pub fn set_overhang_orientation_bias_spec(spec: OverhangBiasSpec);

        /// Retrieves the current overhang-driven conformal/orthogonal surface orientation bias parameters.
        pub fn get_overhang_orientation_bias_spec() -> OverhangBiasSpec;

        /// Selects the non-planar slicing isosurfaces for `object_id` using a bidirectional
        /// branch-aware relaxation algorithm honoring `min_mm`, `max_mm`, `target_mm`,
        /// `first_layer_mm`, and `top_layer_mm`, and extracts their interior-clipped 3D
        /// meshes and boundary contours.
        pub fn select_slicing_isosurfaces(
            object_id: u64,
            adaptivity: f32,
        ) -> SelectedIsosurfacesResult;

        /// Sets the shell & toolpath generation parameters (`ShellPathSpec`) in the C++ backend.
        pub fn set_shell_path_spec(spec: ShellPathSpec);

        /// Retrieves the current shell & toolpath generation parameters (`ShellPathSpec`) from the C++ backend.
        pub fn get_shell_path_spec() -> ShellPathSpec;

        /// Sets the support detection, order-field seeding, and support toolpath generation parameters (`SupportSpec`).
        pub fn set_support_spec(spec: SupportSpec);

        /// Retrieves the current support detection and support toolpath generation parameters (`SupportSpec`).
        pub fn get_support_spec() -> SupportSpec;

        /// Generates outer walls, inner walls, solid infill (`Monotonic` or `Concentric`), and
        /// shell gap fill across all selected non-planar slicing isosurfaces of `object_id`
        /// within the outer shell ($-T_{\text{shell}} \le \phi_{\text{sdf}}(\mathbf{x}) \le 0$).
        pub fn generate_object_shell_toolpaths(
            object_id: u64,
            adaptivity: f32,
        ) -> ObjectToolpathsResult;
    }
}

// SAFETY: `ffi::RenderVertex` is generated by `#[cxx::bridge]` with `#[repr(C)]`
// and consists exclusively of 6 contiguous `f32` fields (24 bytes total, no padding).
unsafe impl Zeroable for ffi::RenderVertex {}
unsafe impl Pod for ffi::RenderVertex {}

impl ffi::RenderVertex {
    /// Vertex buffer layout attributes (`position` at shader location 0,
    /// `normal` at shader location 1) matching the 24-byte C++ `RenderVertex`.
    pub const ATTRIBUTES: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3];

    /// Describes the stride and attribute offsets for a `wgpu::RenderPipeline`.
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES,
        }
    }
}

#[cfg(test)]
pub static TEST_BACKEND_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
mod tests {
    use super::ffi::{self, ClearanceXZ, LayerHeightSpec, RenderVertex, ToolSpec};

    #[test]
    fn verify_render_vertex_abi_layout() {
        assert_eq!(
            std::mem::size_of::<RenderVertex>(),
            6 * std::mem::size_of::<f32>()
        );
        assert_eq!(
            std::mem::align_of::<RenderVertex>(),
            std::mem::align_of::<f32>()
        );

        let sample = [RenderVertex {
            x: 1.0,
            y: 2.0,
            z: 3.0,
            nx: 0.0,
            ny: 0.0,
            nz: 1.0,
        }];
        let bytes: &[u8] = bytemuck::cast_slice(&sample);
        assert_eq!(bytes.len(), 24);
    }

    #[test]
    fn verify_kinematics_tools_and_layer_height_ffi() {
        let _guard = super::TEST_BACKEND_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        ffi::set_nozzle_axis_bed([0.0, 0.0, 1.0]);
        let basis = ffi::query_nozzle_frame([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        assert_eq!(basis.nozzle_axis, [0.0, 0.0, 1.0]);
        assert_eq!(basis.travel_axis, [1.0, 0.0, 0.0]);
        assert_eq!(basis.lateral_axis, [0.0, 1.0, 0.0]);

        ffi::set_gravity_direction_bed([0.0, -0.5, -0.8660254]);
        let g = ffi::query_gravity_direction([0.0, 0.0, 0.0]);
        assert!((g[1] + 0.5).abs() < 1e-5 && (g[2] + 0.8660254).abs() < 1e-5);
        ffi::set_gravity_direction_bed([0.0, 0.0, -1.0]);

        let tools = [
            ToolSpec {
                id: 0,
                name: "T0 - 0.4mm".to_string(),
                nozzle_diameter_mm: 0.40,
                temperature_celsius: 215.0,
                has_rotational_axes: true,
                max_angle_from_bed_normal_deg: 45.0,
                clearance_profile_xz: vec![
                    ClearanceXZ {
                        x_radius_mm: 0.5,
                        z_height_mm: 0.0,
                    },
                    ClearanceXZ {
                        x_radius_mm: 12.0,
                        z_height_mm: 15.0,
                    },
                ],
                ..Default::default()
            },
            ToolSpec {
                id: 1,
                name: "T1 - 0.8mm".to_string(),
                nozzle_diameter_mm: 0.80,
                temperature_celsius: 245.0,
                has_rotational_axes: false,
                max_angle_from_bed_normal_deg: 0.0,
                clearance_profile_xz: vec![
                    ClearanceXZ {
                        x_radius_mm: 1.0,
                        z_height_mm: 0.0,
                    },
                    ClearanceXZ {
                        x_radius_mm: 16.0,
                        z_height_mm: 20.0,
                    },
                ],
                ..Default::default()
            },
        ];
        ffi::set_machine_tools(&tools, 1);
        assert_eq!(ffi::get_active_tool_index(), 1);
        let read_tools = ffi::get_machine_tools();
        assert_eq!(read_tools.len(), 2);
        assert_eq!(read_tools[1].nozzle_diameter_mm, 0.80);
        assert_eq!(read_tools[1].temperature_celsius, 245.0);
        assert!(!read_tools[1].has_rotational_axes);
        assert!(read_tools[0].has_rotational_axes);
        assert!((read_tools[0].max_angle_from_bed_normal_deg - 45.0).abs() < 1e-6);
        assert_eq!(read_tools[1].clearance_profile_xz.len(), 2);

        ffi::set_slice_layer_height(LayerHeightSpec {
            target_mm: 0.22,
            min_mm: 0.08,
            max_mm: 0.36,
            first_layer_mm: 0.25,
            top_layer_mm: 0.15,
        });
        let lh = ffi::get_slice_layer_height();
        assert!((lh.target_mm - 0.22).abs() < 1e-6);
        assert!((lh.min_mm - 0.08).abs() < 1e-6);
        assert!((lh.max_mm - 0.36).abs() < 1e-6);
        assert!((lh.first_layer_mm - 0.25).abs() < 1e-6);
        assert!((lh.top_layer_mm - 0.15).abs() < 1e-6);

        ffi::set_eikonal_solver_iterations(4);
        assert_eq!(ffi::get_eikonal_solver_iterations(), 4);

        // Register a closed 12mm x 12mm x 16mm box mesh resting on the bed
        #[rustfmt::skip]
        let positions: [f32; 24] = [
            -6.0, -6.0,  0.0,
             6.0, -6.0,  0.0,
             6.0,  6.0,  0.0,
            -6.0,  6.0,  0.0,
            -6.0, -6.0, 16.0,
             6.0, -6.0, 16.0,
             6.0,  6.0, 16.0,
            -6.0,  6.0, 16.0,
        ];
        #[rustfmt::skip]
        let indices: [u32; 36] = [
            0, 2, 1,  0, 3, 2, // bottom (Z=0)
            4, 5, 6,  4, 6, 7, // top (Z=16)
            0, 1, 5,  0, 5, 4, // front
            1, 2, 6,  1, 6, 5, // right
            2, 3, 7,  2, 7, 6, // back
            3, 0, 4,  3, 4, 7, // left
        ];

        let info = ffi::register_triangle_mesh("TestBox", &positions, &indices);
        assert!(info.object_id > 0);
        // Fields must not be computed on initial object load
        assert!(ffi::list_object_fields(info.object_id).is_empty());

        // 1. With T1 active (has_rotational_axes = false), tool_tilt_deg must be 0 everywhere
        assert!(ffi::recompute_object_fields(info.object_id, 1.5, 4.0));
        let prog = ffi::get_field_compute_progress();
        assert!(!prog.running);
        assert!((prog.fraction - 1.0).abs() < 1e-4);

        let fields = ffi::list_object_fields(info.object_id);
        assert!(fields.contains(&"sdf".to_string()));
        assert!(fields.contains(&"eikonal".to_string()));
        assert!(fields.contains(&"overhang".to_string()));
        assert!(fields.contains(&"tool_tilt_deg".to_string()));

        let tilt_fixed = ffi::get_object_field_stats(info.object_id, "tool_tilt_deg");
        assert!(tilt_fixed.valid);
        assert!(tilt_fixed.max_value.abs() < 1e-4);

        // 2. Switch to T0 (has_rotational_axes = true, max_angle_from_bed_normal_deg = 30.0)
        //    and add an orthogonality orientation hint tilted at 45 degrees
        let rot_tools = [ToolSpec {
            id: 0,
            name: "T0 - 6-Axis".to_string(),
            nozzle_diameter_mm: 0.40,
            temperature_celsius: 215.0,
            has_rotational_axes: true,
            max_angle_from_bed_normal_deg: 30.0,
            clearance_profile_xz: vec![
                ClearanceXZ {
                    x_radius_mm: 0.5,
                    z_height_mm: 0.0,
                },
                ClearanceXZ {
                    x_radius_mm: 2.0,
                    z_height_mm: 2.5,
                },
                ClearanceXZ {
                    x_radius_mm: 6.0,
                    z_height_mm: 10.0,
                },
            ],
            ..Default::default()
        }];
        ffi::set_machine_tools(&rot_tools, 0);
        ffi::set_spatial_orientation_hints(&[super::ffi::SpatialOrientationHint {
            kind: 1, // Orthogonality
            position_bed: [0.0, 0.0, 12.0],
            direction_bed: [0.70710677, 0.0, 0.70710677],
            influence_radius_mm: 15.0,
            weight: 1.0,
            apply_on_surface_only: false,
        }]);

        assert!(ffi::recompute_object_fields(info.object_id, 1.5, 4.0));
        ffi::set_spatial_orientation_hints(&[]);

        let eik_stats = ffi::get_object_field_stats(info.object_id, "eikonal");
        assert!(eik_stats.valid);
        assert!(eik_stats.min_value >= -0.01 && eik_stats.min_value <= 2.0);
        assert!(eik_stats.max_value >= 12.0);

        // Verify tool tilt is clamped to <= 30.0 deg (max_angle_from_bed_normal_deg)
        let tilt_rot = ffi::get_object_field_stats(info.object_id, "tool_tilt_deg");
        assert!(tilt_rot.valid);
        assert!(tilt_rot.max_value <= 30.05);

        // Extract an intermediate Eikonal isosurface and verify it intersects the object cleanly
        let iso = ffi::extract_field_isosurface(info.object_id, "eikonal", 8.0, 0.0);
        assert!(!iso.interior_indices.is_empty());

        // Clone the object in the backend and verify independent geometry & fields
        let cloned_id = ffi::clone_cad_object(info.object_id, "TestBox (2)");
        assert!(cloned_id > 0 && cloned_id != info.object_id);
        assert_eq!(
            ffi::get_object_vertices(cloned_id).len(),
            ffi::get_object_vertices(info.object_id).len()
        );
        assert!(ffi::get_object_field_stats(cloned_id, "eikonal").valid);

        let _ = ffi::remove_cad_object(info.object_id);
        assert!(ffi::get_object_field_stats(cloned_id, "eikonal").valid);
        let _ = ffi::remove_cad_object(cloned_id);
    }
}
