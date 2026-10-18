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
        /// computing vertex normals and initializing OpenVDB SDF & Eikonal grids.
        pub fn register_triangle_mesh(
            name: &str,
            positions: &[f32],
            indices: &[u32],
        ) -> DecomposedPartInfo;

        /// Retrieves the centered local-space vertex buffer for `object_id`.
        pub fn get_object_vertices(object_id: u64) -> Vec<RenderVertex>;

        /// Retrieves the triangle index buffer for `object_id`.
        pub fn get_object_indices(object_id: u64) -> Vec<u32>;

        /// Removes `object_id` from the C++ backend without disturbing other objects.
        pub fn remove_cad_object(object_id: u64) -> bool;

        /// Exports the original OCCT B-Rep `TopoDS_Shape` for `object_id` to a STEP file.
        pub fn export_object_step_to_path(object_id: u64, path: &str) -> bool;

        /// Lists all scalar `openvdb::FloatGrid` field names available on `object_id`
        /// (e.g., `"sdf"`, `"eikonal"`, `"overhang"`).
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
mod tests {
    use super::ffi::RenderVertex;

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
}
