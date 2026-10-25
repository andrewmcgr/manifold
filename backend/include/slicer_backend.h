#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rust/cxx.h"

namespace manifold {

struct RenderVertex;
struct DecomposedPartInfo;
struct FieldStats;
struct IsoSurfaceMesh;
struct FieldSlice2D;
struct NozzleBasis;
struct ClearanceXZ;
struct ToolSpec;
struct LayerHeightSpec;
struct SelectedLayerInfo;
struct SelectedIsosurfacesResult;
struct SpatialOrientationHint;
struct FieldComputeProgress;
struct LineWidthSpec;
struct ShellPathSpec;
struct ToolpathPoint;
struct ExtrusionPath;
struct LayerToolpaths;
struct ObjectToolpathsResult;

/// Core C++20 compute engine handling CAD B-Rep ingestion (OCCT), assembly
/// decomposition, surface tessellation, STL/3MF triangle mesh registration,
/// OpenVDB Signed Distance Fields (SDF), generic scalar field visualization,
/// and 5/6-axis nozzle & gravity kinematics.
class SlicerBackend {
public:
    SlicerBackend();
    ~SlicerBackend();

    SlicerBackend(const SlicerBackend&) = delete;
    SlicerBackend& operator=(const SlicerBackend&) = delete;

    /// Legacy single-model loader retained for backward compatibility.
    bool loadStepFile(const std::string& filePath, double linearDeflection = 0.05, double angularDeflection = 0.5);

    /// Loads an ISO 10303-21 STEP file, decomposes any assembly hierarchy
    /// (compounds/solids/shells) into independent objects, tessellates each
    /// part in its local centered frame, and computes its OpenVDB SDF & scalar fields.
    std::vector<DecomposedPartInfo> loadStepAssembly(
        const std::string& filePath,
        double linearDeflection = 0.05,
        double angularDeflection = 0.5);

    /// Registers an indexed triangle mesh (e.g., from STL or 3MF), computes
    /// area-weighted normals, centers local coordinates, and computes its
    /// OpenVDB SDF & scalar fields.
    DecomposedPartInfo registerTriangleMesh(
        const std::string& name,
        const float* positions,
        std::size_t positionFloatsCount,
        const std::uint32_t* indices,
        std::size_t indexCount);

    /// Removes a single object from the backend registry without disturbing
    /// any other objects (including sibling parts from the same STEP assembly).
    bool removeObject(std::uint64_t objectId);

    /// Exports an object's underlying OCCT B-Rep TopoDS_Shape to a STEP file
    /// (returns false if the object originated from a pure triangle mesh).
    bool exportObjectStep(std::uint64_t objectId, const std::string& outFilePath) const;

    /// Returns the tessellated vertices for a specific object ID.
    std::vector<RenderVertex> objectVertices(std::uint64_t objectId) const;

    /// Returns the tessellated triangle indices for a specific object ID.
    std::vector<std::uint32_t> objectIndices(std::uint64_t objectId) const;

    /// Returns the per-triangle STEP B-Rep TopoDS_Face index for `objectId` (empty for pure triangle meshes).
    std::vector<std::uint32_t> objectTriangleFaceIds(std::uint64_t objectId) const;

    /// Returns the solved 3D unit tool axis (`3 * vertexBuffer.size()` floats `[ax, ay, az, ...]`)
    /// sampled at each surface vertex of `objectId` from `toolAxisGrid` (or empty if unavailable).
    std::vector<float> objectVertexToolAxes(std::uint64_t objectId) const;

    /// Lists all named OpenVDB scalar fields stored on `objectId`.
    std::vector<std::string> listObjectFields(std::uint64_t objectId) const;

    /// Queries statistics (active voxel count, min/max value, voxel size) for a named field.
    FieldStats getObjectFieldStats(std::uint64_t objectId, const std::string& fieldName) const;

    /// Rebuilds the OpenVDB SDF and derived scalar fields (`eikonal`, `overhang`, `tool_tilt_deg`)
    /// for `objectId` at the requested `voxelSize` and `narrowBandVoxels`.
    bool recomputeObjectFields(std::uint64_t objectId, float voxelSize, float narrowBandVoxels);

    /// Extracts an isosurface triangle mesh from any named scalar field on `objectId`
    /// using `openvdb::tools::volumeToMesh`.
    IsoSurfaceMesh extractFieldIsosurface(
        std::uint64_t objectId,
        const std::string& fieldName,
        float isovalue,
        float adaptivity) const;

    /// Samples a 2D cross-section plane from any named scalar field on `objectId`
    /// using trilinear interpolation (`openvdb::tools::GridSampler`).
    FieldSlice2D sampleFieldSlice(
        std::uint64_t objectId,
        const std::string& fieldName,
        std::uint8_t axis,
        float normalizedPos,
        std::uint32_t resolution) const;

    /// Queries the nozzle's local orthonormal right-handed basis at `positionBed`
    /// for a toolhead moving in direction `travelDirBed` (in bed coordinates).
    NozzleBasis queryNozzleFrame(
        const std::array<float, 3>& positionBed,
        const std::array<float, 3>& travelDirBed) const noexcept;

    /// Queries the unit gravity direction vector in bed coordinates at `positionBed`.
    std::array<float, 3> queryGravityDirection(
        const std::array<float, 3>& positionBed) const noexcept;

    /// Updates the nozzle axis orientation (pointing opposite to nozzle discharge) in bed coordinates.
    void setNozzleAxisBed(const std::array<float, 3>& axisBed) noexcept;

    /// Updates the gravity direction vector in bed coordinates.
    void setGravityDirectionBed(const std::array<float, 3>& gravityBed) noexcept;

    /// Replaces the machine's tool list and sets the active tool index.
    void setMachineTools(const std::vector<ToolSpec>& tools, std::uint32_t activeToolIndex);

    /// Returns the configured list of machine tools.
    const std::vector<ToolSpec>& machineTools() const noexcept;

    /// Returns the active tool index.
    std::uint32_t activeToolIndex() const noexcept;

    /// Sets the variable layer height (`target_mm`, `min_mm`, `max_mm`, `first_layer_mm`, `top_layer_mm`) for slicing.
    void setSliceLayerHeight(const LayerHeightSpec& spec) noexcept;

    /// Returns the current variable layer height specification.
    LayerHeightSpec sliceLayerHeight() const noexcept;

    /// Sets the number of outer constraint & tool-orientation relaxation iterations for the Eikonal solver.
    void setEikonalSolverIterations(std::uint32_t iterations) noexcept;

    /// Returns the configured number of outer Eikonal solver iterations.
    std::uint32_t eikonalSolverIterations() const noexcept;

    /// Configures dual-grid multi-resolution volumetric field evaluation (coarse full-interior core
    /// + fine boundary-shell narrow band).
    void setMultiResolutionConfig(bool enabled, std::uint32_t coarseRatio) noexcept;

    /// Returns whether dual-grid multi-resolution field evaluation is enabled.
    bool multiResolutionEnabled() const noexcept;

    /// Returns the coarse-to-fine voxel ratio used in dual-grid multi-resolution mode.
    std::uint32_t multiResolutionCoarseRatio() const noexcept;

    /// Sets spatial/surface tangency and orthogonality hints for the Eikonal solver.
    void setSpatialOrientationHints(const std::vector<SpatialOrientationHint>& hints);

    /// Returns the configured spatial/surface orientation hints.
    const std::vector<SpatialOrientationHint>& spatialOrientationHints() const noexcept;

    /// Selects the non-planar slicing isosurfaces for `objectId` using a bidirectional
    /// branch-aware relaxation algorithm honoring `min_mm`, `max_mm`, `target_mm`,
    /// `first_layer_mm`, and `top_layer_mm`, and extracts their interior-clipped 3D
    /// meshes and boundary contours.
    SelectedIsosurfacesResult selectSlicingIsosurfaces(
        std::uint64_t objectId,
        float adaptivity) const;

    /// Sets the shell & toolpath generation parameters (`ShellPathSpec`).
    void setShellPathSpec(const ShellPathSpec& spec) noexcept;

    /// Returns the current shell & toolpath generation parameters (`ShellPathSpec`).
    ShellPathSpec shellPathSpec() const noexcept;

    /// Generates outer walls, inner walls, solid infill (`Monotonic` or `Concentric`),
    /// and shell gap fill across all selected non-planar slicing isosurfaces of `objectId`
    /// within the outer shell ($-T_{\text{shell}} \le \phi_{\text{sdf}}(\mathbf{x}) \le 0$).
    ObjectToolpathsResult generateObjectShellToolpaths(
        std::uint64_t objectId,
        float adaptivity) const;

    /// Returns a reference to the most recently loaded object's vertices.
    const std::vector<RenderVertex>& vertices() const noexcept;

    /// Returns a reference to the most recently loaded object's indices.
    const std::vector<std::uint32_t>& indices() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ============================================================================
// CXX FFI Free-Function Entry Points (declared in the shared namespace)
// ============================================================================

bool load_cad_model(rust::Str path);
rust::Vec<RenderVertex> get_tessellated_vertices();
rust::Vec<std::uint32_t> get_tessellated_indices();

rust::Vec<DecomposedPartInfo> load_step_assembly(rust::Str path);
DecomposedPartInfo register_triangle_mesh(
    rust::Str name,
    rust::Slice<const float> positions,
    rust::Slice<const std::uint32_t> indices);
rust::Vec<RenderVertex> get_object_vertices(std::uint64_t object_id);
rust::Vec<std::uint32_t> get_object_indices(std::uint64_t object_id);
rust::Vec<std::uint32_t> get_object_triangle_face_ids(std::uint64_t object_id);
rust::Vec<float> get_object_vertex_tool_axes(std::uint64_t object_id);
bool remove_cad_object(std::uint64_t object_id);
bool export_object_step_to_path(std::uint64_t object_id, rust::Str path);

rust::Vec<rust::String> list_object_fields(std::uint64_t object_id);
FieldStats get_object_field_stats(std::uint64_t object_id, rust::Str field_name);
bool recompute_object_fields(std::uint64_t object_id, float voxel_size, float narrow_band_voxels);
FieldComputeProgress get_field_compute_progress();
IsoSurfaceMesh extract_field_isosurface(
    std::uint64_t object_id,
    rust::Str field_name,
    float isovalue,
    float adaptivity);
FieldSlice2D sample_field_slice(
    std::uint64_t object_id,
    rust::Str field_name,
    std::uint8_t axis,
    float normalized_pos,
    std::uint32_t resolution);

NozzleBasis query_nozzle_frame(
    std::array<float, 3> position_bed,
    std::array<float, 3> travel_dir_bed);
std::array<float, 3> query_gravity_direction(std::array<float, 3> position_bed);
void set_nozzle_axis_bed(std::array<float, 3> axis_bed);
void set_gravity_direction_bed(std::array<float, 3> gravity_bed);
void set_machine_tools(rust::Slice<const ToolSpec> tools, std::uint32_t active_tool_index);
rust::Vec<ToolSpec> get_machine_tools();
std::uint32_t get_active_tool_index();
void set_slice_layer_height(LayerHeightSpec layer_height);
LayerHeightSpec get_slice_layer_height();
void set_eikonal_solver_iterations(std::uint32_t iterations);
std::uint32_t get_eikonal_solver_iterations();
void set_multi_resolution_grid_config(bool enabled, std::uint32_t coarse_ratio);
void set_spatial_orientation_hints(rust::Slice<const SpatialOrientationHint> hints);
SelectedIsosurfacesResult select_slicing_isosurfaces(std::uint64_t object_id, float adaptivity);
void set_shell_path_spec(ShellPathSpec spec);
ShellPathSpec get_shell_path_spec();
ObjectToolpathsResult generate_object_shell_toolpaths(std::uint64_t object_id, float adaptivity);

} // namespace manifold
