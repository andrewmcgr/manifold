#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rust/cxx.h"

namespace manifold {

struct RenderVertex;
struct DecomposedPartInfo;

/// Core C++20 compute engine handling CAD B-Rep ingestion (OCCT), assembly
/// decomposition, surface tessellation, STL/3MF triangle mesh registration,
/// and volumetric Eikonal distance fields (OpenVDB).
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
    /// part in its local centered frame, and initializes its OpenVDB Eikonal grid.
    std::vector<DecomposedPartInfo> loadStepAssembly(
        const std::string& filePath,
        double linearDeflection = 0.05,
        double angularDeflection = 0.5);

    /// Registers an indexed triangle mesh (e.g., from STL or 3MF), computes
    /// area-weighted normals, centers local coordinates, and initializes its
    /// OpenVDB Eikonal FloatGrid.
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
bool remove_cad_object(std::uint64_t object_id);
bool export_object_step_to_path(std::uint64_t object_id, rust::Str path);

} // namespace manifold
