#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rust/cxx.h"

namespace manifold {

struct RenderVertex;

/// Core C++20 compute engine handling CAD B-Rep ingestion (OCCT), surface
/// tessellation, and volumetric Eikonal distance fields (OpenVDB).
class SlicerBackend {
public:
    SlicerBackend();
    ~SlicerBackend();

    SlicerBackend(const SlicerBackend&) = delete;
    SlicerBackend& operator=(const SlicerBackend&) = delete;

    /// Loads an ISO 10303-21 STEP file into an Open CASCADE TopoDS_Shape
    /// and immediately tessellates its faces for GPU rendering and voxelization.
    bool loadStepFile(const std::string& filePath, double linearDeflection = 0.05, double angularDeflection = 0.5);

    /// Initializes an OpenVDB FloatGrid encoding a stubbed Eikonal / geodesic
    /// scalar field over the active model's bounding volume.
    void initializeEikonalField(double voxelSize = 0.25);

    /// Returns a reference to the contiguous CPU buffer of tessellated vertices.
    const std::vector<RenderVertex>& vertices() const noexcept;

    /// Returns a reference to the contiguous CPU buffer of triangle indices.
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

} // namespace manifold
