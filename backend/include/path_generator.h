#pragma once

#include <array>
#include <cstddef>
#include <functional>
#include <openvdb/openvdb.h>

#include "slicer_backend.h"

namespace manifold {

/// Generates non-planar outer walls, inner walls, solid infill (`Monotonic` or `Concentric`),
/// shell gap fill within the object's outer shell ($-T_{\text{shell}} \le \phi_{\text{sdf}}(\mathbf{x}) \le 0$),
/// and non-planar sparse infill (`Monotonic`, `Cubic`, `Gyroid`, `SchwarzD`, `SchwarzP`,
/// `Concentric`, `AllWalls`) within the interior core ($\phi_{\text{sdf}}(\mathbf{x}) < -T_{\text{shell}}$).
///
/// Key invariants:
/// - Walls are generated inwards on each 3D slicing isosurface by insetting from $\phi_{\text{sdf}} = 0$
///   and may be either closed loops or open polylines where the shell boundary $\phi_{\text{sdf}} = -T_{\text{shell}}$
///   or local medial clearance is reached.
/// - Outer walls, inner walls, and solid infill each adapt their local line width within their
///   configured `[min_mm, target_mm, max_mm]` range.
/// - Continuous extrusions shorter than `spec.min_extrusion_length_mm` are filtered out, while
///   individual discretization segments within a continuous extrusion may be shorter.
/// - Every toolpath vertex queries the solved 5/6-axis `toolAxisGrid` (or `defaultNozzleAxisBed`)
///   without ever assuming nozzle = +Z.
ObjectToolpathsResult generateShellToolpathsForLayers(
    const SelectedIsosurfacesResult& selectedLayers,
    const openvdb::FloatGrid& sdfGrid,
    const openvdb::FloatGrid& eikGrid,
    const openvdb::Vec3SGrid* toolAxisGrid,
    const std::array<float, 3>& defaultNozzleAxisBed,
    const LayerHeightSpec& layerHeightSpec,
    const ShellPathSpec& spec,
    const std::function<void(std::size_t, std::size_t)>& progressCb = nullptr,
    const openvdb::FloatGrid* coarseSdfGrid = nullptr,
    const openvdb::FloatGrid* coarseEikGrid = nullptr,
    const openvdb::Vec3SGrid* coarseToolAxisGrid = nullptr,
    float fineShellDepthMm = 0.0f,
    const openvdb::FloatGrid* overhangGrid = nullptr,
    const openvdb::FloatGrid* coarseOverhangGrid = nullptr,
    const std::array<float, 3>& defaultGravityDirectionBed = {0.0f, 0.0f, -1.0f},
    const ToolSpec* activeTool = nullptr,
    float objectMinAxialBed = 0.0f);

} // namespace manifold
