#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <openvdb/openvdb.h>

#include "slicer_backend.h"

namespace manifold {

/// Extensible spatial/surface orientation hint for guiding non-planar layer normals
/// and 5/6-axis tool orientations.
enum class HintKind : std::uint8_t {
    /// Isosurface should be tangent to `directionBed` (i.e. \nabla \tau \perp directionBed).
    Tangency = 0,
    /// Isosurface normal should align with `directionBed` (i.e. \nabla \tau \parallel directionBed).
    Orthogonality = 1,
};

struct OrientationHint {
    HintKind kind = HintKind::Orthogonality;
    std::array<float, 3> positionBed{0.0f, 0.0f, 0.0f};
    std::array<float, 3> directionBed{0.0f, 0.0f, 1.0f};
    float influenceRadiusMm = 10.0f;
    float weight = 1.0f;
    bool applyOnSurfaceOnly = true;
};

/// Mode for computing the local 5/6-axis tool orientation vector \hat{a}(x).
enum class ToolOrientationMode : std::uint8_t {
    /// Follow the constrained Eikonal gradient \nabla \tau(x) (default).
    FollowEikonalGradient = 0,
    /// Prefer tool axis mostly perpendicular to the object boundary (\nabla \phi_sdf).
    MostlyPerpendicularToSurface = 1,
    /// Prefer tool axis mostly parallel to the object boundary (tangent to \phi_sdf = 0).
    MostlyParallelToSurface = 2,
};

/// Compressed Sparse Row (CSR) wavefront schedule grouping mutually non-adjacent
/// active voxels into parallelizable level sets.
struct WavefrontSchedule {
    std::vector<std::size_t> indices;
    std::vector<std::size_t> levelOffsets;

    std::size_t numLevels() const noexcept {
        return levelOffsets.empty() ? 0 : (levelOffsets.size() - 1);
    }
};

/// Flattened representation of all active voxels in the object's OpenVDB domain,
/// providing O(1) 6-neighbor indexing and parallel hyperplane wavefront schedules
/// for multi-threaded TBB fast sweeping, constraint projection, and causal island removal.
struct ActiveVolumeDomain {
    double voxelSize = 0.5;
    float minZBed = 0.0f;
    float maxZBed = 0.0f;
    float maxSpanMm = 1.0f;
    openvdb::math::Transform::Ptr transform;

    /// Active voxel coordinates and world-space (bed-coordinate) centers.
    std::vector<openvdb::Coord> coords;
    std::vector<std::array<float, 3>> posBed;

    /// Signed distance field (\phi <= 0 inside object, \phi > 0 outside).
    std::vector<float> sdf;
    /// Unit outward SDF gradient (\nabla \phi / |\nabla \phi|).
    std::vector<std::array<float, 3>> sdfGrad;
    /// Gravity-aware overhang scalar (-\nabla \phi \cdot \hat{g}_{bed} \in [-1, 1]).
    std::vector<float> overhang;

    /// Solved Eikonal scalar field \tau(x) (mm).
    std::vector<float> tau;
    /// Local isotropic slowness s(x) > 0 for the Eikonal PDE |\nabla \tau| = s(x).
    std::vector<float> slowness;
    /// Preferred unit propagation direction \hat{d}_{target}(x) for directional constraints.
    std::vector<std::array<float, 3>> targetDir;
    /// Solved unit tool axis \hat{a}(x) in bed coordinates (pointing opposite to nozzle discharge).
    std::vector<std::array<float, 3>> toolAxis;
    /// Angle (degrees) between toolAxis[i] and the reference bed normal.
    std::vector<float> toolTiltDeg;

    /// True if voxel i is a fixed seed on the print bed (\tau = 0).
    std::vector<std::uint8_t> isSeed;
    /// True if voxel i belongs to the object interior or immediate boundary (\phi <= 0.75 * dx).
    std::vector<std::uint8_t> isInterior;

    /// 6-connected neighbor indices (-X, +X, -Y, +Y, -Z, +Z), or -1 if inactive.
    std::vector<std::array<std::int32_t, 6>> neighbors6;
    /// OpenVDB Int32Grid mapping active (i, j, k) coordinates to [0, N-1] (-1 if inactive).
    openvdb::Int32Grid::Ptr indexGrid;

    /// 4 diagonal hyperplane schedules (L = +-i +- j + k) for lock-free TBB 8-octant fast sweeping.
    std::array<WavefrontSchedule, 4> octantWavefronts;
    /// Z-slice Red-Black schedule for parallel upward clearance cone enforcement.
    WavefrontSchedule zParityWavefront;

    /// Look up active voxel index by OpenVDB Coord (-1 if not in active domain).
    std::int32_t findVoxelIndex(const openvdb::Coord& ijk) const;

    /// Builds the flattened index and 6-neighbor connectivity table from `sdfGrid` in parallel.
    void initializeFromSdfGrid(
        const openvdb::FloatGrid& sdfGrid,
        float objectMinZ,
        float objectMaxZ,
        float objectMaxSpan);

    /// Builds O(N) counting-sort hyperplane wavefront schedules for all non-seed interior voxels.
    void buildWavefrontSchedules();
};

/// Context passed to tool orientation strategies and Eikonal constraints.
struct EikonalSolverContext {
    const SlicerBackend& backend;
    const ToolSpec& activeTool;
    const LayerHeightSpec& layerHeight;
    std::uint32_t outerIteration = 0;
    std::uint32_t totalIterations = 4;
    ToolOrientationMode orientationMode = ToolOrientationMode::FollowEikonalGradient;
    std::vector<OrientationHint> hints;
    std::function<void(float, const char*)> reportProgress;
};

/// Pluggable strategy for computing and clamping the local 5/6-axis tool orientation
/// field \hat{a}(x) across the active voxel domain.
class ToolOrientationStrategy {
public:
    virtual ~ToolOrientationStrategy() = default;

    /// Computes or updates `domain.toolAxis` and `domain.toolTiltDeg` for iteration `ctx.outerIteration`.
    virtual void updateToolOrientations(
        ActiveVolumeDomain& domain,
        const EikonalSolverContext& ctx) const;
};

/// Abstract base class for modular Eikonal constraints (tool clearance, overhang anti-island,
/// surface/spatial tangency & orthogonality hints, etc.).
class EikonalConstraint {
public:
    virtual ~EikonalConstraint() = default;
    virtual std::string name() const = 0;

    /// Called before each fast-sweeping pass to adjust local slowness `s(x)` or `targetDir(x)`.
    virtual void applyPreSweepModifiers(
        ActiveVolumeDomain& domain,
        const EikonalSolverContext& ctx) const = 0;

    /// Called after each fast-sweeping pass to project/enforce hard geometric or topological
    /// inequalities directly on `domain.tau`.
    virtual void enforceOnField(
        ActiveVolumeDomain& domain,
        const EikonalSolverContext& ctx) const = 0;
};

/// Enforces the active tool's XZ clearance profile so that the revolved clearance envelope
/// oriented along `domain.toolAxis[i]` never intersects the print bed or already-printed
/// isosurfaces (\tau(y) <= \tau(x)).
class ToolClearanceConstraint final : public EikonalConstraint {
public:
    std::string name() const override { return "ToolClearanceConstraint"; }

    void applyPreSweepModifiers(
        ActiveVolumeDomain& domain,
        const EikonalSolverContext& ctx) const override;

    void enforceOnField(
        ActiveVolumeDomain& domain,
        const EikonalSolverContext& ctx) const override;

    /// Computes the maximum allowable layer slope angle \alpha_{max} (radians) relative to the
    /// plane perpendicular to the tool axis from a tool's XZ clearance profile.
    static float computeMaxClearanceSlopeRad(const ToolSpec& tool) noexcept;
};

/// Uses the gravity-aware `"overhang"` field to accelerate outward layer growth into
/// overhanging regions and enforces strict single-minimum causal monotonicity from the
/// bed contact seeds so there are zero disconnected isosurface islands off the bed.
class OverhangIslandConstraint final : public EikonalConstraint {
public:
    std::string name() const override { return "OverhangIslandConstraint"; }

    void applyPreSweepModifiers(
        ActiveVolumeDomain& domain,
        const EikonalSolverContext& ctx) const override;

    void enforceOnField(
        ActiveVolumeDomain& domain,
        const EikonalSolverContext& ctx) const override;
};

/// Applies spatial/surface tangency and orthogonality hints to the Eikonal propagation field.
class OrientationHintConstraint final : public EikonalConstraint {
public:
    std::string name() const override { return "OrientationHintConstraint"; }

    void applyPreSweepModifiers(
        ActiveVolumeDomain& domain,
        const EikonalSolverContext& ctx) const override;

    void enforceOnField(
        ActiveVolumeDomain& domain,
        const EikonalSolverContext& ctx) const override;
};

/// Result grids produced by the iterative constrained Eikonal solver.
struct EikonalSolveResult {
    openvdb::FloatGrid::Ptr eikonalGrid;
    openvdb::FloatGrid::Ptr overhangGrid;
    openvdb::FloatGrid::Ptr toolTiltDegGrid;
    openvdb::Vec3SGrid::Ptr toolAxisGrid;
};

/// Runs the full iterative constrained Eikonal + tool-orientation solver on `sdfGrid`.
EikonalSolveResult solveConstrainedEikonal(
    const openvdb::FloatGrid& sdfGrid,
    float objectMinZ,
    float objectMaxZ,
    float objectMaxSpan,
    float exteriorBandMm,
    const EikonalSolverContext& baseContext);

} // namespace manifold
