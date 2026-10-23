#include "slicer_backend.h"
#include "eikonal_solver.h"
#include "path_generator.h"
#include "manifold-frontend/src/bridge.rs.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <filesystem>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <queue>
#include <set>
#include <unordered_map>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

// Open CASCADE (OCCT) headers
#include <BRepBndLib.hxx>
#include <BRepMesh_IncrementalMesh.hxx>
#include <BRep_Builder.hxx>
#include <BRep_Tool.hxx>
#include <Bnd_Box.hxx>
#include <IFSelect_ReturnStatus.hxx>
#include <Poly_Triangulation.hxx>
#include <STEPControl_Reader.hxx>
#include <STEPControl_StepModelType.hxx>
#include <STEPControl_Writer.hxx>
#include <TopAbs_Orientation.hxx>
#include <TopAbs_ShapeEnum.hxx>
#include <TopExp_Explorer.hxx>
#include <TopLoc_Location.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Compound.hxx>
#include <TopoDS_Face.hxx>
#include <TopoDS_Iterator.hxx>
#include <TopoDS_Shape.hxx>
#include <gp_Dir.hxx>
#include <gp_Pnt.hxx>
#include <gp_Trsf.hxx>
#include <gp_Vec.hxx>

// OpenVDB headers
#include <openvdb/openvdb.h>
#include <openvdb/math/Transform.h>
#include <openvdb/tools/Interpolation.h>
#include <openvdb/tools/MeshToVolume.h>
#include <openvdb/tools/VolumeToMesh.h>

namespace manifold {

namespace {

struct ObjectRecord {
    std::uint64_t id = 0;
    std::string name;
    TopoDS_Shape shape;
    bool hasBrep = false;
    float originX = 0.0f;
    float originY = 0.0f;
    float originZ = 0.0f;
    float minX = 0.0f, minY = 0.0f, minZ = 0.0f;
    float maxX = 0.0f, maxY = 0.0f, maxZ = 0.0f;
    std::vector<RenderVertex> vertexBuffer;
    std::vector<std::uint32_t> indexBuffer;
    /// Per-triangle STEP B-Rep TopoDS_Face index (1 entry per triangle when `hasBrep == true`).
    std::vector<std::uint32_t> triangleFaceIds;
    /// Generic dictionary of named scalar fields (e.g., "sdf", "eikonal", "overhang", "tool_tilt_deg").
    std::map<std::string, openvdb::FloatGrid::Ptr> scalarFields;
    /// Solved 3D unit tool axis field \hat{a}(x) in bed coordinates.
    openvdb::Vec3SGrid::Ptr toolAxisGrid;
    /// Coarse full-interior grids when `isMultiResolution == true`.
    std::map<std::string, openvdb::FloatGrid::Ptr> coarseScalarFields;
    openvdb::Vec3SGrid::Ptr coarseToolAxisGrid;
    bool isMultiResolution = false;
    float fineShellInnerDepthMm = 0.0f;
    std::uint32_t coarseToFineRatio = 1;
    /// Cached selected slicing isosurfaces for this object's current Eikonal field & LayerHeightSpec.
    mutable bool hasCachedSelectedLayers = false;
    mutable float cachedSelectedLayersAdaptivity = -1.0f;
    mutable LayerHeightSpec cachedSelectedLayersSpec{};
    mutable SelectedIsosurfacesResult cachedSelectedLayers{};
};

void computeVertexBounds(
    const std::vector<RenderVertex>& vertices,
    float& minX, float& minY, float& minZ,
    float& maxX, float& maxY, float& maxZ)
{
    if (vertices.empty()) {
        minX = minY = minZ = 0.0f;
        maxX = maxY = maxZ = 0.0f;
        return;
    }

    minX = minY = minZ = std::numeric_limits<float>::max();
    maxX = maxY = maxZ = std::numeric_limits<float>::lowest();

    for (const auto& v : vertices) {
        minX = std::min(minX, v.x);
        minY = std::min(minY, v.y);
        minZ = std::min(minZ, v.z);
        maxX = std::max(maxX, v.x);
        maxY = std::max(maxY, v.y);
        maxZ = std::max(maxZ, v.z);
    }
}

/// Recursively decomposes a TopoDS_Shape assembly hierarchy (compounds and
/// compsolids) into independent leaf solids/shells while preserving composed
/// TopLoc_Location transforms.
void collectDecomposedLeaves(const TopoDS_Shape& shape, std::vector<TopoDS_Shape>& leaves) {
    if (shape.IsNull()) {
        return;
    }

    const TopAbs_ShapeEnum stype = shape.ShapeType();
    if (stype == TopAbs_COMPOUND || stype == TopAbs_COMPSOLID) {
        bool hasSubSolidsOrCompounds = false;
        BRep_Builder builder;
        TopoDS_Compound looseFacesCompound;
        builder.MakeCompound(looseFacesCompound);
        bool hasLooseFaces = false;

        for (TopoDS_Iterator it(shape, Standard_True, Standard_True); it.More(); it.Next()) {
            const TopoDS_Shape& child = it.Value();
            if (child.IsNull()) {
                continue;
            }
            const TopAbs_ShapeEnum ctype = child.ShapeType();
            if (ctype == TopAbs_COMPOUND || ctype == TopAbs_COMPSOLID ||
                ctype == TopAbs_SOLID || ctype == TopAbs_SHELL) {
                hasSubSolidsOrCompounds = true;
                collectDecomposedLeaves(child, leaves);
            } else if (ctype == TopAbs_FACE) {
                builder.Add(looseFacesCompound, child);
                hasLooseFaces = true;
            }
        }

        if (!hasSubSolidsOrCompounds) {
            leaves.push_back(shape);
        } else if (hasLooseFaces) {
            leaves.push_back(looseFacesCompound);
        }
    } else if (stype == TopAbs_SOLID || stype == TopAbs_SHELL || stype == TopAbs_FACE) {
        leaves.push_back(shape);
    }
}

/// Centers a vertex buffer in-place so that its XY centroid is at (0, 0) and
/// its minimum Z is at 0.0, returning the original world offset (originX, originY, originZ).
void recenterVertexBuffer(ObjectRecord& rec) {
    if (rec.vertexBuffer.empty()) {
        rec.originX = rec.originY = rec.originZ = 0.0f;
        rec.minX = rec.minY = rec.minZ = 0.0f;
        rec.maxX = rec.maxY = rec.maxZ = 0.0f;
        return;
    }

    float minX = 0.0f, minY = 0.0f, minZ = 0.0f;
    float maxX = 0.0f, maxY = 0.0f, maxZ = 0.0f;
    computeVertexBounds(rec.vertexBuffer, minX, minY, minZ, maxX, maxY, maxZ);

    rec.originX = 0.5f * (minX + maxX);
    rec.originY = 0.5f * (minY + maxY);
    rec.originZ = minZ;

    for (auto& v : rec.vertexBuffer) {
        v.x -= rec.originX;
        v.y -= rec.originY;
        v.z -= rec.originZ;
    }

    computeVertexBounds(
        rec.vertexBuffer,
        rec.minX, rec.minY, rec.minZ,
        rec.maxX, rec.maxY, rec.maxZ);
}

/// Computes the right-handed orthonormal local frame of the nozzle:
/// - `nozzle_axis` (local +Z): unit vector pointing opposite to the nozzle discharge direction.
/// - `travel_axis` (local +X): unit vector pointing in the direction of toolhead travel.
/// - `lateral_axis` (local +Y): `nozzle_axis × travel_axis` so `travel_axis × lateral_axis = nozzle_axis`.
NozzleBasis computeNozzleBasis(
    const std::array<float, 3>& nozzleAxisInput,
    const std::array<float, 3>& travelDirInput) noexcept
{
    auto normalize3 = [](float x, float y, float z, const std::array<float, 3>& fallback) -> std::array<float, 3> {
        const float len = std::sqrt(x * x + y * y + z * z);
        if (len > 1.0e-7f) {
            return {x / len, y / len, z / len};
        }
        return fallback;
    };

    const auto zNozzle = normalize3(
        nozzleAxisInput[0],
        nozzleAxisInput[1],
        nozzleAxisInput[2],
        {0.0f, 0.0f, 1.0f});

    // Project travelDirInput orthogonal to zNozzle
    const float dotTZ =
        travelDirInput[0] * zNozzle[0] +
        travelDirInput[1] * zNozzle[1] +
        travelDirInput[2] * zNozzle[2];
    float tx = travelDirInput[0] - dotTZ * zNozzle[0];
    float ty = travelDirInput[1] - dotTZ * zNozzle[1];
    float tz = travelDirInput[2] - dotTZ * zNozzle[2];
    float tLen = std::sqrt(tx * tx + ty * ty + tz * tz);

    if (tLen <= 1.0e-6f) {
        // Deterministic fallback orthogonal to zNozzle
        const std::array<float, 3> refAxis =
            (std::abs(zNozzle[0]) < 0.9f) ? std::array<float, 3>{1.0f, 0.0f, 0.0f}
                                          : std::array<float, 3>{0.0f, 1.0f, 0.0f};
        const float dotRef =
            refAxis[0] * zNozzle[0] +
            refAxis[1] * zNozzle[1] +
            refAxis[2] * zNozzle[2];
        tx = refAxis[0] - dotRef * zNozzle[0];
        ty = refAxis[1] - dotRef * zNozzle[1];
        tz = refAxis[2] - dotRef * zNozzle[2];
        tLen = std::sqrt(tx * tx + ty * ty + tz * tz);
    }

    const std::array<float, 3> xTravel = (tLen > 1.0e-7f)
        ? std::array<float, 3>{tx / tLen, ty / tLen, tz / tLen}
        : std::array<float, 3>{1.0f, 0.0f, 0.0f};

    // Right-hand rule: yLateral = zNozzle × xTravel
    const float lx = zNozzle[1] * xTravel[2] - zNozzle[2] * xTravel[1];
    const float ly = zNozzle[2] * xTravel[0] - zNozzle[0] * xTravel[2];
    const float lz = zNozzle[0] * xTravel[1] - zNozzle[1] * xTravel[0];
    const auto yLateral = normalize3(lx, ly, lz, {0.0f, 1.0f, 0.0f});

    NozzleBasis basis{};
    basis.nozzle_axis = zNozzle;
    basis.travel_axis = xTravel;
    basis.lateral_axis = yLateral;
    return basis;
}

std::mutex g_backendMutex;
std::atomic<bool> g_fieldComputeRunning{false};
std::atomic<float> g_fieldComputeFraction{0.0f};
std::mutex g_fieldProgressStageMutex;
std::string g_fieldComputeStage{"Idle"};

void setFieldProgress(bool running, float fraction, const char* stageMsg) {
    g_fieldComputeRunning.store(running, std::memory_order_relaxed);
    g_fieldComputeFraction.store(std::clamp(fraction, 0.0f, 1.0f), std::memory_order_relaxed);
    if (stageMsg != nullptr) {
        std::lock_guard<std::mutex> lock(g_fieldProgressStageMutex);
        g_fieldComputeStage = stageMsg;
    }
}

FieldComputeProgress getFieldProgressSnapshot() {
    FieldComputeProgress out{};
    out.running = g_fieldComputeRunning.load(std::memory_order_relaxed);
    out.fraction = g_fieldComputeFraction.load(std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lock(g_fieldProgressStageMutex);
        out.stage = rust::String(g_fieldComputeStage);
    }
    return out;
}

/// Populates the object's scalar and vector field registry using the iterative
/// constrained 3D Eikonal + tool-orientation solver:
/// 1. `"sdf"`: Full-interior + exterior-band Signed Distance Field via
///    `openvdb::tools::meshToSignedDistanceField`.
/// 2. `"eikonal"`: Iterative, constraint-driven non-planar layer field propagated from
///    the bed contact seeds through the object interior, subject to the active tool's
///    XZ clearance profile, rotational axis limits, and anti-island overhang causality.
/// 3. `"overhang"`: Normalized SDF gradient projected against the queried gravity vector
///    ($-\nabla \phi \cdot \hat{\mathbf{g}}_{\text{bed}} \in [-1, 1]$).
/// 4. `"tool_tilt_deg"`: Solved 5/6-axis tool tilt angle (degrees from bed normal).
void buildObjectScalarFields(
    ObjectRecord& rec,
    const SlicerBackend& backend,
    float requestedVoxelSize = 0.5f,
    float narrowBandVoxels = 6.0f)
{
    rec.scalarFields.clear();
    rec.toolAxisGrid.reset();
    rec.coarseScalarFields.clear();
    rec.coarseToolAxisGrid.reset();
    rec.isMultiResolution = false;
    rec.fineShellInnerDepthMm = 0.0f;
    rec.coarseToFineRatio = 1;
    rec.hasCachedSelectedLayers = false;
    if (rec.vertexBuffer.empty() || rec.indexBuffer.size() < 3) {
        return;
    }

    setFieldProgress(true, 0.04f, "Voxelizing mesh & building OpenVDB Signed Distance Field...");

    const float spanX = rec.maxX - rec.minX;
    const float spanY = rec.maxY - rec.minY;
    const float spanZ = rec.maxZ - rec.minZ;
    const float maxSpan = std::max({spanX, spanY, spanZ, 1.0f});
    const float minBoxSpan = std::max(0.1f, std::min({spanX, spanY, spanZ}));

    std::vector<openvdb::Vec3s> points;
    points.reserve(rec.vertexBuffer.size());
    for (const auto& v : rec.vertexBuffer) {
        points.emplace_back(v.x, v.y, v.z);
    }

    std::vector<openvdb::Vec3I> triangles;
    triangles.reserve(rec.indexBuffer.size() / 3);
    for (std::size_t t = 0; t + 2 < rec.indexBuffer.size(); t += 3) {
        triangles.emplace_back(
            rec.indexBuffer[t + 0],
            rec.indexBuffer[t + 1],
            rec.indexBuffer[t + 2]);
    }
    std::vector<openvdb::Vec4I> quads;

    // Prepare solver context from the backend's active tool, layer height, and hints
    const auto& tools = backend.machineTools();
    const std::uint32_t activeIdx = backend.activeToolIndex();
    ToolSpec activeTool{};
    if (!tools.empty()) {
        activeTool = tools[std::min<std::size_t>(activeIdx, tools.size() - 1)];
    } else {
        activeTool.id = 0;
        activeTool.name = rust::String("T0 - 0.4mm Standard");
        activeTool.nozzle_diameter_mm = 0.40f;
        activeTool.temperature_celsius = 215.0f;
        activeTool.has_rotational_axes = true;
        activeTool.max_angle_from_bed_normal_deg = 45.0f;
    }

    const LayerHeightSpec localLayerHeight = backend.sliceLayerHeight();
    EikonalSolverContext ctx{
        backend,
        activeTool,
        localLayerHeight,
        0,
        backend.eikonalSolverIterations(),
        ToolOrientationMode::FollowEikonalGradient,
        {},
        [](float frac, const char* msg) {
            setFieldProgress(true, frac, msg);
        },
    };

    for (const auto& ffiHint : backend.spatialOrientationHints()) {
        OrientationHint h{};
        h.kind = (ffiHint.kind == 0) ? HintKind::Tangency : HintKind::Orthogonality;
        h.positionBed = ffiHint.position_bed;
        h.directionBed = ffiHint.direction_bed;
        h.influenceRadiusMm = ffiHint.influence_radius_mm;
        h.weight = ffiHint.weight;
        h.applyOnSurfaceOnly = ffiHint.apply_on_surface_only;
        ctx.hints.push_back(h);
    }

    const float shellThicknessMm = std::max(0.4f, backend.shellPathSpec().shell_thickness_mm);
    const bool multiResConfig =
        backend.multiResolutionEnabled() && (backend.multiResolutionCoarseRatio() >= 2);

    if (multiResConfig) {
        const std::uint32_t ratio =
            std::clamp<std::uint32_t>(backend.multiResolutionCoarseRatio(), 2, 8);
        // Because the fine grid only allocates a narrow boundary shell in dual-grid mode
        // (or a sparse thin-wall shell for thin-walled parts), allow fine voxel sizes down to
        // maxSpan / 800.0 for large print volumes.
        const double fineVoxelSize = std::max<double>(
            std::max(requestedVoxelSize, 0.05f),
            static_cast<double>(maxSpan) / 800.0);
        const double coarseVoxelSize = std::max<double>(
            fineVoxelSize * static_cast<double>(ratio),
            static_cast<double>(maxSpan) / 180.0);
        const float inBandFineMm = std::max({
            shellThicknessMm + 2.5f * static_cast<float>(coarseVoxelSize),
            8.0f * static_cast<float>(fineVoxelSize),
        });
        const double estUniformFineVoxels =
            (static_cast<double>(spanX) * static_cast<double>(spanY) * static_cast<double>(spanZ)) /
            (fineVoxelSize * fineVoxelSize * fineVoxelSize);

        if (estUniformFineVoxels >= 120000.0 && coarseVoxelSize >= 1.5 * fineVoxelSize) {
            setFieldProgress(
                true,
                0.06f,
                "Building fine boundary-shell OpenVDB SDF grid (narrow band)...");

            const float exBandFineVoxels = std::clamp(
                std::max(narrowBandVoxels, 6.0f),
                4.0f,
                32.0f);
            const float exBandFineMm = exBandFineVoxels * static_cast<float>(fineVoxelSize);
            const float inBandFineVoxels = std::clamp(
                std::ceil(inBandFineMm / static_cast<float>(fineVoxelSize)),
                8.0f,
                96.0f);

            openvdb::math::Transform::Ptr fineXform =
                openvdb::math::Transform::createLinearTransform(fineVoxelSize);
            openvdb::FloatGrid::Ptr fineSdfGrid =
                openvdb::tools::meshToSignedDistanceField<openvdb::FloatGrid>(
                    *fineXform,
                    points,
                    triangles,
                    quads,
                    exBandFineVoxels,
                    inBandFineVoxels);
            fineSdfGrid->setName("sdf");

            // Check whether the part actually has a deep interior core beyond `inBandFineMm`.
            // For thin-walled or medium-walled curved parts (even with a very large bounding box),
            // `deepCoreFloorVoxels == 0` because 100% of the solid interior already fits inside
            // the sparse narrow-band `fineSdfGrid`. Solving directly on `fineSdfGrid` avoids
            // coarse-grid under-sampling of thin walls while retaining O(Surface Area) sparsity.
            const float innerFloorCutoff = -std::max(
                inBandFineMm - 1.75f * static_cast<float>(fineVoxelSize),
                2.0f * static_cast<float>(fineVoxelSize));
            std::size_t deepCoreFloorVoxels = 0;
            for (auto iter = fineSdfGrid->cbeginValueOn(); iter; ++iter) {
                if (*iter <= innerFloorCutoff) {
                    ++deepCoreFloorVoxels;
                }
            }

            if (deepCoreFloorVoxels < 128) {
                rec.scalarFields["sdf"] = fineSdfGrid;
                EikonalSolveResult fineDirectSolved = solveConstrainedEikonal(
                    *fineSdfGrid,
                    rec.minZ,
                    rec.maxZ,
                    maxSpan,
                    exBandFineMm,
                    ctx);
                rec.scalarFields["eikonal"] = fineDirectSolved.eikonalGrid;
                rec.scalarFields["overhang"] = fineDirectSolved.overhangGrid;
                rec.scalarFields["tool_tilt_deg"] = fineDirectSolved.toolTiltDegGrid;
                rec.toolAxisGrid = fineDirectSolved.toolAxisGrid;
                rec.isMultiResolution = false;
                rec.fineShellInnerDepthMm = 0.0f;
                rec.coarseToFineRatio = 1;
                return;
            }

            if (minBoxSpan > 2.2f * inBandFineMm) {
                setFieldProgress(
                    true,
                    0.18f,
                    "Building coarse full-interior OpenVDB SDF grid (multi-resolution)...");

                const float minExteriorMm = std::max(
                    0.20f * maxSpan,
                    narrowBandVoxels * static_cast<float>(coarseVoxelSize));
                const float exBandCoarseVoxels = std::clamp(
                    minExteriorMm / static_cast<float>(coarseVoxelSize),
                    3.0f,
                    90.0f);
                const float inBandCoarseVoxels = std::clamp(
                    (0.60f * maxSpan) / static_cast<float>(coarseVoxelSize),
                    4.0f,
                    180.0f);

                openvdb::math::Transform::Ptr coarseXform =
                    openvdb::math::Transform::createLinearTransform(coarseVoxelSize);
                openvdb::FloatGrid::Ptr coarseSdfGrid =
                    openvdb::tools::meshToSignedDistanceField<openvdb::FloatGrid>(
                        *coarseXform,
                        points,
                        triangles,
                        quads,
                        exBandCoarseVoxels,
                        inBandCoarseVoxels);
                coarseSdfGrid->setName("sdf");
                rec.coarseScalarFields["sdf"] = coarseSdfGrid;

                EikonalSolverContext coarseCtx = ctx;
                coarseCtx.reportProgress = [](float frac, const char* msg) {
                    setFieldProgress(true, 0.22f + 0.32f * frac, msg);
                };
                EikonalSolveResult coarseSolved = solveConstrainedEikonal(
                    *coarseSdfGrid,
                    rec.minZ,
                    rec.maxZ,
                    maxSpan,
                    minExteriorMm,
                    coarseCtx);

                rec.coarseScalarFields["eikonal"] = coarseSolved.eikonalGrid;
                rec.coarseScalarFields["overhang"] = coarseSolved.overhangGrid;
                rec.coarseScalarFields["tool_tilt_deg"] = coarseSolved.toolTiltDegGrid;
                rec.coarseToolAxisGrid = coarseSolved.toolAxisGrid;

                rec.scalarFields["sdf"] = fineSdfGrid;

                EikonalSolveResult fineSolved = solveFineShellWithCoarseGuidance(
                    *fineSdfGrid,
                    coarseSolved,
                    inBandFineMm,
                    rec.minZ,
                    rec.maxZ,
                    maxSpan,
                    exBandFineMm,
                    ctx);

                rec.scalarFields["eikonal"] = fineSolved.eikonalGrid;
                rec.scalarFields["overhang"] = fineSolved.overhangGrid;
                rec.scalarFields["tool_tilt_deg"] = fineSolved.toolTiltDegGrid;
                rec.toolAxisGrid = fineSolved.toolAxisGrid;
                rec.isMultiResolution = true;
                rec.fineShellInnerDepthMm = inBandFineMm;
                rec.coarseToFineRatio = static_cast<std::uint32_t>(
                    std::max(2.0, std::round(coarseVoxelSize / fineVoxelSize)));
                return;
            }
        }
    }

    // Single-resolution path for compact or thin-walled parts
    const double voxelSize = std::max<double>(
        std::max(requestedVoxelSize, 0.05f),
        static_cast<double>(maxSpan) / 450.0);

    // Maintain a consistent physical exterior band (at least 20% of maxSpan or narrowBandVoxels * voxelSize)
    // and a full interior band (60% of maxSpan) so the SDF isovalue range in millimeters
    // never shrinks when the user selects a finer voxel resolution.
    const float minExteriorMm =
        std::max(0.20f * maxSpan, narrowBandVoxels * static_cast<float>(voxelSize));
    const float exBandVoxels = std::clamp(
        minExteriorMm / static_cast<float>(voxelSize),
        3.0f,
        150.0f);
    const float inBandVoxels = std::clamp(
        (0.60f * maxSpan) / static_cast<float>(voxelSize),
        4.0f,
        300.0f);

    openvdb::math::Transform::Ptr xform =
        openvdb::math::Transform::createLinearTransform(voxelSize);

    // 1. Compute full-interior + exterior Signed Distance Field ("sdf")
    openvdb::FloatGrid::Ptr sdfGrid =
        openvdb::tools::meshToSignedDistanceField<openvdb::FloatGrid>(
            *xform, points, triangles, quads, exBandVoxels, inBandVoxels);
    sdfGrid->setName("sdf");
    rec.scalarFields["sdf"] = sdfGrid;

    // 2. Solve iterative constrained 3D Eikonal + tool orientation fields
    EikonalSolveResult solved = solveConstrainedEikonal(
        *sdfGrid,
        rec.minZ,
        rec.maxZ,
        maxSpan,
        minExteriorMm,
        ctx);

    rec.scalarFields["eikonal"] = solved.eikonalGrid;
    rec.scalarFields["overhang"] = solved.overhangGrid;
    rec.scalarFields["tool_tilt_deg"] = solved.toolTiltDegGrid;
    rec.toolAxisGrid = solved.toolAxisGrid;
}

void computeMeshNormals(
    std::vector<RenderVertex>& vertexBuffer,
    const std::vector<std::uint32_t>& indexBuffer)
{
    const std::size_t numVertices = vertexBuffer.size();
    for (auto& v : vertexBuffer) {
        v.nx = 0.0f;
        v.ny = 0.0f;
        v.nz = 0.0f;
    }

    for (std::size_t t = 0; t + 2 < indexBuffer.size(); t += 3) {
        const std::uint32_t idx0 = indexBuffer[t + 0];
        const std::uint32_t idx1 = indexBuffer[t + 1];
        const std::uint32_t idx2 = indexBuffer[t + 2];
        if (idx0 >= numVertices || idx1 >= numVertices || idx2 >= numVertices) {
            continue;
        }

        const auto& v0 = vertexBuffer[idx0];
        const auto& v1 = vertexBuffer[idx1];
        const auto& v2 = vertexBuffer[idx2];

        const float e1x = v1.x - v0.x;
        const float e1y = v1.y - v0.y;
        const float e1z = v1.z - v0.z;
        const float e2x = v2.x - v0.x;
        const float e2y = v2.y - v0.y;
        const float e2z = v2.z - v0.z;

        const float cx = e1y * e2z - e1z * e2y;
        const float cy = e1z * e2x - e1x * e2z;
        const float cz = e1x * e2y - e1y * e2x;

        for (const std::uint32_t idx : {idx0, idx1, idx2}) {
            vertexBuffer[idx].nx += cx;
            vertexBuffer[idx].ny += cy;
            vertexBuffer[idx].nz += cz;
        }
    }

    for (auto& v : vertexBuffer) {
        const float len = std::sqrt(v.nx * v.nx + v.ny * v.ny + v.nz * v.nz);
        if (len > 1.0e-7f) {
            v.nx /= len;
            v.ny /= len;
            v.nz /= len;
        } else {
            v.nz = 1.0f;
        }
    }
}

void tessellateTopoShape(
    const TopoDS_Shape& shape,
    double linearDeflection,
    double angularDeflection,
    std::vector<RenderVertex>& vertexBuffer,
    std::vector<std::uint32_t>& indexBuffer,
    std::vector<std::uint32_t>& triangleFaceIds)
{
    vertexBuffer.clear();
    indexBuffer.clear();
    triangleFaceIds.clear();

    if (shape.IsNull()) {
        return;
    }

    BRepMesh_IncrementalMesh mesher(shape, linearDeflection, Standard_False, angularDeflection, Standard_True);
    mesher.Perform();

    std::uint32_t currentFaceId = 0;
    for (TopExp_Explorer explorer(shape, TopAbs_FACE); explorer.More(); explorer.Next()) {
        const TopoDS_Face& face = TopoDS::Face(explorer.Current());
        TopLoc_Location location;
        const Handle(Poly_Triangulation)& triangulation = BRep_Tool::Triangulation(face, location);

        if (triangulation.IsNull()) {
            continue;
        }

        const gp_Trsf& trsf = location.Transformation();
        const bool isReversed = (face.Orientation() == TopAbs_REVERSED);
        const Standard_Integer numNodes = triangulation->NbNodes();
        const Standard_Integer numTriangles = triangulation->NbTriangles();

        const auto baseVertexIndex = static_cast<std::uint32_t>(vertexBuffer.size());
        vertexBuffer.resize(baseVertexIndex + static_cast<std::size_t>(numNodes));

        for (Standard_Integer i = 1; i <= numNodes; ++i) {
            const gp_Pnt transformedPt = triangulation->Node(i).Transformed(trsf);
            auto& dst = vertexBuffer[baseVertexIndex + static_cast<std::uint32_t>(i - 1)];
            dst.x = static_cast<float>(transformedPt.X());
            dst.y = static_cast<float>(transformedPt.Y());
            dst.z = static_cast<float>(transformedPt.Z());
            dst.nx = 0.0f;
            dst.ny = 0.0f;
            dst.nz = 0.0f;
        }

        indexBuffer.reserve(indexBuffer.size() + static_cast<std::size_t>(numTriangles) * 3);
        triangleFaceIds.reserve(triangleFaceIds.size() + static_cast<std::size_t>(numTriangles));
        for (Standard_Integer t = 1; t <= numTriangles; ++t) {
            Standard_Integer n1 = 0, n2 = 0, n3 = 0;
            triangulation->Triangle(t).Get(n1, n2, n3);

            if (isReversed) {
                std::swap(n2, n3);
            }

            const std::uint32_t idx0 = baseVertexIndex + static_cast<std::uint32_t>(n1 - 1);
            const std::uint32_t idx1 = baseVertexIndex + static_cast<std::uint32_t>(n2 - 1);
            const std::uint32_t idx2 = baseVertexIndex + static_cast<std::uint32_t>(n3 - 1);

            indexBuffer.push_back(idx0);
            indexBuffer.push_back(idx1);
            indexBuffer.push_back(idx2);
            triangleFaceIds.push_back(currentFaceId);

            const auto& v0 = vertexBuffer[idx0];
            const auto& v1 = vertexBuffer[idx1];
            const auto& v2 = vertexBuffer[idx2];

            const gp_Vec e1(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
            const gp_Vec e2(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);
            const gp_Vec cross = e1.Crossed(e2);

            for (const std::uint32_t idx : {idx0, idx1, idx2}) {
                vertexBuffer[idx].nx += static_cast<float>(cross.X());
                vertexBuffer[idx].ny += static_cast<float>(cross.Y());
                vertexBuffer[idx].nz += static_cast<float>(cross.Z());
            }
        }

        for (Standard_Integer i = 0; i < numNodes; ++i) {
            auto& v = vertexBuffer[baseVertexIndex + static_cast<std::uint32_t>(i)];
            const float len = std::sqrt(v.nx * v.nx + v.ny * v.ny + v.nz * v.nz);
            if (len > 1.0e-7f) {
                v.nx /= len;
                v.ny /= len;
                v.nz /= len;
            } else {
                v.nz = 1.0f;
            }
        }
        ++currentFaceId;
    }
}

ToolSpec makeDefaultTool(std::uint32_t id, const char* name, float diameterMm, float tempC) {
    ToolSpec t{};
    t.id = id;
    t.name = rust::String(name);
    t.nozzle_diameter_mm = diameterMm;
    t.temperature_celsius = tempC;
    t.has_rotational_axes = true;
    t.max_angle_from_bed_normal_deg = 45.0f;
    const float tipOuterR = std::max(diameterMm * 1.25f, 0.4f);
    t.clearance_profile_xz.push_back(ClearanceXZ{tipOuterR, 0.0f});
    t.clearance_profile_xz.push_back(ClearanceXZ{2.5f, 2.0f});
    t.clearance_profile_xz.push_back(ClearanceXZ{9.0f, 6.0f});
    t.clearance_profile_xz.push_back(ClearanceXZ{22.0f, 18.0f});
    t.clearance_profile_xz.push_back(ClearanceXZ{28.0f, 45.0f});
    return t;
}

} // namespace

struct SlicerBackend::Impl {
    std::uint64_t nextObjectId = 1;
    std::uint64_t lastLoadedId = 0;
    std::unordered_map<std::uint64_t, ObjectRecord> objects;
    std::vector<RenderVertex> emptyVertices;
    std::vector<std::uint32_t> emptyIndices;

    // 5/6-axis kinematics state in bed coordinates
    std::array<float, 3> nozzleAxisBed{0.0f, 0.0f, 1.0f};
    std::array<float, 3> gravityDirectionBed{0.0f, 0.0f, -1.0f};

    // Multi-tool registry & variable layer height configuration
    std::vector<ToolSpec> tools{
        makeDefaultTool(0, "T0 - 0.4mm Standard", 0.40f, 215.0f),
        makeDefaultTool(1, "T1 - 0.6mm High-Flow", 0.60f, 230.0f),
    };
    std::uint32_t activeToolIndex = 0;
    LayerHeightSpec sliceLayerHeight{0.20f, 0.08f, 0.32f, 0.20f, 0.20f};
    std::uint32_t eikonalSolverIterations = 4;
    bool multiResolutionEnabled = true;
    std::uint32_t multiResolutionCoarseRatio = 4;
    std::vector<SpatialOrientationHint> spatialOrientationHints;
    ShellPathSpec shellPathSpec{
        1.60f,
        2,
        LineWidthSpec{0.40f, 0.28f, 0.56f},
        LineWidthSpec{0.42f, 0.28f, 0.58f},
        LineWidthSpec{0.42f, 0.26f, 0.60f},
        0,
        0.20f,
        LineWidthSpec{0.45f, 0.30f, 0.65f},
        2,
        45.0f,
        0.40f,
        0.80f,
    };
};

SlicerBackend::SlicerBackend() : impl_(std::make_unique<Impl>()) {
    openvdb::initialize();
}

SlicerBackend::~SlicerBackend() = default;

bool SlicerBackend::loadStepFile(const std::string& filePath, double linearDeflection, double angularDeflection) {
    const auto parts = loadStepAssembly(filePath, linearDeflection, angularDeflection);
    return !parts.empty();
}

std::vector<DecomposedPartInfo> SlicerBackend::loadStepAssembly(
    const std::string& filePath,
    double linearDeflection,
    double angularDeflection)
{
    std::vector<DecomposedPartInfo> results;

    STEPControl_Reader reader;
    const IFSelect_ReturnStatus status = reader.ReadFile(filePath.c_str());
    if (status != IFSelect_RetDone) {
        return results;
    }

    reader.TransferRoots();
    const TopoDS_Shape rootShape = reader.OneShape();
    if (rootShape.IsNull()) {
        return results;
    }

    std::vector<TopoDS_Shape> leafShapes;
    collectDecomposedLeaves(rootShape, leafShapes);
    if (leafShapes.empty()) {
        leafShapes.push_back(rootShape);
    }

    const std::string stem = std::filesystem::path(filePath).stem().string();
    const bool isMultiPart = leafShapes.size() > 1;

    std::size_t partIndex = 1;
    for (const TopoDS_Shape& leaf : leafShapes) {
        ObjectRecord rec;
        rec.id = impl_->nextObjectId++;
        rec.name = isMultiPart
            ? (stem + "::Part_" + std::to_string(partIndex))
            : stem;
        rec.shape = leaf;
        rec.hasBrep = true;

        tessellateTopoShape(
            leaf,
            linearDeflection,
            angularDeflection,
            rec.vertexBuffer,
            rec.indexBuffer,
            rec.triangleFaceIds);

        if (rec.vertexBuffer.empty() || rec.indexBuffer.empty()) {
            continue;
        }

        recenterVertexBuffer(rec);

        gp_Trsf localShift;
        localShift.SetTranslation(gp_Vec(-rec.originX, -rec.originY, -rec.originZ));
        rec.shape.Move(TopLoc_Location(localShift));

        DecomposedPartInfo info;
        info.object_id = rec.id;
        info.name = rust::String(rec.name);
        info.origin_x = rec.originX;
        info.origin_y = rec.originY;
        info.origin_z = rec.originZ;
        info.has_brep = true;

        impl_->lastLoadedId = rec.id;
        impl_->objects.emplace(rec.id, std::move(rec));
        results.push_back(std::move(info));
        ++partIndex;
    }

    return results;
}

DecomposedPartInfo SlicerBackend::registerTriangleMesh(
    const std::string& name,
    const float* positions,
    std::size_t positionFloatsCount,
    const std::uint32_t* indices,
    std::size_t indexCount)
{
    DecomposedPartInfo info;
    info.object_id = 0;
    info.name = rust::String(name);
    info.origin_x = 0.0f;
    info.origin_y = 0.0f;
    info.origin_z = 0.0f;
    info.has_brep = false;

    const std::size_t numVertices = positionFloatsCount / 3;
    if (numVertices == 0 || indexCount < 3) {
        return info;
    }

    ObjectRecord rec;
    rec.id = impl_->nextObjectId++;
    rec.name = name;
    rec.hasBrep = false;
    rec.vertexBuffer.resize(numVertices);
    rec.indexBuffer.assign(indices, indices + indexCount);

    for (std::size_t i = 0; i < numVertices; ++i) {
        auto& v = rec.vertexBuffer[i];
        v.x = positions[i * 3 + 0];
        v.y = positions[i * 3 + 1];
        v.z = positions[i * 3 + 2];
        v.nx = 0.0f;
        v.ny = 0.0f;
        v.nz = 0.0f;
    }

    computeMeshNormals(rec.vertexBuffer, rec.indexBuffer);
    recenterVertexBuffer(rec);

    info.object_id = rec.id;
    info.origin_x = rec.originX;
    info.origin_y = rec.originY;
    info.origin_z = rec.originZ;

    impl_->lastLoadedId = rec.id;
    impl_->objects.emplace(rec.id, std::move(rec));
    return info;
}

bool SlicerBackend::removeObject(std::uint64_t objectId) {
    return impl_->objects.erase(objectId) > 0;
}

bool SlicerBackend::exportObjectStep(std::uint64_t objectId, const std::string& outFilePath) const {
    const auto it = impl_->objects.find(objectId);
    if (it == impl_->objects.end() || !it->second.hasBrep || it->second.shape.IsNull()) {
        return false;
    }

    STEPControl_Writer writer;
    const IFSelect_ReturnStatus transferStatus =
        writer.Transfer(it->second.shape, STEPControl_AsIs);
    if (transferStatus != IFSelect_RetDone) {
        return false;
    }

    const IFSelect_ReturnStatus writeStatus = writer.Write(outFilePath.c_str());
    return writeStatus == IFSelect_RetDone;
}

std::vector<RenderVertex> SlicerBackend::objectVertices(std::uint64_t objectId) const {
    const auto it = impl_->objects.find(objectId);
    if (it == impl_->objects.end()) {
        return {};
    }
    return it->second.vertexBuffer;
}

std::vector<std::uint32_t> SlicerBackend::objectIndices(std::uint64_t objectId) const {
    const auto it = impl_->objects.find(objectId);
    if (it == impl_->objects.end()) {
        return {};
    }
    return it->second.indexBuffer;
}

std::vector<std::uint32_t> SlicerBackend::objectTriangleFaceIds(std::uint64_t objectId) const {
    const auto it = impl_->objects.find(objectId);
    if (it == impl_->objects.end()) {
        return {};
    }
    return it->second.triangleFaceIds;
}

std::vector<std::string> SlicerBackend::listObjectFields(std::uint64_t objectId) const {
    std::vector<std::string> names;
    const auto it = impl_->objects.find(objectId);
    if (it == impl_->objects.end()) {
        return names;
    }
    for (const auto& [name, grid] : it->second.scalarFields) {
        if (grid) {
            names.push_back(name);
        }
    }
    return names;
}

FieldStats SlicerBackend::getObjectFieldStats(std::uint64_t objectId, const std::string& fieldName) const {
    FieldStats stats{};
    stats.valid = false;
    stats.field_name = rust::String(fieldName);
    stats.active_voxel_count = 0;
    stats.min_value = 0.0f;
    stats.max_value = 0.0f;
    stats.background_value = 0.0f;
    stats.voxel_size = 0.0f;

    const auto objIt = impl_->objects.find(objectId);
    if (objIt == impl_->objects.end()) {
        return stats;
    }
    const auto fieldIt = objIt->second.scalarFields.find(fieldName);
    if (fieldIt == objIt->second.scalarFields.end() || !fieldIt->second) {
        return stats;
    }

    const openvdb::FloatGrid::Ptr& grid = fieldIt->second;
    stats.valid = true;
    stats.active_voxel_count = grid->activeVoxelCount();
    stats.background_value = grid->background();
    stats.voxel_size = static_cast<float>(grid->transform().voxelSize()[0]);

    float minVal = std::numeric_limits<float>::max();
    float maxVal = std::numeric_limits<float>::lowest();
    for (auto iter = grid->cbeginValueOn(); iter; ++iter) {
        const float v = *iter;
        minVal = std::min(minVal, v);
        maxVal = std::max(maxVal, v);
    }

    if (objIt->second.isMultiResolution) {
        const auto cIt = objIt->second.coarseScalarFields.find(fieldName);
        if (cIt != objIt->second.coarseScalarFields.end() && cIt->second) {
            stats.active_voxel_count += cIt->second->activeVoxelCount();
            for (auto iter = cIt->second->cbeginValueOn(); iter; ++iter) {
                const float v = *iter;
                minVal = std::min(minVal, v);
                maxVal = std::max(maxVal, v);
            }
        }
    }

    if (minVal <= maxVal) {
        stats.min_value = minVal;
        stats.max_value = maxVal;
    }
    return stats;
}

bool SlicerBackend::recomputeObjectFields(
    std::uint64_t objectId,
    float voxelSize,
    float narrowBandVoxels)
{
    ObjectRecord workingRec;
    SlicerBackend snapshotBackend;
    {
        std::lock_guard<std::mutex> lock(g_backendMutex);
        const auto it = impl_->objects.find(objectId);
        if (it == impl_->objects.end()) {
            return false;
        }
        workingRec.id = it->second.id;
        workingRec.name = it->second.name;
        workingRec.minX = it->second.minX;
        workingRec.minY = it->second.minY;
        workingRec.minZ = it->second.minZ;
        workingRec.maxX = it->second.maxX;
        workingRec.maxY = it->second.maxY;
        workingRec.maxZ = it->second.maxZ;
        workingRec.vertexBuffer = it->second.vertexBuffer;
        workingRec.indexBuffer = it->second.indexBuffer;

        snapshotBackend.impl_->nozzleAxisBed = impl_->nozzleAxisBed;
        snapshotBackend.impl_->gravityDirectionBed = impl_->gravityDirectionBed;
        snapshotBackend.impl_->tools = impl_->tools;
        snapshotBackend.impl_->activeToolIndex = impl_->activeToolIndex;
        snapshotBackend.impl_->sliceLayerHeight = impl_->sliceLayerHeight;
        snapshotBackend.impl_->eikonalSolverIterations = impl_->eikonalSolverIterations;
        snapshotBackend.impl_->multiResolutionEnabled = impl_->multiResolutionEnabled;
        snapshotBackend.impl_->multiResolutionCoarseRatio = impl_->multiResolutionCoarseRatio;
        snapshotBackend.impl_->spatialOrientationHints = impl_->spatialOrientationHints;
        snapshotBackend.impl_->shellPathSpec = impl_->shellPathSpec;
    }

    setFieldProgress(true, 0.01f, "Preparing mesh & OpenVDB grid...");
    buildObjectScalarFields(workingRec, snapshotBackend, voxelSize, narrowBandVoxels);

    bool committed = false;
    {
        std::lock_guard<std::mutex> lock(g_backendMutex);
        const auto it = impl_->objects.find(objectId);
        if (it != impl_->objects.end()) {
            it->second.scalarFields = std::move(workingRec.scalarFields);
            it->second.toolAxisGrid = std::move(workingRec.toolAxisGrid);
            it->second.coarseScalarFields = std::move(workingRec.coarseScalarFields);
            it->second.coarseToolAxisGrid = std::move(workingRec.coarseToolAxisGrid);
            it->second.isMultiResolution = workingRec.isMultiResolution;
            it->second.fineShellInnerDepthMm = workingRec.fineShellInnerDepthMm;
            it->second.coarseToFineRatio = workingRec.coarseToFineRatio;
            it->second.hasCachedSelectedLayers = false;
            committed = true;
        }
    }
    setFieldProgress(false, 1.0f, "Field computation complete.");
    return committed;
}

IsoSurfaceMesh SlicerBackend::extractFieldIsosurface(
    std::uint64_t objectId,
    const std::string& fieldName,
    float isovalue,
    float adaptivity) const
{
    IsoSurfaceMesh out{};
    const auto objIt = impl_->objects.find(objectId);
    if (objIt == impl_->objects.end()) {
        return out;
    }
    const ObjectRecord& rec = objIt->second;
    const auto fieldIt = rec.scalarFields.find(fieldName);
    if (fieldIt == rec.scalarFields.end() || !fieldIt->second) {
        return out;
    }

    const openvdb::FloatGrid* gridPtr = fieldIt->second.get();
    if (rec.isMultiResolution) {
        const auto cIt = rec.coarseScalarFields.find(fieldName);
        if (cIt != rec.coarseScalarFields.end() && cIt->second) {
            const float fineDx = static_cast<float>(gridPtr->transform().voxelSize()[0]);
            if (fieldName == "sdf") {
                if (isovalue < -(rec.fineShellInnerDepthMm - 2.0f * fineDx) ||
                    isovalue > gridPtr->background() - 2.0f * fineDx) {
                    gridPtr = cIt->second.get();
                }
            } else {
                // For full-volume fields ("eikonal", "overhang", "tool_tilt_deg"), extract the
                // carrier surface from the full-interior coarse grid so no inner narrow-band
                // closure wall appears inside the part.
                gridPtr = cIt->second.get();
            }
        }
    }
    const openvdb::FloatGrid& grid = *gridPtr;
    std::vector<openvdb::Vec3s> points;
    std::vector<openvdb::Vec3I> triangles;
    std::vector<openvdb::Vec4I> quads;

    openvdb::tools::volumeToMesh(
        grid,
        points,
        triangles,
        quads,
        static_cast<double>(isovalue),
        static_cast<double>(std::clamp(adaptivity, 0.0f, 1.0f)),
        true);

    if (points.empty() || (triangles.empty() && quads.empty())) {
        return out;
    }

    std::vector<RenderVertex> verts(points.size());
    for (std::size_t i = 0; i < points.size(); ++i) {
        verts[i].x = points[i].x();
        verts[i].y = points[i].y();
        verts[i].z = points[i].z();
        verts[i].nx = 0.0f;
        verts[i].ny = 0.0f;
        verts[i].nz = 0.0f;
    }

    std::vector<std::uint32_t> inds;
    inds.reserve(triangles.size() * 3 + quads.size() * 6);

    // Note: OpenVDB volumeToMesh orders polygon vertices clockwise when viewed from
    // the higher-valued side of the isosurface; reversing to (0, 2, 1) produces
    // standard CCW outward normals.
    for (const auto& tri : triangles) {
        inds.push_back(tri[0]);
        inds.push_back(tri[2]);
        inds.push_back(tri[1]);
    }
    for (const auto& q : quads) {
        inds.push_back(q[0]);
        inds.push_back(q[2]);
        inds.push_back(q[1]);

        inds.push_back(q[0]);
        inds.push_back(q[3]);
        inds.push_back(q[2]);
    }

    computeMeshNormals(verts, inds);

    const auto sdfIt = rec.scalarFields.find("sdf");
    const bool hasSdf = (sdfIt != rec.scalarFields.end() && sdfIt->second != nullptr);

    // For the SDF field itself (or if no SDF exists), all vertices share `isovalue`
    if (fieldName == "sdf" || !hasSdf) {
        out.vertices.reserve(verts.size());
        out.vertex_sdf.reserve(verts.size());
        for (const auto& v : verts) {
            out.vertices.push_back(v);
            out.vertex_sdf.push_back(isovalue);
        }
        out.indices.reserve(inds.size());
        if (isovalue <= 0.0f) {
            out.interior_indices.reserve(inds.size());
        } else {
            out.exterior_indices.reserve(inds.size());
        }
        for (const auto idx : inds) {
            out.indices.push_back(idx);
            if (isovalue <= 0.0f) {
                out.interior_indices.push_back(idx);
            } else {
                out.exterior_indices.push_back(idx);
            }
        }
        return out;
    }

    // For non-SDF scalar fields ("eikonal", "overhang", etc.), evaluate the object's
    // Signed Distance Field at every vertex and split straddling triangles along the
    // exact object surface (\phi_sdf = 0) so the interior (\phi_sdf <= 0) and exterior
    // (\phi_sdf > 0) regions have a crisp geometric boundary and a 3D intersection contour.
    const openvdb::FloatGrid& fineSdfGrid = *sdfIt->second;
    const openvdb::FloatGrid* coarseSdfPtr = nullptr;
    if (rec.isMultiResolution) {
        const auto cSdfIt = rec.coarseScalarFields.find("sdf");
        if (cSdfIt != rec.coarseScalarFields.end() && cSdfIt->second) {
            coarseSdfPtr = cSdfIt->second.get();
        }
    }
    const openvdb::FloatGrid& carrierSdfGrid = (coarseSdfPtr != nullptr) ? *coarseSdfPtr : fineSdfGrid;
    const float sdfBg = carrierSdfGrid.background();
    const float voxelSize = static_cast<float>(carrierSdfGrid.transform().voxelSize()[0]);
    const float fineDx = static_cast<float>(fineSdfGrid.transform().voxelSize()[0]);
    const float fineInnerCutoff = -std::max(rec.fineShellInnerDepthMm - 1.75f * fineDx, 1.5f * fineDx);
    const float fineOuterCutoff = fineSdfGrid.background() - 1.5f * fineDx;
    // Strip the artificial closure shell generated by volumeToMesh where active narrow-band
    // voxels jump discontinuously to the background value at the outer fringe of the band.
    const float outerFringeLimit = sdfBg - 1.75f * voxelSize;

    auto fineSdfAcc = fineSdfGrid.getConstAccessor();
    openvdb::tools::GridSampler<openvdb::FloatGrid::ConstAccessor, openvdb::tools::BoxSampler>
        fineSdfSampler(fineSdfAcc, fineSdfGrid.transform());

    auto carrierSdfAcc = carrierSdfGrid.getConstAccessor();
    openvdb::tools::GridSampler<openvdb::FloatGrid::ConstAccessor, openvdb::tools::BoxSampler>
        carrierSdfSampler(carrierSdfAcc, carrierSdfGrid.transform());

    std::vector<float> vertSdf(verts.size(), 0.0f);
    for (std::size_t i = 0; i < verts.size(); ++i) {
        const openvdb::Vec3d wp(verts[i].x, verts[i].y, verts[i].z);
        const float fSdf = fineSdfSampler.wsSample(wp);
        if (coarseSdfPtr != nullptr && (fSdf <= fineInnerCutoff || fSdf >= fineOuterCutoff)) {
            vertSdf[i] = carrierSdfSampler.wsSample(wp);
        } else {
            vertSdf[i] = fSdf;
        }
    }

    auto lerpVertex = [](const RenderVertex& a, const RenderVertex& b, float t) -> RenderVertex {
        RenderVertex r{};
        r.x = a.x + t * (b.x - a.x);
        r.y = a.y + t * (b.y - a.y);
        r.z = a.z + t * (b.z - a.z);
        r.nx = a.nx + t * (b.nx - a.nx);
        r.ny = a.ny + t * (b.ny - a.ny);
        r.nz = a.nz + t * (b.nz - a.nz);
        const float len = std::sqrt(r.nx * r.nx + r.ny * r.ny + r.nz * r.nz);
        if (len > 1.0e-7f) {
            r.nx /= len;
            r.ny /= len;
            r.nz /= len;
        } else {
            r.nz = 1.0f;
        }
        return r;
    };

    out.vertices.reserve(verts.size() + inds.size() / 3);
    out.vertex_sdf.reserve(verts.size() + inds.size() / 3);
    for (std::size_t i = 0; i < verts.size(); ++i) {
        out.vertices.push_back(verts[i]);
        out.vertex_sdf.push_back(vertSdf[i]);
    }

    out.indices.reserve(inds.size());
    out.interior_indices.reserve(inds.size());
    out.exterior_indices.reserve(inds.size());

    auto pushInteriorTri = [&](std::uint32_t i0, std::uint32_t i1, std::uint32_t i2) {
        out.interior_indices.push_back(i0);
        out.interior_indices.push_back(i1);
        out.interior_indices.push_back(i2);
        out.indices.push_back(i0);
        out.indices.push_back(i1);
        out.indices.push_back(i2);
    };

    auto pushExteriorTri = [&](std::uint32_t i0, std::uint32_t i1, std::uint32_t i2) {
        out.exterior_indices.push_back(i0);
        out.exterior_indices.push_back(i1);
        out.exterior_indices.push_back(i2);
        out.indices.push_back(i0);
        out.indices.push_back(i1);
        out.indices.push_back(i2);
    };

    auto addCutVertex = [&](const RenderVertex& v, float sdfTag) -> std::uint32_t {
        const auto idx = static_cast<std::uint32_t>(out.vertices.size());
        out.vertices.push_back(v);
        out.vertex_sdf.push_back(sdfTag);
        return idx;
    };

    for (std::size_t t = 0; t + 2 < inds.size(); t += 3) {
        const std::uint32_t i0 = inds[t + 0];
        const std::uint32_t i1 = inds[t + 1];
        const std::uint32_t i2 = inds[t + 2];

        const float d0 = vertSdf[i0];
        const float d1 = vertSdf[i1];
        const float d2 = vertSdf[i2];

        // Skip outer narrow-band closure wall triangles
        if (std::max({d0, d1, d2}) >= outerFringeLimit &&
            std::min({d0, d1, d2}) > 0.25f * voxelSize) {
            continue;
        }

        const bool in0 = (d0 <= 0.0f);
        const bool in1 = (d1 <= 0.0f);
        const bool in2 = (d2 <= 0.0f);

        if (in0 && in1 && in2) {
            pushInteriorTri(i0, i1, i2);
            continue;
        }
        if (!in0 && !in1 && !in2) {
            pushExteriorTri(i0, i1, i2);
            continue;
        }

        // Triangle straddles \phi_sdf = 0: cyclically permute (A, B, C) preserving CCW winding
        // so that vertex A is on the solo side of \phi_sdf = 0 and (B, C) are on the opposite side.
        std::uint32_t idxA = i0, idxB = i1, idxC = i2;
        float dA = d0, dB = d1, dC = d2;
        bool soloInside = in0;

        if (in0 == in1) {
            // C is the solo vertex -> cyclic permutation (C, A, B)
            idxA = i2; idxB = i0; idxC = i1;
            dA = d2;   dB = d0;   dC = d1;
            soloInside = in2;
        } else if (in0 == in2) {
            // B is the solo vertex -> cyclic permutation (B, C, A)
            idxA = i1; idxB = i2; idxC = i0;
            dA = d1;   dB = d2;   dC = d0;
            soloInside = in1;
        }

        const float denomAB = dB - dA;
        const float denomAC = dC - dA;
        const float tAB = (std::abs(denomAB) > 1.0e-8f)
            ? std::clamp(-dA / denomAB, 0.0f, 1.0f)
            : 0.5f;
        const float tAC = (std::abs(denomAC) > 1.0e-8f)
            ? std::clamp(-dA / denomAC, 0.0f, 1.0f)
            : 0.5f;

        const RenderVertex vAB = lerpVertex(verts[idxA], verts[idxB], tAB);
        const RenderVertex vAC = lerpVertex(verts[idxA], verts[idxC], tAC);

        // Record the 3D intersection contour segment on \tau(x) = isovalue \cap \phi_sdf(x) = 0
        out.intersection_contour.push_back(vAB.x);
        out.intersection_contour.push_back(vAB.y);
        out.intersection_contour.push_back(vAB.z);
        out.intersection_contour.push_back(vAC.x);
        out.intersection_contour.push_back(vAC.y);
        out.intersection_contour.push_back(vAC.z);

        // Create separate interior (-eps) and exterior (+eps) seam vertices so per-vertex
        // colors have a crisp, non-bleeding boundary right along \phi_sdf = 0.
        const std::uint32_t vAB_in = addCutVertex(vAB, -1.0e-5f);
        const std::uint32_t vAC_in = addCutVertex(vAC, -1.0e-5f);
        const std::uint32_t vAB_ex = addCutVertex(vAB, +1.0e-5f);
        const std::uint32_t vAC_ex = addCutVertex(vAC, +1.0e-5f);

        if (soloInside) {
            pushInteriorTri(idxA, vAB_in, vAC_in);
            pushExteriorTri(vAB_ex, idxB, idxC);
            pushExteriorTri(vAB_ex, idxC, vAC_ex);
        } else {
            pushExteriorTri(idxA, vAB_ex, vAC_ex);
            pushInteriorTri(vAB_in, idxB, idxC);
            pushInteriorTri(vAB_in, idxC, vAC_in);
        }
    }

    return out;
}

FieldSlice2D SlicerBackend::sampleFieldSlice(
    std::uint64_t objectId,
    const std::string& fieldName,
    std::uint8_t axis,
    float normalizedPos,
    std::uint32_t resolution) const
{
    FieldSlice2D slice{};
    slice.valid = false;
    slice.width = 0;
    slice.height = 0;
    slice.origin = {0.0f, 0.0f, 0.0f};
    slice.u_vec = {1.0f, 0.0f, 0.0f};
    slice.v_vec = {0.0f, 1.0f, 0.0f};
    slice.normal = {0.0f, 0.0f, 1.0f};
    slice.plane_coord = 0.0f;
    slice.min_value = 0.0f;
    slice.max_value = 0.0f;
    slice.background_value = 0.0f;

    const auto objIt = impl_->objects.find(objectId);
    if (objIt == impl_->objects.end()) {
        return slice;
    }
    const ObjectRecord& rec = objIt->second;
    const auto fieldIt = rec.scalarFields.find(fieldName);
    if (fieldIt == rec.scalarFields.end() || !fieldIt->second) {
        return slice;
    }

    const openvdb::FloatGrid& grid = *fieldIt->second;
    const float bg = grid.background();
    const float voxel = static_cast<float>(grid.transform().voxelSize()[0]);
    const float maxSpan = std::max({
        rec.maxX - rec.minX,
        rec.maxY - rec.minY,
        rec.maxZ - rec.minZ,
        1.0f});
    const float pad = std::max(0.18f * maxSpan, voxel * 3.0f);

    const float bMinX = rec.minX - pad;
    const float bMinY = rec.minY - pad;
    const float bMinZ = rec.minZ - pad;
    const float bMaxX = rec.maxX + pad;
    const float bMaxY = rec.maxY + pad;
    const float bMaxZ = rec.maxZ + pad;

    const float t = std::clamp(normalizedPos, 0.0f, 1.0f);
    const std::uint32_t res = std::clamp<std::uint32_t>(resolution, 16, 512);

    if (axis == 0) {
        // YZ slice at X = const
        const float xCoord = bMinX + t * (bMaxX - bMinX);
        slice.origin = {xCoord, bMinY, bMinZ};
        slice.u_vec = {0.0f, bMaxY - bMinY, 0.0f};
        slice.v_vec = {0.0f, 0.0f, bMaxZ - bMinZ};
        slice.normal = {1.0f, 0.0f, 0.0f};
        slice.plane_coord = xCoord;
    } else if (axis == 1) {
        // XZ slice at Y = const
        const float yCoord = bMinY + t * (bMaxY - bMinY);
        slice.origin = {bMinX, yCoord, bMinZ};
        slice.u_vec = {bMaxX - bMinX, 0.0f, 0.0f};
        slice.v_vec = {0.0f, 0.0f, bMaxZ - bMinZ};
        slice.normal = {0.0f, 1.0f, 0.0f};
        slice.plane_coord = yCoord;
    } else {
        // XY slice at Z = const (default)
        const float zCoord = bMinZ + t * (bMaxZ - bMinZ);
        slice.origin = {bMinX, bMinY, zCoord};
        slice.u_vec = {bMaxX - bMinX, 0.0f, 0.0f};
        slice.v_vec = {0.0f, bMaxY - bMinY, 0.0f};
        slice.normal = {0.0f, 0.0f, 1.0f};
        slice.plane_coord = zCoord;
    }

    slice.valid = true;
    slice.width = res;
    slice.height = res;
    slice.background_value = bg;

    // Query global active min/max of the field so color normalization stays consistent across slices
    float minVal = std::numeric_limits<float>::max();
    float maxVal = std::numeric_limits<float>::lowest();
    for (auto iter = grid.cbeginValueOn(); iter; ++iter) {
        const float v = *iter;
        minVal = std::min(minVal, v);
        maxVal = std::max(maxVal, v);
    }
    const openvdb::FloatGrid* coarseFieldPtr = nullptr;
    const openvdb::FloatGrid* coarseSdfPtr = nullptr;
    if (rec.isMultiResolution) {
        const auto cFieldIt = rec.coarseScalarFields.find(fieldName);
        if (cFieldIt != rec.coarseScalarFields.end() && cFieldIt->second) {
            coarseFieldPtr = cFieldIt->second.get();
            for (auto iter = coarseFieldPtr->cbeginValueOn(); iter; ++iter) {
                const float v = *iter;
                minVal = std::min(minVal, v);
                maxVal = std::max(maxVal, v);
            }
        }
        const auto cSdfIt = rec.coarseScalarFields.find("sdf");
        if (cSdfIt != rec.coarseScalarFields.end() && cSdfIt->second) {
            coarseSdfPtr = cSdfIt->second.get();
        }
    }
    if (minVal > maxVal) {
        minVal = -1.0f;
        maxVal = 1.0f;
    }
    slice.min_value = minVal;
    slice.max_value = maxVal;

    auto accessor = grid.getConstAccessor();
    openvdb::tools::GridSampler<openvdb::FloatGrid::ConstAccessor, openvdb::tools::BoxSampler>
        sampler(accessor, grid.transform());

    const auto sdfIt = rec.scalarFields.find("sdf");
    const openvdb::FloatGrid* sdfGridPtr =
        (sdfIt != rec.scalarFields.end() && sdfIt->second) ? sdfIt->second.get() : &grid;
    auto sdfAccessor = sdfGridPtr->getConstAccessor();
    openvdb::tools::GridSampler<openvdb::FloatGrid::ConstAccessor, openvdb::tools::BoxSampler>
        sdfSampler(sdfAccessor, sdfGridPtr->transform());

    const openvdb::FloatGrid* fallbackFieldGrid =
        (coarseFieldPtr != nullptr) ? coarseFieldPtr : &grid;
    auto coarseFieldAcc = fallbackFieldGrid->getConstAccessor();
    openvdb::tools::GridSampler<openvdb::FloatGrid::ConstAccessor, openvdb::tools::BoxSampler>
        coarseFieldSampler(coarseFieldAcc, fallbackFieldGrid->transform());

    const openvdb::FloatGrid* fallbackSdfGrid =
        (coarseSdfPtr != nullptr) ? coarseSdfPtr : sdfGridPtr;
    auto coarseSdfAcc = fallbackSdfGrid->getConstAccessor();
    openvdb::tools::GridSampler<openvdb::FloatGrid::ConstAccessor, openvdb::tools::BoxSampler>
        coarseSdfSampler(coarseSdfAcc, fallbackSdfGrid->transform());

    const float fineInnerCutoff = -std::max(rec.fineShellInnerDepthMm - 1.75f * voxel, 1.5f * voxel);
    const float fineOuterCutoff = sdfGridPtr->background() - 1.5f * voxel;

    const std::size_t totalPixels = static_cast<std::size_t>(res) * res;
    slice.values.reserve(totalPixels);
    slice.sdf_values.reserve(totalPixels);
    for (std::uint32_t iv = 0; iv < res; ++iv) {
        const float fv = (static_cast<float>(iv) + 0.5f) / static_cast<float>(res);
        for (std::uint32_t iu = 0; iu < res; ++iu) {
            const float fu = (static_cast<float>(iu) + 0.5f) / static_cast<float>(res);
            const openvdb::Vec3d wp(
                slice.origin[0] + fu * slice.u_vec[0] + fv * slice.v_vec[0],
                slice.origin[1] + fu * slice.u_vec[1] + fv * slice.v_vec[1],
                slice.origin[2] + fu * slice.u_vec[2] + fv * slice.v_vec[2]);
            const float fSdf = sdfSampler.wsSample(wp);
            if (rec.isMultiResolution && coarseSdfPtr != nullptr &&
                (fSdf <= fineInnerCutoff || fSdf >= fineOuterCutoff)) {
                slice.values.push_back(coarseFieldSampler.wsSample(wp));
                slice.sdf_values.push_back(coarseSdfSampler.wsSample(wp));
            } else {
                slice.values.push_back(sampler.wsSample(wp));
                slice.sdf_values.push_back(fSdf);
            }
        }
    }

    return slice;
}

const std::vector<RenderVertex>& SlicerBackend::vertices() const noexcept {
    const auto it = impl_->objects.find(impl_->lastLoadedId);
    if (it == impl_->objects.end()) {
        return impl_->emptyVertices;
    }
    return it->second.vertexBuffer;
}

const std::vector<std::uint32_t>& SlicerBackend::indices() const noexcept {
    const auto it = impl_->objects.find(impl_->lastLoadedId);
    if (it == impl_->objects.end()) {
        return impl_->emptyIndices;
    }
    return it->second.indexBuffer;
}

// ============================================================================
// Thread-safe Global Backend Singleton for FFI Dispatch
// ============================================================================

namespace {
SlicerBackend& globalBackend() {
    static SlicerBackend instance;
    return instance;
}
} // namespace

bool load_cad_model(rust::Str path) {
    std::lock_guard<std::mutex> lock(g_backendMutex);
    const std::string filePath(path.data(), path.size());
    return globalBackend().loadStepFile(filePath);
}

rust::Vec<RenderVertex> get_tessellated_vertices() {
    std::lock_guard<std::mutex> lock(g_backendMutex);
    const auto& src = globalBackend().vertices();
    rust::Vec<RenderVertex> out;
    out.reserve(src.size());
    for (const auto& v : src) {
        out.push_back(v);
    }
    return out;
}

rust::Vec<std::uint32_t> get_tessellated_indices() {
    std::lock_guard<std::mutex> lock(g_backendMutex);
    const auto& src = globalBackend().indices();
    rust::Vec<std::uint32_t> out;
    out.reserve(src.size());
    for (const std::uint32_t idx : src) {
        out.push_back(idx);
    }
    return out;
}

rust::Vec<DecomposedPartInfo> load_step_assembly(rust::Str path) {
    std::lock_guard<std::mutex> lock(g_backendMutex);
    const std::string filePath(path.data(), path.size());
    const auto parts = globalBackend().loadStepAssembly(filePath);
    rust::Vec<DecomposedPartInfo> out;
    out.reserve(parts.size());
    for (const auto& p : parts) {
        out.push_back(p);
    }
    return out;
}

DecomposedPartInfo register_triangle_mesh(
    rust::Str name,
    rust::Slice<const float> positions,
    rust::Slice<const std::uint32_t> indices)
{
    std::lock_guard<std::mutex> lock(g_backendMutex);
    const std::string meshName(name.data(), name.size());
    return globalBackend().registerTriangleMesh(
        meshName,
        positions.data(),
        positions.size(),
        indices.data(),
        indices.size());
}

rust::Vec<RenderVertex> get_object_vertices(std::uint64_t object_id) {
    std::lock_guard<std::mutex> lock(g_backendMutex);
    const auto src = globalBackend().objectVertices(object_id);
    rust::Vec<RenderVertex> out;
    out.reserve(src.size());
    for (const auto& v : src) {
        out.push_back(v);
    }
    return out;
}

rust::Vec<std::uint32_t> get_object_indices(std::uint64_t object_id) {
    std::lock_guard<std::mutex> lock(g_backendMutex);
    const auto src = globalBackend().objectIndices(object_id);
    rust::Vec<std::uint32_t> out;
    out.reserve(src.size());
    for (const std::uint32_t idx : src) {
        out.push_back(idx);
    }
    return out;
}

rust::Vec<std::uint32_t> get_object_triangle_face_ids(std::uint64_t object_id) {
    std::lock_guard<std::mutex> lock(g_backendMutex);
    const auto src = globalBackend().objectTriangleFaceIds(object_id);
    rust::Vec<std::uint32_t> out;
    out.reserve(src.size());
    for (const std::uint32_t fid : src) {
        out.push_back(fid);
    }
    return out;
}

bool remove_cad_object(std::uint64_t object_id) {
    std::lock_guard<std::mutex> lock(g_backendMutex);
    return globalBackend().removeObject(object_id);
}

bool export_object_step_to_path(std::uint64_t object_id, rust::Str path) {
    std::lock_guard<std::mutex> lock(g_backendMutex);
    const std::string filePath(path.data(), path.size());
    return globalBackend().exportObjectStep(object_id, filePath);
}

rust::Vec<rust::String> list_object_fields(std::uint64_t object_id) {
    std::lock_guard<std::mutex> lock(g_backendMutex);
    const auto names = globalBackend().listObjectFields(object_id);
    rust::Vec<rust::String> out;
    out.reserve(names.size());
    for (const auto& n : names) {
        out.push_back(rust::String(n));
    }
    return out;
}

FieldStats get_object_field_stats(std::uint64_t object_id, rust::Str field_name) {
    std::lock_guard<std::mutex> lock(g_backendMutex);
    const std::string name(field_name.data(), field_name.size());
    return globalBackend().getObjectFieldStats(object_id, name);
}

bool recompute_object_fields(std::uint64_t object_id, float voxel_size, float narrow_band_voxels) {
    return globalBackend().recomputeObjectFields(object_id, voxel_size, narrow_band_voxels);
}

FieldComputeProgress get_field_compute_progress() {
    return getFieldProgressSnapshot();
}

IsoSurfaceMesh extract_field_isosurface(
    std::uint64_t object_id,
    rust::Str field_name,
    float isovalue,
    float adaptivity)
{
    std::lock_guard<std::mutex> lock(g_backendMutex);
    const std::string name(field_name.data(), field_name.size());
    return globalBackend().extractFieldIsosurface(object_id, name, isovalue, adaptivity);
}

FieldSlice2D sample_field_slice(
    std::uint64_t object_id,
    rust::Str field_name,
    std::uint8_t axis,
    float normalized_pos,
    std::uint32_t resolution)
{
    std::lock_guard<std::mutex> lock(g_backendMutex);
    const std::string name(field_name.data(), field_name.size());
    return globalBackend().sampleFieldSlice(object_id, name, axis, normalized_pos, resolution);
}

NozzleBasis SlicerBackend::queryNozzleFrame(
    const std::array<float, 3>& /*positionBed*/,
    const std::array<float, 3>& travelDirBed) const noexcept
{
    return computeNozzleBasis(impl_->nozzleAxisBed, travelDirBed);
}

std::array<float, 3> SlicerBackend::queryGravityDirection(
    const std::array<float, 3>& /*positionBed*/) const noexcept
{
    return impl_->gravityDirectionBed;
}

void SlicerBackend::setNozzleAxisBed(const std::array<float, 3>& axisBed) noexcept {
    const float len = std::sqrt(
        axisBed[0] * axisBed[0] +
        axisBed[1] * axisBed[1] +
        axisBed[2] * axisBed[2]);
    if (len > 1.0e-7f) {
        impl_->nozzleAxisBed = {axisBed[0] / len, axisBed[1] / len, axisBed[2] / len};
    } else {
        impl_->nozzleAxisBed = {0.0f, 0.0f, 1.0f};
    }
}

void SlicerBackend::setGravityDirectionBed(const std::array<float, 3>& gravityBed) noexcept {
    const float len = std::sqrt(
        gravityBed[0] * gravityBed[0] +
        gravityBed[1] * gravityBed[1] +
        gravityBed[2] * gravityBed[2]);
    if (len > 1.0e-7f) {
        impl_->gravityDirectionBed = {
            gravityBed[0] / len,
            gravityBed[1] / len,
            gravityBed[2] / len};
    } else {
        impl_->gravityDirectionBed = {0.0f, 0.0f, -1.0f};
    }
}

void SlicerBackend::setMachineTools(
    const std::vector<ToolSpec>& tools,
    std::uint32_t activeToolIndex)
{
    impl_->tools = tools;
    if (impl_->tools.empty()) {
        impl_->tools.push_back(makeDefaultTool(0, "T0 - 0.4mm Standard", 0.40f, 215.0f));
        impl_->activeToolIndex = 0;
    } else {
        impl_->activeToolIndex = std::min<std::uint32_t>(
            activeToolIndex,
            static_cast<std::uint32_t>(impl_->tools.size() - 1));
    }
}

const std::vector<ToolSpec>& SlicerBackend::machineTools() const noexcept {
    return impl_->tools;
}

std::uint32_t SlicerBackend::activeToolIndex() const noexcept {
    return impl_->activeToolIndex;
}

void SlicerBackend::setSliceLayerHeight(const LayerHeightSpec& spec) noexcept {
    LayerHeightSpec s = spec;
    s.min_mm = std::max(s.min_mm, 0.01f);
    s.max_mm = std::max(s.max_mm, s.min_mm);
    s.target_mm = std::clamp(s.target_mm, s.min_mm, s.max_mm);
    s.first_layer_mm = (s.first_layer_mm > 0.0f) ? std::max(s.first_layer_mm, 0.01f) : s.target_mm;
    s.top_layer_mm = (s.top_layer_mm > 0.0f) ? std::max(s.top_layer_mm, 0.01f) : s.target_mm;
    impl_->sliceLayerHeight = s;
}

LayerHeightSpec SlicerBackend::sliceLayerHeight() const noexcept {
    return impl_->sliceLayerHeight;
}

void SlicerBackend::setEikonalSolverIterations(std::uint32_t iterations) noexcept {
    impl_->eikonalSolverIterations = std::clamp<std::uint32_t>(iterations, 1, 16);
}

std::uint32_t SlicerBackend::eikonalSolverIterations() const noexcept {
    return impl_->eikonalSolverIterations;
}

void SlicerBackend::setMultiResolutionConfig(bool enabled, std::uint32_t coarseRatio) noexcept {
    impl_->multiResolutionEnabled = enabled;
    impl_->multiResolutionCoarseRatio = std::clamp<std::uint32_t>(coarseRatio, 2, 8);
}

bool SlicerBackend::multiResolutionEnabled() const noexcept {
    return impl_->multiResolutionEnabled;
}

std::uint32_t SlicerBackend::multiResolutionCoarseRatio() const noexcept {
    return impl_->multiResolutionCoarseRatio;
}

void SlicerBackend::setSpatialOrientationHints(const std::vector<SpatialOrientationHint>& hints) {
    impl_->spatialOrientationHints = hints;
}

const std::vector<SpatialOrientationHint>& SlicerBackend::spatialOrientationHints() const noexcept {
    return impl_->spatialOrientationHints;
}

SelectedIsosurfacesResult SlicerBackend::selectSlicingIsosurfaces(
    std::uint64_t objectId,
    float adaptivity) const
{
    SelectedIsosurfacesResult result{};
    result.valid = false;
    result.branch_count = 0;

    const auto objIt = impl_->objects.find(objectId);
    if (objIt == impl_->objects.end()) {
        return result;
    }
    const ObjectRecord& rec = objIt->second;

    const auto eikIt = rec.scalarFields.find("eikonal");
    const auto sdfIt = rec.scalarFields.find("sdf");
    if (eikIt == rec.scalarFields.end() || !eikIt->second ||
        sdfIt == rec.scalarFields.end() || !sdfIt->second) {
        return result;
    }

    const openvdb::FloatGrid& fineEikGrid = *eikIt->second;
    const openvdb::FloatGrid& fineSdfGrid = *sdfIt->second;

    const openvdb::FloatGrid* coarseEikPtr = nullptr;
    const openvdb::FloatGrid* coarseSdfPtr = nullptr;
    if (rec.isMultiResolution) {
        const auto cEikIt = rec.coarseScalarFields.find("eikonal");
        const auto cSdfIt = rec.coarseScalarFields.find("sdf");
        if (cEikIt != rec.coarseScalarFields.end() && cEikIt->second &&
            cSdfIt != rec.coarseScalarFields.end() && cSdfIt->second) {
            coarseEikPtr = cEikIt->second.get();
            coarseSdfPtr = cSdfIt->second.get();
        }
    }
    const bool useDualGridExtraction = (coarseEikPtr != nullptr && coarseSdfPtr != nullptr);
    const openvdb::FloatGrid& eikGrid = useDualGridExtraction ? *coarseEikPtr : fineEikGrid;
    const openvdb::FloatGrid& sdfGrid = useDualGridExtraction ? *coarseSdfPtr : fineSdfGrid;
    const float voxelSize = static_cast<float>(eikGrid.transform().voxelSize()[0]);
    const float fineVoxelSize = static_cast<float>(fineEikGrid.transform().voxelSize()[0]);
    const float eikBg = eikGrid.background();
    const float sdfBg = sdfGrid.background();

    LayerHeightSpec lh = impl_->sliceLayerHeight;
    const float hMin = std::max(lh.min_mm, 0.01f);
    const float hMax = std::max(lh.max_mm, hMin);
    const float hTarget = std::clamp(lh.target_mm, hMin, hMax);
    const float hFirst = (lh.first_layer_mm > 0.0f) ? std::max(lh.first_layer_mm, 0.01f) : hTarget;
    const float hTop = (lh.top_layer_mm > 0.0f) ? std::max(lh.top_layer_mm, 0.01f) : hTarget;

    if (rec.hasCachedSelectedLayers &&
        std::abs(rec.cachedSelectedLayersSpec.target_mm - lh.target_mm) < 1.0e-5f &&
        std::abs(rec.cachedSelectedLayersSpec.min_mm - lh.min_mm) < 1.0e-5f &&
        std::abs(rec.cachedSelectedLayersSpec.max_mm - lh.max_mm) < 1.0e-5f &&
        std::abs(rec.cachedSelectedLayersSpec.first_layer_mm - lh.first_layer_mm) < 1.0e-5f &&
        std::abs(rec.cachedSelectedLayersSpec.top_layer_mm - lh.top_layer_mm) < 1.0e-5f) {
        return rec.cachedSelectedLayers;
    }

    setFieldProgress(true, 0.05f, "Selecting non-planar slicing isosurfaces...");

    // ------------------------------------------------------------------------
    // 1. Collect Active Interior Voxels, Compute |\nabla \tau|, & Exact \phi_sdf = 0 Bounds
    // ------------------------------------------------------------------------
    struct InteriorVoxel {
        openvdb::Coord ijk;
        float tau = 0.0f;
        float sdf = 0.0f;
        float gradMag = 1.0f;
        float surfTauMin = 0.0f;
        float surfTauMax = 0.0f;
        float cornerSdfAllowance = 0.0f;
        int branchId = 0;
    };

    std::vector<InteriorVoxel> voxels;
    voxels.reserve(eikGrid.activeVoxelCount() / 2 + 64);

    auto sdfAccMain = sdfGrid.getConstAccessor();
    auto eikAccMain = eikGrid.getConstAccessor();

    // Distinguish valid solved voxels from the background value `eikBg`
    const float validTauLimit = eikBg - std::max(0.5f * voxelSize, 0.5f);
    // Include voxels up to +0.50 * voxelSize for 26-connectivity across thin features/corners
    // (and up to +0.85 * voxelSize when the voxel is a thin-wall medial valley minimum between
    // two opposite positive surfaces), while tracking exact \phi_sdf = 0 surface-clamped
    // [surfTauMin, surfTauMax] per voxel.
    const float interiorSdfCutoff = 0.50f * voxelSize;
    const float thinWallSdfCutoff = 0.85f * voxelSize;

    openvdb::Int32Grid voxelIdxGrid(-1);
    voxelIdxGrid.setTransform(eikGrid.transform().copy());
    auto idxAcc = voxelIdxGrid.getAccessor();

    for (auto iter = eikGrid.cbeginValueOn(); iter; ++iter) {
        const float tau = *iter;
        if (!std::isfinite(tau) || tau >= validTauLimit) {
            continue;
        }
        const openvdb::Coord ijk = iter.getCoord();
        const float sdf = sdfAccMain.getValue(ijk);
        if (sdf > interiorSdfCutoff) {
            if (sdf > thinWallSdfCutoff) {
                continue;
            }
            const float margin = sdf + 0.12f * voxelSize;
            const bool isThinWallMedial =
                (sdfAccMain.getValue(ijk.offsetBy(-1, 0, 0)) > margin &&
                 sdfAccMain.getValue(ijk.offsetBy(1, 0, 0)) > margin) ||
                (sdfAccMain.getValue(ijk.offsetBy(0, -1, 0)) > margin &&
                 sdfAccMain.getValue(ijk.offsetBy(0, 1, 0)) > margin) ||
                (sdfAccMain.getValue(ijk.offsetBy(0, 0, -1)) > margin &&
                 sdfAccMain.getValue(ijk.offsetBy(0, 0, 1)) > margin);
            if (!isThinWallMedial) {
                continue;
            }
        }

        const int vIdx = static_cast<int>(voxels.size());
        idxAcc.setValue(ijk, vIdx);
        voxels.push_back(InteriorVoxel{ijk, tau, sdf, 1.0f, tau, tau, 0.0f, 0});
    }

    if (voxels.empty()) {
        return result;
    }

    // Compute finite-difference |\nabla \tau| and exact \phi_sdf = 0 surface crossings per voxel
    const float invVoxel = 1.0f / std::max(voxelSize, 1.0e-5f);
    float globalTauMin = std::numeric_limits<float>::max();
    float globalTauMax = std::numeric_limits<float>::lowest();

    const std::array<openvdb::Coord, 3> axes = {
        openvdb::Coord(1, 0, 0),
        openvdb::Coord(0, 1, 0),
        openvdb::Coord(0, 0, 1),
    };

    for (std::size_t i = 0; i < voxels.size(); ++i) {
        const openvdb::Coord& c = voxels[i].ijk;
        const float t0 = voxels[i].tau;
        const float s0 = voxels[i].sdf;

        std::array<float, 3> gradTau{0.0f, 0.0f, 0.0f};
        std::array<float, 3> gradSdf{0.0f, 0.0f, 0.0f};

        for (int a = 0; a < 3; ++a) {
            const openvdb::Coord& d = axes[static_cast<std::size_t>(a)];
            const float tP = eikAccMain.getValue(c + d);
            const float tM = eikAccMain.getValue(c - d);
            const float sP = sdfAccMain.getValue(c + d);
            const float sM = sdfAccMain.getValue(c - d);
            const bool okP = std::isfinite(tP) && (tP < validTauLimit);
            const bool okM = std::isfinite(tM) && (tM < validTauLimit);
            const bool inP = okP && (sP <= 0.05f * voxelSize);
            const bool inM = okM && (sM <= 0.05f * voxelSize);
            if (inP && inM) {
                gradTau[static_cast<std::size_t>(a)] = 0.5f * (tP - tM) * invVoxel;
            } else if (inP) {
                gradTau[static_cast<std::size_t>(a)] = (tP - t0) * invVoxel;
            } else if (inM) {
                gradTau[static_cast<std::size_t>(a)] = (t0 - tM) * invVoxel;
            } else if (okP && okM) {
                gradTau[static_cast<std::size_t>(a)] = 0.5f * (tP - tM) * invVoxel;
            } else if (okP) {
                gradTau[static_cast<std::size_t>(a)] = (tP - t0) * invVoxel;
            } else if (okM) {
                gradTau[static_cast<std::size_t>(a)] = (t0 - tM) * invVoxel;
            }

            gradSdf[static_cast<std::size_t>(a)] = 0.5f * (sP - sM) * invVoxel;
        }

        const float g = std::sqrt(
            gradTau[0] * gradTau[0] + gradTau[1] * gradTau[1] + gradTau[2] * gradTau[2]);
        voxels[i].gradMag = (g > 1.0e-3f) ? std::clamp(g, 0.20f, 4.0f) : 1.0f;

        if (s0 <= 0.0f) {
            float tMin = t0;
            float tMax = t0;
            // Expand to exact \phi_sdf = 0 crossings along 26-neighbor edges exiting the object
            for (int dz = -1; dz <= 1; ++dz) {
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        if (dx == 0 && dy == 0 && dz == 0) {
                            continue;
                        }
                        const openvdb::Coord nb = c + openvdb::Coord(dx, dy, dz);
                        const float sNb = sdfAccMain.getValue(nb);
                        if (sNb <= 0.0f) {
                            continue;
                        }
                        const float tNb = eikAccMain.getValue(nb);
                        if (!std::isfinite(tNb) || tNb >= validTauLimit) {
                            continue;
                        }
                        const float denom = sNb - s0;
                        if (denom > 1.0e-6f) {
                            const float alpha = std::clamp(-s0 / denom, 0.0f, 1.0f);
                            const float tZero = (1.0f - alpha) * t0 + alpha * tNb;
                            tMin = std::min(tMin, tZero);
                            tMax = std::max(tMax, tZero);
                        }
                    }
                }
            }
            voxels[i].surfTauMin = tMin;
            voxels[i].surfTauMax = tMax;
        } else {
            // Voxel center is in the thin boundary shell (0 < sdf <= 0.50 * voxelSize):
            // compute its \tau values Strictly on the \phi_sdf = 0 surface so exterior voxel
            // centers never overestimate tauEnd into thin air outside the object.
            float tMin = std::numeric_limits<float>::max();
            float tMax = std::numeric_limits<float>::lowest();
            for (int dz = -1; dz <= 1; ++dz) {
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        if (dx == 0 && dy == 0 && dz == 0) {
                            continue;
                        }
                        const int nbIdx = idxAcc.getValue(c + openvdb::Coord(dx, dy, dz));
                        if (nbIdx < 0) {
                            continue;
                        }
                        const float sNb = voxels[static_cast<std::size_t>(nbIdx)].sdf;
                        if (sNb > 0.0f) {
                            continue;
                        }
                        const float tNb = voxels[static_cast<std::size_t>(nbIdx)].tau;
                        const float denom = s0 - sNb;
                        const float alpha = (denom > 1.0e-6f)
                            ? std::clamp(-sNb / denom, 0.0f, 1.0f)
                            : 0.0f;
                        const float tZero = (1.0f - alpha) * tNb + alpha * t0;
                        tMin = std::min(tMin, tZero);
                        tMax = std::max(tMax, tZero);
                    }
                }
            }
            if (tMin <= tMax) {
                voxels[i].surfTauMin = tMin;
                voxels[i].surfTauMax = tMax;
            } else {
                // Sub-voxel corner tip with no strictly negative neighbor center: project onto \phi_sdf = 0
                const float gSdfLen = std::sqrt(
                    gradSdf[0] * gradSdf[0] + gradSdf[1] * gradSdf[1] + gradSdf[2] * gradSdf[2]);
                float dTauToSurf = 0.0f;
                if (gSdfLen > 1.0e-5f) {
                    dTauToSurf = s0 * (
                        gradTau[0] * gradSdf[0] +
                        gradTau[1] * gradSdf[1] +
                        gradTau[2] * gradSdf[2]) / gSdfLen;
                }
                const float tProj = std::max(0.0f, t0 - dTauToSurf);
                voxels[i].surfTauMin = tProj;
                voxels[i].surfTauMax = tProj;
                voxels[i].cornerSdfAllowance = std::clamp(s0, 0.0f, 0.75f * voxelSize);
            }
        }

        globalTauMin = std::min(globalTauMin, voxels[i].surfTauMin);
        globalTauMax = std::max(globalTauMax, voxels[i].surfTauMax);
    }

    // Project exact CAD mesh vertices and triangle centroids onto their containing/nearest
    // interior voxel so sharp 3D corners of the CAD mesh are captured to sub-voxel accuracy.
    openvdb::tools::GridSampler<openvdb::FloatGrid::ConstAccessor, openvdb::tools::BoxSampler>
        eikSurfaceSampler(eikAccMain, eikGrid.transform());
    openvdb::tools::GridSampler<openvdb::FloatGrid::ConstAccessor, openvdb::tools::BoxSampler>
        sdfSurfaceSampler(sdfAccMain, sdfGrid.transform());

    auto registerSurfacePoint = [&](double wx, double wy, double wz) {
        const openvdb::Vec3d wp(wx, wy, wz);
        const float tauSurf = eikSurfaceSampler.wsSample(wp);
        if (!std::isfinite(tauSurf) || tauSurf >= validTauLimit) {
            return;
        }
        const openvdb::Coord ijk = eikGrid.transform().worldToIndexCellCentered(wp);
        int vIdx = idxAcc.getValue(ijk);
        if (vIdx < 0) {
            int bestDistSq = 100;
            for (int dz = -2; dz <= 2; ++dz) {
                for (int dy = -2; dy <= 2; ++dy) {
                    for (int dx = -2; dx <= 2; ++dx) {
                        const int dSq = dx * dx + dy * dy + dz * dz;
                        if (dSq >= bestDistSq) {
                            continue;
                        }
                        const int cand = idxAcc.getValue(ijk + openvdb::Coord(dx, dy, dz));
                        if (cand >= 0) {
                            bestDistSq = dSq;
                            vIdx = cand;
                        }
                    }
                }
            }
        }
        if (vIdx >= 0) {
            auto& vx = voxels[static_cast<std::size_t>(vIdx)];
            vx.surfTauMin = std::min(vx.surfTauMin, tauSurf);
            vx.surfTauMax = std::max(vx.surfTauMax, tauSurf);
            const float sdfAtPt = sdfSurfaceSampler.wsSample(wp);
            if (std::isfinite(sdfAtPt) && sdfAtPt > 0.0f) {
                vx.cornerSdfAllowance = std::max(
                    vx.cornerSdfAllowance,
                    std::clamp(sdfAtPt, 0.0f, 0.75f * voxelSize));
            }
            globalTauMin = std::min(globalTauMin, tauSurf);
            globalTauMax = std::max(globalTauMax, tauSurf);
        }
    };

    for (const auto& v : rec.vertexBuffer) {
        registerSurfacePoint(v.x, v.y, v.z);
    }

    if (globalTauMax <= globalTauMin + 1.0e-4f) {
        return result;
    }

    // ------------------------------------------------------------------------
    // 2. Construct Topological Branch Tree via Descending Superlevel-Set Split Tree
    // ------------------------------------------------------------------------
    // Unlike thin \Delta\tau slabs (which fragment into disconnected 1D strips whenever
    // isosurfaces are tilted across the grid), the superlevel sets U(t) = {x : \tau(x) >= t}
    // are solid 3D volumes that remain 26-connected at arbitrary tilt angles right up to the
    // sharpest 3D corners. Sweeping t downward from \tau_max to \tau_min via Union-Find
    // detects genuine multi-tower saddle splits while preserving exact branch peak isovalues.
    struct MacroBranch {
        int id = 0;
        int primaryParentId = -1;
        std::vector<int> parentBranches;
        std::vector<int> childBranches;
        std::vector<int> voxelIndices;
        float tauStart = 0.0f;
        float tauEnd = 0.0f;
        float startCornerSdfAllowance = 0.0f;
        float peakCornerSdfAllowance = 0.0f;
        bool isRoot = true;
        bool isLeaf = true;

        // 1D profile across [tauStart, tauEnd]
        enum : int { kNumBins = 64 };
        std::array<float, kNumBins> gMean{};
        std::array<float, kNumBins> gMin{};
        std::array<float, kNumBins> gMax{};
        std::array<float, kNumBins + 1> cumArcS{};
        float totalArcLengthMm = 0.0f;

        // Selected layer isovalues and thicknesses on this branch
        std::vector<float> layerIsovalues;
        std::vector<float> layerMeanThickness;
        std::vector<float> layerMinThickness;
        std::vector<float> layerMaxThickness;
        std::vector<bool> layerIsFirst;
        std::vector<bool> layerIsTop;

        float tauToArcS(float tau) const noexcept {
            if (tauEnd <= tauStart + 1.0e-6f) {
                return 0.0f;
            }
            const float u = std::clamp((tau - tauStart) / (tauEnd - tauStart), 0.0f, 1.0f);
            const float fBin = u * static_cast<float>(kNumBins);
            const int b0 = std::clamp(static_cast<int>(std::floor(fBin)), 0, kNumBins - 1);
            const float frac = fBin - static_cast<float>(b0);
            return cumArcS[b0] + frac * (cumArcS[b0 + 1] - cumArcS[b0]);
        }

        float arcSToTau(float s) const noexcept {
            if (totalArcLengthMm <= 1.0e-6f || tauEnd <= tauStart + 1.0e-6f) {
                return tauStart;
            }
            const float clampedS = std::clamp(s, 0.0f, totalArcLengthMm);
            auto it = std::lower_bound(cumArcS.begin(), cumArcS.end(), clampedS);
            int idx = static_cast<int>(std::distance(cumArcS.begin(), it));
            if (idx <= 0) {
                return tauStart;
            }
            if (idx > kNumBins) {
                return tauEnd;
            }
            const int b0 = idx - 1;
            const float ds = cumArcS[b0 + 1] - cumArcS[b0];
            const float frac = (ds > 1.0e-7f) ? (clampedS - cumArcS[b0]) / ds : 0.0f;
            const float u = (static_cast<float>(b0) + std::clamp(frac, 0.0f, 1.0f)) /
                            static_cast<float>(kNumBins);
            return tauStart + u * (tauEnd - tauStart);
        }

        void queryGradRange(float tauA, float tauB, float& outGMin, float& outGMean, float& outGMax) const noexcept {
            if (tauEnd <= tauStart + 1.0e-6f) {
                outGMin = outGMean = outGMax = 1.0f;
                return;
            }
            const float u0 = std::clamp((std::min(tauA, tauB) - tauStart) / (tauEnd - tauStart), 0.0f, 1.0f);
            const float u1 = std::clamp((std::max(tauA, tauB) - tauStart) / (tauEnd - tauStart), 0.0f, 1.0f);
            const int bStart = std::clamp(
                static_cast<int>(std::floor(u0 * static_cast<float>(kNumBins))),
                0,
                kNumBins - 1);
            const int bEnd = std::clamp(
                static_cast<int>(std::floor(u1 * static_cast<float>(kNumBins))),
                bStart,
                kNumBins - 1);
            outGMin = std::numeric_limits<float>::max();
            outGMax = 0.0f;
            float sumMean = 0.0f;
            int cnt = 0;
            for (int b = bStart; b <= bEnd; ++b) {
                outGMin = std::min(outGMin, gMin[b]);
                outGMax = std::max(outGMax, gMax[b]);
                sumMean += gMean[b];
                ++cnt;
            }
            outGMean = (cnt > 0) ? (sumMean / static_cast<float>(cnt)) : 1.0f;
            if (outGMin > outGMax) {
                outGMin = outGMax = outGMean;
            }
        }
    };

    struct SuperComponent {
        float peakTau = std::numeric_limits<float>::lowest();
        float minTau = std::numeric_limits<float>::max();
        float startCornerSdfAllowance = 0.0f;
        float peakCornerSdfAllowance = 0.0f;
        std::vector<int> voxelIndices;
        std::vector<int> childBranches;
        bool active = true;
    };

    const std::size_t numVoxels = voxels.size();
    std::vector<int> descOrder(numVoxels);
    std::iota(descOrder.begin(), descOrder.end(), 0);
    std::sort(descOrder.begin(), descOrder.end(), [&](int a, int b) {
        if (std::abs(voxels[a].tau - voxels[b].tau) > 1.0e-6f) {
            return voxels[a].tau > voxels[b].tau;
        }
        return voxels[a].surfTauMax > voxels[b].surfTauMax;
    });

    std::vector<int> ufParent(numVoxels, -1);
    std::vector<int> rootToComp(numVoxels, -1);
    std::vector<std::uint8_t> swept(numVoxels, 0);
    std::vector<SuperComponent> superComps;
    superComps.reserve(64);

    auto ufFind = [&](int x) -> int {
        int r = x;
        while (ufParent[r] != r) {
            r = ufParent[r];
        }
        int curr = x;
        while (curr != r) {
            const int nxt = ufParent[curr];
            ufParent[curr] = r;
            curr = nxt;
        }
        return r;
    };

    auto mergeSuperComps = [&](int dstCompId, int srcCompId) {
        if (dstCompId == srcCompId) {
            return;
        }
        auto& dst = superComps[static_cast<std::size_t>(dstCompId)];
        auto& src = superComps[static_cast<std::size_t>(srcCompId)];
        if (!src.active) {
            return;
        }
        if (src.peakTau >= dst.peakTau - 0.5f * voxelSize) {
            dst.peakCornerSdfAllowance = std::max(dst.peakCornerSdfAllowance, src.peakCornerSdfAllowance);
        }
        if (src.minTau <= dst.minTau + 0.5f * voxelSize) {
            dst.startCornerSdfAllowance = std::max(dst.startCornerSdfAllowance, src.startCornerSdfAllowance);
        }
        if (dst.childBranches.empty() && src.childBranches.empty()) {
            dst.peakTau = std::max(dst.peakTau, src.peakTau);
        } else if (dst.childBranches.empty() && !src.childBranches.empty()) {
            dst.peakTau = src.peakTau;
        } else if (!dst.childBranches.empty() && !src.childBranches.empty()) {
            dst.peakTau = std::min(dst.peakTau, src.peakTau);
        }
        dst.minTau = std::min(dst.minTau, src.minTau);
        dst.voxelIndices.insert(dst.voxelIndices.end(), src.voxelIndices.begin(), src.voxelIndices.end());
        dst.childBranches.insert(dst.childBranches.end(), src.childBranches.begin(), src.childBranches.end());
        src.voxelIndices.clear();
        src.childBranches.clear();
        src.active = false;
    };

    const float minBranchPersistenceMm = std::max({2.5f * voxelSize, 3.0f * hMax, 1.5f});
    const std::size_t minBranchVoxels = 12;

    std::vector<MacroBranch> rawBranches;

    for (const int v : descOrder) {
        swept[v] = 1;
        ufParent[v] = v;

        const openvdb::Coord& c = voxels[v].ijk;
        std::vector<int> nbRoots;
        std::vector<int> nbComps;

        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0 && dz == 0) {
                        continue;
                    }
                    const int nb = idxAcc.getValue(c + openvdb::Coord(dx, dy, dz));
                    if (nb >= 0 && swept[nb]) {
                        const int r = ufFind(nb);
                        const int cId = rootToComp[r];
                        if (cId >= 0 && superComps[static_cast<std::size_t>(cId)].active) {
                            if (std::find(nbComps.begin(), nbComps.end(), cId) == nbComps.end()) {
                                nbRoots.push_back(r);
                                nbComps.push_back(cId);
                            }
                        }
                    }
                }
            }
        }

        if (nbComps.empty()) {
            // New local maximum peak in the superlevel set
            const int cId = static_cast<int>(superComps.size());
            SuperComponent sc{};
            sc.peakTau = voxels[v].surfTauMax;
            sc.minTau = voxels[v].surfTauMin;
            sc.peakCornerSdfAllowance = voxels[v].cornerSdfAllowance;
            sc.startCornerSdfAllowance = voxels[v].cornerSdfAllowance;
            sc.voxelIndices.push_back(v);
            sc.active = true;
            superComps.push_back(std::move(sc));
            rootToComp[v] = cId;
        } else if (nbComps.size() == 1) {
            // Single continuing superlevel component
            const int r0 = nbRoots[0];
            const int cId = nbComps[0];
            ufParent[v] = r0;
            auto& sc = superComps[static_cast<std::size_t>(cId)];
            if (voxels[v].surfTauMax >= sc.peakTau - 1.5f * voxelSize) {
                sc.peakCornerSdfAllowance = std::max(sc.peakCornerSdfAllowance, voxels[v].cornerSdfAllowance);
            }
            if (voxels[v].surfTauMin <= sc.minTau + 1.5f * voxelSize) {
                sc.startCornerSdfAllowance = std::max(sc.startCornerSdfAllowance, voxels[v].cornerSdfAllowance);
            }
            if (sc.childBranches.empty()) {
                sc.peakTau = std::max(sc.peakTau, voxels[v].surfTauMax);
            }
            sc.minTau = std::min(sc.minTau, voxels[v].surfTauMin);
            sc.voxelIndices.push_back(v);
        } else {
            // Saddle point where >= 2 upper superlevel components meet at tauSplit
            const float tauSplit = voxels[v].tau;
            std::vector<int> sigComps;
            std::vector<int> minorComps;

            for (const int cId : nbComps) {
                const auto& sc = superComps[static_cast<std::size_t>(cId)];
                const float persistence = sc.peakTau - tauSplit;
                const bool isSignificant = !sc.childBranches.empty() ||
                    (persistence >= minBranchPersistenceMm && sc.voxelIndices.size() >= minBranchVoxels);
                if (isSignificant) {
                    sigComps.push_back(cId);
                } else {
                    minorComps.push_back(cId);
                }
            }

            // Union all neighbor roots with v
            const int masterRoot = nbRoots[0];
            for (std::size_t j = 1; j < nbRoots.size(); ++j) {
                const int rj = ufFind(nbRoots[j]);
                if (rj != masterRoot) {
                    ufParent[rj] = masterRoot;
                }
            }
            ufParent[v] = masterRoot;

            if (sigComps.size() <= 1) {
                // Not a genuine topological branch split: merge minor discretization bumps
                // into the main component while preserving the exact global peakTau!
                int mainComp = !sigComps.empty() ? sigComps[0] : nbComps[0];
                if (sigComps.empty()) {
                    for (const int cId : nbComps) {
                        if (superComps[static_cast<std::size_t>(cId)].peakTau >
                            superComps[static_cast<std::size_t>(mainComp)].peakTau) {
                            mainComp = cId;
                        }
                    }
                }
                for (const int cId : nbComps) {
                    if (cId != mainComp) {
                        mergeSuperComps(mainComp, cId);
                    }
                }
                auto& sc = superComps[static_cast<std::size_t>(mainComp)];
                if (voxels[v].surfTauMax >= sc.peakTau - 1.5f * voxelSize) {
                    sc.peakCornerSdfAllowance = std::max(sc.peakCornerSdfAllowance, voxels[v].cornerSdfAllowance);
                }
                if (voxels[v].surfTauMin <= sc.minTau + 1.5f * voxelSize) {
                    sc.startCornerSdfAllowance = std::max(sc.startCornerSdfAllowance, voxels[v].cornerSdfAllowance);
                }
                if (sc.childBranches.empty()) {
                    sc.peakTau = std::max(sc.peakTau, voxels[v].surfTauMax);
                }
                sc.minTau = std::min(sc.minTau, voxels[v].surfTauMin);
                sc.voxelIndices.push_back(v);
                rootToComp[masterRoot] = mainComp;
            } else {
                // Genuine topological split at tauSplit into >= 2 significant upper branches!
                int bestSig = sigComps[0];
                for (const int cId : sigComps) {
                    if (superComps[static_cast<std::size_t>(cId)].peakTau >
                        superComps[static_cast<std::size_t>(bestSig)].peakTau) {
                        bestSig = cId;
                    }
                }
                for (const int mId : minorComps) {
                    mergeSuperComps(bestSig, mId);
                }

                std::vector<int> finalizedChildIds;
                finalizedChildIds.reserve(sigComps.size());
                std::vector<int> absorbedVoxels;
                for (const int cId : sigComps) {
                    auto& sc = superComps[static_cast<std::size_t>(cId)];
                    const float branchSpanTau = sc.peakTau - tauSplit;
                    // If this component is an intermediate trunk between an upper split and tauSplit
                    // whose own tau span is thinner than hMin (e.g. a multi-way saddle split across
                    // adjacent voxels), collapse it into the new parent trunk rather than creating
                    // a degenerate sub-layer intermediate branch.
                    if (!sc.childBranches.empty() && branchSpanTau < hMin) {
                        absorbedVoxels.insert(
                            absorbedVoxels.end(),
                            sc.voxelIndices.begin(),
                            sc.voxelIndices.end());
                        for (const int grandChildId : sc.childBranches) {
                            rawBranches[static_cast<std::size_t>(grandChildId)].tauStart = tauSplit;
                            finalizedChildIds.push_back(grandChildId);
                        }
                        sc.voxelIndices.clear();
                        sc.childBranches.clear();
                        sc.active = false;
                        continue;
                    }

                    const int bId = static_cast<int>(rawBranches.size());
                    MacroBranch mb{};
                    mb.id = bId;
                    mb.tauStart = tauSplit;
                    mb.tauEnd = std::max(sc.peakTau, tauSplit + 1.0e-3f);
                    mb.startCornerSdfAllowance = sc.startCornerSdfAllowance;
                    mb.peakCornerSdfAllowance = sc.peakCornerSdfAllowance;
                    mb.isRoot = false;
                    mb.isLeaf = sc.childBranches.empty();
                    mb.childBranches = std::move(sc.childBranches);
                    mb.voxelIndices = std::move(sc.voxelIndices);
                    sc.active = false;

                    rawBranches.push_back(std::move(mb));
                    finalizedChildIds.push_back(bId);
                }

                // Start a new parent trunk SuperComponent below tauSplit
                const int trunkCompId = static_cast<int>(superComps.size());
                SuperComponent trunk{};
                trunk.peakTau = tauSplit;
                trunk.minTau = voxels[v].surfTauMin;
                trunk.peakCornerSdfAllowance = voxels[v].cornerSdfAllowance;
                trunk.startCornerSdfAllowance = voxels[v].cornerSdfAllowance;
                trunk.voxelIndices = std::move(absorbedVoxels);
                trunk.voxelIndices.push_back(v);
                trunk.childBranches = std::move(finalizedChildIds);
                trunk.active = true;
                superComps.push_back(std::move(trunk));
                rootToComp[masterRoot] = trunkCompId;
            }
        }
    }

    // Finalize remaining active root SuperComponents at the bottom of the sweep
    std::vector<int> activeRootComps;
    for (std::size_t cId = 0; cId < superComps.size(); ++cId) {
        if (superComps[cId].active && !superComps[cId].voxelIndices.empty()) {
            activeRootComps.push_back(static_cast<int>(cId));
        }
    }

    if (activeRootComps.size() > 1) {
        int largestRoot = activeRootComps[0];
        for (const int cId : activeRootComps) {
            if (superComps[static_cast<std::size_t>(cId)].voxelIndices.size() >
                superComps[static_cast<std::size_t>(largestRoot)].voxelIndices.size()) {
                largestRoot = cId;
            }
        }
        std::vector<int> keptRoots;
        for (const int cId : activeRootComps) {
            const auto& sc = superComps[static_cast<std::size_t>(cId)];
            const bool isSig = (cId == largestRoot) || !sc.childBranches.empty() ||
                ((sc.peakTau - sc.minTau) >= minBranchPersistenceMm &&
                 sc.voxelIndices.size() >= minBranchVoxels);
            if (isSig) {
                keptRoots.push_back(cId);
            } else {
                mergeSuperComps(largestRoot, cId);
            }
        }
        activeRootComps = std::move(keptRoots);
    }

    for (const int cId : activeRootComps) {
        auto& sc = superComps[static_cast<std::size_t>(cId)];
        if (!sc.active || sc.voxelIndices.empty()) {
            continue;
        }
        const int bId = static_cast<int>(rawBranches.size());
        MacroBranch mb{};
        mb.id = bId;
        mb.tauStart = std::min(sc.minTau, std::max(0.0f, globalTauMin));
        mb.tauEnd = std::max(sc.peakTau, mb.tauStart + 1.0e-3f);
        mb.startCornerSdfAllowance = sc.startCornerSdfAllowance;
        mb.peakCornerSdfAllowance = sc.peakCornerSdfAllowance;
        mb.isRoot = true;
        mb.isLeaf = sc.childBranches.empty();
        mb.childBranches = std::move(sc.childBranches);
        mb.voxelIndices = std::move(sc.voxelIndices);
        sc.active = false;
        rawBranches.push_back(std::move(mb));
    }

    if (rawBranches.empty()) {
        MacroBranch mb{};
        mb.id = 0;
        mb.tauStart = std::max(0.0f, globalTauMin);
        mb.tauEnd = globalTauMax;
        mb.isRoot = true;
        mb.isLeaf = true;
        for (std::size_t i = 0; i < numVoxels; ++i) {
            mb.voxelIndices.push_back(static_cast<int>(i));
        }
        rawBranches.push_back(std::move(mb));
    }

    // Order branches ascending by tauStart (so root trunk is B0, child branches follow)
    std::vector<int> branchOrder(rawBranches.size());
    std::iota(branchOrder.begin(), branchOrder.end(), 0);
    std::sort(branchOrder.begin(), branchOrder.end(), [&](int a, int b) {
        if (rawBranches[a].isRoot != rawBranches[b].isRoot) {
            return rawBranches[a].isRoot > rawBranches[b].isRoot;
        }
        if (std::abs(rawBranches[a].tauStart - rawBranches[b].tauStart) > 1.0e-4f) {
            return rawBranches[a].tauStart < rawBranches[b].tauStart;
        }
        return rawBranches[a].tauEnd < rawBranches[b].tauEnd;
    });

    std::vector<int> oldToNewBranch(rawBranches.size(), 0);
    for (std::size_t newIdx = 0; newIdx < branchOrder.size(); ++newIdx) {
        oldToNewBranch[static_cast<std::size_t>(branchOrder[newIdx])] = static_cast<int>(newIdx);
    }

    std::vector<MacroBranch> branches(rawBranches.size());
    for (std::size_t newIdx = 0; newIdx < branchOrder.size(); ++newIdx) {
        MacroBranch mb = std::move(rawBranches[static_cast<std::size_t>(branchOrder[newIdx])]);
        mb.id = static_cast<int>(newIdx);
        for (int& ch : mb.childBranches) {
            ch = oldToNewBranch[static_cast<std::size_t>(ch)];
        }
        branches[newIdx] = std::move(mb);
    }

    for (std::size_t b = 0; b < branches.size(); ++b) {
        for (const int ch : branches[b].childBranches) {
            branches[static_cast<std::size_t>(ch)].parentBranches.push_back(static_cast<int>(b));
            branches[static_cast<std::size_t>(ch)].primaryParentId = static_cast<int>(b);
            branches[static_cast<std::size_t>(ch)].isRoot = false;
        }
        branches[b].isLeaf = branches[b].childBranches.empty();
    }

    // Populate 3D branchGrid and dilate by 3 voxels into the exterior narrow band
    // so boundary triangles cut by volumeToMesh always match their branch cleanly.
    openvdb::Int32Grid branchGrid(-1);
    branchGrid.setTransform(eikGrid.transform().copy());
    auto branchAcc = branchGrid.getAccessor();

    for (std::size_t b = 0; b < branches.size(); ++b) {
        for (const int vIdx : branches[b].voxelIndices) {
            voxels[static_cast<std::size_t>(vIdx)].branchId = static_cast<int>(b);
            branchAcc.setValue(voxels[static_cast<std::size_t>(vIdx)].ijk, static_cast<int>(b));
        }
    }

    if (branches.size() > 1) {
        for (int pass = 0; pass < 3; ++pass) {
            std::vector<std::pair<openvdb::Coord, int>> dilationUpdates;
            for (auto iter = branchGrid.cbeginValueOn(); iter; ++iter) {
                const int bId = *iter;
                if (bId < 0) {
                    continue;
                }
                const openvdb::Coord c = iter.getCoord();
                for (int dz = -1; dz <= 1; ++dz) {
                    for (int dy = -1; dy <= 1; ++dy) {
                        for (int dx = -1; dx <= 1; ++dx) {
                            if (dx == 0 && dy == 0 && dz == 0) {
                                continue;
                            }
                            const openvdb::Coord nb = c + openvdb::Coord(dx, dy, dz);
                            if (branchAcc.getValue(nb) < 0) {
                                dilationUpdates.emplace_back(nb, bId);
                            }
                        }
                    }
                }
            }
            for (const auto& [coord, bId] : dilationUpdates) {
                if (branchAcc.getValue(coord) < 0) {
                    branchAcc.setValue(coord, bId);
                }
            }
        }
    }

    // ------------------------------------------------------------------------
    // 3. Build 1D Gradient & Cumulative Arc-Length Profiles S_b(\tau) per Branch
    // ------------------------------------------------------------------------
    auto rebuildBranchArcLengthProfile = [&](MacroBranch& mb) {
        constexpr int M = MacroBranch::kNumBins;
        const float t0 = mb.tauStart;
        const float t1 = std::max(mb.tauEnd, t0 + 1.0e-4f);
        const float span = t1 - t0;

        std::array<double, M> sumG{};
        std::array<int, M> countG{};
        for (int m = 0; m < M; ++m) {
            mb.gMin[m] = std::numeric_limits<float>::max();
            mb.gMax[m] = 0.0f;
            mb.gMean[m] = 1.0f;
        }

        for (const int vIdx : mb.voxelIndices) {
            const float t = voxels[static_cast<std::size_t>(vIdx)].tau;
            const float g = voxels[static_cast<std::size_t>(vIdx)].gradMag;
            const int m = std::clamp(
                static_cast<int>(std::floor(((t - t0) / span) * static_cast<float>(M))),
                0,
                M - 1);
            sumG[m] += static_cast<double>(g);
            countG[m] += 1;
            mb.gMin[m] = std::min(mb.gMin[m], g);
            mb.gMax[m] = std::max(mb.gMax[m], g);
        }

        float fallbackMean = 1.0f;
        double totalSum = 0.0;
        int totalCnt = 0;
        for (int m = 0; m < M; ++m) {
            totalSum += sumG[m];
            totalCnt += countG[m];
        }
        if (totalCnt > 0) {
            fallbackMean = static_cast<float>(totalSum / static_cast<double>(totalCnt));
        }

        for (int m = 0; m < M; ++m) {
            if (countG[m] > 0) {
                mb.gMean[m] = static_cast<float>(sumG[m] / static_cast<double>(countG[m]));
            } else {
                mb.gMean[m] = fallbackMean;
            }
        }

        for (int m = 1; m < M; ++m) {
            if (countG[m] == 0) {
                mb.gMean[m] = mb.gMean[m - 1];
                mb.gMin[m] = mb.gMin[m - 1];
                mb.gMax[m] = mb.gMax[m - 1];
            }
        }
        for (int m = M - 2; m >= 0; --m) {
            if (countG[m] == 0 || mb.gMin[m] > mb.gMax[m]) {
                mb.gMean[m] = mb.gMean[m + 1];
                mb.gMin[m] = mb.gMin[m + 1];
                mb.gMax[m] = mb.gMax[m + 1];
            }
        }
        for (int m = 0; m < M; ++m) {
            if (mb.gMin[m] > mb.gMax[m]) {
                mb.gMin[m] = mb.gMax[m] = mb.gMean[m];
            }
        }

        std::array<float, M> smoothed = mb.gMean;
        for (int pass = 0; pass < 2; ++pass) {
            for (int m = 1; m + 1 < M; ++m) {
                smoothed[m] = 0.25f * mb.gMean[m - 1] + 0.50f * mb.gMean[m] + 0.25f * mb.gMean[m + 1];
            }
            mb.gMean = smoothed;
        }

        const float dTau = span / static_cast<float>(M);
        mb.cumArcS[0] = 0.0f;
        for (int m = 0; m < M; ++m) {
            const float g = std::clamp(mb.gMean[m], 0.20f, 4.0f);
            mb.cumArcS[m + 1] = mb.cumArcS[m] + dTau / g;
        }
        mb.totalArcLengthMm = mb.cumArcS[M];
    };

    for (auto& mb : branches) {
        rebuildBranchArcLengthProfile(mb);
    }

    // ------------------------------------------------------------------------
    // 4. Bidirectional Layer Relaxation Solver per Branch
    // ------------------------------------------------------------------------
    // Keep parent/child branch boundaries anchored at the exact topological saddle
    // isovalue tauSplit so branch level sets never cross into sibling/child voxel domains.
    // Solve the bidirectional layer sequence on each branch b:
    // - Root branches emit their first layer at s_0 = hFirst (is_first_layer = true).
    // - Non-root branches start their interval at s_0 = 0 (which is the parent's junction layer,
    //   already emitted by the parent branch) and emit layers k = 1 .. N_b.
    // - Leaf branches anchor their final layer at s_{N_b} = L_b - hTop (is_top_layer = true).
    // - Internal branches anchor their final layer at the junction s_{N_b} = L_b.
    for (auto& mb : branches) {
        const float L = mb.totalArcLengthMm;
        if (L <= 1.0e-4f) {
            continue;
        }

        const float sStart = mb.isRoot ? std::min(hFirst, 0.50f * L) : 0.0f;
        const float sEnd = mb.isLeaf
            ? std::max(sStart, L - std::min(hTop, 0.50f * L))
            : L;
        const float spanS = std::max(0.0f, sEnd - sStart);

        // Very thin branch case: always emit the anchor layer so no branch is ever dropped
        if (spanS < 0.5f * hMin) {
            const float sAnchor = mb.isRoot ? sStart : sEnd;
            const float tauAnchor = mb.arcSToTau(sAnchor);
            const float thick = std::max(0.01f, sAnchor);
            mb.layerIsovalues.push_back(tauAnchor);
            mb.layerMeanThickness.push_back(thick);
            mb.layerMinThickness.push_back(thick);
            mb.layerMaxThickness.push_back(thick);
            mb.layerIsFirst.push_back(mb.isRoot);
            mb.layerIsTop.push_back(mb.isLeaf);
            continue;
        }

        const int nMin = std::max(1, static_cast<int>(std::ceil(spanS / hMax - 1.0e-5f)));
        const int nMax = std::max(1, static_cast<int>(std::floor(spanS / hMin + 1.0e-5f)));
        const int nIdeal = std::max(1, static_cast<int>(std::round(spanS / hTarget)));
        const int N = (nMin <= nMax) ? std::clamp(nIdeal, nMin, nMax) : nIdeal;

        std::vector<float> sNodes(static_cast<std::size_t>(N) + 1, 0.0f);
        for (int k = 0; k <= N; ++k) {
            const float u = static_cast<float>(k) / static_cast<float>(N);
            sNodes[k] = sStart + u * spanS;
        }

        // Bidirectional (forward bottom-up + backward top-down) Gauss-Seidel relaxation
        // accounting for local gradient variations so physical thickness errors are evenly
        // distributed across the entire branch while strictly respecting [hMin, hMax].
        if (N >= 2) {
            for (int iter = 0; iter < 40; ++iter) {
                // Forward pass: k = 1 .. N - 1 (bottom-up from bed/junction toward peak)
                for (int k = 1; k < N; ++k) {
                    const float sMid = 0.5f * (sNodes[k - 1] + sNodes[k + 1]);
                    const float lo = std::max(sNodes[k - 1] + hMin, sNodes[k + 1] - hMax);
                    const float hi = std::min(sNodes[k - 1] + hMax, sNodes[k + 1] - hMin);
                    sNodes[k] = (lo <= hi) ? std::clamp(sMid, lo, hi) : sMid;
                }

                // Backward pass: k = N - 1 .. 1 (top-down from peak/junction toward bed)
                for (int k = N - 1; k >= 1; --k) {
                    const float sMid = 0.5f * (sNodes[k - 1] + sNodes[k + 1]);
                    const float lo = std::max(sNodes[k - 1] + hMin, sNodes[k + 1] - hMax);
                    const float hi = std::min(sNodes[k - 1] + hMax, sNodes[k + 1] - hMin);
                    sNodes[k] = (lo <= hi) ? std::clamp(sMid, lo, hi) : sMid;
                }
            }
        }

        const int kFirstEmit = mb.isRoot ? 0 : 1;
        for (int k = kFirstEmit; k <= N; ++k) {
            const float sCurr = sNodes[k];
            const float sPrev = (k == 0) ? 0.0f : sNodes[k - 1];
            const float tauCurr = mb.arcSToTau(sCurr);
            const float tauPrev = (k == 0) ? mb.tauStart : mb.arcSToTau(sPrev);
            const float dTau = std::max(0.0f, tauCurr - tauPrev);

            float gMinLocal = 1.0f, gMeanLocal = 1.0f, gMaxLocal = 1.0f;
            mb.queryGradRange(tauPrev, tauCurr, gMinLocal, gMeanLocal, gMaxLocal);

            const float meanH = std::max(0.01f, sCurr - sPrev);
            const float rawMinH = (gMaxLocal > 1.0e-4f) ? (dTau / gMaxLocal) : meanH;
            const float rawMaxH = (gMinLocal > 1.0e-4f) ? (dTau / gMinLocal) : meanH;
            const bool isFirst = (mb.isRoot && k == 0);
            const bool isTop = (mb.isLeaf && k == N);

            const float hiClamp = std::max(2.0f * hMax, meanH);
            const float minH = isFirst ? meanH : std::clamp(std::min(rawMinH, meanH), 0.02f, meanH);
            const float maxH = isFirst ? meanH : std::clamp(std::max(rawMaxH, meanH), meanH, hiClamp);

            mb.layerIsovalues.push_back(tauCurr);
            mb.layerMeanThickness.push_back(meanH);
            mb.layerMinThickness.push_back(minH);
            mb.layerMaxThickness.push_back(maxH);
            mb.layerIsFirst.push_back(isFirst);
            mb.layerIsTop.push_back(isTop);
        }
    }

    // ------------------------------------------------------------------------
    // 5. Flatten, Sort, and Parallel-Extract 3D Clipped Layer Meshes & Contours
    // ------------------------------------------------------------------------
    struct PendingLayer {
        std::uint32_t branchId = 0;
        std::int32_t parentBranchId = -1;
        std::uint32_t branchLayerIndex = 0;
        std::uint32_t branchLayerCount = 0;
        float isovalue = 0.0f;
        float meanThicknessMm = 0.0f;
        float minThicknessMm = 0.0f;
        float maxThicknessMm = 0.0f;
        float sdfClipAllowance = 0.0f;
        bool isFirstLayer = false;
        bool isTopLayer = false;
    };

    std::vector<PendingLayer> pending;
    const float cornerRampTauSpan = std::max(2.0f * voxelSize, 1.5f * hMax);

    for (const auto& mb : branches) {
        const auto count = static_cast<std::uint32_t>(mb.layerIsovalues.size());
        for (std::uint32_t i = 0; i < count; ++i) {
            const float iso = mb.layerIsovalues[i];
            float allowance = 0.0f;
            if (mb.isRoot && mb.startCornerSdfAllowance > 0.0f) {
                const float u = std::clamp(1.0f - (iso - mb.tauStart) / cornerRampTauSpan, 0.0f, 1.0f);
                allowance = std::max(allowance, u * mb.startCornerSdfAllowance);
            }
            if (mb.isLeaf && mb.peakCornerSdfAllowance > 0.0f) {
                const float u = std::clamp(1.0f - (mb.tauEnd - iso) / cornerRampTauSpan, 0.0f, 1.0f);
                allowance = std::max(allowance, u * mb.peakCornerSdfAllowance);
            }

            PendingLayer pl{};
            pl.branchId = static_cast<std::uint32_t>(mb.id);
            pl.parentBranchId = static_cast<std::int32_t>(mb.primaryParentId);
            pl.branchLayerIndex = i;
            pl.branchLayerCount = count;
            pl.isovalue = iso;
            pl.meanThicknessMm = mb.layerMeanThickness[i];
            pl.minThicknessMm = mb.layerMinThickness[i];
            pl.maxThicknessMm = mb.layerMaxThickness[i];
            pl.sdfClipAllowance = allowance;
            pl.isFirstLayer = mb.layerIsFirst[i];
            pl.isTopLayer = mb.layerIsTop[i];
            pending.push_back(pl);
        }
    }

    if (pending.empty()) {
        return result;
    }

    std::sort(pending.begin(), pending.end(), [](const PendingLayer& a, const PendingLayer& b) {
        if (std::abs(a.isovalue - b.isovalue) > 1.0e-5f) {
            return a.isovalue < b.isovalue;
        }
        return a.branchId < b.branchId;
    });

    const std::size_t numLayers = pending.size();
    std::vector<SelectedLayerInfo> extractedLayers(numLayers);
    const std::size_t numBranches = branches.size();
    const bool multiBranch = (numBranches > 1);
    const float outerFringeLimit = sdfBg - 1.75f * voxelSize;
    // Always extract slicing layers at full voxel resolution (adaptivity = 0.0) so
    // OpenVDB never merges adaptive quads across the \phi_sdf = 0 boundary or introduces
    // interior T-junction cracks on top-surface layers.
    (void)adaptivity;

    // Precompute branch subtree reachability and representative 3D voxel sample points per branch
    // so connected surface islands can be assigned as whole topological units without ever dropping
    // individual triangles at parent/child tau-split staircase boundaries.
    std::vector<std::vector<std::uint8_t>> isSubtree(
        numBranches,
        std::vector<std::uint8_t>(numBranches, 0));
    std::vector<std::vector<openvdb::Vec3f>> branchSamplePoints(numBranches);
    if (multiBranch) {
        for (std::size_t b = 0; b < numBranches; ++b) {
            std::vector<int> stack = {static_cast<int>(b)};
            isSubtree[b][b] = 1;
            while (!stack.empty()) {
                const int curr = stack.back();
                stack.pop_back();
                for (const int ch : branches[static_cast<std::size_t>(curr)].childBranches) {
                    if (!isSubtree[b][static_cast<std::size_t>(ch)]) {
                        isSubtree[b][static_cast<std::size_t>(ch)] = 1;
                        stack.push_back(ch);
                    }
                }
            }

            const auto& vList = branches[b].voxelIndices;
            if (!vList.empty()) {
                const std::size_t maxSamples = 256;
                const std::size_t step = std::max<std::size_t>(1, vList.size() / maxSamples);
                branchSamplePoints[b].reserve(std::min(vList.size(), maxSamples));
                for (std::size_t i = 0; i < vList.size(); i += step) {
                    const openvdb::Vec3d wp =
                        eikGrid.transform().indexToWorld(voxels[static_cast<std::size_t>(vList[i])].ijk);
                    branchSamplePoints[b].emplace_back(
                        static_cast<float>(wp.x()),
                        static_cast<float>(wp.y()),
                        static_cast<float>(wp.z()));
                }
            }
        }
    }

    tbb::parallel_for(
        tbb::blocked_range<std::size_t>(0, numLayers),
        [&](const tbb::blocked_range<std::size_t>& r) {
            auto sdfAcc = sdfGrid.getConstAccessor();
            openvdb::tools::GridSampler<openvdb::FloatGrid::ConstAccessor, openvdb::tools::BoxSampler>
                sdfSampler(sdfAcc, sdfGrid.transform());
            auto fineSdfAcc = fineSdfGrid.getConstAccessor();
            openvdb::tools::GridSampler<openvdb::FloatGrid::ConstAccessor, openvdb::tools::BoxSampler>
                fineSdfSampler(fineSdfAcc, fineSdfGrid.transform());
            auto fineEikAcc = fineEikGrid.getConstAccessor();
            openvdb::tools::GridSampler<openvdb::FloatGrid::ConstAccessor, openvdb::tools::BoxSampler>
                fineEikSampler(fineEikAcc, fineEikGrid.transform());
            auto bAcc = branchGrid.getConstAccessor();

            const float fineInnerCutoff =
                -std::max(rec.fineShellInnerDepthMm - 1.75f * fineVoxelSize, 1.5f * fineVoxelSize);
            const float fineOuterCutoff = fineSdfGrid.background() - 1.5f * fineVoxelSize;
            const float shellThicknessMm = std::max(0.4f, impl_->shellPathSpec.shell_thickness_mm);

            auto sampleCompositeSdf = [&](const openvdb::Vec3d& wp) -> float {
                float baseSdf = 0.0f;
                bool usedFineGrid = false;
                if (!useDualGridExtraction) {
                    baseSdf = sdfSampler.wsSample(wp);
                    usedFineGrid = true;
                } else {
                    const float fSdf = fineSdfSampler.wsSample(wp);
                    if (fSdf <= fineInnerCutoff || fSdf >= fineOuterCutoff) {
                        baseSdf = sdfSampler.wsSample(wp);
                    } else {
                        baseSdf = fSdf;
                        usedFineGrid = true;
                    }
                }

                // Compensate for Jensen's positive trilinear interpolation bias across thin curved
                // walls (where \phi_sdf forms a V-shaped medial valley whose slopes have opposite
                // signs on either side of the wall). Monotonic single-sided boundaries of thick
                // parts have identical slope signs on both sides and are completely unaffected.
                if (usedFineGrid &&
                    baseSdf > -0.50f * fineVoxelSize &&
                    baseSdf < 0.90f * fineVoxelSize) {
                    const openvdb::Vec3d idxPt = fineSdfGrid.transform().worldToIndex(wp);
                    const openvdb::Coord c = fineSdfGrid.transform().worldToIndexCellCentered(wp);
                    const float cSdf = fineSdfAcc.getValue(c);
                    const float slopeThresh = 0.15f * fineVoxelSize;
                    float maxValleyCorr = 0.0f;

                    auto evalAxisValleyCorr = [&](int axis, double coordVal) {
                        const int i0 = static_cast<int>(std::floor(coordVal));
                        const float f = static_cast<float>(coordVal - static_cast<double>(i0));
                        auto coordAt = [&](int idxAlongAxis) -> openvdb::Coord {
                            if (axis == 0) return openvdb::Coord(idxAlongAxis, c.y(), c.z());
                            if (axis == 1) return openvdb::Coord(c.x(), idxAlongAxis, c.z());
                            return openvdb::Coord(c.x(), c.y(), idxAlongAxis);
                        };
                        const float sM1 = fineSdfAcc.getValue(coordAt(i0 - 1));
                        const float s0  = fineSdfAcc.getValue(coordAt(i0));
                        const float s1  = fineSdfAcc.getValue(coordAt(i0 + 1));
                        const float s2  = fineSdfAcc.getValue(coordAt(i0 + 2));
                        const float gL = s0 - sM1;
                        const float gR = s2 - s1;
                        if (gL < -slopeThresh && gR > slopeThresh) {
                            const float sLin = (1.0f - f) * s0 + f * s1;
                            const float sVal = std::max(s0 + f * gL, s1 - (1.0f - f) * gR);
                            if (sVal < sLin) {
                                maxValleyCorr = std::max(maxValleyCorr, sLin - sVal);
                            }
                        }
                    };

                    evalAxisValleyCorr(0, idxPt.x());
                    evalAxisValleyCorr(1, idxPt.y());
                    evalAxisValleyCorr(2, idxPt.z());

                    const float valleyMargin = cSdf + slopeThresh;
                    const bool isCellValley =
                        (cSdf < baseSdf) &&
                        ((fineSdfAcc.getValue(c.offsetBy(-1, 0, 0)) > valleyMargin &&
                          fineSdfAcc.getValue(c.offsetBy(1, 0, 0)) > valleyMargin) ||
                         (fineSdfAcc.getValue(c.offsetBy(0, -1, 0)) > valleyMargin &&
                          fineSdfAcc.getValue(c.offsetBy(0, 1, 0)) > valleyMargin) ||
                         (fineSdfAcc.getValue(c.offsetBy(0, 0, -1)) > valleyMargin &&
                          fineSdfAcc.getValue(c.offsetBy(0, 0, 1)) > valleyMargin));
                    if (isCellValley) {
                        maxValleyCorr = std::max(maxValleyCorr, 0.65f * (baseSdf - cSdf));
                    }

                    if (maxValleyCorr > 0.0f) {
                        baseSdf -= std::min(maxValleyCorr, 0.70f * fineVoxelSize);
                    }
                }
                return baseSdf;
            };

            auto lookupVoxelBranch = [&](float x, float y, float z, int targetBranch) -> int {
                const openvdb::Coord c =
                    branchGrid.transform().worldToIndexCellCentered(openvdb::Vec3d(x, y, z));
                const int b0 = bAcc.getValue(c);
                if (b0 >= 0 && isSubtree[static_cast<std::size_t>(targetBranch)][static_cast<std::size_t>(b0)]) {
                    return b0;
                }
                int firstValid = b0;
                for (int dz = -1; dz <= 1; ++dz) {
                    for (int dy = -1; dy <= 1; ++dy) {
                        for (int dx = -1; dx <= 1; ++dx) {
                            const int nb = bAcc.getValue(c + openvdb::Coord(dx, dy, dz));
                            if (nb >= 0) {
                                if (isSubtree[static_cast<std::size_t>(targetBranch)][static_cast<std::size_t>(nb)]) {
                                    return nb;
                                }
                                if (firstValid < 0) {
                                    firstValid = nb;
                                }
                            }
                        }
                    }
                }
                if (firstValid >= 0) {
                    return firstValid;
                }
                for (int dz = -2; dz <= 2; ++dz) {
                    for (int dy = -2; dy <= 2; ++dy) {
                        for (int dx = -2; dx <= 2; ++dx) {
                            const int nb = bAcc.getValue(c + openvdb::Coord(dx, dy, dz));
                            if (nb >= 0) {
                                return nb;
                            }
                        }
                    }
                }
                return -1;
            };

            auto nearestSubtreeDistSq = [&](const openvdb::Vec3f& pt, int targetBranch) -> float {
                float bestDistSq = std::numeric_limits<float>::max();
                for (std::size_t u = 0; u < numBranches; ++u) {
                    if (!isSubtree[static_cast<std::size_t>(targetBranch)][u]) {
                        continue;
                    }
                    for (const auto& sp : branchSamplePoints[u]) {
                        const float dSq = (pt - sp).lengthSqr();
                        if (dSq < bestDistSq) {
                            bestDistSq = dSq;
                        }
                    }
                }
                return bestDistSq;
            };

            auto islandMatchesBranch = [&](int subtreeVotes,
                                           int otherBranchVotes,
                                           const openvdb::Vec3f& islandCentroid,
                                           int targetBranch) -> bool {
                if (!multiBranch) {
                    return true;
                }
                if (subtreeVotes > 0 || otherBranchVotes > 0) {
                    return subtreeVotes >= otherBranchVotes;
                }
                // All sampled voxels on this island belonged to strict ancestors of targetBranch
                // (e.g. a small island right above a saddle split): assign to the closest non-ancestor branch.
                int bestBranch = -1;
                float bestDistSq = std::numeric_limits<float>::max();
                for (std::size_t u = 0; u < numBranches; ++u) {
                    const bool isStrictAncestor =
                        (u != static_cast<std::size_t>(targetBranch)) &&
                        (isSubtree[u][static_cast<std::size_t>(targetBranch)] != 0);
                    if (isStrictAncestor) {
                        continue;
                    }
                    for (const auto& sp : branchSamplePoints[u]) {
                        const float dSq = (islandCentroid - sp).lengthSqr();
                        if (dSq < bestDistSq) {
                            bestDistSq = dSq;
                            bestBranch = static_cast<int>(u);
                        }
                    }
                }
                return bestBranch >= 0 &&
                    isSubtree[static_cast<std::size_t>(targetBranch)][static_cast<std::size_t>(bestBranch)] != 0;
            };

            auto lerpVertex = [](const RenderVertex& a, const RenderVertex& b, float t) -> RenderVertex {
                RenderVertex rv{};
                rv.x = a.x + t * (b.x - a.x);
                rv.y = a.y + t * (b.y - a.y);
                rv.z = a.z + t * (b.z - a.z);
                rv.nx = a.nx + t * (b.nx - a.nx);
                rv.ny = a.ny + t * (b.ny - a.ny);
                rv.nz = a.nz + t * (b.nz - a.nz);
                const float len = std::sqrt(rv.nx * rv.nx + rv.ny * rv.ny + rv.nz * rv.nz);
                if (len > 1.0e-7f) {
                    rv.nx /= len;
                    rv.ny /= len;
                    rv.nz /= len;
                } else {
                    rv.nz = 1.0f;
                }
                return rv;
            };

            for (std::size_t idx = r.begin(); idx != r.end(); ++idx) {
                const PendingLayer& pl = pending[idx];
                SelectedLayerInfo info{};
                info.layer_index = static_cast<std::uint32_t>(idx);
                info.branch_id = pl.branchId;
                info.parent_branch_id = pl.parentBranchId;
                info.branch_layer_index = pl.branchLayerIndex;
                info.branch_layer_count = pl.branchLayerCount;
                info.isovalue = pl.isovalue;
                info.mean_thickness_mm = pl.meanThicknessMm;
                info.min_thickness_mm = pl.minThicknessMm;
                info.max_thickness_mm = pl.maxThicknessMm;
                info.is_first_layer = pl.isFirstLayer;
                info.is_top_layer = pl.isTopLayer;

                std::vector<openvdb::Vec3s> points;
                std::vector<openvdb::Vec3I> triangles;
                std::vector<openvdb::Vec4I> quads;

                openvdb::tools::volumeToMesh(
                    eikGrid,
                    points,
                    triangles,
                    quads,
                    static_cast<double>(pl.isovalue),
                    0.0,
                    true);

                if (!points.empty() && (!triangles.empty() || !quads.empty())) {
                    std::vector<RenderVertex> verts(points.size());
                    for (std::size_t i = 0; i < points.size(); ++i) {
                        verts[i].x = points[i].x();
                        verts[i].y = points[i].y();
                        verts[i].z = points[i].z();
                        verts[i].nx = 0.0f;
                        verts[i].ny = 0.0f;
                        verts[i].nz = 0.0f;
                    }

                    std::vector<std::uint32_t> inds;
                    inds.reserve(triangles.size() * 3 + quads.size() * 6);
                    for (const auto& tri : triangles) {
                        inds.push_back(tri[0]);
                        inds.push_back(tri[2]);
                        inds.push_back(tri[1]);
                    }
                    for (const auto& q : quads) {
                        inds.push_back(q[0]);
                        inds.push_back(q[2]);
                        inds.push_back(q[1]);
                        inds.push_back(q[0]);
                        inds.push_back(q[3]);
                        inds.push_back(q[2]);
                    }

                    if (useDualGridExtraction) {
                        // Conformal edge-midpoint subdivision inside the outer boundary shell band
                        // followed by smooth Newton projection onto the fine Eikonal isosurface
                        // \tau_fine(x) = pl.isovalue. Because edge splitting is keyed per undirected
                        // edge {min(u,v), max(u,v)} and all triangles apply the exact 0/1/2/3-split-edge
                        // templates, zero T-junctions can ever be created (`hole_edges == 0`).
                        const float refineMinSdf =
                            -std::max(shellThicknessMm + 1.5f * voxelSize, 4.0f * fineVoxelSize);
                        const float refineMaxSdf =
                            std::min(1.5f * voxelSize, outerFringeLimit - 0.5f * voxelSize);
                        const float maxShellEdgeSq =
                            (1.35f * fineVoxelSize) * (1.35f * fineVoxelSize);
                        const int numSubdivPasses = std::clamp<int>(
                            static_cast<int>(
                                std::ceil(std::log2(std::max<std::uint32_t>(2, rec.coarseToFineRatio)))),
                            1,
                            3);

                        for (int pass = 0; pass < numSubdivPasses; ++pass) {
                            std::vector<float> passSdf(verts.size(), 0.0f);
                            for (std::size_t i = 0; i < verts.size(); ++i) {
                                passSdf[i] = sampleCompositeSdf(
                                    openvdb::Vec3d(verts[i].x, verts[i].y, verts[i].z));
                            }

                            std::unordered_map<std::uint64_t, std::uint32_t> edgeMidpointMap;
                            edgeMidpointMap.reserve(inds.size());

                            auto considerEdgeForSplit = [&](std::uint32_t u, std::uint32_t v) {
                                const std::uint32_t lo = std::min(u, v);
                                const std::uint32_t hi = std::max(u, v);
                                const std::uint64_t key =
                                    (static_cast<std::uint64_t>(lo) << 32) |
                                    static_cast<std::uint64_t>(hi);
                                if (edgeMidpointMap.find(key) != edgeMidpointMap.end()) {
                                    return;
                                }
                                const float d0 = passSdf[lo];
                                const float d1 = passSdf[hi];
                                if (std::max(d0, d1) < refineMinSdf ||
                                    std::min(d0, d1) > refineMaxSdf) {
                                    return;
                                }
                                const float dx = verts[hi].x - verts[lo].x;
                                const float dy = verts[hi].y - verts[lo].y;
                                const float dz = verts[hi].z - verts[lo].z;
                                if (dx * dx + dy * dy + dz * dz <= maxShellEdgeSq) {
                                    return;
                                }
                                RenderVertex mid{};
                                mid.x = 0.5f * (verts[lo].x + verts[hi].x);
                                mid.y = 0.5f * (verts[lo].y + verts[hi].y);
                                mid.z = 0.5f * (verts[lo].z + verts[hi].z);
                                mid.nx = 0.0f;
                                mid.ny = 0.0f;
                                mid.nz = 1.0f;
                                const auto midIdx = static_cast<std::uint32_t>(verts.size());
                                verts.push_back(mid);
                                edgeMidpointMap.emplace(key, midIdx);
                            };

                            for (std::size_t t = 0; t + 2 < inds.size(); t += 3) {
                                const std::uint32_t i0 = inds[t + 0];
                                const std::uint32_t i1 = inds[t + 1];
                                const std::uint32_t i2 = inds[t + 2];
                                considerEdgeForSplit(i0, i1);
                                considerEdgeForSplit(i1, i2);
                                considerEdgeForSplit(i2, i0);
                            }

                            if (edgeMidpointMap.empty()) {
                                break;
                            }

                            auto lookupMidpoint = [&](std::uint32_t u, std::uint32_t v) -> std::int32_t {
                                const std::uint32_t lo = std::min(u, v);
                                const std::uint32_t hi = std::max(u, v);
                                const std::uint64_t key =
                                    (static_cast<std::uint64_t>(lo) << 32) |
                                    static_cast<std::uint64_t>(hi);
                                const auto it = edgeMidpointMap.find(key);
                                return (it != edgeMidpointMap.end())
                                    ? static_cast<std::int32_t>(it->second)
                                    : -1;
                            };

                            std::vector<std::uint32_t> refinedInds;
                            refinedInds.reserve(inds.size() * 2);

                            auto pushRefinedTri = [&](std::uint32_t a, std::uint32_t b, std::uint32_t c) {
                                refinedInds.push_back(a);
                                refinedInds.push_back(b);
                                refinedInds.push_back(c);
                            };

                            for (std::size_t t = 0; t + 2 < inds.size(); t += 3) {
                                const std::uint32_t i0 = inds[t + 0];
                                const std::uint32_t i1 = inds[t + 1];
                                const std::uint32_t i2 = inds[t + 2];

                                const std::int32_t m01 = lookupMidpoint(i0, i1);
                                const std::int32_t m12 = lookupMidpoint(i1, i2);
                                const std::int32_t m20 = lookupMidpoint(i2, i0);
                                const int splitCount =
                                    (m01 >= 0 ? 1 : 0) + (m12 >= 0 ? 1 : 0) + (m20 >= 0 ? 1 : 0);

                                if (splitCount == 0) {
                                    pushRefinedTri(i0, i1, i2);
                                } else if (splitCount == 3) {
                                    const auto u01 = static_cast<std::uint32_t>(m01);
                                    const auto u12 = static_cast<std::uint32_t>(m12);
                                    const auto u20 = static_cast<std::uint32_t>(m20);
                                    pushRefinedTri(i0, u01, u20);
                                    pushRefinedTri(u01, i1, u12);
                                    pushRefinedTri(u20, u12, i2);
                                    pushRefinedTri(u01, u12, u20);
                                } else if (splitCount == 1) {
                                    // Rotate (A, B, C) so that edge (A, B) is the single split edge
                                    std::uint32_t A = i0, B = i1, C = i2;
                                    std::uint32_t mAB = static_cast<std::uint32_t>(m01);
                                    if (m12 >= 0) {
                                        A = i1; B = i2; C = i0;
                                        mAB = static_cast<std::uint32_t>(m12);
                                    } else if (m20 >= 0) {
                                        A = i2; B = i0; C = i1;
                                        mAB = static_cast<std::uint32_t>(m20);
                                    }
                                    pushRefinedTri(A, mAB, C);
                                    pushRefinedTri(mAB, B, C);
                                } else {
                                    // splitCount == 2: rotate (A, B, C) so (C, A) is the unsplit edge
                                    // and edges (A, B) and (B, C) are split at mAB and mBC.
                                    std::uint32_t A = i0, B = i1, C = i2;
                                    std::uint32_t mAB = 0, mBC = 0;
                                    if (m20 < 0) {
                                        A = i0; B = i1; C = i2;
                                        mAB = static_cast<std::uint32_t>(m01);
                                        mBC = static_cast<std::uint32_t>(m12);
                                    } else if (m01 < 0) {
                                        A = i1; B = i2; C = i0;
                                        mAB = static_cast<std::uint32_t>(m12);
                                        mBC = static_cast<std::uint32_t>(m20);
                                    } else {
                                        A = i2; B = i0; C = i1;
                                        mAB = static_cast<std::uint32_t>(m20);
                                        mBC = static_cast<std::uint32_t>(m01);
                                    }
                                    pushRefinedTri(A, mAB, C);
                                    pushRefinedTri(mAB, mBC, C);
                                    pushRefinedTri(mAB, B, mBC);
                                }
                            }

                            inds.swap(refinedInds);
                        }

                        // Project shell-band vertices along \nabla \tau_fine onto the fine Eikonal
                        // isosurface \tau_fine(x) = pl.isovalue, tapering smoothly to 0 at `fineInnerCutoff`.
                        const float fullProjDepth = -(shellThicknessMm + 0.5f * fineVoxelSize);
                        const float taperSpan =
                            std::max(fullProjDepth - fineInnerCutoff, fineVoxelSize);
                        const float hStep = 0.5f * fineVoxelSize;
                        const float maxStepMm = 0.55f * fineVoxelSize;
                        const float fineEikValidLimit =
                            fineEikGrid.background() - std::max(0.5f * fineVoxelSize, 0.5f);

                        for (std::size_t i = 0; i < verts.size(); ++i) {
                            const openvdb::Vec3d wp0(verts[i].x, verts[i].y, verts[i].z);
                            const float fSdf = fineSdfSampler.wsSample(wp0);
                            if (fSdf <= fineInnerCutoff || fSdf >= 2.0f * fineVoxelSize) {
                                continue;
                            }
                            float wProj = 1.0f;
                            if (fSdf < fullProjDepth) {
                                const float u =
                                    std::clamp((fSdf - fineInnerCutoff) / taperSpan, 0.0f, 1.0f);
                                wProj = u * u * (3.0f - 2.0f * u);
                            }
                            if (wProj <= 1.0e-3f) {
                                continue;
                            }

                            for (int step = 0; step < 2; ++step) {
                                const openvdb::Vec3d wp(verts[i].x, verts[i].y, verts[i].z);
                                const float tVal = fineEikSampler.wsSample(wp);
                                if (!std::isfinite(tVal) || tVal >= fineEikValidLimit) {
                                    break;
                                }
                                const float err = tVal - pl.isovalue;
                                if (std::abs(err) <= 1.0e-4f) {
                                    break;
                                }
                                const float gx =
                                    (fineEikSampler.wsSample(wp + openvdb::Vec3d(hStep, 0.0, 0.0)) -
                                     fineEikSampler.wsSample(wp - openvdb::Vec3d(hStep, 0.0, 0.0))) /
                                    (2.0f * hStep);
                                const float gy =
                                    (fineEikSampler.wsSample(wp + openvdb::Vec3d(0.0, hStep, 0.0)) -
                                     fineEikSampler.wsSample(wp - openvdb::Vec3d(0.0, hStep, 0.0))) /
                                    (2.0f * hStep);
                                const float gz =
                                    (fineEikSampler.wsSample(wp + openvdb::Vec3d(0.0, 0.0, hStep)) -
                                     fineEikSampler.wsSample(wp - openvdb::Vec3d(0.0, 0.0, hStep))) /
                                    (2.0f * hStep);
                                const float gLenSq = gx * gx + gy * gy + gz * gz;
                                if (gLenSq < 0.04f || gLenSq > 16.0f) {
                                    break;
                                }
                                const float gLen = std::sqrt(gLenSq);
                                const float disp =
                                    std::clamp(err / gLen, -maxStepMm, maxStepMm) * wProj;
                                verts[i].x -= disp * (gx / gLen);
                                verts[i].y -= disp * (gy / gLen);
                                verts[i].z -= disp * (gz / gLen);
                            }
                        }
                    }

                    computeMeshNormals(verts, inds);

                    std::vector<float> vertSdf(verts.size(), 0.0f);
                    for (std::size_t i = 0; i < verts.size(); ++i) {
                        vertSdf[i] = sampleCompositeSdf(
                            openvdb::Vec3d(verts[i].x, verts[i].y, verts[i].z));
                    }

                    // Conformal Thin-Wall Valley Edge Split:
                    // If an unclipped isosurface triangle edge straddles a thin curved wall such
                    // that both endpoints lie outside (\phi > 0) while an interior sample along the
                    // edge dips inside the wall (\phi <= 0), split the undirected edge at that
                    // interior valley minimum using the conformal 0/1/2/3-edge templates so
                    // `clipAgainstSdfThreshold` captures both sides of the thin wall without holes.
                    {
                        std::unordered_map<std::uint64_t, std::uint32_t> valleySplitMap;
                        const float maxEndpointSdf = 1.75f * fineVoxelSize;

                        auto considerValleyEdge = [&](std::uint32_t u, std::uint32_t v) {
                            const std::uint32_t lo = std::min(u, v);
                            const std::uint32_t hi = std::max(u, v);
                            const float dLo = vertSdf[lo];
                            const float dHi = vertSdf[hi];
                            if (dLo <= 0.0f || dHi <= 0.0f ||
                                std::min(dLo, dHi) > maxEndpointSdf) {
                                return;
                            }
                            const std::uint64_t key =
                                (static_cast<std::uint64_t>(lo) << 32) |
                                static_cast<std::uint64_t>(hi);
                            if (valleySplitMap.find(key) != valleySplitMap.end()) {
                                return;
                            }

                            float bestT = 0.5f;
                            float bestSdf = std::min(dLo, dHi);
                            for (const float tCand : {0.25f, 0.50f, 0.75f}) {
                                const openvdb::Vec3d wpCand(
                                    verts[lo].x + tCand * (verts[hi].x - verts[lo].x),
                                    verts[lo].y + tCand * (verts[hi].y - verts[lo].y),
                                    verts[lo].z + tCand * (verts[hi].z - verts[lo].z));
                                const float sCand = sampleCompositeSdf(wpCand);
                                if (sCand < bestSdf) {
                                    bestSdf = sCand;
                                    bestT = tCand;
                                }
                            }
                            if (bestSdf <= 0.0f) {
                                const RenderVertex mv = lerpVertex(verts[lo], verts[hi], bestT);
                                const auto midIdx = static_cast<std::uint32_t>(verts.size());
                                verts.push_back(mv);
                                vertSdf.push_back(bestSdf);
                                valleySplitMap.emplace(key, midIdx);
                            }
                        };

                        for (std::size_t t = 0; t + 2 < inds.size(); t += 3) {
                            const std::uint32_t i0 = inds[t + 0];
                            const std::uint32_t i1 = inds[t + 1];
                            const std::uint32_t i2 = inds[t + 2];
                            considerValleyEdge(i0, i1);
                            considerValleyEdge(i1, i2);
                            considerValleyEdge(i2, i0);
                        }

                        if (!valleySplitMap.empty()) {
                            auto lookupValleySplit =
                                [&](std::uint32_t u, std::uint32_t v) -> std::int32_t {
                                const std::uint32_t lo = std::min(u, v);
                                const std::uint32_t hi = std::max(u, v);
                                const std::uint64_t key =
                                    (static_cast<std::uint64_t>(lo) << 32) |
                                    static_cast<std::uint64_t>(hi);
                                const auto it = valleySplitMap.find(key);
                                return (it != valleySplitMap.end())
                                    ? static_cast<std::int32_t>(it->second)
                                    : -1;
                            };

                            std::vector<std::uint32_t> splitInds;
                            splitInds.reserve(inds.size() + valleySplitMap.size() * 6);
                            auto pushSplitTri =
                                [&](std::uint32_t a, std::uint32_t b, std::uint32_t c) {
                                splitInds.push_back(a);
                                splitInds.push_back(b);
                                splitInds.push_back(c);
                            };

                            for (std::size_t t = 0; t + 2 < inds.size(); t += 3) {
                                const std::uint32_t i0 = inds[t + 0];
                                const std::uint32_t i1 = inds[t + 1];
                                const std::uint32_t i2 = inds[t + 2];

                                const std::int32_t m01 = lookupValleySplit(i0, i1);
                                const std::int32_t m12 = lookupValleySplit(i1, i2);
                                const std::int32_t m20 = lookupValleySplit(i2, i0);
                                const int splitCount =
                                    (m01 >= 0 ? 1 : 0) + (m12 >= 0 ? 1 : 0) + (m20 >= 0 ? 1 : 0);

                                if (splitCount == 0) {
                                    pushSplitTri(i0, i1, i2);
                                } else if (splitCount == 3) {
                                    const auto u01 = static_cast<std::uint32_t>(m01);
                                    const auto u12 = static_cast<std::uint32_t>(m12);
                                    const auto u20 = static_cast<std::uint32_t>(m20);
                                    pushSplitTri(i0, u01, u20);
                                    pushSplitTri(u01, i1, u12);
                                    pushSplitTri(u20, u12, i2);
                                    pushSplitTri(u01, u12, u20);
                                } else if (splitCount == 1) {
                                    std::uint32_t A = i0, B = i1, C = i2;
                                    std::uint32_t mAB = static_cast<std::uint32_t>(m01);
                                    if (m12 >= 0) {
                                        A = i1; B = i2; C = i0;
                                        mAB = static_cast<std::uint32_t>(m12);
                                    } else if (m20 >= 0) {
                                        A = i2; B = i0; C = i1;
                                        mAB = static_cast<std::uint32_t>(m20);
                                    }
                                    pushSplitTri(A, mAB, C);
                                    pushSplitTri(mAB, B, C);
                                } else {
                                    std::uint32_t A = i0, B = i1, C = i2;
                                    std::uint32_t mAB = 0, mBC = 0;
                                    if (m20 < 0) {
                                        A = i0; B = i1; C = i2;
                                        mAB = static_cast<std::uint32_t>(m01);
                                        mBC = static_cast<std::uint32_t>(m12);
                                    } else if (m01 < 0) {
                                        A = i1; B = i2; C = i0;
                                        mAB = static_cast<std::uint32_t>(m12);
                                        mBC = static_cast<std::uint32_t>(m20);
                                    } else {
                                        A = i2; B = i0; C = i1;
                                        mAB = static_cast<std::uint32_t>(m20);
                                        mBC = static_cast<std::uint32_t>(m01);
                                    }
                                    pushSplitTri(A, mAB, C);
                                    pushSplitTri(mAB, mBC, C);
                                    pushSplitTri(mAB, B, mBC);
                                }
                            }
                            inds.swap(splitInds);
                        }
                    }

                    auto clipAgainstSdfThreshold = [&](float sdfLimit) {
                        info.vertices.clear();
                        info.indices.clear();
                        info.contour_segments.clear();

                        std::vector<RenderVertex> rawVertices;
                        std::vector<std::uint32_t> rawIndices;
                        struct ContourEdge {
                            std::uint32_t v0;
                            std::uint32_t v1;
                        };
                        std::vector<ContourEdge> rawContours;

                        std::vector<std::int32_t> vertexRemap(verts.size(), -1);
                        std::unordered_map<std::uint64_t, std::uint32_t> edgeCutRemap;

                        auto emitVertex = [&](std::uint32_t oldIdx) -> std::uint32_t {
                            if (vertexRemap[oldIdx] >= 0) {
                                return static_cast<std::uint32_t>(vertexRemap[oldIdx]);
                            }
                            const auto newIdx = static_cast<std::uint32_t>(rawVertices.size());
                            rawVertices.push_back(verts[oldIdx]);
                            vertexRemap[oldIdx] = static_cast<std::int32_t>(newIdx);
                            return newIdx;
                        };
                        auto emitCutEdgeVertex = [&](std::uint32_t u, std::uint32_t v, float dU, float dV) -> std::uint32_t {
                            const std::uint32_t lo = std::min(u, v);
                            const std::uint32_t hi = std::max(u, v);
                            const std::uint64_t key =
                                (static_cast<std::uint64_t>(lo) << 32) | static_cast<std::uint64_t>(hi);
                            const auto it = edgeCutRemap.find(key);
                            if (it != edgeCutRemap.end()) {
                                return it->second;
                            }
                            const float dLo = (u == lo) ? dU : dV;
                            const float dHi = (u == lo) ? dV : dU;
                            const float denom = dHi - dLo;
                            const float t = (std::abs(denom) > 1.0e-8f)
                                ? std::clamp(-dLo / denom, 0.0f, 1.0f)
                                : 0.5f;
                            const RenderVertex cv = lerpVertex(verts[lo], verts[hi], t);
                            const auto newIdx = static_cast<std::uint32_t>(rawVertices.size());
                            rawVertices.push_back(cv);
                            edgeCutRemap.emplace(key, newIdx);
                            return newIdx;
                        };

                        for (std::size_t t = 0; t + 2 < inds.size(); t += 3) {
                            const std::uint32_t i0 = inds[t + 0];
                            const std::uint32_t i1 = inds[t + 1];
                            const std::uint32_t i2 = inds[t + 2];

                            const float d0 = vertSdf[i0] - sdfLimit;
                            const float d1 = vertSdf[i1] - sdfLimit;
                            const float d2 = vertSdf[i2] - sdfLimit;

                            if (std::max({vertSdf[i0], vertSdf[i1], vertSdf[i2]}) >= outerFringeLimit &&
                                std::min({vertSdf[i0], vertSdf[i1], vertSdf[i2]}) > 0.25f * voxelSize) {
                                continue;
                            }

                            const bool in0 = (d0 <= 0.0f);
                            const bool in1 = (d1 <= 0.0f);
                            const bool in2 = (d2 <= 0.0f);
                            if (!in0 && !in1 && !in2) {
                                continue;
                            }

                            if (in0 && in1 && in2) {
                                const std::uint32_t n0 = emitVertex(i0);
                                const std::uint32_t n1 = emitVertex(i1);
                                const std::uint32_t n2 = emitVertex(i2);
                                rawIndices.push_back(n0);
                                rawIndices.push_back(n1);
                                rawIndices.push_back(n2);
                                continue;
                            }

                            std::uint32_t idxA = i0, idxB = i1, idxC = i2;
                            float dA = d0, dB = d1, dC = d2;
                            bool soloInside = in0;

                            if (in0 == in1) {
                                idxA = i2; idxB = i0; idxC = i1;
                                dA = d2;   dB = d0;   dC = d1;
                                soloInside = in2;
                            } else if (in0 == in2) {
                                idxA = i1; idxB = i2; idxC = i0;
                                dA = d1;   dB = d2;   dC = d0;
                                soloInside = in1;
                            }

                            const std::uint32_t vAB_in = emitCutEdgeVertex(idxA, idxB, dA, dB);
                            const std::uint32_t vAC_in = emitCutEdgeVertex(idxA, idxC, dA, dC);
                            rawContours.push_back(ContourEdge{vAB_in, vAC_in});

                            if (soloInside) {
                                const std::uint32_t nA = emitVertex(idxA);
                                rawIndices.push_back(nA);
                                rawIndices.push_back(vAB_in);
                                rawIndices.push_back(vAC_in);
                            } else {
                                const std::uint32_t nB = emitVertex(idxB);
                                const std::uint32_t nC = emitVertex(idxC);
                                rawIndices.push_back(vAB_in);
                                rawIndices.push_back(nB);
                                rawIndices.push_back(nC);
                                rawIndices.push_back(vAB_in);
                                rawIndices.push_back(nC);
                                rawIndices.push_back(vAC_in);
                            }
                        }

                        if (!multiBranch || rawIndices.empty()) {
                            info.vertices.reserve(rawVertices.size());
                            for (const auto& rv : rawVertices) {
                                info.vertices.push_back(rv);
                            }
                            info.indices.reserve(rawIndices.size());
                            for (const auto idxVal : rawIndices) {
                                info.indices.push_back(idxVal);
                            }
                            info.contour_segments.reserve(rawContours.size() * 6);
                            for (const auto& ce : rawContours) {
                                const auto& p0 = rawVertices[ce.v0];
                                const auto& p1 = rawVertices[ce.v1];
                                info.contour_segments.push_back(p0.x);
                                info.contour_segments.push_back(p0.y);
                                info.contour_segments.push_back(p0.z);
                                info.contour_segments.push_back(p1.x);
                                info.contour_segments.push_back(p1.y);
                                info.contour_segments.push_back(p1.z);
                            }
                            return;
                        }

                        // Group clipped triangles into connected 3D surface islands via shared vertices
                        // and classify each island as a whole unit so no interior holes can ever be created.
                        const std::size_t numClippedVerts = rawVertices.size();
                        std::vector<std::uint32_t> vParent(numClippedVerts);
                        std::iota(vParent.begin(), vParent.end(), 0);
                        auto findV = [&](std::uint32_t x) -> std::uint32_t {
                            std::uint32_t root = x;
                            while (vParent[root] != root) {
                                root = vParent[root];
                            }
                            std::uint32_t curr = x;
                            while (curr != root) {
                                const std::uint32_t nxt = vParent[curr];
                                vParent[curr] = root;
                                curr = nxt;
                            }
                            return root;
                        };
                        auto unionV = [&](std::uint32_t a, std::uint32_t b) {
                            const std::uint32_t ra = findV(a);
                            const std::uint32_t rb = findV(b);
                            if (ra != rb) {
                                vParent[rb] = ra;
                            }
                        };

                        for (std::size_t t = 0; t + 2 < rawIndices.size(); t += 3) {
                            unionV(rawIndices[t + 0], rawIndices[t + 1]);
                            unionV(rawIndices[t + 1], rawIndices[t + 2]);
                        }

                        struct IslandStats {
                            int subtreeVotes = 0;
                            int otherBranchVotes = 0;
                            int triCount = 0;
                            openvdb::Vec3d sumCentroids{0.0, 0.0, 0.0};
                            bool keep = false;
                        };
                        std::unordered_map<std::uint32_t, IslandStats> islands;
                        const int targetBranch = static_cast<int>(pl.branchId);

                        for (std::size_t t = 0; t + 2 < rawIndices.size(); t += 3) {
                            const std::uint32_t i0 = rawIndices[t + 0];
                            const std::uint32_t i1 = rawIndices[t + 1];
                            const std::uint32_t i2 = rawIndices[t + 2];
                            const std::uint32_t root = findV(i0);
                            auto& st = islands[root];
                            const float cx = (rawVertices[i0].x + rawVertices[i1].x + rawVertices[i2].x) / 3.0f;
                            const float cy = (rawVertices[i0].y + rawVertices[i1].y + rawVertices[i2].y) / 3.0f;
                            const float cz = (rawVertices[i0].z + rawVertices[i1].z + rawVertices[i2].z) / 3.0f;
                            st.sumCentroids += openvdb::Vec3d(cx, cy, cz);
                            st.triCount += 1;

                            const int vBranch = lookupVoxelBranch(cx, cy, cz, targetBranch);
                            if (vBranch >= 0) {
                                const bool inSub =
                                    isSubtree[static_cast<std::size_t>(targetBranch)][static_cast<std::size_t>(vBranch)] != 0;
                                const bool inAnc =
                                    isSubtree[static_cast<std::size_t>(vBranch)][static_cast<std::size_t>(targetBranch)] != 0;
                                if (inSub) {
                                    st.subtreeVotes += 1;
                                } else if (!inAnc) {
                                    st.otherBranchVotes += 1;
                                }
                            }
                        }

                        bool anyIslandKept = false;
                        std::uint32_t closestIslandRoot = 0;
                        float closestIslandDistSq = std::numeric_limits<float>::max();

                        for (auto& [root, st] : islands) {
                            const openvdb::Vec3d cAvg =
                                st.sumCentroids / static_cast<double>(std::max(1, st.triCount));
                            const openvdb::Vec3f centroid(
                                static_cast<float>(cAvg.x()),
                                static_cast<float>(cAvg.y()),
                                static_cast<float>(cAvg.z()));
                            st.keep = islandMatchesBranch(
                                st.subtreeVotes,
                                st.otherBranchVotes,
                                centroid,
                                targetBranch);
                            if (st.keep) {
                                anyIslandKept = true;
                            }
                            const float dSq = nearestSubtreeDistSq(centroid, targetBranch);
                            if (dSq < closestIslandDistSq) {
                                closestIslandDistSq = dSq;
                                closestIslandRoot = root;
                            }
                        }

                        if (!anyIslandKept && !islands.empty()) {
                            islands[closestIslandRoot].keep = true;
                        }

                        std::vector<std::int32_t> compactRemap(numClippedVerts, -1);
                        auto emitKeptVert = [&](std::uint32_t rawIdx) -> std::uint32_t {
                            if (compactRemap[rawIdx] >= 0) {
                                return static_cast<std::uint32_t>(compactRemap[rawIdx]);
                            }
                            const auto newIdx = static_cast<std::uint32_t>(info.vertices.size());
                            info.vertices.push_back(rawVertices[rawIdx]);
                            compactRemap[rawIdx] = static_cast<std::int32_t>(newIdx);
                            return newIdx;
                        };

                        for (std::size_t t = 0; t + 2 < rawIndices.size(); t += 3) {
                            const std::uint32_t root = findV(rawIndices[t + 0]);
                            if (!islands[root].keep) {
                                continue;
                            }
                            const std::uint32_t n0 = emitKeptVert(rawIndices[t + 0]);
                            const std::uint32_t n1 = emitKeptVert(rawIndices[t + 1]);
                            const std::uint32_t n2 = emitKeptVert(rawIndices[t + 2]);
                            info.indices.push_back(n0);
                            info.indices.push_back(n1);
                            info.indices.push_back(n2);
                        }

                        for (const auto& ce : rawContours) {
                            const std::uint32_t root = findV(ce.v0);
                            auto it = islands.find(root);
                            if (it == islands.end() || !it->second.keep) {
                                continue;
                            }
                            const auto& p0 = rawVertices[ce.v0];
                            const auto& p1 = rawVertices[ce.v1];
                            info.contour_segments.push_back(p0.x);
                            info.contour_segments.push_back(p0.y);
                            info.contour_segments.push_back(p0.z);
                            info.contour_segments.push_back(p1.x);
                            info.contour_segments.push_back(p1.y);
                            info.contour_segments.push_back(p1.z);
                        }
                    };

                    clipAgainstSdfThreshold(pl.sdfClipAllowance);

                    // If trilinear SDF rounding at a sharp 3D corner still left this layer empty,
                    // find the minimum SDF on the branch's isosurface sheet and clip with a small margin.
                    if (info.indices.empty()) {
                        float minBranchSdf = std::numeric_limits<float>::max();
                        const int targetBranch = static_cast<int>(pl.branchId);
                        for (std::size_t t = 0; t + 2 < inds.size(); t += 3) {
                            const std::uint32_t i0 = inds[t + 0];
                            const std::uint32_t i1 = inds[t + 1];
                            const std::uint32_t i2 = inds[t + 2];
                            if (std::max({vertSdf[i0], vertSdf[i1], vertSdf[i2]}) >= outerFringeLimit) {
                                continue;
                            }
                            const float cx = (verts[i0].x + verts[i1].x + verts[i2].x) / 3.0f;
                            const float cy = (verts[i0].y + verts[i1].y + verts[i2].y) / 3.0f;
                            const float cz = (verts[i0].z + verts[i1].z + verts[i2].z) / 3.0f;
                            if (!multiBranch) {
                                minBranchSdf = std::min({minBranchSdf, vertSdf[i0], vertSdf[i1], vertSdf[i2]});
                            } else {
                                const int vBranch = lookupVoxelBranch(cx, cy, cz, targetBranch);
                                if (vBranch >= 0 &&
                                    isSubtree[static_cast<std::size_t>(targetBranch)][static_cast<std::size_t>(vBranch)]) {
                                    minBranchSdf = std::min({minBranchSdf, vertSdf[i0], vertSdf[i1], vertSdf[i2]});
                                }
                            }
                        }
                        if (minBranchSdf <= 0.85f * voxelSize) {
                            clipAgainstSdfThreshold(minBranchSdf + std::min(0.20f * voxelSize, 0.25f * hTop));
                        }
                    }
                }

                extractedLayers[idx] = std::move(info);
            }
        });

    result.valid = true;
    result.branch_count = static_cast<std::uint32_t>(branches.size());
    result.layers.reserve(extractedLayers.size());
    for (auto& layer : extractedLayers) {
        result.layers.push_back(std::move(layer));
    }

    rec.cachedSelectedLayers = result;
    rec.cachedSelectedLayersAdaptivity = adaptivity;
    rec.cachedSelectedLayersSpec = lh;
    rec.hasCachedSelectedLayers = true;
    setFieldProgress(false, 1.0f, "Selected slicing isosurfaces ready.");
    return result;
}

void SlicerBackend::setShellPathSpec(const ShellPathSpec& spec) noexcept {
    ShellPathSpec s = spec;
    s.shell_thickness_mm = std::max(0.05f, s.shell_thickness_mm);
    auto sanitizeW = [](LineWidthSpec& w) {
        w.min_mm = std::max(0.05f, w.min_mm);
        if (w.max_mm < w.min_mm) {
            w.max_mm = w.min_mm;
        }
        w.target_mm = std::clamp(w.target_mm, w.min_mm, w.max_mm);
    };
    sanitizeW(s.outer_wall_width);
    sanitizeW(s.inner_wall_width);
    sanitizeW(s.solid_infill_width);
    sanitizeW(s.sparse_infill_width);
    s.sparse_infill_density = std::clamp(s.sparse_infill_density, 0.0f, 1.0f);
    s.infill_angle_deg = std::clamp(s.infill_angle_deg, 0.0f, 180.0f);
    s.nozzle_diameter_mm = (s.nozzle_diameter_mm > 0.05f) ? s.nozzle_diameter_mm : 0.40f;
    s.min_extrusion_length_mm = std::max(0.0f, s.min_extrusion_length_mm);
    impl_->shellPathSpec = s;
}

ShellPathSpec SlicerBackend::shellPathSpec() const noexcept {
    return impl_->shellPathSpec;
}

ObjectToolpathsResult SlicerBackend::generateObjectShellToolpaths(
    std::uint64_t objectId,
    float adaptivity) const
{
    ObjectToolpathsResult empty{};
    empty.valid = false;

    const auto objIt = impl_->objects.find(objectId);
    if (objIt == impl_->objects.end()) {
        return empty;
    }
    const ObjectRecord& rec = objIt->second;

    const auto eikIt = rec.scalarFields.find("eikonal");
    const auto sdfIt = rec.scalarFields.find("sdf");
    if (eikIt == rec.scalarFields.end() || !eikIt->second ||
        sdfIt == rec.scalarFields.end() || !sdfIt->second) {
        return empty;
    }

    const SelectedIsosurfacesResult selectedLayers =
        selectSlicingIsosurfaces(objectId, adaptivity);
    if (!selectedLayers.valid || selectedLayers.layers.empty()) {
        return empty;
    }

    setFieldProgress(true, 0.08f, "Generating non-planar shell toolpaths...");
    auto progressCb = [](std::size_t done, std::size_t total) {
        const float frac = 0.08f + 0.90f * (static_cast<float>(done) / static_cast<float>(std::max<std::size_t>(1, total)));
        const std::string msg = "Generating shell toolpaths (" + std::to_string(done) + "/" + std::to_string(total) + " layers)...";
        setFieldProgress(true, frac, msg.c_str());
    };

    const openvdb::FloatGrid* coarseSdfPtr = nullptr;
    const openvdb::FloatGrid* coarseEikPtr = nullptr;
    const openvdb::Vec3SGrid* coarseToolAxisPtr = nullptr;
    if (rec.isMultiResolution) {
        const auto cSdfIt = rec.coarseScalarFields.find("sdf");
        const auto cEikIt = rec.coarseScalarFields.find("eikonal");
        if (cSdfIt != rec.coarseScalarFields.end() && cSdfIt->second &&
            cEikIt != rec.coarseScalarFields.end() && cEikIt->second) {
            coarseSdfPtr = cSdfIt->second.get();
            coarseEikPtr = cEikIt->second.get();
            coarseToolAxisPtr = rec.coarseToolAxisGrid.get();
        }
    }

    ObjectToolpathsResult res = generateShellToolpathsForLayers(
        selectedLayers,
        *sdfIt->second,
        *eikIt->second,
        rec.toolAxisGrid.get(),
        impl_->nozzleAxisBed,
        impl_->sliceLayerHeight,
        impl_->shellPathSpec,
        progressCb,
        coarseSdfPtr,
        coarseEikPtr,
        coarseToolAxisPtr,
        rec.fineShellInnerDepthMm);
    setFieldProgress(false, 1.0f, "Shell toolpaths ready.");
    return res;
}

NozzleBasis query_nozzle_frame(
    std::array<float, 3> position_bed,
    std::array<float, 3> travel_dir_bed)
{
    std::lock_guard<std::mutex> lock(g_backendMutex);
    return globalBackend().queryNozzleFrame(position_bed, travel_dir_bed);
}

std::array<float, 3> query_gravity_direction(std::array<float, 3> position_bed) {
    std::lock_guard<std::mutex> lock(g_backendMutex);
    return globalBackend().queryGravityDirection(position_bed);
}

void set_nozzle_axis_bed(std::array<float, 3> axis_bed) {
    std::lock_guard<std::mutex> lock(g_backendMutex);
    globalBackend().setNozzleAxisBed(axis_bed);
}

void set_gravity_direction_bed(std::array<float, 3> gravity_bed) {
    std::lock_guard<std::mutex> lock(g_backendMutex);
    globalBackend().setGravityDirectionBed(gravity_bed);
}

void set_machine_tools(rust::Slice<const ToolSpec> tools, std::uint32_t active_tool_index) {
    std::lock_guard<std::mutex> lock(g_backendMutex);
    std::vector<ToolSpec> copied;
    copied.reserve(tools.size());
    for (const auto& t : tools) {
        copied.push_back(t);
    }
    globalBackend().setMachineTools(copied, active_tool_index);
}

rust::Vec<ToolSpec> get_machine_tools() {
    std::lock_guard<std::mutex> lock(g_backendMutex);
    const auto& src = globalBackend().machineTools();
    rust::Vec<ToolSpec> out;
    out.reserve(src.size());
    for (const auto& t : src) {
        out.push_back(t);
    }
    return out;
}

std::uint32_t get_active_tool_index() {
    std::lock_guard<std::mutex> lock(g_backendMutex);
    return globalBackend().activeToolIndex();
}

void set_slice_layer_height(LayerHeightSpec layer_height) {
    std::lock_guard<std::mutex> lock(g_backendMutex);
    globalBackend().setSliceLayerHeight(layer_height);
}

LayerHeightSpec get_slice_layer_height() {
    std::lock_guard<std::mutex> lock(g_backendMutex);
    return globalBackend().sliceLayerHeight();
}

void set_eikonal_solver_iterations(std::uint32_t iterations) {
    std::lock_guard<std::mutex> lock(g_backendMutex);
    globalBackend().setEikonalSolverIterations(iterations);
}

std::uint32_t get_eikonal_solver_iterations() {
    std::lock_guard<std::mutex> lock(g_backendMutex);
    return globalBackend().eikonalSolverIterations();
}

void set_multi_resolution_grid_config(bool enabled, std::uint32_t coarse_ratio) {
    std::lock_guard<std::mutex> lock(g_backendMutex);
    globalBackend().setMultiResolutionConfig(enabled, coarse_ratio);
}

void set_spatial_orientation_hints(rust::Slice<const SpatialOrientationHint> hints) {
    std::lock_guard<std::mutex> lock(g_backendMutex);
    std::vector<SpatialOrientationHint> copied;
    copied.reserve(hints.size());
    for (const auto& h : hints) {
        copied.push_back(h);
    }
    globalBackend().setSpatialOrientationHints(copied);
}

SelectedIsosurfacesResult select_slicing_isosurfaces(std::uint64_t object_id, float adaptivity) {
    std::lock_guard<std::mutex> lock(g_backendMutex);
    return globalBackend().selectSlicingIsosurfaces(object_id, adaptivity);
}

void set_shell_path_spec(ShellPathSpec spec) {
    std::lock_guard<std::mutex> lock(g_backendMutex);
    globalBackend().setShellPathSpec(spec);
}

ShellPathSpec get_shell_path_spec() {
    std::lock_guard<std::mutex> lock(g_backendMutex);
    return globalBackend().shellPathSpec();
}

ObjectToolpathsResult generate_object_shell_toolpaths(std::uint64_t object_id, float adaptivity) {
    std::lock_guard<std::mutex> lock(g_backendMutex);
    return globalBackend().generateObjectShellToolpaths(object_id, adaptivity);
}

} // namespace manifold
