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
#include <IMeshTools_MeshAlgoType.hxx>
#include <IMeshTools_Parameters.hxx>
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

struct QuantizedVertexKey {
    std::int64_t ix = 0;
    std::int64_t iy = 0;
    std::int64_t iz = 0;

    bool operator==(const QuantizedVertexKey& o) const noexcept {
        return ix == o.ix && iy == o.iy && iz == o.iz;
    }
};

struct QuantizedVertexKeyHash {
    std::size_t operator()(const QuantizedVertexKey& k) const noexcept {
        std::size_t h = static_cast<std::size_t>(k.ix) * 73856093ULL;
        h ^= static_cast<std::size_t>(k.iy) * 19349663ULL;
        h ^= static_cast<std::size_t>(k.iz) * 83492791ULL;
        return h;
    }
};

struct UndirectedEdgeKey {
    std::uint32_t u = 0;
    std::uint32_t v = 0;

    bool operator==(const UndirectedEdgeKey& o) const noexcept {
        return u == o.u && v == o.v;
    }
};

struct UndirectedEdgeKeyHash {
    std::size_t operator()(const UndirectedEdgeKey& k) const noexcept {
        return (static_cast<std::size_t>(k.u) << 32) ^ static_cast<std::size_t>(k.v);
    }
};

struct BoundaryLoopDescriptor {
    std::vector<std::uint32_t> vertices;
    openvdb::Vec3d centroid{0.0, 0.0, 0.0};
    openvdb::Vec3d normal{0.0, 0.0, 1.0};
    double normalLen = 0.0;
    double meanRadius = 0.0;
};

BoundaryLoopDescriptor analyzeBoundaryLoop(
    const std::vector<std::uint32_t>& loopVerts,
    const std::vector<openvdb::Vec3s>& points)
{
    BoundaryLoopDescriptor desc;
    desc.vertices = loopVerts;
    const std::size_t n = loopVerts.size();
    if (n == 0) {
        return desc;
    }

    openvdb::Vec3d c(0.0, 0.0, 0.0);
    for (const std::uint32_t vid : loopVerts) {
        const auto& p = points[vid];
        c += openvdb::Vec3d(p.x(), p.y(), p.z());
    }
    c /= static_cast<double>(n);
    desc.centroid = c;

    double rSum = 0.0;
    openvdb::Vec3d newell(0.0, 0.0, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        const auto& p0 = points[loopVerts[i]];
        const auto& p1 = points[loopVerts[(i + 1) % n]];
        const openvdb::Vec3d d0(p0.x() - c.x(), p0.y() - c.y(), p0.z() - c.z());
        const openvdb::Vec3d d1(p1.x() - c.x(), p1.y() - c.y(), p1.z() - c.z());
        rSum += d0.length();
        newell += d0.cross(d1);
    }
    desc.meanRadius = rSum / static_cast<double>(n);
    desc.normalLen = newell.length();
    if (desc.normalLen > 1.0e-12) {
        desc.normal = newell / desc.normalLen;
    }
    return desc;
}

void triangulateSingleBoundaryLoop(
    const BoundaryLoopDescriptor& desc,
    const std::vector<openvdb::Vec3s>& points,
    std::vector<openvdb::Vec3I>& outPatchTris)
{
    const auto& loop = desc.vertices;
    const std::size_t n = loop.size();
    if (n < 3) {
        return;
    }
    if (n == 3) {
        if (loop[0] != loop[1] && loop[1] != loop[2] && loop[2] != loop[0]) {
            outPatchTris.emplace_back(loop[0], loop[1], loop[2]);
        }
        return;
    }
    if (n == 4) {
        const double d02 = (points[loop[0]] - points[loop[2]]).lengthSqr();
        const double d13 = (points[loop[1]] - points[loop[3]]).lengthSqr();
        if (d02 <= d13) {
            outPatchTris.emplace_back(loop[0], loop[1], loop[2]);
            outPatchTris.emplace_back(loop[0], loop[2], loop[3]);
        } else {
            outPatchTris.emplace_back(loop[0], loop[1], loop[3]);
            outPatchTris.emplace_back(loop[1], loop[2], loop[3]);
        }
        return;
    }

    // Build a 2D projection plane orthogonal to desc.normal
    const openvdb::Vec3d nrm = desc.normal;
    const openvdb::Vec3d refAxis =
        (std::abs(nrm.x()) < 0.9) ? openvdb::Vec3d(1.0, 0.0, 0.0) : openvdb::Vec3d(0.0, 1.0, 0.0);
    openvdb::Vec3d uAxis = (refAxis - refAxis.dot(nrm) * nrm);
    const double uLen = uAxis.length();
    if (uLen > 1.0e-12) {
        uAxis /= uLen;
    } else {
        uAxis = openvdb::Vec3d(1.0, 0.0, 0.0);
    }
    openvdb::Vec3d vAxis = nrm.cross(uAxis);

    std::vector<openvdb::Vec2d> uv(n);
    for (std::size_t i = 0; i < n; ++i) {
        const auto& p = points[loop[i]];
        const openvdb::Vec3d d(p.x() - desc.centroid.x(), p.y() - desc.centroid.y(), p.z() - desc.centroid.z());
        uv[i] = openvdb::Vec2d(d.dot(uAxis), d.dot(vAxis));
    }

    double signedArea2 = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const auto& a = uv[i];
        const auto& b = uv[(i + 1) % n];
        signedArea2 += a.x() * b.y() - b.x() * a.y();
    }
    if (signedArea2 < 0.0) {
        for (auto& p : uv) {
            p.y() = -p.y();
        }
    }

    std::vector<std::size_t> poly(n);
    std::iota(poly.begin(), poly.end(), 0);

    auto cross2d = [](const openvdb::Vec2d& a, const openvdb::Vec2d& b, const openvdb::Vec2d& c) -> double {
        return (b.x() - a.x()) * (c.y() - a.y()) - (b.y() - a.y()) * (c.x() - a.x());
    };
    auto pointInTri2d = [&](const openvdb::Vec2d& p,
                            const openvdb::Vec2d& a,
                            const openvdb::Vec2d& b,
                            const openvdb::Vec2d& c) -> bool {
        const double c0 = cross2d(a, b, p);
        const double c1 = cross2d(b, c, p);
        const double c2 = cross2d(c, a, p);
        return (c0 >= -1.0e-9 && c1 >= -1.0e-9 && c2 >= -1.0e-9);
    };

    while (poly.size() > 3) {
        const std::size_t m = poly.size();
        std::size_t bestIdx = m;
        double bestDiagSq = std::numeric_limits<double>::max();

        for (std::size_t i = 0; i < m; ++i) {
            const std::size_t iPrev = poly[(i + m - 1) % m];
            const std::size_t iCurr = poly[i];
            const std::size_t iNext = poly[(i + 1) % m];
            if (loop[iPrev] == loop[iCurr] || loop[iCurr] == loop[iNext] || loop[iNext] == loop[iPrev]) {
                continue;
            }
            const double earCross = cross2d(uv[iPrev], uv[iCurr], uv[iNext]);
            if (earCross <= 1.0e-9) {
                continue;
            }
            bool anyInside = false;
            for (std::size_t j = 0; j < m; ++j) {
                if (j == (i + m - 1) % m || j == i || j == (i + 1) % m) {
                    continue;
                }
                const std::size_t iOther = poly[j];
                if (loop[iOther] == loop[iPrev] || loop[iOther] == loop[iCurr] || loop[iOther] == loop[iNext]) {
                    continue;
                }
                if (pointInTri2d(uv[iOther], uv[iPrev], uv[iCurr], uv[iNext])) {
                    anyInside = true;
                    break;
                }
            }
            if (!anyInside) {
                const double diagSq = (points[loop[iPrev]] - points[loop[iNext]]).lengthSqr();
                if (diagSq < bestDiagSq) {
                    bestDiagSq = diagSq;
                    bestIdx = i;
                }
            }
        }

        if (bestIdx == m) {
            // Fallback for non-planar or nearly-collinear 3D loops: pick shortest 3D diagonal
            for (std::size_t i = 0; i < m; ++i) {
                const std::size_t iPrev = poly[(i + m - 1) % m];
                const std::size_t iNext = poly[(i + 1) % m];
                const double diagSq = (points[loop[iPrev]] - points[loop[iNext]]).lengthSqr();
                if (diagSq < bestDiagSq) {
                    bestDiagSq = diagSq;
                    bestIdx = i;
                }
            }
            if (bestIdx == m) {
                bestIdx = 0;
            }
        }

        const std::size_t iPrev = poly[(bestIdx + m - 1) % m];
        const std::size_t iCurr = poly[bestIdx];
        const std::size_t iNext = poly[(bestIdx + 1) % m];
        if (loop[iPrev] != loop[iCurr] && loop[iCurr] != loop[iNext] && loop[iNext] != loop[iPrev]) {
            outPatchTris.emplace_back(loop[iPrev], loop[iCurr], loop[iNext]);
        }
        poly.erase(poly.begin() + static_cast<std::ptrdiff_t>(bestIdx));
    }

    if (poly.size() == 3) {
        const std::uint32_t v0 = loop[poly[0]];
        const std::uint32_t v1 = loop[poly[1]];
        const std::uint32_t v2 = loop[poly[2]];
        if (v0 != v1 && v1 != v2 && v2 != v0) {
            outPatchTris.emplace_back(v0, v1, v2);
        }
    }
}

void loftCoaxialBoundaryLoops(
    const BoundaryLoopDescriptor& loopA,
    const BoundaryLoopDescriptor& loopB,
    const std::vector<openvdb::Vec3s>& points,
    std::vector<openvdb::Vec3I>& outPatchTris)
{
    const auto& a = loopA.vertices;
    const auto& bOrig = loopB.vertices;
    const std::size_t N = a.size();
    const std::size_t M = bOrig.size();
    if (N < 3 || M < 3) {
        return;
    }

    // Reverse loopB so its traversal circulates in the same angular direction around loopA.normal
    // while its reversed edges (bRev[k+1] -> bRev[k]) correspond to forward edges of loopB.
    std::vector<std::uint32_t> bRev(M);
    for (std::size_t k = 0; k < M; ++k) {
        bRev[k] = bOrig[(M - k) % M];
    }

    // Find the index k0 in bRev closest to a[0]
    std::size_t bestK0 = 0;
    double bestDistSq = std::numeric_limits<double>::max();
    for (std::size_t k = 0; k < M; ++k) {
        const double dSq = (points[a[0]] - points[bRev[k]]).lengthSqr();
        if (dSq < bestDistSq) {
            bestDistSq = dSq;
            bestK0 = k;
        }
    }

    std::vector<std::uint32_t> b(M);
    for (std::size_t k = 0; k < M; ++k) {
        b[k] = bRev[(bestK0 + k) % M];
    }

    std::size_t i = 0;
    std::size_t m = 0;
    while (i < N || m < M) {
        const std::uint32_t u0 = a[i % N];
        const std::uint32_t u1 = a[(i + 1) % N];
        const std::uint32_t w0 = b[m % M];
        const std::uint32_t w1 = b[(m + 1) % M];

        bool advanceA = false;
        if (i >= N) {
            advanceA = false;
        } else if (m >= M) {
            advanceA = true;
        } else {
            const double diagU0W1 = (points[u0] - points[w1]).lengthSqr();
            const double diagU1W0 = (points[u1] - points[w0]).lengthSqr();
            advanceA = (diagU1W0 < diagU0W1);
        }

        if (advanceA) {
            if (u0 != u1 && u1 != w0 && w0 != u0) {
                outPatchTris.emplace_back(u0, u1, w0);
            }
            ++i;
        } else {
            if (u0 != w1 && w1 != w0 && w0 != u0) {
                outPatchTris.emplace_back(u0, w1, w0);
            }
            ++m;
        }
    }
}

/// Welds coincident vertices across face boundaries and repairs any open boundary loops
/// (such as untessellated B-Rep conical/cylindrical frustum faces or micro-gaps) so that
/// `openvdb::tools::meshToSignedDistanceField` receives a closed, watertight manifold mesh.
void extractWeldedWatertightMesh(
    const std::vector<RenderVertex>& vertexBuffer,
    const std::vector<std::uint32_t>& indexBuffer,
    std::vector<openvdb::Vec3s>& outPoints,
    std::vector<openvdb::Vec3I>& outTriangles,
    std::vector<openvdb::Vec3I>* outPatchTriangles = nullptr)
{
    outPoints.clear();
    outTriangles.clear();
    if (outPatchTriangles != nullptr) {
        outPatchTriangles->clear();
    }
    if (vertexBuffer.empty() || indexBuffer.size() < 3) {
        return;
    }

    // 1. Spatial vertex welding at 1 micrometer (1e-3 mm) resolution
    constexpr double kQuantScale = 1000.0;
    std::unordered_map<QuantizedVertexKey, std::uint32_t, QuantizedVertexKeyHash> weldMap;
    weldMap.reserve(vertexBuffer.size());
    std::vector<std::uint32_t> oldToWelded(vertexBuffer.size(), 0);
    outPoints.reserve(vertexBuffer.size() / 2 + 16);

    for (std::size_t i = 0; i < vertexBuffer.size(); ++i) {
        const auto& v = vertexBuffer[i];
        const QuantizedVertexKey key{
            static_cast<std::int64_t>(std::llround(static_cast<double>(v.x) * kQuantScale)),
            static_cast<std::int64_t>(std::llround(static_cast<double>(v.y) * kQuantScale)),
            static_cast<std::int64_t>(std::llround(static_cast<double>(v.z) * kQuantScale)),
        };
        auto [it, inserted] = weldMap.emplace(key, static_cast<std::uint32_t>(outPoints.size()));
        if (inserted) {
            outPoints.emplace_back(v.x, v.y, v.z);
        }
        oldToWelded[i] = it->second;
    }

    outTriangles.reserve(indexBuffer.size() / 3);
    struct EdgeDirCounts {
        std::uint32_t forwardCount = 0; // u -> v (u < v)
        std::uint32_t reverseCount = 0; // v -> u (u < v)
    };
    std::unordered_map<UndirectedEdgeKey, EdgeDirCounts, UndirectedEdgeKeyHash> edgeCounts;
    edgeCounts.reserve(indexBuffer.size() / 2);

    for (std::size_t t = 0; t + 2 < indexBuffer.size(); t += 3) {
        const std::uint32_t i0 = indexBuffer[t + 0];
        const std::uint32_t i1 = indexBuffer[t + 1];
        const std::uint32_t i2 = indexBuffer[t + 2];
        if (i0 >= vertexBuffer.size() || i1 >= vertexBuffer.size() || i2 >= vertexBuffer.size()) {
            continue;
        }
        const std::uint32_t w0 = oldToWelded[i0];
        const std::uint32_t w1 = oldToWelded[i1];
        const std::uint32_t w2 = oldToWelded[i2];
        if (w0 == w1 || w1 == w2 || w2 == w0) {
            continue;
        }
        outTriangles.emplace_back(w0, w1, w2);

        const std::array<std::pair<std::uint32_t, std::uint32_t>, 3> halfEdges{{
            {w0, w1},
            {w1, w2},
            {w2, w0},
        }};
        for (const auto& [a, b] : halfEdges) {
            const UndirectedEdgeKey ek{ std::min(a, b), std::max(a, b) };
            auto& cnt = edgeCounts[ek];
            if (a < b) {
                ++cnt.forwardCount;
            } else {
                ++cnt.reverseCount;
            }
        }
    }

    // 2. Collect open boundary half-edges (in the opposite direction needed to close the hole)
    std::unordered_map<std::uint32_t, std::vector<std::uint32_t>> openAdj;
    for (const auto& [ek, cnt] : edgeCounts) {
        if (cnt.forwardCount == 1 && cnt.reverseCount == 0) {
            // Existing mesh has u -> v; hole-closing patch needs v -> u
            openAdj[ek.v].push_back(ek.u);
        } else if (cnt.reverseCount == 1 && cnt.forwardCount == 0) {
            // Existing mesh has v -> u; hole-closing patch needs u -> v
            openAdj[ek.u].push_back(ek.v);
        }
    }

    if (!openAdj.empty()) {
        std::vector<BoundaryLoopDescriptor> loops;
        for (auto& [startU, neighbors] : openAdj) {
            while (!neighbors.empty()) {
                const std::uint32_t firstV = neighbors.back();
                neighbors.pop_back();

                std::vector<std::uint32_t> path;
                std::unordered_map<std::uint32_t, std::size_t> posInPath;
                path.push_back(startU);
                posInPath[startU] = 0;

                std::uint32_t curr = firstV;
                while (true) {
                    auto visitedIt = posInPath.find(curr);
                    if (visitedIt != posInPath.end()) {
                        // Extract simple cycle from visitedIt->second .. path.end()
                        const std::size_t cycleStart = visitedIt->second;
                        if (path.size() - cycleStart >= 3) {
                            std::vector<std::uint32_t> cycle(
                                path.begin() + static_cast<std::ptrdiff_t>(cycleStart),
                                path.end());
                            loops.push_back(analyzeBoundaryLoop(cycle, outPoints));
                        }
                        for (std::size_t k = cycleStart; k < path.size(); ++k) {
                            posInPath.erase(path[k]);
                        }
                        path.resize(cycleStart);
                        if (path.empty()) {
                            break;
                        }
                    }

                    path.push_back(curr);
                    posInPath[curr] = path.size() - 1;

                    auto nextIt = openAdj.find(curr);
                    if (nextIt == openAdj.end() || nextIt->second.empty()) {
                        break;
                    }
                    const std::uint32_t nxt = nextIt->second.back();
                    nextIt->second.pop_back();
                    curr = nxt;
                }
            }
        }

        // 3. Pair coaxial opposite-winding loops (such as missing conical/cylindrical hole faces)
        std::vector<bool> loopPaired(loops.size(), false);
        std::vector<openvdb::Vec3I> patchTris;

        for (std::size_t i = 0; i < loops.size(); ++i) {
            if (loopPaired[i] || loops[i].vertices.size() < 6 || loops[i].normalLen <= 1.0e-9) {
                continue;
            }
            std::size_t bestPartner = loops.size();
            double bestPartnerDist = std::numeric_limits<double>::max();

            for (std::size_t j = i + 1; j < loops.size(); ++j) {
                if (loopPaired[j] || loops[j].vertices.size() < 6 || loops[j].normalLen <= 1.0e-9) {
                    continue;
                }
                const std::size_t nMin = std::min(loops[i].vertices.size(), loops[j].vertices.size());
                const std::size_t nMax = std::max(loops[i].vertices.size(), loops[j].vertices.size());
                if (nMin * 2 < nMax) {
                    continue;
                }
                const double nDot = loops[i].normal.dot(loops[j].normal);
                if (nDot > -0.70) {
                    continue;
                }
                const openvdb::Vec3d deltaC = loops[j].centroid - loops[i].centroid;
                const double distC = deltaC.length();
                const double rMax = std::max({loops[i].meanRadius, loops[j].meanRadius, 1.0e-4});
                const double rMin = std::min(loops[i].meanRadius, loops[j].meanRadius);
                if (rMin < 0.15 * rMax || distC > 4.0 * rMax) {
                    continue;
                }
                if (distC > 1.0e-4) {
                    const double axialAlign = std::abs(deltaC.dot(loops[i].normal)) / distC;
                    if (axialAlign < 0.80) {
                        continue;
                    }
                }
                if (distC < bestPartnerDist) {
                    bestPartnerDist = distC;
                    bestPartner = j;
                }
            }

            if (bestPartner < loops.size()) {
                loopPaired[i] = true;
                loopPaired[bestPartner] = true;
                loftCoaxialBoundaryLoops(loops[i], loops[bestPartner], outPoints, patchTris);
            }
        }

        // 4. Close any remaining unpaired boundary loops via 3D planar ear-clipping
        for (std::size_t i = 0; i < loops.size(); ++i) {
            if (!loopPaired[i]) {
                triangulateSingleBoundaryLoop(loops[i], outPoints, patchTris);
            }
        }

        outTriangles.insert(outTriangles.end(), patchTris.begin(), patchTris.end());
        if (outPatchTriangles != nullptr) {
            *outPatchTriangles = std::move(patchTris);
        }
    }

    // 5. Ensure positive global signed volume (outward surface orientation)
    double signedVol6 = 0.0;
    for (const auto& tri : outTriangles) {
        const openvdb::Vec3d p0(outPoints[tri[0]]);
        const openvdb::Vec3d p1(outPoints[tri[1]]);
        const openvdb::Vec3d p2(outPoints[tri[2]]);
        signedVol6 += p0.dot(p1.cross(p2));
    }
    if (signedVol6 < 0.0) {
        for (auto& tri : outTriangles) {
            std::swap(tri[1], tri[2]);
        }
        if (outPatchTriangles != nullptr) {
            for (auto& tri : *outPatchTriangles) {
                std::swap(tri[1], tri[2]);
            }
        }
    }
}
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
    std::vector<openvdb::Vec3I> triangles;
    extractWeldedWatertightMesh(rec.vertexBuffer, rec.indexBuffer, points, triangles, nullptr);
    if (points.empty() || triangles.empty()) {
        return;
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
    const OverhangBiasSpec biasSpec = backend.overhangOrientationBiasSpec();
    OverhangOrientationBiasConfig biasCfg{};
    biasCfg.enabled = biasSpec.enabled;
    biasCfg.weight = std::clamp(biasSpec.weight, 0.0f, 1.0f);
    biasCfg.orthogonalSupportedAngleDeg = std::clamp(biasSpec.orthogonal_supported_angle_deg, 0.0f, 89.0f);
    biasCfg.orthogonalOverhangAngleDeg = std::clamp(biasSpec.orthogonal_overhang_angle_deg, 0.0f, 89.0f);
    biasCfg.transitionWidthDeg = std::clamp(biasSpec.transition_width_deg, 1.0f, 45.0f);

    const ShellPathSpec localShellSpec = backend.shellPathSpec();
    const SupportSpec localSupportSpec = backend.supportSpec();
    const float effectiveSupportBridgeMm =
        (localSupportSpec.max_bridge_length_mm > 0.0f)
            ? localSupportSpec.max_bridge_length_mm
            : localShellSpec.max_bridge_length_mm;

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
        biasCfg,
        localShellSpec.overhang_angle_threshold_deg,
        effectiveSupportBridgeMm,
        localSupportSpec.enabled && localSupportSpec.seed_order_field,
        localSupportSpec.only_unreachable_and_long_bridges,
        backend.orderFieldMode() == 1,
    };

    for (const auto& ffiHint : backend.spatialOrientationHints()) {
        if (ffiHint.kind != 0 && ffiHint.kind != 1) {
            continue;
        }
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
                rec.scalarFields["support_needed"] = fineDirectSolved.supportNeededGrid;
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
                rec.coarseScalarFields["support_needed"] = coarseSolved.supportNeededGrid;
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
                rec.scalarFields["support_needed"] = fineSolved.supportNeededGrid;
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
    rec.scalarFields["support_needed"] = solved.supportNeededGrid;
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

    IMeshTools_Parameters meshParams;
    meshParams.Deflection = std::max(linearDeflection, 1.0e-4);
    meshParams.Angle = std::max(angularDeflection, 1.0e-3);
    meshParams.InParallel = Standard_True;
    meshParams.Relative = Standard_False;
    BRepMesh_IncrementalMesh mesher(shape, meshParams);

    // Check if any face failed the default Watson 2D Delaunay triangulation; if so,
    // retry with Delabella and relaxed surface-deflection / tolerance controls.
    bool anyNullFace = false;
    for (TopExp_Explorer explorer(shape, TopAbs_FACE); explorer.More(); explorer.Next()) {
        const TopoDS_Face& face = TopoDS::Face(explorer.Current());
        TopLoc_Location location;
        if (BRep_Tool::Triangulation(face, location).IsNull()) {
            anyNullFace = true;
            break;
        }
    }
    if (anyNullFace) {
        IMeshTools_Parameters fallbackParams = meshParams;
        fallbackParams.MeshAlgo = IMeshTools_MeshAlgoType_Delabella;
        fallbackParams.ForceFaceDeflection = Standard_True;
        fallbackParams.ControlSurfaceDeflection = Standard_False;
        fallbackParams.AllowQualityDecrease = Standard_True;
        fallbackParams.AdjustMinSize = Standard_True;
        BRepMesh_IncrementalMesh fallbackMesher(shape, fallbackParams);
    }

    std::uint32_t currentFaceId = 0;
    for (TopExp_Explorer explorer(shape, TopAbs_FACE); explorer.More(); explorer.Next()) {
        const TopoDS_Face& face = TopoDS::Face(explorer.Current());
        TopLoc_Location location;
        const Handle(Poly_Triangulation)& triangulation = BRep_Tool::Triangulation(face, location);

        if (triangulation.IsNull()) {
            continue;
        }

        const gp_Trsf& trsf = location.Transformation();
        const bool isReversed =
            (face.Orientation() == TopAbs_REVERSED) != (trsf.IsNegative() == Standard_True);
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

    // Reconstruct any faces that failed OCCT triangulation (such as missing conical chamfer
    // frustums or micro-gaps) by welding boundary vertices and lofting/triangulating open loops.
    std::vector<openvdb::Vec3s> weldedPts;
    std::vector<openvdb::Vec3I> weldedTris;
    std::vector<openvdb::Vec3I> patchTris;
    extractWeldedWatertightMesh(vertexBuffer, indexBuffer, weldedPts, weldedTris, &patchTris);
    if (!patchTris.empty()) {
        for (const auto& tri : patchTris) {
            const auto& p0 = weldedPts[tri[0]];
            const auto& p1 = weldedPts[tri[1]];
            const auto& p2 = weldedPts[tri[2]];
            const openvdb::Vec3s e1 = p1 - p0;
            const openvdb::Vec3s e2 = p2 - p0;
            openvdb::Vec3s nrm = e1.cross(e2);
            const float nLen = nrm.length();
            if (nLen > 1.0e-7f) {
                nrm /= nLen;
            } else {
                nrm = openvdb::Vec3s(0.0f, 0.0f, 1.0f);
            }

            const auto baseIdx = static_cast<std::uint32_t>(vertexBuffer.size());
            vertexBuffer.push_back(RenderVertex{p0.x(), p0.y(), p0.z(), nrm.x(), nrm.y(), nrm.z()});
            vertexBuffer.push_back(RenderVertex{p1.x(), p1.y(), p1.z(), nrm.x(), nrm.y(), nrm.z()});
            vertexBuffer.push_back(RenderVertex{p2.x(), p2.y(), p2.z(), nrm.x(), nrm.y(), nrm.z()});
            indexBuffer.push_back(baseIdx + 0);
            indexBuffer.push_back(baseIdx + 1);
            indexBuffer.push_back(baseIdx + 2);
            triangleFaceIds.push_back(currentFaceId);
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
    t.clearance_mode = 0;
    const float tipOuterR = std::max(diameterMm * 1.25f, 0.4f);
    t.clearance_profile_xz.push_back(ClearanceXZ{tipOuterR, 0.0f});
    t.clearance_profile_xz.push_back(ClearanceXZ{2.5f, 2.0f});
    t.clearance_profile_xz.push_back(ClearanceXZ{9.0f, 6.0f});
    t.clearance_profile_xz.push_back(ClearanceXZ{22.0f, 18.0f});
    t.clearance_profile_xz.push_back(ClearanceXZ{28.0f, 45.0f});
    t.cone_tip_radius_mm = tipOuterR;
    t.cone_half_angle_deg = 35.0f;
    t.cone_height_mm = 45.0f;
    t.rect_tip_half_width_x_mm = tipOuterR;
    t.rect_tip_half_width_y_mm = tipOuterR;
    t.rect_half_angle_x_deg = 35.0f;
    t.rect_half_angle_y_deg = 25.0f;
    t.rect_height_mm = 45.0f;
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
    std::uint8_t orderFieldMode = 0;
    std::uint32_t eikonalSolverIterations = 4;
    bool multiResolutionEnabled = true;
    std::uint32_t multiResolutionCoarseRatio = 4;
    std::vector<SpatialOrientationHint> spatialOrientationHints;
    OverhangBiasSpec overhangBiasSpec{
        true,
        0.75f,
        30.0f,
        60.0f,
        15.0f,
    };
    ConformalSkinSpec conformalSkinSpec{
        false,
        3,
        12.0f,
        60.0f,
    };
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
        45.0f,
        40.0f,
        0.08f,
    };
    SupportSpec supportSpec{
        true,
        -1,
        0,
        false,
        true,
        false,
        40.0f,
        0.20f,
        0.50f,
        2,
        1,
        2.50f,
        0.15f,
        0.40f,
        45.0f,
        2.40f,
        12.00f,
        2,
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

std::uint64_t SlicerBackend::cloneObject(
    std::uint64_t sourceObjectId,
    const std::string& cloneName)
{
    const auto it = impl_->objects.find(sourceObjectId);
    if (it == impl_->objects.end()) {
        return 0;
    }

    const ObjectRecord& src = it->second;
    ObjectRecord rec;
    rec.id = impl_->nextObjectId++;
    rec.name = cloneName.empty() ? (src.name + " (Copy)") : cloneName;
    rec.shape = src.shape;
    rec.hasBrep = src.hasBrep;
    rec.originX = src.originX;
    rec.originY = src.originY;
    rec.originZ = src.originZ;
    rec.minX = src.minX;
    rec.minY = src.minY;
    rec.minZ = src.minZ;
    rec.maxX = src.maxX;
    rec.maxY = src.maxY;
    rec.maxZ = src.maxZ;
    rec.vertexBuffer = src.vertexBuffer;
    rec.indexBuffer = src.indexBuffer;
    rec.triangleFaceIds = src.triangleFaceIds;
    rec.isMultiResolution = src.isMultiResolution;
    rec.fineShellInnerDepthMm = src.fineShellInnerDepthMm;
    rec.coarseToFineRatio = src.coarseToFineRatio;
    rec.hasCachedSelectedLayers = src.hasCachedSelectedLayers;
    rec.cachedSelectedLayersAdaptivity = src.cachedSelectedLayersAdaptivity;
    rec.cachedSelectedLayersSpec = src.cachedSelectedLayersSpec;
    rec.cachedSelectedLayers = src.cachedSelectedLayers;

    for (const auto& kv : src.scalarFields) {
        if (kv.second) {
            rec.scalarFields[kv.first] = kv.second->deepCopy();
        }
    }
    if (src.toolAxisGrid) {
        rec.toolAxisGrid = src.toolAxisGrid->deepCopy();
    }
    for (const auto& kv : src.coarseScalarFields) {
        if (kv.second) {
            rec.coarseScalarFields[kv.first] = kv.second->deepCopy();
        }
    }
    if (src.coarseToolAxisGrid) {
        rec.coarseToolAxisGrid = src.coarseToolAxisGrid->deepCopy();
    }

    const std::uint64_t newId = rec.id;
    impl_->lastLoadedId = newId;
    impl_->objects.emplace(newId, std::move(rec));
    return newId;
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

std::vector<float> SlicerBackend::objectVertexToolAxes(std::uint64_t objectId) const {
    const auto it = impl_->objects.find(objectId);
    if (it == impl_->objects.end()) {
        return {};
    }
    const ObjectRecord& rec = it->second;
    if (!rec.toolAxisGrid || rec.vertexBuffer.empty()) {
        return {};
    }

    const openvdb::Vec3SGrid& grid = *rec.toolAxisGrid;
    const float voxelSize = static_cast<float>(grid.transform().voxelSize()[0]);
    const float inwardStep = 0.60f * voxelSize;
    openvdb::tools::GridSampler<openvdb::Vec3SGrid, openvdb::tools::BoxSampler> sampler(grid);

    std::unique_ptr<openvdb::tools::GridSampler<openvdb::Vec3SGrid, openvdb::tools::BoxSampler>> coarseSampler;
    if (rec.isMultiResolution && rec.coarseToolAxisGrid) {
        coarseSampler = std::make_unique<
            openvdb::tools::GridSampler<openvdb::Vec3SGrid, openvdb::tools::BoxSampler>>(
            *rec.coarseToolAxisGrid);
    }

    const std::array<float, 3> fallback = impl_->nozzleAxisBed;
    auto normalizeVec3 = [](float x, float y, float z, const std::array<float, 3>& fb) -> std::array<float, 3> {
        const float len = std::sqrt(x * x + y * y + z * z);
        if (len > 1.0e-7f) {
            return {x / len, y / len, z / len};
        }
        return fb;
    };
    std::vector<float> out(rec.vertexBuffer.size() * 3, 0.0f);
    for (std::size_t i = 0; i < rec.vertexBuffer.size(); ++i) {
        const RenderVertex& v = rec.vertexBuffer[i];
        const auto nUnit = normalizeVec3(v.nx, v.ny, v.nz, fallback);
        // Sample slightly inside the surface (-0.6 * voxelSize * nUnit) so trilinear BoxSampler
        // reads solved interior voxels rather than exterior air voxels where toolAxis == bedNormal.
        const openvdb::Vec3d wpIn(
            v.x - inwardStep * nUnit[0],
            v.y - inwardStep * nUnit[1],
            v.z - inwardStep * nUnit[2]);
        openvdb::Vec3s sampled = sampler.wsSample(wpIn);
        float lenSq = sampled.x() * sampled.x() + sampled.y() * sampled.y() + sampled.z() * sampled.z();
        if (lenSq < 1.0e-4f) {
            const openvdb::Vec3d wp0(v.x, v.y, v.z);
            sampled = sampler.wsSample(wp0);
            lenSq = sampled.x() * sampled.x() + sampled.y() * sampled.y() + sampled.z() * sampled.z();
            if (lenSq < 1.0e-4f && coarseSampler) {
                sampled = coarseSampler->wsSample(wpIn);
            }
        }
        const auto unitAxis = normalizeVec3(sampled.x(), sampled.y(), sampled.z(), fallback);
        out[i * 3 + 0] = unitAxis[0];
        out[i * 3 + 1] = unitAxis[1];
        out[i * 3 + 2] = unitAxis[2];
    }
    return out;
}

std::vector<float> SlicerBackend::objectVertexSupportNeeded(std::uint64_t objectId) const {
    const auto it = impl_->objects.find(objectId);
    if (it == impl_->objects.end()) {
        return {};
    }
    const ObjectRecord& rec = it->second;
    const auto supIt = rec.scalarFields.find("support_needed");
    if (supIt == rec.scalarFields.end() || !supIt->second || rec.vertexBuffer.empty()) {
        return {};
    }

    const openvdb::FloatGrid& grid = *supIt->second;
    const float voxelSize = static_cast<float>(grid.transform().voxelSize()[0]);
    const float inwardStep = 0.60f * voxelSize;
    openvdb::tools::GridSampler<openvdb::FloatGrid, openvdb::tools::BoxSampler> sampler(grid);

    std::unique_ptr<openvdb::tools::GridSampler<openvdb::FloatGrid, openvdb::tools::BoxSampler>> coarseSampler;
    if (rec.isMultiResolution) {
        const auto cSupIt = rec.coarseScalarFields.find("support_needed");
        if (cSupIt != rec.coarseScalarFields.end() && cSupIt->second) {
            coarseSampler = std::make_unique<
                openvdb::tools::GridSampler<openvdb::FloatGrid, openvdb::tools::BoxSampler>>(
                *cSupIt->second);
        }
    }

    const std::array<float, 3> fallback = impl_->nozzleAxisBed;
    auto normalizeVec3 = [](float x, float y, float z, const std::array<float, 3>& fb) -> std::array<float, 3> {
        const float len = std::sqrt(x * x + y * y + z * z);
        if (len > 1.0e-7f) {
            return {x / len, y / len, z / len};
        }
        return fb;
    };

    std::vector<float> out(rec.vertexBuffer.size(), 0.0f);
    for (std::size_t i = 0; i < rec.vertexBuffer.size(); ++i) {
        const RenderVertex& v = rec.vertexBuffer[i];
        const auto nUnit = normalizeVec3(v.nx, v.ny, v.nz, fallback);
        const auto gDir = queryGravityDirection({v.x, v.y, v.z});
        const float cosGrav = nUnit[0] * gDir[0] + nUnit[1] * gDir[1] + nUnit[2] * gDir[2];
        const openvdb::Vec3d wp0(v.x, v.y, v.z);
        const openvdb::Vec3d wpIn(
            v.x - inwardStep * nUnit[0],
            v.y - inwardStep * nUnit[1],
            v.z - inwardStep * nUnit[2]);
        float val = std::max(sampler.wsSample(wp0), sampler.wsSample(wpIn));
        if (cosGrav > 0.25f) {
            // For downward-facing overhang vertices lying on sharp 90-degree CAD edges,
            // also probe 1.25*dx tangent to the face so edge dilution from adjacent vertical walls is avoided.
            const std::array<float, 3> refAxis =
                (std::abs(nUnit[0]) < 0.9f) ? std::array<float, 3>{1.0f, 0.0f, 0.0f}
                                            : std::array<float, 3>{0.0f, 1.0f, 0.0f};
            const float dRef = refAxis[0] * nUnit[0] + refAxis[1] * nUnit[1] + refAxis[2] * nUnit[2];
            const auto t0 = normalizeVec3(
                refAxis[0] - dRef * nUnit[0],
                refAxis[1] - dRef * nUnit[1],
                refAxis[2] - dRef * nUnit[2],
                {1.0f, 0.0f, 0.0f});
            const auto t1 = normalizeVec3(
                nUnit[1] * t0[2] - nUnit[2] * t0[1],
                nUnit[2] * t0[0] - nUnit[0] * t0[2],
                nUnit[0] * t0[1] - nUnit[1] * t0[0],
                {0.0f, 1.0f, 0.0f});
            const double tStep = 1.25 * static_cast<double>(voxelSize);
            for (const auto& tDir : {t0, t1}) {
                for (double sgn : {-1.0, 1.0}) {
                    const openvdb::Vec3d wpT(
                        wpIn.x() + sgn * tStep * tDir[0],
                        wpIn.y() + sgn * tStep * tDir[1],
                        wpIn.z() + sgn * tStep * tDir[2]);
                    val = std::max(val, sampler.wsSample(wpT));
                }
            }
        } else if (cosGrav < -0.10f && val < 0.95f) {
            // Upward-facing top surfaces do not require support unless the entire component is floating
            val = 0.0f;
        }
        if (val <= 1.0e-4f && coarseSampler) {
            val = std::max(val, coarseSampler->wsSample(wpIn));
        }
        out[i] = std::clamp(val, 0.0f, 1.0f);
    }
    return out;
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
        snapshotBackend.impl_->orderFieldMode = impl_->orderFieldMode;
        snapshotBackend.impl_->eikonalSolverIterations = impl_->eikonalSolverIterations;
        snapshotBackend.impl_->multiResolutionEnabled = impl_->multiResolutionEnabled;
        snapshotBackend.impl_->multiResolutionCoarseRatio = impl_->multiResolutionCoarseRatio;
        snapshotBackend.impl_->spatialOrientationHints = impl_->spatialOrientationHints;
        snapshotBackend.impl_->overhangBiasSpec = impl_->overhangBiasSpec;
        snapshotBackend.impl_->shellPathSpec = impl_->shellPathSpec;
        snapshotBackend.impl_->supportSpec = impl_->supportSpec;
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

rust::Vec<float> get_object_vertex_tool_axes(std::uint64_t object_id) {
    std::lock_guard<std::mutex> lock(g_backendMutex);
    const auto src = globalBackend().objectVertexToolAxes(object_id);
    rust::Vec<float> out;
    out.reserve(src.size());
    for (const float v : src) {
        out.push_back(v);
    }
    return out;
}

rust::Vec<float> get_object_vertex_support_needed(std::uint64_t object_id) {
    std::lock_guard<std::mutex> lock(g_backendMutex);
    const auto src = globalBackend().objectVertexSupportNeeded(object_id);
    rust::Vec<float> out;
    out.reserve(src.size());
    for (const float v : src) {
        out.push_back(v);
    }
    return out;
}

bool remove_cad_object(std::uint64_t object_id) {
    std::lock_guard<std::mutex> lock(g_backendMutex);
    return globalBackend().removeObject(object_id);
}

std::uint64_t clone_cad_object(std::uint64_t source_object_id, rust::Str clone_name) {
    std::lock_guard<std::mutex> lock(g_backendMutex);
    const std::string nameStr(clone_name.data(), clone_name.size());
    return globalBackend().cloneObject(source_object_id, nameStr);
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

void SlicerBackend::setOrderFieldMode(std::uint8_t mode) noexcept {
    impl_->orderFieldMode = (mode == 1) ? 1 : 0;
}

std::uint8_t SlicerBackend::orderFieldMode() const noexcept {
    return impl_->orderFieldMode;
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
    for (auto& kv : impl_->objects) {
        kv.second.hasCachedSelectedLayers = false;
    }
}

const std::vector<SpatialOrientationHint>& SlicerBackend::spatialOrientationHints() const noexcept {
    return impl_->spatialOrientationHints;
}

void SlicerBackend::setOverhangOrientationBiasSpec(const OverhangBiasSpec& spec) noexcept {
    OverhangBiasSpec s = spec;
    s.weight = std::clamp(s.weight, 0.0f, 1.0f);
    s.orthogonal_supported_angle_deg = std::clamp(s.orthogonal_supported_angle_deg, 0.0f, 89.0f);
    s.orthogonal_overhang_angle_deg = std::clamp(s.orthogonal_overhang_angle_deg, 0.0f, 89.0f);
    s.transition_width_deg = (s.transition_width_deg >= 0.5f)
        ? std::clamp(s.transition_width_deg, 1.0f, 45.0f)
        : 15.0f;
    impl_->overhangBiasSpec = s;
}

OverhangBiasSpec SlicerBackend::overhangOrientationBiasSpec() const noexcept {
    return impl_->overhangBiasSpec;
}

void SlicerBackend::setConformalSkinSpec(const ConformalSkinSpec& spec) noexcept {
    ConformalSkinSpec s = spec;
    s.default_skin_layers = (s.default_skin_layers >= 1)
        ? std::clamp<std::uint32_t>(s.default_skin_layers, 1, 12)
        : 3;
    s.min_oblique_angle_deg = (s.min_oblique_angle_deg >= 1.0f)
        ? std::clamp(s.min_oblique_angle_deg, 2.0f, 60.0f)
        : 12.0f;
    s.max_slope_angle_deg = (s.max_slope_angle_deg >= 5.0f)
        ? std::clamp(s.max_slope_angle_deg, 10.0f, 85.0f)
        : 60.0f;
    if (impl_->conformalSkinSpec.auto_enabled != s.auto_enabled ||
        impl_->conformalSkinSpec.default_skin_layers != s.default_skin_layers ||
        std::abs(impl_->conformalSkinSpec.min_oblique_angle_deg - s.min_oblique_angle_deg) > 1.0e-5f ||
        std::abs(impl_->conformalSkinSpec.max_slope_angle_deg - s.max_slope_angle_deg) > 1.0e-5f) {
        impl_->conformalSkinSpec = s;
        for (auto& kv : impl_->objects) {
            kv.second.hasCachedSelectedLayers = false;
        }
    }
}

ConformalSkinSpec SlicerBackend::conformalSkinSpec() const noexcept {
    return impl_->conformalSkinSpec;
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

    // Compute finite-difference |\nabla \tau| and exact \phi_sdf = 0 surface crossings per voxel in parallel
    const float invVoxel = 1.0f / std::max(voxelSize, 1.0e-5f);
    const std::array<openvdb::Coord, 3> axes = {
        openvdb::Coord(1, 0, 0),
        openvdb::Coord(0, 1, 0),
        openvdb::Coord(0, 0, 1),
    };

    tbb::parallel_for(
        tbb::blocked_range<std::size_t>(0, voxels.size(), 512),
        [&](const tbb::blocked_range<std::size_t>& r) {
            auto localEikAcc = eikGrid.getConstAccessor();
            auto localSdfAcc = sdfGrid.getConstAccessor();
            auto localIdxAcc = voxelIdxGrid.getConstAccessor();

            for (std::size_t i = r.begin(); i != r.end(); ++i) {
                const openvdb::Coord& c = voxels[i].ijk;
                const float t0 = voxels[i].tau;
                const float s0 = voxels[i].sdf;

                std::array<float, 3> gradTau{0.0f, 0.0f, 0.0f};
                std::array<float, 3> gradSdf{0.0f, 0.0f, 0.0f};

                for (int a = 0; a < 3; ++a) {
                    const openvdb::Coord& d = axes[static_cast<std::size_t>(a)];
                    const float tP = localEikAcc.getValue(c + d);
                    const float tM = localEikAcc.getValue(c - d);
                    const float sP = localSdfAcc.getValue(c + d);
                    const float sM = localSdfAcc.getValue(c - d);
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
                                const float sNb = localSdfAcc.getValue(nb);
                                if (sNb <= 0.0f) {
                                    continue;
                                }
                                const float tNb = localEikAcc.getValue(nb);
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
                    // Voxel center is in the thin boundary shell (0 < sdf <= 0.85 * voxelSize):
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
                                const int nbIdx = localIdxAcc.getValue(c + openvdb::Coord(dx, dy, dz));
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
            }
        });

    float globalTauMin = std::numeric_limits<float>::max();
    float globalTauMax = std::numeric_limits<float>::lowest();
    for (const auto& vx : voxels) {
        globalTauMin = std::min(globalTauMin, vx.surfTauMin);
        globalTauMax = std::max(globalTauMax, vx.surfTauMax);
    }

    // Project exact CAD mesh vertices and triangle centroids onto their containing/nearest
    // interior voxel so sharp 3D corners of the CAD mesh are captured to sub-voxel accuracy.
    auto fineSdfAccMain = fineSdfGrid.getConstAccessor();
    openvdb::tools::GridSampler<openvdb::FloatGrid::ConstAccessor, openvdb::tools::BoxSampler>
        eikSurfaceSampler(eikAccMain, eikGrid.transform());
    openvdb::tools::GridSampler<openvdb::FloatGrid::ConstAccessor, openvdb::tools::BoxSampler>
        sdfSurfaceSampler(fineSdfAccMain, fineSdfGrid.transform());

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
                    std::clamp(sdfAtPt, 0.0f, 0.75f * fineVoxelSize));
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
        std::vector<std::pair<openvdb::Coord, int>> frontier;
        frontier.reserve(voxels.size() / 4 + 64);
        for (const auto& vx : voxels) {
            if (vx.sdf >= -1.25f * voxelSize) {
                frontier.emplace_back(vx.ijk, vx.branchId);
            }
        }
        std::vector<std::pair<openvdb::Coord, int>> nextFrontier;
        for (int pass = 0; pass < 3; ++pass) {
            nextFrontier.clear();
            for (const auto& [c, bId] : frontier) {
                for (int dz = -1; dz <= 1; ++dz) {
                    for (int dy = -1; dy <= 1; ++dy) {
                        for (int dx = -1; dx <= 1; ++dx) {
                            if (dx == 0 && dy == 0 && dz == 0) {
                                continue;
                            }
                            const openvdb::Coord nb = c + openvdb::Coord(dx, dy, dz);
                            if (branchAcc.getValue(nb) < 0) {
                                nextFrontier.emplace_back(nb, bId);
                            }
                        }
                    }
                }
            }
            frontier.clear();
            for (const auto& [coord, bId] : nextFrontier) {
                if (branchAcc.getValue(coord) < 0) {
                    branchAcc.setValue(coord, bId);
                    frontier.emplace_back(coord, bId);
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

    // Precompute LeafNode [\tau_min, \tau_max] intervals and exterior-air value-mask culling for
    // `eikGrid` so that each layer's `volumeToMesh` call only visits the O(N^{2/3}) leaf nodes
    // that straddle `pl.isovalue` inside/near the object (\phi_sdf <= 2.25 * \Delta x) rather than
    // scanning the entire 3D grid and meshing the outer narrow-band closure bag in empty air.
    using EikTreeType = openvdb::FloatGrid::TreeType;
    using EikLeafType = EikTreeType::LeafNodeType;
    const float eikBgVal = eikGrid.background();
    const float activeLeafMaxSdf = 2.25f * voxelSize;
    const float nearBgMaxSdf = 0.65f * voxelSize;

    std::vector<const EikLeafType*> srcLeaves;
    srcLeaves.reserve(eikGrid.tree().leafCount());
    openvdb::Int32Grid leafIdxGrid(-1);
    auto leafIdxAcc = leafIdxGrid.getAccessor();
    for (auto leafIter = eikGrid.tree().cbeginLeaf(); leafIter; ++leafIter) {
        const auto idx = static_cast<std::int32_t>(srcLeaves.size());
        srcLeaves.push_back(leafIter.getLeaf());
        leafIdxAcc.setValue(leafIter->origin(), idx);
    }

    const std::size_t numEikLeaves = srcLeaves.size();
    std::vector<std::unique_ptr<EikLeafType>> activeMaskedLeaves(numEikLeaves);
    std::vector<std::unique_ptr<EikLeafType>> haloMaskedLeaves(numEikLeaves);
    std::vector<float> leafTauMin(numEikLeaves, std::numeric_limits<float>::max());
    std::vector<float> leafTauMax(numEikLeaves, -std::numeric_limits<float>::max());
    std::vector<std::uint8_t> leafHasActive(numEikLeaves, 0);
    std::vector<std::vector<std::uint32_t>> leafHaloNeighbors(numEikLeaves);

    tbb::parallel_for(
        tbb::blocked_range<std::size_t>(0, numEikLeaves),
        [&](const tbb::blocked_range<std::size_t>& r) {
            auto sdfLeafAcc = sdfGrid.getConstAccessor();
            auto eikLeafAcc = eikGrid.getConstAccessor();
            auto idxLeafAcc = leafIdxGrid.getConstAccessor();

            for (std::size_t lIdx = r.begin(); lIdx != r.end(); ++lIdx) {
                const EikLeafType* srcLeaf = srcLeaves[lIdx];
                auto haloCopy = std::make_unique<EikLeafType>(*srcLeaf);
                haloCopy->getValueMask().setOff();
                haloMaskedLeaves[lIdx] = std::move(haloCopy);

                auto activeCopy = std::make_unique<EikLeafType>(*srcLeaf);
                float tMin = std::numeric_limits<float>::max();
                float tMax = -std::numeric_limits<float>::max();
                bool anyOn = false;

                for (auto vIt = srcLeaf->cbeginValueOn(); vIt; ++vIt) {
                    const openvdb::Index off = vIt.pos();
                    const openvdb::Coord c = vIt.getCoord();
                    const float tau = *vIt;
                    const float sdf = sdfLeafAcc.getValue(c);

                    bool keepActive =
                        std::isfinite(tau) && (tau < validTauLimit) && (sdf <= activeLeafMaxSdf);
                    if (keepActive && sdf > nearBgMaxSdf) {
                        for (int dz = -1; dz <= 1 && keepActive; ++dz) {
                            for (int dy = -1; dy <= 1 && keepActive; ++dy) {
                                for (int dx = -1; dx <= 1; ++dx) {
                                    if (dx == 0 && dy == 0 && dz == 0) {
                                        continue;
                                    }
                                    const float nbTau =
                                        eikLeafAcc.getValue(c + openvdb::Coord(dx, dy, dz));
                                    if (!std::isfinite(nbTau) || nbTau >= validTauLimit) {
                                        keepActive = false;
                                        break;
                                    }
                                }
                            }
                        }
                    }

                    if (!keepActive) {
                        activeCopy->setValueOff(off);
                        continue;
                    }

                    anyOn = true;
                    tMin = std::min(tMin, tau);
                    tMax = std::max(tMax, tau);
                    for (int dz = -1; dz <= 1; ++dz) {
                        for (int dy = -1; dy <= 1; ++dy) {
                            for (int dx = -1; dx <= 1; ++dx) {
                                if (dx == 0 && dy == 0 && dz == 0) {
                                    continue;
                                }
                                const float nbTau =
                                    eikLeafAcc.getValue(c + openvdb::Coord(dx, dy, dz));
                                if (std::isfinite(nbTau) && nbTau < validTauLimit) {
                                    tMin = std::min(tMin, nbTau);
                                    tMax = std::max(tMax, nbTau);
                                }
                            }
                        }
                    }
                }

                // Also expand [tMin, tMax] across all 512 buffer values in `srcLeaf` so if a
                // neighbor leaf's active voxel at local coord 7 straddles into `srcLeaf` at coord 0,
                // `srcLeaf` is guaranteed to be included as an active leaf (never leaving an
                // open unclipped border at \phi_sdf ~ +2.25 * \Delta x).
                if (anyOn) {
                    const float* buf = srcLeaf->buffer().data();
                    for (openvdb::Index i = 0; i < EikLeafType::SIZE; ++i) {
                        const float vTau = buf[i];
                        if (std::isfinite(vTau) && vTau < validTauLimit) {
                            tMin = std::min(tMin, vTau);
                            tMax = std::max(tMax, vTau);
                        }
                    }
                }

                if (anyOn && tMin <= tMax) {
                    leafHasActive[lIdx] = 1;
                    leafTauMin[lIdx] = tMin - 3.5f * voxelSize;
                    leafTauMax[lIdx] = tMax + 3.5f * voxelSize;
                    activeMaskedLeaves[lIdx] = std::move(activeCopy);

                    const openvdb::Coord orig = srcLeaf->origin();
                    constexpr int kLeafDim = static_cast<int>(EikLeafType::DIM);
                    auto& nbList = leafHaloNeighbors[lIdx];
                    nbList.reserve(26);
                    for (int dz = -1; dz <= 1; ++dz) {
                        for (int dy = -1; dy <= 1; ++dy) {
                            for (int dx = -1; dx <= 1; ++dx) {
                                if (dx == 0 && dy == 0 && dz == 0) {
                                    continue;
                                }
                                const openvdb::Coord nbOrig =
                                    orig.offsetBy(dx * kLeafDim, dy * kLeafDim, dz * kLeafDim);
                                const std::int32_t nbIdx = idxLeafAcc.getValue(nbOrig);
                                if (nbIdx >= 0) {
                                    nbList.push_back(static_cast<std::uint32_t>(nbIdx));
                                }
                            }
                        }
                    }
                }
            }
        });

    // ------------------------------------------------------------------------
    // 5b. Conformal SDF Surface Skin Patch Detection & Protrusion Stand-Off
    // ------------------------------------------------------------------------
    const ConformalSkinSpec skinSpec = impl_->conformalSkinSpec;
    std::vector<SpatialOrientationHint> includeSkinHints;
    std::vector<SpatialOrientationHint> excludeSkinHints;
    for (const auto& h : impl_->spatialOrientationHints) {
        if (h.kind == 2) {
            if (h.skin_layer_count == 0 || h.weight <= 0.0f) {
                excludeSkinHints.push_back(h);
            } else {
                includeSkinHints.push_back(h);
            }
        }
    }
    const bool anyConformalSkinActive = skinSpec.auto_enabled || !includeSkinHints.empty();

    struct ConformalSkinSurfVoxel {
        openvdb::Coord ijk;
        openvdb::Vec3f surfPos{0.0f, 0.0f, 0.0f};
        openvdb::Vec3f surfNormal{0.0f, 0.0f, 1.0f};
        float tau = 0.0f;
        int requestedLayers = 0;
        float weight = 1.0f;
        bool isManual = false;
        int branchId = 0;
        int patchId = -1;
    };

    struct ConformalSkinPatch {
        int patchId = 0;
        int skinLayers = 3;
        float skinThicknessMm = 0.60f;
        float tauMin = 0.0f;
        float tauCompletion = 0.0f;
        std::uint32_t branchId = 0;
        std::int32_t parentBranchId = -1;
        openvdb::Vec3f meanNormal{0.0f, 0.0f, 1.0f};
        std::vector<std::size_t> surfVoxelIndices;
    };

    std::vector<ConformalSkinPatch> conformalSkinPatches;
    openvdb::FloatGrid skinErodeDepthGrid(0.0f);
    skinErodeDepthGrid.setTransform(fineSdfGrid.transform().copy());
    openvdb::Int32Grid skinPatchIdGrid(-1);
    skinPatchIdGrid.setTransform(fineSdfGrid.transform().copy());
    openvdb::Vec3SGrid skinPatchNormalGrid(openvdb::Vec3s(0.0f, 0.0f, 1.0f));
    skinPatchNormalGrid.setTransform(fineSdfGrid.transform().copy());
    float maxConformalSkinDepthMm = 0.0f;

    if (anyConformalSkinActive) {
        constexpr float kDegToRad = 3.14159265358979323846f / 180.0f;
        const float cosMaxSlope =
            std::cos(std::clamp(skinSpec.max_slope_angle_deg, 10.0f, 85.0f) * kDegToRad);
        const float cosMinOblique =
            std::cos(std::clamp(skinSpec.min_oblique_angle_deg, 2.0f, 60.0f) * kDegToRad);

        const ToolSpec* activeToolPtr = nullptr;
        float tipRadiusMm = 0.25f;
        float maxToolClearanceHeightMm = 20.0f;
        if (!impl_->tools.empty()) {
            const auto& tool = impl_->tools[std::min<std::size_t>(
                impl_->activeToolIndex, impl_->tools.size() - 1)];
            activeToolPtr = &tool;
            tipRadiusMm = 0.5f * std::max(0.2f, tool.nozzle_diameter_mm);
            if (tool.clearance_mode == 1) {
                tipRadiusMm = std::max(tipRadiusMm, tool.cone_tip_radius_mm);
                maxToolClearanceHeightMm = std::max(4.0f, tool.cone_height_mm);
            } else if (tool.clearance_mode == 2) {
                tipRadiusMm = std::max({
                    tipRadiusMm,
                    tool.rect_tip_half_width_x_mm,
                    tool.rect_tip_half_width_y_mm});
                maxToolClearanceHeightMm = std::max(4.0f, tool.rect_height_mm);
            } else if (!tool.clearance_profile_xz.empty()) {
                tipRadiusMm = std::max(tipRadiusMm, tool.clearance_profile_xz.front().x_radius_mm);
                maxToolClearanceHeightMm = std::max(
                    4.0f, tool.clearance_profile_xz.back().z_height_mm);
            }
        }
        const float wallHalfWidthMm = 0.5f * impl_->shellPathSpec.outer_wall_width.target_mm;

        // Evaluates the active tool's conservative radial clearance envelope R_tool(z)
        // in the nozzle's local frame at height `zLocalMm >= 0` along `nozzle_axis`
        // (AGENTS.md Rule 1 & Rule 3).
        auto evalToolClearanceRadiusAtZ = [&](float zLocalMm) -> float {
            const float z = std::max(0.0f, zLocalMm);
            float rTool = tipRadiusMm;
            if (activeToolPtr != nullptr) {
                const auto& tool = *activeToolPtr;
                if (tool.clearance_mode == 1) {
                    const float slope = std::tan(
                        std::clamp(tool.cone_half_angle_deg, 1.0f, 85.0f) * kDegToRad);
                    rTool = std::max(rTool, tool.cone_tip_radius_mm + z * slope);
                } else if (tool.clearance_mode == 2) {
                    const float slopeX = std::tan(
                        std::clamp(tool.rect_half_angle_x_deg, 1.0f, 85.0f) * kDegToRad);
                    const float slopeY = std::tan(
                        std::clamp(tool.rect_half_angle_y_deg, 1.0f, 85.0f) * kDegToRad);
                    const float hx = tool.rect_tip_half_width_x_mm + z * slopeX;
                    const float hy = tool.rect_tip_half_width_y_mm + z * slopeY;
                    // Conservative circumscribed radius for arbitrary tool travel azimuth
                    rTool = std::max(rTool, std::hypot(hx, hy));
                } else if (!tool.clearance_profile_xz.empty()) {
                    const auto& prof = tool.clearance_profile_xz;
                    if (z <= prof.front().z_height_mm) {
                        rTool = std::max(rTool, prof.front().x_radius_mm);
                    } else if (z >= prof.back().z_height_mm) {
                        const std::size_t n = prof.size();
                        if (n >= 2) {
                            const float dz =
                                prof[n - 1].z_height_mm - prof[n - 2].z_height_mm;
                            const float dr =
                                prof[n - 1].x_radius_mm - prof[n - 2].x_radius_mm;
                            const float slope =
                                (dz > 1.0e-4f) ? std::max(0.0f, dr / dz) : 0.0f;
                            rTool = std::max(
                                rTool,
                                prof.back().x_radius_mm + (z - prof.back().z_height_mm) * slope);
                        } else {
                            rTool = std::max(rTool, prof.back().x_radius_mm);
                        }
                    } else {
                        for (std::size_t k = 0; k + 1 < prof.size(); ++k) {
                            const float z0 = prof[k].z_height_mm;
                            const float z1 = prof[k + 1].z_height_mm;
                            if (z >= z0 && z <= z1) {
                                const float t =
                                    (z1 - z0 > 1.0e-5f) ? ((z - z0) / (z1 - z0)) : 0.0f;
                                const float rk =
                                    prof[k].x_radius_mm +
                                    t * (prof[k + 1].x_radius_mm - prof[k].x_radius_mm);
                                rTool = std::max(rTool, rk);
                                break;
                            }
                        }
                    }
                } else {
                    rTool = std::max(rTool, tipRadiusMm + 0.65f * z);
                }
            } else {
                rTool = std::max(rTool, tipRadiusMm + 0.65f * z);
            }
            return rTool + wallHalfWidthMm + 0.35f * fineVoxelSize;
        };

        auto fineSdfAcc = fineSdfGrid.getConstAccessor();
        openvdb::tools::GridSampler<openvdb::FloatGrid::ConstAccessor, openvdb::tools::BoxSampler>
            fineSdfSampler(fineSdfAcc, fineSdfGrid.transform());

        const float maxPlausibleTau =
            std::min(validTauLimit, globalTauMax + 4.0f * voxelSize);
        const float fineBgLimit =
            fineSdfGrid.background() - std::max(0.5f * fineVoxelSize, 0.25f);

        // Build a flat 3D coarse brick acceleration grid over the object's bounding box storing
        // the minimum \tau of any solid voxel within each brick (dilated by 1 brick so boundary
        // interpolation stencils are conservatively covered). This rejects empty-air and
        // unprinted-future points in O(1) L1/L2 cache reads without touching OpenVDB's RootNode map.
        const float brickSizeMm = std::max(2.0f * voxelSize, 1.0f);
        const float invBrickSize = 1.0f / brickSizeMm;
        const float brickPadMm = 2.0f * voxelSize;
        const float brickOriginX = rec.minX - brickPadMm;
        const float brickOriginY = rec.minY - brickPadMm;
        const float brickOriginZ = rec.minZ - brickPadMm;
        const int brickNx = std::max(
            1,
            static_cast<int>(std::ceil((rec.maxX - rec.minX + 2.0f * brickPadMm) * invBrickSize)) + 1);
        const int brickNy = std::max(
            1,
            static_cast<int>(std::ceil((rec.maxY - rec.minY + 2.0f * brickPadMm) * invBrickSize)) + 1);
        const int brickNz = std::max(
            1,
            static_cast<int>(std::ceil((rec.maxZ - rec.minZ + 2.0f * brickPadMm) * invBrickSize)) + 1);
        const std::size_t brickStrideY = static_cast<std::size_t>(brickNx);
        const std::size_t brickStrideZ =
            static_cast<std::size_t>(brickNx) * static_cast<std::size_t>(brickNy);
        std::vector<float> brickMinTau(
            brickStrideZ * static_cast<std::size_t>(brickNz),
            std::numeric_limits<float>::max());

        for (const auto& vx : voxels) {
            if (vx.sdf > 0.35f * voxelSize) {
                continue;
            }
            const openvdb::Vec3d wp = eikGrid.transform().indexToWorld(vx.ijk);
            const int bx0 = static_cast<int>(std::floor((static_cast<float>(wp.x()) - brickOriginX) * invBrickSize));
            const int by0 = static_cast<int>(std::floor((static_cast<float>(wp.y()) - brickOriginY) * invBrickSize));
            const int bz0 = static_cast<int>(std::floor((static_cast<float>(wp.z()) - brickOriginZ) * invBrickSize));
            const float tVal = std::max(0.0f, std::min(vx.tau, vx.surfTauMin));
            for (int dz = -1; dz <= 1; ++dz) {
                const int bz = bz0 + dz;
                if (bz < 0 || bz >= brickNz) {
                    continue;
                }
                for (int dy = -1; dy <= 1; ++dy) {
                    const int by = by0 + dy;
                    if (by < 0 || by >= brickNy) {
                        continue;
                    }
                    for (int dx = -1; dx <= 1; ++dx) {
                        const int bx = bx0 + dx;
                        if (bx < 0 || bx >= brickNx) {
                            continue;
                        }
                        const std::size_t bIdx =
                            static_cast<std::size_t>(bz) * brickStrideZ +
                            static_cast<std::size_t>(by) * brickStrideY +
                            static_cast<std::size_t>(bx);
                        brickMinTau[bIdx] = std::min(brickMinTau[bIdx], tVal);
                    }
                }
            }
        }

        const float solidBBoxMinX = rec.minX - 0.75f * voxelSize;
        const float solidBBoxMaxX = rec.maxX + 0.75f * voxelSize;
        const float solidBBoxMinY = rec.minY - 0.75f * voxelSize;
        const float solidBBoxMaxY = rec.maxY + 0.75f * voxelSize;
        const float solidBBoxMinZ = rec.minZ - 0.75f * voxelSize;
        const float solidBBoxMaxZ = rec.maxZ + 0.75f * voxelSize;

        auto sampleAnySdfWith = [&](
            const openvdb::Vec3d& wp,
            const openvdb::FloatGrid::ConstAccessor& localSdfAcc,
            const openvdb::tools::GridSampler<
                openvdb::FloatGrid::ConstAccessor,
                openvdb::tools::BoxSampler>& localSdfSampler,
            const openvdb::tools::GridSampler<
                openvdb::FloatGrid::ConstAccessor,
                openvdb::tools::BoxSampler>& localFineSdfSampler) -> float {
            // Fast cell-centered pre-check on `sdfGrid` to avoid 8-corner trilinear BoxSampler
            // probes when the query point is clearly outside the solid boundary.
            const openvdb::Coord cMain = sdfGrid.transform().worldToIndexCellCentered(wp);
            const float cCellVal = localSdfAcc.getValue(cMain);
            if (cCellVal > 0.90f * voxelSize) {
                return cCellVal;
            }
            const float fVal = localFineSdfSampler.wsSample(wp);
            if (std::isfinite(fVal) && std::abs(fVal) < fineBgLimit) {
                return fVal;
            }
            return localSdfSampler.wsSample(wp);
        };

        auto sampleTauAtWith = [&](
            const openvdb::Vec3d& wp,
            const openvdb::FloatGrid::ConstAccessor& localFineEikAcc,
            const openvdb::tools::GridSampler<
                openvdb::FloatGrid::ConstAccessor,
                openvdb::tools::BoxSampler>& localFineEikSampler,
            const openvdb::FloatGrid::ConstAccessor& localMainEikAcc,
            const openvdb::tools::GridSampler<
                openvdb::FloatGrid::ConstAccessor,
                openvdb::tools::BoxSampler>& localMainEikSampler) -> float {
            const float fTau = localFineEikSampler.wsSample(wp);
            if (std::isfinite(fTau) && fTau >= 0.0f && fTau <= maxPlausibleTau) {
                return fTau;
            }
            const float cTau = localMainEikSampler.wsSample(wp);
            if (std::isfinite(cTau) && cTau >= 0.0f && cTau <= maxPlausibleTau) {
                return cTau;
            }
            // Cell-centered neighborhood fallback when a trilinear stencil corner touches background
            const openvdb::Coord cFine =
                fineEikGrid.transform().worldToIndexCellCentered(wp);
            float bestTau = std::numeric_limits<float>::max();
            int bestDistSq = 100;
            for (int dz = -1; dz <= 1; ++dz) {
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        const int d2 = dx * dx + dy * dy + dz * dz;
                        if (d2 > bestDistSq) {
                            continue;
                        }
                        const float v =
                            localFineEikAcc.getValue(cFine + openvdb::Coord(dx, dy, dz));
                        if (std::isfinite(v) && v >= 0.0f && v <= maxPlausibleTau) {
                            if (d2 < bestDistSq || v < bestTau) {
                                bestDistSq = d2;
                                bestTau = v;
                            }
                        }
                    }
                }
            }
            if (bestTau <= maxPlausibleTau) {
                return bestTau;
            }
            const openvdb::Coord cMain =
                eikGrid.transform().worldToIndexCellCentered(wp);
            for (int dz = -1; dz <= 1; ++dz) {
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        const float v =
                            localMainEikAcc.getValue(cMain + openvdb::Coord(dx, dy, dz));
                        if (std::isfinite(v) && v >= 0.0f && v <= maxPlausibleTau) {
                            bestTau = std::min(bestTau, v);
                        }
                    }
                }
            }
            return (bestTau <= maxPlausibleTau) ? bestTau : 0.0f;
        };

        // Checks whether placing the nozzle at the innermost conformal skin layer
        // (`xBase = xSurf - skinDepthMm * nSdf`) to print `xSurf` at `printTau` causes
        // the 3D Tool Clearance Envelope (in the local `NozzleBasis` queried via
        // `queryNozzleFrame`, AGENTS.md Rule 1) to collide with:
        //   1. Any local protrusion rising above the surface tangent plane near the nozzle tip, or
        //   2. Any already-printed solid geometry (`\phi_sdf(y) <= 0` with `\tau(y) <= printTau`)
        //      such as the Benchy cabin or raised port/starboard hull gunwale walls.
        auto hasNozzleClearanceCollision = [&](
            const openvdb::Vec3f& xSurf,
            const openvdb::Vec3f& nSdf,
            float skinDepthMm,
            float printTau,
            const openvdb::FloatGrid::ConstAccessor& localSdfAcc,
            const openvdb::tools::GridSampler<
                openvdb::FloatGrid::ConstAccessor,
                openvdb::tools::BoxSampler>& localSdfSampler,
            const openvdb::tools::GridSampler<
                openvdb::FloatGrid::ConstAccessor,
                openvdb::tools::BoxSampler>& localFineSdfSampler,
            const openvdb::FloatGrid::ConstAccessor& localFineEikAcc,
            const openvdb::tools::GridSampler<
                openvdb::FloatGrid::ConstAccessor,
                openvdb::tools::BoxSampler>& localFineEikSampler,
            const openvdb::FloatGrid::ConstAccessor& localMainEikAcc,
            const openvdb::tools::GridSampler<
                openvdb::FloatGrid::ConstAccessor,
                openvdb::tools::BoxSampler>& localMainEikSampler,
            bool* outNearBoundaryMargin = nullptr) -> bool {
            if (outNearBoundaryMargin != nullptr) {
                *outNearBoundaryMargin = false;
            }
            const NozzleBasis nb = queryNozzleFrame(
                {xSurf.x(), xSurf.y(), xSurf.z()},
                {1.0f, 0.0f, 0.0f});
            const openvdb::Vec3f zAxis(
                nb.nozzle_axis[0], nb.nozzle_axis[1], nb.nozzle_axis[2]);
            const openvdb::Vec3f xAxis(
                nb.travel_axis[0], nb.travel_axis[1], nb.travel_axis[2]);
            const openvdb::Vec3f yAxis(
                nb.lateral_axis[0], nb.lateral_axis[1], nb.lateral_axis[2]);

            // Innermost conformal skin layer position where the nozzle reaches deepest
            const openvdb::Vec3f xBase = xSurf - skinDepthMm * nSdf;

            // Compute maximum axial height along +zAxis until exiting the object bounding box
            const openvdb::Vec3f corners[8] = {
                {solidBBoxMinX, solidBBoxMinY, solidBBoxMinZ},
                {solidBBoxMaxX, solidBBoxMinY, solidBBoxMinZ},
                {solidBBoxMinX, solidBBoxMaxY, solidBBoxMinZ},
                {solidBBoxMaxX, solidBBoxMaxY, solidBBoxMinZ},
                {solidBBoxMinX, solidBBoxMinY, solidBBoxMaxZ},
                {solidBBoxMaxX, solidBBoxMinY, solidBBoxMaxZ},
                {solidBBoxMinX, solidBBoxMaxY, solidBBoxMaxZ},
                {solidBBoxMaxX, solidBBoxMaxY, solidBBoxMaxZ},
            };
            float maxZAlongNozzle = 0.0f;
            for (const auto& c : corners) {
                maxZAlongNozzle = std::max(maxZAlongNozzle, (c - xBase).dot(zAxis));
            }
            const float zSweepMax = std::min(
                maxZAlongNozzle,
                std::min(maxToolClearanceHeightMm, 12.0f));
            const float zStart = std::max(0.60f * hTop, 0.50f * fineVoxelSize);
            if (zSweepMax <= zStart) {
                return false;
            }

            constexpr float kDirs12[12][2] = {
                {1.0f, 0.0f},
                {0.8660254f, 0.5f},
                {0.5f, 0.8660254f},
                {0.0f, 1.0f},
                {-0.5f, 0.8660254f},
                {-0.8660254f, 0.5f},
                {-1.0f, 0.0f},
                {-0.8660254f, -0.5f},
                {-0.5f, -0.8660254f},
                {0.0f, -1.0f},
                {0.5f, -0.8660254f},
                {0.8660254f, -0.5f},
            };

            const float radialStep = std::max({0.75f * fineVoxelSize, 0.50f * voxelSize, 0.45f});
            const float nearTipLimit = skinDepthMm + std::max(1.75f * hTop, 1.35f * fineVoxelSize);
            const float printTauTol = printTau + 0.65f * hTop;
            const float baseCurvTol = std::max(0.45f * hTop, 0.35f * fineVoxelSize);

            auto isBrickPossiblyOccupiedAt = [&](
                const openvdb::Vec3f& pt,
                float maxAllowedTau) -> bool {
                if (pt.x() < solidBBoxMinX || pt.x() > solidBBoxMaxX ||
                    pt.y() < solidBBoxMinY || pt.y() > solidBBoxMaxY ||
                    pt.z() < solidBBoxMinZ || pt.z() > solidBBoxMaxZ) {
                    return false;
                }
                const int bx = static_cast<int>(std::floor((pt.x() - brickOriginX) * invBrickSize));
                const int by = static_cast<int>(std::floor((pt.y() - brickOriginY) * invBrickSize));
                const int bz = static_cast<int>(std::floor((pt.z() - brickOriginZ) * invBrickSize));
                if (bx < 0 || bx >= brickNx ||
                    by < 0 || by >= brickNy ||
                    bz < 0 || bz >= brickNz) {
                    return false;
                }
                const std::size_t bIdx =
                    static_cast<std::size_t>(bz) * brickStrideZ +
                    static_cast<std::size_t>(by) * brickStrideY +
                    static_cast<std::size_t>(bx);
                return brickMinTau[bIdx] <= maxAllowedTau;
            };

            for (float z = zStart; z <= zSweepMax;) {
                const float rClear = evalToolClearanceRadiusAtZ(z);
                const float rMargin =
                    (outNearBoundaryMargin != nullptr && z <= nearTipLimit + 2.0f * hTop)
                        ? (rClear + std::max(0.65f * fineVoxelSize, 0.85f * hTop))
                        : rClear;
                const openvdb::Vec3f pAxis = xBase + z * zAxis;
                const float sliceTauLimit =
                    (z <= nearTipLimit) ? maxPlausibleTau : printTauTol;

                const float zStep = (z < nearTipLimit + 2.0f * hTop)
                    ? std::max(0.55f * fineVoxelSize, 0.65f * hTop)
                    : std::max(0.95f * fineVoxelSize, 0.65f);

                // Quick disk AABB overlap check against the solid object bounding box
                if (pAxis.x() + rMargin < solidBBoxMinX || pAxis.x() - rMargin > solidBBoxMaxX ||
                    pAxis.y() + rMargin < solidBBoxMinY || pAxis.y() - rMargin > solidBBoxMaxY ||
                    pAxis.z() + rMargin < solidBBoxMinZ || pAxis.z() - rMargin > solidBBoxMaxZ) {
                    z += zStep;
                    continue;
                }

                // Check centerline point `pAxis` when above the outer surface
                if (z > skinDepthMm + 0.55f * hTop) {
                    const float hAboveSurfAxis = (pAxis - xSurf).dot(nSdf);
                    if (hAboveSurfAxis > baseCurvTol &&
                        isBrickPossiblyOccupiedAt(pAxis, sliceTauLimit)) {
                        const openvdb::Vec3d wpAxis(pAxis.x(), pAxis.y(), pAxis.z());
                        const float axisSdf = sampleAnySdfWith(
                            wpAxis, localSdfAcc, localSdfSampler, localFineSdfSampler);
                        if (axisSdf <= 0.08f * fineVoxelSize) {
                            if (z <= nearTipLimit) {
                                return true;
                            }
                            const float yTau = sampleTauAtWith(
                                wpAxis,
                                localFineEikAcc,
                                localFineEikSampler,
                                localMainEikAcc,
                                localMainEikSampler);
                            if (yTau <= printTauTol) {
                                return true;
                            }
                        }
                    }
                }

                const int numRings = std::max(1, static_cast<int>(std::ceil(rMargin / radialStep)));
                for (int ri = 1; ri <= numRings; ++ri) {
                    const float r =
                        (static_cast<float>(ri) / static_cast<float>(numRings)) * rMargin;
                    const bool isMarginRing = (r > rClear + 1.0e-4f);
                    if (isMarginRing &&
                        (outNearBoundaryMargin == nullptr || *outNearBoundaryMargin)) {
                        continue;
                    }
                    // Allow slight natural dome convexity without false self-collision
                    const float curvTol = baseCurvTol + 0.03f * r;
                    const bool checkSteepCore = (z > skinDepthMm + 0.75f * hTop);

                    for (const auto& dir : kDirs12) {
                        const openvdb::Vec3f rVec = xAxis * dir[0] + yAxis * dir[1];
                        const openvdb::Vec3f yPos = pAxis + r * rVec;

                        // 1. Cheap geometric plane filter FIRST (3 flops)
                        const float hAboveSurf = (yPos - xSurf).dot(nSdf);
                        const float hAboveBase = hAboveSurf + skinDepthMm;
                        const bool aboveSurf = (hAboveSurf > curvTol);
                        const bool aboveBase = (checkSteepCore && hAboveBase > curvTol);
                        if (!aboveSurf && !aboveBase) {
                            continue;
                        }

                        // 2. O(1) flat 3D brick occupancy + \tau filter (zero OpenVDB queries)
                        const float maxAllowedTau =
                            (aboveSurf && z <= nearTipLimit) ? maxPlausibleTau : printTauTol;
                        if (!isBrickPossiblyOccupiedAt(yPos, maxAllowedTau)) {
                            continue;
                        }

                        // 3. Exact SDF & Eikonal \tau check inside active brick
                        const openvdb::Vec3d wpY(yPos.x(), yPos.y(), yPos.z());
                        const float ySdf = sampleAnySdfWith(
                            wpY, localSdfAcc, localSdfSampler, localFineSdfSampler);
                        if (ySdf > 0.08f * fineVoxelSize) {
                            continue;
                        }

                        bool hitsObstacle = false;
                        if (aboveSurf) {
                            if (z <= nearTipLimit) {
                                hitsObstacle = true;
                            } else {
                                const float yTau = sampleTauAtWith(
                                    wpY,
                                    localFineEikAcc,
                                    localFineEikSampler,
                                    localMainEikAcc,
                                    localMainEikSampler);
                                if (yTau <= printTauTol) {
                                    hitsObstacle = true;
                                }
                            }
                        } else if (aboveBase && ySdf <= -(skinDepthMm + 0.35f * hTop)) {
                            const float yTau = sampleTauAtWith(
                                wpY,
                                localFineEikAcc,
                                localFineEikSampler,
                                localMainEikAcc,
                                localMainEikSampler);
                            if (yTau <= printTauTol) {
                                hitsObstacle = true;
                            }
                        }

                        if (hitsObstacle) {
                            if (!isMarginRing) {
                                return true;
                            }
                            if (outNearBoundaryMargin != nullptr) {
                                *outNearBoundaryMargin = true;
                            }
                        }
                    }
                }

                z += zStep;
            }

            return false;
        };

        // Collect active surface-band voxels from `fineSdfGrid`
        std::vector<std::pair<openvdb::Coord, float>> rawSurfBand;
        rawSurfBand.reserve(voxels.size());
        for (auto iter = fineSdfGrid.cbeginValueOn(); iter; ++iter) {
            const float sdfVal = *iter;
            if (sdfVal >= -1.15f * fineVoxelSize && sdfVal <= 0.65f * fineVoxelSize) {
                rawSurfBand.emplace_back(iter.getCoord(), sdfVal);
            }
        }

        // Parallel candidate evaluation + Stage-1 3D nozzle clearance check across all cores
        std::vector<ConformalSkinSurfVoxel> candidateSlots(rawSurfBand.size());
        std::vector<std::uint8_t> candidateValid(rawSurfBand.size(), 0);

        tbb::parallel_for(
            tbb::blocked_range<std::size_t>(0, rawSurfBand.size(), 256),
            [&](const tbb::blocked_range<std::size_t>& r) {
                auto localSdfAcc = sdfGrid.getConstAccessor();
                openvdb::tools::GridSampler<
                    openvdb::FloatGrid::ConstAccessor,
                    openvdb::tools::BoxSampler>
                    localSdfSampler(localSdfAcc, sdfGrid.transform());
                auto localFineSdfAcc = fineSdfGrid.getConstAccessor();
                openvdb::tools::GridSampler<
                    openvdb::FloatGrid::ConstAccessor,
                    openvdb::tools::BoxSampler>
                    localFineSdfSampler(localFineSdfAcc, fineSdfGrid.transform());
                auto localFineEikAcc = fineEikGrid.getConstAccessor();
                openvdb::tools::GridSampler<
                    openvdb::FloatGrid::ConstAccessor,
                    openvdb::tools::BoxSampler>
                    localFineEikSampler(localFineEikAcc, fineEikGrid.transform());
                auto localMainEikAcc = eikGrid.getConstAccessor();
                openvdb::tools::GridSampler<
                    openvdb::FloatGrid::ConstAccessor,
                    openvdb::tools::BoxSampler>
                    localMainEikSampler(localMainEikAcc, eikGrid.transform());
                auto localBAcc = branchGrid.getConstAccessor();

                auto sampleLocalSdfGradUnit = [&](
                    const openvdb::Vec3d& wp,
                    openvdb::Vec3f& outNormal) -> bool {
                    const double h = 0.5 * static_cast<double>(fineVoxelSize);
                    const float gx =
                        localFineSdfSampler.wsSample(wp + openvdb::Vec3d(h, 0.0, 0.0)) -
                        localFineSdfSampler.wsSample(wp - openvdb::Vec3d(h, 0.0, 0.0));
                    const float gy =
                        localFineSdfSampler.wsSample(wp + openvdb::Vec3d(0.0, h, 0.0)) -
                        localFineSdfSampler.wsSample(wp - openvdb::Vec3d(0.0, h, 0.0));
                    const float gz =
                        localFineSdfSampler.wsSample(wp + openvdb::Vec3d(0.0, 0.0, h)) -
                        localFineSdfSampler.wsSample(wp - openvdb::Vec3d(0.0, 0.0, h));
                    const float len = std::sqrt(gx * gx + gy * gy + gz * gz);
                    if (len < 1.0e-5f) {
                        return false;
                    }
                    outNormal = openvdb::Vec3f(gx / len, gy / len, gz / len);
                    return true;
                };

                for (std::size_t idx = r.begin(); idx != r.end(); ++idx) {
                    const openvdb::Coord ijk = rawSurfBand[idx].first;
                    const float sdfVal = rawSurfBand[idx].second;
                    const openvdb::Vec3d wp = fineSdfGrid.transform().indexToWorld(ijk);
                    openvdb::Vec3f nSdf(0.0f, 0.0f, 1.0f);
                    if (!sampleLocalSdfGradUnit(wp, nSdf)) {
                        continue;
                    }

                    const openvdb::Vec3f xSurf(
                        static_cast<float>(wp.x()) - sdfVal * nSdf.x(),
                        static_cast<float>(wp.y()) - sdfVal * nSdf.y(),
                        static_cast<float>(wp.z()) - sdfVal * nSdf.z());

                    // Query local 5/6-axis nozzle frame in bed coordinates (AGENTS.md Rule 1)
                    const NozzleBasis nozzleBasis = queryNozzleFrame(
                        {xSurf.x(), xSurf.y(), xSurf.z()},
                        {1.0f, 0.0f, 0.0f});
                    const auto& zNozzle = nozzleBasis.nozzle_axis;
                    const float cosNozzle =
                        nSdf.x() * zNozzle[0] + nSdf.y() * zNozzle[1] + nSdf.z() * zNozzle[2];
                    const float relNozzleZ =
                        (xSurf.x() - rec.minX) * zNozzle[0] +
                        (xSurf.y() - rec.minY) * zNozzle[1] +
                        (xSurf.z() - rec.minZ) * zNozzle[2];
                    if (relNozzleZ <= 1.75f * hFirst && cosNozzle < 0.25f) {
                        continue;
                    }

                    // Check manual exclusion hints (`skin_layer_count == 0`)
                    bool excludedByHint = false;
                    for (const auto& eh : excludeSkinHints) {
                        const float dx = xSurf.x() - eh.position_bed[0];
                        const float dy = xSurf.y() - eh.position_bed[1];
                        const float dz = xSurf.z() - eh.position_bed[2];
                        const float rad = std::max(eh.influence_radius_mm, 1.5f * fineVoxelSize);
                        const float distSq = dx * dx + dy * dy + dz * dz;
                        const float normalDist =
                            dx * eh.direction_bed[0] +
                            dy * eh.direction_bed[1] +
                            dz * eh.direction_bed[2];
                        const float maxNormalOffset = std::max(2.5f * fineVoxelSize, 0.25f * rad);
                        const float dotDir =
                            nSdf.x() * eh.direction_bed[0] +
                            nSdf.y() * eh.direction_bed[1] +
                            nSdf.z() * eh.direction_bed[2];
                        if (distSq <= rad * rad &&
                            std::abs(normalDist) <= maxNormalOffset &&
                            dotDir >= 0.80f) {
                            excludedByHint = true;
                            break;
                        }
                    }
                    if (excludedByHint) {
                        continue;
                    }

                    int requestedLayers = 0;
                    bool isManual = false;
                    float bestManualScore = -1.0f;
                    for (const auto& ih : includeSkinHints) {
                        const float dx = xSurf.x() - ih.position_bed[0];
                        const float dy = xSurf.y() - ih.position_bed[1];
                        const float dz = xSurf.z() - ih.position_bed[2];
                        const float rad = std::max(ih.influence_radius_mm, 1.5f * fineVoxelSize);
                        const float distSq = dx * dx + dy * dy + dz * dz;
                        const float normalDist =
                            dx * ih.direction_bed[0] +
                            dy * ih.direction_bed[1] +
                            dz * ih.direction_bed[2];
                        const float maxNormalOffset = std::max(2.5f * fineVoxelSize, 0.25f * rad);
                        const float dotDir =
                            nSdf.x() * ih.direction_bed[0] +
                            nSdf.y() * ih.direction_bed[1] +
                            nSdf.z() * ih.direction_bed[2];
                        if (distSq <= rad * rad &&
                            std::abs(normalDist) <= maxNormalOffset &&
                            dotDir >= 0.80f) {
                            const float score = ih.weight * dotDir / (1.0f + distSq / (rad * rad));
                            if (score > bestManualScore) {
                                bestManualScore = score;
                                requestedLayers = std::clamp<int>(
                                    static_cast<int>(ih.skin_layer_count), 1, 12);
                                isManual = true;
                            }
                        }
                    }

                    if (!isManual && skinSpec.auto_enabled) {
                        if (cosNozzle < cosMaxSlope) {
                            continue;
                        }
                        const float inStep = 0.85f * fineVoxelSize;
                        const openvdb::Vec3d wpIn(
                            xSurf.x() - inStep * nSdf.x(),
                            xSurf.y() - inStep * nSdf.y(),
                            xSurf.z() - inStep * nSdf.z());
                        const double hEik = 0.5 * static_cast<double>(fineVoxelSize);
                        const float tx =
                            sampleTauAtWith(
                                wpIn + openvdb::Vec3d(hEik, 0.0, 0.0),
                                localFineEikAcc,
                                localFineEikSampler,
                                localMainEikAcc,
                                localMainEikSampler) -
                            sampleTauAtWith(
                                wpIn - openvdb::Vec3d(hEik, 0.0, 0.0),
                                localFineEikAcc,
                                localFineEikSampler,
                                localMainEikAcc,
                                localMainEikSampler);
                        const float ty =
                            sampleTauAtWith(
                                wpIn + openvdb::Vec3d(0.0, hEik, 0.0),
                                localFineEikAcc,
                                localFineEikSampler,
                                localMainEikAcc,
                                localMainEikSampler) -
                            sampleTauAtWith(
                                wpIn - openvdb::Vec3d(0.0, hEik, 0.0),
                                localFineEikAcc,
                                localFineEikSampler,
                                localMainEikAcc,
                                localMainEikSampler);
                        const float tz =
                            sampleTauAtWith(
                                wpIn + openvdb::Vec3d(0.0, 0.0, hEik),
                                localFineEikAcc,
                                localFineEikSampler,
                                localMainEikAcc,
                                localMainEikSampler) -
                            sampleTauAtWith(
                                wpIn - openvdb::Vec3d(0.0, 0.0, hEik),
                                localFineEikAcc,
                                localFineEikSampler,
                                localMainEikAcc,
                                localMainEikSampler);
                        const float tLen = std::sqrt(tx * tx + ty * ty + tz * tz);
                        if (tLen < 1.0e-4f) {
                            continue;
                        }
                        const float cosOblique =
                            (tx * nSdf.x() + ty * nSdf.y() + tz * nSdf.z()) / tLen;
                        if (cosOblique >= 0.087f && cosOblique <= cosMinOblique) {
                            requestedLayers = std::clamp<int>(
                                static_cast<int>(skinSpec.default_skin_layers), 1, 12);
                        }
                    }

                    if (requestedLayers <= 0) {
                        continue;
                    }

                    // Ensure sufficient interior solid thickness beneath xSurf along -nSdf
                    while (requestedLayers > 1) {
                        const float probeDepth =
                            static_cast<float>(requestedLayers) * hTop + 0.65f * hMin;
                        const openvdb::Vec3d wpDeep(
                            xSurf.x() - probeDepth * nSdf.x(),
                            xSurf.y() - probeDepth * nSdf.y(),
                            xSurf.z() - probeDepth * nSdf.z());
                        if (sampleAnySdfWith(
                                wpDeep, localSdfAcc, localSdfSampler, localFineSdfSampler) <=
                            -0.20f * hMin) {
                            break;
                        }
                        --requestedLayers;
                    }
                    {
                        const float minProbeDepth = hTop + 0.45f * hMin;
                        const openvdb::Vec3d wpMinDeep(
                            xSurf.x() - minProbeDepth * nSdf.x(),
                            xSurf.y() - minProbeDepth * nSdf.y(),
                            xSurf.z() - minProbeDepth * nSdf.z());
                        if (sampleAnySdfWith(
                                wpMinDeep, localSdfAcc, localSdfSampler, localFineSdfSampler) >
                            0.0f) {
                            continue;
                        }
                    }

                    // Ensure free-air overhead clearance along +nSdf
                    const float skinDepthMm = static_cast<float>(requestedLayers) * hTop;
                    const float overheadProbeDist = std::max(skinDepthMm, 1.25f * fineVoxelSize);
                    const openvdb::Vec3d wpOverhead(
                        xSurf.x() + overheadProbeDist * nSdf.x(),
                        xSurf.y() + overheadProbeDist * nSdf.y(),
                        xSurf.z() + overheadProbeDist * nSdf.z());
                    if (sampleAnySdfWith(
                            wpOverhead, localSdfAcc, localSdfSampler, localFineSdfSampler) <=
                        0.15f * fineVoxelSize) {
                        continue;
                    }

                    const openvdb::Vec3d wpTauSample(
                        xSurf.x() - 0.60f * fineVoxelSize * nSdf.x(),
                        xSurf.y() - 0.60f * fineVoxelSize * nSdf.y(),
                        xSurf.z() - 0.60f * fineVoxelSize * nSdf.z());
                    const float surfTau = sampleTauAtWith(
                        wpTauSample,
                        localFineEikAcc,
                        localFineEikSampler,
                        localMainEikAcc,
                        localMainEikSampler);

                    const openvdb::Coord cBranch =
                        branchGrid.transform().worldToIndexCellCentered(wpTauSample);
                    int vBranch = localBAcc.getValue(cBranch);
                    if (vBranch < 0) {
                        for (int dz = -1; dz <= 1 && vBranch < 0; ++dz) {
                            for (int dy = -1; dy <= 1 && vBranch < 0; ++dy) {
                                for (int dx = -1; dx <= 1 && vBranch < 0; ++dx) {
                                    vBranch =
                                        localBAcc.getValue(cBranch + openvdb::Coord(dx, dy, dz));
                                }
                            }
                        }
                    }

                    ConformalSkinSurfVoxel sv{};
                    sv.ijk = ijk;
                    sv.surfPos = xSurf;
                    sv.surfNormal = nSdf;
                    sv.tau = surfTau;
                    sv.requestedLayers = requestedLayers;
                    sv.weight = 1.0f;
                    sv.isManual = isManual;
                    sv.branchId = std::max(0, vBranch);
                    sv.patchId = -1;
                    candidateSlots[idx] = sv;
                    candidateValid[idx] = 1;
                }
            });

        std::vector<ConformalSkinSurfVoxel> surfVoxels;
        surfVoxels.reserve(rawSurfBand.size() / 4 + 16);
        openvdb::Int32Grid surfVoxelIdxGrid(-1);
        surfVoxelIdxGrid.setTransform(fineSdfGrid.transform().copy());
        auto surfIdxAcc = surfVoxelIdxGrid.getAccessor();
        for (std::size_t idx = 0; idx < rawSurfBand.size(); ++idx) {
            if (!candidateValid[idx]) {
                continue;
            }
            const auto svIdx = static_cast<std::int32_t>(surfVoxels.size());
            surfIdxAcc.setValue(candidateSlots[idx].ijk, svIdx);
            surfVoxels.push_back(candidateSlots[idx]);
        }

        // Lazy GPU dense `(sdf, tau)` narrow-band volume cache + batch clearance dispatcher
        bool denseGpuGridReady = false;
        bool denseGpuGridFailed = false;
        int denseNx = 0;
        int denseNy = 0;
        int denseNz = 0;
        float denseOriginX = solidBBoxMinX;
        float denseOriginY = solidBBoxMinY;
        float denseOriginZ = solidBBoxMinZ;
        float invDenseStep = 1.0f / std::max(fineVoxelSize, 1.0e-3f);
        std::vector<float> denseSdfTau;
        std::vector<std::int32_t> gpuClearanceIntParams;
        std::vector<float> gpuClearanceFloatParams;
        std::vector<float> gpuProfileXzFlat;

        auto ensureDenseGpuGrid = [&]() -> bool {
            if (denseGpuGridReady) {
                return true;
            }
            if (denseGpuGridFailed || !gpu_is_available()) {
                denseGpuGridFailed = true;
                return false;
            }
            float stepMm = fineVoxelSize;
            const float spanX = std::max(voxelSize, solidBBoxMaxX - solidBBoxMinX);
            const float spanY = std::max(voxelSize, solidBBoxMaxY - solidBBoxMinY);
            const float spanZ = std::max(voxelSize, solidBBoxMaxZ - solidBBoxMinZ);
            const double estFineCells =
                (static_cast<double>(spanX) / stepMm + 2.0) *
                (static_cast<double>(spanY) / stepMm + 2.0) *
                (static_cast<double>(spanZ) / stepMm + 2.0);
            if (estFineCells > 4.5e6) {
                stepMm = voxelSize;
            }
            invDenseStep = 1.0f / std::max(stepMm, 1.0e-3f);
            denseNx = std::max(2, static_cast<int>(std::ceil(spanX * invDenseStep)) + 2);
            denseNy = std::max(2, static_cast<int>(std::ceil(spanY * invDenseStep)) + 2);
            denseNz = std::max(2, static_cast<int>(std::ceil(spanZ * invDenseStep)) + 2);
            const std::size_t totalCells =
                static_cast<std::size_t>(denseNx) *
                static_cast<std::size_t>(denseNy) *
                static_cast<std::size_t>(denseNz);
            if (totalCells > 8'000'000u) {
                denseGpuGridFailed = true;
                return false;
            }

            denseSdfTau.assign(totalCells * 2u, 0.0f);
            const std::size_t strideY = static_cast<std::size_t>(denseNx);
            const std::size_t strideZ = strideY * static_cast<std::size_t>(denseNy);

            tbb::parallel_for(
                tbb::blocked_range<int>(0, denseNz, 4),
                [&](const tbb::blocked_range<int>& rz) {
                    auto localSdfAcc = sdfGrid.getConstAccessor();
                    openvdb::tools::GridSampler<
                        openvdb::FloatGrid::ConstAccessor,
                        openvdb::tools::BoxSampler>
                        localSdfSampler(localSdfAcc, sdfGrid.transform());
                    auto localFineSdfAcc = fineSdfGrid.getConstAccessor();
                    openvdb::tools::GridSampler<
                        openvdb::FloatGrid::ConstAccessor,
                        openvdb::tools::BoxSampler>
                        localFineSdfSampler(localFineSdfAcc, fineSdfGrid.transform());
                    auto localFineEikAcc = fineEikGrid.getConstAccessor();
                    openvdb::tools::GridSampler<
                        openvdb::FloatGrid::ConstAccessor,
                        openvdb::tools::BoxSampler>
                        localFineEikSampler(localFineEikAcc, fineEikGrid.transform());
                    auto localMainEikAcc = eikGrid.getConstAccessor();
                    openvdb::tools::GridSampler<
                        openvdb::FloatGrid::ConstAccessor,
                        openvdb::tools::BoxSampler>
                        localMainEikSampler(localMainEikAcc, eikGrid.transform());

                    for (int iz = rz.begin(); iz != rz.end(); ++iz) {
                        const float wz = denseOriginZ + static_cast<float>(iz) * stepMm;
                        for (int iy = 0; iy < denseNy; ++iy) {
                            const float wy = denseOriginY + static_cast<float>(iy) * stepMm;
                            for (int ix = 0; ix < denseNx; ++ix) {
                                const float wx = denseOriginX + static_cast<float>(ix) * stepMm;
                                const std::size_t cIdx =
                                    static_cast<std::size_t>(iz) * strideZ +
                                    static_cast<std::size_t>(iy) * strideY +
                                    static_cast<std::size_t>(ix);
                                const openvdb::Vec3d wp(wx, wy, wz);
                                const float sVal = sampleAnySdfWith(
                                    wp, localSdfAcc, localSdfSampler, localFineSdfSampler);
                                float tVal = -1.0f;
                                if (sVal <= 0.65f * voxelSize) {
                                    tVal = sampleTauAtWith(
                                        wp,
                                        localFineEikAcc,
                                        localFineEikSampler,
                                        localMainEikAcc,
                                        localMainEikSampler);
                                }
                                denseSdfTau[cIdx * 2u + 0u] = sVal;
                                denseSdfTau[cIdx * 2u + 1u] = tVal;
                            }
                        }
                    }
                });

            const NozzleBasis refNb = queryNozzleFrame(
                {rec.minX, rec.minY, rec.minZ},
                {1.0f, 0.0f, 0.0f});
            const int cMode = (activeToolPtr != nullptr)
                ? static_cast<int>(activeToolPtr->clearance_mode)
                : 0;
            float coneTipR = tipRadiusMm;
            float coneSlope = 0.65f;
            float rectHx = tipRadiusMm;
            float rectHy = tipRadiusMm;
            float rectSlopeX = 0.65f;
            float rectSlopeY = 0.65f;
            float profEndSlope = 0.0f;
            gpuProfileXzFlat.clear();
            if (activeToolPtr != nullptr) {
                const auto& tool = *activeToolPtr;
                coneTipR = tool.cone_tip_radius_mm;
                coneSlope = std::tan(
                    std::clamp(tool.cone_half_angle_deg, 1.0f, 85.0f) * kDegToRad);
                rectHx = tool.rect_tip_half_width_x_mm;
                rectHy = tool.rect_tip_half_width_y_mm;
                rectSlopeX = std::tan(
                    std::clamp(tool.rect_half_angle_x_deg, 1.0f, 85.0f) * kDegToRad);
                rectSlopeY = std::tan(
                    std::clamp(tool.rect_half_angle_y_deg, 1.0f, 85.0f) * kDegToRad);
                const auto& prof = tool.clearance_profile_xz;
                for (const auto& pt : prof) {
                    gpuProfileXzFlat.push_back(pt.x_radius_mm);
                    gpuProfileXzFlat.push_back(pt.z_height_mm);
                }
                if (prof.size() >= 2) {
                    const float dz =
                        prof.back().z_height_mm - prof[prof.size() - 2].z_height_mm;
                    const float dr =
                        prof.back().x_radius_mm - prof[prof.size() - 2].x_radius_mm;
                    profEndSlope = (dz > 1.0e-4f) ? std::max(0.0f, dr / dz) : 0.0f;
                }
            }
            const float radialStep =
                std::max({0.75f * fineVoxelSize, 0.50f * voxelSize, 0.45f});

            gpuClearanceIntParams = {
                cMode,
                brickNx,
                brickNy,
                brickNz,
                denseNx,
                denseNy,
                denseNz,
                1};
            gpuClearanceFloatParams = {
                refNb.nozzle_axis[0], refNb.nozzle_axis[1], refNb.nozzle_axis[2], 0.0f,
                refNb.travel_axis[0], refNb.travel_axis[1], refNb.travel_axis[2], 0.0f,
                refNb.lateral_axis[0], refNb.lateral_axis[1], refNb.lateral_axis[2], 0.0f,
                solidBBoxMinX, solidBBoxMinY, solidBBoxMinZ, 0.0f,
                solidBBoxMaxX, solidBBoxMaxY, solidBBoxMaxZ, 0.0f,
                brickOriginX, brickOriginY, brickOriginZ, invBrickSize,
                denseOriginX, denseOriginY, denseOriginZ, invDenseStep,
                tipRadiusMm, wallHalfWidthMm, fineVoxelSize, hTop,
                maxToolClearanceHeightMm, maxPlausibleTau, voxelSize, radialStep,
                coneTipR, coneSlope, rectHx, rectHy,
                rectSlopeX, rectSlopeY, profEndSlope, 0.0f};
            denseGpuGridReady = true;
            return true;
        };

        auto evaluateBatchNozzleClearance = [&](
            const std::vector<int>& voxelIndices,
            float overrideSkinDepthMm,
            float overridePrintTau,
            std::vector<std::uint32_t>& outFlags) {
            const std::size_t count = voxelIndices.size();
            outFlags.assign(count, 0u);
            if (count == 0) {
                return;
            }
            if (count >= 128 && ensureDenseGpuGrid()) {
                std::vector<float> candPosDepth(count * 4u);
                std::vector<float> candNormTau(count * 4u);
                for (std::size_t k = 0; k < count; ++k) {
                    const auto& sv = surfVoxels[static_cast<std::size_t>(voxelIndices[k])];
                    const float depthMm = (overrideSkinDepthMm > 0.0f)
                        ? overrideSkinDepthMm
                        : (static_cast<float>(sv.requestedLayers) * hTop);
                    const float pTau = (overridePrintTau > 0.0f)
                        ? overridePrintTau
                        : (sv.tau + depthMm);
                    candPosDepth[k * 4u + 0u] = sv.surfPos.x();
                    candPosDepth[k * 4u + 1u] = sv.surfPos.y();
                    candPosDepth[k * 4u + 2u] = sv.surfPos.z();
                    candPosDepth[k * 4u + 3u] = depthMm;
                    candNormTau[k * 4u + 0u] = sv.surfNormal.x();
                    candNormTau[k * 4u + 1u] = sv.surfNormal.y();
                    candNormTau[k * 4u + 2u] = sv.surfNormal.z();
                    candNormTau[k * 4u + 3u] = pTau;
                }
                const rust::Slice<const float> posSlice(candPosDepth.data(), candPosDepth.size());
                const rust::Slice<const float> normSlice(candNormTau.data(), candNormTau.size());
                const rust::Slice<const float> brickSlice(brickMinTau.data(), brickMinTau.size());
                const rust::Slice<const float> denseSlice(denseSdfTau.data(), denseSdfTau.size());
                const rust::Slice<const std::int32_t> intSlice(
                    gpuClearanceIntParams.data(), gpuClearanceIntParams.size());
                const rust::Slice<const float> floatSlice(
                    gpuClearanceFloatParams.data(), gpuClearanceFloatParams.size());
                const rust::Slice<const float> profSlice(
                    gpuProfileXzFlat.data(), gpuProfileXzFlat.size());
                rust::Slice<std::uint32_t> flagsSlice(outFlags.data(), outFlags.size());
                if (gpu_sweep_tool_clearance(
                        posSlice,
                        normSlice,
                        brickSlice,
                        denseSlice,
                        intSlice,
                        floatSlice,
                        profSlice,
                        flagsSlice)) {
                    return;
                }
            }

            // Multi-threaded TBB CPU fallback
            tbb::parallel_for(
                tbb::blocked_range<std::size_t>(0, count, 64),
                [&](const tbb::blocked_range<std::size_t>& r) {
                    auto localSdfAcc = sdfGrid.getConstAccessor();
                    openvdb::tools::GridSampler<
                        openvdb::FloatGrid::ConstAccessor,
                        openvdb::tools::BoxSampler>
                        localSdfSampler(localSdfAcc, sdfGrid.transform());
                    auto localFineSdfAcc = fineSdfGrid.getConstAccessor();
                    openvdb::tools::GridSampler<
                        openvdb::FloatGrid::ConstAccessor,
                        openvdb::tools::BoxSampler>
                        localFineSdfSampler(localFineSdfAcc, fineSdfGrid.transform());
                    auto localFineEikAcc = fineEikGrid.getConstAccessor();
                    openvdb::tools::GridSampler<
                        openvdb::FloatGrid::ConstAccessor,
                        openvdb::tools::BoxSampler>
                        localFineEikSampler(localFineEikAcc, fineEikGrid.transform());
                    auto localMainEikAcc = eikGrid.getConstAccessor();
                    openvdb::tools::GridSampler<
                        openvdb::FloatGrid::ConstAccessor,
                        openvdb::tools::BoxSampler>
                        localMainEikSampler(localMainEikAcc, eikGrid.transform());

                    for (std::size_t k = r.begin(); k != r.end(); ++k) {
                        const auto& sv = surfVoxels[static_cast<std::size_t>(voxelIndices[k])];
                        const float depthMm = (overrideSkinDepthMm > 0.0f)
                            ? overrideSkinDepthMm
                            : (static_cast<float>(sv.requestedLayers) * hTop);
                        const float pTau = (overridePrintTau > 0.0f)
                            ? overridePrintTau
                            : (sv.tau + depthMm);
                        bool nearMargin = false;
                        if (hasNozzleClearanceCollision(
                                sv.surfPos,
                                sv.surfNormal,
                                depthMm,
                                pTau,
                                localSdfAcc,
                                localSdfSampler,
                                localFineSdfSampler,
                                localFineEikAcc,
                                localFineEikSampler,
                                localMainEikAcc,
                                localMainEikSampler,
                                &nearMargin)) {
                            outFlags[k] = 1u;
                        } else if (nearMargin) {
                            outFlags[k] = 2u;
                        }
                    }
                });
        };

        if (!surfVoxels.empty()) {
            std::vector<int> allSurfIndices(surfVoxels.size());
            std::iota(allSurfIndices.begin(), allSurfIndices.end(), 0);
            std::vector<std::uint32_t> stage1Flags;
            evaluateBatchNozzleClearance(allSurfIndices, -1.0f, -1.0f, stage1Flags);
            for (std::size_t k = 0; k < surfVoxels.size(); ++k) {
                if ((stage1Flags[k] & 1u) != 0u) {
                    surfVoxels[k].patchId = -2;
                    surfVoxels[k].weight = 0.0f;
                } else if ((stage1Flags[k] & 2u) != 0u) {
                    surfVoxels[k].weight = 0.55f;
                }
            }
        }

        // Connected-component clustering of conformal skin surface voxels into causal patches,
        // followed by Stage-2 causal print-time (`tauCompletion`) 3D tool-clearance pruning.
        if (!surfVoxels.empty()) {
            std::vector<int> queue;
            queue.reserve(surfVoxels.size());
            const float maxStepHeight = 0.95f * fineVoxelSize;

            for (std::size_t i = 0; i < surfVoxels.size(); ++i) {
                if (surfVoxels[i].patchId != -1) {
                    continue;
                }
                const int candidatePatchId = static_cast<int>(conformalSkinPatches.size());
                queue.clear();
                queue.push_back(static_cast<int>(i));
                surfVoxels[i].patchId = candidatePatchId;

                bool hasAnyManual = false;
                double sumLayers = 0.0;

                std::size_t head = 0;
                while (head < queue.size()) {
                    const int currIdx = queue[head++];
                    const auto& cv = surfVoxels[static_cast<std::size_t>(currIdx)];
                    hasAnyManual = hasAnyManual || cv.isManual;
                    sumLayers += static_cast<double>(cv.requestedLayers);

                    // 26-connected neighborhood with coplanar step rejection so recessed
                    // surfaces (e.g. Benchy bow deck) never bridge across a vertical step
                    // onto raised rims (e.g. hull gunwale tops).
                    for (int dz = -1; dz <= 1; ++dz) {
                        for (int dy = -1; dy <= 1; ++dy) {
                            for (int dx = -1; dx <= 1; ++dx) {
                                if (dx == 0 && dy == 0 && dz == 0) {
                                    continue;
                                }
                                const int nbIdx =
                                    surfIdxAcc.getValue(cv.ijk + openvdb::Coord(dx, dy, dz));
                                if (nbIdx < 0) {
                                    continue;
                                }
                                auto& nb = surfVoxels[static_cast<std::size_t>(nbIdx)];
                                if (nb.patchId != -1) {
                                    continue;
                                }
                                if (cv.surfNormal.dot(nb.surfNormal) < 0.75f) {
                                    continue;
                                }
                                const openvdb::Vec3f diff = nb.surfPos - cv.surfPos;
                                if (std::abs(diff.dot(cv.surfNormal)) > maxStepHeight ||
                                    std::abs(diff.dot(nb.surfNormal)) > maxStepHeight) {
                                    continue;
                                }
                                nb.patchId = candidatePatchId;
                                queue.push_back(nbIdx);
                            }
                        }
                    }
                }

                const std::size_t minPatchVoxels = hasAnyManual ? 2u : 8u;
                if (queue.size() < minPatchVoxels) {
                    for (int idx : queue) {
                        surfVoxels[static_cast<std::size_t>(idx)].patchId = -2;
                    }
                    continue;
                }

                const int patchSkinLayers = std::clamp<int>(
                    static_cast<int>(std::lround(sumLayers / static_cast<double>(queue.size()))),
                    1,
                    12);
                const float patchSkinThicknessMm = static_cast<float>(patchSkinLayers) * hTop;

                // Stage-2 Causal Print-Time Pruning:
                // Because the conformal skin patch is printed all at once when the core reaches
                // `tauCompletion` (the maximum \tau across the surviving patch), all core layers
                // up to `tauCompletion` have ALREADY been printed (including adjacent walls such
                // as the Benchy cabin and hull sides up to `tauCompletion`). Iteratively prune
                // any patch voxel whose 3D tool clearance envelope collides at `tauCompletion`.
                std::vector<int> surviving = queue;
                float tauMin = 0.0f;
                float tauCompletion = 0.0f;

                for (int pruneIter = 0; pruneIter < 3; ++pruneIter) {
                    std::vector<float> patchTaus;
                    patchTaus.reserve(surviving.size());
                    for (int idx : surviving) {
                        const float t = surfVoxels[static_cast<std::size_t>(idx)].tau;
                        if (t > 0.0f && t <= maxPlausibleTau) {
                            patchTaus.push_back(t);
                        }
                    }
                    if (surviving.size() < minPatchVoxels || patchTaus.empty()) {
                        surviving.clear();
                        break;
                    }
                    std::sort(patchTaus.begin(), patchTaus.end());
                    tauMin = patchTaus.front();
                    tauCompletion = patchTaus.back();

                    std::vector<std::uint32_t> stage2Flags;
                    evaluateBatchNozzleClearance(
                        surviving,
                        patchSkinThicknessMm,
                        tauCompletion,
                        stage2Flags);

                    std::vector<int> nextSurviving;
                    nextSurviving.reserve(surviving.size());
                    bool anyPruned = false;
                    for (std::size_t k = 0; k < surviving.size(); ++k) {
                        const int idx = surviving[k];
                        auto& sv = surfVoxels[static_cast<std::size_t>(idx)];
                        if ((stage2Flags[k] & 1u) != 0u) {
                            sv.patchId = -2;
                            sv.weight = 0.0f;
                            anyPruned = true;
                        } else {
                            if ((stage2Flags[k] & 2u) != 0u) {
                                sv.weight = std::min(sv.weight, 0.55f);
                            }
                            nextSurviving.push_back(idx);
                        }
                    }
                    surviving = std::move(nextSurviving);
                    if (!anyPruned) {
                        break;
                    }
                }

                if (surviving.size() < minPatchVoxels) {
                    for (int idx : surviving) {
                        surfVoxels[static_cast<std::size_t>(idx)].patchId = -2;
                        surfVoxels[static_cast<std::size_t>(idx)].weight = 0.0f;
                    }
                    continue;
                }

                // Recompute final `tauMin`, `tauCompletion`, `meanNormal`, and `branchId`
                std::vector<float> finalTaus;
                finalTaus.reserve(surviving.size());
                openvdb::Vec3f sumNormal(0.0f, 0.0f, 0.0f);
                std::unordered_map<int, int> branchVotes;
                for (int idx : surviving) {
                    const auto& sv = surfVoxels[static_cast<std::size_t>(idx)];
                    if (sv.tau > 0.0f && sv.tau <= maxPlausibleTau) {
                        finalTaus.push_back(sv.tau);
                    }
                    sumNormal += sv.surfNormal;
                    branchVotes[sv.branchId] += 1;
                }
                if (finalTaus.empty()) {
                    for (int idx : surviving) {
                        surfVoxels[static_cast<std::size_t>(idx)].patchId = -2;
                        surfVoxels[static_cast<std::size_t>(idx)].weight = 0.0f;
                    }
                    continue;
                }
                std::sort(finalTaus.begin(), finalTaus.end());
                tauMin = finalTaus.front();
                tauCompletion = finalTaus.back();

                int bestBranch = 0;
                int bestVotes = -1;
                for (const auto& [bId, cnt] : branchVotes) {
                    if (cnt > bestVotes) {
                        bestVotes = cnt;
                        bestBranch = bId;
                    }
                }
                bestBranch = std::clamp<int>(
                    bestBranch, 0, static_cast<int>(branches.size()) - 1);

                ConformalSkinPatch patch{};
                patch.patchId = candidatePatchId;
                patch.skinLayers = patchSkinLayers;
                patch.skinThicknessMm = patchSkinThicknessMm;
                patch.tauMin = tauMin;
                patch.tauCompletion = tauCompletion;
                patch.branchId = static_cast<std::uint32_t>(bestBranch);
                patch.parentBranchId = static_cast<std::int32_t>(
                    branches[static_cast<std::size_t>(bestBranch)].primaryParentId);
                const float nLen = sumNormal.length();
                patch.meanNormal = (nLen > 1.0e-5f)
                    ? (sumNormal / nLen)
                    : openvdb::Vec3f(0.0f, 0.0f, 1.0f);
                patch.surfVoxelIndices.reserve(surviving.size());
                for (int idx : surviving) {
                    patch.surfVoxelIndices.push_back(static_cast<std::size_t>(idx));
                }
                maxConformalSkinDepthMm = std::max(maxConformalSkinDepthMm, patch.skinThicknessMm);
                conformalSkinPatches.push_back(std::move(patch));
            }
        }

        // Smooth surface weights on each conformal skin patch so the eroded core transitions
        // smoothly at the patch perimeter and tool-clearance stand-off boundary, then populate
        // the 3D narrow-band `skinErodeDepthGrid`, `skinPatchIdGrid`, and `skinPatchNormalGrid`
        // without dilating past rejected collision voxels.
        if (!conformalSkinPatches.empty()) {
            for (int pass = 0; pass < 2; ++pass) {
                std::vector<float> nextWeights(surfVoxels.size(), 0.0f);
                for (std::size_t i = 0; i < surfVoxels.size(); ++i) {
                    const auto& sv = surfVoxels[i];
                    if (sv.patchId < 0) {
                        continue;
                    }
                    float sumW = 2.0f * sv.weight;
                    float cntW = 2.0f;
                    for (int dz = -1; dz <= 1; ++dz) {
                        for (int dy = -1; dy <= 1; ++dy) {
                            for (int dx = -1; dx <= 1; ++dx) {
                                if (dx == 0 && dy == 0 && dz == 0) {
                                    continue;
                                }
                                const openvdb::Coord nbCoord = sv.ijk + openvdb::Coord(dx, dy, dz);
                                const int nbIdx = surfIdxAcc.getValue(nbCoord);
                                if (nbIdx >= 0 &&
                                    surfVoxels[static_cast<std::size_t>(nbIdx)].patchId == sv.patchId) {
                                    sumW += surfVoxels[static_cast<std::size_t>(nbIdx)].weight;
                                    cntW += 1.0f;
                                } else {
                                    const float nbSdf = fineSdfAcc.getValue(nbCoord);
                                    if (nbSdf >= -0.85f * fineVoxelSize &&
                                        nbSdf <= 0.55f * fineVoxelSize) {
                                        // Adjacent non-patch or collision-pruned boundary voxel pulls rim weight toward 0
                                        cntW += 1.0f;
                                    }
                                }
                            }
                        }
                    }
                    nextWeights[i] = std::clamp(sumW / cntW, 0.0f, 1.0f);
                }
                for (std::size_t i = 0; i < surfVoxels.size(); ++i) {
                    if (surfVoxels[i].patchId >= 0) {
                        surfVoxels[i].weight = nextWeights[i];
                    }
                }
            }

            auto erodeAcc = skinErodeDepthGrid.getAccessor();
            auto patchIdAcc = skinPatchIdGrid.getAccessor();
            auto patchNormAcc = skinPatchNormalGrid.getAccessor();
            const float minBandSdf = -(maxConformalSkinDepthMm + 2.5f * fineVoxelSize);
            const float maxBandSdf = 1.5f * fineVoxelSize;
            const float maxProjDistSq = (2.15f * fineVoxelSize) * (2.15f * fineVoxelSize);
            const float maxNearestSurfDistSq = (1.20f * fineVoxelSize) * (1.20f * fineVoxelSize);

            auto sampleSdfGradUnit = [&](
                const openvdb::Vec3d& wp,
                openvdb::Vec3f& outNormal) -> bool {
                const double h = 0.5 * static_cast<double>(fineVoxelSize);
                const float gx =
                    fineSdfSampler.wsSample(wp + openvdb::Vec3d(h, 0.0, 0.0)) -
                    fineSdfSampler.wsSample(wp - openvdb::Vec3d(h, 0.0, 0.0));
                const float gy =
                    fineSdfSampler.wsSample(wp + openvdb::Vec3d(0.0, h, 0.0)) -
                    fineSdfSampler.wsSample(wp - openvdb::Vec3d(0.0, h, 0.0));
                const float gz =
                    fineSdfSampler.wsSample(wp + openvdb::Vec3d(0.0, 0.0, h)) -
                    fineSdfSampler.wsSample(wp - openvdb::Vec3d(0.0, 0.0, h));
                const float len = std::sqrt(gx * gx + gy * gy + gz * gz);
                if (len < 1.0e-5f) {
                    return false;
                }
                outNormal = openvdb::Vec3f(gx / len, gy / len, gz / len);
                return true;
            };

            // Only visit `fineSdfGrid` narrow-band voxels within the neighborhood of actual
            // conformal skin patches rather than computing gradients across the entire object.
            const int bandRadiusVoxels = std::clamp(
                static_cast<int>(std::ceil(maxConformalSkinDepthMm / fineVoxelSize)) + 2,
                2,
                16);
            openvdb::BoolGrid nearPatchMaskGrid(false);
            auto maskAcc = nearPatchMaskGrid.getAccessor();
            std::vector<openvdb::Coord> nearPatchCoords;
            for (const auto& patch : conformalSkinPatches) {
                for (const std::size_t svIdx : patch.surfVoxelIndices) {
                    const openvdb::Coord c0 = surfVoxels[svIdx].ijk;
                    for (int dz = -bandRadiusVoxels; dz <= bandRadiusVoxels; ++dz) {
                        for (int dy = -bandRadiusVoxels; dy <= bandRadiusVoxels; ++dy) {
                            for (int dx = -bandRadiusVoxels; dx <= bandRadiusVoxels; ++dx) {
                                const openvdb::Coord nb = c0 + openvdb::Coord(dx, dy, dz);
                                if (!maskAcc.getValue(nb)) {
                                    maskAcc.setValue(nb, true);
                                    const float sdfVal = fineSdfAcc.getValue(nb);
                                    if (sdfVal >= minBandSdf && sdfVal <= maxBandSdf) {
                                        nearPatchCoords.push_back(nb);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            for (const openvdb::Coord& ijk : nearPatchCoords) {
                const float sdfVal = fineSdfAcc.getValue(ijk);
                const openvdb::Vec3d wp = fineSdfGrid.transform().indexToWorld(ijk);
                openvdb::Vec3f nSdf(0.0f, 0.0f, 1.0f);
                if (!sampleSdfGradUnit(wp, nSdf)) {
                    continue;
                }
                const openvdb::Vec3f xSurf(
                    static_cast<float>(wp.x()) - sdfVal * nSdf.x(),
                    static_cast<float>(wp.y()) - sdfVal * nSdf.y(),
                    static_cast<float>(wp.z()) - sdfVal * nSdf.z());
                const openvdb::Coord cSurf =
                    fineSdfGrid.transform().worldToIndexCellCentered(
                        openvdb::Vec3d(xSurf.x(), xSurf.y(), xSurf.z()));

                float sumWeight = 0.0f;
                float sumInvDist = 0.0f;
                int nearestSurfPatchId = -1;
                float nearestSurfDistSq = std::numeric_limits<float>::max();
                openvdb::Vec3f sumSurfNormal(0.0f, 0.0f, 0.0f);

                for (int dz = -2; dz <= 2; ++dz) {
                    for (int dy = -2; dy <= 2; ++dy) {
                        for (int dx = -2; dx <= 2; ++dx) {
                            const int svIdx =
                                surfIdxAcc.getValue(cSurf + openvdb::Coord(dx, dy, dz));
                            if (svIdx < 0) {
                                continue;
                            }
                            const auto& sv = surfVoxels[static_cast<std::size_t>(svIdx)];
                            const float dotN = nSdf.dot(sv.surfNormal);
                            if (dotN < 0.55f) {
                                continue;
                            }
                            const float dSq = (xSurf - sv.surfPos).lengthSqr();
                            if (dSq > maxProjDistSq) {
                                continue;
                            }
                            const float wDist = 1.0f / (0.15f * fineVoxelSize * fineVoxelSize + dSq);
                            // Always accumulate `sumInvDist` for both valid and collision-pruned
                            // surface voxels so the erosion depth tapers inside the safe patch
                            // boundary and NEVER dilates outward into rejected collision zones.
                            sumInvDist += wDist;
                            if (dSq < nearestSurfDistSq) {
                                nearestSurfDistSq = dSq;
                                nearestSurfPatchId = sv.patchId;
                            }
                            if (sv.patchId >= 0) {
                                sumWeight += wDist * sv.weight * dotN;
                                sumSurfNormal += sv.surfNormal * wDist;
                            }
                        }
                    }
                }

                // Require the closest projected surface voxel to belong to a valid conformal patch
                // (`nearestSurfPatchId >= 0`) and lie within 1.20 fine voxels.
                if (nearestSurfPatchId >= 0 &&
                    nearestSurfDistSq <= maxNearestSurfDistSq &&
                    sumInvDist > 1.0e-6f) {
                    const float avgW = std::clamp(sumWeight / sumInvDist, 0.0f, 1.0f);
                    const float erodeDepth =
                        avgW * conformalSkinPatches[static_cast<std::size_t>(nearestSurfPatchId)].skinThicknessMm;
                    if (erodeDepth > 1.0e-4f) {
                        erodeAcc.setValue(ijk, erodeDepth);
                        patchIdAcc.setValue(ijk, nearestSurfPatchId);
                        const float snLen = sumSurfNormal.length();
                        if (snLen > 1.0e-5f) {
                            patchNormAcc.setValue(
                                ijk,
                                openvdb::Vec3s(
                                    sumSurfNormal.x() / snLen,
                                    sumSurfNormal.y() / snLen,
                                    sumSurfNormal.z() / snLen));
                        }
                    }
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
            auto skinErodeAcc = skinErodeDepthGrid.getConstAccessor();
            openvdb::tools::GridSampler<openvdb::FloatGrid::ConstAccessor, openvdb::tools::BoxSampler>
                skinErodeSampler(skinErodeAcc, skinErodeDepthGrid.transform());
            auto skinPatchIdAcc = skinPatchIdGrid.getConstAccessor();

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

            std::vector<std::uint8_t> leafInLayerState(numEikLeaves, 0);
            std::vector<std::size_t> touchedLeaves;
            touchedLeaves.reserve(std::min<std::size_t>(numEikLeaves, 4096));

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

                openvdb::FloatGrid layerEikGrid(eikBgVal);
                layerEikGrid.setTransform(eikGrid.transform().copy());
                auto& layerTree = layerEikGrid.tree();
                bool hasAnyActiveLeaf = false;

                for (std::size_t lIdx = 0; lIdx < numEikLeaves; ++lIdx) {
                    if (!leafHasActive[lIdx]) {
                        continue;
                    }
                    if (pl.isovalue < leafTauMin[lIdx] || pl.isovalue > leafTauMax[lIdx]) {
                        continue;
                    }
                    leafInLayerState[lIdx] = 2;
                    touchedLeaves.push_back(lIdx);
                    layerTree.addLeaf(new EikLeafType(*activeMaskedLeaves[lIdx]));
                    hasAnyActiveLeaf = true;
                }

                const std::size_t numStraddling = touchedLeaves.size();
                for (std::size_t k = 0; k < numStraddling; ++k) {
                    const std::size_t lIdx = touchedLeaves[k];
                    for (const std::uint32_t nbIdx : leafHaloNeighbors[lIdx]) {
                        if (leafInLayerState[nbIdx] == 0) {
                            leafInLayerState[nbIdx] = 1;
                            touchedLeaves.push_back(nbIdx);
                            layerTree.addLeaf(new EikLeafType(*haloMaskedLeaves[nbIdx]));
                        }
                    }
                }

                for (const std::size_t lIdx : touchedLeaves) {
                    leafInLayerState[lIdx] = 0;
                }
                touchedLeaves.clear();

                std::vector<openvdb::Vec3s> points;
                std::vector<openvdb::Vec3I> triangles;
                std::vector<openvdb::Vec4I> quads;

                if (hasAnyActiveLeaf) {
                    openvdb::tools::volumeToMesh(
                        layerEikGrid,
                        points,
                        triangles,
                        quads,
                        static_cast<double>(pl.isovalue),
                        0.0,
                        true);
                }

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
                    bool anyErodedByConformalSkin = false;
                    for (std::size_t i = 0; i < verts.size(); ++i) {
                        const openvdb::Vec3d wp(verts[i].x, verts[i].y, verts[i].z);
                        float s = sampleCompositeSdf(wp);
                        if (!conformalSkinPatches.empty() &&
                            s > -(maxConformalSkinDepthMm + 2.5f * fineVoxelSize) &&
                            s < 1.5f * fineVoxelSize) {
                            const float erodeDepth = skinErodeSampler.wsSample(wp);
                            if (erodeDepth > 1.0e-4f) {
                                const openvdb::Coord cFine =
                                    fineSdfGrid.transform().worldToIndexCellCentered(wp);
                                int pId = skinPatchIdAcc.getValue(cFine);
                                if (pId < 0) {
                                    for (int dz = -1; dz <= 1 && pId < 0; ++dz) {
                                        for (int dy = -1; dy <= 1 && pId < 0; ++dy) {
                                            for (int dx = -1; dx <= 1 && pId < 0; ++dx) {
                                                pId = skinPatchIdAcc.getValue(
                                                    cFine + openvdb::Coord(dx, dy, dz));
                                            }
                                        }
                                    }
                                }
                                if (pId >= 0 &&
                                    static_cast<std::size_t>(pId) < conformalSkinPatches.size()) {
                                    const auto& patch =
                                        conformalSkinPatches[static_cast<std::size_t>(pId)];
                                    // Erode core layers beneath the conformal skin patch, while leaving
                                    // taller protrusion layers above `patch.tauCompletion` un-eroded.
                                    if (pl.isovalue >= patch.tauMin - patch.skinThicknessMm - 2.0f * hTop &&
                                        pl.isovalue <= patch.tauCompletion + 1.5f * hTop) {
                                        s += erodeDepth;
                                        anyErodedByConformalSkin = true;
                                    }
                                }
                            }
                        }
                        vertSdf[i] = s;
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
                        const float minValleyEdgeSq =
                            (0.30f * fineVoxelSize) * (0.30f * fineVoxelSize);

                        auto considerValleyEdge = [&](std::uint32_t u, std::uint32_t v) {
                            const std::uint32_t lo = std::min(u, v);
                            const std::uint32_t hi = std::max(u, v);
                            const float dLo = vertSdf[lo];
                            const float dHi = vertSdf[hi];
                            if (dLo <= 0.0f || dHi <= 0.0f ||
                                std::min(dLo, dHi) > maxEndpointSdf) {
                                return;
                            }
                            const float edx = verts[hi].x - verts[lo].x;
                            const float edy = verts[hi].y - verts[lo].y;
                            const float edz = verts[hi].z - verts[lo].z;
                            if (edx * edx + edy * edy + edz * edz < minValleyEdgeSq) {
                                return;
                            }
                            const std::uint64_t key =
                                (static_cast<std::uint64_t>(lo) << 32) |
                                static_cast<std::uint64_t>(hi);
                            if (valleySplitMap.find(key) != valleySplitMap.end()) {
                                return;
                            }

                            const float minEndpointSdf = std::min(dLo, dHi);
                            const openvdb::Vec3d wpMid(
                                verts[lo].x + 0.50f * edx,
                                verts[lo].y + 0.50f * edy,
                                verts[lo].z + 0.50f * edz);
                            const float sMid = sampleCompositeSdf(wpMid);
                            float bestT = 0.50f;
                            float bestSdf = std::min(minEndpointSdf, sMid);

                            // Only probe quarter-points (0.25, 0.75) if the midpoint is dipping
                            // below the lower endpoint or already near the wall boundary.
                            if (sMid < minEndpointSdf || minEndpointSdf < 0.85f * fineVoxelSize) {
                                for (const float tCand : {0.25f, 0.75f}) {
                                    const openvdb::Vec3d wpCand(
                                        verts[lo].x + tCand * edx,
                                        verts[lo].y + tCand * edy,
                                        verts[lo].z + tCand * edz);
                                    const float sCand = sampleCompositeSdf(wpCand);
                                    if (sCand < bestSdf) {
                                        bestSdf = sCand;
                                        bestT = tCand;
                                    }
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

                    // Always clip against the exact physical surface (\phi_sdf <= 0.0) first so
                    // cross-sections with valid interior geometry are never inflated outward.
                    clipAgainstSdfThreshold(0.0f);

                    if (info.indices.empty() && !anyErodedByConformalSkin && pl.sdfClipAllowance > 0.0f) {
                        clipAgainstSdfThreshold(pl.sdfClipAllowance);
                    }

                    // If trilinear SDF rounding at a sharp 3D corner still left this layer empty,
                    // find the minimum SDF on the branch's isosurface sheet and clip with a small margin.
                    if (info.indices.empty() && !anyErodedByConformalSkin) {
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

    // ------------------------------------------------------------------------
    // 5c. Extract Conformal SDF Skin Layers (\phi_sdf = -(N_p - m + 0.15) * hTop)
    //     and Schedule Causally at Each Patch's Completion Isovalue \tau_completion
    // ------------------------------------------------------------------------
    if (!conformalSkinPatches.empty()) {
        auto lerpSkinVertex = [](const RenderVertex& a, const RenderVertex& b, float t) -> RenderVertex {
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

        using SdfTreeType = openvdb::FloatGrid::TreeType;
        using SdfLeafType = SdfTreeType::LeafNodeType;
        constexpr int kLeafDim = static_cast<int>(SdfLeafType::DIM);

        std::vector<openvdb::FloatGrid::Ptr> patchSdfGrids(conformalSkinPatches.size());
        struct SkinLayerExtractionJob {
            std::size_t patchIdx;
            int m;
        };
        std::vector<SkinLayerExtractionJob> skinJobs;

        for (std::size_t pIdx = 0; pIdx < conformalSkinPatches.size(); ++pIdx) {
            const auto& patch = conformalSkinPatches[pIdx];
            openvdb::FloatGrid::Ptr patchSdfGrid =
                openvdb::FloatGrid::create(fineSdfGrid.background());
            patchSdfGrid->setTransform(fineSdfGrid.transform().copy());
            auto& patchTree = patchSdfGrid->tree();
            openvdb::Int32Grid addedLeafGrid(0);
            auto addedLeafAcc = addedLeafGrid.getAccessor();

            for (auto pIt = skinPatchIdGrid.cbeginValueOn(); pIt; ++pIt) {
                if (*pIt != patch.patchId) {
                    continue;
                }
                const openvdb::Coord c = pIt.getCoord();
                const openvdb::Coord baseOrigin(
                    c.x() & ~(kLeafDim - 1),
                    c.y() & ~(kLeafDim - 1),
                    c.z() & ~(kLeafDim - 1));
                for (int dz = -1; dz <= 1; ++dz) {
                    for (int dy = -1; dy <= 1; ++dy) {
                        for (int dx = -1; dx <= 1; ++dx) {
                            const openvdb::Coord orig = baseOrigin.offsetBy(
                                dx * kLeafDim, dy * kLeafDim, dz * kLeafDim);
                            if (addedLeafAcc.getValue(orig) == 0) {
                                addedLeafAcc.setValue(orig, 1);
                                if (const SdfLeafType* srcLeaf =
                                        fineSdfGrid.tree().probeConstLeaf(orig)) {
                                    patchTree.addLeaf(new SdfLeafType(*srcLeaf));
                                }
                            }
                        }
                    }
                }
            }

            if (patchTree.leafCount() == 0) {
                continue;
            }
            patchSdfGrids[pIdx] = patchSdfGrid;
            for (int m = 1; m <= patch.skinLayers; ++m) {
                skinJobs.push_back(SkinLayerExtractionJob{pIdx, m});
            }
        }

        std::vector<SelectedLayerInfo> skinJobResults(skinJobs.size());
        std::vector<std::uint8_t> skinJobValid(skinJobs.size(), 0);

        tbb::parallel_for(
            tbb::blocked_range<std::size_t>(0, skinJobs.size()),
            [&](const tbb::blocked_range<std::size_t>& r) {
                auto fineSdfAcc = fineSdfGrid.getConstAccessor();
                openvdb::tools::GridSampler<
                    openvdb::FloatGrid::ConstAccessor,
                    openvdb::tools::BoxSampler>
                    fineSdfSampler(fineSdfAcc, fineSdfGrid.transform());
                auto skinErodeAcc = skinErodeDepthGrid.getConstAccessor();
                openvdb::tools::GridSampler<
                    openvdb::FloatGrid::ConstAccessor,
                    openvdb::tools::BoxSampler>
                    skinErodeSampler(skinErodeAcc, skinErodeDepthGrid.transform());
                auto skinPatchIdAcc = skinPatchIdGrid.getConstAccessor();
                auto skinPatchNormAcc = skinPatchNormalGrid.getConstAccessor();
                openvdb::tools::GridSampler<
                    openvdb::Vec3SGrid::ConstAccessor,
                    openvdb::tools::BoxSampler>
                    skinPatchNormSampler(skinPatchNormAcc, skinPatchNormalGrid.transform());

                auto sampleFineSdfGradUnit = [&](
                    const openvdb::Vec3d& wp,
                    openvdb::Vec3f& outNormal) -> bool {
                    const double h = 0.5 * static_cast<double>(fineVoxelSize);
                    const float gx =
                        fineSdfSampler.wsSample(wp + openvdb::Vec3d(h, 0.0, 0.0)) -
                        fineSdfSampler.wsSample(wp - openvdb::Vec3d(h, 0.0, 0.0));
                    const float gy =
                        fineSdfSampler.wsSample(wp + openvdb::Vec3d(0.0, h, 0.0)) -
                        fineSdfSampler.wsSample(wp - openvdb::Vec3d(0.0, h, 0.0));
                    const float gz =
                        fineSdfSampler.wsSample(wp + openvdb::Vec3d(0.0, 0.0, h)) -
                        fineSdfSampler.wsSample(wp - openvdb::Vec3d(0.0, 0.0, h));
                    const float len = std::sqrt(gx * gx + gy * gy + gz * gz);
                    if (len < 1.0e-5f) {
                        return false;
                    }
                    outNormal = openvdb::Vec3f(gx / len, gy / len, gz / len);
                    return true;
                };

                for (std::size_t jIdx = r.begin(); jIdx != r.end(); ++jIdx) {
                    const auto& job = skinJobs[jIdx];
                    const auto& patch = conformalSkinPatches[job.patchIdx];
                    const int m = job.m;
                    const auto& patchSdfGrid = *patchSdfGrids[job.patchIdx];

                    // Target SDF level set from innermost (m = 1) to outermost (m = N_p):
                    // \phi_m = -(N_p - m + 0.15) * hTop
                    const float targetDepthMm =
                        (static_cast<float>(patch.skinLayers - m) + 0.15f) * hTop;
                    const float phiTarget = -targetDepthMm;
                    const float requiredErodeMm =
                        (static_cast<float>(patch.skinLayers - m) + 0.25f) * hTop;

                    std::vector<openvdb::Vec3s> points;
                    std::vector<openvdb::Vec3I> triangles;
                    std::vector<openvdb::Vec4I> quads;
                    openvdb::tools::volumeToMesh(
                        patchSdfGrid,
                        points,
                        triangles,
                        quads,
                        static_cast<double>(phiTarget),
                        0.0,
                        true);

                    if (points.empty() || (triangles.empty() && quads.empty())) {
                        continue;
                    }

                    std::vector<RenderVertex> verts(points.size());
                    std::vector<float> clipDist(points.size(), 1.0f);
                    for (std::size_t i = 0; i < points.size(); ++i) {
                        openvdb::Vec3d wp(points[i].x(), points[i].y(), points[i].z());
                        openvdb::Vec3f nSdf = patch.meanNormal;
                        sampleFineSdfGradUnit(wp, nSdf);

                        // 1 Newton projection step onto exact \phi_sdf(x) = phiTarget
                        const float curSdf = fineSdfSampler.wsSample(wp);
                        const float err = curSdf - phiTarget;
                        const float step =
                            std::clamp(err, -0.35f * fineVoxelSize, 0.35f * fineVoxelSize);
                        wp -= openvdb::Vec3d(
                            step * nSdf.x(), step * nSdf.y(), step * nSdf.z());
                        sampleFineSdfGradUnit(wp, nSdf);

                        verts[i].x = static_cast<float>(wp.x());
                        verts[i].y = static_cast<float>(wp.y());
                        verts[i].z = static_cast<float>(wp.z());
                        verts[i].nx = nSdf.x();
                        verts[i].ny = nSdf.y();
                        verts[i].nz = nSdf.z();

                        // Check patch membership and normal alignment with the generating surface
                        const openvdb::Coord cFine =
                            fineSdfGrid.transform().worldToIndexCellCentered(wp);
                        int pId = skinPatchIdAcc.getValue(cFine);
                        if (pId != patch.patchId) {
                            for (int dz = -1; dz <= 1 && pId != patch.patchId; ++dz) {
                                for (int dy = -1; dy <= 1 && pId != patch.patchId; ++dy) {
                                    for (int dx = -1; dx <= 1 && pId != patch.patchId; ++dx) {
                                        const int nbP = skinPatchIdAcc.getValue(
                                            cFine + openvdb::Coord(dx, dy, dz));
                                        if (nbP == patch.patchId) {
                                            pId = nbP;
                                        }
                                    }
                                }
                            }
                        }

                        if (pId == patch.patchId) {
                            const openvdb::Vec3s pNormS = skinPatchNormSampler.wsSample(wp);
                            openvdb::Vec3f pNorm(pNormS.x(), pNormS.y(), pNormS.z());
                            const float pNormLen = pNorm.length();
                            if (pNormLen > 1.0e-4f) {
                                pNorm /= pNormLen;
                            } else {
                                pNorm = patch.meanNormal;
                            }
                            const float alignFactor =
                                std::clamp((nSdf.dot(pNorm) - 0.50f) / 0.25f, 0.0f, 1.0f);
                            const float effErode = skinErodeSampler.wsSample(wp) * alignFactor;
                            clipDist[i] = requiredErodeMm - effErode;
                        } else {
                            clipDist[i] = requiredErodeMm;
                        }
                    }

                    std::vector<std::uint32_t> inds;
                    inds.reserve(triangles.size() * 3 + quads.size() * 6);
                    auto pushOrientedTri = [&](std::uint32_t a, std::uint32_t b, std::uint32_t c) {
                        const openvdb::Vec3f pa(verts[a].x, verts[a].y, verts[a].z);
                        const openvdb::Vec3f pb(verts[b].x, verts[b].y, verts[b].z);
                        const openvdb::Vec3f pc(verts[c].x, verts[c].y, verts[c].z);
                        const openvdb::Vec3f triN = (pb - pa).cross(pc - pa);
                        const openvdb::Vec3f avgN(
                            verts[a].nx + verts[b].nx + verts[c].nx,
                            verts[a].ny + verts[b].ny + verts[c].ny,
                            verts[a].nz + verts[b].nz + verts[c].nz);
                        if (triN.dot(avgN) >= 0.0f) {
                            inds.push_back(a);
                            inds.push_back(b);
                            inds.push_back(c);
                        } else {
                            inds.push_back(a);
                            inds.push_back(c);
                            inds.push_back(b);
                        }
                    };
                    for (const auto& tri : triangles) {
                        pushOrientedTri(tri[0], tri[2], tri[1]);
                    }
                    for (const auto& q : quads) {
                        pushOrientedTri(q[0], q[2], q[1]);
                        pushOrientedTri(q[0], q[3], q[2]);
                    }

                    SelectedLayerInfo skinInfo{};
                    skinInfo.branch_id = patch.branchId;
                    skinInfo.parent_branch_id = patch.parentBranchId;
                    skinInfo.branch_layer_index = static_cast<std::uint32_t>(m - 1);
                    skinInfo.branch_layer_count = static_cast<std::uint32_t>(patch.skinLayers);
                    skinInfo.isovalue =
                        patch.tauCompletion + static_cast<float>(m) * 1.0e-3f;
                    skinInfo.mean_thickness_mm = hTop;
                    skinInfo.min_thickness_mm = hTop;
                    skinInfo.max_thickness_mm = hTop;
                    skinInfo.is_first_layer = false;
                    skinInfo.is_top_layer = true;

                    std::vector<std::int32_t> vertexRemap(verts.size(), -1);
                    std::unordered_map<std::uint64_t, std::uint32_t> edgeCutRemap;
                    auto emitSkinVertex = [&](std::uint32_t oldIdx) -> std::uint32_t {
                        if (vertexRemap[oldIdx] >= 0) {
                            return static_cast<std::uint32_t>(vertexRemap[oldIdx]);
                        }
                        const auto newIdx = static_cast<std::uint32_t>(skinInfo.vertices.size());
                        skinInfo.vertices.push_back(verts[oldIdx]);
                        vertexRemap[oldIdx] = static_cast<std::int32_t>(newIdx);
                        return newIdx;
                    };
                    auto emitSkinCutVertex =
                        [&](std::uint32_t u, std::uint32_t v, float dU, float dV) -> std::uint32_t {
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
                        RenderVertex cv = lerpSkinVertex(verts[lo], verts[hi], t);
                        // Ensure cut rim vertex stays strictly inside \phi_sdf <= 0
                        const openvdb::Vec3d wpC(cv.x, cv.y, cv.z);
                        const float cSdf = fineSdfSampler.wsSample(wpC);
                        if (cSdf > -0.02f * hTop) {
                            const float pushIn = cSdf + 0.05f * hTop;
                            cv.x -= pushIn * cv.nx;
                            cv.y -= pushIn * cv.ny;
                            cv.z -= pushIn * cv.nz;
                        }
                        const auto newIdx = static_cast<std::uint32_t>(skinInfo.vertices.size());
                        skinInfo.vertices.push_back(cv);
                        edgeCutRemap.emplace(key, newIdx);
                        return newIdx;
                    };

                    for (std::size_t t = 0; t + 2 < inds.size(); t += 3) {
                        const std::uint32_t i0 = inds[t + 0];
                        const std::uint32_t i1 = inds[t + 1];
                        const std::uint32_t i2 = inds[t + 2];

                        const float d0 = clipDist[i0];
                        const float d1 = clipDist[i1];
                        const float d2 = clipDist[i2];

                        const bool in0 = (d0 <= 0.0f);
                        const bool in1 = (d1 <= 0.0f);
                        const bool in2 = (d2 <= 0.0f);
                        if (!in0 && !in1 && !in2) {
                            continue;
                        }

                        if (in0 && in1 && in2) {
                            const std::uint32_t n0 = emitSkinVertex(i0);
                            const std::uint32_t n1 = emitSkinVertex(i1);
                            const std::uint32_t n2 = emitSkinVertex(i2);
                            skinInfo.indices.push_back(n0);
                            skinInfo.indices.push_back(n1);
                            skinInfo.indices.push_back(n2);
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

                        const std::uint32_t vAB = emitSkinCutVertex(idxA, idxB, dA, dB);
                        const std::uint32_t vAC = emitSkinCutVertex(idxA, idxC, dA, dC);
                        const auto& p0 = skinInfo.vertices[vAB];
                        const auto& p1 = skinInfo.vertices[vAC];
                        skinInfo.contour_segments.push_back(p0.x);
                        skinInfo.contour_segments.push_back(p0.y);
                        skinInfo.contour_segments.push_back(p0.z);
                        skinInfo.contour_segments.push_back(p1.x);
                        skinInfo.contour_segments.push_back(p1.y);
                        skinInfo.contour_segments.push_back(p1.z);

                        if (soloInside) {
                            const std::uint32_t nA = emitSkinVertex(idxA);
                            skinInfo.indices.push_back(nA);
                            skinInfo.indices.push_back(vAB);
                            skinInfo.indices.push_back(vAC);
                        } else {
                            const std::uint32_t nB = emitSkinVertex(idxB);
                            const std::uint32_t nC = emitSkinVertex(idxC);
                            skinInfo.indices.push_back(vAB);
                            skinInfo.indices.push_back(nB);
                            skinInfo.indices.push_back(nC);
                            skinInfo.indices.push_back(vAB);
                            skinInfo.indices.push_back(nC);
                            skinInfo.indices.push_back(vAC);
                        }
                    }

                    if (!skinInfo.vertices.empty() && skinInfo.indices.size() >= 3) {
                        skinJobResults[jIdx] = std::move(skinInfo);
                        skinJobValid[jIdx] = 1;
                    }
                }
            });

        std::vector<SelectedLayerInfo> conformalSkinLayers;
        conformalSkinLayers.reserve(skinJobs.size());
        for (std::size_t jIdx = 0; jIdx < skinJobs.size(); ++jIdx) {
            if (skinJobValid[jIdx]) {
                conformalSkinLayers.push_back(std::move(skinJobResults[jIdx]));
            }
        }

        if (!conformalSkinLayers.empty()) {
            // Merge non-empty core layers with conformal skin layers and sort causally by isovalue
            std::vector<SelectedLayerInfo> mergedLayers;
            mergedLayers.reserve(extractedLayers.size() + conformalSkinLayers.size());
            for (auto& layer : extractedLayers) {
                if (!layer.vertices.empty() && layer.indices.size() >= 3) {
                    mergedLayers.push_back(std::move(layer));
                }
            }
            for (auto& skinLayer : conformalSkinLayers) {
                mergedLayers.push_back(std::move(skinLayer));
            }
            std::stable_sort(
                mergedLayers.begin(),
                mergedLayers.end(),
                [](const SelectedLayerInfo& a, const SelectedLayerInfo& b) {
                    if (std::abs(a.isovalue - b.isovalue) > 1.0e-5f) {
                        return a.isovalue < b.isovalue;
                    }
                    return a.branch_id < b.branch_id;
                });
            for (std::size_t i = 0; i < mergedLayers.size(); ++i) {
                mergedLayers[i].layer_index = static_cast<std::uint32_t>(i);
            }
            extractedLayers = std::move(mergedLayers);
        }
    }

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
    s.overhang_angle_threshold_deg = (s.overhang_angle_threshold_deg > 1.0f)
        ? std::clamp(s.overhang_angle_threshold_deg, 5.0f, 89.0f)
        : 45.0f;
    s.max_bridge_length_mm = (s.max_bridge_length_mm > 0.1f)
        ? std::clamp(s.max_bridge_length_mm, 0.5f, 500.0f)
        : 40.0f;
    s.wave_overhang_overlap_fraction = (s.wave_overhang_overlap_fraction > 1.0e-4f)
        ? std::clamp(s.wave_overhang_overlap_fraction, 0.01f, 0.45f)
        : 0.08f;
    impl_->shellPathSpec = s;
}

ShellPathSpec SlicerBackend::shellPathSpec() const noexcept {
    return impl_->shellPathSpec;
}

void SlicerBackend::setSupportSpec(const SupportSpec& spec) noexcept {
    SupportSpec s = spec;
    s.support_tool_index = std::max<std::int32_t>(-1, s.support_tool_index);
    s.support_style = (s.support_style == 1) ? 1 : 0;
    s.max_bridge_length_mm = std::clamp(s.max_bridge_length_mm, 0.0f, 500.0f);
    s.z_gap_mm = std::clamp(s.z_gap_mm, 0.0f, 5.0f);
    s.xy_gap_mm = std::clamp(s.xy_gap_mm, 0.0f, 10.0f);
    s.interface_layers = std::min<std::uint32_t>(s.interface_layers, 16);
    s.bed_interface_layers = std::min<std::uint32_t>(s.bed_interface_layers, 8);
    s.base_spacing_mm = (s.base_spacing_mm > 0.1f)
        ? std::clamp(s.base_spacing_mm, 0.4f, 20.0f)
        : 2.50f;
    s.interface_spacing_mm = std::isfinite(s.interface_spacing_mm)
        ? std::clamp(s.interface_spacing_mm, 0.0f, 5.0f)
        : 0.15f;
    s.line_width_mm = (s.line_width_mm > 0.05f)
        ? std::clamp(s.line_width_mm, 0.15f, 3.0f)
        : 0.0f;
    s.tree_branch_angle_deg = (s.tree_branch_angle_deg > 1.0f)
        ? std::clamp(s.tree_branch_angle_deg, 15.0f, 65.0f)
        : 45.0f;
    s.tree_tip_diameter_mm = (s.tree_tip_diameter_mm > 0.1f)
        ? std::clamp(s.tree_tip_diameter_mm, 0.80f, 8.00f)
        : 2.40f;
    s.tree_trunk_max_diameter_mm = (s.tree_trunk_max_diameter_mm > 0.5f)
        ? std::clamp(s.tree_trunk_max_diameter_mm, std::max(2.0f, s.tree_tip_diameter_mm), 30.0f)
        : 12.00f;
    s.tree_branch_wall_count = (s.tree_branch_wall_count >= 1)
        ? std::clamp<std::uint32_t>(s.tree_branch_wall_count, 1, 4)
        : 2;
    impl_->supportSpec = s;
}

SupportSpec SlicerBackend::supportSpec() const noexcept {
    return impl_->supportSpec;
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

    const auto ovIt = rec.scalarFields.find("overhang");
    const openvdb::FloatGrid* overhangPtr =
        (ovIt != rec.scalarFields.end() && ovIt->second) ? ovIt->second.get() : nullptr;

    const openvdb::FloatGrid* coarseSdfPtr = nullptr;
    const openvdb::FloatGrid* coarseEikPtr = nullptr;
    const openvdb::Vec3SGrid* coarseToolAxisPtr = nullptr;
    const openvdb::FloatGrid* coarseOverhangPtr = nullptr;
    if (rec.isMultiResolution) {
        const auto cSdfIt = rec.coarseScalarFields.find("sdf");
        const auto cEikIt = rec.coarseScalarFields.find("eikonal");
        if (cSdfIt != rec.coarseScalarFields.end() && cSdfIt->second &&
            cEikIt != rec.coarseScalarFields.end() && cEikIt->second) {
            coarseSdfPtr = cSdfIt->second.get();
            coarseEikPtr = cEikIt->second.get();
            coarseToolAxisPtr = rec.coarseToolAxisGrid.get();
        }
        const auto cOvIt = rec.coarseScalarFields.find("overhang");
        if (cOvIt != rec.coarseScalarFields.end() && cOvIt->second) {
            coarseOverhangPtr = cOvIt->second.get();
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
        rec.fineShellInnerDepthMm,
        overhangPtr,
        coarseOverhangPtr,
        impl_->gravityDirectionBed);

    // Step 2: Generate Support Base (`feature_kind = 7`) and Support Interface (`feature_kind = 8`)
    // toolpaths for unreachable order-field regions projected along gravity `+\hat{g}_{bed}`
    const auto supIt = rec.scalarFields.find("support_needed");
    if (impl_->supportSpec.enabled && supIt != rec.scalarFields.end() && supIt->second && !res.layers.empty()) {
        const openvdb::FloatGrid& supGrid = *supIt->second;
        const openvdb::FloatGrid& sdfGrid = *sdfIt->second;
        const float dx = static_cast<float>(supGrid.transform().voxelSize()[0]);

        const std::array<float, 3> gRaw = queryGravityDirection({0.0f, 0.0f, 0.0f});
        const float gLen = std::sqrt(gRaw[0] * gRaw[0] + gRaw[1] * gRaw[1] + gRaw[2] * gRaw[2]);
        const std::array<float, 3> gDir = (gLen > 1.0e-6f)
            ? std::array<float, 3>{gRaw[0] / gLen, gRaw[1] / gLen, gRaw[2] / gLen}
            : std::array<float, 3>{0.0f, 0.0f, -1.0f};
        const std::array<float, 3> uUp{-gDir[0], -gDir[1], -gDir[2]};

        // Orthonormal basis (tU, tV, uUp) aligned with anti-gravity uUp
        const std::array<float, 3> refAxis =
            (std::abs(uUp[0]) < 0.9f) ? std::array<float, 3>{1.0f, 0.0f, 0.0f}
                                      : std::array<float, 3>{0.0f, 1.0f, 0.0f};
        const float dotRef = refAxis[0] * uUp[0] + refAxis[1] * uUp[1] + refAxis[2] * uUp[2];
        std::array<float, 3> tU{
            refAxis[0] - dotRef * uUp[0],
            refAxis[1] - dotRef * uUp[1],
            refAxis[2] - dotRef * uUp[2]};
        const float tULen = std::sqrt(tU[0] * tU[0] + tU[1] * tU[1] + tU[2] * tU[2]);
        if (tULen > 1.0e-6f) {
            tU[0] /= tULen;
            tU[1] /= tULen;
            tU[2] /= tULen;
        } else {
            tU = {1.0f, 0.0f, 0.0f};
        }
        const std::array<float, 3> tV{
            uUp[1] * tU[2] - uUp[2] * tU[1],
            uUp[2] * tU[0] - uUp[0] * tU[2],
            uUp[0] * tU[1] - uUp[1] * tU[0]};

        // Reference build-plate origin P0 and minimum anti-gravity projection hMinObj
        const std::array<float, 3> pOrigin{0.0f, 0.0f, rec.minZ};
        float minObjH = std::numeric_limits<float>::max();
        for (const auto& v : rec.vertexBuffer) {
            const float h = (v.x - pOrigin[0]) * uUp[0] +
                            (v.y - pOrigin[1]) * uUp[1] +
                            (v.z - pOrigin[2]) * uUp[2];
            minObjH = std::min(minObjH, h);
        }
        if (minObjH >= 1.0e5f) {
            minObjH = 0.0f;
        }

        // Build a 2D column map in the (tU, tV) plane of unreachable boundary contact roofs
        const float cellStepMm = std::clamp(dx, 0.40f, 1.00f);
        struct ColumnSpan {
            float roofH = -1.0f;
            float floorH = 0.0f;
        };
        std::unordered_map<std::uint64_t, ColumnSpan> supportColumns;

        auto packCellKey = [](std::int32_t iu, std::int32_t iv) -> std::uint64_t {
            return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(iu)) << 32) |
                   static_cast<std::uint64_t>(static_cast<std::uint32_t>(iv));
        };
        auto unpackCellKey = [](std::uint64_t key, std::int32_t& iu, std::int32_t& iv) {
            iu = static_cast<std::int32_t>(static_cast<std::uint32_t>(key >> 32));
            iv = static_cast<std::int32_t>(static_cast<std::uint32_t>(key & 0xFFFFFFFFULL));
        };

        auto sdfAcc = sdfGrid.getConstAccessor();
        openvdb::tools::GridSampler<openvdb::FloatGrid::ConstAccessor, openvdb::tools::BoxSampler>
            sdfSampler(sdfAcc, sdfGrid.transform());
        std::unique_ptr<openvdb::tools::GridSampler<openvdb::FloatGrid, openvdb::tools::BoxSampler>>
            coarseSdfSamplerPtr;
        if (coarseSdfPtr != nullptr) {
            coarseSdfSamplerPtr = std::make_unique<
                openvdb::tools::GridSampler<openvdb::FloatGrid, openvdb::tools::BoxSampler>>(
                *coarseSdfPtr);
        }
        auto samplePartSdf = [&](const openvdb::Vec3d& wp) -> float {
            const float fVal = sdfSampler.wsSample(wp);
            if (coarseSdfSamplerPtr && rec.fineShellInnerDepthMm > 0.0f &&
                fVal <= -(rec.fineShellInnerDepthMm - 1.5f * dx)) {
                return coarseSdfSamplerPtr->wsSample(wp);
            }
            return fVal;
        };

        const float zGapMm = std::max(0.05f, impl_->supportSpec.z_gap_mm);
        const float xyGapMm = std::max(0.10f, impl_->supportSpec.xy_gap_mm);
        const float targetLayerH = std::max(0.05f, impl_->sliceLayerHeight.target_mm);

        // Resolve the tool used for support extrusion (`support_tool_index >= 0` selects a specific
        // tool from the multi-tool registry; `< 0` defaults to `activeToolIndex`).
        std::size_t supportToolIdx = impl_->activeToolIndex;
        if (impl_->supportSpec.support_tool_index >= 0 && !impl_->tools.empty()) {
            supportToolIdx = std::min<std::size_t>(
                static_cast<std::size_t>(impl_->supportSpec.support_tool_index),
                impl_->tools.size() - 1);
        } else if (!impl_->tools.empty()) {
            supportToolIdx = std::min<std::size_t>(impl_->activeToolIndex, impl_->tools.size() - 1);
        }
        const float supportToolNozzleDiamMm =
            (!impl_->tools.empty() && supportToolIdx < impl_->tools.size())
                ? std::max(0.10f, impl_->tools[supportToolIdx].nozzle_diameter_mm)
                : std::max(0.10f, impl_->shellPathSpec.nozzle_diameter_mm);
        const float lineWidthMm = std::max(
            0.15f,
            (impl_->supportSpec.line_width_mm > 0.05f)
                ? impl_->supportSpec.line_width_mm
                : supportToolNozzleDiamMm);

        // Build a 3D spatial hash of all generated object extrusion path vertices & segment midpoints
        // (`feature_kind <= 6`) so support generation enforces direct 3D bead-to-bead collision avoidance
        // against actual object extrusions (walls, infill, bridges, and wave overhangs) in addition to
        // the voxelized part SDF!
        const float objGridCellMm = 1.50f;
        struct ObjBeadSample {
            float x = 0.0f;
            float y = 0.0f;
            float z = 0.0f;
            float halfW = 0.21f;
            float halfH = 0.10f;
        };
        auto packGrid3D = [](std::int32_t ix, std::int32_t iy, std::int32_t iz) -> std::uint64_t {
            const std::uint64_t ux = static_cast<std::uint64_t>(static_cast<std::uint32_t>(ix)) & 0x1FFFFFULL;
            const std::uint64_t uy = static_cast<std::uint64_t>(static_cast<std::uint32_t>(iy)) & 0x1FFFFFULL;
            const std::uint64_t uz = static_cast<std::uint64_t>(static_cast<std::uint32_t>(iz)) & 0x1FFFFFULL;
            return (ux << 42) | (uy << 21) | uz;
        };
        std::unordered_map<std::uint64_t, std::vector<ObjBeadSample>> objectExtrusionGrid;
        auto insertObjSample = [&](float px, float py, float pz, float hw, float hh) {
            const auto ix = static_cast<std::int32_t>(std::floor(px / objGridCellMm));
            const auto iy = static_cast<std::int32_t>(std::floor(py / objGridCellMm));
            const auto iz = static_cast<std::int32_t>(std::floor(pz / objGridCellMm));
            objectExtrusionGrid[packGrid3D(ix, iy, iz)].push_back(ObjBeadSample{px, py, pz, hw, hh});
        };
        for (const auto& lyr : res.layers) {
            for (const auto& op : lyr.paths) {
                if (op.feature_kind > 6 || op.points.empty()) {
                    continue;
                }
                for (std::size_t k = 0; k < op.points.size(); ++k) {
                    const auto& pt = op.points[k];
                    const float hw = 0.5f * std::max(0.15f, pt.width_mm);
                    const float hh = 0.5f * std::max(0.05f, pt.height_mm);
                    insertObjSample(pt.x, pt.y, pt.z, hw, hh);
                    if (k > 0) {
                        const auto& prev = op.points[k - 1];
                        const float dxSeg = pt.x - prev.x;
                        const float dySeg = pt.y - prev.y;
                        const float dzSeg = pt.z - prev.z;
                        const float segLen = std::sqrt(dxSeg * dxSeg + dySeg * dySeg + dzSeg * dzSeg);
                        if (segLen > 0.65f) {
                            const int nSub = std::min(8, static_cast<int>(std::ceil(segLen / 0.65f)));
                            for (int s = 1; s < nSub; ++s) {
                                const float frac = static_cast<float>(s) / static_cast<float>(nSub);
                                insertObjSample(
                                    prev.x + frac * dxSeg,
                                    prev.y + frac * dySeg,
                                    prev.z + frac * dzSeg,
                                    hw,
                                    hh);
                            }
                        }
                    }
                }
            }
        }

        // Direct 3D collision check between a candidate support bead at `pBed` (with half-width
        // `supHalfW` and half-height `supHalfH`) and all nearby object extrusion beads.
        // Decomposes separation along anti-gravity `uUp = -\hat{g}_{bed}` (AGENTS.md Rule 2):
        // - Same-layer / overlapping-height beads (`dh < supHalfH + objHalfH + 0.02 mm`) must maintain
        //   center-to-center horizontal separation >= `supHalfW + objHalfW + max(0.08 mm, 0.65 * xyGapMm)`.
        // - Vertically adjacent beads (`dh < 0.45 * zGapMm + supHalfH + objHalfH`) must not overlap
        //   when not in a roof contact zone, and even in a roof contact zone must never physically
        //   interpenetrate (`dh < 0.85 * (supHalfH + objHalfH)` with `dHoriz < supHalfW + objHalfW`).
        auto hasObjectExtrusionClearance = [&](const std::array<float, 3>& pBed,
                                               float supHalfW,
                                               float supHalfH,
                                               bool isRoofContactZone) -> bool {
            if (objectExtrusionGrid.empty()) {
                return true;
            }
            const auto cx = static_cast<std::int32_t>(std::floor(pBed[0] / objGridCellMm));
            const auto cy = static_cast<std::int32_t>(std::floor(pBed[1] / objGridCellMm));
            const auto cz = static_cast<std::int32_t>(std::floor(pBed[2] / objGridCellMm));
            const float latGap = std::max(0.08f, 0.65f * xyGapMm);

            for (std::int32_t dz = -1; dz <= 1; ++dz) {
                for (std::int32_t dy = -1; dy <= 1; ++dy) {
                    for (std::int32_t dxCell = -1; dxCell <= 1; ++dxCell) {
                        auto it = objectExtrusionGrid.find(packGrid3D(cx + dxCell, cy + dy, cz + dz));
                        if (it == objectExtrusionGrid.end()) {
                            continue;
                        }
                        for (const auto& ob : it->second) {
                            const float rx = ob.x - pBed[0];
                            const float ry = ob.y - pBed[1];
                            const float rz = ob.z - pBed[2];
                            const float dh = std::abs(rx * uUp[0] + ry * uUp[1] + rz * uUp[2]);
                            const float vSumH = supHalfH + ob.halfH;
                            if (dh >= vSumH + zGapMm) {
                                continue;
                            }
                            const float dSq = rx * rx + ry * ry + rz * rz;
                            const float dHoriz = std::sqrt(std::max(0.0f, dSq - dh * dh));
                            const float hSumW = supHalfW + ob.halfW;

                            // 1. Same-layer lateral collision (never allowed, even in roof contact zone)
                            if (dh < 0.88f * vSumH) {
                                if (dHoriz < hSumW + latGap) {
                                    return false;
                                }
                            } else if (!isRoofContactZone && dh < vSumH + 0.45f * zGapMm) {
                                // 2. Non-roof vertical proximity collision
                                if (dHoriz < hSumW + 0.35f * xyGapMm) {
                                    return false;
                                }
                            }
                        }
                    }
                }
            }
            return true;
        };

        auto uvHToBed = [&](float uVal, float vVal, float hVal) -> std::array<float, 3> {
            const float worldH = hVal + minObjH;
            return {
                pOrigin[0] + uVal * tU[0] + vVal * tV[0] + worldH * uUp[0],
                pOrigin[1] + uVal * tU[1] + vVal * tV[1] + worldH * uUp[1],
                pOrigin[2] + uVal * tU[2] + vVal * tV[2] + worldH * uUp[2]};
        };
        auto sampleSdfAtUVH = [&](float uVal, float vVal, float hVal) -> float {
            const auto p = uvHToBed(uVal, vVal, hVal);
            return samplePartSdf(openvdb::Vec3d(p[0], p[1], p[2]));
        };

        // Directional clearance check in (u, v, h) coordinates:
        // - Enforces direct 3D bead-to-bead clearance against generated object extrusion paths.
        // - Enforces `minRoofClearance` (~0.75 * zGapMm) vertically against downward-facing
        //   overhang ceilings (and upward-facing floors when `on_bed_only == false`).
        // - Enforces `xyGapMm + 0.5f * lineWidthMm + extraHorizR` horizontally against vertical/sloped
        //   side walls (so the outer edge of the support bead respects `xyGapMm`).
        auto hasSupportClearanceAtUVH = [&](float uVal,
                                            float vVal,
                                            float hVal,
                                            float extraHorizR,
                                            bool isRoofContactZone) -> bool {
            const std::array<float, 3> pBed = uvHToBed(uVal, vVal, hVal);
            if (!hasObjectExtrusionClearance(
                    pBed,
                    0.5f * lineWidthMm + extraHorizR,
                    0.5f * targetLayerH,
                    isRoofContactZone)) {
                return false;
            }
            const float sHere = samplePartSdf(openvdb::Vec3d(pBed[0], pBed[1], pBed[2]));
            const float minRoofClearance = std::min(0.75f * zGapMm, xyGapMm);
            if (sHere < minRoofClearance) {
                return false;
            }
            const float reqHoriz = xyGapMm + 0.5f * lineWidthMm + extraHorizR;
            if (sHere >= reqHoriz) {
                return true;
            }
            const float eps = std::max(0.5f * dx, 0.25f);
            const float gu =
                sampleSdfAtUVH(uVal + eps, vVal, hVal) - sampleSdfAtUVH(uVal - eps, vVal, hVal);
            const float gv =
                sampleSdfAtUVH(uVal, vVal + eps, hVal) - sampleSdfAtUVH(uVal, vVal - eps, hVal);
            const float gh =
                sampleSdfAtUVH(uVal, vVal, hVal + eps) - sampleSdfAtUVH(uVal, vVal, hVal - eps);
            const float gNorm = std::sqrt(gu * gu + gv * gv + gh * gh);
            if (gNorm <= 1.0e-5f) {
                return isRoofContactZone && (sHere >= minRoofClearance);
            }
            const float nUp = gh / gNorm; // < 0 beneath overhang ceiling, > 0 above floor
            const float nHoriz = std::sqrt(std::max(0.0f, 1.0f - nUp * nUp));

            if (impl_->supportSpec.on_bed_only && nUp > 0.30f) {
                return false;
            }

            if (nUp < -0.38f || (!impl_->supportSpec.on_bed_only && nUp > 0.38f)) {
                // Predominantly vertical contact with overhang ceiling (or allowed floor).
                // Verify we are not simultaneously within `xyGapMm + 0.5f * lineWidthMm` of a vertical
                // side wall that extends below `hVal - zGapMm`.
                const float hNorm = std::sqrt(gu * gu + gv * gv);
                if (hNorm > 0.35f * gNorm && hVal > zGapMm + 0.10f) {
                    const float wallStep = xyGapMm + 0.5f * lineWidthMm;
                    const float uWall = uVal - wallStep * (gu / hNorm);
                    const float vWall = vVal - wallStep * (gv / hNorm);
                    const float sWall =
                        sampleSdfAtUVH(uWall, vWall, std::max(0.05f, hVal - zGapMm));
                    if (sWall <= 0.0f) {
                        return false;
                    }
                }
                return true;
            }

            const float reqDirectional = reqHoriz * nHoriz + minRoofClearance * std::abs(nUp);
            return sHere >= reqDirectional;
        };

        auto considerSupportVoxel = [&](const openvdb::Vec3d& wp, float supVal, float sdfVal) {
            if (supVal < 0.50f || std::abs(sdfVal) > 1.50f * dx) {
                return;
            }
            const float relX = static_cast<float>(wp.x()) - pOrigin[0];
            const float relY = static_cast<float>(wp.y()) - pOrigin[1];
            const float relZ = static_cast<float>(wp.z()) - pOrigin[2];
            const float uCoord = relX * tU[0] + relY * tU[1] + relZ * tU[2];
            const float vCoord = relX * tV[0] + relY * tV[1] + relZ * tV[2];
            const float hCoord = (relX * uUp[0] + relY * uUp[1] + relZ * uUp[2]) - minObjH;

            if (hCoord <= zGapMm + targetLayerH) {
                return;
            }

            // Locate the exact \phi = 0 downward-facing overhang ceiling surface `hCeiling`
            // along the vertical column `(uCoord, vCoord)`, rejecting horizontally extrapolated
            // narrow-band voxels that have no solid part overhang above them.
            const float marchStep = std::min(0.45f * dx, 0.25f);
            float hCeiling = -1.0f;
            if (sdfVal <= 0.0f) {
                float hPrev = hCoord;
                float sPrev = sdfVal;
                const float minSearchH = std::max(0.0f, hCoord - 2.5f * dx);
                for (float hProbe = hCoord - marchStep; hProbe >= minSearchH; hProbe -= marchStep) {
                    const float sProbe = sampleSdfAtUVH(uCoord, vCoord, hProbe);
                    if (sProbe > 0.0f) {
                        const float denom = std::max(1.0e-5f, sProbe - sPrev);
                        const float t = std::clamp(-sPrev / denom, 0.0f, 1.0f);
                        hCeiling = hPrev + t * (hProbe - hPrev);
                        break;
                    }
                    hPrev = hProbe;
                    sPrev = sProbe;
                }
            } else {
                float hPrev = hCoord;
                float sPrev = sdfVal;
                const float maxSearchH = hCoord + 1.85f * dx;
                for (float hProbe = hCoord + marchStep; hProbe <= maxSearchH; hProbe += marchStep) {
                    const float sProbe = sampleSdfAtUVH(uCoord, vCoord, hProbe);
                    if (sProbe <= 0.0f) {
                        const float denom = std::max(1.0e-5f, sPrev - sProbe);
                        const float t = std::clamp(sPrev / denom, 0.0f, 1.0f);
                        hCeiling = hPrev + t * (hProbe - hPrev);
                        break;
                    }
                    hPrev = hProbe;
                    sPrev = sProbe;
                }
            }
            if (hCeiling <= zGapMm + targetLayerH) {
                return;
            }

            // Place `roofH` at vertical distance `zGapMm` below the exact \phi = 0 overhang surface,
            // and nudge slightly lower on steep tilted overhangs so 3D surface clearance >= 0.85 * zGapMm.
            float roofH = hCeiling - zGapMm;
            const float sAtRoof = sampleSdfAtUVH(uCoord, vCoord, roofH);
            if (sAtRoof <= 0.0f) {
                return;
            }
            if (sAtRoof < 0.85f * zGapMm) {
                roofH -= (0.90f * zGapMm - sAtRoof);
                if (sampleSdfAtUVH(uCoord, vCoord, roofH) < 0.70f * zGapMm) {
                    return;
                }
            }

            // Ray-march downward from `roofH` along gravity `+\hat{g}_{bed} = -uUp` to find `floorH`
            float floorH = 0.0f;
            float hPrev = roofH;
            float sPrev = sampleSdfAtUVH(uCoord, vCoord, roofH);
            for (float hProbe = roofH - marchStep; hProbe > 0.0f; hProbe -= marchStep) {
                const float sProbe = sampleSdfAtUVH(uCoord, vCoord, hProbe);
                if (sProbe <= 0.0f) {
                    const float denom = std::max(1.0e-5f, sPrev - sProbe);
                    const float t = std::clamp(sPrev / denom, 0.0f, 1.0f);
                    const float hFloorSurf = hPrev + t * (hProbe - hPrev);
                    floorH = hFloorSurf + zGapMm;
                    break;
                }
                hPrev = hProbe;
                sPrev = sProbe;
            }
            if (roofH <= floorH + 0.5f * targetLayerH) {
                return;
            }
            // In ColumnarGrid mode with `on_bed_only` enabled, never place support columns on
            // upper part surfaces (`floorH > 0`). In NonPlanarTree mode (`support_style == 1`),
            // keep the overhang contact cell even when `floorH > 0` so tree branches can grow
            // from `(uCoord, vCoord, roofH)` and steer laterally around the lower model surface
            // down to the build plate (`h = 0`).
            if (impl_->supportSpec.support_style == 0 &&
                impl_->supportSpec.on_bed_only &&
                floorH > 1.0e-3f) {
                return;
            }

            const auto iu = static_cast<std::int32_t>(std::llround(uCoord / cellStepMm));
            const auto iv = static_cast<std::int32_t>(std::llround(vCoord / cellStepMm));
            const std::uint64_t key = packCellKey(iu, iv);
            auto itCol = supportColumns.find(key);
            if (itCol == supportColumns.end()) {
                supportColumns.emplace(key, ColumnSpan{roofH, floorH});
            } else {
                if (std::abs(roofH - itCol->second.roofH) <= 2.5f * dx) {
                    // Same overhang surface: keep the conservative (lower) roofH so no voxel in
                    // the cell violates `zGapMm` against a sloped overhang ceiling.
                    itCol->second.roofH = std::min(itCol->second.roofH, roofH);
                    itCol->second.floorH = std::max(itCol->second.floorH, floorH);
                } else if (roofH > itCol->second.roofH) {
                    itCol->second.roofH = roofH;
                    itCol->second.floorH = std::min(itCol->second.floorH, floorH);
                }
            }
        };

        for (auto iter = supGrid.cbeginValueOn(); iter; ++iter) {
            const float supVal = *iter;
            if (supVal < 0.15f) {
                continue;
            }
            const openvdb::Coord ijk = iter.getCoord();
            const openvdb::Vec3d wp = supGrid.transform().indexToWorld(ijk);
            const float sdfVal = sdfAcc.getValue(ijk);
            considerSupportVoxel(wp, supVal, sdfVal);
        }

        if (!supportColumns.empty()) {
            std::int32_t minIU = std::numeric_limits<std::int32_t>::max();
            std::int32_t maxIU = std::numeric_limits<std::int32_t>::lowest();
            std::int32_t minIV = std::numeric_limits<std::int32_t>::max();
            std::int32_t maxIV = std::numeric_limits<std::int32_t>::lowest();
            for (const auto& [key, span] : supportColumns) {
                std::int32_t iu = 0, iv = 0;
                unpackCellKey(key, iu, iv);
                minIU = std::min(minIU, iu);
                maxIU = std::max(maxIU, iu);
                minIV = std::min(minIV, iv);
                maxIV = std::max(maxIV, iv);
            }

            const float baseSpacingMm = std::max(0.4f, impl_->supportSpec.base_spacing_mm);
            // `interface_spacing_mm` is the edge-to-edge gap (>= 0.0 mm) between adjacent support
            // interface extrusions (`0.0 mm` = 100% solid touching interface roof/floor with pitch == `lineWidthMm`).
            const float interfaceGapMm = std::max(0.0f, impl_->supportSpec.interface_spacing_mm);
            const float interfaceSpacingMm = lineWidthMm + interfaceGapMm;
            const float minSegLenMm = std::max(0.4f, impl_->shellPathSpec.min_extrusion_length_mm);

            std::vector<float> uniqueLayerIsos;
            uniqueLayerIsos.reserve(res.layers.size());
            for (const auto& lyr : res.layers) {
                if (lyr.isovalue <= 0.0f) {
                    continue;
                }
                if (uniqueLayerIsos.empty() ||
                    std::abs(lyr.isovalue - uniqueLayerIsos.back()) >= 1.0e-3f) {
                    uniqueLayerIsos.push_back(lyr.isovalue);
                }
            }
            const float firstSupportIso =
                uniqueLayerIsos.empty() ? targetLayerH : uniqueLayerIsos.front();

            // =========================================================================
            // Non-Planar Tree Support Growth, Obstacle-Avoiding Routing & Coalescence
            // =========================================================================
            struct TreeStation {
                float h = 0.0f;
                float u = 0.0f;
                float v = 0.0f;
                float radius = 1.0f;
                std::array<float, 3> tangentUp{0.0f, 0.0f, 1.0f};
            };
            struct TreeBranch {
                float roofH = 0.0f;
                float floorH = 0.0f;
                bool groundedOnBed = false;
                std::int32_t mergedIntoBranch = -1;
                std::vector<float> forkHeights;
                std::vector<TreeStation> stations; // Ordered top-down (decreasing `h`)
            };
            std::vector<TreeBranch> treeBranches;

            struct CanopyCellLink {
                std::int32_t iu = 0;
                std::int32_t iv = 0;
                float u = 0.0f;
                float v = 0.0f;
                float roofH = 0.0f;
                float floorH = 0.0f;
                std::size_t branchIdx = 0;
            };
            std::vector<CanopyCellLink> treeCanopyCells;
            float treeCanopyTransitionMm = 2.0f;

            const bool useTreeSupport = (impl_->supportSpec.support_style == 1);
            if (useTreeSupport) {
                const bool onBedOnly = impl_->supportSpec.on_bed_only;
                const float branchAngleRad =
                    std::clamp(impl_->supportSpec.tree_branch_angle_deg, 15.0f, 65.0f) *
                    (3.14159265358979323846f / 180.0f);
                const float tanBranchMax = std::tan(branchAngleRad);
                const float tipRadiusMm = std::max(
                    lineWidthMm,
                    0.5f * std::clamp(impl_->supportSpec.tree_tip_diameter_mm, 0.80f, 8.00f));
                const float maxTrunkRadiusMm = std::max(
                    tipRadiusMm,
                    0.5f * std::clamp(impl_->supportSpec.tree_trunk_max_diameter_mm, 2.00f, 30.00f));

                // 1. Cluster overhang roof cells into Tree Tip Seeds
                struct RawRoofCell {
                    std::int32_t iu = 0;
                    std::int32_t iv = 0;
                    float u = 0.0f;
                    float v = 0.0f;
                    float roofH = 0.0f;
                    float floorH = 0.0f;
                };
                std::vector<RawRoofCell> rawCells;
                rawCells.reserve(supportColumns.size());
                float maxRoofH = 0.0f;
                for (const auto& [key, span] : supportColumns) {
                    std::int32_t iu = 0, iv = 0;
                    unpackCellKey(key, iu, iv);
                    rawCells.push_back(RawRoofCell{
                        iu,
                        iv,
                        static_cast<float>(iu) * cellStepMm,
                        static_cast<float>(iv) * cellStepMm,
                        span.roofH,
                        span.floorH});
                    maxRoofH = std::max(maxRoofH, span.roofH);
                }
                std::sort(
                    rawCells.begin(),
                    rawCells.end(),
                    [](const RawRoofCell& a, const RawRoofCell& b) {
                        return a.roofH > b.roofH;
                    });

                struct TipSeed {
                    float sumU = 0.0f;
                    float sumV = 0.0f;
                    float roofH = 0.0f;
                    float floorH = 0.0f;
                    std::uint32_t count = 0;
                    bool spawned = false;
                };
                const float clusterRadiusMm = std::max(1.5f * cellStepMm, 1.35f * tipRadiusMm);
                const float clusterRadiusSq = clusterRadiusMm * clusterRadiusMm;
                treeCanopyTransitionMm =
                    std::max(1.5f * targetLayerH, clusterRadiusMm / std::max(0.35f, tanBranchMax));
                std::vector<TipSeed> tipSeeds;
                for (const auto& rc : rawCells) {
                    bool mergedToSeed = false;
                    for (auto& ts : tipSeeds) {
                        const float cu = ts.sumU / static_cast<float>(ts.count);
                        const float cv = ts.sumV / static_cast<float>(ts.count);
                        const float du = rc.u - cu;
                        const float dv = rc.v - cv;
                        if (du * du + dv * dv <= clusterRadiusSq &&
                            std::abs(rc.roofH - ts.roofH) <= 3.0f * targetLayerH) {
                            ts.sumU += rc.u;
                            ts.sumV += rc.v;
                            ts.roofH = std::max(ts.roofH, rc.roofH);
                            ts.floorH = std::min(ts.floorH, rc.floorH);
                            ++ts.count;
                            mergedToSeed = true;
                            break;
                        }
                    }
                    if (!mergedToSeed) {
                        tipSeeds.push_back(TipSeed{rc.u, rc.v, rc.roofH, rc.floorH, 1, false});
                    }
                }

                // 2. Top-down step-by-step tree growth along gravity `+\hat{g}_{bed}` (`-uUp`)
                struct ActiveNode {
                    std::size_t branchIdx = 0;
                    float u = 0.0f;
                    float v = 0.0f;
                    float radius = 1.0f;
                    float targetRadius = 1.0f;
                    float roofH = 0.0f;
                    float floorH = 0.0f;
                    float origFloorH = 0.0f;
                    bool alive = true;
                };
                std::vector<ActiveNode> activeNodes;

                // Evaluate lookahead clearance along the downward gravity ray so branches
                // start steering around lower part ledges well before colliding with them,
                // without mistaking the overhang ceiling above `nodeRoofH` for a lateral wall.
                auto evaluateLookaheadScore = [&](float uCand,
                                                  float vCand,
                                                  float hCand,
                                                  float rCand,
                                                  float nodeRoofH) -> float {
                    const bool inRoofZone =
                        (nodeRoofH - hCand) <= (3.5f * targetLayerH + zGapMm);
                    const float sNow = sampleSdfAtUVH(uCand, vCand, hCand);
                    const float rCenterClear = std::min(rCand, tipRadiusMm);
                    if (!hasSupportClearanceAtUVH(
                            uCand,
                            vCand,
                            hCand,
                            inRoofZone ? 0.0f : 0.35f * rCenterClear,
                            inRoofZone)) {
                        return -1000.0f + sNow;
                    }
                    const float reqClearance = xyGapMm + 0.65f * rCenterClear;
                    const float roofVertDist = std::max(zGapMm, nodeRoofH - hCand + zGapMm);
                    const float effNow =
                        (sNow < reqClearance && sNow >= 0.72f * roofVertDist)
                            ? reqClearance
                            : sNow;
                    float minLookSdf = effNow;
                    const std::array<float, 5> lookaheadOffsetsMm{1.2f, 2.8f, 5.5f, 9.5f, 15.0f};
                    for (const float dhLook : lookaheadOffsetsMm) {
                        const float hProbe = hCand - dhLook;
                        if (hProbe <= 0.1f) {
                            break;
                        }
                        if (!onBedOnly && hProbe <= 0.5f * targetLayerH) {
                            break;
                        }
                        const float sLook = sampleSdfAtUVH(uCand, vCand, hProbe);
                        const float probeRoofDist =
                            std::max(zGapMm, nodeRoofH - hProbe + zGapMm);
                        if (sLook < 0.80f * probeRoofDist) {
                            minLookSdf = std::min(minLookSdf, sLook);
                        }
                    }
                    float perimBonus = 0.0f;
                    if (!inRoofZone) {
                        const float rProbe =
                            std::min(maxTrunkRadiusMm, std::max(rCand, tipRadiusMm + 0.8f));
                        const float sE = sampleSdfAtUVH(uCand + rProbe, vCand, hCand);
                        const float sW = sampleSdfAtUVH(uCand - rProbe, vCand, hCand);
                        const float sN = sampleSdfAtUVH(uCand, vCand + rProbe, hCand);
                        const float sS = sampleSdfAtUVH(uCand, vCand - rProbe, hCand);
                        const float minPerimSdf = std::min({sE, sW, sN, sS});
                        perimBonus = 0.35f * std::clamp(minPerimSdf - xyGapMm, -2.0f, 1.5f);
                    }
                    float score = std::min(effNow - reqClearance, 3.0f) +
                                  1.35f * std::min(minLookSdf - reqClearance, 4.0f) +
                                  perimBonus;
                    return score;
                };

                // Check whether a trunk disk of radius `rTest` centered at `(uCenter, vCenter)`
                // at height `hVal` has full perimeter clearance around the part. Evaluating
                // `hasSupportClearanceAtUVH` on the perimeter avoids narrow-band SDF saturation
                // at the trunk center so trunks can expand up to `maxTrunkRadiusMm`.
                auto isTrunkPerimeterClear = [&](float uCenter,
                                                 float vCenter,
                                                 float hVal,
                                                 float rTest,
                                                 bool roofZone) -> bool {
                    if (rTest <= 0.5f * lineWidthMm) {
                        return hasSupportClearanceAtUVH(uCenter, vCenter, hVal, 0.0f, roofZone);
                    }
                    const float minRingR = std::min(rTest, tipRadiusMm);
                    for (float rRing = rTest; rRing >= minRingR - 1.0e-4f; rRing -= 0.70f) {
                        const float rSample = std::max(minRingR, rRing);
                        for (int k = 0; k < 24; ++k) {
                            const float ang =
                                static_cast<float>(k) * (2.0f * 3.14159265358979323846f / 24.0f);
                            const float uPt = uCenter + rSample * std::cos(ang);
                            const float vPt = vCenter + rSample * std::sin(ang);
                            if (!hasSupportClearanceAtUVH(uPt, vPt, hVal, 0.0f, roofZone)) {
                                return false;
                            }
                        }
                        if (rSample <= minRingR + 1.0e-4f) {
                            break;
                        }
                    }
                    return true;
                };

                const float dhStep = std::clamp(targetLayerH, 0.15f, 0.35f);
                const float topAnchorBandMm =
                    static_cast<float>(std::max<std::uint32_t>(1, impl_->supportSpec.interface_layers)) *
                    targetLayerH;
                const int numSteps = static_cast<int>(std::ceil(maxRoofH / dhStep)) + 1;
                for (int step = 0; step <= numSteps; ++step) {
                    const float hCurr = std::max(0.0f, maxRoofH - static_cast<float>(step) * dhStep);
                    const float hNext = std::max(0.0f, hCurr - dhStep);
                    const float dhActual = hCurr - hNext;

                    // Spawn any tip seeds whose roofH is reached at `hCurr`
                    for (auto& ts : tipSeeds) {
                        if (!ts.spawned && ts.roofH >= hCurr - 1.0e-4f) {
                            ts.spawned = true;
                            const float u0 = ts.sumU / static_cast<float>(ts.count);
                            const float v0 = ts.sumV / static_cast<float>(ts.count);
                            const float cellArea = static_cast<float>(ts.count) * cellStepMm * cellStepMm;
                            const float areaRadius = std::sqrt(cellArea / 3.14159265f);
                            const float r0 = tipRadiusMm;
                            const float rTarget0 = std::clamp(
                                std::max(tipRadiusMm, 0.85f * areaRadius),
                                tipRadiusMm,
                                std::min(maxTrunkRadiusMm, 1.8f * tipRadiusMm));

                            TreeBranch br{};
                            br.roofH = ts.roofH;
                            br.floorH = onBedOnly ? 0.0f : ts.floorH;
                            br.stations.push_back(TreeStation{
                                ts.roofH,
                                u0,
                                v0,
                                r0,
                                uUp});
                            if (ts.roofH > hCurr + 1.0e-3f) {
                                br.stations.push_back(TreeStation{
                                    hCurr,
                                    u0,
                                    v0,
                                    r0,
                                    uUp});
                            }
                            const std::size_t bIdx = treeBranches.size();
                            treeBranches.push_back(std::move(br));
                            activeNodes.push_back(ActiveNode{
                                bIdx,
                                u0,
                                v0,
                                r0,
                                rTarget0,
                                ts.roofH,
                                onBedOnly ? 0.0f : ts.floorH,
                                ts.floorH,
                                true});
                        }
                    }

                    if (dhActual <= 1.0e-5f) {
                        // Reached the build plate (`hCurr == 0.0f`)
                        for (auto& node : activeNodes) {
                            if (node.alive) {
                                treeBranches[node.branchIdx].groundedOnBed = true;
                            }
                        }
                        break;
                    }

                    const float maxLatStep = dhActual * tanBranchMax;

                    // Step 2a: Obstacle-avoiding steering around the part SDF
                    for (std::size_t nodeIdx = 0; nodeIdx < activeNodes.size(); ++nodeIdx) {
                        auto& node = activeNodes[nodeIdx];
                        if (!node.alive) {
                            continue;
                        }
                        if (!onBedOnly && hNext <= node.floorH + 1.0e-4f) {
                            // Landed on lower part surface (allowed when `on_bed_only == false`)
                            treeBranches[node.branchIdx].stations.push_back(TreeStation{
                                node.floorH,
                                node.u,
                                node.v,
                                node.radius,
                                uUp});
                            node.alive = false;
                            continue;
                        }

                        float bestU = node.u;
                        float bestV = node.v;
                        float usedLatStep = 0.0f;

                        // Keep the branch tip stationary directly beneath the overhang across the
                        // top roof interface layers unless an immediate obstacle below requires early escape.
                        const float maxAnchorBand =
                            (onBedOnly && node.origFloorH > 1.0e-3f)
                                ? std::min(topAnchorBandMm, 0.20f * std::max(0.0f, node.roofH - node.origFloorH))
                                : topAnchorBandMm;
                        const bool holdTipUnderRoof =
                            ((node.roofH - hNext) <= maxAnchorBand) &&
                            hasSupportClearanceAtUVH(node.u, node.v, hNext, 0.0f, true);

                        const float straightScore =
                            evaluateLookaheadScore(node.u, node.v, hNext, node.radius, node.roofH);
                        if (holdTipUnderRoof && straightScore > -500.0f) {
                            usedLatStep = maxLatStep;
                        } else if (straightScore < 0.35f) {
                            // Need obstacle-avoiding lateral steering!
                            float bestScore = straightScore;
                            // Estimate 2D horizontal SDF gradient at (node.u, node.v, hNext)
                            const float eps = std::max(0.5f * dx, 0.35f);
                            const float gx =
                                sampleSdfAtUVH(node.u + eps, node.v, hNext) -
                                sampleSdfAtUVH(node.u - eps, node.v, hNext);
                            const float gy =
                                sampleSdfAtUVH(node.u, node.v + eps, hNext) -
                                sampleSdfAtUVH(node.u, node.v - eps, hNext);
                            const float gNorm = std::sqrt(gx * gx + gy * gy);

                            auto tryCandidate = [&](float cu, float cv, float stepLen) {
                                const float sc =
                                    evaluateLookaheadScore(cu, cv, hNext, node.radius, node.roofH);
                                if (sc > bestScore + 1.0e-4f) {
                                    bestScore = sc;
                                    bestU = cu;
                                    bestV = cv;
                                    usedLatStep = stepLen;
                                }
                            };

                            if (gNorm > 1.0e-5f) {
                                const float dirU = gx / gNorm;
                                const float dirV = gy / gNorm;
                                tryCandidate(
                                    node.u + maxLatStep * dirU,
                                    node.v + maxLatStep * dirV,
                                    maxLatStep);
                            }

                            // Also probe steering directly toward nearby clear active branches
                            for (const auto& other : activeNodes) {
                                if (!other.alive || other.branchIdx == node.branchIdx) {
                                    continue;
                                }
                                const float du = other.u - node.u;
                                const float dv = other.v - node.v;
                                const float d = std::sqrt(du * du + dv * dv);
                                if (d > 1.0e-4f &&
                                    d <= std::max(10.0f, 4.0f * (node.radius + other.radius))) {
                                    const float stepLen = std::min(maxLatStep, d);
                                    tryCandidate(
                                        node.u + stepLen * (du / d),
                                        node.v + stepLen * (dv / d),
                                        stepLen);
                                }
                            }

                            // Probe 16 directions at 0.6x and 1.0x `maxLatStep` (and 1.35x in emergency when `onBedOnly`)
                            const std::array<float, 3> radialFracs{
                                0.60f,
                                1.00f,
                                onBedOnly ? 1.35f : 1.00f};
                            for (const float frac : radialFracs) {
                                const float rStep = frac * maxLatStep;
                                for (int k = 0; k < 16; ++k) {
                                    const float ang =
                                        static_cast<float>(k) * (2.0f * 3.14159265358979323846f / 16.0f);
                                    tryCandidate(
                                        node.u + rStep * std::cos(ang),
                                        node.v + rStep * std::sin(ang),
                                        rStep);
                                }
                            }

                            // If local steps still sit above a wide flat lower obstacle (where the 3D SDF
                            // gradient inside the lower shelf points vertically rather than horizontally),
                            // find the shortest horizontal escape vector to a clear vertical corridor!
                            if (bestScore < 0.35f) {
                                float hBlocked = -1.0f;
                                const float reqClear =
                                    xyGapMm + 0.65f * std::min(node.radius, tipRadiusMm);
                                for (float hProbe = hNext - 0.8f; hProbe >= 0.25f; hProbe -= 0.8f) {
                                    const float probeRoofDist =
                                        std::max(zGapMm, node.roofH - hProbe + zGapMm);
                                    const float sProbe = sampleSdfAtUVH(node.u, node.v, hProbe);
                                    if (sProbe < reqClear && sProbe < 0.80f * probeRoofDist) {
                                        hBlocked = hProbe;
                                        break;
                                    }
                                }
                                if (hBlocked >= 0.0f) {
                                    bool foundCorridor = false;
                                    for (float rSearch = 1.0f; rSearch <= 28.0f && !foundCorridor; rSearch += 1.0f) {
                                        float bestCorridorSdf = -1.0e9f;
                                        float escDirU = 0.0f;
                                        float escDirV = 0.0f;
                                        for (int k = 0; k < 16; ++k) {
                                            const float ang =
                                                static_cast<float>(k) * (2.0f * 3.14159265358979323846f / 16.0f);
                                            const float dirU = std::cos(ang);
                                            const float dirV = std::sin(ang);
                                            const float uEsc = node.u + rSearch * dirU;
                                            const float vEsc = node.v + rSearch * dirV;
                                            const float sBlock = sampleSdfAtUVH(uEsc, vEsc, hBlocked);
                                            const float sBed = sampleSdfAtUVH(
                                                uEsc,
                                                vEsc,
                                                std::max(0.25f, 0.5f * targetLayerH));
                                            const float minCorridor = std::min(sBlock, sBed);
                                            if (minCorridor >= reqClear && minCorridor > bestCorridorSdf) {
                                                bestCorridorSdf = minCorridor;
                                                escDirU = dirU;
                                                escDirV = dirV;
                                                foundCorridor = true;
                                            }
                                        }
                                        if (foundCorridor) {
                                            const float stepLen = std::min(maxLatStep, rSearch);
                                            const float cu = node.u + stepLen * escDirU;
                                            const float cv = node.v + stepLen * escDirV;
                                            const bool inRoofZone =
                                                (node.roofH - hNext) <= (3.5f * targetLayerH + zGapMm);
                                            if (hasSupportClearanceAtUVH(cu, cv, hNext, 0.0f, inRoofZone)) {
                                                bestU = cu;
                                                bestV = cv;
                                                usedLatStep = stepLen;
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        // Verify directional clearance at `(bestU, bestV, hNext)`
                        const bool inRoofZone =
                            (node.roofH - hNext) <= (3.5f * targetLayerH + zGapMm);
                        if (!hasSupportClearanceAtUVH(bestU, bestV, hNext, 0.0f, inRoofZone)) {
                            // Blocked by part geometry even after lateral steering
                            if (!onBedOnly) {
                                treeBranches[node.branchIdx].stations.push_back(TreeStation{
                                    std::max(hNext, node.floorH),
                                    node.u,
                                    node.v,
                                    node.radius,
                                    uUp});
                            }
                            node.alive = false;
                            continue;
                        }

                        node.u = bestU;
                        node.v = bestV;

                        // Step 2b: Mutual attraction toward nearby active branches within merge horizon.
                        // Prioritize the closest merge partner within `2.5 * (r_node + r_other)` so pairs
                        // of branches converge all the way to a shared center (`dist <= maxLatStep`) without
                        // stalling between opposing pulls from more distant branches.
                        bool hasCloseMergePartner = false;
                        const float remBudget = std::max(0.0f, maxLatStep - usedLatStep);
                        if (remBudget > 1.0e-3f && !inRoofZone) {
                            std::size_t bestPartnerIdx = activeNodes.size();
                            float bestPartnerDist = std::numeric_limits<float>::max();
                            float attractU = 0.0f;
                            float attractV = 0.0f;
                            float attractWeight = 0.0f;

                            for (std::size_t otherIdx = 0; otherIdx < activeNodes.size(); ++otherIdx) {
                                const auto& other = activeNodes[otherIdx];
                                if (!other.alive || other.branchIdx == node.branchIdx) {
                                    continue;
                                }
                                const float du = other.u - node.u;
                                const float dv = other.v - node.v;
                                const float dist = std::sqrt(du * du + dv * dv);
                                const float closePartnerHorizon = 2.5f * (node.radius + other.radius);
                                if (dist <= closePartnerHorizon && dist < bestPartnerDist &&
                                    (other.roofH - hNext) > topAnchorBandMm) {
                                    bestPartnerDist = dist;
                                    bestPartnerIdx = otherIdx;
                                }
                                const float horizon =
                                    std::max(8.0f, 3.8f * (node.radius + other.radius));
                                if (dist > 1.0e-3f && dist <= horizon) {
                                    const float w = (other.radius * other.radius) / (dist + 0.5f);
                                    attractU += w * (du / dist);
                                    attractV += w * (dv / dist);
                                    attractWeight += w;
                                }
                            }

                            float dirU = 0.0f;
                            float dirV = 0.0f;
                            float maxNudge = 0.0f;
                            if (bestPartnerIdx < activeNodes.size()) {
                                hasCloseMergePartner = true;
                                const auto& partner = activeNodes[bestPartnerIdx];
                                const float du = partner.u - node.u;
                                const float dv = partner.v - node.v;
                                const float dist = std::sqrt(du * du + dv * dv);
                                if (dist <= node.radius + partner.radius) {
                                    node.targetRadius = std::max(
                                        node.targetRadius,
                                        std::min(maxTrunkRadiusMm, partner.radius));
                                }
                                if (dist > 1.0e-4f) {
                                    dirU = du / dist;
                                    dirV = dv / dist;
                                    const float maxClose =
                                        (bestPartnerIdx < nodeIdx) ? dist : (0.60f * dist);
                                    maxNudge = std::min(remBudget, maxClose);
                                }
                            } else if (attractWeight > 0.0f) {
                                const float aNorm =
                                    std::sqrt(attractU * attractU + attractV * attractV);
                                if (aNorm > 1.0e-5f) {
                                    dirU = attractU / aNorm;
                                    dirV = attractV / aNorm;
                                    maxNudge = 0.75f * remBudget;
                                }
                            }

                            if (maxNudge > 1.0e-4f) {
                                const float candU = node.u + maxNudge * dirU;
                                const float candV = node.v + maxNudge * dirV;
                                const float candScore =
                                    evaluateLookaheadScore(candU, candV, hNext, node.radius, node.roofH);
                                if ((candScore >= 0.25f || candScore >= straightScore - 0.25f) &&
                                    hasSupportClearanceAtUVH(candU, candV, hNext, 0.0f, false)) {
                                    node.u = candU;
                                    node.v = candV;
                                    usedLatStep += maxNudge;
                                }
                            }
                        }

                        // Step 2b.5: Trunk radius expansion toward the build plate (conical trunk
                        // taper plus basal root flare in the bottom 10 mm for bed adhesion), with
                        // perimeter-based wall push-away (including downward lookahead) and clearance
                        // clamping so narrow-band SDF saturation never prevents trunks from expanding
                        // up to `maxTrunkRadiusMm`.
                        if (!inRoofZone) {
                            const bool groundsToBed = onBedOnly || (node.floorH <= 1.0e-3f);
                            const float basalFlareZoneMm = 10.0f;
                            float flareSlope = 0.0f;
                            if (groundsToBed && hNext < basalFlareZoneMm) {
                                const float flareFrac =
                                    std::clamp(1.0f - (hNext / basalFlareZoneMm), 0.0f, 1.0f);
                                flareSlope = 0.30f * (0.35f + 0.65f * flareFrac);
                            }
                            const float dTargetR = (0.18f + flareSlope) * dhActual;
                            node.targetRadius = std::min(
                                maxTrunkRadiusMm,
                                std::max(node.targetRadius, node.radius + dTargetR));

                            const float maxDrStep = 0.42f * maxLatStep;
                            const float desiredR = std::min(
                                maxTrunkRadiusMm,
                                std::min(node.targetRadius, node.radius + maxDrStep));

                            const float remSteer = std::max(0.0f, maxLatStep - usedLatStep);
                            if (remSteer > 1.0e-4f && !hasCloseMergePartner) {
                                const float rProbe = std::min(
                                    maxTrunkRadiusMm,
                                    std::max(node.targetRadius + 0.5f, node.radius + 1.2f));
                                const float hLook =
                                    groundsToBed
                                        ? std::max(0.30f, hNext - std::min(4.5f, 0.65f * hNext))
                                        : hNext;
                                float pushU = 0.0f;
                                float pushV = 0.0f;
                                int blockedCount = 0;
                                for (int k = 0; k < 16; ++k) {
                                    const float ang =
                                        static_cast<float>(k) * (2.0f * 3.14159265358979323846f / 16.0f);
                                    const float ca = std::cos(ang);
                                    const float sa = std::sin(ang);
                                    const float uP = node.u + rProbe * ca;
                                    const float vP = node.v + rProbe * sa;
                                    if (!hasSupportClearanceAtUVH(uP, vP, hNext, 0.0f, false) ||
                                        !hasSupportClearanceAtUVH(uP, vP, hLook, 0.0f, false)) {
                                        pushU -= ca;
                                        pushV -= sa;
                                        ++blockedCount;
                                    }
                                }
                                const float pushNorm = std::hypot(pushU, pushV);
                                if (blockedCount > 0 && blockedCount < 14 && pushNorm > 1.0e-4f) {
                                    const float nudgeDist = 0.85f * remSteer;
                                    const float nudgeU = node.u + nudgeDist * (pushU / pushNorm);
                                    const float nudgeV = node.v + nudgeDist * (pushV / pushNorm);
                                    if (hasSupportClearanceAtUVH(nudgeU, nudgeV, hNext, 0.0f, false) &&
                                        isTrunkPerimeterClear(nudgeU, nudgeV, hNext, node.radius, false)) {
                                        node.u = nudgeU;
                                        node.v = nudgeV;
                                        usedLatStep += nudgeDist;
                                    }
                                }
                            }

                            if (isTrunkPerimeterClear(node.u, node.v, hNext, desiredR, false)) {
                                node.radius = desiredR;
                            } else {
                                float rLow = tipRadiusMm;
                                float rHigh = std::max(tipRadiusMm, desiredR);
                                float bestR = rLow;
                                for (int iter = 0; iter < 5; ++iter) {
                                    const float rMid = 0.5f * (rLow + rHigh);
                                    if (isTrunkPerimeterClear(node.u, node.v, hNext, rMid, false)) {
                                        bestR = rMid;
                                        rLow = rMid;
                                    } else {
                                        rHigh = rMid;
                                    }
                                }
                                node.radius = bestR;
                            }
                        }
                    }

                    // Step 2c: Coalesce (merge) nearby active branches into shared trunks once their
                    // centers have converged close enough that connecting both branches to `(mu, mv)`
                    // at `hNext` respects `maxLatStep` (preventing unsupported horizontal jumps at forks).
                    for (std::size_t i = 0; i < activeNodes.size(); ++i) {
                        if (!activeNodes[i].alive) {
                            continue;
                        }
                        for (std::size_t j = i + 1; j < activeNodes.size(); ++j) {
                            if (!activeNodes[j].alive) {
                                continue;
                            }
                            auto& a = activeNodes[i];
                            auto& b = activeNodes[j];
                            // Only merge branches below their roof contact zone
                            if ((a.roofH - hNext) <= topAnchorBandMm ||
                                (b.roofH - hNext) <= topAnchorBandMm) {
                                continue;
                            }
                            const float du = b.u - a.u;
                            const float dv = b.v - a.v;
                            const float dist = std::sqrt(du * du + dv * dv);
                            if (dist > 1.35f * maxLatStep) {
                                continue;
                            }
                            const auto& aPrev = treeBranches[a.branchIdx].stations.back();
                            const auto& bPrev = treeBranches[b.branchIdx].stations.back();

                            const float areaA = a.radius * a.radius;
                            const float areaB = b.radius * b.radius;
                            const float invArea = 1.0f / std::max(areaA + areaB, 1.0e-4f);
                            const float mu = (areaA * a.u + areaB * b.u) * invArea;
                            const float mv = (areaA * a.v + areaB * b.v) * invArea;

                            // Ensure neither branch exceeds `1.08 * maxLatStep` from `hCurr` to `hNext`
                            const float stepA = std::hypot(mu - aPrev.u, mv - aPrev.v);
                            const float stepB = std::hypot(mu - bPrev.u, mv - bPrev.v);
                            if (stepA > 1.08f * maxLatStep || stepB > 1.08f * maxLatStep) {
                                continue;
                            }

                            if (hasSupportClearanceAtUVH(mu, mv, hNext, 0.0f, false)) {
                                const float maxDrStep = 0.42f * maxLatStep;
                                const float rMergedGoal = std::min(
                                    maxTrunkRadiusMm,
                                    std::max(a.radius, b.radius));
                                float rMerged = tipRadiusMm;
                                if (isTrunkPerimeterClear(mu, mv, hNext, rMergedGoal, false)) {
                                    rMerged = rMergedGoal;
                                } else {
                                    float rLow = tipRadiusMm;
                                    float rHigh = std::max(tipRadiusMm, rMergedGoal);
                                    for (int iter = 0; iter < 5; ++iter) {
                                        const float rMid = 0.5f * (rLow + rHigh);
                                        if (isTrunkPerimeterClear(mu, mv, hNext, rMid, false)) {
                                            rMerged = rMid;
                                            rLow = rMid;
                                        } else {
                                            rHigh = rMid;
                                        }
                                    }
                                }
                                const float rBEnd = std::clamp(
                                    rMerged,
                                    b.radius - maxDrStep,
                                    b.radius + maxDrStep);
                                // Connect child branch `b` into merged trunk `a` at `(mu, mv, hNext)`
                                treeBranches[b.branchIdx].stations.push_back(TreeStation{
                                    hNext,
                                    mu,
                                    mv,
                                    rBEnd,
                                    uUp});
                                treeBranches[b.branchIdx].mergedIntoBranch =
                                    static_cast<std::int32_t>(a.branchIdx);
                                treeBranches[a.branchIdx].forkHeights.push_back(hNext);
                                b.alive = false;

                                a.u = mu;
                                a.v = mv;
                                a.radius = rMerged;
                                a.targetRadius = std::min(
                                    maxTrunkRadiusMm,
                                    std::max({a.targetRadius, b.targetRadius, std::sqrt(areaA + areaB)}));
                                a.roofH = std::max(a.roofH, b.roofH);
                            }
                        }
                    }

                    // Record station at `hNext` for all surviving branches
                    for (auto& node : activeNodes) {
                        if (!node.alive) {
                            continue;
                        }
                        treeBranches[node.branchIdx].stations.push_back(TreeStation{
                            hNext,
                            node.u,
                            node.v,
                            node.radius,
                            uUp});
                        if (hNext <= 1.0e-4f) {
                            treeBranches[node.branchIdx].groundedOnBed = true;
                        }
                    }
                }

                // Enforce bottom-up radius monotonicity (`s[k-1].radius <= s[k].radius`) along
                // every branch and across child-to-parent fork junctions so trunks strictly
                // expand (never narrow) toward the build plate and no upper branch station
                // ever overhangs a narrower lower neck.
                for (int pass = 0; pass < 3; ++pass) {
                    for (auto& br : treeBranches) {
                        if (br.stations.size() < 2) {
                            continue;
                        }
                        if (br.mergedIntoBranch >= 0 &&
                            static_cast<std::size_t>(br.mergedIntoBranch) < treeBranches.size()) {
                            const auto& parentBr =
                                treeBranches[static_cast<std::size_t>(br.mergedIntoBranch)];
                            const float hMerge = br.stations.back().h;
                            for (std::size_t pk = 0; pk + 1 < parentBr.stations.size(); ++pk) {
                                const auto& ps0 = parentBr.stations[pk];
                                const auto& ps1 = parentBr.stations[pk + 1];
                                if (hMerge <= ps0.h + 1.0e-4f && hMerge >= ps1.h - 1.0e-4f) {
                                    const float denom = std::max(1.0e-5f, ps0.h - ps1.h);
                                    const float t = std::clamp((ps0.h - hMerge) / denom, 0.0f, 1.0f);
                                    const float parentR = ps0.radius + t * (ps1.radius - ps0.radius);
                                    br.stations.back().radius =
                                        std::min(br.stations.back().radius, parentR);
                                    break;
                                }
                            }
                        }
                        for (std::size_t k = br.stations.size() - 1; k > 0; --k) {
                            br.stations[k - 1].radius =
                                std::min(br.stations[k - 1].radius, br.stations[k].radius);
                        }
                    }
                }

                // If `onBedOnly` is enabled, discard any stub branch that became trapped in an
                // internal pocket and neither reached the build plate (`h == 0`) nor merged into a
                // trunk that reached the build plate.
                if (onBedOnly && !treeBranches.empty()) {
                    // Propagate `groundedOnBed` upward strictly along actual merged parent links
                    bool changed = true;
                    while (changed) {
                        changed = false;
                        for (auto& br : treeBranches) {
                            if (br.groundedOnBed || br.stations.empty()) {
                                continue;
                            }
                            const auto& endSt = br.stations.back();
                            if (endSt.h <= 1.0e-3f) {
                                br.groundedOnBed = true;
                                changed = true;
                                continue;
                            }
                            if (br.mergedIntoBranch >= 0 &&
                                static_cast<std::size_t>(br.mergedIntoBranch) < treeBranches.size() &&
                                treeBranches[static_cast<std::size_t>(br.mergedIntoBranch)].groundedOnBed) {
                                br.groundedOnBed = true;
                                changed = true;
                            }
                        }
                    }
                    treeBranches.erase(
                        std::remove_if(
                            treeBranches.begin(),
                            treeBranches.end(),
                            [](const TreeBranch& br) {
                                return !br.groundedOnBed || br.stations.size() < 2;
                            }),
                        treeBranches.end());
                }

                // 3. Compute 3D unit centerline tangents `tangentUp` (pointing from trunk to tip)
                //    in bed coordinates so 5/6-axis tools can orient nozzle beads along the branch!
                for (auto& br : treeBranches) {
                    const std::size_t m = br.stations.size();
                    for (std::size_t k = 0; k < m; ++k) {
                        const std::size_t kUpper = (k > 0) ? (k - 1) : 0;
                        const std::size_t kLower = (k + 1 < m) ? (k + 1) : (m - 1);
                        const float du = br.stations[kUpper].u - br.stations[kLower].u;
                        const float dv = br.stations[kUpper].v - br.stations[kLower].v;
                        const float dh = std::max(1.0e-3f, br.stations[kUpper].h - br.stations[kLower].h);
                        std::array<float, 3> rawT{
                            du * tU[0] + dv * tV[0] + dh * uUp[0],
                            du * tU[1] + dv * tV[1] + dh * uUp[1],
                            du * tU[2] + dv * tV[2] + dh * uUp[2]};
                        const float tLen =
                            std::sqrt(rawT[0] * rawT[0] + rawT[1] * rawT[1] + rawT[2] * rawT[2]);
                        if (tLen > 1.0e-5f) {
                            br.stations[k].tangentUp = {
                                rawT[0] / tLen,
                                rawT[1] / tLen,
                                rawT[2] / tLen};
                        } else {
                            br.stations[k].tangentUp = uUp;
                        }
                    }
                }

                // 4. Link every overhang contact cell `(iu, iv)` in `supportColumns` to its nearest
                //    surviving `TreeBranch` so the full overhang contact area is supported at `roofH`
                //    and funnels smoothly into the branch tips across `treeCanopyTransitionMm`.
                if (!treeBranches.empty()) {
                    const float maxCanopyReachSq =
                        (2.8f * clusterRadiusMm + 3.0f) * (2.8f * clusterRadiusMm + 3.0f);
                    treeCanopyCells.reserve(rawCells.size());
                    for (const auto& rc : rawCells) {
                        std::size_t bestIdx = treeBranches.size();
                        float bestDistSq = maxCanopyReachSq;
                        for (std::size_t bIdx = 0; bIdx < treeBranches.size(); ++bIdx) {
                            const auto& br = treeBranches[bIdx];
                            if (br.stations.empty()) {
                                continue;
                            }
                            if (std::abs(br.roofH - rc.roofH) > 4.0f * targetLayerH + 1.5f) {
                                continue;
                            }
                            const float du = rc.u - br.stations.front().u;
                            const float dv = rc.v - br.stations.front().v;
                            const float dSq = du * du + dv * dv;
                            if (dSq <= bestDistSq) {
                                bestDistSq = dSq;
                                bestIdx = bIdx;
                            }
                        }
                        if (bestIdx < treeBranches.size()) {
                            treeCanopyCells.push_back(CanopyCellLink{
                                rc.iu,
                                rc.iv,
                                rc.u,
                                rc.v,
                                rc.roofH,
                                onBedOnly ? 0.0f : rc.floorH,
                                bestIdx});
                        }
                    }
                }
            }

            // Query support tool rotational capabilities for 5/6-axis branch-tangent nozzle orientation
            bool toolHasRotationalAxes = false;
            float toolMaxTiltRad = 0.0f;
            if (!impl_->tools.empty() && supportToolIdx < impl_->tools.size()) {
                toolHasRotationalAxes = impl_->tools[supportToolIdx].has_rotational_axes;
                toolMaxTiltRad =
                    std::clamp(impl_->tools[supportToolIdx].max_angle_from_bed_normal_deg, 0.0f, 85.0f) *
                    (3.14159265358979323846f / 180.0f);
            }

            float lastSupportIso = -1.0e9f;
            for (std::size_t layerIdx = 0; layerIdx < res.layers.size(); ++layerIdx) {
                auto& layer = res.layers[layerIdx];
                const float hLayer = layer.isovalue;
                if (hLayer <= 0.0f) {
                    continue;
                }
                // Avoid emitting duplicate support columns at identical isovalues across parallel branches
                if (std::abs(hLayer - lastSupportIso) < 1.0e-3f) {
                    continue;
                }

                std::uint32_t bedLayerOrdinal = 0;
                for (std::size_t uIdx = 0; uIdx < uniqueLayerIsos.size(); ++uIdx) {
                    if (uniqueLayerIsos[uIdx] <= hLayer + 1.0e-3f) {
                        bedLayerOrdinal = static_cast<std::uint32_t>(uIdx);
                    } else {
                        break;
                    }
                }

                float localLayerH = targetLayerH;
                if (layerIdx < selectedLayers.layers.size() &&
                    selectedLayers.layers[layerIdx].mean_thickness_mm > 0.01f) {
                    localLayerH = selectedLayers.layers[layerIdx].mean_thickness_mm;
                } else if (layerIdx > 0) {
                    const float dh = layer.isovalue - res.layers[layerIdx - 1].isovalue;
                    if (dh > 0.02f && dh < 2.0f) {
                        localLayerH = dh;
                    }
                }
                const float effectiveStepH = std::max(
                    localLayerH,
                    std::max(targetLayerH, impl_->sliceLayerHeight.max_mm));
                const float interfaceBandMm =
                    static_cast<float>(impl_->supportSpec.interface_layers) * effectiveStepH;
                const float bedInterfaceBandMm =
                    firstSupportIso +
                    ((impl_->supportSpec.bed_interface_layers > 0)
                         ? static_cast<float>(impl_->supportSpec.bed_interface_layers - 1) *
                               effectiveStepH
                         : 0.0f);

                std::vector<ExtrusionPath> supportPathsForLayer;
                std::uint32_t supportPassIdx = 0;

                if (useTreeSupport) {
                    // =====================================================================
                    // Emit Non-Planar Tree Support Cross-Sections at `hLayer`
                    // =====================================================================
                    struct LayerBranchDisk {
                        std::size_t branchIdx = 0;
                        float u = 0.0f;
                        float v = 0.0f;
                        float radius = 1.0f;
                        float roofH = 0.0f;
                        float floorH = 0.0f;
                        bool groundedOnBed = false;
                        bool hasForkDiaphragm = false;
                        std::array<float, 3> tangentUp{0.0f, 0.0f, 1.0f};
                    };
                    std::vector<LayerBranchDisk> disks;
                    std::vector<std::int32_t> branchToDiskIdx(treeBranches.size(), -1);

                    for (std::size_t bIdx = 0; bIdx < treeBranches.size(); ++bIdx) {
                        const auto& br = treeBranches[bIdx];
                        if (br.stations.size() < 2) {
                            continue;
                        }
                        const float hTop = br.stations.front().h;
                        const float hBot = br.stations.back().h;
                        const float minHForBranch =
                            (br.mergedIntoBranch >= 0)
                                ? (hBot - 1.0e-4f)
                                : (hBot - 1.0f * effectiveStepH);
                        if (hLayer > hTop + 0.5f * effectiveStepH ||
                            hLayer < minHForBranch) {
                            continue;
                        }
                        const float hClamped = std::clamp(hLayer, hBot, hTop);
                        for (std::size_t k = 0; k + 1 < br.stations.size(); ++k) {
                            const auto& s0 = br.stations[k];     // higher h
                            const auto& s1 = br.stations[k + 1]; // lower h
                            if (hClamped <= s0.h + 1.0e-4f && hClamped >= s1.h - 1.0e-4f) {
                                const float denom = std::max(1.0e-5f, s0.h - s1.h);
                                const float t = std::clamp((s0.h - hClamped) / denom, 0.0f, 1.0f);
                                LayerBranchDisk d{};
                                d.branchIdx = bIdx;
                                d.u = s0.u + t * (s1.u - s0.u);
                                d.v = s0.v + t * (s1.v - s0.v);
                                d.radius = s0.radius + t * (s1.radius - s0.radius);
                                d.roofH = br.roofH;
                                d.floorH = br.floorH;
                                d.groundedOnBed = br.groundedOnBed;
                                for (const float hFork : br.forkHeights) {
                                    if (hLayer >= hFork - 2.0f * effectiveStepH - 1.0e-4f &&
                                        hLayer <= hFork + 2.0f * effectiveStepH + 1.0e-4f) {
                                        d.hasForkDiaphragm = true;
                                        break;
                                    }
                                }
                                if (br.mergedIntoBranch >= 0 &&
                                    hLayer <= hBot + 1.5f * effectiveStepH + 1.0e-4f) {
                                    d.hasForkDiaphragm = true;
                                }
                                d.tangentUp = {
                                    s0.tangentUp[0] + t * (s1.tangentUp[0] - s0.tangentUp[0]),
                                    s0.tangentUp[1] + t * (s1.tangentUp[1] - s0.tangentUp[1]),
                                    s0.tangentUp[2] + t * (s1.tangentUp[2] - s0.tangentUp[2])};
                                const float tNorm = std::sqrt(
                                    d.tangentUp[0] * d.tangentUp[0] +
                                    d.tangentUp[1] * d.tangentUp[1] +
                                    d.tangentUp[2] * d.tangentUp[2]);
                                if (tNorm > 1.0e-5f) {
                                    d.tangentUp[0] /= tNorm;
                                    d.tangentUp[1] /= tNorm;
                                    d.tangentUp[2] /= tNorm;
                                } else {
                                    d.tangentUp = uUp;
                                }
                                branchToDiskIdx[bIdx] = static_cast<std::int32_t>(disks.size());
                                disks.push_back(d);
                                break;
                            }
                        }
                    }

                    // Only deduplicate truly concentric/coincident disks (`dist <= 0.20 * lineWidthMm`)
                    // right at the exact fork point; never suppress a diverging branch disk while it
                    // separates from its parent trunk!
                    std::vector<LayerBranchDisk> uniqueDisks;
                    const float concentricTolSq = (0.20f * lineWidthMm) * (0.20f * lineWidthMm);
                    for (const auto& d : disks) {
                        bool absorbed = false;
                        for (auto& ud : uniqueDisks) {
                            const float du = d.u - ud.u;
                            const float dv = d.v - ud.v;
                            if (du * du + dv * dv <= concentricTolSq) {
                                if (std::abs(d.radius - ud.radius) > 0.25f * lineWidthMm) {
                                    ud.hasForkDiaphragm = true;
                                }
                                ud.radius = std::max(ud.radius, d.radius);
                                ud.roofH = std::max(ud.roofH, d.roofH);
                                ud.groundedOnBed = ud.groundedOnBed || d.groundedOnBed;
                                ud.hasForkDiaphragm = ud.hasForkDiaphragm || d.hasForkDiaphragm;
                                absorbed = true;
                                break;
                            }
                        }
                        if (!absorbed) {
                            uniqueDisks.push_back(d);
                        }
                    }

                    // Helper to compute local nozzle axis for a tree support point `pBed`
                    // honoring Rule 1 (`queryNozzleFrame`) and 5/6-axis branch-tangent tilt
                    const float bedTiltScale =
                        std::clamp(hLayer / std::max(3.0f * targetLayerH, 0.6f), 0.0f, 1.0f);
                    auto makeTreePoint = [&](float uVal,
                                             float vVal,
                                             const std::array<float, 3>& travelDir,
                                             const std::array<float, 3>& branchTangentUp,
                                             bool isRoofContactZone) -> std::pair<bool, ToolpathPoint> {
                        if (!hasSupportClearanceAtUVH(uVal, vVal, hLayer, 0.0f, isRoofContactZone)) {
                            return {false, ToolpathPoint{}};
                        }
                        const std::array<float, 3> pBed = uvHToBed(uVal, vVal, hLayer);
                        const NozzleBasis nb = queryNozzleFrame(pBed, travelDir);
                        std::array<float, 3> nAxis = nb.nozzle_axis;
                        if (toolHasRotationalAxes && toolMaxTiltRad > 1.0e-4f && bedTiltScale > 1.0e-3f) {
                            const float maxAng = bedTiltScale * toolMaxTiltRad;
                            const float cosDot = std::clamp(
                                nAxis[0] * branchTangentUp[0] +
                                    nAxis[1] * branchTangentUp[1] +
                                    nAxis[2] * branchTangentUp[2],
                                -1.0f,
                                1.0f);
                            const float ang = std::acos(cosDot);
                            if (ang > 1.0e-4f) {
                                const float alpha = std::min(1.0f, maxAng / ang);
                                std::array<float, 3> blended{
                                    (1.0f - alpha) * nAxis[0] + alpha * branchTangentUp[0],
                                    (1.0f - alpha) * nAxis[1] + alpha * branchTangentUp[1],
                                    (1.0f - alpha) * nAxis[2] + alpha * branchTangentUp[2]};
                                const float bLen = std::sqrt(
                                    blended[0] * blended[0] +
                                    blended[1] * blended[1] +
                                    blended[2] * blended[2]);
                                if (bLen > 1.0e-5f) {
                                    nAxis = {blended[0] / bLen, blended[1] / bLen, blended[2] / bLen};
                                }
                            }
                        }
                        ToolpathPoint tp{};
                        tp.x = pBed[0];
                        tp.y = pBed[1];
                        tp.z = pBed[2];
                        tp.nx = nAxis[0];
                        tp.ny = nAxis[1];
                        tp.nz = nAxis[2];
                        tp.width_mm = lineWidthMm;
                        tp.height_mm = localLayerH;
                        return {true, tp};
                    };

                    auto pushSupportPath = [&](std::uint8_t featureKind,
                                               bool isClosed,
                                               std::vector<ToolpathPoint>&& pts) {
                        if (pts.size() < 2) {
                            return;
                        }
                        float lenMm = 0.0f;
                        for (std::size_t k = 1; k < pts.size(); ++k) {
                            const float dxSeg = pts[k].x - pts[k - 1].x;
                            const float dySeg = pts[k].y - pts[k - 1].y;
                            const float dzSeg = pts[k].z - pts[k - 1].z;
                            lenMm += std::sqrt(dxSeg * dxSeg + dySeg * dySeg + dzSeg * dzSeg);
                        }
                        if (lenMm < minSegLenMm) {
                            return;
                        }
                        ExtrusionPath ep{};
                        ep.feature_kind = featureKind;
                        ep.wall_index = supportPassIdx++;
                        ep.is_closed = isClosed;
                        ep.total_length_mm = lenMm;
                        ep.mean_width_mm = lineWidthMm;
                        ep.points.reserve(pts.size());
                        for (const auto& pt : pts) {
                            ep.points.push_back(pt);
                        }
                        supportPathsForLayer.push_back(std::move(ep));
                        if (featureKind == 7) {
                            ++res.support_count;
                        } else {
                            ++res.support_interface_count;
                        }
                        ++res.total_path_count;
                    };

                    const std::uint32_t wallCount =
                        std::clamp<std::uint32_t>(impl_->supportSpec.tree_branch_wall_count, 1, 4);
                    const bool swapUV = ((layerIdx % 2) == 1);

                    // Precompute each disk's mode (`isBedIntf`, `isRoofIntf`) and outer boundary
                    // radius `diskOuterR` on `hLayer`, and propagate `hasForkDiaphragm` across
                    // overlapping non-bed disks so all overlapping disks share a single boolean-union
                    // level-set offset family `S_union(u, v) = c` without intersecting rings!
                    const std::size_t numUniqueDisks = uniqueDisks.size();
                    std::vector<bool> diskIsBedIntf(numUniqueDisks, false);
                    std::vector<bool> diskIsRoofIntf(numUniqueDisks, false);
                    std::vector<float> diskOuterR(numUniqueDisks, 0.0f);
                    std::vector<bool> diskEmitted(numUniqueDisks, false);

                    for (std::size_t i = 0; i < numUniqueDisks; ++i) {
                        const auto& d = uniqueDisks[i];
                        diskIsBedIntf[i] =
                            (impl_->supportSpec.bed_interface_layers > 0) &&
                            d.groundedOnBed &&
                            (bedLayerOrdinal < impl_->supportSpec.bed_interface_layers ||
                             hLayer <= bedInterfaceBandMm + 1.0e-4f);
                        diskIsRoofIntf[i] =
                            !diskIsBedIntf[i] &&
                            (impl_->supportSpec.interface_layers > 0) &&
                            (d.roofH - hLayer <= interfaceBandMm + 1.0e-4f);
                        diskOuterR[i] = diskIsBedIntf[i] ? (d.radius + 1.0f) : d.radius;
                    }

                    // Mark `hasForkDiaphragm = true` on any overlapping pair of `Support Base`
                    // disks so overlapping disks always share the same solid `lineWidthMm`
                    // union level-set offset family `S_union(u, v) = c` without hollow gaps.
                    for (std::size_t i = 0; i < numUniqueDisks; ++i) {
                        if (diskIsBedIntf[i] || diskIsRoofIntf[i]) {
                            continue;
                        }
                        for (std::size_t j = i + 1; j < numUniqueDisks; ++j) {
                            if (diskIsBedIntf[j] || diskIsRoofIntf[j]) {
                                continue;
                            }
                            const float distIJ = std::hypot(
                                uniqueDisks[i].u - uniqueDisks[j].u,
                                uniqueDisks[i].v - uniqueDisks[j].v);
                            if (distIJ <= diskOuterR[i] + diskOuterR[j] + 0.25f * lineWidthMm) {
                                uniqueDisks[i].hasForkDiaphragm = true;
                                uniqueDisks[j].hasForkDiaphragm = true;
                            }
                        }
                    }

                    // Emit a ring on `uniqueDisks[diskIdx]` at radius `ringRadius = diskOuterR[diskIdx] - c`,
                    // clipped against all other disks `j != diskIdx` so overlapping base pads and
                    // forking branch rings trace the smooth boolean-union contour `S_union(u, v) = c`
                    // and never intersect each other on the same layer.
                    auto emitRingOnDisk = [&](std::size_t diskIdx,
                                              float ringRadius,
                                              std::uint8_t featureKind,
                                              bool isRoofContactZone) {
                        if (ringRadius < 0.35f * lineWidthMm) {
                            return;
                        }
                        const auto& d = uniqueDisks[diskIdx];
                        const float cOffset = std::max(0.0f, diskOuterR[diskIdx] - ringRadius);
                        const int numSegs = std::clamp<int>(
                            static_cast<int>(std::ceil(2.0f * 3.14159265f * ringRadius / 0.45f)),
                            12,
                            64);
                        std::vector<ToolpathPoint> currentArc;
                        currentArc.reserve(static_cast<std::size_t>(numSegs) + 1);
                        bool allValid = true;
                        for (int s = 0; s <= numSegs; ++s) {
                            const float ang =
                                static_cast<float>(s) * (2.0f * 3.14159265358979323846f / static_cast<float>(numSegs));
                            const float ca = std::cos(ang);
                            const float sa = std::sin(ang);
                            const float uPt = d.u + ringRadius * ca;
                            const float vPt = d.v + ringRadius * sa;

                            bool insideOtherDisk = false;
                            for (std::size_t j = 0; j < numUniqueDisks; ++j) {
                                if (j == diskIdx) {
                                    continue;
                                }
                                if (diskIsRoofIntf[j] && !treeCanopyCells.empty()) {
                                    continue;
                                }
                                const float rJ = diskOuterR[j] - cOffset;
                                const float distIJ = std::hypot(
                                    d.u - uniqueDisks[j].u,
                                    d.v - uniqueDisks[j].v);
                                if (diskIsBedIntf[diskIdx] == diskIsBedIntf[j] &&
                                    distIJ + rJ <= ringRadius + 0.02f * lineWidthMm) {
                                    continue;
                                }
                                const float sinFactor = std::clamp(
                                    distIJ / std::max({ringRadius, rJ, 0.1f}),
                                    0.01f,
                                    1.0f);
                                const float seamMargin = 0.08f * lineWidthMm * sinFactor;
                                const float minAllowedR =
                                    (diskIsBedIntf[diskIdx] == diskIsBedIntf[j])
                                        ? (rJ + seamMargin)
                                        : (diskOuterR[j] + 0.50f * lineWidthMm);
                                if (minAllowedR <= 0.0f) {
                                    continue;
                                }
                                const float duJ = uPt - uniqueDisks[j].u;
                                const float dvJ = vPt - uniqueDisks[j].v;
                                if (duJ * duJ + dvJ * dvJ < minAllowedR * minAllowedR) {
                                    insideOtherDisk = true;
                                    break;
                                }
                            }
                            if (insideOtherDisk) {
                                allValid = false;
                                if (currentArc.size() >= 2) {
                                    pushSupportPath(featureKind, false, std::move(currentArc));
                                }
                                currentArc.clear();
                                continue;
                            }

                            const std::array<float, 3> tangDir{
                                -sa * tU[0] + ca * tV[0],
                                -sa * tU[1] + ca * tV[1],
                                -sa * tU[2] + ca * tV[2]};
                            auto [ok, tp] =
                                makeTreePoint(uPt, vPt, tangDir, d.tangentUp, isRoofContactZone);
                            if (ok) {
                                currentArc.push_back(tp);
                            } else {
                                allValid = false;
                                if (currentArc.size() >= 2) {
                                    pushSupportPath(featureKind, false, std::move(currentArc));
                                }
                                currentArc.clear();
                            }
                        }
                        if (currentArc.size() >= 2) {
                            pushSupportPath(featureKind, allValid, std::move(currentArc));
                        }
                    };

                    // 1. Emit branch / trunk disks at `hLayer`
                    for (std::size_t diskIdx = 0; diskIdx < numUniqueDisks; ++diskIdx) {
                        const auto& d = uniqueDisks[diskIdx];
                        const bool isBedIntf = diskIsBedIntf[diskIdx];
                        const bool isRoofIntf = diskIsRoofIntf[diskIdx];

                        if (isBedIntf) {
                            // Dense flanged bed adhesion floor pad (`feature_kind = 8`), clipped to boolean union
                            diskEmitted[diskIdx] = true;
                            const float flangeR = diskOuterR[diskIdx];
                            for (float rRing = flangeR - 0.5f * lineWidthMm;
                                 rRing >= 0.45f * lineWidthMm;
                                 rRing -= interfaceSpacingMm) {
                                emitRingOnDisk(diskIdx, rRing, 8, false);
                            }
                        } else if (isRoofIntf) {
                            // When `treeCanopyCells` is empty, emit outer framing ring on roof interface;
                            // when `treeCanopyCells` is populated, Step 2 emits the uniform parallel roof
                            // interface lines across the entire overhang contact patch without intersecting rings.
                            if (treeCanopyCells.empty()) {
                                diskEmitted[diskIdx] = true;
                                const float outerR =
                                    std::max(0.5f * lineWidthMm, d.radius - 0.25f * lineWidthMm);
                                emitRingOnDisk(diskIdx, outerR, 8, true);
                            }
                        } else {
                            // Hollow / ribbed organic tree branch & trunk walls (`feature_kind = 7`)
                            diskEmitted[diskIdx] = true;
                            float innerCoreR = d.radius;
                            for (std::uint32_t w = 0; w < wallCount; ++w) {
                                const float rWall =
                                    d.radius - (static_cast<float>(w) + 0.5f) * lineWidthMm;
                                if (rWall < 0.40f * lineWidthMm) {
                                    break;
                                }
                                emitRingOnDisk(diskIdx, rWall, 7, false);
                                innerCoreR = rWall - 0.5f * lineWidthMm;
                            }
                            // Internal concentric structural bracing & fork diaphragm rings on the
                            // exact `c_w = (w + 0.5) * lineWidthMm` union level-set offset lattice
                            // so all branch walls, expanded trunks, and fork junctions belong to a
                            // single continuous non-intersecting offset family across all layers.
                            for (float rInner = innerCoreR - 0.5f * lineWidthMm;
                                 rInner >= 0.45f * lineWidthMm;
                                 rInner -= lineWidthMm) {
                                emitRingOnDisk(diskIdx, rInner, 7, false);
                            }
                        }
                    }

                    // 2. Emit Full-Footprint Roof Contact Interface (`feature_kind = 8`) and
                    //    Canopy Funnel Transition (`feature_kind = 7`) across `treeCanopyCells`
                    //    so every overhanging region makes complete area contact at `roofH` and
                    //    tapers smoothly into the cylindrical tree branches below!
                    if (!treeCanopyCells.empty()) {
                        struct ActiveCanopyCell {
                            std::uint8_t kind = 8; // 8 = roof interface, 7 = funnel transition
                            std::array<float, 3> tangentUp{0.0f, 0.0f, 1.0f};
                        };
                        std::unordered_map<std::uint64_t, ActiveCanopyCell> layerCanopy;
                        std::int32_t cMinIU = std::numeric_limits<std::int32_t>::max();
                        std::int32_t cMaxIU = std::numeric_limits<std::int32_t>::lowest();
                        std::int32_t cMinIV = std::numeric_limits<std::int32_t>::max();
                        std::int32_t cMaxIV = std::numeric_limits<std::int32_t>::lowest();

                        const float maxCanopyDepth = interfaceBandMm + treeCanopyTransitionMm;
                        for (const auto& cc : treeCanopyCells) {
                            const float depthFromRoof = cc.roofH - hLayer;
                            if (depthFromRoof < -0.5f * effectiveStepH ||
                                depthFromRoof > maxCanopyDepth + 1.0e-4f ||
                                hLayer < cc.floorH) {
                                continue;
                            }
                            const bool isRoofIntf =
                                (impl_->supportSpec.interface_layers > 0) &&
                                (depthFromRoof <= interfaceBandMm + 1.0e-4f);
                            float uCell = cc.u;
                            float vCell = cc.v;
                            std::array<float, 3> tang = uUp;
                            if (cc.branchIdx < branchToDiskIdx.size() &&
                                branchToDiskIdx[cc.branchIdx] >= 0) {
                                const auto& bDisk =
                                    disks[static_cast<std::size_t>(branchToDiskIdx[cc.branchIdx])];
                                tang = bDisk.tangentUp;
                                if (!isRoofIntf && treeCanopyTransitionMm > 1.0e-4f) {
                                    const float alpha = std::clamp(
                                        (depthFromRoof - interfaceBandMm) / treeCanopyTransitionMm,
                                        0.0f,
                                        1.0f);
                                    uCell = (1.0f - alpha) * cc.u + alpha * bDisk.u;
                                    vCell = (1.0f - alpha) * cc.v + alpha * bDisk.v;
                                }
                            } else if (!isRoofIntf) {
                                continue;
                            }

                            const auto iu = static_cast<std::int32_t>(std::llround(uCell / cellStepMm));
                            const auto iv = static_cast<std::int32_t>(std::llround(vCell / cellStepMm));
                            cMinIU = std::min(cMinIU, iu);
                            cMaxIU = std::max(cMaxIU, iu);
                            cMinIV = std::min(cMinIV, iv);
                            cMaxIV = std::max(cMaxIV, iv);
                            const std::uint64_t key = packCellKey(iu, iv);
                            const std::uint8_t kKind = isRoofIntf ? 8 : 7;
                            auto itC = layerCanopy.find(key);
                            if (itC == layerCanopy.end()) {
                                layerCanopy.emplace(key, ActiveCanopyCell{kKind, tang});
                            } else if (kKind == 8) {
                                itC->second.kind = 8;
                            }
                        }

                        if (!layerCanopy.empty()) {
                            auto emitCanopyGroup = [&](std::uint8_t featureKind,
                                                       float spacingMm,
                                                       bool groupSwapUV) {
                                const float minU = static_cast<float>(cMinIU) * cellStepMm;
                                const float maxU = static_cast<float>(cMaxIU) * cellStepMm;
                                const float minV = static_cast<float>(cMinIV) * cellStepMm;
                                const float maxV = static_cast<float>(cMaxIV) * cellStepMm;
                                const float outerMin = groupSwapUV ? minU : minV;
                                const float outerMax = groupSwapUV ? maxU : maxV;
                                const float innerMin = groupSwapUV ? minV : minU;
                                const float innerMax = groupSwapUV ? maxV : maxU;
                                const float marchStepMm =
                                    std::min({0.5f * cellStepMm, 0.40f, 0.5f * lineWidthMm});
                                const std::array<float, 3> travelDir = groupSwapUV ? tV : tU;

                                bool zig = false;
                                for (float outerCoord = outerMin;
                                     outerCoord <= outerMax + 1.0e-4f;
                                     outerCoord += spacingMm) {
                                    std::vector<ToolpathPoint> activeRun;
                                    auto flushCanopyRun = [&]() {
                                        if (activeRun.size() >= 2) {
                                            if (zig) {
                                                std::reverse(activeRun.begin(), activeRun.end());
                                            }
                                            pushSupportPath(featureKind, false, std::move(activeRun));
                                            zig = !zig;
                                        }
                                        activeRun.clear();
                                    };

                                    for (float innerCoord = innerMin;
                                         innerCoord <= innerMax + 1.0e-4f;
                                         innerCoord += marchStepMm) {
                                        const float uVal = groupSwapUV ? outerCoord : innerCoord;
                                        const float vVal = groupSwapUV ? innerCoord : outerCoord;

                                        // Never cross inside any branch/trunk disk already emitted in Step 1
                                        bool insideEmittedDisk = false;
                                        for (std::size_t j = 0; j < numUniqueDisks; ++j) {
                                            if (!diskEmitted[j]) {
                                                continue;
                                            }
                                            const float exclR = diskOuterR[j] + 0.55f * lineWidthMm;
                                            const float duJ = uVal - uniqueDisks[j].u;
                                            const float dvJ = vVal - uniqueDisks[j].v;
                                            if (duJ * duJ + dvJ * dvJ < exclR * exclR) {
                                                insideEmittedDisk = true;
                                                break;
                                            }
                                        }
                                        if (insideEmittedDisk) {
                                            flushCanopyRun();
                                            continue;
                                        }

                                        const auto iu =
                                            static_cast<std::int32_t>(std::llround(uVal / cellStepMm));
                                        const auto iv =
                                            static_cast<std::int32_t>(std::llround(vVal / cellStepMm));

                                        // Resolve a single unique canopy kind at `(iu, iv)` so orthogonal
                                        // Support Base (7) and Support Interface (8) scanlines never cross!
                                        bool any7 = false;
                                        bool any8 = false;
                                        std::array<float, 3> cellTang = uUp;
                                        for (std::int32_t du = -1; du <= 1; ++du) {
                                            for (std::int32_t dv = -1; dv <= 1; ++dv) {
                                                auto itC = layerCanopy.find(packCellKey(iu + du, iv + dv));
                                                if (itC != layerCanopy.end()) {
                                                    if (itC->second.kind == 8) {
                                                        any8 = true;
                                                        cellTang = itC->second.tangentUp;
                                                    } else if (itC->second.kind == 7) {
                                                        any7 = true;
                                                        if (!any8) {
                                                            cellTang = itC->second.tangentUp;
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                        bool matched = false;
                                        if (featureKind == 8) {
                                            matched = any8;
                                        } else {
                                            // Keep Support Base (7) separated from orthogonal Support Interface (8)
                                            bool near8 = any8;
                                            if (any7 && !near8) {
                                                for (std::int32_t du = -2; du <= 2 && !near8; ++du) {
                                                    for (std::int32_t dv = -2; dv <= 2; ++dv) {
                                                        auto itC = layerCanopy.find(packCellKey(iu + du, iv + dv));
                                                        if (itC != layerCanopy.end() && itC->second.kind == 8) {
                                                            near8 = true;
                                                            break;
                                                        }
                                                    }
                                                }
                                            }
                                            matched = any7 && !near8;
                                        }
                                        if (!matched) {
                                            flushCanopyRun();
                                            continue;
                                        }
                                        auto [ok, tp] = makeTreePoint(
                                            uVal,
                                            vVal,
                                            travelDir,
                                            cellTang,
                                            featureKind == 8);
                                        if (!ok) {
                                            flushCanopyRun();
                                            continue;
                                        }
                                        activeRun.push_back(tp);
                                    }
                                    flushCanopyRun();
                                }
                            };

                            emitCanopyGroup(7, baseSpacingMm, swapUV);
                            if (impl_->supportSpec.interface_layers > 0) {
                                emitCanopyGroup(8, interfaceSpacingMm, !swapUV);
                            }
                        }
                    }
                } else {
                    // =====================================================================
                    // Emit Classic Columnar Grid Scanline Supports at `hLayer`
                    // =====================================================================
                    auto emitScanlineSupportGroup = [&](std::uint8_t featureKind, float spacingMm, bool swapUV) {
                        const float minU = static_cast<float>(minIU) * cellStepMm;
                        const float maxU = static_cast<float>(maxIU) * cellStepMm;
                        const float minV = static_cast<float>(minIV) * cellStepMm;
                        const float maxV = static_cast<float>(maxIV) * cellStepMm;

                        const float outerMin = swapUV ? minU : minV;
                        const float outerMax = swapUV ? maxU : maxV;
                        const float innerMin = swapUV ? minV : minU;
                        const float innerMax = swapUV ? maxV : maxU;
                        const float marchStepMm =
                            std::min({0.5f * cellStepMm, 0.40f, 0.5f * lineWidthMm});
                        const float bboxMarginMm = 2.0f * cellStepMm;

                        auto isCellValidForKind = [&](float uSample,
                                                      float vSample,
                                                      bool& outRoofContact) -> bool {
                            const auto iu = static_cast<std::int32_t>(std::llround(uSample / cellStepMm));
                            const auto iv = static_cast<std::int32_t>(std::llround(vSample / cellStepMm));
                            bool inColumn = false;
                            bool isInterface = false;
                            outRoofContact = false;
                            for (std::int32_t du = -1; du <= 1; ++du) {
                                for (std::int32_t dv = -1; dv <= 1; ++dv) {
                                    auto itCol = supportColumns.find(packCellKey(iu + du, iv + dv));
                                    if (itCol != supportColumns.end()) {
                                        const auto& col = itCol->second;
                                        if (hLayer >= col.floorH && hLayer <= col.roofH + 0.5f * effectiveStepH) {
                                            inColumn = true;
                                            const bool isRoofIntf =
                                                (impl_->supportSpec.interface_layers > 0) &&
                                                (col.roofH - hLayer <= interfaceBandMm + 1.0e-4f);
                                            const bool isBedIntf =
                                                (impl_->supportSpec.bed_interface_layers > 0) &&
                                                (col.floorH <= 1.0e-3f) &&
                                                (bedLayerOrdinal < impl_->supportSpec.bed_interface_layers ||
                                                 (hLayer - col.floorH) <= bedInterfaceBandMm + 1.0e-4f);
                                            if (isRoofIntf) {
                                                outRoofContact = true;
                                            }
                                            if (isRoofIntf || isBedIntf) {
                                                isInterface = true;
                                            }
                                        }
                                    }
                                }
                            }
                            if (featureKind == 7 && inColumn && !isInterface) {
                                for (std::int32_t du = -2; du <= 2 && !isInterface; ++du) {
                                    for (std::int32_t dv = -2; dv <= 2; ++dv) {
                                        auto itCol = supportColumns.find(packCellKey(iu + du, iv + dv));
                                        if (itCol != supportColumns.end()) {
                                            const auto& col = itCol->second;
                                            if (hLayer >= col.floorH && hLayer <= col.roofH + 0.5f * effectiveStepH) {
                                                const bool isRoofIntf =
                                                    (impl_->supportSpec.interface_layers > 0) &&
                                                    (col.roofH - hLayer <= interfaceBandMm + 1.0e-4f);
                                                const bool isBedIntf =
                                                    (impl_->supportSpec.bed_interface_layers > 0) &&
                                                    (col.floorH <= 1.0e-3f) &&
                                                    (bedLayerOrdinal < impl_->supportSpec.bed_interface_layers ||
                                                     (hLayer - col.floorH) <= bedInterfaceBandMm + 1.0e-4f);
                                                if (isRoofIntf || isBedIntf) {
                                                    isInterface = true;
                                                    break;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            const std::uint8_t wantedKind = isInterface ? 8 : 7;
                            return inColumn && (wantedKind == featureKind);
                        };

                        bool zig = false;
                        float prevRunOuterCoord = -1.0e9f;
                        for (float outerCoord = outerMin; outerCoord <= outerMax + 1.0e-4f; outerCoord += spacingMm) {
                            std::vector<ToolpathPoint> activeRun;

                            auto flushRun = [&]() {
                                if (activeRun.size() < 2) {
                                    activeRun.clear();
                                    return;
                                }
                                float lenMm = 0.0f;
                                for (std::size_t k = 1; k < activeRun.size(); ++k) {
                                    const float dxSeg = activeRun[k].x - activeRun[k - 1].x;
                                    const float dySeg = activeRun[k].y - activeRun[k - 1].y;
                                    const float dzSeg = activeRun[k].z - activeRun[k - 1].z;
                                    lenMm += std::sqrt(dxSeg * dxSeg + dySeg * dySeg + dzSeg * dzSeg);
                                }
                                if (lenMm >= minSegLenMm) {
                                    if (zig) {
                                        std::reverse(activeRun.begin(), activeRun.end());
                                    }
                                    bool chainedToPrev = false;
                                    // Never chain two broken halves of the SAME scanline (`outerCoord == prevRunOuterCoord`)
                                    // across an obstacle or interface patch! Only chain adjacent scanlines
                                    // (`std::abs(outerCoord - prevRunOuterCoord) >= 0.8f * spacingMm`) when the turn
                                    // midpoint also belongs to the same support feature kind.
                                    if (!supportPathsForLayer.empty() &&
                                        supportPathsForLayer.back().feature_kind == featureKind &&
                                        !supportPathsForLayer.back().points.empty() &&
                                        std::abs(outerCoord - prevRunOuterCoord) >= 0.8f * spacingMm &&
                                        std::abs(outerCoord - prevRunOuterCoord) <= 1.25f * spacingMm) {
                                        const ToolpathPoint& prevEnd = supportPathsForLayer.back().points.back();
                                        const ToolpathPoint& currStart = activeRun.front();
                                        const float tx = currStart.x - prevEnd.x;
                                        const float ty = currStart.y - prevEnd.y;
                                        const float tz = currStart.z - prevEnd.z;
                                        const float turnDist = std::sqrt(tx * tx + ty * ty + tz * tz);
                                        if (turnDist <= 1.35f * spacingMm) {
                                            const std::array<float, 3> midBed{
                                                0.5f * (prevEnd.x + currStart.x),
                                                0.5f * (prevEnd.y + currStart.y),
                                                0.5f * (prevEnd.z + currStart.z)};
                                            const float relMidX = midBed[0] - pOrigin[0];
                                            const float relMidY = midBed[1] - pOrigin[1];
                                            const float relMidZ = midBed[2] - pOrigin[2];
                                            const float uMid = relMidX * tU[0] + relMidY * tU[1] + relMidZ * tU[2];
                                            const float vMid = relMidX * tV[0] + relMidY * tV[1] + relMidZ * tV[2];
                                            bool midRoofContact = false;
                                            const openvdb::Vec3d midPt(midBed[0], midBed[1], midBed[2]);
                                            const float minTurnClear =
                                                (featureKind == 8)
                                                    ? std::min(0.75f * zGapMm, xyGapMm)
                                                    : (xyGapMm + 0.5f * lineWidthMm);
                                            if (isCellValidForKind(uMid, vMid, midRoofContact) &&
                                                samplePartSdf(midPt) >= minTurnClear &&
                                                hasObjectExtrusionClearance(
                                                    midBed,
                                                    0.5f * lineWidthMm,
                                                    0.5f * localLayerH,
                                                    featureKind == 8)) {
                                                auto& prevPath = supportPathsForLayer.back();
                                                for (const auto& pt : activeRun) {
                                                    prevPath.points.push_back(pt);
                                                }
                                                prevPath.total_length_mm += turnDist + lenMm;
                                                chainedToPrev = true;
                                            }
                                        }
                                    }
                                    if (!chainedToPrev) {
                                        ExtrusionPath ep{};
                                        ep.feature_kind = featureKind;
                                        ep.wall_index = supportPassIdx++;
                                        ep.is_closed = false;
                                        ep.total_length_mm = lenMm;
                                        ep.mean_width_mm = lineWidthMm;
                                        ep.points.reserve(activeRun.size());
                                        for (const auto& pt : activeRun) {
                                            ep.points.push_back(pt);
                                        }
                                        supportPathsForLayer.push_back(std::move(ep));
                                        if (featureKind == 7) {
                                            ++res.support_count;
                                        } else {
                                            ++res.support_interface_count;
                                        }
                                        ++res.total_path_count;
                                    }
                                    prevRunOuterCoord = outerCoord;
                                    zig = !zig;
                                }
                                activeRun.clear();
                            };

                            for (float innerCoord = innerMin; innerCoord <= innerMax + 1.0e-4f; innerCoord += marchStepMm) {
                                const float uVal = swapUV ? outerCoord : innerCoord;
                                const float vVal = swapUV ? innerCoord : outerCoord;
                                bool isRoofContact = false;
                                if (!isCellValidForKind(uVal, vVal, isRoofContact)) {
                                    flushRun();
                                    continue;
                                }

                                const std::array<float, 3> pBed = uvHToBed(uVal, vVal, hLayer);

                                if (pBed[0] < rec.minX - bboxMarginMm || pBed[0] > rec.maxX + bboxMarginMm ||
                                    pBed[1] < rec.minY - bboxMarginMm || pBed[1] > rec.maxY + bboxMarginMm ||
                                    pBed[2] < rec.minZ - 1.0e-3f || pBed[2] > rec.maxZ + bboxMarginMm) {
                                    flushRun();
                                    continue;
                                }

                                if (!hasSupportClearanceAtUVH(uVal, vVal, hLayer, 0.0f, isRoofContact)) {
                                    flushRun();
                                    continue;
                                }

                                const std::array<float, 3> travelDir = swapUV ? tV : tU;
                                const NozzleBasis nb = queryNozzleFrame(pBed, travelDir);
                                ToolpathPoint tp{};
                                tp.x = pBed[0];
                                tp.y = pBed[1];
                                tp.z = pBed[2];
                                tp.nx = nb.nozzle_axis[0];
                                tp.ny = nb.nozzle_axis[1];
                                tp.nz = nb.nozzle_axis[2];
                                tp.width_mm = lineWidthMm;
                                tp.height_mm = localLayerH;
                                activeRun.push_back(tp);
                            }
                            flushRun();
                        }
                    };

                    const bool alternateDir = ((layerIdx % 2) == 1);
                    emitScanlineSupportGroup(7, baseSpacingMm, alternateDir);
                    if (impl_->supportSpec.interface_layers > 0 ||
                        impl_->supportSpec.bed_interface_layers > 0) {
                        emitScanlineSupportGroup(8, interfaceSpacingMm, !alternateDir);
                    }
                }

                if (!supportPathsForLayer.empty()) {
                    lastSupportIso = hLayer;
                    // Prepend support paths before object walls on this layer
                    std::vector<ExtrusionPath> combined;
                    combined.reserve(supportPathsForLayer.size() + layer.paths.size());
                    for (auto& sp : supportPathsForLayer) {
                        combined.push_back(std::move(sp));
                    }
                    for (auto& op : layer.paths) {
                        combined.push_back(std::move(op));
                    }
                    layer.paths.clear();
                    for (auto& cp : combined) {
                        layer.paths.push_back(std::move(cp));
                    }
                }
            }
        }
    }

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

void set_order_field_mode(std::uint8_t mode) {
    std::lock_guard<std::mutex> lock(g_backendMutex);
    globalBackend().setOrderFieldMode(mode);
}

std::uint8_t get_order_field_mode() {
    std::lock_guard<std::mutex> lock(g_backendMutex);
    return globalBackend().orderFieldMode();
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

void set_overhang_orientation_bias_spec(OverhangBiasSpec spec) {
    std::lock_guard<std::mutex> lock(g_backendMutex);
    globalBackend().setOverhangOrientationBiasSpec(spec);
}

OverhangBiasSpec get_overhang_orientation_bias_spec() {
    std::lock_guard<std::mutex> lock(g_backendMutex);
    return globalBackend().overhangOrientationBiasSpec();
}

void set_conformal_skin_spec(ConformalSkinSpec spec) {
    std::lock_guard<std::mutex> lock(g_backendMutex);
    globalBackend().setConformalSkinSpec(spec);
}

ConformalSkinSpec get_conformal_skin_spec() {
    std::lock_guard<std::mutex> lock(g_backendMutex);
    return globalBackend().conformalSkinSpec();
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

void set_support_spec(SupportSpec spec) {
    std::lock_guard<std::mutex> lock(g_backendMutex);
    globalBackend().setSupportSpec(spec);
}

SupportSpec get_support_spec() {
    std::lock_guard<std::mutex> lock(g_backendMutex);
    return globalBackend().supportSpec();
}

ObjectToolpathsResult generate_object_shell_toolpaths(std::uint64_t object_id, float adaptivity) {
    std::lock_guard<std::mutex> lock(g_backendMutex);
    return globalBackend().generateObjectShellToolpaths(object_id, adaptivity);
}

} // namespace manifold
