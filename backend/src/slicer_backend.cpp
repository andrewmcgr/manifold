#include "slicer_backend.h"
#include "manifold-frontend/src/bridge.rs.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <map>
#include <mutex>
#include <unordered_map>

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
    /// Generic dictionary of named scalar fields (e.g., "sdf", "eikonal", "overhang").
    std::map<std::string, openvdb::FloatGrid::Ptr> scalarFields;
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

/// Populates the object's scalar field registry using the queried nozzle local frame
/// and gravity direction vector in bed coordinates (never assuming nozzle = +Z or gravity = -Z):
/// 1. `"sdf"`: Full-interior + exterior-band Signed Distance Field via
///    `openvdb::tools::meshToSignedDistanceField`.
/// 2. `"eikonal"`: Non-planar conformal slicing distance field derived from height along the
///    queried `nozzle_axis` & SDF.
/// 3. `"overhang"`: Normalized SDF gradient projected against the queried gravity vector
///    ($-\nabla \phi \cdot \hat{\mathbf{g}}_{\text{bed}} \in [-1, 1]$).
void buildObjectScalarFields(
    ObjectRecord& rec,
    const SlicerBackend& backend,
    float requestedVoxelSize = 0.5f,
    float narrowBandVoxels = 6.0f)
{
    rec.scalarFields.clear();
    if (rec.vertexBuffer.empty() || rec.indexBuffer.size() < 3) {
        return;
    }

    const float spanX = rec.maxX - rec.minX;
    const float spanY = rec.maxY - rec.minY;
    const float spanZ = rec.maxZ - rec.minZ;
    const float maxSpan = std::max({spanX, spanY, spanZ, 1.0f});

    // Allow fine voxel sizes up to ~220 voxels along the longest bounding box axis
    const double voxelSize = std::max<double>(
        std::max(requestedVoxelSize, 0.05f),
        static_cast<double>(maxSpan) / 220.0);

    // Maintain a consistent physical exterior band (at least 20% of maxSpan or narrowBandVoxels * voxelSize)
    // and a full interior band (60% of maxSpan) so the SDF isovalue range in millimeters
    // never shrinks when the user selects a finer voxel resolution.
    const float minExteriorMm = std::max(0.20f * maxSpan, narrowBandVoxels * static_cast<float>(voxelSize));
    const float exBandVoxels = std::clamp(
        minExteriorMm / static_cast<float>(voxelSize),
        3.0f,
        120.0f);
    const float inBandVoxels = std::clamp(
        (0.60f * maxSpan) / static_cast<float>(voxelSize),
        4.0f,
        220.0f);

    openvdb::math::Transform::Ptr xform =
        openvdb::math::Transform::createLinearTransform(voxelSize);

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

    // 1. Compute full-interior + exterior Signed Distance Field ("sdf")
    openvdb::FloatGrid::Ptr sdfGrid =
        openvdb::tools::meshToSignedDistanceField<openvdb::FloatGrid>(
            *xform, points, triangles, quads, exBandVoxels, inBandVoxels);
    sdfGrid->setName("sdf");
    rec.scalarFields["sdf"] = sdfGrid;

    // 2. Build "eikonal" (conformal non-planar layer field) and "overhang" (gravity-aligned gradient)
    const float eikonalBg = spanZ + minExteriorMm;
    openvdb::FloatGrid::Ptr eikonalGrid = openvdb::FloatGrid::create(eikonalBg);
    eikonalGrid->setName("eikonal");
    eikonalGrid->setGridClass(openvdb::GRID_LEVEL_SET);
    eikonalGrid->setTransform(xform->copy());

    openvdb::FloatGrid::Ptr overhangGrid = openvdb::FloatGrid::create(0.0f);
    overhangGrid->setName("overhang");
    overhangGrid->setGridClass(openvdb::GRID_LEVEL_SET);
    overhangGrid->setTransform(xform->copy());

    auto sdfAccessor = sdfGrid->getConstAccessor();
    auto eikonalAccessor = eikonalGrid->getAccessor();
    auto overhangAccessor = overhangGrid->getAccessor();

    const float invSpanZ = 1.0f / std::max(spanZ, 1.0f);
    const float invTwoDx = 1.0f / static_cast<float>(2.0 * voxelSize);

    for (auto iter = sdfGrid->cbeginValueOn(); iter; ++iter) {
        const openvdb::Coord ijk = iter.getCoord();
        const float sdfVal = *iter;
        const openvdb::Vec3d worldPos = xform->indexToWorld(ijk);
        const std::array<float, 3> posBed{
            static_cast<float>(worldPos.x()),
            static_cast<float>(worldPos.y()),
            static_cast<float>(worldPos.z())};

        // Query the nozzle's local orthonormal frame and the gravity direction vector in bed coordinates
        const NozzleBasis nozzleFrame = backend.queryNozzleFrame(posBed, {1.0f, 0.0f, 0.0f});
        const std::array<float, 3> gravityDir = backend.queryGravityDirection(posBed);

        // Project position relative to the object base onto the queried nozzle axis
        const float relX = posBed[0];
        const float relY = posBed[1];
        const float relZ = posBed[2] - rec.minZ;
        const float axialHeight =
            relX * nozzleFrame.nozzle_axis[0] +
            relY * nozzleFrame.nozzle_axis[1] +
            relZ * nozzleFrame.nozzle_axis[2];
        const float axialRel = std::clamp(axialHeight * invSpanZ, 0.0f, 1.0f);

        const float lateralU =
            relX * nozzleFrame.travel_axis[0] +
            relY * nozzleFrame.travel_axis[1] +
            relZ * nozzleFrame.travel_axis[2];
        const float lateralV =
            relX * nozzleFrame.lateral_axis[0] +
            relY * nozzleFrame.lateral_axis[1] +
            relZ * nozzleFrame.lateral_axis[2];
        const double radial = std::sqrt(lateralU * lateralU + lateralV * lateralV);

        const float harmonicWave =
            0.12f * maxSpan * axialRel * static_cast<float>(std::sin(3.5 * radial / maxSpan));
        const float conformalOffset = -0.35f * axialRel * sdfVal;
        const float eikonalVal = axialHeight + harmonicWave + conformalOffset;
        eikonalAccessor.setValue(ijk, eikonalVal);

        // Central-difference 3D SDF gradient projected against the queried gravity vector (-g_bed)
        const float gx =
            (sdfAccessor.getValue(ijk.offsetBy(1, 0, 0)) -
             sdfAccessor.getValue(ijk.offsetBy(-1, 0, 0))) *
            invTwoDx;
        const float gy =
            (sdfAccessor.getValue(ijk.offsetBy(0, 1, 0)) -
             sdfAccessor.getValue(ijk.offsetBy(0, -1, 0))) *
            invTwoDx;
        const float gz =
            (sdfAccessor.getValue(ijk.offsetBy(0, 0, 1)) -
             sdfAccessor.getValue(ijk.offsetBy(0, 0, -1))) *
            invTwoDx;
        const float antiGravityProj =
            -(gx * gravityDir[0] + gy * gravityDir[1] + gz * gravityDir[2]);
        overhangAccessor.setValue(ijk, std::clamp(antiGravityProj, -1.0f, 1.0f));
    }

    rec.scalarFields["eikonal"] = eikonalGrid;
    rec.scalarFields["overhang"] = overhangGrid;
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
    std::vector<std::uint32_t>& indexBuffer)
{
    vertexBuffer.clear();
    indexBuffer.clear();

    if (shape.IsNull()) {
        return;
    }

    BRepMesh_IncrementalMesh mesher(shape, linearDeflection, Standard_False, angularDeflection, Standard_True);
    mesher.Perform();

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
    }
}

ToolSpec makeDefaultTool(std::uint32_t id, const char* name, float diameterMm, float tempC) {
    ToolSpec t{};
    t.id = id;
    t.name = rust::String(name);
    t.nozzle_diameter_mm = diameterMm;
    t.temperature_celsius = tempC;
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
    LayerHeightSpec sliceLayerHeight{0.20f, 0.08f, 0.32f};
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
            rec.indexBuffer);

        if (rec.vertexBuffer.empty() || rec.indexBuffer.empty()) {
            continue;
        }

        recenterVertexBuffer(rec);

        gp_Trsf localShift;
        localShift.SetTranslation(gp_Vec(-rec.originX, -rec.originY, -rec.originZ));
        rec.shape.Move(TopLoc_Location(localShift));

        buildObjectScalarFields(rec, *this);

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
    buildObjectScalarFields(rec, *this);

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
    const auto it = impl_->objects.find(objectId);
    if (it == impl_->objects.end()) {
        return false;
    }
    buildObjectScalarFields(it->second, *this, voxelSize, narrowBandVoxels);
    return true;
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

    const openvdb::FloatGrid& grid = *fieldIt->second;
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
    const openvdb::FloatGrid& sdfGrid = *sdfIt->second;
    const float sdfBg = sdfGrid.background();
    const float voxelSize = static_cast<float>(sdfGrid.transform().voxelSize()[0]);
    // Strip the artificial closure shell generated by volumeToMesh where active narrow-band
    // voxels jump discontinuously to the background value at the outer fringe of the band.
    const float outerFringeLimit = sdfBg - 1.75f * voxelSize;

    auto sdfAccessor = sdfGrid.getConstAccessor();
    openvdb::tools::GridSampler<openvdb::FloatGrid::ConstAccessor, openvdb::tools::BoxSampler>
        sdfSampler(sdfAccessor, sdfGrid.transform());

    std::vector<float> vertSdf(verts.size(), 0.0f);
    for (std::size_t i = 0; i < verts.size(); ++i) {
        const openvdb::Vec3d wp(verts[i].x, verts[i].y, verts[i].z);
        vertSdf[i] = sdfSampler.wsSample(wp);
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
            slice.values.push_back(sampler.wsSample(wp));
            slice.sdf_values.push_back(sdfSampler.wsSample(wp));
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
std::mutex g_backendMutex;
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
    std::lock_guard<std::mutex> lock(g_backendMutex);
    return globalBackend().recomputeObjectFields(object_id, voxel_size, narrow_band_voxels);
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
    impl_->sliceLayerHeight = s;
}

LayerHeightSpec SlicerBackend::sliceLayerHeight() const noexcept {
    return impl_->sliceLayerHeight;
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

} // namespace manifold
