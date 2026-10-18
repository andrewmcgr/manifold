#include "slicer_backend.h"
#include "manifold-frontend/src/bridge.rs.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
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
    std::vector<RenderVertex> vertexBuffer;
    std::vector<std::uint32_t> indexBuffer;
    openvdb::FloatGrid::Ptr eikonalGrid;
};

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
            // The compound consists solely of faces/edges; keep it as a single part
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
void recenterVertexBuffer(
    std::vector<RenderVertex>& vertices,
    float& outOriginX,
    float& outOriginY,
    float& outOriginZ)
{
    if (vertices.empty()) {
        outOriginX = 0.0f;
        outOriginY = 0.0f;
        outOriginZ = 0.0f;
        return;
    }

    float minX = std::numeric_limits<float>::max();
    float minY = std::numeric_limits<float>::max();
    float minZ = std::numeric_limits<float>::max();
    float maxX = std::numeric_limits<float>::lowest();
    float maxY = std::numeric_limits<float>::lowest();
    float maxZ = std::numeric_limits<float>::lowest();

    for (const auto& v : vertices) {
        minX = std::min(minX, v.x);
        minY = std::min(minY, v.y);
        minZ = std::min(minZ, v.z);
        maxX = std::max(maxX, v.x);
        maxY = std::max(maxY, v.y);
        maxZ = std::max(maxZ, v.z);
    }

    outOriginX = 0.5f * (minX + maxX);
    outOriginY = 0.5f * (minY + maxY);
    outOriginZ = minZ;

    for (auto& v : vertices) {
        v.x -= outOriginX;
        v.y -= outOriginY;
        v.z -= outOriginZ;
    }
}

/// Initializes a coarse OpenVDB Eikonal scalar field over the centered vertex bounds.
openvdb::FloatGrid::Ptr createEikonalGridForVertices(
    const std::vector<RenderVertex>& vertices,
    double voxelSize = 0.5)
{
    const float backgroundDistance = static_cast<float>(voxelSize * 3.0);
    openvdb::FloatGrid::Ptr grid = openvdb::FloatGrid::create(backgroundDistance);
    grid->setName("eikonal_nonplanar_field");
    grid->setGridClass(openvdb::GRID_LEVEL_SET);
    grid->setTransform(openvdb::math::Transform::createLinearTransform(voxelSize));

    if (vertices.empty()) {
        return grid;
    }

    float minX = std::numeric_limits<float>::max();
    float minY = std::numeric_limits<float>::max();
    float minZ = std::numeric_limits<float>::max();
    float maxX = std::numeric_limits<float>::lowest();
    float maxY = std::numeric_limits<float>::lowest();
    float maxZ = std::numeric_limits<float>::lowest();

    for (const auto& v : vertices) {
        minX = std::min(minX, v.x);
        minY = std::min(minY, v.y);
        minZ = std::min(minZ, v.z);
        maxX = std::max(maxX, v.x);
        maxY = std::max(maxY, v.y);
        maxZ = std::max(maxZ, v.z);
    }

    // Clamp grid resolution for fast interactive initialization on large CAD parts
    const double maxSpan = std::max({maxX - minX, maxY - minY, maxZ - minZ, 1.0f});
    const double effectiveVoxel = std::max(voxelSize, maxSpan / 32.0);
    grid->setTransform(openvdb::math::Transform::createLinearTransform(effectiveVoxel));

    openvdb::FloatGrid::Accessor accessor = grid->getAccessor();
    const openvdb::math::Transform& xform = grid->transform();

    const openvdb::Coord ijkMin = xform.worldToIndexCellCentered(openvdb::Vec3d(minX, minY, minZ));
    const openvdb::Coord ijkMax = xform.worldToIndexCellCentered(openvdb::Vec3d(maxX, maxY, maxZ));

    for (int i = ijkMin.x(); i <= ijkMax.x(); ++i) {
        for (int j = ijkMin.y(); j <= ijkMax.y(); ++j) {
            for (int k = ijkMin.z(); k <= ijkMax.z(); ++k) {
                const openvdb::Coord ijk(i, j, k);
                const openvdb::Vec3d worldPos = xform.indexToWorld(ijk);
                const double radialSq = worldPos.x() * worldPos.x() + worldPos.y() * worldPos.y();
                const double phi = (worldPos.z() - minZ) + 0.05 * std::sin(0.1 * std::sqrt(radialSq));
                accessor.setValue(ijk, static_cast<float>(phi));
            }
        }
    }

    return grid;
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

} // namespace

struct SlicerBackend::Impl {
    std::uint64_t nextObjectId = 1;
    std::uint64_t lastLoadedId = 0;
    std::unordered_map<std::uint64_t, ObjectRecord> objects;
    std::vector<RenderVertex> emptyVertices;
    std::vector<std::uint32_t> emptyIndices;
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

        recenterVertexBuffer(rec.vertexBuffer, rec.originX, rec.originY, rec.originZ);

        // Shift the stored B-Rep shape by the same local recentering translation so
        // exporting the individual part to STEP preserves its local centered frame.
        gp_Trsf localShift;
        localShift.SetTranslation(gp_Vec(-rec.originX, -rec.originY, -rec.originZ));
        rec.shape.Move(TopLoc_Location(localShift));

        rec.eikonalGrid = createEikonalGridForVertices(rec.vertexBuffer);

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

    // Compute area-weighted vertex normals across all triangles
    for (std::size_t t = 0; t + 2 < indexCount; t += 3) {
        const std::uint32_t idx0 = rec.indexBuffer[t + 0];
        const std::uint32_t idx1 = rec.indexBuffer[t + 1];
        const std::uint32_t idx2 = rec.indexBuffer[t + 2];
        if (idx0 >= numVertices || idx1 >= numVertices || idx2 >= numVertices) {
            continue;
        }

        const auto& v0 = rec.vertexBuffer[idx0];
        const auto& v1 = rec.vertexBuffer[idx1];
        const auto& v2 = rec.vertexBuffer[idx2];

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
            rec.vertexBuffer[idx].nx += cx;
            rec.vertexBuffer[idx].ny += cy;
            rec.vertexBuffer[idx].nz += cz;
        }
    }

    for (auto& v : rec.vertexBuffer) {
        const float len = std::sqrt(v.nx * v.nx + v.ny * v.ny + v.nz * v.nz);
        if (len > 1.0e-7f) {
            v.nx /= len;
            v.ny /= len;
            v.nz /= len;
        } else {
            v.nz = 1.0f;
        }
    }

    recenterVertexBuffer(rec.vertexBuffer, rec.originX, rec.originY, rec.originZ);
    rec.eikonalGrid = createEikonalGridForVertices(rec.vertexBuffer);

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

} // namespace manifold
