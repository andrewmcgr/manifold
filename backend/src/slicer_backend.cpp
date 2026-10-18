#include "slicer_backend.h"
#include "manifold-frontend/src/bridge.rs.h"

#include <cmath>
#include <mutex>

// Open CASCADE (OCCT) headers
#include <BRepBndLib.hxx>
#include <BRepMesh_IncrementalMesh.hxx>
#include <BRep_Tool.hxx>
#include <Bnd_Box.hxx>
#include <IFSelect_ReturnStatus.hxx>
#include <Poly_Triangulation.hxx>
#include <STEPControl_Reader.hxx>
#include <TopAbs_Orientation.hxx>
#include <TopAbs_ShapeEnum.hxx>
#include <TopExp_Explorer.hxx>
#include <TopLoc_Location.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Face.hxx>
#include <TopoDS_Shape.hxx>
#include <gp_Dir.hxx>
#include <gp_Pnt.hxx>
#include <gp_Trsf.hxx>
#include <gp_Vec.hxx>

// OpenVDB headers
#include <openvdb/openvdb.h>
#include <openvdb/math/Transform.h>

namespace manifold {

struct SlicerBackend::Impl {
    TopoDS_Shape shape;
    std::vector<RenderVertex> vertexBuffer;
    std::vector<std::uint32_t> indexBuffer;
    openvdb::FloatGrid::Ptr eikonalGrid;

    void tessellate(double linearDeflection, double angularDeflection) {
        vertexBuffer.clear();
        indexBuffer.clear();

        if (shape.IsNull()) {
            return;
        }

        // Generate surface triangulation across all faces in parallel
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

            // Populate positions and initialize accumulator normals to zero
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

            // Extract triangles and accumulate area-weighted vertex normals
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

            // Normalize accumulated vertex normals for this face
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
};

SlicerBackend::SlicerBackend() : impl_(std::make_unique<Impl>()) {
    openvdb::initialize();
}

SlicerBackend::~SlicerBackend() = default;

bool SlicerBackend::loadStepFile(const std::string& filePath, double linearDeflection, double angularDeflection) {
    STEPControl_Reader reader;
    const IFSelect_ReturnStatus status = reader.ReadFile(filePath.c_str());
    if (status != IFSelect_RetDone) {
        return false;
    }

    reader.TransferRoots();
    impl_->shape = reader.OneShape();
    if (impl_->shape.IsNull()) {
        return false;
    }

    impl_->tessellate(linearDeflection, angularDeflection);
    initializeEikonalField();
    return true;
}

void SlicerBackend::initializeEikonalField(double voxelSize) {
    // Create a narrow-band scalar FloatGrid with exterior background value
    const float backgroundDistance = static_cast<float>(voxelSize * 3.0);
    impl_->eikonalGrid = openvdb::FloatGrid::create(backgroundDistance);
    impl_->eikonalGrid->setName("eikonal_nonplanar_field");
    impl_->eikonalGrid->setGridClass(openvdb::GRID_LEVEL_SET);
    impl_->eikonalGrid->setTransform(openvdb::math::Transform::createLinearTransform(voxelSize));

    if (impl_->shape.IsNull()) {
        return;
    }

    Bnd_Box bbox;
    BRepBndLib::Add(impl_->shape, bbox);
    if (bbox.IsVoid()) {
        return;
    }

    Standard_Real xMin = 0.0, yMin = 0.0, zMin = 0.0;
    Standard_Real xMax = 0.0, yMax = 0.0, zMax = 0.0;
    bbox.Get(xMin, yMin, zMin, xMax, yMax, zMax);

    // Placeholder Eikonal field initialization: seed a smoothly curved
    // non-planar scalar potential phi(x, y, z) across the coarse bounding lattice
    openvdb::FloatGrid::Accessor accessor = impl_->eikonalGrid->getAccessor();
    const openvdb::math::Transform& xform = impl_->eikonalGrid->transform();

    const openvdb::Coord ijkMin = xform.worldToIndexCellCentered(openvdb::Vec3d(xMin, yMin, zMin));
    const openvdb::Coord ijkMax = xform.worldToIndexCellCentered(openvdb::Vec3d(xMax, yMax, zMax));

    // Seed boundary condition on the base plate (k == ijkMin.z()) where phi = 0,
    // and initialize interior voxels with a geodesic distance estimate
    for (int i = ijkMin.x(); i <= ijkMax.x(); ++i) {
        for (int j = ijkMin.y(); j <= ijkMax.y(); ++j) {
            for (int k = ijkMin.z(); k <= ijkMax.z(); ++k) {
                const openvdb::Coord ijk(i, j, k);
                const openvdb::Vec3d worldPos = xform.indexToWorld(ijk);

                // Stubbed non-planar Eikonal distance potential:
                // height above build plate modulated by radial harmonic curvature
                const double radialSq = worldPos.x() * worldPos.x() + worldPos.y() * worldPos.y();
                const double phi = (worldPos.z() - zMin) + 0.05 * std::sin(0.1 * std::sqrt(radialSq));
                accessor.setValue(ijk, static_cast<float>(phi));
            }
        }
    }
}

const std::vector<RenderVertex>& SlicerBackend::vertices() const noexcept {
    return impl_->vertexBuffer;
}

const std::vector<std::uint32_t>& SlicerBackend::indices() const noexcept {
    return impl_->indexBuffer;
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

} // namespace manifold
