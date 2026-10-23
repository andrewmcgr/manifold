#include "path_generator.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <queue>
#include <unordered_map>
#include <utility>
#include <vector>

#include <openvdb/tools/Interpolation.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

// Include the CXX-generated header so shared Rust/C++ structs (`ShellPathSpec`,
// `ExtrusionPath`, `LayerToolpaths`, `ObjectToolpathsResult`, etc.) are complete types.
#include "manifold-frontend/src/bridge.rs.h"

namespace manifold {
namespace {

constexpr std::uint8_t kFeatureOuterWall = 0;
constexpr std::uint8_t kFeatureInnerWall = 1;
constexpr std::uint8_t kFeatureSolidInfill = 2;
constexpr std::uint8_t kFeatureGapFill = 3;
constexpr std::uint8_t kFeatureSparseInfill = 4;

constexpr std::uint8_t kPatternConcentric = 1;

constexpr std::uint8_t kSparsePatternMonotonic = 0;
constexpr std::uint8_t kSparsePatternCubic = 1;
constexpr std::uint8_t kSparsePatternGyroid = 2;
constexpr std::uint8_t kSparsePatternSchwarzD = 3;
constexpr std::uint8_t kSparsePatternSchwarzP = 4;
constexpr std::uint8_t kSparsePatternConcentric = 5;
constexpr std::uint8_t kSparsePatternAllWalls = 6;

/// Standard anchor overlap fraction $\eta \in [0, 0.5]$ between infill line caps and adjacent
/// perimeter/shell boundaries. With semicircular nozzle cap radius $0.5 w$, the nozzle centerline
/// is inset by $(0.5 - \eta) w = 0.35 w$ from the physical inner edge of the adjacent boundary.
constexpr float kInfillAnchorOverlapFraction = 0.15f;
constexpr float kInfillEndpointInsetFactor = 0.5f - kInfillAnchorOverlapFraction; // 0.35f

struct Vec3f {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;

    Vec3f operator+(const Vec3f& o) const noexcept { return {x + o.x, y + o.y, z + o.z}; }
    Vec3f operator-(const Vec3f& o) const noexcept { return {x - o.x, y - o.y, z - o.z}; }
    Vec3f operator*(float s) const noexcept { return {x * s, y * s, z * s}; }
};

inline float dot3(const Vec3f& a, const Vec3f& b) noexcept {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline Vec3f cross3(const Vec3f& a, const Vec3f& b) noexcept {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    };
}

inline float lengthSq3(const Vec3f& v) noexcept {
    return v.x * v.x + v.y * v.y + v.z * v.z;
}

inline float length3(const Vec3f& v) noexcept {
    return std::sqrt(lengthSq3(v));
}

inline float dist3(const Vec3f& a, const Vec3f& b) noexcept {
    return length3(a - b);
}

inline Vec3f normalize3(const Vec3f& v, const Vec3f& fallback = {0.0f, 0.0f, 1.0f}) noexcept {
    const float len = length3(v);
    if (len > 1.0e-7f) {
        return {v.x / len, v.y / len, v.z / len};
    }
    return fallback;
}

inline Vec3f lerp3(const Vec3f& a, const Vec3f& b, float t) noexcept {
    return {
        a.x + t * (b.x - a.x),
        a.y + t * (b.y - a.y),
        a.z + t * (b.z - a.z),
    };
}

struct SanitizedLineWidth {
    float target_mm = 0.40f;
    float min_mm = 0.28f;
    float max_mm = 0.56f;
};

SanitizedLineWidth sanitizeLineWidth(const LineWidthSpec& raw) noexcept {
    SanitizedLineWidth out{};
    out.min_mm = std::max(0.05f, raw.min_mm);
    out.max_mm = std::max(out.min_mm, raw.max_mm);
    out.target_mm = std::clamp(raw.target_mm, out.min_mm, out.max_mm);
    return out;
}

struct GridKey3D {
    std::int64_t ix = 0;
    std::int64_t iy = 0;
    std::int64_t iz = 0;

    bool operator==(const GridKey3D& o) const noexcept {
        return ix == o.ix && iy == o.iy && iz == o.iz;
    }
};

struct GridKey3DHash {
    std::size_t operator()(const GridKey3D& k) const noexcept {
        std::size_t h = static_cast<std::size_t>(k.ix) * 73856093u;
        h ^= static_cast<std::size_t>(k.iy) * 19349663u;
        h ^= static_cast<std::size_t>(k.iz) * 83492791u;
        return h;
    }
};

inline GridKey3D toGridKey(const Vec3f& p, float invCell) noexcept {
    return GridKey3D{
        static_cast<std::int64_t>(std::llround(p.x * invCell)),
        static_cast<std::int64_t>(std::llround(p.y * invCell)),
        static_cast<std::int64_t>(std::llround(p.z * invCell)),
    };
}

struct EdgeKey {
    std::uint32_t a = 0;
    std::uint32_t b = 0;

    EdgeKey(std::uint32_t u, std::uint32_t v) noexcept
        : a(std::min(u, v)), b(std::max(u, v)) {}

    bool operator==(const EdgeKey& o) const noexcept {
        return a == o.a && b == o.b;
    }
};

struct EdgeKeyHash {
    std::size_t operator()(const EdgeKey& e) const noexcept {
        return (static_cast<std::size_t>(e.a) << 32) ^ static_cast<std::size_t>(e.b);
    }
};

struct SurfaceVertex {
    Vec3f pos{};
    Vec3f normal{};
    Vec3f nozzleAxis{};
    float sdf = 0.0f;
    float layerHeightMm = 0.20f;
    float dRem = 0.0f;
    float remSpan = 0.0f;
    bool onOuterBoundary = false;
    bool onInnerShellBoundary = false;
};

struct SurfaceMesh {
    std::vector<SurfaceVertex> verts;
    std::vector<std::array<std::uint32_t, 3>> tris;
};

SurfaceVertex lerpSurfaceVertex(const SurfaceVertex& a, const SurfaceVertex& b, float t) noexcept {
    SurfaceVertex out{};
    out.pos = lerp3(a.pos, b.pos, t);
    out.normal = normalize3(lerp3(a.normal, b.normal, t), a.normal);
    out.nozzleAxis = normalize3(lerp3(a.nozzleAxis, b.nozzleAxis, t), a.nozzleAxis);
    out.sdf = a.sdf + t * (b.sdf - a.sdf);
    out.layerHeightMm = a.layerHeightMm + t * (b.layerHeightMm - a.layerHeightMm);
    out.dRem = a.dRem + t * (b.dRem - a.dRem);
    out.remSpan = a.remSpan + t * (b.remSpan - a.remSpan);
    out.onOuterBoundary = false;
    out.onInnerShellBoundary = a.onInnerShellBoundary && b.onInnerShellBoundary;
    return out;
}

/// Welds coincident vertices within `weldEps` and drops degenerate triangles.
SurfaceMesh weldSurfaceMesh(const SurfaceMesh& inMesh, float weldEps = 1.0e-4f) {
    SurfaceMesh out;
    if (inMesh.verts.empty() || inMesh.tris.empty()) {
        return out;
    }

    const float invCell = 1.0f / std::max(weldEps, 1.0e-6f);
    std::unordered_map<GridKey3D, std::uint32_t, GridKey3DHash> cellToVert;
    cellToVert.reserve(inMesh.verts.size());

    std::vector<std::uint32_t> remap(inMesh.verts.size(), 0);
    out.verts.reserve(inMesh.verts.size());

    for (std::size_t i = 0; i < inMesh.verts.size(); ++i) {
        const auto& v = inMesh.verts[i];
        const GridKey3D key = toGridKey(v.pos, invCell);
        auto it = cellToVert.find(key);
        if (it != cellToVert.end()) {
            const std::uint32_t idx = it->second;
            remap[i] = idx;
            out.verts[idx].onOuterBoundary = out.verts[idx].onOuterBoundary || v.onOuterBoundary;
            out.verts[idx].onInnerShellBoundary =
                out.verts[idx].onInnerShellBoundary || v.onInnerShellBoundary;
        } else {
            const auto newIdx = static_cast<std::uint32_t>(out.verts.size());
            out.verts.push_back(v);
            cellToVert.emplace(key, newIdx);
            remap[i] = newIdx;
        }
    }

    out.tris.reserve(inMesh.tris.size());
    for (const auto& tri : inMesh.tris) {
        const std::uint32_t i0 = remap[tri[0]];
        const std::uint32_t i1 = remap[tri[1]];
        const std::uint32_t i2 = remap[tri[2]];
        if (i0 == i1 || i1 == i2 || i2 == i0) {
            continue;
        }
        const Vec3f e1 = out.verts[i1].pos - out.verts[i0].pos;
        const Vec3f e2 = out.verts[i2].pos - out.verts[i0].pos;
        if (lengthSq3(cross3(e1, e2)) <= 1.0e-12f) {
            continue;
        }
        out.tris.push_back({i0, i1, i2});
    }
    return out;
}

/// Conformingly subdivides edges in the already-clipped shell mesh (both edges longer than
/// `maxEdgeLen` and interior cross-strip edges connecting two outer-boundary vertices) so
/// contour extraction and medial-axis detection have interior vertices even on 1-triangle-wide
/// top-layer strips without introducing T-junctions.
template <typename SdfSamplerFn>
SurfaceMesh subdivideShellMesh(
    const SurfaceMesh& inMesh,
    float maxEdgeLen,
    SdfSamplerFn&& sampleSdfAt)
{
    if (inMesh.tris.empty() || inMesh.verts.empty()) {
        return inMesh;
    }

    SurfaceMesh current = inMesh;
    const float maxEdgeSq = maxEdgeLen * maxEdgeLen;

    for (int pass = 0; pass < 2; ++pass) {
        if (current.tris.size() > 6000) {
            break;
        }

        std::unordered_map<EdgeKey, std::uint32_t, EdgeKeyHash> edgeTriCounts;
        edgeTriCounts.reserve(current.tris.size() * 3);
        for (const auto& tri : current.tris) {
            edgeTriCounts[EdgeKey(tri[0], tri[1])] += 1;
            edgeTriCounts[EdgeKey(tri[1], tri[2])] += 1;
            edgeTriCounts[EdgeKey(tri[2], tri[0])] += 1;
        }

        std::unordered_map<EdgeKey, std::uint32_t, EdgeKeyHash> edgeMidpoints;
        edgeMidpoints.reserve(current.tris.size() * 2);

        auto shouldSplitEdge = [&](std::uint32_t u, std::uint32_t v) -> bool {
            const float lenSq = lengthSq3(current.verts[v].pos - current.verts[u].pos);
            if (lenSq > maxEdgeSq) {
                return true;
            }
            // Always split interior edges whose endpoints both lie on the outer boundary
            // (1-triangle-wide thin strips) so a centerline vertex is created inside the strip.
            if (current.verts[u].onOuterBoundary && current.verts[v].onOuterBoundary) {
                auto it = edgeTriCounts.find(EdgeKey(u, v));
                if (it != edgeTriCounts.end() && it->second >= 2 && lenSq > 1.0e-5f) {
                    return true;
                }
            }
            return false;
        };

        auto getMidpoint = [&](std::uint32_t u, std::uint32_t v) -> std::uint32_t {
            const EdgeKey key(u, v);
            auto it = edgeMidpoints.find(key);
            if (it != edgeMidpoints.end()) {
                return it->second;
            }
            SurfaceVertex mid = lerpSurfaceVertex(current.verts[u], current.verts[v], 0.5f);
            auto cntIt = edgeTriCounts.find(key);
            const bool isTrueOuterBoundaryEdge =
                (cntIt != edgeTriCounts.end() && cntIt->second == 1) &&
                current.verts[u].onOuterBoundary &&
                current.verts[v].onOuterBoundary;
            mid.onOuterBoundary = isTrueOuterBoundaryEdge;
            if (!mid.onOuterBoundary) {
                const float wsSdf = sampleSdfAt(mid.pos);
                mid.sdf = 0.5f * (mid.sdf + wsSdf);
            } else {
                mid.sdf = 0.0f;
            }
            const auto newIdx = static_cast<std::uint32_t>(current.verts.size());
            current.verts.push_back(mid);
            edgeMidpoints.emplace(key, newIdx);
            return newIdx;
        };

        std::vector<std::array<std::uint32_t, 3>> nextTris;
        nextTris.reserve(current.tris.size() * 2);
        bool anySplit = false;

        for (const auto& tri : current.tris) {
            const std::uint32_t i0 = tri[0];
            const std::uint32_t i1 = tri[1];
            const std::uint32_t i2 = tri[2];

            const bool s01 = shouldSplitEdge(i0, i1);
            const bool s12 = shouldSplitEdge(i1, i2);
            const bool s20 = shouldSplitEdge(i2, i0);
            const int splitCount = static_cast<int>(s01) + static_cast<int>(s12) + static_cast<int>(s20);

            if (splitCount == 0) {
                nextTris.push_back(tri);
            } else if (splitCount == 3) {
                anySplit = true;
                const std::uint32_t m01 = getMidpoint(i0, i1);
                const std::uint32_t m12 = getMidpoint(i1, i2);
                const std::uint32_t m20 = getMidpoint(i2, i0);
                nextTris.push_back({i0, m01, m20});
                nextTris.push_back({m01, i1, m12});
                nextTris.push_back({m20, m12, i2});
                nextTris.push_back({m01, m12, m20});
            } else if (splitCount == 2) {
                anySplit = true;
                // Rotate indices so edges (a, b) and (b, c) are the two split edges
                std::uint32_t a = i0, b = i1, c = i2;
                if (!s01) {
                    a = i1; b = i2; c = i0;
                } else if (!s12) {
                    a = i2; b = i0; c = i1;
                }
                const std::uint32_t mab = getMidpoint(a, b);
                const std::uint32_t mbc = getMidpoint(b, c);
                nextTris.push_back({mab, b, mbc});
                if (lengthSq3(current.verts[a].pos - current.verts[mbc].pos) <=
                    lengthSq3(current.verts[c].pos - current.verts[mab].pos)) {
                    nextTris.push_back({a, mab, mbc});
                    nextTris.push_back({a, mbc, c});
                } else {
                    nextTris.push_back({a, mab, c});
                    nextTris.push_back({mab, mbc, c});
                }
            } else {
                // splitCount == 1: rotate indices so edge (a, b) is the single split edge
                anySplit = true;
                std::uint32_t a = i0, b = i1, c = i2;
                if (s12) {
                    a = i1; b = i2; c = i0;
                } else if (s20) {
                    a = i2; b = i0; c = i1;
                }
                const std::uint32_t mab = getMidpoint(a, b);
                nextTris.push_back({a, mab, c});
                nextTris.push_back({mab, b, c});
            }
        }

        current.tris = std::move(nextTris);
        if (!anySplit) {
            break;
        }
    }

    return current;
}

/// Clips a surface mesh to the region where `scalarField(v) <= maxAllowedValue`.
/// Any new vertices created on the clipping boundary have `markInnerBoundary` applied if requested.
SurfaceMesh clipSurfaceMeshByScalar(
    const SurfaceMesh& inMesh,
    const std::vector<float>& scalarAtVert,
    float maxAllowedValue,
    bool markInnerBoundary)
{
    SurfaceMesh out;
    if (inMesh.tris.empty() || scalarAtVert.size() != inMesh.verts.size()) {
        return out;
    }

    std::vector<std::int32_t> vertRemap(inMesh.verts.size(), -1);
    std::unordered_map<EdgeKey, std::uint32_t, EdgeKeyHash> cutEdgeCache;

    auto emitExisting = [&](std::uint32_t oldIdx) -> std::uint32_t {
        if (vertRemap[oldIdx] >= 0) {
            return static_cast<std::uint32_t>(vertRemap[oldIdx]);
        }
        const auto newIdx = static_cast<std::uint32_t>(out.verts.size());
        SurfaceVertex sv = inMesh.verts[oldIdx];
        if (markInnerBoundary && std::abs(scalarAtVert[oldIdx] - maxAllowedValue) <= 1.0e-5f) {
            sv.onInnerShellBoundary = true;
        }
        out.verts.push_back(sv);
        vertRemap[oldIdx] = static_cast<std::int32_t>(newIdx);
        return newIdx;
    };

    auto emitCut = [&](std::uint32_t u, std::uint32_t v) -> std::uint32_t {
        const EdgeKey key(u, v);
        auto it = cutEdgeCache.find(key);
        if (it != cutEdgeCache.end()) {
            return it->second;
        }
        const float du = scalarAtVert[u] - maxAllowedValue;
        const float dv = scalarAtVert[v] - maxAllowedValue;
        const float denom = dv - du;
        const float t = (std::abs(denom) > 1.0e-8f)
            ? std::clamp(-du / denom, 0.0f, 1.0f)
            : 0.5f;
        SurfaceVertex cv = lerpSurfaceVertex(inMesh.verts[u], inMesh.verts[v], t);
        if (markInnerBoundary) {
            cv.onInnerShellBoundary = true;
            cv.onOuterBoundary = false;
        }
        const auto newIdx = static_cast<std::uint32_t>(out.verts.size());
        out.verts.push_back(cv);
        cutEdgeCache.emplace(key, newIdx);
        return newIdx;
    };

    for (const auto& tri : inMesh.tris) {
        const std::uint32_t i0 = tri[0];
        const std::uint32_t i1 = tri[1];
        const std::uint32_t i2 = tri[2];

        const float d0 = scalarAtVert[i0] - maxAllowedValue;
        const float d1 = scalarAtVert[i1] - maxAllowedValue;
        const float d2 = scalarAtVert[i2] - maxAllowedValue;

        const bool in0 = (d0 <= 0.0f);
        const bool in1 = (d1 <= 0.0f);
        const bool in2 = (d2 <= 0.0f);

        if (!in0 && !in1 && !in2) {
            continue;
        }
        if (in0 && in1 && in2) {
            out.tris.push_back({emitExisting(i0), emitExisting(i1), emitExisting(i2)});
            continue;
        }

        std::uint32_t idxA = i0, idxB = i1, idxC = i2;
        bool soloInside = in0;
        if (in0 == in1) {
            idxA = i2; idxB = i0; idxC = i1;
            soloInside = in2;
        } else if (in0 == in2) {
            idxA = i1; idxB = i2; idxC = i0;
            soloInside = in1;
        }

        const std::uint32_t vAB = emitCut(idxA, idxB);
        const std::uint32_t vAC = emitCut(idxA, idxC);

        if (soloInside) {
            const std::uint32_t nA = emitExisting(idxA);
            out.tris.push_back({nA, vAB, vAC});
        } else {
            const std::uint32_t nB = emitExisting(idxB);
            const std::uint32_t nC = emitExisting(idxC);
            out.tris.push_back({vAB, nB, nC});
            out.tris.push_back({vAB, nC, vAC});
        }
    }

    return weldSurfaceMesh(out, 5.0e-5f);
}

struct Segment3D {
    SurfaceVertex a{};
    SurfaceVertex b{};
    float widthA = 0.40f;
    float widthB = 0.40f;
};

/// Intersects a single triangle `tri` with the iso-level `isoValue` of `fieldAtVert`,
/// clipping against `validityAtVert >= minValidity`. Returns true and populates `outSeg` if valid.
inline bool intersectTriangleWithIsoLevel(
    const SurfaceMesh& mesh,
    const std::array<std::uint32_t, 3>& tri,
    const std::vector<float>& fieldAtVert,
    float isoValue,
    const std::vector<float>& validityAtVert,
    float minValidity,
    const std::vector<float>& widthAtVert,
    Segment3D& outSeg)
{
    const std::uint32_t idx[3] = {tri[0], tri[1], tri[2]};
    float fp[3] = {
        fieldAtVert[idx[0]] - isoValue,
        fieldAtVert[idx[1]] - isoValue,
        fieldAtVert[idx[2]] - isoValue,
    };

    for (int k = 0; k < 3; ++k) {
        if (std::abs(fp[k]) < 1.0e-7f) {
            fp[k] = 1.0e-7f;
        }
    }

    if ((fp[0] > 0.0f && fp[1] > 0.0f && fp[2] > 0.0f) ||
        (fp[0] < 0.0f && fp[1] < 0.0f && fp[2] < 0.0f)) {
        return false;
    }

    struct Crossing {
        SurfaceVertex v{};
        float validity = 0.0f;
        float width = 0.40f;
    };
    Crossing crossings[3];
    int numCrossings = 0;

    for (int e = 0; e < 3; ++e) {
        const int eNext = (e + 1) % 3;
        if ((fp[e] < 0.0f && fp[eNext] > 0.0f) || (fp[e] > 0.0f && fp[eNext] < 0.0f)) {
            const std::uint32_t u = idx[e];
            const std::uint32_t v = idx[eNext];
            const float denom = fp[eNext] - fp[e];
            const float t = (std::abs(denom) > 1.0e-9f)
                ? std::clamp(-fp[e] / denom, 0.0f, 1.0f)
                : 0.5f;

            Crossing c{};
            c.v = lerpSurfaceVertex(mesh.verts[u], mesh.verts[v], t);
            c.validity = validityAtVert[u] + t * (validityAtVert[v] - validityAtVert[u]);
            c.width = widthAtVert[u] + t * (widthAtVert[v] - widthAtVert[u]);
            crossings[numCrossings++] = c;
            if (numCrossings == 2) {
                break;
            }
        }
    }

    if (numCrossings != 2) {
        return false;
    }

    Crossing c0 = crossings[0];
    Crossing c1 = crossings[1];

    // Orient segment consistently so that higher `field` is to the left (normal × tangent)
    const Vec3f triNormal = normalize3(
        cross3(
            mesh.verts[idx[1]].pos - mesh.verts[idx[0]].pos,
            mesh.verts[idx[2]].pos - mesh.verts[idx[0]].pos),
        c0.v.normal);
    const Vec3f gradDir =
        (mesh.verts[idx[0]].pos * fp[0]) +
        (mesh.verts[idx[1]].pos * fp[1]) +
        (mesh.verts[idx[2]].pos * fp[2]);
    const Vec3f segDir = c1.v.pos - c0.v.pos;
    if (dot3(cross3(triNormal, segDir), gradDir) < 0.0f) {
        std::swap(c0, c1);
    }

    // Clip segment against `validity >= minValidity`
    const float val0 = c0.validity - minValidity;
    const float val1 = c1.validity - minValidity;
    if (val0 < 0.0f && val1 < 0.0f) {
        return false;
    }
    if (val0 < 0.0f || val1 < 0.0f) {
        const float denom = val1 - val0;
        const float tClip = (std::abs(denom) > 1.0e-8f)
            ? std::clamp(-val0 / denom, 0.0f, 1.0f)
            : 0.5f;
        Crossing cMid{};
        cMid.v = lerpSurfaceVertex(c0.v, c1.v, tClip);
        cMid.validity = minValidity;
        cMid.width = c0.width + tClip * (c1.width - c0.width);
        if (val0 < 0.0f) {
            c0 = cMid;
        } else {
            c1 = cMid;
        }
    }

    if (dist3(c0.v.pos, c1.v.pos) <= 1.0e-5f) {
        return false;
    }

    outSeg.a = c0.v;
    outSeg.b = c1.v;
    outSeg.widthA = c0.width;
    outSeg.widthB = c1.width;
    return true;
}

/// Extracts iso-contour segments `field(x) = isoValue` across triangles of `mesh`,
/// restricted to points where `validityField(x) >= minValidity`.
std::vector<Segment3D> extractIsoContourSegments(
    const SurfaceMesh& mesh,
    const std::vector<float>& fieldAtVert,
    float isoValue,
    const std::vector<float>& validityAtVert,
    float minValidity,
    const std::vector<float>& widthAtVert)
{
    std::vector<Segment3D> segments;
    if (mesh.tris.empty() || fieldAtVert.size() != mesh.verts.size()) {
        return segments;
    }

    Segment3D seg{};
    for (const auto& tri : mesh.tris) {
        if (intersectTriangleWithIsoLevel(
                mesh,
                tri,
                fieldAtVert,
                isoValue,
                validityAtVert,
                minValidity,
                widthAtVert,
                seg)) {
            segments.push_back(seg);
        }
    }

    return segments;
}

struct RawChainPoint {
    Vec3f pos{};
    Vec3f nozzleAxis{};
    float widthMm = 0.40f;
    float heightMm = 0.20f;
};

struct RawChain {
    std::vector<RawChainPoint> pts;
    bool isClosed = false;
};

/// Stitches unordered 3D segments into maximal continuous open polylines or closed loops
/// in $O(N)$ time using a 3D spatial hash table over segment endpoints.
std::vector<RawChain> stitchSegmentsIntoChains(
    const std::vector<Segment3D>& segments,
    float snapToleranceMm,
    float closeLoopToleranceMm)
{
    std::vector<RawChain> result;
    if (segments.empty()) {
        return result;
    }

    const std::size_t nSegs = segments.size();
    std::vector<bool> used(nSegs, false);

    auto makePtA = [](const Segment3D& s) -> RawChainPoint {
        return {s.a.pos, s.a.nozzleAxis, s.widthA, s.a.layerHeightMm};
    };
    auto makePtB = [](const Segment3D& s) -> RawChainPoint {
        return {s.b.pos, s.b.nozzleAxis, s.widthB, s.b.layerHeightMm};
    };

    const float cellSize = std::max(snapToleranceMm * 1.05f, 1.0e-3f);
    const float invCell = 1.0f / cellSize;
    const float snapSq = snapToleranceMm * snapToleranceMm;

    struct EndpointRef {
        std::uint32_t segIdx = 0;
        bool isEndB = false;
    };

    std::unordered_map<GridKey3D, std::vector<EndpointRef>, GridKey3DHash> endpointGrid;
    endpointGrid.reserve(nSegs * 2);

    for (std::uint32_t i = 0; i < static_cast<std::uint32_t>(nSegs); ++i) {
        endpointGrid[toGridKey(segments[i].a.pos, invCell)].push_back({i, false});
        endpointGrid[toGridKey(segments[i].b.pos, invCell)].push_back({i, true});
    }

    auto findClosestUnusedEndpoint = [&](const Vec3f& queryPos,
                                         bool preferStartA,
                                         std::size_t& outSegIdx,
                                         bool& outReverseSeg) -> bool {
        const GridKey3D baseKey = toGridKey(queryPos, invCell);
        float bestDistSq = snapSq;
        outSegIdx = nSegs;
        outReverseSeg = false;

        for (std::int64_t dz = -1; dz <= 1; ++dz) {
            for (std::int64_t dy = -1; dy <= 1; ++dy) {
                for (std::int64_t dx = -1; dx <= 1; ++dx) {
                    const GridKey3D nbKey{baseKey.ix + dx, baseKey.iy + dy, baseKey.iz + dz};
                    auto it = endpointGrid.find(nbKey);
                    if (it == endpointGrid.end()) {
                        continue;
                    }
                    for (const EndpointRef& ref : it->second) {
                        if (used[ref.segIdx]) {
                            continue;
                        }
                        const Vec3f& epPos =
                            ref.isEndB ? segments[ref.segIdx].b.pos : segments[ref.segIdx].a.pos;
                        const float dSq = lengthSq3(epPos - queryPos);
                        if (dSq < bestDistSq) {
                            bestDistSq = dSq;
                            outSegIdx = ref.segIdx;
                            // If `preferStartA` (extending forward from tail), matching `a` (`!isEndB`)
                            // means `reverseSeg = false`; matching `b` (`isEndB`) means `reverseSeg = true`.
                            // If `!preferStartA` (extending backward from head), matching `b` (`isEndB`)
                            // means `reverseSeg = false`; matching `a` (`!isEndB`) means `reverseSeg = true`.
                            outReverseSeg = preferStartA ? ref.isEndB : (!ref.isEndB);
                        }
                    }
                }
            }
        }
        return outSegIdx < nSegs;
    };

    for (std::size_t seed = 0; seed < nSegs; ++seed) {
        if (used[seed]) {
            continue;
        }
        used[seed] = true;

        std::vector<RawChainPoint> chain;
        chain.reserve(32);
        chain.push_back(makePtA(segments[seed]));
        chain.push_back(makePtB(segments[seed]));

        // Extend forward from chain.back()
        while (true) {
            std::size_t bestIdx = nSegs;
            bool reverseSeg = false;
            if (!findClosestUnusedEndpoint(chain.back().pos, true, bestIdx, reverseSeg)) {
                break;
            }
            used[bestIdx] = true;
            const auto& seg = segments[bestIdx];
            const RawChainPoint nextStart = reverseSeg ? makePtB(seg) : makePtA(seg);
            const RawChainPoint nextEnd = reverseSeg ? makePtA(seg) : makePtB(seg);

            chain.back().pos = lerp3(chain.back().pos, nextStart.pos, 0.5f);
            chain.back().widthMm = 0.5f * (chain.back().widthMm + nextStart.widthMm);
            chain.back().heightMm = 0.5f * (chain.back().heightMm + nextStart.heightMm);
            chain.back().nozzleAxis = normalize3(
                chain.back().nozzleAxis + nextStart.nozzleAxis,
                chain.back().nozzleAxis);

            if (dist3(chain.back().pos, nextEnd.pos) > 1.0e-5f) {
                chain.push_back(nextEnd);
            }
        }

        // Extend backward from chain.front()
        std::vector<RawChainPoint> revPrefix;
        while (true) {
            const Vec3f head = revPrefix.empty() ? chain.front().pos : revPrefix.back().pos;
            std::size_t bestIdx = nSegs;
            bool reverseSeg = false;
            if (!findClosestUnusedEndpoint(head, false, bestIdx, reverseSeg)) {
                break;
            }
            used[bestIdx] = true;
            const auto& seg = segments[bestIdx];
            const RawChainPoint prevStart = reverseSeg ? makePtB(seg) : makePtA(seg);
            const RawChainPoint prevEnd = reverseSeg ? makePtA(seg) : makePtB(seg);

            RawChainPoint& attachPt = revPrefix.empty() ? chain.front() : revPrefix.back();
            attachPt.pos = lerp3(attachPt.pos, prevEnd.pos, 0.5f);
            attachPt.widthMm = 0.5f * (attachPt.widthMm + prevEnd.widthMm);
            attachPt.heightMm = 0.5f * (attachPt.heightMm + prevEnd.heightMm);
            attachPt.nozzleAxis = normalize3(
                attachPt.nozzleAxis + prevEnd.nozzleAxis,
                attachPt.nozzleAxis);

            if (dist3(prevStart.pos, attachPt.pos) > 1.0e-5f) {
                revPrefix.push_back(prevStart);
            }
        }

        if (!revPrefix.empty()) {
            std::reverse(revPrefix.begin(), revPrefix.end());
            revPrefix.insert(revPrefix.end(), chain.begin(), chain.end());
            chain = std::move(revPrefix);
        }

        if (chain.size() < 2) {
            continue;
        }

        RawChain rc{};
        const float endDist = dist3(chain.front().pos, chain.back().pos);
        if (chain.size() >= 3 && endDist <= closeLoopToleranceMm) {
            rc.isClosed = true;
            if (endDist <= 0.05f * closeLoopToleranceMm && chain.size() >= 4) {
                chain.front().pos = lerp3(chain.front().pos, chain.back().pos, 0.5f);
                chain.front().widthMm = 0.5f * (chain.front().widthMm + chain.back().widthMm);
                chain.front().heightMm = 0.5f * (chain.front().heightMm + chain.back().heightMm);
                chain.pop_back();
            }
        } else {
            rc.isClosed = false;
        }
        rc.pts = std::move(chain);
        result.push_back(std::move(rc));
    }

    return result;
}

/// Converts a `RawChain` into an `ExtrusionPath` if its continuous 3D arc length is at least
/// `minExtrusionLengthMm`. Also simplifies redundant collinear micro-vertices while preserving
/// conformal 3D surface geometry and variable line widths.
bool finalizeExtrusionPath(
    const RawChain& raw,
    std::uint8_t featureKind,
    std::uint32_t wallIndex,
    float minExtrusionLengthMm,
    ExtrusionPath& outPath)
{
    if (raw.pts.size() < 2) {
        return false;
    }

    float totalLength = 0.0f;
    float weightedWidthSum = 0.0f;
    for (std::size_t i = 0; i + 1 < raw.pts.size(); ++i) {
        const float segLen = dist3(raw.pts[i].pos, raw.pts[i + 1].pos);
        totalLength += segLen;
        weightedWidthSum += segLen * 0.5f * (raw.pts[i].widthMm + raw.pts[i + 1].widthMm);
    }
    if (raw.isClosed && raw.pts.size() >= 3) {
        const float closeLen = dist3(raw.pts.back().pos, raw.pts.front().pos);
        totalLength += closeLen;
        weightedWidthSum += closeLen * 0.5f * (raw.pts.back().widthMm + raw.pts.front().widthMm);
    }

    if (totalLength < minExtrusionLengthMm || totalLength <= 1.0e-5f) {
        return false;
    }

    outPath.feature_kind = featureKind;
    outPath.wall_index = wallIndex;
    outPath.is_closed = raw.isClosed;
    outPath.total_length_mm = totalLength;
    outPath.mean_width_mm = weightedWidthSum / totalLength;
    outPath.points.clear();
    outPath.points.reserve(raw.pts.size());

    auto toToolpathPoint = [](const RawChainPoint& pt) -> ToolpathPoint {
        const Vec3f zNozzle = normalize3(pt.nozzleAxis, {0.0f, 0.0f, 1.0f});
        ToolpathPoint tp{};
        tp.x = pt.pos.x;
        tp.y = pt.pos.y;
        tp.z = pt.pos.z;
        tp.nx = zNozzle.x;
        tp.ny = zNozzle.y;
        tp.nz = zNozzle.z;
        tp.width_mm = pt.widthMm;
        tp.height_mm = pt.heightMm;
        return tp;
    };

    const float maxMergedSegLen = 0.55f;
    outPath.points.push_back(toToolpathPoint(raw.pts.front()));
    RawChainPoint lastKept = raw.pts.front();

    for (std::size_t i = 1; i + 1 < raw.pts.size(); ++i) {
        const RawChainPoint& curr = raw.pts[i];
        const RawChainPoint& next = raw.pts[i + 1];

        const float spanLen = dist3(lastKept.pos, next.pos);
        bool keep = (spanLen > maxMergedSegLen) ||
            (std::abs(curr.widthMm - lastKept.widthMm) > 0.015f) ||
            (std::abs(curr.heightMm - lastKept.heightMm) > 0.010f) ||
            (dot3(curr.nozzleAxis, lastKept.nozzleAxis) < 0.9994f);

        if (!keep && spanLen > 1.0e-5f) {
            const Vec3f chordDir = (next.pos - lastKept.pos) * (1.0f / spanLen);
            const Vec3f rel = curr.pos - lastKept.pos;
            const Vec3f perp = rel - (chordDir * dot3(rel, chordDir));
            if (lengthSq3(perp) > (0.010f * 0.010f)) {
                keep = true;
            }
        }

        if (keep) {
            outPath.points.push_back(toToolpathPoint(curr));
            lastKept = curr;
        }
    }

    outPath.points.push_back(toToolpathPoint(raw.pts.back()));
    return true;
}

/// Generates `Monotonic` flood-fill paths on a 3D submesh `infillMesh`
/// using single-pass triangle-to-scanline bucketing.
void generateMonotonicInfillOnSubmesh(
    const SurfaceMesh& infillMesh,
    std::uint32_t layerIndex,
    const SanitizedLineWidth& infillWidth,
    float minExtrusionLengthMm,
    ::rust::Vec<ExtrusionPath>& outPaths,
    float baseAngleDeg = 45.0f,
    std::uint8_t featureKind = kFeatureSolidInfill)
{
    if (infillMesh.tris.empty() || infillMesh.verts.empty()) {
        return;
    }

    // 1. Decompose `infillMesh` into connected triangle islands via shared edges
    const std::size_t nTris = infillMesh.tris.size();
    std::unordered_map<EdgeKey, std::vector<std::uint32_t>, EdgeKeyHash> edgeToTris;
    edgeToTris.reserve(nTris * 3);
    for (std::uint32_t t = 0; t < static_cast<std::uint32_t>(nTris); ++t) {
        const auto& tri = infillMesh.tris[t];
        edgeToTris[EdgeKey(tri[0], tri[1])].push_back(t);
        edgeToTris[EdgeKey(tri[1], tri[2])].push_back(t);
        edgeToTris[EdgeKey(tri[2], tri[0])].push_back(t);
    }

    std::vector<int> triIsland(nTris, -1);
    int numIslands = 0;
    for (std::size_t t = 0; t < nTris; ++t) {
        if (triIsland[t] >= 0) {
            continue;
        }
        const int islandId = numIslands++;
        std::queue<std::uint32_t> q;
        q.push(static_cast<std::uint32_t>(t));
        triIsland[t] = islandId;
        while (!q.empty()) {
            const std::uint32_t curr = q.front();
            q.pop();
            const auto& tri = infillMesh.tris[curr];
            const EdgeKey edges[3] = {
                EdgeKey(tri[0], tri[1]),
                EdgeKey(tri[1], tri[2]),
                EdgeKey(tri[2], tri[0]),
            };
            for (const auto& ek : edges) {
                auto it = edgeToTris.find(ek);
                if (it == edgeToTris.end()) {
                    continue;
                }
                for (std::uint32_t nb : it->second) {
                    if (triIsland[nb] < 0) {
                        triIsland[nb] = islandId;
                        q.push(nb);
                    }
                }
            }
        }
    }

    // Alternating cross-ply angle (+0° / +90° relative to `baseAngleDeg`) per layer
    constexpr float kPi = 3.14159265358979323846f;
    const float baseAngleRad = (baseAngleDeg * (kPi / 180.0f)) +
        ((layerIndex % 2 == 0) ? 0.0f : (0.5f * kPi));
    const float cosA = std::cos(baseAngleRad);
    const float sinA = std::sin(baseAngleRad);

    for (int isl = 0; isl < numIslands; ++isl) {
        SurfaceMesh islandMesh;
        std::unordered_map<std::uint32_t, std::uint32_t> vMap;
        Vec3f areaWeightedNozzle{0.0f, 0.0f, 0.0f};

        for (std::size_t t = 0; t < nTris; ++t) {
            if (triIsland[t] != isl) {
                continue;
            }
            const auto& tri = infillMesh.tris[t];
            std::array<std::uint32_t, 3> mapped{};
            for (int k = 0; k < 3; ++k) {
                auto it = vMap.find(tri[k]);
                if (it != vMap.end()) {
                    mapped[k] = it->second;
                } else {
                    const auto newIdx = static_cast<std::uint32_t>(islandMesh.verts.size());
                    islandMesh.verts.push_back(infillMesh.verts[tri[k]]);
                    vMap.emplace(tri[k], newIdx);
                    mapped[k] = newIdx;
                }
            }
            islandMesh.tris.push_back(mapped);

            const Vec3f& p0 = islandMesh.verts[mapped[0]].pos;
            const Vec3f& p1 = islandMesh.verts[mapped[1]].pos;
            const Vec3f& p2 = islandMesh.verts[mapped[2]].pos;
            const float area2 = length3(cross3(p1 - p0, p2 - p0));
            const Vec3f avgNozzle = normalize3(
                islandMesh.verts[mapped[0]].nozzleAxis +
                islandMesh.verts[mapped[1]].nozzleAxis +
                islandMesh.verts[mapped[2]].nozzleAxis);
            areaWeightedNozzle = areaWeightedNozzle + (avgNozzle * area2);
        }

        if (islandMesh.tris.empty() || islandMesh.verts.empty()) {
            continue;
        }

        // Build tangent orthonormal basis (uAxis, vAxis) orthogonal to the island's local nozzle axis
        const Vec3f zAxis = normalize3(areaWeightedNozzle, {0.0f, 0.0f, 1.0f});
        const Vec3f refAxis = (std::abs(zAxis.x) < 0.9f)
            ? Vec3f{1.0f, 0.0f, 0.0f}
            : Vec3f{0.0f, 1.0f, 0.0f};
        const Vec3f e1 = normalize3(refAxis - (zAxis * dot3(refAxis, zAxis)), {1.0f, 0.0f, 0.0f});
        const Vec3f e2 = normalize3(cross3(zAxis, e1), {0.0f, 1.0f, 0.0f});

        // Rotate in the tangent plane by `baseAngleRad`
        const Vec3f uAxis = normalize3((e1 * cosA) + (e2 * sinA), e1);
        const Vec3f vAxis = normalize3(cross3(zAxis, uAxis), e2);

        std::vector<float> uCoord(islandMesh.verts.size(), 0.0f);
        float uMin = std::numeric_limits<float>::max();
        float uMax = std::numeric_limits<float>::lowest();
        for (std::size_t i = 0; i < islandMesh.verts.size(); ++i) {
            const float u = dot3(islandMesh.verts[i].pos, uAxis);
            uCoord[i] = u;
            uMin = std::min(uMin, u);
            uMax = std::max(uMax, u);
        }

        const float uSpan = uMax - uMin;
        if (uSpan < 0.5f * infillWidth.min_mm) {
            continue;
        }

        // Determine scanline count and adapted line width within [min_mm, max_mm]
        const int nIdeal = std::max(1, static_cast<int>(std::round(uSpan / infillWidth.target_mm)));
        const float adaptedSpacing = std::clamp(
            uSpan / static_cast<float>(nIdeal),
            infillWidth.min_mm,
            infillWidth.max_mm);
        const int numLines = std::max(1, static_cast<int>(std::floor(uSpan / adaptedSpacing + 0.5f)));
        const float lineWidthMm = std::clamp(
            uSpan / static_cast<float>(numLines),
            infillWidth.min_mm,
            infillWidth.max_mm);
        const float startU = uMin + 0.5f * (uSpan - static_cast<float>(numLines - 1) * lineWidthMm);

        const std::vector<float> allValid(islandMesh.verts.size(), 1.0f);
        const std::vector<float> constWidth(islandMesh.verts.size(), lineWidthMm);

        // Single-pass triangle-to-scanline bucketing: each triangle only tests the scanlines
        // `m` that fall within its `[triUMin, triUMax]` interval!
        std::vector<std::vector<Segment3D>> rawScanSegs(static_cast<std::size_t>(numLines));
        Segment3D seg{};

        for (const auto& tri : islandMesh.tris) {
            const float u0 = uCoord[tri[0]];
            const float u1 = uCoord[tri[1]];
            const float u2 = uCoord[tri[2]];
            const float triUMin = std::min({u0, u1, u2});
            const float triUMax = std::max({u0, u1, u2});

            const int mLo = std::max(
                0,
                static_cast<int>(std::ceil((triUMin - startU) / lineWidthMm - 1.0e-5f)));
            const int mHi = std::min(
                numLines - 1,
                static_cast<int>(std::floor((triUMax - startU) / lineWidthMm + 1.0e-5f)));

            for (int m = mLo; m <= mHi; ++m) {
                const float uIso = startU + static_cast<float>(m) * lineWidthMm;
                if (intersectTriangleWithIsoLevel(
                        islandMesh,
                        tri,
                        uCoord,
                        uIso,
                        allValid,
                        0.0f,
                        constWidth,
                        seg)) {
                    rawScanSegs[static_cast<std::size_t>(m)].push_back(seg);
                }
            }
        }

        // Stitch each scanline's segments into ordered 3D scanline chains
        struct ScanSegment {
            RawChain chain;
            float vStart = 0.0f;
            float vEnd = 0.0f;
            bool consumed = false;
        };
        std::vector<std::vector<ScanSegment>> scanlines(static_cast<std::size_t>(numLines));

        for (int m = 0; m < numLines; ++m) {
            const auto& segs = rawScanSegs[static_cast<std::size_t>(m)];
            if (segs.empty()) {
                continue;
            }
            auto chains = stitchSegmentsIntoChains(segs, 0.06f * lineWidthMm, 0.0f);
            for (auto& ch : chains) {
                if (ch.pts.size() < 2) {
                    continue;
                }
                ch.isClosed = false;
                float v0 = dot3(ch.pts.front().pos, vAxis);
                float v1 = dot3(ch.pts.back().pos, vAxis);
                if (v0 > v1) {
                    std::reverse(ch.pts.begin(), ch.pts.end());
                    std::swap(v0, v1);
                }
                scanlines[static_cast<std::size_t>(m)].push_back(ScanSegment{
                    std::move(ch),
                    v0,
                    v1,
                    false,
                });
            }
            std::sort(
                scanlines[static_cast<std::size_t>(m)].begin(),
                scanlines[static_cast<std::size_t>(m)].end(),
                [](const ScanSegment& a, const ScanSegment& b) {
                    return a.vStart < b.vStart;
                });
        }

        // Monotonic flood-fill traversal: step strictly from m = 0 to numLines - 1,
        // linking adjacent scanlines into continuous zig-zag flood paths when their
        // boundary endpoints lie within `maxLinkDist` and preserve monotonic order.
        const float maxLinkDist = 2.25f * infillWidth.max_mm;
        std::uint32_t passCounter = 0;

        for (int m = 0; m < numLines; ++m) {
            for (std::size_t sIdx = 0; sIdx < scanlines[static_cast<std::size_t>(m)].size(); ++sIdx) {
                auto& seedSeg = scanlines[static_cast<std::size_t>(m)][sIdx];
                if (seedSeg.consumed) {
                    continue;
                }
                seedSeg.consumed = true;

                RawChain floodChain = seedSeg.chain;
                bool forwardAlongV = true;
                int currLine = m;

                while (currLine + 1 < numLines) {
                    const Vec3f currEnd = floodChain.pts.back().pos;
                    auto& nextLineSegs = scanlines[static_cast<std::size_t>(currLine + 1)];

                    std::size_t bestNext = nextLineSegs.size();
                    float bestDist = maxLinkDist;
                    bool nextForward = !forwardAlongV;

                    for (std::size_t k = 0; k < nextLineSegs.size(); ++k) {
                        if (nextLineSegs[k].consumed) {
                            continue;
                        }
                        const Vec3f candidateAttach = forwardAlongV
                            ? nextLineSegs[k].chain.pts.back().pos
                            : nextLineSegs[k].chain.pts.front().pos;
                        const float d = dist3(currEnd, candidateAttach);
                        if (d < bestDist) {
                            bestDist = d;
                            bestNext = k;
                            nextForward = !forwardAlongV;
                        }
                    }

                    if (bestNext == nextLineSegs.size()) {
                        break;
                    }

                    nextLineSegs[bestNext].consumed = true;
                    auto nextPts = nextLineSegs[bestNext].chain.pts;
                    if (!nextForward) {
                        std::reverse(nextPts.begin(), nextPts.end());
                    }
                    floodChain.pts.insert(floodChain.pts.end(), nextPts.begin(), nextPts.end());
                    forwardAlongV = nextForward;
                    ++currLine;
                }

                ExtrusionPath ep{};
                if (finalizeExtrusionPath(
                        floodChain,
                        featureKind,
                        passCounter,
                        minExtrusionLengthMm,
                        ep)) {
                    outPaths.push_back(std::move(ep));
                    ++passCounter;
                }
            }
        }
    }
}

/// Decomposes a `SurfaceMesh` into connected triangle islands via shared edges.
std::vector<SurfaceMesh> decomposeMeshIntoIslands(const SurfaceMesh& inMesh) {
    std::vector<SurfaceMesh> islands;
    if (inMesh.tris.empty() || inMesh.verts.empty()) {
        return islands;
    }

    const std::size_t nTris = inMesh.tris.size();
    std::unordered_map<EdgeKey, std::vector<std::uint32_t>, EdgeKeyHash> edgeToTris;
    edgeToTris.reserve(nTris * 3);
    for (std::uint32_t t = 0; t < static_cast<std::uint32_t>(nTris); ++t) {
        const auto& tri = inMesh.tris[t];
        edgeToTris[EdgeKey(tri[0], tri[1])].push_back(t);
        edgeToTris[EdgeKey(tri[1], tri[2])].push_back(t);
        edgeToTris[EdgeKey(tri[2], tri[0])].push_back(t);
    }

    std::vector<int> triIsland(nTris, -1);
    int numIslands = 0;
    for (std::size_t t = 0; t < nTris; ++t) {
        if (triIsland[t] >= 0) {
            continue;
        }
        const int islandId = numIslands++;
        std::queue<std::uint32_t> q;
        q.push(static_cast<std::uint32_t>(t));
        triIsland[t] = islandId;
        while (!q.empty()) {
            const std::uint32_t curr = q.front();
            q.pop();
            const auto& tri = inMesh.tris[curr];
            const EdgeKey edges[3] = {
                EdgeKey(tri[0], tri[1]),
                EdgeKey(tri[1], tri[2]),
                EdgeKey(tri[2], tri[0]),
            };
            for (const auto& ek : edges) {
                auto it = edgeToTris.find(ek);
                if (it == edgeToTris.end()) {
                    continue;
                }
                for (std::uint32_t nb : it->second) {
                    if (triIsland[nb] < 0) {
                        triIsland[nb] = islandId;
                        q.push(nb);
                    }
                }
            }
        }
    }

    islands.resize(static_cast<std::size_t>(numIslands));
    std::vector<std::unordered_map<std::uint32_t, std::uint32_t>> vMaps(
        static_cast<std::size_t>(numIslands));

    for (std::size_t t = 0; t < nTris; ++t) {
        const int isl = triIsland[t];
        if (isl < 0) {
            continue;
        }
        auto& islandMesh = islands[static_cast<std::size_t>(isl)];
        auto& vMap = vMaps[static_cast<std::size_t>(isl)];
        const auto& tri = inMesh.tris[t];
        std::array<std::uint32_t, 3> mapped{};
        for (int k = 0; k < 3; ++k) {
            auto it = vMap.find(tri[k]);
            if (it != vMap.end()) {
                mapped[k] = it->second;
            } else {
                const auto newIdx = static_cast<std::uint32_t>(islandMesh.verts.size());
                islandMesh.verts.push_back(inMesh.verts[tri[k]]);
                vMap.emplace(tri[k], newIdx);
                mapped[k] = newIdx;
            }
        }
        islandMesh.tris.push_back(mapped);
    }

    return islands;
}

/// Computes exact on-surface Euclidean/geodesic distance from all boundary edges of `mesh`
/// (edges incident to 1 triangle or flagged with `onInnerShellBoundary`).
std::vector<float> computeOnSurfaceBoundaryDistance(
    const SurfaceMesh& mesh,
    float searchCellSizeMm)
{
    const std::size_t nV = mesh.verts.size();
    const float kInf = 1.0e9f;
    std::vector<float> dBound(nV, kInf);
    if (nV == 0 || mesh.tris.empty()) {
        return dBound;
    }

    std::vector<std::vector<std::uint32_t>> adj(nV);
    std::unordered_map<EdgeKey, std::uint32_t, EdgeKeyHash> edgeTriCounts;
    edgeTriCounts.reserve(mesh.tris.size() * 3);
    for (const auto& tri : mesh.tris) {
        edgeTriCounts[EdgeKey(tri[0], tri[1])] += 1;
        edgeTriCounts[EdgeKey(tri[1], tri[2])] += 1;
        edgeTriCounts[EdgeKey(tri[2], tri[0])] += 1;
        for (int k = 0; k < 3; ++k) {
            const std::uint32_t u = tri[k];
            const std::uint32_t v = tri[(k + 1) % 3];
            adj[u].push_back(v);
            adj[v].push_back(u);
        }
    }

    struct BoundSeg {
        Vec3f a;
        Vec3f b;
        Vec3f ab;
        float abLenSq;
    };
    std::vector<BoundSeg> boundSegs;
    boundSegs.reserve(nV / 2 + 8);
    std::vector<bool> isBoundVert(nV, false);

    for (const auto& [ek, count] : edgeTriCounts) {
        if (count == 1 ||
            (mesh.verts[ek.a].onInnerShellBoundary && mesh.verts[ek.b].onInnerShellBoundary)) {
            isBoundVert[ek.a] = true;
            isBoundVert[ek.b] = true;
            const Vec3f a = mesh.verts[ek.a].pos;
            const Vec3f b = mesh.verts[ek.b].pos;
            const Vec3f ab = b - a;
            boundSegs.push_back(BoundSeg{a, b, ab, lengthSq3(ab)});
        }
    }

    if (boundSegs.empty()) {
        return dBound;
    }

    const float cellSize = std::max(0.5f, searchCellSizeMm);
    const float invCell = 1.0f / cellSize;
    std::unordered_map<GridKey3D, std::vector<std::uint32_t>, GridKey3DHash> segGrid;
    segGrid.reserve(boundSegs.size() * 2 + 8);
    for (std::size_t sIdx = 0; sIdx < boundSegs.size(); ++sIdx) {
        const auto& seg = boundSegs[sIdx];
        const Vec3f minPt{
            std::min(seg.a.x, seg.b.x),
            std::min(seg.a.y, seg.b.y),
            std::min(seg.a.z, seg.b.z)};
        const Vec3f maxPt{
            std::max(seg.a.x, seg.b.x),
            std::max(seg.a.y, seg.b.y),
            std::max(seg.a.z, seg.b.z)};
        const GridKey3D kMin = toGridKey(minPt, invCell);
        const GridKey3D kMax = toGridKey(maxPt, invCell);
        for (std::int64_t iz = kMin.iz; iz <= kMax.iz; ++iz) {
            for (std::int64_t iy = kMin.iy; iy <= kMax.iy; ++iy) {
                for (std::int64_t ix = kMin.ix; ix <= kMax.ix; ++ix) {
                    segGrid[GridKey3D{ix, iy, iz}].push_back(static_cast<std::uint32_t>(sIdx));
                }
            }
        }
    }

    using QueueItem = std::pair<float, std::uint32_t>;
    std::priority_queue<QueueItem, std::vector<QueueItem>, std::greater<QueueItem>> pq;

    for (std::size_t i = 0; i < nV; ++i) {
        if (isBoundVert[i]) {
            dBound[i] = 0.0f;
            pq.push({0.0f, static_cast<std::uint32_t>(i)});
            continue;
        }
        const Vec3f p = mesh.verts[i].pos;
        const GridKey3D baseKey = toGridKey(p, invCell);
        float bestDist = kInf;
        for (std::int64_t dz = -1; dz <= 1; ++dz) {
            for (std::int64_t dy = -1; dy <= 1; ++dy) {
                for (std::int64_t dx = -1; dx <= 1; ++dx) {
                    const GridKey3D nbKey{baseKey.ix + dx, baseKey.iy + dy, baseKey.iz + dz};
                    auto it = segGrid.find(nbKey);
                    if (it == segGrid.end()) {
                        continue;
                    }
                    for (std::uint32_t sIdx : it->second) {
                        const auto& seg = boundSegs[sIdx];
                        const float tProj = (seg.abLenSq > 1.0e-12f)
                            ? std::clamp(dot3(p - seg.a, seg.ab) / seg.abLenSq, 0.0f, 1.0f)
                            : 0.5f;
                        const Vec3f foot = seg.a + seg.ab * tProj;
                        bestDist = std::min(bestDist, dist3(p, foot));
                    }
                }
            }
        }
        if (bestDist < kInf * 0.5f) {
            dBound[i] = bestDist;
            pq.push({bestDist, static_cast<std::uint32_t>(i)});
        }
    }

    while (!pq.empty()) {
        const auto [dCurr, u] = pq.top();
        pq.pop();
        if (dCurr > dBound[u] + 1.0e-6f) {
            continue;
        }
        for (std::uint32_t v : adj[u]) {
            const float wEdge = dist3(mesh.verts[u].pos, mesh.verts[v].pos);
            if (dCurr + wEdge < dBound[v]) {
                dBound[v] = dCurr + wEdge;
                pq.push({dBound[v], v});
            }
        }
    }

    return dBound;
}

/// Sub-chord SDF validation: ensures no segment or turnaround chord of `chain` crosses into the
/// forbidden shell/exterior domain (`shiftedSdf(p) > maxAllowedSdf`). If a chord crosses outside,
/// splits the chain at the boundary crossing so all emitted sub-chains remain strictly valid.
template <typename SdfSamplerFn>
std::vector<RawChain> reclipAndRefineChainToSdf(
    const RawChain& chain,
    float maxAllowedSdf,
    SdfSamplerFn&& sampleShiftedSdf)
{
    std::vector<RawChain> out;
    if (chain.pts.size() < 2) {
        return out;
    }

    std::vector<RawChainPoint> pts = chain.pts;
    if (chain.isClosed && pts.size() >= 3) {
        pts.push_back(pts.front());
    }

    auto lerpChainPt = [](const RawChainPoint& a, const RawChainPoint& b, float t) -> RawChainPoint {
        RawChainPoint p{};
        p.pos = lerp3(a.pos, b.pos, t);
        p.nozzleAxis = normalize3(lerp3(a.nozzleAxis, b.nozzleAxis, t), a.nozzleAxis);
        p.widthMm = a.widthMm + t * (b.widthMm - a.widthMm);
        p.heightMm = a.heightMm + t * (b.heightMm - a.heightMm);
        return p;
    };

    RawChain active{};
    active.isClosed = false;

    auto flushActive = [&]() {
        if (active.pts.size() >= 2) {
            out.push_back(std::move(active));
        }
        active = RawChain{};
        active.isClosed = false;
    };

    const float maxStepMm = 0.45f;
    for (std::size_t i = 0; i + 1 < pts.size(); ++i) {
        const RawChainPoint& a = pts[i];
        const RawChainPoint& b = pts[i + 1];
        const float segLen = dist3(a.pos, b.pos);
        const int nSteps = std::max(1, static_cast<int>(std::ceil(segLen / maxStepMm)));

        RawChainPoint prevPt = a;
        float prevVal = sampleShiftedSdf(prevPt.pos) - maxAllowedSdf;
        bool prevIn = (prevVal <= 0.0f);

        if (prevIn && active.pts.empty()) {
            active.pts.push_back(prevPt);
        }

        for (int s = 1; s <= nSteps; ++s) {
            const float t = static_cast<float>(s) / static_cast<float>(nSteps);
            const RawChainPoint currPt = (s == nSteps) ? b : lerpChainPt(a, b, t);
            const float currVal = sampleShiftedSdf(currPt.pos) - maxAllowedSdf;
            const bool currIn = (currVal <= 0.0f);

            if (prevIn && currIn) {
                if (active.pts.empty()) {
                    active.pts.push_back(prevPt);
                }
                if (s == nSteps || dist3(active.pts.back().pos, currPt.pos) > 0.35f) {
                    active.pts.push_back(currPt);
                }
            } else if (prevIn && !currIn) {
                const float denom = currVal - prevVal;
                const float tCross = (std::abs(denom) > 1.0e-8f)
                    ? std::clamp(-prevVal / denom, 0.0f, 1.0f)
                    : 0.5f;
                const RawChainPoint exitPt = lerpChainPt(prevPt, currPt, tCross);
                if (active.pts.empty()) {
                    active.pts.push_back(prevPt);
                }
                if (dist3(active.pts.back().pos, exitPt.pos) > 1.0e-5f) {
                    active.pts.push_back(exitPt);
                }
                flushActive();
            } else if (!prevIn && currIn) {
                const float denom = currVal - prevVal;
                const float tCross = (std::abs(denom) > 1.0e-8f)
                    ? std::clamp(-prevVal / denom, 0.0f, 1.0f)
                    : 0.5f;
                const RawChainPoint enterPt = lerpChainPt(prevPt, currPt, tCross);
                active.pts.push_back(enterPt);
                if (dist3(active.pts.back().pos, currPt.pos) > 1.0e-5f) {
                    active.pts.push_back(currPt);
                }
            }

            prevPt = currPt;
            prevVal = currVal;
            prevIn = currIn;
        }
    }

    flushActive();
    if (chain.isClosed && out.size() == 1 && out[0].pts.size() >= 4 &&
        dist3(out[0].pts.front().pos, out[0].pts.back().pos) <= 1.0e-3f) {
        out[0].pts.pop_back();
        out[0].isClosed = true;
    }
    return out;
}

/// Orders polylines using greedy nearest-neighbor traversal to minimize non-extruding travel moves.
void orderChainsGreedyNearestNeighbor(std::vector<RawChain>& chains) {
    if (chains.size() <= 1) {
        return;
    }
    std::vector<RawChain> ordered;
    ordered.reserve(chains.size());
    std::vector<bool> visited(chains.size(), false);

    visited[0] = true;
    ordered.push_back(std::move(chains[0]));

    for (std::size_t step = 1; step < chains.size(); ++step) {
        const Vec3f currPos = ordered.back().pts.back().pos;
        std::size_t bestIdx = chains.size();
        float bestDistSq = std::numeric_limits<float>::max();
        bool reverseBest = false;

        for (std::size_t i = 0; i < chains.size(); ++i) {
            if (visited[i] || chains[i].pts.empty()) {
                continue;
            }
            const float dFrontSq = lengthSq3(chains[i].pts.front().pos - currPos);
            if (dFrontSq < bestDistSq) {
                bestDistSq = dFrontSq;
                bestIdx = i;
                reverseBest = false;
            }
            if (!chains[i].isClosed) {
                const float dBackSq = lengthSq3(chains[i].pts.back().pos - currPos);
                if (dBackSq < bestDistSq) {
                    bestDistSq = dBackSq;
                    bestIdx = i;
                    reverseBest = true;
                }
            }
        }

        if (bestIdx == chains.size()) {
            break;
        }
        visited[bestIdx] = true;
        if (reverseBest) {
            std::reverse(chains[bestIdx].pts.begin(), chains[bestIdx].pts.end());
        }
        ordered.push_back(std::move(chains[bestIdx]));
    }

    chains = std::move(ordered);
}

/// Extracts a parallel scanline family on `islandMesh` along `(uAxis, vAxis)` with pitch `pitchMm`,
/// groups adjacent scanline chords into monotonic components via Union-Find, and chains each
/// component in alternating boustrophedon order while validating turnaround segments against SDF.
template <typename SdfSamplerFn>
std::vector<RawChain> generateScanlineFamilyOnIsland(
    const SurfaceMesh& islandMesh,
    const Vec3f& uAxis,
    const Vec3f& vAxis,
    float pitchMm,
    float lineWidthMm,
    bool usePhaseOffset,
    float phaseOffsetMm,
    float maxLinkDistMm,
    float maxAllowedSdf,
    SdfSamplerFn&& sampleShiftedSdf)
{
    std::vector<RawChain> resultChains;
    if (islandMesh.tris.empty() || islandMesh.verts.empty() || pitchMm <= 1.0e-4f) {
        return resultChains;
    }

    std::vector<float> uCoord(islandMesh.verts.size(), 0.0f);
    float uMin = std::numeric_limits<float>::max();
    float uMax = std::numeric_limits<float>::lowest();
    for (std::size_t i = 0; i < islandMesh.verts.size(); ++i) {
        const float u = dot3(islandMesh.verts[i].pos, uAxis);
        uCoord[i] = u;
        uMin = std::min(uMin, u);
        uMax = std::max(uMax, u);
    }

    const float uSpan = uMax - uMin;
    if (uSpan <= 1.0e-4f) {
        return resultChains;
    }

    float startU = 0.0f;
    int numLines = 0;
    if (usePhaseOffset) {
        float phase = std::fmod(phaseOffsetMm, pitchMm);
        if (phase < 0.0f) {
            phase += pitchMm;
        }
        const float kFirst = std::ceil((uMin - phase) / pitchMm - 1.0e-5f);
        startU = phase + kFirst * pitchMm;
        if (startU > uMax) {
            return resultChains;
        }
        numLines = std::max(1, static_cast<int>(std::floor((uMax - startU) / pitchMm + 1.0e-5f)) + 1);
    } else {
        numLines = std::max(1, static_cast<int>(std::floor(uSpan / pitchMm)) + 1);
        while (numLines > 1 && static_cast<float>(numLines - 1) * pitchMm > uSpan - 1.0e-4f) {
            --numLines;
        }
        startU = uMin + 0.5f * (uSpan - static_cast<float>(numLines - 1) * pitchMm);
    }

    if (numLines <= 0) {
        return resultChains;
    }

    const std::vector<float> allValid(islandMesh.verts.size(), 1.0f);
    const std::vector<float> constWidth(islandMesh.verts.size(), lineWidthMm);

    std::vector<std::vector<Segment3D>> rawScanSegs(static_cast<std::size_t>(numLines));
    Segment3D seg{};

    for (const auto& tri : islandMesh.tris) {
        const float u0 = uCoord[tri[0]];
        const float u1 = uCoord[tri[1]];
        const float u2 = uCoord[tri[2]];
        const float triUMin = std::min({u0, u1, u2});
        const float triUMax = std::max({u0, u1, u2});

        const int mLo = std::max(
            0,
            static_cast<int>(std::ceil((triUMin - startU) / pitchMm - 1.0e-5f)));
        const int mHi = std::min(
            numLines - 1,
            static_cast<int>(std::floor((triUMax - startU) / pitchMm + 1.0e-5f)));

        for (int m = mLo; m <= mHi; ++m) {
            const float uIso = startU + static_cast<float>(m) * pitchMm;
            if (intersectTriangleWithIsoLevel(
                    islandMesh,
                    tri,
                    uCoord,
                    uIso,
                    allValid,
                    0.0f,
                    constWidth,
                    seg)) {
                rawScanSegs[static_cast<std::size_t>(m)].push_back(seg);
            }
        }
    }

    struct ChordNode {
        RawChain chain;
        int scanIdx = 0;
        float vStart = 0.0f;
        float vEnd = 0.0f;
        int nextChord = -1;
        int prevChord = -1;
    };

    std::vector<ChordNode> chords;
    std::vector<std::vector<int>> lineChords(static_cast<std::size_t>(numLines));

    for (int m = 0; m < numLines; ++m) {
        const auto& segs = rawScanSegs[static_cast<std::size_t>(m)];
        if (segs.empty()) {
            continue;
        }
        auto chains = stitchSegmentsIntoChains(segs, std::max(0.04f, 0.08f * lineWidthMm), 0.0f);
        std::vector<ChordNode> localNodes;
        for (auto& ch : chains) {
            if (ch.pts.size() < 2) {
                continue;
            }
            ch.isClosed = false;
            float v0 = dot3(ch.pts.front().pos, vAxis);
            float v1 = dot3(ch.pts.back().pos, vAxis);
            if (v0 > v1) {
                std::reverse(ch.pts.begin(), ch.pts.end());
                std::swap(v0, v1);
            }
            localNodes.push_back(ChordNode{std::move(ch), m, v0, v1, -1, -1});
        }
        std::sort(
            localNodes.begin(),
            localNodes.end(),
            [](const ChordNode& a, const ChordNode& b) { return a.vStart < b.vStart; });
        for (auto& node : localNodes) {
            const int id = static_cast<int>(chords.size());
            chords.push_back(std::move(node));
            lineChords[static_cast<std::size_t>(m)].push_back(id);
        }
    }

    if (chords.empty()) {
        return resultChains;
    }

    auto isTurnaroundValid = [&](const Vec3f& a, const Vec3f& b) -> bool {
        const float d = dist3(a, b);
        if (d > maxLinkDistMm) {
            return false;
        }
        const int steps = std::max(2, static_cast<int>(std::ceil(d / 0.35f)));
        for (int s = 1; s < steps; ++s) {
            const float t = static_cast<float>(s) / static_cast<float>(steps);
            if (sampleShiftedSdf(lerp3(a, b, t)) > maxAllowedSdf) {
                return false;
            }
        }
        return true;
    };

    // Union-Find / 1-to-1 monotonic matching across adjacent scanlines m -> m + 1
    std::vector<int> ufParent(chords.size());
    for (std::size_t i = 0; i < chords.size(); ++i) {
        ufParent[i] = static_cast<int>(i);
    }
    auto ufFind = [&](int x) {
        while (ufParent[static_cast<std::size_t>(x)] != x) {
            ufParent[static_cast<std::size_t>(x)] =
                ufParent[static_cast<std::size_t>(ufParent[static_cast<std::size_t>(x)])];
            x = ufParent[static_cast<std::size_t>(x)];
        }
        return x;
    };
    auto ufUnion = [&](int a, int b) {
        const int ra = ufFind(a);
        const int rb = ufFind(b);
        if (ra != rb) {
            ufParent[static_cast<std::size_t>(rb)] = ra;
        }
    };

    for (int m = 0; m + 1 < numLines; ++m) {
        const auto& currIds = lineChords[static_cast<std::size_t>(m)];
        const auto& nextIds = lineChords[static_cast<std::size_t>(m + 1)];
        for (int cId : currIds) {
            auto& cNode = chords[static_cast<std::size_t>(cId)];
            int bestNext = -1;
            float bestScore = maxLinkDistMm;
            for (int nId : nextIds) {
                auto& nNode = chords[static_cast<std::size_t>(nId)];
                if (nNode.prevChord >= 0) {
                    continue;
                }
                const float overlapLo = std::max(cNode.vStart, nNode.vStart);
                const float overlapHi = std::min(cNode.vEnd, nNode.vEnd);
                const float vGap = std::max(0.0f, overlapLo - overlapHi);
                if (vGap > maxLinkDistMm) {
                    continue;
                }
                const float dFront = dist3(cNode.chain.pts.front().pos, nNode.chain.pts.front().pos);
                const float dBack = dist3(cNode.chain.pts.back().pos, nNode.chain.pts.back().pos);
                const float minEndDist = std::min(dFront, dBack);
                if (minEndDist < bestScore) {
                    bestScore = minEndDist;
                    bestNext = nId;
                }
            }
            if (bestNext >= 0) {
                cNode.nextChord = bestNext;
                chords[static_cast<std::size_t>(bestNext)].prevChord = cId;
                ufUnion(cId, bestNext);
            }
        }
    }

    // Traverse each monotonic component from its lowest scanline `prevChord == -1`
    // in alternating boustrophedon order.
    for (std::size_t i = 0; i < chords.size(); ++i) {
        if (chords[i].prevChord >= 0) {
            continue;
        }
        RawChain compChain = chords[i].chain;
        bool forwardAlongV = true;
        int currId = static_cast<int>(i);

        while (chords[static_cast<std::size_t>(currId)].nextChord >= 0) {
            const int nextId = chords[static_cast<std::size_t>(currId)].nextChord;
            const auto& nextNode = chords[static_cast<std::size_t>(nextId)];
            const Vec3f currEnd = compChain.pts.back().pos;
            const Vec3f nextAttach = forwardAlongV
                ? nextNode.chain.pts.back().pos
                : nextNode.chain.pts.front().pos;

            if (isTurnaroundValid(currEnd, nextAttach)) {
                auto nextPts = nextNode.chain.pts;
                if (forwardAlongV) {
                    std::reverse(nextPts.begin(), nextPts.end());
                }
                compChain.pts.insert(compChain.pts.end(), nextPts.begin(), nextPts.end());
                forwardAlongV = !forwardAlongV;
            } else {
                resultChains.push_back(std::move(compChain));
                compChain = nextNode.chain;
                forwardAlongV = true;
            }
            currId = nextId;
        }

        if (compChain.pts.size() >= 2) {
            resultChains.push_back(std::move(compChain));
        }
    }

    return resultChains;
}

/// Generates non-planar sparse infill toolpaths (`feature_kind == kFeatureSparseInfill`, or
/// `kFeatureSolidInfill` for narrow-region-promoted islands) on the interior core of `baseMesh`.
template <typename SdfSamplerFn, typename NozzleSamplerFn, typename HeightSamplerFn>
void generateSparseInfillForLayer(
    const SurfaceMesh& baseMesh,
    const std::vector<float>& sdfDepth,
    float effectiveShellDepth,
    float sdfZeroShift,
    std::uint32_t layerIndex,
    float layerIsovalue,
    float sparseDensity,
    const SanitizedLineWidth& sparseW,
    std::uint8_t sparsePattern,
    float infillAngleDeg,
    float nozzleDiameterMm,
    const SanitizedLineWidth& solidInfillW,
    std::uint8_t solidInfillPattern,
    float minExtrusionLengthMm,
    float stitchSnap,
    float closeLoopTol,
    SdfSamplerFn&& sampleSdf,
    NozzleSamplerFn&& sampleNozzleAxis,
    HeightSamplerFn&& sampleLocalLayerHeight,
    ::rust::Vec<ExtrusionPath>& outPaths)
{
    if (sparseDensity <= 1.0e-4f || baseMesh.tris.empty() || baseMesh.verts.empty()) {
        return;
    }

    // 1. Clip `baseMesh` to the interior core where `sdfDepth >= effectiveShellDepth`
    //    (`-sdfDepth <= -effectiveShellDepth`), marking the cut contour with `onInnerShellBoundary`.
    std::vector<float> negSdfDepth(sdfDepth.size(), 0.0f);
    for (std::size_t i = 0; i < sdfDepth.size(); ++i) {
        negSdfDepth[i] = -sdfDepth[i];
    }

    SurfaceMesh rawSparseMesh = clipSurfaceMeshByScalar(
        baseMesh,
        negSdfDepth,
        -effectiveShellDepth,
        true);

    if (rawSparseMesh.tris.empty() || rawSparseMesh.verts.empty()) {
        return;
    }

    auto sampleShiftedSdf = [&](const Vec3f& p) -> float {
        return sampleSdf(p) - sdfZeroShift;
    };

    // Split any interior cross-strip edges whose endpoints both lie on `onInnerShellBoundary`
    // so narrow interior peninsulas have interior vertices before computing `dFromShell`.
    {
        for (auto& v : rawSparseMesh.verts) {
            v.onOuterBoundary = v.onInnerShellBoundary;
        }
        rawSparseMesh = subdivideShellMesh(
            rawSparseMesh,
            std::max(0.45f, 1.25f * sparseW.target_mm),
            sampleShiftedSdf);
        for (auto& v : rawSparseMesh.verts) {
            v.onInnerShellBoundary = v.onOuterBoundary;
            v.onOuterBoundary = false;
        }
    }

    // 2. Compute exact on-surface distance `dFromShell` from the physical inner edge of the shell
    //    and inset the sparse nozzle-centerline domain by `(0.5 - eta) * w_sparse = 0.35 * w_sparse`.
    const float endpointInsetMm = kInfillEndpointInsetFactor * sparseW.target_mm;
    const std::vector<float> dFromShell = computeOnSurfaceBoundaryDistance(
        rawSparseMesh,
        std::max(0.8f, 2.5f * sparseW.target_mm));

    std::vector<float> negDFromShell(dFromShell.size(), 0.0f);
    for (std::size_t i = 0; i < dFromShell.size(); ++i) {
        negDFromShell[i] = -dFromShell[i];
    }

    SurfaceMesh sparseMesh = clipSurfaceMeshByScalar(
        rawSparseMesh,
        negDFromShell,
        -endpointInsetMm,
        true);

    if (sparseMesh.tris.empty() || sparseMesh.verts.empty()) {
        return;
    }

    // Sample 5/6-axis nozzle orientation and local layer height on the sparse interior vertices
    for (auto& sv : sparseMesh.verts) {
        sv.nozzleAxis = sampleNozzleAxis(sv.pos);
        sv.sdf = sampleShiftedSdf(sv.pos);
        sv.layerHeightMm = sampleLocalLayerHeight(sv.pos);
    }

    // 3. Decompose `sparseMesh` into connected 3D triangle islands
    std::vector<SurfaceMesh> islands = decomposeMeshIntoIslands(sparseMesh);
    const float narrowDiagThresholdMm = 15.0f * std::max(0.1f, nozzleDiameterMm);
    const float maxAllowedSdf = -(effectiveShellDepth - 0.5f * sparseW.target_mm);
    constexpr float kPi = 3.14159265358979323846f;

    std::uint32_t sparsePassCounter = 0;

    for (auto& islandMesh : islands) {
        if (islandMesh.tris.empty() || islandMesh.verts.empty()) {
            continue;
        }

        // Compute 3D bounding-box diagonal and area-weighted nozzle axis for this island
        Vec3f pMin = islandMesh.verts.front().pos;
        Vec3f pMax = islandMesh.verts.front().pos;
        for (const auto& v : islandMesh.verts) {
            pMin.x = std::min(pMin.x, v.pos.x);
            pMin.y = std::min(pMin.y, v.pos.y);
            pMin.z = std::min(pMin.z, v.pos.z);
            pMax.x = std::max(pMax.x, v.pos.x);
            pMax.y = std::max(pMax.y, v.pos.y);
            pMax.z = std::max(pMax.z, v.pos.z);
        }
        const float islandDiagMm = dist3(pMin, pMax);

        // Narrow-region promotion: islands smaller than `15 * D_nozzle` are promoted to solid infill
        if (islandDiagMm < narrowDiagThresholdMm) {
            if (solidInfillPattern == kPatternConcentric) {
                const std::vector<float> dInward = computeOnSurfaceBoundaryDistance(
                    islandMesh,
                    std::max(0.6f, 2.0f * solidInfillW.target_mm));
                float maxDepth = 0.0f;
                for (float d : dInward) {
                    if (d < 1.0e8f) {
                        maxDepth = std::max(maxDepth, d);
                    }
                }
                const int maxPasses = std::clamp(
                    static_cast<int>(std::ceil(maxDepth / solidInfillW.target_mm)) + 1,
                    1,
                    64);
                const std::vector<float> allValid(islandMesh.verts.size(), 1.0f);
                const std::vector<float> constWidth(islandMesh.verts.size(), solidInfillW.target_mm);
                std::uint32_t passIdx = 0;
                for (int p = 0; p < maxPasses; ++p) {
                    const float targetD = static_cast<float>(p) * solidInfillW.target_mm;
                    if (targetD > maxDepth + 1.0e-4f) {
                        break;
                    }
                    const auto segs = extractIsoContourSegments(
                        islandMesh,
                        dInward,
                        targetD,
                        allValid,
                        0.0f,
                        constWidth);
                    const auto chains = stitchSegmentsIntoChains(segs, stitchSnap, closeLoopTol);
                    for (const auto& ch : chains) {
                        ExtrusionPath ep{};
                        if (finalizeExtrusionPath(
                                ch,
                                kFeatureSolidInfill,
                                passIdx,
                                minExtrusionLengthMm,
                                ep)) {
                            outPaths.push_back(std::move(ep));
                            ++passIdx;
                        }
                    }
                }
            } else {
                generateMonotonicInfillOnSubmesh(
                    islandMesh,
                    layerIndex,
                    solidInfillW,
                    minExtrusionLengthMm,
                    outPaths,
                    infillAngleDeg,
                    kFeatureSolidInfill);
            }
            continue;
        }

        // Compute area-weighted local nozzle axis (obeying AGENTS.md Rule 1)
        Vec3f areaWeightedNozzle{0.0f, 0.0f, 0.0f};
        for (const auto& tri : islandMesh.tris) {
            const Vec3f& p0 = islandMesh.verts[tri[0]].pos;
            const Vec3f& p1 = islandMesh.verts[tri[1]].pos;
            const Vec3f& p2 = islandMesh.verts[tri[2]].pos;
            const float area2 = length3(cross3(p1 - p0, p2 - p0));
            const Vec3f avgNozzle = normalize3(
                islandMesh.verts[tri[0]].nozzleAxis +
                islandMesh.verts[tri[1]].nozzleAxis +
                islandMesh.verts[tri[2]].nozzleAxis);
            areaWeightedNozzle = areaWeightedNozzle + (avgNozzle * area2);
        }
        const Vec3f zAxis = normalize3(areaWeightedNozzle, {0.0f, 0.0f, 1.0f});
        const Vec3f refAxis = (std::abs(zAxis.x) < 0.9f)
            ? Vec3f{1.0f, 0.0f, 0.0f}
            : Vec3f{0.0f, 1.0f, 0.0f};
        const Vec3f e1 = normalize3(refAxis - (zAxis * dot3(refAxis, zAxis)), {1.0f, 0.0f, 0.0f});
        const Vec3f e2 = normalize3(cross3(zAxis, e1), {0.0f, 1.0f, 0.0f});

        std::vector<RawChain> rawSparseChains;

        if (sparsePattern == kSparsePatternMonotonic) {
            const float pitchMm = std::max(
                sparseW.target_mm,
                sparseW.target_mm / std::max(0.01f, sparseDensity));
            const float angleRad = (infillAngleDeg * (kPi / 180.0f)) +
                ((layerIndex % 2 == 0) ? 0.0f : (0.5f * kPi));
            const Vec3f uAxis = normalize3(
                (e1 * std::cos(angleRad)) + (e2 * std::sin(angleRad)),
                e1);
            const Vec3f vAxis = normalize3(cross3(zAxis, uAxis), e2);

            rawSparseChains = generateScanlineFamilyOnIsland(
                islandMesh,
                uAxis,
                vAxis,
                pitchMm,
                sparseW.target_mm,
                false,
                0.0f,
                2.0f * pitchMm,
                maxAllowedSdf,
                sampleShiftedSdf);
        } else if (sparsePattern == kSparsePatternCubic) {
            const float dCubic = std::max(
                sparseW.target_mm,
                (2.0f * sparseW.target_mm) / std::max(0.01f, sparseDensity));
            const float baseRad = infillAngleDeg * (kPi / 180.0f);
            const float phaseShift = 1.41421356237f * layerIsovalue;

            // Family A: +45 deg, phase +sqrt(2) * c_k
            {
                const float angA = baseRad + 0.25f * kPi;
                const Vec3f uAxisA = normalize3(
                    (e1 * std::cos(angA)) + (e2 * std::sin(angA)),
                    e1);
                const Vec3f vAxisA = normalize3(cross3(zAxis, uAxisA), e2);
                auto chainsA = generateScanlineFamilyOnIsland(
                    islandMesh,
                    uAxisA,
                    vAxisA,
                    dCubic,
                    sparseW.target_mm,
                    true,
                    phaseShift,
                    1.75f * dCubic,
                    maxAllowedSdf,
                    sampleShiftedSdf);
                rawSparseChains.insert(
                    rawSparseChains.end(),
                    std::make_move_iterator(chainsA.begin()),
                    std::make_move_iterator(chainsA.end()));
            }
            // Family B: -45 deg, phase -sqrt(2) * c_k
            {
                const float angB = baseRad - 0.25f * kPi;
                const Vec3f uAxisB = normalize3(
                    (e1 * std::cos(angB)) + (e2 * std::sin(angB)),
                    e1);
                const Vec3f vAxisB = normalize3(cross3(zAxis, uAxisB), e2);
                auto chainsB = generateScanlineFamilyOnIsland(
                    islandMesh,
                    uAxisB,
                    vAxisB,
                    dCubic,
                    sparseW.target_mm,
                    true,
                    -phaseShift,
                    1.75f * dCubic,
                    maxAllowedSdf,
                    sampleShiftedSdf);
                rawSparseChains.insert(
                    rawSparseChains.end(),
                    std::make_move_iterator(chainsB.begin()),
                    std::make_move_iterator(chainsB.end()));
            }
        } else if (sparsePattern == kSparsePatternGyroid ||
                   sparsePattern == kSparsePatternSchwarzD ||
                   sparsePattern == kSparsePatternSchwarzP) {
            const float wavelengthMm = std::clamp(
                (2.0f * kPi * sparseW.target_mm) / std::max(0.01f, sparseDensity),
                2.0f * sparseW.target_mm,
                50.0f);
            const float waveK = (2.0f * kPi) / wavelengthMm;
            const float maxTpmsEdge = std::clamp(0.22f * wavelengthMm, 0.35f, 1.50f);

            // Refine island triangles if needed so the trigonometric TPMS field is well-resolved
            SurfaceMesh tpmsMesh = subdivideShellMesh(
                islandMesh,
                maxTpmsEdge,
                sampleShiftedSdf);
            for (auto& sv : tpmsMesh.verts) {
                if (lengthSq3(sv.nozzleAxis) <= 1.0e-4f) {
                    sv.nozzleAxis = sampleNozzleAxis(sv.pos);
                }
                sv.layerHeightMm = sampleLocalLayerHeight(sv.pos);
            }

            std::vector<float> psiField(tpmsMesh.verts.size(), 0.0f);
            for (std::size_t i = 0; i < tpmsMesh.verts.size(); ++i) {
                const Vec3f& p = tpmsMesh.verts[i].pos;
                const float kx = waveK * p.x;
                const float ky = waveK * p.y;
                const float kz = waveK * p.z;
                if (sparsePattern == kSparsePatternGyroid) {
                    psiField[i] = std::sin(kx) * std::cos(ky) +
                        std::sin(ky) * std::cos(kz) +
                        std::sin(kz) * std::cos(kx);
                } else if (sparsePattern == kSparsePatternSchwarzD) {
                    psiField[i] = std::sin(kx) * std::sin(ky) * std::sin(kz) +
                        std::sin(kx) * std::cos(ky) * std::cos(kz) +
                        std::cos(kx) * std::sin(ky) * std::cos(kz) +
                        std::cos(kx) * std::cos(ky) * std::sin(kz);
                } else {
                    psiField[i] = std::cos(kx) + std::cos(ky) + std::cos(kz);
                }
            }

            const std::vector<float> allValid(tpmsMesh.verts.size(), 1.0f);
            const std::vector<float> constWidth(tpmsMesh.verts.size(), sparseW.target_mm);
            const auto tpmsSegs = extractIsoContourSegments(
                tpmsMesh,
                psiField,
                0.0f,
                allValid,
                0.0f,
                constWidth);
            rawSparseChains = stitchSegmentsIntoChains(
                tpmsSegs,
                std::max(0.05f, 0.15f * maxTpmsEdge),
                closeLoopTol);
            orderChainsGreedyNearestNeighbor(rawSparseChains);
        } else {
            // `Concentric` (5) or `AllWalls` (6): inward geodesic contour rings
            const float ringPitchMm = (sparsePattern == kSparsePatternAllWalls &&
                                       sparsePattern != kSparsePatternConcentric)
                ? sparseW.target_mm
                : std::max(sparseW.target_mm, sparseW.target_mm / std::max(0.01f, sparseDensity));
            const std::vector<float> dInward = computeOnSurfaceBoundaryDistance(
                islandMesh,
                std::max(0.8f, 2.0f * ringPitchMm));
            float maxInward = 0.0f;
            for (float d : dInward) {
                if (d < 1.0e8f) {
                    maxInward = std::max(maxInward, d);
                }
            }

            // Since `islandMesh` is already inset by `0.35 * w_sparse` from `effectiveShellDepth`,
            // starting the first concentric ring at `0.15 * w_sparse` inside `islandMesh` places
            // its centerline at `(0.35 + 0.15) * w_sparse = 0.50 * w_sparse` from the shell edge!
            const float firstRingOffset = (0.5f - kInfillEndpointInsetFactor) * sparseW.target_mm;
            const std::vector<float> allValid(islandMesh.verts.size(), 1.0f);
            const std::vector<float> constWidth(islandMesh.verts.size(), sparseW.target_mm);

            const int maxRings = std::clamp(
                static_cast<int>(std::floor((maxInward - firstRingOffset) / ringPitchMm)) + 1,
                0,
                256);
            for (int r = 0; r < maxRings; ++r) {
                const float targetD = firstRingOffset + static_cast<float>(r) * ringPitchMm;
                if (targetD > maxInward + 1.0e-4f) {
                    break;
                }
                const auto ringSegs = extractIsoContourSegments(
                    islandMesh,
                    dInward,
                    targetD,
                    allValid,
                    0.0f,
                    constWidth);
                auto ringChains = stitchSegmentsIntoChains(ringSegs, stitchSnap, closeLoopTol);
                rawSparseChains.insert(
                    rawSparseChains.end(),
                    std::make_move_iterator(ringChains.begin()),
                    std::make_move_iterator(ringChains.end()));
            }
            orderChainsGreedyNearestNeighbor(rawSparseChains);
        }

        // Sub-chord SDF gating and finalization
        std::size_t emittedForIsland = 0;
        for (const auto& rawCh : rawSparseChains) {
            auto gatedChains = reclipAndRefineChainToSdf(rawCh, maxAllowedSdf, sampleShiftedSdf);
            for (const auto& ch : gatedChains) {
                ExtrusionPath ep{};
                if (finalizeExtrusionPath(
                        ch,
                        kFeatureSparseInfill,
                        sparsePassCounter,
                        minExtrusionLengthMm,
                        ep)) {
                    outPaths.push_back(std::move(ep));
                    ++sparsePassCounter;
                    ++emittedForIsland;
                }
            }
        }

        // Sub-wavelength cavity fallback: if a non-empty island produced zero sparse paths
        // (e.g., cavity smaller than one pattern pitch/wavelength), fill it with Monotonic paths
        // so the interior cavity is not left unsupported.
        if (emittedForIsland == 0) {
            generateMonotonicInfillOnSubmesh(
                islandMesh,
                layerIndex,
                sparseW,
                minExtrusionLengthMm,
                outPaths,
                infillAngleDeg,
                kFeatureSparseInfill);
        }
    }
}

} // namespace

ObjectToolpathsResult generateShellToolpathsForLayers(
    const SelectedIsosurfacesResult& selectedLayers,
    const openvdb::FloatGrid& sdfGrid,
    const openvdb::FloatGrid& eikGrid,
    const openvdb::Vec3SGrid* toolAxisGrid,
    const std::array<float, 3>& defaultNozzleAxisBed,
    const LayerHeightSpec& layerHeightSpec,
    const ShellPathSpec& spec,
    const std::function<void(std::size_t, std::size_t)>& progressCb,
    const openvdb::FloatGrid* coarseSdfGrid,
    const openvdb::FloatGrid* coarseEikGrid,
    const openvdb::Vec3SGrid* coarseToolAxisGrid,
    float fineShellDepthMm)
{
    ObjectToolpathsResult result{};
    result.valid = false;
    result.total_path_count = 0;
    result.outer_wall_count = 0;
    result.inner_wall_count = 0;
    result.solid_infill_count = 0;
    result.sparse_infill_count = 0;

    if (!selectedLayers.valid || selectedLayers.layers.empty()) {
        return result;
    }

    const float shellThicknessMm = std::max(0.05f, spec.shell_thickness_mm);
    const std::uint32_t numInnerWalls = spec.inner_wall_count;
    const SanitizedLineWidth outerW = sanitizeLineWidth(spec.outer_wall_width);
    const SanitizedLineWidth innerW = sanitizeLineWidth(spec.inner_wall_width);
    const SanitizedLineWidth infillW = sanitizeLineWidth(spec.solid_infill_width);
    const std::uint8_t infillPattern = spec.solid_infill_pattern;
    const float minExtrusionLengthMm = std::max(0.0f, spec.min_extrusion_length_mm);
    const float sparseDensity = std::clamp(spec.sparse_infill_density, 0.0f, 1.0f);
    const SanitizedLineWidth sparseW = sanitizeLineWidth(spec.sparse_infill_width);
    const std::uint8_t sparsePattern = spec.sparse_infill_pattern;
    const float infillAngleDeg = std::isfinite(spec.infill_angle_deg) ? spec.infill_angle_deg : 45.0f;
    const float nozzleDiameterMm = std::max(0.1f, spec.nozzle_diameter_mm);

    const float wTargetSumConfig =
        outerW.target_mm + static_cast<float>(numInnerWalls) * innerW.target_mm;
    const float wMaxSumConfig =
        outerW.max_mm + static_cast<float>(numInnerWalls) * innerW.max_mm;
    const float remShellAfterTargetWalls = shellThicknessMm - wTargetSumConfig;
    const float effectiveShellDepth =
        (remShellAfterTargetWalls > 0.0f &&
         remShellAfterTargetWalls < infillW.min_mm &&
         shellThicknessMm > wMaxSumConfig)
            ? wTargetSumConfig
            : shellThicknessMm;

    const float hMin = std::max(0.01f, layerHeightSpec.min_mm);
    const float hMax = std::max(hMin, layerHeightSpec.max_mm);
    const float voxelSize = std::max(0.05f, static_cast<float>(sdfGrid.voxelSize()[0]));
    const float coarseVoxelSize = (coarseSdfGrid != nullptr)
        ? std::max(voxelSize, static_cast<float>(coarseSdfGrid->voxelSize()[0]))
        : voxelSize;
    const float fineShellCutoff = (fineShellDepthMm > 0.0f)
        ? -std::max(fineShellDepthMm - 1.75f * voxelSize, 1.5f * voxelSize)
        : -1.0e9f;
    const Vec3f fallbackNozzle = normalize3(
        {defaultNozzleAxisBed[0], defaultNozzleAxisBed[1], defaultNozzleAxisBed[2]},
        {0.0f, 0.0f, 1.0f});

    const std::size_t numLayers = selectedLayers.layers.size();
    std::vector<LayerToolpaths> perLayerResults(numLayers);
    std::atomic<std::size_t> completedLayers{0};

    tbb::parallel_for(
        tbb::blocked_range<std::size_t>(0, numLayers),
        [&](const tbb::blocked_range<std::size_t>& range) {
            auto sdfAcc = sdfGrid.getConstAccessor();
            openvdb::tools::GridSampler<openvdb::FloatGrid::ConstAccessor, openvdb::tools::BoxSampler>
                sdfSampler(sdfAcc, sdfGrid.transform());

            auto eikAcc = eikGrid.getConstAccessor();
            openvdb::tools::GridSampler<openvdb::FloatGrid::ConstAccessor, openvdb::tools::BoxSampler>
                eikSampler(eikAcc, eikGrid.transform());

            using FloatAcc = openvdb::FloatGrid::ConstAccessor;
            using FloatSampler = openvdb::tools::GridSampler<FloatAcc, openvdb::tools::BoxSampler>;
            std::unique_ptr<FloatAcc> cSdfAccPtr;
            std::unique_ptr<FloatSampler> cSdfSamplerPtr;
            if (coarseSdfGrid != nullptr) {
                cSdfAccPtr = std::make_unique<FloatAcc>(coarseSdfGrid->getConstAccessor());
                cSdfSamplerPtr = std::make_unique<FloatSampler>(*cSdfAccPtr, coarseSdfGrid->transform());
            }

            std::unique_ptr<FloatAcc> cEikAccPtr;
            std::unique_ptr<FloatSampler> cEikSamplerPtr;
            if (coarseEikGrid != nullptr) {
                cEikAccPtr = std::make_unique<FloatAcc>(coarseEikGrid->getConstAccessor());
                cEikSamplerPtr = std::make_unique<FloatSampler>(*cEikAccPtr, coarseEikGrid->transform());
            }

            using ToolAcc = openvdb::Vec3SGrid::ConstAccessor;
            using ToolSampler = openvdb::tools::GridSampler<ToolAcc, openvdb::tools::BoxSampler>;
            std::unique_ptr<ToolAcc> tAccPtr;
            std::unique_ptr<ToolSampler> tSamplerPtr;
            if (toolAxisGrid != nullptr) {
                tAccPtr = std::make_unique<ToolAcc>(toolAxisGrid->getConstAccessor());
                tSamplerPtr = std::make_unique<ToolSampler>(*tAccPtr, toolAxisGrid->transform());
            }

            std::unique_ptr<ToolAcc> cToolAccPtr;
            std::unique_ptr<ToolSampler> cToolSamplerPtr;
            if (coarseToolAxisGrid != nullptr) {
                cToolAccPtr = std::make_unique<ToolAcc>(coarseToolAxisGrid->getConstAccessor());
                cToolSamplerPtr = std::make_unique<ToolSampler>(*cToolAccPtr, coarseToolAxisGrid->transform());
            }

            auto sampleSdf = [&](const Vec3f& p) -> float {
                const openvdb::Vec3d wp(p.x, p.y, p.z);
                const float fineSdf = sdfSampler.wsSample(wp);
                if (cSdfSamplerPtr != nullptr && fineSdf <= fineShellCutoff) {
                    return cSdfSamplerPtr->wsSample(wp);
                }
                return fineSdf;
            };

            auto sampleNozzleAxis = [&](const Vec3f& p) -> Vec3f {
                const openvdb::Vec3d wp(p.x, p.y, p.z);
                if (cToolSamplerPtr != nullptr && sdfSampler.wsSample(wp) <= fineShellCutoff) {
                    const openvdb::Vec3s v = cToolSamplerPtr->wsSample(wp);
                    const Vec3f raw{v.x(), v.y(), v.z()};
                    if (lengthSq3(raw) > 1.0e-4f) {
                        return normalize3(raw, fallbackNozzle);
                    }
                }
                if (tSamplerPtr != nullptr) {
                    const openvdb::Vec3s v = tSamplerPtr->wsSample(wp);
                    const Vec3f raw{v.x(), v.y(), v.z()};
                    if (lengthSq3(raw) > 1.0e-4f) {
                        return normalize3(raw, fallbackNozzle);
                    }
                }
                return fallbackNozzle;
            };

            auto sampleEikGradNorm = [&](const Vec3f& p) -> float {
                const openvdb::Vec3d wp(p.x, p.y, p.z);
                const bool useCoarse =
                    (cEikSamplerPtr != nullptr && cSdfSamplerPtr != nullptr &&
                     sdfSampler.wsSample(wp) <= fineShellCutoff);
                const auto& activeEik = useCoarse ? *cEikSamplerPtr : eikSampler;
                const auto& activeSdf = useCoarse ? *cSdfSamplerPtr : sdfSampler;
                const float stepVoxel = useCoarse ? coarseVoxelSize : voxelSize;
                const float h = 0.5f * stepVoxel;
                const float sdfCut = 0.05f * stepVoxel;
                const float t0 = activeEik.wsSample(wp);
                auto axisDeriv = [&](float dx, float dy, float dz) -> float {
                    const openvdb::Vec3d pP(p.x + dx, p.y + dy, p.z + dz);
                    const openvdb::Vec3d pM(p.x - dx, p.y - dy, p.z - dz);
                    const float tP = activeEik.wsSample(pP);
                    const float tM = activeEik.wsSample(pM);
                    const bool inP = activeSdf.wsSample(pP) <= sdfCut;
                    const bool inM = activeSdf.wsSample(pM) <= sdfCut;
                    if (inP && inM) {
                        return (tP - tM) / (2.0f * h);
                    } else if (inP) {
                        return (tP - t0) / h;
                    } else if (inM) {
                        return (t0 - tM) / h;
                    }
                    return (tP - tM) / (2.0f * h);
                };
                const float gx = axisDeriv(h, 0.0f, 0.0f);
                const float gy = axisDeriv(0.0f, h, 0.0f);
                const float gz = axisDeriv(0.0f, 0.0f, h);
                return std::sqrt(gx * gx + gy * gy + gz * gz);
            };

            for (std::size_t layerIdx = range.begin(); layerIdx != range.end(); ++layerIdx) {
                const SelectedLayerInfo& layerInfo = selectedLayers.layers[layerIdx];
                LayerToolpaths layerOut{};
                layerOut.layer_index = layerInfo.layer_index;
                layerOut.branch_id = layerInfo.branch_id;
                layerOut.isovalue = layerInfo.isovalue;

                if (layerInfo.vertices.empty() || layerInfo.indices.size() < 3) {
                    perLayerResults[layerIdx] = std::move(layerOut);
                    const std::size_t done = completedLayers.fetch_add(1, std::memory_order_relaxed) + 1;
                    if (progressCb && (done % 4 == 0 || done == numLayers)) {
                        progressCb(done, numLayers);
                    }
                    continue;
                }

                // ------------------------------------------------------------
                // 1. Build & Weld Initial Clipped Isosurface Mesh S_k
                // ------------------------------------------------------------
                SurfaceMesh baseMesh;
                baseMesh.verts.reserve(layerInfo.vertices.size());
                const float meanLayerH = std::max(0.02f, layerInfo.mean_thickness_mm);

                // First sample only SDF on baseMesh so we can clip out the sparse interior core
                // before sampling toolAxis / eikGradNorm or subdividing triangles!
                for (const auto& rv : layerInfo.vertices) {
                    SurfaceVertex sv{};
                    sv.pos = {rv.x, rv.y, rv.z};
                    sv.normal = normalize3({rv.nx, rv.ny, rv.nz}, fallbackNozzle);
                    sv.nozzleAxis = fallbackNozzle;
                    sv.sdf = sampleSdf(sv.pos);
                    sv.layerHeightMm = meanLayerH;
                    sv.onOuterBoundary = false;
                    sv.onInnerShellBoundary = false;
                    baseMesh.verts.push_back(sv);
                }

                baseMesh.tris.reserve(layerInfo.indices.size() / 3);
                for (std::size_t t = 0; t + 2 < layerInfo.indices.size(); t += 3) {
                    baseMesh.tris.push_back({
                        layerInfo.indices[t + 0],
                        layerInfo.indices[t + 1],
                        layerInfo.indices[t + 2],
                    });
                }

                baseMesh = weldSurfaceMesh(baseMesh, 1.0e-4f);
                if (baseMesh.tris.empty()) {
                    perLayerResults[layerIdx] = std::move(layerOut);
                    const std::size_t done = completedLayers.fetch_add(1, std::memory_order_relaxed) + 1;
                    if (progressCb && (done % 4 == 0 || done == numLayers)) {
                        progressCb(done, numLayers);
                    }
                    continue;
                }

                const float layerMinH = 0.02f;
                const float layerMaxH = std::max({
                    2.0f * hMax,
                    meanLayerH,
                    layerInfo.max_thickness_mm});

                std::vector<float> baseVertGrads(baseMesh.verts.size(), 1.0f);
                double sumLayerInvGrad = 0.0;
                std::size_t cntLayerGrad = 0;
                for (std::size_t i = 0; i < baseMesh.verts.size(); ++i) {
                    const float g = sampleEikGradNorm(baseMesh.verts[i].pos);
                    baseVertGrads[i] = g;
                    if (g > 0.20f && g < 4.0f) {
                        sumLayerInvGrad += 1.0 / static_cast<double>(g);
                        cntLayerGrad += 1;
                    }
                }
                // Use the harmonic mean of |\nabla \tau| so E[meanLayerH * g_harm / g] == meanLayerH
                // without positive Jensen's inequality drift from 1/g convexity.
                const float layerMeanGrad = (cntLayerGrad > 0 && sumLayerInvGrad > 1.0e-8)
                    ? static_cast<float>(static_cast<double>(cntLayerGrad) / sumLayerInvGrad)
                    : 1.0f;

                auto sampleLocalLayerHeight = [&](const Vec3f& p) -> float {
                    const float gradNorm = sampleEikGradNorm(p);
                    if (gradNorm > 0.20f) {
                        return std::clamp(
                            meanLayerH * (layerMeanGrad / gradNorm),
                            layerMinH,
                            layerMaxH);
                    }
                    return meanLayerH;
                };

                for (std::size_t i = 0; i < baseMesh.verts.size(); ++i) {
                    const float g = baseVertGrads[i];
                    baseMesh.verts[i].layerHeightMm = (g > 0.20f)
                        ? std::clamp(meanLayerH * (layerMeanGrad / g), layerMinH, layerMaxH)
                        : meanLayerH;
                }

                // Identify outer boundary edges of S_k (where S_k intersects the outer surface \phi_sdf = 0)
                std::unordered_map<EdgeKey, std::uint32_t, EdgeKeyHash> edgeTriCounts;
                edgeTriCounts.reserve(baseMesh.tris.size() * 3);
                for (const auto& tri : baseMesh.tris) {
                    edgeTriCounts[EdgeKey(tri[0], tri[1])] += 1;
                    edgeTriCounts[EdgeKey(tri[1], tri[2])] += 1;
                    edgeTriCounts[EdgeKey(tri[2], tri[0])] += 1;
                }

                float maxBoundarySdf = -std::numeric_limits<float>::max();
                bool anyOuterBoundary = false;
                for (const auto& [ek, count] : edgeTriCounts) {
                    if (count == 1) {
                        const float edgeSdf = 0.5f * (baseMesh.verts[ek.a].sdf + baseMesh.verts[ek.b].sdf);
                        if (edgeSdf >= -0.85f * voxelSize) {
                            baseMesh.verts[ek.a].onOuterBoundary = true;
                            baseMesh.verts[ek.b].onOuterBoundary = true;
                            maxBoundarySdf = std::max({
                                maxBoundarySdf,
                                baseMesh.verts[ek.a].sdf,
                                baseMesh.verts[ek.b].sdf,
                            });
                            anyOuterBoundary = true;
                        }
                    }
                }

                if (!anyOuterBoundary) {
                    for (const auto& [ek, count] : edgeTriCounts) {
                        if (count == 1) {
                            baseMesh.verts[ek.a].onOuterBoundary = true;
                            baseMesh.verts[ek.b].onOuterBoundary = true;
                            maxBoundarySdf = std::max({
                                maxBoundarySdf,
                                baseMesh.verts[ek.a].sdf,
                                baseMesh.verts[ek.b].sdf,
                            });
                        }
                    }
                }

                // Normalize vertex SDF relative to the outer boundary contour so \phi_sdf = 0 on \Gamma_outer
                const float sdfZeroShift = std::isfinite(maxBoundarySdf)
                    ? std::max(0.0f, maxBoundarySdf)
                    : 0.0f;
                for (auto& v : baseMesh.verts) {
                    if (v.onOuterBoundary) {
                        v.sdf = 0.0f;
                    } else {
                        v.sdf = std::min(-1.0e-4f, v.sdf - sdfZeroShift);
                    }
                }

                // ------------------------------------------------------------
                // 2. Clip S_k to the Outer Shell Domain (-T_shell <= \phi_sdf <= 0)
                //    BEFORE subdividing or sampling nozzleAxis / eikGradNorm!
                // ------------------------------------------------------------
                std::vector<float> sdfDepth(baseMesh.verts.size(), 0.0f);
                for (std::size_t i = 0; i < baseMesh.verts.size(); ++i) {
                    sdfDepth[i] = std::max(0.0f, -baseMesh.verts[i].sdf);
                }

                SurfaceMesh shellMesh = clipSurfaceMeshByScalar(
                    baseMesh,
                    sdfDepth,
                    shellThicknessMm,
                    true);

                if (shellMesh.tris.empty() || shellMesh.verts.empty()) {
                    perLayerResults[layerIdx] = std::move(layerOut);
                    const std::size_t done = completedLayers.fetch_add(1, std::memory_order_relaxed) + 1;
                    if (progressCb && (done % 4 == 0 || done == numLayers)) {
                        progressCb(done, numLayers);
                    }
                    continue;
                }

                // Subdivide the already-clipped shell strip so narrow top-layer strips have interior vertices
                const float targetRefEdge = std::clamp(
                    0.55f * std::min({outerW.target_mm, innerW.target_mm, shellThicknessMm}),
                    0.18f,
                    0.35f);
                shellMesh = subdivideShellMesh(
                    shellMesh,
                    targetRefEdge,
                    [&](const Vec3f& p) {
                        return std::min(-1.0e-4f, sampleSdf(p) - sdfZeroShift);
                    });

                // Sample 5/6-axis nozzle orientation and local layer height only on the surviving shell vertices
                {
                    std::vector<float> shellGrads(shellMesh.verts.size(), 1.0f);
                    double sumShellInvGrad = 0.0;
                    std::size_t cntShellGrad = 0;
                    for (std::size_t i = 0; i < shellMesh.verts.size(); ++i) {
                        shellMesh.verts[i].nozzleAxis = sampleNozzleAxis(shellMesh.verts[i].pos);
                        const float g = sampleEikGradNorm(shellMesh.verts[i].pos);
                        shellGrads[i] = g;
                        if (g > 0.20f && g < 4.0f) {
                            sumShellInvGrad += 1.0 / static_cast<double>(g);
                            cntShellGrad += 1;
                        }
                    }
                    const float shellMeanGrad = (cntShellGrad > 0 && sumShellInvGrad > 1.0e-8)
                        ? static_cast<float>(static_cast<double>(cntShellGrad) / sumShellInvGrad)
                        : layerMeanGrad;
                    for (std::size_t i = 0; i < shellMesh.verts.size(); ++i) {
                        const float g = shellGrads[i];
                        shellMesh.verts[i].layerHeightMm = (g > 0.20f)
                            ? std::clamp(meanLayerH * (shellMeanGrad / g), layerMinH, layerMaxH)
                            : meanLayerH;
                    }
                }

                // ------------------------------------------------------------
                // 3. Compute On-Surface Geodesic Inset Field & Medial Clearance
                // ------------------------------------------------------------
                const std::size_t nV = shellMesh.verts.size();
                std::vector<std::vector<std::uint32_t>> adj(nV);
                std::unordered_map<EdgeKey, std::uint32_t, EdgeKeyHash> shellEdgeTriCounts;
                shellEdgeTriCounts.reserve(shellMesh.tris.size() * 3);
                for (const auto& tri : shellMesh.tris) {
                    shellEdgeTriCounts[EdgeKey(tri[0], tri[1])] += 1;
                    shellEdgeTriCounts[EdgeKey(tri[1], tri[2])] += 1;
                    shellEdgeTriCounts[EdgeKey(tri[2], tri[0])] += 1;
                    for (int k = 0; k < 3; ++k) {
                        const std::uint32_t u = tri[k];
                        const std::uint32_t v = tri[(k + 1) % 3];
                        adj[u].push_back(v);
                        adj[v].push_back(u);
                    }
                }

                // Ensure only vertices belonging to true outer boundary edges (incident to 1 triangle)
                // are marked as `onOuterBoundary`, and collect exact outer boundary segments.
                struct OuterSegment {
                    Vec3f a;
                    Vec3f b;
                    Vec3f ab;
                    Vec3f inwardNormal;
                    float abLenSq;
                };
                std::vector<OuterSegment> outerSegments;
                outerSegments.reserve(nV / 2 + 8);

                std::vector<bool> trueOuterVert(nV, false);
                std::vector<Vec3f> outerInwardNormal(nV, {0.0f, 0.0f, 0.0f});
                for (const auto& tri : shellMesh.tris) {
                    for (int k = 0; k < 3; ++k) {
                        const std::uint32_t u = tri[k];
                        const std::uint32_t v = tri[(k + 1) % 3];
                        const std::uint32_t w = tri[(k + 2) % 3];
                        if (shellMesh.verts[u].onOuterBoundary && shellMesh.verts[v].onOuterBoundary) {
                            auto cntIt = shellEdgeTriCounts.find(EdgeKey(u, v));
                            if (cntIt != shellEdgeTriCounts.end() && cntIt->second == 1) {
                                trueOuterVert[u] = true;
                                trueOuterVert[v] = true;
                                const Vec3f a = shellMesh.verts[u].pos;
                                const Vec3f b = shellMesh.verts[v].pos;
                                const Vec3f ab = b - a;
                                const Vec3f edgeDir = normalize3(ab);
                                const Vec3f toInterior = shellMesh.verts[w].pos - a;
                                const Vec3f inward = normalize3(
                                    toInterior - (edgeDir * dot3(toInterior, edgeDir)),
                                    {1.0f, 0.0f, 0.0f});
                                outerInwardNormal[u] = outerInwardNormal[u] + inward;
                                outerInwardNormal[v] = outerInwardNormal[v] + inward;
                                outerSegments.push_back(OuterSegment{
                                    a,
                                    b,
                                    ab,
                                    inward,
                                    lengthSq3(ab),
                                });
                            }
                        }
                    }
                }
                for (std::size_t i = 0; i < nV; ++i) {
                    shellMesh.verts[i].onOuterBoundary = trueOuterVert[i];
                    if (trueOuterVert[i]) {
                        outerInwardNormal[i] = normalize3(outerInwardNormal[i], {1.0f, 0.0f, 0.0f});
                    }
                }

                if (outerSegments.empty()) {
                    perLayerResults[layerIdx] = std::move(layerOut);
                    const std::size_t done = completedLayers.fetch_add(1, std::memory_order_relaxed) + 1;
                    if (progressCb && (done % 4 == 0 || done == numLayers)) {
                        progressCb(done, numLayers);
                    }
                    continue;
                }

                // Build a 3D spatial hash grid of outer boundary segments so every vertex can
                // query exact point-to-segment distance `dOuter` and opposing-wall clearance `rMedial`.
                const float medialCellSize = std::max(0.6f, 1.5f * shellThicknessMm);
                const float invMedialCell = 1.0f / medialCellSize;
                std::unordered_map<GridKey3D, std::vector<std::uint32_t>, GridKey3DHash> edgeGrid;
                edgeGrid.reserve(outerSegments.size() * 2 + 8);
                for (std::size_t sIdx = 0; sIdx < outerSegments.size(); ++sIdx) {
                    const auto& seg = outerSegments[sIdx];
                    const Vec3f minPt{
                        std::min(seg.a.x, seg.b.x),
                        std::min(seg.a.y, seg.b.y),
                        std::min(seg.a.z, seg.b.z)};
                    const Vec3f maxPt{
                        std::max(seg.a.x, seg.b.x),
                        std::max(seg.a.y, seg.b.y),
                        std::max(seg.a.z, seg.b.z)};
                    const GridKey3D kMin = toGridKey(minPt, invMedialCell);
                    const GridKey3D kMax = toGridKey(maxPt, invMedialCell);
                    for (std::int64_t iz = kMin.iz; iz <= kMax.iz; ++iz) {
                        for (std::int64_t iy = kMin.iy; iy <= kMax.iy; ++iy) {
                            for (std::int64_t ix = kMin.ix; ix <= kMax.ix; ++ix) {
                                edgeGrid[GridKey3D{ix, iy, iz}].push_back(static_cast<std::uint32_t>(sIdx));
                            }
                        }
                    }
                }

                const float kInf = 1.0e9f;
                std::vector<float> dOuter(nV, kInf);
                std::vector<Vec3f> closestOuterNormal(nV, {1.0f, 0.0f, 0.0f});
                std::vector<float> rMedial(nV, shellThicknessMm * 4.0f);

                using QueueItem = std::pair<float, std::uint32_t>;
                std::priority_queue<QueueItem, std::vector<QueueItem>, std::greater<QueueItem>> pq;

                struct CandSeg {
                    std::uint32_t segIdx;
                    float dist;
                    Vec3f foot;
                };
                std::vector<CandSeg> cands;
                cands.reserve(64);

                for (std::size_t i = 0; i < nV; ++i) {
                    const Vec3f p = shellMesh.verts[i].pos;
                    const GridKey3D baseKey = toGridKey(p, invMedialCell);
                    cands.clear();

                    float bestDist = shellMesh.verts[i].onOuterBoundary ? 0.0f : kInf;
                    Vec3f bestFoot = p;
                    Vec3f bestNormal = shellMesh.verts[i].onOuterBoundary
                        ? outerInwardNormal[i]
                        : Vec3f{1.0f, 0.0f, 0.0f};

                    for (std::int64_t dz = -1; dz <= 1; ++dz) {
                        for (std::int64_t dy = -1; dy <= 1; ++dy) {
                            for (std::int64_t dx = -1; dx <= 1; ++dx) {
                                const GridKey3D nbKey{baseKey.ix + dx, baseKey.iy + dy, baseKey.iz + dz};
                                auto it = edgeGrid.find(nbKey);
                                if (it == edgeGrid.end()) {
                                    continue;
                                }
                                for (const std::uint32_t sIdx : it->second) {
                                    const auto& seg = outerSegments[sIdx];
                                    const float tProj = (seg.abLenSq > 1.0e-12f)
                                        ? std::clamp(dot3(p - seg.a, seg.ab) / seg.abLenSq, 0.0f, 1.0f)
                                        : 0.5f;
                                    const Vec3f foot = seg.a + seg.ab * tProj;
                                    const float d = dist3(p, foot);
                                    cands.push_back(CandSeg{sIdx, d, foot});
                                    if (!shellMesh.verts[i].onOuterBoundary && d < bestDist) {
                                        bestDist = d;
                                        bestFoot = foot;
                                        bestNormal = seg.inwardNormal;
                                    }
                                }
                            }
                        }
                    }

                    if (bestDist < kInf * 0.5f) {
                        dOuter[i] = bestDist;
                        closestOuterNormal[i] = bestNormal;
                        pq.push({bestDist, static_cast<std::uint32_t>(i)});

                        const Vec3f v1 = p - bestFoot;
                        float minOppDist = kInf;
                        for (const auto& c : cands) {
                            if (c.dist >= minOppDist) {
                                continue;
                            }
                            const auto& seg = outerSegments[c.segIdx];
                            const float nDot = dot3(bestNormal, seg.inwardNormal);
                            const Vec3f v2 = p - c.foot;
                            const bool opposingRay =
                                (bestDist > 1.0e-4f && c.dist > 1.0e-4f) &&
                                (dot3(v1, v2) < -0.20f * bestDist * c.dist);
                            const bool opposingBoundary =
                                (nDot < -0.15f) &&
                                (c.dist > 1.0e-3f || !shellMesh.verts[i].onOuterBoundary);
                            if (opposingBoundary || opposingRay) {
                                minOppDist = c.dist;
                            }
                        }
                        if (minOppDist < kInf * 0.5f) {
                            rMedial[i] = 0.5f * (bestDist + minOppDist);
                        }
                    }
                }

                // Dijkstra fallback for any vertices beyond the 3x3x3 spatial grid neighborhood
                while (!pq.empty()) {
                    const auto [dCurr, u] = pq.top();
                    pq.pop();
                    if (dCurr > dOuter[u] + 1.0e-6f) {
                        continue;
                    }
                    for (std::uint32_t v : adj[u]) {
                        const float wEdge = dist3(shellMesh.verts[u].pos, shellMesh.verts[v].pos);
                        if (dCurr + wEdge < dOuter[v]) {
                            dOuter[v] = dCurr + wEdge;
                            closestOuterNormal[v] = closestOuterNormal[u];
                            pq.push({dOuter[v], v});
                        }
                    }
                }

                // Also detect medial shock edges directly on the mesh where adjacent vertices
                // originate from opposing outer boundary normals
                for (std::size_t i = 0; i < nV; ++i) {
                    for (std::uint32_t nb : adj[i]) {
                        if (dot3(closestOuterNormal[i], closestOuterNormal[nb]) < -0.15f) {
                            const float edgeLen = dist3(shellMesh.verts[i].pos, shellMesh.verts[nb].pos);
                            const float shockHalfSpan = 0.5f * (dOuter[i] + dOuter[nb] + edgeLen);
                            rMedial[i] = std::min(rMedial[i], shockHalfSpan);
                        }
                    }
                }

                // Multi-source Dijkstra from \Gamma_inner (\phi_sdf = -T_shell) to measure remaining shell distance
                std::vector<float> dInner(nV, kInf);
                bool hasInnerShellBoundary = false;
                for (std::size_t i = 0; i < nV; ++i) {
                    if (shellMesh.verts[i].onInnerShellBoundary) {
                        dInner[i] = 0.0f;
                        pq.push({0.0f, static_cast<std::uint32_t>(i)});
                        hasInnerShellBoundary = true;
                    }
                }
                if (hasInnerShellBoundary) {
                    while (!pq.empty()) {
                        const auto [dCurr, u] = pq.top();
                        pq.pop();
                        if (dCurr > dInner[u] + 1.0e-6f) {
                            continue;
                        }
                        for (std::uint32_t v : adj[u]) {
                            const float wEdge = dist3(shellMesh.verts[u].pos, shellMesh.verts[v].pos);
                            if (dCurr + wEdge < dInner[v]) {
                                dInner[v] = dCurr + wEdge;
                                pq.push({dInner[v], v});
                            }
                        }
                    }
                }

                // Smooth `rMedial` across 1-ring neighborhoods so line widths vary smoothly
                {
                    std::vector<float> rSmooth = rMedial;
                    for (std::size_t i = 0; i < nV; ++i) {
                        if (adj[i].empty()) {
                            continue;
                        }
                        float sum = rMedial[i];
                        for (std::uint32_t nb : adj[i]) {
                            sum += rMedial[nb];
                        }
                        rSmooth[i] = sum / static_cast<float>(adj[i].size() + 1);
                    }
                    rMedial = std::move(rSmooth);
                }

                // Compute local peak `dOuter` across a 2-ring neighborhood so outer-wall
                // offsets on narrow top strips never exceed the actual mesh peak `dOuter`.
                std::vector<float> localPeakDOuter = dOuter;
                for (int ring = 0; ring < 2; ++ring) {
                    std::vector<float> nextPeak = localPeakDOuter;
                    for (std::size_t i = 0; i < nV; ++i) {
                        for (std::uint32_t nb : adj[i]) {
                            nextPeak[i] = std::max(nextPeak[i], localPeakDOuter[nb]);
                        }
                    }
                    localPeakDOuter = std::move(nextPeak);
                }

                // Total available one-sided shell span B(v) at each vertex
                std::vector<float> spanShell(nV, shellThicknessMm);
                std::vector<float> availableSpan(nV, shellThicknessMm);
                for (std::size_t i = 0; i < nV; ++i) {
                    const float sShell = hasInnerShellBoundary
                        ? std::max(dOuter[i] + dInner[i], shellThicknessMm)
                        : std::max(dOuter[i], rMedial[i]);
                    spanShell[i] = sShell;
                    availableSpan[i] = std::min(rMedial[i], sShell);
                }

                // ------------------------------------------------------------
                // 4. Variable Line Width Allocation for Outer & Inner Walls
                // ------------------------------------------------------------
                std::vector<float> localOuterWidth(nV, outerW.target_mm);
                std::vector<float> localInnerWidth(nV, innerW.target_mm);
                std::vector<std::uint32_t> activeInnerCount(nV, numInnerWalls);
                std::vector<float> totalWallSpan(nV, 0.0f);

                for (std::size_t i = 0; i < nV; ++i) {
                    const float B = availableSpan[i];
                    if (B < outerW.min_mm) {
                        localOuterWidth[i] = outerW.min_mm;
                        localInnerWidth[i] = innerW.min_mm;
                        activeInnerCount[i] = 0;
                        totalWallSpan[i] = (B >= 0.5f * outerW.min_mm) ? outerW.min_mm : 0.0f;
                        continue;
                    }

                    std::uint32_t mFit = 0;
                    for (std::uint32_t m = 1; m <= numInnerWalls; ++m) {
                        const float wMinSum = outerW.min_mm + static_cast<float>(m) * innerW.min_mm;
                        if (B >= wMinSum) {
                            mFit = m;
                        } else {
                            break;
                        }
                    }
                    activeInnerCount[i] = mFit;

                    const float wMinSum = outerW.min_mm + static_cast<float>(mFit) * innerW.min_mm;
                    const float wTargetSum =
                        outerW.target_mm + static_cast<float>(mFit) * innerW.target_mm;
                    const float wMaxSum = outerW.max_mm + static_cast<float>(mFit) * innerW.max_mm;

                    if (B < wTargetSum) {
                        const float denom = std::max(1.0e-5f, wTargetSum - wMinSum);
                        const float t = std::clamp((B - wMinSum) / denom, 0.0f, 1.0f);
                        localOuterWidth[i] =
                            outerW.min_mm + t * (outerW.target_mm - outerW.min_mm);
                        localInnerWidth[i] =
                            innerW.min_mm + t * (innerW.target_mm - innerW.min_mm);
                    } else {
                        const float remGap = (rMedial[i] <= spanShell[i])
                            ? 2.0f * (rMedial[i] - wTargetSum)
                            : (B - wTargetSum);
                        if (remGap > 0.0f && remGap < infillW.min_mm && B <= wMaxSum) {
                            const float denom = std::max(1.0e-5f, wMaxSum - wTargetSum);
                            const float t = std::clamp((B - wTargetSum) / denom, 0.0f, 1.0f);
                            localOuterWidth[i] =
                                outerW.target_mm + t * (outerW.max_mm - outerW.target_mm);
                            localInnerWidth[i] =
                                innerW.target_mm + t * (innerW.max_mm - innerW.target_mm);
                        } else {
                            localOuterWidth[i] = outerW.target_mm;
                            localInnerWidth[i] = innerW.target_mm;
                        }
                    }

                    totalWallSpan[i] =
                        localOuterWidth[i] + static_cast<float>(mFit) * localInnerWidth[i];
                }

                // ------------------------------------------------------------
                // 5. Extract Outer Wall (Contacts \phi_sdf = 0) & Inner Walls
                // ------------------------------------------------------------
                const float stitchSnap = std::max(0.05f, 0.25f * targetRefEdge);
                const float closeLoopTol = std::max(0.15f, 0.65f * outerW.target_mm);

                // 5a. Outer wall: centerline at d_outer(x) = 0.5 * w_outer(x) (clamped on thin top crests)
                {
                    std::vector<float> outerLevel(nV, 0.0f);
                    for (std::size_t i = 0; i < nV; ++i) {
                        const float stripHalfSpan = std::min(
                            availableSpan[i],
                            std::max(dOuter[i], localPeakDOuter[i]));
                        const float effOffset = std::min(
                            0.5f * localOuterWidth[i],
                            0.65f * stripHalfSpan);
                        outerLevel[i] = dOuter[i] - effOffset;
                    }
                    const auto outerSegs = extractIsoContourSegments(
                        shellMesh,
                        outerLevel,
                        0.0f,
                        availableSpan,
                        0.02f,
                        localOuterWidth);
                    const auto outerChains =
                        stitchSegmentsIntoChains(outerSegs, stitchSnap, closeLoopTol);

                    for (const auto& ch : outerChains) {
                        ExtrusionPath ep{};
                        if (finalizeExtrusionPath(
                                ch,
                                kFeatureOuterWall,
                                0,
                                minExtrusionLengthMm,
                                ep)) {
                            layerOut.paths.push_back(std::move(ep));
                        }
                    }
                }

                // 5b. Inner walls: j = 1 .. numInnerWalls (may be closed loops or open polylines!)
                for (std::uint32_t j = 1; j <= numInnerWalls; ++j) {
                    const float minSpanForWallJ =
                        outerW.min_mm + static_cast<float>(j) * innerW.min_mm;
                    std::vector<float> innerLevel(nV, 0.0f);
                    for (std::size_t i = 0; i < nV; ++i) {
                        const float centerOffset =
                            localOuterWidth[i] + (static_cast<float>(j) - 0.5f) * localInnerWidth[i];
                        innerLevel[i] = dOuter[i] - centerOffset;
                    }

                    const auto innerSegs = extractIsoContourSegments(
                        shellMesh,
                        innerLevel,
                        0.0f,
                        availableSpan,
                        minSpanForWallJ,
                        localInnerWidth);
                    const auto innerChains =
                        stitchSegmentsIntoChains(innerSegs, stitchSnap, closeLoopTol);

                    for (const auto& ch : innerChains) {
                        ExtrusionPath ep{};
                        if (finalizeExtrusionPath(
                                ch,
                                kFeatureInnerWall,
                                j,
                                minExtrusionLengthMm,
                                ep)) {
                            layerOut.paths.push_back(std::move(ep));
                        }
                    }
                }

                // ------------------------------------------------------------
                // 6. Shell Gap Fill & Solid Infill (`Monotonic` or `Concentric`)
                // ------------------------------------------------------------
                std::vector<float> negDRem(nV, 0.0f);
                std::vector<float> remOneSidedSpan(nV, 0.0f);
                for (std::size_t i = 0; i < nV; ++i) {
                    const float dRem = dOuter[i] - totalWallSpan[i];
                    negDRem[i] = -dRem;
                    remOneSidedSpan[i] = std::max(0.0f, availableSpan[i] - totalWallSpan[i]);
                    shellMesh.verts[i].dRem = std::max(0.0f, dRem);
                    shellMesh.verts[i].remSpan = remOneSidedSpan[i];
                }

                // 6a. Gap Fill in narrow medial channels where a single variable-width
                //     bead in [infillW.min_mm, infillW.max_mm] fills the remaining gap
                {
                    std::vector<float> gapLevel(nV, 0.0f);
                    std::vector<float> gapValidity(nV, -1.0f);
                    std::vector<float> gapWidth(nV, infillW.target_mm);

                    for (std::size_t i = 0; i < nV; ++i) {
                        const bool isMedialLimited = (rMedial[i] < spanShell[i] - 0.05f);
                        const float fullGapWidth = 2.0f * remOneSidedSpan[i];
                        const float gapCenterOffset = remOneSidedSpan[i];

                        gapLevel[i] = (dOuter[i] - totalWallSpan[i]) - gapCenterOffset;
                        if (isMedialLimited &&
                            fullGapWidth >= infillW.min_mm &&
                            fullGapWidth <= infillW.max_mm) {
                            gapValidity[i] = 1.0f;
                            gapWidth[i] = std::clamp(fullGapWidth, infillW.min_mm, infillW.max_mm);
                        } else {
                            gapValidity[i] = -1.0f;
                            gapWidth[i] = infillW.target_mm;
                        }
                    }

                    const auto gapSegs = extractIsoContourSegments(
                        shellMesh,
                        gapLevel,
                        0.0f,
                        gapValidity,
                        0.0f,
                        gapWidth);
                    const auto gapChains =
                        stitchSegmentsIntoChains(gapSegs, stitchSnap, closeLoopTol);

                    std::uint32_t gapIdx = 0;
                    for (const auto& ch : gapChains) {
                        ExtrusionPath ep{};
                        if (finalizeExtrusionPath(
                                ch,
                                kFeatureGapFill,
                                gapIdx,
                                minExtrusionLengthMm,
                                ep)) {
                            layerOut.paths.push_back(std::move(ep));
                            ++gapIdx;
                        }
                    }
                }

                // 6b. Solid Infill in the remaining shell region where `dRem >= 0`
                for (std::size_t i = 0; i < nV; ++i) {
                    const bool isMedialLimited = (rMedial[i] < spanShell[i] - 0.05f);
                    const float fullRemWidth = isMedialLimited
                        ? 2.0f * remOneSidedSpan[i]
                        : remOneSidedSpan[i];
                    if ((isMedialLimited && fullRemWidth <= infillW.max_mm) ||
                        fullRemWidth < infillW.min_mm) {
                        negDRem[i] = 1.0f;
                    } else if (infillPattern != kPatternConcentric) {
                        // Inset Monotonic solid infill scanline endpoints by `(0.5 - eta) * w_solid`
                        // (`0.35 * w_solid`) from the physical inner edge of the walls (and from
                        // \Gamma_inner when `hasInnerShellBoundary` is true) so semicircular nozzle
                        // caps overlap the adjacent perimeter by `eta = 0.15` without over-extruding.
                        const float endpointInset = std::min(
                            kInfillEndpointInsetFactor * infillW.target_mm,
                            0.25f * fullRemWidth);
                        const float dRemFromWalls = (dOuter[i] - totalWallSpan[i]) - endpointInset;
                        const float dRemBoth = hasInnerShellBoundary
                            ? std::min(dRemFromWalls, dInner[i] - endpointInset)
                            : dRemFromWalls;
                        negDRem[i] = -dRemBoth;
                    }
                }

                SurfaceMesh infillSubmesh = clipSurfaceMeshByScalar(
                    shellMesh,
                    negDRem,
                    0.0f,
                    false);

                if (!infillSubmesh.tris.empty() && !infillSubmesh.verts.empty()) {
                    if (infillPattern == kPatternConcentric) {
                        const std::size_t nInfV = infillSubmesh.verts.size();
                        std::vector<float> infDepth(nInfV, 0.0f);
                        std::vector<float> infSpan(nInfV, 0.0f);
                        std::vector<float> infLocalWidth(nInfV, infillW.target_mm);
                        float maxRemSpan = 0.0f;

                        for (std::size_t i = 0; i < nInfV; ++i) {
                            const float dInfill = std::max(0.0f, infillSubmesh.verts[i].dRem);
                            const float totSpan = std::max(dInfill, infillSubmesh.verts[i].remSpan);
                            infDepth[i] = dInfill;
                            infSpan[i] = totSpan;
                            maxRemSpan = std::max(maxRemSpan, totSpan);

                            const int nPasses = std::max(
                                1,
                                static_cast<int>(std::round(totSpan / infillW.target_mm)));
                            infLocalWidth[i] = std::clamp(
                                totSpan / static_cast<float>(nPasses),
                                infillW.min_mm,
                                infillW.max_mm);
                        }

                        const int maxPasses = std::clamp(
                            static_cast<int>(std::ceil(maxRemSpan / infillW.min_mm)),
                            1,
                            64);
                        std::uint32_t passIdx = 0;

                        for (int p = 0; p < maxPasses; ++p) {
                            std::vector<float> passLevel(nInfV, 0.0f);
                            for (std::size_t i = 0; i < nInfV; ++i) {
                                const float centerD =
                                    (static_cast<float>(p) + 0.5f) * infLocalWidth[i];
                                passLevel[i] = infDepth[i] - centerD;
                            }
                            const float minReqSpan =
                                (static_cast<float>(p) + 0.5f) * infillW.min_mm;
                            const auto passSegs = extractIsoContourSegments(
                                infillSubmesh,
                                passLevel,
                                0.0f,
                                infSpan,
                                minReqSpan,
                                infLocalWidth);
                            const auto passChains =
                                stitchSegmentsIntoChains(passSegs, stitchSnap, closeLoopTol);

                            for (const auto& ch : passChains) {
                                ExtrusionPath ep{};
                                if (finalizeExtrusionPath(
                                        ch,
                                        kFeatureSolidInfill,
                                        passIdx,
                                        minExtrusionLengthMm,
                                        ep)) {
                                    layerOut.paths.push_back(std::move(ep));
                                    ++passIdx;
                                }
                            }
                        }
                    } else {
                        generateMonotonicInfillOnSubmesh(
                            infillSubmesh,
                            layerInfo.layer_index,
                            infillW,
                            minExtrusionLengthMm,
                            layerOut.paths,
                            infillAngleDeg,
                            kFeatureSolidInfill);
                    }
                }

                // ------------------------------------------------------------
                // 7. Non-Planar Sparse Infill on Interior Core (\phi_sdf <= -effectiveShellDepth)
                // ------------------------------------------------------------
                if (sparseDensity > 1.0e-4f) {
                    generateSparseInfillForLayer(
                        baseMesh,
                        sdfDepth,
                        effectiveShellDepth,
                        sdfZeroShift,
                        layerInfo.layer_index,
                        layerInfo.isovalue,
                        sparseDensity,
                        sparseW,
                        sparsePattern,
                        infillAngleDeg,
                        nozzleDiameterMm,
                        infillW,
                        infillPattern,
                        minExtrusionLengthMm,
                        stitchSnap,
                        closeLoopTol,
                        sampleSdf,
                        sampleNozzleAxis,
                        sampleLocalLayerHeight,
                        layerOut.paths);
                }

                // Normalize emitted toolpath point heights so the layer's mean segment height
                // matches `meanLayerH` (`layerInfo.mean_thickness_mm`) while preserving relative
                // intra-layer conformal thickness variations (1 / |\nabla \tau|).
                {
                    double sumPtH = 0.0;
                    std::size_t cntPtH = 0;
                    for (const auto& path : layerOut.paths) {
                        for (const auto& pt : path.points) {
                            sumPtH += static_cast<double>(pt.height_mm);
                            ++cntPtH;
                        }
                    }
                    if (cntPtH > 0) {
                        const float rawMeanPtH =
                            static_cast<float>(sumPtH / static_cast<double>(cntPtH));
                        if (rawMeanPtH > 1.0e-4f) {
                            const float hScale = meanLayerH / rawMeanPtH;
                            for (auto& path : layerOut.paths) {
                                for (auto& pt : path.points) {
                                    pt.height_mm =
                                        std::clamp(pt.height_mm * hScale, layerMinH, layerMaxH);
                                }
                            }
                        }
                    }
                }

                perLayerResults[layerIdx] = std::move(layerOut);
                const std::size_t done = completedLayers.fetch_add(1, std::memory_order_relaxed) + 1;
                if (progressCb && (done % 4 == 0 || done == numLayers)) {
                    progressCb(done, numLayers);
                }
            }
        });

    result.valid = true;
    result.layers.reserve(perLayerResults.size());
    for (auto& layer : perLayerResults) {
        for (const auto& path : layer.paths) {
            result.total_path_count += 1;
            if (path.feature_kind == kFeatureOuterWall) {
                result.outer_wall_count += 1;
            } else if (path.feature_kind == kFeatureInnerWall) {
                result.inner_wall_count += 1;
            } else if (path.feature_kind == kFeatureSparseInfill) {
                result.sparse_infill_count += 1;
            } else {
                result.solid_infill_count += 1;
            }
        }
        result.layers.push_back(std::move(layer));
    }

    return result;
}

} // namespace manifold
