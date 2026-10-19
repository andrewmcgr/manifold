#include "path_generator.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
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

constexpr std::uint8_t kPatternConcentric = 1;

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
    out.onOuterBoundary = a.onOuterBoundary && b.onOuterBoundary;
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
        const GridKey3D key{
            static_cast<std::int64_t>(std::llround(v.pos.x * invCell)),
            static_cast<std::int64_t>(std::llround(v.pos.y * invCell)),
            static_cast<std::int64_t>(std::llround(v.pos.z * invCell)),
        };
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

/// Subdivides triangles in the shell band once (4-to-1 midpoint split on triangles with edges
/// longer than `maxEdgeLen`) so that contour extraction and medial-axis detection have
/// fine resolution relative to the extrusion line width.
template <typename SdfSamplerFn>
SurfaceMesh subdivideShellMesh(
    const SurfaceMesh& inMesh,
    float maxEdgeLen,
    float shellSdfFloor,
    SdfSamplerFn&& sampleSdfAt)
{
    if (inMesh.tris.empty() || inMesh.verts.empty()) {
        return inMesh;
    }

    SurfaceMesh current = inMesh;
    const float maxEdgeSq = maxEdgeLen * maxEdgeLen;

    for (int pass = 0; pass < 2; ++pass) {
        if (current.tris.size() > 35000) {
            break;
        }
        bool anySplit = false;
        std::unordered_map<EdgeKey, std::uint32_t, EdgeKeyHash> edgeMidpoints;
        edgeMidpoints.reserve(current.tris.size() * 2);

        auto getMidpoint = [&](std::uint32_t u, std::uint32_t v) -> std::uint32_t {
            const EdgeKey key(u, v);
            auto it = edgeMidpoints.find(key);
            if (it != edgeMidpoints.end()) {
                return it->second;
            }
            SurfaceVertex mid = lerpSurfaceVertex(current.verts[u], current.verts[v], 0.5f);
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

        for (const auto& tri : current.tris) {
            const std::uint32_t i0 = tri[0];
            const std::uint32_t i1 = tri[1];
            const std::uint32_t i2 = tri[2];

            const float maxTriSdf = std::max({
                current.verts[i0].sdf,
                current.verts[i1].sdf,
                current.verts[i2].sdf,
            });
            const float l01 = lengthSq3(current.verts[i1].pos - current.verts[i0].pos);
            const float l12 = lengthSq3(current.verts[i2].pos - current.verts[i1].pos);
            const float l20 = lengthSq3(current.verts[i0].pos - current.verts[i2].pos);
            const float longestSq = std::max({l01, l12, l20});

            if (maxTriSdf >= shellSdfFloor && longestSq > maxEdgeSq) {
                anySplit = true;
                const std::uint32_t m01 = getMidpoint(i0, i1);
                const std::uint32_t m12 = getMidpoint(i1, i2);
                const std::uint32_t m20 = getMidpoint(i2, i0);
                nextTris.push_back({i0, m01, m20});
                nextTris.push_back({m01, i1, m12});
                nextTris.push_back({m20, m12, i2});
                nextTris.push_back({m01, m12, m20});
            } else {
                nextTris.push_back(tri);
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

/// Extracts iso-contour segments `field(x) = isoValue` across triangles of `mesh`,
/// restricted to points where `validityField(x) >= minValidity`.
/// If a contour segment within a triangle straddles `validityField(x) = minValidity`,
/// it is clipped to the valid sub-segment so open walls terminate cleanly at the exact clearance boundary.
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

    for (const auto& tri : mesh.tris) {
        const std::uint32_t idx[3] = {tri[0], tri[1], tri[2]};
        const float f[3] = {
            fieldAtVert[idx[0]] - isoValue,
            fieldAtVert[idx[1]] - isoValue,
            fieldAtVert[idx[2]] - isoValue,
        };

        // Perturb exact zeros slightly to avoid degenerate vertex-on-contour double crossings
        float fp[3] = {f[0], f[1], f[2]};
        for (int k = 0; k < 3; ++k) {
            if (std::abs(fp[k]) < 1.0e-7f) {
                fp[k] = 1.0e-7f;
            }
        }

        if ((fp[0] > 0.0f && fp[1] > 0.0f && fp[2] > 0.0f) ||
            (fp[0] < 0.0f && fp[1] < 0.0f && fp[2] < 0.0f)) {
            continue;
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
            continue;
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
            continue;
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
            continue;
        }

        Segment3D seg{};
        seg.a = c0.v;
        seg.b = c1.v;
        seg.widthA = c0.width;
        seg.widthB = c1.width;
        segments.push_back(seg);
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

/// Stitches unordered 3D segments into maximal continuous open polylines or closed loops.
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

    const float snapSq = snapToleranceMm * snapToleranceMm;

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
            const Vec3f tail = chain.back().pos;
            std::size_t bestIdx = nSegs;
            bool reverseSeg = false;
            float bestDistSq = snapSq;

            for (std::size_t i = 0; i < nSegs; ++i) {
                if (used[i]) {
                    continue;
                }
                const float dA = lengthSq3(segments[i].a.pos - tail);
                if (dA < bestDistSq) {
                    bestDistSq = dA;
                    bestIdx = i;
                    reverseSeg = false;
                }
                const float dB = lengthSq3(segments[i].b.pos - tail);
                if (dB < bestDistSq) {
                    bestDistSq = dB;
                    bestIdx = i;
                    reverseSeg = true;
                }
            }

            if (bestIdx == nSegs) {
                break;
            }
            used[bestIdx] = true;
            const auto& seg = segments[bestIdx];
            const RawChainPoint nextStart = reverseSeg ? makePtB(seg) : makePtA(seg);
            const RawChainPoint nextEnd = reverseSeg ? makePtA(seg) : makePtB(seg);

            // Average shared endpoint attributes
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
        while (true) {
            const Vec3f head = chain.front().pos;
            std::size_t bestIdx = nSegs;
            bool reverseSeg = false;
            float bestDistSq = snapSq;

            for (std::size_t i = 0; i < nSegs; ++i) {
                if (used[i]) {
                    continue;
                }
                // To prepend to `head`, we want the segment's end (`b`) to match `head`
                const float dB = lengthSq3(segments[i].b.pos - head);
                if (dB < bestDistSq) {
                    bestDistSq = dB;
                    bestIdx = i;
                    reverseSeg = false;
                }
                const float dA = lengthSq3(segments[i].a.pos - head);
                if (dA < bestDistSq) {
                    bestDistSq = dA;
                    bestIdx = i;
                    reverseSeg = true;
                }
            }

            if (bestIdx == nSegs) {
                break;
            }
            used[bestIdx] = true;
            const auto& seg = segments[bestIdx];
            const RawChainPoint prevStart = reverseSeg ? makePtB(seg) : makePtA(seg);
            const RawChainPoint prevEnd = reverseSeg ? makePtA(seg) : makePtB(seg);

            chain.front().pos = lerp3(chain.front().pos, prevEnd.pos, 0.5f);
            chain.front().widthMm = 0.5f * (chain.front().widthMm + prevEnd.widthMm);
            chain.front().heightMm = 0.5f * (chain.front().heightMm + prevEnd.heightMm);
            chain.front().nozzleAxis = normalize3(
                chain.front().nozzleAxis + prevEnd.nozzleAxis,
                chain.front().nozzleAxis);

            if (dist3(prevStart.pos, chain.front().pos) > 1.0e-5f) {
                chain.insert(chain.begin(), prevStart);
            }
        }

        if (chain.size() < 2) {
            continue;
        }

        RawChain rc{};
        const float endDist = dist3(chain.front().pos, chain.back().pos);
        if (chain.size() >= 3 && endDist <= closeLoopToleranceMm) {
            rc.isClosed = true;
            // Weld first and last points if they are nearly identical
            if (endDist <= 0.05f * closeLoopToleranceMm && chain.size() >= 4) {
                chain.front().pos = lerp3(chain.front().pos, chain.back().pos, 0.5f);
                chain.front().widthMm = 0.5f * (chain.front().widthMm + chain.back().widthMm);
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
/// `minExtrusionLengthMm`. Also smooths/orthogonalizes local nozzle frames along the path.
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

    for (std::size_t i = 0; i < raw.pts.size(); ++i) {
        const auto& pt = raw.pts[i];
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
        outPath.points.push_back(tp);
    }

    return true;
}

/// Generates `Monotonic` flood-fill paths on the remaining 3D shell submesh `infillMesh`.
void generateMonotonicInfillOnSubmesh(
    const SurfaceMesh& infillMesh,
    std::uint32_t layerIndex,
    const SanitizedLineWidth& infillWidth,
    float minExtrusionLengthMm,
    ::rust::Vec<ExtrusionPath>& outPaths)
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

    // Alternating 45° / 135° cross-ply angle per layer
    const float baseAngleRad = (layerIndex % 2 == 0)
        ? (0.25f * 3.14159265358979323846f)
        : (0.75f * 3.14159265358979323846f);
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

        // Extract ordered scanlines for each monotonic step m = 0 .. numLines - 1
        struct ScanSegment {
            RawChain chain;
            float vStart = 0.0f;
            float vEnd = 0.0f;
            bool consumed = false;
        };
        std::vector<std::vector<ScanSegment>> scanlines(static_cast<std::size_t>(numLines));

        for (int m = 0; m < numLines; ++m) {
            const float uIso = startU + static_cast<float>(m) * lineWidthMm;
            const auto segs = extractIsoContourSegments(
                islandMesh,
                uCoord,
                uIso,
                allValid,
                0.0f,
                constWidth);
            auto chains = stitchSegmentsIntoChains(segs, 0.05f * lineWidthMm, 0.0f);
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
                        // Connect at the corresponding side (+v if `forwardAlongV`, -v otherwise)
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
                        kFeatureSolidInfill,
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

} // namespace

ObjectToolpathsResult generateShellToolpathsForLayers(
    const SelectedIsosurfacesResult& selectedLayers,
    const openvdb::FloatGrid& sdfGrid,
    const openvdb::FloatGrid& eikGrid,
    const openvdb::Vec3SGrid* toolAxisGrid,
    const std::array<float, 3>& defaultNozzleAxisBed,
    const LayerHeightSpec& layerHeightSpec,
    const ShellPathSpec& spec)
{
    ObjectToolpathsResult result{};
    result.valid = false;
    result.total_path_count = 0;
    result.outer_wall_count = 0;
    result.inner_wall_count = 0;
    result.solid_infill_count = 0;

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

    const float hMin = std::max(0.01f, layerHeightSpec.min_mm);
    const float hMax = std::max(hMin, layerHeightSpec.max_mm);
    const float voxelSize = std::max(0.05f, static_cast<float>(sdfGrid.voxelSize()[0]));
    const Vec3f fallbackNozzle = normalize3(
        {defaultNozzleAxisBed[0], defaultNozzleAxisBed[1], defaultNozzleAxisBed[2]},
        {0.0f, 0.0f, 1.0f});

    const std::size_t numLayers = selectedLayers.layers.size();
    std::vector<LayerToolpaths> perLayerResults(numLayers);

    tbb::parallel_for(
        tbb::blocked_range<std::size_t>(0, numLayers),
        [&](const tbb::blocked_range<std::size_t>& range) {
            auto sdfAcc = sdfGrid.getConstAccessor();
            openvdb::tools::GridSampler<openvdb::FloatGrid::ConstAccessor, openvdb::tools::BoxSampler>
                sdfSampler(sdfAcc, sdfGrid.transform());

            auto eikAcc = eikGrid.getConstAccessor();
            openvdb::tools::GridSampler<openvdb::FloatGrid::ConstAccessor, openvdb::tools::BoxSampler>
                eikSampler(eikAcc, eikGrid.transform());

            auto sampleNozzleAxis = [&](const Vec3f& p) -> Vec3f {
                if (toolAxisGrid != nullptr) {
                    auto tAcc = toolAxisGrid->getConstAccessor();
                    openvdb::tools::GridSampler<openvdb::Vec3SGrid::ConstAccessor, openvdb::tools::BoxSampler>
                        tSampler(tAcc, toolAxisGrid->transform());
                    const openvdb::Vec3s v = tSampler.wsSample(openvdb::Vec3d(p.x, p.y, p.z));
                    const Vec3f raw{v.x(), v.y(), v.z()};
                    if (lengthSq3(raw) > 1.0e-4f) {
                        return normalize3(raw, fallbackNozzle);
                    }
                }
                return fallbackNozzle;
            };

            auto sampleSdf = [&](const Vec3f& p) -> float {
                return sdfSampler.wsSample(openvdb::Vec3d(p.x, p.y, p.z));
            };

            auto sampleEikGradNorm = [&](const Vec3f& p) -> float {
                const float h = 0.5f * voxelSize;
                const float gx = (eikSampler.wsSample(openvdb::Vec3d(p.x + h, p.y, p.z)) -
                                  eikSampler.wsSample(openvdb::Vec3d(p.x - h, p.y, p.z))) / (2.0f * h);
                const float gy = (eikSampler.wsSample(openvdb::Vec3d(p.x, p.y + h, p.z)) -
                                  eikSampler.wsSample(openvdb::Vec3d(p.x, p.y - h, p.z))) / (2.0f * h);
                const float gz = (eikSampler.wsSample(openvdb::Vec3d(p.x, p.y, p.z + h)) -
                                  eikSampler.wsSample(openvdb::Vec3d(p.x, p.y, p.z - h))) / (2.0f * h);
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
                    continue;
                }

                // ------------------------------------------------------------
                // 1. Build & Weld Initial Clipped Isosurface Mesh S_k
                // ------------------------------------------------------------
                SurfaceMesh baseMesh;
                baseMesh.verts.reserve(layerInfo.vertices.size());
                const float meanLayerH = std::max(0.02f, layerInfo.mean_thickness_mm);

                for (const auto& rv : layerInfo.vertices) {
                    SurfaceVertex sv{};
                    sv.pos = {rv.x, rv.y, rv.z};
                    sv.normal = normalize3({rv.nx, rv.ny, rv.nz}, fallbackNozzle);
                    sv.nozzleAxis = sampleNozzleAxis(sv.pos);
                    sv.sdf = sampleSdf(sv.pos);
                    const float gradNorm = sampleEikGradNorm(sv.pos);
                    sv.layerHeightMm = (gradNorm > 0.25f)
                        ? std::clamp(meanLayerH / gradNorm, hMin, hMax)
                        : meanLayerH;
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
                    continue;
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

                // If the entire layer is a small corner cap where boundary edges had slightly lower SDF,
                // mark all boundary edges as outer boundary so the cap still generates its outer perimeter.
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

                // Refine triangles within the shell band so line-width offsets have sub-bead resolution
                const float targetRefEdge = std::clamp(
                    0.55f * std::min({outerW.target_mm, innerW.target_mm, shellThicknessMm}),
                    0.18f,
                    0.45f);
                baseMesh = subdivideShellMesh(
                    baseMesh,
                    targetRefEdge,
                    -shellThicknessMm - 1.5f * voxelSize,
                    [&](const Vec3f& p) {
                        return std::min(-1.0e-4f, sampleSdf(p) - sdfZeroShift);
                    });

                // ------------------------------------------------------------
                // 2. Clip S_k to the Outer Shell Domain (-T_shell <= \phi_sdf <= 0)
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
                    continue;
                }

                // ------------------------------------------------------------
                // 3. Compute On-Surface Geodesic Inset Field & Medial Clearance
                // ------------------------------------------------------------
                const std::size_t nV = shellMesh.verts.size();
                std::vector<std::vector<std::uint32_t>> adj(nV);
                for (const auto& tri : shellMesh.tris) {
                    for (int k = 0; k < 3; ++k) {
                        const std::uint32_t u = tri[k];
                        const std::uint32_t v = tri[(k + 1) % 3];
                        adj[u].push_back(v);
                        adj[v].push_back(u);
                    }
                }

                // Compute inward tangent normal at each outer boundary vertex
                std::vector<Vec3f> outerInwardNormal(nV, {0.0f, 0.0f, 0.0f});
                for (const auto& tri : shellMesh.tris) {
                    for (int k = 0; k < 3; ++k) {
                        const std::uint32_t u = tri[k];
                        const std::uint32_t v = tri[(k + 1) % 3];
                        const std::uint32_t w = tri[(k + 2) % 3];
                        if (shellMesh.verts[u].onOuterBoundary && shellMesh.verts[v].onOuterBoundary) {
                            const Vec3f edgeDir = normalize3(shellMesh.verts[v].pos - shellMesh.verts[u].pos);
                            const Vec3f toInterior = shellMesh.verts[w].pos - shellMesh.verts[u].pos;
                            const Vec3f inward = normalize3(
                                toInterior - (edgeDir * dot3(toInterior, edgeDir)),
                                {0.0f, 0.0f, 0.0f});
                            outerInwardNormal[u] = outerInwardNormal[u] + inward;
                            outerInwardNormal[v] = outerInwardNormal[v] + inward;
                        }
                    }
                }
                for (std::size_t i = 0; i < nV; ++i) {
                    if (shellMesh.verts[i].onOuterBoundary) {
                        outerInwardNormal[i] = normalize3(outerInwardNormal[i], {1.0f, 0.0f, 0.0f});
                    }
                }

                // Multi-source Dijkstra from \Gamma_outer (\phi_sdf = 0) across \mathcal{M}_{shell, k}
                const float kInf = 1.0e9f;
                std::vector<float> dOuter(nV, kInf);
                std::vector<std::uint32_t> closestOuterSrc(nV, 0);

                using QueueItem = std::pair<float, std::uint32_t>;
                std::priority_queue<QueueItem, std::vector<QueueItem>, std::greater<QueueItem>> pq;

                std::vector<std::uint32_t> outerBoundaryVerts;
                outerBoundaryVerts.reserve(nV / 4 + 8);

                for (std::size_t i = 0; i < nV; ++i) {
                    if (shellMesh.verts[i].onOuterBoundary) {
                        dOuter[i] = 0.0f;
                        closestOuterSrc[i] = static_cast<std::uint32_t>(i);
                        pq.push({0.0f, static_cast<std::uint32_t>(i)});
                        outerBoundaryVerts.push_back(static_cast<std::uint32_t>(i));
                    }
                }

                if (outerBoundaryVerts.empty()) {
                    perLayerResults[layerIdx] = std::move(layerOut);
                    continue;
                }

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
                            closestOuterSrc[v] = closestOuterSrc[u];
                            pq.push({dOuter[v], v});
                        }
                    }
                }

                // Smooth `dOuter` slightly with the continuous 3D SDF depth (scaled by surface gradient)
                // so iso-contours of `dOuter` are smooth sub-voxel curves without piecewise graph jaggedness
                std::vector<float> dOuterSmooth = dOuter;
                for (int iter = 0; iter < 2; ++iter) {
                    std::vector<float> nextD = dOuterSmooth;
                    for (std::size_t i = 0; i < nV; ++i) {
                        if (shellMesh.verts[i].onOuterBoundary || adj[i].empty()) {
                            continue;
                        }
                        float sumNb = 0.0f;
                        for (std::uint32_t nb : adj[i]) {
                            sumNb += dOuterSmooth[nb];
                        }
                        const float avgNb = sumNb / static_cast<float>(adj[i].size());
                        nextD[i] = 0.70f * dOuterSmooth[i] + 0.30f * avgNb;
                    }
                    dOuterSmooth = std::move(nextD);
                }
                dOuter = std::move(dOuterSmooth);

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

                // Compute medial-axis half-span R_medial(v) from opposing outer boundary segments
                std::vector<float> rMedial(nV, shellThicknessMm * 4.0f);
                const std::size_t bStride = std::max<std::size_t>(1, outerBoundaryVerts.size() / 160);

                for (std::size_t i = 0; i < nV; ++i) {
                    const std::uint32_t srcIdx = closestOuterSrc[i];
                    const Vec3f srcPos = shellMesh.verts[srcIdx].pos;
                    const Vec3f srcNormal = outerInwardNormal[srcIdx];
                    const float d0 = dOuter[i];

                    float minOppDist = kInf;
                    for (std::size_t b = 0; b < outerBoundaryVerts.size(); b += bStride) {
                        const std::uint32_t bIdx = outerBoundaryVerts[b];
                        const Vec3f bPos = shellMesh.verts[bIdx].pos;
                        const Vec3f bNormal = outerInwardNormal[bIdx];

                        // Opposing wall criterion: either inward normals oppose (`dot < -0.15`)
                        // or boundary chord distance is significantly larger than direct distance
                        const float nDot = dot3(srcNormal, bNormal);
                        const float chordDist = dist3(srcPos, bPos);
                        const float dToB = dist3(shellMesh.verts[i].pos, bPos);

                        if (nDot < -0.15f || chordDist > 1.65f * (d0 + dToB)) {
                            minOppDist = std::min(minOppDist, dToB);
                        }
                    }

                    if (minOppDist < kInf * 0.5f) {
                        rMedial[i] = 0.5f * (d0 + minOppDist);
                    }
                }

                // Also detect medial shock edges directly on the mesh where adjacent vertices
                // originate from opposing outer boundary normals
                for (std::size_t i = 0; i < nV; ++i) {
                    const std::uint32_t sI = closestOuterSrc[i];
                    for (std::uint32_t nb : adj[i]) {
                        const std::uint32_t sN = closestOuterSrc[nb];
                        if (dot3(outerInwardNormal[sI], outerInwardNormal[sN]) < -0.15f) {
                            const float edgeLen = dist3(shellMesh.verts[i].pos, shellMesh.verts[nb].pos);
                            const float shockHalfSpan = 0.5f * (dOuter[i] + dOuter[nb] + edgeLen);
                            rMedial[i] = std::min(rMedial[i], shockHalfSpan);
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

                // Total available one-sided shell span B(v) at each vertex
                std::vector<float> spanShell(nV, shellThicknessMm);
                std::vector<float> availableSpan(nV, shellThicknessMm);
                for (std::size_t i = 0; i < nV; ++i) {
                    const float sShell = hasInnerShellBoundary
                        ? std::max(dOuter[i] + dInner[i], shellThicknessMm)
                        : std::max(dOuter[i], shellThicknessMm);
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
                        // Even a minimum-width outer wall barely fits: clamp to min_mm
                        localOuterWidth[i] = outerW.min_mm;
                        localInnerWidth[i] = innerW.min_mm;
                        activeInnerCount[i] = 0;
                        totalWallSpan[i] = (B >= 0.5f * outerW.min_mm) ? outerW.min_mm : 0.0f;
                        continue;
                    }

                    // Determine maximum number of inner walls m in [0..numInnerWalls] that fit in B
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
                        // Compress outer & inner walls smoothly between [min_mm, target_mm] to fit B
                        const float denom = std::max(1.0e-5f, wTargetSum - wMinSum);
                        const float t = std::clamp((B - wMinSum) / denom, 0.0f, 1.0f);
                        localOuterWidth[i] =
                            outerW.min_mm + t * (outerW.target_mm - outerW.min_mm);
                        localInnerWidth[i] =
                            innerW.min_mm + t * (innerW.target_mm - innerW.min_mm);
                    } else {
                        // B >= wTargetSum: check if there is a narrow unfillable medial residual gap
                        const float remGap = (rMedial[i] <= spanShell[i])
                            ? 2.0f * (rMedial[i] - wTargetSum)
                            : (B - wTargetSum);
                        if (remGap > 0.0f && remGap < infillW.min_mm && B <= wMaxSum) {
                            // Expand walls up toward max_mm so no sub-minimum gap is left behind
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
                const float stitchSnap = std::max(0.04f, 0.25f * targetRefEdge);
                const float closeLoopTol = std::max(0.15f, 0.65f * outerW.target_mm);

                // 5a. Outer wall: centerline at d_outer(x) = 0.5 * w_outer(x)
                {
                    std::vector<float> outerLevel(nV, 0.0f);
                    for (std::size_t i = 0; i < nV; ++i) {
                        outerLevel[i] = dOuter[i] - 0.5f * localOuterWidth[i];
                    }
                    const auto outerSegs = extractIsoContourSegments(
                        shellMesh,
                        outerLevel,
                        0.0f,
                        availableSpan,
                        0.5f * outerW.min_mm,
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
                // Signed distance inward from the inner edge of the wall stack:
                // dRem(v) = dOuter(v) - totalWallSpan(v) >= 0 inside the remaining shell region
                std::vector<float> negDRem(nV, 0.0f);
                std::vector<float> remOneSidedSpan(nV, 0.0f);
                for (std::size_t i = 0; i < nV; ++i) {
                    const float dRem = dOuter[i] - totalWallSpan[i];
                    negDRem[i] = -dRem; // <= 0 where dRem >= 0
                    remOneSidedSpan[i] = std::max(0.0f, availableSpan[i] - totalWallSpan[i]);
                    shellMesh.verts[i].dRem = std::max(0.0f, dRem);
                    shellMesh.verts[i].remSpan = remOneSidedSpan[i];
                }

                // 6a. Gap Fill in narrow residual strips/medial channels where a single variable-width
                //     bead in [infillW.min_mm, infillW.max_mm] fills the remaining gap
                {
                    std::vector<float> gapLevel(nV, 0.0f);
                    std::vector<float> gapValidity(nV, -1.0f);
                    std::vector<float> gapWidth(nV, infillW.target_mm);

                    for (std::size_t i = 0; i < nV; ++i) {
                        const bool isMedialLimited = (rMedial[i] < spanShell[i] - 0.05f);
                        const float fullGapWidth = isMedialLimited
                            ? 2.0f * remOneSidedSpan[i]
                            : remOneSidedSpan[i];
                        const float gapCenterOffset = isMedialLimited
                            ? remOneSidedSpan[i]
                            : 0.5f * remOneSidedSpan[i];

                        gapLevel[i] = (dOuter[i] - totalWallSpan[i]) - gapCenterOffset;
                        if (fullGapWidth >= infillW.min_mm && fullGapWidth <= infillW.max_mm) {
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

                // 6b. Solid Infill in the remaining shell region where `dRem >= 0` and
                //     remaining span exceeds single-bead gap fill (or fills the shell interior)
                for (std::size_t i = 0; i < nV; ++i) {
                    const bool isMedialLimited = (rMedial[i] < spanShell[i] - 0.05f);
                    const float fullRemWidth = isMedialLimited
                        ? 2.0f * remOneSidedSpan[i]
                        : remOneSidedSpan[i];
                    // Exclude regions already handled by a single gap-fill bead
                    if (isMedialLimited && fullRemWidth <= infillW.max_mm) {
                        negDRem[i] = 1.0f;
                    }
                }

                SurfaceMesh infillSubmesh = clipSurfaceMeshByScalar(
                    shellMesh,
                    negDRem,
                    0.0f,
                    false);

                if (!infillSubmesh.tris.empty() && !infillSubmesh.verts.empty()) {
                    if (infillPattern == kPatternConcentric) {
                        // Concentric solid infill: continued inward offsetting from the innermost wall
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

                        const int maxPasses = std::max(
                            1,
                            static_cast<int>(std::ceil(maxRemSpan / infillW.min_mm)));
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
                        // Monotonic flood fill across the 3D curved remaining shell submesh
                        generateMonotonicInfillOnSubmesh(
                            infillSubmesh,
                            layerInfo.layer_index,
                            infillW,
                            minExtrusionLengthMm,
                            layerOut.paths);
                    }
                }

                perLayerResults[layerIdx] = std::move(layerOut);
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
            } else {
                result.solid_infill_count += 1;
            }
        }
        result.layers.push_back(std::move(layer));
    }

    return result;
}

} // namespace manifold
