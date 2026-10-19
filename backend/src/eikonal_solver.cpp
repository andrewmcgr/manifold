#include "slicer_backend.h"
#include "manifold-frontend/src/bridge.rs.h"
#include "eikonal_solver.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <queue>
#include <string>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

namespace manifold {

namespace {

constexpr float kInfinityTau = 1.0e6f;
constexpr float kPi = 3.14159265358979323846f;
constexpr float kDegToRad = kPi / 180.0f;
constexpr float kRadToDeg = 180.0f / kPi;
constexpr std::size_t kMinParallelWavefrontSize = 64;

inline float dot3(const std::array<float, 3>& a, const std::array<float, 3>& b) noexcept {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

inline std::array<float, 3> cross3(
    const std::array<float, 3>& a,
    const std::array<float, 3>& b) noexcept
{
    return {
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]};
}

inline float length3(const std::array<float, 3>& v) noexcept {
    return std::sqrt(dot3(v, v));
}

inline std::array<float, 3> normalize3(
    const std::array<float, 3>& v,
    const std::array<float, 3>& fallback = {0.0f, 0.0f, 1.0f}) noexcept
{
    const float len = length3(v);
    if (len > 1.0e-7f) {
        return {v[0] / len, v[1] / len, v[2] / len};
    }
    return fallback;
}

/// Clamps unit vector `u` onto the spherical cone of half-angle `maxAngleRad` around unit axis `refAxis`.
std::array<float, 3> clampUnitVectorToCone(
    const std::array<float, 3>& u,
    const std::array<float, 3>& refAxis,
    float maxAngleRad) noexcept
{
    const float clampedMax = std::clamp(maxAngleRad, 0.0f, 0.495f * kPi);
    if (clampedMax <= 1.0e-6f) {
        return refAxis;
    }

    const float cosTheta = std::clamp(dot3(u, refAxis), -1.0f, 1.0f);
    const float cosMax = std::cos(clampedMax);
    if (cosTheta >= cosMax) {
        return u;
    }

    const std::array<float, 3> perpRaw{
        u[0] - cosTheta * refAxis[0],
        u[1] - cosTheta * refAxis[1],
        u[2] - cosTheta * refAxis[2]};
    const float perpLen = length3(perpRaw);
    if (perpLen <= 1.0e-6f) {
        return refAxis;
    }

    const float sinMax = std::sin(clampedMax);
    const std::array<float, 3> clamped{
        cosMax * refAxis[0] + (sinMax / perpLen) * perpRaw[0],
        cosMax * refAxis[1] + (sinMax / perpLen) * perpRaw[1],
        cosMax * refAxis[2] + (sinMax / perpLen) * perpRaw[2]};
    return normalize3(clamped, refAxis);
}

/// Computes the maximum tilt angle \gamma (radians) from the bed normal at height `hBedMm`
/// above the bed so that none of the XZ clearance profile stages (r_m, z_m) intersect the bed:
///   hBedMm + z_m * cos(\gamma) - r_m * sin(\gamma) >= 0.
float computeBedSafeTiltRad(
    const ToolSpec& tool,
    float hBedMm,
    float userMaxAngleRad) noexcept
{
    float safeMax = std::clamp(userMaxAngleRad, 0.0f, 85.0f * kDegToRad);
    if (safeMax <= 1.0e-5f || tool.clearance_profile_xz.empty()) {
        return safeMax;
    }

    const float h = std::max(0.0f, hBedMm);
    for (const auto& pt : tool.clearance_profile_xz) {
        const float r = std::max(0.0f, pt.x_radius_mm);
        const float z = std::max(0.0f, pt.z_height_mm);
        if (r <= 1.0e-4f) {
            continue;
        }
        const float hypotRZ = std::hypot(r, z);
        const float phi = std::atan2(z, r);
        const float ratio = std::clamp(h / std::max(hypotRZ, 1.0e-5f), 0.0f, 1.0f);
        const float limitRad = phi + std::asin(ratio);
        safeMax = std::min(safeMax, limitRad);
    }

    return std::max(0.0f, safeMax);
}

/// Computes the central-difference gradient of `domain.tau` at active voxel `idx`.
std::array<float, 3> computeVoxelTauGradient(
    const ActiveVolumeDomain& domain,
    std::size_t idx) noexcept
{
    const float dx = static_cast<float>(domain.voxelSize);
    const float centerTau = domain.tau[idx];
    const auto& n6 = domain.neighbors6[idx];

    auto axisDeriv = [&](std::int32_t negIdx, std::int32_t posIdx) -> float {
        const bool hasNeg = (negIdx >= 0 && domain.tau[static_cast<std::size_t>(negIdx)] < 0.5f * kInfinityTau);
        const bool hasPos = (posIdx >= 0 && domain.tau[static_cast<std::size_t>(posIdx)] < 0.5f * kInfinityTau);
        if (hasNeg && hasPos) {
            return (domain.tau[static_cast<std::size_t>(posIdx)] -
                    domain.tau[static_cast<std::size_t>(negIdx)]) /
                   (2.0f * dx);
        }
        if (hasPos) {
            return (domain.tau[static_cast<std::size_t>(posIdx)] - centerTau) / dx;
        }
        if (hasNeg) {
            return (centerTau - domain.tau[static_cast<std::size_t>(negIdx)]) / dx;
        }
        return 0.0f;
    };

    return {
        axisDeriv(n6[0], n6[1]),
        axisDeriv(n6[2], n6[3]),
        axisDeriv(n6[4], n6[5])};
}

/// Solves the 3D Godunov upwind quadratic update at voxel `idx` for |\nabla \tau| = s(x),
/// optionally blending a directional upwind step along `-domain.targetDir[idx]`.
float godunovUpwindUpdate(
    const ActiveVolumeDomain& domain,
    std::size_t idx,
    bool useDirectionalGuidance) noexcept
{
    const float dx = static_cast<float>(domain.voxelSize);
    const float stepCost = std::max(0.15f, domain.slowness[idx]) * dx;
    const auto& n6 = domain.neighbors6[idx];

    auto neighborTau = [&](std::int32_t nIdx) -> float {
        if (nIdx < 0) {
            return kInfinityTau;
        }
        const std::size_t u = static_cast<std::size_t>(nIdx);
        if (domain.isInterior[idx] && !domain.isInterior[u]) {
            return kInfinityTau;
        }
        return domain.tau[u];
    };

    const float ax = std::min(neighborTau(n6[0]), neighborTau(n6[1]));
    const float ay = std::min(neighborTau(n6[2]), neighborTau(n6[3]));
    const float az = std::min(neighborTau(n6[4]), neighborTau(n6[5]));

    std::array<float, 3> a{ax, ay, az};
    std::sort(a.begin(), a.end());

    if (a[0] >= 0.5f * kInfinityTau) {
        return domain.tau[idx];
    }

    // 1. 1D upwind candidate
    float u = a[0] + stepCost;
    if (u > a[1]) {
        // 2. 2D quadratic upwind candidate
        const float diff12 = a[0] - a[1];
        const float disc2 = 2.0f * stepCost * stepCost - diff12 * diff12;
        if (disc2 >= 0.0f) {
            u = 0.5f * (a[0] + a[1] + std::sqrt(disc2));
        }
        if (u > a[2]) {
            // 3. 3D quadratic upwind candidate
            const float sum3 = a[0] + a[1] + a[2];
            const float sqSum3 = a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
            const float disc3 = sum3 * sum3 - 3.0f * (sqSum3 - stepCost * stepCost);
            if (disc3 >= 0.0f) {
                u = (sum3 + std::sqrt(disc3)) / 3.0f;
            }
        }
    }

    if (!useDirectionalGuidance) {
        return u;
    }

    // Directional upwind estimate along -targetDir[idx]
    const auto& d = domain.targetDir[idx];
    float weightSum = 0.0f;
    float upwindTauSum = 0.0f;

    const std::int32_t xUp = (d[0] >= 0.0f) ? n6[0] : n6[1];
    const std::int32_t yUp = (d[1] >= 0.0f) ? n6[2] : n6[3];
    const std::int32_t zUp = (d[2] >= 0.0f) ? n6[4] : n6[5];

    const float wx = std::abs(d[0]);
    const float wy = std::abs(d[1]);
    const float wz = std::abs(d[2]);

    const float tx = neighborTau(xUp);
    const float ty = neighborTau(yUp);
    const float tz = neighborTau(zUp);

    if (wx > 1.0e-4f && tx < 0.5f * kInfinityTau) {
        weightSum += wx;
        upwindTauSum += wx * tx;
    }
    if (wy > 1.0e-4f && ty < 0.5f * kInfinityTau) {
        weightSum += wy;
        upwindTauSum += wy * ty;
    }
    if (wz > 1.0e-4f && tz < 0.5f * kInfinityTau) {
        weightSum += wz;
        upwindTauSum += wz * tz;
    }

    if (weightSum > 1.0e-4f) {
        const float l1Norm = wx + wy + wz;
        const float dirStep = stepCost / std::max(l1Norm, 1.0f);
        const float uDir = (upwindTauSum / weightSum) + dirStep;
        return 0.70f * u + 0.30f * uDir;
    }

    return u;
}

/// Runs multi-threaded 8-octant Gauss-Seidel Fast Sweeping passes over all interior voxels
/// using the precomputed diagonal hyperplane wavefront schedules (Detrixhe et al. 2013).
/// Within each hyperplane level L = s_x i + s_y j + s_z k, no two voxels are 6-neighbors,
/// so every hyperplane slice executes in parallel via `tbb::parallel_for` with zero data races.
void runInteriorFastSweeping(
    ActiveVolumeDomain& domain,
    std::uint32_t numSweepRounds,
    bool useDirectionalGuidance)
{
    auto sweepWavefrontSlice = [&](const WavefrontSchedule& wf, std::size_t lvl) {
        const std::size_t beginOff = wf.levelOffsets[lvl];
        const std::size_t endOff = wf.levelOffsets[lvl + 1];
        const std::size_t count = endOff - beginOff;

        if (count >= kMinParallelWavefrontSize) {
            tbb::parallel_for(
                tbb::blocked_range<std::size_t>(beginOff, endOff, kMinParallelWavefrontSize),
                [&](const tbb::blocked_range<std::size_t>& r) {
                    for (std::size_t pos = r.begin(); pos != r.end(); ++pos) {
                        const std::size_t idx = wf.indices[pos];
                        const float cand = godunovUpwindUpdate(domain, idx, useDirectionalGuidance);
                        if (cand < domain.tau[idx]) {
                            domain.tau[idx] = cand;
                        }
                    }
                });
        } else {
            for (std::size_t pos = beginOff; pos < endOff; ++pos) {
                const std::size_t idx = wf.indices[pos];
                const float cand = godunovUpwindUpdate(domain, idx, useDirectionalGuidance);
                if (cand < domain.tau[idx]) {
                    domain.tau[idx] = cand;
                }
            }
        }
    };

    for (std::uint32_t round = 0; round < numSweepRounds; ++round) {
        for (const auto& wf : domain.octantWavefronts) {
            const std::size_t numLvls = wf.numLevels();
            if (numLvls == 0) {
                continue;
            }
            // Forward octant sweep (+Z hyperplane progression)
            for (std::size_t lvl = 0; lvl < numLvls; ++lvl) {
                sweepWavefrontSlice(wf, lvl);
            }
            // Reverse octant sweep (-Z hyperplane progression)
            for (std::size_t lvl = numLvls; lvl-- > 0;) {
                sweepWavefrontSlice(wf, lvl);
            }
        }
    }
}

/// Extends the solved interior Eikonal field smoothly into the exterior narrow-band voxels
/// (\phi_sdf > 0.75 * dx) in parallel via `tbb::parallel_for` so that `openvdb::tools::volumeToMesh`
/// extracts clean, continuous isosurface sheets across the object boundary (\phi_sdf = 0).
void extendEikonalToExteriorBand(
    ActiveVolumeDomain& domain,
    const SlicerBackend& backend)
{
    const std::size_t n = domain.coords.size();
    std::vector<std::size_t> extIndices;
    extIndices.reserve(n / 2);
    for (std::size_t i = 0; i < n; ++i) {
        if (!domain.isInterior[i]) {
            extIndices.push_back(i);
        }
    }

    const std::size_t extCount = extIndices.size();
    if (extCount == 0) {
        return;
    }

    const float dx = static_cast<float>(domain.voxelSize);
    const std::array<std::array<float, 3>, 6> stepVecs{{
        {dx, 0.0f, 0.0f},
        {-dx, 0.0f, 0.0f},
        {0.0f, dx, 0.0f},
        {0.0f, -dx, 0.0f},
        {0.0f, 0.0f, dx},
        {0.0f, 0.0f, -dx},
    }};

    std::vector<float> nextTau = domain.tau;

    for (int pass = 0; pass < 10; ++pass) {
        tbb::parallel_for(
            tbb::blocked_range<std::size_t>(0, extCount, 256),
            [&](const tbb::blocked_range<std::size_t>& r) {
                for (std::size_t e = r.begin(); e != r.end(); ++e) {
                    const std::size_t idx = extIndices[e];
                    const auto& n6 = domain.neighbors6[idx];
                    float sumWeightedTau = 0.0f;
                    float sumWeight = 0.0f;

                    for (int k = 0; k < 6; ++k) {
                        const std::int32_t nb = n6[static_cast<std::size_t>(k)];
                        if (nb < 0) {
                            continue;
                        }
                        const std::size_t u = static_cast<std::size_t>(nb);
                        const float nbTau = domain.tau[u];
                        if (nbTau >= 0.5f * kInfinityTau) {
                            continue;
                        }
                        const float w = (domain.sdf[u] < domain.sdf[idx]) ? 2.5f : 1.0f;
                        const float deltaTau = dot3(domain.toolAxis[u], stepVecs[static_cast<std::size_t>(k)]);
                        sumWeightedTau += w * std::max(0.0f, nbTau + deltaTau);
                        sumWeight += w;
                    }

                    if (sumWeight > 0.0f) {
                        nextTau[idx] = sumWeightedTau / sumWeight;
                    } else if (pass == 0) {
                        const NozzleBasis basis =
                            backend.queryNozzleFrame(domain.posBed[idx], {1.0f, 0.0f, 0.0f});
                        const std::array<float, 3> rel{
                            domain.posBed[idx][0],
                            domain.posBed[idx][1],
                            domain.posBed[idx][2] - domain.minZBed};
                        nextTau[idx] = std::max(0.0f, dot3(rel, basis.nozzle_axis));
                    } else {
                        nextTau[idx] = domain.tau[idx];
                    }
                }
            });

        tbb::parallel_for(
            tbb::blocked_range<std::size_t>(0, extCount, 512),
            [&](const tbb::blocked_range<std::size_t>& r) {
                for (std::size_t e = r.begin(); e != r.end(); ++e) {
                    const std::size_t idx = extIndices[e];
                    domain.tau[idx] = nextTau[idx];
                }
            });
    }
}

} // namespace

// ============================================================================
// ActiveVolumeDomain Implementation
// ============================================================================

std::int32_t ActiveVolumeDomain::findVoxelIndex(const openvdb::Coord& ijk) const {
    if (!indexGrid) {
        return -1;
    }
    return indexGrid->tree().getValue(ijk);
}

void ActiveVolumeDomain::initializeFromSdfGrid(
    const openvdb::FloatGrid& sdfGrid,
    float objectMinZ,
    float objectMaxZ,
    float objectMaxSpan)
{
    voxelSize = sdfGrid.transform().voxelSize()[0];
    minZBed = objectMinZ;
    maxZBed = objectMaxZ;
    maxSpanMm = std::max(objectMaxSpan, 1.0f);
    transform = sdfGrid.transform().copy();

    const std::size_t activeCount = sdfGrid.activeVoxelCount();
    coords.clear();
    posBed.clear();
    sdf.clear();
    sdfGrad.clear();
    overhang.clear();
    tau.clear();
    slowness.clear();
    targetDir.clear();
    toolAxis.clear();
    toolTiltDeg.clear();
    isSeed.clear();
    isInterior.clear();
    neighbors6.clear();

    coords.reserve(activeCount);
    posBed.reserve(activeCount);
    sdf.reserve(activeCount);

    indexGrid = openvdb::Int32Grid::create(-1);
    indexGrid->setTransform(transform->copy());
    auto idxAccessor = indexGrid->getAccessor();

    for (auto iter = sdfGrid.cbeginValueOn(); iter; ++iter) {
        const openvdb::Coord ijk = iter.getCoord();
        const std::int32_t linearIdx = static_cast<std::int32_t>(coords.size());
        coords.push_back(ijk);
        sdf.push_back(*iter);

        const openvdb::Vec3d wp = transform->indexToWorld(ijk);
        posBed.push_back({
            static_cast<float>(wp.x()),
            static_cast<float>(wp.y()),
            static_cast<float>(wp.z())});
        idxAccessor.setValue(ijk, linearIdx);
    }

    const std::size_t n = coords.size();
    sdfGrad.assign(n, {0.0f, 0.0f, 1.0f});
    overhang.assign(n, 0.0f);
    tau.assign(n, kInfinityTau);
    slowness.assign(n, 1.0f);
    targetDir.assign(n, {0.0f, 0.0f, 1.0f});
    toolAxis.assign(n, {0.0f, 0.0f, 1.0f});
    toolTiltDeg.assign(n, 0.0f);
    isSeed.assign(n, 0);
    isInterior.assign(n, 0);
    neighbors6.resize(n);

    // Parallelize 6-neighbor lookup and central-difference SDF gradient computation via TBB
    const float invTwoDx = 1.0f / static_cast<float>(2.0 * voxelSize);

    tbb::parallel_for(
        tbb::blocked_range<std::size_t>(0, n, 512),
        [&](const tbb::blocked_range<std::size_t>& r) {
            auto localIdxAcc = indexGrid->getConstAccessor();
            auto localSdfAcc = sdfGrid.getConstAccessor();

            for (std::size_t i = r.begin(); i != r.end(); ++i) {
                const openvdb::Coord& c = coords[i];
                neighbors6[i] = {
                    localIdxAcc.getValue(c.offsetBy(-1, 0, 0)),
                    localIdxAcc.getValue(c.offsetBy(1, 0, 0)),
                    localIdxAcc.getValue(c.offsetBy(0, -1, 0)),
                    localIdxAcc.getValue(c.offsetBy(0, 1, 0)),
                    localIdxAcc.getValue(c.offsetBy(0, 0, -1)),
                    localIdxAcc.getValue(c.offsetBy(0, 0, 1))};

                const float gx =
                    (localSdfAcc.getValue(c.offsetBy(1, 0, 0)) -
                     localSdfAcc.getValue(c.offsetBy(-1, 0, 0))) *
                    invTwoDx;
                const float gy =
                    (localSdfAcc.getValue(c.offsetBy(0, 1, 0)) -
                     localSdfAcc.getValue(c.offsetBy(0, -1, 0))) *
                    invTwoDx;
                const float gz =
                    (localSdfAcc.getValue(c.offsetBy(0, 0, 1)) -
                     localSdfAcc.getValue(c.offsetBy(0, 0, -1))) *
                    invTwoDx;
                sdfGrad[i] = normalize3({gx, gy, gz}, {0.0f, 0.0f, 1.0f});
            }
        });
}

void ActiveVolumeDomain::buildWavefrontSchedules() {
    const std::size_t n = coords.size();
    std::vector<std::size_t> activeInterior;
    activeInterior.reserve(n);

    std::int32_t minK = std::numeric_limits<std::int32_t>::max();
    std::int32_t maxK = std::numeric_limits<std::int32_t>::lowest();

    for (std::size_t i = 0; i < n; ++i) {
        if (isInterior[i] && !isSeed[i]) {
            activeInterior.push_back(i);
            minK = std::min(minK, coords[i].z());
            maxK = std::max(maxK, coords[i].z());
        }
    }

    const std::size_t m = activeInterior.size();
    if (m == 0) {
        for (auto& wf : octantWavefronts) {
            wf.indices.clear();
            wf.levelOffsets.clear();
        }
        zParityWavefront.indices.clear();
        zParityWavefront.levelOffsets.clear();
        return;
    }

    // 1. Build the 4 diagonal hyperplane schedules in O(m) counting-sort time
    const std::array<std::pair<int, int>, 4> signs{{
        {1, 1},
        {-1, 1},
        {1, -1},
        {-1, -1},
    }};

    for (std::size_t q = 0; q < 4; ++q) {
        const int sx = signs[q].first;
        const int sy = signs[q].second;

        std::int32_t minL = std::numeric_limits<std::int32_t>::max();
        std::int32_t maxL = std::numeric_limits<std::int32_t>::lowest();
        for (std::size_t idx : activeInterior) {
            const auto& c = coords[idx];
            const std::int32_t lvl = sx * c.x() + sy * c.y() + c.z();
            minL = std::min(minL, lvl);
            maxL = std::max(maxL, lvl);
        }

        const std::size_t numBuckets = static_cast<std::size_t>(maxL - minL + 1);
        std::vector<std::size_t> counts(numBuckets + 1, 0);
        for (std::size_t idx : activeInterior) {
            const auto& c = coords[idx];
            const std::size_t b = static_cast<std::size_t>((sx * c.x() + sy * c.y() + c.z()) - minL);
            ++counts[b + 1];
        }

        for (std::size_t b = 0; b < numBuckets; ++b) {
            counts[b + 1] += counts[b];
        }

        auto& wf = octantWavefronts[q];
        wf.levelOffsets = counts;
        wf.indices.resize(m);
        std::vector<std::size_t> writeCursor = counts;

        for (std::size_t idx : activeInterior) {
            const auto& c = coords[idx];
            const std::size_t b = static_cast<std::size_t>((sx * c.x() + sy * c.y() + c.z()) - minL);
            wf.indices[writeCursor[b]++] = idx;
        }
    }

    // 2. Build the Z-slice Red-Black schedule in O(m) counting-sort time
    const std::size_t numZParityBuckets = static_cast<std::size_t>((maxK - minK + 1) * 2);
    std::vector<std::size_t> zCounts(numZParityBuckets + 1, 0);
    for (std::size_t idx : activeInterior) {
        const auto& c = coords[idx];
        const std::size_t parity = static_cast<std::size_t>((c.x() + c.y() + c.z()) & 1);
        const std::size_t b = static_cast<std::size_t>(c.z() - minK) * 2 + parity;
        ++zCounts[b + 1];
    }
    for (std::size_t b = 0; b < numZParityBuckets; ++b) {
        zCounts[b + 1] += zCounts[b];
    }

    zParityWavefront.levelOffsets = zCounts;
    zParityWavefront.indices.resize(m);
    std::vector<std::size_t> zWriteCursor = zCounts;
    for (std::size_t idx : activeInterior) {
        const auto& c = coords[idx];
        const std::size_t parity = static_cast<std::size_t>((c.x() + c.y() + c.z()) & 1);
        const std::size_t b = static_cast<std::size_t>(c.z() - minK) * 2 + parity;
        zParityWavefront.indices[zWriteCursor[b]++] = idx;
    }
}

// ============================================================================
// ToolOrientationStrategy Implementation (Parallelized via TBB)
// ============================================================================

void ToolOrientationStrategy::updateToolOrientations(
    ActiveVolumeDomain& domain,
    const EikonalSolverContext& ctx) const
{
    const std::size_t n = domain.coords.size();
    if (n == 0) {
        return;
    }

    const bool rotational = ctx.activeTool.has_rotational_axes;
    const float maxAngleRad =
        std::clamp(ctx.activeTool.max_angle_from_bed_normal_deg, 0.0f, 89.0f) * kDegToRad;

    if (!rotational || maxAngleRad <= 1.0e-5f) {
        tbb::parallel_for(
            tbb::blocked_range<std::size_t>(0, n, 512),
            [&](const tbb::blocked_range<std::size_t>& r) {
                for (std::size_t i = r.begin(); i != r.end(); ++i) {
                    const NozzleBasis frame =
                        ctx.backend.queryNozzleFrame(domain.posBed[i], {1.0f, 0.0f, 0.0f});
                    domain.toolAxis[i] = frame.nozzle_axis;
                    domain.toolTiltDeg[i] = 0.0f;
                }
            });
        return;
    }

    std::vector<std::array<float, 3>> clampedAxes(n);
    std::vector<float> localMaxTiltRad(n, 0.0f);

    tbb::parallel_for(
        tbb::blocked_range<std::size_t>(0, n, 256),
        [&](const tbb::blocked_range<std::size_t>& r) {
            for (std::size_t i = r.begin(); i != r.end(); ++i) {
                const NozzleBasis baseFrame =
                    ctx.backend.queryNozzleFrame(domain.posBed[i], {1.0f, 0.0f, 0.0f});
                const std::array<float, 3> bedNormal = baseFrame.nozzle_axis;

                const std::array<float, 3> rel{
                    domain.posBed[i][0],
                    domain.posBed[i][1],
                    domain.posBed[i][2] - domain.minZBed};
                const float hBedMm = std::max(0.0f, dot3(rel, bedNormal));
                const float gammaMaxRad =
                    computeBedSafeTiltRad(ctx.activeTool, hBedMm, maxAngleRad);
                localMaxTiltRad[i] = gammaMaxRad;

                std::array<float, 3> desiredAxis = bedNormal;
                if (domain.isInterior[i]) {
                    const auto gradTau = computeVoxelTauGradient(domain, i);
                    const float gradLen = length3(gradTau);
                    if (gradLen > 1.0e-5f) {
                        desiredAxis = {
                            gradTau[0] / gradLen,
                            gradTau[1] / gradLen,
                            gradTau[2] / gradLen};
                        if (dot3(desiredAxis, bedNormal) < 0.0f) {
                            desiredAxis = {-desiredAxis[0], -desiredAxis[1], -desiredAxis[2]};
                        }
                    }

                    const float surfDist = std::abs(domain.sdf[i]);
                    const float bandMm = 3.0f * static_cast<float>(domain.voxelSize);
                    if (surfDist < bandMm &&
                        ctx.orientationMode != ToolOrientationMode::FollowEikonalGradient) {
                        const float wSurf = 1.0f - (surfDist / bandMm);
                        const auto& nSurf = domain.sdfGrad[i];
                        if (ctx.orientationMode ==
                            ToolOrientationMode::MostlyPerpendicularToSurface) {
                            std::array<float, 3> perp = nSurf;
                            if (dot3(perp, bedNormal) < 0.0f) {
                                perp = {-perp[0], -perp[1], -perp[2]};
                            }
                            desiredAxis = normalize3(
                                {(1.0f - 0.5f * wSurf) * desiredAxis[0] + 0.5f * wSurf * perp[0],
                                 (1.0f - 0.5f * wSurf) * desiredAxis[1] + 0.5f * wSurf * perp[1],
                                 (1.0f - 0.5f * wSurf) * desiredAxis[2] + 0.5f * wSurf * perp[2]},
                                bedNormal);
                        } else if (ctx.orientationMode ==
                                   ToolOrientationMode::MostlyParallelToSurface) {
                            const float proj = dot3(desiredAxis, nSurf);
                            const std::array<float, 3> tang = normalize3(
                                {desiredAxis[0] - proj * nSurf[0],
                                 desiredAxis[1] - proj * nSurf[1],
                                 desiredAxis[2] - proj * nSurf[2]},
                                bedNormal);
                            desiredAxis = normalize3(
                                {(1.0f - 0.5f * wSurf) * desiredAxis[0] + 0.5f * wSurf * tang[0],
                                 (1.0f - 0.5f * wSurf) * desiredAxis[1] + 0.5f * wSurf * tang[1],
                                 (1.0f - 0.5f * wSurf) * desiredAxis[2] + 0.5f * wSurf * tang[2]},
                                bedNormal);
                        }
                    }
                }

                clampedAxes[i] = clampUnitVectorToCone(desiredAxis, bedNormal, gammaMaxRad);
            }
        });

    std::vector<std::array<float, 3>> smoothedAxes = clampedAxes;
    for (int pass = 0; pass < 2; ++pass) {
        tbb::parallel_for(
            tbb::blocked_range<std::size_t>(0, n, 256),
            [&](const tbb::blocked_range<std::size_t>& r) {
                for (std::size_t i = r.begin(); i != r.end(); ++i) {
                    const NozzleBasis baseFrame =
                        ctx.backend.queryNozzleFrame(domain.posBed[i], {1.0f, 0.0f, 0.0f});
                    const std::array<float, 3> bedNormal = baseFrame.nozzle_axis;
                    const auto& n6 = domain.neighbors6[i];

                    std::array<float, 3> acc{
                        2.0f * clampedAxes[i][0],
                        2.0f * clampedAxes[i][1],
                        2.0f * clampedAxes[i][2]};

                    for (const std::int32_t nb : n6) {
                        if (nb >= 0) {
                            const auto& v = clampedAxes[static_cast<std::size_t>(nb)];
                            acc[0] += v[0];
                            acc[1] += v[1];
                            acc[2] += v[2];
                        }
                    }

                    const auto unitAvg = normalize3(acc, bedNormal);
                    smoothedAxes[i] = clampUnitVectorToCone(unitAvg, bedNormal, localMaxTiltRad[i]);
                }
            });
        clampedAxes.swap(smoothedAxes);
    }

    tbb::parallel_for(
        tbb::blocked_range<std::size_t>(0, n, 512),
        [&](const tbb::blocked_range<std::size_t>& r) {
            for (std::size_t i = r.begin(); i != r.end(); ++i) {
                const NozzleBasis baseFrame =
                    ctx.backend.queryNozzleFrame(domain.posBed[i], {1.0f, 0.0f, 0.0f});
                const std::array<float, 3> bedNormal = baseFrame.nozzle_axis;
                domain.toolAxis[i] = clampedAxes[i];
                const float cosTilt = std::clamp(dot3(domain.toolAxis[i], bedNormal), -1.0f, 1.0f);
                domain.toolTiltDeg[i] = std::acos(cosTilt) * kRadToDeg;
            }
        });
}

// ============================================================================
// 1. ToolClearanceConstraint Implementation (Parallelized via TBB)
// ============================================================================

float ToolClearanceConstraint::computeMaxClearanceSlopeRad(const ToolSpec& tool) noexcept {
    if (tool.clearance_profile_xz.empty()) {
        return 45.0f * kDegToRad;
    }

    float r0 = std::max(0.5f * tool.nozzle_diameter_mm, 0.2f);
    for (const auto& pt : tool.clearance_profile_xz) {
        if (pt.z_height_mm <= 1.0e-3f) {
            r0 = std::max(r0, pt.x_radius_mm);
        }
    }

    float minTanAlpha = std::tan(75.0f * kDegToRad);
    bool foundPositiveZ = false;
    for (const auto& pt : tool.clearance_profile_xz) {
        if (pt.z_height_mm > 0.05f) {
            foundPositiveZ = true;
            const float deltaR = std::max(pt.x_radius_mm - r0, 0.05f);
            const float tanAlpha = pt.z_height_mm / deltaR;
            minTanAlpha = std::min(minTanAlpha, tanAlpha);
        }
    }

    if (!foundPositiveZ) {
        return 45.0f * kDegToRad;
    }

    const float alphaRad = std::atan(minTanAlpha);
    return std::clamp(alphaRad, 8.0f * kDegToRad, 75.0f * kDegToRad);
}

void ToolClearanceConstraint::applyPreSweepModifiers(
    ActiveVolumeDomain& domain,
    const EikonalSolverContext& ctx) const
{
    const float alphaMaxRad = computeMaxClearanceSlopeRad(ctx.activeTool);
    const std::size_t n = domain.coords.size();

    tbb::parallel_for(
        tbb::blocked_range<std::size_t>(0, n, 512),
        [&](const tbb::blocked_range<std::size_t>& r) {
            for (std::size_t i = r.begin(); i != r.end(); ++i) {
                if (!domain.isInterior[i]) {
                    continue;
                }
                const auto& aAxis = domain.toolAxis[i];
                const auto gradTau = computeVoxelTauGradient(domain, i);
                const float gradLen = length3(gradTau);

                std::array<float, 3> rawDir = aAxis;
                if (gradLen > 1.0e-5f) {
                    rawDir = {gradTau[0] / gradLen, gradTau[1] / gradLen, gradTau[2] / gradLen};
                    if (dot3(rawDir, aAxis) < 0.0f) {
                        rawDir = {-rawDir[0], -rawDir[1], -rawDir[2]};
                    }
                }

                domain.targetDir[i] = clampUnitVectorToCone(rawDir, aAxis, alphaMaxRad);
            }
        });
}

void ToolClearanceConstraint::enforceOnField(
    ActiveVolumeDomain& domain,
    const EikonalSolverContext& ctx) const
{
    const std::size_t n = domain.coords.size();
    if (n == 0) {
        return;
    }

    const float alphaMaxRad = computeMaxClearanceSlopeRad(ctx.activeTool);
    const float tanAlphaMax = std::max(std::tan(alphaMaxRad), 0.14f);
    const float dx = static_cast<float>(domain.voxelSize);

    const std::array<std::array<float, 3>, 6> neighborOffsets{{
        {-dx, 0.0f, 0.0f},
        {dx, 0.0f, 0.0f},
        {0.0f, -dx, 0.0f},
        {0.0f, dx, 0.0f},
        {0.0f, 0.0f, -dx},
        {0.0f, 0.0f, dx},
    }};

    // 1. Parallel Z-slice Red-Black clearance cone enforcement
    const auto& zWf = domain.zParityWavefront;
    const std::size_t numZLevels = zWf.numLevels();

    auto enforceOnVoxel = [&](std::size_t idx) {
        const auto& aAxis = domain.toolAxis[idx];
        const auto& n6 = domain.neighbors6[idx];

        for (int k = 0; k < 6; ++k) {
            const std::int32_t nb = n6[static_cast<std::size_t>(k)];
            if (nb < 0) {
                continue;
            }
            const std::size_t u = static_cast<std::size_t>(nb);
            if (!domain.isInterior[u] || domain.tau[u] >= 0.5f * kInfinityTau) {
                continue;
            }

            const std::array<float, 3> stepFromU{
                -neighborOffsets[static_cast<std::size_t>(k)][0],
                -neighborOffsets[static_cast<std::size_t>(k)][1],
                -neighborOffsets[static_cast<std::size_t>(k)][2]};
            const float dzAxial = dot3(stepFromU, aAxis);
            const std::array<float, 3> radialVec{
                stepFromU[0] - dzAxial * aAxis[0],
                stepFromU[1] - dzAxial * aAxis[1],
                stepFromU[2] - dzAxial * aAxis[2]};
            const float drRadial = length3(radialVec);

            if (dzAxial > 0.1f * dx) {
                const float minProgress = std::max(0.15f * dx, dzAxial - drRadial * tanAlphaMax);
                domain.tau[idx] = std::max(domain.tau[idx], domain.tau[u] + 0.5f * minProgress);
            } else if (std::abs(dzAxial) <= 0.25f * dx && drRadial > 0.1f * dx) {
                const float maxLateralJump = drRadial * std::sin(alphaMaxRad);
                if (domain.tau[idx] > domain.tau[u] + maxLateralJump) {
                    domain.tau[idx] = domain.tau[u] + maxLateralJump;
                }
            }
        }
    };

    for (int pass = 0; pass < 2; ++pass) {
        for (std::size_t lvl = 0; lvl < numZLevels; ++lvl) {
            const std::size_t beginOff = zWf.levelOffsets[lvl];
            const std::size_t endOff = zWf.levelOffsets[lvl + 1];
            const std::size_t count = endOff - beginOff;

            if (count >= kMinParallelWavefrontSize) {
                tbb::parallel_for(
                    tbb::blocked_range<std::size_t>(beginOff, endOff, kMinParallelWavefrontSize),
                    [&](const tbb::blocked_range<std::size_t>& r) {
                        for (std::size_t pos = r.begin(); pos != r.end(); ++pos) {
                            enforceOnVoxel(zWf.indices[pos]);
                        }
                    });
            } else {
                for (std::size_t pos = beginOff; pos < endOff; ++pos) {
                    enforceOnVoxel(zWf.indices[pos]);
                }
            }
        }
    }

    // 2. Parallel Gather-based XZ clearance profile ring verification in the local nozzle frame:
    // Each interior voxel `u` queries points `idx` below it (`pos[u] - z_m * zNozzle - r_m * radial`)
    // and enforces `newTau[u] >= oldTau[idx] + 0.15 * z_m` without write contention.
    if (!ctx.activeTool.clearance_profile_xz.empty() && domain.indexGrid) {
        const std::vector<float> oldTau = domain.tau;
        const std::array<std::array<float, 2>, 4> ringDirs{{
            {1.0f, 0.0f},
            {-1.0f, 0.0f},
            {0.0f, 1.0f},
            {0.0f, -1.0f},
        }};

        tbb::parallel_for(
            tbb::blocked_range<std::size_t>(0, zWf.indices.size(), 256),
            [&](const tbb::blocked_range<std::size_t>& r) {
                auto localIdxAcc = domain.indexGrid->getConstAccessor();

                for (std::size_t posIdx = r.begin(); posIdx != r.end(); ++posIdx) {
                    const std::size_t u = zWf.indices[posIdx];
                    const auto& pos = domain.posBed[u];
                    const auto& zNozzle = domain.toolAxis[u];

                    const std::array<float, 3> refDir =
                        (std::abs(zNozzle[0]) < 0.9f) ? std::array<float, 3>{1.0f, 0.0f, 0.0f}
                                                      : std::array<float, 3>{0.0f, 1.0f, 0.0f};
                    const float dotRef = dot3(refDir, zNozzle);
                    const std::array<float, 3> xNozzle = normalize3({
                        refDir[0] - dotRef * zNozzle[0],
                        refDir[1] - dotRef * zNozzle[1],
                        refDir[2] - dotRef * zNozzle[2]});
                    const std::array<float, 3> yNozzle = normalize3(cross3(zNozzle, xNozzle));

                    float reqTau = oldTau[u];
                    for (const auto& stage : ctx.activeTool.clearance_profile_xz) {
                        const float rMm = stage.x_radius_mm;
                        const float zMm = stage.z_height_mm;
                        if (zMm < 0.75f * dx || zMm > 3.5f * dx) {
                            continue;
                        }

                        for (const auto& cs : ringDirs) {
                            const openvdb::Vec3d lowerWorld(
                                pos[0] - zMm * zNozzle[0] - rMm * (cs[0] * xNozzle[0] + cs[1] * yNozzle[0]),
                                pos[1] - zMm * zNozzle[1] - rMm * (cs[0] * xNozzle[1] + cs[1] * yNozzle[1]),
                                pos[2] - zMm * zNozzle[2] - rMm * (cs[0] * xNozzle[2] + cs[1] * yNozzle[2]));
                            const openvdb::Coord lowerCoord =
                                domain.transform->worldToIndexCellCentered(lowerWorld);
                            const std::int32_t lowerIdx = localIdxAcc.getValue(lowerCoord);
                            if (lowerIdx >= 0) {
                                const std::size_t idx = static_cast<std::size_t>(lowerIdx);
                                if (domain.isInterior[idx] && oldTau[idx] < 0.5f * kInfinityTau) {
                                    reqTau = std::max(reqTau, oldTau[idx] + 0.15f * zMm);
                                }
                            }
                        }
                    }
                    domain.tau[u] = reqTau;
                }
            });
    }
}

// ============================================================================
// 2. OverhangIslandConstraint Implementation
// ============================================================================

void OverhangIslandConstraint::applyPreSweepModifiers(
    ActiveVolumeDomain& domain,
    const EikonalSolverContext& /*ctx*/) const
{
    const std::size_t n = domain.coords.size();
    const float dx = static_cast<float>(domain.voxelSize);
    const float surfaceShellMm = 4.0f * dx;
    constexpr float kOverhangThreshold = 0.42f;

    tbb::parallel_for(
        tbb::blocked_range<std::size_t>(0, n, 512),
        [&](const tbb::blocked_range<std::size_t>& r) {
            for (std::size_t i = r.begin(); i != r.end(); ++i) {
                if (!domain.isInterior[i]) {
                    domain.slowness[i] = 1.0f;
                    continue;
                }

                const float ov = domain.overhang[i];
                const float distFromSurface = std::abs(domain.sdf[i]);

                if (ov > kOverhangThreshold && distFromSurface < surfaceShellMm) {
                    const float shellFactor = 1.0f - (distFromSurface / surfaceShellMm);
                    const float ovSeverity =
                        std::clamp((ov - kOverhangThreshold) / (1.0f - kOverhangThreshold), 0.0f, 1.0f);
                    domain.slowness[i] =
                        std::clamp(1.0f - 0.42f * shellFactor * ovSeverity, 0.48f, 1.0f);

                    const auto& nOut = domain.sdfGrad[i];
                    const float wBias = 0.35f * shellFactor * ovSeverity;
                    domain.targetDir[i] = normalize3(
                        {(1.0f - wBias) * domain.targetDir[i][0] + wBias * nOut[0],
                         (1.0f - wBias) * domain.targetDir[i][1] + wBias * nOut[1],
                         (1.0f - wBias) * domain.targetDir[i][2] + wBias * nOut[2]},
                        domain.toolAxis[i]);
                } else {
                    domain.slowness[i] = 1.0f;
                }
            }
        });
}

void OverhangIslandConstraint::enforceOnField(
    ActiveVolumeDomain& domain,
    const EikonalSolverContext& /*ctx*/) const
{
    const std::size_t n = domain.coords.size();
    if (n == 0) {
        return;
    }

    const float dx = static_cast<float>(domain.voxelSize);
    const float epsStep = 0.04f * dx;

    struct HeapEntry {
        float tau;
        std::size_t idx;
        bool operator>(const HeapEntry& other) const noexcept {
            return tau > other.tau;
        }
    };

    std::priority_queue<HeapEntry, std::vector<HeapEntry>, std::greater<HeapEntry>> pq;
    std::vector<std::uint8_t> settled(n, 0);
    std::vector<float> causalTau(n, kInfinityTau);

    std::size_t seedCount = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (domain.isSeed[i]) {
            causalTau[i] = domain.tau[i];
            pq.push(HeapEntry{causalTau[i], i});
            ++seedCount;
        }
    }

    if (seedCount == 0) {
        return;
    }

    while (!pq.empty()) {
        const HeapEntry top = pq.top();
        pq.pop();

        const std::size_t u = top.idx;
        if (settled[u]) {
            continue;
        }
        settled[u] = 1;

        const auto& n6 = domain.neighbors6[u];
        for (const std::int32_t nb : n6) {
            if (nb < 0) {
                continue;
            }
            const std::size_t v = static_cast<std::size_t>(nb);
            if (settled[v] || !domain.isInterior[v]) {
                continue;
            }

            const float requiredMin = causalTau[u] + epsStep;
            const float candidate = std::max(domain.tau[v], requiredMin);
            if (candidate < causalTau[v]) {
                causalTau[v] = candidate;
                pq.push(HeapEntry{candidate, v});
            }
        }
    }

    tbb::parallel_for(
        tbb::blocked_range<std::size_t>(0, n, 512),
        [&](const tbb::blocked_range<std::size_t>& r) {
            for (std::size_t i = r.begin(); i != r.end(); ++i) {
                if (domain.isInterior[i] && settled[i]) {
                    domain.tau[i] = causalTau[i];
                }
            }
        });
}

// ============================================================================
// 3. OrientationHintConstraint Implementation (Parallelized via TBB)
// ============================================================================

void OrientationHintConstraint::applyPreSweepModifiers(
    ActiveVolumeDomain& domain,
    const EikonalSolverContext& ctx) const
{
    if (ctx.hints.empty()) {
        return;
    }

    const std::size_t n = domain.coords.size();
    const float surfaceBandMm = 2.5f * static_cast<float>(domain.voxelSize);

    tbb::parallel_for(
        tbb::blocked_range<std::size_t>(0, n, 512),
        [&](const tbb::blocked_range<std::size_t>& r) {
            for (std::size_t i = r.begin(); i != r.end(); ++i) {
                if (!domain.isInterior[i]) {
                    continue;
                }

                const auto& pos = domain.posBed[i];
                std::array<float, 3> blendedDir = domain.targetDir[i];

                for (const auto& hint : ctx.hints) {
                    if (hint.applyOnSurfaceOnly && std::abs(domain.sdf[i]) > surfaceBandMm) {
                        continue;
                    }

                    const std::array<float, 3> diff{
                        pos[0] - hint.positionBed[0],
                        pos[1] - hint.positionBed[1],
                        pos[2] - hint.positionBed[2]};
                    const float dist = length3(diff);
                    const float radius = std::max(hint.influenceRadiusMm, 1.0e-3f);
                    if (dist >= radius) {
                        continue;
                    }

                    const float falloff =
                        (1.0f - dist / radius) * std::clamp(hint.weight, 0.0f, 1.0f);
                    const auto hintVec = normalize3(hint.directionBed, domain.toolAxis[i]);

                    std::array<float, 3> targetFromHint = blendedDir;
                    if (hint.kind == HintKind::Orthogonality) {
                        targetFromHint = hintVec;
                        if (dot3(targetFromHint, domain.toolAxis[i]) < 0.0f) {
                            targetFromHint = {
                                -targetFromHint[0],
                                -targetFromHint[1],
                                -targetFromHint[2]};
                        }
                    } else {
                        const float proj = dot3(blendedDir, hintVec);
                        targetFromHint = normalize3(
                            {blendedDir[0] - proj * hintVec[0],
                             blendedDir[1] - proj * hintVec[1],
                             blendedDir[2] - proj * hintVec[2]},
                            domain.toolAxis[i]);
                    }

                    blendedDir = normalize3(
                        {(1.0f - falloff) * blendedDir[0] + falloff * targetFromHint[0],
                         (1.0f - falloff) * blendedDir[1] + falloff * targetFromHint[1],
                         (1.0f - falloff) * blendedDir[2] + falloff * targetFromHint[2]},
                        domain.toolAxis[i]);
                }

                domain.targetDir[i] = blendedDir;
            }
        });
}

void OrientationHintConstraint::enforceOnField(
    ActiveVolumeDomain& /*domain*/,
    const EikonalSolverContext& /*ctx*/) const
{
}

// ============================================================================
// Top-Level Iterative Constrained Eikonal + Tool Orientation Solver
// ============================================================================

EikonalSolveResult solveConstrainedEikonal(
    const openvdb::FloatGrid& sdfGrid,
    float objectMinZ,
    float objectMaxZ,
    float objectMaxSpan,
    float exteriorBandMm,
    const EikonalSolverContext& baseContext)
{
    auto notifyProgress = [&](float frac, const char* msg) {
        if (baseContext.reportProgress) {
            baseContext.reportProgress(frac, msg);
        }
    };

    notifyProgress(0.20f, "Initializing active volume domain & SDF gradients...");

    ActiveVolumeDomain domain;
    domain.initializeFromSdfGrid(sdfGrid, objectMinZ, objectMaxZ, objectMaxSpan);

    const std::size_t n = domain.coords.size();
    const float dx = static_cast<float>(domain.voxelSize);

    // 1. Evaluate the gravity-aware "overhang" field and initialize bed contact seeds (\tau = 0) in parallel
    notifyProgress(0.26f, "Evaluating overhang field & bed contact seeds...");

    const float interiorCutoff = 0.75f * dx;
    const float seedHeightCutoff = 1.50f * dx;

    tbb::parallel_for(
        tbb::blocked_range<std::size_t>(0, n, 512),
        [&](const tbb::blocked_range<std::size_t>& r) {
            for (std::size_t i = r.begin(); i != r.end(); ++i) {
                const auto& pos = domain.posBed[i];
                const NozzleBasis nozzleFrame =
                    baseContext.backend.queryNozzleFrame(pos, {1.0f, 0.0f, 0.0f});
                const std::array<float, 3> gravityDir =
                    baseContext.backend.queryGravityDirection(pos);

                const float antiGravityProj = -dot3(domain.sdfGrad[i], gravityDir);
                domain.overhang[i] = std::clamp(antiGravityProj, -1.0f, 1.0f);

                domain.toolAxis[i] = nozzleFrame.nozzle_axis;
                domain.targetDir[i] = nozzleFrame.nozzle_axis;

                const bool insideOrBoundary = (domain.sdf[i] <= interiorCutoff);
                domain.isInterior[i] = insideOrBoundary ? 1 : 0;

                if (insideOrBoundary) {
                    const std::array<float, 3> rel{pos[0], pos[1], pos[2] - objectMinZ};
                    const float hAxial = dot3(rel, nozzleFrame.nozzle_axis);
                    if (hAxial <= seedHeightCutoff) {
                        domain.isSeed[i] = 1;
                        domain.tau[i] = std::max(0.0f, hAxial);
                    }
                }
            }
        });

    std::size_t seedCount = 0;
    float minInteriorAxialHeight = std::numeric_limits<float>::max();
    for (std::size_t i = 0; i < n; ++i) {
        if (domain.isSeed[i]) {
            ++seedCount;
        }
        if (domain.isInterior[i]) {
            const NozzleBasis nozzleFrame =
                baseContext.backend.queryNozzleFrame(domain.posBed[i], {1.0f, 0.0f, 0.0f});
            const std::array<float, 3> rel{
                domain.posBed[i][0],
                domain.posBed[i][1],
                domain.posBed[i][2] - objectMinZ};
            minInteriorAxialHeight =
                std::min(minInteriorAxialHeight, dot3(rel, nozzleFrame.nozzle_axis));
        }
    }

    if (seedCount == 0 && minInteriorAxialHeight < 0.5f * kInfinityTau) {
        for (std::size_t i = 0; i < n; ++i) {
            if (!domain.isInterior[i]) {
                continue;
            }
            const NozzleBasis nozzleFrame =
                baseContext.backend.queryNozzleFrame(domain.posBed[i], {1.0f, 0.0f, 0.0f});
            const std::array<float, 3> rel{
                domain.posBed[i][0],
                domain.posBed[i][1],
                domain.posBed[i][2] - objectMinZ};
            const float hAxial = dot3(rel, nozzleFrame.nozzle_axis);
            if (hAxial <= minInteriorAxialHeight + seedHeightCutoff) {
                domain.isSeed[i] = 1;
                domain.tau[i] = std::max(0.0f, hAxial - minInteriorAxialHeight);
                ++seedCount;
            }
        }
    }

    // Build O(N) counting-sort hyperplane wavefront schedules for parallel TBB sweeping
    domain.buildWavefrontSchedules();

    // 2. Step A: Unconstrained Eikonal Fast Sweeping (|\nabla \tau| = 1 from bed seeds)
    notifyProgress(0.32f, "Solving initial unconstrained 3D Eikonal wavefronts...");
    runInteriorFastSweeping(domain, 2, /*useDirectionalGuidance=*/false);

    ToolOrientationStrategy orientationStrategy;
    std::vector<std::unique_ptr<EikonalConstraint>> constraints;
    constraints.push_back(std::make_unique<OverhangIslandConstraint>());
    constraints.push_back(std::make_unique<OrientationHintConstraint>());
    constraints.push_back(std::make_unique<ToolClearanceConstraint>());

    EikonalSolverContext iterCtx = baseContext;
    iterCtx.outerIteration = 0;
    constraints[0]->enforceOnField(domain, iterCtx);

    // 3. Step B: Outer Iterative Tool-Orientation & Constraint Relaxation Loop
    const std::uint32_t totalIters = std::clamp<std::uint32_t>(baseContext.totalIterations, 1, 16);
    for (std::uint32_t iter = 0; iter < totalIters; ++iter) {
        iterCtx.outerIteration = iter;
        iterCtx.totalIterations = totalIters;

        const float iterBaseFrac =
            0.40f + 0.46f * (static_cast<float>(iter) / static_cast<float>(totalIters));
        const float iterSpan = 0.46f / static_cast<float>(totalIters);

        const std::string msgOrient =
            "Iteration " + std::to_string(iter + 1) + "/" + std::to_string(totalIters) +
            ": Updating 5/6-axis tool orientations...";
        notifyProgress(iterBaseFrac, msgOrient.c_str());

        orientationStrategy.updateToolOrientations(domain, iterCtx);

        for (const auto& c : constraints) {
            c->applyPreSweepModifiers(domain, iterCtx);
        }

        const std::string msgSweep =
            "Iteration " + std::to_string(iter + 1) + "/" + std::to_string(totalIters) +
            ": Parallel Eikonal sweeping & clearance constraints...";
        notifyProgress(iterBaseFrac + 0.5f * iterSpan, msgSweep.c_str());

        runInteriorFastSweeping(domain, 1, /*useDirectionalGuidance=*/true);

        for (const auto& c : constraints) {
            c->enforceOnField(domain, iterCtx);
        }
    }

    notifyProgress(0.88f, "Finalizing tool orientations & anti-island causality...");
    orientationStrategy.updateToolOrientations(domain, iterCtx);
    constraints[0]->enforceOnField(domain, iterCtx);

    // 4. Smoothly extend \tau(x) into the exterior narrow band in parallel
    notifyProgress(0.92f, "Extending Eikonal field into exterior narrow band...");
    extendEikonalToExteriorBand(domain, baseContext.backend);

    // 5. Pack solved fields into OpenVDB grids
    notifyProgress(0.96f, "Packing OpenVDB scalar & vector grids...");
    const float spanZ = std::max(objectMaxZ - objectMinZ, 1.0f);
    const float eikonalBg = spanZ + exteriorBandMm;

    openvdb::FloatGrid::Ptr eikonalGrid = openvdb::FloatGrid::create(eikonalBg);
    eikonalGrid->setName("eikonal");
    eikonalGrid->setGridClass(openvdb::GRID_LEVEL_SET);
    eikonalGrid->setTransform(domain.transform->copy());

    openvdb::FloatGrid::Ptr overhangGrid = openvdb::FloatGrid::create(0.0f);
    overhangGrid->setName("overhang");
    overhangGrid->setGridClass(openvdb::GRID_LEVEL_SET);
    overhangGrid->setTransform(domain.transform->copy());

    openvdb::FloatGrid::Ptr toolTiltGrid = openvdb::FloatGrid::create(0.0f);
    toolTiltGrid->setName("tool_tilt_deg");
    toolTiltGrid->setGridClass(openvdb::GRID_FOG_VOLUME);
    toolTiltGrid->setTransform(domain.transform->copy());

    openvdb::Vec3SGrid::Ptr toolAxisGrid =
        openvdb::Vec3SGrid::create(openvdb::Vec3s(0.0f, 0.0f, 1.0f));
    toolAxisGrid->setName("tool_axis");
    toolAxisGrid->setTransform(domain.transform->copy());

    auto eikAccessor = eikonalGrid->getAccessor();
    auto ovAccessor = overhangGrid->getAccessor();
    auto tiltAccessor = toolTiltGrid->getAccessor();
    auto axisAccessor = toolAxisGrid->getAccessor();

    for (std::size_t i = 0; i < n; ++i) {
        const openvdb::Coord& ijk = domain.coords[i];
        const float tauVal = (domain.tau[i] < 0.5f * kInfinityTau) ? domain.tau[i] : eikonalBg;
        eikAccessor.setValue(ijk, tauVal);
        ovAccessor.setValue(ijk, domain.overhang[i]);
        tiltAccessor.setValue(ijk, domain.toolTiltDeg[i]);
        axisAccessor.setValue(
            ijk,
            openvdb::Vec3s(
                domain.toolAxis[i][0],
                domain.toolAxis[i][1],
                domain.toolAxis[i][2]));
    }

    notifyProgress(1.0f, "Field computation complete.");

    EikonalSolveResult result;
    result.eikonalGrid = eikonalGrid;
    result.overhangGrid = overhangGrid;
    result.toolTiltDegGrid = toolTiltGrid;
    result.toolAxisGrid = toolAxisGrid;
    return result;
}

} // namespace manifold
