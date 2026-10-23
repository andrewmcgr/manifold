#include "slicer_backend.h"
#include "manifold-frontend/src/bridge.rs.h"
#include "eikonal_solver.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <numeric>
#include <queue>
#include <string>

#include <openvdb/tools/Interpolation.h>
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
        // Skip the z=0 nozzle tip orifice flat so the tip itself can tilt on lower layers;
        // upper clearance profile stages (z > 0) constrain bed collision as a function of hBedMm.
        if (r <= 1.0e-4f || z <= 1.0e-3f) {
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

/// Solves the 3D Godunov upwind quadratic update at voxel `idx` for the initial
/// unconstrained isotropic Eikonal equation |\nabla \tau| = s(x).
float godunovIsotropicUpdate(
    const ActiveVolumeDomain& domain,
    std::size_t idx) noexcept
{
    if (domain.isSeed[idx]) {
        return domain.tau[idx];
    }
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

    return u;
}

/// Evaluates an upwind-biased directional gradient-integration update at voxel `idx`
/// that aligns \nabla \tau(x) with `domain.slowness[idx] * domain.targetDir[idx]`.
/// Because upwind neighbors ((x_i - x_u) \cdot g_{iu} > 0) receive higher weight than
/// downwind neighbors, opposite neighbors (+dx and -dx) never cancel out `targetDir`,
/// while any plane wave \tau(x) = g \cdot x + c is an exact fixed point.
float directionalGradientIntegrationUpdate(
    const ActiveVolumeDomain& domain,
    std::size_t idx) noexcept
{
    const float dx = static_cast<float>(domain.voxelSize);
    const auto& n6 = domain.neighbors6[idx];
    const float sIdx = std::max(0.15f, domain.slowness[idx]);
    const auto& dIdx = domain.targetDir[idx];

    // Step vectors x_idx - x_u for the 6 face neighbors:
    // k=0: u at -X => x_idx - x_u = (+dx, 0, 0)
    // k=1: u at +X => x_idx - x_u = (-dx, 0, 0), etc.
    const std::array<std::array<float, 3>, 6> stepFromU{{
        {dx, 0.0f, 0.0f},
        {-dx, 0.0f, 0.0f},
        {0.0f, dx, 0.0f},
        {0.0f, -dx, 0.0f},
        {0.0f, 0.0f, dx},
        {0.0f, 0.0f, -dx},
    }};

    float weightedTauSum = 0.0f;
    float weightSum = 0.0f;

    for (int k = 0; k < 6; ++k) {
        const std::int32_t nb = n6[static_cast<std::size_t>(k)];
        if (nb < 0) {
            continue;
        }
        const std::size_t u = static_cast<std::size_t>(nb);
        if (!domain.isInterior[u] || domain.tau[u] >= 0.5f * kInfinityTau) {
            continue;
        }

        const float sU = std::max(0.15f, domain.slowness[u]);
        const auto& dU = domain.targetDir[u];
        const std::array<float, 3> gEdge{
            0.5f * (sIdx * dIdx[0] + sU * dU[0]),
            0.5f * (sIdx * dIdx[1] + sU * dU[1]),
            0.5f * (sIdx * dIdx[2] + sU * dU[2])};

        const float deltaTau = dot3(gEdge, stepFromU[static_cast<std::size_t>(k)]);
        // Strong upwind bias: upstream neighbors (deltaTau > 0) carry weight up to 6.0,
        // whereas downstream neighbors carry weight 0.15 so boundary conditions still couple
        // smoothly without cancelling out the interior directional slope.
        const float w = (deltaTau >= 0.0f)
            ? (1.0f + 5.0f * (deltaTau / dx))
            : 0.15f;

        weightedTauSum += w * (domain.tau[u] + deltaTau);
        weightSum += w;
    }

    if (weightSum > 1.0e-5f) {
        return weightedTauSum / weightSum;
    }
    return domain.tau[idx];
}

/// Runs multi-threaded 8-octant Gauss-Seidel Fast Sweeping passes over all interior voxels
/// using the precomputed diagonal hyperplane wavefront schedules (Detrixhe et al. 2013).
/// When `useDirectionalGuidance` is false, solves the initial unconstrained isotropic Eikonal
/// equation; when true, integrates the constrained directional field `domain.targetDir`.
void runInteriorFastSweeping(
    ActiveVolumeDomain& domain,
    std::uint32_t numSweepRounds,
    bool useDirectionalGuidance)
{
    auto sweepWavefrontSlice = [&](const WavefrontSchedule& wf, std::size_t lvl) {
        const std::size_t beginOff = wf.levelOffsets[lvl];
        const std::size_t endOff = wf.levelOffsets[lvl + 1];
        const std::size_t count = endOff - beginOff;

        auto updateVoxel = [&](std::size_t idx) {
            if (domain.isPinnedSeed[idx]) {
                return;
            }
            if (!useDirectionalGuidance) {
                const float cand = godunovIsotropicUpdate(domain, idx);
                if (cand < domain.tau[idx]) {
                    domain.tau[idx] = cand;
                }
            } else {
                domain.tau[idx] = directionalGradientIntegrationUpdate(domain, idx);
            }
        };

        if (count >= kMinParallelWavefrontSize) {
            tbb::parallel_for(
                tbb::blocked_range<std::size_t>(beginOff, endOff, kMinParallelWavefrontSize),
                [&](const tbb::blocked_range<std::size_t>& r) {
                    for (std::size_t pos = r.begin(); pos != r.end(); ++pos) {
                        updateVoxel(wf.indices[pos]);
                    }
                });
        } else {
            for (std::size_t pos = beginOff; pos < endOff; ++pos) {
                updateVoxel(wf.indices[pos]);
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
/// (\phi_sdf > 0.75 * dx) in parallel via an outward shell-by-shell march so that
/// `openvdb::tools::volumeToMesh` extracts clean, continuous isosurface sheets across
/// the object boundary (\phi_sdf = 0) without flattening the interior gradient.
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
        } else {
            // Store the final interior gradient in targetDir so exterior extension continues the exact slope
            const auto g = computeVoxelTauGradient(domain, i);
            const float gLen = length3(g);
            if (gLen > 1.0e-4f) {
                domain.targetDir[i] = {g[0] / gLen, g[1] / gLen, g[2] / gLen};
                domain.slowness[i] = std::clamp(gLen, 0.25f, 2.5f);
            }
        }
    }
    for (const std::size_t extIdx : extIndices) {
        domain.tau[extIdx] = kInfinityTau;
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
    std::vector<std::array<float, 3>> nextDir = domain.targetDir;
    std::vector<float> nextSlowness = domain.slowness;

    for (int pass = 0; pass < 16; ++pass) {
        tbb::parallel_for(
            tbb::blocked_range<std::size_t>(0, extCount, 256),
            [&](const tbb::blocked_range<std::size_t>& r) {
                for (std::size_t e = r.begin(); e != r.end(); ++e) {
                    const std::size_t idx = extIndices[e];
                    const auto& n6 = domain.neighbors6[idx];
                    float sumWeightedTau = 0.0f;
                    float sumWeight = 0.0f;
                    std::array<float, 3> sumDir{0.0f, 0.0f, 0.0f};
                    float sumSlowness = 0.0f;

                    bool hasStrictlyCloserNeighbor = false;
                    for (int k = 0; k < 6; ++k) {
                        const std::int32_t nb = n6[static_cast<std::size_t>(k)];
                        if (nb < 0) {
                            continue;
                        }
                        const std::size_t u = static_cast<std::size_t>(nb);
                        if (domain.tau[u] < 0.5f * kInfinityTau &&
                            domain.sdf[u] < domain.sdf[idx] - 1.0e-5f) {
                            hasStrictlyCloserNeighbor = true;
                            break;
                        }
                    }

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
                        if (hasStrictlyCloserNeighbor && domain.sdf[u] >= domain.sdf[idx] - 1.0e-5f) {
                            continue;
                        }
                        // Extrapolate outward from strictly more-interior shells so outer/same-shell
                        // neighbors never damp the outward gradient magnitude during the march.
                        const float sdfDiff = std::max(0.05f * dx, domain.sdf[idx] - domain.sdf[u]);
                        const float w = hasStrictlyCloserNeighbor ? (dx / sdfDiff) : 1.0f;
                        const float sU = domain.slowness[u];
                        const float deltaTau =
                            sU * dot3(domain.targetDir[u], stepVecs[static_cast<std::size_t>(k)]);
                        sumWeightedTau += w * std::max(0.0f, nbTau + deltaTau);
                        sumDir[0] += w * domain.targetDir[u][0];
                        sumDir[1] += w * domain.targetDir[u][1];
                        sumDir[2] += w * domain.targetDir[u][2];
                        sumSlowness += w * sU;
                        sumWeight += w;
                    }

                    if (sumWeight > 0.0f) {
                        nextTau[idx] = sumWeightedTau / sumWeight;
                        nextDir[idx] = normalize3(sumDir, domain.targetDir[idx]);
                        nextSlowness[idx] = sumSlowness / sumWeight;
                    } else {
                        nextTau[idx] = domain.tau[idx];
                        nextDir[idx] = domain.targetDir[idx];
                        nextSlowness[idx] = domain.slowness[idx];
                    }
                }
            });

        tbb::parallel_for(
            tbb::blocked_range<std::size_t>(0, extCount, 512),
            [&](const tbb::blocked_range<std::size_t>& r) {
                for (std::size_t e = r.begin(); e != r.end(); ++e) {
                    const std::size_t idx = extIndices[e];
                    domain.tau[idx] = nextTau[idx];
                    domain.targetDir[idx] = nextDir[idx];
                    domain.slowness[idx] = nextSlowness[idx];
                }
            });
    }

    // Fallback for any disconnected exterior narrow-band voxels beyond 16 shells
    tbb::parallel_for(
        tbb::blocked_range<std::size_t>(0, extCount, 512),
        [&](const tbb::blocked_range<std::size_t>& r) {
            for (std::size_t e = r.begin(); e != r.end(); ++e) {
                const std::size_t idx = extIndices[e];
                if (domain.tau[idx] >= 0.5f * kInfinityTau) {
                    const NozzleBasis basis =
                        backend.queryNozzleFrame(domain.posBed[idx], {1.0f, 0.0f, 0.0f});
                    const float h = dot3(domain.posBed[idx], basis.nozzle_axis) - domain.minZBed;
                    domain.tau[idx] = std::max(0.0f, h);
                }
            }
        });
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
    isPinnedSeed.clear();
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
    isPinnedSeed.assign(n, 0);
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

    // Include all interior voxels in the wavefront schedules so directional integration
    // and harmonic hint diffusion can sweep smoothly across the entire interior domain.
    // (During the initial unconstrained isotropic Eikonal sweep, `godunovIsotropicUpdate`
    // keeps `isSeed` voxels fixed at their initial seed values.)
    for (std::size_t i = 0; i < n; ++i) {
        if (isInterior[i]) {
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

                const float hBedMm =
                    std::max(0.0f, dot3(domain.posBed[i], bedNormal) - domain.minZBed);
                const float gammaMaxRad =
                    computeBedSafeTiltRad(ctx.activeTool, hBedMm, maxAngleRad);
                localMaxTiltRad[i] = gammaMaxRad;

                std::array<float, 3> desiredAxis = bedNormal;
                if (domain.isInterior[i]) {
                    // `domain.targetDir[i]` already combines the current Eikonal gradient,
                    // overhang guidance, and diffused soft orientation hints.
                    desiredAxis = domain.targetDir[i];
                    if (ctx.hints.empty()) {
                        const auto gradTau = computeVoxelTauGradient(domain, i);
                        const float gradLen = length3(gradTau);
                        if (gradLen > 1.0e-5f) {
                            std::array<float, 3> gUnit{
                                gradTau[0] / gradLen,
                                gradTau[1] / gradLen,
                                gradTau[2] / gradLen};
                            if (dot3(gUnit, bedNormal) < 0.0f) {
                                gUnit = {-gUnit[0], -gUnit[1], -gUnit[2]};
                            }
                            desiredAxis = normalize3(
                                {0.5f * desiredAxis[0] + 0.5f * gUnit[0],
                                 0.5f * desiredAxis[1] + 0.5f * gUnit[1],
                                 0.5f * desiredAxis[2] + 0.5f * gUnit[2]},
                                bedNormal);
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
                const auto& rawDir = domain.targetDir[i];

                // Strictly clamp the target isosurface normal (including any soft orientation hints)
                // to the tool's clearance cone centered on the ACTUAL solved tool axis `aAxis`.
                // Note: Do not negate `rawDir` when `dot(rawDir, aAxis) < 0` as that would flip
                // its radial azimuth by 180 degrees; `clampUnitVectorToCone` preserves radial azimuth.
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
    const float cosAlphaMax = std::cos(alphaMaxRad);

    // 1. Check whether the discrete gradient \nabla \tau at any interior voxel violates
    // the clearance cone of half-angle `alphaMaxRad` around the ACTUAL tool axis `domain.toolAxis[i]`.
    // Voxels whose gradient already satisfies \angle(\nabla \tau, toolAxis) <= alphaMaxRad are untouched.
    std::vector<std::uint8_t> violatesCone(n, 0);
    std::atomic<bool> anyViolation{false};

    tbb::parallel_for(
        tbb::blocked_range<std::size_t>(0, n, 512),
        [&](const tbb::blocked_range<std::size_t>& r) {
            bool localViol = false;
            for (std::size_t i = r.begin(); i != r.end(); ++i) {
                if (!domain.isInterior[i]) {
                    continue;
                }
                const auto& aAxis = domain.toolAxis[i];
                const auto g = computeVoxelTauGradient(domain, i);
                const float gLen = length3(g);
                if (gLen > 1.0e-4f) {
                    const std::array<float, 3> gUnit{g[0] / gLen, g[1] / gLen, g[2] / gLen};
                    const float cosTheta = dot3(gUnit, aAxis);
                    if (cosTheta < cosAlphaMax - 1.0e-3f) {
                        domain.targetDir[i] = clampUnitVectorToCone(gUnit, aAxis, alphaMaxRad);
                        violatesCone[i] = 1;
                        localViol = true;
                    } else {
                        domain.targetDir[i] = clampUnitVectorToCone(domain.targetDir[i], aAxis, alphaMaxRad);
                    }
                } else {
                    domain.targetDir[i] = clampUnitVectorToCone(domain.targetDir[i], aAxis, alphaMaxRad);
                }
            }
            if (localViol) {
                anyViolation.store(true, std::memory_order_relaxed);
            }
        });

    // If any voxel violated the tool's clearance cone around its actual tool axis,
    // re-integrate the clamped gradient across the octant wavefronts.
    if (anyViolation.load(std::memory_order_relaxed)) {
        runInteriorFastSweeping(domain, 1, /*useDirectionalGuidance=*/true);
    }
}

// ============================================================================
// 2. OverhangIslandConstraint Implementation
// ============================================================================

void OverhangIslandConstraint::applyPreSweepModifiers(
    ActiveVolumeDomain& domain,
    const EikonalSolverContext& ctx) const
{
    const std::size_t n = domain.coords.size();
    const float dx = static_cast<float>(domain.voxelSize);
    const float surfaceShellMm = 4.0f * dx;
    constexpr float kOverhangThreshold = 0.42f;

    tbb::parallel_for(
        tbb::blocked_range<std::size_t>(0, n, 512),
        [&](const tbb::blocked_range<std::size_t>& r) {
            for (std::size_t i = r.begin(); i != r.end(); ++i) {
                domain.slowness[i] = 1.0f;
                if (!domain.isInterior[i]) {
                    continue;
                }

                const auto& aAxis = domain.toolAxis[i];
                const auto gradTau = computeVoxelTauGradient(domain, i);
                const float gradLen = length3(gradTau);
                std::array<float, 3> baseDir = aAxis;
                if (gradLen > 1.0e-5f) {
                    baseDir = {gradTau[0] / gradLen, gradTau[1] / gradLen, gradTau[2] / gradLen};
                    if (dot3(baseDir, aAxis) < 0.0f) {
                        baseDir = {-baseDir[0], -baseDir[1], -baseDir[2]};
                    }
                }
                domain.targetDir[i] = baseDir;

                // `domain.overhang[i]` is -\nabla\phi_sdf \cdot \hat{g}_{bed}, which is negative
                // on downward-facing overhangs (\nabla\phi_sdf aligned with gravity \hat{g}_{bed}).
                const float downwardOverhang = -domain.overhang[i];
                const float distFromSurface = std::abs(domain.sdf[i]);

                if (!domain.isSeed[i] &&
                    downwardOverhang > kOverhangThreshold &&
                    distFromSurface < surfaceShellMm) {
                    const float shellFactor = 1.0f - (distFromSurface / surfaceShellMm);
                    const float ovSeverity = std::clamp(
                        (downwardOverhang - kOverhangThreshold) / (1.0f - kOverhangThreshold),
                        0.0f,
                        1.0f);

                    const std::array<float, 3> gravityDir =
                        ctx.backend.queryGravityDirection(domain.posBed[i]);
                    const auto& nOut = domain.sdfGrad[i];
                    const float gProj = dot3(nOut, gravityDir);
                    const std::array<float, 3> lateralOut = normalize3(
                        {nOut[0] - gProj * gravityDir[0],
                         nOut[1] - gProj * gravityDir[1],
                         nOut[2] - gProj * gravityDir[2]},
                        aAxis);
                    const float wBias = 0.25f * shellFactor * ovSeverity;
                    domain.targetDir[i] = normalize3(
                        {(1.0f - wBias) * domain.targetDir[i][0] + wBias * lateralOut[0],
                         (1.0f - wBias) * domain.targetDir[i][1] + wBias * lateralOut[1],
                         (1.0f - wBias) * domain.targetDir[i][2] + wBias * lateralOut[2]},
                        aAxis);
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

    // Shift interior \tau so that the earliest bed seed voxel starts at \tau = 0
    // while allowing tilted layers across the bed to have \tau >= 0.
    // Ignore pinned inner-core boundary seeds (`isPinnedSeed`) when computing `minSeedTau`.
    float minSeedTau = std::numeric_limits<float>::max();
    for (std::size_t i = 0; i < n; ++i) {
        if (domain.isSeed[i] && !domain.isPinnedSeed[i] && domain.tau[i] < 0.5f * kInfinityTau) {
            minSeedTau = std::min(minSeedTau, domain.tau[i]);
        }
    }
    if (minSeedTau < 0.5f * kInfinityTau && std::abs(minSeedTau) > 1.0e-6f) {
        tbb::parallel_for(
            tbb::blocked_range<std::size_t>(0, n, 512),
            [&](const tbb::blocked_range<std::size_t>& r) {
                for (std::size_t i = r.begin(); i != r.end(); ++i) {
                    if (domain.isInterior[i] && !domain.isPinnedSeed[i] &&
                        domain.tau[i] < 0.5f * kInfinityTau) {
                        domain.tau[i] = std::max(0.0f, domain.tau[i] - minSeedTau);
                    }
                }
            });
    }

    const float dx = static_cast<float>(domain.voxelSize);
    const float epsStep = 0.02f * dx;

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
            causalTau[i] = std::max(0.0f, domain.tau[i]);
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

namespace {

/// Evaluates the direct soft orientation hint weight `wMax \in [0, 1]` and desired unit
/// isosurface normal `outTargetDir` at voxel `i`.
float evaluateSoftHintAtVoxel(
    const ActiveVolumeDomain& domain,
    const EikonalSolverContext& ctx,
    std::size_t i,
    std::array<float, 3>& outTargetDir) noexcept
{
    const auto& pos = domain.posBed[i];
    const auto& baseDir = domain.targetDir[i];
    const NozzleBasis baseFrame =
        ctx.backend.queryNozzleFrame(pos, {1.0f, 0.0f, 0.0f});
    const std::array<float, 3> bedNormal = baseFrame.nozzle_axis;
    const float sdfDepth = std::abs(domain.sdf[i]);

    std::array<float, 3> accumVec{0.0f, 0.0f, 0.0f};
    float maxFalloff = 0.0f;

    for (const auto& hint : ctx.hints) {
        // Use a generous effective influence radius so surface and point hints penetrate well
        // into the volume before harmonic diffusion spreads them smoothly across the object.
        const float radius = std::max(hint.influenceRadiusMm, 3.0f * static_cast<float>(domain.voxelSize));
        const float maxDepth = std::max(2.0f * radius, 0.35f * domain.maxSpanMm);
        if (hint.applyOnSurfaceOnly && sdfDepth > maxDepth) {
            continue;
        }

        const std::array<float, 3> diff{
            pos[0] - hint.positionBed[0],
            pos[1] - hint.positionBed[1],
            pos[2] - hint.positionBed[2]};
        const float dist = length3(diff);
        const float reach = hint.applyOnSurfaceOnly ? (1.5f * radius) : (2.0f * radius);
        if (dist >= reach) {
            continue;
        }

        const float uDist = std::clamp(1.0f - dist / reach, 0.0f, 1.0f);
        // Smooth cubic falloff that stays strong (near 1.0) close to the hint
        float falloff = uDist * uDist * (3.0f - 2.0f * uDist) * std::clamp(hint.weight, 0.0f, 1.0f);
        if (hint.applyOnSurfaceOnly) {
            const float uDepth = std::clamp(1.0f - sdfDepth / maxDepth, 0.0f, 1.0f);
            falloff *= std::sqrt(uDepth);
        }
        if (falloff <= 1.0e-4f) {
            continue;
        }

        const auto hintVec = normalize3(hint.directionBed, bedNormal);
        std::array<float, 3> targetFromHint = baseDir;
        if (hint.kind == HintKind::Orthogonality) {
            // Isosurface normal \nabla \tau aligned with \pm hintVec:
            // - For a 3D point hint (`!applyOnSurfaceOnly`), preserve the user's chosen arrow
            //   direction (only flipping if it points steeply downward into the bed).
            // - For a surface orthogonality hint (`applyOnSurfaceOnly`), `hintVec` is the outward
            //   face normal; if the face points downward (overhang), flip to `-hintVec` so \nabla \tau
            //   points upward while remaining orthogonal to the surface; if the face is a vertical
            //   side wall, bias slightly upward so clamping tilts cleanly toward `+hintVec`.
            targetFromHint = hintVec;
            const float dotBed = dot3(targetFromHint, bedNormal);
            if (hint.applyOnSurfaceOnly) {
                if (dotBed < -0.15f) {
                    targetFromHint = {-targetFromHint[0], -targetFromHint[1], -targetFromHint[2]};
                } else if (std::abs(dotBed) <= 0.15f) {
                    targetFromHint = normalize3(
                        {targetFromHint[0] + 0.25f * bedNormal[0],
                         targetFromHint[1] + 0.25f * bedNormal[1],
                         targetFromHint[2] + 0.25f * bedNormal[2]},
                        bedNormal);
                }
            } else if (dotBed < -0.25f) {
                targetFromHint = {-targetFromHint[0], -targetFromHint[1], -targetFromHint[2]};
            }
        } else {
            // HintKind::Tangency: Isosurface normal \nabla \tau tangent to surface (\nabla \tau \perp hintVec)
            float proj = dot3(baseDir, hintVec);
            std::array<float, 3> tang{
                baseDir[0] - proj * hintVec[0],
                baseDir[1] - proj * hintVec[1],
                baseDir[2] - proj * hintVec[2]};
            if (length3(tang) < 0.10f) {
                // When the selected face normal `hintVec` is nearly parallel to `baseDir`
                // (e.g. a horizontal top or bottom face!), project a horizontal/radial reference
                // direction onto the face's tangent plane so layers tilt to meet the face at 90 degrees.
                std::array<float, 3> radialRef{pos[0], pos[1], 0.0f};
                if (length3(radialRef) < 1.0e-3f) {
                    radialRef = baseFrame.travel_axis;
                } else {
                    radialRef = normalize3(radialRef, baseFrame.travel_axis);
                }
                const float projRad = dot3(radialRef, hintVec);
                tang = {
                    radialRef[0] - projRad * hintVec[0],
                    radialRef[1] - projRad * hintVec[1],
                    radialRef[2] - projRad * hintVec[2]};
            }
            targetFromHint = normalize3(tang, bedNormal);
            if (dot3(targetFromHint, bedNormal) < -0.1f) {
                targetFromHint = {-targetFromHint[0], -targetFromHint[1], -targetFromHint[2]};
            }
        }

        accumVec[0] += falloff * targetFromHint[0];
        accumVec[1] += falloff * targetFromHint[1];
        accumVec[2] += falloff * targetFromHint[2];
        maxFalloff = std::max(maxFalloff, falloff);
    }

    if (maxFalloff > 1.0e-4f) {
        const auto unitHint = normalize3(accumVec, baseDir);
        outTargetDir = normalize3(
            {(1.0f - maxFalloff) * baseDir[0] + maxFalloff * unitHint[0],
             (1.0f - maxFalloff) * baseDir[1] + maxFalloff * unitHint[1],
             (1.0f - maxFalloff) * baseDir[2] + maxFalloff * unitHint[2]},
            bedNormal);
    } else {
        outTargetDir = baseDir;
    }
    return maxFalloff;
}

} // namespace

void OrientationHintConstraint::applyPreSweepModifiers(
    ActiveVolumeDomain& domain,
    const EikonalSolverContext& ctx) const
{
    if (ctx.hints.empty()) {
        return;
    }

    const std::size_t n = domain.coords.size();
    std::vector<std::array<float, 3>> directHintDir(n, {0.0f, 0.0f, 1.0f});
    std::vector<std::array<float, 3>> baseDirs = domain.targetDir;
    std::vector<float> directWeight(n, 0.0f);

    tbb::parallel_for(
        tbb::blocked_range<std::size_t>(0, n, 512),
        [&](const tbb::blocked_range<std::size_t>& r) {
            for (std::size_t i = r.begin(); i != r.end(); ++i) {
                if (!domain.isInterior[i]) {
                    continue;
                }
                std::array<float, 3> hDir = domain.targetDir[i];
                const float w = evaluateSoftHintAtVoxel(domain, ctx, i, hDir);
                directHintDir[i] = hDir;
                directWeight[i] = w;
                if (w > 1.0e-4f) {
                    domain.targetDir[i] = hDir;
                }
            }
        });

    // Harmonically diffuse the hinted target normal field across the interior volume using
    // the parallel octant wavefront schedules so surface-face and 3D point hints smoothly
    // steer the entire surrounding layer field rather than only a thin skin.
    auto diffuseWavefrontSlice = [&](const WavefrontSchedule& wf, std::size_t lvl) {
        const std::size_t beginOff = wf.levelOffsets[lvl];
        const std::size_t endOff = wf.levelOffsets[lvl + 1];
        const std::size_t count = endOff - beginOff;

        auto diffuseVoxel = [&](std::size_t idx) {
            const float wDirect = directWeight[idx];
            if (wDirect >= 0.92f) {
                domain.targetDir[idx] = directHintDir[idx];
                return;
            }

            const auto& n6 = domain.neighbors6[idx];
            // Weak background tether (0.04) toward `baseDirs[idx]` so hints propagate far into
            // the interior while un-hinted regions far away still relax smoothly.
            std::array<float, 3> acc{
                0.04f * baseDirs[idx][0] + 2.5f * wDirect * directHintDir[idx][0],
                0.04f * baseDirs[idx][1] + 2.5f * wDirect * directHintDir[idx][1],
                0.04f * baseDirs[idx][2] + 2.5f * wDirect * directHintDir[idx][2]};

            for (const std::int32_t nb : n6) {
                if (nb >= 0) {
                    const std::size_t u = static_cast<std::size_t>(nb);
                    if (domain.isInterior[u]) {
                        acc[0] += domain.targetDir[u][0];
                        acc[1] += domain.targetDir[u][1];
                        acc[2] += domain.targetDir[u][2];
                    }
                }
            }

            domain.targetDir[idx] = normalize3(acc, baseDirs[idx]);
        };

        if (count >= kMinParallelWavefrontSize) {
            tbb::parallel_for(
                tbb::blocked_range<std::size_t>(beginOff, endOff, kMinParallelWavefrontSize),
                [&](const tbb::blocked_range<std::size_t>& r) {
                    for (std::size_t pos = r.begin(); pos != r.end(); ++pos) {
                        diffuseVoxel(wf.indices[pos]);
                    }
                });
        } else {
            for (std::size_t pos = beginOff; pos < endOff; ++pos) {
                diffuseVoxel(wf.indices[pos]);
            }
        }
    };

    for (int round = 0; round < 2; ++round) {
        for (const auto& wf : domain.octantWavefronts) {
            const std::size_t numLvls = wf.numLevels();
            for (std::size_t lvl = 0; lvl < numLvls; ++lvl) {
                diffuseWavefrontSlice(wf, lvl);
            }
            for (std::size_t lvl = numLvls; lvl-- > 0;) {
                diffuseWavefrontSlice(wf, lvl);
            }
        }
    }
}

void OrientationHintConstraint::enforceOnField(
    ActiveVolumeDomain& domain,
    const EikonalSolverContext& ctx) const
{
    if (ctx.hints.empty()) {
        return;
    }

    // Perform an additional directional wavefront integration round along the clearance-clamped
    // `domain.targetDir` when hints are active so `\tau` converges tightly to the hinted slope.
    runInteriorFastSweeping(domain, 1, /*useDirectionalGuidance=*/true);
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

    // 1. Evaluate the gravity-aware "overhang" field and determine bed contact height along
    // the queried nozzle axis in parallel
    notifyProgress(0.26f, "Evaluating overhang field & bed contact seeds...");

    // 0.90 * dx exceeds the 3D cubic lattice covering radius (\sqrt{3}/2 * dx \approx 0.866 * dx),
    // guaranteeing that any thin curved wall at arbitrary 3D orientation forms a 6-connected
    // interior voxel network.
    const float interiorCutoff = 0.90f * dx;
    const float seedHeightCutoff = 1.50f * dx;

    std::vector<float> axialProjections(n, std::numeric_limits<float>::max());

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
                    axialProjections[i] = dot3(pos, nozzleFrame.nozzle_axis);
                }
            }
        });

    float minInteriorAxialProj = std::numeric_limits<float>::max();
    for (std::size_t i = 0; i < n; ++i) {
        if (domain.isInterior[i] && domain.sdf[i] <= 0.0f) {
            minInteriorAxialProj = std::min(minInteriorAxialProj, axialProjections[i]);
        }
    }
    if (minInteriorAxialProj >= 0.5f * kInfinityTau) {
        for (std::size_t i = 0; i < n; ++i) {
            if (domain.isInterior[i]) {
                minInteriorAxialProj = std::min(minInteriorAxialProj, axialProjections[i]);
            }
        }
    }
    if (minInteriorAxialProj < 0.5f * kInfinityTau) {
        domain.minZBed = minInteriorAxialProj;
    }

    for (std::size_t i = 0; i < n; ++i) {
        if (!domain.isInterior[i]) {
            continue;
        }
        const float hAxial = axialProjections[i] - domain.minZBed;
        if (hAxial <= seedHeightCutoff) {
            domain.isSeed[i] = 1;
            domain.tau[i] = std::max(0.0f, hAxial);
        }
    }

    // Build O(N) counting-sort hyperplane wavefront schedules for parallel TBB sweeping
    domain.buildWavefrontSchedules();

    // 2. Step A: Unconstrained Eikonal Fast Sweeping (|\nabla \tau| = 1 from bed seeds)
    notifyProgress(0.32f, "Solving initial unconstrained 3D Eikonal wavefronts...");
    runInteriorFastSweeping(domain, 2, /*useDirectionalGuidance=*/false);

    // Ensure any thin-wall sections connected only diagonally (26-neighbors) or elevated
    // components are seeded and swept so zero interior voxels remain at `kInfinityTau`.
    bool hasUnreachedInterior = false;
    for (std::size_t i = 0; i < n; ++i) {
        if (domain.isInterior[i] && domain.tau[i] >= 0.5f * kInfinityTau) {
            hasUnreachedInterior = true;
            break;
        }
    }
    if (hasUnreachedInterior && domain.indexGrid) {
        auto idxAcc = domain.indexGrid->getConstAccessor();
        for (int bridgePass = 0; bridgePass < 4; ++bridgePass) {
            bool bridgedAny = false;
            for (std::size_t i = 0; i < n; ++i) {
                if (!domain.isInterior[i] || domain.tau[i] < 0.5f * kInfinityTau) {
                    continue;
                }
                const openvdb::Coord& c = domain.coords[i];
                float bestBridgeTau = kInfinityTau;
                for (int dz = -1; dz <= 1; ++dz) {
                    for (int dy = -1; dy <= 1; ++dy) {
                        for (int dxOff = -1; dxOff <= 1; ++dxOff) {
                            if (dxOff == 0 && dy == 0 && dz == 0) {
                                continue;
                            }
                            const std::int32_t nb =
                                idxAcc.getValue(c + openvdb::Coord(dxOff, dy, dz));
                            if (nb < 0) {
                                continue;
                            }
                            const std::size_t u = static_cast<std::size_t>(nb);
                            if (domain.isInterior[u] && domain.tau[u] < 0.5f * kInfinityTau) {
                                const float stepDist = dx * std::sqrt(static_cast<float>(
                                    dxOff * dxOff + dy * dy + dz * dz));
                                bestBridgeTau = std::min(bestBridgeTau, domain.tau[u] + stepDist);
                            }
                        }
                    }
                }
                if (bestBridgeTau < 0.5f * kInfinityTau) {
                    domain.isSeed[i] = 1;
                    domain.tau[i] = bestBridgeTau;
                    bridgedAny = true;
                }
            }
            if (!bridgedAny) {
                break;
            }
            runInteriorFastSweeping(domain, 1, /*useDirectionalGuidance=*/false);
        }

        // Fallback for any disconnected elevated interior component: seed by axial height
        bool fallbackSeeded = false;
        for (std::size_t i = 0; i < n; ++i) {
            if (domain.isInterior[i] && domain.tau[i] >= 0.5f * kInfinityTau) {
                const float hAxial = std::max(0.0f, axialProjections[i] - domain.minZBed);
                domain.isSeed[i] = 1;
                domain.tau[i] = hAxial;
                fallbackSeeded = true;
            }
        }
        if (fallbackSeeded) {
            runInteriorFastSweeping(domain, 1, /*useDirectionalGuidance=*/false);
        }
    }

    ToolOrientationStrategy orientationStrategy;
    OverhangIslandConstraint overhangConstraint;
    OrientationHintConstraint hintConstraint;
    ToolClearanceConstraint clearanceConstraint;

    EikonalSolverContext iterCtx = baseContext;
    iterCtx.outerIteration = 0;
    overhangConstraint.enforceOnField(domain, iterCtx);

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
            ": Evaluating orientation hints & 5/6-axis tool axes...";
        notifyProgress(iterBaseFrac, msgOrient.c_str());

        // 3a. Initialize targetDir from \nabla \tau + overhang bias, then diffuse soft orientation hints
        overhangConstraint.applyPreSweepModifiers(domain, iterCtx);
        hintConstraint.applyPreSweepModifiers(domain, iterCtx);

        // 3b. Update 5/6-axis tool orientation (clamped strictly to tool rotational angle limits)
        orientationStrategy.updateToolOrientations(domain, iterCtx);

        // 3c. Strictly clamp targetDir (including any hints) to the tool's clearance cone
        //     centered on the ACTUAL solved tool axis `domain.toolAxis[i]`
        clearanceConstraint.applyPreSweepModifiers(domain, iterCtx);

        const std::string msgSweep =
            "Iteration " + std::to_string(iter + 1) + "/" + std::to_string(totalIters) +
            ": Parallel Eikonal sweeping & hard clearance enforcement...";
        notifyProgress(iterBaseFrac + 0.5f * iterSpan, msgSweep.c_str());

        // 3d. Directional wavefront integration + soft hint refinement on \tau
        runInteriorFastSweeping(domain, 2, /*useDirectionalGuidance=*/true);
        hintConstraint.enforceOnField(domain, iterCtx);

        // 3e. Hard constraints always run last so soft hints can never violate clearance or causality
        clearanceConstraint.enforceOnField(domain, iterCtx);
        overhangConstraint.enforceOnField(domain, iterCtx);
    }

    notifyProgress(0.88f, "Finalizing tool orientations & anti-island causality...");
    orientationStrategy.updateToolOrientations(domain, iterCtx);
    clearanceConstraint.enforceOnField(domain, iterCtx);
    overhangConstraint.enforceOnField(domain, iterCtx);

    // 4. Smoothly extend \tau(x) into the exterior narrow band in parallel
    notifyProgress(0.92f, "Extending Eikonal field into exterior narrow band...");
    extendEikonalToExteriorBand(domain, baseContext.backend);

    // 5. Pack solved fields into OpenVDB grids
    notifyProgress(0.96f, "Packing OpenVDB scalar & vector grids...");
    const float spanZ = std::max(objectMaxZ - objectMinZ, 1.0f);
    float maxSolvedTau = spanZ;
    for (std::size_t i = 0; i < n; ++i) {
        if (domain.tau[i] < 0.5f * kInfinityTau) {
            maxSolvedTau = std::max(maxSolvedTau, domain.tau[i]);
        }
    }
    const float eikonalBg = maxSolvedTau + 2.0f * exteriorBandMm + 10.0f;

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

EikonalSolveResult solveFineShellWithCoarseGuidance(
    const openvdb::FloatGrid& fineShellSdfGrid,
    const EikonalSolveResult& coarseSolved,
    float innerShellDepthMm,
    float objectMinZ,
    float objectMaxZ,
    float objectMaxSpan,
    float exteriorBandMm,
    const EikonalSolverContext& baseContext)
{
    if (!coarseSolved.eikonalGrid || !coarseSolved.toolAxisGrid) {
        return solveConstrainedEikonal(
            fineShellSdfGrid,
            objectMinZ,
            objectMaxZ,
            objectMaxSpan,
            exteriorBandMm,
            baseContext);
    }

    auto notifyProgress = [&](float frac, const char* msg) {
        if (baseContext.reportProgress) {
            baseContext.reportProgress(frac, msg);
        }
    };

    notifyProgress(0.55f, "Initializing fine boundary-shell domain from coarse Eikonal solution...");

    ActiveVolumeDomain domain;
    domain.initializeFromSdfGrid(fineShellSdfGrid, objectMinZ, objectMaxZ, objectMaxSpan);

    const std::size_t n = domain.coords.size();
    const float dx = static_cast<float>(domain.voxelSize);
    // 0.90 * dx exceeds the 3D cubic lattice covering radius (\sqrt{3}/2 * dx \approx 0.866 * dx),
    // guaranteeing that any thin curved wall at arbitrary 3D orientation forms a 6-connected
    // interior voxel network.
    const float interiorCutoff = 0.90f * dx;
    const float seedHeightCutoff = 1.50f * dx;
    const float pinDepthThreshold = -std::max(innerShellDepthMm - 1.75f * dx, 2.0f * dx);
    const float coarseValidLimit =
        coarseSolved.eikonalGrid->background() - std::max(0.5f * dx, 0.5f);

    std::vector<float> axialProjections(n, std::numeric_limits<float>::max());
    std::vector<float> coarseTauSampled(n, -1.0f);

    tbb::parallel_for(
        tbb::blocked_range<std::size_t>(0, n, 512),
        [&](const tbb::blocked_range<std::size_t>& r) {
            auto coarseEikAcc = coarseSolved.eikonalGrid->getConstAccessor();
            auto coarseAxisAcc = coarseSolved.toolAxisGrid->getConstAccessor();
            openvdb::tools::GridSampler<
                openvdb::FloatGrid::ConstAccessor,
                openvdb::tools::BoxSampler>
                coarseEikSampler(coarseEikAcc, coarseSolved.eikonalGrid->transform());
            openvdb::tools::GridSampler<
                openvdb::Vec3SGrid::ConstAccessor,
                openvdb::tools::BoxSampler>
                coarseAxisSampler(coarseAxisAcc, coarseSolved.toolAxisGrid->transform());

            for (std::size_t i = r.begin(); i != r.end(); ++i) {
                const auto& pos = domain.posBed[i];
                const openvdb::Vec3d wp(pos[0], pos[1], pos[2]);

                const NozzleBasis nozzleFrame =
                    baseContext.backend.queryNozzleFrame(pos, {1.0f, 0.0f, 0.0f});
                const std::array<float, 3> gravityDir =
                    baseContext.backend.queryGravityDirection(pos);

                const float antiGravityProj = -dot3(domain.sdfGrad[i], gravityDir);
                domain.overhang[i] = std::clamp(antiGravityProj, -1.0f, 1.0f);

                const openvdb::Vec3s sampledAxis = coarseAxisSampler.wsSample(wp);
                const std::array<float, 3> unitAxis = normalize3(
                    {sampledAxis.x(), sampledAxis.y(), sampledAxis.z()},
                    nozzleFrame.nozzle_axis);
                domain.toolAxis[i] = unitAxis;
                domain.targetDir[i] = unitAxis;

                const float cosTilt =
                    std::clamp(dot3(unitAxis, nozzleFrame.nozzle_axis), -1.0f, 1.0f);
                domain.toolTiltDeg[i] = std::acos(cosTilt) * kRadToDeg;

                const float cTauRaw = coarseEikSampler.wsSample(wp);
                const bool hasValidCoarseTau =
                    std::isfinite(cTauRaw) && (cTauRaw < coarseValidLimit);
                const float cTau = hasValidCoarseTau ? std::max(0.0f, cTauRaw) : -1.0f;
                coarseTauSampled[i] = cTau;

                const bool insideOrBoundary = (domain.sdf[i] <= interiorCutoff);
                domain.isInterior[i] = insideOrBoundary ? 1 : 0;
                if (insideOrBoundary) {
                    domain.tau[i] = hasValidCoarseTau ? cTau : kInfinityTau;
                    axialProjections[i] = dot3(pos, nozzleFrame.nozzle_axis);
                }
            }
        });

    for (std::size_t i = 0; i < n; ++i) {
        if (!domain.isInterior[i]) {
            continue;
        }
        const float hAxial = axialProjections[i] - domain.minZBed;
        if (hAxial <= seedHeightCutoff) {
            domain.isSeed[i] = 1;
            domain.tau[i] = std::max(0.0f, hAxial);
            continue;
        }

        // Voxels along the inner shell-core interface border the deep interior where fine voxels
        // are inactive; pin them to the coarse Eikonal solution so the fine shell smoothly matches
        // the coarse core without sweeping the deep interior at fine resolution.
        if (domain.sdf[i] < -1.5f * dx && coarseTauSampled[i] >= 0.0f) {
            bool bordersInactiveCore = (domain.sdf[i] <= pinDepthThreshold);
            if (!bordersInactiveCore) {
                for (const std::int32_t nb : domain.neighbors6[i]) {
                    if (nb < 0) {
                        bordersInactiveCore = true;
                        break;
                    }
                }
            }
            if (bordersInactiveCore) {
                domain.isSeed[i] = 1;
                domain.isPinnedSeed[i] = 1;
                domain.tau[i] = coarseTauSampled[i];
            }
        }
    }

    domain.buildWavefrontSchedules();

    // Propagate arrival time into any fine-only features (such as thin fins or ribs attached
    // to a thick core) that were narrower than the coarse grid resolution.
    runInteriorFastSweeping(domain, 2, /*useDirectionalGuidance=*/false);

    ToolOrientationStrategy orientationStrategy;
    OverhangIslandConstraint overhangConstraint;
    OrientationHintConstraint hintConstraint;
    ToolClearanceConstraint clearanceConstraint;

    EikonalSolverContext iterCtx = baseContext;
    const std::uint32_t fineIters =
        std::clamp<std::uint32_t>( std::min<std::uint32_t>(baseContext.totalIterations, 2), 1, 4);

    for (std::uint32_t iter = 0; iter < fineIters; ++iter) {
        iterCtx.outerIteration = iter;
        iterCtx.totalIterations = fineIters;

        const float frac =
            0.64f + 0.22f * (static_cast<float>(iter) / static_cast<float>(fineIters));
        notifyProgress(frac, "Refining fine boundary-shell Eikonal field & surface orientations...");

        overhangConstraint.applyPreSweepModifiers(domain, iterCtx);
        hintConstraint.applyPreSweepModifiers(domain, iterCtx);
        orientationStrategy.updateToolOrientations(domain, iterCtx);
        clearanceConstraint.applyPreSweepModifiers(domain, iterCtx);

        runInteriorFastSweeping(domain, 1, /*useDirectionalGuidance=*/true);
        hintConstraint.enforceOnField(domain, iterCtx);
        clearanceConstraint.enforceOnField(domain, iterCtx);
        overhangConstraint.enforceOnField(domain, iterCtx);
    }

    // Smoothly blend \tau across the inner transition zone into `coarseTauSampled` so the
    // gradient across the shell-core interface is C^1-continuous.
    const float blendStartDepth = std::max(innerShellDepthMm - 3.5f * dx, 1.5f * dx);
    const float blendSpan = std::max(innerShellDepthMm - blendStartDepth, dx);
    tbb::parallel_for(
        tbb::blocked_range<std::size_t>(0, n, 512),
        [&](const tbb::blocked_range<std::size_t>& r) {
            for (std::size_t i = r.begin(); i != r.end(); ++i) {
                if (!domain.isInterior[i] ||
                    domain.tau[i] >= 0.5f * kInfinityTau ||
                    coarseTauSampled[i] < 0.0f) {
                    continue;
                }
                const float depth = -domain.sdf[i];
                if (depth > blendStartDepth) {
                    const float u = std::clamp((depth - blendStartDepth) / blendSpan, 0.0f, 1.0f);
                    const float wCoarse = u * u * (3.0f - 2.0f * u);
                    domain.tau[i] =
                        (1.0f - wCoarse) * domain.tau[i] + wCoarse * coarseTauSampled[i];
                }
            }
        });

    notifyProgress(0.90f, "Extending fine shell Eikonal field into exterior narrow band...");
    extendEikonalToExteriorBand(domain, baseContext.backend);

    notifyProgress(0.96f, "Packing fine shell OpenVDB scalar & vector grids...");
    const float spanZ = std::max(objectMaxZ - objectMinZ, 1.0f);
    float maxSolvedTau = spanZ;
    for (std::size_t i = 0; i < n; ++i) {
        if (domain.tau[i] < 0.5f * kInfinityTau) {
            maxSolvedTau = std::max(maxSolvedTau, domain.tau[i]);
        }
    }
    const float eikonalBg =
        std::max(maxSolvedTau + 2.0f * exteriorBandMm + 10.0f, coarseSolved.eikonalGrid->background());

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

    notifyProgress(1.0f, "Multi-resolution field computation complete.");

    EikonalSolveResult result;
    result.eikonalGrid = eikonalGrid;
    result.overhangGrid = overhangGrid;
    result.toolTiltDegGrid = toolTiltGrid;
    result.toolAxisGrid = toolAxisGrid;
    return result;
}

} // namespace manifold
