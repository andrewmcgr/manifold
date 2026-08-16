//! Grid-based Fast Marching Method (FMM) [`OrderField`]: [`EikonalOrderField`]
//! discretizes a bounding box into a regular grid, seeds a front from a set
//! of seed points, and marches outward with a heap-based narrow-band FMM
//! solve to approximate (an isosurface-monotonic, grid-discretized
//! approximation of) Euclidean/geodesic distance from the front.
//!
//! v1 uses a uniform propagation speed (`speed(p) == 1.0` everywhere), so
//! this reduces to grid-discretized Euclidean distance from the seed set —
//! the point of landing this now is the FMM machinery itself (grid
//! construction, heap-based marching, front seeding) so a later phase can
//! swap in genuine `speed(p)` shaping without re-architecting.
//!
//! This module is deliberately decoupled from any mesh type: callers supply
//! a bounding box and a seed point set directly, so it can be reused for a
//! mesh-derived contact-surface front (a later phase) or any other seed
//! source (e.g. a synthetic test geometry) without this module depending on
//! `manifold-core`'s or even `manifold-fidget`'s own mesh types.

use crate::order::OrderField;
use glam::DVec3;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// A [`OrderField`] backed by a grid-based narrow-band Fast Marching Method
/// solve: `order(p)` is a trilinearly-interpolated lookup into a
/// precomputed distance grid, seeded from `seeds` and marched outward with
/// uniform speed 1.0.
///
/// Degenerate cases (empty `seeds`, or a query point the front never
/// reaches because it is graph-disconnected on the grid) are a documented
/// best-effort fallback: unreached grid nodes keep their initial
/// `f64::INFINITY` value, so [`EikonalOrderField::order`] can return
/// `f64::INFINITY` (or an interpolated blend that is still `+inf`) rather
/// than panicking — never `unwrap()`/panic on degenerate geometry input.
pub struct EikonalOrderField {
    min_corner: DVec3,
    /// Node counts along x/y/z (at least 1 per axis).
    dims: [usize; 3],
    /// Uniform grid spacing (same for all three axes).
    h: f64,
    /// Distance value per grid node, row-major with x fastest, then y, then
    /// z: `index = x + y * dims[0] + z * dims[0] * dims[1]`.
    distances: Vec<f64>,
}

impl EikonalOrderField {
    /// Builds an [`EikonalOrderField`] over the axis-aligned bounding box
    /// `[min_corner, max_corner]` (corners are normalized component-wise, so
    /// swapped min/max components are tolerated rather than producing a
    /// degenerate/negative-extent grid), seeding the front from `seeds`
    /// (each snapped to its nearest grid node) and marching outward via
    /// narrow-band FMM with uniform speed 1.0.
    ///
    /// `requested_cell_size` drives grid resolution automatically (no
    /// separate hardcoded resolution constant): grid node counts along each
    /// axis are `ceil(extent / requested_cell_size) + 1`. Non-positive or
    /// non-finite `requested_cell_size` falls back to a resolution derived
    /// from the box's largest extent (`max_extent / 10.0`, or `1.0` for a
    /// degenerate point-like box) so construction never panics on bad
    /// input.
    ///
    /// An empty `seeds` slice produces a grid with every node left at
    /// `f64::INFINITY` (no front to march from) — a documented best-effort
    /// fallback rather than a panic; see [`EikonalOrderField::order`].
    pub fn new(
        min_corner: DVec3,
        max_corner: DVec3,
        seeds: &[DVec3],
        requested_cell_size: f64,
    ) -> Self {
        let lo = min_corner.min(max_corner);
        let hi = min_corner.max(max_corner);
        let extent = hi - lo;
        let max_extent = extent.x.max(extent.y).max(extent.z);

        let cell_size = if requested_cell_size.is_finite() && requested_cell_size > 0.0 {
            requested_cell_size
        } else if max_extent > 0.0 {
            max_extent / 10.0
        } else {
            1.0
        };

        let dims = [
            grid_node_count(extent.x, cell_size),
            grid_node_count(extent.y, cell_size),
            grid_node_count(extent.z, cell_size),
        ];

        let total = dims[0] * dims[1] * dims[2];
        let distances = vec![f64::INFINITY; total];

        let mut field = EikonalOrderField {
            min_corner: lo,
            dims,
            h: cell_size,
            distances,
        };

        field.march_from_seeds(seeds);
        field
    }

    fn idx(&self, x: usize, y: usize, z: usize) -> usize {
        x + y * self.dims[0] + z * self.dims[0] * self.dims[1]
    }

    fn node_pos(&self, x: usize, y: usize, z: usize) -> DVec3 {
        self.min_corner + DVec3::new(x as f64 * self.h, y as f64 * self.h, z as f64 * self.h)
    }

    /// Seeds the front (nearest grid node to each seed point, frozen with
    /// its exact distance to that seed) and runs the heap-based narrow-band
    /// FMM march, writing results into `self.distances`.
    fn march_from_seeds(&mut self, seeds: &[DVec3]) {
        if seeds.is_empty() || self.distances.is_empty() {
            // Documented best-effort fallback: no front to march from,
            // leave every node at its initial `f64::INFINITY`.
            return;
        }

        let [nx, ny, nz] = self.dims;
        let mut frozen = vec![false; self.distances.len()];
        let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::new();

        // Seed initialization: snap each seed to its nearest grid node and
        // freeze that node with the exact seed-to-node distance (taking the
        // min if multiple seeds map to the same node).
        for &seed in seeds {
            let fx = ((seed.x - self.min_corner.x) / self.h).round();
            let fy = ((seed.y - self.min_corner.y) / self.h).round();
            let fz = ((seed.z - self.min_corner.z) / self.h).round();
            if !fx.is_finite() || !fy.is_finite() || !fz.is_finite() {
                // Degenerate seed coordinate (e.g. NaN/inf input): skip
                // rather than panic on the `as usize` cast below.
                continue;
            }
            let x = clamp_index(fx, nx);
            let y = clamp_index(fy, ny);
            let z = clamp_index(fz, nz);
            let idx = self.idx(x, y, z);
            let dist = seed.distance(self.node_pos(x, y, z));
            if dist < self.distances[idx] {
                self.distances[idx] = dist;
            }
        }

        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let idx = self.idx(x, y, z);
                    if self.distances[idx].is_finite() {
                        frozen[idx] = true;
                        heap.push(HeapEntry {
                            value: self.distances[idx],
                            x,
                            y,
                            z,
                        });
                    }
                }
            }
        }

        // Narrow-band FMM march: repeatedly pop the smallest tentative
        // value, freeze it (skip if it was already frozen via a cheaper
        // path found later), and relax its unfrozen neighbors.
        while let Some(HeapEntry { value, x, y, z }) = heap.pop() {
            let idx = self.idx(x, y, z);
            if frozen[idx] && self.distances[idx] < value {
                // Stale heap entry superseded by a smaller value already
                // frozen for this node; skip.
                continue;
            }
            frozen[idx] = true;

            for (dx, dy, dz) in NEIGHBOR_OFFSETS {
                let nxp = x as isize + dx;
                let nyp = y as isize + dy;
                let nzp = z as isize + dz;
                if nxp < 0
                    || nyp < 0
                    || nzp < 0
                    || nxp as usize >= nx
                    || nyp as usize >= ny
                    || nzp as usize >= nz
                {
                    continue;
                }
                let (nxu, nyu, nzu) = (nxp as usize, nyp as usize, nzp as usize);
                let nidx = self.idx(nxu, nyu, nzu);
                if frozen[nidx] {
                    continue;
                }

                let candidate = self.solve_update(nxu, nyu, nzu, &frozen);
                if candidate < self.distances[nidx] {
                    self.distances[nidx] = candidate;
                    heap.push(HeapEntry {
                        value: candidate,
                        x: nxu,
                        y: nyu,
                        z: nzu,
                    });
                }
            }
        }
    }

    /// Solves the local Eikonal update at grid node `(x, y, z)` from its
    /// already-frozen axis-neighbors, using the standard Godunov upwind FMM
    /// scheme for uniform speed 1.0 and uniform grid spacing `self.h`.
    fn solve_update(&self, x: usize, y: usize, z: usize, frozen: &[bool]) -> f64 {
        let [nx, ny, nz] = self.dims;

        let mut axis_mins: Vec<f64> = Vec::with_capacity(3);

        let mut push_axis_min = |a: Option<f64>, b: Option<f64>| {
            let m = match (a, b) {
                (Some(a), Some(b)) => a.min(b),
                (Some(a), None) => a,
                (None, Some(b)) => b,
                (None, None) => return,
            };
            axis_mins.push(m);
        };

        let val_at = |x: usize, y: usize, z: usize| -> Option<f64> {
            let idx = self.idx(x, y, z);
            frozen[idx].then(|| self.distances[idx])
        };

        push_axis_min(
            x.checked_sub(1).and_then(|xm| val_at(xm, y, z)),
            (x + 1 < nx).then(|| val_at(x + 1, y, z)).flatten(),
        );
        push_axis_min(
            y.checked_sub(1).and_then(|ym| val_at(x, ym, z)),
            (y + 1 < ny).then(|| val_at(x, y + 1, z)).flatten(),
        );
        push_axis_min(
            z.checked_sub(1).and_then(|zm| val_at(x, y, zm)),
            (z + 1 < nz).then(|| val_at(x, y, z + 1)).flatten(),
        );

        solve_eikonal_quadratic(&axis_mins, self.h)
    }
}

impl OrderField for EikonalOrderField {
    /// Trilinearly interpolates the precomputed FMM distance grid at `p`
    /// (clamped into the grid's bounding box). Nodes the front never
    /// reached remain `f64::INFINITY`; interpolating between such nodes (or
    /// a fully-unreached region, e.g. an empty seed set) yields `+inf`
    /// rather than panicking — this is the documented best-effort fallback
    /// for a query point the front cannot reach.
    fn order(&self, p: DVec3) -> f64 {
        let [nx, ny, nz] = self.dims;
        if nx == 0 || ny == 0 || nz == 0 {
            // Degenerate (empty) grid: nothing to interpolate.
            return f64::INFINITY;
        }

        let local = (p - self.min_corner) / self.h;
        let (x0, tx) = axis_coords(local.x, nx);
        let (y0, ty) = axis_coords(local.y, ny);
        let (z0, tz) = axis_coords(local.z, nz);
        let x1 = (x0 + 1).min(nx - 1);
        let y1 = (y0 + 1).min(ny - 1);
        let z1 = (z0 + 1).min(nz - 1);

        let g = |x: usize, y: usize, z: usize| self.distances[self.idx(x, y, z)];

        let c000 = g(x0, y0, z0);
        let c100 = g(x1, y0, z0);
        let c010 = g(x0, y1, z0);
        let c110 = g(x1, y1, z0);
        let c001 = g(x0, y0, z1);
        let c101 = g(x1, y0, z1);
        let c011 = g(x0, y1, z1);
        let c111 = g(x1, y1, z1);

        let c00 = lerp(c000, c100, tx);
        let c10 = lerp(c010, c110, tx);
        let c01 = lerp(c001, c101, tx);
        let c11 = lerp(c011, c111, tx);

        let c0 = lerp(c00, c10, ty);
        let c1 = lerp(c01, c11, ty);

        lerp(c0, c1, tz)
    }
}

/// Number of grid nodes along one axis, given that axis's extent and the
/// (already-validated positive/finite) cell size: `ceil(extent / cell_size)
/// + 1`, minimum 1 (a zero-extent axis is still a single valid node/plane).
fn grid_node_count(extent: f64, cell_size: f64) -> usize {
    if extent <= 0.0 {
        1
    } else {
        (extent / cell_size).ceil() as usize + 1
    }
}

/// Clamps a rounded floating-point grid coordinate into `[0, count - 1]`.
fn clamp_index(v: f64, count: usize) -> usize {
    if count == 0 {
        return 0;
    }
    let max = (count - 1) as f64;
    v.clamp(0.0, max) as usize
}

/// Splits a continuous (already `/h`-scaled) local axis coordinate into a
/// base node index and an interpolation fraction in `[0, 1]`, clamped to the
/// grid's valid range.
fn axis_coords(local: f64, count: usize) -> (usize, f64) {
    if count <= 1 {
        return (0, 0.0);
    }
    let max = (count - 1) as f64;
    let clamped = local.clamp(0.0, max);
    let base = clamped.floor();
    let base_idx = (base as usize).min(count - 2);
    let frac = (clamped - base_idx as f64).clamp(0.0, 1.0);
    (base_idx, frac)
}

fn lerp(a: f64, b: f64, t: f64) -> f64 {
    if a.is_infinite() && b.is_infinite() && a.signum() == b.signum() {
        // Avoid `inf * 0.0` producing NaN when blending two infinities of
        // the same sign (the common "front hasn't reached this region yet"
        // case) — the interpolated result is still meaningfully "+inf".
        return a;
    }
    a + (b - a) * t
}

/// Solves the standard Godunov upwind FMM quadratic update for uniform grid
/// spacing `h` and unit speed, given the per-axis minimal known
/// (already-frozen) neighbor values in `axis_mins` (0..=3 entries, one per
/// contributing axis).
///
/// Classic algorithm (Sethian): sort contributing axis values ascending,
/// try solving with an increasing number of axes `k = 1..=len`, and accept
/// the first `k` for which the causality condition `u <= a[k]` holds (or
/// `k` exhausts all axes) — this correctly falls back to a lower-dimensional
/// (fewer-axis) update when including a farther axis would violate
/// causality (the upwind condition).
fn solve_eikonal_quadratic(axis_mins: &[f64], h: f64) -> f64 {
    if axis_mins.is_empty() {
        // No frozen neighbor at all: nothing to solve from (should not
        // normally happen for a node the march actually visits, but this
        // is a documented best-effort fallback rather than a panic).
        return f64::INFINITY;
    }

    let mut a: Vec<f64> = axis_mins.to_vec();
    a.sort_by(|x, y| x.total_cmp(y));

    let n = a.len();
    for k in 1..=n {
        let sum_a: f64 = a[..k].iter().sum();
        let sum_a2: f64 = a[..k].iter().map(|v| v * v).sum();
        let kf = k as f64;
        let discriminant = sum_a * sum_a - kf * (sum_a2 - h * h);
        if discriminant < 0.0 {
            // No real solution with this many axes (can happen for
            // ill-conditioned/very-different neighbor values); fall back to
            // the single-axis estimate using the smallest neighbor.
            return a[0] + h;
        }
        let u = (sum_a + discriminant.sqrt()) / kf;
        if k == n || u <= a[k] {
            return u;
        }
    }

    // Unreachable in practice (the loop above always returns by k == n),
    // but keep a safe fallback rather than an unreachable!()/panic.
    a[0] + h
}

/// Axis-aligned grid-neighbor offsets (6-connectivity: +/-1 along each of
/// x/y/z).
const NEIGHBOR_OFFSETS: [(isize, isize, isize); 6] = [
    (-1, 0, 0),
    (1, 0, 0),
    (0, -1, 0),
    (0, 1, 0),
    (0, 0, -1),
    (0, 0, 1),
];

/// Min-heap entry for the FMM march: ordered by ascending `value` (reversed
/// `Ord` so `BinaryHeap`, a max-heap by default, pops the smallest value
/// first).
#[derive(Debug, Clone, Copy, PartialEq)]
struct HeapEntry {
    value: f64,
    x: usize,
    y: usize,
    z: usize,
}

impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        other.value.total_cmp(&self.value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn point_seed_reduces_to_approximate_euclidean_distance() {
        let min_corner = DVec3::new(-5.0, -5.0, -5.0);
        let max_corner = DVec3::new(5.0, 5.0, 5.0);
        let seeds = [DVec3::ZERO];
        let field = EikonalOrderField::new(min_corner, max_corner, &seeds, 0.1);

        for p in [
            DVec3::new(2.0, 0.0, 0.0),
            DVec3::new(0.0, 3.0, 0.0),
            DVec3::new(1.0, 1.0, 1.0),
            DVec3::new(-2.0, -1.0, 0.5),
        ] {
            let expected = p.length();
            let got = field.order(p);
            // Grid discretization + axis-aligned (Manhattan-biased) FMM
            // marching introduces some error relative to true Euclidean
            // distance; allow a generous relative tolerance.
            let tol = (expected * 0.15).max(0.3);
            assert!(
                approx_eq(got, expected, tol),
                "expected ~{expected}, got {got} at {p:?}"
            );
        }
    }

    #[test]
    fn planar_seed_reduces_to_approximate_distance_from_plane() {
        // Seed a line of points along the z=0 plane's x-axis to approximate
        // a planar front; order should grow with |z| (distance from the
        // plane), matching the monotonicity `reconstruct_on_order_field`
        // depends on.
        let min_corner = DVec3::new(-5.0, -5.0, -5.0);
        let max_corner = DVec3::new(5.0, 5.0, 5.0);
        let mut seeds = Vec::new();
        let mut x = -5.0;
        while x <= 5.0 {
            seeds.push(DVec3::new(x, 0.0, 0.0));
            x += 0.2;
        }
        let field = EikonalOrderField::new(min_corner, max_corner, &seeds, 0.1);

        let samples: Vec<f64> = [0.0, 0.5, 1.0, 2.0, 3.0]
            .iter()
            .map(|&z| field.order(DVec3::new(0.0, 0.0, z)))
            .collect();

        for window in samples.windows(2) {
            assert!(
                window[1] > window[0] - 1e-9,
                "expected non-decreasing order values with distance from the seed plane, got {samples:?}"
            );
        }
        assert!(approx_eq(samples[0], 0.0, 0.2));
    }

    #[test]
    fn empty_seed_set_never_panics_and_returns_infinity() {
        let min_corner = DVec3::new(0.0, 0.0, 0.0);
        let max_corner = DVec3::new(1.0, 1.0, 1.0);
        let field = EikonalOrderField::new(min_corner, max_corner, &[], 0.2);

        let value = field.order(DVec3::new(0.5, 0.5, 0.5));
        assert!(value.is_infinite());
    }

    #[test]
    fn degenerate_point_like_bbox_never_panics() {
        let corner = DVec3::new(1.0, 2.0, 3.0);
        let seeds = [corner];
        // min_corner == max_corner: zero-extent box on every axis.
        let field = EikonalOrderField::new(corner, corner, &seeds, 0.1);
        let value = field.order(corner);
        assert!(value.is_finite());
        assert!(approx_eq(value, 0.0, 1e-6));
    }

    #[test]
    fn non_positive_cell_size_falls_back_instead_of_panicking() {
        let min_corner = DVec3::new(-1.0, -1.0, -1.0);
        let max_corner = DVec3::new(1.0, 1.0, 1.0);
        let seeds = [DVec3::ZERO];
        let field = EikonalOrderField::new(min_corner, max_corner, &seeds, -1.0);
        let value = field.order(DVec3::new(0.5, 0.0, 0.0));
        assert!(value.is_finite());
    }

    #[test]
    fn query_point_outside_bbox_is_clamped_and_does_not_panic() {
        let min_corner = DVec3::new(-1.0, -1.0, -1.0);
        let max_corner = DVec3::new(1.0, 1.0, 1.0);
        let seeds = [DVec3::ZERO];
        let field = EikonalOrderField::new(min_corner, max_corner, &seeds, 0.25);
        let value = field.order(DVec3::new(100.0, 100.0, 100.0));
        assert!(value.is_finite());
    }
}
