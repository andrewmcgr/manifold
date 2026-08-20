//! Grid-based Fast Marching Method (FMM) [`OrderField`]: [`EikonalOrderField`]
//! discretizes a bounding box into a regular grid, seeds a front from a set
//! of seed points, and marches outward with a heap-based narrow-band FMM
//! solve to approximate (an isosurface-monotonic, grid-discretized
//! approximation of) Euclidean/geodesic distance from the front.
//!
//! v1 uses a uniform propagation speed (`speed(p) == 1.0` everywhere), so
//! [`EikonalOrderField::new`] reduces to grid-discretized Euclidean distance
//! from the seed set. [`EikonalOrderField::new_with_occupancy`] additionally
//! restricts the march to a caller-supplied solid region, so distance is
//! measured *through material* rather than straight-line through free space
//! — without that restriction, a point on a wall can look deceptively close
//! to a seed across an open air gap, producing isosurfaces that climb the
//! outside of the wall rather than following a printable in-material path.
//! The point of landing the FMM machinery itself (grid construction,
//! heap-based marching, front seeding, occupancy gating) now is so a later
//! phase can swap in genuine non-uniform `speed(p)` shaping without
//! re-architecting.
//!
//! This module is deliberately decoupled from any mesh type: callers supply
//! a bounding box, a seed point set, and (for the occupancy-aware
//! constructor) an opaque `is_solid(p) -> bool` classifier directly, so it
//! can be reused for a mesh-derived contact-surface front or any other seed
//! source (e.g. a synthetic test geometry) without this module depending on
//! `manifold-core`'s or even `manifold-fidget`'s own mesh types.
use crate::height_along::HeightAlong;
use crate::order::OrderField;
use crate::slope_profile::SlopeProfile;
use glam::DVec3;
use rayon::prelude::*;
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
    /// narrow-band FMM with uniform speed 1.0, with every grid node treated
    /// as traversable (see [`EikonalOrderField::new_with_occupancy`] for a
    /// version that restricts the march to a solid region).
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
        Self::new_with_occupancy(min_corner, max_corner, seeds, requested_cell_size, &|_| {
            true
        })
    }

    /// Like [`EikonalOrderField::new`], but restricts the march to grid
    /// nodes for which `is_solid(node_position)` returns `true` — the front
    /// can only propagate through solid material, never take a shortcut
    /// through free space around/above the object.
    ///
    /// Without this restriction (i.e. [`EikonalOrderField::new`], which
    /// passes `|_| true`), `order(p)` reduces to plain Euclidean distance
    /// from the seed set, which can make a point high up on a thin wall
    /// look *close* to a build-plate seed across an open air gap — the
    /// resulting isosurface then climbs the outside of that wall, a path no
    /// real toolpath can follow. Restricting relaxation to solid-classified
    /// nodes forces the measured distance to be an in-material path length,
    /// eliminating that shortcut and making `order` far closer to
    /// monotonically increasing with build height within the object (see
    /// [`crate::order::OrderField`]'s well-posedness discussion — this
    /// remains a discretized approximation, not an exact guarantee, since
    /// grid resolution and thin/non-manifold features can still admit
    /// shortcuts one cell wide).
    ///
    /// Nodes classified non-solid never get a finite distance (they are
    /// simply excluded from the march, including as seeds); trilinear
    /// interpolation of [`EikonalOrderField::order`] at a query point whose
    /// surrounding cell touches a non-solid node can therefore itself
    /// return `f64::INFINITY` — expected right at/outside the solid
    /// boundary, and a documented best-effort limitation for query points
    /// inside material thinner than one grid cell.
    ///
    /// `is_solid` is called once per grid node (up to millions of times for
    /// a large grid) and run in parallel via `rayon`, so it must be `Sync`
    /// as well as `Fn` — a mesh-backed classifier (e.g. sampling a spatial
    /// SDF/BVH per node) is exactly the case this matters for: run
    /// single-threaded, that classification pass alone can dominate wall
    /// time on a large grid.
    pub fn new_with_occupancy(
        min_corner: DVec3,
        max_corner: DVec3,
        seeds: &[DVec3],
        requested_cell_size: f64,
        is_solid: &(dyn Fn(DVec3) -> bool + Sync),
    ) -> Self {
        let (mut field, occupied) =
            Self::build_grid(min_corner, max_corner, requested_cell_size, is_solid);
        field.march_from_seeds(seeds, &occupied);
        field
    }

    /// Like [`EikonalOrderField::new_with_occupancy`], but instead of
    /// seeding from a caller-supplied list of points snapped to their
    /// nearest grid node, freezes *every* solid-classified grid node for
    /// which `is_seed_region(node_position)` returns `true` directly, at
    /// exact distance `0.0`.
    ///
    /// This exists because seeding from a sparse point list only ever
    /// seeds wherever those points happen to land — if the underlying
    /// geometry's own vertices sit on the *boundary* of the region meant
    /// to be the front's starting line (e.g. a flat base face
    /// triangulated with vertices only along its silhouette, as CAD
    /// exporters commonly do — a quad base has only 4 corner vertices),
    /// the seeded front traces that boundary outline, not the region's
    /// interior. The un-seeded interior nodes then have to march in from
    /// that boundary like every other node, so a point already resting
    /// flat on the build plate reports a small but nonzero order instead
    /// of the `0.0` its already-in-contact geometry deserves — the base
    /// layer then reads as an eroded/rounded version of the true
    /// footprint. Seeding by region instead — marking every occupied node
    /// within the contact band as a seed, independent of where the mesh's
    /// own vertices happen to sit — fills the interior of each connected
    /// component of that footprint uniformly, at the grid's own
    /// resolution rather than the mesh's triangulation density.
    pub fn new_with_occupancy_and_seed_region(
        min_corner: DVec3,
        max_corner: DVec3,
        requested_cell_size: f64,
        is_solid: &(dyn Fn(DVec3) -> bool + Sync),
        is_seed_region: &(dyn Fn(DVec3) -> bool + Sync),
    ) -> Self {
        Self::new_with_occupancy_and_seed_region_and_slope_limit(
            min_corner,
            max_corner,
            requested_cell_size,
            is_solid,
            is_seed_region,
            None,
            None,
        )
    }

    /// Like [`EikonalOrderField::new_with_occupancy_and_seed_region`], but
    /// additionally accepts an optional `slope_profile` +
    /// `height_along` pair: when both are `Some`, a grade-limiting
    /// (Lipschitz-extension-style) relaxation pass runs after the FMM
    /// march via [`EikonalOrderField::relax_with_slope_limit`], enforcing
    /// `|T(p) - T(q)| <= max_slope_at(height_along(p)) * h` for every pair
    /// of grid-adjacent nodes. Passing `None` for either argument is a
    /// purely additive, behavior-preserving no-op — identical to
    /// [`EikonalOrderField::new_with_occupancy_and_seed_region`].
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_occupancy_and_seed_region_and_slope_limit(
        min_corner: DVec3,
        max_corner: DVec3,
        requested_cell_size: f64,
        is_solid: &(dyn Fn(DVec3) -> bool + Sync),
        is_seed_region: &(dyn Fn(DVec3) -> bool + Sync),
        slope_profile: Option<&SlopeProfile>,
        height_along: Option<&dyn HeightAlong>,
    ) -> Self {
        let (mut field, occupied) =
            Self::build_grid(min_corner, max_corner, requested_cell_size, is_solid);
        field.march_from_region(is_seed_region, &occupied);
        if let (Some(profile), Some(height_along)) = (slope_profile, height_along) {
            field.relax_with_slope_limit(profile, height_along, &occupied);
        }
        field
    }

    /// Shared grid construction + occupancy classification behind both
    /// [`EikonalOrderField::new_with_occupancy`] and
    /// [`EikonalOrderField::new_with_occupancy_and_seed_region`]: builds an
    /// all-`f64::INFINITY` distance grid over `[min_corner, max_corner]`
    /// and classifies every node's occupancy via `is_solid`, without
    /// seeding a front yet — callers freeze their own initial seed
    /// distances into the returned field before marching.
    fn build_grid(
        min_corner: DVec3,
        max_corner: DVec3,
        requested_cell_size: f64,
        is_solid: &(dyn Fn(DVec3) -> bool + Sync),
    ) -> (Self, Vec<bool>) {
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

        let field = EikonalOrderField {
            min_corner: lo,
            dims,
            h: cell_size,
            distances,
        };

        let [nx, ny, _nz] = dims;
        // Classify every node in parallel: each node's position is
        // independent of every other, so this is embarrassingly parallel
        // and `is_solid` (often a mesh SDF/BVH query) is exactly the
        // expensive-per-call case that benefits most.
        let occupied: Vec<bool> = (0..total)
            .into_par_iter()
            .map(|idx| {
                let z = idx / (nx * ny);
                let y = (idx / nx) % ny;
                let x = idx % nx;
                is_solid(field.node_pos(x, y, z))
            })
            .collect();

        (field, occupied)
    }

    fn idx(&self, x: usize, y: usize, z: usize) -> usize {
        x + y * self.dims[0] + z * self.dims[0] * self.dims[1]
    }

    fn node_pos(&self, x: usize, y: usize, z: usize) -> DVec3 {
        self.min_corner + DVec3::new(x as f64 * self.h, y as f64 * self.h, z as f64 * self.h)
    }

    /// Seeds the front (nearest grid node to each seed point, frozen with
    /// its exact distance to that seed) and runs the heap-based narrow-band
    /// FMM march via [`EikonalOrderField::relax_from_frozen`].
    ///
    /// `occupied[idx]` (indexed via [`EikonalOrderField::idx`]) gates both
    /// seeding and relaxation: a seed snapping to a non-solid node is
    /// dropped rather than frozen, and relaxation never assigns a finite
    /// distance to a non-solid neighbor — so the front only ever
    /// propagates through solid-classified nodes (see
    /// [`EikonalOrderField::new_with_occupancy`]). Passing an all-`true`
    /// mask (as [`EikonalOrderField::new`] does) recovers the original
    /// unconstrained free-space march.
    fn march_from_seeds(&mut self, seeds: &[DVec3], occupied: &[bool]) {
        if seeds.is_empty() || self.distances.is_empty() {
            // Documented best-effort fallback: no front to march from,
            // leave every node at its initial `f64::INFINITY`.
            return;
        }

        let [nx, ny, nz] = self.dims;

        // Seed initialization: snap each seed to its nearest grid node and
        // freeze that node with the exact seed-to-node distance (taking the
        // min if multiple seeds map to the same node). A seed snapping to a
        // non-solid node is dropped: it has nothing solid to seed from.
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
            if !occupied[idx] {
                continue;
            }
            let dist = seed.distance(self.node_pos(x, y, z));
            if dist < self.distances[idx] {
                self.distances[idx] = dist;
            }
        }

        self.relax_from_frozen(occupied);
    }

    /// Like [`EikonalOrderField::march_from_seeds`], but freezes every
    /// occupied node matching `is_seed_region` directly at exact distance
    /// `0.0` instead of snapping a sparse point list — see
    /// [`EikonalOrderField::new_with_occupancy_and_seed_region`]'s doc for
    /// why this fills a region's interior rather than just tracing its
    /// boundary.
    fn march_from_region(
        &mut self,
        is_seed_region: &(dyn Fn(DVec3) -> bool + Sync),
        occupied: &[bool],
    ) {
        if self.distances.is_empty() {
            return;
        }

        let [nx, ny, _nz] = self.dims;
        let total = self.distances.len();
        // Classification is independent per node, same rationale as the
        // occupancy pass in `EikonalOrderField::build_grid`.
        let seeded: Vec<bool> = (0..total)
            .into_par_iter()
            .map(|idx| {
                if !occupied[idx] {
                    return false;
                }
                let z = idx / (nx * ny);
                let y = (idx / nx) % ny;
                let x = idx % nx;
                is_seed_region(self.node_pos(x, y, z))
            })
            .collect();

        for (idx, &is_seed) in seeded.iter().enumerate() {
            if is_seed {
                self.distances[idx] = 0.0;
            }
        }

        self.relax_from_frozen(occupied);
    }

    /// Shared second half of the FMM march behind both
    /// [`EikonalOrderField::march_from_seeds`] and
    /// [`EikonalOrderField::march_from_region`]: given `self.distances`
    /// already holding whatever initial seed distances the caller has
    /// frozen in (everywhere else still `f64::INFINITY`), builds the
    /// initial heap from those frozen nodes and runs the heap-based
    /// narrow-band march to fill in the rest, gated by `occupied`.
    fn relax_from_frozen(&mut self, occupied: &[bool]) {
        let [nx, ny, nz] = self.dims;
        let mut frozen = vec![false; self.distances.len()];
        let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::new();

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
                if frozen[nidx] || !occupied[nidx] {
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

    /// Grade-limiting (Lipschitz-extension-style) post-process, run after
    /// the FMM march has already completed: enforces
    /// `|T(p) - T(q)| <= max_slope_at(height_along(p)) * horizontal_distance(p, q)`
    /// for every pair of grid-adjacent nodes `p`, `q` with nonzero
    /// horizontal displacement (4-connectivity within the horizontal
    /// plane, via [`NEIGHBOR_OFFSETS`] excluding its pure-Z pair;
    /// `horizontal_distance` for these axis-neighbors is just `self.h`).
    /// The pure-vertical neighbor pair is deliberately left unconstrained:
    /// it has zero horizontal displacement, so a horizontal slope limit is
    /// vacuous for it -- constraining it would throttle straight-up
    /// progression even under a maximally tight (near-flat) profile,
    /// starving reachable order values far below the mesh's true height.
    ///
    /// This is a second, independent heap-based label-correcting
    /// relaxation pass over the *entire* finite-distance field (unlike
    /// [`EikonalOrderField::relax_from_frozen`], which seeds only from
    /// nodes frozen so far during the FMM march itself) — it does not call
    /// [`EikonalOrderField::solve_update`]/[`solve_eikonal_quadratic`],
    /// since this is a post-hoc clamp on an already-computed isotropic
    /// distance field, not a re-solve of the Eikonal equation. Popping node
    /// `p` with distance `T(p)`, each occupied grid-neighbor `q` is capped
    /// at `T(p) + slope_multiplier * self.h`; if `q`'s current distance
    /// exceeds that cap, it is lowered and `q` is pushed back onto the
    /// heap. Repeated relaxation from both directions (each node acts as
    /// `p` for its neighbors as it's popped) converges to a fixpoint that
    /// respects the Lipschitz bound symmetrically.
    ///
    /// Degenerate `profile`/`height_along` input never corrupts
    /// `self.distances` with NaN: a `NaN` height (via
    /// [`HeightAlong::height`]) or a `max_slope_at` result within
    /// floating-point epsilon of 90 degrees (near-vertical, i.e.
    /// effectively unconstrained — see [`SlopeProfile`]'s
    /// `UNCONSTRAINED_ANGLE_DEG`) skips relaxation from that node entirely,
    /// rather than letting `tan(90 deg)` or a NaN slope multiplier flow
    /// into the distance field.
    fn relax_with_slope_limit(
        &mut self,
        profile: &SlopeProfile,
        height_along: &dyn HeightAlong,
        occupied: &[bool],
    ) {
        let [nx, ny, nz] = self.dims;
        let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::new();

        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let idx = self.idx(x, y, z);
                    if occupied[idx] && self.distances[idx].is_finite() {
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

        // Angles at/above this threshold are treated as "no limit" to
        // avoid wastefully large (but technically finite) `tan()` values
        // near the 90-degree asymptote.
        const NEAR_VERTICAL_ANGLE_DEG: f64 = 89.999;

        while let Some(HeapEntry { value, x, y, z }) = heap.pop() {
            let idx = self.idx(x, y, z);
            if !occupied[idx] || self.distances[idx] < value {
                // Stale heap entry superseded by a smaller value already
                // recorded for this node; skip.
                continue;
            }
            let t_p = self.distances[idx];

            let height = height_along.height(self.node_pos(x, y, z));
            if height.is_nan() {
                // Degenerate height projection: treat as unconstrained
                // (skip relaxation from this node) rather than letting NaN
                // propagate into `max_slope_at`/the tan() computation.
                continue;
            }
            let max_angle = profile.max_slope_at(height);
            if !max_angle.is_finite() || max_angle >= NEAR_VERTICAL_ANGLE_DEG {
                // Unconstrained (or empty-profile default): no cap to
                // enforce from this node.
                continue;
            }
            let slope_multiplier = max_angle.to_radians().tan();
            if !slope_multiplier.is_finite() {
                continue;
            }

            for (dx, dy, dz) in NEIGHBOR_OFFSETS {
                if dx == 0 && dy == 0 {
                    // Pure-vertical neighbor: zero horizontal displacement,
                    // so a *horizontal* slope limit is vacuous for this
                    // pair -- capping it here would throttle straight-up
                    // progression even for a maximally tight (near-flat)
                    // profile, starving reachable order values far below
                    // the mesh's true height and collapsing most of the
                    // object into a handful of widely-spaced layers with
                    // near-vertical isosurface jumps to compensate. Leave
                    // vertical progression unconstrained by this pass,
                    // same as ordinary flat/planar slicing.
                    continue;
                }
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
                if !occupied[nidx] {
                    continue;
                }

                let candidate = t_p + slope_multiplier * self.h;
                if candidate.is_nan() {
                    continue;
                }
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
    /// reached remain `f64::INFINITY`; those corners are *excluded* from
    /// the interpolation (weights renormalized over the finite corners) so
    /// a query point inside the reached solid isn't poisoned by a void
    /// node just outside the surface — the common case for mesh-surface
    /// queries, where up to half of a cell's corners sit in unmarched air.
    /// Only when *every* corner of the containing cell is unreached (e.g.
    /// an empty seed set, or a genuinely unreached region) does this
    /// return `+inf` — the documented best-effort fallback for a query
    /// point the front cannot reach.
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

        // Trilinear weights per corner; unreached (`+inf`) corners are
        // dropped and the remaining weights renormalized. See doc comment.
        let corners = [
            (g(x0, y0, z0), (1.0 - tx) * (1.0 - ty) * (1.0 - tz)),
            (g(x1, y0, z0), tx * (1.0 - ty) * (1.0 - tz)),
            (g(x0, y1, z0), (1.0 - tx) * ty * (1.0 - tz)),
            (g(x1, y1, z0), tx * ty * (1.0 - tz)),
            (g(x0, y0, z1), (1.0 - tx) * (1.0 - ty) * tz),
            (g(x1, y0, z1), tx * (1.0 - ty) * tz),
            (g(x0, y1, z1), (1.0 - tx) * ty * tz),
            (g(x1, y1, z1), tx * ty * tz),
        ];

        let mut weighted_sum = 0.0;
        let mut weight_total = 0.0;
        for (value, weight) in corners {
            if value.is_finite() {
                weighted_sum += value * weight;
                weight_total += weight;
            }
        }

        if weight_total > 0.0 {
            weighted_sum / weight_total
        } else {
            // Every corner with any interpolation weight is unreached:
            // the front genuinely never got here.
            f64::INFINITY
        }
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

    /// Regression test for a real bug: `order`'s trilinear interpolation
    /// blends a reached (finite) grid node with an unreached (`+inf`)
    /// neighbor at the edge of the marched/occupied region — the ordinary
    /// case with `new_with_occupancy`, not a rare corner case. An early
    /// version produced `inf + (finite - inf) * t` = NaN here, which then
    /// flowed into contour reconstruction and made `i_overlay` panic with
    /// "trying to convert a point[NaN, NaN]". Querying at that reached/
    /// unreached boundary must return a usable value (now: the finite-
    /// corner-weighted interpolation), never NaN.
    #[test]
    fn order_at_reached_unreached_boundary_is_infinite_not_nan() {
        let min_corner = DVec3::new(-1.0, -1.0, -1.0);
        let max_corner = DVec3::new(1.0, 1.0, 1.0);
        let seeds = [DVec3::new(-1.0, -1.0, -1.0)];
        // Only the half of the grid with x <= 0 is "solid"/traversable; the
        // other half is never reached and stays at `f64::INFINITY`. A query
        // point straddling that boundary is exactly the mixed
        // finite/infinite trilinear-interpolation case.
        let is_solid = |p: DVec3| p.x <= 0.0;
        let field =
            EikonalOrderField::new_with_occupancy(min_corner, max_corner, &seeds, 0.25, &is_solid);
        let value = field.order(DVec3::new(0.0, -0.9, -0.9));
        assert!(
            !value.is_nan(),
            "order() at a reached/unreached grid boundary must never be NaN"
        );
    }

    /// Regression test for missing walls on meshes with thin features
    /// (reported on Voron_Design_Cube_v7.stl): a query point *inside* the
    /// reached solid but within one cell of the surface has some of its
    /// eight interpolation corners in unmarched air (`+inf`). The old
    /// interpolation propagated `+inf` whenever *any* corner was
    /// unreached, so large bands of real surface vertices got infinite
    /// order and their walls were silently dropped. Finite corners must
    /// win: only a cell whose corners are *all* unreached may return
    /// `+inf`.
    #[test]
    fn order_near_surface_ignores_unreached_void_corners() {
        let min_corner = DVec3::new(-1.0, -1.0, -1.0);
        let max_corner = DVec3::new(1.0, 1.0, 1.0);
        let seeds = [DVec3::new(-1.0, -1.0, -1.0)];
        // Solid half-space x <= 0; everything x > 0 is unmarched air.
        let is_solid = |p: DVec3| p.x <= 0.0;
        let field =
            EikonalOrderField::new_with_occupancy(min_corner, max_corner, &seeds, 0.25, &is_solid);

        // Just inside the solid, but the containing cell straddles the
        // surface so its x = 0.25 corners are unreached.
        let near_surface = field.order(DVec3::new(-0.05, -0.9, -0.9));
        assert!(
            near_surface.is_finite(),
            "query inside reached solid near the surface must be finite, got {near_surface}"
        );

        // Deep inside the unreached air, every corner is `+inf`: the
        // documented "front cannot reach this point" fallback still holds.
        let in_air = field.order(DVec3::new(0.75, 0.75, 0.75));
        assert!(
            in_air.is_infinite(),
            "query in a fully-unreached region must stay +inf, got {in_air}"
        );
    }

    /// Regression test for "Eikonal seeds only the boundary of the contact
    /// footprint, not its interior": seeding a flat square region from only
    /// its four corner points (mimicking a CAD-triangulated quad base with
    /// no interior vertices) leaves the center of that square with a
    /// nonzero order (it has to march in from the corners), while seeding
    /// the same region via [`EikonalOrderField::new_with_occupancy_and_seed_region`]
    /// freezes every occupied node in the region — including its center —
    /// at exact distance `0.0`.
    #[test]
    fn region_seeding_fills_the_interior_unlike_sparse_corner_point_seeding() {
        let min_corner = DVec3::new(-5.0, -5.0, -5.0);
        let max_corner = DVec3::new(5.0, 5.0, 5.0);
        let cell_size = 0.5;
        // The whole grid is "solid"/traversable; only the z <= -4.0 slab is
        // the "contact region" a base footprint would occupy.
        let is_solid = |_p: DVec3| true;
        let is_seed_region = |p: DVec3| p.z <= -4.0;
        let center = DVec3::new(0.0, 0.0, -4.5);

        let corner_seeded = EikonalOrderField::new_with_occupancy(
            min_corner,
            max_corner,
            &[
                DVec3::new(-5.0, -5.0, -4.5),
                DVec3::new(5.0, -5.0, -4.5),
                DVec3::new(-5.0, 5.0, -4.5),
                DVec3::new(5.0, 5.0, -4.5),
            ],
            cell_size,
            &is_solid,
        );
        let region_seeded = EikonalOrderField::new_with_occupancy_and_seed_region(
            min_corner,
            max_corner,
            cell_size,
            &is_solid,
            &is_seed_region,
        );

        let corner_center_order = corner_seeded.order(center);
        let region_center_order = region_seeded.order(center);

        assert!(
            corner_center_order > 1.0,
            "sparse corner-point seeding should leave the region's center a real distance \
             from the nearest seed corner, got {corner_center_order}"
        );
        assert!(
            approx_eq(region_center_order, 0.0, 1e-6),
            "region seeding should freeze every occupied node in the seed region \
             (including its center) at exact distance 0.0, got {region_center_order}"
        );
    }

    /// A steep slope profile applied to a single-point seed should cap the
    /// distance field far from the seed to the Lipschitz bound implied by
    /// the profile's max angle, well below the unconstrained (isotropic)
    /// Euclidean-ish distance the plain FMM march would otherwise produce.
    #[test]
    fn steep_slope_profile_caps_plateau_distance_from_seed() {
        use crate::height_along::ConstantAxisHeight;
        use crate::slope_profile::SlopeProfile;

        let min_corner = DVec3::new(-5.0, -5.0, -5.0);
        let max_corner = DVec3::new(5.0, 5.0, 5.0);
        let cell_size = 0.25;
        let is_solid = |_p: DVec3| true;
        let is_seed_region = |p: DVec3| p == DVec3::ZERO;

        // A tight 30-degree cap, constant everywhere.
        let profile = SlopeProfile::new(vec![(f64::INFINITY, 30.0)]);
        let height_along = ConstantAxisHeight::new(DVec3::Z, DVec3::ZERO);

        let field = EikonalOrderField::new_with_occupancy_and_seed_region_and_slope_limit(
            min_corner,
            max_corner,
            cell_size,
            &is_solid,
            &is_seed_region,
            Some(&profile),
            Some(&height_along),
        );

        // A far point on the same z-plane as the seed: purely horizontal
        // distance is large (many cells), but height along the axis from
        // the seed is ~0, so the Lipschitz bound from the seed itself is
        // tiny (slope_multiplier * horizontal_distance, with height delta
        // ~0 contributing no extra allowance beyond the per-hop cap
        // chaining outward from the seed). Since the cap chains hop by hop
        // at `tan(30deg) * h` per hop regardless of height, the bound after
        // `n` hops of horizontal distance `n * h` is `n * h * tan(30deg)`.
        let far = DVec3::new(4.0, 0.0, 0.0);
        let hops =
            (far.x - min_corner.x).abs() / cell_size - (0.0 - min_corner.x).abs() / cell_size;
        let n = (far.x / cell_size).round().abs();
        let bound = n * cell_size * 30.0_f64.to_radians().tan() + 1e-6;
        let got = field.order(far);

        assert!(
            got.is_finite(),
            "expected finite capped distance, got {got}"
        );
        assert!(
            got <= bound + cell_size, // small slack for grid snapping/interpolation
            "expected distance <= slope-limited bound ~{bound}, got {got} (hops arg unused: {hops})"
        );
    }

    /// Regression test for a bug where the slope-limit relaxation applied
    /// its horizontal cap to the pure-vertical (Z) neighbor too, throttling
    /// straight-up progression by the same tiny per-hop allowance as
    /// horizontal movement. Under an extremely tight profile (near-flat,
    /// e.g. a few degrees), that collapsed most of a tall column's true
    /// height into a small reachable order range. A point directly above
    /// the seed (zero horizontal displacement) must be able to reach a
    /// distance approximately equal to its full Euclidean/axis distance
    /// from the seed, unconstrained by the horizontal slope cap.
    #[test]
    fn pure_vertical_progression_is_not_throttled_by_a_tight_slope_profile() {
        use crate::height_along::ConstantAxisHeight;
        use crate::slope_profile::SlopeProfile;

        let min_corner = DVec3::new(-1.0, -1.0, 0.0);
        let max_corner = DVec3::new(1.0, 1.0, 20.0);
        let cell_size = 0.5;
        let is_solid = |_p: DVec3| true;
        let is_seed_region = |p: DVec3| p == DVec3::new(0.0, 0.0, 0.0);

        // An extremely tight 1-degree cap, constant everywhere -- the
        // near-flat extreme that most aggressively exposed the bug.
        let profile = SlopeProfile::new(vec![(f64::INFINITY, 1.0)]);
        let height_along = ConstantAxisHeight::new(DVec3::Z, DVec3::ZERO);

        let field = EikonalOrderField::new_with_occupancy_and_seed_region_and_slope_limit(
            min_corner,
            max_corner,
            cell_size,
            &is_solid,
            &is_seed_region,
            Some(&profile),
            Some(&height_along),
        );

        // Directly above the seed: zero horizontal displacement, so the
        // pre-fix bug would have capped this to `tan(1deg) * (z / cell_size) * cell_size`
        // (a tiny fraction of `z`), instead of the true ~z distance.
        let above = DVec3::new(0.0, 0.0, 18.0);
        let got = field.order(above);

        assert!(
            got.is_finite(),
            "expected finite distance directly above the seed, got {got}"
        );
        assert!(
            got > above.z - cell_size * 2.0,
            "pure-vertical progression should reach ~its full axis distance ({}) from the seed \
             unthrottled by the horizontal slope cap, got {got}",
            above.z
        );
    }

    /// Passing `None` for both `slope_profile` and `height_along` must
    /// leave behavior identical to the existing unconstrained
    /// `new_with_occupancy_and_seed_region` march (regression safety).
    #[test]
    fn no_slope_profile_matches_unconstrained_march() {
        let min_corner = DVec3::new(-3.0, -3.0, -3.0);
        let max_corner = DVec3::new(3.0, 3.0, 3.0);
        let cell_size = 0.5;
        let is_solid = |_p: DVec3| true;
        let is_seed_region = |p: DVec3| p == DVec3::ZERO;

        let baseline = EikonalOrderField::new_with_occupancy_and_seed_region(
            min_corner,
            max_corner,
            cell_size,
            &is_solid,
            &is_seed_region,
        );
        let via_new_api = EikonalOrderField::new_with_occupancy_and_seed_region_and_slope_limit(
            min_corner,
            max_corner,
            cell_size,
            &is_solid,
            &is_seed_region,
            None,
            None,
        );

        for p in [
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(0.0, 2.0, 0.0),
            DVec3::new(1.0, 1.0, 1.0),
        ] {
            assert!(approx_eq(baseline.order(p), via_new_api.order(p), 1e-9));
        }
    }

    /// A `HeightAlong` implementation returning NaN must not panic and must
    /// not introduce NaN into the distance field.
    #[test]
    fn nan_height_along_does_not_corrupt_distances() {
        struct NanHeight;
        impl HeightAlong for NanHeight {
            fn height(&self, _p: DVec3) -> f64 {
                f64::NAN
            }
        }

        let min_corner = DVec3::new(-2.0, -2.0, -2.0);
        let max_corner = DVec3::new(2.0, 2.0, 2.0);
        let cell_size = 0.5;
        let is_solid = |_p: DVec3| true;
        let is_seed_region = |p: DVec3| p == DVec3::ZERO;
        let profile = SlopeProfile::new(vec![(f64::INFINITY, 30.0)]);
        let height_along = NanHeight;

        let field = EikonalOrderField::new_with_occupancy_and_seed_region_and_slope_limit(
            min_corner,
            max_corner,
            cell_size,
            &is_solid,
            &is_seed_region,
            Some(&profile),
            Some(&height_along),
        );

        for p in [
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(0.0, 1.5, 0.0),
            DVec3::new(-1.0, -1.0, -1.0),
        ] {
            let value = field.order(p);
            assert!(
                !value.is_nan(),
                "expected no NaN in distance field, got NaN at {p:?}"
            );
        }
    }

    /// An empty-breakpoints profile is unconstrained per
    /// `SlopeProfile::max_slope_at`, so it should be a no-op relative to
    /// passing `None` (within float tolerance).
    #[test]
    fn empty_profile_is_a_no_op() {
        use crate::height_along::ConstantAxisHeight;

        let min_corner = DVec3::new(-2.0, -2.0, -2.0);
        let max_corner = DVec3::new(2.0, 2.0, 2.0);
        let cell_size = 0.5;
        let is_solid = |_p: DVec3| true;
        let is_seed_region = |p: DVec3| p == DVec3::ZERO;
        let empty_profile = SlopeProfile::new(vec![]);
        let height_along = ConstantAxisHeight::new(DVec3::Z, DVec3::ZERO);

        let baseline = EikonalOrderField::new_with_occupancy_and_seed_region(
            min_corner,
            max_corner,
            cell_size,
            &is_solid,
            &is_seed_region,
        );
        let with_empty_profile =
            EikonalOrderField::new_with_occupancy_and_seed_region_and_slope_limit(
                min_corner,
                max_corner,
                cell_size,
                &is_solid,
                &is_seed_region,
                Some(&empty_profile),
                Some(&height_along),
            );

        for p in [
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(0.0, 1.5, 0.0),
            DVec3::new(-1.0, -1.0, -1.0),
        ] {
            assert!(approx_eq(
                baseline.order(p),
                with_empty_profile.order(p),
                1e-6
            ));
        }
    }

    /// End-to-end test: a single-point seed with a very tight (5-degree)
    /// slope profile.
    ///
    /// First establishes the test isn't vacuous: the plain unconstrained
    /// march (no slope-limiting pass) has an axis-neighbor delta right
    /// next to the seed of ~`h` (isotropic speed 1.0), which trivially
    /// exceeds the tight cap `tan(5deg) * h`. Then confirms that building
    /// the *same* grid/seed/profile via
    /// `new_with_occupancy_and_seed_region_and_slope_limit` makes every
    /// grid-adjacent node pair *with nonzero horizontal displacement*
    /// (x- and y-neighbors) in the whole field respect
    /// `|T(p) - T(q)| <= max_slope_at(height_along(p)) * h`. The
    /// pure-vertical (Z) neighbor is deliberately excluded from this check
    /// -- it's intentionally left unconstrained by the relaxation (see
    /// `pure_vertical_progression_is_not_throttled_by_a_tight_slope_profile`),
    /// since a horizontal slope limit is vacuous for a pair with zero
    /// horizontal displacement.
    #[test]
    fn steep_synthetic_field_is_clamped_by_slope_profile_relaxation() {
        use crate::height_along::ConstantAxisHeight;
        use crate::slope_profile::SlopeProfile;

        let min_corner = DVec3::new(-3.0, -3.0, -3.0);
        let max_corner = DVec3::new(3.0, 3.0, 3.0);
        let cell_size = 0.5;
        let is_solid = |_p: DVec3| true;
        let is_seed_region = |p: DVec3| p == DVec3::ZERO;

        // A very tight 5-degree cap, constant everywhere (so the bound is
        // spatially uniform regardless of which of a pair's two nodes it's
        // evaluated at).
        let angle_deg = 5.0;
        let profile = SlopeProfile::new(vec![(f64::INFINITY, angle_deg)]);
        let height_along = ConstantAxisHeight::new(DVec3::Z, DVec3::ZERO);
        let max_delta = angle_deg.to_radians().tan() * cell_size;

        // Baseline: the plain unconstrained march, with no slope-limiting
        // pass applied at all.
        let baseline = EikonalOrderField::new_with_occupancy_and_seed_region(
            min_corner,
            max_corner,
            cell_size,
            &is_solid,
            &is_seed_region,
        );

        // Establish the violation is real (not a vacuous pass): right next
        // to the seed, the unconstrained march's axis-neighbor delta is
        // ~`h` (isotropic speed 1.0), far exceeding the tight cap.
        let seed = DVec3::ZERO;
        let neighbor = DVec3::new(cell_size, 0.0, 0.0);
        let baseline_delta = (baseline.order(neighbor) - baseline.order(seed)).abs();
        assert!(
            baseline_delta > max_delta,
            "expected unconstrained march to violate the tight slope cap: delta {baseline_delta} > cap {max_delta} did not hold"
        );

        // Now build the same grid/seed/profile via the slope-limiting
        // relaxation pass and confirm every grid-adjacent node pair in the
        // field respects the Lipschitz bound.
        let limited = EikonalOrderField::new_with_occupancy_and_seed_region_and_slope_limit(
            min_corner,
            max_corner,
            cell_size,
            &is_solid,
            &is_seed_region,
            Some(&profile),
            Some(&height_along),
        );

        let epsilon = 1e-6;
        let [nx, ny, nz] = limited.dims;
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let idx = limited.idx(x, y, z);
                    let t_p = limited.distances[idx];
                    if !t_p.is_finite() {
                        continue;
                    }
                    let p = limited.node_pos(x, y, z);
                    let bound = profile
                        .max_slope_at(height_along.height(p))
                        .to_radians()
                        .tan()
                        * limited.h
                        + epsilon;

                    for (dx, dy, dz) in [(1isize, 0isize, 0isize), (0, 1, 0)] {
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
                        let nidx = limited.idx(nxp as usize, nyp as usize, nzp as usize);
                        let t_q = limited.distances[nidx];
                        if !t_q.is_finite() {
                            continue;
                        }
                        let delta = (t_p - t_q).abs();
                        assert!(
                            delta <= bound,
                            "slope cap violated at {p:?}: delta {delta} > bound {bound}"
                        );
                    }
                }
            }
        }
    }
}
