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
    /// Per-node gradient estimate of `distances`, same indexing as
    /// `distances`. Computed once in [`EikonalOrderField::compute_gradients`]
    /// after every distance-modifying pass (FMM march, optional slope-limit
    /// relaxation) has finished, from one-sided finite differences against
    /// each axis's smaller-valued (upwind) frozen neighbor -- `0.0` on any
    /// axis with no frozen neighbor on either side. Used by `order()`'s
    /// Hermite interpolation path; left `DVec3::ZERO` at unreached nodes
    /// (never read there, since `order()` falls back to trilinear whenever
    /// any of its 8 corners is non-finite).
    gradients: Vec<DVec3>,
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
        field.compute_gradients(&occupied);
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
        if let Some(profile) = slope_profile {
            let default_height =
                crate::height_along::ConstantAxisHeight::new(glam::DVec3::Z, min_corner);
            let ha: &dyn HeightAlong = height_along.unwrap_or(&default_height);
            field.relax_with_slope_limit(profile, ha, &occupied);
        }
        field.compute_gradients(&occupied);
        field
    }

    /// Two-sided boundary-conforming Eikonal field: marches distances upward
    /// from the bed contact region (`is_bed_seed_region`) and downward from
    /// the top exterior surface (`is_top_seed_region`), and smoothly warps the
    /// top skin layers so they lie parallel to and conform with sloped and curved
    /// top surfaces, while keeping the bulk and base 100% strictly monotonic.
    #[allow(clippy::too_many_arguments)]
    pub fn new_conformal_with_occupancy_and_seed_regions_and_slope_limit(
        min_corner: DVec3,
        max_corner: DVec3,
        requested_cell_size: f64,
        is_solid: &(dyn Fn(DVec3) -> bool + Sync),
        is_bed_seed_region: &(dyn Fn(DVec3) -> bool + Sync),
        is_top_seed_region: &(dyn Fn(DVec3) -> bool + Sync),
        slope_profile: Option<&SlopeProfile>,
        height_along: Option<&dyn HeightAlong>,
    ) -> Self {
        let (mut bed_field, occupied) =
            Self::build_grid(min_corner, max_corner, requested_cell_size, is_solid);
        bed_field.march_from_region(is_bed_seed_region, &occupied);

        let (top_distances, nearest_surface_value) = bed_field.march_downward_with_surface_values(
            is_top_seed_region,
            &occupied,
            &bed_field.distances,
        );

        // Skin thickness: top ~2.0mm or 6 grid cells
        let skin_thickness = (bed_field.h * 6.0).max(1.0);

        for (idx, &is_occ) in occupied.iter().enumerate() {
            if !is_occ {
                bed_field.distances[idx] = f64::INFINITY;
                continue;
            }
            let d_top = top_distances[idx];
            let d_bed = bed_field.distances[idx];
            let surf_val = nearest_surface_value[idx];

            if d_top < skin_thickness && surf_val.is_finite() && d_bed.is_finite() {
                let skin_order = surf_val - d_top;
                let t = (d_top / skin_thickness).clamp(0.0, 1.0);
                let alpha = t * t * (3.0 - 2.0 * t);
                bed_field.distances[idx] = (1.0 - alpha) * skin_order + alpha * d_bed;
            }
        }

        if let Some(profile) = slope_profile {
            let default_height =
                crate::height_along::ConstantAxisHeight::new(glam::DVec3::Z, min_corner);
            let ha: &dyn HeightAlong = height_along.unwrap_or(&default_height);
            bed_field.relax_with_slope_limit(profile, ha, &occupied);
        }
        bed_field.compute_gradients(&occupied);
        bed_field
    }

    /// Helper for conformal top skin: marches downward from `is_top_seed_region`,
    /// propagating both the distance to the top surface and the `surface_values`
    /// of the nearest top seed.
    fn march_downward_with_surface_values(
        &self,
        is_top_seed_region: &(dyn Fn(DVec3) -> bool + Sync),
        occupied: &[bool],
        surface_values: &[f64],
    ) -> (Vec<f64>, Vec<f64>) {
        let total = self.distances.len();
        let mut top_distances = vec![f64::INFINITY; total];
        let mut nearest_surface_value = vec![f64::NAN; total];

        let [nx, ny, nz] = self.dims;
        let seeded: Vec<bool> = (0..total)
            .into_par_iter()
            .map(|idx| {
                if !occupied[idx] {
                    return false;
                }
                let z = idx / (nx * ny);
                let y = (idx / nx) % ny;
                let x = idx % nx;
                is_top_seed_region(self.node_pos(x, y, z))
            })
            .collect();

        let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::new();
        let mut frozen = vec![false; total];

        for (idx, &is_seed) in seeded.iter().enumerate() {
            if is_seed && surface_values[idx].is_finite() {
                top_distances[idx] = 0.0;
                nearest_surface_value[idx] = surface_values[idx];
            }
        }

        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let idx = self.idx(x, y, z);
                    if top_distances[idx].is_finite() {
                        frozen[idx] = true;
                        heap.push(HeapEntry {
                            value: top_distances[idx],
                            x,
                            y,
                            z,
                        });
                    }
                }
            }
        }

        while let Some(HeapEntry { value, x, y, z }) = heap.pop() {
            let idx = self.idx(x, y, z);
            if frozen[idx] && top_distances[idx] < value {
                continue;
            }
            frozen[idx] = true;
            let surf_val = nearest_surface_value[idx];

            for (dx, dy, dz) in NEIGHBOR_OFFSETS {
                let nxp = x as isize + dx;
                let nyp = y as isize + dy;
                let nzp = z as isize + dz;
                if nxp < 0
                    || nyp < 0
                    || nzp < 0
                    || nxp >= nx as isize
                    || nyp >= ny as isize
                    || nzp >= nz as isize
                {
                    continue;
                }
                let (nx_u, ny_u, nz_u) = (nxp as usize, nyp as usize, nzp as usize);
                let n_idx = self.idx(nx_u, ny_u, nz_u);
                if !occupied[n_idx] || frozen[n_idx] {
                    continue;
                }
                let next_dist = value + self.h;
                if next_dist < top_distances[n_idx] {
                    top_distances[n_idx] = next_dist;
                    nearest_surface_value[n_idx] = surf_val;
                    heap.push(HeapEntry {
                        value: next_dist,
                        x: nx_u,
                        y: ny_u,
                        z: nz_u,
                    });
                }
            }
        }

        (top_distances, nearest_surface_value)
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
        let gradients = vec![DVec3::ZERO; total];

        let field = EikonalOrderField {
            min_corner: lo,
            dims,
            h: cell_size,
            distances,
            gradients,
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
        _occupied: &[bool],
    ) {
        let [nx, ny, nz] = self.dims;
        let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::new();

        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let idx = self.idx(x, y, z);
                    if self.distances[idx].is_finite() {
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
            if self.distances[idx] < value {
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

    /// Computes a per-node gradient estimate of `self.distances`, stored in
    /// `self.gradients`, for use by `order()`'s Hermite interpolation path.
    /// Must run *after* every distance-modifying pass (FMM march, and the
    /// optional slope-limit relaxation) has finished -- it reads whatever
    /// `self.distances` holds at call time, so it reflects the final field
    /// rather than an intermediate march state.
    ///
    /// Per axis, uses a one-sided finite difference against whichever
    /// axis-neighbor is finite and smaller (the upwind side, matching the
    /// direction [`EikonalOrderField::solve_update`] itself propagates
    /// from), with the difference's sign depending on which side that
    /// neighbor is on: `(T_node - T_lo) / h` for a low-side neighbor,
    /// `(T_hi - T_node) / h` for a high-side one -- using the unconditional
    /// `(T_node - T_neighbor) / h` form regardless of side flips the
    /// gradient's sign whenever the high side is chosen. `0.0` on an axis
    /// where neither neighbor is finite. Only computed for occupied nodes
    /// with a finite distance --
    /// unreached nodes keep `DVec3::ZERO`, which is never read since
    /// `order()` falls back to plain trilinear whenever any of its 8
    /// corners is non-finite.
    fn compute_gradients(&mut self, occupied: &[bool]) {
        let [nx, ny, nz] = self.dims;
        let h = self.h;

        let val_at = |x: usize, y: usize, z: usize| -> f64 { self.distances[self.idx(x, y, z)] };

        // Direction-aware one-sided difference: whichever side's neighbor is
        // smaller (upwind) is used, but the formula's sign depends on which
        // side that neighbor is actually on -- a low-side neighbor gives a
        // forward estimate `(current - lo) / h`, while a high-side neighbor
        // gives `(hi - current) / h`. Using `(current - neighbor) / h`
        // unconditionally (regardless of side) silently flips the sign
        // whenever the high side is chosen, corrupting the Hermite tangent
        // direction for ~half of all nodes.
        let axis_component = |current: f64, lo: Option<f64>, hi: Option<f64>| -> f64 {
            match (lo, hi) {
                (Some(a), Some(b)) => {
                    if a <= b {
                        (current - a) / h
                    } else {
                        (b - current) / h
                    }
                }
                (Some(a), None) => (current - a) / h,
                (None, Some(b)) => (b - current) / h,
                (None, None) => 0.0,
            }
        };

        let mut gradients = vec![DVec3::ZERO; self.gradients.len()];
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let idx = self.idx(x, y, z);
                    if !occupied[idx] || !self.distances[idx].is_finite() {
                        continue;
                    }
                    let current = self.distances[idx];

                    let x_lo = x.checked_sub(1).and_then(|xm| {
                        let nidx = self.idx(xm, y, z);
                        (occupied[nidx] && self.distances[nidx].is_finite())
                            .then(|| val_at(xm, y, z))
                    });
                    let x_hi = (x + 1 < nx)
                        .then(|| self.idx(x + 1, y, z))
                        .and_then(|nidx| {
                            (occupied[nidx] && self.distances[nidx].is_finite())
                                .then(|| val_at(x + 1, y, z))
                        });

                    let y_lo = y.checked_sub(1).and_then(|ym| {
                        let nidx = self.idx(x, ym, z);
                        (occupied[nidx] && self.distances[nidx].is_finite())
                            .then(|| val_at(x, ym, z))
                    });
                    let y_hi = (y + 1 < ny)
                        .then(|| self.idx(x, y + 1, z))
                        .and_then(|nidx| {
                            (occupied[nidx] && self.distances[nidx].is_finite())
                                .then(|| val_at(x, y + 1, z))
                        });

                    let z_lo = z.checked_sub(1).and_then(|zm| {
                        let nidx = self.idx(x, y, zm);
                        (occupied[nidx] && self.distances[nidx].is_finite())
                            .then(|| val_at(x, y, zm))
                    });
                    let z_hi = (z + 1 < nz)
                        .then(|| self.idx(x, y, z + 1))
                        .and_then(|nidx| {
                            (occupied[nidx] && self.distances[nidx].is_finite())
                                .then(|| val_at(x, y, z + 1))
                        });

                    gradients[idx] = DVec3::new(
                        axis_component(current, x_lo, x_hi),
                        axis_component(current, y_lo, y_hi),
                        axis_component(current, z_lo, z_hi),
                    );
                }
            }
        }
        self.gradients = gradients;
    }
}

impl OrderField for EikonalOrderField {
    /// Interpolates the precomputed FMM distance grid at `p` (clamped into
    /// the grid's bounding box). Nodes the front never reached remain
    /// `f64::INFINITY`.
    ///
    /// Uses a gradient-augmented (Hermite) blend over the containing cell's
    /// 8 corners when every corner is finite: each corner carries both its
    /// distance value and a per-axis gradient estimate computed once in
    /// [`EikonalOrderField::compute_gradients`] from the FMM solve itself
    /// (physically meaningful, magnitude ~1 for a unit-speed Eikonal field)
    /// rather than inferred after the fact -- this is C1-continuous and, by
    /// construction, bounded (unlike a wider finite-difference cubic
    /// stencil, which can overshoot near thin features using only inferred
    /// derivatives). Falls back to plain trilinear (dropping unreached
    /// corners and renormalizing weights over the rest) whenever any of the
    /// 8 corners is non-finite -- the common case for queries near the
    /// surface, where up to half of a cell's corners sit in unmarched air.
    /// Only when *every* corner with nonzero trilinear weight is unreached
    /// (e.g. an empty seed set, or a genuinely unreached region) does this
    /// return `+inf` -- the documented best-effort fallback for a query
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
        let grad = |x: usize, y: usize, z: usize| self.gradients[self.idx(x, y, z)];

        let v000 = g(x0, y0, z0);
        let v100 = g(x1, y0, z0);
        let v010 = g(x0, y1, z0);
        let v110 = g(x1, y1, z0);
        let v001 = g(x0, y0, z1);
        let v101 = g(x1, y0, z1);
        let v011 = g(x0, y1, z1);
        let v111 = g(x1, y1, z1);

        if v000.is_finite()
            && v100.is_finite()
            && v010.is_finite()
            && v110.is_finite()
            && v001.is_finite()
            && v101.is_finite()
            && v011.is_finite()
            && v111.is_finite()
        {
            return hermite_trilinear(
                self.h,
                tx,
                ty,
                tz,
                [v000, v100, v010, v110, v001, v101, v011, v111],
                [
                    grad(x0, y0, z0),
                    grad(x1, y0, z0),
                    grad(x0, y1, z0),
                    grad(x1, y1, z0),
                    grad(x0, y0, z1),
                    grad(x1, y0, z1),
                    grad(x0, y1, z1),
                    grad(x1, y1, z1),
                ],
            );
        }

        // Fallback: plain trilinear, dropping unreached (`+inf`) corners and
        // renormalizing weights over the finite ones. See doc comment.
        let corners = [
            (v000, (1.0 - tx) * (1.0 - ty) * (1.0 - tz)),
            (v100, tx * (1.0 - ty) * (1.0 - tz)),
            (v010, (1.0 - tx) * ty * (1.0 - tz)),
            (v110, tx * ty * (1.0 - tz)),
            (v001, (1.0 - tx) * (1.0 - ty) * tz),
            (v101, tx * (1.0 - ty) * tz),
            (v011, (1.0 - tx) * ty * tz),
            (v111, tx * ty * tz),
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

/// Cubic Hermite basis functions `(h00, h10, h01, h11)` at `t in [0, 1]`.
/// `h00 + h01 == 1` always (partition of unity for the value terms); `h10`,
/// `h11` are bounded (peak magnitude ~0.1925 within `[0, 1]`), so a
/// derivative term `h10(t) * h * gradient` or `h11(t) * h * gradient` can't
/// blow up for a well-behaved (magnitude ~1) gradient -- unlike a
/// finite-difference-inferred cubic weight, which has no such bound.
fn hermite_weights(t: f64) -> (f64, f64, f64, f64) {
    let t2 = t * t;
    let t3 = t2 * t;
    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h10 = t3 - 2.0 * t2 + t;
    let h01 = -2.0 * t3 + 3.0 * t2;
    let h11 = t3 - t2;
    (h00, h10, h01, h11)
}

/// Gradient-augmented trilinear (Hermite) interpolation over a fully-finite
/// 8-corner cell. Corners are ordered `[000, 100, 010, 110, 001, 101, 011,
/// 111]` (x fastest, then y, then z), matching [`EikonalOrderField::order`]'s
/// corner construction.
///
/// Blends along z first (Hermite, using each corner's `z`-gradient
/// component as the paired derivative), then y (Hermite, using the
/// `y`-gradient component), then x (Hermite, using the `x`-gradient
/// component); the transverse gradient components not being blended at a
/// given stage are carried through via plain linear interpolation, since we
/// only have a directional derivative per axis, not the mixed partials a
/// full tricubic Hermite patch would need. This simplified scheme is
/// C1-continuous along axis-aligned directions and, since every stage's
/// weights are individually bounded (see [`hermite_weights`]), cannot
/// overshoot the way a finite-difference-derived cubic can.
fn hermite_trilinear(
    h: f64,
    tx: f64,
    ty: f64,
    tz: f64,
    values: [f64; 8],
    gradients: [DVec3; 8],
) -> f64 {
    let [v000, v100, v010, v110, v001, v101, v011, v111] = values;
    let [g000, g100, g010, g110, g001, g101, g011, g111] = gradients;

    let (h00z, h10z, h01z, h11z) = hermite_weights(tz);
    let hermite_z = |v0: f64, gz0: f64, v1: f64, gz1: f64| {
        h00z * v0 + h10z * h * gz0 + h01z * v1 + h11z * h * gz1
    };
    let lerp_z = |a: f64, b: f64| a + (b - a) * tz;

    // After the z-blend, each of the 4 (x, y) corner pairs carries a
    // blended value plus the transverse (x, y) gradient components carried
    // through via linear interpolation (see doc comment).
    let v_x0y0 = hermite_z(v000, g000.z, v001, g001.z);
    let gx_x0y0 = lerp_z(g000.x, g001.x);
    let gy_x0y0 = lerp_z(g000.y, g001.y);

    let v_x1y0 = hermite_z(v100, g100.z, v101, g101.z);
    let gx_x1y0 = lerp_z(g100.x, g101.x);
    let gy_x1y0 = lerp_z(g100.y, g101.y);

    let v_x0y1 = hermite_z(v010, g010.z, v011, g011.z);
    let gx_x0y1 = lerp_z(g010.x, g011.x);
    let gy_x0y1 = lerp_z(g010.y, g011.y);

    let v_x1y1 = hermite_z(v110, g110.z, v111, g111.z);
    let gx_x1y1 = lerp_z(g110.x, g111.x);
    let gy_x1y1 = lerp_z(g110.y, g111.y);

    let (h00y, h10y, h01y, h11y) = hermite_weights(ty);
    let hermite_y = |v0: f64, gy0: f64, v1: f64, gy1: f64| {
        h00y * v0 + h10y * h * gy0 + h01y * v1 + h11y * h * gy1
    };
    let lerp_y = |a: f64, b: f64| a + (b - a) * ty;

    let v_x0 = hermite_y(v_x0y0, gy_x0y0, v_x0y1, gy_x0y1);
    let gx_x0 = lerp_y(gx_x0y0, gx_x0y1);

    let v_x1 = hermite_y(v_x1y0, gy_x1y0, v_x1y1, gy_x1y1);
    let gx_x1 = lerp_y(gx_x1y0, gx_x1y1);

    let (h00x, h10x, h01x, h11x) = hermite_weights(tx);
    h00x * v_x0 + h10x * h * gx_x0 + h01x * v_x1 + h11x * h * gx_x1
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

    /// Regression test for a sign bug in `compute_gradients`'s one-sided
    /// finite difference: seeding from the *high*-x end means every
    /// interior node's smaller (upwind) neighbor sits on its `hi` side, the
    /// exact branch that used to compute `(current - neighbor) / h`
    /// unconditionally -- flipping the sign of the true gradient (which
    /// should point in -x, toward decreasing distance-to-seed) to +x. A
    /// wrong-signed tangent in `hermite_trilinear` bows the interpolated
    /// value the wrong way inside a cell, producing a local overshoot/dip
    /// on the order of one grid cell in size -- this asserts both the
    /// gradient's raw sign and that `order()` stays monotonic (no such
    /// overshoot) across a cell interior.
    #[test]
    fn gradient_sign_is_correct_when_upwind_neighbor_is_on_the_high_side() {
        let min_corner = DVec3::new(0.0, 0.0, 0.0);
        let max_corner = DVec3::new(1.0, 0.0, 0.0);
        // Seed near the high-x end: every interior node's smaller-valued
        // (upwind) axis-neighbor is therefore on its `hi` (x+1) side.
        let seeds = [DVec3::new(0.95, 0.0, 0.0)];
        let field = EikonalOrderField::new(min_corner, max_corner, &seeds, 0.1);

        let [nx, _, _] = field.dims;
        // Check an interior node (not the seed's own cell) directly: its raw
        // gradient x-component must be negative (distance decreases toward
        // the high-x seed), not positive as the sign-flipped bug produced.
        let mid = nx / 2;
        let idx = field.idx(mid, 0, 0);
        assert!(
            field.gradients[idx].x < 0.0,
            "expected negative x-gradient (field decreases toward the high-x seed), got {:?}",
            field.gradients[idx]
        );

        // Also check end-to-end: sampling densely across a cell interior
        // must stay monotonically non-increasing as x increases toward the
        // seed -- a wrong-signed tangent would bow the Hermite curve the
        // wrong way, producing a local increase (overshoot) inside the
        // cell.
        let mut previous = field.order(DVec3::new(0.05, 0.0, 0.0));
        let mut worst_increase = 0.0_f64;
        let mut x = 0.05;
        while x < 0.9 {
            x += 0.01;
            let value = field.order(DVec3::new(x, 0.0, 0.0));
            if value.is_finite() && previous.is_finite() {
                worst_increase = worst_increase.max(value - previous);
            }
            previous = value;
        }
        assert!(
            worst_increase < 1e-6,
            "order() increased by {worst_increase} while approaching the seed -- likely a wrong-signed gradient tangent"
        );
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

    #[test]
    fn conformal_eikonal_field_aligns_with_sloped_top_surface_and_flat_bottom() {
        let min_corner = DVec3::new(0.0, 0.0, 0.0);
        let max_corner = DVec3::new(10.0, 10.0, 10.0);
        // Sloped wedge: z <= 5.0 + 0.5 * x
        let is_solid = |p: DVec3| p.z <= 5.0 + 0.5 * p.x + 0.1 && p.x >= 0.0 && p.x <= 10.0;
        let is_bed_seed = |p: DVec3| p.z <= 0.25;
        let is_top_seed = |p: DVec3| (p.z - (5.0 + 0.5 * p.x)).abs() <= 0.5;

        let field =
            EikonalOrderField::new_conformal_with_occupancy_and_seed_regions_and_slope_limit(
                min_corner,
                max_corner,
                0.5,
                &is_solid,
                &is_bed_seed,
                &is_top_seed,
                None,
                None,
            );

        // Bed level should be flat ~0.0
        let bed1 = field.order(DVec3::new(2.0, 5.0, 0.1));
        let bed2 = field.order(DVec3::new(8.0, 5.0, 0.1));
        assert!(
            (bed1 - bed2).abs() < 0.2,
            "bed order should be approximately flat"
        );

        // Points on the sloped top surface should match their local surface height
        let top1 = field.order(DVec3::new(2.0, 5.0, 6.0));
        let top2 = field.order(DVec3::new(8.0, 5.0, 9.0));
        assert!(
            (top1 - 6.0).abs() < 0.5,
            "top point 1 order should be near 6.0: {top1}"
        );
        assert!(
            (top2 - 9.0).abs() < 0.5,
            "top point 2 order should be near 9.0: {top2}"
        );

        // Subsurface points 0.5mm below the top surface should be ~0.5mm lower in order
        let sub1 = field.order(DVec3::new(2.0, 5.0, 5.5));
        let sub2 = field.order(DVec3::new(8.0, 5.0, 8.5));
        assert!(
            (top1 - sub1 - 0.5).abs() < 0.3,
            "subsurface 1 should be parallel to top surface"
        );
        assert!(
            (top2 - sub2 - 0.5).abs() < 0.3,
            "subsurface 2 should be parallel to top surface"
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

    #[test]
    fn slope_profile_relaxation_couples_features_across_open_air_gap() {
        // Two vertical pillars separated by 10mm of open air.
        // Pillar A at x in [0, 2], Pillar B at x in [12, 14].
        // Pillar A is seeded at z=0, Pillar B is seeded at z=0 with a delayed offset.
        let min_corner = DVec3::new(0.0, 0.0, 0.0);
        let max_corner = DVec3::new(14.0, 2.0, 10.0);
        let is_solid = |p: DVec3| (p.x <= 2.0 || p.x >= 12.0) && p.y <= 2.0 && p.z <= 10.0;
        let is_seed = |p: DVec3| p.x <= 2.0 && p.z <= 0.5;

        // With a 10-degree slope profile, the maximum order difference across the
        // 10mm horizontal gap (dx = 10mm) is capped at 10 * tan(10 deg) ~ 1.76mm.
        let profile = SlopeProfile::new(vec![(0.0, 10.0)]);

        let field = EikonalOrderField::new_with_occupancy_and_seed_region_and_slope_limit(
            min_corner,
            max_corner,
            1.0,
            &is_solid,
            &is_seed,
            Some(&profile),
            None,
        );

        let order_a = field.order(DVec3::new(1.0, 1.0, 5.0));
        let order_b = field.order(DVec3::new(13.0, 1.0, 5.0));

        let max_allowed_delta = 12.0 * 10.0f64.to_radians().tan() + 1.0;
        assert!(
            (order_a - order_b).abs() <= max_allowed_delta,
            "order difference across open air ({}) exceeded max slope limit bound ({})",
            (order_a - order_b).abs(),
            max_allowed_delta
        );
    }
}
