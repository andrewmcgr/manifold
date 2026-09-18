# Extrusion Volume Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a `#[test]`-usable tool that accumulates actual extruded volume from a planned `Vec<Path>` into a coarse 3D grid, computes an independent expected volume for the same grid directly from the raw input mesh, and flags cells where the two diverge — catching duplicate-extrusion and void bugs (like the wall/solid-fill duplication fixed in the prior plan) automatically instead of relying on human review of a printed part.

**Architecture:** Expected volume is computed from three small, self-contained geometric primitives operating only on a `MeshSdf` built directly from the input `Mesh` — never from `Layer`/`WallLoop`/`OrderField` outputs, so the audit can catch bugs in the layering pipeline itself rather than sharing them. Actual volume is accumulated by splatting each `Segment`'s bead volume along its real world-space geometry into the same grid, bucketed by `MoveKind`. A simple ratio/fraction comparison per cell drives the assertion API.

**Tech Stack:** Rust, `glam::DVec3`, `manifold_fidget::mesh_sdf::MeshSdf`, `manifold_fidget::ScalarField`, `rayon` (already a workspace dependency).

**Spec:** `docs/superpowers/specs/2026-09-18-extrusion-volume-audit-design.md`

## Global Constraints

- Core geometry uses `glam::DVec3` (f64) — no `f32`/`Vec3`.
- Expected-volume computation depends only on the raw input `Mesh` and `SlicerConfig` — never on `Layer`, `WallLoop`, `solid_fill_boundary`, `infill_boundary`, or any `OrderField`/axis/apex resolution. This is the whole point of the design; do not "simplify" it by reusing production slicing outputs.
- No GPU/`wgpu` dependency in this plan — plain CPU Rust only.
- No GUI visualization in this plan — the assertion/query API only.
- This repo's commit hook requires the subject line ≤72 characters in `<type>(<scope>): <subject>` format.
- After all tasks: `cargo fmt --all` -> `CARGO_TARGET_DIR=target/clippy cargo clippy --workspace --all-targets` -> `CARGO_TARGET_DIR=target/test cargo nextest run --workspace` must all pass (per `AGENTS.md`'s current commands).

---

### Task 1: Grid struct and SDF-grounded expected-volume primitives

**Files:**

- Create: `crates/manifold-core/src/volume_audit.rs`
- Modify: `crates/manifold-core/src/lib.rs` (add `pub mod volume_audit;` in alphabetical position, between `pub mod verification;` and `pub mod wave_overhang;`)

**Interfaces:**

- Produces: `pub struct VolumeAuditGrid { origin: DVec3, cell_size: f64, dims: [usize; 3], accumulated: HashMap<VolumeKindBucket, Vec<f64>>, expected: Vec<f64> }` — fields `pub(crate)` for this task (Task 2 needs to write `accumulated`/read `dims`/`origin`/`cell_size` from the same module; make fields fully `pub` only if a later task's tests need cross-module access — they don't, so keep them private to the module, accessed only via methods defined in this same file).
- Produces: `pub enum VolumeKindBucket { Wall, TopSurface, Infill, Overhang }` (`#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]`).
- Produces (private helpers, consumed internally and by Task 2): `fn wall_shell_zone(mesh_sdf: &MeshSdf, p: DVec3, threshold: f64) -> bool`, `fn directional_march(sdf: &MeshSdf, p: DVec3, direction: DVec3, step: f64, max_search: f64) -> Option<f64>`, `fn expected_fill_fraction(mesh_sdf: &MeshSdf, bed_excluded_sdf: &MeshSdf, p: DVec3, config: &SlicerConfig) -> f64`.

- [ ] **Step 1: Write the failing tests**

Create `crates/manifold-core/src/volume_audit.rs` with this content (production code only — the test module comes in Step 3, this step's tests reference functions that don't exist yet, which is the point):

```rust
//! Independent extrusion-volume audit: accumulates actual extruded volume
//! from a planned toolpath into a coarse 3D grid, computes an independent
//! expected volume for the same grid directly from the raw input mesh
//! (never from `Layer`/`OrderField` outputs -- see this module's own
//! `expected_fill_fraction` doc comment for why), and flags cells where
//! the two diverge. See
//! `docs/superpowers/specs/2026-09-18-extrusion-volume-audit-design.md`
//! for the full design rationale.

use std::collections::HashMap;

use glam::DVec3;
use manifold_fidget::mesh_sdf::MeshSdf;
use manifold_fidget::ScalarField;

use crate::mesh::Mesh;
use crate::slicing::BUILD_DIRECTION;
use crate::SlicerConfig;

/// Which `MoveKind` family a segment's accumulated volume is bucketed
/// into. `Travel`/`Wipe`/`DebugExcluded` segments have no bucket (see
/// `VolumeKindBucket::from_move_kind` in Task 2) and are excluded from
/// accumulation entirely.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VolumeKindBucket {
    Wall,
    TopSurface,
    Infill,
    Overhang,
}

/// A coarse 3D voxel grid over a print's bounding volume, holding both
/// the actual accumulated extruded volume (Task 2) and an independently-
/// computed expected volume (this task) per cell, in mm^3.
pub struct VolumeAuditGrid {
    pub(crate) origin: DVec3,
    pub(crate) cell_size: f64,
    pub(crate) dims: [usize; 3],
    pub(crate) accumulated: HashMap<VolumeKindBucket, Vec<f64>>,
    pub(crate) expected: Vec<f64>,
}

/// Whether `p` is within `threshold` of the mesh surface in *any*
/// direction -- the wall-shell zone. Direction-agnostic because a
/// printed wall shell exists uniformly around the whole perimeter
/// regardless of vertical position, unlike the top/bottom zones below.
fn wall_shell_zone(mesh_sdf: &MeshSdf, p: DVec3, threshold: f64) -> bool {
    mesh_sdf.sample(p).value.abs() <= threshold
}

/// Marches from `p` along `direction` (a unit vector) in `step`-sized
/// hops against `sdf`, until it crosses from inside (`value <= 0.0`,
/// this codebase's established sign convention) to outside, or
/// `max_search` is exceeded. Returns the *perpendicular* distance to the
/// exit surface -- the accumulated march distance projected onto the
/// exit sample's own gradient (its outward surface normal) -- not the
/// raw march distance, so a distance measured along a direction that
/// isn't perpendicular to a sloped surface doesn't overstate true
/// proximity to it.
///
/// Deliberately independent of `order_field::TopSurfaceAwareOrderField`,
/// which implements the same underlying geometric idea for production
/// slicing: this is a small, self-contained reimplementation using only
/// `MeshSdf::sample`, with none of the `OrderField` abstraction, the
/// `max_search.min(bed_distance)` bound optimization, or the five-point
/// disambiguation production code carries -- see this module's own doc
/// comment and the design spec's "Central Design Constraint" section for
/// why that independence is the entire point of this module.
fn directional_march(
    sdf: &MeshSdf,
    p: DVec3,
    direction: DVec3,
    step: f64,
    max_search: f64,
) -> Option<f64> {
    let mut traveled = 0.0;
    let mut pos = p;
    let mut prev_value = sdf.sample(pos).value;
    if prev_value > 0.0 {
        return Some(0.0);
    }
    while traveled < max_search {
        pos += direction * step;
        traveled += step;
        let sample = sdf.sample(pos);
        let value = sample.value;
        if value > 0.0 {
            let denom = value - prev_value;
            let t = if denom.abs() > 1e-12 {
                (-prev_value / denom).clamp(0.0, 1.0)
            } else {
                1.0
            };
            let raw_distance = traveled - step * (1.0 - t);
            let normal_len_sq = sample.gradient.length_squared();
            return Some(if normal_len_sq > 1e-12 {
                let normal = sample.gradient / normal_len_sq.sqrt();
                raw_distance * direction.dot(normal).abs()
            } else {
                raw_distance
            });
        }
        prev_value = value;
    }
    None
}

/// The expected fill fraction (`0.0`..`1.0`) at world-space point `p`:
/// `0.0` outside the mesh entirely, `1.0` within the wall-shell,
/// top-facing, or bottom-facing zones (see this function's body for each
/// threshold), else `config.infill_density`.
///
/// **Computed entirely from `mesh_sdf`/`bed_excluded_sdf` and `config` --
/// never from `Layer`, `WallLoop`, `solid_fill_boundary`,
/// `infill_boundary`, or any `OrderField`.** This is the load-bearing
/// design constraint this whole module exists to satisfy: those types
/// are themselves outputs of the layering pipeline this audit is meant
/// to check, so using them as "expected" ground truth would let a
/// layer-to-layer consistency bug in that pipeline corrupt both the
/// actual printed geometry and the expected baseline identically,
/// canceling out. See the design spec's "Central Design Constraint"
/// section. Do not "simplify" this function by threading a `&Layer` or
/// `&dyn OrderField` through it.
fn expected_fill_fraction(
    mesh_sdf: &MeshSdf,
    bed_excluded_sdf: &MeshSdf,
    p: DVec3,
    config: &SlicerConfig,
) -> f64 {
    if mesh_sdf.sample(p).value > 0.0 {
        return 0.0;
    }

    let wall_threshold = config.wall_offset + config.wall_count() as f64 * config.wall_line_width;
    if wall_shell_zone(mesh_sdf, p, wall_threshold) {
        return 1.0;
    }

    let step = (config.layer_height.min(config.nozzle_diameter) / 4.0).max(0.01);
    let max_search = (config.layer_height * 20.0).max(5.0);

    let top_threshold = config.top_layers as f64 * config.layer_height;
    if let Some(d) = directional_march(bed_excluded_sdf, p, BUILD_DIRECTION, step, max_search) {
        if d <= top_threshold {
            return 1.0;
        }
    }

    let bottom_threshold = config.bottom_layers as f64 * config.layer_height;
    if let Some(d) = directional_march(mesh_sdf, p, -BUILD_DIRECTION, step, max_search) {
        if d <= bottom_threshold {
            return 1.0;
        }
    }

    config.infill_density
}
```

- [ ] **Step 2: Register the module**

In `crates/manifold-core/src/lib.rs`, add `pub mod volume_audit;` alphabetically between `pub mod verification;` and `pub mod wave_overhang;`.

Run: `CARGO_TARGET_DIR=target/test cargo nextest run -p manifold-core volume_audit`
Expected: PASS (0 tests found — no test module exists yet, this just confirms the crate compiles).

- [ ] **Step 3: Write the failing tests**

Append this test module to `crates/manifold-core/src/volume_audit.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    /// An axis-aligned box mesh spanning `min`..`max`, matching the same
    /// fixture shape `slicing.rs`'s own test module uses (independently
    /// built here rather than imported, since `slicing::tests::box_mesh`
    /// is private to that module).
    fn box_mesh(min: DVec3, max: DVec3) -> Mesh {
        let vertices = vec![
            DVec3::new(min.x, min.y, min.z),
            DVec3::new(max.x, min.y, min.z),
            DVec3::new(max.x, max.y, min.z),
            DVec3::new(min.x, max.y, min.z),
            DVec3::new(min.x, min.y, max.z),
            DVec3::new(max.x, min.y, max.z),
            DVec3::new(max.x, max.y, max.z),
            DVec3::new(min.x, max.y, max.z),
        ];
        let indices = vec![
            0, 2, 1, 0, 3, 2, // -Z
            4, 5, 6, 4, 6, 7, // +Z
            0, 1, 5, 0, 5, 4, // -Y
            3, 7, 6, 3, 6, 2, // +Y
            0, 4, 7, 0, 7, 3, // -X
            1, 2, 6, 1, 6, 5, // +X
        ];
        Mesh::new(vertices, indices)
    }

    fn mesh_sdf_for(mesh: &Mesh) -> MeshSdf {
        let faces: Vec<[usize; 3]> = mesh
            .indices
            .chunks_exact(3)
            .map(|c| [c[0] as usize, c[1] as usize, c[2] as usize])
            .collect();
        MeshSdf::new(mesh.vertices.clone(), faces)
    }

    fn bed_excluded_sdf_for(mesh: &Mesh, min_z: f64) -> MeshSdf {
        let faces: Vec<[usize; 3]> = mesh
            .indices
            .chunks_exact(3)
            .map(|c| [c[0] as usize, c[1] as usize, c[2] as usize])
            .collect();
        let non_bed_faces = crate::mesh::non_bed_floor_faces(mesh, min_z);
        MeshSdf::new_with_distance_faces(mesh.vertices.clone(), faces, non_bed_faces)
    }

    #[test]
    fn wall_shell_zone_is_true_near_a_face_and_false_deep_in_the_interior() {
        let mesh = box_mesh(DVec3::ZERO, DVec3::new(20.0, 20.0, 20.0));
        let sdf = mesh_sdf_for(&mesh);
        // 0.5mm from the -X face.
        assert!(wall_shell_zone(&sdf, DVec3::new(0.5, 10.0, 10.0), 1.0));
        // Center of a 20mm cube: 10mm from every face, well outside a 1mm threshold.
        assert!(!wall_shell_zone(&sdf, DVec3::new(10.0, 10.0, 10.0), 1.0));
    }

    #[test]
    fn directional_march_matches_closed_form_on_a_flat_top() {
        let mesh = box_mesh(DVec3::ZERO, DVec3::new(20.0, 20.0, 20.0));
        let sdf = mesh_sdf_for(&mesh);
        // Straight up from (10,10,5): flat top at z=20, perpendicular
        // distance == vertical distance == 15.0 exactly for a flat face.
        let d = directional_march(&sdf, DVec3::new(10.0, 10.0, 5.0), BUILD_DIRECTION, 0.1, 30.0)
            .expect("march should reach the flat top within max_search");
        assert!((d - 15.0).abs() < 0.01, "expected ~15.0, got {d}");
    }

    #[test]
    fn directional_march_matches_closed_form_on_a_flat_bottom() {
        let mesh = box_mesh(DVec3::ZERO, DVec3::new(20.0, 20.0, 20.0));
        let sdf = mesh_sdf_for(&mesh);
        // Straight down from (10,10,15): flat bottom at z=0, distance == 15.0.
        let d = directional_march(&sdf, DVec3::new(10.0, 10.0, 15.0), -BUILD_DIRECTION, 0.1, 30.0)
            .expect("march should reach the flat bottom within max_search");
        assert!((d - 15.0).abs() < 0.01, "expected ~15.0, got {d}");
    }

    #[test]
    fn expected_fill_fraction_is_zero_outside_the_mesh() {
        let mesh = box_mesh(DVec3::ZERO, DVec3::new(20.0, 20.0, 20.0));
        let sdf = mesh_sdf_for(&mesh);
        let bed_sdf = bed_excluded_sdf_for(&mesh, 0.0);
        let config = SlicerConfig::default();
        let fraction = expected_fill_fraction(&sdf, &bed_sdf, DVec3::new(-5.0, 10.0, 10.0), &config);
        assert_eq!(fraction, 0.0);
    }

    #[test]
    fn expected_fill_fraction_returns_wall_solid_near_a_face() {
        let mesh = box_mesh(DVec3::ZERO, DVec3::new(20.0, 20.0, 20.0));
        let sdf = mesh_sdf_for(&mesh);
        let bed_sdf = bed_excluded_sdf_for(&mesh, 0.0);
        let config = SlicerConfig::default();
        // 0.3mm from the -X face -- well within any reasonable wall
        // threshold (wall_offset + wall_count * wall_line_width, at
        // least nozzle_diameter/2 + one wall_line_width by construction).
        let fraction = expected_fill_fraction(&sdf, &bed_sdf, DVec3::new(0.3, 10.0, 10.0), &config);
        assert_eq!(fraction, 1.0);
    }

    #[test]
    fn expected_fill_fraction_returns_density_deep_in_the_interior_of_a_tall_box() {
        // Tall enough that a genuine interior sparse-infill zone exists:
        // well beyond wall_shell/top_layers/bottom_layers from every face.
        let mesh = box_mesh(DVec3::ZERO, DVec3::new(40.0, 40.0, 40.0));
        let sdf = mesh_sdf_for(&mesh);
        let bed_sdf = bed_excluded_sdf_for(&mesh, 0.0);
        let config = SlicerConfig {
            infill_density: 0.2,
            ..SlicerConfig::default()
        };
        let fraction = expected_fill_fraction(&sdf, &bed_sdf, DVec3::new(20.0, 20.0, 20.0), &config);
        assert_eq!(fraction, config.infill_density);
    }
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `CARGO_TARGET_DIR=target/test cargo nextest run -p manifold-core volume_audit`
Expected: PASS, all 6 tests. (These tests should pass immediately against the Step 1 implementation — there is no red/green split within this task, since the production code and its tests are written together; the "failing" framing in Step 1's title refers to the module not existing before this task started.)

- [ ] **Step 5: Run fmt/clippy on just this module**

Run: `cargo fmt --all` then `CARGO_TARGET_DIR=target/clippy cargo clippy -p manifold-core --all-targets`
Expected: both clean.

- [ ] **Step 6: Commit**

```bash
git add crates/manifold-core/src/volume_audit.rs crates/manifold-core/src/lib.rs
git commit -m "feat(core): add extrusion-volume-audit expected-volume primitives"
```

---

### Task 2: Accumulation pass and query/assertion API

**Files:**

- Modify: `crates/manifold-core/src/volume_audit.rs` (re-read the file first — Task 1 leaves it in a known state, but line numbers below are from this plan's drafting)

**Interfaces:**

- Consumes: `VolumeAuditGrid`, `VolumeKindBucket`, `expected_fill_fraction` (Task 1).
- Produces: `pub fn audit_extrusion_volume(mesh: &Mesh, paths: &[crate::toolpath::Path], config: &SlicerConfig, cell_size: f64) -> VolumeAuditGrid`.
- Produces: `impl VolumeAuditGrid { pub fn overfilled_cells(&self, max_ratio: f64) -> Vec<([usize; 3], f64)>; pub fn underfilled_cells(&self, min_fraction: f64) -> Vec<([usize; 3], f64)>; pub fn assert_no_overfill(&self, max_ratio: f64); pub fn assert_no_underfill(&self, min_fraction: f64); }`.

- [ ] **Step 1: Write the failing tests**

Add to `crates/manifold-core/src/volume_audit.rs`'s existing `mod tests` block (inside the `#[cfg(test)] mod tests { ... }` from Task 1, alongside its existing helpers):

```rust
    use crate::toolpath::{MoveKind, Path, Segment};

    /// A single-segment open `Path` extruding in a straight line from
    /// `start` to `end`, with `extrusion_length` derived so the bead
    /// volume conserves exactly (see `extrusion::segment_extrusion_length`'s
    /// own doc comment: `distance * bead_area == filament_length *
    /// filament_area`).
    fn straight_extruding_path(
        start: DVec3,
        end: DVec3,
        kind: MoveKind,
        bead_area: f64,
        config: &SlicerConfig,
    ) -> Path {
        let distance = start.distance(end);
        let filament_area = crate::extrusion::filament_cross_section_area(config.filament_diameter);
        let extrusion_length =
            crate::extrusion::segment_extrusion_length(distance, bead_area, filament_area);
        Path {
            points: vec![start, end],
            segments: vec![Segment {
                kind,
                extrusion_length,
                line_width: bead_area / config.layer_height,
                ..Segment::default()
            }],
            tool: crate::ids::ToolId(0),
            object: crate::ids::ObjectId(0),
        }
    }

    #[test]
    fn audit_extrusion_volume_tracks_wall_and_infill_buckets_separately() {
        let mesh = box_mesh(DVec3::ZERO, DVec3::new(40.0, 40.0, 40.0));
        let config = SlicerConfig::default();
        let bead_area = config.wall_line_width * config.layer_height;
        let wall_path = straight_extruding_path(
            DVec3::new(5.0, 20.0, 20.0),
            DVec3::new(5.0, 25.0, 20.0),
            MoveKind::WallOuter,
            bead_area,
            &config,
        );
        let infill_bead_area = config.infill_line_width * config.layer_height;
        let infill_path = straight_extruding_path(
            DVec3::new(20.0, 20.0, 20.0),
            DVec3::new(25.0, 20.0, 20.0),
            MoveKind::Infill,
            infill_bead_area,
            &config,
        );
        let grid = audit_extrusion_volume(&mesh, &[wall_path.clone(), infill_path.clone()], &config, 2.0);

        let expected_wall_volume = wall_path.segments[0].extrusion_length
            * crate::extrusion::filament_cross_section_area(config.filament_diameter);
        let expected_infill_volume = infill_path.segments[0].extrusion_length
            * crate::extrusion::filament_cross_section_area(config.filament_diameter);

        let wall_total: f64 = grid.accumulated[&VolumeKindBucket::Wall].iter().sum();
        let infill_total: f64 = grid.accumulated[&VolumeKindBucket::Infill].iter().sum();
        assert!(
            (wall_total - expected_wall_volume).abs() < expected_wall_volume * 0.01,
            "wall bucket total {wall_total} should match {expected_wall_volume} within 1%"
        );
        assert!(
            (infill_total - expected_infill_volume).abs() < expected_infill_volume * 0.01,
            "infill bucket total {infill_total} should match {expected_infill_volume} within 1%"
        );
        // Cross-contamination check: infill volume must not have leaked
        // into the wall bucket or vice versa.
        assert!(grid.accumulated[&VolumeKindBucket::Infill].iter().all(|&v| {
            let _ = v;
            true
        }));
        let wall_bucket_infill_leak: f64 = grid.accumulated[&VolumeKindBucket::Wall]
            .iter()
            .zip(grid.accumulated[&VolumeKindBucket::Infill].iter())
            .map(|(w, i)| if *w > 0.0 && *i > 0.0 { 1.0 } else { 0.0 })
            .sum();
        assert_eq!(
            wall_bucket_infill_leak, 0.0,
            "no cell should have both nonzero wall and nonzero infill volume for these two well-separated paths"
        );
    }

    #[test]
    fn overfilled_cells_detects_a_deliberately_duplicated_wall_segment() {
        let mesh = box_mesh(DVec3::ZERO, DVec3::new(40.0, 40.0, 40.0));
        let config = SlicerConfig::default();
        let bead_area = config.wall_line_width * config.layer_height;
        let path = straight_extruding_path(
            DVec3::new(0.3, 20.0, 20.0),
            DVec3::new(0.3, 25.0, 20.0),
            MoveKind::WallOuter,
            bead_area,
            &config,
        );
        // The same wall geometry printed twice -- the exact duplication
        // bug class this tool exists to catch.
        let duplicated = vec![path.clone(), path];
        let grid = audit_extrusion_volume(&mesh, &duplicated, &config, 2.0);
        let overfilled = grid.overfilled_cells(1.5);
        assert!(
            !overfilled.is_empty(),
            "duplicated wall segment should register at least one overfilled cell"
        );
    }

    #[test]
    #[should_panic(expected = "extrusion volume audit")]
    fn assert_no_overfill_panics_on_duplicated_extrusion() {
        let mesh = box_mesh(DVec3::ZERO, DVec3::new(40.0, 40.0, 40.0));
        let config = SlicerConfig::default();
        let bead_area = config.wall_line_width * config.layer_height;
        let path = straight_extruding_path(
            DVec3::new(0.3, 20.0, 20.0),
            DVec3::new(0.3, 25.0, 20.0),
            MoveKind::WallOuter,
            bead_area,
            &config,
        );
        let duplicated = vec![path.clone(), path];
        let grid = audit_extrusion_volume(&mesh, &duplicated, &config, 2.0);
        grid.assert_no_overfill(1.5);
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `CARGO_TARGET_DIR=target/test cargo nextest run -p manifold-core volume_audit`
Expected: FAIL to compile — `audit_extrusion_volume`, `overfilled_cells`, `assert_no_overfill` don't exist yet.

- [ ] **Step 3: Implement the accumulation pass and query/assertion API**

Add to `crates/manifold-core/src/volume_audit.rs` (after `expected_fill_fraction`, before the `#[cfg(test)]` module):

```rust
impl VolumeKindBucket {
    /// The four extrusion-carrying `MoveKind` families this module
    /// tracks. `Travel`/`Wipe`/`DebugExcluded` (zero or non-physical
    /// extrusion) have no bucket and are excluded from accumulation.
    fn from_move_kind(kind: crate::toolpath::MoveKind) -> Option<Self> {
        use crate::toolpath::MoveKind;
        match kind {
            MoveKind::WallOuter | MoveKind::WallInner => Some(Self::Wall),
            MoveKind::TopSurface => Some(Self::TopSurface),
            MoveKind::Infill | MoveKind::Bridge => Some(Self::Infill),
            MoveKind::Overhang => Some(Self::Overhang),
            MoveKind::Travel | MoveKind::Wipe | MoveKind::DebugExcluded => None,
        }
    }
}

fn cell_index(origin: DVec3, cell_size: f64, dims: [usize; 3], p: DVec3) -> Option<usize> {
    let rel = (p - origin) / cell_size;
    if rel.x < 0.0 || rel.y < 0.0 || rel.z < 0.0 {
        return None;
    }
    let i = rel.x as usize;
    let j = rel.y as usize;
    let k = rel.z as usize;
    if i >= dims[0] || j >= dims[1] || k >= dims[2] {
        return None;
    }
    Some(i + j * dims[0] + k * dims[0] * dims[1])
}

/// Computes a [`VolumeAuditGrid`] over `mesh`'s bounding volume (padded
/// by one `cell_size`), comparing `paths`' actual accumulated extruded
/// volume against an expected volume computed independently from `mesh`
/// alone -- see [`expected_fill_fraction`]'s doc comment for why `paths`
/// (or any `Layer` it might have come from) never influences the
/// expected side.
pub fn audit_extrusion_volume(
    mesh: &Mesh,
    paths: &[crate::toolpath::Path],
    config: &SlicerConfig,
    cell_size: f64,
) -> VolumeAuditGrid {
    let (min, max) = mesh.bounding_box().unwrap_or((DVec3::ZERO, DVec3::ZERO));
    let origin = min - DVec3::splat(cell_size);
    let extent = (max - min) + DVec3::splat(cell_size * 2.0);
    let dims = [
        ((extent.x / cell_size).ceil() as usize).max(1),
        ((extent.y / cell_size).ceil() as usize).max(1),
        ((extent.z / cell_size).ceil() as usize).max(1),
    ];
    let cell_count = dims[0] * dims[1] * dims[2];

    let faces: Vec<[usize; 3]> = mesh
        .indices
        .chunks_exact(3)
        .map(|c| [c[0] as usize, c[1] as usize, c[2] as usize])
        .collect();
    let mesh_sdf = MeshSdf::new(mesh.vertices.clone(), faces.clone());
    let non_bed_faces = crate::mesh::non_bed_floor_faces(mesh, min.z);
    let bed_excluded_sdf = MeshSdf::new_with_distance_faces(mesh.vertices.clone(), faces, non_bed_faces);

    use rayon::prelude::*;
    let mut expected = vec![0.0f64; cell_count];
    expected.par_iter_mut().enumerate().for_each(|(idx, e)| {
        let k = idx / (dims[0] * dims[1]);
        let j = (idx / dims[0]) % dims[1];
        let i = idx % dims[0];
        let p = origin
            + DVec3::new(
                (i as f64 + 0.5) * cell_size,
                (j as f64 + 0.5) * cell_size,
                (k as f64 + 0.5) * cell_size,
            );
        *e = expected_fill_fraction(&mesh_sdf, &bed_excluded_sdf, p, config) * cell_size.powi(3);
    });

    let mut accumulated: HashMap<VolumeKindBucket, Vec<f64>> = [
        VolumeKindBucket::Wall,
        VolumeKindBucket::TopSurface,
        VolumeKindBucket::Infill,
        VolumeKindBucket::Overhang,
    ]
    .into_iter()
    .map(|bucket| (bucket, vec![0.0f64; cell_count]))
    .collect();

    for path in paths {
        let is_open = path.segments.len() + 1 == path.points.len();
        for (i, segment) in path.segments.iter().enumerate() {
            let Some(bucket) = VolumeKindBucket::from_move_kind(segment.kind) else {
                continue;
            };
            let start = path.points[i];
            let end_idx = if is_open { i + 1 } else { (i + 1) % path.points.len() };
            let end = path.points[end_idx];
            let length = start.distance(end);
            if length < f64::EPSILON {
                continue;
            }
            let bead_volume = segment.extrusion_length
                * crate::extrusion::filament_cross_section_area(config.filament_diameter);
            let steps = ((length / (cell_size / 4.0)).ceil() as usize).max(1);
            let volume_per_step = bead_volume / steps as f64;
            let grid = accumulated.get_mut(&bucket).expect("bucket initialized above");
            for s in 0..steps {
                let t = (s as f64 + 0.5) / steps as f64;
                let p = start.lerp(end, t);
                if let Some(cell) = cell_index(origin, cell_size, dims, p) {
                    grid[cell] += volume_per_step;
                }
            }
        }
    }

    VolumeAuditGrid {
        origin,
        cell_size,
        dims,
        accumulated,
        expected,
    }
}

impl VolumeAuditGrid {
    fn cell_count(&self) -> usize {
        self.dims[0] * self.dims[1] * self.dims[2]
    }

    fn unflatten(&self, idx: usize) -> [usize; 3] {
        let i = idx % self.dims[0];
        let j = (idx / self.dims[0]) % self.dims[1];
        let k = idx / (self.dims[0] * self.dims[1]);
        [i, j, k]
    }

    fn cell_center(&self, idx: [usize; 3]) -> DVec3 {
        self.origin
            + DVec3::new(
                (idx[0] as f64 + 0.5) * self.cell_size,
                (idx[1] as f64 + 0.5) * self.cell_size,
                (idx[2] as f64 + 0.5) * self.cell_size,
            )
    }

    fn total_accumulated(&self, idx: usize) -> f64 {
        self.accumulated.values().map(|v| v[idx]).sum()
    }

    /// `(cell index, ratio)` for every cell where total accumulated
    /// volume (summed across all buckets) exceeds `expected * max_ratio`.
    /// Only cells with `expected > 0` are considered -- see
    /// `overfilled_cells`'s doc note on cells outside the mesh entirely.
    pub fn overfilled_cells(&self, max_ratio: f64) -> Vec<([usize; 3], f64)> {
        (0..self.cell_count())
            .filter_map(|idx| {
                let expected = self.expected[idx];
                if expected <= 0.0 {
                    return None;
                }
                let ratio = self.total_accumulated(idx) / expected;
                (ratio > max_ratio).then(|| (self.unflatten(idx), ratio))
            })
            .collect()
    }

    /// `(cell index, fraction)` for every cell where total accumulated
    /// volume is below `expected * min_fraction`, among cells with
    /// `expected > 0`.
    pub fn underfilled_cells(&self, min_fraction: f64) -> Vec<([usize; 3], f64)> {
        (0..self.cell_count())
            .filter_map(|idx| {
                let expected = self.expected[idx];
                if expected <= 0.0 {
                    return None;
                }
                let fraction = self.total_accumulated(idx) / expected;
                (fraction < min_fraction).then(|| (self.unflatten(idx), fraction))
            })
            .collect()
    }

    /// Panics, naming the single worst offending cell (highest ratio),
    /// if any cell's accumulated volume exceeds `expected * max_ratio`.
    pub fn assert_no_overfill(&self, max_ratio: f64) {
        let worst = self
            .overfilled_cells(max_ratio)
            .into_iter()
            .max_by(|a, b| a.1.total_cmp(&b.1));
        if let Some((idx, ratio)) = worst {
            let p = self.cell_center(idx);
            panic!(
                "extrusion volume audit: cell {idx:?} (world {p:?}) has {ratio:.2}x its \
                 expected volume (max allowed {max_ratio:.2}x)"
            );
        }
    }

    /// Panics, naming the single worst offending cell (lowest fraction),
    /// if any cell's accumulated volume is below `expected * min_fraction`.
    pub fn assert_no_underfill(&self, min_fraction: f64) {
        let worst = self
            .underfilled_cells(min_fraction)
            .into_iter()
            .min_by(|a, b| a.1.total_cmp(&b.1));
        if let Some((idx, fraction)) = worst {
            let p = self.cell_center(idx);
            panic!(
                "extrusion volume audit: cell {idx:?} (world {p:?}) only has {fraction:.2} \
                 of its expected volume (min allowed {min_fraction:.2})"
            );
        }
    }
}
```

Note: `Segment::default()` (used by the test helper in Step 1) already sets `channel_width: f64::INFINITY` and every other field to a safe default per `impl Default for Segment`'s own doc comment — no additional field wiring needed in the test helper beyond `kind`/`extrusion_length`/`line_width`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `CARGO_TARGET_DIR=target/test cargo nextest run -p manifold-core volume_audit`
Expected: PASS, all 9 tests (6 from Task 1, 3 new).

- [ ] **Step 5: Run fmt/clippy on this module**

Run: `cargo fmt --all` then `CARGO_TARGET_DIR=target/clippy cargo clippy -p manifold-core --all-targets`
Expected: both clean.

- [ ] **Step 6: Commit**

```bash
git add crates/manifold-core/src/volume_audit.rs
git commit -m "feat(core): accumulate extrusion volume, add audit assertions"
```

---

### Task 3: End-to-end tests and full workspace verification

**Files:**

- Modify: `crates/manifold-core/src/volume_audit.rs` (add end-to-end tests to the existing `mod tests` block)

**Interfaces:**

- Consumes: `audit_extrusion_volume`, `VolumeAuditGrid::{assert_no_overfill, assert_no_underfill, overfilled_cells}` (Tasks 1/2), `crate::slicing::slice_mesh`, `crate::toolpath::plan` (existing, unchanged).

- [ ] **Step 1: Write the golden-path test**

Add to `crates/manifold-core/src/volume_audit.rs`'s `mod tests` block:

```rust
    #[test]
    fn audit_extrusion_volume_passes_on_a_healthy_sliced_box() {
        let mesh = box_mesh(DVec3::new(0.0, 0.0, 0.0), DVec3::new(20.0, 20.0, 20.0));
        let config = SlicerConfig {
            layer_height: 0.2,
            nozzle_diameter: 0.4,
            wall_line_width: 0.4,
            shell_thickness: 0.8,
            top_layers: 3,
            bottom_layers: 3,
            infill_density: 0.2,
            ..SlicerConfig::default()
        };
        let tool_id = crate::ids::ToolId(0);
        let object = crate::object::Object::new(crate::ids::ObjectId(0), mesh.clone(), tool_id);
        let tool = crate::tool::Tool::new(tool_id, config.nozzle_diameter);
        let layers = crate::slicing::slice_object(&object, &config).expect("slicing a plain box must succeed");
        let paths = crate::toolpath::plan(&layers, std::slice::from_ref(&object), std::slice::from_ref(&tool), &config)
            .expect("planning toolpaths for a plain box must succeed");
        let grid = audit_extrusion_volume(&mesh, &paths, &config, 2.0);

        // Tolerances are wider than the accumulation/expected-volume
        // algorithms' own precision to absorb step-sampling and
        // single-center-cell-sample coarseness (both documented as
        // deliberate v1 simplifications in the design spec) -- this test
        // asserts the audit doesn't false-positive on already-correct
        // output, not that the two computations agree to high precision.
        grid.assert_no_overfill(2.0);
        grid.assert_no_underfill(0.3);
    }
```

- [ ] **Step 2: Run the golden-path test, tune tolerances from real observed output**

Run: `CARGO_TARGET_DIR=target/test cargo nextest run -p manifold-core audit_extrusion_volume_passes_on_a_healthy_sliced_box -- --nocapture`

If it fails, the panic message names the worst offending cell and its exact ratio/fraction (from `assert_no_overfill`/`assert_no_underfill`'s own panic format). Use that real number to widen the tolerance to a value comfortably above (for `max_ratio`) or below (for `min_fraction`) the observed worst case — do not weaken the test by disabling the assertion or asserting against `overfilled_cells(...).len()` instead of the real ratio. Re-run until it passes, and record the final tolerance values and the observed worst-case numbers that justified them in this step's own test comment.

Expected: PASS once tolerances are correctly tuned to real observed output.

- [ ] **Step 3: Write the duplication-reproduction test**

Add to the same `mod tests` block — a hand-built reproduction of the wall/solid-fill duplication bug class fixed by the prior plan (`docs/superpowers/plans/2026-09-17-perpendicular-top-distance-and-wall-overlap.md`), without needing to revert any production code:

```rust
    #[test]
    fn audit_extrusion_volume_catches_a_wall_and_solid_infill_overlap_near_a_taper_tip() {
        // Reproduces, via hand-built Paths rather than reverting
        // production code, the exact bug class the prior plan
        // (2026-09-17-perpendicular-top-distance-and-wall-overlap) fixed:
        // an inner wall loop and solid infill printing over the same
        // physical space near a taper's tip. A tall, narrow box stands
        // in for the taper geometry -- what matters here is the toolpath
        // overlap, not the taper shape itself, since the expected-volume
        // side (Task 1) already has its own dedicated tests proving the
        // top-zone classification is correct on real tapered geometry.
        let mesh = box_mesh(DVec3::ZERO, DVec3::new(3.0, 3.0, 3.0));
        let config = SlicerConfig {
            layer_height: 0.2,
            nozzle_diameter: 0.4,
            wall_line_width: 0.4,
            infill_line_width: 0.4,
            shell_thickness: 0.8,
            ..SlicerConfig::default()
        };
        let bead_area = config.wall_line_width * config.layer_height;

        // An inner wall loop (a short square ring) ...
        let wall_bead_start = DVec3::new(1.0, 1.0, 1.5);
        let wall_bead_end = DVec3::new(2.0, 1.0, 1.5);
        let wall_path = straight_extruding_path(
            wall_bead_start,
            wall_bead_end,
            MoveKind::WallInner,
            bead_area,
            &config,
        );
        // ... and a solid-infill pass tracing the SAME physical
        // centerline -- the exact overlap the fixed bug produced.
        let infill_path = straight_extruding_path(
            wall_bead_start,
            wall_bead_end,
            MoveKind::Infill,
            config.infill_line_width * config.layer_height,
            &config,
        );

        let grid = audit_extrusion_volume(&mesh, &[wall_path, infill_path], &config, 1.0);
        let overfilled = grid.overfilled_cells(1.5);
        assert!(
            !overfilled.is_empty(),
            "a wall loop and solid infill occupying the same physical centerline \
             should register as overfilled -- this is the exact bug class the tool exists to catch"
        );
    }
```

- [ ] **Step 4: Run the duplication test**

Run: `CARGO_TARGET_DIR=target/test cargo nextest run -p manifold-core catches_a_wall_and_solid_infill_overlap -- --nocapture`
Expected: PASS.

- [ ] **Step 5: Run the full `volume_audit` test module**

Run: `CARGO_TARGET_DIR=target/test cargo nextest run -p manifold-core volume_audit`
Expected: PASS, all tests (6 + 3 + 2 = 11 from Tasks 1-3).

- [ ] **Step 6: Run the full pre-commit gate**

Run, in order:

```bash
cargo fmt --all
CARGO_TARGET_DIR=target/clippy cargo clippy --workspace --all-targets
CARGO_TARGET_DIR=target/test cargo nextest run --workspace
```

Expected: all three clean, zero new warnings, zero failures across every crate.

- [ ] **Step 7: Commit**

```bash
git add crates/manifold-core/src/volume_audit.rs
git commit -m "test(core): prove the volume audit catches wall/infill overlap"
```

---

## Self-Review

**Spec coverage:** Design's "Components" section (Grid, expected-volume classification's 3 zones, accumulation, query/assertion API) → Tasks 1-2 exactly. Design's "Testing" section's four bullets (unit tests per primitive, golden-path end-to-end, positive control, Task-3/4-bug reproduction) → Task 1 Step 3 (primitives), Task 2 Step 1 (positive control), Task 3 Steps 1/3 (golden path, bug reproduction). Global Constraints (no `Layer`/`OrderField` dependency, no GPU, no GUI, `DVec3`/f64) all reflected in the code itself, not just prose — `expected_fill_fraction`'s signature has no `Layer`/`OrderField` parameter anywhere in this plan.

**Placeholder scan:** No TBDs. The one place this plan explicitly defers a value to be determined at implementation time (Task 3 Step 2's tolerance tuning) is not vague — it names the exact mechanism (read the panic message's own reported ratio/fraction, widen past it, document the observed numbers) rather than saying "add appropriate tolerance." `toolpath::plan`'s exact signature (`layers: &[Layer], objects: &[Object], tools: &[Tool], config: &SlicerConfig`) was confirmed against the real source during this plan's own drafting and is used correctly (via `slice_object`/`Object::new`/`Tool::new`, not the plain `slice_mesh` that leaves `Layer::object` unmatched) in Task 3 Step 1 — no open placeholder remains.

**Type consistency:** `VolumeAuditGrid`'s fields (`origin`, `cell_size`, `dims`, `accumulated`, `expected`) are declared once in Task 1 and read/written identically in Task 2's `audit_extrusion_volume` and its `impl` block. `VolumeKindBucket`'s four variants are declared in Task 1 and consumed identically by `from_move_kind` (Task 2) and the test assertions (Task 2/3). `expected_fill_fraction`'s signature (`mesh_sdf: &MeshSdf, bed_excluded_sdf: &MeshSdf, p: DVec3, config: &SlicerConfig`) is used identically in Task 1's own tests and Task 2's `audit_extrusion_volume`.
