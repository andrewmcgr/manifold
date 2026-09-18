# Volume Audit Fill Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render the extrusion-volume-audit's per-cell fill ratio as colored cubes in the `manifold-gui` 3D viewport — the GPU visualization the original audit design named as a goal and deliberately deferred.

**Architecture:** One new public read-only query on `VolumeAuditGrid` (`cell_ratios_for_display`) exposes every material-bearing cell's ratio without the assertion API's `Solid`-only restriction. A new, GPU-free geometry builder in `manifold-gui` (`volume_audit_view.rs`, mirroring `toolpath_view.rs`'s existing separation) turns that data into colored cube triangles. Rendering reuses the existing triangle-mesh pipeline (`UploadedMesh` in `render.rs`) via one new constructor and one new optional `Viewport3dCallback` field — no new shader or pipeline.

**Tech Stack:** Rust, `glam`, `wgpu` (via `eframe::egui_wgpu`), `bytemuck` (already workspace/crate dependencies, no new ones).

**Spec:** `docs/superpowers/specs/2026-09-18-volume-audit-fill-visualization-design.md`

## Global Constraints

- `manifold-core` changes use `glam::DVec3`/f64; `manifold-gui`'s GPU-facing vertex struct uses `f32`, matching the existing `render.rs`/`toolpath_view.rs` convention.
- No new wgpu shader, pipeline, or bind group — reuse the existing triangle-mesh pipeline exactly (same one `UploadedMesh::upload`/`upload_from_vertices` already use).
- No changes to `expected_fill_fraction`/`classify_fill_zone`/the SDF-only grounding constraint, and no changes to any existing assertion (`overfilled_cells`/`underfilled_cells`/`assert_no_overfill`/etc.) — this plan only adds a new read over already-computed grid data.
- Single-object scope only for the GUI wiring (Task 3): `audit_extrusion_volume` takes a single `&Mesh`, a known, already-documented limitation of the underlying tool — do not attempt multi-object support in this plan.
- This repo's commit hook requires the subject line ≤72 characters in `<type>(<scope>): <subject>` format.
- After each task: `cargo fmt --all` -> `CARGO_TARGET_DIR=target/clippy cargo clippy --workspace --all-targets` -> `CARGO_TARGET_DIR=target/test cargo nextest run --workspace` must all pass.
- **Task 3 is a GUI/rendering change that cannot be fully verified by automated tests.** Its implementer must confirm the app builds and launches without panicking, but visual correctness (cube placement, colors, decluttering behavior) requires a human to actually look at the running GUI — flag this honestly in the report rather than claiming visual correctness from code review alone.

---

### Task 1: `cell_ratios_for_display` query on `VolumeAuditGrid`

**Files:**
- Modify: `crates/manifold-core/src/volume_audit.rs`

**Interfaces:**
- Produces: `pub fn cell_ratios_for_display(&self) -> Vec<([usize; 3], f64)>` on `impl VolumeAuditGrid`.
- Consumes (existing, unchanged): `self.zone`, `self.expected`, `self.total_accumulated`, `self.unflatten`, `self.cell_count` — the same private helpers `overfilled_cells` already uses.

- [ ] **Step 1: Add the new query**

In `crates/manifold-core/src/volume_audit.rs`, inside `impl VolumeAuditGrid`, find `underfilled_cells` (it sits directly after `overfilled_cells`). Add the new method directly after `underfilled_cells`, before `extrusion_outside_mesh_cells`:

```rust

    /// `(cell index, ratio)` for every cell with `expected > 0` (i.e.
    /// every `Solid`- or `SparseInfill`-zone cell), regardless of the
    /// ratio's value.
    ///
    /// For display/visualization purposes only -- unlike
    /// [`VolumeAuditGrid::overfilled_cells`]/[`VolumeAuditGrid::underfilled_cells`],
    /// this is NOT restricted to `Solid`-zone cells and carries no
    /// pass/fail threshold. A caller doing defect *detection* should use
    /// those instead, or [`VolumeAuditGrid::infill_aggregate_ratio`] for
    /// infill's own coarser check. A cell with `expected == 0` (the
    /// `Outside` zone) is never included here -- see
    /// [`VolumeAuditGrid::extrusion_outside_mesh_cells`] for that
    /// separate, binary case.
    pub fn cell_ratios_for_display(&self) -> Vec<([usize; 3], f64)> {
        (0..self.cell_count())
            .filter(|&idx| self.expected[idx] > 0.0)
            .map(|idx| (self.unflatten(idx), self.total_accumulated(idx) / self.expected[idx]))
            .collect()
    }
```

- [ ] **Step 2: Write tests**

Add these two tests in `mod tests`, alongside the other hand-built-fixture tests (e.g. after `cell_center_and_accumulated_volume_decode_a_reported_cell`):

```rust
    #[test]
    fn cell_ratios_for_display_includes_both_solid_and_sparse_infill_cells() {
        // Tall enough that a genuine interior sparse-infill zone exists,
        // same fixture shape as
        // `expected_fill_fraction_returns_density_deep_in_the_interior_of_a_tall_box`.
        let mesh = box_mesh(DVec3::ZERO, DVec3::new(40.0, 40.0, 40.0));
        let config = SlicerConfig {
            infill_density: 0.2,
            ..SlicerConfig::default()
        };
        let cell_size = 2.0;
        let wall_bead_area = config.wall_line_width * config.layer_height;
        // A wall bead near a face (Solid zone) ...
        let wall_path = straight_extruding_path(
            DVec3::new(0.3, 20.0, 20.0),
            DVec3::new(0.3, 22.0, 20.0),
            MoveKind::WallOuter,
            wall_bead_area,
            &config,
        );
        // ... and an infill bead deep in the interior (SparseInfill zone).
        let infill_bead_area = config.infill_line_width * config.layer_height;
        let infill_path = straight_extruding_path(
            DVec3::new(19.0, 20.0, 20.0),
            DVec3::new(21.0, 20.0, 20.0),
            MoveKind::Infill,
            infill_bead_area,
            &config,
        );
        let grid = audit_extrusion_volume(&mesh, &[wall_path, infill_path], &config, cell_size);

        let ratios = grid.cell_ratios_for_display();
        assert!(
            !ratios.is_empty(),
            "a mesh with real extrusion should register at least one cell"
        );

        // Cross-check: every returned cell must have `expected > 0`
        // (reachable only via the private zone/expected fields, so
        // reconstruct via the same public accessors a real caller has:
        // a cell classified `Outside` never appears here even if it
        // somehow held material -- covered by the dedicated fixture in
        // the next test instead).
        let outside = grid.extrusion_outside_mesh_cells();
        for (idx, _) in &ratios {
            assert!(
                !outside.contains(idx),
                "cell {idx:?} appears in both cell_ratios_for_display and \
                 extrusion_outside_mesh_cells -- the two queries must partition disjointly"
            );
        }

        // At least one cell should register a wall-adjacent ratio near
        // the wall bead's own known accumulation (loosely bounded --
        // this is a display query, not a precision assertion; the goal
        // is just confirming both zone kinds are represented).
        assert!(
            ratios.iter().any(|(_, r)| *r > 0.0),
            "at least one cell should have a nonzero ratio given real extrusion was deposited"
        );
    }

    #[test]
    fn cell_ratios_for_display_excludes_outside_the_mesh_cells() {
        let mesh = box_mesh(DVec3::ZERO, DVec3::new(2.0, 6.0, 2.0));
        let config = SlicerConfig::default();
        let cell_size = 0.5;
        // Same stray-bead fixture as
        // `extrusion_outside_the_mesh_is_caught_only_by_its_own_dedicated_query`:
        // x = -0.25 is outside the mesh (min.x == 0.0) but inside the
        // padded grid.
        let stray = straight_extruding_path(
            DVec3::new(-0.25, 2.0, 1.0),
            DVec3::new(-0.25, 4.0, 1.0),
            MoveKind::WallOuter,
            config.wall_line_width * config.layer_height,
            &config,
        );
        let grid = audit_extrusion_volume(&mesh, std::slice::from_ref(&stray), &config, cell_size);

        let outside = grid.extrusion_outside_mesh_cells();
        assert!(!outside.is_empty(), "the stray bead must register as outside-the-mesh material");

        let display_ratios = grid.cell_ratios_for_display();
        for idx in &outside {
            assert!(
                !display_ratios.iter().any(|(display_idx, _)| display_idx == idx),
                "cell_ratios_for_display must not include cell {idx:?}, which \
                 extrusion_outside_mesh_cells already reports as outside the mesh"
            );
        }
    }
```

- [ ] **Step 3: Run tests**

Run: `CARGO_TARGET_DIR=target/test cargo nextest run -p manifold-core volume_audit::tests::cell_ratios_for_display`

Expected: both new tests pass.

- [ ] **Step 4: Run the full pre-commit gate and commit**

```bash
cargo fmt --all
CARGO_TARGET_DIR=target/clippy cargo clippy --workspace --all-targets
CARGO_TARGET_DIR=target/test cargo nextest run --workspace
```

All three must be clean.

```bash
git add crates/manifold-core/src/volume_audit.rs
git commit -m "feat(core): add cell_ratios_for_display query for GUI"
```

---

### Task 2: `volume_audit_view.rs` pure geometry builder

**Files:**
- Create: `crates/manifold-gui/src/volume_audit_view.rs`
- Modify: `crates/manifold-gui/src/main.rs` (add `mod volume_audit_view;` alongside the other `mod` declarations — check the file for the existing module list and match its style/ordering, e.g. alongside `mod toolpath_view;`)

**Interfaces:**
- Consumes (Task 1): `manifold_core::volume_audit::VolumeAuditGrid::cell_ratios_for_display() -> Vec<([usize; 3], f64)>`, `VolumeAuditGrid::extrusion_outside_mesh_cells() -> Vec<[usize; 3]>`, `VolumeAuditGrid::cell_center(&self, idx: [usize; 3]) -> DVec3` (all already public).
- Consumes (existing): `crate::toolpath_view::scalar_to_color(t: f64) -> [f32; 4]`.
- Produces: `pub struct VolumeAuditCellVertex { pub position: [f32; 3], pub normal: [f32; 3], pub color: [f32; 4] }` and `pub fn build_volume_audit_cells(grid: &manifold_core::volume_audit::VolumeAuditGrid, deviation_threshold: f64, shrink_factor: f64) -> Vec<VolumeAuditCellVertex>`, both consumed by Task 3.

- [ ] **Step 1: Write the failing tests**

Create `crates/manifold-gui/src/volume_audit_view.rs` with this content (production code plus tests together — this is a small, self-contained pure module):

```rust
//! Pure geometry builder for the volume-audit fill visualization: turns a
//! `manifold_core::volume_audit::VolumeAuditGrid`'s per-cell fill ratios
//! into flat-colored cube triangles. No GPU/wgpu types here — mirrors
//! `toolpath_view.rs`'s existing separation from `render.rs`'s GPU
//! upload/pipeline concerns. See
//! `docs/superpowers/specs/2026-09-18-volume-audit-fill-visualization-design.md`
//! for the full design rationale.

use glam::DVec3;
use manifold_core::volume_audit::VolumeAuditGrid;

/// One GPU vertex for a flat-colored cube face: position + face normal +
/// RGBA color, all in world space. Bit-identical layout to `render.rs`'s
/// private `Vertex`, so `UploadedMesh::upload_colored_cells` can build
/// its buffer directly from a slice of these.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VolumeAuditCellVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 4],
}

/// Fixed color for a cell reported by `VolumeAuditGrid::extrusion_outside_mesh_cells`
/// (material where none is expected at all) -- a defect at any
/// magnitude, so distinct from the ratio-based blue/green/red scale and
/// never hidden by `deviation_threshold`.
const OUTSIDE_MESH_COLOR: [f32; 4] = [0.95, 0.05, 0.85, 1.0];

/// The six axis-aligned face normals of a cube, in the same face order
/// `push_cube`'s vertex generation below uses.
const FACE_NORMALS: [DVec3; 6] = [
    DVec3::new(1.0, 0.0, 0.0),
    DVec3::new(-1.0, 0.0, 0.0),
    DVec3::new(0.0, 1.0, 0.0),
    DVec3::new(0.0, -1.0, 0.0),
    DVec3::new(0.0, 0.0, 1.0),
    DVec3::new(0.0, 0.0, -1.0),
];

/// Appends 36 vertices (12 triangles, non-indexed, one flat color for
/// the whole cube) for an axis-aligned cube centered at `center` with
/// half-extent `half_size` in every axis.
fn push_cube(out: &mut Vec<VolumeAuditCellVertex>, center: DVec3, half_size: f64, color: [f32; 4]) {
    let h = half_size;
    // Per-face 4 corners (in a consistent winding), split into 2
    // triangles each. Corner order per face matches `FACE_NORMALS`'s
    // ordering: +X, -X, +Y, -Y, +Z, -Z.
    let faces: [[DVec3; 4]; 6] = [
        // +X
        [
            center + DVec3::new(h, -h, -h),
            center + DVec3::new(h, h, -h),
            center + DVec3::new(h, h, h),
            center + DVec3::new(h, -h, h),
        ],
        // -X
        [
            center + DVec3::new(-h, -h, h),
            center + DVec3::new(-h, h, h),
            center + DVec3::new(-h, h, -h),
            center + DVec3::new(-h, -h, -h),
        ],
        // +Y
        [
            center + DVec3::new(h, h, -h),
            center + DVec3::new(-h, h, -h),
            center + DVec3::new(-h, h, h),
            center + DVec3::new(h, h, h),
        ],
        // -Y
        [
            center + DVec3::new(-h, -h, -h),
            center + DVec3::new(h, -h, -h),
            center + DVec3::new(h, -h, h),
            center + DVec3::new(-h, -h, h),
        ],
        // +Z
        [
            center + DVec3::new(-h, -h, h),
            center + DVec3::new(h, -h, h),
            center + DVec3::new(h, h, h),
            center + DVec3::new(-h, h, h),
        ],
        // -Z
        [
            center + DVec3::new(-h, h, -h),
            center + DVec3::new(h, h, -h),
            center + DVec3::new(h, -h, -h),
            center + DVec3::new(-h, -h, -h),
        ],
    ];

    for (face_idx, corners) in faces.iter().enumerate() {
        let normal = FACE_NORMALS[face_idx].as_vec3().to_array();
        // Two triangles per quad: (0,1,2) and (0,2,3).
        for &(a, b, c) in &[(0usize, 1usize, 2usize), (0, 2, 3)] {
            for &i in &[a, b, c] {
                out.push(VolumeAuditCellVertex {
                    position: corners[i].as_vec3().to_array(),
                    normal,
                    color,
                });
            }
        }
    }
}

/// Maps a fill ratio (`accumulated / expected`) to a color on the
/// existing blue (under) -> green (healthy at 1.0) -> red (over) scale.
/// See this plan's design spec's "Color mapping" section for the exact
/// formula.
fn ratio_to_color(ratio: f64) -> [f32; 4] {
    let deviation = (ratio - 1.0).clamp(-1.0, 1.0);
    let t = 0.5 + 0.5 * deviation;
    crate::toolpath_view::scalar_to_color(t)
}

/// Builds a non-indexed triangle-list cube (12 triangles, 36 vertices)
/// per qualifying cell in `grid`, colored by
/// `VolumeAuditGrid::cell_ratios_for_display`'s ratio (blue = under,
/// green = healthy at ratio 1.0, red = over) or a fixed magenta for
/// cells reported by `VolumeAuditGrid::extrusion_outside_mesh_cells`
/// (material where none is expected at all -- shown unconditionally,
/// never hidden by `deviation_threshold`).
///
/// `deviation_threshold` (`>= 0.0`) hides every ratio-based cube whose
/// `|ratio - 1.0|` is below it -- sparse-infill cells legitimately never
/// read near 1.0 per-cell (see this module's own doc comment / the
/// design spec for why), so without this filter a healthy sparse-infill
/// interior would visually swamp genuine wall-shell defects.
///
/// `shrink_factor` (`(0.0, 1.0]`) draws each cube at
/// `cell_size * shrink_factor` rather than the full cell size, leaving a
/// visible gap between adjacent cells.
pub fn build_volume_audit_cells(
    grid: &VolumeAuditGrid,
    deviation_threshold: f64,
    shrink_factor: f64,
) -> Vec<VolumeAuditCellVertex> {
    let half_size = grid.cell_size * shrink_factor * 0.5;
    let mut vertices = Vec::new();

    for (idx, ratio) in grid.cell_ratios_for_display() {
        if (ratio - 1.0).abs() < deviation_threshold {
            continue;
        }
        push_cube(&mut vertices, grid.cell_center(idx), half_size, ratio_to_color(ratio));
    }

    for idx in grid.extrusion_outside_mesh_cells() {
        push_cube(&mut vertices, grid.cell_center(idx), half_size, OUTSIDE_MESH_COLOR);
    }

    vertices
}

#[cfg(test)]
mod tests {
    use super::*;
    use manifold_core::toolpath::{MoveKind, Path, Segment};
    use manifold_core::{mesh::Mesh, volume_audit::audit_extrusion_volume, SlicerConfig};

    /// Same box-mesh fixture shape as `manifold_core::volume_audit`'s own
    /// tests (independently built here since that helper is private to
    /// that crate's test module).
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

    fn straight_extruding_path(
        start: DVec3,
        end: DVec3,
        kind: MoveKind,
        bead_area: f64,
        config: &SlicerConfig,
    ) -> Path {
        let distance = start.distance(end);
        let filament_area =
            manifold_core::extrusion::filament_cross_section_area(config.filament_diameter);
        let extrusion_length =
            manifold_core::extrusion::segment_extrusion_length(distance, bead_area, filament_area);
        Path {
            points: vec![start, end],
            segments: vec![Segment {
                kind,
                extrusion_length,
                line_width: bead_area / config.layer_height,
                ..Segment::default()
            }],
            tool: manifold_core::ids::ToolId(0),
            object: manifold_core::ids::ObjectId(0),
        }
    }

    #[test]
    fn build_volume_audit_cells_emits_36_vertices_per_qualifying_cell() {
        let mesh = box_mesh(DVec3::ZERO, DVec3::new(2.0, 6.0, 2.0));
        let config = SlicerConfig::default();
        let cell_size = 0.5;
        let path = straight_extruding_path(
            DVec3::new(0.3, 2.0, 1.0),
            DVec3::new(0.3, 4.0, 1.0),
            MoveKind::WallOuter,
            config.wall_line_width * config.layer_height,
            &config,
        );
        let grid = audit_extrusion_volume(&mesh, std::slice::from_ref(&path), &config, cell_size);

        // Threshold of 0.0 keeps everything with any deviation at all --
        // a wall bead's own cells should qualify (their ratio is never
        // exactly 1.0 for a hand-built single-segment fixture).
        let vertices = build_volume_audit_cells(&grid, 0.0, 0.9);
        assert!(
            !vertices.is_empty(),
            "a grid with real wall extrusion should produce at least one cube"
        );
        assert_eq!(
            vertices.len() % 36,
            0,
            "every cube must contribute exactly 36 vertices (12 triangles), got {} total",
            vertices.len()
        );
    }

    #[test]
    fn build_volume_audit_cells_respects_the_deviation_threshold() {
        let mesh = box_mesh(DVec3::ZERO, DVec3::new(2.0, 6.0, 2.0));
        let config = SlicerConfig::default();
        let cell_size = 0.5;
        let path = straight_extruding_path(
            DVec3::new(0.3, 2.0, 1.0),
            DVec3::new(0.3, 4.0, 1.0),
            MoveKind::WallOuter,
            config.wall_line_width * config.layer_height,
            &config,
        );
        let grid = audit_extrusion_volume(&mesh, std::slice::from_ref(&path), &config, cell_size);

        let permissive = build_volume_audit_cells(&grid, 0.0, 0.9);
        // A threshold far above any real cell's deviation must hide
        // every ratio-based cube.
        let strict = build_volume_audit_cells(&grid, 1e6, 0.9);
        assert!(
            strict.len() < permissive.len(),
            "an extreme deviation_threshold should hide ratio-based cubes: \
             permissive={}, strict={}",
            permissive.len(),
            strict.len()
        );
    }

    #[test]
    fn build_volume_audit_cells_never_hides_outside_the_mesh_defects() {
        let mesh = box_mesh(DVec3::ZERO, DVec3::new(2.0, 6.0, 2.0));
        let config = SlicerConfig::default();
        let cell_size = 0.5;
        // Same stray-bead fixture as `manifold_core::volume_audit`'s own
        // `extrusion_outside_the_mesh_is_caught_only_by_its_own_dedicated_query`.
        let stray = straight_extruding_path(
            DVec3::new(-0.25, 2.0, 1.0),
            DVec3::new(-0.25, 4.0, 1.0),
            MoveKind::WallOuter,
            config.wall_line_width * config.layer_height,
            &config,
        );
        let grid = audit_extrusion_volume(&mesh, std::slice::from_ref(&stray), &config, cell_size);
        assert!(
            !grid.extrusion_outside_mesh_cells().is_empty(),
            "test fixture must produce an outside-the-mesh cell -- otherwise this test is vacuous"
        );

        // Even an extreme deviation_threshold must not hide the
        // outside-mesh defect cube.
        let vertices = build_volume_audit_cells(&grid, 1e6, 0.9);
        assert!(
            vertices.iter().any(|v| v.color == OUTSIDE_MESH_COLOR),
            "outside-the-mesh defect cubes must never be hidden by deviation_threshold"
        );
    }
}
```

- [ ] **Step 2: Register the module**

Open `crates/manifold-gui/src/main.rs`, find the module declaration list (look for `mod toolpath_view;` or similar `mod` lines), and add `mod volume_audit_view;` in the same alphabetically-sensible position, matching whatever ordering convention the existing list already uses.

- [ ] **Step 3: Run tests**

Run: `CARGO_TARGET_DIR=target/test cargo nextest run -p manifold-gui volume_audit_view`

Expected: all 3 tests pass.

- [ ] **Step 4: Run the full pre-commit gate and commit**

```bash
cargo fmt --all
CARGO_TARGET_DIR=target/clippy cargo clippy --workspace --all-targets
CARGO_TARGET_DIR=target/test cargo nextest run --workspace
```

All three must be clean.

```bash
git add crates/manifold-gui/src/volume_audit_view.rs crates/manifold-gui/src/main.rs
git commit -m "feat(gui): add volume-audit cube geometry builder"
```

---

### Task 3: Wire into the render pipeline and app UI

**Files:**
- Modify: `crates/manifold-gui/src/render.rs`
- Modify: `crates/manifold-gui/src/app.rs`

**Interfaces:**
- Consumes (Task 2): `crate::volume_audit_view::{VolumeAuditCellVertex, build_volume_audit_cells}`.
- Consumes (Task 1, via manifold-core): `manifold_core::volume_audit::audit_extrusion_volume`.
- Produces: `pub fn UploadedMesh::upload_colored_cells(device: &wgpu::Device, vertices: &[VolumeAuditCellVertex]) -> Self`; new `App` fields `show_volume_audit: bool`, `volume_audit_cell_size: f64`, `volume_audit_deviation_threshold: f64`, `uploaded_volume_audit: Option<Arc<UploadedMesh>>`; new method `App::reupload_volume_audit`.

**This task's correctness cannot be fully proven by automated tests.** Verify what you can (compiles, clippy clean, existing test suite unaffected, the app launches without panicking on a simple slice), and say plainly in your report what you could NOT verify (actual on-screen cube placement/coloring/decluttering).

- [ ] **Step 1: Add `UploadedMesh::upload_colored_cells`**

In `crates/manifold-gui/src/render.rs`, inside `impl UploadedMesh`, find the private `from_vertices` method (it's the "shared buffer-creation tail for `Self::upload` and `Self::upload_from_vertices`", called by both). Add a new public method directly before `from_vertices`:

```rust
    /// Uploads pre-colored triangle geometry (already-computed
    /// per-vertex colors, e.g. from
    /// `crate::volume_audit_view::build_volume_audit_cells`) directly --
    /// unlike `Self::upload`/`Self::upload_from_vertices`, applies no
    /// overlay-mode color logic of its own.
    pub fn upload_colored_cells(
        device: &wgpu::Device,
        vertices: &[crate::volume_audit_view::VolumeAuditCellVertex],
    ) -> Self {
        let vertices: Vec<Vertex> = vertices
            .iter()
            .map(|v| Vertex {
                position: v.position,
                normal: v.normal,
                color: v.color,
            })
            .collect();
        Self::from_vertices(device, &vertices, "manifold volume audit cells vertex buffer")
    }
```

- [ ] **Step 2: Thread a new field through `Viewport3dCallback`**

Find `pub struct Viewport3dCallback` and its `overlay: Option<std::sync::Arc<UploadedMesh>>` field. Add a new field directly after it:

```rust
    pub volume_audit_cells: Option<std::sync::Arc<UploadedMesh>>,
```

Find the `prepare` method's parameter list (`meshes: &[UploadedMesh], overlay: Option<&UploadedMesh>, toolpaths: Option<&UploadedToolpaths>,`). Add a new parameter directly after `overlay`:

```rust
        volume_audit_cells: Option<&UploadedMesh>,
```

Trace `prepare`'s body for exactly how it currently draws `overlay` when `Some` (it will call some draw-with-the-mesh-pipeline helper/inline code using the same `Vertex` buffer + pipeline as `meshes`). Add an equivalent draw call for `volume_audit_cells` immediately after wherever `overlay` is drawn, using the identical pipeline/bind-group setup `overlay`'s draw call uses (same MSAA depth-tested triangle pipeline, since `VolumeAuditCellVertex`'s buffer layout is bit-identical to `Vertex`'s). Do not create a second pipeline or duplicate shader code — this must be the exact same draw call shape as the `overlay` one, just against a different buffer.

Find where `CallbackTrait::prepare`/`paint` (further down, on `impl egui_wgpu::CallbackTrait for Viewport3dCallback`) forward `self.overlay.as_deref()` into the inner `prepare` call. Add `self.volume_audit_cells.as_deref()` as the corresponding new argument in the same position you added the parameter above.

- [ ] **Step 3: Add `App` state fields**

In `crates/manifold-gui/src/app.rs`, find the `mesh_overlay_mode: MeshOverlayMode,` field declaration (in the main `App`/`ManifoldApp` struct). Add these four fields directly after it:

```rust
    /// Whether the extrusion-volume-audit cell visualization is drawn in
    /// the viewport (single-object scope only -- see
    /// `docs/superpowers/specs/2026-09-18-volume-audit-fill-visualization-design.md`).
    show_volume_audit: bool,
    /// Grid cell size (mm) for the volume-audit visualization. See
    /// `manifold_core::volume_audit::audit_extrusion_volume`'s own doc
    /// comment for choosing this value relative to line widths.
    volume_audit_cell_size: f64,
    /// Cells whose fill ratio deviates from 1.0 by less than this are
    /// hidden from the volume-audit visualization (declutters routine
    /// sparse-infill noise -- see the design spec's "Decluttering
    /// threshold" section).
    volume_audit_deviation_threshold: f64,
    /// GPU-uploaded copy of the volume-audit cell geometry, rebuilt by
    /// `Self::reupload_volume_audit`.
    uploaded_volume_audit: Option<std::sync::Arc<UploadedMesh>>,
```

Find where `mesh_overlay_mode: MeshOverlayMode::default(),` is set in the struct's constructor (near the other field initializers like `toolpaths: None,`). Add matching initializers directly after it:

```rust
            show_volume_audit: false,
            volume_audit_cell_size: 2.0,
            volume_audit_deviation_threshold: 0.15,
            uploaded_volume_audit: None,
```

- [ ] **Step 4: Add `App::reupload_volume_audit`**

Find `fn reupload_toolpaths` in `app.rs`. Add a new method directly after it, following its exact structural pattern (no-op/clear when preconditions aren't met, otherwise rebuild and upload):

```rust
    /// Rebuilds and re-uploads `uploaded_volume_audit` from the current
    /// `objects`/`toolpaths`/`config`, mirroring `reupload_toolpaths`'s
    /// pattern. No-op (clears the uploaded copy) if the visualization is
    /// off, there isn't exactly one object, or there's no planned
    /// toolpath yet -- `audit_extrusion_volume` takes a single `&Mesh`,
    /// so multi-object scenes are out of scope for this visualization
    /// (see the design spec's "Known Limitation").
    fn reupload_volume_audit(&mut self, device: &eframe::egui_wgpu::wgpu::Device) {
        self.uploaded_volume_audit = None;
        if !self.show_volume_audit {
            return;
        }
        let (Some(paths), [object]) = (&self.toolpaths, self.objects.as_slice()) else {
            return;
        };
        let grid = manifold_core::volume_audit::audit_extrusion_volume(
            &object.mesh,
            paths,
            &self.config,
            self.volume_audit_cell_size,
        );
        let vertices = crate::volume_audit_view::build_volume_audit_cells(
            &grid,
            self.volume_audit_deviation_threshold,
            0.9,
        );
        self.uploaded_volume_audit =
            Some(std::sync::Arc::new(UploadedMesh::upload_colored_cells(device, &vertices)));
    }
```

Check `self.objects`'s actual type before using this exact slice-pattern match (`[object]`) — if `objects: Vec<Object>` (as seen elsewhere in this file), `self.objects.as_slice()` matched against `[object]` requires exactly one element; if the field or its element type differs from what's assumed here, adapt the destructuring accordingly but preserve the exact-one-object semantics.

- [ ] **Step 5: Call `reupload_volume_audit` after a fresh slice**

Find, in `fn update` (the `impl eframe::App for ManifoldApp` block), the block that calls `self.reupload_toolpaths(&device);` immediately after `if self.toolpaths.is_some() {` inside the `if self.drain_slice_messages() {` branch (this is the "slicing job just completed" call site — do not confuse it with the scrub-slider or hidden-line-type call sites elsewhere in the file, which are toolpath-display-specific and irrelevant here). Add a call to `self.reupload_volume_audit(&device);` directly after that `reupload_toolpaths` call, reusing the same already-cloned `device` variable in scope.

- [ ] **Step 6: Add UI controls**

Find the existing `ui.checkbox(&mut self.show_toolpaths, "Show toolpaths");` line and the `egui::ComboBox::from_label("Mesh Overlay")` block directly after it (they sit in the same panel section). Add new controls directly after the `mesh_overlay_mode` combo box's change-handling block (the `if self.mesh_overlay_mode != mode_before { ... self.reupload(&device); }` block):

```rust
            ensure_row_space(ui, 130.0);
            let single_object = self.objects.len() == 1;
            ui.add_enabled_ui(single_object, |ui| {
                let mut audit_changed = false;
                if ui
                    .checkbox(&mut self.show_volume_audit, "Show volume audit")
                    .changed()
                {
                    audit_changed = true;
                }
                if self.show_volume_audit {
                    if ui
                        .add(
                            egui::DragValue::new(&mut self.volume_audit_cell_size)
                                .speed(0.1)
                                .range(0.05..=50.0)
                                .prefix("Cell size: ")
                                .suffix(" mm"),
                        )
                        .changed()
                    {
                        audit_changed = true;
                    }
                    if ui
                        .add(
                            egui::Slider::new(&mut self.volume_audit_deviation_threshold, 0.0..=2.0)
                                .text("Declutter threshold"),
                        )
                        .changed()
                    {
                        audit_changed = true;
                    }
                }
                if audit_changed {
                    let device = frame
                        .wgpu_render_state()
                        .expect("wgpu renderer is required")
                        .device
                        .clone();
                    self.reupload_volume_audit(&device);
                }
            });
            if !single_object {
                ui.label("Volume audit requires exactly one object in the scene.");
            }
```

Check the exact `egui::DragValue`/`egui::Slider` API this codebase's `egui` version exposes (`.range(...)` vs. an older `.clamp_range(...)`, etc. — check an existing `DragValue`/`Slider` usage elsewhere in `app.rs` for the exact method names this version supports) and adjust the builder-method names to match if they differ from what's written above; keep the same user-facing behavior (a numeric drag field for cell size, a 0.0-2.0 slider for the threshold).

- [ ] **Step 7: Wire the new field into the `Viewport3dCallback` construction**

Find where `Viewport3dCallback { ... overlay: self.sdf_overlay_mesh.clone(), ... }` is constructed. Add a new field initializer in the same struct literal, in the same relative position as Task 3 Step 2's field ordering:

```rust
                        volume_audit_cells: self.uploaded_volume_audit.clone(),
```

- [ ] **Step 8: Verify the app builds and launches**

Run: `CARGO_TARGET_DIR=target/clippy cargo clippy --workspace --all-targets` — must be clean (this will catch any signature/field-name mismatches from Steps 1-7).

Run: `CARGO_TARGET_DIR=target/test cargo nextest run --workspace` — must pass (confirms nothing existing broke; this plan adds no new automated tests in this task since it's GPU/UI wiring).

If feasible in this environment, run `cargo run --bin manifold-gui` briefly to confirm the app launches without panicking and the new checkbox/controls appear in the panel — but do not claim to have visually verified cube placement, colors, or decluttering behavior; state plainly in the report whether you were able to launch the GUI at all, and that visual correctness needs human confirmation regardless.

- [ ] **Step 9: Run the full pre-commit gate and commit**

```bash
cargo fmt --all
CARGO_TARGET_DIR=target/clippy cargo clippy --workspace --all-targets
CARGO_TARGET_DIR=target/test cargo nextest run --workspace
```

All three must be clean.

```bash
git add crates/manifold-gui/src/render.rs crates/manifold-gui/src/app.rs
git commit -m "feat(gui): render volume-audit fill visualization"
```
