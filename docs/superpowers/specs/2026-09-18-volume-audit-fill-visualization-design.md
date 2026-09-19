# Design: Volume Audit Fill Visualization

## Background

The extrusion-volume-audit tool (`crates/manifold-core/src/volume_audit.rs`,
shipped by `2026-09-18-extrusion-volume-audit` and refined by
`2026-09-18-infill-aware-volume-audit`) computes, per grid cell, an
expected volume and an accumulated (actual) volume, and exposes
assertion-scoped queries (`overfilled_cells`/`underfilled_cells`,
restricted to `Solid`-zone cells; `extrusion_outside_mesh_cells`, binary;
`infill_aggregate_ratio`, whole-grid). The original design's Background
named GPU visualization as a goal ("the accumulated volume per cell can
be visualized directly"), deliberately deferred in favor of a test-tool-first
priority. This plan builds that visualization in `manifold-gui`.

## Chosen Approach

Render each grid cell whose accumulated/expected ratio deviates from
1.0 by more than a user-adjustable threshold as a small, flat-colored
cube in the 3D viewport, colored on a blue (under) -> green (healthy) ->
red (over) scale. Reuses the existing triangle-mesh render pipeline
(`UploadedMesh` in `render.rs`) rather than building a new instanced-cube
shader/pipeline: cube geometry is built CPU-side as plain (position,
normal, color) triangles, in the same vertex layout the existing pipeline
already accepts.

### Why reuse the existing pipeline

`render.rs` already has a second, independent triangle-mesh slot
(`overlay: Option<Arc<UploadedMesh>>`, currently used for the SDF
isosurface preview) drawn through the same MSAA depth-tested pipeline as
regular object meshes. Adding a *third* independent slot
(`volume_audit_cells`) for this feature costs one new small `UploadedMesh`
constructor and one new optional field threaded through `prepare`/`paint`
— no new WGSL shader, no new bind groups, no new pipeline. `UploadedMesh`'s
existing two constructors (`upload`, taking a `&Mesh` with a fixed overlay
color scheme; `upload_from_vertices`, taking marching-cubes field vertices
with a fixed color) don't accept arbitrary per-vertex colors from a
caller, so this plan adds a third: `upload_colored_cells`, taking a slice
of a new, plain `#[repr(C)]` vertex struct with an explicit color field.

### Color mapping

**Superseded by direct user feedback after this feature shipped and was visually tested:** a fixed absolute scale (`deviation = (r - 1.0).clamp(-1.0, 1.0)`, `r = 0 -> blue`, `r = 1 -> green`, `r >= 2 -> red`) made nearly every visible cell in a real print read the same narrow shade of orange, because the systematic sparse-infill nominal-density mismatch (see `VolumeAuditGrid::overfilled_cells`'s doc comment) dominates the range with a roughly constant, non-defect deviation -- observed directly at ratio ~1.6-1.8 across most displayed cells. The mapping now auto-normalizes against the actual minimum and maximum ratio observed across every cell in the current audit run: `ratio == 1.0` always maps to the exact green midpoint, and the two sides are scaled independently against that run's own observed extremes (`[min_ratio, 1.0]` stretched to blue..green, `[1.0, max_ratio]` stretched to green..red), so whatever real variation is present is always visible regardless of scale. See `crates/manifold-gui/src/volume_audit_view.rs`'s `ratio_to_color` for the exact formula.

A cell holding material where **none** is expected at all (`expected ==
0`, i.e. the `extrusion_outside_mesh_cells` binary defect case) has no
meaningful ratio to map, so it gets a fixed, distinct color (bright
magenta) instead, and — being an unconditional defect, not a
tunable-tolerance deviation — is **never** filtered by the deviation
threshold below.

### Decluttering threshold

A single `deviation_threshold` (`f64`, GUI-exposed as a slider) hides
every ratio-based cube whose `|deviation| < deviation_threshold`, leaving
only cells that deviate from the healthy baseline by more than the
threshold. Motivating case (from user feedback during design): sparse
interior infill cells, even after the `infill-aware-volume-audit` fix
excluded them from the assertion-scoped per-cell queries, still carry a
per-cell ratio that legitimately isn't close to `1.0` for a correctly
printed sparse pattern (that's the whole reason that plan excluded them
from `overfilled_cells`/`underfilled_cells` in the first place) — without
a threshold, a sparse-infill interior would render as a wall of
mid-saturation cubes, obscuring genuine wall-shell defects. The
outside-mesh binary defect cubes above are explicitly exempted from this
filter since they represent a real defect at any magnitude.

## Components

### `crates/manifold-core/src/volume_audit.rs` — one new public query

```rust
/// `(cell index, ratio)` for every cell with `expected > 0` (i.e. every
/// `Solid`- or `SparseInfill`-zone cell), regardless of the ratio's
/// value. For display/visualization purposes -- unlike
/// [`VolumeAuditGrid::overfilled_cells`]/[`VolumeAuditGrid::underfilled_cells`],
/// this is not restricted to `Solid`-zone cells and carries no
/// pass/fail threshold; a caller doing defect *detection* should use
/// those instead.
pub fn cell_ratios_for_display(&self) -> Vec<([usize; 3], f64)>
```

Implemented directly against `self.zone`/`self.expected`/
`total_accumulated`, the same private fields `overfilled_cells` already
reads — no new SDF computation, no change to the expected/accumulated
model.

### `crates/manifold-gui/src/volume_audit_view.rs` (new) — pure geometry builder

Mirrors `toolpath_view.rs`'s existing separation ("pure geometry builders
— no GPU/wgpu types here").

```rust
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

/// Builds a non-indexed triangle-list cube (12 triangles, 36 vertices)
/// per qualifying cell in `grid`, colored by
/// [`manifold_core::volume_audit::VolumeAuditGrid::cell_ratios_for_display`]'s
/// ratio (blue = under, green = healthy at ratio 1.0, red = over) or a
/// fixed magenta for cells reported by
/// [`manifold_core::volume_audit::VolumeAuditGrid::extrusion_outside_mesh_cells`]
/// (material where none is expected at all -- shown unconditionally,
/// never hidden by `deviation_threshold`).
///
/// `deviation_threshold` (`>= 0.0`) hides every ratio-based cube whose
/// `|ratio - 1.0|` is below it -- see this plan's design spec for why
/// (sparse-infill cells legitimately never read near 1.0 per-cell).
///
/// `shrink_factor` (`(0.0, 1.0]`) draws each cube at
/// `cell_size * shrink_factor` rather than the full cell size, leaving a
/// visible gap between adjacent cells.
pub fn build_volume_audit_cells(
    grid: &manifold_core::volume_audit::VolumeAuditGrid,
    deviation_threshold: f64,
    shrink_factor: f64,
) -> Vec<VolumeAuditCellVertex>
```

Fully unit-testable without a GPU device: given a hand-built
`VolumeAuditGrid` (constructed the same way `volume_audit.rs`'s own tests
do, via `audit_extrusion_volume` on a small hand-built mesh + paths), the
returned vertex count, positions, and colors are asserted directly.

### `crates/manifold-gui/src/render.rs` — one new `UploadedMesh` constructor, one new callback field

```rust
impl UploadedMesh {
    /// Uploads pre-colored triangle geometry (already-computed per-vertex
    /// colors, e.g. from `crate::volume_audit_view::build_volume_audit_cells`)
    /// directly -- unlike `Self::upload`/`Self::upload_from_vertices`,
    /// applies no overlay-mode color logic of its own.
    pub fn upload_colored_cells(
        device: &wgpu::Device,
        vertices: &[crate::volume_audit_view::VolumeAuditCellVertex],
    ) -> Self
}
```

`Viewport3dCallback` gains one new field, `volume_audit_cells: Option<Arc<UploadedMesh>>`,
threaded through `prepare`/`paint` exactly like the existing `overlay`
field (same draw call, same pipeline, independent buffer).

### `crates/manifold-gui/src/app.rs` — state, rebuild, and UI

New `App` fields (naming mirrors the existing `show_toolpaths`/
`uploaded_toolpaths`/`mesh_overlay_mode` triple):
- `show_volume_audit: bool` (default `false`)
- `volume_audit_cell_size: f64` (default: a value at or above
  `config.wall_line_width`, per `audit_extrusion_volume`'s own
  documented lower-bound guidance — e.g. `2.0`, matching the value the
  audit module's own tests already use for real-print audits)
- `volume_audit_deviation_threshold: f64` (default `0.15` — small enough
  to surface real deviations, large enough to decluttre routine
  sparse-infill noise; exact default is a judgment call for the
  implementer, exposed as an adjustable slider regardless)
- `uploaded_volume_audit: Option<Arc<UploadedMesh>>`

New rebuild method, `reupload_volume_audit`, following
`reupload_toolpaths`'s exact pattern: no-op (clears the uploaded copy) if
`self.show_volume_audit` is off, or if `self.toolpaths`/`self.objects`
don't support the audit's single-mesh signature (**scope limit,
inherited from `audit_extrusion_volume`'s existing single-`&Mesh`
signature**: only compute/show when exactly one object is present;
multi-object audits are explicitly out of scope, matching the design
spec's own documented limitation for the underlying tool). Otherwise:
calls `manifold_core::volume_audit::audit_extrusion_volume` with the
sole object's mesh, `self.toolpaths`, `self.config`, and
`self.volume_audit_cell_size`, then
`crate::volume_audit_view::build_volume_audit_cells` with
`self.volume_audit_deviation_threshold`, then uploads via
`UploadedMesh::upload_colored_cells`.

Called from the same places `reupload_toolpaths` already is (after a
successful slice+plan, and whenever `volume_audit_cell_size`/
`volume_audit_deviation_threshold`/`show_volume_audit` changes) — mirror
the existing call sites rather than inventing new ones.

UI: a new checkbox (`show_volume_audit`) plus, when checked, a cell-size
drag-value and a deviation-threshold slider, placed in the same panel
section as the existing `mesh_overlay_mode` dropdown (both are
diagnostic-overlay controls). When more than one object is present,
disable the checkbox and show a short explanatory label instead of
silently doing nothing.

## Global Constraints

- Core geometry: `glam::DVec3`/f64 in `manifold-core`; the GPU-facing
  vertex struct uses `f32` per the existing `Vertex`/
  `ToolpathLineInstance` convention (`manifold-gui` is exempt from
  `manifold-core`'s f64-only rule; it already mixes both by necessity for
  GPU buffers).
- No changes to `expected_fill_fraction`/`classify_fill_zone`/the
  SDF-only grounding constraint — this plan only adds a new *read* over
  already-computed grid data.
- No new wgpu pipeline, shader, or bind group — reuse the existing
  triangle-mesh pipeline exactly.
- Repo pre-commit gate: `cargo fmt --all` -> `CARGO_TARGET_DIR=target/clippy
  cargo clippy --workspace --all-targets` -> `CARGO_TARGET_DIR=target/test
  cargo nextest run --workspace`.

## Known Limitation (disclosed, not fixed by this plan)

Single-object only, inherited directly from `audit_extrusion_volume`'s
existing `&Mesh` (not `&[Mesh]`) signature — already a documented
limitation of the underlying tool from the prior plan's final review,
not introduced here.

## Self-Review

- **Placeholder scan:** no TBDs; the deviation-threshold default is
  explicitly flagged as a judgment call for the implementer rather than
  hidden behind a vague instruction.
- **Internal consistency:** the new `cell_ratios_for_display` query and
  the existing `extrusion_outside_mesh_cells` query are read once each by
  `build_volume_audit_cells`, with no overlap (the former only considers
  `expected > 0` cells, the latter only `expected == 0` cells with
  material) — every rendered cube traces to exactly one of the two.
- **Scope check:** does not touch the audit's expected-volume model,
  assertion API, or any existing test; purely additive.
- **Ambiguity check:** the outside-mesh-defect color's exemption from the
  deviation threshold is stated explicitly, so a future reader can't
  mistake "some magenta cubes ignore the slider" for a bug.
