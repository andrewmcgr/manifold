# StepCalibration Real-Contour Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix `StepCalibration::adaptive_step` (`crates/manifold-core/src/slicing.rs`, added in commit `c281221`) so its layer-spacing estimate for non-`Height` order fields is calibrated from the actual isosurface cross-section at the current order value, instead of a window of scattered points selected only by order-value proximity — which produces real Z-spacing errors up to ±35% of `layer_height` on sloped geometry (confirmed by direct experiment on a shed-roof "wedge" test mesh).

**Architecture:** `StepCalibration` currently stores `(order_value, position)` sample pairs drawn from `wall_meshes[0]`'s raw isosurface point cloud, then at each step selects samples within a flat order-value window and takes the median of forward-probed deltas. This conflates "close in order value" with "close in physical space" — on folded/sloped geometry these are not the same thing, so the window can select spatially unrelated points. The fix: store a *borrowed* reference to `wall_meshes[0]`'s triangle soup (`&[DVec3]` positions + `&[f64]` orders) instead of precomputed samples, and at each `adaptive_step` call, extract the *real* loop(s) at the exact current `order_value` via `manifold_fidget::contour::extract_order_contours_on_mesh_with_debug` (the same function slicing already uses to build each layer's real wall-0 geometry later) — then forward-probe from those actual loop points. This naturally handles topology changes (a cross-section splitting into multiple loops, or merging) since it reads the real extracted topology at that exact order value, not a proxy window.

**Tech Stack:** Rust, `glam::DVec3`, `manifold_fidget::contour::extract_order_contours_on_mesh_with_debug` (already imported in `slicing.rs`), `manifold_core::order_field::numeric_gradient` (already used in this file).

**Spec:** No separate spec file — design was worked out and approved in chat (see conversation preceding this plan). This plan is self-contained.

## Global Constraints

- Core geometry uses `glam::DVec3` (f64) — no `f32`/`Vec3`.
- Keep today's two-phase structure in `slice_mesh_with_progress` unchanged: the sequential order-value stepping loop, followed by the parallel per-layer extraction pass. Do not restructure into a fully interleaved sequential extract-then-step loop — that would sacrifice today's `rayon`-parallel final extraction across all layers, which this plan does not require to fix the calibration accuracy problem.
- `StepCalibration`, `adaptive_step`, and `from_wall_pass` stay private to `slicing.rs`.
- No behavior change for `Height`: `calibration` stays `None` for `is_height`, exactly as today (`Height`'s own arc-length-exact stepping is untouched).
- After the task: `cargo fmt --all` → `cargo clippy --workspace --all-targets` → `cargo test --workspace` must all pass.

---

### Task 1: Rewrite `StepCalibration` to calibrate from the real per-step contour, add a permanent wedge-mesh regression test

**Files:**

- Modify: `crates/manifold-core/src/slicing.rs` (the `StepCalibration` struct/impl, currently around line 540-597 — re-read the file first to find the exact current lines, since this plan does not modify anything before it that would shift lines, but always verify)
- Modify: `crates/manifold-core/src/slicing.rs` test module — replace the 3 existing `step_calibration_adaptive_step_*` tests (currently using `ScaledLinearField` and scattered `(position, order)` pairs) with equivalents using real triangle-soup fixtures, since the new `StepCalibration` API takes triangle positions/orders, not arbitrary point samples
- Modify: `crates/manifold-core/src/slicing.rs` test module — add a permanent `wedge_mesh()` helper and a regression test asserting real Z-spacing stays within a tight tolerance of `layer_height` on that mesh (this plan's proof that the fix works, not a throwaway debug probe)

**Interfaces:**

- Produces: `struct StepCalibration<'a> { triangle_positions: &'a [DVec3], triangle_orders: &'a [f64] }` (private, replaces today's `struct StepCalibration { samples: Vec<(f64, DVec3)> }`)
- Produces: `fn StepCalibration::from_wall_pass(positions: &'a [DVec3], orders: &'a [f64]) -> Self` (private; same argument shapes as today's `from_wall_pass`, so the existing call site `wall_meshes.first().map(|(positions, orders)| StepCalibration::from_wall_pass(positions, orders))` needs no changes)
- Produces: `fn StepCalibration::adaptive_step(&self, field: &dyn OrderField, order_value: f64, layer_height: f64) -> f64` (private; same signature as today, so the existing call site `calibration.as_ref().map(|c| c.adaptive_step(&*field, order_value, layer_height)).unwrap_or(layer_height)` needs no changes)
- Consumes: `manifold_fidget::contour::extract_order_contours_on_mesh_with_debug` (already imported at the top of `slicing.rs`: `use manifold_fidget::contour::{extract_contours, extract_order_contours_on_mesh_with_debug, plane_basis};`), `order_field::numeric_gradient` (already used elsewhere in this file), `BUILD_DIRECTION` (already defined in this file)

- [ ] **Step 1: Write the failing tests**

First, locate the current `StepCalibration` struct and its three existing tests (search for `struct StepCalibration` and `step_calibration_adaptive_step_recovers_true_rate_from_uniform_samples`/`step_calibration_adaptive_step_takes_the_median_across_mixed_rate_regions`/`step_calibration_adaptive_step_falls_back_to_layer_height_with_no_samples` in the test module). Delete those 3 existing tests entirely (they test the old point-sampling API and will not compile against the new triangle-soup API) and replace them with:

```rust
    /// Two coplanar triangles forming a flat rectangular patch on a plane
    /// `z = plane_z`, spanning `[x_min, x_max] x [y_min, y_max]` -- a minimal
    /// closed contour source for `extract_order_contours_on_mesh_with_debug`
    /// to slice at any order value that brackets `plane_z * rate` (see
    /// `ScaledLinearField`, whose `order(p) = rate * p.z`, so slicing this
    /// patch's own triangles at any order_value gives back the same flat
    /// rectangle -- the patch's every vertex already sits exactly on that
    /// isosurface since the whole patch is coplanar in `z`).
    fn flat_patch_triangles(plane_z: f64, x_min: f64, x_max: f64, y_min: f64, y_max: f64) -> Vec<DVec3> {
        let v00 = DVec3::new(x_min, y_min, plane_z);
        let v10 = DVec3::new(x_max, y_min, plane_z);
        let v11 = DVec3::new(x_max, y_max, plane_z);
        let v01 = DVec3::new(x_min, y_max, plane_z);
        vec![v00, v10, v11, v00, v11, v01]
    }

    #[test]
    fn step_calibration_adaptive_step_recovers_true_rate_from_a_real_flat_patch() {
        let rate = 2.5;
        let field = ScaledLinearField { rate };
        let order_value = 4.0 * rate;
        let plane_z = order_value / rate;
        // Two triangles straddling the plane in Z (one vertex below,
        // rest above/on) so `extract_order_contours_on_mesh_with_debug`
        // has a real crossing to extract, not a degenerate coplanar patch.
        let below = DVec3::new(0.0, 0.0, plane_z - 1.0);
        let a = DVec3::new(-2.0, -2.0, plane_z + 1.0);
        let b = DVec3::new(2.0, -2.0, plane_z + 1.0);
        let c = DVec3::new(0.0, 2.0, plane_z + 1.0);
        let triangle_positions = vec![below, a, b, below, b, c, below, c, a];
        let triangle_orders: Vec<f64> = triangle_positions.iter().map(|p| field.order(*p)).collect();

        let calibration = StepCalibration::from_wall_pass(&triangle_positions, &triangle_orders);
        let layer_height = 0.2;
        let step = calibration.adaptive_step(&field, order_value, layer_height);

        // `order(p) = rate * p.z` is exact everywhere, so advancing
        // `layer_height` mm along the field's own gradient (straight up in
        // Z here) always yields exactly `rate * layer_height` more order,
        // regardless of which point on the real contour is probed.
        let expected = rate * layer_height;
        assert!(
            (step - expected).abs() < 1e-6,
            "expected step {expected}, got {step}"
        );
    }

    #[test]
    fn step_calibration_adaptive_step_falls_back_to_layer_height_when_no_contour_crosses_here() {
        let field = ScaledLinearField { rate: 2.5 };
        // A tiny flat patch far from the queried order value: no triangle
        // in this soup brackets it, so extraction finds no loop at all.
        let triangle_positions = flat_patch_triangles(0.0, -1.0, 1.0, -1.0, 1.0);
        let triangle_orders: Vec<f64> = triangle_positions.iter().map(|p| field.order(*p)).collect();
        let calibration = StepCalibration::from_wall_pass(&triangle_positions, &triangle_orders);

        let layer_height = 0.2;
        let step = calibration.adaptive_step(&field, 1000.0, layer_height);
        assert_eq!(
            step, layer_height,
            "with no contour crossing at this order value the step must fall back to the naive raw-unit step"
        );
    }

    #[test]
    fn step_calibration_adaptive_step_falls_back_to_layer_height_with_no_triangles() {
        let field = ScaledLinearField { rate: 2.5 };
        let calibration = StepCalibration::from_wall_pass(&[], &[]);
        let layer_height = 0.2;
        let step = calibration.adaptive_step(&field, 4.0, layer_height);
        assert_eq!(
            step, layer_height,
            "with an empty triangle soup the step must fall back to the naive raw-unit step"
        );
    }
```

Also add, in the same test module (near `big_cube_mesh`, reusing its doc-comment style):

```rust
    /// A 5x5 shed-roof wedge: flat base at z=0, walls up to z=1 on the y=0
    /// side and z=6 on the y=5 side, with a single sloped top face connecting
    /// them -- used to stress boundary-metric tangency/orthogonality blending
    /// with a real directional gradient tilt, unlike a flat-topped cube
    /// (whose flat top/vertical walls never tilt the field's local gradient
    /// away from an axis-aligned direction). Regression fixture for the
    /// `StepCalibration` real-contour rewrite: on this mesh, the old
    /// order-value-window sampling produced real Z-spacing between
    /// consecutive layers ranging from 0.166mm to 0.328mm against a 0.25mm
    /// target (a ~35% swing) -- see `step_calibration_keeps_real_z_spacing_close_to_target_on_a_sloped_wedge`.
    fn wedge_mesh() -> Mesh {
        let vertices = vec![
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(5.0, 0.0, 0.0),
            DVec3::new(5.0, 5.0, 0.0),
            DVec3::new(0.0, 5.0, 0.0),
            DVec3::new(0.0, 0.0, 1.0),
            DVec3::new(5.0, 0.0, 1.0),
            DVec3::new(5.0, 5.0, 6.0),
            DVec3::new(0.0, 5.0, 6.0),
        ];
        let indices = vec![
            0, 2, 1, 0, 3, 2, // -Z
            4, 5, 6, 4, 6, 7, // sloped top
            0, 1, 5, 0, 5, 4, // -Y
            3, 7, 6, 3, 6, 2, // +Y
            0, 4, 7, 0, 7, 3, // -X
            1, 2, 6, 1, 6, 5, // +X
        ];
        Mesh::new(vertices, indices)
    }

    #[test]
    fn step_calibration_keeps_real_z_spacing_close_to_target_on_a_sloped_wedge() {
        let config = SlicerConfig {
            layer_height: 0.25,
            order_field: crate::order_field::OrderFieldKind::AnisotropicFsm,
            fsm_boundary_metrics_enabled: true,
            fsm_top_tangency_aspect: Some(4.0),
            fsm_wall_ortho_aspect: Some(4.0),
            fsm_skin_depth_mm: Some(1.0),
            ..SlicerConfig::default()
        };
        let layers = slice_mesh(&wedge_mesh(), &config).unwrap();
        assert!(layers.len() >= 10, "expected a reasonably tall layer stack, got {}", layers.len());

        // Real physical Z-spacing between consecutive layers, measured from
        // each layer's own wall-0 loop centroid Z (the same quantity a user
        // would perceive as "how far apart are these layers printed").
        // Before this fix, this ranged from 0.166 to 0.328 against a 0.25
        // target on this exact mesh (a ~35% swing); this test locks in a
        // materially tighter bound as the fix's regression guard. The last
        // 1-2 layers near the model's own summit/tip are excluded: layer
        // count and exact tip geometry there are governed by separate
        // crown-layer-insertion logic (see the "Find intermediate feature
        // summits" section of `slice_mesh_with_progress`), not by
        // `StepCalibration`, and are not what this test is proving.
        let mut mean_zs: Vec<f64> = Vec::new();
        for l in &layers {
            let zs: Vec<f64> = l.loops.iter().flat_map(|w| w.points.iter().map(|p| p.z)).collect();
            if zs.is_empty() {
                continue;
            }
            mean_zs.push(zs.iter().sum::<f64>() / zs.len() as f64);
        }
        assert!(mean_zs.len() >= 10);
        let usable = &mean_zs[..mean_zs.len() - 2];
        for window in usable.windows(2) {
            let dz = window[1] - window[0];
            assert!(
                (0.20..=0.30).contains(&dz),
                "expected real Z-spacing within 20% of the 0.25 target, got {dz} between consecutive layers"
            );
        }
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p manifold-core --lib slicing::tests::step_calibration -- --nocapture`
Expected: FAIL to compile — `StepCalibration::from_wall_pass`/`adaptive_step` still have the old signature/behavior, and `flat_patch_triangles`/`wedge_mesh` may already exist or not compile against the still-old `StepCalibration`. The `step_calibration_keeps_real_z_spacing_close_to_target_on_a_sloped_wedge` test in particular is expected to compile and run, but FAIL its assertion (reproducing the real ~35% swing this plan fixes) — this is the RED evidence for the wedge regression specifically; the API-shape tests fail to compile until Step 3 lands.

- [ ] **Step 3: Write the implementation**

Replace the current `StepCalibration` struct and `impl StepCalibration` block (search for `struct StepCalibration` in `crates/manifold-core/src/slicing.rs`) with:

```rust
/// Calibrates the layer-stepping loop's order-value increments against real
/// physical distance for order fields whose local order-per-mm calibration
/// varies from point to point (see the doc comment on the stepping loop in
/// `slice_mesh_with_progress` for why a constant `+= layer_height` step is
/// wrong for those fields, and why `Height` is exempt).
///
/// Wraps a borrowed reference to `wall_meshes[0]`'s precomputed triangle
/// soup (already built once per object, at no extra global-computation cost
/// to reuse here) rather than owning a copy or a precomputed point sample:
/// `adaptive_step` re-extracts the *real* isosurface loop(s) at the exact
/// current `order_value` via `extract_order_contours_on_mesh_with_debug`
/// (the same function `slice_mesh_with_progress`'s own per-layer wall-0
/// extraction already uses later, on the same triangle soup) and
/// forward-probes from those actual points -- rather than an earlier
/// design that selected calibration samples from a flat order-value window
/// over the raw point cloud, which conflated "close in order value" with
/// "close in physical space." On folded/sloped geometry those are not the
/// same thing (a window can span spatially unrelated regions of the
/// isosurface), which produced real Z-spacing swings of up to ~35% of
/// `layer_height` on a sloped test mesh -- see
/// `step_calibration_keeps_real_z_spacing_close_to_target_on_a_sloped_wedge`.
/// Reading the real per-step topology also means this naturally handles a
/// cross-section splitting into multiple loops or merging, since whatever
/// loops actually exist at `order_value` are exactly what gets probed.
struct StepCalibration<'a> {
    triangle_positions: &'a [DVec3],
    triangle_orders: &'a [f64],
}

impl<'a> StepCalibration<'a> {
    /// Wraps one wall pass's precomputed triangle-soup isosurface (already
    /// computed for `wall_meshes` above, at no extra cost).
    fn from_wall_pass(triangle_positions: &'a [DVec3], triangle_orders: &'a [f64]) -> Self {
        Self {
            triangle_positions,
            triangle_orders,
        }
    }

    /// Estimates the order-value delta that advances roughly `layer_height`
    /// mm of real distance from `order_value`, as the median of forward
    /// probes (`field.order(pos + gradient_dir * layer_height) - order_value`)
    /// over every point of the real contour loop(s) `extract_order_contours_on_mesh_with_debug`
    /// finds at exactly `order_value` on this wall pass's isosurface.
    ///
    /// Falls back to `layer_height` verbatim -- today's pre-fix behavior --
    /// when the triangle soup is empty, when no loop actually crosses
    /// `order_value` here (e.g. before the object's own order range begins,
    /// or past where it ends), or when every candidate point's gradient is
    /// degenerate. The result is clamped to `[0.3, 3.0] * layer_height` as a
    /// safety bound against a single unrepresentative point producing a
    /// pathologically small or large step -- mirroring the clamp already
    /// used by `extrusion::local_layer_geometry` for the same reason.
    fn adaptive_step(&self, field: &dyn OrderField, order_value: f64, layer_height: f64) -> f64 {
        if self.triangle_positions.is_empty() {
            return layer_height;
        }

        let (loops, _debug_unclosed) = extract_order_contours_on_mesh_with_debug(
            self.triangle_positions,
            self.triangle_orders,
            order_value,
            BUILD_DIRECTION,
        );

        const MAX_SAMPLES: usize = 24;
        let all_points: Vec<DVec3> = loops.into_iter().flatten().collect();
        let stride = (all_points.len() / MAX_SAMPLES).max(1);

        let mut deltas: Vec<f64> = Vec::new();
        for pos in all_points.iter().step_by(stride) {
            let Some(grad) = order_field::numeric_gradient(field, *pos) else {
                continue;
            };
            let grad_len = grad.length();
            if !grad_len.is_finite() || grad_len < 1e-9 {
                continue;
            }
            let normal = grad / grad_len;
            let advanced = field.order(*pos + normal * layer_height);
            if !advanced.is_finite() {
                continue;
            }
            let delta = advanced - order_value;
            if delta.is_finite() && delta > 0.0 {
                deltas.push(delta);
            }
        }

        if deltas.is_empty() {
            return layer_height;
        }
        deltas.sort_by(f64::total_cmp);
        let median = deltas[deltas.len() / 2];
        median.clamp(0.3 * layer_height, 3.0 * layer_height)
    }
}
```

No changes are needed at either call site (`wall_meshes.first().map(|(positions, orders)| StepCalibration::from_wall_pass(positions, orders))` and `calibration.as_ref().map(|c| c.adaptive_step(&*field, order_value, layer_height)).unwrap_or(layer_height)`) — both argument shapes and the return type are unchanged from the old `StepCalibration`, only its internals and its struct's lifetime parameter changed. If the compiler reports a lifetime error at the `let calibration = ...` binding, add an explicit lifetime bound tying `StepCalibration<'_>` to `wall_meshes`'s own scope — `wall_meshes` is a local `Vec` that lives for the rest of the function, so this should resolve without needing to restructure anything.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p manifold-core --lib slicing::tests::step_calibration -- --nocapture`
Expected: PASS (all `step_calibration_*` tests, including the wedge regression test with real Z-spacing now within the `0.20..=0.30` band).

Then run the full `manifold-core` lib suite once: `cargo test -p manifold-core --lib`
Expected: PASS, no new failures versus the current baseline (confirm the current baseline first with `git stash` + a baseline run if there is any doubt, then `git stash pop`, though this plan's change is narrowly scoped to `StepCalibration` and its own tests so no other test should be affected).

- [ ] **Step 5: Run the full pre-commit gate**

Run, in order:

```bash
cargo fmt --all
cargo clippy --workspace --all-targets
cargo test --workspace
```

This last full-workspace run takes several minutes — if your environment has a tool timeout under ~10 minutes, run it as a background/detached process and poll, don't let it time out silently.

Expected: `fmt` makes no unexpected changes beyond this task's own new code; `clippy` reports 0 new warnings; `test` passes fully.

- [ ] **Step 6: Commit**

```bash
git add crates/manifold-core/src/slicing.rs
git commit -m "fix(core): calibrate layer stepping from the real per-step contour"
```

This repo's commit hook requires the subject line ≤72 characters in `<type>(<scope>): <subject>` format. If `cargo fmt --all` in Step 5 produced any diff beyond this change's own new code, commit that separately: `git commit -m "style: cargo fmt"`.

---

## Self-Review

**Spec coverage:** The design agreed in chat had one specific mechanism change (calibrate from the real per-step contour instead of an order-value-windowed point cloud) plus a proof it actually fixes the reported problem (the wedge regression test). Both are in this single task.

**Placeholder scan:** No TBDs; every step has literal code.

**Type consistency:** `StepCalibration<'a> { triangle_positions: &'a [DVec3], triangle_orders: &'a [f64] }`'s fields are used identically in `from_wall_pass` and `adaptive_step`. `from_wall_pass(triangle_positions: &'a [DVec3], triangle_orders: &'a [f64]) -> Self` and `adaptive_step(&self, field: &dyn OrderField, order_value: f64, layer_height: f64) -> f64` match the existing call sites' argument shapes exactly (verified against current source in `crates/manifold-core/src/slicing.rs` around the `let calibration = ...` and `let step = calibration...` lines), so no other code in `slice_mesh_with_progress` needs to change.

---

### Task 2 (final-review fix wave): bound `adaptive_step`'s per-call cost, strengthen the wedge test against a degenerate-constant regression

Added after the final whole-branch review found two Important issues in Task 1's otherwise-approved implementation:

1. **Unbounded per-layer cost.** `adaptive_step` now calls `extract_order_contours_on_mesh_with_debug` on the *entire* `wall_meshes[0]` triangle soup every stepping-loop iteration -- O(T) per layer, serially, where T can be millions of triangles for a large part at the clamped 0.04-0.10mm marching-cubes cell size (`slicing.rs:854-856`). The old implementation was O(log T) via `partition_point` binary search over a presorted sample array. There is no existing spatial index in this codebase suited to fixing this: `TriangleBvh` (`crates/manifold-fidget/src/geometry.rs:209`) indexes by 3D position for nearest-triangle/distance queries, not by 1D order-value range, which is the actual query shape needed here ("find every triangle whose order range brackets this scalar" -- an interval-stabbing query, not a spatial one).
2. **The wedge regression test only proves smoothness, not absolute correctness.** A regression to a constant-output implementation would still produce ratios of exactly `1.0` and pass. `step_calibration_adaptive_step_recovers_true_rate_from_a_real_flat_patch` catches a *total* regression to the `layer_height` fallback via an analytic oracle, but nothing catches a smooth-but-wrong regression on a real distorted field.

**Files:**

- Modify: `crates/manifold-core/src/slicing.rs` (`StepCalibration` struct/impl, added in Task 1's commit `6731cd1` -- re-read the file first to find the exact current lines)
- Modify: `crates/manifold-core/src/slicing.rs` test module (`step_calibration_keeps_order_step_stable_across_a_sloped_wedge`, added in Task 1)

**Interfaces:**

- Produces: `StepCalibration<'a>` gains two new private fields (`sorted_ranges: Vec<(f64, f64, usize)>`, `prefix_max_order: Vec<f64>`) alongside its existing `triangle_positions`/`triangle_orders`. `from_wall_pass`'s signature is unchanged (`(triangle_positions: &'a [DVec3], triangle_orders: &'a [f64]) -> Self`); `adaptive_step`'s signature is unchanged (`(&self, field: &dyn OrderField, order_value: f64, layer_height: f64) -> f64`). No call site elsewhere in `slicing.rs` needs to change.

- [ ] **Step 1: Write the failing tests**

Add to the `#[cfg(test)] mod tests` block, near the existing `step_calibration_*` tests:

```rust
    #[test]
    fn step_calibration_bracket_index_finds_a_triangle_whose_range_is_far_from_the_binary_search_boundary() {
        // Three small triangles clustered near order 100, plus one large
        // triangle whose own order range is [0, 200] -- i.e. it brackets
        // order_value=50 despite its `min_order` (0) sitting far below the
        // `partition_point` boundary for order_value=50. A naive
        // sort-by-min-order-then-stop-at-the-first-non-bracketing-triangle
        // scan would mistakenly miss it if it terminated too early; the
        // prefix-max-of-max-order structure must still find it.
        let big = vec![
            DVec3::new(-100.0, -100.0, 0.0),
            DVec3::new(100.0, -100.0, 0.0),
            DVec3::new(0.0, 100.0, 200.0),
        ];
        let small_a = vec![
            DVec3::new(0.0, 0.0, 99.0),
            DVec3::new(1.0, 0.0, 100.0),
            DVec3::new(0.0, 1.0, 101.0),
        ];
        let small_b = vec![
            DVec3::new(10.0, 0.0, 99.5),
            DVec3::new(11.0, 0.0, 100.5),
            DVec3::new(10.0, 1.0, 101.5),
        ];
        let field = ScaledLinearField { rate: 1.0 };
        let mut triangle_positions = big;
        triangle_positions.extend(small_a);
        triangle_positions.extend(small_b);
        let triangle_orders: Vec<f64> = triangle_positions.iter().map(|p| field.order(*p)).collect();

        let calibration = StepCalibration::from_wall_pass(&triangle_positions, &triangle_orders);
        let layer_height = 0.2;
        let step = calibration.adaptive_step(&field, 50.0, layer_height);

        // `order(p) = p.z` is exact everywhere, so the correct step is
        // exactly `layer_height` regardless of which bracketing triangle
        // is probed -- this only passes if the big triangle (whose range
        // is far from the small ones clustered at order ~100) is actually
        // found and probed.
        assert!(
            (step - layer_height).abs() < 1e-6,
            "expected step {layer_height}, got {step} -- the wide-range triangle may have been missed"
        );
    }

    #[test]
    fn step_calibration_keeps_order_step_stable_across_a_sloped_wedge_and_not_degenerately_constant() {
        let config = SlicerConfig {
            layer_height: 0.25,
            order_field: crate::order_field::OrderFieldKind::AnisotropicFsm,
            fsm_boundary_metrics_enabled: true,
            fsm_top_tangency_aspect: Some(4.0),
            fsm_wall_ortho_aspect: Some(4.0),
            fsm_skin_depth_mm: Some(1.0),
            ..SlicerConfig::default()
        };
        let layers = slice_mesh(&wedge_mesh(), &config).unwrap();
        assert!(layers.len() >= 10);

        let orders: Vec<f64> = layers.iter().map(|l| l.order).collect();
        let dorders: Vec<f64> = orders.windows(2).map(|w| w[1] - w[0]).collect();
        assert!(dorders.len() >= 9);
        let interior = &dorders[1..dorders.len() - 2];

        // A regression to a constant-output `adaptive_step` (e.g. always
        // returning `layer_height` verbatim, silently disabling
        // calibration entirely) would still pass the existing ratio-based
        // smoothness assertion -- guard against that specific degenerate
        // case directly: the calibrated step must actually differ from
        // the raw `layer_height` on this genuinely distorted field, and
        // must never sit exactly on the `[0.3, 3.0] * layer_height` clamp
        // rails (which would indicate the calibration is being clamped
        // away rather than converging to a real local rate).
        let clamp_lo = 0.3 * config.layer_height;
        let clamp_hi = 3.0 * config.layer_height;
        for &d in interior {
            assert!(
                (d - config.layer_height).abs() > 1e-6,
                "expected the calibrated step to differ from the uncalibrated layer_height, got {d}"
            );
            assert!(
                d > clamp_lo + 1e-9 && d < clamp_hi - 1e-9,
                "expected the calibrated step to sit strictly inside the clamp rails [{clamp_lo}, {clamp_hi}], got {d}"
            );
        }
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p manifold-core --lib slicing::tests::step_calibration -- --nocapture`
Expected: `step_calibration_bracket_index_finds_a_triangle_whose_range_is_far_from_the_binary_search_boundary` FAILS to compile (the `sorted_ranges`/`prefix_max_order` fields and bracket-search behavior don't exist yet in `from_wall_pass`/`adaptive_step`). `step_calibration_keeps_order_step_stable_across_a_sloped_wedge_and_not_degenerately_constant` should compile and PASS already against the current (Task 1) implementation -- it's a strengthening of an already-correct implementation, not expected to be RED on its own; if it unexpectedly fails, investigate before proceeding rather than assuming Step 3 will fix it.

- [ ] **Step 3: Write the implementation**

Replace the current `StepCalibration` struct and its `impl` block with:

```rust
struct StepCalibration<'a> {
    triangle_positions: &'a [DVec3],
    triangle_orders: &'a [f64],
    /// Per-triangle `(min_order, max_order, triangle_index)`, sorted
    /// ascending by `min_order`, paired with `prefix_max_order[i]` =
    /// `max(sorted_ranges[0].1 ..= sorted_ranges[i].1)`. Lets
    /// `adaptive_step` find every triangle whose order range brackets the
    /// current `order_value` in typically-sublinear time -- binary search
    /// to the last triangle with `min_order <= order_value`, then a
    /// backward scan that stops the moment `prefix_max_order` proves no
    /// earlier triangle's `max_order` could reach `order_value` either --
    /// instead of the O(T) full-soup scan `extract_order_contours_on_mesh_with_debug`
    /// would otherwise need every stepping-loop iteration, where T can be
    /// millions of triangles for a large part. There is no existing
    /// spatial index in this codebase suited to this: `TriangleBvh`
    /// indexes by 3D position for nearest-triangle queries, not by 1D
    /// order-value range, which is the actual query shape here (an
    /// "interval stabbing" query). This is a performance structure only
    /// -- it never omits a triangle that genuinely brackets `order_value`;
    /// a missed calibration point degrades accuracy silently, which the
    /// existing empty-`deltas`-falls-back-to-`layer_height` path already
    /// guards against, but is not a substitute for actually finding every
    /// real candidate.
    sorted_ranges: Vec<(f64, f64, usize)>,
    prefix_max_order: Vec<f64>,
}

impl<'a> StepCalibration<'a> {
    /// Wraps one wall pass's precomputed triangle-soup isosurface (already
    /// computed for `wall_meshes` above, at no extra cost), and builds the
    /// order-range bracket index described on the struct.
    fn from_wall_pass(triangle_positions: &'a [DVec3], triangle_orders: &'a [f64]) -> Self {
        let triangle_count = triangle_positions.len() / 3;
        let mut sorted_ranges: Vec<(f64, f64, usize)> = (0..triangle_count)
            .filter_map(|t| {
                let os = &triangle_orders[t * 3..t * 3 + 3];
                if os.iter().any(|o| !o.is_finite()) {
                    return None;
                }
                let min_o = os.iter().cloned().fold(f64::INFINITY, f64::min);
                let max_o = os.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                Some((min_o, max_o, t))
            })
            .collect();
        sorted_ranges.sort_by(|a, b| a.0.total_cmp(&b.0));

        let mut running_max = f64::NEG_INFINITY;
        let prefix_max_order: Vec<f64> = sorted_ranges
            .iter()
            .map(|&(_, max_o, _)| {
                running_max = running_max.max(max_o);
                running_max
            })
            .collect();

        Self {
            triangle_positions,
            triangle_orders,
            sorted_ranges,
            prefix_max_order,
        }
    }

    /// Estimates the order-value delta that advances roughly `layer_height`
    /// mm of real distance from `order_value`, as the median of forward
    /// probes over every point of the real contour loop(s)
    /// `extract_order_contours_on_mesh_with_debug` finds at exactly
    /// `order_value`, restricted to the subset of triangles whose own
    /// order range brackets `order_value` (via `sorted_ranges`/`prefix_max_order`)
    /// rather than the whole triangle soup.
    ///
    /// Falls back to `layer_height` verbatim -- today's pre-fix behavior
    /// -- when the triangle soup is empty, when no triangle's range
    /// brackets `order_value` (e.g. before the object's own order range
    /// begins, or past where it ends), or when every candidate point's
    /// gradient is degenerate. The result is clamped to `[0.3, 3.0] *
    /// layer_height` as a safety bound against a single unrepresentative
    /// point producing a pathologically small or large step -- mirroring
    /// the clamp already used by `extrusion::local_layer_geometry` for the
    /// same reason.
    fn adaptive_step(&self, field: &dyn OrderField, order_value: f64, layer_height: f64) -> f64 {
        if self.sorted_ranges.is_empty() {
            return layer_height;
        }

        // `extract_order_contours_on_mesh_with_debug` internally offsets
        // its own target by `1e-5 * order_value.abs().max(1.0)`; widen the
        // bracket test by twice that margin so a triangle whose range just
        // barely excludes the raw `order_value` but would include the
        // function's own perturbed target is never incorrectly skipped.
        let margin = 2e-5 * order_value.abs().max(1.0);

        let hi = self
            .sorted_ranges
            .partition_point(|&(min_o, _, _)| min_o <= order_value + margin);
        if hi == 0 {
            return layer_height;
        }

        let mut bracketing_triangles: Vec<usize> = Vec::new();
        for i in (0..hi).rev() {
            if self.prefix_max_order[i] < order_value - margin {
                break;
            }
            let (_, max_o, t) = self.sorted_ranges[i];
            if max_o >= order_value - margin {
                bracketing_triangles.push(t);
            }
        }
        if bracketing_triangles.is_empty() {
            return layer_height;
        }

        let mut positions: Vec<DVec3> = Vec::with_capacity(bracketing_triangles.len() * 3);
        let mut orders: Vec<f64> = Vec::with_capacity(bracketing_triangles.len() * 3);
        for &t in &bracketing_triangles {
            positions.extend_from_slice(&self.triangle_positions[t * 3..t * 3 + 3]);
            orders.extend_from_slice(&self.triangle_orders[t * 3..t * 3 + 3]);
        }

        let (loops, _debug_unclosed) = extract_order_contours_on_mesh_with_debug(
            &positions,
            &orders,
            order_value,
            BUILD_DIRECTION,
        );

        const MAX_SAMPLES: usize = 24;
        let all_points: Vec<DVec3> = loops.into_iter().flatten().collect();
        if all_points.is_empty() {
            return layer_height;
        }
        let stride = (all_points.len() / MAX_SAMPLES).max(1);

        let mut deltas: Vec<f64> = Vec::new();
        for pos in all_points.iter().step_by(stride) {
            let Some(grad) = order_field::numeric_gradient(field, *pos) else {
                continue;
            };
            let grad_len = grad.length();
            if !grad_len.is_finite() || grad_len < 1e-9 {
                continue;
            }
            let normal = grad / grad_len;
            let advanced = field.order(*pos + normal * layer_height);
            if !advanced.is_finite() {
                continue;
            }
            let here = field.order(*pos);
            let delta = advanced - here;
            if delta.is_finite() && delta > 0.0 {
                deltas.push(delta);
            }
        }

        if deltas.is_empty() {
            return layer_height;
        }
        deltas.sort_by(f64::total_cmp);
        let median = deltas[deltas.len() / 2];
        median.clamp(0.3 * layer_height, 3.0 * layer_height)
    }
}
```

Replace `step_calibration_keeps_order_step_stable_across_a_sloped_wedge` with `step_calibration_keeps_order_step_stable_across_a_sloped_wedge_and_not_degenerately_constant` from Step 1 above (same test, same fixture, with the two added degenerate-constant/clamp-rail assertions) -- delete the old one, don't leave both.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p manifold-core --lib slicing::tests::step_calibration -- --nocapture`
Expected: PASS (all `step_calibration_*` tests, including the new bracket-index test and the strengthened wedge test).

Then run the full `manifold-core` lib suite once: `cargo test -p manifold-core --lib`
Expected: PASS, no new failures.

- [ ] **Step 5: Measure the actual performance change**

Build a release binary and slice a real, non-trivial `AnisotropicFsm` model (any `.stl` available in the repo or test fixtures large enough to be representative -- check for existing example meshes under `crates/manifold-cli/examples/` or similar; if none exist at a useful scale, generate one, e.g. a scaled-up sphere or cone mesh with a few hundred thousand triangles) with a fine `layer_height` (e.g. 0.1mm) to produce several hundred layers, and time the slice (e.g. `cargo build --release -p manifold-cli` then time `manifold slice ...` or an equivalent existing CLI/example invocation). Compare against the same slice on the commit immediately before this task's changes (`6731cd1`, i.e. Task 1's own final state) to characterize the actual wall-clock difference this task's bracket-index optimization makes versus Task 1's O(T)-per-layer baseline. Record the before/after timing in your report -- this doesn't need to be a micro-benchmark harness, a wall-clock comparison of one representative slice is sufficient to characterize whether the optimization meaningfully helps and whether remaining cost is acceptable.

- [ ] **Step 6: Run the full pre-commit gate**

Run, in order:
```bash
cargo fmt --all
cargo clippy --workspace --all-targets
cargo test --workspace
```
This last run takes several minutes -- run as a background/detached process and poll if your environment's tool timeout is short.

- [ ] **Step 7: Commit**

```bash
git add crates/manifold-core/src/slicing.rs
git commit -m "perf(core): bound adaptive_step's per-layer triangle scan"
```
If `cargo fmt --all` produced any diff beyond this task's own new code, commit that separately: `git commit -m "style: cargo fmt"`.

---

## Self-Review (Task 2 addendum)

**Spec coverage:** Both final-review Important findings (unbounded per-layer cost, weak wedge-test invariant) are addressed by this single task -- the bracket index for the first, the strengthened test plus a new correctness test for the index itself for the second.

**Placeholder scan:** No TBDs. Step 5's performance measurement is exploratory by nature (record actual numbers, not a pre-decided pass/fail threshold) since no baseline measurement exists yet to compare against -- this is intentional, not a placeholder.

**Type consistency:** `StepCalibration<'a>`'s new `sorted_ranges: Vec<(f64, f64, usize)>` and `prefix_max_order: Vec<f64>` fields are constructed identically in `from_wall_pass` and consumed identically in `adaptive_step`. `from_wall_pass`/`adaptive_step`'s public signatures (from the caller's perspective) are unchanged from Task 1, so no other code in `slicing.rs` needs to change.
