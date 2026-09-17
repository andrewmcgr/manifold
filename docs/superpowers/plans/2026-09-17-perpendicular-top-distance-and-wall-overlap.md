# Perpendicular Top-Distance & Wall/Solid-Fill Overlap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix two related regressions in `manifold-core`'s solid-fill mechanism: (1) `seed_proximity`'s top-surface march reports raw vertical travel distance instead of true perpendicular distance to the top surface, which under `SlicerConfig::default()`'s nozzle settings makes steeply tapering tops never get solid fill at all; (2) inner wall loops and solid infill can print over the same physical space near a taper's tip, where the local cross-section is narrower than the full wall stack.

**Architecture:** Redesign `TopSurfaceAwareOrderField::march_to_top` to project its accumulated climb-direction distance onto the local surface normal (from the mesh SDF's own gradient, already computed for free at the exit sample) rather than reporting raw travel distance. Extract the existing per-layer "is this point solid-fill-eligible" computation out of `compute_solid_fill_boundaries` into a reusable function callable independently of `infill_boundary`, and use it at wall-generation time to clip inner wall loops (dropping the portions that fall inside the eligible region) before `infill_boundary`/`solid_fill_boundary` even exist.

**Tech Stack:** Rust, `glam::DVec3`, existing `manifold_fidget::mesh_sdf::MeshSdf`/`FieldSample`, existing `polygon2d` module.

**Spec:** `docs/superpowers/specs/2026-09-17-perpendicular-top-distance-and-wall-overlap-design.md`

**Deviation from spec, discovered during this plan's own drafting (not an implementer deviation — corrected here before any task starts):** the spec's Design section 3 says to reuse `polygon2d::difference` to subtract the eligible region from inner wall loops, by analogy with `toolpath::plan`'s existing bridge/overhang subtraction. That analogy doesn't hold: `toolpath::plan` applies `difference` to `sparse_loops`/`all_solid_loops`, which are **filled-region boundaries** consumed by infill pattern generators (their interior gets filled). A `WallLoop.points` is a **stroke centerline** (a single bead's path to extrude along), not a filled region — a polygon-polygon boolean difference doesn't apply to a curve. Task 3 below instead samples each of the wall loop's own existing points against the eligible region with `polygon2d::point_in_polygon` (already used elsewhere in this file) and keeps only the surviving "outside" runs — the correct primitive for clipping a curve, and one that never synthesizes a new point (so no reprojection/reconstruction step is needed for this operation, unlike everywhere else in this file that deals with newly-computed contour points).

## Global Constraints

- Core geometry uses `glam::DVec3` (f64) — no `f32`/`Vec3`.
- `manifold-core` uses `thiserror` for its `Error` enum; this plan touches no error paths, so no new variants are needed.
- Logging via `tracing`; this plan adds no new logging.
- After all tasks: `cargo fmt --all` -> `cargo clippy --workspace --all-targets` -> `cargo test --workspace` must all pass.
- `SeedKind::Bed`'s distance (`inner.order(p)`) is untouched by this plan — only `SeedKind::Patch`'s distance changes.
- The outer wall (`wall_index == 0`) is never clipped or dropped by this plan, under any circumstance — only `wall_index >= 1` loops are subject to clipping.
- This plan does not touch the plane-basis frame mismatch (`plane_basis(BUILD_DIRECTION)` vs `plane_basis(axis)`) or infill slope correction — both are out of scope, tracked separately (see spec's Backlog).
- This repo's commit hook requires the subject line ≤72 characters in `<type>(<scope>): <subject>` format.

---

### Task 1: Perpendicular-distance march in `TopSurfaceAwareOrderField`

**Files:**

- Modify: `crates/manifold-core/src/order_field.rs` (re-read the file first — line numbers below are from this plan's drafting and may have shifted)

**Interfaces:**

- Modifies: `TopSurfaceAwareOrderField::march_to_top(&self, p: DVec3, initial_value: f64, search_bound: f64) -> Option<f64>` — same signature, corrected return value.
- Consumes: `manifold_fidget::FieldSample { value: f64, gradient: DVec3 }` (already returned by `MeshSdf::sample`, whose `gradient` is already the local outward surface normal — confirmed directly against `MeshSdf::sample`'s implementation in `crates/manifold-fidget/src/mesh_sdf.rs`, which builds `gradient: sign * gradient_dir` where `gradient_dir` always points away from the nearest surface point, consistently oriented outward regardless of whether `p` is inside or outside). No new helper needed — `self.bed_excluded_sdf.sample(pos)` already returns this; the current code just discards `.gradient` and keeps only `.value`.

- [ ] **Step 1: Write the failing test**

Add to `crates/manifold-core/src/order_field.rs`'s test module (near the existing `top_surface_aware_order_field_*` tests):

```rust
    #[test]
    fn top_surface_aware_order_field_reports_perpendicular_not_vertical_distance_on_a_slope() {
        // Same cone as `top_surface_aware_order_field_finds_a_tapering_cone_apex_...`
        // (base_radius=5, apex_z=10, 32 segments), but queried at a
        // non-apex, non-step-aligned point on the slope, where the raw
        // vertical march distance and the true perpendicular distance
        // genuinely differ -- the apex-tip query in the existing test
        // exits on its very first hop, so it can't distinguish "reports
        // vertical distance" from "reports perpendicular distance."
        let base_radius = 5.0;
        let apex_z = 10.0;
        let segments = 32;
        let mut vertices = vec![DVec3::new(0.0, 0.0, 0.0)];
        for i in 0..segments {
            let angle = (i as f64) / (segments as f64) * std::f64::consts::TAU;
            vertices.push(DVec3::new(base_radius * angle.cos(), base_radius * angle.sin(), 0.0));
        }
        vertices.push(DVec3::new(0.0, 0.0, apex_z));
        let apex_idx = vertices.len() as u32 - 1;
        let mut indices = Vec::new();
        for i in 0..segments {
            let a = 1 + i as u32;
            let b = 1 + ((i + 1) % segments) as u32;
            indices.extend_from_slice(&[0, b, a]);
            indices.extend_from_slice(&[a, b, apex_idx]);
        }
        let mesh = Mesh::new(vertices, indices);
        let config = crate::SlicerConfig {
            order_field: OrderFieldKind::Height,
            layer_height: 0.2,
            top_layers: 3,
            ..crate::SlicerConfig::default()
        };
        let field = order_field_for(
            config.order_field,
            &config,
            &mesh,
            &manifold_fidget::slope_profile::SlopeProfile::new(Vec::new()),
        );

        // Query point at (r0=1.03, z0=5.0), on the cone's axis-aligned
        // radial slice: radius(z) = 5.0 * (1.0 - z / 10.0), so this point
        // is inside the solid (radius(5.0) = 2.5 > 1.03). Marching
        // straight up, it exits exactly where radius(z_exit) == 1.03:
        // z_exit = 10.0 * (1.0 - 1.03 / 5.0) = 7.94, so raw vertical
        // march distance = 7.94 - 5.0 = 2.94. The default step
        // (layer_height.min(nozzle_diameter) / 4 = 0.05 here) does not
        // divide 2.94 evenly, so this genuinely exercises the march's
        // interpolation, unlike a step-aligned point.
        let query = DVec3::new(1.03, 0.0, 5.0);
        let (kind, distance) = field.seed_proximity(query).unwrap();
        assert_eq!(kind, manifold_fidget::order::SeedKind::Patch);

        let raw_vertical_distance = 2.94_f64;
        let k = base_radius / apex_z; // radial slope magnitude, dr/dz
        let cos_theta = k / (1.0 + k * k).sqrt(); // angle between climb direction and surface normal
        let expected_perpendicular = raw_vertical_distance * cos_theta;

        assert!(
            (distance - expected_perpendicular).abs() < 0.02,
            "expected perpendicular distance ~{expected_perpendicular}, got {distance} \
             (raw vertical distance would have been {raw_vertical_distance})"
        );
        assert!(
            distance < raw_vertical_distance - 0.1,
            "distance {distance} should be meaningfully less than the raw vertical \
             distance {raw_vertical_distance} on a sloped surface"
        );
    }
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p manifold-core --lib order_field::tests::top_surface_aware_order_field_reports_perpendicular -- --nocapture`
Expected: FAIL — the current `march_to_top` returns the raw vertical distance (~2.94), not the perpendicular one (~1.31), so the `< 0.02` tolerance assertion fails.

- [ ] **Step 3: Implement the perpendicular-distance projection**

Find `march_to_top`'s body (search for `fn march_to_top`). It currently ends its exit branch with:

```rust
            let value = self.bed_excluded_sdf.sample(pos).value;
            if value > 0.0 {
                let denom = value - prev_value;
                let t = if denom.abs() > 1e-12 {
                    (-prev_value / denom).clamp(0.0, 1.0)
                } else {
                    1.0
                };
                return Some(traveled - self.step * (1.0 - t));
            }
            prev_value = value;
```

Replace it with:

```rust
            let exit_sample = self.bed_excluded_sdf.sample(pos);
            let value = exit_sample.value;
            if value > 0.0 {
                let denom = value - prev_value;
                let t = if denom.abs() > 1e-12 {
                    (-prev_value / denom).clamp(0.0, 1.0)
                } else {
                    1.0
                };
                let raw_distance = traveled - self.step * (1.0 - t);

                // Project the raw climb-direction (vertical) travel
                // distance onto the local surface normal at the exit
                // point (`exit_sample.gradient`, already the outward
                // normal -- `MeshSdf::sample` always returns a gradient
                // oriented away from the surface, regardless of which
                // side `pos` is on). For a locally planar exit surface
                // with unit normal `n` and climb direction `dir`, the
                // true perpendicular distance from the query point to
                // that surface is `raw_distance * |dir . n|`, not
                // `raw_distance` itself -- the two only coincide when the
                // surface is perpendicular to the climb direction (a flat
                // top). On a taper, `|dir . n| < 1` and this correction is
                // what makes `seed_proximity`'s reported distance for
                // `SeedKind::Patch` reflect true proximity to the top
                // surface instead of overstating it by the taper's own
                // slope factor.
                let normal_len_sq = exit_sample.gradient.length_squared();
                return Some(if normal_len_sq > 1e-12 {
                    let normal = exit_sample.gradient / normal_len_sq.sqrt();
                    raw_distance * dir.dot(normal).abs()
                } else {
                    // Degenerate normal (should not happen for a
                    // non-empty mesh, but `MeshSdf::sample` can return a
                    // zero gradient for an empty triangle set) -- fall
                    // back to the raw distance rather than dividing by a
                    // near-zero length or reporting zero.
                    raw_distance
                });
            }
            prev_value = value;
```

`dir` is already in scope from the loop's own `let dir = grad / len;` a few lines earlier in the same `while` iteration — no new variable needed.

- [ ] **Step 4: Run the test to verify it passes**

Run: `cargo test -p manifold-core --lib order_field::tests::top_surface_aware_order_field_reports_perpendicular -- --nocapture`
Expected: PASS.

- [ ] **Step 5: Run the full `order_field` module test suite**

Run: `cargo test -p manifold-core --lib order_field:: -- --nocapture`
Expected: PASS, including the two pre-existing `top_surface_aware_order_field_*` tests and the distance-pin test on `step_platform_mesh` (a flat top, where `dir . n == 1.0` exactly, so the projection is a no-op there and that test's exact `1.48` expectation is unaffected).

- [ ] **Step 6: Commit**

```bash
git add crates/manifold-core/src/order_field.rs
git commit -m "fix(core): report perpendicular top-surface distance"
```

---

### Task 2: Extract shared solid-fill-eligibility helpers; regression test at default nozzle settings

**Files:**

- Modify: `crates/manifold-core/src/slicing.rs` (re-read the file first — line numbers below are from this plan's drafting and may have shifted)

**Interfaces:**

- Produces: `fn solid_fill_geometry_params(config: &SlicerConfig) -> SolidFillGeometryParams` (new, `pub(crate)`, in `slicing.rs`) — bundles the five values (`bottom_threshold`, `top_threshold`, `min_solid_area`, `max_along`, `cell_size`) that `compute_solid_fill_boundaries` already computes and Task 3/4 also need, so both compute them identically without duplicating the five formulas.
- Produces: `fn seed_eligible_region(order_field: &dyn OrderField, axis: DVec3, apex: DVec3, target_order: f64, params: &SolidFillGeometryParams, basis1: DVec3, basis2: DVec3, extent_2d: (f64, f64, f64, f64)) -> Vec<Vec<DVec3>>` (new, `pub(crate)`, in `slicing.rs`) — the exact `SeedMarginField` + `extract_contours` + center-sample-disambiguation logic already inside `compute_solid_fill_boundaries`, extracted so it's callable with any extent, not just `infill_boundary`'s own. Note the signature takes `order_field`/`target_order` directly rather than a `&Layer` — Task 3/4 call this *before* a `Layer` exists (walls are extracted before `Layer` is constructed), so there is no `Layer` to borrow from at that call site.
- Consumes (Task 3/4): both of the above.

- [ ] **Step 1: Write the failing test for the default-nozzle regression**

Add to `crates/manifold-core/src/slicing.rs`'s test module (this test does not depend on Task 2's refactor to compile or pass — it only depends on Task 1's fix, already landed — but lives here because it belongs next to `compute_solid_fill_boundaries`'s other tests):

```rust
    #[test]
    fn compute_solid_fill_boundaries_covers_a_steeply_tapering_top_at_default_nozzle_settings() {
        // Same cone as the tapering-cone test added by the prior plan
        // (`compute_solid_fill_boundaries_covers_a_steeply_tapering_top_not_just_flat_ones`),
        // but at `SlicerConfig::default()`'s actual nozzle/wall/infill
        // widths (0.4mm) -- the config that test had to tighten to 0.1mm
        // to pass, because the pre-Task-1 vertical-distance march
        // overstated proximity to the top on this cone's slope by a
        // constant factor. With Task 1's perpendicular-distance fix, no
        // config tightening should be needed.
        let base_radius = 5.0;
        let apex_z = 10.0;
        let segments = 32;
        let mut vertices = vec![DVec3::new(0.0, 0.0, 0.0)];
        for i in 0..segments {
            let angle = (i as f64) / (segments as f64) * std::f64::consts::TAU;
            vertices.push(DVec3::new(base_radius * angle.cos(), base_radius * angle.sin(), 0.0));
        }
        vertices.push(DVec3::new(0.0, 0.0, apex_z));
        let apex_idx = vertices.len() as u32 - 1;
        let mut indices = Vec::new();
        for i in 0..segments {
            let a = 1 + i as u32;
            let b = 1 + ((i + 1) % segments) as u32;
            indices.extend_from_slice(&[0, b, a]);
            indices.extend_from_slice(&[a, b, apex_idx]);
        }
        let mesh = Mesh::new(vertices, indices);
        let config = SlicerConfig {
            order_field: order_field::OrderFieldKind::Height,
            layer_height: 0.2,
            top_layers: 3,
            bottom_layers: 3,
            ..SlicerConfig::default()
        };
        let mut layers = slice_mesh(&mesh, &config).unwrap();
        compute_solid_fill_boundaries(&mut layers, &config);
        assert!(!layers.is_empty());

        let top_threshold_layers = config.top_layers;
        let mut checked = 0;
        for layer in layers.iter().rev().take(top_threshold_layers + 2) {
            if layer.loops.is_empty() {
                continue;
            }
            assert!(
                !layer.solid_fill_boundary.is_empty(),
                "layer at index {} (near the cone's apex, default nozzle settings) \
                 should have solid_fill_boundary",
                layer.index
            );
            checked += 1;
        }
        assert!(
            checked >= top_threshold_layers,
            "expected to actually check {top_threshold_layers} near-apex layers, checked {checked}"
        );
    }
```

- [ ] **Step 2: Run the test to verify it already passes (Task 1's fix is sufficient on its own for this)**

Run: `cargo test -p manifold-core --lib slicing::tests::compute_solid_fill_boundaries_covers_a_steeply_tapering_top_at_default_nozzle -- --nocapture`
Expected: PASS. If it fails, do not proceed to Step 3 — Task 1's fix has a gap; stop and re-examine `march_to_top`'s change before continuing (this test's whole purpose is proving Task 1 alone already closes the default-nozzle gap; do not weaken this test's config to make it pass).

- [ ] **Step 3: Extract `solid_fill_geometry_params` and `seed_eligible_region`**

Find `compute_solid_fill_boundaries` (search for `pub fn compute_solid_fill_boundaries`). Its current body starts:

```rust
pub fn compute_solid_fill_boundaries(layers: &mut [Layer], config: &SlicerConfig) {
    let (axis, apex, _slope) = order_field::resolve_axis_apex_slope(config.order_field, config);
    let (basis1, basis2) = plane_basis(axis);
    let origin = apex;

    // [... doc comment about SEED_MARGIN_TOLERANCE_MM ...]
    const SEED_MARGIN_TOLERANCE_MM: f64 = 1e-3;
    let bottom_threshold =
        config.bottom_layers as f64 * config.layer_height + SEED_MARGIN_TOLERANCE_MM;
    let top_threshold = config.top_layers as f64 * config.layer_height + SEED_MARGIN_TOLERANCE_MM;
    let min_solid_area = 0.25 * config.nozzle_diameter * config.nozzle_diameter;
    let max_along = (config.layer_height * 20.0).max(5.0);
    let cell_size = (config.wall_offset / 2.0)
        .min(config.wall_line_width / 4.0)
        .clamp(0.04, 0.10);

    let results: Vec<(usize, Vec<Vec<DVec3>>)> = layers
        .par_iter()
        .enumerate()
        .map(|(pos, layer)| {
            if layer.infill_boundary.is_empty() {
                return (pos, Vec::new());
            }
            let boundary_2d = { /* ... */ };
            let (mut min_u, mut min_v, mut max_u, mut max_v) = ( /* ... */ );
            for loop_ in &boundary_2d { /* ... */ }
            if !min_u.is_finite() || !min_v.is_finite() {
                return (pos, Vec::new());
            }
            let pad = cell_size * 2.0;
            let width = (max_u - min_u) + pad * 2.0;
            let height = (max_v - min_v) + pad * 2.0;
            let center = apex + basis1 * ((min_u + max_u) * 0.5) + basis2 * ((min_v + max_v) * 0.5);
            let resolution_u = ((width / cell_size).ceil() as usize).max(2);
            let resolution_v = ((height / cell_size).ceil() as usize).max(2);

            let field = SeedMarginField {
                order_field: layer.order_field.as_ref(),
                axis,
                target_order: layer.order,
                max_along,
                bottom_threshold,
                top_threshold,
            };
            let seed_eligible_3d = extract_contours(
                &field, center, basis1, basis2, width, height, resolution_u, resolution_v, 0.0,
            );
            let seed_eligible_2d = if seed_eligible_3d.is_empty() {
                if field.sample(center).value >= 0.0 {
                    boundary_2d.clone()
                } else {
                    Vec::new()
                }
            } else {
                let raw = polygon2d::to_2d(&seed_eligible_3d, basis1, basis2, origin);
                polygon2d::canonicalize(&raw)
            };
            let solid_2d = polygon2d::intersection(&seed_eligible_2d, &boundary_2d);
            let solid_2d = polygon2d::filter_min_area(&solid_2d, min_solid_area);

            let mut references = layer.infill_boundary.clone();
            if references.is_empty() { /* ... */ }
            let solid_3d = order_field::reconstruct_on_order_field_near(
                solid_2d, &references, basis1, basis2, axis, apex, layer.order, max_along,
                layer.order_field.as_ref(),
            );
            (pos, solid_3d)
        })
        .collect();

    for (pos, solid_fill_boundary) in results {
        layers[pos].solid_fill_boundary = solid_fill_boundary;
    }
}
```

Add, immediately above `compute_solid_fill_boundaries`:

```rust
/// The five geometry-derived constants [`compute_solid_fill_boundaries`]
/// and [`seed_eligible_region`] both need, computed identically from
/// `config` so callers never duplicate these five formulas or drift out
/// of sync with each other.
pub(crate) struct SolidFillGeometryParams {
    pub bottom_threshold: f64,
    pub top_threshold: f64,
    pub min_solid_area: f64,
    pub max_along: f64,
    pub cell_size: f64,
}

pub(crate) fn solid_fill_geometry_params(config: &SlicerConfig) -> SolidFillGeometryParams {
    // See `compute_solid_fill_boundaries`'s own doc comment for why this
    // tolerance exists (FMM/bisection numerical noise near a
    // bottom_layers/top_layers boundary).
    const SEED_MARGIN_TOLERANCE_MM: f64 = 1e-3;
    SolidFillGeometryParams {
        bottom_threshold: config.bottom_layers as f64 * config.layer_height
            + SEED_MARGIN_TOLERANCE_MM,
        top_threshold: config.top_layers as f64 * config.layer_height + SEED_MARGIN_TOLERANCE_MM,
        min_solid_area: 0.25 * config.nozzle_diameter * config.nozzle_diameter,
        max_along: (config.layer_height * 20.0).max(5.0),
        cell_size: (config.wall_offset / 2.0)
            .min(config.wall_line_width / 4.0)
            .clamp(0.04, 0.10),
    }
}

/// Computes the 2D region (in the `basis1`/`basis2` plane through `apex`)
/// where a point on `order_field`'s own `target_order` isosurface is
/// close enough to a seed (bed contact, or a top-surface patch) to need
/// solid rather than sparse infill -- the same margin-contouring logic
/// [`compute_solid_fill_boundaries`] already used inline, extracted so it
/// can run against any `extent_2d`, independent of whether
/// `Layer::infill_boundary` exists yet at the call site (wall generation,
/// in `slice_mesh_with_progress`, runs before `infill_boundary` is
/// computed for a layer).
pub(crate) fn seed_eligible_region(
    order_field: &dyn OrderField,
    axis: DVec3,
    apex: DVec3,
    target_order: f64,
    params: &SolidFillGeometryParams,
    basis1: DVec3,
    basis2: DVec3,
    extent_2d: (f64, f64, f64, f64), // (min_u, min_v, max_u, max_v)
) -> Vec<Vec<DVec3>> {
    let (min_u, min_v, max_u, max_v) = extent_2d;
    let pad = params.cell_size * 2.0;
    let width = (max_u - min_u) + pad * 2.0;
    let height = (max_v - min_v) + pad * 2.0;
    let center = apex + basis1 * ((min_u + max_u) * 0.5) + basis2 * ((min_v + max_v) * 0.5);
    let resolution_u = ((width / params.cell_size).ceil() as usize).max(2);
    let resolution_v = ((height / params.cell_size).ceil() as usize).max(2);

    let field = SeedMarginField {
        order_field,
        axis,
        target_order,
        max_along: params.max_along,
        bottom_threshold: params.bottom_threshold,
        top_threshold: params.top_threshold,
    };
    let seed_eligible_3d = extract_contours(
        &field, center, basis1, basis2, width, height, resolution_u, resolution_v, 0.0,
    );
    if seed_eligible_3d.is_empty() {
        // No crossing anywhere in the sampled grid: uniformly eligible or
        // uniformly ineligible. Disambiguate the same way
        // `compute_solid_fill_boundaries` always has -- sample the
        // field's own margin at the region's center. A caller-supplied
        // rectangle covering the query extent (see call sites) means "the
        // whole rectangle" is a safe, if approximate, stand-in for "the
        // whole eligible region" in the uniformly-eligible case, since
        // both call sites intersect this result against their own
        // narrower boundary immediately afterward.
        if field.sample(center).value >= 0.0 {
            vec![vec![
                apex + basis1 * min_u + basis2 * min_v,
                apex + basis1 * max_u + basis2 * min_v,
                apex + basis1 * max_u + basis2 * max_v,
                apex + basis1 * min_u + basis2 * max_v,
            ]]
        } else {
            Vec::new()
        }
    } else {
        seed_eligible_3d
    }
}
```

Then replace `compute_solid_fill_boundaries`'s body with:

```rust
pub fn compute_solid_fill_boundaries(layers: &mut [Layer], config: &SlicerConfig) {
    let (axis, apex, _slope) = order_field::resolve_axis_apex_slope(config.order_field, config);
    let (basis1, basis2) = plane_basis(axis);
    let origin = apex;
    let params = solid_fill_geometry_params(config);

    let results: Vec<(usize, Vec<Vec<DVec3>>)> = layers
        .par_iter()
        .enumerate()
        .map(|(pos, layer)| {
            if layer.infill_boundary.is_empty() {
                return (pos, Vec::new());
            }
            let boundary_2d = {
                let raw = polygon2d::to_2d(&layer.infill_boundary, basis1, basis2, origin);
                polygon2d::canonicalize(&raw)
            };

            let (mut min_u, mut min_v, mut max_u, mut max_v) = (
                f64::INFINITY,
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::NEG_INFINITY,
            );
            for loop_ in &boundary_2d {
                for &[u, v] in loop_ {
                    min_u = min_u.min(u);
                    min_v = min_v.min(v);
                    max_u = max_u.max(u);
                    max_v = max_v.max(v);
                }
            }
            if !min_u.is_finite() || !min_v.is_finite() {
                return (pos, Vec::new());
            }

            let seed_eligible_3d = seed_eligible_region(
                layer.order_field.as_ref(),
                axis,
                apex,
                layer.order,
                &params,
                basis1,
                basis2,
                (min_u, min_v, max_u, max_v),
            );
            let seed_eligible_2d = {
                let raw = polygon2d::to_2d(&seed_eligible_3d, basis1, basis2, origin);
                polygon2d::canonicalize(&raw)
            };
            let solid_2d = polygon2d::intersection(&seed_eligible_2d, &boundary_2d);
            let solid_2d = polygon2d::filter_min_area(&solid_2d, params.min_solid_area);

            let mut references = layer.infill_boundary.clone();
            if references.is_empty() {
                references = layer
                    .loops
                    .iter()
                    .filter(|w| w.wall_index == 0)
                    .map(|w| w.points.clone())
                    .collect();
            }
            let solid_3d = order_field::reconstruct_on_order_field_near(
                solid_2d,
                &references,
                basis1,
                basis2,
                axis,
                apex,
                layer.order,
                params.max_along,
                layer.order_field.as_ref(),
            );
            (pos, solid_3d)
        })
        .collect();

    for (pos, solid_fill_boundary) in results {
        layers[pos].solid_fill_boundary = solid_fill_boundary;
    }
}
```

This is a pure refactor: `seed_eligible_region` reproduces the original inline logic exactly, just parameterized on `extent_2d` instead of reading it from `boundary_2d`'s own bbox inline, and its "uniformly eligible" fallback returns a rectangle covering the query extent instead of `boundary_2d.clone()` directly (the caller intersects with `boundary_2d` immediately afterward either way, so both give the same final `solid_2d` — a rectangle covering `boundary_2d`'s own bbox, intersected with `boundary_2d`, equals `boundary_2d`).

- [ ] **Step 4: Run the full `slicing` module test suite to confirm the refactor changed nothing**

Run: `cargo test -p manifold-core --lib slicing:: -- --nocapture`
Expected: PASS, including every pre-existing `compute_solid_fill_boundaries_*` test and the new one from Step 1 — no behavior change from this refactor.

- [ ] **Step 5: Commit**

```bash
git add crates/manifold-core/src/slicing.rs
git commit -m "refactor(core): extract seed-eligibility for wall generation"
```

---

### Task 3: Clip inner wall loops against the solid-fill-eligible region (Height path)

**Files:**

- Modify: `crates/manifold-core/src/slicing.rs` (re-read the surrounding code first — search for `let is_height` and `for wall_index in 0..wall_count`, since line numbers shift)

**Interfaces:**

- Consumes: `SolidFillGeometryParams`, `seed_eligible_region` (Task 2).
- Produces: `fn clip_wall_loop_against_eligible_region(loop_points: &[DVec3], loop_is_open: bool, eligible_2d: &[Vec<[f64; 2]>], basis1: DVec3, basis2: DVec3, origin: DVec3) -> Vec<(Vec<DVec3>, bool)>` (new, private, in `slicing.rs`) — Task 4 reuses this unchanged.

- [ ] **Step 1: Write the failing tests for the new clipping helper**

Add to `crates/manifold-core/src/slicing.rs`'s test module:

```rust
    #[test]
    fn clip_wall_loop_against_eligible_region_keeps_the_whole_loop_when_nothing_overlaps() {
        let square = vec![
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(10.0, 0.0, 0.0),
            DVec3::new(10.0, 10.0, 0.0),
            DVec3::new(0.0, 10.0, 0.0),
        ];
        // Eligible region is far away from the wall loop -- no overlap.
        let eligible = vec![vec![[100.0, 100.0], [110.0, 100.0], [110.0, 110.0], [100.0, 110.0]]];
        let result = clip_wall_loop_against_eligible_region(
            &square, false, &eligible, DVec3::X, DVec3::Y, DVec3::ZERO,
        );
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0.len(), 4);
        assert!(!result[0].1, "untouched loop should stay closed");
    }

    #[test]
    fn clip_wall_loop_against_eligible_region_drops_the_loop_when_fully_inside() {
        let square = vec![
            DVec3::new(1.0, 1.0, 0.0),
            DVec3::new(2.0, 1.0, 0.0),
            DVec3::new(2.0, 2.0, 0.0),
            DVec3::new(1.0, 2.0, 0.0),
        ];
        // Eligible region fully contains the wall loop.
        let eligible = vec![vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]];
        let result = clip_wall_loop_against_eligible_region(
            &square, false, &eligible, DVec3::X, DVec3::Y, DVec3::ZERO,
        );
        assert!(result.is_empty(), "fully-eligible loop should be dropped entirely");
    }

    #[test]
    fn clip_wall_loop_against_eligible_region_returns_an_open_arc_for_a_partial_overlap() {
        // A loop where two adjacent points (index 2, 3 of 6) fall inside
        // the eligible region and the rest don't; the surviving arc must
        // be the 4 remaining points, contiguous in original order and
        // wrapping across the array boundary (`is_open: true`).
        let hexagon = vec![
            DVec3::new(0.0, -1.0, 0.0),   // 0: outside
            DVec3::new(1.0, -0.5, 0.0),   // 1: outside
            DVec3::new(1.5, 0.0, 0.0),    // 2: inside
            DVec3::new(1.0, 0.5, 0.0),    // 3: inside
            DVec3::new(0.0, 1.0, 0.0),    // 4: outside
            DVec3::new(-1.0, 0.0, 0.0),   // 5: outside
        ];
        let eligible = vec![vec![[0.8, -0.3], [2.0, -0.3], [2.0, 0.8], [0.8, 0.8]]];
        let result = clip_wall_loop_against_eligible_region(
            &hexagon, false, &eligible, DVec3::X, DVec3::Y, DVec3::ZERO,
        );
        assert_eq!(result.len(), 1, "expected exactly one surviving arc, got {result:?}");
        let (points, is_open) = &result[0];
        assert!(is_open, "a partially-clipped closed loop must become an open arc");
        // The surviving arc wraps from index 4 through 5, 0, to 1 --
        // contiguous in the original cyclic order, starting right after
        // the inside->outside transition.
        assert_eq!(
            points,
            &vec![hexagon[4], hexagon[5], hexagon[0], hexagon[1]]
        );
    }

    #[test]
    fn clip_wall_loop_against_eligible_region_never_touches_an_already_open_loop_beyond_dropping_points() {
        let open_path = vec![
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(2.0, 0.0, 0.0),
            DVec3::new(3.0, 0.0, 0.0),
        ];
        // Only the last point falls inside the eligible region.
        let eligible = vec![vec![[2.5, -1.0], [10.0, -1.0], [10.0, 1.0], [2.5, 1.0]]];
        let result = clip_wall_loop_against_eligible_region(
            &open_path, true, &eligible, DVec3::X, DVec3::Y, DVec3::ZERO,
        );
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, vec![open_path[0], open_path[1], open_path[2]]);
        assert!(result[0].1, "a clipped open loop stays open");
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p manifold-core --lib slicing::tests::clip_wall_loop -- --nocapture`
Expected: FAIL to compile — `clip_wall_loop_against_eligible_region` doesn't exist yet.

- [ ] **Step 3: Implement `clip_wall_loop_against_eligible_region`**

Add near `SeedMarginField` (or any other module-level helper in `slicing.rs`):

```rust
/// Splits `loop_points` into the sub-runs of points that fall outside
/// every polygon in `eligible_2d`, dropping the runs that fall inside.
/// See this plan's own top-level "Deviation from spec" note for why this
/// samples the loop's own existing points with
/// [`polygon2d::point_in_polygon`] rather than using a filled-region
/// boolean difference: `eligible_2d` is a filled region, but
/// `loop_points` is a stroke centerline, not a filled region itself, so a
/// polygon-polygon difference does not apply. Never synthesizes a new
/// point -- at the wall extraction resolution (`cell_size`, `0.04..0.10`
/// mm between points), sampling only the existing vertices is accurate
/// enough without a dedicated curve-polygon intersection routine.
///
/// A run spanning the entire original point sequence with nothing
/// dropped is returned closed (`is_open` matches `loop_is_open`); every
/// other returned run is open (`is_open: true`), including a surviving
/// arc of an originally-closed loop.
fn clip_wall_loop_against_eligible_region(
    loop_points: &[DVec3],
    loop_is_open: bool,
    eligible_2d: &[Vec<[f64; 2]>],
    basis1: DVec3,
    basis2: DVec3,
    origin: DVec3,
) -> Vec<(Vec<DVec3>, bool)> {
    if loop_points.is_empty() || eligible_2d.is_empty() {
        return vec![(loop_points.to_vec(), loop_is_open)];
    }

    let inside: Vec<bool> = loop_points
        .iter()
        .map(|&p| {
            let uv = [(p - origin).dot(basis1), (p - origin).dot(basis2)];
            eligible_2d.iter().any(|poly| polygon2d::point_in_polygon(uv, poly))
        })
        .collect();

    if inside.iter().all(|&i| !i) {
        return vec![(loop_points.to_vec(), loop_is_open)];
    }
    if inside.iter().all(|&i| i) {
        return Vec::new();
    }

    let n = loop_points.len();
    // For a closed loop, rotate the scan to start right after an
    // inside->outside transition, so a surviving run can never straddle
    // the array's wrap-around boundary and no merge step is needed. An
    // open loop already scans start-to-end with no wrap-around.
    let start = if loop_is_open {
        0
    } else {
        (0..n)
            .find(|&i| !inside[i] && inside[(i + n - 1) % n])
            .unwrap_or(0)
    };

    let mut runs: Vec<Vec<DVec3>> = Vec::new();
    let mut current: Vec<DVec3> = Vec::new();
    for offset in 0..n {
        let idx = (start + offset) % n;
        if inside[idx] {
            if !current.is_empty() {
                runs.push(std::mem::take(&mut current));
            }
        } else {
            current.push(loop_points[idx]);
        }
    }
    if !current.is_empty() {
        runs.push(current);
    }

    runs.into_iter().map(|pts| (pts, true)).collect()
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p manifold-core --lib slicing::tests::clip_wall_loop -- --nocapture`
Expected: PASS, all 4 tests.

- [ ] **Step 5: Wire the clipping into the Height-path wall loop**

Find the Height-path wall-generation loop (search for `if is_height {` inside the per-layer `.map(|(index, &order_value)| {` closure). It currently pushes a `WallLoop` for every `wall_index in 0..wall_count` unconditionally:

```rust
                for wall_index in 0..wall_count {
                    let iso = -(config.wall_offset + wall_index as f64 * config.wall_line_width);
                    let wall_loops = extract_contours(
                        &*side_sdf, origin, basis1, basis2, extent, extent, resolution, resolution,
                        iso,
                    );
                    loops.extend(wall_loops.iter().cloned().enumerate().map(|(li, points)| {
                        let arc_fraction = compute_arc_fractions(&points);
                        let channel_width = polygon2d::channel_widths_3d(
                            &points,
                            &wall_loops,
                            li,
                            2.0 * config.wall_line_width,
                        );
                        WallLoop {
                            is_open: false,
                            wall_index,
                            island: 0,
                            unsupported: vec![false; points.len()],
                            top_surface: Vec::new(),
                            arc_fraction,
                            // ... (line_widths, channel_width, points -- unchanged)
                        }
                    }));
                }
```

Replace the loop body's `wall_index > 0` case so it clips before constructing `WallLoop`s. `field` (the resolved `Arc<dyn OrderField>`), `axis`, and `params` are not yet in scope at this point in the function — add them once, right before this `if is_height { ... }` block (they're needed by both this task and Task 4's non-Height branch, which sits in the `else` immediately after):

```rust
            let (axis, apex, _) = order_field::resolve_axis_apex_slope(config.order_field, config);
            let params = solid_fill_geometry_params(config);
            let mut loops = Vec::new();
            if is_height {
                for wall_index in 0..wall_count {
                    let iso = -(config.wall_offset + wall_index as f64 * config.wall_line_width);
                    let wall_loops = extract_contours(
                        &*side_sdf, origin, basis1, basis2, extent, extent, resolution, resolution,
                        iso,
                    );
                    for (li, points) in wall_loops.iter().cloned().enumerate() {
                        let arc_fraction = compute_arc_fractions(&points);
                        let channel_width = polygon2d::channel_widths_3d(
                            &points,
                            &wall_loops,
                            li,
                            2.0 * config.wall_line_width,
                        );
                        if wall_index == 0 {
                            loops.push(WallLoop {
                                is_open: false,
                                wall_index,
                                island: 0,
                                unsupported: vec![false; points.len()],
                                top_surface: Vec::new(),
                                arc_fraction,
                                line_widths: vec![config.wall_line_width; points.len()],
                                channel_width,
                                points,
                            });
                            continue;
                        }

                        // Inner wall (wall_index >= 1): clip against the
                        // region solid infill will already cover, so this
                        // wall and solid infill never print over the same
                        // physical space (see this plan's Task 3/4).
                        let points_2d: Vec<[f64; 2]> = points
                            .iter()
                            .map(|&p| [(p - origin).dot(basis1), (p - origin).dot(basis2)])
                            .collect();
                        let (mut min_u, mut min_v, mut max_u, mut max_v) = (
                            f64::INFINITY,
                            f64::INFINITY,
                            f64::NEG_INFINITY,
                            f64::NEG_INFINITY,
                        );
                        for &[u, v] in &points_2d {
                            min_u = min_u.min(u);
                            min_v = min_v.min(v);
                            max_u = max_u.max(u);
                            max_v = max_v.max(v);
                        }
                        let eligible_3d = if min_u.is_finite() {
                            seed_eligible_region(
                                field.as_ref(),
                                axis,
                                apex,
                                order_value,
                                &params,
                                basis1,
                                basis2,
                                (min_u, min_v, max_u, max_v),
                            )
                        } else {
                            Vec::new()
                        };
                        let eligible_2d = {
                            let raw = polygon2d::to_2d(&eligible_3d, basis1, basis2, origin);
                            polygon2d::canonicalize(&raw)
                        };
                        let clipped = clip_wall_loop_against_eligible_region(
                            &points, false, &eligible_2d, basis1, basis2, origin,
                        );
                        for (clipped_points, clipped_is_open) in clipped {
                            let n_pts = clipped_points.len();
                            if n_pts < 2 {
                                continue;
                            }
                            let arc_fraction = compute_arc_fractions(&clipped_points);
                            loops.push(WallLoop {
                                is_open: clipped_is_open,
                                wall_index,
                                island: 0,
                                unsupported: vec![false; n_pts],
                                top_surface: Vec::new(),
                                arc_fraction,
                                line_widths: vec![config.wall_line_width; n_pts],
                                channel_width: vec![f64::INFINITY; n_pts],
                                points: clipped_points,
                            });
                        }
                    }
                }
```

`channel_width: vec![f64::INFINITY; n_pts]` for the clipped case (rather than recomputing `polygon2d::channel_widths_3d` on the sub-arc) is a deliberate, documented simplification: `INFINITY` is already this codebase's "no channel-width constraint" convention (used unconditionally in the non-Height path today), so this never under- or over-constrains a clipped inner wall's bead width — it just skips a refinement that clipped loops don't currently get anywhere else in this file either.

The `field` and `order_value` variables referenced above are already in scope at this point in the existing code (`field` is the resolved order field bound before the per-layer parallel loop; `order_value` is the closure's own loop variable) — re-read the surrounding ~50 lines before this edit to confirm their exact names haven't shifted.

- [ ] **Step 6: Run the Height-path wall generation tests**

Run: `cargo test -p manifold-core --lib slicing::tests:: -- --nocapture`
Expected: PASS, no regressions in any existing Height-mode wall/layer test (none of them use a geometry where an inner wall loop overlaps a solid-fill-eligible region, so none should observe any change in loop count/shape from this step).

- [ ] **Step 7: Commit**

```bash
git add crates/manifold-core/src/slicing.rs
git commit -m "fix(core): clip Height-mode inner walls near solid fill"
```

---

### Task 4: Clip inner wall loops against the solid-fill-eligible region (non-Height path)

**Files:**

- Modify: `crates/manifold-core/src/slicing.rs` (re-read the surrounding code first — search for `for w in 1..wall_count` inside the non-Height (`else`) branch of the per-layer wall-generation loop)

**Interfaces:**

- Consumes: `SolidFillGeometryParams`, `seed_eligible_region`, `clip_wall_loop_against_eligible_region` (Tasks 2/3).

- [ ] **Step 1: Write the failing test**

Add to `crates/manifold-core/src/slicing.rs`'s test module — this exercises the non-Height path specifically (`OrderFieldKind::AnisotropicFsm` with `fsm_seed_surfaces_enabled: false`, the same FSM-without-native-seeds configuration the prior plan's final fix wave used, so `TopSurfaceAwareOrderField` wraps it and this task's clipping applies):

```rust
    #[test]
    fn non_height_inner_walls_are_clipped_where_solid_fill_will_cover_the_same_area() {
        let base_radius = 5.0;
        let apex_z = 10.0;
        let segments = 32;
        let mut vertices = vec![DVec3::new(0.0, 0.0, 0.0)];
        for i in 0..segments {
            let angle = (i as f64) / (segments as f64) * std::f64::consts::TAU;
            vertices.push(DVec3::new(base_radius * angle.cos(), base_radius * angle.sin(), 0.0));
        }
        vertices.push(DVec3::new(0.0, 0.0, apex_z));
        let apex_idx = vertices.len() as u32 - 1;
        let mut indices = Vec::new();
        for i in 0..segments {
            let a = 1 + i as u32;
            let b = 1 + ((i + 1) % segments) as u32;
            indices.extend_from_slice(&[0, b, a]);
            indices.extend_from_slice(&[a, b, apex_idx]);
        }
        let mesh = Mesh::new(vertices, indices);
        let config = SlicerConfig {
            order_field: order_field::OrderFieldKind::AnisotropicFsm,
            layer_height: 1.0,
            nozzle_diameter: 1.0,
            fsm_seed_surfaces_enabled: false,
            top_layers: 3,
            bottom_layers: 3,
            shell_thickness: 3.0,
            wall_line_width: 1.0,
            ..SlicerConfig::default()
        };
        let mut layers = slice_mesh(&mesh, &config).unwrap();
        compute_solid_fill_boundaries(&mut layers, &config);

        // For every layer, no inner-wall point (wall_index >= 1) should
        // fall inside that layer's own solid_fill_boundary -- this is
        // the direct invariant this plan's fix exists to guarantee.
        let (axis, apex, _) = order_field::resolve_axis_apex_slope(config.order_field, &config);
        let (basis1, basis2) = plane_basis(axis);
        let mut checked_any_inner_wall = false;
        for layer in &layers {
            if layer.solid_fill_boundary.is_empty() {
                continue;
            }
            let solid_2d = {
                let raw = polygon2d::to_2d(&layer.solid_fill_boundary, basis1, basis2, apex);
                polygon2d::canonicalize(&raw)
            };
            for wall in layer.loops.iter().filter(|w| w.wall_index >= 1 && w.wall_index < 990) {
                checked_any_inner_wall = true;
                for &p in &wall.points {
                    let uv = [(p - apex).dot(basis1), (p - apex).dot(basis2)];
                    assert!(
                        !solid_2d.iter().any(|poly| polygon2d::point_in_polygon(uv, poly)),
                        "layer {} wall_index {} has a point inside its own \
                         solid_fill_boundary -- overlap with solid infill",
                        layer.index,
                        wall.wall_index
                    );
                }
            }
        }
        assert!(
            checked_any_inner_wall,
            "expected at least one non-empty inner wall loop to actually check \
             (test setup produced no wall_index >= 1 loops at all)"
        );
    }
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p manifold-core --lib slicing::tests::non_height_inner_walls_are_clipped -- --nocapture`
Expected: FAIL — before this task's fix, inner walls in the non-Height path are never checked against `solid_fill_boundary` at all, so on this cone's tapering tip some inner-wall point should land inside `solid_fill_boundary`. (If it unexpectedly passes on this exact geometry/config, the test's `checked_any_inner_wall` assertion should still catch a setup that never exercised the code path — do not treat an unexpected pass as success without confirming `checked_any_inner_wall` was actually `true`.)

- [ ] **Step 3: Wire the clipping into the non-Height-path wall loop**

Find the non-Height-path's `for w in 1..wall_count` loop (search for the fallback comment `// If the interior cavity has narrowed or capped under a roof, fall back to the` or `let extracted_w_loops = suppress_close_redundant_loops(...)`). Its tail currently looks like:

```rust
                    let extracted_w_loops =
                        suppress_close_redundant_loops(extracted_w_loops, config.max_bead_width());
                    let extracted_w_loops = drop_fragmented_wall_loops(
                        extracted_w_loops,
                        w - 1,
                        &loops,
                        &outers,
                        origin,
                        basis1,
                        basis2,
                    );

                    for pts in extracted_w_loops {
                        let arc_fraction = compute_arc_fractions(&pts);
                        let n_pts = pts.len();
                        let mid_2d = [(pts[0] - origin).dot(basis1), (pts[0] - origin).dot(basis2)];
                        let island = outers
                            .iter()
                            .position(|out| polygon2d::point_in_polygon(mid_2d, out))
                            .unwrap_or(0);
                        loops.push(WallLoop {
                            is_open: false,
                            wall_index: w,
                            island,
                            unsupported: vec![false; n_pts],
                            top_surface: Vec::new(),
                            arc_fraction,
                            line_widths: vec![config.wall_line_width; n_pts],
                            channel_width: vec![f64::INFINITY; n_pts],
                            points: pts,
                        });
                    }
```

Replace the `for pts in extracted_w_loops { ... }` block with:

```rust
                    for pts in extracted_w_loops {
                        let mid_2d = [(pts[0] - origin).dot(basis1), (pts[0] - origin).dot(basis2)];
                        let island = outers
                            .iter()
                            .position(|out| polygon2d::point_in_polygon(mid_2d, out))
                            .unwrap_or(0);

                        // `w` is always `>= 1` in this loop (the caller
                        // iterates `for w in 1..wall_count`), so every
                        // loop constructed here is an inner wall subject
                        // to clipping (see Task 3's Height-path note for
                        // why this and Task 4's are not the same code
                        // path -- Height uses flat-plane marching squares
                        // per wall depth, this uses order-contour
                        // extraction on a precomputed wall isosurface
                        // mesh, but both produce the same `WallLoop`
                        // shape and both need the same clip).
                        let points_2d: Vec<[f64; 2]> = pts
                            .iter()
                            .map(|&p| [(p - origin).dot(basis1), (p - origin).dot(basis2)])
                            .collect();
                        let (mut min_u, mut min_v, mut max_u, mut max_v) = (
                            f64::INFINITY,
                            f64::INFINITY,
                            f64::NEG_INFINITY,
                            f64::NEG_INFINITY,
                        );
                        for &[u, v] in &points_2d {
                            min_u = min_u.min(u);
                            min_v = min_v.min(v);
                            max_u = max_u.max(u);
                            max_v = max_v.max(v);
                        }
                        let eligible_3d = if min_u.is_finite() {
                            seed_eligible_region(
                                field.as_ref(),
                                axis,
                                apex,
                                order_value,
                                &params,
                                basis1,
                                basis2,
                                (min_u, min_v, max_u, max_v),
                            )
                        } else {
                            Vec::new()
                        };
                        let eligible_2d = {
                            let raw = polygon2d::to_2d(&eligible_3d, basis1, basis2, origin);
                            polygon2d::canonicalize(&raw)
                        };
                        let clipped = clip_wall_loop_against_eligible_region(
                            &pts, false, &eligible_2d, basis1, basis2, origin,
                        );
                        for (clipped_points, clipped_is_open) in clipped {
                            let n_pts = clipped_points.len();
                            if n_pts < 2 {
                                continue;
                            }
                            let arc_fraction = compute_arc_fractions(&clipped_points);
                            loops.push(WallLoop {
                                is_open: clipped_is_open,
                                wall_index: w,
                                island,
                                unsupported: vec![false; n_pts],
                                top_surface: Vec::new(),
                                arc_fraction,
                                line_widths: vec![config.wall_line_width; n_pts],
                                channel_width: vec![f64::INFINITY; n_pts],
                                points: clipped_points,
                            });
                        }
                    }
```

`apex` and `params` come from Task 3's Step 5 addition just before the `if is_height { ... } else { ... }` split — confirm they're in scope from the non-Height (`else`) branch too (both branches are inside the same per-layer closure, after that shared addition, so they should be); if the existing code structures `is_height`/non-Height as two entirely separate closures rather than one `if`/`else`, move the `let (axis, apex, _) = ...` and `let params = ...` lines to before whichever split point actually exists, so both branches see them.

- [ ] **Step 4: Run the test to verify it passes**

Run: `cargo test -p manifold-core --lib slicing::tests::non_height_inner_walls_are_clipped -- --nocapture`
Expected: PASS.

- [ ] **Step 5: Run the full `slicing` module test suite**

Run: `cargo test -p manifold-core --lib slicing:: -- --nocapture`
Expected: PASS, no regressions.

- [ ] **Step 6: Commit**

```bash
git add crates/manifold-core/src/slicing.rs
git commit -m "fix(core): clip non-Height inner walls near solid fill"
```

---

### Task 5: Full workspace verification

**Files:** None (verification only).

- [ ] **Step 1: Run the full pre-commit gate**

Run, in order (the last step takes several minutes — run as a background/detached process and poll if your environment's tool timeout is short):

```bash
cargo fmt --all
cargo clippy --workspace --all-targets
cargo test --workspace
```

Expected: all three clean, zero new warnings, zero failures across every crate (`manifold-core`, `manifold-fidget`, `manifold-cli`, `manifold-gui`, `manifold-printer`).

- [ ] **Step 2: Manual sanity check on the CLI**

If a sample non-trivial STL/3MF mesh is available in the repo or environment (check for one under any `examples/`, `test-data/`, or similar directory; if none exists, skip this step and note that in your report — do not fabricate a mesh path), run:

```bash
cargo run -p manifold-cli --release -- <mesh-path> -o /tmp/verify.gcode
```

Expected: completes without error or panic. This is a smoke check, not a substitute for the unit/integration tests above.

- [ ] **Step 3: Commit (if Step 1 required any fixes)**

If `cargo fmt --all` or fixing a clippy warning produced any diff not already covered by Tasks 1-4's commits, commit it:

```bash
git add -A
git commit -m "chore(core): fmt/clippy fixes for top-distance and wall-overlap fix"
```

If there is nothing to commit, skip this step.

---

## Self-Review

**Spec coverage:** Design section 1 (perpendicular-distance march) → Task 1. Design section 2 (extract `seed_eligible_region`) → Task 2. Design section 3 (subtract eligible region from inner walls) → Tasks 3/4, with the corrected clipping mechanism documented as a deviation at the top of this plan. Testing section's three bullets → the default-nozzle regression test (Task 2), the perpendicular-distance unit test (Task 1), and the no-overlap invariant test (Task 4, verified directly on `Layer::loops`/`solid_fill_boundary` rather than on emitted `toolpath::Path`s, which is a more direct and precise check of the actual guarantee than reconstructing overlap from final G-code paths). The spec's "Known Open Question" (does `WallLoop` support open paths) is resolved, not deferred: `WallLoop.is_open` already exists and is already consumed end-to-end by `toolpath.rs`'s segment generation and `gap_fill.rs` — confirmed directly against both call sites before writing this plan, so no additional sub-task for open-path support was needed.

**Placeholder scan:** No TBDs. Every step has complete, real code. The two "re-read the file first, line numbers may have shifted" notes (Tasks 3 and 4) mirror the same accepted convention the prior merged plan used in its own Task 1, not a placeholder — they point at a `search for` anchor, not an unresolved decision.

**Type consistency:** `SolidFillGeometryParams` is defined once (Task 2) and consumed identically (`&params` or its individual fields) in Task 2's own refactor and in Tasks 3/4. `seed_eligible_region`'s signature (`order_field: &dyn OrderField, axis: DVec3, apex: DVec3, target_order: f64, params: &SolidFillGeometryParams, basis1: DVec3, basis2: DVec3, extent_2d: (f64, f64, f64, f64)) -> Vec<Vec<DVec3>>`) is used identically at all three call sites (Task 2's refactor, Task 3, Task 4). `clip_wall_loop_against_eligible_region`'s signature is defined once (Task 3) and reused unchanged (Task 4).
