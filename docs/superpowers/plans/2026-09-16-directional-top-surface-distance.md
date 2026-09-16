# Directional Top-Surface Distance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix a real, previously-parked Critical regression: `compute_solid_fill_boundaries` currently only recognizes a "top surface" seed for classification purposes via `PatchAwareOrderField`'s angle-thresholded (`seed_max_angle_deg`, default 10°) mesh-face clustering (`SeedPatch`/`UnionFind`/`detect_top_surface_patches`, in `crates/manifold-core/src/order_field.rs`). A tapering or rounded top narrows through steeper angles than that threshold as it approaches its apex, so no patch is ever detected there — that region gets no `solid_fill_boundary` coverage at all, and prints sparse infill directly against the converging inner walls instead of solid skin. This plan replaces the angle-thresholded clustering approach entirely with a direct, per-point ray-march along the field's own local climb direction, which works for any top-surface geometry — flat, tapered, domed, doesn't matter.

**Architecture:** `SeedKind::Bed`'s distance (`order(p)` itself) works for any geometry because the underlying field is *seeded from* the bed: its value at any point is literally "distance from the bed along climb direction," already computed for the whole domain by the field's own solve. There is no equivalent "distance from the top" field, since fields are seeded from the bed only. Rather than approximating one with discrete surface-patch clustering (which only works for near-flat surfaces), compute it directly: for a query point `p`, march along the local climb gradient (`order_field::numeric_gradient`, already used by `StepCalibration`) in small hops, testing each hop against a bed-contact-excluded mesh SDF (so the march can't be fooled by "exiting" near the bed itself) until it exits the solid or exceeds a bounded search distance — the accumulated distance at exit is the real physical distance to this point's own local top surface. A cheap Euclidean-distance pre-filter (the same bed-excluded SDF's own nearest-surface query, always a lower bound on any directional distance to that surface) skips the expensive march entirely for the common case of points deep in the interior, far from any exposed surface.

This directly obsoletes `SeedPatch`/`UnionFind`/`detect_top_surface_patches` (~250 lines added in the prior top-surface-seeds plan) — they were approximating exactly this quantity, badly, only for near-flat surfaces. This plan deletes them rather than keeping both mechanisms side by side.

**Tech Stack:** Rust, `glam::DVec3`, `manifold_fidget::mesh_sdf::MeshSdf::new_with_distance_faces` (already used in `slicing.rs` for an analogous bed-exclusion SDF), `order_field::numeric_gradient` (already used by `StepCalibration`).

**Spec:** No separate spec file — design was worked out and approved in chat (see conversation preceding this plan). This plan is self-contained.

## Global Constraints

- Core geometry uses `glam::DVec3` (f64) — no `f32`/`Vec3`.
- `TopSurfaceAwareOrderField` (replacing `PatchAwareOrderField`), its fields, and its methods stay private to `order_field.rs`.
- No behavior change for `AnisotropicFsm` with `fsm_seed_surfaces_enabled: true` — it keeps its own native `seed_proximity` (unwrapped), exactly as today.
- `order_field_for`/`order_field_for_with_sdf`'s existing public signatures must stay unchanged.
- After all tasks: `cargo fmt --all` → `cargo clippy --workspace --all-targets` → `cargo test --workspace` must all pass.
- **Session-size note for whoever executes this plan**: if dispatching via `subagent-driven-development` from a session that has already done a lot of work (long conversation history), watch for `"Prompt is too long"` or similar context-overflow failures on subagent dispatch — this happened during this plan's own design session. If dispatches keep failing for that reason, execute this plan from a fresh session instead; the plan file and this document are fully self-contained and don't depend on any prior conversation context.

---

### Task 1: Extract the shared bed-exclusion face filter; replace `PatchAwareOrderField` with `TopSurfaceAwareOrderField`

**Files:**

- Modify: `crates/manifold-core/src/mesh.rs` (add the shared face-filter function)
- Modify: `crates/manifold-core/src/slicing.rs` (replace its inline `non_bed_floor_faces` filter closure with a call to the new shared function — cosmetic dedup, not a behavior change; re-read the file first, since prior work in this session shifted line numbers)
- Modify: `crates/manifold-core/src/order_field.rs` (delete `SeedPatch`/`UnionFind`/`detect_top_surface_patches`/`PatchAwareOrderField`; add `TopSurfaceAwareOrderField`; update `order_field_for_with_sdf`'s wiring)
- Modify: `crates/manifold-fidget/src/order.rs` (update `OrderField::seed_proximity`'s doc comment)

**Interfaces:**

- Produces: `pub(crate) fn non_bed_floor_faces(mesh: &Mesh, min_z: f64) -> Vec<[usize; 3]>` in `mesh.rs` — the exact filter logic already inline in `slicing.rs` (downward-facing, near-`min_z` triangles excluded), extracted so both `slicing.rs` and `order_field.rs` can use it without duplicating the logic.
- Produces: `struct TopSurfaceAwareOrderField { inner: Box<dyn OrderField>, bed_excluded_sdf: MeshSdf, max_search: f64, step: f64 }` (private, in `order_field.rs`), implementing `OrderField` — `order()` delegates unchanged to `inner`; `seed_proximity()` returns the smaller of `inner.order(p)` (bed distance) and the ray-march result (top distance).
- Removes: `struct SeedPatch`, `struct UnionFind`, `fn detect_top_surface_patches`, `struct PatchAwareOrderField` and their tests (`detect_top_surface_patches_*`, `patch_aware_order_field_*`) from `order_field.rs`.

- [ ] **Step 1: Write the failing tests**

First, locate `struct Mesh` in `crates/manifold-core/src/mesh.rs` and add, right after its `impl Mesh` block (after `bounding_box`):

```rust
/// Triangle indices of `mesh` excluding downward-facing bed-contact
/// triangles resting on the build plate (any vertex at `z <= min_z + 0.02`)
/// -- the same "keep every top ceiling, roof, and side wall; drop only the
/// literal bed floor" exclusion `slicing::slice_mesh_with_progress` already
/// applies when building its own bed-exclusion SDF for wall/infill
/// boundary extraction, extracted here so both that use and
/// `order_field::order_field_for_with_sdf`'s directional top-surface
/// distance can share the same filter logic instead of duplicating it.
/// Returns the full unfiltered face list when the mesh has no bed-contact
/// faces at all (i.e. `min_z` isn't actually touched by any downward face
/// -- a floating or non-flat-bottomed mesh), matching the "no exclusion
/// needed" case callers already handle by reusing their original SDF.
pub(crate) fn non_bed_floor_faces(mesh: &Mesh, min_z: f64) -> Vec<[usize; 3]> {
    mesh.indices
        .chunks_exact(3)
        .filter_map(|chunk| {
            let [i0, i1, i2] = [chunk[0] as usize, chunk[1] as usize, chunk[2] as usize];
            let v0 = mesh.vertices[i0];
            let v1 = mesh.vertices[i1];
            let v2 = mesh.vertices[i2];
            let normal = (v1 - v0).cross(v2 - v0);
            let normal_len_sq = normal.length_squared();
            if normal_len_sq > 1e-12
                && normal.z < 0.0
                && (v0.z <= min_z + 0.02 || v1.z <= min_z + 0.02 || v2.z <= min_z + 0.02)
            {
                let nz_sq = normal.z * normal.z;
                if nz_sq >= 0.998 * normal_len_sq {
                    return None;
                }
            }
            Some([i0, i1, i2])
        })
        .collect()
}

#[cfg(test)]
mod bed_floor_face_tests {
    use super::*;

    #[test]
    fn non_bed_floor_faces_excludes_only_the_downward_bed_contact_cap() {
        // A unit cube: bottom cap at z=0 (must be excluded), everything
        // else (top cap, 4 side walls) must survive.
        let vertices = vec![
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(1.0, 1.0, 0.0),
            DVec3::new(0.0, 1.0, 0.0),
            DVec3::new(0.0, 0.0, 1.0),
            DVec3::new(1.0, 0.0, 1.0),
            DVec3::new(1.0, 1.0, 1.0),
            DVec3::new(0.0, 1.0, 1.0),
        ];
        let indices = vec![
            0, 2, 1, 0, 3, 2, // -Z (bed contact, downward)
            4, 5, 6, 4, 6, 7, // +Z (top, upward)
            0, 1, 5, 0, 5, 4, // -Y
            3, 7, 6, 3, 6, 2, // +Y
            0, 4, 7, 0, 7, 3, // -X
            1, 2, 6, 1, 6, 5, // +X
        ];
        let mesh = Mesh::new(vertices, indices);
        let faces = non_bed_floor_faces(&mesh, 0.0);
        // 12 total triangles minus the 2 bottom-cap triangles = 10.
        assert_eq!(faces.len(), 10, "expected exactly the 2 bottom-cap triangles excluded");
    }
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p manifold-core --lib mesh::bed_floor_face_tests -- --nocapture`
Expected: FAIL to compile — `non_bed_floor_faces` doesn't exist yet.

- [ ] **Step 3: Add `non_bed_floor_faces` to `mesh.rs`**

Insert the function from Step 1 (without the test module) into `crates/manifold-core/src/mesh.rs`, after `impl Mesh { ... }`'s closing brace. Add the test module too.

Run: `cargo test -p manifold-core --lib mesh:: -- --nocapture`
Expected: PASS.

- [ ] **Step 4: Refactor `slicing.rs` to use the shared function**

In `crates/manifold-core/src/slicing.rs`, find the `non_bed_floor_faces` local variable (search for `let non_bed_floor_faces: Vec<[usize; 3]> = mesh` — it's built via an inline `.filter_map(...)` closure with exactly the same logic Step 1 extracted). Replace that whole `let non_bed_floor_faces: Vec<[usize; 3]> = mesh...collect();` block with:

```rust
    let non_bed_floor_faces: Vec<[usize; 3]> = crate::mesh::non_bed_floor_faces(mesh, min.z);
```

Run: `cargo test -p manifold-core --lib slicing:: -- --nocapture`
Expected: PASS, no behavior change (this is a pure extraction — the logic is byte-for-byte identical, just no longer duplicated inline).

- [ ] **Step 5: Write the failing tests for `TopSurfaceAwareOrderField`**

In `crates/manifold-core/src/order_field.rs`'s test module, add:

```rust
    #[test]
    fn top_surface_aware_order_field_finds_a_tapering_cone_apex_that_a_flat_angle_threshold_would_miss() {
        // A cone: base radius 5 at z=0, apex at z=10 -- the surface slope
        // near the apex is far steeper than any reasonable
        // `seed_max_angle_deg` threshold (a cone this narrow has a
        // half-angle of atan(5/10) ≈ 26.6 degrees from vertical, i.e. ≈
        // 63.4 degrees from horizontal -- nowhere near the old mechanism's
        // 10-degree-from-horizontal cutoff). No mesh face here is "near
        // flat," so the deleted patch-clustering mechanism would find
        // nothing; the ray-march must still find the real top surface.
        let base_radius = 5.0;
        let apex_z = 10.0;
        let segments = 32;
        let mut vertices = vec![DVec3::new(0.0, 0.0, 0.0)]; // 0: base center
        for i in 0..segments {
            let angle = (i as f64) / (segments as f64) * std::f64::consts::TAU;
            vertices.push(DVec3::new(base_radius * angle.cos(), base_radius * angle.sin(), 0.0));
        }
        vertices.push(DVec3::new(0.0, 0.0, apex_z)); // last: apex
        let apex_idx = vertices.len() as u32 - 1;
        let mut indices = Vec::new();
        for i in 0..segments {
            let a = 1 + i as u32;
            let b = 1 + ((i + 1) % segments) as u32;
            // Base cap (downward normal, bed contact).
            indices.extend_from_slice(&[0, b, a]);
            // Side wall up to the apex.
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
        // A point right at the cone's own apex tip (order = height above
        // bed = 10.0 for HeightOrderField) must be classified as a top-surface
        // (Patch) seed with a near-zero distance -- it IS the top.
        let (kind, distance) = field.seed_proximity(DVec3::new(0.0, 0.0, 9.99)).unwrap();
        assert_eq!(
            kind,
            manifold_fidget::order::SeedKind::Patch,
            "expected the cone's steep apex to be classified as a top-surface seed"
        );
        assert!(
            distance < 0.1,
            "expected near-zero distance to the top surface right at the apex, got {distance}"
        );
    }

    #[test]
    fn top_surface_aware_order_field_prefers_bed_when_top_is_genuinely_far() {
        let mesh = step_platform_mesh();
        let config = crate::SlicerConfig {
            order_field: OrderFieldKind::Height,
            layer_height: 0.2,
            ..crate::SlicerConfig::default()
        };
        let field = order_field_for(
            config.order_field,
            &config,
            &mesh,
            &manifold_fidget::slope_profile::SlopeProfile::new(Vec::new()),
        );
        // A point right at the base, far from the platform's own top (z=6):
        // bed must win.
        let (kind, _distance) = field.seed_proximity(DVec3::new(0.5, 0.5, 0.1)).unwrap();
        assert_eq!(kind, manifold_fidget::order::SeedKind::Bed);
    }
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `cargo test -p manifold-core --lib order_field::tests::top_surface_aware -- --nocapture`
Expected: FAIL — the cone-apex test currently gets `SeedKind::Bed` (the existing angle-thresholded patch mechanism finds no patch anywhere near this cone's steep sides, so it falls through to bed-only classification), reproducing the exact regression this plan fixes.

- [ ] **Step 7: Delete the obsoleted patch-clustering mechanism**

In `crates/manifold-core/src/order_field.rs`, delete:

- `struct SeedPatch` and its doc comment.
- `struct UnionFind` and its `impl`.
- `fn detect_top_surface_patches` and its doc comment.
- `struct PatchAwareOrderField` and its `impl OrderField for PatchAwareOrderField`.
- Their tests: `detect_top_surface_patches_finds_the_flat_top_of_a_cube_but_not_its_base`, `detect_top_surface_patches_returns_empty_for_a_mesh_with_no_upward_faces`, `detect_top_surface_patches_separates_two_disjoint_flat_regions_at_different_heights`, `patch_aware_order_field_reports_patch_seed_near_a_raised_top_and_bed_seed_elsewhere`, `order_field_for_height_kind_reports_patch_seed_near_a_flat_raised_top`, `order_field_for_anisotropic_fsm_with_seed_surfaces_disabled_still_gets_generic_patch_wrapping` (the last two will be superseded by Step 5's new tests, which cover the same "wrapping happens for Height and for FSM-without-seeds" ground via the cone/step-platform meshes).

Keep `is_upward_within_angle` — it's unused after this deletion and should also be removed if nothing else in the file calls it (check with a search before deleting; if something else still uses it, leave it).

Keep `step_platform_mesh()` (the test helper) — Step 5's second test reuses it.

- [ ] **Step 8: Implement `TopSurfaceAwareOrderField`**

Insert, where `PatchAwareOrderField` used to be:

```rust
/// Wraps any [`OrderField`], replacing its default `seed_proximity`
/// (bed-only) with a top-surface-aware version that works for any exposed
/// top geometry -- flat, tapered, domed, doesn't matter -- unlike the
/// earlier patch-clustering approach (`SeedPatch`/`UnionFind`/
/// `detect_top_surface_patches`, removed by this change), which only
/// recognized near-flat surfaces within a fixed angle threshold of
/// horizontal and reduced each cluster to one approximate scalar.
///
/// For a query point `p`, marches along the field's own local climb
/// direction ([`order_field::numeric_gradient`]) in `step`-sized hops,
/// testing each hop against `bed_excluded_sdf` (a variant of the mesh's
/// SDF with bed-contact faces excluded from distance evaluation, so the
/// march can't be fooled by "exiting" near the bed itself -- see
/// [`crate::mesh::non_bed_floor_faces`]) until it exits the solid or
/// exceeds `max_search` -- the accumulated distance at exit is the real
/// physical distance to this point's own local top surface, along the
/// direction that actually matters for non-planar layer spacing.
///
/// Bounded by a cheap pre-filter: `bed_excluded_sdf`'s own ordinary
/// Euclidean nearest-surface distance is always `<=` any climb-direction
/// distance to that same surface, so when it already exceeds `max_search`
/// the expensive directional march is skipped entirely -- this keeps the
/// cost of the common case (points deep in the interior, far from any
/// exposed surface) to one cheap BVH query, paying for the full march only
/// within a thin shell near the object's own exterior.
struct TopSurfaceAwareOrderField {
    inner: Box<dyn OrderField>,
    bed_excluded_sdf: MeshSdf,
    max_search: f64,
    step: f64,
}

impl TopSurfaceAwareOrderField {
    fn march_to_top(&self, p: DVec3) -> Option<f64> {
        let mut traveled = 0.0;
        let mut pos = p;
        while traveled < self.max_search {
            let grad = order_field::numeric_gradient(self.inner.as_ref(), pos)?;
            let len = grad.length();
            if !len.is_finite() || len < 1e-9 {
                return None;
            }
            let dir = grad / len;
            pos += dir * self.step;
            traveled += self.step;
            if self.bed_excluded_sdf.sample(pos).value > 0.0 {
                return Some(traveled);
            }
        }
        None
    }
}

impl OrderField for TopSurfaceAwareOrderField {
    fn order(&self, p: DVec3) -> f64 {
        self.inner.order(p)
    }

    fn seed_proximity(&self, p: DVec3) -> Option<(SeedKind, f64)> {
        let bed_distance = self.inner.order(p);

        let top_distance = if self.bed_excluded_sdf.sample(p).value > self.max_search {
            None
        } else {
            self.march_to_top(p)
        };

        Some(match top_distance {
            Some(td) if td < bed_distance => (SeedKind::Patch, td),
            _ => (SeedKind::Bed, bed_distance),
        })
    }
}
```

Note: `MeshSdf` needs importing in `order_field.rs` if it isn't already (check the top of the file — `fsm_field_for`/`eikonal_field_for` already take `Option<&MeshSdf>` parameters, so `use manifold_fidget::mesh_sdf::MeshSdf;` almost certainly already exists; add it if not).

- [ ] **Step 9: Wire `TopSurfaceAwareOrderField` into `order_field_for_with_sdf`**

Find the tail of `order_field_for_with_sdf` (after the `AnisotropicFsm`-with-seeds-enabled early return, where `PatchAwareOrderField` used to be constructed). Replace that block with:

```rust
    let Some((min, _max)) = mesh.bounding_box() else {
        return inner;
    };
    let faces: Vec<[usize; 3]> = mesh
        .indices
        .chunks_exact(3)
        .map(|chunk| [chunk[0] as usize, chunk[1] as usize, chunk[2] as usize])
        .collect();
    if faces.is_empty() {
        return inner;
    }
    let non_bed_faces = crate::mesh::non_bed_floor_faces(mesh, min.z);
    // `MeshSdf` has no `Clone` impl, so always rebuild rather than trying
    // to reuse `sdf` unchanged even when `non_bed_faces` happens to equal
    // the full face list -- the extra BVH build only happens once per
    // object per slice and is not worth threading a `Clone` bound through
    // `MeshSdf` to avoid.
    let bed_excluded_sdf = MeshSdf::new_with_distance_faces(mesh.vertices.clone(), faces, non_bed_faces);
    let max_search = (config.layer_height * 20.0).max(5.0);
    let step = (config.layer_height.min(config.nozzle_diameter) / 4.0).max(0.01);
    Box::new(TopSurfaceAwareOrderField {
        inner,
        bed_excluded_sdf,
        max_search,
        step,
    })
```

`MeshSdf` has no `Clone` impl (confirmed directly against `crates/manifold-fidget/src/mesh_sdf.rs`'s struct definition), so the code above always rebuilds rather than trying to reuse the passed-in `sdf` — no further check needed.

- [ ] **Step 10: Update the doc comment on `OrderField::seed_proximity`**

In `crates/manifold-fidget/src/order.rs`, update the doc comment (currently describing `PatchAwareOrderField`) to describe `TopSurfaceAwareOrderField`'s directional-ray-march mechanism instead — same structure as the existing comment, updated to name the new type and describe marching along climb direction against a bed-excluded SDF rather than clustering near-flat mesh faces.

- [ ] **Step 11: Run tests to verify they pass**

Run: `cargo test -p manifold-core --lib order_field:: -- --nocapture`
Expected: PASS (all `order_field` module tests, including both new `top_surface_aware_*` tests).

Then run the full `manifold-core` lib suite once: `cargo test -p manifold-core --lib`
Expected: PASS, no new failures.

- [ ] **Step 12: Commit**

```bash
git add crates/manifold-core/src/mesh.rs crates/manifold-core/src/slicing.rs crates/manifold-core/src/order_field.rs crates/manifold-fidget/src/order.rs
git commit -m "feat(core): replace patch clustering with a top-surface ray-march"
```

This repo's commit hook requires the subject line ≤72 characters in `<type>(<scope>): <subject>` format.

---

### Task 2: End-to-end regression test on a tapering top; full workspace verification

**Files:**

- Modify: `crates/manifold-core/src/slicing.rs` test module (add one end-to-end test proving `compute_solid_fill_boundaries` now produces solid fill on a genuinely tapering top, the actual user-visible symptom this plan fixes)

**Interfaces:**

- Consumes: `Layer::solid_fill_boundary`, `compute_solid_fill_boundaries`, `slice_mesh` (all existing, unchanged signatures)

- [ ] **Step 1: Write the failing test**

Add to `crates/manifold-core/src/slicing.rs`'s test module:

```rust
    #[test]
    fn compute_solid_fill_boundaries_covers_a_steeply_tapering_top_not_just_flat_ones() {
        // A cone, same shape as order_field.rs's own cone-apex test, tall
        // enough and with a small enough top_layers window that "near the
        // apex" is a real, nonempty band of layers, not just the single
        // topmost layer (which any mechanism would trivially get right).
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

        // The topmost several layers (well within top_layers * layer_height
        // of the apex) must have nonempty solid_fill_boundary -- before this
        // fix, none of them would, since a cone's steep sides never trigger
        // the old angle-thresholded patch detection anywhere near the apex.
        let top_threshold_layers = config.top_layers;
        let mut checked = 0;
        for layer in layers.iter().rev().take(top_threshold_layers + 2) {
            if layer.loops.is_empty() {
                continue; // Skip a possible empty apex-cap layer.
            }
            assert!(
                !layer.solid_fill_boundary.is_empty(),
                "layer at index {} (near the cone's apex) should have solid_fill_boundary",
                layer.index
            );
            checked += 1;
        }
        assert!(checked >= top_threshold_layers, "expected to actually check {top_threshold_layers} near-apex layers, checked {checked}");
    }
```

- [ ] **Step 2: Run the test to verify it fails, then passes**

Run: `cargo test -p manifold-core --lib slicing::tests::compute_solid_fill_boundaries_covers_a_steeply_tapering_top -- --nocapture`

If Task 1 is already committed on this branch, this should PASS immediately (Task 1's fix already makes this true) — in that case, this step's RED/GREEN pair is: temporarily comment out the `TopSurfaceAwareOrderField` wrapping in `order_field_for_with_sdf` (revert to just `return inner;` unconditionally) to confirm the test genuinely fails without Task 1's fix, then restore it and confirm PASS. Record both outputs in your report as this task's RED/GREEN evidence, since Task 1 already landed the actual fix and this task's only job is proving it end-to-end.

- [ ] **Step 3: Run the full pre-commit gate**

Run, in order:

```bash
cargo fmt --all
cargo clippy --workspace --all-targets
cargo test --workspace
```

This last run takes several minutes — run as a background/detached process and poll if your environment's tool timeout is short.

- [ ] **Step 4: Commit**

```bash
git add crates/manifold-core/src/slicing.rs
git commit -m "test(core): prove solid fill covers a steeply tapering top"
```

---

## Self-Review

**Spec coverage:** The design agreed in chat has two parts — (1) the ray-march mechanism itself, replacing the angle-thresholded patch clustering (Task 1), and (2) proof it fixes the actual user-visible symptom end-to-end (Task 2). Both covered.

**Placeholder scan:** No TBDs. `MeshSdf`'s lack of a `Clone` impl was verified directly during this plan's drafting, so Step 9's construction code has no open question left in it.

**Type consistency:** `TopSurfaceAwareOrderField { inner: Box<dyn OrderField>, bed_excluded_sdf: MeshSdf, max_search: f64, step: f64 }`'s fields are used identically in its own `impl` and in the construction at the end of `order_field_for_with_sdf`. `non_bed_floor_faces(mesh: &Mesh, min_z: f64) -> Vec<[usize; 3]>`'s signature is identical across Task 1's Step 1 test, Step 4's `slicing.rs` call site, and Step 9's `order_field.rs` call site.
