# Universal Top-Surface Seed Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `OrderField::seed_proximity` return real top-surface (`SeedKind::Patch`) results for every order field kind, not only `AnisotropicFsmOrderField` with `fsm_seed_surfaces_enabled` — fixing 4 currently-failing tests in the uncommitted `compute_solid_fill_boundaries` rewrite (`crates/manifold-core/src/slicing.rs`), which relies on `seed_proximity` to classify top-surface solid fill for every order field kind, not just FSM.

**Architecture:** Add a pure-geometry top-surface patch detector (`detect_top_surface_patches`) that clusters a mesh's upward-facing, non-bed-contact triangles into connected components via edge-adjacency union-find, independent of any order field's internal solving mechanism. Wrap the field `order_field_for_with_sdf` already builds in a new `PatchAwareOrderField` decorator that delegates `order()` unchanged and answers `seed_proximity` from the detected patches — for every kind except `AnisotropicFsm` with `fsm_seed_surfaces_enabled` (which already has a more accurate, solver-consistent native answer via `AnisotropicFsmOrderField::seed_proximity`). This separates "where are the top-surface seeds" (always computed, geometric, O(faces), cheap) from "does the solver bend the field around them" (unchanged — still exclusively `fsm_seed_surfaces_enabled`).

**Tech Stack:** Rust, `glam::DVec3`, existing `manifold_core::mesh::Mesh` / `manifold_fidget::order::{OrderField, SeedKind}`.

**Spec:** No separate spec file — design was worked out and approved in chat (see conversation preceding this plan). This plan is self-contained.

## Global Constraints

- Core geometry uses `glam::DVec3` (f64) — no `f32`/`Vec3`.
- `manifold-core` has no UI/CLI dependencies; keep this change entirely inside `crates/manifold-core/src/order_field.rs`.
- No new public API surface beyond what's already exported — `PatchAwareOrderField`, `SeedPatch`, `UnionFind`, and `detect_top_surface_patches` are all private to `order_field.rs`.
- Follow existing conventions in the file: doc comments explaining *why*, not *what*; reuse `is_upward_within_angle` rather than duplicating the angle test.
- After all tasks: `cargo fmt --all` → `cargo clippy --workspace --all-targets` → `cargo test --workspace` must all pass, including the 4 currently-failing tests in `slicing.rs`.

---

### Task 1: `SeedPatch` + `UnionFind` + `detect_top_surface_patches`

**Files:**

- Modify: `crates/manifold-core/src/order_field.rs` (add new private items after `is_upward_within_angle`, which currently ends around line 285 — read the file to find the exact current line before inserting)
- Test: `crates/manifold-core/src/order_field.rs` (inline `#[cfg(test)] mod tests`, same file)

**Interfaces:**

- Produces: `struct SeedPatch { center: DVec3, radius: f64, order_value: f64 }` (private)
- Produces: `fn detect_top_surface_patches(mesh: &crate::mesh::Mesh, min_z: f64, seed_tolerance: f64, seed_max_angle_deg: f64, order_fn: &dyn Fn(DVec3) -> f64) -> Vec<SeedPatch>` (private, used by Task 2)
- Consumes: `is_upward_within_angle(normal: DVec3, max_angle_deg: f64) -> bool` (already in this file)

- [ ] **Step 1: Write the failing tests**

Add to the `#[cfg(test)] mod tests` block in `crates/manifold-core/src/order_field.rs`:

```rust
    #[test]
    fn detect_top_surface_patches_finds_the_flat_top_of_a_cube_but_not_its_base() {
        // A 10x10x10 cube from (0,0,0) to (10,10,10): base at z=0 (bed
        // contact, must be excluded), flat top at z=10 (must be found).
        let mesh = crate::mesh::Mesh::new(
            vec![
                DVec3::new(0.0, 0.0, 0.0),
                DVec3::new(10.0, 0.0, 0.0),
                DVec3::new(10.0, 10.0, 0.0),
                DVec3::new(0.0, 10.0, 0.0),
                DVec3::new(0.0, 0.0, 10.0),
                DVec3::new(10.0, 0.0, 10.0),
                DVec3::new(10.0, 10.0, 10.0),
                DVec3::new(0.0, 10.0, 10.0),
            ],
            vec![
                // Bottom (z=0), normal points down.
                0, 2, 1, 0, 3, 2,
                // Top (z=10), normal points up.
                4, 5, 6, 4, 6, 7,
            ],
        );
        let patches = detect_top_surface_patches(&mesh, 0.0, 0.05, 10.0, &|p| p.z);
        assert_eq!(patches.len(), 1, "expected exactly one top patch, got {patches:?}");
        assert!(
            (patches[0].order_value - 10.0).abs() < 1e-9,
            "expected top patch order_value near 10.0, got {}",
            patches[0].order_value
        );
    }

    #[test]
    fn detect_top_surface_patches_returns_empty_for_a_mesh_with_no_upward_faces() {
        // A single vertical wall triangle: no face qualifies as upward-facing.
        let mesh = crate::mesh::Mesh::new(
            vec![
                DVec3::new(0.0, 0.0, 0.0),
                DVec3::new(1.0, 0.0, 0.0),
                DVec3::new(0.0, 0.0, 1.0),
            ],
            vec![0, 1, 2],
        );
        let patches = detect_top_surface_patches(&mesh, 0.0, 0.05, 10.0, &|p| p.z);
        assert!(patches.is_empty());
    }

    #[test]
    fn detect_top_surface_patches_separates_two_disjoint_flat_regions_at_different_heights() {
        // Two separate flat squares (not edge-connected to each other) at
        // z=5 and z=10, both well clear of the bed at z=0.
        let mesh = crate::mesh::Mesh::new(
            vec![
                DVec3::new(0.0, 0.0, 5.0),
                DVec3::new(1.0, 0.0, 5.0),
                DVec3::new(1.0, 1.0, 5.0),
                DVec3::new(0.0, 1.0, 5.0),
                DVec3::new(100.0, 0.0, 10.0),
                DVec3::new(101.0, 0.0, 10.0),
                DVec3::new(101.0, 1.0, 10.0),
                DVec3::new(100.0, 1.0, 10.0),
            ],
            vec![0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7],
        );
        let mut patches = detect_top_surface_patches(&mesh, 0.0, 0.05, 10.0, &|p| p.z);
        patches.sort_by(|a, b| a.order_value.partial_cmp(&b.order_value).unwrap());
        assert_eq!(patches.len(), 2);
        assert!((patches[0].order_value - 5.0).abs() < 1e-9);
        assert!((patches[1].order_value - 10.0).abs() < 1e-9);
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p manifold-core --lib order_field::tests::detect_top_surface_patches -- --nocapture`
Expected: FAIL with "cannot find function `detect_top_surface_patches`" (and `SeedPatch` not found).

- [ ] **Step 3: Write the implementation**

Insert directly after the `is_upward_within_angle` function body in `crates/manifold-core/src/order_field.rs`:

```rust
/// A connected cluster of upward-facing, non-bed-contact mesh faces (see
/// `is_upward_within_angle`) treated as a top-surface seed by
/// `PatchAwareOrderField::seed_proximity`. Unlike `AnisotropicFsmOrderField`'s
/// PDE-solver patches (`patch_seed_values`), this is a pure post-hoc
/// geometric classification -- it never feeds back into any field solve, so
/// it works for every `OrderField` kind, not just `AnisotropicFsm`.
#[derive(Debug, Clone, PartialEq)]
struct SeedPatch {
    center: DVec3,
    radius: f64,
    order_value: f64,
}

/// Minimal union-find (disjoint set) over `0..n`, path-compressed, used to
/// group mesh faces into edge-connected components.
struct UnionFind {
    parent: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    fn union(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            self.parent[ra] = rb;
        }
    }
}

/// Clusters `mesh`'s upward-facing, non-bed-contact triangles (CAD normal
/// within `seed_max_angle_deg` of horizontal-up per `is_upward_within_angle`,
/// and not touching the bed within `seed_tolerance` of `min_z`) into
/// edge-connected components, each becoming one [`SeedPatch`] with a
/// consensus `order_value`: the *maximum* `order_fn` value across the
/// patch's vertices. `patch_seed_values` uses the same maximum convention
/// to avoid creating an artificial local minimum when a patch is fed back
/// into `AnisotropicFsmOrderField`'s solve as a boundary condition; that
/// specific concern doesn't apply here since this function's result is only
/// ever used as a read-only classification threshold in
/// `PatchAwareOrderField::seed_proximity`, never fed back into a solve --
/// the maximum is reused anyway to keep top-surface classification
/// consistent with that convention.
fn detect_top_surface_patches(
    mesh: &crate::mesh::Mesh,
    min_z: f64,
    seed_tolerance: f64,
    seed_max_angle_deg: f64,
    order_fn: &dyn Fn(DVec3) -> f64,
) -> Vec<SeedPatch> {
    let face_count = mesh.indices.len() / 3;
    let face_vertices = |face: usize| -> [DVec3; 3] {
        let base = face * 3;
        [
            mesh.vertices[mesh.indices[base] as usize],
            mesh.vertices[mesh.indices[base + 1] as usize],
            mesh.vertices[mesh.indices[base + 2] as usize],
        ]
    };
    let qualifies = |face: usize| -> bool {
        let [v0, v1, v2] = face_vertices(face);
        if v0.z <= min_z + seed_tolerance
            && v1.z <= min_z + seed_tolerance
            && v2.z <= min_z + seed_tolerance
        {
            return false;
        }
        let normal = (v1 - v0).cross(v2 - v0);
        is_upward_within_angle(normal, seed_max_angle_deg)
    };

    let mut uf = UnionFind::new(face_count);
    let mut edges: std::collections::HashMap<(u32, u32), Vec<usize>> =
        std::collections::HashMap::new();
    for face in 0..face_count {
        if !qualifies(face) {
            continue;
        }
        let base = face * 3;
        let idx = [
            mesh.indices[base],
            mesh.indices[base + 1],
            mesh.indices[base + 2],
        ];
        for &(a, b) in &[(idx[0], idx[1]), (idx[1], idx[2]), (idx[2], idx[0])] {
            let key = (a.min(b), a.max(b));
            if let Some(others) = edges.get(&key) {
                for &other in others {
                    uf.union(face, other);
                }
            }
            edges.entry(key).or_default().push(face);
        }
    }

    let mut groups: std::collections::HashMap<usize, Vec<DVec3>> = std::collections::HashMap::new();
    for face in 0..face_count {
        if !qualifies(face) {
            continue;
        }
        let root = uf.find(face);
        groups.entry(root).or_default().extend(face_vertices(face));
    }

    groups
        .into_values()
        .map(|vertices| {
            let center = vertices.iter().copied().sum::<DVec3>() / vertices.len() as f64;
            let radius = vertices
                .iter()
                .map(|&v| (v - center).length())
                .fold(0.0_f64, f64::max);
            let order_value = vertices
                .iter()
                .map(|&v| order_fn(v))
                .fold(f64::NEG_INFINITY, f64::max);
            SeedPatch {
                center,
                radius,
                order_value,
            }
        })
        .collect()
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p manifold-core --lib order_field::tests::detect_top_surface_patches -- --nocapture`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add crates/manifold-core/src/order_field.rs
git commit -m "feat(core): add order-field-agnostic top-surface patch detector"
```

---

### Task 2: `PatchAwareOrderField` decorator

**Files:**

- Modify: `crates/manifold-core/src/order_field.rs`

**Interfaces:**

- Consumes: `SeedPatch`, `detect_top_surface_patches` (Task 1); `manifold_fidget::order::{OrderField, SeedKind}` (already imported in this file)
- Produces: `struct PatchAwareOrderField { inner: Box<dyn OrderField>, patches: Vec<SeedPatch>, footprint_tolerance: f64 }` (private, used by Task 3)

- [ ] **Step 1: Write the failing test**

Add to `#[cfg(test)] mod tests`:

```rust
    #[test]
    fn patch_aware_order_field_reports_patch_seed_near_a_raised_top_and_bed_seed_elsewhere() {
        struct Fixed(f64);
        impl OrderField for Fixed {
            fn order(&self, p: DVec3) -> f64 {
                p.z - self.0
            }
        }
        let field = PatchAwareOrderField {
            inner: Box::new(Fixed(0.0)),
            patches: vec![SeedPatch {
                center: DVec3::new(0.0, 0.0, 10.0),
                radius: 1.0,
                order_value: 10.0,
            }],
            footprint_tolerance: 0.5,
        };

        // Right at the patch's own center/height: order() == order_value,
        // patch distance is 0, must win over the (much larger) bed distance.
        let (kind, distance) = field.seed_proximity(DVec3::new(0.0, 0.0, 10.0)).unwrap();
        assert_eq!(kind, SeedKind::Patch);
        assert!(distance.abs() < 1e-9);

        // Far outside the patch's footprint radius + tolerance: falls back
        // to bed distance even though the height matches the patch.
        let (kind, distance) = field.seed_proximity(DVec3::new(50.0, 50.0, 10.0)).unwrap();
        assert_eq!(kind, SeedKind::Bed);
        assert!((distance - 10.0).abs() < 1e-9);

        // Near the bed: bed distance (order() itself) wins.
        let (kind, distance) = field.seed_proximity(DVec3::new(0.0, 0.0, 0.1)).unwrap();
        assert_eq!(kind, SeedKind::Bed);
        assert!((distance - 0.1).abs() < 1e-9);
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p manifold-core --lib order_field::tests::patch_aware_order_field_reports_patch_seed_near_a_raised_top_and_bed_seed_elsewhere -- --nocapture`
Expected: FAIL with "cannot find struct `PatchAwareOrderField`" / "cannot find value `SeedKind`" (add `use manifold_fidget::order::SeedKind;` to the test module if not already imported at file scope — check the top of `order_field.rs` first; it very likely already imports `SeedKind` since `SeedMarginField` in `slicing.rs` uses it and this file resolves `OrderFieldKind`).

- [ ] **Step 3: Write the implementation**

Insert directly after `detect_top_surface_patches`:

```rust
/// Wraps any [`OrderField`] to add real top-surface [`OrderField::seed_proximity`]
/// support via [`detect_top_surface_patches`], without touching `order()` or
/// the wrapped field's own geometry. This separates "where are the
/// top-surface seeds" (always computed here, geometric, cheap) from "does
/// the solver bend the field around them" (still exclusively
/// `AnisotropicFsmOrderField::with_seed_metadata`, gated on
/// `SlicerConfig::fsm_seed_surfaces_enabled` in `fsm_field_for`).
struct PatchAwareOrderField {
    inner: Box<dyn OrderField>,
    patches: Vec<SeedPatch>,
    footprint_tolerance: f64,
}

impl OrderField for PatchAwareOrderField {
    fn order(&self, p: DVec3) -> f64 {
        self.inner.order(p)
    }

    fn seed_proximity(&self, p: DVec3) -> Option<(SeedKind, f64)> {
        let bed_distance = self.inner.order(p);
        let patch_distance = self
            .patches
            .iter()
            .filter(|patch| (p - patch.center).length() <= patch.radius + self.footprint_tolerance)
            .map(|patch| (self.inner.order(p) - patch.order_value).abs())
            .fold(None, |acc: Option<f64>, d| Some(acc.map_or(d, |best| best.min(d))));
        Some(match patch_distance {
            Some(pd) if pd < bed_distance => (SeedKind::Patch, pd),
            _ => (SeedKind::Bed, bed_distance),
        })
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p manifold-core --lib order_field::tests::patch_aware_order_field_reports_patch_seed_near_a_raised_top_and_bed_seed_elsewhere -- --nocapture`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/manifold-core/src/order_field.rs
git commit -m "feat(core): add PatchAwareOrderField seed_proximity decorator"
```

---

### Task 3: Wire the decorator into `order_field_for_with_sdf`

**Files:**

- Modify: `crates/manifold-core/src/order_field.rs:88-107` (the `order_field_for_with_sdf` function body shown in exploration — re-read the file first, since Task 1/2 edits shift line numbers)

**Interfaces:**

- Consumes: `PatchAwareOrderField`, `detect_top_surface_patches` (Task 2); `OrderFieldKind` (existing, `PartialEq, Eq` already derived); `SlicerConfig::fsm_seed_surfaces_enabled`, `SlicerConfig::fsm_seed_max_angle_deg()`, `SlicerConfig::layer_height` (all existing fields/methods on `crate::SlicerConfig`); `mesh.bounding_box() -> Option<(DVec3, DVec3)>` (existing on `crate::mesh::Mesh`)
- Produces: `order_field_for_with_sdf` and `order_field_for` keep their existing public signatures unchanged — only the returned `Box<dyn OrderField>`'s behavior changes (adds real `seed_proximity`) for kinds other than `AnisotropicFsm` with seed surfaces enabled

- [ ] **Step 1: Write the failing tests**

Add to `#[cfg(test)] mod tests` in `crates/manifold-core/src/order_field.rs`:

```rust
    #[test]
    fn order_field_for_height_kind_reports_patch_seed_near_a_flat_raised_top() {
        // A step mesh: base block 10x10x4 (z 0..4), with a smaller raised
        // platform 4x4x2 on top of it (z 4..6) -- the platform's own flat
        // top (z=6) is a patch distinct from the bed (z=0).
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
        // A point right on the raised platform's own top surface.
        let (kind, _distance) = field.seed_proximity(DVec3::new(5.0, 5.0, 6.0)).unwrap();
        assert_eq!(
            kind,
            manifold_fidget::order::SeedKind::Patch,
            "expected the raised platform's own top to be classified as a Patch seed"
        );
    }

    #[test]
    fn order_field_for_anisotropic_fsm_with_seed_surfaces_disabled_still_gets_generic_patch_wrapping() {
        let mesh = step_platform_mesh();
        let config = crate::SlicerConfig {
            order_field: OrderFieldKind::AnisotropicFsm,
            layer_height: 0.2,
            fsm_seed_surfaces_enabled: false,
            ..crate::SlicerConfig::default()
        };
        let field = order_field_for(
            config.order_field,
            &config,
            &mesh,
            &manifold_fidget::slope_profile::SlopeProfile::new(Vec::new()),
        );
        let (kind, _distance) = field.seed_proximity(DVec3::new(5.0, 5.0, 6.0)).unwrap();
        assert_eq!(kind, manifold_fidget::order::SeedKind::Patch);
    }

    /// A 10x10x4 base block (z 0..4) with a 4x4x2 platform (z 4..6) centered
    /// on top, used to test patch detection through the full
    /// `order_field_for` pipeline. Base at (3,3) to (7,7) in XY.
    fn step_platform_mesh() -> crate::mesh::Mesh {
        let base = crate::mesh::Mesh::new(
            vec![
                DVec3::new(0.0, 0.0, 0.0),
                DVec3::new(10.0, 0.0, 0.0),
                DVec3::new(10.0, 10.0, 0.0),
                DVec3::new(0.0, 10.0, 0.0),
                DVec3::new(0.0, 0.0, 4.0),
                DVec3::new(10.0, 0.0, 4.0),
                DVec3::new(10.0, 10.0, 4.0),
                DVec3::new(0.0, 10.0, 4.0),
            ],
            vec![0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7],
        );
        let platform = crate::mesh::Mesh::new(
            vec![
                DVec3::new(3.0, 3.0, 4.0),
                DVec3::new(7.0, 3.0, 4.0),
                DVec3::new(7.0, 7.0, 4.0),
                DVec3::new(3.0, 7.0, 4.0),
                DVec3::new(3.0, 3.0, 6.0),
                DVec3::new(7.0, 3.0, 6.0),
                DVec3::new(7.0, 7.0, 6.0),
                DVec3::new(3.0, 7.0, 6.0),
            ],
            vec![0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7],
        );
        let offset = base.vertices.len() as u32;
        let mut vertices = base.vertices;
        vertices.extend(platform.vertices);
        let mut indices = base.indices;
        indices.extend(platform.indices.iter().map(|i| i + offset));
        crate::mesh::Mesh::new(vertices, indices)
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p manifold-core --lib order_field::tests::order_field_for_height_kind_reports_patch_seed_near_a_flat_raised_top order_field::tests::order_field_for_anisotropic_fsm_with_seed_surfaces_disabled_still_gets_generic_patch_wrapping -- --nocapture`
Expected: FAIL — both assert `SeedKind::Patch` but currently get `SeedKind::Bed` (the whole point of this plan).

- [ ] **Step 3: Write the implementation**

Modify `order_field_for_with_sdf` in `crates/manifold-core/src/order_field.rs`:

```rust
pub fn order_field_for_with_sdf(
    kind: OrderFieldKind,
    config: &SlicerConfig,
    mesh: &Mesh,
    slope_profile: &manifold_fidget::slope_profile::SlopeProfile,
    sdf: Option<&MeshSdf>,
) -> Box<dyn OrderField> {
    let inner: Box<dyn OrderField> = match kind {
        OrderFieldKind::Height => Box::new(HeightOrderField::new(BUILD_DIRECTION)),
        OrderFieldKind::Conical => Box::new(ConicalOrderField::new(
            config.order_field_apex,
            config.order_field_axis,
            config.order_field_slope,
        )),
        OrderFieldKind::Eikonal | OrderFieldKind::DualIso => {
            Box::new(eikonal_field_for(config, mesh, slope_profile, sdf))
        }
        OrderFieldKind::AnisotropicFsm => Box::new(fsm_field_for(config, mesh, slope_profile, sdf)),
    };

    if kind == OrderFieldKind::AnisotropicFsm && config.fsm_seed_surfaces_enabled {
        // AnisotropicFsmOrderField::seed_proximity already reflects the
        // solver's own patch metadata (with_seed_metadata) -- wrapping it
        // here would shadow that more accurate, solve-consistent answer
        // with a purely geometric approximation.
        return inner;
    }

    let Some((min, _max)) = mesh.bounding_box() else {
        return inner;
    };
    let seed_tolerance = config.layer_height.abs().max(f64::EPSILON) / 2.0;
    let footprint_tolerance = (config.layer_height * 20.0).max(5.0);
    let patches = detect_top_surface_patches(
        mesh,
        min.z,
        seed_tolerance,
        config.fsm_seed_max_angle_deg(),
        &|p| inner.order(p),
    );
    if patches.is_empty() {
        return inner;
    }
    Box::new(PatchAwareOrderField {
        inner,
        patches,
        footprint_tolerance,
    })
}
```

Also update `OrderField::seed_proximity`'s doc comment in `crates/manifold-fidget/src/order.rs` (currently says "`Height`, `Conical`, `Eikonal`, `DualIso` -- none of these support patch seeds today"), since that's now false:

```rust
    /// Order-space distance from `p` to whichever seed (the bed contact,
    /// or -- for a field that supports it -- a top-surface patch) is
    /// locally responsible for this region: approximately physical
    /// distance along the climb direction for a well-behaved field.
    ///
    /// Default: `order(p)` itself, i.e. bed-distance only -- exact for a
    /// field with no patch seeds. `manifold_core::order_field::order_field_for`
    /// wraps every concrete field (`HeightOrderField`, `ConicalOrderField`,
    /// `EikonalOrderField`, and `AnisotropicFsmOrderField` when
    /// `fsm_seed_surfaces_enabled` is `false`) in a `PatchAwareOrderField`
    /// decorator that overrides this method using purely geometric
    /// top-surface detection, independent of the field's own solving
    /// mechanism. `AnisotropicFsmOrderField` overrides this method natively
    /// instead (see its own doc) when `fsm_seed_surfaces_enabled` is `true`,
    /// since its patch metadata is already solver-consistent.
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p manifold-core --lib order_field:: -- --nocapture`
Expected: PASS (all `order_field` module tests, including the two new ones and the three from Tasks 1-2).

- [ ] **Step 5: Commit**

```bash
git add crates/manifold-core/src/order_field.rs crates/manifold-fidget/src/order.rs
git commit -m "feat(core): wire PatchAwareOrderField into order_field_for for every kind"
```

---

### Task 4: Verify the originally-failing `slicing.rs` tests now pass

**Files:**

- None modified — this task only runs and confirms, per `principle-prove-it-works`. The already-uncommitted `crates/manifold-core/src/slicing.rs` (the `SeedMarginField`/`compute_solid_fill_boundaries` rewrite from the prior session) is not touched by this task; it should now work correctly given Tasks 1-3.

**Interfaces:**

- Consumes: everything from Tasks 1-3 (no new interfaces produced)

- [ ] **Step 1: Run the 4 previously-failing tests by name**

Run:

```bash
cargo test -p manifold-core --lib \
  slicing::tests::slice_mesh_height_mode_generates_solid_fill_boundary_for_stepped_horizontal_surfaces \
  slicing::tests::slice_mesh_dual_iso_generates_solid_skin_and_no_sparse_infill_at_bottom_and_top_layers \
  slicing::tests::compute_solid_fill_boundaries_covers_only_top_and_bottom_layers_leaving_the_interior_empty \
  slicing::tests::compute_solid_fill_boundaries_propagates_top_and_bottom_in_the_correct_direction \
  -- --nocapture
```

Expected: PASS (all 4). If any still fail, stop and re-diagnose before proceeding — do not adjust these tests to match wrong behavior (see `systematic-debugging`/`principle-fix-root-causes`).

- [ ] **Step 2: Run the full workspace test suite**

Run: `cargo test --workspace`
Expected: PASS, 0 failures. This run takes several minutes; if invoking from an agent harness with a tool timeout under ~5 minutes, run it as a background/detached command and poll for completion rather than let it time out.

- [ ] **Step 3: Run the required pre-commit gate**

Run, in order:

```bash
cargo fmt --all
cargo clippy --workspace --all-targets
cargo test --workspace
```

Expected: `fmt` makes no unexpected changes beyond this plan's own new code style; `clippy` reports no new warnings; `test` passes fully (repeat of Step 2, now after `fmt`/`clippy` in case either touched behavior — it shouldn't, but this is the project's documented required order per `AGENTS.md`).

- [ ] **Step 4: Commit if `cargo fmt` changed anything**

```bash
git add -A
git commit -m "style: cargo fmt"
```

(Skip this step if `cargo fmt --all` produced no diff.)

---

## Self-Review

**Spec coverage:** The design agreed in chat had three parts — (1) a generic geometric patch detector, (2) a decorator that plugs it into `seed_proximity` without touching `order()`, (3) wiring it into the single `order_field_for` construction point so every downstream consumer (`slicing::compute_solid_fill_boundaries`, `infill`, `toolpath`) benefits automatically. Task 1 covers (1), Task 2 covers (2), Task 3 covers (3) plus the doc-comment update this design implies. Task 4 proves the originally-reported regression is actually fixed, per `principle-prove-it-works`.

**Placeholder scan:** No TBDs; every step has literal code. Line-number references in Task 3's Files section explicitly say to re-read the file first since earlier tasks shift line numbers.

**Type consistency:** `SeedPatch { center: DVec3, radius: f64, order_value: f64 }` (Task 1) is used identically in Task 2's `PatchAwareOrderField.patches: Vec<SeedPatch>` and its `seed_proximity` body, and in Task 3's `detect_top_surface_patches` call feeding `PatchAwareOrderField`'s `patches` field. `detect_top_surface_patches(mesh, min_z, seed_tolerance, seed_max_angle_deg, order_fn)` signature is identical across Tasks 1 and 3. `order_field_for`/`order_field_for_with_sdf`'s existing public signatures are unchanged (verified against current source in `crates/manifold-core/src/order_field.rs:77-107`).
