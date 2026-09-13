# End-of-Print Pressure Bleed Wipe and Clearance Move Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute an automated reverse pressure-bleed wipe along the final extruded bead, shear the molten meniscus, and perform a collision-free 3D-clamped clearance move outside the object envelope before running user `end_gcode`.

**Architecture:** Extend `SlicerConfig` in `manifold-core` with end-of-print wipe and clearance configuration. In `crates/manifold-core/src/toolpath.rs`, implement `append_end_of_print_wipe_and_clearance` which calculates $L_{\text{wipe}}$ from fluid dynamics pressure advance and junction velocity, appends a reverse wipe segment ($\Delta E = 0$), an outward meniscus shear step, and an arrangement-aware clearance travel move ($D_{\text{clear}} = \frac{1}{2} \times \text{arrangement\_clearance}$) with Z-lift clamped within `machine.build_volume`. In `crates/manifold-core/src/gcode.rs`, ensure the transition from wipe to clearance travel triggers retraction and suppresses duplicate end retraction. In `crates/manifold-gui`, expose configuration in the settings panel.

**Tech Stack:** Rust (edition 2021), `manifold-core`, `glam::DVec3`, `egui`/`eframe`.

**Spec:** `docs/superpowers/specs/2025-05-13-end-of-print-wipe-and-clearance-design.md`

## Global Constraints

- Core geometry uses `glam::DVec3` (f64) everywhere in `manifold-core` — no `f32`/`Vec3` in core.
- All slicing and toolpath domain logic lives exclusively in `crates/manifold-core`.
- `manifold-core` uses `thiserror` for errors; front-ends use `anyhow` at boundaries.
- Build volume limits in $X, Y$, and $Z$ must never be exceeded; if at or near $Z_{\text{max}}$, vertical lift is suppressed.
- Standard workspace checks: `cargo fmt --all` -> `cargo clippy --workspace --all-targets` -> `cargo test --workspace --release`.

---

### File Structure Map

- **`crates/manifold-core/src/lib.rs`**:
  - Add configuration fields to `SlicerConfig`: `end_of_print_wipe_enabled`, `end_of_print_wipe_distance`, `end_of_print_clearance_z_lift`.
  - Add helper methods on `SlicerConfig`: `end_of_print_wipe_enabled(&self) -> bool`, `end_of_print_clearance_z_lift(&self) -> f64`.
  - Wire `toolpath::append_end_of_print_wipe_and_clearance` into `plan_toolpaths_with_progress`.
- **`crates/manifold-core/src/toolpath.rs`**:
  - Implement dynamic pressure-bleed calculation helper: `calculate_end_of_print_wipe_distance(config, fluid_engine, terminal_speed, line_width, layer_height, final_segment_len) -> f64`.
  - Implement clearance waypoint calculator: `calculate_end_of_print_clearance(final_pt, final_dir, mesh_sdf, order_field, machine, config) -> (DVec3, DVec3)`.
  - Implement `append_end_of_print_wipe_and_clearance(paths, objects, machine, config) -> Result<()>`.
- **`crates/manifold-core/src/gcode.rs`**:
  - Verify and test that `emit_with_machine` emits the unextruded reverse wipe, executes retraction before clearance travel, suppresses duplicate retraction at loop end, and executes `end_gcode`.
- **`crates/manifold-gui/src/app.rs`**:
  - Expose end-of-print wipe toggle and Z-lift controls in the slicer settings panel under an appropriate collapsible section.

---

### Task 1: Add Configuration Fields to `SlicerConfig` in `manifold-core`

**Files:**

- Modify: `crates/manifold-core/src/lib.rs`

**Interfaces:**

- Produces:
  - `SlicerConfig.end_of_print_wipe_enabled: bool` (default `true`)
  - `SlicerConfig.end_of_print_wipe_distance: Option<f64>` (default `None`)
  - `SlicerConfig.end_of_print_clearance_z_lift: Option<f64>` (default `None`)
  - `SlicerConfig::end_of_print_wipe_enabled(&self) -> bool`
  - `SlicerConfig::end_of_print_clearance_z_lift(&self) -> f64` (defaults to `2.0`)

- [ ] **Step 1: Write unit tests for new `SlicerConfig` fields and defaults**

In `crates/manifold-core/src/lib.rs` inside `mod tests`:

```rust
#[test]
fn default_end_of_print_wipe_config_is_sane() {
    let config = SlicerConfig::default();
    assert!(config.end_of_print_wipe_enabled());
    assert_eq!(config.end_of_print_wipe_distance, None);
    assert_eq!(config.end_of_print_clearance_z_lift(), 2.0);
}

#[test]
fn end_of_print_wipe_config_deserialization_defaults() {
    let json = r#"{
        "layer_height": 0.2,
        "nozzle_diameter": 0.4,
        "wall_line_width": 0.4,
        "shell_thickness": 0.8
    }"#;
    let config: SlicerConfig = serde_json::from_str(json).unwrap();
    assert!(config.end_of_print_wipe_enabled());
    assert_eq!(config.end_of_print_clearance_z_lift(), 2.0);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p manifold-core -- default_end_of_print_wipe_config_is_sane --release`
Expected: FAIL (fields and methods do not exist)

- [ ] **Step 3: Implement fields and accessor methods on `SlicerConfig`**

In `crates/manifold-core/src/lib.rs`:
Add fields to `SlicerConfig`:

```rust
    /// Whether to perform an automated pressure-bleed wipe and clearance move at the end of the print.
    /// Defaults to true.
    #[serde(default = "default_true")]
    pub end_of_print_wipe_enabled: bool,

    /// Optional manual override for the end wipe distance in mm.
    /// When None, dynamically derived from the fluid pressure advance model.
    #[serde(default)]
    pub end_of_print_wipe_distance: Option<f64>,

    /// Vertical Z lift in mm applied during the final clearance move. Defaults to 2.0 mm when None.
    #[serde(default)]
    pub end_of_print_clearance_z_lift: Option<f64>,
```

Add helper methods in `impl SlicerConfig`:

```rust
    pub fn end_of_print_wipe_enabled(&self) -> bool {
        self.end_of_print_wipe_enabled
    }

    pub fn end_of_print_clearance_z_lift(&self) -> f64 {
        self.end_of_print_clearance_z_lift.unwrap_or(2.0).max(0.0)
    }
```

Ensure `Default for SlicerConfig` initializes:

```rust
    end_of_print_wipe_enabled: true,
    end_of_print_wipe_distance: None,
    end_of_print_clearance_z_lift: None,
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p manifold-core -- default_end_of_print_wipe_config_is_sane --release`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/manifold-core/src/lib.rs
git commit -m "feat(core): add end-of-print wipe and clearance configuration fields"
```

---

### Task 2: Implement Dynamic Pressure-Bleed Wipe Distance Calculation

**Files:**

- Modify: `crates/manifold-core/src/toolpath.rs`

**Interfaces:**

- Produces:
  - `pub fn calculate_end_of_print_wipe_distance(config: &SlicerConfig, fluid_engine: Option<&FluidDynamicsEngine>, terminal_velocity_mm_s: f64, bead_width: f64, bead_height: f64, final_segment_len: f64) -> f64`

- [ ] **Step 1: Write unit tests for dynamic wipe distance calculation**

In `crates/manifold-core/src/toolpath.rs` inside `mod tests`:

```rust
#[test]
fn calculate_end_of_print_wipe_distance_scales_with_velocity_and_pa() {
    let mut config = SlicerConfig::default();
    config.nozzle_diameter = 0.4;
    config.pressure_advance = Some(0.05);

    // At low velocity
    let wipe_low_v = calculate_end_of_print_wipe_distance(&config, None, 10.0, 0.4, 0.2, 10.0);
    // At high velocity (more stored pressure)
    let wipe_high_v = calculate_end_of_print_wipe_distance(&config, None, 60.0, 0.4, 0.2, 10.0);

    assert!(wipe_high_v > wipe_low_v);
    assert!(wipe_low_v >= 0.5 * config.nozzle_diameter);
    assert!(wipe_high_v <= 5.0);
}

#[test]
fn calculate_end_of_print_wipe_distance_clamps_to_segment_length() {
    let mut config = SlicerConfig::default();
    config.nozzle_diameter = 0.4;
    config.pressure_advance = Some(0.10);

    // Segment is only 0.25mm long
    let short_seg_len = 0.25;
    let wipe_dist = calculate_end_of_print_wipe_distance(&config, None, 100.0, 0.4, 0.2, short_seg_len);

    assert!(wipe_dist <= short_seg_len);
    assert!(wipe_dist > 0.0);
}

#[test]
fn calculate_end_of_print_wipe_distance_uses_manual_override() {
    let mut config = SlicerConfig::default();
    config.end_of_print_wipe_distance = Some(3.5);

    let wipe_dist = calculate_end_of_print_wipe_distance(&config, None, 20.0, 0.4, 0.2, 10.0);
    assert_eq!(wipe_dist, 3.5);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p manifold-core -- calculate_end_of_print_wipe_distance --release`
Expected: FAIL (`calculate_end_of_print_wipe_distance` not defined)

- [ ] **Step 3: Implement `calculate_end_of_print_wipe_distance` in `toolpath.rs`**

In `crates/manifold-core/src/toolpath.rs`:

```rust
pub fn calculate_end_of_print_wipe_distance(
    config: &SlicerConfig,
    fluid_engine: Option<&crate::fluid_dynamics::FluidDynamicsEngine>,
    terminal_velocity_mm_s: f64,
    bead_width: f64,
    bead_height: f64,
    final_segment_len: f64,
) -> f64 {
    if let Some(manual) = config.end_of_print_wipe_distance {
        return manual.clamp(0.0, final_segment_len.max(0.0));
    }

    let nozzle_dia = config.nozzle_diameter;
    let l_min = 0.5 * nozzle_dia;
    let l_max = 5.0f64.min(final_segment_len.max(0.0));

    if l_max <= l_min {
        return l_max;
    }

    let q = (bead_width * bead_height * terminal_velocity_mm_s).max(0.0);
    let c_pa = if let Some(engine) = fluid_engine {
        engine.dynamic_pressure_advance(q, 0.0)
    } else {
        config.pressure_advance.unwrap_or(0.0)
    };

    let static_wipe = 1.0 * nozzle_dia;
    let l_calc = (c_pa * terminal_velocity_mm_s) + static_wipe;

    l_calc.clamp(l_min, l_max)
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p manifold-core -- calculate_end_of_print_wipe_distance --release`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/manifold-core/src/toolpath.rs
git commit -m "feat(core): implement dynamic pressure bleed wipe distance calculation"
```

---

### Task 3: Implement Clearance Waypoint Generation with Arrangement Separation and 3D Clamping

**Files:**

- Modify: `crates/manifold-core/src/toolpath.rs`

**Interfaces:**

- Produces:
  - `pub fn calculate_end_of_print_clearance(final_pt: DVec3, final_reverse_dir: DVec3, mesh_sdf: Option<&MeshSdf>, order_field: Option<&dyn OrderField>, machine: &Machine, config: &SlicerConfig) -> (DVec3, DVec3)`
  - Returns `(shear_step_point, final_clearance_point)` where final clearance point is strictly clamped inside `machine.build_volume` in $X, Y$, and $Z$, and offset by $\frac{1}{2}\times\text{arrangement\_clearance}$.

- [ ] **Step 1: Write unit tests for clearance waypoint generation and boundary clamping**

In `crates/manifold-core/src/toolpath.rs` inside `mod tests`:

```rust
#[test]
fn calculate_end_of_print_clearance_clamps_to_max_z() {
    let machine = Machine::new(
        BoundingVolume::Aabb {
            min: DVec3::ZERO,
            max: DVec3::new(200.0, 200.0, 100.0), // Z max is 100.0
        },
        vec![Tool::new(crate::ids::ToolId(0), 0.4)],
    );
    let config = SlicerConfig::default();

    // Final point is already near Z max: Z = 99.5
    let final_pt = DVec3::new(100.0, 100.0, 99.5);
    let reverse_dir = -DVec3::X;

    let (_shear, clear) = calculate_end_of_print_clearance(final_pt, reverse_dir, None, None, &machine, &config);

    // Clearance point must not exceed Z max (100.0 - 0.5 safety margin = 99.5)
    assert!(clear.z <= 99.5);
    assert!(clear.x >= 0.5 && clear.x <= 199.5);
    assert!(clear.y >= 0.5 && clear.y <= 199.5);
}

#[test]
fn calculate_end_of_print_clearance_uses_arrangement_clearance() {
    let mut machine = Machine::new(
        BoundingVolume::Aabb {
            min: DVec3::ZERO,
            max: DVec3::new(200.0, 200.0, 200.0),
        },
        vec![Tool::new(crate::ids::ToolId(0), 0.4)],
    );
    machine.arrangement_clearance = Some(14.0); // Half is 7.0mm
    let config = SlicerConfig::default();

    let final_pt = DVec3::new(100.0, 100.0, 50.0);
    let reverse_dir = DVec3::X; // moving in -X, so reverse is +X

    let (shear, clear) = calculate_end_of_print_clearance(final_pt, reverse_dir, None, None, &machine, &config);

    // Clearance distance from shear point in XY should be 7.0mm
    let xy_dist = DVec2::new(clear.x - shear.x, clear.y - shear.y).length();
    assert!((xy_dist - 7.0).abs() < 1e-3);
    // Z lift of 2.0mm applied
    assert!((clear.z - (shear.z + 2.0)).abs() < 1e-3);
}

#[test]
fn calculate_end_of_print_clearance_clamps_to_max_xy_boundaries() {
    let machine = Machine::new(
        BoundingVolume::Aabb {
            min: DVec3::ZERO,
            max: DVec3::new(100.0, 100.0, 100.0),
        },
        vec![Tool::new(crate::ids::ToolId(0), 0.4)],
    );
    let config = SlicerConfig::default();

    // Final point right at edge of bed: X = 98.0
    let final_pt = DVec3::new(98.0, 50.0, 10.0);
    let reverse_dir = DVec3::X; // pointing towards +X (beyond 100.0)

    let (_shear, clear) = calculate_end_of_print_clearance(final_pt, reverse_dir, None, None, &machine, &config);

    assert!(clear.x <= 99.5);
    assert!(clear.x >= 0.5);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p manifold-core -- calculate_end_of_print_clearance --release`
Expected: FAIL (`calculate_end_of_print_clearance` not defined)

- [ ] **Step 3: Implement `calculate_end_of_print_clearance` in `toolpath.rs`**

In `crates/manifold-core/src/toolpath.rs`:

```rust
pub fn calculate_end_of_print_clearance(
    final_pt: DVec3,
    reverse_dir: DVec3,
    mesh_sdf: Option<&MeshSdf>,
    order_field: Option<&dyn OrderField>,
    machine: &Machine,
    config: &SlicerConfig,
) -> (DVec3, DVec3) {
    let nozzle_dia = config.nozzle_diameter;
    let (b_min, b_max) = match &machine.build_volume {
        BoundingVolume::Aabb { min, max } => (*min, *max),
    };
    let margin = 0.5;
    let safe_min = b_min + DVec3::splat(margin);
    let safe_max = b_max - DVec3::splat(margin);

    // Compute outward shear direction from CAD normal + isosurface normal
    let (cad_normal, iso_normal) = if let Some(sdf) = mesh_sdf {
        let sample = sdf.sample(final_pt);
        let cad_n = sample.gradient.try_normalize();
        let iso_n = order_field
            .and_then(|f| crate::order_field::numeric_gradient(f, final_pt))
            .and_then(|g| g.try_normalize())
            .map(|n| if n.z < 0.0 { -n } else { n })
            .unwrap_or(DVec3::Z);
        (cad_n, iso_n)
    } else {
        (None, DVec3::Z)
    };

    let exit_dir = if let Some(cad_n) = cad_normal {
        let blended = cad_n * 1.5 + iso_normal * 0.5;
        blended.try_normalize().unwrap_or(cad_n)
    } else {
        let rev_xy = DVec3::new(reverse_dir.x, reverse_dir.y, 0.0);
        rev_xy.try_normalize().unwrap_or(DVec3::X)
    };

    // 1. Shear step (1.5 * nozzle_diameter)
    let shear_dist = 1.5 * nozzle_dia;
    let mut shear_pt = final_pt + exit_dir * shear_dist;
    shear_pt = shear_pt.clamp(safe_min, safe_max);

    // 2. Clearance move: half of arrangement clearance
    let clear_dist = (0.5 * machine.arrangement_clearance()).max(1.0);
    let mut clear_xy = DVec2::new(exit_dir.x, exit_dir.y);
    if clear_xy.length_squared() < 1e-4 {
        clear_xy = DVec2::X;
    } else {
        clear_xy = clear_xy.normalize();
    }

    let z_lift = config.end_of_print_clearance_z_lift();
    let target_z = (shear_pt.z + z_lift).min(safe_max.z);

    let mut clear_pt = DVec3::new(
        shear_pt.x + clear_xy.x * clear_dist,
        shear_pt.y + clear_xy.y * clear_dist,
        target_z,
    );
    clear_pt = clear_pt.clamp(safe_min, safe_max);

    (shear_pt, clear_pt)
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p manifold-core -- calculate_end_of_print_clearance --release`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/manifold-core/src/toolpath.rs
git commit -m "feat(core): implement arrangement-aware clearance waypoint calculation with 3D clamping"
```

---

### Task 4: Implement `append_end_of_print_wipe_and_clearance` and Wire into Pipeline

**Files:**

- Modify: `crates/manifold-core/src/toolpath.rs`
- Modify: `crates/manifold-core/src/lib.rs`

**Interfaces:**

- Produces:
  - `pub fn append_end_of_print_wipe_and_clearance(paths: &mut Vec<Path>, objects: &[Object], machine: &Machine, config: &SlicerConfig) -> Result<()>`
  - Appends reverse wipe segment, outward shear step, and clearance travel move to the toolpath sequence.
  - Automatically invoked inside `plan_toolpaths_with_progress` if `config.end_of_print_wipe_enabled()` is true.

- [ ] **Step 1: Write integration tests for `append_end_of_print_wipe_and_clearance`**

In `crates/manifold-core/src/toolpath.rs` inside `mod tests`:

```rust
#[test]
fn append_end_of_print_wipe_appends_unextruded_wipe_and_clearance_segments() {
    let mut config = SlicerConfig::default();
    config.end_of_print_wipe_enabled = true;
    config.nozzle_diameter = 0.4;
    config.wall_line_width = 0.4;
    config.layer_height = 0.2;

    let machine = Machine::new(
        BoundingVolume::Aabb {
            min: DVec3::ZERO,
            max: DVec3::new(200.0, 200.0, 200.0),
        },
        vec![Tool::new(crate::ids::ToolId(0), 0.4)],
    );

    // Create a path with 2 extrusion segments
    let mut path = Path::new(crate::ids::ToolId(0));
    path.points = vec![
        DVec3::new(10.0, 10.0, 1.0),
        DVec3::new(20.0, 10.0, 1.0),
        DVec3::new(20.0, 20.0, 1.0),
    ];
    path.segments = vec![
        Segment {
            kind: MoveKind::Perimeter,
            order: 1.0,
            bead_width: 0.4,
            bead_height: 0.2,
            point_range: 0..2,
        },
        Segment {
            kind: MoveKind::Perimeter,
            order: 1.0,
            bead_width: 0.4,
            bead_height: 0.2,
            point_range: 1..3,
        },
    ];

    let mut paths = vec![path];
    let objects = vec![];

    append_end_of_print_wipe_and_clearance(&mut paths, &objects, &machine, &config).unwrap();

    let last_path = paths.last().unwrap();
    // Must have added wipe and travel segments
    let wipe_seg = last_path.segments.iter().find(|s| s.kind == MoveKind::Wipe);
    assert!(wipe_seg.is_some(), "must have a Wipe segment");

    let last_seg = last_path.segments.last().unwrap();
    assert_eq!(last_seg.kind, MoveKind::Travel);

    // Final point must be elevated and cleared
    let final_pt = last_path.points.last().unwrap();
    assert!(final_pt.z >= 3.0); // 1.0 + 2.0 Z lift
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p manifold-core -- append_end_of_print_wipe_appends --release`
Expected: FAIL (`append_end_of_print_wipe_and_clearance` not defined)

- [ ] **Step 3: Implement `append_end_of_print_wipe_and_clearance` and wire into `plan_toolpaths_with_progress`**

In `crates/manifold-core/src/toolpath.rs`:

```rust
pub fn append_end_of_print_wipe_and_clearance(
    paths: &mut Vec<Path>,
    objects: &[crate::object::Object],
    machine: &crate::machine::Machine,
    config: &SlicerConfig,
) -> Result<()> {
    if !config.end_of_print_wipe_enabled() || paths.is_empty() {
        return Ok(());
    }

    // Find the last path with extruding segments
    let last_path_idx = match paths.iter().rposition(|p| {
        p.segments.iter().any(|s| s.kind != MoveKind::Travel && s.kind != MoveKind::Wipe)
    }) {
        Some(idx) => idx,
        None => return Ok(()),
    };

    let path = &mut paths[last_path_idx];
    if path.points.len() < 2 || path.segments.is_empty() {
        return Ok(());
    }

    let last_extruding_seg = match path.segments.iter().rfind(|s| s.kind != MoveKind::Travel && s.kind != MoveKind::Wipe) {
        Some(s) => s.clone(),
        None => return Ok(()),
    };

    let p_start = path.points[last_extruding_seg.point_range.start];
    let p_end = path.points[last_extruding_seg.point_range.end - 1];
    let seg_vec = p_end - p_start;
    let seg_len = seg_vec.length();
    if seg_len < 1e-4 {
        return Ok(());
    }

    let reverse_dir = -seg_vec / seg_len;
    let terminal_v = config.print_speed().min(config.wall_print_speed());
    let wipe_dist = calculate_end_of_print_wipe_distance(
        config,
        None,
        terminal_v,
        last_extruding_seg.bead_width,
        last_extruding_seg.bead_height,
        seg_len,
    );

    let p_wipe = p_end + reverse_dir * wipe_dist;
    let (p_shear, p_clear) = calculate_end_of_print_clearance(
        p_wipe,
        reverse_dir,
        None,
        None,
        machine,
        config,
    );

    // Append points and segments to the path
    let wipe_start_idx = path.points.len() - 1;
    path.points.push(p_wipe);
    path.segments.push(Segment {
        kind: MoveKind::Wipe,
        order: last_extruding_seg.order,
        bead_width: 0.0,
        bead_height: 0.0,
        point_range: wipe_start_idx..wipe_start_idx + 2,
    });

    let shear_start_idx = path.points.len() - 1;
    path.points.push(p_shear);
    path.segments.push(Segment {
        kind: MoveKind::Travel,
        order: last_extruding_seg.order,
        bead_width: 0.0,
        bead_height: 0.0,
        point_range: shear_start_idx..shear_start_idx + 2,
    });

    let clear_start_idx = path.points.len() - 1;
    path.points.push(p_clear);
    path.segments.push(Segment {
        kind: MoveKind::Travel,
        order: last_extruding_seg.order,
        bead_width: 0.0,
        bead_height: 0.0,
        point_range: clear_start_idx..clear_start_idx + 2,
    });

    Ok(())
}
```

In `crates/manifold-core/src/lib.rs` inside `plan_toolpaths_with_progress`:
Right before `toolpath::validate_within_bounds(&paths, &workspace.machine.build_volume)?;`:

```rust
    toolpath::append_end_of_print_wipe_and_clearance(
        &mut paths,
        &workspace.objects,
        &workspace.machine,
        &workspace.config,
    )?;
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p manifold-core -- append_end_of_print_wipe --release`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/manifold-core/src/toolpath.rs crates/manifold-core/src/lib.rs
git commit -m "feat(core): wire end-of-print wipe and clearance into toolpath planning pipeline"
```

---

### Task 5: Verify G-code Emission & Retraction Handover in `gcode.rs`

**Files:**

- Modify: `crates/manifold-core/src/gcode.rs`

**Interfaces:**

- Consumes: `Path` containing `MoveKind::Wipe` and terminal `MoveKind::Travel`.
- Produces: G-code where the reverse wipe is emitted with $\Delta E = 0$, followed by retraction before travel to $P_{\text{clear}}$, and `end_gcode` emitted with `retracted == true`.

- [ ] **Step 1: Write integration tests for end-of-print wipe G-code emission**

In `crates/manifold-core/src/gcode.rs` inside `mod tests`:

```rust
#[test]
fn emit_end_of_print_wipe_retracts_before_clearance_and_executes_end_gcode() {
    let mut config = SlicerConfig::default();
    config.use_firmware_retraction = true;
    config.end_gcode = "M104 S0\nM140 S0\nPRINT_END".to_string();

    let machine = Machine::new(
        BoundingVolume::Aabb {
            min: DVec3::ZERO,
            max: DVec3::new(200.0, 200.0, 200.0),
        },
        vec![Tool::new(crate::ids::ToolId(0), 0.4)],
    );

    let mut path = Path::new(crate::ids::ToolId(0));
    path.points = vec![
        DVec3::new(10.0, 10.0, 1.0),
        DVec3::new(20.0, 10.0, 1.0),
        DVec3::new(18.0, 10.0, 1.0), // wipe
        DVec3::new(18.0, 11.0, 1.0), // shear
        DVec3::new(18.0, 18.0, 3.0), // clearance
    ];
    path.segments = vec![
        Segment {
            kind: MoveKind::Perimeter,
            order: 1.0,
            bead_width: 0.4,
            bead_height: 0.2,
            point_range: 0..2,
        },
        Segment {
            kind: MoveKind::Wipe,
            order: 1.0,
            bead_width: 0.0,
            bead_height: 0.0,
            point_range: 1..3,
        },
        Segment {
            kind: MoveKind::Travel,
            order: 1.0,
            bead_width: 0.0,
            bead_height: 0.0,
            point_range: 2..4,
        },
        Segment {
            kind: MoveKind::Travel,
            order: 1.0,
            bead_width: 0.0,
            bead_height: 0.0,
            point_range: 3..5,
        },
    ];

    let gcode = emit_with_machine(&[path], &config, Some(&machine), None);

    // Verify wipe move has no E parameter
    assert!(gcode.contains("G1 X18.000 Y10.000 Z1.000"));
    // Retract G10 must appear before the travel moves and before PRINT_END
    let g10_pos = gcode.rfind("G10").expect("G10 retract must be emitted");
    let print_end_pos = gcode.find("PRINT_END").expect("PRINT_END must be present");
    assert!(g10_pos < print_end_pos);

    // No duplicate second G10 between clearance and PRINT_END
    let after_clearance = &gcode[g10_pos + 3..print_end_pos];
    assert!(!after_clearance.contains("G10"), "must not emit redundant second retraction");
}
```

- [ ] **Step 2: Run test to verify behavior**

Run: `cargo test -p manifold-core -- emit_end_of_print_wipe --release`
Expected: PASS (or identify any adjustment in `gcode.rs`)

- [ ] **Step 3: If any adjustment needed in `gcode.rs`, implement it cleanly**

Ensure `gcode::emit_with_machine` treats `MoveKind::Wipe` as unextruded ($\Delta E = 0$) and handles retract transition smoothly.

- [ ] **Step 4: Run test to confirm PASS**

Run: `cargo test -p manifold-core -- emit_end_of_print_wipe --release`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/manifold-core/src/gcode.rs
git commit -m "test(core): verify end-of-print wipe G-code emission and single retraction handover"
```

---

### Task 6: Expose End-of-Print Wipe Controls in `manifold-gui`

**Files:**

- Modify: `crates/manifold-gui/src/app.rs`

**Interfaces:**

- Exposes UI toggles in `crates/manifold-gui/src/app.rs` for `config.end_of_print_wipe_enabled`, `config.end_of_print_clearance_z_lift`.

- [ ] **Step 1: Add UI controls in `manifold-gui` settings panel**

In `crates/manifold-gui/src/app.rs`, find the Retraction / Travel settings section and add:

```rust
ui.checkbox(&mut self.config.end_of_print_wipe_enabled, "End-of-print pressure bleed wipe");
if self.config.end_of_print_wipe_enabled {
    ui.horizontal(|ui| {
        ui.label("Clearance Z-lift:");
        let mut z_lift = self.config.end_of_print_clearance_z_lift();
        if ui.add(egui::DragValue::new(&mut z_lift).speed(0.1).range(0.0..=20.0).suffix(" mm")).changed() {
            self.config.end_of_print_clearance_z_lift = Some(z_lift);
        }
    });
}
```

- [ ] **Step 2: Verify GUI compilation and headless tests**

Run: `cargo test -p manifold-gui --release`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add crates/manifold-gui/src/app.rs
git commit -m "feat(gui): expose end-of-print wipe and clearance Z-lift settings in GUI"
```

---

### Task 7: Full Workspace Lint, Format, and Regression Gate

**Files:**

- All touched files.

- [ ] **Step 1: Run format check**

Run: `cargo fmt --all -- --check`
If formatting needed: `cargo fmt --all`

- [ ] **Step 2: Run clippy across all workspace targets**

Run: `cargo clippy --workspace --all-targets`
Expected: Zero warnings or errors.

- [ ] **Step 3: Run full workspace test suite in release mode**

Run: `cargo test --workspace --release`
Expected: 100% tests pass (all unit tests, integration tests, CLI tests, GUI tests, printer tests).

- [ ] **Step 4: Commit any cleanup**

```bash
git add -u
git commit -m "chore: format and clean workspace lints for end-of-print wipe"
```
