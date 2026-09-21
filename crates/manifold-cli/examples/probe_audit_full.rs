//! Scratch probe: full-dump version of `probe_volume_audit_shell` -- prints
//! ALL underfilled / overfilled / outside-mesh audit cells (no truncation)
//! plus per-layer loop and toolpath-kind structure for the first layers.
//!
//! ```sh
//! CARGO_TARGET_DIR=target/release cargo run --release -p manifold-cli --example probe_audit_full -- \
//!     /path/to/mesh.stl /path/to/profile.json [cell_size]
//! ```

use manifold_core::ids::{ObjectId, ToolId};
use manifold_core::machine::Machine;
use manifold_core::object::{center_on_bed, Object};
use manifold_core::toolpath::MoveKind;
use manifold_core::volume_audit::{audit_extrusion_volume, VolumeKindBucket};
use manifold_core::{slicing, stl, toolpath};

#[derive(serde::Deserialize)]
struct Profile {
    machine: Machine,
    config: manifold_core::SlicerConfig,
}

fn main() -> anyhow::Result<()> {
    let mesh_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "TestObj1.stl".to_string());
    let profile_path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "examples/profile.json".to_string());
    let cell_size: f64 = std::env::args()
        .nth(3)
        .map(|s| s.parse().expect("cell_size must be a number"))
        .unwrap_or(2.0);

    let mesh = stl::load_stl(std::fs::File::open(&mesh_path)?)?;
    let json = std::fs::read_to_string(&profile_path)?;
    let profile: Profile = serde_json::from_str(&json)?;

    let object = Object::new(ObjectId(0), mesh, ToolId(0));
    let mut objects = vec![object];
    center_on_bed(&mut objects, &profile.machine.build_volume);
    let object = &objects[0];

    let (mut mmin, mut mmax) = (
        glam::DVec3::splat(f64::INFINITY),
        glam::DVec3::splat(f64::NEG_INFINITY),
    );
    for &v in &object.mesh.vertices {
        let w = object.transform.transform_point(v);
        mmin = mmin.min(w);
        mmax = mmax.max(w);
    }
    println!(
        "World mesh bbox: x[{:.1},{:.1}] y[{:.1},{:.1}] z[{:.1},{:.1}]",
        mmin.x, mmax.x, mmin.y, mmax.y, mmin.z, mmax.z
    );

    println!(
        "Slicing {mesh_path} with {profile_path} (order_field={:?}, cell_size={cell_size})...",
        profile.config.order_field
    );
    let layers = slicing::slice_object(object, &profile.config)?;
    println!("Sliced into {} layers", layers.len());
    let paths = toolpath::plan(
        &layers,
        std::slice::from_ref(object),
        &profile.machine.tools,
        &profile.config,
    )?;
    println!("Planned into {} paths", paths.len());

    // Per-layer loop structure for the first few layers.
    for layer in layers.iter().take(5) {
        let mut wall_counts: std::collections::BTreeMap<usize, (usize, usize)> =
            std::collections::BTreeMap::new();
        for l in &layer.loops {
            let e = wall_counts.entry(l.wall_index).or_insert((0, 0));
            e.0 += 1;
            e.1 += l.points.len();
        }
        println!(
            "layer {:3} order {:7.3}: loops {} (wall idx -> loops/pts: {}) solid_fill_boundary {} infill_boundary {}",
            layer.index,
            layer.order,
            layer.loops.len(),
            wall_counts
                .iter()
                .map(|(w, (n, p))| format!("w{w}:{n}/{p}"))
                .collect::<Vec<_>>()
                .join(" "),
            layer.solid_fill_boundary.len(),
            layer.infill_boundary.len()
        );
    }

    let world_mesh = manifold_core::mesh::Mesh::new(
        object
            .mesh
            .vertices
            .iter()
            .map(|&v| object.transform.transform_point(v))
            .collect(),
        object.mesh.indices.clone(),
    );

    let grid = audit_extrusion_volume(&world_mesh, &paths, &profile.config, cell_size);

    println!("\n--- ALL underfilled Solid-zone cells (fraction < 0.9) ---");
    let under = grid.underfilled_cells(0.9);
    println!("{} cells", under.len());
    for (idx, fraction) in &under {
        let c = grid.cell_center(*idx);
        println!(
            "  idx {idx:?} world({:.1},{:.1},{:.1}) fraction={fraction:.3}",
            c.x, c.y, c.z
        );
    }

    println!("\n--- ALL outside-mesh cells ---");
    let outside = grid.extrusion_outside_mesh_cells();
    println!("{} cells", outside.len());
    for idx in &outside {
        let c = grid.cell_center(*idx);
        let w = grid.accumulated_volume(*idx, VolumeKindBucket::Wall);
        let t = grid.accumulated_volume(*idx, VolumeKindBucket::TopSurface);
        let i = grid.accumulated_volume(*idx, VolumeKindBucket::Infill);
        let o = grid.accumulated_volume(*idx, VolumeKindBucket::Overhang);
        println!(
            "  idx {idx:?} world({:.1},{:.1},{:.1}) wall={w:.3} top={t:.3} infill={i:.3} overhang={o:.3}",
            c.x, c.y, c.z
        );
    }

    println!("\n--- ALL overfilled cells (ratio > 1.0) ---");
    let over = grid.overfilled_cells(1.0);
    println!("{} cells", over.len());
    for (idx, ratio) in &over {
        let c = grid.cell_center(*idx);
        println!(
            "  idx {idx:?} world({:.1},{:.1},{:.1}) ratio={ratio:.3}",
            c.x, c.y, c.z
        );
    }

    // Toolpath move-kind census per layer + vertical extrusion detection.
    println!("\n--- Toolpath move kinds, first 5 layers ---");
    for layer in layers.iter().take(5) {
        let mut kinds: std::collections::BTreeMap<String, (usize, usize)> =
            std::collections::BTreeMap::new();
        for p in &paths {
            for s in &p.segments {
                if (s.order - layer.order).abs() < 1e-9 {
                    let e = kinds.entry(format!("{:?}", s.kind)).or_insert((0, 0));
                    e.0 += 1;
                    e.1 += p.points.len();
                }
            }
        }
        println!(
            "layer {:3} order {:7.3}: {}",
            layer.index,
            layer.order,
            kinds
                .iter()
                .map(|(k, (n, p))| format!("{k}:{n}segs/{p}pts"))
                .collect::<Vec<_>>()
                .join(" ")
        );
    }

    // Vertical extrusion segments: |dz| between consecutive points beyond
    // half a layer height, on an extruding (non-travel, non-wipe) segment.
    println!("\n--- Vertical extrusion segments (|dz| > 0.5mm) ---");
    let mut n_vert = 0;
    for p in &paths {
        let n_pts = p.points.len();
        if n_pts == 0 {
            continue;
        }
        for i in 0..p.segments.len() {
            let s = &p.segments[i];
            if s.kind == MoveKind::Travel || s.kind == MoveKind::Wipe {
                continue;
            }
            let j = (i + 1) % n_pts;
            let a = p.points[i];
            let b = p.points[j];
            let dz = (b.z - a.z).abs();
            if dz > 0.5 {
                n_vert += 1;
                if n_vert <= 200 {
                    let mid = (a + b) * 0.5;
                    println!(
                        "  seg {:?} layer-order {:7.3} dz={dz:7.2} midworld({:.1},{:.1},{:.1})",
                        s.kind, s.order, mid.x, mid.y, mid.z
                    );
                }
            }
        }
    }
    println!("total vertical extrusion segments: {n_vert}");
    Ok(())
}
