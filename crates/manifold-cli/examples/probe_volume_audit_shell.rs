//! Scratch probe: slice a real mesh with a real saved profile.json, run the
//! extrusion-volume audit against the resulting toolpaths, and report the
//! worst underfilled Solid-zone cells near the bed -- investigating a
//! reported bug where large parts of the first few layers' shell go
//! missing, but the volume-audit GUI overlay didn't flag it. Not part of
//! the `manifold` binary.
//!
//! ```sh
//! cargo run --release -p manifold-cli --example probe_volume_audit_shell -- \
//!     /path/to/mesh.stl /path/to/profile.json [cell_size]
//! ```

use manifold_core::ids::{ObjectId, ToolId};
use manifold_core::machine::Machine;
use manifold_core::object::{center_on_bed, Object};
use manifold_core::volume_audit::{audit_extrusion_volume, VolumeKindBucket};
use manifold_core::{slicing, stl, toolpath, SlicerConfig};

#[derive(serde::Deserialize)]
struct Profile {
    machine: Machine,
    config: SlicerConfig,
}

fn main() -> anyhow::Result<()> {
    let mesh_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "TestObj1.stl".to_string());
    let profile_path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "/Users/amcgregor/3D/profile.json".to_string());
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

    println!(
        "Slicing {mesh_path} with profile {profile_path} (order_field={:?}, cell_size={cell_size})...",
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

    // Audit against the world-space mesh -- object.mesh is local-space,
    // but the planned toolpaths are world-space (see the same fix already
    // applied to the GUI's own volume-audit visualization).
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

    println!("\n--- Solid-zone underfill, sorted by fraction (worst first) ---");
    let mut under = grid.underfilled_cells(0.9);
    under.sort_by(|a, b| a.1.total_cmp(&b.1));
    println!(
        "{} Solid-zone cells below 90% of expected volume (of the grid's total)",
        under.len()
    );
    for (idx, fraction) in under.iter().take(40) {
        let center = grid.cell_center(*idx);
        let wall = grid.accumulated_volume(*idx, VolumeKindBucket::Wall);
        let top = grid.accumulated_volume(*idx, VolumeKindBucket::TopSurface);
        let infill = grid.accumulated_volume(*idx, VolumeKindBucket::Infill);
        let overhang = grid.accumulated_volume(*idx, VolumeKindBucket::Overhang);
        println!(
            "  cell {idx:?} world({:.1},{:.1},{:.1}) fraction={fraction:.3} wall={wall:.4} top={top:.4} infill={infill:.4} overhang={overhang:.4}",
            center.x, center.y, center.z
        );
    }

    println!("\n--- Solid-zone underfill restricted to z < 3mm (near-bed layers) ---");
    let near_bed: Vec<_> = under
        .iter()
        .filter(|(idx, _)| grid.cell_center(*idx).z < 3.0)
        .collect();
    println!("{} near-bed Solid-zone cells below 90%", near_bed.len());
    for (idx, fraction) in near_bed.iter().take(40) {
        let center = grid.cell_center(*idx);
        println!(
            "  cell {idx:?} world({:.1},{:.1},{:.1}) fraction={fraction:.3}",
            center.x, center.y, center.z
        );
    }

    println!("\n--- Outside-the-mesh material (extrusion into open air) ---");
    let outside = grid.extrusion_outside_mesh_cells();
    println!("{} cells", outside.len());
    let mut zhist: std::collections::BTreeMap<i32, usize> = std::collections::BTreeMap::new();
    for idx in &outside {
        let c = grid.cell_center(*idx);
        *zhist.entry((c.z / 2.0) as i32).or_default() += 1;
    }
    println!("z-band (2mm) histogram:");
    for (zb, n) in &zhist {
        println!("  z {:4}..{:4} : {}", 2 * zb, 2 * zb + 2, n);
    }
    println!("\ncentral-region (world x,y in 166..184) outside-mesh cells, kind volumes:");
    for idx in outside.iter().filter(|idx| {
        let c = grid.cell_center(**idx);
        (166.0..184.0).contains(&c.x) && (166.0..184.0).contains(&c.y)
    }) {
        let c = grid.cell_center(*idx);
        let wall = grid.accumulated_volume(*idx, VolumeKindBucket::Wall);
        let top = grid.accumulated_volume(*idx, VolumeKindBucket::TopSurface);
        let infill = grid.accumulated_volume(*idx, VolumeKindBucket::Infill);
        let overhang = grid.accumulated_volume(*idx, VolumeKindBucket::Overhang);
        println!(
            "  cell {idx:?} world({x:.1},{y:.1},{z:.1}) wall={wall:.3} top={top:.3} infill={infill:.3} overhang={overhang:.3}",
            x = c.x,
            y = c.y,
            z = c.z,
            idx = *idx,
            wall = wall,
            top = top,
            infill = infill,
            overhang = overhang
        );
    }

    println!("\n--- Overfill ratio (worst first, top 20) ---");
    let mut over = grid.overfilled_cells(0.0);
    over.sort_by(|a, b| b.1.total_cmp(&a.1));
    for (idx, ratio) in over.iter().take(20) {
        let center = grid.cell_center(*idx);
        println!(
            "  cell {idx:?} world({:.1},{:.1},{:.1}) ratio={ratio:.3}",
            center.x, center.y, center.z
        );
    }

    Ok(())
}
