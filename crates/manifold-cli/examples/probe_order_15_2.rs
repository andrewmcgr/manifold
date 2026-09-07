//! Regression probe for wave overhangs, bridges, and non-planar inner wall reprojection.
use manifold_core::ids::{ObjectId, ToolId};
use manifold_core::machine::Machine;
use manifold_core::object::Object;
use manifold_core::slicing;
use manifold_core::toolpath;
use manifold_core::SlicerConfig;
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Debug, Deserialize)]
struct ProfileJson {
    machine: Machine,
    config: SlicerConfig,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let profile_path = PathBuf::from("/Users/amcgregor/3D/profile.json");
    let model_path = PathBuf::from("/Users/amcgregor/3D/Thingy2.stl");

    let file = std::fs::File::open(&profile_path)?;
    let profile: ProfileJson = serde_json::from_reader(file)?;
    let config = profile.config;
    let machine = profile.machine;

    let mesh_file = std::fs::File::open(&model_path)?;
    let mesh = manifold_core::stl::load_stl(mesh_file)?;
    let mut objects = vec![Object::new(ObjectId(0), mesh, ToolId(0))];
    manifold_core::object::center_on_bed(&mut objects, &machine.build_volume);

    let slope_profile = machine.slope_profile();
    let layers = slicing::slice_mesh(&objects[0].mesh, &config)?;

    let tools = machine.tools.clone();
    let paths = toolpath::plan_with_progress(
        &layers,
        &objects,
        &tools,
        &config,
        None,
        &slope_profile,
        &mut |_| {},
    )?;

    // Check 1: Verify wave overhangs on order 15.2 (layer 75)
    let l152_paths: Vec<_> = paths
        .iter()
        .filter(|p| {
            p.segments
                .iter()
                .any(|s| (s.order - 15.2).abs() < 1e-4 && s.kind == toolpath::MoveKind::Overhang)
        })
        .collect();
    println!(
        "Order 15.2: found {} paths with Overhang moves",
        l152_paths.len()
    );
    assert!(
        !l152_paths.is_empty(),
        "Order 15.2 should have wave overhang paths!"
    );

    let mut bridge_orders = std::collections::BTreeMap::<String, usize>::new();
    for p in &paths {
        for s in &p.segments {
            if s.kind == toolpath::MoveKind::Bridge {
                *bridge_orders.entry(format!("{:.3}", s.order)).or_default() += 1;
            }
        }
    }
    println!("Bridge segments by order: {:?}", bridge_orders);

    // Check 3: Verify no flat horizontal chords across the arch valley on order 17.8
    let mut flat_chords = 0;
    for path in &paths {
        for (seg_idx, segment) in path.segments.iter().enumerate() {
            if segment.kind == toolpath::MoveKind::WallInner && (segment.order - 17.8).abs() < 1e-4
            {
                let p0 = path.points[seg_idx];
                let p1 = path.points[(seg_idx + 1) % path.points.len()];
                if ((p0.x < 170.0 && p1.x > 180.0) || (p0.x > 180.0 && p1.x < 170.0))
                    && (p0.z - 17.8).abs() < 0.1
                    && (p1.z - 17.8).abs() < 0.1
                {
                    println!(
                        "ERROR: Found flat chord spanning arch at Z=17.8: p0={:?}, p1={:?}",
                        p0, p1
                    );
                    flat_chords += 1;
                }
            }
        }
    }
    println!(
        "Order 17.8: flat chords at Z=17.8 spanning across the arch: {}",
        flat_chords
    );
    assert_eq!(
        flat_chords, 0,
        "No inner wall should span horizontally across the valley at Z=17.8!"
    );

    println!("All regression checks passed!");
    Ok(())
}
