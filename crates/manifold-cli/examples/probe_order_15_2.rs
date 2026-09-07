//! Comprehensive test suite for Thingy2 and Thingy3 with 3D walls and topological seeding
use manifold_core::ids::{ObjectId, ToolId};
use manifold_core::machine::Machine;
use manifold_core::object::Object;
use manifold_core::slicing;
use manifold_core::toolpath;
use manifold_core::SlicerConfig;
use manifold_fidget::ScalarField;
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Debug, Deserialize)]
struct ProfileJson {
    machine: Machine,
    config: SlicerConfig,
}

fn check_thingy2(
    config: &SlicerConfig,
    machine: &Machine,
) -> Result<(), Box<dyn std::error::Error>> {
    let model_path = PathBuf::from("/Users/amcgregor/3D/Thingy2.stl");
    let mesh_file = std::fs::File::open(&model_path)?;
    let mesh = manifold_core::stl::load_stl(mesh_file)?;
    let mut objects = vec![Object::new(ObjectId(0), mesh, ToolId(0))];
    manifold_core::object::center_on_bed(&mut objects, &machine.build_volume);

    let slope_profile = machine.slope_profile();
    let layers = slicing::slice_mesh(&objects[0].mesh, config)?;
    let tools = machine.tools.clone();
    let paths = toolpath::plan_with_progress(
        &layers,
        &objects,
        &tools,
        config,
        None,
        &slope_profile,
        &mut |_| {},
    )?;

    // Check 1: Wave overhangs on order 15.2
    let l152_paths = paths
        .iter()
        .filter(|p| {
            p.segments
                .iter()
                .any(|s| (s.order - 15.2).abs() < 1e-4 && s.kind == toolpath::MoveKind::Overhang)
        })
        .count();
    println!(
        "Thingy2 Order 15.2: found {} paths with Overhang moves",
        l152_paths
    );
    assert!(
        l152_paths > 0,
        "Thingy2 Order 15.2 should have wave overhang paths!"
    );

    // Check 2: No flat chords at Z=17.8 spanning across the arch
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
                    flat_chords += 1;
                }
            }
        }
    }
    println!(
        "Thingy2 Order 17.8: flat chords across arch: {}",
        flat_chords
    );
    assert_eq!(
        flat_chords, 0,
        "Thingy2 Order 17.8 should have no flat chords!"
    );

    Ok(())
}

fn check_thingy3(
    config: &SlicerConfig,
    machine: &Machine,
    weight: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    let model_path = PathBuf::from("/Users/amcgregor/3D/Thingy3.stl");
    let mesh_file = std::fs::File::open(&model_path)?;
    let mesh = manifold_core::stl::load_stl(mesh_file)?;
    let mut objects = vec![Object::new(ObjectId(0), mesh, ToolId(0))];
    manifold_core::object::center_on_bed(&mut objects, &machine.build_volume);

    let mut cfg = config.clone();
    cfg.eikonal_surface_order_weight = Some(weight);

    let slope_profile = machine.slope_profile();
    let layers = slicing::slice_mesh(&objects[0].mesh, &cfg)?;
    let tools = machine.tools.clone();
    let paths = toolpath::plan_with_progress(
        &layers,
        &objects,
        &tools,
        &cfg,
        None,
        &slope_profile,
        &mut |_| {},
    )?;

    // Check for outside segments
    let mut outside_count = 0;
    for p in &paths {
        let count = p.points.len();
        for (si, s) in p.segments.iter().enumerate() {
            if s.kind == toolpath::MoveKind::Travel {
                continue;
            }
            let p0 = p.points[si];
            let p1 = p.points[(si + 1) % count];
            let mid = (p0 + p1) * 0.5;

            let l_idx = layers.iter().position(|l| (l.order - s.order).abs() < 1e-4);
            if let Some(li) = l_idx {
                if let Some(sdf) = &layers[li].mesh_sdf {
                    let d = sdf.sample(mid).value;
                    if d > 0.40 {
                        outside_count += 1;
                    }
                }
            }
        }
    }
    println!(
        "Thingy3 (weight = {:.1}): outside segments (SDF > 0.40mm) = {}",
        weight, outside_count
    );
    assert!(
        outside_count < 100,
        "Should have minimal/negligible outside points!"
    );

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let profile_path = PathBuf::from("/Users/amcgregor/3D/profile.json");
    let file = std::fs::File::open(&profile_path)?;
    let profile: ProfileJson = serde_json::from_reader(file)?;
    let config = profile.config;
    let machine = profile.machine;

    println!("Running checks on Thingy2.stl...");
    check_thingy2(&config, &machine)?;

    println!("\nRunning checks on Thingy3.stl (weight = 0.0)...");
    check_thingy3(&config, &machine, 0.0)?;

    println!("\nRunning checks on Thingy3.stl (weight = 0.5)...");
    check_thingy3(&config, &machine, 0.5)?;

    println!("\nAll checks passed successfully!");
    Ok(())
}
