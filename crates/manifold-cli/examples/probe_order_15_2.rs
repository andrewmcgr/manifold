//! Automated verification of Tangent Surface categorization on Thingy3.stl and Thingy2.stl
use manifold_core::ids::{ObjectId, ToolId};
use manifold_core::machine::Machine;
use manifold_core::object::Object;
use manifold_core::slicing;
use manifold_core::toolpath::{self, MoveKind};
use manifold_core::SlicerConfig;
use manifold_fidget::ScalarField;
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Debug, Deserialize)]
struct ProfileJson {
    machine: Machine,
    config: SlicerConfig,
}

fn test_mesh(model_name: &str) -> Result<(), Box<dyn std::error::Error>> {
    let profile_path = PathBuf::from("/Users/amcgregor/3D/profile.json");
    let model_path = PathBuf::from(format!("/Users/amcgregor/3D/{}", model_name));

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

    println!("\n==========================================");
    println!("Testing Tangent Surfaces on {}", model_name);
    println!("==========================================");

    let mut true_downward_overhangs = 0;
    let mut false_upward_overhangs = 0;
    let mut wall_outer_count = 0;
    let mut infill_count = 0;
    let mut debug_excluded_count = 0;

    for (pi, p) in paths.iter().enumerate() {
        let n = p.points.len();
        for (si, s) in p.segments.iter().enumerate() {
            match s.kind {
                MoveKind::WallOuter => wall_outer_count += 1,
                MoveKind::Infill => infill_count += 1,
                MoveKind::DebugExcluded => debug_excluded_count += 1,
                MoveKind::Overhang => {
                    let p0 = p.points[si];
                    let p1 = p.points[(si + 1) % n];
                    let mid = (p0 + p1) * 0.5;

                    let l_idx = layers
                        .iter()
                        .position(|l| (l.order - s.order).abs() < 1e-4)
                        .unwrap();
                    let sdf = layers[l_idx].mesh_sdf.as_ref().unwrap();

                    let eps = 0.02;
                    let dx = sdf.sample(mid + glam::DVec3::X * eps).value
                        - sdf.sample(mid - glam::DVec3::X * eps).value;
                    let dy = sdf.sample(mid + glam::DVec3::Y * eps).value
                        - sdf.sample(mid - glam::DVec3::Y * eps).value;
                    let dz = sdf.sample(mid + glam::DVec3::Z * eps).value
                        - sdf.sample(mid - glam::DVec3::Z * eps).value;
                    let len = (dx * dx + dy * dy + dz * dz).sqrt();
                    let nz = if len > 1e-6 { dz / len } else { 0.0 };

                    // An overhang segment must face downward (nz <= 0.15)
                    if nz > 0.15 {
                        false_upward_overhangs += 1;
                        if false_upward_overhangs <= 5 {
                            println!("  False upward overhang (nz={:.3}): path #{}, seg #{}, order={:.3}, mid={:.2?}",
                                nz, pi, si, s.order, mid);
                        }
                    } else {
                        true_downward_overhangs += 1;
                    }
                }
                _ => {}
            }
        }
    }

    println!("Move counts for {}:", model_name);
    println!(
        "  True downward overhang segments: {}",
        true_downward_overhangs
    );
    println!(
        "  False upward overhang segments: {}",
        false_upward_overhangs
    );
    println!("  WallOuter segments: {}", wall_outer_count);
    println!("  Infill segments: {}", infill_count);
    println!("  DebugExcluded segments: {}", debug_excluded_count);

    assert_eq!(
        debug_excluded_count, 0,
        "No DebugExcluded moves should exist!"
    );
    assert_eq!(
        false_upward_overhangs, 0,
        "No upward-facing surfaces should be tagged as Overhang!"
    );
    assert!(
        true_downward_overhangs > 0,
        "Should have true downward overhang segments!"
    );
    assert!(wall_outer_count > 0, "Should have WallOuter segments!");

    println!("All checks passed for {}!", model_name);
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    test_mesh("Thingy3.stl")?;
    test_mesh("Thingy2.stl")?;
    println!("\nAll tangent surface checks passed cleanly!");
    Ok(())
}
