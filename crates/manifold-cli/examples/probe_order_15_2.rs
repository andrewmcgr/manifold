//! Verify wave overhang paths on Thingy3.stl and Thingy2.stl with new surface planning
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

    println!("\nTesting {}", model_name);
    let mut total_overhang_paths = 0;
    let mut total_overhang_segs = 0;
    let mut mid_air_paths = 0;

    for (pi, p) in paths.iter().enumerate() {
        for s in &p.segments {
            if s.kind == toolpath::MoveKind::Overhang {
                total_overhang_paths += 1;
                total_overhang_segs += p
                    .segments
                    .iter()
                    .filter(|seg| seg.kind == toolpath::MoveKind::Overhang)
                    .count();

                let mut min = glam::DVec3::splat(f64::INFINITY);
                let mut max = glam::DVec3::splat(f64::NEG_INFINITY);
                for pt in &p.points {
                    min = min.min(*pt);
                    max = max.max(*pt);
                }
                // Check if this overhang is isolated in the middle (x between 10 and 20) with z < 22 while order > 26
                if min.x > 9.0 && max.x < 21.0 && max.z < 22.0 && s.order > 26.0 {
                    mid_air_paths += 1;
                    println!("  MID-AIR Overhang Path #{}: order={:.3}, z=[{:.2}, {:.2}], x=[{:.2}, {:.2}], y=[{:.2}, {:.2}]",
                        pi, s.order, min.z, max.z, min.x, max.x, min.y, max.y);
                }
                break;
            }
        }
    }
    println!("  Total Overhang paths: {}", total_overhang_paths);
    println!("  Total Overhang segments: {}", total_overhang_segs);
    println!("  Mid-air detached Overhang paths: {}", mid_air_paths);

    assert_eq!(
        mid_air_paths, 0,
        "No mid-air detached overhang paths should exist!"
    );
    assert!(total_overhang_paths > 0, "Should have wave overhang paths!");

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    test_mesh("Thingy3.stl")?;
    test_mesh("Thingy2.stl")?;
    println!("\nAll mesh tests passed cleanly!");
    Ok(())
}
