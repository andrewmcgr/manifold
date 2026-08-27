use glam::DVec3;
use manifold_core::bounds::BoundingVolume;
use manifold_core::ids::{ObjectId, ToolId};
use manifold_core::machine::Machine;
use manifold_core::object::Object;
use manifold_core::{slicing, stl, SlicerConfig};

fn slice_and_report(
    label: &str,
    mesh: &manifold_core::mesh::Mesh,
    _machine: &Machine,
    config: &SlicerConfig,
) -> anyhow::Result<()> {
    let object = Object::new(ObjectId(0), mesh.clone(), ToolId(0));
    let objects = vec![object];
    let layers = slicing::slice_workspace(&objects, &[ObjectId(0)], config)?;
    let wave_plan = manifold_core::wave_overhang::plan_wave_overhangs(&layers, config, ToolId(0));

    println!("--- {label} Wave Overhang Paths ---");
    let mut total_wave_paths = 0usize;
    for (k, paths) in wave_plan.paths_by_layer.iter().enumerate() {
        if !paths.is_empty() {
            total_wave_paths += paths.len();
            let l = &layers[k];
            println!(
                "  Layer k={:2} order={:.4}: {} wave overhang paths",
                k,
                l.order,
                paths.len()
            );
            for (pi, p) in paths.iter().enumerate() {
                let min_pt = p
                    .points
                    .iter()
                    .fold(DVec3::splat(f64::INFINITY), |a, b| a.min(*b));
                let max_pt = p
                    .points
                    .iter()
                    .fold(DVec3::splat(f64::NEG_INFINITY), |a, b| a.max(*b));
                println!(
                    "    Path {pi}: {} pts, bbox: min={min_pt:.3?} max={max_pt:.3?}",
                    p.points.len()
                );
            }
        }
    }
    println!("Total wave overhang paths: {total_wave_paths}\n");

    Ok(())
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let stl_path = args.next().expect("usage: <stl path> <profile.json path>");
    let profile_path = args.next().expect("usage: <stl path> <profile.json path>");

    let file = std::fs::File::open(&stl_path)?;
    let mesh = stl::load_stl(file)?;

    let profile_json: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(&profile_path)?)?;
    let mut machine: Machine = serde_json::from_value(profile_json["machine"].clone())?;
    let config: SlicerConfig = serde_json::from_value(profile_json["config"].clone())?;

    machine.build_volume = BoundingVolume::Aabb {
        min: DVec3::splat(-1_000.0),
        max: DVec3::splat(1_000.0),
    };

    println!("slicing {stl_path} with real profile {profile_path}...");
    println!();

    slice_and_report("as-configured (2 walls)", &mesh, &machine, &config)?;

    Ok(())
}
