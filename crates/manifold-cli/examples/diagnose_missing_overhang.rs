use manifold_core::ids::{ObjectId, ToolId};
use manifold_core::machine::Machine;
use manifold_core::object::{center_on_bed, Object};
use manifold_core::{slicing, stl, toolpath, SlicerConfig};

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let stl_path = args.next().expect("usage: <stl path> <profile.json path>");
    let profile_path = args.next().expect("usage: <stl path> <profile.json path>");

    let file = std::fs::File::open(&stl_path)?;
    let mesh = stl::load_stl(file)?;

    let profile_json: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(&profile_path)?)?;
    let machine: Machine = serde_json::from_value(profile_json["machine"].clone())?;
    let mut config: SlicerConfig = serde_json::from_value(profile_json["config"].clone())?;

    println!("Slicing {stl_path} centered on bed with real machine + profile {profile_path}...");

    config.eikonal_conform_top_surfaces = true;

    let object = Object::new(ObjectId(0), mesh, ToolId(0));
    let mut objects = vec![object];
    center_on_bed(&mut objects, &machine.build_volume);

    let layers = slicing::slice_workspace(&objects, &[ObjectId(0)], &config)?;
    println!("Sliced into {} layers", layers.len());

    let paths = toolpath::plan(&layers, &objects, &machine.tools, &config)?;
    println!("Planned into {} paths", paths.len());

    toolpath::validate_within_bounds(&paths, &machine.build_volume)?;
    println!("SUCCESS: Sliced and validated within machine build volume with 0 errors!");

    Ok(())
}
