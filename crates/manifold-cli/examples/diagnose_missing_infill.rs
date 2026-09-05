//! Scratch diagnostic: slice a mesh with a real saved profile.json (same
//! shape manifold-gui's `Profile` serializes) and dump per-layer info near
//! order ~14 to investigate a reported missing-infill layer on
//! Voron_Design_Cube_v7.stl. Not part of the `manifold` binary.

use manifold_core::machine::Machine;
use manifold_core::object::{center_on_bed, Object};
use manifold_core::{slice_to_gcode, SlicerConfig, Workspace};
use std::io::BufReader;

#[derive(serde::Deserialize)]
struct Profile {
    machine: Machine,
    config: SlicerConfig,
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let profile_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/Users/amcgregor/3D/profile.json".to_string());
    let mesh_path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "Voron_Design_Cube_v7.stl".to_string());
    let output_path = std::env::args()
        .nth(3)
        .unwrap_or_else(|| "/tmp/diag.gcode".to_string());

    let json = std::fs::read_to_string(&profile_path)?;
    let profile: Profile = serde_json::from_str(&json)?;

    let file = std::fs::File::open(&mesh_path)?;
    let mesh = manifold_core::stl::load_stl(BufReader::new(file))?;
    let mut objects = vec![Object::new(
        manifold_core::ids::ObjectId(0),
        mesh,
        manifold_core::ids::ToolId(0),
    )];
    let mut machine = profile.machine;
    center_on_bed(&mut objects, &machine.build_volume);
    // center_on_bed may need the machine mutably elsewhere; kept immutable
    // here since build_volume isn't mutated by this call.
    let _ = &mut machine;

    let workspace = Workspace::new(objects, machine, profile.config);
    let gcode = slice_to_gcode(&workspace)?;
    std::fs::write(&output_path, gcode)?;
    println!("wrote {}", output_path);
    Ok(())
}
