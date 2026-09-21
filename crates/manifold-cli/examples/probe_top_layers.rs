//! Scratch probe: dump loop structure (3D z-extent) for the top layers
//! (layer orders ~11.9-14.2) where the vertical-extrusion defect lives.
//!
//! ```sh
//! CARGO_TARGET_DIR=target/release cargo run --release -p manifold-cli --example probe_top_layers -- \
//!     /path/to/mesh.stl /path/to/profile.json
//! ```
use manifold_core::ids::{ObjectId, ToolId};
use manifold_core::machine::Machine;
use manifold_core::object::{center_on_bed, Object};

use manifold_core::{slicing, stl};

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
    let mesh = stl::load_stl(std::fs::File::open(&mesh_path)?)?;
    let json = std::fs::read_to_string(&profile_path)?;
    let profile: Profile = serde_json::from_str(&json)?;

    let object = Object::new(ObjectId(0), mesh, ToolId(0));
    let mut objects = vec![object];
    center_on_bed(&mut objects, &profile.machine.build_volume);
    let object = &objects[0];

    let layers = slicing::slice_object(object, &profile.config)?;
    println!("{} layers", layers.len());
    println!("showing layers with order > 11.0");
    for layer in layers.iter().filter(|l| l.order > 11.0) {
        println!("\nlayer {:3} order {:8.3}", layer.index, layer.order);
        for l in &layer.loops {
            let zmin = l.points.iter().map(|p| p.z).fold(f64::INFINITY, f64::min);
            let zmax = l
                .points
                .iter()
                .map(|p| p.z)
                .fold(f64::NEG_INFINITY, f64::max);
            let xmin = l.points.iter().map(|p| p.x).fold(f64::INFINITY, f64::min);
            let xmax = l
                .points
                .iter()
                .map(|p| p.x)
                .fold(f64::NEG_INFINITY, f64::max);
            let ymin = l.points.iter().map(|p| p.y).fold(f64::INFINITY, f64::min);
            let ymax = l
                .points
                .iter()
                .map(|p| p.y)
                .fold(f64::NEG_INFINITY, f64::max);
            println!(
                "  loop wall{:<2} isl{:2} open={} n={:<5} z[{zmin:6.2},{zmax:6.2}] x[{xmin:6.1},{xmax:6.1}] y[{ymin:6.1},{ymax:6.1}]",
                l.wall_index,
                l.island,
                l.is_open,
                l.points.len()
            );
        }
        for (i, poly) in layer.infill_boundary.iter().enumerate() {
            let zmin = poly.iter().map(|p| p.z).fold(f64::INFINITY, f64::min);
            let zmax = poly.iter().map(|p| p.z).fold(f64::NEG_INFINITY, f64::max);
            let xmin = poly.iter().map(|p| p.x).fold(f64::INFINITY, f64::min);
            let xmax = poly.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max);
            let ymin = poly.iter().map(|p| p.y).fold(f64::INFINITY, f64::min);
            let ymax = poly.iter().map(|p| p.y).fold(f64::NEG_INFINITY, f64::max);
            println!(
                "  infill_boundary[{i}] n={:<5} z[{zmin:6.2},{zmax:6.2}] x[{xmin:6.1},{xmax:6.1}] y[{ymin:6.1},{ymax:6.1}]",
                poly.len()
            );
        }
    }
    Ok(())
}
