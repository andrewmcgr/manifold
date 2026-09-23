//! Scratch probe: dump the material/air mask of the TOP region of Test5.stl
//! directly from the world-space MeshSdf (no slicing). '#' = inside mesh
//! (SDF<0), ' ' = air. Object-coord grid mapped through the object transform.
//!
//! ```sh
//! CARGO_TARGET_DIR=target/build cargo run --release -p manifold-cli --example probe_topmask -- \
//!     Test5.stl examples/profile.json
//! ```
use manifold_core::ids::{ObjectId, ToolId};
use manifold_core::machine::Machine;
use manifold_core::object::{center_on_bed, Object};
use manifold_fidget::mesh_sdf::MeshSdf;
use manifold_fidget::ScalarField;

#[derive(serde::Deserialize)]
struct Profile {
    machine: Machine,
    #[allow(dead_code)]
    config: manifold_core::SlicerConfig,
}

fn main() -> anyhow::Result<()> {
    let mesh_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "Test5.stl".to_string());
    let profile_path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "examples/profile.json".to_string());
    let mesh = manifold_core::stl::load_stl(std::fs::File::open(&mesh_path)?)?;
    let json = std::fs::read_to_string(&profile_path)?;
    let profile: Profile = serde_json::from_str(&json)?;

    let object = Object::new(ObjectId(0), mesh, ToolId(0));
    let mut objects = vec![object];
    center_on_bed(&mut objects, &profile.machine.build_volume);
    let object = &objects[0];

    let world_mesh = manifold_core::mesh::Mesh::new(
        object
            .mesh
            .vertices
            .iter()
            .map(|&v| object.transform.transform_point(v))
            .collect(),
        object.mesh.indices.clone(),
    );
    let faces: Vec<[usize; 3]> = world_mesh
        .indices
        .as_chunks::<3>()
        .0
        .iter()
        .map(|c| [c[0] as usize, c[1] as usize, c[2] as usize])
        .collect();
    let sdf = MeshSdf::new(world_mesh.vertices.clone(), faces.clone());

    let res = 0.5f64;
    for zi in (20..=30).step_by(1) {
        let z = zi as f64;
        println!("\n=== z = {z:.0} ===");
        for yi in 0..=59 {
            let y = yi as f64 * res;
            let mut row = String::new();
            for xi in 0..=59 {
                let x = xi as f64 * res;
                let p = object.transform.transform_point(glam::DVec3::new(x, y, z));
                let s = sdf.sample(p).value;
                row.push(if s < 0.0 { '#' } else { ' ' });
            }
            // label every 4th row (2mm)
            if yi % 4 == 0 {
                println!("  y={y:4.1} {row}");
            } else {
                println!("        {row}");
            }
        }
    }
    Ok(())
}
