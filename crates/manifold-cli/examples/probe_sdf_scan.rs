//! Scratch probe: sample the object's MeshSdf (same ground truth as the
//! volume audit) on coarse grids to map the internal void structure.
//!
//! ```sh
//! CARGO_TARGET_DIR=target/release cargo run --release -p manifold-cli --example probe_sdf_scan -- \
//!     /path/to/mesh.stl /path/to/profile.json
//! ```

use manifold_core::ids::{ObjectId, ToolId};
use manifold_core::machine::Machine;
use manifold_core::object::{center_on_bed, Object};
use manifold_fidget::mesh_sdf::MeshSdf;
use manifold_fidget::ScalarField;

#[derive(serde::Deserialize)]
struct Profile {
    machine: Machine,
}

fn main() -> anyhow::Result<()> {
    let mesh_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "TestObj1.stl".to_string());
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
    let sdf = MeshSdf::new(world_mesh.vertices.clone(), faces);
    let inside = |p: glam::DVec3| sdf.sample(p).value <= 0.0;

    // (x,z) cross-section at mid-Y.
    let y0 = 175.0;
    println!("(x,z) cross-section at y={y0} (world), inside=#");
    println!("x range [165,185], z range [0,14]");
    for i in (0..=28).rev() {
        let z = 14.0 * i as f64 / 28.0;
        let mut row = String::new();
        for j in 0..=40 {
            let x = 165.0 + 20.0 * j as f64 / 40.0;
            row.push(if inside(glam::DVec3::new(x, y0, z)) {
                '#'
            } else {
                '.'
            });
        }
        println!("{z:5.2} {row}");
    }
    println!("       165        170        175        180        185");

    // (y,z) cross-sections at a few X planes.
    for x0 in [170.0_f64, 175.0, 180.0] {
        println!("\n(y,z) cross-section at x={x0}, inside=#");
        for i in (0..=28).rev() {
            let z = 14.0 * i as f64 / 28.0;
            let mut row = String::new();
            for j in 0..=90 {
                let y = 152.5 + 45.0 * j as f64 / 90.0;
                row.push(if inside(glam::DVec3::new(x0, y, z)) {
                    '#'
                } else {
                    '.'
                });
            }
            println!("{z:5.2} {row}");
        }
        println!("       152.5                     175                      197.5");
    }
    Ok(())
}
