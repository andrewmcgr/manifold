//! Scratch probe: sample the raw MeshSdf finely across the midline strip to
//! establish ground truth for defect-1 (midline underfill in the first layers).
//!
//! ```sh
//! CARGO_TARGET_DIR=target/release cargo run --release -p manifold-cli --example probe_center_sdf -- \
//!     TestObj1.stl examples/profile.json
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
    let val = |p: glam::DVec3| sdf.sample(p).value;
    let inside = |p: glam::DVec3| val(p) <= 0.0;

    // 1. Fine x-scan across the midline at mid-Y, for the bottom few layers.
    let y0 = 175.0;
    println!("=== fine x-scan at y={y0} (value = SDF dist, negative=inside) ===");
    for z in [0.2_f64, 0.4, 0.6, 0.8, 1.0, 1.5] {
        print!("{z:4.1}: ");
        for i in 0..=16 {
            let x = 171.0 + 0.5 * i as f64; // x in [171,179] step 0.5
            let v = val(glam::DVec3::new(x, y0, z));
            if v.is_finite() {
                print!("{v:7.3} ");
            } else {
                print!("    +inf ");
            }
        }
        println!();
    }
    println!("       x: 171.0  171.5  172.0  172.5  173.0  173.5  174.0  174.5  175.0");
    println!("          175.5  176.0  176.5  177.0  177.5  178.0  178.5  179.0");

    // 2. Binary inside/outside at z=0.2 and z=0.4 across the midline.
    for z in [0.2_f64, 0.4, 0.6] {
        println!("\n=== inside(#)/void(.) at z={z}, y={y0}, x in [171,179] step 0.2 ===");
        let mut row = String::new();
        for i in 0..=40 {
            let x = 171.0 + 8.0 * i as f64 / 40.0;
            row.push(if inside(glam::DVec3::new(x, y0, z)) {
                '#'
            } else {
                '.'
            });
        }
        println!("{z:4.1}: {row}");
        println!("       171       173       175       177       179");
    }

    // 3. 2D map of the midline strip at z=0.2 (x[171,179] x y[171,179]).
    println!("\n=== 2D slice at z=0.2, x[171,179] (cols), y[179,171] (rows, top->bottom) ===");
    for i in (0..=24).rev() {
        let y = 171.0 + 8.0 * i as f64 / 24.0;
        let mut row = String::new();
        for j in 0..=32 {
            let x = 171.0 + 8.0 * j as f64 / 32.0;
            row.push(if inside(glam::DVec3::new(x, y, 0.2)) {
                '#'
            } else {
                '.'
            });
        }
        println!("{y:6.1} {row}");
    }
    println!("       171       173       175       177       179");

    // 4. Same 2D map a bit higher (z=0.6) to see how the channel evolves.
    println!("\n=== 2D slice at z=0.6, x[171,179] x y[171,179] ===");
    for i in (0..=24).rev() {
        let y = 171.0 + 8.0 * i as f64 / 24.0;
        let mut row = String::new();
        for j in 0..=32 {
            let x = 171.0 + 8.0 * j as f64 / 32.0;
            row.push(if inside(glam::DVec3::new(x, y, 0.6)) {
                '#'
            } else {
                '.'
            });
        }
        println!("{y:6.1} {row}");
    }
    println!("       171       173       175       177       179");

    Ok(())
}
