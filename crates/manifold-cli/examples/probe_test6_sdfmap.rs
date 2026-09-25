//! Scratch probe: rasterize the all-face world SDF of Test6.stl on the
//! layer plane to see where the w3 (SDF <= -1.4) sheet exists at each z.
//!
//! Prints a 0.5mm grid over world x/y in 160..190:
//!   '#' sdf <= -1.4   (w3 core: the infill-boundary sheet should live here)
//!   'o' -1.4 < sdf <= -1.0   (between w3 and w2)
//!   '.' otherwise
//!
//! ```sh
//! CARGO_TARGET_DIR=target/build cargo run --release -q -p manifold-cli \
//!     --example probe_test6_sdfmap -- Test6.stl examples/profile.json
//! ```

use manifold_core::ids::{ObjectId, ToolId};
use manifold_core::machine::Machine;
use manifold_core::object::{center_on_bed, Object};
use manifold_fidget::mesh_sdf::MeshSdf;
use manifold_fidget::ScalarField;

#[derive(serde::Deserialize)]
struct Profile {
    machine: Machine,
    config: manifold_core::SlicerConfig,
}

fn main() -> anyhow::Result<()> {
    let mesh_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "Test6.stl".to_string());
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
    let wc = profile.config.wall_count();
    println!(
        "wall_count={} infill_depth(w3)={:.2}",
        wc,
        0.2 + wc as f64 * 0.4
    );
    // Raw SDF line scans for ground truth.
    let scan = |label: &str, pts: Vec<glam::DVec3>| {
        let mut line = String::from(label);
        for p in pts {
            let v = sdf.sample(p).value;
            line.push_str(&format!(" {v:6.2}"));
        }
        println!("{line}");
    };
    for z in [1.0f64, 1.65, 2.37, 3.0] {
        let pts: Vec<_> = (158..=192)
            .map(|i| glam::DVec3::new(i as f64, 175.0, z))
            .collect();
        scan(&format!("z={z} y=175  x:"), pts);
        let pts: Vec<_> = (158..=192)
            .map(|i| glam::DVec3::new(175.0, i as f64, z))
            .collect();
        scan(&format!("z={z} x=175  y:"), pts);
    }
    {
        // Cross-section SDF maps through the groove region (y fixed).
        for &y in &[168.0f64, 175.0, 182.0] {
            println!("\n=== SDF cross-section at y={y} (x 170..191, z 0.5..4.9, 0.5mm) ===");
            println!("      x: 170         175         180         185         190");
            let mut z = 4.9;
            while z >= 0.5 - 1e-9 {
                let mut row = String::new();
                let mut x = 170.0;
                while x <= 191.0 + 1e-9 {
                    let v = sdf.sample(glam::DVec3::new(x, y, z)).value;
                    row.push(if v > 0.5 {
                        '.'
                    } else if v > 0.0 {
                        'o'
                    } else if v > -0.7 {
                        '+'
                    } else {
                        '#'
                    });
                    x += 0.5;
                }
                println!("{z:4.1} {row}");
                z -= 0.5;
            }
        }
    }
    Ok(())
}
