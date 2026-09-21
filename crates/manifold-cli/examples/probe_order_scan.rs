//! Scratch probe: sample the layer's order field along vertical lines at
//! the bore void's x-faces, to test the "two roots at the same order value"
//! theory for the zigzag AllWalls infill reconstruction.
//!
//! ```sh
//! CARGO_TARGET_DIR=target/release cargo run --release -p manifold-cli --example probe_order_scan -- \
//!     /path/to/mesh.stl /path/to/profile.json [layer_index]
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
    let layer_index: usize = std::env::args()
        .nth(3)
        .map(|s| s.parse().expect("layer index"))
        .unwrap_or(66);
    let mesh = stl::load_stl(std::fs::File::open(&mesh_path)?)?;
    let json = std::fs::read_to_string(&profile_path)?;
    let profile: Profile = serde_json::from_str(&json)?;

    let object = Object::new(ObjectId(0), mesh, ToolId(0));
    let mut objects = vec![object];
    center_on_bed(&mut objects, &profile.machine.build_volume);
    let object = &objects[0];

    let layers = slicing::slice_object(object, &profile.config)?;
    let layer = &layers[layer_index];
    println!("layer {} order {:.6}", layer.index, layer.order);

    // Vertical lines at the two x-faces of the central bore (y-mid),
    // matching where the zigzag AllWalls rings sit.
    let samples: [(f64, f64); 6] = [
        (173.48, 175.10),
        (173.51, 175.15),
        (173.40, 175.41),
        (173.37, 175.47),
        (173.5, 176.5),
        (173.4, 177.5),
    ];
    let target = layer.order;
    for (sx, sy) in samples {
        print!("({sx:.2},{sy:.2}) target={target:.4}: ");
        let mut crossings: Vec<f64> = Vec::new();
        let mut prev_z = 0.0f64;
        let mut prev_r = f64::NAN;
        for i in 0..=280 {
            let z = i as f64 * 0.05; // 0 ..= 14 in 0.05 steps
            let r = layer.order_field.order(glam::DVec3::new(sx, sy, z)) - target;
            if r.is_finite()
                && prev_r.is_finite()
                && ((prev_r > 0.0 && r <= 0.0) || (prev_r < 0.0 && r >= 0.0))
            {
                // linear interp for the crossing z
                let tc = prev_r / (prev_r - r);
                crossings.push(prev_z + tc * (z - prev_z));
            }
            prev_z = z;
            prev_r = r;
        }
        print!(
            "roots={:?} ",
            crossings
                .iter()
                .map(|c| format!("{c:.3}"))
                .collect::<Vec<_>>()
                .join(",")
        );
        // also sample a few key heights
        for z in [
            0.5f64, 1.0, 1.5, 1.66, 2.0, 4.0, 6.0, 8.0, 10.0, 11.0, 11.4, 11.5, 11.9, 12.5, 13.5,
        ] {
            let r = layer.order_field.order(glam::DVec3::new(sx, sy, z)) - target;
            print!(" z{z:.2}:{r:+7.3} |");
        }
        println!();
    }
    Ok(())
}
