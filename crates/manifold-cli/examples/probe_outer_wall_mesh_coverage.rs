//! Scratch driver to trigger the `MANIFOLD_DEBUG_CORNER` temporary
//! instrumentation in `slicing::slice_mesh` and confirm/deny whether
//! `outer_wall_mesh` (the narrow-band marching-cubes triangle soup feeding
//! DualIso's wall-0 contour extraction) has any vertex coverage in the
//! vent-slot corner region where wall-0 contour extraction produces an
//! open (non-closing) loop.
//!
//! ```sh
//! MANIFOLD_DEBUG_CORNER=1 cargo run --release -p manifold-cli --example probe_outer_wall_mesh_coverage -- \
//!     /Users/amcgregor/3D/Voron_Design_Cube_v7.stl /Users/amcgregor/3D/profile.json
//! ```

use glam::DVec3;
use manifold_core::ids::{ObjectId, ToolId};
use manifold_core::machine::Machine;
use manifold_core::object::{self, Object};
use manifold_core::{slicing, stl, SlicerConfig};

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "Voron_Design_Cube_v7.stl".to_string());
    let profile_path = std::env::args().nth(2);

    let mut config = match &profile_path {
        Some(p) => {
            let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(p)?)?;
            let _machine: Machine = serde_json::from_value(v["machine"].clone())?;
            let config: SlicerConfig = serde_json::from_value(v["config"].clone())?;
            config
        }
        None => SlicerConfig::default(),
    };
    config.order_field = manifold_core::order_field::OrderFieldKind::DualIso;

    let mesh = stl::load_stl(std::fs::File::open(&path)?)?;
    let mut objects = vec![Object::new(ObjectId(0), mesh, ToolId(0))];
    let build_volume = manifold_core::bounds::BoundingVolume::Aabb {
        min: DVec3::ZERO,
        max: DVec3::new(200.0, 200.0, 200.0),
    };
    object::center_on_bed(&mut objects, &build_volume);
    let obj = &objects[0];
    let mut mesh = obj.mesh.clone();
    for v in &mut mesh.vertices {
        *v = obj.transform.transform_point(*v);
    }

    println!("slicing (this will trip TEMP-INSTRUMENT prints to stderr if MANIFOLD_DEBUG_CORNER is set)...");
    let layers = slicing::slice_mesh(&mesh, &config)?;
    println!("done, {} layers", layers.len());

    let mut open_wall0_count = 0usize;
    for layer in &layers {
        for wall in &layer.loops {
            if wall.wall_index == 0 && wall.is_open {
                open_wall0_count += 1;
                println!(
                    "OPEN wall-0 loop at layer order {:.4}: {} points",
                    layer.order,
                    wall.points.len()
                );
            }
        }
    }
    println!(
        "open wall-0 loop count: {} (should be 0 on a watertight mesh)",
        open_wall0_count
    );

    Ok(())
}
