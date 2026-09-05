//! Scratch diagnostic: slice `Voron_Design_Cube_v7.stl` with a real saved
//! `profile.json` and, for every layer whose order falls in a given range,
//! report wall-0 loop fragmentation (loop count, open/closed, point counts)
//! and infill-boundary emptiness restricted to a spatial corner region --
//! to investigate a report of fragmented outer-wall segments and missing
//! infill near the min-X/min-Y corner (mesh bbox min = (85, 85)) spanning
//! order ~9 to ~15.
//!
//! ```sh
//! cargo run --release -p manifold-cli --example diagnose_corner_fragmentation -- \
//!     /Users/amcgregor/3D/Voron_Design_Cube_v7.stl /Users/amcgregor/3D/profile.json \
//!     85.0 85.0 20.0 9.0 15.5
//! ```

use glam::DVec3;
use manifold_core::ids::{ObjectId, ToolId};
use manifold_core::machine::Machine;
use manifold_core::object::{self, Object};
use manifold_core::{slicing, stl, SlicerConfig};

fn main() -> anyhow::Result<()> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "Voron_Design_Cube_v7.stl".to_string());
    let profile_path = std::env::args().nth(2);
    let corner_x: f64 = std::env::args().nth(3).map_or(85.0, |s| s.parse().unwrap());
    let corner_y: f64 = std::env::args().nth(4).map_or(85.0, |s| s.parse().unwrap());
    let radius: f64 = std::env::args().nth(5).map_or(20.0, |s| s.parse().unwrap());
    let order_lo: f64 = std::env::args().nth(6).map_or(9.0, |s| s.parse().unwrap());
    let order_hi: f64 = std::env::args().nth(7).map_or(15.5, |s| s.parse().unwrap());

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

    println!(
        "slicing... (corner=({corner_x:.1},{corner_y:.1}) radius={radius:.1} order range [{order_lo:.2},{order_hi:.2}])"
    );
    let layers = slicing::slice_mesh(&mesh, &config)?;
    println!("done, {} layers", layers.len());

    let near_corner = |p: DVec3| -> bool {
        let dx = p.x - corner_x;
        let dy = p.y - corner_y;
        (dx * dx + dy * dy).sqrt() <= radius
    };

    for layer in &layers {
        if layer.order < order_lo || layer.order > order_hi {
            continue;
        }

        let wall0_loops: Vec<&manifold_core::slicing::WallLoop> =
            layer.loops.iter().filter(|w| w.wall_index == 0).collect();
        let wall0_near: Vec<&&manifold_core::slicing::WallLoop> = wall0_loops
            .iter()
            .filter(|w| w.points.iter().any(|&p| near_corner(p)))
            .collect();

        let infill_near = layer
            .infill_boundary
            .iter()
            .filter(|loop_pts| loop_pts.iter().any(|&p| near_corner(p)))
            .count();

        println!(
            "order={:8.4}  wall0_loops_total={:3}  wall0_loops_near_corner={:3}  infill_loops_total={:3}  infill_loops_near_corner={:3}",
            layer.order,
            wall0_loops.len(),
            wall0_near.len(),
            layer.infill_boundary.len(),
            infill_near,
        );
        for w in &wall0_near {
            println!(
                "    wall0 loop: open={:5} points={:5} first={:?} last={:?}",
                w.is_open,
                w.points.len(),
                w.points.first(),
                w.points.last(),
            );
        }
    }

    Ok(())
}
