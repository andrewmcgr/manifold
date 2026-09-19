//! Scratch instrumentation probe (Task 1 of the first-layer-infill-boundary-dropout
//! investigation): dumps per-loop wall_index/island/point-count/bbox for layers 0-3
//! of the real repro, to understand exactly how the object's cross-section is
//! partitioned into islands at each wall depth.

use manifold_core::ids::{ObjectId, ToolId};
use manifold_core::machine::Machine;
use manifold_core::object::{center_on_bed, Object};
use manifold_core::{slicing, stl, SlicerConfig};

#[derive(serde::Deserialize)]
struct Profile {
    machine: Machine,
    config: SlicerConfig,
}

fn main() -> anyhow::Result<()> {
    let mesh_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "TestObj1.stl".to_string());
    let profile_path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "/Users/amcgregor/3D/profile.json".to_string());
    let max_layer: usize = std::env::args()
        .nth(3)
        .map(|s| s.parse().unwrap())
        .unwrap_or(3);

    let mesh = stl::load_stl(std::fs::File::open(&mesh_path)?)?;
    let json = std::fs::read_to_string(&profile_path)?;
    let profile: Profile = serde_json::from_str(&json)?;

    let object = Object::new(ObjectId(0), mesh, ToolId(0));
    let mut objects = vec![object];
    center_on_bed(&mut objects, &profile.machine.build_volume);

    let layers = slicing::slice_object(&objects[0], &profile.config)?;
    println!("Sliced into {} layers", layers.len());

    for layer in layers.iter().take(max_layer + 1) {
        println!("\n=== layer {} order {:.3} ===", layer.index, layer.order);
        for l in &layer.loops {
            let (mut xmin, mut xmax) = (f64::INFINITY, f64::NEG_INFINITY);
            let (mut ymin, mut ymax) = (f64::INFINITY, f64::NEG_INFINITY);
            for p in &l.points {
                xmin = xmin.min(p.x);
                xmax = xmax.max(p.x);
                ymin = ymin.min(p.y);
                ymax = ymax.max(p.y);
            }
            println!(
                "  wall_index={} island={} is_open={} pts={} bbox_x=[{:.2},{:.2}] bbox_y=[{:.2},{:.2}]",
                l.wall_index, l.island, l.is_open, l.points.len(), xmin, xmax, ymin, ymax
            );
        }
        println!("  infill_boundary polys: {}", layer.infill_boundary.len());
        for (i, poly) in layer.infill_boundary.iter().enumerate() {
            let (mut xmin, mut xmax) = (f64::INFINITY, f64::NEG_INFINITY);
            for p in poly {
                xmin = xmin.min(p.x);
                xmax = xmax.max(p.x);
            }
            println!(
                "    poly[{i}]: {} pts, bbox_x=[{:.2},{:.2}]",
                poly.len(),
                xmin,
                xmax
            );
        }
    }

    Ok(())
}
