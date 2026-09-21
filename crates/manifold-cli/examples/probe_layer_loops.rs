//! Scratch probe: dump per-loop / per-infill-boundary geometry (wall index,
//! island, point count, bbox, area) for the first N layers, to inspect how
//! the bore at the x-midline shows up in the layer topology.
//!
//! ```sh
//! CARGO_TARGET_DIR=target/release cargo run --release -p manifold-cli --example probe_layer_loops -- \
//!     /path/to/mesh.stl /path/to/profile.json [max_layer_index]
//! ```

use manifold_core::ids::{ObjectId, ToolId};
use manifold_core::machine::Machine;
use manifold_core::object::{center_on_bed, Object};
use manifold_core::{slicing, stl, SlicerConfig};

#[derive(serde::Deserialize)]
struct Profile {
    machine: Machine,
    config: SlicerConfig,
}

fn loop_stats(pts: &[glam::DVec3]) -> (f64, f64, f64, f64, f64, f64, f64, f64) {
    let mut minx = f64::INFINITY;
    let mut miny = f64::INFINITY;
    let mut minz = f64::INFINITY;
    let mut maxx = f64::NEG_INFINITY;
    let mut maxy = f64::NEG_INFINITY;
    let mut maxz = f64::NEG_INFINITY;
    for p in pts {
        minx = minx.min(p.x);
        miny = miny.min(p.y);
        minz = minz.min(p.z);
        maxx = maxx.max(p.x);
        maxy = maxy.max(p.y);
        maxz = maxz.max(p.z);
    }
    // 2D area in XY (shoelace).
    let mut area = 0.0;
    for i in 0..pts.len() {
        let a = &pts[i];
        let b = &pts[(i + 1) % pts.len()];
        area += a.x * b.y - b.x * a.y;
    }
    (
        minx,
        miny,
        minz,
        maxx,
        maxy,
        maxz,
        area.abs() * 0.5,
        pts.len() as f64,
    )
}

fn main() -> anyhow::Result<()> {
    let mesh_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "TestObj1.stl".to_string());
    let profile_path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "examples/profile.json".to_string());
    let max_layer_index: usize = std::env::args()
        .nth(3)
        .map(|s| s.parse().expect("max_layer_index must be a number"))
        .unwrap_or(4);

    let mesh = stl::load_stl(std::fs::File::open(&mesh_path)?)?;
    let json = std::fs::read_to_string(&profile_path)?;
    let profile: Profile = serde_json::from_str(&json)?;

    let object = Object::new(ObjectId(0), mesh, ToolId(0));
    let mut objects = vec![object];
    center_on_bed(&mut objects, &profile.machine.build_volume);
    let object = &objects[0];

    let layers = slicing::slice_object(object, &profile.config)?;
    println!("Sliced into {} layers\n", layers.len());

    for layer in layers.iter().take(max_layer_index + 1) {
        println!("=== layer {:3} order {:7.3} ===", layer.index, layer.order);
        for (i, loop_) in layer.loops.iter().enumerate() {
            let (minx, miny, minz, maxx, maxy, maxz, area, npts) = loop_stats(&loop_.points);
            println!(
                "  loop{:3} w{} island{} n={:5} area={:8.2} bbox x[{:.1},{:.1}] y[{:.1},{:.1}] z[{:.2},{:.2}]",
                i, loop_.wall_index, loop_.island, npts, area, minx, maxx, miny, maxy, minz, maxz
            );
        }
        for (i, poly) in layer.infill_boundary.iter().enumerate() {
            let (minx, miny, _minz, maxx, maxy, _maxz, area, npts) = loop_stats(poly);
            println!(
                "  infill_boundary{:3} n={:5} area={:8.2} bbox x[{:.1},{:.1}] y[{:.1},{:.1}]",
                i, npts, area, minx, maxx, miny, maxy
            );
        }
        for (i, poly) in layer.solid_fill_boundary.iter().enumerate() {
            let (minx, miny, _minz, maxx, maxy, _maxz, area, npts) = loop_stats(poly);
            println!(
                "  solid_fill_boundary{:3} n={:5} area={:8.2} bbox x[{:.1},{:.1}] y[{:.1},{:.1}]",
                i, npts, area, minx, maxx, miny, maxy
            );
        }
    }
    Ok(())
}
