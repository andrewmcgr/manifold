//! Scratch probe: find extruding paths that contain vertical segments
//! (|dz| > 0.5mm between consecutive points) and dump their full point
//! lists, to see what kind of geometry the planner produced.
//!
//! ```sh
//! CARGO_TARGET_DIR=target/release cargo run --release -p manifold-cli --example probe_vertical_paths -- \
//!     /path/to/mesh.stl /path/to/profile.json [max_paths]
//! ```

use manifold_core::ids::{ObjectId, ToolId};
use manifold_core::machine::Machine;
use manifold_core::object::{center_on_bed, Object};
use manifold_core::toolpath::MoveKind;
use manifold_core::{slicing, stl, toolpath};

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
    let max_paths: usize = std::env::args()
        .nth(3)
        .map(|s| s.parse().expect("max_paths must be a number"))
        .unwrap_or(3);

    let mesh = stl::load_stl(std::fs::File::open(&mesh_path)?)?;
    let json = std::fs::read_to_string(&profile_path)?;
    let profile: Profile = serde_json::from_str(&json)?;

    let object = Object::new(ObjectId(0), mesh, ToolId(0));
    let mut objects = vec![object];
    center_on_bed(&mut objects, &profile.machine.build_volume);
    let object = &objects[0];

    let layers = slicing::slice_object(object, &profile.config)?;
    let paths = toolpath::plan(
        &layers,
        std::slice::from_ref(object),
        &profile.machine.tools,
        &profile.config,
    )?;
    println!("planned paths: {}", paths.len());
    // Fingerprint: sum of first-point coords across all paths.
    let fp: f64 = paths
        .iter()
        .filter(|p| !p.points.is_empty())
        .map(|p| p.points[0].x + p.points[0].y + p.points[0].z)
        .sum();
    println!("fingerprint first-points: {fp:.6}");

    let mut shown = 0usize;
    for (pi, p) in paths.iter().enumerate() {
        let n_pts = p.points.len();
        if n_pts == 0 {
            continue;
        }
        let mut vert = 0usize;
        let mut max_dz = 0.0f64;
        for i in 0..p.segments.len() {
            let s = &p.segments[i];
            let j = (i + 1) % n_pts;
            let dz = (p.points[j].z - p.points[i].z).abs();
            if dz > max_dz {
                max_dz = dz;
            }
            if dz > 0.5 && s.kind != MoveKind::Travel {
                vert += 1;
            }
        }
        if vert == 0 {
            continue;
        }
        shown += 1;
        let kinds: std::collections::HashSet<String> =
            p.segments.iter().map(|s| format!("{:?}", s.kind)).collect();
        let mut orders: Vec<f64> = p.segments.iter().map(|s| s.order).collect();
        orders.sort_by(|a, b| a.total_cmp(b));
        orders.dedup_by(|a, b| a.total_cmp(b) == std::cmp::Ordering::Equal);
        let zmin = p.points.iter().map(|q| q.z).fold(f64::INFINITY, f64::min);
        let zmax = p
            .points
            .iter()
            .map(|q| q.z)
            .fold(f64::NEG_INFINITY, f64::max);
        let xmin = p.points.iter().map(|q| q.x).fold(f64::INFINITY, f64::min);
        let xmax = p
            .points
            .iter()
            .map(|q| q.x)
            .fold(f64::NEG_INFINITY, f64::max);
        let ymin = p.points.iter().map(|q| q.y).fold(f64::INFINITY, f64::min);
        let ymax = p
            .points
            .iter()
            .map(|q| q.y)
            .fold(f64::NEG_INFINITY, f64::max);
        println!(
            "path {pi}: n={} kinds={kinds:?} orders={orders:?} vert_segs={vert} max_dz={max_dz:.2} z[{zmin:.2},{zmax:.2}] x[{xmin:.1},{xmax:.1}] y[{ymin:.1},{ymax:.1}]",
            n_pts
        );
        if shown <= max_paths {
            for i in 0..p.segments.len() {
                let s = &p.segments[i];
                let j = (i + 1) % n_pts;
                println!(
                    "  seg{:4} {:?} order {:8.3} ({:.2},{:.2},{:.3}) -> ({:.2},{:.2},{:.3})",
                    i,
                    s.kind,
                    s.order,
                    p.points[i].x,
                    p.points[i].y,
                    p.points[i].z,
                    p.points[j].x,
                    p.points[j].y,
                    p.points[j].z
                );
            }
            if shown == max_paths {
                break;
            }
        }
    }
    let mut total_vert = 0usize;
    for p in &paths {
        let n_pts = p.points.len();
        if n_pts == 0 {
            continue;
        }
        for i in 0..p.segments.len() {
            let s = &p.segments[i];
            if s.kind == MoveKind::Travel || s.kind == MoveKind::Wipe {
                continue;
            }
            let j = (i + 1) % n_pts;
            if (p.points[j].z - p.points[i].z).abs() > 0.5 {
                total_vert += 1;
            }
        }
    }
    println!("total vertical extrusion segments: {total_vert}");
    Ok(())
}
