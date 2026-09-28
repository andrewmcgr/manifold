//! Scratch probe: Test5 planned-path kind census for z >= 29.0, plus
//! top-layer ib/sf/wall dump (mirrors probe_test6_topkinds for Test5).
//!
//! Usage: probe_test5_topz <mesh.stl> <profile.json> [--empty-slope]

use manifold_core::ids::{ObjectId, ToolId};
use manifold_core::machine::Machine;
use manifold_core::object::{center_on_bed, Object};
use manifold_core::toolpath::plan_with_progress;
use std::collections::BTreeMap;

#[derive(serde::Deserialize)]
struct Profile {
    machine: Machine,
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

    let empty_slope = std::env::args().any(|a| a == "--empty-slope");
    let slope_profile = if empty_slope {
        manifold_fidget::slope_profile::SlopeProfile::new(Vec::new())
    } else {
        profile.machine.slope_profile()
    };
    let layers = manifold_core::slicing::slice_object_with_progress(
        object,
        &profile.config,
        &slope_profile,
        &mut |_| {},
    )?;
    let n_layers = layers.len();
    println!("layers={n_layers}");

    println!("\n--- top 6 layers ---");
    for i in (n_layers.saturating_sub(6)..n_layers).rev() {
        let l = &layers[i];
        print!(
            "L{:3} order={:8.4} ib={} sf={}",
            l.index,
            l.order,
            l.infill_boundary.len(),
            l.solid_fill_boundary.len()
        );
        for w in l.loops.iter().filter(|w| w.wall_index == 0).take(6) {
            let z0 = w.points.iter().map(|p| p.z).fold(f64::INFINITY, f64::min);
            let z1 = w
                .points
                .iter()
                .map(|p| p.z)
                .fold(f64::NEG_INFINITY, f64::max);
            print!(" [w0 n={} z {z0:.2}..{z1:.2}]", w.points.len());
        }
        println!();
    }

    let paths = plan_with_progress(
        &layers,
        &objects,
        &profile.machine.tools,
        &profile.config,
        Some(&profile.machine),
        &slope_profile,
        &mut |_| {},
    )?;

    let mut agg: BTreeMap<(String, u32), (usize, f64)> = BTreeMap::new();
    let mut n_above = 0usize;
    for p in &paths {
        for i in 0..p.segments.len() {
            let s = &p.segments[i];
            if s.extrusion_length <= 0.0005 {
                continue;
            }
            let dest = if p.segments.len() == p.points.len() {
                (i + 1) % p.points.len()
            } else {
                (i + 1).min(p.points.len() - 1)
            };
            let a = p.points[i];
            let b = p.points[dest];
            let zm = (a.z + b.z) * 0.5;
            if zm < 29.0 {
                continue;
            }
            if zm > 30.0 {
                n_above += 1;
            }
            let zb = (zm * 10.0).round() as u32;
            let k = format!("{:?}", s.kind);
            let e = agg.entry((k, zb)).or_insert((0, 0.0));
            e.0 += 1;
            e.1 += s.extrusion_length;
        }
    }
    println!("\n--- kind x z-band, z >= 29.0 (segs above z=30.0: {n_above}) ---");
    for ((k, zb), (n, e)) in &agg {
        println!("z={:6.1} {k:12} {:5} {:8.3}", *zb as f64 / 10.0, n, e);
    }
    Ok(())
}
