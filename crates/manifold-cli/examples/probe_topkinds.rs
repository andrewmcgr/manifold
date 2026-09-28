//! Top-band kind census: for every planned path that carries extrusion at
//! order >= 2.8, report kind / extruded segs / E / point z-range / x-y bbox,
//! grouped per layer order. Also dumps wall-0 `top_surface` tag counts for
//! every layer (true-count / point-count per loop).
//!
//! Usage: probe_topkinds <mesh.stl> [profile.json] [--empty-slope]
//!
//! Temporary diagnostic for the Test6 top-surface defect.

use manifold_core::ids::{ObjectId, ToolId};
use manifold_core::object::{center_on_bed, Object};

#[derive(serde::Deserialize)]
struct Profile {
    machine: manifold_core::machine::Machine,
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

    let mut on_progress = |_f: f64| {};
    let paths = manifold_core::toolpath::plan_with_progress(
        &layers,
        &objects,
        &profile.machine.tools,
        &profile.config,
        Some(&profile.machine),
        &slope_profile,
        &mut on_progress,
    )?;

    // ---- wall loop census (all wall indices, order >= 4.2) -----------------
    println!("== wall loops (all indices, order >= 4.2) ==");
    for (i, layer) in layers.iter().enumerate() {
        if layer.order < 4.2 {
            continue;
        }
        for (j, wall) in layer.loops.iter().enumerate() {
            let n = wall.points.len();
            let tagged = wall.top_surface.iter().filter(|&&t| t).count();
            let xmin = wall
                .points
                .iter()
                .map(|p| p.x)
                .fold(f64::INFINITY, f64::min);
            let xmax = wall
                .points
                .iter()
                .map(|p| p.x)
                .fold(f64::NEG_INFINITY, f64::max);
            let ymin = wall
                .points
                .iter()
                .map(|p| p.y)
                .fold(f64::INFINITY, f64::min);
            let ymax = wall
                .points
                .iter()
                .map(|p| p.y)
                .fold(f64::NEG_INFINITY, f64::max);
            let zmin = wall
                .points
                .iter()
                .map(|p| p.z)
                .fold(f64::INFINITY, f64::min);
            let zmax = wall
                .points
                .iter()
                .map(|p| p.z)
                .fold(f64::NEG_INFINITY, f64::max);
            let lo = layer.order;
            println!(
                "L{i} w{}[{j}] order={lo:.4} n={n} top_tagged={tagged} x=[{xmin:.2}..{xmax:.2}] y=[{ymin:.2}..{ymax:.2}] z=[{zmin:.3}..{zmax:.3}]",
                wall.wall_index
            );
        }
    }

    // ---- kind census over top-band paths ----------------------------------
    println!("== top-band kind census (order >= 2.8) ==");
    type Bucket = Vec<(String, usize, f64, f64, f64, f64, f64, f64, f64)>;
    let mut buckets: std::collections::BTreeMap<i32, Bucket> = std::collections::BTreeMap::new();
    for path in &paths {
        let segs: Vec<&manifold_core::toolpath::Segment> = path
            .segments
            .iter()
            .filter(|s| s.extrusion_length > 0.0)
            .collect();
        if segs.is_empty() {
            continue;
        }
        let omin = segs.iter().map(|s| s.order).fold(f64::INFINITY, f64::min);
        if omin < 2.8 {
            continue;
        }
        // Group the path's extruded segments by kind.
        let mut by_kind: std::collections::BTreeMap<&str, (usize, f64)> =
            std::collections::BTreeMap::new();
        for s in &segs {
            let k = match s.kind {
                manifold_core::toolpath::MoveKind::WallOuter => "WallOuter",
                manifold_core::toolpath::MoveKind::WallInner => "WallInner",
                manifold_core::toolpath::MoveKind::Infill => "Infill",
                manifold_core::toolpath::MoveKind::Bridge => "Bridge",
                manifold_core::toolpath::MoveKind::Overhang => "Overhang",
                manifold_core::toolpath::MoveKind::TopSurface => "TopSurface",
                manifold_core::toolpath::MoveKind::Travel => "Travel",
                manifold_core::toolpath::MoveKind::Wipe => "Wipe",
                manifold_core::toolpath::MoveKind::DebugExcluded => "DebugExcluded",
            };
            by_kind.entry(k).or_default().0 += 1;
            by_kind.entry(k).or_default().1 += s.extrusion_length;
        }
        let zmin = path
            .points
            .iter()
            .map(|p| p.z)
            .fold(f64::INFINITY, f64::min);
        let zmax = path
            .points
            .iter()
            .map(|p| p.z)
            .fold(f64::NEG_INFINITY, f64::max);
        let xmin = path
            .points
            .iter()
            .map(|p| p.x)
            .fold(f64::INFINITY, f64::min);
        let xmax = path
            .points
            .iter()
            .map(|p| p.x)
            .fold(f64::NEG_INFINITY, f64::max);
        let ymin = path
            .points
            .iter()
            .map(|p| p.y)
            .fold(f64::INFINITY, f64::min);
        let ymax = path
            .points
            .iter()
            .map(|p| p.y)
            .fold(f64::NEG_INFINITY, f64::max);
        let key = (omin * 20.0).round() as i32;
        for (k, (n, e)) in &by_kind {
            buckets.entry(key).or_default().push((
                k.to_string(),
                *n,
                *e,
                xmin,
                xmax,
                ymin,
                ymax,
                zmin,
                zmax,
            ));
        }
    }
    for (key, rows) in &buckets {
        let order = *key as f64 / 20.0;
        println!("-- order ~{order:.2} --");
        let mut rows: Vec<_> = rows.iter().collect();
        rows.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        for (k, n, e, xmin, xmax, ymin, ymax, zmin, zmax) in rows {
            println!(
                "  {k:<12} paths-rows={n} e={e:9.3}  x=[{xmin:.2}..{xmax:.2}] y=[{ymin:.2}..{ymax:.2}] z=[{zmin:.3}..{zmax:.3}]"
            );
        }
    }
    // ---- per-kind z-profile over 2.8..5.0 ---------------------------------
    let mut zprof: std::collections::BTreeMap<(i32, &str), (usize, f64)> =
        std::collections::BTreeMap::new();
    for path in &paths {
        for (p, s) in path.points.iter().zip(&path.segments) {
            if s.extrusion_length <= 0.0 {
                continue;
            }
            let z = p.z;
            if !(2.8..=5.01).contains(&z) {
                continue;
            }
            let k: &str = match s.kind {
                manifold_core::toolpath::MoveKind::WallOuter => "WallOuter",
                manifold_core::toolpath::MoveKind::WallInner => "WallInner",
                manifold_core::toolpath::MoveKind::Infill => "Infill",
                manifold_core::toolpath::MoveKind::Bridge => "Bridge",
                manifold_core::toolpath::MoveKind::Overhang => "Overhang",
                manifold_core::toolpath::MoveKind::TopSurface => "TopSurface",
                manifold_core::toolpath::MoveKind::Travel => "Travel",
                manifold_core::toolpath::MoveKind::Wipe => "Wipe",
                manifold_core::toolpath::MoveKind::DebugExcluded => "DebugExcluded",
            };
            let zb = (z * 20.0).round() as i32;
            let e = zprof.entry((zb, k)).or_default();
            e.0 += 1;
            e.1 += s.extrusion_length;
        }
    }
    println!("== per-kind z-profile (0.05 bands, z 2.8..5.0) ==");
    for ((zb, k), (n, e)) in &zprof {
        if *e > 0.05 {
            println!("  z~{:.2} {k:<12} segs={n:6} e={e:9.3}", *zb as f64 / 20.0);
        }
    }
    Ok(())
}
