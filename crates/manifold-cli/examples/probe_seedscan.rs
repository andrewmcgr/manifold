//! Targeted seed-margin probe for the Test6 top-face solid-fill loss.
//!
//! Usage: probe_seedscan <stl> <profile.json>
//!
//! Prints:
//!   Part 1: vertical scanlines (sdf, order, seed_proximity kind+d, margins)
//!   Part 2: order values on the z=5.0 top face (left slab)
//!   Part 3: seed margins on each top layer's own plane (z = layer.order)
//!   Part 4: found_ib loop bboxes on L22-L24
//!
//! The seed-margin threshold is `top_layers * layer_height + 1e-3` = 0.601
//! for the machine profile (top_layers=3, layer_height=0.2).

use manifold_core::ids::{ObjectId, ToolId};
use manifold_core::machine::Machine;
use manifold_core::object::{center_on_bed, Object};
use manifold_fidget::order::SeedKind;
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
    let config = &profile.config;
    let machine = &profile.machine;

    let layers = manifold_core::slicing::slice_object_with_progress(
        object,
        config,
        &machine.slope_profile(),
        &mut |_| {},
    )?;

    let layer_height = config.layer_height;
    let thresh = config.top_layers as f64 * layer_height + 1e-3;
    let bottom_thresh = config.bottom_layers as f64 * layer_height + 1e-3;
    println!(
        "layer_height={:.3} top_threshold={:.4} bottom_threshold={:.4}",
        layer_height, thresh, bottom_thresh
    );

    // ---- Part 1: vertical scanlines inside the left slab. ----
    let z_scan_lo: f64 = std::env::var("ZZ_LO")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3.6);
    let z_scan_hi: f64 = std::env::var("ZZ_HI")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5.25);
    let face_z: f64 = std::env::var("ZZ_FACE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5.0);
    let slab_box: (f64, f64, f64, f64) = match std::env::var("ZZ_BOX").ok() {
        Some(b) => {
            let v: Vec<f64> = b.split(',').map(|s| s.parse().unwrap()).collect();
            (v[0], v[1], v[2], v[3])
        }
        None => (166.5, 174.5, 166.5, 183.5),
    };
    let scan_pts: Vec<(f64, f64)> = std::env::var("ZZ_SCAN")
        .ok()
        .map(|b| {
            let v: Vec<f64> = b.split(',').map(|s| s.parse().unwrap()).collect();
            (0..v.len()).step_by(2).map(|i| (v[i], v[i + 1])).collect()
        })
        .unwrap_or_else(|| vec![(168.0, 175.0), (172.0, 172.0)]);
    println!("\n--- Part 1: vertical scanlines (x,y world) ---");
    for (sx, sy) in scan_pts {
        println!("line ({sx}, {sy}):");
        let mut z = z_scan_lo;
        while z <= z_scan_hi {
            let p = glam::DVec3::new(sx, sy, z);
            let l0 = layers.last().unwrap();
            let sdf = l0
                .mesh_sdf
                .as_ref()
                .map(|sdf| sdf.sample(p).value)
                .unwrap_or(f64::NAN);
            let o = l0.order_field.order(p);
            match l0.order_field.seed_proximity(p) {
                Some((kind, d)) => {
                    let kn = match kind {
                        SeedKind::Bed => "bed",
                        SeedKind::Patch => "patch",
                    };
                    println!(
                        "  z={z:5.2} sdf={sdf:8.4} order={o:8.4} prox={kn:5} d={d:7.4} top-margin={:9.4} bottom-margin={:9.4}",
                        thresh - d,
                        bottom_thresh - d
                    );
                }
                None => println!("  z={z:5.2} sdf={sdf:8.4} order={o:8.4} prox=None"),
            }
            z += 0.05;
        }
    }

    // ---- Part 2: order on the z=5.0 top face (left slab only). ----
    println!("\n--- Part 2: top face z=5.0 (left slab, sdf in [-0.05,0.05]) ---");
    {
        let l0 = layers.last().unwrap();
        let mut n = 0usize;
        let (mut omin, mut omax) = (f64::INFINITY, f64::NEG_INFINITY);
        for xi in 0..17 {
            for yi in 0..18 {
                let p = glam::DVec3::new(
                    slab_box.0 + xi as f64 * 0.5,
                    slab_box.1 + yi as f64 * 0.5,
                    face_z,
                );
                let s = l0
                    .mesh_sdf
                    .as_ref()
                    .map(|sdf| sdf.sample(p).value)
                    .unwrap_or(f64::NAN);
                if s.is_finite() && (-0.05..=0.05).contains(&s) {
                    let o = l0.order_field.order(p);
                    n += 1;
                    omin = omin.min(o);
                    omax = omax.max(o);
                }
            }
        }
        println!("face points {n}; order in [{omin:.4}, {omax:.4}]");
    }

    // ---- Part 3: seed margins on each top layer's own plane. ----
    println!("\n--- Part 3: seed margins at z = layer.order ---");
    for (i, l) in layers.iter().enumerate() {
        if i < 13 {
            continue;
        }
        for (label, x0, x1, y0, y1) in [
            ("slab", 166.5, 174.5, 166.5, 183.5),
            ("groove", 176.5, 184.5, 166.5, 183.9),
        ] {
            let mut n = 0usize;
            let (mut pmin, mut pmax) = (f64::INFINITY, f64::NEG_INFINITY);
            let mut npos = 0usize;
            let mut nbed = 0usize;
            let mut npatch = 0usize;
            let mut nnone = 0usize;
            for xi in 0..17 {
                for yi in 0..18 {
                    let p = glam::DVec3::new(
                        x0 + xi as f64 * ((x1 - x0) / 16.0),
                        y0 + yi as f64 * ((y1 - y0) / 17.0),
                        l.order,
                    );
                    match l.order_field.seed_proximity(p) {
                        Some((kind, d)) => {
                            n += 1;
                            let m = thresh - d;
                            if m > 0.0 {
                                npos += 1;
                            }
                            match kind {
                                SeedKind::Bed => nbed += 1,
                                SeedKind::Patch => npatch += 1,
                            }
                            pmin = pmin.min(m);
                            pmax = pmax.max(m);
                        }
                        None => nnone += 1,
                    }
                }
            }
            if n > 0 || nnone > 0 {
                let (pmn, pmx) = (pmin.min(9.0), pmax.max(-9.0));
                println!(
                    "L{:3} order={:7.4} {:6}: pts={:3} none={} bed={} patch={} margin in [{:+9.4},{:+9.4}] positive={}",
                    i,
                    l.order,
                    label,
                    n,
                    nnone,
                    nbed,
                    npatch,
                    pmn,
                    pmx,
                    npos
                );
            }
        }
    }

    // ---- Part 4: found_ib loop bboxes on L22-L24. ----
    println!("\n--- Part 4: found_ib loops (L22-L24) ---");
    for (i, l) in layers.iter().enumerate() {
        if !(22..=24).contains(&i) {
            continue;
        }
        for (li, loop_) in l.infill_boundary.iter().enumerate() {
            let mut xmin = f64::INFINITY;
            let mut xmax = f64::NEG_INFINITY;
            let mut ymin = f64::INFINITY;
            let mut ymax = f64::NEG_INFINITY;
            for pt in loop_ {
                xmin = xmin.min(pt.x);
                xmax = xmax.max(pt.x);
                ymin = ymin.min(pt.y);
                ymax = ymax.max(pt.y);
            }
            println!(
                "L{:3} loop {:2}: n={:4} x [{:.2}..{:.2}] y [{:.2}..{:.2}]",
                i,
                li,
                loop_.len(),
                xmin,
                xmax,
                ymin,
                ymax
            );
        }
    }

    // ---- Part 5: patch component map just above the face. ----
    // At z=5.10 (0.1 above the face) every interior point of the left slab
    // reports prox=patch with d = |order(p) - seed_value|; where d==0 the
    // implied seed_value equals order(p). Raster the slab and print the
    // implied seed value per column to expose how many patch components
    // tile the face and what value each carries.
    println!("\n--- Part 5: implied patch seed value raster (z=5.10) ---");
    {
        let l0 = layers.last().unwrap();
        let zf: f64 = 5.10;
        for yi in 0..29 {
            let mut row = String::new();
            let y = 161.0 + yi as f64 * 0.5;
            for xi in 0..29 {
                let x = 161.0 + xi as f64 * 0.5;
                let p = glam::DVec3::new(x, y, zf);
                let s = l0
                    .mesh_sdf
                    .as_ref()
                    .map(|sdf| sdf.sample(p).value)
                    .unwrap_or(f64::NAN);
                let v = match l0.order_field.seed_proximity(p) {
                    Some((kind, d)) => {
                        let o = l0.order_field.order(p);
                        let val = o + d; // seed_value >= order above the face
                        let m = match kind {
                            SeedKind::Bed => 'b',
                            SeedKind::Patch => 'p',
                        };
                        format!("{val:4.2}{m}")
                    }
                    _ => ".      ".to_string(),
                };
                let _ = s;
                row.push_str(&v);
                row.push(' ');
            }
            println!("y={y:6.1}: {row}");
        }
        // legend: value column is the implied seed_value; check distinct values
        let mut vals: Vec<f64> = Vec::new();
        for yi in 0..29 {
            for xi in 0..29 {
                let p = glam::DVec3::new(161.0 + xi as f64 * 0.5, 161.0 + yi as f64 * 0.5, zf);
                if let Some((_, d)) = l0.order_field.seed_proximity(p) {
                    let v = l0.order_field.order(p) + d;
                    if !vals.iter().any(|w| (w - v).abs() < 0.005) {
                        vals.push(v);
                    }
                }
            }
        }
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("distinct implied seed values: {vals:?}");
    }

    Ok(())
}
