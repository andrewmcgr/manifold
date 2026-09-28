//! Scratch probe: Test6 top-surface defect, Phase 1 mapping.
//!
//! Slice Test6.stl with the machine profile and, for the top layers:
//!  Part B: kind x z-band census of planned paths
//!  Part C: top-face (z=5.0) coverage per layer, by segment kind
//!  Part D: seed-eligibility (order + seed_proximity + sdf) maps on the
//!         top face and the groove floor
//!  Part E: planned material in the groove air (z >= 3.4) per layer/kind
//!
//! ```sh
//! CARGO_TARGET_DIR=target/build cargo run -p manifold-cli --example probe_topmask -- \
//!     Test6.stl examples/profile.json
//! ```

use manifold_core::ids::{ObjectId, ToolId};
use manifold_core::machine::Machine;
use manifold_core::object::{center_on_bed, Object};
use manifold_fidget::ScalarField;

#[derive(serde::Deserialize)]
struct Profile {
    machine: Machine,
    config: manifold_core::SlicerConfig,
}

fn kind_id(k: manifold_core::toolpath::MoveKind) -> u32 {
    match k {
        manifold_core::toolpath::MoveKind::WallOuter => 0,
        manifold_core::toolpath::MoveKind::WallInner => 1,
        manifold_core::toolpath::MoveKind::Infill => 2,
        manifold_core::toolpath::MoveKind::Bridge => 3,
        manifold_core::toolpath::MoveKind::Overhang => 4,
        manifold_core::toolpath::MoveKind::TopSurface => 5,
        manifold_core::toolpath::MoveKind::Travel => 6,
        manifold_core::toolpath::MoveKind::Wipe => 7,
        manifold_core::toolpath::MoveKind::DebugExcluded => 8,
    }
}

/// Attribute each path to the layer whose order its first point's z is closest
/// to (same convention as probe_layer_dump), so adjacent cap layers
/// (0.0075 apart) do not double-count each other's paths.
fn by_layer<'a>(
    planned: &'a [manifold_core::toolpath::Path],
    layers: &[manifold_core::slicing::Layer],
) -> Vec<Vec<&'a manifold_core::toolpath::Path>> {
    let mut out: Vec<Vec<&manifold_core::toolpath::Path>> = vec![Vec::new(); layers.len()];
    for p in planned {
        if p.points.is_empty() {
            continue;
        }
        let z = p.points[0].z;
        let li = layers
            .iter()
            .enumerate()
            .min_by(|a, b| {
                (a.1.order - z)
                    .abs()
                    .partial_cmp(&(b.1.order - z).abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0);
        out[li].push(p);
    }
    out
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
    let planned = manifold_core::toolpath::plan_with_progress(
        &layers,
        &objects,
        &machine.tools,
        config,
        Some(machine),
        &machine.slope_profile(),
        &mut |_| {},
    )?;

    let loop_area = |loops: &Vec<Vec<glam::DVec3>>| -> f64 {
        loops
            .iter()
            .map(|lp| {
                let n = lp.len();
                if n < 3 {
                    return 0.0;
                }
                let mut a = 0.0f64;
                for i in 0..n {
                    let p = lp[i];
                    let q = lp[(i + 1) % n];
                    a += p.x * q.y - q.x * p.y;
                }
                a.abs() / 2.0
            })
            .sum()
    };
    let paths_by_layer = by_layer(&planned, &layers);

    // Part A: top-layer laydump.
    println!("--- Part A: top layers ---");
    for l in layers.iter().rev().take(7) {
        let paths = &paths_by_layer[l.index];
        let (mut segs, mut ext) = (0u64, 0.0f64);
        for p in paths {
            for s in &p.segments {
                segs += 1;
                ext += s.extrusion_length;
            }
        }
        println!(
            "L{:3} order {:6.4} ib={}({:.1}mm2) sf={}({:.1}mm2) paths={} segs={} e={:.2}mm",
            l.index,
            l.order,
            l.infill_boundary.len(),
            loop_area(&l.infill_boundary),
            l.solid_fill_boundary.len(),
            loop_area(&l.solid_fill_boundary),
            paths.len(),
            segs,
            ext
        );
    }

    // Part B: kind x z-band census for the top layers.
    println!("\n--- Part B: kind census, layers with order >= 3.9 ---");
    let kind_names = [
        "WallOuter",
        "WallInner",
        "Infill",
        "Bridge",
        "Overhang",
        "TopSurface",
        "Travel",
        "Wipe",
        "DebugExcluded",
        "?",
    ];
    for l in layers.iter().rev() {
        if l.order < 3.9 {
            break;
        }
        let mut census: std::collections::HashMap<(u32, u32), (u64, f64)> =
            std::collections::HashMap::new();
        for p in &paths_by_layer[l.index] {
            for (i, s) in p.segments.iter().enumerate() {
                let start = match p.points.get(i) {
                    Some(pt) => *pt,
                    None => continue,
                };
                let zb = (start.z * 10.0).round() as u32;
                let k = kind_id(s.kind);
                let e = census.entry((k, zb)).or_insert((0u64, 0.0f64));
                e.0 += 1;
                e.1 += s.extrusion_length;
            }
        }
        let mut any = false;
        println!("L{:3} (order {:6.4}):", l.index, l.order);
        let mut entries: Vec<(u32, u32, u64, f64)> = census
            .iter()
            .map(|(&kk, &vv)| (kk.0, kk.1, vv.0, vv.1))
            .collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
        for (k, zb, n, e) in &entries {
            if *k >= 6 {
                continue; // travel/wipe noise
            }
            println!(
                "    {:<12} z={:5.1}: {:5} segs {:8.2}mm E",
                kind_names[*k as usize],
                (*zb) as f64 / 10.0,
                n,
                e
            );
            any = true;
        }
        if !any {
            println!("    (no extruded segments)");
        }
    }

    // Part C: top-face (z=5.0) coverage per top layer, by kind.
    println!("\n--- Part C: top-face coverage at z=5.0 (0.5mm grid, seg start within 0.35mm, z>=4.8) ---");
    for l in layers.iter().rev().take(3) {
        let paths = &paths_by_layer[l.index];
        let mut by_kind: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
        let mut covered = 0u32;
        let mut total = 0u32;
        let mut x = 166.25f64;
        while x <= 183.75 {
            let mut y = 166.25f64;
            while y <= 183.75 {
                let mut found: Vec<u32> = Vec::new();
                for p in paths {
                    for (i, s) in p.segments.iter().enumerate() {
                        let start = match p.points.get(i) {
                            Some(pt) => *pt,
                            None => continue,
                        };
                        if start.z < 4.8 {
                            continue;
                        }
                        let dx = start.x - x;
                        let dy = start.y - y;
                        if dx * dx + dy * dy <= 0.35 * 0.35 {
                            let k = kind_id(s.kind);
                            if k <= 5 && !found.contains(&k) {
                                found.push(k);
                            }
                        }
                    }
                }
                if !found.is_empty() {
                    covered += 1;
                    for k in found {
                        *by_kind.entry(k).or_insert(0) += 1;
                    }
                }
                total += 1;
                y += 0.5;
            }
            x += 0.5;
        }
        let mut ks: Vec<(u32, u32)> = by_kind.into_iter().collect();
        ks.sort();
        println!(
            "L{:3} (order {:6.4}): covered {}/{} grid pts; per-kind: {}",
            l.index,
            l.order,
            covered,
            total,
            ks.iter()
                .map(|(k, n)| format!("{}={}", kind_names[*k as usize], n))
                .collect::<Vec<_>>()
                .join(" ")
        );
    }

    // Part D: seed-eligibility maps.
    println!("\n--- Part D: seed maps (top face z=5.0 and groove floor z=3.0) ---");
    let top_pts: Vec<glam::DVec3> = (0..=18)
        .flat_map(|i| {
            (0..=12).map(move |j| glam::DVec3::new(166.0 + i as f64, 169.0 + j as f64, 5.0))
        })
        .collect();
    let floor_pts: Vec<glam::DVec3> = (0..=6)
        .flat_map(|i| {
            (0..=12).map(move |j| glam::DVec3::new(177.0 + i as f64, 169.0 + j as f64, 3.0))
        })
        .collect();
    for l in layers.iter().rev() {
        if l.order < 4.1 {
            break;
        }
        let mut n_elig = 0usize;
        let mut o_min = f64::INFINITY;
        let mut o_max = f64::NEG_INFINITY;
        let mut s_min = f64::INFINITY;
        let mut s_max = f64::NEG_INFINITY;
        for p in &top_pts {
            let o = l.order_field.order(*p);
            let prox = l.order_field.seed_proximity(*p);
            let s = l
                .mesh_sdf
                .as_ref()
                .map(|sdf| sdf.sample(*p).value)
                .unwrap_or(f64::NAN);
            if prox.is_some() {
                n_elig += 1;
            }
            o_min = o_min.min(o);
            o_max = o_max.max(o);
            s_min = s_min.min(s);
            s_max = s_max.max(s);
        }
        println!(
            "L{:3} (order {:6.4}) top face: eligible {}/{}  order in [{:.4},{:.4}]  sdf in [{:.4},{:.4}]",
            l.index,
            l.order,
            n_elig,
            top_pts.len(),
            o_min,
            o_max,
            s_min,
            s_max
        );
    }
    for l in layers.iter() {
        if (l.order - 2.9).abs() > 0.5 {
            continue;
        }
        let mut n_elig = 0usize;
        let mut o_min = f64::INFINITY;
        let mut o_max = f64::NEG_INFINITY;
        let mut s_min = f64::INFINITY;
        let mut s_max = f64::NEG_INFINITY;
        for p in &floor_pts {
            let o = l.order_field.order(*p);
            let prox = l.order_field.seed_proximity(*p);
            let s = l
                .mesh_sdf
                .as_ref()
                .map(|sdf| sdf.sample(*p).value)
                .unwrap_or(f64::NAN);
            if prox.is_some() {
                n_elig += 1;
            }
            o_min = o_min.min(o);
            o_max = o_max.max(o);
            s_min = s_min.min(s);
            s_max = s_max.max(s);
        }
        println!(
            "L{:3} (order {:6.4}) groove floor: eligible {}/{}  order in [{:.4},{:.4}]  sdf in [{:.4},{:.4}]",
            l.index,
            l.order,
            n_elig,
            floor_pts.len(),
            o_min,
            o_max,
            s_min,
            s_max
        );
    }

    // Part E: planned material in the groove air (z >= 3.4) for the top layers.
    println!(
        "\n--- Part E: groove-air material (x 176.1..185.9, y 166.1..183.9, z>=3.4, E>0.01) ---"
    );
    for l in layers.iter().rev() {
        if l.order < 3.9 {
            break;
        }
        let mut total = 0.0f64;
        let mut by_kind: std::collections::HashMap<u32, (u64, f64)> =
            std::collections::HashMap::new();
        for p in &paths_by_layer[l.index] {
            for (i, s) in p.segments.iter().enumerate() {
                if s.extrusion_length <= 0.01 {
                    continue;
                }
                let start = match p.points.get(i) {
                    Some(pt) => *pt,
                    None => continue,
                };
                let end = p.points.get(i + 1).copied().unwrap_or(start);
                let in_box = |pt: glam::DVec3| -> bool {
                    pt.x >= 176.1 && pt.x <= 185.9 && pt.y >= 166.1 && pt.y <= 183.9
                };
                if !(in_box(start) || in_box(end)) {
                    continue;
                }
                if start.z < 3.4 && end.z < 3.4 {
                    continue;
                }
                total += s.extrusion_length;
                let k = kind_id(s.kind);
                let e = by_kind.entry(k).or_insert((0u64, 0.0f64));
                e.0 += 1;
                e.1 += s.extrusion_length;
            }
        }
        if total <= 0.0 {
            continue;
        }
        let mut ks: Vec<(u32, (u64, f64))> = by_kind.into_iter().collect();
        ks.sort_by_key(|a| a.0);
        println!(
            "L{:3} (order {:6.4}): {:8.2}mm E in groove air; {}",
            l.index,
            l.order,
            total,
            ks.iter()
                .map(|(k, (n, e))| format!("{}={}seg/{:.2}mm", kind_names[*k as usize], n, e))
                .collect::<Vec<_>>()
                .join(" ")
        );
    }
    // Part F: where each top layer's order isosurface physically sits, and
    // what the seed margin reads there (the exact SeedMarginField logic:
    // margin = threshold - dist; eligible iff margin > 0).
    println!("\n--- Part F: isosurface z + seed margin per top layer (1mm grid) ---");
    let top_threshold = config.top_layers as f64 * config.layer_height + 1e-3;
    let bottom_threshold = config.bottom_layers as f64 * config.layer_height + 1e-3;
    println!(
        "top_threshold={:.4} bottom_threshold={:.4} (layer_height={:?} top_layers={} bottom_layers={})",
        top_threshold,
        bottom_threshold,
        config.layer_height,
        config.top_layers,
        config.bottom_layers
    );
    let solve_z = |l: &manifold_core::slicing::Layer, x: f64, y: f64| -> Option<f64> {
        let target = l.order;
        let mut z_prev = 2.0f64;
        let mut f_prev = l.order_field.order(glam::DVec3::new(x, y, z_prev)) - target;
        let mut z = z_prev + 0.05;
        while z <= 6.0 {
            let f = l.order_field.order(glam::DVec3::new(x, y, z)) - target;
            if f_prev.is_finite() && f.is_finite() && f_prev * f <= 0.0 {
                let (mut a, mut b) = (z_prev, z);
                for _ in 0..40 {
                    let m = (a + b) * 0.5;
                    let fm = l.order_field.order(glam::DVec3::new(x, y, m)) - target;
                    if f_prev * fm <= 0.0 {
                        b = m;
                    } else {
                        a = m;
                        f_prev = fm;
                    }
                }
                return Some((a + b) * 0.5);
            }
            z_prev = z;
            f_prev = f;
            z += 0.05;
        }
        None
    };
    for l in layers.iter() {
        if !(14..=16).contains(&l.index) && !(22..=26).contains(&l.index) {
            continue;
        }
        let mut solved = 0usize;
        let mut z_min = f64::INFINITY;
        let mut z_max = f64::NEG_INFINITY;
        let mut sdf_min = f64::INFINITY;
        let mut sdf_max = f64::NEG_INFINITY;
        let mut n_patch = 0usize;
        let mut n_bed = 0usize;
        let mut n_none = 0usize;
        let mut d_min = f64::INFINITY;
        let mut d_max = f64::NEG_INFINITY;
        let mut n_patch_eligible = 0usize;
        let mut n_patch_eligible_solid = 0usize;
        let mut y = 166.0f64;
        while y <= 184.0 {
            let mut x = 166.0f64;
            while x <= 184.0 {
                if let Some(zs) = solve_z(l, x, y) {
                    solved += 1;
                    let p = glam::DVec3::new(x, y, zs);
                    let s = l
                        .mesh_sdf
                        .as_ref()
                        .map(|sdf| sdf.sample(p).value)
                        .unwrap_or(f64::NAN);
                    z_min = z_min.min(zs);
                    z_max = z_max.max(zs);
                    sdf_min = sdf_min.min(s);
                    sdf_max = sdf_max.max(s);
                    match l.order_field.seed_proximity(p) {
                        Some((manifold_fidget::order::SeedKind::Patch, d)) => {
                            n_patch += 1;
                            d_min = d_min.min(d);
                            d_max = d_max.max(d);
                            if d <= top_threshold {
                                n_patch_eligible += 1;
                                if s < 0.0 {
                                    n_patch_eligible_solid += 1;
                                }
                            }
                        }
                        Some((manifold_fidget::order::SeedKind::Bed, _)) => n_bed += 1,
                        None => n_none += 1,
                    }
                }
                x += 1.0;
            }
            y += 1.0;
        }
        println!(
            "L{:3} (order {:6.4}): solved {}/361  isosurf z in [{:.3},{:.3}]  sdf in [{:.3},{:.3}]  patch={} bed={} none={}  patch d in [{:.3},{:.3}]  patch-eligible d<={:.3}: {} ({} on solid)",
            l.index,
            l.order,
            solved,
            z_min,
            z_max,
            sdf_min,
            sdf_max,
            n_patch,
            n_bed,
            n_none,
            d_min,
            d_max,
            top_threshold,
            n_patch_eligible,
            n_patch_eligible_solid
        );
    }

    // Part G: order-field value across the top face (z=5.0) and the groove
    // floor (z=3.0) -- shows tangential order variation of each horizontal
    // face (solid points only).
    println!("\n--- Part G: order map on horizontal faces (solid pts, 0.5mm grid) ---");
    let order_map = |label: &str,
                     z_face: f64,
                     x0: f64,
                     x1: f64,
                     y0: f64,
                     y1: f64,
                     layers: &[manifold_core::slicing::Layer]| {
        let l = &layers[26]; // any layer: the order field is the same object
        let mut n = 0usize;
        let mut o_min = f64::INFINITY;
        let mut o_max = f64::NEG_INFINITY;
        let mut o_sum = 0.0f64;
        let bins = 12usize;
        let mut bin_min = vec![f64::INFINITY; bins * bins];
        let mut bin_max = vec![f64::NEG_INFINITY; bins * bins];
        let mut y = y0;
        while y <= y1 + 1e-6 {
            let mut x = x0;
            while x <= x1 + 1e-6 {
                let p = glam::DVec3::new(x, y, z_face);
                let s = l
                    .mesh_sdf
                    .as_ref()
                    .map(|sdf| sdf.sample(p).value)
                    .unwrap_or(f64::NAN);
                if s < -0.05 {
                    let o = l.order_field.order(p);
                    if o.is_finite() {
                        n += 1;
                        o_min = o_min.min(o);
                        o_max = o_max.max(o);
                        o_sum += o;
                        let bi = (((x - x0) / (x1 - x0) * (bins as f64 - 1.0)).round() as usize)
                            .min(bins - 1);
                        let bj = (((y - y0) / (y1 - y0) * (bins as f64 - 1.0)).round() as usize)
                            .min(bins - 1);
                        let idx = bi * bins + bj;
                        bin_min[idx] = bin_min[idx].min(o);
                        bin_max[idx] = bin_max[idx].max(o);
                    }
                }
                x += 0.5;
            }
            y += 0.5;
        }
        if n == 0 {
            println!("{label}: no solid face points");
            return;
        }
        println!(
            "{}: {} solid pts, order in [{:.4},{:.4}] mean {:.4}",
            label,
            n,
            o_min,
            o_max,
            o_sum / n as f64
        );
        // ASCII map: one char per bin, '#' = face order spans this layer's band region
        let mut rows: Vec<String> = Vec::new();
        for bj in (0..bins).rev() {
            let mut row = String::new();
            for bi in 0..bins {
                let idx = bi * bins + bj;
                if bin_min[idx].is_finite() {
                    row.push_str(&format!("{:5.2}", bin_min[idx]));
                } else {
                    row.push_str("    .");
                }
            }
            rows.push(row);
        }
        for r in rows {
            println!("  {r}");
        }
    };
    order_map(
        "top face z=5.0 (left slab)",
        5.0,
        166.0,
        174.0,
        166.0,
        184.0,
        &layers,
    );
    order_map(
        "groove floor z=3.0",
        3.0,
        177.0,
        184.0,
        169.0,
        181.0,
        &layers,
    );
    Ok(())
}
