//! Scratch probe for the Test6.stl top-surface defect (machine profile).
//!
//! Census of planned path kinds at the top of the part:
//!   Part A: top layers - wall loops, top_surface/unsupported flag counts
//!   Part B: kind x z-band census (extruded segments)
//!   Part C: top-face (z=5.0) coverage by kind
//!   Part D: TopSurface segments in the groove region (world x 176..186)
//!   Part E: Overhang segments near the hole tops
//!
//! ```sh
//! CARGO_TARGET_DIR=target/build cargo run -p manifold-cli --release \
//!     --example probe_test6_topkinds -- Test6.stl examples/profile.json
//! ```

use manifold_core::ids::{ObjectId, ToolId};
use manifold_core::machine::Machine;
use manifold_core::object::{center_on_bed, Object};
use manifold_core::toolpath::{plan_with_progress, MoveKind};
use manifold_fidget::mesh_sdf::MeshSdf;
use std::collections::BTreeMap;

#[derive(serde::Deserialize)]
struct Profile {
    machine: Machine,
    config: manifold_core::SlicerConfig,
}

fn kind_name(k: &MoveKind) -> &'static str {
    match k {
        MoveKind::WallOuter => "WallOuter",
        MoveKind::WallInner => "WallInner",
        MoveKind::Infill => "Infill",
        MoveKind::Bridge => "Bridge",
        MoveKind::Overhang => "Overhang",
        MoveKind::TopSurface => "TopSurface",
        MoveKind::Travel => "Travel",
        MoveKind::Wipe => "Wipe",
        MoveKind::DebugExcluded => "DebugExcluded",
    }
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

    let world_mesh = manifold_core::mesh::Mesh::new(
        object
            .mesh
            .vertices
            .iter()
            .map(|&v| object.transform.transform_point(v))
            .collect(),
        object.mesh.indices.clone(),
    );
    let faces: Vec<[usize; 3]> = world_mesh
        .indices
        .as_chunks::<3>()
        .0
        .iter()
        .map(|c| [c[0] as usize, c[1] as usize, c[2] as usize])
        .collect();
    let _sdf = MeshSdf::new(world_mesh.vertices.clone(), faces);

    let slope_profile = profile.machine.slope_profile();
    let layers = manifold_core::slicing::slice_object_with_progress(
        object,
        &profile.config,
        &slope_profile,
        &mut |_| {},
    )?;
    println!("layers: {} (machine slope profile)", layers.len());

    // ---- Part A: top layers: wall loops + flags ----
    println!("\n--- Part A: top layers (wall-0 loops, ts/unsup counts) ---");
    for layer in layers.iter().rev().take(9) {
        let ib = layer.infill_boundary.len();
        let sf = layer.solid_fill_boundary.len();
        print!(
            "L{:3} order={:7.4} ib={} sf={} walls:",
            layer.index, layer.order, ib, sf
        );
        for wall in layer.loops.iter().filter(|w| w.wall_index == 0).take(6) {
            let n = wall.points.len();
            let ts = wall.top_surface.iter().filter(|&&b| b).count();
            let us = wall.unsupported.iter().filter(|&&b| b).count();
            let z0 = wall
                .points
                .iter()
                .map(|p| p.z)
                .fold(f64::INFINITY, f64::min);
            let z1 = wall
                .points
                .iter()
                .map(|p| p.z)
                .fold(f64::NEG_INFINITY, f64::max);
            print!(" [w0 n={n} ts={ts} us={us} z {z0:.2}..{z1:.2}]");
        }
        println!();
    }

    let mut on_progress = |_f: f64| {};
    let paths = plan_with_progress(
        &layers,
        &objects,
        &profile.machine.tools,
        &profile.config,
        Some(&profile.machine),
        &slope_profile,
        &mut on_progress,
    )?;

    let mut extruded = 0usize;
    let mut total_e = 0.0f64;
    for p in &paths {
        for s in &p.segments {
            if s.extrusion_length > 0.0005 {
                extruded += 1;
                total_e += s.extrusion_length;
            }
        }
    }
    println!("\nplanned: {} extruded segs, {:.3} mm E", extruded, total_e);

    // ---- Part B: kind x z-band census ----
    println!("\n--- Part B: kind x z-band (extruded segs, E>0.0005) ---");
    // (z-band, kind) -> (count, E)
    let mut census: BTreeMap<(i64, &'static str), (usize, f64)> = BTreeMap::new();
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
            let midz = (p.points[i].z + p.points[dest].z) * 0.5;
            let zb = (midz * 5.0).round() as i64;
            let e = census.entry((zb, kind_name(&s.kind))).or_default();
            e.0 += 1;
            e.1 += s.extrusion_length;
        }
    }
    let mut kinds: Vec<&'static str> = census
        .keys()
        .map(|k| k.1)
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    kinds.sort();
    let mut zbands: Vec<i64> = census
        .keys()
        .map(|k| k.0)
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    zbands.sort();
    for zb in zbands.iter().rev().take(14) {
        let z = *zb as f64 / 5.0;
        let mut row = format!("z {:4.1}:", z);
        for k in &kinds {
            if let Some((n, e)) = census.get(&(*zb, *k)) {
                if *n > 0 {
                    row.push_str(&format!(" {k:11} {:>5} {:>8.2}", n, e));
                }
            }
        }
        println!("{row}");
    }

    // ---- Part C: top-face coverage by kind ----
    println!("\n--- Part C: top-face (z=5.0) coverage by kind ---");
    // Top face: world x 160..190, y 160..190, minus groove (x 176..186)
    // and minus the six hole disks (r > 1.4).
    let hole_centers = [
        (167.0, 175.0),
        (171.0, 168.08),
        (171.0, 181.92),
        (179.0, 181.92),
        (179.0, 168.08),
        (183.0, 175.0),
    ];
    // collect material segments near the top face (mid z >= 4.7)
    let mut top_segs: Vec<(f64, f64, &'static str)> = Vec::new(); // (x, y, kind)
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
            let midz = (a.z + b.z) * 0.5;
            if midz < 4.7 {
                continue;
            }
            // subdivide long segments for coverage
            let len = a.distance(b);
            let steps = (len / 0.25).ceil().max(1.0) as usize;
            for st in 0..=steps {
                let t = st as f64 / steps as f64;
                top_segs.push((
                    a.x + t * (b.x - a.x),
                    a.y + t * (b.y - a.y),
                    kind_name(&s.kind),
                ));
            }
        }
    }
    let mut total = 0usize;
    let mut by_kind: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut y = 160.25;
    while y <= 189.75 {
        let mut x = 160.25;
        while x <= 189.75 {
            let in_groove = (176.0..186.0).contains(&x);
            let in_hole = hole_centers.iter().any(|&(cx, cy)| {
                let dx: f64 = x - cx;
                let dy: f64 = y - cy;
                dx.hypot(dy) < 1.4
            });
            if !in_groove && !in_hole {
                total += 1;
                let mut hit: Option<&'static str> = None;
                let mut any: BTreeMap<&'static str, usize> = BTreeMap::new();
                for (sx, sy, k) in &top_segs {
                    if (sx - x).abs() <= 0.3 && (sy - y).abs() <= 0.3 {
                        *any.entry(*k).or_insert(0) += 1;
                        if hit.is_none() && *k == "TopSurface" {
                            hit = Some("TopSurface");
                        }
                    }
                }
                if !any.is_empty() {
                    // primary kind = the one with the most hits; count TopSurface
                    // separately even if minority
                    let mut best = ("WallOuter", 0usize);
                    for (k, n) in &any {
                        if *n > best.1 {
                            best = (*k, *n);
                        }
                    }
                    *by_kind.entry(best.0).or_insert(0) += 1;
                    if any.get("TopSurface").copied().unwrap_or(0) > 0 {
                        *by_kind.entry("__topsurface_any__").or_insert(0) += 1;
                    }
                }
            }
            x += 0.5;
        }
        y += 0.5;
    }
    println!("top-face sample points: {total}");
    for (k, n) in by_kind.iter().rev() {
        println!(
            "  {:16} {:6.1}%  ({})",
            k,
            100.0 * *n as f64 / total.max(1) as f64,
            n
        );
    }

    // ---- Part D: TopSurface segments in the groove region ----
    println!(
        "\n--- Part D: TopSurface-kind segs in groove region (world x 176..186, z 3.4..4.9) ---"
    );
    let mut dcount = 0usize;
    let mut de = 0.0f64;
    let mut dby_z: BTreeMap<i64, (usize, f64)> = BTreeMap::new();
    for p in &paths {
        for i in 0..p.segments.len() {
            let s = &p.segments[i];
            if s.extrusion_length <= 0.0005 || kind_name(&s.kind) != "TopSurface" {
                continue;
            }
            let dest = if p.segments.len() == p.points.len() {
                (i + 1) % p.points.len()
            } else {
                (i + 1).min(p.points.len() - 1)
            };
            let midz = (p.points[i].z + p.points[dest].z) * 0.5;
            let midx = (p.points[i].x + p.points[dest].x) * 0.5;
            if (3.4..4.9).contains(&midz) && (175.5..186.5).contains(&midx) {
                dcount += 1;
                de += s.extrusion_length;
                let zb = (midz * 5.0).round() as i64;
                let e = dby_z.entry(zb).or_default();
                e.0 += 1;
                e.1 += s.extrusion_length;
            }
        }
    }
    println!("count={dcount} E={de:.3}");
    for (zb, (n, e)) in dby_z {
        let z = zb as f64 / 5.0;
        println!("  z {z:.1}: {n} segs {e:.3} mm");
    }

    // ---- Part E: Overhang segs near hole tops ----
    println!("\n--- Part E: Overhang-kind segs within 2.5mm of hole centers, z 4.0..5.1 ---");
    let mut ecount = 0usize;
    let mut ee = 0.0f64;
    let mut eby_hole: BTreeMap<usize, (usize, f64)> = BTreeMap::new();
    for p in &paths {
        for i in 0..p.segments.len() {
            let s = &p.segments[i];
            if s.extrusion_length <= 0.0005 || kind_name(&s.kind) != "Overhang" {
                continue;
            }
            let dest = if p.segments.len() == p.points.len() {
                (i + 1) % p.points.len()
            } else {
                (i + 1).min(p.points.len() - 1)
            };
            let midz = (p.points[i].z + p.points[dest].z) * 0.5;
            let midx = (p.points[i].x + p.points[dest].x) * 0.5;
            let midy = (p.points[i].y + p.points[dest].y) * 0.5;
            if !(4.0..5.1).contains(&midz) {
                continue;
            }
            for (h, (cx, cy)) in hole_centers.iter().enumerate() {
                if (midx - cx).hypot(midy - cy) <= 2.5 {
                    ecount += 1;
                    ee += s.extrusion_length;
                    let e2 = eby_hole.entry(h).or_default();
                    e2.0 += 1;
                    e2.1 += s.extrusion_length;
                    break;
                }
            }
        }
    }
    println!("count={ecount} E={ee:.3}");
    for (h, (n, e)) in eby_hole {
        println!(
            "  hole {h} ({},{}): {} segs {:.3} mm",
            hole_centers[h].0, hole_centers[h].1, n, e
        );
    }

    // ---- Part F: per-layer kind census for top layers ----
    println!("\n--- Part F: per-layer kind census (L13..L26) ---");
    for layer in layers.iter().rev().take(14) {
        let mut per: BTreeMap<&'static str, (usize, f64)> = BTreeMap::new();
        for p in &paths {
            for i in 0..p.segments.len() {
                let s = &p.segments[i];
                if s.extrusion_length <= 0.0005 {
                    continue;
                }
                if (s.order - layer.order).abs() > 1e-6 {
                    continue;
                }
                let e2 = per.entry(kind_name(&s.kind)).or_default();
                e2.0 += 1;
                e2.1 += s.extrusion_length;
            }
        }
        let n_paths: usize = paths
            .iter()
            .filter(|p| {
                p.segments
                    .iter()
                    .any(|s| (s.order - layer.order).abs() <= 1e-6 && s.extrusion_length > 0.0005)
            })
            .count();
        print!(
            "L{:3} order={:7.4} paths={}",
            layer.index, layer.order, n_paths
        );
        for (k, (n, e)) in &per {
            print!(" {k:11} {:>6} {:>8.2}", n, e);
        }
        println!();
    }

    // ---- Part G: found_ib / solid_fill / wall loop bboxes at floor+top levels ----
    println!("\n--- Part G: loop bboxes (L13..L16, L22..L26) ---");
    for layer in &layers {
        let in_range = (13..=16).contains(&layer.index) || (22..=26).contains(&layer.index);
        if !in_range {
            continue;
        }
        let dump = |label: &str, loops: &[Vec<glam::DVec3>]| {
            print!(
                "L{} o={:.4} {} = {} loops",
                layer.index,
                layer.order,
                label,
                loops.len()
            );
            for (i, lp) in loops.iter().enumerate() {
                if lp.is_empty() {
                    continue;
                }
                let (mut x0, mut x1, mut y0, mut y1, mut z0, mut z1) = (
                    f64::INFINITY,
                    f64::NEG_INFINITY,
                    f64::INFINITY,
                    f64::NEG_INFINITY,
                    f64::INFINITY,
                    f64::NEG_INFINITY,
                );
                for p in lp {
                    x0 = x0.min(p.x);
                    x1 = x1.max(p.x);
                    y0 = y0.min(p.y);
                    y1 = y1.max(p.y);
                    z0 = z0.min(p.z);
                    z1 = z1.max(p.z);
                }
                print!(
                    " [{} n={} x{:.1}..{:.1} y{:.1}..{:.1} z{:.2}..{:.2}]",
                    i,
                    lp.len(),
                    x0,
                    x1,
                    y0,
                    y1,
                    z0,
                    z1
                );
            }
            println!();
        };
        dump("ib", &layer.infill_boundary);
        dump("sf", &layer.solid_fill_boundary);
        let wall_pts: Vec<Vec<glam::DVec3>> =
            layer.loops.iter().map(|w| w.points.clone()).collect();
        dump("walls", &wall_pts);
        for (i, w) in layer.loops.iter().enumerate() {
            let ts = w.top_surface.iter().filter(|&&f| f).count();
            println!(
                "  L{} wall[{}] wall_index={} n={} top_surface_pts={}",
                layer.index,
                i,
                w.wall_index,
                w.points.len(),
                ts
            );
        }
    }

    // ---- Part H: seed_proximity at slab/groove points across z ----
    println!("\n--- Part H: seed_proximity (d; margin=0.601-d; Patch~0 Bed~4.4) ---");
    let threshold = 3.0 * 0.2 + 1e-3;
    for layer in layers.iter().rev().take(4) {
        for (label, p) in [
            ("face  (170,175,5.00)", glam::DVec3::new(170.0, 175.0, 5.00)),
            ("faceE (170,175,4.80)", glam::DVec3::new(170.0, 175.0, 4.80)),
            ("intrl (170,175,4.60)", glam::DVec3::new(170.0, 175.0, 4.60)),
            ("intrl (170,175,4.40)", glam::DVec3::new(170.0, 175.0, 4.40)),
            ("florr (181,175,3.00)", glam::DVec3::new(181.0, 175.0, 3.00)),
            ("florr (181,175,2.80)", glam::DVec3::new(181.0, 175.0, 2.80)),
        ] {
            let sp = layer.order_field.seed_proximity(p);
            match sp {
                Some((_kind, d)) => {
                    let margin = threshold - d;
                    println!(
                        "L{:2} o={:.4} {:14} -> d={:7.3} margin={:+8.3}",
                        layer.index, layer.order, label, d, margin
                    );
                }
                None => {
                    println!(
                        "L{:2} o={:.4} {:14} -> None",
                        layer.index, layer.order, label
                    );
                }
            }
        }
    }

    // ---- Part I: order + seed_proximity along the top-face and floor
    // columns, plus SeedMarginField-style reconstruction for L24/L26 ----
    use manifold_core::order_field::{reconstruct_on_order_field, resolve_axis_apex_slope};
    let top = layers.last().unwrap();
    let max_along = (profile.config.layer_height * 20.0).max(5.0);
    println!("\n--- Part I: order/seed columns + L24/L26 reconstruction ---");
    for (label, x, y, zs) in [
        (
            "face (170,175)",
            170.0,
            175.0,
            [4.40, 4.4829, 4.55, 4.60, 4.70, 4.80, 4.90, 4.95, 5.00],
        ),
        (
            "floor (181,175)",
            181.0,
            175.0,
            [2.50, 2.60, 2.7282, 2.80, 2.90, 2.95, 3.00, 3.05, 3.10],
        ),
    ] {
        for z in zs {
            let p = glam::DVec3::new(x, y, z);
            let o = top.order_field.order(p);
            let sp = top.order_field.seed_proximity(p);
            match sp {
                Some((kind, d)) => {
                    println!(
                        "{label:14} z={:6.4} order={:8.4} seed={:?} d={:7.4} top_margin={:+8.4} bot_margin={:+8.4}",
                        z, o, kind, d, 0.601 - d, 0.601 - d
                    );
                }
                None => {
                    println!("{label:14} z={z:6.4} order={o:8.4} seed=None");
                }
            }
        }
    }
    let (axis, apex, _slope) = resolve_axis_apex_slope(profile.config.order_field, &profile.config);
    for li in [14usize, 15, 24, 26] {
        if li >= layers.len() {
            continue;
        }
        let layer = &layers[li];
        for (label, u, v) in [("face", 170.0, 175.0), ("floor", 181.0, 175.0)] {
            let contours = vec![vec![[u, v]]];
            let pts = reconstruct_on_order_field(
                contours,
                glam::DVec3::X,
                glam::DVec3::Y,
                axis,
                apex,
                layer.order,
                max_along,
                layer.order_field.as_ref(),
            );
            let rp = pts[0][0];
            let sp = layer.order_field.seed_proximity(rp);
            match sp {
                Some((kind, d)) => {
                    println!(
                        "L{} {label:5} reconstruct -> z={:8.4} seed={:?} d={:7.4} top_margin={:+8.4}",
                        li, rp.z, kind, d, 0.601 - d
                    );
                }
                None => {
                    println!("L{li} {label:5} reconstruct -> z={:.4} seed=None", rp.z);
                }
            }
        }
    }

    // ---- Part J: SeedMarginField-style margin maps at L14/L15/L24/L26 ----
    // '#' = positive margin (seed-eligible), '.' = negative, '?' = none.
    let (axis_j, apex_j, _slope_j) =
        resolve_axis_apex_slope(profile.config.order_field, &profile.config);
    for li in [14usize, 15, 24, 26] {
        if li >= layers.len() {
            continue;
        }
        let layer = &layers[li];
        println!(
            "L{} o={:.4} margin map (u=162..188 step 2, v=162..188 step 2):",
            li, layer.order
        );
        for vi in 0..=13 {
            let v = 162.0 + vi as f64 * 2.0;
            let mut row = format!("v{:4.0} ", v);
            for ui in 0..=13 {
                let u = 162.0 + ui as f64 * 2.0;
                let contours = vec![vec![[u, v]]];
                let pts = reconstruct_on_order_field(
                    contours,
                    glam::DVec3::X,
                    glam::DVec3::Y,
                    axis_j,
                    apex_j,
                    layer.order,
                    max_along,
                    layer.order_field.as_ref(),
                );
                let rp = pts[0][0];
                match layer.order_field.seed_proximity(rp) {
                    Some((_kind, d)) => {
                        let m = 0.601 - d;
                        row.push(if m > 0.0 { '#' } else { '.' });
                    }
                    None => row.push('?'),
                }
            }
            println!("{row}");
        }
    }

    // ---- Part K: kind census + spatial map for z >= 4.3 ----
    println!("\n--- Part K: all segs z>=4.3 by kind (count, E, bbox) ---");
    use std::collections::BTreeMap as BK;
    type KindAgg = BK<&'static str, (usize, f64, f64, f64, f64, f64, f64)>;
    type CellMap = BK<(i32, i32), BK<&'static str, (usize, f64)>>;
    let mut kagg: KindAgg = BK::new();
    let mut map: CellMap = BK::new();
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
            let midz = (a.z + b.z) * 0.5;
            if midz < 4.3 {
                continue;
            }
            let midx = (a.x + b.x) * 0.5;
            let midy = (a.y + b.y) * 0.5;
            let k = kind_name(&s.kind);
            let e = kagg.entry(k).or_insert((
                0,
                0.0,
                f64::INFINITY,
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::NEG_INFINITY,
                f64::NEG_INFINITY,
            ));
            e.0 += 1;
            e.1 += s.extrusion_length;
            e.2 = e.2.min(midx);
            e.3 = e.3.min(midy);
            e.4 = e.4.max(midx);
            e.5 = e.5.max(midy);
            e.6 = e.6.max(midz);
            if midz >= 4.85 {
                let cell = ((midx - 160.0) as i32, (midy - 160.0) as i32);
                let e2 = map.entry(cell).or_default();
                let e3 = e2.entry(k).or_insert((0, 0.0));
                e3.0 += 1;
                e3.1 += s.extrusion_length;
            }
        }
    }
    for (k, (n, e, x0, y0, x1, y1, z1)) in &kagg {
        println!(
            "  {k:11} {:>6} segs {:9.3} mm E  x {x0:.1}..{x1:.1} y {y0:.1}..{y1:.1} zmax {z1:.2}",
            n, e
        );
    }
    println!("top-face kind map (z>=4.85, 1mm cells, x=col 160..189, y=row 160..189; W=WallOuter I=WallInner S=TopSurface i=Infill O=Overhang B=Bridge o=other):");
    let legend: Vec<&'static str> = vec![];
    for yi in (0..30).rev() {
        let mut row = format!("y{:>3} ", yi + 160);
        for xi in 0..30 {
            let cell = (xi, yi);
            let mut ch = '.';
            if let Some(per) = map.get(&cell) {
                // pick dominant kind by E
                let mut best = ("", 0f64);
                for (k, (n, e)) in per.iter() {
                    let _ = n;
                    if *e > best.1 {
                        best = (k, *e);
                    }
                }
                ch = match best.0 {
                    "WallOuter" => 'W',
                    "WallInner" => 'I',
                    "TopSurface" => 'S',
                    "Infill" => 'i',
                    "Overhang" => 'O',
                    "Bridge" => 'B',
                    _ => 'o',
                };
            }
            row.push(ch);
        }
        println!("{row}");
    }
    let _ = legend;
    let _ = map;
    // ---- Part M: top-face order structure + slab attribution ----
    let of = &top.order_field;
    println!("\n--- Part M1: order near the top face ---");
    for z in [3.60, 4.00, 4.40, 4.60, 4.80, 4.90, 4.95, 5.00, 5.05] {
        let mut row = format!("z={:5.2} (y=175): ", z);
        for x in (163..=187).step_by(2) {
            row.push_str(&format!(
                "{:>7.4}",
                of.order(glam::DVec3::new(x as f64, 175.0, z))
            ));
        }
        println!("{row}");
    }
    println!("face z=5.0 grid (x=col 163..187 step2, y=row 163..187 step2):");
    for y in (163..=187).step_by(2) {
        let mut row = format!("y={:5.0}: ", y);
        for x in (163..=187).step_by(2) {
            row.push_str(&format!(
                "{:>7.4}",
                of.order(glam::DVec3::new(x as f64, y as f64, 5.0))
            ));
        }
        println!("{row}");
    }

    println!("\n--- Part M2: top-material (mid z>=4.35) by kind x owning layer slab ---");
    let layer_orders: Vec<f64> = layers.iter().map(|l| l.order).collect();
    let owner = |o: f64| -> i64 {
        for (i, &lo) in layer_orders.iter().enumerate() {
            let hi = layer_orders.get(i + 1).copied().unwrap_or(f64::INFINITY);
            if o >= lo && o < hi {
                return i as i64;
            }
        }
        -1
    };
    use std::collections::BTreeMap as BM2;
    let mut attr: BM2<(&'static str, i64), (i32, f64)> = BM2::new();
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
            let mid = (a + b) * 0.5;
            if mid.z < 4.35 {
                continue;
            }
            let o = of.order(mid);
            let k = kind_name(&s.kind);
            let e2 = attr.entry((k, owner(o))).or_insert((0, 0.0));
            e2.0 += 1;
            e2.1 += s.extrusion_length;
        }
    }
    for ((k, li), (n, e)) in &attr {
        println!("  {k:11} L{:3} {:>6} segs {:9.3} mm E", *li, n, e);
    }

    println!("\n--- Part M3: face z=5.0 grid points by owning slab ---");
    let mut fb: BM2<i64, i32> = BM2::new();
    for y in (163..=187).step_by(2) {
        for x in (163..=187).step_by(2) {
            let o = of.order(glam::DVec3::new(x as f64, y as f64, 5.0));
            *fb.entry(owner(o)).or_insert(0) += 1;
        }
    }
    for (li, n) in &fb {
        println!("  L{:3} {:>4} face pts", *li, n);
    }

    println!("\n--- Part M4: seed columns at (170,175) z 4.40..5.05, top 4 layers ---");
    for layer in layers.iter().rev().take(4) {
        for k in 0..14 {
            let z = 4.40 + k as f64 * 0.05;
            let p = glam::DVec3::new(170.0, 175.0, z);
            let o = layer.order_field.order(p);
            match layer.order_field.seed_proximity(p) {
                Some((kind, d)) => {
                    println!(
                        "L{:2} z={:5.2} o={:7.4} seed={:?} d={:7.4}",
                        layer.index, z, o, kind, d
                    );
                }
                None => {
                    println!("L{:2} z={:5.2} o={:7.4} seed=None", layer.index, z, o);
                }
            }
        }
        println!();
    }

    println!("\n--- Part M5: top layers ib/sf ---");
    for layer in layers.iter().rev().take(6) {
        println!(
            "L{:2} o={:.4} ib={} sf={}",
            layer.index,
            layer.order,
            layer.infill_boundary.len(),
            layer.solid_fill_boundary.len()
        );
    }

    Ok(())
}
