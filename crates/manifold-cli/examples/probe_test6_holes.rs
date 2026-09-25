//! Scratch probe for the Test6.stl hole-infill bug:
//! 30mm tile, six 3mm through-holes in a circular pattern, inset trough
//! on the top, chamfers incl. two hole tops.
//!
//! Observed defect (volume audit, 0.4mm cells): sparse in-air infill in ALL
//! six holes (orders ~1.8/2.2/2.6 and 4.6), plus solid/top material in
//! 3-4 holes at the top band (orders 4.6/5.0).
//!
//! This probe:
//!   Part 1: discover the 6 hole centers from the world SDF (air pockets),
//!   Part 2: per-layer even-odd occupancy at hole centers (infill + solid),
//!   Part 3: per-loop dump for hit layers (which loops contain each hole),
//!   Part 4: extruded segments inside each hole cylinder, by kind + order,
//!   Part 5: SDF air confirmation at hole centers.
//!
//! ```sh
//! CARGO_TARGET_DIR=target/build cargo run -p manifold-cli --example probe_test6_holes -- \
//!     Test6.stl examples/profile.json
//! ```

use manifold_core::ids::{ObjectId, ToolId};
use manifold_core::machine::Machine;
use manifold_core::object::{center_on_bed, Object};
use manifold_fidget::mesh_sdf::MeshSdf;
use manifold_fidget::ScalarField;

#[derive(serde::Deserialize)]
struct Profile {
    machine: Machine,
    config: manifold_core::SlicerConfig,
}

fn even_odd_contains(pts: &[[f64; 2]], x: f64, y: f64) -> bool {
    let mut inside = false;
    let n = pts.len();
    if n < 3 {
        return false;
    }
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = (pts[i][0], pts[i][1]);
        let (xj, yj) = (pts[j][0], pts[j][1]);
        if (yi > y) != (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi {
            inside = !inside;
        }
        j = i;
    }
    inside
}
fn signed_area2(pts: &[[f64; 2]]) -> f64 {
    let mut a = 0.0;
    let n = pts.len();
    for i in 0..n {
        let p = pts[i];
        let q = pts[(i + 1) % n];
        a += p[0] * q[1] - q[0] * p[1];
    }
    a / 2.0
}
fn bbox2(pts: &[[f64; 2]]) -> (f64, f64, f64, f64) {
    let mut x0 = f64::INFINITY;
    let mut x1 = f64::NEG_INFINITY;
    let mut y0 = f64::INFINITY;
    let mut y1 = f64::NEG_INFINITY;
    for p in pts {
        x0 = x0.min(p[0]);
        x1 = x1.max(p[0]);
        y0 = y0.min(p[1]);
        y1 = y1.max(p[1]);
    }
    (x0, x1, y0, y1)
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
    let sdf = MeshSdf::new(world_mesh.vertices.clone(), faces.clone());

    // ---- Part 1: discover hole centers (air pockets at mid height) ----
    // Tile is ~30x30 at world ~ (160..190, 160..190); scan at z = 2.5 (mid).
    let z_scan = 2.5;
    let mut air: Vec<(f64, f64)> = Vec::new();
    let mut y = 160.0;
    while y <= 190.0 {
        let mut x = 160.0;
        while x <= 190.0 {
            let v = sdf.sample(glam::DVec3::new(x, y, z_scan)).value;
            if v > 0.5 {
                air.push((x, y));
            }
            x += 0.2;
        }
        y += 0.2;
    }
    // grid cluster (0.5mm connectivity via flood fill on the 0.2mm grid)
    let idx = |x: f64, y: f64| -> Option<usize> {
        let ix = ((x - 160.0) / 0.2).round() as usize;
        let iy = ((y - 160.0) / 0.2).round() as usize;
        Some(ix * 151 + iy)
    };
    let mut seen = vec![false; 151 * 151];
    let mut centers: Vec<(f64, f64, usize)> = Vec::new();
    for &(px, py) in &air {
        let i0 = idx(px, py).unwrap();
        if seen[i0] {
            continue;
        }
        // flood fill
        let mut stack = vec![(px, py)];
        let mut sum_x = 0.0f64;
        let mut sum_y = 0.0f64;
        let mut n = 0usize;
        while let Some((qx, qy)) = stack.pop() {
            let i = idx(qx, qy).unwrap();
            if seen[i] {
                continue;
            }
            seen[i] = true;
            sum_x += qx;
            sum_y += qy;
            n += 1;
            for (dx, dy) in [(0.2, 0.0), (-0.2, 0.0), (0.0, 0.2), (0.0, -0.2)] {
                let nx = qx + dx;
                let ny = qy + dy;
                let Some(j) = idx(nx, ny) else { continue };
                if !seen[j] {
                    let v = sdf.sample(glam::DVec3::new(nx, ny, z_scan)).value;
                    if v > 0.5 {
                        stack.push((nx, ny));
                    }
                }
            }
        }
        if n >= 4 {
            centers.push((sum_x / n as f64, sum_y / n as f64, n));
        }
    }
    centers.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap()
            .then_with(|| a.1.partial_cmp(&b.1).unwrap())
    });
    println!("--- Part 1: air pockets at z={z_scan} (SDF>0.5) ---");
    for (i, (cx, cy, n)) in centers.iter().enumerate() {
        println!(
            "  hole{i} center=({cx:.2},{cy:.2}) cells={n} area~{:.1}mm2",
            *n as f64 * 0.04
        );
    }
    assert!(!centers.is_empty(), "no air pockets found");

    let hole_centers: Vec<(f64, f64)> = centers.iter().map(|(x, y, _)| (*x, *y)).collect();

    // ---- Part 5 (early): SDF air confirmation at centers ----
    println!("\n--- Part 5: SDF at hole centers (air = positive) ---");
    for (i, (cx, cy)) in hole_centers.iter().enumerate() {
        let vals: Vec<String> = (0..5)
            .map(|k| {
                let z = 0.5 + 1.0 * k as f64;
                format!(
                    "{z:.0}:{:.2}",
                    sdf.sample(glam::DVec3::new(*cx, *cy, z)).value
                )
            })
            .collect();
        println!("  hole{i}: {}", vals.join(" "));
    }

    // ---- Part 2: slice with machine slope profile; per-layer parity ----
    println!("\n--- slicing (machine slope profile) ---");
    let slope_profile = profile.machine.slope_profile();
    let layers = manifold_core::slicing::slice_object_with_progress(
        object,
        &profile.config,
        &slope_profile,
        &mut |_| {},
    )?;
    println!("layers: {}", layers.len());

    let max_order = layers.iter().fold(0.0f64, |m, l| m.max(l.order));
    println!("\n--- Part 2: per-layer even-odd at hole centers ---");
    for layer in &layers {
        let ib2: Vec<Vec<[f64; 2]>> = layer
            .infill_boundary
            .iter()
            .map(|l| l.iter().map(|&p| [p.x, p.y]).collect())
            .collect();
        let sf2: Vec<Vec<[f64; 2]>> = layer
            .solid_fill_boundary
            .iter()
            .map(|l| l.iter().map(|&p| [p.x, p.y]).collect())
            .collect();
        let mut ih = 0usize;
        let mut ih_holes = Vec::new();
        for (h, &(cx, cy)) in hole_centers.iter().enumerate() {
            let c = ib2
                .iter()
                .filter(|loop_| even_odd_contains(loop_, cx, cy))
                .count();
            if c % 2 == 1 {
                ih += 1;
                ih_holes.push(h);
            }
        }
        let mut sh = 0usize;
        let mut sh_holes = Vec::new();
        for (h, &(cx, cy)) in hole_centers.iter().enumerate() {
            let c = sf2
                .iter()
                .filter(|loop_| even_odd_contains(loop_, cx, cy))
                .count();
            if c % 2 == 1 {
                sh += 1;
                sh_holes.push(h);
            }
        }
        let cap = max_order - layer.order;
        let flag = if ih > 0 || sh > 0 { " <<<" } else { "" };
        println!(
            "L{:>3} o={:5.2} ib={} sf={} inf_hits={}/{} {:?} solid_hits={}/{} {:?} (cap {:.2}){}",
            layer.index,
            layer.order,
            ib2.len(),
            sf2.len(),
            ih,
            hole_centers.len(),
            ih_holes,
            sh,
            hole_centers.len(),
            sh_holes,
            cap,
            flag
        );
    }

    // ---- Part 3: loop dumps for hit layers ----
    println!("\n--- Part 3: infill/solid loops at hit layers ---");
    let dump_orders: Vec<f64> = std::env::var("PROBE_T6_DUMP_ORDERS")
        .ok()
        .map(|s| s.split(',').filter_map(|t| t.trim().parse().ok()).collect())
        .unwrap_or_default();
    for layer in &layers {
        let ib2: Vec<Vec<[f64; 2]>> = layer
            .infill_boundary
            .iter()
            .map(|l| l.iter().map(|&p| [p.x, p.y]).collect())
            .collect();
        let sf2: Vec<Vec<[f64; 2]>> = layer
            .solid_fill_boundary
            .iter()
            .map(|l| l.iter().map(|&p| [p.x, p.y]).collect())
            .collect();
        let hit = hole_centers.iter().any(|&(cx, cy)| {
            ib2.iter().filter(|l| even_odd_contains(l, cx, cy)).count() % 2 == 1
                || sf2.iter().filter(|l| even_odd_contains(l, cx, cy)).count() % 2 == 1
        });
        let force = dump_orders.iter().any(|o| (o - layer.order).abs() < 0.05);
        if !hit && !force {
            continue;
        }
        println!("L{} o={:.2}:", layer.index, layer.order);
        for (i, loop_) in ib2.iter().enumerate() {
            let (x0, x1, y0, y1) = bbox2(loop_);
            let contains: Vec<usize> = hole_centers
                .iter()
                .enumerate()
                .filter(|&(_h, &(cx, cy))| even_odd_contains(loop_, cx, cy))
                .map(|(h, _)| h)
                .collect();
            let sdf0 = if !loop_.is_empty() {
                sdf.sample(glam::DVec3::new(loop_[0][0], loop_[0][1], layer.order))
                    .value
            } else {
                0.0
            };
            let sa = signed_area2(loop_);
            println!(
                "  ib[{}] n={} x[{:6.2}..{:6.2}] y[{:6.2}..{:6.2}] sa={:+8.0} {} sdf~{:+.2} holes={:?}",
                i,
                loop_.len(),
                x0,
                x1,
                y0,
                y1,
                sa,
                if sa > 0.0 { "OUTER" } else { "hole " },
                sdf0,
                contains
            );
        }
        for (i, loop_) in sf2.iter().enumerate() {
            let (x0, x1, y0, y1) = bbox2(loop_);
            let contains: Vec<usize> = hole_centers
                .iter()
                .enumerate()
                .filter(|&(_h, &(cx, cy))| even_odd_contains(loop_, cx, cy))
                .map(|(h, _)| h)
                .collect();
            let sa = signed_area2(loop_);
            println!(
                "  sf[{}] n={} x[{:6.2}..{:6.2}] y[{:6.2}..{:6.2}] sa={:+8.0} {} holes={:?}",
                i,
                loop_.len(),
                x0,
                x1,
                y0,
                y1,
                sa,
                if sa > 0.0 { "OUTER" } else { "hole " },
                contains
            );
        }
    }

    // ---- Part 4b: replicate the TPMS generator's island construction ----
    use manifold_core::polygon2d;
    println!("\n--- Part 4b: generator island reconstruction ---");
    for layer in &layers {
        let loops3 = &layer.infill_boundary;
        if loops3.is_empty() {
            continue;
        }
        let b1 = glam::DVec3::new(1.0, 0.0, 0.0);
        let b2 = glam::DVec3::new(0.0, 1.0, 0.0);
        let apex = glam::DVec3::ZERO;
        let loops_2d = polygon2d::canonicalize(&polygon2d::to_2d(loops3, b1, b2, apex));
        let mut outers: Vec<usize> = Vec::new();
        let mut holes: Vec<usize> = Vec::new();
        for (i, l) in loops_2d.iter().enumerate() {
            if polygon2d::signed_area(l) > 0.0 {
                outers.push(i);
            } else {
                holes.push(i);
            }
        }
        let mut assigned = vec![false; holes.len()];
        let mut islands: Vec<Vec<usize>> = outers.iter().map(|&o| vec![o]).collect();
        for (o_pos, &o) in outers.iter().enumerate() {
            for (h_pos, &h) in holes.iter().enumerate() {
                if !assigned[h_pos] && polygon2d::point_in_polygon(loops_2d[h][0], &loops_2d[o]) {
                    islands[o_pos].push(h);
                    assigned[h_pos] = true;
                }
            }
        }
        let unassigned: Vec<usize> = holes
            .iter()
            .enumerate()
            .filter(|(i, _)| !assigned[*i])
            .map(|(_, &h)| h)
            .collect();
        let orient: Vec<String> = loops_2d
            .iter()
            .map(|l| {
                if polygon2d::signed_area(l) > 0.0 {
                    "O"
                } else {
                    "h"
                }
                .to_string()
            })
            .collect();
        let isls: Vec<String> = islands
            .iter()
            .map(|is| {
                is.iter()
                    .map(|i| i.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            })
            .collect();
        println!(
            "L{} o={:.2} orient={} islands=[{}] unassigned_holes={:?}",
            layer.index,
            layer.order,
            orient.join(""),
            isls.join(" | "),
            unassigned
        );
        for (hidx, &(cx, cy)) in hole_centers.iter().enumerate() {
            let p = [cx, cy];
            let members: Vec<usize> = (0..loops_2d.len())
                .filter(|&i| polygon2d::point_in_polygon(p, &loops_2d[i]))
                .collect();
            let filled_by: Vec<usize> = islands
                .iter()
                .enumerate()
                .filter(|(_, is)| {
                    let loops: Vec<Vec<[f64; 2]>> =
                        is.iter().map(|&i| loops_2d[i].clone()).collect();
                    polygon2d::contains_point(&loops, p)
                })
                .map(|(i, _)| i)
                .collect();
            println!(
                "   hole{}: in_loops={:?} filled_by_islands={:?}",
                hidx, members, filled_by
            );
        }
    }
    use manifold_core::toolpath;
    println!("\n--- planning toolpaths ---");
    if std::env::var("PROBE_T6_SKIP_PLAN").is_ok() {
        return Ok(());
    }
    let mut on_progress = |_f: f64| {};
    let paths = toolpath::plan_with_progress(
        &layers,
        &objects,
        &profile.machine.tools,
        &profile.config,
        Some(&profile.machine),
        &slope_profile,
        &mut on_progress,
    )?;
    let total_paths = paths.len();
    let mut total_all = 0usize;
    let mut per_hole: Vec<std::collections::HashMap<(i32, String), (usize, f64)>> =
        vec![std::collections::HashMap::new(); hole_centers.len()];
    for path in &paths {
        let pts = &path.points;
        if pts.is_empty() {
            continue;
        }
        let npts = pts.len();
        for (i, seg) in path.segments.iter().enumerate() {
            if seg.extrusion_length <= 0.0 {
                continue;
            }
            total_all += 1;
            let a = pts[i];
            let b = pts[i % npts];
            let mid = glam::DVec3::new((a.x + b.x) / 2.0, (a.y + b.y) / 2.0, (a.z + b.z) / 2.0);
            for (h, &(cx, cy)) in hole_centers.iter().enumerate() {
                let in_cyl = |p: glam::DVec3| {
                    let dx = p.x - cx;
                    let dy = p.y - cy;
                    dx * dx + dy * dy <= 1.4 * 1.4
                };
                if in_cyl(a) || in_cyl(mid) {
                    let key = (
                        (seg.order * 100.0).round() as i32,
                        format!("{:?}", seg.kind),
                    );
                    let e = per_hole[h].entry(key).or_default();
                    e.0 += 1;
                    e.1 += seg.extrusion_length;
                }
            }
        }
    }
    println!("\n--- Part 4: extruded segs inside each hole cylinder (r<=1.4) ---");
    println!("total paths: {total_paths}, total extruded segs: {total_all}");
    for (h, map) in per_hole.iter().enumerate() {
        let tot = map.values().map(|(n, _)| *n).sum::<usize>();
        let mm = map.values().map(|(_, mm)| mm).sum::<f64>();
        if map.is_empty() {
            println!("  hole{h}: 0 segs");
            continue;
        }
        println!("  hole{h}: {tot} segs, {mm:.2}mm extruded:");
        let mut rows: Vec<_> = map.iter().collect();
        rows.sort_by(|a, b| a.0 .0.cmp(&b.0 .0).then_with(|| a.0 .1.cmp(&b.0 .1)));
        for ((order, kind), (n, mm)) in &rows {
            println!(
                "    o={:.2} {kind:<12} n={n:<4} e={mm:.2}mm",
                *order as f64 / 100.0
            );
        }
    }

    Ok(())
}
