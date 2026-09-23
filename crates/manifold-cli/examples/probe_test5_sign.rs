//! Scratch probe for the Test5.stl bug investigation:
//!
//! Part 1: audit `MeshSdf` sign classification in the top channel region.
//! Compare the default sign strategy (Pseudonormal + `fast_parity_sign`
//! cross-check) against the O(N) winding-number ground truth on a 3D grid,
//! for both the plain all-faces SDF and the `bed_open_sdf` variant
//! (downward bed-floor faces excluded from distance).
//!
//! Part 2: run the real slice pipeline (`slice_object`, which includes the
//! `compute_solid_fill_boundaries` post-pass) and dump, per layer: loop
//! counts by wall index, infill/solid boundary areas, and flag loops whose
//! entire 3D bbox sits inside the open channel region (object x 8..22,
//! y 8..22) — the region that is air, not material.
//!
//! ```sh
//! CARGO_TARGET_DIR=target/build cargo run --release -p manifold-cli --example probe_test5_sign -- \
//!     Test5.stl examples/profile.json
//! ```

use manifold_core::ids::{ObjectId, ToolId};
use manifold_core::machine::Machine;
use manifold_core::object::{center_on_bed, Object};
use manifold_fidget::mesh_sdf::{MeshSdf, SignMethod};

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

    let mut min_z = f64::INFINITY;
    for v in &world_mesh.vertices {
        min_z = min_z.min(v.z);
    }
    // Replicate mesh::non_bed_floor_faces (pub(crate)) for the bed-open SDF.
    let dist_faces: Vec<[usize; 3]> = faces
        .iter()
        .filter_map(|&[i0, i1, i2]| {
            let v0 = world_mesh.vertices[i0];
            let v1 = world_mesh.vertices[i1];
            let v2 = world_mesh.vertices[i2];
            let normal = (v1 - v0).cross(v2 - v0);
            let nl2 = normal.length_squared();
            if nl2 > 1e-12
                && normal.z < 0.0
                && (v0.z <= min_z + 0.02 || v1.z <= min_z + 0.02 || v2.z <= min_z + 0.02)
                && normal.z * normal.z >= 0.998 * nl2
            {
                return None;
            }
            Some([i0, i1, i2])
        })
        .collect();
    println!(
        "faces: {} total, {} bed-floor faces excluded from distance ({:.1}% excluded)",
        faces.len(),
        faces.len() - dist_faces.len(),
        100.0 * (faces.len() - dist_faces.len()) as f64 / faces.len() as f64
    );

    let sdf = MeshSdf::new(world_mesh.vertices.clone(), faces.clone());
    let mut sdf_w = MeshSdf::new(world_mesh.vertices.clone(), faces.clone());
    sdf_w.set_sign_method(SignMethod::WindingNumber);
    let sdf_bed =
        MeshSdf::new_with_distance_faces(world_mesh.vertices.clone(), faces.clone(), dist_faces);

    let t0 = std::time::Instant::now();
    // Object-coord grid over the top channel region; map through the
    // object's own transform to world coords.
    let mut default_disagree = 0u32;
    let mut bed_disagree = 0u32;
    let mut default_inside_air = 0u32;
    let mut bed_inside_air = 0u32;
    let mut shown = 0u32;
    let mut per_z: std::collections::HashMap<i32, (u32, u32)> = std::collections::HashMap::new();
    for zi in 0..=22 {
        let z = 19.0 + 0.5 * zi as f64;
        for xi in 0..=36 {
            let x = 6.0 + 0.5 * xi as f64;
            for yi in 0..=36 {
                let y = 6.0 + 0.5 * yi as f64;
                let p = object.transform.transform_point(glam::DVec3::new(x, y, z));
                let vd = sdf.sample(p).value;
                let vw = sdf_w.sample(p).value;
                let vb = sdf_bed.sample(p).value;
                let d = (vd < 0.0) != (vw < 0.0);
                let b = (vb < 0.0) != (vw < 0.0);
                if d {
                    default_disagree += 1;
                    if vd < 0.0 {
                        default_inside_air += 1;
                    }
                }
                if b {
                    bed_disagree += 1;
                    if vb < 0.0 {
                        bed_inside_air += 1;
                    }
                }
                let e = per_z.entry(zi).or_default();
                if d {
                    e.0 += 1;
                }
                if b {
                    e.1 += 1;
                }
                if (d || b) && shown < 40 {
                    shown += 1;
                    println!(
                        "obj({:4.1},{:4.1},{:4.1}) winding={:+6.3} default={:+7.3} bedopen={:+7.3}{}",
                        x,
                        y,
                        z,
                        vw,
                        vd,
                        vb,
                        if vd < 0.0 && vw > 0.0 {
                            "   <== default says INSIDE, winding says OUTSIDE"
                        } else if vb < 0.0 && vw > 0.0 {
                            "   <== bed-open says INSIDE, winding says OUTSIDE"
                        } else {
                            ""
                        }
                    );
                }
            }
        }
    }
    println!(
        "sign audit done in {:.1}s: default-disagree={} ({} of those call AIR inside), bed-open-disagree={} ({} call AIR inside)",
        t0.elapsed().as_secs_f64(),
        default_disagree,
        default_inside_air,
        bed_disagree,
        bed_inside_air
    );
    let mut per_z_sorted: Vec<_> = per_z.iter().filter(|e| e.1 .0 + e.1 .1 > 0).collect();
    per_z_sorted.sort_by_key(|e| e.0);
    for (&zi, &(d, b)) in &per_z_sorted {
        println!(
            "  z={:4.1}: default-disagree={} bedopen-disagree={}",
            19.0 + 0.5 * zi as f64,
            d,
            b
        );
    }

    // ---- Part 2: real pipeline layer dump ----
    println!("\n--- slicing (this takes ~2 min in release) ---");
    let slope_profile = profile.machine.slope_profile();
    let layers = manifold_core::slicing::slice_object_with_progress(
        object,
        &profile.config,
        &slope_profile,
        &mut |_| {},
    )?;
    println!("layers: {} (with machine slope profile)", layers.len());

    // Channel region in world coords: object x 8..22, y 8..22 (map corners).
    let ch_min = object
        .transform
        .transform_point(glam::DVec3::new(8.0, 8.0, 0.0));
    let ch_max = object
        .transform
        .transform_point(glam::DVec3::new(22.0, 22.0, 30.0));

    for (i, layer) in layers.iter().enumerate() {
        let by_wall: std::collections::HashMap<usize, (usize, f64, f64, f64, f64)> = layer
            .loops
            .iter()
            .fold(std::collections::HashMap::new(), |mut m, w| {
                let e =
                    m.entry(w.wall_index)
                        .or_insert((0, f64::MAX, f64::MAX, f64::MIN, f64::MIN));
                e.0 += 1;
                for p in &w.points {
                    e.1 = e.1.min(p.x);
                    e.2 = e.2.min(p.y);
                    e.3 = e.3.max(p.x);
                    e.4 = e.4.max(p.y);
                }
                m
            });
        let walls: Vec<(usize, &usize, f64, f64, f64, f64)> = by_wall
            .iter()
            .map(|(w, (n, x0, y0, x1, y1))| (*w, n, *x0, *y0, *x1, *y1))
            .collect();
        let infill_area: f64 = layer
            .infill_boundary
            .iter()
            .map(|l| polygon_area_3d(l))
            .sum();
        let solid_area: f64 = layer
            .solid_fill_boundary
            .iter()
            .map(|l| polygon_area_3d(l))
            .sum();
        // loops fully inside the channel box (in world coords)?
        let ch_loops: Vec<(usize, f64, f64)> = layer
            .loops
            .iter()
            .filter(|w| {
                w.points.iter().all(|p| {
                    p.x >= ch_min.x && p.x <= ch_max.x && p.y >= ch_min.y && p.y <= ch_max.y
                })
            })
            .map(|w| (w.wall_index, w.points[0].z, w.points.len() as f64))
            .collect();
        let mut line = format!(
            "L{:3} o={:6.2} walls[{}] infill={:7.1}mm^2({}loops) solid={:7.1}mm^2({}loops)",
            i,
            layer.order,
            walls
                .iter()
                .map(|(w, n, _, _, _, _)| format!("w{w}={n}"))
                .collect::<Vec<_>>()
                .join(", "),
            infill_area,
            layer.infill_boundary.len(),
            solid_area,
            layer.solid_fill_boundary.len()
        );
        if !ch_loops.is_empty() {
            let desc: Vec<String> = ch_loops
                .iter()
                .take(6)
                .map(|(w, z, n)| format!("w{w}@z{z:.2}(n{n:.0})"))
                .collect();
            line.push_str(&format!(
                "  CHANNEL-BOX LOOPS: {}{}",
                desc.join(" "),
                if ch_loops.len() > 6 { " ..." } else { "" }
            ));
        }
        println!("{line}");
    }

    // ---- Part 3: seed-proximity probe on selected layers ----
    let th_b = profile.config.bottom_layers as f64 * profile.config.layer_height + 1e-3;
    let th_t = profile.config.top_layers as f64 * profile.config.layer_height + 1e-3;
    println!("\nthresholds: bottom={th_b:.4} top={th_t:.4}");
    for li in [0usize, 1, 2, 155, 130] {
        let layer = &layers[li];
        let f = layer.order_field.as_ref();
        println!("\n--- layer {} (order {:.2}) ---", li, layer.order);
        let mut shown = 0usize;
        'outer: for w in &layer.loops {
            if w.points.is_empty() {
                continue;
            }
            for k in 0..4 {
                let idx = (k * w.points.len() / 4) % w.points.len();
                let p = w.points[idx];
                let o = f.order(p);
                let sp = f.seed_proximity(p);
                let (kind, d): (&str, f64) = match sp {
                    Some((ref kk, dd)) => (
                        match kk {
                            manifold_fidget::order::SeedKind::Bed => "Bed",
                            manifold_fidget::order::SeedKind::Patch => "Patch",
                        },
                        dd,
                    ),
                    None => ("None", f64::NAN),
                };
                println!(
                    "  w{} pt({:7.2},{:7.2},{:6.2}) order={:8.3} seed=({}, {:9.3}) bed-margin={:+9.3} patch-margin={:+9.3}",
                    w.wall_index,
                    p.x,
                    p.y,
                    p.z,
                    o,
                    kind,
                    d,
                    th_b - d,
                    th_t - d
                );
                shown += 1;
                if shown >= 12 {
                    break 'outer;
                }
            }
        }
        for (k, l) in layer.infill_boundary.iter().enumerate() {
            let x0 = l.iter().map(|p| p.x).fold(f64::INFINITY, f64::min);
            let x1 = l.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max);
            let y0 = l.iter().map(|p| p.y).fold(f64::INFINITY, f64::min);
            let y1 = l.iter().map(|p| p.y).fold(f64::NEG_INFINITY, f64::max);
            let z0 = l.iter().map(|p| p.z).fold(f64::INFINITY, f64::min);
            let z1 = l.iter().map(|p| p.z).fold(f64::NEG_INFINITY, f64::max);
            println!(
                "  infill-loop{}: x[{x0:7.2}..{x1:7.2}] y[{y0:7.2}..{y1:7.2}] z[{z0:6.2}..{z1:6.2}] n={}",
                k,
                l.len()
            );
        }
    }
    // ---- Part 4: rasterize bottom-layer seed margin + isolated post-pass ----
    use manifold_core::slicing::Layer;
    let base = &layers[0];
    let f = base.order_field.as_ref();
    for zprobe in [0.0f64, 0.2, 0.56] {
        println!(
            "\nmargin raster z={zprobe:.2} (flat probe; '+' eligible, '-' sparse, 'i' inf/none):"
        );
        let mut y = 0.0;
        while y <= 28.0 + 1e-9 {
            let mut row = String::new();
            let mut x = 0.0;
            while x <= 30.0 + 1e-9 {
                let p = object
                    .transform
                    .transform_point(glam::DVec3::new(x, y, zprobe));
                let d = match f.seed_proximity(p) {
                    Some((_, d)) => d,
                    None => f64::NAN,
                };
                row.push(if d.is_nan() || d.is_infinite() {
                    'i'
                } else if 0.601 - d >= 0.0 {
                    '+'
                } else {
                    '-'
                });
                x += 1.0;
            }
            println!("  y={y:4.1} {row}");
            y += 1.0;
        }
    }

    let mut solo = vec![Layer {
        order_field: std::sync::Arc::clone(&base.order_field),
        order: base.order,
        infill_boundary: base.infill_boundary.clone(),
        loops: base.loops.clone(),
        ..Default::default()
    }];
    manifold_core::slicing::compute_solid_fill_boundaries(&mut solo, &profile.config);
    let n = solo[0].solid_fill_boundary.len();
    println!("\nISOOLATED compute_solid_fill_boundaries(layer 0 data): solid loops = {n}");
    for l in &solo[0].solid_fill_boundary {
        let x0 = l.iter().map(|p| p.x).fold(f64::INFINITY, f64::min);
        let x1 = l.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max);
        let y0 = l.iter().map(|p| p.y).fold(f64::INFINITY, f64::min);
        let y1 = l.iter().map(|p| p.y).fold(f64::NEG_INFINITY, f64::max);
        println!(
            "  solid-loop: x[{x0:7.2}..{x1:7.2}] y[{y0:7.2}..{y1:7.2}] n={}",
            l.len()
        );
    }

    // control: same geometry, plain HeightOrderField
    let height = std::sync::Arc::new(manifold_fidget::order::HeightOrderField::new(
        glam::DVec3::new(0.0, 0.0, 1.0),
    ));
    let mut solo_h = vec![Layer {
        order_field: height,
        order: base.order,
        infill_boundary: base.infill_boundary.clone(),
        loops: base.loops.clone(),
        ..Default::default()
    }];
    manifold_core::slicing::compute_solid_fill_boundaries(&mut solo_h, &profile.config);
    println!(
        "control HeightOrderField(layer 0 data): solid loops = {}",
        solo_h[0].solid_fill_boundary.len()
    );
    // ---- Part 5: central-hole infill test (symptom b) ----
    // Sparse infill in the hole requires the hole interior to be inside
    // infill_boundary at some layer. Test the hole center object(15,15)
    // (even-odd over all infill loops, in world XY).
    let hcenter = object
        .transform
        .transform_point(glam::DVec3::new(15.0, 15.0, 0.0));
    let hx = hcenter.x;
    let hy = hcenter.y;
    println!("\n--- Part 5: central-hole infill test, center world ({hx:.1},{hy:.1}) ---");
    for (i, layer) in layers.iter().enumerate() {
        if layer.infill_boundary.is_empty() {
            continue;
        }
        let mut containing: Vec<usize> = Vec::new();
        for (k, l) in layer.infill_boundary.iter().enumerate() {
            if poly_contains_2d(l, hx, hy) {
                containing.push(k);
            }
        }
        if containing.len() % 2 == 1 {
            let zmin = layer
                .infill_boundary
                .iter()
                .flatten()
                .map(|p| p.z)
                .fold(f64::INFINITY, f64::min);
            let zmax = layer
                .infill_boundary
                .iter()
                .flatten()
                .map(|p| p.z)
                .fold(f64::NEG_INFINITY, f64::max);
            println!(
                "HOLE-INSIDE L{:3} o={:6.2} infill-z[{zmin:5.2}..{zmax:5.2}] contains-loop={:?}",
                i, layer.order, containing
            );
            for &k in &containing {
                let l = &layer.infill_boundary[k];
                let x0 = l.iter().map(|p| p.x).fold(f64::INFINITY, f64::min);
                let x1 = l.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max);
                let y0 = l.iter().map(|p| p.y).fold(f64::INFINITY, f64::min);
                let y1 = l.iter().map(|p| p.y).fold(f64::NEG_INFINITY, f64::max);
                println!(
                    "    loop{k} x[{x0:6.2}..{x1:6.2}] y[{y0:6.2}..{y1:6.2}] n={}",
                    l.len()
                );
            }
        }
    }
    // 1052-signature loop search across BOTH layer.loops and infill_boundary.
    for (i, layer) in layers.iter().enumerate() {
        let scan = |tag: &str, pts: &[glam::DVec3]| {
            let n = pts.len();
            if !(950..=1150).contains(&n) {
                return;
            }
            let x0 = pts.iter().map(|p| p.x).fold(f64::INFINITY, f64::min);
            let x1 = pts.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max);
            let y0 = pts.iter().map(|p| p.y).fold(f64::INFINITY, f64::min);
            let y1 = pts.iter().map(|p| p.y).fold(f64::NEG_INFINITY, f64::max);
            let w = x1 - x0;
            let h = y1 - y0;
            if (11.5f64..15.0).contains(&w) && (11.5f64..15.0).contains(&h) {
                println!(
                    "1052-SIG {tag} L{i} n={n} x[{x0:.2}..{x1:.2}] y[{y0:.2}..{y1:.2}] w={w:.2} h={h:.2}"
                );
            }
        };
        for (k, l) in layer.infill_boundary.iter().enumerate() {
            scan(&format!("infill{k}"), l);
        }
        for w in &layer.loops {
            scan(&format!("wall{}", w.wall_index), &w.points);
        }
    }
    // ---- Part 6: in-air wall-loop audit for the top layers ----
    use manifold_fidget::ScalarField;
    println!("\n--- Part 6: wall loops with >=3 pts in deep-hole interior (world x171..179 y171..179 z23..29), all layers ---");
    let mut total_in_box = 0usize;
    for (i, layer) in layers.iter().enumerate() {
        for w in &layer.loops {
            if w.points.is_empty() {
                continue;
            }
            let mut in_box = 0usize;
            let mut air = 0usize;
            let mut in_box_sdfs: Vec<f64> = Vec::new();
            for p in &w.points {
                let sv = sdf.sample(*p).value;
                if (171.0f64..179.0).contains(&p.x)
                    && (171.0f64..179.0).contains(&p.y)
                    && (23.0f64..29.0).contains(&p.z)
                {
                    in_box += 1;
                    in_box_sdfs.push(sv);
                }
                if sv > 0.0 {
                    air += 1;
                }
            }
            if in_box >= 3 {
                total_in_box += 1;
                let air_frac = air as f64 / w.points.len() as f64;
                let z0 = w.points.iter().map(|p| p.z).fold(f64::INFINITY, f64::min);
                let z1 = w
                    .points
                    .iter()
                    .map(|p| p.z)
                    .fold(f64::NEG_INFINITY, f64::max);
                let smin = in_box_sdfs.iter().cloned().fold(f64::INFINITY, f64::min);
                let smax = in_box_sdfs
                    .iter()
                    .cloned()
                    .fold(f64::NEG_INFINITY, f64::max);
                println!(
                    "L{i} o={:.2} w{} n={} in_box={in_box} air={air_frac:.2} smin={smin:.3} smax={smax:.3} z[{z0:.2}..{z1:.2}]",
                    layer.order,
                    w.wall_index,
                    w.points.len()
                );
            }
        }
    }
    println!("total loops with >=3 pts in slot-air box: {total_in_box}");
    // ---- Part 7: hole-interior occupancy at the hole layers ----
    println!("\n--- Part 7: hole-interior occupancy at layers order 24.0..28.5 ---");
    let hole_pts: [(f64, f64); 5] = [
        (175.0, 175.0),
        (173.0, 173.0),
        (177.0, 177.0),
        (173.0, 177.0),
        (177.0, 173.0),
    ];
    for (i, layer) in layers.iter().enumerate() {
        if !(24.0f64..28.5).contains(&layer.order) {
            continue;
        }
        let mut infill_hits = 0usize;
        for (hx, hy) in hole_pts {
            let n_inside = layer
                .infill_boundary
                .iter()
                .filter(|lp| poly_contains_2d(lp, hx, hy))
                .count();
            if n_inside % 2 == 1 {
                infill_hits += 1;
            }
        }
        let mut solid_hits = 0usize;
        for (hx, hy) in hole_pts {
            let n_inside = layer
                .solid_fill_boundary
                .iter()
                .filter(|lp| poly_contains_2d(lp, hx, hy))
                .count();
            if n_inside % 2 == 1 {
                solid_hits += 1;
            }
        }
        println!(
            "L{i} o={:.2} infill_hits={infill_hits}/5 solid_hits={solid_hits}/5 infill_loops={} solid_loops={} wall_loops={}",
            layer.order,
            layer.infill_boundary.len(),
            layer.solid_fill_boundary.len(),
            layer.loops.len()
        );
    }
    // ---- Part 8: plan toolpaths from existing layers; kind of hole-interior segs ----
    use manifold_core::toolpath;
    let slope_profile = profile.machine.slope_profile();
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
    let mut kind_counts: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    let mut order_hist: std::collections::BTreeMap<String, usize> =
        std::collections::BTreeMap::new();
    let mut total = 0usize;
    let total_paths = paths.len();
    let mut total_all = 0usize;
    for path in &paths {
        let pts = &path.points;
        if pts.is_empty() {
            continue;
        }
        for (i, seg) in path.segments.iter().enumerate() {
            if seg.extrusion_length <= 0.0 {
                continue;
            }
            total_all += 1;
            let p = pts[i % pts.len()];
            let end = pts[(i + 1) % pts.len()];
            let mid = p.lerp(end, 0.5);
            let in_box = |q: glam::DVec3| {
                (171.0f64..179.0).contains(&q.x)
                    && (171.0f64..179.0).contains(&q.y)
                    && (23.0f64..29.0).contains(&q.z)
            };
            if in_box(p) || in_box(mid) {
                total += 1;
                let k = format!("{:?}", seg.kind);
                *kind_counts.entry(k.clone()).or_insert(0) += 1;
                *order_hist.entry(format!("{:.2}", seg.order)).or_insert(0) += 1;
            }
        }
    }
    println!(
        "\n--- Part 8: extruded segs in deep-hole interior (world x171-179 y171-179 z23-29) ---"
    );
    println!("total paths: {total_paths}, total extruded segs (all): {total_all}");
    println!("extruded segs in box (start or mid): {total}");
    for (kind, count) in &kind_counts {
        println!("  kind {kind}: {count}");
    }
    for (order, count) in &order_hist {
        println!("  order {order}: {count}");
    }
    // ---- Part 8b: hole x/y column, z 22..31, per-order ----
    let mut total_xy = 0usize;
    let mut xy_kind: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut xy_order_hist: std::collections::BTreeMap<String, usize> =
        std::collections::BTreeMap::new();
    for path in &paths {
        let pts = &path.points;
        if pts.is_empty() {
            continue;
        }
        for (i, seg) in path.segments.iter().enumerate() {
            if seg.extrusion_length <= 0.0 {
                continue;
            }
            let p = pts[i % pts.len()];
            let end = pts[(i + 1) % pts.len()];
            let mid = p.lerp(end, 0.5);
            let in_xy = |q: glam::DVec3| {
                (171.0f64..179.0).contains(&q.x)
                    && (171.0f64..179.0).contains(&q.y)
                    && (22.0f64..31.0).contains(&q.z)
            };
            if in_xy(p) || in_xy(mid) {
                total_xy += 1;
                let k = format!("{:?}", seg.kind);
                *xy_kind.entry(k).or_insert(0) += 1;
                *xy_order_hist
                    .entry(format!("{:.2}", seg.order))
                    .or_insert(0) += 1;
            }
        }
    }
    println!("\\n--- Part 8b: segs in hole x/y column (x171-179 y171-179 z22-31) ---");
    println!("total: {total_xy}");
    for (order, count) in &xy_order_hist {
        println!("  order {order}: {count}");
    }
    for (kind, count) in &xy_kind {
        println!("  kind {kind}: {count}");
    }

    // ---- Part 9: per-loop parity audit at the 5/5 layers ----
    println!("\n--- Part 9: per-loop parity at hole layers ---");
    for (i, layer) in layers.iter().enumerate() {
        if !(24.0f64..25.5).contains(&layer.order) && !(26.0f64..28.5).contains(&layer.order) {
            continue;
        }
        let n_inside = layer
            .infill_boundary
            .iter()
            .filter(|lp| poly_contains_2d(lp, 175.0, 175.0))
            .count();
        println!(
            "L{i} o={:.2} loops={} center_inside={n_inside} parity={}",
            layer.order,
            layer.infill_boundary.len(),
            n_inside % 2 == 1
        );
        for (j, lp) in layer.infill_boundary.iter().enumerate() {
            let mut x0 = f64::INFINITY;
            let mut x1 = f64::NEG_INFINITY;
            let mut y0 = f64::INFINITY;
            let mut y1 = f64::NEG_INFINITY;
            for p in lp {
                x0 = x0.min(p.x);
                x1 = x1.max(p.x);
                y0 = y0.min(p.y);
                y1 = y1.max(p.y);
            }
            let inside = poly_contains_2d(lp, 175.0, 175.0);
            println!(
                "  [{j}] n={} x[{x0:.2}..{x1:.2}] y[{y0:.2}..{y1:.2}] center_in={inside}",
                lp.len()
            );
        }
    }

    // ---- Part 10: figure-8 forensics on the L151/L153 outer loop ----
    println!("\\n--- Part 10: pinched-outer forensics ---");
    let mut analyzed = 0usize;
    for (i, layer) in layers.iter().enumerate() {
        if analyzed >= 2 {
            break;
        }
        if !matches!(layer.order, 26.3..26.5 | 26.7..26.9) {
            continue;
        }
        analyzed += 1;
        let lp = layer
            .infill_boundary
            .iter()
            .max_by_key(|l| l.len())
            .expect("biggest infill loop");
        let n = lp.len();
        // (a) min distance from center to any loop segment
        let mut dmin2 = f64::INFINITY;
        let c = glam::DVec3::new(175.0, 175.0, 0.0);
        for k in 0..n {
            let a = lp[k];
            let b = lp[(k + 1) % n];
            let ab = b - a;
            let ab2 = ab.length_squared();
            let t = if ab2 > 0.0 {
                ((c - a).dot(ab) / ab2).clamp(0.0, 1.0)
            } else {
                0.0
            };
            dmin2 = dmin2.min((a + ab * t - c).length_squared());
        }
        println!(
            "L{i} o={:.4} outer n={n} min_dist_center={} mm",
            layer.order,
            dmin2.sqrt()
        );
        // (b) points in the central 8x8mm window
        let mut central: Vec<usize> = (0..n)
            .filter(|&k| {
                let p = lp[k];
                (171.0..179.0).contains(&p.x) && (171.0..179.0).contains(&p.y)
            })
            .collect();
        central.truncate(40);
        println!("L{i} central-window points (of {n}): {}", central.len());
        for &k in central.iter() {
            let p = lp[k];
            println!("  [{k}] ({:.3}, {:.3}, {:.3})", p.x, p.y, p.z);
        }
        // (c) even-odd parity map on a 7x7 grid over 165..185
        let grid = [165.0, 169.0, 173.0, 175.0, 177.0, 181.0, 185.0];
        println!("L{i} even-odd map over this loop only (1=in, 0=out):");
        for &gy in grid.iter().rev() {
            let mut row = String::new();
            for &gx in &grid {
                row.push(if poly_contains_2d(lp, gx, gy) {
                    '1'
                } else {
                    '0'
                });
            }
            println!("  y={gy:.0} {row}");
        }
        // (d) cusp / self-touch scan: sharpest direction reversals and zero-length segs
        let mut reversals: Vec<(f64, usize)> = Vec::new();
        let mut zero_segs = 0usize;
        for k in 0..n {
            let a = lp[(k + n - 1) % n];
            let p = lp[k];
            let b = lp[(k + 1) % n];
            let in_dir = p - a;
            let out_dir = b - p;
            if in_dir.length() < 1e-6 || out_dir.length() < 1e-6 {
                zero_segs += 1;
                continue;
            }
            let dot = in_dir.dot(out_dir) / (in_dir.length() * out_dir.length());
            if dot < -0.5 {
                reversals.push((dot, k));
            }
        }
        reversals.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
        println!(
            "L{i} zero-length segs: {zero_segs}; sharp reversals (dot<-0.5): {}",
            reversals.len()
        );
        for (dot, k) in reversals.iter().take(6) {
            let p = lp[*k];
            println!(
                "  [{k}] dot={dot:.4} at ({:.3}, {:.3}, {:.3})",
                p.x, p.y, p.z
            );
        }
    }

    // ---- Part 10b: slot-cap layer parity (order 23.0..24.0) ----
    println!("\\n--- Part 10b: slot-cap parity (center 175,175) ---");
    for (i, layer) in layers.iter().enumerate() {
        if !(21.3f64..22.0).contains(&layer.order) {
            continue;
        }
        let containing: Vec<(usize, usize)> = layer
            .infill_boundary
            .iter()
            .enumerate()
            .filter(|(_, lp)| poly_contains_2d(lp, 175.0, 175.0))
            .map(|(j, lp)| (j, lp.len()))
            .collect();
        let parity: usize = containing.len();
        println!(
            "L{i} o={:.2} loops={} center_loops={:?} parity={}",
            layer.order,
            layer.infill_boundary.len(),
            containing,
            parity % 2
        );
    }
    // ---- Part 10c: SDF sign column at the 5 hole test points ----
    println!("\\n--- Part 10c: SDF column at test points (z 20.5..30.5) ---");
    for (px, py) in [
        (173.0, 173.0),
        (175.0, 175.0),
        (177.0, 177.0),
        (173.0, 177.0),
        (177.0, 173.0),
    ] {
        let mut row = String::new();
        row.push_str(&format!("({:.0},{:.0}) ", px, py));
        let mut z = 20.5f64;
        while z <= 30.51 {
            let v = sdf.sample(glam::DVec3::new(px, py, z)).value;
            row.push(if v.is_finite() {
                if v < 0.0 {
                    'M'
                } else {
                    'A'
                }
            } else {
                '.'
            });
            z += 0.5;
        }
        println!("{}  (z=20.5..30.5 step .5, M=material A=air)", row);
    }

    Ok(())
}

fn polygon_area_3d(loop_pts: &[glam::DVec3]) -> f64 {
    // Shoelace area in XY (world coords; object placement is a pure
    // translation, so world-XY area == object-XY area).
    let mut a: f64 = 0.0;
    for w in loop_pts.windows(2) {
        a += w[0].x * w[1].y - w[1].x * w[0].y;
    }
    a.abs() * 0.5
}

/// Even-odd point-in-polygon test in world XY for a 3D loop.
fn poly_contains_2d(loop_pts: &[glam::DVec3], px: f64, py: f64) -> bool {
    let n = loop_pts.len();
    if n < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let xi = loop_pts[i].x;
        let yi = loop_pts[i].y;
        let xj = loop_pts[j].x;
        let yj = loop_pts[j].y;
        if (yi > py) != (yj > py) {
            let xint = (xj - xi) * (py - yi) / (yj - yi) + xi;
            if px < xint {
                inside = !inside;
            }
        }
        j = i;
    }
    inside
}
