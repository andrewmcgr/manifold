//! Diagnostic probe: inspect order 15.2 for Thingy2.stl with profile.json

use manifold_core::ids::{ObjectId, ToolId};
use manifold_core::machine::Machine;
use manifold_core::object::Object;
use manifold_core::polygon2d;
use manifold_core::toolpath;
use manifold_core::wave_overhang;
use manifold_core::{slicing, stl, SlicerConfig};
use manifold_fidget::slope_profile::SlopeProfile;
use manifold_fidget::ScalarField;

#[derive(Debug, serde::Deserialize)]
struct Profile {
    machine: Machine,
    config: SlicerConfig,
}

fn main() -> anyhow::Result<()> {
    let profile_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/Users/amcgregor/3D/profile.json".to_string());
    let mesh_path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "/Users/amcgregor/3D/Thingy2.stl".to_string());

    let profile_json = std::fs::read_to_string(&profile_path)?;
    let profile: Profile = serde_json::from_str(&profile_json)?;
    let config = profile.config;
    let machine = profile.machine;

    let mesh = stl::load_stl(std::fs::File::open(&mesh_path)?)?;
    println!("\nMesh bounding box: {:?}", mesh.bounding_box());
    let mut min_z = f64::INFINITY;
    let mut max_z = f64::NEG_INFINITY;
    for v in &mesh.vertices {
        min_z = min_z.min(v.z);
        max_z = max_z.max(v.z);
    }
    println!("Mesh Z range: [{:.2}, {:.2}]", min_z, max_z);

    let mut objects = vec![Object::new(ObjectId(0), mesh, ToolId(0))];
    manifold_core::object::center_on_bed(&mut objects, &machine.build_volume);

    let slope_profile = SlopeProfile::new(machine.eikonal_slope_profile.clone());
    let layers = slicing::slice_workspace_with_progress(
        &objects,
        &[ObjectId(0)],
        &config,
        &slope_profile,
        &mut |_| {},
    )?;

    // Find layer near order 15.2
    let l75_idx = layers
        .iter()
        .position(|l| (l.order - 15.2).abs() < 1e-3)
        .expect("layer 15.2 not found");
    let l74_idx = l75_idx - 1;

    let l75 = &layers[l75_idx];
    let l74 = &layers[l74_idx];

    println!(
        "Layer {} (order {:.3}) has {} loops, infill_boundary: {} loops, solid_fill_boundary: {} loops",
        l74.index,
        l74.order,
        l74.loops.len(),
        l74.infill_boundary.len(),
        l74.solid_fill_boundary.len()
    );
    println!(
        "Layer {} (order {:.3}) has {} loops, infill_boundary: {} loops, solid_fill_boundary: {} loops",
        l75.index,
        l75.order,
        l75.loops.len(),
        l75.infill_boundary.len(),
        l75.solid_fill_boundary.len()
    );

    for (i, w) in l75.loops.iter().enumerate() {
        let mut min = glam::DVec3::splat(f64::INFINITY);
        let mut max = glam::DVec3::splat(f64::NEG_INFINITY);
        for p in &w.points {
            min = min.min(*p);
            max = max.max(*p);
        }
        println!(
            "  L75 loop {}: wall_index={}, island={}, pts={}, min={:.2?}, max={:.2?}, is_open={}",
            i,
            w.wall_index,
            w.island,
            w.points.len(),
            min,
            max,
            w.is_open
        );
    }

    let (axis, apex, _) =
        manifold_core::order_field::resolve_axis_apex_slope(config.order_field, &config);
    let (basis1, basis2) = manifold_fidget::contour::plane_basis(axis);

    let l75_w0: Vec<Vec<glam::DVec3>> = l75
        .loops
        .iter()
        .filter(|w| w.wall_index == 0)
        .map(|w| w.points.clone())
        .collect();
    let l74_w0: Vec<Vec<glam::DVec3>> = l74
        .loops
        .iter()
        .filter(|w| w.wall_index == 0)
        .map(|w| w.points.clone())
        .collect();

    println!("\nLayer 74 wall0 loops:");
    for (i, pts) in l74_w0.iter().enumerate() {
        let mut min = glam::DVec3::splat(f64::INFINITY);
        let mut max = glam::DVec3::splat(f64::NEG_INFINITY);
        for p in pts {
            min = min.min(*p);
            max = max.max(*p);
        }
        println!(
            "  L74 loop {}: pts={}, min={:.2?}, max={:.2?}",
            i,
            pts.len(),
            min,
            max
        );
    }

    let cur_b = polygon2d::canonicalize(&polygon2d::to_2d(&l75_w0, basis1, basis2, apex));
    let prev_b = polygon2d::canonicalize(&polygon2d::to_2d(&l74_w0, basis1, basis2, apex));

    println!(
        "cur_b loops: {}, prev_b loops: {}",
        cur_b.len(),
        prev_b.len()
    );
    let raw_overhang = polygon2d::difference(&cur_b, &prev_b);
    println!("raw_overhang loops: {}", raw_overhang.len());
    for (i, loop_2d) in raw_overhang.iter().enumerate() {
        let area = polygon2d::signed_area(loop_2d);
        println!(
            "  overhang loop {}: area = {:.4}, pts = {}",
            i,
            area,
            loop_2d.len()
        );
    }

    let min_overhang_area = 0.25 * config.nozzle_diameter * config.nozzle_diameter;
    let filtered = polygon2d::filter_min_area(&raw_overhang, min_overhang_area);
    println!(
        "filtered overhang loops (>= {:.4}): {}",
        min_overhang_area,
        filtered.len()
    );

    let shapes = wave_overhang::group_loops_into_polygon_shapes(&filtered);
    println!("Grouped into {} polygon shape(s)", shapes.len());

    let max_along = manifold_core::order_field::max_along_for(&config);

    for (si, shape) in shapes.iter().enumerate() {
        let mut min_pt = [f64::INFINITY, f64::INFINITY];
        let mut max_pt = [f64::NEG_INFINITY, f64::NEG_INFINITY];
        for &[u, v] in &shape.outer {
            min_pt[0] = min_pt[0].min(u);
            min_pt[1] = min_pt[1].min(v);
            max_pt[0] = max_pt[0].max(u);
            max_pt[1] = max_pt[1].max(v);
        }
        println!(
            "Shape {}: outer pts={}, holes={}, bbox: u=[{:.2}, {:.2}], v=[{:.2}, {:.2}]",
            si,
            shape.outer.len(),
            shape.holes.len(),
            min_pt[0],
            max_pt[0],
            min_pt[1],
            max_pt[1]
        );
        let mut c_u = 0.0;
        let mut c_v = 0.0;
        for &[u, v] in &shape.outer {
            c_u += u;
            c_v += v;
        }
        let len = shape.outer.len().max(1) as f64;
        let c_u = c_u / len;
        let c_v = c_v / len;

        println!("  outer centroid: u={:.2}, v={:.2}", c_u, c_v);
        let pt_in_outer = wave_overhang::PolygonShape2D {
            outer: shape.outer.clone(),
            holes: shape.holes.clone(),
        }
        .contains_point([c_u, c_v]);
        println!("  contains_point(centroid) = {}", pt_in_outer);

        let without_near = manifold_core::order_field::reconstruct_on_order_field(
            vec![vec![[c_u, c_v]]],
            basis1,
            basis2,
            axis,
            apex,
            l75.order,
            max_along,
            l75.order_field.as_ref(),
        );
        if let Some(p_3d) = without_near.first().and_then(|pts| pts.first()) {
            if let Some(sdf) = &l75.mesh_sdf {
                let sdf_val = sdf.sample(*p_3d).value;
                println!(
                    "  WITHOUT NEAR: p_3d={:.2?}, sdf.sample(p_3d).value = {:.4}",
                    p_3d, sdf_val
                );
            }
        } else {
            println!("  WITHOUT NEAR: reconstruct returned empty!");
        }

        // Check seed segments
        let mut seed_segments = Vec::new();
        let n = shape.outer.len();
        let search_dist = (config.nozzle_diameter * 1.25).max(0.4);
        for i in 0..n {
            let p0 = shape.outer[i];
            let p1 = shape.outer[(i + 1) % n];
            let mid = [(p0[0] + p1[0]) * 0.5, (p0[1] + p1[1]) * 0.5];
            let near_prev = polygon2d_contains_or_near(mid, &prev_b, search_dist);
            if near_prev {
                seed_segments.push((p0, p1));
            }
        }
        println!("  seed segments bordering prev_b: {}", seed_segments.len());

        let wavelength = (config.nozzle_diameter - config.wave_overhang_overlap()).max(0.10);
        let seeds_2d: Vec<wave_overhang::LineSegment2D> = seed_segments
            .iter()
            .map(|&(p0, p1)| wave_overhang::LineSegment2D { p0, p1 })
            .collect();
        let polylines =
            wave_overhang::generate_wave_overhang_paths_2d(shape, &seeds_2d, wavelength, &config);
        println!("  generated wave polylines 2D: {}", polylines.len());
        for (pi, poly) in polylines.iter().enumerate() {
            println!("    poly {}: pts={}", pi, poly.len());
        }
    }

    println!("\nTesting reconstruction of Shape 0 and Shape 3 wave polylines with reconstruct_on_order_field_near:");
    for (si, shape) in [(0, &shapes[0]), (3, &shapes[3])] {
        let wavelength = (config.nozzle_diameter - config.wave_overhang_overlap()).max(0.10);
        let n = shape.outer.len();
        let search_dist = (config.nozzle_diameter * 1.25).max(0.4);
        let mut seed_segments = Vec::new();
        for i in 0..n {
            let p0 = shape.outer[i];
            let p1 = shape.outer[(i + 1) % n];
            let mid = [(p0[0] + p1[0]) * 0.5, (p0[1] + p1[1]) * 0.5];
            if polygon2d_contains_or_near(mid, &prev_b, search_dist) {
                seed_segments.push(wave_overhang::LineSegment2D { p0, p1 });
            }
        }
        let polylines = wave_overhang::generate_wave_overhang_paths_2d(
            shape,
            &seed_segments,
            wavelength,
            &config,
        );
        let references: Vec<Vec<glam::DVec3>> = l75
            .loops
            .iter()
            .filter(|w| w.wall_index == 0)
            .map(|w| w.points.clone())
            .collect();
        let polylines_3d = manifold_core::order_field::reconstruct_on_order_field_near(
            polylines,
            &references,
            basis1,
            basis2,
            axis,
            apex,
            l75.order,
            max_along,
            l75.order_field.as_ref(),
        );
        println!(
            "  Shape {}: generated {} 3D wave polylines",
            si,
            polylines_3d.len()
        );
        let mut valid_in_solid = 0;
        for (pi, p3d) in polylines_3d.iter().enumerate() {
            let in_solid = p3d.iter().all(|p| {
                l75.mesh_sdf
                    .as_ref()
                    .map(|s| s.sample(*p).value <= 0.35)
                    .unwrap_or(true)
            });
            if in_solid {
                valid_in_solid += 1;
            } else {
                let max_sdf = p3d
                    .iter()
                    .map(|p| {
                        l75.mesh_sdf
                            .as_ref()
                            .map(|s| s.sample(*p).value)
                            .unwrap_or(0.0)
                    })
                    .fold(f64::NEG_INFINITY, f64::max);
                println!(
                    "    poly {}: pts={}, NOT in solid, max_sdf={:.4}",
                    pi,
                    p3d.len(),
                    max_sdf
                );
            }
        }
        println!(
            "  Shape {}: {} / {} wave polylines are within solid tolerance",
            si,
            valid_in_solid,
            polylines_3d.len()
        );
    }

    println!("\nAnalyzing bridge topology of Shape 0:");
    let shape0 = &shapes[0];
    let n0 = shape0.outer.len();
    let search_dist = (config.nozzle_diameter * 1.25).max(0.4);
    let mut contact_runs = Vec::new();
    let mut current_run = Vec::new();
    let mut in_contact = false;
    for i in 0..n0 {
        let p0 = shape0.outer[i];
        let p1 = shape0.outer[(i + 1) % n0];
        let mid = [(p0[0] + p1[0]) * 0.5, (p0[1] + p1[1]) * 0.5];
        let near = polygon2d_contains_or_near(mid, &prev_b, search_dist);
        if near {
            current_run.push(i);
            in_contact = true;
        } else if in_contact {
            contact_runs.push(current_run.clone());
            current_run.clear();
            in_contact = false;
        }
    }
    if !current_run.is_empty() {
        contact_runs.push(current_run);
    }
    // Handle wrap-around
    if contact_runs.len() > 1
        && contact_runs.first().unwrap().contains(&0)
        && contact_runs.last().unwrap().contains(&(n0 - 1))
    {
        let last = contact_runs.pop().unwrap();
        contact_runs[0].extend(last);
    }
    println!(
        "Shape 0 has {} distinct contact anchor group(s)",
        contact_runs.len()
    );
    for (gi, group) in contact_runs.iter().enumerate() {
        let mut min = [f64::INFINITY, f64::INFINITY];
        let mut max = [f64::NEG_INFINITY, f64::NEG_INFINITY];
        for &idx in group {
            let p0 = shape0.outer[idx];
            let p1 = shape0.outer[(idx + 1) % n0];
            min[0] = min[0].min(p0[0]).min(p1[0]);
            min[1] = min[1].min(p0[1]).min(p1[1]);
            max[0] = max[0].max(p0[0]).max(p1[0]);
            max[1] = max[1].max(p0[1]).max(p1[1]);
        }
        println!(
            "  Anchor group {}: {} segs, bbox: u=[{:.2}, {:.2}], v=[{:.2}, {:.2}]",
            gi,
            group.len(),
            min[0],
            max[0],
            min[1],
            max[1]
        );
    }
    let plan = wave_overhang::plan_wave_overhangs(&layers, &config, ToolId(0));
    let l75_paths = &plan.paths_by_layer[l75_idx];
    println!(
        "plan_wave_overhangs produced {} paths for layer 75",
        l75_paths.len()
    );
    for (i, p) in l75_paths.iter().enumerate() {
        println!("  wave path {}: pts={}", i, p.points.len());
    }
    println!("\nWall points unsupported tag counts for layer 75 with fixed plan:");
    for (wi, wall) in l75.loops.iter().enumerate() {
        let tags = &plan.wall_overhang_tags_by_layer[l75_idx][wi];
        let tagged_count = tags.iter().filter(|&&t| t).count();
        if tagged_count > 0 {
            println!(
                "  wall {}: wall_index={}, island={}, total_pts={}, tagged_overhang={}",
                wi,
                wall.wall_index,
                wall.island,
                wall.points.len(),
                tagged_count
            );
        }
    }

    println!("\nScanning all layers for overhangs and bridge conditions...");
    for k in 1..layers.len() {
        let prev_k = k - 1;
        let l_cur = &layers[k];
        let l_prev = &layers[prev_k];
        let w0_cur: Vec<Vec<glam::DVec3>> = l_cur
            .loops
            .iter()
            .filter(|w| w.wall_index == 0)
            .map(|w| w.points.clone())
            .collect();
        let w0_prev: Vec<Vec<glam::DVec3>> = l_prev
            .loops
            .iter()
            .filter(|w| w.wall_index == 0)
            .map(|w| w.points.clone())
            .collect();
        if w0_cur.is_empty() || w0_prev.is_empty() {
            continue;
        }
        let cur_b = polygon2d::canonicalize(&polygon2d::to_2d(&w0_cur, basis1, basis2, apex));
        let prev_b = polygon2d::canonicalize(&polygon2d::to_2d(&w0_prev, basis1, basis2, apex));
        let raw = polygon2d::difference(&cur_b, &prev_b);
        let filtered = polygon2d::filter_min_area(&raw, min_overhang_area);
        if filtered.is_empty() {
            continue;
        }
        let shapes = wave_overhang::group_loops_into_polygon_shapes(&filtered);
        for shape in &shapes {
            let n = shape.outer.len();
            let mut contact_runs = Vec::new();
            let mut current_run = Vec::new();
            let mut in_contact = false;
            for i in 0..n {
                let p0 = shape.outer[i];
                let p1 = shape.outer[(i + 1) % n];
                let mid = [(p0[0] + p1[0]) * 0.5, (p0[1] + p1[1]) * 0.5];
                let near = polygon2d_contains_or_near(mid, &prev_b, search_dist);
                if near {
                    current_run.push(i);
                    in_contact = true;
                } else if in_contact {
                    contact_runs.push(current_run.clone());
                    current_run.clear();
                    in_contact = false;
                }
            }
            if !current_run.is_empty() {
                contact_runs.push(current_run);
            }
            if contact_runs.len() > 1
                && contact_runs.first().unwrap().contains(&0)
                && contact_runs.last().unwrap().contains(&(n - 1))
            {
                let last = contact_runs.pop().unwrap();
                contact_runs[0].extend(last);
            }
            if contact_runs.len() >= 2 {
                println!("  Layer {:3} (order {:6.3}): overhang area={:6.2} has {} SEPARATED CONTACT ANCHORS! (BRIDGE CANDIDATE)",
                    k, l_cur.order, polygon2d::signed_area(&shape.outer), contact_runs.len());
            }
        }
    }

    let overhang_footprint_2d = filtered.clone();

    // Infill: clip against overhang footprint
    let infill_2d = polygon2d::to_2d(&l75.infill_boundary, basis1, basis2, apex);
    let diff_infill_2d = polygon2d::difference(&infill_2d, &overhang_footprint_2d);
    let diff_infill_3d = polygon2d::from_2d(diff_infill_2d, basis1, basis2, apex);
    let infill_pts: usize = diff_infill_3d.iter().map(|l| l.len()).sum();
    println!("  Infill boundary: original {} pts across {} loops, after clipping: {} pts across {} loops",
        l75.infill_boundary.iter().map(|l| l.len()).sum::<usize>(), l75.infill_boundary.len(),
        infill_pts, diff_infill_3d.len());
    let tools = machine.tools.clone();
    let paths = toolpath::plan_with_progress(
        &layers,
        &objects,
        &tools,
        &config,
        None,
        &slope_profile,
        &mut |_| {},
    )?;

    println!("\nExtrusion segments on layer 75 crossing the void (X between 170.0 and 180.0):");
    let mut void_segments_by_kind: std::collections::BTreeMap<String, usize> =
        std::collections::BTreeMap::new();
    for (pi, p) in paths.iter().enumerate() {
        let n = p.points.len();
        if n < 2 {
            continue;
        }
        for (si, s) in p.segments.iter().enumerate() {
            if (s.order - l75.order).abs() > 1e-6 {
                continue;
            }
            if s.kind == toolpath::MoveKind::Travel {
                continue;
            }
            let p0 = p.points[si];
            let p1 = p.points[(si + 1) % n];
            let mid_x = (p0.x + p1.x) * 0.5;
            if mid_x > 170.0 && mid_x < 180.0 {
                *void_segments_by_kind
                    .entry(format!("{:?}", s.kind))
                    .or_default() += 1;
                if void_segments_by_kind.values().sum::<usize>() <= 20 {
                    println!(
                        "  Path {} seg {} ({:?}): p0={:.2?}, p1={:.2?}, mid_x={:.2}",
                        pi, si, s.kind, p0, p1, mid_x
                    );
                }
            }
        }
    }
    println!(
        "Total void segments by kind on layer 75: {:?}",
        void_segments_by_kind
    );

    println!("\nTesting clipping of inner walls against overhang footprint:");
    for w_idx in 0..l75.loops.len() {
        let wall = &l75.loops[w_idx];
        if wall.wall_index > 0 {
            let w_2d = polygon2d::to_2d(std::slice::from_ref(&wall.points), basis1, basis2, apex);
            let diff_2d = polygon2d::difference(&w_2d, &filtered);
            let diff_3d = manifold_core::order_field::reconstruct_on_order_field_near(
                diff_2d,
                &l75_w0,
                basis1,
                basis2,
                axis,
                apex,
                l75.order,
                max_along,
                l75.order_field.as_ref(),
            );
            let pts_count: usize = diff_3d.iter().map(|l| l.len()).sum();
            println!(
                "  w{}: original {} pts, clipped against void: {} pts across {} piece(s)",
                wall.wall_index,
                wall.points.len(),
                pts_count,
                diff_3d.len()
            );
            for (pi, piece) in diff_3d.iter().enumerate() {
                let mut min_x = f64::INFINITY;
                let mut max_x = f64::NEG_INFINITY;
                for p in piece {
                    min_x = min_x.min(p.x);
                    max_x = max_x.max(p.x);
                }
                println!(
                    "    piece {}: pts={}, x=[{:.2}, {:.2}]",
                    pi,
                    piece.len(),
                    min_x,
                    max_x
                );
            }
        }
    }
    let p2491 = &paths[2491];
    let n2491 = p2491.points.len();
    for (si, s) in p2491.segments.iter().enumerate() {
        let p0 = p2491.points[si];
        let p1 = p2491.points[(si + 1) % n2491];
        let mid_x = (p0.x + p1.x) * 0.5;
        if mid_x > 170.0 && mid_x < 180.0 {
            println!(
                "  seg {}: {:?}, p0=[{:.2}, {:.2}, {:.2}], p1=[{:.2}, {:.2}, {:.2}], len={:.2}",
                si,
                s.kind,
                p0.x,
                p0.y,
                p0.z,
                p1.x,
                p1.y,
                p1.z,
                p0.distance(p1)
            );
        }
    }
    let l78 = layers
        .iter()
        .find(|l| (l.order - 15.8).abs() < 1e-3)
        .unwrap();
    for (pi, p) in paths.iter().enumerate() {
        let matching: Vec<_> = p
            .segments
            .iter()
            .filter(|s| (s.order - l78.order).abs() < 1e-6)
            .collect();
        if matching
            .iter()
            .any(|s| s.kind == toolpath::MoveKind::DebugExcluded)
        {
            println!(
                "  Path {}: pts={}, kinds={:?}",
                pi,
                p.points.len(),
                matching.iter().fold(
                    std::collections::BTreeMap::<String, usize>::new(),
                    |mut acc, s| {
                        *acc.entry(format!("{:?}", s.kind)).or_default() += 1;
                        acc
                    }
                )
            );
            for pt in &p.points {
                let d = l78
                    .mesh_sdf
                    .as_ref()
                    .map(|s| s.sample(*pt).value)
                    .unwrap_or(0.0);
                if d > 0.35 {
                    println!("    pt outside mesh: pt={:.2?}, sdf={:.4}", pt, d);
                }
            }
        }
    }

    Ok(())
}

fn polygon2d_contains_or_near(pt: [f64; 2], loops: &[Vec<[f64; 2]>], eps: f64) -> bool {
    let eps_sq = eps * eps;
    for loop_ in loops {
        if point_in_single_loop(pt, loop_) {
            return true;
        }
        let n = loop_.len();
        for i in 0..n {
            let seg = wave_overhang::LineSegment2D {
                p0: loop_[i],
                p1: loop_[(i + 1) % n],
            };
            if seg.dist_sq_to_point(pt) <= eps_sq {
                return true;
            }
        }
    }
    false
}

fn point_in_single_loop(pt: [f64; 2], loop_: &[[f64; 2]]) -> bool {
    if loop_.len() < 3 {
        return false;
    }
    let [x, y] = pt;
    let mut inside = false;
    let mut j = loop_.len() - 1;
    for i in 0..loop_.len() {
        let [xi, yi] = loop_[i];
        let [xj, yj] = loop_[j];
        let intersect = ((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
        if intersect {
            inside = !inside;
        }
        j = i;
    }
    inside
}
