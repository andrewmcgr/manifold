//! Scratch diagnostic: locates upward-facing flat top surfaces, then samples
//! the AnisotropicFsm order field along vertical columns beneath their
//! centroids to check for non-monotonicity along BUILD_DIRECTION (Z).
//! Usage: cargo run --release --bin probe_top_seeds -- <mesh.stl> <profile.json>

use manifold_core::{order_field, stl, SlicerConfig};
use manifold_fidget::mesh_sdf::MeshSdf;
use std::fs;
use std::io::Cursor;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mesh_path = &args[1];
    let profile_path = &args[2];

    let mesh_data = fs::read(mesh_path).expect("read mesh");
    let mesh = stl::load_stl(Cursor::new(mesh_data)).expect("parse stl");

    let config_json = fs::read_to_string(profile_path).expect("read profile");
    let val: serde_json::Value = serde_json::from_str(&config_json).expect("parse json");
    let config: SlicerConfig = if val.get("config").is_some() {
        serde_json::from_value(val["config"].clone()).expect("parse config subfield")
    } else {
        serde_json::from_value(val).expect("parse direct config")
    };

    let Some((min, max)) = mesh.bounding_box() else {
        eprintln!("empty mesh");
        return;
    };
    eprintln!("bbox: min={:?} max={:?}", min, max);
    eprintln!(
        "fsm_seed_surfaces_enabled={}",
        config.fsm_seed_surfaces_enabled
    );
    eprintln!("fsm_seed_max_angle_deg={}", config.fsm_seed_max_angle_deg());
    eprintln!(
        "fsm_top_tangency_aspect={}",
        config.fsm_top_tangency_aspect()
    );
    eprintln!("fsm_wall_ortho_aspect={}", config.fsm_wall_ortho_aspect());
    eprintln!("fsm_max_sweeps={}", config.fsm_max_sweeps());

    // Find near-horizontal upward-facing triangles (candidate "flat top" patches),
    // cluster loosely by rounding centroid to 1mm and grouping by z-band.
    let verts = &mesh.vertices;
    let idx = &mesh.indices;
    let mut tops: Vec<(f64, f64, f64, f64)> = Vec::new(); // (cx, cy, cz, area)
    for tri in idx.as_chunks::<3>().0 {
        let a = verts[tri[0] as usize];
        let b = verts[tri[1] as usize];
        let c = verts[tri[2] as usize];
        let n = (b - a).cross(c - a);
        let area = n.length() * 0.5;
        if area < 1e-9 {
            continue;
        }
        let n = n / (n.length());
        // upward-facing within ~5 degrees of vertical
        if n.z > 0.995 {
            let cen = (a + b + c) / 3.0;
            tops.push((cen.x, cen.y, cen.z, area));
        }
    }
    eprintln!("found {} near-horizontal upward triangles", tops.len());

    // Group by z rounded to 0.05mm, sum area, and report top clusters by area.
    use std::collections::BTreeMap;
    let mut by_z: BTreeMap<i64, (f64, f64, f64, usize)> = BTreeMap::new(); // key -> (sum_area, sum_cx*area, sum_cy*area, count)
    for (cx, cy, cz, area) in &tops {
        let key = (cz * 20.0).round() as i64; // 0.05mm buckets
        let e = by_z.entry(key).or_insert((0.0, 0.0, 0.0, 0));
        e.0 += area;
        e.1 += cx * area;
        e.2 += cy * area;
        e.3 += 1;
    }
    let mut clusters: Vec<(f64, f64, f64, f64, usize)> = by_z
        .into_iter()
        .map(|(key, (a, sx, sy, n))| (key as f64 / 20.0, sx / a, sy / a, a, n))
        .collect();
    clusters.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());
    eprintln!("top z-clusters by area (z, centroid_x, centroid_y, area, tri_count):");
    for c in clusters.iter().take(15) {
        eprintln!(
            "  z={:.4} cx={:.3} cy={:.3} area={:.4} n={}",
            c.0, c.1, c.2, c.3, c.4
        );
    }

    // Build the order field once.
    let faces: Vec<[usize; 3]> = idx
        .as_chunks::<3>()
        .0
        .iter()
        .map(|chunk| [chunk[0] as usize, chunk[1] as usize, chunk[2] as usize])
        .collect();
    let sdf = MeshSdf::new(verts.clone(), faces);
    let machine = manifold_core::machine::Machine::default();
    let slope_profile = machine.slope_profile();
    let field = order_field::order_field_for_with_sdf(
        config.order_field,
        &config,
        &mesh,
        &slope_profile,
        Some(&sdf),
    );

    // Sample vertical columns under the largest few clusters (excluding the very
    // top overall max cluster which may just be the mesh apex) and print order()
    // values from just above the bed to just above the cluster height, looking
    // for non-monotonicity.
    let sample_step = 0.05;
    for c in clusters.iter().take(6) {
        let (cz, cx, cy, area, n) = *c;
        if area < 0.5 {
            continue;
        }
        eprintln!(
            "\n--- column under cluster z={:.4} cx={:.3} cy={:.3} area={:.4} n={} ---",
            cz, cx, cy, area, n
        );
        let mut z = min.z + sample_step * 0.5;
        let mut prev: Option<f64> = None;
        let mut last_nonmono_z: Option<f64> = None;
        while z <= (cz + 1.0).min(max.z) {
            let p = glam::DVec3::new(cx, cy, z);
            let val = field.order(p);
            let flag = match prev {
                Some(pv) if val.is_finite() && pv.is_finite() && val < pv - 1e-6 => {
                    last_nonmono_z = Some(z);
                    " <-- NONMONOTONIC (decreased)"
                }
                _ => "",
            };
            if !flag.is_empty() || (z - min.z).abs() < 1e-9 || (z - cz).abs() < sample_step * 2.0 {
                eprintln!("  z={:.4} order={:.5}{}", z, val, flag);
            }
            if val.is_finite() {
                prev = Some(val);
            }
            z += sample_step;
        }
        if let Some(nz) = last_nonmono_z {
            eprintln!("  ** non-monotonicity detected near z={:.4} **", nz);
        } else {
            eprintln!("  (monotonic along this column)");
        }
    }
}
