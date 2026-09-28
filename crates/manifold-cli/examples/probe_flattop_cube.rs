//! Scratch probe: does a plain flat-topped cube get a proper top-surface
//! skin? Slices a generated 20x20x5 cube with the given profile and dumps
//! top layers, per-kind z census, and seed margins at key points.
//!
//! Usage: probe_flattop_cube <profile.json> [--empty-slope]

use glam::DVec3;
use manifold_core::ids::{ObjectId, ToolId};
use manifold_core::machine::Machine;
use manifold_core::mesh::Mesh;
use manifold_core::object::{center_on_bed, Object};
use manifold_core::toolpath::plan_with_progress;

#[derive(serde::Deserialize)]
struct Profile {
    machine: Machine,
    config: manifold_core::SlicerConfig,
}

fn cube_mesh(w: f64, h: f64, d: f64) -> Mesh {
    let x0 = 0.0;
    let y0 = 0.0;
    let z0 = 0.0;
    let x1 = w;
    let y1 = d;
    let z1 = h;
    // 8 vertices
    let v = vec![
        DVec3::new(x0, y0, z0), // 0
        DVec3::new(x1, y0, z0), // 1
        DVec3::new(x1, y1, z0), // 2
        DVec3::new(x0, y1, z0), // 3
        DVec3::new(x0, y0, z1), // 4
        DVec3::new(x1, y0, z1), // 5
        DVec3::new(x1, y1, z1), // 6
        DVec3::new(x0, y1, z1), // 7
    ];
    // outward-facing triangles (normal = (b-a) x (c-a) away from interior)
    let t: Vec<[u32; 3]> = vec![
        [0, 2, 1], // bottom z0 (normal -z): (2-0)x(1-0) = (d,0,0)x(w,0,0) = (0,0,-dw) -> -z ok
        [0, 3, 2],
        [4, 5, 6], // top z1 (normal +z): (5-4)x(6-4)=(w,0,0)x(w,d,0)=(0,0,wd) -> +z ok
        [4, 6, 7],
        [0, 1, 5], // front y0 (normal -y): (1-0)x(5-0)=(w,0,0)x(w,0,h)=(0,-wh,0) ok
        [0, 5, 4],
        [1, 2, 6], // right x1 (normal +x): (2-1)x(6-1)=(0,d,0)x(0,d,h)=(dh,0,0) ok
        [1, 6, 5],
        [2, 3, 7], // back y1 (normal +y): (3-2)x(7-2)=(-w,0,0)x(-w,0,h)=(0,wh,0) ok
        [2, 7, 6],
        [3, 0, 4], // left x0 (normal -x): (0-3)x(4-3)=(w,0,0)x(w,0,h)=(0,-wh,0)...
        [3, 4, 7],
    ];
    let indices: Vec<u32> = t.iter().flatten().copied().collect();
    Mesh::new(v, indices)
}

fn main() -> anyhow::Result<()> {
    let profile_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "examples/profile.json".to_string());
    let mesh = cube_mesh(20.0, 5.0, 20.0);
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
    let paths = plan_with_progress(
        &layers,
        &objects,
        &profile.machine.tools,
        &profile.config,
        Some(&profile.machine),
        &slope_profile,
        &mut |_| {},
    )?;
    let n_layers = layers.len();
    println!(
        "layers={} (cube 20x20x5 at origin, centered on bed)",
        n_layers
    );

    // Part A: top 5 layers
    println!("\n--- Part A: top 5 layers ---");
    for i in (n_layers.saturating_sub(5)..n_layers).rev() {
        let l = &layers[i];
        let ib_n = l.infill_boundary.len();
        let sf_n = l.solid_fill_boundary.len();
        let mut walls = vec![];
        for w in l.loops.iter().filter(|w| w.wall_index == 0).take(6) {
            let pts = &w.points;
            if pts.is_empty() {
                continue;
            }
            let zmin = pts.iter().map(|p| p.z).fold(f64::INFINITY, |a, b| a.min(b));
            let zmax = pts
                .iter()
                .map(|p| p.z)
                .fold(f64::NEG_INFINITY, |a, b| a.max(b));
            walls.push(format!("w0 n={} z {zmin:.2}..{zmax:.2}", pts.len()));
        }
        println!(
            "L{i} order={:.4} ib={} sf={} walls: {}",
            l.order,
            ib_n,
            sf_n,
            walls.join(" ")
        );
    }

    // Part B: face-order range on z=max (via last layer's order field)
    let field = layers.last().unwrap().order_field.clone();
    let bb = object.mesh.bounding_box().unwrap();
    let ztop = bb.1.z;
    let mut omin = f64::INFINITY;
    let mut omax = f64::NEG_INFINITY;
    let mut amin = (0.0, 0.0);
    let mut amax = (0.0, 0.0);
    let nv = 40;
    for i in 0..nv {
        for j in 0..nv {
            let x = bb.0.x + (bb.1.x - bb.0.x) * (i as f64 / (nv - 1) as f64);
            let y = bb.0.y + (bb.1.y - bb.0.y) * (j as f64 / (nv - 1) as f64);
            let p = DVec3::new(x, y, ztop - 0.05);
            let o = field.order(p);
            if o < omin {
                omin = o;
                amin = (x, y);
            }
            if o > omax {
                omax = o;
                amax = (x, y);
            }
        }
    }
    println!("\n--- Part B: face-order just below top face (z={ztop:.2}) ---");
    println!(
        "order range [{omin:.4}, {omax:.4}] width={:.4} min@({:.1},{:.1}) max@({:.1},{:.1})",
        omax - omin,
        amin.0,
        amin.1,
        amax.0,
        amax.1
    );

    // Part C: per-kind z census (z >= ztop - 1.2)
    println!("\n--- Part C: kind x z-band (z >= {:.2}) ---", ztop - 1.2);
    let mut agg: std::collections::BTreeMap<(String, u32), (usize, f64)> =
        std::collections::BTreeMap::new();
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
            if zm < ztop - 1.2 {
                continue;
            }
            let zb = (zm * 10.0).round() as u32;
            let k = format!("{:?}", s.kind);
            let e = agg.entry((k, zb)).or_insert((0, 0.0));
            e.0 += 1;
            e.1 += s.extrusion_length;
        }
    }
    for ((k, zb), (n, e)) in &agg {
        println!("z={:6.1} {k:12} {:5} {:8.3}", *zb as f64 / 10.0, n, e);
    }

    // Part D: seed margins at face-footprint points on top layer planes
    println!("\n--- Part D: seed_proximity at (center, ztop-0.2k) ---");
    let cx = (bb.0.x + bb.1.x) * 0.5;
    let cy = (bb.0.y + bb.1.y) * 0.5;
    for k in 0..=4 {
        let p = DVec3::new(cx, cy, ztop - 0.2 * k as f64);
        let sp = field.seed_proximity(p);
        let o = field.order(p);
        match sp {
            Some((kind, d)) => println!(
                "z={:.2} order={o:.4} seed={kind:?} d={d:.4}",
                ztop - 0.2 * k as f64
            ),
            None => println!("z={:.2} order={o:.4} seed=None", ztop - 0.2 * k as f64),
        }
    }

    let _ = paths.len();
    Ok(())
}
