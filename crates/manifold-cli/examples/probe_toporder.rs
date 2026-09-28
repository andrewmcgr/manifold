//! Scratch probe: where does the slicer's order_max (vertex-sampled) sit
//! relative to the true top-face order peak?
//!
//! For non-height order fields, slicing.rs computes (order_min, order_max)
//! as min/max of `field.order(v)` over MESH VERTICES. If the order field
//! peaks in the INTERIOR of a top face (boundary-metric / top-patch
//! distortion), vertex sampling underestimates the roof order and the cap
//! layers stop short of it, leaving part of the top face unprinted.
//!
//! Prints:
//!   Part A: top-8 vertices by order (x,y,z,order) + global vertex min/max
//!   Part B: order on the top face (z = max vertex z), two samplings:
//!            on-face (sdf in [-0.02,0.02]) and just-below (sdf in [-0.05,-0.01])
//!            with min/max/argmax and fraction of points above vertex max
//!   Part C: ASCII map of (order - vertex_max) on the just-below sampling,
//!            0.5mm grid, '#' > +0.05, '+' > 0, '.' <= 0
//!   Part D: x/y histogram of vertices near z = max_z - 0.4 and max_z - 0.4
//!            (feature identification for Test6's z=4.6/2.6 vertex sets)
//!
//! Usage: probe_toporder <mesh.stl> [profile.json] [--empty-slope]

use manifold_core::ids::{ObjectId, ToolId};
use manifold_core::machine::Machine;
use manifold_core::object::{center_on_bed, Object};
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
    let field = &layers.last().unwrap().order_field;
    // Vertices in WORLD space (the order field and mesh SDF are built from
    // the transformed mesh); object.mesh.vertices are local pre-transform.
    let world_verts: Vec<glam::DVec3> = object
        .mesh
        .vertices
        .iter()
        .map(|&v| object.transform.transform_point(v))
        .collect();

    // ---- Part A: vertex order extremes -----------------------------------
    println!("layers={} last_orders={:?}", layers.len(), {
        layers
            .iter()
            .rev()
            .take(4)
            .map(|l| format!("{:.4}", l.order))
            .collect::<Vec<_>>()
            .join(",")
    });
    let vorders: Vec<(f64, glam::DVec3)> = world_verts
        .iter()
        .map(|v| (field.order(*v), *v))
        .filter(|(o, _)| o.is_finite())
        .collect();
    let mut sorted = vorders;
    sorted.sort_by(|a, b| b.0.total_cmp(&a.0));
    println!("\n--- Part A: vertex order extremes (slicer order_min/max = these) ---");
    println!(
        "vertex order min={:.4} max={:.4} (n={})",
        sorted.last().unwrap().0,
        sorted.first().unwrap().0,
        sorted.len()
    );
    println!("top-8 vertices by order:");
    for (o, v) in sorted.iter().take(8) {
        println!("  order={o:8.4}  x={:7.3} y={:7.3} z={:7.3}", v.x, v.y, v.z);
    }

    // ---- Part B/C: top-face order ----------------------------------------
    let max_z = world_verts
        .iter()
        .map(|v| v.z)
        .fold(f64::NEG_INFINITY, f64::max);
    let vmax = sorted.first().unwrap().0;
    let (x0, x1) = (
        world_verts
            .iter()
            .map(|v| v.x)
            .fold(f64::INFINITY, f64::min),
        world_verts
            .iter()
            .map(|v| v.x)
            .fold(f64::NEG_INFINITY, f64::max),
    );
    let (y0, y1) = (
        world_verts
            .iter()
            .map(|v| v.y)
            .fold(f64::INFINITY, f64::min),
        world_verts
            .iter()
            .map(|v| v.y)
            .fold(f64::NEG_INFINITY, f64::max),
    );
    let step = 0.25;
    let sdf = &layers.last().unwrap().mesh_sdf;

    let mut on_face: Vec<(f64, f64, f64)> = Vec::new();
    let mut below_face: Vec<(f64, f64, f64)> = Vec::new();
    let mut xi = 0.0;
    while xi <= (x1 - x0 + 1e-9) {
        let mut yi = 0.0;
        while yi <= (y1 - y0 + 1e-9) {
            let x = x0 + xi;
            let y = y0 + yi;
            if let Some(sdf) = sdf {
                let p_on = glam::DVec3::new(x, y, max_z);
                let s_on = sdf.sample(p_on).value;
                if (-0.02..=0.02).contains(&s_on) {
                    on_face.push((field.order(p_on), x, y));
                }
                let p_bl = glam::DVec3::new(x, y, max_z - 0.02);
                let s_bl = sdf.sample(p_bl).value;
                if (-0.05..=-0.01).contains(&s_bl) {
                    below_face.push((field.order(p_bl), x, y));
                }
            }
            yi += step;
        }
        xi += step;
    }
    let mut_of = |v: &[(f64, f64, f64)]| -> (f64, f64, f64) {
        v.iter()
            .max_by(|a, b| a.0.total_cmp(&b.0))
            .cloned()
            .unwrap_or((f64::NAN, f64::NAN, f64::NAN))
    };
    let (omax, mxx, mxy) = mut_of(&on_face);
    let (omin, minx, miny) = on_face
        .iter()
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .map(|t| (t.0, t.1, t.2))
        .unwrap_or((f64::NAN, f64::NAN, f64::NAN));
    let (bmax, bxx, bxy) = mut_of(&below_face);
    let n_above = on_face.iter().filter(|t| t.0 > vmax).count();
    let n_above_b = below_face.iter().filter(|t| t.0 > vmax).count();
    println!("\n--- Part B: top face z={max_z:.3} (vertex max order {vmax:.4}) ---");
    println!(
        "on-face samples={}: order in [{omin:.4}, {omax:.4}] argmax=({mxx:.2},{mxy:.2}) min@({minx:.2},{miny:.2}) above_vertex_max={n_above}",
        on_face.len()
    );
    println!(
        "just-below samples={}: order max={bmax:.4} @({bxx:.2},{bxy:.2}) above_vertex_max={n_above_b}",
        below_face.len()
    );

    // ---- Part C: ASCII map (just-below sampling), 0.5mm grid --------------
    println!("\n--- Part C: map of (order - vertex_max), 0.5mm grid (y down) ---");
    let gstep = 0.5;
    let cols = ((x1 - x0) / gstep + 1.0) as usize;
    let mut rows: Vec<(f64, f64, f64)> = Vec::new();
    let mut xi = 0.0;
    while xi <= (x1 - x0 + 1e-9) {
        let mut yi = 0.0;
        while yi <= (y1 - y0 + 1e-9) {
            let x = x0 + xi;
            let y = y0 + yi;
            if let Some(sdf) = sdf {
                let p = glam::DVec3::new(x, y, max_z - 0.02);
                let s = sdf.sample(p).value;
                if (-0.05..=-0.01).contains(&s) {
                    rows.push((field.order(p) - vmax, x, y));
                }
            }
            yi += gstep;
        }
        xi += gstep;
    }
    let grid = if cols > 0 {
        rows.iter()
            .map(|(d, x, y)| {
                let cx = (((x - x0) / gstep).round() as i64) as usize;
                let cy = (((y - y0) / gstep).round() as i64) as usize;
                (cx, cy, *d)
            })
            .fold(
                vec![vec![None; cols]; ((y1 - y0) / gstep + 1.0) as usize],
                |mut m, (cx, cy, d)| {
                    if cx < m[0].len() && cy < m.len() {
                        m[cy][cx] = Some(d);
                    }
                    m
                },
            )
    } else {
        Vec::new()
    };
    for (cy, row) in grid.iter().enumerate() {
        let y = y0 + cy as f64 * gstep;
        let line: String = row
            .iter()
            .map(|c| match c {
                Some(d) if *d > 0.05 => '#',
                Some(d) if *d > 0.0 => '+',
                Some(_) => '.',
                None => ' ',
            })
            .collect();
        println!("y={y:7.2} {line}");
    }
    println!("(x left-to-right from {x0:.1}; # > +0.05 above vertex max, + above, . at/below)");

    // ---- Part E: vertical order profiles + layer-plane physical heights --
    println!("\\n--- Part E: vertical order profiles near the top face ---");
    let samples: [(f64, f64, &str); 6] = [
        (165.0, 165.0, "left-slab-nw"),
        (168.0, 173.0, "left-slab-center"),
        (172.0, 178.0, "left-slab-se"),
        (186.5, 175.0, "right-strip"),
        (187.25, 182.75, "face-argmax"),
        (186.5, 160.25, "face-min"),
    ];
    for (sx, sy, label) in samples {
        print!("  {label:>16}: ");
        for z in [4.0, 4.2, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 4.98, 5.0] {
            let p = glam::DVec3::new(sx, sy, z);
            let o = field.order(p);
            let s = sdf.as_ref().map(|s| s.sample(p).value).unwrap_or(f64::NAN);
            print!(" z{z:.2}={o:.4}(sdf {s:+.3})");
        }
        println!();
    }
    // Physical z of each of the last layer planes at the sample columns.
    println!("  layer-plane physical z (bisection on order(z)=c):");
    let last_orders: Vec<f64> = layers
        .iter()
        .rev()
        .take(5)
        .map(|l| l.order)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    for c in &last_orders {
        print!("    order {c:.4}:");
        for (sx, sy, label) in samples {
            // bisect z in [3.0, 5.0] for order(z) = c
            let mut lo = 3.0f64;
            let mut hi = 5.0f64;
            let mut found = f64::NAN;
            let mut flo = field.order(glam::DVec3::new(sx, sy, lo));
            let fhi = field.order(glam::DVec3::new(sx, sy, hi));
            if (flo - c).abs() < 1e-6 {
                found = lo;
            } else if (fhi - c).abs() < 1e-6 {
                found = hi;
            } else if (flo - c) * (fhi - c) < 0.0 {
                for _ in 0..40 {
                    let mid = (lo + hi) * 0.5;
                    let fm = field.order(glam::DVec3::new(sx, sy, mid));
                    if (flo - c) * (fm - c) <= 0.0 {
                        hi = mid;
                    } else {
                        lo = mid;
                        flo = fm;
                    }
                }
                found = (lo + hi) * 0.5;
            }
            print!(" {label}={found:.4}",);
        }
        println!();
    }
    // w0-sheet (z=4.8) order range over the left slab footprint.
    if let Some(sdf) = sdf {
        let mut mn = f64::INFINITY;
        let mut mx = f64::NEG_INFINITY;
        let mut n = 0usize;
        let mut xi = 0.0;
        while xi <= 14.01 {
            let mut yi = 0.0;
            while yi <= 30.01 {
                let p = glam::DVec3::new(160.0 + xi, 160.0 + yi, 4.8);
                if sdf.sample(p).value < -0.15 {
                    let o = field.order(p);
                    mn = mn.min(o);
                    mx = mx.max(o);
                    n += 1;
                }
                yi += 0.5;
            }
            xi += 0.5;
        }
        println!("  w0-sheet(z=4.8) left-slab interior: order in [{mn:.4}, {mx:.4}] (n={n})");
    }

    // ---- Part F: seed_proximity map near the top face ----------------------
    println!("\\n--- Part F: seed_proximity map (x/y 2mm, z 4.8..5.1) ---");
    let _cols = ((x1 - x0) / 2.0 + 1.0) as usize;
    let zs: [f64; 8] = [4.80, 4.85, 4.90, 4.95, 5.00, 5.05, 5.10, 5.15];
    let legend = " ".repeat(0);
    let _ = legend;
    let mut header = String::from("        ");
    for z in zs {
        header.push_str(&format!(" z{z:.2}"));
    }
    println!("{header}");
    let mut yi = 0.0;
    while yi <= (y1 - y0 + 1e-9) {
        let y = y0 + yi;
        let mut line = format!("y={y:6.1} ");
        for z in zs {
            let mut cell = String::new();
            let mut xi = 0.0;
            while xi <= (x1 - x0 + 1e-9) {
                let x = x0 + xi;
                let p = glam::DVec3::new(x, y, z);
                match field.seed_proximity(p) {
                    Some((manifold_fidget::order::SeedKind::Patch, d)) => {
                        // 3-char cell: P + 2-digit d*100 clamped
                        let code = ((d * 100.0).round() as u32).min(99);
                        cell.push('P');
                        cell.push(char::from(b'0' + (code / 10) as u8));
                        cell.push(char::from(b'0' + (code % 10) as u8));
                    }
                    Some((manifold_fidget::order::SeedKind::Bed, _)) => cell.push_str("B  "),
                    None => cell.push_str("   "),
                }
                xi += 2.0;
            }
            line.push_str(&cell);
            println!("{line}");
        }
        yi += 4.0;
    }
    println!("(Pxz = Patch with d=x*0.1+z*0.01 mm; B = Bed)");

    // ---- Part D: vertex features 0.4 below the top face -------------------
    println!(
        "\n--- Part D: vertices near z={:.1} and z={:.1} (x/y 2mm bins) ---",
        max_z - 0.4,
        max_z - 0.4
    );
    for target in [max_z - 0.4] {
        let mut bins: std::collections::BTreeMap<(i64, i64), usize> =
            std::collections::BTreeMap::new();
        for v in &world_verts {
            if (v.z - target).abs() < 0.05 {
                let b = ((v.x / 2.0).round() as i64, (v.y / 2.0).round() as i64);
                *bins.entry(b).or_default() += 1;
            }
        }
        for (&(bx, by), &n) in &bins {
            println!(
                "  z~{:.1}: bin(x={:.0}..{:.0}, y={:.0}..{:.0}) n={}",
                target,
                bx as f64 * 2.0,
                bx as f64 * 2.0 + 2.0,
                by as f64 * 2.0,
                by as f64 * 2.0 + 2.0,
                n
            );
        }
    }

    Ok(())
}
