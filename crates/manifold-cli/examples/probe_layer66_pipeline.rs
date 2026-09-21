//! Scratch probe: replay the toolpath::plan infill pipeline for one layer
//! (default 66) and dump the intermediate geometry after each stage, to
//! localize where the vertical-ladder fold first appears. Also traces
//! `solve_along_near`-equivalent scans at the exact folded/clean points.
//!
//! ```sh
//! CARGO_TARGET_DIR=target/release cargo run --release -p manifold-cli --example probe_layer66_pipeline -- \
//!     /path/to/mesh.stl /path/to/profile.json [layer_index]
//! ```
use manifold_core::ids::{ObjectId, ToolId};
use manifold_core::machine::Machine;
use manifold_core::object::{center_on_bed, Object};
use manifold_core::{infill, order_field, slicing, stl};

#[derive(serde::Deserialize)]
struct Profile {
    machine: Machine,
    config: manifold_core::SlicerConfig,
}

fn bounds3(pts: &[DVec3v]) -> String {
    if pts.is_empty() {
        return "<empty>".into();
    }
    let mut lo = glam::DVec3::splat(f64::INFINITY);
    let mut hi = glam::DVec3::splat(f64::NEG_INFINITY);
    for p in pts {
        lo = lo.min(*p);
        hi = hi.max(*p);
    }
    format!(
        "z[{},{}] x[{},{}] y[{},{}] n={}",
        lo.z,
        hi.z,
        lo.x,
        hi.x,
        lo.y,
        hi.y,
        pts.len()
    )
}

type DVec3v = glam::DVec3;

fn main() -> anyhow::Result<()> {
    let mesh_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "TestObj1.stl".to_string());
    let profile_path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "examples/profile.json".to_string());
    let layer_index: usize = std::env::args()
        .nth(3)
        .map(|s| s.parse().expect("layer index"))
        .unwrap_or(66);
    let mesh = stl::load_stl(std::fs::File::open(&mesh_path)?)?;
    let json = std::fs::read_to_string(&profile_path)?;
    let profile: Profile = serde_json::from_str(&json)?;
    let config = profile.config;

    let object = Object::new(ObjectId(0), mesh, ToolId(0));
    let mut objects = vec![object];
    center_on_bed(&mut objects, &profile.machine.build_volume);
    let object = &objects[0];

    let layers = slicing::slice_object(object, &config)?;
    let layer = &layers[layer_index];
    let target = layer.order;
    let field: &dyn manifold_fidget::order::OrderField = layer.order_field.as_ref();
    println!("layer {} order {:.6}", layer.index, target);
    println!(
        "solid_fill_boundary: {}",
        layer
            .solid_fill_boundary
            .iter()
            .map(|p| bounds3(p))
            .collect::<Vec<_>>()
            .join(" | ")
    );
    println!(
        "infill_boundary:      {}",
        layer
            .infill_boundary
            .iter()
            .map(|p| bounds3(p))
            .collect::<Vec<_>>()
            .join(" | ")
    );

    // Stage 1: InfillRegion::from_layer
    let region = infill::InfillRegion::from_layer(layer, &config);
    println!("\n-- InfillRegion::from_layer (sparse) --");
    for (i, l) in region.loops.iter().enumerate() {
        println!("  loop[{i}]: {}", bounds3(l));
    }

    // Stage 2: narrow/sparse partition (nozzle*15 3D extent), as in plan().
    let (sparse, narrow): (Vec<Vec<DVec3v>>, Vec<Vec<DVec3v>>) =
        region.loops.clone().into_iter().partition(|l| {
            let min = l
                .iter()
                .cloned()
                .fold(glam::DVec3::splat(f64::INFINITY), |a, p| a.min(p));
            let max = l
                .iter()
                .cloned()
                .fold(glam::DVec3::splat(f64::NEG_INFINITY), |a, p| a.max(p));
            (max - min).length() >= config.nozzle_diameter * 15.0
        });
    println!(
        "\n-- partition (extent >= {:.1}mm) --",
        config.nozzle_diameter * 15.0
    );
    for (i, l) in sparse.iter().enumerate() {
        println!("  sparse[{i}]: {}", bounds3(l));
    }
    for (i, l) in narrow.iter().enumerate() {
        println!("  narrow[{i}]: {}", bounds3(l));
    }

    // Stage 3: AllWalls on the narrow loops (the solid generator).
    let solid_loops: Vec<Vec<DVec3v>> = layer
        .solid_fill_boundary
        .clone()
        .into_iter()
        .chain(narrow.clone())
        .collect();
    let solid_region = infill::InfillRegion {
        loops: solid_loops.clone(),
    };
    let solid_gen = infill::generator_for(config.solid_infill_pattern());
    let paths = solid_gen.generate(&solid_region, &config, layer, &object.transform, 1.0);
    println!("\n-- AllWalls solid paths ({} paths) --", paths.len());
    for (i, p) in paths.iter().enumerate() {
        let kinds: std::collections::BTreeSet<String> =
            p.segments.iter().map(|s| format!("{:?}", s.kind)).collect();
        let zmin = p.points.iter().map(|q| q.z).fold(f64::INFINITY, f64::min);
        let zmax = p
            .points
            .iter()
            .map(|q| q.z)
            .fold(f64::NEG_INFINITY, f64::max);
        let xmin = p.points.iter().map(|q| q.x).fold(f64::INFINITY, f64::min);
        let xmax = p
            .points
            .iter()
            .map(|q| q.x)
            .fold(f64::NEG_INFINITY, f64::max);
        let ymin = p.points.iter().map(|q| q.y).fold(f64::INFINITY, f64::min);
        let ymax = p
            .points
            .iter()
            .map(|q| q.y)
            .fold(f64::NEG_INFINITY, f64::max);
        let mut vert = 0;
        for i in 0..p.segments.len() {
            let s = &p.segments[i];
            if s.kind == manifold_core::toolpath::MoveKind::Travel {
                continue;
            }
            let j = (i + 1) % p.points.len();
            if (p.points[j].z - p.points[i].z).abs() > 0.5 {
                vert += 1;
            }
        }
        println!(
            "  path[{i}] n={} kinds={kinds:?} z[{zmin:.3},{zmax:.3}] x[{xmin:.2},{xmax:.2}] y[{ymin:.2},{ymax:.2}] vert={vert}",
            p.points.len()
        );
        if vert > 0 {
            // transitions only (|dz| > 0.5) with the point index, so the
            // folded vs clean runs can be read off directly.
            for (si, s) in p.segments.iter().enumerate() {
                let j = (si + 1) % p.points.len();
                let dz = p.points[j].z - p.points[si].z;
                if s.kind != manifold_core::toolpath::MoveKind::Travel && dz.abs() > 0.5 {
                    println!(
                        "    seg{si} pt{si}({:.3},{:.3},{:.3}) -> pt{j}({:.3},{:.3},{:.3}) dz={dz:.3}",
                        p.points[si].x,
                        p.points[si].y,
                        p.points[si].z,
                        p.points[j].x,
                        p.points[j].y,
                        p.points[j].z
                    );
                }
            }
        }
    }

    // Stage 4: order-field column profiles + exact solve_along_near trace at
    // the folded (floor z=1.662 / z=1.899) and clean (z~11.3-11.6) points of
    // the right void-face strip (path[5]).
    let (axis, apex, _slope) = order_field::resolve_axis_apex_slope(config.order_field, &config);
    let (basis1, basis2) = manifold_fidget::contour::plane_basis(axis);
    let bracket = order_field::max_along_for(&config);
    println!("\n-- column scans + solver trace (bracket={bracket:.3}, target={target:.6}) --");
    let refs: Vec<(f64, f64, f64, usize)> = solid_loops
        .iter()
        .enumerate()
        .flat_map(|(li, loop3)| {
            loop3.iter().map(move |&p| {
                let rel = p - apex;
                (rel.dot(basis1), rel.dot(basis2), rel.dot(axis), li)
            })
        })
        .collect();
    let test_pts: [(f64, f64, &str); 7] = [
        (176.554, 177.900, "folded z=1.662"),
        (176.687, 177.478, "folded z=1.662"),
        (176.472, 175.128, "folded z=1.662"),
        (176.958, 176.766, "folded z=1.899"),
        (177.001, 176.398, "folded z=1.899"),
        (176.961, 177.306, "clean  z=11.573"),
        (176.883, 177.007, "clean  z=11.269"),
    ];
    for (x, y, label) in test_pts {
        let p2 = DVec3v::new(x, y, 0.0);
        let rel = p2 - apex;
        let (u, v) = (rel.dot(basis1), rel.dot(basis2));
        // seed selection: nearest reference in (u, v), as in
        // reconstruct_on_order_field_near.
        let &(su, sv, salong, sloop) = refs
            .iter()
            .min_by(|a, b| {
                let da = (a.0 - u).powi(2) + (a.1 - v).powi(2);
                let db = (b.0 - u).powi(2) + (b.1 - v).powi(2);
                da.total_cmp(&db)
            })
            .unwrap();
        println!(
            "\n== ({x:.3},{y:.3}) [{label}] seed_loop={sloop} seed_uv=({su:.3},{sv:.3}) seed_z={salong:.4}"
        );

        // Column profile: finiteness transitions + sampled order values.
        let mut prev_finite = false;
        let mut prev_z = 0.0;
        let mut trans = Vec::new();
        let mut z = 0.0;
        while z <= 14.0 + 1e-9 {
            let val = field.order(DVec3v::new(x, y, z));
            let fin = val.is_finite();
            if fin != prev_finite {
                trans.push(format!("z={z:.2} {}/", if fin { "finite" } else { "+inf" }));
            }
            prev_finite = fin;
            prev_z = z;
            z += 0.05;
        }
        let _ = prev_z;
        let samples = [0.5f64, 1.0, 1.5, 2.0, 3.0, 11.0, 11.4, 11.8, 12.2, 13.0];
        let vals: Vec<String> = samples
            .iter()
            .map(|&zz| {
                let o = field.order(DVec3v::new(x, y, zz));
                if o.is_finite() {
                    format!("z={zz} o={o:.3}")
                } else {
                    format!("z={zz} o=+inf")
                }
            })
            .collect();
        println!("  transitions: {}", trans.join(", "));
        println!("  values:      {}", vals.join(", "));

        // Exact solve_along_near replica (scan + bisection) with tracing.
        let seed = DVec3v::new(x, y, salong);
        let residual = |along: f64| field.order(seed + axis * along) - target;
        let f_zero = residual(0.0);
        println!(
            "  f_zero(residual at seed) = {}",
            if f_zero.is_finite() {
                format!("{f_zero:.4}")
            } else {
                "+inf".into()
            }
        );
        if f_zero.abs() <= 1e-9 {
            println!("  => Exact(0.0) [seed already on isosurface]");
            continue;
        }
        let min_bound = -bracket;
        let max_bound = bracket;
        let mut best_along = 0.0_f64;
        let mut best_residual = if f_zero.is_finite() {
            f_zero.abs()
        } else {
            f64::INFINITY
        };
        let mut found_any_finite = f_zero.is_finite();
        let mut consider = |along: f64, r: f64| {
            if r.is_finite() {
                found_any_finite = true;
                if r.abs() < best_residual {
                    best_residual = r.abs();
                    best_along = along;
                }
            }
        };
        let steps = 16;
        let mut lo = 0.0;
        let mut hi = 0.0;
        let mut f_lo = f_zero;
        let mut _f_hi = f_zero;
        let mut bracketed = false;
        let mut scan_line = Vec::new();
        for i in 0..=steps {
            let t = min_bound + (max_bound - min_bound) * (i as f64 / steps as f64);
            let r = residual(t);
            consider(t, r);
            let rstr = if r.is_finite() {
                format!("{r:+.3}")
            } else {
                "+inf".into()
            };
            scan_line.push(format!("t{t:+.2}:{rstr}"));
            if r.is_finite() {
                if !bracketed {
                    lo = t;
                    f_lo = r;
                    bracketed = true;
                } else if (f_lo > 0.0 && r <= 0.0) || (f_lo < 0.0 && r >= 0.0) {
                    hi = t;
                    _f_hi = r;
                    break;
                } else {
                    lo = t;
                    f_lo = r;
                }
            }
        }
        println!("  scan: {}", scan_line.join(" "));
        if bracketed
            && f_lo.is_finite()
            && _f_hi.is_finite()
            && ((f_lo <= 0.0 && _f_hi >= 0.0) || (f_lo >= 0.0 && _f_hi <= 0.0))
        {
            let mut iters = 0;
            let mut last = Vec::new();
            let result;
            loop {
                iters += 1;
                let mid = (lo + hi) * 0.5;
                let f_mid = residual(mid);
                consider(mid, f_mid);
                if iters <= 6 || iters % 8 == 0 {
                    last.push(format!(
                        "it{iters} mid={mid:+.4} f={}",
                        if f_mid.is_finite() {
                            format!("{f_mid:+.4}",)
                        } else {
                            "+inf".into()
                        }
                    ));
                }
                if f_mid.abs() <= 1e-9 || (hi - lo).abs() <= 1e-9 {
                    result = mid;
                    break;
                }
                if (f_lo <= 0.0 && f_mid <= 0.0) || (f_lo >= 0.0 && f_mid >= 0.0) {
                    lo = mid;
                    f_lo = f_mid;
                } else {
                    hi = mid;
                }
                if iters >= 64 {
                    result = (lo + hi) * 0.5;
                    break;
                }
            }
            println!(
                "  bisection: bracket[lo={lo:+.3}(f={f_lo:+.3}) hi={hi:+.3}] -> Exact(t={result:+.6}) z_out={:.4} ({} iters)",
                seed.z + result,
                iters
            );
            println!("  bisection trace: {}", last.join(" "));
        } else if found_any_finite && best_residual.is_finite() {
            println!(
                "  => ClosestObserved(t={best_along:+.4}) z_out={:.4} (best_residual={best_residual:.4})",
                seed.z + best_along
            );
        } else {
            println!("  => None (seed kept) z_out={:.4}", seed.z);
        }
    }
    Ok(())
}
