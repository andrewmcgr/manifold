//! Scratch diagnostic for missing *external* (wall-0) walls when slicing
//! with an Eikonal order field plus a slope-limit profile. Not part of
//! the crate's test suite -- run manually:
//!
//! ```sh
//! cargo run --release -p manifold-cli --example verify_missing_walls -- Voron_Design_Cube_v7.stl 15
//! ```
//!
//! Slices the mesh with `OrderFieldKind::Eikonal` and a uniform
//! `(0.0, <degrees>)` slope profile, then checks per layer that every
//! sampled surface point whose order value falls inside that layer's
//! order band `(layer.order - layer_height, layer.order]` has *some*
//! wall-0 loop point within a small distance. Surface points are sampled
//! densely across every triangle (not just mesh vertices — coarse CAD
//! meshes have huge triangles, so vertex-only checks miss localized
//! dropouts, e.g. near level-set topology changes where a side hole
//! meets a bore). A cluster of samples in a layer's band with no nearby
//! wall-0 geometry means the external wall is missing there.
//!
//! Vertices on *plateau* faces (surface normal nearly parallel to the
//! order-field gradient — e.g. the bed-contact base and flat tops) are
//! skipped: those faces are covered by solid fill, not by wall-0 loops,
//! so wall proximity is the wrong criterion there.

use glam::DVec3;
use manifold_core::bounds::BoundingVolume;
use manifold_core::ids::{ObjectId, ToolId};
use manifold_core::machine::Machine;
use manifold_core::object::Object;
use manifold_core::order_field::OrderFieldKind;
use manifold_core::tool::Tool;
use manifold_core::{slicing, stl, SlicerConfig};
use manifold_fidget::slope_profile::SlopeProfile;

fn main() -> anyhow::Result<()> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "Voron_Design_Cube_v7.stl".to_string());
    let degrees: f64 = std::env::args()
        .nth(2)
        .map(|s| s.parse().expect("degrees must be a number"))
        .unwrap_or(15.0);

    let file = std::fs::File::open(&path)?;
    let mesh = stl::load_stl(file)?;
    let vertices = mesh.vertices.clone();
    println!(
        "loaded {path}: {} vertices, {} triangles",
        vertices.len(),
        mesh.indices.len() / 3
    );

    let object = Object::new(ObjectId(0), mesh, ToolId(0));
    let objects = vec![object];
    let _machine = Machine::new(
        BoundingVolume::Aabb {
            min: DVec3::splat(-1_000.0),
            max: DVec3::splat(1_000.0),
        },
        vec![Tool::new(ToolId(0), 0.4)],
    );

    let config = SlicerConfig {
        order_field: OrderFieldKind::Eikonal,
        ..SlicerConfig::default()
    };
    let slope_profile = SlopeProfile::from_angle(degrees);

    println!("slicing with Eikonal order field, uniform slope limit {degrees} deg...");
    let layers = slicing::slice_workspace_with_progress(
        &objects,
        &[ObjectId(0)],
        &config,
        &slope_profile,
        &mut |_| {},
    )?;
    println!("sliced {} layer(s)", layers.len());

    let layer_height = config.layer_height;
    // Coverage tolerance: an external-surface vertex should have wall-0
    // geometry within a couple of bead widths (wall_offset + sampling
    // slack). Missing walls show up as distances of many mm.
    let tolerance = 2.0;

    // Dense surface samples: subdivide each triangle on a barycentric
    // grid at roughly `sample_spacing` resolution, carrying the face
    // normal for plateau detection. Coarse CAD meshes concentrate their
    // vertices at feature corners, so vertex-only checks miss localized
    // wall dropouts in the middle of large faces.
    let sample_spacing = 0.75;
    let mesh = &objects[0].mesh;
    let mut samples: Vec<(DVec3, DVec3)> = Vec::new(); // (point, face normal)
    for tri in mesh.indices.chunks_exact(3) {
        let (a, b, c) = (
            vertices[tri[0] as usize],
            vertices[tri[1] as usize],
            vertices[tri[2] as usize],
        );
        let n = (b - a).cross(c - a);
        if n.length_squared() == 0.0 {
            continue;
        }
        let n = n.normalize();
        let divisions = ((a.distance(b).max(b.distance(c)).max(c.distance(a))) / sample_spacing)
            .ceil() as usize;
        let divisions = divisions.max(1);
        for i in 0..=divisions {
            for j in 0..=(divisions - i) {
                let u = i as f64 / divisions as f64;
                let v = j as f64 / divisions as f64;
                let w = 1.0 - u - v;
                samples.push((a * u + b * v + c * w, n));
            }
        }
    }
    println!(
        "{} surface samples (spacing ~{sample_spacing} mm)",
        samples.len()
    );

    // A sample is on a plateau face when its normal is nearly parallel
    // (either sign) to the order-field gradient: base/top faces covered
    // by solid fill rather than wall-0 loops.
    let is_plateau = |field: &dyn manifold_fidget::order::OrderField, v: DVec3, n: DVec3| {
        let eps = 0.05;
        let g = DVec3::new(
            field.order(v + DVec3::X * eps) - field.order(v - DVec3::X * eps),
            field.order(v + DVec3::Y * eps) - field.order(v - DVec3::Y * eps),
            field.order(v + DVec3::Z * eps) - field.order(v - DVec3::Z * eps),
        );
        if !g.is_finite() || g.length_squared() == 0.0 {
            return false;
        }
        n.dot(g.normalize()).abs() > 0.5
    };

    let mut bad_layers = 0usize;
    let mut total_uncovered = 0usize;
    let mut worst: Option<(f64, usize, DVec3)> = None;

    // Vertices whose order is non-finite are unreached by the Eikonal
    // march entirely -- they belong to NO layer band and get no walls at
    // all, so count them separately rather than silently skipping.
    {
        let field = &layers[0].order_field;
        let mut unreached = 0usize;
        let mut example = DVec3::ZERO;
        let mut min_o = f64::INFINITY;
        let mut max_o = f64::NEG_INFINITY;
        for &v in &vertices {
            let o = field.order(v);
            if o.is_finite() {
                min_o = min_o.min(o);
                max_o = max_o.max(o);
            } else {
                unreached += 1;
                example = v;
            }
        }
        println!(
            "vertex order range: [{min_o:.3}, {max_o:.3}]; layer orders [{:.3}, {:.3}]",
            layers.first().unwrap().order,
            layers.last().unwrap().order
        );
        if unreached > 0 {
            println!(
                "WARNING: {unreached}/{} vertices have non-finite order (unreached by Eikonal march), e.g. ({:.2}, {:.2}, {:.2})",
                vertices.len(),
                example.x,
                example.y,
                example.z
            );
        }
    }

    for layer in &layers {
        let field = &layer.order_field;
        let wall0_points: Vec<DVec3> = layer
            .loops
            .iter()
            .filter(|l| l.wall_index == 0)
            .flat_map(|l| l.points.iter().copied())
            .collect();

        // Material printed by this layer spans (order - h, order]: the
        // bead deposited on the contour at `order` fills downward toward
        // the previous layer. Attributing the band this way sends
        // bed-adjacent surface vertices (order just above 0) to layer 1 —
        // the layer that physically prints them — rather than to the
        // degenerate order-0 layer whose contour is empty by design.
        let band_lo = layer.order - layer_height;
        let band_hi = layer.order;
        let mut uncovered = 0usize;
        let mut layer_worst: Option<(f64, DVec3)> = None;
        let mut in_band = 0usize;

        for &(v, n) in &samples {
            let o = field.order(v);
            if !o.is_finite() || o <= band_lo || o > band_hi {
                continue;
            }
            if is_plateau(field.as_ref(), v, n) {
                continue;
            }
            in_band += 1;
            let nearest = wall0_points
                .iter()
                .map(|p| p.distance(v))
                .fold(f64::INFINITY, f64::min);
            if nearest > tolerance {
                uncovered += 1;
                if layer_worst.is_none_or(|(d, _)| nearest > d) {
                    layer_worst = Some((nearest, v));
                }
            }
        }

        if uncovered > 0 {
            bad_layers += 1;
            total_uncovered += uncovered;
            let (d, v) = layer_worst.unwrap();
            println!(
                "layer {:3} (order {:8.3}): {}/{} band samples uncovered, worst {:.2} mm at ({:.2}, {:.2}, {:.2}); wall0 pts={}",
                layer.index, layer.order, uncovered, in_band, d, v.x, v.y, v.z,
                wall0_points.len()
            );
            if worst.is_none_or(|(wd, _, _)| d > wd) {
                worst = Some((d, layer.index, v));
            }
        }
    }

    println!();
    if bad_layers == 0 {
        println!("PASS: every in-band mesh vertex has wall-0 geometry within {tolerance} mm");
    } else {
        let (d, li, v) = worst.unwrap();
        println!(
            "FAIL: {bad_layers} layer(s) with uncovered external surface ({total_uncovered} vertices total); worst {d:.2} mm at layer {li}, near ({:.2}, {:.2}, {:.2})",
            v.x, v.y, v.z
        );
    }
    Ok(())
}
