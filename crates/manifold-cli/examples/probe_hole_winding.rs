//! Scratch probe: reproduce the "hole loses its inner wall / infill crosses
//! the hole" defect (Voron cube @ Eikonal, shell 0.8, AllWalls infill).
//!
//! For the first few layers, prints every wall loop's signed area when
//! projected into the shared global `plane_basis(BUILD_DIRECTION)` frame
//! (the frame `slicing.rs` uses for `polygon2d::to_2d` before
//! `inward_offset`). Hole loops must have the opposite sign from their
//! containing outer loop for i_overlay's NonZero fill rule to treat them
//! as holes; same-sign nesting silently erases the hole from every
//! offset/boolean result. Also prints the layer's `infill_boundary` loops
//! the same way, so we can see whether the hole survived into the infill
//! region.
//!
//! ```sh
//! cargo run --release -p manifold-cli --example probe_hole_winding -- Voron_Design_Cube_v7.stl 15
//! ```

use glam::DVec3;
use manifold_core::ids::{ObjectId, ToolId};
use manifold_core::object::Object;
use manifold_core::order_field::OrderFieldKind;
use manifold_core::{infill, slicing, stl, SlicerConfig};
use manifold_fidget::contour::plane_basis;
use manifold_fidget::slope_profile::SlopeProfile;

fn signed_area_2d(points: &[DVec3], basis1: DVec3, basis2: DVec3) -> f64 {
    let uv: Vec<(f64, f64)> = points
        .iter()
        .map(|&p| (p.dot(basis1), p.dot(basis2)))
        .collect();
    let mut area = 0.0;
    for i in 0..uv.len() {
        let (u0, v0) = uv[i];
        let (u1, v1) = uv[(i + 1) % uv.len()];
        area += u0 * v1 - u1 * v0;
    }
    area / 2.0
}

fn centroid(points: &[DVec3]) -> DVec3 {
    points.iter().copied().sum::<DVec3>() / points.len().max(1) as f64
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "Voron_Design_Cube_v7.stl".to_string());
    let degrees: f64 = std::env::args()
        .nth(2)
        .map(|s| s.parse().expect("degrees must be a number"))
        .unwrap_or(15.0);

    let mesh = stl::load_stl(std::fs::File::open(&path)?)?;
    let object = Object::new(ObjectId(0), mesh, ToolId(0));
    let objects = vec![object];
    let config = SlicerConfig {
        order_field: OrderFieldKind::Eikonal,
        shell_thickness: 0.8,
        infill_pattern: infill::InfillPatternKind::AllWalls,
        ..SlicerConfig::default()
    };
    println!("wall_count = {}", config.wall_count());
    let slope_profile = SlopeProfile::new(vec![(0.0, degrees)]);

    let layers = slicing::slice_workspace_with_progress(
        &objects,
        &[ObjectId(0)],
        &config,
        &slope_profile,
        &mut |_| {},
    )?;
    println!("sliced {} layer(s)", layers.len());

    // Mirrors slicing::BUILD_DIRECTION (pub(crate)): (0, 0, 1).
    let (basis1, basis2) = plane_basis(DVec3::new(0.0, 0.0, 1.0));

    for layer in layers.iter().take(4) {
        println!("\n=== layer {} order {:.3} ===", layer.index, layer.order);
        for l in &layer.loops {
            let area = signed_area_2d(&l.points, basis1, basis2);
            let c = centroid(&l.points);
            println!(
                "  wall{} loop {:5} pts  signed_area {:10.2}  centroid ({:7.2}, {:7.2}, {:6.2})",
                l.wall_index,
                l.points.len(),
                area,
                c.x,
                c.y,
                c.z
            );
        }
        for (i, b) in layer.infill_boundary.iter().enumerate() {
            let area = signed_area_2d(b, basis1, basis2);
            let c = centroid(b);
            println!(
                "  infill_boundary[{i}] {:5} pts  signed_area {:10.2}  centroid ({:7.2}, {:7.2}, {:6.2})",
                b.len(),
                area,
                c.x,
                c.y,
                c.z
            );
        }
    }
    Ok(())
}
