//! Probe: AnisotropicFsm order-field behavior at the bottom-midline slab
//! (defect-1 investigation). The bottom-layer wall loops split at the
//! midline (x~173.5-176.5 at layer 0; x~174.4-175.6 at order 0.742) and the
//! infill_boundary excludes the midline strip entirely, even though the mesh
//! SDF says the midline is solid down to z~1.0. This probe samples the
//! production order field at the bottom midline: (a) vertical columns
//! (midline vs a solid-floor control), (b) a 2D x/z value grid at the
//! midline plane, to look for an undefined/abnormal band like the void +inf
//! quirk found for defect 2.
//!
//! Usage: probe_bottom_midline_order <mesh.stl> <profile.json> [midline_x]

use glam::DVec3;
use manifold_core::ids::{ObjectId, ToolId};
use manifold_core::machine::Machine;
use manifold_core::object::{center_on_bed, Object};
use manifold_core::order_field::{order_field_for, OrderFieldKind};

fn main() -> anyhow::Result<()> {
    let mesh_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "TestObj1.stl".to_string());
    let profile_path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "examples/profile.json".to_string());
    let midline_x: f64 = std::env::args()
        .nth(3)
        .map(|s| s.parse().expect("midline_x must be a number"))
        .unwrap_or(175.0);

    #[derive(serde::Deserialize)]
    struct Profile {
        machine: Machine,
        config: manifold_core::SlicerConfig,
    }

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

    let slope_profile = manifold_fidget::slope_profile::SlopeProfile::new(Vec::new());
    let field = order_field_for(
        OrderFieldKind::AnisotropicFsm,
        &profile.config,
        &world_mesh,
        &slope_profile,
    );

    println!(
        "order field: {:?}, midline x = {midline_x}",
        profile.config.order_field
    );

    // (a) Vertical columns through the bottom slab.
    println!("\n=== vertical columns (order value at each z) ===");
    for (label, x) in [
        ("midline", midline_x),
        ("midline-1.5", midline_x - 1.5),
        ("midline+1.5", midline_x + 1.5),
        ("control x=midline-5", midline_x - 5.0),
    ] {
        print!("{label:20} x={x:7.1}:");
        let mut z = 0.2f64;
        while z <= 1.65 {
            let v = field.order(DVec3::new(x, 175.0, z));
            print!(" z{z:4.1}={:8.4}", v);
            z += 0.1;
        }
        println!();
    }

    // (b) 2D x/z value grid at the midline plane (y = 175).
    println!("\n=== 2D grid at y=175 (rows = z, cols = x step 0.5) ===");
    let x0 = midline_x - 4.0;
    print!(
        "{:6} ",
        (0..=8)
            .map(|i| format!("{:7.1}", x0 + 0.5 * i as f64))
            .collect::<Vec<_>>()
            .join(" ")
    );
    println!();
    let mut z = 0.0f64;
    while z <= 2.05 {
        let mut row = format!("{z:4.1}:  ");
        let mut x = x0;
        while x <= x0 + 4.01 {
            let v = field.order(DVec3::new(x, 175.0, z));
            let txt = if !v.is_finite() {
                "  +inf".to_string()
            } else if v.is_nan() {
                "   NaN".to_string()
            } else {
                format!("{v:7.4}")
            };
            row.push_str(&format!("{txt} "));
            x += 0.5;
        }
        println!("{row}");
        z += 0.1;
    }

    // (c) Seed proximity at a few slab points, to see which seed surface
    // the field attributes the slab to.
    println!("\n=== seed_proximity ===");
    for p in [
        DVec3::new(midline_x, 175.0, 0.5),
        DVec3::new(midline_x, 175.0, 1.0),
        DVec3::new(midline_x - 5.0, 175.0, 0.5),
    ] {
        let sp = field.seed_proximity(p);
        println!("({:.1},{:.1},{:.1}) -> {:?}", p.x, p.y, p.z, sp);
    }

    Ok(())
}
