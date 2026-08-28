//! Probe the Eikonal order field along vertical columns at reported
//! floating-island centroids, comparing conform-top ON vs OFF, to see
//! whether the island's surface order was lowered or the material
//! beneath it was raised by the conformal blend.
//!
//! ```sh
//! cargo run --release -p manifold-cli --example probe_island_order -- \
//!     pug_v4_m_sop_85mm.stl /Users/amcgregor/3D/profile.json 122.30 96.12
//! ```

use glam::DVec3;
use manifold_core::ids::{ObjectId, ToolId};
use manifold_core::machine::Machine;
use manifold_core::object::{self, Object};
use manifold_core::order_field::{order_field_for, OrderFieldKind};
use manifold_core::{stl, SlicerConfig};
use manifold_fidget::slope_profile::SlopeProfile;

fn main() -> anyhow::Result<()> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "pug_v4_m_sop_85mm.stl".to_string());
    let profile_path = std::env::args().nth(2);
    let x: f64 = std::env::args()
        .nth(3)
        .map_or(122.30, |s| s.parse().unwrap());
    let y: f64 = std::env::args()
        .nth(4)
        .map_or(96.12, |s| s.parse().unwrap());
    let z0: f64 = std::env::args().nth(5).map_or(3.5, |s| s.parse().unwrap());
    let z1: f64 = std::env::args().nth(6).map_or(8.5, |s| s.parse().unwrap());

    let (mut config, slope_profile) = match &profile_path {
        Some(p) => {
            let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(p)?)?;
            let machine: Machine = serde_json::from_value(v["machine"].clone())?;
            let config: SlicerConfig = serde_json::from_value(v["config"].clone())?;
            let slope = machine.slope_profile();
            (config, slope)
        }
        None => (SlicerConfig::default(), SlopeProfile::from_angle(15.0)),
    };
    config.order_field = OrderFieldKind::Eikonal;

    let mesh = stl::load_stl(std::fs::File::open(&path)?)?;
    let mut objects = vec![Object::new(ObjectId(0), mesh, ToolId(0))];
    let build_volume = manifold_core::bounds::BoundingVolume::Aabb {
        min: DVec3::ZERO,
        max: DVec3::new(200.0, 200.0, 200.0),
    };
    object::center_on_bed(&mut objects, &build_volume);
    let obj = &objects[0];
    let mut mesh = obj.mesh.clone();
    for v in &mut mesh.vertices {
        *v = obj.transform.transform_point(*v);
    }

    config.eikonal_conform_top_surfaces = false;
    config.eikonal_conform_bottom_surfaces = false;
    let field_off = order_field_for(OrderFieldKind::Eikonal, &config, &mesh, &slope_profile);

    config.eikonal_conform_top_surfaces = true;
    let field_on = order_field_for(OrderFieldKind::Eikonal, &config, &mesh, &slope_profile);

    println!("column at ({x:.2}, {y:.2}), z {z0:.2}..{z1:.2}:");
    println!(
        "{:>6}  {:>10}  {:>10}  {:>8}",
        "z", "off", "top-on", "delta"
    );
    let mut z = z0;
    while z <= z1 + 1e-9 {
        let p = DVec3::new(x, y, z);
        let o_off = field_off.order(p);
        let o_on = field_on.order(p);
        println!("{z:6.2}  {o_off:10.3}  {o_on:10.3}  {:8.3}", o_on - o_off);
        z += 0.1;
    }
    Ok(())
}
