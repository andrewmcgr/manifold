//! Probe the DualIso Eikonal order field over an XY grid at a fixed Z near
//! the vent-slot corner where wall-0 contour extraction produces an open
//! (non-closing) loop, to check whether some query points come back
//! non-finite (`f64::INFINITY`) -- which `extract_order_contours_on_mesh_with_debug`
//! would then filter as `Side::Invalid`, explaining a gap in the raw segment
//! set fed to the stitcher.
//!
//! ```sh
//! cargo run --release -p manifold-cli --example probe_vent_slot_order -- \
//!     /Users/amcgregor/3D/Voron_Design_Cube_v7.stl /Users/amcgregor/3D/profile.json \
//!     107.0 113.5 84.5 92.5 14.2
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
        .unwrap_or_else(|| "Voron_Design_Cube_v7.stl".to_string());
    let profile_path = std::env::args().nth(2);
    let x0: f64 = std::env::args()
        .nth(3)
        .map_or(107.0, |s| s.parse().unwrap());
    let x1: f64 = std::env::args()
        .nth(4)
        .map_or(113.5, |s| s.parse().unwrap());
    let y0: f64 = std::env::args().nth(5).map_or(84.5, |s| s.parse().unwrap());
    let y1: f64 = std::env::args().nth(6).map_or(92.5, |s| s.parse().unwrap());
    let order_value: f64 = std::env::args().nth(7).map_or(14.2, |s| s.parse().unwrap());

    let (config, slope_profile) = match &profile_path {
        Some(p) => {
            let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(p)?)?;
            let machine: Machine = serde_json::from_value(v["machine"].clone())?;
            let config: SlicerConfig = serde_json::from_value(v["config"].clone())?;
            let slope = machine.slope_profile();
            (config, slope)
        }
        None => (SlicerConfig::default(), SlopeProfile::from_angle(15.0)),
    };

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

    let field = order_field_for(OrderFieldKind::DualIso, &config, &mesh, &slope_profile);

    println!("scanning x [{x0:.2}..{x1:.2}] y [{y0:.2}..{y1:.2}] at z (order) ~= {order_value:.4}");

    let step = 0.1;
    let mut n_total = 0usize;
    let mut n_nonfinite = 0usize;
    let mut y = y0;
    while y <= y1 + 1e-9 {
        let mut row = String::new();
        let mut x = x0;
        while x <= x1 + 1e-9 {
            let p = DVec3::new(x, y, order_value);
            let o = field.order(p);
            n_total += 1;
            if !o.is_finite() {
                n_nonfinite += 1;
                row.push_str("   INF");
            } else {
                row.push_str(&format!(" {:5.2}", o));
            }
            x += step;
        }
        println!("y={y:6.2}: {row}");
        y += step;
    }

    println!("\n{n_nonfinite} / {n_total} sampled points had a non-finite order() value");

    Ok(())
}
