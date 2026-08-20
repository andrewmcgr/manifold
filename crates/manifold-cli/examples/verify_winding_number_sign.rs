//! Scratch verification for the generalized-winding-number sign fallback
//! in `manifold_fidget::mesh_sdf::MeshSdf::sign_at` (replacing the old
//! nearest-tied-feature heuristic `sign_via_tie_break`). Not part of the
//! crate's public surface or test suite -- run manually against the real
//! test mesh that originally reproduced the bug:
//!
//! ```sh
//! cargo run --release -p manifold-cli --example verify_winding_number_sign -- Voron_Design_Cube_v7.stl
//! ```
//!
//! Checks:
//! 1. (Primary) The specific previously-wrong point
//!    (723.223, 314.132, 21.2), confirmed on this mesh to be a
//!    near-tied-distance, opposite-sign ambiguous case for the old
//!    nearest-tied-feature heuristic, now gets a sign consistent with a
//!    majority vote of ray-crossing parity checks along several
//!    directions -- an independent ground truth computed straight from
//!    the triangle soup, sharing no code path with `MeshSdf::sign_at`.
//! 2. (Informational only) Reports, per layer of a full slice, whether
//!    wall0 loops exist with no corresponding wall1. This mesh is a
//!    thin lattice/skeletal calibration cube with many cross-sections
//!    narrower than `2 * wall_line_width`, so a legitimately missing
//!    wall1 (too thin to fit a second wall pass) is expected and not by
//!    itself evidence of a sign bug -- Check 1 is the actual regression
//!    test for this fix; this is left as a secondary sanity signal.

use glam::DVec3;
use manifold_core::{slicing, stl, SlicerConfig};
use manifold_fidget::mesh_sdf::MeshSdf;
use manifold_fidget::slope_profile::SlopeProfile;
use manifold_fidget::ScalarField;
use std::fs::File;
use std::io::BufReader;

fn parity_sign(mesh: &manifold_core::mesh::Mesh, p: DVec3) -> f64 {
    // Independent ground truth: cast several rays in different
    // directions from p to infinity and count triangle crossings via
    // Möller–Trumbore; a point is inside iff a (large) majority of rays
    // report an odd crossing count. Using several directions and a
    // majority vote sidesteps axis-aligned degenerate hits.
    let dirs = [
        DVec3::new(1.0, 0.0173, 0.0091),
        DVec3::new(0.0091, 1.0, 0.0173),
        DVec3::new(0.0173, 0.0091, 1.0),
        DVec3::new(1.0, 1.0, 1.0).normalize(),
        DVec3::new(-1.0, 1.0, 1.0).normalize(),
    ];
    let mut inside_votes = 0;
    for &dir in &dirs {
        let mut crossings = 0u32;
        for face in mesh.indices.chunks(3) {
            let a = mesh.vertices[face[0] as usize];
            let b = mesh.vertices[face[1] as usize];
            let c = mesh.vertices[face[2] as usize];
            if ray_triangle_hit(p, dir, a, b, c) {
                crossings += 1;
            }
        }
        if crossings % 2 == 1 {
            inside_votes += 1;
        }
    }
    if inside_votes * 2 > dirs.len() {
        -1.0
    } else {
        1.0
    }
}

#[allow(clippy::many_single_char_names)]
fn ray_triangle_hit(orig: DVec3, dir: DVec3, a: DVec3, b: DVec3, c: DVec3) -> bool {
    const EPS: f64 = 1e-9;
    let edge1 = b - a;
    let edge2 = c - a;
    let h = dir.cross(edge2);
    let det = edge1.dot(h);
    if det.abs() < EPS {
        return false;
    }
    let inv_det = 1.0 / det;
    let s = orig - a;
    let u = s.dot(h) * inv_det;
    if !(0.0..=1.0).contains(&u) {
        return false;
    }
    let q = s.cross(edge1);
    let v = dir.dot(q) * inv_det;
    if v < 0.0 || u + v > 1.0 {
        return false;
    }
    let t = edge2.dot(q) * inv_det;
    t > EPS
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "Voron_Design_Cube_v7.stl".to_string());
    let file = File::open(&path).unwrap_or_else(|e| panic!("failed to open {path}: {e}"));
    let mesh = stl::load_stl(BufReader::new(file)).expect("failed to parse STL");

    println!("Loaded {} with {} triangles", path, mesh.indices.len() / 3);

    // --- Check 1: the specific previously-ambiguous point. ---
    let sdf = MeshSdf::new(mesh.vertices.clone(), {
        let mut faces = Vec::with_capacity(mesh.indices.len() / 3);
        for f in mesh.indices.chunks(3) {
            faces.push([f[0] as usize, f[1] as usize, f[2] as usize]);
        }
        faces
    });

    let p = DVec3::new(723.223, 314.132, 21.2);
    let sample = sdf.sample(p);
    let sdf_sign = sample.value.signum();
    let truth_sign = parity_sign(&mesh, p);
    println!(
        "point {p:?}: MeshSdf value={:.6} (sign={sdf_sign}), ray-parity ground truth sign={truth_sign}",
        sample.value
    );
    if sdf_sign == truth_sign {
        println!("PASS: winding-number fallback sign now matches ray-parity ground truth");
    } else {
        println!("FAIL: sign still disagrees with ray-parity ground truth");
    }

    // --- Check 2: full slice, look for missing wall1 loops. ---
    let config = SlicerConfig {
        layer_height: 0.2,
        ..Default::default()
    };
    let slope_profile = SlopeProfile::new(Vec::new());

    let layers = slicing::slice_mesh_with_progress(&mesh, &config, &slope_profile, &mut |_| {})
        .expect("slice failed");

    let mut missing_wall1_layers = 0usize;
    for (i, layer) in layers.iter().enumerate() {
        let wall0 = layer.loops.iter().filter(|w| w.wall_index == 0).count();
        let wall1 = layer.loops.iter().filter(|w| w.wall_index == 1).count();
        if wall0 > 0 && wall1 == 0 {
            missing_wall1_layers += 1;
            println!("layer {i}: wall0={wall0} wall1=0 (missing!)");
        }
    }
    println!(
        "{} / {} layers have wall0 but no wall1",
        missing_wall1_layers,
        layers.len()
    );
}
