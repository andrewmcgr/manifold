//! Scratch diagnostic: reproduce `bed_open_sdf`'s `distance_faces` filter
//! (excluding downward-facing bed-contact floor triangles, mirroring
//! `slicing.rs`'s `non_bed_floor_faces` construction exactly) directly on a
//! loaded mesh, then sweep a fine grid over the known corrupted region
//! (vent-slot corner near world (85,85), z in [7.5, 8.5]) sampling
//! `MeshSdf::nearest` and `ScalarField::sample` at adjacent grid points to
//! check whether the *nearest triangle index* (and its sign) flips
//! discontinuously between neighboring query points -- confirming or
//! refuting the "sharp-tipped triangular vent slot -> nearest-face BVH
//! ambiguity" theory before attempting a fix.
//!
//! ```sh
//! cargo run --release -p manifold-cli --example probe_nearest_face_flip -- \
//!     /Users/amcgregor/3D/Voron_Design_Cube_v7.stl 85.0 85.0 10.0 7.5 8.5 0.02
//! ```

use glam::DVec3;
use manifold_core::ids::{ObjectId, ToolId};
use manifold_core::object::{self, Object};
use manifold_core::stl;
use manifold_fidget::mesh_sdf::MeshSdf;
use manifold_fidget::ScalarField;

fn main() -> anyhow::Result<()> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "Voron_Design_Cube_v7.stl".to_string());
    let cx: f64 = std::env::args().nth(2).map_or(85.0, |s| s.parse().unwrap());
    let cy: f64 = std::env::args().nth(3).map_or(85.0, |s| s.parse().unwrap());
    let radius: f64 = std::env::args().nth(4).map_or(10.0, |s| s.parse().unwrap());
    let z_lo: f64 = std::env::args().nth(5).map_or(7.5, |s| s.parse().unwrap());
    let z_hi: f64 = std::env::args().nth(6).map_or(8.5, |s| s.parse().unwrap());
    let step: f64 = std::env::args().nth(7).map_or(0.02, |s| s.parse().unwrap());

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

    let min_z = mesh
        .vertices
        .iter()
        .map(|v| v.z)
        .fold(f64::INFINITY, f64::min);

    let faces: Vec<[usize; 3]> = mesh
        .indices
        .chunks_exact(3)
        .map(|c| [c[0] as usize, c[1] as usize, c[2] as usize])
        .collect();

    // Mirror slicing.rs's `non_bed_floor_faces` filter exactly.
    let non_bed_floor_faces: Vec<[usize; 3]> = faces
        .iter()
        .filter_map(|&[i0, i1, i2]| {
            let v0 = mesh.vertices[i0];
            let v1 = mesh.vertices[i1];
            let v2 = mesh.vertices[i2];
            let normal = (v1 - v0).cross(v2 - v0);
            let normal_len_sq = normal.length_squared();
            if normal_len_sq > 1e-12
                && normal.z < 0.0
                && (v0.z <= min_z + 0.02 || v1.z <= min_z + 0.02 || v2.z <= min_z + 0.02)
            {
                let nz_sq = normal.z * normal.z;
                if nz_sq >= 0.998 * normal_len_sq {
                    return None;
                }
            }
            Some([i0, i1, i2])
        })
        .collect();

    println!(
        "faces total={} non_bed_floor_faces={} (excluded={})",
        faces.len(),
        non_bed_floor_faces.len(),
        faces.len() - non_bed_floor_faces.len()
    );

    let sdf =
        MeshSdf::new_with_distance_faces(mesh.vertices.clone(), faces, non_bed_floor_faces.clone());
    let dump_face = |idx: usize| {
        if idx == usize::MAX || idx >= non_bed_floor_faces.len() {
            return;
        }
        let [i0, i1, i2] = non_bed_floor_faces[idx];
        let v0 = mesh.vertices[i0];
        let v1 = mesh.vertices[i1];
        let v2 = mesh.vertices[i2];
        let n = (v1 - v0).cross(v2 - v0);
        println!(
            "  face[{idx}] verts=({v0:?},{v1:?},{v2:?}) normal={:?}",
            n.normalize_or_zero()
        );
    };

    let n = ((2.0 * radius / step) as usize).max(1);
    let nz = (((z_hi - z_lo) / step) as usize).max(1);
    let mut flip_count = 0usize;
    let mut total = 0usize;

    // Full 3D grid: index (xi, yi, zi) -> (face_idx, value).
    let grid_at = |xi: usize, yi: usize, zi: usize| -> Option<(usize, f64)> {
        let x = cx - radius + xi as f64 * step;
        let y = cy - radius + yi as f64 * step;
        let z = z_lo + zi as f64 * step;
        let p = DVec3::new(x, y, z);
        sdf.nearest(p).map(|(face_idx, _closest, _dist_sq)| {
            let sample = sdf.sample(p);
            (face_idx, sample.value)
        })
    };

    let mut report_flip = |axis: &str,
                           xi: usize,
                           yi: usize,
                           zi: usize,
                           f0: usize,
                           f1: usize,
                           v0: f64,
                           v1: f64| {
        let x = cx - radius + xi as f64 * step;
        let y = cy - radius + yi as f64 * step;
        let z = z_lo + zi as f64 * step;
        let jump = (v1 - v0).abs();
        flip_count += 1;
        println!(
            "FLIP[{axis}] at ({x:.3},{y:.3},{z:.3}) face {f0}->{f1} value {v0:.5}->{v1:.5} jump={jump:.5}"
        );
        if flip_count <= 3 {
            dump_face(f0);
            dump_face(f1);
        }
    };

    for zi in 0..=nz {
        for yi in 0..=n {
            for xi in 0..=n {
                let Some((f0, v0)) = grid_at(xi, yi, zi) else {
                    continue;
                };
                total += 1;
                // A true 1-Lipschitz field can only change by <= step between
                // adjacent samples `step` apart. Flag anything bigger as a
                // discontinuity, not just a face-index change (which alone is
                // harmless if the value is still continuous).
                if xi < n {
                    if let Some((f1, v1)) = grid_at(xi + 1, yi, zi) {
                        if f0 != f1 && (v0.signum() != v1.signum() || (v1 - v0).abs() > step * 3.0)
                        {
                            report_flip("x", xi, yi, zi, f0, f1, v0, v1);
                        }
                    }
                }
                if yi < n {
                    if let Some((f1, v1)) = grid_at(xi, yi + 1, zi) {
                        if f0 != f1 && (v0.signum() != v1.signum() || (v1 - v0).abs() > step * 3.0)
                        {
                            report_flip("y", xi, yi, zi, f0, f1, v0, v1);
                        }
                    }
                }
                if zi < nz {
                    if let Some((f1, v1)) = grid_at(xi, yi, zi + 1) {
                        if f0 != f1 && (v0.signum() != v1.signum() || (v1 - v0).abs() > step * 3.0)
                        {
                            report_flip("z", xi, yi, zi, f0, f1, v0, v1);
                        }
                    }
                }
            }
        }
    }

    println!("total samples={total} discontinuous flips={flip_count}");

    Ok(())
}
