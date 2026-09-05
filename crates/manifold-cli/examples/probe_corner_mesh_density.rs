//! Scratch: dump raw watertight mesh triangle density near the min-X/min-Y
//! corner across Z, to check whether the source STL itself is degenerate
//! (self-intersecting / duplicate faces) there, vs the slicer inventing
//! spurious detail on otherwise-clean geometry.
use glam::DVec3;
use manifold_core::ids::{ObjectId, ToolId};
use manifold_core::object::{self, Object};
use manifold_core::stl;

fn main() -> anyhow::Result<()> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "Voron_Design_Cube_v7.stl".to_string());
    let x0: f64 = std::env::args().nth(2).map_or(85.0, |s| s.parse().unwrap());
    let x1: f64 = std::env::args().nth(3).map_or(95.0, |s| s.parse().unwrap());
    let y0: f64 = std::env::args().nth(4).map_or(85.0, |s| s.parse().unwrap());
    let y1: f64 = std::env::args().nth(5).map_or(95.0, |s| s.parse().unwrap());
    let z0: f64 = std::env::args().nth(6).map_or(9.0, |s| s.parse().unwrap());
    let z1: f64 = std::env::args().nth(7).map_or(15.5, |s| s.parse().unwrap());

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

    let faces: Vec<[usize; 3]> = mesh
        .indices
        .chunks_exact(3)
        .map(|c| [c[0] as usize, c[1] as usize, c[2] as usize])
        .collect();
    println!(
        "mesh has {} vertices, {} faces total",
        mesh.vertices.len(),
        faces.len()
    );
    let mut n_in_box = 0usize;
    let mut zmin = f64::MAX;
    let mut zmax = f64::MIN;
    for f in &faces {
        let verts: Vec<DVec3> = f.iter().map(|&i| mesh.vertices[i]).collect();
        let tri_min = DVec3::new(
            verts.iter().map(|v| v.x).fold(f64::MAX, f64::min),
            verts.iter().map(|v| v.y).fold(f64::MAX, f64::min),
            verts.iter().map(|v| v.z).fold(f64::MAX, f64::min),
        );
        let tri_max = DVec3::new(
            verts.iter().map(|v| v.x).fold(f64::MIN, f64::max),
            verts.iter().map(|v| v.y).fold(f64::MIN, f64::max),
            verts.iter().map(|v| v.z).fold(f64::MIN, f64::max),
        );
        let in_box = tri_min.x <= x1
            && tri_max.x >= x0
            && tri_min.y <= y1
            && tri_max.y >= y0
            && tri_min.z <= z1
            && tri_max.z >= z0;
        if in_box {
            n_in_box += 1;
            for v in &verts {
                zmin = zmin.min(v.z);
                zmax = zmax.max(v.z);
            }
        }
    }
    println!("faces touching box x[{x0},{x1}] y[{y0},{y1}] z[{z0},{z1}]: {n_in_box}");
    println!("z range of those faces' vertices: [{zmin:.3}, {zmax:.3}]");

    // Check for near-duplicate faces (possible self-intersection / boolean leftover)
    let mut dup_count = 0usize;
    'outer: for (i, fa) in faces.iter().enumerate() {
        let a: Vec<DVec3> = fa.iter().map(|&idx| mesh.vertices[idx]).collect();
        let in_box = a
            .iter()
            .any(|v| v.x >= x0 && v.x <= x1 && v.y >= y0 && v.y <= y1 && v.z >= z0 && v.z <= z1);
        if !in_box {
            continue;
        }
        for fb in faces.iter().skip(i + 1) {
            let b: Vec<DVec3> = fb.iter().map(|&idx| mesh.vertices[idx]).collect();
            let centroid_dist = ((a[0] + a[1] + a[2]) / 3.0).distance((b[0] + b[1] + b[2]) / 3.0);
            if centroid_dist < 1e-4 {
                dup_count += 1;
                if dup_count > 5 {
                    break 'outer;
                }
            }
        }
    }
    println!("near-duplicate-centroid face pairs found (capped scan): {dup_count}");

    Ok(())
}
