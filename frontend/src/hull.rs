use bevy::prelude::*;
use crate::bridge::ffi::RenderVertex;

/// Maximum number of faces in the simplified convex hull used for "Place on Face".
pub const MAX_SIMPLIFIED_HULL_FACES: usize = 27;

/// Represents a single polygonal support facet on the simplified convex hull ($\le 27$ faces).
#[derive(Clone, Debug)]
pub struct SupportFace {
    pub face_index: usize,
    /// Outward-pointing unit normal in the object's local coordinate frame.
    pub normal: Vec3,
    /// Plane signed offset $d$ such that $n \cdot x \le d$ for all hull vertices.
    pub plane_offset: f32,
    /// Area-weighted centroid of the facet in local coordinates.
    pub centroid: Vec3,
    /// Total surface area of this support facet ($mm^2$).
    pub area: f32,
    /// Ordered convex polygon vertices on the facet boundary (local coordinates).
    pub polygon: Vec<Vec3>,
}

/// Simplified 3D convex hull ($\le 27$ faces) computed for an object so the user
/// can interactively pick a flat support face and orient the part onto the print bed.
#[derive(Clone, Debug, Default)]
pub struct SimplifiedConvexHull {
    pub faces: Vec<SupportFace>,
}

impl SimplifiedConvexHull {
    /// Builds a simplified convex hull with at most `max_faces` (default 27) facets
    /// from an object's local-space vertex cloud.
    pub fn from_vertices(vertices: &[RenderVertex], max_faces: usize) -> Self {
        let cap = max_faces.clamp(4, MAX_SIMPLIFIED_HULL_FACES);
        if vertices.len() < 4 {
            return Self::default();
        }

        // 1. Generate a candidate set of well-distributed outward unit directions:
        //    We start with the 26 canonical k-DOP directions (6 axis + 12 edge-bevel + 8 corner)
        //    plus up to 38 geodesic directions, then evaluate support planes and cluster/decimate
        //    the resulting non-empty hull facets down to at most `cap` (<= 27) largest-area faces.
        let candidate_dirs = generate_candidate_hull_directions(vertices);

        // 2. For each candidate direction, find its tightest bounding half-space
        //    n . p <= d where d = max_{v in V} (n . v).
        let mut min_pt = Vec3::splat(f32::INFINITY);
        let mut max_pt = Vec3::splat(f32::NEG_INFINITY);
        for v in vertices {
            let p = Vec3::new(v.x, v.y, v.z);
            min_pt = min_pt.min(p);
            max_pt = max_pt.max(p);
        }
        let extent = (max_pt - min_pt).max_element().max(1.0);

        let mut halfspaces: Vec<(Vec3, f32)> = Vec::with_capacity(candidate_dirs.len());
        for dir in candidate_dirs {
            let mut max_dot = f32::NEG_INFINITY;
            for v in vertices {
                let p = Vec3::new(v.x, v.y, v.z);
                max_dot = max_dot.max(dir.dot(p));
            }
            // Deduplicate nearly identical half-spaces
            let is_dup = halfspaces.iter().any(|(n, d)| {
                n.dot(dir) > 0.996 && (d - max_dot).abs() < extent * 1e-3
            });
            if !is_dup {
                halfspaces.push((dir, max_dot));
            }
        }

        // 3. Realize the 3D polytope facets by clipping a large polygon on each half-space
        //    against all other half-spaces.
        let mut faces = realize_halfspace_polytope(&halfspaces, min_pt, max_pt);

        // 4. Iteratively decimate / remove the smallest-area half-space and re-realize
        //    until at most `cap` (<= 27) dominant support faces remain.
        while faces.len() > cap && halfspaces.len() > 6 {
            // Find the realized face with the smallest area (protecting the 6 primary axis directions if possible)
            if let Some((smallest_idx, _)) = faces
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.area.partial_cmp(&b.1.area).unwrap_or(std::cmp::Ordering::Equal))
            {
                let remove_normal = faces[smallest_idx].normal;
                if let Some(hs_pos) = halfspaces
                    .iter()
                    .position(|(n, _)| n.dot(remove_normal) > 0.999)
                {
                    halfspaces.remove(hs_pos);
                } else {
                    faces.remove(smallest_idx);
                    break;
                }
                faces = realize_halfspace_polytope(&halfspaces, min_pt, max_pt);
            } else {
                break;
            }
        }

        // Sort faces by descending area and assign sequential face indices
        faces.sort_by(|a, b| b.area.partial_cmp(&a.area).unwrap_or(std::cmp::Ordering::Equal));
        faces.truncate(cap);
        for (idx, f) in faces.iter_mut().enumerate() {
            f.face_index = idx;
        }

        Self { faces }
    }

    /// Raycasts a world-space ray against the object's simplified convex hull
    /// (transformed by `object_transform`), returning `(face_index, hit_distance)`.
    pub fn raycast(&self, object_transform: &Transform, ray: Ray3d) -> Option<(usize, f32)> {
        let mut best_hit: Option<(usize, f32)> = None;

        for face in &self.faces {
            if face.polygon.len() < 3 {
                continue;
            }
            let v0 = object_transform.transform_point(face.polygon[0]);
            for i in 1..(face.polygon.len() - 1) {
                let v1 = object_transform.transform_point(face.polygon[i]);
                let v2 = object_transform.transform_point(face.polygon[i + 1]);

                if let Some(t) = ray_intersect_triangle(ray.origin, *ray.direction, v0, v1, v2) {
                    if best_hit.map_or(true, |(_, best_t)| t < best_t) {
                        best_hit = Some((face.face_index, t));
                    }
                }
            }
        }

        best_hit
    }
}

/// Computes the world rotation quaternion that aligns `local_face_normal` (currently
/// rotated by `current_rotation`) so that it points straight down (`-Vec3::Z`) onto the bed.
pub fn rotation_to_place_face_on_bed(current_rotation: Quat, local_face_normal: Vec3) -> Quat {
    let world_normal = (current_rotation * local_face_normal).normalize_or_zero();
    if world_normal.length_squared() < 1e-6 {
        return current_rotation;
    }

    let target = -Vec3::Z;
    let dot = world_normal.dot(target).clamp(-1.0, 1.0);
    let delta_rot = if dot < -0.9999 {
        // 180-degree flip around an orthogonal horizontal axis
        let ortho = if world_normal.x.abs() < 0.9 {
            world_normal.cross(Vec3::X).normalize()
        } else {
            world_normal.cross(Vec3::Y).normalize()
        };
        Quat::from_axis_angle(ortho, std::f32::consts::PI)
    } else {
        Quat::from_rotation_arc(world_normal, target)
    };

    (delta_rot * current_rotation).normalize()
}

/// Generates candidate outward unit normals combining the 26 k-DOP directions
/// (axes, 45° bevels, corners) with dominant triangle normals sampled from the mesh.
fn generate_candidate_hull_directions(vertices: &[RenderVertex]) -> Vec<Vec3> {
    let mut dirs = Vec::with_capacity(48);

    // 26 canonical directions: (dx, dy, dz) in {-1, 0, 1}^3 \ {(0,0,0)}
    for ix in -1..=1 {
        for iy in -1..=1 {
            for iz in -1..=1 {
                if ix == 0 && iy == 0 && iz == 0 {
                    continue;
                }
                let v = Vec3::new(ix as f32, iy as f32, iz as f32).normalize();
                dirs.push(v);
            }
        }
    }

    // Also sample up to 16 distinct vertex normals from the CAD tessellation
    // to capture oblique planar CAD faces accurately.
    let step = (vertices.len() / 64).max(1);
    for v in vertices.iter().step_by(step) {
        let n = Vec3::new(v.nx, v.ny, v.nz);
        if n.length_squared() > 0.5 {
            let unit_n = n.normalize();
            if dirs.iter().all(|existing| existing.dot(unit_n) < 0.96) {
                dirs.push(unit_n);
                if dirs.len() >= 42 {
                    break;
                }
            }
        }
    }

    dirs
}

/// Clips a bounding polygon on each half-space `(n_i, d_i)` against all other
/// half-spaces `n_j . x <= d_j` using Sutherland-Hodgman 3D polygon clipping.
fn realize_halfspace_polytope(
    halfspaces: &[(Vec3, f32)],
    min_pt: Vec3,
    max_pt: Vec3,
) -> Vec<SupportFace> {
    let center = 0.5 * (min_pt + max_pt);
    let radius = (max_pt - min_pt).length().max(1.0) * 2.0;
    let eps = radius * 1e-5;

    let mut faces = Vec::new();

    for (i, &(normal, offset)) in halfspaces.iter().enumerate() {
        // Construct a large initial square on the plane `normal . x = offset`
        let up = if normal.z.abs() < 0.9 { Vec3::Z } else { Vec3::X };
        let tangent1 = normal.cross(up).normalize();
        let tangent2 = normal.cross(tangent1).normalize();
        let plane_origin = normal * offset + (center - normal * normal.dot(center));

        let mut poly = vec![
            plane_origin + radius * (-tangent1 - tangent2),
            plane_origin + radius * (tangent1 - tangent2),
            plane_origin + radius * (tangent1 + tangent2),
            plane_origin + radius * (-tangent1 + tangent2),
        ];

        // Clip against every other half-space `n_j . x <= d_j`
        for (j, &(clip_n, clip_d)) in halfspaces.iter().enumerate() {
            if i == j {
                continue;
            }
            poly = clip_polygon_by_halfspace(&poly, clip_n, clip_d, eps);
            if poly.len() < 3 {
                break;
            }
        }

        if poly.len() < 3 {
            continue;
        }

        let (area, centroid) = compute_polygon_area_and_centroid(&poly);
        if area > (radius * radius * 1e-6) {
            faces.push(SupportFace {
                face_index: faces.len(),
                normal,
                plane_offset: offset,
                centroid,
                area,
                polygon: poly,
            });
        }
    }

    faces
}

fn clip_polygon_by_halfspace(poly: &[Vec3], clip_n: Vec3, clip_d: f32, eps: f32) -> Vec<Vec3> {
    if poly.is_empty() {
        return Vec::new();
    }

    let mut out = Vec::with_capacity(poly.len() + 1);
    let n = poly.len();

    for k in 0..n {
        let a = poly[k];
        let b = poly[(k + 1) % n];
        let da = clip_n.dot(a) - clip_d;
        let db = clip_n.dot(b) - clip_d;

        let a_inside = da <= eps;
        let b_inside = db <= eps;

        if a_inside && b_inside {
            out.push(b);
        } else if a_inside && !b_inside {
            let denom = da - db;
            if denom.abs() > 1e-12 {
                let t = (da / denom).clamp(0.0, 1.0);
                out.push(a.lerp(b, t));
            }
        } else if !a_inside && b_inside {
            let denom = da - db;
            if denom.abs() > 1e-12 {
                let t = (da / denom).clamp(0.0, 1.0);
                out.push(a.lerp(b, t));
            }
            out.push(b);
        }
    }

    out
}

fn compute_polygon_area_and_centroid(poly: &[Vec3]) -> (f32, Vec3) {
    if poly.len() < 3 {
        return (0.0, Vec3::ZERO);
    }

    let p0 = poly[0];
    let mut total_area = 0.0_f32;
    let mut weighted_centroid = Vec3::ZERO;

    for i in 1..(poly.len() - 1) {
        let p1 = poly[i];
        let p2 = poly[i + 1];
        let tri_area = 0.5 * (p1 - p0).cross(p2 - p0).length();
        let tri_center = (p0 + p1 + p2) / 3.0;
        total_area += tri_area;
        weighted_centroid += tri_area * tri_center;
    }

    if total_area > 1e-9 {
        (total_area, weighted_centroid / total_area)
    } else {
        (0.0, p0)
    }
}

/// Möller–Trumbore ray-triangle intersection test.
pub fn ray_intersect_triangle(
    ray_origin: Vec3,
    ray_dir: Vec3,
    v0: Vec3,
    v1: Vec3,
    v2: Vec3,
) -> Option<f32> {
    let e1 = v1 - v0;
    let e2 = v2 - v0;
    let h = ray_dir.cross(e2);
    let a = e1.dot(h);
    if a.abs() < 1e-7 {
        return None;
    }
    let f = 1.0 / a;
    let s = ray_origin - v0;
    let u = f * s.dot(h);
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = s.cross(e1);
    let v = f * ray_dir.dot(q);
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = f * e2.dot(q);
    if t > 1e-4 {
        Some(t)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simplified_hull_max_27_faces_and_bed_alignment() {
        // Generate a dense sphere of 200 vertices
        let mut verts = Vec::new();
        for i in 0..20 {
            let theta = (i as f32 / 20.0) * std::f32::consts::PI;
            for j in 0..10 {
                let phi = (j as f32 / 10.0) * std::f32::consts::TAU;
                let n = Vec3::new(theta.sin() * phi.cos(), theta.sin() * phi.sin(), theta.cos());
                verts.push(RenderVertex {
                    x: n.x * 25.0,
                    y: n.y * 25.0,
                    z: n.z * 25.0 + 25.0,
                    nx: n.x,
                    ny: n.y,
                    nz: n.z,
                });
            }
        }

        let hull = SimplifiedConvexHull::from_vertices(&verts, MAX_SIMPLIFIED_HULL_FACES);
        assert!(!hull.faces.is_empty());
        assert!(
            hull.faces.len() <= MAX_SIMPLIFIED_HULL_FACES,
            "Expected <= 27 hull faces, got {}",
            hull.faces.len()
        );

        // Verify that aligning any hull face to the bed points its world normal along -Z
        for face in &hull.faces {
            let rot = rotation_to_place_face_on_bed(Quat::IDENTITY, face.normal);
            let aligned_normal = rot * face.normal;
            assert!((aligned_normal - Vec3::NEG_Z).length() < 1e-3);
        }
    }
}
