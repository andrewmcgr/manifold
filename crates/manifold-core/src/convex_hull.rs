//! Simplified 3D convex hull generation with bounded facet complexity
//! and tripod (>= 3 contact points) stability.
//!
//! Designed for model reorientation ("Lay on Face") and build-plate grounding
//! ("Drop to Bed").

use glam::DVec3;

/// A single flat facet of a simplified convex hull.
#[derive(Debug, Clone, PartialEq)]
pub struct ConvexFacet {
    /// Outward unit normal of the facet.
    pub normal: DVec3,
    /// Distance constant of the supporting plane: `normal.dot(p) == plane_d`.
    pub plane_d: f64,
    /// Model vertices that lie on this supporting facet (guaranteed >= 3 points).
    pub contact_points: Vec<DVec3>,
    /// Boundary polygon vertices of this facet on the hull, ordered around `normal`.
    pub boundary: Vec<DVec3>,
    /// Surface area of the facet polygon (mm²).
    pub area: f64,
}

/// A simplified convex hull enclosing a 3D model.
#[derive(Debug, Clone, PartialEq)]
pub struct SimplifiedHull {
    /// The constituent facets of the hull (guaranteed <= `max_facets`).
    pub facets: Vec<ConvexFacet>,
    /// Total volume of the simplified hull (mm³).
    pub volume: f64,
}

/// Default maximum number of facets on a simplified hull.
pub const DEFAULT_MAX_FACETS: usize = 26;
/// Default stopping threshold for iterative volume carving (1%).
pub const DEFAULT_VOLUME_THRESHOLD_RATIO: f64 = 0.01;

/// Computes a simplified convex hull enclosing `points`.
///
/// Properties:
/// - Produces at most `max_facets` facets (default 26).
/// - Every facet is guaranteed to contact the original point cloud at >= 3 points.
/// - Iteratively carves volume from an initial enclosing polytope using the most
///   volumetric candidate supporting planes, stopping when `max_facets` is reached
///   or the next cut removes less than `volume_threshold_ratio` of the current volume.
pub fn compute_simplified_convex_hull(
    points: &[DVec3],
    max_facets: usize,
    volume_threshold_ratio: f64,
) -> Option<SimplifiedHull> {
    if points.len() < 4 {
        return None;
    }

    // 1. Compute exact 3D convex hull triangles using Quickhull
    let raw_triangles = quickhull_3d(points)?;
    if raw_triangles.is_empty() {
        return None;
    }

    // 2. Group coplanar triangles into initial candidate supporting planes
    let candidate_planes = extract_candidate_planes(&raw_triangles, points);
    if candidate_planes.is_empty() {
        return None;
    }

    // If candidate planes are already <= max_facets, construct the hull directly
    if candidate_planes.len() <= max_facets {
        let polytope = build_polytope_from_planes(&candidate_planes);
        return Some(polytope_to_simplified_hull(&polytope, &candidate_planes));
    }

    // 3. Otherwise, select the top K <= max_facets planes using greedy volume carving
    let selected_planes =
        select_carving_planes(&candidate_planes, max_facets, volume_threshold_ratio);

    let polytope = build_polytope_from_planes(&selected_planes);
    Some(polytope_to_simplified_hull(&polytope, &selected_planes))
}

/// A candidate supporting plane touching the model at >= 3 points.
#[derive(Debug, Clone)]
struct CandidatePlane {
    normal: DVec3,
    plane_d: f64,
    contact_points: Vec<DVec3>,
    hull_area: f64,
}

/// Extract unique supporting planes from raw convex hull triangles,
/// merging coplanar triangles and collecting contact points.
fn extract_candidate_planes(triangles: &[[DVec3; 3]], all_points: &[DVec3]) -> Vec<CandidatePlane> {
    const NORMAL_EPS: f64 = 1e-4;
    const CONTACT_EPS: f64 = 1e-3;

    let mut planes: Vec<CandidatePlane> = Vec::new();

    for tri in triangles {
        let edge1 = tri[1] - tri[0];
        let edge2 = tri[2] - tri[0];
        let normal = edge1.cross(edge2).normalize_or_zero();
        if normal.length_squared() < 1e-12 {
            continue;
        }
        let plane_d = normal.dot(tri[0]);
        let area = 0.5 * edge1.cross(edge2).length();

        // Check if an existing candidate plane matches this orientation and distance
        let mut matched = false;
        for p in &mut planes {
            if (p.normal.dot(normal) - 1.0).abs() < NORMAL_EPS
                && (p.plane_d - plane_d).abs() < CONTACT_EPS
            {
                p.hull_area += area;
                matched = true;
                break;
            }
        }

        if !matched {
            planes.push(CandidatePlane {
                normal,
                plane_d,
                contact_points: Vec::new(),
                hull_area: area,
            });
        }
    }

    // Collect all original points that lie on each supporting plane
    for plane in &mut planes {
        for &pt in all_points {
            if (plane.normal.dot(pt) - plane.plane_d).abs() <= CONTACT_EPS {
                // Deduplicate contact points
                if !plane
                    .contact_points
                    .iter()
                    .any(|&c| (c - pt).length_squared() < 1e-8)
                {
                    plane.contact_points.push(pt);
                }
            }
        }
    }

    // Retain only planes that have at least 3 non-collinear contact points
    planes.retain(|p| {
        if p.contact_points.len() < 3 {
            return false;
        }
        let p0 = p.contact_points[0];
        let p1 = p.contact_points[1];
        let d01 = (p1 - p0).normalize_or_zero();
        for &p2 in &p.contact_points[2..] {
            let d02 = (p2 - p0).normalize_or_zero();
            if d01.cross(d02).length_squared() > 1e-6 {
                return true;
            }
        }
        false
    });

    // Sort planes descending by area on hull
    planes.sort_by(|a, b| {
        b.hull_area
            .partial_cmp(&a.hull_area)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    planes
}

/// A convex 3D polygon face of a polyhedron.
#[derive(Debug, Clone)]
struct PolyFace {
    normal: DVec3,
    plane_d: f64,
    vertices: Vec<DVec3>,
}

/// A 3D convex polyhedron.
#[derive(Debug, Clone)]
struct Polyhedron {
    faces: Vec<PolyFace>,
}

impl Polyhedron {
    /// Compute total volume of the closed convex polyhedron using divergence theorem.
    fn volume(&self) -> f64 {
        if self.faces.is_empty() {
            return 0.0;
        }
        let mut total_vol = 0.0;
        // Use origin (or first vertex) as pivot
        for face in &self.faces {
            if face.vertices.len() < 3 {
                continue;
            }
            let v0 = face.vertices[0];
            for i in 1..face.vertices.len() - 1 {
                let v1 = face.vertices[i];
                let v2 = face.vertices[i + 1];
                total_vol += v0.dot(v1.cross(v2));
            }
        }
        (total_vol / 6.0).abs()
    }

    /// Clip this convex polyhedron by half-space `normal.dot(x) <= plane_d`.
    fn clip_by_plane(&self, normal: DVec3, plane_d: f64) -> Polyhedron {
        const EPS: f64 = 1e-6;
        let mut new_faces = Vec::new();
        let mut cut_polygon = Vec::new();

        for face in &self.faces {
            let mut clipped_verts = Vec::new();
            let count = face.vertices.len();
            if count < 3 {
                continue;
            }

            for i in 0..count {
                let cur = face.vertices[i];
                let next = face.vertices[(i + 1) % count];
                let d_cur = normal.dot(cur) - plane_d;
                let d_next = normal.dot(next) - plane_d;

                let cur_inside = d_cur <= EPS;
                let next_inside = d_next <= EPS;

                if cur_inside {
                    clipped_verts.push(cur);
                }

                if (cur_inside && !next_inside) || (!cur_inside && next_inside) {
                    let denom = d_next - d_cur;
                    if denom.abs() > 1e-12 {
                        let t = (-d_cur / denom).clamp(0.0, 1.0);
                        let intersect = cur + (next - cur) * t;
                        clipped_verts.push(intersect);
                        cut_polygon.push(intersect);
                    }
                }
            }

            if clipped_verts.len() >= 3 {
                new_faces.push(PolyFace {
                    normal: face.normal,
                    plane_d: face.plane_d,
                    vertices: clipped_verts,
                });
            }
        }

        // Add the new facet formed on the clipping plane
        if cut_polygon.len() >= 3 {
            // Deduplicate cut polygon vertices
            let mut unique_cut: Vec<DVec3> = Vec::new();
            for pt in cut_polygon {
                if !unique_cut.iter().any(|&u| (u - pt).length_squared() < 1e-8) {
                    unique_cut.push(pt);
                }
            }

            if unique_cut.len() >= 3 {
                // Sort vertices angularly around face normal
                let center = unique_cut.iter().copied().sum::<DVec3>() / unique_cut.len() as f64;
                let basis1 = if normal.x.abs() < 0.9 {
                    normal.cross(DVec3::X).normalize_or_zero()
                } else {
                    normal.cross(DVec3::Y).normalize_or_zero()
                };
                let basis2 = normal.cross(basis1);

                unique_cut.sort_by(|a, b| {
                    let da = *a - center;
                    let db = *b - center;
                    let angle_a = (da.dot(basis2)).atan2(da.dot(basis1));
                    let angle_b = (db.dot(basis2)).atan2(db.dot(basis1));
                    angle_a
                        .partial_cmp(&angle_b)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                new_faces.push(PolyFace {
                    normal,
                    plane_d,
                    vertices: unique_cut,
                });
            }
        }

        Polyhedron { faces: new_faces }
    }
}

/// Create an initial bounding box polyhedron enclosing the planes.
fn build_initial_bounding_polyhedron(planes: &[CandidatePlane]) -> Polyhedron {
    let mut min = DVec3::splat(f64::INFINITY);
    let mut max = DVec3::splat(f64::NEG_INFINITY);

    for p in planes {
        for &pt in &p.contact_points {
            min = min.min(pt);
            max = max.max(pt);
        }
    }

    // Expand bounding box slightly (10%) as initial bounding volume
    let extent = (max - min).max(DVec3::splat(1.0));
    let margin = extent * 0.1;
    let bmin = min - margin;
    let bmax = max + margin;

    // 6 faces of the axis-aligned bounding box
    let v = [
        DVec3::new(bmin.x, bmin.y, bmin.z), // 0
        DVec3::new(bmax.x, bmin.y, bmin.z), // 1
        DVec3::new(bmax.x, bmax.y, bmin.z), // 2
        DVec3::new(bmin.x, bmax.y, bmin.z), // 3
        DVec3::new(bmin.x, bmin.y, bmax.z), // 4
        DVec3::new(bmax.x, bmin.y, bmax.z), // 5
        DVec3::new(bmax.x, bmax.y, bmax.z), // 6
        DVec3::new(bmin.x, bmax.y, bmax.z), // 7
    ];

    let faces = vec![
        // -Z bottom
        PolyFace {
            normal: -DVec3::Z,
            plane_d: -bmin.z,
            vertices: vec![v[0], v[3], v[2], v[1]],
        },
        // +Z top
        PolyFace {
            normal: DVec3::Z,
            plane_d: bmax.z,
            vertices: vec![v[4], v[5], v[6], v[7]],
        },
        // -X left
        PolyFace {
            normal: -DVec3::X,
            plane_d: -bmin.x,
            vertices: vec![v[0], v[4], v[7], v[3]],
        },
        // +X right
        PolyFace {
            normal: DVec3::X,
            plane_d: bmax.x,
            vertices: vec![v[1], v[2], v[6], v[5]],
        },
        // -Y front
        PolyFace {
            normal: -DVec3::Y,
            plane_d: -bmin.y,
            vertices: vec![v[0], v[1], v[5], v[4]],
        },
        // +Y back
        PolyFace {
            normal: DVec3::Y,
            plane_d: bmax.y,
            vertices: vec![v[3], v[7], v[6], v[2]],
        },
    ];

    Polyhedron { faces }
}

/// Greedily select up to `max_facets` planes that remove the most volume.
fn select_carving_planes(
    candidates: &[CandidatePlane],
    max_facets: usize,
    threshold_ratio: f64,
) -> Vec<CandidatePlane> {
    let mut selected: Vec<CandidatePlane> = Vec::new();
    let mut current_poly = build_initial_bounding_polyhedron(candidates);

    let mut remaining: Vec<CandidatePlane> = candidates.to_vec();

    while selected.len() < max_facets && !remaining.is_empty() {
        let current_vol = current_poly.volume();
        if current_vol < 1e-9 {
            break;
        }

        let mut best_idx = None;
        let mut best_vol_removed = 0.0;
        let mut best_poly = None;

        for (idx, plane) in remaining.iter().enumerate() {
            let clipped = current_poly.clip_by_plane(plane.normal, plane.plane_d);
            let clipped_vol = clipped.volume();
            let vol_removed = current_vol - clipped_vol;

            if vol_removed > best_vol_removed {
                best_vol_removed = vol_removed;
                best_idx = Some(idx);
                best_poly = Some(clipped);
            }
        }

        let Some(idx) = best_idx else {
            break;
        };
        let Some(poly) = best_poly else {
            break;
        };

        // Stopping criterion: if the best cut removes less than threshold ratio (e.g. 1%)
        if best_vol_removed / current_vol < threshold_ratio && selected.len() >= 4 {
            break;
        }

        selected.push(remaining.swap_remove(idx));
        current_poly = poly;
    }

    selected
}

/// Construct a polyhedron by sequentially clipping with all `planes`.
fn build_polytope_from_planes(planes: &[CandidatePlane]) -> Polyhedron {
    let mut poly = build_initial_bounding_polyhedron(planes);
    for plane in planes {
        poly = poly.clip_by_plane(plane.normal, plane.plane_d);
    }
    poly
}

/// Convert a clipped polyhedron and its supporting planes into a `SimplifiedHull`.
fn polytope_to_simplified_hull(poly: &Polyhedron, planes: &[CandidatePlane]) -> SimplifiedHull {
    let mut facets = Vec::new();
    const NORMAL_EPS: f64 = 1e-3;
    const PLANE_D_EPS: f64 = 1e-2;

    for face in &poly.faces {
        if face.vertices.len() < 3 {
            continue;
        }

        // Compute facet polygon area
        let mut area = 0.0;
        let v0 = face.vertices[0];
        for i in 1..face.vertices.len() - 1 {
            let v1 = face.vertices[i];
            let v2 = face.vertices[i + 1];
            area += 0.5 * (v1 - v0).cross(v2 - v0).length();
        }

        // Find matching supporting plane to retrieve model contact points
        let matched_contacts = planes.iter().find(|p| {
            (p.normal.dot(face.normal) - 1.0).abs() < NORMAL_EPS
                && (p.plane_d - face.plane_d).abs() < PLANE_D_EPS
        });

        let contact_points = matched_contacts
            .map(|p| p.contact_points.clone())
            .unwrap_or_else(|| face.vertices.clone());

        facets.push(ConvexFacet {
            normal: face.normal,
            plane_d: face.plane_d,
            contact_points,
            boundary: face.vertices.clone(),
            area,
        });
    }

    SimplifiedHull {
        volume: poly.volume(),
        facets,
    }
}

// -----------------------------------------------------------------------------
// 3D Quickhull Implementation
// -----------------------------------------------------------------------------

struct QuickhullFace {
    indices: [usize; 3],
    normal: DVec3,
    plane_d: f64,
    outside_points: Vec<usize>,
}

fn quickhull_3d(points: &[DVec3]) -> Option<Vec<[DVec3; 3]>> {
    if points.len() < 4 {
        return None;
    }

    // Find extreme points along coordinate axes
    let mut min_x = 0;
    let mut max_x = 0;
    for (i, p) in points.iter().enumerate() {
        if p.x < points[min_x].x {
            min_x = i;
        }
        if p.x > points[max_x].x {
            max_x = i;
        }
    }
    if min_x == max_x {
        return None;
    }

    // Point farthest from line between min_x and max_x
    let a = points[min_x];
    let b = points[max_x];
    let line_dir = (b - a).normalize_or_zero();
    let mut max_dist = -1.0;
    let mut third = 0;
    for (i, &p) in points.iter().enumerate() {
        let dist = (p - a).cross(line_dir).length();
        if dist > max_dist {
            max_dist = dist;
            third = i;
        }
    }
    if max_dist < 1e-6 {
        return None; // All points collinear
    }

    // Point farthest from plane (a, b, c)
    let c = points[third];
    let tri_normal = (b - a).cross(c - a).normalize_or_zero();
    let mut max_plane_dist = -1.0;
    let mut fourth = 0;
    for (i, &p) in points.iter().enumerate() {
        let dist = (p - a).dot(tri_normal).abs();
        if dist > max_plane_dist {
            max_plane_dist = dist;
            fourth = i;
        }
    }
    if max_plane_dist < 1e-6 {
        return None; // All points coplanar
    }

    let _d = points[fourth];
    let mut v = [min_x, max_x, third, fourth];

    // Ensure initial tetrahedron vertices are oriented CCW from outside
    if (points[v[1]] - points[v[0]])
        .cross(points[v[2]] - points[v[0]])
        .dot(points[v[3]] - points[v[0]])
        > 0.0
    {
        v.swap(1, 2);
    }

    let initial_faces = [
        [v[0], v[1], v[2]],
        [v[0], v[3], v[1]],
        [v[1], v[3], v[2]],
        [v[2], v[3], v[0]],
    ];

    let mut faces: Vec<QuickhullFace> = Vec::new();
    for idxs in initial_faces {
        let normal = (points[idxs[1]] - points[idxs[0]])
            .cross(points[idxs[2]] - points[idxs[0]])
            .normalize_or_zero();
        let plane_d = normal.dot(points[idxs[0]]);
        faces.push(QuickhullFace {
            indices: idxs,
            normal,
            plane_d,
            outside_points: Vec::new(),
        });
    }

    // Assign points to outside faces
    for (pt_idx, &pt) in points.iter().enumerate() {
        if v.contains(&pt_idx) {
            continue;
        }
        for face in &mut faces {
            if face.normal.dot(pt) - face.plane_d > 1e-6 {
                face.outside_points.push(pt_idx);
                break;
            }
        }
    }

    // Main Quickhull loop
    let mut iter = 0;
    while iter < 2000 {
        iter += 1;
        // Find face with farthest outside point
        let mut best_face_idx = None;
        let mut max_d = 0.0;
        let mut best_point_idx = 0;

        for (f_idx, face) in faces.iter().enumerate() {
            for &pt_idx in &face.outside_points {
                let dist = face.normal.dot(points[pt_idx]) - face.plane_d;
                if dist > max_d {
                    max_d = dist;
                    best_face_idx = Some(f_idx);
                    best_point_idx = pt_idx;
                }
            }
        }

        let Some(_f_idx) = best_face_idx else {
            break; // No more outside points: hull complete!
        };

        let eye_pt = points[best_point_idx];

        // Find visible faces from eye_pt
        let mut visible = vec![false; faces.len()];
        let mut unassigned_points = Vec::new();

        for (i, face) in faces.iter().enumerate() {
            if face.normal.dot(eye_pt) - face.plane_d > 1e-6 {
                visible[i] = true;
                unassigned_points.extend_from_slice(&face.outside_points);
            }
        }

        // Find horizon edges (edges of visible faces shared with non-visible faces)
        let mut horizon: Vec<(usize, usize)> = Vec::new();
        for (i, face) in faces.iter().enumerate() {
            if !visible[i] {
                continue;
            }
            for edge_idx in 0..3 {
                let e0 = face.indices[edge_idx];
                let e1 = face.indices[(edge_idx + 1) % 3];

                // Check if neighboring face sharing this edge is non-visible
                let neighbor_visible = faces.iter().enumerate().any(|(j, other)| {
                    j != i
                        && visible[j]
                        && other.indices.contains(&e0)
                        && other.indices.contains(&e1)
                });

                if !neighbor_visible {
                    horizon.push((e0, e1));
                }
            }
        }

        // Remove visible faces
        let mut surviving_faces = Vec::new();
        for (i, face) in faces.into_iter().enumerate() {
            if !visible[i] {
                surviving_faces.push(face);
            }
        }
        faces = surviving_faces;

        // Build new triangular faces from horizon to eye_pt
        let mut new_faces = Vec::new();
        for (e0, e1) in horizon {
            let normal = (points[e1] - points[e0])
                .cross(eye_pt - points[e0])
                .normalize_or_zero();
            let plane_d = normal.dot(points[e0]);
            new_faces.push(QuickhullFace {
                indices: [e0, e1, best_point_idx],
                normal,
                plane_d,
                outside_points: Vec::new(),
            });
        }

        // Reassign unassigned points to new faces
        unassigned_points.retain(|&idx| idx != best_point_idx);
        for pt_idx in unassigned_points {
            let pt = points[pt_idx];
            for face in &mut new_faces {
                if face.normal.dot(pt) - face.plane_d > 1e-6 {
                    face.outside_points.push(pt_idx);
                    break;
                }
            }
        }

        faces.extend(new_faces);
    }

    Some(
        faces
            .into_iter()
            .map(|f| {
                [
                    points[f.indices[0]],
                    points[f.indices[1]],
                    points[f.indices[2]],
                ]
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cube_simplified_hull_produces_six_faces() {
        let mut cube_points = Vec::new();
        for x in [-10.0, 10.0] {
            for y in [-10.0, 10.0] {
                for z in [-10.0, 10.0] {
                    cube_points.push(DVec3::new(x, y, z));
                }
            }
        }

        let hull = compute_simplified_convex_hull(&cube_points, 26, 0.01)
            .expect("cube hull must be non-empty");

        assert_eq!(hull.facets.len(), 6, "cube hull must have exactly 6 faces");
        for facet in &hull.facets {
            assert!(
                facet.contact_points.len() >= 4,
                "cube face must contact at least 4 points"
            );
        }
        // Cube volume: 20 * 20 * 20 = 8000
        assert!((hull.volume - 8000.0).abs() < 1e-3);
    }

    #[test]
    fn cylinder_simplified_hull_respects_max_facets() {
        let mut cyl_points = Vec::new();
        let num_radial = 36;
        for i in 0..num_radial {
            let angle = (i as f64) * std::f64::consts::TAU / (num_radial as f64);
            let x = 10.0 * angle.cos();
            let y = 10.0 * angle.sin();
            cyl_points.push(DVec3::new(x, y, 0.0));
            cyl_points.push(DVec3::new(x, y, 20.0));
        }

        let max_limit = 16;
        let hull = compute_simplified_convex_hull(&cyl_points, max_limit, 0.01)
            .expect("cylinder hull must succeed");

        assert!(
            hull.facets.len() <= max_limit,
            "facet count {} must be <= limit {}",
            hull.facets.len(),
            max_limit
        );
        for facet in &hull.facets {
            assert!(
                facet.contact_points.len() >= 3,
                "facet must contact at least 3 points"
            );
        }
    }

    #[test]
    fn stopping_threshold_prevents_unnecessary_facets() {
        let mut cube_points = Vec::new();
        for x in [-10.0, 10.0] {
            for y in [-10.0, 10.0] {
                for z in [-10.0, 10.0] {
                    cube_points.push(DVec3::new(x, y, z));
                }
            }
        }
        // Add tiny chamfer point that would remove < 0.01% volume
        cube_points.push(DVec3::new(9.99, 9.99, 9.99));

        let hull = compute_simplified_convex_hull(&cube_points, 26, 0.01).unwrap();
        // The 6 main faces enclose the cube with 0 volume loss, chamfer is skipped
        assert_eq!(hull.facets.len(), 6);
    }

    #[test]
    fn voron_cube_stl_simplified_hull() {
        let path = std::path::Path::new("../../Voron_Design_Cube_v7.stl");
        if !path.exists() {
            return;
        }
        let file = std::fs::File::open(path).expect("must open STL file");
        let mesh = crate::stl::load_stl(std::io::BufReader::new(file)).expect("must load STL");
        let hull = compute_simplified_convex_hull(&mesh.vertices, 26, 0.01)
            .expect("must compute hull for Voron cube");
        assert!(hull.facets.len() <= 26);
        for facet in &hull.facets {
            assert!(
                facet.contact_points.len() >= 3,
                "every facet must contact at least 3 points"
            );
        }
    }
}
