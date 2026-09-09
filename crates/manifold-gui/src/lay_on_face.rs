//! "Lay on Face" and "Drop to Bed" interactive orientation tools.

use eframe::egui::{self, Color32, Pos2, Stroke};
use glam::{DQuat, DVec3, Mat4};
use manifold_core::convex_hull::{ConvexFacet, SimplifiedHull};
use manifold_core::object::Object;
use manifold_core::transform::Transform;

/// Computes the alignment transform to orient a facet flat on the bed.
///
/// Rotates the object so the facet's normal points in the $-\mathbf{Z}$ direction
/// (downwards into the bed), translates along $+Z$ so the contact points rest
/// flush at `bed_z`, and preserves the facet centroid's position in the XY plane.
pub fn orient_facet_to_bed(object: &Object, facet: &ConvexFacet, bed_z: f64) -> Transform {
    let (scale, rotation, _) = object.transform.0.to_scale_rotation_translation();

    // Compute facet centroid in local coordinates
    let centroid_local = if !facet.contact_points.is_empty() {
        facet.contact_points.iter().copied().sum::<DVec3>() / facet.contact_points.len() as f64
    } else if !facet.boundary.is_empty() {
        facet.boundary.iter().copied().sum::<DVec3>() / facet.boundary.len() as f64
    } else {
        DVec3::ZERO
    };

    // Original world-space XY location of the centroid
    let orig_world_centroid = object.transform.transform_point(centroid_local);

    // World-space facet normal
    let world_normal = (rotation * facet.normal).normalize_or_zero();

    // Rotation arc to align world_normal -> -Z
    let target_dir = -DVec3::Z;
    let align_rot = if world_normal.dot(target_dir) < -0.999999 {
        // Antiparallel: rotate 180 degrees around X axis
        DQuat::from_axis_angle(DVec3::X, std::f64::consts::PI)
    } else {
        DQuat::from_rotation_arc(world_normal, target_dir)
    };

    let new_rotation = align_rot * rotation;

    // Offset of the centroid under new_rotation and scale
    let rotated_centroid_offset = new_rotation * (centroid_local * scale);

    // Keep the centroid at the exact same location in the XY plane
    let new_tx = orig_world_centroid.x - rotated_centroid_offset.x;
    let new_ty = orig_world_centroid.y - rotated_centroid_offset.y;

    // Drop to bed: find minimum Z among all mesh vertices under new_rotation & scale
    let mut min_z_offset = f64::INFINITY;
    for &v in &object.mesh.vertices {
        let rot_v = new_rotation * (v * scale);
        min_z_offset = min_z_offset.min(rot_v.z);
    }

    let new_tz = if min_z_offset.is_finite() {
        bed_z - min_z_offset
    } else {
        bed_z
    };

    Transform::from_scale_rotation_translation(
        scale,
        new_rotation,
        DVec3::new(new_tx, new_ty, new_tz),
    )
}

/// Drops an object vertically so its lowest vertex rests on the bed at `bed_z`,
/// without changing its rotation or scale.
pub fn drop_object_to_bed(object: &Object, bed_z: f64) -> Transform {
    let (scale, rotation, mut translation) = object.transform.0.to_scale_rotation_translation();
    let mut min_z = f64::INFINITY;
    for &v in &object.mesh.vertices {
        let world_v = object.transform.transform_point(v);
        min_z = min_z.min(world_v.z);
    }
    if min_z.is_finite() {
        translation.z += bed_z - min_z;
    }
    Transform::from_scale_rotation_translation(scale, rotation, translation)
}

/// Raycast against the simplified convex hull's facets to find the hovered facet.
/// Returns the index of the closest front-facing facet intersecting the ray.
pub fn pick_hull_facet(
    hull: &SimplifiedHull,
    object: &Object,
    ray_orig: DVec3,
    ray_dir: DVec3,
) -> Option<usize> {
    let inv = object.transform.0.inverse();
    let local_orig = inv.transform_point3(ray_orig);
    let local_dir = inv.transform_vector3(ray_dir);
    let local_len = local_dir.length();
    if local_len < 1e-12 {
        return None;
    }
    let local_dir_norm = local_dir / local_len;

    let mut best_t = f64::INFINITY;
    let mut best_idx = None;

    for (idx, facet) in hull.facets.iter().enumerate() {
        // Backface culling: ray must oppose outward normal
        if facet.normal.dot(local_dir_norm) >= -1e-6 {
            continue;
        }

        // Ray-plane intersection: normal . (orig + t * dir) = plane_d
        let denom = facet.normal.dot(local_dir_norm);
        if denom.abs() < 1e-12 {
            continue;
        }
        let t = (facet.plane_d - facet.normal.dot(local_orig)) / denom;
        if t <= 1e-4 || t >= best_t {
            continue;
        }

        let hit_pt = local_orig + local_dir_norm * t;

        // Check if hit_pt is inside the facet's 2D boundary polygon
        if point_in_convex_polygon_3d(&facet.boundary, facet.normal, hit_pt) {
            best_t = t;
            best_idx = Some(idx);
        }
    }

    best_idx
}

/// Point-in-polygon test for a planar 3D convex polygon.
fn point_in_convex_polygon_3d(polygon: &[DVec3], normal: DVec3, p: DVec3) -> bool {
    if polygon.len() < 3 {
        return false;
    }
    for i in 0..polygon.len() {
        let v0 = polygon[i];
        let v1 = polygon[(i + 1) % polygon.len()];
        let edge = v1 - v0;
        let to_p = p - v0;
        let cross = edge.cross(to_p);
        if cross.dot(normal) < -1e-5 {
            return false;
        }
    }
    true
}

/// Render the simplified convex hull overlay on the 3D viewport canvas.
pub fn render_hull_overlay(
    painter: &egui::Painter,
    hull: &SimplifiedHull,
    object: &Object,
    view_proj: Mat4,
    rect: egui::Rect,
    camera_eye: DVec3,
    hovered_facet_idx: Option<usize>,
) {
    for (idx, facet) in hull.facets.iter().enumerate() {
        // World-space boundary vertices
        let world_verts: Vec<DVec3> = facet
            .boundary
            .iter()
            .map(|&v| object.transform.transform_point(v))
            .collect();

        // Project boundary to screen
        let mut screen_points: Vec<Pos2> = Vec::new();
        let mut all_in_front = true;
        for &pt in &world_verts {
            let clip = view_proj * pt.as_vec3().extend(1.0);
            if clip.w <= 0.0 {
                all_in_front = false;
                break;
            }
            let ndc = clip.truncate() / clip.w;
            screen_points.push(Pos2::new(
                rect.min.x + (ndc.x * 0.5 + 0.5) * rect.width(),
                rect.min.y + (1.0 - (ndc.y * 0.5 + 0.5)) * rect.height(),
            ));
        }

        if !all_in_front || screen_points.len() < 3 {
            continue;
        }

        // World-space normal for backface culling
        let world_normal = (object.transform.0.transform_vector3(facet.normal)).normalize_or_zero();
        let center = world_verts.iter().copied().sum::<DVec3>() / world_verts.len() as f64;
        let to_cam = (camera_eye - center).normalize_or_zero();
        if world_normal.dot(to_cam) <= 0.0 {
            continue; // Backface culled
        }

        let is_hovered = hovered_facet_idx == Some(idx);
        let fill = if is_hovered {
            Color32::from_rgba_unmultiplied(255, 205, 50, 120)
        } else {
            Color32::from_rgba_unmultiplied(50, 160, 240, 45)
        };
        let stroke = if is_hovered {
            Stroke::new(2.5_f32, Color32::from_rgb(255, 225, 100))
        } else {
            Stroke::new(1.2_f32, Color32::from_rgba_unmultiplied(120, 200, 255, 180))
        };

        painter.add(egui::Shape::convex_polygon(screen_points, fill, stroke));

        // Draw contact point markers
        for &cp in &facet.contact_points {
            let world_cp = object.transform.transform_point(cp);
            let clip = view_proj * world_cp.as_vec3().extend(1.0);
            if clip.w > 0.0 {
                let ndc = clip.truncate() / clip.w;
                let screen_cp = Pos2::new(
                    rect.min.x + (ndc.x * 0.5 + 0.5) * rect.width(),
                    rect.min.y + (1.0 - (ndc.y * 0.5 + 0.5)) * rect.height(),
                );
                let dot_color = if is_hovered {
                    Color32::from_rgb(255, 255, 100)
                } else {
                    Color32::from_rgb(100, 220, 255)
                };
                painter.circle_filled(screen_cp, 3.5, dot_color);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use manifold_core::ids::{ObjectId, ToolId};
    use manifold_core::mesh::Mesh;

    #[test]
    fn drop_object_to_bed_aligns_minimum_z() {
        let vertices = vec![
            DVec3::new(0.0, 0.0, 50.0),
            DVec3::new(10.0, 0.0, 60.0),
            DVec3::new(0.0, 10.0, 70.0),
        ];
        let mesh = Mesh::new(vertices.clone(), vec![0, 1, 2]);
        let mut object = Object::new(ObjectId(0), mesh, ToolId(0));
        object.transform = Transform::from_translation(DVec3::new(5.0, 5.0, 10.0));

        let bed_z = 0.0;
        let dropped_transform = drop_object_to_bed(&object, bed_z);

        // Minimum Z was 50.0 + 10.0 = 60.0.
        // After dropping to bed_z = 0.0, minimum vertex should be at 0.0.
        let lowest = vertices
            .iter()
            .map(|&v| dropped_transform.transform_point(v).z)
            .fold(f64::INFINITY, f64::min);
        assert!((lowest - bed_z).abs() < 1e-6);
    }

    #[test]
    fn orient_facet_to_bed_preserves_xy_centroid() {
        let vertices = vec![
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(20.0, 0.0, 0.0),
            DVec3::new(20.0, 20.0, 0.0),
            DVec3::new(0.0, 20.0, 0.0),
            DVec3::new(0.0, 0.0, 20.0),
            DVec3::new(20.0, 0.0, 20.0),
            DVec3::new(20.0, 20.0, 20.0),
            DVec3::new(0.0, 20.0, 20.0),
        ];
        let mesh = Mesh::new(vertices.clone(), vec![0, 1, 2]);
        let mut object = Object::new(ObjectId(0), mesh, ToolId(0));
        object.transform = Transform::from_translation(DVec3::new(100.0, 50.0, 15.0));

        // Facet along +X (x = 20)
        let facet = ConvexFacet {
            normal: DVec3::X,
            plane_d: 20.0,
            contact_points: vec![
                DVec3::new(20.0, 0.0, 0.0),
                DVec3::new(20.0, 20.0, 0.0),
                DVec3::new(20.0, 20.0, 20.0),
                DVec3::new(20.0, 0.0, 20.0),
            ],
            boundary: vec![
                DVec3::new(20.0, 0.0, 0.0),
                DVec3::new(20.0, 20.0, 0.0),
                DVec3::new(20.0, 20.0, 20.0),
                DVec3::new(20.0, 0.0, 20.0),
            ],
            area: 400.0,
        };

        let orig_world_centroid = object
            .transform
            .transform_point(DVec3::new(20.0, 10.0, 10.0));
        let bed_z = 0.0;

        let oriented_transform = orient_facet_to_bed(&object, &facet, bed_z);

        // Under oriented transform, the facet contacts the bed at bed_z
        let new_world_centroid = oriented_transform.transform_point(DVec3::new(20.0, 10.0, 10.0));

        // The XY position of the centroid must not move
        assert!(
            (new_world_centroid.x - orig_world_centroid.x).abs() < 1e-6,
            "centroid X shifted from {} to {}",
            orig_world_centroid.x,
            new_world_centroid.x
        );
        assert!(
            (new_world_centroid.y - orig_world_centroid.y).abs() < 1e-6,
            "centroid Y shifted from {} to {}",
            orig_world_centroid.y,
            new_world_centroid.y
        );
        assert!(
            (new_world_centroid.z - bed_z).abs() < 1e-6,
            "centroid Z should be at bed_z {}",
            bed_z
        );
    }
}
