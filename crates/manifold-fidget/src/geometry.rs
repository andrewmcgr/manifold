//! Pure geometry utilities: point-to-triangle closest-point queries and a
//! spatial index (BVH) over a set of triangles for nearest-triangle
//! queries.
//!
//! This module is independent of the `ScalarField`/`MeshSdf` primitives
//! elsewhere in this crate — it has no dependency on them and can be used
//! standalone.

use glam::DVec3;

/// A triangle defined by its three vertex positions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Triangle {
    pub a: DVec3,
    pub b: DVec3,
    pub c: DVec3,
}

impl Triangle {
    pub fn new(a: DVec3, b: DVec3, c: DVec3) -> Self {
        Triangle { a, b, c }
    }

    /// Axis-aligned bounding box of this triangle.
    fn bounds(&self) -> Aabb {
        Aabb::from_points(&[self.a, self.b, self.c])
    }

    fn centroid(&self) -> DVec3 {
        (self.a + self.b + self.c) / 3.0
    }
}

/// Closest point on a triangle's surface to a query point `p`, using the
/// standard barycentric Voronoi-region test (Ericson, "Real-Time Collision
/// Detection", section 5.1.5).
///
/// Handles all 7 Voronoi regions (3 vertices, 3 edges, 1 face). Degenerate
/// (zero-area / collinear) triangles are not treated specially by the
/// algorithm itself, but the barycentric math stays well-defined and never
/// panics or divides by a value that can be exactly zero without being
/// caught by an earlier region check, so the routine always returns some
/// point on/among the triangle's vertices/edges rather than panicking.
pub fn closest_point_on_triangle(p: DVec3, tri: &Triangle) -> DVec3 {
    let (a, b, c) = (tri.a, tri.b, tri.c);

    // Check vertex region outside A.
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a; // barycentric (1,0,0)
    }

    // Check vertex region outside B.
    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return b; // barycentric (0,1,0)
    }

    // Check edge region of AB.
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let denom = d1 - d3;
        let v = if denom.abs() > f64::EPSILON {
            d1 / denom
        } else {
            0.0
        };
        return a + ab * v;
    }

    // Check vertex region outside C.
    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c; // barycentric (0,0,1)
    }

    // Check edge region of AC.
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let denom = d2 - d6;
        let w = if denom.abs() > f64::EPSILON {
            d2 / denom
        } else {
            0.0
        };
        return a + ac * w;
    }

    // Check edge region of BC.
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let denom = (d4 - d3) + (d5 - d6);
        let w = if denom.abs() > f64::EPSILON {
            (d4 - d3) / denom
        } else {
            0.0
        };
        return b + (c - b) * w;
    }

    // Interior/face region: barycentric (u, v, w).
    let denom = va + vb + vc;
    if denom.abs() <= f64::EPSILON {
        // Degenerate (zero-area) triangle: fall back to the closest of the
        // three vertices rather than dividing by (near-)zero.
        let da = a.distance_squared(p);
        let db = b.distance_squared(p);
        let dc = c.distance_squared(p);
        return if da <= db && da <= dc {
            a
        } else if db <= dc {
            b
        } else {
            c
        };
    }
    let v = vb / denom;
    let w = vc / denom;
    a + ab * v + ac * w
}

/// Axis-aligned bounding box.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Aabb {
    min: DVec3,
    max: DVec3,
}

impl Aabb {
    fn from_points(points: &[DVec3]) -> Self {
        let mut min = points[0];
        let mut max = points[0];
        for &p in &points[1..] {
            min = min.min(p);
            max = max.max(p);
        }
        Aabb { min, max }
    }

    fn union(&self, other: &Aabb) -> Aabb {
        Aabb {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    /// Squared distance from `p` to the nearest point in/on this box (0 if
    /// `p` is inside). Used as a lower bound for BVH pruning.
    fn distance_squared_to(&self, p: DVec3) -> f64 {
        let dx = (self.min.x - p.x).max(0.0).max(p.x - self.max.x);
        let dy = (self.min.y - p.y).max(0.0).max(p.y - self.max.y);
        let dz = (self.min.z - p.z).max(0.0).max(p.z - self.max.z);
        dx * dx + dy * dy + dz * dz
    }
}

enum Node {
    Leaf {
        bounds: Aabb,
        indices: Vec<usize>,
    },
    Internal {
        bounds: Aabb,
        left: Box<Node>,
        right: Box<Node>,
    },
}

impl Node {
    fn bounds(&self) -> Aabb {
        match self {
            Node::Leaf { bounds, .. } => *bounds,
            Node::Internal { bounds, .. } => *bounds,
        }
    }
}

/// Bounding-volume hierarchy over a set of triangles, supporting
/// nearest-triangle-to-a-point queries.
///
/// A simple median-split BVH (partition by centroid along the box's
/// longest axis) is used rather than a more elaborate SAH build: at the
/// mesh sizes this crate deals with (visualization/debug tooling, not a
/// production path-tracer), build quality has negligible query-time impact
/// and a median split is much simpler to get right.
pub struct TriangleBvh {
    triangles: Vec<Triangle>,
    root: Option<Node>,
}

const LEAF_SIZE: usize = 4;

impl TriangleBvh {
    /// Builds a BVH over `triangles`. Degenerate/zero-area triangles are
    /// kept (not filtered out) so index-based lookups on the caller's side
    /// stay valid; they are simply included as their (possibly
    /// degenerate) bounding box.
    pub fn build(triangles: Vec<Triangle>) -> Self {
        let mut indices: Vec<usize> = (0..triangles.len()).collect();
        let root = if indices.is_empty() {
            None
        } else {
            Some(Self::build_node(&triangles, &mut indices))
        };
        TriangleBvh { triangles, root }
    }

    fn build_node(triangles: &[Triangle], indices: &mut [usize]) -> Node {
        let bounds = indices
            .iter()
            .map(|&i| triangles[i].bounds())
            .reduce(|a, b| a.union(&b))
            .expect("build_node is never called with an empty index slice");

        if indices.len() <= LEAF_SIZE {
            return Node::Leaf {
                bounds,
                indices: indices.to_vec(),
            };
        }

        // Split along the box's longest axis by median centroid.
        let extent = bounds.max - bounds.min;
        let axis = if extent.x >= extent.y && extent.x >= extent.z {
            0
        } else if extent.y >= extent.z {
            1
        } else {
            2
        };

        indices.sort_by(|&i, &j| {
            let ci = triangles[i].centroid();
            let cj = triangles[j].centroid();
            let (vi, vj) = match axis {
                0 => (ci.x, cj.x),
                1 => (ci.y, cj.y),
                _ => (ci.z, cj.z),
            };
            vi.partial_cmp(&vj).unwrap_or(std::cmp::Ordering::Equal)
        });

        let mid = indices.len() / 2;
        let (left_indices, right_indices) = indices.split_at_mut(mid);

        // Guard against a degenerate split (e.g. all centroids identical)
        // that would otherwise recurse forever: fall back to a leaf.
        if left_indices.is_empty() || right_indices.is_empty() {
            return Node::Leaf {
                bounds,
                indices: indices.to_vec(),
            };
        }

        let left = Box::new(Self::build_node(triangles, left_indices));
        let right = Box::new(Self::build_node(triangles, right_indices));
        Node::Internal {
            bounds,
            left,
            right,
        }
    }

    /// Returns the index (into the triangle list this BVH was built from)
    /// of the triangle nearest to `p`, along with the closest point on
    /// that triangle and the squared distance to it.
    ///
    /// Returns `None` if the BVH is empty.
    pub fn nearest(&self, p: DVec3) -> Option<(usize, DVec3, f64)> {
        let root = self.root.as_ref()?;
        let mut best: Option<(usize, DVec3, f64)> = None;
        Self::nearest_recurse(&self.triangles, root, p, &mut best);
        best
    }

    fn nearest_recurse(
        triangles: &[Triangle],
        node: &Node,
        p: DVec3,
        best: &mut Option<(usize, DVec3, f64)>,
    ) {
        let bound_dist_sq = node.bounds().distance_squared_to(p);
        if let Some((_, _, best_dist_sq)) = best {
            if bound_dist_sq > *best_dist_sq {
                return; // this whole subtree can't beat the current best
            }
        }

        match node {
            Node::Leaf { indices, .. } => {
                for &idx in indices {
                    let closest = closest_point_on_triangle(p, &triangles[idx]);
                    let dist_sq = closest.distance_squared(p);
                    let is_better = match best {
                        Some((_, _, best_dist_sq)) => dist_sq < *best_dist_sq,
                        None => true,
                    };
                    if is_better {
                        *best = Some((idx, closest, dist_sq));
                    }
                }
            }
            Node::Internal { left, right, .. } => {
                // Visit the nearer child first so pruning is more effective.
                let left_dist = left.bounds().distance_squared_to(p);
                let right_dist = right.bounds().distance_squared_to(p);
                if left_dist <= right_dist {
                    Self::nearest_recurse(triangles, left, p, best);
                    Self::nearest_recurse(triangles, right, p, best);
                } else {
                    Self::nearest_recurse(triangles, right, p, best);
                    Self::nearest_recurse(triangles, left, p, best);
                }
            }
        }
    }

    /// Number of triangles indexed by this BVH.
    pub fn len(&self) -> usize {
        self.triangles.len()
    }

    /// Whether this BVH indexes zero triangles.
    pub fn is_empty(&self) -> bool {
        self.triangles.is_empty()
    }

    /// Access to the underlying triangle at `index`.
    pub fn triangle(&self, index: usize) -> &Triangle {
        &self.triangles[index]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_triangle() -> Triangle {
        // Right triangle in the z=0 plane: A=(0,0,0), B=(1,0,0), C=(0,1,0).
        Triangle::new(
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(0.0, 1.0, 0.0),
        )
    }

    fn approx_eq(a: DVec3, b: DVec3, tol: f64) -> bool {
        a.distance(b) <= tol
    }

    #[test]
    fn vertex_region_a_returns_vertex_a() {
        let tri = unit_triangle();
        // Far in the direction away from both AB and AC from A.
        let p = DVec3::new(-1.0, -1.0, 0.0);
        let closest = closest_point_on_triangle(p, &tri);
        assert!(approx_eq(closest, tri.a, 1e-9));
    }

    #[test]
    fn vertex_region_b_returns_vertex_b() {
        let tri = unit_triangle();
        let p = DVec3::new(2.0, -1.0, 0.0);
        let closest = closest_point_on_triangle(p, &tri);
        assert!(approx_eq(closest, tri.b, 1e-9));
    }

    #[test]
    fn vertex_region_c_returns_vertex_c() {
        let tri = unit_triangle();
        let p = DVec3::new(-1.0, 2.0, 0.0);
        let closest = closest_point_on_triangle(p, &tri);
        assert!(approx_eq(closest, tri.c, 1e-9));
    }

    #[test]
    fn edge_region_ab_projects_onto_edge() {
        let tri = unit_triangle();
        // Above the midpoint of AB, offset in -y (outside the triangle).
        let p = DVec3::new(0.5, -1.0, 0.0);
        let closest = closest_point_on_triangle(p, &tri);
        assert!(approx_eq(closest, DVec3::new(0.5, 0.0, 0.0), 1e-9));
    }

    #[test]
    fn edge_region_ac_projects_onto_edge() {
        let tri = unit_triangle();
        let p = DVec3::new(-1.0, 0.5, 0.0);
        let closest = closest_point_on_triangle(p, &tri);
        assert!(approx_eq(closest, DVec3::new(0.0, 0.5, 0.0), 1e-9));
    }

    #[test]
    fn edge_region_bc_projects_onto_edge() {
        let tri = unit_triangle();
        // BC edge goes from (1,0,0) to (0,1,0): the line x+y=1.
        // Query point beyond BC, along its outward normal direction.
        let p = DVec3::new(1.0, 1.0, 0.0);
        let closest = closest_point_on_triangle(p, &tri);
        assert!(approx_eq(closest, DVec3::new(0.5, 0.5, 0.0), 1e-9));
    }

    #[test]
    fn face_region_returns_projection_onto_plane() {
        let tri = unit_triangle();
        // Directly above the interior of the triangle.
        let p = DVec3::new(0.25, 0.25, 5.0);
        let closest = closest_point_on_triangle(p, &tri);
        assert!(approx_eq(closest, DVec3::new(0.25, 0.25, 0.0), 1e-9));
    }

    #[test]
    fn degenerate_zero_area_triangle_does_not_panic() {
        // All three vertices coincide.
        let tri = Triangle::new(DVec3::ZERO, DVec3::ZERO, DVec3::ZERO);
        let closest = closest_point_on_triangle(DVec3::new(3.0, 4.0, 5.0), &tri);
        assert!(approx_eq(closest, DVec3::ZERO, 1e-9));

        // Collinear vertices.
        let tri = Triangle::new(
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(2.0, 0.0, 0.0),
        );
        let closest = closest_point_on_triangle(DVec3::new(1.0, 5.0, 0.0), &tri);
        // Should land somewhere on the segment, not panic/NaN.
        assert!(closest.is_finite());
    }

    fn brute_force_nearest(triangles: &[Triangle], p: DVec3) -> (usize, DVec3, f64) {
        let mut best: Option<(usize, DVec3, f64)> = None;
        for (idx, tri) in triangles.iter().enumerate() {
            let closest = closest_point_on_triangle(p, tri);
            let dist_sq = closest.distance_squared(p);
            let is_better = match best {
                Some((_, _, best_dist_sq)) => dist_sq < best_dist_sq,
                None => true,
            };
            if is_better {
                best = Some((idx, closest, dist_sq));
            }
        }
        best.expect("triangles must be non-empty")
    }

    fn scattered_triangles() -> Vec<Triangle> {
        vec![
            Triangle::new(
                DVec3::new(0.0, 0.0, 0.0),
                DVec3::new(1.0, 0.0, 0.0),
                DVec3::new(0.0, 1.0, 0.0),
            ),
            Triangle::new(
                DVec3::new(5.0, 5.0, 5.0),
                DVec3::new(6.0, 5.0, 5.0),
                DVec3::new(5.0, 6.0, 5.0),
            ),
            Triangle::new(
                DVec3::new(-3.0, 0.0, 2.0),
                DVec3::new(-2.0, 0.0, 2.0),
                DVec3::new(-3.0, 1.0, 2.0),
            ),
            Triangle::new(
                DVec3::new(10.0, -10.0, 0.0),
                DVec3::new(11.0, -10.0, 0.0),
                DVec3::new(10.0, -9.0, 0.0),
            ),
            Triangle::new(
                DVec3::new(0.0, 0.0, -5.0),
                DVec3::new(1.0, 0.0, -5.0),
                DVec3::new(0.0, 1.0, -5.0),
            ),
            Triangle::new(
                DVec3::new(2.0, 2.0, 2.0),
                DVec3::new(3.0, 2.0, 2.0),
                DVec3::new(2.0, 3.0, 2.0),
            ),
        ]
    }

    #[test]
    fn bvh_nearest_matches_brute_force_for_several_query_points() {
        let triangles = scattered_triangles();
        let bvh = TriangleBvh::build(triangles.clone());

        let queries = [
            DVec3::new(0.1, 0.1, 0.1),
            DVec3::new(5.5, 5.5, 5.0),
            DVec3::new(-2.5, 0.5, 2.0),
            DVec3::new(10.5, -9.5, 0.0),
            DVec3::new(0.2, 0.2, -4.9),
            DVec3::new(2.4, 2.4, 2.1),
            DVec3::new(100.0, 100.0, 100.0),
            DVec3::new(-50.0, -50.0, -50.0),
        ];

        for &p in &queries {
            let (expected_idx, expected_point, expected_dist_sq) =
                brute_force_nearest(&triangles, p);
            let (actual_idx, actual_point, actual_dist_sq) =
                bvh.nearest(p).expect("BVH is non-empty");

            // Distances must match exactly (both use the same closest-point
            // routine); the index may legitimately differ only if two
            // triangles are equidistant, which doesn't happen for this
            // well-separated test set.
            assert!(approx_eq(actual_point, expected_point, 1e-9));
            assert!((actual_dist_sq - expected_dist_sq).abs() <= 1e-9);
            assert_eq!(actual_idx, expected_idx);
        }
    }

    #[test]
    fn empty_bvh_returns_none() {
        let bvh = TriangleBvh::build(Vec::new());
        assert!(bvh.nearest(DVec3::new(0.0, 0.0, 0.0)).is_none());
        assert!(bvh.is_empty());
    }
}
