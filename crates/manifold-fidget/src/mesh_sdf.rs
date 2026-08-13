//! Mesh-derived signed distance field: [`MeshSdf`] wraps a triangle mesh's
//! [`TriangleBvh`](crate::geometry::TriangleBvh) and implements
//! [`ScalarField`] by combining nearest-triangle distance queries with a
//! runtime-selectable sign strategy ([`SignMethod`]).
//!
//! See `MESH_SDF_VISUALIZATION.md` Phase A for the design this module
//! implements.

use crate::geometry::{Triangle, TriangleBvh};
use crate::{FieldSample, ScalarField};
use glam::DVec3;

/// Squared-length tolerance below which a vector is treated as zero (used
/// both for "query point coincides with the closest surface point" and for
/// degenerate triangle edges when computing incident-angle weights).
const DEGENERATE_EPSILON: f64 = 1e-12;

/// Strategy used to determine the sign (inside/outside) of a [`MeshSdf`]
/// sample.
///
/// Only [`SignMethod::Pseudonormal`] is implemented in this pass.
///
/// `WindingNumber` is the designed next variant (see
/// `MESH_SDF_VISUALIZATION.md` Phase A): a hierarchical solid-angle
/// evaluation over the BVH that is robust to non-watertight / non-manifold
/// meshes, unlike pseudonormals. It is deliberately **not implemented
/// here** — it is omitted from this enum rather than added as an
/// `unimplemented!()` stub, specifically so that adding it later is a pure
/// enum-variant addition (plus a new match arm in [`MeshSdf::sign_at`]),
/// not a breaking rename or restructuring of [`MeshSdf`]'s public shape.
/// The BVH build and the pseudonormal precompute already live in their own
/// method ([`MeshSdf::new`]'s helpers), isolated from the query path
/// ([`MeshSdf::sample`]), so a winding-number variant that needs different
/// (or no) precompute can be slotted in without touching either.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SignMethod {
    /// Angle-weighted pseudonormals, precomputed per vertex and per face at
    /// construction time. Sign at query time is
    /// `sign(dot(query - nearest_point, pseudonormal_at_nearest_feature))`.
    /// Cheap, but only correct on watertight, consistently-wound meshes.
    Pseudonormal,
}

/// A [`ScalarField`] backed by a triangle mesh's signed distance: distance
/// to the nearest triangle (via a [`TriangleBvh`]), signed by `sign_method`.
pub struct MeshSdf {
    bvh: TriangleBvh,
    /// Flat face normal per triangle, same indexing as the BVH's
    /// (originally-supplied) triangle order. Used for face- and
    /// edge-Voronoi-region queries (see the simplification note on
    /// [`MeshSdf::feature_normal`]).
    face_normals: Vec<DVec3>,
    /// Vertex indices (into `vertex_positions`) for each triangle, same
    /// indexing as `face_normals`/the BVH.
    face_vertex_indices: Vec<[usize; 3]>,
    /// Original mesh vertex positions.
    vertex_positions: Vec<DVec3>,
    /// Angle-weighted pseudonormal per vertex, same indexing as
    /// `vertex_positions`.
    vertex_pseudonormals: Vec<DVec3>,
    sign_method: SignMethod,
}

impl MeshSdf {
    /// Builds a `MeshSdf` from `vertices` and `faces` (each face a triple of
    /// indices into `vertices`, consistently wound).
    ///
    /// This isolates the one-time O(triangles) precompute (BVH build +
    /// angle-weighted pseudonormals) from the query path in
    /// [`MeshSdf::sample`].
    ///
    /// Does not panic/error on malformed input: an empty `faces` list
    /// produces a `MeshSdf` whose `sample()` always returns a sentinel (see
    /// [`ScalarField::sample`] impl below); degenerate (zero-area)
    /// triangles are kept (matching [`TriangleBvh::build`]'s behavior) and
    /// contribute a zero face normal / zero angle weight rather than
    /// panicking.
    pub fn new(vertices: Vec<DVec3>, faces: Vec<[usize; 3]>) -> Self {
        let triangles: Vec<Triangle> = faces
            .iter()
            .map(|f| Triangle::new(vertices[f[0]], vertices[f[1]], vertices[f[2]]))
            .collect();

        let face_normals: Vec<DVec3> = triangles
            .iter()
            .map(|tri| {
                let n = (tri.b - tri.a).cross(tri.c - tri.a);
                if n.length_squared() > DEGENERATE_EPSILON {
                    n.normalize()
                } else {
                    // Degenerate (zero-area/collinear) triangle: no
                    // well-defined face normal. Zero contributes nothing to
                    // vertex pseudonormal weighting and signals "no face
                    // information" for face/edge-region sign queries on
                    // this triangle.
                    DVec3::ZERO
                }
            })
            .collect();

        let vertex_pseudonormals =
            Self::compute_vertex_pseudonormals(&vertices, &faces, &face_normals);

        let bvh = TriangleBvh::build(triangles);

        MeshSdf {
            bvh,
            face_normals,
            face_vertex_indices: faces,
            vertex_positions: vertices,
            vertex_pseudonormals,
            sign_method: SignMethod::Pseudonormal,
        }
    }

    /// Precomputes angle-weighted vertex pseudonormals: for each face
    /// incident to a vertex, the face normal is weighted by the angle the
    /// face subtends at that vertex, then the weighted sum is normalized.
    /// This is the standard construction (Baerentzen & Aanaes) for a
    /// pseudonormal that gives a consistent sign for points in a vertex's
    /// Voronoi region regardless of which incident face happens to be
    /// nearest.
    fn compute_vertex_pseudonormals(
        vertices: &[DVec3],
        faces: &[[usize; 3]],
        face_normals: &[DVec3],
    ) -> Vec<DVec3> {
        let mut accum = vec![DVec3::ZERO; vertices.len()];
        for (face, &normal) in faces.iter().zip(face_normals.iter()) {
            for corner in 0..3 {
                let v0 = vertices[face[corner]];
                let v1 = vertices[face[(corner + 1) % 3]];
                let v2 = vertices[face[(corner + 2) % 3]];
                let e1 = v1 - v0;
                let e2 = v2 - v0;
                if e1.length_squared() <= DEGENERATE_EPSILON
                    || e2.length_squared() <= DEGENERATE_EPSILON
                {
                    // Degenerate edge at this corner: no well-defined angle,
                    // contribute nothing.
                    continue;
                }
                let angle = e1.angle_between(e2);
                accum[face[corner]] += normal * angle;
            }
        }
        accum
            .into_iter()
            .map(|n| {
                if n.length_squared() > DEGENERATE_EPSILON {
                    n.normalize()
                } else {
                    // Isolated vertex, or all incident faces degenerate:
                    // no well-defined pseudonormal. Zero is a safe fallback
                    // that `sign_at` treats as "no vertex information" (it
                    // falls back to the face normal in that case, see
                    // below).
                    DVec3::ZERO
                }
            })
            .collect()
    }

    /// Number of triangles in this SDF's mesh.
    pub fn len(&self) -> usize {
        self.bvh.len()
    }

    /// Whether this SDF's mesh has zero triangles.
    pub fn is_empty(&self) -> bool {
        self.bvh.is_empty()
    }

    /// Changes the sign strategy used by subsequent [`ScalarField::sample`]
    /// calls. Runtime-switchable by design (not a compile-time generic) so
    /// callers (e.g. a GUI toggle) can change it after construction without
    /// rebuilding the `MeshSdf`.
    pub fn set_sign_method(&mut self, method: SignMethod) {
        self.sign_method = method;
    }

    /// Current sign strategy.
    pub fn sign_method(&self) -> SignMethod {
        self.sign_method
    }

    /// The normal to use for sign determination at `closest` on triangle
    /// `face_idx`.
    ///
    /// Simplification (documented per the task spec): this crate has only a
    /// single "current nearest triangle" available in the query path, not a
    /// full mesh half-edge/adjacency structure. The textbook pseudonormal
    /// construction uses per-edge pseudonormals (average of the two
    /// adjacent faces' normals) for edge-Voronoi-region queries; instead,
    /// this implementation uses:
    /// - the precomputed vertex pseudonormal when `closest` coincides with
    ///   one of the triangle's vertices (vertex-Voronoi region), and
    /// - the nearest triangle's own flat face normal for both the face
    ///   interior *and* edge regions.
    ///
    /// This is a standard, acceptable approximation (edge regions are a
    /// measure-zero part of the surface and the face-normal fallback is
    /// only off by the angle between the two adjacent faces), but it is
    /// less accurate than full per-edge pseudonormals near sharp edges.
    fn feature_normal(&self, face_idx: usize, closest: DVec3) -> DVec3 {
        let verts = self.face_vertex_indices[face_idx];
        for &vertex_idx in &verts {
            if closest.distance_squared(self.vertex_positions[vertex_idx]) <= DEGENERATE_EPSILON {
                let pn = self.vertex_pseudonormals[vertex_idx];
                if pn.length_squared() > DEGENERATE_EPSILON {
                    return pn;
                }
                // Fall through to face normal if the vertex pseudonormal
                // itself is degenerate.
                break;
            }
        }
        self.face_normals[face_idx]
    }

    /// Sign (+1.0 or -1.0) for a query point `p` whose nearest triangle is
    /// `face_idx`, with closest surface point `closest`.
    fn sign_at(&self, face_idx: usize, closest: DVec3, p: DVec3) -> f64 {
        match self.sign_method {
            SignMethod::Pseudonormal => {
                let normal = self.feature_normal(face_idx, closest);
                let diff = p - closest;
                // On the surface (diff ~ zero) or a degenerate normal: sign
                // is immaterial since value ~= 0 either way, default to
                // positive (outside).
                if diff.dot(normal) < 0.0 {
                    -1.0
                } else {
                    1.0
                }
            }
        }
    }
}

impl ScalarField for MeshSdf {
    /// Samples the mesh SDF at `p`.
    ///
    /// Fallback for an empty mesh (no triangles): returns a large sentinel
    /// value (`f64::MAX`, "infinitely far outside") with a zero gradient,
    /// rather than panicking — there is no meaningful nearest point to
    /// report.
    ///
    /// If `p` lies exactly on the surface (`p == closest`), the gradient
    /// direction is arbitrary (`(p - closest)` cannot be normalized); the
    /// nearest triangle's feature normal is used instead so the gradient is
    /// still a finite, sensible unit vector rather than NaN.
    fn sample(&self, p: DVec3) -> FieldSample {
        match self.bvh.nearest(p) {
            None => FieldSample {
                value: f64::MAX,
                gradient: DVec3::ZERO,
            },
            Some((face_idx, closest, dist_sq)) => {
                let dist = dist_sq.sqrt();
                let sign = self.sign_at(face_idx, closest, p);

                let diff = p - closest;
                let gradient_dir = if diff.length_squared() > DEGENERATE_EPSILON {
                    diff.normalize()
                } else {
                    // p is (numerically) exactly on the surface: fall back
                    // to the feature normal so the gradient is a finite
                    // unit vector instead of NaN from normalizing zero.
                    let normal = self.feature_normal(face_idx, closest);
                    if normal.length_squared() > DEGENERATE_EPSILON {
                        normal
                    } else {
                        DVec3::X
                    }
                };

                FieldSample {
                    value: sign * dist,
                    gradient: sign * gradient_dir,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    /// Unit cube spanning [0,1]^3, as 12 triangles with outward winding.
    fn cube_mesh() -> (Vec<DVec3>, Vec<[usize; 3]>) {
        let vertices = vec![
            DVec3::new(0.0, 0.0, 0.0), // 0
            DVec3::new(1.0, 0.0, 0.0), // 1
            DVec3::new(1.0, 1.0, 0.0), // 2
            DVec3::new(0.0, 1.0, 0.0), // 3
            DVec3::new(0.0, 0.0, 1.0), // 4
            DVec3::new(1.0, 0.0, 1.0), // 5
            DVec3::new(1.0, 1.0, 1.0), // 6
            DVec3::new(0.0, 1.0, 1.0), // 7
        ];
        let faces = vec![
            // -Z (bottom), outward normal -Z
            [0, 2, 1],
            [0, 3, 2],
            // +Z (top), outward normal +Z
            [4, 5, 6],
            [4, 6, 7],
            // -Y (front), outward normal -Y
            [0, 1, 5],
            [0, 5, 4],
            // +Y (back), outward normal +Y
            [3, 7, 6],
            [3, 6, 2],
            // -X (left), outward normal -X
            [0, 4, 7],
            [0, 7, 3],
            // +X (right), outward normal +X
            [1, 2, 6],
            [1, 6, 5],
        ];
        (vertices, faces)
    }

    fn cube_sdf() -> MeshSdf {
        let (vertices, faces) = cube_mesh();
        MeshSdf::new(vertices, faces)
    }

    #[test]
    fn point_outside_face_has_positive_sign_and_correct_distance() {
        let sdf = cube_sdf();
        let sample = sdf.sample(DVec3::new(0.5, 0.5, 2.0));
        assert!(sample.value > 0.0);
        assert!(approx_eq(sample.value, 1.0, 1e-9));
    }

    #[test]
    fn point_inside_has_negative_sign_and_correct_distance() {
        let sdf = cube_sdf();
        // Center of the cube: nearest surface is any face, distance 0.5.
        let sample = sdf.sample(DVec3::new(0.5, 0.5, 0.5));
        assert!(sample.value < 0.0);
        assert!(approx_eq(sample.value, -0.5, 1e-9));
        assert!(sample.gradient.is_finite());
        // Gradient should be a unit vector (magnitude ~1), even though the
        // nearest point is ambiguous (medial axis / cube center).
        assert!(approx_eq(sample.gradient.length(), 1.0, 1e-9));
    }

    #[test]
    fn point_on_surface_has_near_zero_value_and_finite_gradient() {
        let sdf = cube_sdf();
        let sample = sdf.sample(DVec3::new(0.5, 0.5, 1.0));
        assert!(approx_eq(sample.value, 0.0, 1e-9));
        assert!(sample.gradient.is_finite());
        assert!(approx_eq(sample.gradient.length(), 1.0, 1e-9));
    }

    #[test]
    fn point_near_a_corner_has_correct_sign_and_distance() {
        let sdf = cube_sdf();
        // Just outside the (1,1,1) corner, along the diagonal.
        let p = DVec3::new(1.1, 1.1, 1.1);
        let sample = sdf.sample(p);
        assert!(sample.value > 0.0);
        let expected = (p - DVec3::new(1.0, 1.0, 1.0)).length();
        assert!(approx_eq(sample.value, expected, 1e-6));
    }

    #[test]
    fn point_inside_near_a_corner_has_negative_sign() {
        let sdf = cube_sdf();
        // Just inside the (0,0,0) corner, along the diagonal.
        let p = DVec3::new(0.05, 0.05, 0.05);
        let sample = sdf.sample(p);
        assert!(sample.value < 0.0);
    }

    #[test]
    fn empty_mesh_returns_sentinel_without_panicking() {
        let sdf = MeshSdf::new(Vec::new(), Vec::new());
        assert!(sdf.is_empty());
        let sample = sdf.sample(DVec3::new(0.0, 0.0, 0.0));
        assert_eq!(sample.value, f64::MAX);
        assert_eq!(sample.gradient, DVec3::ZERO);
    }

    #[test]
    fn set_sign_method_round_trips() {
        let mut sdf = cube_sdf();
        assert_eq!(sdf.sign_method(), SignMethod::Pseudonormal);
        sdf.set_sign_method(SignMethod::Pseudonormal);
        assert_eq!(sdf.sign_method(), SignMethod::Pseudonormal);
    }
}
