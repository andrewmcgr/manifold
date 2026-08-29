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
use std::collections::HashMap;

/// Squared-length tolerance below which a vector is treated as zero (used
/// both for "query point coincides with the closest surface point" and for
/// degenerate triangle edges when computing incident-angle weights).
const DEGENERATE_EPSILON: f64 = 1e-12;

/// Strategy used to determine the sign (inside/outside) of a [`MeshSdf`]
/// sample.
///
/// [`SignMethod::Pseudonormal`] is the default, cheap fast path.
/// [`SignMethod::WindingNumber`] is a slower, O(triangles)-per-query but
/// robust alternative — see its doc comment. [`MeshSdf::sign_at`] also uses
/// the winding-number computation internally as the fallback for
/// [`SignMethod::Pseudonormal`]'s ambiguous-tie case (see `sign_at`'s doc
/// comment and the `AMBIGUOUS_COS_THRESHOLD` note), since the plain
/// most-decisive-tied-feature heuristic that previously lived there
/// (`sign_via_tie_break`) could still pick the wrong feature when two
/// non-adjacent, near-tied-distance features disagreed on sign — confirmed
/// on Voron_Design_Cube_v7.stl at point (723.223, 314.132, 21.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SignMethod {
    /// Angle-weighted pseudonormals, precomputed per vertex and per face at
    /// construction time. Sign at query time is
    /// `sign(dot(query - nearest_point, pseudonormal_at_nearest_feature))`.
    /// Cheap, but only correct on watertight, consistently-wound meshes.
    /// Falls back to [`MeshSdf::winding_number_sign`] for ambiguous ties
    /// (see [`MeshSdf::sign_at`]).
    Pseudonormal,
    /// Generalized winding number (Jacobson, Kavan &amp; Sorkine-Hornung):
    /// sums the signed solid angle every triangle in the mesh subtends at
    /// the query point (`sum / (4*pi)`), which is exactly 1.0 inside and
    /// 0.0 outside a closed, consistently-wound surface for *any* query
    /// point, with no notion of "nearest feature" or ties to break —
    /// robust to non-watertight / non-manifold meshes and immune to the
    /// nearest-feature-tie failure mode that [`SignMethod::Pseudonormal`]
    /// can hit. O(triangles) per query (no BVH acceleration), so it is
    /// intended for the rare ambiguous case or for correctness-over-speed
    /// use, not as the default for large meshes.
    WindingNumber,
}

/// A [`ScalarField`] backed by a triangle mesh's signed distance: distance
/// to the nearest triangle (via a [`TriangleBvh`]), signed by `sign_method`.
pub struct MeshSdf {
    bvh: TriangleBvh,
    /// Optional BVH of the watertight mesh when `bvh` is built from a subset of faces
    /// (e.g. `distance_faces` excluding horizontal caps). Used for fast O(log N) ray-parity checks.
    watertight_bvh: Option<TriangleBvh>,
    /// Flat face normal per triangle, same indexing as the BVH's
    /// (originally-supplied) triangle order. Used for face- and
    /// edge-Voronoi-region queries (see the simplification note on
    /// [`MeshSdf::feature_normal`]).
    face_normals: Vec<DVec3>,
    /// Vertex indices (into `vertex_positions`) for each triangle, same
    /// indexing as `face_normals`/the BVH.
    face_vertex_indices: Vec<[usize; 3]>,
    /// Vertex indices for the watertight boundary mesh, used exclusively by
    /// [`MeshSdf::winding_number_sign`].
    watertight_face_vertex_indices: Vec<[usize; 3]>,
    /// Original mesh vertex positions.
    vertex_positions: Vec<DVec3>,
    /// Angle-weighted pseudonormal per vertex, same indexing as
    /// `vertex_positions`.
    vertex_pseudonormals: Vec<DVec3>,
    /// Averaged face normal per undirected edge (key: the edge's two vertex
    /// indices, sorted so `(a, b)` and `(b, a)` map to the same entry).
    /// Used for edge-Voronoi-region sign queries (see
    /// [`MeshSdf::feature_normal`]) — the standard pseudonormal
    /// construction for edges is the average of the (up to two) adjacent
    /// faces' normals, which is only correct when both share the edge;
    /// boundary edges (only one adjacent face, e.g. a non-watertight mesh)
    /// fall back to that single face's normal.
    edge_pseudonormals: HashMap<(usize, usize), DVec3>,
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
        let edge_pseudonormals = Self::compute_edge_pseudonormals(&faces, &face_normals);

        let bvh = TriangleBvh::build(triangles);

        MeshSdf {
            bvh,
            watertight_bvh: None,
            face_normals,
            face_vertex_indices: faces.clone(),
            watertight_face_vertex_indices: faces,
            vertex_positions: vertices,
            vertex_pseudonormals,
            edge_pseudonormals,
            sign_method: SignMethod::Pseudonormal,
        }
    }

    /// Builds a `MeshSdf` using a dedicated set of `distance_faces` for distance/BVH
    /// and feature-normal queries, while retaining the full `watertight_faces` for
    /// generalized winding-number sign evaluation ([`MeshSdf::winding_number_sign`]).
    ///
    /// This allows excluding non-side faces (such as downward-facing bed-contact
    /// triangles) from distance calculations without breaking the topological
    /// closure required for reliable interior/exterior winding-number classification.
    pub fn new_with_distance_faces(
        vertices: Vec<DVec3>,
        watertight_faces: Vec<[usize; 3]>,
        distance_faces: Vec<[usize; 3]>,
    ) -> Self {
        let triangles: Vec<Triangle> = distance_faces
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
                    DVec3::ZERO
                }
            })
            .collect();

        let vertex_pseudonormals =
            Self::compute_vertex_pseudonormals(&vertices, &distance_faces, &face_normals);
        let edge_pseudonormals = Self::compute_edge_pseudonormals(&distance_faces, &face_normals);

        let bvh = TriangleBvh::build(triangles);

        let watertight_bvh = if distance_faces.len() == watertight_faces.len() {
            None
        } else {
            let watertight_triangles: Vec<Triangle> = watertight_faces
                .iter()
                .map(|f| Triangle::new(vertices[f[0]], vertices[f[1]], vertices[f[2]]))
                .collect();
            Some(TriangleBvh::build(watertight_triangles))
        };

        MeshSdf {
            bvh,
            watertight_bvh,
            face_normals,
            face_vertex_indices: distance_faces,
            watertight_face_vertex_indices: watertight_faces,
            vertex_positions: vertices,
            vertex_pseudonormals,
            edge_pseudonormals,
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

    /// Precomputes averaged edge normals: for each undirected edge, sums
    /// the face normals of every triangle incident to it (1 for a boundary
    /// edge, 2 for a proper manifold edge, more only for a non-manifold
    /// mesh) and normalizes. This is the piece the original face-normal-only
    /// `feature_normal` fallback was missing — a query point whose closest
    /// surface point lies on an edge shared by two triangles with very
    /// different normals (e.g. a vertical wall meeting a flat cap at a
    /// sharp angle) previously used just the BVH's arbitrarily-chosen
    /// nearest triangle's own normal, which can be nearly orthogonal to the
    /// true outward direction and flip the sign.
    fn compute_edge_pseudonormals(
        faces: &[[usize; 3]],
        face_normals: &[DVec3],
    ) -> HashMap<(usize, usize), DVec3> {
        let mut accum: HashMap<(usize, usize), DVec3> = HashMap::new();
        for (face, &normal) in faces.iter().zip(face_normals.iter()) {
            for corner in 0..3 {
                let a = face[corner];
                let b = face[(corner + 1) % 3];
                let key = if a < b { (a, b) } else { (b, a) };
                *accum.entry(key).or_insert(DVec3::ZERO) += normal;
            }
        }
        accum
            .into_iter()
            .map(|(key, n)| {
                let normal = if n.length_squared() > DEGENERATE_EPSILON {
                    n.normalize()
                } else {
                    // Both adjacent faces degenerate, or their normals
                    // exactly cancel (e.g. a zero-thickness fold): no
                    // well-defined edge normal. `feature_normal` treats a
                    // zero result as "no edge information" and falls back
                    // to the nearest triangle's own face normal.
                    DVec3::ZERO
                };
                (key, normal)
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

    /// Returns the nearest triangle index, closest point on that triangle,
    /// and squared distance to it.
    pub fn nearest(&self, p: DVec3) -> Option<(usize, DVec3, f64)> {
        self.bvh.nearest(p)
    }

    /// The normal to use for sign determination at `closest` on triangle
    /// `face_idx`.
    ///
    /// Uses the standard three-region pseudonormal construction:
    /// - the precomputed vertex pseudonormal when `closest` coincides with
    ///   one of the triangle's vertices (vertex-Voronoi region);
    /// - the precomputed edge pseudonormal (averaged normal of the faces
    ///   sharing that edge) when `closest` lies on one of the triangle's
    ///   edges, identified via `closest`'s barycentric coordinates having a
    ///   near-zero component (edge-Voronoi region); and
    /// - the nearest triangle's own flat face normal otherwise (face
    ///   interior), or as a fallback if the vertex/edge pseudonormal is
    ///   itself degenerate.
    ///
    /// The edge case matters because the BVH may report any one of
    /// multiple equidistant triangles as "nearest" for a point whose
    /// closest surface point sits exactly on a shared edge; using only that
    /// triangle's own face normal (as a prior, simpler version of this
    /// method did) is only a good approximation when the two faces sharing
    /// the edge have similar normals, and can flip the sign entirely when
    /// they're near-perpendicular (e.g. a vertical wall meeting a flat
    /// cap).
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

        if let Some(edge_key) = Self::edge_containing_point(verts, &self.vertex_positions, closest)
        {
            if let Some(&en) = self.edge_pseudonormals.get(&edge_key) {
                if en.length_squared() > DEGENERATE_EPSILON {
                    return en;
                }
            }
        }

        self.face_normals[face_idx]
    }

    /// If `closest` (already known to lie within triangle `verts`, and
    /// already checked *not* to coincide with any of its vertices) lies on
    /// one of the triangle's three edges, returns that edge's key (sorted
    /// vertex-index pair) for an [`MeshSdf::edge_pseudonormals`] lookup.
    /// Returns `None` if `closest` is strictly in the triangle's interior.
    ///
    /// Identifies the edge via barycentric coordinates: `closest`'s
    /// barycentric weight for a vertex is ~0 exactly when `closest` lies on
    /// the opposite edge.
    fn edge_containing_point(
        verts: [usize; 3],
        vertex_positions: &[DVec3],
        closest: DVec3,
    ) -> Option<(usize, usize)> {
        let a = vertex_positions[verts[0]];
        let b = vertex_positions[verts[1]];
        let c = vertex_positions[verts[2]];

        let v0 = b - a;
        let v1 = c - a;
        let v2 = closest - a;
        let d00 = v0.dot(v0);
        let d01 = v0.dot(v1);
        let d11 = v1.dot(v1);
        let d20 = v2.dot(v0);
        let d21 = v2.dot(v1);
        let denom = d00 * d11 - d01 * d01;
        if denom.abs() <= DEGENERATE_EPSILON {
            // Degenerate (zero-area/collinear) triangle: barycentric
            // coordinates aren't well-defined, so there's no meaningful
            // edge to report.
            return None;
        }

        // Barycentric weights for (a, b, c) respectively; `bary_b` is the
        // weight on `b`, etc. Weight ~0 on a vertex means `closest` lies on
        // the edge opposite that vertex.
        let bary_c = (d11 * d20 - d01 * d21) / denom;
        let bary_b = (d00 * d21 - d01 * d20) / denom;
        let bary_a = 1.0 - bary_b - bary_c;

        const EDGE_EPSILON: f64 = 1e-9;
        if bary_a.abs() <= EDGE_EPSILON {
            let (x, y) = (verts[1], verts[2]);
            Some(if x < y { (x, y) } else { (y, x) })
        } else if bary_b.abs() <= EDGE_EPSILON {
            let (x, y) = (verts[0], verts[2]);
            Some(if x < y { (x, y) } else { (y, x) })
        } else if bary_c.abs() <= EDGE_EPSILON {
            let (x, y) = (verts[0], verts[1]);
            Some(if x < y { (x, y) } else { (y, x) })
        } else {
            None
        }
    }

    fn sign_at(&self, _face_idx: usize, closest: DVec3, p: DVec3) -> f64 {
        match self.sign_method {
            SignMethod::Pseudonormal => {
                let diff = p - closest;
                let dist = diff.length();
                if dist <= DEGENERATE_EPSILON.sqrt() {
                    // On the surface: sign is immaterial since value ~= 0
                    // either way, default to positive (outside).
                    return 1.0;
                }

                self.fast_parity_sign(p)
            }
            SignMethod::WindingNumber => self.winding_number_sign(p),
        }
    }

    /// Fast BVH-accelerated multi-ray parity sign check: casts rays along 3 slightly
    /// non-axis-aligned directions through the BVH in $O(\log N)$ time, using majority vote.
    pub fn fast_parity_sign(&self, p: DVec3) -> f64 {
        const DIRS: [DVec3; 3] = [
            DVec3::new(1.0, 0.0173, 0.0091),
            DVec3::new(0.0091, 1.0, 0.0173),
            DVec3::new(0.0173, 0.0091, 1.0),
        ];
        let bvh = self.watertight_bvh.as_ref().unwrap_or(&self.bvh);
        let mut inside_votes = 0;
        for dir in DIRS {
            if bvh.ray_crossings(p, dir) % 2 == 1 {
                inside_votes += 1;
            }
        }
        if inside_votes >= 2 {
            -1.0
        } else {
            1.0
        }
    }

    /// Generalized winding number sign for a query point `p`: sums the
    /// signed solid angle every triangle in the mesh subtends at `p` (via
    /// the Van Oosterom & Strackee tangent formula for a triangle's solid
    /// angle, which is exact and avoids the branch-cut/atan2-per-edge
    /// pitfalls of naive spherical-excess formulas) and normalizes by
    /// `4*pi`. For a closed, consistently-wound surface this winding
    /// number is exactly 1.0 for any point strictly inside and 0.0 for any
    /// point strictly outside, regardless of how many features are
    /// equidistant from `p` — unlike a nearest-feature approach, it never
    /// needs to decide "which" feature to trust.
    ///
    /// O(triangles): every triangle contributes regardless of distance to
    /// `p`, so this is only used for [`SignMethod::WindingNumber`] itself
    /// or as the rare ambiguous-tie fallback in [`MeshSdf::sign_at`], never
    /// on the hot common-case path.
    ///
    /// Degenerate (zero-area) triangles contribute (numerically) zero
    /// solid angle and are harmless to include unconditionally.
    fn winding_number_sign(&self, p: DVec3) -> f64 {
        let mut solid_angle_sum = 0.0f64;
        for verts in &self.watertight_face_vertex_indices {
            let a = self.vertex_positions[verts[0]] - p;
            let b = self.vertex_positions[verts[1]] - p;
            let c = self.vertex_positions[verts[2]] - p;

            let a_len = a.length();
            let b_len = b.length();
            let c_len = c.length();
            if a_len <= DEGENERATE_EPSILON.sqrt()
                || b_len <= DEGENERATE_EPSILON.sqrt()
                || c_len <= DEGENERATE_EPSILON.sqrt()
            {
                // p coincides with a vertex: solid angle is ill-defined;
                // this triangle's contribution is skipped (matches the
                // "on the surface, sign is immaterial" handling
                // elsewhere).
                continue;
            }

            // Van Oosterom & Strackee (1983): solid angle subtended by a
            // triangle with vertices a, b, c (relative to the query
            // point) is 2*atan2(numerator, denominator) where:
            let numerator = a.dot(b.cross(c));
            let denominator =
                a_len * b_len * c_len + a.dot(b) * c_len + b.dot(c) * a_len + c.dot(a) * b_len;
            solid_angle_sum += 2.0 * numerator.atan2(denominator);
        }

        let winding_number = solid_angle_sum / (4.0 * std::f64::consts::PI);
        // Inside the mesh: winding_number ~= 1.0 -> sign = -1.0 (inside).
        // Outside the mesh: winding_number ~= 0.0 -> sign = +1.0 (outside).
        if winding_number >= 0.5 {
            -1.0
        } else {
            1.0
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

    /// Regression test for a real-world tie: two triangles sharing a
    /// shared vertex (but not sharing an edge with each other in a way
    /// that matters here) can report the *exact* same minimal distance to
    /// a distant query point via two different edges, one of which (here,
    /// the near-flat interior "cap" edge) has a pseudonormal that is
    /// nearly perpendicular to the true displacement direction — a weak,
    /// unreliable signal that used to flip the sign to "inside" for a
    /// point that is unambiguously far outside the mesh. Coordinates are
    /// taken verbatim from the failing case (a wall triangle meeting a
    /// flat bottom cap on a real STL mesh).
    #[test]
    fn tied_distance_to_two_edges_from_a_shared_vertex_resolves_to_the_decisive_normal() {
        let v59 = DVec3::new(131.707106590271, 93.84999990463257, 5.703026968363187e-11);
        let v58 = DVec3::new(130.8071050643921, 92.99156427383423, 0.7644127607915876);
        let v60 = DVec3::new(130.8071050643921, 92.94999980926514, 5.703037990183537e-11);
        let v371 = DVec3::new(
            129.07702159881592,
            93.60407066345215,
            5.7030269185100044e-11,
        );

        // Index 0=v59, 1=v58, 2=v60, 3=v371.
        let vertices = vec![v59, v58, v60, v371];
        let faces = vec![
            [0, 1, 2], // wall triangle (59, 58, 60)
            [2, 3, 0], // cap triangle (60, 371, 59)
        ];
        let sdf = MeshSdf::new(vertices, faces);

        let bad_point = DVec3::new(149.55099233709944, 74.59991149106668, 1.0);
        let sample = sdf.sample(bad_point);

        // This point is far outside the mesh (well beyond its x/y extent);
        // the correct sign is positive (outside).
        assert!(
            sample.value > 0.0,
            "expected positive (outside) sign, got {}",
            sample.value
        );
        assert!(approx_eq(sample.value, 26.248457115795734, 1e-6));
    }

    #[test]
    fn set_sign_method_round_trips() {
        let mut sdf = cube_sdf();
        assert_eq!(sdf.sign_method(), SignMethod::Pseudonormal);
        sdf.set_sign_method(SignMethod::Pseudonormal);
        assert_eq!(sdf.sign_method(), SignMethod::Pseudonormal);
    }
}
