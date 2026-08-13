# MESH_SDF_VISUALIZATION.md

Implementation plan: converting a `manifold-core::Mesh` into a signed
distance field (SDF) and adding isosurface/slice visualization tools for
it, integrated into `manifold-gui`. This builds directly on the
`fidget`-based scalar-field work started in `NON_PLANAR_SLICING.md`
(open question 4: "where the SDF comes from") and promotes the
`manifold-fidget` crate (formerly `manifold-fidget-spike`) from a research
scratch space to a real dependency of `manifold-gui`.

## Motivation

`NON_PLANAR_SLICING.md` assumed an SDF representation of the object but
left "mesh -> SDF" as an open question. Before any `manifold-core`
integration, we want working, testable mesh-to-SDF conversion plus tools
to *see* the result (isosurface extraction, slice views) — both to build
confidence in the conversion itself and because these are useful
debugging tools independent of the eventual toolpath-planning use case.

## Why this can't just be a `fidget::context::Tree`

`fidget`'s SDFs are closed-form expression trees built from arithmetic
ops. Nearest-triangle-on-a-mesh is a discrete spatial query (BVH lookup),
not an arithmetic expression — it cannot be expressed as a `Tree` and
compiled/JIT'd the same way a sphere or CSG primitive can. So mesh-derived
SDFs need their own evaluation path, and any code consuming "a scalar
field" (isosurface extraction, slice sampling) needs to be written against
a trait, not against `fidget::context::Tree` directly, so it works for
both a `Tree`-backed field and a mesh-derived one.

## Phases

### Phase A — `ScalarField` trait + mesh SDF (`manifold-fidget`)

- Introduce a `ScalarField` trait generalizing the existing free functions
  `evaluate()`/`angle_field()` (currently hardcoded to `&Tree`) to work
  over any field: `fn sample(&self, p: DVec3) -> FieldSample`. Keep
  `FieldSample { value, gradient }` as-is. Provide a `Tree`-backed impl so
  existing sphere-spike code/tests keep working unchanged.
- `MeshSdf` struct wrapping a mesh's triangles:
  - Spatial index (BVH, or a simple uniform grid if a BVH is more than
    needed for typical part sizes — pick whichever is simpler to get
    right first; a BVH is the more standard/robust choice) over triangles
    for nearest-triangle queries.
  - Distance = distance from query point to nearest point on the nearest
    triangle (standard point-to-triangle closest-point routine).
  - Gradient = normalized `(query_point - nearest_point)` — exact away
    from the medial axis, no finite differences needed.
- **Sign method as a runtime-switchable strategy, not a compile-time
  generic** — a `SignMethod` enum (or small trait object) stored on
  `MeshSdf`, settable/toggleable after construction:
  - `SignMethod::Pseudonormal` (implemented first): angle-weighted
    pseudonormals precomputed per vertex/edge/face at `MeshSdf`
    construction time (one-time O(triangles) setup); sign at query time
    is `sign(dot(query - nearest_point, pseudonormal_at_nearest_feature))`.
    Cheap (piggybacks on the same nearest-triangle query used for
    distance) but only correct on watertight, consistently-wound meshes —
    degrades near holes/cracks/flipped triangles.
  - `SignMethod::WindingNumber` (deliberately **not implemented in this
    pass** — stub variant, or omit from the enum with a doc comment
    marking it as the designed extension point): more expensive
    (hierarchical solid-angle evaluation over the BVL) but robust to
    non-watertight/non-manifold meshes, which are expected to be common
    once real mesh import exists. Structure `MeshSdf` (BVH already
    computed, feature precompute step already isolated) so adding this
    variant later doesn't require restructuring the nearest-query path.
  - The GUI toggle (Phase D) exposes this as a runtime setting, not a
    rebuild.
- Unit tests: build a mesh for a known primitive (e.g. an explicit cube
  as 12 triangles) and check distance/sign/gradient at known
  inside/outside/on-surface points, plus a point on the medial axis
  (interior) to confirm gradient behavior is sane (magnitude ~1, no
  NaN/panics) even where "nearest point" is ambiguous.

### Phase B — Isosurface extraction (`manifold-fidget`)

- Hand-rolled marching cubes over a regular grid, generic over
  `&dyn ScalarField` (or a generic `<F: ScalarField>`) so it works
  unmodified for both `Tree`-backed toy fields and `MeshSdf`.
- Per-vertex normals come directly from the field's gradient at the
  extracted vertex position — no separate normal-recomputation pass.
- Output: a triangle soup (positions + normals) in a form `manifold-gui`'s
  existing mesh-upload path can consume directly (see Phase D).
- Unit test: isosurface of a sphere `Tree` field at radius `r` — spot
  check that a sample of extracted vertices lie at distance `r` from the
  center (within grid-resolution tolerance).

### Phase C — Slice sampling (`manifold-fidget`)

- Sample a `&dyn ScalarField` over a 2D grid on an arbitrary plane
  (defined by an origin point + two orthonormal basis vectors), producing
  a `Vec<f32>` value grid (row-major, with grid dimensions). No GUI/wgpu
  dependency here — this is pure sampling, so it's independently testable
  and reusable if slice rendering ever needs a different consumer (e.g. a
  future in-viewport texture instead of a side-panel image).

### Phase D — GUI integration (`manifold-gui`)

- Add `manifold-fidget` as a workspace dependency of `manifold-gui`.
- New "SDF" debug panel:
  - Object picker (reuse whatever object-selection concept
    `manifold-gui` already has for the loaded mesh).
  - Sign-method toggle (Pseudonormal now; winding number entry
    disabled/greyed with a tooltip noting it's not yet implemented).
  - Iso-level control (numeric slider/field) and a "recompute" trigger —
    isosurface extraction is too expensive to redo every frame, so this
    is recompute-on-demand (button, or on parameter change with a
    debounce), not a per-frame operation.
  - Isosurface render: upload the Phase B triangle soup through the
    existing `render.rs` mesh pipeline (reusing `UploadedMesh`/
    `MeshPaintCallback`), rendered as a distinct overlay (e.g.
    semi-transparent or different color) so it's visually distinguishable
    from the real mesh, using the field-gradient normals from Phase B
    for shading (flat-shaded, matching the existing mesh renderer's
    style) rather than recomputing face normals.
  - Slice view: plane/Z control, Phase C grid rendered as a heatmap
    (`egui::ColorImage` -> `TextureHandle`) shown **in the side panel**
    for this pass — not embedded in the 3D viewport.

## Explicitly deferred (not blocking this plan)

- **In-viewport (main 3D window) slice-plane rendering.** Side-panel
  heatmap is the first pass; an in-viewport rendering of the slice plane
  is wanted eventually but is follow-up work, not part of Phase D.
- `SignMethod::WindingNumber` actual implementation (structure allows it;
  not built now).
- Live/per-frame SDF or isosurface recomputation (interactive dragging of
  iso-level/plane redoing marching cubes every frame) — recompute-on-demand
  only for this pass.
- GPU-accelerated marching cubes / use of `fidget::wgpu`.
- Non-mesh SDF sources (CSG-authored `Tree` shapes) exposed in the GUI
  panel — the panel targets mesh-derived SDFs first.

## Status

**Approved plan, not yet broken into subtasks.** Phases A-D above are
handed to task breakdown next; this document should be kept up to date
if scope changes during implementation, and promoted/cross-referenced
from `ROADMAP.md` once implementation begins in earnest.
