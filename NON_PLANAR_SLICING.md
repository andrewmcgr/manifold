## NON_PLANAR_SLICING.md

Research spike: a functional-representation, field-driven toolpath
generation approach for Manifold's core non-planar slicing pipeline
(replacing/extending the current mesh -> `Layer[]` -> `Path[]` -> Gcode
pipeline in `manifold-core`). This is a **research and design document,
not an implementation plan** — no phase here should be treated as
committed work until its open questions are resolved and it's promoted
into `ROADMAP.md`.

## Motivation

Manifold's current slicing (`manifold-core::slicing`) cuts each object
with flat horizontal planes at a fixed layer height. That is fine for a
placeholder pipeline, but it is the opposite of what "non-planar slicer"
means for this project long-term: toolpaths that deform off the flat XY
plane, following printability constraints (overhangs, surface quality,
strength) rather than a stack of horizontal cuts.

## The proposed approach

1. **Represent the object as a signed distance field (SDF)**, not a
   discretized/grid volume — a function `f(p) -> f64` where `f(p) < 0`
   inside the shape, `> 0` outside, `0` on the boundary. Isosurfaces of
   this field at different offsets give inward/outward-offset shells of
   the object (useful for shells/perimeters), and `f(p) = 0` is the
   object's own surface.
2. **Represent desired deposition order as a second scalar field**,
   `order(p) -> f64`, monotonically increasing in the direction printing
   should proceed. Isosurfaces of `order` are the (potentially curved)
   layers — printing surfaces in increasing `order` never asks the
   toolhead to deposit material through/above not-yet-printed material,
   provided `order` correctly encodes printability constraints.
3. **`order` is derived from a local printing-direction vector field**,
   which itself depends on printability constraints — most importantly
   overhang angle (are we depositing onto existing material or into
   open air, given current gravity/toolhead orientation?), but
   eventually also strength/surface-quality objectives. `order` is
   obtained by integrating the direction field (see S³-Slicer below) —
   this is why it "may need to be updated frequently as slicing
   progresses": the direction field can be locally re-optimized as
   already-deposited geometry changes what's printable next.
4. **The angle-field primitive**: "given an isosurface of the SDF and a
   vector, what is an isosurface of the angle between the two?" — i.e.
   at each point on (or near) the surface, compute the angle between the
   local surface normal (`normalize(gradient(f))`) and a reference vector
   `v` (e.g. current gravity direction relative to the object). This
   derived scalar field `angle(p) = angle_between(normalize(grad f(p)), v)`
   directly answers: is this an overhang (`angle` near 0 when `v` points
   "down" and the surface faces down = unsupported)? Is this surface
   exposed vs. interior? This is the core building block for
   printability-constrained direction-field construction described in
   the literature below.

## Literature (this is prior art, not novel research)

- **S³-Slicer** — Zhang, Fang, Huang, Dutta, Lefebvre, Kilic, Wang,
  *"S³-Slicer: A General Slicing Framework for Multi-Axis 3D Printing"*,
  ACM TOG (SIGGRAPH Asia 2022), vol. 41, no. 6, article 277. **Best Paper
  Award.**
  - Paper: <https://dl.acm.org/doi/10.1145/3550454.3555516>
  - Project page: <https://guoxinfang.github.io/S3_Slicer>
  - Reference implementation (C++/Qt, MSVC + Intel oneMKL):
    <https://github.com/zhangty019/S3_DeformFDM>
  - Core idea, in their words: compute an optimal local printing-direction
    vector field satisfying fabrication objectives (support-free,
    strength reinforcement, surface quality) as a rotation-driven
    deformation of the input model; the *height field* of the deformed
    model maps back to a *scalar field* on the original (undeformed)
    shape; **isosurfaces of that scalar field are the curved layers**.
    Optimized via quaternion fields. This is precisely our proposed
    `order` field, validated at a top venue — not a novel unknown for us
    to invent, but an approach to adapt (their deformation-based
    construction is one way to build `order`; there may be simpler/more
    direct ways to build it straight from the angle-field primitive
    without a full deformation solve — open question below).
  - Related follow-on work referenced from the same group: curved-layer
    support generation (ICRA 2023), surface toolpath alignment with
    stress/fiber direction — "Reinforced FDM" (SIGGRAPH Asia 2020).
- **S⁴-Slicer** — simplified reimplementation by Joshua Bird:
  <https://github.com/jyjblrd/S4_Slicer> (video:
  <https://www.youtube.com/watch?v=M51bMMVWbC8>). Likely a much more
  approachable read than the full research codebase before diving into
  the original paper's math.
- Adjacent literature confirming this is an active, multi-group research
  area (not a one-off result): "Vector field-based curved layer slicing
  and path planning for multi-axis printing" (ScienceDirect, 2022);
  multiple 2024-2025 MDPI/Springer papers on curved-layer slicing for
  fiber-reinforced composites and surface-quality-enhanced curved
  slicing — same core pattern (direction field -> integrated scalar
  field -> isosurfaces), different objective functions.

## Crate candidate: `fidget`

<https://docs.rs/fidget> (mkeeter, MPL-2.0, actively maintained, v0.5) —
the strongest Rust crate match found so far for "functional rather than
explicit grid" plus "GPU-accelerated":

- Shapes are **closed-form expression trees** (`fidget::context::Tree`/
  `Context`), built from arithmetic ops — a genuine functional SDF
  representation, not a voxel grid. Matches our `f(p) -> f64` model
  directly.
- **`fidget::jit`**: compiles expression trees to native machine code for
  fast CPU evaluation (default feature).
- **`fidget::wgpu`**: GPU-accelerated evaluation/rendering, on the same
  `wgpu` stack `manifold-gui` already depends on — no new GPU API
  surface to learn.
- **Interval evaluation**: conservatively proves large spatial regions
  empty/full and can simplify the tape within a region — exactly the
  kind of pruning a deposition-order-aware toolpath search wants (skip
  regions that can't matter for the current pass).
- Evaluators return **values and partial derivatives** (gradient/normal)
  per point — directly what the angle-field primitive needs.
- **`fidget::mesh`**: Manifold Dual Contouring, implicit surface -> mesh,
  potentially reusable for extracting a printable curve/patch from an
  `order` isosurface, or adaptable for testing.

### Open questions on `fidget` (must resolve before committing)

1. **Composability of derived fields**: fidget's gradient/derivative
   output is numeric (per-evaluator-call), not a new `Tree` node you can
   feed back into fidget's own tape compiler/JIT/mesher. So `angle(p)`
   and `order(p)` (if built from angle-field integration) likely can't
   be expressed as fidget `Tree`s and get JIT/GPU-accelerated "for free"
   the same way the base SDF can — they'd need their own evaluation path
   (sampling SDF value + gradient via fidget, then computing
   angle/order in our own code), with our own spatial acceleration
   (could still borrow fidget's interval-arithmetic ideas, or build a
   thinner custom evaluator). Need to read fidget's `eval`/`var` modules
   closely, and possibly ask upstream (issues/discussions), before
   assuming this is a solved problem.
2. **Does `order` need to be re-evaluated as a *field* at all, or just
   sampled locally during toolpath walking?** S³-Slicer computes a
   global scalar field once (per deformation solve) and re-solves the
   deformation when constraints change materially. If our `order` field
   only needs local re-optimization as printing progresses (per the
   user's framing: "may need to be updated frequently as slicing
   progresses"), a full global re-solve every update could be too slow;
   worth investigating incremental/local update strategies vs. a
   from-scratch S³-Slicer-style solve.
3. **GPU compute pipeline shape**: `fidget::wgpu` is oriented at
   rendering (interactive preview). Whether it exposes a general
   GPU-compute entry point suitable for extracting/walking `order`
   isosurfaces (not just rasterizing to an image) needs checking against
   its actual API, not assumed from the module name.
4. **Where the SDF comes from**: `manifold-core` currently loads
   triangle meshes (STL/3MF) — an explicit boundary representation, not
   an SDF. Converting mesh -> SDF (or working with a hybrid: SDF for
   printability-field computation, mesh for import/final geometry) is
   its own sub-problem — needs either a mesh-to-SDF conversion step
   (signed distance to nearest triangle, with correct sign via
   winding/normal tests) or accepting SDFs as a first-class *input*
   modality alongside meshes (e.g. constructive solid geometry built
   directly as a `fidget::Tree`, bypassing mesh import entirely for
   SDF-native models).
5. **Licensing**: `fidget` is MPL-2.0 (file-level copyleft) — compatible
   with typical open-source use but distinct from `thiserror`/`anyhow`/
   `glam`'s permissive licenses already in the workspace; confirm this
   is acceptable for the project before depending on it.

## Alternatives to evaluate alongside `fidget`

Not yet researched in depth — listed so this spike doesn't anchor on the
first crate found:

- Hand-rolled SDF composition (closures/enum-of-ops) with `rayon` for
  CPU parallelism and no GPU path initially — simplest, most control,
  but reinvents JIT/interval-arithmetic infrastructure `fidget` already
  has.
- `wgpu` compute shaders written directly (WGSL) against a hand-designed
  SDF scene-description buffer, bypassing `fidget` entirely — more
  control over the GPU pipeline shape, more implementation work, no
  functional/JIT tree abstraction for free.
- Other implicit-surface/CSG-kernel prior art worth a literature/crate
  pass: ImplicitCAD, Antimony/Sitri, OpenVCAD (non-Rust, but relevant
  algorithmic prior art for functional CAD kernels using SDFs).

## Suggested spike structure (once the reading above is done)

1. Read S³-Slicer paper + `S3_DeformFDM`/`S4_Slicer` source closely;
   confirm whether the deformation-based `order` field construction is
   necessary, or whether a more direct angle-field-driven construction
   (closer to the user's original framing) is viable and simpler.
2. Prototype `fidget`: build a toy SDF (sphere/box), evaluate value +
   gradient at sample points, and hand-verify the angle-field primitive
   (`angle_between(normalize(grad f(p)), v)`) works as expected — outside
   `manifold-core`, in a scratch example, before any workspace
   integration.
3. Resolve the "open questions on `fidget`" above with direct
   experimentation/upstream questions, not assumption.
4. Only after 1-3: write an actual phased implementation plan and
   promote it into `ROADMAP.md` as new phases, following the existing
   "Research notes / crate decisions" + numbered-phase convention.

## Status

**Research spike, in progress.** No `manifold-core` domain-logic changes
should be made based on this document alone — see "Suggested spike
structure" above for the gate before this becomes an implementation
plan.
