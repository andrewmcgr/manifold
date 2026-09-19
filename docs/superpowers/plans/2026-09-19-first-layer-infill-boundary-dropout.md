# First-Layer Infill Boundary Dropout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Find and fix the root cause of `Layer::infill_boundary` being entirely absent from roughly half a real object's footprint at its first few layers, even though the mesh SDF confirms real solid geometry exists there and the wall loops are correctly generated across the whole footprint.

**Architecture:** This is a root-cause debugging plan, not a from-scratch feature. Task 1 investigates (no fix yet) and produces a minimized, committable regression test that fails against current `main`. Task 2 fixes the confirmed root cause and makes Task 1's test pass. Task 3 adds any additional regression coverage the fix's blast radius warrants, re-verifies against the real reported case, and runs the final gate.

**Tech Stack:** Rust, existing `manifold-core` slicing pipeline (`crates/manifold-core/src/slicing.rs`), `polygon2d` module, `manifold-cli` scratch probe tooling.

**Spec:** `docs/superpowers/specs/2026-09-19-first-layer-infill-boundary-dropout-design.md`

## Global Constraints

- Core geometry: `glam::DVec3`/f64 only, no `f32`/`Vec3`.
- Do not touch `compute_solid_fill_boundaries` (`slicing.rs:3959`) unless the investigation proves it's actually implicated — current evidence says it isn't; it correctly returns empty when handed an already-empty `infill_boundary`.
- Do not touch wall-loop generation or `toolpath::plan` — confirmed unaffected by direct evidence (walls print correctly across the whole footprint at every layer checked).
- This repo's commit hook requires the subject line ≤72 characters in `<type>(<scope>): <subject>` format. Direct commits to `main` are blocked — this plan's work happens on `bug/first-layer-infill-boundary-dropout` (already created).
- After each task: `cargo fmt --all` -> `CARGO_TARGET_DIR=target/clippy cargo clippy --workspace --all-targets` -> `CARGO_TARGET_DIR=target/test cargo nextest run --workspace` must all pass. Full-workspace test runs on this repo can take several minutes on a rebuild — that's normal, not a hang.

---

### Task 1: Root-cause investigation and a failing regression test

**Files:**
- Likely modify: `crates/manifold-core/src/slicing.rs` (add a new `#[cfg(test)]` test only — no production code changes in this task)
- Read (not modify unless the investigation proves necessary): the candidate locations in the design spec's "Candidate Root-Cause Locations" section
- Available as evidence-gathering tools (already committed, do not need to recreate): `crates/manifold-cli/examples/probe_first_layer_dropout.rs`, `crates/manifold-cli/examples/probe_volume_audit_shell.rs`, `crates/manifold-cli/examples/probe_layer_dropouts.rs`
- The real repro files: `/Users/amcgregor/work/Manifold/TestObj1.stl`, `/Users/amcgregor/3D/profile.json` — both exist on disk, available for direct investigation via the probe tools above (e.g. `cargo run --release -p manifold-cli --example probe_first_layer_dropout -- /Users/amcgregor/work/Manifold/TestObj1.stl /Users/amcgregor/3D/profile.json 178.0 3`)

**Interfaces:**
- Produces: a written root-cause report (in your task report file) identifying the exact function/line where the affected region's geometry is dropped from `infill_boundary`, with evidence (not guessing).
- Produces: one new `#[test]` in `crates/manifold-core/src/slicing.rs`'s existing `mod tests` block, committed in a RED state (fails against current `main`/this branch's un-fixed code) — prefer a small, synthetic mesh fixture reproducing the same defect class; if genuinely impractical, a targeted test loading the real `TestObj1.stl`/`profile.json` files directly (matching the existing `diagnose_missing_overhang.rs`-style pattern of loading real saved profiles when synthetic repro isn't practical — see the design spec's Acceptance section for the exact bar).
- Later tasks consume: your root-cause report and the new test's exact name/location.

- [ ] **Step 1: Reproduce and orient**

Read the design spec in full: `docs/superpowers/specs/2026-09-19-first-layer-infill-boundary-dropout-design.md`. It already establishes (with evidence, not guessing): walls are unaffected, the mesh SDF confirms real geometry exists in the missing region, `compute_solid_fill_boundaries` is correctly propagating an already-empty `infill_boundary` rather than causing the emptiness itself, and `Layer::infill_boundary` is the actual missing artifact (confined to `x < 178` at layers 0-2 of the real repro, out of a full mesh bbox of `x ∈ [165, 185]`).

Run the existing probe to confirm you can reproduce the symptom yourself before touching any code:

```bash
cargo build --release -p manifold-cli --example probe_first_layer_dropout
./target/release/examples/probe_first_layer_dropout /Users/amcgregor/work/Manifold/TestObj1.stl /Users/amcgregor/3D/profile.json 178.0 3
```

(If `cargo build --release` writes to a different target dir than `./target/release/...` in your environment, adjust the binary path accordingly — check `CARGO_TARGET_DIR` if set, or just use `cargo run --release -p manifold-cli --example probe_first_layer_dropout -- ...` directly instead of building+running separately.)

You should see the same pattern the design spec describes: `infill_boundary in region: false` at layers 0-2, `true` at layer 3; wall counts present and matching in-region at every layer.

- [ ] **Step 2: Trace the actual derivation, using the candidate list as a starting point, not a checklist to blindly work through**

The design spec's "Candidate Root-Cause Locations" section names specific line ranges in `slicing.rs` where `infill_boundary` gets derived via `polygon2d::inward_offset` and per-island fallback logic. Use `read_symbol`/`read_enclosing`/direct reading to understand the actual control flow around each candidate, and use the debugger's basic tool — temporary `eprintln!`/`tracing::debug!` instrumentation compiled into a scratch probe run against the real repro — to see exactly which branch executes for the affected region and what value it produces there (empty vs. non-empty, and why).

Do not guess or pattern-match to what "sounds right" from the design spec's descriptions — verify with real instrumented output against the real repro, the same way `probe_first_layer_dropout.rs` itself was built on verified evidence rather than assumption.

Specific things worth checking directly, since they're the most likely failure shapes given the symptom (a real per-island region silently vanishing, not misclassified):
- Is the affected `x > 178` region a topologically separate island (a disconnected polygon) from the `x < 178` region at the wall-loop level, at these layers? (Check `layer.loops`' actual point sets/bounding boxes per polygon, not just aggregate wall counts — the earlier probe only checked aggregate loop-touches-region, not island separation.)
- If it is a separate island: does the per-island "deepest wall" / narrow-region fallback logic (`slicing.rs:6649-6652` and its surrounding function) correctly handle a *second, independent* island, or does something assume a single connected region and silently drop all but the first/largest one?
- Does `clean_first_layer_geometry`'s `filter_min_area` (`slicing.rs` around line 1774, threshold `nozzle_diameter^2 * 2`) get applied more than once, or does whatever calls it get invoked per-layer for layers 0-2 (not just the literal first layer, despite the function's name/doc)? If so, is the affected region's inward-offset area small enough to fall below that threshold even though the region itself is clearly not a "sub-bead micro-sliver" (its wall loops span thousands of points, matching the earlier probe's wall-loop point counts)?

- [ ] **Step 3: Confirm the root cause with a minimal experiment**

Once you have a hypothesis, prove it directly — e.g., temporarily patch the suspected function to behave differently and confirm the symptom resolves (then revert the temporary patch), or construct a small synthetic two-island mesh fixture and confirm it reproduces the SAME defect class (empty `infill_boundary` for one island despite real geometry existing there) using the *existing, unmodified* pipeline. Do not proceed to Step 4 until you have real, reproduced evidence pointing at one specific function/branch, not a plausible-sounding theory.

- [ ] **Step 4: Write the regression test**

Add a new `#[test]` function to `crates/manifold-core/src/slicing.rs`'s existing `mod tests` block (search for `#[cfg(test)]` / `mod tests` in that file — it already has many similar slicing-pipeline tests to mirror the style of, e.g. `compute_solid_fill_boundaries_covers_only_top_and_bottom_layers_leaving_the_interior_empty` for structure/naming conventions).

Prefer a small synthetic mesh (two separate islands, or whatever minimal shape your Step 3 confirmed reproduces the defect) over the real 100KB `TestObj1.stl`, for the same reasons the rest of this codebase's test suite avoids large real meshes (see `AGENTS.md`/the design spec's Plan E backlog item on test-suite performance). If a synthetic reduction genuinely isn't practical (document concretely why, don't just default to the real file for convenience), write a targeted test loading `/Users/amcgregor/work/Manifold/TestObj1.stl` + `/Users/amcgregor/3D/profile.json` directly — note this makes the test environment-dependent on files outside the repo, which is a real cost; if you go this route, at minimum extract just the relevant STL sub-region into a small fixture file committed under a `tests/fixtures/` (or similar, check for existing fixture conventions in this repo first) directory instead of depending on the user's own local file paths.

The test must:
- Fail (red) against the current, unmodified pipeline — confirm this by running it before any fix.
- Assert directly on `layer.infill_boundary` (or `layer.solid_fill_boundary`, whichever is more directly affected by the confirmed root cause) containing geometry in the region that's currently missing it — not an indirect proxy.

- [ ] **Step 5: Run the gate and commit**

```bash
cargo fmt --all
CARGO_TARGET_DIR=target/clippy cargo clippy --workspace --all-targets
CARGO_TARGET_DIR=target/test cargo nextest run --workspace
```

The new test is EXPECTED to fail at this point (red) — that's correct for this task; do not weaken it to pass artificially. Confirm every OTHER test still passes (no accidental regressions from any temporary instrumentation you added and should have since removed).

Commit (the new failing test is intentional and expected — say so in the commit message):

```bash
git add crates/manifold-core/src/slicing.rs
git commit -m "test(core): add failing repro for infill-boundary island drop"
```

Write your full report — the root-cause finding (function/line, with evidence), what you tried and ruled out, the exact test name/location, and full command output — to a report file (name it however the subagent-driven-development skill's `task-brief`/report conventions expect for this plan).

---

### Task 2: Fix the confirmed root cause

**Files:**
- `crates/manifold-core/src/slicing.rs` (the specific function Task 1 identified)

**Interfaces:**
- Consumes: Task 1's root-cause report and failing test (exact name/location).
- Produces: the fix, making Task 1's test pass.

- [ ] **Step 1: Read Task 1's report in full**

Do not re-investigate from scratch — Task 1 already did the diagnostic work. Read its report file for the exact function, the exact reason the region gets dropped, and the reasoning behind why that's a bug (not intentional filtering).

- [ ] **Step 2: Implement the fix**

Fix the confirmed root cause at its actual source (per `principle-fix-root-causes` — no guard/workaround that merely stops the specific symptom without addressing why it happens). If the root cause turns out to be in per-island handling not correctly iterating/preserving multiple independent islands, fix that generally (so it's correct for 2, 3, or N islands, not hardcoded for exactly 2) rather than specifically patching around `TestObj1.stl`'s particular shape.

If your fix touches shared geometry-processing logic (e.g. `polygon2d` helpers, or a function used by multiple call sites in `slicing.rs`), check for other call sites and confirm your fix doesn't change behavior for the single-island case (the overwhelmingly common case, already covered by many passing tests) — run the full test suite specifically watching for any newly-broken test in `slicing.rs`'s existing suite, not just Task 1's new one.

- [ ] **Step 3: Verify Task 1's test now passes**

Run the specific test Task 1 added and confirm it's green. Do not modify the test's assertions to make it pass artificially — if the fix doesn't make the EXACT test Task 1 wrote pass, either the fix is incomplete or Task 1's root-cause diagnosis needs revisiting (escalate/ledger this rather than weakening the test).

- [ ] **Step 4: Run the gate and commit**

```bash
cargo fmt --all
CARGO_TARGET_DIR=target/clippy cargo clippy --workspace --all-targets
CARGO_TARGET_DIR=target/test cargo nextest run --workspace
```

All tests must pass, including Task 1's now-green regression test and every pre-existing test in the suite.

```bash
git add crates/manifold-core/src/slicing.rs
git commit -m "fix(core): <describe the actual root cause fixed>"
```

---

### Task 3: Additional regression coverage, real-world verification, and final gate

**Files:**
- Possibly `crates/manifold-core/src/slicing.rs` (additional targeted tests, if the fix's blast radius warrants them)
- No production code changes expected in this task unless Task 2's re-verification below surfaces something new

**Interfaces:**
- Consumes: Task 2's fix.
- Produces: confirmation the real reported case is resolved, using the real files and the probe tooling already built this session.

- [ ] **Step 1: Consider additional test coverage**

Given the fix, are there adjacent cases worth a quick additional test (e.g. 3+ islands, an island that's genuinely below the sliver-filter threshold and SHOULD still be filtered, an island exactly at the threshold boundary)? Use judgment — don't manufacture tests for their own sake, but if the root cause was in general per-island logic, a 3-island case is cheap insurance that the fix generalizes rather than just covering the 2-island case Task 1's regression test happens to use.

- [ ] **Step 2: Re-verify against the real reported case**

Run, using the real repro files:

```bash
cargo run --release -p manifold-cli --example probe_first_layer_dropout -- /Users/amcgregor/work/Manifold/TestObj1.stl /Users/amcgregor/3D/profile.json 178.0 3
cargo run --release -p manifold-cli --example probe_volume_audit_shell -- /Users/amcgregor/work/Manifold/TestObj1.stl /Users/amcgregor/3D/profile.json 1.2
```

Confirm `probe_first_layer_dropout`'s output now shows `infill_boundary in region: true` at layers 0-2 (matching the rest of the footprint), and confirm `probe_volume_audit_shell`'s underfill listing no longer shows the dense cluster of near-*zero*-fill cells at `x ∈ [180, 183.6]`, `z < 3mm` that the design spec documented. Per the design spec's Acceptance section: some cells near the object's thin midline may still legitimately read below 100% (the user explicitly noted the object is under 1.2mm thick there) — that's expected and correct, not a regression; the proof here is about the dense near-*zero* cluster specifically, not perfect fill everywhere.

Include this real-world before/after comparison in your task report.

- [ ] **Step 3: Run the full gate**

```bash
cargo fmt --all
CARGO_TARGET_DIR=target/clippy cargo clippy --workspace --all-targets
CARGO_TARGET_DIR=target/test cargo nextest run --workspace
```

- [ ] **Step 4: Commit any additional tests**

```bash
git add crates/manifold-core/src/slicing.rs
git commit -m "test(core): add additional island-count regression coverage"
```

(Skip this commit if Step 1 concluded no additional tests were warranted — say so in your report either way.)
