# CODE_STYLE.md

Rust project — follow standard Rust conventions
(https://doc.rust-lang.org/1.0.0/style/) and idiomatic best practices.
Nothing here overrides `rustfmt`/`clippy` defaults; this file only notes
project-specific choices.

## Naming Conventions

- Crates: `manifold-<role>` (kebab-case), e.g. `manifold-core`,
  `manifold-cli`, `manifold-gui`. Library crate names use the corresponding
  snake_case in code (`manifold_core`).
- Modules: one noun per pipeline stage (`mesh`, `slicing`, `toolpath`,
  `gcode`), snake_case, matching the domain concept they own.
- Types: `UpperCamelCase` (`Mesh`, `SlicerConfig`, `Layer`, `Path`).
- Functions/variables: `snake_case`. Pipeline entry points read as verbs
  (`slice_mesh`, `plan`, `emit`).
- Error variants: `UpperCamelCase`, named after the failure domain
  (`InvalidMesh`, `Slicing`, `Toolpath`, `Io`).

## File Organization

- `manifold-core` is the only crate holding slicing domain logic. CLI/GUI
  crates are thin front-ends that call into it — do not duplicate slicing
  logic in a front-end.
- Each pipeline stage (mesh, slicing, toolpath, gcode) is its own module
  file directly under `src/`; keep that flat structure until a stage
  genuinely needs sub-modules (then convert to `stage/mod.rs` + files).
- Public API surface of `manifold-core` is re-exported from `lib.rs`
  (`pub use error::{Error, Result}`); keep `lib.rs` as the readable
  top-level pipeline description (`slice_to_gcode`), not a dumping ground.

## Import Style

- Group by: `std` first, then external crates, then internal
  (`crate::...`) — this matches default `rustfmt`/`cargo fmt` grouping.
  Run `cargo fmt --all` before committing; don't hand-format imports.
- Prefer explicit `use crate::{module::Type, other::Thing};` grouping over
  repeated single-item `use` lines.

## Code Patterns

- Geometry uses `glam::DVec3` (f64) throughout `manifold-core` — do not mix
  in `f32`/`Vec3` for domain math; front-ends convert at the boundary if a
  renderer needs `f32`.
- Config objects (`SlicerConfig`) derive `Serialize`/`Deserialize` and
  `Default` so they can be persisted and constructed without boilerplate.
  Add new slicing parameters there, not as loose function arguments.
- Pipeline functions take `&Type` inputs and return `Result<Owned>`
  (e.g. `slice_mesh(&Mesh, &SlicerConfig) -> Result<Vec<Layer>>`) — no
  hidden mutable state, no globals.

## Error Handling

- `manifold-core` (library code) uses `thiserror` — a single `Error` enum
  in `error.rs` with `#[error("...")]` messages and `#[from]` conversions
  (see `Error::Io`). Add new failure modes as new variants there, not as
  `String`/`Box<dyn Error>`.
- `manifold-cli` / `manifold-gui` (application code) use `anyhow::Result`
  at their boundaries (`main() -> anyhow::Result<()>`) and propagate with
  `?`. Do not introduce `anyhow` inside `manifold-core`.

## Logging

- Use `tracing` macros (`tracing::info!`, `tracing::warn!`, etc.) with
  structured fields, e.g. `tracing::info!(input = %path.display(), "...")`
  — prefer structured fields over interpolating everything into the
  message string.
- Initialize once per binary with `tracing_subscriber::fmt::init()` in
  `main()`; `manifold-core` should only emit `tracing` events, never
  initialize a subscriber itself.

## Testing

- Unit tests live in a `#[cfg(test)] mod tests` block at the bottom of the
  file they test (see `manifold-core/src/lib.rs`), following stdlib/Rust
  convention.
- Test names describe the behavior asserted, snake_case, no `test_`
  prefix (e.g. `default_config_is_sane`).
- Run the full suite with `cargo test --workspace` before committing.

## Do's and Don'ts

- ✅ Put new slicing/toolpath/Gcode logic in `manifold-core`.
- ✅ Add new config fields to `SlicerConfig`.
- ✅ Run `cargo fmt --all` and `cargo clippy --workspace --all-targets`
  before committing.
- ❌ Don't add UI, CLI parsing, or `anyhow` dependencies to
  `manifold-core`.
- ❌ Don't use `f32`/`Vec3` for core geometry — use `glam::DVec3`.
- ❌ Don't panic/`unwrap()` on user-controlled input (mesh files, CLI
  args) — return a typed `Error` instead.
