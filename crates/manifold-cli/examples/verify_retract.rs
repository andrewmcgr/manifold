//! Scratch verification for retract/unretract Gcode emission
//! (`gcode::emit`'s `G10`/`G11` insertion). Not part of the crate's public
//! surface or test suite -- run manually against the real test STL:
//!
//! ```sh
//! cargo run --release -p manifold-cli --example verify_retract -- pug_v4_l_sop_85mm.stl
//! ```
use glam::DVec3;
use manifold_core::bounds::BoundingVolume;
use manifold_core::ids::{ObjectId, ToolId};
use manifold_core::machine::Machine;
use manifold_core::object::Object;
use manifold_core::tool::Tool;
use manifold_core::{gcode, slicing, stl, toolpath, SlicerConfig};

fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).expect("usage: <stl path>");
    let file = std::fs::File::open(&path)?;
    let mesh = stl::load_stl(file)?;

    let object = Object::new(ObjectId(0), mesh, ToolId(0));
    let objects = vec![object];

    // Effectively unbounded build volume so this scratch check isolates
    // gcode-emission behavior from the unrelated build-volume-placement
    // issue (see verify_flat_nozzle.rs for the same workaround).
    let machine = Machine::new(
        BoundingVolume::Aabb {
            min: DVec3::splat(-1_000.0),
            max: DVec3::splat(1_000.0),
        },
        vec![Tool::new(ToolId(0), 0.4)],
    );

    let config = SlicerConfig::default();
    let layers = slicing::slice_workspace(&objects, &[ObjectId(0)], &config)?;
    let paths = toolpath::plan(&layers, &objects, &machine.tools, &config)?;
    let out = gcode::emit(&paths, &config);

    let g10_count = out.lines().filter(|l| *l == "G10").count();
    let g11_count = out.lines().filter(|l| *l == "G11").count();
    let travel_lines = out
        .lines()
        .filter(|l| l.starts_with("G0 ") && !l.contains('E'))
        .count();

    println!("G10 (retract) count: {g10_count}");
    println!("G11 (unretract) count: {g11_count}");
    println!("travel (G0, no E) move count: {travel_lines}");

    // Sanity: every G10 must be immediately followed (later in the file) by
    // exactly one matching G11 before the next G10 -- i.e. they alternate,
    // never double up.
    let mut retracted = true; // matches gcode::emit's initial assumption
    let mut violations = 0usize;
    for line in out.lines() {
        match line {
            "G10" => {
                if retracted {
                    violations += 1;
                }
                retracted = true;
            }
            "G11" => {
                if !retracted {
                    violations += 1;
                }
                retracted = false;
            }
            _ => {}
        }
    }
    println!("alternation violations: {violations}");

    Ok(())
}
