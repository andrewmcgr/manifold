//! Diagnostic tool: exports a 2D slice of order field values as CSV for visualization.
//! Usage: cargo run --release --bin probe_order_field -- <mesh.stl> <profile.json> <output.csv>
//!
//! Slices through the mesh's bounding box at Z = max_z - 5mm (captures near-tip/tip geometry),
//! samples the order field on a grid, and writes (X, Y, order_value) to CSV.
//! Infinite values are written as "inf"; unreachable regions show as flat white in plots.

use manifold_core::{order_field, stl, SlicerConfig};
use manifold_fidget::mesh_sdf::MeshSdf;
use std::fs;
use std::io::{Cursor, Write};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!("Usage: {} <mesh.stl> <profile.json> <output.csv>", args[0]);
        std::process::exit(1);
    }

    let mesh_path = &args[1];
    let profile_path = &args[2];
    let output_path = &args[3];

    // Load mesh
    let mesh_data = fs::read(mesh_path).expect("Failed to read mesh file");
    let cursor = Cursor::new(mesh_data);
    let mesh = stl::load_stl(cursor).expect("Failed to parse STL");

    // Load config (handle both {"machine": ..., "config": ...} and direct SlicerConfig)
    let config_json = fs::read_to_string(profile_path).expect("Failed to read profile");
    let val: serde_json::Value = serde_json::from_str(&config_json).expect("Failed to parse JSON");
    let config: SlicerConfig = if val.get("config").is_some() {
        serde_json::from_value(val["config"].clone()).expect("Failed to parse config subfield")
    } else {
        serde_json::from_value(val).expect("Failed to parse direct config")
    };

    let Some((min, max)) = mesh.bounding_box() else {
        eprintln!("Empty mesh");
        std::process::exit(1);
    };

    // Build order field (same as slicing pipeline)
    let faces: Vec<[usize; 3]> = mesh
        .indices
        .chunks_exact(3)
        .map(|chunk| [chunk[0] as usize, chunk[1] as usize, chunk[2] as usize])
        .collect();
    let sdf = MeshSdf::new(mesh.vertices.clone(), faces);
    let machine = manifold_core::machine::Machine::default();
    let slope_profile = machine.slope_profile();

    let field = order_field::order_field_for_with_sdf(
        config.order_field,
        &config,
        &mesh,
        &slope_profile,
        Some(&sdf),
    );

    // Slice near the top (within 5mm of max_z) to capture the arch region
    let slice_z = (max.z - 5.0).max(min.z + 0.1);
    eprintln!("Mesh bounds: Z ∈ [{}, {}]", min.z, max.z);
    eprintln!("Slicing at Z = {}", slice_z);

    // Grid resolution: 0.1mm spacing
    let grid_spacing = 0.1;
    let x_min = (min.x - 1.0).floor();
    let x_max = (max.x + 1.0).ceil();
    let y_min = (min.y - 1.0).floor();
    let y_max = (max.y + 1.0).ceil();

    let x_count = ((x_max - x_min) / grid_spacing).ceil() as usize;
    let y_count = ((y_max - y_min) / grid_spacing).ceil() as usize;

    eprintln!(
        "Grid: {} × {} points (X ∈ [{}, {}], Y ∈ [{}, {}])",
        x_count, y_count, x_min, x_max, y_min, y_max
    );

    // Sample and write
    let mut file = fs::File::create(output_path).expect("Failed to create output file");
    writeln!(file, "x,y,order").expect("Failed to write header");

    for i in 0..x_count {
        let x = x_min + (i as f64) * grid_spacing;
        for j in 0..y_count {
            let y = y_min + (j as f64) * grid_spacing;
            let p = glam::DVec3::new(x, y, slice_z);
            let order = field.order(p);
            let order_str = if order.is_infinite() {
                "inf".to_string()
            } else if order.is_nan() {
                "nan".to_string()
            } else {
                format!("{:.6}", order)
            };
            writeln!(file, "{:.6},{:.6},{}", x, y, order_str).expect("Failed to write row");
        }
    }

    eprintln!("Wrote {} rows to {}", x_count * y_count, output_path);
    eprintln!("\nTo visualize:");
    eprintln!("  python3 << 'EOF'");
    eprintln!("import pandas as pd; import matplotlib.pyplot as plt; import numpy as np");
    eprintln!("df = pd.read_csv('{}')", output_path);
    eprintln!("df_finite = df[df['order'] != 'inf']");
    eprintln!("if len(df_finite) > 0:");
    eprintln!("  df_finite['order'] = pd.to_numeric(df_finite['order'])");
    eprintln!(
        "  plt.scatter(df_finite['x'], df_finite['y'], c=df_finite['order'], cmap='viridis', s=1)"
    );
    eprintln!("  plt.colorbar(label='Order')");
    eprintln!(
        "  plt.axis('equal'); plt.title('Order field at Z={}'); plt.savefig('{}.png')",
        slice_z, output_path
    );
    eprintln!("EOF");
}
