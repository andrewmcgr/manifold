use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let backend_dir = manifest_dir.join("../backend");

    // Generate the CXX C++ bridge header/source and compile our FFI glue.
    // Note: in environments where OCCT / OpenVDB are installed, the CMake target
    // or pkg-config paths are linked alongside the CXX bridge target.
    let mut build = cxx_build::bridge("src/bridge.rs");
    build
        .file(backend_dir.join("src/slicer_backend.cpp"))
        .file(backend_dir.join("src/eikonal_solver.cpp"))
        .include(backend_dir.join("include"))
        .std("c++20")
        .flag_if_supported("-Wno-deprecated-declarations");

    // Standard Unix & macOS Homebrew include/library search paths for OCCT and OpenVDB
    for prefix in ["/usr/local", "/usr", "/opt/homebrew"] {
        let include_dir = PathBuf::from(prefix).join("include");
        let occt_include = include_dir.join("opencascade");
        if occt_include.exists() {
            build.include(&occt_include);
        }
        if include_dir.exists() {
            build.include(&include_dir);
        }
        let lib_dir = PathBuf::from(prefix).join("lib");
        if lib_dir.exists() {
            println!("cargo:rustc-link-search=native={}", lib_dir.display());
        }
    }

    // Allow skipping full C++ compilation in lightweight CI / cargo check environments
    // by setting MANIFOLD_SKIP_CPP_BUILD=1 if system CAD libraries are not installed.
    if std::env::var("MANIFOLD_SKIP_CPP_BUILD").is_err() {
        build.compile("slicer_backend_cxx");

        for lib in [
            "TKernel",
            "TKMath",
            "TKBRep",
            "TKTopAlgo",
            "TKGeomBase",
            "TKGeomAlgo",
            "TKMesh",
            "TKXSBase",
            "TKDESTEP",
            "openvdb",
            "tbb",
        ] {
            println!("cargo:rustc-link-lib=dylib={lib}");
        }
    }

    println!("cargo:rerun-if-changed=src/bridge.rs");
    println!("cargo:rerun-if-changed=../backend/include/slicer_backend.h");
    println!("cargo:rerun-if-changed=../backend/include/eikonal_solver.h");
    println!("cargo:rerun-if-changed=../backend/src/slicer_backend.cpp");
    println!("cargo:rerun-if-changed=../backend/src/eikonal_solver.cpp");
    println!("cargo:rerun-if-changed=../backend/CMakeLists.txt");
}
