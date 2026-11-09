use std::path::{Path, PathBuf};

fn lib_exists_in_dirs(lib_dirs: &[PathBuf], name: &str) -> bool {
    let candidates = [
        format!("lib{name}.dylib"),
        format!("lib{name}.so"),
        format!("lib{name}.a"),
    ];
    for dir in lib_dirs {
        for cand in &candidates {
            if dir.join(cand).exists() {
                return true;
            }
        }
    }
    false
}

fn main() {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let backend_dir = manifest_dir.join("../backend");

    // Generate the CXX C++ bridge header/source and compile our FFI glue.
    let mut build = cxx_build::bridge("src/bridge.rs");
    build
        .file(backend_dir.join("src/slicer_backend.cpp"))
        .file(backend_dir.join("src/eikonal_solver.cpp"))
        .file(backend_dir.join("src/path_generator.cpp"))
        .include(backend_dir.join("include"))
        .std("c++20")
        .flag_if_supported("-Wno-deprecated-declarations");

    // Collect candidate prefixes across macOS Homebrew, Linuxbrew, and standard Linux/Unix distros
    let mut prefixes: Vec<PathBuf> = Vec::new();
    if let Ok(hb_prefix) = std::env::var("HOMEBREW_PREFIX") {
        prefixes.push(PathBuf::from(hb_prefix));
    }
    for p in [
        "/opt/homebrew",
        "/home/linuxbrew/.linuxbrew",
        "/usr/local",
        "/usr",
    ] {
        let pb = PathBuf::from(p);
        if !prefixes.contains(&pb) {
            prefixes.push(pb);
        }
    }

    let mut discovered_lib_dirs: Vec<PathBuf> = Vec::new();
    for prefix in &prefixes {
        let include_dir = prefix.join("include");
        for sub in ["opencascade", "OpenCASCADE", "Imath", "OpenEXR"] {
            let sub_inc = include_dir.join(sub);
            if sub_inc.exists() {
                build.include(&sub_inc);
            }
        }
        if include_dir.exists() {
            build.include(&include_dir);
        }

        for lib_rel in [
            "lib",
            "lib64",
            "lib/x86_64-linux-gnu",
            "lib/aarch64-linux-gnu",
        ] {
            let lib_dir = prefix.join(lib_rel);
            if lib_dir.exists() && !discovered_lib_dirs.contains(&lib_dir) {
                println!("cargo:rustc-link-search=native={}", lib_dir.display());
                discovered_lib_dirs.push(lib_dir);
            }
        }
    }

    // Allow skipping full C++ compilation in lightweight CI / cargo check environments
    // by setting MANIFOLD_SKIP_CPP_BUILD=1 if system CAD libraries are not installed.
    if std::env::var("MANIFOLD_SKIP_CPP_BUILD").is_err() {
        build.compile("slicer_backend_cxx");

        // Core OpenCASCADE libraries present across OCCT 7.6, 7.7, and 7.8+
        for lib in [
            "TKernel",
            "TKMath",
            "TKBRep",
            "TKTopAlgo",
            "TKGeomBase",
            "TKGeomAlgo",
            "TKMesh",
            "TKXSBase",
        ] {
            println!("cargo:rustc-link-lib=dylib={lib}");
        }

        // OCCT 7.8+ consolidates STEP into TKDESTEP; OCCT 7.6/7.7 (Ubuntu/Debian) uses TKSTEP*
        if lib_exists_in_dirs(&discovered_lib_dirs, "TKDESTEP") {
            println!("cargo:rustc-link-lib=dylib=TKDESTEP");
        }
        for legacy_step_lib in ["TKSTEP", "TKSTEPAttr", "TKSTEP209", "TKSTEPBase"] {
            if lib_exists_in_dirs(&discovered_lib_dirs, legacy_step_lib) {
                println!("cargo:rustc-link-lib=dylib={legacy_step_lib}");
            }
        }

        println!("cargo:rustc-link-lib=dylib=openvdb");
        println!("cargo:rustc-link-lib=dylib=tbb");

        // On Linux, OpenVDB headers inline `Imath` half-float conversions that may reference libImath
        if Path::new("/etc/os-release").exists() && lib_exists_in_dirs(&discovered_lib_dirs, "Imath")
        {
            println!("cargo:rustc-link-lib=dylib=Imath");
        }
    }

    println!("cargo:rerun-if-changed=src/bridge.rs");
    for rel in [
        "include/slicer_backend.h",
        "include/eikonal_solver.h",
        "include/path_generator.h",
        "src/slicer_backend.cpp",
        "src/eikonal_solver.cpp",
        "src/path_generator.cpp",
        "CMakeLists.txt",
    ] {
        println!(
            "cargo:rerun-if-changed={}",
            backend_dir.join(rel).display()
        );
    }
}
