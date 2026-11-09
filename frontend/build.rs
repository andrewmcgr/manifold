use std::path::{Path, PathBuf};

fn lib_exists_in_dirs(lib_dirs: &[PathBuf], name: &str) -> bool {
    let candidates = [
        format!("lib{name}.dylib"),
        format!("lib{name}.so"),
        format!("lib{name}.a"),
        format!("lib{name}.dll.a"),
        format!("{name}.lib"),
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

fn find_matching_lib_stem(lib_dirs: &[PathBuf], prefix: &str) -> Option<String> {
    for dir in lib_dirs {
        let Ok(entries) = std::fs::read_dir(dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let file_name = entry.file_name();
            let Some(name_str) = file_name.to_str() else {
                continue;
            };
            // Match e.g. libImath-3_1.dll.a, Imath-3_1.lib, libImath.so, etc.
            let stripped = name_str.strip_prefix("lib").unwrap_or(name_str);
            if !stripped.starts_with(prefix) {
                continue;
            }
            for suffix in [".dll.a", ".dylib", ".so", ".a", ".lib"] {
                if let Some(stem) = stripped.strip_suffix(suffix) {
                    return Some(stem.to_string());
                }
            }
        }
    }
    None
}

fn main() {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let backend_dir = manifest_dir.join("../backend");
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_env = std::env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();

    // Generate the CXX C++ bridge header/source and compile our FFI glue.
    let mut build = cxx_build::bridge("src/bridge.rs");
    build
        .file(backend_dir.join("src/slicer_backend.cpp"))
        .file(backend_dir.join("src/eikonal_solver.cpp"))
        .file(backend_dir.join("src/path_generator.cpp"))
        .include(backend_dir.join("include"))
        .std("c++20")
        .flag_if_supported("-Wno-deprecated-declarations");

    if target_os == "windows" {
        build.define("NOMINMAX", None);
        build.define("_USE_MATH_DEFINES", None);
        build.define("OPENVDB_DLL", None);
        if target_env == "msvc" {
            build.flag_if_supported("/EHsc");
            build.flag_if_supported("/bigobj");
            build.flag_if_supported("/utf-8");
        } else {
            build.flag_if_supported("-Wa,-mbig-obj");
        }
    }

    // Collect candidate prefixes across macOS Homebrew, Linuxbrew, Windows (vcpkg / MSYS2 / Conda), and Unix distros
    let mut prefixes: Vec<PathBuf> = Vec::new();
    for env_var in [
        "MANIFOLD_DEPS_PREFIX",
        "HOMEBREW_PREFIX",
        "VCPKG_ROOT",
        "MINGW_PREFIX",
        "CONDA_PREFIX",
    ] {
        if let Ok(val) = std::env::var(env_var) {
            let pb = PathBuf::from(&val);
            if env_var == "VCPKG_ROOT" {
                let triplet = std::env::var("VCPKG_DEFAULT_TRIPLET")
                    .unwrap_or_else(|_| "x64-windows".to_string());
                let vcpkg_installed = pb.join("installed").join(triplet);
                if vcpkg_installed.exists() {
                    prefixes.push(vcpkg_installed);
                }
            }
            if !prefixes.contains(&pb) {
                prefixes.push(pb);
            }
        }
    }
    for p in [
        "/opt/homebrew",
        "/home/linuxbrew/.linuxbrew",
        "/usr/local",
        "/usr",
        "C:/msys64/ucrt64",
        "C:/msys64/mingw64",
        "C:/vcpkg/installed/x64-windows",
    ] {
        let pb = PathBuf::from(p);
        if pb.exists() && !prefixes.contains(&pb) {
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
        if lib_exists_in_dirs(&discovered_lib_dirs, "tbb") {
            println!("cargo:rustc-link-lib=dylib=tbb");
        } else if lib_exists_in_dirs(&discovered_lib_dirs, "tbb12") {
            println!("cargo:rustc-link-lib=dylib=tbb12");
        } else {
            println!("cargo:rustc-link-lib=dylib=tbb");
        }

        // On Linux and Windows, OpenVDB headers inline `Imath` half-float conversions that reference libImath
        if Path::new("/etc/os-release").exists() || target_os == "windows" {
            if lib_exists_in_dirs(&discovered_lib_dirs, "Imath") {
                println!("cargo:rustc-link-lib=dylib=Imath");
            } else if let Some(imath_stem) = find_matching_lib_stem(&discovered_lib_dirs, "Imath-") {
                println!("cargo:rustc-link-lib=dylib={imath_stem}");
            }
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
