use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

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

fn copy_file_if_changed(src: &Path, dst: &Path) -> std::io::Result<()> {
    let src_bytes = std::fs::read(src)?;
    if let Ok(dst_bytes) = std::fs::read(dst) {
        if dst_bytes == src_bytes {
            return Ok(());
        }
    }
    if let Some(parent) = dst.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(dst, &src_bytes)
}

fn write_file_if_changed(dst: &Path, content: &str) -> std::io::Result<()> {
    if let Ok(existing) = std::fs::read_to_string(dst) {
        if existing == content {
            return Ok(());
        }
    }
    if let Some(parent) = dst.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(dst, content)
}

fn find_executable_in_path(name: &str) -> Option<PathBuf> {
    let candidate = Path::new(name);
    if candidate.is_absolute() || candidate.components().count() > 1 {
        if candidate.is_file() {
            return Some(candidate.to_path_buf());
        }
        if cfg!(windows) && candidate.extension().is_none() {
            let with_exe = candidate.with_extension("exe");
            if with_exe.is_file() {
                return Some(with_exe);
            }
        }
        return None;
    }

    let path_var = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path_var) {
        let full = dir.join(name);
        if full.is_file() {
            return Some(full);
        }
        if cfg!(windows) && full.extension().is_none() {
            for ext in ["exe", "cmd"] {
                let with_ext = dir.join(format!("{name}.{ext}"));
                if with_ext.is_file() {
                    return Some(with_ext);
                }
            }
        }
    }
    None
}

fn cmake_path(p: &Path) -> String {
    p.to_string_lossy().replace('\\', "/")
}

fn resolve_cxx_compiler_launcher() -> Option<String> {
    for var_name in ["CXX_WRAPPER", "CC_WRAPPER"] {
        if let Ok(val) = std::env::var(var_name) {
            let trimmed = val.trim();
            if trimmed.is_empty() {
                return None;
            }
            if let Some(resolved) = find_executable_in_path(trimmed) {
                return Some(cmake_path(&resolved));
            }
            return Some(trimmed.to_string());
        }
    }
    find_executable_in_path("ccache").map(|p| cmake_path(&p))
}

fn try_build_cpp_with_cmake(
    out_dir: &Path,
    backend_dir: &Path,
    extra_include_dirs: &[PathBuf],
    compiler_path: &Path,
    is_like_msvc: bool,
    target_os: &str,
    target_env: &str,
    profile: &str,
) -> bool {
    if std::env::var_os("MANIFOLD_CPP_LEGACY_CC").is_some() {
        return false;
    }
    let Some(cmake_exe) = find_executable_in_path("cmake") else {
        return false;
    };

    // Stabilize cxx-build generated headers & source so unrelated C++ edits do not bump their mtimes.
    let cxx_src_dir = out_dir.join("cxxbridge");
    let cxx_stable_dir = out_dir.join("cxxbridge_stable");
    let bridge_h_src = cxx_src_dir.join("include/manifold-frontend/src/bridge.rs.h");
    let bridge_h_dst = cxx_stable_dir.join("include/manifold-frontend/src/bridge.rs.h");
    let rust_cxx_h_src = cxx_src_dir.join("include/rust/cxx.h");
    let rust_cxx_h_dst = cxx_stable_dir.join("include/rust/cxx.h");
    let bridge_cc_src = cxx_src_dir.join("sources/manifold-frontend/src/bridge.rs.cc");
    let bridge_cc_dst = cxx_stable_dir.join("sources/manifold-frontend/src/bridge.rs.cc");

    if copy_file_if_changed(&bridge_h_src, &bridge_h_dst).is_err()
        || copy_file_if_changed(&rust_cxx_h_src, &rust_cxx_h_dst).is_err()
        || copy_file_if_changed(&bridge_cc_src, &bridge_cc_dst).is_err()
    {
        return false;
    }
    let bridge_alias_src = cxx_src_dir.join("include/manifold-frontend/src/bridge.rs");
    if bridge_alias_src.exists() {
        let _ = copy_file_if_changed(
            &bridge_alias_src,
            &cxx_stable_dir.join("include/manifold-frontend/src/bridge.rs"),
        );
    }

    let resolved_compiler = compiler_path
        .to_str()
        .and_then(find_executable_in_path)
        .unwrap_or_else(|| compiler_path.to_path_buf());
    let compiler_cmake = cmake_path(&resolved_compiler);

    let generator: Option<&str> = if find_executable_in_path("ninja").is_some() {
        Some("Ninja")
    } else if target_os == "windows" && target_env == "gnu" {
        Some("MinGW Makefiles")
    } else if target_os != "windows" {
        Some("Unix Makefiles")
    } else {
        None
    };

    let launcher = resolve_cxx_compiler_launcher();
    let use_pch = std::env::var_os("MANIFOLD_DISABLE_PCH").is_none()
        && !launcher
            .as_deref()
            .is_some_and(|l| l.to_ascii_lowercase().contains("sccache"));

    let mut include_dirs = Vec::with_capacity(extra_include_dirs.len() + 3);
    include_dirs.push(backend_dir.join("include"));
    include_dirs.push(cxx_stable_dir.join("include"));
    let crate_link_dir = cxx_src_dir.join("crate");
    if crate_link_dir.exists() {
        include_dirs.push(crate_link_dir);
    }
    include_dirs.extend(extra_include_dirs.iter().cloned());

    let mut include_cmake_lines = String::new();
    for dir in &include_dirs {
        include_cmake_lines.push_str(&format!("    \"{}\"\n", cmake_path(dir)));
    }

    let windows_defs = if target_os == "windows" {
        "target_compile_definitions(slicer_backend_cxx PRIVATE NOMINMAX _USE_MATH_DEFINES OPENVDB_DLL)\n"
    } else {
        ""
    };

    let compiler_flags = if is_like_msvc {
        "target_compile_options(slicer_backend_cxx PRIVATE /EHsc /bigobj /utf-8)\n"
    } else if target_os == "windows" {
        "target_compile_options(slicer_backend_cxx PRIVATE -Wno-deprecated-declarations -Wa,-mbig-obj)\n"
    } else {
        "target_compile_options(slicer_backend_cxx PRIVATE -Wno-deprecated-declarations)\n"
    };

    let pch_block = if use_pch {
        r#"if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.16")
    target_precompile_headers(slicer_backend_cxx PRIVATE
        <algorithm>
        <array>
        <atomic>
        <cmath>
        <cstdint>
        <functional>
        <limits>
        <memory>
        <queue>
        <string>
        <unordered_map>
        <unordered_set>
        <utility>
        <vector>
        <openvdb/openvdb.h>
        <openvdb/tools/Interpolation.h>
        <tbb/blocked_range.h>
        <tbb/parallel_for.h>
    )
endif()
"#
    } else {
        ""
    };

    let cmake_lists = format!(
        r#"cmake_minimum_required(VERSION 3.16)
project(slicer_backend_cxx LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(BRIDGE_CC "{bridge_cc}")
add_library(slicer_backend_cxx STATIC
    "${{BRIDGE_CC}}"
    "{slicer_cpp}"
    "{eikonal_cpp}"
    "{pathgen_cpp}"
)

set_source_files_properties("${{BRIDGE_CC}}" PROPERTIES SKIP_PRECOMPILE_HEADERS ON)

target_include_directories(slicer_backend_cxx PRIVATE
{include_cmake_lines})

{windows_defs}{compiler_flags}{pch_block}
set_target_properties(slicer_backend_cxx PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${{CMAKE_BINARY_DIR}}/lib"
    ARCHIVE_OUTPUT_DIRECTORY_DEBUG "${{CMAKE_BINARY_DIR}}/lib"
    ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${{CMAKE_BINARY_DIR}}/lib"
    ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO "${{CMAKE_BINARY_DIR}}/lib"
    ARCHIVE_OUTPUT_DIRECTORY_MINSIZEREL "${{CMAKE_BINARY_DIR}}/lib"
)
"#,
        bridge_cc = cmake_path(&bridge_cc_dst),
        slicer_cpp = cmake_path(&backend_dir.join("src/slicer_backend.cpp")),
        eikonal_cpp = cmake_path(&backend_dir.join("src/eikonal_solver.cpp")),
        pathgen_cpp = cmake_path(&backend_dir.join("src/path_generator.cpp")),
    );

    let cmake_src_dir = out_dir.join("cmake_src");
    let cmake_build_dir = out_dir.join("cmake_build");
    if write_file_if_changed(&cmake_src_dir.join("CMakeLists.txt"), &cmake_lists).is_err() {
        return false;
    }

    let build_type = if profile == "release" {
        "Release"
    } else {
        "RelWithDebInfo"
    };

    let relwithdeb_cxx_flags = if is_like_msvc {
        None
    } else {
        Some("-O2 -g1 -DNDEBUG")
    };

    let sig = format!(
        "cmake={}\ngen={}\ncompiler={}\nbuild_type={}\nlauncher={}\npch={}\nrelwithdeb={}\nenv_cxxflags={}\ncmakelists={}\n",
        cmake_exe.display(),
        generator.unwrap_or("default"),
        compiler_cmake,
        build_type,
        launcher.as_deref().unwrap_or(""),
        use_pch,
        relwithdeb_cxx_flags.unwrap_or(""),
        std::env::var("CXXFLAGS").unwrap_or_default(),
        cmake_lists
    );

    let sig_path = cmake_build_dir.join(".config_signature");
    let cache_path = cmake_build_dir.join("CMakeCache.txt");
    let need_configure = !cache_path.exists()
        || std::fs::read_to_string(&sig_path).ok().as_deref() != Some(sig.as_str());

    if need_configure {
        if cmake_build_dir.exists() {
            let _ = std::fs::remove_dir_all(&cmake_build_dir);
        }
        if std::fs::create_dir_all(&cmake_build_dir).is_err() {
            return false;
        }

        let mut cfg_cmd = Command::new(&cmake_exe);
        if let Some(gen) = generator {
            cfg_cmd.arg("-G").arg(gen);
            if gen == "MinGW Makefiles" {
                cfg_cmd.arg("-DCMAKE_SH=CMAKE_SH-NOTFOUND");
            }
        }
        cfg_cmd
            .arg("-S")
            .arg(&cmake_src_dir)
            .arg("-B")
            .arg(&cmake_build_dir)
            .arg(format!("-DCMAKE_CXX_COMPILER={compiler_cmake}"))
            .arg(format!("-DCMAKE_BUILD_TYPE={build_type}"));

        if let Some(flags) = relwithdeb_cxx_flags {
            cfg_cmd.arg(format!("-DCMAKE_CXX_FLAGS_RELWITHDEBINFO={flags}"));
        }
        if let Some(ref l) = launcher {
            cfg_cmd.arg(format!("-DCMAKE_CXX_COMPILER_LAUNCHER={l}"));
        }

        let Ok(cfg_out) = cfg_cmd.output() else {
            return false;
        };
        if !cfg_out.status.success() {
            let _ = std::fs::remove_dir_all(&cmake_build_dir);
            return false;
        }
        let _ = std::fs::write(&sig_path, &sig);
    }

    let mut build_cmd = Command::new(&cmake_exe);
    build_cmd
        .arg("--build")
        .arg(&cmake_build_dir)
        .arg("--config")
        .arg(build_type)
        .arg("--parallel");

    if launcher
        .as_deref()
        .is_some_and(|l| l.to_ascii_lowercase().contains("ccache"))
        && std::env::var_os("CCACHE_SLOPPINESS").is_none()
    {
        build_cmd.env(
            "CCACHE_SLOPPINESS",
            "pch_defines,time_macros,include_file_ctime,include_file_mtime",
        );
    }

    let Ok(build_out) = build_cmd.output() else {
        return false;
    };
    if !build_out.status.success() {
        let mut stderr = std::io::stderr();
        let _ = stderr.write_all(&build_out.stdout);
        let _ = stderr.write_all(&build_out.stderr);
        std::process::exit(build_out.status.code().unwrap_or(1));
    }

    let lib_dir = cmake_build_dir.join("lib");
    let built_unix = lib_dir.join("libslicer_backend_cxx.a").exists();
    let built_msvc = lib_dir.join("slicer_backend_cxx.lib").exists();
    if !built_unix && !built_msvc {
        return false;
    }

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=static=slicer_backend_cxx");
    true
}

fn main() {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let backend_dir = manifest_dir.join("../backend");
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_env = std::env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
    let profile = std::env::var("PROFILE").unwrap_or_else(|_| "debug".to_string());

    // Generate the CXX C++ bridge header/source and configure fallback compiler flags.
    let mut build = cxx_build::bridge("src/bridge.rs");
    build
        .file(backend_dir.join("src/slicer_backend.cpp"))
        .file(backend_dir.join("src/eikonal_solver.cpp"))
        .file(backend_dir.join("src/path_generator.cpp"))
        .include(backend_dir.join("include"))
        .std("c++20")
        .opt_level(if profile == "release" { 3 } else { 2 })
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
        "D:/a/_temp/msys64/ucrt64",
        "C:/msys64/mingw64",
        "C:/vcpkg/installed/x64-windows",
    ] {
        let pb = PathBuf::from(p);
        if pb.exists() && !prefixes.contains(&pb) {
            prefixes.push(pb);
        }
    }

    let mut discovered_include_dirs: Vec<PathBuf> = Vec::new();
    let mut discovered_lib_dirs: Vec<PathBuf> = Vec::new();
    for prefix in &prefixes {
        let include_dir = prefix.join("include");
        for sub in ["opencascade", "OpenCASCADE", "Imath", "OpenEXR"] {
            let sub_inc = include_dir.join(sub);
            if sub_inc.exists() {
                build.include(&sub_inc);
                if !discovered_include_dirs.contains(&sub_inc) {
                    discovered_include_dirs.push(sub_inc);
                }
            }
        }
        if include_dir.exists() {
            build.include(&include_dir);
            if !discovered_include_dirs.contains(&include_dir) {
                discovered_include_dirs.push(include_dir);
            }
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
        let compiler = build.get_compiler();
        let built_via_cmake = try_build_cpp_with_cmake(
            &out_dir,
            &backend_dir,
            &discovered_include_dirs,
            compiler.path(),
            compiler.is_like_msvc(),
            &target_os,
            &target_env,
            &profile,
        );
        if !built_via_cmake {
            build.compile("slicer_backend_cxx");
        }

        // Core OpenCASCADE libraries present across OCCT 7.6, 7.7, and 7.8+
        for lib in [
            "TKernel",
            "TKMath",
            "TKG2d",
            "TKG3d",
            "TKBRep",
            "TKTopAlgo",
            "TKGeomBase",
            "TKGeomAlgo",
            "TKMesh",
            "TKXSBase",
        ] {
            println!("cargo:rustc-link-lib=dylib={lib}");
        }

        // OCCT XCAF (Extended Data Exchange: STEPCAFControl_Reader, XCAFDoc_ShapeTool/ColorTool/MaterialTool)
        for xcaf_lib in [
            "TKCDF",
            "TKLCAF",
            "TKCAF",
            "TKService",
            "TKV3d",
            "TKVCAF",
            "TKXCAF",
            "TKXDESTEP",
        ] {
            if lib_exists_in_dirs(&discovered_lib_dirs, xcaf_lib) {
                println!("cargo:rustc-link-lib=dylib={xcaf_lib}");
            }
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

    for env_var in [
        "MANIFOLD_SKIP_CPP_BUILD",
        "MANIFOLD_CPP_LEGACY_CC",
        "MANIFOLD_DISABLE_PCH",
        "MANIFOLD_DEPS_PREFIX",
        "CXX",
        "CXX_WRAPPER",
        "CC_WRAPPER",
    ] {
        println!("cargo:rerun-if-env-changed={env_var}");
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
