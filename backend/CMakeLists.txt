cmake_minimum_required(VERSION 3.20)
project(slicer_backend VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Append standard Unix and macOS Homebrew/system search paths
list(APPEND CMAKE_PREFIX_PATH
    "/usr/local"
    "/usr"
    "/opt/homebrew"
    "/opt/local"
)

# 1. Locate Open CASCADE Technology (OCCT) — LGPL-2.1 with OCCT Exception
find_package(OpenCASCADE REQUIRED)
message(STATUS "Found OpenCASCADE ${OpenCASCADE_VERSION} at ${OpenCASCADE_INCLUDE_DIR}")

# 2. Locate OpenVDB (and IlmBase/Imath + TBB) — MPL-2.0 / Apache-2.0
list(APPEND CMAKE_MODULE_PATH
    "/usr/local/lib/cmake/OpenVDB"
    "/usr/lib/cmake/OpenVDB"
    "/opt/homebrew/lib/cmake/OpenVDB"
)
find_package(OpenVDB REQUIRED)
find_package(TBB REQUIRED)

set(OCCT_LIBS
    TKernel
    TKMath
    TKBRep
    TKTopAlgo
    TKGeomBase
    TKGeomAlgo
    TKMesh
    TKXSBase
    TKDESTEP
    TKSTEP
    TKSTEPAttr
    TKSTEP209
    TKSTEPBase
)

add_library(slicer_backend STATIC
    src/slicer_backend.cpp
)

target_include_directories(slicer_backend
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        ${OpenCASCADE_INCLUDE_DIR}
)

# Filter OCCT libraries that exist in the active OCCT version (e.g., 7.7 vs 7.8+)
set(RESOLVED_OCCT_LIBS "")
foreach(lib IN LISTS OCCT_LIBS)
    if(TARGET ${lib})
        list(APPEND RESOLVED_OCCT_LIBS ${lib})
    endif()
endforeach()

target_link_libraries(slicer_backend
    PUBLIC
        ${RESOLVED_OCCT_LIBS}
        OpenVDB::openvdb
        TBB::tbb
)

install(TARGETS slicer_backend
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)
install(DIRECTORY include/ DESTINATION include)
