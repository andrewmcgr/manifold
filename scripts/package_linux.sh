#!/usr/bin/env bash
set -euo pipefail

# Packages a Linux distribution directory and tarball for Manifold2.
# Usage:
#   ./scripts/package_linux.sh [--debug | --release] [--skip-build] [--prefix <dir>]

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="release"
SKIP_BUILD=0
STAGE_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --debug)
            PROFILE="debug"
            shift
            ;;
        --release)
            PROFILE="release"
            shift
            ;;
        --skip-build)
            SKIP_BUILD=1
            shift
            ;;
        --prefix)
            STAGE_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

if [[ -z "${STAGE_DIR}" ]]; then
    STAGE_DIR="${REPO_ROOT}/target/${PROFILE}/package/manifold-linux"
fi

if [[ "${SKIP_BUILD}" -eq 0 ]]; then
    echo "==> Building manifold-frontend (${PROFILE})..."
    if [[ "${PROFILE}" == "release" ]]; then
        cargo build --manifest-path "${REPO_ROOT}/frontend/Cargo.toml" --release
    else
        cargo build --manifest-path "${REPO_ROOT}/frontend/Cargo.toml"
    fi
fi

BIN_PATH="${REPO_ROOT}/target/${PROFILE}/manifold-frontend"
if [[ ! -x "${BIN_PATH}" ]]; then
    echo "Error: Binary not found at ${BIN_PATH}" >&2
    exit 1
fi

echo "==> Staging Linux package in ${STAGE_DIR}..."
rm -rf "${STAGE_DIR}"
mkdir -p \
    "${STAGE_DIR}/bin" \
    "${STAGE_DIR}/share/applications" \
    "${STAGE_DIR}/share/icons/hicolor/256x256/apps" \
    "${STAGE_DIR}/share/icons/hicolor/1024x1024/apps" \
    "${STAGE_DIR}/share/manifold/test-objects"

cp "${BIN_PATH}" "${STAGE_DIR}/bin/manifold-frontend"
chmod +x "${STAGE_DIR}/bin/manifold-frontend"
ln -sf "manifold-frontend" "${STAGE_DIR}/bin/manifold"

cp "${REPO_ROOT}/packaging/linux/manifold.desktop" "${STAGE_DIR}/share/applications/manifold.desktop"
cp "${REPO_ROOT}/frontend/assets/app_icon.png" "${STAGE_DIR}/share/icons/hicolor/256x256/apps/manifold.png"
cp "${REPO_ROOT}/frontend/assets/app_icon_1024.png" "${STAGE_DIR}/share/icons/hicolor/1024x1024/apps/manifold.png"

if [[ -d "${REPO_ROOT}/Test Objects" ]]; then
    cp -R "${REPO_ROOT}/Test Objects/." "${STAGE_DIR}/share/manifold/test-objects/"
fi

TARBALL="${STAGE_DIR}.tar.gz"
tar -C "$(dirname "${STAGE_DIR}")" -czf "${TARBALL}" "$(basename "${STAGE_DIR}")"
echo "==> Successfully packaged Linux archive: ${TARBALL}"
