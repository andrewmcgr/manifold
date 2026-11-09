#!/usr/bin/env bash
set -euo pipefail

# Assembles a macOS application bundle (Manifold.app) for Manifold2.
# Usage:
#   ./scripts/bundle_macos.sh [--debug | --release] [--skip-build] [--out-dir <dir>]

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="release"
SKIP_BUILD=0
OUT_DIR=""

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
        --out-dir)
            OUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

if [[ -z "${OUT_DIR}" ]]; then
    OUT_DIR="${REPO_ROOT}/target/${PROFILE}/bundle/osx"
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

APP_DIR="${OUT_DIR}/Manifold.app"
CONTENTS_DIR="${APP_DIR}/Contents"
MACOS_DIR="${CONTENTS_DIR}/MacOS"
RESOURCES_DIR="${CONTENTS_DIR}/Resources"

echo "==> Assembling ${APP_DIR}..."
rm -rf "${APP_DIR}"
mkdir -p "${MACOS_DIR}" "${RESOURCES_DIR}"

cp "${BIN_PATH}" "${MACOS_DIR}/manifold-frontend"
chmod +x "${MACOS_DIR}/manifold-frontend"
ln -sf "manifold-frontend" "${MACOS_DIR}/Manifold"

cp "${REPO_ROOT}/packaging/macos/Info.plist" "${CONTENTS_DIR}/Info.plist"
printf "APPL????" > "${CONTENTS_DIR}/PkgInfo"

cp "${REPO_ROOT}/frontend/assets/Manifold.icns" "${RESOURCES_DIR}/Manifold.icns"
cp "${REPO_ROOT}/frontend/assets/app_icon_1024.png" "${RESOURCES_DIR}/app_icon_1024.png"

if [[ -d "${REPO_ROOT}/Test Objects" ]]; then
    mkdir -p "${RESOURCES_DIR}/Test Objects"
    cp -R "${REPO_ROOT}/Test Objects/." "${RESOURCES_DIR}/Test Objects/"
fi

if command -v codesign >/dev/null 2>&1; then
    echo "==> Applying ad-hoc code signature to ${APP_DIR}..."
    codesign --force --deep --sign - "${APP_DIR}"
fi

echo "==> Successfully bundled: ${APP_DIR}"
