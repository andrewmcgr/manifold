#!/usr/bin/env bash
# Build and package a minimal standalone macOS .app bundle for Manifold GUI.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

RELEASE_FLAG="--release"
BUILD_PROFILE="release"
OPEN_AFTER_BUILD=false
ZIP_BUNDLE=false

for arg in "$@"; do
    case "$arg" in
        --debug)
            RELEASE_FLAG=""
            BUILD_PROFILE="debug"
            ;;
        --release)
            RELEASE_FLAG="--release"
            BUILD_PROFILE="release"
            ;;
        --open)
            OPEN_AFTER_BUILD=true
            ;;
        --zip)
            ZIP_BUNDLE=true
            ;;
        --help|-h)
            echo "Usage: $0 [--release|--debug] [--open] [--zip]"
            echo ""
            echo "Options:"
            echo "  --release  Build release profile (default)"
            echo "  --debug    Build debug profile (faster build, slower runtime)"
            echo "  --open     Launch Manifold.app after building"
            echo "  --zip      Package Manifold.app into target/Manifold-macos-app.zip"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg" >&2
            echo "Run '$0 --help' for usage." >&2
            exit 1
            ;;
    esac
done

cd "$WORKSPACE_ROOT"

echo "==> Building manifold-gui ($BUILD_PROFILE profile)..."
if [ -n "$RELEASE_FLAG" ]; then
    cargo build --release -p manifold-gui
else
    cargo build -p manifold-gui
fi

BIN_PATH="target/$BUILD_PROFILE/manifold-gui"
APP_DIR="target/$BUILD_PROFILE/Manifold.app"

if [ ! -f "$BIN_PATH" ]; then
    echo "Error: Binary not found at $BIN_PATH" >&2
    exit 1
fi

echo "==> Assembling $APP_DIR..."
rm -rf "$APP_DIR"
mkdir -p "$APP_DIR/Contents/MacOS"
mkdir -p "$APP_DIR/Contents/Resources"

cp "$BIN_PATH" "$APP_DIR/Contents/MacOS/manifold-gui"
chmod +x "$APP_DIR/Contents/MacOS/manifold-gui"

# Determine version from Cargo.toml or git tag
VERSION="$(cargo metadata --no-deps --format-version 1 2>/dev/null | grep -o '"version":"[^"]*"' | head -n 1 | cut -d'"' -f4 || echo "0.1.0")"

cat << EOF > "$APP_DIR/Contents/Info.plist"
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>manifold-gui</string>
    <key>CFBundleIdentifier</key>
    <string>io.github.manifold.gui</string>
    <key>CFBundleName</key>
    <string>Manifold</string>
    <key>CFBundleDisplayName</key>
    <string>Manifold</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>${VERSION}</string>
    <key>CFBundleVersion</key>
    <string>${VERSION}</string>
    <key>LSMinimumSystemVersion</key>
    <string>11.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSSupportsAutomaticGraphicsSwitching</key>
    <true/>
</dict>
</plist>
EOF

echo "==> Created $APP_DIR"

if [ "$ZIP_BUNDLE" = true ]; then
    ZIP_NAME="Manifold-macos-app.zip"
    echo "==> Packaging $APP_DIR into target/$BUILD_PROFILE/$ZIP_NAME..."
    (cd "target/$BUILD_PROFILE" && zip -r -y "$ZIP_NAME" "Manifold.app" >/dev/null)
    echo "==> Created target/$BUILD_PROFILE/$ZIP_NAME"
fi

if [ "$OPEN_AFTER_BUILD" = true ]; then
    echo "==> Launching $APP_DIR..."
    open "$APP_DIR"
fi
