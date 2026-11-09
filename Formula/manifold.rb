class Manifold < Formula
  desc "Multi-axis, non-planar additive manufacturing slicer powered by OpenVDB and OpenCASCADE"
  homepage "https://github.com/andrewmcgr/manifold"
  url "https://github.com/andrewmcgr/manifold/archive/refs/tags/v0.1.0.tar.gz"
  sha256 "0000000000000000000000000000000000000000000000000000000000000000"
  license "MIT"
  head "https://github.com/andrewmcgr/manifold.git", branch: "main"

  depends_on "cmake" => :build
  depends_on "pkg-config" => :build
  depends_on "rust" => :build
  depends_on "imath"
  depends_on "opencascade"
  depends_on "openvdb"
  depends_on "tbb"

  on_linux do
    depends_on "alsa-lib"
    depends_on "fontconfig"
    depends_on "gtk+3"
    depends_on "libx11"
    depends_on "libxcursor"
    depends_on "libxi"
    depends_on "libxkbcommon"
    depends_on "libxrandr"
    depends_on "systemd"
    depends_on "vulkan-loader"
    depends_on "wayland"
  end

  def install
    ENV["HOMEBREW_PREFIX"] = HOMEBREW_PREFIX.to_s
    system "cargo", "install", *std_cargo_args(path: "frontend")
    bin.install_symlink "manifold-frontend" => "manifold"

    pkgshare.install "Test Objects/Manifold_M_Cube.3mf"
    pkgshare.install "Test Objects/Manifold_M_Cube_FullSlope.3mf"

    if OS.mac?
      system "./scripts/bundle_macos.sh", "--release", "--skip-build", "--out-dir", prefix.to_s
      # Point the app bundle binary to the installed bin/manifold-frontend to avoid duplicating binaries
      rm "#{prefix}/Manifold.app/Contents/MacOS/manifold-frontend"
      ln_s "#{bin}/manifold-frontend", "#{prefix}/Manifold.app/Contents/MacOS/manifold-frontend"
    else
      (share/"applications").install "packaging/linux/manifold.desktop"
      (share/"icons/hicolor/256x256/apps").install "frontend/assets/app_icon.png" => "manifold.png"
      (share/"icons/hicolor/1024x1024/apps").install "frontend/assets/app_icon_1024.png" => "manifold.png"
    end
  end

  def caveats
    on_macos do
      <<~EOS
        Manifold.app has been installed to:
          #{opt_prefix}/Manifold.app

        To link it into /Applications, run:
          ln -sf "#{opt_prefix}/Manifold.app" /Applications/Manifold.app
      EOS
    end
  end

  test do
    ENV["MANIFOLD_DISABLE_GPU"] = "1"
    assert_match "Usage:", shell_output("#{bin}/manifold-frontend --help")
    system "#{bin}/manifold-frontend", "slice",
           "#{pkgshare}/Manifold_M_Cube.3mf",
           "-o", "#{testpath}/out.gcode"
    assert_predicate testpath/"out.gcode", :exist?
  end
end
