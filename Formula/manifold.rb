class Manifold < Formula
  desc "Multi-axis, non-planar additive manufacturing slicer powered by OpenVDB and OpenCASCADE"
  homepage "https://github.com/andrewmcgr/manifold"
  version "0.15.1"
  license "MIT"

  on_macos do
    on_arm do
      url "https://github.com/andrewmcgr/manifold/releases/download/v0.15.1/Manifold-macos-arm64.zip", using: :nounzip
      # Homebrew verifies this checksum, which matches the release's SHA256SUMS.txt
      sha256 "421e11a26d530c01781227539a8c0162905a7655543eb88bd9bb198557f094c3"
    end
  end

  on_linux do
    on_intel do
      url "https://github.com/andrewmcgr/manifold/releases/download/v0.15.1/Manifold-linux-x86_64.tar.gz"
      sha256 "73da3e8a8a1d9b6b3940d22d9e6340af113cecfda051d4a8fc195c9cbdc7e010"
    end
  end

  depends_on arch: :arm64 if OS.mac?
  depends_on arch: :x86_64 if OS.linux?

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
    if OS.mac?
      # Homebrew would auto-enter the zip's single top-level Manifold.app dir, so extract manually
      system "/usr/bin/ditto", "-x", "-k", cached_download, buildpath
      system "/usr/bin/codesign", "--verify", "--deep", "--strict", "Manifold.app"
      prefix.install "Manifold.app"
      bin.install_symlink prefix/"Manifold.app/Contents/MacOS/manifold-frontend"
      bin.install_symlink prefix/"Manifold.app/Contents/MacOS/manifold-frontend" => "manifold"
      pkgshare.install_symlink prefix/"Manifold.app/Contents/Resources/Test Objects" => "test-objects"
    else
      prefix.install Dir["manifold-linux/*"]
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
    assert_match "USAGE:", shell_output("#{bin}/manifold-frontend --help")
    system "#{bin}/manifold-frontend", "slice",
           "#{pkgshare}/test-objects/Manifold_M_Cube.3mf",
           "-o", "#{testpath}/out.gcode"
    assert_predicate testpath/"out.gcode", :exist?
  end
end
