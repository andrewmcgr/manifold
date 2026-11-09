Param(
    [ValidateSet("release", "debug")]
    [string]$Profile = "release",
    [switch]$SkipBuild,
    [string]$BinDir = "",
    [string]$StageDir = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

if ([string]::IsNullOrEmpty($StageDir)) {
    $StageDir = Join-Path $RepoRoot "target\$Profile\package\manifold-windows-x86_64"
}

if (-not $SkipBuild) {
    Write-Host "==> Building manifold-frontend ($Profile)..."
    if ($Profile -eq "release") {
        cargo build --manifest-path "$RepoRoot\frontend\Cargo.toml" --release
    } else {
        cargo build --manifest-path "$RepoRoot\frontend\Cargo.toml"
    }
    if ($LASTEXITCODE -ne 0) {
        throw "cargo build failed with exit code $LASTEXITCODE"
    }
}

$BinCandidates = @()
if (-not [string]::IsNullOrEmpty($BinDir)) {
    $BinCandidates += (Join-Path $RepoRoot "$BinDir\manifold-frontend.exe")
    $BinCandidates += (Join-Path $BinDir "manifold-frontend.exe")
}
$BinCandidates += (Join-Path $RepoRoot "target\$Profile\manifold-frontend.exe")
$BinCandidates += (Join-Path $RepoRoot "target\x86_64-pc-windows-gnu\$Profile\manifold-frontend.exe")
$BinCandidates += (Join-Path $RepoRoot "target\x86_64-pc-windows-msvc\$Profile\manifold-frontend.exe")

$BinPath = $null
foreach ($Cand in $BinCandidates) {
    if (Test-Path $Cand) {
        $BinPath = $Cand
        break
    }
}
if (-not $BinPath) {
    throw "Binary not found in any candidate path: $($BinCandidates -join ', ')"
}

Write-Host "==> Staging Windows Scoop / portable package in $StageDir..."
if (Test-Path $StageDir) {
    Remove-Item -Recurse -Force $StageDir
}
New-Item -ItemType Directory -Force -Path $StageDir | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $StageDir "Test Objects") | Out-Null

Copy-Item $BinPath (Join-Path $StageDir "manifold-frontend.exe")
Copy-Item $BinPath (Join-Path $StageDir "manifold.exe")
Copy-Item (Join-Path $RepoRoot "frontend\assets\app_icon.png") (Join-Path $StageDir "manifold.png")
Copy-Item (Join-Path $RepoRoot "frontend\assets\app_icon_1024.png") (Join-Path $StageDir "manifold_1024.png")

if (Test-Path (Join-Path $RepoRoot "Test Objects")) {
    Copy-Item (Join-Path $RepoRoot "Test Objects\*") (Join-Path $StageDir "Test Objects\") -Recurse -Force
}

# Bundle required runtime DLLs from MSYS2 / vcpkg / Conda prefix if present so the Scoop zip is self-contained
$CandidateDllDirs = @()
foreach ($EnvVar in @("MANIFOLD_DEPS_PREFIX", "MINGW_PREFIX", "CONDA_PREFIX")) {
    $Val = [Environment]::GetEnvironmentVariable($EnvVar)
    if (-not [string]::IsNullOrEmpty($Val)) {
        $BinCandidate = Join-Path $Val "bin"
        if (Test-Path $BinCandidate) {
            $CandidateDllDirs += $BinCandidate
        }
    }
}
if (-not [string]::IsNullOrEmpty($env:VCPKG_ROOT)) {
    $Triplet = if ($env:VCPKG_DEFAULT_TRIPLET) { $env:VCPKG_DEFAULT_TRIPLET } else { "x64-windows" }
    $VcpkgBin = Join-Path $env:VCPKG_ROOT "installed\$Triplet\bin"
    if (Test-Path $VcpkgBin) {
        $CandidateDllDirs += $VcpkgBin
    }
}
foreach ($FallbackBin in @("D:\a\_temp\msys64\ucrt64\bin", "C:\msys64\ucrt64\bin", "C:\msys64\mingw64\bin", "C:\vcpkg\installed\x64-windows\bin")) {
    if ((Test-Path $FallbackBin) -and ($CandidateDllDirs -notcontains $FallbackBin)) {
        $CandidateDllDirs += $FallbackBin
    }
}

# Use objdump recursively when available to capture all transitive non-system DLL dependencies,
# then also scan fallback glob patterns.
$CopiedDlls = @{}
$Objdump = Get-Command "objdump.exe" -ErrorAction SilentlyContinue
if ($Objdump -and $CandidateDllDirs.Count -gt 0) {
    $Queue = [System.Collections.Generic.Queue[string]]::new()
    $Queue.Enqueue((Join-Path $StageDir "manifold-frontend.exe"))
    while ($Queue.Count -gt 0) {
        $CurrentBin = $Queue.Dequeue()
        $Lines = & $Objdump.Source -p $CurrentBin 2>$null | Select-String "DLL Name:\s+(\S+)"
        foreach ($Match in $Lines) {
            $DllName = $Match.Matches[0].Groups[1].Value
            if (-not $CopiedDlls.ContainsKey($DllName)) {
                foreach ($DllDir in $CandidateDllDirs) {
                    $CandidatePath = Join-Path $DllDir $DllName
                    if (Test-Path $CandidatePath) {
                        $DestPath = Join-Path $StageDir $DllName
                        Copy-Item $CandidatePath $DestPath -Force
                        $CopiedDlls[$DllName] = $true
                        $Queue.Enqueue($DestPath)
                        break
                    }
                }
            }
        }
    }
}

foreach ($DllDir in $CandidateDllDirs) {
    $Patterns = @(
        "libTK*.dll", "TK*.dll",
        "libopenvdb*.dll", "openvdb*.dll",
        "libtbb*.dll", "tbb*.dll",
        "libImath*.dll", "Imath*.dll",
        "libHalf*.dll", "Half*.dll",
        "libboost_iostreams*.dll", "boost_iostreams*.dll",
        "libboost_system*.dll", "boost_system*.dll",
        "libblosc*.dll", "blosc*.dll",
        "zlib1.dll", "libzstd.dll", "liblz4.dll", "libbz2*.dll", "liblzma*.dll",
        "libgcc_s_seh-1.dll", "libstdc++-6.dll", "libwinpthread-1.dll",
        "libjemalloc*.dll", "libfmt*.dll", "libspdlog*.dll"
    )
    foreach ($Pat in $Patterns) {
        Get-ChildItem -Path $DllDir -Filter $Pat -ErrorAction SilentlyContinue | ForEach-Object {
            if (-not $CopiedDlls.ContainsKey($_.Name)) {
                Copy-Item $_.FullName (Join-Path $StageDir $_.Name) -Force
                $CopiedDlls[$_.Name] = $true
            }
        }
    }
    break
}

Write-Host "==> Bundled $($CopiedDlls.Count) runtime DLLs into $StageDir"

$ZipPath = "$StageDir.zip"
if (Test-Path $ZipPath) {
    Remove-Item -Force $ZipPath
}
Compress-Archive -Path "$StageDir\*" -DestinationPath $ZipPath -Force
Write-Host "==> Successfully packaged Windows archive: $ZipPath"
