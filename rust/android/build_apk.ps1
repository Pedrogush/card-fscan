<#
.SYNOPSIS
  Builds the debug-signed card-fscan (rs) APK without Gradle.

.DESCRIPTION
  1. Stages assets into rust/android/assets/ (name index, OCR model, optional sample photo).
  2. Cross-compiles the Rust crate to libfscan_android.so with cargo-ndk (release mode).
  3. Packs manifest + assets with aapt2, adds the .so uncompressed, zipaligns
     (16 KiB page alignment for .so files) and signs with the debug keystore.

  Output: rust/android/build/card-fscan-rs-debug.apk

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File rust\android\build_apk.ps1
  powershell -ExecutionPolicy Bypass -File rust\android\build_apk.ps1 -Sample C:\photos\c4_0001.jpg
  powershell -ExecutionPolicy Bypass -File rust\android\build_apk.ps1 -Abis arm64-v8a,x86_64 -Install
#>
param(
    # Optional photo bundled as assets/sample.jpg (also read from $env:FSCAN_SAMPLE).
    [string]$Sample = $env:FSCAN_SAMPLE,
    # Android ABIs to build. arm64-v8a = real phones; x86_64 = emulator.
    [string[]]$Abis = @('arm64-v8a'),
    # Install on the connected device with adb after building.
    [switch]$Install
)

$ErrorActionPreference = 'Stop'

function Step($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
# Runs a native program and stops the script if it fails.
function Run {
    # No named parameters on purpose: flags like -P must reach the program untouched.
    $exe = $args[0]
    $argList = @($args | Select-Object -Skip 1)
    & $exe @argList
    if ($LASTEXITCODE -ne 0) { throw "command failed ($LASTEXITCODE): $exe $argList" }
}
function Latest($dir) {
    Get-ChildItem $dir -Directory | Sort-Object { [version]($_.Name -replace '[^0-9.].*$', '') } | Select-Object -Last 1
}

# ---------------------------------------------------------------- paths/tools
$AndroidDir = $PSScriptRoot                         # rust/android
$RustDir    = Split-Path $AndroidDir -Parent        # rust
$RepoDir    = Split-Path $RustDir -Parent           # repo root
$AssetsDir  = Join-Path $AndroidDir 'assets'
$BuildDir   = Join-Path $AndroidDir 'build'
$OutApk     = Join-Path $BuildDir 'card-fscan-rs-debug.apk'

if (-not $env:ANDROID_HOME) { $env:ANDROID_HOME = Join-Path $env:LOCALAPPDATA 'Android\Sdk' }
$Sdk = $env:ANDROID_HOME
if (-not $env:ANDROID_NDK_HOME) { $env:ANDROID_NDK_HOME = (Latest (Join-Path $Sdk 'ndk')).FullName }
$BuildTools = (Latest (Join-Path $Sdk 'build-tools')).FullName
$AndroidJar = Join-Path $Sdk 'platforms\android-36\android.jar'
if (-not $env:JAVA_HOME) {
    $jdk = Get-ChildItem (Join-Path $env:USERPROFILE '.jdks') -Directory -ErrorAction SilentlyContinue | Select-Object -Last 1
    if ($jdk) { $env:JAVA_HOME = $jdk.FullName }
}
if (-not $env:JAVA_HOME) { throw 'JAVA_HOME is not set and no JDK found in ~/.jdks' }
$env:PATH = "$env:USERPROFILE\.cargo\bin;$env:JAVA_HOME\bin;$env:PATH"
# NOTE (windows-gnu host without MinGW): jni's build script depends on
# walkdir -> winapi-util, and cc's jobserver; newer versions of those use
# raw-dylib, which needs dlltool + an assembler on the host. Cargo.lock pins
# winapi-util 0.1.9 and jobserver 0.1.32 (see README "Known quirk").
# Separate target dir so this does not block other cargo builds of the workspace.
if (-not $env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR = Join-Path $RustDir 'target\android-build' }

$Aapt2     = Join-Path $BuildTools 'aapt2.exe'
$Zipalign  = Join-Path $BuildTools 'zipalign.exe'
$Apksigner = Join-Path $BuildTools 'apksigner.bat'
$Jar       = Join-Path $env:JAVA_HOME 'bin\jar.exe'
$Keytool   = Join-Path $env:JAVA_HOME 'bin\keytool.exe'
foreach ($t in @($Aapt2, $Zipalign, $Apksigner, $Jar, $Keytool, $AndroidJar)) {
    if (-not (Test-Path $t)) { throw "missing tool: $t" }
}
Write-Host "SDK: $Sdk`nNDK: $env:ANDROID_NDK_HOME`nbuild-tools: $BuildTools`nJDK: $env:JAVA_HOME"

# ---------------------------------------------------------------- 1. assets
Step 'Staging assets'
if (Test-Path $AssetsDir) { Remove-Item -Recurse -Force $AssetsDir }
New-Item -ItemType Directory -Force $AssetsDir | Out-Null

# names_v1.json -> names_v1.json.gz (gzip with .NET, no external tool needed)
$namesSrc = Join-Path $RepoDir 'testdata\names\names_v1.json'
$namesDst = Join-Path $AssetsDir 'names_v1.json.gz'
$in = [IO.File]::OpenRead($namesSrc)
$out = [IO.File]::Create($namesDst)
$gz = New-Object IO.Compression.GZipStream($out, [IO.Compression.CompressionMode]::Compress)
try { $in.CopyTo($gz) } finally { $gz.Dispose(); $out.Dispose(); $in.Dispose() }

# OCR model + dictionary
$modelDirs = @(
    (Join-Path $RustDir 'models'),
    'C:\Users\Pedro\AppData\Local\Temp\claude\C--Claude-My-projects-MTG-fscan\f23b9e8a-535b-4b8d-93ba-95f8c5eca9da\scratchpad\models'
)
$modelDir = $modelDirs | Where-Object { Test-Path (Join-Path $_ 'en_PP-OCRv4_rec_mobile.onnx') } | Select-Object -First 1
if (-not $modelDir) { throw "OCR model en_PP-OCRv4_rec_mobile.onnx not found in: $($modelDirs -join ', ')" }
Copy-Item (Join-Path $modelDir 'en_PP-OCRv4_rec_mobile.onnx') (Join-Path $AssetsDir 'rec.onnx')
Copy-Item (Join-Path $modelDir 'en_PP-OCRv4_rec_mobile.dict.txt') (Join-Path $AssetsDir 'rec.dict.txt')

if ($Sample) {
    if (-not (Test-Path $Sample)) { throw "sample photo not found: $Sample" }
    Copy-Item $Sample (Join-Path $AssetsDir 'sample.jpg')
    Write-Host "bundled sample photo: $Sample"
}
Get-ChildItem $AssetsDir | ForEach-Object { '{0,-20} {1,12:N0} bytes' -f $_.Name, $_.Length }

# ---------------------------------------------------------------- 2. Rust -> .so
Step "Compiling Rust for $($Abis -join ', ') (cargo-ndk, release)"
$LibStage = Join-Path $BuildDir 'apk-lib'      # becomes the APK's lib/ folder
if (Test-Path $LibStage) { Remove-Item -Recurse -Force $LibStage }
New-Item -ItemType Directory -Force $BuildDir | Out-Null
$ndkArgs = @()
foreach ($abi in $Abis) { $ndkArgs += @('-t', $abi) }
Push-Location $RustDir
try {
    # -P 26: link against Android API level 26 (= minSdk in AndroidManifest.xml)
    Run cargo ndk @ndkArgs -P 26 build --release -p fscan-android
} finally { Pop-Location }

# Copy only our library (rten also emits an unneeded librten-*.so) into
# build/apk-lib/lib/<abi>/ and strip it: the release profile keeps line-table
# debug info (~90 MB); the unstripped copy stays in target/ for crash symbolication.
$Triples = @{ 'arm64-v8a' = 'aarch64-linux-android'; 'x86_64' = 'x86_64-linux-android' }
$Strip = Join-Path $env:ANDROID_NDK_HOME 'toolchains\llvm\prebuilt\windows-x86_64\bin\llvm-strip.exe'
foreach ($abi in $Abis) {
    if (-not $Triples.ContainsKey($abi)) { throw "unsupported ABI: $abi" }
    $src = Join-Path $env:CARGO_TARGET_DIR "$($Triples[$abi])\release\libfscan_android.so"
    $dstDir = Join-Path $LibStage "lib\$abi"
    New-Item -ItemType Directory -Force $dstDir | Out-Null
    Run $Strip --strip-unneeded -o (Join-Path $dstDir 'libfscan_android.so') $src
    '{0,-12} libfscan_android.so {1,12:N0} bytes (stripped)' -f $abi, (Get-Item (Join-Path $dstDir 'libfscan_android.so')).Length
}

# ---------------------------------------------------------------- 3. APK
Step 'Packaging APK (aapt2 link)'
$Unaligned = Join-Path $BuildDir 'unaligned.apk'
$Aligned   = Join-Path $BuildDir 'aligned.apk'
Remove-Item -Force $Unaligned, $Aligned, $OutApk -ErrorAction SilentlyContinue
# aapt2 compiles AndroidManifest.xml to binary XML, writes resources.arsc and
# copies assets/ into the zip. android.jar provides the framework definitions.
Run $Aapt2 link -o $Unaligned --manifest (Join-Path $AndroidDir 'AndroidManifest.xml') -I $AndroidJar -A $AssetsDir

# Add lib/<abi>/libfscan_android.so *uncompressed* (jar flag 0 = store, M = no
# META-INF/MANIFEST.MF) so Android can mmap it straight from the APK.
Run $Jar uf0M $Unaligned -C $LibStage lib

Step 'Aligning (zipalign -P 16: 16 KiB page alignment for .so)'
Run $Zipalign -P 16 -f 4 $Unaligned $Aligned

Step 'Signing with the debug keystore'
$Keystore = Join-Path $env:USERPROFILE '.android\debug.keystore'
if (-not (Test-Path $Keystore)) {
    Write-Host "creating $Keystore"
    New-Item -ItemType Directory -Force (Split-Path $Keystore) | Out-Null
    Run $Keytool -genkeypair -v -keystore $Keystore -storepass android -alias androiddebugkey `
        -keypass android -keyalg RSA -keysize 2048 -validity 10000 -dname 'CN=Android Debug,O=Android,C=US'
}
Run $Apksigner sign --ks $Keystore --ks-pass pass:android --ks-key-alias androiddebugkey `
    --key-pass pass:android --out $OutApk $Aligned
Remove-Item -Force $Unaligned, $Aligned, "$OutApk.idsig" -ErrorAction SilentlyContinue

Step 'Verifying'
Run $Zipalign -c -P 16 4 $OutApk
Run $Apksigner verify $OutApk
$size = (Get-Item $OutApk).Length / 1MB
Write-Host ("OK: {0} ({1:N1} MB)" -f $OutApk, $size) -ForegroundColor Green

if ($Install) {
    Step 'Installing with adb'
    Run (Join-Path $Sdk 'platform-tools\adb.exe') install -r $OutApk
}
