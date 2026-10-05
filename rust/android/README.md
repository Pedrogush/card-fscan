# fscan-android — the Rust Android app

`fscan-android` is the Android app of the Rust track: a single screen
(title, **Run sample scan** button, text area) that runs the `fscan-core`
pipeline on a photo and prints the result. It is written entirely in Rust —
there is no Java/Kotlin and no Gradle.

Package `io.github.pedrogush.cardfscan.rs`, label "card-fscan (rs)", so it
installs side by side with the Kotlin app. minSdk 26, targetSdk 36, arm64-v8a
(x86_64 optional, for emulators).

## Build the APK

From the repo root, in PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File rust\android\build_apk.ps1
# bundle a photo as assets/sample.jpg:
powershell -ExecutionPolicy Bypass -File rust\android\build_apk.ps1 -Sample C:\path\to\photo.jpg
# also build for the x86_64 emulator and install on the connected device:
powershell -ExecutionPolicy Bypass -File rust\android\build_apk.ps1 -Abis arm64-v8a,x86_64 -Install
```

Output: `rust/android/build/card-fscan-rs-debug.apk` (debug-signed). Install
with `adb install -r rust\android\build\card-fscan-rs-debug.apk`.

Requirements: Rust with the `aarch64-linux-android` target
(`rustup target add aarch64-linux-android`), `cargo install cargo-ndk`,
Android SDK with an NDK, build-tools and `platforms/android-36`, and a JDK
(for `jar`, `keytool`, `apksigner`). The script finds them through
`ANDROID_HOME`, `ANDROID_NDK_HOME`, `JAVA_HOME` or the default locations.
The Rust code is compiled in **release** mode because the OCR is far too slow
unoptimised; "debug APK" only means it is signed with the debug key
(`%USERPROFILE%\.android\debug.keystore`, created if missing — never in the repo).

## How it works

1. **NativeActivity.** Android ships a ready-made activity class,
   `android.app.NativeActivity`, whose only job is to load a native library.
   `AndroidManifest.xml` declares that activity, `hasCode="false"` (no
   `classes.dex`) and `android.app.lib_name = fscan_android`, so Android loads
   `lib/arm64-v8a/libfscan_android.so` from the APK.
2. **android-activity / winit / eframe.** The `android-activity` crate
   (pulled in by winit's `android-native-activity` feature) implements the C
   callbacks NativeActivity expects, starts a thread and calls our
   `android_main` (in `src/lib.rs`). That hands the `AndroidApp` to `eframe`,
   which opens an OpenGL ES surface (`glow`) and draws the egui UI
   (`src/app.rs`).
3. **The scan.** Pressing the button spawns a background thread that reads
   `names_v1.json.gz`, `rec.onnx`, `rec.dict.txt` (and `sample.jpg` if
   bundled) through the APK `AssetManager`, then runs `src/scan.rs`, which only
   uses `fscan-core` and is plain portable Rust. Errors and even panics end up
   as text in the text area.
4. **cargo-ndk** sets up the NDK's clang/linker for the Android target and
   runs `cargo build`; `-P 26` targets API level 26. The script then copies
   `libfscan_android.so` into `build/apk-lib/lib/<abi>/`, stripping debug
   info with the NDK's `llvm-strip` (the unstripped file stays in
   `rust/target/android-build/` for symbolicating crashes). The first build
   takes ~30 min on 2 cores (rten in release); later ones are incremental.
5. **aapt2 link** turns the manifest into binary XML and builds a bare APK
   (a zip) containing it plus everything in `assets/`.
6. **jar uf0M** adds `lib/<abi>/libfscan_android.so` *uncompressed*, so
   Android can map it straight from the APK (`extractNativeLibs="false"`).
7. **zipalign -P 16 -f 4** aligns the `.so` to 16 KiB pages (required by
   newer Android devices) and everything else to 4 bytes.
8. **apksigner** signs it with the debug key; the script then re-checks it
   with `zipalign -c` and `apksigner verify`.

`assets/` and `build/` are generated (gitignored). Assets come from
`testdata/names/names_v1.json` (gzipped by the script) and
`rust/models/en_PP-OCRv4_rec_mobile.{onnx,dict.txt}`.

## Getting a photo onto the phone

Without a bundled `sample.jpg`, the app uses the first `.jpg` in its
external files directory:

```sh
adb shell mkdir -p /sdcard/Android/data/io.github.pedrogush.cardfscan.rs/files
adb push c4_0001.jpg /sdcard/Android/data/io.github.pedrogush.cardfscan.rs/files/
```

(Install and launch the app once first so Android creates the directory.)
Logs: `adb logcat -s fscan`.

## Known quirk: two pinned build-time crates

The Rust host here is `x86_64-pc-windows-gnu` without MinGW. Build scripts
run on the host, and two of them on the Android path would pull crates that
use `raw-dylib` on Windows (needs `dlltool` + an assembler, error
"error calling dlltool"):

- `jni` -> `walkdir` -> `winapi-util` >= 0.1.10 -> `windows-sys 0.61`
- `android-activity` -> `cc` -> `jobserver` >= 0.1.33 -> `getrandom 0.3/0.4`

`Cargo.lock` pins `winapi-util 0.1.9` and `jobserver 0.1.32`. If a
`cargo update` brings the error back:

```sh
cargo update -p winapi-util --precise 0.1.9
cargo update -p jobserver --precise 0.1.32
```

## On the host

All Android-only code and dependencies are behind
`cfg(target_os = "android")`, so `cargo check -p fscan-android` on Windows
builds just the portable `scan` module. The crate is in the workspace
`members` but not `default-members`, so plain `cargo build` skips it.
