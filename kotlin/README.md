# card-fscan — Kotlin track

Reads photos of shingled Magic: The Gathering columns (see `../spec/SPEC.md`) and writes one
JSON inventory per photo. The same pipeline runs on the desktop (CLI, used for evaluation)
and on Android (Jetpack Compose app).

## Architecture

```
kotlin/
├── core/   pure Kotlin/JVM library: the whole pipeline   (compiles against OpenCV + ONNX Runtime APIs)
├── cli/    desktop app  `scan --out <dir> <image>...`   (ships desktop OpenCV + ONNX Runtime)
├── app/    Android app (Compose)                        (ships Android OpenCV + ONNX Runtime)
└── models/ latin_PP-OCRv5_rec_mobile.onnx (+ NOTICE, licence)

photo ──► MarkerDetector ──► MarkerSelection ──► ColumnRectifier (per column) ──► warp 12 px/mm
          ArUco 4x4_50       drop stray ids,     homography from 8 header        840 x 2880 px
                             decide C4 / C6      corners; stop card (id 40)
                                                          │
          ┌───────────────────────────────────────────────┘
          ▼
   TextLineFinder ──► SlotCropper ──► PaddleTextRecognizer ──► NameIndex.match ──► ScanReport (JSON)
   one name line per   5.5 mm crop     PP-OCRv5 latin, ONNX     clean, normalise,
   slot, chosen        around it       Runtime, CTC greedy      indel ratio, accept
   jointly (DP)        (+ fallbacks)   decode, batched          rule of SPEC 3
```

`CardScanner` (core) wires these together. Key ideas:

- **Geometry is pure Kotlin** (`geometry/`): strip constants, slot boxes and a small
  homography solver, unit-tested without native code.
- **Joint text-line finding** (`vision/TextLineFinder.kt`): within the spec's widened slot
  band (-1..11 mm below the nominal card top) each pixel row gets a "text energy" (sum of
  horizontal gradients). A dynamic programme picks one line per slot, in order down the
  column, at least 4 mm apart, maximising total energy. This absorbs placement jitter and
  stops a slot from re-reading the previous card's name when that name sits low.
- **Staged OCR** (`ScanOptions.stages`): every slot is read once from a colour crop centred on
  its line; slots not auto-accepted are retried with an inverted CLAHE crop, then the fixed
  spec band. The "better" reading wins (auto > review > empty, then match score).
- **Matching** (`match/`): exactly SPEC section 3, checked against all 74 vectors of
  `testdata/names/match_cases.json`. The indel ratio uses the bit-parallel LCS algorithm
  (Hyyrö 2004): about 1 ms to score one query against all 62,659 keys, so brute force is
  fine and no prefilter is needed.
- **Stricter than the spec, never looser**: an all-caps OCR line is never auto-accepted.
  Some special frames show a headline such as "THE PROSPERITY POST" in the name band, and
  the headline word can be a *different* real card name.
- **Stray markers**: header markers must match the median header-marker size; ids 8..11
  only switch to C6 when they lie on the header row; a stop-card marker must project to a
  roughly 30 mm square. Card art occasionally decodes as a marker.

## Build, test, run

Requirements: JDK 17+ (JDK 21 used), Android SDK with platform 36 and NDK 29 for the app.
Point Gradle to the SDK with `kotlin/local.properties` (not committed):

```
sdk.dir=C:/Users/<you>/AppData/Local/Android/Sdk
```

All commands run from `kotlin/` (on Windows use `gradlew.bat` or Git Bash):

```sh
./gradlew test                      # unit tests (core): geometry, matching vectors, OCR, JSON
./gradlew :cli:installDist          # builds cli/build/install/scan/{bin,lib,models}
./gradlew :app:assembleDebug        # app/build/outputs/apk/debug/app-debug.apk (arm64)
./gradlew :app:assembleDebug -Pabis=arm64-v8a,x86_64   # also runs on the x86_64 emulator
```

The CLI (SPEC section 5):

```sh
# from the repo root
kotlin/cli/build/install/scan/bin/scan --out kotlin/results/smoke testdata/smoke/*.jpg
uv run --project tools tools/eval.py --manifest testdata/smoke/manifest.json --results kotlin/results/smoke

# or through Gradle (paths relative to the repo root)
./gradlew :cli:run --args="--out kotlin/results/smoke testdata/smoke/c4_0001.jpg"
```

Options: `--index <names_v1.json[.gz]>` (default: found from the repo), `--model <onnx>`,
`--threads <n>` (ONNX Runtime threads, default 2), `--debug <dir>` (dumps warped columns
and every OCR crop), `--preset fixed|located`.

The app: install the APK, wait for "Ready", then **Pick photo** (system Photo Picker) or
**Take photo** (the system camera app writes a full-resolution JPEG into the app's
cache through a `FileProvider`, so no camera permission is needed). The result lists every
column and slot. The model and `names_v1.json` are copied into the APK assets at build time
by the `copyScannerAssets` task in `app/build.gradle.kts`. Card images are never bundled.

## Evaluation results

RESULTS_PLACEHOLDER

## Kotlin for newcomers

A tour of the language and build features this code uses, with a file where each one
appears.

**Language**

| Feature | What it does | Where |
| --- | --- | --- |
| `val` / `var` | read-only vs. mutable variable. Prefer `val`. | everywhere |
| `data class` | a class with `equals`, `hashCode`, `toString` and `copy()` generated from its properties | `ScanReport.kt`, `match/NameIndex.kt` (`Candidate`, `MatchResult`) |
| `copy(...)` | a new object with some properties changed (immutable updates) | `CardScanner.guardHeadline`, `app/ScannerViewModel.kt` |
| `object` | a singleton: one instance, created on first use | `geometry/StripGeometry.kt`, `match/TextNormalizer.kt` |
| `companion object` | "static" members of a class (`Homography.fit(...)`) | `geometry/Homography.kt`, `match/NameIndex.kt` |
| `enum class` | a fixed set of values | `geometry/StripGeometry.kt` (`Config`), `vision/SlotCropper.kt` (`Enhance`) |
| `interface` | a contract several classes can implement | `ocr/TextRecognizer.kt` |
| Null safety: `T?`, `?.`, `?:`, `!!` | `String?` may be null; `a?.b` is null-safe access; `a ?: b` is "a, or b if a is null"; `!!` asserts non-null | `CardScanner.kt` (`slotReport`), `cli/Main.kt` |
| `when` | a richer `switch` that is also an expression | `match/TextNormalizer.kt`, `ocr/OnnxMetadata.kt` |
| Default and named arguments | `slotBox(1, widen = 1.5)`; no overloads needed | `geometry/StripGeometry.kt`, `vision/SlotCropper.kt` (`CropVariant`) |
| Lambdas and collection functions | `map`, `filter`, `associate`, `groupBy`, `maxBy`, `flatMap`, `chunked` work on any list | `CardScanner.kt`, `vision/MarkerSelection.kt` |
| `it` | the implicit name of a single lambda parameter | everywhere |
| Destructuring | `val (key, v) = pair` unpacks a `Pair` or data class | `CardScanner.readSlots` |
| Extension functions | add a function to an existing type without subclassing | `MatchCasesTest.kt` (`JsonObject.str`) |
| `use { }` | closes a resource when the block ends, even on errors (try-with-resources) | `ocr/PaddleTextRecognizer.kt`, `cli/Main.kt` |
| `by lazy` | compute a value on first use, then cache it | `test/.../TestPaths.kt` |
| `buildString`, `buildList` | build a String or List with a mutable builder, return the immutable result | `match/TextNormalizer.kt`, `cli/Main.kt` |
| `require` / `check` / `error` | throw `IllegalArgumentException` / `IllegalStateException` with a message | `geometry/StripGeometry.kt`, `ocr/PaddleTextRecognizer.kt` |
| `Nothing` | the type of an expression that never returns (throws or exits) | `cli/Main.kt` (`usage`) |
| `operator fun times` | lets you write `a * b` for your own types | `geometry/Homography.kt` |
| Annotations `@Serializable`, `@SerialName` | the kotlinx.serialization compiler plugin generates the JSON code | `ScanReport.kt`, `match/NameIndex.kt` |
| `runCatching` / `fold` | wraps a call into a `Result` of success or exception | `app/ScannerViewModel.kt` |
| String templates | `"$x"` and `"${a.b}"` inside strings | everywhere |
| Gotcha: `x += f()` | reads `x` before calling `f()`; if `f` changes `x`, the update is lost | `ocr/OnnxMetadata.kt` (`skip`) |
| Gotcha: signed bytes | `Byte` is -128..127; use `b.toInt() and 0xFF` for pixel values | `ocr/PaddleTextRecognizer.kt` |

**Coroutines and Compose (app)**

| Concept | Where |
| --- | --- |
| `ViewModel` survives screen rotation, so heavy state (model, name index) loads once | `app/ScannerViewModel.kt` |
| `viewModelScope.launch(Dispatchers.Default) { }` runs work on a background thread pool and is cancelled with the ViewModel | `app/ScannerViewModel.kt` |
| `StateFlow` holds the current UI state; the UI collects it with `collectAsStateWithLifecycle()` | `ScannerViewModel.kt`, `MainActivity.kt` |
| `@Composable` functions describe UI from state; Compose re-runs them when state changes | `app/MainActivity.kt` |
| `remember { mutableStateOf(...) }` keeps a value across recompositions | `MainActivity.kt` (`pendingPhoto`) |
| `rememberLauncherForActivityResult` + `PickVisualMedia` / `TakePicture` replace `startActivityForResult` | `MainActivity.kt` |
| `FileProvider` gives another app (the camera) a `content://` uri to write into | `AndroidManifest.xml`, `res/xml/file_paths.xml` |

**Gradle and Android build**

| Concept | Where |
| --- | --- |
| Gradle wrapper: `gradlew` downloads the pinned Gradle version, so nobody installs Gradle | `gradlew`, `gradle/wrapper/gradle-wrapper.properties` |
| Multi-project build: `include(":core", ":cli", ":app")` | `settings.gradle.kts` |
| Kotlin DSL build scripts (`*.gradle.kts`) | every `build.gradle.kts` |
| Version catalog: every version in one file, used as `libs.xxx` | `gradle/libs.versions.toml` |
| `implementation` vs `api` vs `compileOnly` vs `testImplementation`: who sees a dependency and when | `core/build.gradle.kts` (core compiles against OpenCV/ONNX Runtime but each app ships its own runtime) |
| `application` plugin: `run` and `installDist` tasks and start scripts | `cli/build.gradle.kts` |
| A typed custom task with `@InputFiles` / `@OutputDirectory`, registered as generated assets | `app/build.gradle.kts` (`CopyScannerAssets`) |
| `compileSdk` / `minSdk` / `targetSdk`; ABI filters; NDK only for stripping native libraries | `app/build.gradle.kts` |
| Build properties: `-Pabis=...` read with `project.findProperty` | `app/build.gradle.kts` |
| `gradle.properties`: JVM heap, worker count | `gradle.properties` |

## Licences

Code: MIT (repo root `LICENSE`). Model and libraries: see `THIRD_PARTY.md` and
`models/NOTICE`.
