# card-fscan — Rust track

A pure-Rust implementation of the card-fscan pipeline (`spec/SPEC.md`): find
the ArUco header markers, rectify each column, crop each card's name bar,
read it with a small OCR model and match it against the MTGJSON name index.
The same library runs in a desktop CLI (for evaluation) and in an Android app.

```
                         rust/  (Cargo workspace)
 ┌──────────────────────────────────────────────────────────────────────────┐
 │ core/  crate fscan-core (library, no platform code)                       │
 │                                                                           │
 │  photo (GrayImage)                                                        │
 │    │  aruco.rs      DICT_4X4_50 detector: threshold -> contours -> quads  │
 │    │                -> 6x6 cell decode -> sub-pixel edge-fit corners      │
 │    ▼                                                                      │
 │  pipeline.rs::locate   config C4/C6, column pairs, stop card  ──┐         │
 │    │  homography.rs  8-point normalised DLT (nalgebra)          │         │
 │    │  geometry.rs    SPEC §1 constants, slot boxes               │ Layout  │
 │    ▼                                                             ◄─┘       │
 │  pipeline.rs::crops    per slot: warp band at 12 px/mm (image_ops.rs),    │
 │    │                   find the name line (+ drift tracking down the      │
 │    │                   column), cut off the mana cost                     │
 │    ▼                                                                      │
 │  ocr.rs   trait Recognizer; RtenRecognizer = PP-OCRv5 rec ONNX on `rten`  │
 │    │      + CTC greedy decode         (pass 2: fallback crops for misses) │
 │    ▼                                                                      │
 │  matching.rs  SPEC §3 clean/normalize, bit-parallel Indel ratio,          │
 │    │          per-oracle best, accept rule                                │
 │    ▼                                                                      │
 │  output.rs   PhotoResult -> SPEC §4 JSON (serde)                          │
 └──────────────────────────────────────────────────────────────────────────┘
 cli/      crate fscan-cli -> binary `fscan scan --out <dir> <image>...`
 android/  crate fscan-android -> libfscan_android.so (NativeActivity + egui)
 models/   en_PP-OCRv5_rec_mobile.onnx + .dict.txt (Apache-2.0)
```

## Build, test, run

Requirements: Rust stable (1.85+, edition 2024). The host toolchain used here
is `x86_64-pc-windows-gnu`; every dependency is pure Rust, so no C compiler is
needed for the host build. `.cargo/config.toml` limits builds to 2 jobs.

```sh
cd rust
cargo test                       # unit tests + SPEC §3 vectors (testdata/names/match_cases.json)
cargo build --release -p fscan-cli
./target/release/fscan scan --out results/smoke ../testdata/smoke/*.jpg
# options: --names <names_v1.json[.gz]> --model <rec.onnx> --dump-crops <dir>
cd .. && uv run --project tools tools/eval.py --manifest testdata/smoke/manifest.json --results rust/results/smoke
```

Always use `--release`: the OCR model runs ~30x slower unoptimised (even the
test profile optimises dependencies, see `[profile.dev.package."*"]`).

Debug helpers (`core/examples/`):

- `ocr_bench --manifest <m> --model <onnx|rten> <photos>` — OCR bake-off on
  the pipeline's own crops, scored against ground truth. Env knobs:
  `OCR_BATCH`, `OCR_HEIGHT`, `OCR_XSCALE`, `TEXT_MM`, `TRIM`, `CLAHE`.
- `column_dump <out dir> <photos>` — every column rectified to strip mm.

### Android APK

See `android/README.md`. In short (PowerShell, from the repo root):

```powershell
powershell -ExecutionPolicy Bypass -File rust\android\build_apk.ps1 [-Sample photo.jpg] [-Install]
```

It cross-compiles `fscan-android` for `aarch64-linux-android` with
`cargo ndk` (release), packages the manifest + assets (gzipped name index,
OCR model, optional sample photo) with `aapt2`, adds the `.so`
uncompressed, aligns and debug-signs it. Output:
`rust/android/build/card-fscan-rs-debug.apk` (~20 MB, ~10.6 MB of it the
stripped native library). Phase 1 app: one screen, "Run sample scan" runs the
whole pipeline on `assets/sample.jpg` or the first `.jpg` in the app's
external files dir and prints config/status/columns/slots/auto/timing.

## Design choices

- **ArUco**: no mature pure-Rust detector existed (`aruco-rs` 0.1 is a young
  WASM-oriented port; `calib-targets-aruco` decodes only rectified grids), so
  `core/src/aruco.rs` implements the OpenCV recipe: detect on a ≤1600 px
  downscale (adaptive mean threshold, two window sizes), `imageproc`
  border-following contours, a quad fit (farthest-point corners + every
  contour point near an edge), 6x6 cell sampling on the full-resolution
  photo through a homography, 2-means binarisation, ≤2 border errors, exact
  payload match in 4 rotations (OpenCV's default correction for this
  dictionary is 0 bits), then sub-pixel corners from line fits to the four
  edges. Unit test: warped synthetic markers come back with < 0.6 px corner
  error (≈0.03 px in practice).
- **Stray markers** (card art can decode as a marker): a column only counts
  when its two ids form a pair with the printed spacing (40 mm apart, 18 mm
  wide, ±20%); ids 8-11 only switch the config to C6 when they sit where a
  header marker should be; marker 40 only counts as a stop card if it
  projects to a ~30 mm square centred on the column.
- **OCR bake-off** (pipeline crops of smoke `c4_0003` + `c4_0008`, 151
  lines, `ocr_bench`, greedy CTC, same crops for all):

  | model (runtime: rten) | exact text | auto & correct | wrong |
  | --- | --- | --- | --- |
  | PP-OCRv5 en mobile rec (7.9 MB) | **90.1%** | **95.4%** | 0 |
  | PP-OCRv4 en mobile rec (7.7 MB) | 50.3% | 90.7% | 1 |
  | ocrs `text-recognition.rten` (9.7 MB) | 11.3% | 66.2% | 0 |

  (v4/ocrs were measured before the mana-cost trim, which mostly helps
  exact text.) PP-OCRv5 also keeps spaces; PP-OCRv4 often drops them.
  Runtime: [`rten`](https://github.com/robertknight/rten) loads the ONNX file
  directly and is pure Rust, so the identical code runs on Android.
  `tract` was not needed; `ort` (ONNX Runtime) needs prebuilt C++ binaries
  and was ruled out for Android parity — on this machine onnxruntime is
  roughly 2x faster per line than rten (see "Speed").
- **Crops**: the spec's slot band is 9-12 mm tall, but the name text is only
  ~3 mm of it; resized to the model's 48 px input the letters were ~13 px and
  recognition failed (~2% auto). So each band (padded 1.5 mm as the spec
  allows) is narrowed to a 6.4 mm window centred on the row band with the most
  horizontal-gradient energy, and cut on the right at the blank gap before the
  mana cost. This took the auto rate from ~2% to ~95% and halved OCR work.
- **Drift tracking**: the homography comes from two 18 mm markers at the
  top and is extrapolated 200 mm down; with tilt + lens distortion it was off
  by up to ~9 mm at slot 20 (bottom slots read the card above → wrong
  autos). Each column is now walked top-down and every slot's text search is
  centred on the drift predicted (robust line fit) from the name lines found
  above it. Slot numbering is untouched.
- **Two passes**: all primary crops are read first; only slots that did not
  auto-accept are re-read from fallbacks (inverted crop, second-best text
  line, untrimmed line) and the best match wins.
- **Stricter-only guards** (never loosen SPEC §3): a reading with < 3
  letters/digits is never auto (fragments like `X` matched the card "X"); a
  trailing mana-cost token read as letters (`Dig Through Time 6UU`) is also
  tried stripped.
- **Matching**: Indel ratio via bit-parallel LCS (Hyyrö), exact rational
  tie-breaks; ~1 ms per query over 62,659 keys. Scores are rounded like the
  reference (float32, 4 decimals) so boundary cases agree.

## Results

Scored with `tools/eval.py` (2026-10-05, code at commit `d850406` + README;
model `en_PP-OCRv5_rec_mobile`). "auto" = auto-accepted *and* correct over
scored slots (excludes `lang: other` and sideways split cards, per eval).

| set | images | slots | auto rate | wrong | review / empty | column-count errors | status errors |
| --- | --- | --- | --- | --- | --- | --- | --- |
| smoke | 8 | 593 | **96.99%** | **0** | 37 / 8 | 0 | 0 |
| dev | 60 | 5255 | **95.76%** | **0** | 403 / 50 | 0 | 0 |

Dev by category: en_modern 97.8%, multiface 96.3%, old frames 94.8%,
Portuguese 97.9%, "special" frames 82.7% (showcase / Universes Beyond cards
whose name bar shows a flavour name or a newspaper masthead — these correctly
go to review). By difficulty: easy 96.5%, normal 96.2%, hard 94.2%.

### Speed (release build, this laptop's 4-core CPU, one photo at a time)

| | per photo | of which OCR |
| --- | --- | --- |
| C4 (80 slots) | ~8 s | ~7 s (~90 ms per line) |
| C6 (120 slots) | ~11 s | ~10 s |

Measured on smoke with the machine mostly idle (one unrelated process held a
core). Marker detection is ~0.1-0.15 s, cropping ~0.25 s, matching ~0.1 s,
JPEG decode ~0.3 s. During the dev run other builds shared the CPU and the
mean was 18.5 s/photo. On the same crops onnxruntime is about 2x faster per
line than rten, so the runtime, not the pipeline, is the main lever (see
Phase 2 ideas below).

### Known failure modes / next steps

- OCR time dominates. Options: crop width is already trimmed to the name;
  next would be int8 quantisation, rten tuning for depthwise convs, an
  `ort` backend on desktop, or batching all columns.
- Reviews are mostly cards whose name bar doesn't hold the oracle name
  (special frames, flavour names, sideways splits/battles, `lang: other`)
  plus blur on hard images (`Toraga Treesneaker`).
- Drift tracking assumes the first few slots of a column are close to the
  header-only geometry; a fuller fix would refine the homography with the
  strip's printed slot ticks.

## Rust for newcomers (as used in this code)

**Crates and workspaces.** A *crate* is a compilation unit: a library
(`core`, imported as `fscan_core`) or a binary (`cli` builds `fscan`). The
top-level `Cargo.toml` is a *workspace*: one `Cargo.lock`, one `target/`,
shared dependency versions in `[workspace.dependencies]` that members inherit
with `dep.workspace = true` (see `core/Cargo.toml`). `default-members` keeps
plain `cargo build` away from the Android crate. Each file under
`core/src/` is a *module* declared with `pub mod ...` in `core/src/lib.rs`.

**Ownership and borrowing.** Every value has one owner; passing it by value
*moves* it. Most functions here borrow instead: `fn scan(&self, photo:
&GrayImage, ...)` (pipeline.rs) reads the photo without taking it, so the CLI
can reuse it. `&mut` is an exclusive, writable borrow (`let slot = &mut best[..]` in
`NameIndex::top_candidates` updates one element of a vector in place). The compiler checks that borrows never outlive the
owner — that is why `Scanner::crops` can hand `&self` to many threads at once
(shared `&` borrows are fine; `Read` results are *moved* into the output).
`clone()` makes an explicit copy when two owners are really needed (e.g. the
crops kept for pass 2 in `scan`).

**`Result` and `?`.** Fallible functions return `Result<T, Error>`.
`core/src/lib.rs` defines one `Error` enum; `#[derive(thiserror::Error)]`
writes `Display` for it and `#[from]` lets `?` convert e.g. a
`serde_json::Error` automatically. `?` means "if this is an error, return it
from the current function, otherwise unwrap the value" (see
`NameIndex::from_bytes`). The CLI uses `anyhow` instead, which wraps any
error and lets `.with_context(|| ...)` add a message (`cli/src/main.rs`).
`Option<T>` is the same idea for "maybe absent" (`find_marker`,
`best_pair`): `left.zip(right).and_then(...)` in `locate` reads "if both
markers exist, try to build the geometry".

**Traits.** A trait is an interface. `ocr::Recognizer` has one method,
`recognize`; `RtenRecognizer` implements it, and the column-dump example
implements a do-nothing `NoOcr`. `Scanner` stores a `Box<dyn Recognizer>`
(a pointer to *some* type implementing the trait, chosen at runtime). The
`Send + Sync` bounds promise the compiler it's safe to share across threads.
Derives like `#[derive(Debug, Clone, Serialize)]` generate standard trait
implementations — `output.rs` gets its whole JSON writer from `Serialize`.

**Iterators and closures.** Much of the code is iterator chains:
`s.nfkd().filter(..).flat_map(char::to_lowercase)` in `matching::normalize`
streams characters without temporary strings; `.map(...).collect()` builds a
`Vec`. `|x| ...` is a closure (an inline function that can capture local
variables, e.g. `to_row` in `text_rows`). `rayon` turns `iter()` into
`par_iter()` to spread work over cores (`Scanner::crops`, `match_all`) —
the borrow checker guarantees this is data-race free.

**Enums and `match`.** `MatchStatus`, `CropVariant`, `Config` are enums;
`match` must handle every variant, so adding one makes the compiler point at
every place to update.

**Tests.** `#[cfg(test)] mod tests` at the end of a file holds unit tests
(compiled only for `cargo test`); `core/tests/match_cases.rs` is an
integration test that only sees the public API.

**Android cross-compile flow.** `rustup target add aarch64-linux-android`
installs the standard library for phones; `cargo ndk -t arm64-v8a build`
points Cargo at the NDK's clang linker. `android/Cargo.toml` makes the crate
a `cdylib` (a C-style `.so`), and `#[cfg(target_os = "android")]` code plus
`[target.'cfg(target_os = "android")'.dependencies]` keep phone-only parts
(egui/winit/android-activity) out of the desktop build. Android's built-in
`NativeActivity` loads `libfscan_android.so` and calls our `android_main`
(`android/src/lib.rs`) — no Java needed. `build_apk.ps1` then does what
Gradle would: `aapt2` for the manifest/assets, zip in the `.so`, `zipalign`,
`apksigner`.

## Profiling and performance in Rust (as used in this code)

Phase 2 numbers and the full story are in [`PERF.md`](PERF.md). The tools
and patterns:

**Release profiles.** `cargo build --release` turns on the optimiser; the
OCR model is ~30x slower without it. `Cargo.toml` sets
`[profile.release] debug = "line-tables-only"` (fast code that still gives
file:line in panics and profilers) and `[profile.dev.package."*"]
opt-level = 3`, so even `cargo test` builds optimise the heavy dependencies
while our own crates stay quick to compile. Always time release builds.

**Timing with `Instant`.** `std::time::Instant::now()` / `.elapsed()` is a
monotonic clock costing tens of nanoseconds, cheap enough to leave in. The
`profile` module (`core/src/profile.rs`) wraps it: `prof.time(Stage::Warp,
|| warp_rect(...))` runs a closure and charges its duration to a stage.
Stages are counted in `AtomicU64`s so rayon worker threads can add to the
same `Profile` without a lock (`fetch_add` with `Ordering::Relaxed`: we only
need the final sums, not ordering between threads). `fscan bench --reps N`
(`cli/src/main.rs`) prints mean / p50 / p95 / share per stage, and every
JSON carries the breakdown in `timing_ms`.

**Wall time vs CPU time.** On a busy machine wall-clock time is noise.
`cli/src/cpu.rs` reads the process CPU time (a two-function FFI declaration of
`GetProcessTimes` on Windows, `/proc/self/stat` on Linux/Android), which
measures *work* and barely moves when other programs compete for the cores.
Optimise work first, then parallelism.

**Parallelism.** `rayon` turns `iter()` into `par_iter()` for independent
work (slot crops per column, matching). For the OCR model, rten would
otherwise split each layer over all cores, but this model's layers are too
small to split well, and it degrades badly under load. Instead
`ocr::run_on_workers` starts one scoped thread per core
(`std::thread::scope`, so they may borrow the crops) that pull whole batches
from an `AtomicUsize` counter, and each batch runs single-threaded on a
per-thread rten pool (`thread_local!`). We avoided rayon there on purpose:
a rayon thread that blocks inside another pool keeps stealing work, so far
more model runs than cores were in flight.

**Doing less work.** The biggest wins were algorithmic, not
micro-optimisation: a small fast model reads every slot and the accurate
one only re-reads the doubtful ones (`Scanner::scan`, pass 1b); fallback
crops run as a cascade that stops as soon as a slot is accepted and skips
variants that would duplicate the primary crop (`slot_crop`); the JPEG
decoder is asked for the luma plane only (`pipeline::decode_gray`).

**Allocation and memory layout.**
- `std::mem::take(&mut crop.image)` moves an image out of a struct (leaving
  an empty one) instead of `clone()`-ing ~50 KB per slot (`Scanner::scan`).
- `warp_rect` (`core/src/image_ops.rs`) allocates the output once and writes
  rows through `chunks_exact_mut` instead of a bounds-checked `put_pixel` per
  pixel, and steps the projective coordinates incrementally.
- `Vec::with_capacity` when the final size is known.
- Data-oriented layout in `NameIndex`: the 62k keys live back to back in one
  `Vec<u8>` grouped by oracle id ("struct of arrays"), so a query streams
  through memory instead of chasing one heap pointer per `String`; this
  halved the matching time. A length-based pruning we tried first was
  *slower*: it barely pruned and it scattered the memory accesses. Measure
  (`core/examples/match_bench.rs`), don't guess.
