# Performance (Phase 2)

Goal: cut read time per photo as far as possible without losing accuracy,
with the phone (aarch64, typically 8 mixed big/little cores) as the real
target. All numbers are from the release build on this laptop unless noted.

**Machine.** Intel Core i5-3330 (Ivy Bridge, 2012), 4 cores / 4 threads,
3.0 GHz, AVX but **no AVX2/FMA**. rten picks its GEMM kernel at run time;
without AVX2+FMA it falls back to its portable `GenericKernel`, so this CPU
runs the OCR models ~8 GFLOP/s per core. On aarch64 rten uses NEON kernels,
so per-core speed on a phone is relatively better than this laptop suggests
(see "Phone estimate").

**Load during the work.** Other agents generated the 272-image `large`
set during most of Phase 2 (two Python workers at *normal* priority, ~2.3
cores), and a stray `find.exe / -name collection_full_trade_*.json` (started
16:35, not ours, left alone) used ~1 core all day. So: optimisation
decisions were made on **process CPU time** (work; insensitive to load) and
same-window A/B runs; the final before/after table below was re-measured
after the generator finished (only `find.exe` still running).

## Instrumentation

- `core/src/profile.rs`: a `Profile` with one `AtomicU64` nanosecond counter
  per stage plus event counters; timers are plain `Instant`s (overhead far
  below 0.1%). Stages inside parallel loops accumulate thread time.
- Every result JSON carries `timing_ms` with wall-clock steps (`decode`,
  `gray`, `detect`, `crop`, `ocr`, `match`, `retry`, `json`, `scan`,
  `total`), `stages` (thread time: detect, layout, warp, line_find, trim,
  enhance, ocr_prep, ocr_infer, ctc, clean, match) and `counters` (slots, OCR
  calls/batches/lines, lines read by the fast model, crops per slot, retry
  slots/crops, OCR input width).
- `fscan bench --reps N <images>` prints mean / p50 / p95 / % of total per
  stage, the counters, and **process CPU time per photo**
  (`cli/src/cpu.rs`).
- `scripts/gate.sh <label> [binary] [sets]` runs the accuracy gate (scan +
  `tools/eval.py`) and appends a line to `results/gate_summary.txt`;
  `scripts/profile_summary.py <results dir>` averages `timing_ms`.
- `core/examples/match_bench.rs` times the matcher on real OCR strings.

## Before / after (quiet window, smoke, 8 photos x 3 reps)

Measured 2026-10-06 ~01:20 after the generator had finished (only the stray
`find.exe` busy, ~1 core). "Before" = commit `8f0cf65` (Phase 1 pipeline +
instrumentation, identical behaviour to Phase 1 `d850406`); "after" = commit
`b5e7b2d`. Commands: `fscan bench --reps 3 testdata/smoke/*.jpg`.
A second baseline run right after gave 7.9 s/photo (no drift).

Wall-clock ms per photo:

| stage | before mean | before p50 / p95 | after mean | after p50 / p95 | change |
| --- | ---: | ---: | ---: | ---: | ---: |
| JPEG decode | 165 | 148 / 282 | 112 | 102 / 200 | -32% |
| grayscale | 178 | 157 / 296 | 0 | 0 / 0 | (luma decode) |
| marker detection + layout | 139 | 123 / 211 | 125 | 122 / 150 | -10% |
| slot crops (warp, line finding, trim) | 197 | 178 / 304 | 115 | 110 / 163 | -42% |
| OCR, all passes | 7269 | 7201 / 9781 | 3467 | 3724 / 4360 | **-52%** |
| matching, all passes | 143 | 124 / 270 | 94 | 92 / 127 | -34% |
| (of which fallback pass) | 1647 | 1827 / 2239 | 1002 | 1098 / 1308 | -39% |
| JSON | 0.2 | | 0.3 | | |
| **total per photo** | **8145** | 8010 / 11113 | **3915** | 4133 / 4952 | **-52% (2.1x)** |

Thread time per photo (summed over threads):

| stage | before | after |
| --- | ---: | ---: |
| warp | 763 | 321 |
| name-line finding / mana trim | 5 / 15 | 5 / 16 |
| OCR prep (resize + tensor) | 91 | 110 |
| OCR inference | 7165 (rten intra-op: 1 batch at a time) | 11307 (4 batches at a time, each single-threaded) |
| CTC decode | 3.5 | 37 (6.9k-class tiny model) |
| clean + matching | 532 | 340 |

Counters per photo (before -> after): OCR lines 97.0 -> 114.9 (of which
81.6 by the fast model, 33.3 by the accurate one); OCR calls 2 -> 3; batches
12.6 -> 30.4; retried slots 7.0 -> 6.5; crops per slot 1.2 -> 1.4.

**Process CPU time per photo** (work, all threads): before ~14.4 s
(measured with the instrumented baseline during the loaded period; CPU time
is insensitive to load), after **8.7 s** (-40%). OCR work estimated from
the models' multiply-accumulates (v5: 660 MMAC, tiny: 184 MMAC per 48x320
line) and the counters: ~68 -> ~41 GMAC per dev photo (-40%).

### ms/photo per set (full `fscan scan`, release, same quiet conditions)

| set | photos | Phase 1 | final | speed-up |
| --- | ---: | ---: | ---: | ---: |
| smoke | 8 | 7924 | 4839 | 1.6x |
| dev | 60 | 8638 | 4757 | 1.8x |
| large | 272 | 13757 * | 4447 | 3.1x |

\* The Phase 1 `large` run (23:10-00:11) started right after the generator
finished; it is slower per photo than its dev run for reasons I could not
pin down (possibly other activity in that hour; rten's intra-op threading
degrades sharply under any extra load, see #1 below), so the fair speed-up
is the smoke/dev 1.6-2.1x. Under load the new scheduler matters more: with
~3 of 4 cores busy elsewhere, Phase 1 took 58 s/photo and the new code
13-18 s.

## Accuracy gate (every kept change; full log in `results/gate_summary.txt`)

| build | smoke auto / wrong | dev auto / wrong | large auto / wrong | column / status errors |
| --- | --- | --- | --- | --- |
| Phase 1 baseline | 96.99% / 0 | 95.76% / 0 | 95.89% / **17** | 0 / 0 |
| cascade fallbacks + threading | 96.99% / 0 | 95.80% / 0 | | 0 / 0 |
| + fast warp, luma decode | 96.99% / 0 | 95.74% / 0 | | 0 / 0 |
| + PP-OCRv6 tiny first pass (no trust rule) | 96.99% / **1** | 95.86% / 0 | | rejected |
| + trust rule (name-bar shape) | 96.99% / 0 | 95.84% / 0 | | 0 / 0 |
| + single fallback round | 96.99% / 0 | 95.84% / 0 | | 0 / 0 |
| + batch 4 (commit `af3a42a`) | 96.99% / 0 | 95.82% / 0 | 95.90% / **20** | 0 / 0 |
| + broad all-caps guard | 96.81% / 0 | 95.32% / 0 | | rejected (auto rate) |
| **final** (`b5e7b2d`: decoy, fragment, space and camel-case guards) | **96.99% / 0** | **95.78% / 0** | **95.89% / 1** | **0 / 0** |

The `large` set (272 photos, 24,496 slots, new card pool) exposed problems
the Phase 1 baseline already had:

- **16 wrong in the baseline**: the *Breaking News* newspaper frame prints
  the masthead "THE PROSPERITY POST" where the name normally is, on many
  different cards; PP-OCRv5 reads `PRASPERITY` (score 90.0) -> auto
  "Prosperity". Fixed with a frame-decoy rule (an all-caps reading whose best
  match is a known decoy name is never auto) plus short all-caps fragments
  (`BAT`). A broad "never auto an all-caps reading" rule cost 0.5 pt on dev
  because several special frames legitimately print names in capitals
  ("CURIOSITY", "THE FOURTH DOCTOR"), so it was narrowed.
- **Two new risks from the fast model**, which drops spaces and garbles
  letters more often: `WasteLand` -> "Wasteland" (truth "Waste Land", a
  different card) and `BalÍightning` -> "Blightning" (truth "Ball
  Lightning"). Fixed by (a) never auto-accepting between two names equal up
  to spaces, and (b) not trusting a fast-model read that switches from lower
  to upper case inside a word unless it equals its match up to spaces.
- **1 remaining "wrong" is a test-data error**: `large/c4_0196.jpg` col 4
  slot 2 is labelled *Efígie Desdenhosa* (Scorn Effigy, PT, scryfall id
  9981a703-…), but Scryfall's image for that printing shows the name bar
  **"Coroa Rúnica"** (Runed Crown) - the reader reads exactly what is
  printed. Every correct reader will fail this slot; reported to the
  coordinator rather than worked around.

All guards only ever turn an auto into review (they cannot create an auto),
so they cannot raise the wrong count.

## What was tried, in order (each gated on smoke + dev)

| # | change | effect | gate (smoke / dev auto, wrong) | kept |
| --- | --- | --- | --- | --- |
| 0 | Phase 1 baseline | ~14.4 s CPU / photo, OCR 91% | 96.99% / 95.76%, 0 | — |
| 1 | OCR threading: whole batches in parallel, one per core, each single-threaded (`Threading::InterBatch`), instead of rten splitting every layer over all cores | same CPU; under load wall 13-18 s vs **58 s** for intra-op (rten's layer-level parallelism collapses when cores are busy) | with #2 | yes |
| 1a | …first version used rayon `par_iter` | rayon threads blocked inside rten's pool kept stealing more batches → far more runs in flight than cores; replaced by scoped OS threads pulling from an atomic counter | — | replaced |
| 2 | fallback variants as a cascade, skipping duplicate crops (untrimmed only if the trim cut something) | OCR lines/photo 106 → 95 (smoke); rescues are rare (16 of 293 retried slots) | 96.99% / 95.80%, 0 | partly (see #8) |
| 3 | fast warp (incremental homogeneous coords, direct buffer writes) | warp thread time 1596 → 513 ms/photo | with #4: 96.99% / 95.74%, 0 | yes |
| 4 | luma-only JPEG decode (zune-jpeg `ColorSpace::Luma`) | decode+gray 921 → 227 ms | (with #3) | yes |
| 5 | int8 dynamic quantisation (onnxruntime `quantize_dynamic`, rten runs `ConvInteger`) | per-tensor: 16/80 auto on c4_0001 (vs 78) and 5x slower here; per-channel: 17/80; MatMul-only: accuracy ok but MatMul is <1% of the model | — | no |
| 6 | matching: length-bound pruning | **slower** (24.8 vs 11 ms/query): bound rarely prunes (3rd-best score ~60-80) and scattered memory access | exactness test passes | no |
| 6b | matching: flat, oracle-grouped key layout (one `Vec<u8>`, sequential scan, no per-query scratch array) | 2x faster than per-`String` layout, results identical (test) | with #7 | yes |
| 7 | **reader cascade**: PP-OCRv6 *tiny* (1.1M params, 184 MMAC/line vs 660 for v5) reads every slot; PP-OCRv5 re-reads the rest | CPU 14.6 → 8.5 s on 2 photos | dev 95.86% / 0 wrong, but **smoke 1 wrong** (tiny read the "Prosperity Post" showcase masthead `/PROSPERITY8O0` → auto "Prosperity") | fixed by 7b |
| 7a | trust a tiny auto only at score ≥ 95, else v5 must confirm | passes (96.99% / 95.80%, 0) but 40% of slots go to v5 (147 lines/photo) | pass | replaced |
| 7b | trust a tiny auto at ≥ 95 **or** when the text looks like a name bar (starts upper-case, has lower-case); otherwise v5 must also auto-accept or the slot goes to review | 124 lines/photo, ~44% less OCR work than v5 alone | 96.99% / 95.84%, 0 | yes |
| 8 | fallback variants in one parallel round instead of a 3-round cascade | wall −8%, CPU +5% (cascade rarely stops early; small sequential rounds idle the cores) | 96.99% / 95.84%, 0 | yes |
| 9 | adaptive batch size (`min(batch, lines / workers)`) and default batch 4 | keeps all cores busy on small rounds; finer granularity for big/little phone cores; CPU unchanged on x86 (batch 1/4/8 all equal) | 96.99% / 95.82%, 0 | yes |
| — | not tried / rejected: smaller model input height (PP-OCRv5 needs 48 px: 32 px fails in an AveragePool), horizontal squeeze of the line (0.75x lost 10 pts of exact reads) | | | |

## Phone estimate (no device yet)

`fscan-core` + `fscan-android` cross-compile for `aarch64-linux-android`
(cargo-ndk, release) and the APK still builds and verifies:
`rust/android/build/card-fscan-rs-debug.apk`, 24.4 MB (it now also ships the
4.5 MB fast model as `fast.onnx`).

Reasoning, for a C4 photo (80 slots, 12 MP):

- OCR work after Phase 2: ~35 GMAC = ~70 GFLOP (smoke counters); before:
  ~61 GMAC.
- This laptop runs rten's portable kernel at ~9 GFLOP/s per core (measured:
  ~7.7 s of OCR CPU for ~70 GFLOP). On aarch64 rten uses its NEON GEMM
  kernels; for a Cortex-A76/A78 big core at 2.2-2.6 GHz (peak ~35-40
  GFLOP/s fp32) I assume 15-20 GFLOP/s effective on this model, and 3-5
  GFLOP/s for a Cortex-A55 little core.
- Typical mid-range SoC (2 big + 6 little, e.g. Dimensity 7050 / Helio G99
  class): ~2 x 17 + 6 x 4 = ~58 GFLOP/s aggregate; with the work-queue
  scheduler (whole batches of 4 lines, no per-layer synchronisation, so slow
  cores just take fewer batches) and ~25% loss for thermal throttling and
  tail effects: ~40-45 GFLOP/s -> **OCR ~1.6-1.8 s**.
- Non-OCR: 12 MP luma decode ~0.2-0.3 s, marker detection + crops ~0.3-0.5 s
  on big cores, matching ~0.1 s.

**Estimate: ~2.5-4 s per C4 photo, ~4-6 s per C6 photo (24 MP, 120 slots)**
on a mid-range phone; a pessimistic case (A55-heavy SoC, sustained
throttling) is about twice that. Phase 1 code would be roughly 2x slower on
the same phone (≈1.75x the OCR work, and rten's intra-op threading suffers
most on big/little cores). This is an estimate to be replaced by a
measurement: the app prints scan/OCR timings, and the first device run
should be recorded here.

## Ideas not done (next steps)

- Feed the fallback round to the fast model first (it is now ~25% of wall
  time for ~6 slots/photo; most of those slots are unreadable by design:
  foreign printings, special frames).
- Calibrate the batch size and worker count on a real phone
  (`RtenRecognizer::batch_size`, `available_parallelism`); consider pinning
  OCR to big cores only if little cores turn out to hurt.
- Width: crops could be cut on the left as well (~5% of OCR width).
- Desktop only: rten has no AVX(1)-only kernel; on this CPU `ort` (ONNX
  Runtime, now possible with MSVC) ran the same model ~2x faster per line.
  Not pursued: the phone is the target and `ort` complicates the Android
  build.
