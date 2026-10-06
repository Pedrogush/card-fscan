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

BASELINE_AND_FINAL

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

LARGE_AND_PHONE
