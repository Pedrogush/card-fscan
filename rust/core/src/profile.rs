//! Lightweight per-stage instrumentation.
//!
//! A [`Profile`] holds one atomic nanosecond counter per [`Stage`] plus a few
//! event counters. Atomics let rayon worker threads add to the same profile
//! without locks; each update is one `fetch_add`, and a timer is two
//! `Instant::now()` calls (~20-40 ns each), so the overhead is far below
//! 0.1% of a scan.
//!
//! Stages that run inside parallel loops accumulate *thread time* (summed
//! over all threads), so they can add up to more than the wall-clock time of
//! the enclosing step. The top-level steps in [`Timing`] are wall-clock.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use crate::output::{Counters, StageTimes};

/// Fine-grained pipeline stages (thread time).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// Marker detection: downscale + threshold + contours + decode + refine.
    Detect,
    /// Pairing markers, homographies, stop card.
    Layout,
    /// Perspective warp of slot bands.
    Warp,
    /// Finding the name line (row energy) incl. drift prediction.
    LineFind,
    /// Cutting off the mana cost (column energy).
    Trim,
    /// CLAHE and inversion.
    Enhance,
    /// OCR input preparation: resize to model height + tensor packing.
    OcrPrep,
    /// The model itself (`rten` run).
    OcrInfer,
    /// CTC greedy decoding.
    Ctc,
    /// SPEC §3 cleaning + normalisation.
    Clean,
    /// Scoring against the name index + accept rule.
    Match,
}

impl Stage {
    pub const ALL: [Stage; 11] = [
        Stage::Detect,
        Stage::Layout,
        Stage::Warp,
        Stage::LineFind,
        Stage::Trim,
        Stage::Enhance,
        Stage::OcrPrep,
        Stage::OcrInfer,
        Stage::Ctc,
        Stage::Clean,
        Stage::Match,
    ];
}

/// Event counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Count {
    /// Calls to `Recognizer::recognize`.
    OcrCalls,
    /// Model runs (batches).
    OcrBatches,
    /// Lines (crops) recognised.
    OcrLines,
    /// Slots that went to the fallback pass.
    RetrySlots,
    /// Fallback crops produced.
    RetryCrops,
    /// Sum of model input widths in pixels (a proxy for OCR work; includes padding).
    OcrPixelsWide,
    /// Lines read by the fast first-pass reader (included in `OcrLines`).
    FastLines,
}

const N_STAGES: usize = Stage::ALL.len();
const N_COUNTS: usize = 7;

#[derive(Default)]
pub struct Profile {
    nanos: [AtomicU64; N_STAGES],
    counts: [AtomicU64; N_COUNTS],
}

impl Profile {
    pub fn new() -> Self {
        Self::default()
    }

    /// Run `f`, charging its duration to `stage`, and return its result.
    #[inline]
    pub fn time<T>(&self, stage: Stage, f: impl FnOnce() -> T) -> T {
        let t = Instant::now();
        let out = f();
        self.add(stage, t);
        out
    }

    /// Charge the time since `since` to `stage`.
    #[inline]
    pub fn add(&self, stage: Stage, since: Instant) {
        self.nanos[stage as usize].fetch_add(since.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    #[inline]
    pub fn count(&self, what: Count, n: u64) {
        self.counts[what as usize].fetch_add(n, Ordering::Relaxed);
    }

    pub fn ms(&self, stage: Stage) -> f64 {
        self.nanos[stage as usize].load(Ordering::Relaxed) as f64 / 1e6
    }

    pub fn get(&self, what: Count) -> u64 {
        self.counts[what as usize].load(Ordering::Relaxed)
    }

    pub fn stage_times(&self) -> StageTimes {
        StageTimes {
            detect: self.ms(Stage::Detect),
            layout: self.ms(Stage::Layout),
            warp: self.ms(Stage::Warp),
            line_find: self.ms(Stage::LineFind),
            trim: self.ms(Stage::Trim),
            enhance: self.ms(Stage::Enhance),
            ocr_prep: self.ms(Stage::OcrPrep),
            ocr_infer: self.ms(Stage::OcrInfer),
            ctc: self.ms(Stage::Ctc),
            clean: self.ms(Stage::Clean),
            match_: self.ms(Stage::Match),
        }
    }

    pub fn counters(&self, slots: usize) -> Counters {
        let lines = self.get(Count::OcrLines);
        Counters {
            slots: slots as u64,
            ocr_calls: self.get(Count::OcrCalls),
            ocr_batches: self.get(Count::OcrBatches),
            ocr_lines: lines,
            crops_per_slot: if slots == 0 { 0.0 } else { lines as f64 / slots as f64 },
            retry_slots: self.get(Count::RetrySlots),
            retry_crops: self.get(Count::RetryCrops),
            ocr_input_px_wide: self.get(Count::OcrPixelsWide),
            fast_lines: self.get(Count::FastLines),
        }
    }
}

/// Milliseconds since `t` as a float (sub-millisecond precision).
#[inline]
pub fn ms_since(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}
