//! Single-line text recognition (no text detection: every slot crop is one
//! line) with CTC greedy decoding, running ONNX/RTen models on the pure-Rust
//! `rten` runtime.
//!
//! Two model families are supported so they can be compared (see README):
//! - PaddleOCR PP-OCR recognition models (`*_rec_mobile.onnx`, Apache-2.0),
//!   input `[N, 3, 48, W]` in -1..1, output `[N, W/8, classes]` softmax.
//! - The `ocrs` recognition model (`text-recognition.rten`, MIT/Apache),
//!   input `[N, 1, 64, W]` in -0.5..0.5, output `[W/4, N, classes]` log-softmax.

use std::path::Path;

use image::GrayImage;
use image::imageops::{FilterType, resize};
use std::sync::Arc;

use std::sync::atomic::{AtomicUsize, Ordering};
use rten::{Model, RunOptions, ThreadPool};
use rten_tensor::prelude::*;
use rten_tensor::{NdTensor, NdTensorView, Tensor};

use crate::Error;
use crate::profile::{Count, Profile, Stage};
use std::time::Instant;

/// One recognised line.
#[derive(Debug, Clone, PartialEq)]
pub struct OcrLine {
    pub text: String,
    /// Mean probability of the emitted characters (0..1).
    pub confidence: f32,
}

/// Anything that turns line images into text. The pipeline only depends on
/// this trait, so tests can plug in a fake recogniser.
///
/// `Send + Sync` lets one recogniser be shared between threads.
pub trait Recognizer: Send + Sync {
    /// Read every line image. Implementations charge their work to `prof`.
    fn recognize(&self, lines: &[GrayImage], prof: &Profile) -> Result<Vec<OcrLine>, Error>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelKind {
    PaddleRec,
    Ocrs,
}

impl ModelKind {
    fn input_height(self) -> u32 {
        match self {
            ModelKind::PaddleRec => 48,
            ModelKind::Ocrs => 64,
        }
    }
}

/// A recognition model plus its character set.
pub struct RtenRecognizer {
    model: Model,
    kind: ModelKind,
    /// `alphabet[i]` is the character for class `i + 1` (class 0 is the CTC
    /// blank).
    alphabet: Vec<char>,
    /// Lines are processed in batches of this size.
    pub batch_size: usize,
    /// Widest input (pixels after resizing) a line may have.
    pub max_width: u32,
    /// Height lines are resized to (the model's training height by default).
    pub input_height: u32,
    /// Horizontal squeeze applied after resizing to `input_height` (1.0 =
    /// keep the aspect ratio). Narrower inputs are proportionally faster.
    pub width_scale: f64,
    /// How batches use the CPU cores; see [`Threading`].
    pub threading: Threading,
}

/// Two ways to spread OCR over cores.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Threading {
    /// One batch at a time; rten splits each operator across its own thread
    /// pool. Good for big models, poor for this small one: most layers are
    /// too small to split well.
    IntraOp,
    /// Several batches at once on rayon's pool (one per core), each run
    /// single-threaded. Every core does useful work all the time.
    InterBatch,
}

/// Run `f` on every item using `workers` plain OS threads that pull the next
/// unclaimed item from a shared atomic counter, and return the results in
/// item order.
///
/// Why not rayon's `par_iter`? rten hands each run to the given (one-thread)
/// pool and waits; a *rayon* thread that waits like that keeps stealing other
/// items meanwhile, so far more runs than cores end up in flight. Scoped
/// threads (`std::thread::scope`) may borrow local data, and exactly
/// `workers` runs execute at once.
fn run_on_workers<T: Sync, R: Send>(
    items: &[T],
    workers: usize,
    f: impl Fn(&T) -> Result<R, Error> + Sync,
) -> Result<Vec<R>, Error> {
    let next = AtomicUsize::new(0);
    let mut slots: Vec<Option<Result<R, Error>>> = (0..items.len()).map(|_| None).collect();
    let per_thread: Vec<Vec<(usize, Result<R, Error>)>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..workers.clamp(1, items.len().max(1)))
            .map(|_| {
                scope.spawn(|| {
                    let mut done = Vec::new();
                    loop {
                        let k = next.fetch_add(1, Ordering::Relaxed);
                        let Some(item) = items.get(k) else { break };
                        done.push((k, f(item)));
                    }
                    done
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("OCR worker panicked")).collect()
    });
    for (k, r) in per_thread.into_iter().flatten() {
        slots[k] = Some(r);
    }
    slots.into_iter().map(|r| r.expect("every item was processed")).collect()
}

thread_local! {
    /// A one-thread rten pool per worker thread, so a model run started on
    /// this thread stays single-threaded (rten would otherwise use its global,
    /// all-cores pool). `thread_local!` gives each OS thread its own copy.
    static SERIAL_POOL: Arc<ThreadPool> = Arc::new(ThreadPool::with_num_threads(1));
}

/// The `ocrs` model's character set (copied from the ocrs crate, MIT/Apache).
const OCRS_ALPHABET: &str = " 0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~EABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

impl RtenRecognizer {
    /// Load a model from disk. For PaddleOCR models the dictionary is read
    /// from `<model stem>.dict.txt` next to it (one character per line).
    pub fn load(path: &Path) -> Result<Self, Error> {
        let bytes = std::fs::read(path).map_err(|e| Error::Io(path.display().to_string(), e))?;
        let is_rten = path.extension().is_some_and(|e| e == "rten");
        let dict = if is_rten {
            None
        } else {
            let dict_path = path.with_extension("dict.txt");
            Some(std::fs::read_to_string(&dict_path).map_err(|e| Error::Io(dict_path.display().to_string(), e))?)
        };
        Self::from_bytes(bytes, dict.as_deref())
    }

    /// `dict = None` means the built-in `ocrs` model and alphabet.
    pub fn from_bytes(model_bytes: Vec<u8>, dict: Option<&str>) -> Result<Self, Error> {
        let model = Model::load(model_bytes).map_err(|e| Error::Model(e.to_string()))?;
        let (kind, alphabet) = match dict {
            Some(d) => (ModelKind::PaddleRec, parse_dict(d)),
            None => (ModelKind::Ocrs, OCRS_ALPHABET.chars().collect()),
        };
        let input_height = kind.input_height();
        Ok(RtenRecognizer { model, kind, alphabet, batch_size: 8, max_width: 640, input_height, width_scale: 1.0, threading: Threading::InterBatch })
    }

    /// Number of concurrent model runs in [`Threading::InterBatch`] mode.
    fn workers(&self) -> usize {
        std::thread::available_parallelism().map_or(4, |n| n.get())
    }

    pub fn kind(&self) -> ModelKind {
        self.kind
    }

    /// Resize lines to the model height and pack them into one NCHW tensor,
    /// padded on the right to the widest line.
    fn prepare(&self, lines: &[GrayImage]) -> NdTensor<f32, 4> {
        let h = self.input_height;
        let resized: Vec<GrayImage> = lines
            .iter()
            .map(|img| {
                let w = ((self.width_scale * img.width() as f64 * h as f64 / img.height().max(1) as f64).round() as u32)
                    .clamp(h / 2, self.max_width);
                resize(img, w, h, FilterType::Triangle)
            })
            .collect();
        // Round the width up to a multiple of 8 (the model's downsampling).
        let width = resized.iter().map(|r| r.width()).max().unwrap_or(h).div_ceil(8) * 8;
        let channels = match self.kind {
            ModelKind::PaddleRec => 3,
            ModelKind::Ocrs => 1,
        };
        let (pad, norm): (f32, fn(u8) -> f32) = match self.kind {
            ModelKind::PaddleRec => (0.0, |v| v as f32 / 127.5 - 1.0),
            ModelKind::Ocrs => (-0.5, |v| v as f32 / 255.0 - 0.5),
        };
        let mut t = NdTensor::full([lines.len(), channels, h as usize, width as usize], pad);
        for (n, img) in resized.iter().enumerate() {
            for (x, y, px) in img.enumerate_pixels() {
                let v = norm(px[0]);
                for c in 0..channels {
                    t[[n, c, y as usize, x as usize]] = v;
                }
            }
        }
        t
    }

    fn run_batch(&self, lines: &[GrayImage], prof: &Profile) -> Result<Vec<OcrLine>, Error> {
        let t = Instant::now();
        let input: Tensor<f32> = self.prepare(lines).into();
        prof.add(Stage::OcrPrep, t);
        prof.count(Count::OcrBatches, 1);
        prof.count(Count::OcrPixelsWide, (input.size(0) * input.size(3)) as u64);
        let t = Instant::now();
        let opts = match self.threading {
            Threading::IntraOp => None,
            Threading::InterBatch => {
                // `RunOptions` is `#[non_exhaustive]` (no struct literal from
                // outside rten), so use its builder method.
                Some(RunOptions::default().with_thread_pool(Some(SERIAL_POOL.with(Arc::clone))))
            }
        };
        let output = self
            .model
            .run_one(input.view().into(), opts)
            .map_err(|e| Error::Model(e.to_string()))?;
        prof.add(Stage::OcrInfer, t);
        let t = Instant::now();
        let mut probs: NdTensor<f32, 3> =
            output.try_into().map_err(|_| Error::Model("expected a 3-D recognition output".into()))?;
        if self.kind == ModelKind::Ocrs {
            // [seq, batch, class] -> [batch, seq, class]
            probs.permute([1, 0, 2]);
        }
        let log_probs = self.kind == ModelKind::Ocrs;
        let out = (0..lines.len()).map(|n| ctc_greedy(probs.slice(n), &self.alphabet, log_probs)).collect();
        prof.add(Stage::Ctc, t);
        Ok(out)
    }
}

impl Recognizer for RtenRecognizer {
    fn recognize(&self, lines: &[GrayImage], prof: &Profile) -> Result<Vec<OcrLine>, Error> {
        prof.count(Count::OcrCalls, 1);
        prof.count(Count::OcrLines, lines.len() as u64);
        // Batch lines of similar aspect ratio together so little compute is
        // wasted on right padding, then put the results back in input order.
        let mut order: Vec<usize> = (0..lines.len()).collect();
        let aspect = |i: usize| lines[i].width() as f64 / lines[i].height().max(1) as f64;
        order.sort_by(|&a, &b| aspect(a).total_cmp(&aspect(b)));
        // Small rounds (e.g. the few slots a fallback pass re-reads) would
        // fill only one or two batches and leave the other cores idle, so the
        // batch shrinks until every worker gets one.
        let batch = match self.threading {
            Threading::IntraOp => self.batch_size,
            Threading::InterBatch => self.batch_size.min(lines.len().div_ceil(self.workers())),
        };
        let chunks: Vec<&[usize]> = order.chunks(batch.max(1)).collect();
        let run = |chunk: &&[usize]| {
            let batch: Vec<GrayImage> = chunk.iter().map(|&i| lines[i].clone()).collect();
            self.run_batch(&batch, prof)
        };
        let results: Vec<Vec<OcrLine>> = match self.threading {
            // `collect::<Result<_, _>>()` stops at the first error.
            Threading::IntraOp => chunks.iter().map(run).collect::<Result<_, _>>()?,
            Threading::InterBatch => run_on_workers(&chunks, self.workers(), run)?,
        };
        let mut out = vec![None; lines.len()];
        for (chunk, lines_out) in chunks.iter().zip(results) {
            for (&i, line) in chunk.iter().zip(lines_out) {
                out[i] = Some(line);
            }
        }
        // Every slot was filled above, so `flatten` drops nothing.
        Ok(out.into_iter().flatten().collect())
    }
}

fn parse_dict(d: &str) -> Vec<char> {
    // One char per line. A line holding just " " is the space class; the
    // file ends with a newline, so drop only the final empty line.
    let mut lines: Vec<&str> = d.split('\n').collect();
    if lines.last() == Some(&"") {
        lines.pop();
    }
    lines.iter().map(|l| l.trim_end_matches('\r').chars().next().unwrap_or(' ')).collect()
}

/// Greedy CTC decoding: take the best class at each time step, merge repeats,
/// drop blanks (class 0).
fn ctc_greedy(seq: NdTensorView<f32, 2>, alphabet: &[char], log_probs: bool) -> OcrLine {
    let mut text = String::new();
    let mut confs = Vec::new();
    let mut prev = 0usize;
    for t in 0..seq.size(0) {
        let row = seq.slice(t);
        let (best, p) = row
            .iter()
            .copied()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap_or((0, 0.0));
        if best != 0 && best != prev {
            if let Some(&c) = alphabet.get(best - 1) {
                text.push(c);
                confs.push(if log_probs { p.exp() } else { p });
            }
        }
        prev = best;
    }
    let confidence = if confs.is_empty() { 0.0 } else { confs.iter().sum::<f32>() / confs.len() as f32 };
    OcrLine { text: text.trim().to_string(), confidence }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_decoding_merges_repeats_and_drops_blanks() {
        // classes: 0 blank, 1 'a', 2 'b'
        let probs = NdTensor::from_data(
            [6, 3],
            vec![
                0.1, 0.8, 0.1, // a
                0.1, 0.8, 0.1, // a (repeat, merged)
                0.9, 0.05, 0.05, // blank
                0.1, 0.8, 0.1, // a (new after blank)
                0.1, 0.1, 0.8, // b
                0.9, 0.05, 0.05, // blank
            ],
        );
        let line = ctc_greedy(probs.view(), &['a', 'b'], false);
        assert_eq!(line.text, "aab");
        assert!((line.confidence - 0.8).abs() < 1e-6);
    }

    #[test]
    fn dict_keeps_space_class() {
        assert_eq!(parse_dict("a\nb\n \n"), vec!['a', 'b', ' ']);
    }
}
