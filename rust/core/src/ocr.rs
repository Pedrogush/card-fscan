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
use rten::Model;
use rten_tensor::prelude::*;
use rten_tensor::{NdTensor, NdTensorView, Tensor};

use crate::Error;

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
    fn recognize(&self, lines: &[GrayImage]) -> Result<Vec<OcrLine>, Error>;
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
        Ok(RtenRecognizer { model, kind, alphabet, batch_size: 16, max_width: 640 })
    }

    pub fn kind(&self) -> ModelKind {
        self.kind
    }

    /// Resize lines to the model height and pack them into one NCHW tensor,
    /// padded on the right to the widest line.
    fn prepare(&self, lines: &[GrayImage]) -> NdTensor<f32, 4> {
        let h = self.kind.input_height();
        let resized: Vec<GrayImage> = lines
            .iter()
            .map(|img| {
                let w = ((img.width() as f64 * h as f64 / img.height().max(1) as f64).round() as u32)
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

    fn run_batch(&self, lines: &[GrayImage]) -> Result<Vec<OcrLine>, Error> {
        let input: Tensor<f32> = self.prepare(lines).into();
        let output = self
            .model
            .run_one(input.view().into(), None)
            .map_err(|e| Error::Model(e.to_string()))?;
        let mut probs: NdTensor<f32, 3> =
            output.try_into().map_err(|_| Error::Model("expected a 3-D recognition output".into()))?;
        if self.kind == ModelKind::Ocrs {
            // [seq, batch, class] -> [batch, seq, class]
            probs.permute([1, 0, 2]);
        }
        let log_probs = self.kind == ModelKind::Ocrs;
        Ok((0..lines.len())
            .map(|n| ctc_greedy(probs.slice(n), &self.alphabet, log_probs))
            .collect())
    }
}

impl Recognizer for RtenRecognizer {
    fn recognize(&self, lines: &[GrayImage]) -> Result<Vec<OcrLine>, Error> {
        let mut out = Vec::with_capacity(lines.len());
        for chunk in lines.chunks(self.batch_size.max(1)) {
            out.extend(self.run_batch(chunk)?);
        }
        Ok(out)
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
