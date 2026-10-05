//! fscan-core: the whole card-fscan pipeline (see `spec/SPEC.md`).

pub mod aruco;
pub mod geometry;
pub mod homography;
pub mod image_ops;
pub mod matching;
pub mod ocr;
pub mod output;
pub mod pipeline;

pub use pipeline::{ScanParams, Scanner};
// Re-exported so callers can write `fscan_core::Error`.


/// Errors returned by the library. `thiserror` derives `Display` and `From`.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("I/O error on {0}: {1}")]
    Io(String, #[source] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("image error: {0}")]
    Image(#[from] image::ImageError),
    #[error("OCR model error: {0}")]
    Model(String),
    #[error("invalid input: {0}")]
    Invalid(String),
}
