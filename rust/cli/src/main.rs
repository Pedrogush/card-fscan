//! `fscan scan --out <dir> [--names <names_v1.json>] [--model <rec.onnx>] <image>...`
//!
//! Writes `<dir>/<image basename>.json` per photo (SPEC §5) and prints one
//! timing line per photo.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use fscan_core::Scanner;
use fscan_core::matching::NameIndex;
use fscan_core::ocr::RtenRecognizer;

struct Args {
    out: PathBuf,
    names: PathBuf,
    model: PathBuf,
    images: Vec<PathBuf>,
}

const USAGE: &str = "usage: fscan scan --out <dir> [--names <names_v1.json[.gz]>] [--model <rec model>] <image>...";

/// Hand-rolled argument parsing keeps the dependency list short.
fn parse_args() -> Result<Args> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut args = std::env::args().skip(1);
    if args.next().as_deref() != Some("scan") {
        bail!(USAGE);
    }
    let mut out = None;
    let mut names = root.join("testdata/names/names_v1.json");
    let mut model = root.join("rust/models/en_PP-OCRv4_rec_mobile.onnx");
    let mut images = Vec::new();
    while let Some(a) = args.next() {
        // `ok_or_else` turns a missing value into an error; `?` returns it.
        let mut value = || args.next().with_context(|| format!("{a} needs a value"));
        match a.as_str() {
            "--out" => out = Some(PathBuf::from(value()?)),
            "--names" => names = PathBuf::from(value()?),
            "--model" => model = PathBuf::from(value()?),
            "-h" | "--help" => bail!(USAGE),
            _ => images.push(PathBuf::from(a)),
        }
    }
    let out = out.context(USAGE)?;
    if images.is_empty() {
        bail!(USAGE);
    }
    Ok(Args { out, names, model, images })
}

fn main() -> Result<()> {
    let args = parse_args()?;
    std::fs::create_dir_all(&args.out).with_context(|| format!("creating {}", args.out.display()))?;

    let t = Instant::now();
    let index = NameIndex::load(&args.names).with_context(|| format!("loading {}", args.names.display()))?;
    let recognizer = RtenRecognizer::load(&args.model).with_context(|| format!("loading {}", args.model.display()))?;
    let scanner = Scanner::new(index, Box::new(recognizer));
    eprintln!("loaded {} names and the OCR model in {} ms", scanner.index.len(), t.elapsed().as_millis());

    let mut total_ms = 0u128;
    for path in &args.images {
        let t = Instant::now();
        let name = path.file_name().context("image path has no file name")?.to_string_lossy().to_string();
        let photo = image::open(path).with_context(|| format!("reading {}", path.display()))?.to_luma8();
        let decode_ms = t.elapsed().as_millis();
        let result = scanner.scan(&photo, &name)?;
        let out_path = args.out.join(format!("{name}.json"));
        std::fs::write(&out_path, serde_json::to_string_pretty(&result)?)?;
        let ms = t.elapsed().as_millis();
        total_ms += ms;
        let slots: usize = result.columns.iter().map(|c| c.slots.len()).sum();
        let auto = result.columns.iter().flat_map(|c| &c.slots).filter(|s| s.status == fscan_core::output::SlotStatus::Auto).count();
        println!(
            "{name}: {:?} {:?} auto {auto}/{slots}  {ms} ms (decode {decode_ms}, detect {}, crop {}, ocr {}, match {})",
            result.config, result.status, result.timing_ms.detect, result.timing_ms.crop, result.timing_ms.ocr, result.timing_ms.match_
        );
    }
    println!("{} photos, mean {} ms/photo", args.images.len(), total_ms / args.images.len() as u128);
    Ok(())
}
