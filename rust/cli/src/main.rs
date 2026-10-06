//! card-fscan desktop CLI.
//!
//! ```text
//! fscan scan  --out <dir> [options] <image>...     one JSON per photo (SPEC §5)
//! fscan bench [--reps N] [--out <dir>] [options] <image>...   per-stage timing table
//! options: --names <names_v1.json[.gz]>  --model <rec model>  --dump-crops <dir>
//!          --fast-model <rec model> (default: PP-OCRv6 tiny)  --no-fast (accurate model only)
//!          --ocr-batch N  --intra-op (rten's own threading, the Phase 1 behaviour)
//! ```

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use fscan_core::Scanner;
use fscan_core::matching::NameIndex;
use fscan_core::ocr::{RtenRecognizer, Threading};
use fscan_core::output::{PhotoResult, SlotStatus, Timing};
use fscan_core::pipeline::decode_gray;

mod cpu;

#[derive(PartialEq)]
enum Mode {
    Scan,
    Bench,
}

struct Args {
    mode: Mode,
    out: Option<PathBuf>,
    names: PathBuf,
    model: PathBuf,
    fast_model: Option<PathBuf>,
    dump: Option<PathBuf>,
    reps: usize,
    ocr_batch: Option<usize>,
    intra_op: bool,
    images: Vec<PathBuf>,
}

const USAGE: &str = "usage: fscan scan --out <dir> [--names <names_v1.json[.gz]>] [--model <rec model>] [--dump-crops <dir>] <image>...\n       fscan bench [--reps N] [--out <dir>] [same options] <image>...";

/// Hand-rolled argument parsing keeps the dependency list short.
fn parse_args() -> Result<Args> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut args = std::env::args().skip(1);
    let mode = match args.next().as_deref() {
        Some("scan") => Mode::Scan,
        Some("bench") => Mode::Bench,
        _ => bail!(USAGE),
    };
    let mut parsed = Args {
        mode,
        out: None,
        names: root.join("testdata/names/names_v1.json"),
        model: root.join("rust/models/en_PP-OCRv5_rec_mobile.onnx"),
        fast_model: Some(root.join("rust/models/PP-OCRv6_rec_tiny.onnx")),
        dump: None,
        reps: 3,
        ocr_batch: None,
        intra_op: false,
        images: Vec::new(),
    };
    while let Some(a) = args.next() {
        // A closure that fetches the option's value or fails with a message;
        // `?` returns that error from `parse_args`.
        let mut value = || args.next().with_context(|| format!("{a} needs a value"));
        match a.as_str() {
            "--out" => parsed.out = Some(PathBuf::from(value()?)),
            "--names" => parsed.names = PathBuf::from(value()?),
            "--model" => parsed.model = PathBuf::from(value()?),
            "--fast-model" => parsed.fast_model = Some(PathBuf::from(value()?)),
            "--no-fast" => parsed.fast_model = None,
            "--dump-crops" => parsed.dump = Some(PathBuf::from(value()?)),
            "--ocr-batch" => parsed.ocr_batch = Some(value()?.parse().context("--ocr-batch needs a number")?),
            "--intra-op" => parsed.intra_op = true,
            "--reps" => parsed.reps = value()?.parse().context("--reps needs a number")?,
            "-h" | "--help" => bail!(USAGE),
            _ => parsed.images.push(PathBuf::from(a)),
        }
    }
    if parsed.images.is_empty() || (parsed.mode == Mode::Scan && parsed.out.is_none()) {
        bail!(USAGE);
    }
    Ok(parsed)
}

/// Decode, scan and serialise one photo, filling the CLI-side timings.
fn run_one(scanner: &Scanner, path: &Path) -> Result<(PhotoResult, String)> {
    let name = path.file_name().context("image path has no file name")?.to_string_lossy().to_string();
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let t = Instant::now();
    let (photo, decode_ms, gray_ms) = decode_gray(&bytes)?;
    let mut result = scanner.scan(&photo, &name)?;
    // Time the serialisation, then store that time and serialise again (the
    // second pass is what gets written; it costs the same ~1 ms).
    let t_json = Instant::now();
    serde_json::to_string_pretty(&result)?;
    let tm = &mut result.timing_ms;
    tm.json = t_json.elapsed().as_secs_f64() * 1e3;
    tm.decode = decode_ms;
    tm.gray = gray_ms;
    tm.total = t.elapsed().as_secs_f64() * 1e3;
    let json = serde_json::to_string_pretty(&result)?;
    Ok((result, json))
}

fn main() -> Result<()> {
    let args = parse_args()?;
    if let Some(out) = &args.out {
        std::fs::create_dir_all(out).with_context(|| format!("creating {}", out.display()))?;
    }

    let t = Instant::now();
    let index = NameIndex::load(&args.names).with_context(|| format!("loading {}", args.names.display()))?;
    let mut recognizer =
        RtenRecognizer::load(&args.model).with_context(|| format!("loading {}", args.model.display()))?;
    if let Some(b) = args.ocr_batch {
        recognizer.batch_size = b;
    }
    if args.intra_op {
        recognizer.threading = Threading::IntraOp;
    }
    let mut scanner = Scanner::new(index, Box::new(recognizer));
    if let Some(path) = &args.fast_model {
        let fast = RtenRecognizer::load(path).with_context(|| format!("loading {}", path.display()))?;
        scanner.fast_recognizer = Some(Box::new(fast));
    }
    if let Some(dir) = &args.dump {
        std::fs::create_dir_all(dir)?;
        scanner.params.dump_dir = Some(dir.clone());
    }
    eprintln!("loaded {} names and the OCR model in {} ms", scanner.index.len(), t.elapsed().as_millis());

    let reps = if args.mode == Mode::Bench { args.reps.max(1) } else { 1 };
    let mut timings: Vec<Timing> = Vec::new();
    let mut cpu: Vec<f64> = Vec::new();
    for rep in 0..reps {
        for path in &args.images {
            let c0 = cpu::process_cpu_ms();
            let (result, json) = run_one(&scanner, path)?;
            cpu.push(cpu::process_cpu_ms() - c0);
            if let Some(out) = &args.out {
                std::fs::write(out.join(format!("{}.json", result.image)), json)?;
            }
            if args.mode == Mode::Scan || rep == 0 {
                print_line(&result);
            }
            timings.push(result.timing_ms);
        }
    }
    if args.mode == Mode::Bench {
        print_table(&timings, reps);
        let (mean, p50, p95) = stats(&mut cpu);
        println!("
Process CPU time per photo (all threads, insensitive to other load): mean {mean:.0} ms, p50 {p50:.0}, p95 {p95:.0}");
    } else {
        let mean = timings.iter().map(|t| t.total).sum::<f64>() / timings.len() as f64;
        println!("{} photos, mean {mean:.0} ms/photo", timings.len());
    }
    Ok(())
}

fn print_line(r: &PhotoResult) {
    let slots: usize = r.columns.iter().map(|c| c.slots.len()).sum();
    let auto = r.columns.iter().flat_map(|c| &c.slots).filter(|s| s.status == SlotStatus::Auto).count();
    let t = &r.timing_ms;
    println!(
        "{}: {:?} {:?} auto {auto}/{slots}  {:.0} ms (decode {:.0}, gray {:.0}, detect {:.0}, crop {:.0}, ocr {:.0}, match {:.0}, retry {:.0}; {} lines)",
        r.image, r.config, r.status, t.total, t.decode, t.gray, t.detect, t.crop, t.ocr, t.match_, t.retry, t.counters.ocr_lines
    );
}

/// Mean, median and 95th percentile of `values`.
fn stats(values: &mut [f64]) -> (f64, f64, f64) {
    values.sort_by(f64::total_cmp);
    let n = values.len();
    let mean = values.iter().sum::<f64>() / n as f64;
    let pct = |q: f64| values[((n - 1) as f64 * q).round() as usize];
    (mean, pct(0.5), pct(0.95))
}

fn print_table(timings: &[Timing], reps: usize) {
    // Each row: label and a function pulling that number out of a Timing.
    // `fn(&Timing) -> f64` is a plain function pointer; the closures below
    // capture nothing, so they coerce to it.
    let wall: [(&str, fn(&Timing) -> f64); 10] = [
        ("decode (JPEG)", |t| t.decode),
        ("grayscale", |t| t.gray),
        ("detect+layout", |t| t.detect),
        ("crops (pass 1)", |t| t.crop),
        ("ocr (both passes)", |t| t.ocr),
        ("match (both passes)", |t| t.match_),
        ("  of which retry pass", |t| t.retry),
        ("json", |t| t.json),
        ("scan (inside core)", |t| t.scan),
        ("TOTAL", |t| t.total),
    ];
    let threads: [(&str, fn(&Timing) -> f64); 11] = [
        ("marker detection", |t| t.stages.detect),
        ("layout/homography", |t| t.stages.layout),
        ("warp", |t| t.stages.warp),
        ("name-line finding", |t| t.stages.line_find),
        ("mana trim", |t| t.stages.trim),
        ("clahe/invert", |t| t.stages.enhance),
        ("ocr prep (resize+tensor)", |t| t.stages.ocr_prep),
        ("ocr inference", |t| t.stages.ocr_infer),
        ("ctc decode", |t| t.stages.ctc),
        ("clean+normalise", |t| t.stages.clean),
        ("matching", |t| t.stages.match_),
    ];
    let counters: [(&str, fn(&Timing) -> f64); 8] = [
        ("slots", |t| t.counters.slots as f64),
        ("ocr calls", |t| t.counters.ocr_calls as f64),
        ("ocr batches", |t| t.counters.ocr_batches as f64),
        ("ocr lines", |t| t.counters.ocr_lines as f64),
        ("  of which fast model", |t| t.counters.fast_lines as f64),
        ("crops per slot", |t| t.counters.crops_per_slot),
        ("retry slots", |t| t.counters.retry_slots as f64),
        ("ocr input px wide", |t| t.counters.ocr_input_px_wide as f64),
    ];
    let total_mean = timings.iter().map(|t| t.total).sum::<f64>() / timings.len() as f64;
    println!("\n{} runs ({} reps). Wall-clock ms per photo:", timings.len(), reps);
    println!("| stage | mean | p50 | p95 | % of total |\n| --- | ---: | ---: | ---: | ---: |");
    let row = |label: &str, f: fn(&Timing) -> f64, pct: bool| {
        let mut v: Vec<f64> = timings.iter().map(f).collect();
        let (mean, p50, p95) = stats(&mut v);
        let share = if pct { format!("{:.1}%", 100.0 * mean / total_mean) } else { String::new() };
        println!("| {label} | {mean:.1} | {p50:.1} | {p95:.1} | {share} |");
    };
    for (label, f) in wall {
        row(label, f, true);
    }
    println!("\nThread time ms per photo (summed over threads; parallel stages can exceed wall time):");
    println!("| stage | mean | p50 | p95 | % of total wall |\n| --- | ---: | ---: | ---: | ---: |");
    for (label, f) in threads {
        row(label, f, true);
    }
    println!("\nCounters per photo:");
    println!("| counter | mean | p50 | p95 | |\n| --- | ---: | ---: | ---: | --- |");
    for (label, f) in counters {
        row(label, f, false);
    }
}
