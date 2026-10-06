//! Matching micro-benchmark on real OCR output:
//! `cargo run --release -p fscan-core --example match_bench -- <results dir>`
//! Times the flat-layout and the reference search over every `raw_text` found in
//! the result JSONs, and checks they agree.

use std::time::Instant;

use fscan_core::matching::{NameIndex, clean, normalize};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::args().nth(1).ok_or("results dir")?;
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let index = NameIndex::load(&root.join("testdata/names/names_v1.json"))?;
    let mut keys = Vec::new();
    for entry in std::fs::read_dir(&dir)? {
        let path = entry?.path();
        if path.extension().is_some_and(|e| e == "json") {
            let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path)?)?;
            for col in v["columns"].as_array().into_iter().flatten() {
                for slot in col["slots"].as_array().into_iter().flatten() {
                    let key = normalize(&clean(slot["raw_text"].as_str().unwrap_or("")));
                    if !key.is_empty() {
                        keys.push(key);
                    }
                }
            }
        }
    }
    let t = Instant::now();
    let fast: Vec<_> = keys.iter().map(|k| index.top_candidates(k, 3)).collect();
    let fast_ms = t.elapsed().as_secs_f64() * 1e3;
    let t = Instant::now();
    let slow: Vec<_> = keys.iter().map(|k| index.top_candidates_exhaustive(k, 3)).collect();
    let slow_ms = t.elapsed().as_secs_f64() * 1e3;
    assert!(fast == slow, "flat and reference search disagree");
    println!(
        "{} queries: flat layout {:.3} ms/query, per-String reference {:.3} ms/query (single thread)",
        keys.len(),
        fast_ms / keys.len() as f64,
        slow_ms / keys.len() as f64
    );
    Ok(())
}
