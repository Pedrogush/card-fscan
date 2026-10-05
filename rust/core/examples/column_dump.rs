//! Debug helper: write each readable column of a photo, rectified to strip
//! millimetres at 6 px/mm (0..70 x 0..330 mm), to `<out>/<stem>_col<j>.png`,
//! with a grey tick at every nominal card top edge.
//!
//! `cargo run --release -p fscan-core --example column_dump -- <out dir> <photo>...`

use std::path::PathBuf;

use fscan_core::geometry::{HEADER_Y, Rect, SLOT_PITCH};
use fscan_core::image_ops::warp_rect;
use fscan_core::matching::NameIndex;
use fscan_core::ocr::{OcrLine, Recognizer};
use fscan_core::{Error, Scanner};

/// `locate` needs a Scanner, which needs a recogniser; this one never runs.
struct NoOcr;
impl Recognizer for NoOcr {
    fn recognize(&self, lines: &[image::GrayImage]) -> Result<Vec<OcrLine>, Error> {
        Ok(lines.iter().map(|_| OcrLine { text: String::new(), confidence: 0.0 }).collect())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let out = PathBuf::from(args.next().ok_or("out dir")?);
    std::fs::create_dir_all(&out)?;
    let index = NameIndex::from_bytes(br#"{"version":1,"entries":[]}"#, false)?;
    let scanner = Scanner::new(index, Box::new(NoOcr));
    let ppm = 6.0;
    for photo_path in args {
        let photo = image::open(&photo_path)?.to_luma8();
        let layout = scanner.locate(&photo);
        let stem = PathBuf::from(&photo_path).file_stem().unwrap().to_string_lossy().to_string();
        for c in &layout.columns {
            let rect = Rect { x0: 0.0, y0: 0.0, x1: 70.0, y1: 330.0 };
            let mut img = warp_rect(&photo, &c.geometry.mm_to_img, rect, ppm);
            for i in 0..20 {
                let y = ((HEADER_Y + SLOT_PITCH * i as f64) * ppm) as u32;
                for x in 0..12 {
                    img.put_pixel(x, y, image::Luma([128]));
                }
            }
            img.save(out.join(format!("{stem}_col{}.png", c.geometry.column)))?;
        }
        println!("{stem}: {:?} {:?}, {} columns", layout.config, layout.status, layout.columns.len());
    }
    Ok(())
}
