//! The egui UI and the Android-specific I/O (APK assets, external files dir).

use std::ffi::CString;
use std::io::Read;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use eframe::egui;
use winit::platform::android::activity::AndroidApp;

use crate::scan::{self, ScanInputs};

const PACKAGE: &str = "io.github.pedrogush.cardfscan.rs";

/// Starts eframe with our app. Blocks until the activity is destroyed.
pub fn run(android_app: AndroidApp) -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        android_app: Some(android_app.clone()),
        ..Default::default()
    };
    eframe::run_native(
        "card-fscan (rs)",
        options,
        Box::new(move |_cc| Ok(Box::new(FscanApp::new(android_app)))),
    )
}

/// State shared between the UI thread and the background scan thread.
#[derive(Default)]
struct Shared {
    running: bool,
    text: String,
}

struct FscanApp {
    android_app: AndroidApp,
    shared: Arc<Mutex<Shared>>,
}

impl FscanApp {
    fn new(android_app: AndroidApp) -> Self {
        let shared = Shared {
            running: false,
            text: "Press \"Run sample scan\" to scan the sample photo.".to_owned(),
        };
        Self { android_app, shared: Arc::new(Mutex::new(shared)) }
    }

    /// Spawns the scan on a worker thread so the UI keeps rendering.
    fn start_scan(&self, ctx: &egui::Context) {
        {
            let mut s = self.shared.lock().unwrap();
            if s.running {
                return;
            }
            s.running = true;
            s.text = "Loading assets and running the scan...".to_owned();
        }
        let app = self.android_app.clone();
        let shared = self.shared.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let t = Instant::now();
            // catch_unwind: a panic in the pipeline becomes a message, not a crash.
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                load_inputs(&app).and_then(scan::run)
            }));
            let text = match outcome {
                Ok(Ok(report)) => {
                    format!("{report}\nWall clock incl. loading: {} ms", t.elapsed().as_millis())
                }
                Ok(Err(msg)) => format!("ERROR: {msg}"),
                Err(panic) => {
                    let msg = panic
                        .downcast_ref::<String>()
                        .map(String::as_str)
                        .or_else(|| panic.downcast_ref::<&str>().copied())
                        .unwrap_or("unknown panic");
                    format!("ERROR: the scan panicked: {msg}")
                }
            };
            log::info!("scan finished:\n{text}");
            if let Ok(mut s) = shared.lock() {
                s.running = false;
                s.text = text;
            }
            ctx.request_repaint();
        });
    }
}

impl eframe::App for FscanApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        // Keep content clear of the Android status bar (winit has no safe-area API yet).
        egui::Panel::top("status_bar_space").show(ui, |ui| ui.set_height(32.0));
        egui::CentralPanel::default().show(ui, |ui| {
            ui.heading("card-fscan (rs)");
            ui.add_space(8.0);
            let (running, text) = {
                let s = self.shared.lock().unwrap();
                (s.running, s.text.clone())
            };
            ui.horizontal(|ui| {
                let button = egui::Button::new("Run sample scan").min_size(egui::vec2(200.0, 48.0));
                if ui.add_enabled(!running, button).clicked() {
                    self.start_scan(ui.ctx());
                }
                if running {
                    ui.spinner();
                }
            });
            ui.add_space(8.0);
            egui::ScrollArea::both().auto_shrink(false).show(ui, |ui| {
                ui.add(egui::Label::new(egui::RichText::new(text).monospace()).selectable(true));
            });
        });
    }
}

/// Reads a whole file from the APK's `assets/` folder.
fn read_asset(app: &AndroidApp, name: &str) -> Result<Option<Vec<u8>>, String> {
    let cname = CString::new(name).map_err(|e| e.to_string())?;
    let Some(mut asset) = app.asset_manager().open(&cname) else {
        return Ok(None);
    };
    let mut buf = Vec::new();
    asset.read_to_end(&mut buf).map_err(|e| format!("reading asset {name}: {e}"))?;
    Ok(Some(buf))
}

fn require_asset(app: &AndroidApp, name: &str) -> Result<Vec<u8>, String> {
    read_asset(app, name)?.ok_or_else(|| format!("asset {name} is missing from the APK"))
}

/// Photo source: asset `sample.jpg`, else the first `.jpg` in the app's
/// external files dir, else an explanatory error.
fn load_photo(app: &AndroidApp) -> Result<(Vec<u8>, String), String> {
    if let Some(bytes) = read_asset(app, "sample.jpg")? {
        return Ok((bytes, "sample.jpg".to_owned()));
    }
    let dir = app
        .external_data_path()
        .unwrap_or_else(|| PathBuf::from(format!("/sdcard/Android/data/{PACKAGE}/files")));
    let mut jpgs: Vec<PathBuf> = std::fs::read_dir(&dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| {
                    p.extension()
                        .and_then(|x| x.to_str())
                        .is_some_and(|x| x.eq_ignore_ascii_case("jpg") || x.eq_ignore_ascii_case("jpeg"))
                })
                .collect()
        })
        .unwrap_or_default();
    jpgs.sort();
    let Some(path) = jpgs.into_iter().next() else {
        return Err(format!(
            "no sample photo: push one with\n  adb push photo.jpg {}/\n(or rebuild the APK with a sample.jpg asset)",
            dir.display()
        ));
    };
    let bytes = std::fs::read(&path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    Ok((bytes, name))
}

fn load_inputs(app: &AndroidApp) -> Result<ScanInputs, String> {
    let (photo, photo_name) = load_photo(app)?;
    let names_gz = require_asset(app, "names_v1.json.gz")?;
    let model = require_asset(app, "rec.onnx")?;
    let dict = String::from_utf8(require_asset(app, "rec.dict.txt")?)
        .map_err(|e| format!("rec.dict.txt is not UTF-8: {e}"))?;
    // The fast first-pass model is optional: without it the app still works,
    // just slower.
    let fast = match (read_asset(app, "fast.onnx")?, read_asset(app, "fast.dict.txt")?) {
        (Some(m), Some(d)) => {
            Some((m, String::from_utf8(d).map_err(|e| format!("fast.dict.txt is not UTF-8: {e}"))?))
        }
        _ => None,
    };
    Ok(ScanInputs { names_gz, model, dict, fast, photo, photo_name })
}
