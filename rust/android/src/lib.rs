//! card-fscan Android app (Rust track).
//!
//! On Android this crate is built as `libfscan_android.so`. The stock
//! Android `NativeActivity` (declared in `AndroidManifest.xml`, no Java of
//! our own) loads that library and the `android-activity` crate (via winit)
//! calls our `android_main` entry point, which starts an egui UI.
//!
//! On any other target only [`scan`] is compiled, so `cargo check -p
//! fscan-android` / `cargo test -p fscan-android` work on the host.

pub mod scan;

#[cfg(target_os = "android")]
mod app;

/// Entry point called by `android-activity` on its own thread once the
/// NativeActivity has started. `#[no_mangle]` keeps the symbol name exactly
/// `android_main` so the glue code can find it.
#[cfg(target_os = "android")]
#[unsafe(no_mangle)]
fn android_main(android_app: winit::platform::android::activity::AndroidApp) {
    android_logger::init_once(
        android_logger::Config::default()
            .with_max_level(log::LevelFilter::Info)
            .with_tag("fscan"),
    );
    if let Err(e) = app::run(android_app) {
        log::error!("eframe exited with error: {e}");
    }
}
