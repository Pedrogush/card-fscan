//! Process CPU time (user + kernel), to measure *work* independently of how
//! busy the machine is: when other programs compete for the cores, wall-clock
//! time inflates but CPU time barely moves.

/// CPU time used by this process so far, in milliseconds.
#[cfg(windows)]
pub fn process_cpu_ms() -> f64 {
    // A minimal FFI declaration of two kernel32 functions (std already links
    // kernel32, so no extra crate is needed). `unsafe` is required because the
    // compiler cannot check foreign code; we pass valid pointers to
    // stack-allocated structs, which is all GetProcessTimes needs.
    #[repr(C)]
    #[derive(Default)]
    struct FileTime {
        low: u32,
        high: u32,
    }
    unsafe extern "system" {
        fn GetCurrentProcess() -> isize;
        fn GetProcessTimes(h: isize, c: *mut FileTime, e: *mut FileTime, k: *mut FileTime, u: *mut FileTime) -> i32;
    }
    let (mut c, mut e, mut k, mut u) = Default::default();
    let ok = unsafe { GetProcessTimes(GetCurrentProcess(), &mut c, &mut e, &mut k, &mut u) };
    if ok == 0 {
        return 0.0;
    }
    // FILETIME counts 100 ns ticks.
    let ticks = |t: &FileTime| ((t.high as u64) << 32 | t.low as u64) as f64;
    (ticks(&k) + ticks(&u)) / 1e4
}

/// CPU time used by this process so far, in milliseconds (Linux/Android:
/// fields 14 and 15 of `/proc/self/stat`, in clock ticks of 10 ms).
#[cfg(not(windows))]
pub fn process_cpu_ms() -> f64 {
    let Ok(stat) = std::fs::read_to_string("/proc/self/stat") else { return 0.0 };
    // The command name (field 2) may contain spaces; skip past its ')'.
    let rest = stat.rsplit_once(')').map_or("", |(_, r)| r);
    let fields: Vec<&str> = rest.split_whitespace().collect();
    let tick = |i: usize| fields.get(i).and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0);
    // After ')', field 3 (state) has index 0, so utime (14) is 11, stime (15) is 12.
    (tick(11) + tick(12)) * 10.0
}
