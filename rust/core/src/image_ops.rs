//! Small image-processing helpers on 8-bit grayscale images: sampling,
//! downscaling, perspective warps, CLAHE and inversion.

use image::{GrayImage, Luma};

use crate::geometry::Rect;
use crate::homography::Homography;

/// Bilinear sample at a sub-pixel position (pixel centres at integer + 0.5
/// are NOT used here: pixel (x, y) covers x..x+1, its value sits at x, y).
/// Positions outside the image are clamped to the edge.
pub fn sample_bilinear(img: &GrayImage, x: f64, y: f64) -> f64 {
    let (w, h) = img.dimensions();
    let x = x.clamp(0.0, (w - 1) as f64);
    let y = y.clamp(0.0, (h - 1) as f64);
    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let fx = x - x0 as f64;
    let fy = y - y0 as f64;
    // `img.as_raw()` is the row-major pixel buffer; indexing it directly is
    // much faster than `get_pixel` in a hot loop.
    let raw = img.as_raw();
    let at = |xx: u32, yy: u32| raw[(yy * w + xx) as usize] as f64;
    let top = at(x0, y0) * (1.0 - fx) + at(x1, y0) * fx;
    let bot = at(x0, y1) * (1.0 - fx) + at(x1, y1) * fx;
    top * (1.0 - fy) + bot * fy
}

/// Shrink by an integer factor, averaging each `f x f` block.
pub fn downscale(img: &GrayImage, f: u32) -> GrayImage {
    if f <= 1 {
        return img.clone();
    }
    let (w, h) = img.dimensions();
    let (sw, sh) = (w / f, h / f);
    let raw = img.as_raw();
    let mut out = GrayImage::new(sw, sh);
    let area = f * f;
    for sy in 0..sh {
        for sx in 0..sw {
            let mut sum = 0u32;
            for dy in 0..f {
                let row = ((sy * f + dy) * w + sx * f) as usize;
                sum += raw[row..row + f as usize].iter().map(|&v| v as u32).sum::<u32>();
            }
            out.put_pixel(sx, sy, Luma([(sum / area) as u8]));
        }
    }
    out
}

/// Summed-area table with one extra row/column of zeros, so the sum over
/// `x0..x1, y0..y1` is `s[y1][x1] - s[y0][x1] - s[y1][x0] + s[y0][x0]`.
struct Integral {
    w: usize,
    sums: Vec<u64>,
}

impl Integral {
    fn new(img: &GrayImage) -> Self {
        let (w, h) = (img.width() as usize, img.height() as usize);
        let stride = w + 1;
        let mut sums = vec![0u64; stride * (h + 1)];
        let raw = img.as_raw();
        for y in 0..h {
            let mut row_sum = 0u64;
            for x in 0..w {
                row_sum += raw[y * w + x] as u64;
                sums[(y + 1) * stride + x + 1] = sums[y * stride + x + 1] + row_sum;
            }
        }
        Integral { w: stride, sums }
    }

    fn sum(&self, x0: usize, y0: usize, x1: usize, y1: usize) -> u64 {
        let s = &self.sums;
        s[y1 * self.w + x1] + s[y0 * self.w + x0] - s[y0 * self.w + x1] - s[y1 * self.w + x0]
    }
}

/// Adaptive "mean - c" threshold (like OpenCV's ADAPTIVE_THRESH_MEAN_C with
/// THRESH_BINARY_INV): output is 255 where the pixel is darker than the mean
/// of its `(2r+1)^2` neighbourhood by more than `c`, else 0.
pub fn adaptive_threshold_inv(img: &GrayImage, r: u32, c: f64) -> GrayImage {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let integral = Integral::new(img);
    let raw = img.as_raw();
    let r = r as usize;
    let mut out = GrayImage::new(w as u32, h as u32);
    let out_raw: &mut [u8] = &mut out;
    for y in 0..h {
        let (y0, y1) = (y.saturating_sub(r), (y + r + 1).min(h));
        for x in 0..w {
            let (x0, x1) = (x.saturating_sub(r), (x + r + 1).min(w));
            let n = ((x1 - x0) * (y1 - y0)) as f64;
            let mean = integral.sum(x0, y0, x1, y1) as f64 / n;
            if (raw[y * w + x] as f64) < mean - c {
                out_raw[y * w + x] = 255;
            }
        }
    }
    out
}

/// Resample the strip-mm rectangle `rect` into a new image at `ppm` pixels per
/// mm. `mm_to_img` maps strip millimetres to source-image pixels.
///
/// This is the hottest preprocessing loop (~100k output pixels per slot), so
/// it avoids per-pixel overhead: along one output row the homogeneous source
/// coordinates `(X, Y, W)` are linear in the column index, so they are
/// computed as `start + u * step` (no 3x3 matrix product per pixel), the
/// output is written straight into a `Vec<u8>` (no bounds-checked
/// `put_pixel`), and bilinear interpolation reads the raw source buffer.
pub fn warp_rect(src: &GrayImage, mm_to_img: &Homography, rect: Rect, ppm: f64) -> GrayImage {
    let w = (rect.width() * ppm).round().max(1.0) as u32;
    let h = (rect.height() * ppm).round().max(1.0) as u32;
    let (sw, sh) = src.dimensions();
    let raw = src.as_raw();
    let m = &mm_to_img.m;
    let (max_x, max_y) = ((sw - 1) as f64, (sh - 1) as f64);
    let mut out = vec![0u8; (w * h) as usize];
    // `chunks_exact_mut` hands out one output row at a time as a mutable slice.
    for (v, row) in out.chunks_exact_mut(w as usize).enumerate() {
        let ymm = rect.y0 + (v as f64 + 0.5) / ppm;
        let xmm0 = rect.x0 + 0.5 / ppm;
        // Homogeneous source coordinates at u = 0, and their change per column.
        let start = [
            m[(0, 0)] * xmm0 + m[(0, 1)] * ymm + m[(0, 2)],
            m[(1, 0)] * xmm0 + m[(1, 1)] * ymm + m[(1, 2)],
            m[(2, 0)] * xmm0 + m[(2, 1)] * ymm + m[(2, 2)],
        ];
        let step = [m[(0, 0)] / ppm, m[(1, 0)] / ppm, m[(2, 0)] / ppm];
        for (u, px) in row.iter_mut().enumerate() {
            let u = u as f64;
            let inv_w = 1.0 / (start[2] + u * step[2]);
            // Source value for pixel (i, j) sits at (i + 0.5, j + 0.5).
            let x = ((start[0] + u * step[0]) * inv_w - 0.5).clamp(0.0, max_x);
            let y = ((start[1] + u * step[1]) * inv_w - 0.5).clamp(0.0, max_y);
            let (x0, y0) = (x as u32, y as u32); // floor: x, y >= 0
            let (x1, y1) = ((x0 + 1).min(sw - 1), (y0 + 1).min(sh - 1));
            let (fx, fy) = ((x - x0 as f64) as f32, (y - y0 as f64) as f32);
            let at = |xx: u32, yy: u32| raw[(yy * sw + xx) as usize] as f32;
            let top = at(x0, y0) + (at(x1, y0) - at(x0, y0)) * fx;
            let bot = at(x0, y1) + (at(x1, y1) - at(x0, y1)) * fx;
            *px = (top + (bot - top) * fy + 0.5) as u8;
        }
    }
    GrayImage::from_raw(w, h, out).expect("buffer has w * h bytes")
}

pub fn invert(img: &GrayImage) -> GrayImage {
    let mut out = img.clone();
    out.iter_mut().for_each(|v| *v = 255 - *v);
    out
}

/// Contrast Limited Adaptive Histogram Equalisation (as in OpenCV's
/// `createCLAHE`): equalise each tile's histogram with its peaks clipped at
/// `clip_limit` times the mean bin height, then blend neighbouring tiles'
/// lookup tables bilinearly so tile borders don't show.
pub fn clahe(img: &GrayImage, tiles_x: u32, tiles_y: u32, clip_limit: f64) -> GrayImage {
    let (w, h) = img.dimensions();
    let tiles_x = tiles_x.clamp(1, w.max(1));
    let tiles_y = tiles_y.clamp(1, h.max(1));
    let tw = w.div_ceil(tiles_x);
    let th = h.div_ceil(tiles_y);
    let raw = img.as_raw();

    // One 256-entry lookup table per tile.
    let mut luts = vec![[0u8; 256]; (tiles_x * tiles_y) as usize];
    for ty in 0..tiles_y {
        for tx in 0..tiles_x {
            let (x0, y0) = (tx * tw, ty * th);
            let (x1, y1) = ((x0 + tw).min(w), (y0 + th).min(h));
            let mut hist = [0u32; 256];
            for y in y0..y1 {
                for x in x0..x1 {
                    hist[raw[(y * w + x) as usize] as usize] += 1;
                }
            }
            let n = ((x1 - x0) * (y1 - y0)).max(1);
            let clip = ((clip_limit * n as f64 / 256.0) as u32).max(1);
            let mut excess = 0u32;
            for b in hist.iter_mut() {
                if *b > clip {
                    excess += *b - clip;
                    *b = clip;
                }
            }
            let bonus = excess / 256;
            let mut rest = excess % 256;
            let lut = &mut luts[(ty * tiles_x + tx) as usize];
            let mut cdf = 0u32;
            for (i, b) in hist.iter().enumerate() {
                let mut v = b + bonus;
                if rest > 0 {
                    v += 1;
                    rest -= 1;
                }
                cdf += v;
                lut[i] = ((cdf as f64 * 255.0 / n as f64).round()).min(255.0) as u8;
            }
        }
    }

    let mut out = GrayImage::new(w, h);
    for y in 0..h {
        // Position relative to tile centres, clamped at the image border.
        let fy = ((y as f64 + 0.5) / th as f64 - 0.5).clamp(0.0, (tiles_y - 1) as f64);
        let ty0 = fy.floor() as u32;
        let ty1 = (ty0 + 1).min(tiles_y - 1);
        let ay = fy - ty0 as f64;
        for x in 0..w {
            let fx = ((x as f64 + 0.5) / tw as f64 - 0.5).clamp(0.0, (tiles_x - 1) as f64);
            let tx0 = fx.floor() as u32;
            let tx1 = (tx0 + 1).min(tiles_x - 1);
            let ax = fx - tx0 as f64;
            let v = raw[(y * w + x) as usize] as usize;
            let l = |tx: u32, ty: u32| luts[(ty * tiles_x + tx) as usize][v] as f64;
            let top = l(tx0, ty0) * (1.0 - ax) + l(tx1, ty0) * ax;
            let bot = l(tx0, ty1) * (1.0 - ax) + l(tx1, ty1) * ax;
            out.put_pixel(x, y, Luma([(top * (1.0 - ay) + bot * ay).round() as u8]));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Point;

    #[test]
    fn warp_identity_scale() {
        // A 4x4 image warped with "1 mm == 1 px" at 1 px/mm is unchanged.
        let img = GrayImage::from_fn(4, 4, |x, y| Luma([(x * 10 + y * 50) as u8]));
        let r = Rect { x0: 0.0, y0: 0.0, x1: 4.0, y1: 4.0 };
        let out = warp_rect(&img, &Homography::identity(), r, 1.0);
        assert_eq!(out, img);
    }

    #[test]
    fn warp_matches_reference_sampling() {
        // Compare against the straightforward per-pixel formula on a
        // perspective map.
        let img = GrayImage::from_fn(200, 150, |x, y| Luma([((x * 7 + y * 13) % 256) as u8]));
        let h = Homography { m: nalgebra::Matrix3::new(3.1, 0.2, 10.0, -0.1, 2.9, 5.0, 1e-4, 2e-4, 1.0) };
        let r = Rect { x0: 1.0, y0: 2.0, x1: 50.0, y1: 40.0 };
        let fast = warp_rect(&img, &h, r, 1.5);
        for (u, v, px) in fast.enumerate_pixels() {
            let p = h.apply(Point::new(r.x0 + (u as f64 + 0.5) / 1.5, r.y0 + (v as f64 + 0.5) / 1.5));
            let want = sample_bilinear(&img, p.x - 0.5, p.y - 0.5);
            assert!((px[0] as f64 - want).abs() <= 1.0, "({u},{v}): {} vs {want}", px[0]);
        }
    }

    #[test]
    fn threshold_marks_dark_spot() {
        let mut img = GrayImage::from_pixel(9, 9, Luma([200]));
        img.put_pixel(4, 4, Luma([20]));
        let t = adaptive_threshold_inv(&img, 2, 7.0);
        assert_eq!(t.get_pixel(4, 4)[0], 255);
        assert_eq!(t.get_pixel(0, 0)[0], 0);
    }

    #[test]
    fn clahe_stretches_low_contrast() {
        let img = GrayImage::from_fn(64, 16, |x, _| Luma([100 + (x / 8) as u8]));
        let out = clahe(&img, 2, 1, 40.0);
        let (lo, hi) = out.iter().fold((255u8, 0u8), |(lo, hi), &v| (lo.min(v), hi.max(v)));
        assert!(hi - lo > 100, "range {lo}..{hi}");
    }
}
