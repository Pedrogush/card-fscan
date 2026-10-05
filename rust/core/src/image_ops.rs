//! Small image-processing helpers on 8-bit grayscale images: sampling,
//! downscaling, perspective warps, CLAHE and inversion.

use image::{GrayImage, Luma};

use crate::geometry::{Point, Rect};
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
pub fn warp_rect(src: &GrayImage, mm_to_img: &Homography, rect: Rect, ppm: f64) -> GrayImage {
    let w = (rect.width() * ppm).round().max(1.0) as u32;
    let h = (rect.height() * ppm).round().max(1.0) as u32;
    let mut out = GrayImage::new(w, h);
    for v in 0..h {
        let ymm = rect.y0 + (v as f64 + 0.5) / ppm;
        for u in 0..w {
            let xmm = rect.x0 + (u as f64 + 0.5) / ppm;
            let p = mm_to_img.apply(Point::new(xmm, ymm));
            // Output pixel centres map to source positions; the source value
            // for pixel (i, j) sits at (i + 0.5, j + 0.5).
            let val = sample_bilinear(src, p.x - 0.5, p.y - 0.5);
            out.put_pixel(u, v, Luma([val.round() as u8]));
        }
    }
    out
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

    #[test]
    fn warp_identity_scale() {
        // A 4x4 image warped with "1 mm == 1 px" at 1 px/mm is unchanged.
        let img = GrayImage::from_fn(4, 4, |x, y| Luma([(x * 10 + y * 50) as u8]));
        let r = Rect { x0: 0.0, y0: 0.0, x1: 4.0, y1: 4.0 };
        let out = warp_rect(&img, &Homography::identity(), r, 1.0);
        assert_eq!(out, img);
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
