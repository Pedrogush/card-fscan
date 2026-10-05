//! A focused detector for OpenCV's ArUco `DICT_4X4_50` markers.
//!
//! No mature pure-Rust ArUco crate existed, so this follows the classic
//! OpenCV recipe, trimmed to what card-fscan needs:
//!
//! 1. Downscale the photo (markers are large) and run an adaptive threshold.
//! 2. Trace connected outlines (`imageproc::contours`) and keep the ones that
//!    are convex quadrilaterals of a plausible size.
//! 3. For each quad, sample the 6x6 cell grid on the full-resolution photo
//!    through a perspective transform, binarise the cells, require a black
//!    border and look the 4x4 payload up in the dictionary (4 rotations).
//! 4. Refine the four corners by fitting straight lines to the marker's edges
//!    at full resolution and intersecting them (sub-pixel accuracy).

use image::GrayImage;
use imageproc::contours::{BorderType, find_contours};

use crate::geometry::Point;
use crate::homography::Homography;
use crate::image_ops::{adaptive_threshold_inv, downscale, sample_bilinear};

/// The 50 markers of OpenCV's `DICT_4X4_50`, as 16-bit row-major payloads
/// (bit 15 = top-left cell, 1 = white), in the marker's canonical rotation.
///
/// Extracted with `cv2.aruco.generateImageMarker` from OpenCV
/// (Apache-2.0, `modules/objdetect/src/aruco/predefined_dictionaries.hpp`).
pub const DICT_4X4_50: [u16; 50] = [
    0xB532, 0x0F9A, 0x332D, 0x9946, 0x549E, 0x79CD, 0x9E2E, 0xC4F2, 0xFEDA, 0xCF56, //
    0xF991, 0x11A7, 0x0EB7, 0x2A0F, 0x24B1, 0x263E, 0x4665, 0x6600, 0x6C5E, 0x76AF, //
    0x868B, 0xB02B, 0xCCD5, 0xDD82, 0xFE47, 0x9471, 0xACE4, 0xA554, 0x2123, 0x346F, //
    0x4415, 0x57B2, 0x9ECF, 0xF0CB, 0x08AE, 0x0929, 0x1875, 0x04FF, 0x0DF6, 0x1C5A, //
    0x1718, 0x2A28, 0x328C, 0x38B2, 0x24E8, 0x2EEB, 0x2D3F, 0x4B64, 0x502E, 0x5013,
];

/// Marker grid size in cells: 4 payload cells plus a 1-cell black border.
const GRID: usize = 6;

/// A detected marker. Corners follow OpenCV: top-left, top-right,
/// bottom-right, bottom-left of the marker as printed, in image pixels.
#[derive(Debug, Clone, PartialEq)]
pub struct Marker {
    pub id: u16,
    pub corners: [Point; 4],
}

impl Marker {
    pub fn center(&self) -> Point {
        let (sx, sy) = self.corners.iter().fold((0.0, 0.0), |(x, y), p| (x + p.x, y + p.y));
        Point::new(sx / 4.0, sy / 4.0)
    }

    /// Mean side length in pixels.
    pub fn side(&self) -> f64 {
        (0..4).map(|i| self.corners[i].dist(self.corners[(i + 1) % 4])).sum::<f64>() / 4.0
    }
}

/// Tunables. `Default` gives values that work on 12-50 MP photos.
#[derive(Debug, Clone)]
pub struct DetectorParams {
    /// The photo is shrunk so its long side is at most this many pixels
    /// before thresholding (decoding still uses the full image).
    pub work_size: u32,
    /// Adaptive threshold window radii (in work-image pixels) to try.
    pub threshold_radii: Vec<u32>,
    pub threshold_c: f64,
    /// Smallest accepted marker side, as a fraction of the work image's
    /// long side.
    pub min_side_rate: f64,
    /// Max wrong cells allowed in the 20-cell black border.
    pub max_border_errors: u32,
    /// Max payload bit errors to correct (OpenCV's default for this
    /// dictionary is 0).
    pub max_payload_errors: u32,
}

impl Default for DetectorParams {
    fn default() -> Self {
        DetectorParams {
            work_size: 1600,
            threshold_radii: vec![7, 15],
            threshold_c: 7.0,
            min_side_rate: 0.015,
            max_border_errors: 2,
            max_payload_errors: 0,
        }
    }
}

/// Detect all `DICT_4X4_50` markers in a grayscale photo.
pub fn detect(img: &GrayImage, params: &DetectorParams) -> Vec<Marker> {
    let long = img.width().max(img.height());
    let factor = long.div_ceil(params.work_size).max(1);
    let small = downscale(img, factor);
    let scale = factor as f64;
    let min_side = params.min_side_rate * small.width().max(small.height()) as f64;

    let mut found: Vec<(Marker, u32)> = Vec::new();
    for &r in &params.threshold_radii {
        let bin = adaptive_threshold_inv(&small, r, params.threshold_c);
        for contour in find_contours::<i32>(&bin) {
            if contour.border_type != BorderType::Outer || contour.points.len() < (4.0 * min_side) as usize {
                continue;
            }
            let pts: Vec<Point> = contour.points.iter().map(|p| Point::new(p.x as f64, p.y as f64)).collect();
            let Some(quad) = fit_quad(&pts, min_side) else { continue };
            // Contour pixels are the dark border's outermost pixels; the
            // marker's true edge is half a pixel further out. Map pixel
            // positions to full-res coordinates (centre of each block).
            let quad_full = quad.map(|p| Point::new((p.x + 0.5) * scale, (p.y + 0.5) * scale));
            let quad_full = expand(&quad_full, 0.5 * scale);
            let Some((marker, errors)) = decode(img, &quad_full, params) else { continue };
            // The same marker is usually found by several thresholds; keep the
            // cleanest reading.
            match found.iter_mut().find(|(m, _)| m.id == marker.id && m.center().dist(marker.center()) < 0.5 * m.side()) {
                Some(existing) if existing.1 <= errors => {}
                Some(existing) => *existing = (marker, errors),
                None => found.push((marker, errors)),
            }
        }
    }
    found
        .into_iter()
        .map(|(m, _)| Marker { corners: refine_corners(img, &m.corners), ..m })
        .collect()
}

/// Approximate a closed outline by a convex quadrilateral, or `None` when it
/// isn't one. Corners are returned clockwise on screen (y down).
fn fit_quad(pts: &[Point], min_side: f64) -> Option<[Point; 4]> {
    let n = pts.len() as f64;
    let c = pts.iter().fold(Point::default(), |a, p| Point::new(a.x + p.x / n, a.y + p.y / n));
    let far = |from: Point| pts.iter().copied().max_by(|a, b| a.dist(from).total_cmp(&b.dist(from)));
    // Two opposite corners: farthest from the centroid, then farthest from it.
    let a = far(c)?;
    let b = far(a)?;
    // The other two: farthest on either side of the diagonal a-b.
    let signed = |p: Point| (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
    let c1 = pts.iter().copied().max_by(|p, q| signed(*p).total_cmp(&signed(*q)))?;
    let c2 = pts.iter().copied().min_by(|p, q| signed(*p).total_cmp(&signed(*q)))?;
    if signed(c1) <= 0.0 || signed(c2) >= 0.0 {
        return None;
    }
    // a -> c1 -> b -> c2 goes around the quad; make it clockwise on screen.
    let mut quad = [a, c1, b, c2];
    if cross(quad[0], quad[1], quad[2]) < 0.0 {
        quad.swap(1, 3);
    }
    // Convex with sides of reasonable, similar length.
    let sides: Vec<f64> = (0..4).map(|i| quad[i].dist(quad[(i + 1) % 4])).collect();
    let (smin, smax) = sides.iter().fold((f64::MAX, 0.0f64), |(lo, hi), &s| (lo.min(s), hi.max(s)));
    if smin < min_side || smax > 4.0 * smin {
        return None;
    }
    if (0..4).any(|i| cross(quad[i], quad[(i + 1) % 4], quad[(i + 2) % 4]) <= 0.0) {
        return None;
    }
    // Every outline point must lie close to one of the four edges.
    let tol = (0.06 * smin).max(1.5);
    let near_edge = |p: &Point| (0..4).any(|i| dist_to_segment(*p, quad[i], quad[(i + 1) % 4]) <= tol);
    if !pts.iter().all(near_edge) {
        return None;
    }
    Some(quad)
}

fn cross(a: Point, b: Point, c: Point) -> f64 {
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
}

fn dist_to_segment(p: Point, a: Point, b: Point) -> f64 {
    let (dx, dy) = (b.x - a.x, b.y - a.y);
    let len2 = dx * dx + dy * dy;
    let t = if len2 > 0.0 { (((p.x - a.x) * dx + (p.y - a.y) * dy) / len2).clamp(0.0, 1.0) } else { 0.0 };
    p.dist(Point::new(a.x + t * dx, a.y + t * dy))
}

/// Push each corner `d` pixels outward from the quad's centre.
fn expand(q: &[Point; 4], d: f64) -> [Point; 4] {
    let c = Point::new(q.iter().map(|p| p.x).sum::<f64>() / 4.0, q.iter().map(|p| p.y).sum::<f64>() / 4.0);
    q.map(|p| {
        let len = p.dist(c).max(1e-9);
        // Corners are ~sqrt(2) further out than edge midpoints.
        let k = d * std::f64::consts::SQRT_2 / len;
        Point::new(p.x + (p.x - c.x) * k, p.y + (p.y - c.y) * k)
    })
}

/// Sample the 6x6 grid inside `quad` (clockwise, any starting corner) and
/// try to read it as a dictionary marker. Returns the marker with its corners
/// re-ordered to the printed orientation, plus the number of corrected cells.
fn decode(img: &GrayImage, quad: &[Point; 4], params: &DetectorParams) -> Option<(Marker, u32)> {
    let g = GRID as f64;
    let unit = [Point::new(0.0, 0.0), Point::new(g, 0.0), Point::new(g, g), Point::new(0.0, g)];
    let h = Homography::from_points(&unit, quad)?;

    // Mean intensity of the central part of each cell (a 4x4 sub-grid of
    // samples covering the middle 60%, away from blurry cell borders).
    let mut cells = [[0.0f64; GRID]; GRID];
    for (r, row) in cells.iter_mut().enumerate() {
        for (c, cell) in row.iter_mut().enumerate() {
            let mut sum = 0.0;
            for i in 0..4 {
                for j in 0..4 {
                    let p = h.apply(Point::new(c as f64 + 0.2 + 0.2 * j as f64, r as f64 + 0.2 + 0.2 * i as f64));
                    sum += sample_bilinear(img, p.x - 0.5, p.y - 0.5);
                }
            }
            *cell = sum / 16.0;
        }
    }
    let threshold = two_means_threshold(cells.iter().flatten().copied())?;

    let mut border_errors = 0;
    let mut bits = [[false; GRID]; GRID];
    for r in 0..GRID {
        for c in 0..GRID {
            bits[r][c] = cells[r][c] > threshold;
            let on_border = r == 0 || c == 0 || r == GRID - 1 || c == GRID - 1;
            if on_border && bits[r][c] {
                border_errors += 1;
            }
        }
    }
    if border_errors > params.max_border_errors {
        return None;
    }

    // Try the 4 possible starting corners. Starting at corner k of the quad
    // is the same as rotating the sampled grid; see `rotate`.
    let mut grid = bits;
    for k in 0..4 {
        let code = payload(&grid);
        if let Some((id, dist)) = lookup(code, params.max_payload_errors) {
            let corners = [quad[k % 4], quad[(k + 1) % 4], quad[(k + 2) % 4], quad[(k + 3) % 4]];
            return Some((Marker { id, corners }, border_errors + dist));
        }
        grid = rotate(&grid);
    }
    None
}

/// The grid as seen when the next corner (clockwise) is taken as top-left:
/// new cell (r, c) is old cell (c, N-1-r).
fn rotate(g: &[[bool; GRID]; GRID]) -> [[bool; GRID]; GRID] {
    let mut out = [[false; GRID]; GRID];
    for (r, row) in out.iter_mut().enumerate() {
        for (c, v) in row.iter_mut().enumerate() {
            *v = g[c][GRID - 1 - r];
        }
    }
    out
}

/// The inner 4x4 cells as a 16-bit number, row-major, MSB first.
fn payload(g: &[[bool; GRID]; GRID]) -> u16 {
    let mut v = 0u16;
    for row in &g[1..GRID - 1] {
        for &bit in &row[1..GRID - 1] {
            v = (v << 1) | bit as u16;
        }
    }
    v
}

/// Dictionary lookup allowing up to `max_errors` flipped bits; returns the
/// id and the Hamming distance. Ambiguous corrections are rejected.
fn lookup(code: u16, max_errors: u32) -> Option<(u16, u32)> {
    let mut best: Option<(u16, u32)> = None;
    let mut ambiguous = false;
    for (id, &word) in DICT_4X4_50.iter().enumerate() {
        let d = (word ^ code).count_ones();
        if d > max_errors {
            continue;
        }
        match best {
            Some((_, bd)) if bd < d => {}
            Some((_, bd)) if bd == d => ambiguous = true,
            _ => {
                best = Some((id as u16, d));
                ambiguous = false;
            }
        }
    }
    if ambiguous { None } else { best }
}

/// Threshold separating dark and light cells (1-D 2-means). `None` when the
/// cells have too little contrast to be a marker.
fn two_means_threshold(values: impl Iterator<Item = f64> + Clone) -> Option<f64> {
    let (lo, hi) = values.clone().fold((f64::MAX, f64::MIN), |(lo, hi), v| (lo.min(v), hi.max(v)));
    if hi - lo < 30.0 {
        return None;
    }
    let mut t = (lo + hi) / 2.0;
    for _ in 0..10 {
        let (mut sd, mut nd, mut sl, mut nl) = (0.0, 0, 0.0, 0);
        for v in values.clone() {
            if v > t {
                sl += v;
                nl += 1;
            } else {
                sd += v;
                nd += 1;
            }
        }
        if nd == 0 || nl == 0 {
            break;
        }
        t = (sd / nd as f64 + sl / nl as f64) / 2.0;
    }
    Some(t)
}

/// Sub-pixel corner refinement: fit a line to each of the 4 outer edges
/// (strongest gradient along the edge normal, at many points) and intersect
/// neighbouring lines.
fn refine_corners(img: &GrayImage, q: &[Point; 4]) -> [Point; 4] {
    let side = (0..4).map(|i| q[i].dist(q[(i + 1) % 4])).sum::<f64>() / 4.0;
    let cell = side / GRID as f64;
    let search = (0.35 * cell).max(2.0);
    let mut lines = Vec::with_capacity(4);
    for i in 0..4 {
        let (a, b) = (q[i], q[(i + 1) % 4]);
        let len = a.dist(b);
        let (dx, dy) = ((b.x - a.x) / len, (b.y - a.y) / len);
        // Outward normal for a clockwise (on screen) quad.
        let (nx, ny) = (dy, -dx);
        let mut edge_pts = Vec::new();
        let samples = 24;
        for s in 0..samples {
            // Stay away from the corners, where two edges mix.
            let t = 0.15 + 0.7 * s as f64 / (samples - 1) as f64;
            let base = Point::new(a.x + t * (b.x - a.x), a.y + t * (b.y - a.y));
            let at = |d: f64| sample_bilinear(img, base.x + nx * d - 0.5, base.y + ny * d - 0.5);
            let step = 0.5;
            let n = (2.0 * search / step) as i32;
            // Inside is dark, outside light: look for the largest rise.
            let mut best = (0.0, f64::MIN);
            let mut grads = Vec::with_capacity(n as usize + 1);
            for k in 0..=n {
                let d = -search + k as f64 * step;
                let g = at(d + step) - at(d - step);
                grads.push(g);
                if g > best.1 {
                    best = (d, g);
                }
            }
            if best.1 < 10.0 {
                continue;
            }
            // Parabola through the peak and its neighbours for sub-sample accuracy.
            let k = ((best.0 + search) / step).round() as usize;
            let mut d = best.0;
            if k > 0 && k + 1 < grads.len() {
                let (g0, g1, g2) = (grads[k - 1], grads[k], grads[k + 1]);
                let denom = g0 - 2.0 * g1 + g2;
                if denom.abs() > 1e-9 {
                    d += step * (0.5 * (g0 - g2) / denom).clamp(-1.0, 1.0);
                }
            }
            edge_pts.push(Point::new(base.x + nx * d, base.y + ny * d));
        }
        match fit_line(&edge_pts) {
            Some(l) if edge_pts.len() >= 6 => lines.push(l),
            _ => return *q,
        }
    }
    let mut out = *q;
    for i in 0..4 {
        // Corner i is where edge (i-1) meets edge i.
        match intersect(lines[(i + 3) % 4], lines[i]) {
            Some(p) if p.dist(q[i]) < 0.5 * cell => out[i] = p,
            _ => return *q,
        }
    }
    out
}

/// Total-least-squares line through points: (point on line, unit direction).
fn fit_line(pts: &[Point]) -> Option<(Point, (f64, f64))> {
    if pts.len() < 2 {
        return None;
    }
    let n = pts.len() as f64;
    let cx = pts.iter().map(|p| p.x).sum::<f64>() / n;
    let cy = pts.iter().map(|p| p.y).sum::<f64>() / n;
    let (mut sxx, mut sxy, mut syy) = (0.0, 0.0, 0.0);
    for p in pts {
        let (dx, dy) = (p.x - cx, p.y - cy);
        sxx += dx * dx;
        sxy += dx * dy;
        syy += dy * dy;
    }
    // Direction of the largest eigenvector of the 2x2 covariance matrix.
    let angle = 0.5 * (2.0 * sxy).atan2(sxx - syy);
    Some((Point::new(cx, cy), (angle.cos(), angle.sin())))
}

fn intersect(l1: (Point, (f64, f64)), l2: (Point, (f64, f64))) -> Option<Point> {
    let ((p, d), (q, e)) = (l1, l2);
    let det = d.0 * e.1 - d.1 * e.0;
    if det.abs() < 1e-9 {
        return None;
    }
    let t = ((q.x - p.x) * e.1 - (q.y - p.y) * e.0) / det;
    Some(Point::new(p.x + t * d.0, p.y + t * d.1))
}

/// Render marker `id` as a `cells_px`-per-cell image with a white quiet zone
/// of one cell (used by tests and the Android demo).
pub fn render_marker(id: u16, cells_px: u32) -> GrayImage {
    let code = DICT_4X4_50[id as usize];
    let size = (GRID as u32 + 2) * cells_px;
    GrayImage::from_fn(size, size, |x, y| {
        let (c, r) = (x / cells_px, y / cells_px);
        let white = if (1..=4).contains(&(r as i32 - 1)) && (1..=4).contains(&(c as i32 - 1)) {
            let bit = (r - 2) * 4 + (c - 2);
            code >> (15 - bit) & 1 == 1
        } else {
            !(1..=6).contains(&r) || !(1..=6).contains(&c)
        };
        image::Luma([if white { 255 } else { 0 }])
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;

    #[test]
    fn dictionary_codes_are_distinct_under_rotation() {
        // No marker may equal another marker, or any marker in another
        // orientation (including itself), or ids would be ambiguous.
        let as_grid = |code: u16| {
            let mut g = [[false; GRID]; GRID];
            for i in 0..16 {
                g[1 + i / 4][1 + i % 4] = code >> (15 - i) & 1 == 1;
            }
            g
        };
        for (i, &a) in DICT_4X4_50.iter().enumerate() {
            let mut g = as_grid(a);
            for k in 0..4 {
                for (j, &b) in DICT_4X4_50.iter().enumerate() {
                    if k > 0 || i != j {
                        assert_ne!(payload(&g), b, "marker {i} rotated {k} times equals marker {j}");
                    }
                }
                g = rotate(&g);
            }
        }
    }

    /// Paste markers into a larger image through a known homography and check
    /// that ids and corners come back.
    #[test]
    fn detects_warped_markers() {
        let (w, h) = (1200u32, 900u32);
        let ids = [0u16, 7, 40];
        // Each marker is placed with its own perspective transform.
        let placements: Vec<[Point; 4]> = vec![
            [Point::new(100.3, 120.7), Point::new(330.2, 140.1), Point::new(318.8, 362.9), Point::new(92.5, 350.4)],
            [Point::new(700.0, 400.0), Point::new(900.0, 420.0), Point::new(880.0, 640.0), Point::new(690.0, 610.0)],
            // Rotated by ~90 degrees: printed top-left is at the image's top-right.
            [Point::new(620.0, 100.0), Point::new(630.0, 300.0), Point::new(430.0, 310.0), Point::new(420.0, 110.0)],
        ];
        let mut img = GrayImage::from_pixel(w, h, Luma([210]));
        for (&id, quad) in ids.iter().zip(&placements) {
            let m = render_marker(id, 20); // 160 px, marker body 20..140
            let src = [Point::new(20.0, 20.0), Point::new(140.0, 20.0), Point::new(140.0, 140.0), Point::new(20.0, 140.0)];
            // Extend the quad to include the white quiet zone.
            let hm = Homography::from_points(&src, quad).unwrap();
            let inv = hm.inverse().unwrap();
            for y in 0..h {
                for x in 0..w {
                    let p = inv.apply(Point::new(x as f64 + 0.5, y as f64 + 0.5));
                    if p.x >= 0.0 && p.y >= 0.0 && p.x < 160.0 && p.y < 160.0 {
                        let v = sample_bilinear(&m, p.x - 0.5, p.y - 0.5);
                        img.put_pixel(x, y, Luma([(20.0 + v * 0.85) as u8]));
                    }
                }
            }
        }
        let found = detect(&img, &DetectorParams { work_size: 600, ..Default::default() });
        assert_eq!(found.len(), 3, "{found:?}");
        for (&id, quad) in ids.iter().zip(&placements) {
            let m = found.iter().find(|m| m.id == id).unwrap_or_else(|| panic!("id {id} missing"));
            for k in 0..4 {
                let err = m.corners[k].dist(quad[k]);
                assert!(err < 0.6, "id {id} corner {k}: {:?} vs {:?} ({err:.2} px)", m.corners[k], quad[k]);
            }
        }
    }
}
