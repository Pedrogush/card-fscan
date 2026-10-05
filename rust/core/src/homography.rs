//! Planar homographies (3x3 projective transforms) estimated with the
//! normalised Direct Linear Transform, using `nalgebra` for the linear algebra.

use nalgebra::{Matrix3, SMatrix, SymmetricEigen, Vector3};

use crate::geometry::Point;

/// Maps points of one plane to another: `dst ~ H * [x, y, 1]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Homography {
    pub m: Matrix3<f64>,
}

impl Homography {
    pub fn identity() -> Self {
        Homography { m: Matrix3::identity() }
    }

    /// Least-squares homography from >= 4 point pairs (normalised DLT).
    /// Returns `None` for degenerate input (e.g. collinear points).
    pub fn from_points(src: &[Point], dst: &[Point]) -> Option<Self> {
        assert_eq!(src.len(), dst.len(), "point lists must pair up");
        if src.len() < 4 {
            return None;
        }
        // Hartley normalisation: centre the points and scale them to an
        // average distance of sqrt(2). This keeps the system well conditioned.
        let (ts, src_n) = normalise(src)?;
        let (td, dst_n) = normalise(dst)?;

        // Each pair gives two rows of A (9 columns); we accumulate A^T A
        // directly so the matrix stays 9x9 whatever the number of points.
        let mut ata = SMatrix::<f64, 9, 9>::zeros();
        for (p, q) in src_n.iter().zip(&dst_n) {
            let r1 = [-p.x, -p.y, -1.0, 0.0, 0.0, 0.0, q.x * p.x, q.x * p.y, q.x];
            let r2 = [0.0, 0.0, 0.0, -p.x, -p.y, -1.0, q.y * p.x, q.y * p.y, q.y];
            for row in [r1, r2] {
                for i in 0..9 {
                    for j in 0..9 {
                        ata[(i, j)] += row[i] * row[j];
                    }
                }
            }
        }
        // The solution is the eigenvector of A^T A with the smallest eigenvalue.
        let eig = SymmetricEigen::new(ata);
        let (min_idx, _) = eig
            .eigenvalues
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.total_cmp(b.1))?;
        let h = eig.eigenvectors.column(min_idx);
        let hn = Matrix3::from_row_slice(h.as_slice());
        // Undo the normalisation: H = Td^-1 * Hn * Ts.
        let m = td.try_inverse()? * hn * ts;
        let scale = m[(2, 2)];
        if scale.abs() < 1e-12 {
            return None;
        }
        Some(Homography { m: m / scale })
    }

    pub fn apply(&self, p: Point) -> Point {
        let v = self.m * Vector3::new(p.x, p.y, 1.0);
        Point::new(v.x / v.z, v.y / v.z)
    }

    pub fn inverse(&self) -> Option<Self> {
        self.m.try_inverse().map(|m| Homography { m })
    }

    /// Compose: first apply `first`, then `self`.
    pub fn after(&self, first: &Homography) -> Homography {
        Homography { m: self.m * first.m }
    }
}

/// Similarity transform that centres `pts` and scales them to mean distance
/// sqrt(2), plus the transformed points.
fn normalise(pts: &[Point]) -> Option<(Matrix3<f64>, Vec<Point>)> {
    let n = pts.len() as f64;
    let cx = pts.iter().map(|p| p.x).sum::<f64>() / n;
    let cy = pts.iter().map(|p| p.y).sum::<f64>() / n;
    let mean_d = pts.iter().map(|p| (p.x - cx).hypot(p.y - cy)).sum::<f64>() / n;
    if mean_d < 1e-12 {
        return None;
    }
    let s = std::f64::consts::SQRT_2 / mean_d;
    let t = Matrix3::new(s, 0.0, -s * cx, 0.0, s, -s * cy, 0.0, 0.0, 1.0);
    let out = pts.iter().map(|p| Point::new(s * (p.x - cx), s * (p.y - cy))).collect();
    Some((t, out))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_h() -> Homography {
        Homography { m: Matrix3::new(11.3, 0.4, 812.0, -0.25, 11.6, 401.0, 1.2e-5, -3.0e-5, 1.0) }
    }

    #[test]
    fn round_trip_eight_points() {
        let h = sample_h();
        let src: Vec<Point> = crate::geometry::header_corners_mm(crate::geometry::Side::Left)
            .into_iter()
            .chain(crate::geometry::header_corners_mm(crate::geometry::Side::Right))
            .collect();
        let dst: Vec<Point> = src.iter().map(|&p| h.apply(p)).collect();
        let est = Homography::from_points(&src, &dst).unwrap();
        // Checks far from the header too, where errors are amplified.
        for p in [Point::new(0.0, 0.0), Point::new(70.0, 240.0), Point::new(35.0, 120.0)] {
            assert!(est.apply(p).dist(h.apply(p)) < 1e-6, "{p:?}");
        }
        let inv = est.inverse().unwrap();
        let p = Point::new(12.5, 200.0);
        assert!(inv.apply(est.apply(p)).dist(p) < 1e-9);
    }

    #[test]
    fn degenerate_returns_none() {
        let pts = [Point::new(0.0, 0.0); 4];
        assert!(Homography::from_points(&pts, &pts).is_none());
    }
}
