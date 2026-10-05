package io.github.pedrogush.cardfscan.core.geometry

import kotlin.math.abs
import kotlin.math.sqrt

/**
 * A 3x3 projective transform, stored row-major in [m] (9 values).
 *
 * Written in plain Kotlin (no OpenCV) so the geometry can be unit-tested without native code.
 */
class Homography(val m: DoubleArray) {
    init {
        require(m.size == 9) { "a homography has 9 entries" }
    }

    fun apply(p: Point2): Point2 {
        val w = m[6] * p.x + m[7] * p.y + m[8]
        return Point2((m[0] * p.x + m[1] * p.y + m[2]) / w, (m[3] * p.x + m[4] * p.y + m[5]) / w)
    }

    /** Matrix product: (this * other) maps a point through [other] first, then through this. */
    operator fun times(other: Homography): Homography {
        val a = m
        val b = other.m
        val r = DoubleArray(9)
        for (i in 0..2) for (j in 0..2) {
            r[3 * i + j] = (0..2).sumOf { k -> a[3 * i + k] * b[3 * k + j] }
        }
        return Homography(r)
    }

    fun inverse(): Homography {
        val a = m[0]; val b = m[1]; val c = m[2]
        val d = m[3]; val e = m[4]; val f = m[5]
        val g = m[6]; val h = m[7]; val i = m[8]
        val det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
        require(abs(det) > 1e-15) { "singular homography" }
        val inv = doubleArrayOf(
            e * i - f * h, c * h - b * i, b * f - c * e,
            f * g - d * i, a * i - c * g, c * d - a * f,
            d * h - e * g, b * g - a * h, a * e - b * d,
        )
        return Homography(DoubleArray(9) { inv[it] / det })
    }

    // `companion object` holds "static" members: call them as Homography.fit(...).
    companion object {
        fun scale(s: Double) = Homography(doubleArrayOf(s, 0.0, 0.0, 0.0, s, 0.0, 0.0, 0.0, 1.0))

        fun translate(dx: Double, dy: Double) = Homography(doubleArrayOf(1.0, 0.0, dx, 0.0, 1.0, dy, 0.0, 0.0, 1.0))

        /**
         * Least-squares homography mapping [src] onto [dst] (at least 4 pairs), via the
         * normalised Direct Linear Transform with h33 = 1. Points are first moved to their
         * centroid and scaled (Hartley normalisation) so the equations stay well conditioned
         * even when one side is in thousands of pixels and the other in millimetres.
         */
        fun fit(src: List<Point2>, dst: List<Point2>): Homography {
            require(src.size == dst.size && src.size >= 4) { "need >= 4 point pairs" }
            val ns = normaliser(src)
            val nd = normaliser(dst)
            val s = src.map(ns::apply)
            val d = dst.map(nd::apply)

            // Normal equations A^T A h = A^T b for the 8 unknowns.
            val ata = Array(8) { DoubleArray(8) }
            val atb = DoubleArray(8)
            fun addRow(row: DoubleArray, rhs: Double) {
                for (r in 0 until 8) {
                    atb[r] += row[r] * rhs
                    for (c in 0 until 8) ata[r][c] += row[r] * row[c]
                }
            }
            for (k in s.indices) {
                val (x, y) = s[k]
                val (u, v) = d[k]
                addRow(doubleArrayOf(x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y), u)
                addRow(doubleArrayOf(0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y), v)
            }
            val h = solve(ata, atb)
            val normalised = Homography(DoubleArray(9) { if (it < 8) h[it] else 1.0 })
            // Undo the normalisation: H = Nd^-1 * Hn * Ns, then rescale so h33 = 1.
            val full = nd.inverse() * normalised * ns
            return Homography(DoubleArray(9) { full.m[it] / full.m[8] })
        }

        private fun normaliser(points: List<Point2>): Homography {
            val cx = points.sumOf { it.x } / points.size
            val cy = points.sumOf { it.y } / points.size
            val meanDist = points.sumOf { sqrt((it.x - cx) * (it.x - cx) + (it.y - cy) * (it.y - cy)) } / points.size
            val s = if (meanDist > 0) sqrt(2.0) / meanDist else 1.0
            return scale(s) * translate(-cx, -cy)
        }

        /** Gaussian elimination with partial pivoting. */
        private fun solve(a: Array<DoubleArray>, b: DoubleArray): DoubleArray {
            val n = b.size
            val mat = Array(n) { r -> a[r].copyOf() }
            val rhs = b.copyOf()
            for (col in 0 until n) {
                val pivot = (col until n).maxBy { abs(mat[it][col]) }
                require(abs(mat[pivot][col]) > 1e-12) { "degenerate point configuration" }
                mat[col] = mat[pivot].also { mat[pivot] = mat[col] }
                rhs[col] = rhs[pivot].also { rhs[pivot] = rhs[col] }
                for (r in col + 1 until n) {
                    val f = mat[r][col] / mat[col][col]
                    if (f == 0.0) continue
                    for (c in col until n) mat[r][c] -= f * mat[col][c]
                    rhs[r] -= f * rhs[col]
                }
            }
            val x = DoubleArray(n)
            for (r in n - 1 downTo 0) {
                var sum = rhs[r]
                for (c in r + 1 until n) sum -= mat[r][c] * x[c]
                x[r] = sum / mat[r][r]
            }
            return x
        }
    }
}
