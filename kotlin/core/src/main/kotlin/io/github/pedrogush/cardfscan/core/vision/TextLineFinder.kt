package io.github.pedrogush.cardfscan.core.vision

import io.github.pedrogush.cardfscan.core.geometry.StripGeometry
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc
import kotlin.math.roundToInt

/**
 * Finds the name-text line of every slot of a warped column at once.
 *
 * **Text energy.** Letters are mostly vertical strokes, so each pixel row of the column gets
 * the sum of |d/dx| of the grey image over x 5..45 mm (the left of the name bar, clear of the
 * mana cost). Name bars and borders are smooth, so text rows stand out. The energy of a
 * candidate line centred on row r is the sum over a [TEXT_HEIGHT_MM] window.
 *
 * **Joint choice.** Choosing each slot's strongest window independently fails when a card's
 * name sits low in its band (some frames do this) or the next card is placed high: the
 * visible part of the *previous* card's name can then win inside the next slot's band, and
 * two slots read the same name. Physically the lines are ordered down the column and cannot
 * overlap, so we pick one centre per slot, each at least [MIN_GAP_MM] below the previous
 * one, maximising the total energy. That is a small dynamic programme (Viterbi-style):
 * best[i](r) = E(r) + max over r' <= r - gap of best[i-1](r').
 */
class TextLineFinder {
    private val px = StripGeometry.PX_PER_MM

    /**
     * @return for slots 1..[nSlots], the text centre in mm below that slot's nominal top
     *   edge, always inside the window-fitting part of the spec's band.
     */
    fun find(column: Mat, nSlots: Int): DoubleArray {
        if (nSlots == 0) return DoubleArray(0)
        val profile = rowEnergy(column)
        val win = (TEXT_HEIGHT_MM * px).roundToInt()
        // energy[r] = sum of profile over the window centred on row r (prefix sums).
        val prefix = DoubleArray(profile.size + 1)
        for (i in profile.indices) prefix[i + 1] = prefix[i] + profile[i]
        fun energy(r: Int): Double {
            val a = (r - win / 2).coerceIn(0, profile.size)
            val b = (r + win - win / 2).coerceIn(0, profile.size)
            return prefix[b] - prefix[a]
        }

        val lo = CropVariant.MIN_MM + TEXT_HEIGHT_MM / 2 // centre range relative to the slot top
        val hi = CropVariant.MAX_MM - TEXT_HEIGHT_MM / 2
        val gap = (MIN_GAP_MM * px).roundToInt()
        val firstRow = IntArray(nSlots) { ((StripGeometry.slotTop(it + 1) + lo) * px).roundToInt() }
        val count = ((hi - lo) * px).roundToInt() + 1

        // score[i][k]: best total for slots 0..i with slot i centred on row firstRow[i] + k.
        val score = Array(nSlots) { DoubleArray(count) }
        val from = Array(nSlots) { IntArray(count) { -1 } }
        for (k in 0 until count) score[0][k] = energy(firstRow[0] + k)
        for (i in 1 until nSlots) {
            // Running maximum over the previous slot's rows, consumed in row order.
            var bestPrev = Double.NEGATIVE_INFINITY
            var bestPrevK = -1
            var p = 0
            for (k in 0 until count) {
                val row = firstRow[i] + k
                while (p < count && firstRow[i - 1] + p <= row - gap) {
                    if (score[i - 1][p] > bestPrev) {
                        bestPrev = score[i - 1][p]
                        bestPrevK = p
                    }
                    p++
                }
                if (bestPrevK < 0) {
                    score[i][k] = Double.NEGATIVE_INFINITY
                } else {
                    score[i][k] = bestPrev + energy(row)
                    from[i][k] = bestPrevK
                }
            }
        }
        // Backtrack from the best end state.
        val centres = DoubleArray(nSlots)
        var k = score[nSlots - 1].indices.maxBy { score[nSlots - 1][it] }
        for (i in nSlots - 1 downTo 0) {
            centres[i] = lo + k / px
            if (i > 0) k = from[i][k]
        }
        return centres
    }

    private fun rowEnergy(column: Mat): DoubleArray {
        val x0 = (5.0 * px).roundToInt()
        val x1 = (45.0 * px).roundToInt()
        val roi = column.submat(Rect(x0, 0, x1 - x0, column.rows()))
        val gray = Mat()
        Imgproc.cvtColor(roi, gray, Imgproc.COLOR_BGR2GRAY)
        val dx = Mat()
        Imgproc.Sobel(gray, dx, CvType.CV_32F, 1, 0, 3)
        Core.absdiff(dx, Scalar(0.0), dx)
        val rows = Mat()
        Core.reduce(dx, rows, 1, Core.REDUCE_SUM, CvType.CV_32F)
        val out = FloatArray(rows.rows())
        rows.get(0, 0, out)
        listOf(roi, gray, dx, rows).forEach(Mat::release)
        return DoubleArray(out.size) { out[it].toDouble() }
    }

    companion object {
        /** Approximate height of a name's letters (cap height plus descenders). */
        const val TEXT_HEIGHT_MM = 3.0
        /** Two name lines are at least this far apart (centre to centre). */
        const val MIN_GAP_MM = 4.0
    }
}
