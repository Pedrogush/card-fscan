package io.github.pedrogush.cardfscan.core.vision

import io.github.pedrogush.cardfscan.core.geometry.StripGeometry
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.roundToInt

/**
 * Finds the name-text line of every slot of a warped column at once.
 *
 * **Text energy.** Letters are mostly short vertical strokes, so each pixel row of the column
 * gets the sum of |d/dx| of the grey image over x 5..45 mm (the left of the name bar, clear
 * of the mana cost), after removing vertical edges taller than [VERTICAL_LINE_MM] (frame
 * borders). Name bars are smooth, so text rows stand out. The energy of a
 * candidate line centred on row r is the sum over a [TEXT_HEIGHT_MM] window.
 *
 * **Joint choice.** Choosing each slot's strongest window independently fails when a card's
 * name sits low in its band (some frames do this) or the next card is placed high: the
 * visible part of the *previous* card's name can then win inside the next slot's band, and
 * two slots read the same name. Physically the lines are ordered down the column and cannot
 * overlap, so we pick one centre per slot, each at least [MIN_GAP_MM] below the previous
 * one, maximising the total energy. That is a small dynamic programme (Viterbi-style):
 * best[i](r) = w(r) E(r) + max over r' <= r - gap of best[i-1](r').
 *
 * **Position prior** w: a card's own name sits 2..8.5 mm below its top edge (old frames high,
 * some special frames low) plus placement jitter. A line hugging the top or bottom of the
 * widened band is more likely a neighbour's half-hidden name, so its energy is scaled down
 * linearly to [EDGE_WEIGHT] at the band edges.
 *
 * **Stop card**: the last card before a stop card cannot show text below the stop card's top
 * edge, so its candidates end there.
 */
/**
 * The name line chosen for one slot.
 *
 * @property centreMm text centre, mm below the slot's nominal top edge (-1..11).
 * @property ambiguous the line hugs the edge of the band while another comparably strong line
 *   exists in the same band: we cannot tell which card the text belongs to, so this slot must
 *   not be auto-accepted.
 */
data class TextLine(val centreMm: Double, val ambiguous: Boolean)

class TextLineFinder {
    private val px = StripGeometry.PX_PER_MM

    /** @return the name line of each of slots 1..[nSlots]. */
    fun find(column: Mat, nSlots: Int, stopCardTopMm: Double? = null): List<TextLine> {
        if (nSlots == 0) return emptyList()
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

        // Candidate centres span the whole spec band; the crop itself is clamped inside it later.
        val lo = CropVariant.MIN_MM
        val hi = CropVariant.MAX_MM
        val gap = (MIN_GAP_MM * px).roundToInt()
        val firstRow = IntArray(nSlots) { ((StripGeometry.slotTop(it + 1) + lo) * px).roundToInt() }
        val count = ((hi - lo) * px).roundToInt() + 1
        fun weight(offsetMm: Double): Double = when {
            offsetMm < PRIOR_LOW_MM -> EDGE_WEIGHT + (1 - EDGE_WEIGHT) * (offsetMm - lo) / (PRIOR_LOW_MM - lo)
            offsetMm > PRIOR_HIGH_MM -> EDGE_WEIGHT + (1 - EDGE_WEIGHT) * (hi - offsetMm) / (hi - PRIOR_HIGH_MM)
            else -> 1.0
        }.coerceIn(EDGE_WEIGHT, 1.0)
        // Last allowed centre row of the final slot when a stop card follows it.
        val lastRowLimit = stopCardTopMm?.let { ((it - TEXT_HEIGHT_MM / 2) * px).roundToInt() } ?: Int.MAX_VALUE
        fun lineScore(slot: Int, k: Int): Double {
            val row = firstRow[slot] + k
            if (slot == nSlots - 1 && row > lastRowLimit) return Double.NEGATIVE_INFINITY
            return weight(lo + k / px) * energy(row)
        }

        // score[i][k]: best total for slots 0..i with slot i centred on row firstRow[i] + k.
        val score = Array(nSlots) { DoubleArray(count) }
        val from = Array(nSlots) { IntArray(count) { -1 } }
        for (k in 0 until count) score[0][k] = lineScore(0, k)
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
                    score[i][k] = bestPrev + lineScore(i, k)
                    from[i][k] = bestPrevK
                }
            }
        }
        // Backtrack from the best end state.
        val chosen = IntArray(nSlots)
        var k = score[nSlots - 1].indices.maxBy { score[nSlots - 1][it] }
        for (i in nSlots - 1 downTo 0) {
            chosen[i] = k
            if (i > 0) k = from[i][k]
        }
        return (0 until nSlots).map { i ->
            val c = chosen[i]
            val offset = lo + c / px
            val onEdge = offset < PRIOR_LOW_MM || offset > PRIOR_HIGH_MM
            // Strongest line in the same band at least one gap away from the chosen one.
            val rival = (0 until count).filter { abs(it - c) >= gap }.maxOfOrNull { energy(firstRow[i] + it) } ?: 0.0
            TextLine(offset, ambiguous = onEdge && rival >= AMBIGUOUS_RATIO * energy(firstRow[i] + c))
        }
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
        // Remove long vertical edges (frame borders, box sides): an opening with a tall thin
        // kernel keeps only structures at least VERTICAL_LINE_MM tall, which letters are not.
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(1.0, VERTICAL_LINE_MM * px))
        val lines = Mat()
        Imgproc.morphologyEx(dx, lines, Imgproc.MORPH_OPEN, kernel)
        Core.subtract(dx, lines, dx)
        val rows = Mat()
        Core.reduce(dx, rows, 1, Core.REDUCE_SUM, CvType.CV_32F)
        val out = FloatArray(rows.rows())
        rows.get(0, 0, out)
        listOf(roi, gray, dx, rows, kernel, lines).forEach(Mat::release)
        return DoubleArray(out.size) { out[it].toDouble() }
    }

    companion object {
        /** Approximate height of a name's letters (cap height plus descenders). */
        const val TEXT_HEIGHT_MM = 3.0
        /** Vertical edges at least this tall are frame lines, not letters. */
        const val VERTICAL_LINE_MM = 5.0
        /** Two name lines are at least this far apart (centre to centre). */
        const val MIN_GAP_MM = 4.0
        /** Text-centre offsets (mm below the slot top) that get the full weight. */
        const val PRIOR_LOW_MM = 2.0
        const val PRIOR_HIGH_MM = 8.5
        const val EDGE_WEIGHT = 0.6
        /** An edge line is ambiguous when a rival line has at least this fraction of its energy. */
        const val AMBIGUOUS_RATIO = 0.6
    }
}
