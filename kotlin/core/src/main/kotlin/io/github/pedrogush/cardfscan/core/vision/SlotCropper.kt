package io.github.pedrogush.cardfscan.core.vision

import io.github.pedrogush.cardfscan.core.geometry.StripGeometry
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.CLAHE
import org.opencv.imgproc.Imgproc
import kotlin.math.roundToInt

/** How a crop is enhanced before OCR (SPEC section 2, steps 5-6). */
enum class Enhance {
    /** The warped colour pixels as they are. */
    COLOR,
    /** Grey + CLAHE local contrast enhancement. */
    CLAHE,
    /** Grey + CLAHE, then inverted (for white text on dark name bars). */
    INVERTED,
}

/**
 * One way of cutting a slot out of the warped column.
 *
 * Bands are measured in mm from the slot's nominal card top edge (`25 + 10 (i-1)` mm).
 * The spec band is 0.5..9.5; implementations may widen it by up to 1.5 mm each side, so every
 * band must stay within [MIN_MM]..[MAX_MM].
 *
 * - With [locate] = false the crop is the fixed band [topMm]..[bottomMm].
 * - With [locate] = true the name text is first found inside [topMm]..[bottomMm] (see
 *   [SlotCropper.locateText]) and the crop is [heightMm] tall, centred on it. This absorbs
 *   placement jitter and makes the text fill more of the OCR model's 48 px input.
 */
data class CropVariant(
    val name: String,
    val enhance: Enhance,
    val topMm: Double = MIN_MM,
    val bottomMm: Double = MAX_MM,
    val locate: Boolean = false,
    val heightMm: Double = 5.5,
    /** Shift (mm) applied to the located centre, to try a second guess above or below. */
    val shiftMm: Double = 0.0,
) {
    init {
        require(topMm >= MIN_MM - 1e-9 && bottomMm <= MAX_MM + 1e-9 && topMm < bottomMm) {
            "crop band $topMm..$bottomMm mm is outside the spec's $MIN_MM..$MAX_MM"
        }
    }

    companion object {
        const val MIN_MM = StripGeometry.SLOT_INSET - StripGeometry.MAX_SLOT_WIDEN // -1.0
        const val MAX_MM = StripGeometry.SLOT_PITCH - StripGeometry.SLOT_INSET + StripGeometry.MAX_SLOT_WIDEN // 11.0
    }
}

/** Cuts slot crops out of a column warped at 12 px/mm. */
class SlotCropper {
    private val clahe: CLAHE = Imgproc.createCLAHE(2.0, Size(8.0, 2.0))
    private val px = StripGeometry.PX_PER_MM

    /** Returns the crop, plus the band (mm from the slot top) it was taken from. */
    fun crop(column: Mat, slot: Int, variant: CropVariant): Mat {
        val top = StripGeometry.slotTop(slot)
        var bandTop = variant.topMm
        var bandBottom = variant.bottomMm
        if (variant.locate) {
            val centre = locateText(column, slot, variant.topMm, variant.bottomMm) + variant.shiftMm
            val half = variant.heightMm / 2
            // Keep the crop inside the allowed band, sliding it rather than shrinking it.
            bandTop = (centre - half).coerceIn(variant.topMm, variant.bottomMm - variant.heightMm)
            bandBottom = bandTop + variant.heightMm
        }
        val x0 = (StripGeometry.SLOT_X0 * px).roundToInt()
        val x1 = (StripGeometry.SLOT_X1 * px).roundToInt()
        val y0 = ((top + bandTop) * px).roundToInt().coerceIn(0, column.rows() - 1)
        val y1 = ((top + bandBottom) * px).roundToInt().coerceIn(y0 + 1, column.rows())
        val roi = column.submat(Rect(x0, y0, x1 - x0, y1 - y0))
        val out = enhance(roi, variant.enhance)
        roi.release()
        return out
    }

    private fun enhance(roi: Mat, how: Enhance): Mat {
        val out = Mat()
        when (how) {
            Enhance.COLOR -> roi.copyTo(out)
            Enhance.CLAHE, Enhance.INVERTED -> {
                val gray = Mat()
                Imgproc.cvtColor(roi, gray, Imgproc.COLOR_BGR2GRAY)
                clahe.apply(gray, out)
                if (how == Enhance.INVERTED) Core.bitwise_not(out, out)
                gray.release()
            }
        }
        return out
    }

    /**
     * Finds the vertical centre (mm from the slot top) of the name text inside the band
     * [fromMm]..[toMm].
     *
     * Letters are mostly vertical strokes, so each pixel row gets the sum of |d/dx| of the
     * grey image over the left part of the name bar (x 5..45 mm, clear of the mana cost).
     * Name bars and borders are smooth, so the text rows stand out; the 3 mm window with the
     * largest total is taken as the text line.
     */
    fun locateText(column: Mat, slot: Int, fromMm: Double, toMm: Double): Double {
        val top = StripGeometry.slotTop(slot)
        val y0 = ((top + fromMm) * px).roundToInt().coerceIn(0, column.rows() - 1)
        val y1 = ((top + toMm) * px).roundToInt().coerceIn(y0 + 1, column.rows())
        val x0 = (5.0 * px).roundToInt()
        val x1 = (45.0 * px).roundToInt()
        val roi = column.submat(Rect(x0, y0, x1 - x0, y1 - y0))
        val gray = Mat()
        Imgproc.cvtColor(roi, gray, Imgproc.COLOR_BGR2GRAY)
        val dx = Mat()
        Imgproc.Sobel(gray, dx, CvType.CV_32F, 1, 0, 3)
        Core.absdiff(dx, Scalar(0.0), dx)
        val rows = Mat()
        Core.reduce(dx, rows, 1, Core.REDUCE_SUM, CvType.CV_32F)
        val profile = FloatArray(rows.rows())
        rows.get(0, 0, profile)
        listOf(roi, gray, dx, rows).forEach(Mat::release)

        val win = (TEXT_HEIGHT_MM * px).roundToInt().coerceAtMost(profile.size)
        var sum = 0.0
        for (i in 0 until win) sum += profile[i]
        var bestSum = sum
        var bestStart = 0
        for (start in 1..profile.size - win) {
            sum += profile[start + win - 1] - profile[start - 1]
            if (sum > bestSum) {
                bestSum = sum
                bestStart = start
            }
        }
        return fromMm + (bestStart + win / 2.0) / px
    }

    companion object {
        /** Approximate height of a name's letters (cap height plus descenders). */
        const val TEXT_HEIGHT_MM = 3.0
    }
}
