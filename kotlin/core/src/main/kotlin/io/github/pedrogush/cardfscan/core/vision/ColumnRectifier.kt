package io.github.pedrogush.cardfscan.core.vision

import io.github.pedrogush.cardfscan.core.geometry.Homography
import io.github.pedrogush.cardfscan.core.geometry.Point2
import io.github.pedrogush.cardfscan.core.geometry.StripGeometry
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

/**
 * One column's mapping from photo pixels to strip millimetres (SPEC section 2, step 3).
 *
 * @property imageToMm homography fitted on the 8 header-marker corners.
 */
class ColumnRectifier(val imageToMm: Homography) {

    /** Strip-mm position of an image point. */
    fun toMm(p: Point2): Point2 = imageToMm.apply(p)

    /**
     * Warps the column to the canonical image: [StripGeometry.PX_PER_MM] px/mm covering
     * x 0..70 and y 0..[heightMm] (840 x 2880 px for the default 240 mm).
     */
    fun warp(photo: Mat, heightMm: Double = StripGeometry.STRIP_HEIGHT): Mat {
        val toPx = Homography.scale(StripGeometry.PX_PER_MM) * imageToMm
        val m = Mat(3, 3, CvType.CV_64F).apply { put(0, 0, *toPx.m) }
        val out = Mat()
        val size = Size(StripGeometry.STRIP_WIDTH * StripGeometry.PX_PER_MM, heightMm * StripGeometry.PX_PER_MM)
        Imgproc.warpPerspective(photo, out, m, size, Imgproc.INTER_LINEAR)
        m.release()
        return out
    }

    /**
     * If [stopMarker] lies inside this column, returns the slot (1..20) the stop card occupies
     * (SPEC section 2, step 4); otherwise null.
     *
     * The marker's top edge is the mean y of its two highest projected corners, so a stop card
     * placed upside down (marker corners rotated) still gives the right answer.
     */
    fun stopCardSlot(stopMarker: Marker): Int? {
        val mm = stopMarker.corners.map(::toMm)
        val cx = mm.sumOf { it.x } / 4
        if (cx < 0.0 || cx >= StripGeometry.STRIP_WIDTH) return null
        val topY = mm.map { it.y }.sorted().take(2).average()
        return StripGeometry.stopCardSlot(topY)
    }

    companion object {
        /** Fits the column homography from the left and right header markers. */
        fun fromHeaderMarkers(left: Marker, right: Marker): ColumnRectifier {
            val image = left.corners + right.corners
            val mm = StripGeometry.LEFT_MARKER_CORNERS + StripGeometry.RIGHT_MARKER_CORNERS
            return ColumnRectifier(Homography.fit(image, mm))
        }
    }
}
