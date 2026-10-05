package io.github.pedrogush.cardfscan.core.vision

import io.github.pedrogush.cardfscan.core.geometry.Point2
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import org.opencv.objdetect.ArucoDetector
import org.opencv.objdetect.DetectorParameters
import org.opencv.objdetect.Objdetect
import kotlin.math.hypot

/** One detected ArUco marker: its id and 4 corners (image px) in OpenCV order TL, TR, BR, BL. */
data class Marker(val id: Int, val corners: List<Point2>) {
    val perimeter: Double get() = corners.indices.sumOf { i ->
        val a = corners[i]
        val b = corners[(i + 1) % 4]
        hypot(a.x - b.x, a.y - b.y)
    }
}

/**
 * Finds DICT_4X4_50 markers in a photo (SPEC section 2, step 1).
 *
 * Corners are refined to sub-pixel accuracy: the homography is fitted on an 18 mm tall header
 * and extrapolated ~200 mm down the column, so corner noise is amplified about ten-fold.
 */
class MarkerDetector {
    private val detector: ArucoDetector = run {
        val params = DetectorParameters()
        params.set_cornerRefinementMethod(Objdetect.CORNER_REFINE_SUBPIX)
        ArucoDetector(Objdetect.getPredefinedDictionary(Objdetect.DICT_4X4_50), params)
    }

    fun detect(image: Mat): List<Marker> {
        val gray = if (image.channels() == 1) image else Mat().also { Imgproc.cvtColor(image, it, Imgproc.COLOR_BGR2GRAY) }
        val corners = ArrayList<Mat>()
        val ids = Mat()
        detector.detectMarkers(gray, corners, ids)
        val markers = corners.indices.map { i ->
            val c = FloatArray(8)
            corners[i].get(0, 0, c)
            Marker(ids.get(i, 0)[0].toInt(), (0 until 4).map { k -> Point2(c[2 * k].toDouble(), c[2 * k + 1].toDouble()) })
        }
        corners.forEach(Mat::release)
        ids.release()
        if (gray !== image) gray.release()
        return markers
    }
}
