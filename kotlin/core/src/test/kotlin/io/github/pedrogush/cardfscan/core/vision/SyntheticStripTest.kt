package io.github.pedrogush.cardfscan.core.vision

import io.github.pedrogush.cardfscan.core.geometry.Config
import io.github.pedrogush.cardfscan.core.geometry.Homography
import io.github.pedrogush.cardfscan.core.geometry.Point2
import io.github.pedrogush.cardfscan.core.geometry.StripGeometry
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.opencv.objdetect.Objdetect
import kotlin.math.cos
import kotlin.math.roundToInt
import kotlin.math.sin
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull

/**
 * Draws a printed strip (header markers + a stop card) in millimetre space, photographs it
 * through a known homography, and checks the real OpenCV pipeline recovers the geometry.
 */
class SyntheticStripTest {
    private val pxPerMm = 8.0

    @BeforeTest
    fun loadOpenCv() = nu.pattern.OpenCV.loadLocally()

    private fun drawMarker(canvas: Mat, id: Int, xMm: Double, yMm: Double, sizeMm: Double) {
        val side = (sizeMm * pxPerMm).roundToInt()
        val marker = Mat()
        Objdetect.generateImageMarker(Objdetect.getPredefinedDictionary(Objdetect.DICT_4X4_50), id, side, marker, 1)
        val bgr = Mat()
        Imgproc.cvtColor(marker, bgr, Imgproc.COLOR_GRAY2BGR)
        bgr.copyTo(canvas.submat(Rect((xMm * pxPerMm).roundToInt(), (yMm * pxPerMm).roundToInt(), side, side)))
    }

    @Test
    fun recoversColumnAndStopCard() {
        // The strip in mm space: 70 x 300 mm at 8 px/mm, white.
        val strip = Mat((300 * pxPerMm).toInt(), (70 * pxPerMm).toInt(), CvType.CV_8UC3, Scalar(255.0, 255.0, 255.0))
        drawMarker(strip, 0, 6.0, 3.5, 18.0)
        drawMarker(strip, 1, 46.0, 3.5, 18.0)
        // Stop card in slot 14: card top at 25 + 130 mm; marker 3 mm lower, x 16.5 mm into the card.
        val stopSlot = 14
        drawMarker(strip, 40, 3.5 + 16.5, StripGeometry.slotTop(stopSlot) + 3.0, 30.0)

        // A camera: rotate 4 degrees, 11 px/mm, mild perspective.
        val a = Math.toRadians(4.0)
        val mmToPhoto = Homography(
            doubleArrayOf(11 * cos(a), -11 * sin(a), 300.0, 11 * sin(a), 11 * cos(a), 150.0, 1e-5, 2e-5, 1.0),
        )
        val stripPxToPhoto = mmToPhoto * Homography.scale(1 / pxPerMm)
        val photo = Mat()
        val m = Mat(3, 3, CvType.CV_64F).apply { put(0, 0, *stripPxToPhoto.m) }
        Imgproc.warpPerspective(strip, photo, m, Size(1400.0, 3600.0), Imgproc.INTER_LINEAR, 0, Scalar(90.0, 90.0, 90.0))

        val markers = MarkerDetector().detect(photo)
        val header = MarkerSelection.headerMarkers(markers)
        assertEquals(setOf(0, 1), header.keys)
        assertEquals(Config.C4, MarkerSelection.config(header))

        val rectifier = ColumnRectifier.fromHeaderMarkers(header.getValue(0), header.getValue(1))
        // A point far down the column maps back to within 1 mm. The header corners span only
        // 18 mm, so sub-pixel corner noise is amplified ~10x at y = 220 mm; the text-line
        // finder absorbs errors of this size.
        val probe = Point2(35.0, 220.0)
        val back = rectifier.toMm(mmToPhoto.apply(probe))
        assertEquals(probe.x, back.x, 1.0)
        assertEquals(probe.y, back.y, 1.0)

        val stop = assertNotNull(markers.firstOrNull { it.id == StripGeometry.STOP_CARD_ID })
        assertEquals(stopSlot, rectifier.stopCardSlot(stop))
    }
}
