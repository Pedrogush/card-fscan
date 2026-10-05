package io.github.pedrogush.cardfscan.core.ocr

import io.github.pedrogush.cardfscan.core.TestPaths
import org.junit.jupiter.api.Assumptions.assumeTrue
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc
import java.io.File
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals

class PaddleTextRecognizerTest {
    private val model = File(TestPaths.modelsDir, "latin_PP-OCRv5_rec_mobile.onnx")

    @BeforeTest
    fun loadOpenCv() = nu.pattern.OpenCV.loadLocally()

    /** Black text on a light background, roughly the size of a warped name bar. */
    private fun render(text: String): Mat {
        val img = Mat(66, 732, CvType.CV_8UC3, Scalar(225.0, 230.0, 235.0))
        Imgproc.putText(img, text, Point(10.0, 46.0), Imgproc.FONT_HERSHEY_DUPLEX, 1.3, Scalar(20.0, 20.0, 20.0), 2, Imgproc.LINE_AA)
        return img
    }

    @Test
    fun metadataDictionaryIsReadIntact() {
        assumeTrue(model.exists(), "model not found: $model")
        val dict = OnnxMetadata.read(model.readBytes()).getValue("character").split('\n').dropLastWhile { it.isEmpty() }
        // 502 characters + blank + space = the model's 504 output classes.
        assertEquals(502, dict.size)
        // The last entry is U+1D713, outside the Basic Multilingual Plane: the case JNI corrupts.
        assertEquals(0x1D713, dict.last().codePointAt(0))
    }

    @Test
    fun readsRenderedTextIncludingSpaces() {
        assumeTrue(model.exists(), "model not found: $model")
        PaddleTextRecognizer(model.readBytes()).use { ocr ->
            val lines = ocr.recognize(listOf(render("Lightning Bolt"), render("Serra Angel")))
            assertEquals("Lightning Bolt", lines[0].text.trim())
            assertEquals("Serra Angel", lines[1].text.trim())
        }
    }
}
