package io.github.pedrogush.cardfscan.core.ocr

import org.opencv.core.Mat

/** Text read from one single-line crop, with the mean per-character confidence (0..1). */
data class OcrLine(val text: String, val confidence: Double)

/**
 * Anything that can read single text lines. An `interface` lets tests or a future
 * engine plug in without touching the pipeline.
 */
interface TextRecognizer : AutoCloseable {
    /** Reads every crop (BGR or gray 8-bit [Mat]s); the result list has the same order. */
    fun recognize(crops: List<Mat>): List<OcrLine>
}
