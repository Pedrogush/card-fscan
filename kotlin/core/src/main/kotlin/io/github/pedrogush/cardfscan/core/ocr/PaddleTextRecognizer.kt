package io.github.pedrogush.cardfscan.core.ocr

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.nio.FloatBuffer
import kotlin.math.ceil
import kotlin.math.roundToInt

/**
 * PP-OCR (PaddleOCR) text-line recognition model run with ONNX Runtime.
 *
 * The model takes a batch of BGR images 48 px high, normalised to [-1, 1], and outputs per
 * time step (one per 8 px of width) a probability for each class: class 0 is the CTC
 * "blank", then the characters of the dictionary, then a space. The dictionary is stored in
 * the ONNX file's metadata under the key `character` (RapidOCR exports do this); it is read
 * with [OnnxMetadata] and checked against the model's output size.
 *
 * @param modelBytes the .onnx file contents.
 * @param threads ONNX Runtime intra-op threads.
 * @param widthStretch horizontal stretch applied when resizing to 48 px high. Card names are
 *   long relative to their height; stretching gives CTC more time steps per character.
 * @param batchSize crops per inference call.
 */
class PaddleTextRecognizer(
    modelBytes: ByteArray,
    threads: Int = 2,
    private val widthStretch: Double = 1.0,
    private val batchSize: Int = 16,
) : TextRecognizer {
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val classes: List<String>
    private val inputName: String

    init {
        val options = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(threads)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        session = env.createSession(modelBytes, options)
        inputName = session.inputNames.first()
        val dict = OnnxMetadata.read(modelBytes)["character"]
            ?: error("model has no 'character' metadata; supply a RapidOCR-style export")
        classes = listOf("") + dict.split('\n').dropLastWhile { it.isEmpty() } + listOf(" ")
        val modelClasses = (session.outputInfo.values.first().info as TensorInfo).shape.last()
        check(modelClasses == classes.size.toLong()) {
            "dictionary has ${classes.size} classes but the model outputs $modelClasses"
        }
    }

    override fun recognize(crops: List<Mat>): List<OcrLine> {
        val prepared = crops.map(::toModelInput)
        val results = arrayOfNulls<OcrLine>(crops.size)
        // Sort by width so each batch pads as little as possible, then restore the order.
        val order = prepared.indices.sortedBy { prepared[it].cols() }
        for (chunk in order.chunked(batchSize)) {
            val lines = runBatch(chunk.map { prepared[it] })
            chunk.forEachIndexed { k, idx -> results[idx] = lines[k] }
        }
        prepared.forEach(Mat::release)
        return results.map { it!! } // `!!` asserts non-null: every slot was filled above.
    }

    /** Resizes to 48 px high (BGR, 8-bit). */
    private fun toModelInput(crop: Mat): Mat {
        val bgr = Mat()
        if (crop.channels() == 1) Imgproc.cvtColor(crop, bgr, Imgproc.COLOR_GRAY2BGR) else crop.copyTo(bgr)
        val w = ceil(HEIGHT.toDouble() * crop.cols() / crop.rows() * widthStretch).toInt().coerceIn(HEIGHT, MAX_WIDTH)
        val out = Mat()
        Imgproc.resize(bgr, out, Size(w.toDouble(), HEIGHT.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
        bgr.release()
        return out
    }

    private fun runBatch(images: List<Mat>): List<OcrLine> {
        val width = images.maxOf { it.cols() }
        val plane = HEIGHT * width
        // Zero after normalisation = mid-grey padding, as PaddleOCR does.
        val data = FloatArray(images.size * 3 * plane)
        val row = ByteArray(width * 3)
        images.forEachIndexed { n, img ->
            val base = n * 3 * plane
            for (y in 0 until HEIGHT) {
                img.get(y, 0, row)
                for (x in 0 until img.cols()) {
                    for (c in 0 until 3) {
                        val v = row[3 * x + c].toInt() and 0xFF // bytes are signed in Kotlin/Java
                        data[base + c * plane + y * width + x] = v / 127.5f - 1f
                    }
                }
            }
        }
        val shape = longArrayOf(images.size.toLong(), 3, HEIGHT.toLong(), width.toLong())
        OnnxTensor.createTensor(env, FloatBuffer.wrap(data), shape).use { input ->
            session.run(mapOf(inputName to input)).use { output ->
                val tensor = output[0] as OnnxTensor
                val dims = tensor.info.shape // [N, T, C]
                val steps = dims[1].toInt()
                val nClasses = dims[2].toInt()
                val probs = tensor.floatBuffer
                return images.indices.map { n -> decode(probs, n * steps * nClasses, steps, nClasses) }
            }
        }
    }

    /** CTC greedy decoding: arg-max per step, merge repeats, drop blanks. */
    private fun decode(probs: FloatBuffer, offset: Int, steps: Int, nClasses: Int): OcrLine {
        val sb = StringBuilder()
        var confSum = 0.0
        var count = 0
        var prev = 0
        for (t in 0 until steps) {
            var best = 0
            var bestP = probs.get(offset + t * nClasses)
            for (c in 1 until nClasses) {
                val p = probs.get(offset + t * nClasses + c)
                if (p > bestP) {
                    bestP = p
                    best = c
                }
            }
            if (best != 0 && best != prev && best < classes.size) {
                sb.append(classes[best])
                confSum += bestP
                count++
            }
            prev = best
        }
        val conf = if (count == 0) 0.0 else confSum / count
        return OcrLine(sb.toString(), (conf * 1e4).roundToInt() / 1e4)
    }

    override fun close() {
        session.close()
    }

    companion object {
        const val HEIGHT = 48
        const val MAX_WIDTH = 1600
    }
}
