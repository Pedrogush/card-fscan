package io.github.pedrogush.cardfscan.core

import io.github.pedrogush.cardfscan.core.geometry.Config
import io.github.pedrogush.cardfscan.core.geometry.StripGeometry
import io.github.pedrogush.cardfscan.core.match.MatchResult
import io.github.pedrogush.cardfscan.core.match.NameIndex
import io.github.pedrogush.cardfscan.core.match.SlotStatus
import io.github.pedrogush.cardfscan.core.match.TextNormalizer
import io.github.pedrogush.cardfscan.core.ocr.OcrLine
import io.github.pedrogush.cardfscan.core.ocr.TextRecognizer
import io.github.pedrogush.cardfscan.core.vision.ColumnRectifier
import io.github.pedrogush.cardfscan.core.vision.CropVariant
import io.github.pedrogush.cardfscan.core.vision.Enhance
import io.github.pedrogush.cardfscan.core.vision.Marker
import io.github.pedrogush.cardfscan.core.vision.MarkerDetector
import io.github.pedrogush.cardfscan.core.vision.MarkerSelection
import io.github.pedrogush.cardfscan.core.vision.SlotCropper
import io.github.pedrogush.cardfscan.core.vision.TextLineFinder
import org.opencv.core.Mat
import org.opencv.imgcodecs.Imgcodecs
import java.io.File

/**
 * Tunables of the reader. The defaults are what the eval runs use.
 *
 * @property stages crop variants tried in order: a slot only goes on to the next stage while
 *   it is not yet auto-accepted, so most slots cost a single OCR call.
 */
data class ScanOptions(
    val stages: List<List<CropVariant>> = DEFAULT_STAGES,
    /** Folder to dump warped columns and crops into, for debugging (null = off). */
    val debugDir: File? = null,
) {
    companion object {
        /** Named stage lists, selectable from the CLI (`--preset`) for experiments. */
        val PRESETS: Map<String, List<List<CropVariant>>> = mapOf(
            "fixed" to listOf(
                listOf(CropVariant("color", Enhance.COLOR, 0.5, 9.5)),
                listOf(CropVariant("clahe", Enhance.CLAHE, 0.5, 9.5), CropVariant("inverted", Enhance.INVERTED, 0.5, 9.5)),
            ),
            "located" to listOf(
                listOf(CropVariant("loc", Enhance.COLOR, locate = true)),
                listOf(CropVariant("loc-inv", Enhance.INVERTED, locate = true)),
                listOf(CropVariant("fixed", Enhance.COLOR, 0.5, 9.5)),
            ),
        )
        val DEFAULT_STAGES: List<List<CropVariant>> = PRESETS.getValue("located")
    }
}

/**
 * The whole pipeline of SPEC section 2 for one photo: markers -> columns -> rectify ->
 * stop card -> slot crops -> OCR -> match -> [ScanReport].
 *
 * Create one scanner and reuse it: the name index and the OCR model are loaded once.
 */
class CardScanner(
    private val index: NameIndex,
    private val recognizer: TextRecognizer,
    private val options: ScanOptions = ScanOptions(),
) {
    private val detector = MarkerDetector()
    private val cropper = SlotCropper()
    private val lineFinder = TextLineFinder()

    fun scanFile(file: File): ScanReport {
        val start = System.nanoTime()
        val image = Imgcodecs.imread(file.absolutePath, Imgcodecs.IMREAD_COLOR)
        require(!image.empty()) { "cannot read image $file" }
        val decodeMs = msSince(start)
        try {
            val report = scan(image, file.name)
            val timing = linkedMapOf("total" to msSince(start), "decode" to decodeMs)
            timing.putAll(report.timingMs - "total")
            return report.copy(timingMs = timing)
        } finally {
            image.release()
        }
    }

    /** Scans an already decoded BGR photo. */
    fun scan(image: Mat, imageName: String): ScanReport {
        val start = System.nanoTime()
        val timing = linkedMapOf<String, Long>()

        // 1. Markers; stray ids decoded from card art are filtered out (see MarkerSelection).
        val markers = detector.detect(image)
        val byId = MarkerSelection.headerMarkers(markers)
        val stopMarkers = markers.filter { it.id == StripGeometry.STOP_CARD_ID }
        timing["markers"] = msSince(start)

        // 2. Config and column presence.
        val config: Config = MarkerSelection.config(byId)
        val plans = (1..config.columns).map { j -> planColumn(j, byId, stopMarkers) }
        val photoOk = plans.all { it.rectifier != null }

        // 3. Warp each present column and read its slots.
        val t1 = System.nanoTime()
        val columns = plans.filter { it.rectifier != null }.associate { it.column to it.rectifier!!.warp(image) }
        timing["warp"] = msSince(t1)
        debugColumns(imageName, columns)

        val readings = readSlots(columns, plans, imageName, timing)
        columns.values.forEach(Mat::release)

        val columnReports = plans.map { plan -> buildColumnReport(plan, readings) }
        timing["total"] = msSince(start)
        return ScanReport(
            image = imageName,
            config = config.name,
            status = if (photoOk) "ok" else "retake",
            columns = columnReports,
            timingMs = timing,
        )
    }

    /** What we know about column [column] before OCR. */
    private class ColumnPlan(
        val column: Int,
        val rectifier: ColumnRectifier?,
        val stopSlot: Int?,
        val error: String?,
    ) {
        /** Slots that hold cards: up to the one above the stop card, else all 20. */
        val nCards: Int get() = if (rectifier == null) 0 else (stopSlot?.minus(1) ?: StripGeometry.CARDS_PER_COLUMN)
    }

    private fun planColumn(j: Int, byId: Map<Int, Marker>, stopMarkers: List<Marker>): ColumnPlan {
        val left = byId[StripGeometry.leftMarkerId(j)]
        val right = byId[StripGeometry.rightMarkerId(j)]
        if (left == null || right == null) {
            val why = when {
                left == null && right == null -> "column markers not found"
                left == null -> "left marker ${StripGeometry.leftMarkerId(j)} not found"
                else -> "right marker ${StripGeometry.rightMarkerId(j)} not found"
            }
            return ColumnPlan(j, null, null, why)
        }
        val rectifier = ColumnRectifier.fromHeaderMarkers(left, right)
        // A stop card belongs to this column when its centre projects inside x 0..70.
        val stops = stopMarkers.mapNotNull(rectifier::stopCardSlot)
        val stopSlot = stops.minOrNull()
        return ColumnPlan(j, rectifier, stopSlot, null)
    }

    /** One OCR + match attempt on one slot. */
    private class Reading(val variant: CropVariant, val ocr: OcrLine, val match: MatchResult)

    private data class SlotKey(val column: Int, val slot: Int)

    /**
     * Reads every slot, stage by stage. All crops of a stage go to the recognizer as one
     * batch; slots that are auto-accepted drop out of later stages.
     */
    private fun readSlots(
        columns: Map<Int, Mat>,
        plans: List<ColumnPlan>,
        imageName: String,
        timing: MutableMap<String, Long>,
    ): Map<SlotKey, Reading> {
        var cropNs = 0L
        var ocrNs = 0L
        var matchNs = 0L
        var ocrCalls = 0L
        val best = HashMap<SlotKey, Reading>()
        var t = System.nanoTime()
        val lines: Map<Int, DoubleArray> = plans.filter { it.rectifier != null }
            .associate { it.column to lineFinder.find(columns.getValue(it.column), it.nCards) }
        cropNs += System.nanoTime() - t
        var pending = plans.filter { it.rectifier != null }
            .flatMap { p -> (1..p.nCards).map { SlotKey(p.column, it) } }
        for (stage in options.stages) {
            if (pending.isEmpty()) break
            val jobs = pending.flatMap { key -> stage.map { v -> key to v } }
            t = System.nanoTime()
            val crops = jobs.map { (key, v) ->
                cropper.crop(columns.getValue(key.column), key.slot, v, lines.getValue(key.column)[key.slot - 1])
            }
            debugCrops(imageName, jobs, crops)
            cropNs += System.nanoTime() - t
            t = System.nanoTime()
            val texts = recognizer.recognize(crops)
            ocrNs += System.nanoTime() - t
            ocrCalls += crops.size
            crops.forEach(Mat::release)
            t = System.nanoTime()
            jobs.forEachIndexed { i, (key, v) ->
                val reading = Reading(v, texts[i], guardSpaces(guardHeadline(texts[i].text, index.match(texts[i].text))))
                best[key] = better(best[key], reading)
            }
            matchNs += System.nanoTime() - t
            pending = pending.filter { best[it]?.match?.status != SlotStatus.AUTO }
        }
        timing["crop"] = cropNs / 1_000_000
        timing["ocr"] = ocrNs / 1_000_000
        timing["match"] = matchNs / 1_000_000
        timing["ocr_crops"] = ocrCalls
        return best
    }

    /**
     * Stricter than the spec, never looser: an all-caps reading is never auto-accepted.
     * Real names print in mixed case; all-caps bands on special frames ("Breaking News"
     * headlines such as THE PROSPERITY POST) often show a word that is a *different* card's
     * name. The few caps-style name frames just go to review.
     */
    private fun guardHeadline(raw: String, match: MatchResult): MatchResult =
        if (match.status == SlotStatus.AUTO && TextNormalizer.looksLikeHeadline(raw)) match.copy(status = SlotStatus.REVIEW) else match

    /** Stricter than the spec: the accepted name must also win when spaces are ignored (see [NameIndex.isRobustToSpaces]). */
    private fun guardSpaces(match: MatchResult): MatchResult =
        if (match.status == SlotStatus.AUTO && !index.isRobustToSpaces(match)) match.copy(status = SlotStatus.REVIEW) else match

    /** "Keep the better-scoring result" (SPEC section 2, step 6): auto beats review beats empty, then match score. */
    private fun better(a: Reading?, b: Reading): Reading {
        if (a == null) return b
        fun rank(r: Reading) = when (r.match.status) {
            SlotStatus.AUTO -> 2
            SlotStatus.REVIEW -> 1
            SlotStatus.EMPTY -> 0
        }
        val ra = rank(a)
        val rb = rank(b)
        if (ra != rb) return if (rb > ra) b else a
        val sa = a.match.best?.score ?: 0.0
        val sb = b.match.best?.score ?: 0.0
        return if (sb > sa) b else a
    }

    private fun buildColumnReport(plan: ColumnPlan, readings: Map<SlotKey, Reading>): ColumnReport {
        if (plan.rectifier == null) {
            return ColumnReport(plan.column, "error", stopCard = false, nCards = 0, slots = emptyList(), reason = plan.error)
        }
        val slots = (1..plan.nCards).map { i -> slotReport(i, readings[SlotKey(plan.column, i)]) }
        val allAuto = slots.all { it.status == "auto" }
        return ColumnReport(
            column = plan.column,
            status = if (allAuto) "ok" else "error",
            stopCard = plan.stopSlot != null,
            nCards = plan.nCards,
            slots = slots,
        )
    }

    private fun slotReport(slot: Int, reading: Reading?): SlotReport {
        val match = reading?.match
        val auto = match?.status == SlotStatus.AUTO
        val best = match?.best
        return SlotReport(
            slot = slot,
            status = (match?.status ?: SlotStatus.EMPTY).name.lowercase(),
            rawText = reading?.ocr?.text ?: "",
            name = if (auto) best?.name else null,
            oracleId = if (auto) best?.oracleId else null,
            lang = if (auto) best?.lang else null,
            score = best?.score?.let { Math.round(it * 100) / 100.0 },
            candidates = match?.candidates.orEmpty().map {
                CandidateReport(it.name, it.oracleId, Math.round(it.score * 100) / 100.0)
            },
            ocrConfidence = reading?.ocr?.confidence,
            variant = reading?.variant?.name,
        )
    }

    private fun debugColumns(imageName: String, columns: Map<Int, Mat>) {
        val dir = options.debugDir ?: return
        dir.mkdirs()
        val stem = imageName.substringBeforeLast('.')
        columns.forEach { (j, m) -> Imgcodecs.imwrite(File(dir, "${stem}_col$j.jpg").path, m) }
    }

    private fun debugCrops(imageName: String, jobs: List<Pair<SlotKey, CropVariant>>, crops: List<Mat>) {
        val dir = options.debugDir ?: return
        val stem = imageName.substringBeforeLast('.')
        jobs.forEachIndexed { i, (key, v) ->
            val name = "${stem}_col${key.column}_s%02d_%s.png".format(key.slot, v.name)
            Imgcodecs.imwrite(File(dir, name).path, crops[i])
        }
    }

    private fun msSince(t0: Long) = (System.nanoTime() - t0) / 1_000_000
}
