package io.github.pedrogush.cardfscan.core.match

import kotlinx.serialization.Serializable
import kotlinx.serialization.SerialName
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.decodeFromStream
import java.io.File
import java.io.InputStream
import java.util.zip.GZIPInputStream

/** One row of `names_v1.json` (SPEC section 3). `@Serializable` generates the JSON reader for it. */
@Serializable
data class NameEntry(
    val key: String,
    val name: String,
    @SerialName("oracle_id") val oracleId: String,
    val lang: String,
)

@Serializable
private data class NameIndexFile(val version: Int, val source: String, val entries: List<NameEntry>)

/** A possible identity for a crop: one oracle_id with its best-scoring printed name. */
data class Candidate(val name: String, val oracleId: String, val lang: String, val key: String, val score: Double)

enum class SlotStatus { AUTO, REVIEW, EMPTY }

data class MatchResult(
    val status: SlotStatus,
    val cleaned: String,
    val key: String,
    /** Top 3 distinct oracle_ids, best first (empty when [status] is EMPTY). */
    val candidates: List<Candidate> = emptyList(),
    val runnerUpScore: Double = 0.0,
) {
    val best: Candidate? get() = candidates.firstOrNull()
}

/**
 * The in-memory name index and the matcher of SPEC section 3.
 *
 * Construction precomputes, for every entry, its key as ASCII bytes and the position of its
 * oracle_id in the sorted list of distinct oracle_ids, so [match] is a tight loop over arrays.
 */
class NameIndex(val entries: List<NameEntry>, val source: String = "") {
    private val keyBytes: Array<ByteArray> = Array(entries.size) { entries[it].key.toByteArray(Charsets.US_ASCII) }
    /** Distinct oracle_ids, sorted ascending: the index in this list doubles as the tie-breaker. */
    private val oracleIds: List<String> = entries.map { it.oracleId }.distinct().sorted()
    private val oraclePosition: Map<String, Int> = oracleIds.withIndex().associate { (i, id) -> id to i }
    private val entryOracle: IntArray = IntArray(entries.size) { oraclePosition.getValue(entries[it].oracleId) }

    /** Keys with spaces removed, for [isRobustToSpaces]. */
    private val compactKeyBytes: Array<ByteArray> by lazy {
        Array(entries.size) { entries[it].key.replace(" ", "").toByteArray(Charsets.US_ASCII) }
    }

    val size get() = entries.size

    /**
     * Extra safety check on top of SPEC section 3 (it can only turn `auto` into `review`).
     *
     * OCR sometimes drops or invents a space, and some names differ only by a space
     * ("Wasteland" vs the Un-card "Waste Land"). We recompute the scores with spaces removed
     * from both the query and every key, and require the accepted oracle_id to still lead
     * every other oracle_id by [AUTO_LEAD].
     */
    fun isRobustToSpaces(result: MatchResult): Boolean {
        val best = result.best ?: return false
        val query = result.key.replace(" ", "")
        if (query.isEmpty()) return false
        val pattern = IndelPattern(query)
        val bestOracle = oraclePosition.getValue(best.oracleId)
        var own = -1f
        var other = -1f
        for (e in compactKeyBytes.indices) {
            val s = pattern.ratio(compactKeyBytes[e])
            if (entryOracle[e] == bestOracle) own = maxOf(own, s) else other = maxOf(other, s)
        }
        return round4(own) - round4(other) >= AUTO_LEAD - EPS
    }

    fun match(raw: String): MatchResult {
        if (!TextNormalizer.hasLetter(raw)) return MatchResult(SlotStatus.EMPTY, "", "")
        val cleaned = TextNormalizer.clean(raw)
        val key = TextNormalizer.normalize(cleaned)
        if (key.isEmpty()) return MatchResult(SlotStatus.EMPTY, cleaned, key)

        // Best score per oracle_id, and the first entry (in file order) that reaches it.
        val pattern = IndelPattern(key)
        val bestScore = FloatArray(oracleIds.size) { -1f }
        val bestEntry = IntArray(oracleIds.size) { -1 }
        for (e in keyBytes.indices) {
            val s = pattern.ratio(keyBytes[e])
            val o = entryOracle[e]
            if (s > bestScore[o]) {
                bestScore[o] = s
                bestEntry[o] = e
            }
        }

        val top = topOracles(bestScore, 3)
        val candidates = top.map { o ->
            val e = entries[bestEntry[o]]
            Candidate(e.name, e.oracleId, e.lang, e.key, round4(bestScore[o]))
        }
        val best = candidates.first()
        val runnerUp = candidates.getOrNull(1)?.score ?: 0.0
        return MatchResult(acceptStatus(best, runnerUp), cleaned, key, candidates, runnerUp)
    }

    /** Indices of the [n] highest scores; ties go to the lower index, i.e. the smaller oracle_id. */
    private fun topOracles(scores: FloatArray, n: Int): List<Int> {
        val top = IntArray(n) { -1 }
        for (o in scores.indices) {
            val s = scores[o]
            // Insert o into the small sorted `top` array if it beats the current n-th entry.
            var pos = n
            while (pos > 0 && (top[pos - 1] < 0 || s > scores[top[pos - 1]])) pos--
            if (pos < n) {
                for (k in n - 1 downTo pos + 1) top[k] = top[k - 1]
                top[pos] = o
            }
        }
        return top.filter { it >= 0 }
    }

    companion object {
        const val AUTO_MIN = 90.0
        const val AUTO_LEAD = 5.0
        const val SHORT_KEY_LEN = 6
        const val SHORT_MIN = 95.0
        /** Threshold comparisons tolerate float noise (SPEC section 3, determinism details). */
        const val EPS = 1e-6

        fun acceptStatus(best: Candidate, runnerUp: Double): SlotStatus {
            var accept = best.score >= AUTO_MIN - EPS && best.score - runnerUp >= AUTO_LEAD - EPS
            if (best.key.length <= SHORT_KEY_LEN && best.score < SHORT_MIN - EPS) accept = false
            return if (accept) SlotStatus.AUTO else SlotStatus.REVIEW
        }

        /**
         * The reference scores in float32 and then rounds to 4 decimals before the accept test.
         * Float32 noise (~4e-6 near 95) is larger than the 1e-6 tolerance, so we must round the
         * same way to agree on boundary cases such as a lead of exactly 5.
         */
        fun round4(score: Float): Double =
            java.math.BigDecimal(score.toDouble()).setScale(4, java.math.RoundingMode.HALF_EVEN).toDouble()

        private val json = Json { ignoreUnknownKeys = true }

        /** Reads `names_v1.json` (or `.json.gz`) from a stream. */
        @OptIn(kotlinx.serialization.ExperimentalSerializationApi::class)
        fun load(input: InputStream, gzipped: Boolean = false): NameIndex {
            val stream = if (gzipped) GZIPInputStream(input) else input
            // `use` closes the stream when the block ends, even on exceptions (like try-with-resources).
            val file = stream.buffered().use { json.decodeFromStream<NameIndexFile>(it) }
            return NameIndex(file.entries, file.source)
        }

        fun load(file: File): NameIndex = load(file.inputStream(), gzipped = file.name.endsWith(".gz"))
    }
}
