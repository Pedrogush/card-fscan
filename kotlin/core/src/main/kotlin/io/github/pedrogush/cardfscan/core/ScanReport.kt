package io.github.pedrogush.cardfscan.core

import kotlinx.serialization.EncodeDefault
import kotlinx.serialization.ExperimentalSerializationApi
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

/*
 * The per-photo output JSON of SPEC section 4. Each `@Serializable data class` maps 1:1 to a
 * JSON object; `@SerialName` gives the snake_case key where the Kotlin name is camelCase.
 */

@Serializable
data class ScanReport(
    @SerialName("spec_version") val specVersion: Int = 1,
    val image: String,
    val config: String?,
    val status: String, // ok | retake
    val columns: List<ColumnReport>,
    @SerialName("timing_ms") val timingMs: Map<String, Long>,
) {
    fun toJson(): String = JSON.encodeToString(serializer(), this)

    companion object {
        @OptIn(ExperimentalSerializationApi::class)
        val JSON = Json {
            prettyPrint = true
            prettyPrintIndent = "  "
            encodeDefaults = true
            explicitNulls = true
        }
        fun fromJson(text: String): ScanReport = JSON.decodeFromString(serializer(), text)
    }
}

@Serializable
data class ColumnReport(
    val column: Int,
    val status: String, // ok | error
    @SerialName("stop_card") val stopCard: Boolean,
    @SerialName("n_cards") val nCards: Int,
    val slots: List<SlotReport>,
    /** Why the column is in error (not in the spec; extra keys are ignored by the scorer). */
    val reason: String? = null,
)

@Serializable
data class SlotReport(
    val slot: Int,
    val status: String, // auto | review | empty
    @SerialName("raw_text") val rawText: String,
    val name: String?,
    @SerialName("oracle_id") val oracleId: String?,
    val lang: String?,
    val score: Double?,
    val candidates: List<CandidateReport>,
    /** OCR confidence of the chosen reading and which crop variant produced it (diagnostics). */
    @SerialName("ocr_conf") val ocrConfidence: Double? = null,
    val variant: String? = null,
)

@Serializable
data class CandidateReport(
    val name: String,
    @SerialName("oracle_id") val oracleId: String,
    val score: Double,
)
