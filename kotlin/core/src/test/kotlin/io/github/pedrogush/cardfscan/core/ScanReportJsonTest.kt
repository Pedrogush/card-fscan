package io.github.pedrogush.cardfscan.core

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/** The output must have exactly the keys and types of SPEC section 4. */
class ScanReportJsonTest {
    private val report = ScanReport(
        image = "c4_0001.jpg",
        config = "C4",
        status = "ok",
        columns = listOf(
            ColumnReport(
                column = 1, status = "error", stopCard = true, nCards = 2,
                slots = listOf(
                    SlotReport(
                        1, "auto", "Lightning Bolt {R}", "Lightning Bolt", "4457ed35-7c10-48c8-9776-456485fdf070", "en", 100.0,
                        listOf(CandidateReport("Lightning Bolt", "4457ed35-7c10-48c8-9776-456485fdf070", 100.0)),
                    ),
                    SlotReport(2, "review", "Ser Angl", null, null, null, 84.2, emptyList()),
                ),
            ),
        ),
        timingMs = mapOf("total" to 1234L),
    )

    @Test
    fun keysFollowTheSpec() {
        val root = Json.parseToJsonElement(report.toJson()).jsonObject
        assertEquals(1, root.getValue("spec_version").jsonPrimitive.content.toInt())
        assertTrue(root.keys.containsAll(listOf("image", "config", "status", "columns", "timing_ms")))
        val column = root.getValue("columns").jsonArray[0].jsonObject
        assertTrue(column.keys.containsAll(listOf("column", "status", "stop_card", "n_cards", "slots")))
        val auto = column.getValue("slots").jsonArray[0].jsonObject
        for (k in listOf("slot", "status", "raw_text", "name", "oracle_id", "lang", "score", "candidates")) {
            assertTrue(k in auto, "slot key $k")
        }
        // Non-auto slots carry explicit nulls for name / oracle_id / lang.
        val review = column.getValue("slots").jsonArray[1].jsonObject
        assertEquals(JsonNull, review["name"])
        assertEquals(JsonNull, review["oracle_id"])
        assertEquals(JsonNull, review["lang"])
        assertEquals(1234L, root.getValue("timing_ms").jsonObject.getValue("total").jsonPrimitive.content.toLong())
    }

    @Test
    fun roundTrips() {
        assertEquals(report, ScanReport.fromJson(report.toJson()))
    }
}
