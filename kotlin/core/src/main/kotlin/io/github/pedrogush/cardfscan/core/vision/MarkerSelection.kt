package io.github.pedrogush.cardfscan.core.vision

import io.github.pedrogush.cardfscan.core.geometry.Config
import io.github.pedrogush.cardfscan.core.geometry.Point2
import io.github.pedrogush.cardfscan.core.geometry.StripGeometry
import kotlin.math.abs

/**
 * Picks the real header markers out of everything the detector found.
 *
 * Card art occasionally decodes as a stray DICT_4X4_50 marker. Taken literally, SPEC
 * section 2 step 2 would then switch a C4 photo to C6 ("any id 8..11 present") and call it a
 * retake. So we only trust markers that are geometrically plausible:
 *
 * - **Size**: all header markers are 18 mm squares seen from the same height, so a candidate
 *   whose side differs from the median header side by more than [SIZE_TOLERANCE] is dropped.
 *   When an id appears twice, the copy closest to the median size wins.
 * - **Position** (ids 8..11 only): they must lie on the header row, to the right of
 *   column 4, when projected through a column-1..4 homography.
 */
object MarkerSelection {
    const val SIZE_TOLERANCE = 0.35

    fun headerMarkers(markers: List<Marker>): Map<Int, Marker> {
        val candidates = markers.filter { it.id in 0..11 }
        if (candidates.isEmpty()) return emptyMap()
        val median = candidates.map { it.perimeter }.sorted()[candidates.size / 2]
        return candidates
            .filter { abs(it.perimeter / median - 1.0) <= SIZE_TOLERANCE }
            .groupBy { it.id }
            .mapValues { (_, copies) -> copies.minBy { abs(it.perimeter - median) } }
    }

    /** Decides C4/C6, ignoring ids 8..11 that are not on the header row. */
    fun config(header: Map<Int, Marker>): Config {
        val extra = header.filterKeys { it in 8..11 }.values
        if (extra.isEmpty()) return Config.C4
        val reference = (1..4).firstNotNullOfOrNull { j ->
            val l = header[StripGeometry.leftMarkerId(j)]
            val r = header[StripGeometry.rightMarkerId(j)]
            if (l != null && r != null) j to ColumnRectifier.fromHeaderMarkers(l, r) else null
        } ?: return Config.C6 // nothing to check against: follow the spec literally
        val (j, rectifier) = reference
        val onHeaderRow = extra.any { m ->
            val c = rectifier.toMm(centre(m.corners))
            // Strip coordinates of column j; columns 5-6 start at x = 70 * (5 - j).
            val x = c.x + StripGeometry.STRIP_WIDTH * (j - 1)
            c.y in -15.0..40.0 && x in 4 * StripGeometry.STRIP_WIDTH - 10..6 * StripGeometry.STRIP_WIDTH + 10
        }
        return if (onHeaderRow) Config.C6 else Config.C4
    }

    fun centre(points: List<Point2>) = Point2(points.sumOf { it.x } / points.size, points.sumOf { it.y } / points.size)
}
