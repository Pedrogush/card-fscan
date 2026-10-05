package io.github.pedrogush.cardfscan.core.geometry

import kotlin.math.roundToInt

/** A point in 2-D (pixels or millimetres, depending on context). */
data class Point2(val x: Double, val y: Double)

/** An axis-aligned rectangle; [x0],[y0] is the top-left corner. */
data class Box(val x0: Double, val y0: Double, val x1: Double, val y1: Double) {
    val width get() = x1 - x0
    val height get() = y1 - y0
}

/** The two print-pack layouts: 4 or 6 columns of strips (SPEC section 1). */
enum class Config(val columns: Int) {
    C4(4),
    C6(6),
}

/**
 * Every physical constant of the print pack (SPEC section 1), in millimetres,
 * in *strip coordinates*: origin at the strip's top-left, x right, y down the column.
 *
 * An `object` is a singleton: there is exactly one StripGeometry, used as `StripGeometry.SLOT_PITCH`.
 */
object StripGeometry {
    const val STRIP_WIDTH = 70.0
    const val STRIP_HEIGHT = 240.0
    const val HEADER_Y = 25.0
    const val SLOT_PITCH = 10.0
    const val CARDS_PER_COLUMN = 20

    /** Canonical warp resolution (SPEC section 2, step 3). */
    const val PX_PER_MM = 12.0

    const val STOP_CARD_ID = 40
    /** The stop card's marker top sits this far below the card's top edge. */
    const val STOP_MARKER_OFFSET = 3.0

    /** Name-bar crop across the card (SPEC section 2, step 5). */
    const val SLOT_X0 = 4.5
    const val SLOT_X1 = 65.5
    const val SLOT_INSET = 0.5
    /** The spec allows widening each slot crop by up to this much above and below. */
    const val MAX_SLOT_WIDEN = 1.5

    fun leftMarkerId(column: Int) = 2 * (column - 1)
    fun rightMarkerId(column: Int) = 2 * (column - 1) + 1

    /** Header marker corners in OpenCV order: top-left, top-right, bottom-right, bottom-left. */
    val LEFT_MARKER_CORNERS = listOf(Point2(6.0, 3.5), Point2(24.0, 3.5), Point2(24.0, 21.5), Point2(6.0, 21.5))
    val RIGHT_MARKER_CORNERS = listOf(Point2(46.0, 3.5), Point2(64.0, 3.5), Point2(64.0, 21.5), Point2(46.0, 21.5))

    /** Top edge (mm) of card / slot [slot] (1-based). */
    fun slotTop(slot: Int): Double = HEADER_Y + SLOT_PITCH * (slot - 1)

    /**
     * The crop rectangle for [slot] (1-based), optionally widened vertically by [widen] mm
     * on each side (clamped to the spec's 1.5 mm).
     */
    fun slotBox(slot: Int, widen: Double = 0.0): Box {
        require(slot in 1..CARDS_PER_COLUMN) { "slot must be 1..20, was $slot" }
        val w = widen.coerceIn(0.0, MAX_SLOT_WIDEN)
        val top = slotTop(slot)
        return Box(SLOT_X0, top + SLOT_INSET - w, SLOT_X1, top + SLOT_PITCH - SLOT_INSET + w)
    }

    /**
     * Which slot a stop card occupies, given the y (mm) of its marker's top edge
     * (SPEC section 2, step 4). Returns null when the position is outside slots 1..20.
     */
    fun stopCardSlot(markerTopY: Double): Int? {
        val k = ((markerTopY - STOP_MARKER_OFFSET - HEADER_Y) / SLOT_PITCH).roundToInt() + 1
        return k.takeIf { it in 1..CARDS_PER_COLUMN }
    }

    /** Which config a set of detected marker ids implies (SPEC section 2, step 2). */
    fun configFor(markerIds: Set<Int>): Config =
        if (markerIds.any { it in 8..11 }) Config.C6 else Config.C4
}
