package io.github.pedrogush.cardfscan.core.geometry

import kotlin.math.cos
import kotlin.math.sin
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull

class GeometryTest {
    @Test
    fun slotBoxesFollowTheSpec() {
        val first = StripGeometry.slotBox(1)
        assertEquals(Box(4.5, 25.5, 65.5, 34.5), first)
        val last = StripGeometry.slotBox(20)
        assertEquals(Box(4.5, 215.5, 65.5, 224.5), last)
        // Widening is clamped to 1.5 mm per side.
        assertEquals(Box(4.5, 24.0, 65.5, 36.0), StripGeometry.slotBox(1, widen = 5.0))
        assertFailsWith<IllegalArgumentException> { StripGeometry.slotBox(21) }
    }

    @Test
    fun stopCardSlotFromMarkerTop() {
        // Stop card in slot k has its card top at 25 + 10(k-1) and marker top 3 mm lower.
        for (k in 1..20) {
            val y = 25.0 + 10 * (k - 1) + 3.0
            assertEquals(k, StripGeometry.stopCardSlot(y))
            assertEquals(k, StripGeometry.stopCardSlot(y + 2.0)) // placement jitter
            assertEquals(k, StripGeometry.stopCardSlot(y - 2.0))
        }
        assertNull(StripGeometry.stopCardSlot(400.0))
    }

    @Test
    fun configFromMarkerIds() {
        assertEquals(Config.C4, StripGeometry.configFor(setOf(0, 1, 2, 3, 4, 5, 6, 7)))
        assertEquals(Config.C6, StripGeometry.configFor(setOf(0, 1, 9)))
        assertEquals(0, StripGeometry.leftMarkerId(1))
        assertEquals(11, StripGeometry.rightMarkerId(6))
    }

    @Test
    fun homographyRoundTripOnSyntheticCorners() {
        // A made-up camera: rotate 7 degrees, scale to ~11 px/mm, add perspective, shift.
        val a = Math.toRadians(7.0)
        val truth = Homography(
            doubleArrayOf(
                11.2 * cos(a), -11.2 * sin(a), 1500.0,
                11.2 * sin(a), 11.2 * cos(a), 400.0,
                2e-5, -1e-5, 1.0,
            ),
        )
        val stripCorners = StripGeometry.LEFT_MARKER_CORNERS + StripGeometry.RIGHT_MARKER_CORNERS
        val imageCorners = stripCorners.map(truth::apply)

        // Fit image -> strip (what the pipeline does) and check it inverts the camera everywhere,
        // including far down the column where errors would be largest.
        val fitted = Homography.fit(imageCorners, stripCorners)
        for (p in stripCorners + listOf(Point2(35.0, 120.0), Point2(4.5, 225.0), Point2(65.5, 240.0))) {
            val back = fitted.apply(truth.apply(p))
            assertEquals(p.x, back.x, 1e-6)
            assertEquals(p.y, back.y, 1e-6)
        }
        // Inverse and composition agree with each other.
        val identity = fitted * fitted.inverse()
        for (i in 0..8) assertEquals(if (i % 4 == 0) 1.0 else 0.0, identity.m[i] / identity.m[8], 1e-9)
    }
}
