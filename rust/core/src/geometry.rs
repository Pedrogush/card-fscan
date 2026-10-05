//! The physical standard from SPEC §1 and §2: strip layout in millimetres.
//!
//! Strip coordinates: origin at the strip's top-left corner, x to the right,
//! y down the column (towards card 20).

/// A point in some 2-D coordinate system (mm or pixels, depending on context).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl Point {
    pub const fn new(x: f64, y: f64) -> Self {
        Point { x, y }
    }

    pub fn dist(self, other: Point) -> f64 {
        (self.x - other.x).hypot(self.y - other.y)
    }
}

/// Axis-aligned rectangle `x0..x1, y0..y1`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    pub x0: f64,
    pub y0: f64,
    pub x1: f64,
    pub y1: f64,
}

impl Rect {
    pub fn width(&self) -> f64 {
        self.x1 - self.x0
    }
    pub fn height(&self) -> f64 {
        self.y1 - self.y0
    }
}

pub const PX_PER_MM: f64 = 12.0;
pub const STRIP_W: f64 = 70.0;
pub const STRIP_H: f64 = 240.0;
pub const HEADER_Y: f64 = 25.0;
pub const SLOT_PITCH: f64 = 10.0;
pub const CARDS_PER_COLUMN: usize = 20;
pub const STOP_CARD_ID: u16 = 40;
/// The stop marker's top edge sits this far below the stop card's top edge.
pub const STOP_MARKER_OFFSET: f64 = 3.0;
/// Side of the stop card's marker (5 mm cells).
pub const STOP_MARKER_SIZE: f64 = 30.0;
/// Slot crops: x 4.5..65.5, inset 0.5 mm from the 10 mm band.
pub const SLOT_X0: f64 = 4.5;
pub const SLOT_X1: f64 = 65.5;
pub const SLOT_INSET: f64 = 0.5;

/// Which header marker of a column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Left,
    Right,
}

/// Marker id of column `j` (1-based) on the given side.
pub fn marker_id(column: usize, side: Side) -> u16 {
    let base = 2 * (column as u16 - 1);
    match side {
        Side::Left => base,
        Side::Right => base + 1,
    }
}

/// The four corners (OpenCV order: TL, TR, BR, BL) of a header marker in
/// strip millimetres.
pub fn header_corners_mm(side: Side) -> [Point; 4] {
    let (x0, x1) = match side {
        Side::Left => (6.0, 24.0),
        Side::Right => (46.0, 64.0),
    };
    let (y0, y1) = (3.5, 21.5);
    [Point::new(x0, y0), Point::new(x1, y0), Point::new(x1, y1), Point::new(x0, y1)]
}

/// The nominal 10 mm band of slot `i` (1-based), inset by 0.5 mm, widened in
/// y by `pad_mm` on each side (SPEC allows up to 1.5 mm for placement jitter).
pub fn slot_box_mm(i: usize, pad_mm: f64) -> Rect {
    let top = HEADER_Y + SLOT_PITCH * (i as f64 - 1.0);
    Rect {
        x0: SLOT_X0,
        x1: SLOT_X1,
        y0: top + SLOT_INSET - pad_mm,
        y1: top + SLOT_PITCH - SLOT_INSET + pad_mm,
    }
}

/// SPEC §2.4: the slot occupied by a stop card whose marker top edge is at
/// `y_top` mm. The column then holds `slot - 1` cards.
pub fn stop_card_slot(y_top: f64) -> i64 {
    ((y_top - STOP_MARKER_OFFSET - HEADER_Y) / SLOT_PITCH).round() as i64 + 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slot_boxes() {
        let s1 = slot_box_mm(1, 0.0);
        assert_eq!((s1.x0, s1.y0, s1.x1, s1.y1), (4.5, 25.5, 65.5, 34.5));
        let s20 = slot_box_mm(20, 1.5);
        assert_eq!((s20.y0, s20.y1), (214.0, 226.0));
        // Matches the handoff doc's slot_box(): MARGIN + 1 .. MARGIN + CARD_W - 1.
        assert_eq!(s20.width(), 61.0);
    }

    #[test]
    fn marker_ids() {
        assert_eq!(marker_id(1, Side::Left), 0);
        assert_eq!(marker_id(1, Side::Right), 1);
        assert_eq!(marker_id(6, Side::Right), 11);
    }

    #[test]
    fn stop_slot() {
        // Stop card placed exactly in slot k: card top at 25 + 10(k-1).
        for k in 1..=20 {
            let y_top = HEADER_Y + 10.0 * (k as f64 - 1.0) + STOP_MARKER_OFFSET;
            assert_eq!(stop_card_slot(y_top), k);
            // +/- 2 mm placement jitter still rounds to the same slot.
            assert_eq!(stop_card_slot(y_top + 2.0), k);
            assert_eq!(stop_card_slot(y_top - 2.0), k);
        }
    }
}
