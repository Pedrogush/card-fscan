"""1:1 SVG sheet of rack header stickers (ArUco DICT_4X4_50), per spec/SPEC.md section 1.

Each sticker is exactly the header plate: 64 x 24.5 mm (strip x 3..67, y 0..24.5). Cut on the solid
outline and stick it flush with the plate's edges, the "card 1" edge towards the cards. Markers then
sit at strip x 6..24 / 46..64, y 3.5..21.5. Print at 100% (check the 100 mm ruler).
"""
import os
import cv2
import shingle_cad as sc

STICKER_W = sc.RACK_W - 2 * 3.0                  # header plate width between the wall slabs
STICKER_H = sc.U_HEADER_END - sc.U_FRONT          # 24.5
X0_STRIP = (sc.RACK_W - STICKER_W) / 2           # strip x of the sticker's left edge (3)
MARKER, CELLS = 18.0, 6
MARKERS_X = (6.0, 46.0)                          # strip x of left / right marker (SPEC 1)
MARKER_Y = 3.5                                   # strip y of marker tops (SPEC 1)
DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


def marker_rects(mid, x, y):
    bits = cv2.aruco.generateImageMarker(DICT, mid, CELLS)
    c = MARKER / CELLS
    out = [f'<rect x="{x:.3f}" y="{y:.3f}" width="{MARKER}" height="{MARKER}" fill="#000"/>']
    for r in range(CELLS):
        for k in range(CELLS):
            if bits[r, k]:
                out.append(f'<rect x="{x + k * c:.3f}" y="{y + r * c:.3f}" width="{c:.3f}" '
                           f'height="{c:.3f}" fill="#fff"/>')
    return out


def sticker(col, ox, oy):
    s = [f'<g transform="translate({ox} {oy})">',
         f'<rect x="0" y="0" width="{STICKER_W}" height="{STICKER_H}" fill="#fff" stroke="#000" stroke-width="0.15"/>']
    for i, mx in enumerate(MARKERS_X):
        s += marker_rects(2 * (col - 1) + i, mx - X0_STRIP, MARKER_Y)
    cx = STICKER_W / 2
    s.append(f'<text x="{cx}" y="10" font-family="Arial" font-size="3.2" font-weight="bold" text-anchor="middle">COL</text>')
    s.append(f'<text x="{cx}" y="18.5" font-family="Arial" font-size="8" font-weight="bold" text-anchor="middle">{col}</text>')
    s.append(f'<text x="{cx}" y="{STICKER_H - 0.8}" font-family="Arial" font-size="1.6" fill="#777" '
             f'text-anchor="middle">card 1 edge</text>')
    s.append('</g>')
    return s


def ruler(ox, oy):
    s = [f'<g transform="translate({ox} {oy})" stroke="#000" stroke-width="0.2">',
         '<line x1="0" y1="0" x2="100" y2="0"/>']
    for mm in range(0, 101):
        h = 4 if mm % 10 == 0 else (2.5 if mm % 5 == 0 else 1.5)
        s.append(f'<line x1="{mm}" y1="0" x2="{mm}" y2="{h}"/>')
    s.append('</g>')
    s.append(f'<text x="{ox + 50}" y="{oy + 9}" font-family="Arial" font-size="3" text-anchor="middle">'
             'Print check: this ruler must measure exactly 100 mm</text>')
    return s


def sheet(path, columns=range(1, 7)):
    W, H = 210, 297
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}mm" height="{H}mm" viewBox="0 0 {W} {H}">',
         f'<text x="20" y="20" font-family="Arial" font-size="5" font-weight="bold">card-fscan rack header stickers (1:1)</text>',
         f'<text x="20" y="27" font-family="Arial" font-size="3">Print at 100% / Actual size. Cut on the outline; '
         f'stick flush with the header plate, "card 1 edge" towards the cards.</text>']
    for i, col in enumerate(columns):
        ox = 20 + (i % 2) * (STICKER_W + 26)
        oy = 40 + (i // 2) * (STICKER_H + 14)
        s += sticker(col, ox, oy)
    s += ruler(55, 250)
    s.append('</svg>')
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(s))


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "header_stickers.svg")
    sheet(out)
    print("wrote", os.path.normpath(out), f"(sticker {STICKER_W} x {STICKER_H} mm)")
