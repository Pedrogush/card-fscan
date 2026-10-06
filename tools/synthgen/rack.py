"""Rack mode (SPEC section 1b, fab/ v2): cards standing in tilted-slot racks.

Geometry follows ``fab/source/shingle_cad.py`` (the source of truth):
  * racks 70 wide, butted; each has two walls: a 3 mm slab (strip x 0..3 and
    67..70) plus a 1.5 mm grooved comb (x 3..4.5 and 65.5..67), all tops at
    the marker plane;
  * a white header plate / sticker at strip x 3..67, y 0..24.5 carrying the
    SPEC 1 markers;
  * card i (card-local s down its length, t across) tilted THETA = 20 deg: its
    top edge at strip y 25 + 10(i-1) in the marker plane, descending away from
    the header until it rests on the floor (FLOOR_T above the board).

World frame used here: X, Y = strip coordinates of the column (plus 70 (j-1)
and the rack's small placement affine), Z = depth BELOW the marker plane
(the camera sits at Z = -h, h = lens height above the marker plane, which is
how the stand's height table is defined). Everything is rendered as planar
polygons straight into an undistorted ("ideal") image with exact per-polygon
homographies, back to front, then lens distortion + the usual photometric
pipeline are applied by render.photograph(ideal=...).
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

import render as R

THETA = math.radians(20.0)
S_T, C_T = math.sin(THETA), math.cos(THETA)
CARD_T = 0.3
FLOOR_T = 3.0
H_RACK = FLOOR_T + R.CARD_H * S_T + CARD_T * C_T + 0.05   # board -> marker plane, 33.43 mm
SLAB_T, COMB_T = 3.0, 1.5
HEADER_Y1 = 24.5                     # header plate: strip y 0 .. 24.5 (u -25 .. -0.5)
FLOOR_Y0 = 25.0 + 78.0               # rack floor plate from u = 78
RACK_Y1 = 25.0 + 277.0               # rack back end (u = 277)
GROOVE_LO, GROOVE_HI = -0.65, 0.35   # groove band around each card plane (shingle_cad)
FOOT_DEPTH, FOOT_W, FOOT_H = 100.0, 200.0, 8.0
DEBUG_STASH: dict[str, Any] | None = None   # set to {} to capture the undistorted render

WALL_GREYS = [(120, 124, 128), (95, 98, 104), (150, 152, 155), (70, 72, 76)]  # BGR printed plastic
FLOOR_COLS = [(110, 160, 200), (120, 124, 128), (60, 62, 66)]                 # MDF, grey, black


# ---------------------------------------------------------------------------
# textures

def render_header_sticker(col: int) -> np.ndarray:
    """fab/header_stickers.svg sticker: 64 x 24.5 mm, markers at SPEC 1 spots."""
    s = R.SCENE_PX_PER_MM
    w, h = int(64.0 * s), int(round(HEADER_Y1 * s))
    img = Image.new("RGB", (w, h), (242, 242, 238))
    d = ImageDraw.Draw(img)
    cx = 32.0 * s
    R._text_centered(d, cx, 7.6 * s, "COL", 3.3 * s, R.INK, bold=True)
    R._text_centered(d, cx, 12.6 * s, str(col), 8.2 * s, R.INK, bold=True)
    R._text_centered(d, cx, 22.4 * s, "card 1 edge", 1.7 * s, (119, 119, 119))
    arr = np.array(img)[:, :, ::-1].copy()
    msz = int(R.MARKER_SIZE * s)
    for mid, (x0, y0) in zip(R.marker_ids(col), (R.MARKER_L, R.MARKER_R)):
        m = R._marker_bitmap(mid, msz)
        px, py = int(round((x0 - 3.0) * s)), int(round(y0 * s))
        arr[py:py + msz, px:px + msz][m == 0] = R.INK[::-1]
    return arr


def raw_card(jpeg: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Scryfall image at native resolution + rounded-corner alpha."""
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("cannot decode card image")
    h, w = img.shape[:2]
    return img, R._rounded_mask(w, h, 2.6 * w / R.CARD_W)


# ---------------------------------------------------------------------------
# painting into the ideal image

class Canvas:
    def __init__(self, cam: R.Camera, base: np.ndarray):
        self.cam = cam
        self.img = base
        self.h, self.w = base.shape[:2]

    def _roi(self, pts: np.ndarray) -> tuple[int, int, int, int] | None:
        x0 = max(0, int(math.floor(pts[:, 0].min())) - 2)
        y0 = max(0, int(math.floor(pts[:, 1].min())) - 2)
        x1 = min(self.w, int(math.ceil(pts[:, 0].max())) + 2)
        y1 = min(self.h, int(math.ceil(pts[:, 1].max())) + 2)
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, y0, x1, y1

    def tex(self, tex: np.ndarray, alpha: np.ndarray | None, corners3d: np.ndarray,
            shade: float = 1.0) -> None:
        """Paint ``tex`` whose corners (TL, TR, BR, BL) sit at ``corners3d``."""
        dst = self.cam.project_ideal3d(corners3d)
        roi = self._roi(dst)
        if roi is None:
            return
        x0, y0, x1, y1 = roi
        h, w = tex.shape[:2]
        src = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
        M = cv2.getPerspectiveTransform(src, (dst - [x0, y0]).astype(np.float32))
        size = (x1 - x0, y1 - y0)
        warped = cv2.warpPerspective(tex, M, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        a_src = alpha if alpha is not None else np.full((h, w), 255, np.uint8)
        a = cv2.warpPerspective(a_src, M, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                borderValue=0).astype(np.float32) / 255.0
        roi_img = self.img[y0:y1, x0:x1].astype(np.float32)
        out = roi_img * (1 - a[:, :, None]) + warped.astype(np.float32) * shade * a[:, :, None]
        self.img[y0:y1, x0:x1] = np.clip(out + 0.5, 0, 255).astype(np.uint8)

    def poly(self, pts3d: np.ndarray, bgr: tuple[float, float, float]) -> None:
        """Flat-coloured convex planar polygon (anti-aliased edges)."""
        dst = self.cam.project_ideal3d(pts3d)
        roi = self._roi(dst)
        if roi is None:
            return
        x0, y0, x1, y1 = roi
        mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
        p = np.round((dst - [x0, y0]) * 16).astype(np.int32)
        cv2.fillPoly(mask, [p], 255, lineType=cv2.LINE_AA, shift=4)
        a = mask.astype(np.float32)[:, :, None] / 255.0
        roi_img = self.img[y0:y1, x0:x1].astype(np.float32)
        out = roi_img * (1 - a) + np.array(bgr, np.float32)[None, None, :] * a
        self.img[y0:y1, x0:x1] = np.clip(out + 0.5, 0, 255).astype(np.uint8)


def _xy(m: np.ndarray, x: float, y: float, z: float) -> list[float]:
    p = m @ np.array([x, y, 1.0])
    return [p[0] / p[2], p[1] / p[2], z]


def card_corners(m: np.ndarray, i: int, dx: float, slide: float) -> np.ndarray:
    """3D corners TL, TR, BR, BL of card in slot i (slide > 0 = deeper along the groove)."""
    ytop = R.HEADER + R.PITCH * (i - 1)

    def P(s_: float, t: float) -> list[float]:
        ss = s_ + slide
        return _xy(m, R.CARD_X + dx + t, ytop + ss * C_T, ss * S_T)
    return np.array([P(0, 0), P(0, R.CARD_W), P(R.CARD_H, R.CARD_W), P(R.CARD_H, 0)])


def card_point(m: np.ndarray, i: int, dx: float, slide: float, s_: float, t: float) -> list[float]:
    ytop = R.HEADER + R.PITCH * (i - 1)
    ss = s_ + slide
    return _xy(m, R.CARD_X + dx + t, ytop + ss * C_T, ss * S_T)


# ---------------------------------------------------------------------------
# scene

def paint_rack(cv: Canvas, m: np.ndarray, col: dict[str, Any], placements: list[list[float]],
               cards: list[tuple[np.ndarray, np.ndarray]], stop: tuple[np.ndarray, np.ndarray],
               wall_bgr: tuple[int, int, int], floor_bgr: tuple[int, int, int], overhang_shade: float) -> None:
    cam = cv.cam
    Hf = H_RACK - FLOOR_T
    # rack floor plate (between the slabs)
    cv.poly(np.array([_xy(m, SLAB_T, FLOOR_Y0, Hf), _xy(m, 70 - SLAB_T, FLOOR_Y0, Hf),
                      _xy(m, 70 - SLAB_T, RACK_Y1, Hf), _xy(m, SLAB_T, RACK_Y1, Hf)]), floor_bgr)
    # cards back to front: card 1 first, each next one lies on top
    n_items = len(cards) + (1 if col["stop_card"] else 0)
    for k in range(n_items):
        i = k + 1
        dx, slide, _ = placements[k]
        if k < len(cards):
            tex, alpha = cards[k]
        else:
            tex, alpha = stop
        if i < n_items and overhang_shade > 0:
            # the next card overhangs this one by ~3.4 mm: darken the lower part of the band
            tex = tex.copy()
            h = tex.shape[0]
            s0, s1 = int(5.5 / R.CARD_H * h), int(11.0 / R.CARD_H * h)
            ramp = 1.0 - overhang_shade * np.linspace(0, 1, s1 - s0, dtype=np.float32)
            tex[s0:s1] = np.clip(tex[s0:s1].astype(np.float32) * ramp[:, None, None], 0, 255).astype(np.uint8)
        cv.tex(tex, alpha, card_corners(m, i, dx, slide))

    # inner comb faces (vertical planes x = 4.5 and 65.5), only the parts above the cards
    face_bgr = tuple(c * 0.55 for c in wall_bgr)
    cam_local = np.linalg.inv(m) @ np.array([cam.C[0], cam.C[1], 1.0])
    cam_x = cam_local[0] / cam_local[2]
    for xf, visible in ((SLAB_T + COMB_T, cam_x > SLAB_T + COMB_T), (70 - SLAB_T - COMB_T, cam_x < 70 - SLAB_T - COMB_T)):
        if not visible:
            continue
        for k in range(n_items):
            i = k + 1
            y0 = R.HEADER + R.PITCH * (i - 1)
            if k < n_items - 1:
                y1 = y0 + R.PITCH
                z1 = R.PITCH * S_T / C_T
                pts = [_xy(m, xf, y0, 0), _xy(m, xf, y1, 0), _xy(m, xf, y1, z1)]
            else:
                yb, zb = y0 + R.CARD_H * C_T, R.CARD_H * S_T
                pts = [_xy(m, xf, y0, 0), _xy(m, xf, RACK_Y1, 0), _xy(m, xf, RACK_Y1, Hf), _xy(m, xf, yb, Hf),
                       _xy(m, xf, yb, zb)]
            cv.poly(np.array(pts), face_bgr)
        if n_items == 0:
            cv.poly(np.array([_xy(m, xf, R.HEADER, 0), _xy(m, xf, RACK_Y1, 0), _xy(m, xf, RACK_Y1, Hf),
                              _xy(m, xf, R.HEADER, H_RACK)]), face_bgr)

    # wall tops: slabs, combs, and the groove openings in the comb tops
    groove_bgr = tuple(c * 0.35 for c in wall_bgr)
    for xa, xb in ((0.0, SLAB_T), (70 - SLAB_T, 70.0)):
        cv.poly(np.array([_xy(m, xa, 0, 0), _xy(m, xb, 0, 0), _xy(m, xb, RACK_Y1, 0), _xy(m, xa, RACK_Y1, 0)]),
                wall_bgr)
    for xa, xb in ((SLAB_T, SLAB_T + COMB_T), (70 - SLAB_T - COMB_T, 70 - SLAB_T)):
        cv.poly(np.array([_xy(m, xa, HEADER_Y1, 0), _xy(m, xb, HEADER_Y1, 0), _xy(m, xb, RACK_Y1, 0),
                          _xy(m, xa, RACK_Y1, 0)]), wall_bgr)
        for i in range(1, R.N_SLOTS + 1):
            yt = R.HEADER + R.PITCH * (i - 1)
            ga, gb = yt + GROOVE_LO / S_T, yt + GROOVE_HI / S_T   # groove band where it meets the top plane
            cv.poly(np.array([_xy(m, xa, ga, 0), _xy(m, xb, ga, 0), _xy(m, xb, gb, 0), _xy(m, xa, gb, 0)]),
                    groove_bgr)


def scene_points(rack_m: list[np.ndarray], entry: dict[str, Any], placements: list[list[list[float]]]) -> dict[str, Any]:
    """3D points used for framing and truth: header marker corners, name zone, stop markers."""
    markers: dict[int, np.ndarray] = {}
    need = []
    for j, m in enumerate(rack_m):
        for mid, c in R.marker_corners_strip(j + 1).items():
            p = np.array([_xy(m, x, y, 0.0) for x, y in c])
            markers[mid] = p
            need.append(p)
        need.append(np.array([_xy(m, 0, R.HEADER, 0), _xy(m, 70, R.HEADER, 0),
                              _xy(m, 0, R.HEADER + 200, 0), _xy(m, 70, R.HEADER + 200, 0)]))
    stops = []
    for j, col in enumerate(entry["columns"]):
        if col["stop_card"]:
            k = col["n_cards"] + 1
            dx, slide, _ = placements[j][k - 1]
            x0, y0 = R.STOP_MARKER_XY
            sz = R.STOP_MARKER_SIZE
            p = np.array([card_point(rack_m[j], k, dx, slide, s_, t)
                          for s_, t in ((y0, x0), (y0, x0 + sz), (y0 + sz, x0 + sz), (y0 + sz, x0))])
            stops.append(p)
            need.append(p)
    return {"markers": markers, "stops": stops, "need": np.vstack(need)}


def render_entry_rack(entry: dict[str, Any], sf: Any, G: Any) -> tuple[bytes, dict[str, Any], dict[str, Any]]:
    """Rack-mode counterpart of gen.render_entry (G = the gen module, for helpers)."""
    cfg = R.CONFIGS[entry["config"]]
    lv = G.LEVELS[entry["level"]]
    rk = G.RACK_LEVELS[entry["level"]]
    ncols = cfg["cols"]
    rng = np.random.default_rng([entry["seed"], 1, entry["attempt"]])

    # racks pushed against the foot: tiny block offset/rotation, butted edge to edge
    block_rot = G._sym(rng, rk["rack_rot"])
    block_dx, block_dy = G._sym(rng, rk["rack_xy"]), G._sym(rng, rk["rack_xy"])
    block = R.affine(block_rot, (ncols * R.STRIP_W / 2, 0.0), block_dx, block_dy)
    rack_m, tray_info = [], []
    x_acc = 0.0
    for j in range(ncols):
        if j > 0:
            x_acc += float(rng.uniform(0.0, rk["rack_gap"]))
        dy = G._sym(rng, rk["rack_jit_y"])
        rot = G._sym(rng, rk["rack_jit_rot"])
        rack_m.append(block @ R.affine(rot, (R.STRIP_W / 2, 0.0), j * R.STRIP_W + x_acc, dy))
        tray_info.append([G._round(x_acc, 3), G._round(dy, 3), G._round(rot, 3)])

    # cards in grooves: lateral play and how far down the groove they sit
    placements: list[list[list[float]]] = []
    for col in entry["columns"]:
        n = col["n_cards"] + (1 if col["stop_card"] else 0)
        placements.append([[G._tnorm(rng, rk["card_dx"]), -abs(G._tnorm(rng, rk["card_slide"])), 0.0]
                           for _ in range(n)])

    pts = scene_points(rack_m, entry, placements)
    cam, cam_info = G.sample_camera(cfg, lv, ncols, rng, pts["need"])

    # board / mat at the board plane (H_RACK below the markers)
    W, Hh = cfg["size"]
    Hb = cam.H_at(H_RACK)
    Hb_inv = np.linalg.inv(Hb)
    border = np.array([[0, 0], [W, 0], [W, Hh], [0, Hh], [W / 2, 0], [W / 2, Hh], [0, Hh / 2], [W, Hh / 2]], float)
    bp = R.apply_h(Hb_inv, border)
    scene = R.Scene(bp[:, 0].min() - 20, bp[:, 1].min() - 20, bp[:, 0].max() + 20, bp[:, 1].max() + 20)
    mat_kind = ["flat", "wood", "felt"][int(rng.integers(0, 3))]
    R.render_mat(scene, mat_kind, rng)
    s = R.SCENE_PX_PER_MM
    ppm = cam.f_px / (abs(cam.C[2]) + H_RACK)
    sigma_aa = max(0.0, 0.5 * (s / ppm) - 0.35)
    mat_src = cv2.GaussianBlur(scene.img, (0, 0), sigma_aa)
    M = Hb @ np.linalg.inv(scene.mat_to_px())
    ideal = cv2.warpPerspective(mat_src, M, (W, Hh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    del scene, mat_src
    cv = Canvas(cam, ideal)

    # stand foot in front of the racks (top 8 mm above the board)
    foot_rgb = [(60, 60, 60), (40, 110, 200), (230, 230, 230), (30, 30, 30)][int(rng.integers(0, 4))]
    xc = ncols * R.STRIP_W / 2
    zf = H_RACK - FOOT_H
    cv.poly(np.array([_xy(block, xc - FOOT_W / 2, -FOOT_DEPTH, zf), _xy(block, xc + FOOT_W / 2, -FOOT_DEPTH, zf),
                      _xy(block, xc + FOOT_W / 2, 0, zf), _xy(block, xc - FOOT_W / 2, 0, zf)]), foot_rgb)

    wall_bgr = WALL_GREYS[int(rng.integers(0, len(WALL_GREYS)))]
    floor_bgr = FLOOR_COLS[int(rng.integers(0, len(FLOOR_COLS)))]
    overhang = float(rng.uniform(*lv["shadow"]))
    stop_tex, stop_alpha = R.render_stop_card()
    stop_tex = cv2.GaussianBlur(stop_tex, (0, 0), 0.4)
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    # Painter's order across racks: the solid double walls between racks hide
    # the far side, so draw racks from the outside in towards the camera
    # (left of the lens: left to right; right of it: right to left; the rack
    # under the lens last). Within a rack: floor, cards 1..n, the visible part
    # of the camera-facing comb faces, wall tops, header.
    cam_x = float(cam.C[0])
    centres = [float(_xy(m, 35.0, 120.0, 0)[0]) for m in rack_m]
    under = min(range(ncols), key=lambda j: abs(centres[j] - cam_x))
    order = [j for j in range(ncols) if j < under] + [j for j in reversed(range(ncols)) if j > under] + [under]
    for j in order:
        col = entry["columns"][j]
        cards = []
        for slot in col["slots"]:
            u = slot["image_uri"]
            if u not in cache:
                cache[u] = raw_card(sf.image(u))
            cards.append(cache[u])
        paint_rack(cv, rack_m[j], col, placements[j], cards, (stop_tex, stop_alpha), wall_bgr, floor_bgr, overhang)
        # header plate with its sticker, flush with the wall tops
        sticker = cv2.GaussianBlur(render_header_sticker(j + 1), (0, 0), 0.4)
        m = rack_m[j]
        cv.tex(sticker, None, np.array([_xy(m, 3, 0, 0), _xy(m, 67, 0, 0), _xy(m, 67, HEADER_Y1, 0),
                                        _xy(m, 3, HEADER_Y1, 0)]))

    # occluder for retake images: a fingertip / stray card back lying on the header
    occl_info = None
    neg = entry.get("negative")
    if neg and neg["kind"] == "occluded_marker":
        mk = np.vstack([pts["markers"][i] for i in neg["ids"]])
        cx, cy = mk[:, 0].mean(), mk[:, 1].mean()
        span = float(np.ptp(mk[:, 0]))
        kind = "hand" if rng.random() < 0.5 else "card_back"
        if kind == "hand":
            rot = float(rng.uniform(-5, 5))
            w_mm, h_mm = (30.0, 30.0) if len(neg["ids"]) == 1 else (span * 1.25 + 4.0, 34.0)
            tex = np.zeros((int(h_mm * s), int(w_mm * s), 3), np.uint8)
            tex[:] = (120, 150, 205)
            alpha = np.zeros(tex.shape[:2], np.uint8)
            cv2.ellipse(alpha, (tex.shape[1] // 2, tex.shape[0] // 2), (tex.shape[1] // 2 - 2, tex.shape[0] // 2 - 2),
                        0, 0, 360, 255, -1)
        else:
            rot = float(rng.uniform(-8, 8))
            w_mm, h_mm = span + 5.0, 26.0
            tex = np.zeros((int(h_mm * s), int(w_mm * s), 3), np.uint8)
            tex[:] = (40, 70, 110)
            cv2.ellipse(tex, (tex.shape[1] // 2, tex.shape[0] // 2), (tex.shape[1] // 3, tex.shape[0] // 3),
                        0, 0, 360, (60, 110, 170), -1)
            alpha = R._rounded_mask(tex.shape[1], tex.shape[0], 2 * s)
        om = R.affine(rot, (w_mm / 2, h_mm / 2), cx - w_mm / 2, cy - h_mm / 2)
        cv.tex(tex, alpha, np.array([_xy(om, 0, 0, -1.0), _xy(om, w_mm, 0, -1.0), _xy(om, w_mm, h_mm, -1.0),
                                     _xy(om, 0, h_mm, -1.0)]))
        occl_info = kind

    # photometric look (same sampling as flat mode)
    n_glare = int(rng.integers(lv["glare"][0], lv["glare"][1] + 1))
    glare = []
    for _ in range(n_glare):
        glare.append((float(rng.uniform(0, W)), float(rng.uniform(0, Hh)),
                      float(rng.uniform(8, 30) * cam_info["px_per_mm"]), float(rng.uniform(lv["glare"][2], lv["glare"][3]))))
    look = {
        "light_grad": (G._sym(rng, lv["grad"]), G._sym(rng, lv["grad"])),
        "vignette": float(rng.uniform(*lv["vignette"])),
        "glare": glare,
        "exposure": float(rng.uniform(*lv["exposure"])),
        "defocus_sigma_px": float(rng.uniform(*lv["defocus"])),
        "motion_px": float(rng.uniform(*lv["motion"])),
        "motion_angle_deg": float(rng.uniform(0, 180)),
        "read_noise": float(rng.uniform(*lv["read"])),
        "shot_noise": float(rng.uniform(*lv["shot"])),
        "wb_gains_bgr": [1.0 + G._sym(rng, lv["wb"]) for _ in range(3)],
    }
    quality = int(rng.integers(lv["jpeg"][0], lv["jpeg"][1] + 1))
    if DEBUG_STASH is not None:
        DEBUG_STASH["ideal"] = cv.img.copy()
    img = R.photograph(None, cam, look, rng, ideal=cv.img)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    jpeg = buf.tobytes()

    flat = [p for pl in placements for p in pl]
    difficulty = {
        "level": entry["level"], **cam_info,
        "blur_px": G._round(look["defocus_sigma_px"], 3), "motion_px": G._round(look["motion_px"], 3),
        "motion_angle_deg": G._round(look["motion_angle_deg"], 1),
        "card_dx_max_mm": G._round(max((abs(p[0]) for p in flat), default=0.0), 3),
        "card_slide_max_mm": G._round(max((abs(p[1]) for p in flat), default=0.0), 3),
        "tray_block": [G._round(block_dx, 3), G._round(block_dy, 3), G._round(block_rot, 3)],
        "trays": tray_info,
        "light_grad": [G._round(v, 4) for v in look["light_grad"]], "vignette": G._round(look["vignette"], 4),
        "glare": [[G._round(g[0], 1), G._round(g[1], 1), G._round(g[2], 1), G._round(g[3], 3)] for g in glare],
        "exposure": G._round(look["exposure"], 4),
        "read_noise": G._round(look["read_noise"], 3), "shot_noise": G._round(look["shot_noise"], 4),
        "wb_gains_bgr": [G._round(v, 4) for v in look["wb_gains_bgr"]], "jpeg_quality": quality,
        "mat": mat_kind, "overhang_shade": G._round(overhang, 3),
    }
    if occl_info:
        difficulty["occluder"] = occl_info
    aux = {
        "markers": {mid: cam.project(p) for mid, p in pts["markers"].items()},
        "stops": [cam.project(p) for p in pts["stops"]],
        "placements": placements,
        "rack_m": rack_m, "cam": cam,
    }
    return jpeg, difficulty, aux
