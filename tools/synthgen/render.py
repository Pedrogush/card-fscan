"""Scene + camera renderer for synthetic shingle photos.

Coordinate frames (all millimetres unless noted):
  * card-local: origin at the card's top-left corner, 63 x 88.
  * tray/strip: SPEC section 1 strip coordinates (origin strip top-left).
    The tray board is 70 x 310 with the strip glued on its top 240 mm.
  * mat: x across the columns, y down the columns. Nominally strip j's
    origin sits at (70 (j-1), 0).
  * scene raster: the mat rendered at SCENE_PX_PER_MM over a window.
  * image: the final photo pixels after the camera model.

All randomness comes from the numpy Generator passed in; the same inputs give
bit-identical output on the same machine and library versions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---- SPEC section 1 constants ------------------------------------------------
STRIP_W, STRIP_H, TRAY_H = 70.0, 240.0, 310.0
HEADER, PITCH, N_SLOTS = 25.0, 10.0, 20
CARD_W, CARD_H, CARD_X = 63.0, 88.0, 3.5
MARKER_L = (6.0, 3.5)
MARKER_R = (46.0, 3.5)
MARKER_SIZE = 18.0
STOP_ID = 40
STOP_MARKER_XY = (16.5, 3.0)
STOP_MARKER_SIZE = 30.0
CARD_CORNER_R = 3.0

SCENE_PX_PER_MM = 14  # scene raster density; output is ~11-12 px/mm
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

CONFIGS = {
    "C4": {"cols": 4, "size": (4032, 3024), "frame_w_mm": 360.0},
    "C6": {"cols": 6, "size": (5712, 4284), "frame_w_mm": 480.0},
}

INK = (28, 28, 28)
GREY_INK = (140, 140, 140)  # 0x8C8C8C as in the print pack
LIGHT_LINE = (204, 204, 204)


def marker_ids(col: int) -> tuple[int, int]:
    return 2 * (col - 1), 2 * (col - 1) + 1


def marker_corners_strip(col: int) -> dict[int, np.ndarray]:
    """id -> 4x2 corners (TL, TR, BR, BL) in strip mm, per SPEC 1."""
    out = {}
    for mid, (x0, y0) in zip(marker_ids(col), (MARKER_L, MARKER_R)):
        out[mid] = np.array([[x0, y0], [x0 + MARKER_SIZE, y0], [x0 + MARKER_SIZE, y0 + MARKER_SIZE],
                             [x0, y0 + MARKER_SIZE]], dtype=np.float64)
    return out


def _font(size_px: float) -> ImageFont.FreeTypeFont:
    # Pillow's bundled font: identical on every machine with the pinned Pillow.
    return ImageFont.load_default(size=max(4, int(round(size_px))))


def _text_centered(draw: ImageDraw.ImageDraw, cx: float, top: float, text: str, size_px: float,
                   fill: tuple[int, int, int], bold: bool = False) -> None:
    f = _font(size_px)
    sw = max(1, int(size_px / 14)) if bold else 0
    x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=f, stroke_width=sw)
    draw.text((cx - (x1 - x0) / 2 - x0, top - y0), text, font=f, fill=fill, stroke_width=sw, stroke_fill=fill)


def _marker_bitmap(mid: int, size_px: int) -> np.ndarray:
    return cv2.aruco.generateImageMarker(ARUCO_DICT, mid, size_px, borderBits=1)


# ---- printed artwork --------------------------------------------------------

def render_tray(col: int, board_rgb: tuple[int, int, int], paper_rgb: tuple[int, int, int]) -> np.ndarray:
    """Tray board + glued strip for column ``col`` (BGR, SCENE_PX_PER_MM)."""
    s = SCENE_PX_PER_MM
    w, h = int(STRIP_W * s), int(TRAY_H * s)
    img = Image.new("RGB", (w, h), board_rgb)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, w - 1, int(STRIP_H * s) - 1], fill=paper_rgb)

    def line(x0: float, y0: float, x1: float, y1: float, width_mm: float, fill: tuple[int, int, int]) -> None:
        d.line([(x0 * s, y0 * s), (x1 * s, y1 * s)], fill=fill, width=max(1, int(round(width_mm * s))))

    # side guides and slot ticks (from the print pack vectors)
    line(CARD_X, HEADER, CARD_X, STRIP_H, 0.15, LIGHT_LINE)
    line(STRIP_W - CARD_X, HEADER, STRIP_W - CARD_X, STRIP_H, 0.15, LIGHT_LINE)
    for i in range(N_SLOTS):
        y = HEADER + PITCH * i
        if i == 0:
            line(CARD_X, y, STRIP_W - CARD_X, y, 0.5, INK)
        else:
            line(CARD_X, y, STRIP_W - CARD_X, y, 0.12, LIGHT_LINE)
        line(0, y, CARD_X, y, 0.45, INK)
        line(STRIP_W - CARD_X, y, STRIP_W, y, 0.45, INK)
        _text_centered(d, 1.75 * s, (y + 1.6) * s, str(i + 1), 1.9 * s, INK, bold=True)
        _text_centered(d, 68.25 * s, (y + 1.6) * s, str(i + 1), 1.9 * s, INK, bold=True)
    # header text
    _text_centered(d, 35 * s, 6.9 * s, "COL", 3.4 * s, INK, bold=True)
    _text_centered(d, 35 * s, 11.7 * s, str(col), 11.0 * s, INK, bold=True)
    _text_centered(d, 35 * s, 27.6 * s, "CARD 1 TOP EDGE ON THIS LINE", 2.3 * s, GREY_INK, bold=True)
    _text_centered(d, 35 * s, 30.6 * s, "next card 10 mm lower, ON TOP", 2.3 * s, GREY_INK)
    # dashed line + footer
    x = CARD_X
    while x < STRIP_W - CARD_X:
        line(x, 229.0, min(x + 1.0, STRIP_W - CARD_X), 229.0, 0.15, GREY_INK)
        x += 2.0
    _text_centered(d, 35 * s, 230.7 * s, "card 20 body continues past strip end", 2.2 * s, GREY_INK)
    a, b = marker_ids(col)
    _text_centered(d, 35 * s, 236.3 * s, f"Shingle strip v1 | col {col} | ArUco 4x4_50 ids {a}/{b}", 2.0 * s, GREY_INK)

    arr = np.array(img)[:, :, ::-1].copy()  # RGB -> BGR
    msz = int(MARKER_SIZE * s)
    for mid, (x0, y0) in zip(marker_ids(col), (MARKER_L, MARKER_R)):
        m = _marker_bitmap(mid, msz)
        px, py = int(round(x0 * s)), int(round(y0 * s))
        roi = arr[py:py + msz, px:px + msz]
        roi[m == 0] = INK[::-1]
    return arr


def _rounded_mask(w: int, h: int, r: float) -> np.ndarray:
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, w - 1, h - 1], radius=r, fill=255)
    return np.array(mask)


def render_stop_card() -> tuple[np.ndarray, np.ndarray]:
    """Stop card artwork (BGR) and alpha, card-sized, at SCENE_PX_PER_MM."""
    s = SCENE_PX_PER_MM
    w, h = int(CARD_W * s), int(CARD_H * s)
    img = Image.new("RGB", (w, h), (244, 244, 240))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, w - 1, h - 1], radius=CARD_CORNER_R * s, outline=(150, 150, 150),
                        width=max(1, int(0.2 * s)))
    _text_centered(d, CARD_W / 2 * s, 38.2 * s, "FIM", 13.0 * s, INK, bold=True)
    _text_centered(d, CARD_W / 2 * s, 56.2 * s, "END OF COLUMN", 4.9 * s, INK, bold=True)
    _text_centered(d, CARD_W / 2 * s, 67.5 * s, "stop card | ArUco 4x4_50 id 40", 3.1 * s, GREY_INK)
    _text_centered(d, CARD_W / 2 * s, 72.5 * s, "marker must stay at the TOP", 3.1 * s, GREY_INK)
    arr = np.array(img)[:, :, ::-1].copy()
    msz = int(STOP_MARKER_SIZE * s)
    m = _marker_bitmap(STOP_ID, msz)
    px, py = int(round(STOP_MARKER_XY[0] * s)), int(round(STOP_MARKER_XY[1] * s))
    arr[py:py + msz, px:px + msz][m == 0] = INK[::-1]
    return arr, _rounded_mask(w, h, CARD_CORNER_R * s)


def prepare_card(jpeg: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Decode a Scryfall image and scale it to card size at SCENE_PX_PER_MM."""
    s = SCENE_PX_PER_MM
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("cannot decode card image")
    w, h = int(CARD_W * s), int(CARD_H * s)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
    return img, _rounded_mask(w, h, 2.6 * s)


# ---- geometry helpers --------------------------------------------------------

def affine(rot_deg: float, pivot: tuple[float, float], tx: float, ty: float) -> np.ndarray:
    """3x3: rotate about ``pivot`` (local coords) then place pivot at pivot+(tx,ty)."""
    c, s_ = math.cos(math.radians(rot_deg)), math.sin(math.radians(rot_deg))
    px, py = pivot
    m = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]], dtype=np.float64)
    pre = np.array([[1, 0, -px], [0, 1, -py], [0, 0, 1]], dtype=np.float64)
    post = np.array([[1, 0, px + tx], [0, 1, py + ty], [0, 0, 1]], dtype=np.float64)
    return post @ m @ pre


def apply_h(m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    p = np.hstack([pts, np.ones((len(pts), 1))]) @ m.T
    return p[:, :2] / p[:, 2:3]


@dataclass
class Placed:
    """A rectangular raster placed on the mat via a mm->mm affine."""
    img: np.ndarray            # BGR at SCENE_PX_PER_MM in its local frame
    alpha: np.ndarray | None   # uint8 or None (opaque)
    local_to_mat: np.ndarray   # 3x3, local mm -> mat mm
    shadow: float = 0.0        # strength of the thin contact shadow


@dataclass
class Scene:
    """Mat raster covering [x0, x1] x [y0, y1] mm."""
    x0: float
    y0: float
    x1: float
    y1: float
    img: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        s = SCENE_PX_PER_MM
        self.w = int(math.ceil((self.x1 - self.x0) * s))
        self.h = int(math.ceil((self.y1 - self.y0) * s))

    def mat_to_px(self) -> np.ndarray:
        s = SCENE_PX_PER_MM
        return np.array([[s, 0, -self.x0 * s], [0, s, -self.y0 * s], [0, 0, 1]], dtype=np.float64)

    def paste(self, p: Placed) -> None:
        s = SCENE_PX_PER_MM
        # local px -> local mm -> mat mm -> scene px
        m = self.mat_to_px() @ p.local_to_mat @ np.diag([1 / s, 1 / s, 1.0])
        h, w = p.img.shape[:2]
        corners = apply_h(m, np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64))
        bx0 = max(0, int(math.floor(corners[:, 0].min())) - 4)
        by0 = max(0, int(math.floor(corners[:, 1].min())) - 4)
        bx1 = min(self.w, int(math.ceil(corners[:, 0].max())) + 4)
        by1 = min(self.h, int(math.ceil(corners[:, 1].max())) + 4)
        if bx1 <= bx0 or by1 <= by0:
            return
        roi_m = np.array([[1, 0, -bx0], [0, 1, -by0], [0, 0, 1]], dtype=np.float64) @ m
        size = (bx1 - bx0, by1 - by0)
        src = cv2.warpAffine(p.img, roi_m[:2], size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        a_src = p.alpha if p.alpha is not None else np.full((h, w), 255, np.uint8)
        a = cv2.warpAffine(a_src, roi_m[:2], size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                           borderValue=0).astype(np.float32) / 255.0
        roi = self.img[by0:by1, bx0:bx1].astype(np.float32)
        if p.shadow > 0:
            # thin contact shadow just outside the edge (strongest above it)
            sh = cv2.warpAffine(a_src, (np.array([[1, 0, 0], [0, 1, -0.35 * s], [0, 0, 1]]) @ roi_m)[:2], size,
                                flags=cv2.INTER_LINEAR, borderValue=0).astype(np.float32) / 255.0
            sh = cv2.GaussianBlur(sh, (0, 0), 0.3 * s)
            roi *= (1.0 - p.shadow * sh)[:, :, None]
        roi = roi * (1.0 - a[:, :, None]) + src.astype(np.float32) * a[:, :, None]
        self.img[by0:by1, bx0:bx1] = np.clip(roi + 0.5, 0, 255).astype(np.uint8)


# ---- mat backgrounds ---------------------------------------------------------

def render_mat(scene: Scene, kind: str, rng: np.random.Generator) -> None:
    h, w = scene.h, scene.w
    if kind == "flat":
        base = rng.uniform(40, 200, 3)
        tint = np.array(base, dtype=np.float32)
        noise = cv2.resize(rng.standard_normal((max(2, h // 40), max(2, w // 40))).astype(np.float32), (w, h),
                           interpolation=cv2.INTER_LINEAR)
        img = tint[None, None, :] + 6.0 * noise[:, :, None]
    elif kind == "wood":
        base = np.array([rng.uniform(40, 80), rng.uniform(80, 120), rng.uniform(130, 175)], np.float32)  # BGR
        ys = np.arange(h, dtype=np.float32)[:, None]
        xs = np.arange(w, dtype=np.float32)[None, :]
        freq = rng.uniform(0.004, 0.008)
        warp = cv2.resize(rng.standard_normal((max(2, h // 300), max(2, w // 300))).astype(np.float32), (w, h),
                          interpolation=cv2.INTER_CUBIC)
        angle = rng.uniform(-0.15, 0.15)
        grain = np.sin((ys + angle * xs) * freq * 2 * np.pi + 6.0 * warp)
        fine = cv2.resize(rng.standard_normal((h // 4 + 1, max(2, w // 60))).astype(np.float32), (w, h),
                          interpolation=cv2.INTER_LINEAR)
        lum = 1.0 + 0.10 * grain + 0.05 * fine
        img = base[None, None, :] * lum[:, :, None]
    else:  # grey felt / cutting mat
        base = rng.uniform(70, 150)
        noise = rng.standard_normal((h // 2 + 1, w // 2 + 1)).astype(np.float32)
        noise = cv2.resize(cv2.GaussianBlur(noise, (0, 0), 1.0), (w, h), interpolation=cv2.INTER_LINEAR)
        hue = rng.uniform(-8, 8, 3).astype(np.float32)
        img = (base + hue)[None, None, :] + 10.0 * noise[:, :, None]
    scene.img = np.clip(img, 0, 255).astype(np.uint8)


# ---- camera ------------------------------------------------------------------

@dataclass
class Camera:
    width: int
    height: int
    f_px: float
    R: np.ndarray       # world(mat) -> camera rotation
    C: np.ndarray       # camera centre in mat coords (z negative = above the mat)
    k1: float

    @property
    def H(self) -> np.ndarray:
        """Homography mat mm -> ideal (undistorted) pixel."""
        K = np.array([[self.f_px, 0, self.width / 2], [0, self.f_px, self.height / 2], [0, 0, 1]])
        t = -self.R @ self.C
        return K @ np.column_stack([self.R[:, 0], self.R[:, 1], t])

    def H_at(self, z: float) -> np.ndarray:
        """Homography for the plane Z = z (Z points away from the camera, i.e.
        z > 0 is below the reference plane) -> ideal pixel."""
        K = np.array([[self.f_px, 0, self.width / 2], [0, self.f_px, self.height / 2], [0, 0, 1]])
        t = -self.R @ self.C + self.R[:, 2] * z
        return K @ np.column_stack([self.R[:, 0], self.R[:, 1], t])

    def project_ideal3d(self, pts: np.ndarray) -> np.ndarray:
        """3D points (X, Y, Z down) -> ideal (undistorted) pixels."""
        P = (np.asarray(pts, np.float64).reshape(-1, 3) - self.C) @ self.R.T
        return np.column_stack([self.f_px * P[:, 0] / P[:, 2] + self.width / 2,
                                self.f_px * P[:, 1] / P[:, 2] + self.height / 2])

    def project(self, pts_mat: np.ndarray) -> np.ndarray:
        """Mat mm (Nx2, on the reference plane) or 3D points (Nx3) -> distorted image pixels."""
        pts_mat = np.asarray(pts_mat, dtype=np.float64)
        if pts_mat.ndim == 2 and pts_mat.shape[1] == 3:
            ideal = self.project_ideal3d(pts_mat)
        else:
            ideal = apply_h(self.H, pts_mat)
        cx, cy, f = self.width / 2, self.height / 2, self.f_px
        x = (ideal[:, 0] - cx) / f
        y = (ideal[:, 1] - cy) / f
        r2 = x * x + y * y
        d = 1 + self.k1 * r2
        return np.column_stack([x * d * f + cx, y * d * f + cy])

    def undistort_grid(self, rows: slice) -> tuple[np.ndarray, np.ndarray]:
        """Distorted pixel grid rows -> ideal (undistorted) pixel coords (float64)."""
        cx, cy, f = self.width / 2, self.height / 2, self.f_px
        v = np.arange(rows.start, rows.stop, dtype=np.float64)[:, None]
        u = np.arange(self.width, dtype=np.float64)[None, :]
        xd = np.broadcast_to((u - cx) / f, (len(v), self.width))
        yd = np.broadcast_to((v - cy) / f, (len(v), self.width))
        xu, yu = xd.copy(), yd.copy()
        if self.k1 != 0.0:
            for _ in range(6):
                d = 1 + self.k1 * (xu * xu + yu * yu)
                xu = xd / d
                yu = yd / d
        return xu * f + cx, yu * f + cy

    def unproject_grid(self, rows: slice) -> tuple[np.ndarray, np.ndarray]:
        """Distorted pixel grid rows -> mat mm (float64)."""
        cx, cy, f = self.width / 2, self.height / 2, self.f_px
        v = np.arange(rows.start, rows.stop, dtype=np.float64)[:, None]
        u = np.arange(self.width, dtype=np.float64)[None, :]
        xd = np.broadcast_to((u - cx) / f, (len(v), self.width))
        yd = np.broadcast_to((v - cy) / f, (len(v), self.width))
        xu, yu = xd.copy(), yd.copy()
        if self.k1 != 0.0:
            for _ in range(6):
                d = 1 + self.k1 * (xu * xu + yu * yu)
                xu = xd / d
                yu = yd / d
        Hi = np.linalg.inv(self.H)
        px = xu * f + cx
        py = yu * f + cy
        X = Hi[0, 0] * px + Hi[0, 1] * py + Hi[0, 2]
        Y = Hi[1, 0] * px + Hi[1, 1] * py + Hi[1, 2]
        Z = Hi[2, 0] * px + Hi[2, 1] * py + Hi[2, 2]
        return X / Z, Y / Z


def rot_matrix(tilt_deg: float, tilt_dir_deg: float, yaw_deg: float) -> np.ndarray:
    """Camera rotation: yaw about the optical axis, then a tilt of ``tilt_deg``
    about a horizontal axis at azimuth ``tilt_dir_deg``."""
    def rz(a: float) -> np.ndarray:
        c, s = math.cos(a), math.sin(a)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def rx(a: float) -> np.ndarray:
        c, s = math.cos(a), math.sin(a)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    phi = math.radians(tilt_dir_deg)
    tilt = rz(phi) @ rx(math.radians(tilt_deg)) @ rz(-phi)
    return rz(math.radians(yaw_deg)) @ tilt


def scene_bounds(cam: Camera, margin_mm: float = 15.0) -> tuple[float, float, float, float]:
    w, h = cam.width, cam.height
    border = []
    for t in np.linspace(0, 1, 17):
        border += [(t * w, 0), (t * w, h), (0, t * h), (w, t * h)]
    border_arr = np.array(border)
    # invert the distortion approximately via the grid routine on single points
    X, Y = [], []
    for u, v in border_arr:
        cx, cy, f = w / 2, h / 2, cam.f_px
        xd, yd = (u - cx) / f, (v - cy) / f
        xu, yu = xd, yd
        for _ in range(6):
            d = 1 + cam.k1 * (xu * xu + yu * yu)
            xu, yu = xd / d, yd / d
        p = apply_h(np.linalg.inv(cam.H), np.array([[xu * f + cx, yu * f + cy]]))[0]
        X.append(p[0])
        Y.append(p[1])
    return min(X) - margin_mm, min(Y) - margin_mm, max(X) + margin_mm, max(Y) + margin_mm


def photograph(scene: Scene | None, cam: Camera, look: dict[str, Any], rng: np.random.Generator,
               ideal: np.ndarray | None = None) -> np.ndarray:
    """Project the scene through the camera and apply the photometric model.

    Flat mode passes the mat ``scene``; rack mode passes ``ideal``, an already
    perspective-rendered undistorted image, and only lens distortion is applied
    geometrically. ``look`` holds already-sampled parameters."""
    s = SCENE_PX_PER_MM
    if ideal is not None:
        src = ideal
    else:
        assert scene is not None
        ppm = cam.f_px / abs(cam.C[2])
        # anti-alias the scene for the ~0.8x downsample
        sigma_aa = max(0.0, 0.5 * (s / ppm) - 0.35)
        src = cv2.GaussianBlur(scene.img, (0, 0), sigma_aa) if sigma_aa > 0.05 else scene.img
    W, H = cam.width, cam.height
    out = np.empty((H, W, 3), np.uint8)

    # lighting field parameters
    gx, gy = look["light_grad"]
    vig = look["vignette"]
    glare = look["glare"]  # list of (u, v, radius_px, strength)
    exposure = look["exposure"]
    uu = (np.arange(W, dtype=np.float32) - W / 2) / (W / 2)
    step = 256
    for r0 in range(0, H, step):
        rows = slice(r0, min(H, r0 + step))
        if ideal is not None:
            px, py = cam.undistort_grid(rows)
            mx, my = px.astype(np.float32), py.astype(np.float32)
        else:
            X, Y = cam.unproject_grid(rows)
            mx = ((X - scene.x0) * s).astype(np.float32)
            my = ((Y - scene.y0) * s).astype(np.float32)
        chunk = cv2.remap(src, mx, my, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        vv = ((np.arange(rows.start, rows.stop, dtype=np.float32) - H / 2) / (H / 2))[:, None]
        r2 = (uu[None, :] ** 2 * (W / H) ** 2 + vv ** 2) / (1 + (W / H) ** 2) * 2  # ~1 at the corners
        gain = exposure * (1 + gx * uu[None, :] + gy * vv) * (1 - vig * r2 / 2)
        f = chunk.astype(np.float32) * gain[:, :, None]
        if glare:
            add = np.zeros(gain.shape, np.float32)
            ug = np.arange(W, dtype=np.float32)[None, :]
            vg = np.arange(rows.start, rows.stop, dtype=np.float32)[:, None]
            for (gu, gv, rad, st) in glare:
                add += st * np.exp(-((ug - gu) ** 2 + (vg - gv) ** 2) / (2 * rad * rad))
            add = np.minimum(add, 0.95)
            f = f + (255.0 - f) * add[:, :, None]
        out[rows] = np.clip(f + 0.5, 0, 255).astype(np.uint8)

    # optics: defocus + motion blur
    if look["defocus_sigma_px"] > 0.05:
        out = cv2.GaussianBlur(out, (0, 0), look["defocus_sigma_px"])
    L = look["motion_px"]
    if L >= 0.5:
        k = int(math.ceil(L)) * 2 + 1
        ker = np.zeros((k, k), np.float32)
        c = k // 2
        ang = math.radians(look["motion_angle_deg"])
        for t in np.linspace(-L / 2, L / 2, max(3, int(L * 4))):
            x, y = c + t * math.cos(ang), c + t * math.sin(ang)
            x0, y0 = int(math.floor(x)), int(math.floor(y))
            fx, fy = x - x0, y - y0
            ker[y0, x0] += (1 - fx) * (1 - fy)
            ker[y0, min(k - 1, x0 + 1)] += fx * (1 - fy)
            ker[min(k - 1, y0 + 1), x0] += (1 - fx) * fy
            ker[min(k - 1, y0 + 1), min(k - 1, x0 + 1)] += fx * fy
        ker /= ker.sum()
        out = cv2.filter2D(out, -1, ker, borderType=cv2.BORDER_REFLECT)

    # sensor noise + white balance, in row chunks
    wb = np.array(look["wb_gains_bgr"], np.float32)
    shot, read = look["shot_noise"], look["read_noise"]
    for r0 in range(0, H, step):
        rows = slice(r0, min(H, r0 + step))
        f = out[rows].astype(np.float32)
        n = rng.standard_normal(f.shape, dtype=np.float32)
        f = f + n * np.sqrt(read * read + shot * f)
        f = f * wb[None, None, :]
        out[rows] = np.clip(f + 0.5, 0, 255).astype(np.uint8)
    return out
