"""Tiny z-buffer rasterizer + the scenes for the Shingle Scanner preview renders."""
import math
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import shingle_cad as sc

rng = np.random.default_rng(7)


# ------------------------------------------------------------------ renderer
class Camera:
    def __init__(self, eye, target, up=(0, 0, 1), fov_deg=35, size=(1600, 1200), ss=2, ortho=None):
        self.eye = np.array(eye, float)
        f = np.array(target, float) - self.eye
        self.fwd = f / np.linalg.norm(f)
        r = np.cross(self.fwd, up)
        self.right = r / np.linalg.norm(r)
        self.up = np.cross(self.right, self.fwd)
        self.W, self.H = size[0] * ss, size[1] * ss
        self.size, self.ss = size, ss
        self.fpx = (self.W / 2) / math.tan(math.radians(fov_deg) / 2)
        self.ortho = ortho

    def project(self, P):
        d = P - self.eye
        xc, yc, zc = d @ self.right, d @ self.up, d @ self.fwd
        if self.ortho:
            k = self.W / self.ortho
            return self.W / 2 + xc * k, self.H / 2 - yc * k, zc
        return self.W / 2 + self.fpx * xc / zc, self.H / 2 - self.fpx * yc / zc, zc


def render(meshes, cam, bg_top=(0.93, 0.94, 0.96), bg_bot=(0.80, 0.82, 0.86), outline=True,
           lights=((-0.4, -0.6, 1.0), (0.7, 0.3, 0.4))):
    W, H = cam.W, cam.H
    inv = np.zeros((H, W))                      # 1/depth (perspective) or -depth (ortho); bigger = closer
    inv[:] = -np.inf
    col = np.zeros((H, W, 3))
    nrm = np.zeros((H, W, 3))
    L = [np.array(l) / np.linalg.norm(l) for l in lights]
    for m in meshes:
        T = m.tris
        if len(T) == 0:
            continue
        P = T.reshape(-1, 3)
        sx, sy, zc = cam.project(P)
        sx, sy, zc = sx.reshape(-1, 3), sy.reshape(-1, 3), zc.reshape(-1, 3)
        N = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
        Nl = np.linalg.norm(N, axis=1, keepdims=True)
        N = N / np.where(Nl == 0, 1, Nl)
        toward = cam.eye - T.mean(1) if not cam.ortho else -cam.fwd[None, :].repeat(len(T), 0)
        N[np.einsum("ij,ij->i", N, toward) < 0] *= -1
        shade = 0.42 + 0.50 * np.clip(N @ L[0], 0, 1) + 0.18 * np.clip(N @ L[1], 0, 1)
        base = np.array(m.color[:3])
        for i in range(len(T)):
            if not cam.ortho and zc[i].min() < 15:
                continue
            x, y, z = sx[i], sy[i], zc[i]
            x0, x1 = int(max(0, math.floor(x.min()))), int(min(W - 1, math.ceil(x.max())))
            y0, y1 = int(max(0, math.floor(y.min()))), int(min(H - 1, math.ceil(y.max())))
            if x0 > x1 or y0 > y1:
                continue
            den = (y[1] - y[2]) * (x[0] - x[2]) + (x[2] - x[1]) * (y[0] - y[2])
            if abs(den) < 1e-9:
                continue
            gy, gx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
            px, py = gx + 0.5, gy + 0.5
            w0 = ((y[1] - y[2]) * (px - x[2]) + (x[2] - x[1]) * (py - y[2])) / den
            w1 = ((y[2] - y[0]) * (px - x[2]) + (x[0] - x[2]) * (py - y[2])) / den
            w2 = 1 - w0 - w1
            ins = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
            if not ins.any():
                continue
            if cam.ortho:
                q = -(w0 * z[0] + w1 * z[1] + w2 * z[2])
            else:
                q = w0 / z[0] + w1 / z[1] + w2 / z[2]
            sub = inv[y0:y1 + 1, x0:x1 + 1]
            win = ins & (q > sub)
            if not win.any():
                continue
            sub[win] = q[win]
            col[y0:y1 + 1, x0:x1 + 1][win] = base * shade[i]
            nrm[y0:y1 + 1, x0:x1 + 1][win] = N[i]
    empty = ~np.isfinite(inv)
    t = np.linspace(0, 1, H)[:, None, None]
    bg = np.array(bg_top) * (1 - t) + np.array(bg_bot) * t
    img = np.where(empty[..., None], np.broadcast_to(bg, (H, W, 3)), col)
    if outline:
        depth = np.where(empty, 1e9, (-inv if cam.ortho else 1 / np.where(empty, 1, inv)))
        edge = np.zeros((H, W), bool)
        for ax in (0, 1):
            dz = np.abs(np.roll(depth, -1, ax) - 2 * depth + np.roll(depth, 1, ax))
            edge |= dz > (0.6 + 0.0025 * np.minimum(depth, 5000))
            dn = 1 - np.sum(nrm * np.roll(nrm, -1, ax), axis=2)
            edge |= (dn > 0.12) & ~empty & ~np.roll(empty, -1, ax)
            edge |= empty != np.roll(empty, -1, ax)
        img[edge] = img[edge] * 0.25
    im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    return im.resize(cam.size, Image.LANCZOS)


def draw_lines(im, cam, segs, color=(200, 40, 40), width=3, dash=None):
    d = ImageDraw.Draw(im)
    s = 1 / cam.ss
    for a, b in segs:
        a, b = np.array(a, float), np.array(b, float)
        n = 60
        pts = []
        for i in range(n + 1):
            p = a + (b - a) * i / n
            x, y, z = cam.project(p[None, :])
            pts.append((x[0] * s, y[0] * s))
        for i in range(n):
            if dash and (i // dash) % 2:
                continue
            d.line([pts[i], pts[i + 1]], fill=color, width=width)


def label(im, xy, text, size=30, fill=(30, 30, 30), anchor="la", box=True):
    f = None
    for name in ("DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "arialbd.ttf"):
        try:
            f = ImageFont.truetype(name, size)
            break
        except OSError:
            pass
    f = f or ImageFont.load_default()
    d = ImageDraw.Draw(im)
    if box:
        bb = d.textbbox(xy, text, font=f, anchor=anchor)
        d.rounded_rectangle([bb[0] - 10, bb[1] - 6, bb[2] + 10, bb[3] + 6], 8, fill=(255, 255, 255))
    d.text(xy, text, font=f, fill=fill, anchor=anchor)


# ------------------------------------------------------------------ scene pieces
PLA = (0.72, 0.76, 0.80)
CARRIAGE = (0.95, 0.55, 0.20)
WALL = (0.42, 0.47, 0.54)
MDF = (0.78, 0.64, 0.46)
BOARD = (0.86, 0.77, 0.62)
STEEL = (0.55, 0.56, 0.58)
PHONE = (0.12, 0.12, 0.14)
FRAMES = {"W": (0.93, 0.90, 0.80), "U": (0.25, 0.45, 0.75), "B": (0.30, 0.28, 0.30),
          "R": (0.78, 0.25, 0.20), "G": (0.25, 0.55, 0.30), "M": (0.82, 0.68, 0.30)}
ART = [(0.45, 0.55, 0.62), (0.55, 0.45, 0.38), (0.35, 0.45, 0.35), (0.60, 0.52, 0.45), (0.40, 0.38, 0.50)]


def rack_to_world(m, col_x):
    return m.moved((col_x, sc.RACK_Y0, 0))


def quad_local(s0, s1, t0, t1, n):
    a, b, c, d = [s0, t0, n], [s1, t0, n], [s1, t1, n], [s0, t1, n]
    return np.array([[a, b, c], [a, c, d]], float)


def card_meshes(k, frame_rgb, col_x, slide=0.0):
    M, off = sc.card_frame(k, slide)
    M = M @ np.diag([1, -1, 1])     # card face read upright from the camera side: name on the +x... reader's left
    hw = sc.CARD_W / 2
    out = []

    def add(tris, color):
        m = sc.Mesh(tris, color).transformed(M, off)
        out.append(rack_to_world(m, col_x))
    add(sc.box(0, sc.CARD_L, -hw, hw, -sc.CARD_T, 0), (0.08, 0.08, 0.09))          # black border
    add(quad_local(2.5, 85.5, -28.8, 28.8, 0.03), frame_rgb)                       # frame
    light = tuple(min(1, 0.55 + 0.45 * c) for c in frame_rgb)
    add(quad_local(3.6, 9.2, -27.5, 27.5, 0.06), light)                           # name bar
    L = rng.uniform(16, 40)
    add(quad_local(5.2, 7.6, -25.5, -25.5 + L, 0.09), (0.10, 0.10, 0.10))         # name text
    for j in range(int(rng.integers(1, 4))):                                       # mana symbols
        add(quad_local(5.0, 7.8, 24.5 - j * 3.6, 21.7 - j * 3.6 + 0.0, 0.09), (0.2, 0.2, 0.2))
    add(quad_local(10.3, 46.5, -26.5, 26.5, 0.06), ART[int(rng.integers(len(ART)))])  # art
    add(quad_local(47.5, 52.5, -27.5, 27.5, 0.06), light)                         # type line
    add(quad_local(53.5, 80.0, -27, 27, 0.06), (0.92, 0.89, 0.82))                # text box
    return out


def marker_decal(mid, x0, y0, size, z, color_scale=1.0):
    bits = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), mid, 6)
    cell = size / 6
    tris_b, tris_w = [], []
    for r in range(6):
        for c in range(6):
            x, y = x0 + size - (c + 1) * cell, y0 + r * cell   # camera's image-right is world -x
            q = np.array([[[x, y, z], [x + cell, y, z], [x + cell, y + cell, z]],
                          [[x, y, z], [x + cell, y + cell, z], [x, y + cell, z]]])
            (tris_b if bits[r, c] == 0 else tris_w).append(q)
    return [sc.Mesh(np.concatenate(tris_b), (0.05, 0.05, 0.05)),
            sc.Mesh(np.concatenate(tris_w), (0.97, 0.97, 0.97))]


def rack_meshes(j, col_x, colour="W", n_cards=20, hide_left=False, slide_card=None):
    out = []
    if not hide_left:
        out.append(rack_to_world(sc.Mesh(sc.wall_local(-1).tris, WALL), col_x))
    out.append(rack_to_world(sc.Mesh(sc.wall_local(+1).tris, WALL), col_x))
    out.append(rack_to_world(sc.Mesh(sc.floor_local().tris, MDF), col_x))
    out.append(rack_to_world(sc.Mesh(sc.header_local().tris, (0.97, 0.97, 0.95)), col_x))
    zt = sc.H + 0.05
    # SPEC 1: markers at strip x 6..24 / 46..64, y 3.5..21.5 (u = y - 25). Rack-local x = 35 - strip x
    # (the camera's image-right is world -x).
    y0 = 3.5 - 25.0
    for m in marker_decal(2 * (j - 1), 11, y0, 18, zt) + marker_decal(2 * (j - 1) + 1, -29, y0, 18, zt):
        out.append(rack_to_world(m, col_x))
    for k in range(1, n_cards + 1):
        out += card_meshes(k, FRAMES[colour], col_x)
    if slide_card:
        out += card_meshes(slide_card[0], FRAMES[colour], col_x, slide=slide_card[1])
    return out


def stand_meshes(h, with_phone=True, with_board=True):
    out = []
    if with_board:
        out.append(sc.Mesh(sc.box(-300, 300, -75, 375, -12, 0), BOARD))
    out.append(sc.Mesh(sc.foot(), PLA))
    for i in (1, 2, 3):
        out.append(sc.Mesh(sc.post_segment(i), PLA if i != 2 else (0.66, 0.71, 0.76)))
    out.append(sc.Mesh(sc.carriage(h), CARRIAGE))
    zp = sc.pin_z(h)
    out.append(sc.Mesh(sc.box(-30, 30, -2.4, 2.4, zp - 2.4, zp + 2.4), STEEL))
    out.append(sc.Mesh(sc.fence(18.0, h), CARRIAGE))
    if with_phone:
        top = zp + sc.PIN_TO_PLATE
        # phone lying screen up in landscape, main lens 18 mm from its long edge (against the fence)
        # and 16 mm from its end, over the window centre
        lx, ly = 0.0, sc.ARM_REACH
        out.append(sc.Mesh(sc.box(lx - 16, lx + 144, ly - 18, ly + 57, top, top + 8.5), PHONE))
        out.append(sc.Mesh(sc.box(lx - 14, lx + 142, ly - 16, ly + 55, top + 8.5, top + 8.6), (0.20, 0.30, 0.45)))
    return out


def lens_point(h):
    return np.array([0.0, sc.ARM_REACH, sc.pin_z(h) + sc.PIN_TO_PLATE])


COLOURS = ["W", "U", "B", "R", "G", "M"]


def station_racks(n_cols=6, partial=None):
    out = []
    for j in range(1, n_cols + 1):
        cx = ((n_cols + 1) / 2 - j) * sc.RACK_W   # column 1 on the operator's left
        n = 20 if not partial or j != n_cols else partial
        out += rack_meshes(j, cx, COLOURS[j - 1], n_cards=n)
    return out
