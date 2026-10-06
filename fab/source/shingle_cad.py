"""Shingle Scanner v2 hardware: camera stand + tilted-slot card racks.

One source of truth for (a) printable STL parts and (b) preview renders.
Units: mm. World: X across columns, Y along columns (away from the post), Z up,
origin at the post centre on the baseboard surface.
"""
import math
import struct
import numpy as np

# ------------------------------------------------------------------ parameters
# card + rack
CARD_W, CARD_L, CARD_T = 63.0, 88.0, 0.3
N_SLOTS, PITCH = 20, 10.0          # cards per rack, offset between name bars
THETA = math.radians(20)           # card tilt (gravity-loading vs. text foreshortening)
GROOVE_LO, GROOVE_HI = -0.65, 0.35 # groove band around each card plane (card sits in [-0.3, 0])
FLOOR_T = 3.0                      # floor plate (3 mm MDF/hardboard or printed)
RACK_W = 70.0                      # = column pitch; racks butt edge to edge
SLAB_T, COMB_T = 3.0, 1.5          # wall: outer slab + grooved comb layer
HEADER_T = 3.0                     # marker header plate thickness
# SPEC 1: strip y = u + 25. Rack front = strip y 0, so the header markers land at the spec's
# y 3.5..21.5 (u -21.5..-3.5) and the frame centre (strip y 120) sits under the lens.
U_FRONT, U_BACK = -25.0, 277.0     # rack extent along the column (u = 0: card 1 top edge)
U_HEADER_END = -0.5                # header plate runs right up to card 1 (white margin below the markers)
U_FLOOR = 78.0                     # floor plate starts here
S, C = math.sin(THETA), math.cos(THETA)
H = FLOOR_T + CARD_L * S + CARD_T * C + 0.05   # all name-bar top edges sit at this height
X_SLAB_IN = RACK_W / 2 - SLAB_T            # 32.0
X_COMB_IN = X_SLAB_IN - COMB_T             # 30.5

# stand
MAT = round(H)                    # engraved heights are lens height above the rack tops
PIN_TO_PLATE = 25.0
H_MIN, H_MAX, H_STEP = 230, 400, 10
POST_W, CLEAR, PIN_HOLE = 30.0, 0.3, 5.4
JOINTS, POST_TOP = (135.0, 275.0), 425.0
PEG_W, PEG_LEN, PEG_CLEAR = 14.0, 20.0, 0.2
ARM_REACH, FOOT_FRONT = 160.0, 40.0
SLEEVE_WALL, SLEEVE_H = 6.0, 2 * PIN_TO_PLATE
PLATE_T, WINDOW, ARM_W, ARM_T = 5.0, 70.0, 30.0, 12.0
# Phone plate, relative to the lens axis (x across columns, y along them). The phone lies screen up
# in LANDSCAPE (its long side across the columns = the frame's long side), lens over the window,
# body extending towards +x (the column-1 side). A slotted fence registers the phone's long edge.
PLATE_X0, PLATE_X1, PLATE_Y0, PLATE_Y1 = -45.0, 150.0, -45.0, 65.0
FENCE_W, FENCE_H, FENCE_X0, FENCE_X1 = 6.0, 8.0, -40.0, 140.0
FENCE_D_MIN, FENCE_D_MAX = 10.0, 28.0   # lens centre to phone long edge (adjustable range)
FENCE_SLOT_X, M3 = (60.0, 125.0), 3.4
RACK_Y0 = FOOT_FRONT - U_FRONT    # world Y of u = 0 (rack front butts the foot)


def pin_z(h):
    return h - PIN_TO_PLATE + MAT


# ------------------------------------------------------------------ mesh basics
class Mesh:
    def __init__(self, tris=None, color=(0.7, 0.7, 0.7), name=""):
        self.tris = np.zeros((0, 3, 3)) if tris is None else np.asarray(tris, float).reshape(-1, 3, 3)
        self.color, self.name = color, name

    def transformed(self, M, off=(0, 0, 0)):
        M = np.asarray(M, float)
        t = self.tris @ M.T + np.asarray(off, float)
        if np.linalg.det(M) < 0:
            t = t[:, ::-1]
        return Mesh(t, self.color, self.name)

    def moved(self, off):
        return self.transformed(np.eye(3), off)

    def __add__(self, other):
        return Mesh(np.concatenate([self.tris, other.tris]), self.color, self.name)


def box(x0, x1, y0, y1, z0, z1):
    v = np.array([[x, y, z] for z in (z0, z1) for y in (y0, y1) for x in (x0, x1)], float)
    quads = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
    t = []
    for a, b, c, d in quads:
        t += [[v[a], v[b], v[c]], [v[a], v[c], v[d]]]
    return np.array(t)


def rect_solid(adds, subs=()):
    """Watertight mesh of (union of add boxes) minus (union of sub boxes), all axis aligned."""
    allb = list(adds) + list(subs)
    xs = sorted({b[0] for b in allb} | {b[1] for b in allb})
    ys = sorted({b[2] for b in allb} | {b[3] for b in allb})
    zs = sorted({b[4] for b in allb} | {b[5] for b in allb})
    cx = (np.array(xs[:-1]) + np.array(xs[1:])) / 2
    cy = (np.array(ys[:-1]) + np.array(ys[1:])) / 2
    cz = (np.array(zs[:-1]) + np.array(zs[1:])) / 2
    X, Y, Z = np.meshgrid(cx, cy, cz, indexing="ij")

    def inside(b):
        return (X > b[0]) & (X < b[1]) & (Y > b[2]) & (Y < b[3]) & (Z > b[4]) & (Z < b[5])
    occ = np.zeros(X.shape, bool)
    for b in adds:
        occ |= inside(b)
    for b in subs:
        occ &= ~inside(b)
    grids = [np.array(xs), np.array(ys), np.array(zs)]
    tris = []
    for ax in range(3):
        pad = [(0, 0)] * 3
        pad[ax] = (1, 1)
        o = np.pad(occ, pad)
        lo = np.take(o, range(0, o.shape[ax] - 1), axis=ax)
        hi = np.take(o, range(1, o.shape[ax]), axis=ax)
        for idx in np.argwhere(lo != hi):
            outward = 1 if lo[tuple(idx)] else -1        # solid on the low side -> normal +ax
            a1, a2 = [a for a in range(3) if a != ax]
            p = grids[ax][idx[ax]]
            u0, u1 = grids[a1][idx[a1]], grids[a1][idx[a1] + 1]
            w0, w1 = grids[a2][idx[a2]], grids[a2][idx[a2] + 1]
            q = []
            for (u, w) in ((u0, w0), (u1, w0), (u1, w1), (u0, w1)):
                pt = [0, 0, 0]
                pt[ax], pt[a1], pt[a2] = p, u, w
                q.append(pt)
            q = np.array(q, float)
            n = np.cross(q[1] - q[0], q[2] - q[0])
            if n[ax] * outward < 0:
                q = q[::-1]
            tris += [[q[0], q[1], q[2]], [q[0], q[2], q[3]]]
    return np.array(tris)


def _area(poly):
    p = np.asarray(poly)
    return 0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])


def ear_clip(poly):
    pts = [tuple(p) for p in poly]
    if _area(pts) < 0:
        pts = pts[::-1]
    idx = list(range(len(pts)))
    out = []

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    guard = 0
    while len(idx) > 3 and guard < 10000:
        guard += 1
        for i in range(len(idx)):
            a, b, c = pts[idx[i - 1]], pts[idx[i]], pts[idx[(i + 1) % len(idx)]]
            if cross(a, b, c) <= 1e-9:
                continue
            ok = True
            for j in idx:
                p = pts[j]
                if p in (a, b, c):
                    continue
                if cross(a, b, p) >= -1e-9 and cross(b, c, p) >= -1e-9 and cross(c, a, p) >= -1e-9:
                    ok = False
                    break
            if ok:
                out.append((a, b, c))
                idx.pop(i)
                break
        else:
            break
    if len(idx) == 3:
        out.append(tuple(pts[i] for i in idx))
    return out


def prism(poly, e0, e1):
    """Extrude a simple 2D polygon (a, b) along e in local (a, b, e) coordinates."""
    pts = [tuple(p) for p in poly]
    if _area(pts) < 0:
        pts = pts[::-1]
    tris = []
    for a, b, c in ear_clip(pts):
        tris.append([[a[0], a[1], e1], [b[0], b[1], e1], [c[0], c[1], e1]])
        tris.append([[a[0], a[1], e0], [c[0], c[1], e0], [b[0], b[1], e0]])
    n = len(pts)
    for i in range(n):
        p, q = pts[i], pts[(i + 1) % n]
        A, B = [p[0], p[1], e0], [q[0], q[1], e0]
        Cc, D = [q[0], q[1], e1], [p[0], p[1], e1]
        tris += [[A, B, Cc], [A, Cc, D]]
    return np.array(tris, float)


def clip_halfplane(poly, a, b, c):
    """Keep the part of poly where a*x + b*y + c >= 0."""
    out = []
    n = len(poly)
    for i in range(n):
        p, q = poly[i], poly[(i + 1) % n]
        fp, fq = a * p[0] + b * p[1] + c, a * q[0] + b * q[1] + c
        if fp >= 0:
            out.append(p)
        if (fp >= 0) != (fq >= 0):
            t = fp / (fp - fq)
            out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    clean = []
    for p in out:
        if not clean or math.dist(p, clean[-1]) > 1e-6:
            clean.append(p)
    if len(clean) > 1 and math.dist(clean[0], clean[-1]) < 1e-6:
        clean.pop()
    return clean if len(clean) >= 3 and abs(_area(clean)) > 1e-3 else []


# ------------------------------------------------------------------ rack
def d_coeffs():
    # d(u, z) = (z - H) cos + u sin ; a card plane k is d = c_k
    return S, C, -H * C


def c_k(k):
    return (k - 1) * PITCH * S


def comb_polygon():
    return [(U_FRONT, 0), (U_FLOOR, 0), (U_FLOOR, FLOOR_T), (U_BACK, FLOOR_T), (U_BACK, H),
            (U_HEADER_END, H), (U_HEADER_END, H - HEADER_T), (U_FRONT, H - HEADER_T)]


def rib_polygons():
    a, b, c0 = d_coeffs()
    ribs = []
    for k in range(0, N_SLOTS + 1):
        poly = comb_polygon()
        if k >= 1:   # d >= c_k + GROOVE_HI
            poly = clip_halfplane(poly, a, b, c0 - (c_k(k) + GROOVE_HI))
        if k <= N_SLOTS - 1 and poly:   # d <= c_{k+1} + GROOVE_LO
            poly = clip_halfplane(poly, -a, -b, -c0 + (c_k(k + 1) + GROOVE_LO))
        if poly:
            ribs.append(poly)
    return ribs


# (u, z, e) -> rack-local (x, u, z) with e = x : columns e->x, u->y, z->z
M_WALL = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float)


def wall_local(side=+1):
    """Wall in rack-local coords (x lateral, y = u, z). side=+1 right wall, -1 left."""
    slab = prism([(U_FRONT, 0), (U_BACK, 0), (U_BACK, H), (U_FRONT, H)], X_SLAB_IN, RACK_W / 2)
    parts = [slab]
    for poly in rib_polygons():
        parts.append(prism(poly, X_COMB_IN, X_SLAB_IN + 0.05))
    m = Mesh(np.concatenate(parts)).transformed(M_WALL)
    if side < 0:
        m = m.transformed(np.diag([-1, 1, 1]))
    return m


def floor_local():
    return Mesh(box(-X_SLAB_IN, X_SLAB_IN, U_FLOOR, U_BACK, 0, FLOOR_T))


def header_local():
    return Mesh(box(-X_SLAB_IN, X_SLAB_IN, U_FRONT, U_HEADER_END, H - HEADER_T, H))


def card_frame(k, slide=0.0):
    """Matrix/offset mapping card-local (s, t, n) to rack-local (x, y=u, z)."""
    M = np.array([[0, 1, 0], [C, 0, S], [-S, 0, C]], float)
    off = np.array([0, (k - 1) * PITCH, H]) + slide * np.array([0, -C, S])
    return M, off


# ------------------------------------------------------------------ stand
def post_segment(i):
    z0, z1 = (0.0,) + JOINTS, JOINTS + (POST_TOP,)
    z0, z1 = z0[i - 1], z1[i - 1]
    hw = POST_W / 2
    adds = [(-hw, hw, -hw, hw, z0, z1)]
    subs = []
    if i < 3:
        pw = (PEG_W - PEG_CLEAR) / 2
        adds.append((-pw, pw, -pw, pw, z1, z1 + PEG_LEN))
    if i > 1:
        sw = (PEG_W + PEG_CLEAR) / 2
        subs.append((-sw, sw, -sw, sw, z0 - 1, z0 + PEG_LEN + 0.6))
    r = PIN_HOLE / 2
    for h in range(H_MIN, H_MAX + 1, H_STEP):
        z = pin_z(h)
        subs.append((-hw - 1, hw + 1, -r, r, z - r, z + r))          # pin channel along X
        if h % 50 == 0:                                               # height mark notches
            subs.append((-hw - 1, -hw + 1.0, -r - 6, -r - 2, z - 0.6, z + 0.6))
            subs.append((hw - 1.0, hw + 1, -r - 6, -r - 2, z - 0.6, z + 0.6))
    return rect_solid(adds, subs)


def foot():
    boss = POST_W + 2 * CLEAR + 14
    s = POST_W / 2 + CLEAR
    adds = [(-100, 100, FOOT_FRONT - 100, FOOT_FRONT, 0, 8), (-boss / 2, boss / 2, -boss / 2, boss / 2, 0, 30)]
    subs = [(-s, s, -s, s, -1, 40),                                    # post socket (through)
            (-1.5, 1.5, FOOT_FRONT - 3, FOOT_FRONT + 1, -1, 9)]       # centre notch on the front edge
    for x in (-85, 85):
        for y in (FOOT_FRONT - 85, FOOT_FRONT - 15):
            subs.append((x - 2.5, x + 2.5, y - 2.5, y + 2.5, -1, 9))  # screw holes
    return rect_solid(adds, subs)


def fence_slot_y():
    """Y range (lens-relative) of the fence bolt slots: fence centre = -(d + FENCE_W / 2)."""
    return -(FENCE_D_MAX + FENCE_W / 2), -(FENCE_D_MIN + FENCE_W / 2)


def carriage(h):
    """Carriage pinned at lens height h, in world coords."""
    zp = pin_z(h)
    zb, top = zp - PIN_TO_PLATE, zp + PIN_TO_PLATE
    inner = POST_W / 2 + CLEAR
    outer = inner + SLEEVE_WALL
    r = PIN_HOLE / 2
    w2 = WINDOW / 2
    ly = ARM_REACH
    y_plate0 = ly + PLATE_Y0
    adds = [(-outer, outer, -outer, outer, zb, top),
            (-ARM_W / 2, ARM_W / 2, outer - 0.01, y_plate0 + 0.01, top - ARM_T, top),
            (PLATE_X0, PLATE_X1, y_plate0, ly + PLATE_Y1, top - PLATE_T, top)]
    subs = [(-inner, inner, -inner, inner, zb - 1, top + 1),
            (-outer - 1, outer + 1, -r, r, zp - r, zp + r),
            (-w2, w2, ly - w2, ly + w2, top - PLATE_T - 1, top + 1)]
    for a in (-1, 1):   # crosshair notches pointing at the window centre
        x0 = w2 if a > 0 else -w2 - 10
        subs.append((x0, x0 + 10, ly - 0.6, ly + 0.6, top - 1, top + 1))
        y0 = ly + w2 if a > 0 else ly - w2 - 10
        subs.append((-0.6, 0.6, y0, y0 + 10, top - 1, top + 1))
    s0, s1 = fence_slot_y()
    for fx in FENCE_SLOT_X:   # M3 slots for the adjustable fence
        subs.append((fx - M3 / 2, fx + M3 / 2, ly + s0 - M3 / 2, ly + s1 + M3 / 2, top - PLATE_T - 1, top + 1))
    body = rect_solid(adds, subs)
    # gusset under the arm (overlaps the sleeve and arm; slicers merge it)
    g = prism([(outer - 1, zb), (outer - 1, top - ARM_T + 0.5), (outer + 70, top - ARM_T + 0.5)], -4, 4)
    g = Mesh(g).transformed(np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float)).tris
    return np.concatenate([body, g])


def fence(d=18.0, h=360):
    """Phone fence (separate part) in world coords, set for lens-to-phone-edge distance d."""
    top = pin_z(h) + PIN_TO_PLATE
    yc = ARM_REACH - (d + FENCE_W / 2)
    adds = [(FENCE_X0, FENCE_X1, yc - FENCE_W / 2, yc + FENCE_W / 2, top, top + FENCE_H)]
    subs = [(fx - M3 / 2, fx + M3 / 2, yc - M3 / 2, yc + M3 / 2, top - 1, top + FENCE_H + 1)
            for fx in FENCE_SLOT_X]
    return rect_solid(adds, subs)


# ------------------------------------------------------------------ STL
def write_stl(path, tris, name="part"):
    tris = np.asarray(tris, float)
    with open(path, "wb") as f:
        f.write(name.encode()[:80].ljust(80, b"\0"))
        f.write(struct.pack("<I", len(tris)))
        for t in tris:
            n = np.cross(t[1] - t[0], t[2] - t[0])
            L = np.linalg.norm(n)
            n = n / L if L > 0 else n
            f.write(struct.pack("<12fH", *n, *t[0], *t[1], *t[2], 0))
