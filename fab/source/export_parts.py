"""Export printable STLs in print orientation (flat side on the bed, Z up)."""
import os
import numpy as np
import shingle_cad as sc

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")   # STLs live in fab/
os.makedirs(OUT, exist_ok=True)


def to_bed(tris):
    t = np.asarray(tris, float).copy()
    mn = t.reshape(-1, 3).min(0)
    mx = t.reshape(-1, 3).max(0)
    t -= [(mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2, mn[2]]
    return t


def wall_print(side):
    m = sc.wall_local(side)              # rack-local (x, u, z)
    # bed X = u, bed Y = z (height of the wall), bed Z = distance from the outer slab face
    if side > 0:   # proper rotations only (det +1), so nothing gets mirrored
        M = np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]], float)
    else:
        M = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], float)
    assert abs(np.linalg.det(M) - 1) < 1e-9
    return to_bed(m.transformed(M).tris)


def coupon(side):
    """Short wall section with 5 grooves, widths 0.8 / 0.9 / 1.0 / 1.1 / 1.2 mm (left to right)."""
    s, c = sc.S, sc.C
    Hc = 26.0
    rect = [(0, 0), (120, 0), (120, Hc), (6, Hc), (0, Hc - 6)]      # cut corner marks the 0.8 end
    widths = [0.8, 0.9, 1.0, 1.1, 1.2]
    # groove k: band around d = (z - Hc) c + u s = 10 (k+1) s
    bands = [(10 * (k + 1) * s - w / 2, 10 * (k + 1) * s + w / 2) for k, w in enumerate(widths)]
    ribs = []
    for k in range(len(bands) + 1):
        poly = rect
        if k > 0:
            poly = sc.clip_halfplane(poly, s, c, -Hc * c - bands[k - 1][1])
        if k < len(bands) and poly:
            poly = sc.clip_halfplane(poly, -s, -c, Hc * c + bands[k][0])
        if poly:
            ribs.append(sc.prism(poly, 3.0 - 0.05, 4.5))
    slab = sc.prism(rect, 0, 3.0)
    t = np.concatenate([slab] + ribs)
    if side < 0:
        t = sc.Mesh(t).transformed(np.diag([-1, 1, 1])).tris
    return to_bed(t)


def post_print(i):
    return to_bed(sc.post_segment(i))


def carriage_print():
    t = sc.carriage(360)
    # plate top face down on the bed
    return to_bed(sc.Mesh(t).transformed(np.diag([1, -1, -1])).tris)


PARTS = {
    "rack_wall_left": (lambda: wall_print(-1), "1 per rack"),
    "rack_wall_right": (lambda: wall_print(+1), ""),
    "rack_floor": (lambda: to_bed(sc.floor_local().tris), "or cut from 3 mm MDF/hardboard"),
    "rack_header": (lambda: to_bed(sc.header_local().tris), "or cut from 3 mm MDF/hardboard"),
    "groove_test_left": (lambda: coupon(-1), "print FIRST"),
    "groove_test_right": (lambda: coupon(+1), "print FIRST"),
    "stand_foot": (lambda: to_bed(sc.foot()), ""),
    "stand_post_1": (lambda: post_print(1), ""),
    "stand_post_2": (lambda: post_print(2), ""),
    "stand_post_3": (lambda: post_print(3), ""),
    "stand_carriage": (carriage_print, ""),
    "stand_phone_fence": (lambda: to_bed(sc.fence()), "2 x M3x16 + nuts"),
}


def edge_report(t):
    from collections import Counter
    t = np.round(np.asarray(t), 4)
    cnt = Counter()
    for tri in t:
        for a, b in ((0, 1), (1, 2), (2, 0)):
            e = tuple(sorted((tuple(tri[a]), tuple(tri[b]))))
            cnt[e] += 1
    bad = sum(1 for v in cnt.values() if v != 2)
    return bad


if __name__ == "__main__":
    for name, (fn, _) in PARTS.items():
        t = fn()
        sc.write_stl(os.path.join(OUT, f"{name}.stl"), t, name)
        ext = t.reshape(-1, 3).max(0) - t.reshape(-1, 3).min(0)
        # signed volume (positive = outward normals)
        vol = np.sum(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2]))) / 6
        print(f"{name:20s} {len(t):6d} tris  {ext[0]:6.1f} x {ext[1]:6.1f} x {ext[2]:6.1f} mm"
              f"  vol {vol / 1000:7.1f} cm3  odd-edges {edge_report(t)}")
    print("H (name-bar plane above board) =", round(sc.H, 2), " MAT =", sc.MAT)
