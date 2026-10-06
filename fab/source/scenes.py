import sys
import time
import numpy as np
import shingle_cad as sc
import render as R
from render import Camera, render, draw_lines, label

import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "renders")
os.makedirs(OUT, exist_ok=True)
H_EX = 360   # C6 with a 26 mm-equivalent main lens
which = sys.argv[1:] or ["stand", "station", "rack", "phone"]
SIZE = (1600, 1200)


def frame_corners(W):
    Hf = W * 0.75
    c = np.array([0, sc.ARM_REACH, sc.H])
    return [c + [sx * W / 2, sy * Hf / 2, 0] for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]


if "stand" in which:
    t = time.time()
    cam = Camera(eye=(-640, 820, 760), target=(10, 70, 215), fov_deg=38, size=SIZE)
    im = render(R.stand_meshes(H_EX, with_phone=False), cam)
    label(im, (40, 40), "Camera stand  (shown at the 360 mm hole: C6, 26 mm lens)", 34)
    label(im, (40, 100), "orange = carriage + phone plate (lens window)   grey = foot + 3 post segments   pin = steel", 24, box=False)
    im.save(f"{OUT}/1_stand.png")
    print("stand", round(time.time() - t, 1))

if "station" in which:
    t = time.time()
    cam = Camera(eye=(780, 1180, 820), target=(0, 140, 175), fov_deg=39, size=SIZE)
    meshes = R.stand_meshes(H_EX) + R.station_racks(6)
    im = render(meshes, cam)
    lp = R.lens_point(H_EX)
    cs = frame_corners(480)
    draw_lines(im, cam, [(lp, c) for c in cs], (210, 50, 50), 2, dash=3)
    draw_lines(im, cam, [(cs[i], cs[(i + 1) % 4]) for i in range(4)], (210, 50, 50), 3)
    label(im, (40, 40), "Scanning station, C6: 6 racks x 20 cards = 120 per photo", 34)
    label(im, (40, 100), "red = what the phone sees (480 x 360 mm)", 24, box=False, fill=(180, 40, 40))
    im.save(f"{OUT}/2_station.png")
    print("station", round(time.time() - t, 1))

if "rack" in which:
    t = time.time()
    meshes = R.rack_meshes(1, 0, "U", n_cards=13, hide_left=True, slide_card=(14, 45))
    # the removed left wall, lying beside the rack as it is printed (grooves up)
    wl = sc.wall_local(-1)
    M = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], float)   # rotate so the grooved face points up
    lying = wl.transformed(M)
    mn = lying.tris.reshape(-1, 3).min(0)
    lying = lying.moved((-120 - mn[0], sc.RACK_Y0 - mn[1] + sc.U_FRONT + 20, -mn[2]))
    meshes.append(R.sc.Mesh(lying.tris, R.WALL))
    meshes.append(sc.Mesh(sc.box(-200, 120, -20, 420, -10, 0), R.BOARD))
    cam = Camera(eye=(-430, -10, 330), target=(-30, 175, 5), fov_deg=40, size=SIZE)
    im = render(meshes, cam)
    label(im, (40, 40), "One rack, near wall removed: cards slide down 20-degree grooves", 32)
    label(im, (40, 96), "card 14 is being slid in; all name bars end up at the same height", 24, box=False)
    label(im, (40, 1130), "left: the removed wall, grooves up, the way it is printed", 24, box=False)
    im.save(f"{OUT}/3_rack.png")
    print("rack", round(time.time() - t, 1))

if "phone" in which:
    t = time.time()
    lp = R.lens_point(H_EX)
    f_eq = 26.0
    fov = 2 * np.degrees(np.arctan(34.61 / 2 / f_eq))
    cam = Camera(eye=lp - [0, 0, 1], target=lp - [0, 0, 100], up=(0, -1, 0), fov_deg=fov,
                 size=(1600, 1200), ss=2)
    meshes = R.station_racks(6, partial=13) + [sc.Mesh(sc.box(-300, 300, -75, 375, -12, 0), R.BOARD)]
    meshes += [m for m in R.stand_meshes(H_EX, with_phone=False, with_board=False)]
    im = render(meshes, cam, outline=False)
    label(im, (30, 30), "What the phone sees (simulated, C6 at 360 mm): 120 name bars + 12 markers", 28)
    label(im, (30, 1140), "rack 6 holds 13 cards, so it needs a stop card in slot 14", 24)
    im.save(f"{OUT}/4_phone_view.png")
    print("phone", round(time.time() - t, 1))
