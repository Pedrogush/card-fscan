"""card-fscan synthetic photo generator (tools/synthgen v1).

Renders phone photos of C4 / C6 shingled columns on SPEC v1 strips, with real
Scryfall card images, plus the exact expected output (SPEC section 5 manifest).

  uv run tools/synthgen/gen.py --set smoke --n 8 --seed 1
  uv run tools/synthgen/gen.py --set dev --n 60 --seed 2
  uv run tools/synthgen/gen.py --set smoke --regen        # rebuild images from manifest
  uv run tools/synthgen/gen.py --set smoke --regen --debug c4_0001  # + slot crops in _debug/

Determinism: every image has a seed. Card choice uses default_rng([seed, 0])
and is stored in the manifest; everything rendered (placement jitter, camera,
lighting, noise) uses default_rng([seed, 1, attempt]). ``--regen`` only needs
the manifest and the (cached or re-downloaded) card images.

Each image is verified after encoding: cv2.aruco.ArucoDetector must find every
expected marker id (and none of the deliberately occluded ones) with corners
within 4 px of the true projection. If a non-negative image fails, the attempt
counter is bumped and the image re-rendered (recorded in the manifest).
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import render as R  # noqa: E402
from scryfall import Scryfall, build_pool  # noqa: E402

ROOT = HERE.parent.parent
TESTDATA = ROOT / "testdata"
GENERATOR = "tools/synthgen v1"

# Category mix per slot (sums to 1). ~70% EN, ~25% PT, ~5% other/tricky.
MIX = {
    "en_old": 0.14, "en_modern": 0.38, "en_special": 0.09, "en_multiface": 0.09,
    "pt": 0.25, "other": 0.04, "en_split": 0.01,
}

# Upper bounds / ranges per difficulty level. Values are sampled uniformly
# inside these ranges (signed quantities symmetric around 0).
LEVELS: dict[str, dict[str, Any]] = {
    "easy": dict(tilt=(0.0, 1.0), yaw=0.5, decentre=3.0, k1=0.0, defocus=(0.3, 0.6), motion=(0.0, 0.0),
                 read=(1.0, 2.0), shot=(0.02, 0.05), grad=0.08, vignette=(0.0, 0.10), glare=(0, 0, 0.0, 0.0),
                 wb=0.03, exposure=(0.97, 1.03), jpeg=(92, 95), card_dy=0.5, card_dx=0.3, card_rot=0.3,
                 tray_xy=1.0, tray_rot=0.3, tray_jit=(0.5, 0.1), shadow=(0.10, 0.25)),
    "normal": dict(tilt=(0.0, 2.5), yaw=1.0, decentre=6.0, k1=0.004, defocus=(0.5, 1.0), motion=(0.0, 1.5),
                   read=(2.0, 4.0), shot=(0.05, 0.12), grad=0.18, vignette=(0.08, 0.22), glare=(0, 2, 0.08, 0.22),
                   wb=0.06, exposure=(0.92, 1.06), jpeg=(88, 95), card_dy=1.5, card_dx=0.7, card_rot=1.0,
                   tray_xy=2.0, tray_rot=1.0, tray_jit=(1.0, 0.2), shadow=(0.15, 0.35)),
    "hard": dict(tilt=(2.0, 4.0), yaw=2.0, decentre=10.0, k1=0.010, defocus=(0.8, 1.5), motion=(0.5, 3.0),
                 read=(3.0, 6.0), shot=(0.10, 0.25), grad=0.30, vignette=(0.18, 0.32), glare=(1, 3, 0.18, 0.40),
                 wb=0.10, exposure=(0.85, 1.08), jpeg=(85, 90), card_dy=2.0, card_dx=1.0, card_rot=1.5,
                 tray_xy=3.0, tray_rot=1.5, tray_jit=(1.5, 0.3), shadow=(0.2, 0.45)),
}

EVAL_NOTES = {
    "slot_lang": "en | pt | other. 'other' = a printing in a language outside the index (ja, de, ...). "
                 "Readers may return review/empty for these; they are excluded from the auto-rate "
                 "denominator but an auto with the wrong oracle_id still counts as wrong.",
    "tags": "slot tags: split_sideways = split card whose names run along the long edge (expect review).",
    "expect_column_status": "present only on negative columns (e.g. 19 cards, no stop card): the reader "
                            "must report that column as 'error'. Its n_cards is not scored.",
    "retake": "expect_status 'retake' images have a hidden marker; only the photo status is scored.",
    "extra_fields": "scryfall_id, image_uri, category, placement [dx_mm, dy_mm, rot_deg] are for "
                    "regeneration/analysis and are not part of the scored truth.",
}


# --------------------------------------------------------------------------- #
# Planning (cards + structure), stored in the manifest


def _round(x: float, n: int = 4) -> float:
    return float(round(float(x), n))


def plan_recipes(set_name: str, n: int, rng: np.random.Generator) -> list[dict[str, Any]]:
    """Structure of each image: config, level, partial / negative kind."""
    if set_name == "smoke" and n == 8:
        return [
            dict(config="C4", level="easy", partial=None, negative=None),
            dict(config="C4", level="easy", partial=None, negative=None),
            dict(config="C4", level="normal", partial=None, negative=None),
            dict(config="C4", level="normal", partial=None, negative=None),
            dict(config="C4", level="normal", partial="one", negative=None),
            dict(config="C6", level="normal", partial=None, negative=None),
            dict(config="C4", level="easy", partial=None, negative="occluded_marker"),
            dict(config="C4", level="hard", partial=None, negative=None),
        ]
    n_c6 = round(n / 3)
    configs = ["C6"] * n_c6 + ["C4"] * (n - n_c6)
    n_easy, n_hard = round(n * 0.25), round(n * 0.25)
    levels = ["easy"] * n_easy + ["hard"] * n_hard + ["normal"] * (n - n_easy - n_hard)
    configs = [configs[i] for i in rng.permutation(n)]
    levels = [levels[i] for i in rng.permutation(n)]
    n_partial = max(1, round(n * 0.17))
    n_retake = max(1, round(n * 0.05))
    n_wrong = max(1, round(n * 0.05))
    kinds = (["partial"] * n_partial + ["occluded_marker"] * n_retake + ["wrong_count"] * n_wrong
             + [None] * (n - n_partial - n_retake - n_wrong))
    kinds = [kinds[i] for i in rng.permutation(n)]
    out = []
    for c, lv, k in zip(configs, levels, kinds):
        partial = None
        if k == "partial":
            partial = "two" if rng.random() < 0.3 else "one"
        out.append(dict(config=c, level=lv, partial=partial,
                        negative=k if k in ("occluded_marker", "wrong_count") else None))
    return out


def plan_image(recipe: dict[str, Any], seed: int, pool: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    rng = np.random.default_rng([seed, 0])
    ncols = R.CONFIGS[recipe["config"]]["cols"]
    counts = [20] * ncols
    stops = [False] * ncols
    if recipe["partial"] == "one":
        counts[-1] = int(rng.integers(1, 18))
        stops[-1] = True
    elif recipe["partial"] == "two":
        counts[-2] = int(rng.integers(1, 18))
        stops[-2] = True
        counts[-1] = 0
        stops[-1] = True
    negative = None
    expect_status = "ok"
    if recipe["negative"] == "wrong_count":
        j = int(rng.integers(1, ncols + 1))
        counts[j - 1] = 19
        stops[j - 1] = False
        negative = {"kind": "wrong_count", "column": j}
    elif recipe["negative"] == "occluded_marker":
        j = int(rng.integers(1, ncols + 1))
        ids = list(R.marker_ids(j))
        which = int(rng.integers(0, 3))  # left, right or both
        hidden = ids if which == 2 else [ids[which]]
        negative = {"kind": "occluded_marker", "column": j, "ids": hidden}
        expect_status = "retake"

    cats = list(MIX)
    probs = np.array([MIX[c] for c in cats])
    probs = probs / probs.sum()
    columns = []
    for j in range(1, ncols + 1):
        slots = []
        for i in range(1, counts[j - 1] + 1):
            cat = cats[int(rng.choice(len(cats), p=probs))]
            card = pool[cat][int(rng.integers(0, len(pool[cat])))]
            slot = {"slot": i, "oracle_id": card["oracle_id"], "name": card["name"], "lang": card["lang"],
                    "category": cat, "scryfall_id": card["scryfall_id"], "image_uri": card["image_uri"]}
            if cat == "en_split":
                slot["tags"] = ["split_sideways"]
            slots.append(slot)
        col: dict[str, Any] = {"column": j, "stop_card": stops[j - 1], "n_cards": counts[j - 1], "slots": slots}
        if negative and negative["kind"] == "wrong_count" and negative["column"] == j:
            col["expect_column_status"] = "error"
        columns.append(col)
    return {
        "seed": seed, "attempt": 0, "config": recipe["config"], "level": recipe["level"],
        "expect_status": expect_status, "negative": negative, "columns": columns,
    }


# --------------------------------------------------------------------------- #
# Rendering


def _sym(rng: np.random.Generator, a: float) -> float:
    return float(rng.uniform(-a, a)) if a > 0 else 0.0


def _tnorm(rng: np.random.Generator, a: float) -> float:
    """Truncated normal in [-a, a] with sd a/2 (most cards near the line)."""
    if a <= 0:
        return 0.0
    return float(np.clip(rng.normal(0.0, a / 2), -a, a))


def sample_camera(cfg: dict[str, Any], lv: dict[str, Any], ncols: int, rng: np.random.Generator,
                  need_pts: np.ndarray) -> tuple[R.Camera, dict[str, Any]]:
    W, H = cfg["size"]
    for _ in range(200):
        f35 = float(rng.uniform(23.5, 27.5))
        h_ideal = cfg["frame_w_mm"] * f35 / 34.6
        h = round(h_ideal / 10.0) * 10.0 + float(rng.uniform(-2.0, 2.0))
        f_px = W * f35 / 34.6
        tilt = float(rng.uniform(*lv["tilt"]))
        tilt_dir = float(rng.uniform(0, 360))
        yaw = _sym(rng, lv["yaw"])
        dcx, dcy = _sym(rng, lv["decentre"]), _sym(rng, lv["decentre"])
        k1 = _sym(rng, lv["k1"])
        Rm = R.rot_matrix(tilt, tilt_dir, yaw)
        # place the camera so the optical axis hits the frame centre (+ decentre)
        target = np.array([ncols * R.STRIP_W / 2 + dcx, 120.0 + dcy, 0.0])
        axis_world = Rm.T @ np.array([0.0, 0.0, 1.0])
        C = target - axis_world * (h / axis_world[2])
        cam = R.Camera(W, H, f_px, Rm, C, k1)
        px = cam.project(need_pts)
        ppm = f_px / h
        m = 5.0 * ppm
        if (px[:, 0].min() >= m and px[:, 1].min() >= m and px[:, 0].max() <= W - m and px[:, 1].max() <= H - m):
            info = dict(f35_mm=_round(f35, 2), height_mm=_round(h, 2), px_per_mm=_round(ppm, 3),
                        tilt_deg=_round(tilt, 3), tilt_dir_deg=_round(tilt_dir, 1), yaw_deg=_round(yaw, 3),
                        decentre_mm=[_round(dcx, 2), _round(dcy, 2)], k1=_round(k1, 5))
            return cam, info
    raise RuntimeError("could not place camera with all required points in frame")


def render_entry(entry: dict[str, Any], sf: Scryfall) -> tuple[bytes, dict[str, Any], dict[str, Any]]:
    """Render one manifest entry. Returns (jpeg, difficulty, aux) where aux has
    the true marker corners in image px and the per-column strip->image maps."""
    cfg = R.CONFIGS[entry["config"]]
    lv = LEVELS[entry["level"]]
    ncols = cfg["cols"]
    rng = np.random.default_rng([entry["seed"], 1, entry["attempt"]])

    # trays: block offset/rotation + small per-tray jitter, butted edge to edge
    block_rot = _sym(rng, lv["tray_rot"])
    block_dx, block_dy = _sym(rng, lv["tray_xy"]), _sym(rng, lv["tray_xy"])
    jit_xy, jit_rot = lv["tray_jit"]
    block = R.affine(block_rot, (ncols * R.STRIP_W / 2, 0.0), block_dx, block_dy)
    tray_m: list[np.ndarray] = []
    x_acc = 0.0
    tray_info = []
    for j in range(ncols):
        gap = float(rng.uniform(0.0, 0.6 * jit_xy))
        x_acc += gap if j > 0 else 0.0
        dy = _sym(rng, jit_xy)
        rot = _sym(rng, jit_rot)
        local = R.affine(rot, (R.STRIP_W / 2, 0.0), j * R.STRIP_W + x_acc, dy)
        tray_m.append(block @ local)
        tray_info.append([_round(x_acc, 3), _round(dy, 3), _round(rot, 3)])

    # card placement
    placements: list[list[list[float]]] = []
    for col in entry["columns"]:
        pl = []
        n = col["n_cards"] + (1 if col["stop_card"] else 0)
        for _ in range(n):
            pl.append([_tnorm(rng, lv["card_dx"]), _tnorm(rng, lv["card_dy"]), _tnorm(rng, lv["card_rot"])])
        placements.append(pl)

    # points that must be inside the frame (>= 5 mm margin): header markers,
    # stop-card markers, and the name zone corners
    need = []
    for j in range(ncols):
        for c in R.marker_corners_strip(j + 1).values():
            need.append(R.apply_h(tray_m[j], c))
        need.append(R.apply_h(tray_m[j], np.array([[0, R.HEADER], [R.STRIP_W, R.HEADER],
                                                   [0, R.HEADER + 200], [R.STRIP_W, R.HEADER + 200]])))
    stop_corner_local = np.array([[R.STOP_MARKER_XY[0], R.STOP_MARKER_XY[1]],
                                  [R.STOP_MARKER_XY[0] + R.STOP_MARKER_SIZE, R.STOP_MARKER_XY[1]],
                                  [R.STOP_MARKER_XY[0] + R.STOP_MARKER_SIZE, R.STOP_MARKER_XY[1] + R.STOP_MARKER_SIZE],
                                  [R.STOP_MARKER_XY[0], R.STOP_MARKER_XY[1] + R.STOP_MARKER_SIZE]])

    def card_m(j: int, i: int) -> np.ndarray:
        dx, dy, rot = placements[j][i - 1]
        top = R.HEADER + R.PITCH * (i - 1)
        return tray_m[j] @ R.affine(rot, (R.CARD_W / 2, R.CARD_H / 2), R.CARD_X + dx, top + dy)

    stop_mats: dict[int, np.ndarray] = {}
    for j, col in enumerate(entry["columns"]):
        if col["stop_card"]:
            stop_mats[j] = card_m(j, col["n_cards"] + 1)
            need.append(R.apply_h(stop_mats[j], stop_corner_local))
    need_pts = np.vstack(need)

    cam, cam_info = sample_camera(cfg, lv, ncols, rng, need_pts)

    # scene
    x0, y0, x1, y1 = R.scene_bounds(cam)
    scene = R.Scene(x0, y0, x1, y1)
    mat_kind = ["flat", "wood", "felt"][int(rng.integers(0, 3))]
    R.render_mat(scene, mat_kind, rng)
    # stand foot (front edge on the strips' top line, centred)
    foot_rgb = [(60, 60, 60), (40, 110, 200), (230, 230, 230), (30, 30, 30)][int(rng.integers(0, 4))]
    foot = np.full((int(100 * R.SCENE_PX_PER_MM), int(200 * R.SCENE_PX_PER_MM), 3), foot_rgb, np.uint8)
    scene.paste(R.Placed(foot, None, block @ R.affine(0, (0, 0), ncols * R.STRIP_W / 2 - 100, -100.0)))

    board_rgb = [(235, 235, 232), (190, 150, 110), (225, 225, 225)][int(rng.integers(0, 3))]
    paper = int(rng.integers(236, 250))
    paper_rgb = (paper, paper, paper - int(rng.integers(0, 6)))
    shadow = float(rng.uniform(*lv["shadow"]))
    stop_img, stop_alpha = R.render_stop_card()
    card_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for j, col in enumerate(entry["columns"]):
        scene.paste(R.Placed(R.render_tray(j + 1, board_rgb, paper_rgb), None, tray_m[j]))
        for slot in col["slots"]:
            url = slot["image_uri"]
            if url not in card_cache:
                card_cache[url] = R.prepare_card(sf.image(url))
            img, alpha = card_cache[url]
            scene.paste(R.Placed(img, alpha, card_m(j, slot["slot"]), shadow=shadow))
        if col["stop_card"]:
            scene.paste(R.Placed(stop_img, stop_alpha, stop_mats[j], shadow=shadow))

    # occluder for retake images: a hand-ish blob or a stray face-down card
    occl_info = None
    neg = entry.get("negative")
    if neg and neg["kind"] == "occluded_marker":
        j = neg["column"] - 1
        corners = R.marker_corners_strip(neg["column"])
        pts = np.vstack([R.apply_h(tray_m[j], corners[i]) for i in neg["ids"]])
        cx, cy = pts.mean(axis=0)
        # Cover exactly the hidden marker(s) (+ margin) and nothing of the
        # neighbouring marker, which must stay detectable.
        span = float(np.ptp(pts[:, 0]))
        kind = "hand" if rng.random() < 0.5 else "card_back"
        s = R.SCENE_PX_PER_MM
        if kind == "hand":
            rot = float(rng.uniform(-5, 5))
            w_mm, h_mm = span * 1.25 + 4.0, 34.0  # ellipse covering the square(s)
            img = np.zeros((int(h_mm * s), int(w_mm * s), 3), np.uint8)
            img[:] = (120, 150, 205)
            alpha = np.zeros(img.shape[:2], np.uint8)
            cv2.ellipse(alpha, (img.shape[1] // 2, img.shape[0] // 2), (img.shape[1] // 2 - 2, img.shape[0] // 2 - 2),
                        0, 0, 360, 255, -1)
            shade = cv2.GaussianBlur(alpha, (0, 0), 3 * s).astype(np.float32) / 255.0
            img = (img.astype(np.float32) * (0.75 + 0.25 * shade[:, :, None])).astype(np.uint8)
            if len(neg["ids"]) == 1:
                w_mm = h_mm = 30.0
                img = cv2.resize(img, (int(w_mm * s), int(h_mm * s)))
                alpha = cv2.resize(alpha, (int(w_mm * s), int(h_mm * s)))
        else:
            rot = float(rng.uniform(-8, 8))
            w_mm, h_mm = span + 5.0, 26.0
            img = np.zeros((int(h_mm * s), int(w_mm * s), 3), np.uint8)
            img[:] = (40, 70, 110)
            cv2.ellipse(img, (img.shape[1] // 2, img.shape[0] // 2), (img.shape[1] // 3, img.shape[0] // 3),
                        0, 0, 360, (60, 110, 170), -1)
            alpha = R._rounded_mask(img.shape[1], img.shape[0], 2 * s)
        m = R.affine(rot, (w_mm / 2, h_mm / 2), cx - w_mm / 2, cy - h_mm / 2)
        scene.paste(R.Placed(img, alpha, m, shadow=0.3))
        occl_info = kind

    # photometric look
    W, H = cfg["size"]
    n_glare = int(rng.integers(lv["glare"][0], lv["glare"][1] + 1))
    glare = []
    for _ in range(n_glare):
        glare.append((float(rng.uniform(0, W)), float(rng.uniform(0, H)),
                      float(rng.uniform(8, 30) * cam_info["px_per_mm"]), float(rng.uniform(lv["glare"][2], lv["glare"][3]))))
    look = {
        "light_grad": (_sym(rng, lv["grad"]), _sym(rng, lv["grad"])),
        "vignette": float(rng.uniform(*lv["vignette"])),
        "glare": glare,
        "exposure": float(rng.uniform(*lv["exposure"])),
        "defocus_sigma_px": float(rng.uniform(*lv["defocus"])),
        "motion_px": float(rng.uniform(*lv["motion"])),
        "motion_angle_deg": float(rng.uniform(0, 180)),
        "read_noise": float(rng.uniform(*lv["read"])),
        "shot_noise": float(rng.uniform(*lv["shot"])),
        "wb_gains_bgr": [1.0 + _sym(rng, lv["wb"]) for _ in range(3)],
    }
    quality = int(rng.integers(lv["jpeg"][0], lv["jpeg"][1] + 1))
    img = R.photograph(scene, cam, look, rng)
    del scene
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    jpeg = buf.tobytes()

    all_dy = [abs(p[1]) for pl in placements for p in pl]
    all_dx = [abs(p[0]) for pl in placements for p in pl]
    all_rot = [abs(p[2]) for pl in placements for p in pl]
    difficulty = {
        "level": entry["level"], **cam_info,
        "blur_px": _round(look["defocus_sigma_px"], 3), "motion_px": _round(look["motion_px"], 3),
        "motion_angle_deg": _round(look["motion_angle_deg"], 1),
        "jitter_mm": _round(max(all_dy, default=0.0), 3), "jitter_x_mm": _round(max(all_dx, default=0.0), 3),
        "card_rot_max_deg": _round(max(all_rot, default=0.0), 3),
        "tray_block": [_round(block_dx, 3), _round(block_dy, 3), _round(block_rot, 3)],
        "trays": tray_info,
        "light_grad": [_round(v, 4) for v in look["light_grad"]], "vignette": _round(look["vignette"], 4),
        "glare": [[_round(g[0], 1), _round(g[1], 1), _round(g[2], 1), _round(g[3], 3)] for g in glare],
        "exposure": _round(look["exposure"], 4),
        "read_noise": _round(look["read_noise"], 3), "shot_noise": _round(look["shot_noise"], 4),
        "wb_gains_bgr": [_round(v, 4) for v in look["wb_gains_bgr"]], "jpeg_quality": quality,
        "mat": mat_kind, "shadow": _round(shadow, 3),
    }
    if occl_info:
        difficulty["occluder"] = occl_info

    # truth for verification / debug
    markers: dict[int, np.ndarray] = {}
    for j in range(ncols):
        for mid, c in R.marker_corners_strip(j + 1).items():
            markers[mid] = cam.project(R.apply_h(tray_m[j], c))
    stop_px = [cam.project(R.apply_h(m, stop_corner_local)) for m in stop_mats.values()]
    aux = {"markers": markers, "stops": stop_px, "placements": placements,
           "strip_to_img": [(cam, tray_m[j]) for j in range(ncols)]}
    return jpeg, difficulty, aux


def verify(entry: dict[str, Any], jpeg: bytes, aux: dict[str, Any]) -> tuple[bool, str, dict[int, np.ndarray]]:
    gray = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)
    det = cv2.aruco.ArucoDetector(R.ARUCO_DICT, cv2.aruco.DetectorParameters())
    corners, ids, _ = det.detectMarkers(gray)
    found: dict[int, list[np.ndarray]] = {}
    if ids is not None:
        for c, i in zip(corners, ids.flatten()):
            found.setdefault(int(i), []).append(c.reshape(4, 2))
    ncols = R.CONFIGS[entry["config"]]["cols"]
    hidden = set(entry["negative"]["ids"]) if entry.get("negative", {}) and entry["negative"]["kind"] == "occluded_marker" else set()
    expected = {i for i in range(2 * ncols)} - hidden
    n_stop = sum(1 for c in entry["columns"] if c["stop_card"])
    problems = []
    missing = expected - set(found)
    if missing:
        problems.append(f"missing ids {sorted(missing)}")
    leaked = hidden & set(found)
    if leaked:
        problems.append(f"occluded ids detected {sorted(leaked)}")
    extra = set(found) - expected - {R.STOP_ID} - hidden
    if extra:
        problems.append(f"unexpected ids {sorted(extra)}")
    if len(found.get(R.STOP_ID, [])) != n_stop:
        problems.append(f"stop markers found {len(found.get(R.STOP_ID, []))} != {n_stop}")
    worst = 0.0
    for mid in expected & set(found):
        err = min(float(np.abs(c - aux["markers"][mid]).max()) for c in found[mid])
        worst = max(worst, err)
    for sp in aux["stops"]:
        if found.get(R.STOP_ID):
            err = min(float(np.abs(c - sp).max()) for c in found[R.STOP_ID])
            worst = max(worst, err)
    if worst > 4.0:
        problems.append(f"corner error {worst:.2f}px")
    first = {k: v[0] for k, v in found.items()}
    return (not problems), ("; ".join(problems) or f"ok (max corner err {worst:.2f}px)"), first


def debug_crops(entry: dict[str, Any], jpeg: bytes, found: dict[int, np.ndarray], out_dir: Path, stem: str) -> None:
    """Warp each column with the homography from the detected header corners
    (SPEC 2 step 3) and save the canonical column plus a few slot crops."""
    out_dir.mkdir(parents=True, exist_ok=True)
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    ppm = 12.0
    for col in entry["columns"]:
        j = col["column"]
        a, b = R.marker_ids(j)
        if a not in found or b not in found:
            continue
        src = np.vstack([found[a], found[b]]).astype(np.float32)
        cs = R.marker_corners_strip(j)
        dst = (np.vstack([cs[a], cs[b]]) * ppm).astype(np.float32)
        Hm, _ = cv2.findHomography(src, dst, 0)
        canvas = cv2.warpPerspective(img, Hm, (int(70 * ppm), int(340 * ppm)), flags=cv2.INTER_LINEAR)
        vis = canvas.copy()
        for i in range(1, 21):
            y0 = (25 + 10 * (i - 1) + 0.5) * ppm
            y1 = (25 + 10 * i - 0.5) * ppm
            cv2.rectangle(vis, (int(4.5 * ppm), int(y0)), (int(65.5 * ppm), int(y1)), (0, 0, 255), 2)
        cv2.imwrite(str(out_dir / f"{stem}_col{j}.jpg"), cv2.resize(vis, None, fx=0.5, fy=0.5,
                                                                      interpolation=cv2.INTER_AREA))
        for slot in col["slots"]:
            i = slot["slot"]
            if i not in (1, 2, 10, 19, 20) and i != col["n_cards"]:
                continue
            y0 = int((25 + 10 * (i - 1) + 0.5) * ppm)
            y1 = int((25 + 10 * i - 0.5) * ppm)
            crop = canvas[y0:y1, int(4.5 * ppm):int(65.5 * ppm)]
            cv2.imwrite(str(out_dir / f"{stem}_col{j}_s{i:02d}.png"), crop)


# --------------------------------------------------------------------------- #
# Driver


def _work(args: tuple[str, dict[str, Any], str, bool, bool]) -> tuple[str, dict[str, Any], str, int]:
    key, entry, set_dir, regen, debug = args
    cv2.setNumThreads(1)
    sf = Scryfall()
    stem = Path(key).stem
    attempts = 0
    while True:
        jpeg, diff, aux = render_entry(entry, sf)
        ok, msg, found = verify(entry, jpeg, aux)
        if ok:
            break
        if regen:
            raise RuntimeError(f"{key}: verification failed on regen: {msg}")
        attempts += 1
        if attempts > 10:
            raise RuntimeError(f"{key}: verification keeps failing: {msg}")
        print(f"  {key}: attempt {entry['attempt']} rejected ({msg}); retrying", flush=True)
        entry = {**entry, "attempt": entry["attempt"] + 1}
    if regen and entry.get("difficulty") and entry["difficulty"] != diff:
        raise RuntimeError(f"{key}: regenerated parameters differ from manifest (non-deterministic?)")
    # placement detail on each slot
    for j, col in enumerate(entry["columns"]):
        for slot in col["slots"]:
            slot["placement"] = [_round(v, 3) for v in aux["placements"][j][slot["slot"] - 1]]
    entry = {**entry, "difficulty": diff}
    (Path(set_dir) / Path(key).name).write_bytes(jpeg)
    if debug:
        debug_crops(entry, jpeg, found, Path(set_dir) / "_debug", stem)
    return key, entry, msg, len(jpeg)


def write_readme(set_dir: Path, set_name: str, manifest: dict[str, Any]) -> None:
    imgs = manifest["images"]
    n = len(imgs)
    by_cfg: dict[str, int] = {}
    by_lv: dict[str, int] = {}
    langs: dict[str, int] = {}
    partial = retake = wrong = 0
    slots = 0
    for e in imgs.values():
        by_cfg[e["config"]] = by_cfg.get(e["config"], 0) + 1
        by_lv[e["level"]] = by_lv.get(e["level"], 0) + 1
        retake += e["expect_status"] == "retake"
        wrong += bool(e["negative"] and e["negative"]["kind"] == "wrong_count")
        partial += any(c["stop_card"] for c in e["columns"])
        for c in e["columns"]:
            for s in c["slots"]:
                slots += 1
                langs[s["lang"]] = langs.get(s["lang"], 0) + 1
    seed = manifest.get("seed")
    text = f"""# testdata/{set_name}

Synthetic phone photos of shingled columns, generated by `tools/synthgen` ({GENERATOR}).
Only `manifest.json` is committed; the JPEGs contain Wizards of the Coast card art
(images from Scryfall) and are regenerated locally:

```
uv run --project tools tools/synthgen/gen.py --set {set_name} --regen
```

That downloads the card images once into `testdata/cache/` (gitignored) and
rebuilds every image bit-for-bit from the per-image seeds in the manifest (same
machine and pinned library versions). The set was created with
`gen.py --set {set_name} --n {n} --seed {seed}`.

Contents: {n} images ({', '.join(f'{v} {k}' for k, v in sorted(by_cfg.items()))};
{', '.join(f'{v} {k}' for k, v in sorted(by_lv.items()))}), {slots} true card slots
({', '.join(f'{k} {v} ({100 * v / max(1, slots):.0f}%)' for k, v in sorted(langs.items()))}).
{partial} images with stop cards (partial columns), {retake} retake images (a marker
hidden), {wrong} images with a 19-card column without stop card (column must be `error`).

Score a reader with:

```
uv run --project tools tools/eval.py --manifest testdata/{set_name}/manifest.json --results <dir>
```

See `manifest.json` -> `eval_notes` for how `lang: other`, tags and negative cases are scored.
"""
    (set_dir / "README.md").write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True, dest="set_name")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--regen", action="store_true", help="rebuild images from an existing manifest")
    ap.add_argument("--only", nargs="*", default=None, help="image stems to (re)render, e.g. c4_0001")
    ap.add_argument("--debug", nargs="*", default=None,
                    help="write slot crops for these stems (no args: all) to testdata/<set>/_debug")
    ap.add_argument("--workers", type=int, default=1, help="render processes (<= 2 recommended)")
    args = ap.parse_args()

    set_dir = TESTDATA / args.set_name
    set_dir.mkdir(parents=True, exist_ok=True)
    man_path = set_dir / "manifest.json"
    sf = Scryfall()
    if args.regen:
        manifest = json.loads(man_path.read_text(encoding="utf-8"))
    else:
        pool = build_pool(sf)
        rng = np.random.default_rng(args.seed)
        recipes = plan_recipes(args.set_name, args.n, rng)
        images: dict[str, Any] = {}
        for k, rec in enumerate(recipes, start=1):
            seed = args.seed * 100_000 + k
            key = f"{args.set_name}/{rec['config'].lower()}_{k:04d}.jpg"
            images[key] = plan_image(rec, seed, pool)
        manifest = {
            "spec_version": 1, "set": args.set_name, "generator": GENERATOR, "seed": args.seed,
            "card_images": "Scryfall 'large' JPEG (672x936, ~10.7 px/mm); card art (c) Wizards of the Coast",
            "eval_notes": EVAL_NOTES, "images": images,
        }

    # prefetch card images sequentially (polite to the CDN), then render
    urls = sorted({s["image_uri"] for e in manifest["images"].values() for c in e["columns"] for s in c["slots"]})
    t0 = time.time()
    for i, u in enumerate(urls):
        sf.image(u)
        if (i + 1) % 200 == 0:
            print(f"  images {i + 1}/{len(urls)}", flush=True)
    print(f"{len(urls)} card images ready ({time.time() - t0:.0f}s)", flush=True)

    keys = list(manifest["images"])
    if args.only:
        keys = [k for k in keys if Path(k).stem in args.only]
    debug_all = args.debug is not None and len(args.debug) == 0
    jobs = [(k, manifest["images"][k], str(set_dir), args.regen,
             debug_all or (args.debug is not None and Path(k).stem in args.debug)) for k in keys]
    t0 = time.time()
    if args.workers > 1:
        with mp.get_context("spawn").Pool(args.workers) as p:
            results = p.imap(_work, jobs)
            done = [_report(r, t0) for r in results]
    else:
        done = [_report(_work(j), t0) for j in jobs]
    if not args.regen:
        for key, entry in done:
            manifest["images"][key] = entry
        man_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8",
                            newline="\n")
        write_readme(set_dir, args.set_name, manifest)
        print(f"wrote {man_path}")


def _report(r: tuple[str, dict[str, Any], str, int], t0: float) -> tuple[str, dict[str, Any]]:
    key, entry, msg, size = r
    d = entry["difficulty"]
    print(f"{key}: {entry['level']:6} {entry['expect_status']:6} {d['px_per_mm']:.2f}px/mm tilt {d['tilt_deg']:.1f} "
          f"blur {d['blur_px']:.2f} q{d['jpeg_quality']} {size / 1e6:.1f}MB attempt {entry['attempt']} - {msg} "
          f"[{time.time() - t0:.0f}s]", flush=True)
    return key, entry


if __name__ == "__main__":
    main()
