"""The pictures of the BEV system for the evaluation suite (Evaluation): every view returns a `Page` of panels (pure data).

Views (names as in `BEVTask.views()`):
    overview    camera with the projected ground truth | GT bird's-eye view | prediction (+ boxes) | three error maps
                (drivable area, vehicle footprint, lane dividers): class colour right, tinted green false, tinted blue missed, black background, dark grey not evaluated
    outputs     the detailed result page: GT / prediction / errors, dt = +-0.5 s with the unknown cells, splat mass, scores, speed
    stages      the stages of the chain: camera heads, lift, camera BEV, temporal fusion, radar, trunk (needs the taps); a model with the 1 m centre-line head
                gets two panels more (25 presence against the GT, 26 direction arrows: `centreline_panels`)
    dashboard   one compact picture of every stage of one frame (the video frame)
Bird's-eye maps are drawn with x forward (up) and +y (left of the car) on the left; arrays are (x, y) = (rows, columns).
Colour scheme (Mirko, 2026-11-03): black plot background, dark-grey road with a black boundary, white lane lines, light-grey crossings, a calmer green for walkways, dark-red
objects (vehicles, pedestrians, two-wheelers) and the same red translucent where they lie outside the observed area (camera x radar x not hidden); boxes: GT outline red
(translucent outside), the dashed GT outline on a prediction white, predicted boxes amber. The error maps (Mirko, 2026-11-04) use the same class colours: right = the class colour, false = mixed with green, missed = mixed with blue, black
background, no red (`ERR`, `class_colors`).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from Evaluation import Page, Panel, Prediction, Sample, Shape, View, ViewContext
from Evaluation.render.figure import error_counts
from Evaluation.render.taps import energy, pca_rgb
from evaluation.metrics import FRONT_HALF_FOV, MOVING_SPEED, angle_error, dilate, gt_boxes_from_meta, in_front, line_cells

# The colour scheme of the bird's-eye pictures (Mirko's proposal of 2026-11-03): black plot background, dark-grey road with a black boundary, white lane markings, light-grey
# crossings, dark-red objects (vehicles, pedestrians, two-wheelers) and the same red, translucent, where they lie outside the observed area, a calmer green for walkways.
BG = (0.0, 0.0, 0.0)
ROAD_EDGE = (0.0, 0.0, 0.0)
DARK_RED, OUT_RED, OUT_ALPHA = (0.62, 0.04, 0.07), (0.95, 0.16, 0.16), 0.45
OBJECT_LAYERS = ("vehicle", "pedestrian", "two_wheeler")
GT_BOX_COLOR, GT_BOX_OUT = (0.80, 0.12, 0.14), (0.95, 0.30, 0.30, 0.55)     # outline of a ground-truth box inside / outside the observed area (RGBA: translucent)
GT_OVERLAY, DET_COLOR = (0.93, 0.93, 0.93), (1.0, 0.72, 0.15)                 # dashed GT outline on a prediction panel; predicted boxes
COLORS = {"drivable_area": (0.23, 0.23, 0.25), "carpark_area": (0.31, 0.33, 0.40), "walkway": (0.31, 0.60, 0.43),
          "ped_crossing": (0.74, 0.74, 0.74), "stop_line": (0.60, 0.60, 0.64),
          "vehicle": DARK_RED, "pedestrian": DARK_RED, "two_wheeler": DARK_RED}
LINE_COLORS = {"lane_divider": (0.97, 0.97, 0.97), "road_divider": (0.98, 0.82, 0.30)}
GROUP_COLOR = {"vehicle": (0.95, 0.15, 0.15), "pedestrian": (0.9, 0.2, 0.9), "two_wheeler": (1.0, 0.6, 0.1)}
LINE_RGB = {"lane_divider": (1.0, 0.9, 0.1), "road_divider": (1.0, 0.5, 0.0), "centreline": (0.1, 0.9, 0.9)}
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
MAP_LEGEND = [(n, c) for n, c in list(COLORS.items()) + list(LINE_COLORS.items())] + [("object outside the observed area", OUT_RED)]

# The error maps in the convention of the ground-truth picture (Mirko, 2026-11-04): right = the class colour, a false cell (predicted, not in the ground truth) = the class colour
# mixed with green, a missed cell = mixed with blue, nothing = the black background; never red (red is for moving targets). Not evaluated = dark grey.
GREEN, BLUE, TINT, SKIP_DARK = (0.20, 0.85, 0.35), (0.25, 0.45, 1.0), 0.5, (0.17, 0.17, 0.15)


def tinted(c, toward, amount: float = TINT) -> tuple:
    return tuple(float(x) * (1 - amount) + float(y) * amount for x, y in zip(c, toward))


def class_colors(c) -> Dict[str, tuple]:
    return dict(tp=tuple(c), fp=tinted(c, GREEN), fn=tinted(c, BLUE), tn=BG, skip=SKIP_DARK)


ERR = {"drivable_area": class_colors(COLORS["drivable_area"]), "vehicle": class_colors(COLORS["vehicle"]),
       "lane_divider": class_colors(LINE_COLORS["lane_divider"]), "centreline": class_colors(LINE_RGB["centreline"])}
ERR_LEGEND = [("right (the class colour, as in the ground truth)", ERR["drivable_area"]["tp"]), ("false: predicted, not in the ground truth (class colour + green)", ERR["drivable_area"]["fp"]),
              ("missed: ground truth, not predicted (class colour + blue)", ERR["drivable_area"]["fn"]), ("not evaluated", SKIP_DARK)]


# ------------------------------------------------------------------------------------------------ geometry helpers
def layer_index(spec) -> Dict[str, tuple]:
    return {name: (i, kind, key) for i, (name, kind, key) in enumerate(spec)}


def masks(g) -> Dict[str, np.ndarray]:
    cam = np.asarray(g["camera"], np.float32) > 0
    seen = np.asarray(g["undefined"], np.float32) == 0
    keep = (np.asarray(g["ignore"], np.float32) == 0) if "ignore" in g else np.ones_like(cam)
    return {"fov": cam, "seen": seen, "obj_fov": cam & keep, "out": ~cam}


def gt_boxes(g, groups) -> List[dict]:
    return [b for b in gt_boxes_from_meta(g["meta"]["boxes"]) if b["group"] in groups]


def extent(grid) -> Tuple[float, float, float, float]:
    return (grid.y_min, grid.y_max, grid.x_min, grid.x_max)


def observed_area(g) -> np.ndarray:
    """cells the rig observes in this GT frame (the fusion area of D37: camera cells not hidden behind a box, inside the radar fan); all True for a GT without these layers"""
    cam = np.asarray(g["camera"], np.float32) > 0 if "camera" in g else None
    if cam is None:
        return np.ones_like(np.asarray(g["undefined"]), bool) if "undefined" in g else np.ones((120, 104), bool)
    area = cam
    if "occluded" in g:
        area = area & ~(np.asarray(g["occluded"], np.float32) > 0)
    if "radar" in g:
        area = area & (np.asarray(g["radar"], np.float32) > 0)
    return area


def box_shapes(boxes, color, ls="-", lw=1.2, score_min=None, observed=None) -> List[Shape]:
    """box dicts (cx, cy, length, width, yaw | radius) -> overlay shapes in bird's-eye plot coordinates (horizontal = y). observed: a bool grid (observed_area) -> `color` for a
    box with at least half of its footprint inside, translucent GT_BOX_OUT for the others (the colour argument is then the one of the observed boxes)"""
    out = []
    if observed is not None:
        from evaluation.bev_fusion_area import footprint_share
    for b0 in boxes:
        b = b0
        if score_min is not None and b.get("score", 1.0) < score_min:
            continue
        color_b = color if observed is None or footprint_share(b, observed) >= 0.5 else GT_BOX_OUT
        if "radius" in b:
            out.append(Shape("circle", (b["cy"], b["cx"], max(b["radius"], 0.3)), color_b, ls, lw))
        else:
            c, s = math.cos(b["yaw"]), math.sin(b["yaw"])
            hl, hw = b["length"] / 2, b["width"] / 2
            pts = [(b["cx"] + c * a - s * w, b["cy"] + s * a + c * w) for a, w in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))]
            out.append(Shape("polygon", [(y, x) for x, y in pts], color_b, ls, lw))
            out.append(Shape("line", [(b["cy"], b["cx"]), (b["cy"] + s * hl, b["cx"] + c * hl)], color_b, ls, lw))
    return out


NOCARE_COLOR, MOVING_RED = (0.66, 0.58, 0.95), (0.95, 0.20, 0.20)
OUTSIDE_COLOR = (0.66, 0.58, 0.95, 0.35)                                # boxes outside the scored window: faint violet, not counted


def error_box_shapes(dets, gts_all, shares, area, thr: float = 2.0, min_score: float = 0.3):
    """Boxes of a prediction panel with the errors told apart by colour (the convention of the maps: right = the class colour, false = tinted green, missed = tinted blue):
      detection that matches a ground-truth box (within `thr` m)     amber, solid      its ground-truth box: white, dashed, thin
      false detection                                                 amber tinted green, solid, thick
      missed ground-truth box                                         white tinted blue, dashed, thick
      'no care' (D38: less than half of the footprint in the area the rig observes: such a box is neither a hit nor a miss nor a false alarm): violet, thin
      outside the scored window of the numbers (x < 60 m, +-32.3 deg): faint violet, not counted
    -> (shapes, counts {"tp","fp","fn","nocare"}): the counts are those of the window, as the numbers count. gts_all: EVERY annotated box of the scored classes with its footprint
    share (bev_fusion_area.gt_with_shares); the detections are drawn from score `min_score` on (what the pictures always did)."""
    from evaluation.bev_errors import classify_frame
    from evaluation.bev_fusion_area import prepare
    shown = [d for d in dets if d.get("score", 1.0) >= min_score]
    inside_d = [d for d in shown if in_front(d)]
    kept, scored = prepare(inside_d, list(gts_all), list(shares), area, include_partial=False)         # as the metric: ALL annotated boxes, the ones beyond the window only neutralise
    neutral_d = [d for d in inside_d if not any(d is k for k in kept)]
    neutral_g = [g for g in gts_all if in_front(g) and not any(g is k for k in scored)]               # counted: the window's no-care boxes
    r = classify_frame(kept, scored, sorted({b["group"] for b in list(kept) + list(scored)}), thr, 0.0)
    out: List[Shape] = box_shapes([b for b in list(shown) + list(gts_all) if not in_front(b)], OUTSIDE_COLOR, "-", 0.6)
    out += box_shapes(neutral_g + neutral_d, NOCARE_COLOR, "-", 0.8)
    for d, g in r["tp"]:
        out += box_shapes([g], GT_OVERLAY, "--", 0.7) + box_shapes([d], DET_COLOR, "-", 1.4)
    for d, _kind, _other in r["fp"]:
        out += box_shapes([d], tinted(DET_COLOR, GREEN), "-", 2.0)
    for g, _kind, _other in r["fn"]:
        out += box_shapes([g], tinted(GT_OVERLAY, BLUE, 0.6), "--", 2.0)
    return out, {"tp": len(r["tp"]), "fp": len(r["fp"]), "fn": len(r["fn"]), "nocare": len(neutral_g) + len(neutral_d)}


def box_caption(c) -> str:
    """the legend sentence of a prediction panel's boxes from the counts of error_box_shapes (None: the plain boxes of the fallback)"""
    if c is None:
        return "boxes: amber = detections (score >= 0.3), dashed = GT (error colours not available for this frame)"
    return f"boxes right {c['tp']} (amber), false {c['fp']} (green tint), missed {c['fn']} (blue tint, dashed), no care {c['nocare']} (violet), scored window"


def past_box_shapes(boxes_then, meta_then, meta_now, alpha: float = 0.30, color=GT_OVERLAY) -> List[Shape]:
    """the boxes of an earlier frame (`boxes_then` in the ego frame of `meta_then`) moved into the ego frame of the current frame through the two recorded poses (T_e2g) and
    drawn faded: the history of the scene behind the present. Boxes that moved with the ego car stay where the world kept them."""
    T = np.linalg.inv(np.asarray(meta_now["T_e2g"], float)) @ np.asarray(meta_then["T_e2g"], float)
    dyaw = math.atan2(T[1, 0], T[0, 0])
    moved = []
    for b in boxes_then:
        x, y, *_ = T @ np.array([b["cx"], b["cy"], 0.0, 1.0])
        moved.append(dict(b, cx=float(x), cy=float(y), **({"yaw": b["yaw"] + dyaw} if "yaw" in b else {})))
    return box_shapes(moved, tuple(color) + (alpha,), ":", 0.9)


def radar_points(rad) -> Shape:
    """the radar detections of the window as ONE point layer: colour = the direction of the radial velocity (cool blue = approaching, warm orange = receding, grey = about
    standing still; saturation grows with |vr| up to 6 m/s), transparency = the strength of the return (RCS, relative to this frame: at least 50 % opaque for the weakest)"""
    vr = np.nan_to_num(np.asarray(rad["vr"], float), nan=0.0)                         # a Doppler that was not measured counts as standing
    k = np.clip(np.abs(vr) / 6.0, 0, 1)[:, None]                                      # 0 = standing (grey), 1 = 6 m/s and more (saturated)
    base = np.where((vr < 0)[:, None], np.asarray((0.15, 0.65, 1.0)), np.asarray((1.0, 0.50, 0.10)))
    rgba = np.ones((len(vr), 4))
    rgba[:, :3] = np.asarray((0.55, 0.55, 0.55)) * (1 - k) + base * k
    rcs = np.asarray(rad["rcs"], float) if "rcs" in rad else np.zeros_like(vr)
    ok = np.isfinite(rcs)
    lo, hi = (np.percentile(rcs[ok], 5), np.percentile(rcs[ok], 95)) if ok.sum() > 1 else (0.0, 1.0)
    rgba[:, 3] = 0.5 + 0.5 * np.clip(np.where(ok, (rcs - lo) / max(hi - lo, 1e-6), 0.0), 0, 1)
    return Shape("points", np.c_[np.asarray(rad["y"], float), np.asarray(rad["x"], float)], rgba, "-", 0.6)


def speed_dir_rgb(v, vmax: float) -> np.ndarray:
    """a velocity field (2, X, Y) m/s -> RGB (X, Y, 3): hue = the direction of travel (red +x ahead, green-cyan to the left, blue-violet behind, see the wheel note), brightness =
    the speed (black = standing still, full colour = `vmax` and faster)"""
    import matplotlib.colors as mc
    v = np.asarray(v, np.float32)
    sp = np.hypot(v[0], v[1])
    hsv = np.stack([(np.arctan2(v[1], v[0]) / (2 * np.pi)) % 1.0, np.ones_like(sp), np.clip(sp / max(vmax, 1e-6), 0, 1)], -1)
    return mc.hsv_to_rgb(hsv).astype(np.float32)


def bce_map(p, g, eps: float = 1e-4) -> np.ndarray:
    """the binary cross-entropy of every cell of a probability map against the ground truth (what the layer's training loss is made of, per cell)"""
    p = np.clip(np.asarray(p, np.float64), eps, 1 - eps)
    g = np.clip(np.asarray(g, np.float64), 0.0, 1.0)
    return -(g * np.log(p) + (1 - g) * np.log(1 - p))


def loss_overlay(base_rgb, loss, vmax: float = 2.0, strength: float = 0.85) -> np.ndarray:
    """a loss map as a transparent heat (a ramp without red: yellow, then white; opacity = the loss relative to `vmax`) over a picture: where the loss sits, on the ground truth it
    belongs to"""
    t = np.clip(np.asarray(loss, np.float32) / vmax, 0, 1)
    yellow, white = np.asarray((0.95, 0.85, 0.15), np.float32), np.ones(3, np.float32)
    heat = np.where((t < 0.6)[..., None], yellow, yellow + (white - yellow) * ((t - 0.6) / 0.4)[..., None])
    a = (t * strength)[..., None]
    return (np.asarray(base_rgb, np.float32) * (1 - a) + heat * a).astype(np.float32)


def moving_layers(prob, vel, gt_prob, gt_vel, groups=OBJECT_LAYERS):
    """the moving objects as one pixel layer: (predicted, ground truth) boolean maps = a cell of any object class (prob > 0.5) whose velocity is above 0.5 m/s. The stationary
    objects are the object cells that are not moving."""
    shape = np.asarray(gt_vel[0]).shape

    def objs(pr):
        have = [np.asarray(pr[n], np.float32) > 0.5 for n in groups if n in pr]
        return np.any(have, axis=0) if have else np.zeros(shape, bool)               # a model / GT without object layers: nothing moves
    po, go = objs(prob), objs(gt_prob)
    ps = np.hypot(vel[0], vel[1]) > MOVING_SPEED if vel is not None else np.zeros(shape, bool)
    gs = np.hypot(gt_vel[0], gt_vel[1]) > MOVING_SPEED
    return (po & ps), (go & gs), (po & ~ps), (go & ~gs)


def fov_shapes() -> List[Shape]:
    return [Shape("line", [(0, 0), (sg * 60 * math.tan(FRONT_HALF_FOV), 60)], "0.62", ":", 0.7) for sg in (-1, 1)]


def bev(grid, kind: str, title: str, data, shapes: Sequence[Shape] = (), fov: bool = True, **kw) -> Panel:
    """a bird's-eye panel: extent, +y on the left, x forward"""
    return Panel(kind, title, data, extent=extent(grid), xlim=(grid.y_max, grid.y_min), ylim=(grid.x_min, grid.x_max),
                 shapes=(fov_shapes() if fov else []) + list(shapes), **kw)


def _edge(mask: np.ndarray) -> np.ndarray:
    """the cells of a boolean map that touch a cell outside it (4-neighbourhood, the map border counts as inside)"""
    pad = np.pad(mask, 1, mode="edge")
    inner = pad[1:-1, 1:-1] & pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:]
    return mask & ~inner


def composite(prob: Dict[str, np.ndarray], dist: Dict[str, np.ndarray], line_width: float = 0.35, observed: Optional[np.ndarray] = None) -> np.ndarray:
    """RGB (X, Y, 3) on the black background: the areas (alpha = probability where the model says yes, i.e. >= 0.5, in the order of COLORS; below that the cell stays
    black, so the non-road area is black as in the ground-truth picture, Mirko 2026-11-04), a black boundary along the drivable area, then crossings, the object
    layers (dark red; outside `observed`, a bool grid, the same red translucent) and the lines (from the distance field)"""
    shape = next(iter(prob.values())).shape
    img = np.zeros(shape + (3,), np.float32) + np.asarray(BG, np.float32)

    def paint(c, a):
        nonlocal img
        img = img * (1 - a[..., None]) + np.asarray(c, np.float32) * a[..., None]

    def decided(p):                                                              # the layer paints only where the model says yes: no faint haze in the non-road area
        p = np.clip(p, 0, 1)
        return np.where(p >= 0.5, p, 0.0).astype(np.float32)
    for name in ("drivable_area", "carpark_area", "walkway"):
        if name in prob:
            paint(COLORS[name], decided(prob[name]))
    if "drivable_area" in prob:                                                 # the road boundary, black, over the neighbouring walkway / car park
        paint(ROAD_EDGE, _edge(prob["drivable_area"] > 0.5).astype(np.float32))
    for name, c in COLORS.items():
        if name in ("drivable_area", "carpark_area", "walkway") or name not in prob:
            continue
        a = decided(prob[name])
        if name in OBJECT_LAYERS and observed is not None:
            paint(OUT_RED, a * np.where(observed, 0.0, OUT_ALPHA))
            paint(c, a * np.where(observed, 1.0, 0.0))
        else:
            paint(c, a)
    for name, c in LINE_COLORS.items():
        if name in dist:
            paint(c, np.clip(1 - dist[name] / line_width, 0, 1))
    return img


def tolerant_error_rgb(P, G, valid, tol_cells: float = 1.0, colors: Optional[Dict[str, tuple]] = None) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """error picture of two thin cell sets with a tolerance (a cell within tol_cells of the other set counts as found), in the colours of `colors`
    (default: the lane divider's, ERR): right = the class colour, false = predicted but nothing of the ground truth near (class colour + green), missed = ground truth
    without prediction near (class colour + blue), black background, dark grey = not evaluated. -> (rgb, (tp, fp, fn))"""
    P, G, valid = np.asarray(P, bool), np.asarray(G, bool), np.asarray(valid, bool)
    c = colors or ERR["lane_divider"]
    DG, DP = dilate(G, tol_cells), dilate(P, tol_cells)
    img = np.zeros(P.shape + (3,), np.float32) + np.asarray(c["tn"], np.float32)
    ok = (P & DG | G & DP) & valid
    fp, fn = P & ~DG & valid, G & ~DP & valid
    img[ok] = c["tp"]
    img[fp] = c["fp"]
    img[fn] = c["fn"]
    img[~valid] = c["skip"]
    return img, (int((P & DG & valid).sum()), int(fp.sum()), int(fn.sum()))


def _box_corners(b) -> np.ndarray:
    cx, cy, cz = b["center"]
    w, l, h = b["size"]
    c, s = math.cos(b["yaw"]), math.sin(b["yaw"])
    loc = np.array([[sx * l / 2, sy * w / 2, sz * h / 2] for sz in (-1, 1) for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1))])
    return np.stack([cx + c * loc[:, 0] - s * loc[:, 1], cy + s * loc[:, 0] + c * loc[:, 1], cz + loc[:, 2]], 1)


def _densify(pl, step=1.0):
    pl = np.asarray(pl, float)
    out = [pl[0]]
    for a, b in zip(pl[:-1], pl[1:]):
        n = max(1, int(np.hypot(*(b - a)) / step))
        out.extend(a + (b - a) * (i / n) for i in range(1, n + 1))
    return np.array(out)


def _project(meta, xyz, color, lw, near=0.5) -> List[Shape]:
    """a 3D polyline in the ego frame -> line shapes in ORIGINAL image pixels, every segment clipped at the near plane"""
    K, T_c2e = np.asarray(meta["K"], float), np.asarray(meta["T_c2e"], float)
    P = (np.c_[xyz, np.ones(len(xyz))] @ np.linalg.inv(T_c2e).T)[:, :3]
    out = []
    for a, b in zip(P[:-1], P[1:]):
        if a[2] < near and b[2] < near:
            continue
        if a[2] < near or b[2] < near:
            t = (near - a[2]) / (b[2] - a[2])
            c = a + t * (b - a)
            a, b = (c, b) if a[2] < near else (a, c)
        uv = [(K[0, 0] * p[0] / p[2] + K[0, 2], K[1, 1] * p[1] / p[2] + K[1, 2]) for p in (a, b)]
        out.append(Shape("line", uv, color, "-", lw))
    return out


def camera_gt_shapes(g) -> List[Shape]:
    """the ground truth drawn into the original front-camera image: 3D boxes (vehicle red, pedestrian magenta, two-wheeler
    orange, grey = ignored: no lidar / radar point) and the exact map lines (lane dividers yellow, road dividers orange,
    centre-lines cyan). If the paint and the lines do not meet, the GT or the projection is wrong."""
    meta = g["meta"]
    out: List[Shape] = []
    for layer, pls in meta.get("vectors", {}).items():
        for pl in pls:
            d = _densify(pl)
            keep = (d[:, 0] > -5) & (d[:, 0] < 70) & (np.abs(d[:, 1]) < 30)
            for run_ in np.split(np.arange(len(d)), np.where(np.diff(keep.astype(int)) != 0)[0] + 1):
                if keep[run_[0]] and len(run_) >= 2:
                    out += _project(meta, np.c_[d[run_], np.zeros(len(run_))], LINE_RGB.get(layer, "c"), 0.9 if layer == "centreline" else 1.4)
    for b in meta["boxes"]:
        if 0.5 < b["center"][0] < 70:
            xyz = _box_corners(b)
            col = GROUP_COLOR.get(b["group"], "w") if b.get("observed", True) else (0.7, 0.7, 0.7)
            for i, j in EDGES:
                out += _project(meta, xyz[[i, j]], col, 1.0 if b.get("observed", True) else 0.8)
    return out


def _caption(counts, what) -> str:
    tp, fp, fn = counts
    return f"{what}: right {tp}  false {fp}  missed {fn}"


def centreline_panels(grid, g, o, valid, step: int = 4, length: float = 2.0) -> List[Panel]:
    """The 1 m centre-line head of a frame (no panel for a model without it, or a GT without the layer): (1) its presence against the GT centre-line cells with the
    tolerance of the numbers (+-1 m: the cyan class colour right, tinted green false, tinted blue missed, dark grey not evaluated), (2) its direction: arrows of `length` m on every `step`-th GT cell,
    GT cyan (the colour of the camera panel's centre-lines), predicted amber, on the predicted presence (grey). o: one dt of the Prediction (`prob`, `dir`)."""
    if not o.get("dir") or "centreline" not in o["prob"] or "centreline" not in g or "centreline_dir" not in g:
        return []
    prob, G = o["prob"]["centreline"], np.asarray(g["centreline"], np.float32) > 0.5
    err, counts = tolerant_error_rgb(prob > 0.5, G, valid, tol_cells=1.0 / grid.res, colors=ERR["centreline"])
    pd, gd = np.asarray(o["dir"]["centreline"], np.float32), np.asarray(g["centreline_dir"], np.float32)
    D = G & valid & (np.hypot(gd[0], gd[1]) > 0.5)                      # the cells the angle is scored on (metrics.SegMeter.add_directions: a GT direction needs |dir| > 0.5)
    cells = np.argwhere(D)
    ang = np.degrees(angle_error(pd, gd)[D])
    arrows = []
    for ix, iy in cells[::step]:
        x0, y0 = grid.x_min + (ix + 0.5) * grid.res, grid.y_min + (iy + 0.5) * grid.res
        for vec, color in ((gd[:, ix, iy], LINE_RGB["centreline"]), (pd[:, ix, iy] / max(float(np.hypot(*pd[:, ix, iy])), 1e-6), DET_COLOR)):
            arrows.append(Shape("arrow", (y0, x0, y0 + length * vec[1], x0 + length * vec[0]), color, "-", 1.0))
    return [bev(grid, "rgb", "25 " + _caption(counts, "centre-line head, 1 m (+-1 m)"), err),
            bev(grid, "rgb", f"26 centre-line direction on the GT cells: GT cyan, predicted amber (mean error {ang.mean() if len(ang) else float('nan'):.1f} deg, {len(ang)} cells)",
                prob[..., None] * np.asarray((0.45, 0.45, 0.45), np.float32), arrows)]


# ------------------------------------------------------------------------------------------------ views
class BevView(View):
    """shared context of the BEV views: the spec of the head layers, the grid and the class names"""

    def __init__(self, task):
        self.t = task

    # -- unpack
    def _gt0(self, sample):
        return sample.reference[0.0]

    def _preds(self, pred: Prediction, dt=0.0):
        return pred.outputs["dt"][dt]

    def pred_boxes(self, sample: Sample, o, dt=0.0, past: bool = False):
        """the boxes of a prediction panel with the errors in colour (error_box_shapes), the ground truth of the frame behind them and, for dt 0, the ground-truth boxes of the
        frame half a second earlier faded (the history of the scene) -> (shapes, counts)"""
        from evaluation.bev_fusion_area import fusion_area, gt_with_shares
        g = sample.reference[dt]
        try:
            gts, shares = gt_with_shares(g, self.t.groups)
            shapes, c = error_box_shapes(o["dets"], gts, shares, fusion_area(g))
        except Exception:                                             # noqa: BLE001  (a GT without the fusion layers, an odd box: the picture must not cost the evaluation its cards)
            return box_shapes(gt_boxes(g, self.t.groups), GT_OVERLAY, "--", 0.7) + box_shapes([d for d in o["dets"] if d.get("score", 1.0) >= 0.3], DET_COLOR), None
        if past and dt == 0.0 and -0.5 in sample.reference:
            try:
                shapes = past_box_shapes(gt_boxes(sample.reference[-0.5], self.t.groups), sample.reference[-0.5]["meta"], g["meta"]) + shapes
            except Exception:                                         # noqa: BLE001  (no poses in the meta, a singular pose)
                pass
        return shapes, c


class OverviewView(BevView):
    name = "overview"
    description = "camera with ground truth | GT | prediction | error maps (drivable area, vehicle, lane dividers)"

    def render(self, sample: Sample, pred: Prediction, ctx: ViewContext) -> Page:
        t, grid = self.t, self.t.grid
        g0 = self._gt0(sample)
        o = self._preds(pred)
        m = masks(g0)
        gtb = gt_boxes(g0, t.groups)
        k = layer_index(t.spec)
        gt_prob = {n: np.asarray(g0[k[n][2]], np.float32) for n in o["prob"]}
        gt_dist = {n: np.asarray(g0[n + "_dist"], np.float32) for n in o["dist"]}
        bshapes, bx = self.pred_boxes(sample, o, 0.0, past=True)
        P, G = o["prob"]["drivable_area"] > 0.5, np.asarray(g0["drivable_area"], np.float32) > 0.5
        e_drv = _err(P, G, m["fov"], "drivable_area")
        Pv, Gv = o["prob"]["vehicle"] > 0.5, np.asarray(g0["vehicle"], np.float32) > 0.5
        e_veh = _err(Pv, Gv, m["obj_fov"], "vehicle")
        e_lane, c_lane = tolerant_error_rgb(line_cells(o["dist"]["lane_divider"], grid.res), line_cells(gt_dist["lane_divider"], grid.res),
                                            m["fov"])
        cam = sample.get("camera")
        try:
            cam_shapes = camera_gt_shapes(g0)
        except KeyError:                                          # a GT generator without calibration / vectors in its meta
            cam_shapes = []
        cam_panel = (Panel("image", "front camera (original) with the ground truth: boxes + map lines", cam, shapes=cam_shapes,
                           note="paint and lines must meet; boxes on the vehicles") if cam is not None
                     else Panel("text", "front camera", "not available"))
        panels = [cam_panel,
                  bev(grid, "rgb", "ground truth (areas, lines, vehicles)", composite(gt_prob, gt_dist, observed=observed_area(g0)), box_shapes(gtb, GT_BOX_COLOR, observed=observed_area(g0))),
                  bev(grid, "rgb", f"prediction: {box_caption(bx)}; faded = 0.5 s ago",
                      composite(o["prob"], o["dist"], observed=observed_area(g0)), bshapes),
                  bev(grid, "rgb", _caption(error_counts(P, G, m["fov"]), "drivable area (cells)"), e_drv),
                  bev(grid, "rgb", _caption(error_counts(Pv, Gv, m["obj_fov"]), "vehicle footprint (cells)"), e_veh),
                  bev(grid, "rgb", _caption(c_lane, "lane dividers (+-0.5 m)"), e_lane)]
        return Page(title="BEV overview", panels=panels, ncols=3, legend=ERR_LEGEND, panel_size=(5.0, 4.4))


def _err(P, G, valid, cls: str = "drivable_area"):
    from Evaluation.render.figure import error_rgb
    return error_rgb(P, G, valid, colors=ERR[cls])



class OutputsView(BevView):
    name = "outputs"
    description = "detailed results: GT, prediction, errors, dt +-0.5 s, splat mass, scores, speed"

    def layer_panels(self, grid, o0, g0, gt_prob, gt_dist, m, vel, gvel):
        """the semantic output as two layers, each ground truth beside the system result: (a) MOVING objects (vehicles, pedestrians, two-wheelers with a velocity above
        0.5 m/s) as pixels, red, right = red, false = red + green, missed = red + blue; (b) the STATIONARY world (areas, lines, the objects that stand); then the loss of
        the drivable-area / vehicle layers and the velocity error as a heat over the GT. Cells of unknown velocity (flow_weight 0: GT velocity 0 means "not known") are left out
        of the moving / stationary split and of the velocity error. A failure here costs the page these panels, nothing else."""
        try:
            return self._layer_panels(grid, o0, g0, gt_prob, gt_dist, m, vel, gvel)
        except Exception:                                                     # noqa: BLE001
            return [Panel("text", "25 - 31 moving / stationary layers and loss heat", "not available for this frame")]

    def _layer_panels(self, grid, o0, g0, gt_prob, gt_dist, m, vel, gvel):
        if vel is None:                                                       # this subject predicts no velocity: "moving" and "stationary" cannot be compared
            return [Panel("text", "25 - 28 moving objects / stationary world", "this subject has no velocity output: no moving / stationary comparison"),
                    *self._heat_panels(grid, o0, g0, gt_prob, gt_dist, m, None, gvel)]
        known = np.asarray(g0["flow_weight"], np.float32) > 0 if "flow_weight" in g0 else np.ones(np.asarray(gvel[0]).shape, bool)
        pm, gm, ps, gs = (x & known for x in moving_layers(o0["prob"], vel, gt_prob, gvel))
        valid = m["obj_fov"] & known
        stat_pred = {n: (a if n not in OBJECT_LAYERS else np.where(ps, a, 0.0)) for n, a in o0["prob"].items()}
        stat_gt = {n: (a if n not in OBJECT_LAYERS else np.where(gs, a, 0.0)) for n, a in gt_prob.items()}
        obs = observed_area(g0)
        heat = self._heat_panels(grid, o0, g0, gt_prob, gt_dist, m, vel, gvel)
        return [bev(grid, "rgb", "25 moving objects, ground truth (velocity above 0.5 m/s, known velocity)", np.where(gm[..., None], np.asarray(MOVING_RED, np.float32), np.asarray(BG, np.float32))),
                bev(grid, "rgb", _caption(error_counts(pm, gm, valid), "26 moving objects, system (red right, + green false, + blue missed)"), _layer_err(pm, gm, valid, MOVING_RED)),
                bev(grid, "rgb", "27 stationary world, ground truth (areas, lines, standing objects)", composite(stat_gt, gt_dist, observed=obs)),
                bev(grid, "rgb", "28 stationary world, system", composite(stat_pred, o0["dist"], observed=obs))] + heat

    def _heat_panels(self, grid, o0, g0, gt_prob, gt_dist, m, vel, gvel):
        """the loss of the drivable-area / vehicle layers and the velocity error as a heat over the GT"""
        known = np.asarray(g0["flow_weight"], np.float32) > 0 if "flow_weight" in g0 else np.ones(np.asarray(gvel[0]).shape, bool)
        obs = observed_area(g0)
        heat = []
        for name, title in (("drivable_area", "29 drivable area"), ("vehicle", "30 vehicle footprint")):
            if name in o0["prob"] and name in gt_prob:
                lm = bce_map(o0["prob"][name], gt_prob[name])
                v_ = m["fov"] if name == "drivable_area" else m["obj_fov"]
                heat.append(bev(grid, "rgb", f"{title}: loss per cell (BCE) as a heat over the GT (yellow, white = high); mean on the evaluated cells ({'camera cells' if name == 'drivable_area' else 'object cells'}) "
                                f"{float(lm[v_].mean()) if v_.any() else float('nan'):.3f}", loss_overlay(composite(gt_prob, gt_dist, observed=obs), np.where(v_, lm, 0.0))))
        if vel is not None:
            objs = np.any([np.asarray(gt_prob[n]) > 0.5 for n in OBJECT_LAYERS if n in gt_prob], axis=0) if any(n in gt_prob for n in OBJECT_LAYERS) else np.zeros(known.shape, bool)
            lv = np.hypot(vel[0] - gvel[0], vel[1] - gvel[1]) * (objs & known)
            heat.append(bev(grid, "rgb", "31 velocity error on the object cells of known velocity (|v_pred - v_GT|, white = 4 m/s) as a heat over the GT",
                            loss_overlay(composite(gt_prob, gt_dist, observed=obs), lv, vmax=4.0)))
        return heat

    def render(self, sample: Sample, pred: Prediction, ctx: ViewContext) -> Page:
        t, grid = self.t, self.t.grid
        g0 = self._gt0(sample)
        gtb = gt_boxes(g0, t.groups)
        o0 = self._preds(pred)
        k = layer_index(t.spec)
        gt_prob = {n: np.asarray(g0[k[n][2]], np.float32) for n in o0["prob"]}
        gt_dist = {n: np.asarray(g0[n + "_dist"], np.float32) for n in o0["dist"]}
        m = masks(g0)
        bs0, bx0 = self.pred_boxes(sample, o0, 0.0, past=True)
        P, G = o0["prob"]["drivable_area"] > 0.5, np.asarray(g0["drivable_area"], np.float32) > 0.5
        pv, gv = o0["prob"]["vehicle"] > 0.5, np.asarray(g0["vehicle"], np.float32) > 0.5
        panels = [bev(grid, "rgb", "13 ground truth, dt 0 (areas, lines, vehicles)", composite(gt_prob, gt_dist, observed=observed_area(g0)), box_shapes(gtb, GT_BOX_COLOR, observed=observed_area(g0))),
                  bev(grid, "rgb", f"14 prediction, dt 0: {box_caption(bx0)}",
                      composite(o0["prob"], o0["dist"], observed=observed_area(g0)), bs0),
                  bev(grid, "rgb", "15 drivable area: class colour = right, green tint = false, blue tint = missed, dark grey = not seen by the camera", _err(P, G, m["fov"], "drivable_area")),
                  bev(grid, "rgb", "16 vehicle footprint: same convention (class colour right, green tint false, blue tint missed)", _err(pv, gv, m["obj_fov"], "vehicle"))]
        for dt in (0.5, -0.5):
            if dt not in pred.outputs["dt"]:
                panels.append(Panel("text", f"dt {dt:+.1f}", "not decoded"))
                continue
            od = pred.outputs["dt"][dt]
            img = composite(od["prob"], od["dist"], observed=observed_area(sample.reference[dt]))
            shade = od["mass"] < 0.5 if od["mass"] is not None else np.zeros(img.shape[:2], bool)
            img = np.where(shade[..., None], img * 0.45 + np.array([0.35, 0.25, 0.0]) * 0.55, img)
            bsd, bxd = self.pred_boxes(sample, od, dt)
            panels.append(bev(grid, "rgb", f"prediction at dt {dt:+.1f} s; brown = unknown cells (splat mass < 0.5); {box_caption(bxd)}", img, bsd))
        if 0.5 in pred.outputs["dt"] and pred.outputs["dt"][0.5]["mass"] is not None:
            panels.append(bev(grid, "raster", "19 splat mass at dt +0.5 s (1 = a source cell lands here, 0 = nothing does)",
                              pred.outputs["dt"][0.5]["mass"], fov=False, cmap="viridis", vmin=0, vmax=1.5, colorbar=True))
        scores = np.array([d["score"] for d in o0["dets"]], float)
        hist = np.histogram(scores, bins=np.linspace(0, 1, 21))[0]
        panels.append(Panel("bars", f"20 detection scores in bins of 0.05 (>= 0.3: {int((scores >= 0.3).sum())}; GT in view: {len(gtb)})",
                            {f"{b * 0.05:.2f}": int(h) for b, h in enumerate(hist)}))
        vel, gvel = o0.get("vhat"), (np.asarray(g0["velocity"], np.float32) if "velocity" in g0 else None)
        veh = np.asarray(g0["vehicle"], np.float32) > 0.5
        gsp = np.hypot(gvel[0], gvel[1]) if gvel is not None else None
        vmax = max(2.0, float(np.percentile(gsp[veh], 99)) if gsp is not None and veh.any() else 2.0)
        wheel = "hue = direction of travel (red ahead, green to the left, cyan behind, violet to the right), brightness = speed"
        if gsp is not None:
            panels.append(bev(grid, "rgb", f"21 GT velocity of the vehicle cells: {wheel} (full colour = {vmax:.1f} m/s)", speed_dir_rgb(np.where(veh, 1.0, 0.0) * gvel, vmax),
                              box_shapes(gtb, "0.55", "--", 0.7)))
        if vel is not None:
            sp = np.hypot(vel[0], vel[1])
            panels.append(bev(grid, "rgb", "22 predicted velocity on the same cells (same scale and colours)", speed_dir_rgb(np.where(veh, 1.0, 0.0) * vel, vmax),
                              box_shapes(gtb, "0.55", "--", 0.7)))
            panels.append(bev(grid, "rgb", f"23 predicted velocity everywhere (full colour = {vmax:.1f} m/s, the same scale as 21 / 22; the predicted maximum is {sp.max():.1f} m/s; the static world should stay black)", speed_dir_rgb(vel, vmax),
                              box_shapes(gtb, "0.55", "--", 0.7)))
            txt = ("no vehicle cells" if gsp is None or not veh.any() else
                   f"vehicle cells: GT speed mean {gsp[veh].mean():.2f}, predicted mean {sp[veh].mean():.2f} m/s,\n"
                   f"correlation {np.corrcoef(gsp[veh], sp[veh])[0, 1] if gsp[veh].std() > 0 and sp[veh].std() > 0 else float('nan'):.2f}")
            panels.append(Panel("text", "24 speed on vehicle cells: predicted against GT", txt))
        if gvel is not None:
            panels += self.layer_panels(grid, o0, g0, gt_prob, gt_dist, m, vel, gvel)
        return Page(title="BEV outputs", panels=panels, ncols=4, legend=ERR_LEGEND, panel_size=(4.4, 4.0))


def _layer_err(P, G, valid, color):
    from Evaluation.render.figure import error_rgb
    return error_rgb(P, G, valid, colors=class_colors(color))



class StagesView(BevView):
    name = "stages"
    description = "the stages of the chain (needs the taps)"
    needs_taps = True

    def render(self, sample: Sample, pred: Prediction, ctx: ViewContext) -> Page:
        t, grid = self.t, self.t.grid
        g0 = self._gt0(sample)
        boxes = box_shapes(gt_boxes(g0, t.groups), GT_OVERLAY, "--", 0.7)
        batch, tp = sample.inputs, pred.taps
        cam = np.asarray(g0["camera"], np.float32) > 0
        panels: List[Panel] = []
        img = sample.get("camera_model")
        if img is not None:
            panels.append(Panel("image", "1 CAM_FRONT as the model sees it (crop)", img, note="not black, not shifted, the front camera"))
        if "prob" in batch and t.depth_centres is not None:
            prob = batch["prob"][0, 0].float().numpy()
            dc = t.depth_centres()
            exp_d = (prob * dc[:, None, None]).sum(0)
            ent = -(prob * np.log(prob + 1e-9)).sum(0) / math.log(prob.shape[0])
            h, w = (img.shape[:2] if img is not None else (256, 704))
            big = np.kron(exp_d, np.ones((16, 16)))[:h, :w]
            if img is not None:
                panels.append(Panel("image", "2 camera head: expected depth per pixel group (blue 1 m .. red 60 m)", _overlay(img, big),
                                    note="depth grows from the bottom of the image upwards"))
            else:
                panels.append(Panel("raster", "2 camera head: expected depth per pixel group (m)", big, cmap="turbo", vmin=1, vmax=60,
                                    colorbar=True, origin="upper", note="depth grows from the bottom of the image upwards"))
            panels.append(Panel("raster", "3 depth uncertainty (entropy / max; 0 = sure, 1 = flat)", ent, cmap="viridis", vmin=0, vmax=1,
                                colorbar=True, origin="upper", note="the road is sure, sky and object edges are uncertain"))
        rad = batch["radar"]
        pts = [radar_points({k: v.numpy() for k, v in rad.items()})]
        panels.append(bev(grid, "rgb", f"4 radar detections of the window ({len(rad['x'])}): colour = direction of the Doppler (cool = approaching, warm = receding), opacity = strength (RCS)",
                          np.zeros((2, 2, 3)), pts + boxes, note="moving cars carry a Doppler colour that differs from their surroundings"))

        def emap(name, title, note=""):
            if name in tp:
                a = np.asarray(tp[name])
                return bev(grid, "raster", title, energy(a) if a.ndim == 3 else a, boxes, colorbar=True, note=note)
            return None
        lift = tp.get("lift")
        if lift is not None:
            nz = len(t.z_levels)
            panels.append(emap("lift", "5 camera BEV after the lift (feature energy, all heights)", "energy only inside the camera cone"))
            per = lift.reshape(nz, -1, lift.shape[-2], lift.shape[-1])
            e_z = {f"{z:g}": float((energy(per[kz]) * cam).sum() / max(cam.sum(), 1)) for kz, z in enumerate(t.z_levels)}
            panels.append(Panel("bars", "6 where the energy sits in height (mean over camera cells; z in m)", e_z,
                                note="road level and vehicles carry the energy; one single height is a red flag"))
        for name, title, note in (("cam_reduce", "7 camera BEV after reduce, current frame (energy)", "the same wedge"),
                                  ("cam_temporal", "8 camera BEV after temporal fusion (energy)", "the cone is filled further and smoother"),
                                  ("temporal_agg", "9 what the past frames supply (energy)", "small where nothing moved, larger at objects"),
                                  ("radar_feat", "11 radar branch features (energy)", "a smeared version of the raster")):
            p = emap(name, title, note)
            if p is not None:
                panels.append(p)
        if "radar_raster" in tp:
            panels.append(bev(grid, "raster", "10 radar raster: detections per cell (count channel)", np.asarray(tp["radar_raster"])[0], boxes,
                              cmap="viridis", colorbar=True))
        if "trunk" in tp:
            panels.append(bev(grid, "rgb", "12 trunk output: feature PCA as RGB (same colour = same kind of cell)", pca_rgb(np.asarray(tp["trunk"])),
                              boxes, note="colour regions: road, off-road, objects"))
        if "camera_visibility" in tp:
            panels.append(bev(grid, "raster", "camera visibility (cells the camera sees; 1 = visible)", np.asarray(tp["camera_visibility"]),
                              boxes, cmap="gray", vmin=0, vmax=1))
        panels += centreline_panels(grid, g0, self._preds(pred), cam)
        return Page(title="BEV stages", panels=[p for p in panels if p is not None], ncols=4, panel_size=(4.4, 4.0))


class DashboardView(BevView):
    name = "dashboard"
    description = "every stage of one frame in one compact picture (video frame)"
    needs_taps = True

    def render(self, sample: Sample, pred: Prediction, ctx: ViewContext) -> Page:
        t, grid = self.t, self.t.grid
        g0 = self._gt0(sample)
        gtb = gt_boxes(g0, t.groups)
        boxes = box_shapes(gtb, GT_OVERLAY, "--", 0.7)
        batch, tp = sample.inputs, pred.taps
        o0 = self._preds(pred)
        k = layer_index(t.spec)
        panels: List[Panel] = []
        img = sample.get("camera_model")
        if img is not None and "prob" in batch and t.depth_centres is not None:
            ed = (batch["prob"][0, 0].float().numpy() * t.depth_centres()[:, None, None]).sum(0)
            h, w = img.shape[:2]
            panels.append(Panel("image", "input + camera head depth (blue near, red far)", _overlay(img, np.kron(ed, np.ones((16, 16)))[:h, :w])))
        rad = batch["radar"]
        panels.append(bev(grid, "rgb", f"radar detections ({len(rad['x'])}): colour = Doppler direction, opacity = strength", np.zeros((2, 2, 3)),
                          [radar_points({k: v.numpy() for k, v in rad.items()})] + boxes))
        for name, title in (("cam_temporal", "camera BEV after lift + temporal fusion (energy)"), ("radar_feat", "radar branch (energy)")):
            if name in tp:
                a = np.asarray(tp[name])
                panels.append(bev(grid, "raster", title, energy(a) if a.ndim == 3 else a, boxes))
        if "trunk" in tp:
            panels.append(bev(grid, "rgb", "trunk output (feature PCA as RGB)", pca_rgb(np.asarray(tp["trunk"])), boxes))
        gt_prob = {n: np.asarray(g0[k[n][2]], np.float32) for n in o0["prob"]}
        gt_dist = {n: np.asarray(g0[n + "_dist"], np.float32) for n in o0["dist"]}
        panels.append(bev(grid, "rgb", "ground truth now", composite(gt_prob, gt_dist, observed=observed_area(g0)), box_shapes(gtb, GT_BOX_COLOR, observed=observed_area(g0))))
        bsn, bxn = self.pred_boxes(sample, o0, 0.0, past=True)
        panels.append(bev(grid, "rgb", f"prediction now: {box_caption(bxn)}; faded = 0.5 s ago",
                          composite(o0["prob"], o0["dist"], observed=observed_area(g0)), bsn))
        for dt in (0.5, -0.5):
            if dt in pred.outputs["dt"]:
                od = pred.outputs["dt"][dt]
                bsd, bxd = self.pred_boxes(sample, od, dt)
                panels.append(bev(grid, "rgb", f"prediction at dt {dt:+.1f} s: {box_caption(bxd)}",
                                  composite(od["prob"], od["dist"], observed=observed_area(sample.reference[dt])), bsd))
        if o0.get("vhat") is not None:
            panels.append(bev(grid, "rgb", "predicted velocity: hue = direction, brightness = speed (full colour = 10 m/s)", speed_dir_rgb(o0["vhat"], 10.0), boxes))
        return Page(title="BEV dashboard", panels=panels, ncols=5, panel_size=(4.2, 3.9))


def _overlay(img: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """camera image with the expected depth blended in (turbo, 1..60 m)"""
    import matplotlib.cm as cm
    rgb = cm.turbo(np.clip((depth - 1) / 59.0, 0, 1))[..., :3]
    return (np.clip(0.5 * img.astype(np.float32) / 255.0 + 0.5 * rgb, 0, 1) * 255).astype(np.uint8)


# ------------------------------------------------------------------------------------------------ self-test of the pure picture helpers
def selftest():
    # radar points: opacity >= 50 % and rising with the strength, colour by the sign of the Doppler
    rad = {"x": np.array([10.0, 20.0, 30.0]), "y": np.array([0.0, 1.0, 2.0]), "vr": np.array([-15.0, 0.0, 15.0]), "rcs": np.array([0.0, 10.0, 20.0])}
    c = radar_points(rad).color
    assert c.shape == (3, 4) and abs(c[0, 3] - 0.5) < 1e-6 and abs(c[2, 3] - 1.0) < 1e-6 and 0.5 < c[1, 3] < 1.0, c
    assert c[0, 2] > c[0, 0] and c[2, 0] > c[2, 2], "approaching = cool (blue), receding = warm (red)"
    one = radar_points({"x": np.array([5.0]), "y": np.array([0.0]), "vr": np.array([1.0]), "rcs": np.array([3.0])}).color
    assert one.shape == (1, 4) and one[0, 3] >= 0.5, "a single return"
    # speed and direction: hue = direction, brightness = speed
    v = np.zeros((2, 2, 3), np.float32)
    v[:, 0, 0], v[:, 0, 1], v[:, 0, 2] = (10.0, 0.0), (0.0, 5.0), (-5.0, 0.0)
    img = speed_dir_rgb(v, 10.0)
    assert np.allclose(img[0, 0], (1, 0, 0)) and np.allclose(img[1, 1], 0), "ahead at full speed = red, standing = black"
    assert abs(img[0, 1].max() - 0.5) < 1e-6 and img[0, 1].argmax() == 1, "left at half speed = half-bright green side"
    assert abs(img[0, 2].max() - 0.5) < 1e-6 and img[0, 2][0] < img[0, 2][1] and img[0, 2][1] == img[0, 2][2], "behind = cyan"
    # moving objects as one layer
    prob = {"vehicle": np.array([[1.0, 1.0], [0.0, 0.0]]), "pedestrian": np.zeros((2, 2)), "drivable_area": np.ones((2, 2))}
    gprob = {"vehicle": np.array([[1.0, 1.0], [1.0, 0.0]]), "pedestrian": np.zeros((2, 2)), "drivable_area": np.ones((2, 2))}
    vel = np.zeros((2, 2, 2))
    vel[0, 0, 0] = 3.0                                                   # predicted: cell (0,0) moves
    gvel = np.zeros((2, 2, 2))
    gvel[0, 0, 0], gvel[0, 1, 0] = 3.0, 2.0                               # ground truth: cells (0,0) and (1,0) move
    pm, gm, ps, gs = moving_layers(prob, vel, gprob, gvel)
    assert pm.tolist() == [[True, False], [False, False]] and gm.tolist() == [[True, False], [True, False]]
    assert ps.tolist() == [[False, True], [False, False]] and gs.tolist() == [[False, True], [False, False]]
    # past boxes: the car drove 5 m forward since, so a box 10 m ahead then is 5 m ahead now, faded
    T_then, T_now = np.eye(4), np.eye(4)
    T_now[0, 3] = 5.0
    sh = past_box_shapes([{"group": "vehicle", "cx": 10.0, "cy": 0.0, "length": 4.0, "width": 2.0, "yaw": 0.0}], {"T_e2g": T_then}, {"T_e2g": T_now})
    poly = next(x for x in sh if x.kind == "polygon")
    assert abs(np.mean([p[1] for p in poly.xy]) - 5.0) < 1e-6 and abs(poly.color[3] - 0.30) < 1e-9
    T_now2 = np.eye(4)
    T_now2[:2, :2] = [[0, -1], [1, 0]]                                    # the car turned left by 90 degrees: a box straight ahead is now on its right
    sh2 = next(x for x in past_box_shapes([{"group": "vehicle", "cx": 10.0, "cy": 0.0, "length": 4.0, "width": 2.0, "yaw": 0.0}], {"T_e2g": T_then}, {"T_e2g": T_now2}) if x.kind == "polygon")
    assert abs(np.mean([p[0] for p in sh2.xy]) + 10.0) < 1e-6 and abs(np.mean([p[1] for p in sh2.xy])) < 1e-6, "turned frame: y = -10, x = 0"
    # errors in colour: a box right, one false, one missed, one 'no care' on each side
    area = np.ones((120, 104), bool)
    area[60:] = False                                                    # x >= 30 m is not observed
    def box(x, y, g="vehicle", s=None):
        b = {"group": g, "cx": float(x), "cy": float(y), "length": 4.0, "width": 2.0, "yaw": 0.0, "vx": 0.0, "vy": 0.0}
        if s is not None:
            b["score"] = s
        return b
    gts = [box(10, 0), box(20, 5), box(45, 0)]                             # right, missed, no care (x = 45 m, outside the observed area)
    shares = [1.0, 1.0, 0.0]
    dets = [box(10.2, 0, s=0.9), box(15, -8, s=0.8), box(46, 0, s=0.9), box(12, 12, s=0.1)]       # right, false, on the no-care box, under the score threshold
    shapes, cnt = error_box_shapes(dets, gts, shares, area)
    assert cnt == {"tp": 1, "fp": 1, "fn": 1, "nocare": 2}, cnt
    def poly_at(cx, cy):                                                   # the polygon of the box centred at (cx, cy)
        return next(x for x in shapes if x.kind == "polygon" and abs(np.mean([q[1] for q in x.xy]) - cx) < 0.05 and abs(np.mean([q[0] for q in x.xy]) - cy) < 0.05)
    rgb = lambda x: tuple(np.round(np.asarray(x.color, float), 3))        # noqa: E731
    fp_, fn_, ok_, nc_ = poly_at(15, -8), poly_at(20, 5), poly_at(10.2, 0), poly_at(45, 0)
    assert rgb(fp_) == tuple(np.round(tinted(DET_COLOR, GREEN), 3)) and fp_.ls == "-" and fp_.lw == 2.0, "a false detection: amber + green, solid, thick"
    assert rgb(fn_) == tuple(np.round(tinted(GT_OVERLAY, BLUE, 0.6), 3)) and fn_.ls == "--" and fn_.lw == 2.0, "a missed box: white + blue, dashed, thick"
    assert rgb(ok_) == tuple(np.round(DET_COLOR, 3)) and ok_.ls == "-" and ok_.lw == 1.4, "the right detection keeps the plain amber"
    assert rgb(nc_) == tuple(np.round(NOCARE_COLOR, 3)), "a no-care box is violet"
    assert rgb(poly_at(10, 0)) == tuple(np.round(GT_OVERLAY, 3)) and poly_at(10, 0).ls == "--", "the GT box of a right detection: white, dashed"
    # the scored window: a false detection 56 deg off the axis and a GT box beyond 60 m are neither counted nor coloured as errors
    dets2 = dets + [box(10, 15, s=0.9)]
    shapes2, cnt2 = error_box_shapes(dets2, gts + [box(70, 0), box(10, -15)], shares + [1.0, 1.0], area)
    assert cnt2 == cnt, (cnt2, cnt)
    # a ground-truth box just beyond the window (60.8 m) neutralises a detection inside it, as in the metric (the detection is ignored, not a false alarm)
    shapes3, cnt3 = error_box_shapes(dets + [box(59.4, 0, s=0.9)], gts + [box(60.8, 0)], shares + [1.0], area)
    assert cnt3 == dict(cnt, nocare=cnt["nocare"] + 1), (cnt3, cnt)
    far = [x for x in shapes2 if x.kind == "polygon" and abs(np.mean([q[1] for q in x.xy]) - 70) < 0.3][0]
    assert abs(np.asarray(far.color, float)[3] - OUTSIDE_COLOR[3]) < 1e-9, "outside the window: faint violet"
    # robustness: a NaN Doppler / RCS, a model without object layers, no radar strength at all
    nanrad = radar_points({"x": np.array([1.0, 2.0, 3.0]), "y": np.zeros(3), "vr": np.array([np.nan, 5.0, -5.0]), "rcs": np.array([1.0, np.nan, 3.0])}).color
    assert np.isfinite(nanrad).all() and (nanrad[:, 3] >= 0.5).all(), nanrad
    assert np.isfinite(radar_points({"x": np.array([1.0]), "y": np.zeros(1), "vr": np.zeros(1)}).color).all(), "no rcs key"
    pm0, gm0, ps0, gs0 = moving_layers({"drivable_area": np.ones((2, 2))}, None, {"drivable_area": np.ones((2, 2))}, np.ones((2, 2, 2)))
    assert not (pm0.any() or gm0.any() or ps0.any() or gs0.any()), "no object layers: nothing moves"
    # past boxes keep their heading through the pose: a box with yaw 0.5 seen from a car that turned by 90 degrees has yaw 0.5 - 90 deg
    b_then = {"group": "vehicle", "cx": 10.0, "cy": 0.0, "length": 4.0, "width": 2.0, "yaw": 0.5}
    got = next(x for x in past_box_shapes([b_then], {"T_e2g": np.eye(4)}, {"T_e2g": T_now2}) if x.kind == "polygon")
    want = next(x for x in box_shapes([dict(b_then, cx=0.0, cy=-10.0, yaw=0.5 - math.pi / 2)], (0, 0, 0, 0.3), ":", 0.9) if x.kind == "polygon")
    assert np.allclose(got.xy, want.xy, atol=1e-6), (got.xy, want.xy)
    # the heat of a loss has no red tint (hue yellow to white): green is at least 0.85 of red at every opacity
    for loss_v in (0.3, 1.0, 1.7, 2.0):
        px = loss_overlay(np.zeros((1, 1, 3), np.float32), np.array([[loss_v]]), vmax=2.0, strength=1.0)[0, 0]
        assert px[1] >= 0.85 * px[0], (loss_v, px)
    # radar: standing = grey, approaching = blue, receding = orange
    rr = radar_points({"x": np.array([1.0, 2.0, 3.0]), "y": np.zeros(3), "vr": np.array([-8.0, 0.0, 8.0]), "rcs": np.array([1.0, 2.0, 3.0])}).color
    assert rr[1, 0] == rr[1, 1] == rr[1, 2] and rr[0, 2] > rr[0, 0] and rr[2, 0] > rr[2, 2], rr
    # the loss heat: the BCE of known cells and the overlay (no loss = the picture itself)
    lm = bce_map(np.array([[0.5, 0.99], [0.01, 0.5]]), np.array([[1.0, 1.0], [1.0, 0.0]]))
    assert abs(lm[0, 0] - math.log(2)) < 1e-9 and abs(lm[0, 1] + math.log(0.99)) < 1e-9 and abs(lm[1, 0] + math.log(0.01)) < 1e-9 and abs(lm[1, 1] - math.log(2)) < 1e-9
    base = np.full((2, 2, 3), 0.3, np.float32)
    out = loss_overlay(base, np.array([[0.0, 0.0], [0.0, 4.0]]), vmax=2.0)
    assert np.allclose(out[0, 0], 0.3) and out[1, 1].mean() > 0.6, out
    print("bev_views selftest ok: radar opacity / direction, speed-direction colours, moving layer, faded past boxes through the poses, error colours of the boxes")

