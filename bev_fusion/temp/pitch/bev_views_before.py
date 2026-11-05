"""The pictures of the BEV system for the evaluation suite (evalsuite): every view returns a `Page` of panels (pure data).

Views (names as in `BEVTask.views()`):
    overview    camera with the projected ground truth | GT bird's-eye view | prediction (+ boxes) | three error maps
                (drivable area, vehicle footprint, lane dividers): grey right, RED false, BLUE missed, cream not evaluated
    outputs     the detailed result page: GT / prediction / errors, dt = +-0.5 s with the unknown cells, splat mass, scores, speed
    stages      the stages of the chain: camera heads, lift, camera BEV, temporal fusion, radar, trunk (needs the taps)
    dashboard   one compact picture of every stage of one frame (the video frame)
Bird's-eye maps are drawn with x forward (up) and +y (left of the car) on the left; arrays are (x, y) = (rows, columns).
"""
from __future__ import annotations

import math
import textwrap
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from evalsuite import Page, Panel, Prediction, Sample, Shape, View, ViewContext
from evalsuite.render.figure import ERROR_LEGEND, FN_COLOR, FP_COLOR, SKIP_COLOR, TP_COLOR, error_counts
from evalsuite.render.taps import energy, pca_rgb
from evaluation.metrics import FRONT_HALF_FOV, dilate, gt_boxes_from_meta, line_cells

COLORS = {"drivable_area": (0.80, 0.80, 0.80), "carpark_area": (0.70, 0.78, 0.90), "walkway": (0.62, 0.85, 0.62),
          "ped_crossing": (1.00, 0.70, 0.35), "stop_line": (0.80, 0.68, 0.92),
          "vehicle": (0.15, 0.35, 0.90), "pedestrian": (0.85, 0.15, 0.75), "two_wheeler": (0.10, 0.75, 0.80)}
LINE_COLORS = {"lane_divider": (0.15, 0.15, 0.15), "road_divider": (0.85, 0.10, 0.10)}
GROUP_COLOR = {"vehicle": (0.95, 0.15, 0.15), "pedestrian": (0.9, 0.2, 0.9), "two_wheeler": (1.0, 0.6, 0.1)}
LINE_RGB = {"lane_divider": (1.0, 0.9, 0.1), "road_divider": (1.0, 0.5, 0.0), "centreline": (0.1, 0.9, 0.9)}
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
MAP_LEGEND = [(n, c) for n, c in list(COLORS.items()) + list(LINE_COLORS.items())]


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


def box_shapes(boxes, color, ls="-", lw=1.2, score_min=None) -> List[Shape]:
    """box dicts (cx, cy, length, width, yaw | radius) -> overlay shapes in bird's-eye plot coordinates (horizontal = y)"""
    out = []
    for b in boxes:
        if score_min is not None and b.get("score", 1.0) < score_min:
            continue
        if "radius" in b:
            out.append(Shape("circle", (b["cy"], b["cx"], max(b["radius"], 0.3)), color, ls, lw))
        else:
            c, s = math.cos(b["yaw"]), math.sin(b["yaw"])
            hl, hw = b["length"] / 2, b["width"] / 2
            pts = [(b["cx"] + c * a - s * w, b["cy"] + s * a + c * w) for a, w in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))]
            out.append(Shape("polygon", [(y, x) for x, y in pts], color, ls, lw))
            out.append(Shape("line", [(b["cy"], b["cx"]), (b["cy"] + s * hl, b["cx"] + c * hl)], color, ls, lw))
    return out


def fov_shapes() -> List[Shape]:
    return [Shape("line", [(0, 0), (sg * 60 * math.tan(FRONT_HALF_FOV), 60)], "0.4", ":", 0.6) for sg in (-1, 1)]


def bev(grid, kind: str, title: str, data, shapes: Sequence[Shape] = (), fov: bool = True, **kw) -> Panel:
    """a bird's-eye panel: extent, +y on the left, x forward"""
    return Panel(kind, title, data, extent=extent(grid), xlim=(grid.y_max, grid.y_min), ylim=(grid.x_min, grid.x_max),
                 shapes=(fov_shapes() if fov else []) + list(shapes), **kw)


def composite(prob: Dict[str, np.ndarray], dist: Dict[str, np.ndarray], line_width: float = 0.35) -> np.ndarray:
    """RGB (X, Y, 3) of the probabilities (alpha = probability) in the order of COLORS, lines from the distance field"""
    shape = next(iter(prob.values())).shape
    img = np.ones(shape + (3,), np.float32)
    for name, c in COLORS.items():
        if name in prob:
            a = np.clip(prob[name], 0, 1)[..., None]
            img = img * (1 - a) + np.asarray(c, np.float32) * a
    for name, c in LINE_COLORS.items():
        if name in dist:
            a = np.clip(1 - dist[name] / line_width, 0, 1)[..., None]
            img = img * (1 - a) + np.asarray(c, np.float32) * a
    return img


def tolerant_error_rgb(P, G, valid, tol_cells: float = 1.0) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """error picture of two thin cell sets with a tolerance (a cell within tol_cells of the other set counts as found):
    grey = found, RED = predicted but nothing of the ground truth near, BLUE = ground truth without prediction near,
    cream = not evaluated. -> (rgb, (tp, fp, fn))"""
    P, G, valid = np.asarray(P, bool), np.asarray(G, bool), np.asarray(valid, bool)
    DG, DP = dilate(G, tol_cells), dilate(P, tol_cells)
    img = np.ones(P.shape + (3,), np.float32)
    ok = (P & DG | G & DP) & valid
    fp, fn = P & ~DG & valid, G & ~DP & valid
    img[ok] = TP_COLOR
    img[fp] = FP_COLOR
    img[fn] = FN_COLOR
    img[~valid] = SKIP_COLOR
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
        dets = [d for d in o["dets"] if d["score"] >= 0.3]
        P, G = o["prob"]["drivable_area"] > 0.5, np.asarray(g0["drivable_area"], np.float32) > 0.5
        e_drv = _err(P, G, m["fov"])
        Pv, Gv = o["prob"]["vehicle"] > 0.5, np.asarray(g0["vehicle"], np.float32) > 0.5
        e_veh = _err(Pv, Gv, m["obj_fov"])
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
                  bev(grid, "rgb", "ground truth (areas, lines, vehicles)", composite(gt_prob, gt_dist), box_shapes(gtb, "tab:blue")),
                  bev(grid, "rgb", "prediction (red boxes: score >= 0.3, dashed: GT)", composite(o["prob"], o["dist"]),
                      box_shapes(gtb, "k", "--", 0.7) + box_shapes(dets, "tab:red")),
                  bev(grid, "rgb", _caption(error_counts(P, G, m["fov"]), "drivable area (cells)"), e_drv),
                  bev(grid, "rgb", _caption(error_counts(Pv, Gv, m["obj_fov"]), "vehicle footprint (cells)"), e_veh),
                  bev(grid, "rgb", _caption(c_lane, "lane dividers (+-0.5 m)"), e_lane)]
        return Page(title="BEV overview", panels=panels, ncols=3, legend=ERROR_LEGEND, panel_size=(5.0, 4.4))


def _err(P, G, valid):
    from evalsuite.render.figure import error_rgb
    return error_rgb(P, G, valid)


def _doppler_colors(vr) -> np.ndarray:
    import matplotlib.cm as cm
    return cm.coolwarm((np.clip(np.asarray(vr, float), -15, 15) + 15) / 30.0)


class OutputsView(BevView):
    name = "outputs"
    description = "detailed results: GT, prediction, errors, dt +-0.5 s, splat mass, scores, speed"

    def render(self, sample: Sample, pred: Prediction, ctx: ViewContext) -> Page:
        t, grid = self.t, self.t.grid
        g0 = self._gt0(sample)
        gtb = gt_boxes(g0, t.groups)
        o0 = self._preds(pred)
        k = layer_index(t.spec)
        gt_prob = {n: np.asarray(g0[k[n][2]], np.float32) for n in o0["prob"]}
        gt_dist = {n: np.asarray(g0[n + "_dist"], np.float32) for n in o0["dist"]}
        m = masks(g0)
        P, G = o0["prob"]["drivable_area"] > 0.5, np.asarray(g0["drivable_area"], np.float32) > 0.5
        pv, gv = o0["prob"]["vehicle"], np.asarray(g0["vehicle"], np.float32)
        rgb = np.ones(pv.shape + (3,), np.float32)
        rgb[..., 0], rgb[..., 1], rgb[..., 2] = 1 - pv * 0.8, 1 - np.maximum(pv, gv) * 0.6, 1 - gv * 0.8
        panels = [bev(grid, "rgb", "13 ground truth, dt 0 (areas, lines, vehicles)", composite(gt_prob, gt_dist), box_shapes(gtb, "tab:blue")),
                  bev(grid, "rgb", "14 prediction, dt 0 (red boxes: score >= 0.3, dashed black: GT)", composite(o0["prob"], o0["dist"]),
                      box_shapes(gtb, "k", "--", 0.7) + box_shapes([d for d in o0["dets"] if d["score"] >= 0.3], "tab:red")),
                  bev(grid, "rgb", "15 drivable area: right grey, false red, missed blue, cream = not seen by the camera", _err(P, G, m["fov"])),
                  bev(grid, "rgb", "16 vehicle footprint: prediction red, GT blue, both = dark", rgb)]
        for dt in (0.5, -0.5):
            if dt not in pred.outputs["dt"]:
                panels.append(Panel("text", f"dt {dt:+.1f}", "not decoded"))
                continue
            od = pred.outputs["dt"][dt]
            img = composite(od["prob"], od["dist"])
            shade = od["mass"] < 0.5 if od["mass"] is not None else np.zeros(img.shape[:2], bool)
            img = np.where(shade[..., None], img * 0.45 + np.array([0.35, 0.25, 0.0]) * 0.55, img)
            panels.append(bev(grid, "rgb", f"prediction at dt {dt:+.1f} s; brown = unknown cells (splat mass < 0.5); dashed = GT then", img,
                              box_shapes(gt_boxes(sample.reference[dt], t.groups), "k", "--", 0.7) +
                              box_shapes([d for d in od["dets"] if d["score"] >= 0.3], "tab:red")))
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
        if gsp is not None:
            panels.append(bev(grid, "raster", "21 GT speed |v| of the vehicle cells (m/s)", np.where(veh, gsp, 0),
                              box_shapes(gtb, "c", "--", 0.7), cmap="viridis", vmax=vmax, colorbar=True))
        if vel is not None:
            sp = np.hypot(vel[0], vel[1])
            panels.append(bev(grid, "raster", "22 predicted speed |v| on the same cells (same colour scale)", np.where(veh, sp, 0),
                              box_shapes(gtb, "c", "--", 0.7), cmap="viridis", vmax=vmax, colorbar=True))
            panels.append(bev(grid, "raster", f"23 predicted speed everywhere (scale to its own max {sp.max():.2f} m/s)", sp,
                              box_shapes(gtb, "c", "--", 0.7), cmap="viridis", colorbar=True))
            txt = ("no vehicle cells" if gsp is None or not veh.any() else
                   f"vehicle cells: GT speed mean {gsp[veh].mean():.2f}, predicted mean {sp[veh].mean():.2f} m/s,\n"
                   f"correlation {np.corrcoef(gsp[veh], sp[veh])[0, 1] if gsp[veh].std() > 0 and sp[veh].std() > 0 else float('nan'):.2f}")
            panels.append(Panel("text", "24 speed on vehicle cells: predicted against GT", txt))
        return Page(title="BEV outputs", panels=panels, ncols=4, legend=ERROR_LEGEND, panel_size=(4.4, 4.0))


class StagesView(BevView):
    name = "stages"
    description = "the stages of the chain (needs the taps)"
    needs_taps = True

    def render(self, sample: Sample, pred: Prediction, ctx: ViewContext) -> Page:
        t, grid = self.t, self.t.grid
        g0 = self._gt0(sample)
        boxes = box_shapes(gt_boxes(g0, t.groups), "k", "--", 0.7)
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
        pts = [Shape("points", np.c_[rad["y"].numpy(), rad["x"].numpy()], _doppler_colors(rad["vr"].numpy()), "-", 0.6)]
        panels.append(bev(grid, "rgb", f"4 radar detections of the window ({len(rad['x'])}): colour = Doppler vr (blue -15 .. red +15 m/s)",
                          np.ones((2, 2, 3)), pts + boxes, note="moving cars carry a Doppler colour that differs from their surroundings"))

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
        return Page(title="BEV stages", panels=[p for p in panels if p is not None], ncols=4, panel_size=(4.4, 4.0))


class DashboardView(BevView):
    name = "dashboard"
    description = "every stage of one frame in one compact picture (video frame)"
    needs_taps = True

    def render(self, sample: Sample, pred: Prediction, ctx: ViewContext) -> Page:
        t, grid = self.t, self.t.grid
        g0 = self._gt0(sample)
        gtb = gt_boxes(g0, t.groups)
        boxes = box_shapes(gtb, "k", "--", 0.7)
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
        panels.append(bev(grid, "rgb", f"radar detections ({len(rad['x'])}), colour = Doppler", np.ones((2, 2, 3)),
                          [Shape("points", np.c_[rad["y"].numpy(), rad["x"].numpy()], _doppler_colors(rad["vr"].numpy()), "-", 0.6)] + boxes))
        for name, title in (("cam_temporal", "camera BEV after lift + temporal fusion (energy)"), ("radar_feat", "radar branch (energy)")):
            if name in tp:
                a = np.asarray(tp[name])
                panels.append(bev(grid, "raster", title, energy(a) if a.ndim == 3 else a, boxes))
        if "trunk" in tp:
            panels.append(bev(grid, "rgb", "trunk output (feature PCA as RGB)", pca_rgb(np.asarray(tp["trunk"])), boxes))
        gt_prob = {n: np.asarray(g0[k[n][2]], np.float32) for n in o0["prob"]}
        gt_dist = {n: np.asarray(g0[n + "_dist"], np.float32) for n in o0["dist"]}
        panels.append(bev(grid, "rgb", "ground truth now", composite(gt_prob, gt_dist), box_shapes(gtb, "tab:blue")))
        panels.append(bev(grid, "rgb", "prediction now (red boxes >= 0.3, dashed = GT)", composite(o0["prob"], o0["dist"]),
                          boxes + box_shapes([d for d in o0["dets"] if d["score"] >= 0.3], "tab:red")))
        for dt in (0.5, -0.5):
            if dt in pred.outputs["dt"]:
                od = pred.outputs["dt"][dt]
                panels.append(bev(grid, "rgb", f"prediction at dt {dt:+.1f} s (dashed = GT then)", composite(od["prob"], od["dist"]),
                                  box_shapes(gt_boxes(sample.reference[dt], t.groups), "k", "--", 0.7) +
                                  box_shapes([d for d in od["dets"] if d["score"] >= 0.3], "tab:red")))
        if o0.get("vhat") is not None:
            sp = np.hypot(o0["vhat"][0], o0["vhat"][1])
            panels.append(bev(grid, "raster", "predicted speed field (m/s)", sp, boxes, cmap="viridis", colorbar=True, vmax=max(2.0, float(sp.max()))))
        return Page(title="BEV dashboard", panels=panels, ncols=5, panel_size=(4.2, 3.9))


def _overlay(img: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """camera image with the expected depth blended in (turbo, 1..60 m)"""
    import matplotlib.cm as cm
    rgb = cm.turbo(np.clip((depth - 1) / 59.0, 0, 1))[..., :3]
    return (np.clip(0.5 * img.astype(np.float32) / 255.0 + 0.5 * rgb, 0, 1) * 255).astype(np.uint8)
