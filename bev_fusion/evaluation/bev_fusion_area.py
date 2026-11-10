"""Scoring on the FUSION AREA (decisions D37 / D38, first cut, additive): what our front camera + radar rig can observe is scored, the rest is "don't care" on both sides.

    fusion area       the cells of the GT layers `camera` (ground-plane projection into CAM_FRONT) x `radar` (the radar fan) x not hidden behind a box (`occluded`);
                      today the radar fan contains every camera cell, so it equals the camera cells that are not hidden (about 60 % of the grid)
    footprint share   f of a box = the share of its footprint (sampled every 0.25 m) that lies in the fusion area; outside the grid counts as outside. A GT box uses its four
                      corners (as GtGenerator.coverage.footprint_share, the stored `share_fusion` of GT v7); a detection uses its length / width / yaw, a pedestrian detection a disc
    GT classes        care f >= 0.5 | partial 0.1 <= f < 0.5 | don't care f < 0.1      (D38 (b): 0.1 and 0.5; a convention backed by records/footprint_share_study_2026-11-01.md)
    detections        ignored when more than half of the detection's own footprint lies outside the fusion area (the KITTI rule), or when it sits within 2 m of a neutral
                      GT box (partial / don't care in the headline, also one outside the scored window) of its class and within 2 m of no scored one: a detection on a don't-care
                      object is neither a hit nor a false alarm. The 2 m is the centre distance whatever the box size (the TP threshold of the errors): a detection 2 - 4 m from a
                      neutral box is NOT ignored and can be a false alarm at ap@4

    BevFusionDetMetric   `detf` (GT = care boxes; partial and don't-care boxes are neutral) and `detfp` (GT = care + partial; don't care neutral): the same numbers as `det`
                         (AP, ATE, ASE, AOE, AVE, velocity skill). Unlike `det`, the lidar-based `observed` flag of the nuScenes boxes plays no role (our rig has no lidar):
                         a box is judged by what camera and radar can observe
    BevCoverageMetric    `cov`: per class of the COMPLETE GT set (every annotated box in the scored window) the number of boxes (n_gt), the share found by ANY detection within 2 m
                         (one-to-one, nearest first over ALL boxes incl. the neutral ones and every score: the recall ceiling `recall2m`, not the recall of `detf` at 2 m) and the mean centre error of the found ones (`ate`). Classes: all, care,
                         partial, dont_care (footprint share in the fusion area); camera = f_cam >= 0.5 for the camera cells that are not hidden; with the radar returns of the
                         annotation (nuScenes `num_radar_pts`, evaluation/bev_annotation_pts.py): cam_radar (camera and >= 1 return), cam_only (camera, no return), radar_only
                         (not camera, >= 1 return), neither. Without that table the four are left out. This is the split "things a camera can see" vs "things camera and
                         radar can see" of the complete GT set.
The old numbers (`det`, `seg_fov`) keep their meaning and label (D37: old numbers keep their label, never mixed with the new ones).
    python evaluation/bev_fusion_area.py --selftest"""
from __future__ import annotations

import math
import os
import sys
from typing import Dict, Optional, Sequence

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, ".."), os.path.join(ROOT, "..", "autoresearch")]
from evalsuite import Metric, Prediction, Sample  # noqa: E402
from evaluation.bev_task import BevDetMetric  # noqa: E402
from evaluation.metrics import dt_key, gt_boxes_from_meta, in_front  # noqa: E402

RES, X0, Y0 = 0.5, 0.0, -26.0                     # the front grid: axis 0 = x forward, axis 1 = y (left positive)
CARE, PARTIAL = 0.5, 0.1                          # D38 (b)
NEAR = 2.0                                        # m: the TP threshold of the errors; a detection this close to a neutral box of its class is neutral
STEP = 0.25                                       # m: footprint sampling of a detection
CORNER_STEP = 0.125                               # m: sampling of a GT box (4 x 4 per cell, at least 12 per axis): the same as GtGenerator.coverage, so the shares equal the stored `share_fusion`


def fusion_area(g) -> np.ndarray:
    return (np.asarray(g["camera"], np.float32) > 0) & (np.asarray(g["radar"], np.float32) > 0) & ~(np.asarray(g["occluded"], np.float32) > 0)


def camera_area(g) -> np.ndarray:
    return (np.asarray(g["camera"], np.float32) > 0) & ~(np.asarray(g["occluded"], np.float32) > 0)


SIZE_MAX = 30.0                                   # m: a predicted size beyond this is clamped (a diverging model must not make the sampling explode); NaN counts as outside


def _share(px, py, area) -> float:
    i, j = np.floor((px - X0) / RES).astype(int), np.floor((py - Y0) / RES).astype(int)
    ok = (i >= 0) & (i < area.shape[0]) & (j >= 0) & (j < area.shape[1])
    hit = np.zeros(px.shape, bool)
    hit[ok] = area[i[ok], j[ok]]
    return float(hit.mean())


def corner_share(corners, area: np.ndarray) -> float:
    """share of a rectangle given by its four consecutive corners (ego frame, x forward) that lies in `area`: the rule of GtGenerator.coverage.footprint_share"""
    c = np.asarray(corners, float)[:, :2]
    e1, e2 = c[1] - c[0], c[3] - c[0]
    n1, n2 = max(int(math.ceil(np.hypot(*e1) / CORNER_STEP)), 12), max(int(math.ceil(np.hypot(*e2) / CORNER_STEP)), 12)
    u, v = (np.arange(n1) + 0.5) / n1, (np.arange(n2) + 0.5) / n2
    xy = (c[0] + u[:, None, None] * e1 + v[None, :, None] * e2).reshape(-1, 2)
    return _share(xy[:, 0], xy[:, 1], area)


def footprint_share(b: dict, area: np.ndarray) -> float:
    """share of the footprint of box dict `b` (cx, cy and radius | length, width, yaw) that lies in the boolean grid `area`; a NaN box is outside"""
    if any(not math.isfinite(b.get(k, 0.0)) for k in ("cx", "cy", "radius", "length", "width", "yaw")):
        return 0.0
    if "radius" in b:
        r = min(b["radius"], SIZE_MAX)
        n = max(3, int(math.ceil(2 * r / STEP)))
        u = (np.arange(n) + 0.5) / n * 2 * r - r
        X, Y = np.meshgrid(u, u)
        k = X * X + Y * Y <= r * r
        px, py = X[k] + b["cx"], Y[k] + b["cy"]
        if px.size == 0:
            px, py = np.array([b["cx"]]), np.array([b["cy"]])
    else:
        L, W = min(b["length"], SIZE_MAX), min(b["width"], SIZE_MAX)
        nl, nw = max(3, int(math.ceil(L / STEP))), max(3, int(math.ceil(W / STEP)))
        ul, uw = (np.arange(nl) + 0.5) / nl * L - L / 2, (np.arange(nw) + 0.5) / nw * W - W / 2
        U, V = np.meshgrid(ul, uw)
        c, s = math.cos(b["yaw"]), math.sin(b["yaw"])
        px, py = b["cx"] + U.ravel() * c - V.ravel() * s, b["cy"] + U.ravel() * s + V.ravel() * c
    return _share(px, py, area)


def gt_with_shares(g, groups=None):
    """(box dicts, footprint shares) of EVERY annotated box of a GT dict (observed or not, inside the window or not), optionally of `groups` only; the share from the corners"""
    area = fusion_area(g)
    raw = g["meta"]["boxes"]
    boxes = gt_boxes_from_meta(raw, observed_only=False)
    keep = [i for i, b in enumerate(boxes) if groups is None or b["group"] in groups]
    return [boxes[i] for i in keep], [corner_share(raw[i]["corners"], area) if "corners" in raw[i] else footprint_share(boxes[i], area) for i in keep]


def gt_class(f: float) -> str:
    return "care" if f >= CARE else "partial" if f >= PARTIAL else "dont_care"


def _dist(a: dict, b: dict) -> float:
    return math.hypot(a["cx"] - b["cx"], a["cy"] - b["cy"])


def prepare(dets: Sequence[dict], gts_all: Sequence[dict], shares: Sequence[float], area: np.ndarray, include_partial: bool):
    """(detections kept, GT boxes scored) of one frame under the fusion-area rule. gts_all: every annotated box with its footprint share; the scored ones are the
    care (and, include_partial, the partial) boxes INSIDE the scored window; every other box, also one outside the window, is neutral for the detections"""
    scored, neutral = [], []
    for b, f in zip(gts_all, shares):
        c = gt_class(f)
        (scored if (c == "care" or (include_partial and c == "partial")) and in_front(b) else neutral).append(b)
    keep = []
    for d in dets:
        if any(b["group"] == d["group"] and _dist(d, b) <= NEAR for b in scored):
            keep.append(d)                                                 # a detection of a scored box is scored, wherever its own footprint lies
        elif footprint_share(d, area) >= 0.5 and not any(b["group"] == d["group"] and _dist(d, b) <= NEAR for b in neutral):
            keep.append(d)                                                 # else: at least half in the fusion area and not on a neutral object (neither hit nor false alarm)
    return keep, scored


class BevFusionDetMetric(BevDetMetric):
    """`det` numbers under the fusion-area rule (see the module text); the reduce / direction code is the parent's"""

    def __init__(self, groups, name="detf", include_partial=False, source="dets"):
        super().__init__(groups, name, source)
        self.include_partial = include_partial

    def frame(self, sample: Sample, pred: Prediction):
        out = {}
        for dt, o in pred.outputs["dt"].items():
            dets = o.get(self.source)
            if dets is None:
                continue
            g = sample.reference[dt]
            gts_all, shares = gt_with_shares(g, self.groups)
            d, gts = prepare(self.meter.keep(dets), gts_all, shares, fusion_area(g), self.include_partial)
            fr = self.meter.frame_rows(d, gts, self.groups)
            out[f"{dt_key(dt)}|rows"] = fr["rows"]
            out[f"{dt_key(dt)}|n_gt"] = fr["n_gt"].astype(np.int32)
            out[f"{dt_key(dt)}|vel"] = self.meter.frame_velocity(d, gts, self.groups)
        return out


COV_CLASSES = ("all", "care", "partial", "dont_care", "camera", "cam_radar", "cam_only", "radar_only", "neither")
RADAR_CLASSES = ("cam_radar", "cam_only", "radar_only", "neither")


def coverage_classes(f_fusion: float, f_cam: float, radar_pts: Optional[int]):
    """the classes of one GT box (see the module text); radar_pts None = no annotation table"""
    cl = ["all", gt_class(f_fusion)]
    cam = f_cam >= CARE
    if cam:
        cl.append("camera")
    if radar_pts is not None:
        cl.append(("cam_radar" if radar_pts > 0 else "cam_only") if cam else ("radar_only" if radar_pts > 0 else "neither"))
    return cl


class BevCoverageMetric(Metric):
    """see the module text; a Metric of the evalsuite (frame / reduce / score / support), counts are additive per frame"""
    name = "cov"
    headline = ()
    better: Dict[str, str] = {}

    def __init__(self, groups: Sequence[str], pts=None):
        self.groups, self.pts = list(groups), pts

    def direction(self, key):
        last = key.rsplit("/", 1)[-1]
        return "low" if last == "ate" else "high"

    def frame(self, sample: Sample, pred: Prediction):
        g = sample.reference.get(0.0)
        o = pred.outputs["dt"].get(0.0)
        if g is None or o is None or o.get("dets") is None:
            return {}
        raw = g["meta"]["boxes"]
        gts = gt_boxes_from_meta(raw, observed_only=False)
        area, cam = fusion_area(g), camera_area(g)
        dets = [d for d in o["dets"] if in_front(d)]
        rows = np.zeros((len(self.groups), len(COV_CLASSES), 3))               # n_gt, n_found, sum of the centre errors of the found ones
        used, found = set(), {}
        pairs = sorted((_dist(d, b), di, bi) for di, d in enumerate(dets) for bi, b in enumerate(gts) if d["group"] == b["group"] and _dist(d, b) <= NEAR)
        for dist, di, bi in pairs:                                             # one-to-one, nearest first, every score: the recall ceiling
            if di not in used and bi not in found:
                used.add(di)
                found[bi] = dist
        for bi, (b, rb) in enumerate(zip(gts, raw)):
            if b["group"] not in self.groups or not in_front(b):
                continue
            pts = self.pts(rb["token"]) if self.pts else None
            for cl in coverage_classes(corner_share(rb["corners"], area), corner_share(rb["corners"], cam), None if pts is None else pts[1]):
                k = COV_CLASSES.index(cl)
                rows[self.groups.index(b["group"]), k, 0] += 1
                if bi in found:
                    rows[self.groups.index(b["group"]), k, 1] += 1
                    rows[self.groups.index(b["group"]), k, 2] += found[bi]
        return {"cov": rows}

    def reduce(self, counts, keys: Optional[Sequence[str]] = None):
        tot = sum((c["cov"] for c in counts if "cov" in c), np.zeros((len(self.groups), len(COV_CLASSES), 3)))
        out: Dict[str, float] = {}
        for gi, grp in enumerate(self.groups):
            for k, cl in enumerate(COV_CLASSES):
                if cl in RADAR_CLASSES and not self.pts:
                    continue
                n, hit, err = tot[gi, k]
                out[f"dt+0.00/{grp}/{cl}/n_gt"] = float(n)
                out[f"dt+0.00/{grp}/{cl}/recall2m"] = float(hit / n) if n else float("nan")
                out[f"dt+0.00/{grp}/{cl}/ate"] = float(err / hit) if hit else float("nan")
        return {k: out.get(k, float("nan")) for k in keys} if keys is not None else out

    def score(self, counts):
        return {}

    def support(self, counts):
        return {}


def selftest():
    cam, rad, occ = (np.zeros((4, 4), np.float32) for _ in range(3))
    cam[:2], rad[1:3], occ[1, 1] = 1, 1, 1                                    # camera rows 0-1, radar rows 1-2, hidden cell (1, 1)
    fa, ca = fusion_area({"camera": cam, "radar": rad, "occluded": occ}), camera_area({"camera": cam, "radar": rad, "occluded": occ})
    assert fa.sum() == 3 and fa[1, 0] and fa[1, 2] and fa[1, 3] and not fa[1, 1] and not fa[0].any() and not fa[2].any(), "fusion area = camera x radar x not hidden"
    assert ca.sum() == 7 and ca[0].all() and not ca[1, 1], "camera area = camera x not hidden"
    area = np.zeros((120, 104), bool)
    area[:20, :] = True                                                       # observed: x < 10 m
    box = lambda cx, cy, l=4.0, w=2.0, yaw=0.0, grp="vehicle": dict(group=grp, cx=cx, cy=cy, length=l, width=w, yaw=yaw)   # noqa: E731
    assert footprint_share(box(5, 0), area) == 1.0 and footprint_share(box(15, 0), area) == 0.0
    assert abs(footprint_share(box(10, 0), area) - 0.5) < 0.04, footprint_share(box(10, 0), area)
    assert footprint_share(box(5, 0, yaw=0.7), area) == 1.0, "a turned box inside"
    assert footprint_share(box(5, 40), area) == 0.0, "outside the grid counts as outside"
    ped = dict(group="pedestrian", cx=10.0, cy=0.0, radius=0.4)
    assert 0.35 < footprint_share(ped, area) < 0.65
    assert abs(corner_share([(9, -1), (9, 1), (11, 1), (11, -1)], area) - 0.5) < 0.03 and corner_share([(1, -1), (1, 1), (3, 1), (3, -1)], area) == 1.0, "corner share"
    assert gt_class(0.5) == "care" and gt_class(0.49) == "partial" and gt_class(0.1) == "partial" and gt_class(0.09) == "dont_care"
    # three GT boxes (care, partial, don't care), a detection on each, plus a false detection inside and one outside the area
    def corners(b):                                                           # the four consecutive corners of a box dict, as the GT stores them
        c, s_ = math.cos(b["yaw"]), math.sin(b["yaw"])
        return [(b["cx"] + sx * b["length"] / 2 * c - sy * b["width"] / 2 * s_, b["cy"] + sx * b["length"] / 2 * s_ + sy * b["width"] / 2 * c) for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
    care, partial, dc = box(5, 0), box(11.4, 5), box(30, -5)
    gts = [care, partial, dc]
    fs = [footprint_share(b, area) for b in gts]
    assert [gt_class(f) for f in fs] == ["care", "partial", "dont_care"], fs
    dets = [dict(b, score=1.0) for b in gts] + [dict(box(6, 20), score=1.0), dict(box(40, 0), score=1.0)]
    shares = [corner_share(corners(b), area) for b in gts]
    assert [gt_class(f) for f in shares] == ["care", "partial", "dont_care"], shares
    keep, scored = prepare(dets, gts, shares, area, include_partial=False)
    assert len(scored) == 1 and len(keep) == 2, (len(scored), len(keep))      # the care box + its detection and the false one inside; the neutral ones and the outside one go
    keep2, scored2 = prepare(dets, gts, shares, area, include_partial=True)
    assert len(scored2) == 2 and len(keep2) == 3, (len(scored2), len(keep2))
    # a detection on the care box that also sits within 2 m of a neutral box of its class stays (it is near a scored one)
    keep3, _ = prepare([dict(box(6.5, 0), score=1.0)], [care, box(8.0, 0.0, l=1.0)], [1.0, 1.0], area, include_partial=False)
    assert len(keep3) == 1
    area2 = np.ones((140, 104), bool)
    area2[118:] = False
    # a neutral box just outside the window (x >= 60) still neutralises a detection next to it
    k4, s4 = prepare([dict(box(58.5, 0), score=1.0)], [dict(box(60.5, 0))], [0.0], area2, include_partial=False)
    assert k4 == [] and s4 == [], "a detection within 2 m of a neutral box outside the window is ignored"
    assert footprint_share(dict(box(5, 0), length=float("nan")), area) == 0.0, "a NaN box is outside"
    assert 0.0 <= footprint_share(dict(box(5, 0), length=1e6, width=1e6), area) <= 1.0, "a huge predicted box is clamped, not sampled"
    assert coverage_classes(1.0, 1.0, 3) == ["all", "care", "camera", "cam_radar"]
    assert coverage_classes(1.0, 1.0, 0) == ["all", "care", "camera", "cam_only"]
    assert coverage_classes(0.0, 0.0, 2) == ["all", "dont_care", "radar_only"]
    assert coverage_classes(0.3, 0.3, 0) == ["all", "partial", "neither"]
    assert coverage_classes(1.0, 1.0, None) == ["all", "care", "camera"]
    print("bev_fusion_area selftest ok: footprint shares, GT classes, neutral detections, coverage classes")


def coverage_table(n: int = 300, sig: Optional[str] = None, seed: int = 0, pts=None):
    """how the COMPLETE GT set (every annotated box in the scored window) of `n` cached frames splits by what the rig can observe: counts per class and share of the class's total"""
    from evaluation import bev_gtcheck as G
    from evalsuite.contracts import FrameRef
    task = G.build_task()
    metric = BevCoverageMetric(task.groups, pts if pts is not None else task._pts())
    counts, sig = [], G.resolve_sig(sig)
    for i, (name, s) in enumerate(G.pick(n, sig, seed)):
        counts.append(metric.frame(Sample(ref=FrameRef("gt", i, name), inputs=None, reference=s), Prediction(outputs={"dt": {0.0: {"dets": []}}})))
    res = metric.reduce(counts)
    shown = [c for c in COV_CLASSES if metric.pts or c not in RADAR_CLASSES]
    print(f"{len(counts)} frames of signature {sig}; annotated boxes in the scored window (x < 60 m, +-32.3 deg), observed or not")
    print(f"{'class':14s}" + "".join(f"{c:>12s}" for c in shown))
    for grp in task.groups:
        tot = res[f"dt+0.00/{grp}/all/n_gt"]
        print(f"{grp:14s}" + "".join(f"{int(res[f'dt+0.00/{grp}/{c}/n_gt']):7d}{100 * res[f'dt+0.00/{grp}/{c}/n_gt'] / tot if tot else 0:4.0f} %" for c in shown))
    return res


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    elif "--coverage" in sys.argv:
        coverage_table(int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 300)
    else:
        print(__doc__)
