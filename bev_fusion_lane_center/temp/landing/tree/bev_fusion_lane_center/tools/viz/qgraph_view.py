"""Pictures and videos of the lane head of E273 (pixel maps, grid-cell nodes, Hermite links, soft sections, map assignment): ONE figure per keyframe, all bird's-eye panels on the
same 0.5 m canvas (1 m maps scaled x2, 2 m maps x4 by nearest neighbour, so the cells line up). The input is the prediction dict of `training/grid/qformat.py` (the contract) and the
lane GT of the frame (`GtGenerator.lane_gt`); today the prediction is the ORACLE (`qformat.oracle_prediction`) or a DEGRADED oracle made here to check the error tints, later the network's
dict through `pred_fn(token)`.

    python qgraph_view.py frame <token | scene-0221:12> --out DIR [--pred oracle|degraded] [--no-cams]
    python qgraph_view.py scene <VAL scene id | scene name> --out DIR [--video] [--pred ...] [--frames N] [--every K]
    python qgraph_view.py list                    the VAL scenes (id as in `grid_view.py --scene`, name, location, frames, description)
    python qgraph_view.py --selftest              a synthetic frame (needs no stored GT; imports the scorer)

Panels (1920 x 1080 with the six cameras; metres, x right, y forward, ego at the origin):
  cameras   the GT lane graph projected into the six cameras (front large; the projection is `GtGenerator.lane_graph_view.camera_image`; a camera that is not on disk is a grey tile)
  1 m       centre-lane cells: GT faint grey, correct dark, false positive light red, miss light blue (a cell counts when the other map has a cell within 1 cell); without
            `pred["maps"]` the cells of the delivered nodes are drawn instead
  2 m, 0.5 m  road and intersection cells (exact), dividers and boundary (one hue per class), the same scheme
  nodes     a marker per flag (stop, exit, split, merge, start, end, window) and the Hermite links over the GT graph in thin grey; a node is false when no GT node lies within 2 m
            (red halo), a GT node is missed when no delivered node lies within 2 m (blue halo); false links red, missed GT links blue; GT junctions circled faintly
  ids       left the GT pieces (one stable colour per piece id), right the delivered nodes and curves in the expected colour sum_j P(j | node) colour_j (dustbin grey; an unsure
            assignment blends), a black diamond where the section changes along a link
Title and numbers panel: the scorer's numbers (`evaluation/qgraph_eval.py`; the graph numbers only for VAL frames, against the old graph GT of the lane suite). The counts of false / missed nodes and links behind the tints are the picture's own rule (nearest node within 2 m).
Read-out: the scorer's rule (`qgraph_eval.kept`: e >= 0.5, p >= 0.5 between kept nodes). Map layers of `pred["maps"]` as `qraster` makes them: lines05 = mask (3) + distance (3),
centre1 channel 0, road2 = road, intersection; values inside [0, 1] are probabilities (on at > 0.5), others logits (on at > 0).

Minimal example (python):
    import qgraph_view as QV
    g = QV.load_gt(token); pred = QV.make_pred(g, "degraded", token)              # or the network's dict
    QV.render_frame(token, pred, "frame.png", cams=True)                          # returns the RGB array too
    QV.render_scene("scene-0221", "out_dir", pred_fn=lambda t: QV.make_pred(QV.load_gt(t), "oracle", t), video=True)
Cameras come from the CRN infos of the VAL frames (calibration, ego pose per camera); other tokens get grey tiles. CPU only, about 3 s per frame.
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")                                    # a shared machine
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import argparse  # noqa: E402
import colorsys  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
import zlib  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402
from scipy.ndimage import maximum_filter  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _sub in ("..", "training/grid", "tools/video", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, _sub))
import qformat as QF  # noqa: E402
import qraster as QR  # noqa: E402
import qtargets as QT  # noqa: E402

HOME = os.path.expanduser("~")
NUSC = os.path.join(HOME, "data", "nuscenes_full")
EXP = os.path.join(HOME, "exp_gcar_b123")
CAMS = ("CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT")
RED, BLUE, GREY = np.array([1.0, 0.2, 0.2]), np.array([0.2, 0.4, 1.0]), np.array([0.6, 0.6, 0.6])
ALPHA = 0.4                                                           # the error tints
HUE = {"lane_divider": (0.10, 0.62, 0.25), "road_divider": (0.85, 0.60, 0.0), "boundary": (0.50, 0.20, 0.70)}
MARK = {"stop": ("s", "#d62728", 72), "exit": ("s", "#e8b400", 72), "split": ("^", "#ff7f0e", 56), "merge": ("v", "#1f77b4", 56), "start": ("o", "#2ca02c", 40),
        "end": ("o", "#7f7f7f", 40), "win_in": ("P", "#111111", 90), "win_out": ("X", "#111111", 90)}
_CACHE = {}


# ------------------------------------------------------------------------------------------------------------------------------------------ data
def load_gt(token):
    """the stored lane GT graph of a keyframe (`GtGenerator.lane_gt.load`)"""
    from GtGenerator import lane_gt
    return lane_gt.load(token)


def _infos():
    """CRN infos of the VAL frames by sample token (calibration and ego pose of every camera, scene name), loaded once (5 s)"""
    if "infos" not in _CACHE:
        with open(os.path.join(NUSC, "nuscenes_infos_val_crn.pkl"), "rb") as f:
            _CACHE["infos"] = {i["sample_token"]: i for i in pickle.load(f)}
    return _CACHE["infos"]


def val_scenes():
    """-> {scene id: (scene name, [sample tokens in time order])} of the VAL frames; the id is that of `poses_val.npz`, as in `grid_view.py --scene`"""
    if "scenes" not in _CACHE:
        pz = np.load(os.path.join(EXP, "poses_val.npz"))
        with open(os.path.join(EXP, "feats", "val_tokens.json")) as f:
            toks = json.load(f)
        with open(os.path.join(ROOT, "data", "splits.json")) as f:
            rows = json.load(f)["val"]
        out = {}
        for r in sorted(rows, key=lambda r: pz["ts"][r]):
            out.setdefault(int(pz["scene"][r]), []).append(toks[r])
        _CACHE["scenes"] = {k: (_infos()[v[0]]["scene_name"], v) for k, v in sorted(out.items())}
    return _CACHE["scenes"]


def resolve_scene(s):
    """a VAL scene id ("11") or name ("scene-0221") -> (id, name, tokens)"""
    for k, (name, toks) in val_scenes().items():
        if str(k) == str(s) or name == s:
            return k, name, toks
    raise KeyError(f"{s!r} is not a VAL scene; `qgraph_view.py list` shows them")


def frame_info(token):
    """-> (scene id, scene name, index of the frame, frames of the scene) or None when the token is not a VAL frame"""
    if token not in _infos():
        return None
    for k, (name, toks) in val_scenes().items():
        if token in toks:
            return k, name, toks.index(token), len(toks)
    return None


class _Nusc:
    """the surface of the devkit that `lane_graph_view.camera_image` reads, served from the camera infos of one keyframe (no 30 s devkit load)"""

    def __init__(self, cam_infos):
        self.c = cam_infos

    def get(self, table, key):
        c = self.c[key]
        return {"sample_data": dict(calibrated_sensor_token=key, ego_pose_token=key, filename=c["filename"]), "calibrated_sensor": c["calibrated_sensor"], "ego_pose": c["ego_pose"]}[table]


def camera_tiles(token, g, sizes, log=print):
    """-> {camera: RGB tile} with the GT graph projected (the GT chat's code); grey tiles with a note when the token has no camera info or an image is missing"""
    import cv2
    info = _infos().get(token)
    tiles = {}
    if info is not None:
        from GtGenerator import lane_graph_view as V
        from GtGenerator.nuscenes_adapter import pose_matrix

        class _Gt:
            nusc, dataroot = _Nusc(info["cam_infos"]), NUSC
        sample = dict(data={c: c for c in CAMS})
        T_front = pose_matrix(info["cam_infos"]["CAM_FRONT"]["ego_pose"])
    for ch in CAMS:
        w, h = sizes[ch]
        if info is not None:
            if not os.path.exists(os.path.join(NUSC, info["cam_infos"][ch]["filename"])):
                log(f"camera {ch} of {token}: image not on disk (grey tile)")
            img = V.camera_image(_Gt, sample, g, ch, T_front, scale=1.0)[..., ::-1]
        else:
            log(f"camera {ch} of {token}: no camera info for this token (grey tile)")
            img = np.full((900, 1600, 3), 60, np.uint8)
        tile = cv2.resize(np.ascontiguousarray(img), (w, h), interpolation=cv2.INTER_AREA)
        cv2.rectangle(tile, (0, 0), (int(0.42 * w), 15), (0, 0, 0), -1)               # covers the caption camera_image drew at full size
        cv2.putText(tile, ch, (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        tiles[ch] = tile
    return tiles


# ------------------------------------------------------------------------------------------------------------------------------------ rasters
def centre_mask(g):
    """(100, 100) bool: the 1 m cells that hold a point of a lane piece (the rows of `qraster.centre_points`)"""
    P, _ = QR.centre_points(g)
    m = np.zeros((100, 100), bool)
    if len(P):
        c = np.clip(np.floor(P + 50.0).astype(int), 0, 99)
        m[c[:, 0], c[:, 1]] = True
    return m


def gt_layers(g):
    """the GT masks: centre (100, 100), lines (3, 200, 200), road and inter (50, 50)"""
    lm, _ = QR.lines05(g)
    r2 = QR.areas2(g)
    return dict(centre=centre_mask(g), lines=lm > 0, road=r2[0] > 0, inter=r2[1] > 0)


def oracle_maps(g):
    """`pred["maps"]` of a perfect network, the layout of the contract (events as `qraster`; the distance channels of the lines are the GT's)"""
    lm, ld = QR.lines05(g)
    c1 = np.zeros((6, 100, 100), np.float32)
    c1[0] = centre_mask(g)
    return dict(lines05=np.concatenate([lm, ld]), centre1=c1, events1=QR.events1(g), road2=QR.areas2(g))


def _on(x):
    x = np.asarray(x, np.float32)
    return x > 0.5 if (x.size == 0 or (x.min() >= 0 and x.max() <= 1)) else x > 0


def pred_layers(maps):
    return dict(centre=_on(maps["centre1"][0]), lines=_on(maps["lines05"][:3]), road=_on(maps["road2"][0]), inter=_on(maps["road2"][1]))


def layer_triple(p, t, tol=0):
    """-> correct, false positive, miss of a predicted mask against a GT mask; a cell counts when the other mask has a cell within `tol` cells"""
    near = (lambda m: maximum_filter(m, size=2 * tol + 1)) if tol else (lambda m: m)
    return p & near(t), p & ~near(t), t & ~near(p)


def paint(img, m, rgb, a=1.0):
    img[m] = (1 - a) * img[m] + a * np.asarray(rgb, float)


def up(a, k):
    return np.repeat(np.repeat(a, k, 0), k, 1)


def layer_img(img, p, t, tol, under, ok):
    """paint one layer into an RGB canvas: GT under, correct, false positive tint, miss tint; -> (P, R) of the layer"""
    c, fp, ms = layer_triple(p, t, tol)
    paint(img, t, under)
    paint(img, c, ok)
    paint(img, fp, RED, ALPHA)
    paint(img, ms, BLUE, ALPHA)
    return (c.sum() / p.sum() if p.any() else None), (1 - ms.sum() / t.sum() if t.any() else None)


def centre_panel(gt, pr, d, cmp, nt):
    """(200, 200, 3) canvas of the 1 m centre cells; without maps the cells of the delivered nodes (and the missed GT nodes)"""
    img = np.ones((100, 100, 3))
    if pr is not None:
        pe, re = layer_img(img, pr["centre"], gt["centre"], 1, 0.84, 0.2)
        return up(img, 2), (pe, re)
    p = np.zeros((100, 100), bool)
    cell = lambda xy: np.clip(np.floor(np.asarray(xy).reshape(-1, 2) + 50.0).astype(int), 0, 99)
    c = cell(d["nodes"]["pos"])
    p[c[:, 0], c[:, 1]] = True
    ok = p & maximum_filter(gt["centre"], size=3)
    miss = np.zeros((100, 100), bool)
    m = cell(nt["pos"][cmp["missed_n"]])
    miss[m[:, 0], m[:, 1]] = True
    paint(img, gt["centre"], 0.84)
    paint(img, ok, 0.2)
    paint(img, p & ~ok, RED, ALPHA + 0.3)
    paint(img, miss, BLUE, ALPHA + 0.3)
    return up(img, 2), (ok.sum() / max(p.sum(), 1), 1 - cmp["missed_n"].mean() if len(cmp["missed_n"]) else 1.0)


def area_panel(gt, pr):
    """(200, 200, 3) canvas: road and intersection cells (2 m, exact) and the three line classes (0.5 m, one cell of tolerance); -> canvas, {layer: (P, R)}"""
    img = np.ones((50, 50, 3))
    sc = {}
    if pr is None:
        paint(img, gt["road"], 0.93)
        paint(img, gt["inter"], 0.85)
    else:
        sc["road"] = layer_img(img, pr["road"], gt["road"], 0, 0.93, 0.78)
        sc["inter"] = layer_img(img, pr["inter"], gt["inter"], 0, 0.85, 0.52)
    img = up(img, 4)
    for k, name in enumerate(QR.LINES):
        hue = np.array(HUE[name])
        if pr is None:
            paint(img, gt["lines"][k], 0.5 * hue + 0.5)
        else:
            sc[name] = layer_img(img, pr["lines"][k], gt["lines"][k], 1, 0.4 * hue + 0.6, hue)
    return img, sc


def underlay(gt):
    """(200, 200, 3): road and intersection in faint grey, under the node and id panels"""
    img = np.ones((50, 50, 3))
    paint(img, gt["road"], 0.95)
    paint(img, gt["inter"], 0.90)
    return up(img, 4)


# ------------------------------------------------------------------------------------------------------------------------------ the prediction
def select(pred, keep, lkeep=None):
    """a pred with the nodes `keep` (bool (M,)) and the links between them that also pass `lkeep` (bool (L,)); the table of the oracle is dropped"""
    n, l = pred["nodes"], pred["links"]
    new = -np.ones(len(keep), int)
    new[keep] = np.arange(int(keep.sum()))
    lk = keep[l["src"]] & keep[l["dst"]] & (np.ones(len(l["src"]), bool) if lkeep is None else lkeep)
    links = {k: v[lk] for k, v in l.items() if k not in ("src", "dst", "curves")}
    links.update(src=new[l["src"][lk]], dst=new[l["dst"][lk]], curves=None if l["curves"] is None else [c for c, f in zip(l["curves"], lk) if f])
    a = pred["assign"]
    assign = None if a is None else dict(a, P=a["P"][keep], road=a["road"][keep])
    return dict(pred, nodes={k: v[keep] for k, v in n.items()}, links=links, assign=assign, table=None)


def readout(pred):
    """the delivered graph as a pred: the nodes and links the scorer keeps (`qgraph_eval.kept`, the one place of the read-out rule)"""
    import qgraph_eval as QE
    keep, L = QE.kept(pred)
    lk = np.zeros(len(pred["links"]["src"]), bool)
    lk[L] = True
    return select(pred, keep, lk)


def curves_of(pred):
    """the curve of every link: the delivered ones, else the Hermite curve of the two node poses and the handle scales"""
    n, l = pred["nodes"], pred["links"]
    if l["curves"] is not None:
        return l["curves"]
    return [QT.hermite(n["pos"][a], n["dirv"][a], n["pos"][b], n["dirv"][b], x, y) for a, b, x, y in zip(l["src"], l["dst"], l["a"], l["b"])]


def degrade(pred, seed=0):
    """the oracle with known faults, so that the tints can be checked: 5 % spurious nodes and 3 % spurious links (red), 15 % of the nodes dropped and 10 % of the links cut (blue),
    10 % of the nodes shifted by 1.5 m, the section ids (and the map piece) of the two biggest pieces swapped, the assignment of 20 % of the nodes blurred; curves recomputed from the nodes.
    A frame with fewer than two pieces (a map hole, a lone piece) has nothing to swap and is returned as it is"""
    rng = np.random.RandomState(seed)
    n = {k: v.copy() for k, v in pred["nodes"].items()}
    l = {k: v.copy() for k, v in pred["links"].items() if k != "curves"}
    a = dict(pred["assign"])
    M, K = len(n["pos"]), a["P"].shape[1] - 1
    if M < 3 or K < 2:
        return pred
    gpos = pred["nodes"]["pos"]
    # spurious nodes at 4 - 6 m from the lanes, spurious links between near, not linked nodes
    new = []
    while len(new) < max(int(0.05 * M), 3):
        q = gpos[rng.randint(M)] + rng.uniform(4, 6) * np.array([np.cos(t := rng.uniform(0, 2 * np.pi)), np.sin(t)])
        if np.abs(q).max() < 49 and np.linalg.norm(gpos - q, axis=1).min() > 3:
            new.append(q)
    new = np.array(new)
    ang = rng.uniform(0, 2 * np.pi, len(new))
    n["pos"] = np.concatenate([n["pos"], new])
    n["dirv"] = np.concatenate([n["dirv"], np.stack([np.cos(ang), np.sin(ang)], 1)])
    n["e"] = np.concatenate([n["e"], np.full(len(new), 0.9)])
    n["flags"] = np.concatenate([n["flags"], np.zeros((len(new), 8))])
    n["s"] = np.concatenate([n["s"], np.zeros(len(new))])
    n["cell"] = QT.cell_of(n["pos"])
    n["section"] = np.concatenate([n["section"], np.full(len(new), n["section"].max() + 1)])
    dust = np.zeros((len(new), K + 1))
    dust[:, K] = 1
    a["P"] = np.concatenate([a["P"], dust])
    a["road"] = np.concatenate([a["road"], np.tile(np.eye(a["road"].shape[1])[-1], (len(new), 1))])
    kd = cKDTree(gpos)
    have = set(zip(l["src"].tolist(), l["dst"].tolist())) | set(zip(l["dst"].tolist(), l["src"].tolist()))
    extra = []
    for i in rng.permutation(M):
        if len(extra) >= max(int(0.03 * len(l["src"])), 2):
            break
        for j in kd.query(gpos[i], 6)[1][1:]:
            if (i, j) not in have and np.linalg.norm(gpos[i] - gpos[j]) > 4:
                extra.append((i, j))
                have.add((i, j))
                break
    for k in ("src", "dst"):
        l[k] = np.concatenate([l[k], [e[0 if k == "src" else 1] for e in extra]]).astype(int)
    for k, v in (("p", 0.9), ("long", False), ("a", 1 / 3), ("b", 1 / 3)):
        l[k] = np.concatenate([l[k], np.full(len(extra), v, l[k].dtype)])
    M2 = len(n["pos"])
    # shift, drop, cut
    sh = rng.rand(M2) < 0.10
    t = rng.uniform(0, 2 * np.pi, int(sh.sum()))
    n["pos"][sh] += 1.5 * np.stack([np.cos(t), np.sin(t)], 1)
    for i in rng.randint(0, M, max(int(0.15 * M / 3), 1)):                  # 15 % of the nodes dropped, in runs of three along a lane (a lone node leaves no gap at the 2 m rule)
        n["e"][i:i + 3] = 0.2
    l["p"][rng.rand(len(l["src"])) < 0.10] = 0.2
    # swap the two biggest pieces; blur the assignment of 20 % of the nodes
    cnt = np.bincount(n["section"][:M], minlength=K)
    ia, ib = np.argsort(-cnt)[:2]
    for x, y in ((ia, ib), (ib, ia)):
        sel = np.nonzero(n["section"][:M] == x)[0]
        a["P"][sel[:, None], [[x, y]]] = a["P"][sel[:, None], [[y, x]]]
    sa, sb = n["section"][:M] == ia, n["section"][:M] == ib
    n["section"][:M][sa], n["section"][:M][sb] = ib, ia
    for m in np.nonzero(rng.rand(M2) < 0.20)[0]:
        other = np.zeros(K + 1)
        other[rng.randint(K + 1)] = 1
        a["P"][m] = 0.5 * a["P"][m] + 0.5 * other
    l["curves"] = None
    out = dict(pred, nodes=n, links=l, assign=a, table=None)
    if pred["maps"] is not None:
        out["maps"] = degrade_maps(pred["maps"], rng)
    return out


def degrade_maps(maps, rng):
    """the mask layers of the maps with known faults: the forward half shifted, a block erased, a blob added"""
    m = {k: v.copy() for k, v in maps.items()}

    def mess(x, shift):
        n = x.shape[0]
        fwd = np.zeros_like(x, bool)
        fwd[:, n // 2 + n // 20:] = x[:, n // 2 + n // 20:] > 0.5
        x[fwd] = 0
        x += np.roll(fwd, shift, axis=0).astype(x.dtype)
        b = n // 10
        i, j = rng.randint(n // 4, 3 * n // 4 - b), rng.randint(n // 20, n // 2 - b)
        x[i:i + b, j:j + b] = 0
        i, j = rng.randint(n // 8, 7 * n // 8 - b), rng.randint(n // 8, 7 * n // 8 - b)
        x[i:i + b, j] = 1
    mess(m["centre1"][0], 2)
    for k in range(3):
        mess(m["lines05"][k], 3)
    mess(m["road2"][0], 1)
    mess(m["road2"][1], 1)
    return m


def make_pred(g, kind="oracle", token=""):
    """the oracle prediction of the GT graph `g` (with dense maps) or the degraded one (seeded by the token)"""
    pred = QF.oracle_prediction(g)
    pred["maps"] = oracle_maps(g)
    return pred if kind == "oracle" else degrade(pred, zlib.crc32(token.encode()))


# ------------------------------------------------------------------------------------------------------------------------------------ analysis
def compare_graph(d, nt, tol=2.0):
    """the delivered graph `d` against the GT node table `nt` (card section 4): a node is false when no GT node lies within `tol`, a GT node is missed when no delivered node lies within
    `tol`; a link is true when its two nodes match the two nodes of a GT link, a GT link is missed when no delivered link matches it. -> dict of bool arrays + the matching"""
    pos, gpos = d["nodes"]["pos"], nt["pos"]
    dn, ig = cKDTree(gpos).query(pos) if len(pos) and len(gpos) else (np.full(len(pos), np.inf), np.zeros(len(pos), int))
    dm = cKDTree(pos).query(gpos)[0] if len(pos) and len(gpos) else np.full(len(gpos), np.inf)
    ok = dn <= tol
    gl = list(zip(nt["src"].tolist(), nt["dst"].tolist()))
    src, dst = d["links"]["src"], d["links"]["dst"]
    gset = set(gl)
    hit = np.array([bool(ok[a] and ok[b]) and (ig[a], ig[b]) in gset for a, b in zip(src, dst)], bool)
    found = {(ig[a], ig[b]) for a, b, h in zip(src, dst, hit) if h}
    return dict(false_n=~ok, missed_n=dm > tol, false_l=~hit, missed_l=np.array([x not in found for x in gl], bool), match=ig, gt_links=gl)


def scorer_numbers(pred, g, token):
    """the numbers of the scorer (`qgraph_eval`: node, assignment and, for a VAL frame, graph metrics against the old graph GT of the lane suite); the graph keys are missing for other tokens"""
    import qgraph_eval as QE
    out = dict(QE.node_metrics(pred, g), **QE.assign_metrics(pred, g))
    if "frames" not in _CACHE:
        _CACHE["frames"] = {f.token: f for f in QE.val_frames()}
    if token in _CACHE["frames"]:
        out.update(QE.graph_metrics(QE.readout(pred)[0], _CACHE["frames"][token]))
    return out


def piece_rgb(key):
    """the colour of a piece id: the hue of `GtGenerator.lane_graph_view` (so the camera panels and the ids panels agree), a little darker for a light background"""
    return np.array(colorsys.hsv_to_rgb(((zlib.crc32(str(key).encode()) * 2654435761) % 180) / 180.0, 0.78, 0.80))


def expected_rgb(assign):
    """(M, 3) the colour sum_j P(j | node) colour_j, the dustbin grey"""
    P = assign["P"]
    cols = np.array([piece_rgb(k) for k in assign["map_ids"]]).reshape(-1, 3)
    return P[:, :-1] @ cols + P[:, -1:] * GREY


# ------------------------------------------------------------------------------------------------------------------------------------- drawing
def layout(cams):
    """pixel rectangles (x, y, w, h; origin top left) of the panels; the BEV panels are S x S with S a multiple of 200, so that a 0.5 m canvas pixel is S / 200 whole screen pixels"""
    S = 400 if cams else 600
    x0, y0, row_h, col_w = (640 if cams else 0), 66, S + 38, S + 24
    rects = {name: (x0 + (i % 3) * col_w + 24, y0 + (i // 3) * row_h, S, S) for i, name in enumerate(("centre", "areas", "nodes", "ids_gt", "ids_pred", "numbers"))}
    W, H = (1920, 1080) if cams else (x0 + 3 * col_w + 12, y0 + 2 * row_h + 150)
    ly = y0 + 2 * row_h - 14
    lay = dict(size=(W, H), bev=rects, legend=(x0 + 24, ly, 3 * col_w - 24, H - ly - 6))
    if cams:
        lay["cams"] = {"CAM_FRONT": (0, 44, 640, 360), "CAM_FRONT_LEFT": (0, 408, 320, 180), "CAM_FRONT_RIGHT": (320, 408, 320, 180), "CAM_BACK_LEFT": (0, 592, 320, 180),
                       "CAM_BACK_RIGHT": (320, 592, 320, 180), "CAM_BACK": (0, 776, 320, 180)}
    return lay


def _axes(fig, rect, size, title=None):
    x, y, w, h = rect
    W, H = size
    ax = fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])
    if title:
        ax.set_title(title, fontsize=7.5, loc="left", pad=7)
    return ax


def _bev(ax):
    ax.set_xlim(-50, 50), ax.set_ylim(-50, 50), ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([-50, -25, 0, 25, 50]), ax.set_yticks([-50, -25, 0, 25, 50])
    ax.tick_params(labelsize=5.5, length=2, pad=1)
    ax.scatter([0], [0], marker="^", c="k", s=22, zorder=1.5)                          # the ego car, under the nodes
    return ax


def _show(ax, img):
    ax.imshow(np.transpose(img, (1, 0, 2)), origin="lower", extent=(-50, 50, -50, 50), interpolation="nearest", zorder=0)
    return _bev(ax)


def _lines(ax, polys, **kw):
    ax.add_collection(LineCollection([np.asarray(p)[:, :2] for p in polys if len(p) > 1], **kw))


def draw_nodes(ax, g, nt, d, cmp, gt):
    _show(ax, underlay(gt))
    for e in g["events"]:
        if e["type"] in ("split", "merge", "cross"):
            ax.add_patch(Circle(e["pos"], 3.0, fill=False, ec="0.55", ls=":", lw=0.8, zorder=1))
    _lines(ax, [p["pts"] for p in g["pieces"]], colors="0.62", linewidths=0.9, zorder=2)
    ax.scatter(*nt["pos"].T, s=5, c="0.55", zorder=2)
    miss = [nt["pts"][nt["path"][i]] for i in np.nonzero(cmp["missed_l"])[0]]
    _lines(ax, miss, colors=BLUE, linewidths=4.5, alpha=0.45, zorder=3)
    ax.scatter(*nt["pos"][cmp["missed_n"]].T, s=170, c=[BLUE], alpha=0.45, linewidths=0, zorder=3)
    curves = curves_of(d)
    long = d["links"]["long"]
    fl = cmp["false_l"]
    _lines(ax, [c for c, f in zip(curves, fl) if f], colors=RED, linewidths=4.5, alpha=0.45, zorder=4)
    _lines(ax, [c for c, f in zip(curves, long) if not f], colors="#26324d", linewidths=1.2, zorder=5)
    _lines(ax, [c for c, f in zip(curves, long) if f], colors="#7a1fa2", linewidths=1.4, linestyles="--", zorder=5)
    pos, fb = d["nodes"]["pos"], d["nodes"]["flags"] >= 0.5
    ax.scatter(*pos[cmp["false_n"]].T, s=170, c=[RED], alpha=0.45, linewidths=0, zorder=6)
    ax.scatter(*pos.T, s=6 + 8 * d["nodes"]["e"], c="#26324d", zorder=7)
    for f in sorted(MARK, key=lambda f: -MARK[f][2]):
        sel = fb[:, QT.FI[f]]
        if sel.any():
            m, c, s = MARK[f]
            ax.scatter(*pos[sel].T, marker=m, s=s, c=c, edgecolors="k", linewidths=0.5, zorder=8, clip_on=False)


def draw_ids(ax_gt, ax_pr, g, gt, d):
    _show(ax_gt, underlay(gt))
    for p in g["pieces"]:
        q, c = np.asarray(p["pts"]), piece_rgb(p["piece"])
        if len(q) < 2:
            continue
        ax_gt.plot(q[:, 0], q[:, 1], color=c, lw=2.4, solid_capstyle="round", zorder=3)
        m = len(q) // 2
        ax_gt.annotate("", xy=q[min(m + 1, len(q) - 1)], xytext=q[m], arrowprops=dict(arrowstyle="-|>", color=c, lw=1.2), zorder=4)
        if p["length"] > 8:
            ax_gt.text(q[m, 0], q[m, 1], " " + str(p["piece"])[:4], fontsize=5, color="0.2", zorder=5)
    _show(ax_pr, underlay(gt))
    n, l = d["nodes"], d["links"]
    col = expected_rgb(d["assign"]) if d["assign"] is not None else np.array([piece_rgb(s) for s in n["section"]]).reshape(-1, 3)
    segs, cs, marks = [], [], []
    for c, a, b in zip(curves_of(d), l["src"], l["dst"]):
        c = np.asarray(c)
        t = np.linspace(0, 1, len(c) - 1)[:, None]
        segs += list(np.stack([c[:-1], c[1:]], 1))
        cs += [col[b]] * len(t)                                          # a link has the colour of the section it leads into
        if n["section"][a] != n["section"][b]:
            marks.append(c[len(c) // 2])
    if segs:
        ax_pr.add_collection(LineCollection(segs, colors=cs, linewidths=2.4, zorder=3, capstyle="round"))
    ax_pr.scatter(*n["pos"].T, s=14 + 20 * n["e"], c=col, edgecolors="0.25", linewidths=0.4, zorder=4)
    if marks:
        ax_pr.scatter(*np.array(marks).T, marker="D", s=22, c="k", zorder=6)


def _f(v, nd=3):
    return "-" if v is None or v != v else f"{v:.{nd}f}"


def draw_numbers(ax, sn, sc, ctx):
    """the text panel: the scorer's numbers `sn` (missing keys show "-"), the counts behind the tints, the pixel layers `sc` {name: (P, R)}"""
    ax.axis("off")
    f = lambda k, nd=3: f"{_f(sn.get(k), nd):>6}"
    L = ["scorer (qgraph_eval.py)",
         f" nodes   P{f('node_P')} R{f('node_R')} F1{f('node_F1')}  pos err{f('pos_err_med', 2)} m",
         f" links   short P{f('link_short_P')} R{f('link_short_R')}   long P{f('link_long_P')} R{f('link_long_R')}",
         f" graph   F1@1.0{f('F1@1.0')}  Q{f('Q')}  lanes found{f('lanes_found')}",
         f"         topo F1{f('topo_F1')}  junction F1{f('junction_F1')}",
         f" assign  lane_acc{f('lane_acc')}  road_acc{f('road_acc')}  dust P{f('dust_P')} R{f('dust_R')}", "",
         "what the tints show (nearest node within 2 m)",
         f" delivered {ctx['n_nodes']} nodes, {ctx['n_links']} links ({ctx['n_long']} long)", f" GT        {ctx['gt_nodes']} nodes, {ctx['gt_links']} links",
         f" false nodes {ctx['false_n']}, missed {ctx['missed_n']};  false links {ctx['false_l']}, missed {ctx['missed_l']}", ""]
    if sc:
        L.append("pixel layers (P / R)")
        L += [f"  {k:13s}{_f(v[0]):>5} / {_f(v[1]):>5}" for k, v in sc.items()]
    else:
        L.append("no dense maps delivered (pred['maps'] is None)")
    ax.text(0, 1, "\n".join(L), va="top", family="monospace", fontsize=7.2, transform=ax.transAxes)


def draw_legend(ax):
    ax.axis("off"), ax.set_xlim(0, 1), ax.set_ylim(0, 1)
    for i, f in enumerate(MARK):
        m, c, _ = MARK[f]
        x = 0.01 + i * 0.105
        ax.scatter([x], [0.82], marker=m, s=40, c=c, edgecolors="k", linewidths=0.5, transform=ax.transAxes, clip_on=False)
        ax.text(x + 0.012, 0.82, f, fontsize=7, va="center", transform=ax.transAxes)
    sw = [("GT (faint grey)", 0.84 * np.ones(3)), ("correct (dark)", 0.2 * np.ones(3)), ("false positive / false node, link", 1 - ALPHA + ALPHA * RED),
          ("miss / missed node, link", 1 - ALPHA + ALPHA * BLUE)]
    for i, (t, c) in enumerate(sw):
        x = 0.01 + i * 0.25
        ax.add_patch(plt.Rectangle((x, 0.52), 0.015, 0.16, fc=c, ec="0.5", lw=0.4, transform=ax.transAxes))
        ax.text(x + 0.02, 0.60, t, fontsize=7, va="center", transform=ax.transAxes)
    for i, (k, h) in enumerate(HUE.items()):
        x = 0.01 + i * 0.16
        ax.plot([x, x + 0.02], [0.32, 0.32], color=h, lw=3, transform=ax.transAxes)
        ax.text(x + 0.025, 0.32, k, fontsize=7, va="center", transform=ax.transAxes)
    ax.plot([0.52, 0.54], [0.32, 0.32], color="#26324d", lw=1.4, transform=ax.transAxes), ax.text(0.545, 0.32, "short link", fontsize=7, va="center", transform=ax.transAxes)
    ax.plot([0.62, 0.64], [0.32, 0.32], color="#7a1fa2", lw=1.4, ls="--", transform=ax.transAxes), ax.text(0.645, 0.32, "long link", fontsize=7, va="center", transform=ax.transAxes)
    ax.scatter([0.73], [0.32], marker="D", s=18, c="k", transform=ax.transAxes), ax.text(0.742, 0.32, "section changes", fontsize=7, va="center", transform=ax.transAxes)
    ax.text(0.01, 0.08, "ids: left the GT pieces (colour = hash of the piece id, as in the cameras); right the delivered nodes in the expected colour sum_j P(j | node) colour_j, dustbin grey.  "
            "junctions (GT split / merge): faint dotted circles.  1 m and 0.5 m cells: tolerance of one cell.", fontsize=6.5, transform=ax.transAxes)


def render_frame(token, pred, out_png, cams=True, g=None, note="", log=print):
    """one figure of a keyframe: -> the RGB array (H, W, 3); written to `out_png` (a path, or None). `g` = the lane GT graph (loaded when not given)"""
    g = g if g is not None else load_gt(token)
    nt = QT.node_table(g)
    d = readout(pred)
    cmp = compare_graph(d, nt)
    sn = scorer_numbers(pred, g, token)
    gt = gt_layers(g)
    pr = pred_layers(pred["maps"]) if pred["maps"] is not None else None
    lay = layout(cams)
    W, H = lay["size"]
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    if cams:
        tiles = camera_tiles(token, g, {c: lay["cams"][c][2:] for c in CAMS}, log)
        for c in CAMS:
            x, y, _, h = lay["cams"][c]
            fig.figimage(tiles[c], xo=x, yo=H - y - h, origin="upper", zorder=0)
    B = lay["bev"]
    ctx = dict(n_nodes=len(d["nodes"]["pos"]), n_links=len(d["links"]["src"]), n_long=int(d["links"]["long"].sum()), gt_nodes=len(nt["pos"]), gt_links=len(nt["src"]),
               false_n=int(cmp["false_n"].sum()), missed_n=int(cmp["missed_n"].sum()), false_l=int(cmp["false_l"].sum()), missed_l=int(cmp["missed_l"].sum()))
    img, (pc, rc) = centre_panel(gt, pr, d, cmp, nt)
    _show(_axes(fig, B["centre"], lay["size"], f"1 m centre cells ({'maps' if pr is not None else 'node cells'}): P {_f(pc, 2)} R {_f(rc, 2)}"), img)
    img, sc = area_panel(gt, pr)
    _show(_axes(fig, B["areas"], lay["size"], "2 m road / intersection, 0.5 m dividers / boundary"), img)
    sc = dict(centre=(pc, rc), **sc)
    draw_nodes(_axes(fig, B["nodes"], lay["size"], "nodes by type, Hermite links, false (red) / missed (blue)"), g, nt, d, cmp, gt)
    draw_ids(_axes(fig, B["ids_gt"], lay["size"], "ids: GT pieces"), _axes(fig, B["ids_pred"], lay["size"], "ids: delivered, expected colour"), g, gt, d)
    draw_numbers(_axes(fig, B["numbers"], lay["size"]), sn, sc if pr is not None else {}, ctx)
    draw_legend(_axes(fig, lay["legend"], lay["size"]))
    fi = frame_info(token)
    where = f"{fi[1]} (VAL {fi[0]}) frame {fi[2] + 1}/{fi[3]}" if fi else g.get("location", "")
    fig.text(8 / W, 1 - 20 / H, f"{token}   {where}   {g.get('location', '')}   GT lane signature {g.get('signature', '?')}   {note}", fontsize=10, va="center")
    fig.text(8 / W, 1 - 38 / H, "   ".join(f"{k} {_f(sn.get(key))}" for k, key in (("node F1", "node_F1"), ("F1@1.0", "F1@1.0"), ("topo F1", "topo_F1"), ("junction F1", "junction_F1"),
                                                                                   ("lane_acc", "lane_acc"))) + "   [qgraph_eval]", fontsize=10, va="center", family="monospace")
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    if out_png:
        os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
        fig.savefig(out_png, dpi=100)
    plt.close(fig)
    return arr


def render_scene(scene, out_dir, pred_fn=None, kind="oracle", video=False, frames=None, every=1, cams=True, fps=2, log=print):
    """the keyframes of a VAL scene in time order -> PNGs in out_dir (and an mp4); `pred_fn(token)` gives the prediction (default: `make_pred(load_gt(token), kind, token)`); `kind` also labels the files"""
    _, name, toks = resolve_scene(scene)
    toks = toks[::every][:frames]
    pred_fn = pred_fn or (lambda t: make_pred(load_gt(t), kind, t))
    arrs = []
    for i, t in enumerate(toks):
        g = load_gt(t)
        p = os.path.join(out_dir, f"{name}_{kind}_f{i:02d}_{t[:8]}.png")
        arrs.append(render_frame(t, pred_fn(t), p, cams=cams, g=g, log=log))
        log(f"{p}")
    if video and arrs:
        import make_video as MV
        mp4 = MV.frames_to_video(arrs, os.path.join(out_dir, f"{name}_{kind}.mp4"), fps=fps)
        log(f"video {mp4}")
    return arrs


# ------------------------------------------------------------------------------------------------------------------------------------- selftest
def _synthetic():
    """a small lane GT graph: a lane that splits into a straight piece and a connector through an intersection, a lane of its own beside them, dividers, a boundary, a road and an intersection polygon"""
    ys = lambda a, b: np.stack([np.zeros(int((b - a) / 0.5) + 1), np.arange(a, b + 1e-9, 0.5)], 1)
    A = QT._piece("A", ys(-48, 0), succ=["B", "C"], start_open=True)
    B = QT._piece("B", ys(0, 30), end_open=True, pred=["A"])
    D = QT._piece("D", np.stack([np.full(81, 10.0), np.arange(-30, 10.1, 0.5)], 1))                  # a lane of its own, off the axis in x and y: a flipped image shows
    s = np.linspace(0, 1, 61)
    C = QT._piece("C", np.stack([22 * s ** 2, 30 * s], 1), kind="connector", pred=["A"], end_open=True, inter=True)
    ev = [dict(type="window", io="in", pos=[0.0, -48.0], **{"in": [], "out": ["A"]}, dirs_in=[], dirs_out=[[0, 1]], partial=False),
          dict(type="split", pos=[0.0, 0.0], **{"in": ["A"], "out": ["B", "C"]}, dirs_in=[[0, 1]], dirs_out=[[0, 1], [0, 1]], partial=False),
          dict(type="stop", pos=[0.0, 0.0], on="A", s=48.0, n_succ=2, **{"in": ["A"], "out": ["B", "C"]}, dirs_in=[[0, 1]], dirs_out=[[0, 1], [0, 1]], partial=False),
          dict(type="window", io="out", pos=[0.0, 30.0], **{"in": ["B"], "out": []}, dirs_in=[[0, 1]], dirs_out=[], partial=False),
          dict(type="window", io="out", pos=[22.0, 30.0], **{"in": ["C"], "out": []}, dirs_in=[[0, 1]], dirs_out=[], partial=False)]
    line = lambda x: dict(pts=np.stack([np.full(201, x), np.linspace(-50, 50, 201)], 1).astype(np.float32), start_open=True, end_open=True, corners=[])
    sq = lambda x0, y0, x1, y1: np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)
    return dict(pieces=[A, B, C, D], events=ev, lines=dict(lane_divider=[line(-1.75)], road_divider=[], boundary=[line(-5.0), line(5.0)]), location="synthetic", sample="synthetic",
                areas=dict(road=[dict(id=None, exterior=sq(-6, -50, 6, 50), holes=[], area=0.0), dict(id=None, exterior=sq(0, 0, 40, 34), holes=[], area=0.0)],
                           intersection=[dict(id=None, exterior=sq(-6, -6, 30, 34), holes=[], area=0.0)]), signature="selftest", roads=[], half=50.0, frame="lane")


def _selftest():
    import tempfile
    # known answers of the rules
    p, t = np.zeros((1, 9), bool), np.zeros((1, 9), bool)
    p[0, [0, 6]], t[0, 1] = True, True
    c, fp, ms = layer_triple(p, t, 1)
    assert c[0, 0] and fp[0, 6] and not fp[0, 0] and not ms.any(), "tolerance of one cell: (0) is near the GT cell (1), (6) is false"
    c, fp, ms = layer_triple(t, p, 0)
    assert not c.any() and fp[0, 1] and ms[0, [0, 6]].all(), "no tolerance: the shifted cell is false and the GT cells are missed"
    assert np.allclose(expected_rgb(dict(P=np.array([[0.5, 0.5, 0.0], [0.0, 0.0, 1.0]]), map_ids=["a", "b"])),
                       [0.5 * piece_rgb("a") + 0.5 * piece_rgb("b"), GREY]), "expected colour: blend of the two pieces, dustbin grey"
    assert (piece_rgb("00645d3e") == piece_rgb("00645d3e")).all() and not np.allclose(piece_rgb("a"), piece_rgb("b")), "a stable colour per piece id"
    g = _synthetic()
    pred = make_pred(g, "oracle")
    nt = QT.node_table(g)
    d = readout(pred)
    cmp = compare_graph(d, nt)
    sn = scorer_numbers(pred, g, "synthetic")
    assert not cmp["false_n"].any() and not cmp["missed_n"].any() and not cmp["false_l"].any() and not cmp["missed_l"].any(), "the oracle has no error"
    assert sn["node_F1"] == 1 and sn["lane_acc"] == 1 and "F1@1.0" not in sn, sn                  # the scorer's numbers; no graph numbers for a token that is no VAL frame
    # three GT nodes in a row, the middle delivered node 3.5 m off: it is false, the GT node is missed, both links are false and both GT links missed; within 2 m is a match
    gt3 = dict(pos=np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]]), src=np.array([0, 1]), dst=np.array([1, 2]))
    d3 = dict(nodes=dict(pos=np.array([[0.0, 0.0], [10.0, 3.5], [20.0, 0.0]])), links=dict(src=np.array([0, 1]), dst=np.array([1, 2])))
    c3 = compare_graph(d3, gt3)
    assert c3["false_n"].tolist() == [False, True, False] and c3["missed_n"].tolist() == [False, True, False], c3
    assert c3["false_l"].all() and c3["missed_l"].all(), "both links touch the wrong node"
    d3["nodes"]["pos"][1] = [10.0, 1.9]
    c3 = compare_graph(d3, gt3)
    assert not (c3["false_n"].any() or c3["missed_n"].any() or c3["false_l"].any() or c3["missed_l"].any()), "within 2 m is a match"
    # the degraded oracle has the faults it promises
    bad = make_pred(g, "degraded", "selftest")
    d = readout(bad)
    cmp = compare_graph(d, nt)
    sn = scorer_numbers(bad, g, "synthetic")
    assert cmp["false_n"].sum() >= 1 and cmp["missed_n"].any() and cmp["false_l"].any() and cmp["missed_l"].any() and sn["node_F1"] < 1 and sn["lane_acc"] < 1, (cmp, sn)
    # the pictures (a synthetic token has no camera info and is no VAL frame: nothing is read from the data); no panel is blank, the tints are where the rules put them
    # (red pixels = false positives, blue = misses: the oracle has neither in the centre panel, which holds only greys and the two tints)
    _CACHE["infos"], _CACHE["frames"] = {}, {}
    empty = dict(pred, nodes=dict(pred["nodes"], e=np.zeros(len(pred["nodes"]["e"]))))
    out, logs = tempfile.mkdtemp(), []
    for kind, pr, cams in (("oracle", pred, False), ("degraded", bad, False), ("no_maps_no_assign", dict(pred, maps=None, assign=None), False), ("empty", empty, False), ("cams", pred, True)):
        lay = layout(cams)
        arr = render_frame("synthetic", pr, os.path.join(out, kind + ".png"), cams=cams, g=g, log=logs.append)
        assert arr.shape == (lay["size"][1], lay["size"][0], 3) and os.path.getsize(os.path.join(out, kind + ".png")) > 5000, kind
        for name, (x, y, w, h) in lay["bev"].items():
            assert len(np.unique(arr[y:y + h, x:x + w].reshape(-1, 3), axis=0)) > 3, f"{kind}: panel {name} is blank"
        x, y, w, h = lay["bev"]["centre"]
        a = arr[y:y + h, x:x + w].astype(int)
        red, blue = int(((a[..., 0] - a[..., 2] > 40) & (a[..., 0] - a[..., 1] > 40)).sum()), int(((a[..., 2] - a[..., 0] > 40) & (a[..., 2] - a[..., 1] > 40)).sum())
        if kind in ("oracle", "cams"):
            assert red == 0 and blue == 0, (kind, red, blue)
            # alignment: the pixel under every delivered node is a dark (correct) cell of the centre panel (the points beyond 48 m have no cell) and a node marker (not the pale
            # underlay) in the nodes panel: an image that is flipped or transposed against the vectors fails
            pos = readout(pr)["nodes"]["pos"]
            pix = lambda name, q: arr[int(lay["bev"][name][1] + (50 - q[1]) / 100 * lay["bev"][name][3]), int(lay["bev"][name][0] + (q[0] + 50) / 100 * lay["bev"][name][2])].astype(int)
            dark = np.mean([pix("centre", q).sum() < 200 for q in pos if np.abs(q).max() < 47.5])
            marked = np.mean([pix("nodes", q).sum() < 500 for q in pos])
            assert dark > 0.97 and marked > 0.97, (kind, dark, marked)
        if kind == "degraded":
            assert red > 0 and blue > 0, (kind, red, blue)
        if kind == "cams":
            assert len(logs) == 6 and "no camera info" in logs[0] and abs(arr[44:404, 0:640].mean() - 60) < 15, "six grey tiles with a note in the log"
    # a frame without lanes (a map hole) and a frame with one piece: drawn, and nothing to degrade
    g0 = dict(g, pieces=[], events=[], lines=dict(lane_divider=[], road_divider=[], boundary=[]), areas=dict(road=[], intersection=[]))
    p0 = make_pred(g0, "degraded", "x")
    render_frame("synthetic", p0, os.path.join(out, "empty_gt.png"), cams=False, g=g0)
    p1 = make_pred(dict(g, pieces=g["pieces"][3:], events=[]), "degraded", "x")
    assert p1["nodes"]["e"].min() == 1 and os.path.getsize(os.path.join(out, "empty_gt.png")) > 5000, "one piece: returned as the oracle"
    print("qgraph_view selftest ok:", out)


# ------------------------------------------------------------------------------------------------------------------------------------------ CLI
def _frame_token(s):
    if ":" in s:                                                        # scene-0221:12
        name, i = s.split(":")
        return resolve_scene(name)[2][int(i)]
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    for name in ("frame", "scene"):
        p = sub.add_parser(name)
        p.add_argument("what", help="sample token or scene-0221:12" if name == "frame" else "VAL scene id or name")
        p.add_argument("--out", required=True)
        p.add_argument("--pred", default="oracle", choices=("oracle", "degraded"))
        p.add_argument("--no-cams", action="store_true")
        if name == "scene":
            p.add_argument("--video", action="store_true")
            p.add_argument("--frames", type=int, default=None)
            p.add_argument("--every", type=int, default=1)
            p.add_argument("--fps", type=int, default=2)
    sub.add_parser("list")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if a.cmd == "frame":
        tok = _frame_token(a.what)
        render_frame(tok, make_pred(load_gt(tok), a.pred, tok), os.path.join(a.out, f"{tok[:8]}_{a.pred}.png"), cams=not a.no_cams)
        print("written", os.path.join(a.out, f"{tok[:8]}_{a.pred}.png"))
    elif a.cmd == "scene":
        render_scene(a.what, a.out, kind=a.pred, video=a.video, frames=a.frames, every=a.every, cams=not a.no_cams, fps=a.fps)
    elif a.cmd == "list":
        with open(os.path.join(NUSC, "v1.0-trainval", "scene.json")) as f:
            desc = {s["name"]: s for s in json.load(f)}
        for k, (name, toks) in val_scenes().items():
            print(f"{k:3d}  {name}  {len(toks):3d} frames  {desc[name]['description']}")
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
