"""The layer table of the map head (training/layers.py) and the 1 m centre-line row (plan 2026-11-03 section 2, "Head spec by kind"); CPU, seconds.

  STEP A, the table changes no number: every old layout (centrelines x dense_objects x stationary) gives the same channel list as the OLD layer_spec
     body (a reference copy below), the OLD map_terms body (a reference copy) gives the same terms and groups as the table-driven one on a synthetic batch
     (to 1e-9, and through BEVLoss.terms_at for the main, camera-aux and radar-aux heads), build_layer_targets reads the same GT layers, and a model with
     the option off has exactly the state_dict keys and shapes of before (golden hashes made from the code of commit 479100d, the tiny test_slots config).
  STEP B, centreline_1m (off by default): the pooling has a known answer on a hand-made 4 x 4 example (mask max-pool, direction = the mask-weighted mean,
     truncated distance min-pool); the truncated distance equals scipy's exact transform; the pooled targets of a synthetic frame; the loss is finite, an
     oracle prediction costs ~0 and a reversed direction does not; through the tiny model the output 'sem_1m' has the shape (B, 4, X / 2, Y / 2), the loss is
     finite, and a gradient reaches every new parameter and the trunk; a model / loss disagreement on the flag is refused.
Run (from the bev_fusion folder): wsl ~/miniconda3/envs/gcar/bin/python tests/test_layers.py   (PYTHONPATH = the repository root for GtGenerator)
"""
import hashlib
import math
import os
import sys
from itertools import product

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
for sub in ("tests/support", "tests", "training"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import layers as LY  # noqa: E402
import losses as L  # noqa: E402
import synthetic_world as sw  # noqa: E402
from feature_cache import CachedBEVFusionModel  # noqa: E402
from lift_pd import Grid  # noqa: E402
from model import ModelConfig, check_config  # noqa: E402
from test_slots import batch_for  # noqa: E402

torch.set_num_threads(2)
fails = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        fails.append(msg)


# ------------------------------------------------------------------------------------------------ reference copies of the OLD code (commit 479100d)
def old_layer_spec(centrelines=False, dense_objects=True, stationary=False):
    spec = [(a, "area", a) for a in LY.AREAS]
    for ln in LY.LINES:
        spec += [(ln, "line", ln)]
    spec += [(ln + "_dist", "dist", ln + "_dist") for ln in LY.LINES]
    if centrelines:
        spec += [("centreline", "centreline", "centreline"), ("centreline_cos", "dir", "centreline_dir"),
                 ("centreline_sin", "dir", "centreline_dir")]
    if dense_objects:
        spec += [(o, "object", o) for o in LY.OBJECTS]
    if stationary:
        spec += [("radar_stationary", "stationary", "radar_stationary")]
    return spec


def old_radar_aux_spec(stationary=True):
    return [(o, "object", o) for o in LY.OBJECTS] + ([("radar_stationary", "stationary", "radar_stationary")] if stationary else [])


def old_target_keys(centrelines):
    keys = list(LY.AREAS) + list(LY.LINES) + [ln + "_dist" for ln in LY.LINES] + list(LY.OBJECTS) + ["velocity"]
    return keys + (["centreline", "centreline_dir"] if centrelines else [])


def old_map_terms(logits, T, spec, weights, cfg):
    idx = {}
    for i, (_, kind, _) in enumerate(spec):
        idx.setdefault(kind, []).append(i)
    names = {k: [spec[i][0] for i in v] for k, v in idx.items()}

    def ch(kind):
        return logits[:, idx[kind]]

    def tg(kind):
        return torch.cat([T[spec[i][2]] for i in idx[kind]], 1)
    terms, groups = {}, {}
    if "area" in idx:
        x, t, w = ch("area"), tg("area"), weights["area"]
        bce, iou = L.soft_bce(x, t, w), L.soft_iou_loss(torch.sigmoid(x), t, w)
        for j, n in enumerate(names["area"]):
            terms[f"area_bce.{n}"], terms[f"area_iou.{n}"] = bce[j], iou[j]
        groups["areas"] = (bce + cfg.w_area_iou * iou).mean()
    if "line" in idx:
        w = weights["line"]
        dice = L.tolerant_dice_loss(torch.sigmoid(ch("line")), tg("line"), w, cfg.line_tolerance)
        hub = L.dist_huber(ch("dist"), tg("dist"), w, cfg.huber_delta, cfg.dist_near, cfg.dist_w_far, cfg.dist_max)
        for j, n in enumerate(names["line"]):
            terms[f"line_dice.{n}"] = dice[j]
        for j, n in enumerate(names["dist"]):
            terms[f"line_huber.{n}"] = hub[j]
        groups["lines"] = dice.mean() + hub.mean()
    if "centreline" in idx:
        w = weights["line"]
        dice = L.tolerant_dice_loss(torch.sigmoid(ch("centreline")), tg("centreline"), w, cfg.line_tolerance)[0]
        dl = L.direction_l1(ch("dir"), T["centreline_dir"], T["centreline"], w)
        terms["centreline_dice"], terms["centreline_dir"] = dice, dl
        groups["lines"] = groups.get("lines", 0.0) + dice + cfg.w_direction * dl
    if "object" in idx:
        x, t, w = ch("object"), tg("object"), weights["object"]
        bce, iou = L.soft_bce(x, t, w), L.soft_iou_loss(torch.sigmoid(x), t, w)
        for j, n in enumerate(names["object"]):
            terms[f"obj_bce.{n}"], terms[f"obj_iou.{n}"] = bce[j], iou[j]
        groups["objects"] = (bce + iou).mean()
    if "stationary" in idx and "radar_stationary" in T:
        x, t, w = ch("stationary"), T["radar_stationary"], weights["stationary"]
        bce = L.soft_bce(x, t, w)[0]
        dice = L.tolerant_dice_loss(torch.sigmoid(x), t, w, 0)[0]
        terms["stationary_bce"], terms["stationary_dice"] = bce, dice
        groups["stationary"] = bce + dice
    return terms, groups


class OldBEVLoss(L.BEVLoss):
    """BEVLoss as it was before the table (commit 479100d): the old specs, the old terms_at (without the S5c refine branch, which the table does not
    touch) and the old forward, all on old_map_terms; the Kendall weights, the dt weights and the total come out of the same copy."""

    def __init__(self, cfg, grid):
        super().__init__(cfg, grid)
        c = self.cfg
        self.spec, self.cam_spec, self.radar_spec = old_layer_spec(c.centrelines, c.dense_objects), old_layer_spec(False, True), old_radar_aux_spec(c.radar_stationary)

    def terms_at(self, p, gts, dt):
        cfg = self.cfg
        dev = next(v for v in p.values() if torch.is_tensor(v)) if "obj" not in p else p["obj"]["hm"]
        T = L.build_layer_targets(gts, cfg, p.get("mass"), dt, dev.device)
        terms, groups, fixed = {}, {}, {}

        def add(g, v):
            groups[g] = groups.get(g, 0.0) + v
        if "sem" in p:
            w = {"area": T["w_map"], "line": T["w_lines"], "object": T["w_obj"]}
            t, g = old_map_terms(p["sem"], T, self.spec, w, cfg)
            terms.update(("sem." + k, v) for k, v in t.items())
            for k, v in g.items():
                add(k, v)
        tt = None
        if "obj" in p:
            lo, tt = L.object_head_terms(p["obj"], T["boxes"], T["w_obj"], self.grid, self.groups, cfg.w_obj_vel, cfg.max_obj, cfg.w_obj_mov, hygiene=cfg.hygiene_2026_11_02)
            terms.update(("objhead." + k, v) for k, v in lo.items())
            add("objects", lo["total"])
        if "v_hat" in p and abs(float(dt)) < 1e-6:
            terms["motion"] = L.motion_term(p["v_hat"], T, cfg)
            lv = [L.motion_term(v, T, cfg) for v in (p.get("v_levels") or {}).values()]
            for i, t in enumerate(lv):
                terms[f"motion_l{i}"] = t
            add("motion", (terms["motion"] + sum(lv)) / (1 + len(lv)))
            if cfg.consistency and tt is not None:
                terms["motion_consistency"] = L.consistency_term(p["obj"]["vel"], p["v_hat"], tt)
                add("motion", cfg.w_consistency * terms["motion_consistency"])
        if "cam_aux" in p:
            wc = T["w_cam"]
            t, g = old_map_terms(p["cam_aux"], T, self.cam_spec, {"area": wc, "line": wc, "object": wc * T["w_keep"]}, cfg)
            terms.update(("cam_aux." + k, v) for k, v in t.items())
            fixed["cam_aux"] = cfg.w_cam_aux * sum(g.values())
        if "radar_aux" in p:
            wr = T["w_radar"]
            t, g = old_map_terms(p["radar_aux"], T, self.radar_spec, {"object": wr * T["w_keep"], "stationary": wr}, cfg)
            terms.update(("radar_aux." + k, v) for k, v in t.items())
            fixed["radar_aux"] = cfg.w_radar_aux * sum(g.values())
        return terms, groups, fixed

    def forward(self, preds, gts, step_frac=1.0):
        out, group_tot, aux = {}, {}, 0.0
        for dt, p in preds.items():
            w = L.dt_weight(dt, step_frac, self.cfg)
            if w <= 0:
                continue
            terms, groups, fixed = self.terms_at(p, gts[dt], dt)
            tag = f"@{float(dt):+.1f}"
            out.update((k + tag, v) for k, v in terms.items())
            for g, v in groups.items():
                group_tot[g] = group_tot.get(g, 0.0) + w * v
            for k, v in fixed.items():
                aux = aux + w * v
        total, s = self.task(group_tot)
        out.update(("group." + g, v) for g, v in group_tot.items())
        out.update(("s." + g, v) for g, v in s.items())
        out["aux"] = aux
        out["total"] = total + aux
        return out


# ------------------------------------------------------------------------------------------------ the synthetic frame
grid = Grid()
X, Y = grid.nx, grid.ny
scene = L._demo_scene(sw)
gts = {dt: [L.synthetic_gt(scene.step(dt) if dt else scene, grid)] for dt in (0.0, 0.5)}


def logits(n, h=X, w=Y, seed=3):
    return torch.randn(1, n, h, w, generator=torch.Generator().manual_seed(seed))


def same(a, b):
    """two dicts of terms: the same keys (and order), values equal to 1e-9"""
    return list(a) == list(b) and all(abs(float(a[k]) - float(b[k])) <= 1e-9 for k in a)


# ------------------------------------------------------------------------------------------------ STEP A
print("STEP A: the table reproduces the old layouts")
base = [("drivable_area", "area", "drivable_area"), ("ped_crossing", "area", "ped_crossing"), ("walkway", "area", "walkway"),
        ("stop_line", "area", "stop_line"), ("carpark_area", "area", "carpark_area"), ("lane_divider", "line", "lane_divider"),
        ("road_divider", "line", "road_divider"), ("lane_divider_dist", "dist", "lane_divider_dist"), ("road_divider_dist", "dist", "road_divider_dist"),
        ("vehicle", "object", "vehicle"), ("pedestrian", "object", "pedestrian"), ("two_wheeler", "object", "two_wheeler")]
check(L.layer_spec() == base, "default layout = the 12 channels of the first design, in that order (areas, line masks, line distances, objects)")
for cl, do, st in product((False, True), repeat=3):
    check(L.layer_spec(cl, do, st) == old_layer_spec(cl, do, st), f"layer_spec(centrelines={cl}, dense_objects={do}, stationary={st}) = the old body")
check(all(L.radar_aux_spec(s) == old_radar_aux_spec(s) for s in (False, True)), "radar_aux_spec = the old body (stationary on / off)")
check(L.layer_spec_1m(False) == [], "the 1 m branch has no channel while the option is off")
check(L.BEVLoss(L.LossConfig(centreline_1m=True)).spec == base, "the option does not touch the 0.5 m layout (the same 12 channels with it on)")
check(all(not k.endswith("@1m") for _, _, k in L.layer_spec(True, True)), "no 0.5 m channel points at a pooled target")
for bad, why in ((dict(name="x", kind="blob"), "kind"), (dict(name="x", kind="area", level="2m"), "level"), (dict(name="x", kind="area", dist=True), "dist on an area")):
    try:
        LY.Layer(**bad)
        check(False, f"a row with a bad {why} is refused")
    except ValueError:
        check(True, f"a row with a bad {why} is refused")

for cl, do in product((False, True), repeat=2):                   # the losses on the same logits / targets / weights
    cfg = L.LossConfig(centrelines=cl, dense_objects=do)
    T = L.build_layer_targets(gts[0.0], cfg)
    w = {"area": T["w_map"], "line": T["w_lines"], "object": T["w_obj"]}
    spec = L.layer_spec(cl, do)
    got, gg = L.map_terms(logits(len(spec)), T, spec, w, cfg)
    ref, rg = old_map_terms(logits(len(spec)), T, old_layer_spec(cl, do), w, cfg)
    check(same(got, ref) and same(gg, rg) and len(got) > 0, f"map_terms = the old body: centrelines={cl} dense_objects={do} ({len(got)} terms, groups {list(gg)})")
derived = {"boxes", "obj_cells", "w_map", "w_lines", "w_obj", "w_cam", "w_radar", "w_keep", "w_fusion", "radar_stationary"}      # made from the layers, not read
for cl in (False, True):
    check(set(L.build_layer_targets(gts[0.0], L.LossConfig(centrelines=cl))) == set(old_target_keys(cl)) | derived,
          f"build_layer_targets reads the GT layers of before (centrelines={cl})")
# through BEVLoss.terms_at: main, camera-aux and radar-aux heads against the old body on the same targets
for cl, rs in product((False, True), repeat=2):
    cfg = L.LossConfig(centrelines=cl, radar_stationary=rs)
    lm = L.BEVLoss(cfg, grid)
    p = {"sem": logits(len(lm.spec)), "cam_aux": logits(len(lm.cam_spec), seed=4), "radar_aux": logits(len(lm.radar_spec), seed=5)}
    terms, groups, fixed = lm.terms_at(p, gts[0.0], 0.0)
    T = L.build_layer_targets(gts[0.0], cfg)
    ref = {}
    ref.update(("sem." + k, v) for k, v in old_map_terms(p["sem"], T, old_layer_spec(cl, True), {"area": T["w_map"], "line": T["w_lines"], "object": T["w_obj"]}, cfg)[0].items())
    wc = T["w_cam"]
    ref.update(("cam_aux." + k, v) for k, v in old_map_terms(p["cam_aux"], T, old_layer_spec(False, True), {"area": wc, "line": wc, "object": wc * T["w_keep"]}, cfg)[0].items())
    wr = T["w_radar"]
    ref.update(("radar_aux." + k, v) for k, v in old_map_terms(p["radar_aux"], T, old_radar_aux_spec(rs), {"object": wr * T["w_keep"], "stationary": wr}, cfg)[0].items())
    check(same(terms, ref), f"BEVLoss.terms_at (centrelines={cl}, radar_stationary={rs}): sem, cam_aux and radar_aux terms = the old body ({len(ref)} terms)")

# the whole BEVLoss at dt 0 and 0.5 against the old copy: every term, the dt-weighted groups, the Kendall terms (s_g set to non-trivial values), aux and the total
for cl, rs in product((False, True), repeat=2):
    cfg = L.LossConfig(centrelines=cl, radar_stationary=rs)
    new, old = L.BEVLoss(cfg, grid), OldBEVLoss(cfg, grid)
    s_g = torch.tensor([0.3, -0.7, 0.9, -1.1])
    for lm in (new, old):
        lm.task.s.data.copy_(s_g)
    g = torch.Generator().manual_seed(7)
    preds = {}
    for dt in (0.0, 0.5):
        q = {"sem": logits(len(new.spec), seed=11), "cam_aux": logits(len(new.cam_spec), seed=12), "radar_aux": logits(len(new.radar_spec), seed=13),
             "obj": {k: torch.randn(1, c, X, Y, generator=g) for k, c in (("hm", 3), ("off", 2), ("size", 2), ("rot", 2), ("rad", 1), ("vel", 2))}}
        if dt == 0.0:
            q["v_hat"] = torch.randn(1, 2, X, Y, generator=g)
        else:
            q["mass"] = torch.rand(1, 1, X, Y, generator=g)
        preds[dt] = q
    got, ref = new(preds, gts, 0.7), old(preds, gts, 0.7)
    groups_ = [k for k in ref if k.startswith(("group.", "s."))] + ["aux", "total"]
    check(same(got, ref) and len(groups_) >= 8 and abs(float(ref["total"])) > 1.0,
          f"whole BEVLoss = the old copy: centrelines={cl} radar_stationary={rs} ({len(ref)} entries: terms, {len(groups_) - 2} group / Kendall, aux, total {float(ref['total']):.3f})")

# the network: the state_dict of a model with the option off is that of before (golden: key + shape rows of the code of 479100d)
GRID = {"x_max": 16.0, "y_min": -8.0, "y_max": 8.0}                 # 32 x 32 cells
TINY = dict(pretrained="", trunk_pretrained="", grid_kwargs=GRID, oversample=1, c_cam=16, c_rad=8, c_fuse=24, trunk_width=0.25, p_drop=0.0)
GOLDEN = {"default": ({}, 209, "b7e31c1084af7885"), "centrelines": ({"centrelines": True}, 209, "3c6612bfd4da4774"),
          "no dense objects": ({"dense_objects": False}, 209, "04c62437d5e98199"), "no aux heads": ({"aux_heads": False}, 193, "caf7192e586544e3")}


def sd_rows(m):
    return [f"{k} {tuple(v.shape)}" for k, v in m.state_dict().items()]


for name, (kw, n, sha) in GOLDEN.items():
    rows = sd_rows(CachedBEVFusionModel(ModelConfig(**dict(TINY, **kw))))
    check(len(rows) == n and hashlib.sha256("\n".join(rows).encode()).hexdigest()[:16] == sha, f"state_dict keys + shapes of the {name} model = before ({len(rows)} tensors)")

# ------------------------------------------------------------------------------------------------ STEP B
print("STEP B: the 1 m centre-line row")
check(not L.LossConfig().centreline_1m and not ModelConfig().centreline_1m, "the option is off by default in LossConfig and ModelConfig")
check(L.layer_spec_1m(True) == [("centreline", "line", "centreline@1m"), ("centreline_dist", "dist", "centreline_dist@1m"),
                                ("centreline_cos", "dir", "centreline_dir@1m"), ("centreline_sin", "dir", "centreline_dir@1m")],
      "the 1 m row gives presence, distance, (cos, sin) with the targets 'centreline@1m', 'centreline_dist@1m', 'centreline_dir@1m'")

# the pooling, a hand-made 4 x 4 example (0.5 m cells)
m4 = torch.zeros(1, 1, 4, 4)
m4[0, 0, 0, 0] = 1
m4[0, 0, 2, 2] = m4[0, 0, 2, 3] = m4[0, 0, 3, 3] = 1
d4 = torch.zeros(1, 2, 4, 4)
d4[:, :, 0, 0] = torch.tensor([0.6, 0.8]).view(1, 2)                         # block (0, 0): one line cell
d4[:, :, 2, 2] = torch.tensor([1.0, 0.0]).view(1, 2)                         # block (1, 1): three line cells, headings x, y, y
d4[:, :, 2, 3] = d4[:, :, 3, 3] = torch.tensor([0.0, 1.0]).view(1, 2)
pm, pd, pu = LY.pool_polyline(m4, d4)
check(pm.shape == (1, 1, 2, 2) and pd.shape == (1, 1, 2, 2) and pu.shape == (1, 2, 2, 2), "pooled shapes (B, 1 / 1 / 2, X / 2, Y / 2)")
check(torch.equal(pm[0, 0], torch.tensor([[1.0, 0.0], [0.0, 1.0]])), "mask: max-pool of the 2 x 2 blocks")
check(torch.allclose(pd[0, 0], torch.tensor([[0.0, 0.5], [0.5, 0.0]])), "distance: the block's nearest line cell (0 / 0.5 / 0.5 / 0 m)")
check(torch.allclose(pu[0, :, 0, 0], torch.tensor([0.6, 0.8])) and torch.allclose(pu[0, :, 1, 1], torch.tensor([1 / 3, 2 / 3]))
      and float(pu[0, :, 0, 1].abs().sum() + pu[0, :, 1, 0].abs().sum()) == 0.0,
      "direction: the mask-weighted mean of the block ((0.6, 0.8) is not diluted by the 3 empty cells; (1/3, 2/3) of three cells); 0 in an empty block")
try:
    LY.pool_polyline(torch.zeros(1, 1, 5, 4))
    check(False, "an odd grid is refused")
except ValueError:
    check(True, "an odd grid is refused")

# the truncated distance field against the exact transform
one_cell = torch.zeros(1, 1, 13, 13)
one_cell[0, 0, 6, 6] = 1
td = LY.truncated_distance(one_cell)[0, 0]
want = {(0, 0): 0.0, (1, 0): 0.5, (1, 1): math.sqrt(2) / 2, (2, 1): math.sqrt(5) / 2, (3, 0): 1.5, (4, 0): 2.0, (3, 4): 2.0, (0, -5): 2.0}
check(all(abs(float(td[6 + dx, 6 + dy]) - v) < 1e-6 for (dx, dy), v in want.items()), "truncated distance of one cell: 0, 0.5, 0.707, 1.118, 1.5 m, capped at 2 m (also at 2.5 m)")
try:
    from scipy.ndimage import distance_transform_edt
    rng = np.random.default_rng(0)
    mm = (rng.random((2, 1, 40, 36)) < 0.03).astype(np.float32)
    ref = np.stack([np.minimum(distance_transform_edt(1 - mm[b, 0]) * LY.GT_RES, LY.DIST_MAX) for b in range(2)])[:, None]
    check(np.abs(LY.truncated_distance(torch.from_numpy(mm)).numpy() - ref).max() < 1e-5, "truncated distance = scipy's exact transform (random masks, 2 m cap)")
except ImportError:
    print("  skip scipy is not installed: the exact-transform comparison")

# the targets of a synthetic frame
cfg1 = L.LossConfig(centreline_1m=True)
T0, T1 = L.build_layer_targets(gts[0.0], L.LossConfig()), L.build_layer_targets(gts[0.0], cfg1)
check(not any("@1m" in k for k in T0), "option off: no pooled target, no weight 'w_lines@1m'")
pm, pd, pu = LY.pool_polyline(T1["centreline"], T1["centreline_dir"])
check(all(torch.equal(T1[k], v) for k, v in (("centreline@1m", pm), ("centreline_dist@1m", pd), ("centreline_dir@1m", pu)))
      and T1["centreline@1m"].shape == (1, 1, X // 2, Y // 2) and T1["centreline_dir@1m"].shape == (1, 2, X // 2, Y // 2) and float(pm.sum()) > 10,
      f"build_layer_targets: the 1 m targets are the pooled 0.5 m ones ({int(pm.sum())} line cells of {pm.numel()}); the 0.5 m centre-line layers are read without centrelines=True")
check(torch.allclose(T1["w_lines@1m"], torch.nn.functional.avg_pool2d(T1["w_lines"], 2)) and all(torch.equal(T1[k], T0[k]) for k in T0 if torch.is_tensor(T0[k])),
      "the 1 m weight is the mean of the 0.5 m line weight; every 0.5 m target and weight is unchanged by the option")
check(float(pd[pm > 0].max()) == 0.0 and float((pd[pm == 0]).min()) > 0.0, "distance is 0 exactly on the line cells, > 0 elsewhere")

# the loss: an oracle costs ~0, a reversed direction does not
spec1 = L.layer_spec_1m(True)
orc = torch.cat([20.0 * (2 * T1["centreline@1m"] - 1), T1["centreline_dist@1m"], T1["centreline_dir@1m"]], 1)     # logits, metres, (cos, sin)
w1 = {"line": T1["w_lines@1m"]}
t_or, g_or = L.map_terms(orc, T1, spec1, w1, cfg1)
check(list(t_or) == ["line_dice.centreline", "line_huber.centreline_dist", "centreline_dir@1m"] and list(g_or) == ["lines"],
      "terms: Dice + Huber of the presence / distance rows, the direction term; one 'lines' group")
check(float(t_or["line_dice.centreline"]) < 1e-4 and float(t_or["line_huber.centreline_dist"]) < 1e-6 and float(t_or["centreline_dir@1m"]) < 1e-5,
      f"oracle prediction: Dice {float(t_or['line_dice.centreline']):.1e}, Huber {float(t_or['line_huber.centreline_dist']):.1e}, direction {float(t_or['centreline_dir@1m']):.1e}")
rev = orc.clone()
rev[:, 2:] = -rev[:, 2:]
t_rv, _ = L.map_terms(rev, T1, spec1, w1, cfg1)
check(float(t_rv["centreline_dir@1m"]) > 0.9 and abs(float(t_rv["line_dice.centreline"]) - float(t_or["line_dice.centreline"])) < 1e-9,
      f"reversed direction: direction term {float(t_rv['centreline_dir@1m']):.2f}, the presence term does not move")
far = orc.clone()
far[:, 1:2] = far[:, 1:2] + 0.8
hub_far = float(L.map_terms(far, T1, spec1, w1, cfg1)[0]["line_huber.centreline_dist"])
check(hub_far > 0.02, f"a distance field 0.8 m off costs (Huber {hub_far:.3f}; the cells beyond the 2 m cap are free)")

# the tiny model with the option on: output, loss, gradient
TINY1 = dict(TINY, centreline_1m=True)


def seeded(**kw):
    torch.manual_seed(0)
    return CachedBEVFusionModel(ModelConfig(**dict(TINY, **kw)))


off, on = seeded(), seeded(centreline_1m=True)
extra = sorted(set(on.state_dict()) - set(off.state_dict()))
check(extra == [f"head_1m.net.{k}" for k in ("0.weight", "1.bias", "1.num_batches_tracked", "1.running_mean", "1.running_var", "1.weight", "3.bias", "3.weight")]
      and not set(off.state_dict()) - set(on.state_dict()), "option on: exactly one new SemanticHead (conv, BN, conv) in the state_dict, nothing else changes")
for kw in ({}, {"aux_heads": False, "dense_objects": False, "object_direction": True}):          # the new head is built last: at one seed every other tensor is the same on / off
    sa, sb = seeded(**kw).state_dict(), seeded(centreline_1m=True, **kw).state_dict()
    check(list(sb)[:len(sa)] == list(sa) and all(k.startswith("head_1m.") for k in list(sb)[len(sa):]) and all(torch.equal(sa[k], sb[k]) for k in sa),
          f"same seed, option on / off ({kw or 'default'}): all {len(sa)} shared tensors equal in value, key order and shape, the 1 m head's keys come last")
on.train()
with torch.no_grad():
    for q in on.parameters():                                       # zero-start layers cut the gradient at init by design (as test_slots does)
        if q.requires_grad and q.abs().max() == 0:
            q.normal_(0, 1e-2)
batch = batch_for(on.cfg)
res = on(batch, dts=(0.0, 0.5))
gx, gy = on.grid.nx, on.grid.ny
check(all(res["out"][dt]["sem_1m"].shape == (2, 4, gx // 2, gy // 2) for dt in (0.0, 0.5)) and res["out"][0.0]["sem"].shape == (2, 12, gx, gy),
      f"output 'sem_1m' (B, 4, {gx // 2}, {gy // 2}) at every dt next to 'sem' (B, 12, {gx}, {gy})")
check(all("sem_1m" not in r["out"][0.0] for r in (off(batch_for(off.cfg), dts=(0.0,)),)), "option off: no 'sem_1m' output")
tgrid = on.grid
tscene = sw.Scene([sw.Box(9.0, -1.75, 4.5, 1.8, 1.5, 0.0, 8.0, 0.0, "car", 10.0)], lanes=[-5.25, -1.75, 1.75, 5.25], v=8.0, omega=0.05)
tg = {dt: [L.synthetic_gt(tscene.step(dt) if dt else tscene, tgrid)] * 2 for dt in (0.0, 0.5)}
lm = L.BEVLoss(L.LossConfig(centreline_1m=True), tgrid)
out = lm({dt: {k: v for k, v in p.items() if k in ("sem", "sem_1m", "mass")} for dt, p in res["out"].items()}, tg, 1.0)
t1 = {k: v for k, v in out.items() if k.startswith("sem1m.")}
check(len(t1) == 6 and all(torch.isfinite(v) for v in out.values() if torch.is_tensor(v)), f"BEVLoss with the option on: finite, {len(t1)} sem1m terms ({', '.join(sorted({k.split('@')[0] for k in t1}))})")
on.zero_grad()
sum(t1.values()).backward()                                         # the 1 m terms alone: nothing else reaches the new parameters
new = [(n, q) for n, q in on.named_parameters() if n.startswith("head_1m.")]
check(len(new) == 5 and all(q.grad is not None and torch.isfinite(q.grad).all() and q.grad.abs().sum() > 0 for _, q in new),
      f"a gradient reaches every new parameter ({len(new)} tensors) of the 1 m branch")
check(all(any(q.grad is not None and q.grad.abs().sum() > 0 for q in mod.parameters()) for mod in (on.trunk, on.sem_head.refine)),
      "... and flows on through the sem head's state into the trunk")
on.eval()                                                           # at dt != 0 the branch reads the MOVED state the sem head decodes, not the trunk state S
with torch.no_grad():
    S, v_, w_ = (lambda r: (r["features"], batch["v"].to(r["features"]), batch["omega"].to(r["features"])))(on(batch, dts=(0.0,)))
    for dt in (0.5, -0.5, 0.0):
        x = on.sem_head(S, dt, v_, w_, return_aux=True)[1]["state"]
        want, wrong = on.head_1m(torch.nn.functional.avg_pool2d(x, 2)), on.head_1m(torch.nn.functional.avg_pool2d(S, 2))
        got = on(batch, dts=(dt,))["out"][dt]["sem_1m"]
        check(torch.allclose(got, want, atol=1e-6) and float((wrong - want).abs().max()) > 1e-3,
              f"dt {dt:+.1f}: sem_1m = head_1m(avg_pool(the sem head's moved state)) (max |diff| {float((got - want).abs().max()):.1e}); reading the unmoved trunk state would differ by {float((wrong - want).abs().max()):.2f}")
for lcfg, mdl, why in ((L.LossConfig(), on, "model has the branch, loss does not"), (L.LossConfig(centreline_1m=True), off, "loss has the row, model does not")):
    try:
        with torch.no_grad():
            L.BEVLoss(lcfg, tgrid)({0.0: {k: v for k, v in mdl(batch_for(mdl.cfg), dts=(0.0,))["out"][0.0].items() if k in ("sem", "sem_1m")}}, {0.0: tg[0.0]}, 1.0)
        check(False, f"a disagreement of the flag is refused ({why})")
    except ValueError:
        check(True, f"a disagreement of the flag is refused ({why})")

check_config(ModelConfig(**TINY1))
try:
    check_config(ModelConfig(**dict(TINY1, grid_kwargs=dict(GRID, x_max=15.5))))                    # 31 x 32 cells
    check(False, "check_config refuses an odd grid for centreline_1m")
except ValueError:
    check(True, "check_config refuses an odd grid for centreline_1m")

print("test_layers " + ("FAILED: " + "; ".join(fails) if fails else "passed"))
sys.exit(1 if fails else 0)
