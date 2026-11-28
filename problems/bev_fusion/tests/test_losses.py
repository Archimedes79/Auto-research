"""Tests of net/loss/losses.py (its former self-test).

    python tests/test_losses.py      (from the problem folder, with its module path)
"""
import math

import numpy as np
import torch
import torch.nn as nn

from net.loss.losses import BEVLoss, LossConfig, TASK_GROUPS, TaskWeights, build_layer_targets, dist_huber, dt_weight, object_head_terms, soft_bce, tolerant_dice_loss
from net.model.layers import DIST_MAX
from net.model.layers import layer_spec, radar_aux_spec
from net.model.lift_pd import Grid
from net.prep.targets import NUSC_GROUPS, head_box


def _selftest_fusion_area():
    """LossConfig.fusion_area_only (D38 (b)), CPU, no optimiser step (a backward pass on leaf tensors only; seconds):
    off by default and then every weight is positive outside camera x radar; on, the fused weights are x (camera x radar) and the aux
    weights stay; an object whose centre lies outside is ignored; through the whole BEVLoss at dt 0 and dt 0.5 (splat mass) the gradient
    of every fused output is exactly 0 outside the fusion area, the aux heads' gradients are identical to those with the option off."""
    from dataclasses import replace
    from tests.support import synthetic_world as sw
    from net.model.object_head import ObjectHead
    torch.set_num_threads(2)
    grid, cfg = Grid(), LossConfig()
    X, Y = grid.nx, grid.ny
    assert cfg.fusion_area_only is False, "the option is off by default: every stored run is reproduced"
    scene = sw.demo_scene()
    gts = {dt: [sw.synthetic_gt(scene.step(dt) if dt else scene, grid)] for dt in (0.0, 0.5)}
    T = build_layer_targets(gts[0.0], cfg)
    fa = torch.as_tensor(gts[0.0][0]["camera"] * gts[0.0][0]["radar"])[None, None]
    assert float(fa.min()) == 0.0 and float(fa.max()) == 1.0 and torch.equal(T["w_fusion"], fa)
    for k in ("w_map", "w_lines", "w_obj"):
        assert float(T[k][fa == 0].max()) > 0, f"{k}: off by default, so cells outside the fusion area are supervised"
    Tf = build_layer_targets(gts[0.0], replace(cfg, fusion_area_only=True))
    for k in ("w_map", "w_lines", "w_obj"):
        assert torch.equal(Tf[k], T[k] * fa), k
        assert float(Tf[k][fa == 0].abs().max()) == 0.0 and float(Tf[k][fa > 0].sum()) > 0, k
    for k in ("w_cam", "w_radar", "w_keep", "w_fusion"):
        assert torch.equal(Tf[k], T[k]), k
    cam_, occ_, rad_ = (torch.as_tensor(gts[0.0][0][k])[None, None] for k in ("camera", "occluded", "radar"))
    assert torch.equal(T["w_cam"], cam_ * (1 - occ_)) and torch.equal(T["w_radar"], rad_), "the aux weights are the sensors' own areas"
    assert float(((cam_ > 0) & (rad_ == 0)).sum()) > 0, "the demo scene has camera cells the radar does not see (the aux weights differ there)"
    # an object whose centre cell lies outside the fusion area is ignored (the moving car: blank the camera under its centre cell)
    cx0 = gts[0.0][0]["meta"]["boxes"][0]["center"]
    ix, iy = int((cx0[0] - grid.x_min) / grid.res), int((cx0[1] - grid.y_min) / grid.res)
    gcam = dict(gts[0.0][0], camera=gts[0.0][0]["camera"].copy())
    gcam["camera"][ix - 4:ix + 5, iy - 4:iy + 5] = 0.0
    oh = ObjectHead(8, groups=NUSC_GROUPS)(torch.zeros(1, 8, X, Y))
    npos = {}
    for tag, g_, on in (("off", gts[0.0], False), ("off_blank", [gcam], False), ("on", gts[0.0], True), ("on_blank", [gcam], True)):
        Tg = build_layer_targets(g_, replace(cfg, fusion_area_only=on))
        npos[tag] = int(object_head_terms(oh, Tg["boxes"], Tg["w_obj"], grid, NUSC_GROUPS)[1]["mask"].sum())
    assert npos["off_blank"] == npos["off"] and npos["on_blank"] == npos["on"] - 1 > 0, npos

    # the gradient through the whole BEVLoss: leaf tensors as the model's outputs, dt 0 and dt 0.5 (unknown cells: mass 0 in a strip)
    def grads(on):
        torch.manual_seed(0)
        g = torch.Generator().manual_seed(0)
        lm = BEVLoss(replace(cfg, fusion_area_only=on))
        head = ObjectHead(8, groups=NUSC_GROUPS)
        preds, leaves = {}, {}
        for dt in (0.0, 0.5):
            maps = head(torch.randn(1, 8, X, Y, generator=g))
            p = {"sem": torch.randn(1, len(lm.spec), X, Y, generator=g), "obj": {k: v.detach().clone() for k, v in maps.items()},
                 "cam_aux": torch.randn(1, len(lm.cam_spec), X, Y, generator=g), "radar_aux": torch.randn(1, len(lm.radar_spec), X, Y, generator=g),
                 "mass": torch.ones(1, 1, X, Y)}
            if dt == 0.0:
                p["v_hat"] = torch.randn(1, 2, X, Y, generator=g)
            else:
                p["mass"][..., :20, :] = 0.0
            for k in ("sem", "cam_aux", "radar_aux", "v_hat"):
                if k in p:
                    p[k].requires_grad_()
            for v in p["obj"].values():
                v.requires_grad_()
            preds[dt] = p
            leaves[dt] = p
        out = lm(preds, gts, 1.0)
        out["total"].backward()
        return {dt: {**{k: p[k].grad for k in ("sem", "cam_aux", "radar_aux", "v_hat") if k in p},
                     **{"obj." + k: v.grad for k, v in p["obj"].items()}} for dt, p in leaves.items()}
    g_off, g_on = grads(False), grads(True)
    outside = (fa == 0)
    for dt in (0.0, 0.5):
        for k, gr in g_on[dt].items():
            if k in ("cam_aux", "radar_aux"):
                assert torch.equal(gr, g_off[dt][k]), f"dt {dt} {k}: the aux heads keep their own area (identical gradient with the option off)"
                continue
            m = outside.expand_as(gr)
            assert float(gr[m].abs().max()) == 0.0, f"dt {dt} {k}: gradient outside the fusion area must be exactly 0"
            assert float(gr[~m].abs().max()) > 0.0, f"dt {dt} {k}: no gradient inside the fusion area"
        for k in ("sem", "obj.hm"):
            gr = g_off[dt][k]
            assert float(gr[outside.expand_as(gr)].abs().max()) > 0.0, f"dt {dt} {k}: with the option off the outside is supervised"
        gr = g_on[dt]["radar_aux"]
        assert float(gr[outside.expand_as(gr)].abs().max()) > 0.0, "the radar aux head keeps supervising outside the fusion area"
    print("fusion_area_only selftest passed: off by default; on, the fused weights are x camera x radar, an object centre outside is ignored, "
          "the gradient of every fused output is exactly 0 outside the fusion area at dt 0 and 0.5, the aux gradients equal those without the option")


def _selftest():
    import time
    from dataclasses import replace
    from tests.support import synthetic_world as sw
    from net.model.bev_trunk import SemanticHead
    from net.model.object_head import ObjectHead
    from net.model.timed_head import TimedHead, TimeMover
    torch.manual_seed(0)
    torch.set_num_threads(1)                          # small tensors on a shared CPU: threads only add contention
    t0 = time.time()
    grid, cfg = Grid(), LossConfig()
    X, Y = grid.nx, grid.ny
    spec = layer_spec(cfg.centrelines, cfg.dense_objects)
    print("layer spec (semantic head channels):", ", ".join(f"{i}:{n}" for i, (n, _, _) in enumerate(spec)))
    ones = torch.ones(1, 1, X, Y)

    # 1 BCE on soft targets = the analytic value (also with arbitrary weights)
    z, tv = math.log(3.0), 0.25
    exp = -(tv * math.log(0.75) + (1 - tv) * math.log(0.25))
    w = torch.rand(1, 1, X, Y)
    got = soft_bce(torch.full((1, 2, X, Y), z), torch.full((1, 2, X, Y), tv), w)
    assert torch.allclose(got, torch.tensor(exp), atol=1e-5), (got, exp)
    # 2 tolerant Dice: 0 for a 1-cell shift (either side), > 0 for a 3-cell shift
    G = torch.zeros(1, 1, X, Y)
    G[0, 0, 10:110, 50] = 1.0
    for s, zero in ((1, True), (-1, True), (0, True), (3, False)):
        P = torch.roll(G, s, 3)
        d = float(tolerant_dice_loss(P, G, ones, 1)[0])
        assert (abs(d) < 1e-6) if zero else d > 0.9, (s, d)
    d3 = float(tolerant_dice_loss(torch.roll(G, 3, 3), G, ones, 1)[0])
    # 3 Huber 0 for the exact field; > 0 for a shifted one; a prediction beyond the truncation is free where GT = d_max
    gt0 = sw.synthetic_gt(sw.Scene([]), grid)
    D = torch.as_tensor(gt0["lane_divider_dist"])[None, None]
    assert float(dist_huber(D, D, ones)[0]) == 0.0
    assert float(dist_huber(D + 0.3, D, ones)[0]) > 0.0
    assert float(dist_huber(torch.where(D >= DIST_MAX, D + 5, D), D, ones)[0]) == 0.0

    # the synthetic sample: one moving car, a parked car, a parked truck, a walking pedestrian; GT at dt = 0, +-0.5 s
    scene = sw.demo_scene()
    dts = (0.0, 0.5, -0.5)
    gts = {dt: [sw.synthetic_gt(scene.step(dt) if dt else scene, grid)] for dt in dts}
    T = build_layer_targets(gts[0.0], cfg)
    assert T["drivable_area"].shape == (1, 1, X, Y) and T["velocity"].shape == (1, 2, X, Y)
    assert float(T["w_map"].min()) == 0.0 and float(T["w_map"].max()) == 1.0
    assert abs(float(T["w_map"][T["w_map"] > 0].min()) - cfg.w_occluded_map) < 1e-6, "occluded map cells: 0.5"
    assert "radar_stationary" in T
    gno = {k: v for k, v in gts[0.0][0].items() if k != "radar_stationary"}
    assert "radar_stationary" not in build_layer_targets([gno], cfg), "hook skipped when the GT has no such layer"
    mass = torch.ones(1, 1, X, Y)
    mass[..., :20, :] = 0.0
    Tu = build_layer_targets(gts[0.5], cfg, mass, 0.5)
    Tk = build_layer_targets(gts[0.5], cfg, None, 0.5)
    sel = Tk["w_map"][..., :20, :] > 0
    assert torch.allclose(Tu["w_map"][..., :20, :][sel], cfg.w_unknown * Tk["w_map"][..., :20, :][sel])

    _selftest_fusion_area()                           # D38 (b): `python net/loss/losses.py --fusion-area` runs it alone

    # predictions as leaf tensors: weight-0 cells get exactly zero gradient; doubling the map leaves every term
    cam_n, rad_n = len(layer_spec(False, True)), len(radar_aux_spec(True))

    def leaf_preds(Xn, seed):
        g = torch.Generator().manual_seed(seed)
        r = lambda *s: torch.randn(*s, generator=g)
        p = {}
        for dt in (0.0, 0.5):
            q = {"sem": r(1, len(spec), Xn, Y), "cam_aux": r(1, cam_n, Xn, Y), "radar_aux": r(1, rad_n, Xn, Y),
                 "obj": {k: r(1, c, Xn, Y) for k, c in (("hm", 3), ("off", 2), ("size", 2), ("rot", 2), ("rad", 1),
                                                        ("vel", 2))}}
            if dt == 0.0:
                q["v_hat"] = r(1, 2, Xn, Y)
            else:
                q["mass"] = torch.rand(1, 1, Xn, Y, generator=g)
            p[dt] = q
        return p

    def leaves(p):
        for q in p.values():
            for k, v in q.items():
                for t_ in (v.values() if isinstance(v, dict) else [v]):
                    if k != "mass":
                        yield k, t_
    gz = {dt: [dict(gts[dt][0])] for dt in (0.0, 0.5)}
    for dt in gz:                                     # an undefined block (y < -21 m) and undefined first / last rows
        g = gz[dt][0] = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in gz[dt][0].items()}
        for k in ("camera", "radar", "occluded"):
            g[k][:, :10] = 0
            g[k][[0, -1], :] = 0
        g["undefined"][:, :10] = 1
        g["undefined"][[0, -1], :] = 1
    lossf = BEVLoss(cfg, grid)
    p = leaf_preds(X, 1)
    for _, t_ in leaves(p):
        t_.requires_grad_(True)
    res = lossf(p, gz, 1.0)
    res["total"].backward()
    for k, t_ in leaves(p):
        if t_.dim() == 4:
            assert float(t_.grad[..., :10].abs().max()) == 0.0, f"gradient in weight-0 cells: {k}"
            assert float(t_.grad[..., [0, -1], :].abs().max()) == 0.0, f"gradient in weight-0 rows: {k}"
            if k in ("sem", "cam_aux", "radar_aux", "hm", "v_hat"):
                assert float(t_.grad.abs().max()) > 0, f"no gradient at all: {k}"
    assert lossf.task.s.grad is not None and float(lossf.task.s.grad.abs().min()) > 0, "gradient reaches every s_g"

    span = grid.x_max - grid.x_min
    grid2 = replace(grid, x_max=grid.x_max + span)

    def doubled(g):
        o = {}
        for k, v in g.items():
            if k == "meta":
                bx2 = [dict(b, center=b["center"] + np.array([span, 0, 0]), corners=b["corners"] + [span, 0])
                       for b in v["boxes"]]
                o[k] = dict(v, boxes=v["boxes"] + bx2)
            else:
                o[k] = np.concatenate([v, v], axis=v.ndim - 2)
        return o
    p1 = leaf_preds(X, 2)
    p2 = {dt: {k: ({kk: torch.cat([vv, vv], 2) for kk, vv in v.items()} if isinstance(v, dict) else
                   torch.cat([v, v], 2) if v.dim() == 4 else torch.cat([v, v], 1)) for k, v in q.items()}
          for dt, q in p1.items()}
    with torch.no_grad():
        r1 = BEVLoss(cfg, grid)(p1, gz, 1.0)
        r2 = BEVLoss(cfg, grid2)(p2, {dt: [doubled(gz[dt][0])] for dt in gz}, 1.0)
    assert set(r1) == set(r2)
    worst = max(abs(float(r1[k]) - float(r2[k])) / max(1.0, abs(float(r1[k]))) for k in r1)
    assert worst < 1e-4, [(k, float(r1[k]), float(r2[k])) for k in r1 if abs(float(r1[k]) - float(r2[k])) > 1e-4]

    # Kendall weights stay inside the clamp under strong pressure either way; fixed weights; dt weights and warm-up
    tw = TaskWeights()
    opt = torch.optim.SGD(tw.parameters(), lr=5.0)
    for i in range(40):
        big = 1e3 if i < 20 else 1e-6
        tot, s = tw({g: torch.tensor(big) for g in TASK_GROUPS})
        opt.zero_grad()
        tot.backward()
        opt.step()
        v = tw.values().detach()
        assert float(v.min()) >= -3.0 and float(v.max()) <= 1.0
    assert abs(float(tw.values().detach()[0]) + 3.0) < 1e-6, "pushed to the lower bound"
    tf, _ = TaskWeights(mode="fixed", fixed=(1, 2, 3, 4))({"areas": torch.tensor(1.0), "motion": torch.tensor(1.0)})
    assert float(tf) == 5.0
    assert dt_weight(0.0, 0.0) == 1.0 and dt_weight(0.5, 0.1) == 0.0 and abs(dt_weight(-0.5, 0.3) - 0.25) < 1e-9
    assert dt_weight(0.46, 0.5) == 0.5 and dt_weight(-1.0, 1.0) == 0.25

    # overfit one synthetic sample: a free state, ONE shared mover, the semantic, object and both auxiliary heads as
    # TimedHeads at dt = 0, +-0.5 s; ~100 Adam steps; the total is finite and decreases
    torch.manual_seed(0)
    C = 16
    state = nn.Parameter(0.1 * torch.randn(1, C, X, Y))
    mover = TimeMover(C, grid)
    sem = TimedHead(SemanticHead(C, len(spec)), C, mover)
    obj = TimedHead(ObjectHead(C, NUSC_GROUPS, hidden=16), C, mover)                 # small: CPU test
    cam = TimedHead(SemanticHead(C, cam_n), C, mover)
    rad = TimedHead(SemanticHead(C, rad_n), C, mover)
    lossf = BEVLoss(cfg, grid)
    params = [state] + list(lossf.parameters())
    for m in (sem, obj, cam, rad):
        params += [q for q in m.parameters() if all(q is not r_ for r_ in params)]
    opt = torch.optim.Adam(params, lr=1e-2)
    hist = []
    for it in range(100):
        preds = {}
        for dt in dts:
            y, aux = sem(state, dt, scene.v, scene.omega, return_aux=True)
            q = {"sem": y, "mass": aux["mass"], "obj": obj(state, dt, scene.v, scene.omega),
                 "cam_aux": cam(state, dt, scene.v, scene.omega), "radar_aux": rad(state, dt, scene.v, scene.omega)}
            if dt == 0.0:
                q["v_hat"] = aux["v_hat"]
            preds[dt] = q
        res = lossf(preds, gts, 1.0)
        assert torch.isfinite(res["total"]), it
        raw = sum(res["group." + g].item() for g in TASK_GROUPS) + res["aux"].item()
        hist.append((res["total"].item(), raw))
        opt.zero_grad()
        res["total"].backward()
        torch.nn.utils.clip_grad_norm_(params, 10.0)
        opt.step()
    (t_first, r_first), (t_last, r_last) = hist[0], hist[-1]
    assert t_last < t_first and r_last < 0.7 * r_first, hist[::10]
    # an unobserved box (no lidar / radar point): not a heatmap positive, no gradient at its peak, footprint weight 0
    gu = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in gts[0.0][0].items()}
    bx0 = gu["meta"]["boxes"][0]
    gu["meta"] = dict(gu["meta"], boxes=[dict(bx0, observed=False)] + gu["meta"]["boxes"][1:])
    fp = gu["instance"] == 0
    gu["ignore"] = fp.astype(np.float32)
    gu[bx0["group"]] = np.where(fp, 0.0, gu[bx0["group"]]).astype(np.float32)
    Tu = build_layer_targets([gu], cfg)
    assert float(Tu["w_obj"][0, 0][torch.as_tensor(fp)].abs().max()) == 0.0, "unobserved footprint must be ignored"
    hm_u = torch.zeros(1, 3, X, Y, requires_grad=True)
    lo_u, tt_u = object_head_terms({"hm": hm_u, **{k: torch.zeros(1, c, X, Y) for k, c in (
        ("off", 2), ("size", 2), ("rot", 2), ("rad", 1), ("vel", 2))}}, Tu["boxes"], Tu["w_obj"], grid)
    lo_u["total"].backward()
    hb0 = head_box(bx0)
    ix, iy = int((hb0["cx"] - grid.x_min) // grid.res), int((hb0["cy"] - grid.y_min) // grid.res)
    gi = [g_ for g_, _ in NUSC_GROUPS].index(bx0["group"])
    assert float(tt_u["hm"][0, gi, ix, iy]) < 1.0 and float(hm_u.grad[0, gi, ix, iy].abs()) == 0.0, "unobserved peak"
    # dt weights: every nominal |dt| needs a listed weight; [[0, 1]] alone (the old no-time control) gave dt = +-0.5 the weight 1
    assert [dt_weight(d) for d in (0.0, 0.5, -0.5, 1.0)] == [1.0, 0.5, 0.5, 0.25]
    assert dt_weight(0.5, cfg=LossConfig(dt_weights=((0.0, 1.0), (0.5, 0.0)))) == 0.0
    try:
        dt_weight(0.5, cfg=LossConfig(dt_weights=((0.0, 1.0),)))
        raise AssertionError("an |dt| without a listed weight must be refused")
    except ValueError:
        pass
    print(f"losses selftest ok ({time.time() - t0:.0f} s): soft BCE = analytic {exp:.5f}, tolerant Dice 0 at a 1-cell "
          f"shift / {d3:.2f} at 3 cells, Huber 0 for the exact field, weight-0 cells: zero gradient (all heads), "
          f"doubled map: every term unchanged (max rel. diff {worst:.1e}), Kendall s in [-3, 1], dt weights + warm-up, "
          f"overfit 100 steps: total {t_first:.3f} -> {t_last:.3f}, unweighted sum {r_first:.3f} -> {r_last:.3f}; "
          f"{len(res)} terms; s = " + ", ".join(f"{g} {float(res['s.' + g]):+.2f}" for g in TASK_GROUPS))


if __name__ == "__main__":
    _selftest_fusion_area()
    _selftest()
