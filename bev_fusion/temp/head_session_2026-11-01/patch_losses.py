"""One-off patch of training/losses.py: move the inline fusion-area test into _selftest_fusion_area(), add the demo-scene helper,
the CLI flag and the refine note (exact-string, each anchor must match once)."""
import os

PATH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "training", "losses.py"))
src = open(PATH, encoding="utf-8").read()

# 1 the inline block -> a call
a = src.index("    # D38 (b) option fusion_area_only: the fused weights are x (camera x radar)")
b = src.index("    assert npos[\"off_blank\"]")
b = src.index("\n", b) + 1
assert src.count("    # D38 (b) option fusion_area_only:") == 1
src = src[:a] + "    _selftest_fusion_area()                           # D38 (b): `python training/losses.py --fusion-area` runs it alone\n" + src[b:]

# 2 helper + standalone self-test before _selftest
HELPER = '''def _demo_scene(sw):
    """the synthetic sample of the self-tests: one moving car, a parked car, a parked truck, a walking pedestrian"""
    B_ = sw.Box
    return sw.Scene([B_(20.0, -1.75, 4.5, 1.8, 1.5, 0.0, 8.0, 0.0, "car", 10.0),
                     B_(28.0, -5.5, 4.5, 1.8, 1.5, 0.0, 0.0, 0.0, "car", 10.0),
                     B_(38.0, 3.5, 8.0, 2.5, 3.0, 0.05, 0.0, 0.0, "truck", 20.0),
                     B_(15.0, 8.5, 0.6, 0.6, 1.8, math.pi / 2, 0.0, 1.2, "pedestrian", -5.0)],
                    lanes=[-5.25, -1.75, 1.75, 5.25], v=10.0, omega=0.1)


def _selftest_fusion_area():
    """LossConfig.fusion_area_only (D38 (b)), CPU, no optimiser step (a backward pass on leaf tensors only; seconds):
    off by default and then every weight is positive outside camera x radar; on, the fused weights are x (camera x radar) and the aux
    weights stay; an object whose centre lies outside is ignored; through the whole BEVLoss at dt 0 and dt 0.5 (splat mass) the gradient
    of every fused output is exactly 0 outside the fusion area, the aux heads' gradients are identical to those with the option off."""
    import os
    import sys
    from dataclasses import replace
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
    import synthetic_world as sw
    from object_head import ObjectHead
    torch.set_num_threads(2)
    grid, cfg = Grid(), LossConfig()
    X, Y = grid.nx, grid.ny
    assert cfg.fusion_area_only is False, "the option is off by default: every stored run is reproduced"
    scene = _demo_scene(sw)
    gts = {dt: [synthetic_gt(scene.step(dt) if dt else scene, grid)] for dt in (0.0, 0.5)}
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


'''
anchor = "def _selftest():\n    import os\n    import sys\n    import time\n"
assert src.count(anchor) == 1
src = src.replace(anchor, HELPER + anchor, 1)

# 3 CLI
old = 'if __name__ == "__main__":\n    _selftest()\n'
assert src.count(old) == 1
src = src.replace(old, 'if __name__ == "__main__":\n    import sys\n    if "--fusion-area" in sys.argv:\n        _selftest_fusion_area()\n    else:\n        _selftest()\n', 1)

# 4 refine note
old = "    camera and not occluded, or radar), so every other box is ignored, not background.\"\"\""
assert src.count(old) == 1
src = src.replace(old, "    camera and not occluded, or radar), so every other box is ignored, not background. It does NOT follow\n"
                  "    LossConfig.fusion_area_only (startable.py refuses model.refine together with it).\"\"\"", 1)
open(PATH, "w", encoding="utf-8", newline="\n").write(src)
print("patched", PATH)
