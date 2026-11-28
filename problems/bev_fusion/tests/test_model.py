"""Tests of net/model/model.py (its former self-test).

    python -P -m bev_fusion.tests.test_model      (from the problem folder, with its module path)
"""
import torch

from bev_fusion.net import netpath
from bev_fusion.net.model.model import BEVFusionModel, ModelConfig
from bev_fusion.net.model.temporal_fusion import TemporalFusion
from bev_fusion.net.prep.targets import sample_targets


def _selftest():
    import os
    from bev_fusion.net.loss.losses import BEVLoss, sample_targets
    from bev_fusion.net.prep.nuscenes_dataset import FrontRadarDataset, collate
    torch.manual_seed(0)
    os.chdir(netpath.SITE)                                                   # the site folder: the default weights are named relative to it (models/pretrained)
    assert BEVFusionModel(ModelConfig(pretrained="")).fusion.p_drop == 0.2
    for off in (dict(camera_off=True), dict(radar_kind="off")):    # one sensor off: no modality dropout (it would blank the sample)
        assert BEVFusionModel(ModelConfig(pretrained="", **off)).fusion.p_drop == 0.0, off
    try:
        BEVFusionModel(ModelConfig(pretrained="", camera_off=True, lane_branch="res4"))
        raise AssertionError("lane_branch with camera_off was accepted")
    except ValueError:
        pass
    ds = FrontRadarDataset(netpath.nuscenes_root(mini=True), "v1.0-mini", n_cam=3, cam_stride=2, with_gt=True)
    batch = collate([ds[10], ds[200]])                       # mid-scene keyframes: GT of k-1, k, k+1 exist
    gts = sample_targets(batch["gt"])
    dts = tuple(sorted(gts))
    assert dts == (-0.5, 0.0, 0.5) and all(len(g) == 2 for g in gts.values()), {k: len(g) for k, g in gts.items()}
    model = BEVFusionModel()
    model.train()
    res = model(batch, dts=dts)
    o = res["out"][0.0]
    X, Y = model.grid.nx, model.grid.ny
    assert o["sem"].shape == (2, len(model.spec), X, Y) and o["obj"]["hm"].shape == (2, 3, X, Y), o["sem"].shape
    assert set(res["out"]) == set(dts) and "v_hat" in o and "cam_aux" in o and "v_hat" not in res["out"][0.5]
    crit = BEVLoss(grid=model.grid)
    L = crit(res["out"], gts, step_frac=1.0)
    assert torch.isfinite(L["total"]), {k: float(v) for k, v in L.items() if torch.is_tensor(v)}
    L["total"].backward()
    assert model.camera.depth_net.depth_conv[4].weight.grad is None, "frozen camera heads get no gradient"
    for name, mod in (("reduce", model.cam_reduce), ("camera temporal", model.cam_temporal), ("radar", model.radar),
                      ("fusion", model.fusion), ("trunk", model.trunk), ("time mover", model.mover),
                      ("semantic head", model.sem_head), ("object head", model.obj_head)):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in mod.parameters()), f"no gradient: {name}"
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    groups = ", ".join(f"{k[6:]} {float(v.detach()):.2f}" for k, v in L.items() if k.startswith("group."))
    # a second call with the returned state and the step since (sequence mode)
    with torch.no_grad():
        model.eval()
        res2 = model(batch, state=res["state"], step=(torch.full((2,), 1 / 12), torch.tensor([5.0, 8.0]),
                                                      torch.tensor([0.0, 0.1])))
    assert res2["out"][0.0]["sem"].shape == o["sem"].shape and "cam_aux" not in res2["out"][0.0]
    # the camera memory writes only where the camera observes (obs_mass = LiftPD.visibility), nothing outside the view
    if isinstance(model.cam_temporal, TemporalFusion):
        with torch.no_grad():
            vis = model.camera_visibility(batch, o["sem"])                                  # (B, 1, X, Y)
            st = model(batch)["state"]["cam"]
            from bev_fusion.net.model.temporal_fusion import read_memory as _rm
            _, conf = _rm(st["M"], st["W"], st["pose"], model.cam_temporal.ego_pts, vis.shape[-2], vis.shape[-1],
                          model.cam_temporal.mem_grid)
        assert 0.05 < float(vis.mean()) < 0.95, f"visibility covers part of the grid ({float(vis.mean()):.2f})"
        assert float(conf[vis < 0.5].max()) < 1e-4 and float(conf[vis > 0.5].mean()) > 0.1, \
            "memory written inside the camera view only"
    # per-sample true keyframe gap: the ego path equals the scalar path for shared values; a 0.45 s gap moves 0.9 x
    with torch.no_grad():
        vv, ww = batch["v"].float(), batch["omega"].float()
        base = model(batch, dts=(0.5,))["out"][0.5]
        same = model(batch, dts=(0.5,), ego={0.5: (torch.full((2,), 0.5), vv, ww)})["out"][0.5]
        assert torch.allclose(same["sem"], base["sem"], atol=1e-5) and torch.allclose(same["mass"], base["mass"])
        v10, w0 = torch.full((2,), 10.0), torch.zeros(2)
        unk = lambda gaps: (model(batch, dts=(0.5,), ego={0.5: (torch.tensor(gaps), v10, w0)})["out"][0.5]["mass"]
                            < 0.5).float().mean((1, 2, 3))
        u_ref, u_gap = unk([0.5, 0.5]), unk([0.5, 0.45])
    assert abs(float(u_gap[0] - u_ref[0])) < 1e-6 and abs(float(u_gap[1] / u_ref[1]) - 0.9) < 0.05, (u_ref, u_gap)
    del model, res, res2
    # every swappable block (shapes only: no checkpoint, no gradient)
    swaps = [{"lift": "p"}, {"cam_temporal": "none"}, {"rad_temporal": "window_memory"}, {"radar_kind": "delta_slots", "c_rad": 64}, {"radar_kind": "off"},
             {"camera_off": True}, {"cam_buffer": "recurrent"}]
    tried = []
    for kw in swaps:
        m = BEVFusionModel(ModelConfig(pretrained="", **kw)).eval()
        with torch.no_grad():
            r = m(batch, dts=(0.0, 0.5))
        assert r["out"][0.5]["sem"].shape == o["sem"].shape, kw
        tried.append("=".join(str(x) for x in next(iter(kw.items()))))
        del m, r
    print(f"model selftest ok: two mini samples through the whole chain at dt {dts}, BEVLoss total "
          f"{float(L['total'].detach()):.2f} ({groups}); {n_train / 1e6:.1f} M of {n_all / 1e6:.1f} M parameters trainable "
          f"(camera heads frozen, no gradient); gradients reach reduce, camera temporal, radar, fusion, trunk, mover, "
          f"both heads; sequence mode with state; per-sample keyframe gap (ego = scalar path; 0.45 s gap: unknown band "
          f"{float(u_gap[1] / u_ref[1]):.2f} x the 0.5 s band); swapped blocks run: {', '.join(tried)}")


if __name__ == "__main__":
    _selftest()
