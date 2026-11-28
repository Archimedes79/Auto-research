"""Tests of evaluation/bev_modules.py (its former self-test).

    python -P -m bev_fusion.tests.test_bev_modules      (from the problem folder, with its module path)
"""
from __future__ import annotations
import os
import tempfile

import numpy as np
import torch

from bev_fusion.evaluation.bev_modules import Recorder, average_response, chain_modules, changed_nothing, draw_panels, energy, frame_panels, mean_over_frames, rel_change, response_row, stat, tensors_of, velocity_rows, zero_cam, zero_doppler, zero_radar


def selftest():
    # numbers of tensors: known answers
    t = torch.zeros(2, 3, 4, 4)
    t[:, 0] = torch.randn(2, 4, 4)
    t[:, 1] = 1.0                                                                # constant channel: dead; channel 2 all zero: dead
    s = stat(t)
    assert s["shape"] == [2, 3, 4, 4] and s["finite"] and abs(s["dead_channels"] - 2 / 3) < 1e-6, s
    assert abs(s["zero_share"] - 1 / 3) < 1e-6 and s["absmax"] >= 1.0
    assert not stat(torch.tensor([[[[1.0, float("nan")]]]]))["finite"] and stat(torch.ones(5))["dead_channels"] is None
    nest = tensors_of({"a": (torch.ones(1, 2), torch.zeros(3, dtype=torch.int64)), "b": [torch.ones(2)]})
    assert set(nest) == {"a[0]", "b[0]"}, set(nest)
    assert abs(rel_change(torch.tensor([3.0, 4.0]), torch.tensor([3.0, 4.0])) ) < 1e-9 and abs(rel_change(torch.tensor([3.0, 4.0]), torch.zeros(2)) - 1.0) < 1e-6
    assert abs(rel_change(torch.tensor([3.0, 4.0]), torch.tensor([3.0, 0.0])) - 0.8) < 1e-6
    assert energy(torch.ones(1, 4, 5, 6)).shape == (5, 6) and abs(energy(torch.ones(1, 4, 5, 6))[0, 0] - 2.0) < 1e-6
    assert mean_over_frames([{"std": 1.0, "shape": [1], "finite": True}, {"std": 3.0, "shape": [2], "finite": False}]) == {"std": 2.0, "shape": [1], "finite": 0.5}
    assert average_response([{"a": {"x": 1.0, "y": None}}, {"a": {"x": 3.0, "y": None}}]) == {"a:x": 2.0, "a:y": None}
    # the velocity field on static and moving cells: planted offsets
    gv = torch.zeros(2, 6, 6)
    gv[0, :2] = 4.0                                                              # 12 moving cells at (4, 0) m/s
    und = torch.zeros(6, 6, dtype=torch.bool)
    und[5] = True                                                                # one row undefined: not counted
    field = gv + torch.tensor([0.3, 0.4])[:, None, None]                         # every cell is off by (0.3, 0.4): EPE 0.5
    row = velocity_rows("f", 7.0, {"v_hat": field}, gv, und)[0]
    assert row["static_cells"] == 18 and row["moving_cells"] == 12 and abs(row["static_mean_vx"] - 0.3) < 1e-6 and abs(row["static_mean_vy"] - 0.4) < 1e-6, row
    assert abs(row["static_epe"] - 0.5) < 1e-6 and abs(row["moving_epe"] - 0.5) < 1e-6 and abs(row["moving_gt_speed"] - 4.0) < 1e-6 and row["static_std_x"] < 1e-6 and row["ego_v_mps"] == 7.0
    assert velocity_rows("f", 0.0, {"v": torch.zeros(2, 3, 3)}, torch.zeros(2, 3, 3), torch.zeros(3, 3).bool())[0]["moving_epe"] is None, "no moving cell: None, not nan"
    # every perturbation changes exactly what it names (the cam_off bug of t3: a flag that changed nothing)
    batch = {"prob": torch.ones(1, 2), "ctx": torch.ones(1, 2), "lane_img": torch.ones(1, 3, dtype=torch.uint8), "lane_img_old": torch.ones(1, 2, 3, dtype=torch.uint8),
             "radar_hits": torch.ones(1, 3), "v": torch.ones(1),
             "radar": {"x": torch.arange(5.0), "vr": torch.ones(5), "y": torch.ones(5)}}
    snapshot = {k: (v.clone() if torch.is_tensor(v) else {kk: vv.clone() for kk, vv in v.items()}) for k, v in batch.items()}
    c, r, d = zero_cam(batch), zero_radar(batch), zero_doppler(batch)
    assert c["prob"].abs().sum() == 0 and c["ctx"].abs().sum() == 0 and c["lane_img"].sum() == 0 and c["lane_img_old"].sum() == 0 and c["radar"] is batch["radar"] and c["radar_hits"].sum() == 3 and c["v"].sum() == 1
    assert r["radar"]["x"].numel() == 0 and r["radar"]["vr"].numel() == 0 and r["radar_hits"].abs().sum() == 0 and r["prob"].sum() == 2 and r["lane_img"].sum() == 3 and r["lane_img_old"].sum() == 6
    assert d["radar"]["vr"].abs().sum() == 0 and d["radar"]["x"].sum() == 10 and d["radar"]["y"].sum() == 5 and d["prob"].sum() == 2
    assert all(torch.equal(batch[k], snapshot[k]) if torch.is_tensor(batch[k]) else all(torch.equal(batch[k][kk], snapshot[k][kk]) for kk in batch[k]) for k in batch), "the batch is not changed in place"
    # radar_per_frame (t25): the sets of radar_seq are perturbed too (the recursive unroll's radar branch reads those), the batch is not changed in place
    seq = {"radar": {"x": torch.arange(5.0), "vr": torch.ones(5)}, "radar_seq": [{"x": torch.arange(5.0), "vr": torch.ones(5)}, {"x": torch.arange(3.0), "vr": torch.ones(3)}]}
    rs, ds = zero_radar(seq), zero_doppler(seq)
    assert rs["radar"]["x"].numel() == 0 and len(rs["radar_seq"]) == 2 and all(s["x"].numel() == 0 and s["vr"].numel() == 0 for s in rs["radar_seq"])
    assert ds["radar"]["vr"].abs().sum() == 0 and all(s["vr"].abs().sum() == 0 for s in ds["radar_seq"]) and ds["radar_seq"][1]["x"].sum() == 3
    assert seq["radar_seq"][0]["vr"].sum() == 5 and seq["radar_seq"][1]["x"].numel() == 3 and seq["radar"]["x"].numel() == 5, "radar_seq is not changed in place"
    assert "radar_seq" not in zero_radar(batch) and "radar_seq" not in zero_doppler(batch), "a batch without radar_seq stays without"
    # response row: a stage that follows its input changes, one that does not stays
    base = {"radar": {"out": torch.ones(2, 3)}, "fusion": {"out": torch.ones(2, 3)}}
    new = {"radar": {"out": torch.zeros(2, 3)}, "fusion": {"out": torch.ones(2, 3)}}
    heads = {"sem": torch.ones(2), "obj": {"hm": torch.ones(2), "vel": torch.ones(2)}, "v_hat": None}
    rr = response_row(base, heads, new, {"sem": torch.ones(2), "obj": {"hm": torch.zeros(2), "vel": torch.ones(2)}, "v_hat": None})
    assert abs(rr["radar"]["out"] - 1.0) < 1e-6 and rr["fusion"]["out"] == 0.0 and rr["OUT"]["hm"] > 0.99 and rr["OUT"]["sem"] == 0.0 and rr["OUT"]["v_hat"] is None, rr
    # the recorder: forward hooks and the wrapped trunk.features; closing restores everything
    import torch.nn as nn

    class Trunk(nn.Module):
        def features(self, x):
            return x * 2

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.lift, self.radar, self.trunk = nn.Identity(), nn.Identity(), Trunk()

        def forward(self, x):
            return self.trunk.features(self.radar(self.lift(x)))
    m = Toy()
    plain = m.trunk.features
    rec = Recorder(m)
    y = m(torch.ones(1, 2, 3, 3))
    assert set(rec.store) == {"lift", "radar", "trunk"} and torch.equal(rec.store["trunk"]["out"], y) and y.sum() == 36, set(rec.store)
    snap = rec.copy()
    rec.clear()
    assert not rec.store and "out" in snap["lift"]
    rec.close()
    assert m.trunk.features == plain and "features" not in m.trunk.__dict__, "the trunk is as before (no instance attribute left behind)"
    assert chain_modules(m) == ["lift", "radar", "trunk"] and chain_modules(nn.Sequential(nn.Identity())) == ["0"], "the chain is the model's own children"
    m.lift2 = nn.Identity()                                                      # a dual camera's second lift is probed without a list that has to know it
    rec2 = Recorder(m)
    m(torch.ones(1, 2, 3, 3))
    m.lift2(torch.ones(1))
    assert "lift2" in rec2.store
    rec2.close()
    assert stat(torch.ones(1))["std"] == 0.0, "a single element has no spread (not nan)"
    t_cpu = torch.ones(1, 2, 2, 2)
    stored = tensors_of(t_cpu)["out"]
    t_cpu.mul_(5)
    assert float(stored.mean()) == 1.0, "a later in-place operation does not change the record"
    assert changed_nothing({"a": {"x": 0.0, "y": None}}) and not changed_nothing({"a": {"x": 0.0}, "OUT": {"sem": 0.2}})
    # pictures: the stage panels and the figure
    stages = {"fusion": {"out": torch.rand(1, 4, 6, 5)}}
    out0 = {"sem": torch.randn(1, 2, 6, 5), "obj": {"hm": torch.randn(1, 3, 6, 5), "mov": torch.randn(1, 3, 6, 5)}, "v_hat": torch.randn(1, 2, 6, 5),
            "v_levels": {"l0": torch.randn(1, 2, 6, 5)}}
    gt = {"drivable_area": np.zeros((6, 5)), "lane_divider": np.zeros((6, 5)), "vehicle": np.zeros((6, 5)), "flow": np.zeros((2, 6, 5))}
    panels = frame_panels(stages, torch.rand(1, 4, 6, 5), out0, gt, ["drivable_area", "lane_divider"])
    titles = [p[0] for p in panels]
    assert "fusion |feat|" in titles and "M level l0 |v|" in titles and "pred moving prob (hurdle)" in titles and "GT lane_divider" in titles, titles
    with tempfile.TemporaryDirectory() as d_:
        path = os.path.join(d_, "f.png")
        draw_panels(path, "toy", panels)
        assert os.path.getsize(path) > 2000
    print("bev_modules selftest ok: tensor numbers, dead channels, static / moving velocity cells, each perturbation changes only what it names, response rows, recorder, pictures")


if __name__ == "__main__":
    selftest()
