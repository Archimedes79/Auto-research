"""Contract test of every kind of every network slot (training/architecture.md) and of the main switches (camera_off, cam_adapter,
head_channels, trunk_detail; the lane branch and the refiner have their own tests: test_lane_branch_chain.py, studies/early/box_refine.py,
because they need pre-trained weights or GT boxes): each kind is built on a
small grid inside the cached chain (CachedBEVFusionModel), run on a synthetic cached batch through the precompute step, and must
(1) refuse no valid combination and refuse the invalid ones in model.check_config, (2) pass interface.check_inputs / check_outputs,
(3) give finite outputs including the dense boxes and the soft line map, (4) send a gradient to every trainable parameter on the
second call of a clip (the first fills the memory: the radar branch has no past frames, so its memory blocks read nothing before),
and (5) carry a temporal state into the next call. A new kind is a builder entry in model.py plus its line in KINDS below.
Run (from the bev_fusion folder): wsl ~/miniconda3/envs/gcar/bin/python tests/test_slots.py   (CPU, about 1-2 min, no data, no GPU)"""
import dataclasses
import math
import os
import sys
import time

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "training"))
from bevfusion_camera import DEPTHS, bevfusion_ida  # noqa: E402
from camera_encoder import nuscenes_front_ida  # noqa: E402
from feature_cache import FEAT_HW, N_CTX, N_PROB, CachedBEVFusionModel  # noqa: E402
from interface import RADAR_FIELDS, check_inputs, check_outputs  # noqa: E402
from lift_pd import _front_camera  # noqa: E402
from model import ModelConfig, check_config  # noqa: E402
from nuscenes_dataset import DST_HW  # noqa: E402
from precompute import MAPS, Precompute  # noqa: E402

GRID = {"x_max": 16.0, "y_min": -8.0, "y_max": 8.0}             # 32 x 32 cells
BASE = dict(pretrained="", trunk_pretrained="", grid_kwargs=GRID, oversample=1, c_cam=16, c_rad=8, c_fuse=24, trunk_width=0.25,
            p_drop=0.0, dense_outputs=True)                  # no modality dropout (it would zero a branch at random)
KINDS = [("camera_features", "dualbev"), ("camera_features", "bevfusion"), ("camera_features", "dual"),
         ("camera_off", True), ("cam_adapter", True), ("head_channels", 16), ("trunk_detail", 8),
         ("lift", "pd"), ("lift", "p"), ("lift", "s"),
         ("cam_collapse", "zconv"), ("cam_collapse", "zsum"),
         ("cam_temporal", "window_memory"), ("cam_temporal", "ego_gru"), ("cam_temporal", "none"),
         ("rad_temporal", "none"), ("rad_temporal", "window_memory"), ("rad_temporal", "ego_gru"),
         ("radar_kind", "raster"), ("radar_kind", "pillars"), ("radar_kind", "off"),
         ("fusion", "concat"), ("fusion", "gate"), ("fusion", "cmx"),
         ("trunk_kind", "dualbev"), ("trunk_kind", "second"),
         ("object_direction", True), ("object_moving", True), ("motion_block", True)]
# parts a kind builds but cannot train, with the reason (a finding, not hidden): the radar branch has no past frames, so the camera's
# window block reads no window (relevance, temperature) and its velocity head reaches the output only through the memory write of a
# later call; the measured Doppler is the radar memory's own static gate (rad_static_gate, checked by gate_check below)
KNOWN_UNUSED = {("rad_temporal", "window_memory"): ("rad_temporal.log_tau", "rad_temporal.head.", "rad_temporal.relevance."),
                ("radar_kind", "off"): ("radar.", "aux_rad.net.0.weight", "aux_rad.net.1.weight", "trunk.fuser.pr."),   # the study arm keeps the radar
                # block's weights (same checkpoint layout), never runs it; the radar aux head sees a zero radar BEV
                ("camera_off", True): ("cam_reduce.", "cam_temporal.head.net.0.weight", "cam_temporal.relevance.weight"),
                ("radar_kind", "delta_slots"): ("radar.decoder.",)}   # trained by the reconstruction side loss in train.py, not the outputs
                # camera_off: the camera BEV is zero, its encoder never runs and the weights that see only that zero input get none
COMBOS = [{"camera_features": "dual", "cam_temporal": "ego_gru"}, {"fusion": "cmx", "radar_kind": "off"},
          {"rad_temporal": "window_memory", "radar_kind": "pillars"}, {"camera_off": True, "fusion": "gate"},
          {"camera_features": "bevfusion", "cam_collapse": "zsum", "trunk_kind": "second"}]
COMBOS += [{"radar_kind": "delta_slots", "c_rad": 64}, {"radar_kind": "delta_slots", "c_rad": 64, "rad_temporal": "window_memory"},
           {"rad_temporal": "window_memory", "rad_static_gate": True}, {"rad_temporal": "ego_gru", "rad_static_gate": True}]
INVALID = [dict(rad_static_gate=True), dict(rad_static_gate=True, rad_temporal="window_memory", radar_kind="off"), dict(rad_static_vr=0.0),
           dict(camera_features="dual", lift="s"), dict(camera_features="bevfusion", lift="s"), dict(lift="xx"),
           dict(rad_temporal="xx"), dict(lane_branch="res4", camera_off=True), dict(trunk_kind="cmx")]

COMBOS += [{"motion_block": True, "motion_strides": (4, 2)}, {"motion_block": True, "box_vel_from_field": True}, {"motion_block": True, "box_vel_from_field": True, "object_moving": True}]   # t15
COMBOS += [{"camera_features": "dual", "radar_kind": "delta_slots", "c_rad": 64, "rad_temporal": "window_memory", "rad_static_gate": True,
            "object_direction": True, "object_moving": True, "motion_block": True, "box_vel_from_field": True}]   # t16: the full network (configs/t16_full.json)
INVALID += [dict(box_vel_from_field=True), dict(motion_block=True, radar_kind="off"), dict(motion_block=True, camera_off=True),
            dict(motion_block=True, motion_strides=(1, 2)), dict(motion_block=True, motion_strides=(4, 3))]


def batch_for(cfg, B=2, Fn=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    K, T, _ = _front_camera(torch.float32)
    K = K.clone()
    K[:2] *= 2                                                   # the synthetic camera at the original 1600 x 900 resolution
    bevf = cfg.camera_features == "bevfusion"                    # the cached BEVFusion features: own depth bins, stride 8, own crop
    n_d, hw_f = (len(DEPTHS), (32, 88)) if bevf else (N_PROB, FEAT_HW)
    prob = torch.rand(B, Fn, n_d, *hw_f, generator=g)
    rig = torch.eye(4).repeat(5, 1, 1)                           # five radars as on the car: front, two front corners, two rear
    for i, (x, y, yaw) in enumerate([(3.4, 0.0, 0.0), (2.4, 0.8, 1.04), (2.4, -0.8, -1.04), (-1.0, 0.9, 2.1), (-1.0, -0.9, -2.1)]):
        rig[i, :2, :2] = torch.tensor([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        rig[i, :2, 3] = torch.tensor([x, y])
    b = {"K": K.expand(B, Fn, 3, 3).clone(), "T_cam_car": T.expand(B, Fn, 4, 4).clone(), "radar_poses": rig.expand(B, 5, 4, 4).clone(),
         "ida": bevfusion_ida() if bevf else nuscenes_front_ida(dst_hw=DST_HW),
         "hw": DST_HW, "prob": prob / prob.sum(2, keepdim=True), "ctx": torch.randn(B, Fn, N_CTX, *hw_f, generator=g),
         "v": torch.full((B,), 8.0), "omega": torch.full((B,), 0.05),
         "steps": [[]] + [[(torch.full((B,), 1 / 12), torch.full((B,), 8.0), torch.full((B,), 0.05))] * k for k in range(1, Fn)],
         "radar": dict({f: torch.randn(40, generator=g) for f in RADAR_FIELDS},          # the contract's detection fields
                       x=torch.rand(40, generator=g) * 15, y=torch.rand(40, generator=g) * 14 - 7, r=torch.rand(40, generator=g) * 15,
                       dt=torch.rand(40, generator=g) * 0.5, b=torch.arange(40) % B)}
    if cfg.camera_features == "dual":
        p2 = torch.rand(B, Fn, len(DEPTHS), 32, 88, generator=g)
        b.update(prob2=p2 / p2.sum(2, keepdim=True), ctx2=torch.randn(B, Fn, N_CTX, 32, 88, generator=g), ida2=bevfusion_ida())
    return b


def one(name, value, extra=None):
    cfg = ModelConfig(**dict(BASE, **{name: value}, **(extra or {})))
    torch.manual_seed(0)
    m = CachedBEVFusionModel(cfg).train()
    with torch.no_grad():                                        # zero-start branches (zero last layer) cut the gradient at init by
        for q in m.parameters():                                 # design: give them small values, so the test checks the path
            if q.requires_grad and q.abs().max() == 0:
                q.normal_(0, 1e-2)
    batch = batch_for(cfg)
    errs = check_inputs(batch)
    assert not errs, (name, value, errs)
    maps = Precompute.from_model(m)(dict(batch))                 # the precompute step on the model's own grid: every map, in the contract
    assert not check_inputs(maps), (name, value, check_inputs(maps))
    assert all(maps[k].shape == (2, 1, m.grid.nx, m.grid.ny) and 0 <= maps[k].min() and maps[k].max() <= 1 for k in MAPS), (name, value)
    dts = (0.0, 0.5)
    with torch.no_grad():                                        # the first call of a clip fills the memory (radar has no past frames:
        state = m(batch, dts=(0.0,))["state"]                    # its memory blocks read nothing before it)
    res = m(batch, dts=dts, state=state, step=(1 / 12, 8.0, 0.05))
    errs = check_outputs(res, dts)
    assert not errs, (name, value, errs)
    p0 = res["out"][0.0]
    assert "boxes" in p0 and "lines" in p0 and p0["lines"].shape[1] == len(m._dist_idx), (name, value)
    total = res["features"].sum() * 0
    for dt, p in res["out"].items():
        for k, v in p.items():
            for t in (v.values() if isinstance(v, dict) else [v]):
                if torch.is_tensor(t) and t.is_floating_point():
                    assert torch.isfinite(t).all(), (name, value, dt, k)
                    if k not in ("boxes", "lines"):              # the gradient through what the loss reads (the dense outputs:
                        total = total + (t.float() * torch.randn_like(t.float())).mean()   # extract.py's own test)
    total.backward()
    dead = [n for n, q in m.named_parameters() if q.requires_grad
            and (q.grad is None or not torch.isfinite(q.grad).all() or q.grad.abs().max() == 0)]
    known = tuple(k for kv in [(name, value)] + list((extra or {}).items()) for k in KNOWN_UNUSED.get(kv, ()))
    unexpected = [n for n in dead if not n.startswith(known)]
    assert not unexpected, (name, value, extra, "no gradient:", unexpected[:6], len(unexpected))
    if not extra:                                                # a combination may legitimately train a part the single kind cannot
        names = [n for n, _ in m.named_parameters()]
        assert all(any(n.startswith(k) for n in dead) for k in known if any(p.startswith(k) for p in names)), \
            (name, value, "a known-unused part now trains: update KNOWN_UNUSED", dead)
    # the dense outputs are the planner's door into the network: a loss on them must reach every parameter that a loss on the maps they
    # are built from reaches (heat map -> peak weight, box parameters; distance channels -> soft line map)
    p3 = m(batch, dts=dts, state=state, step=(1 / 12, 8.0, 0.05))["out"][0.0]
    params = [q for q in m.parameters() if q.requires_grad]

    def reached(*tensors):
        s = sum((t.float() * torch.randn_like(t.float())).mean() for t in tensors)
        gr = torch.autograd.grad(s, params, allow_unused=True, retain_graph=True)
        return {n for (n, q), g in zip(m.named_parameters(), gr) if q.requires_grad and g is not None and torch.isfinite(g).all() and g.abs().max() > 0}
    from_maps = reached(*[v for k, v in p3["obj"].items() if k in ("hm", "off", "size", "rot", "vel")], p3["sem"][:, m._dist_idx])
    from_dense = reached(*[v for v in p3["boxes"].values() if torch.is_tensor(v) and v.is_floating_point()], p3["lines"])
    lost = sorted(from_maps - from_dense)
    assert from_dense and not lost, (name, value, extra, "the dense outputs lose the gradient of", lost[:6], len(lost), "of", len(from_maps))
    with torch.no_grad():                                        # a second call carries the temporal state
        res2 = m(batch, dts=(0.0,), state=res["state"], step=(1 / 12, 8.0, 0.05))
    assert not check_outputs(res2, (0.0,)), (name, value)
    return sum(q.numel() for q in m.parameters() if q.requires_grad)


def gate_check():
    """the Doppler static gate with a known answer: the same detections, once static (vr 0) and once moving (vr 6 m/s); the radar memory's
    confidence after the first call. Without the gate both write the same, with it the moving ones write (almost) nothing and the static
    ones as much as without the gate."""
    w = {}
    for gate in (False, True):
        m = CachedBEVFusionModel(ModelConfig(**dict(BASE, rad_temporal="window_memory", rad_static_gate=gate))).eval()
        for name, vr in (("static", 0.0), ("moving", 6.0)):
            b = batch_for(m.cfg)
            b["radar"]["vr"] = torch.full_like(b["radar"]["vr"], vr)
            with torch.no_grad():
                w[gate, name] = float(m(b, dts=(0.0,))["state"]["rad"]["W"].sum())
    assert w[False, "static"] > 0 and abs(w[False, "static"] - w[False, "moving"]) < 1e-4 * w[False, "static"], w
    assert abs(w[True, "static"] - w[False, "static"]) < 1e-4 * w[False, "static"] and w[True, "moving"] < 1e-3 * w[True, "static"], w
    print(f"  ok  radar static gate: memory confidence static {w[True, 'static']:.2f} / moving {w[True, 'moving']:.4f} (without the gate "
          f"{w[False, 'static']:.2f} / {w[False, 'moving']:.2f})")


def main():
    t0 = time.time()
    gate_check()
    for bad in INVALID:
        try:
            check_config(ModelConfig(**dict(BASE, **bad)))
            raise AssertionError(f"check_config accepted {bad}")
        except ValueError:
            pass
    for name, value in KINDS:
        n = one(name, value)
        print(f"  ok  {name:16s} {str(value):14s} {n / 1e3:8.1f} k trainable parameters")
    for combo in COMBOS:
        (name, value), *rest = combo.items()
        one(name, value, dict(rest))
        print(f"  ok  combination {combo}")
    print(f"test_slots passed ({time.time() - t0:.0f} s): {len(KINDS)} slot kinds and {len(COMBOS)} combinations build, pass both "
          f"interfaces, give finite dense outputs, reach every trainable parameter and carry state; {len(INVALID)} invalid combinations refused")


if __name__ == "__main__":
    main()
