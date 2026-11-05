"""usage: probe_off.py <code root containing bev_fusion/ and autoresearch/> <out.pt>
Builds the cached chain (option off) from a code tree, runs train-mode forward + backward on a synthetic batch (lane branch rgb, pair buffer, delta_slots, motion block), saves
state_dict keys/tensors at init, outputs, losses, grads."""
import dataclasses, os, sys, hashlib
root, out = sys.argv[1], sys.argv[2]
sys.path[:0] = [os.path.join(root, "bev_fusion", "training"), os.path.join(root, "bev_fusion", "tests"), os.path.join(root, "bev_fusion"), os.path.join(root, "autoresearch")]
import torch
torch.set_num_threads(2)
import model as M
from feature_cache import CachedBEVFusionModel
import test_slots as TS
print("model from", M.__file__)
res = {}
for label, extra in (("dual_pair_full", dict(camera_features="dual", radar_kind="delta_slots", c_rad=64, rad_temporal="window_memory", rad_static_gate=True, object_direction=True,
                                              object_moving=True, motion_block=True, box_vel_from_field=True, cam_buffer="pair", lane_branch="rgb")),
                     ("dualbev_plain", dict(camera_features="dualbev")), ("bevfusion_adapter", dict(camera_features="bevfusion")),
                     ("dual_window", dict(camera_features="dual", cam_temporal="ego_gru"))):
    cfg = M.ModelConfig(**dict(TS.BASE, **extra))
    torch.manual_seed(123)
    m = CachedBEVFusionModel(cfg).train()
    with torch.no_grad():
        for q in m.parameters():
            if q.requires_grad and q.abs().max() == 0:
                q.normal_(0, 1e-2)
    sd = {k: v.clone() for k, v in m.state_dict().items()}
    Fn = 2 if cfg.cam_buffer == "pair" else 3
    b = TS.batch_for(cfg, B=2, Fn=Fn)
    if cfg.lane_branch:
        b["lane_img"] = torch.randint(0, 256, (2, 3, *TS.DST_HW), dtype=torch.uint8, generator=torch.Generator().manual_seed(4))
    b["flip"] = torch.tensor([True, False])
    torch.manual_seed(5)
    r = m(b, dts=(0.0, 0.5), step=(1 / 12, 8.0, 0.05)) if cfg.cam_buffer == "recurrent" else m(b, dts=(0.0, 0.5))
    tot = r["features"].sum() * 0
    outs = {}
    for dt, p in r["out"].items():
        for k, v in p.items():
            for kk, t in (v.items() if isinstance(v, dict) else [("", v)]):
                if torch.is_tensor(t) and t.is_floating_point():
                    outs[f"{dt}/{k}/{kk}"] = t.detach().clone()
                    tot = tot + (t.float() * torch.linspace(-1, 1, t.numel()).view_as(t)).mean()
    tot.backward()
    grads = {n: q.grad.clone() for n, q in m.named_parameters() if q.grad is not None}
    res[label] = dict(sd=sd, outs=outs, grads=grads, keys=list(m.state_dict().keys()), pnames=[n for n, _ in m.named_parameters()], res_keys=sorted(r.keys()), feat=r["features"].detach().clone())
    print(label, "params", len(grads), "outs", len(outs), "res keys", sorted(r.keys()))
torch.save(res, out)
