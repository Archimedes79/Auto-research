"""torch.compile feasibility on CPU: dynamo graph breaks (backend eager, no inductor, no GPU) of the t20 forward and of its parts, static shapes, small grid, B = 1.
usage: compile_explain.py"""
import collections
import re
import sys
import time

import common as C
import torch
import torch._dynamo as dynamo

cfg, model, loss, batch, _ = C.make(B=1, n_boxes=15, grid_kwargs={"x_max": 24.0, "y_min": -12.0, "y_max": 12.0})
from trainkit.util import detach_tree  # noqa: E402

dts = (-0.5, 0.0, 0.5)
gts = C.T.sample_targets(batch["gt"])
ego = C.T.batch_ego(batch["gap_steps"], dts, "cpu")
# a warm state
with torch.no_grad():
    state = model(batch, dts=(0.0,), ego=None)["state"]
state = detach_tree(state)
step = (1 / 12, 8.0, 0.05)


def summarize(name, fn, *a, **k):
    dynamo.reset()
    t0 = time.time()
    try:
        ex = dynamo.explain(fn)(*a, **k)
    except Exception as e:                                   # noqa: BLE001
        print(f"[{name}] explain failed: {type(e).__name__}: {str(e)[:300]}")
        return None
    reasons = collections.Counter()
    for br in ex.break_reasons:
        r = br.reason.split("\n")[0][:110]
        loc = ""
        if br.user_stack:
            fr = br.user_stack[-1]
            loc = f"{fr.filename.split('/')[-1]}:{fr.lineno}"
        reasons[(r, loc)] += 1
    print(f"[{name}] graphs {ex.graph_count}, graph breaks {ex.graph_break_count}, ops in graphs {ex.op_count}  ({time.time() - t0:.0f} s)")
    for (r, loc), n in reasons.most_common(12):
        print(f"      x{n}  {loc:28s} {r}")
    return ex


# 1 the whole forward (batch dict, 3 dts, ego, state, step)
summarize("model.forward (all)", model.forward, batch, dts=dts, state=state, step=step, ego=ego)
# 2 parts on their real inputs
with torch.no_grad():
    from model import Precompute  # noqa: F401
    b2 = dict(batch)
cam_frames = model.camera_bevs(batch)
summarize("camera_bevs (lift x per sample, dual)", model.camera_bevs, batch)
v_levels = None
steps_ = [[tuple(torch.as_tensor(t, dtype=cam_frames.dtype) for t in s) for s in f] for f in batch["steps"]]
from motion_block import block_inputs  # noqa: E402
summarize("motion_block", model.motion_block, *block_inputs(cam_frames, steps_, batch["radar"], model.grid))
summarize("radar encoder", model.radar, batch["radar"], 1, with_state=True)
cam, _ = model._branch_temporal(model.cam_temporal, cam_frames, steps_, state.get("cam"), step, None)
rad, _ = model.radar(batch["radar"], 1, with_state=True)[0], None
S = model.trunk.features(model.fusion(cam, rad)) if model.fusion is not None else model.trunk.features(cam, rad)
S = model.head_reduce(S) if model.head_reduce is not None else S
summarize("trunk.features(fusion(...))", lambda c, r: model.trunk.features(model.fusion(c, r)), cam, rad)
v = torch.full((1,), 8.0)
w = torch.full((1,), 0.05)
vf = torch.zeros(1, 2, model.grid.nx, model.grid.ny)
summarize("sem_head (mover + splat + head), one dt", lambda s: model.sem_head(s, 0.5, v, w, return_aux=True, v_field=vf), S)
summarize("splat.splat_bev only", lambda s: __import__("splat").splat_bev(s, model.grid, 0.5, v, w, head=lambda _s: vf), S)
# 3 the loss
res = model(batch, dts=dts, state=state, step=step, ego=ego)
preds = {dt: {k: ({a: b.float() for a, b in vv.items()} if isinstance(vv, dict) else vv.float()) for k, vv in p.items()} for dt, p in res["out"].items()}
summarize("loss (BEVLoss.forward)", loss, preds, gts, 0.5)
import losses  # noqa: E402
T0 = losses.build_layer_targets(gts[0.0], loss.cfg, preds[0.0].get("mass"), 0.0, "cpu")
summarize("map_terms (sem, one dt)", losses.map_terms, preds[0.0]["sem"], T0, loss.spec, {"area": T0["w_map"], "line": T0["w_lines"], "object": T0["w_obj"]}, loss.cfg, 0.5)
summarize("object_head_terms (one dt)", losses.object_head_terms, preds[0.0]["obj"], T0["boxes"], T0["w_obj"], loss.grid, loss.groups, loss.cfg.w_obj_vel, loss.cfg.max_obj,
          loss.cfg.w_obj_mov, hygiene=loss.cfg.hygiene_2026_11_02)
summarize("motion_term (one level)", losses.motion_term, preds[0.0]["v_hat"], T0, loss.cfg)
