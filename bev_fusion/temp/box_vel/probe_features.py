"""t11 box velocity, steps 1-3: what the trained chain computes at the supervised GT box centres.

For n frames of a split, each run cold (like the stored t8 evaluation), with the run's own code snapshot (evaluation/bev_adapter
setup_paths + load_run: last checkpoint, EMA weights) and that snapshot's training loss (train.forward_loss, train_terms=False):
  per frame   every loss term at dt 0 (objhead.vel@+0.0 among them), the vel term of the zero predictor on the same boxes, the motion
              term and the motion term of v_hat = 0, the share of active (pre-activation > 0) hidden units per object-head branch over
              all cells and over the supervised centres, the number of hidden units on at >= 1 cell
  per box     (the boxes object_head_terms supervises at dt 0: observed, centre cell with w_obj > 0) group, target velocity, the trunk
              feature S, the object head's shared feature, the vel branch's hidden pre-activation, its output, v_hat at the centre cell
Memory: the datasets read a light devkit of n_scenes scenes per split (autogt/light_nusc.py, ~1 GB) instead of the full trainval
devkit (8+ GB); the frames are spread over those scenes. -> npz for probe_fit.py. Run (from bev_fusion/, CPU):
  OMP_NUM_THREADS=2 ~/miniconda3/envs/gcar/bin/python temp/box_vel/probe_features.py runs/t8_ctl_s0 val 300 temp/box_vel/out/t8_ctl_s0_val.npz [n_scenes 30]
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

BRANCHES = ("hm", "off", "size", "rot", "vel")


def pick(names, k):
    """k names spread over the list"""
    if k >= len(names):
        return list(names)
    return [names[i] for i in np.linspace(0, len(names) - 1, k).round().astype(int)]


def datasets(run, T, split, n_scenes):
    """the run's train / val datasets (train.make_datasets of the snapshot) on a light devkit of n_scenes scenes of `split`"""
    import nuscenes.nuscenes as nn_mod
    from autogt.light_nusc import light_nuscenes
    from evaluation.bev_adapter import freeze_scenes
    with open(os.path.join(run, "config.json")) as f:
        cfg = T.TrainConfig(**json.load(f))
    freeze_scenes(cfg)
    names = pick(cfg.val_scenes if split == "val" else cfg.train_scenes, n_scenes)
    light = light_nuscenes(cfg.data_root, cfg.version, scenes=names)
    orig = nn_mod.NuScenes
    nn_mod.NuScenes = lambda version, dataroot=None, verbose=False: light          # make_datasets imports it at call time
    try:
        cfg.train_scenes, cfg.val_scenes = (names, []) if split == "train" else ([], names)
        tr, va = T.make_datasets(cfg)
    finally:
        nn_mod.NuScenes = orig
    return (tr if split == "train" else va), names


def light_load(run, T, device):
    """the last checkpoint's EMA weights in the model (as evaluation/bev_adapter.load_run, without the optimiser and the EMA object:
    ~0.6 GB less) -> cfg, model (eval), loss, step, checkpoint path"""
    import glob
    import torch
    with open(os.path.join(run, "config.json")) as f:
        cfg = T.TrainConfig(**json.load(f))
    model, loss, _ = T.build(cfg)
    ckpt = sorted(glob.glob(os.path.join(run, "ckpt_*.pt")))[-1]
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    st = {k[6:]: v for k, v in ck["state"].items() if k.startswith("model.")}
    st.update(ck["ema"])                                 # EMA tensors over the state (the integer buffers come from the state)
    miss, extra = model.load_state_dict(st, strict=False)
    if miss or extra:
        raise ValueError(f"{ckpt}: missing {miss[:5]} ({len(miss)}), unexpected {extra[:5]} ({len(extra)})")
    loss.load_state_dict({k[5:]: v for k, v in ck["state"].items() if k.startswith("loss.")})
    step = int(ck["step"])
    del ck, st
    return cfg, model.to(device).eval(), loss.to(device), step, ckpt


def main(run, split, n, out_path, n_scenes=30, device="cpu"):
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    from evaluation.bev_adapter import _chain, setup_paths
    setup_paths(run)
    ns = _chain()
    T = ns.T
    import losses as L                                   # the run's snapshot (setup_paths put it first on sys.path)
    ds, scene_names = datasets(run, T, split, n_scenes)
    cfg, model, loss, step, ckpt = light_load(run, T, device)
    head, grid = model.obj_head.head, model.grid
    cap = {}

    def hook(name):
        def f(_m, _i, o):
            cap.setdefault(name, []).append(o.detach())
        return f
    hs = [head.shared.register_forward_hook(hook("shared"))]
    hs += [getattr(head, b)[0].register_forward_hook(hook(b)) for b in BRANCHES]
    idx = sorted(set(np.linspace(0, len(ds) - 1, min(n, len(ds))).round().astype(int).tolist()))
    rows, frames = {k: [] for k in ("group", "t_vel", "S", "x", "z_vel", "vel", "vhat", "frame")}, []
    print(f"{run} {split}: {len(scene_names)} scenes, {len(ds)} samples, {len(idx)} frames, checkpoint {os.path.basename(ckpt)}", flush=True)
    t0 = time.time()
    for fi, i in enumerate(idx):
        item = ds[i]
        batch = T.collate_cached([item]) if cfg.feature_cache else T.collate([item])
        cap.clear()
        with torch.no_grad():
            out, res, gts = T.forward_loss(model, loss, batch, 1.0, device, cfg.amp, cfg, train_terms=False)
        dts = sorted(res["out"])
        i0 = dts.index(0.0)                              # the object head runs once per dt, in this order
        p = res["out"][0.0]
        Tt = L.build_layer_targets(gts[0.0], loss.cfg, None, 0.0, device)
        _, t = L.object_head_terms(p["obj"], Tt["boxes"], Tt["w_obj"], grid, loss.groups, loss.cfg.w_obj_vel, loss.cfg.max_obj)
        m = t["mask"][0] > 0
        ind = t["ind"][0][m]
        S = res["features"][0].flatten(1)                # (C, X*Y)
        x = cap["shared"][i0][0].flatten(1)
        zs = {b: cap[b][i0][0].flatten(1) for b in BRANCHES}
        vel = p["obj"]["vel"][0].flatten(1)
        vhat = p["v_hat"][0].flatten(1)
        tv = t["vel"][0][m]
        zero_vel = loss.cfg.w_obj_vel * float(tv.abs().sum(-1).mean()) if len(tv) else 0.0   # head_loss l1: |ex| + |ey|, mean over boxes
        fr = {"token": item["token"], "n_box": int(m.sum()),
              "vel_term": float(out.get("objhead.vel@+0.0", float("nan"))), "vel_term_zero": zero_vel,
              "motion": float(out.get("motion@+0.0", float("nan"))),
              "motion_zero": float(L.motion_term(torch.zeros_like(p["v_hat"]), Tt, loss.cfg))}
        for b in BRANCHES:
            z = zs[b]
            fr[f"active_all_{b}"] = float((z > 0).float().mean())
            fr[f"units_alive_{b}"] = int(((z > 0).any(dim=1)).sum())          # hidden units on at >= 1 cell of this frame
            fr[f"active_ctr_{b}"] = float((z[:, ind] > 0).float().mean()) if len(ind) else float("nan")
        frames.append(fr)
        if len(ind):
            rows["group"].append(t["group"][0][m].cpu().numpy())
            rows["t_vel"].append(tv.cpu().numpy())
            rows["S"].append(S[:, ind].t().cpu().numpy().astype(np.float16))
            rows["x"].append(x[:, ind].t().cpu().numpy().astype(np.float16))
            rows["z_vel"].append(zs["vel"][:, ind].t().cpu().numpy().astype(np.float16))
            rows["vel"].append(vel[:, ind].t().cpu().numpy())
            rows["vhat"].append(vhat[:, ind].t().cpu().numpy())
            rows["frame"].append(np.full(len(ind), fi))
        if fi % 25 == 0:
            print(f"{fi + 1}/{len(idx)} frames, {(time.time() - t0) / (fi + 1):.2f} s/frame; vel term {fr['vel_term']:.3f} (zero "
                  f"predictor {fr['vel_term_zero']:.3f}), vel units alive {fr['units_alive_vel']}/64, off {fr['units_alive_off']}/64", flush=True)
    for h in hs:
        h.remove()
    arr = {k: np.concatenate(v) for k, v in rows.items() if v}
    for k in sorted(frames[0]):
        arr["frame_" + k] = np.array([f[k] for f in frames])
    arr["groups"] = np.array([g for g, _ in loss.groups])
    arr["scenes"] = np.array(scene_names)
    arr["step"] = np.array(step)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, **arr)
    print(f"wrote {out_path}: {len(frames)} frames, {len(arr.get('group', []))} boxes, step {step}", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], int(a[2]), a[3], int(a[4]) if len(a) > 4 else 30)
