"""Module probe: what does every module of the chain output, does it respond to its input, is it related to the GT? One run, a few val frames,
CPU or GPU (no checkpoint is changed, the network runs through the same `train.forward_loss` as the evaluation, cold memory per frame).

    python evaluation/bev_modules.py RUN [--frames 3] [--device cpu] [--split val] [--threads 2] [--out DIR]     (also: bev_suite.py RUN modules ...)
    python evaluation/bev_modules.py --selftest                                                                    (synthetic, no run needed)

Writes OUT (default RUN/eval/modules/):
    modules.json        per module output (every float tensor inside it, named by its path): shape, finite, mean, std, absmax, share of zeros, share of dead
                        channels (std over batch and cells < 1e-7); also "OUT dt0:<head output>", "radar_side:*", "features(S)" and, from a second (warm) frame
                        that reads what the first one wrote, "WARM cam_temporal / rad_temporal / fusion": the memories do something
    response.json       the response test: remove the camera (probability and context tensors), all radar detections, or only the Doppler of the detections,
                        and give the relative change ||a - b|| / ||a|| of the memories, the fusion, the trunk, the motion block and of the heads' outputs
                        (a stage that does not change when its input is removed is not reading it)
    static_check.json   the velocity field (v_hat and every v_levels.*) on static GT cells (speed < 0.3 m/s) and on moving ones (>= 1 m/s): mean offset, spread, EPE
                        (a field with an offset along the ego velocity on static cells has not learnt to subtract the ego motion)
    frame_<k>.png       one panel per stage, the GT beside the heads (x forward = up)
The probed modules are the model's direct children (`chain_modules`: lift AND lift2 of a dual camera, lane, ...), the trunk through its `.features()`; `bev_timing.py`
times the same. What to know when reading the numbers:
  * a module called once per dt (the heads) is recorded at its FIRST call (dt -0.5 for `sem_head:` / `obj_head:`); `OUT dt0:` holds the dt-0 outputs
  * `aux_cam` / `aux_rad` only run while training: no output here (the `radar_aux` tap of `qual debug` runs the head on the radar branch for a look)
  * the response row of a stage uses its first two tensors (motion_block: the two coarsest levels; `OUT:v_hat` is the finest)
  * `zero_doppler` sets the radial velocity of every detection to 0, which in the ego-compensated frame reads "every detection stands still", not "no Doppler";
    the camera perturbation also zeroes `lane_img` / `img`; a perturbation that changes nothing at all prints a WARNING (the t3 cam_off class of bug)
  * "static" cells of the velocity check are all defined cells with GT speed < 0.3 m/s, empty ground included, not only static objects
  * the velocity-field and response numbers of a few frames are a probe, not a measurement: no confidence interval
Every perturbation is itself tested (`--selftest`): it changes exactly the keys it names and nothing else."""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, ".."), os.path.join(ROOT, "..", "autoresearch")]

MODULES = ("lift", "cam_reduce", "cam_reduce2", "cam_merge", "cam_temporal", "radar", "rad_temporal", "fusion", "trunk", "head_reduce",
           "motion_block", "sem_head", "obj_head", "aux_cam", "aux_rad")     # the chain of the t16 configuration, for reading; the probe takes the model's own children (chain_modules)
MEMORIES = ("cam_temporal", "rad_temporal", "fusion")                       # shown again on the warm frame
RESPONSE_STAGES = ("cam_temporal", "radar", "rad_temporal", "fusion", "trunk", "motion_block")
STATIC_SPEED, MOVING_SPEED = 0.3, 1.0                                       # m/s: GT cells below / above these speeds are static / moving


# ------------------------------------------------------------------------------------------------ numbers of tensors
def tensors_of(o, prefix=""):
    """every float tensor inside a module output (tensor, tuple, list, dict), named by its path"""
    out = {}
    if torch.is_tensor(o):
        if o.is_floating_point():
            out[prefix or "out"] = o.detach().float().cpu().clone()               # a copy: float32 on the CPU is no copy, a later in-place op would change the record
    elif isinstance(o, dict):
        for k, v in o.items():
            out.update(tensors_of(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(o, (tuple, list)):
        for i, v in enumerate(o):
            out.update(tensors_of(v, f"{prefix}[{i}]" if prefix else f"[{i}]"))
    return out


def chain_modules(model):
    """the names of the model's direct child modules in definition order: everything the chain is made of (lift AND lift2 of a dual camera, lane, cam_adapter, ...),
    not a fixed list. aux_cam / aux_rad are children but only run while training (model.py), so they have no output in an evaluation-mode probe."""
    return [n for n, _ in model.named_children()]


def wrap_method(obj, name, wrapper):
    """replace obj.<name> by wrapper(original) on the instance; -> restore() that puts back exactly what was there (the class method, or the instance attribute)"""
    had = name in obj.__dict__
    original = getattr(obj, name)
    setattr(obj, name, wrapper(original))

    def restore():
        if had:
            setattr(obj, name, original)
        else:
            obj.__dict__.pop(name, None)
    return restore


def stat(t):
    """shape, finite, mean, std, absmax, share of exact zeros and share of dead channels (dim 1 of a >= 3-D tensor; None otherwise) of one tensor"""
    t = t.float()
    dead = None
    if t.dim() >= 3:
        c = t.reshape(t.shape[0], t.shape[1], -1)
        dead = float((c.std(dim=(0, 2)) < 1e-7).float().mean()) if c.numel() > 1 else None
    return {"shape": list(t.shape), "finite": bool(torch.isfinite(t).all()), "mean": float(t.mean()), "std": float(t.std()) if t.numel() > 1 else 0.0,
            "absmax": float(t.abs().max()), "zero_share": float((t == 0).float().mean()), "dead_channels": dead}


def rel_change(a, b):
    """||a - b|| / ||a||: 0 = unchanged, 1 = the change is as large as the signal"""
    a, b = a.float(), b.float()
    return float((a - b).norm() / (a.norm() + 1e-9))


def energy(t):
    """(X, Y) map: channel norm of a (C, X, Y) / (1, C, X, Y) / (X, Y) tensor"""
    t = t.detach().float().cpu()
    while t.dim() > 3:
        t = t[0]
    return (t.norm(dim=0) if t.dim() == 3 else t).numpy()


def mean_over_frames(lst):
    """list of stat dicts (one per frame) -> one dict: numeric fields averaged, the others (shape) from the first frame"""
    return {k: (float(np.mean([d[k] for d in lst])) if isinstance(lst[0][k], (int, float)) and lst[0][k] is not None else lst[0][k]) for k in lst[0]}


def average_response(lst):
    """list over frames of {stage: {tensor: relative change}} -> {"stage:tensor": mean over frames} (None when never defined)"""
    o = {}
    for ch in lst:
        for n, d in ch.items():
            for p, v in d.items():
                o.setdefault(f"{n}:{p}", []).append(v)
    return {k: (float(np.mean([x for x in v if x is not None])) if any(x is not None for x in v) else None) for k, v in o.items()}


# ------------------------------------------------------------------------------------------------ the velocity field on static and moving cells
def velocity_rows(frame, ego_v, fields, gv, undefined):
    """fields {level name: (2, X, Y) tensor}, gv the GT velocity (2, X, Y) in m/s, undefined (X, Y) bool -> one row per level: on the static cells (GT speed < 0.3,
    defined) the mean field (vx, vy), its spread and the mean speed (zero for a field that knows the cells stand still), on the moving cells (>= 1 m/s) the
    end-point error and the mean GT speed"""
    gv = gv.float()
    speed = gv.norm(dim=0)
    static, moving = (speed < STATIC_SPEED) & ~undefined, (speed >= MOVING_SPEED) & ~undefined
    rows = []
    for lvl, v in fields.items():
        v = v.float()
        vs = v[:, static]
        rows.append({"frame": frame, "level": lvl, "ego_v_mps": float(ego_v), "static_cells": int(static.sum()),
                     "static_mean_vx": float(vs[0].mean()) if vs.shape[1] else None, "static_mean_vy": float(vs[1].mean()) if vs.shape[1] else None,
                     "static_std_x": float(vs[0].std()) if vs.shape[1] > 1 else None, "static_std_y": float(vs[1].std()) if vs.shape[1] > 1 else None,
                     "static_epe": float(vs.norm(dim=0).mean()) if vs.shape[1] else None, "moving_cells": int(moving.sum()),
                     "moving_epe": float((v[:, moving] - gv[:, moving]).norm(dim=0).mean()) if moving.any() else None,
                     "moving_gt_speed": float(gv[:, moving].norm(dim=0).mean()) if moving.any() else None})
    return rows


# ------------------------------------------------------------------------------------------------ the response test: remove an input
def zero_cam(batch):
    """every camera input times 0: the cached probability / context tensors the lift reads, and the images the lane branch and the camera student read (`lane_img`,
    `lane_img_old` of student_frames 'all', `img`) - a camera
    path that is left in the batch would make the test say 'the camera does not matter' (the t3 cam_off bug)"""
    return {**batch, **{k: batch[k] * 0 for k in ("prob", "ctx", "prob2", "ctx2", "lane_img", "lane_img_old", "img") if k in batch and torch.is_tensor(batch[k])}}


def zero_radar(batch):
    """no radar detection at all (every list of the detection dict empty) and the hit / static maps zero"""
    r = batch["radar"]
    keep = torch.zeros(r["x"].shape[0], dtype=torch.bool)
    out = {**batch, "radar": {k: v[keep] for k, v in r.items()}}
    out.update({k: batch[k] * 0 for k in ("radar_hits", "radar_static") if k in batch and torch.is_tensor(batch[k])})
    return out


def zero_doppler(batch):
    """the radial velocity of every detection 0 (in the ego-compensated frame that reads 'every detection stands still'): positions, ranges and rcs stay"""
    return {**batch, "radar": {**batch["radar"], "vr": batch["radar"]["vr"] * 0}}


PERTURBATIONS = {"zero_cam": zero_cam, "zero_radar": zero_radar, "zero_doppler": zero_doppler}


def changed_nothing(row):
    """True when no number of a response row moved: the perturbation did not reach the network (a key it did not remove, a flag that changed nothing)"""
    return all((v or 0.0) <= 1e-9 for d in row.values() for v in d.values())


def response_row(base_stages, base_out, new_stages, new_out):
    """{stage: {tensor: relative change}} for the stages both runs recorded (first two tensors of each) and the heads' outputs"""
    ch = {}
    for n in RESPONSE_STAGES:
        if n in base_stages and n in new_stages:
            ch[n] = {p: rel_change(base_stages[n][p], new_stages[n][p]) for p in list(base_stages[n])[:2] if p in new_stages[n]}
    both = lambda k: (base_out.get(k) is not None and new_out.get(k) is not None)      # noqa: E731
    ch["OUT"] = {"sem": rel_change(base_out["sem"], new_out["sem"]), "hm": rel_change(base_out["obj"]["hm"], new_out["obj"]["hm"]),
                 "vel": rel_change(base_out["obj"]["vel"], new_out["obj"]["vel"]) if "vel" in base_out["obj"] else None,
                 "v_hat": rel_change(base_out["v_hat"], new_out["v_hat"]) if both("v_hat") else None}
    return ch


# ------------------------------------------------------------------------------------------------ pictures
def frame_panels(stages, features, out0, gt, layer_names):
    """[(title, 2-D array, colormap)] of one frame: the stages' energy, the heads beside the GT"""
    panels = []

    def add(title, arr, cmap="viridis"):
        panels.append((title, np.asarray(arr), cmap))
    for n in ("cam_temporal", "radar", "rad_temporal", "fusion", "trunk"):
        if stages.get(n):
            add(f"{n} |feat|", energy(list(stages[n].values())[0]))
    add("features S |.|", energy(features))
    sem = out0["sem"][0].float().cpu()
    for lay in ("drivable_area", "lane_divider"):
        if lay in layer_names:
            add(f"pred {lay} (sigmoid)", torch.sigmoid(sem[layer_names.index(lay)]).numpy(), "magma")
            add(f"GT {lay}", np.asarray(gt[lay]), "magma")
    add("pred object heat (max over groups)", torch.sigmoid(out0["obj"]["hm"][0].float().cpu()).max(0).values.numpy(), "magma")
    add("GT vehicle", np.asarray(gt["vehicle"]), "magma")
    if out0.get("v_hat") is not None:
        add("v_hat |v| (motion block field)", out0["v_hat"][0].float().cpu().norm(dim=0).numpy(), "plasma")
    add("GT |flow| (m/s)", np.linalg.norm(np.asarray(gt["flow"]), axis=0), "plasma")
    for lv, v in (out0.get("v_levels") or {}).items():
        add(f"M level {lv} |v|", v[0].float().cpu().norm(dim=0).numpy(), "plasma")
    if "mov" in out0["obj"]:
        add("pred moving prob (hurdle)", torch.sigmoid(out0["obj"]["mov"][0].float().cpu()).max(0).values.numpy(), "magma")
    return panels


def draw_panels(path, title, panels, cols=4):
    """one figure, `cols` panels per row, x forward = up"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nr = (len(panels) + cols - 1) // cols
    fig, axs = plt.subplots(nr, cols, figsize=(3.4 * cols, 3.6 * nr), squeeze=False)
    for ax in axs.ravel():
        ax.axis("off")
    for ax, (t, arr, cmap) in zip(axs.ravel(), panels):
        ax.imshow(arr.T[::-1] if arr.ndim == 2 else arr, cmap=cmap)
        ax.set_title(t, fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=70)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ the probe of a run
class Recorder:
    """forward hooks that store the tensors of every probed module's output in `store` {module: {path: tensor}}; the trunk is called through `.features()`
    (not forward), so that method is wrapped. `close()` removes everything."""

    def __init__(self, model, modules=None):
        self.model, self.store, self._hooks, self._restore = model, {}, [], None
        for name in (chain_modules(model) if modules is None else modules):
            m = getattr(model, name, None)
            if m is not None and name != "trunk":
                self._hooks.append(m.register_forward_hook(self._hook(name)))
        trunk = getattr(model, "trunk", None)
        if trunk is not None and hasattr(trunk, "features"):
            def wrapper(original):
                def features(*args, **kw):
                    out = original(*args, **kw)
                    self.store.setdefault("trunk", tensors_of(out))
                    return out
                return features
            self._restore = wrap_method(trunk, "features", wrapper)

    def _hook(self, name):
        def hook(mod, inputs, output):                                  # returns None: a hook that returns a value REPLACES the module's output
            self.store.setdefault(name, tensors_of(output))
        return hook

    def clear(self):
        self.store.clear()

    def copy(self):
        return {n: {p: t.clone() for p, t in d.items()} for n, d in self.store.items()}

    def close(self):
        for h in self._hooks:
            h.remove()
        if self._restore is not None:
            self._restore()
            self._restore = None


def probe(run, frames=3, device="cpu", split="val", out_dir=None, threads=2, ckpt_dir=None):
    """the probe of one run (see the module text); -> {"modules": ..., "response": ..., "static": [...]}, files written to out_dir"""
    torch.set_num_threads(threads)
    run = os.path.abspath(run)
    out_dir = out_dir or os.path.join(run, "eval", "modules")
    os.makedirs(out_dir, exist_ok=True)
    from evaluation import bev_adapter as A
    A.setup_paths(run)
    src, subj, _ = A.build(run, device=device, split=split, workers=0, ckpt_dir=ckpt_dir)
    ns, model, loss, cfg = subj.ns, subj.model, subj.loss, subj.cfg
    T, dev = ns.T, subj.device
    print("run", run, "step", subj.step, "checkpoint", os.path.basename(subj.ckpt), "device", dev, flush=True)
    rec = Recorder(model)
    refs = []
    for s in src.scenes():
        fr = src.frames(s)
        refs += [(s, len(fr) // 2), (s, len(fr) // 3)]
    if len(refs) < frames:
        print(f"WARNING: only {len(refs)} frames available (2 per scene), not {frames}", flush=True)

    def forward(batch, **kw):
        rec.clear()
        with torch.no_grad():
            _, res, _ = T.forward_loss(model, loss, batch, 1.0, dev, cfg.amp, cfg, train_terms=False, **kw)
        return res

    modules, response, rows = {}, {}, []
    try:
        _probe_frames(refs[:frames], src, subj, rec, forward, modules, response, rows, out_dir, run)
    finally:
        rec.close()
    return _write(modules, response, rows, out_dir)


def _probe_frames(refs, src, subj, rec, forward, modules, response, rows, out_dir, run):
    ns, model, cfg = subj.ns, subj.model, subj.cfg
    T, dev = ns.T, subj.device
    for k, (scene, idx) in enumerate(refs):
        ref = src.frames(scene)[idx]
        sample = src.load(ref)
        batch = T.to_device(sample.inputs, dev)
        res = forward(batch)
        base = rec.copy()
        out0 = res["out"][0.0]
        for n, d in base.items():
            for p, t in d.items():
                modules.setdefault(f"{n}:{p}", []).append(stat(t))
        for p, t in tensors_of(out0, "out").items():
            modules.setdefault(f"OUT dt0:{p}", []).append(stat(t))
        for p, t in tensors_of(res.get("radar_side", {}), "radar_side").items():
            modules.setdefault(f"radar_side:{p}", []).append(stat(t))
        modules.setdefault("features(S)", []).append(stat(res["features"]))
        gt = sample.reference[0.0]
        fields = {"v_hat": out0["v_hat"][0].cpu()} if out0.get("v_hat") is not None else {}
        fields.update({f"v_levels.{kk}": vv[0].cpu() for kk, vv in (out0.get("v_levels") or {}).items()})
        rows += velocity_rows(ref.key, batch["v"][0], fields, torch.as_tensor(np.asarray(gt["velocity"])), torch.as_tensor(np.asarray(gt["undefined"])).bool())
        if idx + 1 < len(src.frames(scene)):                                    # the warm frame: the next one with the state of this one
            s2 = src.load(src.frames(scene)[idx + 1])
            forward(T.to_device(s2.inputs, dev), state=ns.detach_tree(res["state"]), step=ns.clip_step(sample.inputs, dev))
            for n, d in rec.store.items():
                for p, t in d.items():
                    if n in MEMORIES:
                        modules.setdefault(f"WARM {n}:{p}", []).append(stat(t))
        for pname, perturb in PERTURBATIONS.items():
            r2 = forward(perturb(batch))
            row = response_row(base, out0, rec.copy(), r2["out"][0.0])
            if changed_nothing(row):
                print(f"WARNING: {pname} changed nothing in frame {ref.key}: the perturbation does not reach the network (a key it does not remove?)", flush=True)
            response.setdefault(pname, []).append(row)
        draw_panels(os.path.join(out_dir, f"frame_{k}.png"), f"{os.path.basename(run)} step {subj.step}  frame {ref.key}",
                    frame_panels(base, res["features"], out0, gt, [s_[0] for s_ in model.spec]))
        print("frame", k, ref.key, "done", flush=True)


def _write(modules, response, rows, out_dir):
    summary = {k: mean_over_frames(v) for k, v in modules.items()}
    resp = {k: average_response(v) for k, v in response.items()}
    for name, obj in (("modules.json", summary), ("response.json", resp), ("static_check.json", rows)):
        with open(os.path.join(out_dir, name), "w") as f:
            json.dump(obj, f, indent=1)
    return {"modules": summary, "response": resp, "static": rows, "out_dir": out_dir}


def print_report(r):
    print("\nSTATIC / MOVING CELLS of the velocity field (per frame and level):")
    for row in r["static"]:
        print("  ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if k != "frame"})
    print("\nMODULE OUTPUTS (mean over frames):")
    for k, v in r["modules"].items():
        print(f"  {k:48s} std {v['std']:10.4g} absmax {v['absmax']:10.4g} zero {v['zero_share']:.2f} dead_ch {v['dead_channels']} finite {v['finite']}")
    print("\nRESPONSE (relative change of the stage output when the input is removed):")
    for p, d in r["response"].items():
        print(" ", p)
        for k, v in d.items():
            print(f"    {k:40s} {v if v is None else round(v, 4)}")
    print("written", r["out_dir"])


# ------------------------------------------------------------------------------------------------ self-test
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="?")
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--split", default="val")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default=None)
    ap.add_argument("--ckpt-dir", default=None, help="the folder of a moved checkpoint")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        selftest()
        return 0
    if not a.run:
        ap.error("RUN is required (or --selftest)")
    print_report(probe(a.run, a.frames, a.device, a.split, a.out, a.threads, a.ckpt_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
