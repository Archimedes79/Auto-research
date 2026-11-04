"""t12: train ONLY the motion block (training/motion_block.py) on a frozen, trained base model, and compare its velocity with
the base's own velocity head on the same validation frames (Mirko 2026-11-01: "use an existing training and just train
this block if it improves anything").

The base (a finished run: config.json + checkpoint, EMA weights) is frozen entirely and run in eval mode (BatchNorm with
its running statistics). Two hooks, no edit of the model code:
  * model.camera_bevs is wrapped: after the frozen camera BEVs of the window are made, the block gets the current frame,
    an older window frame splatted to now by the ego motion only (splat.splat_window without a velocity head), the fixed
    radar raster of the current keyframe and the age of the older frame -> the velocity of every pyramid level
  * model.mover.motion (the velocity head of the time mover, `TimeMover.motion`) is replaced by the block's last level
So the block's velocity moves the trunk state to -0.5 / +0.5 s for the timed heads and is the "v_hat" of the motion
loss; the outputs at dt = 0 do not change. Training loss = the base's loss (BEVLoss with its frozen Kendall weights, the
GT of -0.5 / 0 / +0.5 s; the gradient reaches only the block: the motion term and the maps / boxes decoded at +-0.5 s)
+ the motion term on the coarser levels with the weight of the motion group (deep supervision, as tests/test_flow_fused.py).
Every keyframe runs cold (seq_len 1: no world memory carried), the same for both arms.

Validation (the same `val_subset` frames for both arms): velocity error (EPE, m/s) on supervised object cells (all,
moving > 1 m/s, static, moving vehicles), at the GT box centres (the field vs the object head's own box velocity), and the
loss terms of the base's loss (groups, the +-0.5 s terms). -> <out>/metrics.json, <out>/log.jsonl, <out>/block.pt
Usage (from the bev_fusion folder): python training/train_motion_block.py --base runs/t10_dual_s0 --ckpt ckpt_00003000.pt
    --out runs/t12_mb_s0 [--seed 0] [--steps 1500] [--old 2] [--val 300] [--lr 1e-3]
"""
import argparse
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from losses import _gather, build_layer_targets, motion_term, object_head_terms, sample_targets  # noqa: E402
from motion_block import MotionBlock, backward_warp, radar_raster  # noqa: E402
from splat import splat_window  # noqa: E402
from train import build, forward_loss, make_datasets, make_loader, read_config, val_subset, warm_start  # noqa: E402


class Hook(torch.nn.Module):
    """stands in for TimeMover.motion: returns the velocity the wrapped camera_bevs computed for this batch."""

    def __init__(self):
        super().__init__()
        self.v = None

    def forward(self, S):
        assert self.v is not None and self.v.shape[-2:] == S.shape[-2:], (None if self.v is None else self.v.shape, S.shape)
        return self.v


def attach(model, block, old):
    """wrap model.camera_bevs so the block runs on every batch; returns (hook, orig motion head, ctx dict)."""
    orig_cb, orig_motion, hook, ctx = model.camera_bevs, model.mover.motion, Hook(), {"on": True, "levels": None}

    def camera_bevs(batch):
        with torch.no_grad():
            frames = orig_cb(batch)
        B = frames.shape[0]
        steps = [tuple(torch.as_tensor(t, dtype=frames.dtype, device=frames.device) for t in s)
                 for s in batch["steps"][old]]
        with torch.no_grad():
            w, m = splat_window([(frames[:, old], steps)], model.grid, head=None, mass_gate=True)
            rad = radar_raster(batch["radar"], B, model.grid)
        dt = sum(s[0] for s in steps).abs().reshape(B)
        inputs = (frames[:, 0], w[:, 0], m[:, 0].clamp(max=1.0), rad, dt)
        ctx["inputs"] = tuple(t[:1].detach() for t in inputs)              # one sample, for the timing at the end
        ctx["raw"] = inputs                                                # for the warp check of the base evaluation
        if ctx["on"]:
            levels = block(*inputs)
            ctx["levels"], hook.v = levels, levels[-1]
        return frames
    model.camera_bevs = camera_bevs
    return hook, orig_motion, ctx


@torch.no_grad()
def block_timing(block, inputs, n=30):
    """ms per frame of the block alone (batch 1, eval mode) and the GPU memory in use at that moment (another job on the
    GPU makes the number larger: it is an idle-GPU timing only when the memory line says nothing else runs)."""
    import subprocess
    for _ in range(5):
        block(*inputs)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        block(*inputs)
    torch.cuda.synchronize()
    try:
        used = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True).stdout.split()[0])
    except Exception:                                   # no nvidia-smi: the number stays unlabelled
        used = -1
    return {"block_ms_per_frame": 1000 * (time.time() - t0) / n, "gpu_used_mb_at_timing": used}


def pin_gt(ds, sig):
    """read the GT of an earlier generator signature, READ-ONLY: the dataset keeps only the samples whose cache file of that
    signature exists, and a missing file raises instead of being built by the current generator (which would store new GT
    under the old name). The base was trained and evaluated on its own signature (runs/<base>/gt.json); the live generator
    may have changed since (autogt edits change the signature). -> (samples kept, samples before)"""
    n0 = len(ds.samples)
    ds.gt_signature = lambda: sig
    ds.samples = [t for t in ds.samples if os.path.exists(os.path.join(ds.cache_dir, f"{t}_{sig}.npz"))]
    orig = ds._gt

    def _gt(token):
        if not os.path.exists(os.path.join(ds.cache_dir, f"{token}_{sig}.npz")):
            raise RuntimeError(f"GT {token}_{sig} not cached: pinned GT is read-only")
        return orig(token)
    ds._gt = _gt
    return len(ds.samples), n0


def set_arm(model, arm, hook, orig_motion, ctx):
    ctx["on"] = arm == "block"
    model.mover.motion = hook if arm == "block" else orig_motion


def coarse_terms(levels, gts, loss, device):
    T = build_layer_targets(gts[0.0], loss.cfg, None, 0.0, device)
    return [motion_term(v, T, loss.cfg) for v in levels[:-1]], T


@torch.no_grad()
def evaluate(model, loss, loader, cfg, device, arm, hook, orig_motion, ctx):
    set_arm(model, arm, hook, orig_motion, ctx)
    acc, n = {}, 0
    sums = {k: [0.0, 0] for k in ("all", "moving", "static", "moving_vehicle", "centre_moving_field", "centre_moving_objhead",
                                  "centre_moving_zero", "warp_gt", "warp_zero")}

    def add(k, e):
        sums[k][0] += float(e.sum())
        sums[k][1] += int(e.numel())
    for batch in loader:
        out, res, gts = forward_loss(model, loss, batch, 1.0, device, "off", cfg, train_terms=False)
        for k, v in out.items():
            if torch.is_tensor(v) and v.dim() == 0:
                acc[k] = acc.get(k, 0.0) + float(v)
        n += 1
        p = res["out"][0.0]
        T = build_layer_targets(gts[0.0], loss.cfg, None, 0.0, device)
        v, g = p["v_hat"].float(), T["velocity"].float()
        sup = (T["w_obj"][:, 0] > 0) & (T["obj_cells"][:, 0] > 0)
        e = (v - g).norm(dim=1)
        spd = g.norm(dim=1)
        add("all", e[sup])
        add("moving", e[sup & (spd > 1.0)])
        add("static", e[sup & (spd <= 1.0)])
        if "vehicle" in T:
            add("moving_vehicle", e[sup & (spd > 1.0) & (T["vehicle"][:, 0] > 0.5)])
        _, tt = object_head_terms(p["obj"], T["boxes"], T["w_obj"], model.grid, loss.groups, loss.cfg.w_obj_vel,
                                  loss.cfg.max_obj, loss.cfg.w_obj_mov)
        ind, msk = tt["ind"].to(device), tt["mask"].to(device) > 0
        gc = _gather(g, ind)                                      # (B, N, 2) GT velocity at the box centre cells
        mov = msk & (gc.norm(dim=-1) > 1.0)
        add("centre_moving_field", (_gather(v, ind) - gc).norm(dim=-1)[mov])
        add("centre_moving_objhead", (_gather(p["obj"]["vel"].float(), ind) - gc).norm(dim=-1)[mov])
        add("centre_moving_zero", gc.norm(dim=-1)[mov])
        if arm == "base":       # layer 0 on the real features: the older frame warped with the GT velocity must match the
            cur, w, m, _, dt = ctx["raw"]          # current one better than with v = 0 (moving object cells, both landed)
            cells = sup & (spd > 1.0) & (m[:, 0] > 0.5)
            for name, vv in (("warp_gt", g), ("warp_zero", torch.zeros_like(g))):
                add(name, (backward_warp(w, vv, dt, model.grid.res) - cur).abs().mean(1)[cells])
    res = {k: v / max(1, n) for k, v in acc.items()}
    res.update({f"epe_{k}": (s / c if c else float("nan")) for k, (s, c) in sums.items()})
    res.update({f"n_{k}": c for k, (s, c) in sums.items()})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--old", type=int, default=2, help="window frame used as the older frame (2 = the oldest, ~0.33 s)")
    ap.add_argument("--val", type=int, default=300, help="validation samples (spread over all val scenes)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--gt", default="base", choices=["base", "live"],
                    help="base: the GT signature of the base run (runs/<base>/gt.json), read-only; live: the current generator")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = read_config(os.path.join(a.base, "config.json"))
    cfg.seq_len, cfg.seq_bptt, cfg.workers = 1, False, a.workers
    model, loss, _ = build(cfg)
    fresh = warm_start(os.path.join(a.base, a.ckpt), model, loss)
    assert not [k for k in fresh if not k.startswith("camera.")], fresh[:5]
    model.to(device).eval()
    loss.to(device).eval()
    for p in list(model.parameters()) + list(loss.parameters()):
        p.requires_grad_(False)
    block = MotionBlock(model.cfg.c_cam, model.grid).to(device)
    hook, orig_motion, ctx = attach(model, block, a.old)
    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    n_par = sum(p.numel() for p in block.parameters())
    print(f"base {a.base}/{a.ckpt} frozen; motion block {n_par / 1e3:.0f} k parameters; older frame {a.old}; device {device}",
          flush=True)

    ds_train, ds_val = make_datasets(cfg)
    sig = json.load(open(os.path.join(a.base, "gt.json")))["gt_signature"] if a.gt == "base" else None
    if sig:
        for name, ds in (("train", ds_train), ("val", ds_val)):
            kept, n0 = pin_gt(ds, sig)
            print(f"GT pinned to the base's {sig} (read-only): {name} {kept} of {n0} samples cached", flush=True)
    gen = torch.Generator().manual_seed(a.seed)
    train_loader = make_loader(ds_train, cfg, True, generator=gen)
    val_loader = make_loader(val_subset(ds_val, a.val), cfg, False, workers=2)
    opt = torch.optim.AdamW(block.parameters(), lr=a.lr, weight_decay=1e-4)
    ck_path, step = os.path.join(a.out, "block.pt"), 0
    metrics_path = os.path.join(a.out, "metrics.json")
    metrics = json.load(open(metrics_path)) if os.path.exists(metrics_path) else {}
    if os.path.exists(ck_path):
        ck = torch.load(ck_path, map_location=device, weights_only=False)
        block.load_state_dict(ck["block"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        print(f"resumed at step {step}", flush=True)
    if "base" not in metrics:
        t0 = time.time()
        metrics["base"] = evaluate(model, loss, val_loader, cfg, device, "base", hook, orig_motion, ctx)
        print(f"base evaluated ({time.time() - t0:.0f} s): moving EPE {metrics['base']['epe_moving']:.3f}, moving vehicles "
              f"{metrics['base']['epe_moving_vehicle']:.3f}", flush=True)
        json.dump(metrics, open(metrics_path, "w"), indent=1)

    set_arm(model, "block", hook, orig_motion, ctx)
    block.train()
    t0, it = time.time(), iter(train_loader)
    while step < a.steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)
        f = step / a.steps
        for g in opt.param_groups:
            g["lr"] = a.lr * min(1.0, (step + 1) / 100) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * f)))
        out, res, gts = forward_loss(model, loss, batch, 1.0, device, "off", cfg, train_terms=False)
        coarse, _ = coarse_terms(ctx["levels"], gts, loss, device)
        w_motion = torch.exp(-out["s.motion"]) if "s.motion" in out else 1.0
        total = out["total"] + w_motion * sum(coarse)
        if not torch.isfinite(total):
            print(f"step {step}: non-finite loss, skipped", flush=True)
            opt.zero_grad(set_to_none=True)
            step += 1
            continue
        opt.zero_grad(set_to_none=True)
        total.backward()
        gn = torch.nn.utils.clip_grad_norm_(block.parameters(), 5.0)
        opt.step()
        step += 1
        if step % 50 == 0 or step == 1:
            rec = {"step": step, "total": float(total.detach()), "motion": float(out.get("motion@+0.0", float("nan"))),
                   "coarse": [float(c) for c in coarse], "grad_norm": float(gn), "lr": opt.param_groups[0]["lr"],
                   "s_per_step": (time.time() - t0) / 50}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            t0 = time.time()
        if step % 500 == 0 or step == a.steps:
            torch.save({"block": block.state_dict(), "opt": opt.state_dict(), "step": step, "args": vars(a)}, ck_path + ".tmp")
            os.replace(ck_path + ".tmp", ck_path)
    block.eval()
    metrics["block"] = evaluate(model, loss, val_loader, cfg, device, "block", hook, orig_motion, ctx)
    metrics["meta"] = {"base": a.base, "ckpt": a.ckpt, "steps": step, "seed": a.seed, "old_frame": a.old,
                       "val_samples": len(val_subset(ds_val, a.val)), "block_params": n_par, "lr": a.lr,
                       "batch_size": cfg.batch_size, "split": "val", "memory": "cold (seq_len 1)",
                       "gt_signature": ds_val.gt_signature()}
    if device == "cuda":
        metrics["meta"]["peak_gpu_gb"] = torch.cuda.max_memory_reserved() / 1e9
        metrics["meta"].update(block_timing(block, ctx["inputs"]))
    json.dump(metrics, open(metrics_path, "w"), indent=1)
    b, k = metrics["base"], metrics["block"]
    print("== VAL, same frames, EPE m/s: " + ", ".join(f"{n} base {b['epe_' + n]:.3f} block {k['epe_' + n]:.3f}"
                                                      for n in ("moving", "moving_vehicle", "static", "all")), flush=True)
    print(f"== box centres, moving: field {k['epe_centre_moving_field']:.3f} (base field {b['epe_centre_moving_field']:.3f}), "
          f"object head {k['epe_centre_moving_objhead']:.3f}, v = 0 {k['epe_centre_moving_zero']:.3f}", flush=True)
    print(f"== total_main base {b.get('total_main', float('nan')):.4f} block {k.get('total_main', float('nan')):.4f}; "
          f"metrics {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
