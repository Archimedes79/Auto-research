"""Evaluate a training run (runs/<name>) on the validation split under its EMA weights, with evaluation/metrics.py.

    runs/<name>/config.json + the last ckpt_*.pt (EMA weights) -> the cached model -> every val sample (or --n spread
    evenly over all val scenes) at dt = -0.5 / 0 / +0.5 s (the true keyframe gap, as in training: train.forward_loss)
      -> SegMeter: map areas, lines (zero crossing of the distance field), dense object coverage; masks "fov" (camera
         cells) and "seen" (camera or radar cells, the training mask of the map layers); known / unknown by splat mass
      -> DetMeter: object head boxes (decode, threshold 0.05, top 100) against the observed GT boxes, front field of view
      -> the per-scene records of the headline numbers (paired scene bootstrap between runs: --compare)
      -> the prior baseline (--prior N): per-cell mean of the GT layers over N train samples, as a constant prediction
         (the strong simple baseline of a map: what the position in the grid alone gives); the null baseline is IoU 0
Output in runs/<name>/eval/: metrics.json (flat, metrics.report), table.txt, scene_records.pkl, images/*.png (--images N),
latency.json (--time, only on an idle GPU: the GPU's utilisation is checked first and written next to the numbers).
Usage (from the bev_fusion folder):
    wsl ~/miniconda3/envs/gcar/bin/python evaluation/evaluate_run.py runs/t1_base [--n 0] [--prior 500] [--images 8] [--time]
    wsl ~/miniconda3/envs/gcar/bin/python evaluation/evaluate_run.py --compare runs/a runs/b [--boot 1000]
"""
import argparse
import glob
import json
import math
import os
import pickle
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
# the model code: the run's own snapshot (runs/<name>/code/, written by the launcher from a pinned commit) when it exists,
# so a run is evaluated with the code it was trained with even after the shared training/ code changed; else training/
_RUN = next((a for a in sys.argv[1:] if not a.startswith("-") and os.path.isdir(os.path.join(a, "code"))), None)
CODE = os.path.abspath(os.path.join(_RUN, "code")) if _RUN else os.path.join(ROOT, "training")
sys.path.insert(0, CODE)
sys.path.insert(0, ROOT)
from evaluation.metrics import (DetMeter, SegMeter, FRONT_HALF_FOV, format_table, gt_boxes_from_meta,  # noqa: E402
                                match, ap_nuscenes, ms_per_frame, report)
from losses import NUSC_GROUPS  # noqa: E402
from object_head import decode  # noqa: E402
from train import (EMA, TrainConfig, build, forward_loss, load_checkpoint, make_datasets, make_loader,  # noqa: E402
                   param_groups, val_subset)
try:                                                  # sequence training (snapshots before 2026-10-26 do not have it)
    from train import clip_step, detach_tree  # noqa: E402
except ImportError:
    clip_step = detach_tree = None

HEADLINE_SEG = (("drivable_area", "fov"), ("walkway", "fov"), ("lane_divider", "fov"), ("lane_divider:zc", "fov"),
                ("vehicle", "fov"))
SEG_METRIC = {"drivable_area": "iou@0", "walkway": "iou@0", "lane_divider": "iou@0.5", "lane_divider:zc": "f1@0.5",
              "vehicle": "iou@0"}


# ------------------------------------------------------------------------------------------------ helpers
def load_run(run, device, ckpt_dir=None):
    with open(os.path.join(run, "config.json")) as f:
        raw = json.load(f)
    cfg = TrainConfig(**raw)
    model, loss, _ = build(cfg)
    model, loss = model.to(device), loss.to(device)
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir or run, "ckpt_*.pt")))
    if not ckpts:
        raise FileNotFoundError(f"no checkpoint in {ckpt_dir or run}")
    opt = torch.optim.AdamW(param_groups(model, loss, cfg.lr, cfg.weight_decay), lr=cfg.lr)   # as train(), to load
    ema = EMA(model, cfg.ema)
    step, _ = load_checkpoint(ckpts[-1], model, loss, opt, ema)
    ema.apply(model)                                            # evaluate the EMA weights
    model.eval()
    return cfg, model, loss, step, ckpts[-1]


def layer_index(spec):
    return {name: (i, kind, key) for i, (name, kind, key) in enumerate(spec)}


def predictions(p, b, spec):
    """one sample's dict of probability maps (areas, lines, objects) and distance fields from the model output p."""
    sem = p["sem"][b].float()
    prob, dist = {}, {}
    for name, (i, kind, key) in layer_index(spec).items():
        if kind in ("area", "object", "line"):
            prob[name] = torch.sigmoid(sem[i]).cpu().numpy()
        elif kind == "dist":
            dist[name[:-len("_dist")]] = sem[i].clamp(min=0).cpu().numpy()
    return prob, dist


def masks(g):
    cam = np.asarray(g["camera"], np.float32) > 0
    seen = np.asarray(g["undefined"], np.float32) == 0
    keep = (np.asarray(g["ignore"], np.float32) == 0) if "ignore" in g else np.ones_like(cam)
    return {"fov": cam, "seen": seen, "obj_fov": cam & keep, "out": ~cam}      # out: whole grid outside the current view


def add_sample(meters, prob, dist, g, spec, dt, mass):
    """all SegMeter updates of one sample at one dt; meters: {"fov": SegMeter, "seen": SegMeter, "obj": SegMeter}."""
    k = layer_index(spec)
    m = masks(g)
    areas = {n: prob[n] for n, (_, kind, _) in k.items() if kind in ("area", "line")}     # line = the 1-cell mask head
    objs = {n: prob[n] for n, (_, kind, _) in k.items() if kind == "object"}
    gt_a = {n: np.asarray(g[k[n][2]], np.float32) for n in areas}
    gt_o = {n: np.asarray(g[k[n][2]], np.float32) for n in objs}
    gt_d = {n: np.asarray(g[n + "_dist"], np.float32) for n in dist}
    for mname in ("fov", "seen", "out"):
        if mname not in meters:
            continue
        meters[mname].add(areas, gt_a, mask=m[mname], mass=mass, dt=dt)
        meters[mname].add_lines(dist, gt_d, mask=m[mname], mass=mass, dt=dt)
    meters["obj"].add(objs, gt_o, mask=m["obj_fov"], mass=mass, dt=dt)


def gt_boxes(g):
    return [b for b in gt_boxes_from_meta(g["meta"]["boxes"]) if b["group"] in dict(NUSC_GROUPS)]


def prior_maps(train_ds, n, spec):
    """per-cell mean of every evaluated GT layer over n train samples (spread evenly), dt = 0 keyframes."""
    idx = np.linspace(0, len(train_ds) - 1, n).round().astype(int)
    k = layer_index(spec)
    acc, accd = {}, {}
    for i in idx:
        g = train_ds._gt(train_ds.samples[i])["cur"]
        for name, (_, kind, key) in k.items():
            if kind in ("area", "object", "line"):
                acc[name] = acc.get(name, 0) + np.asarray(g[key], np.float64)
            elif kind == "dist":
                accd[name[:-5]] = accd.get(name[:-5], 0) + np.asarray(g[key], np.float64)
    return {n: (v / len(idx)).astype(np.float32) for n, v in acc.items()}, \
           {n: (v / len(idx)).astype(np.float32) for n, v in accd.items()}


def gpu_busy():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout.strip().split(",")
        return {"util_percent": float(out[0]), "mem_mib": float(out[1])}
    except Exception as e:                                    # noqa: BLE001
        return {"error": str(e)}


# ------------------------------------------------------------------------------------------------ images
COLORS = {"drivable_area": (0.80, 0.80, 0.80), "carpark_area": (0.70, 0.78, 0.90), "walkway": (0.62, 0.85, 0.62),
          "ped_crossing": (1.00, 0.70, 0.35), "stop_line": (0.80, 0.68, 0.92),
          "vehicle": (0.15, 0.35, 0.90), "pedestrian": (0.85, 0.15, 0.75), "two_wheeler": (0.10, 0.75, 0.80)}
LINE_COLORS = {"lane_divider": (0.15, 0.15, 0.15), "road_divider": (0.85, 0.10, 0.10)}


def composite(prob, dist, line_width=0.35):
    """RGB (X, Y, 3) of the probabilities (alpha = probability) in the order of COLORS, lines from the distance field."""
    shape = next(iter(prob.values())).shape
    img = np.ones(shape + (3,), np.float32)
    for name, c in COLORS.items():
        if name in prob:
            a = np.clip(prob[name], 0, 1)[..., None]
            img = img * (1 - a) + np.asarray(c, np.float32) * a
    for name, c in LINE_COLORS.items():
        if name in dist:
            a = np.clip(1 - dist[name] / line_width, 0, 1)[..., None]
            img = img * (1 - a) + np.asarray(c, np.float32) * a
    return img


def draw_boxes(ax, boxes, color, lw=1.2, ls="-", score_min=None):
    from matplotlib.patches import Circle, Polygon
    for b in boxes:
        if score_min is not None and b.get("score", 1.0) < score_min:
            continue
        if "radius" in b:
            ax.add_patch(Circle((b["cy"], b["cx"]), max(b["radius"], 0.3), fill=False, ec=color, lw=lw, ls=ls))
        else:
            c, s = math.cos(b["yaw"]), math.sin(b["yaw"])
            hl, hw = b["length"] / 2, b["width"] / 2
            pts = [(b["cx"] + c * a - s * w, b["cy"] + s * a + c * w) for a, w in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))]
            ax.add_patch(Polygon([(y, x) for x, y in pts], closed=True, fill=False, ec=color, lw=lw, ls=ls))
            ax.plot([b["cy"], b["cy"] + s * hl], [b["cx"], b["cx"] + c * hl], color=color, lw=lw)


def bev_axes(ax, grid, title):
    ax.set_xlim(grid.y_max, grid.y_min)                                # +y (left of the car) on the left
    ax.set_ylim(grid.x_min, grid.x_max)
    ax.set_aspect("equal")
    for sgn in (-1, 1):
        ax.plot([0, sgn * 60 * math.tan(FRONT_HALF_FOV)], [0, 60], color="0.4", lw=0.6, ls=":")
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)


def show(ax, img, grid):
    """(X, Y[, 3]) grid array: rows = x (forward, up), columns = y; bev_axes inverts the y axis (+y on the left)."""
    ax.imshow(img, origin="lower", interpolation="nearest", extent=(grid.y_min, grid.y_max, grid.x_min, grid.x_max),
              aspect="auto")


def save_image(path, token, nusc, root, grid, prob, dist, gtd, dets, gts, probs_next, dets_next, gt_next, gts_next,
               mass_next, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    fig, axs = plt.subplots(2, 3, figsize=(15, 10.5))
    sd = nusc.get("sample_data", nusc.get("sample", token)["data"]["CAM_FRONT"])
    axs[0, 0].imshow(Image.open(os.path.join(root, sd["filename"])).convert("RGB").resize((800, 450)))
    axs[0, 0].set_title("CAM_FRONT (input, current frame)", fontsize=9)
    axs[0, 0].axis("off")
    gt_prob = {n: np.asarray(gtd[n], np.float32) for n in prob if n in gtd}
    gt_dist = {n: np.asarray(gtd[n + "_dist"], np.float32) for n in dist}
    panels = [(axs[0, 1], composite(gt_prob, gt_dist), gts, None, "GT, dt 0"),
              (axs[0, 2], composite(prob, dist), dets, gts, "prediction, dt 0 (boxes: score >= 0.3; GT dashed)"),
              (axs[1, 1], composite(probs_next[0], probs_next[1]), dets_next, gts_next, "prediction, dt +0.5 s (GT dashed)")]
    for ax, img, boxes, ref, t in panels:
        show(ax, img, grid)
        bev_axes(ax, grid, t)
        if ref is not None:
            draw_boxes(ax, ref, "k", lw=0.8, ls="--")
        draw_boxes(ax, boxes, "tab:blue" if ref is None else "tab:red", score_min=None if ref is None else 0.3)
    # error map of the drivable area at dt 0 on the camera cells: TP grey, FP red, FN blue
    P, G = prob["drivable_area"] > 0.5, np.asarray(gtd["drivable_area"], np.float32) > 0.5
    cam = np.asarray(gtd["camera"], np.float32) > 0
    err = np.ones(P.shape + (3,), np.float32)
    err[P & G & cam] = (0.7, 0.7, 0.7)
    err[P & ~G & cam] = (0.9, 0.2, 0.2)
    err[~P & G & cam] = (0.2, 0.4, 0.95)
    err[~cam] = (0.95, 0.95, 0.88)
    show(axs[1, 0], err, grid)
    bev_axes(axs[1, 0], grid, "drivable area dt 0: TP grey, FP red, FN blue (camera cells)")
    im = axs[1, 2].imshow(np.clip(mass_next, 0, 1.5), origin="lower", interpolation="nearest", cmap="viridis", vmin=0,
                          vmax=1.5, extent=(grid.y_min, grid.y_max, grid.x_min, grid.x_max), aspect="auto")
    fig.colorbar(im, ax=axs[1, 2], fraction=0.04)
    bev_axes(axs[1, 2], grid, "splat mass at dt +0.5 s (< 0.5 = unknown cells)")
    from matplotlib.patches import Patch
    handles = [Patch(color=c, label=n) for n, c in list(COLORS.items()) + list(LINE_COLORS.items())]
    axs[0, 0].legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=8, frameon=False)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ evaluation
def perturb(batch, control):
    """in place: the sanity controls of evaluate_run (the model must lose what the control takes away, else the input is not used)."""
    B = len(batch["tokens"])
    if control == "shuffle_radar":
        batch["radar"]["b"] = (batch["radar"]["b"] + 1) % B
    elif control == "zero_radar":
        batch["radar"] = {k: v[:0] for k, v in batch["radar"].items()}
    elif control == "shuffle_cam":
        batch["prob"], batch["ctx"] = torch.roll(batch["prob"], 1, 0), torch.roll(batch["ctx"], 1, 0)
    elif control == "zero_cam":
        batch["prob"], batch["ctx"] = torch.zeros_like(batch["prob"]), torch.zeros_like(batch["ctx"])
    else:
        raise ValueError(f"unknown control {control}")


def evaluate_run(run, n=0, prior=0, images=0, timing=False, batch_size=4, workers=4, device="cuda", warm=False, on_train=False,
                 control="", ckpt_dir=None):
    """warm=False: independent samples with an empty world memory ("cold", the default of every table so far); warm=True:
    the real deployment, every val scene run in time order with the world memory carried from keyframe to keyframe over the
    WHOLE scene (batch 1; a keyframe whose predecessor is not the previous sample starts with an empty memory); the files
    get the suffix _warm (metrics_warm.json, ...) and the run's cold files stay."""
    device = device if torch.cuda.is_available() else "cpu"
    os.chdir(ROOT)
    cfg, model, loss, step, ckpt = load_run(run, device, ckpt_dir)
    cfg.batch_size, cfg.workers = (1 if warm else batch_size), workers
    cfg.seq_len = 1                         # the loader yields single samples; with warm=True the state is carried below
                                            # (what a clip-trained model gains from the memory over 3 keyframes: eval_memory.py)
    tag = ("_warm" if warm else "") + ("_train" if on_train else "") + (f"_ctl_{control}" if control else "")   # on_train: n TRAIN samples (overfitting check)
    if not cfg.radar_cache and os.path.isdir("data/cache/radar"):
        cfg.radar_cache = "data/cache/radar"                    # exact (feature_cache radar-selftest): only faster
    train_ds, val_ds = make_datasets(cfg)
    ds = val_subset(train_ds if on_train else val_ds, n) if n else val_ds
    loader = make_loader(ds, cfg, False)
    spec = model.spec
    grid = model.grid
    nusc = val_ds.nusc
    scene_of = {}
    meters = {"fov": SegMeter(grid), "seen": SegMeter(grid), "obj": SegMeter(grid)}
    det = DetMeter(classes=[g for g, _ in NUSC_GROUPS])
    use_refined = bool(cfg.model.get("refine")) and hasattr(model, "boxes")           # S5c refiner: second box set
    det_ref = DetMeter(classes=[g for g, _ in NUSC_GROUPS]) if use_refined else None
    pri = {"fov": SegMeter(grid), "seen": SegMeter(grid), "obj": SegMeter(grid)} if prior else None
    pmaps = prior_maps(train_ds, prior, spec) if prior else None
    scenes = {}                                                  # scene -> {"seg": {(layer, subset): SegMeter}, "det": frames}
    img_idx = set(np.linspace(0, len(ds) - 1, images).round().astype(int).tolist()) if images else set()
    out_dir = os.path.join(run, "eval")
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    tot, nb, i0, t0 = {}, 0, 0, time.time()
    state, prev_batch, prev_tok = None, None, None
    with torch.no_grad():
        for batch in loader:
            kw = {}
            if control:
                perturb(batch, control)
            if warm:                                                  # carry the world memory along the scene
                tok0 = batch["tokens"][0]
                cont = prev_tok is not None and nusc.get("sample", tok0)["prev"] == prev_tok
                if not cont:
                    state = None
                kw = {"state": state, "step": clip_step(prev_batch, device) if (cont and state is not None) else None}
            out, res, gts = forward_loss(model, loss, batch, 1.0, device, cfg.amp, cfg, train_terms=False, **kw)
            if warm:
                state, prev_batch, prev_tok = detach_tree(res["state"]), batch, batch["tokens"][0]
            for key, v in out.items():
                if key == "total" or key.startswith("group."):
                    tot[key] = tot.get(key, 0.0) + float(v)
            nb += 1
            B = len(batch["tokens"])
            refined = ({dt: model.boxes(res, dt, k=100, threshold=0.05, rescore=True) for dt in res["out"]}
                       if use_refined else {})
            for b in range(B):
                tok = batch["tokens"][b]
                scene = scene_of.setdefault(tok, nusc.get("scene", nusc.get("sample", tok)["scene_token"])["name"])
                rec = scenes.setdefault(scene, {"seg": {}, "det": {}, "detr": {}})
                per_dt = {}
                for dt, p in res["out"].items():
                    g = gts[dt][b]
                    mass = p["mass"][b, 0].float().cpu().numpy() if (p.get("mass") is not None and abs(dt) > 1e-6) else None
                    prob, dist = predictions(p, b, spec)
                    add_sample(meters, prob, dist, g, spec, dt, mass)
                    dets = decode({k: v[b:b + 1].float().cpu() for k, v in p["obj"].items()}, grid, NUSC_GROUPS,
                                  k=100, threshold=0.05)[0]
                    gb = gt_boxes(g)
                    det.add(dets, gb, dt=dt)
                    rec["det"].setdefault(dt, []).append(([d for d in dets], gb))
                    if use_refined:
                        det_ref.add(refined[dt][b], gb, dt=dt)
                        rec["detr"].setdefault(dt, []).append(([d for d in refined[dt][b]], gb))
                    if abs(dt) < 1e-6:                        # per-scene headline records (dt 0)
                        for layer, sub in HEADLINE_SEG:
                            sm = rec["seg"].setdefault(layer, SegMeter(grid, tolerances=(0.0, 0.5)))
                            m = masks(g)
                            if layer.endswith(":zc"):
                                ln = layer[:-3]
                                sm.add_lines({ln: dist[ln]}, {ln: np.asarray(g[ln + "_dist"], np.float32)}, mask=m["fov"])
                            else:
                                gkey = layer_index(spec)[layer][2]
                                sm.add({layer: prob[layer]}, {layer: np.asarray(g[gkey], np.float32)},
                                       mask=m["obj_fov"] if layer == "vehicle" else m["fov"])
                    if pri is not None:
                        add_sample(pri, pmaps[0], pmaps[1], g, spec, dt, mass)
                    per_dt[dt] = (prob, dist, refined[dt][b] if use_refined else dets, gb, mass)     # images: the final boxes
                if (i0 + b) in img_idx and 0.5 in per_dt:
                    g0 = gts[0.0][b]
                    prob, dist, dets, gb, _ = per_dt[0.0]
                    pn, dn, detn, gbn, massn = per_dt[0.5]
                    save_image(os.path.join(out_dir, "images", f"{i0 + b:05d}_{scene}.png"), tok, nusc,
                               os.path.expanduser(cfg.data_root), grid, prob, dist, g0, dets, gb, (pn, dn), detn,
                               gts[0.5][b], gbn, massn, f"{os.path.basename(run.rstrip('/'))} step {step}, val sample "
                               f"{i0 + b} ({scene}), EMA weights")
            i0 += B
            if nb % 50 == 0:
                print(f"{i0}/{len(ds)} samples, {time.time() - t0:.0f} s", flush=True)
    flat = report(seg=None, det=det)
    if use_refined:
        flat.update({"detr/" + k[len("det/"):]: v for k, v in report(seg=None, det=det_ref).items()})
    for name, m in meters.items():
        flat.update({f"seg_{name}/" + k: v for k, v in report(seg=m).items()})
    if pri is not None:
        for name, m in pri.items():
            flat.update({f"prior_{name}/" + k: v for k, v in report(seg=m).items()})
    flat.update({f"val_loss/{k}": v / max(1, nb) for k, v in tot.items()})
    flat["info/step"], flat["info/n_samples"], flat["info/n_scenes"] = step, i0, len(scenes)
    lat = None
    if timing:
        lat = latency(model, loader, device)
        flat.update({f"ms/{k}": v for k, v in lat.items() if not isinstance(v, dict)})
    with open(os.path.join(out_dir, f"metrics{tag}.json"), "w") as f:
        json.dump({"run": run, "checkpoint": os.path.basename(ckpt), "split": cfg.val_split, "weights": "EMA",
                   "memory": "carried over the whole scene" if warm else "empty (cold)",
                   "code": os.path.relpath(CODE, ROOT),
                   "metrics": flat, "latency": lat}, f, indent=1, default=float)
    seg_rec = {s: {layer: sm.c for layer, sm in r["seg"].items()} for s, r in scenes.items()}
    det_rec = {s: r["det"] for s, r in scenes.items()}
    detr_rec = {s: r["detr"] for s, r in scenes.items()}
    with open(os.path.join(out_dir, f"scene_records{tag}.pkl"), "wb") as f:
        pickle.dump({"seg": seg_rec, "det": det_rec, "detr": detr_rec}, f)
    table = headline_table(flat)
    with open(os.path.join(out_dir, f"table{tag}.txt"), "w") as f:
        f.write(table + "\n")
    print(table)
    print(f"done: {i0} val samples, {len(scenes)} scenes, {time.time() - t0:.0f} s -> {out_dir}")
    return flat


def latency(model, loader, device):
    """batch-1 forward of the cached model (camera heads excluded: they are a fixed cost of the pre-trained ResNet-50,
    timed separately) at dt 0 and at (-0.5, 0, +0.5); refused numbers on a busy GPU are flagged."""
    busy = gpu_busy()
    batch = next(iter(loader))
    from train import to_device
    b1 = to_device({k: (v[:1] if torch.is_tensor(v) and k not in ("ida",) else v) for k, v in batch.items()}, device)
    b1["radar"] = {k: v[batch["radar"]["b"] == 0].to(device) for k, v in batch["radar"].items()}
    b1["steps"] = [[tuple(t[:1].to(device) for t in s) for s in fr] for fr in batch["steps"]]
    b1["v"], b1["omega"] = batch["v"][:1].to(device), batch["omega"][:1].to(device)
    out = {"gpu_before": busy}
    out["bev_dt0_median"] = ms_per_frame(lambda: model(b1, dts=(0.0,)), n=50, warmup=10, batch1=False)["median"]
    out["bev_3dt_median"] = ms_per_frame(lambda: model(b1, dts=(-0.5, 0.0, 0.5)), n=50, warmup=10, batch1=False)["median"]
    if hasattr(model, "boxes") and model.cfg.refine:           # plus decoding and refining the boxes at dt 0
        out["bev_dt0_refined_boxes_median"] = ms_per_frame(lambda: model.boxes(model(b1, dts=(0.0,)), 0.0), n=50,
                                                           warmup=10, batch1=False)["median"]
    try:
        from camera_encoder import FrontCameraEncoder
        cam = FrontCameraEncoder().to(device).eval()
        x = torch.randn(1, 3, 256, 704, device=device)
        cv = torch.zeros(1, 27, device=device)
        out["camera_heads_median"] = ms_per_frame(lambda: cam.image_features(x, cv), n=30, warmup=5, batch1=False)["median"]
        del cam
    except Exception as e:                                    # noqa: BLE001
        out["camera_heads_error"] = str(e)
    out["gpu_after"] = gpu_busy()
    out["idle"] = bool(busy.get("util_percent", 100) < 10)
    return out


def time_run(run, device="cuda"):
    """latency of a run's chain (batch 1, EMA weights) -> runs/<name>/eval/latency.json; only valid on an idle GPU (the GPU
    load before and after is recorded, 'idle' says whether it was below 10 %). Does not touch metrics.json."""
    device = device if torch.cuda.is_available() else "cpu"
    os.chdir(ROOT)
    cfg, model, loss, step, ckpt = load_run(run, device, ckpt_dir)
    cfg.batch_size, cfg.workers, cfg.seq_len = 4, 2, 1
    if not cfg.radar_cache and os.path.isdir("data/cache/radar"):
        cfg.radar_cache = "data/cache/radar"
    _, val_ds = make_datasets(cfg)
    lat = latency(model, make_loader(val_subset(val_ds, 8), cfg, False), device)
    os.makedirs(os.path.join(run, "eval"), exist_ok=True)
    with open(os.path.join(run, "eval", "latency.json"), "w") as f:
        json.dump({"run": run, "checkpoint": os.path.basename(ckpt), "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
                   "latency": lat}, f, indent=1, default=float)
    print(json.dumps(lat, indent=1, default=float))
    return lat


def headline_table(flat):
    keys = []
    for pre in ("seg_fov", "prior_fov"):
        for dt in ("dt+0.00", "dt+0.50", "dt-0.50"):
            for layer, metric in (("drivable_area", "iou@0"), ("drivable_area", "iou@0.5"), ("ped_crossing", "iou@0"),
                                  ("walkway", "iou@0"), ("carpark_area", "iou@0"), ("lane_divider", "iou@0.5"),
                                  ("lane_divider", "iou@1"), ("road_divider", "iou@1"), ("lane_divider:zc", "f1@0.5"),
                                  ("road_divider:zc", "f1@0.5"), ("lane_divider:zc", "field_mae")):
                keys.append(f"{pre}/seg/{dt}/{layer}/all/all/{metric}")
    for pre in ("seg_obj", "prior_obj"):
        for dt in ("dt+0.00", "dt+0.50"):
            for layer in ("vehicle", "pedestrian", "two_wheeler"):
                keys.append(f"{pre}/seg/{dt}/{layer}/all/all/iou@0")
    for dp in ("det", "detr"):                                  # detr = boxes refined by S5c (only for models with a refiner)
        for dt in ("dt+0.00", "dt+0.50", "dt-0.50"):
            keys.append(f"{dp}/{dt}/mean/all/mAP")
            for cls in ("vehicle", "pedestrian", "two_wheeler"):
                for m in ("ap@2", "ap@4", "ate", "ase", "aoe", "ave", "n_gt"):
                    keys.append(f"{dp}/{dt}/{cls}/all/{m}")
    keys += [k for k in flat if k.startswith(("val_loss/", "ms/", "info/"))]
    lines = []
    for k in keys:
        if k in flat:
            v = flat[k]
            lines.append(f"{k:60s} {v:10.4f}" if isinstance(v, float) else f"{k:60s} {v!s:>10}")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ paired comparison
def _seg_value(counts, metric):
    """metric from summed SegMeter counts of one layer (subset all, bin all)."""
    tol = {"iou@0": 0, "iou@0.5": 1, "f1@0": 0, "f1@0.5": 1}[metric]
    tp_p, fp, tp_g, fn = counts[tol]
    if metric.startswith("iou"):
        tp = (tp_p + tp_g) / 2
        return tp / max(tp + fp + fn, 1e-9)
    p, r = tp_p / max(tp_p + fp, 1e-9), tp_g / max(tp_g + fn, 1e-9)
    return 2 * p * r / max(p + r, 1e-9)


def scene_counts(rec, layer):
    """scene -> (2, 4) counts of the layer (dt 0, subset all, bin all)."""
    out = {}
    for s, layers in rec["seg"].items():
        c = layers.get(layer, {})
        key = (0.0, layer, "all", "all")
        if key in c:
            out[s] = c[key]["cnt"]
    return out


def scene_flags(frames_by_scene, scenes, cls, thr=2.0):
    """scene -> (scores, true-positive flags, n_gt) of one class (front field of view, matching at thr metres)."""
    from evaluation.metrics import in_front
    out = {}
    for s in scenes:
        frames = [([d for d in ds if d["group"] == cls and in_front(d)], [g for g in gs if g["group"] == cls and in_front(g)])
                  for ds, gs in frames_by_scene[s]]
        m, n_gt = match(frames, thr)
        out[s] = (np.array([x[0] for x in m], float), np.array([x[1] for x in m], float), n_gt)
    return out


def ap_from_flags(flags, scenes):
    """nuScenes AP of the detections of the listed scenes (repeats allowed): one sort of the concatenated flags."""
    sc = np.concatenate([flags[s][0] for s in scenes])
    fl = np.concatenate([flags[s][1] for s in scenes])
    n_gt = sum(flags[s][2] for s in scenes)
    return ap_nuscenes(fl[np.argsort(-sc, kind="stable")], n_gt)


def det_ap(frames_by_scene, scenes, cls, thr=2.0):
    frames = [([d for d in ds if d["group"] == cls], [g for g in gs if g["group"] == cls])
              for s in scenes for ds, gs in frames_by_scene[s]]
    from evaluation.metrics import in_front
    frames = [([d for d in ds if in_front(d)], [g for g in gs if in_front(g)]) for ds, gs in frames]
    m, n_gt = match(frames, thr)
    return ap_nuscenes([x[1] for x in m], n_gt)


def compare(run_a, run_b, n_boot=1000, seed=0, det_a="det", det_b="det", dt=0.0):
    """paired scene bootstrap of the headline numbers: b - a, mean and 95 % interval. det_a / det_b: the box set of each
    run ('det' = plain decode, 'detr' = refined by S5c); dt: the target time of the detection numbers (the map numbers are
    at dt 0)."""
    ra = pickle.load(open(os.path.join(run_a, "eval", "scene_records.pkl"), "rb"))
    rb = pickle.load(open(os.path.join(run_b, "eval", "scene_records.pkl"), "rb"))
    rng = np.random.default_rng(seed)
    lines = [f"paired scene bootstrap ({n_boot} resamples), b - a; a = {run_a} ({det_a}), b = {run_b} ({det_b}); val; "
             f"maps at dt 0, boxes at dt {dt:+.1f}"]
    res = {}
    for layer, _ in HEADLINE_SEG:
        metric = SEG_METRIC[layer]
        ca, cb = scene_counts(ra, layer), scene_counts(rb, layer)
        sc = sorted(set(ca) & set(cb))
        A, Bc = np.stack([ca[s] for s in sc]), np.stack([cb[s] for s in sc])
        point = _seg_value(Bc.sum(0), metric) - _seg_value(A.sum(0), metric)
        ds = []
        for _ in range(n_boot):
            ix = rng.integers(0, len(sc), len(sc))
            ds.append(_seg_value(Bc[ix].sum(0), metric) - _seg_value(A[ix].sum(0), metric))
        lo, hi = np.percentile(ds, [2.5, 97.5])
        res[f"{layer} {metric}"] = (point, lo, hi, _seg_value(A.sum(0), metric), _seg_value(Bc.sum(0), metric))
    fa = {s: r[dt] for s, r in ra[det_a].items() if dt in r}
    fb = {s: r[dt] for s, r in rb[det_b].items() if dt in r}
    sc = sorted(set(fa) & set(fb))
    for cls in ("vehicle", "pedestrian"):
        # the greedy matching of a detection only involves its own frame, so the true-positive flags of every scene are fixed:
        # compute them once, a resample only re-sorts the concatenated flags (equal to re-running det_ap, but fast)
        fl_a, fl_b = scene_flags(fa, sc, cls), scene_flags(fb, sc, cls)
        a0, b0 = ap_from_flags(fl_a, sc), ap_from_flags(fl_b, sc)
        ds = []
        for _ in range(n_boot):
            ix = rng.integers(0, len(sc), len(sc))
            s_ix = [sc[i] for i in ix]
            ds.append(ap_from_flags(fl_b, s_ix) - ap_from_flags(fl_a, s_ix))
        lo, hi = np.percentile(ds, [2.5, 97.5])
        res[f"{cls} AP@2"] = (b0 - a0, lo, hi, a0, b0)
    for k, (d, lo, hi, a, b) in res.items():
        lines.append(f"{k:28s} a {a:.4f}  b {b:.4f}  b-a {d:+.4f}  95% [{lo:+.4f}, {hi:+.4f}]"
                     + ("  (excludes 0)" if lo > 0 or hi < 0 else ""))
    txt = "\n".join(lines)
    print(txt)
    return res, txt


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?")
    ap.add_argument("--n", type=int, default=0, help="val samples spread over all scenes (0 = all)")
    ap.add_argument("--prior", type=int, default=0, help="train samples of the prior-map baseline (0 = off)")
    ap.add_argument("--images", type=int, default=0)
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--time-only", action="store_true", help="only the latency -> eval/latency.json (idle GPU)")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--on-train", action="store_true", help="evaluate --n TRAIN samples instead of val (overfitting check, files *_train)")
    ap.add_argument("--control", default="", choices=["", "shuffle_radar", "zero_radar", "shuffle_cam", "zero_cam"],
                    help="sanity control: perturb one input at evaluation (files *_ctl_<name>)")
    ap.add_argument("--ckpt-dir", default=None, help="read the checkpoint from this folder (e.g. the D: copy of a moved run)")
    ap.add_argument("--warm", action="store_true", help="carry the world memory over whole scenes (batch 1, time order) -> *_warm files")
    ap.add_argument("--compare", nargs=2)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--det-a", default="det", help="box set of run a: det (plain) or detr (refined)")
    ap.add_argument("--det-b", default="det")
    ap.add_argument("--dt", type=float, default=0.0, help="target time of the detection comparison")
    a = ap.parse_args()
    if a.time_only:
        time_run(a.run)
    elif a.compare:
        os.chdir(ROOT)
        compare(*a.compare, n_boot=a.boot, det_a=a.det_a, det_b=a.det_b, dt=a.dt)
    else:
        evaluate_run(a.run, a.n, a.prior, a.images, a.time, a.batch, a.workers, warm=a.warm, on_train=a.on_train, control=a.control, ckpt_dir=a.ckpt_dir)
