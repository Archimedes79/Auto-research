"""Observability report of a training run: what every stage of the chain produces on every N-th validation frame.

    wsl ~/miniconda3/envs/gcar/bin/python evaluation/observe_run.py runs/t1_r4_soa [--every 20] [--max 12] [--device cuda]
    wsl ~/miniconda3/envs/gcar/bin/python evaluation/observe_run.py runs/t1_r4_soa --compare runs/t1_r0   (two runs, same frames)

The run is loaded as evaluate_run.py does (pinned code snapshot, EMA weights); the frames are run one by one through the
model exactly as in validation (for sequence-trained runs the two previous keyframes first, so the world memory is warm).
Forward hooks record the stages (nothing is recomputed on the side), and per frame two pictures are written:
    stages.png   input -> camera heads -> lift -> temporal fusion -> radar branch -> fusion / trunk (energy = L2 norm over
                 the channels of a feature map; the GT boxes are drawn dashed on every BEV panel, so an alignment error of a
                 stage is visible at once)
    outputs.png  GT | prediction | errors | motion | the same at dt = +-0.5 s (with the splat's "unknown" cells) | boxes
plus a table of automatic plausibility checks (heuristic thresholds: they flag, a human judges) and index.html that lists
all frames with a guide to every panel (what it should look like, what is a red flag). Output: runs/<name>/observe/.
Numbers in the checks come from the same tensors as the pictures; the pictures are not part of git.
"""
import argparse
import html
import json
import math
import os
import sys
import textwrap
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate_run as ev  # noqa: E402  (sets sys.path to the run's pinned code snapshot)
from evaluate_run import (COLORS, NUSC_GROUPS, composite, decode, draw_boxes, gt_boxes, layer_index,  # noqa: E402
                          load_run, masks, predictions, show)
from video_encode import encode  # noqa: E402
from train import clip_step, collate, collate_cached, forward_loss, make_datasets, to_device  # noqa: E402

ROOT = ev.ROOT


def bev_axes(ax, grid, title):
    ev.bev_axes(ax, grid, textwrap.fill(title, 46))


# ------------------------------------------------------------------------------------------------ hooks
class Tap:
    """Records the stages of one forward pass by hooks on the model's own modules (the last call wins; a list for the
    modules called once per sample / frame)."""

    def __init__(self, model):
        self.d, self.handles = {}, []
        self.model = model

        def out_hook(name, pick=lambda o: o, many=False):
            def h(mod, args, out):
                v = pick(out).detach().float().cpu()
                if many:
                    self.d.setdefault(name, []).append(v)
                else:
                    self.d[name] = v
            return h

        def pre_hook(name):
            def h(mod, args):
                self.d[name] = args[0].detach().float().cpu()
            return h

        def reg(mod, hook, pre=False):
            if mod is not None:
                self.handles.append(mod.register_forward_pre_hook(hook) if pre else mod.register_forward_hook(hook))
        reg(model.lift, out_hook("lift", many=True))
        reg(model.cam_reduce, out_hook("cam_frames", many=True))
        reg(model.cam_temporal, out_hook("cam_temporal", pick=lambda o: o[0]))
        reg(model.cam_temporal, lambda m, a, o: self.d.__setitem__(
            "cam_aux", {k: v.detach().float().cpu() for k, v in o[2].items() if torch.is_tensor(v)}) if len(o) > 2 else None)
        reg(model.radar, out_hook("radar_feat"))
        net = getattr(model.radar, "net", None)
        if net is not None:
            reg(net, pre_hook("radar_raster"), pre=True)
        reg(model.fusion, out_hook("fused"))

    def reset(self):
        self.d = {}

    def close(self):
        for h in self.handles:
            h.remove()


# ------------------------------------------------------------------------------------------------ map helpers
def energy(t):
    """(C, X, Y) -> (X, Y) L2 norm over the channels."""
    return np.sqrt((np.asarray(t, np.float64) ** 2).sum(0)).astype(np.float32)


def pca_rgb(t):
    """(C, X, Y) -> (X, Y, 3): first three principal components of the per-cell feature vectors, scaled to [0, 1]."""
    C, X, Y = t.shape
    f = np.asarray(t, np.float64).reshape(C, -1).T
    f = f - f.mean(0)
    _, _, vt = np.linalg.svd(f[:: max(1, len(f) // 4000)], full_matrices=False)
    p = f @ vt[:3].T
    lo, hi = np.percentile(p, 2, 0), np.percentile(p, 98, 0)
    return np.clip((p - lo) / np.maximum(hi - lo, 1e-9), 0, 1).reshape(X, Y, 3).astype(np.float32)


def draw_map(ax, grid, img, title, boxes=None, cmap="magma", vmax=None, cbar=True, fig=None, vmin=0.0):
    if img.ndim == 2:
        vm = vmax if vmax is not None else max(float(np.percentile(img, 99.5)), 1e-6)
        im = ax.imshow(img, origin="lower", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vm,
                       extent=(grid.y_min, grid.y_max, grid.x_min, grid.x_max), aspect="auto")
        if cbar and fig is not None:
            fig.colorbar(im, ax=ax, fraction=0.04, pad=0.01).ax.tick_params(labelsize=6)
    else:
        show(ax, img, grid)
    bev_axes(ax, grid, title)
    if boxes:
        draw_boxes(ax, boxes, "cyan", lw=0.7, ls="--")


def camera_image(nusc, token, root, ida, hw):
    """CAM_FRONT warped like the model's input (ida: original pixel -> model pixel), a PIL image of size hw[::-1]."""
    from PIL import Image
    sd = nusc.get("sample_data", nusc.get("sample", token)["data"]["CAM_FRONT"])
    im = Image.open(os.path.join(root, sd["filename"])).convert("RGB")
    H, W = int(hw[0]), int(hw[1])
    try:
        inv = np.linalg.inv(np.asarray(ida, np.float64))
        return im.transform((W, H), Image.AFFINE, data=tuple(inv[:2].flatten()), resample=Image.BILINEAR)
    except Exception:                                          # noqa: BLE001
        return im.resize((W, H))


# ------------------------------------------------------------------------------------------------ one frame
def run_frame(model, loss, cfg, ds, i, device, cold):
    """(batch of the target frame, model result, GTs, tap dict) of val sample i, through its clip when sequence-trained."""
    toks = [ds.samples[i]]
    pos = {t: k for k, t in enumerate(ds.samples)}
    while not cold and len(toks) < max(1, cfg.seq_len):
        prev = ds.nusc.get("sample", toks[0])["prev"]
        if not prev or prev not in pos:
            break
        toks.insert(0, prev)
    fn = collate_cached if cfg.feature_cache else collate
    tap = Tap(model)
    state, prev_batch, res = None, None, None
    try:
        for t, tok in enumerate(toks):
            batch = fn([ds[pos[tok]]])
            step = clip_step(prev_batch, device) if prev_batch is not None else None
            tap.reset()
            with torch.no_grad():
                _, res, gts = forward_loss(model, loss, batch, 1.0, device, cfg.amp, cfg, train_terms=False,
                                           state=state, step=step)
            state, prev_batch = res["state"], batch
    finally:
        tap.close()
    with torch.no_grad():
        vis = model.camera_visibility(to_device(batch, device), res["features"]) if hasattr(model, "camera_visibility") else None
    return batch, res, gts, tap.d, (None if vis is None else vis[0, 0].cpu().numpy()), len(toks)


def stats(name, t):
    a = np.asarray(t, np.float64)
    return {"stage": name, "shape": "x".join(str(s) for s in a.shape), "mean": float(a.mean()), "std": float(a.std()),
            "abs_max": float(np.abs(a).max()), "nonzero": float((a != 0).mean()), "finite": bool(np.isfinite(a).all())}


# ------------------------------------------------------------------------------------------------ pictures
def stages_figure(path, model, cfg, grid, batch, res, gts, tap, vis, nusc, root, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g0 = gts[0.0][0]
    boxes = gt_boxes(g0)
    cam = np.asarray(g0["camera"], np.float32) > 0
    fig, axs = plt.subplots(3, 4, figsize=(19, 14))
    # row 1 ---- input and camera heads
    from camera_encoder import depth_centres
    prob = batch["prob"][0, 0].float().numpy() if "prob" in batch else None
    img = camera_image(nusc, batch["tokens"][0], os.path.expanduser(cfg.data_root), batch["ida"], batch["hw"])
    axs[0, 0].imshow(img)
    axs[0, 0].set_title("1 CAM_FRONT as the model sees it (crop)", fontsize=9)
    axs[0, 0].axis("off")
    if prob is not None:
        dc = depth_centres().numpy()
        exp_d = (prob * dc[:, None, None]).sum(0)
        ent = -(prob * np.log(prob + 1e-9)).sum(0) / math.log(prob.shape[0])
        axs[0, 1].imshow(img)
        im = axs[0, 1].imshow(np.kron(exp_d, np.ones((16, 16))), alpha=0.55, cmap="turbo", vmin=1, vmax=60,
                              extent=(0, img.size[0], img.size[1], 0))
        fig.colorbar(im, ax=axs[0, 1], fraction=0.04).ax.tick_params(labelsize=6)
        axs[0, 1].set_title("2 camera head: expected depth per pixel group (m)", fontsize=9)
        axs[0, 1].axis("off")
        im = axs[0, 2].imshow(ent, cmap="viridis", vmin=0, vmax=1)
        fig.colorbar(im, ax=axs[0, 2], fraction=0.04).ax.tick_params(labelsize=6)
        axs[0, 2].set_title("3 depth uncertainty (entropy / max; 0 = sure, 1 = flat)", fontsize=9)
    else:
        axs[0, 1].axis("off")
        axs[0, 2].axis("off")
    rad = batch["radar"]
    ax = axs[0, 3]
    sc = ax.scatter(rad["y"].numpy(), rad["x"].numpy(), c=rad["vr"].numpy(), s=np.clip(rad["rcs"].numpy() + 20, 4, 60),
                    cmap="coolwarm", vmin=-15, vmax=15)
    fig.colorbar(sc, ax=ax, fraction=0.04).ax.tick_params(labelsize=6)
    bev_axes(ax, grid, f"4 radar detections of the window ({len(rad['x'])}), colour = Doppler vr (m/s)")
    draw_boxes(ax, boxes, "k", lw=0.7, ls="--")
    # row 2 ---- camera BEV
    lift = tap.get("lift")
    if lift:
        v = lift[0][0].numpy()                                      # frame 0: (nz * C, X, Y)
        e_all = energy(v)
        draw_map(axs[1, 0], grid, e_all, "5 camera BEV after the lift (feature energy, all heights)", boxes, fig=fig)
        nz = len(model.cfg.z_levels)
        per = v.reshape(nz, -1, v.shape[-2], v.shape[-1])
        e_z = [float((energy(per[k]) * cam).sum() / max(cam.sum(), 1)) for k in range(nz)]
        axs[1, 1].bar([f"{z:g}" for z in model.cfg.z_levels], e_z, color="tab:blue")
        axs[1, 1].set_title("6 where the energy sits in height (mean over camera cells; z in m)", fontsize=9)
        axs[1, 1].tick_params(labelsize=7)
    cf = tap.get("cam_frames")
    if cf:
        draw_map(axs[1, 2], grid, energy(cf[0][0].numpy()), "7 camera BEV after reduce, current frame (energy)", boxes,
                 fig=fig)
    ct = tap.get("cam_temporal")
    if ct is not None:
        draw_map(axs[1, 3], grid, energy(ct[0].numpy()), "8 camera BEV after temporal fusion (energy)", boxes, fig=fig)
        if vis is not None:
            axs[1, 3].contour(np.linspace(grid.y_min, grid.y_max, vis.shape[1]), np.linspace(grid.x_min, grid.x_max,
                              vis.shape[0]), vis, levels=[0.5], colors="lime", linewidths=0.7)
    # row 3 ---- temporal change, radar branch, fusion
    aux = tap.get("cam_aux")
    if aux is not None and "agg" in aux:
        conf = aux["mem_conf"][0, 0].numpy() if "mem_conf" in aux else None
        draw_map(axs[2, 0], grid, energy(aux["agg"][0].numpy()),
                 "9 what the past frames supply: window frames moved to now by ego motion + velocity head (energy; green = "
                 "world memory confident)", boxes, fig=fig)
        if conf is not None and conf.max() > 0.5:
            axs[2, 0].contour(np.linspace(grid.y_min, grid.y_max, conf.shape[1]), np.linspace(grid.x_min, grid.x_max, conf.shape[0]),
                              conf, levels=[0.5], colors="lime", linewidths=0.7)
    rr = tap.get("radar_raster")
    if rr is not None:
        draw_map(axs[2, 1], grid, rr[0, 0].numpy(), "10 radar raster: detections per cell (count channel)", boxes,
                 cmap="viridis", fig=fig)
    rf = tap.get("radar_feat")
    if rf is not None:
        draw_map(axs[2, 2], grid, energy(rf[0].numpy()), "11 radar branch features (energy)", boxes, fig=fig)
    S = res["features"][0].float().cpu().numpy()
    draw_map(axs[2, 3], grid, pca_rgb(S), "12 trunk output: feature PCA as RGB (same colour = same kind of cell)", boxes)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=70)
    plt.close(fig)


def outputs_figure(path, model, cfg, grid, batch, res, gts, nusc, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    spec = model.spec
    k = layer_index(spec)
    g0 = gts[0.0][0]
    gtb = gt_boxes(g0)
    p0 = res["out"][0.0]
    prob, dist = predictions(p0, 0, spec)
    gt_prob = {n: np.asarray(g0[k[n][2]], np.float32) for n in prob}
    gt_dist = {n: np.asarray(g0[n + "_dist"], np.float32) for n in dist}
    dets = decode({a: b[:1].float().cpu() for a, b in p0["obj"].items()}, grid, NUSC_GROUPS, k=100, threshold=0.05)[0]
    cam = np.asarray(g0["camera"], np.float32) > 0
    fig, axs = plt.subplots(3, 4, figsize=(19, 14))
    show(axs[0, 0], composite(gt_prob, gt_dist), grid)
    bev_axes(axs[0, 0], grid, "13 ground truth, dt 0 (areas, lines, vehicles)")
    draw_boxes(axs[0, 0], gtb, "tab:blue")
    show(axs[0, 1], composite(prob, dist), grid)
    bev_axes(axs[0, 1], grid, "14 prediction, dt 0 (red boxes: score >= 0.3, dashed black: GT)")
    draw_boxes(axs[0, 1], gtb, "k", lw=0.7, ls="--")
    draw_boxes(axs[0, 1], dets, "tab:red", score_min=0.3)
    P, G = prob["drivable_area"] > 0.5, np.asarray(g0["drivable_area"], np.float32) > 0.5
    err = np.ones(P.shape + (3,), np.float32)
    err[P & G & cam] = (0.7, 0.7, 0.7)
    err[P & ~G & cam] = (0.9, 0.2, 0.2)
    err[~P & G & cam] = (0.2, 0.4, 0.95)
    err[~cam] = (0.95, 0.95, 0.88)
    show(axs[0, 2], err, grid)
    bev_axes(axs[0, 2], grid, "15 drivable area: right grey, false red, missed blue, beige = not seen by the camera")
    pv = prob.get("vehicle")
    gv = np.asarray(g0["vehicle"], np.float32)
    rgb = np.ones(pv.shape + (3,), np.float32)
    rgb[..., 0] = 1 - pv * 0.8
    rgb[..., 1] = 1 - np.maximum(pv, gv) * 0.6
    rgb[..., 2] = 1 - gv * 0.8
    show(axs[0, 3], rgb, grid)
    bev_axes(axs[0, 3], grid, "16 vehicle footprint: prediction red, GT blue, both = dark")
    for col, dt in ((0, 0.5), (1, -0.5)):
        ax = axs[1, col]
        p = res["out"][dt]
        pr, ds = predictions(p, 0, spec)
        show(ax, composite(pr, ds), grid)
        mass = p["mass"][0, 0].float().cpu().numpy()
        shade = np.zeros(mass.shape + (4,), np.float32)
        shade[mass < 0.5] = (0.35, 0.25, 0.0, 0.55)
        ax.imshow(shade, origin="lower", interpolation="nearest", extent=(grid.y_min, grid.y_max, grid.x_min, grid.x_max),
                  aspect="auto")
        bev_axes(ax, grid, f"{17 + col} prediction at dt {dt:+.1f} s; brown = unknown cells (splat mass < 0.5); dashed = GT then")
        draw_boxes(ax, gt_boxes(gts[dt][0]), "k", lw=0.7, ls="--")
        dd = decode({a: b[:1].float().cpu() for a, b in p["obj"].items()}, grid, NUSC_GROUPS, k=100, threshold=0.05)[0]
        draw_boxes(ax, dd, "tab:red", score_min=0.3)
    mass = res["out"][0.5]["mass"][0, 0].float().cpu().numpy()
    draw_map(axs[1, 2], grid, mass, "19 splat mass at dt +0.5 s (1 = a source cell lands here, 0 = nothing does)", None,
             cmap="viridis", fig=fig, vmax=1.5)
    sc = np.array([d["score"] for d in dets])
    ax = axs[1, 3]
    ax.hist(sc, bins=np.linspace(0, 1, 21), color="tab:red")
    ax.set_yscale("log")
    ax.set_title(textwrap.fill(f"20 detection scores (n >= 0.3: {int((sc >= 0.3).sum())}; GT boxes in view: {len(gtb)})", 46),
                 fontsize=9)
    ax.tick_params(labelsize=7)
    # motion: GT velocity (m/s, on the object cells) against the predicted velocity field
    vel = p0["v_hat"][0].float().cpu().numpy() if p0.get("v_hat") is not None else None
    gvel = np.asarray(g0["velocity"], np.float32) if "velocity" in g0 else None
    veh = np.asarray(g0["vehicle"], np.float32) > 0.5
    gsp = np.hypot(gvel[0], gvel[1]) if gvel is not None else None
    vmax = max(2.0, float(np.percentile(gsp[veh], 99)) if gsp is not None and veh.any() else 2.0)
    if gsp is not None:
        draw_map(axs[2, 0], grid, np.where(veh, gsp, 0), "21 GT speed |v| of the vehicle cells (m/s)", gtb, cmap="viridis", fig=fig,
                 vmax=vmax)
    if vel is not None:
        sp = np.hypot(vel[0], vel[1])
        draw_map(axs[2, 1], grid, np.where(veh, sp, 0), "22 predicted speed |v| on the same cells (same colour scale)", gtb,
                 cmap="viridis", fig=fig, vmax=vmax)
        draw_map(axs[2, 2], grid, sp, f"23 predicted speed everywhere (scale to its own max {sp.max():.2f} m/s)", gtb, cmap="viridis",
                 fig=fig)
        ax = axs[2, 3]
        if gsp is not None and veh.any():
            ax.scatter(gsp[veh], sp[veh], s=4, alpha=0.4)
            top = max(vmax, float(sp[veh].max()), 1.0)
            ax.plot([0, top], [0, top], "k--", lw=0.8)
            ax.set_xlabel("GT speed (m/s)", fontsize=8)
            ax.set_ylabel("predicted speed (m/s)", fontsize=8)
        ax.set_title(textwrap.fill("24 speed on vehicle cells: predicted against GT (diagonal = perfect)", 46), fontsize=9)
        ax.tick_params(labelsize=7)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=70)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ checks
def checks(model, cfg, grid, batch, res, gts, tap, vis):
    """[(name, value, status 'ok' / 'warn' / 'bad', what it means)]; thresholds are heuristics that flag, not judge."""
    out = []

    def add(name, val, status, note):
        out.append((name, val, status, note))
    g0 = gts[0.0][0]
    cam = np.asarray(g0["camera"], np.float32) > 0
    for name, t in [("camera lift", tap.get("lift", [None])[0] if tap.get("lift") else None),
                    ("camera temporal", tap.get("cam_temporal")), ("radar branch", tap.get("radar_feat")),
                    ("trunk output", res["features"].float().cpu())]:
        if t is None:
            continue
        a = np.asarray(t)
        fin = bool(np.isfinite(a).all())
        dead = float((a.reshape(-1, *a.shape[-2:]).std((1, 2)) < 1e-8).mean())
        add(f"{name}: finite, dead channels", f"{'finite' if fin else 'NaN/Inf'}, {dead:.0%} dead",
            "bad" if not fin else ("warn" if dead > 0.5 else "ok"), "no NaN; most channels should vary over the map")
    lift = tap.get("lift")
    if lift and vis is not None:
        e = energy(lift[0][0].numpy())
        out_frac = float(e[vis < 0.5].sum() / max(e.sum(), 1e-9))
        add("camera BEV energy outside the camera's field of view", f"{out_frac:.1%}", "ok" if out_frac < 0.05 else "warn",
            "the lift can only fill cells that project into the image")
    if "prob" in batch:
        p = batch["prob"][0, 0].float().numpy()
        from camera_encoder import depth_centres
        ed = (p * depth_centres().numpy()[:, None, None]).sum(0)
        add("depth head: median expected depth", f"{np.median(ed):.1f} m", "ok" if 4 < np.median(ed) < 40 else "warn",
            "a road scene: pixel groups mostly 5-40 m away")
    rad = batch["radar"]
    n = len(rad["x"])
    from evaluation.metrics import in_front
    boxes = [b for b in gt_boxes(g0) if in_front(b)]                  # the field of view DetMeter scores
    inside = 0
    for x, y in zip(rad["x"].numpy(), rad["y"].numpy()):
        for b in boxes:
            yaw = b.get("yaw", 0.0)
            c, s = math.cos(yaw), math.sin(yaw)
            dx, dy = x - b["cx"], y - b["cy"]
            u, w = c * dx + s * dy, -s * dx + c * dy
            hl, hw = (b["length"] / 2, b["width"] / 2) if "length" in b else (b["radius"], b["radius"])
            if abs(u) <= hl + 1.0 and abs(w) <= hw + 1.0:
                inside += 1
                break
    add("radar detections", f"{n} in window", "ok" if n > 20 else "warn", "forward radars: tens to hundreds per 0.5 s")
    add("radar detections on or within 1 m of a GT box", f"{inside / max(n, 1):.0%}", "ok",
        "context only: most radar points are static structure, not objects")
    aux, cf = tap.get("cam_aux"), tap.get("cam_frames")
    if aux is not None and cf and "agg" in aux:
        share = float(energy(aux["agg"][0].numpy()).sum() / max(energy(cf[0][0].numpy()).sum(), 1e-9))
        add("temporal fusion: energy of the moved past frames / current frame", f"{share:.2f}", "ok" if 0.05 < share < 3 else "warn",
            "~0: the window supplies nothing (frames missing or splat empty); >3: the past dominates")
        if "mem_conf" in aux:
            cov = float((aux["mem_conf"] > 0.5).float().mean())
            add("world memory: cells it is confident about", f"{cov:.0%}", "ok",
                "0 % on the first keyframe of a clip, grows as the clip runs (the camera cone fills in)")
    spec = model.spec
    k = layer_index(spec)
    prob, _ = predictions(res["out"][0.0], 0, spec)
    gd = float((np.asarray(g0["drivable_area"], np.float32) > 0.5)[cam].mean())
    pd = float((prob["drivable_area"] > 0.5)[cam].mean())
    add("drivable area, share of camera cells: prediction vs GT", f"{pd:.2f} vs {gd:.2f}", "ok" if abs(pd - gd) < 0.15 else "warn",
        "a gross offset means a systematic bias")
    P, G = prob["drivable_area"] > 0.5, np.asarray(g0["drivable_area"], np.float32) > 0.5
    iou = float((P & G & cam).sum() / max((P | G)[cam].sum(), 1))
    add("drivable area IoU on camera cells", f"{iou:.2f}", "ok" if iou > 0.5 else "warn", "this frame only")
    dets = [d for d in decode({a: b[:1].float().cpu() for a, b in res["out"][0.0]["obj"].items()}, grid, NUSC_GROUPS, k=100,
                              threshold=0.3)[0] if in_front(d)]
    left = list(boxes)
    hit = 0
    for d in sorted(dets, key=lambda d: -d["score"]):
        j = min(range(len(left)), key=lambda q: math.hypot(left[q]["cx"] - d["cx"], left[q]["cy"] - d["cy"]), default=None)
        if j is not None and math.hypot(left[j]["cx"] - d["cx"], left[j]["cy"] - d["cy"]) < 2.0:
            left.pop(j)
            hit += 1
    add("boxes in the front field of view (score >= 0.3, centre within 2 m): found / detected / GT", f"{hit} / {len(dets)} / {len(boxes)}",
        "ok" if hit >= 0.5 * len(boxes) and len(dets) <= 2 * max(hit, 1) + 2 else "warn",
        "far and small objects are in the GT: recall is not expected to be 100 %")
    for dt in (0.5, -0.5):
        if dt in res["out"]:
            m = res["out"][dt]["mass"][0, 0].float().cpu().numpy()
            add(f"splat: known cells at dt {dt:+.1f} s", f"{(m >= 0.5).mean():.1%}", "ok" if 0.3 < (m >= 0.5).mean() < 0.999 else "warn",
                "cells with no source: the ego drives into new ground, a few percent must be unknown")
    p0 = res["out"][0.0]
    if p0.get("v_hat") is not None and "velocity" in g0:
        v = p0["v_hat"][0].float().cpu().numpy()
        gv = np.asarray(g0["velocity"], np.float32)
        sp, gsp = np.hypot(v[0], v[1]), np.hypot(gv[0], gv[1])
        veh = np.asarray(g0["vehicle"], np.float32) > 0.5
        mov = veh & (gsp > 1.0)
        if mov.any():
            r = float(sp[mov].mean() / max(gsp[mov].mean(), 1e-6))
            add("velocity head: predicted / GT speed on moving vehicle cells", f"{sp[mov].mean():.2f} / {gsp[mov].mean():.2f} m/s",
                "ok" if r > 0.5 else ("warn" if r > 0.2 else "bad"), "the head should reproduce the speed of moving vehicles")
        else:
            add("velocity head: |v| on vehicle cells / elsewhere", f"{sp[veh].mean() if veh.any() else 0:.2f} / {sp[~veh].mean():.2f} m/s",
                "ok", "no moving vehicle in this frame: both should be near 0")
    return out


# ------------------------------------------------------------------------------------------------ html
GUIDE = """
<h2>How to read the pictures</h2>
<table>
<tr><th>#</th><th>panel</th><th>what you should see</th><th>red flag</th></tr>
<tr><td>1</td><td>input image</td><td>the front camera, cropped and scaled like the network input</td><td>black, shifted, wrong camera</td></tr>
<tr><td>2-3</td><td>camera head</td><td>depth grows from the bottom of the image upwards; the road is sure (dark in 3), sky and object edges are uncertain</td><td>depth flat, or the near road far away</td></tr>
<tr><td>4</td><td>radar points</td><td>points in front of the car, dashed GT boxes of vehicles; moving cars carry a Doppler colour that differs from their surroundings</td><td>all points one colour in a busy scene, points far outside the GT</td></tr>
<tr><td>5-6</td><td>lift</td><td>energy only inside the camera cone (a wedge), strongest in the visible near field; the bar chart: which heights carry the energy (road level and vehicles)</td><td>energy outside the wedge, one single height, empty near field</td></tr>
<tr><td>7-8</td><td>camera BEV, temporal</td><td>the same wedge; after the temporal fusion the cone is filled further and smoother; the dashed GT boxes sit on energy blobs</td><td>energy that does not follow the GT boxes, the past erasing the present</td></tr>
<tr><td>9</td><td>temporal change</td><td>small where nothing moved, larger at objects and at the cone's border</td><td>everything changes, or nothing at all</td></tr>
<tr><td>10-11</td><td>radar</td><td>raster: dots where the detections are; the radar branch smears them a little with a 3x3 context</td><td>features without raster dots, raster shifted against the GT boxes</td></tr>
<tr><td>12</td><td>trunk output</td><td>colour regions: road, off-road, objects; borders follow the scene</td><td>noise or one single colour</td></tr>
<tr><td>13-15</td><td>result, dt 0</td><td>prediction looks like the GT; the error map is mostly grey; red = false area, blue = missed area</td><td>large red or blue regions, boxes in empty space</td></tr>
<tr><td>16</td><td>velocity field</td><td>near 0 on the static scene, arrows on vehicles that move; pointing along the car axis</td><td>arrows everywhere, sideways arrows</td></tr>
<tr><td>17-18</td><td>dt +-0.5 s</td><td>the scene as it will be / was; the cone's unseen cells are hatched (unknown); objects move along their speed</td><td>objects not moving at all, content in the hatched cells</td></tr>
<tr><td>19-20</td><td>vehicles, scores</td><td>footprint overlap violet; scores: a few high detections, many near 0</td><td>only red or only blue footprints, a wall of mid scores</td></tr>
</table>
"""


def write_html(out_dir, run, step, cfg, frames, summary):
    css = ("body{font-family:Segoe UI,Arial,sans-serif;margin:24px;max-width:1500px}table{border-collapse:collapse;font-size:13px}"
           "td,th{border:1px solid #ccc;padding:3px 8px;text-align:left}.ok{background:#dff5df}.warn{background:#fff1c4}"
           ".bad{background:#f8d0d0}img{max-width:100%}h3{margin-top:28px}")
    h = [f"<html><head><meta charset='utf-8'><title>{html.escape(run)} observability</title><style>{css}</style></head><body>",
         f"<h1>Observability report: {html.escape(os.path.basename(run.rstrip('/')))}</h1>",
         f"<p>checkpoint step {step}, EMA weights, validation split, every {summary['every']}-th frame "
         f"({len(frames)} frames shown). Pipeline: front camera (frozen DualBEV heads: depth + context) -> lift to BEV -> "
         f"camera temporal fusion -> radar branch -> fusion -> trunk -> time-decoding heads (maps, boxes, motion).</p>",
         "<h2>Checks over all shown frames</h2><table><tr><th>check</th><th>values per frame</th><th>flags</th><th>meaning</th></tr>"]
    names = []
    for f in frames:
        for c in f["checks"]:
            if c[0] not in names:
                names.append(c[0])
    for n in names:
        row = [(f["checks"][[c[0] for c in f["checks"]].index(n)]) for f in frames if n in [c[0] for c in f["checks"]]]
        worst = "bad" if any(r[2] == "bad" for r in row) else "warn" if any(r[2] == "warn" for r in row) else "ok"
        nw = sum(r[2] != "ok" for r in row)
        h.append(f"<tr class='{worst}'><td>{html.escape(n)}</td><td>{html.escape(' | '.join(str(r[1]) for r in row))}</td>"
                 f"<td>{nw} of {len(row)}</td><td>{html.escape(row[0][3])}</td></tr>")
    h.append("</table>")
    if summary.get("videos"):
        h.append("<h2>Scene videos (all keyframes of a scene in order, the world memory carried over)</h2>")
        for v in summary["videos"]:
            h.append(f"<p>{html.escape(v['scene'])} ({v['info']['frames']} frames, {v['info']['codec']})<br>"
                     f"<video src='{v['path']}' controls loop width='1260'></video></p>")
    h.append(GUIDE)
    for f in frames:
        h.append(f"<h3>frame {f['idx']}: {html.escape(f['scene'])}, sample {html.escape(f['token'][:8])}, "
                 f"{f['n_clip']} keyframe(s) run before the picture</h3>")
        h.append(f"<a href='{f['stages']}'><img src='{f['stages']}'></a><a href='{f['outputs']}'><img src='{f['outputs']}'></a>")
        h.append("<table><tr><th>check</th><th>value</th><th>meaning</th></tr>")
        for n, v, s, note in f["checks"]:
            h.append(f"<tr class='{s}'><td>{html.escape(n)}</td><td>{html.escape(str(v))}</td><td>{html.escape(note)}</td></tr>")
        h.append("</table>")
    h.append("</body></html>")
    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(h))


# ------------------------------------------------------------------------------------------------ main
def _freeze_scenes(cfg):
    """Runs of 2026-10-26 (t1_*) have no scene lists and meant the 428 / 82 scenes of nuScenes parts 1-6; after parts 7-10 were
    added (and their camera features moved to D:) the same config means 700 / 150 scenes and most val frames would have no
    cached features. An empty list is replaced by the frozen one (data/scenes_parts1_6.json, data/README.md)."""
    path = os.path.join(ROOT, "data", "scenes_parts1_6.json")
    if os.path.exists(path) and not cfg.val_scenes and cfg.feature_cache:
        frozen = json.load(open(path))
        cfg.train_scenes, cfg.val_scenes = cfg.train_scenes or frozen["train"], frozen["val"]
        print("config without scene lists: using the frozen scenes of parts 1-6 "
              f"({len(cfg.train_scenes)} train / {len(cfg.val_scenes)} val)", flush=True)


def dashboard_figure(path, model, cfg, grid, batch, res, gts, tap, nusc, title):
    """One compact picture of every stage of one frame (a video frame): input -> depth -> camera BEV (after the temporal
    fusion) -> radar -> trunk -> outputs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from camera_encoder import depth_centres
    spec = model.spec
    g0 = gts[0.0][0]
    boxes = gt_boxes(g0)
    fig, axs = plt.subplots(2, 5, figsize=(21, 9))
    img = camera_image(nusc, batch["tokens"][0], os.path.expanduser(cfg.data_root), batch["ida"], batch["hw"])
    axs[0, 0].imshow(img)
    if "prob" in batch:
        ed = (batch["prob"][0, 0].float().numpy() * depth_centres().numpy()[:, None, None]).sum(0)
        axs[0, 0].imshow(np.kron(ed, np.ones((16, 16))), alpha=0.5, cmap="turbo", vmin=1, vmax=60,
                         extent=(0, img.size[0], img.size[1], 0))
    axs[0, 0].set_title("input + camera head depth (blue near, red far)", fontsize=9)
    axs[0, 0].axis("off")
    rad = batch["radar"]
    axs[0, 1].scatter(rad["y"].numpy(), rad["x"].numpy(), c=rad["vr"].numpy(), s=np.clip(rad["rcs"].numpy() + 20, 4, 40),
                      cmap="coolwarm", vmin=-15, vmax=15)
    bev_axes(axs[0, 1], grid, f"radar detections ({len(rad['x'])}), colour = Doppler")
    draw_boxes(axs[0, 1], boxes, "k", lw=0.7, ls="--")
    ct = tap.get("cam_temporal")
    if ct is not None:
        draw_map(axs[0, 2], grid, energy(ct[0].numpy()), "camera BEV after lift + temporal fusion (energy)", boxes, cbar=False)
    rf = tap.get("radar_feat")
    if rf is not None:
        draw_map(axs[0, 3], grid, energy(rf[0].numpy()), "radar branch (energy)", boxes, cbar=False)
    draw_map(axs[0, 4], grid, pca_rgb(res["features"][0].float().cpu().numpy()), "trunk output (feature PCA as RGB)", boxes)
    p0 = res["out"][0.0]
    prob, dist = predictions(p0, 0, spec)
    k = layer_index(spec)
    gt_prob = {n: np.asarray(g0[k[n][2]], np.float32) for n in prob}
    gt_dist = {n: np.asarray(g0[n + "_dist"], np.float32) for n in dist}
    show(axs[1, 0], composite(gt_prob, gt_dist), grid)
    bev_axes(axs[1, 0], grid, "ground truth now")
    draw_boxes(axs[1, 0], boxes, "tab:blue")
    show(axs[1, 1], composite(prob, dist), grid)
    bev_axes(axs[1, 1], grid, "prediction now (red boxes >= 0.3, dashed = GT)")
    draw_boxes(axs[1, 1], boxes, "k", lw=0.7, ls="--")
    dets = decode({a: b[:1].float().cpu() for a, b in p0["obj"].items()}, grid, NUSC_GROUPS, k=100, threshold=0.3)[0]
    draw_boxes(axs[1, 1], dets, "tab:red")
    for col, dt in ((2, 0.5), (3, -0.5)):
        if dt not in res["out"]:
            axs[1, col].axis("off")
            continue
        p = res["out"][dt]
        pr, ds = predictions(p, 0, spec)
        show(axs[1, col], composite(pr, ds), grid)
        bev_axes(axs[1, col], grid, f"prediction at dt {dt:+.1f} s (dashed = GT then)")
        draw_boxes(axs[1, col], gt_boxes(gts[dt][0]), "k", lw=0.7, ls="--")
        draw_boxes(axs[1, col], decode({a: b[:1].float().cpu() for a, b in p["obj"].items()}, grid, NUSC_GROUPS, k=100,
                                       threshold=0.3)[0], "tab:red")
    if p0.get("v_hat") is not None:
        v = p0["v_hat"][0].float().cpu().numpy()
        draw_map(axs[1, 4], grid, np.hypot(v[0], v[1]), "predicted speed field (m/s)", boxes, cmap="viridis", cbar=True, fig=fig,
                 vmax=max(2.0, float(np.hypot(v[0], v[1]).max())))
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=60)
    plt.close(fig)


def load_all(run, device):
    """everything a report needs, loaded once: (cfg, model, loss, step, val dataset, device)."""
    os.chdir(ROOT)
    device = device if torch.cuda.is_available() else "cpu"
    cfg, model, loss, step, ckpt = load_run(run, device)
    cfg.workers = 0
    _freeze_scenes(cfg)
    _, val_ds = make_datasets(cfg)
    return cfg, model, loss, step, val_ds, device


def video(run, scene=None, device="cuda", out=None, fps=2.0, max_frames=0, ctx=None):
    """All keyframes of one validation scene (a 20 s clip, 2 Hz) run in order with the model state (world memory)
    carried from frame to frame; one dashboard picture per frame -> video.mp4 (+ the frames, index_video.html)."""
    cfg, model, loss, step, val_ds, device = ctx or load_all(run, device)
    grid, nusc = model.grid, val_ds.nusc
    pos = {t: k for k, t in enumerate(val_ds.samples)}
    names = {nusc.get("scene", nusc.get("sample", t)["scene_token"])["name"] for t in val_ds.samples}
    if scene in (None, "auto"):
        scene = sorted(names)[len(names) // 2]
    elif scene not in names:
        raise ValueError(f"{scene} is not in the validation scenes of this config")
    first = [t for t in val_ds.samples if nusc.get("scene", nusc.get("sample", t)["scene_token"])["name"] == scene]
    toks = [first[0]]
    while True:
        nxt = nusc.get("sample", toks[-1])["next"]
        if not nxt or nxt not in pos:
            break
        toks.append(nxt)
    if max_frames:
        toks = toks[:max_frames]
    out_dir = out or os.path.join(run, "observe", f"video_{scene}")
    os.makedirs(os.path.join(out_dir, "frames"), exist_ok=True)
    fn = collate_cached if cfg.feature_cache else collate
    tap = Tap(model)
    state, prev_batch, t0, files = None, None, time.time(), []
    try:
        for n, tok in enumerate(toks):
            batch = fn([val_ds[pos[tok]]])
            st = clip_step(prev_batch, device) if prev_batch is not None else None
            tap.reset()
            with torch.no_grad():
                _, res, gts = forward_loss(model, loss, batch, 1.0, device, cfg.amp, cfg, train_terms=False, state=state, step=st)
            state, prev_batch = res["state"], batch
            path = os.path.join(out_dir, "frames", f"{n:03d}.png")
            dashboard_figure(path, model, cfg, grid, batch, res, gts, tap.d, nusc,
                             f"{os.path.basename(run.rstrip('/'))} step {step}  {scene}  keyframe {n + 1}/{len(toks)} "
                             f"(t = {n * 0.5:.1f} s, memory carried over {n} earlier keyframes)")
            files.append(path)
            print(f"video frame {n + 1}/{len(toks)} {time.time() - t0:.0f} s", flush=True)
    finally:
        tap.close()
    info = encode(files, os.path.join(out_dir, "video.mp4"), fps)
    print(f"encoded {info}", flush=True)
    print(f"done: {len(files)} frames -> {info['path']}")
    return info


def observe(run, every=20, max_frames=12, start=0, device="cuda", cold=False, out=None, per_scene=False, videos=0, fps=2.0):
    ctx = load_all(run, device)
    cfg, model, loss, step, val_ds, device = ctx
    grid, nusc = model.grid, val_ds.nusc
    out_dir = out or os.path.join(run, "observe")
    os.makedirs(os.path.join(out_dir, "img"), exist_ok=True)
    if per_scene:                                              # the middle keyframe of every (evenly chosen) scene
        by_scene = {}
        for k, tok in enumerate(val_ds.samples):
            by_scene.setdefault(nusc.get("sample", tok)["scene_token"], []).append(k)
        mids = [v[len(v) // 2] for v in by_scene.values()]
        idx = [mids[int(j)] for j in np.linspace(0, len(mids) - 1, min(max_frames, len(mids))).round()]
        every = len(val_ds) // max(len(by_scene), 1)
    else:
        idx = list(range(start, len(val_ds), every))[:max_frames]
    frames, t0 = [], time.time()
    for n, i in enumerate(idx):
        try:
            batch, res, gts, tap, vis, n_clip = run_frame(model, loss, cfg, val_ds, i, device, cold)
        except KeyError as e:                                  # a frame of the window is missing in the feature cache
            print(f"frame {i} skipped: no cached camera features for {e}", flush=True)
            continue
        tok = batch["tokens"][0]
        scene = nusc.get("scene", nusc.get("sample", tok)["scene_token"])["name"]
        ttl = f"{os.path.basename(run.rstrip('/'))} step {step}, val frame {i} ({scene}, {tok[:8]})"
        sp, op = f"img/{i:05d}_stages.png", f"img/{i:05d}_outputs.png"
        stages_figure(os.path.join(out_dir, sp), model, cfg, grid, batch, res, gts, tap, vis, nusc, None, ttl)
        outputs_figure(os.path.join(out_dir, op), model, cfg, grid, batch, res, gts, nusc, ttl)
        frames.append({"idx": i, "token": tok, "scene": scene, "stages": sp, "outputs": op, "n_clip": n_clip,
                       "checks": checks(model, cfg, grid, batch, res, gts, tap, vis)})
        print(f"frame {i} ({n + 1}/{len(idx)}) {time.time() - t0:.0f} s", flush=True)
    vids = []
    if videos:                                                 # scenes spread evenly over the validation set
        names = sorted({nusc.get("scene", nusc.get("sample", t)["scene_token"])["name"] for t in val_ds.samples})
        for sc in [names[int(j)] for j in np.linspace(0, len(names) - 1, videos + 2).round()[1:-1]]:
            vd = os.path.join(out_dir, f"video_{sc}")
            info = video(run, sc, device, vd, fps, ctx=ctx)
            vids.append({"scene": sc, "path": os.path.relpath(info["path"], out_dir).replace(os.sep, "/"), "info": info})
    summary = {"every": every, "videos": vids}
    write_html(out_dir, run, step, cfg, frames, summary)
    with open(os.path.join(out_dir, "checks.json"), "w") as f:
        json.dump([{k: v for k, v in fr.items()} for fr in frames], f, indent=1, default=str)
    print(f"done: {len(frames)} frames -> {os.path.join(out_dir, 'index.html')}")
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--every", type=int, default=20)
    ap.add_argument("--max", type=int, default=12)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cold", action="store_true", help="no previous keyframes: run each frame without carried memory")
    ap.add_argument("--out", default=None)
    ap.add_argument("--per-scene", action="store_true", help="one frame (the middle keyframe) per validation scene, spread "
                    "evenly over at most --max scenes, instead of every N-th frame")
    ap.add_argument("--video", default=None, metavar="SCENE", help="a video of all keyframes of one scene (name, or auto)")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--videos", type=int, default=0, help="with the frame report: also K scene videos, scenes spread evenly")
    a = ap.parse_args()
    if a.video:
        video(a.run, a.video, a.device, a.out, a.fps)
    else:
        observe(a.run, a.every, a.max, a.start, a.device, a.cold, a.out, a.per_scene, a.videos, a.fps)
