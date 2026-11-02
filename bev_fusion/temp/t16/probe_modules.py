"""scratch of the t16 bring-up, stage S4 (D44): what does every module of the full network output, does it respond to its input,
and is it related to the GT? One run, a few val frames, CPU or GPU.

    python bev_fusion/temp/t16/probe_modules.py runs/t16_small_s0 [--frames 3] [--device cpu] [--out DIR]

Writes OUT/modules.json (per module: shape, finite, mean, std, share of dead channels, share of zeros), OUT/response.json (relative change
of the chain's stages when one sensor is removed: the response test), OUT/frame_<k>.png (one panel per stage with the GT beside the heads).
The network is run through train.forward_loss exactly as the evaluation does (cold memory per frame; a warm second frame for the memories).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))                      # bev_fusion/
sys.path.insert(0, ROOT)

MODULES = ["lift", "cam_reduce", "cam_reduce2", "cam_merge", "cam_temporal", "radar", "rad_temporal", "fusion", "trunk", "head_reduce",
           "motion_block", "sem_head", "obj_head", "aux_cam", "aux_rad"]


def tensors_of(o, prefix=""):
    """every float tensor inside a module output (tensor, tuple, list, dict), named by its path"""
    out = {}
    if torch.is_tensor(o):
        if o.is_floating_point():
            out[prefix or "out"] = o.detach().float().cpu()
    elif isinstance(o, dict):
        for k, v in o.items():
            out.update(tensors_of(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(o, (tuple, list)):
        for i, v in enumerate(o):
            out.update(tensors_of(v, f"{prefix}[{i}]" if prefix else f"[{i}]"))
    return out


def stat(t):
    t = t.float()
    flat = t.reshape(t.shape[0], -1) if t.dim() > 1 else t.reshape(1, -1)
    c = t.reshape(t.shape[0], t.shape[1], -1) if t.dim() >= 3 else None
    dead = float((c.std(dim=(0, 2)) < 1e-7).float().mean()) if c is not None else None
    return {"shape": list(t.shape), "finite": bool(torch.isfinite(t).all()), "mean": float(t.mean()), "std": float(t.std()),
            "absmax": float(t.abs().max()), "zero_share": float((t == 0).float().mean()), "dead_channels": dead}


def rel_change(a, b):
    a, b = a.float(), b.float()
    return float((a - b).norm() / (a.norm() + 1e-9))


def energy(t):
    """(X, Y) map: channel-norm of a (C, X, Y) / (1, C, X, Y) / (X, Y) tensor"""
    t = t.detach().float().cpu()
    while t.dim() > 3:
        t = t[0]
    return (t.norm(dim=0) if t.dim() == 3 else t).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", default=None)
    ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    run = os.path.abspath(a.run)
    out_dir = a.out or os.path.join(run, "eval", "modules")
    os.makedirs(out_dir, exist_ok=True)

    from evaluation import bev_adapter as A
    A.setup_paths(run)
    src, subj, task = A.build(run, device=a.device, split=a.split, workers=0)
    ns, model, loss, cfg = subj.ns, subj.model, subj.loss, subj.cfg
    T = ns.T
    print("run", run, "step", subj.step, "checkpoint", os.path.basename(subj.ckpt), "device", subj.device, flush=True)

    store = {}
    hooks = []
    for name in MODULES:
        m = getattr(model, name, None)
        if m is None:
            continue
        def hook(mod, i, o, n=name):                               # returns None: a hook that returns a value REPLACES the module's output
            store.setdefault(n, tensors_of(o))
        hooks.append(m.register_forward_hook(hook))

    scenes = src.scenes()
    refs = []
    for s in scenes:
        fr = src.frames(s)
        refs += [(s, len(fr) // 2), (s, len(fr) // 3)]
    refs = refs[:a.frames]

    modules, response, rows = {}, {}, []
    dts = (-0.5, 0.0, 0.5)
    for k, (scene, idx) in enumerate(refs):
        ref = src.frames(scene)[idx]
        sample = src.load(ref)
        batch = T.to_device(sample.inputs, subj.device)
        store.clear()
        with torch.no_grad():
            _, res, _ = T.forward_loss(model, loss, batch, 1.0, subj.device, cfg.amp, cfg, train_terms=False)
        base = {n: {p: t.clone() for p, t in d.items()} for n, d in store.items()}
        out0 = res["out"][0.0]
        for n, d in base.items():
            for p, t in d.items():
                modules.setdefault(f"{n}:{p}", []).append(stat(t))
        for p, t in tensors_of(out0, "out").items():
            modules.setdefault(f"OUT dt0:{p}", []).append(stat(t))
        if "radar_side" in res:
            for p, t in tensors_of(res["radar_side"], "radar_side").items():
                modules.setdefault(f"radar_side:{p}", []).append(stat(t))
        modules.setdefault("features(S)", []).append(stat(res["features"]))
        # the warm frame: the next frame with the temporal state of this one (the memories read what the first call wrote)
        s2 = src.load(src.frames(scene)[idx + 1])
        store.clear()
        with torch.no_grad():
            _, resw, _ = T.forward_loss(model, loss, T.to_device(s2.inputs, subj.device), 1.0, subj.device, cfg.amp, cfg, train_terms=False,
                                        state=ns.detach_tree(res["state"]), step=ns.clip_step(sample.inputs, subj.device))
        for n, d in store.items():
            for p, t in d.items():
                if n in ("cam_temporal", "rad_temporal", "fusion"):
                    modules.setdefault(f"WARM {n}:{p}", []).append(stat(t))
        store.clear()

        # response test: remove one sensor, compare the stages
        def run_with(mod):
            b2 = {kk: (v.clone() if torch.is_tensor(v) else (dict(v) if isinstance(v, dict) else v)) for kk, v in batch.items()}
            mod(b2)
            store.clear()
            with torch.no_grad():
                _, r2, _ = T.forward_loss(model, loss, b2, 1.0, subj.device, cfg.amp, cfg, train_terms=False)
            return r2, {n: {p: t.clone() for p, t in d.items()} for n, d in store.items()}

        def zero_cam(b2):
            for kk in ("prob", "ctx", "prob2", "ctx2"):
                if kk in b2 and torch.is_tensor(b2[kk]):
                    b2[kk] = b2[kk] * 0

        def zero_radar(b2):
            r = dict(b2["radar"])
            n = r["x"].shape[0]
            keep = torch.zeros(n, dtype=torch.bool)               # no detection at all: an empty radar
            b2["radar"] = {kk: v[keep] for kk, v in r.items()}
            for kk in ("radar_hits", "radar_static"):
                if kk in b2 and torch.is_tensor(b2[kk]):
                    b2[kk] = b2[kk] * 0

        def zero_doppler(b2):
            r = dict(b2["radar"])
            r["vr"] = r["vr"] * 0
            b2["radar"] = r

        for pname, mod in (("zero_cam", zero_cam), ("zero_radar", zero_radar), ("zero_doppler", zero_doppler)):
            r2, st2 = run_with(mod)
            ch = {}
            for n in ("cam_temporal", "radar", "rad_temporal", "fusion", "trunk", "motion_block"):
                if n in base and n in st2:
                    ch[n] = {p: rel_change(base[n][p], st2[n][p]) for p in list(base[n])[:2] if p in st2[n]}
            o2 = r2["out"][0.0]
            ch["OUT"] = {"sem": rel_change(out0["sem"], o2["sem"]), "hm": rel_change(out0["obj"]["hm"], o2["obj"]["hm"]),
                         "vel": rel_change(out0["obj"]["vel"], o2["obj"]["vel"]) if "vel" in out0["obj"] else None,
                         "v_hat": rel_change(out0["v_hat"], o2["v_hat"]) if out0.get("v_hat") is not None else None}
            response.setdefault(pname, []).append(ch)

        # one image per frame: the stages and the GT beside the heads
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        gt = sample.reference[0.0]
        sem = out0["sem"][0].float().cpu()
        spec = model.spec
        names = [s[0] for s in spec]
        panels = []

        def add(title, arr, cmap="viridis"):
            panels.append((title, np.asarray(arr), cmap))
        for n in ("cam_temporal", "radar", "rad_temporal", "fusion", "trunk"):
            if n in base and base[n]:
                add(f"{n} |feat|", energy(list(base[n].values())[0]))
        add("features S |.|", energy(res["features"]))
        for lay in ("drivable_area", "lane_divider"):
            if lay in names:
                add(f"pred {lay} (sigmoid)", torch.sigmoid(sem[names.index(lay)]).numpy(), "magma")
                add(f"GT {lay}", np.asarray(gt[lay]), "magma")
        hm = out0["obj"]["hm"][0].float().cpu()
        add("pred object heat (max over groups)", torch.sigmoid(hm).max(0).values.numpy(), "magma")
        add("GT vehicle", np.asarray(gt["vehicle"]), "magma")
        vh = out0.get("v_hat")
        if vh is not None:
            add("v_hat |v| (motion block field)", vh[0].float().cpu().norm(dim=0).numpy(), "plasma")
        add("GT |flow| (m/s)", np.linalg.norm(np.asarray(gt["flow"]), axis=0), "plasma")
        if out0.get("v_levels"):
            for lv, v in out0["v_levels"].items():
                add(f"M level {lv} |v|", v[0].float().cpu().norm(dim=0).numpy(), "plasma")
        if "mov" in out0["obj"]:
            add("pred moving prob (hurdle)", torch.sigmoid(out0["obj"]["mov"][0].float().cpu()).max(0).values.numpy(), "magma")
        nc = 4
        nr = (len(panels) + nc - 1) // nc
        fig, axs = plt.subplots(nr, nc, figsize=(3.4 * nc, 3.6 * nr))
        for ax in np.ravel(axs):
            ax.axis("off")
        for ax, (title, arr, cmap) in zip(np.ravel(axs), panels):
            ax.imshow(arr.T[::-1] if arr.ndim == 2 else arr, cmap=cmap)       # x forward = up
            ax.set_title(title, fontsize=8)
        fig.suptitle(f"{os.path.basename(run)} step {subj.step}  frame {ref.key}", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"frame_{k}.png"), dpi=70)
        plt.close(fig)
        print("frame", k, ref.key, "done", flush=True)

    def mean_of(lst):
        keys = lst[0].keys()
        return {kk: (float(np.mean([d[kk] for d in lst])) if isinstance(lst[0][kk], (int, float, bool)) and lst[0][kk] is not None else lst[0][kk])
                for kk in keys}
    summary = {k: mean_of(v) for k, v in modules.items()}
    for h in hooks:
        h.remove()

    def avg_resp(lst):
        o = {}
        for ch in lst:
            for n, d in ch.items():
                for p, v in d.items():
                    o.setdefault(f"{n}:{p}", []).append(v)
        return {k: (float(np.mean([x for x in v if x is not None])) if any(x is not None for x in v) else None) for k, v in o.items()}
    resp = {k: avg_resp(v) for k, v in response.items()}
    with open(os.path.join(out_dir, "modules.json"), "w") as f:
        json.dump(summary, f, indent=1)
    with open(os.path.join(out_dir, "response.json"), "w") as f:
        json.dump(resp, f, indent=1)
    print("\nMODULE OUTPUTS (mean over frames):")
    for k, v in summary.items():
        print(f"  {k:48s} std {v['std']:10.4g} absmax {v['absmax']:10.4g} zero {v['zero_share']:.2f} dead_ch {v['dead_channels']} finite {v['finite']}")
    print("\nRESPONSE (relative change of the stage output when the input is removed):")
    for p, d in resp.items():
        print(" ", p)
        for k, v in d.items():
            print(f"    {k:40s} {v if v is None else round(v, 4)}")


if __name__ == "__main__":
    main()
