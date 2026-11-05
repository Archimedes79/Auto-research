"""scratch (open finding #5, BatchNorm over dt; records/bn_over_dt_2026-11-03.md): does the mixture of dt statistics in the running statistics of the `shared` BatchNorm
of the semantic / object head hurt the dt 0 output? EMA weights of runs/t16_full_s0 (ckpt_00006000.pt), dt 0 loss and tolerant IoU on val clips, three ways:
  A  eval mode (the running statistics, a mixture of dt -0.5 / 0 / +0.5)
  B  the two BatchNorm modules in train mode (batch statistics of the dt 0 input of each batch of 3 samples), running statistics NOT updated (momentum 0)
  C  eval mode with running statistics re-estimated from dt 0 inputs only (cumulative average over the OTHER half of the clips: no labels, no overlap)
plus the size of the mixture: per-dt input statistics of the two modules (eval forward, all three dts) against the running statistics.
RULE written before the numbers (the record repeats it): the mixture is called negligible when, for B and C against A, (1) every dt 0 group loss differs by less than 2 paired
standard errors over the 25 batches of 3 clips AND by less than 1 % of its value, and (2) the lane divider IoU@1 (tolerance 1 m) differs by less than the band 0.010 of the card.
Output: temp/bn_dt/measure.json. Run from bev_fusion (GPU or CPU): $HOME/miniconda3/envs/gcar/bin/python temp/bn_dt/measure.py [--device cuda] [--clips 75]"""
import json
import math
import os
import statistics
import sys
import time

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path[:0] = [os.path.join(ROOT, "training"), ROOT, os.path.join(ROOT, ".."), os.path.join(ROOT, "..", "autoresearch")]
os.chdir(ROOT)
import losses as L  # noqa: E402
import train as T  # noqa: E402
from evaluation.metrics import SegMeter  # noqa: E402

DEVICE = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else "cuda"
N_EVAL = int(sys.argv[sys.argv.index("--clips") + 1]) if "--clips" in sys.argv else 75
RUN = "runs/t16_full_s0"
LAYERS = ("lane_divider", "road_divider", "drivable_area", "vehicle")
FLAT = {"total", "group.areas", "group.lines", "group.objects", "group.motion"}


def build_model(cfg):
    model, loss, _ = T.build(cfg)
    ck = torch.load(os.path.join(RUN, "ckpt_00006000.pt"), map_location="cpu", weights_only=False)
    rep = model.load_state_dict(ck["ema"], strict=False)
    assert not rep.unexpected_keys and all(k.endswith("num_batches_tracked") for k in rep.missing_keys), rep
    loss.load_state_dict({k[len("loss."):]: v for k, v in ck["state"].items() if k.startswith("loss.")})
    return model.to(DEVICE), loss.to(DEVICE)


def heads_bn(model):
    return {"sem": model.sem_head.head.net[1], "obj": model.obj_head.head.shared[1]}


def loaders(cfg):
    _, val_ds = T.make_datasets(cfg)
    clips = T.ClipDataset(val_ds, cfg.seq_len)
    sub = T.val_subset(clips, cfg.eval_batches * cfg.batch_size)        # the trainer's own fixed subset of clips (their GT is cached)
    mk = lambda idx: T.make_loader(torch.utils.data.Subset(clips, idx), cfg, False, workers=0, generator=torch.Generator().manual_seed(cfg.seed))   # noqa: E731
    return mk(sub.indices[0::2][:N_EVAL]), mk(sub.indices[1::2][:N_EVAL]), len(sub.indices)


def clip_losses(model, loss, clip, cfg, meter, dt0_only=True):
    """one clip (consecutive keyframes) like train.run_clip (state carried), the GT pruned to the dt 0 side; -> list of dicts of the scalar terms"""
    state, prev, rows = None, None, []
    for batch in clip:
        if dt0_only:
            batch = dict(batch, gt=[{"cur": g["cur"]} for g in batch["gt"]])
        step = T.clip_step(prev, DEVICE) if prev is not None else None
        out, res, gts = T.forward_loss(model, loss, batch, 1.0, DEVICE, "off", cfg, train_terms=False, state=state, step=step)
        rows.append({k: float(v) for k, v in out.items() if torch.is_tensor(v) and v.numel() == 1 and (k in FLAT or k.endswith("@+0.0"))})
        if meter is not None:
            sem = res["out"][0.0]["sem"].float()
            T_ = L.build_layer_targets(gts[0.0], loss.cfg, None, 0.0, sem.device)
            names = [s[0] for s in model.spec]
            for b in range(sem.shape[0]):
                meter.add({n: torch.sigmoid(sem[b, names.index(n)]) for n in LAYERS if n in names}, {n: T_[n][b, 0] for n in LAYERS if n in names}, dt=0.0)
        state, prev = res["state"], batch
    return rows


def evaluate(model, loss, loader, cfg):
    meter = SegMeter(model.grid)
    per_batch = []
    with torch.no_grad():
        for item in loader:
            clip = item if isinstance(item, list) else [item]
            rows = clip_losses(model, loss, clip, cfg, meter)
            per_batch.append({k: statistics.mean(r[k] for r in rows) for k in rows[0]})
    iou = {n: meter.result()[0.0][n]["all"]["all"] for n in LAYERS}
    return per_batch, iou


def summarize(per_batch):
    keys = sorted(per_batch[0])
    return {k: statistics.mean(b[k] for b in per_batch) for k in keys}


def paired(a, b, keys):
    """b - a over the batches: mean, standard error"""
    out = {}
    for k in keys:
        d = [y[k] - x[k] for x, y in zip(a, b)]
        out[k] = {"diff": statistics.mean(d), "se": statistics.stdev(d) / math.sqrt(len(d)), "rel_pct": 100 * statistics.mean(d) / statistics.mean(x[k] for x in a)}
    return out


def dt_statistics(model, loss, loader, cfg):
    """channel means / variances of the BN input per call (= per dt: -0.5, 0, +0.5, the heads are called in the order of the sorted dts) in an eval forward, averaged over frames"""
    acc = {n: {} for n in ("sem", "obj")}
    calls = {n: 0 for n in acc}
    hooks = []
    for name, bn in heads_bn(model).items():
        def pre(mod, args, name=name):
            x = args[0].float()
            i = calls[name] % 3
            calls[name] += 1
            m, v = x.mean((0, 2, 3)), x.var((0, 2, 3), unbiased=False)
            a = acc[name].setdefault(i, [0, 0.0, 0.0])
            a[0] += 1
            a[1] = a[1] + m.double()
            a[2] = a[2] + v.double()
        hooks.append(bn.register_forward_pre_hook(pre))
    with torch.no_grad():
        for item in loader:
            state, prev = None, None
            for batch in item:
                step = T.clip_step(prev, DEVICE) if prev is not None else None
                _, res, _ = T.forward_loss(model, loss, batch, 1.0, DEVICE, "off", cfg, train_terms=False, state=state, step=step)
                state, prev = res["state"], batch
    for h in hooks:
        h.remove()
    res = {}
    for name, bn in heads_bn(model).items():
        run_m, run_v = bn.running_mean.double(), bn.running_var.double()
        res[name] = {}
        for i, dt in enumerate((-0.5, 0.0, 0.5)):
            n, m, v = acc[name][i]
            m, v = m / n, v / n
            res[name][f"{dt:+.1f}"] = {"mean_shift_in_run_sigma": float(((m - run_m).abs() / run_v.sqrt()).mean()), "var_ratio_to_running": float((v / run_v).mean()),
                                       "mean_of_means": float(m.mean()), "mean_of_vars": float(v.mean())}
        res[name]["running"] = {"mean_of_means": float(run_m.mean()), "mean_of_vars": float(run_v.mean())}
    return res


def main():
    t0 = time.time()
    cfg = T.read_config(os.path.join(RUN, "config.json"))
    cfg.workers = 0
    cfg.device = DEVICE
    model, loss = build_model(cfg)
    eval_loader, calib_loader, n_sub = loaders(cfg)
    print(f"model built, {len(eval_loader.dataset)} eval clips and {len(calib_loader.dataset)} calibration clips of the trainer's {n_sub}-clip val subset ({time.time() - t0:.0f} s)", flush=True)
    bns = heads_bn(model)
    saved = {n: (bn.running_mean.clone(), bn.running_var.clone(), bn.num_batches_tracked.clone()) for n, bn in bns.items()}
    res = {"run": RUN, "ckpt": "ckpt_00006000.pt (EMA weights)", "eval_clips": len(eval_loader.dataset), "calib_clips": len(calib_loader.dataset)}

    model.eval()
    a_rows, a_iou = evaluate(model, loss, eval_loader, cfg)
    print(f"A eval mode: {summarize(a_rows)['total']:.4f} ({time.time() - t0:.0f} s)", flush=True)

    for bn in bns.values():                                               # B: batch statistics of the dt 0 input, running statistics untouched
        bn.train()
        bn.momentum = 0.0
    b_rows, b_iou = evaluate(model, loss, eval_loader, cfg)
    for n, bn in bns.items():
        assert torch.equal(bn.running_mean, saved[n][0]) and torch.equal(bn.running_var, saved[n][1]), "mode B changed the running statistics"
        bn.eval()
        bn.momentum = 0.1
    print(f"B batch statistics: {summarize(b_rows)['total']:.4f} ({time.time() - t0:.0f} s)", flush=True)

    for bn in bns.values():                                               # C: running statistics from dt 0 inputs only (the other half of the clips)
        bn.reset_running_stats()
        bn.momentum = None
        bn.train()
    with torch.no_grad():
        for item in calib_loader:
            clip_losses(model, loss, item, cfg, None)
    for bn in bns.values():
        bn.eval()
    res["running_stats_shift_dt0_vs_mixture"] = {n: {"mean_shift_in_mixture_sigma": float(((bn.running_mean - saved[n][0]).abs() / saved[n][1].sqrt()).mean()),
                                                     "var_ratio_dt0_to_mixture": float((bn.running_var / saved[n][1]).mean())} for n, bn in bns.items()}
    c_rows, c_iou = evaluate(model, loss, eval_loader, cfg)
    print(f"C dt 0 running statistics: {summarize(c_rows)['total']:.4f} ({time.time() - t0:.0f} s)", flush=True)

    for n, bn in bns.items():                                             # back to the checkpoint's statistics for the per-dt look
        bn.running_mean.copy_(saved[n][0])
        bn.running_var.copy_(saved[n][1])
        bn.num_batches_tracked.copy_(saved[n][2])
        bn.momentum = 0.1
    model.eval()
    res["per_dt_input_statistics"] = dt_statistics(model, loss, calib_loader, cfg)

    keys = sorted(a_rows[0])
    res["A_eval"], res["B_batch_stats"], res["C_dt0_running"] = summarize(a_rows), summarize(b_rows), summarize(c_rows)
    res["iou_A"], res["iou_B"], res["iou_C"] = a_iou, b_iou, c_iou
    res["B_minus_A"], res["C_minus_A"] = paired(a_rows, b_rows, keys), paired(a_rows, c_rows, keys)
    res["n_batches"] = len(a_rows)
    with open("temp/bn_dt/measure.json", "w") as f:
        json.dump(res, f, indent=1)
    print(f"done ({time.time() - t0:.0f} s)", flush=True)
    for name, d in (("B", res["B_minus_A"]), ("C", res["C_minus_A"])):
        for k in ("total", "group.areas", "group.lines", "group.objects", "group.motion"):
            print(f"{name} - A  {k:14s} {d[k]['diff']:+.4f} (se {d[k]['se']:.4f}, {d[k]['rel_pct']:+.2f} %)")
    for n in ("lane_divider", "road_divider", "drivable_area", "vehicle"):
        print(f"{n:14s} iou@1  A {a_iou[n]['iou@1']:.4f}  B {b_iou[n]['iou@1']:.4f}  C {c_iou[n]['iou@1']:.4f}   iou@0.5  A {a_iou[n]['iou@0.5']:.4f}  B {b_iou[n]['iou@0.5']:.4f}  C {c_iou[n]['iou@0.5']:.4f}")


if __name__ == "__main__":
    main()
