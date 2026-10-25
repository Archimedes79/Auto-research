"""Loop 10: training of the learned trace network by imitation of the ground-truth graph (noisy states, corrective expert actions, ring-limited context).
Usage (gcar, GPU slot): p10_train.py --name e94_a --iters 12000 [--no_cov] [--no_ptr] [--no_range] [--p_neg 0.2] [--seed 0]"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p10_data as PD  # noqa: E402
import p10_expert as EX  # noqa: E402
import p10_net as PN  # noqa: E402

OUT = "/mnt/c/ML-Experiment/bev_fusion_lane_center/runs"
EV_W = torch.tensor([1.0, 2.0, 3.0, 3.0, 2.0])


def seed_loss(logit, gd):
    tgt = torch.exp(-(gd ** 2) / (2 * 0.75 ** 2))
    bce = F.binary_cross_entropy_with_logits(logit, tgt, reduction="none")
    return (bce * (1.0 + 9.0 * tgt)).mean()


def dir_loss(dirh, gdir, gd):
    m = (gd < 1.0) & (gdir.norm(dim=1) > 0.5)
    pn = F.normalize(dirh, dim=1)
    cos = (pn * gdir).sum(1)
    return ((1 - cos) * m).sum() / m.sum().clamp(min=1)


def masked_mean(x, m):
    return (x * m).sum() / m.sum().clamp(min=1.0)


def step_loss(out, S, cfg):
    geo, valid = S["geo"], S["valid"]
    w = 1.0 + S["rng"] / 25.0 if cfg.range_weight else torch.ones_like(S["rng"])
    ce_h = F.cross_entropy(out["head"].float().flatten(0, 1), S["k"].flatten(), reduction="none").view_as(geo)
    ce_c = F.cross_entropy(out["step"].float().flatten(0, 1), S["c"].flatten(), reduction="none").view_as(geo)
    l_head = masked_mean(ce_h * w, geo)
    l_step = 0.3 * masked_mean(ce_c * w, geo)
    l_rp = masked_mean(F.smooth_l1_loss(out["res_phi"].float(), S["res_phi"], reduction="none", beta=0.3), geo)
    l_rl = masked_mean(F.smooth_l1_loss(out["res_len"].float(), S["res_len"], reduction="none", beta=0.3), geo)
    l_off = masked_mean(F.smooth_l1_loss(out["off"].float(), S["off"], reduction="none", beta=0.3), geo)
    bce = F.binary_cross_entropy_with_logits(out["on"].float(), S["on"], reduction="none")
    l_on = masked_mean(bce * torch.where(S["on"] > 0.5, 1.0, cfg.lam_fp) * w, valid)
    ce_e = F.cross_entropy(out["ev"].float().flatten(0, 1), S["ev"].flatten(), weight=EV_W.to(S["ev"].device), reduction="none").view_as(geo)
    l_ev = masked_mean(ce_e * w, valid)
    fk = ((S["ev"] == EX.FORK) & (geo > 0.5)).float()
    l_psi = masked_mean(F.smooth_l1_loss(out["psi"].float(), S["psi"], reduction="none", beta=0.3), fk)
    mg = ((S["ev"] == EX.MERGE) & (S["ptr"] >= 0) & (geo > 0.5)).float()
    M = out["ptr"].shape[-1] - 1
    tp = torch.where(S["ptr"] >= 0, S["ptr"], torch.full_like(S["ptr"], M))
    ce_p = F.cross_entropy(out["ptr"].float().flatten(0, 1), tp.flatten(), reduction="none").view_as(geo)
    l_ptr = masked_mean(ce_p, mg)
    tot = l_head + l_step + 2.0 * l_rp + 0.5 * l_rl + l_off + cfg.w_on * l_on + cfg.w_ev * l_ev + 0.5 * l_psi + 0.5 * l_ptr
    with torch.no_grad():
        kh = out["head"].argmax(-1)
        met = dict(acc_head=float(masked_mean((kh == S["k"]).float(), geo)), acc_head1=float(masked_mean(((kh - S["k"]).abs() <= 1).float(), geo)),
                   acc_step=float(masked_mean((out["step"].argmax(-1) == S["c"]).float(), geo)), acc_ev=float(masked_mean((out["ev"].argmax(-1) == S["ev"]).float(), valid)),
                   acc_on=float(masked_mean(((out["on"] > 0).float() == S["on"]).float(), valid)), off_mae=float(masked_mean((out["off"] - S["off"]).abs(), geo)),
                   acc_ptr=float(masked_mean((out["ptr"].argmax(-1) == tp).float(), mg)))
    parts = dict(head=float(l_head), step=float(l_step), rp=float(l_rp), rl=float(l_rl), off=float(l_off), on=float(l_on), ev=float(l_ev), psi=float(l_psi), ptr=float(l_ptr))
    return tot, parts, met


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--iters", type=int, default=12000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_cov", action="store_true")
    ap.add_argument("--no_ptr", action="store_true")
    ap.add_argument("--no_range", action="store_true")
    ap.add_argument("--no_range_weight", action="store_true")
    ap.add_argument("--lam_fp", type=float, default=3.0)
    ap.add_argument("--w_on", type=float, default=1.0)
    ap.add_argument("--w_ev", type=float, default=1.0)
    ap.add_argument("--w_seed", type=float, default=0.5)
    ap.add_argument("--w_dir", type=float, default=0.2)
    ap.add_argument("--ns", type=int, default=32)
    ap.add_argument("--init", default=None)
    ap.add_argument("--p_wd", type=float, default=0.0, help="share of wrong-direction states (head faces against the direction implied by its flag; target: stop)")
    ap.add_argument("--nav", action="store_true", help="E103: navigation attribute inside the network (SD raster branch + route head on the step token)")
    ap.add_argument("--freeze_trace", action="store_true", help="E103a: train only the navigation branch on top of a frozen trace network (--init)")
    ap.add_argument("--w_route", type=float, default=1.0)
    ap.add_argument("--max_frames", type=int, default=0, help="train on a fixed random subset of this many frames (0 = all)")
    ap.add_argument("--fold", type=int, default=-1, help="hold out fold f of the cross-fitted dense heads (E102 cross-fitting)")
    ap.add_argument("--p_drop", type=float, default=0.1, help="share of the context pieces dropped in training (context robustness, E95)")
    ap.add_argument("--stem_init", default=None, help="loop 16: initialise ONLY the stem from the model.pt of this run (shared warm start of the ladder); the rest starts from scratch")
    ap.add_argument("--stem_lr", type=float, default=1.0, help="learning-rate factor of the stem (loop 16 uses 0.3, like E120)")
    a = ap.parse_args()
    a.range_weight = not a.no_range_weight
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    out_dir = f"{OUT}/{a.name}"
    os.makedirs(out_dir, exist_ok=True)
    dev = "cuda"
    net = PN.TraceNet(use_cov=not a.no_cov, use_ptr=not a.no_ptr, use_range=not a.no_range, nav=a.nav).to(dev)
    if a.init:
        miss = net.load_state_dict(torch.load(f"{OUT}/{a.init}/model.pt", map_location=dev), strict=not a.nav)
        print("init from", a.init, "missing keys:", len(getattr(miss, "missing_keys", [])), flush=True)
    if a.freeze_trace:
        for n_, p_ in net.named_parameters():
            p_.requires_grad = n_.startswith(("sd_cnn", "route_mlp"))
    print(f"params {PN.count_params(net) / 1e6:.2f} M", flush=True)
    rows = None
    if a.fold >= 0:  # cross-fitting: fold f of the dense head's out-of-fold split is held out
        ex = set(int(r) for r in np.load(f"{OUT}/e37_fold{a.fold}/ids_oof.npy"))
        rows = [r for r in range(8414) if r not in ex]
        print(f"fold {a.fold} held out: training on {len(rows)} frames", flush=True)
    if a.max_frames:  # a fixed random subset of the frames (fits into the page cache: the full run is bound by disk reads)
        pool = np.arange(8414) if rows is None else np.asarray(rows)
        rows = np.random.RandomState(0).permutation(pool)[: a.max_frames]
    ds = PD.TraceData("train", rows=rows, ns=a.ns, seed=a.seed, p_drop=a.p_drop, nav=a.nav, p_wd=a.p_wd)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True, persistent_workers=True, prefetch_factor=3)
    if a.stem_init:
        sd0 = torch.load(f"{OUT}/{a.stem_init}/model.pt", map_location=dev)
        net.stem.load_state_dict({k[5:]: v for k, v in sd0.items() if k.startswith("stem.")})
        print("stem initialised from", a.stem_init, flush=True)
    stem_ps = [p_ for n_, p_ in net.named_parameters() if p_.requires_grad and n_.startswith("stem.")]
    rest_ps = [p_ for n_, p_ in net.named_parameters() if p_.requires_grad and not n_.startswith("stem.")]
    groups = [dict(params=rest_ps, lr=a.lr)] + ([dict(params=stem_ps, lr=a.lr * a.stem_lr)] if stem_ps else [])
    opt = torch.optim.AdamW(groups, lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g_["lr"] for g_ in groups], total_steps=a.iters, pct_start=0.1)
    scaler = torch.cuda.amp.GradScaler()
    it, t0 = 0, time.time()
    run = {}
    log = open(f"{out_dir}/log.txt", "a")
    while it < a.iters:
        for b in dl:
            x = b["x"].to(dev, non_blocking=True).float()
            S = {k: v.to(dev, non_blocking=True) for k, v in b.items() if k not in ("x", "gd", "gdir", "row", "sd")}
            sd = b["sd"].to(dev, non_blocking=True).float() if a.nav else None
            gd, gdir = b["gd"].to(dev), b["gdir"].to(dev)
            net.train()
            if a.freeze_trace:  # BatchNorm statistics of the frozen trace network stay as they are
                for n_, m_ in net.named_modules():
                    if not n_.startswith(("sd_cnn", "route_mlp")):
                        m_.eval()
            with torch.autocast("cuda", dtype=torch.float16):
                feats, seed_lg, dirh = net.encode(x)
                out = net(feats, S, qc=S["qc"], sd=sd, route_detach=a.freeze_trace)
            tot_s, parts, met = step_loss(out, S, a)
            l_seed = seed_loss(seed_lg.float(), gd)
            l_dir = dir_loss(dirh.float(), gdir, gd)
            tot = tot_s + a.w_seed * l_seed + a.w_dir * l_dir
            if a.nav:
                lr_ = F.binary_cross_entropy_with_logits(out["route"].float(), S["route"], reduction="none", pos_weight=torch.tensor(2.0, device=dev))
                l_route = masked_mean(lr_, S["valid"])
                tot = a.w_route * l_route if a.freeze_trace else tot + a.w_route * l_route  # frozen trace network: only the navigation branch is trained
                parts["route"] = float(l_route)
                met["acc_route"] = float(masked_mean(((out["route"] > 0).float() == S["route"]).float(), S["valid"]))
                met["route_pos"] = float(masked_mean(S["route"], S["valid"]))
            opt.zero_grad(set_to_none=True)
            scaler.scale(tot).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            parts.update(seed=float(l_seed), dir=float(l_dir), tot=float(tot))
            for k, v in {**parts, **met}.items():
                run[k] = 0.98 * run.get(k, v) + 0.02 * v
            it += 1
            if it % 100 == 0 or it == 1:
                msg = f"it {it} {time.time() - t0:.0f}s " + " ".join(f"{k} {v:.3f}" for k, v in run.items())
                print(msg, flush=True)
                log.write(msg + "\n")
                log.flush()
            if it % 2000 == 0 or it == a.iters:
                torch.save(net.state_dict(), f"{out_dir}/model.pt")
            if it >= a.iters:
                break
    torch.save(net.state_dict(), f"{out_dir}/model.pt")
    json.dump(dict(args=vars(a), final=run, seconds=time.time() - t0), open(f"{out_dir}/train_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
