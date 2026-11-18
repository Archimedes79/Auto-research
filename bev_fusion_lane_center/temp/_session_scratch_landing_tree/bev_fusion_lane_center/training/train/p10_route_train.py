"""Loop 10, E103c: graph transformer route head on the stored tokens of a frozen trace network (p10_tokens.py), trained with the loop-9 map faults (pose error, dropped roads, per-road bias, wrong map, no map).
The SD map enters as a raster (no registration step); the label of a vertex is 'lies on the route lane' (ground-truth route heat); wrong / missing maps have the target 0 (abstention).
Usage (gcar, GPU): p10_route_train.py --tok tok_e94_b_oof0 --name e103c --iters 3000"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p8_lib as L  # noqa: E402
import p10_net as PN  # noqa: E402
import sd_lib as SL  # noqa: E402

HOME = os.path.expanduser("~")
OUT = L.OUT
NMAX = int(os.environ.get("NMAX", 320))


def draw_cell(sd0, sdl, row, rng, p_wrong, p_none, p_bias):
    u = rng.rand()
    kind = "w" if u < p_wrong else ("0" if u < p_wrong + p_none else "n")
    sd = sd0 if kind != "w" else sdl[(int(row) + 397) % len(sdl)]
    if sd is None:
        kind = "0"
    if kind == "0":
        return np.zeros((5, 200, 200), np.float32), kind, np.zeros(2), 0.0
    sigma = float(rng.choice([0.0, 1.0, 2.0, 3.0]))
    drop = 0.3 if rng.rand() < 0.3 else 0.0
    kw = dict(bias=float(rng.choice([1.0, 2.0]))) if (kind == "n" and rng.rand() < p_bias) else {}
    sdm, dd, RR = L.draw_sd(sd, row, sigma, drop, int(rng.randint(1, 1 << 20)), **kw)
    return SL.raster_sd(sdm), kind, np.asarray(dd, np.float64), float(np.arctan2(RR[1, 0], RR[0, 0]))


class TokData(torch.utils.data.Dataset):
    def __init__(self, files, p_wrong=0.08, p_none=0.08, p_bias=0.3, seed=0):
        self.items = []
        self.part = None
        for f in files:
            z = pickle.load(open(f"{HOME}/exp_gcar_b123/p10/{f}.pkl", "rb"))
            self.part = z["part"]
            self.items += z["items"]
        self.p = (p_wrong, p_none, p_bias)
        self.seed = seed
        self.sdl = L.cache("sd", self.part)
        self.rh = None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        if self.rh is None:
            self.rh = np.load(f"{L.D}/route_heat_{self.part}.npy", mmap_mode="r")
        it = self.items[i]
        rng = np.random.RandomState((self.seed * 7919 + i * 104729 + np.random.randint(1 << 30)) % (2 ** 31))
        raster, kind, dpose, yaw = draw_cell(self.sdl[it["row"]], self.sdl, it["row"], rng, *self.p)
        n = min(len(it["xy"]), NMAX)
        xy = np.zeros((NMAX, 2), np.float32)
        tok = np.zeros((NMAX, 128), np.float32)
        cp = np.zeros((NMAX, 7, 2), np.float32)
        mask = np.zeros(NMAX, bool)
        y = np.zeros(NMAX, np.float32)
        xy[:n], tok[:n], cp[:n], mask[:n] = it["xy"][:n], it["tok"][:n], it["cp"][:n], True
        if kind == "n":
            heat = np.asarray(self.rh[it["row"]])
            r_ = np.clip(np.round((xy[:n, 0] + 50) / 0.5 - 0.5).astype(int), 0, 199)
            c_ = np.clip(np.round((xy[:n, 1] + 50) / 0.5 - 0.5).astype(int), 0, 199)
            y[:n] = heat[r_, c_] >= 128
        return dict(xy=torch.from_numpy(xy), tok=torch.from_numpy(tok), cp=torch.from_numpy(cp), mask=torch.from_numpy(mask), y=torch.from_numpy(y), sd=torch.from_numpy(raster.astype(np.float16)), kind=0 if kind == "n" else (1 if kind == "w" else 2), dpose=torch.tensor(dpose, dtype=torch.float32), yaw=torch.tensor(yaw, dtype=torch.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok", required=True, help="token files (comma separated, without .pkl) of the training frames")
    ap.add_argument("--name", required=True)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--p_wrong", type=float, default=0.08)
    ap.add_argument("--p_none", type=float, default=0.08)
    ap.add_argument("--p_bias", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--loc", action="store_true", help="E103d: learned rough localisation (pose of the SD raster) inside the network")
    ap.add_argument("--corr", action="store_true", help="E103e: differentiable correlation-volume localisation (implies --loc)")
    ap.add_argument("--xattn", action="store_true", help="E104: cross-attention from the vertex tokens to the SD-map tokens")
    ap.add_argument("--w_pose", type=float, default=1.0)
    ap.add_argument("--tf", type=float, default=0.5, help="share of the iterations in which the route stage sees the raster warped by the TRUE pose (teacher forcing)")
    a = ap.parse_args()
    a.loc = a.loc or a.corr
    torch.manual_seed(a.seed)
    dev = "cuda"
    ds = TokData(a.tok.split(","), a.p_wrong, a.p_none, a.p_bias, a.seed)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True, persistent_workers=True)
    net = (PN.RouteXNet() if a.xattn else (PN.RouteCorrNet() if a.corr else (PN.RouteLocNet() if a.loc else PN.RouteGraphNet()))).to(dev)
    print(f"frames {len(ds)}, params {PN.count_params(net) / 1e6:.2f} M", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.iters, pct_start=0.1)
    scaler = torch.cuda.amp.GradScaler()
    d = f"{OUT}/{a.name}"
    os.makedirs(d, exist_ok=True)
    it, t0, run = 0, time.time(), {}
    log = open(f"{d}/log.txt", "a")
    while it < a.iters:
        for b in dl:
            b = {k: v.to(dev) for k, v in b.items()}
            net.train()
            pose_l = torch.tensor(0.0, device=dev)
            with torch.autocast("cuda", dtype=torch.float16):
                if a.loc:
                    tf = np.random.rand() < a.tf
                    lg, dh, yh = net(b["tok"], b["xy"], b["cp"], b["sd"].float(), b["mask"], pose_in=(b["dpose"], b["yaw"]) if tf else None)
                else:
                    lg = net(b["tok"], b["xy"], b["cp"], b["sd"].float(), b["mask"])
            loss_el = F.binary_cross_entropy_with_logits(lg.float(), b["y"], reduction="none", pos_weight=torch.tensor(2.0, device=dev))
            m = b["mask"].float()
            loss = (loss_el * m).sum() / m.sum().clamp(min=1)
            if a.loc:
                # pose supervision (right maps only; wrong / missing maps have no true pose); the pose head sees the un-warped raster in every iteration (teacher forcing only changes the route stage)
                if tf:
                    with torch.autocast("cuda", dtype=torch.float16):
                        _, d_hat, y_hat = net(b["tok"], b["xy"], b["cp"], b["sd"].float(), b["mask"])
                else:
                    d_hat, y_hat = dh, yh
                ok = (b["kind"] == 0).float()
                e_d = F.smooth_l1_loss(d_hat.float(), b["dpose"], reduction="none", beta=0.5).sum(1)
                e_y = F.smooth_l1_loss(torch.rad2deg(y_hat.float()) / 2.0, torch.rad2deg(b["yaw"]) / 2.0, reduction="none", beta=0.5)
                pose_l = ((e_d + e_y) * ok).sum() / ok.sum().clamp(min=1)
                loss = loss + a.w_pose * pose_l
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            with torch.no_grad():
                pred = (lg > 0).float()
                right = b["kind"] == 0
                acc = ((pred == b["y"]).float() * m).sum() / m.sum()
                pos = (b["y"] * m).sum()
                rec = ((pred * b["y"]) * m).sum() / pos.clamp(min=1)
                wr = (b["kind"] != 0).float()[:, None] * m
                fp_wrong = ((pred * wr).sum() / wr.sum().clamp(min=1)) if wr.sum() > 0 else torch.tensor(0.0)
            extra = {}
            if a.loc:
                with torch.no_grad():
                    okm = (b["kind"] == 0)
                    if okm.any():
                        extra = dict(d_err=float((d_hat.float() - b["dpose"]).norm(dim=1)[okm].mean()), yaw_err_deg=float(torch.rad2deg((y_hat.float() - b["yaw"]).abs())[okm].mean()), pose_loss=float(pose_l))
            for k_, v_ in dict(loss=float(loss), acc=float(acc), recall=float(rec), flagged_wrong=float(fp_wrong), **extra).items():
                run[k_] = 0.98 * run.get(k_, v_) + 0.02 * v_
            it += 1
            if it % 100 == 0 or it == 1:
                msg = f"it {it} {time.time() - t0:.0f}s " + " ".join(f"{k} {v:.3f}" for k, v in run.items())
                print(msg, flush=True)
                log.write(msg + "\n")
                log.flush()
            if it >= a.iters:
                break
    torch.save(net.state_dict(), f"{d}/model.pt")
    json.dump(dict(args=vars(a), final=run, seconds=time.time() - t0), open(f"{d}/train_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
