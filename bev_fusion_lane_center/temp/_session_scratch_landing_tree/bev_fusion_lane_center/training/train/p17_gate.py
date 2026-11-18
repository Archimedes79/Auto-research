"""Loop 17 / E134: a LEARNED gated memory in front of the (frozen) loop-16 builder, trained through the builder loss.
Stabilizer: the warped memory y_{t-1} (42 channels, ego-motion warp as in p13_temporal_val.py) and the current frame x_t give per-pixel gates (a small conv net on heat / direction of both, a validity mask, the maxdecay decision and a
feature-mismatch map); y_t = g * m + (1 - g) * x_t separately for the 39 feature channels and the 3 map channels. The gate is initialised so that it reproduces the training-free spatial hold of E131 (decision of maxdecay, +-6 in the logit, decay 0.95);
the conv output starts at zero. y_t is the next memory. Differentiable end to end: warp = grid_sample, gate = convs, builder = frozen loop-16 network, loss = builder loss (p16_loss) of frame t. Unroll 3 frames, memory detached before the first.
Training uses the failure-phase frames (70 %) and clean-phase frames (30 %) of the VAL failure series of the scenes that are NOT in the held-out fold; apply writes the fused series for the held-out scenes with the model that did not see them.
Usage (gcar, GPU): p17_gate.py train --fold 0 --folds 2 --name e17_gate_k0 [--steps 1500]   |   p17_gate.py apply --names e17_gate_k0,e17_gate_k1 --perts dark,blur,cam_front3_off --out gate
  | p17_gate.py apply --init_only --perts ... --out gate0   (the untrained gate: must reproduce maxdecay)"""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p13_temporal_val as TV  # noqa: E402
import p16_common as C  # noqa: E402
import p16_data as PD  # noqa: E402
import p16_loss as PL  # noqa: E402
import p17_eval as EV  # noqa: E402

DEV = "cuda"
OUTD = EV.OUTD


class Stabilizer(nn.Module):
    def __init__(self, hid=32):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(11, hid, 3, padding=1), nn.GELU(), nn.Conv2d(hid, hid, 3, padding=2, dilation=2), nn.GELU(), nn.Conv2d(hid, 2, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.a_logit = nn.Parameter(torch.tensor(math.log(0.95 / 0.05)))

    def forward(self, x, m, valid):
        """x, m: [1, 42, H, W] (m warped, direction rotated), valid [1, 1, H, W] -> y [1, 42, H, W]"""
        a = torch.sigmoid(self.a_logit)
        wd = a * m[:, 39:40]
        use = ((valid > 0.5) & (wd > x[:, 39:40])).float()
        mis = (x[:, :39] - m[:, :39]).abs().mean(1, keepdim=True)
        inp = torch.cat([x[:, 39:42], m[:, 39:42], use, valid, mis, x[:, :39].abs().mean(1, keepdim=True), m[:, :39].abs().mean(1, keepdim=True)], 1)
        o = self.net(inp)
        base = 6.0 * (use * 2 - 1)
        gm, gf = torch.sigmoid(base + o[:, 0:1]), torch.sigmoid(base + o[:, 1:2])
        md = torch.cat([m[:, :39], wd, m[:, 40:42]], 1)
        return torch.cat([gf * md[:, :39] + (1 - gf) * x[:, :39], gm * md[:, 39:42] + (1 - gm) * x[:, 39:42]], 1)


def warp_mem(y, P, rp, rc):
    """memory y [1, 42, H, W] of frame row rp -> frame row rc"""
    Rp, tp, Rc, tc = P["R"][rp], P["t"][rp], P["R"][rc], P["t"][rc]
    w, valid = TV.warp(y[0], TV.make_grid(Rp, tp, Rc, tc))
    rel = torch.as_tensor(np.asarray(Rc, np.float64).T @ np.asarray(Rp, np.float64), device=DEV, dtype=torch.float32)
    w = w.clone()
    w[40:42] = (rel @ w[40:42].reshape(2, -1)).reshape(2, 200, 200)
    return w[None], valid[None, None].float()


def chain(model, ds, i, n_back, P):
    """fused input y_t of series frame i, unrolled over up to n_back earlier frames of the same scene (memory of the first one = none)"""
    idx = [i]
    for _ in range(n_back):
        p = int(ds.meta["prev"][idx[-1]])
        if p < 0:
            break
        idx.append(p)
    idx = idx[::-1]
    y = None
    for k, j in enumerate(idx):
        x = torch.from_numpy(np.asarray(ds.mm17[j], np.float32)).to(DEV)[None]
        if y is None:
            y = x
        else:
            m, v = warp_mem(y.detach() if k == 1 else y, P, int(ds.meta["rows"][idx[k - 1]]), int(ds.meta["rows"][j]))
            y = model(x, m, v)
    return y


def train(a):
    torch.manual_seed(a.seed)
    rng = np.random.RandomState(a.seed)
    net, cfg = EV.load_net(a.design, a.init, a.set)
    cfg.temporal = False
    net.cfg.temporal = False
    if a.mapnet:  # a map-conditioned builder behind the gate: the 5 SD channels are appended to the fused 42 channels
        import p17_mapnet as MN

        MN.widen(net)
        net.load_state_dict(torch.load(f"{C.OUT}/e123/{a.mapnet}/model.pt", map_location=DEV), strict=False)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    model = Stabilizer().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.1)
    sets = []
    for pn in a.perts.split(","):
        ds = EV.SeriesData(f"{pn}_single", with_prev=False)
        scenes = sorted(set(int(s) for s in ds.meta["scene"]))
        held = {s for k, s in enumerate(scenes) if k % a.folds == a.fold}
        ok = [i for i in range(len(ds)) if int(ds.meta["scene"][i]) not in held]
        sets.append((ds, [i for i in ok if ds.meta["phase"][i] == 1], [i for i in ok if ds.meta["phase"][i] == 0]))
    P = sets[0][0].P17
    mw = None
    if a.mapnet:
        import p8_lib as L8

        mw = MN.MapWrap(sets[0][0], "val", P, train=False)
        mw.sdl = L8.cache("sd", "val")
    od = f"{C.OUT}/e123/{a.name}"
    os.makedirs(od, exist_ok=True)
    log = open(f"{od}/log.txt", "a")
    run, t0 = {}, time.time()
    for it in range(1, a.steps + 1):
        ys, items = [], []
        for _ in range(a.bs):
            ds, fail, clean = sets[int(rng.randint(len(sets)))]
            pool = fail if rng.rand() < 0.7 else clean
            i = int(pool[rng.randint(len(pool))])
            yy = chain(model, ds, i, a.unroll, P)
            if mw is not None:
                row_ = int(ds.meta["rows"][i])
                u_ = rng.rand()
                mode_ = "none" if u_ < 0.1 else ("wrong" if u_ < 0.2 else "exact")
                ras = torch.from_numpy(mw.raster(row_, rng, mode_, float(rng.choice([0.0, 1.0, 2.0])))).to(DEV)[None]
                yy = torch.cat([yy, ras], 1)
            ys.append(yy)
            items.append(ds[i])
        b = PD.collate(items)
        b = {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in b.items()}
        with torch.autocast("cuda", dtype=torch.float16):
            res = net(torch.cat(ys, 0).float(), gt=None, share=0.0, prior=None)
        loss, parts = PL.total_loss(net, res, b, {})
        if not torch.isfinite(loss):
            opt.zero_grad(set_to_none=True)
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run["loss"] = 0.97 * run.get("loss", float(loss)) + 0.03 * float(loss)
        if it % 50 == 0 or it == 1:
            msg = f"it {it}/{a.steps} {time.time() - t0:.0f}s loss {run['loss']:.3f} decay {float(torch.sigmoid(model.a_logit)):.3f}"
            print(msg, flush=True)
            log.write(msg + "\n")
            log.flush()
    torch.save(model.state_dict(), f"{od}/gate.pt")
    print("DONE", f"{time.time() - t0:.0f} s", flush=True)


@torch.no_grad()
def apply(a):
    P = None
    models = []
    for nm in (a.names.split(",") if a.names else []):
        m = Stabilizer().to(DEV)
        m.load_state_dict(torch.load(f"{C.OUT}/e123/{nm}/gate.pt", map_location=DEV))
        models.append(m.eval())
    if a.init_only:
        models = [Stabilizer().to(DEV).eval()]
    folds = len(models)
    for pn in a.perts.split(","):
        ds = EV.SeriesData(f"{pn}_{a.src}", with_prev=False)
        P = ds.P17
        scenes = sorted(set(int(s) for s in ds.meta["scene"]))
        fold_of = {s: k % folds for k, s in enumerate(scenes)}
        n = len(ds)
        tag = f"{pn}_{a.out}"
        mm = np.lib.format.open_memmap(f"{OUTD}/{tag}.npy", mode="w+", dtype=np.float16, shape=(n, 42, 200, 200))
        y = None
        for i in range(n):
            x = torch.from_numpy(np.asarray(ds.mm17[i], np.float32)).to(DEV)[None]
            p = int(ds.meta["prev"][i])
            if p < 0 or y is None:
                yi = x
            else:
                m, v = warp_mem(y, P, int(ds.meta["rows"][p]), int(ds.meta["rows"][i]))
                yi = models[fold_of[int(ds.meta["scene"][i])]](x, m, v)
            mm[i] = yi[0].cpu().numpy().astype(np.float16)
            y = yi
        mm.flush()
        np.savez(f"{OUTD}/{tag}.meta.npz", **ds.meta)
        print(f"[{pn}] wrote {tag}", flush=True)
    print("DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "apply"])
    ap.add_argument("--design", default="D7")
    ap.add_argument("--init", default="e123_D7_r3_f1")
    ap.add_argument("--set", nargs="*", default=["M=288", "K0=144"])
    ap.add_argument("--name", default="e17_gate_k0")
    ap.add_argument("--names", default="")
    ap.add_argument("--perts", default="dark,blur,cam_front3_off")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--folds", type=int, default=2)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--unroll", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="gate")
    ap.add_argument("--init_only", action="store_true")
    ap.add_argument("--src", default="single", help="apply: the series (single, single_cleanref) the gate is run over")
    ap.add_argument("--mapnet", default="", help="run name of a map-conditioned builder (47 channels) to train the gate behind")
    a = ap.parse_args()
    train(a) if a.mode == "train" else apply(a)
