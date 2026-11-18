"""Loop 17 / E133b: failure-frame fine-tuning of a loop-16 builder, scene-disjoint.
Batches are half clean TRAIN frames (as in p16_train.py) and half failure-phase frames of the VAL failure series (dark / blur / three front cameras off; inputs as single frame, spatial hold and EMA + monitor hold) of the scenes that are
NOT in the held-out fold. Fold k of F: scene index (sorted scene ids of the series) % F == k is held out and evaluated by p17_eval.py --ckpt <name> (all frames are written; the evaluation of a fold uses only its scenes, p17_merge.py joins the folds).
Usage (gcar, GPU): p17_ft.py --name e17_ft_k0 --init e123_D7_r3_f1 --fold 0 --folds 2 --u 1.5 --lr 3e-4"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p16_common as C  # noqa: E402
import p16_data as PD  # noqa: E402
import p16_loss as PL  # noqa: E402
import p16_time as PT_  # noqa: E402
import p16_train as PT  # noqa: E402
import p17_eval as EV  # noqa: E402


class Mix(torch.utils.data.Dataset):
    def __init__(self, clean, fails, share=0.5, n=10 ** 7, seed=0):
        self.clean, self.fails, self.share, self.n, self.seed = clean, fails, share, n, seed

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.RandomState((self.seed * 1000003 + i) % (2 ** 32))
        if rng.rand() >= self.share:
            return self.clean[int(rng.randint(len(self.clean)))]
        ds, idx = self.fails[int(rng.randint(len(self.fails)))]
        return ds[int(idx[rng.randint(len(idx))])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", default="D7")
    ap.add_argument("--name", required=True)
    ap.add_argument("--init", default="e123_D7_r3_f1")
    ap.add_argument("--set", nargs="*", default=["M=288", "K0=144"])
    ap.add_argument("--series", default="dark_single,dark_maxdecay_0.95,dark_ema_hold_mon_0.5,blur_single,blur_maxdecay_0.95,blur_ema_hold_mon_0.5,cam_front3_off_single,cam_front3_off_maxdecay_0.95,cam_front3_off_ema_hold_mon_0.5")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--folds", type=int, default=2)
    ap.add_argument("--u", type=float, default=1.5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--stem_lr", type=float, default=0.3)
    ap.add_argument("--share", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--warm_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = "cuda"
    od = f"{C.OUT}/e123/{a.name}"
    os.makedirs(od, exist_ok=True)
    net, cfg = EV.load_net(a.design, a.init, a.set)
    net.train()
    iters = max(10, int(a.u * PT.IT_PER_U))
    stem_ps = [p for n, p in net.named_parameters() if n.startswith("stem.")]
    rest = [p for n, p in net.named_parameters() if not n.startswith("stem.")]
    opt = torch.optim.AdamW([dict(params=rest, lr=a.lr), dict(params=stem_ps, lr=a.lr * a.stem_lr)], weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr, a.lr * a.stem_lr], total_steps=iters, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda")
    clean = PD.BuilderData("train", ks=None, train=True, with_prev=cfg.temporal)
    fails = []
    for tag in a.series.split(","):
        ds = EV.SeriesData(tag, with_prev=bool(cfg.temporal))
        scenes = sorted(set(int(s) for s in ds.meta["scene"]))
        held = {s for k, s in enumerate(scenes) if k % a.folds == a.fold}
        idx = [i for i in range(len(ds)) if ds.meta["phase"][i] == 1 and int(ds.meta["scene"][i]) not in held]
        fails.append((ds, idx))
    print(f"fold {a.fold}/{a.folds}: failure items per series {[len(i) for _, i in fails]}, held-out scenes {sorted(held)}", flush=True)
    mix = Mix(clean, fails, a.share, n=iters * PT.BS + 1000, seed=a.seed)
    dl = torch.utils.data.DataLoader(mix, batch_size=PT.BS, shuffle=False, num_workers=a.workers, collate_fn=PD.collate, drop_last=True, persistent_workers=a.workers > 0, prefetch_factor=3 if a.workers > 0 else None)
    log = open(f"{od}/log.txt", "a")
    run, it, t0, bad = {}, 0, time.time(), 0
    status = "ok"
    for b in dl:
        b = {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}
        net.train()
        share = 0.5 * max(0.0, 1.0 - it / max(1.0, a.warm_frac * iters)) if cfg.warm else 0.0
        with torch.autocast("cuda", dtype=torch.float16):
            res = net(b["x"].float(), gt=b if share > 0 else None, share=share, prior=(PT_.prior_from_prev(net, b, train=True) if cfg.temporal else None))
        loss, parts = PL.total_loss(net, res, b, {})
        if not torch.isfinite(loss):
            bad += 1
            opt.zero_grad(set_to_none=True)
            if bad > 20:
                status = "KILL nan"
                break
            continue
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        for k, v in parts.items():
            if np.isfinite(v):
                run[k] = 0.97 * run.get(k, v) + 0.03 * v
        it += 1
        if it % 100 == 0 or it == 1:
            msg = f"it {it}/{iters} {time.time() - t0:.0f}s " + " ".join(f"{k} {v:.3f}" for k, v in run.items())
            print(msg, flush=True)
            log.write(msg + "\n")
            log.flush()
        if it >= iters:
            break
    torch.save(net.state_dict(), f"{od}/model.pt")
    json.dump(dict(status=status, args=vars(a), seconds=time.time() - t0, final_loss={k: float(v) for k, v in run.items()}), open(f"{od}/status.json", "w"))
    print("DONE", status, f"{time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
