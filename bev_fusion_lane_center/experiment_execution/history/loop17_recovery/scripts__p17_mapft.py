"""Loop 17 / E138: train the map-conditioned builder (p17_mapnet.py) from the champion: TRAIN frames with a random map situation (pose error 0 - 3 m, wrong map 10 %, no map 10 %) and perception loss on purpose (40 %).
Time recursion off (E132: no effect). Usage (gcar, GPU): p17_mapft.py --name e17_map_a --u 6 --lr 5e-4"""
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
import p16_train as PT  # noqa: E402
import p17_eval as EV  # noqa: E402
import p17_mapnet as MN  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", default="D7")
    ap.add_argument("--name", required=True)
    ap.add_argument("--init", default="e123_D7_r3_f1")
    ap.add_argument("--set", nargs="*", default=["M=288", "K0=144"])
    ap.add_argument("--u", type=float, default=6.0)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--stem_lr", type=float, default=0.6)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--p_degrade", type=float, default=0.4)
    ap.add_argument("--p_none", type=float, default=0.1)
    ap.add_argument("--p_wrong", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    dev = "cuda"
    od = f"{C.OUT}/e123/{a.name}"
    os.makedirs(od, exist_ok=True)
    net, cfg = EV.load_net(a.design, a.init, a.set)
    cfg.temporal = False
    net.cfg.temporal = False
    MN.widen(net)
    net.train()
    iters = max(10, int(a.u * PT.IT_PER_U))
    stem_ps = [p for n, p in net.named_parameters() if n.startswith("stem.")]
    rest = [p for n, p in net.named_parameters() if not n.startswith("stem.")]
    opt = torch.optim.AdamW([dict(params=rest, lr=a.lr), dict(params=stem_ps, lr=a.lr * a.stem_lr)], weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr, a.lr * a.stem_lr], total_steps=iters, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda")
    base = PD.BuilderData("train", ks=None, train=True, with_prev=False)
    P = dict(np.load(f"{PD.D}/poses_train.npz"))
    ds = MN.MapWrap(base, "train", P, train=True, p_none=a.p_none, p_wrong=a.p_wrong, p_degrade=a.p_degrade, seed=a.seed)
    dl = torch.utils.data.DataLoader(ds, batch_size=PT.BS, shuffle=True, num_workers=a.workers, collate_fn=PD.collate, drop_last=True, persistent_workers=True, prefetch_factor=3)
    log = open(f"{od}/log.txt", "a")
    run, it, t0, bad, status = {}, 0, time.time(), 0, "ok"
    while it < iters and status == "ok":
        for b in dl:
            b = {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}
            net.train()
            share = 0.5 * max(0.0, 1.0 - it / max(1.0, 0.1 * iters)) if cfg.warm else 0.0
            with torch.autocast("cuda", dtype=torch.float16):
                res = net(b["x"].float(), gt=b if share > 0 else None, share=share, prior=None)
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
