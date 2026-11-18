"""Loop 17 / E138: train the map-conditioned builder (p17_mapnet.py) from the champion: TRAIN frames with a random map situation (pose error 0 - 3 m, wrong map 10 %, no map 10 %) and perception loss on purpose (40 %).
Time recursion off (E132: no effect). Usage (gcar, GPU): p17_mapft.py --name e17_map_a --u 6 --lr 5e-4
Resume (added 2026-10-24 after the E209 crash lost 2 h): every --ckpt_every steps runs/e123/<name>/ckpt.pt is written atomically (model, optimiser, schedule, scaler, step, loss averages, RNG);
starting the same command again continues from it (--resume 0 ignores it). model.pt / status.json are written only when the last step has finished."""
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
    ap.add_argument("--real_fail", type=int, default=0, help="1: half of the frames are real failure frames of the TRAIN rows (seven types, p17_dense_train.py)")
    ap.add_argument("--share", type=float, default=0.5)
    ap.add_argument("--fail_perts", default="dark,blur,cam_front3_off,fog,radar_off,extr3,cam_front_off", help="failure types of the real-failure training rows (leave types out to test unseen ones)")
    ap.add_argument("--init_ckpt", default=None, help="start from the map-conditioned weights of this run (47 input channels)")
    ap.add_argument("--ckpt_every", type=int, default=400, help="steps between checkpoints (0 = none)")
    ap.add_argument("--resume", type=int, default=1, help="1: continue from runs/e123/<name>/ckpt.pt when it exists")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    dev = "cuda"
    od = f"{C.OUT}/e123/{a.name}"
    os.makedirs(od, exist_ok=True)
    net, cfg = EV.load_net(a.design, a.init, a.set)
    cfg.temporal = False
    net.cfg.temporal = False
    MN.widen(net)
    if a.init_ckpt:
        net.load_state_dict(torch.load(f"{C.OUT}/e123/{a.init_ckpt}/model.pt", map_location=dev), strict=False)
    net.train()
    iters = max(10, int(a.u * PT.IT_PER_U))
    stem_ps = [p for n, p in net.named_parameters() if n.startswith("stem.")]
    rest = [p for n, p in net.named_parameters() if not n.startswith("stem.")]
    opt = torch.optim.AdamW([dict(params=rest, lr=a.lr), dict(params=stem_ps, lr=a.lr * a.stem_lr)], weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr, a.lr * a.stem_lr], total_steps=iters, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda")
    log = open(f"{od}/log.txt", "a")
    run, it, t0, bad, status = {}, 0, time.time(), 0, "ok"
    ck = f"{od}/ckpt.pt"
    if a.resume and os.path.exists(ck):
        s = torch.load(ck, map_location=dev, weights_only=False)
        net.load_state_dict(s["model"])
        opt.load_state_dict(s["opt"])
        sched.load_state_dict(s["sched"])
        scaler.load_state_dict(s["scaler"])
        it, run = s["it"], s["run"]
        torch.set_rng_state(s["rng_torch"].cpu())
        np.random.set_state(s["rng_np"])
        msg = f"resumed from {ck} at step {it}/{iters}"
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()
        del s
    base = PD.BuilderData("train", ks=None, train=True, with_prev=False)
    P = dict(np.load(f"{PD.D}/poses_train.npz"))
    ds = MN.MapWrap(base, "train", P, train=True, p_none=a.p_none, p_wrong=a.p_wrong, p_degrade=a.p_degrade, seed=a.seed)
    if a.real_fail:
        fail = MN.MapWrap(MN.RealFail(base, a.fail_perts.split(",")), "train", P, train=True, p_none=a.p_none, p_wrong=a.p_wrong, p_degrade=0.0, seed=a.seed + 1)
        ds = MN.Mix2(ds, fail, a.share, n=iters * PT.BS + 1000, seed=a.seed)
    # real-failure mix: fixed index order, so a resumed run continues at the batch where it stopped (the clean-only mode reshuffles)
    sampler = list(range(min(it * PT.BS, max(0, len(ds) - 4 * PT.BS)), len(ds))) if (a.real_fail and it) else None
    dl = torch.utils.data.DataLoader(ds, batch_size=PT.BS, shuffle=(not a.real_fail) if sampler is None else False, sampler=sampler, num_workers=a.workers, collate_fn=PD.collate,
                                     drop_last=True, persistent_workers=True, prefetch_factor=3)
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
            if a.ckpt_every and it % a.ckpt_every == 0 and it < iters:
                tmp = ck + ".tmp"
                torch.save(dict(model=net.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), scaler=scaler.state_dict(), it=it, run=run,
                                rng_torch=torch.get_rng_state(), rng_np=np.random.get_state()), tmp)
                os.replace(tmp, ck)
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
