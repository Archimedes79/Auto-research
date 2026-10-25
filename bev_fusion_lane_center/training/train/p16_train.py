"""Loop 16: train one design of the ladder at a fidelity and evaluate it on the fixed VAL subset (validity sweep, metric card).
Usage (gcar, GPU): p16_train.py --design D3 --name e123_D3_r0_00 --u 1 [--seed 0] [--init e123_D3_r0_00] [--lr 1e-3] [--set key=value ...]
1 u = 1200 iterations x 8 frames (= 9,600 frames, the same data volume as 600 iterations x 16 frames of the control arm)."""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p16_builder as PB  # noqa: E402
import p16_common as C  # noqa: E402
import p16_data as PD  # noqa: E402
import p16_infer as PI  # noqa: E402
import p16_loss as PL  # noqa: E402
import p16_time as PT_  # noqa: E402

DESIGNS = {  # factors A..G of the L8 array (plan section 3)
    "D1": dict(constr="G", pattern="fixed", seeds="lattice", dedup="geo", ctx="local", warm=False, revisit=False),
    "D2": dict(constr="G", pattern="fixed", seeds="lattice", dedup="learned", ctx="global", warm=True, revisit=True),
    "D3": dict(constr="G", pattern="teardrop", seeds="anchors", dedup="geo", ctx="local", warm=True, revisit=True),
    "D4": dict(constr="G", pattern="teardrop", seeds="anchors", dedup="learned", ctx="global", warm=False, revisit=False),
    "D5": dict(constr="L", pattern="fixed", seeds="anchors", dedup="geo", ctx="global", warm=False, revisit=True),
    "D6": dict(constr="L", pattern="fixed", seeds="anchors", dedup="learned", ctx="local", warm=True, revisit=False),
    "D7": dict(constr="L", pattern="teardrop", seeds="lattice", dedup="geo", ctx="global", warm=True, revisit=False),
    "D8": dict(constr="L", pattern="teardrop", seeds="lattice", dedup="learned", ctx="local", warm=False, revisit=True),
}
# literature baseline arms beside the ladder (never halved): B1 = geometric link prior on pieces (TopoLogic form, K30), B3 = one pooled score per ray (line-of-interest verifier, K40)
DESIGNS["B1"] = dict(DESIGNS["D7"], link_mode="geo")
DESIGNS["B3"] = dict(DESIGNS["D3"], ver_mode="loi")
DESIGNS["H"] = dict(DESIGNS["D7"], constr="H")  # hybrid challenge: L pieces and growth (3 levels) with the link head on the ends of the last level; parents / factors are set at rung 1
IT_PER_U = 1200
BS = 8


def parse_set(items):
    out = {}
    for s in items or []:
        k, v = s.split("=")
        try:
            v = json.loads(v)
        except Exception:
            pass
        out[k] = v
    return out


def build(design, extra=None):
    kw = dict(DESIGNS[design])
    kw.update(extra or {})
    cfg = PB.DefaultCfg(**{k: v for k, v in kw.items() if k not in ("w",)})
    anch = np.load(f"{C.LADDER}/anchors128.npy") if cfg.seeds == "anchors" else None
    return PB.Builder(cfg, anchors=anch), cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--u", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default=None, help="continue from the checkpoint of this run (refinement)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--stem_lr", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key=value")
    ap.add_argument("--w", nargs="*", default=[], help="loss weight overrides key=value")
    ap.add_argument("--warm_frac", type=float, default=0.4)
    ap.add_argument("--eval", default="val250", help="val250 | val | none")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = "cuda"
    od = f"{C.OUT}/e123/{a.name}"
    os.makedirs(od, exist_ok=True)
    extra = parse_set(a.set)
    net, cfg = build(a.design, extra)
    net = net.to(dev)
    if a.init:
        sd = torch.load(f"{C.OUT}/e123/{a.init}/model.pt", map_location=dev)
        own = net.state_dict()
        skipped = [k for k, v in sd.items() if k in own and own[k].shape != v.shape]  # a changed ray ladder (SPF3) changes the shape of the geometry embedding: it restarts
        sd = {k: v for k, v in sd.items() if k not in skipped}
        if skipped:
            print("shape mismatch, not loaded:", skipped, flush=True)
        miss = net.load_state_dict(sd, strict=False)
        print("init from", a.init, "missing", len(miss.missing_keys), "unexpected", len(miss.unexpected_keys), flush=True)
    iters = max(10, int(a.u * IT_PER_U)) if not a.smoke else 20
    stem_ps = [p for n, p in net.named_parameters() if n.startswith("stem.")]
    rest = [p for n, p in net.named_parameters() if not n.startswith("stem.")]
    opt = torch.optim.AdamW([dict(params=rest, lr=a.lr), dict(params=stem_ps, lr=a.lr * a.stem_lr)], weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr, a.lr * a.stem_lr], total_steps=iters, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda")
    ds, dl = PD.loader("train", bs=BS, workers=a.workers, with_prev=cfg.temporal)
    wts = {k: float(v) for k, v in parse_set(a.w).items()}
    log = open(f"{od}/log.txt", "a")
    print(f"design {a.design} cfg {json.dumps(cfg.asdict())} iters {iters} params {sum(p.numel() for p in net.parameters()) / 1e6:.2f} M", flush=True)
    run, it, t0, bad = {}, 0, time.time(), 0
    status = "ok"
    while it < iters:
        for b in dl:
            b = {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}
            net.train()
            share = 0.5 * max(0.0, 1.0 - it / max(1.0, a.warm_frac * iters)) if cfg.warm else 0.0
            with torch.autocast("cuda", dtype=torch.float16):
                res = net(b["x"].float(), gt=b if share > 0 else None, share=share, prior=(PT_.prior_from_prev(net, b, train=True) if cfg.temporal else None))
            loss, parts = PL.total_loss(net, res, b, wts)
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
            gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            parts["gn"] = float(gn)
            for k, v in parts.items():
                if np.isfinite(v):
                    run[k] = 0.97 * run.get(k, v) + 0.03 * v
            it += 1
            if it % 100 == 0 or it == 1:
                msg = f"it {it}/{iters} {time.time() - t0:.0f}s share {share:.2f} " + " ".join(f"{k} {v:.3f}" for k, v in run.items())
                print(msg, flush=True)
                log.write(msg + "\n")
                log.flush()
            if it >= iters:
                break
        if status != "ok":
            break
    torch.save(net.state_dict(), f"{od}/model.pt")
    train_sec = time.time() - t0
    card = dict(run=a.name, kind="design", design=a.design, cfg=cfg.asdict(), u=a.u, iters=iters, seed=a.seed, init=a.init, args=vars(a), status=status, train_seconds=train_sec, final_loss={k: float(v) for k, v in run.items()})
    if a.eval != "none" and status == "ok":
        t1 = time.time()
        if a.eval == "val250":
            ks = C.val_subset()
        else:
            ks = None
        ds_v = PD.BuilderData("val_clean", ks=ks, train=False, with_prev=cfg.temporal)
        graphs, gts, sec = PI.predict(net, ds_v, ks, workers=a.workers)
        ev = PI.evaluate_sweep(graphs, gts)
        card.update(eval=a.eval, n=len(gts), metrics=ev["best"], metrics_tau05=ev["at05"], sweep=ev["sweep"], **{k: ev[k] for k in ("F1@Sp17", "FL@R85")}, eval_seconds=time.time() - t1)
        m = ev["best"]
        print(a.name, f"S {m['S']:.4f} tau {m['tau']} F1@1 {m['F1@1.0']:.4f} F1@.5 {m['F1@0.5']:.4f} J {m['J']:.3f} Sp {m['Sp']:.3f} v100 {m['vert100']:.1f} FL {m['false_len_frac']:.3f}", flush=True)
    C.write_card(a.name, card)
    json.dump(dict(status=status), open(f"{od}/status.json", "w"))


if __name__ == "__main__":
    main()
