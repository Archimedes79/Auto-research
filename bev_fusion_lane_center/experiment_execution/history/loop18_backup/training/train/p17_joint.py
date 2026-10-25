"""Loop 17 / E136: joint end-to-end fine-tune of the builder and the route head along the edges.
Chain per frame: builder (trainable) -> assembly (beam, discrete, no gradient: it only decides WHICH nodes and edges exist) -> route input (the builder's node tokens gathered at the surviving vertices and linearly interpolated along the edges: a
fixed sparse matrix, so the gradient of the route loss reaches the builder) -> route head with cross-attention to the SD-map raster (trainable) -> route label per point (BCE, as p10_route_train.py with the loop-9 map faults).
Loss = route BCE + lambda * builder loss (p16_loss, keeps the graph quality). Time recursion off, builder in eval mode (BN statistics fixed), only weights learn.
Usage (gcar, GPU): p17_joint.py --name e17_joint_a --iters 600 --bs 4 --lam 1.0 [--route p16_f2hr_route_s0]"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p10_net as PN  # noqa: E402
import p10_route_train as RTR  # noqa: E402
import p16_common as C  # noqa: E402
import p16_data as PD  # noqa: E402
import p16_infer as PI  # noqa: E402
import p16_loss as PL  # noqa: E402
import p16_tok_resample as TR  # noqa: E402
import p16_tokens as TK  # noqa: E402
import p17_eval as EV  # noqa: E402
import p8_lib as L  # noqa: E402

DEV = "cuda"


def resample_matrix(it, step):
    """the interpolation of p16_tok_resample.resample_item as a matrix [n_new, n_old] (same order of the new points)"""
    xy = it["xy"].astype(np.float64)
    has = it["has"].astype(bool)
    n = len(xy)
    rows = [({i: 1.0} if has[i] else {}) for i in range(n)]
    seen = set()
    for path in it["pieces"]:
        for a, b in zip(path[:-1], path[1:]):
            key = (min(a, b), max(a, b))
            if key in seen:
                continue
            seen.add(key)
            lo, hi = key
            m = int(np.ceil(float(np.linalg.norm(xy[hi] - xy[lo])) / step)) - 1
            for j in range(1, m + 1):
                w = j / (m + 1)
                if has[lo] and has[hi]:
                    rows.append({lo: 1 - w, hi: w})
                elif has[lo]:
                    rows.append({lo: 1.0})
                elif has[hi]:
                    rows.append({hi: 1.0})
                else:
                    rows.append({})
    M = np.zeros((len(rows), n), np.float32)
    for r, d in enumerate(rows):
        for c, w in d.items():
            M[r, c] = w
    return M


def route_item(r, ax, g, step=4.0):
    """structure of the route input of one frame (numpy): the resampled item and the interpolation matrix; tokens are added by the caller (differentiably)"""
    V = np.asarray(ax["V"], np.float32).reshape(-1, 2)
    src = np.asarray(ax["src"], int)
    has = src >= 0
    keep = np.where(has, r["pi"][np.clip(src, 0, None)], 0.5).astype(np.float32)
    it0 = dict(xy=V, tok=np.zeros((len(V), 128), np.float16), has=has, cp=TK.cross_section(V, ax["edges"]), pieces=[list(map(int, p)) for p in g["nodes"]], edges=sorted((int(x), int(y)) for x, y in ax["edges"]), keep=keep)
    it1 = TR.resample_item(it0, step)
    M = resample_matrix(it0, step)
    assert len(M) == len(it1["xy"])
    return it0, it1, M, src, has


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", default="D7")
    ap.add_argument("--name", required=True)
    ap.add_argument("--init", default="e123_D7_r3_f1")
    ap.add_argument("--route", default="p16_f2hr_route_s0")
    ap.add_argument("--set", nargs="*", default=["M=192", "K0=96"])
    ap.add_argument("--asm", nargs="*", default=[f"{k}={v}" for k, v in EV.CHAMP_ASM.items()])
    ap.add_argument("--tau", type=float, default=0.7)
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr_b", type=float, default=1e-4)
    ap.add_argument("--lr_r", type=float, default=3e-4)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_builder_grad", action="store_true", help="control: the same training, but the route loss does not reach the builder (tokens detached)")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.RandomState(a.seed)
    asm = {k: float(v) for k, v in (x.split("=") for x in a.asm)}
    net, cfg = EV.load_net(a.design, a.init, a.set)
    cfg.temporal = False
    net.cfg.temporal = False
    net.eval()
    rnet = PN.RouteXNet().to(DEV)
    rnet.load_state_dict(torch.load(f"{L.OUT}/{a.route}/model.pt", map_location=DEV))
    rnet.train()
    stem = [p for n, p in net.named_parameters() if n.startswith("stem.")]
    rest = [p for n, p in net.named_parameters() if not n.startswith("stem.")]
    opt = torch.optim.AdamW([dict(params=rest, lr=a.lr_b), dict(params=stem, lr=a.lr_b * 0.3), dict(params=rnet.parameters(), lr=a.lr_r)], weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr_b, a.lr_b * 0.3, a.lr_r], total_steps=a.iters, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda")
    ds, dl = PD.loader("train", bs=a.bs, workers=a.workers, with_prev=False)
    sdl = L.cache("sd", "train")
    rh = np.load(f"{L.D}/route_heat_train.npy", mmap_mode="r")
    od = f"{L.OUT}/{a.name}"
    os.makedirs(od, exist_ok=True)
    log = open(f"{od}/log.txt", "a")
    run, it, t0 = {}, 0, time.time()
    while it < a.iters:
        for b in dl:
            b = {k: (v.to(DEV, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}
            with torch.autocast("cuda", dtype=torch.float16):
                res = net(b["x"].float(), gt=None, share=0.0, prior=None)
            lb, parts = PL.total_loss(net, res, b, {})
            lr_, pos = [], []
            gk = 0
            for i in range(b["x"].shape[0]):
                row = int(b["row"][i])
                r = PI.to_numpy(res, i)
                g, ax = PI.assemble(r, None, a.tau, cfg, aux=True, **asm)
                if not len(ax["V"]):
                    continue
                it0, it1, M, src, has = route_item(r, ax, g)
                h = res["h"][i].float()
                if a.no_builder_grad:
                    h = h.detach()
                tok_old = torch.zeros(len(it0["xy"]), 128, device=DEV)
                if has.any():
                    hh = h[torch.as_tensor(src[has], device=DEV)]
                    tok_old[torch.as_tensor(np.where(has)[0], device=DEV), : hh.shape[1]] = hh
                tok = torch.from_numpy(M).to(DEV) @ tok_old
                raster, kind, _, _ = RTR.draw_cell(sdl[row], sdl, row, rng, 0.08, 0.08, 0.3)
                n = len(it1["xy"])
                y = np.zeros(n, np.float32)
                if kind == "n":
                    r_ = np.clip(np.round((it1["xy"][:, 0] + 50) / 0.5 - 0.5).astype(int), 0, 199)
                    c_ = np.clip(np.round((it1["xy"][:, 1] + 50) / 0.5 - 0.5).astype(int), 0, 199)
                    y = (np.asarray(rh[row])[r_, c_] >= 128).astype(np.float32)
                xy_t = torch.from_numpy(it1["xy"])[None].to(DEV)
                cp_t = torch.from_numpy(it1["cp"])[None].to(DEV)
                sd_t = torch.from_numpy(raster)[None].to(DEV)
                mask = torch.ones(1, n, dtype=torch.bool, device=DEV)
                with torch.autocast("cuda", dtype=torch.float16):
                    lg = rnet(tok[None], xy_t, cp_t, sd_t, mask)
                pos.append(float(y.mean()))
                lr_.append(F.binary_cross_entropy_with_logits(lg.float()[0], torch.from_numpy(y).to(DEV), pos_weight=torch.tensor(2.0, device=DEV)))
                gk += 1
            lroute = torch.stack(lr_).mean() if lr_ else torch.tensor(0.0, device=DEV)
            loss = lroute + a.lam * lb
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gb = torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            torch.nn.utils.clip_grad_norm_(rnet.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            for k, v in dict(loss=float(loss), route=float(lroute), builder=float(lb), grad_builder=float(gb) if torch.isfinite(gb) else run.get("grad_builder", 0.0), pos_rate=float(np.mean(pos)) if pos else 0.0, scale=float(scaler.get_scale())).items():
                run[k] = 0.95 * run.get(k, v) + 0.05 * v
            it += 1
            if it % 20 == 0 or it == 1:
                msg = f"it {it}/{a.iters} {time.time() - t0:.0f}s " + " ".join(f"{k} {v:.3f}" for k, v in run.items())
                print(msg, flush=True)
                log.write(msg + "\n")
                log.flush()
            if it >= a.iters:
                break
    os.makedirs(f"{C.OUT}/e123/{a.name}", exist_ok=True)
    torch.save(net.state_dict(), f"{C.OUT}/e123/{a.name}/model.pt")
    torch.save(rnet.state_dict(), f"{od}/model.pt")
    json.dump(dict(args=vars(a), final=run, seconds=time.time() - t0), open(f"{od}/train_summary.json", "w"), indent=1)
    print("DONE", f"{time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
