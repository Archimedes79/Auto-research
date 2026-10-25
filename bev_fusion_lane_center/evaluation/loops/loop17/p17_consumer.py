"""Loop 17 / E212 (internal E144): the planner-side consumer under sensor failure. The consumer of loops 14 / 16 (ConsB, mode route+ego: the edges of the lane graph with the route probability of the route head and the ego motion -> the driven path at 5 ... 40 m)
is trained as in p14_chain_consumer.py on the training chain data of the champion (out of ~/exp_gcar_b123/p14/p16chain_f2hr_train.pt, 3 seeds) and read on the graphs of the failure-half frames of a failure series (builder -> assembly -> route input -> route head -> consumer),
error of the predicted path point at 20 m against the driven path (frames whose driven path reaches 20 m). Series: p17_series.py files; builder: the champion (default) or a map-conditioned builder (--mapnet).
Usage (gcar, GPU): p17_consumer.py --out e144_champ --series dark_single,dark_maxdecay_0.95,... [--mapnet 1 --ckpt e17_map_b --map_mode exact] [--cells exact,none]"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p10_net as PN  # noqa: E402
import p14_chain_consumer as CC  # noqa: E402
import p16_chain_data as CF  # noqa: E402
import p16_common as C  # noqa: E402
import p16_data as PD  # noqa: E402
import p16_infer as PI  # noqa: E402
import p16_time as PTM  # noqa: E402
import p17_eval as EV  # noqa: E402
import p17_joint as PJ  # noqa: E402
import p8_lib as L  # noqa: E402

DEV = "cuda"
E17 = EV.E17
HOME = os.path.expanduser("~")
MODE = "route+ego"


def consumers(seeds, epochs=25):
    """the consumers of the champion's training chain data, trained once and kept"""
    d = f"{E17}/consumer/models"
    os.makedirs(d, exist_ok=True)
    nets = []
    ztr = None
    for s in range(seeds):
        f = f"{d}/{MODE}_s{s}.pt"
        net = CC.ConsB(CC.nfeat(MODE), False).to(DEV)
        if os.path.exists(f):
            net.load_state_dict(torch.load(f, map_location=DEV))
        else:
            if ztr is None:
                ztr = torch.load(f"{HOME}/exp_gcar_b123/p14/p16chain_f2hr_train.pt", weights_only=False)
                st_tr, p_tr, k_tr, _ = CC.to_dev(ztr)
            net = CC.train(st_tr, p_tr, k_tr, MODE, epochs, s)
            torch.save(net.state_dict(), f)
        nets.append(net.eval())
    return nets


@torch.no_grad()
def items_of(net, cfg, ds, idx, asm, tau):
    items = []
    for i in idx:
        b = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in PD.collate([ds[i]]).items()}
        row = int(ds.rows[int(b["k"][0])])
        prior = PTM.prior_from_prev(net, b, train=False) if cfg.temporal else None
        with torch.autocast("cuda", dtype=torch.float16):
            res = net(b["x"].float(), gt=None, share=0.0, prior=prior)
        r = PI.to_numpy(res, 0)
        g, ax = PI.assemble(r, None, tau, cfg, aux=True, **asm)
        if not len(ax["V"]):
            items.append(dict(row=row, k=len(items), xy=np.zeros((1, 2), np.float32), tok=np.zeros((1, 128), np.float16), has=np.zeros(1, bool), cp=np.zeros((1, 7, 2), np.float32), pieces=[], edges=[], keep=np.zeros(1, np.float32)))
            continue
        it0, it1, M, src, has = PJ.route_item(r, ax, g)
        h = res["h"][0].float().cpu().numpy()
        tok_old = np.zeros((len(it0["xy"]), 128), np.float32)
        if has.any():
            hh = h[src[has]]
            tok_old[np.where(has)[0], : hh.shape[1]] = hh
        items.append(dict(row=row, k=len(items), xy=it1["xy"], tok=(M @ tok_old).astype(np.float16), has=it1["has"], cp=it1["cp"], pieces=it1["pieces"], edges=it1["edges"], keep=it1["keep"]))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", default="D7")
    ap.add_argument("--name", default="e123_D7_r3_f1")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--set", nargs="*", default=["M=288", "K0=144"])
    ap.add_argument("--asm", nargs="*", default=[f"{k}={v}" for k, v in EV.CHAMP_ASM.items()])
    ap.add_argument("--tau", type=float, default=0.7)
    ap.add_argument("--series", required=True)
    ap.add_argument("--route", default="p16_f2hr_route_s0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prev", type=int, default=0)
    ap.add_argument("--mapnet", type=int, default=0)
    ap.add_argument("--map_mode", default="exact")
    ap.add_argument("--map_sigma", type=float, default=0.0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--phase", type=int, default=1, help="1: failure half, 0: clean half")
    ap.add_argument("--tag_suffix", default="")
    a = ap.parse_args()
    asm = {k: float(v) for k, v in (x.split("=") for x in a.asm)}
    net, cfg = EV.load_net(a.design, a.name, a.set)
    if a.mapnet:
        import p17_mapnet as MN

        MN.widen(net)
        a.prev = 0
    if a.ckpt:
        net.load_state_dict(torch.load(a.ckpt if a.ckpt.endswith(".pt") else f"{C.OUT}/e123/{a.ckpt}/model.pt", map_location="cuda"), strict=False)
    if not a.prev:
        cfg.temporal = False
        net.cfg.temporal = False
    net.eval()
    rnet = PN.RouteXNet().to(DEV)
    rnet.load_state_dict(torch.load(f"{L.OUT}/{a.route}/model.pt", map_location=DEV))
    rnet.eval()
    cons = consumers(a.seeds)
    os.makedirs(f"{E17}/{a.out}", exist_ok=True)
    res = {}
    t0 = time.time()
    for tag in a.series.split(","):
        ds = EV.SeriesData(tag, with_prev=bool(cfg.temporal))
        if a.mapnet:
            ds = MN.MapWrap(ds, "val", ds.P17, train=False, mode=a.map_mode, sigma=a.map_sigma)
        idx = [i for i in range(len(ds)) if int(ds.meta["phase"][i]) == a.phase]
        items = items_of(net, cfg, ds, idx, asm, a.tau)
        static = CF.frame_static(items, "val", DEV)
        samples, meta = [], []
        for k in range(len(items)):
            if not static["v"][k].any():
                continue
            for cname, (kind, sigma) in CF.CELLS_TEST.items():
                if cname in ("exact", "none"):
                    samples.append((k, kind, sigma, 0.0, 7 + k))
                    meta.append(cname)
        dsc = CF.CellData(items, "val", samples)
        p = CF.route_edge_probs(rnet, dsc, static, DEV, workers=2)
        k_t = torch.tensor([s[0] for s in samples], device=DEV)
        static = {k_: (v_.to(DEV) if torch.is_tensor(v_) else v_) for k_, v_ in static.items()}  # loop 18: the restored code left the static tensors on the cpu (indexing with cuda indices failed)
        p = p.to(DEV) if torch.is_tensor(p) else p
        errs = {}
        for si, cn in enumerate(cons):
            o = CC.predict(cn, static, p, k_t, MODE)
            r = CC.evaluate(o, static, k_t, meta)
            for c, v in r.items():
                errs.setdefault(c, []).append((v["err20"], v["n"]))
        res[tag + a.tag_suffix] = {c: dict(err20=float(np.mean([e for e, _ in v])), err20_seeds=[float(e) for e, _ in v], n=v[0][1]) for c, v in errs.items()}
        print(f"[{tag}{a.tag_suffix}] " + " ".join(f"{c}: {r['err20']:.2f} m (n {r['n']}, seeds {', '.join(f'{x:.2f}' for x in r['err20_seeds'])})" for c, r in res[tag + a.tag_suffix].items()) + f"  {time.time() - t0:.0f} s", flush=True)
        json.dump(res, open(f"{E17}/{a.out}/consumer.json", "w"), indent=1)
    print("DONE")


if __name__ == "__main__":
    main()
