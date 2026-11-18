"""Loop 16: token files of a builder run for the navigation head and the planner consumer (same structure as p10_tokens.py / p15_make_tokens.py output):
items = dict(row, k, xy [n,2], tok [n,128] fp16, has [n], cp [n,7,2], pieces [vertex paths], edges [vertex edges], keep [n]).
A vertex inherits the 96-d token of the node it was built from (zero-padded to 128), the cross-section points are 7 offsets across the local lane direction.
Usage (gcar, GPU): p16_tokens.py --design D3 --name e123_D3_r3_00 --out tok_p16_xxx --split val_clean|test_clean|train [--tau 0.5] [--n 0] [--set k=v ...]
Writes ~/exp_gcar_b123/p10/<out>_<split>.pkl (train = every frame of the training split, in-sample for the builder: the caveat of E120 / E121)."""
import argparse
import os
import pickle
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p16_common as C  # noqa: E402
import p16_data as PD  # noqa: E402
import p16_infer as PI  # noqa: E402
import p16_train as PT  # noqa: E402

OFFS = np.array([-2.5, -1.5, -0.75, 0.0, 0.75, 1.5, 2.5], np.float32)


def cross_section(V, edges):
    """7 points across the lane at every vertex: the normal of the mean direction of the incident edges (a vertex without an edge: x axis)"""
    n = len(V)
    t = np.zeros((n, 2))
    for a, b in edges:
        d = V[b] - V[a]
        d = d / max(np.linalg.norm(d), 1e-6)
        t[a] += d
        t[b] += d
    nn = np.linalg.norm(t, axis=1, keepdims=True)
    t = np.where(nn > 1e-6, t / np.maximum(nn, 1e-6), np.array([[1.0, 0.0]]))
    nrm = np.stack([-t[:, 1], t[:, 0]], 1)
    return (V[:, None, :] + OFFS[None, :, None] * nrm[:, None, :]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--asm", nargs="*", default=[], help="assembly keywords, e.g. cov_r=0.5 vtx_r=1.5 junc_thr=0.5")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    import json

    c = json.load(open(f"{C.LADDER}/cards/{a.name}.json", encoding="utf-8-sig"))
    tau = a.tau if a.tau is not None else c["metrics"]["tau"]
    extra = dict(c["cfg"])  # the run's own configuration (flips and switches included)
    extra.update(PT.parse_set(a.set))
    net, cfg = PT.build(a.design, extra)
    asm = {k: float(v) for k, v in (x.split("=") for x in a.asm)}
    net = net.cuda()
    net.load_state_dict(torch.load(f"{C.OUT}/e123/{a.name}/model.pt", map_location="cuda"), strict=False)
    net.eval()
    ds = PD.BuilderData(a.split, train=False, with_prev=cfg.temporal)
    if a.n:
        ds.ks = np.sort(np.random.RandomState(0).choice(len(ds), min(a.n, len(ds)), replace=False))
    dl = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False, num_workers=a.workers, collate_fn=PD.collate)
    items, t0 = [], time.time()
    with torch.no_grad():
        for b in dl:
            bb = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in b.items()}
            prior = None
            if cfg.temporal:
                import p16_time as PT_

                prior = PT_.prior_from_prev(net, bb, train=False)
            with torch.autocast("cuda", dtype=torch.float16):
                res = net(bb["x"].float(), prior=prior)
            pair = None
            if cfg.dedup == "learned":
                import p16_loss as PL

                _, pr = PL.dup_learned(net, res, bb)
                pair = dict(ki=pr["ki"].cpu().numpy(), p=torch.sigmoid(pr["logit"]).float().cpu().numpy())
            for i in range(bb["x"].shape[0]):
                r = PI.to_numpy(res, i)
                g, ax = PI.assemble(r, None, tau, cfg, aux=True, pair=None if pair is None else dict(ki=pair["ki"][i], p=pair["p"][i]), **asm)
                V = np.asarray(ax["V"], np.float32).reshape(-1, 2)
                n = len(V)
                h = res["h"][i].float().cpu().numpy()
                tok = np.zeros((n, 128), np.float16)
                has = np.zeros(n, bool)
                for v, s in enumerate(ax["src"]):
                    if s >= 0:
                        tok[v, : h.shape[1]] = h[s]
                        has[v] = True
                items.append(dict(row=int(b["row"][i]), k=int(b["k"][i]), xy=V, tok=tok, has=has, cp=cross_section(V, ax["edges"]), pieces=[list(map(int, p)) for p in g["nodes"]], edges=sorted((int(x), int(y)) for x, y in ax["edges"]), keep=np.asarray([float(r["pi"][s_]) if s_ >= 0 else 0.5 for s_ in ax["src"]], np.float32)))
    d = f"{os.path.expanduser('~')}/exp_gcar_b123/p10"
    os.makedirs(d, exist_ok=True)
    fn = f"{d}/{a.out}_{a.split}.pkl"
    pickle.dump(dict(part=ds.part, items=items, stats=dict(tau=tau, design=a.design, name=a.name, asm=asm)), open(fn, "wb"))
    print(f"written {fn}: {len(items)} frames, {np.mean([len(it['xy']) for it in items]):.0f} vertices per frame, tau {tau}, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
