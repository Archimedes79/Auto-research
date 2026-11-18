"""Loop 14 / E117: a learned planner-side consumer of the LEARNED chain (system B: trace network E94a -> route head E104b), trained and evaluated on the data of p14_chain_data.py.
Tokens = the 6 m edges of the traced graph (start / middle / end point). Optional inputs per token: route probability of the route head (route), probabilities of the piece verifier and junction verifier as soft attributes (soft), the pieces below 0.6 removed from the
token list (hard), the mean step token of the edge ends (tok, 128 numbers through a linear layer), the ego motion of the frame (ego). Consumer as E115: input MLP, 2-layer transformer encoder (d 64), 8 learned path queries (5 ... 40 m of the driven path) cross-attending
to the tokens, loss = Euclid error of the predicted path points, 3 seeds. Error at 20 m on the TEST frames with a driven path of >= 20 m (546 frames per cell), cells exact / s2 / wrong / none.
Usage (gcar): p14_chain_consumer.py --data chain_a --out e117_a"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p8_lib as L  # noqa: E402

HOME = os.path.expanduser("~")
DEV = "cuda"
NQ = 8
HARD = 0.6

MODES = {
    "geo": dict(),
    "route": dict(route=1),
    "route+ego": dict(route=1, ego=1),
    "route+soft": dict(route=1, soft=1),
    "route+soft+ego": dict(route=1, soft=1, ego=1),
    "route+hard+ego": dict(route=1, hard=1, ego=1),
    "route+tok+ego": dict(route=1, tok=1, ego=1),
    "route+soft+tok+ego": dict(route=1, soft=1, tok=1, ego=1),
    "route+tokreg+ego": dict(route=1, tok=1, ego=1, tokreg=1),  # E117b: step token with dropout 0.5 and Gaussian noise (sd = 0.5 x the std of the token feature) during training
}


def to_dev(z):
    st = {k: v.to(DEV) for k, v in z["static"].items()}
    return st, z["p"].to(DEV), z["k"].to(DEV), z["cell"]


def feats(st, k, p, mode):
    """-> x [B,NT,F], tokens [B,NT,128] or None, mask [B,NT]"""
    f = MODES[mode]
    geo = st["geo"][k]
    mask = st["mask"][k].clone()
    cols = [geo]
    if f.get("route"):
        cols.append(p.float()[..., None])
    if f.get("soft"):
        cols += [st["pp"][k][..., None], st["jp"][k][..., None]]
    if f.get("hard"):
        mask &= st["pp"][k] >= HARD
    if f.get("ego"):
        cols.append(st["ego"][k][:, None, :].expand(-1, geo.shape[1], -1))
    x = torch.cat(cols, -1) * mask[..., None]
    t = st["tok"][k].float() if f.get("tok") else None
    return x, t, mask


def nfeat(mode):
    f = MODES[mode]
    return 6 + f.get("route", 0) + 2 * f.get("soft", 0) + 4 * f.get("ego", 0)


class ConsB(nn.Module):
    def __init__(self, nf, tok=False, d=64, nl=2, nq=NQ):
        super().__init__()
        self.emb = nn.Sequential(nn.Linear(nf, d), nn.GELU(), nn.Linear(d, d))
        self.tp = nn.Sequential(nn.Linear(128, d), nn.GELU()) if tok else None
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 2 * d, 0.0, batch_first=True, norm_first=True), nl)
        self.q = nn.Parameter(torch.randn(nq, d) * 0.1)
        self.att = nn.MultiheadAttention(d, 4, batch_first=True)
        self.out = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2))

    def forward(self, x, t, mask):
        mask = mask.clone()
        mask[:, 0] |= ~mask.any(1)
        h = self.emb(x)
        if self.tp is not None:
            h = h + self.tp(t)
        h = self.enc(h, src_key_padding_mask=~mask)
        q = self.q[None].expand(len(x), -1, -1)
        o, _ = self.att(q, h, h, key_padding_mask=~mask)
        return self.out(o) * 50.0


def train(st, p, k, mode, epochs, seed, lr=2e-3, bs=256, d=64, nl=2):
    torch.manual_seed(seed)
    net = ConsB(nfeat(mode), bool(MODES[mode].get("tok")), d=d, nl=nl).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
    n = len(k)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * (n // bs), pct_start=0.1)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n - bs + 1, bs):
            ix = perm[i : i + bs]
            kk = k[ix]
            x, t, m = feats(st, kk, p[ix], mode)
            if t is not None and MODES[mode].get("tokreg"):
                t = nn.functional.dropout(t, 0.5) + 0.5 * t.std(dim=(0, 1), keepdim=True) * torch.randn_like(t)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = net(x, t, m).float()
            err = (o - st["y"][kk]).norm(dim=-1)
            vv = st["v"][kk].float()
            loss = (err * vv).sum() / vv.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sch.step()
    return net


@torch.no_grad()
def predict(net, st, p, k, mode, bs=512):
    net.eval()
    out = []
    for i in range(0, len(k), bs):
        kk = k[i : i + bs]
        x, t, m = feats(st, kk, p[i : i + bs], mode)
        out.append(net(x, t, m).cpu())
    return torch.cat(out)


def evaluate(o, st, k, cell, groups=None):
    kc = k.cpu()
    err = (o - st["y"].cpu()[kc]).norm(dim=-1).numpy()
    v = st["v"].cpu()[kc].numpy()
    cu = st["curved"].cpu()[kc].numpy()
    cell = np.array(cell)
    res = {}
    for c in sorted(set(cell)):
        sel = (cell == c) & v[:, 3]
        res[c] = dict(err20=float(err[sel, 3].mean()), n=int(sel.sum()), err20_curved=float(err[sel & cu, 3].mean()) if (sel & cu).any() else None, n_curved=int((sel & cu).sum()), err20_straight=float(err[sel & ~cu, 3].mean()), err_all=float(err[sel][v[sel]].mean()))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="chain_a")
    ap.add_argument("--out", default="e117_a")
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--eval_split", default="test", help="loop 16b: 'val' evaluates on the validation frames (the test frames stay untouched)")
    ap.add_argument("--d", type=int, default=64, help="E117c capacity control: width of the consumer")
    ap.add_argument("--nl", type=int, default=2, help="E117c capacity control: number of transformer layers")
    a = ap.parse_args()
    d = f"{HOME}/exp_gcar_b123/p14"
    os.makedirs(f"{L.OUT}/{a.out}", exist_ok=True)
    ztr = torch.load(f"{d}/{a.data}_train.pt", weights_only=False)
    zte = torch.load(f"{d}/{a.data}_{a.eval_split}.pt", weights_only=False)
    st_tr, p_tr, k_tr, c_tr = to_dev(ztr)
    st_te, p_te, k_te, c_te = to_dev(zte)
    print(f"train samples {len(k_tr)} test samples {len(k_te)}; train cells {dict(zip(*np.unique(c_tr, return_counts=True)))}", flush=True)
    res = {}
    t0 = time.time()
    for mode in a.modes.split(","):
        for seed in range(a.seeds):
            net = train(st_tr, p_tr, k_tr, mode, a.epochs, seed, d=a.d, nl=a.nl)
            os.makedirs(f"{L.OUT}/{a.out}/models", exist_ok=True)
            torch.save(net.state_dict(), f"{L.OUT}/{a.out}/models/{mode}_s{seed}.pt")
            o = predict(net, st_te, p_te, k_te, mode)
            r = evaluate(o, st_te, k_te, c_te)
            res[f"{mode}/seed{seed}"] = r
            print(f"{mode:20s} seed {seed}: " + " ".join(f"{c} {r[c]['err20']:.2f}" for c in ("exact", "s2", "wrong", "none")) + f"  curved(exact) {r['exact']['err20_curved']:.2f}  ({time.time() - t0:.0f} s)", flush=True)
            json.dump(res, open(f"{L.OUT}/{a.out}/result.json", "w"), indent=1)
    res["_args"] = vars(a)
    json.dump(res, open(f"{L.OUT}/{a.out}/result.json", "w"), indent=1)
    print("DONE")


if __name__ == "__main__":
    main()
