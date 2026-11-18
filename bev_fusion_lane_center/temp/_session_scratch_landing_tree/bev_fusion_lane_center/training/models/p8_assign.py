"""Loop 8 / E83: graph assignment network (relation-biased transformer over the SEGMENTS of the perceived sparse lane graph + one scene token with the SD-map context).
Per segment it predicts  keep (is a true lane: precision control, 'better short than false')  and  route (is on the navigation route: lane-level assignment).
Training: samples of p8_dataset.py (--plan train, --tag _seg); selection on --plan dev; test on --plan test.
Usage (gcar): p8_assign.py --name e83_a --tag _seg [--epochs 25] [--seed 0] [--test 1] [--load e83_a]"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p8_lib as L  # noqa: E402
import p8_seg as S  # noqa: E402

DEV = "cuda"
NMAX = S.NSEG_MAX
NREL = 5


class Layer(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h, self.dk = h, d // h
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.rb = nn.Linear(NREL, h)  # relation (successor, predecessor, left, right, same piece) -> attention bias per head

    def forward(self, x, rel, mask):
        B, T, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.h, self.dk).unbind(2)
        att = torch.einsum("bthd,bshd->bhts", q, k) / self.dk ** 0.5
        att = att + self.rb(rel).permute(0, 3, 1, 2)
        att = att.masked_fill(~mask[:, None, None, :], -1e9)
        x = x + self.o(torch.einsum("bhts,bshd->bthd", att.softmax(-1), v).reshape(B, T, d))
        return x + self.ff(self.ln2(x))


class AssignNet(nn.Module):
    def __init__(self, dn, ds, d=96, h=4, nl=3, p_drop=0.0):
        super().__init__()
        self.emb = nn.Sequential(nn.Linear(dn, d), nn.GELU(), nn.Linear(d, d))
        self.semb = nn.Sequential(nn.Linear(ds, d), nn.GELU(), nn.Linear(d, d))
        self.layers = nn.ModuleList([Layer(d, h) for _ in range(nl)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 2))
        self.drop = nn.Dropout(p_drop)
        self.no_rel = False

    def forward(self, node, rel, scene, mask):
        B, N, _ = node.shape
        if self.no_rel:
            rel = torch.zeros_like(rel)
        x = torch.cat([self.semb(scene)[:, None], self.emb(node)], 1)
        m = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=node.device), mask], 1)
        r = F.pad(rel, (0, 0, 1, 0, 1, 0))
        x = self.drop(x)
        for ly in self.layers:
            x = ly(x, r, m)
        x = self.ln(x)
        s = x[:, :1].expand(-1, N, -1)
        return self.head(torch.cat([x[:, 1:], s], -1))  # [B,N,2]: keep, route


def to_tensors(samples, abstain=0):
    """pad all samples to NMAX segments -> dict of tensors on the CPU"""
    n = len(samples)
    dn = samples[0]["node"].shape[1]
    node = np.zeros((n, NMAX, dn), np.float16)
    rel = np.zeros((n, NMAX, NMAX, NREL), np.uint8)
    mask = np.zeros((n, NMAX), bool)
    tl, rl, ln = np.zeros((n, NMAX), np.float32), np.zeros((n, NMAX), np.float32), np.zeros((n, NMAX), np.float32)
    scene = np.zeros((n, samples[0]["scene"].shape[0]), np.float32)
    wr = np.zeros(n, np.float32)
    for i, s in enumerate(samples):
        m = len(s["tl"])
        node[i, :m] = s["node"]
        rel[i, :m, :m] = s["rel"]
        mask[i, :m] = True
        tl[i, :m], rl[i, :m], ln[i, :m] = s["tl"], s["rl"], s["length"]
        scene[i] = s["scene"]
        wr[i] = float(s["kind"] == "n" and s["has_route"])
        if abstain and s["kind"] in ("w", "0"):  # loop 9: a wrong or missing map must not produce a route: target 0 for every edge
            wr[i] = float(abstain)  # loss weight of the abstention samples (1 = as the others)
            rl[i, :m] = 0.0
    return dict(node=torch.from_numpy(node), rel=torch.from_numpy(rel), mask=torch.from_numpy(mask), tl=torch.from_numpy(tl), rl=torch.from_numpy(rl), ln=torch.from_numpy(ln),
                scene=torch.from_numpy(scene), wr=torch.from_numpy(wr))


def batch(T, idx):
    b = {k: v[idx] for k, v in T.items()}
    nm = max(int(b["mask"].sum(1).max()), 1)  # crop to the longest sample of the batch (valid segments are packed at the front)
    for k in ("node", "mask", "tl", "rl", "ln"):
        b[k] = b[k][:, :nm]
    b["rel"] = b["rel"][:, :nm, :nm]
    return {k: v.to(DEV) for k, v in b.items()}


def loss_fn(o, b, w_route=1.0, w_keep=1.0):
    w = torch.clamp(b["ln"] / 4.0, 0.2, 2.0) * b["mask"]
    lk = (F.binary_cross_entropy_with_logits(o[..., 0], (b["tl"] >= 0.6).float(), reduction="none") * w).sum() / w.sum()
    wr = w * b["wr"][:, None]
    lr = (F.binary_cross_entropy_with_logits(o[..., 1], (b["rl"] >= 0.5).float(), reduction="none") * wr).sum() / wr.sum().clamp(min=1.0)
    return w_keep * lk + w_route * lr, lk.item(), lr.item()


def seg_points(s, sel, ds=None):
    """points of the selected segments: the stored 1 m vertices (ds None) or resampled at ds without the end vertex"""
    out = []
    for j in np.nonzero(sel)[0]:
        q = s["spts"][j, : s["snp"][j]].astype(np.float64)
        if ds is None:
            out.append(q)
        else:
            seg = np.linalg.norm(np.diff(q, axis=0), axis=1)
            t = np.arange(0.0, seg.sum() - 1e-6, ds)
            if len(t) == 0:
                t = np.array([0.0])
            cs = np.concatenate([[0.0], np.cumsum(seg)])
            out.append(np.stack([np.interp(t, cs, q[:, 0]), np.interp(t, cs, q[:, 1])], 1))
    return np.concatenate(out) if out else np.zeros((0, 2))


def decode_eval(samples, P, srcs, tk=0.5, tr=0.5, oracle=False):
    """P: [n,NMAX,2] probabilities. Per cell: lane precision/recall of the kept segments (0.5 m points, tolerance 1 m), route success of the route segments (E59/E77 definition)."""
    res = {}
    for i, s in enumerate(samples):
        src = srcs[s["src"]]
        part, row = src["part"], s["row"]
        m = len(s["tl"])
        if oracle:
            keep = s["tl"] >= 0.6
            rt = (s["rl"] >= 0.5) & keep
        else:
            keep = P[i, :m, 0] >= tk
            rt = (P[i, :m, 1] >= tr) & keep
        gl = L.gt_lane_points(part, row)
        key = (s["src"].split("_", 1)[1], s["cell"])
        c = res.setdefault(key, dict(n=0, nr=0, succ=0, cov=0.0, prec=0.0, tp_p=0, n_p=0, tp_g=0, n_g=0, false_len=0.0, kept_len=0.0, pieces=0))
        pr = L.lane_pr(seg_points(s, keep, 0.5), gl)
        for k_ in ("tp_p", "n_p", "tp_g", "n_g"):
            c[k_] += pr[k_]
        c["n"] += 1
        c["kept_len"] += float(s["length"][keep].sum())
        c["false_len"] += float(s["length"][keep & (s["tl"] < 0.5)].sum())
        c["pieces"] += int(keep.sum())
        if s["kind"] == "n" and s["has_route"]:
            r = L.route_success(seg_points(s, rt), L.cache("rpts", part)[int(row)])
            if r is not None:
                c["nr"] += 1
                c["succ"] += r["succ"]
                c["cov"] += r["cov"]
                c["prec"] += r["prec"]
    out = {}
    for key, c in res.items():
        nr = max(c["nr"], 1)
        p_, r_ = c["tp_p"] / max(c["n_p"], 1), c["tp_g"] / max(c["n_g"], 1)
        out["/".join(key)] = dict(n=c["n"], n_route=c["nr"], route_success=c["succ"] / nr, route_cov=c["cov"] / nr, route_prec=c["prec"] / nr, lane_P=p_, lane_R=r_, lane_F1=2 * p_ * r_ / max(p_ + r_, 1e-9),
                                  false_len_per_row=c["false_len"] / c["n"], kept_len_per_row=c["kept_len"] / c["n"], segments_per_row=c["pieces"] / c["n"])
    return out


def load_samples(plan, loc, tag=""):
    return pickle.load(open(f"{L.D}/p8/ds_{plan}_{loc}{tag}.pkl", "rb"))


def srcs_for(samples):
    names = sorted({s["src"] for s in samples})
    return {n: L.load_src(n) for n in names}


def dev_score(res):
    r = [v["route_success"] for k, v in res.items() if v["n_route"] > 0]
    l = [v["lane_F1"] for k, v in res.items()]
    return float(np.mean(r)), float(np.mean(l))


def predict_norm(net, T, mu, sd_, no_sd=0, bs=128):
    net.eval()
    out = np.zeros((len(T["node"]), NMAX, 2), np.float32)
    with torch.no_grad():
        for i in range(0, len(T["node"]), bs):
            idx = torch.arange(i, min(i + bs, len(T["node"])))
            b = batch(T, idx)
            node = (b["node"].float() - mu.to(DEV)) / sd_.to(DEV) * b["mask"][..., None]
            scene = b["scene"]
            if no_sd:
                node[..., S.SEG_GEO :] = 0
                scene = torch.zeros_like(scene)
            o = torch.sigmoid(net(node, b["rel"].float(), scene, b["mask"])).cpu().numpy()
            out[i : i + len(idx), : o.shape[1]] = o
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--loc", default="tmpl")
    ap.add_argument("--tag", default="_seg")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--d", type=int, default=96)
    ap.add_argument("--nl", type=int, default=3)
    ap.add_argument("--drop", type=float, default=0.0)
    ap.add_argument("--w_route", type=float, default=1.0)
    ap.add_argument("--w_keep", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test", type=int, default=0)
    ap.add_argument("--load", default="")
    ap.add_argument("--train_plan", default="train")
    ap.add_argument("--fnoise", type=float, default=0.0, help="Gaussian noise (std, in units of the normalised features) added to the node features in training")
    ap.add_argument("--tokdrop", type=float, default=0.0, help="share of segments dropped at random in training")
    ap.add_argument("--no_rel", type=int, default=0, help="ablation: no segment-segment relations (successor / predecessor / parallel / same piece) in the attention")
    ap.add_argument("--clean_only", type=int, default=0, help="train on the out-of-fold clean rows only (no perturbed train rows)")
    ap.add_argument("--no_sd", type=int, default=0, help="ablation: zero all SD-relative node features and the scene features")
    ap.add_argument("--abstain", type=float, default=0, help="loop 9: route target 0 (weight = this value) for wrong-map and no-map samples")
    ap.add_argument("--no_bias", type=int, default=0, help="loop 9: drop the training samples with a per-road lateral bias")
    ap.add_argument("--devplan", default="dev", help="DEV plan (dev or devq)")
    ap.add_argument("--dev_tag", default="", help="tag of the DEV samples (default: --tag)")
    a = ap.parse_args()
    out = f"{L.OUT}/{a.name}"
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    t0 = time.time()
    dev_s = load_samples(a.devplan, a.loc, a.dev_tag or a.tag)
    dev_src = srcs_for(dev_s)
    Tdev = to_tensors(dev_s)
    if a.load:
        ck = torch.load(f"{L.OUT}/{a.load}/model.pt", map_location=DEV, weights_only=False)
        net = AssignNet(ck["dn"], ck["ds"], ck["d"], 4, ck["nl"]).to(DEV)
        net.load_state_dict(ck["state"])
        net.no_rel = bool(ck["args"].get("no_rel", 0))
        a.no_sd = int(ck["args"].get("no_sd", 0))
        mu, sd_ = ck["mu"], ck["sd"]
    else:
        tr = load_samples(a.train_plan, a.loc, a.tag)
        if a.clean_only:
            tr = [x for x in tr if x["src"] == "train_oof"]
        if a.no_bias:
            tr = [x for x in tr if not x.get("sd_kw", {}).get("bias")]
        Ttr = to_tensors(tr, a.abstain)
        dn, ds = Ttr["node"].shape[2], Ttr["scene"].shape[1]
        nodes = Ttr["node"].float()[Ttr["mask"]]
        mu, sd_ = nodes.mean(0), nodes.std(0) + 1e-3
        net = AssignNet(dn, ds, a.d, 4, a.nl, a.drop).to(DEV)
        net.no_rel = bool(a.no_rel)
        opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
        steps = a.epochs * (len(tr) // a.bs)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
        print(f"train samples {len(tr)} dev {len(dev_s)} params {sum(p.numel() for p in net.parameters())}", flush=True)
        for ep in range(a.epochs):
            net.train()
            perm = torch.randperm(len(tr))
            tl_ = []
            for i in range(0, len(tr) - a.bs + 1, a.bs):
                b = batch(Ttr, perm[i : i + a.bs])
                if a.tokdrop > 0:
                    b["mask"] = b["mask"] & (torch.rand_like(b["ln"]) > a.tokdrop)
                node = (b["node"].float() - mu.to(DEV)) / sd_.to(DEV) * b["mask"][..., None]
                if a.fnoise > 0:
                    node = node + a.fnoise * torch.randn_like(node) * b["mask"][..., None]
                scene = b["scene"]
                if a.no_sd:
                    node[..., S.SEG_GEO :] = 0
                    scene = torch.zeros_like(scene)
                o = net(node, b["rel"].float(), scene, b["mask"])
                loss, lk, lr_ = loss_fn(o, b, a.w_route, a.w_keep)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                sch.step()
                tl_.append((lk, lr_))
            if ep % 5 == 4 or ep == a.epochs - 1:
                Pd = predict_norm(net, Tdev, mu, sd_, a.no_sd)
                rs = decode_eval(dev_s, Pd, dev_src)
                rsu, lf = dev_score(rs)
                print(f"ep {ep + 1} loss keep {np.mean([x[0] for x in tl_]):.4f} route {np.mean([x[1] for x in tl_]):.4f} | DEV route success {rsu:.4f} lane F1 {lf:.4f} ({time.time() - t0:.0f} s)", flush=True)
        torch.save(dict(state=net.state_dict(), dn=dn, ds=ds, d=a.d, nl=a.nl, mu=mu, sd=sd_, args=vars(a)), f"{out}/model.pt")
    Pd = predict_norm(net, Tdev, mu, sd_, a.no_sd)
    rs = decode_eval(dev_s, Pd, dev_src)
    rsu, lf = dev_score(rs)
    res = dict(args=vars(a), dev=rs, dev_route=rsu, dev_lane_F1=lf)
    np.save(f"{out}/dev_pred.npy", Pd.astype(np.float16))
    print(f"DEV route success {rsu:.4f} lane F1 {lf:.4f}")
    if a.test:
        te = load_samples("test", a.loc, a.tag)
        te_src = srcs_for(te)
        Pt = predict_norm(net, to_tensors(te), mu, sd_, a.no_sd)
        res["test"] = decode_eval(te, Pt, te_src)
        res["test_oracle"] = decode_eval(te, Pt, te_src, oracle=True)
        np.save(f"{out}/test_pred.npy", Pt.astype(np.float16))
        for k, v in sorted(res["test"].items()):
            print(f"TEST {k:22s} n {v['n']:4d} route {v['route_success']:.3f} (oracle {res['test_oracle'][k]['route_success']:.3f}) lane P {v['lane_P']:.3f} R {v['lane_R']:.3f} F1 {v['lane_F1']:.3f}")
    json.dump(res, open(f"{out}/result.json", "w"), indent=1)


if __name__ == "__main__":
    main()
