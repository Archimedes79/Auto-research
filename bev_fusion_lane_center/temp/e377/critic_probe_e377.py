"""critic probe (E377 code review, read-only): the HEAD versions of jtoken / jtoken_train (git show, in memory) against the working tree; loss + grads on synthetic and real
TRAIN frames, 20-step smoke runs bit for bit, the near label against an independent float64 numpy version on real frames, n_near per-frame spread."""
import json, os, subprocess, sys, tempfile, types
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch
torch.set_num_threads(2)
ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for sub_ in ("training", "training/grid", "training/data", "training/data/prep", "training/train", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub_))
import jtoken as WK
import jtoken_train as WKT


def head_mod(rel, name, deps=None):
    src = subprocess.check_output(["git", "show", "HEAD:bev_fusion_lane_center/" + rel], cwd=ROOT).decode()
    m = types.ModuleType(name)
    m.__file__ = os.path.join(ROOT, rel)
    saved = {k: sys.modules.get(k) for k in (deps or {})}
    sys.modules.update(deps or {})
    try:
        exec(compile(src, m.__file__, "exec"), m.__dict__)
    finally:
        for k, v in saved.items():
            sys.modules[k] = v
    return m


HK = head_mod("training/grid/jtoken.py", "jtoken_head")
HKT = head_mod("training/train/jtoken_train.py", "jtoken_train_head", {"jtoken": HK})
assert HKT.JK is HK and WKT.JK is WK and not hasattr(HK, "near_none") and hasattr(WK, "near_none")
print("HEAD modules built in memory; HEAD has no near_none, the working tree has it")

RECIPES = [dict(), dict(none_weight=0.25), dict(none_weight=0.25, pos_loss="max"), dict(pos_loss="max"), dict(none_weight=0.5), dict(none_weight=0.73)]


def grads(model, loss):
    return [g.clone() for g in torch.autograd.grad(loss, [p for p in model.parameters()], retain_graph=True, allow_unused=True) if g is not None]


def compare(model, out, b, tag):
    bad = []
    for r in RECIPES:
        lh, ih = HK.jtoken_loss(out, b, **r)
        gh = grads(model, lh)
        for v in (dict(r), dict(r, none_weight_near=r.get("none_weight", 1.0))):
            lw, iw = WK.jtoken_loss(out, b, **v)
            gw = grads(model, lw)
            same = torch.equal(lh, lw) and len(gh) == len(gw) and all(torch.equal(x, y) for x, y in zip(gh, gw)) and all(ih[k] == iw[k] for k in ih)
            if not same:
                bad.append((tag, v, float(lh), float(lw)))
    print(tag + ": loss, info and every gradient bit for bit HEAD vs WT for", len(RECIPES), "recipes x (unset, near = none weight):", "OK" if not bad else bad)
    return not bad


ok = True
for s in range(3):
    torch.manual_seed(s)
    m = WK.JTokenHead()
    b = WK.synthetic_batch(4, 96, seed=s)
    out = m(b)
    ok &= compare(m, out, b, "synthetic seed %d" % s)
for r in (dict(), dict(none_weight=0.25)):
    torch.manual_seed(0)
    m = WK.JTokenHead()
    b = WK.with_tokens_np(WK.synthetic_batch(2, 64, seed=7))
    lh, ih = HKT.make_loss(None, teacher="rule", **r)(m, b)
    torch.manual_seed(0)
    m2 = WK.JTokenHead()
    lw, iw = WKT.make_loss(None, teacher="rule", **r)(m2, b)
    diff = {k: (ih[k], iw.get(k)) for k in ih if ih[k] != iw.get(k)}
    print("teacher", r, "HEAD vs WT bit for bit:", torch.equal(lh, lw), diff)
    ok &= torch.equal(lh, lw)

with tempfile.TemporaryDirectory(dir=os.path.join(ROOT, "temp", "e377")) as runs:
    for name, extra, smk in (("champ", ["--none-weight", "0.25"], 48), ("max", ["--none-weight", "0.25", "--pos-loss", "max"], 48), ("max1", ["--pos-loss", "max"], 48),
                             ("dflt", [], 48), ("champ384", ["--none-weight", "0.25"], 384), ("teach", ["--none-weight", "0.25", "--teacher", "rule"], 48),
                             ("drop", ["--none-weight", "0.25", "--token-drop", "0.3"], 48)):
        res = {}
        for tag, mod in (("head", HKT), ("wt", WKT)):
            n = name + "_" + tag
            base = ["--name", n, "--seed", "0", "--smoke", "--bs", "2", "--workers", "0"]
            mod.train(mod.parse_args(base + extra), runs, 500, smk)
            ck = torch.load(os.path.join(runs, n, "model.pt"), map_location="cpu", weights_only=False)
            rows = [r for r in map(json.loads, open(os.path.join(runs, n, "log.jsonl"))) if "total" in r]
            res[tag] = (ck, rows)
        (ch, rh), (cw, rw) = res["head"], res["wt"]
        same_w = all(torch.equal(v, cw["state"][k]) for k, v in ch["state"].items())
        tot_h, tot_w = [r["total"] for r in rh], [r["total"] for r in rw]
        same_rows = tot_h == tot_w and all(r[k] == w[k] for r, w in zip(rh, rw) for k in ("nll_s", "nll_l", "ce1", "ce3"))
        extra_args = {k: v for k, v in cw["args"].items() if k not in ch["args"]}
        has_nn = any("n_near" in r for r in rw)
        print("smoke", name, "K", smk, ": final weights equal", same_w, ", logged totals and parts equal", same_rows, tot_w, ", new args keys", extra_args, ", n_near logged", has_nn)
        ok &= same_w and same_rows

import lane_ptok_data as LP
for seed in (0, 1):
    ds = LP.PTokData("train", seed, "e292_a")
    rng = np.random.RandomState(1)
    idx = rng.choice(len(ds), 120, replace=False)
    items = [ds[int(i)] for i in idx]
    bt = LP.collate(items)
    pos, dirv, valid, yt = bt["pos"].float(), bt["dirv"].float(), bt["valid"].bool(), bt["ytype"].long()
    near = WK.near_none(pos, dirv, valid, yt)
    nn_np = []
    for it in items:
        p, d, v, t = np.asarray(it["pos"], float), np.asarray(it["dirv"], float), np.asarray(it["valid"], bool), np.asarray(it["ytype"])
        asg = v & (t != 0)
        dist = np.linalg.norm(p[:, None] - p[None], axis=2)
        nn_np.append(v & (t == 0) & ((dist <= 2.0) & (d @ d.T >= 0.5) & asg[None, :]).any(1))
    nn_np = np.stack(nn_np)
    per = near.sum(1).numpy()
    dn = np.linalg.norm(np.asarray(bt["dirv"], float), axis=-1)[np.asarray(bt["valid"], bool)]
    ndiff = int((near.numpy() != nn_np).sum())
    print("TRAIN cache", seed, len(ds), "frames, pos dtype", bt["pos"].dtype, "dirv norm on valid tokens", dn.min(), dn.max(), "near torch float32 vs numpy float64 differing tokens", ndiff, "of", near.numel())
    print("  n_near per frame mean", per.mean(), "SD", per.std(), "min", per.min(), "max", per.max(), "(120 random frames); SD of an 8-frame batch mean", per.std() / np.sqrt(8), "of 24 frames", per.std() / np.sqrt(24))
    bb = {k: v[:2] for k, v in bt.items() if torch.is_tensor(v)}
    torch.manual_seed(0)
    m = WK.JTokenHead()
    out = m(bb)
    ok &= compare(m, out, bb, "real TRAIN cache %d, 2 frames" % seed)
    for w, wp in ((0.25, 0.73), (1.0, 0.73)):
        lw, iw = WK.jtoken_loss(out, bb, none_weight=w, none_weight_near=wp)
        K = out["cand"].shape[-1]
        yt2 = bb["ytype"].long()
        v2 = out["valid"]
        P, D = bb["pos"].double(), bb["dirv"].double()
        dist = (P[:, :, None] - P[:, None]).norm(dim=-1)
        cosm = (D[:, :, None] * D[:, None]).sum(-1)
        asg = v2 & (yt2 != 0)
        nr = v2 & (yt2 == 0) & ((dist <= 2.0) & (cosm >= 0.5) & asg[:, None, :]).any(-1)
        tokw = torch.where(nr, torch.tensor(wp, dtype=torch.float64), torch.tensor(w, dtype=torch.float64))
        lp = out["logp"].detach().double()
        G = (bb["y"] > 0)[:, None] & out["cand"]
        val = []
        for c in range(2):
            rows = out["cand"][:, c].any(-1) & v2
            fr = []
            for f in range(2):
                tot, n = 0.0, 0
                for i in torch.nonzero(rows[f]).flatten().tolist():
                    g = G[f, c, i]
                    tot += float(-torch.logsumexp(lp[f, c, i, :K][g], 0)) if g.any() else float(-tokw[f, i] * lp[f, c, i, K])
                    n += 1
                if n:
                    fr.append(tot / n)
            val.append(sum(fr) / len(fr))
        for t in (out["types1"], out["types"]):
            ce = torch.nn.functional.cross_entropy(t.detach().double().flatten(0, 1), yt2.flatten(), reduction="none").view_as(yt2)
            wt = torch.where(yt2 == 0, tokw, torch.ones_like(tokw))
            val.append(float(((ce * wt * v2).sum(1) / (wt * v2).sum(1)).mean()))
        got = [iw[k] for k in ("nll_s", "nll_l", "ce1", "ce3")]
        mx = max(abs(a - b) for a, b in zip(got, val))
        print("  independent float64 loss parts at none weight", w, "near", wp, "on 2 real frames: max abs diff", mx, "n_near", iw["n_near"], "independent", float(nr.sum(1).double().mean()), "parts", got)
        ok &= mx < 1e-4
print("ALL OK" if ok else "SOMETHING DIFFERS")
