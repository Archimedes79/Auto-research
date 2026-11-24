"""E390 step 1 (card experiments/E390_sd-map-logical-road-graph): the probes of the SD road-graph encoding: the rule nulls and the Bayes ceiling (no training), the training of the arms g / n, the VAL read.

    wsl bash tools/env/_py14.sh sdgraph_probe.py nulls                       # VAL at every probe level: P1 copy / neighbour vote / Bayes ceiling (fitted on the TRAIN exact graphs), P2 the link rule, P3 the route rule
                                                                             #   -> runs/e390_sdgraph/nulls_val.json (never overwritten)
    wsl bash tools/env/_py14.sh sdgraph_probe.py train --arm g|n --seed S    # the encoder + probe heads on the TRAIN draws -> runs/e390_<arm>_s<S>/model.pt, train.json
    wsl bash tools/env/_py14.sh sdgraph_probe.py read  --arm g|n --seed S    # VAL at every level + the invariance test (P4) -> runs/e390_<arm>_s<S>/probe_card.json (paired with the nulls on the same draws)
    wsl bash tools/env/_py14.sh sdgraph_probe.py selftest                    # metrics by hand, the bootstrap, a tiny training run on the toy graph

VAL draws: the error of frame k at level L comes from a generator seeded by (390, k, the level's index): the same draws for every arm and the nulls (paired). Metrics (pooled over the frames):
  P1  accuracy of the lane class over the ROAD nodes of exact length >= 10 m
  P2  AUC (and AP) of the link score over the candidate pairs (input gap <= 10 m, not an input link): positives = the DROPPED true links, negatives = the non-links; the deciding set leaves out the pairs whose two nodes
      both carry an input route flag (route order would give their link away), the full set is reported
  P3  AUC of the on-route score over `p3_set` (nodes without an input route flag within 30 m of a flagged one): positives = the hidden route nodes
Statistics: a scene bootstrap (B 1000, seed 0) of each metric and of g minus a null / n on the same draws (95 % interval).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for sub_ in ("training/data", "training/grid", "evaluation", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub_)))
import jsdgraph as JS  # noqa: E402
import lane_sdgraph_data as SG  # noqa: E402

LEVELS = {                                                                    # the VAL levels of the card (section 1.3)
    "base": dict(count_rate=0.3, drop=0.3, hide=0.3),
    "ce0": dict(count_rate=0.0, drop=0.3, hide=0.3),
    "ce01": dict(count_rate=0.1, drop=0.3, hide=0.3),
    "s1": dict(count_rate=0.3, drop=0.3, hide=0.3, sigma=1.0),
    "s2": dict(count_rate=0.3, drop=0.3, hide=0.3, sigma=2.0),
    "s5": dict(count_rate=0.3, drop=0.3, hide=0.3, sigma=5.0),
    "d01": dict(count_rate=0.3, drop=0.1, hide=0.3),
}
MIN_LEN = 10.0
B_BOOT = 1000
CFG = dict(d=64, heads=4, layers=2, lr=1e-3, epochs=20, bs=32, draws=2, seed_draw=7)


def runs_dir(arm, seed):
    return os.path.join(ROOT, "runs", f"e390_{arm}_s{seed}")


# ------------------------------------------------------------------------------------------------------------------------------ the draws
def train_draw(G, rng):
    """one TRAIN error draw (card 1.2): count error {0, 0.1, 0.3}, unknown p 0.1 half the time, link drop {0, 0.1, 0.3}, route hide {0, 0.3}, smooth deformation {0, 1, 2} m, Douglas-Peucker U(0, 1.5) m, a rigid
    transform (+-10 m, +-30 deg)"""
    return SG.perturb(G, rng, count_rate=rng.choice([0.0, 0.1, 0.3]), unknown_p=0.1 if rng.random() < 0.5 else 0.0, drop=rng.choice([0.0, 0.1, 0.3]), hide=rng.choice([0.0, 0.3]),
                      sigma=rng.choice([0.0, 1.0, 2.0]), eps=rng.uniform(0, 1.5), rigid=(rng.uniform(-10, 10), rng.uniform(-10, 10), np.deg2rad(rng.uniform(-30, 30))))


def val_draw(G, k, level, rigid=None):
    li = list(LEVELS).index(level)
    return SG.perturb(G, np.random.default_rng([390, k, li]), rigid=rigid, **LEVELS[level])


def lengths(G):
    return np.array([sum(np.linalg.norm(np.diff(p, axis=0), axis=1).sum() for p in gs) for gs in G["geo"]])


def targets(G, Gin, T, P):
    """the per-frame targets: lane (N,) class or -1 (not a road of length >= MIN_LEN), link (N, N) 1 true / 0 non-link / -1 not a candidate or an input link, dec (N, N) the deciding P2 pairs, route (N,) 1 / 0 over p3_set, -1 else"""
    N = len(G["kind"])
    lane = np.where((G["kind"] == 0) & (lengths(G) >= MIN_LEN), T["lanes"], -1)
    link = np.full((N, N), -1, np.int64)
    a, b = SG.candidates(P)
    link[a, b] = 0
    for i, j in T["succ"]:
        if link[i, j] == 0:
            link[i, j] = 1
    for i, j in Gin["succ"]:
        link[i, j] = -1                                                      # an input link is given, not predicted
    flag = Gin["route_in"] >= 0
    dec = (link >= 0) & ~(flag[:, None] & flag[None, :])
    route = np.full(N, -1, np.int64)
    s = SG.p3_set(Gin)
    route[s] = (T["route"][s] >= 0).astype(np.int64)
    return dict(lane=lane, link=link, dec=dec, route=route)


def item(G, Gin, T):
    f = SG.featurize(Gin)
    f.update(targets(G, Gin, T, f["pair"]))
    return f


# ------------------------------------------------------------------------------------------------------------------------------ metrics and the scene bootstrap
def auc(score, lab):
    """the AUC by ranks (ties averaged); nan without both classes"""
    score, lab = np.asarray(score, float), np.asarray(lab, int)
    npos, nneg = int(lab.sum()), int((1 - lab).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    r = np.empty(len(score))
    s = score[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        r[order[i: j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    return float((r[lab == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def ap(score, lab):
    score, lab = np.asarray(score, float), np.asarray(lab, int)
    if lab.sum() == 0:
        return float("nan")
    o = np.argsort(-score, kind="mergesort")
    hit = lab[o]
    prec = np.cumsum(hit) / np.arange(1, len(hit) + 1)
    return float((prec * hit).sum() / hit.sum())


class Rows:
    """per-frame records of one read: P1 (pred, true), P2 (score, label, deciding), P3 (score, label); `metric(name, frames)` pools them over a frame list"""

    def __init__(self):
        self.p1, self.p2, self.p3 = {}, {}, {}

    def metric(self, name, fr=None):
        fr = range(len(self.p1)) if fr is None else fr
        if name == "p1":
            p = np.concatenate([self.p1[k][0] for k in fr]) if len(fr) else np.zeros(0)
            t = np.concatenate([self.p1[k][1] for k in fr]) if len(fr) else np.zeros(0)
            return float((p == t).mean()) if len(t) else float("nan")
        src = self.p2 if name.startswith("p2") else self.p3
        sel = [src[k] for k in fr]
        if not sel:
            return float("nan")
        s = np.concatenate([x[0] for x in sel])
        lab = np.concatenate([x[1] for x in sel])
        if name in ("p2", "p2ap"):
            m = np.concatenate([x[2] for x in sel])
            s, lab = s[m], lab[m]
        return ap(s, lab) if name.endswith("ap") else auc(s, lab)

    def counts(self):
        return dict(p1_nodes=int(sum(len(v[1]) for v in self.p1.values())), p2_pos_dec=int(sum(v[1][v[2]].sum() for v in self.p2.values())), p2_neg_dec=int(sum((1 - v[1][v[2]]).sum() for v in self.p2.values())),
                    p3_pos=int(sum(v[1].sum() for v in self.p3.values())), p3_neg=int(sum((1 - v[1]).sum() for v in self.p3.values())))


def boot(scene, fn, B=B_BOOT, seed=0):
    """a scene bootstrap of fn(frame indices) -> (value on all frames, lo, hi)"""
    scene = np.asarray(scene)
    us = np.unique(scene)
    by = {s: np.flatnonzero(scene == s) for s in us}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        fr = np.concatenate([by[s] for s in rng.choice(us, len(us))])
        vals.append(fn(fr))
    vals = np.array(vals, float)
    vals = vals[np.isfinite(vals)]
    return float(fn(np.arange(len(scene)))), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired(scene, a, b, name, B=B_BOOT):
    return boot(scene, lambda fr: a.metric(name, fr) - b.metric(name, fr), B)


# ------------------------------------------------------------------------------------------------------------------------------ the nulls and the Bayes ceiling
def noise_matrix(rate):
    """O[t, o] = P(input class o | true class t) at the count error rate (the clipping of `perturb`)"""
    O = np.eye(SG.N_LANE) * (1 - rate)
    for t in range(SG.N_LANE):
        for d in (-1, 1):
            o = t + d
            o = 1 if o < 0 else (SG.N_LANE - 2 if o > SG.N_LANE - 1 else o)
            O[t, o] += rate / 2
    return O


def bayes_fit(graphs):
    """from the TRAIN exact graphs: the prior of the lane class and, per relation (succ, pred, opp), P(the neighbour road's class | own class) (Laplace +1)"""
    prior = np.ones(SG.N_LANE)
    M = {r: np.ones((SG.N_LANE, SG.N_LANE)) for r in ("succ", "pred", "opp")}
    for G in graphs:
        L, K = G["lanes"], G["kind"]
        prior += np.bincount(L[K == 0], minlength=SG.N_LANE)
        for a, b in G["succ"]:
            if K[a] == 0 and K[b] == 0:
                M["succ"][L[a], L[b]] += 1
                M["pred"][L[b], L[a]] += 1
        for a, b in G["oppl"]:
            if K[a] == 0 and K[b] == 0:
                M["opp"][L[a], L[b]] += 1
    return dict(prior=(prior / prior.sum()).tolist(), **{r: (m / m.sum(1, keepdims=True)).tolist() for r, m in M.items()})


def bayes_p1(Gin, fit, rate):
    """the posterior argmax of every road's class from its own input class and its INPUT succ / pred / opposite roads' input classes (naive Bayes under the known noise; unknown inputs skipped)"""
    O = noise_matrix(rate)
    lin, K = Gin["lanes_in"], Gin["kind"]
    lp = np.log(np.array(fit["prior"]))
    nb = {i: [] for i in range(len(K))}
    for a, b in Gin["succ"]:
        nb[a].append(("succ", b))
        nb[b].append(("pred", a))
    for a, b in Gin["oppl"]:
        nb[a].append(("opp", b))
    out = np.zeros(len(K), int)
    for i in range(len(K)):
        if K[i] != 0:
            continue
        s = lp.copy()
        if lin[i] < SG.N_LANE:
            s += np.log(O[:, lin[i]] + 1e-12)
        for r, j in nb[i]:
            if K[j] == 0 and lin[j] < SG.N_LANE:
                s += np.log(np.array(fit[r]) @ O[:, lin[j]] + 1e-12)
        out[i] = int(np.argmax(s))
    return out


def null_rows(graphs, level, fit):
    """the rule rows of one level: copy / vote / bayes (P1), the link rule (P2), the route rule (P3)"""
    rows = {k: Rows() for k in ("copy", "vote", "bayes", "rule")}
    for k, G in enumerate(graphs):
        Gin, T = val_draw(G, k, level)
        P = SG.pair_features(Gin)
        tg = targets(G, Gin, T, P)
        m = tg["lane"] >= 0
        c, v = SG.null_p1(Gin)
        b = bayes_p1(Gin, fit, LEVELS[level].get("count_rate", 0.0))
        for name, pred in (("copy", c), ("vote", v), ("bayes", b)):
            rows[name].p1[k] = (pred[m], tg["lane"][m])
        sel = tg["link"] >= 0
        rows["rule"].p2[k] = (SG.null_p2(P)[sel], tg["link"][sel], tg["dec"][sel])
        s3 = SG.null_p3(Gin, P)
        idx = np.flatnonzero(tg["route"] >= 0)
        rows["rule"].p3[k] = (np.array([s3[i] for i in idx], float), tg["route"][idx])
        for name in ("copy", "vote", "bayes"):
            rows[name].p2[k], rows[name].p3[k] = rows["rule"].p2[k], rows["rule"].p3[k]
        rows["rule"].p1[k] = rows["copy"].p1[k]
    return rows


def nulls(log=print):
    out_path = os.path.join(SG.CACHE, "nulls_val.json")
    assert not os.path.exists(out_path), f"never overwritten: {out_path}"
    tr, va = SG.load("train"), SG.load("val")
    fit = bayes_fit(tr["graphs"])
    res = dict(bayes_fit=fit, levels={})
    for level in LEVELS:
        t0 = time.time()
        rows = null_rows(va["graphs"], level, fit)
        sc = va["scene"]
        L = dict(counts=rows["rule"].counts())
        for name in ("copy", "vote", "bayes"):
            L[f"p1_{name}"] = boot(sc, lambda fr, r=rows[name]: r.metric("p1", fr))
        for name in ("p2", "p2ap", "p2all", "p3"):
            L[f"{name}_rule"] = boot(sc, lambda fr, n=name: rows["rule"].metric(n, fr))
        res["levels"][level] = L
        log(f"{level}: " + ", ".join(f"{k} {v[0]:.4f} [{v[1]:.4f}, {v[2]:.4f}]" for k, v in L.items() if k != "counts") + f"; {L['counts']} ({time.time() - t0:.0f} s)", flush=True)
    json.dump(res, open(out_path, "w"), indent=1)
    log(f"written {os.path.relpath(out_path, ROOT)}")
    return res


# ------------------------------------------------------------------------------------------------------------------------------ training and read
def build_models(arm, seed):
    torch.manual_seed(seed)
    enc = JS.SDGraphEncoder(SG.NODE_F, SG.EDGE_F, rel=len(SG.REL), d=CFG["d"], heads=CFG["heads"], layers=CFG["layers"] if arm == "g" else 0)
    heads = JS.ProbeHeads(CFG["d"], SG.EDGE_F)
    return enc, heads


def forward(enc, heads, b):
    pair = b["pair"].float()
    h = enc(b["x"].float(), b["rel"], pair, b["valid"])
    return heads.lanes(h), heads.links(h, pair), heads.route(h)


def losses(out, b):
    l1, l2, l3 = out
    lane, link, route = b["lane"], b["link"], b["route"]
    m1 = lane >= 0
    loss1 = F.cross_entropy(l1[m1], lane[m1]) if m1.any() else l1.sum() * 0
    m2 = link >= 0
    y2 = link[m2].float()
    pw = torch.clamp((1 - y2).sum() / y2.sum().clamp_min(1), 1.0, 10.0)
    loss2 = F.binary_cross_entropy_with_logits(l2[m2], y2, pos_weight=pw) if m2.any() else l2.sum() * 0
    m3 = route >= 0
    loss3 = F.binary_cross_entropy_with_logits(l3[m3], route[m3].float()) if m3.any() else l3.sum() * 0
    return loss1 + loss2 + loss3, dict(p1=float(loss1), p2=float(loss2), p3=float(loss3))


def train(arm, seed, graphs=None, epochs=None, out=None, log=print):
    """the arm's encoder + heads on the TRAIN draws: CFG['draws'] draws per frame per epoch block, regenerated every epoch from a generator seeded by (seed, epoch, frame); Adam; the final model is kept"""
    out = out or runs_dir(arm, seed)
    assert not os.path.exists(os.path.join(out, "model.pt")), f"never overwritten: {out}"
    graphs = graphs if graphs is not None else SG.load("train")["graphs"]
    graphs = [G for G in graphs if len(G["kind"]) >= 2]
    enc, heads = build_models(arm, seed)
    opt = torch.optim.Adam(list(enc.parameters()) + list(heads.parameters()), lr=CFG["lr"])
    epochs = epochs or CFG["epochs"]
    hist, t0 = [], time.time()
    for ep in range(epochs):
        rng = np.random.default_rng([seed, ep, CFG["seed_draw"]])
        order = rng.permutation(len(graphs))
        tot, n = np.zeros(4), 0
        for s in range(0, len(order), CFG["bs"]):
            items = []
            for k in order[s: s + CFG["bs"]]:
                Gin, T = train_draw(graphs[k], rng)
                items.append(item(graphs[k], Gin, T))
            b = JS.collate(items)
            loss, parts = losses(forward(enc, heads, b), b)
            assert torch.isfinite(loss), f"the loss is not finite (epoch {ep}): the card's kill"
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += [float(loss), parts["p1"], parts["p2"], parts["p3"]]
            n += 1
        hist.append(dict(epoch=ep, loss=round(tot[0] / n, 5), p1=round(tot[1] / n, 5), p2=round(tot[2] / n, 5), p3=round(tot[3] / n, 5), s=round(time.time() - t0, 1)))
        log(json.dumps(hist[-1]), flush=True)
    os.makedirs(out, exist_ok=True)
    params = sum(p.numel() for p in enc.parameters()) + sum(p.numel() for p in heads.parameters())
    torch.save(dict(enc=enc.state_dict(), heads=heads.state_dict(), arm=arm, seed=seed, cfg=CFG, version=SG.VERSION), os.path.join(out, "model.pt"))
    json.dump(dict(arm=arm, seed=seed, cfg=CFG, params=params, params_encoder=sum(p.numel() for p in enc.parameters()), hist=hist), open(os.path.join(out, "train.json"), "w"), indent=1)
    return enc, heads


def load_models(arm, seed):
    ck = torch.load(os.path.join(runs_dir(arm, seed), "model.pt"), map_location="cpu", weights_only=False)
    enc, heads = build_models(arm, seed)
    enc.load_state_dict(ck["enc"])
    heads.load_state_dict(ck["heads"])
    return enc.eval(), heads.eval()


@torch.no_grad()
def model_rows(enc, heads, graphs, level, bs=32):
    rows = Rows()
    for s in range(0, len(graphs), bs):
        its, ks = [], list(range(s, min(len(graphs), s + bs)))
        for k in ks:
            Gin, T = val_draw(graphs[k], k, level)
            its.append(item(graphs[k], Gin, T))
        b = JS.collate(its)
        l1, l2, l3 = forward(enc, heads, b)
        for r, k in enumerate(ks):
            n = len(its[r]["x"])
            lane = its[r]["lane"]
            m = lane >= 0
            rows.p1[k] = (l1[r, :n].argmax(-1).numpy()[m], lane[m])
            sel = its[r]["link"] >= 0
            rows.p2[k] = (l2[r, :n, :n].numpy()[sel], its[r]["link"][sel], its[r]["dec"][sel])
            idx = np.flatnonzero(its[r]["route"] >= 0)
            rows.p3[k] = (l3[r, :n].numpy()[idx], its[r]["route"][idx])
    return rows


@torch.no_grad()
def invariance(enc, heads, graphs, draws=10):
    """P4: per VAL frame, `draws` rigid transforms (+-10 m, +-30 deg) of the base-level input: the largest embedding change / the mean |embedding|, and the argmax / sign flips of the probe outputs where the margin
    exceeds 1e-4"""
    worst, flips = 0.0, 0
    for k, G in enumerate(graphs):
        rng = np.random.default_rng([3904, k])
        ref = None
        for d in range(draws + 1):
            rig = None if d == 0 else (rng.uniform(-10, 10), rng.uniform(-10, 10), np.deg2rad(rng.uniform(-30, 30)))
            Gin, T = val_draw(G, k, "base", rigid=rig)
            b = JS.collate([item(G, Gin, T)])
            pair = b["pair"].float()
            h = enc(b["x"].float(), b["rel"], pair, b["valid"])
            o = (heads.lanes(h), heads.links(h, pair), heads.route(h))
            if ref is None:
                ref = (h, o)
                continue
            worst = max(worst, float((h - ref[0]).abs().max() / ref[0].abs().mean().clamp_min(1e-9)))
            top2 = ref[1][0].topk(2, -1).values
            sure = (top2[..., 0] - top2[..., 1]) > 1e-4
            flips += int(((o[0].argmax(-1) != ref[1][0].argmax(-1)) & sure).sum())
            for x, y in ((o[1], ref[1][1]), (o[2], ref[1][2])):
                flips += int(((x > 0) != (y > 0))[y.abs() > 1e-4].sum())
    return dict(max_rel_change=worst, flips=flips, frames=len(graphs), draws=draws)


def read(arm, seed, log=print):
    out_path = os.path.join(runs_dir(arm, seed), "probe_card.json")
    assert not os.path.exists(out_path), f"never overwritten: {out_path}"
    enc, heads = load_models(arm, seed)
    va = SG.load("val")
    tr = SG.load("train")
    fit = bayes_fit(tr["graphs"])
    sc = va["scene"]
    card = dict(arm=arm, seed=seed, levels={}, version=SG.VERSION)
    for level in LEVELS:
        t0 = time.time()
        r = model_rows(enc, heads, va["graphs"], level)
        nr = null_rows(va["graphs"], level, fit)
        L = dict(counts=r.counts())
        for name in ("p1", "p2", "p2ap", "p2all", "p3"):
            L[name] = boot(sc, lambda fr, n=name: r.metric(n, fr))
        L["p1_minus_copy"] = paired(sc, r, nr["copy"], "p1")
        L["p1_minus_vote"] = paired(sc, r, nr["vote"], "p1")
        L["p1_minus_bayes"] = paired(sc, r, nr["bayes"], "p1")
        L["p2_minus_rule"] = paired(sc, r, nr["rule"], "p2")
        L["p3_minus_rule"] = paired(sc, r, nr["rule"], "p3")
        card["levels"][level] = L
        log(f"{arm}_s{seed} {level}: " + ", ".join(f"{k} {v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}]" for k, v in L.items() if k != "counts") + f" ({time.time() - t0:.0f} s)", flush=True)
    card["p4"] = invariance(enc, heads, va["graphs"])
    log(f"P4: {card['p4']}")
    json.dump(card, open(out_path, "w"), indent=1)
    log(f"written {os.path.relpath(out_path, ROOT)}")
    return card


def paired_models(a, b, level, log=print):
    """g minus n on the same draws (the graph clause): P1 and P2 at a level"""
    va = SG.load("val")
    ra, rb = model_rows(*load_models(*a), va["graphs"], level), model_rows(*load_models(*b), va["graphs"], level)
    return {n: paired(va["scene"], ra, rb, n) for n in ("p1", "p2", "p3")}


# ------------------------------------------------------------------------------------------------------------------------------ selftest
def selftest():
    assert auc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) == 0.75 and auc([1, 1], [0, 1]) == 0.5 and np.isnan(auc([1, 2], [1, 1]))
    assert abs(ap([0.9, 0.8, 0.1], [1, 0, 1]) - (1 + 2 / 3) / 2) < 1e-12
    O = noise_matrix(0.3)
    assert np.allclose(O.sum(1), 1) and O[0, 1] == 0.3 and O[3, 2] == 0.3 and O[1, 0] == 0.15
    v, lo, hi = boot(np.array([0, 0, 1, 1]), lambda fr: float(len(fr)), B=50)
    assert v == 4 and lo == hi == 4
    G = SG.toy()
    graphs = [G] * 8
    import tempfile
    with tempfile.TemporaryDirectory(dir=os.path.join(ROOT, "temp")) as td:
        enc, heads = train("g", 0, graphs=graphs, epochs=3, out=os.path.join(td, "g"), log=lambda *a, **k: None)
        Gin, T = val_draw(G, 0, "base")
        it = item(G, Gin, T)
        b = JS.collate([it])
        l1, l2, l3 = forward(enc.eval(), heads.eval(), b)
        assert l1.shape == (1, 5, 4) and l2.shape == (1, 5, 5) and l3.shape == (1, 5)
        assert torch.isfinite(l1).all() and torch.isfinite(l2).all()
    fit = bayes_fit([G])
    Gc, _ = SG.perturb(G, np.random.default_rng(0))
    assert (bayes_p1(Gc, fit, 0.0)[G["kind"] == 0] == G["lanes"][G["kind"] == 0]).all(), "at error 0 the Bayes ceiling is the input"
    print("sdgraph_probe selftest ok")


def main(argv=None):
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("cmd", choices=("nulls", "train", "read", "selftest"))
    ap_.add_argument("--arm", choices=("g", "n"))
    ap_.add_argument("--seed", type=int, default=0)
    a = ap_.parse_args(argv)
    torch.set_num_threads(2)
    if a.cmd == "selftest":
        return selftest()
    if a.cmd == "nulls":
        return nulls()
    if a.cmd == "train":
        return train(a.arm, a.seed)
    return read(a.arm, a.seed)


if __name__ == "__main__":
    main()
