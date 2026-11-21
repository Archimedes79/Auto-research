"""E391 step 2 (card experiments/E391_sd-map-fusion-association): the association of the champion's perceived tokens to the E390 road-graph segments: data, nulls, training, read.

    wsl bash tools/env/_py14.sh mapassoc_probe.py nulls --seed S [--split val|holdout]   # no fusion training: the zero-shot E390 P6 head, the nearest rule, the logistic rule (both fitted on TRAIN tokens), the oracle
                                                                                         #   -> runs/e391_assoc/nulls_<split>_s<S>.json (never overwritten)
    wsl bash tools/env/_py14.sh mapassoc_probe.py gatenull --seed S [--split val|holdout] # card v2.1: the zero-shot P6 head + a NO-MAP false-token gate (logistic on the token's own outputs, fitted on TRAIN)
                                                                                         #   -> runs/e391_assoc/fit_gate_s<S>.json, nulls_gate_<split>_s<S>.json
    wsl bash tools/env/_py14.sh mapassoc_probe.py train --arm z|P|F1|F2|F3 --seed S       # -> runs/e391_<arm>_s<S>/model.pt, train.json (GPU when visible; z also on the CPU)
    wsl bash tools/env/_py14.sh mapassoc_probe.py read  --arm A --seed S [--split val|holdout]   # -> runs/e391_<arm>_s<S>/card_<split>.json, paired with the nulls on the same draws
    wsl bash tools/env/_py14.sh mapassoc_probe.py selftest

Data (`AssocData`): the cached B' tokens of cache seed S (E365, K 512), the E390 road graph of the frame (v6 cache) under a map draw, the E390 features with the map's stated count-error rate (`featurize(stated_rate=True)`,
the encoder of `runs/e390_g7_s<S>`, frozen), the token-segment geometry (`prox_features` with the tokens as proxies), the labels of `lane_mapassoc_data` (the association, the node-level route role, the delivered mask).
Targets per token: assoc = the segment, or -1 'unassigned' (a false token, or its segment missing from the map), -2 ignored (invalid, unlabelled); lanes = the segment's logical lane class (present road segments);
route = E389's on_lane (tokens with a GT node). VAL draws: one generator per frame, seeded by (391, k), for every level (nested); TRAIN draws: the step-2 mix (simple errors only). The TRAIN holdout (every 10th
TRAIN scene, as E390 v7) is never trained on and is read first. Metrics (delivered tokens, pooled over frames, a scene bootstrap B 1000): assoc_acc (deciding), its parts seg_acc / none_recall (missing
segment) / false_assign (a false token given a segment) / false_unassigned (a token with a present segment left unassigned); lanes_acc; route_f1.
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
for sub_ in ("training/data", "training/data/prep", "training/grid", "training/train", "evaluation", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub_)))
import jmapassoc as JM  # noqa: E402
import jsdgraph as JS  # noqa: E402
import lane_mapassoc_data as MA  # noqa: E402
import lane_sdgraph_data as SG  # noqa: E402
import sdgraph_probe as SP  # noqa: E402

LEVELS = {                                                                     # card v2, section 1.2: the deciding level dec and the reported levels (simple errors only: step 3 has the error models)
    "dec": dict(pose=1.0, count_rate=0.1, drop=0.1, hide=0.3, node_drop=0.1),
    "exact": dict(),
    "pose2": dict(pose=2.0, count_rate=0.1, drop=0.1, hide=0.3, node_drop=0.1),
    "nd2": dict(pose=1.0, count_rate=0.1, drop=0.1, hide=0.3, node_drop=0.2),
}
CFG = dict(steps=3000, bs=8, lr=1e-3, warmup=0.05, clip=3.0, workers=4, seed_draw=391)
B_BOOT = 1000


def train_kw(rng):
    """one TRAIN map draw of step 2 (simple errors only): pose {0, 1, 2}, count error {0, 0.1}, link drop {0, 0.1}, route hide {0, 0.3}, missing segments {0, 0.1}"""
    return dict(pose=float(rng.choice([0.0, 1.0, 2.0])), count_rate=float(rng.choice([0.0, 0.1])), drop=float(rng.choice([0.0, 0.1])), hide=float(rng.choice([0.0, 0.3])), node_drop=float(rng.choice([0.0, 0.1])))


class AssocData(torch.utils.data.Dataset):
    """the items of a split and cache seed: `level` = a VAL / holdout level (fixed draws) or None = the TRAIN mix; `frames` = the frame indices; `need_map` False for the no-map arm z"""

    def __init__(self, split, seed, frames=None, level=None, need_map=True):
        import lane_ptok_data as LP
        self.ds = LP.PTokData(split, seed, "e292_a", None, 0.0)
        self.sg = SG.load(split)
        self.lab = MA.load(split, seed)
        assert list(self.ds.tokens) == list(self.sg["tokens"]) == list(self.lab["tokens"])
        self.frames = list(range(len(self.ds))) if frames is None else list(frames)
        self.level, self.need_map, self.seed = level, need_map, seed

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, j):
        i = self.frames[j]
        it = self.ds[i]
        G = self.sg["graphs"][i]
        rec = self.lab["frames"][i]
        if self.level is None:
            rng = np.random.default_rng([CFG["seed_draw"], self.seed, i, int(torch.randint(1 << 30, (1,)))])
            kw = train_kw(rng)
        else:
            rng = np.random.default_rng([391, i])
            kw = LEVELS[self.level]
        Gin, T = SG.perturb(G, rng, **kw)
        pos, dirv = np.asarray(it["pos"], float), np.asarray(it["dirv"], float)
        out = {k: np.asarray(it[k]) for k in ("tok", "pos", "dirv", "e", "flags", "valid")}
        seg = rec["seg"].astype(int)
        present = T["present"]
        kind = np.full(len(seg), -1, int)                                     # 0 a present segment, 1 its segment missing from the map, 2 a false token
        lab = seg >= 0
        pres_lab = lab & present[np.clip(seg, 0, None)]
        kind[pres_lab] = 0
        kind[lab & ~pres_lab] = 1
        kind[seg == MA.FALSE] = 2
        assoc = np.where(kind == 0, seg, np.where(kind >= 1, -1, -2))
        lanes = np.where((kind == 0) & (G["kind"][np.clip(seg, 0, None)] == 0), G["lanes"][np.clip(seg, 0, None)], -1)
        out.update(assoc=assoc.astype(np.int64), kind=kind.astype(np.int64), lanes=lanes.astype(np.int64), route=rec["on_lane"].astype(np.int64),
                   kept=np.asarray(rec.get("kept", it["valid"]), bool), frame=np.int64(i))
        if self.need_map:
            Gin = dict(Gin, prox=pos, pdir=dirv)
            f = SG.featurize(Gin, stated_rate=True)
            out["map"] = dict(x=f["x"], rel=f["rel"], pair=f["pair"], present=f["present"])
            out["pa"] = f["pa"]
            out["lanes_in"] = Gin["lanes_in"]
            out["route_in"] = Gin["route_in"]
            out["seg_kind"] = G["kind"]
        return out


def collate(items):
    b = {k: torch.as_tensor(np.stack([it[k] for it in items])) for k in ("tok", "pos", "dirv", "e", "flags", "valid", "assoc", "kind", "lanes", "route", "kept", "frame")}
    if "map" in items[0]:
        mb = JS.collate([it["map"] for it in items])
        N = mb["x"].shape[1]
        K = items[0]["pa"].shape[0]
        pa = np.zeros((len(items), K, N, items[0]["pa"].shape[-1]), np.float32)
        lin = np.full((len(items), N), -1, np.int64)
        rin = np.full((len(items), N), -1, np.int64)
        skind = np.full((len(items), N), -1, np.int64)
        for r, it in enumerate(items):
            n = it["pa"].shape[1]
            pa[r, :, :n] = it["pa"]
            lin[r, :n], rin[r, :n], skind[r, :n] = it["lanes_in"], it["route_in"], it["seg_kind"]
        b.update(map=mb, pa=torch.as_tensor(pa), lanes_in=torch.as_tensor(lin), route_in=torch.as_tensor(rin), seg_kind=torch.as_tensor(skind))
    return b


def encode(enc, b, dev):
    """the frozen E390 encoder on the batch's map -> (h (B, N, 64), seg_valid (B, N))"""
    mb = b["map"]
    with torch.no_grad():
        h = enc(mb["x"].float().to(dev), mb["rel"].to(dev), mb["pair"].float().to(dev), mb["valid"].to(dev))
    return h, mb["valid"].to(dev)


def to_dev(b, dev):
    return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}


def losses(out, b):
    lan, rou = b["lanes"], b["route"]
    m1 = lan >= 0
    l_lanes = F.cross_entropy(out["lanes"][m1], lan[m1]) if m1.any() else out["lanes"].sum() * 0
    m2 = rou >= 0
    l_route = F.binary_cross_entropy_with_logits(out["route"][m2], rou[m2].float()) if m2.any() else out["route"].sum() * 0
    parts = dict(lanes=float(l_lanes), route=float(l_route))
    loss = l_lanes + l_route
    if "assoc" in out:
        a = b["assoc"]
        tgt = torch.where(a == -1, torch.full_like(a, out["assoc"].shape[-1] - 1), a)
        m3 = a >= -1
        l_assoc = F.cross_entropy(out["assoc"][m3], tgt[m3]) if m3.any() else out["assoc"].sum() * 0
        parts["assoc"] = float(l_assoc)
        loss = loss + l_assoc
    return loss, parts


def runs_dir(arm, seed):
    return os.path.join(ROOT, "runs", f"e391_{arm}_s{seed}")


def train_frames(seed):
    lab = MA.load("train", seed)
    ho = set(lab["holdout"])
    return [i for i, sc in enumerate(lab["scene"]) if sc not in ho]


def train(arm, seed, steps=None, frames=None, out=None, log=print):
    """the fusion head of an arm on the TRAIN frames outside the holdout (AdamW, warm-up + cosine, clip; the E390 encoder runs/e390_g7_s<S> frozen); the final model is kept"""
    out = out or runs_dir(arm, seed)
    assert not os.path.exists(os.path.join(out, "model.pt")), f"never overwritten: {out}"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    steps = steps or CFG["steps"]
    data = AssocData("train", seed, frames=frames if frames is not None else train_frames(seed), need_map=arm != "z")
    dl = torch.utils.data.DataLoader(data, batch_size=CFG["bs"], shuffle=True, drop_last=True, num_workers=CFG["workers"], collate_fn=collate, persistent_workers=CFG["workers"] > 0)
    head = JM.MapAssocHead(arm).to(dev)
    enc = None
    if arm != "z":
        enc, _ = SP.load_models("g7", seed)
        enc = enc.to(dev).eval()
    opt = torch.optim.AdamW(head.parameters(), lr=CFG["lr"])
    warm = int(CFG["warmup"] * steps)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + np.cos(np.pi * (s - warm) / max(1, steps - warm))))
    hist, t0, step, it = [], time.time(), 0, iter(dl)
    first = None
    while step < steps:
        try:
            b = next(it)
        except StopIteration:
            it = iter(dl)
            b = next(it)
        b = to_dev(b, dev)
        if arm != "z":
            h, sv = encode(enc, b, dev)
            o = head(b, h, sv, b["pa"])
        else:
            o = head(b)
        loss, parts = losses(o, b)
        assert torch.isfinite(loss), f"the loss is not finite at step {step} (the card's kill)"
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), CFG["clip"])
        opt.step()
        sched.step()
        step += 1
        if step == 100:
            first = float(loss)
        if step % 100 == 0:
            hist.append(dict(step=step, loss=round(float(loss), 5), **{k: round(v, 5) for k, v in parts.items()}, s=round(time.time() - t0, 1)))
            log(json.dumps(hist[-1]), flush=True)
    os.makedirs(out, exist_ok=True)
    torch.save(dict(state=head.state_dict(), arm=arm, seed=seed, cfg=CFG), os.path.join(out, "model.pt"))
    params = sum(p.numel() for p in head.parameters())
    last = float(np.mean([r["loss"] for r in hist[-5:]])) if hist else float("nan")
    json.dump(dict(arm=arm, seed=seed, cfg=CFG, params=params, frames=len(data), loss_step100=first, loss_last=last, learn_ratio=last / first if first else None, hist=hist), open(os.path.join(out, "train.json"), "w"), indent=1)
    assert first is None or last < 0.8 * first, f"no learning: loss {last} against {first} at step 100 (the card's kill)"
    return head


# ------------------------------------------------------------------------------------------------------------------------------ predictions, nulls, metrics
class Rows:
    """per frame: assoc (pred, target, kind), lanes (pred, target), route (prob, target), over the DELIVERED tokens"""

    def __init__(self):
        self.a, self.l, self.r = {}, {}, {}

    def add(self, k, b, r, ap=None, lp=None, rp=None):
        kept = b["kept"][r].numpy()
        tg, kd = b["assoc"][r].numpy(), b["kind"][r].numpy()
        if ap is not None:
            m = kept & (tg >= -1)
            self.a[k] = (ap[m], tg[m], kd[m])
        if lp is not None:
            lt = b["lanes"][r].numpy()
            m = kept & (lt >= 0)
            self.l[k] = (lp[m], lt[m])
        if rp is not None:
            rt = b["route"][r].numpy()
            m = kept & (rt >= 0)
            self.r[k] = (rp[m], rt[m])

    def metric(self, name, fr):
        fr = [k for k in fr]
        if name in ("assoc_acc", "nonfalse_acc", "seg_acc", "none_recall", "false_assign", "false_unassigned"):
            sel = [self.a[k] for k in fr if k in self.a]
            if not sel:
                return float("nan")
            p, t, kd = (np.concatenate(x) for x in zip(*sel))
            if name == "assoc_acc":
                m, v = np.ones(len(t), bool), p == t
            elif name == "nonfalse_acc":                                          # card v2.1: the tokens with a GT segment only (present or missing), false tokens left out
                m, v = kd <= 1, p == t
            elif name == "seg_acc":
                m, v = kd == 0, p == t
            elif name == "none_recall":
                m, v = kd == 1, p == -1
            elif name == "false_assign":
                m, v = kd == 2, p != -1
            else:
                m, v = kd == 0, p == -1
            return float(v[m].mean()) if m.any() else float("nan")
        if name == "lanes_acc":
            sel = [self.l[k] for k in fr if k in self.l]
            if not sel:
                return float("nan")
            p, t = (np.concatenate(x) for x in zip(*sel))
            return float((p == t).mean()) if len(t) else float("nan")
        sel = [self.r[k] for k in fr if k in self.r]
        if not sel:
            return float("nan")
        p, t = (np.concatenate(x) for x in zip(*sel))
        yhat = p >= 0.5
        tp, fp, fn = (yhat & (t == 1)).sum(), (yhat & (t == 0)).sum(), (~yhat & (t == 1)).sum()
        return float(2 * tp / max(2 * tp + fp + fn, 1))


METRICS = ("assoc_acc", "nonfalse_acc", "seg_acc", "none_recall", "false_assign", "false_unassigned", "lanes_acc", "route_f1")


def boot(scene, frames, fn, B=B_BOOT, seed=0):
    sc = np.asarray([scene[k] for k in frames])
    fr = np.asarray(frames)
    us = np.unique(sc)
    by = {s: fr[sc == s] for s in us}
    rng = np.random.default_rng(seed)
    vals = [fn(np.concatenate([by[s] for s in rng.choice(us, len(us))])) for _ in range(B)]
    vals = np.array(vals, float)
    vals = vals[np.isfinite(vals)]
    v = float(fn(fr))
    if not len(vals):
        return v, float("nan"), float("nan")
    return v, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def read_through(b, r, a_pred):
    """the attributes read through an association (a null's or the oracle's): lanes = the assigned segment's INPUT lane class (unknown / not a road / unassigned -> class 1 = 2 lanes), route = the assigned segment
    carries an input route flag"""
    lin, rin, sk = b["lanes_in"][r].numpy(), b["route_in"][r].numpy(), b["seg_kind"][r].numpy()
    ok = a_pred >= 0
    j = np.clip(a_pred, 0, None)
    lanes = np.where(ok & (sk[j] == 0) & (lin[j] >= 0) & (lin[j] < SG.N_LANE), lin[j], 1)
    route = (ok & (rin[j] >= 0)).astype(float)
    return lanes, route


def fit_p6(seed, n=800, log=print):
    """the nearest rule's tau and the logistic association null, fitted on TRAIN tokens (frames outside the holdout, the dec level), E390's functions with the tokens as proxies"""
    fr = train_frames(seed)
    rng = np.random.default_rng(3911)
    pick = sorted(rng.choice(fr, min(n, len(fr)), replace=False).tolist())
    data = AssocData("train", seed, frames=pick, level="dec")
    tau_grid = np.r_[np.arange(0.5, 20.01, 0.5), np.inf]
    X, Y, cache = [], [], []
    for j in range(len(data)):
        it = data[j]
        pa, tg = it["pa"], it["assoc"]
        Gin = dict(kind=it["seg_kind"], lanes_in=it["lanes_in"], present=it["map"]["present"])
        m = it["kept"] & (tg >= -1)
        cache.append((pa[m], tg[m]))
        f = SP.p6_simple(Gin, pa[m])
        pres = Gin["present"]
        lab = np.arange(len(pres))[None, :] == tg[m][:, None]
        X.append(f[:, pres].reshape(-1, SP.P6_LR_F))
        Y.append(lab[:, pres].reshape(-1))
    acc = [np.mean(np.concatenate([SP.p6_rule(None, pa, t) == y for pa, y in cache])) for t in tau_grid]
    tau = float(tau_grid[int(np.argmax(acc))])
    from sklearn.linear_model import LogisticRegression
    X, Y = np.concatenate(X), np.concatenate(Y).astype(int)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    lrm = LogisticRegression(max_iter=2000).fit((X - mu) / sd, Y)
    lr = dict(mean=mu.tolist(), std=sd.tolist(), coef=lrm.coef_[0].tolist(), bias=float(lrm.intercept_[0]))
    best = []
    for j, (pa, y) in enumerate(cache):
        it = data[j]
        Gin = dict(kind=it["seg_kind"], lanes_in=it["lanes_in"], present=it["map"]["present"])
        s = np.where(Gin["present"][None, :], SP.p6_lr_score(SP.p6_simple(Gin, pa), lr), -np.inf)
        best.append((s.max(1), np.where(np.isfinite(s.max(1)), s.argmax(1), -1), y))
    grid = np.linspace(-12, 4, 65)
    acc_lr = [np.mean(np.concatenate([np.where(bb >= thr, jj, -1) == yy for bb, jj, yy in best])) for thr in grid]
    lr["none_thr"] = float(grid[int(np.argmax(acc_lr))])
    log(f"fit on {len(pick)} TRAIN frames: tau {tau}, logistic none threshold {lr['none_thr']}", flush=True)
    return dict(tau=tau, tau_acc=[round(float(a), 4) for a in acc], lr=lr, frames=len(pick))


def gate_x(b, r):
    """the no-map false-token gate's inputs of row r: the token's own perception outputs (sigmoid(e), sigmoid(flags), tok), no map"""
    e, fl, tok = b["e"][r].float(), b["flags"][r].float(), b["tok"][r].float()
    return torch.cat([e.sigmoid()[:, None], fl.sigmoid(), tok], -1).numpy()


def gate_prob(gate, X):
    return 1.0 / (1.0 + np.exp(-(((X - np.asarray(gate["mean"])) / np.asarray(gate["std"])) @ np.asarray(gate["coef"]) + gate["bias"])))


def p6_assign(z6, r, N):
    pz = z6[r].argmax(-1).numpy()
    return np.where(pz == N, -1, pz)


def fit_gate(seed, n=800, log=print):
    """card v2.1 (critic item 1): the null 'zero-shot E390 P6 head + a NO-MAP false-token gate': a logistic on the token's own features (no map) predicting a false token, fitted on TRAIN tokens
    (frames outside the holdout, the dec level, the tokens the fusion arms are trained on); the threshold maximizes the combined association accuracy on the same TRAIN tokens"""
    from sklearn.linear_model import LogisticRegression
    fr = train_frames(seed)
    rng = np.random.default_rng(3912)
    pick = sorted(rng.choice(fr, min(n, len(fr)), replace=False).tolist())
    data = AssocData("train", seed, frames=pick, level="dec")
    dl = torch.utils.data.DataLoader(data, batch_size=8, shuffle=False, num_workers=CFG["workers"], collate_fn=collate)
    enc, heads6 = SP.load_models("g7", seed)
    enc, heads6 = enc.eval(), heads6.eval()
    X, PZ, TG, KD = [], [], [], []
    with torch.no_grad():
        for b in dl:
            h, sv = encode(enc, b, "cpu")
            z6 = heads6.assoc(h, b["pa"].float(), sv)
            for r in range(len(b["frame"])):
                tg = b["assoc"][r].numpy()
                m = b["kept"][r].numpy() & (tg >= -1)
                X.append(gate_x(b, r)[m])
                PZ.append(p6_assign(z6, r, h.shape[1])[m])
                TG.append(tg[m])
                KD.append(b["kind"][r].numpy()[m])
    X, PZ, TG, KD = (np.concatenate(v) for v in (X, PZ, TG, KD))
    y = (KD == 2).astype(int)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    lrm = LogisticRegression(max_iter=3000).fit((X - mu) / sd, y)
    gate = dict(mean=mu.tolist(), std=sd.tolist(), coef=lrm.coef_[0].tolist(), bias=float(lrm.intercept_[0]))
    p = gate_prob(gate, X)
    grid = np.round(np.arange(0.05, 0.951, 0.05), 2)
    acc = [float(np.mean(np.where(p >= t, -1, PZ) == TG)) for t in grid]
    gate.update(thr=float(grid[int(np.argmax(acc))]), acc_grid=[round(a, 4) for a in acc], acc_p6zero=round(float(np.mean(PZ == TG)), 4), frames=len(pick), tokens=int(len(y)), false_rate=round(float(y.mean()), 4))
    log(f"gate fit on {len(pick)} TRAIN frames, {len(y)} tokens (false {y.mean():.3f}): threshold {gate['thr']}, TRAIN acc {max(acc):.4f} (P6 zero-shot {gate['acc_p6zero']})", flush=True)
    return gate


def load_gate(seed, log=print):
    path = os.path.join(MA.CACHE, f"fit_gate_s{seed}.json")
    if not os.path.exists(path):
        json.dump(fit_gate(seed, log=log), open(path, "w"), indent=1)
    return json.load(open(path))


def gatenull(seed, split="val", log=print):
    """card v2.1: the gate null (p6gate) on the same draws as the other nulls -> runs/e391_assoc/nulls_gate_<split>_s<S>.json (never overwritten); also nonfalse_acc of p6zero / p6gate"""
    out_path = os.path.join(MA.CACHE, f"nulls_gate_{split}_s{seed}.json")
    assert not os.path.exists(out_path), f"never overwritten: {out_path}"
    fit = dict(json.load(open(os.path.join(MA.CACHE, f"fit_s{seed}.json"))), gate=load_gate(seed, log=log))
    res = dict(seed=seed, split=split, gate_thr=fit["gate"]["thr"], levels={})
    for level in LEVELS:
        t0 = time.time()
        rows, keys, scene = null_and_model_rows(split, seed, level, fit=fit, log=log)
        L = {}
        for name in ("p6zero", "p6gate"):
            for m in METRICS:
                L[f"{m}_{name}"] = boot(scene, keys, lambda fr, rr=rows[name], mm=m: rr.metric(mm, fr))
        res["levels"][level] = L
        log(f"{level}: " + ", ".join(f"{k} {v[0]:.4f}" for k, v in L.items()) + f" ({time.time() - t0:.0f} s)", flush=True)
    json.dump(res, open(out_path, "w"), indent=1)
    log(f"written {os.path.relpath(out_path, ROOT)}")
    return res


def eval_frames(split, seed):
    if split == "val":
        return None, MA.load("val", seed)["scene"]
    lab = MA.load("train", seed)
    ho = set(lab["holdout"])
    return [i for i, sc in enumerate(lab["scene"]) if sc in ho], lab["scene"]


def null_and_model_rows(split, seed, level, head=None, fit=None, log=print):
    """the rows of one level: the model head (when given) and the nulls (zero-shot E390 P6 head, nearest, logistic, oracle) on the same draws"""
    frames, scene = eval_frames(split, seed)
    data = AssocData(split if split == "val" else "train", seed, frames=frames, level=level)
    dl = torch.utils.data.DataLoader(data, batch_size=8, shuffle=False, num_workers=CFG["workers"], collate_fn=collate)
    enc, heads6 = SP.load_models("g7", seed)
    enc, heads6 = enc.eval(), heads6.eval()
    gate = fit.get("gate")
    rows = {k: Rows() for k in ("model", "p6zero", "nearest", "logistic", "oracle") + (("p6gate",) if gate else ())}
    with torch.no_grad():
        for b in dl:
            h, sv = encode(enc, b, "cpu")
            z6 = heads6.assoc(h, b["pa"].float(), sv)
            o = None
            if head is not None:
                o = head(b, h, sv, b["pa"]) if head.arm != "z" else head(b)
            N = h.shape[1]
            for r in range(len(b["frame"])):
                k = int(b["frame"][r])
                tg = b["assoc"][r].numpy()
                pz = p6_assign(z6, r, N)
                Gin = dict(kind=b["seg_kind"][r].numpy(), lanes_in=b["lanes_in"][r].numpy(), present=sv[r].numpy())
                pa = b["pa"][r].numpy()
                pn = SP.p6_rule(None, pa, fit["tau"])
                pl = SP.p6_lr(Gin, pa, fit["lr"])
                cand = [("p6zero", pz), ("nearest", pn), ("logistic", pl), ("oracle", np.where(tg >= -1, tg, -1))]
                if gate:
                    cand.append(("p6gate", np.where(gate_prob(gate, gate_x(b, r)) >= gate["thr"], -1, pz)))
                for name, ap in cand:
                    lanes, route = read_through(b, r, ap)
                    rows[name].add(k, b, r, ap, lanes, route)
                if o is not None:
                    ap = None
                    if "assoc" in o:
                        ap = o["assoc"][r].argmax(-1).numpy()
                        ap = np.where(ap == o["assoc"].shape[-1] - 1, -1, ap)
                    rows["model"].add(k, b, r, ap, o["lanes"][r].argmax(-1).numpy(), torch.sigmoid(o["route"][r]).numpy())
    keys = sorted(rows["oracle"].a)
    return rows, keys, scene


def nulls(seed, split="val", log=print):
    out_path = os.path.join(MA.CACHE, f"nulls_{split}_s{seed}.json")
    assert not os.path.exists(out_path), f"never overwritten: {out_path}"
    fit_path = os.path.join(MA.CACHE, f"fit_s{seed}.json")
    fit = json.load(open(fit_path)) if os.path.exists(fit_path) else fit_p6(seed, log=log)
    if not os.path.exists(fit_path):
        json.dump(fit, open(fit_path, "w"), indent=1)
    res = dict(seed=seed, split=split, fit=dict(tau=fit["tau"], none_thr=fit["lr"]["none_thr"]), levels={})
    for level in LEVELS:
        t0 = time.time()
        rows, keys, scene = null_and_model_rows(split, seed, level, fit=fit, log=log)
        L = {}
        for name in ("p6zero", "nearest", "logistic", "oracle"):
            for m in METRICS:
                L[f"{m}_{name}"] = boot(scene, keys, lambda fr, rr=rows[name], mm=m: rr.metric(mm, fr))
        L["counts"] = dict(frames=len(keys), tokens=int(sum(len(v[1]) for v in rows["oracle"].a.values())), absent=int(sum((v[2] == 1).sum() for v in rows["oracle"].a.values())),
                           false=int(sum((v[2] == 2).sum() for v in rows["oracle"].a.values())), lanes=int(sum(len(v[1]) for v in rows["oracle"].l.values())), route=int(sum(len(v[1]) for v in rows["oracle"].r.values())))
        res["levels"][level] = L
        log(f"{level}: " + ", ".join(f"{k} {v[0]:.4f}" for k, v in L.items() if k != "counts") + f"; {L['counts']} ({time.time() - t0:.0f} s)", flush=True)
    json.dump(res, open(out_path, "w"), indent=1)
    log(f"written {os.path.relpath(out_path, ROOT)}")
    return res


def load_head(arm, seed):
    ck = torch.load(os.path.join(runs_dir(arm, seed), "model.pt"), map_location="cpu", weights_only=False)
    head = JM.MapAssocHead(arm)
    head.load_state_dict(ck["state"])
    return head.eval()


def read(arm, seed, split="val", log=print):
    out_path = os.path.join(runs_dir(arm, seed), f"card_{split}.json")
    assert not os.path.exists(out_path), f"never overwritten: {out_path}"
    tj = json.load(open(os.path.join(runs_dir(arm, seed), "train.json")))
    assert tj["learn_ratio"] is not None and tj["learn_ratio"] < 0.8, f"{arm}_s{seed} failed the card's kill (learn ratio {tj['learn_ratio']}): not read"
    fit = dict(json.load(open(os.path.join(MA.CACHE, f"fit_s{seed}.json"))), gate=json.load(open(os.path.join(MA.CACHE, f"fit_gate_s{seed}.json"))))
    head = load_head(arm, seed)
    card = dict(arm=arm, seed=seed, split=split, levels={})
    for level in LEVELS:
        t0 = time.time()
        rows, keys, scene = null_and_model_rows(split, seed, level, head=head, fit=fit, log=log)
        L = {}
        for m in METRICS:
            L[m] = boot(scene, keys, lambda fr, mm=m: rows["model"].metric(mm, fr))
            for name in ("p6zero", "p6gate", "nearest", "logistic"):
                L[f"{m}_minus_{name}"] = boot(scene, keys, lambda fr, mm=m, nn=name: rows["model"].metric(mm, fr) - rows[nn].metric(mm, fr))
        card["levels"][level] = L
        log(f"{arm}_s{seed} {split} {level}: " + ", ".join(f"{m} {L[m][0]:.4f}" for m in METRICS) + f" ({time.time() - t0:.0f} s)", flush=True)
    json.dump(card, open(out_path, "w"), indent=1)
    log(f"written {os.path.relpath(out_path, ROOT)}")
    return card


def selftest():
    r = Rows()
    r.a = {0: (np.array([1, -1, 2]), np.array([1, -1, -1]), np.array([0, 1, 2])), 1: (np.array([-1]), np.array([0]), np.array([0]))}
    assert r.metric("assoc_acc", [0, 1]) == 0.5 and r.metric("none_recall", [0, 1]) == 1.0 and r.metric("false_assign", [0, 1]) == 1.0 and r.metric("false_unassigned", [0, 1]) == 0.5
    assert abs(r.metric("nonfalse_acc", [0, 1]) - 2 / 3) < 1e-9
    g = dict(mean=[0.0, 0.0], std=[1.0, 1.0], coef=[2.0, 0.0], bias=0.0)
    assert np.allclose(gate_prob(g, np.array([[0.0, 5.0], [10.0, 0.0]])), [0.5, 1 / (1 + np.exp(-20))])
    r.r = {0: (np.array([0.9, 0.2, 0.7]), np.array([1, 0, 0]))}
    assert abs(r.metric("route_f1", [0]) - 2 / 3) < 1e-9
    v, lo, hi = boot({0: 0, 1: 0, 2: 1}, [0, 1, 2], lambda fr: float(len(fr)), B=20)
    assert v == 3
    print("mapassoc_probe selftest ok")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("nulls", "gatenull", "train", "read", "selftest"))
    ap.add_argument("--arm", choices=JM.ARMS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="val", choices=("val", "holdout"))
    a = ap.parse_args(argv)
    torch.set_num_threads(2)
    if a.cmd == "selftest":
        return selftest()
    if a.cmd == "nulls":
        return nulls(a.seed, a.split)
    if a.cmd == "gatenull":
        return gatenull(a.seed, a.split)
    if a.cmd == "train":
        return train(a.arm, a.seed)
    return read(a.arm, a.seed, a.split)


if __name__ == "__main__":
    main()
