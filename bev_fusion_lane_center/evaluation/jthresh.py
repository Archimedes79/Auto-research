"""E388 (D125): the decode thresholds of the token head LEARNED by gradient instead of chosen on a grid. Three scalars tau_short, tau_long, tau_keep (sigmoid of u,
u = 0 at the start: 0.5 each) are fitted by Adam on a soft F1 of the delivered decision, on a labelled TRAIN dump of a frozen run; the result is a decode cell for
`jtoken_eval.py score` (the real scorer reads VAL once at it). The none gate is not part of it (off, as the champion).

    wsl bash tools/env/_py14.sh jthresh.py dump --run R --seed S [--split train --every 3]   # GPU: a LIVE dump of the run (B' = the run's own b_state) with the token loss's labels
                                                                                              #   -> runs/R/token_<split>_every<E>_live_labels.pkl (never overwritten)
    wsl bash tools/env/_py14.sh jthresh.py fit --run R --seed S [--split train --every 3]    # CPU: the fit -> runs/R/thresh_fit_<split>_every<E>.json (never overwritten); prints the cell
    wsl bash tools/env/_py14.sh jthresh.py selftest                                          # CPU: synthetic frames with a known optimum, hand numbers, mutants

The surrogate (card E388): per frame, the valid tokens i (p_none_i, g_i = assigned to a GT node by the token loss's label step), the dump's candidate pairs (p >= 0.05 inside
the gate; class long = the head's token chord > 8 m, as the decode) with y = the token loss's GT successor label, and EVERY GT link (i, j) with its class. keep k_i =
sigmoid((tau_keep - p_none_i) / T); link s_ij = sigmoid((p_ij - tau_class) / T) k_i k_j (both ends kept, as the decode). Soft counts over all frames: TP_c = sum s y,
FP_c = sum s (1 - y), FN_c = sum over the GT links of class c of (1 - s), a GT link with no candidate having s = 0; softF1 = 2 TP / (2 TP + FP + FN); keep likewise over the
valid tokens with g. Loss = 3 - softF1_short - softF1_long - softF1_keep. Adam (lr 0.05) on u, full batch, STEPS steps, the step of the lowest loss kept; the link taus are
floored at the dump's 0.05 and all three rounded to 3 decimals before they are written."""
import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for sub_ in ("evaluation", "training", "net/prep", "net/model", "net/loss", "training/data", "training/train", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub_)))

T, LR, STEPS, FLOOR = 0.05, 0.05, 400, 0.05                                   # the card's numbers
RECORD = (0.3, 0.2, 0.6)                                                      # the champion's hand-tuned cell (short, long, keep; none gate off)


def label_path(runs, run, split, every):
    return os.path.join(runs, run, f"token_{split}_every{every}_live_labels.pkl")


def fit_path(runs, run, split, every):
    return os.path.join(runs, run, f"thresh_fit_{split}_every{every}.json")


# ------------------------------------------------------------------------------------------------------------------------------ the labelled live dump (GPU)
def label_frames(out, lb, keep_p):
    """the frames of one head batch: valid, p_none, g (gt_id >= 0), the candidates (src, dst, p, long, y) as `jtoken_eval.batch_dump` writes them (p >= keep_p inside the gate)
    and every GT link (src, dst, long) of the token loss's labels `y` between valid tokens, its class from the head's own chord"""
    import jtoken as JK
    pr = JK.probs(out)
    npy = lambda t: t.detach().cpu().numpy()  # noqa: E731
    p, cand, longm, pnone = npy(pr["p"]), npy(pr["cand"]), npy(pr["longm"]), npy(pr["pt"][..., 0])
    y, valid, gid, yt = npy(lb["y"]).astype(bool), npy(out["valid"]).astype(bool), npy(lb["label"]["gt_id"]), npy(lb["ytype"])
    frames = []
    for b in range(len(p)):
        cs, cd = np.nonzero((p[b] >= keep_p) & cand[b])
        gs, gd = np.nonzero(y[b] & valid[b][:, None] & valid[b][None, :])
        frames.append(dict(token=lb["token"][b], valid=valid[b], pnone=pnone[b].astype(np.float32), g=(yt[b] > 0) & valid[b], g_gtid=(gid[b] >= 0) & valid[b],
                           cand=dict(src=cs.astype(np.int16), dst=cd.astype(np.int16), p=p[b][cs, cd].astype(np.float32), long=longm[b][cs, cd], y=y[b][cs, cd]),
                           gt=dict(src=gs.astype(np.int16), dst=gd.astype(np.int16), long=longm[b][gs, gd])))
    return frames


def dump(a, runs=None):
    import jtoken_eval as E
    import jtoken_train as JKT
    import lane_live_tok as LT
    runs = runs or E.RUNS
    assert torch.cuda.is_available() or a.cpu, "the live label dump runs B' on the GPU (--cpu only for tests)"
    dev = "cpu" if a.cpu else "cuda"
    if dev == "cuda":
        import gpu_gate
        gpu_gate.wait_for_room(who=f"jthresh dump {a.run}")
    path = label_path(runs, a.run, a.split, a.every)
    assert not os.path.exists(path), f"a stored dump is never overwritten: {path}"
    ck = torch.load(os.path.join(runs, a.run, "model.pt"), map_location="cpu", weights_only=False)
    m, args = JKT.load_model(a.run, dev, runs)
    assert args["seed"] == a.seed, f"--seed {a.seed} does not match the run (seed {args['seed']})"
    kw = E.cache_kw(args)
    ds = E.open_cache(a.split, kw["seed"], kw["ident"], kw["K"], kw.get("cache"))
    idx = E.sample(len(ds), a.split, a.every)[: a.max or None]
    src, b_source = E.live_source(os.path.join(runs, a.run, "model.pt"), ck, kw, ds.K)
    src = src.to(dev)
    data = E.live_data(a.split, kw["seed"], kw["ident"])
    assert list(data.tokens) == list(ds.tokens) and [int(r) for r in data.rows] == [int(r) for r in ds.rows], "the live frames are not the frames of the cache, in its order"
    frames = []
    for b in E.batches(data, idx, collate=LT.collate):
        with torch.no_grad():
            tks = [src.tokens(b["x"][c:c + E.LIVE_BS])[1] for c in range(0, len(b["x"]), E.LIVE_BS)]
            tk = {k: torch.cat([t[k] for t in tks]) for k in tks[0]}
            lb = src.labelled(tk, b["table"], b["token"], b["row"], token_drop=0.0, gate=True)
            frames += label_frames(E.run_head(m, lb, dev), lb, E.KEEP_P)
    meta = dict(run=a.run, seed=a.seed, split=a.split, every=a.every, idx=[int(i) for i in idx], tokens=[ds.tokens[i] for i in idx], live=True, b_source=b_source, device=dev, keep_p=E.KEEP_P, steps=args.get("steps"))
    E.check_b(meta, ck, kw, runs)
    with open(path, "wb") as fh:
        pickle.dump(dict(meta=meta, frames=frames), fh)
    n_c, n_g = np.mean([len(f["cand"]["src"]) for f in frames]), np.mean([len(f["gt"]["src"]) for f in frames])
    print(f"written {os.path.relpath(path, ROOT)}: {len(frames)} frames, candidates per frame {n_c:.1f}, GT links per frame {n_g:.1f}, GT nodes per frame {np.mean([f['g'].sum() for f in frames]):.1f}")
    return path


# ------------------------------------------------------------------------------------------------------------------------------ the surrogate and the fit (CPU)
def flatten(frames):
    """the frames as one set of tensors: tokens (pnone, g) over the valid tokens of every frame, candidates (global ends, p, long, y), and per class the number of GT links that
    are no candidate (their s is 0: full FNs)"""
    off, pn, g, cs, cd, cp, cl, cy, miss = 0, [], [], [], [], [], [], [], np.zeros(2, int)
    for f in frames:
        K = len(f["pnone"])
        pn.append(f["pnone"]); g.append(f["g"].astype(np.float32))
        c = f["cand"]
        cs.append(c["src"].astype(int) + off); cd.append(c["dst"].astype(int) + off); cp.append(c["p"]); cl.append(c["long"].astype(int)); cy.append(c["y"].astype(np.float32))
        have = set(zip(c["src"].tolist(), c["dst"].tolist()))
        for s_, d_, l_ in zip(f["gt"]["src"].tolist(), f["gt"]["dst"].tolist(), f["gt"]["long"].tolist()):
            if (s_, d_) not in have:
                miss[int(l_)] += 1
        valid = f["valid"]
        pn[-1] = np.where(valid, pn[-1], np.float32(2.0))                     # a padding token: p_none 2 (never kept: k ~ 0) and g 0
        off += K
    t = lambda xs, dt=torch.float64: torch.as_tensor(np.concatenate(xs), dtype=dt)  # noqa: E731
    valid = torch.as_tensor(np.concatenate([f["valid"] for f in frames]))
    return dict(pnone=t(pn), g=t(g), valid=valid, src=t(cs, torch.long), dst=t(cd, torch.long), p=t(cp), long=t(cl, torch.long), y=t(cy), miss=torch.as_tensor(miss, dtype=torch.float64))


def soft_terms(D, tau, temp=T, sign=1.0):
    """-> (loss, dict of the soft F1s) for tau = (short, long, keep) (tensors); sign = -1 is a test mutant (the decision flipped: kept below the threshold)"""
    k = torch.sigmoid(sign * (tau[2] - D["pnone"]) / temp) * D["valid"]
    thr = torch.stack([tau[0], tau[1]])[D["long"]]
    s = torch.sigmoid(sign * (D["p"] - thr) / temp) * k[D["src"]] * k[D["dst"]]
    f1 = {}
    for c, name in ((0, "short"), (1, "long")):
        m = (D["long"] == c).double()
        tp, fp = (s * D["y"] * m).sum(), (s * (1 - D["y"]) * m).sum()
        fn = ((1 - s) * D["y"] * m).sum() + D["miss"][c]
        f1[name] = 2 * tp / (2 * tp + fp + fn).clamp_min(1e-9)
    tp, fp, fn = (k * D["g"]).sum(), (k * (1 - D["g"])).sum(), ((1 - k) * D["g"] * D["valid"]).sum()
    f1["keep"] = 2 * tp / (2 * tp + fp + fn).clamp_min(1e-9)
    return 3 - f1["short"] - f1["long"] - f1["keep"], f1


def hard_f1(D, tau):
    """the token-level F1s of the HARD decision at tau (the surrogate at T -> 0): what the decode delivers on the labels"""
    k = (D["pnone"] < tau[2]) & D["valid"]
    thr = torch.as_tensor([tau[0], tau[1]], dtype=torch.float64)[D["long"]]
    s = ((D["p"] >= thr) & k[D["src"]] & k[D["dst"]]).double()
    out = {}
    for c, name in ((0, "short"), (1, "long")):
        m = (D["long"] == c).double()
        tp, fp, fn = (s * D["y"] * m).sum(), (s * (1 - D["y"]) * m).sum(), ((1 - s) * D["y"] * m).sum() + D["miss"][c]
        out[name] = float(2 * tp / max(float(2 * tp + fp + fn), 1e-9))
    kd = k.double()
    tp, fp, fn = (kd * D["g"]).sum(), (kd * (1 - D["g"])).sum(), ((1 - kd) * D["g"] * D["valid"]).sum()
    out["keep"] = float(2 * tp / max(float(2 * tp + fp + fn), 1e-9))
    return out


def fit_taus(D, steps=STEPS, lr=LR, temp=T, sign=1.0, init=(0.5, 0.5, 0.5)):
    """Adam on u (tau = sigmoid(u), u = logit(init) at the start: 0.5 each by default), full batch -> (taus (short, long, keep) of the lowest-loss step: links floored at FLOOR,
    all rounded to 3 decimals, the step, the curve every 20 steps, the raw taus). sign = -1 is a test mutant (the decision's sign flipped)"""
    u = torch.logit(torch.tensor(init, dtype=torch.float64)).requires_grad_(True)
    opt = torch.optim.Adam([u], lr=lr)
    best, curve = (float("inf"), None, 0), []
    for i in range(steps + 1):
        tau = torch.sigmoid(u)
        loss, f1 = soft_terms(D, tau, temp, sign)
        assert torch.isfinite(loss), f"the loss is not finite at step {i} (the card's kill)"
        if float(loss) < best[0]:
            best = (float(loss), tau.detach().clone(), i)
        if i % 20 == 0:
            curve.append(dict(step=i, loss=round(float(loss), 6), tau=[round(float(x), 4) for x in tau], **{k: round(float(v), 5) for k, v in f1.items()}))
        if i < steps:
            opt.zero_grad()
            loss.backward()
            opt.step()
    raw = [float(x) for x in best[1]]
    taus = [round(max(raw[0], FLOOR), 3), round(max(raw[1], FLOOR), 3), round(raw[2], 3)]
    return taus, best[2], curve, raw


def fit(a, runs=None):
    import jtoken_eval as E
    runs = runs or E.RUNS
    path = fit_path(runs, a.run, a.split, a.every)
    assert not os.path.exists(path), f"a stored fit is never overwritten: {path}"
    assert a.split == "train", "the thresholds are fitted on TRAIN only (card E388)"
    with open(label_path(runs, a.run, a.split, a.every), "rb") as fh:
        d = pickle.load(fh)
    assert d["meta"]["seed"] == a.seed and d["meta"]["run"] == a.run
    D = flatten(d["frames"])
    taus, step, curve, raw = fit_taus(D)                                      # THE fit (neutral start 0.5 / 0.5 / 0.5): its taus are the arm's
    assert max(abs(x - 0.5) for x in raw) > 0.01, "no tau moved by more than 0.01 from 0.5 (the card's kill: no gradient reaches them)"
    taus_h, step_h, _, raw_h = fit_taus(D, init=RECORD)                       # the identifiability check: the same fit started from the hand cell
    soft = lambda t: {k: round(float(v), 5) for k, v in soft_terms(D, torch.as_tensor(t, dtype=torch.float64))[1].items()}  # noqa: E731
    obj = lambda h: h["short"] + h["long"] + h["keep"]  # noqa: E731
    tf1 = dict(learned=hard_f1(D, taus), record=hard_f1(D, RECORD), neutral=hard_f1(D, (0.5, 0.5, 0.5)), from_hand_cell=hard_f1(D, taus_h))
    gap = max(abs(x - y) for x, y in zip(taus, taus_h))
    labels = []
    if gap > 0.02:
        labels.append(f"surrogate not identifying (the fits from 0.5 and from the hand cell end {gap:.3f} apart)")
    if obj(tf1["learned"]) < obj(tf1["record"]) - 0.002:
        labels.append(f"the fit loses to the hand cell on its own TRAIN objective (hard sum {obj(tf1['learned']):.4f} < {obj(tf1['record']):.4f})")
    if taus[1] < 0.15:
        labels.append(f"tau_long {taus[1]} < 0.15: the dump's 0.05 floor truncates soft mass near it")
    n_gt = sum(len(f["gt"]["src"]) for f in d["frames"])
    gfr = np.mean(np.concatenate([(f["g"] == f["g_gtid"])[f["valid"]] for f in d["frames"]])) if "g_gtid" in d["frames"][0] else None
    res = dict(run=a.run, seed=a.seed, split=a.split, every=a.every, frames=len(d["frames"]), dump=os.path.relpath(label_path(runs, a.run, a.split, a.every), ROOT),
               config=dict(T=T, lr=LR, steps=STEPS, floor=FLOOR, init=0.5, select="the step of the lowest TRAIN loss", none_gate="off"),
               taus=dict(short=taus[0], long=taus[1], keep=taus[2]), raw=raw, step=step, cell=f"{taus[0]:g},{taus[1]:g},{taus[2]:g},off",
               from_hand_cell=dict(taus=taus_h, raw=raw_h, step=step_h, max_gap=round(gap, 4)), labels=labels,
               gt_links=n_gt, gt_links_without_candidate=dict(short=int(D["miss"][0]), long=int(D["miss"][1]), share=round(float(D["miss"].sum()) / max(n_gt, 1), 4)),
               keep_label_agreement_ytype_vs_gt_id=gfr,
               train_token_f1=tf1, train_soft_f1=dict(learned=soft(taus), record=soft(RECORD)), curve=curve)
    with open(path, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"{a.run}: learned taus short {taus[0]} long {taus[1]} keep {taus[2]} (step {step}; from the hand cell {taus_h}, gap {gap:.3f}); TRAIN hard token F1 learned {tf1['learned']}, "
          f"hand cell {tf1['record']}; GT links without a candidate {res['gt_links_without_candidate']}; labels {labels or 'none'}; cell {res['cell']}; written {os.path.relpath(path, ROOT)}")
    return res


# ------------------------------------------------------------------------------------------------------------------------------ selftest
def _toy(seed=0, n_frames=6, K=40):
    """frames with a known optimum: GT tokens p_none in U(0, 0.45), others U(0.65, 1); short GT pairs p in U(0.55, 1), short non-GT U(0.05, 0.4); long GT U(0.35, 1), long
    non-GT U(0.05, 0.25); every candidate end is a GT token; one GT link per frame without a candidate (a full FN)"""
    rng = np.random.RandomState(seed)
    frames = []
    for _ in range(n_frames):
        valid = np.ones(K, bool)
        valid[-4:] = False
        g = np.zeros(K, bool)
        g[:20] = True
        pnone = np.where(g, rng.uniform(0, 0.45, K), rng.uniform(0.65, 1, K)).astype(np.float32)
        src, dst, p, lg, y = [], [], [], [], []
        for i in range(0, 18):
            for step_, cls, pos in ((1, 0, True), (2, 0, False), (3, 1, True), (4, 1, False)):    # every (src, dst) pair once
                src.append(i); dst.append((i + step_) % 20); lg.append(cls); y.append(pos)
                lo, hi = ((0.55, 1) if cls == 0 else (0.35, 1)) if pos else ((0.05, 0.4) if cls == 0 else (0.05, 0.25))
                p.append(rng.uniform(lo, hi))
        cand = dict(src=np.array(src, np.int16), dst=np.array(dst, np.int16), p=np.array(p, np.float32), long=np.array(lg, bool), y=np.array(y, bool))
        gs = cand["src"][cand["y"]].tolist() + [18]
        gd = cand["dst"][cand["y"]].tolist() + [19]
        gl = cand["long"][cand["y"]].tolist() + [False]
        frames.append(dict(token="t", valid=valid, pnone=pnone, g=g & valid, cand=cand, gt=dict(src=np.array(gs, np.int16), dst=np.array(gd, np.int16), long=np.array(gl, bool))))
    return frames


def selftest():
    import math
    # (1) hand numbers: one frame, two tokens kept, one short candidate (GT) and one GT link without a candidate
    f = dict(token="h", valid=np.array([True, True, True]), pnone=np.array([0.1, 0.2, 0.9], np.float32), g=np.array([True, True, False]),
             cand=dict(src=np.array([0], np.int16), dst=np.array([1], np.int16), p=np.array([0.8], np.float32), long=np.array([False]), y=np.array([True])),
             gt=dict(src=np.array([0, 1], np.int16), dst=np.array([1, 2], np.int16), long=np.array([False, False])))
    D = flatten([f])
    assert D["miss"].tolist() == [1.0, 0.0], D["miss"]
    tau = torch.tensor([0.3, 0.2, 0.5], dtype=torch.float64)
    sig = lambda x: 1 / (1 + math.exp(-x))  # noqa: E731
    k0, k1, k2 = sig((0.5 - 0.1) / T), sig((0.5 - 0.2) / T), sig((0.5 - 0.9) / T)
    s = sig((0.8 - 0.3) / T) * k0 * k1                                       # both ends kept (a mutant without the k product: s = sig(...) alone)
    f1s = 2 * s / (2 * s + 0 + (1 - s) + 1)                                   # TP s, FP 0, FN (1 - s) + the GT link without a candidate
    f1k = 2 * (k0 + k1) / (2 * (k0 + k1) + k2 + (2 - k0 - k1))
    loss, got = soft_terms(D, tau)
    assert abs(float(got["short"]) - f1s) < 1e-9 and abs(float(got["keep"]) - f1k) < 1e-9, (got, f1s, f1k)
    assert float(got["long"]) == 0.0, "no long link and no long GT link: F1 0"
    assert hard_f1(D, (0.3, 0.2, 0.5))["short"] == 2 / 3, "hard: TP 1, FN 1 (the link without a candidate)"
    # the global offsets and padding: frame 2 has its candidate on ITS tokens 0 -> 1 (not frame 1's) and a padding token with p_none 0.1 (never kept)
    f2 = dict(f, valid=np.array([True, True, False]), pnone=np.array([0.9, 0.9, 0.1], np.float32), g=np.array([False, False, False]),
              gt=dict(src=np.array([], np.int16), dst=np.array([], np.int16), long=np.array([], bool)))
    D2 = flatten([f, f2])
    assert D2["src"].tolist() == [0, 3] and D2["dst"].tolist() == [1, 4], (D2["src"], D2["dst"])
    h2 = hard_f1(D2, (0.3, 0.2, 0.5))
    assert h2["short"] == 0.5 and h2["keep"] == 1.0, h2                       # frame 2: its GT link's ends are not kept (p_none 0.9): one more FN (wrong offsets would deliver it on frame 1's kept tokens); its padding token (0.1) is never kept: no FP
    # the both-ends keep: an end above tau_keep cuts the link
    assert hard_f1(D, (0.3, 0.2, 0.15))["short"] == 0.0, "an end not kept (p_none 0.2 >= 0.15): the link is not delivered"
    # (2) the fit on frames with a known optimum: every tau inside its window, hard token F1 1 on the candidates (the missing GT links stay FNs)
    D = flatten(_toy())
    taus, step, curve, raw = fit_taus(D)
    assert 0.4 <= taus[0] <= 0.55 and 0.25 <= taus[1] <= 0.35 and 0.45 <= taus[2] <= 0.65, taus
    h = hard_f1(D, taus)
    assert h["keep"] == 1.0 and h["long"] > 0.95 and h["short"] > 0.95, h
    assert curve[0]["loss"] > curve[-1]["loss"] and step > 0
    th = fit_taus(D, init=RECORD)[0]                                          # the identifiability check: from the hand cell the fit lands in the same windows
    assert 0.4 <= th[0] <= 0.55 and 0.25 <= th[1] <= 0.35 and 0.45 <= th[2] <= 0.65, th
    # (3) mutants: the decision's sign flipped does not find the windows; no floor lets a link tau below the dump's 0.05
    tm = fit_taus(D, sign=-1.0)[0]
    assert not (0.4 <= tm[0] <= 0.55 and 0.25 <= tm[1] <= 0.35), f"a sign-flipped decision must not fit: {tm}"
    allpos = [dict(fr, cand=dict(fr["cand"], y=np.ones_like(fr["cand"]["y"])), gt=dict(src=fr["cand"]["src"], dst=fr["cand"]["dst"], long=fr["cand"]["long"])) for fr in _toy(1)]
    tf = fit_taus(flatten(allpos))[0]
    assert tf[0] >= FLOOR and tf[1] >= FLOOR, f"the floor: {tf}"
    # (4) the label frames from a head output: candidates = p >= keep_p inside the gate, GT links from y between valid tokens, class from the chord
    import jtoken as JK
    torch.manual_seed(0)
    b = JK.synthetic_batch(2, 48, seed=1)
    out = JK.JTokenHead().eval()(b)
    lb = dict(b, token=["a", "b"], label=dict(gt_id=torch.where(b["ytype"] > 0, torch.arange(48)[None].expand(2, 48), -torch.ones(2, 48, dtype=torch.long))))
    fr = label_frames(out, lb, 0.05)
    assert np.array_equal(fr[0]["g"], (b["ytype"][0] > 0).numpy() & fr[0]["valid"]), "the keep label is the loss's own ytype > 0"
    pr = JK.probs(out)
    assert len(fr) == 2 and int(((pr["p"][0] >= 0.05) & pr["cand"][0]).sum()) == len(fr[0]["cand"]["src"])
    gy = (b["y"][0].bool() & out["valid"][0][:, None] & out["valid"][0][None, :])
    assert int(gy.sum()) == len(fr[0]["gt"]["src"]) and np.array_equal(fr[0]["cand"]["y"], b["y"][0].bool().numpy()[fr[0]["cand"]["src"], fr[0]["cand"]["dst"]])
    assert np.array_equal(fr[0]["gt"]["long"], (out["chord"][0] > JK.SHORT).numpy()[fr[0]["gt"]["src"], fr[0]["gt"]["dst"]])
    print(f"jthresh selftest ok: hand numbers (soft and hard F1, the GT link without a candidate = a full FN, both ends kept), the fit finds the known windows {taus} (step {step}), "
          f"mutants (sign flipped: {tm}; the floor holds: {tf}), label frames from a head output (candidates, GT links, chord class)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("dump", "fit"):
        s = sub.add_parser(name)
        s.add_argument("--run", required=True)
        s.add_argument("--seed", type=int, required=True)
        s.add_argument("--split", default="train")
        s.add_argument("--every", type=int, default=3)
        if name == "dump":
            s.add_argument("--max", type=int, default=0)
            s.add_argument("--cpu", action="store_true")
    sub.add_parser("selftest")
    a = ap.parse_args(argv)
    torch.set_num_threads(2)
    return dict(dump=dump, fit=fit, selftest=lambda _a: selftest())[a.cmd](a)


if __name__ == "__main__":
    main()
