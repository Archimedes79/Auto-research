"""E391b (card experiments/E391b_existence-gate-on-the-p6-association): the fit population of E391 corrected (the DELIVERED tokens), PART A an existence gate on the zero-shot E390 P6 association (CPU, no head trained), PART B E391's protocol rerun with the loss on the delivered tokens only.

    wsl bash tools/env/_py14.sh evaluation/mapassoc_exist.py pnone --seed S           # the E373 p_none of every TRAIN frame on the CPU -> runs/e391_assoc/exist_pnone_s<S>.pkl; asserted equal to the stored delivered masks of the holdout frames
    wsl bash tools/env/_py14.sh evaluation/mapassoc_exist.py fit   --seed S           # part A: TRAIN frames outside the holdout, delivered tokens: g0 / g1n / g1 + thresholds -> exist_fit_s<S>.pkl / .json (never overwritten)
    wsl bash tools/env/_py14.sh evaluation/mapassoc_exist.py hold  --seed S           # part A: the TRAIN holdout re-weighted to the VAL composition, the stop rule -> exist_hold_s<S>.json
    wsl bash tools/env/_py14.sh evaluation/mapassoc_exist.py read  --seed S           # part A: ONE VAL read (refused unless g1 passed the stop rule on BOTH seeds) -> exist_read_val_s<S>.json; `outcome` prints the card's outcome
    wsl bash tools/env/_py14.sh evaluation/mapassoc_exist.py outcome
    wsl bash tools/env/_py14.sh evaluation/mapassoc_exist.py train_d --arm P|F1|F2|F3 --seed S               # part B: `mapassoc_probe.train` with `AssocDataD` -> runs/e391b_<arm>d_s<S>/
    wsl bash tools/env/_py14.sh evaluation/mapassoc_exist.py read_d  --arm A --seed S --split holdout|val   # part B: `mapassoc_probe.read` on that model -> runs/e391b_<arm>d_s<S>/card_<split>.json
    wsl bash tools/env/_py14.sh evaluation/mapassoc_exist.py choose_d | verdict_d                             # part B: E391's mechanical choice on the holdout cards; the verdict by E391 5.1 item 4 and what it means for the E391 record
    wsl bash tools/env/_py14.sh evaluation/mapassoc_exist.py selftest

THE FINDING behind it: `lane_mapassoc_data.build` stores the delivered mask `kept` (the champion's p_none < 0.6) only for the holdout frames; for the other TRAIN frames `AssocData` falls back to ALL valid tokens (386 scored tokens per frame, 68.5 % false, against 141 and
23.5 % for the delivered set): E391's arms and gate nulls were fitted on that population. `AssocDataD` takes `kept = valid & (p_none < KEEP_P)` for every frame (the E373 head on the CPU, `exist_pnone_s<S>.pkl`; VAL from the stored dump), and in training ignores the targets of the tokens that are
not delivered (assoc -2, lanes -1, route -1). A frame's scored tokens: the delivered tokens that carry an association label (`assoc >= -1`); kind 0 = a present segment, 1 = its segment missing from the map, 2 = a false token. The gates' target is 'unassigned' (assoc == -1: kind 1 or 2);
an arm assigns the zero-shot P6 argmax to every token whose predicted probability of 'unassigned' is below its threshold. Arms: g0 (logistic regression on the own outputs: sigmoid(e), sigmoid(flags), tok, p_none), g1n (a gradient-boosted tree on the same), g1 (the tree on
those + a summary of the P6 logits + the token-map geometry); p6gate (E391's logistic gate, fitted on the wrong population) and the oracle gate (a perfect existence decision, then the P6 argmax) are references. Lanes: the map's lane class where the assigned segment is a road with a known class, else the no-map head
z's argmax (g arms and the oracle; also for a token assigned to a segment that is not a road or has no known class) / class 1 (p6gate, as E391's nulls). Nothing label-dependent (kind, assoc, gt_id) is a feature. The re-weighting: the per-kind rates of a frame set combined with the kind shares of the VAL dec level of the
same seed (`nulls_val_s<S>.json`, `levels.dec.counts`: VAL LABEL COUNTS are used by `fit` for the threshold and by `hold` for the stop rule, as the card says); it corrects the composition, not the shift of the features (the zero-shot P6 head was itself trained on the fit frames)."""
import argparse
import json
import os
import pickle
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")                                  # a shared machine: the tree fit uses 2 threads, not every core
os.environ.setdefault("MKL_NUM_THREADS", "2")
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
for sub_ in ("training/data", "training/data/prep", "training/grid", "training/train", "evaluation", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub_)))
import lane_mapassoc_data as MA  # noqa: E402
import lane_sdgraph_data as SG  # noqa: E402
import mapassoc_probe as MP  # noqa: E402
import sdgraph_probe as SP  # noqa: E402

WORKERS = 2                                                                    # the loader workers of part A's `collect` (a shared machine); part B trains and reads with E391's own setting (4)
BAR_A = {0: 0.8427, 1: 0.8397}                                                 # E391 section 5.1: assoc_acc at dec on VAL (strongest null p6gate + 0.25 x (1 - null))
BAR_L = {0: 0.9182, 1: 0.9044}                                                 # E391 section 5: lanes_acc (z + 0.25 x (1 - z))
BAR_R = {0: 0.8771, 1: 0.8750}                                                 # E391 section 5: route_f1
BAR_HOLD = {0: dict(assoc_acc=0.8472, lanes_acc=0.9266, route_f1=0.8664), 1: dict(assoc_acc=0.8499, lanes_acc=0.9282, route_f1=0.8553)}     # E391 5.1 item 3: the holdout bars of the choice
FU_MAX, FU_FIT, HOLD_MARGIN, BAND = 0.05, 0.04, 0.02, 0.01                    # the guard, the margin a threshold keeps to it on TRAIN validation scenes, the stop-rule margin over the VAL bar, the band
MIN_COUNTS = dict(tokens=20000, false=3000, absent=2000)                       # E391 section 1.3
GRID = np.round(np.arange(0.02, 0.981, 0.02), 2)                               # the thresholds on the predicted probability of 'unassigned'
N_FIT = 2000
TREE = dict(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, l2_regularization=1.0, early_stopping=False, random_state=0)   # fixed by the card, not tuned
ARMS = ("g0", "g1n", "g1")
GEO_NAMES = ("p_unassigned", "p_top1", "p_top2", "margin", "entropy", "d_min", "cos_nearest", "n_within_2m", "n_within_4m")
FUSION = ("P", "F1", "F2", "F3")


# ------------------------------------------------------------------------------------------------------------------------------ io
def write_json(path, obj):
    """atomic: a .tmp renamed; the final name is never overwritten (the caller asserts it first)"""
    tmp = path + ".tmp"
    json.dump(obj, open(tmp, "w"), indent=1)
    os.replace(tmp, path)


def write_pickle(path, obj):
    tmp = path + ".tmp"
    pickle.dump(obj, open(tmp, "wb"))
    os.replace(tmp, path)


def path(name, seed, ext):
    return os.path.join(MA.CACHE, f"{name}_s{seed}.{ext}")


# ------------------------------------------------------------------------------------------------------------------------------ the delivered mask
def pnone_table(split, seed):
    """{frame index: p_none (K,) float32} of the champion's E373 stage-1 head: TRAIN from the file of `pnone`, VAL from the stored dump (`MA.e373_pnone`)"""
    if split == "train":
        f = path("exist_pnone", seed, "pkl")
        assert os.path.exists(f), f"no {os.path.relpath(f, ROOT)}: run `pnone --seed {seed}` first"
        pn = pickle.load(open(f, "rb"))
        assert len(pn) == len(MA.load("train", seed)["frames"]), f"{os.path.basename(f)} does not cover every TRAIN frame ({len(pn)})"
        return pn
    import lane_ptok_data as LP
    ds = LP.PTokData(split, seed, "e292_a", None, 0.0)
    return MA.e373_pnone(split, seed, list(range(len(ds))), ds)


def pnone(seed, log=print):
    """the p_none of every TRAIN frame by the E373 head on the CPU (about 30 min), saved atomically; the kill: the stored `kept` of a holdout frame must be `valid & p_none < KEEP_P` (at most 0.1 % of its tokens may differ)"""
    out = path("exist_pnone", seed, "pkl")
    assert not os.path.exists(out), f"never overwritten: {out}"
    import lane_ptok_data as LP
    ds = LP.PTokData("train", seed, "e292_a", None, 0.0)
    lab = MA.load("train", seed)
    assert list(ds.tokens) == list(lab["tokens"])
    pn = MA.e373_pnone("train", seed, list(range(len(ds))), ds, log=log)
    assert len(pn) == len(ds) and all(np.isfinite(v).all() for v in pn.values()), "a non-finite p_none"
    bad = tot = 0
    for i, rec in enumerate(lab["frames"]):
        if "kept" in rec:
            valid = np.asarray(ds[i]["valid"], bool)
            bad += int(((valid & (pn[i] < MA.KEEP_P)) != np.asarray(rec["kept"], bool)).sum())
            tot += int(valid.sum())
    assert tot > 0 and bad / tot <= 0.001, f"the p_none pass does not reproduce the stored delivered masks of the holdout: {bad} of {tot} valid tokens differ"
    write_pickle(out, pn)
    log(f"pnone seed {seed}: {len(pn)} frames; the stored holdout masks are reproduced ({bad} of {tot} tokens differ); written {os.path.relpath(out, ROOT)}", flush=True)
    return dict(frames=len(pn), differ=bad, valid_tokens=tot)


class AssocDataD(MP.AssocData):
    """`MP.AssocData` with the DELIVERED mask for every frame: where the labels carry no `kept` (the TRAIN frames outside the holdout), kept = valid & (p_none < KEEP_P); `self.pn` the p_none of every frame. Training (level None): the targets of the tokens that are not
    delivered are ignored (assoc -2, lanes -1, route -1), so the loss sees the population that is scored"""

    def __init__(self, split, seed, frames=None, level=None, need_map=True):
        super().__init__(split, seed, frames=frames, level=level, need_map=need_map)
        self.pn = pnone_table(split, seed)

    def __getitem__(self, j):
        out = super().__getitem__(j)
        i = int(out["frame"])
        if "kept" not in self.lab["frames"][i]:
            out["kept"] = np.asarray(out["valid"], bool) & (self.pn[i] < MA.KEEP_P)
        if self.level is None:
            k = out["kept"]
            out["assoc"] = np.where(k, out["assoc"], -2).astype(out["assoc"].dtype)
            out["lanes"] = np.where(k, out["lanes"], -1).astype(out["lanes"].dtype)
            out["route"] = np.where(k, out["route"], -1).astype(out["route"].dtype)
        return out


# ------------------------------------------------------------------------------------------------------------------------------ features
def feats(b, r, z6, sv):
    """-> (own (K, F1), geo (K, 9)) of row r of the batch: own = sigmoid(e), sigmoid(flags), tok (E391's gate inputs; p_none is appended by `collect`); geo = the P6 summary (softmax of the zero-shot logits over the present segments and the 'unassigned' slot: its
    probability, the largest and the second segment probability, their margin, the entropy) and the token-map geometry (the smallest distance to a PRESENT segment in m, the cos of the heading difference there, the number of present segments within 2 m / 4 m)"""
    own = MP.gate_x(b, r)
    P = torch.softmax(z6[r].float(), -1).numpy()
    seg = np.sort(P[:, :-1], axis=1) if P.shape[1] > 1 else np.zeros((len(P), 1))
    top1 = seg[:, -1]
    top2 = seg[:, -2] if seg.shape[1] > 1 else np.zeros(len(P))
    ent = -(P * np.log(P + 1e-12)).sum(1)
    pa = b["pa"][r].numpy()
    present = sv[r].numpy().astype(bool)
    d = np.where(present[None, :], pa[..., 0] * 10.0, np.inf)
    if d.shape[1]:
        j = d.argmin(1)
        dmin = np.minimum(d.min(1), 50.0)
        cosn = np.where(np.isfinite(d.min(1)), pa[np.arange(len(pa)), j, 1], 0.0)
    else:
        dmin, cosn = np.full(len(pa), 50.0), np.zeros(len(pa))
    geo = np.stack([P[:, -1], top1, top2, top1 - top2, ent, dmin, cosn, (d <= 2.0).sum(1), (d <= 4.0).sum(1)], 1).astype(np.float64)
    return own, geo


def collect(split, seed, level, frames, with_z=False, with_gate=False, log=print):
    """-> a list of per-frame records over all K tokens: k (the frame index), kept (the DELIVERED mask), tg, kd, lt (lanes target), pz (the zero-shot P6 assignment), own (with p_none as the last column), geo, lanes_in, seg_kind, and (when asked) zl (the no-map head z's lane argmax),
    pg (the p6gate probability of a false token)"""
    data = AssocDataD(split, seed, frames=frames, level=level)
    dl = torch.utils.data.DataLoader(data, batch_size=8, shuffle=False, num_workers=WORKERS, collate_fn=MP.collate)
    enc, heads6 = SP.load_models("g7", seed)
    enc, heads6 = enc.eval(), heads6.eval()
    zhead = MP.load_head("z", seed) if with_z else None
    gate = MP.load_gate(seed, log=log) if with_gate else None
    out, t0 = [], time.time()
    with torch.no_grad():
        for b in dl:
            h, sv = MP.encode(enc, b, "cpu")
            z6 = heads6.assoc(h, b["pa"].float(), sv)
            oz = zhead(b) if zhead is not None else None
            N = h.shape[1]
            for r in range(len(b["frame"])):
                k = int(b["frame"][r])
                own, geo = feats(b, r, z6, sv)
                own_pn = np.concatenate([own, data.pn[k].astype(np.float64)[:, None]], 1)
                rec = dict(k=k, kept=b["kept"][r].numpy(), tg=b["assoc"][r].numpy(), kd=b["kind"][r].numpy(), lt=b["lanes"][r].numpy(), pz=MP.p6_assign(z6, r, N),
                           own=own_pn.astype(np.float32), geo=geo.astype(np.float32), lanes_in=b["lanes_in"][r].numpy(), seg_kind=b["seg_kind"][r].numpy())
                if oz is not None:
                    rec["zl"] = oz["lanes"][r].argmax(-1).numpy()
                if gate is not None:
                    rec["pg"] = MP.gate_prob(gate, own)
                out.append(rec)
            if len(out) % 400 < 8:
                log(f"  {split} {level}: {len(out)}/{len(data)} frames, {time.time() - t0:.0f} s", flush=True)
    return out


# ------------------------------------------------------------------------------------------------------------------------------ arms, rates, re-weighting
def scored(rec):
    """the tokens a frame scores: delivered with an association label"""
    return rec["kept"] & (rec["tg"] >= -1)


def X_of(rec, arm, m):
    """the feature rows of the tokens m: g0 and g1n the own outputs (+ p_none), g1 also the P6 summary and the geometry"""
    return rec["own"][m] if arm in ("g0", "g1n") else np.concatenate([rec["own"][m], rec["geo"][m]], 1)


def design(recs, arm):
    """(X, y) over the scored tokens of the records; y = 'unassigned' (assoc == -1)"""
    X, y = [], []
    for rec in recs:
        m = scored(rec)
        X.append(X_of(rec, arm, m))
        y.append(rec["tg"][m] == -1)
    return np.concatenate(X), np.concatenate(y).astype(int)


def make_model(arm):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)) if arm == "g0" else HistGradientBoostingClassifier(**TREE)


def p_unassigned(model, rec, arm):
    """the predicted probability of 'unassigned' for every token of the frame (computed on the delivered ones, 0 elsewhere)"""
    p = np.zeros(len(rec["kept"]))
    k = rec["kept"]
    if k.any():
        p[k] = model.predict_proba(X_of(rec, arm, k))[:, 1]
    return p


def assignment(rec, p, thr):
    """the arm's association: the P6 argmax where the probability of 'unassigned' is below thr, else -1"""
    return np.where(p >= thr, -1, rec["pz"])


def oracle_gate(rec):
    """the reference: a perfect existence decision, then the P6 argmax"""
    return np.where(rec["tg"] == -1, -1, rec["pz"])


def lanes_of(ap, rec, fallback):
    """the lane class through the association: the assigned segment's INPUT class where it is a road with a known class, else `fallback` (the no-map head z for the g arms, the constant 1 for p6gate, as `MP.read_through`)"""
    ok = ap >= 0
    j = np.clip(ap, 0, None)
    lin, sk = rec["lanes_in"], rec["seg_kind"]
    known = ok & (sk[j] == 0) & (lin[j] >= 0) & (lin[j] < SG.N_LANE)
    return np.where(known, lin[j], fallback)


def make_rows(recs, ap_of, fallback_of):
    rows = MP.Rows()
    for rec in recs:
        ap = ap_of(rec)
        m = scored(rec)
        rows.a[rec["k"]] = (ap[m], rec["tg"][m], rec["kd"][m])
        m2 = rec["kept"] & (rec["lt"] >= 0)
        rows.l[rec["k"]] = (lanes_of(ap, rec, fallback_of(rec))[m2], rec["lt"][m2])
    return rows


def kind_rates(rows, keys=None):
    """the per-kind rates of an association over the frames: r0 = a token with a present segment assigned right, r1 / r2 = a missing-segment / a false token left unassigned, fu = a token with a present segment left unassigned (false_unassigned), n0 / n1 / n2 = the counts"""
    sel = [rows.a[k] for k in (rows.a if keys is None else keys) if k in rows.a]
    p, t, kd = (np.concatenate(x) for x in zip(*sel))
    r = {}
    for i, name in ((0, "r0"), (1, "r1"), (2, "r2")):
        m = kd == i
        r[name] = float(((p == t) if i == 0 else (p == -1))[m].mean()) if m.any() else float("nan")
        r["n" + str(i)] = int(m.sum())
    r["fu"] = float((p[kd == 0] == -1).mean()) if (kd == 0).any() else float("nan")
    return r


def reweight(rates, w):
    """assoc_acc of the VAL composition w = (share of kind 0, 1, 2) from the per-kind rates (finite rates asserted)"""
    assert all(np.isfinite(rates[k]) for k in ("r0", "r1", "r2", "fu")), f"a kind without tokens: {rates}"
    return float(w[0] * rates["r0"] + w[1] * rates["r1"] + w[2] * rates["r2"])


def val_weights(seed):
    """the kind shares of the VAL dec level of the seed (E391's nulls file: VAL LABEL COUNTS) -> (w, the counts)"""
    c = json.load(open(os.path.join(MA.CACHE, f"nulls_val_s{seed}.json")))["levels"]["dec"]["counts"]
    n0 = c["tokens"] - c["absent"] - c["false"]
    return (n0 / c["tokens"], c["absent"] / c["tokens"], c["false"] / c["tokens"]), c


def choose_threshold(curve):
    """curve = [(thr, assoc_rw, fu)] on the grid: the threshold with the highest assoc_rw among those with fu <= FU_FIT (the first on a tie), None when there is none (the arm cannot meet the guard); non-finite values are refused"""
    assert all(np.isfinite(a) and np.isfinite(fu) for _, a, fu in curve), f"non-finite point on the curve: {curve}"
    ok = [(a, -i, thr) for i, (thr, a, fu) in enumerate(curve) if fu <= FU_FIT]
    return max(ok)[2] if ok else None


def stop_pass(seed, assoc_rw, fu):
    """the stop rule of card 1.5 for one arm and seed on the holdout: the re-weighted assoc_acc >= the VAL bar + the margin 0.02 AND the re-weighted false_unassigned <= 0.05"""
    return bool(assoc_rw >= BAR_A[seed] + HOLD_MARGIN - 1e-9 and fu <= FU_MAX)


def operating_curve(recs, model, arm, w):
    """[(thr, assoc_rw, fu, r2)] of an arm over the grid on the records (the re-weighted numbers)"""
    ps = [p_unassigned(model, rec, arm) for rec in recs]
    out = []
    for thr in GRID:
        rows = MP.Rows()
        for rec, p in zip(recs, ps):
            m = scored(rec)
            rows.a[rec["k"]] = (assignment(rec, p, thr)[m], rec["tg"][m], rec["kd"][m])
        r = kind_rates(rows)
        out.append((float(thr), reweight(r, w), r["fu"], r["r2"]))
    return out


def fit_split(scene, holdout, frames, seed, n=N_FIT):
    """the fit frames: n frames outside the holdout scenes drawn by the generator (3913, seed), a quarter of their SCENES held out for the threshold -> (fit frames, validation frames); asserted: the holdout scenes are not in the draw, the two sets share no scene"""
    ho = set(holdout)
    rng = np.random.default_rng([3913, seed])
    pick = sorted(rng.choice(frames, min(n, len(frames)), replace=False).tolist())
    assert all(scene[k] not in ho for k in pick), "a TRAIN frame of the holdout is in the fit"
    scenes = sorted({int(scene[k]) for k in pick})
    vs = {int(s) for s in rng.choice(scenes, max(1, len(scenes) // 4), replace=False)}
    fit_fr, val_fr = [k for k in pick if int(scene[k]) not in vs], [k for k in pick if int(scene[k]) in vs]
    assert not ({int(scene[k]) for k in fit_fr} & {int(scene[k]) for k in val_fr}) and fit_fr and val_fr, "the validation scenes overlap the fit scenes"
    return fit_fr, val_fr


def shares(recs):
    """the kind shares (0, 1, 2) and the scored tokens per frame of the records"""
    kd = np.concatenate([rec["kd"][scored(rec)] for rec in recs])
    return [float((kd == i).mean()) for i in (0, 1, 2)], float(len(kd) / max(len(recs), 1))


# ------------------------------------------------------------------------------------------------------------------------------ part A: commands
def fit(seed, log=print):
    out_pkl, out_json = path("exist_fit", seed, "pkl"), path("exist_fit", seed, "json")
    assert not os.path.exists(out_pkl) and not os.path.exists(out_json), f"never overwritten: {out_pkl}"
    lab = MA.load("train", seed)
    fit_fr, val_fr = fit_split(lab["scene"], lab["holdout"], MP.train_frames(seed), seed)
    log(f"fit: {len(fit_fr) + len(val_fr)} TRAIN frames outside the holdout ({len(fit_fr)} fitted, {len(val_fr)} validation)", flush=True)
    recs = collect("train", seed, "dec", fit_fr + val_fr, log=log)
    by = {rec["k"]: rec for rec in recs}
    w, counts = val_weights(seed)
    sh_fit, tpf = shares([by[k] for k in fit_fr])
    log(f"the fit population (delivered): {tpf:.1f} scored tokens per frame, kind shares {np.round(sh_fit, 3).tolist()} (VAL {np.round(w, 3).tolist()})", flush=True)
    models, thr = {}, {}
    info = dict(seed=seed, fit_frames=len(fit_fr), val_frames=len(val_fr), weights=list(w), val_counts=counts, fit_shares=sh_fit, fit_tokens_per_frame=tpf, tree=TREE, arms={})
    for arm in ARMS:
        X, y = design([by[k] for k in fit_fr], arm)
        assert len(y) >= MIN_COUNTS["tokens"] and int(y.sum()) >= MIN_COUNTS["false"], f"{arm}: too few delivered TRAIN tokens for the classifier ({len(y)}, unassigned {int(y.sum())})"
        models[arm] = make_model(arm).fit(X, y)
        curve = operating_curve([by[k] for k in val_fr], models[arm], arm, w)
        thr[arm] = choose_threshold([(t, a, fu) for t, a, fu, _ in curve])
        at = {c[0]: c for c in curve}
        info["arms"][arm] = dict(tokens=int(len(y)), unassigned=int(y.sum()), threshold=thr[arm], curve=[[round(v, 4) for v in c] for c in curve])
        log(f"{arm}: {len(y)} tokens ({int(y.sum())} unassigned); threshold {thr[arm]}" + (f", validation assoc_rw {at[thr[arm]][1]:.4f}, fu {at[thr[arm]][2]:.4f}" if thr[arm] is not None else " (cannot meet the guard)"), flush=True)
    write_pickle(out_pkl, dict(models=models, thr=thr, info=info))
    write_json(out_json, info)
    log("written", os.path.relpath(out_pkl, ROOT))
    return info


def check_p6gate_reproduces(rows, seed, split, scene_keys, log=print):
    """the pipeline check: E391's p6gate through THIS code reproduces the stored nulls file (assoc_acc, false_unassigned, lanes_acc at dec, unweighted) to 1e-6"""
    f = os.path.join(MA.CACHE, f"nulls_gate_{split}_s{seed}.json")
    st = json.load(open(f))["levels"]["dec"]
    for m in ("assoc_acc", "false_unassigned", "lanes_acc"):
        mine, ref = rows.metric(m, scene_keys), st[f"{m}_p6gate"][0]
        assert abs(mine - ref) < 1e-6, f"the pipeline does not reproduce E391's p6gate {m} on {split} seed {seed}: {mine} against the stored {ref}"
    log(f"pipeline check: p6gate reproduces {os.path.basename(f)} (assoc_acc, false_unassigned, lanes_acc to 1e-6)", flush=True)


def hold(seed, log=print):
    out = path("exist_hold", seed, "json")
    assert not os.path.exists(out), f"never overwritten: {out}"
    fitd = pickle.load(open(path("exist_fit", seed, "pkl"), "rb"))
    frames, scene = MP.eval_frames("holdout", seed)
    recs = collect("train", seed, "dec", frames, with_z=True, with_gate=True, log=log)
    w, counts = val_weights(seed)
    gate = MP.load_gate(seed, log=log)
    res = dict(seed=seed, frames=len(recs), weights=list(w), val_counts=counts, bar=BAR_A[seed], margin=HOLD_MARGIN, arms={})
    p6g = make_rows(recs, lambda rec: np.where(rec["pg"] >= gate["thr"], -1, rec["pz"]), lambda rec: 1)
    check_p6gate_reproduces(p6g, seed, "holdout", sorted(p6g.a), log)
    r = kind_rates(p6g)
    res["arms"]["p6gate"] = dict(rates=r, assoc_rw=reweight(r, w), fu=r["fu"], threshold=gate["thr"], reference=True)
    ro = kind_rates(make_rows(recs, oracle_gate, lambda rec: rec["zl"]))
    res["arms"]["oracle"] = dict(rates=ro, assoc_rw=reweight(ro, w), fu=ro["fu"], reference=True)
    for arm in ARMS:
        thr = fitd["thr"][arm]
        model = fitd["models"][arm]
        if thr is None:
            res["arms"][arm] = dict(threshold=None, stop=False, why="the threshold search found no value with false_unassigned <= 0.04 on the TRAIN validation scenes", curve=[[round(v, 4) for v in c] for c in operating_curve(recs, model, arm, w)])
            continue
        ps = {rec["k"]: p_unassigned(model, rec, arm) for rec in recs}
        rows = make_rows(recs, lambda rec: assignment(rec, ps[rec["k"]], thr), lambda rec: rec["zl"])
        r = kind_rates(rows)
        a = reweight(r, w)
        res["arms"][arm] = dict(threshold=thr, rates=r, assoc_rw=a, fu=r["fu"], stop=stop_pass(seed, a, r["fu"]), lanes_acc=rows.metric("lanes_acc", sorted(rows.l)),
                                curve=[[round(v, 4) for v in c] for c in operating_curve(recs, model, arm, w)])
    write_json(out, res)
    for name, d in res["arms"].items():
        if d.get("rates"):
            log(f"seed {seed} {name}: re-weighted assoc_acc {d['assoc_rw']:.4f} (stop bar {BAR_A[seed] + HOLD_MARGIN:.4f}), false_unassigned {d['fu']:.4f} (<= {FU_MAX}), per kind r0 {d['rates']['r0']:.3f} r1 {d['rates']['r1']:.3f} r2 {d['rates']['r2']:.3f}, STOP RULE "
                + ("reference" if d.get("reference") else "PASS" if d["stop"] else "FAIL"), flush=True)
        else:
            log(f"seed {seed} {name}: {d['why']}", flush=True)
    log("written", os.path.relpath(out, ROOT))
    return res


def passed_arms(cache=None):
    """the arms that passed the stop rule on BOTH seeds (refuses when a hold file is missing)"""
    ok = {a: True for a in ARMS}
    for s in (0, 1):
        f = os.path.join(cache or MA.CACHE, f"exist_hold_s{s}.json")
        assert os.path.exists(f), f"no hold file of seed {s}: the stop rule is not read, no VAL read"
        h = json.load(open(f))
        for a in ARMS:
            ok[a] &= bool(h["arms"][a].get("stop", False))
    return [a for a in ARMS if ok[a]]


def clauses(seed, m, paired, fallback=None):
    """clause A of card 1.5 for g1 on one seed from the VAL metrics m[arm][metric] = (v, lo, hi) and the paired intervals: assoc_acc >= the bar, the paired interval against g0 AND g1n above 0, false_unassigned <= 0.05;
    also whether the band is cleared (>= bar + 0.01) and whether a no-map arm (g0 or g1n) meets bar and guard on its own"""
    g = m["g1"]
    A = bool(g["assoc_acc"][0] >= BAR_A[seed] and paired["assoc_acc_g1_minus_g0"][1] > 0 and paired["assoc_acc_g1_minus_g1n"][1] > 0 and g["false_unassigned"][0] <= FU_MAX)
    nomap = {a: bool(m[a]["assoc_acc"][0] >= BAR_A[seed] and m[a]["false_unassigned"][0] <= FU_MAX and not (fallback or {}).get(a, False)) for a in ("g0", "g1n")}
    return dict(A=A, band=bool(g["assoc_acc"][0] >= BAR_A[seed] + BAND), nomap=nomap)


def outcome_of(cl):
    """the outcome of card 1.5 from the two seeds' clauses (None = no VAL read: g1 failed the stop rule): CLOSED / NO-MAP GATE SUFFICES / NO FUSION / INCONCLUSIVE / GATE WORKS, in this order"""
    if cl is None:
        return "CLOSED"
    if any(all(cl[s]["nomap"][a] for s in (0, 1)) for a in ("g0", "g1n")):
        return "NO-MAP GATE SUFFICES"
    if not cl[0]["A"] and not cl[1]["A"]:
        return "NO FUSION"
    if cl[0]["A"] != cl[1]["A"] or not (cl[0]["band"] and cl[1]["band"]):
        return "INCONCLUSIVE"
    return "GATE WORKS"


def closed_case(arms):
    """part A closes when g1 fails the stop rule on either seed: 'not separable at the bar' when no arm passed on both seeds, else the case that a no-map arm passed (g1's P6 / geometry inputs are in-sample at fit)"""
    if not arms:
        return "CLOSED: not shown separable at the bar with these inputs"
    return "CLOSED: g1 failed the stop rule while " + " and ".join(arms) + " passed on both seeds (the separation exists without the map inputs)"


def read(seed, log=print, nulls_only=False):
    """part A's ONE VAL read (refused unless g1 passed the stop rule on both seeds) -> exist_read_val_s<S>.json; nulls_only (`nullread`): the VAL read of the corrected no-map null g0 (with p6gate, the oracle gate and z) WITHOUT the stop-rule gate, for part B's record text -> exist_null_val_s<S>.json"""
    out = path("exist_null_val" if nulls_only else "exist_read_val", seed, "json")
    assert not os.path.exists(out), f"never overwritten: {out}"
    if nulls_only:
        arms, run_arms = [], ("g0",)
    else:
        arms, run_arms = passed_arms(), ARMS
        assert "g1" in arms, f"the stop rule: g1 did not pass on both seeds (passed: {arms or 'none'}): E391b part A closes, no VAL read"
    fitd = pickle.load(open(path("exist_fit", seed, "pkl"), "rb"))
    gate = MP.load_gate(seed, log=log)
    recs = collect("val", seed, "dec", None, with_z=True, with_gate=True, log=log)
    scene = MA.load("val", seed)["scene"]
    cand = {"p6gate": (lambda rec: np.where(rec["pg"] >= gate["thr"], -1, rec["pz"]), lambda rec: 1)}
    fallback = {}
    for arm in run_arms:
        ps = {rec["k"]: p_unassigned(fitd["models"][arm], rec, arm) for rec in recs}
        thr = fitd["thr"][arm]
        assert thr is not None or arm != "g1", "g1 has no threshold"
        fallback[arm] = thr is None
        thr = 0.5 if thr is None else thr                                       # a null that cannot meet the guard on the TRAIN validation scenes is read at 0.5 and flagged (and is never 'a no-map arm that meets the bar')
        cand[arm] = (lambda rec, ps=ps, thr=thr: assignment(rec, ps[rec["k"]], thr), lambda rec: rec["zl"])
    cand["oracle"] = (oracle_gate, lambda rec: rec["zl"])
    rows = {n: make_rows(recs, a, f) for n, (a, f) in cand.items()}
    rows["z"] = make_rows(recs, lambda rec: np.full(len(rec["kept"]), -1), lambda rec: rec["zl"])
    keys = sorted(rows["p6gate"].a)
    check_p6gate_reproduces(rows["p6gate"], seed, "val", keys, log)
    zst = json.load(open(os.path.join(ROOT, "runs", f"e391_z_s{seed}", "card_val.json")))["levels"]["dec"]["lanes_acc"][0]
    assert abs(rows["z"].metric("lanes_acc", keys) - zst) < 1e-6, f"z lanes_acc through this code {rows['z'].metric('lanes_acc', keys)} is not the stored {zst}"
    cnt = dict(tokens=sum(len(v[1]) for v in rows["p6gate"].a.values()), false=sum(int((v[2] == 2).sum()) for v in rows["p6gate"].a.values()), absent=sum(int((v[2] == 1).sum()) for v in rows["p6gate"].a.values()))
    assert all(cnt[k] >= MIN_COUNTS[k] for k in MIN_COUNTS), f"minimum counts missed: {cnt}"
    assert cnt == {k: val_weights(seed)[1][k] for k in cnt}, f"the VAL counts differ from E391's nulls file: {cnt}"
    res = dict(seed=seed, arms=arms, counts=cnt, metrics={}, paired={}, threshold_fallback=fallback)
    for n, rw in rows.items():
        if n == "z":
            res["metrics"][n] = {"lanes_acc": MP.boot(scene, keys, lambda fr, rr=rw: rr.metric("lanes_acc", fr))}
            continue
        res["metrics"][n] = {m: MP.boot(scene, keys, lambda fr, rr=rw, mm=m: rr.metric(mm, fr)) for m in ("assoc_acc", "nonfalse_acc", "seg_acc", "none_recall", "false_assign", "false_unassigned", "lanes_acc")}
    pairs = (("g0", "p6gate"),) if nulls_only else (("g1", "g0"), ("g1", "g1n"), ("g1", "p6gate"), ("g1n", "g0"), ("g0", "p6gate"), ("g1", "oracle"))
    for n, ref in pairs:
        res["paired"][f"assoc_acc_{n}_minus_{ref}"] = MP.boot(scene, keys, lambda fr, a_=rows[n], b_=rows[ref]: a_.metric("assoc_acc", fr) - b_.metric("assoc_acc", fr))
    if not nulls_only:
        res["clauses"] = clauses(seed, res["metrics"], res["paired"], fallback)
    write_json(out, res)
    M = res["metrics"]
    fmt = lambda n: f"{n} assoc_acc {M[n]['assoc_acc'][0]:.4f} [{M[n]['assoc_acc'][1]:.4f}, {M[n]['assoc_acc'][2]:.4f}], false_unassigned {M[n]['false_unassigned'][0]:.4f}, lanes_acc {M[n]['lanes_acc'][0]:.4f}"  # noqa: E731
    log(f"seed {seed} VAL dec ({'null read' if nulls_only else 'read'}; bar {BAR_A[seed]}, guard {FU_MAX}): " + "; ".join(fmt(n) for n in ("g0", "g1n", "g1", "p6gate", "oracle") if n in M) + f"; z lanes_acc {M['z']['lanes_acc'][0]:.4f}"
        + ("" if nulls_only else f"; paired g1 - g0 {res['paired']['assoc_acc_g1_minus_g0']}, g1 - g1n {res['paired']['assoc_acc_g1_minus_g1n']}; clauses {res['clauses']}"), flush=True)
    log("written", os.path.relpath(out, ROOT))
    return res


def outcome(log=print):
    """the outcome of part A from the files: CLOSED when a hold file says g1 failed, else from the two VAL reads"""
    try:
        arms = passed_arms()
    except AssertionError as e:
        log(f"OUTCOME not readable: {e}")
        return None
    if "g1" not in arms:
        o = closed_case(arms)
        log(f"OUTCOME {o}")
        return o
    cl = {s: json.load(open(path("exist_read_val", s, "json")))["clauses"] for s in (0, 1)}
    o = outcome_of(cl)
    log(f"OUTCOME {o}: clauses {cl}")
    return o


# ------------------------------------------------------------------------------------------------------------------------------ part B: E391's protocol with the loss on the delivered tokens
def runs_dir_d(arm, seed):
    return os.path.join(ROOT, "runs", f"e391b_{arm}d_s{seed}")


class patched:
    """a context in which `mapassoc_probe` builds `AssocDataD` and puts a fusion arm's folder under runs/e391b_<arm>d_s<S>"""

    def __enter__(self):
        self.old = (MP.AssocData, MP.runs_dir)
        MP.AssocData, MP.runs_dir = AssocDataD, runs_dir_d

    def __exit__(self, *a):
        MP.AssocData, MP.runs_dir = self.old


def train_d(arm, seed, log=print):
    assert arm in FUSION
    with patched():
        return MP.train(arm, seed, out=runs_dir_d(arm, seed), log=log)


def read_d(arm, seed, split, log=print):
    """E391's `read` on a part B model; the VAL read is refused unless the mechanical choice exists and names this arm (the holdout reads of all eight come first)"""
    assert arm in FUSION and split in ("holdout", "val")
    if split == "val":
        f = os.path.join(MA.CACHE, "exist_choice_d.json")
        assert os.path.exists(f), "no choice on the holdout yet (`choose_d`): no VAL read"
        c = json.load(open(f))
        assert arm in (c["assoc_arm"], c["attr_arm"]), f"{arm} was not chosen on the holdout ({c['assoc_arm']} / {c['attr_arm']}): no VAL read"
    with patched():
        return MP.read(arm, seed, split, log=log)


def card_state(arm, seed, split):
    """'card' when the arm's card of the split exists, 'killed' when its train.json shows E391's kill (learn_ratio None or >= 0.8: `MP.read` refuses such a model: a missing card FAILS every bar), else 'missing' (not trained, not read yet, or crashed)"""
    d = runs_dir_d(arm, seed)
    if os.path.exists(os.path.join(d, f"card_{split}.json")):
        return "card"
    tj = os.path.join(d, "train.json")
    if os.path.exists(tj):
        lr = json.load(open(tj)).get("learn_ratio")
        if lr is None or lr >= 0.8:
            return "killed"
    return "missing"


def hval(arm, seed, m):
    """a holdout card's dec value (a killed arm has no card: NaN, which fails every bar)"""
    f = os.path.join(runs_dir_d(arm, seed), "card_holdout.json")
    return float(json.load(open(f))["levels"]["dec"][m][0]) if os.path.exists(f) else float("nan")


def choose_from(val):
    """E391 5.1 item 3, mechanical, from val(arm, seed, metric): -> dict(assoc_arm, attr_arm, why): the association arm F1 if it beats P on holdout assoc_acc by more than 0.01 on BOTH seeds, else P; the attribute arm the first of P, F1, F2, F3 that passes the holdout association bars (assoc_acc, false_unassigned <= 0.05)
    AND the attribute bars on both seeds; none: the best attribute mean among those passing the association bars; none: the association arm"""
    d = {s: val("F1", s, "assoc_acc") - val("P", s, "assoc_acc") for s in (0, 1)}
    assoc = "F1" if all(d[s] > 0.01 for s in (0, 1)) else "P"
    a_ok = [a for a in FUSION if all(val(a, s, "assoc_acc") >= BAR_HOLD[s]["assoc_acc"] and val(a, s, "false_unassigned") <= FU_MAX for s in (0, 1))]
    passing = [a for a in a_ok if all(val(a, s, m) >= BAR_HOLD[s][m] for s in (0, 1) for m in ("lanes_acc", "route_f1"))]
    mean = {a: sum(val(a, s, m) for s in (0, 1) for m in ("lanes_acc", "route_f1")) / 4 for a in FUSION}
    if passing:
        return dict(assoc_arm=assoc, attr_arm=passing[0], why="the simplest passing the holdout association and attribute bars")
    if a_ok:
        return dict(assoc_arm=assoc, attr_arm=max(a_ok, key=lambda a: mean[a] if mean[a] == mean[a] else -1.0), why="none passes the attribute bars: the best attribute mean among those passing the association bars")
    return dict(assoc_arm=assoc, attr_arm=assoc, why="none passes the holdout association bars: the association arm (one head)")


def choose_d(log=print, allow_missing=False):
    """the mechanical choice on the eight holdout cards; refused while a card is missing (not trained, not read, crashed) unless `allow_missing` (the arm NOT RUN by the time cut: its seed fails every bar); a choice is never overwritten"""
    miss = [f"{a}d_s{s}" for a in FUSION for s in (0, 1) if card_state(a, s, "holdout") == "missing"]
    assert not miss or allow_missing, f"no holdout card (and no kill) of {miss}: the choice is not made on NaNs (--allow-missing only for an arm that was NOT RUN by the time cut)"
    c = choose_from(hval)
    c["table"] = {a: {s: {m: hval(a, s, m) for m in ("assoc_acc", "false_unassigned", "lanes_acc", "route_f1")} for s in (0, 1)} for a in FUSION}
    c["not_run"] = miss
    f = os.path.join(MA.CACHE, "exist_choice_d.json")
    assert not os.path.exists(f), f"never overwritten: {f}"
    write_json(f, c)
    log(" ".join(sorted({c["assoc_arm"], c["attr_arm"]}, key=FUSION.index)))
    return c


def verdict_b(assoc, attr):
    """E391 5.1 item 4 in its order from the VAL clauses of the association arm (assoc[s] = bool A) and of the attribute arm (attr[s] = dict(A, L, R)):
    NO FUSION (the association arm fails A on both seeds) / INCONCLUSIVE (a deciding clause passes on one seed only) / PARTLY (A passes on both seeds, an attribute clause fails on both) / PASS"""
    if not assoc[0] and not assoc[1]:
        return "NO FUSION"
    deciding = [(assoc[0], assoc[1])] + [(attr[0][c], attr[1][c]) for c in ("A", "L", "R")]
    if any(x != y for x, y in deciding):
        return "INCONCLUSIVE"
    return "PASS" if all(x for x, _ in deciding) else "PARTLY"


RECORD = {"PASS": "E391's NO FUSION is REVERSED UNDER E391's BARS (the fit population, not the architecture, was the problem)",
          "PARTLY": "E391's NO FUSION is REVERSED UNDER E391's BARS for the association (it passes clause A on both seeds; the named attribute clause fails)",
          "NO FUSION": "E391's NO FUSION is NOT REVERSED by the delivered-mask loss (it stands as recorded; a residual shift remains: the delivered TRAIN false share 23.5 - 24 % against 33 % on VAL, features in-sample for B' and the E373 head)",
          "INCONCLUSIVE": "E391's NO FUSION stays UNRESOLVED (the recorded verdict with its caveat)"}


def null_sentence(cache=None):
    """card 1.5: the corrected no-map comparator for the record text, from the `nullread` files (g0 on VAL, point estimates)"""
    v = {}
    for sd in (0, 1):
        f = os.path.join(cache or MA.CACHE, f"exist_null_val_s{sd}.json")
        if not os.path.exists(f):
            return "corrected no-map null g0: NOT READ (no nullread file): the record text is incomplete"
        m = json.load(open(f))["metrics"]["g0"]
        v[sd] = (m["assoc_acc"][0], m["false_unassigned"][0])
    if all(v[sd][0] >= BAR_A[sd] and v[sd][1] <= FU_MAX for sd in (0, 1)):
        return f"map effect over the corrected no-map gate: NOT SHOWN (g0 alone meets the association bar and the guard on both seeds: assoc_acc {v[0][0]:.4f} / {v[1][0]:.4f}, false_unassigned {v[0][1]:.4f} / {v[1][1]:.4f})"
    return (f"the corrected no-map gate g0 (VAL, point estimates) reaches assoc_acc {v[0][0]:.4f} / {v[1][0]:.4f} against the bars {BAR_A[0]} / {BAR_A[1]}, false_unassigned {v[0][1]:.4f} / {v[1][1]:.4f} (guard {FU_MAX}): "
            "it does not meet the bar and the guard on both seeds")


def vclauses(arm, seed):
    """clause A (assoc_acc >= bar, the paired interval against p6gate above 0, false_unassigned <= 0.05), L, R of a chosen arm from its VAL card at dec; a missing card (a killed arm) fails everything"""
    st = card_state(arm, seed, "val")
    assert st != "missing", f"no VAL card (and no kill) of {arm}d_s{seed}: the VAL read is not done, no verdict"
    if st == "killed":
        return dict(A=False, L=False, R=False)
    L = json.load(open(os.path.join(runs_dir_d(arm, seed), "card_val.json")))["levels"]["dec"]
    return dict(A=bool(L["assoc_acc"][0] >= BAR_A[seed] and L["assoc_acc_minus_p6gate"][1] > 0 and L["false_unassigned"][0] <= FU_MAX), L=bool(L["lanes_acc"][0] >= BAR_L[seed]), R=bool(L["route_f1"][0] >= BAR_R[seed]))


def verdict_d(log=print):
    c = json.load(open(os.path.join(MA.CACHE, "exist_choice_d.json")))
    a_cl = {s: vclauses(c["assoc_arm"], s) for s in (0, 1)}
    t_cl = {s: vclauses(c["attr_arm"], s) for s in (0, 1)}
    v = verdict_b({s: a_cl[s]["A"] for s in (0, 1)}, t_cl)
    marg = {s: json.load(open(os.path.join(runs_dir_d(c["assoc_arm"], s), "card_val.json")))["levels"]["dec"]["assoc_acc"][0] - BAR_A[s] for s in (0, 1) if card_state(c["assoc_arm"], s, "val") == "card"}
    log(f"PART B VERDICT {v}: association arm {c['assoc_arm']} {a_cl}, attribute arm {c['attr_arm']} {t_cl}; margin of the association arm's assoc_acc over the bar {({s: round(m, 4) for s, m in marg.items()})} (a pass by less than 0.01 is within the band; the VAL interval is about +-0.025); "
        f"for the E391 record: {RECORD[v]}; {null_sentence()}")
    return v


# ------------------------------------------------------------------------------------------------------------------------------ test
def toy_rec(k, n=60, rng=None):
    """a toy frame of n tokens, about 20 % of them NOT delivered, with one perfectly separating feature: the own outputs' first column is 0.9 for an 'unassigned' token and 0.1 else"""
    rng = rng or np.random.default_rng(k)
    kd = rng.choice([0, 1, 2], n, p=[0.6, 0.1, 0.3])
    tg = np.where(kd == 0, rng.integers(0, 3, n), -1)
    own = np.concatenate([np.where(kd >= 1, 0.9, 0.1)[:, None] + 0.01 * rng.normal(size=(n, 1)), rng.normal(size=(n, 3))], 1)
    return dict(k=k, kept=rng.random(n) > 0.2, tg=tg, kd=kd, lt=np.where(kd == 0, 1, -1), pz=np.where(kd == 0, tg, rng.integers(0, 3, n)), own=own, geo=rng.normal(size=(n, 9)), lanes_in=np.array([2, 1, 0]), seg_kind=np.zeros(3, int),
                zl=np.full(n, 1))


def selftest():
    # features by hand: one frame row, K = 3 tokens, N = 3 segments (the third absent)
    b = dict(e=torch.zeros(1, 3), flags=torch.zeros(1, 3, 2), tok=torch.ones(1, 3, 1), pa=torch.zeros(1, 3, 3, 8))
    pa = b["pa"]
    pa[0, :, :, 0] = torch.tensor([[0.1, 0.3, 0.0], [0.5, 0.2, 0.0], [5.0, 5.0, 0.0]])        # distance / 10: segment 2 is absent (zeros) and must not count
    pa[0, :, :, 1] = torch.tensor([[0.9, 0.1, 1.0], [0.2, 0.8, 1.0], [0.0, 0.0, 1.0]])
    sv = torch.tensor([[True, True, False]])
    z6 = torch.tensor([[[2.0, 0.0, float("-inf"), 0.0], [0.0, 3.0, float("-inf"), 0.0], [0.0, 0.0, float("-inf"), 4.0]]])
    own, geo = feats(b, 0, z6, sv)
    assert own.shape == (3, 1 + 2 + 1) and geo.shape == (3, 9)
    P = torch.softmax(z6[0].float(), -1).numpy()
    seg = np.sort(P[:, :-1], axis=1)
    assert np.allclose(geo[:, 0], P[:, -1]) and np.allclose(geo[:, 1], seg[:, -1]) and np.allclose(geo[:, 2], seg[:, -2]) and np.allclose(geo[:, 3], geo[:, 1] - geo[:, 2]) and geo[0, 2] > 0 and np.allclose(geo[:, 4], -(P * np.log(P + 1e-12)).sum(1))
    assert np.allclose(geo[:, 5], [1.0, 2.0, 50.0]) and np.allclose(geo[:, 6], [0.9, 0.8, 0.0]) and np.allclose(geo[:, 7], [1, 1, 0]) and np.allclose(geo[:, 8], [2, 1, 0]), geo[:, 5:]
    pa[0, 2, :, 0] = 5.5                                                                         # a token whose nearest present segment is farther than 50 m is clipped; a map with no present segment gives 50 / 0 / 0
    assert feats(b, 0, z6, torch.tensor([[False, False, False]]))[1][2, 5:].tolist() == [50.0, 0.0, 0.0, 0.0]
    d4 = dict(b, pa=pa.clone())
    d4["pa"][0, 0, 0, 0] = 0.4                                                                   # exactly 4 m is within 4 m
    assert feats(d4, 0, z6, sv)[1][0, 8] == 2
    # the rates and the re-weighting by hand
    rows = MP.Rows()
    rows.a[0] = (np.array([5, 5, 7, -1, -1, 4, -1, -1, -1, 2]), np.array([5, 5, 6, 3, -1, -1, -1, -1, -1, -1]), np.array([0, 0, 0, 0, 1, 1, 2, 2, 2, 2]))
    r = kind_rates(rows)
    assert (r["n0"], r["n1"], r["n2"]) == (4, 2, 4) and abs(r["r0"] - 0.5) < 1e-9 and abs(r["r1"] - 0.5) < 1e-9 and abs(r["r2"] - 0.75) < 1e-9 and abs(r["fu"] - 0.25) < 1e-9, r
    assert abs(reweight(r, (0.5, 0.1, 0.4)) - 0.6) < 1e-9 and abs(rows.metric("assoc_acc", [0]) - reweight(r, (0.4, 0.2, 0.4))) < 1e-9
    try:
        reweight(dict(r, r2=float("nan")), (0.5, 0.1, 0.4))
        raise SystemExit("a NaN rate was re-weighted")
    except AssertionError:
        pass
    # the VAL shares from a file
    old = MA.CACHE
    with __import__("tempfile").TemporaryDirectory() as tmp:
        MA.CACHE = tmp
        try:
            json.dump(dict(levels=dict(dec=dict(counts=dict(tokens=1000, absent=100, false=300)))), open(os.path.join(tmp, "nulls_val_s0.json"), "w"))
            w, c = val_weights(0)
            assert np.allclose(w, (0.6, 0.1, 0.3)), w
            # passed_arms: both seeds pass -> the arm, one seed fails -> not, a missing file refuses
            for s, stops in ((0, dict(g0=False, g1n=True, g1=True)), (1, dict(g0=False, g1n=False, g1=True))):
                write_json(os.path.join(tmp, f"exist_hold_s{s}.json"), dict(arms={a: dict(stop=v) for a, v in stops.items()}))
            assert passed_arms() == ["g1"], passed_arms()
            os.remove(os.path.join(tmp, "exist_hold_s1.json"))
            try:
                passed_arms()
                raise SystemExit("passed_arms accepted a missing hold file")
            except AssertionError:
                pass
            write_json(os.path.join(tmp, "exist_hold_s1.json"), dict(arms=dict(g0=dict(stop=True), g1n=dict(stop=True), g1=dict(stop=False))))
            assert passed_arms() == ["g1n"], "g1 fails on seed 1: only g1n passes on both seeds (and the read refuses without g1)"
            write_json(os.path.join(tmp, "exist_hold_s1.json"), dict(arms=dict(g0=dict(stop=False), g1n=dict(stop=False), g1=dict(stop=False))))
            assert passed_arms() == []
        finally:
            MA.CACHE = old
    # the threshold search and the stop rule
    assert choose_threshold([(0.1, 0.80, 0.10), (0.2, 0.83, 0.04), (0.3, 0.84, 0.05), (0.4, 0.83, 0.02)]) == 0.2
    assert choose_threshold([(0.1, 0.80, 0.10), (0.2, 0.83, 0.06)]) is None and choose_threshold([(0.1, 0.8, 0.01), (0.2, 0.8, 0.01)]) == 0.1
    try:
        choose_threshold([(0.1, float("nan"), 0.01)])
        raise SystemExit("a NaN point was accepted")
    except AssertionError:
        pass
    assert stop_pass(0, 0.8627, 0.05) and not stop_pass(0, 0.8626, 0.05) and not stop_pass(0, 0.90, 0.0501) and stop_pass(1, 0.8597, 0.0) and not stop_pass(1, 0.8596, 0.0)
    # the arm's association (the threshold edge: p >= thr is unassigned) and the lanes through it
    rec = dict(pz=np.array([0, 1, -1, 2]), lanes_in=np.array([2, -1, 0, 5]), seg_kind=np.array([0, 0, 1, 0]))
    assert assignment(rec, np.array([0.1, 0.5, 0.1, 0.3]), 0.5).tolist() == [0, -1, -1, 2]
    ap = np.array([0, 1, -1, 3])
    assert lanes_of(ap, rec, 1).tolist() == [2, 1, 1, 1] and lanes_of(ap, rec, np.array([7, 8, 9, 6])).tolist() == [2, 8, 9, 6], "an unknown class (-1), a class beyond N_LANE, an unassigned token: the fall-back"
    assert oracle_gate(dict(tg=np.array([-1, 2, -1]), pz=np.array([1, 2, 0]))).tolist() == [-1, 2, -1]
    assert lanes_of(np.array([0, 2]), rec, 1).tolist() == [2, 1] and lanes_of(np.array([0, 2]), rec, np.array([7, 9])).tolist() == [2, 9], "a token assigned to a segment that is not a road: the fall-back"
    # the toy pipeline: a perfectly separating feature through design, the classifiers, p_unassigned, the curve and the threshold
    recs = [toy_rec(k) for k in range(40)]
    for arm in ARMS:
        X, y = design(recs, arm)
        assert X.shape[1] == (4 if arm != "g1" else 13) and abs(y.mean() - np.mean(np.concatenate([r["tg"][r["kept"]] == -1 for r in recs]))) < 1e-9 and y.sum() > 0 and (y == 0).sum() > 0, "the target is 'unassigned' (kind 1 and 2) over the DELIVERED tokens"
        assert len(y) == sum(int(r["kept"].sum()) for r in recs) < sum(len(r["kept"]) for r in recs), "the undelivered tokens are not in the design"
        mdl = make_model(arm).fit(X, y)
        assert (mdl.predict_proba(X)[:, 1] > 0.5).astype(int).tolist() == y.tolist(), f"{arm} does not separate the toy"
        for rec in recs[:2]:
            p = p_unassigned(mdl, rec, arm)
            assert (p[(rec["kd"] >= 1) & rec["kept"]] > 0.5).all() and (p[(rec["kd"] == 0) & rec["kept"]] < 0.5).all() and (p[~rec["kept"]] == 0).all()
        curve = operating_curve(recs, mdl, arm, (0.6, 0.1, 0.3))
        mid = [c for c in curve if 0.3 <= c[0] <= 0.7]
        assert all(abs(c[1] - 1.0) < 1e-9 and c[2] == 0.0 for c in mid), (arm, mid[:2])                  # a perfect gate: assoc 1.0, no true token dropped
        assert choose_threshold([(t, a, fu) for t, a, fu, _ in curve]) is not None
    assert make_model("g0").__class__.__name__ == "Pipeline" and make_model("g1").__class__.__name__ == "HistGradientBoostingClassifier" and make_model("g1").get_params()["random_state"] == 0
    ro = make_rows(recs, oracle_gate, lambda rec: rec["zl"])
    assert abs(reweight(kind_rates(ro), (0.6, 0.1, 0.3)) - 1.0) < 1e-9 and abs(ro.metric("assoc_acc", sorted(ro.a)) - 1.0) < 1e-9, "the oracle gate: a perfect existence decision, then the P6 argmax"
    kr = kind_rates(ro)
    assert kr["n0"] + kr["n1"] + kr["n2"] == sum(int(r["kept"].sum()) for r in recs), "the rates count the delivered tokens only"
    assert all(len(ro.l[r["k"]][0]) == int((r["kept"] & (r["lt"] >= 0)).sum()) for r in recs), "the lanes rows exclude the undelivered tokens"
    sh, tpf = shares(recs)
    kd_all = np.concatenate([r["kd"][r["kept"]] for r in recs])
    assert np.allclose(sh, [(kd_all == i).mean() for i in (0, 1, 2)]) and abs(tpf - len(kd_all) / len(recs)) < 1e-9, "the shares are the delivered shares"
    # the fit split: the holdout is never drawn, the validation scenes are whole scenes, the draw is deterministic
    scene = np.repeat(np.arange(40), 5)
    allf = [i for i in range(200) if scene[i] not in {0, 10, 20, 30}]
    f1, v1 = fit_split(scene, [0, 10, 20, 30], allf, 3, n=100)
    f2, v2 = fit_split(scene, [0, 10, 20, 30], allf, 3, n=100)
    assert (f1, v1) == (f2, v2) and not {int(scene[k]) for k in f1 + v1} & {0, 10, 20, 30} and not {int(scene[k]) for k in f1} & {int(scene[k]) for k in v1} and len(f1) + len(v1) == 100
    try:
        fit_split(scene, [0, 10, 20, 30], allf + [1], 3, n=500)
        raise SystemExit("a holdout frame in the pool was accepted")
    except AssertionError:
        pass
    # the clauses of card 1.5 and the outcome table
    mk = lambda a, fu=0.04: {"assoc_acc": (a, a - 0.02, a + 0.02), "false_unassigned": (fu, 0, 0), "lanes_acc": (0.9, 0, 0)}  # noqa: E731
    pr = lambda d0, d1: {"assoc_acc_g1_minus_g0": (0.02, d0, 0.04), "assoc_acc_g1_minus_g1n": (0.01, d1, 0.03)}  # noqa: E731
    cl = clauses(0, dict(g0=mk(0.80), g1n=mk(0.82), g1=mk(0.86)), pr(0.01, 0.0001))
    assert cl["A"] and cl["band"] and not cl["nomap"]["g0"] and not cl["nomap"]["g1n"]
    assert not clauses(0, dict(g0=mk(0.80), g1n=mk(0.82), g1=mk(0.8426)), pr(0.01, 0.01))["A"], "the bar edge"
    assert clauses(0, dict(g0=mk(0.80), g1n=mk(0.82), g1=mk(0.8427)), pr(0.01, 0.01))["A"] and not clauses(0, dict(g0=mk(0.80), g1n=mk(0.82), g1=mk(0.8427)), pr(0.01, 0.01))["band"]
    assert not clauses(0, dict(g0=mk(0.80), g1n=mk(0.82), g1=mk(0.86)), pr(0.0, 0.01))["A"] and not clauses(0, dict(g0=mk(0.80), g1n=mk(0.82), g1=mk(0.86)), pr(0.01, -0.001))["A"], "an interval lower bound of 0 is not above 0"
    assert not clauses(0, dict(g0=mk(0.80), g1n=mk(0.82), g1=mk(0.86, 0.0501)), pr(0.01, 0.01))["A"] and clauses(0, dict(g0=mk(0.80), g1n=mk(0.86, 0.05), g1=mk(0.86)), pr(0.01, 0.01))["nomap"]["g1n"]
    good = dict(A=True, band=True, nomap=dict(g0=False, g1n=False))
    bad = dict(A=False, band=False, nomap=dict(g0=False, g1n=False))
    nm = dict(good, nomap=dict(g0=True, g1n=False))
    assert outcome_of(None) == "CLOSED" and outcome_of({0: good, 1: good}) == "GATE WORKS" and outcome_of({0: bad, 1: bad}) == "NO FUSION" and outcome_of({0: good, 1: bad}) == "INCONCLUSIVE"
    assert outcome_of({0: good, 1: dict(good, band=False)}) == "INCONCLUSIVE" and outcome_of({0: nm, 1: nm}) == "NO-MAP GATE SUFFICES" and outcome_of({0: nm, 1: dict(good)}) == "GATE WORKS"
    assert outcome_of({0: dict(nm, A=False), 1: dict(nm, A=False)}) == "NO-MAP GATE SUFFICES", "a no-map arm on its own meets the bar on both seeds: it comes before NO FUSION"
    n1 = dict(good, nomap=dict(g0=False, g1n=True))
    assert outcome_of({0: n1, 1: n1}) == "NO-MAP GATE SUFFICES" and outcome_of({0: nm, 1: n1}) == "GATE WORKS", "the SAME no-map arm on both seeds"
    assert not clauses(0, dict(g0=mk(0.86, 0.0501), g1n=mk(0.82), g1=mk(0.86)), pr(0.01, 0.01))["nomap"]["g0"] and clauses(0, dict(g0=mk(0.86, 0.05), g1n=mk(0.82), g1=mk(0.86)), pr(0.01, 0.01))["nomap"]["g0"]
    assert not clauses(0, dict(g0=mk(0.86), g1n=mk(0.82), g1=mk(0.86)), pr(0.01, 0.01), fallback=dict(g0=True, g1n=False))["nomap"]["g0"], "a null read at the fallback threshold is not 'a no-map arm that meets the bar'"
    assert closed_case([]).startswith("CLOSED: not shown separable") and "g0 and g1n passed" in closed_case(["g0", "g1n"]) and "g1n passed" in closed_case(["g1n"])
    # part B: the choice and the verdict of E391 5.1
    tab = {a: {s: dict(assoc_acc=0.80, false_unassigned=0.10, lanes_acc=0.90, route_f1=0.50) for s in (0, 1)} for a in FUSION}
    val = lambda a, s, m: tab[a][s][m]  # noqa: E731
    assert choose_from(val) == dict(assoc_arm="P", attr_arm="P", why="none passes the holdout association bars: the association arm (one head)")
    tab["F1"] = {s: dict(assoc_acc=0.90, false_unassigned=0.04, lanes_acc=0.95, route_f1=0.50) for s in (0, 1)}
    c = choose_from(val)
    assert c["assoc_arm"] == "F1" and c["attr_arm"] == "F1" and c["why"].startswith("none passes the attribute bars")
    tab["F2"] = {s: dict(assoc_acc=0.90, false_unassigned=0.04, lanes_acc=0.95, route_f1=0.90) for s in (0, 1)}
    assert choose_from(val)["attr_arm"] == "F2" and choose_from(val)["why"].startswith("the simplest")
    tab["F1"][1]["assoc_acc"] = 0.80 + 0.005
    assert choose_from(val)["assoc_arm"] == "P", "F1 does not beat P by more than 0.01 on both seeds"
    t2 = {a: {s: dict(assoc_acc=0.70, false_unassigned=0.10, lanes_acc=0.90, route_f1=0.50) for s in (0, 1)} for a in FUSION}
    t2["F1"] = {s: dict(assoc_acc=0.90, false_unassigned=0.04, lanes_acc=0.95, route_f1=0.50) for s in (0, 1)}
    t2["F2"] = {s: dict(assoc_acc=0.90, false_unassigned=0.04, lanes_acc=0.95, route_f1=0.60) for s in (0, 1)}
    c2 = choose_from(lambda a, s, m: t2[a][s][m])
    assert c2["assoc_arm"] == "F1" and c2["attr_arm"] == "F2" and c2["why"].startswith("none passes the attribute bars"), "two arms pass the association bars, none the attribute bars: the higher attribute mean"
    t2["F1"][1]["route_f1"], t2["F2"][1]["route_f1"] = 0.90, 0.90
    t2["F1"][0]["route_f1"], t2["F2"][0]["route_f1"] = 0.90, 0.90
    assert choose_from(lambda a, s, m: t2[a][s][m])["attr_arm"] == "F1", "two arms pass everything: the first in the order P, F1, F2, F3"
    t2["F1"][0]["assoc_acc"] = 0.84
    assert choose_from(lambda a, s, m: t2[a][s][m])["attr_arm"] == "F2", "an arm that passes the association bar on one seed only does not pass"
    assert "CONFIRMED" not in " ".join(RECORD.values()) and "NOT REVERSED" in RECORD["NO FUSION"] and RECORD["PASS"].count("UNDER E391's BARS") == 1 and RECORD["PARTLY"].count("UNDER E391's BARS") == 1, "the card's wording of the record text"
    with __import__("tempfile").TemporaryDirectory() as tmp2:
        for sd, (a, fu) in {0: (0.85, 0.04), 1: (0.84, 0.04)}.items():
            write_json(os.path.join(tmp2, f"exist_null_val_s{sd}.json"), dict(metrics=dict(g0=dict(assoc_acc=(a, 0, 0), false_unassigned=(fu, 0, 0)))))
        assert null_sentence(tmp2).startswith("map effect over the corrected no-map gate: NOT SHOWN"), "g0 meets the bar and the guard on both seeds"
        write_json(os.path.join(tmp2, "exist_null_val_s1.json"), dict(metrics=dict(g0=dict(assoc_acc=(0.80, 0, 0), false_unassigned=(0.04, 0, 0)))))
        assert "does not meet the bar" in null_sentence(tmp2) and "0.8000" in null_sentence(tmp2)
        write_json(os.path.join(tmp2, "exist_null_val_s0.json"), dict(metrics=dict(g0=dict(assoc_acc=(0.85, 0, 0), false_unassigned=(0.0501, 0, 0)))))
        assert "does not meet the bar" in null_sentence(tmp2), "the guard is part of 'meets'"
        os.remove(os.path.join(tmp2, "exist_null_val_s1.json"))
        assert "NOT READ" in null_sentence(tmp2)
    ok = dict(A=True, L=True, R=True)
    assert verdict_b({0: False, 1: False}, {0: ok, 1: ok}) == "NO FUSION" and verdict_b({0: True, 1: False}, {0: ok, 1: ok}) == "INCONCLUSIVE" and verdict_b({0: True, 1: True}, {0: ok, 1: dict(ok, R=False)}) == "INCONCLUSIVE"
    assert verdict_b({0: True, 1: True}, {0: dict(ok, R=False), 1: dict(ok, R=False)}) == "PARTLY" and verdict_b({0: True, 1: True}, {0: ok, 1: ok}) == "PASS"
    assert verdict_b({0: True, 1: True}, {0: dict(ok, A=False, L=False), 1: dict(ok, A=False, L=False)}) == "PARTLY"
    assert verdict_b({0: False, 1: False}, {0: ok, 1: dict(ok, A=False)}) == "NO FUSION", "NO FUSION comes before INCONCLUSIVE"
    # the delivered mask: where the labels carry no kept, valid & p_none < KEEP_P; the holdout's stored mask is kept; the FRAME INDEX (not the list position) keys the labels and the p_none table
    fake = AssocDataD.__new__(AssocDataD)
    fake.lab = dict(frames={5: {}, 3: dict(kept=np.array([True, False]))})
    fake.pn = {5: np.array([0.1, 0.7, 0.59], np.float32), 3: np.array([0.9, 0.9], np.float32)}
    fake.level = None
    fake.frames = [5, 3]
    base = MP.AssocData.__getitem__

    def stub(self, j):
        i = self.frames[j]
        big = i == 5
        return dict(frame=np.int64(i), valid=np.array([True, True, False]) if big else np.array([True, True]), kept=np.array([True, True, False]) if big else np.array([True, False]),
                    assoc=np.array([1, -1, 2]) if big else np.array([1, -1]), lanes=np.array([0, 1, 2]) if big else np.array([0, 1]), route=np.array([1, 0, 1]) if big else np.array([1, 0]))
    try:
        MP.AssocData.__getitem__ = stub
        o5, o3 = fake[0], fake[1]
        assert o5["kept"].tolist() == [True, False, False] and o5["assoc"].tolist() == [1, -2, -2] and o5["lanes"].tolist() == [0, -1, -1] and o5["route"].tolist() == [1, -1, -1], "no stored mask: valid & p_none < 0.6 (the third token is not valid, the second has p_none 0.7); the targets of what is not delivered are ignored"
        assert o3["kept"].tolist() == [True, False] and o3["assoc"].tolist() == [1, -2] and o3["lanes"].tolist() == [0, -1] and o3["route"].tolist() == [1, -1], "the stored mask of a holdout frame is kept"
        fake.level = "dec"
        assert fake[0]["assoc"].tolist() == [1, -1, 2], "an evaluation level keeps the targets"
    finally:
        MP.AssocData.__getitem__ = base
    # part B glue on fake cards: the kinds of clause, a killed arm, a missing card refuses, the choice gate of the VAL read
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        old_rd, old_cache = runs_dir_d, MA.CACHE
        globals()["runs_dir_d"] = lambda a, s: os.path.join(tmp, f"{a}d_s{s}")
        MA.CACHE = tmp
        try:
            def card(arm, seed, split, assoc=0.85, lo=0.01, fu=0.04, lanes=0.93, route=0.88):
                d = runs_dir_d(arm, seed)
                os.makedirs(d, exist_ok=True)
                json.dump(dict(levels=dict(dec=dict(assoc_acc=(assoc, 0, 0), assoc_acc_minus_p6gate=(0.03, lo, 0.05), false_unassigned=(fu, 0, 0), lanes_acc=(lanes, 0, 0), route_f1=(route, 0, 0)))), open(os.path.join(d, f"card_{split}.json"), "w"))
            card("F1", 0, "val")
            assert vclauses("F1", 0) == dict(A=True, L=True, R=True)
            card("F1", 0, "val", assoc=0.8426)
            assert vclauses("F1", 0)["A"] is False, "the association bar (seed 0: 0.8427)"
            card("F1", 0, "val", lo=0.0)
            assert vclauses("F1", 0)["A"] is False, "the paired interval against p6gate must be above 0"
            card("F1", 0, "val", fu=0.0501)
            assert vclauses("F1", 0)["A"] is False, "false_unassigned <= 0.05"
            card("F1", 0, "val", lanes=0.9181, route=0.8770)
            assert vclauses("F1", 0) == dict(A=True, L=False, R=False), "the L bar 0.9182 and the R bar 0.8771 of seed 0"
            card("F1", 1, "val", lanes=0.9044, route=0.8750, assoc=0.8397)
            assert vclauses("F1", 1) == dict(A=True, L=True, R=True), "the seed-1 bars 0.8397 / 0.9044 / 0.8750"
            try:
                vclauses("F2", 0)
                raise SystemExit("a missing VAL card was read as a failure")
            except AssertionError:
                pass
            os.makedirs(runs_dir_d("F2", 0), exist_ok=True)
            json.dump(dict(learn_ratio=0.9), open(os.path.join(runs_dir_d("F2", 0), "train.json"), "w"))
            assert vclauses("F2", 0) == dict(A=False, L=False, R=False) and card_state("F2", 0, "holdout") == "killed" and card_state("F3", 0, "holdout") == "missing"
            json.dump(dict(learn_ratio=0.2), open(os.path.join(runs_dir_d("F2", 0), "train.json"), "w"))
            assert card_state("F2", 0, "holdout") == "missing", "a trained arm with a good learn ratio but no card is missing, not killed"
            try:
                choose_d(log=lambda *a, **k: None)
                raise SystemExit("choose_d ran on missing cards")
            except AssertionError:
                pass
            for arm in FUSION:
                for sd in (0, 1):
                    card(arm, sd, "holdout", assoc=0.80, fu=0.10, lanes=0.90, route=0.50)
            c = choose_d(log=lambda *a, **k: None)
            assert c["assoc_arm"] == "P" and c["not_run"] == []
            try:
                choose_d(log=lambda *a, **k: None)
                raise SystemExit("a choice was overwritten")
            except AssertionError:
                pass
            try:
                read_d("F2", 0, "val")
                raise SystemExit("a VAL read of an arm that was not chosen")
            except AssertionError as e:
                assert "was not chosen" in str(e)
            os.remove(os.path.join(tmp, "exist_choice_d.json"))
            try:
                read_d("P", 0, "val")
                raise SystemExit("a VAL read without a choice")
            except AssertionError as e:
                assert "no choice" in str(e)
        finally:
            globals()["runs_dir_d"], MA.CACHE = old_rd, old_cache
    print("mapassoc_exist selftest ok")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("pnone", "fit", "hold", "read", "nullread", "outcome", "train_d", "read_d", "choose_d", "verdict_d", "selftest"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arm", choices=FUSION)
    ap.add_argument("--split", choices=("holdout", "val"), default="val")
    ap.add_argument("--allow-missing", action="store_true", help="choose_d: an arm that was NOT RUN by the time cut fails every bar")
    a = ap.parse_args(argv)
    torch.set_num_threads(2)
    if a.cmd == "selftest":
        return selftest()
    if a.cmd == "outcome":
        return outcome()
    if a.cmd == "choose_d":
        return choose_d(allow_missing=a.allow_missing)
    if a.cmd == "nullread":
        return read(a.seed, nulls_only=True)
    if a.cmd == "verdict_d":
        return verdict_d()
    if a.cmd in ("train_d", "read_d"):
        assert a.arm, "--arm is required"
        return train_d(a.arm, a.seed) if a.cmd == "train_d" else read_d(a.arm, a.seed, a.split)
    return {"pnone": pnone, "fit": fit, "hold": hold, "read": read}[a.cmd](a.seed)


if __name__ == "__main__":
    main()
