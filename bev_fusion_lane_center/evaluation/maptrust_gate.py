"""E394 v1, arm g (card experiments/E394_*): a LABEL-FREE map-perception consistency gate on E391's fusion head F1, so that a wrong map does not make the lanes per direction worse than no map. CPU.

    wsl bash tools/env/_py14.sh maptrust_gate.py fit --seed S [--n 800] [--out DIR]   # -> runs/e394_gate/fit_s<S>.json (the logistic and its threshold; never overwritten)
    wsl bash tools/env/_py14.sh maptrust_gate.py read --seed S --split holdout|val [--every N] [--levels a,b|card]   # -> runs/e394_gate/read_<split>_s<S>[_every<N>].json (never overwritten)
    wsl bash tools/env/_py14.sh maptrust_gate.py selftest

Per frame, from the delivered tokens (`kept`) and the INPUT map only (no label is read): `assigned_mass` = the mean of 1 - p(unassigned) of F1's association, `max_prob` = the mean of F1's best segment
probability, `p6_rate` = the share of tokens the zero-shot E390 association (p6) gives a segment (the geometric agreement of map and perception), `lane_agree` = among the tokens F1 assigns to a road
segment of known lane class, the share whose lane class read by the NO-MAP head z equals the map's (Laplace (a + 1) / (n + 2)). A logistic on the four standardized features predicts `gate_label(spec)`
= the map would harm the lanes read through it = the map of another scene (see `gate_label`: chosen on TRAIN). Fit: 800 TRAIN frames outside the holdout (seed 3941), each with its own draw of the catalogue mix `train_spec` (seeded by (394, frame, 1)).
Threshold: the TRUST_Q quantile of p(wrong) over the RIGHT-map draws of the fit, so that about 96 % of the draws that are not the map of another scene are trusted on TRAIN (card v1: chosen on TRAIN as the quantile with the most room on both clauses W and F; the clause F needs 95 % on the read).
The gated head g: a trusted frame keeps F1's association and lanes; a distrusted frame falls back to z's lanes and leaves every token unassigned (the card's fall-back, D135).
`read`: the heads a (= F1 ungated), z and g on the same draws as `maperror_probe` (`ErrAssocData`, seeded by (394, frame)), per level: lanes_all of a / z / g with the scene bootstrap (B 1000, the same
resamples for every quantity: paired) and the paired differences g - z, g - a, a - z, the trust rate of g, wrong_assign / assign_rate of a and g; `--levels card` = the two deciding levels (dec, wrong_scene+dec)
and the report-only failure map of the card (section 1.3). Tests: evaluation/tests/test_maptrust_gate.py.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for sub_ in ("training/data", "training/data/prep", "net/prep", "net/model", "net/loss", "training/train", "evaluation", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub_)))
import lane_maperror_data as ME  # noqa: E402
import lane_sdgraph_data as SG  # noqa: E402

OUT = os.path.join(ROOT, "runs", "e394_gate")
FEATS = ("assigned_mass", "max_prob", "p6_rate", "lane_agree")
TRUST_Q = 0.96
FIT_N, FIT_SEED = 800, 3941
CARD_LEVELS = ("exact", "dec", "wrong_scene+dec", "wrong_scene=1", "shift=4.0", "shift=8.0", "yaw=4.0", "pose=4.0", "outdated=0.3", "outdated=1.0", "count_big=0.3", "ghost=0.6",
               "wrong_route=1", "node_drop=0.4")


def gate_label(spec):
    """the frame label of the fit: the map would HARM the lanes read through it = the map of another scene (card v1: chosen on TRAIN; `is_wrong` as the label - a shift or turn of
    4 - 8 m / deg too - gave AUROC 0.79 and rejected 44 % of the wrong maps at 97.5 % trust, while E394 step 0 shows that F1 reads the lanes as well as on the right map under those shifts
    (lanes_all 0.982 at 8 m): only the map of another scene breaks the read among the modes of the TRAIN mix)"""
    return bool(spec.get("wrong_scene"))


def frame_features(assoc_logits, p6_assign, z_lanes, kept, lanes_in, seg_kind):
    """(4,) the label-free features of one frame (module docstring). assoc_logits (K, N + 1) F1's logits (last = unassigned), p6_assign (K,) the zero-shot segment or -1, z_lanes (K,) the no-map
    head's lane class, kept (K,) bool the delivered tokens, lanes_in / seg_kind (N,) the INPUT map's lane class and kind"""
    kept = np.asarray(kept, bool)
    if not kept.any():
        return np.array([0.0, 0.0, 0.0, 0.5])
    p = torch.softmax(torch.as_tensor(assoc_logits, dtype=torch.float64), -1).numpy()[kept]
    assigned = 1.0 - p[:, -1]
    best = p[:, :-1].max(-1) if p.shape[1] > 1 else np.zeros(len(p))
    ap = p.argmax(-1)
    ap = np.where(ap == p.shape[1] - 1, -1, ap)
    j = np.clip(ap, 0, None)
    lin, sk = np.asarray(lanes_in), np.asarray(seg_kind)
    known = (ap >= 0) & (sk[j] == 0) & (lin[j] >= 0) & (lin[j] < SG.N_LANE)
    agree = (np.asarray(z_lanes)[kept] == lin[j]) & known
    return np.array([float(assigned.mean()), float(best.mean()), float((np.asarray(p6_assign)[kept] >= 0).mean()), (float(agree.sum()) + 1.0) / (float(known.sum()) + 2.0)])


def gate_prob(fit, X):
    """p(wrong map) of feature rows X (n, 4) under a fit"""
    Z = (np.asarray(X, float) - np.asarray(fit["mean"])) / np.asarray(fit["std"])
    return 1.0 / (1.0 + np.exp(-(Z @ np.asarray(fit["coef"]) + fit["bias"])))


def fit_logistic(X, y, q=TRUST_Q):
    """the logistic on standardized features and its threshold: the q quantile of p(wrong) over the rows with y = 0 (the right maps)"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    X, y = np.asarray(X, float), np.asarray(y, int)
    assert 0 < y.sum() < len(y), "the fit needs right and wrong maps"
    mu, sd = X.mean(0), X.std(0) + 1e-9
    m = LogisticRegression(max_iter=3000).fit((X - mu) / sd, y)
    fit = dict(feats=list(FEATS), mean=mu.tolist(), std=sd.tolist(), coef=m.coef_[0].tolist(), bias=float(m.intercept_[0]))
    p = gate_prob(fit, X)
    fit.update(thr=float(np.quantile(p[y == 0], q)), trust_q=q, auroc=float(roc_auc_score(y, p)), rows=int(len(y)), wrong_share=float(y.mean()),
               trusted_right=float((p[y == 0] <= np.quantile(p[y == 0], q)).mean()), rejected_wrong=float((p[y == 1] > np.quantile(p[y == 0], q)).mean()))
    return fit


def combine(trusted, ap, l1, zl):
    """the gated head's read of one frame: a trusted frame keeps F1's association and lanes, a distrusted one leaves every token unassigned and takes the no-map head's lanes (the fall-back, D135)"""
    return (ap, l1) if trusted else (np.full_like(ap, -1), zl)


def provenance(MP, seed):
    """the weights the gate reads (repository-relative): E391's F1 and z, the E390 encoder (critic 2026-11-24, should-fix 6)"""
    rel = lambda p: os.path.relpath(p, ROOT).replace(os.sep, "/")  # noqa: E731
    import sdgraph_probe as SP
    return dict(F1=rel(MP.runs_dir("F1", seed)), z=rel(MP.runs_dir("z", seed)), encoder=rel(SP.runs_dir("g7", seed)))         # the folders the loaders read


def _models(seed):
    import mapassoc_probe as MP
    import sdgraph_probe as SP
    enc, heads6 = SP.load_models("g7", seed)
    return MP, enc.eval(), heads6.eval(), MP.load_head("F1", seed), MP.load_head("z", seed)


def _frames_out(MP, enc, heads6, f1, z, b):
    """per row of a batch: (frame, features, F1 assignment, F1 lanes, z lanes)"""
    h, sv = MP.encode(enc, b, "cpu")
    z6 = heads6.assoc(h, b["pa"].float(), sv)
    o1, oz = f1(b, h, sv, b["pa"]), z(b)
    N = h.shape[1]
    out = []
    for r in range(len(b["frame"])):
        ap6 = MP.p6_assign(z6, r, N)
        zl = oz["lanes"][r].argmax(-1).numpy()
        x = frame_features(o1["assoc"][r].numpy(), ap6, zl, b["kept"][r].numpy(), b["lanes_in"][r].numpy(), b["seg_kind"][r].numpy())
        ap = o1["assoc"][r].argmax(-1).numpy()
        ap = np.where(ap == o1["assoc"].shape[-1] - 1, -1, ap)
        out.append((int(b["frame"][r]), x, ap, o1["lanes"][r].argmax(-1).numpy(), zl))
    return out


def fit(seed, n=FIT_N, out=OUT, log=print):
    import maperror_probe as MEP
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"fit_s{seed}.json")
    assert not os.path.exists(path), f"never overwritten: {path}"
    MP, enc, heads6, f1, z = _models(seed)
    fr = MP.train_frames(seed)
    pick = sorted(np.random.default_rng(FIT_SEED).choice(fr, min(n, len(fr)), replace=False).tolist())
    specs = {i: ME.train_spec(np.random.default_rng([394, i, 1])) for i in pick}
    plan = MEP.plan_for("holdout", seed)

    class FitData(MEP.ErrAssocData):
        def __getitem__(self, j):
            self.spec = specs[self.frames[j]]
            return super().__getitem__(j)

    data = FitData("train", seed, {}, frames=pick, plan=plan)
    dl = torch.utils.data.DataLoader(data, batch_size=8, shuffle=False, num_workers=0, collate_fn=MEP.collate)
    X, y, t0 = [], [], time.time()
    with torch.no_grad():
        for b in dl:
            for k, x, *_ in _frames_out(MP, enc, heads6, f1, z, b):
                X.append(x)
                y.append(int(gate_label(specs[k])))
    res = fit_logistic(np.array(X), np.array(y))
    res.update(seed=seed, frames=len(pick), fit_seed=FIT_SEED, heads=provenance(MP, seed), seconds=round(time.time() - t0))
    json.dump(res, open(path + ".tmp", "w"), indent=1)
    os.replace(path + ".tmp", path)
    log(f"fit s{seed}: {len(pick)} TRAIN frames (wrong {res['wrong_share']:.3f}), AUROC {res['auroc']:.4f}, thr {res['thr']:.4f}, trusted right {res['trusted_right']:.3f}, "
        f"rejected wrong {res['rejected_wrong']:.3f}, coef {np.round(res['coef'], 3).tolist()} -> {os.path.relpath(path, ROOT)}", flush=True)
    return res


def read(seed, split, every=1, levels="card", out=OUT, log=print):
    import maperror_probe as MEP
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"read_{split}_s{seed}{f'_every{every}' if every != 1 else ''}.json")
    assert not os.path.exists(path), f"never overwritten: {path}"
    g = json.load(open(os.path.join(out, f"fit_s{seed}.json")))
    MP, enc, heads6, f1, z = _models(seed)
    frames, scene = MP.eval_frames(split, seed)
    frames = (list(range(len(scene))) if frames is None else frames)[:: max(1, every)]
    plan = MEP.plan_for(split, seed)
    names = CARD_LEVELS if levels == "card" else tuple(levels.split(","))
    specs = dict(MEP.spec_list(["all"]))
    assert all(n in specs for n in names), [n for n in names if n not in specs]
    res = dict(seed=seed, split=split, every=every, frames=len(frames), fit=os.path.relpath(os.path.join(out, f"fit_s{seed}.json"), ROOT), thr=g["thr"], heads=provenance(MP, seed), levels={})
    for name in names:
        t0 = time.time()
        data = MEP.ErrAssocData(split if split == "val" else "train", seed, specs[name], frames=frames, plan=plan)
        dl = torch.utils.data.DataLoader(data, batch_size=8, shuffle=False, num_workers=0, collate_fn=MEP.collate)
        rows = {h: MP.Rows() for h in ("a", "g")}
        lall = {h: MEP.Lall() for h in ("a", "z", "g")}
        trust = {}
        with torch.no_grad():
            for b in dl:
                for r, (k, x, ap, l1, zl) in enumerate(_frames_out(MP, enc, heads6, f1, z, b)):
                    ok = bool(gate_prob(g, x[None])[0] <= g["thr"])
                    trust[k] = ok
                    rows["a"].add(k, b, r, ap, l1)
                    ag, lg = combine(ok, ap, l1, zl)
                    rows["g"].add(k, b, r, ag, lg)
                    lall["a"].add(k, b, r, l1)
                    lall["z"].add(k, b, r, zl)
                    lall["g"].add(k, b, r, lg)
        keys = sorted(trust)
        bt = lambda fn: [round(v, 5) for v in MP.boot(scene, keys, fn)]  # noqa: E731   (the same resamples for every quantity: seed 0)
        L = {f"lanes_all_{h}": bt(lambda fr, hh=h: lall[hh].acc(fr)) for h in ("a", "z", "g")}
        for x_, y_ in (("g", "z"), ("g", "a"), ("a", "z")):
            L[f"d_{x_}_{y_}"] = bt(lambda fr, xx=x_, yy=y_: lall[xx].acc(fr) - lall[yy].acc(fr))
        L["trust"] = round(float(np.mean([trust[k] for k in keys])), 4)
        for h in ("a", "g"):
            L.update({f"{m}_{h}": round(v, 5) for m, v in MEP.extra_metrics(rows[h], keys).items()})
            L[f"assoc_acc_{h}"] = round(rows[h].metric("assoc_acc", keys), 5)
        L["tokens"] = int(sum(len(v[1]) for v in lall["z"].d.values()))
        L["frames"] = len(keys)
        res["levels"][name] = L
        log(f"{name}: lanes_all a {L['lanes_all_a'][0]:.4f} z {L['lanes_all_z'][0]:.4f} g {L['lanes_all_g'][0]:.4f} | g - z {L['d_g_z'][0]:+.4f} [{L['d_g_z'][1]:+.4f}, {L['d_g_z'][2]:+.4f}] | "
            f"trust {L['trust']:.3f} | wrong_assign a {L['wrong_assign_a']:.3f} g {L['wrong_assign_g']:.3f} ({time.time() - t0:.0f} s)", flush=True)
        json.dump(res, open(path + ".part", "w"), indent=1)
    os.replace(path + ".part", path)
    log(f"written {os.path.relpath(path, ROOT)}")
    return res


def selftest():
    import test_maptrust_gate as T
    T.main()
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="maptrust_gate", description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("fit", "read", "selftest"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="holdout", choices=("holdout", "val"))
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--levels", default="card")
    ap.add_argument("--n", type=int, default=FIT_N, help="fit: TRAIN frames (a smoke test: fewer)")
    ap.add_argument("--out", default=OUT, help="the output folder (a smoke test: a scratch folder)")
    a = ap.parse_args(argv)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    a.out = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)          # relative to the problem folder (`_py14.sh` runs from the script's folder)
    if a.cmd == "selftest":
        return selftest()
    if a.cmd == "fit":
        fit(a.seed, a.n, a.out)
    else:
        read(a.seed, a.split, a.every, a.levels, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
