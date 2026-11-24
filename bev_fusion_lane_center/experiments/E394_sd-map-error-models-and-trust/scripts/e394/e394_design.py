"""E394 v1.1 design of gate g on TRAIN only (the critic's points 2 and 4 of 2026-11-24): (1) the 800 fit draws of `maptrust_gate.fit` (the catalogue mix), in-sample and 5-fold
cross-validated; (2) 400 OTHER TRAIN frames (outside the holdout and outside the fit draws) read at the card's two deciding specs exactly - `dec` (RIGHT) and `wrong_scene+dec` (WRONG) -
for every threshold quantile of the grid: the trust at RIGHT (clause F), lanes_all of g against z at WRONG (clause W) and of g against a at RIGHT (clause R), from the logistic fitted on the
800 mix draws. No holdout or VAL frame is read (the wrong-scene maps are borrowed by `maperror_probe.plan_for('holdout')`: map INPUTS of other TRAIN scenes, possibly holdout scenes, no labels).
v1.1 (the critic's second look): also (3) the FEATURE-SET BOUND - a logistic fitted directly (in-sample) on the held frames at RIGHT (label 0) and WRONG (label 1): the best clause W at a trust
of at least 0.95 at RIGHT that these four features allow at all; and every held frame's features and counts are stored (`rows_right`, `rows_wrong`). The 400 frames are frame-disjoint from the fit
draws, not scene-disjoint. Writes runs/e394_gate/design_train_s<S>[_<tag>].json (never overwritten). Run from the problem folder (through run.py as `E394.gdesign_s<S>`):
wsl bash tools/env/_py14.sh experiments/E394_sd-map-error-models-and-trust/scripts/e394/e394_design.py [seed] [tag]"""
import json
import os
import sys

import numpy as np
import torch

ROOT = os.environ.get("BFLC_ROOT") or os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import maptrust_gate as G  # noqa: E402

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
TAG = sys.argv[2] if len(sys.argv) > 2 else ""
OUTP = os.path.join(G.OUT, f"design_train_s{SEED}{'_' + TAG if TAG else ''}.json")
assert not os.path.exists(OUTP), f"never overwritten: {OUTP}"
torch.set_num_threads(2)
MEP = __import__("maperror_probe")
MP, enc, heads6, f1, z = G._models(SEED)
fr = MP.train_frames(SEED)
fit_pick = sorted(np.random.default_rng(G.FIT_SEED).choice(fr, min(G.FIT_N, len(fr)), replace=False).tolist())
rest = sorted(set(fr) - set(fit_pick))
held = sorted(np.random.default_rng(3942).choice(rest, min(400, len(rest)), replace=False).tolist())
plan = MEP.plan_for("holdout", SEED)
specs = dict(MEP.spec_list(["all"]))


def rows_for(frames, spec_of):
    class D(MEP.ErrAssocData):
        def __getitem__(self, j):
            self.spec = spec_of(self.frames[j])
            return super().__getitem__(j)
    dl = torch.utils.data.DataLoader(D("train", SEED, {}, frames=frames, plan=plan), batch_size=8, shuffle=False, num_workers=0, collate_fn=MEP.collate)
    out = []
    with torch.no_grad():
        for b in dl:
            for r, (k, x, ap, l1, zl) in enumerate(G._frames_out(MP, enc, heads6, f1, z, b)):
                la = b["lanes_all"][r].numpy()
                m = b["kept"][r].numpy().astype(bool) & (la >= 0)
                out.append(dict(k=k, x=x.tolist(), n=int(m.sum()), c1=int((l1[m] == la[m]).sum()), cz=int((zl[m] == la[m]).sum())))
    return out


mix_spec = {i: G.ME.train_spec(np.random.default_rng([394, i, 1])) for i in fit_pick}
mix = rows_for(fit_pick, lambda i: mix_spec[i])
right = rows_for(held, lambda i: specs["dec"])
wrong = rows_for(held, lambda i: specs["wrong_scene+dec"])
X = np.array([r["x"] for r in mix])
y = np.array([int(G.gate_label(mix_spec[r["k"]])) for r in mix])
fit = G.fit_logistic(X, y)
pm = G.gate_prob(fit, X)
XR, XW = np.array([r["x"] for r in right]), np.array([r["x"] for r in wrong])
pr, pw = G.gate_prob(fit, XR), G.gate_prob(fit, XW)
acc = lambda rows, ok, use1: sum((r["c1"] if (o if use1 else False) else r["cz"]) for r, o in zip(rows, ok)) / max(sum(r["n"] for r in rows), 1)  # noqa: E731
lz_r = sum(r["cz"] for r in right) / sum(r["n"] for r in right)
la_r = sum(r["c1"] for r in right) / sum(r["n"] for r in right)
lz_w = sum(r["cz"] for r in wrong) / sum(r["n"] for r in wrong)
la_w = sum(r["c1"] for r in wrong) / sum(r["n"] for r in wrong)
grid = []
for q in (0.90, 0.925, 0.95, 0.96, 0.975, 0.99, 0.995, 1.0):
    thr = float(np.quantile(pm[y == 0], q))
    okr, okw = pr <= thr, pw <= thr
    grid.append(dict(q=q, thr=thr, trust_right=float(okr.mean()), trust_wrong=float(okw.mean()),
                     g_right=acc(right, okr, True), g_wrong=acc(wrong, okw, True), g_minus_z_wrong=acc(wrong, okw, True) - lz_w, g_minus_a_right=acc(right, okr, True) - la_r))
# 5-fold cross-validation on the mix draws (the in-sample check of the fit itself): the wrong-scene draws' g - z and the rejection at q = TRUST_Q
folds = np.random.default_rng(0).permutation(len(mix)) % 5
cv_ok = np.zeros(len(mix), bool)
for f in range(5):
    tr, te = folds != f, folds == f
    ff = G.fit_logistic(X[tr], y[tr])
    p_tr = G.gate_prob(ff, X[tr])
    cv_ok[te] = G.gate_prob(ff, X[te]) <= np.quantile(p_tr[y[tr] == 0], G.TRUST_Q)
ins_ok = pm <= fit["thr"]
ws = y == 1
mixed = lambda ok, sel: (sum(mix[i]["c1"] if ok[i] else mix[i]["cz"] for i in np.where(sel)[0]) - sum(mix[i]["cz"] for i in np.where(sel)[0])) / sum(mix[i]["n"] for i in np.where(sel)[0])  # noqa: E731
# (3) the feature-set bound: the four features fitted directly on the deciding levels (in-sample, the most these features can do for a frame-level logistic)
Xb = np.vstack([XR, XW])
yb = np.r_[np.zeros(len(XR)), np.ones(len(XW))].astype(int)
fb = G.fit_logistic(Xb, yb)
pbr, pbw = G.gate_prob(fb, XR), G.gate_prob(fb, XW)
bound = []
for q in (0.90, 0.95, 0.96, 0.975, 0.99):
    thr = float(np.quantile(pbr, q))
    okr, okw = pbr <= thr, pbw <= thr
    bound.append(dict(q=q, trust_right=float(okr.mean()), g_minus_z_wrong=acc(wrong, okw, True) - lz_w, g_minus_a_right=acc(right, okr, True) - la_r))
ok95 = [b_ for b_ in bound if b_["trust_right"] >= 0.95]
res_bound = dict(auroc=fb["auroc"], grid=bound, best_w_at_trust95=max(b_["g_minus_z_wrong"] for b_ in ok95) if ok95 else None)
res = dict(seed=SEED, fit_frames=len(fit_pick), held_frames=len(held), auroc_mix=fit["auroc"], trust_q=G.TRUST_Q, bound=res_bound,
           rows_right=right, rows_wrong=wrong,
           mix_in_sample=dict(g_minus_z_wrong_scene=mixed(ins_ok, ws), rejects_wrong_scene=float(1 - ins_ok[ws].mean())),
           mix_cv5=dict(g_minus_z_wrong_scene=mixed(cv_ok, ws), rejects_wrong_scene=float(1 - cv_ok[ws].mean())),
           held_right=dict(a=la_r, z=lz_r, a_minus_z=la_r - lz_r), held_wrong=dict(a=la_w, z=lz_w, a_minus_z=la_w - lz_w), grid=grid)
os.makedirs(G.OUT, exist_ok=True)
json.dump(res, open(OUTP + ".tmp", "w"), indent=1)
os.replace(OUTP + ".tmp", OUTP)
print(json.dumps({k: v for k, v in res.items() if k not in ("grid", "rows_right", "rows_wrong", "bound")}, indent=None))
print("bound (fitted on the deciding levels, in-sample): AUROC", round(fb["auroc"], 4), "best W at trust >= 0.95:", res_bound["best_w_at_trust95"])
for b_ in bound:
    print(f"  bound q {b_['q']}: trust RIGHT {b_['trust_right']:.3f} | g - z WRONG {b_['g_minus_z_wrong']:+.4f} | g - a RIGHT {b_['g_minus_a_right']:+.4f}")
for g in grid:
    print(f"q {g['q']}: trust RIGHT {g['trust_right']:.3f} | g - z WRONG {g['g_minus_z_wrong']:+.4f} (rejects {1 - g['trust_wrong']:.3f}) | g - a RIGHT {g['g_minus_a_right']:+.4f}")
