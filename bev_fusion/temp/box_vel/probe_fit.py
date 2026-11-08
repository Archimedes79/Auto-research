"""t11 box velocity, step 3: read-outs of the frozen features at the supervised vehicle centres (npz of probe_features.py).

fit = a train-split npz, test = a val-split npz (other scenes). At the GT centres (perfect localisation: an upper reference for the
matched-box AVE of the evaluation, labelled so), vehicles only, moving = |v_gt| >= 0.5 m/s (evaluation/metrics.py MOVING_SPEED):
  as stored    the run's own vel output and v_hat at the centres
  ridge        closed-form ridge of (vx, vy) on the standardised feature (S = trunk 96 ch, x = object head shared 64 ch), lambda by 5-fold
               CV over frames of the fit set
  logistic     moving / static on the same feature (AUC, BCE against the base-rate entropy: the V3 / V4 test of t11_vel_hurdle)
  L1 linear    the linear L1 (median) regression by Adam on the full fit set: how far from 0 the median read-out gets
  branch       the vel branch itself (3 x 3 conv is a 1 x 1 here: Linear 64 -> 64, ReLU, Linear 64 -> 2, PyTorch default init) trained from
               scratch on the frozen shared feature x with batches of 9 frames' centres, Adam lr 3e-4, 1500 steps, for the losses
               l1 (today), huber (delta 1 m/s), l2, moving (L1 on moving centres only = the hurdle's regression): active share of its hidden
               units at the test centres and its skill
Prints one table; -> temp/box_vel/out/probe_fit_<tag>.json
Run (from bev_fusion/): OMP_NUM_THREADS=2 ~/miniconda3/envs/gcar/bin/python temp/box_vel/probe_fit.py fit.npz test.npz tag
"""
import json
import os
import sys

import numpy as np

MOVING = 0.5


def load(path, group="vehicle", part=None):
    """part: None = all centres; 'first' / 'second' = the frames below / above the median frame index (the frames of a npz run in
    scene order, so the halves share at most one scene): a within-file fit / test split when no train-split file exists"""
    z = np.load(path, allow_pickle=True)
    g = list(z["groups"]).index(group)
    k = z["group"] == g
    if part is not None:
        cut = np.median(z["frame"])
        k = k & ((z["frame"] <= cut) if part == "first" else (z["frame"] > cut))
    return {n: z[n][k].astype(np.float32) for n in ("t_vel", "S", "x", "z_vel", "vel", "vhat")} | {"frame": z["frame"][k], "step": int(z["step"])}


def skill(pred, tv):
    """vehicle numbers at the centres: AVE moving, ave0 moving, skill moving, mean |pred| on static, n moving / static"""
    sp = np.linalg.norm(tv, axis=1)
    mv, st = sp >= MOVING, sp < MOVING
    e = np.linalg.norm(pred - tv, axis=1)
    ave, ave0 = float(e[mv].mean()), float(sp[mv].mean())
    return {"ave_moving": ave, "ave0_moving": ave0, "skill_moving": 1 - ave / ave0, "ave_static": float(e[st].mean()),
            "n_moving": int(mv.sum()), "n_static": int(st.sum())}


def standardise(a, b):
    mu, sd = a.mean(0), a.std(0) + 1e-6
    return (a - mu) / sd, (b - mu) / sd


def ridge(Xf, yf, Xt, frames, lams=(1e-2, 1e-1, 1, 10, 100, 1000)):
    """closed-form ridge with intercept, lambda by 5-fold CV over frames"""
    def fit(X, y, lam):
        Xa = np.c_[X, np.ones(len(X))]
        R = lam * np.eye(Xa.shape[1])
        R[-1, -1] = 0
        return np.linalg.solve(Xa.T @ Xa + R, Xa.T @ y)

    def pred(W, X):
        return np.c_[X, np.ones(len(X))] @ W
    uf = np.unique(frames)
    fold = {f: i % 5 for i, f in enumerate(np.random.default_rng(0).permutation(uf))}
    fi = np.array([fold[f] for f in frames])
    cv = {}
    for lam in lams:
        err = []
        for k in range(5):
            if not (fi == k).any() or (fi == k).all():
                continue
            W = fit(Xf[fi != k], yf[fi != k], lam)
            err.append(np.linalg.norm(pred(W, Xf[fi == k]) - yf[fi == k], axis=1).mean())
        cv[lam] = float(np.mean(err))
    best = min(cv, key=cv.get)
    return pred(fit(Xf, yf, best), Xt), best


def logistic(Xf, mf, Xt, mt, steps=3000):
    import torch
    lin = torch.nn.Linear(Xf.shape[1], 1)
    opt = torch.optim.Adam(lin.parameters(), 1e-2, weight_decay=1e-4)
    X, y = torch.tensor(Xf), torch.tensor(mf, dtype=torch.float32)
    for _ in range(steps):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(lin(X)[:, 0], y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        lt = lin(torch.tensor(Xt))[:, 0]
        p = torch.sigmoid(lt).numpy()
        bce = float(torch.nn.functional.binary_cross_entropy_with_logits(lt, torch.tensor(mt, dtype=torch.float32)))
    r = mt.mean()
    h0 = float(-(r * np.log(r) + (1 - r) * np.log(1 - r)))
    order = np.argsort(p)
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    npos, nneg = mt.sum(), (1 - mt).sum()
    auc = float((ranks[mt > 0].sum() - npos * (npos + 1) / 2) / (npos * nneg))
    return p, {"auc": auc, "bce": bce, "base_entropy": h0, "base_rate_test": float(r)}


def l1_linear(Xf, yf, Xt, steps=3000):
    import torch
    lin = torch.nn.Linear(Xf.shape[1], 2)
    torch.nn.init.zeros_(lin.weight)
    torch.nn.init.zeros_(lin.bias)
    opt = torch.optim.Adam(lin.parameters(), 1e-2)
    X, y = torch.tensor(Xf), torch.tensor(yf)
    for _ in range(steps):
        loss = (lin(X) - y).abs().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return lin(torch.tensor(Xt)).numpy(), float(lin.weight.norm())


def branch(xf, yf, frames, xt, kind, steps=1500, seed=0):
    """the vel branch (1 x 1 version) from PyTorch's default init on the frozen shared feature, Adam lr 3e-4, batches of 9 frames"""
    import torch
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(xf.shape[1], 64), torch.nn.ReLU(), torch.nn.Linear(64, 2))
    torch.nn.init.zeros_(net[2].bias)
    opt = torch.optim.Adam(net.parameters(), 3e-4)
    X, Y = torch.tensor(xf), torch.tensor(yf)
    mov = (Y.norm(dim=1) >= MOVING).float()
    uf = np.unique(frames)
    rng = np.random.default_rng(seed)
    alive = []
    for it in range(steps):
        pick = np.isin(frames, rng.choice(uf, min(9, len(uf)), replace=False))
        xb, yb, mb = X[pick], Y[pick], mov[pick]
        out = net(xb)
        e = out - yb
        if kind == "l1":
            loss = e.abs().sum(1).mean()
        elif kind == "huber":
            loss = torch.nn.functional.huber_loss(out, yb, reduction="none", delta=1.0).sum(1).mean()
        elif kind == "l2":
            loss = (e ** 2).sum(1).mean()
        elif kind == "moving":
            loss = (e.abs().sum(1) * mb).sum() / mb.sum().clamp(min=1)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 100 == 0 or it == steps - 1:
            with torch.no_grad():
                alive.append((it, float((net[0](torch.tensor(xt)) > 0).float().mean())))
    with torch.no_grad():
        h = net[0](torch.tensor(xt))
        return net(torch.tensor(xt)).numpy(), alive, float((h > 0).float().mean())


def main(fit_path, test_path, tag):
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    same = fit_path == test_path                                   # one file: the first half of its frames fits, the second tests
    F, T = load(fit_path, part="first" if same else None), load(test_path, part="second" if same else None)
    tv = T["t_vel"]
    res = {"fit": fit_path, "test": test_path, "step": T["step"], "n_fit": len(F["t_vel"]), "n_test": len(tv)}
    res["stored_vel"] = skill(T["vel"], tv)
    res["stored_vhat"] = skill(T["vhat"], tv)
    res["zero"] = skill(np.zeros_like(tv), tv)
    res["vel_units_active_at_centres"] = float((T["z_vel"] > 0).mean())
    mt = (np.linalg.norm(tv, axis=1) >= MOVING).astype(np.float32)
    mf = (np.linalg.norm(F["t_vel"], axis=1) >= MOVING).astype(np.float32)
    for feat in ("S", "x"):
        Xf, Xt = standardise(F[feat], T[feat])
        pr, lam = ridge(Xf, F["t_vel"], Xt, F["frame"])
        res[f"ridge_{feat}"] = skill(pr, tv) | {"lambda": lam}
        p, lg = logistic(Xf, mf, Xt, mt)
        res[f"logistic_{feat}"] = lg
        res[f"ridge_{feat}_gated"] = skill(pr * (p >= 0.5)[:, None], tv)
        pl, wn = l1_linear(Xf, F["t_vel"], Xt)
        res[f"l1_linear_{feat}"] = skill(pl, tv) | {"weight_norm": wn}
    for kind in ("l1", "huber", "l2", "moving"):
        pb, alive, act = branch(F["x"], F["t_vel"], F["frame"], T["x"], kind)
        res[f"branch_{kind}"] = skill(pb, tv) | {"active_at_test_centres": act, "active_curve": alive}
    out = os.path.join(os.path.dirname(test_path), f"probe_fit_{tag}.json")
    json.dump(res, open(out, "w"), indent=1)
    print(f"{tag}: fit {res['n_fit']} / test {res['n_test']} vehicle centres (step {res['step']}); vel units active at the test centres "
          f"{res['vel_units_active_at_centres']:.4f}")
    print(f"{'read-out':22s} {'AVE mov':>8s} {'ave0 mov':>8s} {'skill':>7s} {'err static':>10s}  extra")
    for k, v in res.items():
        if isinstance(v, dict) and "skill_moving" in v:
            extra = {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()
                     if kk in ("lambda", "weight_norm", "active_at_test_centres")}
            print(f"{k:22s} {v['ave_moving']:8.3f} {v['ave0_moving']:8.3f} {v['skill_moving']:7.3f} {v['ave_static']:10.3f}  {extra}")
        elif k.startswith("logistic"):
            print(f"{k:22s} AUC {v['auc']:.3f}, BCE {v['bce']:.3f} vs base-rate entropy {v['base_entropy']:.3f} (moving share {v['base_rate_test']:.2f})")
    print(f"-> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
