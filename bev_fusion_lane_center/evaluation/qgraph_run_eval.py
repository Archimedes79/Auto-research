"""Full scoring of a trained lane head (E273) on the 873 VAL frames, the control g and the reference d, and the verdict by the card's rule.

    wsl bash tools/env/_py14.sh qgraph_run_eval.py score --run e273_p_s0 [--seed 0] [--n 0]      # p: graph, nodes, assignment (exact map and map error 1-2 m); g: E272 chain + geometric assigner
    wsl bash tools/env/_py14.sh qgraph_run_eval.py verdict --runs e273_p_s0 e273_p_s1               # the rule of the card, mean of the seeds (by hand: the numbers are printed)

p = the lane head (`runs/<run>/model.pt`), its delivered graph read out by `evaluation/qgraph_eval.py` (e >= 0.5, p >= 0.5; the named output step); frames = `qgraph_eval.val_frames()` (VAL only,
never TEST). g = the E272 chain of the same seed (`runs/e272_j_s<seed>`, read out at the card threshold) whose delivered vertices are assigned to the same lane-level map tokens by the
GEOMETRIC start-value assigner of E272 (sigma 1.5 m, lambda 2, dustbin 4: `qassign.Assigner` without its learned part); d = the stored numbers of the lane suite run of `recall_sp0_dp`
(graph reference). The map with an error is `map_sim.make_map(g, "error", token)`: the same fixed perturbation for p and g. Output: `runs/<exp>/<run>/card.json`, `results.md` (exp = e273 by default, `--exp e274` for E274; `verdict274` is the rule of the E274 card).
`lane_acc` is the e-weighted share of delivered nodes within 1 m of a GT lane whose most probable map piece is a piece of that lane (qgraph_eval.assign_metrics); the p - g interval is a paired
bootstrap over scenes of the per-frame (right, total) counts.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for sub in ("evaluation", "training", "training/grid", "training/data", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub)))
import lane_gt_data as LD  # noqa: E402
import map_sim as MS  # noqa: E402
import qgraph_eval as QE  # noqa: E402
import qpredict as QP  # noqa: E402

# the reference graph d = recall_sp0_dp: lane suite run graphs_recall_sp0_dp_carry_20261104_181557 (VAL 873)
D_REF = dict(Q=0.7424, F1_1=0.8781, lanes_found=0.7809, topo_F1=0.7405, topo_P=0.8107, topo_R=0.6815, junction_F1=0.2730, junction_P=0.2125, junction_R=0.3815, dup_frac=0.0766, vert100=10.94,
             source="runs/suite/graphs_recall_sp0_dp/quant/graphs_recall_sp0_dp_carry_20261104_181557")


def geo_pred(V, H, w, mp, sigma=1.5, lam=2.0, c_bin=4.0):
    """the prediction dict of the control g: vertices as nodes, assigned to the map tokens by E272's geometric start-value assigner (numpy, lane frame)"""
    V, H = np.asarray(V, float).reshape(-1, 2), np.asarray(H, float).reshape(-1, 2)
    M, T, K = len(V), len(mp["pos"]), mp["K"]
    d = V[:, None, :] - mp["pos"][None, :, :].astype(float)
    cost = (d * d).sum(-1) / sigma ** 2 + lam * (1.0 - H @ mp["tan"].T.astype(float))
    logit = -np.concatenate([cost, np.full((M, 1), c_bin)], 1)
    logit -= logit.max(1, keepdims=True)
    Pt = np.exp(logit)
    Pt /= Pt.sum(1, keepdims=True)
    P = np.zeros((M, K + 1))
    for k in range(K):
        P[:, k] = Pt[:, :T][:, mp["pidx"] == k].sum(1)
    P[:, K] = Pt[:, T]
    R = len(mp["roads"])
    PR = np.zeros((M, R + 1))
    for k in range(K):
        PR[:, mp["road"][k]] += P[:, k]
    PR[:, R] = P[:, K]
    nodes = dict(pos=V, dirv=H, e=np.ones(M), flags=np.zeros((M, 8)), s=np.zeros(M), cell=np.zeros((M, 2), int), section=np.arange(M))
    links = dict(src=np.zeros(0, int), dst=np.zeros(0, int), p=np.zeros(0), long=np.zeros(0, bool), a=np.zeros(0), b=np.zeros(0), curves=None)
    return dict(nodes=nodes, links=links, assign=dict(P=P, map_ids=list(mp["ids"]), road=PR, road_ids=list(mp["roads"])), maps=None)


def chain_vertices(seed, n=0, device="cuda"):
    """the delivered vertices of the E272 chain of this seed on the VAL frames (read out at the threshold of its card)"""
    import chain as CH
    import chain_data as CD
    import vertex_eval as VE
    cfg = json.load(open(os.path.join(ROOT, "runs", f"e272_j_s{seed}", "config.json")))
    model = CH.GridChain(width=cfg.get("width", 48))
    model.load_state_dict(torch.load(os.path.join(ROOT, "runs", f"e272_j_s{seed}", "model.pt"), map_location="cpu")["state"])
    model.eval().to(device)
    thr = float(json.load(open(os.path.join(ROOT, "runs", "e272", f"e272_j_s{seed}", "card.json")))["thr"])
    nval = len(CD.PD.BuilderData("val_clean", train=False).rows)
    ks = list(range(nval))[: n or None]
    _, dl = CD.loader("val_clean", bs=4, workers=2, shuffle=False, ks=ks)
    out = []
    with torch.no_grad():
        for b in dl:
            o = model(b["x"].to(device), b["step"].to(device), b["has"].to(device))
            for i in range(o["fused"].shape[0]):
                V, H, w = VE.read_out(o["fused"][i].cpu(), thr)
                V, H, w = VE.inside(V, H, w)
                out.append((V, H, w))
    return out


def predict_val(run, mode, n=0, device="cuda", workers=3):
    m = QP.load_model(os.path.join(ROOT, "runs", run), device)
    ds, dl = LD.loader("val_clean", bs=4, workers=workers, shuffle=False, map_mode=mode, pixels=False, strict=True)
    preds = []
    for k, b in enumerate(dl):
        preds += QP.predict(m, b)
        if n and len(preds) >= n:
            break
    return ds, preds[: n or None]


def bootstrap_diff(rec_a, rec_b, scenes, B=1000, seed=0):
    """paired scene bootstrap of (a - b) of a ratio with per-frame (right, total) counts -> (mean, lo, hi)"""
    a = np.array([c["lane"] for c in rec_a], float)
    b = np.array([c["lane"] for c in rec_b], float)
    sc = np.asarray(scenes)
    us = np.unique(sc)
    rng = np.random.RandomState(seed)

    def ratio(x, idx):
        return x[idx, 0].sum() / max(x[idx, 1].sum(), 1e-9)
    full = np.arange(len(a))
    d0 = ratio(a, full) - ratio(b, full)
    ds = []
    for _ in range(B):
        pick = rng.choice(us, len(us))
        idx = np.concatenate([np.flatnonzero(sc == s) for s in pick])
        ds.append(ratio(a, idx) - ratio(b, idx))
    return float(d0), float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))


def score(run, seed, n=0, device="cuda", exp="e273"):
    t0 = time.time()
    frames = QE.val_frames()[: n or None]
    share = QE.stored_share(frames)
    assert share == 1.0, f"only {share:.3f} of the VAL frames have a stored lane GT"
    card = dict(run=run, seed=seed, split="val", frames=len(frames), date=time.strftime("%Y-%m-%d %H:%M"), d_ref=D_REF)
    res, recs = {}, {}
    for mode in ("exact", "error"):
        ds, preds = predict_val(run, mode, n, device)
        assert [int(ds.bd.rows[k]) for k in ds.ks][: len(frames)] == [f.row for f in frames], "the frame order of the loader differs from the scorer's"
        out, rec = QE.score_frames(preds, frames, records=True)
        res[mode], recs[mode] = out, rec
    card["p"] = dict(graph=res["exact"]["graph"], nodes=res["exact"]["nodes"], assign_exact=res["exact"]["assign"], assign_error=res["error"]["assign"])
    # the control g
    vs = chain_vertices(seed, n, device)
    g_rec = {}
    g_res = {}
    for mode in ("exact", "error"):
        rec = []
        for f, (V, H, w) in zip(frames, vs):
            g = f.lane()
            mp = MS.make_map(g, mode, token=f.token)
            rec.append(QE.assign_counts(geo_pred(V, H, w, mp), QE._ref(g), 0.5))
            f._lane = f._ref = None
        g_rec[mode] = rec
        g_res[mode] = QE.assign_numbers(QE.pool_counts(rec))
    card["g"] = dict(assign_exact=g_res["exact"], assign_error=g_res["error"], vertices_per_frame=float(np.mean([len(v[0]) for v in vs])))
    scenes = [f.scene for f in frames]
    card["diff"] = {mode: dict(zip(("mean", "lo", "hi"), bootstrap_diff(recs[mode]["assign"], g_rec[mode], scenes))) for mode in ("exact", "error")}
    card["seconds"] = round(time.time() - t0, 1)
    out_dir = os.path.join(ROOT, "runs", exp, run)                                                      # runs/e273/<run> (E273) or runs/e274/<run> (E274)
    os.makedirs(out_dir, exist_ok=True)
    json.dump(card, open(os.path.join(out_dir, "card.json"), "w"), indent=1, default=float)
    write_md(card, os.path.join(out_dir, "results.md"))
    return card


def write_md(c, path):
    p, g, dd = c["p"], c["g"], c["d_ref"]
    gr = p["graph"]
    L = [f"# {c['run']} (seed {c['seed']}), VAL {c['frames']} frames, {c['date']}", "",
         "| | p (lane head) | g (E272 chain + geometric assigner) | d (recall_sp0_dp) |", "|---|---|---|---|",
         f"| lane_acc, exact map | {p['assign_exact'].get('lane_acc', float('nan')):.4f} | {g['assign_exact']['lane_acc']:.4f} | |",
         f"| lane_acc, map error 1-2 m | {p['assign_error'].get('lane_acc', float('nan')):.4f} | {g['assign_error']['lane_acc']:.4f} | |",
         f"| p - g, map error (95 % scene bootstrap) | {c['diff']['error']['mean']:+.4f} [{c['diff']['error']['lo']:+.4f}, {c['diff']['error']['hi']:+.4f}] | | |",
         f"| p - g, exact map | {c['diff']['exact']['mean']:+.4f} [{c['diff']['exact']['lo']:+.4f}, {c['diff']['exact']['hi']:+.4f}] | | |",
         f"| topo F1 | {gr.get('topo_F1', float('nan')):.4f} | | {dd['topo_F1']:.4f} |", f"| F1@1.0 | {gr.get('F1@1.0', float('nan')):.4f} | | {dd['F1_1']:.4f} |",
         f"| junction F1 | {gr.get('junction_F1', float('nan')):.4f} | | {dd['junction_F1']:.4f} |", f"| lanes found | {gr.get('lanes_found', float('nan')):.4f} | | {dd['lanes_found']:.4f} |",
         f"| Q | {gr.get('Q', float('nan')):.4f} | | {dd['Q']:.4f} |",
         f"| vertices per 100 m: node ids / Douglas-Peucker 0.3 m (budget 14.3 on the DP basis) | {gr.get('vert100', float('nan')):.1f} / {gr.get('vert100_dp', float('nan')):.1f} | | {dd['vert100']:.1f} / |",
         f"| pieces per GT piece, id switches per 100 m | {gr.get('pieces_per_gt_piece', float('nan')):.3f}, {gr.get('id_switches_per_100m', float('nan')):.3f} | | |", "", "graph numbers of p: `qgraph_eval.graph_metrics` (the lane suite's functions); d from " + dd["source"] + ".", "",
         "```json", json.dumps({k: c["p"][k] for k in ("graph", "nodes")}, indent=1, default=float)[:3000], "```"]
    open(path, "w", encoding="utf-8").write("\n".join(L) + "\n")


def verdict(runs):
    """the rule of the card (section 0): mean over the seeds"""
    cards = [json.load(open(os.path.join(ROOT, "runs", "e273", r, "card.json"))) for r in runs]
    m = lambda f: float(np.mean([f(c) for c in cards]))  # noqa: E731
    d_err = m(lambda c: c["diff"]["error"]["mean"])
    d_ex = m(lambda c: c["diff"]["exact"]["mean"])
    topo = m(lambda c: c["p"]["graph"].get("topo_F1", float("nan")))
    f1 = m(lambda c: c["p"]["graph"].get("F1@1.0", float("nan")))
    jf = m(lambda c: c["p"]["graph"].get("junction_F1", float("nan")))
    guards = dict(lane_acc_exact=d_ex >= -0.01, topo_F1=topo >= D_REF["topo_F1"] - 0.03, F1_1=f1 >= D_REF["F1_1"] - 0.03)
    seeds_same_side = all((c["diff"]["error"]["mean"] > 0.03) for c in cards) or all((abs(c["diff"]["error"]["mean"]) <= 0.03) for c in cards) or all((c["diff"]["error"]["mean"] < -0.03) for c in cards)
    if not all(guards.values()) or d_err < -0.03:
        v = "CUT"
    elif d_err > 0.03:
        v = "KEEP"
    else:
        v = "TIE"
    out = dict(verdict=v, lane_acc_err_p_minus_g=d_err, lane_acc_exact_p_minus_g=d_ex, topo_F1=topo, F1_1=f1, junction_F1=jf, guards=guards, seeds_same_side=seeds_same_side, runs=runs,
               note="by hand: card.validate cannot read assignment numbers; both seeds must lie on the same side of the band")
    print(json.dumps(out, indent=1))
    return out


def verdict274(var, ctl, exp_var="e274", exp_ctl="e273", metric="topo_F1", band=0.02, root=None):
    """the rule of the E274 card (section 0), by hand like E273: the variant's gain of `metric` over the control (per seed, same seed number), guards on the assignment and the delivered graph, and
    the system pass against the reference d (reported). KEEP: gain above the band on both seeds and every guard held; CUT: a gain below -band on any seed or a guard broken; TIE: both gains
    inside the band; anything else INCONCLUSIVE (add a seed). Guards: lane_acc_exact p - g >= -0.01, lane_acc_err p - g >= +0.03 (mean of the seeds), F1@1.0 not below the control by more than 0.01 on
    any seed and, when the headline is not topo_F1 (arm p against r), topo_F1 not below the control by more than 0.01 on any seed."""
    root = root or ROOT
    load = lambda exp, r: json.load(open(os.path.join(root, "runs", exp, r, "card.json")))  # noqa: E731
    cv, cc = [load(exp_var, r) for r in var], [load(exp_ctl, r) for r in ctl]
    assert [c["seed"] for c in cv] == [c["seed"] for c in cc], "variant and control must list the same seeds in the same order"
    gain = [a["p"]["graph"][metric] - b["p"]["graph"][metric] for a, b in zip(cv, cc)]
    mean = lambda f: float(np.mean([f(c) for c in cv]))  # noqa: E731
    f1 = [a["p"]["graph"]["F1@1.0"] - b["p"]["graph"]["F1@1.0"] for a, b in zip(cv, cc)]
    guards = dict(lane_acc_exact_p_minus_g=mean(lambda c: c["diff"]["exact"]["mean"]) >= -0.01, lane_acc_err_p_minus_g=mean(lambda c: c["diff"]["error"]["mean"]) >= 0.03,
                  F1_1_not_below_control_by_0_01=all(g >= -0.01 for g in f1))
    topo_gain = [a["p"]["graph"]["topo_F1"] - b["p"]["graph"]["topo_F1"] for a, b in zip(cv, cc)]
    if metric != "topo_F1":
        guards["topo_F1_not_below_control_by_0_01"] = all(g >= -0.01 for g in topo_gain)
    side = ("up" if all(g > band for g in gain) else "down" if any(g < -band for g in gain) else "inside" if all(abs(g) <= band for g in gain) else "mixed")
    v = "CUT" if (side == "down" or not all(guards.values())) else "KEEP" if side == "up" else "TIE" if side == "inside" else "INCONCLUSIVE (seeds differ: add a seed)"
    g = lambda c, k: c["p"]["graph"][k]  # noqa: E731
    sysm = dict(topo_F1=mean(lambda c: g(c, "topo_F1")), F1_1=mean(lambda c: g(c, "F1@1.0")), junction_F1=mean(lambda c: g(c, "junction_F1")), node_R=mean(lambda c: c["p"]["nodes"]["node_R"]), node_P=mean(lambda c: c["p"]["nodes"]["node_P"]),
                vert100_dp=mean(lambda c: g(c, "vert100_dp")), dup_frac=mean(lambda c: g(c, "dup_frac")), long_link_P=mean(lambda c: c["p"]["nodes"]["link_long_P"]), long_link_R=mean(lambda c: c["p"]["nodes"]["link_long_R"]))
    sysm["system_pass_vs_d"] = bool(sysm["topo_F1"] >= D_REF["topo_F1"] - 0.03 and sysm["F1_1"] >= D_REF["F1_1"] - 0.03)
    out = dict(verdict=v, metric=metric, band=band, gain_per_seed=gain, side=side, guards=guards, F1_1_gain_per_seed=f1, topo_F1_gain_per_seed=topo_gain, variant_numbers=sysm, variant=var, control=ctl,
               note="by hand: card.validate cannot read assignment numbers; the written rule of the E274 card (section 0), both seeds on the same side; density and system pass are reported, not judged, for arms r and p")
    print(json.dumps(out, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--run", required=True)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--n", type=int, default=0)
    s.add_argument("--exp", default="e273", help="the experiment folder of the scoring card: runs/<exp>/<run>/")
    v = sub.add_parser("verdict")
    v.add_argument("--runs", nargs="+", required=True)
    w = sub.add_parser("verdict274")
    w.add_argument("--var", nargs="+", required=True, help="the variant runs (both seeds), cards in runs/<exp-var>/")
    w.add_argument("--ctl", nargs="+", required=True, help="the control runs (both seeds)")
    w.add_argument("--exp-var", default="e274")
    w.add_argument("--exp-ctl", default="e273")
    w.add_argument("--metric", default="topo_F1")
    w.add_argument("--band", type=float, default=0.02)
    a = ap.parse_args()
    if a.cmd == "score":
        c = score(a.run, a.seed, a.n, "cuda" if torch.cuda.is_available() else "cpu", exp=a.exp)
        print(open(os.path.join(ROOT, "runs", a.exp, a.run, "results.md"), encoding="utf-8").read())
    elif a.cmd == "verdict274":
        verdict274(a.var, a.ctl, a.exp_var, a.exp_ctl, a.metric, a.band)
    else:
        verdict(a.runs)


if __name__ == "__main__":
    main()
