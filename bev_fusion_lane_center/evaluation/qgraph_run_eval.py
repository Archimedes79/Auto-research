"""Full scoring of a trained lane head (E273) on the 873 VAL frames, the control g and the reference d, and the verdict by the card's rule.

    wsl bash tools/env/_py14.sh qgraph_run_eval.py score --run e273_p_s0 [--seed 0] [--n 0]      # p: graph, nodes, assignment (exact map and map error 1-2 m); g: E272 chain + geometric assigner
    wsl bash tools/env/_py14.sh qgraph_run_eval.py verdict --runs e273_p_s0 e273_p_s1               # the rule of the card, mean of the seeds (by hand: the numbers are printed)

E275: `score --exp e275 --readouts plain@0.5 close:1.5@0.5 close:1.5@0.4 plain@0.4 [--curve 1.0 1.25 1.75 2.0] [--dump]` scores several read-outs (`qgraph_readout.py`) of one model on one prediction pass,
one card per read-out in `runs/e275/<run>.<spec>/`; `verdict275` = the written rule of the E275 card (tested by `tests/test_verdict275.py`).

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
import qgraph_readout as RO  # noqa: E402
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


def score(run, seed, n=0, device="cuda", exp="e273", readouts=None, curve=(), dump=None, max_drop=0.20, repro=None, repro_tol=0.005):
    """one prediction pass per map mode, then every read-out in `readouts` (qgraph_readout specs, 'plain@0.5' / 'close:1.5@0.5') is scored on it. readouts=None is the E273 / E274 way: one
    card runs/<exp>/<run>/ at e >= 0.5. With readouts: one card per spec, runs/<exp>/<run>.<spec name>/ (card['run'] stays the model; card['readout'] holds the step's counts and ms per frame; the
    control g is computed once and shared). curve: radii of close-node suppression scored on the delivered graph only (exact map, e >= 0.5) -> runs/<exp>/<run>.curve.json. dump: path for the
    candidate nodes (e >= 0.2) of the exact-map pass (scratch pickle for the analysis). max_drop: the kill criterion of the E275 card, a close-node spec that deletes a larger share of the
    nodes delivered at its threshold stops the scoring before any card is written. repro: path of the E274 sweep json of this model; the close:1.5@0.5 read-out must reproduce its entry
    'pos1.5_e0.5' (topo F1, F1@1.0, junction F1, lanes found, node P / R) within repro_tol, else the scoring stops after the exact-map pass, before the control and before any card is written."""
    t0 = time.time()
    if repro:                                                                                       # before any GPU work; a relative path is relative to the project root (the scorer runs in evaluation/)
        repro = repro if os.path.isabs(repro) else os.path.join(ROOT, repro)
        if not os.path.isfile(repro):
            raise SystemExit(f"--repro: the sweep file {repro} does not exist (stopped before any prediction pass)")
    frames = QE.val_frames()[: n or None]
    share = QE.stored_share(frames)
    assert share == 1.0, f"only {share:.3f} of the VAL frames have a stored lane GT"
    card = dict(run=run, seed=seed, split="val", frames=len(frames), date=time.strftime("%Y-%m-%d %H:%M"), d_ref=D_REF)
    specs = [RO.parse(s) for s in readouts] if readouts else [RO.parse("plain@0.5")]
    res, recs, info, curve_out = {s: {} for s in specs}, {s: {} for s in specs}, {}, {}
    for mode in ("exact", "error"):
        ds, preds = predict_val(run, mode, n, device)
        assert [int(ds.bd.rows[k]) for k in ds.ks][: len(frames)] == [f.row for f in frames], "the frame order of the loader differs from the scorer's"
        if mode == "exact":
            if dump:
                os.makedirs(os.path.dirname(dump), exist_ok=True)
                RO.dump_nodes(preds, [f.row for f in frames], dump)
            for s in specs:
                if s.kind != "close":
                    continue
                before, after = RO.counts(preds, s)
                info[s] = dict(spec=RO.name(s), radius=s.radius, thr_e=s.thr, nodes_before=before, nodes_after=after, dropped_frac=1.0 - after / max(before, 1), tied_pairs_within_radius=RO.tied_pairs(preds, s), ms_per_frame=RO.timed(preds, s))
                print(f"{run} {RO.name(s)}: {before} nodes delivered, {after} after the step ({100 * info[s]['dropped_frac']:.1f} % dropped), {info[s]['ms_per_frame']:.3f} ms per frame", flush=True)
                if info[s]["dropped_frac"] > max_drop:                                                    # the kill criterion of the card is for the judged spec (threshold 0.5); the others only warn
                    msg = f"{RO.name(s)} drops {100 * info[s]['dropped_frac']:.1f} % of the delivered nodes (limit {100 * max_drop:.0f} %)"
                    if s.thr >= 0.5:
                        raise SystemExit(f"KILL CRITERION: {msg}: nothing written, search the code")
                    print(f"WARNING (reported spec, not judged): {msg}", flush=True)
            for r in curve:                                                                              # graph numbers only, at e >= 0.5
                o = QE.score_frames([RO.apply(p, RO.Spec("close", float(r), 0.5)) for p in preds], frames)
                curve_out[f"{r:g}"] = {k: o["graph"][k] for k in ("topo_F1", "F1@1.0", "junction_F1", "lanes_found", "Q", "dup_frac", "vert100_dp")} | {k: o["nodes"][k] for k in ("node_P", "node_R")}
                print(f"{run} close {r:g} m, e>=0.5: topo {curve_out[f'{r:g}']['topo_F1']:.3f} F1@1 {curve_out[f'{r:g}']['F1@1.0']:.3f} junction {curve_out[f'{r:g}']['junction_F1']:.3f} "
                      f"lanes {curve_out[f'{r:g}']['lanes_found']:.3f} node R {curve_out[f'{r:g}']['node_R']:.3f}", flush=True)
        for s in specs:
            out, rec = QE.score_frames([RO.apply(p, s) for p in preds], frames, thr_e=s.thr, records=True)
            res[s][mode], recs[s][mode] = out, rec
            if mode == "exact" and repro and s == RO.Spec("close", 1.5, 0.5):                          # layer 1: the named step must reproduce the scratch sweep of E274
                ref = json.load(open(repro))["pos1.5_e0.5"]
                mine = dict(topo_F1=out["graph"]["topo_F1"], F1_1=out["graph"]["F1@1.0"], junction_F1=out["graph"]["junction_F1"], lanes_found=out["graph"]["lanes_found"],
                            node_P=out["nodes"]["node_P"], node_R=out["nodes"]["node_R"])
                dif = {k: abs(mine[k] - ref[k]) for k in mine}
                print(f"{run} close1.5_t0.5 against the E274 sweep: max abs difference {max(dif.values()):.2e} ({repro})", flush=True)
                if max(dif.values()) > repro_tol:
                    raise SystemExit(f"KILL CRITERION: the step does not reproduce the E274 sweep (tolerance {repro_tol}): {dif}: nothing written, search the code")
                info[s]["reproduces_e274_sweep_max_abs_diff"] = max(dif.values())
        del preds
    # the control g (does not depend on the read-out of p)
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
    scenes = [f.scene for f in frames]
    cards = {}
    for s in specs:
        c = dict(card)
        c["p"] = dict(graph=res[s]["exact"]["graph"], nodes=res[s]["exact"]["nodes"], assign_exact=res[s]["exact"]["assign"], assign_error=res[s]["error"]["assign"])
        if res[s]["exact"].get("tier"):                                                                  # E276: the tier numbers (only with the l4 lane GT)
            c["p"]["tier"] = res[s]["exact"]["tier"]
        c["g"] = dict(assign_exact=g_res["exact"], assign_error=g_res["error"], vertices_per_frame=float(np.mean([len(v[0]) for v in vs])))
        c["diff"] = {mode: dict(zip(("mean", "lo", "hi"), bootstrap_diff(recs[s][mode]["assign"], g_rec[mode], scenes))) for mode in ("exact", "error")}
        if readouts:
            c["readout"] = {**dict(spec=RO.name(s), kind=s.kind, radius=s.radius, thr_e=s.thr), **(info.get(s) or {})}
            old = next((json.load(open(p)) for p in (os.path.join(ROOT, "runs", e, run, "card.json") for e in ("e273", "e274")) if os.path.exists(p)), None)
            if s == RO.Spec("plain", None, 0.5) and old is not None and old.get("frames") == len(frames):    # layer 1: the unchanged read-out must give the stored card (same frames)
                diffs = [abs(c["p"]["graph"][k] - old["p"]["graph"][k]) for k in ("topo_F1", "F1@1.0", "junction_F1", "lanes_found", "Q", "dup_frac")]
                diffs += [abs(c["diff"][m]["mean"] - old["diff"][m]["mean"]) for m in ("exact", "error")]
                c["readout"]["reproduces_stored_card_max_abs_diff"] = max(diffs)
                print(f"{run} plain@0.5 against the stored card: max abs difference {max(diffs):.2e}", flush=True)
        c["seconds"] = round(time.time() - t0, 1)
        out_dir = os.path.join(ROOT, "runs", exp, f"{run}.{RO.name(s)}" if readouts else run)         # runs/e273/<run> (E273), runs/e274/<run> (E274), runs/e275/<run>.<spec> (E275)
        os.makedirs(out_dir, exist_ok=True)
        json.dump(c, open(os.path.join(out_dir, "card.json"), "w"), indent=1, default=float)
        write_md(c, os.path.join(out_dir, "results.md"))
        cards[s] = c
    if curve_out:
        json.dump(dict(run=run, seed=seed, thr_e=0.5, frames=len(frames), curve=curve_out), open(os.path.join(ROOT, "runs", exp, f"{run}.curve.json"), "w"), indent=1, default=float)
    return cards[specs[0]] if not readouts else cards


def write_md(c, path):
    p, g, dd = c["p"], c["g"], c["d_ref"]
    gr = p["graph"]
    L = [f"# {c['run']}{' [' + c['readout']['spec'] + ']' if 'readout' in c else ''} (seed {c['seed']}), VAL {c['frames']} frames, {c['date']}", "",
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
    if "readout" in c:                                                                                  # E275: the read-out step and its counts
        L.insert(1, "")
        L.insert(2, "read-out: " + ", ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in c["readout"].items() if v is not None))
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


def verdict274(var, ctl, exp_var="e274", exp_ctl="e273", metric="topo_F1", band=0.02, root=None, verbose=True, topo_guard=True):
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
    if metric != "topo_F1" and topo_guard:                                                    # the E274 card's guard for arm p against r; the E275 card has none (verdict275 passes False)
        guards["topo_F1_not_below_control_by_0_01"] = all(g >= -0.01 for g in topo_gain)
    side = ("up" if all(g > band for g in gain) else "down" if any(g < -band for g in gain) else "inside" if all(abs(g) <= band for g in gain) else "mixed")
    v = "CUT" if (side == "down" or not all(guards.values())) else "KEEP" if side == "up" else "TIE" if side == "inside" else "INCONCLUSIVE (seeds differ: add a seed)"
    g = lambda c, k: c["p"]["graph"][k]  # noqa: E731
    sysm = dict(topo_F1=mean(lambda c: g(c, "topo_F1")), F1_1=mean(lambda c: g(c, "F1@1.0")), junction_F1=mean(lambda c: g(c, "junction_F1")), node_R=mean(lambda c: c["p"]["nodes"]["node_R"]), node_P=mean(lambda c: c["p"]["nodes"]["node_P"]),
                vert100_dp=mean(lambda c: g(c, "vert100_dp")), dup_frac=mean(lambda c: g(c, "dup_frac")), long_link_P=mean(lambda c: c["p"]["nodes"]["link_long_P"]), long_link_R=mean(lambda c: c["p"]["nodes"]["link_long_R"]))
    sysm["system_pass_vs_d"] = bool(sysm["topo_F1"] >= D_REF["topo_F1"] - 0.03 and sysm["F1_1"] >= D_REF["F1_1"] - 0.03)
    out = dict(verdict=v, metric=metric, band=band, gain_per_seed=gain, side=side, guards=guards, F1_1_gain_per_seed=f1, topo_F1_gain_per_seed=topo_gain, variant_numbers=sysm, variant=var, control=ctl,
               note="by hand: card.validate cannot read assignment numbers; the written rule of the E274 card (section 0), both seeds on the same side; density and system pass are reported, not judged, for arms r and p")
    if verbose:
        print(json.dumps(out, indent=1))
    return out


def verdict275(var, ctl, exp_var="e275", exp_ctl="e273", root=None, lanes_tol=0.03, verbose=True, spec="close1.5_t0.5", frames=873):
    """the rule of the E275 card (section 0) for the close-node suppression (variant `var`: the cards of the step, control `ctl`: the same models read out as before; both lists in the same seed
    order). Two calls of verdict274 (topo_F1, band 0.02; junction_F1, band 0.03: each with the assignment guards and the F1@1.0 guard of that function, WITHOUT the topo guard of the E274 card) and
    one more guard: lanes_found not below the control by more than `lanes_tol` on any seed. KEEP: both headlines up on both seeds and every guard held. CUT: either headline down on a seed or any
    guard broken. TIE: both inside their bands and the guards held. Everything else: INCONCLUSIVE (goes to Mirko: there is no further seed without retraining).
    What it judges is checked first: two seeds (0 and 1) in the same order, `frames` frames in every card, the variant cards carry the read-out `spec`, the control cards none (or plain_t0.5)."""
    root = root or ROOT
    load = lambda exp, r: json.load(open(os.path.join(root, "runs", exp, r, "card.json")))  # noqa: E731
    cv, cc = [load(exp_var, r) for r in var], [load(exp_ctl, r) for r in ctl]
    assert [c["seed"] for c in cv] == [c["seed"] for c in cc] == [0, 1], f"the rule needs seeds 0 and 1 in this order, got {[c['seed'] for c in cv]} against {[c['seed'] for c in cc]}"
    assert all(c["frames"] == frames for c in cv + cc), f"every card must hold {frames} frames: {[c['frames'] for c in cv + cc]}"
    assert all(c.get("readout", {}).get("spec") == spec for c in cv), f"the variant cards must carry the read-out {spec}: {[c.get('readout', {}).get('spec') for c in cv]}"
    assert all(c.get("readout", {}).get("spec", "plain_t0.5") == "plain_t0.5" for c in cc), "the control cards must be read out as before (no read-out step)"
    t = verdict274(var, ctl, exp_var, exp_ctl, "topo_F1", 0.02, root, verbose=False, topo_guard=False)
    j = verdict274(var, ctl, exp_var, exp_ctl, "junction_F1", 0.03, root, verbose=False, topo_guard=False)
    lanes = [a["p"]["graph"]["lanes_found"] - b["p"]["graph"]["lanes_found"] for a, b in zip(cv, cc)]
    guards = dict(t["guards"])
    guards["lanes_found_not_below_control_by_%g" % lanes_tol] = all(x >= -lanes_tol for x in lanes)
    sides = (t["side"], j["side"])
    if "down" in sides or not all(guards.values()):
        v = "CUT"
    elif sides == ("up", "up"):
        v = "KEEP"
    elif sides == ("inside", "inside"):
        v = "TIE"
    else:
        v = "INCONCLUSIVE (the two headlines or the seeds disagree; goes to Mirko)"
    out = dict(verdict=v, topo_F1=dict(side=t["side"], gain_per_seed=t["gain_per_seed"]), junction_F1=dict(side=j["side"], gain_per_seed=j["gain_per_seed"]), lanes_found_gain_per_seed=lanes,
               F1_1_gain_per_seed=t["F1_1_gain_per_seed"], guards=guards, variant_numbers=t["variant_numbers"], variant=var, control=ctl,
               note="by hand: the written rule of the E275 card (section 0); in-sample on VAL; density and system pass are reported, not judged")
    if verbose:
        print(json.dumps(out, indent=1))
    return out


def verdict276(var, ctl, exp_var="e276", exp_ctl="e276", kind="t", root=None, band=0.02, topo_tol=0.05, recall_tol=0.02, density_cut=0.25, verbose=True, frames=873):
    """the rule of the E276 card (section 0), by hand. kind 't': arm t (`var`) against the baseline B scored on GT l4 (`ctl`, cards runs/<exp_ctl>/<run>/); kind 's': arm s against arm t.
    Both lists in the same seed order (0, 1), `frames` frames in every card, the tier numbers present. Per seed: gain = t12_F1_1(var) - t12_F1_1(ctl). Guards: t1_R_1 not below the control by more
    than `recall_tol`; kind 't': topo_F1 not below it by more than `topo_tol` (the price of not demanding unseen parts); both: assignment `lane_acc_exact` p - g >= -0.01 and `lane_acc_err` p - g >= +0.03
    (mean of the seeds, as E274). kind 't': KEEP when both gains exceed `band`, TIE when both are inside, CUT when a gain is below -band or a guard broke, else INCONCLUSIVE. kind 's': KEEP when
    the delivered nodes per 100 m of GT lane (`p.nodes.nodes_per_100m`) fall by at least `density_cut` on both seeds AND t12_F1_1 stays within `band` below the control on both seeds AND the guards hold; CUT when t12_F1_1 falls by more than `band`
    on a seed or a guard broke; else TIE (no density gain: t stays)."""
    root = root or ROOT
    load = lambda exp, r: json.load(open(os.path.join(root, "runs", exp, r, "card.json")))  # noqa: E731
    cv, cc = [load(exp_var, r) for r in var], [load(exp_ctl, r) for r in ctl]
    assert kind in ("t", "s"), kind
    assert [c["seed"] for c in cv] == [c["seed"] for c in cc] == [0, 1], f"the rule needs seeds 0 and 1 in this order, got {[c['seed'] for c in cv]} against {[c['seed'] for c in cc]}"
    assert all(c["frames"] == frames for c in cv + cc), f"every card must hold {frames} frames: {[c['frames'] for c in cv + cc]}"
    assert all("tier" in c["p"] for c in cv + cc), "a card without tier numbers (scored on a lane GT without the l4 labels)"
    tier = lambda c, k: c["p"]["tier"][k]  # noqa: E731
    gain = [tier(a, "t12_F1_1") - tier(b, "t12_F1_1") for a, b in zip(cv, cc)]
    r1 = [tier(a, "t1_R_1") - tier(b, "t1_R_1") for a, b in zip(cv, cc)]
    topo = [a["p"]["graph"]["topo_F1"] - b["p"]["graph"]["topo_F1"] for a, b in zip(cv, cc)]
    dens = [1.0 - a["p"]["nodes"]["nodes_per_100m"] / b["p"]["nodes"]["nodes_per_100m"] for a, b in zip(cv, cc)]       # the density the arm s changes: DELIVERED NODES per 100 m of GT lane (the oracle diagnostic P1 of the card:
    # vert100_dp counts the Douglas-Peucker vertices of the GEOMETRY and is 15 for a perfect network whatever the node target, so it is reported, not judged)
    mean = lambda f: float(np.mean([f(c) for c in cv]))  # noqa: E731
    guards = {}
    guards["lane_acc_exact_p_minus_g"] = mean(lambda c: c["diff"]["exact"]["mean"]) >= -0.01
    guards["lane_acc_err_p_minus_g"] = mean(lambda c: c["diff"]["error"]["mean"]) >= 0.03
    guards["t1_R_1_not_below_control_by_%g" % recall_tol] = all(x >= -recall_tol for x in r1)
    if kind == "t":
        guards["topo_F1_not_below_control_by_%g" % topo_tol] = all(x >= -topo_tol for x in topo)
        side = "up" if all(g > band for g in gain) else "down" if any(g < -band for g in gain) else "inside" if all(abs(g) <= band for g in gain) else "mixed"
        v = "CUT" if (side == "down" or not all(guards.values())) else "KEEP" if side == "up" else "TIE" if side == "inside" else "INCONCLUSIVE (seeds differ; goes to Mirko)"
    else:
        dens_ok = all(d >= density_cut for d in dens)
        side = "no_loss" if all(g >= -band for g in gain) else "loss"
        v = "CUT" if (side == "loss" or not all(guards.values())) else "KEEP" if dens_ok else "TIE (the density did not fall by %g: arm t stays)" % density_cut
    out = dict(verdict=v, kind=kind, band=band, side=side, t12_F1_1_gain_per_seed=gain, t1_R_1_gain_per_seed=r1, topo_F1_gain_per_seed=topo, nodes_per_100m_fall_per_seed=dens,
               vert100_dp_fall_per_seed=[1.0 - a["p"]["graph"]["vert100_dp"] / b["p"]["graph"]["vert100_dp"] for a, b in zip(cv, cc)], guards=guards,
               variant_numbers={k: mean(lambda c, k=k: c["p"]["tier"][k]) for k in ("t12_F1_1", "t12_R_1", "t12_P_1", "t1_R_1", "t1_R_1all", "z1_P_1", "z2_P_1", "z3_P_1")} | dict(
                   topo_F1=mean(lambda c: c["p"]["graph"]["topo_F1"]), junction_F1=mean(lambda c: c["p"]["graph"]["junction_F1"]), vert100_dp=mean(lambda c: c["p"]["graph"]["vert100_dp"])),
               variant=var, control=ctl, note="by hand: the written rule of the E276 card (section 0); the tier numbers are not in PROBLEM.md (D59 point 3); in-sample on VAL, no TEST")
    if verbose:
        print(json.dumps(out, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--run", required=True)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--n", type=int, default=0)
    s.add_argument("--exp", default=None, help="the experiment folder of the scoring card: runs/<exp>/<run>/ (default e273; required with --readouts, e.g. e275)")
    s.add_argument("--readouts", nargs="+", default=None, help="E275: read-out specs (plain@0.5 close:1.5@0.5 ...); one card per spec in runs/<exp>/<run>.<spec>/ (qgraph_readout.py)")
    s.add_argument("--curve", nargs="*", type=float, default=[], help="E275: radii of close-node suppression scored on the graph only (e >= 0.5) -> runs/<exp>/<run>.curve.json")
    s.add_argument("--dump", action="store_true", help="E275: write the candidate nodes of the exact-map pass to temp/<exp>/pred_<run>.pkl (scratch, for the analysis)")
    s.add_argument("--max-drop", type=float, default=0.20, help="E275 kill criterion: stop when a close-node spec (threshold >= 0.5) drops more than this share of the delivered nodes")
    s.add_argument("--repro", default=None, help="E275: the E274 sweep json of this model (temp/e274/sweep_<run>.json): close:1.5@0.5 must reproduce its pos1.5_e0.5 entry within 0.005")
    s.add_argument("--sparse", nargs=2, type=float, default=None, metavar=("EPS", "CHORD"), help="E276: score the node numbers against the SPARSE GT nodes the model was trained on (qtargets.node_table)")
    y = sub.add_parser("verdict276")
    y.add_argument("--kind", choices=("t", "s"), required=True, help="t: arm t against B scored on l4; s: arm s against arm t")
    y.add_argument("--var", nargs="+", required=True, help="the variant runs (seeds 0 and 1), cards in runs/<exp-var>/")
    y.add_argument("--ctl", nargs="+", required=True, help="the control runs (seeds 0 and 1), cards in runs/<exp-ctl>/")
    y.add_argument("--exp-var", default="e276")
    y.add_argument("--exp-ctl", default="e276")
    x = sub.add_parser("verdict275")
    x.add_argument("--var", nargs="+", required=True, help="the cards of the step, runs/<exp-var>/<name>/ (both seeds)")
    x.add_argument("--ctl", nargs="+", required=True, help="the control cards (both seeds)")
    x.add_argument("--exp-var", default="e275")
    x.add_argument("--exp-ctl", default="e273")
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
        if a.readouts and not a.exp:
            ap.error("--readouts needs --exp (e.g. e275): the default folder of the E273 cards must not receive read-out cards")
        a.exp = a.exp or "e273"
        if a.sparse:
            QE.NODE_SPARSE = (float(a.sparse[0]), float(a.sparse[1]))
        dump = os.path.join(ROOT, "temp", a.exp, f"pred_{a.run}.pkl") if a.dump else None
        c = score(a.run, a.seed, a.n, "cuda" if torch.cuda.is_available() else "cpu", exp=a.exp, readouts=a.readouts, curve=a.curve, dump=dump, max_drop=a.max_drop, repro=a.repro)
        if not a.readouts:
            print(open(os.path.join(ROOT, "runs", a.exp, a.run, "results.md"), encoding="utf-8").read())
    elif a.cmd == "verdict274":
        verdict274(a.var, a.ctl, a.exp_var, a.exp_ctl, a.metric, a.band)
    elif a.cmd == "verdict275":
        verdict275(a.var, a.ctl, a.exp_var, a.exp_ctl)
    elif a.cmd == "verdict276":
        verdict276(a.var, a.ctl, a.exp_var, a.exp_ctl, a.kind)
    else:
        verdict(a.runs)


if __name__ == "__main__":
    main()
