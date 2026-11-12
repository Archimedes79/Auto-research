"""scratch (Mirko 2026-11-10 'starte die diskutierten Experimente'; D79; E283 section 7 'next CPU step'): how much of the oracle gain of p19 survives when the ENDS are as noisy as the network's, with pairs decided from the noisy ends?
CPU, VAL 873, baseline B + hard close-node step 1.5 m as the delivered graph, one process per seed (argument 0 / 1). A design check, in-sample on VAL, by hand: no verdict on a model. An oracle is a sanity check, not an upper bound.
Stage 1 and the delivered graph are those of p19_clean_ends_oracle.py (executed from its source up to its stage 2); only the connectors change, wired as in p19.

Variants (all with ONE cubic Hermite curve, tangent length = chord, per pair with out-of-fold probability >= 0.5; the classifier is HistGradientBoosting on the pair geometry of p19 / pair_geometry.py, re-fitted on the NOISY
features of the other fold, 2 folds of whole physical junctions):
  V0  delivered                   the graph of B + hard step
  V3  clean ends, classifier      p19's V3 (reference: no end noise)
  N1  measured spread             every GT end moved along its lane by N(-0.45, 1.2) m toward the junction (measured, E283 section 5: median 0.2 - 0.6 m too far out, q25 .. q75 1.6 m wide, 9 - 18 % 2 m or more out),
                                  across by N(0, 0.3) m (median lateral error 0.12 - 0.15 m, section 5) and its heading turned by N(0, 6 deg) (median 3.5 deg, p90 11 deg, section 6)
  N2  double spread               the same with 2 x the standard deviations (the bias stays): sensitivity
  N3  N1 + missing ends           N1 and 25 % of the ends dropped (the flag is missing at 25 - 33 % of the GT ends, section 2); pairs through a dropped end are lost
  N4  N3 + false ends             N3 and a false end (position 2 - 6 m along the lane from a true end of the same side, heading N(0, 20 deg)) for 50 % of the remaining true ends of each side (half of the flagged nodes sit at
                                  no GT end, section 2); a pair that contains a false end is a false pair; the classifier sees them like true ends (the closest thing to the network the CPU can give)
Scored by the scorer's own functions (graph_counts / graph_numbers) as in p19.

PRE-REGISTERED RULE (written before the run; the rule of D79 for an end head, applied to the oracle): a variant HOLDS when against V0 on BOTH seeds: topo F1 > +0.02, F1@1.0 not lower than -0.01, dup_frac not higher than +0.02,
vertices per 100 m (Douglas-Peucker) not higher than +1.0, junction F1 not lower than -0.03 (lanes found and Q are reported, not judged).
  Reading: N4 holds -> an end head that is no better than the delivered flags would already pay: write the card, GPU in the next window. N4 fails, N3 or N1 holds -> the head must be better than the delivered flags (state the
  recall / precision it needs: the largest noise that still holds is the head's target); N1 fails -> the potential of the oracle came from exact ends: no GPU hour, the lever is closed (the card says so).
    wsl env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 bash tools/env/_py14.sh temp/e284/p20_noisy_ends_oracle.py <seed>"""
import json
import os
import pickle
import sys
import time

_src = open(os.path.join(os.environ["BFLC_ROOT"], "temp", "e284", "p19_clean_ends_oracle.py")).read()
_head = _src.split("# ------------------------------------------------------------------ stage 2")[0]
_tail = _src.split("# ------------------------------------------------------------------ stage 2")[1].split("dump = pickle.load")[0].split("\n", 1)[1]
exec(compile(_head, "p19_head", "exec"))      # stage 1: AN, fold, PROB (clean), frames, POSE, SPEC, QR, SEED, KEYS, hermite, allowed, sang, side, Counter, ...
exec(compile(_tail, "p19_tail", "exec"))      # tan_of, ang, build


def feats(ent, ext, cand):
    """the pair features of pair_geometry.analyse for ends given as (pos, tan) lists and a candidate list [(a, b)]"""
    E, TE = np.array([e[0] for e in ent]), np.array([e[1] for e in ent])
    X_, TX = np.array([x[0] for x in ext]), np.array([x[1] for x in ext])
    sa, sb = side(E, TE), side(X_, TX)
    na, nb = Counter(a for a, b in cand), Counter(b for a, b in cand)
    turn = np.array([sang(ent[a][1], ext[b][1]) for a, b in cand])
    L = np.array([np.linalg.norm(ext[b][0] - ent[a][0]) for a, b in cand])
    rk_turn_a, rk_len_a, rk_turn_b = np.zeros(len(cand)), np.zeros(len(cand)), np.zeros(len(cand))
    for a in set(x[0] for x in cand):
        idx = [i for i, (aa, bb) in enumerate(cand) if aa == a]
        for arr, vals in ((rk_turn_a, np.abs(turn[idx])), (rk_len_a, L[idx])):
            r_ = np.argsort(np.argsort(vals))
            arr[idx] = r_ / max(len(idx) - 1, 1)
    for b in set(x[1] for x in cand):
        idx = [i for i, (aa, bb) in enumerate(cand) if bb == b]
        r_ = np.argsort(np.argsort(np.abs(turn[idx])))
        rk_turn_b[idx] = r_ / max(len(idx) - 1, 1)
    F = []
    for i, (a, b) in enumerate(cand):
        pa, ta, pb, tb = ent[a][0], ent[a][1], ext[b][0], ext[b][1]
        ch = pb - pa
        na_ = np.array([-ta[1], ta[0]])
        nb_ = np.array([-tb[1], tb[0]])
        F.append([turn[i], abs(turn[i]), L[i], ch @ ta, ch @ na_, ch @ tb, ch @ nb_, sang(ta, ch), sang(tb, ch), sa[a][0], sa[a][1], sb[b][0], sb[b][1], na[a], nb[b], rk_turn_a[i], rk_len_a[i], rk_turn_b[i], len(ent), len(ext)])
    return np.array(F, float)


def rot(t, deg):
    c_, s_ = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    v = np.array([c_ * t[0] - s_ * t[1], s_ * t[0] + c_ * t[1]])
    return v / max(np.linalg.norm(v), 1e-9)


def perturb(r, rng, sd_along, sd_cross, sd_ang, p_miss, p_false):
    """noisy ends of one frame: -> ent2, ext2 (lists of (pos, tan)), orig_a, orig_b (index of the true end or -1 for a false end)"""
    out = []
    for side_, sgn in ((r["ent"], 1.0), (r["ext"], -1.0)):       # entry: toward the junction = +tan; exit: toward the junction = -tan
        ends, orig = [], []
        for i, (p, t) in enumerate(side_):
            if rng.random() < p_miss:
                continue
            off = rng.normal(-0.45, sd_along)
            n = np.array([-t[1], t[0]])
            ends.append((p + sgn * off * t + n * rng.normal(0.0, sd_cross), rot(t, rng.normal(0.0, sd_ang))))
            orig.append(i)
        for k in range(int(sum(rng.random() < p_false for _ in range(len(ends))))):
            j = rng.integers(len(ends)) if ends else None
            if j is None:
                break
            p, t = ends[j]
            ends.append((p + sgn * t * rng.uniform(-6.0, -2.0), rot(t, rng.normal(0.0, 20.0))))
            orig.append(-1)
        out.append((ends, orig))
    return out[0][0], out[1][0], out[0][1], out[1][1]


NOISE = {"N1 measured spread": (1.2, 0.3, 6.0, 0.0, 0.0), "N2 double spread": (2.4, 0.6, 12.0, 0.0, 0.0),
         "N3 N1 + missing ends": (1.2, 0.3, 6.0, 0.25, 0.0), "N4 N3 + false ends": (1.2, 0.3, 6.0, 0.25, 0.5)}
NV = {}                                    # variant -> {frame: (r2, pairs predicted)}
nstat = {}
for vi, (vname, par) in enumerate(NOISE.items()):
    cache, X_all, y_all, fold_all = {}, [], [], []
    for b in fr:
        rng = np.random.default_rng(10000 * (vi + 1) + b)
        ent2, ext2, oa, ob = perturb(AN[b], rng, *par)
        true_pairs = {AN[b]["cand"][i] for i in np.flatnonzero(AN[b]["y"] == 1)}
        cand2 = [(a, c_) for a in range(len(ent2)) for c_ in range(len(ext2)) if allowed(ent2[a][0], ent2[a][1], ext2[c_][0], ext2[c_][1])]
        if not cand2:
            continue
        y2 = np.array([(oa[a] >= 0 and ob[c_] >= 0 and (oa[a], ob[c_]) in true_pairs) for a, c_ in cand2], int)
        cache[b] = dict(r=dict(ent=ent2, ext=ext2), cand=cand2, y=y2, X=feats(ent2, ext2, cand2), n_true=len(true_pairs))
    P2 = {}
    for k_ in (0, 1):
        tr = [b for b in cache if fold[b] != k_]
        te = [b for b in cache if fold[b] == k_]
        Xt, yt = np.concatenate([cache[b]["X"] for b in tr]), np.concatenate([cache[b]["y"] for b in tr])
        clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, max_leaf_nodes=15, random_state=0).fit(Xt, yt)
        for b in te:
            P2[b] = clf.predict_proba(cache[b]["X"])[:, 1]
    yy = np.concatenate([cache[b]["y"] for b in cache])
    pp = np.concatenate([P2[b] for b in cache])
    ntrue = sum(cache[b]["n_true"] for b in cache) + 0
    nstat[vname] = dict(frames=len(cache), cand=int(len(yy)), true_in_cand=int(yy.sum()), auc=float(roc_auc_score(yy, pp)),
                        precision=float(((pp >= 0.5) & (yy == 1)).sum() / max((pp >= 0.5).sum(), 1)), recall_of_true_pairs=float(((pp >= 0.5) & (yy == 1)).sum() / max(sum(cache[b]["n_true"] for b in cache), 1)))
    print(f"{vname}: {nstat[vname]}", flush=True)
    NV[vname] = {b: (cache[b]["r"], [cache[b]["cand"][i] for i in np.flatnonzero(P2[b] >= 0.5)]) for b in cache}

dump = pickle.load(open(os.path.join(ROOT, "temp", "e276", f"predfull_e273_p_s{SEED}.pkl"), "rb"))
VAR = ["V0 delivered", "V3 clean ends, classifier"] + list(NOISE)
rec = {v: [] for v in VAR}
added = {v: [0, 0] for v in VAR}
for b, (f, d) in enumerate(zip(frames, dump)):
    pred = QR.apply(dict(nodes=dict(pos=d["pos"], dirv=d["dirv"], e=d["e"], flags=d["flags"], section=d["section"], s=d.get("s")),
                         links=dict(src=d["link_src"], dst=d["link_dst"], p=d["link_p"], long=d["link_long"], a=d["link_a"], b=d["link_b"])), SPEC)
    graph, bk = QE.readout(pred, 0.5, 0.5)
    rec["V0 delivered"].append(QE.graph_counts(graph, f))
    long_mask = np.asarray(pred["links"]["long"], bool)
    r = AN.get(b)
    jobs = {"V3 clean ends, classifier": (r, [r["cand"][i] for i in np.flatnonzero(PROB[b] >= 0.5)] if r is not None else [])}
    for vname in NOISE:
        jobs[vname] = NV[vname].get(b, (r, []))
    for v, (rr, pr) in jobs.items():
        raw, n_add, n_wired = build(graph.lines, graph.edges, graph.nodes, bk["kind"], bk["piece_links"], long_mask, rr, pr)
        added[v][0] += n_add
        added[v][1] += n_wired
        rec[v].append(QE.graph_counts(raw, f))
    f._lane = f._ref = None
    if (b + 1) % 100 == 0:
        print(f"  seed {SEED} scored {b + 1}/{len(frames)}  {time.time() - t0:.0f} s", flush=True)
num = {v: QE.graph_numbers(rec[v]) for v in VAR}
out = dict(seed=SEED, noise_stats=nstat, curves_added={v: dict(curves=added[v][0], wired_both_ends=added[v][1]) for v in VAR[1:]},
           numbers={v: {k: float(num[v][k]) for k in KEYS if k in num[v]} for v in VAR})
json.dump(out, open(os.path.join(ROOT, "temp", "e284", f"p20_noisy_ends_oracle_s{SEED}.json"), "w"), indent=1)
print(f"seed {SEED}: curves added (wired at both ends): " + ", ".join(f"{v.split(' ')[0]} {added[v][0]} ({added[v][1]})" for v in VAR[1:]))
print(f"{'':30s}" + "".join(f"{k:>12s}" for k in KEYS))
for v in VAR:
    print(f"{v:30s}" + "".join(f"{num[v][k]:12.4f}" for k in KEYS))
print("differences to V0:")
for v in VAR[1:]:
    print(f"{v:30s}" + "".join(f"{num[v][k] - num['V0 delivered'][k]:+12.4f}" for k in KEYS))


def holds(v):
    g = lambda k: num[v][k] - num["V0 delivered"][k]  # noqa: E731
    return g("topo_F1") > 0.02 and g("F1@1.0") >= -0.01 and g("dup_frac") <= 0.02 and g("vert100_dp") <= 1.0 and g("junction_F1") >= -0.03


for v in VAR[1:]:
    print(f"RULE seed {SEED}: {v} HOLDS: {holds(v)}")
