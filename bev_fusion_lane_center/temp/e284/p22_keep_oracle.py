"""scratch (E283 section 8 follow-up of p20, same day; D79): p20 REPLACED every delivered long link by the curves of the (noisy) end head, which punishes a head with gaps (N3: 25 % ends missing, N4: + false ends) more than a
real system would be: the natural integration keeps the delivered long links that the head does not cover. CPU, VAL 873, both seeds (argument 0 / 1), baseline B + hard step 1.5 m; the noisy ends, the pair classifier (re-fitted on the
noisy features, out-of-fold) and the wiring are those of p20 (executed from its source). By hand: a design check, no verdict on a model.
Variants (p20's noisy ends, curves for pairs with out-of-fold probability >= 0.5, wired as in p19):
  K1  N1 ends   + keep   the delivered long-link pieces are KEPT unless covered: a delivered long link is covered when an added curve starts within 4 m of its first point and ends within 4 m of its last point
  K3  N3 ends   + keep   (25 % of the ends missing)
  K4  N4 ends   + keep   (25 % missing and false ends: the closest the CPU can get to a real head)
  The replace variants of p20 (N1, N3, N4) are scored again next to them for the same frames.
PRE-REGISTERED RULE (written before the run; the rule of D79 / p20): a variant HOLDS when against V0 on BOTH seeds: topo F1 > +0.02, F1@1.0 not lower than -0.01, dup_frac not higher than +0.02, vertices per 100 m (DP) not higher than
+1.0, junction F1 not lower than -0.03.
  Reading: K4 holds -> the card of the end head with the keep integration, GPU in the window; K4 fails, K3 holds -> the head must keep the end recall of 75 % and cut the false ends (the card says so; a GPU diagnostic only);
  K3 fails -> the lever is closed for the GPU: the end head pays only with end recall far above what the node flags give today. K1 alone holding is the p20 N1 result again (exact-ish ends needed).
    wsl env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 bash tools/env/_py14.sh temp/e284/p22_keep_oracle.py <seed>"""
import json
import os
import pickle
import sys
import time

_src = open(os.path.join(os.environ["BFLC_ROOT"], "temp", "e284", "p20_noisy_ends_oracle.py")).read()
_head = _src.split("dump = pickle.load(")[0]
sys.argv = [sys.argv[0], sys.argv[1]]
exec(compile(_head, "p20_head", "exec"))     # stage 1 of p19, feats / perturb / NOISE / NV (noisy ends and out-of-fold pairs of N1 .. N4), build, QE, QR, SPEC, frames, KEYS, time


def build_keep(lines, edges, nodes, kinds, links, long_mask, r, pairs):
    """like p19's build, but a delivered long-link piece is dropped only when an added curve covers it (start within 4 m of its first point, end within 4 m of its last point)"""
    curves = []
    for a, b in pairs:
        pa, ta = r["ent"][a]
        pb, tb = r["ext"][b]
        curves.append((pa, ta, pb, tb))
    keep = []
    for i, k in enumerate(kinds):
        if k == "link" and bool(long_mask[links[i][0]]):
            P = np.asarray(lines[i], float)
            if any(np.linalg.norm(P[0] - pa) <= 4.0 and np.linalg.norm(P[-1] - pb) <= 4.0 for pa, ta, pb, tb in curves):
                continue
        keep.append(i)
    new = {old: j for j, old in enumerate(keep)}
    L2 = [np.asarray(lines[i], float) for i in keep]
    N2 = [list(nodes[i]) for i in keep]
    E2 = [(new[a], new[b]) for a, b in edges if a in new and b in new]
    ends = [(P[-1], tan_of(P, True), P[0], tan_of(P, False)) for P in L2]
    wired = 0
    for a, b in pairs:
        pa, ta = r["ent"][a]
        pb, tb = r["ext"][b]
        m = len(L2)
        L2.append(hermite(pa, ta, pb, tb, 1.0, 0.25))
        N2.append([10000 + a, 20000 + b])
        w_in = w_out = False
        for i, (le, te, fs, ts) in enumerate(ends):
            if np.linalg.norm(le - pa) <= 4.0 and ang(te, ta) <= 45.0:
                E2.append((i, m))
                w_in = True
            if np.linalg.norm(fs - pb) <= 4.0 and ang(ts, tb) <= 45.0:
                E2.append((m, i))
                w_out = True
        wired += int(w_in and w_out)
    return dict(pts=L2, edges=E2, nodes=N2), len(pairs), wired


dump = pickle.load(open(os.path.join(ROOT, "temp", "e276", f"predfull_e273_p_s{SEED}.pkl"), "rb"))
MAP = {"K1 N1 + keep": "N1 measured spread", "K3 N3 + keep": "N3 N1 + missing ends", "K4 N4 + keep": "N4 N3 + false ends"}
VAR = ["V0 delivered"] + list(MAP) + list(MAP.values())
rec = {v: [] for v in VAR}
added = {v: [0, 0] for v in VAR}
for b, (f, d) in enumerate(zip(frames, dump)):
    pred = QR.apply(dict(nodes=dict(pos=d["pos"], dirv=d["dirv"], e=d["e"], flags=d["flags"], section=d["section"], s=d.get("s")),
                         links=dict(src=d["link_src"], dst=d["link_dst"], p=d["link_p"], long=d["link_long"], a=d["link_a"], b=d["link_b"])), SPEC)
    graph, bk = QE.readout(pred, 0.5, 0.5)
    rec["V0 delivered"].append(QE.graph_counts(graph, f))
    long_mask = np.asarray(pred["links"]["long"], bool)
    for kv, nv in MAP.items():
        rr, pr = NV[nv].get(b, (AN.get(b), []))
        raw, n_add, n_w = build_keep(graph.lines, graph.edges, graph.nodes, bk["kind"], bk["piece_links"], long_mask, rr, pr)           # rr may be None when pr is empty (r is read only per pair)
        added[kv][0] += n_add
        added[kv][1] += n_w
        rec[kv].append(QE.graph_counts(raw, f))
        raw2, n2, w2 = build(graph.lines, graph.edges, graph.nodes, bk["kind"], bk["piece_links"], long_mask, rr, pr)
        added[nv][0] += n2
        added[nv][1] += w2
        rec[nv].append(QE.graph_counts(raw2, f))
    f._lane = f._ref = None
    if (b + 1) % 100 == 0:
        print(f"  seed {SEED} scored {b + 1}/{len(frames)}  {time.time() - t0:.0f} s", flush=True)
num = {v: QE.graph_numbers(rec[v]) for v in VAR}
out = dict(seed=SEED, curves_added={v: dict(curves=added[v][0], wired_both_ends=added[v][1]) for v in VAR[1:]}, numbers={v: {k: float(num[v][k]) for k in KEYS if k in num[v]} for v in VAR})
json.dump(out, open(os.path.join(ROOT, "temp", "e284", f"p22_keep_oracle_s{SEED}.json"), "w"), indent=1)
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
