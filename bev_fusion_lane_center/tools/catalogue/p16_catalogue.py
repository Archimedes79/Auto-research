"""Loop 16: append the finished runs of the ledger to the experiment table (plan section 7(e)).
Writes out/loop2/e123_ladder/catalogue16.json = rows (theme, id, what, result, tag) and basis (id -> bb, k, sp); e2e_catalogue.py and e2e_basis.py read it at the end of their import (small loader block),
build_experiment_table.py then renders reports/Experiment_Table_EN.md.  Ids run on from E124 in the order of the launch times; a run row = one finished ledger row (control arm, noise runs and exploratory runs included, as 'C' / 'D' rows).
Tag rule: E = a design that is alive or won (learned, differentiable); D = cut, below the control after its rung, or a baseline that lost; C = control arm, noise run, analysis.
Usage (gcar): p16_catalogue.py"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p16_common as C  # noqa: E402
import p16_triage as T  # noqa: E402

BB = dict(base="BB01 BB03 BB22 BB23 BB24 BB26 BB29 BB30", tree="BB24 BB25", link="BB13 BB14", spf="BB27", rev="BB08 BB13", ctx="BB11 BB13", time="BB28 BB10")


def basis_of(r):
    bb = ["BB01", "BB03", "BB22", "BB23", "BB26", "BB29", "BB30"]
    sets = " ".join(r.get("sets", [])) + " " + " ".join(r.get("ws", []))
    if r["design"] in ("D3", "D4", "D1", "D2", "B3"):
        bb += ["BB24"]
    if r["design"] in ("D5", "D6", "D7", "D8", "B1", "H"):
        bb += ["BB13", "BB14"]
    if "spf2" in sets or "vc=" in sets:
        bb += ["BB27"]
    if "revisit" in sets or r["design"] in ("D2", "D3", "D5", "D8"):
        bb += ["BB08"]
    if "stage" in sets:
        bb += ["BB25"]
    if "temporal" in sets:
        bb += ["BB28", "BB10"]
    return dict(bb=sorted(set(bb)), k=r["basis"].split(), sp=["SP1", "SP4", "SP5"])


def main():
    s = T.load()
    cut = {}
    for rg, c in s["cuts"].items():
        for d, dec in c.get("decisions", {}).items():
            cut[d] = dec
    rows, basis = [], {}
    n = 124
    ctrl = T.control_table()
    rows.append(("T9", "E122", "Control arm R: the trace network E94 trained from the shared E94 stem at 1 / 2 / 4 / 8 units (1 u = 600 iterations of 16 frames), seeds 0 and 1, fixed VAL subset of 250 frames; the full E94 runs on the same subset",
                 "S (composite of the plan) " + ", ".join(f"{ctrl[(0, u)]['S']:.3f}" for u in (1, 2, 4, 8, 17) if ctrl.get((0, u))) + " (seed 0), seed 1 " + ", ".join(f"{ctrl[(1, u)]['S']:.3f}" for u in (1, 2, 4, 8, 17) if ctrl.get((1, u))) + "; the shared stem carries the skill, the curve is flat from 2 u; E94 F1@1 0.883 on this subset, 21.8 vertices per 100 m, J 0.39", "C"))
    basis["E122"] = dict(bb=["BB01", "BB11", "BB19", "BB21", "BB14"], k=["K16", "K15"], sp=["SP5"])
    runs = sorted(s["runs"].items(), key=lambda kv: kv[1].get("launched", ""))
    for rid, r in runs:
        c = T.card(r["name"])
        if not c or "metrics" not in c:
            continue
        m = c["metrics"]
        cap = T.card(r["name"] + ".m192")
        tag = "E"
        if r["type"] in ("N", "X"):
            tag = "C"
        elif r["design"] in ("B1", "B3"):
            tag = "D"
        elif cut.get(r["design"]) == "CUT":
            tag = "D"
        eid = f"E{n}"
        n += 1
        what = f"{rid} ({r['type']}): {r['change']}"
        res = f"S {m['S']:.3f}, F1@1 {m['F1@1.0']:.3f}, F1@0.5 {m['F1@0.5']:.3f}, J {m['J']:.3f}, {m['vert100']:.1f} vertices per 100 m, false length {m['false_len_frac']:.3f}, tau {m['tau']}; VAL subset of 250 frames, {r['u']} u"
        if cap and "metrics" in cap:
            res += f"; with beam 192 and 96 seeds S {cap['metrics']['S']:.3f}"
        rows.append(("T9", eid, what, res, tag))
        basis[eid] = basis_of(r)
    # loop 16b: validation-only diagnoses and refinements after the champion freeze (hand-written rows, numbers from out/loop2/e123_ladder/diag and the route-head folders)
    extra = [
        ("Route head on the champion's own training frames (in-sample) against validation: success 0.760 (1052 frames) against 0.798; control on out-of-fold training frames 0.945 against 0.881. Not an in-sample effect (tag C).", "C", ["BB11", "BB13"], ["K185"]),
        ("Route head with the token content set to zero (positions, cross sections, SD map stay): champion 0.803 / 0.797 / 0.793 / 0.784, control 0.904 / 0.891 / 0.885 / 0.870 (map error 0 / 1 / 2 / 3 m); the head does not read the node features (tag C).", "C", ["BB11", "BB13"], ["K185"]),
        ("Edge-level diagnosis of the route flagging (p16_route_diag2.py): the same miss rate of route edges in both graphs (13.0 % / 13.5 %), confident misses, edges next to a fork vertex missed 28.7 % (champion) against 18.4 %; mean missed edge 8.1 m against 4.6 m; coverage ceiling 0.972 -> 0.872 against 0.993 -> 0.936 (tag C).", "C", ["BB11"], ["K185"]),
        ("Route head asked along the edges (points every <= 4 m, interpolated tokens), two seeds, validation: champion 0.888 / 0.889 (exact), 0.881 / 0.886 (1 m), 0.886 / 0.883 (2 m), 0.883 / 0.876 (3 m); control with the same treatment 0.876 / 0.894; vertex-based 0.798 and 0.881 / 0.875. The route gap of the sparse graph is closed (tag E).", "E", ["BB11", "BB13", "BB10"], ["K185"]),
        ("Point spacing of the route attribute, champion, seed 0, validation: 8 m (98 points per frame) 0.872, 4 m (162) 0.888, 2 m (284) 0.899, vertex-based (68) 0.798 (tag C).", "C", ["BB11"], ["K185"]),
        ("Planner consumer on validation (3 seeds, exact map): path error at 20 m champion with the vertex head 0.31 m, with the head along the edges 0.26 m, control 0.27 m (tag E).", "E", ["BB11", "BB13"], ["K185"]),
        ("Relaxed child selection near predicted forks at inference (smaller dominance and non-maximum-suppression radii around junction peaks): S 0.750, J 0.278 - 0.279 against 0.749 / 0.278; oracle junction heat J 0.282: the second branch of a shallow fork is not in the tree (tag D).", "D", ["BB24", "BB26"], ["K183"]),
        ("Inference-time ensemble of two builders (champion f1 + replicate s2, node union): F1@1 0.871 -> 0.885, lanes found 0.695 -> 0.778, J 0.278 -> 0.177, S 0.749 -> 0.734; only confident nodes of s2: S 0.738 - 0.743 (tag D).", "D", ["BB24", "BB26"], ["K183"]),
        ("Hysteresis threshold in the assembly (nodes with validity 0.3 - 0.6 admitted when their parent is, tau 0.7 - 0.9): lanes found 0.695 -> 0.70 - 0.71, J 0.278 -> 0.257 - 0.271, S 0.745 - 0.748 against 0.749 (VAL250); plain threshold 0.9: S 0.754, J 0.308, lanes found 0.649 (a flat knob, tau stays 0.7): no gain (tag D).", "D", ["BB26"], ["K183"]),
        ("Fast runtime: builder in a CUDA graph (82 -> 19 ms under load; idle 69 -> 12 ms with the time recursion, 6 ms single frame), assembly vectorised 50 -> 20 ms under load (15 ms idle) with identical graphs, route head in padded CUDA graphs 6 -> 2 ms; chain 33 ms per frame (f1), 29 ms (j1) against 594 ms for the control (tag E).", "E", ["BB26"], ["K186", "K187"]),
        ("Quality against compute time (VAL250, idle GPU): j1 192 / 96 S 0.747 in 20 ms, j1 288 / 144 S 0.748 in 21 ms, f1 288 / 144 S 0.749 in 27 ms; below 192 / 96 recall collapses (96 / 48: S 0.70); the time recursion is a tie in S (tag C).", "C", ["BB28", "BB24"], ["K186"]),
    ]
    for what, tg, bbs, ks in extra:
        eid = f"E{n}"
        n += 1
        rows.append(("T9", eid, "Loop 16b (validation only, after the champion freeze): " + what.split(" (tag")[0].split(":")[0], what.split(":", 1)[1].rsplit(" (tag", 1)[0].strip() if ":" in what else what, tg))
        basis[eid] = dict(bb=bbs, k=ks, sp=["SP1", "SP4", "SP5"])
    json.dump(dict(theme=["T9", "Loop 16: fully differentiable lane-graph builders (growth and link), search ladder", "loop 16"], rows=rows, basis=basis), open(f"{C.LADDER}/catalogue16.json", "w"), indent=1)
    print("rows", len(rows))


if __name__ == "__main__":
    main()
