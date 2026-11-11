"""Where does a junction pair head fail? Splits of the end and pair recall of a head's dump by the attributes of the ground truth (E300; what might be overlooked when only one number is read).

    wsl bash tools/env/_py14.sh jsplits.py score --run e284_h2_s0 [--run e287_p2_s0 ...]        CPU, VAL 873, from runs/<run>/ends_val.pkl (`qends_eval.py dump` / `jends_eval.py dump`)
    wsl bash tools/env/_py14.sh jsplits.py selftest

Per GT pair: delivered = a keypoint of the dump matches its entry and one its exit (within 4 m and 30 degrees) and the head's logit of that keypoint pair is >= 0 (p >= 0.5, allowed). Per GT end: found = a valid keypoint matches it.
Splits (rows of the printed tables): the ego drove the pair (tier 1, non-causal label) or not; city (boston / singapore: the traffic side differs); lanes of the entry arm (1, 2, 3+); the turn class of the pair (straight < 35 degrees, left, right,
other); for the ends: painted (stop line / crossing near, `qjunction.painted_flags`) or not, visible (camera sees it, no box hides it: the l4 label `vis`) or not, and the distance from the ego (< 20, 20 - 35, > 35 m).
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for sub in ("evaluation", "training", "training/grid", "training/data", "training/data/prep", "tools/viz", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub)))
import jgt as J  # noqa: E402
import jrule_control as JR  # noqa: E402


def turn_class(t):
    return "straight" if abs(t) < 35 else ("left" if 35 <= t <= 150 else ("right" if -150 <= t <= -35 else "other"))


def rng_class(d):
    return "<20 m" if d < 20 else ("20-35 m" if d < 35 else ">35 m")


def score(a):
    import junction_paint as JP
    import qgraph_view as QV
    import qjunction as QJ
    out = {}
    for run in a.run:
        frames = {f["token"]: f for f in pickle.load(open(os.path.join(ROOT, "runs", run, "ends_val.pkl"), "rb"))}
        P, E = {}, {}                                                   # split value -> [n, hit]
        def add(D, key, hit):
            d = D.setdefault(key, [0, 0])
            d[0] += 1
            d[1] += int(hit)
        for sid, (name, toks) in QV.val_scenes().items():
            for t in toks:
                if t not in frames:
                    continue
                g = QV.load_gt(t)
                jg = J.junction_gt(g)
                if jg is None:
                    continue
                f = frames[t]
                K = f["logit"].shape[0]
                kp, kd = f["kp_pos"], f["kp_dir"]
                ia = JR._match(kp[:K], kd[:K], f["valid"][0], jg["ent"])
                ib = JR._match(kp[K:], kd[K:], f["valid"][1], jg["ext"])
                try:
                    pe, px = QJ.painted_flags(jg, JP.load(t))
                except FileNotFoundError:
                    pe, px = np.zeros(len(jg["ent"]), bool), np.zeros(len(jg["ext"]), bool)
                city = "boston" if "boston" in str(g.get("location", "")) else "singapore"
                for r in jg["pairs"]:
                    i, j = int(r[0]), int(r[1])
                    hit = ia[i] >= 0 and ib[j] >= 0 and bool(f["allowed"][ia[i], ib[j]]) and f["logit"][ia[i], ib[j]] >= 0
                    add(P, ("all", ""), hit)
                    add(P, ("driven", "yes" if r[9] else "no"), hit)
                    add(P, ("city", city), hit)
                    add(P, ("lanes of the entry arm", str(min(int(r[6]), 3)) if r[6] < 3 else "3+"), hit)
                    add(P, ("turn", turn_class(r[8])), hit)
                for kind, EE, M, fl, vis in (("entry", jg["ent"], ia, pe, jg["ent_vis"]), ("exit", jg["ext"], ib, px, jg["ext_vis"])):
                    for q, e in enumerate(EE):
                        found = M[q] >= 0
                        add(E, (kind, "all", ""), found)
                        add(E, (kind, "painted", "yes" if fl[q] else "no"), found)
                        add(E, (kind, "visible", "yes" if vis[q] else "no"), found)
                        add(E, (kind, "range", rng_class(np.linalg.norm(e[:2]))), found)
                        add(E, (kind, "city", city), found)
        res = dict(pairs={f"{k[0]}: {k[1]}".strip(": "): dict(n=v[0], recall=v[1] / max(v[0], 1)) for k, v in sorted(P.items())},
                   ends={f"{k[0]} {k[1]}: {k[2]}".strip(": "): dict(n=v[0], recall=v[1] / max(v[0], 1)) for k, v in sorted(E.items())})
        out[run] = res
        print(run)
        for k, v in res["pairs"].items():
            print(f"  pair recall  {k:32s} n {v['n']:6d}  {v['recall']:.3f}")
        for k, v in res["ends"].items():
            print(f"  end found    {k:32s} n {v['n']:6d}  {v['recall']:.3f}")
    os.makedirs(os.path.join(ROOT, "runs", "e300"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "runs", "e300", "splits_" + "_".join(a.run) + ".json"), "w"), indent=1)


def selftest():
    assert turn_class(10) == "straight" and turn_class(90) == "left" and turn_class(-90) == "right" and turn_class(170) == "other"
    assert rng_class(10) == "<20 m" and rng_class(30) == "20-35 m" and rng_class(40) == ">35 m"
    print("jsplits selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("score")
    s.add_argument("--run", nargs="+", required=True)
    sub.add_parser("selftest")
    a = ap.parse_args()
    selftest() if a.cmd == "selftest" else score(a)
