"""Loop 17: out-of-fold summary of a fold-wise fine-tune (p17_ft.py / p17_eval.py --out <prefix>_k0, _k1): for every frame the graph of the run whose training did NOT contain its scene.
Prints and appends (tables/<md>) one row per series variant: failure-phase and clean-phase S, F1@1, J, vertices, lanes found, with a paired scene bootstrap of F1@1 against a reference run (--ref <out>/<tag>).
Usage (gcar): p17_merge.py --prefix e133b --perts dark,blur,cam_front3_off --variants single,maxdecay_0.95,ema_hold_mon_0.5 --ref e131_hold --ref0 e130_champion --md tables/e133b.md --folds 2"""
import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p16_common as C  # noqa: E402
import p17_compare as CMP  # noqa: E402

E17 = CMP.E17


def oof(prefix, tag, folds):
    runs = [pickle.load(open(f"{E17}/{prefix}_k{k}/{tag}.pkl", "rb")) for k in range(folds)]
    scenes = sorted(set(int(s) for s in runs[0]["scene"]))
    held = {s: k % folds for k, s in enumerate(scenes)}  # scene -> the fold that held it out (= the run to take)
    n = len(runs[0]["rows"])
    graphs, comp = [None] * n, [None] * n
    for i in range(n):
        k = held[int(runs[0]["scene"][i])]
        graphs[i], comp[i] = runs[k]["graphs"][i], runs[k]["comp"][i]
    return dict(rows=np.asarray(runs[0]["rows"]), scene=np.asarray(runs[0]["scene"]), fail=np.asarray(runs[0]["phase"]) == 1, graphs=graphs, comp=np.stack(comp))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--perts", default="dark,blur,cam_front3_off")
    ap.add_argument("--variants", default="single,maxdecay_0.95,ema_hold_mon_0.5")
    ap.add_argument("--ref", default="e131_hold", help="reference run for the paired bootstrap (variants other than single)")
    ap.add_argument("--ref0", default="e130_champion", help="reference run for the variant single")
    ap.add_argument("--folds", type=int, default=2)
    ap.add_argument("--md", default="")
    a = ap.parse_args()
    lines = ["| series | phase | n | S | F1@1 | J | vertices / 100 m | lanes found | F1@1 vs reference (paired scene bootstrap) |", "|---|---|---|---|---|---|---|---|---|"]
    for pn in a.perts.split(","):
        for v in a.variants.split(","):
            tag = f"{pn}_{v}"
            d = oof(a.prefix, tag, a.folds)
            cs = CMP.cards(d)
            refdir = a.ref0 if v == "single" else a.ref
            rr = pickle.load(open(f"{E17}/{refdir}/{tag}.pkl", "rb"))
            ref = dict(rows=np.asarray(rr["rows"]), scene=np.asarray(rr["scene"]), fail=np.asarray(rr["phase"]) == 1, graphs=rr["graphs"], comp=rr["comp"])
            b0, lo, hi, nf, ns = CMP.boot(d, ref)
            for nm in ("cl", "fail"):
                c = cs[nm]
                n = int((d["fail"] if nm == "fail" else ~d["fail"]).sum())
                extra = f"{b0:+.3f} [{lo:+.3f}, {hi:+.3f}] ({ns} scenes)" if nm == "fail" else ""
                lines.append(f"| {tag} | {nm} | {n} | {c['S']:.3f} | {c['F1@1.0']:.3f} | {c['J']:.3f} | {c['vert100']:.1f} | {c['lanes_found']:.2f} | {extra} |")
            pickle.dump(d, open(f"{E17}/{a.prefix}_oof_{tag}.pkl", "wb"))
    txt = "\n".join(lines)
    print(txt)
    if a.md:
        os.makedirs(os.path.dirname(f"{E17}/{a.md}"), exist_ok=True)
        open(f"{E17}/{a.md}", "a", encoding="utf-8").write(txt + "\n\n")


if __name__ == "__main__":
    main()
