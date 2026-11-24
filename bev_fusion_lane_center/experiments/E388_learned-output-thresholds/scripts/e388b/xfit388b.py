"""E388b reading (ii) (D126 item 2; diagnostic, by hand, NO verdict, the E388 NO stands), CPU, scratch: a scene-split cross-fit of the E388 surrogate ON VAL.
The VAL scenes of the run's live dump are split BEFORE any fit by the parity of the scene's rank among the sorted distinct scene ids (rank even = half A, odd = half B).
For each direction (fit on A, read B; fit on B, read A): the three taus by `jthresh.fit_taus` (the E388 surrogate and config: neutral start 0.5, T 0.05, lr 0.05, 400 steps, lowest loss, links floored at 0.05,
3 decimals) on the labelled VAL live dump's frames of the fit half; then the REAL scorer (`jtoken_eval.score_cell` on the stored VAL live dump token_val_live.pkl) at the fitted cell and at the hand cell
0.3 / 0.2 / 0.6 / off, on the frames of the read half; the paired scene bootstrap (`qgraph_eval.paired` + `paired_junction`) on that half.
H2 (the surrogate is the wrong target) predicts: the taus run to the 0.05 floor on VAL halves too and the fitted cell loses to the hand cell on the held-out half.
usage: python xfit388b.py <seed>  -> temp/e388b/xfit_s<S>.json"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

R = os.environ["BFLC_ROOT"]
for sub in ("evaluation", "training", "training/grid", "training/data", "training/train", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(R, sub)))
import jthresh as JT  # noqa: E402
import jtoken_eval as E  # noqa: E402

S = int(sys.argv[1])
run = f"e372_s4_s{S}"
out_path = os.path.join(R, "temp", "e388b", f"xfit_s{S}.json")
assert not os.path.exists(out_path), f"never overwritten: {out_path}"
a = argparse.Namespace(run=run, seed=S, split="val", every=1, drop=0.0, drop_seed=0, live=True)
r = E.read_dump(a, E.RUNS, 0)
frames = r["frames"]
with open(JT.label_path(E.RUNS, run, "val", 1), "rb") as fh:
    lab = pickle.load(fh)
assert [f["token"] for f in lab["frames"]] == [f.token for f in frames] == [f["token"] for f in r["fr"]], "the labelled dump and the live dump are not the same VAL frames in order"
scenes = np.array([f.scene for f in frames])
uniq = np.unique(scenes)
half = {sc: int(i % 2) for i, sc in enumerate(uniq)}                           # 0 = A (even rank), 1 = B (odd rank)
h = np.array([half[sc] for sc in scenes])
idx = {0: np.nonzero(h == 0)[0], 1: np.nonzero(h == 1)[0]}
print(f"{run}: VAL {len(frames)} frames, {len(uniq)} scenes; half A {len(idx[0])} frames ({int(np.sum(np.arange(len(uniq)) % 2 == 0))} scenes), half B {len(idx[1])} frames", flush=True)

HAND = (0.3, 0.2, 0.6, False)
fits = {}
for k, name in ((0, "A"), (1, "B")):
    D = JT.flatten([lab["frames"][i] for i in idx[k]])
    taus, step, curve, raw = JT.fit_taus(D)
    fits[name] = dict(taus=taus, raw=raw, step=step, gt_links_without_candidate_share=round(float(D["miss"].sum()) / max(sum(len(lab["frames"][i]["gt"]["src"]) for i in idx[k]), 1), 4),
                      token_f1_learned=JT.hard_f1(D, taus), token_f1_hand=JT.hard_f1(D, HAND[:3]))
    print(f"fit on half {name}: taus {taus} (raw {[round(x, 4) for x in raw]}, step {step}); token F1 learned {fits[name]['token_f1_learned']} hand {fits[name]['token_f1_hand']}", flush=True)

cells = {"hand": HAND, "fit_A": (*fits["A"]["taus"], False), "fit_B": (*fits["B"]["taus"], False)}
res = {}
for name, cell in cells.items():
    print(f"scoring {name} {cell} on all {len(frames)} VAL frames", flush=True)
    res[name] = E.score_cell(r, cell, frames)

KEYS = ("topo_F1", "topo_P", "topo_R", "F1@1.0", "lanes_found", "dup_frac", "Q", "junction_F1", "junction_P", "junction_R")


def sub(rec, ii):
    return dict(graph=[rec["graph"][i] for i in ii])


reads = {}
for fit_name, read_k, read_name in (("fit_A", 1, "B"), ("fit_B", 0, "A")):
    ii = idx[read_k]
    fr = [frames[i] for i in ii]
    ra, rb = sub(res[fit_name]["rec"], ii), sub(res["hand"]["rec"], ii)
    p = E.QE.paired(ra, rb, fr)
    p.update(E.paired_junction(ra, rb, fr))
    reads[f"{fit_name}_on_{read_name}"] = {k: {kk: float(p[k][kk]) for kk in ("a", "b", "diff", "lo", "hi")} for k in KEYS if k in p}
    t = reads[f"{fit_name}_on_{read_name}"]
    print(f"{fit_name} read on half {read_name} ({len(ii)} frames), fitted minus hand: " + ", ".join(f"{k} {t[k]['diff']:+.4f} [{t[k]['lo']:+.4f}, {t[k]['hi']:+.4f}]" for k in ("topo_F1", "junction_F1", "lanes_found", "dup_frac") if k in t), flush=True)

doc = dict(run=run, seed=S, split="val", frames=len(frames), scenes=len(uniq), split_rule="scene rank parity among the sorted distinct scene ids (even = A, odd = B), written before any fit",
           halves=dict(A=len(idx[0]), B=len(idx[1])), config=dict(T=JT.T, lr=JT.LR, steps=JT.STEPS, floor=JT.FLOOR, init=0.5), fits=fits, cells={k: list(v) for k, v in cells.items()}, reads=reads,
           label="diagnostic, by hand, no verdict (E388 NO stands)")
with open(out_path, "w") as fh:
    json.dump(doc, fh, indent=1)
print(f"written {os.path.relpath(out_path, R)}")
