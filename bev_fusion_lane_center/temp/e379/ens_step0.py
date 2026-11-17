"""E379 step-0 measurement (scratch, CPU, no training): what does averaging two independently trained champion-recipe heads do? Heads on the same cache seed S: A = runs/e373_cb_s<S> (E373, head seed 2 / 3) and B = runs/e367_b25_s<S> (E367, head seed 0 / 1), both
the E365 head with none weight 0.25, 12000 steps. Their VAL dumps (`token_val.pkl`: p_none per token, the candidate links with p >= 0.05 with their handle scales) are averaged: p_none = the mean; a link's p = the mean over the two heads with 0 where a head wrote no candidate; the handle
scales = the mean over the heads that wrote the link. Decode and score as the card of the champion (the D115 cell: short 0.3 / long 0.2 / keep 0.6 / none gate off), the standard scorer on the 873 VAL frames; paired scene bootstrap of the ensemble against each single head.
usage: python ens_step0.py <seed>"""
import argparse
import os
import sys

import numpy as np

R = os.environ["BFLC_ROOT"]
sys.path.insert(0, os.path.join(R, "evaluation"))
import jtoken_eval as JE  # noqa: E402

S = int(sys.argv[1])
SPLIT = sys.argv[2] if len(sys.argv) > 2 else "val"        # val: the 873 frames; train: the every-3 sample of the TRAIN dumps (291 frames)
EVERY = 3 if SPLIT == "train" else 1
LT, LTL, KT = 0.3, 0.2, 0.6


def load(run):
    a = argparse.Namespace(run=run, model="model.pt", seed=S, split=SPLIT, every=EVERY, drop=0.0, drop_seed=0, max=0)
    return JE.read_dump(a, JE.RUNS, 0)


def mean_frame(fa, fb, K=512):
    out = dict(fa)
    out["pnone"] = ((fa["pnone"].astype(np.float64) + fb["pnone"].astype(np.float64)) / 2).astype(np.float32)
    out["ptype"] = ((fa["ptype"].astype(np.float32) + fb["ptype"].astype(np.float32)) / 2).astype(np.float16)
    acc = {}
    for f in (fa, fb):
        c = f["cand"]
        for s_, d_, p_, l_, a_, b_ in zip(c["src"], c["dst"], c["p"], c["long"], c["a"], c["b"]):
            acc.setdefault((int(s_), int(d_)), []).append((float(p_), bool(l_), float(a_), float(b_)))
    keys = sorted(acc)
    src = np.array([k[0] for k in keys], np.int16)
    dst = np.array([k[1] for k in keys], np.int16)
    p = np.array([sum(x[0] for x in acc[k]) / 2 for k in keys], np.float32)
    lg = np.array([acc[k][0][1] for k in keys], bool)
    a = np.array([np.mean([x[2] for x in acc[k]]) for k in keys], np.float32)
    b = np.array([np.mean([x[3] for x in acc[k]]) for k in keys], np.float32)
    out["cand"] = dict(src=src, dst=dst, p=p, long=lg, a=a, b=b)
    return out


def score(r, frs):
    fr = [JE.decode_variant(f, v, LT, KT, False, LTL) for f, v in zip(frs, r["valid"])]
    return JE.score_preds([JE.frame_prediction(t, f, "rec", KT) for t, f in zip(r["toks"], fr)], r["frames"], log=lambda *x, **k: None)


ra, rb = load(f"e373_cb_s{S}"), load(f"e367_b25_s{S}")
assert all(a["token"] == b["token"] and np.array_equal(a["cell"], b["cell"]) for a, b in zip(ra["fr"], rb["fr"])), "the two dumps are not the same tokens"
sa, sb = score(ra, ra["fr"]), score(rb, rb["fr"])
se = score(ra, [mean_frame(a, b) for a, b in zip(ra["fr"], rb["fr"])])
frames = ra["frames"]
print(f"seed {S}, D115 cell, {SPLIT.upper()} {len(frames)} frames: A = e373_cb_s{S}, B = e367_b25_s{S}, E = the mean of the two")
for k in JE.KEYS + ("junction_P", "junction_R"):
    print("  %-12s A %.4f  B %.4f  E %.4f | E - A %+.4f  E - B %+.4f  (A - B %+.4f)" % (k, sa["graph"][k], sb["graph"][k], se["graph"][k], se["graph"][k] - sa["graph"][k], se["graph"][k] - sb["graph"][k], sa["graph"][k] - sb["graph"][k]))
for name, ref in (("A", sa), ("B", sb)):
    p = JE.QE.paired(se["rec"], ref["rec"], frames)
    pj = JE.paired_junction(se["rec"], ref["rec"], frames)
    print("  paired E - %s:" % name, {k: {x: round(v, 4) for x, v in p[k].items() if x in ("diff", "lo", "hi")} for k in ("topo_F1", "F1@1.0", "lanes_found") if k in p}, "junction_F1", {x: round(v, 4) for x, v in pj["junction_F1"].items() if x in ("diff", "lo", "hi")})
p = JE.QE.paired(sa["rec"], sb["rec"], frames)
print("  paired A - B (the head-seed spread):", {k: {x: round(v, 4) for x, v in p[k].items() if x in ("diff", "lo", "hi")} for k in ("topo_F1", "lanes_found") if k in p})
