"""E376 diagnostic (scratch, CPU): where do the champion's delivered links come from - sources WITH an in-class GT successor group G(i) (where a max-positive loss
acts) or sources with an EMPTY G(i) (where it does not)? On one stored VAL dump at the D115 cell (short 0.3 / long 0.2 / keep 0.6 / none gate off).
usage: bash tools/env/_py14.sh e376_emptyg.py <run> <seed>"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import jtoken_eval as E  # noqa: E402

run, seed = sys.argv[1], int(sys.argv[2])
a = argparse.Namespace(run=run, model="model.pt", seed=seed, split="val", every=1, drop=0.0, drop_seed=0)
r = E.read_dump(a, E.RUNS, 0)
ds, idx = r["ds"], r["idx"]
CL = ("short", "long")
links = {c: dict(nonempty=0, empty_assigned=0, empty_unassigned=0) for c in CL}           # delivered links by the source's in-class group
srcs = {c: dict(nonempty=[0, 0], empty_assigned=[0, 0], empty_unassigned=[0, 0]) for c in CL}   # kept sources: [count, sources emitting >= 2 links in the class]
in_g = {c: [0, 0] for c in CL}                                                              # links from non-empty-G sources: [dst in G(i), all]
junc = dict(points=0, with_empty_src_link=0)
gid_all = np.asarray(ds.c.f["gt_id"][idx])
for k, (i, f, v) in enumerate(zip(idx, r["fr"], r["valid"])):
    it = ds[int(i)]
    y = np.asarray(it["y"]) > 0
    yl = np.asarray(it["ylong"]).astype(bool)
    G = {"short": y & ~yl, "long": yl}
    gid = gid_all[k]
    fr = E.decode_variant(f, v, 0.3, 0.6, False, 0.2)
    keep, c = fr["keep"], fr["rec"]
    s, d, lg = c["src"].astype(int), c["dst"].astype(int), np.asarray(c["long"], bool)
    from_empty = np.zeros(len(s), bool)
    for ci, cl in enumerate(CL):
        m = lg if ci == 1 else ~lg
        ne = G[cl].any(1)
        grp = np.where(ne, 0, np.where(gid >= 0, 1, 2))                                      # 0 nonempty, 1 empty but assigned (an end), 2 empty and unassigned (duplicate / false)
        names = ("nonempty", "empty_assigned", "empty_unassigned")
        for gi, nm in enumerate(names):
            links[cl][nm] += int((grp[s[m]] == gi).sum())
            nout = np.bincount(s[m], minlength=len(keep))
            sel = keep & (grp == gi)
            srcs[cl][nm][0] += int(sel.sum())
            srcs[cl][nm][1] += int((sel & (nout >= 2)).sum())
        mm = m & ne[s]
        in_g[cl][0] += int(G[cl][s[mm], d[mm]].sum())
        in_g[cl][1] += int(mm.sum())
        from_empty |= m & ~ne[s]
    deg = np.bincount(s, minlength=len(keep)) + np.bincount(d, minlength=len(keep))
    jp = np.flatnonzero(deg >= 3)
    junc["points"] += len(jp)
    touched = set(s[from_empty].tolist()) | set(d[from_empty].tolist())
    junc["with_empty_src_link"] += sum(int(x) in touched for x in jp)
out = dict(run=run, seed=seed, frames=len(idx), cell="short 0.3 / long 0.2 / keep 0.6 / gate off", links=links, kept_sources=srcs, links_from_nonempty_into_G=in_g, token_degree3=junc,
           note="G(i) = the training label y split by ylong (lane_ptok_data); a link 'from empty' has an in-class empty G(i) at its source; token_degree3 = kept tokens with in + out degree >= 3 in the delivered token graph (not the scorer's junction points)")
print(json.dumps(out, indent=1))
os.makedirs(os.path.join(ROOT, "temp", "e376", "out"), exist_ok=True)
json.dump(out, open(os.path.join(ROOT, "temp", "e376", "out", f"emptyg_{run}.json"), "w"), indent=1)
