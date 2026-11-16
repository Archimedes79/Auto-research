"""scratch (continuation of err_taxonomy.py): the taxonomy counted every GT lane point alike. The scorer's tier numbers (E276, the l4 labels: tier 1 / 2 / 3 = how much the lane matters for the car, vis = a camera sees it or it is near and unhidden) say
where the loss sits in terms of RELEVANCE. Baseline B + the hard close-node step 1.5 m (the E275 read-out), full dump, VAL 873: the pooled tier numbers t<k>_R_1 (recall of the VISIBLE GT points of tier k within 1.0 m), t<k>_R_1all (all points of tier k),
z<k>_P_1 (precision of the delivered points whose nearest GT point within 2.5 m has tier k), t12_R_1 / t12_P_1 / t12_F1_1 (tiers 1 + 2, the primary numbers of the E276 card), the visible metres per tier. Plus the headline numbers of the same graphs.
    wsl env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 bash tools/env/_py14.sh temp/e284/tier_view.py <seed> [--spec close:1.5@0.5]"""
import json
import os
import pickle
import sys

import numpy as np

ROOT = os.environ["BFLC_ROOT"]
for sub in ("training", "training/grid", "training/data", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import qgraph_eval as QE  # noqa: E402
import qgraph_readout as QR  # noqa: E402

SEED = int(sys.argv[1])
SPECTXT = sys.argv[sys.argv.index("--spec") + 1] if "--spec" in sys.argv else "close:1.5@0.5"
SPEC = QR.parse(SPECTXT)
frames = QE.val_frames()
dump = pickle.load(open(os.path.join(ROOT, "temp", "e276", f"predfull_e273_p_s{SEED}.pkl"), "rb"))
cs = []
for n, (f, d) in enumerate(zip(frames, dump)):
    pred = dict(nodes=dict(pos=d["pos"], dirv=d["dirv"], e=d["e"], flags=d["flags"], section=d["section"], s=d.get("s")),
                links=dict(src=d["link_src"], dst=d["link_dst"], p=d["link_p"], long=d["link_long"], a=d["link_a"], b=d["link_b"]))
    graph, bk = QE.readout(QR.apply(pred, SPEC), 0.5, 0.5)
    cs.append(QE.graph_counts(graph, f))
    f._lane = f._ref = None
    if (n + 1) % 200 == 0:
        print(f"  frame {n + 1}/{len(frames)}", flush=True)
g = QE.graph_numbers(cs)
tc = dict(rec=np.sum([c["tier"]["rec"] for c in cs if "tier" in c], 0), prec=np.sum([c["tier"]["prec"] for c in cs if "tier" in c], 0))
t = QE.tier_numbers(tc)
out = dict(seed=SEED, spec=SPECTXT, headline={k: float(g[k]) for k in QE.HEADLINE if k in g}, tiers={k: float(v) for k, v in t.items()})
json.dump(out, open(os.path.join(ROOT, "temp", "e284", f"tier_view_s{SEED}.json"), "w"), indent=1)
print("headline:", json.dumps({k: round(v, 3) for k, v in out["headline"].items()}))
for k in (1, 2, 3):
    print(f"tier {k}: visible GT {t[f't{k}_len']:8.0f} m  recall of the visible points {t[f't{k}_R_1']:.3f}  recall of all points {t[f't{k}_R_1all']:.3f}")
for k in (1, 2, 3):
    print(f"zone {k}: delivered {t[f'z{k}_n']:8.0f} m  precision {t[f'z{k}_P_1']:.3f}")
print(f"zone 0 (delivered points with no GT lane within 2.5 m): {t['z0_n']:.0f} m")
print(f"tiers 1 + 2: R {t['t12_R_1']:.3f}  P {t['t12_P_1']:.3f}  F1 {t['t12_F1_1']:.3f}")
