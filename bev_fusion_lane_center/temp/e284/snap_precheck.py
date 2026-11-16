"""scratch (Mirko 2026-11-08 'work alone, improve the extraction by error analysis and fixes'; the taxonomy err_taxonomy.py on 40 frames: the largest geometric loss of the delivered graph is OFFSET, GT points 1 - 2.5 m from a delivered point
(14.8 % of the GT length) and delivered points 1 - 2.5 m from the GT (7 - 16 %), while the pixel centre map p lies on the GT within 0.1 - 0.15 m): RIDGE SNAPPING, a read-out step that moves the delivered polylines onto the ridge of the head's own centre map.
CPU, the model of baseline B (runs/e273_p_s<seed>) on every 6th VAL frame, the hard close-node step 1.5 m (E275) and the read-out at 0.5, then three graphs scored by the E273 scorer on the SAME frames:
  none   the delivered graph as it is
  A      every polyline point moved along the local normal to the ridge of p (the p-weighted centroid within +-1.0 m of the profile maximum over +-2.0 m, only where the ridge is real: maximum >= 0.6 and >= 0.25 above the profile's
         lowest end value), the displacement median-filtered over 9 points (4 m) and capped at 2.0 m
  B      the same, but only for the polylines of the SECTION kind (the connector links of the junction mouth stay as delivered)
PRE-REGISTERED RULE (written before the run): proceed to seed 1 and a card when A or B raises F1@1.0 by >= 0.020 and lowers neither lanes found nor topology F1 by more than 0.005 on these frames; otherwise close the lever. In-sample on VAL, a design check, no verdict.
    wsl env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 bash tools/env/_py14.sh temp/e284/snap_precheck.py <seed> [--every 6] [--rows N]"""
import json
import os
import sys
import time

import numpy as np
from scipy.ndimage import map_coordinates, median_filter

ROOT = os.environ["BFLC_ROOT"]
for sub in ("training", "training/grid", "training/data", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import qgraph_eval as QE  # noqa: E402
import qgraph_readout as QR  # noqa: E402
import qpredict as QP  # noqa: E402
from lane_task import LaneGraph  # noqa: E402

SEED = int(sys.argv[1])
EVERY = int(sys.argv[sys.argv.index("--every") + 1]) if "--every" in sys.argv else 6
N_ROWS = int(sys.argv[sys.argv.index("--rows") + 1]) if "--rows" in sys.argv else 0
SPEC = QR.parse("close:1.5@0.5")
S = np.arange(-2.0, 2.01, 0.25)
KEYS = ("Q", "F1@1.0", "R@1.0", "P@1.0", "lanes_found", "topo_F1", "topo_P", "topo_R", "junction_F1", "junction_P", "junction_R", "vert100_dp")


def snap_line(line, pmap):
    n = len(line)
    if n < 3:
        return line, 0.0
    tan = np.gradient(line, axis=0)
    for _ in range(2):                                                    # a little smoothing of the tangent
        tan = np.c_[np.convolve(tan[:, 0], np.ones(5) / 5, "same"), np.convolve(tan[:, 1], np.ones(5) / 5, "same")]
    tn = np.linalg.norm(tan, axis=1, keepdims=True)
    tan = tan / np.where(tn > 1e-9, tn, 1.0)
    nrm = np.c_[-tan[:, 1], tan[:, 0]]
    xy = line[:, None, :] + S[None, :, None] * nrm[:, None, :]
    v = map_coordinates(pmap, [(xy[..., 0] + 49.5).ravel(), (xy[..., 1] + 49.5).ravel()], order=1, mode="nearest").reshape(n, len(S))
    am = v.argmax(1)
    vmax = v.max(1)
    lo = np.minimum(v[:, 0], v[:, -1])
    ok = (vmax >= 0.6) & (vmax - lo >= 0.25)
    w = np.where(np.abs(S[None, :] - S[am][:, None]) <= 1.0, v, 0.0)
    cen = (w * S[None]).sum(1) / (w.sum(1) + 1e-9)
    d = np.where(ok, np.clip(cen, -2.0, 2.0), 0.0)
    if n >= 9:
        d = median_filter(d, size=9, mode="nearest")
    return line + d[:, None] * nrm, float(np.abs(d).mean())


def snapped_graph(graph, bk, pmap, kinds):
    lines, moved = [], []
    for i, l in enumerate(graph.lines):
        l = np.asarray(l, float)
        if bk["kind"][i] in kinds:
            new, m = snap_line(l, pmap)
            lines.append(new)
            moved.append(m)
        else:
            lines.append(l)
    return LaneGraph(lines, graph.edges, True, graph.nodes, graph.meta), (float(np.mean(moved)) if moved else 0.0)


fn = QQ = QP.make_pred_fn(f"e273_p_s{SEED}", "exact", "cpu", keep_maps=True)
frames = QE.val_frames()[::EVERY]
if N_ROWS:
    frames = frames[:N_ROWS]
rec = {k: [] for k in ("none", "A", "B")}
moved_all = {"A": [], "B": []}
t0 = time.time()
for n, f in enumerate(frames):
    pred = fn(f.token)
    pmap = np.asarray(pred["maps"]["centre1"][0], float)
    q = QR.apply(pred, SPEC)
    graph, bk = QE.readout(q, 0.5, 0.5)
    rec["none"].append(QE.graph_counts(graph, f))
    for name, kinds in (("A", ("section", "link")), ("B", ("section",))):
        g2, mv = snapped_graph(graph, bk, pmap, kinds)
        rec[name].append(QE.graph_counts(g2, f))
        moved_all[name].append(mv)
    f._lane = f._ref = None
    if (n + 1) % 20 == 0:
        print(f"  frame {n + 1}/{len(frames)}  {time.time() - t0:.0f} s", flush=True)
num = {k: QE.graph_numbers(v) for k, v in rec.items()}
out = dict(seed=SEED, frames=len(frames), every=EVERY, numbers={k: {m: float(num[k][m]) for m in KEYS if m in num[k]} for k in num}, mean_displacement_m={k: float(np.mean(v)) for k, v in moved_all.items()})
json.dump(out, open(os.path.join(ROOT, "temp", "e284", f"snap_precheck_s{SEED}.json"), "w"), indent=1)
print(f"{len(frames)} frames, seed {SEED}")
print(f"{'':6s}" + "".join(f"{m:>12s}" for m in KEYS))
for k in ("none", "A", "B"):
    print(f"{k:6s}" + "".join(f"{num[k][m]:12.4f}" if m in num[k] else f"{'-':>12s}" for m in KEYS))
for k in ("A", "B"):
    print(f"{k} - none" + "".join(f"{num[k][m] - num['none'][m]:+12.4f}" if m in num[k] else f"{'-':>12s}" for m in KEYS), f"  (mean displacement {np.mean(moved_all[k]):.2f} m)")
g = lambda k, m: num[k][m] - num["none"][m]  # noqa: E731
passed = [k for k in ("A", "B") if g(k, "F1@1.0") >= 0.020 and g(k, "lanes_found") >= -0.005 and g(k, "topo_F1") >= -0.005]
print("RULE:", "passed by " + ", ".join(passed) + " -> proceed to seed 1 and a card" if passed else "not passed -> close the lever")
