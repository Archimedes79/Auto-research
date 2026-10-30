"""t4_soa_front: compare detection records of several systems in the SAME front area (front-camera sector, x in [0, X_MAX) m,
half angle 32.3 deg) from their scene_records.pkl (det[scene][dt=0] = [(dets, gt boxes)]); AP over 0.5 / 1 / 2 / 4 m by
centre distance (evaluation.metrics.DetMeter), mAP2 = mean over vehicle and pedestrian, plus range bins and a paired scene
bootstrap of every system against the first one. Usage (bev_fusion folder):
  python temp/t4/compare_front.py <out.md> <x_max> <label=records.pkl> [<label=records.pkl> ...]
"""
import os
import pickle
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path[:0] = [ROOT]
from evaluation.metrics import DetMeter  # noqa: E402

out_md, x_max = sys.argv[1], float(sys.argv[2])
systems = [a.split("=", 1) for a in sys.argv[3:]]
THR = (0.5, 1, 2, 4)
CL = ("vehicle", "pedestrian")
N_BOOT = int(os.environ.get("BOOT", "100"))


def load(path):
    r = pickle.load(open(path, "rb"))["det"]
    return {s: d[0.0] for s, d in r.items()}


def meter(frames):
    m = DetMeter(classes=list(CL), x_max=x_max)
    for dets, gt in frames:
        m.add(dets, gt, dt=0.0)
    return m.result()[0.0]


def maps(res, binlab="all"):
    ap = {c: {t: res[c][binlab]["ap@%g" % t] for t in THR} for c in CL}
    return ap, float(np.mean([ap[c][t] for c in CL for t in THR]))


recs = {lab: load(p) for lab, p in systems}
scenes = sorted(set.intersection(*[set(r) for r in recs.values()]))
for lab, r in recs.items():
    assert set(r) == set(scenes), (lab, len(r), len(scenes))
    assert [len(r[s]) for s in scenes] == [len(recs[systems[0][0]][s]) for s in scenes], lab


def frames_of(lab, sc):
    return [f for s in sc for f in recs[lab][s]]


full = {lab: meter(frames_of(lab, scenes)) for lab in recs}
rng = np.random.default_rng(0)
draws = [list(rng.choice(len(scenes), len(scenes))) for _ in range(N_BOOT)]
boot = {lab: [] for lab in recs}
for dr in draws:
    sc = [scenes[i] for i in dr]
    for lab in recs:
        boot[lab].append(maps(meter(frames_of(lab, sc)))[1])
ref = systems[0][0]
n_frames = sum(len(recs[ref][s]) for s in scenes)
lines = [f"# Front-area comparison (x in [0, {x_max:g}) m, +-32.3 deg), cold VAL, {n_frames} frames of {len(scenes)} scenes", "",
         "AP by centre distance over 0.5 / 1 / 2 / 4 m (nuScenes style), mAP2 = mean of vehicle and pedestrian AP over the four distances; "
         f"interval of the difference to `{ref}` = paired scene bootstrap ({N_BOOT} draws, 2.5 / 97.5 %).", "",
         "| system | mAP2 | d vs ref [95% interval] | veh AP@0.5 | veh AP@1 | veh AP@2 | veh AP@4 | ped AP@0.5 | ped AP@1 | ped AP@2 | ped AP@4 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
for lab in recs:
    ap, m2 = maps(full[lab])
    d = np.array(boot[lab]) - np.array(boot[ref])
    dtxt = "-" if lab == ref else f"{m2 - maps(full[ref])[1]:+.3f} [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]"
    lines.append(f"| {lab} | {m2:.3f} | {dtxt} | " + " | ".join(f"{ap[c][t]:.3f}" for c in CL for t in THR) + " |")
lines += ["", "mAP2 by distance of the object (range bins of the metric):", "",
          "| system | " + " | ".join(f"{lo:g}-{min(hi, x_max):g} m" for lo, hi in ((0, 20), (20, 40), (40, 60)) if lo < x_max) + " |",
          "|---|" + "---|" * sum(1 for lo in (0, 20, 40) if lo < x_max)]
for lab in recs:
    cells = []
    for i, (lo, hi) in enumerate(((0, 20), (20, 40), (40, 60))):
        if lo < x_max:
            cells.append(f"{maps(full[lab], '%g-%g' % (lo, hi))[1]:.3f}" if "%g-%g" % (lo, hi) in full[lab]["vehicle"] else "n/a")
    lines.append(f"| {lab} | " + " | ".join(cells) + " |")
n_gt = {c: full[ref][c]["all"]["n_gt"] for c in CL}
lines += ["", f"GT boxes in the area: vehicle {n_gt['vehicle']}, pedestrian {n_gt['pedestrian']}."]
open(out_md, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
