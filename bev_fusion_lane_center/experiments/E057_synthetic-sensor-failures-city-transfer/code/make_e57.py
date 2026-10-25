"""Bericht E57 (Schleife 5): Robustheit gegen synthetische Sensorstoerungen (eingefrorenes Fundament + sauber trainierter Kopf).  Liest out/loop2/e57_<stoerung>/{metrics.json, prob_test.npy, ids_test.npy};
Text loop2/loop5_text_e57.py (TEXT57).  Aufruf (gcar): make_e57.py [figs|report]"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arch_figs5 as AF5  # noqa: E402
import dense_head as DH  # noqa: E402
from report2 import f3, make_report2  # noqa: E402

OUT = "/mnt/c/ML-Experiment/bev_fusion_lane_center/runs"
FIG = f"{OUT}/figs"
os.makedirs(FIG, exist_ok=True)
_ns = {}
if os.path.exists("/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/reports_build/loop5_text_e57.py"):
    exec(open("/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/reports_build/loop5_text_e57.py", encoding="utf-8").read(), _ns)
TEXT = _ns.get("TEXT57", {})

# (Ordner, Beschriftung, Referenz-Ordner, Beschreibung der Stoerung)
CONDS = [
    ("stored", "clean (reference)", "stored", "stored features of the 833 TEST samples"),
    ("cam_front_off", "front camera black", "stored", "CAM_FRONT image set to 0"),
    ("cam_front3_off", "3 forward cameras black", "stored", "CAM_FRONT_LEFT / FRONT / FRONT_RIGHT set to 0"),
    ("dark", "night (x 0.15 + noise)", "stored", "images x 0.15 plus Gaussian noise 0.02"),
    ("fog", "fog (contrast x 0.45 + haze)", "stored", "images x 0.45 + 0.43 haze"),
    ("blur", "blur (sigma 3 px)", "stored_s2", "Gaussian blur of all six images (every 2nd TEST sample)"),
    ("radar_off", "radar nearly off (3 points)", "stored_s2", "all but 3 radar points removed (every 2nd TEST sample)"),
    ("extr3", "camera extrinsics jitter 3 deg", "stored_s2", "random rotation sigma 3 deg per axis, camera and sample (every 2nd TEST sample)"),
]


def jl(p):
    return json.load(open(p)) if os.path.exists(p) else None


def mpath(folder):
    return f"{OUT}/e57_{folder}" if folder != "stored_s2" else f"{OUT}/e57_stored_s2"


def get(folder):
    d = jl(f"{mpath(folder)}/metrics.json")
    return d


def rows_table():
    rows = []
    cl = {"stored": get("stored"), "stored_s2": get("stored_s2")}
    for f, lab, ref, desc in CONDS:
        d = get(f)
        if not d:
            continue
        px = d["pixel"]["0.6"]
        g = d.get("graph", {})
        c = cl.get(ref)
        rel = f3(g["F1@1.0"] / c["graph"]["F1@1.0"], 2) if (c and g and "graph" in c) else "-"
        relr = f3(g["route_f"] / c["graph"]["route_f"], 2) if (c and g and "graph" in c) else "-"
        rows.append([lab, d["n"], f3(px["F1@1.0"]), f"{f3(px['P@1.0'])} / {f3(px['R@1.0'])}", f3(g.get("F1@1.0")), f3(g.get("route_f")), f3(g.get("lanes_found")), f"{f3(g.get('split6_F1'))} / {f3(g.get('merge6_F1'))}", f3(g.get("dir_acc")), f"{rel} / {relr}"])
    return rows


def bars():
    labs, gf, rf, ff = [], [], [], []
    cl = {"stored": get("stored"), "stored_s2": get("stored_s2")}
    for f, lab, ref, desc in CONDS:
        d = get(f)
        if not d or "graph" not in d:
            continue
        labs.append(lab.replace(" (", "\n("))
        gf.append(d["graph"]["F1@1.0"])
        rf.append(d["graph"]["route_f"])
        ff.append(d["graph"]["lanes_found"])
    fig, ax = plt.subplots(figsize=(13, 4.2))
    x = np.arange(len(labs))
    ax.bar(x - 0.27, gf, 0.26, label="point F1 @1 m", color="#1f77b4")
    ax.bar(x, rf, 0.26, label="route F", color="#2ca02c")
    ax.bar(x + 0.27, ff, 0.26, label="lanes found", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(labs, fontsize=7.5)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    ax.set_title("E57: system output (strict tracer on the e37_final heat map) under sensor perturbations - TEST", fontsize=10)
    fig.tight_layout()
    p = f"{FIG}/e57_bars.png"
    fig.savefig(p, dpi=100)
    plt.close(fig)
    return p


def qualitative(n=3):
    avail = [(f, lab) for f, lab, ref, desc in CONDS if os.path.exists(f"{mpath(f)}/prob_test.npy")]
    rows_all = [np.load(f"{mpath(f)}/ids_test.npy") for f, _ in avail]
    common = sorted(set(rows_all[0].tolist()).intersection(*[set(r.tolist()) for r in rows_all[1:]]))
    if len(common) < n:
        return None
    sel = [common[int(i)] for i in np.linspace(len(common) * 0.1, len(common) * 0.85, n)]
    gts = DH.load_gt_points("b123", "val", sel)
    fig, axs = plt.subplots(n, len(avail), figsize=(2.15 * len(avail), 2.25 * n), squeeze=False)
    for c, (f, lab) in enumerate(avail):
        prob = np.load(f"{mpath(f)}/prob_test.npy", mmap_mode="r")
        ids = np.load(f"{mpath(f)}/ids_test.npy").tolist()
        for r, row in enumerate(sel):
            ax = axs[r, c]
            ax.imshow(np.asarray(prob[ids.index(row)], np.float32).T, origin="lower", extent=[-50, 50, -50, 50], cmap="magma", vmin=0, vmax=1)
            ax.plot(gts[r][:, 0], gts[r][:, 1], ".", color="#00e5ff", ms=0.6, alpha=0.6)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(lab, fontsize=6.5)
    fig.suptitle("Heat map of the dense head (colour) with the ground-truth centerlines (cyan dots) for the same TEST samples under each perturbation", fontsize=8)
    fig.tight_layout()
    p = f"{FIG}/e57_qualitative.png"
    fig.savefig(p, dpi=90)
    plt.close(fig)
    return p


def main(mode):
    head = ["perturbation", "n", "pixel F1 @1 m (tau 0.6)", "P / R", "graph point F1 @1 m", "route F", "lanes found", "split / merge F1 @6 m", "direction acc.", "relative to clean: point F1 / route F"]
    rows = rows_table()
    if mode == "figs":
        for r in rows:
            print(r)
        return
    images = []
    for fn in (bars, qualitative):
        try:
            p = fn()
            if p:
                images.append((p, "System output under each perturbation (TEST)." if fn is bars else "Qualitative: the same TEST samples (rows) under each perturbation (columns) - the dense head's heat map against the ground truth. Dark images remove nearly all lanes; a lost front camera removes the forward sector; radar loss is mild, extrinsic jitter costs about 20 %, blur removes most of the thin lanes."))
        except Exception as ex:  # noqa: BLE001
            print(fn.__name__, "Fehler:", ex)
    t = TEXT
    make_report2("E57", t.get("title", "E57"), t.get("question", ""), t.get("setup", []), [("System under synthetic sensor perturbations (TEST; frozen GaussianCaR + e37_final trained on clean data only)", head, rows)], images, t.get("findings", ["(to be written)"]), t.get("limits", []), arch=AF5.robust_arch, arch_height=3.0)
    print("E57 written")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "figs")
