"""scratch (2026-11-17, derived from render_q1115.py for the report of the token champion): one picture per chosen VAL frame: front camera with the GT lane graph projected (GtGenerator's camera view) + the BEV of the delivered graph of B (plain), E312 and E324 s
against the l6 lane GT; own lane (tier 1 of the l4 labels, visible part) drawn as a pale band under the GT; a number table of the frame.  CPU only.
usage: wsl bash tools/env/_py14.sh render_q1115.py <spec.json> <out dir>      spec = [{"name": "01_typical", "k": 123, "title": "..."}, ...]  (k = position in the 873 VAL frames)"""
import json
import os
import pickle
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

ROOT = os.environ["BFLC_ROOT"]
for sub in ("tools/viz", "evaluation", "training/grid", "training", "training/data", "evaluation/sparse_lanes", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub)))
import qgraph_view as QV  # noqa: E402

HERE = os.path.join(ROOT, "temp", "qual_report_1115")
H17 = os.path.join(ROOT, "temp", "qual_report_1117")
P1 = pickle.load(open(os.path.join(H17, "pass1_with_champion.pkl"), "rb"))
FR, SC = P1["frames"], P1["scenes"]
TAB = {r["k"]: r for r in json.load(open(os.path.join(HERE, "perframe_table.json")))}
CH = json.load(open(os.path.join(H17, "perframe_table.json")))["sys"]
NAMES = (("B", "plain lane head B"), ("champion_b25_cb_s0", "CHAMPION: token head, none weight 0.25 (E373 decode)"), ("E324s", "E324 s (previous system, rule steps, snapped)"))
BOX = 48.0


def resample(p, ds=0.5):
    p = np.asarray(p, float)
    if len(p) < 2:
        return p
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    if seg[-1] < 1e-6:
        return p[:1]
    s = np.arange(0, seg[-1] + 1e-9, ds)
    return np.c_[np.interp(s, seg, p[:, 0]), np.interp(s, seg, p[:, 1])]


def panel(ax, rec, key, title, fs=8):
    ax.set_xlim(-BOX, BOX)
    ax.set_ylim(-BOX, BOX)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=fs, linespacing=1.15)
    ax.grid(alpha=0.2)
    ax.tick_params(labelsize=6, length=2, pad=1)
    gts, G = [], []
    for pc in rec["gt"]:
        q = np.asarray(pc["pts"], float)
        gts.append(q)
        G.append(resample(q))
        t1 = pc["tier"] == 1                                                    # own lane: pale band under the GT (tier 1 of the l4 labels, the part a camera sees: darker)
        if t1.any():
            ax.plot(q[t1, 0], q[t1, 1], color="#ffd23f", lw=9, alpha=0.55, solid_capstyle="round", zorder=0)
    G = np.vstack(G) if G else np.zeros((0, 2))
    lines = [resample(l) for l in rec["sys"][key]["lines"] if len(l) > 1]
    P = np.vstack(lines) if lines else np.zeros((0, 2))
    tg = cKDTree(G) if len(G) else None
    tp = cKDTree(P) if len(P) else None
    for q in gts:
        ax.plot(q[:, 0], q[:, 1], color="#b8b8b8", lw=3.0, solid_capstyle="round", zorder=1)
        for m in range(10, len(q) - 1, 20):
            ax.annotate("", xy=q[m + 1], xytext=q[m], arrowprops=dict(arrowstyle="-|>", color="#555555", lw=0.8), zorder=2)
    if tp is not None and len(G):                                               # GT not covered within 2 m: blue
        miss = tp.query(G)[0] > 2.0
        ax.scatter(G[miss, 0], G[miss, 1], s=3, color="#1f4fff", zorder=3, linewidths=0)
    for q in lines:                                                             # delivered: green <= 1 m, orange <= 2 m, red beyond
        d = tg.query(q)[0] if tg is not None else np.full(len(q), 9.0)
        col = np.where(d <= 1.0, 0, np.where(d <= 2.0, 1, 2))
        for c, cc in enumerate(("#1a9a2a", "#f0a000", "#e02020")):
            m = col == c
            if m.any():
                ax.scatter(q[m, 0], q[m, 1], s=2.5, color=cc, zorder=4, linewidths=0)
    for q in lines:
        for m in range(15, len(q) - 1, 30):
            ax.annotate("", xy=q[m + 1], xytext=q[m], arrowprops=dict(arrowstyle="-|>", color="#0b4d12", lw=0.8), zorder=5)
    ax.plot([0], [0], marker="^", color="k", ms=6, zorder=6)


def fmt(x, nd=2):
    return "-" if x is None or x != x else f"{x:.{nd}f}"


def render(spec, out_dir):
    k = spec["k"]
    rec, t = FR[k], TAB[k]
    sc = SC[rec["scene"]]
    fig = plt.figure(figsize=(11.0, 7.6), dpi=100)
    gs = fig.add_gridspec(2, 3, height_ratios=[0.86, 1.0], hspace=0.12, wspace=0.07, left=0.035, right=0.995, top=0.915, bottom=0.04)
    axc = fig.add_subplot(gs[0, :2])
    axc.axis("off")
    try:
        g = QV.load_gt(rec["token"])
        tiles = QV.camera_tiles(rec["token"], g, {c: (800, 450) for c in QV.CAMS}, log=lambda *a: None)
        img = tiles["CAM_FRONT"]
        h = img.shape[0]
        axc.imshow(img[int(h * 0.30):int(h * 0.93)])                            # the road part of the front image
    except Exception as ex:                                                     # noqa: BLE001
        axc.text(0.5, 0.5, f"no camera tile ({type(ex).__name__})", ha="center")
    axc.set_title("front camera, GT lane graph projected (not the delivered graph)", fontsize=8, pad=2)
    axt = fig.add_subplot(gs[0, 2])
    axt.axis("off")
    kk = k
    ch = CH["champion_b25_cb_s0"]
    rows = [("", "plain B", "champion", "E324 s")]
    rows.append(("topo F1", fmt(t["B_topo"]), fmt(ch["topo_F1"][kk]), fmt(t["E324s_topo"])))
    rows.append(("topo P / R", "-", f"{fmt(ch['topo_P'][kk])} / {fmt(ch['topo_R'][kk])}", "-"))
    rows.append(("own lane R", fmt(t["B_ownR"]), "-", fmt(t["E324s_ownR"])))
    rows.append(("maneuvers", *[f"{rec['sys'][s]['man_ok']}/{rec['sys'][s]['man_n']}" for s in ("B", "champion_b25_cb_s0", "E324s")]))
    tb = axt.table(cellText=rows, cellLoc="center", colWidths=[0.34, 0.20, 0.26, 0.20], loc="center", bbox=[0.0, 0.05, 1.0, 0.9])
    tb.auto_set_font_size(False)
    tb.set_fontsize(7.5)
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0 or c == 0:
            cell.set_text_props(weight="bold")
        if c == 0:
            cell.get_text().set_ha("left")
            cell._loc = "left"
    for j, (key, name) in enumerate(NAMES):
        a = fig.add_subplot(gs[1, j])
        tp = ch["topo_F1"][kk] if key == "champion_b25_cb_s0" else t[key + "_topo"]
        panel(a, rec, key, f"{name}\ntopo F1 {fmt(tp)}   maneuvers {rec['sys'][key]['man_ok']}/{rec['sys'][key]['man_n']}")
        if j:
            a.set_yticklabels([])
    desc = sc["description"]
    fig.text(0.5, 0.987, spec["title"], fontsize=10, weight="bold", ha="center", va="top")
    fig.text(0.5, 0.958, f"VAL scene {rec['scene']} ({sc['name']}: {desc[:80]}), frame token {rec['token'][:8]}, row {rec['row']}", fontsize=8, ha="center", va="top", color="#333333")
    fig.text(0.035, 0.012, "GT grey (arrows = travel direction), pale yellow band = own lane (tier 1); delivered points green <= 1 m, orange <= 2 m, red > 2 m from GT; blue = GT not covered within 2 m. Seed 0, VAL.", fontsize=6.5, color="#444444")
    p = os.path.join(out_dir, spec["name"] + ".png")
    fig.savefig(p)
    plt.close(fig)
    return p


if __name__ == "__main__":
    specs = json.load(open(sys.argv[1]))
    out_dir = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    for s in specs:
        print(render(s, out_dir), flush=True)
