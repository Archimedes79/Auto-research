"""qual_frames: the qualitative picture of chosen VAL frames, several systems side by side (promoted 2026-11-23 from the report scratch of 2026-11-17, D140).
One picture per frame: the front camera with the GT lane graph projected (GtGenerator's camera view through `qgraph_view`), a number table per system, and one BEV
panel per system: its delivered graph against the lane GT (GT grey with travel arrows, own lane = a pale band, delivered points green <= 1 m / orange <= 2 m / red
beyond, GT not covered within 2 m blue). CPU only. The frames are chosen OUTSIDE this tool by a written rule (`--spec`), never by eye.

    wsl bash tools/env/_py14.sh tools/viz/qual_frames.py --frames F.pkl --spec S.json --systems "B=plain lane head B,champ=token champion" --out DIR [--table T.json]
    wsl bash tools/env/_py14.sh tools/viz/qual_frames.py --selftest

--frames: a pickle {"frames": [rec, ...], "scenes": {scene: {"name", "description"}}} with, per VAL frame k, rec = {"scene", "token", "row",
          "gt": [{"pts": (n, 2) metres, "tier": (n,) int (1 = own lane)}], "sys": {key: {"lines": [(m, 2) ...], "man_ok": int, "man_n": int}}}
          (the graphs of each system on the same frames, e.g. from `evaluation/jtoken_eval.py graphs` and the per-frame read-outs)
--spec:   [{"name": "01_typical_good", "k": 123, "title": "why this frame"}, ...]   (k = the position in the frames list)
--table:  optional {key: {"<metric name>": [value per frame k], ...}, ...}: one table row per metric name (e.g. "topo F1"); maneuvers come from --frames
--systems: key=label pairs in panel order (2 - 4 systems)."""
import argparse
import json
import os
import pickle
import sys
import tempfile
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
BOX = 48.0                                                                       # half width of a BEV panel, m
NEAR, FAR = 1.0, 2.0                                                             # the colour bands of a delivered point (m from the GT)


def resample(p, ds=0.5):
    p = np.asarray(p, float)
    if len(p) < 2:
        return p
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    if seg[-1] < 1e-6:
        return p[:1]
    s = np.arange(0, seg[-1] + 1e-9, ds)
    return np.c_[np.interp(s, seg, p[:, 0]), np.interp(s, seg, p[:, 1])]


def bands(points, gt_points):
    """the colour band of each delivered point: 0 = within NEAR of the GT, 1 = within FAR, 2 = beyond (all 2 without GT)"""
    points = np.asarray(points, float).reshape(-1, 2)
    if not len(gt_points):
        return np.full(len(points), 2)
    d = cKDTree(gt_points).query(points)[0]
    return np.where(d <= NEAR, 0, np.where(d <= FAR, 1, 2))


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
        t1 = np.asarray(pc.get("tier", np.zeros(len(q))), int) == 1               # own lane: pale band under the GT
        if t1.any():
            ax.plot(q[t1, 0], q[t1, 1], color="#ffd23f", lw=9, alpha=0.55, solid_capstyle="round", zorder=0)
    G = np.vstack(G) if G else np.zeros((0, 2))
    lines = [resample(l) for l in rec["sys"][key]["lines"] if len(l) > 1]
    P = np.vstack(lines) if lines else np.zeros((0, 2))
    for q in gts:
        ax.plot(q[:, 0], q[:, 1], color="#b8b8b8", lw=3.0, solid_capstyle="round", zorder=1)
        for m in range(10, len(q) - 1, 20):
            ax.annotate("", xy=q[m + 1], xytext=q[m], arrowprops=dict(arrowstyle="-|>", color="#555555", lw=0.8), zorder=2)
    if len(P) and len(G):                                                       # GT not covered within FAR: blue
        miss = cKDTree(P).query(G)[0] > FAR
        ax.scatter(G[miss, 0], G[miss, 1], s=3, color="#1f4fff", zorder=3, linewidths=0)
    for q in lines:
        col = bands(q, G)
        for c, cc in enumerate(("#1a9a2a", "#f0a000", "#e02020")):
            m = col == c
            if m.any():
                ax.scatter(q[m, 0], q[m, 1], s=2.5, color=cc, zorder=4, linewidths=0)
        for m in range(15, len(q) - 1, 30):
            ax.annotate("", xy=q[m + 1], xytext=q[m], arrowprops=dict(arrowstyle="-|>", color="#0b4d12", lw=0.8), zorder=5)
    ax.plot([0], [0], marker="^", color="k", ms=6, zorder=6)


def fmt(x, nd=2):
    return "-" if x is None or x != x else f"{x:.{nd}f}"


def camera(ax, token):
    """the front camera of the frame with the GT projected; a text in place of the image when the tile cannot be made (no dataset, no GT store)"""
    ax.axis("off")
    try:
        for sub in ("tools/viz", "evaluation", "net/prep", "net/model", "net/loss", "training", "training/data", "evaluation/sparse_lanes", ".."):
            p = os.path.normpath(os.path.join(ROOT, sub))
            if p not in sys.path:
                sys.path.insert(0, p)
        import qgraph_view as QV
        g = QV.load_gt(token)
        img = QV.camera_tiles(token, g, {c: (800, 450) for c in QV.CAMS}, log=lambda *a: None)["CAM_FRONT"]
        h = img.shape[0]
        ax.imshow(img[int(h * 0.30):int(h * 0.93)])                             # the road part of the front image
    except Exception as ex:                                                     # noqa: BLE001
        ax.text(0.5, 0.5, f"no camera tile ({type(ex).__name__})", ha="center")
    ax.set_title("front camera, GT lane graph projected (not the delivered graph)", fontsize=8, pad=2)


def render(spec, data, systems, table, out_dir):
    k = spec["k"]
    rec = data["frames"][k]
    sc = data.get("scenes", {}).get(rec["scene"], {})
    n = len(systems)
    fig = plt.figure(figsize=(3.7 * n, 7.6), dpi=100)
    gs = fig.add_gridspec(2, n, height_ratios=[0.86, 1.0], hspace=0.12, wspace=0.07, left=0.035, right=0.995, top=0.915, bottom=0.04)
    camera(fig.add_subplot(gs[0, :max(1, n - 1)]), rec["token"])
    axt = fig.add_subplot(gs[0, n - 1] if n > 1 else gs[0, 0])
    axt.axis("off")
    metrics = sorted({m for key, _ in systems for m in table.get(key, {})})
    rows = [[""] + ["\n".join(textwrap.wrap(label, 14)[:3]) for _, label in systems]]
    for m in metrics:
        rows.append([m] + [fmt(table.get(key, {}).get(m, [None] * (k + 1))[k]) for key, _ in systems])
    rows.append(["maneuvers"] + [f"{rec['sys'][key]['man_ok']}/{rec['sys'][key]['man_n']}" for key, _ in systems])
    tb = axt.table(cellText=rows, cellLoc="center", loc="center", bbox=[0.0, 0.05, 1.0, 0.9])
    tb.auto_set_font_size(False)
    tb.set_fontsize(7.5)
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0 or c == 0:
            cell.set_text_props(weight="bold")
    for j, (key, label) in enumerate(systems):
        a = fig.add_subplot(gs[1, j])
        first = metrics[0] if metrics else None
        val = table.get(key, {}).get(first, [None] * (k + 1))[k] if first else None
        panel(a, rec, key, f"{label}\n{first + ' ' + fmt(val) + '   ' if first else ''}maneuvers {rec['sys'][key]['man_ok']}/{rec['sys'][key]['man_n']}")
        if j:
            a.set_yticklabels([])
    fig.text(0.5, 0.987, spec.get("title", spec["name"]), fontsize=10, weight="bold", ha="center", va="top")
    fig.text(0.5, 0.958, f"VAL scene {rec['scene']} ({sc.get('name', '?')}: {sc.get('description', '')[:80]}), frame token {str(rec['token'])[:8]}, row {rec.get('row', k)}",
             fontsize=8, ha="center", va="top", color="#333333")
    fig.text(0.035, 0.012, f"GT grey (arrows = travel direction), pale yellow band = own lane; delivered points green <= {NEAR:g} m, orange <= {FAR:g} m, red > {FAR:g} m "
             f"from GT; blue = GT not covered within {FAR:g} m.", fontsize=6.5, color="#444444")
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, spec["name"] + ".png")
    fig.savefig(p)
    plt.close(fig)
    return p


def parse_systems(s):
    out = []
    for part in s.split(","):
        key, _, label = part.partition("=")
        out.append((key.strip(), (label or key).strip()))
    assert 1 <= len(out) <= 4, "--systems: 1 to 4 key=label pairs"
    return out


def selftest():
    line = np.c_[np.linspace(-30, 30, 61), np.zeros(61)]
    rec = {"scene": "s1", "token": "not-a-token", "row": 0,
           "gt": [{"pts": line, "tier": np.ones(61, int)}, {"pts": line + [0, 20], "tier": np.zeros(61, int)}],
           "sys": {"a": {"lines": [line + [0, 0.5]], "man_ok": 1, "man_n": 2}, "b": {"lines": [line + [0, 1.5], line + [0, 9]], "man_ok": 0, "man_n": 2}}}
    data = {"frames": [rec], "scenes": {"s1": {"name": "scene-0001", "description": "synthetic"}}}
    assert list(bands([[0, 0.5], [0, 1.5], [0, 5]], resample(line))) == [0, 1, 2], "the colour bands: <= 1 m, <= 2 m, beyond"
    assert list(bands([[0, 0]], np.zeros((0, 2)))) == [2], "no GT: every point beyond"
    assert parse_systems("a=first,b") == [("a", "first"), ("b", "b")]
    with tempfile.TemporaryDirectory() as td:
        p = render({"name": "t", "k": 0, "title": "selftest"}, data, parse_systems("a=system A,b=system B"), {"a": {"topo F1": [0.9]}, "b": {"topo F1": [0.4]}}, td)
        assert os.path.getsize(p) > 10_000, "a picture is written"
        fp, sp = os.path.join(td, "f.pkl"), os.path.join(td, "s.json")
        pickle.dump(data, open(fp, "wb"))
        json.dump([{"name": "u", "k": 0, "title": "cli"}], open(sp, "w"))
        assert main(["--frames", fp, "--spec", sp, "--systems", "a=A,b=B", "--out", td]) == [os.path.join(td, "u.png")]
    print("qual_frames selftest ok: colour bands, system parsing, one picture through render and one through the command line (no camera tile without the dataset)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--frames")
    ap.add_argument("--spec")
    ap.add_argument("--systems")
    ap.add_argument("--table", default=None)
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    assert a.frames and a.spec and a.systems and a.out, "--frames, --spec, --systems and --out are needed (or --selftest)"
    data = pickle.load(open(a.frames, "rb"))
    table = json.load(open(a.table)) if a.table else {}
    out = [render(s, data, parse_systems(a.systems), table, a.out) for s in json.load(open(a.spec))]
    for p in out:
        print(p, flush=True)
    return out


if __name__ == "__main__":
    main()
