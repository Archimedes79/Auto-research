"""Pictures of the junction GT layer (E285) and of the junction maps head (E286): one figure per keyframe, bird's-eye (x right, y forward, metres), two panels of the same frame.

    python junction_view.py frame <token | scene-0221:12> --out DIR [--pred RUN]      the GT layer; with --pred the head's maps next to it (a trained `runs/<RUN>/model.pt` of `jmaps_train.py`)
    python junction_view.py --selftest

Left panel (the layer): lane GT lines (grey), the nuScenes intersection polygons (light grey, id None dashed), the junction ZONE (blue outline), the conflict CORE (red fill), painted stop lines (orange by
type) and pedestrian crossings (black bars). Right panel: entries (green triangles, heading arrow) and exits (magenta squares) with the arm number and lane rank, the true pairs as Hermite curves (thick =
the ego drove it), arm axes with the lane counts (in / out). With --pred, a third and fourth panel show the head's entry / exit heat, zone / core and stop lines and the keypoints it delivers (3 x 3 peaks).
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for sub in ("tools/viz", "training", "training/grid", "training/data", "training/data/prep", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))

STOP_COL = ["#ff7f0e", "#d62728", "#9467bd", "#8c564b", "#e377c2"]


def _panel_layer(ax, g, jg, paint):
    for pc in g["pieces"]:
        p = np.asarray(pc["pts"])
        ax.plot(p[:, 0], p[:, 1], color="#b8b8b8", lw=0.8, zorder=1)
    for a in g["areas"]["intersection"]:
        e = np.asarray(a["exterior"])
        ax.plot(*np.vstack([e, e[:1]]).T, color="#999999", lw=0.8, ls="-" if a.get("id") is not None else "--", zorder=1)
    if jg is not None:
        for z in jg["zones"]:
            ax.plot(*z["poly"].T, color="#1f77b4", lw=1.6, zorder=3)
        core = jg.get("core")
        for q in ([] if core is None else (core.geoms if hasattr(core, "geoms") else [core])):
            if q.geom_type == "Polygon":
                ax.fill(*np.asarray(q.exterior.coords).T, color="#d62728", alpha=0.25, zorder=2)
    if paint:
        for s in paint["stop_line"]:
            ax.fill(*np.asarray(s["poly"]).T, color=STOP_COL[s["type"] % 5], zorder=4)
        for p in paint["ped_crossing"]:
            ax.fill(*np.asarray(p).T, color="black", alpha=0.35, zorder=2)
    ax.set_title("GT layer: zone (blue), conflict core (red), stop lines (orange / red), crossings (black)", fontsize=8)


def _panel_ends(ax, g, jg):
    import qends as QE
    for pc in g["pieces"]:
        p = np.asarray(pc["pts"])
        ax.plot(p[:, 0], p[:, 1], color="#d0d0d0", lw=0.7, zorder=1)
    if jg is None:
        ax.set_title("no junction in this frame", fontsize=8)
        return
    for r in jg["pairs"]:
        a, b = int(r[0]), int(r[1])
        c = QE.curve(jg["ent"][a, :2], jg["ent"][a, 2:], jg["ext"][b, :2], jg["ext"][b, 2:])
        ax.plot(c[:, 0], c[:, 1], color="#2ca02c" if r[9] else "#7f7f7f", lw=2.4 if r[9] else 0.9, alpha=0.9 if r[9] else 0.6, zorder=2)
    for i, e in enumerate(jg["ent"]):
        ax.scatter(e[0], e[1], marker="^", s=40, color="#2ca02c", zorder=5)
        ax.arrow(e[0], e[1], 2.0 * e[2], 2.0 * e[3], head_width=0.7, color="#2ca02c", zorder=5)
        ax.text(e[0], e[1], f"{jg['ent_arm'][i]}.{jg['ent_rank'][i]}", fontsize=6, zorder=6)
    for i, e in enumerate(jg["ext"]):
        ax.scatter(e[0], e[1], marker="s", s=34, color="#c2185b", zorder=5)
        ax.arrow(e[0], e[1], 2.0 * e[2], 2.0 * e[3], head_width=0.7, color="#c2185b", zorder=5)
        ax.text(e[0], e[1], f"{jg['ext_arm'][i]}.{jg['ext_rank'][i]}", fontsize=6, zorder=6)
    for k, a in enumerate(jg["arms"]):
        ax.text(a["c"][0], a["c"][1], f"arm {k}: {a['n_in']} in / {a['n_out']} out", fontsize=6, color="#1f77b4", zorder=6, ha="center")
    o = jg["order"]
    ax.set_title(f"ends: entry (green) / exit (magenta), label arm.rank; pairs (thick = driven); order rule {o['monotone']}/{o['multi']} monotone, {o['identity']} identity", fontsize=7)


def _panel_pred(axs, out, b=0):
    import jmaps as JM
    j1, j05 = out["j1"][b].float().sigmoid().numpy(), out["j05"][b].float().sigmoid().numpy()
    ext = [-50, 50, -50, 50]
    axs[0].imshow(np.maximum(j1[2], j1[3]).T, origin="lower", extent=ext, cmap="magma", vmin=0, vmax=1)
    axs[0].set_title("head: entry / exit heat (max) + keypoints (thr 0.3)", fontsize=8)
    kp = JM.keypoints(dict(j1=out["j1"][b:b + 1]), thr=0.3)[0]
    for e in kp["ent"].numpy():
        axs[0].scatter(e[0], e[1], marker="^", s=30, color="#2ca02c")
    for e in kp["ext"].numpy():
        axs[0].scatter(e[0], e[1], marker="s", s=26, color="#00bcd4")
    rgb = np.zeros((100, 100, 3))
    rgb[..., 2], rgb[..., 0] = j1[0], j1[1]
    axs[1].imshow(rgb.transpose(1, 0, 2), origin="lower", extent=ext)
    axs[1].imshow(np.ma.masked_less(j05[0], 0.5).T, origin="lower", extent=ext, cmap="autumn", vmin=0, vmax=1)
    axs[1].set_title("head: zone (blue), core (red), stop line p > 0.5 (orange)", fontsize=8)


def render(token, out_dir, pred=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import jgt as JG
    import junction_paint as JP
    import qgraph_view as QV
    g = QV.load_gt(token)
    jg = JG.junction_gt(g)
    try:
        paint = JP.load(token)
    except FileNotFoundError:
        paint = None
    n = 4 if pred else 2
    fig, axs = plt.subplots(1, n, figsize=(6.2 * n, 6.4))
    _panel_layer(axs[0], g, jg, paint)
    _panel_ends(axs[1], g, jg)
    if pred:
        import torch
        import jmaps as JM
        import lane_gt_data as LD
        ck = torch.load(os.path.join(ROOT, "runs", pred, "model.pt"), map_location="cpu")
        m = JM.JModel(width=48, d=ck["args"].get("d", 64), ctx=ck["args"].get("ctx", "dil"))
        m.load_state_dict(ck["state"])
        m.eval()
        ds = LD.QData("val_clean", pixels=False)
        k = ds.tokens.index(token)
        row = [i for i, kk in enumerate(ds.ks) if int(ds.bd.rows[kk]) == k]
        item = ds[row[0]]
        with torch.no_grad():
            _, out = m(item["x"].float()[None], joint=False)
        _panel_pred(axs[2:], out)
    for ax in axs:
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
    fig.suptitle(f"{token[:8]}  {g.get('location')}  lane GT {g.get('signature')}", fontsize=9)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{token[:8]}_junction.png")
    fig.savefig(path, dpi=80)
    plt.close(fig)
    return path


def _selftest():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import jgt as JG

    def run(a, b, n=12):
        return np.stack([np.linspace(a[0], b[0], n), np.linspace(a[1], b[1], n)], 1)
    s, nth = run((0, -30), (0, -8)), run((0, 8), (0, 30))
    conn = run(s[-1], nth[0], 16)
    pts = np.vstack([s, conn, nth])
    g = dict(pieces=[dict(piece="p0", pts=pts, tier=np.full(len(pts), 1), segments=[dict(kind="lane", i0=0, i1=len(s) - 1), dict(kind="connector", i0=len(s), i1=len(s) + len(conn) - 1),
                                                                                   dict(kind="lane", i0=len(s) + len(conn), i1=len(pts) - 1)])], areas=dict(intersection=[]))
    jg = JG.junction_gt(g)
    fig, axs = plt.subplots(1, 2)
    _panel_layer(axs[0], g, jg, None)
    _panel_ends(axs[1], g, jg)
    plt.close(fig)
    print("junction_view selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("frame")
    p.add_argument("what")
    p.add_argument("--out", required=True)
    p.add_argument("--pred", default=None)
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    else:
        import qgraph_view as QV
        tok = QV._frame_token(a.what)
        print("written", render(tok, a.out, a.pred))
