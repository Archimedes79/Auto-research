"""Pictures of the junction GT layer (E285), of the junction maps head (E286) and of the pair head (E287): one figure per keyframe, bird's-eye (x right, y forward, metres), panels of the same frame.

    python junction_view.py frame <token | scene-0221:12> --out DIR [--pred RUN] [--pairs RUN] [--cam]      the GT layer; --pred = the maps head of a trained `runs/<RUN>/model.pt` (`jmaps_train.py`),
                                                                                                         --pairs = the pair head dump `runs/<RUN>/ends_val.pkl` (`jends_eval.py dump` / `qends_eval.py dump`),
                                                                                                         --cam = the front camera with the GT lane graph projected
    python junction_view.py scene <VAL scene id | name> --out DIR [--pred RUN] [--pairs RUN] [--cam] [--every K] [--frames N] [--fps 2]     an mp4 of the keyframes
    python junction_view.py --selftest

Panels. GT layer: lane GT lines (grey), the nuScenes intersection polygons (light grey, id None dashed), the junction ZONE (blue outline), the conflict CORE (red fill), painted stop lines (orange by type) and pedestrian
crossings (black bars). GT ends: entries (green triangles, heading arrow) and exits (magenta squares) with the arm number and lane rank, the true pairs as Hermite curves (thick = the ego drove it), arm axes with the lane counts
(in / out). With --pred: the head's entry / exit heat with the keypoints (3 x 3 peaks above 0.3: matched to a GT end within 3 m = filled, else hollow), and its zone (blue) / core (red) / stop-line (orange) maps. With --pairs: the pairs
the head delivers (logit >= 0): green = a true pair, red = a false pair, blue dashed = a true pair that was missed (an end not found or the pair not delivered).
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for sub in ("tools/viz", "training", "training/grid", "training/data", "training/data/prep", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))

STOP_COL = ["#ff7f0e", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
_CACHE = {}


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
    ax.set_title("GT layer: zone (blue), conflict core (red), stop lines (orange), crossings (black)", fontsize=8)


def _panel_ends(ax, g, jg, labels=True):
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
        if labels:
            ax.text(e[0] + 0.6, e[1] + 0.6, f"{jg['ent_arm'][i]}.{jg['ent_rank'][i]}", fontsize=6, zorder=6)
    for i, e in enumerate(jg["ext"]):
        ax.scatter(e[0], e[1], marker="s", s=34, color="#c2185b", zorder=5)
        ax.arrow(e[0], e[1], 2.0 * e[2], 2.0 * e[3], head_width=0.7, color="#c2185b", zorder=5)
        if labels:
            ax.text(e[0] + 0.6, e[1] - 1.4, f"{jg['ext_arm'][i]}.{jg['ext_rank'][i]}", fontsize=6, zorder=6)
    o = jg["order"]
    ax.set_title(f"GT ends: entry (green) / exit (magenta), arm.rank; pairs (thick = driven); order rule {o['monotone']}/{o['multi']} monotone", fontsize=8)


def _match_idx(kp, G, tol=3.0):
    """index of the nearest keypoint within tol metres for every GT end (-1 none)"""
    out = -np.ones(len(G), int)
    for g_, e in enumerate(G):
        if len(kp):
            d = np.linalg.norm(kp[:, :2] - e[:2], axis=1)
            if d.min() <= tol:
                out[g_] = int(d.argmin())
    return out


def _panel_maps(axs, out, jg, b=0):
    import jmaps as JM
    j1, j05 = out["j1"][b].float().sigmoid().numpy(), out["j05"][b].float().sigmoid().numpy()
    ext = [-50, 50, -50, 50]
    axs[0].imshow(np.maximum(j1[2], j1[3]).T, origin="lower", extent=ext, cmap="magma", vmin=0, vmax=1)
    kp = JM.keypoints(dict(j1=out["j1"][b:b + 1]), thr=0.3)[0]
    for name, marker, col, G in (("ent", "^", "#2ca02c", None if jg is None else jg["ent"]), ("ext", "s", "#00bcd4", None if jg is None else jg["ext"])):
        k = kp[name].numpy()
        matched = set(_match_idx(k, G).tolist()) if G is not None and len(G) else set()
        for i, e in enumerate(k):
            hit = i in matched
            axs[0].scatter(e[0], e[1], marker=marker, s=36, facecolors=col if hit else "none", edgecolors=col, linewidths=1.4)
    axs[0].set_title("head: entry / exit heat (max), keypoints (filled = matched within 3 m)", fontsize=8)
    rgb = np.zeros((100, 100, 3))
    rgb[..., 2], rgb[..., 0] = j1[0], j1[1]
    axs[1].imshow(rgb.transpose(1, 0, 2), origin="lower", extent=ext)
    axs[1].imshow(np.ma.masked_less(j05[0], 0.5).T, origin="lower", extent=ext, cmap="autumn", vmin=0, vmax=1)
    axs[1].set_title("head: zone (blue), core (red), stop line p > 0.5 (orange)", fontsize=8)


def _panel_pairs(ax, g, jg, frame):
    import qends as QE
    from jrule_control import _match
    for pc in g["pieces"]:
        p = np.asarray(pc["pts"])
        ax.plot(p[:, 0], p[:, 1], color="#d0d0d0", lw=0.7, zorder=1)
    if jg is None or frame is None:
        ax.set_title("no junction in this frame" if jg is None else "no dump for this frame", fontsize=8)
        return
    K = frame["logit"].shape[0]
    kp, kd = frame["kp_pos"], frame["kp_dir"]
    ia = _match(kp[:K], kd[:K], frame["valid"][0], jg["ent"])
    ib = _match(kp[K:], kd[K:], frame["valid"][1], jg["ext"])
    inv_a = {int(k): g_ for g_, k in enumerate(ia) if k >= 0}
    inv_b = {int(k): g_ for g_, k in enumerate(ib) if k >= 0}
    gt = {(int(r[0]), int(r[1])) for r in jg["pairs"]}
    got, n_ok, n_bad = set(), 0, 0
    p = 1 / (1 + np.exp(-frame["logit"])) * frame["allowed"]
    for i, j in zip(*np.where(p >= 0.5)):
        key = (inv_a.get(int(i), -1), inv_b.get(int(j), -1))
        ok = key in gt
        c = QE.curve(kp[i], kd[i], kp[K + j], kd[K + j])
        ax.plot(c[:, 0], c[:, 1], color="#2ca02c" if ok else "#d62728", lw=2.2 if ok else 1.2, alpha=0.95 if ok else 0.8, zorder=3)
        if ok:
            got.add(key)
            n_ok += 1
        else:
            n_bad += 1
    for (a, b) in sorted(gt - got):
        c = QE.curve(jg["ent"][a, :2], jg["ent"][a, 2:], jg["ext"][b, :2], jg["ext"][b, 2:])
        ax.plot(c[:, 0], c[:, 1], color="#1f77b4", lw=1.0, ls="--", alpha=0.8, zorder=2)
    for i in np.where(frame["valid"][0])[0]:
        ax.scatter(kp[i, 0], kp[i, 1], marker="^", s=22, color="#2ca02c", zorder=4)
    for j in np.where(frame["valid"][1])[0]:
        ax.scatter(kp[K + j, 0], kp[K + j, 1], marker="s", s=18, color="#c2185b", zorder=4)
    ax.set_title(f"pairs delivered: {n_ok} true (green), {n_bad} false (red); missed {len(gt - got)} of {len(gt)} (blue dashed)", fontsize=8)


def _panel_cam(ax, token, g):
    import qgraph_view as QV
    sizes = {c: (640, 360) for c in QV.CAMS}
    t = QV.camera_tiles(token, g, sizes, log=lambda *_: None)["CAM_FRONT"]
    ax.imshow(t[..., ::-1] if False else t)
    ax.axis("off")
    ax.set_title("front camera with the GT lane graph projected", fontsize=8)


def _maps_model(run):
    import torch
    import jmaps as JM
    if ("m", run) not in _CACHE:
        ck = torch.load(os.path.join(ROOT, "runs", run, "model.pt"), map_location="cpu")
        m = JM.JModel(width=48, d=ck["args"].get("d", 64), ctx=ck["args"].get("ctx", "dil"))
        m.load_state_dict(ck["state"])
        m.eval()
        _CACHE[("m", run)] = m
    return _CACHE[("m", run)]


def _item(token):
    import lane_gt_data as LD
    if "ds" not in _CACHE:
        _CACHE["ds"] = LD.QData("val_clean", pixels=False)
        _CACHE["row"] = {_CACHE["ds"].tokens[int(_CACHE["ds"].bd.rows[kk])]: i for i, kk in enumerate(_CACHE["ds"].ks)}
    return _CACHE["ds"][_CACHE["row"][token]]


def _dump(run):
    import pickle
    if ("d", run) not in _CACHE:
        _CACHE[("d", run)] = {f["token"]: f for f in pickle.load(open(os.path.join(ROOT, "runs", run, "ends_val.pkl"), "rb"))}
    return _CACHE[("d", run)]


def render(token, out_dir, pred=None, pairs=None, cam=False):
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
    panels = ["layer", "ends"] + (["heat", "zone"] if pred else []) + (["pairs"] if pairs else []) + (["cam"] if cam else [])
    ncol = len(panels) if len(panels) <= 3 else 3
    nrow = (len(panels) + ncol - 1) // ncol
    fig, axs = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 6.4 * nrow), squeeze=False)
    axd = {name: axs[i // ncol][i % ncol] for i, name in enumerate(panels)}
    for i in range(len(panels), nrow * ncol):
        axs[i // ncol][i % ncol].axis("off")
    _panel_layer(axd["layer"], g, jg, paint)
    _panel_ends(axd["ends"], g, jg)
    if pred:
        import torch
        item = _item(token)
        with torch.no_grad():
            _, out = _maps_model(pred)(item["x"].float()[None], joint=False)
        _panel_maps([axd["heat"], axd["zone"]], out, jg)
    if pairs:
        _panel_pairs(axd["pairs"], g, jg, _dump(pairs).get(token))
    if cam:
        _panel_cam(axd["cam"], token, g)
    for name, ax in axd.items():
        if name != "cam":
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


def render_scene(scene, out_dir, pred=None, pairs=None, cam=False, every=1, frames=0, fps=2):
    """one mp4 (`<scene name>_junction.mp4`, mp4v, `fps` frames per second) of the keyframes of a VAL scene (id or name), every `every`-th, at most `frames` (0 = all); the same figure as `render`"""
    import cv2
    import qgraph_view as QV
    sid, name, toks = QV.resolve_scene(scene)
    toks = toks[::every]
    if frames:
        toks = toks[:frames]
    os.makedirs(out_dir, exist_ok=True)
    tmp = os.path.join(out_dir, "_frames")
    writer = None
    path = os.path.join(out_dir, f"{name}_junction.mp4")
    for t in toks:
        img = cv2.imread(render(t, tmp, pred, pairs, cam))
        if writer is None:
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (img.shape[1], img.shape[0]))
        writer.write(img)
    writer.release()
    for f in os.listdir(tmp):
        os.remove(os.path.join(tmp, f))
    os.rmdir(tmp)
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
    fig, axs = plt.subplots(1, 3)
    _panel_layer(axs[0], g, jg, None)
    _panel_ends(axs[1], g, jg)
    # a pair dump that delivers the true pair and one false pair, and misses nothing
    K = 4
    kp = np.zeros((2 * K, 2)); kd = np.zeros((2 * K, 2)); kd[:, 1] = 1.0
    kp[0], kp[K] = jg["ent"][0, :2], jg["ext"][0, :2]
    kp[1], kp[K + 1] = jg["ent"][0, :2] + [5, 0], jg["ext"][0, :2] + [20, 0]
    logit = np.full((K, K), -9.0); logit[0, 0] = 3.0; logit[1, 1] = 2.0
    valid = np.zeros((2, K), bool); valid[0, :2] = valid[1, :2] = True
    _panel_pairs(axs[2], g, jg, dict(kp_pos=kp, kp_dir=kd, logit=logit, allowed=np.ones((K, K), bool), valid=valid))
    plt.close(fig)
    print("junction_view selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    for nm in ("frame", "scene"):
        p = sub.add_parser(nm)
        p.add_argument("what", help="sample token or scene-0221:12" if nm == "frame" else "VAL scene id or name")
        p.add_argument("--out", required=True)
        p.add_argument("--pred", default=None)
        p.add_argument("--pairs", default=None)
        p.add_argument("--cam", action="store_true")
        if nm == "scene":
            p.add_argument("--every", type=int, default=1)
            p.add_argument("--frames", type=int, default=0)
            p.add_argument("--fps", type=int, default=2)
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    elif a.cmd == "scene":
        print("written", render_scene(a.what, a.out, a.pred, a.pairs, a.cam, a.every, a.frames, a.fps))
    else:
        import qgraph_view as QV
        print("written", render(QV._frame_token(a.what), a.out, a.pred, a.pairs, a.cam))
