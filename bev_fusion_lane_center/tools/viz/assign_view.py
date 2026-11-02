"""Picture and video of the ASSIGNMENT of the sparse lane vertices to the SD-map roads (stage 4 of training/grid/, see training/grid/README.md).

    wsl bash tools/env/_py14.sh assign_view.py [registry name] [--scene 23] [--frames 6] [--every 2] [--out DIR] [--video]

Real data, CPU only: the delivered lane graph of a registry system (default recall_sp0_dp, cached outputs: `evaluate.graphs_of`, the vertices of its lanes with their
headings) and the SD map of the same frame (`p8_lib.cache("sd", ...)`, exact registration; the pose error is not shown here: it belongs to SoftRegister). The assignment is
`training/grid/assign.py` with its START values (untrained scalars), so the picture shows what the idea does before any learning, not a trained result.

Left panel (ego frame, metres, x right, y forward as the other lane pictures): grey = SD roads (dashed = lane slots of the road, blue = driven route), pale green = ground-truth
lanes, dots = vertices coloured by the road they are assigned to (size = confidence, faint = unsure), a thin line from each vertex to the map point it is pulled to,
red cross = vertex that belongs to no road (dustbin > 0.5). Right panel: the assignment matrix (vertices ordered by road and position along it, columns = roads + none),
and below it per road: perceived mass (bars) and whether ground-truth lanes lie on that road (dot).
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")  # a shared machine
import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "training", "grid"))
import assign as AS  # noqa: E402

CMAP = plt.get_cmap("tab10")


def to_grid(p):
    """(x right, y forward) -> (x forward, y left)"""
    p = np.asarray(p, np.float32)
    return np.stack([p[..., 1], -p[..., 0]], -1)


def from_grid(p):
    p = np.asarray(p)
    return np.stack([-p[..., 1], p[..., 0]], -1)


def vertices_of(g, conf=None):
    """the vertices of a lane graph (pts in x right, y forward) as grid-frame arrays: pos (N, 2), heading (N, 2), weight (N,), lane index (N,)"""
    pos, hdg, w, lane = [], [], [], []
    for i, q in enumerate(g["pts"]):
        q = np.asarray(q, np.float64)
        if len(q) < 2:
            continue
        t = np.gradient(q, axis=0)
        t = t / np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
        c = 1.0 if conf is None or conf[i] is None else float(conf[i])
        pos.append(to_grid(q)), hdg.append(to_grid(t)), w.append(np.full(len(q), c)), lane.append(np.full(len(q), i))
    if not pos:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32), np.zeros(0, np.float32), np.zeros(0, int)
    return np.concatenate(pos).astype(np.float32), np.concatenate(hdg).astype(np.float32), np.concatenate(w).astype(np.float32), np.concatenate(lane)


def gt_cover(mp, gt_pts, r=2.5):
    """per map id: share of its tokens with a ground-truth lane vertex within r metres (the reference for 'is the road really perceived')"""
    from scipy.spatial import cKDTree
    g = np.concatenate([to_grid(np.asarray(p)) for p in gt_pts if len(p)]) if len(gt_pts) else np.zeros((0, 2))
    out = np.zeros(mp["n_ids"])
    if len(g) == 0:
        return out
    d, _ = cKDTree(g).query(mp["pos"])
    for k in range(mp["n_ids"]):
        m = mp["rid"] == k
        out[k] = float((d[m] < r).mean()) if m.any() else 0.0
    return out


def panel_arrays(g, conf, gt, sd, module):
    """everything one frame needs: the assignment of the vertices of graph g to the SD map sd"""
    mp = AS.map_tokens(sd)
    pos, hdg, w, lane = vertices_of(g, conf)
    t = AS.to_torch(mp)
    with torch.no_grad():
        out = module(torch.as_tensor(pos), torch.as_tensor(hdg), torch.as_tensor(w), t)
    return dict(mp=mp, pos=pos, hdg=hdg, w=w, lane=lane, out={k: v.numpy() for k, v in out.items()}, cover=gt_cover(mp, gt["pts"]), sd=sd)


def draw(fig, d, title=""):
    mp, pos, out = d["mp"], d["pos"], d["out"]
    R = mp["n_ids"]
    P = out["P_road"]
    best = P[:, :R].argmax(1) if R else np.zeros(len(pos), int)
    unassigned = P[:, R] > 0.5 if len(pos) else np.zeros(0, bool)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], height_ratios=[1.6, 1.0], hspace=0.28, wspace=0.18)
    ax = fig.add_subplot(gs[:, 0])
    ax.set_aspect("equal")
    ax.set_xlim(-50, 50), ax.set_ylim(-50, 50)
    ax.set_facecolor("white")
    ax.grid(color="0.93", lw=0.5)
    for i, q in enumerate(d["gt_pts"]):
        if len(q) > 1:
            ax.plot(q[:, 0], q[:, 1], color=(0.65, 0.85, 0.65), lw=3.0, solid_capstyle="round", zorder=1)
    col = {}
    for key, (polys, off) in (("roads", (d["sd"]["roads"], 0)), ("route", (d["sd"]["route"], len(d["sd"]["roads"])))):
        for k, (poly, cnt) in enumerate(polys):
            poly = np.asarray(poly)
            if len(poly) < 2:
                continue
            kid = off + k
            col[kid] = CMAP(kid % 10)
            if key == "route":
                ax.plot(poly[:, 0], poly[:, 1], color=(0.15, 0.3, 0.9), lw=1.6, alpha=0.55, zorder=2)
                continue
            ax.plot(poly[:, 0], poly[:, 1], color="0.75", lw=7.0, solid_capstyle="round", alpha=0.6, zorder=0)
            ax.plot(poly[:, 0], poly[:, 1], color=col[kid], lw=1.2, zorder=2)
            m = len(poly) // 2
            ax.annotate("", xy=poly[m + 1] if m + 1 < len(poly) else poly[-1], xytext=poly[m], arrowprops=dict(arrowstyle="-|>", color=col[kid], lw=1.2), zorder=2)
            ax.text(poly[m, 0] + 1.0, poly[m, 1], f"r{kid}", fontsize=8, color=col[kid], zorder=6, weight="bold")
    if len(pos):
        a = out["A"][:, :-1]
        a = a / np.maximum(a.sum(1, keepdims=True), 1e-9)
        tgt = from_grid(a @ mp["pos"])
        xy = from_grid(pos)
        for i in range(len(pos)):
            if not unassigned[i]:
                ax.plot([xy[i, 0], tgt[i, 0]], [xy[i, 1], tgt[i, 1]], color=col.get(int(best[i]), "0.5"), lw=0.6, alpha=0.5, zorder=3)
        for lane in np.unique(d["lane"]):
            m = d["lane"] == lane
            ax.plot(xy[m, 0], xy[m, 1], color="0.3", lw=0.6, alpha=0.4, zorder=3)
        ok = ~unassigned
        ax.scatter(xy[ok, 0], xy[ok, 1], s=14 + 40 * d["w"][ok], c=[col.get(int(b), "0.5") for b in best[ok]], alpha=0.25 + 0.75 * P[ok, :R].max(1) if R else 0.5,
                   edgecolor="k", linewidth=0.4, zorder=5)
        ax.scatter(xy[unassigned, 0], xy[unassigned, 1], marker="x", c="red", s=36, zorder=6)
    ax.scatter([0], [0], marker="^", c="k", s=60, zorder=7)
    ax.set_title(title, fontsize=9, loc="left")
    # matrix
    ax2 = fig.add_subplot(gs[0, 1])
    if len(pos):
        order = np.lexsort((pos[:, 0], np.where(unassigned, R, best)))
        ax2.imshow(P[order], aspect="auto", cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
    ax2.set_xticks(range(R + 1))
    ax2.set_xticklabels([f"r{k}" for k in range(R)] + ["none"], fontsize=6, rotation=90)
    ax2.set_ylabel("vertices (by road, then position)", fontsize=7)
    ax2.set_title("assignment P(road | vertex)", fontsize=8, loc="left")
    ax2.tick_params(axis="y", labelsize=6)
    # per road mass vs truth
    ax3 = fig.add_subplot(gs[1, 1])
    if R:
        lens = np.array([max(float((mp["rid"] == k).sum()) * 2.0, 1.0) for k in range(R)])
        pres = AS.present(torch.as_tensor(out["mass"]), torch.as_tensor(lens, dtype=torch.float32)).numpy()
        ax3.bar(range(R), pres, color=[col.get(k, "0.7") for k in range(R)], edgecolor="k", linewidth=0.4)
        ax3.scatter(range(R), np.where(d["cover"] > 0.3, 1.04, np.nan), marker="v", c="green", s=22, zorder=5)
    ax3.set_ylim(0, 1.15)
    ax3.set_xticks(range(R))
    ax3.set_xticklabels([f"r{k}" for k in range(R)], fontsize=6)
    ax3.tick_params(axis="y", labelsize=6)
    ax3.set_title("road perceived (bar)   truth lanes on it (green mark)", fontsize=8, loc="left")
    return fig


def load(name, split, ks):
    import evaluate as EV
    import p8_lib as L

    e = EV.registry()[name]
    import lane_prototype as LP
    g, conf, gts, info = LP.delivered(e, split, ks)                    # the delivered graph: assembled + the registry output stage, with the lane confidences
    rows = np.array(json.load(open(f"{EV.R}/data/splits.json"))[EV.SPLITS[split][1]])
    part = {"val": "val", "test": "test", "train": "train"}[split]
    sds = L.cache("sd", part)
    return g, conf, gts, [sds[int(rows[k])] for k in ks]


def scene_ks(split, scene, frames, every):
    import evaluate as EV

    rows = np.array(json.load(open(f"{EV.R}/data/splits.json"))[EV.SPLITS[split][1]])
    pz = np.load(f"{EV.HOME}/exp_gcar_b123/poses_{split}.npz")
    sc, ts = pz["scene"][rows], pz["ts"][rows]
    ix = np.nonzero(sc == scene)[0]
    ix = sorted(ix, key=lambda i: ts[i])[::every]
    return [int(i) for i in ix][:frames]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?", default="recall_sp0_dp")
    ap.add_argument("--split", default="val")
    ap.add_argument("--scene", type=int, default=23)
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--out", default=None)
    ap.add_argument("--video", action="store_true")
    a = ap.parse_args()
    out_dir = os.path.join(ROOT, a.out) if a.out else os.path.join(ROOT, "runs", "grid_view", a.name)
    os.makedirs(out_dir, exist_ok=True)
    ks = scene_ks(a.split, a.scene, a.frames, a.every)
    g, conf, gts, sds = load(a.name, a.split, ks)
    module = AS.Assigner()
    frames = []
    for n, k in enumerate(ks):
        if sds[n] is None:
            print("frame", k, "has no SD map (no route), skipped")
            continue
        d = panel_arrays(g[n], conf[n], gts[n], sds[n], module)
        d["gt_pts"] = [np.asarray(p) for p in gts[n]["pts"]]
        fig = plt.figure(figsize=(11.5, 6.6), dpi=90)
        draw(fig, d, title=f"{a.name}  {a.split} scene {a.scene}  frame {n + 1}/{len(ks)}   (assignment with START values, not trained)")
        path = os.path.join(out_dir, f"scene{a.scene}_f{n:02d}.png")
        fig.savefig(path, bbox_inches="tight")
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
        share = float((d["out"]["P_road"][:, :-1].sum(1) > 0.5).mean()) if len(d["pos"]) else float("nan")
        print(f"frame {k}: {len(d['pos'])} vertices, {share:.2f} assigned to a road, roads {d['mp']['n_ids']}  -> {os.path.relpath(path, ROOT)}")
    if a.video and frames:
        import make_video as MV
        h = min(f.shape[0] for f in frames), min(f.shape[1] for f in frames)
        MV.frames_to_video([f[:h[0], :h[1]] for f in frames], os.path.join(out_dir, f"scene{a.scene}.mp4"), fps=2)


if __name__ == "__main__":
    main()
