"""where does the delivered system fail? (scratch, CPU) The delivered graph of B + the hard close-node step + the connectors of a pair-head run, against the lane GT, on the 873 VAL frames.
Writes temp/jr/media/errors/: stats.json (per-frame numbers), distribution.png (per-frame coverage and pair recall, pair probability of true and false candidates), and galleries of frames
(missed pairs, false pairs, worst coverage): three panels per frame: GT lanes with ends | delivered graph against GT (green = within 1 m of a GT point, red = no GT there, blue = GT point nobody covers) | pairs (green true, red false, blue dashed missed).
usage (from the repo root through tools/env/_py14.sh): fail_gallery.py <pair run> <seed> [thr]"""
import json
import os
import pickle
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.environ.setdefault("BFLC_ROOT", ROOT)
for sub in ("evaluation", "training", "training/grid", "training/data", "tools/viz", ".."):
    sys.path.insert(0, os.path.join(ROOT, sub))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import jgt as J  # noqa: E402
import jrule_control as JR  # noqa: E402
import qends as QENDS  # noqa: E402
import qends_eval as QEV  # noqa: E402
import qgraph_eval as QGE  # noqa: E402
import qgraph_readout as QR  # noqa: E402
import qgraph_view as QV  # noqa: E402

run, seed = sys.argv[1], int(sys.argv[2])
THR = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
OUT = os.path.join(ROOT, "temp", "jr", "media", "errors")
os.makedirs(OUT, exist_ok=True)
HIT = 1.0
BOX = 50.0

frames = QGE.val_frames()
dump = pickle.load(open(os.path.join(ROOT, "temp", "e276", f"predfull_e273_p_s{seed}.pkl"), "rb"))
ends = {f["row"]: f for f in pickle.load(open(os.path.join(ROOT, "runs", run, "ends_val.pkl"), "rb"))}
spec = QR.parse("close:1.5@0.5")


def cover(raw, ref):
    """(delivered points, hit flags, GT points, covered flags) inside the box; points every 0.5 m (the scorer's tier counts)"""
    dp = [QGE._densify(p) for p in raw["pts"] if len(p)]
    P = np.concatenate(dp) if dp else np.zeros((0, 2))
    P = P[(np.abs(P) < BOX).all(1)]
    G = ref.pts[(np.abs(ref.pts) < BOX).all(1)]
    hp = (ref.tree.query(P)[0] <= HIT) if len(P) else np.zeros(0, bool)
    cg = (cKDTree(P).query(G)[0] <= HIT) if len(P) else np.zeros(len(G), bool)
    return P, hp, G, cg


def pairs_of(e, jg):
    """delivered pairs (index pairs of keypoints), their matched GT pair or None, and the counts true / false / missed"""
    K = e["logit"].shape[0]
    kp, kd = e["kp_pos"], e["kp_dir"]
    ve = e["valid"][0] & (e["kp_p"][:K] >= 0.5)
    vx = e["valid"][1] & (e["kp_p"][K:] >= 0.5)
    inv_a = inv_b = {}
    gt = set()
    if jg is not None:
        ma, mb = JR._match(kp[:K], kd[:K], ve, jg["ent"]), JR._match(kp[K:], kd[K:], vx, jg["ext"])
        inv_a, inv_b = {int(k): g for g, k in enumerate(ma) if k >= 0}, {int(k): g for g, k in enumerate(mb) if k >= 0}
        gt = {(int(p[0]), int(p[1])) for p in jg["pairs"]}
    p = 1.0 / (1.0 + np.exp(-e["logit"])) * e["allowed"]
    ia, ib = np.nonzero(p >= THR)
    deliv, got = [], set()
    for i, j in zip(ia, ib):
        key = (inv_a.get(int(i), -1), inv_b.get(int(j), -1))
        ok = key in gt
        deliv.append((int(i), int(j), ok))
        if ok:
            got.add(key)
    # all candidates with both ends found: probability and label
    cand = []
    for i, j in zip(*np.nonzero(e["allowed"] & ve[:, None] & vx[None, :])):
        a, b = inv_a.get(int(i), -1), inv_b.get(int(j), -1)
        if a >= 0 and b >= 0:
            cand.append((float(p[i, j]), (a, b) in gt))
    return deliv, gt - got, gt, cand


rec = []
raws = {}
t0 = __import__("time").time()
for b, (f, d) in enumerate(zip(frames, dump)):
    pred = QR.apply(dict(nodes=dict(pos=d["pos"], dirv=d["dirv"], e=d["e"], flags=d["flags"], section=d["section"], s=d.get("s")),
                         links=dict(src=d["link_src"], dst=d["link_dst"], p=d["link_p"], long=d["link_long"], a=d["link_a"], b=d["link_b"])), spec)
    graph, bk = QGE.readout(pred, 0.5, 0.5)
    e = ends[f.row]
    k = e["kp_pos"].shape[0] // 2
    p = 1.0 / (1.0 + np.exp(-e["logit"])) * e["allowed"]
    ia, ib = np.nonzero(p >= THR)
    prs = [(e["kp_pos"][i], e["kp_dir"][i], e["kp_pos"][k + j], e["kp_dir"][k + j]) for i, j in zip(ia, ib)]
    raw, n_c, n_w = QEV.add_connectors(graph.lines, graph.edges, graph.nodes, bk["kind"], bk["piece_links"], np.asarray(pred["links"]["long"], bool), prs)
    ref = f.ref()
    jg = J.junction_gt(QV.load_gt(f.token))
    base_raw = QGE.to_raw(graph)
    cg0 = cover(base_raw, ref)[3]
    P, hp, G, cg = cover(raw, ref)
    deliv, missed, gtp, cand = pairs_of(e, jg)
    rec.append(dict(k=b, token=f.token, scene=int(f.scene), n_gt_pairs=len(gtp), n_true=sum(1 for x in deliv if x[2]), n_false=sum(1 for x in deliv if not x[2]), n_missed=len(missed),
                    recall_base=float(cg0.mean()) if len(cg0) else float("nan"), recall_head=float(cg.mean()) if len(cg) else float("nan"), precision_head=float(hp.mean()) if len(hp) else float("nan"),
                    cand=cand, has_junction=jg is not None))
    raws[b] = raw
    if (b + 1) % 100 == 0:
        print(f"  {b + 1}/{len(frames)}  {__import__('time').time() - t0:.0f} s", flush=True)
json.dump([{k_: v for k_, v in r.items() if k_ != "cand"} for r in rec], open(os.path.join(OUT, "stats.json"), "w"), indent=0)

# ---------------------------------------------------------------------------------------------- distribution
J_ = [r for r in rec if r["n_gt_pairs"] >= 4]
fig, axs = plt.subplots(1, 3, figsize=(17, 4.6))
ax = axs[0]
ax.hist([r["recall_base"] for r in J_], bins=np.linspace(0, 1, 26), alpha=0.6, label="plain graph (B + hard close-node step)", color="#7f7f7f")
ax.hist([r["recall_head"] for r in J_], bins=np.linspace(0, 1, 26), alpha=0.6, label=f"+ connectors (p >= {THR})", color="#2ca02c")
ax.set_xlabel("share of the GT lane points within 1 m of the delivered graph")
ax.set_ylabel(f"junction frames (>= 4 GT pairs): {len(J_)}")
ax.set_title("coverage of the GT lane graph per frame")
ax.legend(fontsize=8)
ax = axs[1]
fr = [r["n_true"] / r["n_gt_pairs"] for r in J_]
ax.hist(fr, bins=np.linspace(0, 1, 21), color="#1f77b4")
ax.set_xlabel("true pairs delivered / GT pairs of the frame")
ax.set_ylabel("junction frames")
ax.set_title(f"pair recall per frame (mean {np.mean(fr):.2f}; frames with none delivered: {sum(1 for x in fr if x == 0)})")
ax = axs[2]
pt = np.array([c[0] for r in rec for c in r["cand"] if c[1]])
pf = np.array([c[0] for r in rec for c in r["cand"] if not c[1]])
bins = np.linspace(0, 1, 21)
ax.hist(pf, bins=bins, alpha=0.6, color="#d62728", label=f"false pairs of two found ends (n {len(pf)})")
ax.hist(pt, bins=bins, alpha=0.6, color="#2ca02c", label=f"true pairs (n {len(pt)})")
ax.set_yscale("log")
ax.axvline(0.5, color="k", lw=1, ls="--")
ax.set_xlabel("pair probability p")
ax.set_title("pair probability, candidates whose two ends were found")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "distribution.png"), dpi=80)
plt.close(fig)
share_true_below = float((pt < 0.5).mean())
print(f"true pairs of two found ends: {len(pt)}, below p 0.5: {share_true_below:.2f}, below 0.2: {(pt < 0.2).mean():.2f}; false: {len(pf)}, above 0.5: {(pf >= 0.5).mean():.3f}, above 0.2: {(pf >= 0.2).mean():.3f}")


# ---------------------------------------------------------------------------------------------- galleries
def draw_frame(axs, r):
    f = frames[r["k"]]
    g = QV.load_gt(f.token)
    jg = J.junction_gt(g)
    ref = f.ref()
    e = ends[f.row]
    raw = raws[r["k"]]
    P, hp, G, cg = cover(raw, ref)
    ax = axs[0]
    for pc in g["pieces"]:
        p = np.asarray(pc["pts"])
        ax.plot(p[:, 0], p[:, 1], color="#b0b0b0", lw=0.9)
    if jg is not None:
        for ent in jg["ent"]:
            ax.scatter(ent[0], ent[1], marker="^", s=26, color="#2ca02c", zorder=4)
        for ext in jg["ext"]:
            ax.scatter(ext[0], ext[1], marker="s", s=22, color="#c2185b", zorder=4)
    ax.set_title(f"{f.token[:8]}  scene {r['scene']}  GT lanes, entries (green) and exits (magenta); {r['n_gt_pairs']} GT pairs", fontsize=8)
    ax = axs[1]
    ax.scatter(G[~cg, 0], G[~cg, 1], s=3, color="#1f77b4", zorder=2, label="GT not covered")
    ax.scatter(P[hp, 0], P[hp, 1], s=3, color="#2ca02c", zorder=3, label="delivered, near GT")
    ax.scatter(P[~hp, 0], P[~hp, 1], s=5, color="#d62728", zorder=4, label="delivered, no GT near")
    ax.set_title(f"delivered graph: GT coverage {r['recall_head']:.2f}, precision {r['precision_head']:.2f} (plain graph coverage {r['recall_base']:.2f})", fontsize=8)
    ax.legend(fontsize=6, loc="lower right", markerscale=3)
    ax = axs[2]
    for pc in g["pieces"]:
        p = np.asarray(pc["pts"])
        ax.plot(p[:, 0], p[:, 1], color="#d0d0d0", lw=0.7, zorder=1)
    deliv, missed, gt, _ = pairs_of(e, jg)
    k = e["kp_pos"].shape[0] // 2
    for i, j, ok in deliv:
        c = QENDS.curve(e["kp_pos"][i], e["kp_dir"][i], e["kp_pos"][k + j], e["kp_dir"][k + j])
        ax.plot(c[:, 0], c[:, 1], color="#2ca02c" if ok else "#d62728", lw=2.2 if ok else 1.3, zorder=3)
    if jg is not None:
        for (a, b) in sorted(missed):
            c = QENDS.curve(jg["ent"][a, :2], jg["ent"][a, 2:], jg["ext"][b, :2], jg["ext"][b, 2:])
            ax.plot(c[:, 0], c[:, 1], color="#1f77b4", lw=1.0, ls="--", zorder=2)
    ax.set_title(f"pairs: {r['n_true']} true (green), {r['n_false']} false (red), {r['n_missed']} of {r['n_gt_pairs']} missed (blue dashed)", fontsize=8)
    for ax in axs:
        ax.set_xlim(-BOX, BOX)
        ax.set_ylim(-BOX, BOX)
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)


def gallery(name, sel):
    fig, axss = plt.subplots(len(sel), 3, figsize=(16.5, 5.6 * len(sel)), squeeze=False)
    for row, r in zip(axss, sel):
        draw_frame(row, r)
    fig.tight_layout()
    path = os.path.join(OUT, f"{name}.png")
    fig.savefig(path, dpi=75)
    plt.close(fig)
    print("wrote", path, [r["token"][:8] for r in sel])


def pick(rows, key, n, reverse=True):
    out, seen = [], set()
    for r in sorted(rows, key=key, reverse=reverse):
        if r["scene"] in seen:
            continue
        seen.add(r["scene"])
        out.append(r)
        if len(out) == n:
            break
    return out


big = [r for r in rec if r["n_gt_pairs"] >= 8]
gallery("misses", pick(big, lambda r: r["n_missed"], 3))
gallery("false_pairs", pick(rec, lambda r: r["n_false"], 2))
gallery("worst_coverage", pick([r for r in rec if r["n_gt_pairs"] >= 4], lambda r: -r["recall_head"], 2))
med = sorted(big, key=lambda r: r["n_true"] / r["n_gt_pairs"])[len(big) // 2]
gallery("typical", [med])
print("done")


# ---------------------------------------------------------------------------------------------- contact sheet: 12 random junction frames, the delivered graph against GT
def sheet(name, sel, ncol=4):
    nrow = (len(sel) + ncol - 1) // ncol
    fig, axs = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 5.4 * nrow), squeeze=False)
    for ax in axs.ravel():
        ax.axis("off")
    for ax, r in zip(axs.ravel(), sel):
        ax.axis("on")
        f = frames[r["k"]]
        ref = f.ref()
        P, hp, G, cg = cover(raws[r["k"]], ref)
        ax.scatter(G[~cg, 0], G[~cg, 1], s=2, color="#1f77b4", zorder=2)
        ax.scatter(P[hp, 0], P[hp, 1], s=2, color="#2ca02c", zorder=3)
        ax.scatter(P[~hp, 0], P[~hp, 1], s=3, color="#d62728", zorder=4)
        ax.set_title(f"{f.token[:8]}: coverage {r['recall_head']:.2f}, precision {r['precision_head']:.2f}, pairs {r['n_true']}/{r['n_gt_pairs']}", fontsize=8)
        ax.set_xlim(-BOX, BOX)
        ax.set_ylim(-BOX, BOX)
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
    fig.suptitle("12 random junction frames (>= 6 GT pairs): green = delivered near GT, red = delivered without GT, blue = GT not covered", fontsize=10)
    fig.tight_layout()
    path = os.path.join(OUT, f"{name}.png")
    fig.savefig(path, dpi=70)
    plt.close(fig)
    print("wrote", path)


rng = np.random.default_rng(1)
pool = [r for r in rec if r["n_gt_pairs"] >= 6]
sheet("random12", [pool[i] for i in sorted(rng.choice(len(pool), 12, replace=False))])


# ---------------------------------------------------------------------------------------------- error videos (same three panels per keyframe of a scene)
def video(name, fps=2):
    import cv2
    sid, nm, toks = QV.resolve_scene(name)
    rows = [r for r in rec if r["token"] in set(toks)]
    if not rows:
        print("no frames for", name)
        return
    path = os.path.join(OUT, f"{nm}_errors.webm")
    wr = None
    for r in rows:
        fig, axs = plt.subplots(1, 3, figsize=(16.5, 5.6))
        draw_frame(axs, r)
        fig.tight_layout()
        fig.canvas.draw()
        img = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba())[:, :, :3], cv2.COLOR_RGB2BGR)
        plt.close(fig)
        w = 1400
        h = int(round(img.shape[0] * w / img.shape[1] / 2) * 2)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        if wr is None:
            wr = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"vp09"), fps, (w, h))
        wr.write(img)
    wr.release()
    print("wrote", path, len(rows), "frames", round(os.path.getsize(path) / 1e6, 2), "MB")


for nm in ("scene-0096", "scene-0221", "scene-0017", "scene-0108"):
    video(nm)
print("videos done")
