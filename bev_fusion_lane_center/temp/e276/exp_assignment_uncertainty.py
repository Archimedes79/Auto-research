"""E276 analyses of the video notes 2 and 8 from a FULL prediction dump (qgraph_readout.dump_full, `score --dump-full`; exact map): the assignment of the delivered graph to the map elements (is it
1:1?) and the uncertainty of the links (are the link probabilities calibrated, what does abstaining cost?). CPU only.
    wsl env CUDA_VISIBLE_DEVICES= bash tools/env/_py14.sh exp_assignment_uncertainty.py <dump pkl> <l4 store folder> [--rows N] [--pictures K]
Assignment (e >= 0.5 nodes): per node the map piece of the largest probability or the dustbin; per delivered SECTION (nodes with one section id, >= 3 nodes) the majority map piece; counts:
  node accuracy (the scorer's rule: the map piece belongs to the GT lane of the node), section correctness, map pieces CLAIMED by two or more sections (duplicates: the assignment is not 1:1),
  GT pieces (>= 10 m inside the window) claimed by no section (not covered), share of delivered nodes sent to the dustbin and how many of them sit on a GT lane.
Links (both ends delivered, e >= 0.5, p >= 0.05, ends matched one-to-one to GT nodes within 2 m): a link is RIGHT when the two GT nodes are joined by a GT link; reliability by probability bin and
the expected calibration error (ECE), risk-coverage (precision of the links kept at probability >= t against the share of the GT links delivered), per kind (short / long) and per tier of the source node.
Pictures: top views with the delivered links drawn by their probability (width and opacity), dashed below 0.5, green right / red wrong, GT lanes grey (results/E276/figures/).
"""
import gzip
import json
import os
import pickle
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = os.environ["BFLC_ROOT"]
for sub in ("training", "training/grid", "training/data", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import qgraph_eval as QE  # noqa: E402
from lane_task import measures as MS  # noqa: E402

dump_path, store = sys.argv[1], sys.argv[2]
dump_path = dump_path if os.path.isabs(dump_path) else os.path.join(ROOT, dump_path)           # `_py14.sh` runs in another folder: a relative path is relative to the project root
n_rows = int(sys.argv[sys.argv.index("--rows") + 1]) if "--rows" in sys.argv else 0
n_pics = int(sys.argv[sys.argv.index("--pictures") + 1]) if "--pictures" in sys.argv else 0
dump = pickle.load(open(dump_path, "rb"))
frames = QE.val_frames()[: len(dump)]
if n_rows:
    dump, frames = dump[:n_rows], frames[:n_rows]
tag = os.path.basename(dump_path).replace("predfull_", "").replace(".pkl", "")
A = dict(nodes=0, node_right=0, node_dust=0, node_dust_on_lane=0, sections=0, sec_right=0, claims=[], gt_pieces=0, gt_uncovered=0, per_sec_pieces=[])
bins = np.linspace(0, 1, 11)
rel = np.zeros((10, 3))                                           # per bin: n, sum p, n right
links = []                                                        # (p, right, long, tier)
GT_LINKS = [0]
pics = []
for f, d in zip(frames, dump):
    assert d["row"] == f.row
    g = pickle.load(gzip.open(os.path.join(store, f.token + ".pkl.gz"), "rb"))
    ref = QE.LaneRef(g)
    keep = d["e"] >= 0.5
    P = d["pos"][keep]
    kidx = np.flatnonzero(keep)
    # ------------------------------------------------------------------ assignment
    if "assign_P" in d and len(P):
        ids = d["map_ids"]
        Kn = len(ids)
        amax = d["assign_P"][keep].argmax(1)
        dist, tied, _ = ref.near(P)
        id2k = {}
        for k, i in enumerate(ids):
            id2k.setdefault(i, set()).add(k)
        gt_ids = ref.ids
        sec = d["section"][keep]
        A["nodes"] += len(P)
        right = np.zeros(len(P), bool)
        for n in range(len(P)):
            if dist[n] <= QE.LANE_NEAR:
                ok = set().union(*(id2k.get(gt_ids[i], {Kn}) for i in tied[n]))
                right[n] = amax[n] in ok
        on_lane = dist <= QE.LANE_NEAR
        A["node_right"] += int(right[on_lane].sum())
        A["nodes_on_lane"] = A.get("nodes_on_lane", 0) + int(on_lane.sum())
        dust = amax == Kn
        A["node_dust"] += int(dust.sum())
        A["node_dust_on_lane"] += int((dust & on_lane).sum())
        claimed = {}
        for s in np.unique(sec):
            m = (sec == s) & ~dust
            if (sec == s).sum() < 3 or not m.any():
                continue
            cnt = np.bincount(amax[m], minlength=Kn + 1)
            maj = int(cnt.argmax())
            A["sections"] += 1
            A["per_sec_pieces"].append(int((cnt > 0.1 * m.sum()).sum()))
            sec_nodes = np.flatnonzero((sec == s) & on_lane)
            if len(sec_nodes):
                gtp = np.bincount(np.concatenate([tied[n] for n in sec_nodes]).astype(int), minlength=len(gt_ids))
                gmaj = int(gtp.argmax())
                A["sec_right"] += int(maj in id2k.get(gt_ids[gmaj], {Kn}))
            claimed.setdefault(ids[maj], []).append(int(s))             # by map piece id (the parts of one piece share an id)
        A["claims"] += [len(v) for v in claimed.values()]
        lengths = {i: p["length"] for i, p in zip(gt_ids, g["pieces"])}
        covered = set(claimed.keys())
        for i, L in lengths.items():
            if L >= 10.0:
                A["gt_pieces"] += 1
                A["gt_uncovered"] += int(i not in covered)
    # ------------------------------------------------------------------ links
    if len(P) and len(d["link_src"]):
        G, nt = ref.nt["pos"], ref.nt
        pairs = MS.match_1to1(P, G, QE.NODE_TOL)
        g_of_p = -np.ones(len(P), int)
        if len(pairs):
            g_of_p[pairs[:, 0]] = pairs[:, 1]
        local = -np.ones(len(d["e"]), int)
        local[kidx] = np.arange(len(kidx))
        gt_link = {(int(a), int(b)) for a, b in zip(nt["src"], nt["dst"])}
        p_of_g = -np.ones(len(G), int)
        if len(pairs):
            p_of_g[pairs[:, 1]] = pairs[:, 0]
        GT_LINKS[0] += sum(1 for a, b in gt_link if p_of_g[a] >= 0 and p_of_g[b] >= 0)             # the GT links between matched nodes: the denominator of the coverage
        for s_, t_, p_, lg in zip(d["link_src"], d["link_dst"], d["link_p"], d["link_long"]):
            a, b = local[s_], local[t_]
            if a < 0 or b < 0 or g_of_p[a] < 0 or g_of_p[b] < 0:
                continue
            ok = (int(g_of_p[a]), int(g_of_p[b])) in gt_link
            tier = int(nt["tier"][g_of_p[a]]) if ref.labelled else 0
            links.append((float(p_), bool(ok), bool(lg), tier, f.k))
            i = min(int(p_ * 10), 9)
            rel[i] += (1, p_, ok)
    if n_pics and len(pics) < n_pics and len(P) > 30 and len(d["link_src"]):
        pics.append((f, g, d, kidx))
# ---------------------------------------------------------------------- numbers
out = {}
if A["nodes"]:
    cl = np.array(A["claims"]) if A["claims"] else np.zeros(1)
    out["assignment"] = dict(delivered_nodes=A["nodes"], node_accuracy_on_gt_lane=A["node_right"] / max(A["nodes_on_lane"], 1), share_sent_to_dustbin=A["node_dust"] / A["nodes"],
                             share_of_dustbin_nodes_on_a_gt_lane=A["node_dust_on_lane"] / max(A["node_dust"], 1), sections=A["sections"], section_correct=A["sec_right"] / max(A["sections"], 1),
                             map_pieces_claimed=int(len(cl)), claimed_by_two_or_more_sections=float((cl >= 2).mean()), claims_per_piece_mean=float(cl.mean()),
                             gt_pieces_ge_10m=A["gt_pieces"], gt_pieces_claimed_by_no_section=A["gt_uncovered"] / max(A["gt_pieces"], 1),
                             map_pieces_per_section_mean=float(np.mean(A["per_sec_pieces"])) if A["per_sec_pieces"] else None)
if links:
    L = np.array([(p, r, lg, t) for p, r, lg, t, _ in links], float)
    ece = sum(rel[i, 0] / rel[:, 0].sum() * abs(rel[i, 1] / rel[i, 0] - rel[i, 2] / rel[i, 0]) for i in range(10) if rel[i, 0])
    out["links"] = dict(n=len(L), right_share=float(L[:, 1].mean()), ece=float(ece), reliability=[dict(p_lo=float(bins[i]), n=int(rel[i, 0]), mean_p=float(rel[i, 1] / rel[i, 0]) if rel[i, 0] else None,
                                                                                                      right=float(rel[i, 2] / rel[i, 0]) if rel[i, 0] else None) for i in range(10)])
    rc = []
    gt_total = max(float(GT_LINKS[0]), 1.0)
    for t in (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9):
        m = L[:, 0] >= t
        rc.append(dict(threshold=t, links=int(m.sum()), precision=float(L[m, 1].mean()) if m.any() else None, share_of_gt_links_delivered=float(L[m, 1].sum() / gt_total)))
    out["links"]["risk_coverage"] = rc
    for name, sel in (("short", L[:, 2] == 0), ("long", L[:, 2] == 1), ("tier 1", L[:, 3] == 1), ("tier 2", L[:, 3] == 2), ("tier 3", L[:, 3] == 3)):
        if sel.any():
            m = sel & (L[:, 0] >= 0.5)
            out["links"][f"p>=0.5 {name}"] = dict(n=int(m.sum()), precision=float(L[m, 1].mean()) if m.any() else None, candidates=int(sel.sum()), right_share_of_candidates=float(L[sel, 1].mean()))
json.dump(dict(dump=tag, result=out), open(os.path.join(ROOT, "temp", "e276", f"assignment_uncertainty_{tag}.json"), "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float)[:6000])
# ---------------------------------------------------------------------- pictures
if pics:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(os.path.join(ROOT, "results", "E276", "figures"), exist_ok=True)
    cols = min(3, len(pics))
    fig, axes = plt.subplots((len(pics) + cols - 1) // cols, cols, figsize=(7 * cols, 7 * ((len(pics) + cols - 1) // cols)), squeeze=False)
    for ax, (f, g, d, kidx) in zip(axes.ravel(), pics):
        ref = QE.LaneRef(g)
        ax.plot(ref.pts[:, 0], ref.pts[:, 1], ".", ms=1, color="#c9ccd1")
        G, nt = ref.nt["pos"], ref.nt
        keep = d["e"] >= 0.5
        P = d["pos"][keep]
        local = -np.ones(len(d["e"]), int)
        local[kidx] = np.arange(len(kidx))
        pairs = MS.match_1to1(P, G, QE.NODE_TOL)
        g_of_p = -np.ones(len(P), int)
        if len(pairs):
            g_of_p[pairs[:, 0]] = pairs[:, 1]
        gt_link = {(int(a), int(b)) for a, b in zip(nt["src"], nt["dst"])}
        for s_, t_, p_, lg, a_, b_ in zip(d["link_src"], d["link_dst"], d["link_p"], d["link_long"], d["link_a"], d["link_b"]):
            a, b = local[s_], local[t_]
            if a < 0 or b < 0:
                continue
            ok = g_of_p[a] >= 0 and g_of_p[b] >= 0 and (int(g_of_p[a]), int(g_of_p[b])) in gt_link
            matched = g_of_p[a] >= 0 and g_of_p[b] >= 0
            col = "#2ca02c" if ok else ("#d62728" if matched else "#ff7f0e")
            ax.plot([d["pos"][s_][0], d["pos"][t_][0]], [d["pos"][s_][1], d["pos"][t_][1]], "-" if p_ >= 0.5 else "--", color=col, alpha=float(max(0.15, min(1.0, p_))), lw=0.6 + 2.2 * float(p_))
        ax.scatter(P[:, 0], P[:, 1], s=6, c="k", alpha=0.6)
        ax.plot([0], [0], "s", color="#1f77b4")
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
        ax.set_aspect("equal")
        ax.set_title(f"{f.token[:8]}  links: width / opacity = probability, dashed < 0.5; green right, red wrong, orange end not matched", fontsize=7)
    fig.tight_layout()
    path = os.path.join(ROOT, "results", "E276", "figures", f"links_uncertainty_{tag}.png")
    fig.savefig(path, dpi=70)
    print(path)
