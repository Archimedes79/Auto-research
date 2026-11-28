"""One command that checks every stage of a lane run on its stored live dump and says what goes wrong:

    python tools/diagnose/diagnose.py <run> [--split val] [--out DIR] [--no-gpu]

<run> is a run folder of the problem's run root with config.json (the decode) and token_<split>_live.pkl (the step `predict`).
Written to <runs root>/diagnose/<run>/: diagnose.json, summary.md and pictures.
  stages     per ground-truth point (one per metre): pixel evidence -> confident node -> token -> kept node -> delivered line (the pixel and
             node stages need the GPU: B' of the run's model.pt)
  positions  delivered nodes in cells the true lane does not cross, median distance, slope of predicted against required cell offset
  forks      forks per frame, forks holding a jump that another branch reaches anyway, polylines per frame and their length
  failures   the frames with the lowest share of true lane length covered, pictured (delivered lines by distance to the truth, direction
             arrows, true lanes missed in blue)
Reads only; no record, nothing of a run folder changes."""
import argparse
import json
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROBLEM = os.path.normpath(os.path.join(HERE, "..", ".."))
REPO = os.path.normpath(os.path.join(PROBLEM, "..", ".."))
for sub in json.load(open(os.path.join(PROBLEM, "problem.json")))["pythonpath"]:
    sys.path.insert(0, os.path.join(PROBLEM, sub))
sys.path.insert(0, os.path.join(REPO, "autoresearch"))
sys.path.insert(0, REPO)

import GtGenerator  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

R_STAGE = 1.5
STAGES = ("pixel evidence", "confident node", "token", "kept node", "delivered line")


def resample(p, ds):
    p = np.asarray(p, float)[:, :2]
    if len(p) < 2:
        return p, np.zeros_like(p)
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    if s[-1] < 1e-6:
        return p[:1], np.zeros_like(p[:1])
    t = np.r_[np.arange(0, s[-1], ds), s[-1]]
    q = np.c_[np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])]
    tan = np.gradient(q, axis=0)
    return q, tan / np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-9)


def near(points, to):
    return np.full(len(to), np.inf) if not len(points) else cKDTree(points).query(to)[0]


def intersection_flags(token, pts):
    p, flag = [], []
    for piece in GtGenerator.load_lane(token)["pieces"]:
        q = np.asarray(piece["pts"], float)[:, :2]
        f = np.zeros(len(q), bool)
        for s in piece["segments"]:
            if s["in_intersection"]:
                f[s["i0"]: s["i1"] + 1] = True
        p.append(q)
        flag.append(f)
    return np.concatenate(flag)[cKDTree(np.vstack(p)).query(pts)[1]]


def redundant_links(src, dst, hops=5):
    out = {}
    for l, (a, b) in enumerate(zip(src, dst)):
        out.setdefault(a, []).append((b, l))
    drop, forks, with_drop = [], 0, 0
    for a, links in out.items():
        if len(links) < 2:
            continue
        forks += 1
        found = False
        for b, l in links:
            seen, front = {a}, [c for c, _ in links if c != b]
            for _ in range(hops):
                if b in front:
                    drop.append(l)
                    found = True
                    break
                seen.update(front)
                front = [c for q in front for c, _ in out.get(q, []) if c not in seen]
        with_drop += found
    return drop, forks, with_drop


def picture(path, title, gt, own, lines):
    fig, ax = plt.subplots(figsize=(7.0, 7.8), dpi=110)
    for q, t in own:
        ax.plot(q[t, 0], q[t, 1], color="#ffd23f", lw=12, alpha=0.5, solid_capstyle="round", zorder=0)
    G = np.vstack([resample(g, 0.25)[0] for g in gt])
    ax.add_collection(LineCollection(gt, colors="#b8b8b8", linewidths=5.5, capstyle="round", zorder=1))
    dense = [resample(l, 0.25)[0] for l in lines if len(l) > 1]
    P = np.vstack(dense) if dense else np.zeros((0, 2))
    for g in gt:
        q = resample(g, 0.25)[0]
        far = np.r_[False, near(P, q) > 2.0, False]
        for a, b in zip(np.flatnonzero(~far[:-1] & far[1:]), np.flatnonzero(far[:-1] & ~far[1:])):
            ax.plot(q[a:b, 0], q[a:b, 1], color="#1f4fff", lw=2.8, solid_capstyle="round", zorder=2)
    tree = cKDTree(G)
    placed = []
    for q in sorted(dense, key=len, reverse=True):
        d = tree.query((q[:-1] + q[1:]) / 2)[0]
        cols = ["#1a9a2a" if x <= 1 else "#f0a000" if x <= 2 else "#e02020" for x in d]
        ax.add_collection(LineCollection([[q[i], q[i + 1]] for i in range(len(q) - 1)], colors=cols, linewidths=2.2, capstyle="round", zorder=3))
        for i in range(min(len(q) // 2, 14), len(q) - 2, 8):
            at, t = q[i], q[min(i + 8, len(q) - 1)] - q[max(i - 8, 0)]
            t = t / max(np.linalg.norm(t), 1e-9)
            n = np.array([-t[1], t[0]])
            if np.abs(at).max() > 47.5 or any(np.linalg.norm(at - o) < 11 for o in placed):
                continue
            ax.add_patch(Polygon([at + t * 1.9, at - t * 1.1 + n * 1.4, at - t * 0.3, at - t * 1.1 - n * 1.4], closed=True, facecolor="#0b4d12", edgecolor="white", lw=0.6, zorder=5))
            placed.append(at)
    ax.plot(0, 0, marker="^", color="#111827", ms=11, mec="white", zorder=8)
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10, loc="left")
    fig.legend(handles=[Line2D([], [], color="#b8b8b8", lw=5, label="true lane"), Line2D([], [], color="#ffd23f", lw=8, alpha=0.5, label="own lane"),
                        Line2D([], [], color="#1a9a2a", lw=2.5, label="within 1 m"), Line2D([], [], color="#f0a000", lw=2.5, label="1 - 2 m"),
                        Line2D([], [], color="#e02020", lw=2.5, label="over 2 m"), Line2D([], [], color="#1f4fff", lw=2.5, label="true lane missed (2 m)")],
               loc="lower center", ncol=3, fontsize=8.5, frameon=False)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.955, bottom=0.1)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run")
    ap.add_argument("--split", default="val", choices=("val",))
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-gpu", action="store_true", help="skip the pixel and node stages")
    ap.add_argument("--worst", type=int, default=6, help="pictures of the frames with the least true lane length covered")
    a = ap.parse_args(argv)
    import adapter as AD
    import bflc_paths as BP
    import lane_readout as RO
    run_dir = os.path.join(BP.RUNS, a.run)
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    dec = cfg["decode"]
    out_dir = a.out or os.path.join(BP.RUNS, "diagnose", a.run)
    os.makedirs(out_dir, exist_ok=True)
    frames = AD.frames_of(a.split)
    dump = pickle.load(open(os.path.join(run_dir, f"token_{a.split}_live.pkl"), "rb"))["frames"]
    assert [f["token"] for f in dump] == [f.token for f in frames], "the dump is not of these frames"
    QE = AD._qe()
    gpu = not a.no_gpu
    if gpu:
        import torch
        import lane_net as LN
        import lane_tok_data as TD
        data = TD.TokData(a.split, 2.0, feats=True)
        assert list(data.tokens) == [f.token for f in frames]
        net = LN.LaneNet.from_checkpoint(os.path.join(run_dir, "model.pt"), "cuda")[0]
        c1 = -49.5 + np.arange(100)
        GX, GY = np.meshgrid(c1, c1, indexing="ij")
    cov = {s: [] for s in STAGES}
    inter, per_frame = [], []
    pos_rows = dict(off=[], dist=[], p=[], r=[])
    fk = dict(forks=0, with_jump=0, links=0, jumps=0, lengths=[], polylines=0)
    maps = {}
    for k, (fr, d) in enumerate(zip(frames, dump)):
        if gpu and k % 4 == 0:
            x = torch.stack([data[i]["x"] for i in range(k, min(k + 4, len(frames)))]).cuda().float()
            with torch.no_grad():
                P = net.b.pyr(x)
                no = net.tokens(x)[0]
            maps = dict(pix=P["centre1"][:, 0].float().sigmoid().cpu().numpy(), e=no["e"].float().sigmoid().cpu().numpy(), pos=no["pos"].float().cpu().numpy())
        t, v = d["tokens_np"], np.asarray(d["tokens_np"]["valid"], bool)
        pred = RO.graph(t, d, v, RO.Decode(**dec))
        graph, bk = QE.readout(pred, 0.5, 0.5)
        lines = [np.asarray(l, float)[:, :2] for l in QE.to_raw(graph)["pts"] if len(l) > 1]
        gt = [np.asarray(p, float)[:, :2] for p in fr.gt["pts"] if len(p) > 1]
        if not gt:
            continue
        G = np.vstack([resample(g, 1.0)[0] for g in gt])
        pos = np.asarray(t["pos"], float)
        keep = v & (np.asarray(d["pnone"]) < np.float32(dec["keep_thr"]))
        delivered = np.vstack([resample(l, 0.5)[0] for l in lines]) if lines else np.zeros((0, 2))
        sets = {"token": pos[v], "kept node": pos[keep], "delivered line": delivered}
        if gpu:
            i = k % 4
            conf = maps["e"][i] >= 0.5
            sets["pixel evidence"] = np.c_[GX[maps["pix"][i] > 0.5], GY[maps["pix"][i] > 0.5]]
            sets["confident node"] = np.c_[maps["pos"][i, 0][conf], maps["pos"][i, 1][conf]]
        for s, q in sets.items():
            cov[s] += list(near(q, G) <= R_STAGE)
        inter += list(intersection_flags(fr.token, G))
        per_frame.append(dict(k=k, scene=int(fr.scene), token=fr.token, covered=float(np.mean(near(delivered, G) <= 2.0)),
                              good=float(np.mean(near(np.vstack([resample(g, 0.25)[0] for g in gt]), delivered) <= 1.0)) if len(delivered) else 0.0))
        # positions of the delivered nodes against their 2 m cells
        dense = [resample(g, 0.1) for g in gt]
        D = np.vstack([q for q, _ in dense])
        T = np.vstack([u for _, u in dense])
        tree = cKDTree(D)
        npos = np.asarray(pred["nodes"]["pos"], float).reshape(-1, 2)
        cell = np.asarray(pred["nodes"]["cell"], int).reshape(-1, 2)
        used = sorted({n for chain in bk["piece_nodes"] for n in chain})
        for n in used:
            c = np.array([-49.0 + 2 * cell[n, 0], -49.0 + 2 * cell[n, 1]])
            cand = np.asarray(tree.query_ball_point(c, 1.4143), int)
            inside = cand[np.all(np.abs(D[cand] - c) <= 1.0, axis=1)] if len(cand) else cand
            dist, j = tree.query(npos[n])
            pos_rows["off"].append(len(inside) == 0)
            pos_rows["dist"].append(dist)
            if len(inside) and dist < 2.0:
                j = inside[np.argmin(np.linalg.norm(D[inside] - c, axis=1))]
                normal = np.array([-T[j, 1], T[j, 0]])
                pos_rows["p"].append(float((npos[n] - c) @ normal))
                pos_rows["r"].append(float((D[j] - c) @ normal))
        src, dst = (np.asarray(pred["links"][x]).astype(int) for x in ("src", "dst"))
        drop, forks, with_drop = redundant_links(src, dst)
        fk["forks"] += forks
        fk["with_jump"] += with_drop
        fk["links"] += len(src)
        fk["jumps"] += len(set(drop))
        fk["polylines"] += len(lines)
        fk["lengths"] += [float(np.linalg.norm(np.diff(l, axis=0), axis=1).sum()) for l in lines]
    inter = np.asarray(inter, bool)
    stages = {}
    for s in STAGES:
        if cov[s]:
            c = np.asarray(cov[s], bool)
            stages[s] = dict(all=float(c.mean()), outside_intersections=float(c[~inter].mean()), inside_intersections=float(c[inter].mean()))
    p, r = np.asarray(pos_rows["p"]), np.asarray(pos_rows["r"])
    positions = dict(nodes=len(pos_rows["off"]), off_lane_cell_share=float(np.mean(pos_rows["off"])), median_distance_m=float(np.median(pos_rows["dist"])),
                     share_over_half_m=float(np.mean(np.asarray(pos_rows["dist"]) > 0.5)), offset_slope=float(np.polyfit(r, p, 1)[0]) if len(r) > 2 else None)
    forks = dict(per_frame=fk["forks"] / len(frames), share_with_a_jump=fk["with_jump"] / max(fk["forks"], 1), jump_share_of_links=fk["jumps"] / max(fk["links"], 1),
                 polylines_per_frame=fk["polylines"] / len(frames), median_polyline_m=float(np.median(fk["lengths"])) if fk["lengths"] else None)
    worst = sorted(per_frame, key=lambda x: x["covered"] * 0.5 + x["good"] * 0.5)[: a.worst]
    for n, w in enumerate(worst):
        fr, d = frames[w["k"]], dump[w["k"]]
        t, v = d["tokens_np"], np.asarray(d["tokens_np"]["valid"], bool)
        pred = RO.graph(t, d, v, RO.Decode(**dec))
        lines = [np.asarray(l, float)[:, :2] for l in QE.to_raw(QE.readout(pred, 0.5, 0.5)[0])["pts"] if len(l) > 1]
        own = [(np.asarray(pc["pts"], float), np.asarray(pc.get("tier", []), int) == 1) for pc in GtGenerator.load_lane(fr.token)["pieces"]]
        picture(os.path.join(out_dir, f"worst_{n + 1}.png"), f"{a.run}, scene {w['scene']}, {str(fr.token)[:8]}: covered {w['covered']:.0%}, true length within 1 m {w['good']:.0%}",
                [np.asarray(g, float)[:, :2] for g in fr.gt["pts"] if len(g) > 1], [(q, m) for q, m in own if len(m) and m.any()], lines)
    res = json.load(open(os.path.join(run_dir, "result.json"))) if os.path.exists(os.path.join(run_dir, "result.json")) else None
    out = dict(run=a.run, split=a.split, frames=len(frames), radius_m=R_STAGE, stages=stages, positions=positions, forks=forks, worst=worst)
    json.dump(out, open(os.path.join(out_dir, "diagnose.json"), "w"), indent=1)
    md = [f"# {a.run}: stage check on {a.split} ({len(frames)} frames)", "", "## Where the true lanes get lost (share of ground-truth points within 1.5 m of the stage)", "",
          "| stage | all | outside intersections | inside intersections |", "|---|---|---|---|"]
    md += [f"| {s} | {v['all']:.1%} | {v['outside_intersections']:.1%} | {v['inside_intersections']:.1%} |" for s, v in stages.items()]
    md += ["", "Node targets leave intersections out by design: connectors there come from long links.", "", "## Node positions (delivered nodes)", ""]
    md += [f"- {k.replace('_', ' ')}: {v:.3f}" for k, v in positions.items() if isinstance(v, float)]
    md += ["", "## Forks and fragments", ""] + [f"- {k.replace('_', ' ')}: {v:.3f}" for k, v in forks.items() if isinstance(v, float)]
    md += ["", "## Worst frames", ""] + [f"![{n + 1}](worst_{n + 1}.png)" for n in range(len(worst))]
    if res:
        md += ["", "## Scores", "", ", ".join(f"{k.split('.')[-1]} {v:.4f}" for k, v in res.get("numbers", {}).items() if k.split(".")[-1] in ("topo_F1", "F1@1.0", "lanes_found", "junction_F1", "vert100_dp"))]
    open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("written", out_dir)
    print(json.dumps(dict(stages=stages, positions=positions, forks=forks)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
