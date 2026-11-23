"""E389 / E389b report: the example frames of the route-association head in the style of the reports of 2026-11-18, chosen by a WRITTEN RULE and drawn from the stored dumps (no model run, CPU).

    bash tools/env/_py14.sh route_frames_fig.py --seed 0 --out <absolute dir>   # _py14.sh changes the directory: give an absolute --out; needs torch (the gcar python) and WSL for the camera
    bash tools/env/_py14.sh route_frames_fig.py --selftest                       # the rule, the colour rule, the pieces of the error route, no data files

Per frame one png: the front camera with the GT lane graph projected (the old helper `tools/viz/qgraph_view.py`), a small table of the frame, and below the BEV panels side by side with shared axes: the head WITHOUT the map (z),
the head WITH the map (m), and for the wrong-map case the head with the WRONG route. In a panel with a map the SD route is drawn as its PIECES (in driving order, the map error applied piece by piece as the
loader does): a thick translucent band in its own colour, the number of the piece at its start, an arrow for the direction of travel. The PERCEPTION RESULT is the delivered graph of the champion (the nodes at p_none < 0.6 and
the links of the E375 decode of the E373 dump: link 0.3 short / 0.2 long, no none gate; Hermite curves as the scorer reads them). COLOUR RULE (the owner's): a delivered node the head flags as on the route whose GT node IS on the
route ('tp') has the colour of the route piece it belongs to (the nearest piece of the exact route); a link between two such nodes of the same piece has that colour too. Flagged but not on the route ('fp') = amber with a
cross; on the route but not flagged ('fn') = the piece colour, pale, hollow marker, dashed link; not flagged and not on the route ('tn') = thin grey. No red. The head without the map sees no pieces: they are drawn thin and dashed
as the TRUE route (not seen by it), so that the colours still mean something.
The frames are chosen by the rule (fixed before any picture was looked at; they are NOT re-picked): typical = the fork frame whose gain m - z is closest to the median gain of the fork frames, gain = the largest m - z,
loss = the smallest m - z, wrong = the largest drop of m from the mode error to the mode wrong; candidates = frames with a route and at least MIN_NODES covered nodes. Writes `e389_frame_<case>.png` and `e389_frames.json`.
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for sub_ in ("evaluation", "training/grid", "training/data", "tools/viz"):
    sys.path.insert(0, os.path.join(ROOT, sub_))
MIN_NODES = 30
LINK_THR, LINK_LONG, KEEP = 0.3, 0.2, 0.6                                           # the E375 decode of the champion (D115 keep 0.6)
PALETTE = ["#332288", "#44AA99", "#AA4499", "#117733", "#88CCEE", "#882255", "#999933", "#8C510A"]   # Paul Tol's muted set + a brown (colour-blind safe), pieces 1 - 8, repeated after that; no red, no amber
AMBER, GREY, WRONG = "#E69F00", "#a0a0a0", "#222222"                                                  # a false positive; a delivered node that is not on the route and not flagged; the bands of a WRONG map (not a palette colour)
OUT_KEYS = ("tp", "fn", "fp", "tn")


def piece_color(k):
    return PALETTE[k % len(PALETTE)]


def f1_of(tp, fp, fn):
    return 2.0 * tp / max(2.0 * tp + fp + fn, 1e-12)


def frame_f1(cov, lab, flag):
    """F1 of one frame over the covered nodes: cov (K,) bool, lab (K,) 0 / 1, flag (K,) bool -> (f1, tp, fp, fn)"""
    tp, fp, fn = int((cov & (lab == 1) & flag).sum()), int((cov & (lab == 0) & flag).sum()), int((cov & (lab == 1) & ~flag).sum())
    return f1_of(tp, fp, fn), tp, fp, fn


def outcome(lab, flag):
    """'tp' flagged and on the route, 'tn' not flagged and off the route, 'fp' flagged but off the route, 'fn' not flagged but on the route (arrays; a label -1 = no GT node counts as off the route)"""
    on = np.asarray(lab) == 1
    flag = np.asarray(flag, bool)
    return np.where(flag & on, "tp", np.where(~flag & ~on, "tn", np.where(flag, "fp", "fn")))


def choose(gain, drop, fork, ok):
    """the written rule -> {case: frame index}; gain = F1_m - F1_z per frame, drop = F1_m(error) - F1_m(wrong), fork (F,) bool, ok (F,) bool = candidate frames (a route and >= MIN_NODES covered nodes)"""
    gain, drop = np.where(ok, gain, np.nan), np.where(ok, drop, np.nan)
    fk = np.where(fork & ok)[0]
    med = float(np.median(gain[fk]))
    return dict(typical=int(fk[np.argmin(np.abs(gain[fk] - med))]), gain=int(np.nanargmax(gain)), loss=int(np.nanargmin(gain)), wrong=int(np.nanargmax(drop)))


def limits(pts, margin=4.0, min_half=25.0):
    """the crop of a frame: a SQUARE box round the points plus a margin (at least +-min_half m, at most +-50 m): -> (xlim, ylim)"""
    lo, hi = pts.min(0) - margin, pts.max(0) + margin
    c, half = (lo + hi) / 2.0, min(max(float((hi - lo).max()) / 2.0, min_half), 50.0)
    c = np.clip(c, -50.0 + half, 50.0 - half)
    return (float(c[0] - half), float(c[0] + half)), (float(c[1] - half), float(c[1] + half))


# ------------------------------------------------------------------------------------------------------------------------------ the route pieces and the colour rule
def error_pieces(polys, token):
    """the pieces of the FIXED `error` route of a frame (`lane_route_data.error_route`), one polyline per piece, the pose error and the degradation applied PIECE BY PIECE exactly as the loader does it: -> [(poly (n, 2), count (n,))]
    in driving order; sampled with `lane_route_data.sample_route` they give the route table the head saw (selftest)"""
    import lane_route_data as LRD
    import sd_lib as SD
    rng = LRD.token_rng(token)
    shift, yaw = LRD.pose_error(rng, *LRD.ERROR_SHIFT)
    out = []
    for poly, cnt in SD.degrade_sd(dict(route=polys, roads=[]), rng)["route"]:
        poly = np.asarray(poly, float)
        if len(poly) >= 2:
            out.append((poly @ LRD.rot(yaw).T + shift, np.asarray(cnt, float)))
    return out


def piece_of(pos, polys):
    """(K,) int: the index of the route piece nearest to each position (-1 without a piece); polys = [(poly, count)] of the exact route"""
    from scipy.spatial import cKDTree
    pos = np.asarray(pos, float).reshape(-1, 2)
    pts, ids = [], []
    for k, (poly, _) in enumerate(polys):
        poly = np.asarray(poly, float)
        if len(poly) < 2:
            continue
        s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(poly, axis=0), axis=1))]
        t = np.arange(0.0, s[-1] + 1e-9, 0.5) if s[-1] > 1e-6 else np.zeros(1)
        pts.append(np.c_[np.interp(t, s, poly[:, 0]), np.interp(t, s, poly[:, 1])])
        ids.append(np.full(len(t), k))
    if not pts:
        return -np.ones(len(pos), int)
    return np.concatenate(ids)[cKDTree(np.concatenate(pts)).query(pos)[1]]


def label_positions(bands, xl, yl, min_d=6.0, inset=2.5):
    """one (x, y) per band for its number: the first of ten candidate points along the band that is inside the axes (with an inset) and at least min_d m from the ego (0, 0) and from the labels placed before; the first inside point if none is free"""
    out = []
    for poly, _ in bands:
        poly = np.asarray(poly, float)
        cand = [poly[min(int(f * (len(poly) - 1)), len(poly) - 1)] for f in np.linspace(0, 0.9, 10)]
        ins = [c for c in cand if xl[0] + inset <= c[0] <= xl[1] - inset and yl[0] + inset <= c[1] <= yl[1] - inset]
        free = [c for c in ins if np.hypot(*c) >= min_d and all(np.hypot(*(c - q)) >= min_d for q in out)]
        out.append((free or ins or [np.clip(cand[0], [xl[0] + inset, yl[0] + inset], [xl[1] - inset, yl[1] - inset])])[0])
    return out


def node_style(oc, k):
    """(colour, filled, marker, size) of a delivered node by the colour rule: tp = the colour of its piece (filled), fn = the same colour pale and hollow, fp = amber cross, tn = small grey"""
    if oc == "tp":
        return piece_color(k), True, "o", 30
    if oc == "fn":
        return piece_color(k), False, "o", 30
    if oc == "fp":
        return AMBER, True, "X", 42
    return GREY, True, "o", 6


def link_style(oa, ka, ob, kb):
    """(colour, linewidth, linestyle, alpha) of a link by the colour rule: both ends flagged on the route and of the SAME piece = that colour; both missed on the same piece = pale dashed; both false positives = amber; else thin grey"""
    if oa == ob == "tp" and ka == kb:
        return piece_color(ka), 2.6, "-", 1.0
    if oa == ob == "fn" and ka == kb:
        return piece_color(ka), 1.6, (0, (3, 2)), 0.5
    if oa == ob == "fp":
        return AMBER, 1.6, "-", 1.0
    return GREY, 0.8, "-", 0.8


# ------------------------------------------------------------------------------------------------------------------------------ the picture
def delivered_graph(f, valid, pos, dirv):
    """the delivered graph of a dump frame of the champion at its E375 decode: -> (kept (K,) bool, links = [(src, dst, curve (n, 2))] with Hermite curves as the scorer's read-out builds them)"""
    import jtoken_eval as JTE
    import qtargets as QT
    d = JTE.decode_variant(f, valid, LINK_THR, KEEP, False, LINK_LONG)
    pos, dirv = np.asarray(pos, float), np.asarray(dirv, float)
    r = d["rec"]
    links = [(int(s), int(t), QT.hermite(pos[s], dirv[s], pos[t], dirv[t], float(a), float(b), 0.5)) for s, t, a, b in zip(r["src"], r["dst"], r["a"], r["b"])]
    return np.asarray(d["keep"], bool), links


def draw_panel(ax, title, pos, lab, flag, kept, links, pk, bands, truth, gt_lines, cov, xl, yl, band_col=None):
    """one BEV panel: the GT lanes (grey, arrows), the route (bands = what the head saw, numbered, with arrows; truth = the exact pieces thin and dashed for a head without a map), the delivered links and nodes by the colour rule"""
    ax.set_xlim(*xl)
    ax.set_ylim(*yl)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    ax.set_title(title, fontsize=11, linespacing=1.2)
    ax.tick_params(labelsize=9)
    for q in gt_lines:
        ax.plot(q[:, 0], q[:, 1], color="#c9c9c9", lw=3.0, solid_capstyle="round", zorder=1)
        for m in range(10, len(q) - 1, 24):
            ax.annotate("", xy=q[m + 1], xytext=q[m], arrowprops=dict(arrowstyle="-|>", color="#777777", lw=0.8), zorder=2)
    for k, ((poly, _), (lx, ly)) in enumerate(zip(bands, label_positions(bands, xl, yl))):
        c = band_col or piece_color(k)                                                # a WRONG map is drawn in one dark colour: its pieces are not the pieces of the true route
        ax.plot(poly[:, 0], poly[:, 1], "-", color=c, lw=10, alpha=0.30, solid_capstyle="round", zorder=1.5)
        m = len(poly) // 2
        if len(poly) > m + 1:
            ax.annotate("", xy=poly[m + 1], xytext=poly[m], arrowprops=dict(arrowstyle="-|>", color=c, lw=2.0, mutation_scale=16), zorder=2.5)
        ax.text(lx, ly, str(k + 1), fontsize=9, color="white", weight="bold", ha="center", va="center", zorder=7, bbox=dict(boxstyle="circle,pad=0.25", fc=c, ec="white", lw=0.8))
    for k, (poly, _) in enumerate(truth):
        ax.plot(poly[:, 0], poly[:, 1], "--", color=piece_color(k), lw=1.2, alpha=0.9, zorder=1.6)
    oc = outcome(lab, flag)
    for s, t, cv in links:
        if not (kept[s] and kept[t]):
            continue
        col, lw, ls, al = link_style(oc[s], pk[s], oc[t], pk[t])
        ax.plot(cv[:, 0], cv[:, 1], color=col, lw=lw, ls=ls, alpha=al, zorder=3 if col == GREY else 4)
        if lw > 1.2 and len(cv) > 3:
            m = len(cv) // 2
            ax.annotate("", xy=cv[m + 1], xytext=cv[m], arrowprops=dict(arrowstyle="-|>", color=col, lw=0.8, alpha=al), zorder=4.5)
    for i in np.where(kept)[0]:
        col, filled, mk, sz = node_style(oc[i], pk[i])
        if oc[i] == "fp" and not cov[i]:                                              # flagged, but the token has no GT node: not scored, drawn small and pale
            ax.scatter(pos[i, 0], pos[i, 1], s=16, c=AMBER, marker="x", linewidths=0.9, alpha=0.55, zorder=5)
        elif filled:
            ax.scatter(pos[i, 0], pos[i, 1], s=sz, c=col, marker=mk, linewidths=0.5 if mk == "X" else 0, edgecolors="white" if mk == "X" else col, zorder=5)
        else:
            ax.scatter(pos[i, 0], pos[i, 1], s=sz, facecolors="white", edgecolors=col, linewidths=1.6, alpha=0.75, marker=mk, zorder=5)
    ax.plot([0], [0], marker="^", c="k", ms=8, ls="none", zorder=8)
    ax.set_xlabel("x (m)", fontsize=10)
    return {k: int((cov & (oc == k)).sum()) for k in OUT_KEYS}


def draw_frame(path, title, sub, cam, table_rows, panels, gt_lines, xl, yl, key):
    """one png: the camera (top left, or a note without one), the table (top right), the panels side by side (panels = [dict(title, flag, bands, truth)] drawn with `draw_panel`), the key line at the bottom"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(panels)
    fig = plt.figure(figsize=(11.0, 8.2), dpi=110)
    ncol = 3 if n == 3 else 4                                                          # two panels are centred and as large as three
    gs = fig.add_gridspec(2, ncol, height_ratios=[0.8, 1.0], width_ratios=[1, 1, 1] if n == 3 else [0.5, 1, 1, 0.5], hspace=0.25, wspace=0.12, left=0.06, right=0.99, top=0.885, bottom=0.16)
    top = gs[0, :]
    sub_gs = top.subgridspec(1, 2, width_ratios=[2.0, 1.0] if n > 1 else [1, 1], wspace=0.06)
    axc = fig.add_subplot(sub_gs[0, 0])
    axc.axis("off")
    if cam is not None:
        axc.imshow(cam)
    else:
        axc.text(0.5, 0.5, "kein Kamerabild verfügbar", ha="center", va="center", fontsize=11)
    axc.set_title("Frontkamera, GT-Spurgraph eingeblendet (nicht das Ergebnis; die Kamerafarben haben keine Bedeutung)", fontsize=9.5, pad=2)
    axt = fig.add_subplot(sub_gs[0, 1])
    axt.axis("off")
    tb = axt.table(cellText=table_rows, cellLoc="center", loc="center", bbox=[0.0, 0.02, 1.0, 0.96], colWidths=[0.5] + [0.5 / (len(table_rows[0]) - 1)] * (len(table_rows[0]) - 1))
    tb.auto_set_font_size(False)
    tb.set_fontsize(9)
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0 or c == 0:
            cell.set_text_props(weight="bold")
    counts = []
    for j, p in enumerate(panels):
        ax = fig.add_subplot(gs[1, j + (0 if n == 3 else 1)])
        counts.append(draw_panel(ax, p["title"], p["pos"], p["lab"], p["flag"], p["kept"], p["links"], p["pk"], p["bands"], p["truth"], gt_lines, p["cov"], xl, yl, p.get("band_col")))
        if j:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("y (m, nach vorn)", fontsize=10)
    fig.text(0.5, 0.985, title, fontsize=13, weight="bold", ha="center", va="top")
    fig.text(0.5, 0.945, sub, fontsize=9.5, ha="center", va="top", color="#333333")
    from matplotlib.lines import Line2D
    sw = [Line2D([0], [0], marker="o", ls="none", c=PALETTE[0], ms=8, label="Farbe des Routenstücks: auf der Route und markiert (Knoten und Verbindung)"),
          Line2D([0], [0], marker="X", ls="none", c=AMBER, ms=8, label="markiert, aber nicht auf der Route"),
          Line2D([0], [0], marker="o", ls="none", mfc="white", mec=PALETTE[0], mew=1.6, ms=8, label="auf der Route, aber nicht markiert (hell, hohl)"),
          Line2D([0], [0], c=GREY, lw=1.2, label="nicht markiert, nicht auf der Route"),
          Line2D([0], [0], marker="x", ls="none", c=AMBER, ms=6, alpha=0.55, label="markiert, ohne GT-Knoten (nicht gewertet)"),
          Line2D([0], [0], c=PALETTE[1], lw=8, alpha=0.30, label="Routenstück der Karte (Nummer, Pfeil = Fahrtrichtung)"),
          Line2D([0], [0], c=PALETTE[2], lw=1.2, ls="--", label="wahre Route (ohne Karte: vom Kopf nicht gesehen)")]
    if n == 3:
        sw.append(Line2D([0], [0], c=WRONG, lw=8, alpha=0.30, label="Route einer ANDEREN Szene (falsche Karte)"))
    fig.legend(handles=sw, loc="lower left", bbox_to_anchor=(0.05, 0.0), ncol=2, fontsize=8.5, frameon=False, columnspacing=1.5, handletextpad=0.5, labelspacing=0.35)
    fig.savefig(path)
    plt.close(fig)
    return counts


def selftest():
    rng = np.random.RandomState(0)
    F = 40
    gain = rng.randn(F) * 0.1
    drop = rng.rand(F)
    fork = rng.rand(F) > 0.5
    ok = np.ones(F, bool)
    ok[3] = False
    gain[3] = 9.0                                                                       # a non-candidate frame must never be chosen
    drop[3] = 9.0
    c = choose(gain, drop, fork, ok)
    assert c["gain"] == int(np.nanargmax(np.where(ok, gain, np.nan))) and c["gain"] != 3 and c["loss"] == int(np.nanargmin(np.where(ok, gain, np.nan)))
    assert c["wrong"] == int(np.nanargmax(np.where(ok, drop, np.nan))) and fork[c["typical"]] and ok[c["typical"]]
    med = np.median(gain[fork & ok])
    assert abs(gain[c["typical"]] - med) <= min(abs(gain[j] - med) for j in np.where(fork & ok)[0]) + 1e-12
    lab = np.array([1, 1, 0, 0, -1, -1])
    flag = np.array([True, False, True, False, True, False])
    assert list(outcome(lab, flag)) == ["tp", "fn", "fp", "tn", "fp", "tn"]
    f, tp, fp, fn = frame_f1(np.array([1, 1, 1, 1, 0, 0], bool), lab, flag)
    assert (tp, fp, fn) == (1, 1, 1) and abs(f - 0.5) < 1e-12
    for h in PALETTE + [AMBER, GREY]:
        r, g, b = (int(h[i:i + 2], 16) for i in (1, 3, 5))
        assert not (r > 150 and g < 110 and b < 110), f"a red colour {h}"
    assert len(set(PALETTE)) == 8 and AMBER not in PALETTE and WRONG not in PALETTE and piece_color(8) == piece_color(0) and piece_color(6) != piece_color(0)
    # THE COLOUR RULE on a hand-made frame with two route pieces: piece 1 along y at x = 0 (y 0 .. 40), piece 2 along x at y = 20 (x 0 .. 30)
    y = np.arange(0.0, 40.0, 1.0)
    x = np.arange(0.0, 30.0, 1.0)
    polys = [(np.c_[np.zeros_like(y), y], np.ones(len(y))), (np.c_[x, np.full_like(x, 20.0)], np.ones(len(x)))]
    pos = np.array([[0.4, 10.0], [20.0, 20.5], [20.0, 25.0], [25.0, 19.7], [10.0, 5.0], [0.3, 30.0]])
    lab = np.array([1, 1, 0, 1, 0, 1])
    flag = np.array([True, True, True, False, False, False])
    pk, oc = piece_of(pos, polys), outcome(lab, flag)
    assert list(pk[[0, 1, 3]]) == [0, 1, 1], pk                                        # the nodes near piece 1, piece 2, piece 2
    assert list(oc) == ["tp", "tp", "fp", "fn", "tn", "fn"], oc
    assert node_style(oc[0], pk[0])[:2] == (PALETTE[0], True)                           # the flagged node of piece 1 has the colour of piece 1
    assert node_style(oc[1], pk[1])[:2] == (PALETTE[1], True)                           # the flagged node of piece 2 has the colour of piece 2: a different one
    assert node_style(oc[2], pk[2])[0] == AMBER and node_style(oc[2], pk[2])[2] == "X"  # a flagged off-route node: amber cross
    assert node_style(oc[3], pk[3])[:2] == (PALETTE[1], False) and node_style(oc[5], pk[5])[:2] == (PALETTE[0], False)   # a missed node: its piece colour, hollow
    assert node_style(oc[4], pk[4])[0] == GREY and node_style(oc[4], pk[4])[3] < 10     # not flagged, not on the route: small grey
    assert link_style("tp", 1, "tp", 1)[0] == PALETTE[1] and link_style("tp", 0, "tp", 1)[0] == GREY and link_style("tp", 1, "fp", -1)[0] == GREY      # a link: one colour only when both ends are on the same piece
    assert link_style("fn", 0, "fn", 0)[2] != "-" and link_style("fn", 0, "fn", 0)[3] < 1 and link_style("fp", -1, "fp", -1)[0] == AMBER and link_style("tn", -1, "tn", -1)[0] == GREY
    assert piece_of(np.zeros((1, 2)), [])[0] == -1
    two = [(np.c_[np.zeros(20), np.arange(20.0)], None), (np.c_[np.zeros(20), np.arange(20.0)], None), (np.c_[np.full(5, 99.0), np.arange(5.0)], None)]      # two bands that start at the same point, one outside the box
    lp = label_positions(two, (-25.0, 25.0), (-5.0, 45.0))
    assert np.hypot(*(lp[0] - lp[1])) >= 6.0 and all(np.hypot(*q) >= 6.0 for q in lp[:2]) and all(-22.5 <= q[0] <= 22.5 and -2.5 <= q[1] <= 42.5 for q in lp), lp
    xl, yl = limits(np.array([[0.0, 0.0], [10.0, 20.0]]))
    assert xl == (-20.0, 30.0) and yl == (-15.0, 35.0), (xl, yl)                         # a square box of +-25 m round the centre (5, 10)
    # the pieces of the error route: sampled with the loader's `sample_route` they are exactly the route table the head saw (`error_route`), piece by piece
    import lane_route_data as LRD
    pl = [(np.c_[np.zeros_like(y), y], np.full(len(y), 2.0)), (np.c_[x * 0.5, 40.0 + 0.0 * x], np.full(len(x), 2.0)), (np.c_[np.full(15, 10.0), np.arange(15.0)], np.full(15, 3.0))]
    for tok in ("tokA", "tokB"):
        got, want = LRD.sample_route(error_pieces(pl, tok)), LRD.error_route(pl, tok)
        assert all(np.array_equal(a, b) for a, b in zip(got, want)), f"the pieces of the error route are not the route the head saw ({tok})"
    print("route_frames_fig selftest ok")


def load_gt_lines(token):
    import qgraph_view as QV
    g = QV.load_gt(token)
    return g, [np.asarray(p["pts"], float) for p in g["pieces"] if len(p["pts"]) > 1]


def camera_of(token, g):
    import qgraph_view as QV
    try:
        tiles = QV.camera_tiles(token, g, {c: (800, 450) for c in QV.CAMS}, log=lambda *a: None)
        img = tiles["CAM_FRONT"]
        h = img.shape[0]
        return img[int(h * 0.30):int(h * 0.93)]                                     # the road part of the front image, as the old figures
    except Exception:                                                               # noqa: BLE001
        return None


def main(a):
    import jroute_eval as JE
    seed = a.seed
    side = JE.load_side(seed, mode="wrong")
    pn, _ = JE.champion_pnone(side, seed)
    dump = pickle.load(open(os.path.join(JE.RUNS, f"e373_cb_s{seed}", "token_val.pkl"), "rb"))["frames"]
    assert [f["token"] for f in dump[:len(side["tokens"])]] == side["tokens"]
    dlv = JE.delivered_kept(side["valid"], pn, JE.KEEP_THR)

    def pflags(run, mode):
        d = pickle.load(open(os.path.join(JE.RUNS, run, f"route_{mode}.pkl"), "rb"))
        assert [f["token"] for f in d["frames"]] == side["tokens"], run
        return np.stack([np.asarray(f["p_road"]) >= JE.THR for f in d["frames"]])

    fl = dict(z=pflags(f"e389_z_s{seed}", "error"), m=pflags(f"e389_m_s{seed}", "error"), mw=pflags(f"e389_m_s{seed}", "wrong"))
    cov = dlv & side["valid"] & (side["gt_id"] >= 0) & side["has_route"][:, None]
    F = len(side["tokens"])
    f1 = {k: np.array([frame_f1(cov[i], side["lab_road"][i], fl[k][i])[0] for i in range(F)]) for k in fl}
    ok = side["has_route"] & (cov.sum(1) >= MIN_NODES)
    pick = choose(f1["m"] - f1["z"], f1["m"] - f1["mw"], side["decision"] & side["has_route"], ok)
    os.makedirs(a.out, exist_ok=True)
    info = dict(seed=seed, min_nodes=MIN_NODES, candidates=int(ok.sum()), cases={})
    titles = {"typical": "Typischer Frame mit Abzweig (mittlerer Gewinn der Karte)", "gain": "Größter Gewinn der Karte", "loss": "Größter Verlust der Karte", "wrong": "Größter Absturz mit falscher Karte"}
    key = None
    for case, i in pick.items():
        ds, tok = side["ds"], side["tokens"][i]
        exact = ds.polys(i)
        seen = error_pieces(exact, tok)
        got = [np.asarray(v) for v in __import__("lane_route_data").sample_route(seen)[:2]]
        want = ds.raw_route(i, "error")                                             # the route before the cap of 160 samples
        assert np.array_equal(got[0], np.asarray(want[0])), f"frame {i}: the drawn pieces are not the route the head saw"
        wrong = error_pieces(ds.polys(int(ds.plan[i])), side["tokens"][int(ds.plan[i])])
        pos, dirv, lab, valid = (np.asarray(side[k][i]) for k in ("pos", "dirv", "lab_road", "valid"))
        kept, links = delivered_graph(dump[i], valid, pos, dirv)
        pk = piece_of(pos, exact)
        g, gt_lines = load_gt_lines(tok)
        pts = np.concatenate([pos[cov[i]]] + [p for p, _ in exact]) if cov[i].any() else pos[kept]
        xl, yl = limits(pts)
        base = dict(pos=pos, lab=lab, kept=kept, links=links, pk=pk, cov=cov[i])
        panels = [dict(base, title=f"Kopf ohne Karte (z)\nF1 Fahrbahn {f1['z'][i]:.3f}", flag=fl["z"][i], bands=[], truth=exact),
                  dict(base, title=f"Kopf mit Karte (m)\nF1 Fahrbahn {f1['m'][i]:.3f}", flag=fl["m"][i], bands=seen, truth=[])]
        if case == "wrong":
            panels.append(dict(base, title=f"Kopf mit FALSCHER Karte (m)\nF1 Fahrbahn {f1['mw'][i]:.3f}", flag=fl["mw"][i], bands=wrong, truth=exact, band_col=WRONG))
        oc = {n: outcome(lab, p["flag"]) for n, p in zip(("z", "m", "mw"), panels)}
        cnt = {n: {k: int((cov[i] & (oc[n] == k)).sum()) for k in OUT_KEYS} for n in oc}
        names = dict(z="ohne\nKarte", m="mit\nKarte", mw="falsche\nKarte")
        rows = [["", *[names[n] for n in oc]], ["F1 Fahrbahn", *[f"{f1[n][i]:.3f}" for n in oc]], ["richtig markiert", *[str(cnt[n]["tp"]) for n in oc]], ["übersehen", *[str(cnt[n]["fn"]) for n in oc]],
                ["fälschlich markiert", *[str(cnt[n]["fp"]) for n in oc]], ["erfasste Knoten", *[str(int(cov[i].sum())) for _ in oc]]]
        sc = f"Szene {int(side['scene'][i])}, Frame {i}, {len(exact)} Routenstücke der wahren Route"
        fn = f"e389_frame_{case}.png"
        draw_frame(os.path.join(a.out, fn), titles[case], sc + " (VAL, Seed 0)", camera_of(tok, g), rows, panels, gt_lines, xl, yl, key)
        info["cases"][case] = dict(frame=i, scene=int(side["scene"][i]), nodes=int(cov[i].sum()), fork=bool(side["decision"][i]), f1_z=float(f1["z"][i]), f1_m=float(f1["m"][i]), f1_m_wrong=float(f1["mw"][i]),
                                   gain=float(f1["m"][i] - f1["z"][i]), drop=float(f1["m"][i] - f1["mw"][i]), pieces=len(exact), file=fn)
    json.dump(info, open(os.path.join(a.out, "e389_frames.json"), "w"), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        assert a.out and os.path.isabs(a.out), "--out <absolute dir>"
        main(a)
