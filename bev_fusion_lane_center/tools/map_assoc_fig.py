"""E390 / E391 report: example frames of the SD-MAP ASSOCIATION in the style of the E389 figures (`route_frames_fig.py`), chosen by a WRITTEN RULE, drawn from the stored models (CPU, no training, no GPU).

    bash tools/env/_py14.sh map_assoc_fig.py --seed 0 --out <absolute dir>    # _py14.sh changes the directory: give an absolute --out; needs torch (the gcar python) and WSL for the camera
    bash tools/env/_py14.sh map_assoc_fig.py --selftest                       # the colour rule, the frame rule, the node choice, no data files

Data: `mapassoc_probe.AssocData("val", seed, level="dec")` (the deciding map level: pose 1 m, count errors 0.1, link drop 0.1, route hide 0.3, segments missing 0.1; one generator per frame seeded (391, frame)) gives per
delivered token its target (`assoc` = the true road-graph segment, -1 unassigned, -2 ignored) and its `kind` (0 the segment is in the map, 1 its segment is MISSING from the map, 2 a false token with no GT node). Two heads
read the same frames: the ZERO-SHOT E390 association head (`sdgraph_probe.load_models("g7", seed)` -> `heads6.assoc`, `mapassoc_probe.p6_assign`) and E391's fusion head F1 (`mapassoc_probe.load_head("F1", seed)`), exactly
as `mapassoc_probe.null_and_model_rows` calls them. The links are the delivered graph of the champion (E373 dump, E375 decode; `route_frames_fig.delivered_graph`).

Three BEV panels with shared axes: the TRUTH (every token by its true segment), the zero-shot head, F1. THE MAP is drawn as its SEGMENTS: every present node with a delivered labelled token of the frame on it or within
`NEAR_M` m of it, its polylines as a thick translucent band in its OWN colour (8 colours, in the order of the distance of the node to the ego, repeated after 8), its number at a free point (`place_labels`: off the ego, the other numbers and the tokens), direction arrows; an
intersection is one colour for all its connectors. COLOUR RULE (the owner's): a delivered token the head assigns to its CORRECT segment = filled, in the colour of that segment, and a link whose two ends are such tokens of
the SAME segment has that colour too; assigned to a WRONG segment, or a false token (no GT node) that got a segment = amber cross; its segment is in the map but the token is left unassigned = the colour of its true segment,
hollow and pale (dashed link); its segment is MISSING from the map and it is left unassigned (correct) = small filled grey-blue dot; a false token left unassigned (correct) = tiny grey dot; other links thin grey. In the
truth panel: a token of a present segment is filled in the colour of its segment, a token of a missing segment hollow dark grey, a false token tiny grey. No red. Tokens without a label (assoc -2) are not drawn and not counted.

The frames are chosen by the rule below (fixed before any picture was looked at; they are NOT re-picked): candidates = VAL frames with at least MIN_TOK delivered labelled tokens (kept and assoc >= -1); per frame
the association accuracy of F1 and of the zero-shot head (E391's assoc_acc: the share of those tokens assigned to their segment or, when the segment is missing or the token false, left unassigned); typical = the candidate
whose F1 accuracy is closest to the median of the candidates', gain = the largest F1 minus zero-shot, loss = the smallest (F1 minus zero-shot). Writes `e391_frame_<case>.png` and `e391_frames.json`.
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import route_frames_fig as RF  # noqa: E402  (also puts evaluation/, training/grid, training/data, tools/viz on sys.path)
from route_frames_fig import AMBER, GREY, PALETTE, camera_of, delivered_graph, limits, load_gt_lines, piece_color  # noqa: E402

MIN_TOK = 30                                                                         # delivered labelled tokens a candidate frame needs
NEAR_M = 8.0                                                                         # a node is drawn when a delivered labelled token is within this distance of its geometry
BLUEGREY, DARK = "#5E7FA3", "#444444"                                                # a missing-segment token left unassigned (correct); a missing-segment token in the truth panel
CLASSES = ("ok", "wrong", "miss", "none", "false")


# ------------------------------------------------------------------------------------------------------------------------------ the rules
def classify(pred, tgt, kind):
    """the outcome of each token (arrays): 'ok' assigned to its own segment, 'wrong' assigned to another segment (a token of a missing segment or a false token that got a segment too), 'miss' its segment is in the map
    but it is left unassigned, 'none' its segment is missing and it is left unassigned (correct), 'false' a false token left unassigned (correct)"""
    pred, tgt, kind = np.asarray(pred), np.asarray(tgt), np.asarray(kind)
    return np.where(pred >= 0, np.where(pred == tgt, "ok", "wrong"), np.where(kind == 0, "miss", np.where(kind == 1, "none", "false")))


def truth_classes(kind):
    """the truth panel's classes: 'ok' a token of a segment in the map (drawn in its colour), 'absent' its segment is missing from the map, 'false' a false token"""
    return np.where(np.asarray(kind) == 0, "ok", np.where(np.asarray(kind) == 1, "absent", "false"))


def token_style(cls, seg, colours):
    """-> dict(color, filled, marker, size, alpha) of a token by the colour rule (colours: segment -> colour)"""
    if cls == "ok":
        return dict(color=colours[seg], filled=True, marker="o", size=30, alpha=1.0)
    if cls == "miss":
        return dict(color=colours[seg], filled=False, marker="o", size=30, alpha=0.75)
    if cls == "wrong":
        return dict(color=AMBER, filled=True, marker="X", size=42, alpha=1.0)
    if cls == "none":
        return dict(color=BLUEGREY, filled=True, marker="o", size=9, alpha=1.0)
    if cls == "absent":
        return dict(color=DARK, filled=False, marker="o", size=22, alpha=0.9)
    return dict(color=GREY, filled=True, marker="o", size=5, alpha=1.0)


def link_style(ca, sa, cb, sb, colours):
    """(colour, linewidth, linestyle, alpha) of a link by the colour rule: both ends 'ok' of the SAME segment = that colour; both 'miss' of the same segment = pale dashed; both 'wrong' = amber; else thin grey"""
    if ca == cb == "ok" and sa == sb:
        return colours[sa], 2.6, "-", 1.0
    if ca == cb == "miss" and sa == sb:
        return colours[sa], 1.6, (0, (3, 2)), 0.5
    if ca == cb == "wrong":
        return AMBER, 1.6, "-", 1.0
    return GREY, 0.8, "-", 0.8


def frame_stats(pred, tgt, kind, kept):
    """the numbers of one frame over its delivered LABELLED tokens (kept and tgt >= -1): n, acc (E391's assoc_acc), and the counts with their denominators: wrong_present (a token of a present segment assigned to another one,
    of n0), unassigned_present (left unassigned, of n0), wrong_absent (a token of a missing segment that got a segment, of n1), false_assigned (of n2); class_counts over the five classes"""
    pred, tgt, kind = np.asarray(pred), np.asarray(tgt), np.asarray(kind)
    m = np.asarray(kept, bool) & (tgt >= -1)
    p, t, k = pred[m], tgt[m], kind[m]
    cl = classify(p, t, k)
    n = int(m.sum())
    return dict(n=n, acc=float((p == t).mean()) if n else float("nan"), n0=int((k == 0).sum()), n1=int((k == 1).sum()), n2=int((k == 2).sum()),
                wrong_present=int(((k == 0) & (cl == "wrong")).sum()), unassigned_present=int((cl == "miss").sum()), wrong_absent=int(((k == 1) & (cl == "wrong")).sum()),
                false_assigned=int(((k == 2) & (cl == "wrong")).sum()), class_counts={c: int((cl == c).sum()) for c in CLASSES})


def choose(acc_f1, acc_z, n_tok):
    """the written rule -> {case: frame index}; acc_f1, acc_z (F,) the association accuracies of F1 and of the zero-shot head, n_tok (F,) the delivered labelled tokens; candidates = n_tok >= MIN_TOK"""
    ok = np.asarray(n_tok) >= MIN_TOK
    f1, z = np.where(ok, acc_f1, np.nan), np.where(ok, acc_z, np.nan)
    gain = f1 - z
    med = float(np.nanmedian(f1))
    return dict(typical=int(np.nanargmin(np.abs(f1 - med))), gain=int(np.nanargmax(gain)), loss=int(np.nanargmin(gain))), int(ok.sum())


def dense(poly, step=1.0):
    """a polyline resampled every `step` m (its first and last point kept)"""
    poly = np.asarray(poly, float)
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(poly, axis=0), axis=1))]
    if s[-1] < 1e-9:
        return poly[:1]
    t = np.unique(np.r_[np.arange(0.0, s[-1], step), s[-1]])
    return np.c_[np.interp(t, s, poly[:, 0]), np.interp(t, s, poly[:, 1])]


def select_nodes(geo, present, pos, true_seg, near=NEAR_M):
    """the nodes to draw: present, with a polyline of >= 2 points, and (a token's true segment, or a token within `near` m of its geometry); geo = [list of polylines per node], pos (K, 2) and true_seg (K,) the delivered labelled
    tokens -> the node indices in the order of the distance of the node to the ego (0, 0)"""
    from scipy.spatial import cKDTree
    tree = cKDTree(np.asarray(pos, float).reshape(-1, 2)) if len(pos) else None
    holds = set(int(s) for s in np.asarray(true_seg) if s >= 0)
    out = []
    for n, gs in enumerate(geo):
        gs = [dense(p) for p in gs if len(p) >= 2]
        if not present[n] or not gs:
            continue
        pts = np.concatenate(gs)
        if n in holds or (tree is not None and tree.query(pts)[0].min() <= near):
            out.append((float(np.linalg.norm(pts, axis=1).min()), n))
    return [n for _, n in sorted(out)]


def arrow_index(poly, xl, yl, min_d=7.0, inset=2.0):
    """the vertex of a polyline where its direction arrow starts: the first of five points along it (middle first) that is inside the axes and at least min_d m from the ego (the ego triangle must not be mistaken for an
    arrow); None for a polyline shorter than 6 m or without such a point"""
    poly = np.asarray(poly, float)
    if len(poly) < 4 or np.linalg.norm(poly[-1] - poly[0]) < 6.0:
        return None
    for f in (0.5, 0.35, 0.65, 0.2, 0.8):
        m = min(int(f * (len(poly) - 1)), len(poly) - 2)
        x, y = poly[m]
        if xl[0] + inset <= x <= xl[1] - inset and yl[0] + inset <= y <= yl[1] - inset and np.hypot(x, y) >= min_d:
            return m
    return None


def place_labels(bands, xl, yl, tok_pos, min_d=9.5, tok_d=3.0, inset=2.5):
    """one (x, y) per band for its number (`route_frames_fig.label_positions` with every metre of the band as a candidate and the tokens kept free): the first point from the band's start that is inside the axes (inset),
    at least min_d m from the ego and from the numbers placed before and tok_d m from every token; else, of the points that keep min_d, the one farthest from the tokens; else the inside point farthest from the numbers
    placed; bands = [polyline (n, 2)]"""
    tok = np.asarray(tok_pos, float).reshape(-1, 2)
    out = []
    for poly in bands:
        cand = dense(poly, 1.0)
        ins = [c for c in cand if xl[0] + inset <= c[0] <= xl[1] - inset and yl[0] + inset <= c[1] <= yl[1] - inset]
        far = [c for c in ins if np.hypot(*c) >= min_d and all(np.hypot(*(c - q)) >= min_d for q in out)]
        free = [c for c in far if not len(tok) or np.linalg.norm(tok - c, axis=1).min() >= tok_d]
        if free:
            out.append(free[0])
        elif far:                                                                       # no point keeps clear of the tokens: the one farthest from them
            out.append(max(far, key=lambda c: np.linalg.norm(tok - c, axis=1).min() if len(tok) else 0.0))
        elif ins:
            out.append(max(ins, key=lambda c: min([np.hypot(*(c - q)) for q in out] + [np.hypot(*c)])))
        else:
            out.append(np.clip(cand[0], [xl[0] + inset, yl[0] + inset], [xl[1] - inset, yl[1] - inset]))
    return out


def segment_colours(nodes):
    """node index -> colour: the palette in the order given (the distance to the ego), repeated after 8"""
    return {n: piece_color(k) for k, n in enumerate(nodes)}


# ------------------------------------------------------------------------------------------------------------------------------ the picture
def draw_panel(ax, title, pos, cls, seg, drawn, links, colours, bands, labels, gt_lines, xl, yl):
    """one BEV panel: the GT lanes (light grey, arrows), the map's segments (bands with number and arrows), the delivered links and the tokens by the colour rule; cls / seg (K,) = the class and the colour segment of
    each token, drawn (K,) bool = the delivered labelled tokens"""
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
    for (n, polys), (lx, ly), k in zip(bands, labels, range(len(bands))):
        c = colours[n]
        for poly in polys:
            ax.plot(poly[:, 0], poly[:, 1], "-", color=c, lw=10, alpha=0.30, solid_capstyle="round", zorder=1.5)
            m = arrow_index(poly, xl, yl)
            if m is not None:
                ax.annotate("", xy=poly[m + 1], xytext=poly[m], arrowprops=dict(arrowstyle="-|>", color=c, lw=2.0, mutation_scale=13), zorder=2.5)
        ax.text(lx, ly, str(k + 1), fontsize=8, color="white", weight="bold", ha="center", va="center", zorder=7, bbox=dict(boxstyle="circle,pad=0.2", fc=c, ec="white", lw=0.8))
    for s, t, cv in links:
        if not (drawn[s] and drawn[t]):
            continue
        col, lw, ls, al = link_style(cls[s], seg[s], cls[t], seg[t], colours)
        ax.plot(cv[:, 0], cv[:, 1], color=col, lw=lw, ls=ls, alpha=al, zorder=3 if col == GREY else 4)
        if lw > 1.2 and len(cv) > 3:
            m = len(cv) // 2
            ax.annotate("", xy=cv[m + 1], xytext=cv[m], arrowprops=dict(arrowstyle="-|>", color=col, lw=0.8, alpha=al), zorder=4.5)
    for i in np.where(drawn)[0]:
        st = token_style(cls[i], seg[i], colours)
        if st["filled"]:
            ax.scatter(pos[i, 0], pos[i, 1], s=st["size"], c=st["color"], marker=st["marker"], linewidths=0.5 if st["marker"] == "X" else 0, edgecolors="white" if st["marker"] == "X" else st["color"], zorder=5)
        else:
            ax.scatter(pos[i, 0], pos[i, 1], s=st["size"], facecolors="white", edgecolors=st["color"], linewidths=1.6, alpha=st["alpha"], marker=st["marker"], zorder=5)
    ax.plot([0], [0], marker="^", c="k", ms=8, ls="none", zorder=8)
    ax.set_xlabel("x (m)", fontsize=10)


def draw_frame(path, title, sub, cam, table_rows, panels, gt_lines, xl, yl, bands, colours):
    """one png: the camera (top left), the table (top right), the three panels side by side (`draw_panel`), the swatch legend at the bottom"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig = plt.figure(figsize=(11.0, 8.0), dpi=110)
    gs = fig.add_gridspec(2, 3, height_ratios=[0.8, 1.0], hspace=0.28, wspace=0.12, left=0.06, right=0.99, top=0.86, bottom=0.2)
    sub_gs = gs[0, :].subgridspec(1, 2, width_ratios=[1.75, 1.25], wspace=0.05)
    axc = fig.add_subplot(sub_gs[0, 0])
    axc.axis("off")
    if cam is not None:
        axc.imshow(cam)
    else:
        axc.text(0.5, 0.5, "kein Kamerabild verfügbar", ha="center", va="center", fontsize=11)
    axc.set_title("Frontkamera mit GT-Spurgraph (nicht das Ergebnis; Kamerafarben ohne Bedeutung)", fontsize=8.5, pad=2)
    axt = fig.add_subplot(sub_gs[0, 1])
    axt.axis("off")
    nc = len(table_rows[0])
    tb = axt.table(cellText=table_rows, cellLoc="center", loc="center", bbox=[0.0, 0.0, 1.0, 1.0], colWidths=[0.58] + [0.42 / (nc - 1)] * (nc - 1))
    tb.auto_set_font_size(False)
    tb.set_fontsize(8.5)
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0 or c == 0:
            cell.set_text_props(weight="bold")
    labels = place_labels([max(polys, key=len) for _, polys in bands], xl, yl, panels[0]["pos"][panels[0]["drawn"]])
    for j, p in enumerate(panels):
        ax = fig.add_subplot(gs[1, j])
        draw_panel(ax, p["title"], p["pos"], p["cls"], p["seg"], p["drawn"], p["links"], colours, bands, labels, gt_lines, xl, yl)
        if j:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("y (m, nach vorn)", fontsize=10)
    fig.text(0.5, 0.985, title, fontsize=13, weight="bold", ha="center", va="top")
    fig.text(0.5, 0.944, sub, fontsize=9.5, ha="center", va="top", color="#333333", linespacing=1.4)
    sw = [Line2D([0], [0], marker="o", ls="none", c=PALETTE[0], ms=8, label="richtig zugeordnet: Farbe des Segments (Verbindung im selben Segment ebenso)"),
          Line2D([0], [0], marker="X", ls="none", c=AMBER, ms=8, label="falschem Segment zugeordnet (auch ein falsches Token mit Segment)"),
          Line2D([0], [0], marker="o", ls="none", mfc="white", mec=PALETTE[0], mew=1.6, ms=8, alpha=0.75, label="Segment da, Token nicht zugeordnet (Farbe des wahren Segments, blass, hohl)"),
          Line2D([0], [0], marker="o", ls="none", c=BLUEGREY, ms=5, label="Segment fehlt in der Karte, richtig nicht zugeordnet"),
          Line2D([0], [0], marker="o", ls="none", c=GREY, ms=3, label="falsches Token, richtig nicht zugeordnet"),
          Line2D([0], [0], marker="o", ls="none", mfc="white", mec=DARK, mew=1.6, ms=7, label="nur Wahrheit: Segment des Tokens fehlt in der Karte"),
          Line2D([0], [0], c=PALETTE[1], lw=8, alpha=0.30, label="Segment der Karte (Nummer nach Abstand zum Ego, Pfeil = Richtung)"),
          Line2D([0], [0], c=GREY, lw=1.2, label="übrige Verbindung des gelieferten Graphen")]
    fig.legend(handles=sw, loc="lower left", bbox_to_anchor=(0.02, 0.0), ncol=2, fontsize=7.8, frameon=False, columnspacing=1.2, handletextpad=0.5, labelspacing=0.4)
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------------------------------------ the heads and the frames
def score_all(seed, level="dec"):
    """both heads on every VAL frame of the level: -> (data, {frame: dict(z=pred, f1=pred)}) with the predictions per token (-1 = unassigned), computed as `mapassoc_probe.null_and_model_rows` does"""
    import torch
    import mapassoc_probe as MP
    import sdgraph_probe as SP
    torch.set_num_threads(2)
    data = MP.AssocData("val", seed, level=level)
    dl = torch.utils.data.DataLoader(data, batch_size=8, shuffle=False, num_workers=0, collate_fn=MP.collate)
    enc, heads6 = SP.load_models("g7", seed)
    enc, heads6 = enc.eval(), heads6.eval()
    head = MP.load_head("F1", seed)
    preds = {}
    with torch.no_grad():
        for b in dl:
            h, sv = MP.encode(enc, b, "cpu")
            z6 = heads6.assoc(h, b["pa"].float(), sv)
            o = head(b, h, sv, b["pa"])
            N = h.shape[1]
            for r in range(len(b["frame"])):
                pf = o["assoc"][r].argmax(-1).numpy()
                pf = np.where(pf == o["assoc"].shape[-1] - 1, -1, pf)
                preds[int(b["frame"][r])] = dict(z=MP.p6_assign(z6, r, N), f1=pf)
    return data, preds


def perturbed_map(data, i, level="dec"):
    """the input map of frame i exactly as AssocData draws it -> Gin (geo, present, kind)"""
    import mapassoc_probe as MP
    import lane_sdgraph_data as SG
    Gin, _ = SG.perturb(data.sg["graphs"][i], np.random.default_rng([391, i]), **MP.LEVELS[level])
    return Gin


def table_of(st_z, st_f):
    def frac(k, d):
        return [f"{s[k]} von {s[d]}" for s in (st_z, st_f)]
    return [["", "Null-Schuss\nE390", "E391 F1"], ["Zuordnungsgenauigkeit", *[f"{s['acc']:.3f}" for s in (st_z, st_f)]],
            ["falsches Segment\n(Segment in der Karte)", *frac("wrong_present", "n0")], ["Segment in der Karte,\nnicht zugeordnet", *frac("unassigned_present", "n0")],
            ["Segment fehlt,\ntrotzdem zugeordnet", *frac("wrong_absent", "n1")], ["falsche Token\nzugeordnet", *frac("false_assigned", "n2")],
            ["gelieferte Token\nmit Label", *[str(s["n"]) for s in (st_z, st_f)]]]


def main(a):
    seed = a.seed
    import mapassoc_probe as MP
    import lane_mapassoc_data as MA
    data, preds = score_all(seed)
    F = len(data)
    scene = MA.load("val", seed)["scene"]
    dump = pickle.load(open(os.path.join(RF.ROOT, "runs", f"e373_cb_s{seed}", "token_val.pkl"), "rb"))["frames"]
    assert [f["token"] for f in dump[:F]] == list(data.ds.tokens), "the dump frames are not the frames of the association data"
    items = [data[j] for j in range(F)]
    stats = {k: [frame_stats(preds[i][k], items[i]["assoc"], items[i]["kind"], items[i]["kept"]) for i in range(F)] for k in ("z", "f1")}
    pick, n_cand = choose(np.array([s["acc"] for s in stats["f1"]]), np.array([s["acc"] for s in stats["z"]]), np.array([s["n"] for s in stats["f1"]]))
    os.makedirs(a.out, exist_ok=True)
    info = dict(seed=seed, level="dec", level_args=MP.LEVELS["dec"], min_tok=MIN_TOK, frames=F, candidates=n_cand, cases={})
    titles = {"typical": "Typischer Frame (mittlere Zuordnungsgenauigkeit von F1)", "gain": "Größter Gewinn von F1 gegenüber dem Null-Schuss-Kopf", "loss": "Größter Verlust von F1 gegenüber dem Null-Schuss-Kopf"}
    for case, i in pick.items():
        it, tok = items[i], data.ds.tokens[i]
        pos, dirv, valid = np.asarray(it["pos"], float), np.asarray(it["dirv"], float), np.asarray(it["valid"])
        tgt, kind, kept = it["assoc"], it["kind"], it["kept"]
        drawn = kept & (tgt >= -1)
        dkept, links = delivered_graph(dump[i], valid, pos, dirv)
        Gin = perturbed_map(data, i)
        nodes = select_nodes(Gin["geo"], Gin["present"], pos[drawn], tgt[drawn])
        colours = segment_colours(nodes)
        bands = [(n, [np.asarray(p, float) for p in Gin["geo"][n] if len(p) >= 2]) for n in nodes]
        seg_t = np.where(tgt >= 0, tgt, 0)
        cl_t = truth_classes(kind)
        panels = [dict(title="Wahrheit\nToken nach ihrem wahren Segment", cls=cl_t, seg=seg_t),
                  dict(title=f"Null-Schuss-Kopf von E390\nZuordnung {stats['z'][i]['acc']:.3f}", cls=classify(preds[i]["z"], tgt, kind), seg=seg_t),
                  dict(title=f"E391 F1 (Fusionskopf)\nZuordnung {stats['f1'][i]['acc']:.3f}", cls=classify(preds[i]["f1"], tgt, kind), seg=seg_t)]
        for p in panels:
            p.update(pos=pos, drawn=drawn, links=links)
        g, gt_lines = load_gt_lines(tok)
        xl, yl = limits(pos[drawn] if drawn.any() else pos[kept])
        fn = f"e391_frame_{case}.png"
        sub = (f"Szene {int(scene[i])}, Frame {i} (VAL, Seed 0), {len(nodes)} Segmente der Karte gezeigt, {int(drawn.sum())} gelieferte Token mit Label\n"
               "Karte mit Fehlern (Stufe dec): Lage 1 m, Spuranzahl 10 %, Verbindungen 10 %, Route 30 % verborgen, Segmente 10 % fehlen")
        draw_frame(os.path.join(a.out, fn), titles[case], sub, camera_of(tok, g), table_of(stats["z"][i], stats["f1"][i]), panels, gt_lines, xl, yl, bands, colours)
        info["cases"][case] = dict(frame=i, scene=int(scene[i]), token=tok, nodes_drawn=len(nodes), file=fn, kept_agree=bool((np.asarray(dkept) == kept)[valid].all()),
                                   acc_z=stats["z"][i]["acc"], acc_f1=stats["f1"][i]["acc"], gain=stats["f1"][i]["acc"] - stats["z"][i]["acc"], z=stats["z"][i], f1=stats["f1"][i])
    json.dump(info, open(os.path.join(a.out, "e391_frames.json"), "w"), indent=1)
    print(json.dumps(info, indent=1))


# ------------------------------------------------------------------------------------------------------------------------------ the selftest
def selftest():
    # THE COLOUR RULE on a hand-made frame: three segments (0 at 5 m, 1 at 12 m, 2 at 20 m from the ego, along x at y = 5, 12, 20), a segment 3 that is missing from the map and a far one (4)
    xs = np.arange(0.0, 30.0, 1.0)
    geo = [[np.c_[xs, np.full_like(xs, y)]] for y in (5.0, 12.0, 20.0, 15.0, 200.0)]
    present = np.array([True, True, True, False, True])
    pos = np.array([[10.0, 20.2], [10.0, 12.0], [10.0, 5.3], [3.0, 90.0], [12.0, 14.0], [18.0, 60.0]])
    tgt = np.array([2, 1, 0, -1, -1, -1])
    kind = np.array([0, 0, 0, 1, 2, 2])
    pred = np.array([2, 0, -1, -1, 1, -1])
    nodes = select_nodes(geo, present, pos, tgt)
    assert nodes == [0, 1, 2], nodes                                                   # in the order of the distance to the ego; the missing segment (3) and the far one (4) are not drawn
    col = segment_colours(nodes)
    assert len({col[n] for n in nodes}) == 3 and col[0] == PALETTE[0] and col[2] == PALETTE[2]
    cl = list(classify(pred, tgt, kind))
    assert cl == ["ok", "wrong", "miss", "none", "wrong", "false"], cl
    s = token_style(cl[0], tgt[0], col)                                                # a correct token of segment 2: filled, the colour of segment 2
    assert (s["color"], s["filled"], s["marker"]) == (col[2], True, "o")
    s = token_style(cl[1], tgt[1], col)                                                # a wrong assignment: amber cross
    assert (s["color"], s["marker"]) == (AMBER, "X") and token_style(cl[4], tgt[4], col)["marker"] == "X"      # also a false token that got a segment
    s = token_style(cl[2], tgt[2], col)                                                # present but unassigned: the colour of its true segment, hollow and pale
    assert (s["color"], s["filled"], s["alpha"] < 1.0) == (col[0], False, True)
    s = token_style(cl[3], tgt[3], col)                                                # a missing segment, left unassigned: a small filled grey-blue dot
    assert (s["color"], s["filled"], s["size"] < 12) == (BLUEGREY, True, True)
    s = token_style(cl[5], tgt[5], col)
    assert (s["color"], s["size"] < 8) == (GREY, True)
    assert list(truth_classes(kind)) == ["ok", "ok", "ok", "absent", "false", "false"] and token_style("absent", -1, col)["filled"] is False
    assert list(classify(np.array([-1, 3]), np.array([-1, -1]), np.array([1, 1]))) == ["none", "wrong"]            # a missing segment given a segment is wrong
    assert link_style("ok", 2, "ok", 2, col)[0] == col[2] and link_style("ok", 2, "ok", 1, col)[0] == GREY and link_style("ok", 2, "wrong", 2, col)[0] == GREY
    assert link_style("miss", 0, "miss", 0, col)[2] != "-" and link_style("wrong", 0, "wrong", 1, col)[0] == AMBER and link_style("none", -1, "none", -1, col)[0] == GREY
    m = arrow_index(np.c_[np.arange(30.0), np.zeros(30)], (-25.0, 25.0), (-25.0, 25.0))                            # the middle (x = 14) is free
    two = [np.c_[np.zeros(30), np.arange(30.0)], np.c_[np.zeros(30), np.arange(30.0)], np.c_[np.full(5, 99.0), np.arange(5.0)]]               # two bands that start at the same point, one outside the box
    lp = place_labels(two, (-25.0, 25.0), (-5.0, 45.0), np.array([[0.0, 9.0]]))
    assert np.hypot(*(lp[0] - lp[1])) >= 9.5 and all(np.hypot(*q) >= 9.5 for q in lp[:2]) and all(-22.5 <= q[0] <= 22.5 and -2.5 <= q[1] <= 42.5 for q in lp), lp
    assert np.hypot(*(lp[0] - np.array([0.0, 9.0]))) >= 3.0                                                                               # a number keeps off the tokens
    assert m == 14 and arrow_index(np.c_[np.arange(8.0), np.zeros(8)], (-25.0, 25.0), (-25.0, 25.0)) is None and arrow_index(np.c_[np.arange(3.0), np.zeros(3)], (-9, 9), (-9, 9)) is None   # too close to the ego / too short
    # the numbers of a frame: only delivered labelled tokens count
    kept = np.array([True, True, True, True, True, False])
    st = frame_stats(pred, tgt, kind, kept)
    assert st["n"] == 5 and abs(st["acc"] - 2 / 5) < 1e-12 and (st["n0"], st["n1"], st["n2"]) == (3, 1, 1), st                     # ok + none are right
    assert (st["wrong_present"], st["unassigned_present"], st["wrong_absent"], st["false_assigned"]) == (1, 1, 0, 1)
    assert st["class_counts"] == dict(ok=1, wrong=2, miss=1, none=1, false=0)
    assert frame_stats(pred, np.array([2, 1, 0, -2, -1, -1]), kind, kept)["n"] == 4                                              # an unlabelled token (-2) is not counted
    # THE FRAME RULE on synthetic accuracies: a non-candidate frame is never chosen
    rng = np.random.RandomState(0)
    f1, z = rng.rand(40), rng.rand(40)
    n = np.full(40, 50)
    n[3], f1[3], z[3] = 10, 0.5, -5.0                                                  # a huge gain and the median F1, but only 10 tokens
    n[4], f1[4], z[4] = 29, 5.0, 0.0
    n[5], f1[5], z[5] = 40, -5.0, 5.0
    ch, nc = choose(f1, z, n)
    ok = n >= MIN_TOK
    assert nc == int(ok.sum()) and 3 not in ch.values() and 4 not in ch.values()
    assert ch["gain"] == int(np.argmax(np.where(ok, f1 - z, -np.inf))) and ch["loss"] == int(np.argmin(np.where(ok, f1 - z, np.inf))) == 5
    med = np.median(f1[ok])
    assert abs(f1[ch["typical"]] - med) <= min(abs(f1[j] - med) for j in np.where(ok)[0]) + 1e-12
    for h in PALETTE + [AMBER, GREY, BLUEGREY, DARK]:                                  # no red
        r, g, b = (int(h[i:i + 2], 16) for i in (1, 3, 5))
        assert not (r > 150 and g < 110 and b < 110), f"a red colour {h}"
    assert len(set(PALETTE)) == 8 and len({AMBER, GREY, BLUEGREY, DARK} & set(PALETTE)) == 0 and segment_colours(list(range(9)))[8] == PALETTE[0]
    print("map_assoc_fig selftest ok")


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
