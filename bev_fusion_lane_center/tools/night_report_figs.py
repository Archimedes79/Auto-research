"""Figures of the night report 2026-11-25 (D148), drawn from stored result files and stored models (CPU only: no training, no GPU, no run.py job).

    bash tools/env/_py14.sh night_report_figs.py --out <absolute dir>                  # drawing cut + the four charts (needs files that are in git only)
    bash tools/env/_py14.sh night_report_figs.py --out <absolute dir> --frames         # also the two example frames (needs runs/, the caches of E365 / E373 / E390 / E391 and nuScenes)
    bash tools/env/_py14.sh night_report_figs.py --selftest                            # no data files

champion_architecture.svg: a cut of `training/architecture.svg` (drawn by `tools/arch/arch_drawing.py`): the main path, the evaluation and the training stages plus the legend; the side branch and the reference systems are left out.
e395_shift.png: E395 step 2 and addendum A1, F1@0.5 and F1@1.0 of the moved against the unmoved nodes (paired 95 % interval) against the card's bars 0.033 / 0.010.
e391b_assoc.png: E391b part B: association accuracy and share of true tokens left unassigned, the null gate g0, E391's F1 and E391b's F1d against the card's bar and guard.
e393_arms.png: E393: topo F1 of the arms n, e, p minus c (paired 95 % interval) against +0.010; p seed 1 is VOID and not drawn.
t25_arms.png: BEV card t25: `detf` mean mAP and velocity error of moving objects, arm minus control (compare files), against the card's limits.
e391b_frame_{typical,loss}.png + e391b_frames.json: two VAL frames (seed 0, deciding map level) in the style of `map_assoc_fig.py`: truth, E391's F1, E391b's F1d. The frame rule was written before any picture was looked at:
candidates = VAL frames with at least `map_assoc_fig.MIN_TOK` delivered labelled tokens; typical = the candidate whose F1d accuracy is closest to the median; loss = the smallest F1d minus F1. The pooled accuracy of both heads is
compared with the stored cards before anything is drawn.
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
REPO = os.path.abspath(os.path.join(ROOT, ".."))
DATA = os.path.join(ROOT, "results", "E2E_Report_2026-11-25", "data", "runs")
DATA23 = os.path.join(ROOT, "results", "E2E_Report_2026-11-23", "data", "runs")
T25 = os.path.join(REPO, "bev_fusion", "experiments", "t25_one_frame_one_scan")
BLUE, BLUE2, GREY, DARK = "#0072B2", "#56B4E9", "#9a9a9a", "#444444"
E395_ARMS = [("p0", "p0: Catmull-Rom-Loss (Schritt 2)"), ("p2", "p2: nach zwei Runden (Schritt 2)"), ("p0n", "p0n: Knoten-Loss (Schritt 2)"), ("p0nt", "p0nt: p0n, Kopf-Seeds 4 / 5 (A1)"),
             ("p0ntb", "p0ntb: p0n, Kopf-Seeds 6 / 7 (A1)")]
E395_BARS = (("F1@0.5", 0.033), ("F1@1.0", 0.010))                                    # the card's bars (E395 section 3)
E393_BAR = 0.010
T25_PANELS = (("detf.dt+0.00/mean/all/mAP", "`detf` mittlere mAP (rechts = besser)", ((-0.010, "HOLDS ab -0.010"), (-0.020, "FAILS unter -0.020", "right"))),
              ("vel.dt+0.00/centre/field/moving/epe", "Geschwindigkeits-Fehler bewegter Objekte, m/s (links = besser)", ((0.10, "Schutzgrenze +0.10"), (0.20, "Schnitt +0.20"))))
G0 = re.compile(r"g0 \(VAL, point estimates\) reaches assoc_acc ([\d.]+) / ([\d.]+) against the bars ([\d.]+) / ([\d.]+), false_unassigned ([\d.]+) / ([\d.]+) \(guard ([\d.]+)")
ROW = r"^%s\s+a \S+\s+b \S+\s+b-a ([+-][\d.]+)\s+95%% \[([+-][\d.]+), ([+-][\d.]+)\]"


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def arch_cut(svg, top=869, legend_y=1486, legend_h=80):
    """the drawing's text -> a new svg that shows its upper `top` units and, below them, its legend strip (two views of the same content)"""
    m = re.match(r"\s*(?:<\?xml[^>]*\?>\s*)?<svg\b([^>]*)>(.*)</svg>\s*$", svg, re.S)
    assert m, "not an svg document"
    attrs, inner = m.group(1), m.group(2)
    width = int(float(re.search(r'viewBox="0 0 ([\d.]+) [\d.]+"', attrs).group(1)))
    font = re.search(r'font-family="[^"]*"', attrs)
    view = lambda y, vy, h: '<svg x="0" y="%d" width="%d" height="%d" viewBox="0 %d %d %d">%s</svg>' % (y, width, h, vy, width, h, inner)   # noqa: E731
    return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="%d" height="%d" %s>%s%s</svg>\n' % (
        width, top + legend_h, width, top + legend_h, font.group(0) if font else "", view(0, 0, top), view(top, legend_y, legend_h))


def e395_rows(read, metric):
    """read(arm, seed) -> the card's `paired_M_vs_U` -> [(label, [(seed, diff, lo, hi)])]"""
    return [(lab, [(s, *[read(a, s)[metric][k] for k in ("diff", "lo", "hi")]) for s in (0, 1)]) for a, lab in E395_ARMS]


def compare_row(text, row):
    m = re.search(ROW % re.escape(row), text, re.M)
    assert m, "row %s not in the compare file" % row
    return tuple(float(x) for x in m.groups())


def parse_g0(text):
    m = G0.search(text)
    assert m, "the g0 sentence of the verdict is not found"
    v = [float(x) for x in m.groups()]
    return dict(acc=v[0:2], bar=v[2:4], fu=v[4:6], guard=v[6])


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _clean(ax):
    ax.grid(axis="x", alpha=0.25)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def dots(ax, rows, lines=(), void=None, fmt="{:+.4f}", zero=True, two=True):
    """rows [(label, [(seed, diff, lo, hi)])] as dots with intervals, the value written at the right end of every interval; `lines` [(x, text[, side])] dashed, the text right ('left', the default) or left ('right') of its
    line; void {(label, seed): text}"""
    n = len(rows)
    xs = [v for _, pts in rows for p in pts for v in p[1:] if v is not None] + [ln[0] for ln in lines] + ([0.0] if zero else [])
    lo_x, hi_x = min(xs), max(xs)
    span = hi_x - lo_x
    for i, (lab, pts) in enumerate(rows):
        y = n - 1 - i
        for s, d, lo, hi in pts:
            yy = y + ((0.18 if s == 0 else -0.18) if two else 0.0)
            col = BLUE if s == 0 else BLUE2
            if lo is not None:
                ax.plot([lo, hi], [yy, yy], "-", color=col, lw=2.2, solid_capstyle="round")
            ax.plot([d], [yy], "o", color=col, ms=6.0, zorder=3, label="Seed %d" % s)
            ax.text((hi if hi is not None else d) + 0.012 * span, yy, fmt.format(d), va="center", ha="left", fontsize=8.5, color=DARK, zorder=4, bbox=dict(facecolor="white", edgecolor="none", pad=0.3))
        for s in (0, 1):
            if void and (lab, s) in void:
                ax.text(lo_x + 0.02 * span, y - 0.18, void[(lab, s)], va="center", ha="left", fontsize=8.5, color=DARK, style="italic")
    if zero:
        ax.axvline(0, color=DARK, lw=1.0)
    for x, text, *side in lines:
        ax.axvline(x, color=DARK, lw=0.9, ls=(0, (4, 3)))
        if text:
            ax.text(x, n - 0.42, " %s " % text, fontsize=8.5, color=DARK, va="bottom", ha=side[0] if side else "left", zorder=4, bbox=dict(facecolor="white", edgecolor="none", pad=0.4))
    ax.set_xlim(lo_x - 0.05 * span, hi_x + 0.17 * span)
    ax.set_ylim(-0.55, n - 0.05)
    ax.set_yticks(range(n))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=9.5)
    ax.tick_params(axis="x", labelsize=8.5)
    _clean(ax)


def _legend(fig, ax):
    h, l = ax.get_legend_handles_labels()
    seen = dict(zip(l, h))
    fig.legend([seen[k] for k in sorted(seen)], sorted(seen), loc="lower center", ncol=2, fontsize=9.5, frameon=False)


def stacked(panels, path, height, legend=True):
    """panels [(title, kwargs of dots)] one under the other"""
    plt = _plt()
    fig, axs = plt.subplots(len(panels), 1, figsize=(7.6, height), squeeze=False)
    for ax, (title, kw) in zip(axs[:, 0], panels):
        dots(ax, **kw)
        ax.set_title(title.replace("`", ""), fontsize=10.5, loc="left")
    if legend:
        _legend(fig, axs[0, 0])
    fig.tight_layout(rect=(0, 0.45 / height if legend else 0.0, 1, 1), h_pad=1.6)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def charts(out):
    read395 = lambda a, s: load(os.path.join(DATA, "e395_step2", "e395_%s_s%d.json" % (a, s)))["paired_M_vs_U"]        # noqa: E731
    stacked([("E395 %s: verschoben minus unverschoben (Latte +%.3f)" % (m, b), dict(rows=e395_rows(read395, m), lines=((b, "Latte", "right"),))) for m, b in E395_BARS],
            os.path.join(out, "e395_shift.png"), 7.0)
    text = open(os.path.join(DATA, "e391b_partB", "verdict_d.md"), encoding="utf-8").read()
    g0 = parse_g0(text)
    old = [load(os.path.join(DATA23, "e391_F1_s%d" % s, "card_val.json"))["levels"]["dec"] for s in (0, 1)]
    new = [load(os.path.join(DATA, "e391b_F1d_s%d" % s, "card_val.json"))["levels"]["dec"] for s in (0, 1)]
    panels = []
    for key, g0key, title, lines in (("assoc_acc", "acc", "assoc_acc (rechts = besser)", ((min(g0["bar"]), "Latte S1 / S0", "right"), (max(g0["bar"]), ""))),
                                     ("false_unassigned", "fu", "wahre Token nicht zugeordnet (links = besser)", ((g0["guard"], "Grenze %.2f" % g0["guard"]),))):
        rows = [("g0: Tor ohne Karte (Punktwert)", [(s, g0[g0key][s], None, None) for s in (0, 1)]),
                ("E391 F1: Loss auf allen Token", [(s, *old[s][key]) for s in (0, 1)]),
                ("E391b F1d: Loss auf gelieferten", [(s, *new[s][key]) for s in (0, 1)])]
        panels.append(("E391b: " + title, dict(rows=rows, lines=lines, fmt="{:.4f}", zero=False)))
    stacked(panels, os.path.join(out, "e391b_assoc.png"), 5.4)
    rows = []
    for a, lab in (("n", "n: ohne Anker, Pyramide fest"), ("e", "e: ohne Anker, Pyramide trainiert"), ("p", "p: mit Anker, Pyramide trainiert")):
        pts = []
        for s in (0, 1):
            if (a, s) == ("p", 1):
                continue
            card = load(os.path.join(DATA, "e393_%s_s%d" % (a, s), "token_card_rec_lt0.3_ltl0.2_kt0.6_nng_live_vs_e393_c_s%d.json" % s))
            d = card["info"]["paired_vs_ref_run"]["topo_F1"]
            pts.append((s, d["diff"], d["lo"], d["hi"]))
        rows.append((lab, pts))
    stacked([("E393: topo F1 des Arms minus c (VAL, 95-%-Intervall)", dict(rows=rows, lines=((E393_BAR, "Latte +0.010"),), void={(rows[2][0], 1): "Seed 1 ungültig (CUDA-Fehler)"}))],
            os.path.join(out, "e393_arms.png"), 3.1)
    cmp = {a: open(os.path.join(T25, "compare_ctl_vs_%s.txt" % a), encoding="utf-8").read() for a in ("R", "A1", "A2")}
    names = {"R": "R: Kamera rekursiv", "A1": "A1: ein Radar-Scan", "A2": "A2: ein Scan + Gedächtnis"}
    stacked([("t25: %s" % title, dict(rows=[(names[a], [(0, *compare_row(cmp[a], row))]) for a in ("R", "A1", "A2")], lines=lines, two=False)) for row, title, lines in T25_PANELS],
            os.path.join(out, "t25_arms.png"), 4.2, legend=False)


def frames(out, seed=0):
    import pickle

    import numpy as np
    import torch
    import lane_mapassoc_data as MA
    import map_assoc_fig as MF
    import mapassoc_exist as ME
    import mapassoc_probe as MP
    import sdgraph_probe as SP
    torch.set_num_threads(2)
    data = MP.AssocData("val", seed, level="dec")
    dl = torch.utils.data.DataLoader(data, batch_size=8, shuffle=False, num_workers=0, collate_fn=MP.collate)
    enc, _ = SP.load_models("g7", seed)
    enc = enc.eval()
    heads = {"old": MP.load_head("F1", seed)}
    with ME.patched():                                                                  # the same loader, the folder of E391b part B
        assert MP.runs_dir("F1", seed).endswith("e391b_F1d_s%d" % seed)
        heads["new"] = MP.load_head("F1", seed)
    preds = {}
    with torch.no_grad():
        for b in dl:
            h, sv = MP.encode(enc, b, "cpu")
            o = {k: head(b, h, sv, b["pa"])["assoc"] for k, head in heads.items()}
            for r in range(len(b["frame"])):
                p = {k: v[r].argmax(-1).numpy() for k, v in o.items()}
                preds[int(b["frame"][r])] = {k: np.where(p[k] == o[k].shape[-1] - 1, -1, p[k]) for k in p}
    F = len(data)
    items = [data[j] for j in range(F)]
    stats = {k: [MF.frame_stats(preds[i][k], items[i]["assoc"], items[i]["kind"], items[i]["kept"]) for i in range(F)] for k in ("old", "new")}
    stored = {"old": load(os.path.join(DATA23, "e391_F1_s%d" % seed, "card_val.json")), "new": load(os.path.join(DATA, "e391b_F1d_s%d" % seed, "card_val.json"))}
    pooled = {}
    for k in ("old", "new"):
        pooled[k] = sum(s["acc"] * s["n"] for s in stats[k] if s["n"]) / sum(s["n"] for s in stats[k])
        want = stored[k]["levels"]["dec"]["assoc_acc"][0]
        print("pooled assoc_acc %s: %.4f, stored card %.4f" % (k, pooled[k], want), flush=True)
        assert abs(pooled[k] - want) < 5e-4, "the heads do not reproduce the stored card"
    pick, n_cand = MF.choose(np.array([s["acc"] for s in stats["new"]]), np.array([s["acc"] for s in stats["old"]]), np.array([s["n"] for s in stats["new"]]))
    scene = MA.load("val", seed)["scene"]
    dump = pickle.load(open(os.path.join(MF.RF.ROOT, "runs", "e373_cb_s%d" % seed, "token_val.pkl"), "rb"))["frames"]
    assert [f["token"] for f in dump[:F]] == list(data.ds.tokens), "the dump frames are not the frames of the association data"
    info = dict(seed=seed, level="dec", min_tok=MF.MIN_TOK, frames=F, candidates=n_cand, pooled={k: round(v, 4) for k, v in pooled.items()}, cases={})
    titles = {"typical": "Typischer Frame (mittlere Zuordnungsgenauigkeit von F1d)", "loss": "Größter Verlust von E391b F1d gegen E391 F1"}
    for case in ("typical", "loss"):
        i = pick[case]
        it, tok = items[i], data.ds.tokens[i]
        pos, dirv, valid = np.asarray(it["pos"], float), np.asarray(it["dirv"], float), np.asarray(it["valid"])
        tgt, kind, kept = it["assoc"], it["kind"], it["kept"]
        drawn = kept & (tgt >= -1)
        _, links = MF.delivered_graph(dump[i], valid, pos, dirv)
        Gin = MF.perturbed_map(data, i)
        nodes = MF.select_nodes(Gin["geo"], Gin["present"], pos[drawn], tgt[drawn])
        colours = MF.segment_colours(nodes)
        bands = [(n, [np.asarray(p, float) for p in Gin["geo"][n] if len(p) >= 2]) for n in nodes]
        seg_t = np.where(tgt >= 0, tgt, 0)
        panels = [dict(title="Wahrheit\nToken nach ihrem wahren Segment", cls=MF.truth_classes(kind), seg=seg_t),
                  dict(title="E391 F1 (Loss auf allen Token)\nZuordnung %.3f" % stats["old"][i]["acc"], cls=MF.classify(preds[i]["old"], tgt, kind), seg=seg_t),
                  dict(title="E391b F1d (Loss auf gelieferten)\nZuordnung %.3f" % stats["new"][i]["acc"], cls=MF.classify(preds[i]["new"], tgt, kind), seg=seg_t)]
        for p in panels:
            p.update(pos=pos, drawn=drawn, links=links)
        g, gt_lines = MF.load_gt_lines(tok)
        xl, yl = MF.limits(pos[drawn] if drawn.any() else pos[kept])
        table = MF.table_of(stats["old"][i], stats["new"][i])
        table[0] = ["", "E391 F1", "E391b F1d"]
        fn = "e391b_frame_%s.png" % case
        sub = ("Szene %d, Frame %d (VAL, Seed 0), %d Segmente der Karte gezeigt, %d gelieferte Token mit Label\n"
               "Karte mit Fehlern: Lage 1 m, je 10 %% Spurzahlen / Verbindungen / Segmente falsch bzw. fehlend, 30 %% Route verborgen") % (int(scene[i]), i, len(nodes), int(drawn.sum()))
        MF.draw_frame(os.path.join(out, fn), titles[case], sub, MF.camera_of(tok, g), table, panels, gt_lines, xl, yl, bands, colours)
        info["cases"][case] = dict(frame=i, scene=int(scene[i]), tokens=int(drawn.sum()), file=fn, acc_old=round(stats["old"][i]["acc"], 3), acc_new=round(stats["new"][i]["acc"], 3))
        print(case, info["cases"][case], flush=True)
    with open(os.path.join(out, "e391b_frames.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=1)


def selftest():
    svg = '<?xml version="1.0"?>\n<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 200" width="100" height="200" font-family="A, B"><rect width="100" height="200"/><text y="150">x</text></svg>\n'
    cut = arch_cut(svg, top=60, legend_y=140, legend_h=20)
    assert 'viewBox="0 0 100 80" width="100" height="80" font-family="A, B"' in cut and 'viewBox="0 0 100 60"' in cut and 'y="60" width="100" height="20" viewBox="0 140 100 20"' in cut and cut.count("<text") == 2, cut
    import xml.dom.minidom
    xml.dom.minidom.parseString(cut)
    line = "detf.dt+0.00/mean/all/mAP    a 0.4135  b 0.4121  b-a -0.0013  95% [-0.0071, +0.0046]  draws 1000/1000\n"
    assert compare_row("head\n" + line, "detf.dt+0.00/mean/all/mAP") == (-0.0013, -0.0071, 0.0046)
    g = parse_g0("... g0 (VAL, point estimates) reaches assoc_acc 0.7 / 0.6 against the bars 0.8 / 0.9, false_unassigned 0.06 / 0.04 (guard 0.05): ...")
    assert g == dict(acc=[0.7, 0.6], bar=[0.8, 0.9], fu=[0.06, 0.04], guard=0.05)
    rows = e395_rows(lambda a, s: {"F1@0.5": dict(diff=0.01 * (s + 1), lo=0.0, hi=0.03)}, "F1@0.5")
    assert len(rows) == len(E395_ARMS) and rows[0][1][1] == (1, 0.02, 0.0, 0.03)
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "a.png")
        stacked([("t", dict(rows=rows, lines=((0.033, "Latte"),), void={(rows[0][0], 1): "ungültig"})), ("u", dict(rows=[("g", [(0, 0.7, None, None)])], zero=False, fmt="{:.4f}"))], p, 4.0)
        assert os.path.getsize(p) > 5000
    for h in (BLUE, BLUE2, GREY, DARK):                                                 # no red
        r, gg, b = (int(h[i:i + 2], 16) for i in (1, 3, 5))
        assert not (r > 150 and gg < 110 and b < 110), "a red colour %s" % h
    print("night_report_figs selftest ok")


def main(a):
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(ROOT, "training", "architecture.svg"), encoding="utf-8") as f:
        cut = arch_cut(f.read())
    with open(os.path.join(a.out, "champion_architecture.svg"), "w", encoding="utf-8", newline="\n") as f:
        f.write(cut)
    charts(a.out)
    if a.frames:
        frames(a.out)
    print("written", a.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--frames", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        assert a.out and os.path.isabs(a.out), "--out <absolute dir>"
        main(a)
        sys.exit(0)
