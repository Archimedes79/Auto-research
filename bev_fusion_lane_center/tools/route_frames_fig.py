"""E389 report: the example frames of the route-association head, chosen by a WRITTEN RULE and drawn from the stored dumps (no model run, CPU).

    bash tools/env/_py14.sh route_frames_fig.py --seed 0 --out <absolute dir>     # _py14.sh changes the directory: give an absolute --out
    python tools/route_frames_fig.py --selftest                                  # the rule, the outcome colours and the route segments on hand-made data, no data files

Reads `runs/e389_{z,m}_s<S>/route_{error,wrong}.pkl` (p_road per token), the champion's delivered set (`jroute_eval.champion_pnone`, p_none < 0.6) and the route cache; per frame the F1_road of the covered
nodes of z and of m at the mode error and of m at the mode wrong. Frames with fewer than MIN_NODES covered nodes are not candidates. The rule (fixed before the pictures were looked at; the frames are NOT
re-picked after looking at a picture):
  typical = the fork frame whose F1 gain m - z is closest to the median gain of the fork frames;  gain = the frame with the largest m - z;  loss = the frame with the smallest m - z;
  wrong = the frame with the largest drop of m from the mode error to the mode wrong.
Writes one png per case (two panels stacked, z above, m below, German labels, the legend below the axes; the covered delivered nodes coloured by outcome; the SD route as the head saw it as a thick translucent
line, the exact route as a thin dashed one) and `e389_frames.json` with the numbers of the chosen frames (the report captions read it).
The route table of a frame is the polylines of its route pieces CONCATENATED (pieces in driving order): it is drawn as one line per run of neighbouring samples (`route_segments`), never as one connected line
(consecutive samples of different pieces / branches are far apart or turn sharply, and joining them draws spikes).
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
MIN_NODES = 30
MAX_GAP = 3.0                                                                      # m: the sampling step of the route table is 2 m; further apart = another piece / branch
MAX_TURN_COS = 0.5                                                                 # cos 60 degrees: a sharper change of the travel direction between neighbours starts a new segment
OUT_COL = dict(tp="#009E73", tn="#b9b9b9", fp="#E69F00", fn="#0072B2")           # outcome colours: Okabe-Ito (readable with a colour weakness), no red
OUT_TXT = dict(tp="auf der Route, markiert (richtig)", tn="nicht auf der Route, nicht markiert (richtig)", fp="nicht auf der Route, aber markiert (Fehler)", fn="auf der Route, nicht markiert (Fehler)")


def f1_of(tp, fp, fn):
    return 2.0 * tp / max(2.0 * tp + fp + fn, 1e-12)


def frame_f1(cov, lab, flag):
    """F1 of one frame over the covered nodes: cov (K,) bool, lab (K,) 0 / 1, flag (K,) bool -> (f1, tp, fp, fn)"""
    tp, fp, fn = int((cov & (lab == 1) & flag).sum()), int((cov & (lab == 0) & flag).sum()), int((cov & (lab == 1) & ~flag).sum())
    return f1_of(tp, fp, fn), tp, fp, fn


def outcome(lab, flag):
    """'tp' flagged and on the route, 'tn' not flagged and off the route, 'fp' flagged but off the route, 'fn' not flagged but on the route (arrays)"""
    return np.where(flag & (lab == 1), "tp", np.where(~flag & (lab == 0), "tn", np.where(flag, "fp", "fn")))


def route_segments(pos, dirv, valid, max_gap=MAX_GAP, min_cos=MAX_TURN_COS):
    """the route table (M, 2) positions, (M, 2) unit directions, (M,) valid -> a list of (n, 2) arrays: the valid samples in table order, cut wherever two consecutive samples are further apart than max_gap
    or the travel direction turns by more than acos(min_cos); a segment of one sample is dropped"""
    p, d = np.asarray(pos, float)[np.asarray(valid, bool)], np.asarray(dirv, float)[np.asarray(valid, bool)]
    if len(p) < 2:
        return []
    cut = (np.linalg.norm(p[1:] - p[:-1], axis=1) > max_gap) | ((d[1:] * d[:-1]).sum(1) < min_cos)
    idx = np.concatenate([[0], np.where(cut)[0] + 1, [len(p)]])
    return [p[a:b] for a, b in zip(idx[:-1], idx[1:]) if b - a >= 2]


def choose(gain, drop, fork, ok):
    """the written rule -> {case: frame index}; gain = F1_m - F1_z per frame, drop = F1_m(error) - F1_m(wrong), fork (F,) bool, ok (F,) bool = candidate frames (a route and >= MIN_NODES covered nodes)"""
    gain, drop = np.where(ok, gain, np.nan), np.where(ok, drop, np.nan)
    fk = np.where(fork & ok)[0]
    med = float(np.median(gain[fk]))
    return dict(typical=int(fk[np.argmin(np.abs(gain[fk] - med))]), gain=int(np.nanargmax(gain)), loss=int(np.nanargmin(gain)), wrong=int(np.nanargmax(drop)))


def limits(pos, cov, margin=4.0):
    """the crop of a frame: the covered nodes plus a margin, within +-50 m"""
    p = pos[cov]
    lo, hi = np.maximum(p.min(0) - margin, -50.0), np.minimum(p.max(0) + margin, 50.0)
    return (float(lo[0]), float(hi[0])), (float(lo[1]), float(hi[1]))


def draw(path, title, side, i, routes, rows):
    """one png: two panels (rows = [(title, flag (K,) bool, show_sd)]) of frame i, the legend below the axes; routes = dict(exact=table, shown=table) with pos, dir, valid. The SD route is drawn ONLY in the panels
    whose head saw it (show_sd): the head without the map saw none"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    cov = side["_cov"][i]
    pos, lab = side["pos"][i], side["lab_road"][i]
    xl, yl = limits(pos, cov)
    fig, axs = plt.subplots(len(rows), 1, figsize=(5.6, 6.5 * len(rows) + 1.8), sharex=True, sharey=True)
    for ax, (name, flag, show_sd) in zip(np.atleast_1d(axs), rows):
        oc = outcome(lab, flag)
        for seg in (route_segments(routes["shown"]["pos"], routes["shown"]["dir"], routes["shown"]["valid"]) if show_sd else []):
            ax.plot(seg[:, 0], seg[:, 1], "-", c="#1d2330", lw=7, alpha=0.28, solid_capstyle="round", zorder=1)
        for seg in route_segments(routes["exact"]["pos"], routes["exact"]["dir"], routes["exact"]["valid"]):
            ax.plot(seg[:, 0], seg[:, 1], "--", c="#555555", lw=1.1, zorder=2)
        for k, c in OUT_COL.items():
            m = cov & (oc == k)
            ax.scatter(pos[m, 0], pos[m, 1], s=26, c=c, linewidths=0, zorder=3)
        ax.plot([0], [0], marker="^", c="#1d2330", ms=9, ls="none", zorder=4)
        n = {k: int((cov & (oc == k)).sum()) for k in OUT_COL}
        ax.set_aspect("equal")
        ax.set_xlim(*xl)
        ax.set_ylim(*yl)
        ax.set_title(name, fontsize=14)
        ax.set_xlabel(f"x (m)\nPunkte: grün {n['tp']} | blau {n['fn']} | orange {n['fp']}", fontsize=12)
        ax.set_ylabel("y (m, nach vorn)", fontsize=12)
        ax.tick_params(labelsize=11)
        ax.grid(alpha=0.2)
    handles = [Line2D([0], [0], marker="o", ls="none", c=OUT_COL[k], ms=7, label=OUT_TXT[k]) for k in OUT_COL]
    handles += [Line2D([0], [0], c="#1d2330", lw=7, alpha=0.28, label="SD-Route (nur unten gesehen)"), Line2D([0], [0], c="#555555", lw=1.1, ls="--", label="exakte Route (Wahrheit)"),
                Line2D([0], [0], marker="^", ls="none", c="#1d2330", ms=8, label="Fahrzeug")]
    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=(0, 0.14, 1, 0.95), h_pad=2.5)
    fig.subplots_adjust(hspace=0.5)                                                  # room for the x label of the upper panel above the title of the lower one
    fig.legend(handles=handles, loc="lower center", ncol=1, fontsize=12, frameon=False)
    fig.savefig(path, dpi=110)
    plt.close(fig)


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
    lab = np.array([1, 1, 0, 0])
    flag = np.array([True, False, True, False])
    assert list(outcome(lab, flag)) == ["tp", "fn", "fp", "tn"]
    f, tp, fp, fn = frame_f1(np.ones(4, bool), lab, flag)
    assert (tp, fp, fn) == (1, 1, 1) and abs(f - 0.5) < 1e-12
    for h in OUT_COL.values():
        r, g, b = (int(h[i:i + 2], 16) for i in (1, 3, 5))
        assert not (r > 150 and g < 110 and b < 110), f"a red colour {h}"
    # the route table: two pieces (a straight one along y, then a piece that starts 20 m to the side) -> two segments, not one line through the jump
    y = np.arange(0.0, 20.0, 2.0)
    a = np.stack([np.zeros_like(y), y], 1)
    b = np.stack([np.full_like(y, 20.0), y], 1)
    pos = np.concatenate([a, b])
    dr = np.tile([0.0, 1.0], (len(pos), 1))
    segs = route_segments(pos, dr, np.ones(len(pos), bool))
    assert len(segs) == 2 and len(segs[0]) == 10 and len(segs[1]) == 10, [len(s) for s in segs]
    # one continuous piece with the 2 m step stays ONE segment; the joint of two connected pieces (same step) too
    assert len(route_segments(a, dr[:10], np.ones(10, bool))) == 1
    assert len(route_segments(np.concatenate([a, a[-1] + a[1:] - a[0]]), np.tile([0.0, 1.0], (19, 1)), np.ones(19, bool))) == 1
    # a sharp turn between close samples starts a new segment (a branch that leaves at once), invalid samples are dropped, a single sample is no segment
    dr2 = dr.copy()
    dr2[10:] = [1.0, 0.0]
    p2 = np.concatenate([a, a[-1] + 2.0 * np.arange(1, 11)[:, None] * np.array([[1.0, 0.0]])])
    assert len(route_segments(p2, dr2, np.ones(len(p2), bool))) == 2
    v = np.ones(len(pos), bool)
    v[5:] = False
    assert len(route_segments(pos, dr, v)) == 1 and len(route_segments(pos[:1], dr[:1], np.ones(1, bool))) == 0
    xl, yl = limits(np.array([[0.0, 0.0], [10.0, 20.0]]), np.array([True, True]))
    assert xl == (-4.0, 14.0) and yl == (-4.0, 24.0)
    print("route_frames_fig selftest ok")


def main(a):
    sys.path.insert(0, os.path.join(ROOT, "evaluation"))
    import pickle
    import jroute_eval as JE
    seed = a.seed
    side = JE.load_side(seed, mode="wrong")
    pn, _ = JE.champion_pnone(side, seed)
    dlv = JE.delivered_kept(side["valid"], pn, JE.KEEP_THR)

    def pflags(run, mode):
        d = pickle.load(open(os.path.join(JE.RUNS, run, f"route_{mode}.pkl"), "rb"))
        assert [f["token"] for f in d["frames"]] == side["tokens"], run
        return np.stack([np.asarray(f["p_road"]) >= JE.THR for f in d["frames"]])

    fl = dict(z=pflags(f"e389_z_s{seed}", "error"), m=pflags(f"e389_m_s{seed}", "error"), mw=pflags(f"e389_m_s{seed}", "wrong"))
    cov = dlv & side["valid"] & (side["gt_id"] >= 0) & side["has_route"][:, None]
    side["_cov"] = cov
    F = len(side["tokens"])
    f1 = {k: np.array([frame_f1(cov[i], side["lab_road"][i], fl[k][i])[0] for i in range(F)]) for k in fl}
    ok = side["has_route"] & (cov.sum(1) >= MIN_NODES)
    pick = choose(f1["m"] - f1["z"], f1["m"] - f1["mw"], side["decision"] & side["has_route"], ok)
    os.makedirs(a.out, exist_ok=True)
    info = dict(seed=seed, min_nodes=MIN_NODES, candidates=int(ok.sum()), cases={})
    titles = {"typical": "Typischer Frame mit Abzweig (mittlerer Gewinn)", "gain": "Größter Gewinn der Karte", "loss": "Größter Verlust der Karte", "wrong": "Größter Absturz mit falscher Karte"}
    for case, i in pick.items():
        ds = side["ds"]
        tab = {m_: ds.table(i, m_) for m_ in ("exact", "error", "wrong")}
        tab = {k: dict(pos=np.asarray(v["pos"]), dir=np.asarray(v["dir"]), valid=np.asarray(v["valid"], bool)) for k, v in tab.items()}
        shown = "wrong" if case == "wrong" else "error"
        mrow = "mw" if case == "wrong" else "m"
        extra = f" (mit richtiger Karte: {f1['m'][i]:.3f})" if case == "wrong" else ""
        rows = [(f"Kopf ohne Karte (z)\nF1 Fahrbahn {f1['z'][i]:.3f}", fl["z"][i], False),
                (f"Kopf mit {'FALSCHER ' if case == 'wrong' else ''}Karte (m)\nF1 Fahrbahn {f1[mrow][i]:.3f}{extra}", fl[mrow][i], True)]
        fn = f"e389_frame_{case}.png"
        draw(os.path.join(a.out, fn), f"{titles[case]}\nFrame {i}, Szene {int(side['scene'][i])}", side, i, dict(exact=tab["exact"], shown=tab[shown]), rows)
        info["cases"][case] = dict(frame=i, scene=int(side["scene"][i]), nodes=int(cov[i].sum()), fork=bool(side["decision"][i]), f1_z=float(f1["z"][i]), f1_m=float(f1["m"][i]), f1_m_wrong=float(f1["mw"][i]),
                                   gain=float(f1["m"][i] - f1["z"][i]), drop=float(f1["m"][i] - f1["mw"][i]), file=fn)
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
