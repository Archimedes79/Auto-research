"""E389 report: the example frames of the route-association head, chosen by a WRITTEN RULE and drawn from the stored dumps (no model run, CPU).

    python tools/route_frames_fig.py --seed 0 --out <dir>          # run from bev_fusion_lane_center/ with the gcar python
    python tools/route_frames_fig.py --selftest                      # the rule and the outcome colours on hand-made frames, no data

Reads `runs/e389_{z,m}_s<S>/route_{error,wrong}.pkl` (p_road per token), the champion's delivered set (`jroute_eval.champion_pnone`, p_none < 0.6) and the route cache; per frame the F1_road of the covered
nodes of z and of m at the mode error and of m at the mode wrong. Frames with fewer than MIN_NODES covered nodes are not candidates. The rule (fixed before the pictures were looked at):
  typical = the fork frame whose F1 gain m - z is closest to the median gain of the fork frames;  gain = the frame with the largest m - z;  loss = the frame with the smallest m - z;
  wrong = the frame with the largest drop of m from the mode error to the mode wrong.
Writes one png per case (z left, m right; the covered delivered nodes coloured by outcome: green flagged and on the route, grey not flagged and off the route, amber flagged but off the route, blue
not flagged but on the route; the SD route as the map head saw it as a black line, the exact route dashed) and `e389_frames.json` with the numbers of the chosen frames (the captions read it).
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
MIN_NODES = 30
OUT_COL = dict(tp="#2f6b3c", tn="#b9b9b9", fp="#d9962b", fn="#31507f")           # outcome colours (no red)


def f1_of(tp, fp, fn):
    return 2.0 * tp / max(2.0 * tp + fp + fn, 1e-12)


def frame_f1(cov, lab, flag):
    """F1 of one frame over the covered nodes: cov (K,) bool, lab (K,) 0 / 1, flag (K,) bool -> (f1, tp, fp, fn)"""
    tp, fp, fn = int((cov & (lab == 1) & flag).sum()), int((cov & (lab == 0) & flag).sum()), int((cov & (lab == 1) & ~flag).sum())
    return f1_of(tp, fp, fn), tp, fp, fn


def outcome(lab, flag):
    """'tp' flagged and on the route, 'tn' not flagged and off the route, 'fp' flagged but off the route, 'fn' not flagged but on the route (arrays)"""
    return np.where(flag & (lab == 1), "tp", np.where(~flag & (lab == 0), "tn", np.where(flag, "fp", "fn")))


def choose(gain, drop, fork, ok):
    """the written rule -> {case: frame index}; gain = F1_m - F1_z per frame, drop = F1_m(error) - F1_m(wrong), fork (F,) bool, ok (F,) bool = candidate frames (a route and >= MIN_NODES covered nodes)"""
    gain, drop = np.where(ok, gain, np.nan), np.where(ok, drop, np.nan)
    fk = np.where(fork & ok)[0]
    med = float(np.median(gain[fk]))
    return dict(typical=int(fk[np.argmin(np.abs(gain[fk] - med))]), gain=int(np.nanargmax(gain)), loss=int(np.nanargmin(gain)), wrong=int(np.nanargmax(drop)))


def draw(path, title, side, i, flags, routes, rows):
    """one png: two panels (rows = [(name, flag (K,) bool, the F1 text)]) of frame i"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cov = side["_cov"][i]
    pos, lab = side["pos"][i], side["lab_road"][i]
    fig, axs = plt.subplots(1, len(rows), figsize=(5.2 * len(rows), 6.2), sharex=True, sharey=True)
    for ax, (name, flag, txt) in zip(np.atleast_1d(axs), rows):
        oc = outcome(lab, flag)
        for k, c in OUT_COL.items():
            m = cov & (oc == k)
            ax.scatter(pos[m, 0], pos[m, 1], s=14, c=c, label=f"{k}: {int(m.sum())}", linewidths=0)
        ex, er = routes["exact"], routes["shown"]
        if ex["valid"].any():
            ax.plot(ex["pos"][ex["valid"], 0], ex["pos"][ex["valid"], 1], "--", c="#888888", lw=1.2, label="exact route")
        if er["valid"].any():
            ax.plot(er["pos"][er["valid"], 0], er["pos"][er["valid"], 1], "-", c="#1d2330", lw=1.4, label="SD route as seen")
        ax.plot([0], [0], marker="^", c="#1d2330", ms=7, ls="none")
        ax.set_aspect("equal")
        ax.set_xlim(-30, 30)
        ax.set_ylim(-20, 50)
        ax.set_title(f"{name}  {txt}", fontsize=10)
        ax.grid(alpha=0.2)
    np.atleast_1d(axs)[0].legend(fontsize=7, loc="lower left")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
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
    for case, i in pick.items():
        ds = side["ds"]
        tab = {m_: ds.table(i, m_) for m_ in ("exact", "error", "wrong")}
        tab = {k: dict(pos=np.asarray(v["pos"]), valid=np.asarray(v["valid"], bool)) for k, v in tab.items()}
        shown = "wrong" if case == "wrong" else "error"
        mrow = "mw" if case == "wrong" else "m"
        rows = [("z (no map)", fl["z"][i], f"F1 road {f1['z'][i]:.3f}"), (f"m (SD route, mode {shown})", fl[mrow][i], f"F1 road {f1[mrow][i]:.3f}")]
        title = {"typical": "typical fork frame (median gain)", "gain": "largest gain of m over z", "loss": "largest loss of m against z", "wrong": "largest drop under the wrong map"}[case]
        fn = f"e389_frame_{case}.png"
        draw(os.path.join(a.out, fn), f"{title}: frame {i}, scene {int(side['scene'][i])}", side, i, fl, dict(exact=tab["exact"], shown=tab[shown]), rows)
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
        assert a.out, "--out <dir>"
        main(a)
