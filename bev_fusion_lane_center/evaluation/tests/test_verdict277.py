"""Test of `qgraph_run_eval.verdict277` (the written rule of the E277 card: the head trained with the soft close-node step against the baseline read out as before AND against the baseline with the
hard step) with fake scoring cards in a temporary folder: KEEP (replaces the hard step) / TIE (helps but trails the hard step) / CUT / INCONCLUSIVE; the trail tolerance (0.01) on EACH seed; the checks of what
is judged (seeds, frames, read-out of the soft and hard cards).

    wsl bash tools/env/_py14.sh test_verdict277.py
"""
import contextlib
import io
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import qgraph_run_eval as QR  # noqa: E402
import test_verdict275 as T5  # noqa: E402  (the fake-card helpers)


def run(root, soft, ctl, hard, runs=None, hard_runs=None, **kw):
    names_s = [f"s{k}.plain_t0.5" for k in range(len(soft))]
    names_c = [f"c{k}" for k in range(len(ctl))]
    names_h = [f"h{k}.close1.5_t0.5" for k in range(len(hard))]
    for n, c in zip(names_s, soft):
        T5.put(root, "es", n, c)
    for k, (n, c) in enumerate(zip(names_c, ctl)):
        T5.put(root, "ec", n, dict(c, run=runs[k] if runs else f"m{k}"))
    for k, (n, c) in enumerate(zip(names_h, hard)):
        T5.put(root, "eh", n, dict(c, run=hard_runs[k] if hard_runs else f"m{k}"))
    with contextlib.redirect_stdout(io.StringIO()):
        return QR.verdict277(names_s, names_c, names_h, "es", "ec", "eh", root=root, **kw)


def refused(root, soft, ctl, hard, **kw):
    try:
        run(root, soft, ctl, hard, **kw)
    except AssertionError:
        return True
    return False


def main():
    ctl = [T5.card(0), T5.card(1)]
    hard = [T5.var(0, topo=0.626, junc=0.27, lanes=0.627, readout="close1.5_t0.5"), T5.var(1, topo=0.625, junc=0.25, lanes=0.614, readout="close1.5_t0.5")]

    def soft(topo0, topo1, junc=0.26, lanes=0.62, **kw):
        return [T5.var(0, readout="plain_t0.5", topo=topo0, junc=junc, lanes=lanes, **kw), T5.var(1, readout="plain_t0.5", topo=topo1, junc=junc, lanes=lanes, **kw)]

    with tempfile.TemporaryDirectory() as root:
        hard_b = [T5.var(0, topo=0.650, junc=0.27, readout="close1.5_t0.5"), T5.var(1, topo=0.650, junc=0.25, readout="close1.5_t0.5")]       # a stronger hard step (fake), for the trail cases
        hard_c = [T5.var(0, topo=0.640, junc=0.27, readout="close1.5_t0.5"), T5.var(1, topo=0.640, junc=0.25, readout="close1.5_t0.5")]
        # the soft step reaches the hard step (trail 0.004 / 0.003; gain over the baseline +0.022): KEEP, it replaces the hard read-out
        r = run(root, soft(0.622, 0.622), ctl, hard)
        assert r["verdict"].startswith("KEEP") and r["verdict275"] == "KEEP", r
        assert all(abs(t - e) < 1e-9 for t, e in zip(r["topo_F1_trail_behind_hard_per_seed"], [0.004, 0.003])), r
        # it beats the baseline on both seeds but trails the hard step by 0.025 on ONE seed (0.010 on the other): TIE, the hard step stays
        r = run(root, soft(0.625, 0.640), ctl, hard_b)
        assert r["verdict275"] == "KEEP" and r["verdict"].startswith("TIE"), r
        # exactly the tolerance (0.01) is still a replacement; just above it is not
        r = run(root, soft(0.630, 0.630), ctl, hard_c)
        assert r["verdict"].startswith("KEEP"), r
        r = run(root, soft(0.6299, 0.630), ctl, hard_c)
        assert r["verdict"].startswith("TIE"), r
        # no gain over the baseline (topo +0.003): the verdict of verdict275 stays (TIE inside the bands is only reached with a small junction gain too; here the junction gain is large: INCONCLUSIVE)
        r = run(root, soft(0.603, 0.601), ctl, hard)
        assert r["verdict"] == r["verdict275"] and not r["verdict"].startswith("KEEP"), r
        # no effect over the baseline in BOTH headlines (verdict275 says TIE): the card says INCONCLUSIVE, and the word TIE stays for "helps but trails the hard step" (critic pass 2026-11-08)
        r = run(root, soft(0.605, 0.605, junc=0.13), ctl, hard)
        assert r["verdict275"] == "TIE" and r["verdict"].startswith("INCONCLUSIVE"), r
        # a broken assignment guard is a CUT whatever the topo numbers
        r = run(root, soft(0.630, 0.630, ex=-0.02), ctl, hard)
        assert r["verdict"] == "CUT", r
        # the soft step beats the hard one: trail negative, KEEP
        r = run(root, soft(0.640, 0.640), ctl, hard)
        assert r["verdict"].startswith("KEEP") and all(t < 0 for t in r["topo_F1_trail_behind_hard_per_seed"]), r
        # what is judged: seeds in the order 0, 1; 873 frames in every card incl. the hard cards; the read-out of the hard cards
        assert refused(root, soft(0.62, 0.62)[::-1], ctl, hard), "seed order of the soft cards"
        assert refused(root, soft(0.62, 0.62), ctl, hard[::-1]), "seed order of the hard cards"
        assert refused(root, soft(0.62, 0.62), ctl, [T5.var(0, topo=0.626, junc=0.27, readout="close1.5_t0.5", frames=150), hard[1]]), "frames of a hard card"
        assert refused(root, soft(0.62, 0.62), ctl, [T5.var(0, topo=0.626, junc=0.27, readout="plain_t0.5"), hard[1]]), "read-out of a hard card"
        assert refused(root, soft(0.62, 0.62), ctl, hard, runs=["m0", "m1"], hard_runs=["m0", "other"]), "the hard-step cards must be the control's models"
        assert refused(root, [T5.var(0, readout="close1.5_t0.5", topo=0.62, junc=0.26), soft(0.62, 0.62)[1]], ctl, hard), "read-out of a soft card"
    print("verdict277 tests ok")


if __name__ == "__main__":
    main()
