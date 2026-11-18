"""Test of `qgraph_run_eval.verdict275` (the written rule of the E275 card: close-node suppression against the same models read out as before) with fake scoring cards in a temporary folder:
KEEP / TIE / CUT / INCONCLUSIVE; the bands (gains that lie between 0.02 and 0.03); every guard (assignment gain and exact map on the MEAN of the seeds, F1@1.0 and lanes found per seed, the
pairing of the seeds, a lanes-found gain is no violation); the cases where the card and the E274 rule differ (NO topo guard in the junction call); the checks of what is judged (two seeds,
873 frames, the read-out of the cards).

    wsl bash tools/env/_py14.sh test_verdict275.py
"""
import contextlib
import io
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import qgraph_run_eval as QR  # noqa: E402


def card(seed, topo=0.60, f1=0.83, junc=0.12, lanes=0.64, err=0.07, ex=-0.005, frames=873, readout=None):
    c = dict(seed=seed, frames=frames, p=dict(graph=dict(topo_F1=topo, **{"F1@1.0": f1}, junction_F1=junc, lanes_found=lanes, vert100_dp=30.0, dup_frac=0.2),
                                              nodes=dict(node_R=0.83, node_P=0.87, link_long_P=0.18, link_long_R=0.49)), diff=dict(error=dict(mean=err), exact=dict(mean=ex)))
    if readout:
        c["readout"] = dict(spec=readout)
    return c


def var(seed, readout="close1.5_t0.5", **kw):
    return card(seed, readout=readout, **kw)


def put(root, exp, run, c):
    d = os.path.join(root, "runs", exp, run)
    os.makedirs(d, exist_ok=True)
    json.dump(c, open(os.path.join(d, "card.json"), "w"))


def run(root, var_cards, ctl_cards, **kw):
    names_v = [f"v{k}.close1.5_t0.5" for k in range(len(var_cards))]
    names_c = [f"c{k}" for k in range(len(ctl_cards))]
    for n, c in zip(names_v, var_cards):
        put(root, "ev", n, c)
    for n, c in zip(names_c, ctl_cards):
        put(root, "ec", n, c)
    with contextlib.redirect_stdout(io.StringIO()):
        return QR.verdict275(names_v, names_c, "ev", "ec", root=root, **kw)


def refused(root, var_cards, ctl_cards, **kw):
    try:
        run(root, var_cards, ctl_cards, **kw)
    except AssertionError:
        return True
    return False


def main():
    ctl = [card(0), card(1)]
    good = dict(topo=0.63, junc=0.26, lanes=0.62)                      # the in-sample numbers of the E274 sweep: topo +0.03, junction +0.14, lanes found -0.02
    with tempfile.TemporaryDirectory() as root:
        o = run(root, [var(0, **good), var(1, **good)], ctl)
        assert o["verdict"] == "KEEP" and all(o["guards"].values()), o
        assert "topo_F1_not_below_control_by_0_01" not in o["guards"]                                                       # the E274 guard is not in the E275 card
        # sides
        assert run(root, [var(0, topo=0.61, junc=0.13), var(1, topo=0.59, junc=0.11)], ctl)["verdict"] == "TIE"             # both inside their bands (the 0.59 seed is -0.01 on topo)
        assert run(root, [var(0, topo=0.61, junc=0.26), var(1, topo=0.61, junc=0.26)], ctl)["verdict"].startswith("INCONCLUSIVE")     # junction up, topo inside
        assert run(root, [var(0, topo=0.63, junc=0.13), var(1, topo=0.63, junc=0.13)], ctl)["verdict"].startswith("INCONCLUSIVE")     # topo up, junction inside
        assert run(root, [var(0, topo=0.63, junc=0.26), var(1, topo=0.61, junc=0.26)], ctl)["verdict"].startswith("INCONCLUSIVE")     # the seeds disagree on topo (up and inside)
        assert run(root, [var(0, topo=0.63, junc=0.26), var(1, topo=0.55, junc=0.26)], ctl)["verdict"] == "CUT"                # topo clearly down on one seed
        assert run(root, [var(0, topo=0.63, junc=0.26), var(1, topo=0.63, junc=0.08)], ctl)["verdict"] == "CUT"                # junction clearly down on one seed
        assert run(root, [var(0, topo=0.63, junc=0.26), var(1, topo=0.61, junc=0.13)], ctl)["verdict"].startswith("INCONCLUSIVE")     # mixed topo, junction (up, inside)
        assert run(root, [var(0, topo=0.63, junc=0.13), var(1, topo=0.61, junc=0.11)], ctl)["verdict"].startswith("INCONCLUSIVE")     # topo (up, inside) = mixed, junction inside: no TIE
        # the bands: gains between 0.02 and 0.03
        o = run(root, [var(0, topo=0.625, junc=0.145), var(1, topo=0.625, junc=0.145)], ctl)                                 # topo +0.025 (above 0.02), junction +0.025 (inside 0.03)
        assert o["topo_F1"]["side"] == "up" and o["junction_F1"]["side"] == "inside" and o["verdict"].startswith("INCONCLUSIVE")
        assert run(root, [var(0, topo=0.625, junc=0.155), var(1, topo=0.625, junc=0.155)], ctl)["verdict"] == "KEEP"          # topo +0.025 and junction +0.035: both above their own bands
        # NO topo guard: the card gives (inside, up) / (inside, inside) for a topo gain of -0.015, the E274 rule would say CUT
        o = run(root, [var(0, topo=0.585, junc=0.26), var(1, topo=0.585, junc=0.26)], ctl)
        assert o["verdict"].startswith("INCONCLUSIVE") and o["topo_F1"]["side"] == "inside" and o["junction_F1"]["side"] == "up", o
        assert run(root, [var(0, topo=0.585, junc=0.13), var(1, topo=0.585, junc=0.13)], ctl)["verdict"] == "TIE"
        o = run(root, [var(0, topo=0.63, junc=0.26), var(1, topo=0.585, junc=0.26)], ctl)                                    # topo +0.03 and -0.015: mixed
        assert o["verdict"].startswith("INCONCLUSIVE") and o["topo_F1"]["side"] == "mixed", o
        # guards: the assignment numbers are the MEAN of the two seeds, F1@1.0 and lanes found are per seed
        assert run(root, [var(0, **good, ex=-0.02), var(1, **good, ex=0.01)], ctl)["verdict"] == "KEEP"                     # exact map: seed 0 below -0.01, the mean -0.005 holds
        assert run(root, [var(0, **good, ex=-0.05), var(1, **good, ex=-0.05)], ctl)["verdict"] == "CUT"                     # the exact map gets worse
        assert run(root, [var(0, **good, err=0.02), var(1, **good, err=0.08)], ctl)["verdict"] == "KEEP"                    # map-error gain: seed 0 below +0.03, the mean +0.05 holds
        assert run(root, [var(0, **good, err=0.01), var(1, **good, err=0.01)], ctl)["verdict"] == "CUT"                     # the assignment gain is lost
        assert run(root, [var(0, **good, f1=0.81), var(1, **good, f1=0.86)], ctl)["verdict"] == "CUT"                       # F1@1.0 -0.02 on one seed (the mean would hold)
        assert run(root, [var(0, **good, f1=0.825), var(1, **good, f1=0.825)], ctl)["verdict"] == "KEEP"                    # -0.005 is inside the guard
        low = dict(topo=0.63, junc=0.26)
        o = run(root, [var(0, **low, lanes=0.58), var(1, **low, lanes=0.58)], ctl)                                          # lanes found -0.06
        assert o["verdict"] == "CUT" and o["guards"]["lanes_found_not_below_control_by_0.03"] is False, o
        assert run(root, [var(0, **low, lanes=0.60), var(1, **low, lanes=0.68)], ctl)["verdict"] == "CUT"                   # -0.04 on one seed, +0.04 on the other (the mean would hold)
        assert run(root, [var(0, **low, lanes=0.62), var(1, **low, lanes=0.62)], ctl)["verdict"] == "KEEP"                  # -0.02 is inside the guard
        assert run(root, [var(0, **low, lanes=0.70), var(1, **low, lanes=0.70)], ctl)["verdict"] == "KEEP"                  # lanes found UP by 0.06 is no violation
        # the seeds are paired with their own control: control lanes found 0.64 (seed 0) and 0.70 (seed 1)
        ctl2 = [card(0, lanes=0.64), card(1, lanes=0.70)]
        assert run(root, [var(0, **low, lanes=0.62), var(1, **low, lanes=0.68)], ctl2)["verdict"] == "KEEP"                 # -0.02 and -0.02
        assert run(root, [var(0, **low, lanes=0.70), var(1, **low, lanes=0.64)], ctl2)["verdict"] == "CUT"                  # +0.06 and -0.06
        # what is judged
        assert refused(root, [var(0, **good)], [card(0)])                                                                   # one seed only
        assert refused(root, [var(1, **good), var(0, **good)], [card(1), card(0)])                                          # seeds in the wrong order
        assert refused(root, [var(0, **good), var(2, **good)], [card(0), card(2)])                                          # a third seed instead of seed 1
        assert refused(root, [var(0, **good), var(1, **good, frames=12)], ctl)                                              # a card of 12 frames
        assert refused(root, [var(0, readout="plain_t0.5", **good), var(1, readout="plain_t0.5", **good)], ctl)             # not the step
        assert refused(root, [var(0, readout="close1.5_t0.4", **good), var(1, readout="close1.5_t0.4", **good)], ctl)        # the 0.4 variant is reported, not judged
        assert refused(root, [card(0, **good), card(1, **good)], ctl)                                                       # no read-out key at all
        assert refused(root, [var(0, **good), var(1, **good)], [card(0, readout="close1.5_t0.5"), card(1, readout="close1.5_t0.5")])   # a control that is itself a step
        assert not refused(root, [var(0, **good), var(1, **good)], [card(0, readout="plain_t0.5"), card(1, readout="plain_t0.5")])    # the plain read-out as control is fine
        # seeds of variant and control must match
        assert refused(root, [var(0, **good), var(1, **good)], [card(0), card(2)])
    print("verdict275: KEEP / TIE / CUT / INCONCLUSIVE, the bands between 0.02 and 0.03, no topo guard, the guards (mean of the seeds for the assignment, per seed for F1@1.0 and lanes found, pairing, "
          "a lanes-found gain is fine), the checks of seeds / frames / read-out: ok")


if __name__ == "__main__":
    main()
