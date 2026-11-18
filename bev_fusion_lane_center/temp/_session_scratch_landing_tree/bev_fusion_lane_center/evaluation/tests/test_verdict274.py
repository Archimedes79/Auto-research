"""Test of `qgraph_run_eval.verdict274` (the written rule of the E274 card) with fake scoring cards in a temporary folder: KEEP / TIE / CUT / INCONCLUSIVE, every guard, the topo guard of arm p against r,
the seed check. The cards hold only the fields the function reads.

    wsl bash tools/env/_py14.sh test_verdict274.py
"""
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import qgraph_run_eval as QR  # noqa: E402


def card(seed, topo=0.60, f1=0.83, junc=0.12, err=0.07, ex=-0.005):
    return dict(seed=seed, p=dict(graph=dict(topo_F1=topo, **{"F1@1.0": f1}, junction_F1=junc, vert100_dp=30.0, dup_frac=0.2),
                                  nodes=dict(node_R=0.83, node_P=0.87, link_long_P=0.18, link_long_R=0.49)), diff=dict(error=dict(mean=err), exact=dict(mean=ex)))


def put(root, exp, run, c):
    d = os.path.join(root, "runs", exp, run)
    os.makedirs(d, exist_ok=True)
    json.dump(c, open(os.path.join(d, "card.json"), "w"))


def run(root, var_cards, ctl_cards, **kw):
    for k, c in enumerate(var_cards):
        put(root, "ev", f"v{k}", c)
    for k, c in enumerate(ctl_cards):
        put(root, "ec", f"c{k}", c)
    import io
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        return QR.verdict274(["v0", "v1"], ["c0", "c1"], "ev", "ec", root=root, **kw)


def main():
    ctl = [card(0), card(1)]
    with tempfile.TemporaryDirectory() as root:
        assert run(root, [card(0, topo=0.66), card(1, topo=0.65)], ctl)["verdict"] == "KEEP"
        assert run(root, [card(0, topo=0.61), card(1, topo=0.59)], ctl)["verdict"] == "TIE"
        assert run(root, [card(0, topo=0.55), card(1, topo=0.55)], ctl)["verdict"] == "CUT"
        assert run(root, [card(0, topo=0.66), card(1, topo=0.61)], ctl)["verdict"].startswith("INCONCLUSIVE")
        assert run(root, [card(0, topo=0.66), card(1, topo=0.55)], ctl)["verdict"] == "CUT"          # one seed clearly down: CUT (the card: a gain below -band is a CUT)
        assert run(root, [card(0, topo=0.55), card(1, topo=0.61)], ctl)["verdict"] == "CUT"          # down and inside: CUT
        # guards break a KEEP
        up = dict(topo=0.66)
        assert run(root, [card(0, **up, err=0.01), card(1, **up, err=0.01)], ctl)["verdict"] == "CUT"        # the assignment gain is lost
        assert run(root, [card(0, **up, ex=-0.05), card(1, **up, ex=-0.05)], ctl)["verdict"] == "CUT"        # the exact map gets worse
        assert run(root, [card(0, **up, f1=0.80), card(1, **up, f1=0.80)], ctl)["verdict"] == "CUT"          # F1@1.0 falls by more than 0.01
        assert run(root, [card(0, **up, f1=0.825), card(1, **up, f1=0.825)], ctl)["verdict"] == "KEEP"       # 0.005 is inside the guard
        # arm p against r: junction F1 is the headline, band 0.03, the topo guard of the card
        rr = [card(0, topo=0.65, junc=0.15), card(1, topo=0.65, junc=0.15)]
        good = [card(0, topo=0.65, junc=0.22), card(1, topo=0.65, junc=0.21)]
        bad = [card(0, topo=0.57, junc=0.22), card(1, topo=0.57, junc=0.22)]                      # junction F1 +0.07 but topo F1 -0.08: the failing input of the critic pass
        assert run(root, good, rr, metric="junction_F1", band=0.03)["verdict"] == "KEEP"
        o = run(root, bad, rr, metric="junction_F1", band=0.03)
        assert o["verdict"] == "CUT" and o["guards"]["topo_F1_not_below_control_by_0_01"] is False, o
        # the seeds of variant and control must match
        try:
            run(root, [card(0, topo=0.66), card(2, topo=0.65)], ctl)
            raise SystemExit("seed mismatch not detected")
        except AssertionError:
            pass
        # the system pass is reported, not judged
        o = run(root, [card(0, topo=0.73, f1=0.87), card(1, topo=0.73, f1=0.87)], ctl)
        assert o["variant_numbers"]["system_pass_vs_d"] is True and o["verdict"] == "KEEP"
    print("verdict274: KEEP / TIE / CUT / INCONCLUSIVE, the four guards (assignment gain, exact map, F1@1.0, topo F1 for arm p), the seed check and the system pass: ok")


if __name__ == "__main__":
    main()
