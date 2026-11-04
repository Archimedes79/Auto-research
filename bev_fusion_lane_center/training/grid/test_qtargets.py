"""G0 of the lane head (E273, card section 3.4): the representation oracle on REAL lane GT frames. Nodes + links of `qtargets.node_table` are decoded back to curves and compared with the GT.

    wsl bash tools/env/_py14.sh test_qtargets.py [--n 200] [--marker 0]

Frames: a seeded sample of the stored lane GT (`data/autogt/lane/<signature>`, the signature of the autogt source in this checkout). Gates (card): GT lane points within 0.3 m of the decoded
curves >= 0.997 (1 m: 0.999); connector points within 1 m >= 0.97; decoded points within 0.3 m of the GT >= 0.97; frames with more than one node per cell: dropped normal candidates <= 1 % of the nodes;
every node has an offset inside its cell; links: no link longer than 100 m that is not a long link; every piece with a node has exactly one section start.
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, ".."))
sys.path.insert(0, os.path.join(ROOT, "training", "grid"))
import qtargets as QT  # noqa: E402


def frames(n, seed=0):
    from autogt import lane_gt
    toks = sorted(lane_gt.held(os.path.join(lane_gt.DATA, lane_gt.signature())))
    rng = np.random.RandomState(seed)
    sel = rng.choice(len(toks), min(n, len(toks)), replace=False)
    return [lane_gt.load(toks[i]) for i in sorted(sel)]


def run(n=200, marker=0.0):
    acc = dict(all=[], lane=[], connector=[], prec=[], lane_1m=[], connector_1m=[], prec_1m=[])
    st = dict(nodes=0, dropped=0, ev=0, links=0, long=0, ev_merged=0, cells2=0)
    bad = []
    for g in frames(n):
        nt = QT.node_table(g, marker_dist=marker)
        s = nt["stats"]
        for k in ("nodes", "dropped", "links", "long", "ev_merged"):
            st[k] += s[k]
        st["ev"] += s["events"]
        gt = QT.grid_targets(nt)
        assert gt["exists"].sum() == s["nodes"], "two nodes in one cell"
        assert np.abs(gt["offset"]).max() <= QT.RES / 2 + 1e-6, "an offset leaves its cell"
        # one section start per piece that has a node
        per = {}
        for m in range(s["nodes"]):
            per.setdefault(int(nt["piece"][m]), 0)
            per[int(nt["piece"][m])] += int(nt["sstart"][m])
        assert all(v == 1 for v in per.values()), per
        if len(nt["dist"]) and (nt["dist"][~nt["long"]] > QT.SHORT + 1e-6).any():
            bad.append("short link too long")
        c = QT.coverage(QT.decode(nt), g)
        for k in acc:
            acc[k].append(c[k])
    r = {k: float(np.mean(v)) for k, v in acc.items()}
    print(f"frames {len(acc['all'])}: nodes per frame {st['nodes'] / len(acc['all']):.0f}, links {st['links'] / len(acc['all']):.0f} (long {st['long'] / len(acc['all']):.1f}), events {st['ev'] / len(acc['all']):.0f}, "
          f"dropped candidates {st['dropped'] / max(st['nodes'], 1):.4f} of the nodes, merged event places {st['ev_merged']}")
    print("coverage (mean over frames):", {k: round(v, 4) for k, v in r.items()})
    return r, st, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--marker", type=float, default=0.0)
    a = ap.parse_args()
    r, st, bad = run(a.n, a.marker)
    assert not bad, bad
    assert r["lane"] >= 0.997 and r["lane_1m"] >= 0.999, r
    assert r["connector_1m"] >= 0.97, r
    assert r["prec"] >= 0.97, r
    assert st["dropped"] <= 0.01 * st["nodes"], st
    print("test_qtargets ok")


if __name__ == "__main__":
    main()
