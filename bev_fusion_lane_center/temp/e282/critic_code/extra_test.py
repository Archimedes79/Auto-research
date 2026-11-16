"""proposed additions to test_zones.py (run from work/training/grid so that the mutated copy is imported)"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))

import numpy as np, torch
from test_zones import *   # noqa

def run():
    g = toy(labelled=False)
    b = batch_of([g, g])
    zone, gz, kind = b["zone"].long(), b["zone_gt"], b["zone_kind"]
    QG.CFG.update(node_loss="zone", zone_bias=0.0)
    try:
        # winners for a fixed logit pattern: away from the GT cell in every zone
        e = -5.0 * gz
        win, info = QG.zone_targets(e, b, 0.0)
        nz = int(win.sum())
        # (1) l_e is normalised by the number of zones
        no = fake_no(b, e=e.clone().requires_grad_())
        l, inf = QG.node_loss(no, b)
        l_e_hand = QG.focal_bce(e, win, norm=float(nz))
        assert abs(inf["e"] - float(l_e_hand)) < 1e-5, ("normalisation", inf["e"], float(l_e_hand))
        # (2) offset term only on the winners: error 0.3 on the off-GT winners -> 0.6 * k / n_zones ; error on a non-winner zone cell only -> 0
        offw = (win > 0.5) & (gz < 0.5)
        k = int(offw.sum())
        assert k > 0
        no = fake_no(b, e=e.clone().requires_grad_())
        no["off"] = (b["offset_dense"] + 0.3 * offw[:, None].float()).clone().requires_grad_()
        no["dir_raw"] = b["dirv_dense"].clone().requires_grad_()
        _, inf = QG.node_loss(no, b)
        assert abs(inf["off"] - 0.6 * k / nz) < 1e-5, ("offset on winners", inf["off"], 0.6 * k / nz)
        nonwin = (zone >= 0) & (win < 0.5)
        no = fake_no(b, e=e.clone().requires_grad_())
        no["off"] = (b["offset_dense"] + 0.3 * nonwin[:, None].float()).clone().requires_grad_()
        no["dir_raw"] = b["dirv_dense"].clone().requires_grad_()
        _, inf = QG.node_loss(no, b)
        assert inf["off"] < 1e-6, ("offset must not be trained on non-winner zone cells", inf["off"])
        # (3) win_off_gt is the share over FREE zones only
        free_ids = [(i, z) for i in range(2) for z in range(int(zone[i].max()) + 1) if int(kind[i][zone[i] == z][0]) == 1]
        multi = [(i, z) for (i, z) in free_ids if int((zone[i] == z).sum()) >= 2]
        expect = len(multi) / len(free_ids)
        assert 0 < expect < 1
        assert abs(float(info["win_off_gt"]) - expect) < 1e-6, ("win_off_gt", float(info["win_off_gt"]), expect)
    finally:
        QG.CFG.update(node_loss="e273", zone_bias=0.05)
    print("extra tests ok")

run()
