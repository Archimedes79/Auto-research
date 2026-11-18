"""E357 throw-away probe (5 VAL frames, CPU, 2 threads): what the lane GT pieces look like, node / link counts on the 2.5 m lattice, the two candidate rules for a junction link."""
import time

import numpy as np
import torch

torch.set_num_threads(2)
import qgraph_eval as QE  # noqa: E402
import qtargets as QT  # noqa: E402

frames = QE.val_frames()[::200]
QT.N, QT.RES = 40, 2.5
for f in frames:
    g = f.lane()
    t0 = time.time()
    nt = QT.node_table(g)
    t1 = time.time()
    sc = QT.fit_long_scales(nt)
    t2 = time.time()
    kinds = sorted({p["kind"] for p in g["pieces"]})
    mixed = sum(len({s["kind"] for s in p["segments"]}) > 1 for p in g["pieces"])
    gk = np.concatenate([np.zeros(0, int)] + [np.array([any(s["kind"] == "connector" and s["i0"] <= i <= s["i1"] for s in p["segments"]) for i in range(len(p["pts"]))], int) for p in g["pieces"]])
    conn = gk[nt["gid"]].astype(bool)
    pk = np.array([p["kind"] == "connector" for p in g["pieces"]])
    src, dst, lg = nt["src"], nt["dst"], nt["long"]
    path_rule = np.array([gk[p].mean() > 0.5 for p in nt["path"]], bool)
    pair_rule = conn[src] | conn[dst] | (lg & (nt["piece"][src] != nt["piece"][dst]))
    piece_rule = pk[nt["piece"][src]] | pk[nt["piece"][dst]] | (lg & (nt["piece"][src] != nt["piece"][dst]))
    print(f"k {f.k}: pieces {len(g['pieces'])} kinds {kinds} mixed-segment pieces {mixed}; nodes {len(nt['pos'])} links {len(src)} long {int(lg.sum())}; node_table {t1 - t0:.3f} s fit {t2 - t1:.3f} s")
    print(f"   junction links: path rule {int(path_rule.sum())}, pair rule (node on connector point) {int(pair_rule.sum())} agree {int((path_rule == pair_rule).sum())}/{len(src)}, piece-kind rule {int(piece_rule.sum())} agree {int((path_rule == piece_rule).sum())}/{len(src)}")
    print(f"   chord of links: min {nt['dist'].min():.2f} share < 1 m {(nt['dist'] < 1).mean():.3f}; long across pieces {int((lg & (nt['piece'][src] != nt['piece'][dst])).sum())}, long inside a piece {int((lg & (nt['piece'][src] == nt['piece'][dst])).sum())}; nan scales among long {int(np.isnan(sc[lg, 0]).sum())}")
    print("   piece keys:", sorted(g["pieces"][0].keys()))
    f._lane = None
