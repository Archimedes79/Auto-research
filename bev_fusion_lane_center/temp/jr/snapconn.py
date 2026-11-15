"""connectors whose ends are SNAPPED to the delivered lane pieces (scratch variant of qends_eval.add_connectors, E311 idea): the keypoints decide WHICH entry joins which exit, the geometry of the connector starts at the end of the delivered piece that is wired to the entry (nearest piece end within WIRE_POS and WIRE_ANG) and ends at the start of the piece wired to the exit;
an end with no piece in reach keeps its keypoint. Same signature and return value as `qends_eval.add_connectors`."""
import numpy as np

import qends as QEND
import qends_eval as QEV


def add_connectors_snap(lines, edges, nodes, kinds, links, long_mask, pairs):
    keep = []
    for i, k in enumerate(kinds):
        if k == "link" and bool(long_mask[links[i][0]]):
            P = np.asarray(lines[i], float)
            if any(np.linalg.norm(P[0] - pr[0]) <= QEV.COVER_POS and np.linalg.norm(P[-1] - pr[2]) <= QEV.COVER_POS for pr in pairs):
                continue
        keep.append(i)
    new = {old: j for j, old in enumerate(keep)}
    L2 = [np.asarray(lines[i], float) for i in keep]
    N2 = [list(nodes[i]) for i in keep]
    E2 = [(new[a], new[b]) for a, b in edges if a in new and b in new]
    ends = [(P[-1], QEV._tan(P, True), P[0], QEV._tan(P, False)) for P in L2]
    wired = 0
    for n, pr in enumerate(pairs):
        pa, ta, pb, tb = (np.asarray(x, float) for x in pr[:4])
        w_in = w_out = None
        d_in = d_out = 1e9
        for i, (le, te, fs, ts) in enumerate(ends):
            d = np.linalg.norm(le - pa)
            if d <= QEV.WIRE_POS and QEV._ang(te, ta) <= QEV.WIRE_ANG and d < d_in:
                w_in, d_in = i, d
            d = np.linalg.norm(fs - pb)
            if d <= QEV.WIRE_POS and QEV._ang(ts, tb) <= QEV.WIRE_ANG and d < d_out:
                w_out, d_out = i, d
        if w_in is not None:
            pa, ta = ends[w_in][0], ends[w_in][1]
        if w_out is not None:
            pb, tb = ends[w_out][2], ends[w_out][3]
        m = len(L2)
        if len(pr) > 4:
            L2.append(QEND.curve_s(pa, ta, pb, tb, float(pr[4]), float(pr[5])))
        else:
            L2.append(QEND.curve(pa, ta, pb, tb, 1.0, 0.25))
        N2.append([10000 + n, 20000 + n])
        if w_in is not None:
            E2.append((w_in, m))
        if w_out is not None:
            E2.append((m, w_out))
        wired += int(w_in is not None and w_out is not None)
    return dict(pts=L2, edges=E2, nodes=N2), len(pairs), wired
