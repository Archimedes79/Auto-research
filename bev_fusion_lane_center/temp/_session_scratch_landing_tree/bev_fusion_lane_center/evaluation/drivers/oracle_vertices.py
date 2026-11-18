"""Vertex-count oracles on the ground-truth lanes of the TEST rows: Douglas-Peucker, greedy longest chord, and the Imai-Iri minimum (shortest path in the graph of valid chords).
A chord (i, j) is valid if all samples between lie within eps of the segment (Hausdorff, vertices only). Samples every 0.5 m, clipped to +-48 m. Usage (gcar): oracle_vertices.py [n_frames]"""
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import graph_eval as GE  # noqa: E402
import p8_lib as L  # noqa: E402

JMAX = 90  # longest chord 45 m


def seg_dist(P, a, b):
    ab = b - a
    t = np.clip(((P - a) @ ab) / max(float(ab @ ab), 1e-12), 0, 1)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1)


def counts(q, eps):
    n = len(q)
    valid = [[] for _ in range(n)]
    for i in range(n - 1):
        for j in range(i + 1, min(n, i + JMAX + 1)):
            if j == i + 1 or seg_dist(q[i + 1 : j], q[i], q[j]).max() <= eps:
                valid[i].append(j)
    # Imai-Iri: shortest path over valid chords (BFS by levels)
    dist = np.full(n, 10**9)
    dist[0] = 0
    for i in range(n):
        if dist[i] >= 10**9:
            continue
        for j in valid[i]:
            dist[j] = min(dist[j], dist[i] + 1)
    ii = dist[-1] + 1  # vertices
    # greedy longest chord from the start (largest valid j)
    i, g = 0, 1
    while i < n - 1:
        i = max(valid[i]) if valid[i] else i + 1
        g += 1
    return int(ii), int(g)


def dp(p, eps):
    keep = np.zeros(len(p), bool)
    keep[0] = keep[-1] = True
    st = [(0, len(p) - 1)]
    while st:
        a, b = st.pop()
        if b <= a + 1:
            continue
        d = seg_dist(p[a + 1 : b], p[a], p[b])
        i = int(np.argmax(d))
        if d[i] > eps:
            keep[a + 1 + i] = True
            st += [(a, a + 1 + i), (a + 1 + i, b)]
    return int(keep.sum())


def main():
    nfr = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    z = pickle.load(open(os.path.expanduser("~/exp_gcar_b123/p10/tok_e94_a_test_clean.pkl"), "rb"))
    gt = L.cache("gt", "val")
    items = z["items"][:: max(len(z["items"]) // nfr, 1)][:nfr]
    for eps in (0.3, 0.5):
        tot = 0.0
        c = dict(dp=0, ii=0, greedy=0)
        for it in items:
            for p in gt[int(it["row"])]["pts"]:
                q = GE.resample(np.asarray(p, float), 0.5)
                if len(q) < 3:
                    continue
                q = q[(np.abs(q) < 48).all(1)]
                if len(q) < 3:
                    continue
                tot += np.linalg.norm(np.diff(q, axis=0), axis=1).sum()
                ii, g = counts(q, eps)
                c["dp"] += dp(q, eps)
                c["ii"] += ii
                c["greedy"] += g
        print(f"eps {eps} m, {len(items)} frames, {tot:.0f} m of lane: " + ", ".join(f"{k} {100 * v / tot:.1f} per 100 m" for k, v in c.items()))


if __name__ == "__main__":
    main()
