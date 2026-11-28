"""The entry / exit clusters of the junction ends of the lane GT and the Hermite curve of a pair (what stays of the junction end table of E284, whose head is in the tag v1).

    reps, ids = cluster(items)     # items = [(point (2,), unit heading (2,))]: ends within 1.5 m and 40 degrees of a cluster's first member are one end; read by evaluation/jmaneuver.py
    curve(pa, ta, pb, tb)          # the cubic Hermite curve of a pair (tangent length = k x chord); curve_s: the same with handle scales in units of the chord

Nothing here uses the image, the map or the network. Run the self-test: python -P -m lane_center.net.prep.qends
"""
import numpy as np



def cluster(items, tol_pos=1.5, tol_ang=40.0):
    """greedy clustering of (position, tangent) items -> (list of representatives, cluster id of every item)"""
    reps, ids = [], []
    for pos, tan in items:
        for j, (rp, rt) in enumerate(reps):
            if np.linalg.norm(pos - rp) <= tol_pos and np.degrees(np.arccos(np.clip(tan @ rt, -1, 1))) <= tol_ang:
                ids.append(j)
                break
        else:
            reps.append((pos, tan))
            ids.append(len(reps) - 1)
    return reps, ids


def curve(a, ta, b, tb, k=1.0, step=0.25):
    """cubic Hermite curve from a (heading ta) to b (heading tb), handles k x chord (the shape of the E283 oracle: k = 1.0), points about `step` metres apart"""
    L = float(np.linalg.norm(b - a))
    n = max(8, int(1.6 * L / step))
    s = np.linspace(0, 1, n)[:, None]
    h00, h10, h01, h11 = 2 * s ** 3 - 3 * s ** 2 + 1, s ** 3 - 2 * s ** 2 + s, -2 * s ** 3 + 3 * s ** 2, s ** 3 - s ** 2
    return h00 * a + h10 * (k * L * ta) + h01 * b + h11 * (k * L * tb)


def curve_s(a, ta, b, tb, sa=1.0 / 3.0, sb=1.0 / 3.0, step=0.25):
    """the cubic Bezier / Hermite curve from a (heading ta) to b (heading tb) with the handle scales sa, sb in units of the chord (the control points a + sa x chord x ta and b - sb x chord x tb; sa = sb = 1/3 is `curve` with k = 1;
    the scales of `qtargets.fit_long_scales` / E303 are these), points about `step` metres apart"""
    L = float(np.linalg.norm(b - a))
    n = max(8, int(1.6 * L / step))
    s = np.linspace(0, 1, n)[:, None]
    q1, q2 = a + sa * L * ta, b - sb * L * tb
    return (1 - s) ** 3 * a + 3 * (1 - s) ** 2 * s * q1 + 3 * (1 - s) * s ** 2 * q2 + s ** 3 * b


def _selftest():
    up = np.array([0.0, 1.0])
    reps, ids = cluster([(np.array([0.0, 0.0]), up), (np.array([1.0, 0.0]), up), (np.array([5.0, 0.0]), up), (np.array([0.5, 0.0]), np.array([1.0, 0.0]))])
    assert ids == [0, 0, 1, 2] and len(reps) == 3, (ids, len(reps))        # 1 m apart: one end; 5 m apart or 90 degrees off: its own
    c = curve(np.array([0.0, 0.0]), np.array([0.0, 1.0]), np.array([4.0, 4.0]), np.array([1.0, 0.0]))
    c2 = curve_s(np.array([0.0, 0.0]), np.array([0.0, 1.0]), np.array([4.0, 4.0]), np.array([1.0, 0.0]))
    assert np.allclose(c, c2), 'curve_s with the scales 1/3 is curve with k = 1'
    assert np.allclose(c[0], [0, 0]) and np.allclose(c[-1], [4, 4]) and len(c) >= 8
    print("qends selftest ok")


if __name__ == "__main__":
    _selftest()
