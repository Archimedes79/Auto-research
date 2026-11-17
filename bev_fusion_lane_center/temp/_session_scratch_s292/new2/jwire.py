"""E325 / E322: the wiring of a connector end to a delivered piece end, as functions that can be replaced (step 1: the zero-parameter twin of the nearest-piece rule and the helper both the read-out and the
head's features use; the head itself, its data and its training are later steps of the card of E325).

    keep, ends = kept_ends(lines, kinds, links, long_mask, pairs, wire_pos)   # the pieces that survive the long-link cover drop of `qends_eval.add_connectors` and their ends (last point, heading there,
                                                                              # first point, heading there); the indices of a `wire_choice` index THIS list
    cands = candidates(ends, pos, tan, entry, wire_pos)                       # the places in reach of a connector end: [(distance, index, point, tangent)], nearest first, ties in the list order, at most MAX_CAND
    pick = twin_pick(ends, pos, tan, entry, wire_pos)                         # the zero-parameter twin: the argmax of softmax(-d^2 / (2 sigma^2)) over the candidates, the index into `ends` or None
    choice = twin_choice(ends, pairs, wire_pos)                               # one (entry index or None, exit index or None) per pair: `add_connectors(wire_choice=choice)` reproduces the rule bit for bit

The twin is not an approximation of the rule: the logit -d^2 / (2 sigma^2) falls strictly with the distance, so the argmax is the nearest candidate and a tie keeps the first in the list order, which is the
strict `d < d_in` of the rule. It exists so that the read-out path through `wire_choice` is verified bit for bit (it must reproduce the E312 cards) before any learned head is put into it; the learned head
replaces only the logits. The gate (reach 4 m, 45 degrees, at most 8 places) is an index choice and stays. Run the self-test: python jwire.py
"""
import numpy as np

WIRE_SIGMA = 1.2                                                                                  # m: the scale of the fixed dustbin logit of the card (-R^2 / (2 sigma^2) = -5.556 at R = 4 m)
MAX_CAND = 8


def kept_ends(lines, kinds, links, long_mask, pairs, wire_pos, cover_pos):
    """-> (keep, ends): `keep` = the indices of the pieces that stay (a LONG link whose first point lies within `cover_pos` of a pair's entry end and whose last point lies within `cover_pos` of its exit end is
    dropped: a connector covers it), `ends` = [(last point, heading at the end, first point, heading at the start)] of the kept pieces in order. The one place the read-out and the features of a head compute it."""
    import qends_eval as QE
    keep = []
    for i, k in enumerate(kinds):
        if k == "link" and bool(long_mask[links[i][0]]):
            P = np.asarray(lines[i], float)
            if any(np.linalg.norm(P[0] - pr[0]) <= cover_pos and np.linalg.norm(P[-1] - pr[2]) <= cover_pos for pr in pairs):
                continue
        keep.append(i)
    ends = []
    for i in keep:
        P = np.asarray(lines[i], float)
        ends.append((P[-1], QE._tan(P, True), P[0], QE._tan(P, False)))
    return keep, ends


def candidates(ends, pos, tan, entry, wire_pos, wire_ang=45.0):
    """the piece ends in reach of a connector end at `pos` with heading `tan` (entry = True: compare with the LAST point of the pieces, False: with the FIRST point): within `wire_pos` m and `wire_ang` degrees,
    [(distance, index into `ends`, point, tangent)] nearest first (a stable sort: a tie keeps the list order), cut at MAX_CAND"""
    import qends_eval as QE
    out = []
    pos, tan = np.asarray(pos, float), np.asarray(tan, float)
    for i, en in enumerate(ends):
        q, t = (en[0], en[1]) if entry else (en[2], en[3])
        d = float(np.linalg.norm(q - pos))
        if d <= wire_pos and QE._ang(t, tan) <= wire_ang:
            out.append((d, i, q, t))
    out.sort(key=lambda c: c[0])
    return out[:MAX_CAND]


def twin_pick(ends, pos, tan, entry, wire_pos, wire_ang=45.0):
    """the zero-parameter twin of the rule: the argmax of softmax(-d^2 / (2 sigma^2)) over `candidates`, as an index into `ends` (None: no candidate in reach, the fixed dustbin case)"""
    c = candidates(ends, pos, tan, entry, wire_pos, wire_ang)
    if not c:
        return None
    logit = np.array([-(d ** 2) / (2.0 * WIRE_SIGMA ** 2) for d, _, _, _ in c])
    w = np.exp(logit - logit.max())
    w = w / w.sum()
    return c[int(np.argmax(w))][1]                                                                  # np.argmax keeps the first of equal weights: the list order of the nearest-first sort


def twin_choice(ends, pairs, wire_pos, wire_ang=45.0):
    """one (entry index or None, exit index or None) per pair, from the ends of the pair (pr[0], pr[1] = the entry end and heading, pr[2], pr[3] = the exit end and heading)"""
    return [(twin_pick(ends, pr[0], pr[1], True, wire_pos, wire_ang), twin_pick(ends, pr[2], pr[3], False, wire_pos, wire_ang)) for pr in pairs]


def rule_pick(ends, pos, tan, entry, wire_pos, wire_ang=45.0):
    """the nearest-piece rule as `add_connectors` writes it (strict `d < d_in`, the first of equal distances): the independent reference of the self-test"""
    import qends_eval as QE
    best, d_best = None, np.inf
    pos, tan = np.asarray(pos, float), np.asarray(tan, float)
    for i, en in enumerate(ends):
        q, t = (en[0], en[1]) if entry else (en[2], en[3])
        d = np.linalg.norm(q - pos)
        if d <= wire_pos and QE._ang(t, tan) <= wire_ang and d < d_best:
            best, d_best = i, d
    return best


def _selftest():
    import os
    import sys
    root = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    sys.path.insert(0, os.path.join(root, "evaluation"))
    import qends_eval as QE
    rng = np.random.default_rng(0)
    R = 4.0

    def scene(n_pieces):
        lines, kinds, links = [], [], []
        for i in range(n_pieces):
            a = rng.uniform(-6, 6, 2)
            h = rng.uniform(0, 2 * np.pi)
            L = rng.uniform(3, 20)
            t = np.linspace(0, L, 8)
            lines.append(a + np.c_[t * np.cos(h), t * np.sin(h)])
            kinds.append("lane")
            links.append((i,))
        return lines, kinds, links, np.zeros(n_pieces, bool)

    # 1. the twin equals the rule on random scenes (both sides), including scenes with co-located ends
    n_cmp = n_some = 0
    for _ in range(600):
        lines, kinds, links, lm = scene(int(rng.integers(0, 14)))
        if len(lines) > 3 and rng.random() < 0.5:
            lines[2][-1] = lines[1][-1]                                                                  # co-located ends (a merge): the same place, two list entries
        _, ends = kept_ends(lines, kinds, links, lm, [], R, R)
        pos = rng.uniform(-8, 8, 2)
        tan = np.array([np.cos(a := rng.uniform(0, 2 * np.pi)), np.sin(a)])
        for entry in (True, False):
            r, t = rule_pick(ends, pos, tan, entry, R), twin_pick(ends, pos, tan, entry, R)
            assert r == t, (r, t, entry)
            n_cmp += 1
            n_some += r is not None
    assert n_some > 100, n_some                                                                          # the comparison is not vacuous
    # 2. exactly at the reach: d = R is in reach (the rule's `<=`), and a tie keeps the first entry
    ends0 = [(np.array([4.0, 0.0]), np.array([0.0, 1.0]), np.array([9.0, 9.0]), np.array([0.0, 1.0])), (np.array([4.0, 0.0]), np.array([0.0, 1.0]), np.array([9.0, 9.0]), np.array([0.0, 1.0]))]
    assert twin_pick(ends0, [0.0, 0.0], [0.0, 1.0], True, R) == 0 and rule_pick(ends0, [0.0, 0.0], [0.0, 1.0], True, R) == 0
    assert twin_pick(ends0, [0.0, 0.0], [0.0, 1.0], True, R - 1e-9) is None                                # just out of reach: no candidate
    assert twin_pick(ends0, [0.0, 0.0], [1.0, 0.0], True, R) is None                                       # 90 degrees off: gated out
    assert twin_pick([], [0.0, 0.0], [0.0, 1.0], False, R) is None
    # 3. the cut at MAX_CAND keeps the nearest ones; the chosen index refers to the original list
    far = [(np.array([0.1 * (k + 1), 0.0]), np.array([0.0, 1.0]), np.array([50.0, 50.0]), np.array([0.0, 1.0])) for k in range(12)]
    assert len(candidates(far, [0.0, 0.0], [0.0, 1.0], True, R)) == MAX_CAND and twin_pick(far, [0.0, 0.0], [0.0, 1.0], True, R) == 0
    # 4. kept_ends equals the pieces `add_connectors` keeps: a long link covered by a pair is dropped, the others stay in order
    lines = [np.c_[np.zeros(6), np.linspace(-20, -2, 6)], np.c_[np.zeros(6) + 8, np.linspace(2, 20, 6)], np.c_[np.linspace(0, 8, 6), np.linspace(-2, 2, 6)]]
    kinds, links, lm = ["lane", "lane", "link"], [(0,), (1,), (0,)], np.array([True])
    pr = [(np.array([0.0, -2.5]), np.array([0.0, 1.0]), np.array([8.0, 2.5]), np.array([0.0, 1.0]))]
    keep, ends = kept_ends(lines, kinds, links, lm, pr, R, R)
    assert keep == [0, 1] and len(ends) == 2 and np.allclose(ends[0][0], lines[0][-1]) and np.allclose(ends[1][2], lines[1][0])
    assert kept_ends(lines, kinds, links, lm, [], R, R)[0] == [0, 1, 2]
    # 5. add_connectors through the twin choice = the rule, bit for bit, on random scenes with random pairs (snap and its refinements; scenes with covered long links); mutations
    def unit():
        a = rng.uniform(0, 2 * np.pi)
        return np.array([np.cos(a), np.sin(a)])

    def same(x, y):
        return x[1:] == y[1:] and x[0]["edges"] == y[0]["edges"] and len(x[0]["pts"]) == len(y[0]["pts"]) and all(np.array_equal(p, q) for p, q in zip(x[0]["pts"], y[0]["pts"]))

    n_graphs = n_covered = n_swapped_differs = 0
    for _ in range(300):
        lines, kinds, links, _lm = scene(int(rng.integers(1, 12)))
        kinds = ["link" if rng.random() < 0.3 else "lane" for _ in kinds]
        lm = np.array([k == "link" for k in kinds])
        pairs = [(rng.uniform(-8, 8, 2), unit(), rng.uniform(-8, 8, 2), unit()) for _ in range(int(rng.integers(1, 5)))]
        for i in [k for k, v in enumerate(kinds) if v == "link"][:1]:                                   # a pair on the ends of a long link: the link is covered and dropped
            P = lines[i]
            pairs.append((P[0] + rng.normal(0, 0.3, 2), unit(), P[-1] + rng.normal(0, 0.3, 2), unit()))
        kw = dict(snap=True, snap_nogeom=bool(rng.random() < 0.3), snap_dedup=bool(rng.random() < 0.3), snap_both=bool(rng.random() < 0.3), snap_guard=bool(rng.random() < 0.3))
        base = QE.add_connectors(lines, [], [[i] for i in range(len(lines))], kinds, links, lm, pairs, **kw)
        keep, ends = kept_ends(lines, kinds, links, lm, pairs, QE.WIRE_POS, QE.COVER_POS)
        n_covered += len(keep) < len(lines)
        wc = twin_choice(ends, pairs, QE.WIRE_POS)
        assert same(base, QE.add_connectors(lines, [], [[i] for i in range(len(lines))], kinds, links, lm, pairs, wire_choice=wc, **kw)), "the twin choice must reproduce the rule bit for bit"
        sw = [(b, a) if a is not None and b is not None and a != b else (a, b) for a, b in wc]
        if sw != wc:
            n_swapped_differs += not same(base, QE.add_connectors(lines, [], [[i] for i in range(len(lines))], kinds, links, lm, pairs, wire_choice=sw, **kw))
        n_graphs += 1
    assert n_covered > 5 and n_swapped_differs > 3, (n_covered, n_swapped_differs)                       # the comparison covers the cover drop, and a changed choice changes the graph (the test sees the choice)
    args_ok = (lines, [], [[i] for i in range(len(lines))], kinds, links, lm, pairs)
    for bad, kwargs in (("a wire_choice of the wrong length", dict(snap=True, wire_choice=[(None, None)] * (len(pairs) + 1))), ("a wire_choice without snap", dict(wire_choice=[(None, None)] * len(pairs))),
                        ("an index outside the kept pieces", dict(snap=True, wire_choice=[(len(ends), None)] + [(None, None)] * (len(pairs) - 1)))):
        try:
            QE.add_connectors(*args_ok, **kwargs)
        except AssertionError:
            continue
        raise AssertionError(f"accepted {bad}")
    print(f"jwire selftest ok ({n_cmp} twin / rule comparisons, {n_some} with a candidate; {n_graphs} graphs through add_connectors, {n_covered} with a covered long link)")


if __name__ == "__main__":
    _selftest()
