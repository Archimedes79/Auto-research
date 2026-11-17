"""E350 (promoted from the scratch checks of E325, `cmp_twin.py` and `live_cmp.py`): the two checks of the read-out path on a pair run that come before any wiring head is trained or scored on it.

    wsl bash tools/env/_py14.sh jwire_check.py twin --run e324_s_s0 --seed 0 --tag twin [--thr 0.1]            # CPU: the card of `jends_eval.py score --run <run> --seed <seed> --thr 0.1 --snap --wire twin --tag <tag>` against the
                                                                                                                 # stored card of the same run (`ends_card_snap_thr<thr>.json`)
    wsl bash tools/env/_py14.sh jwire_check.py live --seed 0 --pair-run e324_s_s0 [--max 24] [--workers 2] [--device cpu]   # the live pass (B and the pair run, `jwire_data.live_records`) on the first 24 `val_clean` frames against the
                                                                                                                 # records of the stored dump (`jwire_data.py val --seed <seed> --pair-run <run>` first); on a GPU (the default when there is
                                                                                                                 # one) it waits for room like `jends_eval.py dump` (`gpu_gate`); `--device cpu` needs neither (24 frames, a few minutes)
    wsl bash tools/env/_py14.sh jwire_check.py selftest

Acceptance rules (the exit code is 0 when they hold):
  twin: EVERY field of the twin card equals the stored card (all keys of the stored card except `tag`, `wire`, `wire_run`, which differ by design; a key only the twin card has, `full` / `spec` of the scorer since E292, was
        not recorded in E312's stored cards and is not compared; the twin card must say `wire` = twin and the tag asked for, the stored card must be the rule's, both of the run and seed asked for): numbers, curves, wired
        ends, `ends_within_1m...`, the options; and the per-frame counts of its `.counts.pkl` (scenes, base and head of every frame) equal the stored ones. Exact equality, no tolerance (the zero-parameter twin is the rule bit for bit).
  live: all fields of all frames' records (ends, attrs, pairs, kp_p, gts; atol 2e-3) equal, except at most ONE pair in ONE frame: the pair lists of that frame differ by exactly one row (a pair near p = 0.1 that one pass kept
        and the other dropped: numerics of another batch size, accepted in E325 step 2), the other rows of pairs / kp_p / gts, and ends / attrs, are equal. A second such frame, a pair that differs in value, or any other field
        refuses.
"""
import argparse
import json
import os
import pickle
import sys
from types import SimpleNamespace as NS

import numpy as np

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for sub in ("evaluation", "training", "training/grid", "training/data", "training/data/prep", "training/train", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub)))

IGNORED = ("tag", "wire", "wire_run")                                                                 # keys of a card that differ between the twin and the stored card by design
FIELDS = ("ends", "attrs", "pairs", "kp_p", "gts")                                                    # the arrays of a record (`jwire_data.record_of`)
ATOL = 2e-3


def same(x, y):
    """exact equality of nested dicts / lists / tuples / arrays / numbers / strings"""
    if isinstance(x, dict):
        return isinstance(y, dict) and x.keys() == y.keys() and all(same(x[k], y[k]) for k in x)
    if isinstance(x, (list, tuple)):
        return isinstance(y, (list, tuple)) and len(x) == len(y) and all(same(p, q) for p, q in zip(x, y))
    if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
        a, b = np.asarray(x), np.asarray(y)
        return bool(np.array_equal(a, b, equal_nan=a.dtype.kind in "fc" and b.dtype.kind in "fc"))
    return x == y or (x != x and y != y)                                                              # NaN equals NaN


def card_diff(stored, twin):
    """-> the keys of the stored card (the keys of IGNORED not counted) in which the twin card differs; a key that only the twin card has (`full`, `spec`: written by the scorer since E292, absent from E312's stored cards)
    was not recorded then and is not compared"""
    return sorted(k for k in set(stored) - set(IGNORED) if not same(stored[k], twin.get(k)))


def counts_diff(a, b):
    """-> the parts of two `.counts.pkl` (scenes, base, head: one dict per frame) that differ"""
    return sorted(k for k in set(a) | set(b) if not same(a.get(k), b.get(k)))


def twin_check(run, seed, tag, thr=0.1):
    """-> (identical, [what differs]) for the twin card `<tag>` of a run against its stored snapped card"""
    import qends_eval as QE
    names = [QE.card_name(NS(snap=True, thr=thr, tag=t)) for t in (None, tag)]
    d = os.path.join(ROOT, "runs", run)
    cards = [json.load(open(os.path.join(d, n))) for n in names]
    counts = [pickle.load(open(os.path.join(d, n[:-5] + ".counts.pkl"), "rb")) for n in names]
    stored, twin = cards
    diff = [f"card {k}" for k in card_diff(stored, twin)] + [f"counts {k}" for k in counts_diff(*counts)]
    for what, c in (("stored", stored), ("twin", twin)):
        if c["run"] != run:
            diff.append(f"the {what} card is of run {c['run']}, not {run}")
        if c["seed"] != seed:
            diff.append(f"the {what} card is of seed {c['seed']}, not {seed}")
    if twin.get("wire") != "twin" or twin.get("tag") != tag:
        diff.append(f"the card is not the twin's (wire {twin.get('wire')}, tag {twin.get('tag')})")
    if stored.get("wire") is not None:
        diff.append(f"the stored card is not the rule's (wire {stored.get('wire')})")
    return not diff, diff


def _close(a, b, atol):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return a.shape == b.shape and np.allclose(a, b, atol=atol, equal_nan=True)


def frame_diff(a, b, atol=ATOL):
    """-> [] when the records a, b of a frame are equal, ['one pair'] when they differ only by ONE pair (the rule of the module docstring), else the names of the fields that differ"""
    bad = [k for k in FIELDS if not _close(a[k], b[k], atol)]
    if bad and set(bad) <= {"pairs", "kp_p", "gts"} and abs(len(a["pairs"]) - len(b["pairs"])) == 1:
        big, small = (a, b) if len(a["pairs"]) > len(b["pairs"]) else (b, a)
        for n in range(len(big["pairs"])):
            keep = np.arange(len(big["pairs"])) != n
            if all(_close(np.asarray(big[k])[keep], small[k], atol) for k in ("pairs", "kp_p", "gts")):
                return ["one pair"]
    return bad


def compare_records(live, ref, atol=ATOL):
    """-> (accepted, frames compared, {row: what differs}) for the live records against the records of the stored dump (`ref`: row -> record); the rule is the module docstring's"""
    diff = {}
    for r in live:
        q = ref.get(r["row"])
        d = ["no stored record"] if q is None else frame_diff(r, q, atol)
        if d:
            diff[r["row"]] = d
    ok = bool(live) and (not diff or (len(diff) == 1 and list(diff.values())[0] == ["one pair"]))
    return ok, len(live), diff


def live_check(seed, pair_run, nmax=24, workers=2, device=None):
    import jwire_data as JD
    path = JD.cache_path("val", seed, pair_run=pair_run)
    assert os.path.exists(path), f"{path} is missing: `jwire_data.py val --seed {seed} --pair-run {pair_run}` builds the stored-dump records first"
    ref = {r["row"]: r for r in JD.load_records(path, seed, pair_run)}
    import torch
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if dev.startswith("cuda"):                                                                            # D100: next to two trainings a pass can crash one of them (CUDA unknown error), as `jends_eval.dump` waits
        import gpu_gate
        gpu_gate.wait_for_room(who=f"jwire_check live {pair_run}")
    live = JD.live_records(seed, "val_clean", 0.1, nmax, workers, device=dev, pair_run=pair_run)
    return compare_records(live, ref)


def _selftest():
    import tempfile
    # live: planted differences against a stored dump of 4 frames
    rng = np.random.default_rng(0)

    def rec(row):
        n = 3
        return dict(row=row, ends=rng.normal(size=(5, 8)), attrs=rng.normal(size=(5, 6)).astype(np.float32), pairs=rng.normal(size=(n, 8)), kp_p=rng.uniform(size=(n, 2)), gts=rng.normal(size=(n, 2, 4)))

    ref = {r["row"]: r for r in (rec(i) for i in range(4))}
    live = lambda: [{k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in ref[i].items()} for i in range(4)]  # noqa: E731

    def plus_pair(r):
        return dict(r, pairs=np.r_[r["pairs"], rng.normal(size=(1, 8))], kp_p=np.r_[r["kp_p"], rng.uniform(size=(1, 2))], gts=np.r_[r["gts"], rng.normal(size=(1, 2, 4))])
    ok, n, d = compare_records(live(), ref)
    assert ok and n == 4 and d == {}, d                                                               # equal
    L = live()
    L[1]["ends"] = L[1]["ends"] + 1e-4
    assert compare_records(L, ref)[0], "a difference below the tolerance is equal"
    L = live()
    L[2] = plus_pair(L[2])
    ok, _, d = compare_records(L, ref)
    assert ok and d == {2: ["one pair"]}, d                                                           # one pair near p = 0.1 in one frame: tolerated (the live pass has one more)
    L = live()
    ref2 = dict(ref)
    ref2[3] = plus_pair(ref[3])
    ok, _, d = compare_records(L, ref2)
    assert ok and d == {3: ["one pair"]}, d                                                           # ... or one fewer
    L = live()
    L[0], L[2] = plus_pair(L[0]), plus_pair(L[2])
    ok, _, d = compare_records(L, ref)
    assert not ok and sorted(d) == [0, 2], d                                                          # a second frame with a pair: refused
    L = live()
    L[1] = plus_pair(plus_pair(L[1]))
    assert not compare_records(L, ref)[0], "two extra pairs in a frame are refused"
    L = live()
    L[1] = plus_pair(L[1])
    L[1]["pairs"][0] += 0.5
    assert not compare_records(L, ref)[0], "one extra pair AND another pair that differs in value is not 'one pair'"
    L = live()
    L[1]["pairs"] = L[1]["pairs"] + np.array([0.5] + [0.0] * 7)
    ok, _, d = compare_records(L, ref)
    assert not ok and d == {1: ["pairs"]}, d                                                          # a pair that differs in value (same count)
    L = live()
    L[2] = dict(plus_pair(L[2]), ends=L[2]["ends"] + 0.5)
    ok, _, d = compare_records(L, ref)
    assert not ok and "ends" in d[2] and "one pair" not in d[2], d                                    # the pair is not the only difference
    for field, plant in (("attrs", lambda x: x + 0.1), ("ends", lambda x: x + 0.1), ("kp_p", lambda x: x * 0.5), ("gts", lambda x: x + 0.1), ("pairs", lambda x: x + 0.1)):
        L = live()
        L[0][field] = plant(L[0][field])
        ok, _, d = compare_records(L, ref)
        assert not ok and d == {0: [field]}, (field, d)                                                   # each field alone refuses
    L = live()
    L[3]["row"] = 99
    assert not compare_records(L, ref)[0] and not compare_records([], ref)[0], "a frame without a stored record, or no frame at all, is refused"
    # twin: planted cards of a run in a temp ROOT
    global ROOT
    old = ROOT
    with tempfile.TemporaryDirectory() as t:
        ROOT = t
        try:
            os.makedirs(os.path.join(t, "runs", "rx"))
            card = lambda **k: dict(dict(run="rx", seed=0, thr=0.1, frames=3, curves=100, wired_both_ends=60, numbers=dict(head=dict(topo_F1=0.7, dup_frac=0.18), base=dict(topo_F1=0.6)), snap=True, full="f.pkl", tag=None, wire=None), **k)  # noqa: E731
            counts = lambda x=1: dict(scenes=[1, 2, 3], base=[dict(tp=1.0, n=2)] * 3, head=[dict(tp=x, n=2)] * 3)  # noqa: E731

            def put(stored, twin, cs, ct):
                for stem, c, k in (("ends_card_snap_thr0.1", stored, cs), ("ends_card_twin_snap_thr0.1", twin, ct)):
                    json.dump(c, open(os.path.join(t, "runs", "rx", stem + ".json"), "w"))
                    pickle.dump(k, open(os.path.join(t, "runs", "rx", stem + ".counts.pkl"), "wb"))
            twin = card(tag="twin", wire="twin", wire_run=None)
            put(card(), twin, counts(), counts())
            assert twin_check("rx", 0, "twin") == (True, []), twin_check("rx", 0, "twin")              # equal, the provenance keys differ by design (and the stored card may lack wire_run)
            old_card = {k: v for k, v in card().items() if k not in ("full", "tag", "wire")}
            put(old_card, twin, counts(), counts())
            assert twin_check("rx", 0, "twin") == (True, []), "a stored card of E312's age has no full / tag / wire"
            put(card(), {k: v for k, v in twin.items() if k != "curves"}, counts(), counts())
            assert twin_check("rx", 0, "twin")[1] == ["card curves"], "a field the twin card lost is a difference"
            put(card(), dict(twin, full="g.pkl"), counts(), counts())
            assert twin_check("rx", 0, "twin")[1] == ["card full"], "another graph dump"
            put(card(), dict(twin, numbers=dict(head=dict(topo_F1=0.7 + 1e-12, dup_frac=0.18), base=dict(topo_F1=0.6))), counts(), counts())
            assert twin_check("rx", 0, "twin")[1] == ["card numbers"]                                  # one digit of one number
            put(card(), dict(twin, curves=101), counts(), counts())
            assert twin_check("rx", 0, "twin")[1] == ["card curves"]
            put(card(), twin, counts(), counts(1.5))
            assert twin_check("rx", 0, "twin")[1] == ["counts head"]                                   # one per-frame count
            put(card(), twin, counts(), dict(counts(), base=[dict(tp=1.0, n=3)] * 3))
            assert twin_check("rx", 0, "twin")[1] == ["counts base"]
            put(card(), twin, counts(), dict(counts(), scenes=[1, 2, 4]))
            assert twin_check("rx", 0, "twin")[1] == ["counts scenes"]
            put(card(), dict(twin, tag="other"), counts(), counts())
            assert "not the twin's" in twin_check("rx", 0, "twin")[1][0]                               # the tag asked for is the card's
            put(card(wire="head"), twin, counts(), counts())
            assert twin_check("rx", 0, "twin")[1] == ["the stored card is not the rule's (wire head)"]   # the stored card is the rule's (`wire` is ignored by the field comparison)
            for over, what in ((dict(run="ry"), "run"), (dict(seed=1), "seed")):                       # both cards of another run / seed: equal to each other, not the ones asked for
                put(card(**over), dict(twin, **over), counts(), counts())
                ok, d = twin_check("rx", 0, "twin")
                assert not ok and len(d) == 2 and all(f"card is of {what}" in x for x in d), (what, d)
            put(card(), dict(twin, wire="head"), counts(), counts())
            assert "not the twin's" in twin_check("rx", 0, "twin")[1][0]
            put(card(), twin, counts(), counts())
            ok, d = twin_check("rx", 1, "twin")
            assert not ok and "seed" in d[0], d                                                        # the seed asked for must be the card's
        finally:
            ROOT = old
    print("jwire_check selftest ok")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("twin")
    t.add_argument("--run", required=True)
    t.add_argument("--seed", type=int, required=True)
    t.add_argument("--tag", required=True)
    t.add_argument("--thr", type=float, default=0.1)
    v = sub.add_parser("live")
    v.add_argument("--seed", type=int, required=True)
    v.add_argument("--pair-run", required=True)
    v.add_argument("--max", type=int, default=24)
    v.add_argument("--workers", type=int, default=2)
    v.add_argument("--device", default=None)
    sub.add_parser("selftest")
    a = ap.parse_args(argv)
    if a.cmd == "selftest":
        _selftest()
        return 0
    if a.cmd == "twin":
        ok, diff = twin_check(a.run, a.seed, a.tag, a.thr)
        print(f"twin {a.tag} of {a.run} (seed {a.seed}, thr {a.thr}) against the stored snapped card: " + ("IDENTICAL (numbers, curves, wired ends, per-frame counts)" if ok else "DIFFERENT: " + "; ".join(diff)))
        return 0 if ok else 1
    ok, n, diff = live_check(a.seed, a.pair_run, a.max, a.workers, a.device)
    print(f"live pass of {a.pair_run} (seed {a.seed}) on {n} val_clean frames against the stored-dump records: " + ("ACCEPTED" if ok else "REFUSED") + (f"; frames that differ: {diff}" if diff else "; all fields equal"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
