"""E350 (promoted from the scratch checks of E325, `cmp_twin.py` and `live_cmp.py`): the two checks of the read-out path on a pair run that come before any wiring head is trained or scored on it.

    wsl bash tools/env/_py14.sh jwire_check.py twin --run e324_s_s0 --seed 0 --tag twin [--thr 0.1] [--read-out snap|primary]   # CPU: the card of `jends_eval.py score --run <run> --seed <seed> --thr 0.1 --snap --wire twin --tag <tag>` against the
                                                                                                                 # stored card of the same run (`ends_card_snap_thr<thr>.json`); `--read-out primary` (E358, with `--attach --snap-nogeom` in the
                                                                                                                 # scoring): the cards `ends_card_[<tag>_]snap_nogeom_attach_thr<thr>.json`
    wsl bash tools/env/_py14.sh jwire_check.py live --seed 0 --pair-run e324_s_s0 [--ends kp|att] [--max 24] [--workers 2] [--device cpu]   # the live pass (B and the pair run, `jwire_data.live_records`) on the first 24 `val_clean` frames against the
                                                                                                                 # records of the stored dump (`jwire_data.py val --seed <seed> --pair-run <run> [--ends att]` first); on a GPU (the default when
                                                                                                                 # there is one) it waits for room like `jends_eval.py dump` (`gpu_gate`); `--device cpu` needs neither (24 frames, a few minutes)
    wsl bash tools/env/_py14.sh jwire_check.py selftest
`--read-out` and `--ends` are one choice (snap = kp, primary = att): either fixes the other on both commands, both given must agree, none = snap / kp (E325 / E350).

Acceptance rules (the exit code is 0 when they hold):
  twin: EVERY field of the twin card equals the stored card (all keys of the stored card except `tag`, `wire`, `wire_run`, which differ by design; a key only the twin card has, `full` / `spec` of the scorer since E292, was
        not recorded in E312's stored cards and is not compared; the twin card must say `wire` = twin and the tag asked for, the stored card must be the rule's, both of the run and seed asked for and of the read-out asked for:
        `attach` and `snap_nogeom` (in `snap_options`) true under primary, false under snap): numbers, curves, wired
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
READ_OUT_ENDS = dict(snap="kp", primary="att")                                                        # the read-out of a stored card and the end kind of the records of its pair run (as `jwire_verdict.READ_OUT_ENDS`)
ATOL = 2e-3


def read_out_ends(read_out=None, ends=None):
    """-> (read_out, ends) from the two options of the command line: either one fixes the other, both given must agree, none = (snap, kp)"""
    read_out = read_out or {v: k for k, v in READ_OUT_ENDS.items()}.get(ends, "snap")
    assert ends in (None, READ_OUT_ENDS[read_out]), f"--read-out {read_out} and --ends {ends} name different things (snap = kp, primary = att)"
    return read_out, READ_OUT_ENDS[read_out]


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


def twin_check(run, seed, tag, thr=0.1, read_out="snap"):
    """-> (identical, [what differs]) for the twin card `<tag>` of a run against its stored card of the read-out (snap: `ends_card_snap_thr<thr>.json`; primary: `ends_card_snap_nogeom_attach_thr<thr>.json`)"""
    import qends_eval as QE
    primary = read_out == "primary"
    names = [QE.card_name(NS(snap=True, thr=thr, tag=t, attach=primary, snap_nogeom=primary)) for t in (None, tag)]
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
        if bool(c.get("attach")) != primary or ("snap_nogeom" in (c.get("snap_options") or [])) != primary:
            diff.append(f"the {what} card is not the {read_out} read-out (attach {c.get('attach')}, snap_options {c.get('snap_options')})")
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


def live_check(seed, pair_run, nmax=24, workers=2, device=None, ends="kp"):
    import jwire_data as JD
    path = JD.cache_path("val", seed, pair_run=pair_run, ends=ends)
    assert os.path.exists(path), f"{path} is missing: `jwire_data.py val --seed {seed} --pair-run {pair_run} --ends {ends}` builds the stored-dump records first"
    ref = {r["row"]: r for r in JD.load_records(path, seed, pair_run, ends)}
    import torch
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if dev.startswith("cuda"):                                                                            # D100: next to two trainings a pass can crash one of them (CUDA unknown error), as `jends_eval.dump` waits
        import gpu_gate
        gpu_gate.wait_for_room(who=f"jwire_check live {pair_run}")
    live = JD.live_records(seed, "val_clean", 0.1, nmax, workers, device=dev, pair_run=pair_run, ends=ends)
    return compare_records(live, ref)


def _selftest():
    import contextlib
    import io
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

            def put(stored, twin, cs, ct, st="snap_", tag="twin"):
                for stem, c, k in ((f"ends_card_{st}thr0.1", stored, cs), (f"ends_card_{tag}_{st}thr0.1", twin, ct)):
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
            # E358, the PRIMARY read-out: the cards are named `..._snap_nogeom_attach_...` (as the scorer names them), the options of both must be attach + snap_nogeom, nothing falls back to the snapped names
            import qends_eval as QE
            assert QE.card_name(NS(snap=True, thr=0.1, tag=None, attach=True, snap_nogeom=True)) == "ends_card_snap_nogeom_attach_thr0.1.json" and QE.card_name(NS(snap=True, thr=0.1, tag="twinatt", attach=True, snap_nogeom=True)) == "ends_card_twinatt_snap_nogeom_attach_thr0.1.json"
            assert QE.card_name(NS(snap=True, thr=0.1, tag=None)) == "ends_card_snap_thr0.1.json" and QE.card_name(NS(snap=True, thr=0.1, tag="twin")) == "ends_card_twin_snap_thr0.1.json"
            prim = dict(attach=True, snap_options=["snap_nogeom"])
            ptwin = card(tag="twinatt", wire="twin", wire_run=None, **prim)
            pt = lambda: twin_check("rx", 0, "twinatt", read_out="primary")  # noqa: E731
            pput = lambda stored, twin, cs=None, ct=None: put(stored, twin, cs or counts(), ct or counts(), "snap_nogeom_attach_", "twinatt")  # noqa: E731
            try:
                pt()                                                                                     # only the snapped names exist so far: the primary read-out does not fall back to them
                raise AssertionError("the primary read-out must not read the snapped cards")
            except FileNotFoundError:
                pass
            pput(card(**prim), ptwin)
            assert pt() == (True, []), pt()
            pput(card(), ptwin)                                                                          # a stored SNAPPED card under the primary name
            assert pt()[1] == ["the stored card is not the primary read-out (attach None, snap_options None)"], pt()
            pput(card(**prim), dict(ptwin, attach=False))
            assert pt()[1] == ["card attach", "the twin card is not the primary read-out (attach False, snap_options ['snap_nogeom'])"], pt()
            pput(card(**prim), dict(ptwin, snap_options=[]))
            assert pt()[1] == ["card snap_options", "the twin card is not the primary read-out (attach True, snap_options [])"], pt()
            pput(card(**prim), dict(ptwin, numbers=dict(head=dict(topo_F1=0.7 + 1e-12, dup_frac=0.18), base=dict(topo_F1=0.6))))
            assert pt()[1] == ["card numbers"], pt()                                                      # the primary twin is compared as strictly as the snapped one
            pput(card(**prim), ptwin, ct=counts(1.5))
            assert pt()[1] == ["counts head"], pt()
            put(card(**prim), dict(twin, **prim), counts(), counts())                                      # a card with attach / snap_nogeom is not the snapped read-out either
            assert twin_check("rx", 0, "twin")[1] == ["the stored card is not the snap read-out (attach True, snap_options ['snap_nogeom'])", "the twin card is not the snap read-out (attach True, snap_options ['snap_nogeom'])"], twin_check("rx", 0, "twin")
            # the command line: either option fixes the other, both must agree
            assert [read_out_ends(*x) for x in ((None, None), ("snap", None), ("primary", None), (None, "kp"), (None, "att"), ("primary", "att"), ("snap", "kp"))] == [("snap", "kp"), ("snap", "kp"), ("primary", "att"), ("snap", "kp"), ("primary", "att"), ("primary", "att"), ("snap", "kp")]
            for bad in (("primary", "kp"), ("snap", "att")):
                try:
                    read_out_ends(*bad)
                    raise AssertionError(f"{bad} must be refused")
                except AssertionError as e:
                    assert "name different things" in str(e), e
            pput(card(**prim), ptwin)
            with contextlib.redirect_stdout(io.StringIO()) as out:
                for argv in (["--read-out", "primary"], ["--ends", "att"], ["--read-out", "primary", "--ends", "att"]):
                    assert main(["twin", "--run", "rx", "--seed", "0", "--tag", "twinatt"] + argv) == 0, argv
                assert main(["twin", "--run", "rx", "--seed", "0", "--tag", "twin"]) == 1, "the snapped read-out of cards that are primary ones"
            for argv in (["twin", "--run", "rx", "--seed", "0", "--tag", "t", "--ends", "bogus"], ["live", "--seed", "0", "--pair-run", "rx", "--ends", "bogus"], ["live", "--seed", "0", "--pair-run", "rx", "--read-out", "bogus"]):
                with contextlib.redirect_stderr(io.StringIO()):
                    try:
                        main(argv)
                        raise AssertionError(f"{argv} must be refused by the command line")
                    except SystemExit as ex:
                        assert ex.code == 2, argv
            assert out.getvalue().count("against the stored primary card: IDENTICAL") == 3 and "against the stored snap card: DIFFERENT" in out.getvalue(), out.getvalue()
            # live: the end kind reaches the cache of the stored records, its meta check, the live pass and the message of a missing cache
            import jwire_data as JD
            from unittest import mock
            seen, recs = [], [dict(row=1, **{k: np.zeros((1, 2)) for k in FIELDS})]
            cache = os.path.join(t, "cache.pkl")
            open(cache, "wb").write(b"x")
            with mock.patch.object(JD, "cache_path", lambda kind, seed, nmax=0, thr=0.1, pair_run=None, ends="kp": seen.append(("path", kind, seed, pair_run, ends)) or cache), \
                    mock.patch.object(JD, "load_records", lambda path, seed, pair_run=None, ends="kp": seen.append(("load", seed, pair_run, ends)) or recs), \
                    mock.patch.object(JD, "live_records", lambda seed, split, thr, nmax, workers, device=None, pair_run=None, ends="kp": seen.append(("live", split, nmax, pair_run, ends)) or recs):
                assert live_check(0, "e324_s_s0", 24, 2, "cpu", "att")[0] and live_check(0, "e324_s_s0", 24, 2, "cpu")[0]
            assert seen == [("path", "val", 0, "e324_s_s0", "att"), ("load", 0, "e324_s_s0", "att"), ("live", "val_clean", 24, "e324_s_s0", "att"), ("path", "val", 0, "e324_s_s0", "kp"), ("load", 0, "e324_s_s0", "kp"), ("live", "val_clean", 24, "e324_s_s0", "kp")], seen
            calls = []
            with mock.patch.object(sys.modules[__name__], "live_check", lambda *x: calls.append(x) or (True, 24, {})), contextlib.redirect_stdout(io.StringIO()):
                assert main(["live", "--seed", "0", "--pair-run", "e324_s_s0", "--read-out", "primary"]) == 0 and main(["live", "--seed", "0", "--pair-run", "e324_s_s0", "--ends", "att"]) == 0 and main(["live", "--seed", "0", "--pair-run", "e312_p_s0"]) == 0
            assert [c[0:2] + c[-1:] for c in calls] == [(0, "e324_s_s0", "att"), (0, "e324_s_s0", "att"), (0, "e312_p_s0", "kp")], calls
            with mock.patch.object(JD, "cache_path", lambda *x, **k: os.path.join(t, "missing.pkl")):
                try:
                    live_check(0, "e324_s_s0", 24, 2, "cpu", "att")
                    raise AssertionError("a missing stored-dump cache must be refused")
                except AssertionError as e:
                    assert "--ends att" in str(e) and "builds the stored-dump records" in str(e), e
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
    for s in (t, v):
        s.add_argument("--read-out", default=None, choices=tuple(READ_OUT_ENDS), help="snap (default): the snapped read-out, keypoint ends; primary (E358): the end-to-end read-out, attached ends")
        s.add_argument("--ends", default=None, choices=tuple(READ_OUT_ENDS.values()), help="the end kind of the read-out: kp = snap (default), att = primary")
    sub.add_parser("selftest")
    a = ap.parse_args(argv)
    if a.cmd == "selftest":
        _selftest()
        return 0
    read_out, ends = read_out_ends(a.read_out, a.ends)
    if a.cmd == "twin":
        ok, diff = twin_check(a.run, a.seed, a.tag, a.thr, read_out)
        print(f"twin {a.tag} of {a.run} (seed {a.seed}, thr {a.thr}) against the stored {read_out} card: " + ("IDENTICAL (numbers, curves, wired ends, per-frame counts)" if ok else "DIFFERENT: " + "; ".join(diff)))
        return 0 if ok else 1
    ok, n, diff = live_check(a.seed, a.pair_run, a.max, a.workers, a.device, ends)
    print(f"live pass of {a.pair_run} (seed {a.seed}, {ends} ends) on {n} val_clean frames against the stored-dump records: " + ("ACCEPTED" if ok else "REFUSED") + (f"; frames that differ: {diff}" if diff else "; all fields equal"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
