"""Spec tests of `jtest_eval.py` (E343, the one TEST use of E324 s): the rules that decide what is written, read and reported, on toy data (CPU, no model, the TEST split is never opened).
Write helper (E321's artefacts are refused, atomic), dump validation and reuse, the reading rule at its boundaries, the D79 guards against E312, the glue of `score_dumps` (which read-out gets which graph, which pair dump, which ends), the control's checks of
`score_seed` (kill criterion, reproduction of E312's counts), the card and results.md of the arms, the stored-card comparison, and the sha256 of E321's artefacts.

    wsl bash tools/env/_py14.sh test_jtest_eval.py
"""
import argparse
import contextlib
import json
import os
import pickle
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..")))
import jtest_eval as J  # noqa: E402


@contextlib.contextmanager
def patched(*pairs):
    """(object, name, value) triples set for the block and put back after it"""
    old = [(o, n, getattr(o, n)) for o, n, _ in pairs]
    for o, n, v in pairs:
        setattr(o, n, v)
    try:
        yield
    finally:
        for o, n, v in old:
            setattr(o, n, v)


def raises(exc, f, *a, **k):
    try:
        f(*a, **k)
    except exc:
        return True
    return False


# ------------------------------------------------------------------------------------------------------------------------------ the write helper
def test_write_helper_refuses_e321_and_is_atomic():
    """the guard runs on a COPY of the layout (ROOT = a temporary folder with the two protected folders in it): a broken guard writes into the temporary folder, never into the real artefacts of E321"""
    assert J.PROTECTED == ("temp/e321", "runs/eval/e312_test"), "the protected folders are E321's dumps and cards"
    with tempfile.TemporaryDirectory() as d, patched((J, "ROOT", d)):
        for rel in J.PROTECTED + ("temp/e343",):
            os.makedirs(os.path.join(d, rel, "test"))
        for rel in ("temp/e321/test/never_written.pkl", "runs/eval/e312_test/never_written.json", "temp/e343/../e321/test/never_written.pkl", "temp/e321/val_check/never_written.json", "temp/e321/new_folder/never_written.txt"):
            path = os.path.join(d, rel)
            for write in (lambda p: J.save_pickle(p, [1]), lambda p: J.save_json(p, {"a": 1}), lambda p: J.save_text(p, "x")):
                assert raises(PermissionError, write, path), rel
        assert [f for _, _, fs in os.walk(d) for f in fs] == [] and not os.path.exists(os.path.join(d, "temp", "e321", "new_folder")), "a refused write creates nothing: no file, no temporary file, no folder"
        link = os.path.join(d, "link")
        try:
            os.symlink(os.path.join(d, "temp", "e321"), link)
        except OSError:
            link = None                                                                                # no symlinks on this system
        if link:
            assert J.is_protected(os.path.join(link, "test", "x.pkl")) and raises(PermissionError, J.save_text, os.path.join(link, "x.txt"), "x"), "a symlink into E321's folder is refused by its real path"
        assert not J.is_protected(os.path.join(d, "temp", "e343", "test", "x.pkl")) and not J.is_protected(os.path.join(d, "runs", "eval", "e324s_test", "card_s0.json")) and not J.is_protected(os.path.join(d, "temp", "e3210", "x"))
        p = os.path.join(d, "temp", "e343", "test", "dump.pkl")
        J.save_pickle(p, [1, 2])
        assert pickle.load(open(p, "rb")) == [1, 2]
        try:
            with J.atomic(p) as tmp:                                                                   # a crash in the middle: the old file stays whole, no temporary file is left
                open(tmp, "wb").write(b"trunc")
                raise RuntimeError("crash")
        except RuntimeError:
            pass
        assert pickle.load(open(p, "rb")) == [1, 2] and os.listdir(os.path.dirname(p)) == ["dump.pkl"]


def test_e321_artefacts_unchanged():
    """the sha256 of E321's dumps and cards equal the card's (the four pkl files are re-computed here; the dress rehearsal and the TEST run check the same before and after)"""
    present = [r for r in J.E321_SHA256 if os.path.exists(os.path.join(J.ROOT, r))]
    assert J.e321_changed() == [], J.e321_changed()
    print(f"  e321 artefacts: {len(present)} of {len(J.E321_SHA256)} present, all equal to the card's sha256")


# ------------------------------------------------------------------------------------------------------------------------------ dumps: validation and reuse
def toy_dump(n=None, split="val"):
    rows = J.split_rows(split).tolist()
    return [dict(row=r) for r in rows[: n or None]]


def test_dump_validation_and_reuse():
    full = toy_dump()
    assert J.validate_dump(full, "val", "toy") == full
    assert raises(AssertionError, J.validate_dump, full[:-1], "val", "truncated"), "a truncated dump is refused"
    assert raises(AssertionError, J.validate_dump, full + [dict(row=-1)], "val", "too long")
    assert raises(AssertionError, J.validate_dump, full[::-1], "val", "reversed")
    assert raises(AssertionError, J.validate_dump, [dict(row=r) for r in J.split_rows("test").tolist()[:873]], "val", "the rows of the other split")
    assert raises(AssertionError, J.validate_dump, full[:5] + full[6:] + [full[5]], "val", "one row moved")
    assert len(J.validate_dump(full, "val", "toy", n=8)) == 8 and raises(AssertionError, J.validate_dump, full[:7], "val", "toy", 8), "a smoke run takes the first n frames of a longer dump"
    with tempfile.TemporaryDirectory() as d:
        p, calls = os.path.join(d, "b.pkl"), []
        pickle.dump(full, open(p, "wb"))
        got = J.get_dump(p, lambda: calls.append(1), "val", "B")
        assert len(got) == 873 and not calls, "an existing dump is reused, not made again"
        assert raises(SystemExit, J.get_dump, os.path.join(d, "missing.pkl"), None, "val", "B"), "a dump named by a flag must exist"
        q = os.path.join(d, "made.pkl")
        assert len(J.get_dump(q, lambda: (calls.append(1), pickle.dump(full, open(q, "wb"))), "val", "B")) == 873 and calls == [1], "a missing default dump is made once"
        t = os.path.join(d, "trunc.pkl")
        pickle.dump(full[:800], open(t, "wb"))
        assert raises(AssertionError, J.get_dump, t, lambda: calls.append(2), "val", "B") and calls == [1], "a truncated dump on disk is refused, not repaired"


# ------------------------------------------------------------------------------------------------------------------------------ the reading rule
def seeds(D, lo, guards=(True, True), hi=0.03):
    return [dict(D=d, lo=l, hi=hi, guards=g) for d, l, g in zip(D, lo, guards)]


def test_reading_boundaries():
    assert J.reading(seeds((0.0100, 0.0100), (0.0001, 0.0001))) == "FULL RETENTION", "D = 0.0100 exactly is full retention"
    assert J.reading(seeds((0.0100, 0.0099), (0.0001, 0.0001))) == "NOT CONTRADICTED", "D just under 0.0100 on one seed is not full"
    assert J.reading(seeds((0.0200, 0.0200), (0.0001, 0.0))) == "NOT CONTRADICTED", "lo = 0 is not > 0: not full retention"
    assert J.reading(seeds((0.0200, 0.0200), (0.0100, -0.0049))) == "NOT CONTRADICTED"
    assert J.reading(seeds((0.0200, 0.0200), (0.0100, -0.0050))) == "CONTRADICTED", "lo = -0.0050 exactly is not > -0.005"
    assert J.reading(seeds((0.0100, 0.0001), (0.0100, 0.0001))) == "NOT CONTRADICTED" and J.reading(seeds((0.0100, 0.0), (0.0100, 0.0001))) == "CONTRADICTED", "D > 0 is strict"
    assert J.reading(seeds((0.02, 0.02), (0.01, 0.01), guards=(True, False))) == "CONTRADICTED", "a guard miss on one seed is contradicted"
    assert J.reading(seeds((0.02, 0.02), (0.01, 0.01), guards=(False, False))) == "CONTRADICTED"
    assert J.reading(seeds((0.02, float("nan")), (0.01, 0.01))) == "CONTRADICTED", "a nan D fails every comparison"


def test_reading_fallback_without_interval():
    nolo = (None, None)
    assert J.reading(seeds((0.0100, 0.0100), nolo)) == "FULL RETENTION (without interval)"
    assert J.reading(seeds((0.0100, 0.0099), nolo)) == "NOT CONTRADICTED (without interval)"
    assert J.reading(seeds((0.0050, 0.0050), nolo)) == "NOT CONTRADICTED (without interval)" and J.reading(seeds((0.0050, 0.0049), nolo)) == "CONTRADICTED (without interval)"
    assert J.reading(seeds((0.0200, 0.0200), nolo, guards=(True, False))) == "CONTRADICTED (without interval)", "the guards hold in the fallback too"
    # the per-frame counts of E312 reproduce on one seed only: no interval on EITHER seed, the interval of the one seed that has it is not used
    assert J.reading(seeds((0.0200, 0.0075), (-0.02, None))) == "NOT CONTRADICTED (without interval)", "seed 0's interval would contradict, the fallback ignores it"
    assert J.reading(seeds((0.0200, 0.0075), (0.01, float("nan")))) == "NOT CONTRADICTED (without interval)", "a nan interval is no interval"
    assert raises(AssertionError, J.reading, [])


def test_guards_against_e312():
    ref = dict(zip(J.KEYS, [0.5] * len(J.KEYS)))
    ref.update(vert100_dp=19.0)
    arm = lambda **d: dict(ref, **{k: ref[k] + v for k, v in d.items()})  # noqa: E731
    assert J.guards_vs(arm(), ref)["holds"]
    assert J.guards_vs(arm(**{"F1@1.0": -0.0099}), ref)["holds"] and not J.guards_vs(arm(**{"F1@1.0": -0.0101}), ref)["holds"]
    assert J.guards_vs(arm(dup_frac=0.0199), ref)["holds"] and not J.guards_vs(arm(dup_frac=0.0201), ref)["holds"]
    assert J.guards_vs(arm(vert100_dp=1.0), ref)["holds"] and not J.guards_vs(arm(vert100_dp=1.5), ref)["holds"]
    assert J.guards_vs(arm(junction_F1=-0.0299), ref)["holds"] and not J.guards_vs(arm(junction_F1=-0.0301), ref)["holds"]
    g = J.guards_vs(arm(dup_frac=0.015), ref)
    assert g["holds"] and not g["dup_frac_e324_val_bound_ok"], "E324's own VAL bound (0.01) is printed, D79's (0.02) decides"
    assert J.guards_vs(arm(topo_F1=-0.2), ref)["holds"], "the topo F1 gain is the reading's business, not a guard"


# ------------------------------------------------------------------------------------------------------------------------------ comparing numbers and counts
def test_match_and_counts_maxdiff():
    ex, rh = J.EXACT, J.REHEARSAL
    assert J.match(dict(a=1.0, b=dict(c=2.0)), dict(a=1.0, b=dict(c=2.0)), ex)[0]
    ok, worst, where = J.match(dict(a=1.0, b=dict(c=2.0)), dict(a=1.0, b=dict(c=2.5)), ex)
    assert not ok and worst == 0.5 and where == "/b/c"
    assert not J.match(dict(a=1.0), dict(a=1.0, b=2.0), ex)[0], "a number of the stored card that is missing fails"
    assert J.match(dict(n=5774), dict(n=5773), rh)[0] and not J.match(dict(n=5774), dict(n=5773), ex)[0], "integers: relative 5e-4 (E321's 11379 against 11378), exactly under EXACT"
    assert J.match(dict(x=0.6931), dict(x=0.6912), rh)[0] and not J.match(dict(x=0.6931), dict(x=0.6912), ex)[0]
    assert J.match(dict(topo_F1=0.6909), dict(topo_F1=0.6912), rh)[0] and not J.match(dict(topo_F1=0.6931), dict(topo_F1=0.6912), rh)[0], "topo F1 has the tighter tolerance"
    assert not J.match(dict(x=float("nan")), dict(x=1.0), rh)[0]
    c = dict(rec=dict(a=1.0), dp=np.array([1.0, 2.0]), tier=None, lane=dict(m=3))
    assert J.counts_maxdiff(c, dict(rec=dict(a=1.0), dp=np.array([1.0, 2.0]), tier=None, lane=dict(m=3))) == 0.0
    assert abs(J.counts_maxdiff(c, dict(rec=dict(a=1.0), dp=np.array([1.0, 2.0 + 1e-12]), tier=None, lane=dict(m=3))) - 1e-12) < 1e-15
    assert J.counts_maxdiff(c, dict(rec=dict(a=1.0), dp=np.array([1.0, 2.0]), tier=None)) == float("inf"), "another structure"
    assert J.counts_maxdiff([c, c], [c]) == float("inf") and J.counts_maxdiff(np.array([np.nan]), np.array([np.nan])) == 0.0 and J.counts_maxdiff(np.array([np.nan]), np.array([1.0])) == float("inf")


# ------------------------------------------------------------------------------------------------------------------------------ the glue of the scoring: which read-out reads which graph, pair dump and ends
class Frame:
    def __init__(self, row, scene):
        self.row, self.scene, self._lane, self._ref = row, scene, None, None


class Graph:
    def __init__(self, marker):
        self.marker, self.lines, self.edges, self.nodes = marker, [], [], []


def toy_pair_dump(row, off):
    kp = np.array([[off + i, 0.0] for i in range(4)])
    return dict(row=row, kp_pos=kp, kp_dir=np.tile([0.0, 1.0], (4, 1)), att_pos=kp + 100.0, att_dir=np.tile([1.0, 0.0], (4, 1)), logit=np.array([[9.0, -9.0], [-9.0, -9.0]]), allowed=np.ones((2, 2)), scale=None)


def toy_graph_dump(row, marker):
    return dict(row=row, pos=marker, dirv=0, e=0, flags=0, section=0, link_src=0, link_dst=0, link_p=0, link_long=[True], link_a=0, link_b=0)


def test_score_dumps_glue():
    """the variants of E324 s + the stack + the control, on two frames, with the read-out and the scorer replaced by spies: arm w reads the keypoints, p the attached ends with snap_nogeom, k the graph of c2, the control its own pair dump;
    a dump list that two names share (b and ctl) is read out once per frame"""
    import qends_eval as QEV
    import qgraph_eval as QE
    import qgraph_readout as QR
    calls, reads = [], []
    frames = [Frame(10, 0), Frame(11, 0)]
    b = [toy_graph_dump(f.row, "B") for f in frames]
    c2 = [toy_graph_dump(f.row, "C2") for f in frames]
    ends = {"own": {f.row: toy_pair_dump(f.row, 0.0) for f in frames}, "ctl": {f.row: toy_pair_dump(f.row, 50.0) for f in frames}}

    def readout(pred, *a):
        reads.append(pred["marker"])
        g = Graph(pred["marker"])
        g.lines = pred["marker"]                                                                       # the toy graph carries its marker in `lines`: the spy of add_connectors sees which graph the connectors sit on
        return g, dict(kind=[], piece_links=[])

    def add_connectors(lines, edges, nodes, kinds, links, long_mask, pairs, snap=False, snap_nogeom=False, **k):
        assert not k, f"an option the stored cards do not use: {k}"
        calls.append((lines, snap, snap_nogeom, float(pairs[0][0][0]), len(pairs)))
        return dict(pts=[lines]), len(pairs), 1
    variants = [dict(v, man=None) for v in J.variants_of(J.SETUPS["e324_s_s"], True, True)]
    with patched((QR, "apply", lambda pred, spec: dict(marker=pred["nodes"]["pos"], links=dict(long=pred["links"]["long"]))), (QE, "readout", readout), (QE, "to_raw", lambda g: dict(pts=[g.marker])),
                 (QE, "graph_counts", lambda g, f: dict(marker=g.marker if isinstance(g, Graph) else g["pts"][0])), (QEV, "add_connectors", add_connectors)):
        sc = J.score_dumps(frames, {"b": b, "c2": c2, "ctl": b}, ends, variants, log=lambda *a: None)
    want = [("B", True, False, 0.0, 1), ("B", True, True, 0.0, 1), ("B", True, True, 100.0, 1), ("C2", True, False, 0.0, 1), ("B", True, False, 50.0, 1)]      # snapped, snap_nogeom, primary, stack, e312_snapped
    assert calls == want * 2, calls
    assert reads.count("B") == 2 and reads.count("C2") == 2, reads
    assert sc["curves"] == {n: 2 for n in ("snapped", "snap_nogeom", "primary", "stack", "e312_snapped")} == sc["wired"] and sc["man"] == {}
    assert [c["marker"] for c in sc["counts"]["c2_plain"]] == ["C2", "C2"] and [c["marker"] for c in sc["counts"]["plain"]] == ["B", "B"] and [c["marker"] for c in sc["counts"]["stack"]] == ["C2", "C2"]
    calls.clear()                                                                                       # E312's setup: the unsnapped read-out is the only one without snap
    with patched((QR, "apply", lambda pred, spec: dict(marker=pred["nodes"]["pos"], links=dict(long=pred["links"]["long"]))), (QE, "readout", readout), (QE, "to_raw", lambda g: dict(pts=[g.marker])),
                 (QE, "graph_counts", lambda g, f: dict(marker=g.marker if isinstance(g, Graph) else g["pts"][0])), (QEV, "add_connectors", add_connectors)):
        J.score_dumps(frames, {"b": b}, ends, [dict(v, man=None) for v in J.variants_of(J.SETUPS["e312_p_s"])], log=lambda *a: None)
    assert calls == [("B", False, False, 0.0, 1), ("B", True, False, 0.0, 1)] * 2, calls


def test_pairs_use_attached_ends_only_when_asked():
    e = toy_pair_dump(0, 0.0)
    kp, at = J.pairs_of(e, 0.1)[0], J.pairs_of(e, 0.1, attach=True)[0]
    assert np.allclose(kp[0], [0, 0]) and np.allclose(kp[2], [2, 0]) and np.allclose(at[0], [100, 100]) and np.allclose(at[2], [102, 100]) and len(kp) == 4
    assert len(J.pairs_of(dict(e, scale=np.full((2, 2, 2), 0.3)), 0.1)[0]) == 6 and len(J.pairs_of(e, 0.9999)) == 0


# ------------------------------------------------------------------------------------------------------------------------------ the control's checks of one seed, the card, results.md
def toy_numbers(counts):
    cs = list(counts)
    x, d, v = float(np.mean([c["x"] for c in cs])), float(np.mean([c.get("d", 0.2) for c in cs])), float(np.mean([c.get("v", 19.0) for c in cs]))
    return {**{k: x for k in J.KEYS}, "dup_frac": d, "vert100_dp": v}


def toy_sc(variants, xs, n):
    return dict(counts={v["name"]: [dict(x=xs[v["name"]]) for _ in range(n)] for v in variants}, man={v["name"]: [(1, 2, 3, 4)] for v in variants if v["man"]},
                curves={v["name"]: 7 for v in variants if v["ends"]}, wired={v["name"]: 5 for v in variants if v["ends"]})


def toy_card_file(d, head, base):
    p = os.path.join(d, "control_card.json")
    num = lambda x: dict({k: x for k in J.KEYS}, vert100_dp=19.0, dup_frac=0.2)  # noqa: E731
    json.dump(dict(variants=dict(snapped=dict(numbers=dict(head=num(head), base=num(base))))), open(p, "w"))
    return p


def run_toy_seed(d, xs, card_base=0.5, ctl_head=0.6, stack=True, scene_of=lambda r: r % 3):
    """score_seed on toy files (all dumps exist: none may be made) with the scorer replaced; returns what it returns"""
    import qgraph_eval as QE
    full = toy_dump()
    frames = [Frame(r["row"], scene_of(r["row"])) for r in full]
    for name in ("predfull_e273_p_s0.pkl", "ends_e324_s_s0.pkl", "predfull_e292_c2_s0.pkl", "ctl_ends.pkl"):
        pickle.dump(full, open(os.path.join(d, name), "wb"))
    a = argparse.Namespace(split="val", max=0, device="cpu", run_prefix="e324_s_s", b_dump=None, control_ends_dump=os.path.join(d, "ctl_ends.pkl"), control_card=toy_card_file(d, ctl_head, card_base))
    boom = lambda *x, **k: (_ for _ in ()).throw(AssertionError("a dump was made"))  # noqa: E731
    with patched((J, "dump_dir", lambda split, n=0: d), (J, "dump_b", boom), (J, "dump_ends", boom), (J, "pyramid_same", lambda *x: True), (QE, "graph_numbers", toy_numbers),
                 (J, "score_dumps", lambda fr, graphs, ends, variants, thr=J.THR, n=0, log=print: toy_sc(variants, xs, len(fr)))):
        return J.score_seed(a, J.SETUPS["e324_s_s"], 0, frames, stack), frames


XS = dict(plain=0.5, snapped=0.62, snap_nogeom=0.61, primary=0.60, c2_plain=0.52, stack=0.64, e312_snapped=0.6)


def test_score_seed_control_checks():
    with tempfile.TemporaryDirectory() as d:
        r, _ = run_toy_seed(d, XS)                                                                                       # existing dumps reused (dump_b / dump_ends raise when called)
        c = r["control"]
        assert c["plain_check"]["ok"] and c["repro"]["ok"] and c["repro"]["max_abs_diff"] < 1e-12 and c["counts"] is r["sc"]["counts"]["e312_snapped"], "E312's counts reproduce: they are used"
        assert [v["name"] for v in r["variants"]][-1] == "e312_snapped"
        r, _ = run_toy_seed(d, dict(XS, e312_snapped=0.6 + 1e-7))
        assert not r["control"]["repro"]["ok"] and r["control"]["counts"] is None, "a difference of 1e-7 is no reproduction (1e-9): no per-frame counts, no interval"
        assert raises(SystemExit, run_toy_seed, d, XS, card_base=0.51), "the plain graph differs from the control card's numbers.base by 0.01 (VAL tolerance 3e-3): KILL before any number is printed"
        r, _ = run_toy_seed(d, XS, card_base=0.5 + 5e-4)
        assert r["control"]["plain_check"]["ok"], "a fresh CPU dump within the VAL tolerance passes on VAL"
        assert J.tol_of("test") is J.EXACT and J.tol_of("val") is J.REHEARSAL, "TEST: 1e-9 for the plain-graph check"
        r, _ = run_toy_seed(d, XS, stack=False)
        assert "stack" not in r["sc"]["counts"] and "c2_plain" not in r["sc"]["counts"] and all(v["graph"] != "c2" for v in r["variants"]), "--no-stack drops arm k only"


def test_card_arms_and_results_md():
    import qgraph_eval as QE
    setup = J.SETUPS["e324_s_s"]
    with tempfile.TemporaryDirectory() as d, patched((QE, "graph_numbers", toy_numbers)):
        cards = []
        for seed, gain in ((0, 0.016), (1, 0.014)):
            xs = dict(XS, snapped=0.60 + gain, e312_snapped=0.60)
            r, frames = run_toy_seed(d, xs)
            cards.append(J.make_card(seed, f"e324_s_s{seed}", "val", frames, r["sc"], 20, setup, r["variants"], r["control"], True))
        c = cards[0]
        w = c["compare"]["vs_e312"]["w"]
        assert abs(w["difference"]["topo_F1"] - 0.016) < 1e-12 and w["guards"]["holds"] and set(c["compare"]["vs_e312"]) == {"w", "p", "k"} and "k_minus_w" in c["compare"] and "w_minus_p" in c["compare"]
        assert abs(w["paired"]["topo_F1"]["diff"] - 0.016) < 1e-12 and abs(w["paired"]["topo_F1"]["lo"] - 0.016) < 1e-9 and abs(w["paired"]["topo_F1"]["hi"] - 0.016) < 1e-9, "toy: every frame is 0.016 better, so the interval is a point"
        assert abs(c["compare"]["k_minus_w"]["topo_F1"]["diff"] - 0.024) < 1e-12 and abs(c["compare"]["w_minus_p"]["topo_F1"]["diff"] - 0.016) < 1e-12
        assert c["variants"]["stack"]["base"] == "c2_plain" and abs(c["variants"]["stack"]["difference"]["topo_F1"] - 0.12) < 1e-12 and "e312_snapped" not in c["variants"], "k is compared with c2's plain graph; the control is not a variant of the card"
        assert set(c["maneuver"]) == {"plain", "p0.1s", "p0.1an", "c2 plain", "c2 p0.1s"}, "snap_nogeom has no maneuver label"
        assert J.reading([J.reading_inputs(x) for x in cards]) == "FULL RETENTION"
        txt = J.results_md(cards, "val", setup)
        assert "**FULL RETENTION**" in txt and "may become the system of record" in txt and "w - p" in txt and "E2E gap" in txt and "not verified against a stored card" in txt and "snap_nogeom" in txt
        # without E312's per-frame counts: no interval, the fallback label
        cards_nc = []
        for seed, gain in ((0, 0.016), (1, 0.014)):
            r, frames = run_toy_seed(d, dict(XS, snapped=0.60 + gain, e312_snapped=0.6 + 1e-7))
            cards_nc.append(J.make_card(seed, f"e324_s_s{seed}", "val", frames, r["sc"], 20, setup, r["variants"], r["control"], False))
        assert "paired" not in cards_nc[0]["compare"]["vs_e312"]["w"] and J.reading([J.reading_inputs(x) for x in cards_nc]) == "FULL RETENTION (without interval)"
        assert "**FULL RETENTION (without interval)**" in J.results_md(cards_nc, "val", setup)
        assert "not applied" in J.results_md(cards[:1], "val", setup), "the rule needs both seeds"
        assert json.loads(json.dumps(cards[0])), "the card is JSON"


def test_intervals_only_when_the_counts_reproduced_on_every_seed():
    import qgraph_eval as QE
    setup = J.SETUPS["e324_s_s"]
    with tempfile.TemporaryDirectory() as d, patched((QE, "graph_numbers", toy_numbers)):
        ok, frames = run_toy_seed(d, XS)
        bad, _ = run_toy_seed(d, dict(XS, e312_snapped=0.6 + 1e-7))
        assert J.counts_reproduced([ok, ok]) and not J.counts_reproduced([ok, bad]) and not J.counts_reproduced([bad, ok]) and not J.counts_reproduced([bad, bad]), "one seed that does not reproduce: no interval on either"
        c_no = J.make_card(0, "e324_s_s0", "val", frames, ok["sc"], 20, setup, ok["variants"], ok["control"], False)
        c_yes = J.make_card(0, "e324_s_s0", "val", frames, ok["sc"], 20, setup, ok["variants"], ok["control"], True)
        assert "paired" not in c_no["compare"]["vs_e312"]["w"] and "paired" in c_yes["compare"]["vs_e312"]["w"], "use_counts decides, not only the presence of the counts"


def test_inputs_per_split():
    """paths only: which dumps and which control a split reads (the TEST command names B's dump and E312's card; E312's pair dump only by flag; B's dump is never made on TEST)"""
    ns = lambda **k: argparse.Namespace(**dict(dict(split="val", max=0, run_prefix="e324_s_s", b_dump=None, control_ends_dump=None, control_card=None), **k))  # noqa: E731
    setup = J.SETUPS["e324_s_s"]
    r = lambda p: os.path.relpath(p, J.ROOT).replace(os.sep, "/")  # noqa: E731
    i = J.seed_inputs(ns(), setup, 1, True, False)
    assert i["b_made"] and r(i["b"]) == "temp/e343/val/predfull_e273_p_s1.pkl" and r(i["ends"]) == "temp/e343/val/ends_e324_s_s1.pkl" and r(i["c2"]) == "temp/e343/val/predfull_e292_c2_s1.pkl"
    assert r(i["ctl_ends"]) == "runs/e312_p_s1/ends_val.pkl" and r(i["ctl_b"]) == "temp/e276/predfull_e273_p_s1.pkl" and r(i["card"]) == "runs/e312_p_s1/ends_card_snap_thr0.1.json", "VAL: the control is E312's stored dumps and card"
    t = J.seed_inputs(ns(split="test", b_dump="temp/e321/test/predfull_e273_p_s{seed}.pkl", control_card="runs/eval/e312_test/card_s{seed}.json"), setup, 0, True, False)
    assert not t["b_made"] and t["b"] == "temp/e321/test/predfull_e273_p_s0.pkl" and t["ctl_ends"] is None and t["ctl_b"] == t["b"] and t["card"].endswith("runs/eval/e312_test/card_s0.json"), "TEST: no re-score of E312 without the flag"
    assert r(t["ends"]) == "temp/e343/test/ends_e324_s_s0.pkl" and r(t["c2"]) == "temp/e343/test/predfull_e292_c2_s0.pkl", "the new dumps go to temp/e343/test"
    t2 = J.seed_inputs(ns(split="test", control_ends_dump="temp/e321/test/ends_e312_p_s{seed}.pkl"), setup, 1, False, True)
    assert not t2["b_made"] and t2["ctl_ends"] == "temp/e321/test/ends_e312_p_s1.pkl" and t2["ctl_b"] == t2["b"] and t2["c2"] is None and r(t2["card"]) == "runs/eval/e312_test/card_s1.json", "TEST by flag: E312's graph is the same B dump"
    v2 = J.seed_inputs(ns(control_ends_dump="x/ends_{seed}.pkl"), setup, 0, True, True)
    assert v2["ctl_ends"] == "x/ends_0.pkl" and v2["ctl_b"] == v2["b"], "a control pair dump named by a flag is scored on the winner's B dump, on VAL too"
    assert not J.seed_inputs(ns(split="test"), setup, 0, True, False)["b_made"], "B is never dumped on TEST: E321's dump is reused"
    assert r(J.seed_inputs(ns(max=8), setup, 0, True, False)["b"]) == "temp/e343/val_max8/predfull_e273_p_s0.pkl", "a smoke run has its own folder"


def test_cmd_all_end_to_end_on_toy_data():
    """`cmd_all` with the dumps and the scorer replaced: both seeds scored, the intervals dropped on BOTH seeds when one seed's E312 counts did not reproduce, cards and results.md written, E321's hashes checked before and after, the label printed"""
    import io
    import qgraph_eval as QE
    setup = J.SETUPS["e324_s_s"]
    full = toy_dump()
    frames = [Frame(r["row"], r["row"] % 3) for r in full]
    a = argparse.Namespace(split="val", max=873, run_prefix="e324_s_s", seeds=[0, 1], boot=20, no_stack=False, device="cpu", b_dump=None, control_ends_dump=None, control_card=None, test_id="")

    def result(seed, e312, counts):
        variants = J.variants_of(setup, True, True)
        sc = toy_sc(variants, dict(XS, snapped=0.60 + (0.016 if seed == 0 else 0.014), e312_snapped=e312), len(frames))
        control = dict(card="control.json", numbers=toy_numbers([dict(x=0.6)]), plain_check=dict(max_abs_diff=0.0, where="", tolerance="-", ok=True), repro=dict(max_abs_diff=0.0, where="", ok=counts), counts=sc["counts"]["e312_snapped"] if counts else None)
        return dict(seed=seed, run=f"e324_s_s{seed}", sc=sc, variants=variants, control=control)

    def run(counts_of_seed1):
        guards, buf = [], io.StringIO()
        with tempfile.TemporaryDirectory() as d, patched((J, "frames_of", lambda split: frames), (J, "score_seed", lambda a_, st, seed, fr, stack: result(seed, 0.6, seed == 0 or counts_of_seed1)), (J, "out_dir", lambda st, split, n=0: d),
                                                          (J, "guard_e321", guards.append), (QE, "graph_numbers", toy_numbers)):
            old, sys.stdout = sys.stdout, buf
            try:
                rc = J.cmd_all(a)
            finally:
                sys.stdout = old
            out = (rc, sorted(os.listdir(d)), json.load(open(os.path.join(d, "card_s0.json"))), open(os.path.join(d, "results.md")).read())
        return out + (guards, buf.getvalue())
    rc, files, card0, md, guards, text = run(True)
    assert rc == 0 and files == ["card_s0.json", "card_s1.json", "results.md"] and guards == ["before the run", "after the run"] and "paired" in card0["compare"]["vs_e312"]["w"]
    assert "**FULL RETENTION**" in md and "READING: FULL RETENTION" in text, text[-300:]
    rc, files, card0, md, guards, text = run(False)
    assert "paired" not in card0["compare"]["vs_e312"]["w"] and "**FULL RETENTION (without interval)**" in md and "NO interval against E312 on any seed" in text, "seed 1 did not reproduce: no interval on seed 0 either"


def test_variants_are_the_stored_readouts():
    """the options of every variant equal the options of the stored VAL card it reproduces (`score --snap`, `--snap --snap-nogeom`, `--attach --snap --snap-nogeom`): the stored cards' spec fields are checked in `verify` when the file has them"""
    v = {x["name"]: x for x in J.variants_of(J.SETUPS["e324_s_s"], True, True)}
    assert (v["snapped"]["snap"], v["snapped"]["nogeom"], v["snapped"]["attach"]) == (True, False, False)
    assert (v["snap_nogeom"]["snap"], v["snap_nogeom"]["nogeom"], v["snap_nogeom"]["attach"]) == (True, True, False)
    assert (v["primary"]["snap"], v["primary"]["nogeom"], v["primary"]["attach"]) == (True, True, True)
    assert (v["stack"]["graph"], v["stack"]["snap"], v["stack"]["base"]) == ("c2", True, "c2_plain") and v["e312_snapped"]["ends"] == "ctl" and v["e312_snapped"]["graph"] == "ctl"
    e = {x["name"]: x for x in J.variants_of(J.SETUPS["e312_p_s"])}
    assert [x["man"] for x in e.values()] == ["plain", "p0.1", "p0.1s"] and (e["unsnapped"]["snap"], e["snapped"]["snap"]) == (False, True) and not any(x["attach"] or x["nogeom"] for x in e.values()), "E312's variants stay as they were"


def test_check_stored_reads_the_stored_cards():
    """`check_stored` on a toy stored card in a temporary run folder: equal passes, a number off by 1e-6 fails EXACT but passes the rehearsal tolerance, curves must be equal under EXACT, a stored spec field that differs from the variant's fails"""
    setup = dict(J.SETUPS["e312_p_s"], stored={"snapped": "ends_card_snap_thr0.1.json"}, stored_man=None)
    num = lambda x: {k: x for k in J.KEYS}  # noqa: E731
    new = dict(snapped=dict(base="plain", curves=100, wired_both_ends=60, numbers=dict(base=num(0.5), head=num(0.6)), difference=num(0.1)))
    card = dict(variants=new, maneuver={"plain": dict(recall=0.5)})
    old = dict(curves=100, wired_both_ends=60, numbers=dict(base=num(0.5), head=num(0.6)), difference=num(0.1), thr=0.1, spec="close:1.5@0.5", snap=True, attach=False, attach_r=None, snap_options=[], wire=None)
    with tempfile.TemporaryDirectory() as d, patched((J, "ROOT", d)):
        os.makedirs(os.path.join(d, "runs", "r0"))
        write = lambda o: json.dump(o, open(os.path.join(d, "runs", "r0", "ends_card_snap_thr0.1.json"), "w"))  # noqa: E731
        write(old)
        assert J.check_stored(setup, "r0", 0, card, {"counts": {}}, J.EXACT, [])[0]
        write(dict(old, numbers=dict(base=num(0.5), head=num(0.6 + 1e-6))))
        assert not J.check_stored(setup, "r0", 0, card, {"counts": {}}, J.EXACT, [])[0] and J.check_stored(setup, "r0", 0, card, {"counts": {}}, J.REHEARSAL)[0]
        write(dict(old, curves=11378, wired_both_ends=5968))
        card["variants"]["snapped"].update(curves=11379, wired_both_ends=5969)                                       # E321's fresh CPU dump: 11379 against 11378, 5969 against 5968 (relative 9e-5, 2e-4)
        assert not J.check_stored(setup, "r0", 0, card, {"counts": {}}, J.EXACT, [])[0] and J.check_stored(setup, "r0", 0, card, {"counts": {}}, J.REHEARSAL)[0], "integer counts: equal under EXACT, relative 5e-4 in the rehearsal"
        write(dict(old, curves=11300, wired_both_ends=5968))
        assert not J.check_stored(setup, "r0", 0, card, {"counts": {}}, J.REHEARSAL)[0], "11379 against 11300 curves is outside the relative 5e-4"
        card["variants"]["snapped"].update(curves=100, wired_both_ends=60)
        write(dict(old, snap=False))
        assert not J.check_stored(setup, "r0", 0, card, {"counts": {}}, J.REHEARSAL)[0], "the stored spec says unsnapped, the variant is snapped"


def main():
    n = 0
    for k, f in sorted(globals().items()):
        if k.startswith("test_") and callable(f):
            f()
            n += 1
    print(f"test_jtest_eval: {n} tests passed")


if __name__ == "__main__":
    main()
