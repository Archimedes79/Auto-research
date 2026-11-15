"""E325 step 1: answer the code critic (findings F1 - F6). Scripted edits of the LIVE files, written as .new_s292 next to them; the caller moves them over after looking at the result.
usage (problem folder): python3 temp/_session_scratch_s292/patch_fix2.py"""
import re


def sub(path, pairs):
    s = open(path, encoding="utf-8").read()
    for a, b in pairs:
        assert s.count(a) == 1, (path, a[:90], s.count(a))
        s = s.replace(a, b)
    return s


def write(path, s):
    open(path + ".new_s292", "w", encoding="utf-8", newline="\n").write(s)


# ---------------------------------------------------------------- qends_eval.py
p = "evaluation/qends_eval.py"
s = sub(p, [
    ("COVER_POS = 4.0\n",
     "COVER_POS = 4.0\n\n\n"
     "def cover_pos(wire_pos=None):\n"
     "    \"\"\"the radius within which a connector's ends cover (drop) a long link of B: COVER_POS, and under a tight wiring radius the smaller of the two (N1, critic 2026-11-13); the one place `add_connectors` and the\n"
     "    caller that builds the kept list for a `wire_choice` (E325) read it\"\"\"\n"
     "    return COVER_POS if wire_pos is None else min(COVER_POS, float(wire_pos))\n"),
    ("    assert wire_choice is None or (snap and len(wire_choice) == len(pairs)), \"wire_choice needs snap and one entry per pair\"\n",
     "    assert wire_choice is None or (snap and not snap_all and not snap_guard and len(wire_choice) == len(pairs)), \"wire_choice needs snap and one entry per pair, and neither snap_all (it wires every piece in reach) nor snap_guard (it drops the choice)\"\n"),
    ("    keep, ends = JW.kept_ends(lines, kinds, links, long_mask, pairs, wp, cp_)\n", "    keep, ends = JW.kept_ends(lines, kinds, links, long_mask, pairs, cp_)\n"),
    ("                assert all(w is None or (0 <= w < len(ends)) for w in (w_in, w_out)), f\"wire_choice[{n}] = {(w_in, w_out)}: an index outside the {len(ends)} kept pieces\"\n",
     "                assert all(w is None or (isinstance(w, (int, np.integer)) and not isinstance(w, bool) and 0 <= w < len(ends)) for w in (w_in, w_out)), f\"wire_choice[{n}] = {(w_in, w_out)}: not an integer index into the {len(ends)} kept pieces\"\n"
     "                # the gate (reach, heading) is NOT checked here: the caller chooses among the kept pieces, the gate is the input side of its choice (`jwire.candidates`)\n"),
    # --wire checks next to score_target (before the dump and the checkpoints are loaded)
    ("    wire = getattr(a, \"wire\", None)\n    assert wire in (None, \"twin\") and (wire is None or (bool(getattr(a, \"snap\", False)) and getattr(a, \"attach_r\", None) is None)), \"--wire twin needs --snap and no --attach-r\"\n", ""),
    ("    name, spec_s, tag = score_target(a, run_dir)                                                                        # before anything long runs\n",
     "    name, spec_s, tag = score_target(a, run_dir)                                                                        # before anything long runs\n"
     "    wire = getattr(a, \"wire\", None)\n"
     "    assert wire in (None, \"twin\") and (wire is None or (bool(getattr(a, \"snap\", False)) and not getattr(a, \"snap_all\", False) and not getattr(a, \"snap_guard\", False) and getattr(a, \"attach_r\", None) is None)), \"--wire twin needs --snap, no --attach-r, no --snap-all / --snap-guard\"\n"),
    ("pairs, WIRE_POS, COVER_POS)[1], pairs, WIRE_POS)", "pairs, cover_pos())[1], pairs, WIRE_POS, WIRE_ANG)"),
])
# add_connectors: the cover radius through the helper
m = re.search(r"    cp_ = COVER_POS if wire_pos is None else min\(COVER_POS, wp\)( +#.*)?\n", s)
assert m, "cp_ line"
s = s.replace(m.group(0), "    cp_ = cover_pos(wire_pos)" + (m.group(1) or "") + "\n")
write(p, s)

# ---------------------------------------------------------------- jwire.py
p = "training/grid/jwire.py"
s = sub(p, [
    ("    keep, ends = kept_ends(lines, kinds, links, long_mask, pairs, wire_pos)   # the pieces that survive the long-link cover drop of `qends_eval.add_connectors` and their ends (last point, heading there,\n",
     "    keep, ends = kept_ends(lines, kinds, links, long_mask, pairs, cover_pos)  # the pieces that survive the long-link cover drop of `qends_eval.add_connectors` and their ends (last point, heading there,\n"),
    ("def kept_ends(lines, kinds, links, long_mask, pairs, wire_pos, cover_pos):\n", "def kept_ends(lines, kinds, links, long_mask, pairs, cover_pos):\n"),
    ("def candidates(ends, pos, tan, entry, wire_pos, wire_ang=45.0):\n", "def candidates(ends, pos, tan, entry, wire_pos, wire_ang=None):\n"),
    ("    import qends_eval as QE\n    out = []\n    pos, tan = np.asarray(pos, float), np.asarray(tan, float)\n",
     "    import qends_eval as QE\n    wire_ang = QE.WIRE_ANG if wire_ang is None else wire_ang                                          # the module value at call time (a sweep may set QE.WIRE_ANG)\n    out = []\n    pos, tan = np.asarray(pos, float), np.asarray(tan, float)\n"),
    ("def twin_pick(ends, pos, tan, entry, wire_pos, wire_ang=45.0):\n", "def twin_pick(ends, pos, tan, entry, wire_pos, wire_ang=None):\n"),
    ("def twin_choice(ends, pairs, wire_pos, wire_ang=45.0):\n", "def twin_choice(ends, pairs, wire_pos, wire_ang=None):\n"),
    ("def rule_pick(ends, pos, tan, entry, wire_pos, wire_ang=45.0):\n", "def rule_pick(ends, pos, tan, entry, wire_pos, wire_ang=None):\n"),
    ("    import qends_eval as QE\n    best, d_best = None, np.inf\n", "    import qends_eval as QE\n    wire_ang = QE.WIRE_ANG if wire_ang is None else wire_ang\n    best, d_best = None, np.inf\n"),
    ("The gate (reach 4 m, 45 degrees, at most 8 places) is an index choice and stays.",
     "The gate (reach 4 m, 45 degrees, at most 8 places) is an index choice and stays; `add_connectors(wire_choice=)` does not check it (the caller chooses among the kept pieces). Because the candidates are sorted nearest first and the twin's\nlogits fall with the distance, the twin's argmax is always the first candidate and the cut at 8 never matters for it (the softmax is the place where a head's logits come in)."),
    # tests
    ("        _, ends = kept_ends(lines, kinds, links, lm, [], R, R)\n        pos = rng.uniform(-8, 8, 2)\n        tan = np.array([np.cos(a := rng.uniform(0, 2 * np.pi)), np.sin(a)])\n        for entry in (True, False):\n            r, t = rule_pick(ends, pos, tan, entry, R), twin_pick(ends, pos, tan, entry, R)\n            assert r == t, (r, t, entry)\n            n_cmp += 1\n            n_some += r is not None\n    assert n_some > 100, n_some                                                                          # the comparison is not vacuous\n",
     "        _, ends = kept_ends(lines, kinds, links, lm, [], R)\n        for entry in (True, False):\n            pos = rng.uniform(-8, 8, 2)\n            tan = np.array([np.cos(a := rng.uniform(0, 2 * np.pi)), np.sin(a)])\n            if lines and rng.random() < 0.7:                                                               # near a piece end, roughly along it: several candidates in reach\n                k_ = int(rng.integers(len(lines)))\n                pos = np.asarray(lines[k_][-1] if entry else lines[k_][0], float) + rng.normal(0, 1.5, 2)\n                t0, th = QE._tan(np.asarray(lines[k_], float), entry), rng.normal(0, 0.3)\n                tan = np.array([t0[0] * np.cos(th) - t0[1] * np.sin(th), t0[0] * np.sin(th) + t0[1] * np.cos(th)])\n            r, t = rule_pick(ends, pos, tan, entry, R), twin_pick(ends, pos, tan, entry, R)\n            assert r == t, (r, t, entry)\n            n_cmp += 1\n            n_some += r is not None\n            n_multi += len(candidates(ends, pos, tan, entry, R)) >= 2\n    assert n_some > 300 and n_multi > 150, (n_some, n_multi)                                             # the comparison is not vacuous: many ends with a candidate, many with two or more\n"),
    ("    n_cmp = n_some = 0\n", "    n_cmp = n_some = n_multi = 0\n"),
    ("    assert len(candidates(far, [0.0, 0.0], [0.0, 1.0], True, R)) == MAX_CAND and twin_pick(far, [0.0, 0.0], [0.0, 1.0], True, R) == 0\n",
     "    assert len(candidates(far, [0.0, 0.0], [0.0, 1.0], True, R)) == MAX_CAND and twin_pick(far, [0.0, 0.0], [0.0, 1.0], True, R) == 0\n"
     "    assert twin_pick(far[::-1], [0.0, 0.0], [0.0, 1.0], True, R) == 11 and candidates(far[::-1], [0.0, 0.0], [0.0, 1.0], True, R)[0][1] == 11      # nearest LAST in the list: the sort comes before the cut\n"),
    ("    keep, ends = kept_ends(lines, kinds, links, lm, pr, R, R)\n    assert keep == [0, 1] and len(ends) == 2 and np.allclose(ends[0][0], lines[0][-1]) and np.allclose(ends[1][2], lines[1][0])\n    assert kept_ends(lines, kinds, links, lm, [], R, R)[0] == [0, 1, 2]\n",
     "    keep, ends = kept_ends(lines, kinds, links, lm, pr, R)\n    assert keep == [0, 1] and len(ends) == 2 and np.allclose(ends[0][0], lines[0][-1]) and np.allclose(ends[1][2], lines[1][0])\n    assert kept_ends(lines, kinds, links, lm, [], R)[0] == [0, 1, 2]\n"
     "    # the cover radius is the parameter and it is pinned: the link ends 3.9 m from the pair's ends are covered at 4.0 m, not at 3.5 m; 4.1 m away they are not covered at 4.0 m\n"
     "    for off, rad, dropped in ((3.9, 4.0, True), (3.9, 3.5, False), (4.1, 4.0, False)):\n"
     "        prs = [(lines[2][0] + np.array([0.0, off]), np.array([0.0, 1.0]), lines[2][-1] + np.array([0.0, off]), np.array([0.0, 1.0]))]\n"
     "        assert (2 not in kept_ends(lines, kinds, links, lm, prs, rad)[0]) == dropped, (off, rad)\n"
     "        assert (2 not in kept_ends(lines, kinds, links, lm, prs, QE.cover_pos(rad))[0]) == dropped\n"
     "    assert QE.cover_pos() == QE.COVER_POS and QE.cover_pos(1.0) == 1.0 and QE.cover_pos(9.0) == QE.COVER_POS\n"),
    ("    for bad, kwargs in ((\"a wire_choice of the wrong length\",",
     "    # a choice that is NOT the rule gives the intended graph: two pieces ending at (0, -1) (nearest) and (0, -3), the entry keypoint at (0, 0): choosing piece 1 starts the curve at its end and wires it, not piece 0\n"
     "    c_lines = [np.c_[np.zeros(6), np.linspace(-20, -1, 6)], np.c_[np.zeros(6), np.linspace(-30, -3, 6)]]\n"
     "    c_args = (c_lines, [], [[0], [1]], [\"lane\", \"lane\"], [(0,), (1,)], np.zeros(2, bool), [(np.array([0.0, 0.0]), np.array([0.0, 1.0]), np.array([10.0, 10.0]), np.array([0.0, 1.0]))])\n"
     "    r_rule, r_pick = QE.add_connectors(*c_args, snap=True)[0], QE.add_connectors(*c_args, snap=True, wire_choice=[(1, None)])[0]\n"
     "    assert (0, 2) in r_rule[\"edges\"] and np.allclose(r_rule[\"pts\"][2][0], [0.0, -1.0])                  # the rule: the nearest piece\n"
     "    assert (1, 2) in r_pick[\"edges\"] and (0, 2) not in r_pick[\"edges\"] and np.allclose(r_pick[\"pts\"][2][0], [0.0, -3.0]) and not any(e[0] == 2 for e in r_pick[\"edges\"]), r_pick[\"edges\"]\n"
     "    assert (0, 2) not in QE.add_connectors(*c_args, snap=True, wire_choice=[(None, None)])[0][\"edges\"] and np.allclose(QE.add_connectors(*c_args, snap=True, wire_choice=[(None, None)])[0][\"pts\"][2][0], [0.0, 0.0])        # a None keeps the keypoint and wires nothing\n"
     "    for bad, kwargs in ((\"snap_all together with a wire_choice\", dict(snap=True, snap_all=True, wire_choice=[(None, None)])), (\"snap_guard together with a wire_choice\", dict(snap=True, snap_guard=True, wire_choice=[(None, None)])),\n"
     "                        (\"a float index\", dict(snap=True, wire_choice=[(1.0, None)])), (\"a bool index\", dict(snap=True, wire_choice=[(True, None)]))):\n"
     "        try:\n"
     "            QE.add_connectors(*c_args, **kwargs)\n"
     "        except AssertionError:\n"
     "            continue\n"
     "        raise AssertionError(f\"accepted {bad}\")\n"
     "    for bad, kwargs in ((\"a wire_choice of the wrong length\","),
    ("; {n_graphs} graphs through add_connectors, {n_covered} with a covered long link)\")", "; {n_multi} of them with two or more candidates; {n_graphs} graphs through add_connectors, {n_covered} with a covered long link)\")"),
])
write(p, s)

# ---------------------------------------------------------------- README (jends row)
p = "evaluation/README.md"
s = sub(p, [("(the pyramid of the maps run must be the pyramid of `--base`: asserted)", "(the pyramid of the maps run must be the pyramid of `--base`: asserted); E325: `--wire twin`")])
write(p, s)
print("patched")
