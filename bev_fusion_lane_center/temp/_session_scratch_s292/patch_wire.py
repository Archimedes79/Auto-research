"""E325 step 1 (scratch, scripted text surgery on COPIES in temp/_session_scratch_s292/new2/): `qends_eval.add_connectors(wire_choice=...)` and `cmd_score --wire twin`, with `jwire.kept_ends` as the one place that
computes the kept pieces and their ends. Apply AFTER the E292 code critic is answered (it reads the same files); the source is the LIVE file at the time of the run.
usage (from the problem folder): python3 temp/_session_scratch_s292/patch_wire.py"""
import os

NEW = "temp/_session_scratch_s292/new2"
os.makedirs(NEW, exist_ok=True)


def sub(path, pairs):
    s = open(path, encoding="utf-8").read()
    for a, b in pairs:
        assert s.count(a) == 1, (path, a[:80], s.count(a))
        s = s.replace(a, b)
    open(os.path.join(NEW, os.path.basename(path)), "w", encoding="utf-8", newline="\n").write(s)


p = "evaluation/qends_eval.py"
s = open(p, encoding="utf-8").read()
i0 = s.index("            d_in = d_out = np.inf\n            for i, (le, te, fs, ts) in enumerate(ends):\n")
i1 = s.index("            if snap_both and (w_in is None or w_out is None):\n")
old_search = s[i0:i1]
assert old_search.count("\n") == 8, old_search
new_search = ("            if wire_choice is not None:                                                                  # E325: the pieces are CHOSEN by the caller (a head, or the zero-parameter twin of the rule)\n"
              "                w_in, w_out = wire_choice[n]\n"
              "                assert all(w is None or (0 <= w < len(ends)) for w in (w_in, w_out)), f\"wire_choice[{n}] = {(w_in, w_out)}: an index outside the {len(ends)} kept pieces\"\n"
              "            else:\n"
              + "".join("    " + l + "\n" for l in old_search.rstrip("\n").split("\n")))

sub(p, [
    ("import qends as QEND  # noqa: E402\n", "import jwire as JW  # noqa: E402\nimport qends as QEND  # noqa: E402\n"),
    ("snap_both=False, wire_pos=None):\n", "snap_both=False, wire_pos=None, wire_choice=None):\n"),
    ("    wp = WIRE_POS if wire_pos is None else float(wire_pos)\n",
     "    # E325: wire_choice = None (the nearest-piece rule, bit for bit) or ONE (entry piece index or None, exit piece index or None) per pair, the indices into the KEPT pieces (`jwire.kept_ends`: after the long-link\n"
     "    # cover drop below); needs snap. A head or the zero-parameter twin (`jwire.twin_choice`) fills it; the rest of the snapped read-out (nogeom, guard, dedup, both) is unchanged.\n"
     "    assert wire_choice is None or (snap and len(wire_choice) == len(pairs)), \"wire_choice needs snap and one entry per pair\"\n"
     "    wp = WIRE_POS if wire_pos is None else float(wire_pos)\n"),
    ("    keep = []\n    for i, k in enumerate(kinds):\n        if k == \"link\" and bool(long_mask[links[i][0]]):\n            P = np.asarray(lines[i], float)\n            if any(np.linalg.norm(P[0] - pr[0]) <= cp_ and np.linalg.norm(P[-1] - pr[2]) <= cp_ for pr in pairs):\n                continue\n        keep.append(i)\n",
     "    keep, ends = JW.kept_ends(lines, kinds, links, long_mask, pairs, wp, cp_)\n"),
    ("    ends = [(P[-1], _tan(P, True), P[0], _tan(P, False)) for P in L2]\n", ""),
    (old_search, new_search),
    # cmd_score: --wire twin
    ("        raw, n_c, n_w = add_connectors(graph.lines, graph.edges, graph.nodes, bk[\"kind\"], bk[\"piece_links\"], np.asarray(pred[\"links\"][\"long\"], bool), pairs, snap=",
     "        wc = None\n"
     "        if wire == \"twin\":                                                                                                      # E325 arm t: the zero-parameter twin chooses the pieces; must reproduce the card of record bit for bit\n"
     "            wc = JW.twin_choice(JW.kept_ends(graph.lines, bk[\"kind\"], bk[\"piece_links\"], np.asarray(pred[\"links\"][\"long\"], bool), pairs, WIRE_POS, COVER_POS)[1], pairs, WIRE_POS)\n"
     "        raw, n_c, n_w = add_connectors(graph.lines, graph.edges, graph.nodes, bk[\"kind\"], bk[\"piece_links\"], np.asarray(pred[\"links\"][\"long\"], bool), pairs, wire_choice=wc, snap="),
    ("    spec = QR.parse(spec_s)\n", "    spec = QR.parse(spec_s)\n    wire = getattr(a, \"wire\", None)\n    assert wire in (None, \"twin\") and (wire is None or (bool(getattr(a, \"snap\", False)) and getattr(a, \"attach_r\", None) is None)), \"--wire twin needs --snap and no --attach-r\"\n"),
    ("    assert not (full or base or spec) or tag, \"--full / --base / --spec score another graph than B's: give a --tag (it names the card, so that the card of record of the run stays)\"\n",
     "    assert not (full or base or spec or getattr(a, \"wire\", None)) or tag, \"--full / --base / --spec / --wire score another graph than the card of record: give a --tag (it names the card, so that the card of record of the run stays)\"\n"),
    ("    out[\"tag\"] = tag\n", "    out[\"tag\"] = tag\n    out[\"wire\"] = wire\n"),
    ("    s.add_argument(\"--tag\", default=None, help=\"E292: names the card",
     "    s.add_argument(\"--wire\", default=None, choices=(None, \"twin\"), help=\"E325: choose the wired pieces through add_connectors(wire_choice=...): twin = the zero-parameter twin of the nearest-piece rule (needs --snap and --tag); must reproduce the E312 cards\")\n"
     "    s.add_argument(\"--tag\", default=None, help=\"E292: names the card"),
])
sub("evaluation/jends_eval.py", [
    ("tag=getattr(a, \"tag\", None), skip_base_check=True,", "tag=getattr(a, \"tag\", None), wire=getattr(a, \"wire\", None), skip_base_check=True,"),
    ("            s.add_argument(\"--tag\", default=None, help=\"E292: names the card",
     "            s.add_argument(\"--wire\", default=None, choices=(None, \"twin\"), help=\"E325: the zero-parameter twin of the nearest-piece rule chooses the wired pieces (needs --snap and --tag)\")\n"
     "            s.add_argument(\"--tag\", default=None, help=\"E292: names the card"),
])
print("patched:", sorted(os.listdir(NEW)))
