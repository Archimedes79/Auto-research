p = "evaluation/qends_eval.py"
s = open(p, encoding="utf-8").read()


def sub(old, new):
    global s
    assert old in s, old[:80]
    s = s.replace(old, new, 1)


sub("        w_in = w_out = None\n        if snap:\n            pa, ta, pb, tb", "        w_in = w_out = None\n        default_wire = True                                                                                     # wire every piece in reach (the E284 / E287 way); the snapped read-out wires only the chosen piece per end\n        if snap:\n            default_wire = snap_all\n            pa, ta, pb, tb")
sub("                    pa, ta, pb, tb = pk\n                    w_in = w_out = None", "                    pa, ta, pb, tb = pk\n                    w_in = w_out = None\n                    default_wire = True")
sub("        if snap and not snap_all:\n            if w_in is not None:", "        if not default_wire:\n            if w_in is not None:")
# card file name: the unsnapped card of record only for thr 0.5, never overwritten by another threshold
sub('''    snap = bool(getattr(a, "snap", False))
    out["snap"] = snap
    json.dump(out, open(os.path.join(run_dir, f"ends_card_snap_thr{a.thr}.json" if snap else "ends_card.json"), "w"), indent=1)       # E311: a snapped read-out never overwrites the card of record''',
    '''    snap = bool(getattr(a, "snap", False))
    out["snap"] = snap
    opts = [k for k in ("snap_all", "snap_guard", "snap_dedup") if getattr(a, k, False)]
    out["snap_options"] = opts
    # the card of record (`ends_card.json`) is the unsnapped read-out at the threshold 0.5 only: another threshold or a snapped read-out gets its own file (the E308 incident of 2026-11-12: a scratch sweep emptied the card of record)
    name = "ends_card.json" if (not snap and a.thr == 0.5) else "ends_card_" + ("snap" + "".join("_" + k[5:] for k in opts) + "_" if snap else "") + f"thr{a.thr}.json"
    json.dump(out, open(os.path.join(run_dir, name), "w"), indent=1)''')
sub('pairs, snap=bool(getattr(a, "snap", False)))', 'pairs, snap=bool(getattr(a, "snap", False)), snap_all=bool(getattr(a, "snap_all", False)), snap_guard=bool(getattr(a, "snap_guard", False)), snap_dedup=bool(getattr(a, "snap_dedup", False)))')
sub('    s.add_argument("--snap", action="store_true", help="E311: draw the connectors from the end of the delivered piece wired to the entry to the start of the piece wired to the exit (the keypoints only decide which entry joins which exit)")',
    '    s.add_argument("--snap", action="store_true", help="E311: draw the connectors from the end of the delivered piece wired to the entry to the start of the piece wired to the exit (the keypoints only decide which entry joins which exit); the card goes to ends_card_snap_thr<t>.json")\n    for k, h in (("--snap-all", "with --snap: wire every piece in reach of the snapped end"), ("--snap-guard", "with --snap: a degenerate snapped curve keeps its keypoints"), ("--snap-dedup", "with --snap: no second curve between the same two snapped pieces")):\n        s.add_argument(k, action="store_true", help=h)')
# usage lines at the top
sub("    wsl bash tools/env/_py14.sh qends_eval.py score --run e284_diag_s0 --seed 0 [--thr 0.5]", "    wsl bash tools/env/_py14.sh qends_eval.py score --run e284_diag_s0 --seed 0 [--thr 0.5] [--snap [--snap-all] [--snap-guard] [--snap-dedup]]")
open(p, "w", encoding="utf-8").write(s)
p = "evaluation/jends_eval.py"
s = open(p, encoding="utf-8").read()
old = 'snap=bool(getattr(a, "snap", False))))'
assert old in s
s = s.replace(old, 'snap=bool(getattr(a, "snap", False)), snap_all=bool(getattr(a, "snap_all", False)), snap_guard=bool(getattr(a, "snap_guard", False)), snap_dedup=bool(getattr(a, "snap_dedup", False))))', 1)
old = '            s.add_argument("--snap", action="store_true", help="E311:'
i = s.index(old)
j = s.index("\n", i)
s = s[:j + 1] + '            for k_ in ("--snap-all", "--snap-guard", "--snap-dedup"):\n                s.add_argument(k_, action="store_true", help="refinement of --snap (see qends_eval.add_connectors)")\n' + s[j + 1:]
open(p, "w", encoding="utf-8").write(s)
print("ok")
