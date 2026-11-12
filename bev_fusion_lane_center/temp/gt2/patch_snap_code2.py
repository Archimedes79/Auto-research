p = "evaluation/qends_eval.py"
s = open(p, encoding="utf-8").read()
old = '    json.dump(out, open(os.path.join(run_dir, "ends_card.json"), "w"), indent=1)'
assert old in s
s = s.replace(old, '    snap = bool(getattr(a, "snap", False))\n    out["snap"] = snap\n    json.dump(out, open(os.path.join(run_dir, f"ends_card_snap_thr{a.thr}.json" if snap else "ends_card.json"), "w"), indent=1)       # E311: a snapped read-out never overwrites the card of record', 1)
open(p, "w", encoding="utf-8").write(s)
p = "evaluation/jends_eval.py"
s = open(p, encoding="utf-8").read()
old = 'chord=bool(getattr(a, "chord", False))))'
assert old in s
s = s.replace(old, 'chord=bool(getattr(a, "chord", False)), snap=bool(getattr(a, "snap", False))))', 1)
old = '            s.add_argument("--chord", action="store_true", help="E303: the chord-tangent connectors even when the dump has learned handle scales (the shape control: same pairs)")'
assert old in s
s = s.replace(old, old + '\n            s.add_argument("--snap", action="store_true", help="E311: connectors drawn from the end of the delivered piece wired to the entry to the start of the piece wired to the exit (writes ends_card_snap_thr<t>.json, the card of record stays)")', 1)
open(p, "w", encoding="utf-8").write(s)
print("ok")
