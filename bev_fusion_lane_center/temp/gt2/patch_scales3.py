p = "evaluation/jends_eval.py"
s = open(p, encoding="utf-8").read()
old = "    QE.cmd_score(SimpleNamespace(run=a.run, seed=a.seed, thr=a.thr, full=None, base=None, skip_base_check=True))"
new = "    QE.cmd_score(SimpleNamespace(run=a.run, seed=a.seed, thr=a.thr, full=None, base=None, skip_base_check=True, chord=bool(getattr(a, \"chord\", False))))"
assert old in s
s = s.replace(old, new, 1)
old = "        if name == \"score\":\n            s.add_argument(\"--seed\", type=int, required=True)"
new = "        if name == \"score\":\n            s.add_argument(\"--seed\", type=int, required=True)\n            s.add_argument(\"--chord\", action=\"store_true\", help=\"E303: the chord-tangent connectors even when the dump has learned handle scales (the shape control: same pairs)\")"
assert old in s
s = s.replace(old, new, 1)
open(p, "w", encoding="utf-8").write(s)
print("ok")
