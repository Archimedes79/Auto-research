p = "evaluation/README.md"
lines = open(p, encoding="utf-8").read().split("\n")
i = next(k for k, l in enumerate(lines) if l.startswith("| `jpair_pr.py`"))
lines.insert(i + 1, "| `jmaneuver.py` | E310: maneuver coverage of a delivered lane graph: a GT entry -> exit maneuver (connector run of the lane GT) counts as realised when 90 % of its points have a delivered point within 1.5 m in the same direction (40 degrees); variants plain / connectors at p >= t / oracle pairs / oracle + fitted handle scales; the number that matches the pictures next to the identity-strict pair numbers (`score --run --seed [--variants ...] [--rows]`, `selftest`) |")
open(p, "w", encoding="utf-8").write("\n".join(lines))
p = "tools/maintenance/run_gates.sh"
s = open(p, encoding="utf-8").read()
old = 'run "head: pair precision at a fixed recall (E302)"       jpair_pr.py selftest\n'
assert old in s
s = s.replace(old, old + 'run "head: maneuver coverage of a delivered graph (E310)"   jmaneuver.py selftest\n', 1)
open(p, "w", encoding="utf-8").write(s)
print("ok")
