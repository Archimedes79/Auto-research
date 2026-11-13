p = "evaluation/README.md"
s = open(p, encoding="utf-8").read()
old = "| `readout_replay.py` | E288:"
assert old in s
new = ("| `jsplits.py` | E300: where a junction pair head fails: end found / pair recall of a dump split by the attributes of the GT (driven or not, city, entries in use in the maneuver, turn class, range of the entry, painted, visible); strict one-to-one matching at `--kp-thr`. `score --run <runs>`, `selftest`. |\n"
       "| `jpair_pr.py` | E302: pair precision at a fixed pair recall: the precision-recall curve of a pair head from its dump (`--kp-thr`, strict matching) and the precision of one run at the recall of the control's operating point (`--control`). `score --run <run> [--control <run>]`, `selftest`. |\n")
s = s.replace(old, new + old, 1)
s = s.replace("scored on the SAME keypoints as a head's dump: pair recall / precision of the rule and of the head. `score --run <runs>`, `selftest`. |", "scored on the SAME keypoints as a head's dump (same keypoint threshold `--kp-thr`, strict one-to-one matching): pair recall / precision of the rule and of the head and the ceiling (both ends found). `score --run <runs>`, `selftest`. |")
open(p, "w", encoding="utf-8").write(s)
p = "tools/maintenance/run_gates.sh"
s = open(p, encoding="utf-8").read()
s = s.replace('run "head: failure splits of the pair head (E300)"        jsplits.py selftest\n', 'run "head: failure splits of the pair head (E300)"        jsplits.py selftest\nrun "head: pair precision at a fixed recall (E302)"       jpair_pr.py selftest\n')
open(p, "w", encoding="utf-8").write(s)
print("ok")
