p = "evaluation/README.md"
lines = open(p, encoding="utf-8").read().split("\n")
i = next(k for k, l in enumerate(lines) if l.startswith("| `jmaneuver.py`"))
lines.insert(i + 1, "| `jdiag.py` | the diagnostics that the cards of E307 / E308 / E310 / E313 cite: `common` (pair heads on the SAME GT ends: AUC and recall / precision), `curve` (the connector from the keypoints / the true ends against the GT connector, chord tangent and fitted scales), `kperr` (position / heading error of the matched keypoints), `maps` (two maps arms against each other: keypoint F1, crowded / isolated / far entry recall, zone IoU; `selftest`) |")
open(p, "w", encoding="utf-8").write("\n".join(lines))
p = "tools/maintenance/run_gates.sh"
s = open(p, encoding="utf-8").read()
old = 'run "head: maneuver coverage of a delivered graph (E310)"   jmaneuver.py selftest\n'
assert old in s
s = s.replace(old, old + 'run "head: diagnostics of the junction thread (E307 - E313)"  jdiag.py selftest\n', 1)
open(p, "w", encoding="utf-8").write(s)
import glob, re
def edit(path, pairs):
    t = open(path, encoding="utf-8").read()
    for a, b in pairs:
        t = t.replace(a, b)
    open(path, "w", encoding="utf-8").write(t)
edit("experiments/E307_narrow-end-heat/experiment.md", [("comparison script `temp/gt2/maps_cmp.py`", "comparison `wsl bash tools/env/_py14.sh jdiag.py maps e307_n e286_j1`")])
edit("experiments/E313_heat-sigma-0.7/experiment.md", [("`temp/gt2/maps_cmp.py e313_s e307_n`", "`jdiag.py maps e313_s e307_n`"), ("(`maps_cmp.py e313_s e307_n`, the numbers of `jmaps_eval.py score`)", "(`jdiag.py maps e313_s e307_n`, the numbers of `jmaps_eval.py score`)")])
edit("experiments/E310_maneuver-coverage-and-connector-headroom/experiment.md", [("`temp/gt2/curve_vs_gt.py e308_t_s0`", "`jdiag.py curve e308_t_s0`"), ("and `temp/gt2/curve_vs_gt.py <run>` (scratch)", "and `jdiag.py curve <run>`")])
edit("experiments/E308_pair-head-with-node-tokens/experiment.md", [("`temp/gt2/pair_diag.py` / `pair_diag_common.py`", "`jdiag.py common`"), ("(`temp/gt2/pair_diag.py`, to be promoted into `jpair_pr.py` when used a second time)", "(`jdiag.py common <run>`)"), ("`pair_diag_common.py`", "`jdiag.py common`"), ("scratch scripts `temp/gt2/pair_diag.py` / `jdiag.py common`", "`jdiag.py common`")])
print("ok")
