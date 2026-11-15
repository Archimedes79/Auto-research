def insert_after(path, startswith_marker, rows):
    lines = open(path, encoding="utf-8").read().split("\n")
    idx = max(i for i, l in enumerate(lines) if l.startswith(startswith_marker))
    for r in reversed(rows):
        lines.insert(idx + 1, r)
    open(path, "w", encoding="utf-8").write("\n".join(lines))


insert_after("evaluation/README.md", "| `jends_eval.py`", [
    "| `jrule_control.py` | E287: the rule control of the pair decision (groups and ranks of the keypoints, a connected exit group per turn class, lane matching by normalised rank, extreme lanes for the turns) scored on the SAME keypoints as a head's dump: pair recall / precision of the rule and of the head. `score --run <runs>`, `selftest`. |",
    "| `readout_replay.py` | E288: replays node read-out specs (`plain`, `close`, `mnms:<R>[s<sigma>]` of `qgraph_readout`) on the stored full prediction dump of B and scores the delivered graph by the lane suite, CPU only. `score --seed <S> --specs ...`, `selftest`. |",
])
g = "tools/maintenance/run_gates.sh"
t = open(g, encoding="utf-8").read()
old = 'run "head: pair head v2 read-out (E287)"                 jends_eval.py selftest\n'
assert old in t
t = t.replace(old, old + 'run "head: rule control of the pair decision (E287)"      jrule_control.py selftest\nrun "head: read-out replay (E288)"                       readout_replay.py selftest\n', 1)
open(g, "w", encoding="utf-8").write(t)
print("ok")
