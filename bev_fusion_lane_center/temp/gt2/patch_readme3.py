def insert_after(path, startswith_marker, rows):
    lines = open(path, encoding="utf-8").read().split("\n")
    idx = max(i for i, l in enumerate(lines) if l.startswith(startswith_marker))
    for r in reversed(rows):
        lines.insert(idx + 1, r)
    open(path, "w", encoding="utf-8").write("\n".join(lines))


insert_after("training/grid/README.md", "| `jmaps.py`", [
    "| `jends.py` | E287 (D82): the junction pair head v2: keypoints from the 1 m entry / exit heat of `jmaps.py`, pair logits with the order-rule features (rank in the entry / exit group, group sizes) and the evidence along the Hermite curve, the soft crossing penalty (non-crossing pairs inside one pair of groups). Same output keys as `qendnet.EndHead`. Test: `python jends.py`. |",
])
insert_after("evaluation/README.md", "| `jmaps_eval.py`", [
    "| `jends_eval.py` | E287: the pair head v2 on the 873 VAL frames: `dump` (keypoints, pair logits, the head's own recall / precision of ends and pairs, `ends_val.pkl`), `score` (the delivered graph of B with the head's connectors by the E273 scorer through `qends_eval.cmd_score`, D79's rule). |",
])
p = "training/README.md"
s = open(p, encoding="utf-8").read()
b = "`jmaps_train.py` (E286:"
assert b in s
s = s.replace(b, "`jends_train.py` (E287: the pair head v2 on the frozen junction maps run), " + b, 1)
open(p, "w", encoding="utf-8").write(s)
g = "tools/maintenance/run_gates.sh"
t = open(g, encoding="utf-8").read()
old = 'run "head: junction pictures (E285)"                     viz/junction_view.py --selftest\n'
assert old in t
t = t.replace(old, old + 'run "head: pair head v2, order penalty (E287)"            grid/jends.py\nrun "head: pair head v2 read-out (E287)"                 jends_eval.py selftest\n', 1)
open(g, "w", encoding="utf-8").write(t)
print("ok")
