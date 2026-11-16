import subprocess, sys
row = "| `jdup.py` | E324 diagnosis: where the duplicate points of `dup_frac` come from: the counting of `topo_metrics.dup_counts` with every duplicate point attributed (connector / B piece: CC, CB, BC, BB; for CB the arc length from the connector's end and the kind of the B piece), split by the truth of the connector (true pair / false pair) and by the pair probability, for the snapped read-out of E312 and the E320 PRIMARY read-out (`score --seed`, `selftest`; CPU, VAL, about a minute; needs the GT ends of `jwiring.py gt-ends`); the totals reproduce the dup_frac of the cards |\n"
anchor = "| `readout_replay.py` | E288:"
def put(text):
    assert text.count(anchor) == 1
    return text.replace(anchor, row + anchor)
p = "evaluation/README.md"
cur = open(p, encoding="utf-8").read()
open(p, "w", encoding="utf-8").write(put(cur))
head = subprocess.run(["git", "show", "HEAD:bev_fusion_lane_center/evaluation/README.md"], capture_output=True, text=True, encoding="utf-8", cwd="..").stdout
open("temp/e324/README_blob.md", "w", encoding="utf-8").write(put(head))
print("ok")
