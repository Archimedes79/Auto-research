import subprocess
row = "| `jtime.py` | E299 step 0 (time for the junction head, CPU, VAL): `ceiling` (the true pairs delivered at t-1 and missed at t, split into logit-only flips and end-miss flips by the predicted-keypoint association across the two frames, the flip rate), `replay --w 0.5` (the zero-parameter average of the pair logit with the ego-warped logit of the counterpart pair of the previous keyframe: pair precision at the recall of the original at logit 0 and -2.2, snapped topo F1 of the E312 read-out), `selftest`; needs the GT ends of `jwiring.py gt-ends` (labels only) |\n"
anchor = "| `readout_replay.py` | E288:"
head = subprocess.run(["git", "show", "HEAD:bev_fusion_lane_center/evaluation/README.md"], capture_output=True, text=True, encoding="utf-8", cwd="..").stdout
assert head.count(anchor) == 1
open("temp/e299/README_blob.md", "w", encoding="utf-8").write(head.replace(anchor, row + anchor))
cur = open("evaluation/README.md", encoding="utf-8").read()
assert cur.count(anchor) == 1
open("evaluation/README.md", "w", encoding="utf-8").write(cur.replace(anchor, row + anchor))
print("ok")
