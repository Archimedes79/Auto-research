p = "/mnt/c/Auto-research/bev_fusion_lane_center/temp/jr/make_paper_report.py"
t = open(p, encoding="utf-8").read()
old = 'wr("03_gt.md", gt)'
new = '''gt = gt.replace("# 3. Required ground truth\n", "# 3. Required ground truth\n\n**In short.** nuScenes provides the HD map, the recorded poses and the camera and radar data, but no lane-graph labels. A generator derives them: directed centre lines with split, merge, stop and exit events, which lane the car really drives, what the cameras can see, and for junctions the zone, the arms and the valid entry-to-exit pairs. The same graph code makes the training targets and the scoring reference, so a number is always measured against the label it was trained on. The token head additionally needs the 2.0 m lattice table, a static assignment of tokens to GT nodes and the link targets (below). The picture shows the labels of one junction frame; the details, the checks and the limits follow.\n", 1)
wr("03_gt.md", gt)'''
assert t.count(old) == 1
t = t.replace(old, new)
old2 = 'wr("05_quant.md", q)'
new2 = '''q = q.replace("## A. The ladder: what each end-to-end step gave (topo F1, VAL, s0 / s1)\n", "## A. The ladder: what each end-to-end step gave (topo F1, VAL, s0 / s1)\n\n![The ladder of end-to-end systems and the token head, topo F1 on VAL for both seeds. The dashed line is the written floor of the gate (plain lane head + 0.01). Look at the token head S1 (left of the line) against the earlier systems; the hatched rows are not measured yet (PENDING).](parts/figures/ladder.png)\n", 1)
wr("05_quant.md", q)'''
assert t.count(old2) == 1
t = t.replace(old2, new2)
open(p, "w", encoding="utf-8").write(t)
print("ok")
