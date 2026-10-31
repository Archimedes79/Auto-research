p = "temp/loop21/fig_status_frames.py"
s = open(p, encoding="utf-8").read()
R = [
 ('"1  input: BEV features dec32 (frozen GaussianCaR,\nmean |value| of 32 channels, 0.5 m cells)"', '"1  BEV features (frozen), mean |value|"'),
 ('"2  dense lane heat (dense head on the features)"', '"2  dense lane heat"'),
 ('"3  dense direction (hue = heading, brightness = heat)"', '"3  dense direction (hue = heading)"'),
 ('f"4  builder hypothesis tree: {len(s0)} lattice seeds, each with child nodes;\ncolour = path validity pi, grey = pi below tau {tau} (not used)"', 'f"4  hypothesis tree, colour = path validity"'),
 ('"5  junction head: split heat (red), merge heat (blue);\ngreen = ground truth with its splits / merges"', '"5  junction heat: split red, merge blue"'),
 ('f"6  assembled graph (path union, anchoring, pruning):\n{nv(G[\'recall\'][i])} vertices (node ids)"', 'f"6  assembled: {nv(G[\'recall\'][i])} vertices"'),
 ('f"7  after the output stage simplify_dp 0.2 m:\n{nv(G[\'recall_dp\'][i])} vertices"', 'f"7  after simplify_dp: {nv(G[\'recall_dp\'][i])} vertices"'),
 ('f"camera front, VAL frame {i}, {LOCN[int(loc[i])]}\n(green ground truth, red recall_dp: for orientation)"', 'f"camera, frame {i}, {LOCN[int(loc[i])]}"'),
 ('ax2.set_title(ttl + f", F1@1.0 {f1(g, gts[i]):.2f}", fontsize=9)', 'ax2.set_title(ttl + f", F1 {f1(g, gts[i]):.2f}", fontsize=9)'),
 ('fig.suptitle("Inside the network: system `recall` on one VAL frame (ego frame, x right, y forward, +-50 m; 1 - 3 inputs, 4 - 5 builder outputs, 6 - 7 assembly)", fontsize=10)', 'fig.suptitle("system recall on one VAL frame (ego frame, +-50 m): 1 - 3 inputs, 4 - 5 builder outputs, 6 - 7 assembly", fontsize=10)'),
]
for a, b in R:
    assert s.count(a) == 1, a[:50]
    s = s.replace(a, b)
open(p, "w", encoding="utf-8").write(s)
print("ok")
