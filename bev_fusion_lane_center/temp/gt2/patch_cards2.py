def sub(path, old, new, count=1):
    s = open(path, encoding="utf-8").read()
    assert old in s, (path, old[:90])
    open(path, "w", encoding="utf-8").write(s.replace(old, new, count))

p = "experiments/E302_capacitated-pair-assignment/experiment.md"
sub(p, "loss = NLL of the true pairs under P + the degree CE + the crossing penalty of E287; read-out at P >= 0.5 x the row mass",
    "pair probability q = P / max(1, min(row mass, column mass)) (the share of the smaller capacity the pair uses, in [0, 1]), loss = BCE (pos_weight 3) on q + 0.5 x the degree CE (classes 0, 1, 2, 3+; 0 = a keypoint that is no GT end) + the crossing penalty of E287, no dual softmax; read-out q >= 0.5 (= logit >= 0 of the dump, so the E287 read-out and counts are unchanged)")
s = open(p, encoding="utf-8").read()
s = s.replace("## 0. The card (what the tools run)", "Written after the first code review of 2026-11-12 (what the review changed: the plan entry is a mass and is divided by the smaller capacity before it is read as a probability; the card first said \"NLL under P, read-out at 0.5 x the row mass\"). The control p2 and this arm are both trained after commit 15f30cbc (the rank features come from real peaks in training now: an older p2 is not a valid control).\n\n## 0. The card (what the tools run)", 1)
open(p, "w", encoding="utf-8").write(s)

p = "experiments/E305_e2e-maps-and-pairs/experiment.md"
s = open(p, encoding="utf-8").read()
s = s.replace("## 0. The card (what the tools run)", "Control and variant are both trained after commit 15f30cbc (real-peak rank features); the maps head and the pair head share one optimiser (lr 1e-3, one-cycle, one global gradient clip of 3.0): a pretrained head and a fresh one under one schedule is part of what the card tests, the kill criterion covers a drift of the maps.\n\n## 0. The card (what the tools run)", 1)
open(p, "w", encoding="utf-8").write(s)

p = "experiments/E306_exact-polygon-targets/experiment.md"
sub(p, "of 3125 cells that lie more than half inside the polygon, the legacy fill gets 1122 wrong (36 %), the new area rule (`jgt.fill_poly`, `BFLC_RASTER=exact`, 8 x 8 supersampling) 71 (2.3 %).",
    "of 3125 cells that lie more than half inside the polygon, the legacy fill gets 1122 wrong (36 %), the new supersampled area rule (`jgt.fill_poly`, `BFLC_RASTER=exact`, 8 x 8 supersampling; \"exact\" is relative to the legacy fill: the code review found about 3 % of the cells differ from the analytic area rule) 71 (2.3 %).")
sub(p, "`BFLC_RASTER=exact` selects the area rule (a cell is in when more than half of it is inside, thin paint: more than a quarter).", "`BFLC_RASTER=exact` selects the area rule (a cell is in when more than half of it is inside; a polygon thinner than one cell - 2 x area / perimeter - gets the quarter rule so that a thin stop line stays visible; on 1573 VAL stop lines the median width is 3.7 m: almost all use the half rule).")
print("ok")
