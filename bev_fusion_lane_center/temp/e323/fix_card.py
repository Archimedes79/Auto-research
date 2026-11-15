p = "experiments/E323_duplicates-operating-point/experiment.md"
s = open(p, encoding="utf-8").read()
R = [
 ("0.11 (seed 1, 11845)", "0.111 (seed 1, 11845)"),
 ("runs/e320_a_s1/ends_card_snap_nogeom_attach_thr0.11.json`; against", "runs/e320_a_s1/ends_card_snap_nogeom_attach_thr0.111.json`; against"),
 ("0.083, 0.09, 0.12, 0.11. The counts files", "0.083, 0.09, 0.12, 0.111. The counts files"),
 ("E320 at E312's count 0.120 / 0.110 (11383 / 11845 against 11378 / 11849, +0.04 % / -0.03 %)", "E320 at E312's count 0.120 / 0.111 (11383 / 11845 against 11378 / 11849, +0.04 % / -0.03 %; the first version of this card printed 0.110 for seed 1, a display rounding of the count script: 0.110 delivers 11885 curves)"),
 ("`runs/e320_a_s1/ends_card_snap_nogeom_attach_thr0.11.json` (each with its `.counts.pkl`)", "`runs/e320_a_s1/ends_card_snap_nogeom_attach_thr0.111.json` (each with its `.counts.pkl`)"),
 ("| a(t2 = 0.11) |", "| a(t2 = 0.111) |"),
 ("| 11849 | 12209 | 12209 | 12208 | 11885 |", "| 11849 | 12209 | 12209 | 12208 | 11845 |"),
 ("| 0.6864 | 0.6868 | 0.6762 | 0.6889 | 0.6888 |", "| 0.6864 | 0.6868 | 0.6762 | 0.6889 | 0.6889 |"),
 ("| 0.1557 | 0.1562 | 0.1617 | 0.1651 | 0.1641 |", "| 0.1557 | 0.1562 | 0.1617 | 0.1651 | 0.1640 |"),
 ("| 0.3143 | 0.3153 | 0.2972 | 0.3005 | 0.3011 |", "| 0.3143 | 0.3153 | 0.2972 | 0.3005 | 0.3008 |"),
 ("| 20.10 | 20.08 | 20.25 | 19.84 | 19.87 |", "| 20.10 | 20.08 | 20.25 | 19.84 | 19.88 |"),
 ("+0.0024 [-0.0081, +0.0105]; +0.0084 [+0.0025, +0.0143] held", "+0.0025 [-0.0080, +0.0107]; +0.0083 [+0.0025, +0.0141] held"),
 ("| -0.0254 held | -0.0132 held |", "| -0.0254 held | -0.0135 held |"),
 ("0.1919 (-0.0012 against its own 0.1931) and 0.1641 (-0.0010)", "0.1919 (-0.0012 against its own 0.1931) and 0.1640 (-0.0011)"),
 ("0.1951 / 0.1656 predicted, 0.1919 / 0.1641 read", "0.1951 / 0.1656 predicted, 0.1919 / 0.1640 read"),
]
for a, b in R:
    n = s.count(a)
    print(n, a[:60])
    assert n >= 1, a
    s = s.replace(a, b)
open(p, "w", encoding="utf-8").write(s)
