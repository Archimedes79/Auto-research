P = "/mnt/c/Auto-research/bev_fusion/experiments/t26_one_long_run/experiment.md"
s = open(P).read()

# 1 the base bullet -> measured
a = s.index("  - **Where the base stands**")
b = s.index("\n", a)
new_base = ("  - **Where the base stands, MEASURED 2026-11-15 (G0, GT v13, the 1620 common frames of 43 scenes, claim region, cold, CPU: `t26.ref_maps_base`, `temp/t26/ref_maps_base/out.md`; ours = `runs/t20_base_s0` ckpt 6000 EMA, trained on GT v9, scored on v13 with `--allow-gt-mismatch`; d1 = ours - reference at threshold 0.5 [95 % scene bootstrap], d2 = each at its own best threshold, in-sample):** "
            "drivable IoU@0 0.842 / 0.884, d1 -0.041 [-0.056, -0.022], d2 -0.041; pedestrian crossing 0.289 / 0.642, d1 -0.352 [-0.403, -0.312]; walkway 0.427 / 0.668, d1 -0.241 [-0.278, -0.211], d2 -0.234; car park 0.453 / 0.714, d1 -0.261 [-0.356, -0.183], d2 -0.271; "
            "lane divider IoU@0.5 0.477 / 0.764, d1 -0.286 [-0.337, -0.234], d2 -0.362 [-0.412, -0.307]; road divider IoU@0.5 0.605 / 0.714, d1 -0.109 [-0.171, -0.050], d2 -0.171 [-0.225, -0.119] (n_gt cells 4.40 M / 241 k / 993 k / 97 k / 144 k / 40 k). The base is BEHIND on all six rows by the written rule except that drivable area (d1 -0.041 against the margin 0.030, interval wholly below 0) is BEHIND too; the gaps are LARGER than the orientation table of the first draft (GT v8, t16 network, 0 - 60 m: lane divider -0.235, crossing -0.266): inside the claim region the reference is better still. The old-region reproduction row of the same run (0 - 60 m clipped, as table 1) gives lane divider 0.464 / 0.694 and drivable 0.823 / 0.844, in line with the stored v8 table (0.455 / 0.690, 0.818 / 0.858). "
            "**GT shift of the base (v9 -> v13), VAL 1000 (`t26.base_eval`, `temp/t26/base_run/eval/by_n/card_n1000.json`, evaluation GT `0.5_4_a107e0b4e8`):** `detf` mAP 0.4092 -> 0.4092, `det` mAP 0.3414 -> 0.3414, lane divider IoU@0.5 0.4256 -> 0.4257, road divider 0.4985 -> 0.4985, drivable 0.7878 -> 0.7878, walkway 0.3962 -> 0.3962, crossing 0.2616 -> 0.2616, car park 0.2943 -> 0.2943 (-0.0001): the headline rows do not move with the GT version; the velocity row was not in that evaluation (no `--velocity`) and is made by `t26.base_eval_vel` (the velocity rows are where v13 differs: stationary flag, flow). The base on the 43 scenes (1620 frames, `t26.base_eval43`): `detf` mAP 0.4459, `det` 0.3782, lane divider IoU@0.5 0.4637, drivable 0.8232 (a different frame set, not comparable with VAL 1000). "
            "The reference gains from its best threshold by up to +0.073 (lane divider) and +0.069 (road divider): in the d2 column.")
s = s[:a] + new_base + s[b:]

# 2 forecast
a = s.index("- **Prediction (the lead's")
b = s.index("\n", a)
new_pred = ("- **Prediction (the lead's, written before the run; restated 2026-11-15 on the measured base gaps).** Against the base: `detf` mAP +0.03 (0.00 .. +0.07). Measured gaps to GaussianCaR (d1 of the base, section above): crossing 0.352, walkway 0.241, car park 0.261, lane divider 0.286 (d2 0.362), road divider 0.109 (d2 0.171), drivable 0.041. Share of the gap closed: crossing / walkway / car park / lane divider 40 % (20 .. 60 %), road divider and drivable 50 % (30 .. 70 %). That predicts d1 after the run: crossing -0.211 (-0.282 .. -0.141), walkway -0.145 (-0.193 .. -0.096), car park -0.157 (-0.209 .. -0.104), lane divider -0.172 (-0.229 .. -0.114), road divider -0.055 (-0.076 .. -0.033), drivable -0.021 (-0.029 .. -0.012). "
            "**Expected verdict: BEHIND on crossing, walkway, car park, lane divider and road divider; drivable NOT SEPARATED** (its predicted d1 lies inside the margin 0.030 but not its interval). The object row against CRN: the forecast is written after the CRN gates and the CRN score of the base, before the launch. The lead's chance for AT SOA is about 2 %, for BEYOND SOA below 1 % (the measured gaps are larger than the first draft assumed). The statement is expected to fail; the run measures by how much. "
            "The learning-curve read-outs at 6000 / 12000 / 24000 / 36000 steps say whether the share closed grows with the budget: the claim that matters is not AT SOA but the slope.")
s = s[:a] + new_pred + s[b:]

# 3 run id appended
x = '"run_ids": ["t26.base_eval", "t26.base_eval43", "t26.ref_maps_base"]}},'
assert s.count(x) == 1
s = s.replace(x, '"run_ids": ["t26.base_eval", "t26.base_eval43", "t26.ref_maps_base", "t26.base_eval_vel"]}},')
y = "`t26.base_eval43` and `t26.one_s0.eval43`"
if y in s:
    pass
z = "`t26.base_eval43` (the same networks on the 1620 frames of the 43 common scenes)"
open(P, "w").write(s)
print("ok", len(s))
