"""Fills the E111 placeholders: section text, summary, card, vals JSON, state text."""
import json

T = "/mnt/c/ML-Experiment/bev_fusion_lane_center/reports/source/text"

E111_TEXT = """**Terms used here.** *Seed*: the random initialisation and the random order of the training states; everything else is the recipe of E94a (10 000 iterations, batch 16 x 32 states, the same code and data; at inference 192 seeds per frame, thresholds 0.3 / 0.5, duplicate suppression 5). Seed 0 is the network of loops 10 and 11, seed 1 the second run of this loop (it used three data-loader workers instead of five, which changes the speed, not the recipe). *Paired difference*: scene-level bootstrap over the same rows (95 % interval).

**Hypothesis.** Everything about the learned extractor in loops 10 and 11 rests on one training run. If a second run differed by as much as the gap to the rule pipeline (F1 -0.008 on VAL, -0.017 on TEST) or as the effects compared in those loops (0.01 - 0.02), those comparisons could not be decided. For the dense head the seed noise was 0.0006 - 0.0025 F1 (E55). **Method.** The trace network is trained again with seed 1 and evaluated exactly as E94a on all VAL (873) and TEST (833) frames: point F1 at 1 m, precision, recall, split / merge F1 at 6 m, direction accuracy and route F of the graph; paired differences between the two runs and against the rule pipeline's loose and final graph. The route head (E104b) and the memory block were not retrained or re-run for the second run.

[[table: tab_e111]]

[[fig: report_final/figures/p12_e111.png | E111: point F1 at 1 m, direction accuracy and split F1 at 6 m of the two runs of the trace network (bars with scene-level 95 % intervals) against the rule pipeline's loose (red) and final (green) graph, VAL and TEST.]]

**Result.** (1) **The two runs agree on the lane graph to within 0.002 F1.** VAL 0.883 and 0.884 (paired difference seed 1 - seed 0 +0.001 [-0.004, +0.006]), TEST 0.851 and 0.850 (-0.001 [-0.005, +0.005]); precision and recall differ by at most 0.005. (2) **Direction and route F agree as well:** direction accuracy 0.886 / 0.883 (VAL) and 0.863 / 0.862 (TEST), paired -0.002 [-0.011, +0.006] and 0.000 [-0.006, +0.006]; route F 0.744 / 0.742 (VAL) and 0.668 / 0.667 (TEST). (3) **The junction metrics vary more:** split F1 at 6 m 0.357 / 0.341 (VAL) and 0.385 / 0.375 (TEST), merge F1 0.459 / 0.441 and 0.428 / 0.424; paired differences between -0.003 and -0.017, all intervals contain 0. (4) **The gap to the rule pipeline is the same for both runs:** seed 1 against the loose graph F1 -0.008 [-0.017, +0.002] (VAL) and -0.018 [-0.027, -0.009] (TEST) (seed 0: -0.008 [-0.015, 0.000], -0.017), against the final graph -0.015 and -0.026, direction -0.061 and -0.070 against the loose graph; split / merge F1 better than the loose graph (+0.056 / +0.053 split) and far worse than the final graph (-0.347 / -0.340), as for seed 0.

**Conclusion.** The seed noise of the trace network is small: about 0.001 - 0.002 F1 and direction accuracy and 0.01 - 0.02 on the junction metrics, an order of magnitude below the gap to the rule pipeline (-0.017 F1 on TEST). **The gap of the learned graph in loops 10 and 11 is not seed noise**, and F1 comparisons of 0.01 or more between inference variants of one network are decidable with respect to the seed; the junction metrics are only decidable above about 0.02, and nothing here says anything about the seed noise of the route head, the memory block or the route success of the chain (the route head was trained on the tokens of the first run). **Limits.** Two runs of one recipe: the spread of the seed distribution is estimated from one difference, so the 0.001 - 0.002 is a rough size; the scene-level intervals (+-0.035 F1 for one network) are much wider than the seed difference, the paired intervals of the two runs on the same scenes are +-0.005; the second run used another number of data-loader workers.
"""

E111_SUM = ("A second run of the trace network with the same recipe (seed 1) gives the same lane graph to within 0.002 F1 (VAL 0.884 against 0.883, TEST 0.850 against 0.851; paired differences within [-0.005, +0.006]) and the same direction accuracy (0.883 / 0.862 against 0.886 / 0.863); the junction metrics differ by 0.003 - 0.017 (all intervals contain 0). "
            "The seed noise of the learned extractor is therefore small against its gap to the rule pipeline (F1 -0.017 on TEST), which is not seed noise; the route head and the memory block were not retrained for the second run.")

c = open(f"{T}/04g_loop12_c.md", encoding="utf-8").read()
assert c.count("@@E111T@@") == 1 and c.count("@@E111S@@") == 1
c = c.replace("@@E111T@@", E111_TEXT.strip("\n")).replace("@@E111S@@", E111_SUM)
open(f"{T}/04g_loop12_c.md", "w", encoding="utf-8").write(c)

V = dict(
    e111_par="A second run of the trace network (seed 1) reproduces the first to within 0.002 F1, so the gap of the learned graph to the rule pipeline (-0.017 F1 on TEST) is not seed noise.",
    e111_row="a second run of the trace network reproduces F1 to within 0.002 (seed noise small against the 0.017 gap to the rule graph)",
    e111_lim="The seed noise of the trace network rests on two runs (the route head and the memory block were not retrained for the second run; the second run used three data-loader workers instead of five).",
    e111_88="F1 0.884 / 0.850 (VAL / TEST) against 0.883 / 0.851 for the first run, paired differences within [-0.005, +0.006]; direction 0.883 / 0.862 against 0.886 / 0.863; junction F1 differ by 0.003 - 0.017",
    e111_88b="the F1 gap of the learned graph to the rule pipeline (-0.017 on TEST) is not seed noise; junction metrics are only decidable above about 0.02",
)
json.dump(V, open("/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/misc/_text12_vals.json", "w"), indent=1)

# card E111
card = '''    dict(
        id="E111", title="Seed noise of the learned trace network: a second training run with the same recipe",
        loop=12, verdict="kept",
        literature=["E55 (seed repetition of the dense head, loop 4)", "E94 (the trace network, loop 10)"],
        architecture="The trace network E94a (1.67 M parameters) trained again with seed 1 (10 000 iterations, batch 16 x 32 states, the same code, data and recipe; three data-loader workers instead of five) and evaluated like E94a on all VAL (873) and TEST (833) frames: point F1 at 1 m, split / merge F1 at 6 m, direction accuracy, route F; paired scene-level bootstrap between the two runs and against the rule pipeline's loose and final graph. The route head E104b and the memory block were not retrained.",
        hypothesis="If a second run differed by as much as the gap to the rule pipeline (-0.008 F1 VAL, -0.017 TEST), the loop-10 and loop-11 comparisons of the learned graph rested on chance. Status: refuted (the seed noise is about 0.001 - 0.002 F1, an order of magnitude below the gap).",
        result="Seed 0 / seed 1, VAL: F1 at 1 m 0.883 / 0.884, direction 0.886 / 0.883, split F1 0.357 / 0.341, merge F1 0.459 / 0.441, route F 0.744 / 0.742; TEST: 0.851 / 0.850, 0.863 / 0.862, 0.385 / 0.375, 0.428 / 0.424, 0.668 / 0.667. Paired difference seed 1 - seed 0: F1 +0.001 [-0.004, +0.006] (VAL), -0.001 [-0.005, +0.005] (TEST); direction -0.002 [-0.011, +0.006], 0.000 [-0.006, +0.006]; split F1 -0.016 [-0.039, +0.009], -0.011 [-0.038, +0.023]; merge F1 -0.017 [-0.043, +0.007], -0.003 [-0.023, +0.016]. Seed 1 against the rule pipeline's loose graph: F1 -0.008 [-0.017, +0.002] (VAL), -0.018 [-0.027, -0.009] (TEST); against the final graph -0.015 and -0.026.",
        conclusion_quantitative="The seed noise of the lane graph is about 0.001 - 0.002 F1 and direction accuracy and 0.003 - 0.017 on the junction metrics (all intervals contain 0), an order of magnitude below the gap to the rule pipeline (-0.017 F1 on TEST); the gap is the same for both runs.",
        conclusion_qualitative="The disadvantage of the learned graph against the rule graph in loops 10 and 11 is not seed noise, and F1 comparisons of 0.01 or more between inference variants of one network are decidable with respect to the seed; the junction metrics are only decidable above about 0.02. Two runs give a rough size of the noise, not a distribution; the route head, the memory block and the route success of the chain were not repeated.",
        figures=["report_final/figures/p12_e111.png"], captions=["Point F1 at 1 m, direction accuracy and split F1 at 6 m of the two runs of the trace network against the rule pipeline's loose and final graph, VAL and TEST (scene-level 95 % intervals)."],
        source="scripts/p10_train.py (--seed 1), p10_infer.py, p10_final_eval.py, p12_seed_pair.py, fig_p12.py seeds; out/loop2/e94_s1 (final_eval.json, seed_pair_vs_e94_a.json); section 7.35",
    ),
]
'''
p = "/mnt/c/ML-Experiment/bev_fusion_lane_center/scripts/_mkcards12.py"
s = open(p, encoding="utf-8").read()
assert s.count("\n]\njson.dump(cards") == 1
s = s.replace("\n]\njson.dump(cards", "\n" + card.rstrip("\n").rstrip("]").rstrip("\n").rstrip(",") + ",\n]\njson.dump(cards") if False else s.replace("\n]\njson.dump(cards", "\n" + card.rsplit("]", 1)[0] + "]\njson.dump(cards")
open(p, "w", encoding="utf-8").write(s)

st = open("/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/misc/_state12.txt", encoding="utf-8").read()
st = st.replace("@@E111STATE@@", "E111 second run of the trace network (seed 1, same recipe, 3 workers): VAL / TEST F1@1m 0.884 / 0.850 vs 0.883 / 0.851 (paired -0.005 .. +0.006), direction 0.883 / 0.862 vs 0.886 / 0.863, junction F1 differ 0.003-0.017 (n.s.): seed noise ~0.001-0.002 F1, the -0.017 TEST gap to the rule graph is not seed noise (e94_s1, p12_seed_pair.py).")
open("/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/misc/_state12.txt", "w", encoding="utf-8").write(st)
print("ok")
