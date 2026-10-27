"""Week 3 report (loops 18-20, 2026-10-24 .. 2026-10-27): HTML builder.  Run in gcar:  tools/env/_py14.sh build_week3_report.py   (after fig_w3_results.py)
Numbers that come from result files are read here (cards in runs/eval, reports/source/weekly/week3_numbers.json written by fig_w3_results.py); numbers of earlier
reviewed summaries are quoted with the experiment id in the text and are checked by the numbers reviewer against the experiment summaries.
Output: reports/weekly/Week3_Loops18-20_Report_EN.html  (PDF: reports/build/reports/print_weekly.ps1)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import weekly_lib as W  # noqa: E402

R = W.ROOT
F = "reports/source/figures/"
OUT = R + "/reports/weekly/Week3_Loops18-20_Report_EN.html"
os.makedirs(os.path.dirname(OUT), exist_ok=True)
NUM = json.load(open(R + "/reports/source/weekly/week3_numbers.json"))
T = W.Doc.tag
OK, NEG, TIE, INFO = (lambda t: T("ok", t)), (lambda t: T("neg", t)), (lambda t: T("tie", t)), (lambda t: T("info", t))


def card(name, tag=""):
    p = f"{R}/runs/eval/{name}/val{'_' + tag if tag else ''}.json"
    return json.load(open(p, encoding="utf-8-sig"))["metrics"] if os.path.exists(p) else None


def f3(x):
    return "n/a" if x is None else f"{x:.3f}"


def f4(x):
    return "n/a" if x is None else f"{x:.4f}"


seeds = NUM.get("e240_seeds", [])
nS = len(seeds)
qrc, qeq = NUM["e240_rc_Q"], NUM["e240_eq_Q"]
lrc, leq = NUM["e240_rc_lanes_found"], NUM["e240_eq_lanes_found"]
dq = [a - b for a, b in zip(qrc["values"], qeq["values"])]
dl = [a - b for a, b in zip(lrc["values"], leq["values"])]
mean = lambda v: sum(v) / len(v)  # noqa: E731

d = W.Doc("Measure first: the seed floor, the matched budget, and what the ground truth can and cannot explain",
          "Week 3 of the lane-centre-line graph project, loops 18 to 20: the last gains of loop 18, the measurement work of loop 19 (noise floor, equal vertex budgets, a data fix, "
          "a replicate of the map benefit) and the four iterations of loop 20 that also tested the research framework itself.",
          "Week 3 · Loops 18 - 20 · 24 - 27 October 2026",
          "Camera + radar BEV features (nuScenes) → sparse lane graph → map assignment → planner. Numbers are validation (VAL, 873 frames, 22 scenes) unless marked TEST.")

# ------------------------------------------------------------------ 1 at a glance
d.h1("Contents", new_page=False, toc=False)
d.raw("<!--TOC-->")
d.h1("1  At a glance", new_page=False)
d.p("Weeks 1 and 2 built the chain and made it differentiable; loop 18 closed its biggest weaknesses (junctions, the map, the speed). In the rest of week 3 we stopped adding blocks and "
    "<b>measured what our numbers are worth</b>: how large the seed noise is, how much of an apparent gain is only a different vertex budget, whether a damaged training target matters, "
    "whether the map benefit survives other error draws, and whether the map offsets of the ground truth cap the metric. Most answers are ties or negatives; the measurement floor itself is the main result.")
d.stats([
    (f"±{(qrc['sd'] + qeq['sd']) / 2:.4f}","seed SD of Q (E240)", f"{nS} paired seeds of the recall fine-tune on VAL; the band of 0.010 is about 4 SDs. Lanes found SD {(lrc['sd'] + leq['sd']) / 2:.4f}"),
    (f"+{mean(dl):.3f} / +{mean(dq):.3f}", "recall weight 3: lanes found / Q", f"j1rc minus j1eq, mean of {nS} pairs: lanes found higher in every pair, Q +{mean(dq):.3f} below the tool band 0.010 (TIE)"),
    ("0 of 4", "ideas that beat the plain fine-tune this week", "exact distance map (E237), clamp 8 m (E244 arm b), per-range thresholds (E246 killed at step 0), Poisson set loss (E255: Q -0.073 at an equal vertex budget)"),
    ("51 %", "train frames with a damaged distance target", "the stored map broke the 0.5 m bound in 4,315 of 8,414 train frames (TF32); fine-tuning on the exact map changed Q by -0.0017 (E237)"),
    ("+0.02 … +0.04", "F1@1 gain of map registration + trust", "on two new map-error draws, failure half; all six cells above zero (E238)"),
    ("0.4 m = Q -0.05", "price of a ground-truth shift", "shifting all ground-truth lines by 0.3 / 0.5 / 1.0 m costs Q 0.023 / 0.070 / 0.215 (E252); the Boston ground truth is not worse than Singapore's"),
])
d.h2("What we learned this week")
d.ul([
    "<b>The noise band is no longer a guess.</b> Seed SD of Q about 0.0025 (SD of the paired difference 0.0043) and of lanes found about 0.004 were measured on four paired fine-tunes (E240). The 0.010 band is 4 seed SDs and 2.3 paired SDs for Q; "
    "a single pair cannot see anything below about 0.01.",
    "<b>The recall weight is a recall lever, not a quality lever.</b> Lanes found +0.05 to +0.07 and R@1 far +0.04 to +0.05 in every one of four pairs; Q +0.006 +- 0.002 (standard error), positive in all pairs but below the 0.010 band (TIE by the rule, not shown to be zero). Compared at the same vertex budget the "
    "recall variant's Q edge over the champion shrinks from +0.0054 to +0.0025 (E236): <b>every Q comparison must state vertices per 100 m</b>, because Q falls by about 0.0025 per 0.25 vertices.",
    "<b>A damaged training target (TF32 distance map, 51 % of the frames) cost the fine-tune nothing measurable</b> (E237); Mirko's rule is still to train on the exact map from now on, without retraining the foundation (D14).",
    "<b>The map benefit is not an artefact of one error draw</b> (E238): +0.018 … +0.042 F1@1 on two new draws of the 2 m map error, intervals above zero.",
    "<b>Four loss and threshold ideas of the cycle were killed or tied</b> (clamp, per-range thresholds, soft targets not trained, the Poisson set loss): the far-recall gap is not a threshold or a clamp problem. The Poisson set likelihood looked fine at its own 18.7 vertices per 100 m and lost 0.073 Q once held to the same 14.3 vertices as the controls (E255): <b>the matched vertex budget decides such verdicts</b>.",
    "<b>The map offsets that other work found on dividers do not show on our lane centrelines</b> (E252): Boston scores higher than Singapore, correct predictions sit closer to the ground truth in Boston, per-line medians are small.",
    "<b>The research framework held, and it also hurt in specific places</b>: the derived-state tools (D16) worked, a stalled launcher could be resumed, and a shared machine was the largest source of lost time: the planned timing repeat (E253) had to wait five hours for another chat's 7-hour training to end (section 6), and the session was valid once it ran.",
])
d.callout("How reliable are these numbers?",
          "Everything is on the 22 VAL scenes; the seed floor is estimated from four pairs, each comparison is one or two seeds, tuning of tau and the vertex radius is done on VAL subsets and said so. "
          "TEST was not touched in loops 19 and 20 (the one check of loop 18, on 25 October, is logged in <code>evaluation/TEST_USAGE_LOG.md</code>). Every summary was read by a fresh numbers reviewer; the corrections are in section 9.", "warn")

# ------------------------------------------------------------------ 2 setup
d.h1("2  Setup and rules", new_page=True)
d.p("<b>Task and data.</b> From frozen GaussianCaR camera + radar BEV features a differentiable builder predicts a sparse graph of lane centre lines (forks, merges, direction) within ±50 m; "
    "the map assignment and the planner consumer sit behind it. nuScenes, 255 scenes of Boston and Singapore split by scene: TRAIN 8,414 / VAL 873 (22 scenes) / TEST 833 frames. "
    "VAL contains two locations only: Boston-seaport (10 scenes, 396 frames) and Singapore One-North (12 scenes, 477 frames).")
d.fig(F + "p18_arch.png", "The chain (unchanged in loops 19 and 20; drawing of loop 18). Grey = frozen or cached, blue = learned and differentiable, green = memory, orange = new in loop 18, hatched = rule step at inference.")
d.p("<b>Metrics.</b> Headline: <b>Q</b> (the composite S without the sparsity term) at about 14.3 vertices per 100 m, topology F1, lanes found, R@1 far (beyond 40 m), route success, ms per frame. "
    "Rules of the week: a Q comparison names its vertex budget; seed noise is measured, not assumed; a verdict comes from the written rule of the experiment card through <code>decide.py</code> where the card has arms with metric cards.")
d.p("<b>The improver.</b> Experiments run through one launcher (<code>run.py</code>: run record, code tag, chains with <code>--after</code>), the schedule and the loop table are derived from the run records (D16, deployed 26 October), "
    "every card is checked by a referee before its run, every summary by a fresh numbers reviewer.")

# ------------------------------------------------------------------ 3 loop 18
d.h1("3  Loop 18 in brief (24 - 25 October)", new_page=True)
d.p("Loop 18 has its own 12-page report (<i>Loop18_Fork_Pruning_Localisation_Report_EN.pdf</i>); this is the part that the rest of the week builds on.")
d.stats([
    ("0.265 → 0.409", "junction score J, VAL", "fork pruning (K190) un-links forks that the junction head does not support; control 0.408; no retraining"),
    ("0.701 → 0.735", "Q at equal sparsity", "champion with pruning, S 0.746 → 0.769 (control 0.725)"),
    ("2.5 → 0.8 m", "pose error of the SD map", "rough localisation with trust gate (K188 / K189), map error 2 m, night / blur failure half"),
    ("26 → 17.5 ms", "chain per frame", "numba union + pruning, 873 of 873 graphs identical (K191); with time recursion 24 ms; control 628 ms"),
    ("0 of 3", "joint end-to-end fine-tunes that helped", "route loss through the builder adds nothing over the route head alone (E202, E214, E232)"),
    ("TEST once", "champion S 0.748 against control 0.702", "J 0.422 against 0.404; lanes found 0.653 against 0.762 remains the open gap (E230, logged)"),
])
d.fig(F + "p18_forkprune.png", "Fork pruning: the junction gap was a precision gap (loop 18, E215).")
d.fig(F + "p18_registration.png", "Rough localisation and map trust (loop 18, E213): the map benefit becomes nearly independent of the map error.")

# ------------------------------------------------------------------ 4 loop 19
d.h1("4  Loop 19: measuring the floor (25 - 26 October)", new_page=True)
d.p("Loop 19 had one question: <i>what are our numbers worth?</i> Twelve cards, refereed before the first run; the full test suite passed 14 of 14 before and after. The loop table shows 33 runs and 11.3 GPU hours (it also counts the first loop-20 runs, D17); "
    "7 are failed launchers, on a machine that other chats shared (section 6).")
d.h2("4.1  Seed noise of the recall fine-tune (E240) and the early checkpoints (E241)")
rows = []
for s in seeds:
    a, c = card(f"e24_j1rc_s{s}"), card(f"e24_j1eq_s{s}")
    rows.append([f"seed {s}", f3(a["Q"]), f3(c["Q"]), f"{a['Q'] - c['Q']:+.4f}", f3(a["lanes_found"]), f3(c["lanes_found"]), f"{a['lanes_found'] - c['lanes_found']:+.3f}",
                 f3(a["R@1_far"]), f3(c["R@1_far"]), f3(a["topo_F1"]), f3(c["topo_F1"]), f"{a['vert100']:.2f} / {c['vert100']:.2f}"])
vrc = [card(f"e24_j1rc_s{s}")["vert100"] for s in seeds]
veq = [card(f"e24_j1eq_s{s}")["vert100"] for s in seeds]
rows.append(["mean", f3(qrc["mean"]), f3(qeq["mean"]), f"{mean(dq):+.4f}", f3(lrc["mean"]), f3(leq["mean"]), f"{mean(dl):+.3f}", "", "", "", "", f"{mean(vrc):.2f} / {mean(veq):.2f}"])
rows.append(["sample SD (n - 1)", f4(qrc["sd"]), f4(qeq["sd"]), "", f4(lrc["sd"]), f4(leq["sd"]), "", "", "", "", "", ""])
d.table(rows, header=["", "Q j1rc", "Q j1eq", "diff", "lanes j1rc", "lanes j1eq", "diff", "R@1far rc", "R@1far eq", "topo rc", "topo eq", "vert100 rc / eq"], num=range(1, 12), hl=(nS,), widths=None)
d.p(f"VAL 873, fork pruning 0.4, tau 0.7, <code>evaluate.py</code> cards. Seeds 0 and 1 are the stored loop-18 fine-tunes re-evaluated through the one entry point (they reproduce the stored numbers); seed 2 and 3 are new. "
    f"Decision by the tool: {TIE('TIE')} (Q +{mean(dq):.4f} against the band 0.010), lanes found +{mean(dl):.3f} (rule: at least +0.04, met in every pair). "
    "Prediction of the card: Q +0.006, hit. Every SD in this report is a sample SD (n - 1). j1rc carries +0.16 vertices per 100 m on average (+0.04 / +0.22 / +0.34 / +0.06 per pair), worth about 0.002 of the Q gap at the E236 slope. "
    f"Q is positive in all four pairs (+0.006, standard error 0.002, paired t about 2.9 on 3 degrees of freedom): below the decision band of 0.010, so a TIE by the rule, but not shown to be zero.")
d.fig(F + "w3_seed_floor.png", "E240: per-seed values of the equal-budget fine-tune (j1eq) and the recall-weight-3 fine-tune (j1rc); right: the gap along the fine-tune for the seeds with intermediate checkpoints (E241).")
d.p("<b>E241 (early kill).</b> With the intermediate checkpoints the lanes-found effect of the recall weight has its full size already after a quarter of the fine-tune, the Q gap does not separate from noise before the end. "
    "With the four cross-pairings of two seeds each the cut test at 50 % passed (j1rc above j1eq by 0.05 - 0.08 lanes found in 4 of 4, no wrong cut), but the effect is as large at 25 % as at 50 % (+0.065 to +0.078 against a final +0.052 to +0.058: 20 to 50 % more; part of it is the +0.4 to +0.6 vertices that j1rc carries at 25 - 75 %) and Q stays between +0.001 and +0.012 without a stable sign of separation; "
    "four pairings that share runs only bound the wrong-cut rate to about 0.5, so the rule stays a candidate for a 12-pairing replay and covers loss-weight type effects only (E241).")
d.h2("4.2  Equal vertex budget (E236)")
d.p(f"The tau sweep showed that the validity threshold does not set the vertex budget; the vertex radius <code>vtx_r</code> does, and Q falls by about 0.0025 for every 0.25 vertices per 100 m removed. "
    f"At 14.3 ± 0.15 vertices and pruning 0.4 the recall variant rc1 beats the champion by only Q +0.0025 (seed twin +0.0003) instead of the registered +0.0054, while lanes found (+0.068) and R@1 far (+0.060) survive. {TIE('TIE')} "
    "For the fast pair, j1rc over j1 holds at matched vertices (Q +0.0149 on one pair). PROBLEM.md and decision D4 now quote Q with its vertex budget.")
d.h2("4.3  The damaged distance map (E251, E237) and Mirko's rule (D14)")
a0, a1, c0_, c1_ = card("e24_j1ex_s0"), card("e24_j1ex_s1"), card("e24_j1eq_s0"), card("e24_j1eq_s1")
d.p("The stored b123 distance map was made in an old environment whose matrix products run in TF32 on the RTX 3060; it breaks its own 0.5 m bound in 4,315 of 8,414 train frames and 861 of 1,706 val frames. "
    "A CPU rebuild is exact (E251). Fine-tuning the fast variant on the exact map instead gives:")
d.table([["stored map (E240 seeds 0 / 1)", f3(c0_["Q"]), f3(c1_["Q"]), f3(c0_["F1@1.0"]), f3(c1_["F1@1.0"])],
         ["exact map (E237 arm a)", f3(a0["Q"]), f3(a1["Q"]), f3(a0["F1@1.0"]), f3(a1["F1@1.0"])]],
        header=["j1eq fine-tune", "Q seed 0", "Q seed 1", "F1@1.0 seed 0", "F1@1.0 seed 1"], num=range(1, 5), hl=(1,))
d.p(f"{TIE('TIE')} (Q a - c -0.0017, inside the band). 25.7 % of the train frames and 8.7 % of the supervised pixels are off by more than 0.25 m between the two maps. "
    "Mirko's answer (D14): a defective map is not used for training; the foundation network is not retrained. The loader default was switched to the exact map; runs after the switch need a fresh equal-budget control on the same map.")
d.h2("4.4  The map benefit on new error draws (E238)")
d.fig(F + "w3_registration.png", "E238: F1@1 gain of registration + trust over the unregistered map, failure half of the failure series (154 frames, 22 scenes), three draws of the 2 m scene-constant map error.")
d.p(f"{OK('KEEP')} Draw 1 +0.019 / +0.022 / +0.025, draw 2 +0.029 / +0.018 / +0.042 (night / blur / three cameras off), stored draw +0.027 / +0.028 / +0.030; registered minus no map +0.029 … +0.043. "
    "The clean half pays nothing for it (+0.002 … +0.006). The lowest interval bound of the two new draws is +0.002; one cell of the stored draw reaches 0.0.")
d.h2("4.5  Loss and threshold ideas of the cycle (E244 - E247)")
d.table([
    ["E244 dead zone", "the 3 m clamp of the recall term leaves 6.4 - 7.6 % of far ground-truth points without gradient (30.4 - 33.5 % of the missed far points)", "diagnostic only (kill threshold 15 % not met)"],
    ["E244 arm b", "clamp 8 m: Q 0.733, lanes found 0.678, R@1 far 0.691 = the plain fine-tune", NEG("CUT as a lever")],
    ["E244 arm a (first try)", "Poisson set likelihood, cut at iteration 3200 of 4800 on a shared machine; 50 % checkpoint Q 0.673 at 15.1 vertices against 0.730", TIE("partial, negative")],
    ["E245", "36 - 38 % of rejected far hits lie in the 0.6 - 1.0 m band (gate 20 %)", INFO("alive, not trained")],
    ["E246", "per-range thresholds: in-sample ceiling R@1 far +0.006 (gate +0.015)", NEG("killed at step 0")],
    ["E247", "gated evidence update: step 0 needs a quiet GPU", INFO("not run")]],
    header=["card", "result", "state"], widths=["16%", "62%", "22%"])
d.h2("4.6  Timing (E243)")
d.p("The first timing repeat was not valid (machine 2 - 3 ms above the stored values, spread above 1.5 ms in 5 of 6 configurations, other chats' jobs running). Only the ordering survives: "
    "j1 20.26 < f1 20.76 < j1eq 20.86 < rc1 22.25 < j1rc 23.43 < f1 with recursion 27.13 ms; j1 is not a speed variant. The repeat on a quiet machine is in 5.4 (E253): same ordering, absolute values 2 - 3 ms lower.")

# ------------------------------------------------------------------ 5 loop 20
d.h1("5  Loop 20: four iterations (26 - 27 October)", new_page=True)
d.p("Mirko's request: when the other training is ready, do four iterations of automatic research here, test the methodology and the framework, improve the framework, and report. The t1 training of the BEV-system chats had finished, "
    "so the loop started at once. Each iteration is a card with a written decision rule and a summary; the cards of E252, E255 and E253 went through a referee before their runs, the E256 card (a CPU follow-up that cannot change a model) did not.")
d.table([
    ["1", "E252", "Does the map offset of the ground truth cap the metric by city?", "CPU, 10 min", "see 5.1"],
    ["2", "E240 pair 3", "closes the seed floor with a fourth pair (and E241 pair 3)", "2 trainings, ~20 min each", "see 5.2"],
    ["3", "E255", "Poisson set loss, full schedule, compared at a matched vertex budget", "1 training + grids", "see 5.3"],
    ["4", "E253 (after a wait: E256)", "timing repeat on a quiet machine with a machine guard; it had to wait five hours for another chat's 7-hour training, and E256 (do the D4 candidates keep their order in each city?) used the wait", "45 min exclusive; CPU 10 min", "see 5.4 / 5.5"]],
    header=["#", "card", "question", "cost", "result"], widths=["4%", "12%", "50%", "20%", "14%"])
d.h2("5.1  Map offsets of the ground truth by city (E252)")
d.p("The BEV-system chat found map-to-image offsets of 0.3 - 1 m on dividers, worst in Boston (41 % of Boston lane dividers within 0.25 m of the paint against 82 % in Singapore-Queenstown). "
    "If the lane centrelines inherited that, the metric with a 1 m tolerance would be capped in Boston and correct predictions would show a lateral bias there. The card predicted a Boston-minus-Singapore gap of Q -0.03. The referee asked for a null, an offset signature, a synthetic-shift scale and a per-polyline statistic; all were run.")
d.fig(F + "w3_city.png", "E252: Q by city with 95 % scene intervals (left) and the share of correct points more than 0.5 m from the ground-truth line (right), three models.")
e252 = NUM.get("e252", {})
ch = e252.get("champion", {})
if ch:
    b, s = ch["boston-seaport"], ch["singapore-onenorth"]
    d.p(f"{NEG('prediction missed in sign')} Q is higher in Boston: champion {b['Q']:.4f} against {s['Q']:.4f} (gap +{b['Q'] - s['Q']:.4f}); j1eq +0.049, j1rc +0.070. A permutation of the city labels over the 22 scenes "
        "gives a null SD of 0.027 (two-sided p 0.056); F1@0.5 and F1@1.0 gaps +0.053 / +0.040. The synthetic shift of the ground truth costs Q 0.0225 / 0.0698 / 0.2147 at 0.3 / 0.5 / 1.0 m, so an offset of 0.5 m or more would be clearly visible (the 0.3 m shift, 0.0225, is below the null SD of the city gap). "
        "Per ground-truth line, the median lateral offset of the matched predictions is 0.157 m in Boston and 0.221 m in Singapore (share above 0.3 m: 0.283 against 0.380); predicted points without ground truth within 1.5 m: 0.026 against 0.056. "
        "<b>The cap is not supported.</b> VAL has two locations only and cannot speak for Queenstown or Hollandvillage; a model trained on the ground truth can hide a systematic offset.")
d.h2("5.2  Seed pair 3 (E240) - the fourth pair")
d.p("Pair 3 had been cancelled the night before when the shared machine slowed training twenty-fold; the stalled ids were resumed through the launcher (E240.a3, E240.c3). It was pinned to the stored distance map with <code>LCR_DIST_DIR</code> so that the four pairs stay comparable "
    "(the loader default is now the exact map). Pair 3: Q 0.7391 (j1rc) against 0.7316 (j1eq), lanes found 0.7393 against 0.6815. With four pairs the tool still says "
    + TIE("TIE") + " (Q a - c +0.0062, prediction +0.006 hit); the Q seed SDs moved by at most 0.0006 against the three-pair estimate (0.0036 / 0.0022 became 0.0030 / 0.0020; the paired SD 0.0052 became 0.0043), so the Q floor is stable; the lanes-found SDs moved by up to 0.0007. The table in 4.1 and the tiles of section 1 include the pair. "
    "The intermediate checkpoints of pair 3 gave the E241 cross-pairings (section 4.1).")
d.h2("5.3  Poisson set likelihood, full schedule (E255)")
m143 = {n: card(n, "m143") for n in ("e24_j1ex_s0", "e24_j1ex_s1", "e25_j1phd2_s0")}
if m143["e25_j1phd2_s0"]:
    rows = []
    for n, lab in (("e24_j1ex_s0", "j1eq, exact map, seed 0"), ("e24_j1ex_s1", "j1eq, exact map, seed 1"), ("e25_j1phd2_s0", "Poisson loss, seed 0")):
        m = m143[n]
        rows.append([lab, f3(m["Q"]), f"{m['vert100']:.2f}", f3(m["lanes_found"]), f3(m["R@1_far"]), f3(m["topo_F1"]), f3(m["F1@1.0"])])
    d.table(rows, header=["arm (matched to 14.3 vertices / 100 m on val250, read on VAL 873)", "Q", "vert100", "lanes found", "R@1 far", "topo F1", "F1@1.0"], num=range(1, 7), hl=(2,))
    d.p(f"{NEG('CUT')} by <code>decide.py</code>: Q a - c = {m143['e25_j1phd2_s0']['Q'] - 0.5 * (m143['e24_j1ex_s0']['Q'] + m143['e24_j1ex_s1']['Q']):+.4f} (rule: CUT at -0.020 on one seed; the topo F1 guard is violated by -0.127). "
        "The prediction -0.045 ± 0.020 missed by -0.028: the loss was worse than the half-schedule reading. The procedure: sweep tau and the vertex radius on 250 tuning frames, interpolate each curve to 14.3 vertices per 100 m, take the best tau, evaluate on VAL 873 and report the 623 untouched frames (Q 0.6519 against 0.7226 / 0.7241: the two tuned scalars did not overfit the 250 frames; the held-out frames come from the same 22 scenes). "
        "The stored cards of all three models were reproduced first (layer 1 passed).")
    d.p("<b>Why:</b> at its own settings the Poisson arm sits at 18.7 vertices per 100 m with Q 0.723 - a quarter more vertices than the controls' 15.1 - because the set likelihood spreads the validity scores. Squeezed from 18.7 to 14.3 vertices it loses 0.07 Q (0.016 per vertex removed), while the controls lose 0.007 over their smaller squeeze from 15.1 to 14.3 (0.0085 per vertex): about twice the cost per vertex, and lanes found falls 0.640 to 0.551. "
        "Our reading is that the set likelihood spreads the validity scores so that thinning removes true lanes; no score-calibration analysis was made. K193 is closed as a loss swap; a calibration step is untested. One seed is enough for the CUT rule (the gap is 3.6 times the CUT threshold, about 17 paired SDs of the E240 contrast).")
else:
    d.p("E255 did not finish in this report's window; see the experiment file.")
d.fig(F + "w3_matched_vertex.png", "E255 / E244: Q against the vertex budget on the tuning subset (left; thin lines = other tau values, dashed line 14.3) and the first, cut attempt at its 50 % checkpoint (right).")
d.h2("5.4  Timing on a quiet machine (E253)")
d.p("The repeat needs the machine alone for 45 minutes. The BEV-system chat's 7-hour training ran in the planned window, so the run waited until it reported the end (07:19 on 27 October) and started at 07:20. "
    "Six configurations, six rounds in round-robin order, a fresh process per replicate, 200 timed frames each; the guards of the driver (GPU use, WSL load average, Windows CPU) voided no replicate.")
d.table([
    ["j1 192 / 96, pruning 0.3 (a)", "18.52", "0.44", "17.97 - 19.17", "7.8"],
    ["f1 288 / 144, pruning 0.3 (c)", "18.75", "0.29", "18.55 - 19.34", "7.5"],
    ["j1eq, pruning 0.4", "18.49", "0.21", "18.15 - 18.72", "7.7"],
    ["j1rc, pruning 0.4", "21.14", "0.39", "20.74 - 21.82", "8.9"],
    ["rc1 288 / 144, pruning 0.4", "20.81", "0.26", "20.46 - 21.20", "8.8"],
    ["f1 with time recursion (as trained)", "24.85", "0.76", "24.71 - 26.69", "7.8"]],
    header=["configuration (ms per frame, idle machine)", "median", "replicate SD", "range of 6", "assembly ms"], num=range(1, 5))
d.p(f"{OK('session valid')} 6 of 6 replicates per configuration, replicate SD at most 0.6 ms in 5 of 6 configurations (the recursion run has 0.76 ms), the stored loop-18 values reproduced within 1.0 ms. "
    f"{TIE('TIE')} j1 minus f1 = -0.23 ms (per-round differences -0.83 to +0.21; prediction -0.3 hit; the rule needs -1.0 ms for a speed claim): <b>j1 is not a speed variant</b>. "
    "The recall weight costs +2.65 ms (j1rc against j1eq; prediction +2.2) and the recursion +6.1 ms, as in the invalid first session, but all absolute values are 2 - 3 ms lower on the quiet machine. "
    "The chain therefore runs at 18.5 - 18.8 ms without recursion and 24.9 ms with it.")
d.fig(F + "w3_timing.png", "Replicate medians per configuration, ms per frame: the invalid session of loop 19 (left, shared machine) and the repeat with the machine guard (right).")
d.h2("5.5  The D4 candidates by city (E256, used the wait for the quiet machine)")
d.p("E252 showed that all models score higher in Boston than in Singapore. D4 chooses between candidates on pooled numbers, so we asked whether their order holds in each city (CPU, 10 minutes). Rule: a pair that is ordered one way in Boston and the other way in Singapore by more than 0.010 in both cities is a flip.")
d.table([
    ["fast_recall (j1rc 192 / 96)", "0.7757", "0.7061", "+0.070", "0.774 / 0.702", "15.0 / 15.0"],
    ["recall (rc1 + pruning 0.4)", "0.7700", "0.7137", "+0.056", "0.803 / 0.736", "15.6 / 15.5"],
    ["control (E94 trace network)", "0.7661", "0.7119", "+0.054", "0.830 / 0.773", "21.7 / 21.9"],
    ["champion (j1 + pruning 0.3)", "0.7625", "0.7101", "+0.052", "0.734 / 0.660", "14.9 / 14.8"],
    ["fast (j1, 192 / 96)", "0.7515", "0.7064", "+0.045", "0.709 / 0.636", "15.0 / 15.0"]],
    header=["candidate", "Q Boston", "Q Singapore", "gap", "lanes found B / S", "vertices B / S"], num=range(1, 6))
d.p(f"{TIE('no flip, order not established')} No pair flips, but the rule cannot fire easily: the whole Singapore spread is 0.008, below the 0.010 flip threshold. What the split adds: the pooled advantage of fast_recall over fast (+0.011) comes from Boston (+0.024 against -0.000 in Singapore; one model pair, 10 Boston scenes, scene intervals about +-0.04: no more than the seed-to-seed spread of E240), while the lanes-found gain (+0.065 in both cities) is the same everywhere. "
    "So D4 may cite 'lanes found +0.06' as evidence (Q is higher in Boston only); the order of the five Q values inside the 0.012 pooled spread is carried by 10 Boston scenes and is not established. Two cities only; the vertex budgets are not matched here (the control sits at 21.7).")

# ------------------------------------------------------------------ 6 framework
d.h1("6  The framework: what we tested and what we changed", new_page=True)
d.p("The method (frame, research, hypotheses and plan, loop, close), the launcher, the ledger and the cards were used as they are, with the derived-state tools of D16. What worked, what hurt, and what changed:")
d.table([
    ["worked", "a stalled or failed run id could be resumed by the launcher (E240 pair 3); the schedule and the loop table are derived from run records and do not go stale; the referee found real flaws in all 3 refereed loop-20 cards (null and scale for E252, an unequal matching rule for E255, an unusable load guard for E253); the numbers reviewer found 10 mismatches or overclaims in loop 19's four summaries, all corrected"],
    ["hurt: shared machine", "other chats' jobs held the load average at 30 - 64 for hours: a 4800-iteration fine-tune ran at 1 s per iteration instead of 0.3 s, stalled for 40 minutes on 100 iterations, and a timing session became void. No lease exists between chats. <b>Change:</b> a Windows-side guard (CPU %, WSL load, GPU) and repair rounds in the timing driver; the lease itself is a decision for the tools (open item N1)"],
    ["hurt: launcher", "five launchers started within seconds raced on the snapshot commit; a killed launcher's stale heartbeat blocked relaunch; both are fixed or worked around by D15 / D16"],
    ["hurt: bookkeeping", "<code>project.json</code> (protected) lists the E-id range of each loop, so loop 20 is invisible to <code>ledger.py tail --loop 20</code> and to the loop table: D17 is open; stale planned rows of the previous loop stayed in the schedule; <code>experiment.json</code> run ids were wrong once (E237)"],
    ["hurt: reproducibility", "a decided data change (D14) switched the loader default in the middle of a seed series: pair 3 had to be pinned to the old map by an environment variable; the card of every run records which map it used"],
    ["hurt: exclusive windows", "the timing window (E253) was agreed by message with the BEV-system chat and then overrun by its training start: the run waited five hours and a CPU iteration (E256) used the time. A lease (N1) is the fix, not more messages. The cut test of E241 (cross-pairings) had no tool support and was decided by hand: a subcommand of <code>decide.py</code> is open (N15)"],
    ["changed", "matched-vertex procedure as code (<code>temp/loop20/e255_matched.py</code>: stored-card reproduction first, upper-envelope interpolation instead of a maximum over grid points, held-out frames reported); timing driver with machine sidecar files; E252 follow-up controls; 12 system items (N1 - N12) written into the plan with a 'done when' each; D17 filed"],
], header=["", ""], widths=["16%", "84%"], idcol=0)

# ------------------------------------------------------------------ 7 where the chain stands
d.h1("7  Where the chain stands")
d.table([
    ["champion (D7 f1 + fork pruning 0.3)", "0.735", "0.741", "0.695", "0.709", "0.409", "24.9 (18.8 without recursion; E253)"],
    ["recall (rc1 + pruning 0.4)", "0.740", "0.742", "0.767", "0.771", "0.428", "20.8 (E253)"],
    ["fast_recall (j1rc 192 / 96)", "0.739", "0.737", "0.736", "0.727", "0.447", "21.1 (E253)"],
    ["control (E94 trace network)", "0.737", "0.763", "0.800", "0.811", "0.408", "628"]],
    header=["system (registry name)", "Q", "topo F1", "lanes found", "R@1 far", "J", "ms per frame"], num=range(1, 7))
d.p("VAL 873, <code>evaluate.py</code>, stated at each system's own operating point (14.8 - 15.5 vertices per 100 m; the control sits at 21.8, so its row is not at the same budget); at a matched 14.3 the champion has Q 0.7306 and rc1 0.7331 (E236). Nothing was promoted this week: "
    "decision D4 (which fast variant to deploy) stays with Mirko, now with the seed floor and the matched budget as evidence; D5 (purge) and D17 (the loop list in <code>project.json</code>) are open too.")

# ------------------------------------------------------------------ 8 failed, limits, next
d.h1("8  What failed, limits, next steps")
d.h2("What did not work (kept in the record)")
d.ul(["The recall clamp (E244 arm b), per-range thresholds (E246), the Poisson set loss (E244 arm a partial, E255 final: -0.073 Q at 14.3 vertices), and a quick timing session on a shared machine (E243, invalid).",
      "The map-offset cap (E252): the hypothesis came out on the wrong side of the null."])
d.h2("Limits")
d.ul(["Two cities in VAL; 22 scenes; the bands rest on four seed pairs (the standard error of an SD with n = 4 is about 40 %).",
      "Tuning of tau and vtx_r on VAL subsets (declared), one or two seeds per arm; the four seeds of E240 were trained at three code commits and under different machine loads.",
      "The ground-truth question is answered for lane centrelines on two cities only; the divider offsets of the other chat remain unexplained for the dividers themselves."])
d.h2("Next steps")
d.ul(["Switch to reporting every metric by city next to the pooled number; third city once the b456 loader exists (S19.9).",
      "A machine lease between chats before the next timing session (N1; E253 only got its quiet window because the BEV chat reported the end of its training), and metric-specific noise bands as a DECISIONS entry (D18) now that four pairs exist: seed SDs Q 0.0025, lanes found 0.004, R@1 far 0.003, topo F1 0.001; a band is about 2 paired SDs (Q 0.009, lanes found 0.012), not 1 SD.",
      "E245: the ece metric and the soft-target training; E247 step 0 on a quiet GPU."])

# ------------------------------------------------------------------ 9 corrected claims
d.h1("9  Corrected claims of the week")
d.table([
    ["E244 summary", "'the training loss is healthy'", "the Poisson term plateaus from iteration 300 and the recall term it replaced rose from 0.27 to 1.05; not evidence of health"],
    ["E237 summary", "'the exact and stored maps are interchangeable'", "an absence of evidence for an effect above about 0.01-0.016 with 2 seeds, for the fine-tune stage only"],
    ["E240 summary", "'exceeds 8 paired SDs' for lanes found and R@1 far", "8.6 and 6.9 (n = 3 pairs, a rough ratio); the vertex correction 'about +0.004 at equal vertices' is an estimate, the pairs do not show the slope"],
    ["E241 summary", "'Q overstates at 25 %'", "within about one paired SD of the E240 mean in pair 2; in pair 3 (cross-pairing rc2-eq3 +0.0109) just above it; Q is not separable from noise at a quarter of the schedule; the lanes-found prediction at 50 % missed on size"],
    ["this report, first draft", "'ranking confirmed' (E256), 'would be visible' (E252), 'controls lose 0.006 over the same squeeze' (E255), 'TEST not touched this week', 'seed SDs moved by at most 0.0006'", "found by the numbers reviewer: the E256 rule cannot fire on a 0.008 spread; only a 0.5 m shift is clearly visible; the controls lose 0.007 over a squeeze of 0.8 vertices, the Poisson arm 0.07 over 4.4; the loop-18 TEST check of 25 October lies in the week; the Q paired SD moved by 0.0009"],
    ["E252 card (first verdict)", "per-scene medians as the offset statistic", "the map error is per element; a per-polyline statistic and a null were added after the referee"],
    ["E255 card", "'one seed is enough for CUT at -0.010'", "that is about 2 paired SDs; CUT now needs -0.020 on one seed"],
    ["E240 summary (loop 19)", "seed SD from three pairs (Q 0.0036 / 0.0022; 8.6 and 6.9 paired SDs)", "four pairs: Q 0.0030 / 0.0020, paired 0.0043; lanes found 10.1 and R@1 far 7.6 paired SDs (exact; the loop-19 ratios were 8.6 and 7.0, not 6.9)"],
    ["E256 draft (author)", "'pooled advantage of fast_recall over fast +0.0197'", "+0.0111 (0.7392 against 0.7281), found before the reviewer pass"],
    ["E255 draft (author)", "'about 20 paired SDs'; native point described as the E244 reading", "17 paired SDs (0.0726 / 0.0043); the native point is the registry card (18.71 vertices, Q 0.7227)"],
], header=["where", "claim", "correction"], widths=["14%", "30%", "56%"])

# ------------------------------------------------------------------ 10 index
d.h1("10  Experiment index")
d.table([
    ["E236", "matched vertex budget re-rank of the recall variants", TIE("TIE")],
    ["E237", "fine-tune on the exact distance map", TIE("TIE")],
    ["E238", "registration + trust on new map-error draws", OK("KEEP")],
    ["E239", "CONFIRM replication", INFO("waits for the b456 loader")],
    ["E240", "seed noise of the recall fine-tune, four pairs", TIE("TIE")],
    ["E241", "early-kill replay of checkpoints", OK("KEEP: worth a 12-pairing replay")],
    ["E242", "prediction calibration", INFO("merged into the process")],
    ["E243", "timing repeat (loop 19)", NEG("invalid session")],
    ["E244", "clamp and Poisson set loss", NEG("arm b cut, arm a partial")],
    ["E245", "quality-focal node validity", INFO("alive, not trained")],
    ["E246", "range-binned thresholds", NEG("killed at step 0")],
    ["E247", "gated evidence update", INFO("not run")],
    ["E251", "CPU-exact distance map", OK("done, PASS")],
    ["E252", "map offset of the ground truth by city", NEG("cap not supported")],
    ["E253", "timing repeat with a machine guard", TIE("TIE, session valid")],
    ["E256", "D4 candidates by city", TIE("no flip, order not established")],
    ["E255", "Poisson set loss, full schedule, matched vertex budget", NEG("CUT")],
], header=["id", "question", "outcome"], widths=["8%", "62%", "30%"])
d.save(OUT)
