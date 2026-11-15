"""one-off patch of build_total_report.py, experiments_table.json and pack_total.sh (2026-11-15 08:30): the Update of 2026-11-15.
E317 TIE, E318 CUT, E319 gate not met, E325 KEEP, E341 ADOPT, E334 / E348 NOT adopted, E331 / E332 NOT adopted, E333 / E339 / E340 withdrawn (D92), E338 TIE,
E343 on hold, E345 running, E342 part A, planned cards E344 / E346 / E347 / E349 / E350, decisions D92 - D101, the Next table of HANDOVER.md.
Every number is copied from a card (experiments/E3xx_*/experiment.md), a run's result file or temp/e348/rule.log; what is not found is written '-'.
Apply ONCE:   BFLC_ROOT=$PWD python3 temp/jr/patch_total_5.py   (from bev_fusion_lane_center, in WSL)."""
import json
import os
import re

ROOT = os.environ["BFLC_ROOT"]
HERE = os.path.join(ROOT, "temp", "jr")
p = os.path.join(HERE, "build_total_report.py")
s = open(p, encoding="utf-8").read()
assert "update_section" not in s, "patch_total_5 was applied already"


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:80])
    s = s.replace(old, new, 1)


def rep_re(start, end_re, new):
    """replace from the literal `start` up to (not including) the first match of the regex `end_re` after it"""
    global s
    i = s.index(start)
    m = re.compile(end_re, re.M).search(s, i + len(start))
    assert m, end_re
    s = s[:i] + new + s[m.start():]


# ------------------------------------------------------------------------------------------------------------------ 1. the experiment table (json)
TPATH = os.path.join(HERE, "experiments_table.json")
T = json.load(open(TPATH, encoding="utf-8"))
BYID = {e["id"]: e for e in T}


def row(id, name, question, design, verdict, basis, result, source, diff, hard, lesson, **extra):
    d = {"id": id, "name": name, "question": question, "design": design, "verdict": verdict, "verdict_basis": basis, "result": result, "source": source,
         "differentiable": diff, "hard_step": hard, "in_final_system": False, "lesson": lesson}
    d.update(extra)
    return d


NEW = [
    row("E317", "Geometry loss on the delivered connector",
        "Does a loss on the delivered Bezier curve against the GT connector, plus an end-position loss, improve topo F1 and make the end geometry of the snap unnecessary?",
        "Arm g (--geom-w 0.5 on the E312 recipe, 3000 steps) vs E312 of the same seed; snapped read-out of record, unsnapped and snap-nogeom reported; seeds 0/1; VAL 873; runs E317.g_s0 (interrupted and resumed) and E317.g_s1 on 2026-11-14.",
        "TIE", "by hand",
        "Snapped topo F1 over E312 (s0 / s1): +0.0077 / +0.0098 (0.6902 / 0.6963 vs 0.6825 / 0.6864; KEEP needed more than +0.01 on both seeds). Snap-nogeom over u': +0.0199 / +0.0212; unsnapped +0.0159 / +0.0183; every interval excludes 0; guards held (dup_frac at most +0.0075).",
        "section 4", "yes",
        "snap of connector ends kept as the read-out of record; the loss never reaches the wiring half of the snap (69 - 73 % of the snap gap)",
        "The loss makes the shape a constraint (position error -0.04 m, heading error -1.2 degrees). It is already part of the E324 s recipe (--geom-w 0.5), so the TIE is for the E312 recipe and no separate adoption is needed."),
    row("E318", "Position uncertainty per keypoint",
        "Does a learned position scale per keypoint (Laplace NLL, scale fed to the pair logit) add information or topo F1 on top of the E317 geometry loss?",
        "Arm u (--geom-w 0.5 --unc) vs control E317 g of the same seed; both clauses of the informativeness requirement on both seeds; seeds 0/1; VAL 873; runs E318.u_s0 and E318.u_s1 (resumed after a crash) on 2026-11-15.",
        "CUT", "by hand",
        "Snapped topo F1 vs E317 g (s0 / s1): -0.0009 / -0.0037 (inside the band). Clause 1 fails on both seeds: Spearman(sigma, end error) +0.247 / +0.234 against the bar 0.3. Clause 2 holds: pair precision of the most confident minus the most unsure fifth +0.3995 / +0.3510 (bar +0.03). Kill not fired (0.0 % at the clamp).",
        "section 4", "yes", "none added (an uncertainty output); named steps unchanged",
        "The scale is information (more than the range or the heat probability for the end error) but under the bar and moves no graph number; the same loss and head is not proposed a third time; a scale used by the wiring or the close-node step needs its own card."),
    row("E319", "Heading-guided forward/backward exchange in maps head",
        "Does a differentiable block exchanging features ahead of and behind each cell along the lane heading (DirX) improve junction maps and the delivered graph?",
        "Arm d (maps with --dirx, then the E312 pair recipe) vs E312; the maps stage is a gate for the pair stage; seeds 0/1; VAL 873; runs E319.m_s1 and E319.m_s0 on 2026-11-15.",
        "GATE_NOT_MET", "rule",
        "Gate not met on both seeds, so no pair stage was run and there is no topo F1: maps keypoint F1 vs E307 +0.0026 / +0.0082 (gate +0.01); entry recall beyond 35 m -0.0035 / -0.0073 (gate +0.01). Kill (-0.03) not fired.",
        "section 5", "yes", "none added; named hard steps unchanged (peak choice, thresholds, snap)",
        "The block (+0.330 GMACs, 33 218 parameters) did not help the maps the pair head reads, on top of dilated blocks that already see 38 m; the head without the block stays; closed at the gate."),
    row("E325", "Learned wiring head replaces nearest-piece rule",
        "Can a small softmax head over candidate delivered pieces replace the nearest-piece rule that wires connector ends, at equal topo F1 (within 0.005 of E312 snapped)?",
        "Arm w: 545-parameter MLP head, 3 head seeds per E312 seed, trained on stored records of TRAIN frames, vs c = E312 nearest-piece rule and twin t; seeds 0/1; VAL 873; runs E325.w_s0 and E325.w_s1 on 2026-11-14.",
        "KEEP", "by hand",
        "Snapped topo F1 over E312 (s0 / s1): +0.0023 / +0.0032 (0.6848 / 0.6896 vs 0.6825 / 0.6864), paired intervals [+0.0014, +0.0034] / [+0.0016, +0.0048]; guards held (dup_frac +0.0008 / +0.0013). Agreement with the oracle choice on the real choices: 0.780 / 0.786 against the rule's 0.757 / 0.751; the distance-only head 0.753 / 0.762. Twin t (rule as softmax) reproduces the E312 cards exactly.",
        "sections 0, 6", "partly",
        "argmax of the head per end (a learned selection, named), candidate gate (4 m / 45 degrees, at most 8 places, an index choice), p >= 0.1 threshold, keypoint peak choice, close-node step, Hermite sampling and Douglas-Peucker in the scorer",
        "The gain is the learned score, not the distance. The head is trained on stored records of the frozen system, not jointly (the gradient of the topology never reaches it), and its weights are not edge probabilities. Transfer to E324 s: card E350 (planned)."),
    row("E330", "System inventory and simplification plan", "What do parameters, MACs, step times, GT layers, hard steps and dead options of the E312 system cost, and which removals could simplify it?",
        "CPU analysis of files, code and run logs, no training; blocks counted by jmodel_cost.py (batch 1); removal arms E331-E335 with a schedule; seed 0; VAL.", "DIAGNOSTIC", "by hand",
        BYID["E330"]["result"], BYID["E330"]["source"], "n/a", BYID["E330"].get("hard_step", ""),
        BYID["E330"]["lesson"] + " Overtaken on 2026-11-14: E332 arm a was NOT adopted and E333 was withdrawn (D92); the saving that was adopted is E341's 1.477 GMACs."),
    row("E331", "Pair stage without evidence along the curve",
        "Can the six evidence features sampled along each candidate pair's Hermite curve be removed from the pair head with a topo F1 loss of at most 0.01?",
        "Arm e (--no-evidence on the E312 recipe, E307 maps, 3000 steps) vs E312 of the same seed; adopt = loss at most 0.01 on both seeds and both read-outs (jrule_removal.py); seeds 0/1; VAL 873; ran 2026-11-14.",
        "NOT_ADOPTED", "by hand",
        "Unsnapped topo F1 vs E312 (s0 / s1): -0.0232 / -0.0124 (bound 0.01; F1@1.0 -0.0183 / -0.0122 also beyond); snapped -0.0029 / -0.0007 holds. Pair precision at E312's recall 0.772 / 0.745 against 0.743 / 0.724: the loss sits at the operating point p >= 0.1, not in the ranking. Maps keypoint F1 -0.031 / -0.020 against E307 (kill -0.05 not fired).",
        "section 5", "partly", "snapped p >= 0.1 read-out (E311 rule step, reported, must hold); the unsnapped read-out counts (D85)",
        "The evidence is worth 0.012 - 0.023 topo F1 on the differentiable read-out and 0.001 - 0.003 on the snapped one; the removal would save about 0.003 GMACs: it does not pay."),
    row("E332", "Maps head without auxiliary maps",
        "Can the junction maps head drop its 0.5 m paint branch (arm a) and the zone / core supervision (arm b) with a topo F1 loss of at most 0.01?",
        "Arm a (--no-paint, which needs --no-evidence, then the pair stage) vs E312; arm b only if a is adopted; seeds 0/1; VAL 873; ran 2026-11-14.",
        "NOT_ADOPTED", "by hand",
        "Arm a unsnapped topo F1 vs E312 (s0 / s1): -0.0189 / -0.0106 (bound 0.01); snapped -0.0045 / -0.0015 holds; arm a minus E331 +0.0043 / +0.0019 unsnapped (within noise). Stage-1 kill not fired: maps keypoint F1 0.7409 / 0.7355 vs E307 0.7233 / 0.7051. Arm b does not run. Maps head 3.256 -> 1.779 GMACs.",
        "section 5", "partly", "snapped p >= 0.1 read-out (E311 rule step, reported, must hold); the unsnapped read-out counts (D85)",
        "The loss is the evidence, not the paint branch: this opened E341 (the evidence with its four non-paint inputs)."),
    row("E333", "Slimmer maps head, width 40", "Can the junction maps head run at width 40 instead of 64 (1.967 instead of 3.256 GMACs) with a topo F1 loss of at most 0.01?",
        "Arm w (--d 40, maps 3000 steps, then the E312 pair stage) vs E312 of the same seed; seeds 0/1; VAL 873; never run.", "WITHDRAWN", "none",
        "Withdrawn, never run (D92, 2026-11-14: the width of the heads stays 64; no run narrows a head). The card stays as the record of the idea.",
        "status line", "partly", "none", "Compute is saved elsewhere: E341 (the 0.5 m paint branch), the host-side bookkeeping (E342) and the loader (E349)."),
    row("E334", "One joint training stage",
        "Is the separate maps stage needed, or does one joint stage of maps and pair head from an untrained head (3000 steps; 6000 conditional) match E312?",
        "Arm o3 (--e2e --scales --maps-w 1.0, untrained head, 3000 steps), o6 only if o3 loses 0.01 - 0.03; vs E312; seeds 0/1; VAL 873; runs E334.o_s0 and E334.o_s1 on 2026-11-14 / 15; o6 not run.",
        "NOT_ADOPTED", "by hand",
        "NOT adopted by ONE guard missed by 0.0004: dup_frac of the unsnapped read-out on seed 0 +0.0104 against +0.01. Topo F1 inside the band on both seeds and both read-outs: unsnapped -0.0011 / -0.0034, snapped -0.0024 / -0.0049. Pair precision at E312's recall 0.686 / 0.646 against 0.743 / 0.724; maps keypoint F1 -0.0625 / -0.0570 against E307 (kill -0.10).",
        "section 2", "partly", "snapped p >= 0.1 read-out (E311 rule step, reported, must hold); the unsnapped read-out counts (D85)",
        "The pair head is worse in a way the rule does not see (pair precision, maneuver coverage, junction F1); no re-run on the strength of a near miss; the saving is at most the 0.85 h of the first stage per seed."),
    row("E335", BYID["E335"]["name"], BYID["E335"]["question"], BYID["E335"]["design"], BYID["E335"]["verdict"], BYID["E335"]["verdict_basis"], BYID["E335"]["result"], BYID["E335"]["source"],
        BYID["E335"]["differentiable"], BYID["E335"]["hard_step"], BYID["E335"]["lesson"]),
    row("E336", "Recursive temporal state in the shared trunk",
        "Does an ego-warped feature state of the previous keyframe, carried through a zero-gated recursion in the maps-head trunk, lift topo F1 of E312 by 0.01?",
        "Arm r (state in the shared trunk, 2-keyframe sequences, state noise, depth 1) and z (state always cold) vs c = E312; seeds 0/1; VAL; never run.",
        "CLOSED_BY_DATA", "rule",
        "Closed by data, never run: the CPU ceiling of an end prior (E299 section 5, rule written before the run): a keypoint of the previous keyframe, ego-warped, matches 4.83 % of the GT ends of t (bar 0.05, a narrow miss on both seeds); a probability carry-over adds 5.5 false ends per true one (bar 1 : 1).",
        "status line", "partly", "snapped connector read-out at p >= 0.1 as in E312; state of the previous frame computed without gradient (truncated at depth 1)",
        "Not measured: a state that sums two sub-threshold looks (needs the heat maps; named in E299 section 5). The design stays as the record."),
    row("E337", "Attached ends against the snap's geometry",
        "Is the +0.0069 / +0.0062 topo F1 that E324's attached-end read-out gives up against the snap caused by the snap's geometry or by wiring?",
        "Step 0 only, from stored cards and one scan of stored dumps: CPU, nothing trained, 0 GPU h; the planned replay was dropped (the same piece is wired in 96 % of the ends); seeds 0/1; VAL.",
        "DIAGNOSTIC", "by hand",
        "Answered from stored cards: the gap k - a = +0.006932 / +0.006155 topo F1 is GEOMETRY. The snap's geometry on the same wiring is +0.009281 / +0.009378; the attachment recovers 25.3 % / 34.4 % of it. The attached end and the own keypoint wire the SAME delivered piece for 96.2 % / 96.6 % of the ends that have a piece in reach.",
        "sections 1, 2", "partly",
        "nearest-piece wiring at the read-out, candidate set of the attachment, keypoint peak choice, p >= 0.1 threshold, close-node step (E292)",
        "The attached end is a convex combination of the keypoint and raw nodes of B (att_acc 0.742 / 0.711 at step 6000), so it often does not coincide with a piece end. The sharpening lever (an entropy or margin loss on the attachment weights) stays a stub, not scheduled until E292 and the E325 head are measured."),
    row("E338", "Stack check: E324 s heads on the graph of c2",
        "Is the gain of c2 (B fine-tuned 8250 steps, pyramid frozen) additive to the gain of E324 s?",
        "CPU, no training: the pair heads of E324 s read out on c2's delivered graph instead of B's (own keypoints, snapped, p >= 0.1) vs s on B's graph; additive prediction s + c2 - c; seeds 0/1; VAL 873.",
        "TIE", "rule",
        "sc 0.7066 / 0.7016 against s 0.6981 / 0.7010: D = +0.0085 / +0.0006 (the KEEP bound is met on seed 0 only; no seed below -0.003). Additive prediction 0.7046 / 0.7030: a hit on every pre-registered number. Against E312 snapped: +0.0242 [+0.0147, +0.0333] / +0.0152 [+0.0081, +0.0218].",
        "sections 5, 6", "partly", "close-node step on c2's graph, nearest-piece wiring and snap geometry, p >= 0.1, keypoint peak choice, Douglas-Peucker",
        "c2's own gain on seed 1 is only +0.0020 over E312, so the stack rule fails there by c2's gain, not by an interaction; c2 and E324 s stay two separate candidates; adoption is Mirko's decision (D94)."),
    row("E339", "Narrow maps head taught by the wide one", "Can a narrow maps head (width 40) taught by the wide one by soft-target distillation match E312? It runs only if the plain narrow head E333 is not adopted.",
        "Fallback of E333 (arm S4b): teacher stage and student; seeds 0/1; VAL; never run.", "WITHDRAWN", "none",
        "Withdrawn, never run (D92: Mirko does not want the width narrowed). The distillation code was taken out again (commit 50460b87).", "status line", "partly", "none",
        "The literature scan of 2026-11-14 (DWPose, channel-wise distillation) stays as the record."),
    row("E340", "Still narrower maps head, width 32", "Can the junction maps head run at width 32 instead of 40? Follow-up that runs only if E333 is adopted.",
        "No new code beyond the existing --d flag; seeds 0/1; VAL; never run.", "WITHDRAWN", "none",
        "Withdrawn, never run (D92).", "status line", "partly", "none", "The 0.5 m branch alone is 1.108 of the 1.629 GMACs at width 32: a width cut cannot go below about half of the head's MACs."),
    row("E341", "Evidence without the 0.5 m paint branch",
        "How much of the evidence loss of E332 comes from the two paint inputs and how much from the four others (lane heat, zone, core)? Keep the four, drop the two, and drop the branch at the same width.",
        "Arm p: pair stage of E312 on the E332 arm-a maps (--no-paint, zone / core still supervised), four evidence inputs; seeds 0/1; VAL 873; runs E341.p_s0 and E341.p_s1 on 2026-11-14.",
        "ADOPT", "by hand",
        "ADOPT on the E312 recipe (jrule_removal.py exit 0). Unsnapped topo F1 vs E312 (s0 / s1): +0.0000 / -0.0080 (bound 0.010), snapped +0.0005 / -0.0060; every guard held (dup_frac at most +0.0050). Saves 1.477 GMACs of the system's 7.955 (-18.6 %) and 0.037 M of 2.632 M parameters.",
        "section 3", "partly", "snapped p >= 0.1 read-out (E311 rule step, reported, must hold); the unsnapped read-out counts (D85)",
        "Seed 1 unsnapped sits 0.002 inside the bound and its interval [-0.0167, -0.0001] does not exclude a loss beyond it: ADOPT says within the band, not no loss. The ms (about 1.3 of 17.09, an estimate) are not measured. Transfer to the E324 s recipe: E348."),
    row("E342", "Pair head host time",
        "Where does the junction forward spend its 17.09 ms: the pair head costs 9.72 ms with next to no multiply-adds, so is it host bookkeeping and kernel launches?",
        "Part A (CPU, bit for bit on stored dumps): jgt._groups vectorised. Part B (idle GPU, Sunday night): a profile of the pair head. No training.",
        "PART_A_INSTALLED", "by hand",
        "Part A installed (commit 27fee60b): 0.80 ms -> 0.095 ms per set (8.4 times); labels equal on 3 322 real sets; per frame about 2.2 ms -> 0.75 ms = -1.4 ms of the 17.09 ms (8 %). Part B not run. Finding (section 4): the training step is bound by the loader's CPU work, not by the GPU.",
        "sections 1, 4", "n/a", "rank_matrix bookkeeping (single linkage of the ends on the host, no gradient: named hard bookkeeping, D82)",
        "The first idle-GPU timing after the install is the new baseline; the 17.09 ms of 2026-11-13 predates the patch."),
    row("E343", "The one TEST use of E324 s",
        "Does E324 s (with the stack E338 as a second arm) keep its VAL gain over E312 on unseen TEST scenes? Report only: nothing selected or tuned on TEST.",
        "Arms w (snapped own keypoints), p (PRIMARY, attached ends), k (the stack E338 on c2's graph); control = E321's stored TEST cards of E312; TEST 21 scenes; card prediction for w +0.008 / +0.007 over E312; one logged use (D95); the VAL check is the gate.",
        "ON_HOLD", "none",
        "NO TEST number exists; no TEST run, no log row, no grant. Criterion 1 passed on both seeds (max difference 0.00e+00). Dress rehearsal, CPU branch: failed the pre-registered tolerance. GPU branch: seed 0 PASS, seed 1 FAIL by ONE number: vertices per 100 m differs from the stored VAL cards by 4.87e-3 (snapped) against the tolerance 3e-3 (the residual is the CPU dump of B). Arm k reproduces the stored card exactly.",
        "section 2", "partly", "snap wiring and keypoint geometry of arm w (a rule step); arm p keeps the nearest-piece wiring; named in every report",
        "By the card's letter the check failed. HOLD: Mirko decides between (a) kill and (b) a recorded tolerance of 1e-2 on vertices per 100 m; until his word no log row and no grant. Code of 71 files is frozen until the TEST run ends."),
    row("E344", "Plan: from five rule steps to none (Q-Lane)",
        "How to get from the named rule steps of the delivered graph to none: a query head on the 1 m grid that allows position uncertainty and optimises smoothness and connections together?",
        "Plan only: track A closes gaps in the system we have, track B is Q-Lane (variant B0: points on the 2 m grid, attention pooling, soft pruning), track C the sensors (D50); gates, costs and questions for Mirko with defaults.",
        "PLAN", "none", "Nothing built, nothing run. Q0 (CPU, ground truth only) is the card E346 and may start under D86; Q1 and later wait for a recorded decision of Mirko on D84 point 6 and on the replacement of B's heads.",
        "status line", "n/a", "the plan names the rule steps counted from the code (2026-11-14)", "The prior art E271 / E273 / E292 is cited in the card."),
    row("E345", "Pruning sensitivity, step 0 of D99",
        "How small can the heads be when they are pruned by importance? A sensitivity table without retraining on the stored E312 and E324 s models, one unit removed at a time (junction side).",
        "Probe, units, ablation and controls fixed in the card; the sweep and full-VAL points with paired scene bootstraps; four model seeds; GPU about 8 h in four jobs; no training, TEST untouched.",
        "RUNNING", "none", "No table yet: the card folder holds no results.md. The probe jobs of the E312 side are done in the ledger (E345.a / d / p / s / v / sv / b_e312, status ok); the E324 s side and the table are still to come.",
        "sections 0, 4", "n/a", "none (a measurement)", "The B side on c2 is a later card; menc and asg are read by no named read-out, so they cannot be classed yet."),
    row("E346", "Ceiling of the one-graph code",
        "How much does B's node + link code lose on all 873 VAL frames against a clean control, where does the loss sit, and do a 1 m grid or node chains through the junction work?",
        "CPU, no training, built in scratch (temp/e346): arms g0, g1, l2, l1 through run.py; decided by hand (GT replay); cell size set at runtime, no frozen file touched.",
        "BUILT_NOT_RUN", "none",
        "No 873-frame result. Smoke on 31 frames (every 29th): topo F1 g0 1.0000, g1 0.9854, l2 0.9684, l1 0.9687; l2 loses 0.0170 against g1 (bar 0.03); connectors with a collision cell 57.5 % at 2 m and 53.1 % at 1 m (bar 5 %). Open kill item: test_qtargets is red at HEAD (dropped candidates 1.042 % against 1 %); the arms run anyway, labelled.",
        "section 4", "n/a", "none (a measurement of the GT code)", "Stage Q0 of E344; the lane lead's card."),
    row("E347", "Junction ends on the 2 m node grid",
        "Can one grid carry the junction ends (no 1 m branch), and does a one-to-one target make the peak rule unnecessary?",
        "Stage 1: the maps head with its output at 2 m (arm g Gaussian heat, arm o one-to-one target) vs E332 arm a; stage 2: the pair stage on the 2 m maps vs E341; seeds 0/1; VAL; plan card.",
        "PLANNED", "none", "No number (plan card, nothing started; the card's predictions are guesses). Staged copies in temp/e347, installed after the TEST end of E343 (code freeze).",
        "sections 0, 1", "partly", "3 x 3 peak rule and threshold (arm g); arm o drops the peak rule", "Plan of the lane lead (2026-11-14)."),
    row("E348", "E341 removal on the system of record",
        "Does the removal of the 0.5 m paint branch (E341, adopted on the E312 recipe) also hold on the E324 s recipe (6000 steps, geometry loss, attachment)? D94: the transfer needs its own run.",
        "Arm w: E324 s recipe on the E332 arm-a maps, vs E324 s of the same seed; read-outs snapped, PRIMARY and attached; jrule_removal.py; seeds 0/1; VAL 873; runs E348.w_s0 and E348.w_s1 on 2026-11-15 (s1 crashed at step 5038 and was resumed).",
        "NOT_ADOPTED", "rule",
        "NOT adopted by the written rule (jrule_removal.py exit 1, temp/e348/rule.log; the card has no result section yet): the one failing guard is PRIMARY seed 1 dup_frac +0.0109 against the bound +0.01. Topo F1 vs E324 s (s0 / s1), all inside the band: snapped -0.0028 / -0.0006, PRIMARY -0.0014 / +0.0022, attached -0.0013 / +0.0014.",
        "temp/e348/rule.log, section 3", "partly", "nearest-piece wiring in the PRIMARY read-out, candidate set of the attachment, keypoint peak choice, p >= 0.1, close-node step",
        "Named deviation: E348.w_s1 crashed at step 5038 of 6000 (ENOMEM in a data worker) and was resumed from its checkpoint; bit-identity with an uninterrupted run is not claimed."),
    row("E349", "Pair-stage GT cache",
        "Can the per-frame ground truth of the pair stage (junction GT, ends, pairs, pair scales, connector polylines) be computed once so that the loader stops limiting the step?",
        "A cache of the environment-independent geometry and a pair-only item path vs the uncached loader with 4 workers; every field identical under both training environments; rule: at least 0.9 s less per batch of 8; 0 GPU h; plan card.",
        "PLANNED", "none", "No number (plan card, nothing built). Control: 1.83 s per batch of 8 with 4 workers on the loaded box, a training step of about 2.0 s; the card predicts 1.5 s less per batch.",
        "section 0", "n/a", "none (inputs identical)", "Makes every later arm cheaper (E348 follow-ups, pruning step 1 of D99); waits for the E343 freeze."),
    row("E350", "Wiring head on the system of record",
        "Does the E325 learned wiring head, trained on E324 s and read out at its snapped graph, keep equal accuracy on the system of record? D94: the transfer needs its own run.",
        "Head trained on frozen records of E324 s, snapped read-out only (the PRIMARY read-out keeps the nearest-piece rule); a VAL cross-validation gate before the GPU pass; c2 and the stack only if KEEP; plan card.",
        "PLANNED", "none", "No number (plan card). Expected gain at most about +0.003 topo F1 on a read-out that keeps its other rule steps: an end-to-end card, not an accuracy card. Phase B waits for the TEST end of E343 (code freeze).",
        "sections 0, 1", "partly", "argmax of the head per end; the candidate lists; the PRIMARY read-out keeps the nearest-piece rule", "KEEP would make the head the wiring of the snapped read-out of the system of record."),
]
for r_ in NEW:
    BYID[r_["id"]] = r_
KEY = lambda e: (int(re.match(r"E(\d+)", e["id"]).group(1)), e["id"])  # noqa: E731
T = sorted(BYID.values(), key=KEY)
json.dump(T, open(TPATH, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

# ------------------------------------------------------------------------------------------------------------------ 2. the builder
# phases, verdict classes and names, filter, css
rep('("Simplification of the system, D84 point 4 (E330 - E340)", 330, 340)]', '("System inventory, simplification arms, stack check (E330 - E340)", 330, 340), ("Update 2026-11-15: removals on the system of record, the TEST of E324 s, plans (E341 - E350)", 341, 350)]')
rep('VCLS = {"KEEP": "ok", "TIE": "info", "CUT": "no"}', 'VCLS = {"KEEP": "ok", "TIE": "info", "CUT": "no", "ADOPT": "ok", "NOT_ADOPTED": "no", "ON_HOLD": "info"}')
rep('"SUPERSEDED": "superseded"}', '"SUPERSEDED": "superseded", "ADOPT": "ADOPT", "NOT_ADOPTED": "NOT adopted", "ON_HOLD": "on hold", "WITHDRAWN": "withdrawn", "GATE_NOT_MET": "gate not met, closed", "CLOSED_BY_DATA": "closed by data", "PART_A_INSTALLED": "part A installed, part B open", "PLAN": "plan", "PLANNED": "planned", "RUNNING": "running", "BUILT_NOT_RUN": "built, not run"}')
rep('table.exp tr[data-v="CUT"] td:first-child { border-left: 3px solid var(--bad); }', 'table.exp tr[data-v="CUT"] td:first-child { border-left: 3px solid var(--bad); } table.exp tr[data-v="ADOPT"] td:first-child { border-left: 3px solid var(--good); } table.exp tr[data-v="NOT_ADOPTED"] td:first-child { border-left: 3px solid var(--bad); } table.exp tr[data-v="ON_HOLD"] td:first-child { border-left: 3px solid var(--warn); }')
rep('<button data-v="CUT">CUT</button>', '<button data-v="CUT">CUT</button><button data-v="ADOPT">ADOPT</button><button data-v="NOT_ADOPTED">NOT adopted</button>')
rep("['KEEP', 'TIE', 'CUT'].indexOf(r.dataset.v) < 0", "['KEEP', 'TIE', 'CUT', 'ADOPT', 'NOT_ADOPTED'].indexOf(r.dataset.v) < 0")
rep("Colour at the left edge: green KEEP, amber TIE (kept and refined), red CUT.", "Colour at the left edge: green KEEP or ADOPT, amber TIE (kept and refined) or on hold, red CUT or NOT adopted (the removal arms are ADOPT / NOT adopted by their written rule).")

# the head of the page
rep("state of 2026-11-12, night - lane-centre graph from camera + radar on frozen GaussianCaR features - VAL, plus the one TEST use of E312 (E321)", "state of 2026-11-15 (first built 2026-11-12) - lane-centre graph from camera + radar on frozen GaussianCaR features - VAL, plus the one TEST use of E312 (E321); the TEST of E324 s (E343) is on hold")
rep("The best working system is E312 (junction maps", 'The system of record since D94 is E324 s ({pair(head(C_E324S, "topo_F1"))} snapped topo F1 on VAL against {pair(head(C_E312S, "topo_F1"))} for E312), a refinement of E312; the base system E312 (junction maps')
rep('<div class="tile good"><div class="k">the same without the snap rule (the differentiable read-out)</div>', '<div class="tile good"><div class="k">system of record E324 s, snapped: topo F1 (s0 / s1)</div><div class="v">{pair(head(C_E324S, "topo_F1"))}</div><div class="n">E312 recipe + geometry loss + learned attachment + 6000 pair steps (D94); VAL, two seeds; no TEST yet (E343 on hold)</div></div>\n<div class="tile good"><div class="k">the same without the snap rule (the differentiable read-out)</div>')
rep('<nav class="idx"><a href="#sum">Summary</a><a href="#best">Best approaches</a>', '<nav class="idx"><a href="#sum">Summary</a><a href="#update">Update 2026-11-15</a><a href="#best">Best approaches</a>')

# summary
rep("E320 (learned attachment) is measured (written token CUT by a guard, a tie with the snap; D87: kept and refined), E322 (soft wiring): no accuracy headroom, by D87 built at the equal-accuracy bar, E317 / E318 / E319 are held (D85).</li>",
    "E320 (learned attachment) is measured (written token CUT by a guard, a tie with the snap; D87: kept and refined); the learned wiring head E325 is KEEP on E312 (its final choice is still an argmax, a named output selection); E317 is a TIE, E318 CUT, E319 closed at its gate. Section Update 2026-11-15 has the numbers.</li>")
rep("<b>Not done:</b> no later design (E320 and after) has been run on TEST; E315 - E323 are VAL results (E315, E316 TIE, E320 CUT by a guard, E322 / E323 read-outs); the simplification arms E331 - E334 are running.</li></ul></section>",
    "<b>Not done:</b> no later design has been run on TEST: E343 (the one TEST use of E324 s, D95) is on hold after its dress rehearsal failed by one guard number, so there is no TEST number for E324 s; everything after E321 is VAL.</li>\n"
    "<li><b>System of record since D94.</b> E324 s (the E320 recipe at 6000 pair steps), VAL snapped topo F1 {pair(head(C_E324S, \"topo_F1\"))}; c2 (B fine-tuned) and the stack E338 are candidates next to it; every new arm is compared with E324 s of the same seed. The protected files (PROBLEM.md, registry) change only after the TEST check.</li></ul></section>\n\n{update_section()}")

# best approaches: recommendation + the table
rep("<b>Recommendation.</b> E312 is the system of record.", "<b>Recommendation.</b> E324 s (own keypoints, snapped) is the system of record since D94 (VAL); E312 is the control of the one TEST use so far and the older reference.")
rep("the one-piece wiring stays a named selection on the output (E322 gate G0: a learned choice has no headroom). The next gains come from simplification (E331 - E334), the free 6000-step pair training (E316, +0.007) and the geometry loss (the E320 recipe's own snapped read-out is +0.005 / +0.006 over E312).",
    "the one-piece wiring can be replaced by the learned head of E325 (KEEP on E312: +0.0023 / +0.0032 snapped topo F1; its final choice is still an argmax, a named selection on the output). The simplification arms are measured (E331, E332, E334 not adopted, E333 withdrawn, E341 adopted); the further gains come from the stack with c2 (E338, a candidate), the learned wiring head on the system of record (E350, planned) and the plans listed in the Update.")
rep_re("def best_table():", r"^# -+ the experiment table", r'''def dif(a, b, k="topo_F1"):
    return [None if None in (x, y) else x - y for x, y in zip(head(a, k), head(b, k))]


def best_table():
    rows = []

    def row(rank, name, what, cs, man, diffy, role):
        d = diff(cs, "topo_F1")
        rows.append([rank, f"<b>{name}</b><br><span class='cap'>{what}</span>", pair(head(cs, "topo_F1")), pair(d, sg) if all(v is not None for v in d) else "-",
                     pair(diff(cs, "lanes_found"), sg) if cs[0] else "-", " / ".join(f"{v:.1f} %" for v in man) if all(v is not None for v in man) else "-", diffy, role])
    C_E325W = cards("runs/e312_p_s{s}/ends_card_e325w_snap_thr0.1.json")
    C_E341P = cards("runs/e341_p_s{s}/ends_card_snap_thr0.1.json")
    row("1", "E324 s, snapped own keypoints (system of record, D94)", "E312 recipe + geometry loss (<code>--geom-w 0.5</code>) + learned attachment + 6000 pair steps; read out like E312 (snapped, p &gt;= 0.1)", C_E324S, [None, None], pill("info", "as E312 + a learned attachment; read-out steps named"),
        f"use this: the VAL reference of every new arm; {pair(dif(C_E324S, C_E312S), sg)} topo F1 over E312, guards held; c2 (B fine-tuned) and the stack E338 are candidates next to it; adoption in the protected files and any TEST need Mirko's recorded OK (E343 on hold)")
    row("2", "E312, snapped (the base system; control of the one TEST use)", "E307 maps (heat 1.0 m) + pair head <code>--e2e --scales</code> + connectors at p &gt;= 0.1 + snap", C_E312S, [cov(MAN[0]), cov(MAN[1])], pill("info", "all but the named read-out steps"), "the older system of record and the control of E321 (section TEST); the recipe E324 s refines")
    row("3", "E325, E312 + the learned wiring head", "the nearest-piece rule of the snap replaced by a 545-parameter softmax head over the candidate places; trained on stored records, not jointly; the final choice is an argmax", C_E325W, [None, None], pill("info", "as E312; the wiring is a learned argmax"),
        f"KEEP: {pair(dif(C_E325W, C_E312S), sg)} snapped topo F1 over E312 at equal guards; the end-to-end replacement of a rule step; the transfer to E324 s is E350 (planned)")
    row("4", "E341, E312 recipe without the 0.5 m paint branch", "evidence along the curve with its four non-paint inputs; maps head -1.477 GMACs (-18.6 % of the system), -0.037 M parameters", C_E341P, [None, None], pill("info", "as E312"),
        f"ADOPT on the E312 recipe: {pair(dif(C_E341P, C_E312S), sg)} snapped topo F1 against E312, inside the band; the transfer to E324 s (E348) was NOT adopted by its written rule (one dup_frac guard)")
    row("5", "E298, driven pairs weighted 2", "E312 with the positive pair terms of driven maneuvers weighted 1 + 2", C_E298S, [None, None], pill("info", "same as E312"), "use when the ego's own maneuvers matter more than all pairs (card: driven pair recall up; topo F1 within the 0.01 band of E312)")
    row("6", "E314, heat 0.7 m", "the E312 recipe on the E313 maps", C_E314S, [cov(MAN14[0]), cov(MAN14[1])], pill("info", "same as E312"), "TIE with E312: more lanes found, fewer isolated ends; the option when lanes found matters more than junction precision")
    row("7", "E312, not snapped", "the same checkpoints, connectors drawn from the keypoints", C_E312U, [cov(MAN[0], "p0.1"), cov(MAN[1], "p0.1")], pill("ok", "differentiable up to the threshold and the close step"), "the differentiable reference: geometry as good, topology precision lower")
    row("8", "E303, handle scales + snap", "E287 pair head with two learned handle scales per pair", C_E303S, [None, None], pill("info", "snap + threshold"), "kept inside E312")
    row("9", "E305, maps and pairs trained together + snap", "the pair loss reaches the maps head through the soft-argmax", C_E305S, [None, None], pill("info", "snap + threshold"), "kept inside E312")
    row("10", "E287 p2, pair head (no snap, p &gt;= 0.5)", "junction maps (E286) + pair head v2", C_E287, [None, None], pill("ok", "up to threshold"), "first working junction layer; superseded")
    row("11", "E284, end head on B's node tokens", "the first junction end head", C_E284, [None, None], pill("ok", "up to threshold"), "baseline of the junction thread; superseded")
    return tbl(["rank", "approach", "topo F1 (s0 / s1)", "vs plain graph", "lanes found vs plain", "maneuvers realised, %", "end-to-end differentiable?", "use"], rows, num=(2, 3, 4, 5))


''')

# the hard steps: state cells
rep("has no headroom (the one-piece rule is a count effect and a named selection on the output, allowed by D55)\",", "has no accuracy headroom (the one-piece rule is a count effect and a named selection on the output, allowed by D55). <b>E325 (KEEP, 2026-11-14): a 545-parameter learned head replaces the nearest-piece rule at equal-or-better accuracy on E312 (+0.0023 / +0.0032 snapped topo F1); its final choice is still an argmax and it is trained on stored records, not jointly; E350 plans the transfer to E324 s</b>\",")
rep('pill("info", "highest priority"), "designed; code after E317 / E318"],', 'pill("info", "highest priority"), "geometry: E320 / E324 measured; wiring: E325 KEEP on E312, E350 planned"],')
rep('"E318 for sigma; delivery of probabilities to be planned"', '"E318 (position scale) was CUT: the scale is informative but moves no graph number; delivery of probabilities to the planner still to be planned"')
rep('(945, 374, "to change: E320 + probabilities")', '(945, 374, "to change: wiring head (E325 / E350) + probabilities")')
rep("So the first step is E320, and E317 (a geometry loss on the delivered curve) is its training signal. The order of work: E317, E318, E320, then delivery of probabilities instead of a threshold, then the Matrix-NMS layer in the forward path; the unstructured set head stays deferred (D84).",
    "The first step was E320 with E317's geometry loss as its training signal (both measured, see above and the Update of 2026-11-15); what is left is the wiring (the learned head E325 / E350, whose output is still an argmax), delivery of probabilities instead of a threshold, and the close-node step; the unstructured set head stays deferred (D84).")
rep("each has a differentiable test in the queue.</div>", "each had a differentiable test (measured: E317 TIE, E318 CUT, E319 closed at its gate; see the Update of 2026-11-15).</div>")
rep("<td><b>E317</b>: Chamfer loss between the delivered Hermite curve and the true connector; training only</td></tr>", "<td><b>E317</b>: Chamfer loss between the delivered Hermite curve and the true connector; training only. Measured: TIE, kept; it lifts the end geometry by +0.0199 / +0.0212 topo F1 and is part of the E324 s recipe</td></tr>")
rep("<td><b>E318</b>: a log sigma per end with a Laplace NLL; sigma enters the pair logit and goes to the planner</td></tr>", "<td><b>E318</b>: a log sigma per end with a Laplace NLL; sigma enters the pair logit and goes to the planner. Measured: CUT (Spearman of sigma with the end error +0.247 / +0.234 against the bar 0.3; no topo F1 gain)</td></tr>")
rep("<b>E299</b> in time (pair flips 19 %)</td></tr></tbody></table></div>", "<b>E299</b> in time (pair flips 19 %). Measured: E319 closed at its gate (maps keypoint F1 +0.0026 / +0.0082, bar +0.01); E336 (recursive state) closed by data</td></tr></tbody></table></div>")

# TEST section, compute, next, weaknesses, footer
rep("no promotion (the champion decision stays with Mirko), E312 stays the system of record.</div></section>",
    "no promotion (the champion decision stays with Mirko); since D94 the VAL reference is E324 s, while the protected champion entries still name E312 until the TEST check of E324 s (D95).</div>\n"
    "<div class=\"note warn\"><b>The second TEST use is on hold (E343).</b> Mirko's OK of D95 allows ONE use for E324 s (with the stack E338 as a second arm), control E312's stored TEST cards. The VAL gate of the card (the code must reproduce the stored VAL cards, then a dress rehearsal with fresh dumps on the TEST device) failed by one guard number: vertices per 100 m differs by 4.87e-3 on seed 1 against the pre-registered 3e-3 (the residual is the CPU dump of B); by the card's letter the check failed. <b>No TEST run was started and no TEST number of E324 s exists</b>; Mirko decides between a kill and a recorded tolerance of 1e-2 on that guard metric.</div></section>")
rep("the read-out runs on the CPU.</p></section>", "the read-out runs on the CPU. Since the last build: E341 removes 1.477 GMACs and 0.037 M parameters (the ms are an estimate until E342 part B), E342 part A took about 1.4 ms of the 17.09 ms per frame (the pair head's host bookkeeping), and the efficiency facts of the trainings (GPU about 15 % busy, loader-bound step, E349) are in the Update of 2026-11-15.</p></section>")
rep("<section id=\"next\"><h2>Next: queued, running, decided</h2>{q_table_total()}", "<section id=\"next\"><h2>Next: queued, running, decided</h2>{next_table()}<h3>Queued, running and decided cards</h3>{q_table_total()}")
rep("KEEP when the snapped topo F1 over E312 rises by more than 0.01 on both seeds with the guards held; E317 / E320 also report the unsnapped number.", "KEEP when the snapped topo F1 over its control rises by more than 0.01 on both seeds with the guards held; since D94 the control is E324 s of the same seed (E312 for the cards written before); the removal arms are ADOPT / NOT adopted by the removal rule of D84 point 4 (loss at most 0.01 on both seeds and both read-outs, the differentiable one first, guards held).")
rep("TEST was used once, for E312 only (E321); E320 and every later design have not been run on TEST and would need their own recorded OK.</li></ul>", "TEST was used once, for E312 only (E321); no later design has been run on TEST: the second use (E343, E324 s, D95) is on hold after the failed dress rehearsal and would need its own log row; no TEST number of E324 s exists.</li></ul>")
rep("<h3>Open decisions</h3><ul class=\"tight\"><li>TEST for any later design", "<h3>Open decisions</h3><ul class=\"tight\"><li>E343 hold: kill, or a recorded tolerance of 1e-2 on vertices per 100 m (Mirko); TEST for any later design")
rep("Built by temp/jr/build_total_report.py (prefix: build_e2e_report.py)", "Built by temp/jr/build_total_report.py (prefix: build_e2e_report.py; patched by patch_total_2 - 5)")

# the next-section tables (q_table_total replaced: the states of 2026-11-15)
rep_re("def q_table_total():", r"^# -+ media \(the E2E set", r'''def q_table_total():
    rows = [["E315, E316", "maps trained 6000 steps (E315); the E312 pair head trained 6000 steps (E316)", "yes, same model", "0 ms (training only)", "measured: TIE, kept (E316 +0.0073 / +0.0074 snapped over E312); the 6000 pair steps are part of E324 s"],
            ["E317", "geometry loss on the delivered connector: Chamfer between the Hermite curve and the true connector", "yes: the loss on the output reaches positions, headings, scales and, through the soft-argmax, the maps", "0 ms", "measured: TIE, kept (snapped +0.0077 / +0.0098 over E312); part of the E324 s recipe"],
            ["E318", "E317 + a position scale per keypoint (Laplace NLL)", "yes", "about 0 ms", "measured: CUT by the informativeness clause (Spearman +0.247 / +0.234 against 0.3); no topo F1 gain"],
            ["E319", "heading-guided forward / backward exchange in the junction maps head", "yes", "+0.33 GMACs (+10 % of the maps head), +33 k parameters", "closed at its gate (maps keypoint F1 +0.0026 / +0.0082, bar +0.01); no pair stage run"],
            ["E320", "learned attachment of the connector ends to B's nodes + the E317 geometry loss: replaces the geometry of the snap", "yes (the one-piece wiring on the output stays)", "+9.6 k parameters, +3.9 MMACs", "MEASURED: written token CUT by the duplicates guard (dup_frac +0.0184 on seed 0, +0.0093 on seed 1; topo F1 ties the snap); E323: not the curve count; D87: kept and refined, follow-up E324 (the duplicates)"],
            ["E322, E325", "a learned choice of the delivered piece a connector end is wired to (E322 gate, E325 head)", "yes (the final choice is an argmax: a named selection)", "545 parameters, at most 0.12 MMACs per frame", "E322 G0: no accuracy headroom; E325 head: KEEP on E312 (+0.0023 / +0.0032 snapped); E350 transfers it to E324 s (planned)"],
            ["E331 - E334", "simplification arms (D84 point 4): evidence off, no paint branch, slim maps head, one joint stage", "removals", "lower cost", "E331 NOT adopted (unsnapped -0.0232 / -0.0124), E332 arm a NOT adopted (-0.0189 / -0.0106), E333 withdrawn (D92), E334 NOT adopted (one guard, dup_frac +0.0104 against 0.01)"],
            ["E341, E348", "the evidence without the 0.5 m paint branch (-1.477 GMACs): on the E312 recipe (E341) and on the E324 s recipe (E348)", "removal", "-1.477 GMACs, -0.037 M parameters", "E341 ADOPT on the E312 recipe; E348 NOT adopted by its written rule (PRIMARY seed 1 dup_frac +0.0109 against 0.01)"],
            ["E338", "stack check: E324 s heads on the graph of c2", "n/a (read-out, CPU)", "0", "TIE, kept: sc 0.7066 / 0.7016, additive prediction hit; c2 and E324 s stay separate candidates"],
            ["E343", "the one TEST of E324 s (D95)", "n/a", "CPU scoring, GPU dumps", "ON HOLD after the failed dress rehearsal (vertices per 100 m, one number); no TEST number"],
            ["E345", "pruning by importance, step 0 (D99): a sensitivity table without retraining", "n/a (measurement)", "GPU about 8 h in four jobs", "running; no table yet"],
            ["E342, E349", "host time of the pair head (E342 part A installed, part B open); a GT cache for the pair stage (E349, plan)", "n/a", "E342 A: -1.4 ms per frame", "E342 part A installed; part B and E349 not run"],
            ["E344, E346, E347, E350", "Q-Lane plan; ceiling of the one-graph code; ends on the 2 m node grid; the wiring head on E324 s", "plans", "-", "E344, E347, E350 plan cards; E346 built in scratch, smoke on 31 frames"],
            ["E299, E336", "time: the previous keyframe, ego-warped", "yes", "small", "E336 closed by data: the CPU ceiling of an end prior misses its bar (4.83 % of the GT ends matched, bar 0.05); E299 not built"],
            ["D84 items", "TEST pipeline for E312, simplification by one-at-a-time removals (E330 - E335)", "n/a", "removals lower the cost", "TEST of E312 done once (E321); removals: E341 adopted, E331 / E332 / E334 not adopted, E333 withdrawn, E335 CUT"]]
    return tbl(["id", "question", "differentiable", "run-time cost", "state (2026-11-15)"], rows)


def next_table():
    rows = [["1", "E343 HOLD (Mirko)", "the dress-rehearsal check failed by one guard number (vertices per 100 m 4.9e-3 against 3e-3, a device effect of B's dump); Mirko decides between (a) a kill and (b) a recorded tolerance of 1e-2; until his word no log row, no grant, no TEST"],
            ["2", "E346 (CPU, scratch build running, no frozen file)", "the 873-frame arms of the one-graph ceiling; only the 31-frame smoke exists so far"],
            ["2", "E318 / E348 results", "in the Update of 2026-11-15: E318 CUT by the written rule; E348 NOT adopted (rule log; the card's result section is still to be written)"],
            ["3", "WAIT for the TEST end of E343 (freeze of its 71 code files)", "E349 (GT cache), E350 phase B (wiring head on E324 s), E347 (staged copies in temp/e347, install after), D97 (removal of the old reference chain), dead code (D93)"]]
    return ("<p>From the line <b>Next</b> of HANDOVER.md (2026-11-15): <code>1. E343 HOLD (Mirko); 2. E346 (CPU, scratch build running; no frozen file); E318 / E348 results; 3. WAIT for E343's TEST end (freeze): E349, E350 phase B, E347 (staged copies in temp/e347, install after), D97, dead code.</code></p>"
            + tbl(["#", "item", "state"], rows))


''')

# the update section, before the page
UPDATE = r'''# ------------------------------------------------------------------------------------------------------------------ update 2026-11-15 (patch_total_5.py)
C_E324S = cards("runs/e324_s_s{s}/ends_card_snap_thr0.1.json")


def pf(vals, fmt=f3):
    return "-" if any(v is None for v in vals) else " / ".join(fmt(v) for v in vals)


def tv(path, key="topo_F1"):
    c = J(path)
    return None if not c else c["numbers"]["head"][key]


def readout_table():
    """topo F1 and dup_frac of the arms of the last days against their controls, read from the result files (VAL, 873 frames, p >= 0.1)"""
    r12s, r12u = "runs/e312_p_s{s}/ends_card_snap_thr0.1.json", "runs/e312_p_s{s}/ends_card_thr0.1.json"
    r324 = "runs/e324_s_s{s}/ends_card_snap_thr0.1.json"
    spec = [("E324 s (system of record)", "snapped, own keypoints", "E312", r324, r12s),
            ("E317 g (geometry loss)", "snapped", "E312", "runs/e317_g_s{s}/ends_card_snap_thr0.1.json", r12s),
            ("E317 g", "unsnapped", "E312", "runs/e317_g_s{s}/ends_card_thr0.1.json", r12u),
            ("E318 u (position scale)", "snapped", "E317 g", "runs/e318_u_s{s}/ends_card_snap_thr0.1.json", "runs/e317_g_s{s}/ends_card_snap_thr0.1.json"),
            ("E325 w (learned wiring head)", "snapped", "E312", "runs/e312_p_s{s}/ends_card_e325w_snap_thr0.1.json", r12s),
            ("E341 p (no 0.5 m paint branch)", "snapped", "E312", "runs/e341_p_s{s}/ends_card_snap_thr0.1.json", r12s),
            ("E341 p", "unsnapped", "E312", "runs/e341_p_s{s}/ends_card_thr0.1.json", r12u),
            ("E334 o (one joint stage)", "snapped", "E312", "runs/e334_o_s{s}/ends_card_snap_thr0.1.json", r12s),
            ("E334 o", "unsnapped", "E312", "runs/e334_o_s{s}/ends_card_thr0.1.json", r12u),
            ("E331 e (no evidence)", "unsnapped", "E312", "runs/e331_e_s{s}/ends_card_thr0.1.json", r12u),
            ("E332 a (no paint branch, no evidence)", "unsnapped", "E312", "runs/e332_pa_s{s}/ends_card_thr0.1.json", r12u),
            ("E338 sc (E324 s heads on c2's graph)", "snapped, own keypoints", "E324 s", "runs/e324_s_s{s}/ends_card_e338_snap_thr0.1.json", r324),
            ("E348 w (E324 s recipe, no paint branch)", "snapped, own keypoints", "E324 s", "runs/e348_w_s{s}/ends_card_snap_thr0.1.json", r324),
            ("E348 w", "PRIMARY (attached ends, one-piece wiring)", "E324 s", "runs/e348_w_s{s}/ends_card_snap_nogeom_attach_thr0.1.json", "runs/e324_s_s{s}/ends_card_snap_nogeom_attach_thr0.1.json"),
            ("E348 w", "attached ends, proximity wiring", "E324 s", "runs/e348_w_s{s}/ends_card_attach_thr0.1.json", "runs/e324_s_s{s}/ends_card_attach_thr0.1.json")]
    rows = []
    for arm, ro, ctl, pa, pb in spec:
        va, vb = [tv(pa.format(s=s)) for s in (0, 1)], [tv(pb.format(s=s)) for s in (0, 1)]
        da, db = [tv(pa.format(s=s), "dup_frac") for s in (0, 1)], [tv(pb.format(s=s), "dup_frac") for s in (0, 1)]
        d = [None if None in (x, y) else x - y for x, y in zip(va, vb)]
        dd = [None if None in (x, y) else x - y for x, y in zip(da, db)]
        dds = " / ".join("-" if v is None else (f"<span style='color:var(--bad)'><b>{v:+.4f}</b></span>" if v > 0.01 else f"{v:+.4f}") for v in dd)
        rows.append([f"<b>{esc(arm)}</b>", esc(ro), esc(ctl), pf(va), pf(vb), pf(d, sg), dds])
    return tbl(["arm", "read-out", "control (same seed)", "topo F1 arm (s0 / s1)", "topo F1 control", "difference", "dup_frac difference (guard +0.01)"], rows, num=(3, 4, 5, 6))


def update_section():
    E = esc
    res = [["E317", "geometry loss on the delivered connector (E312 recipe)", pill("info", "TIE, kept"),
            "snapped topo F1 over E312 +0.0077 / +0.0098 (KEEP needs more than +0.01 on both seeds); snap-nogeom read-out over u' +0.0199 / +0.0212; unsnapped +0.0159 / +0.0183; every guard held",
            "the loss makes the shape a constraint and is already part of the E324 s recipe; no separate adoption"],
           ["E318", "position scale per keypoint (Laplace NLL) on top of E317", pill("no", "CUT"),
            "snapped topo F1 against the control E317 g -0.0009 / -0.0037 (inside the band); Spearman(sigma, end error) +0.247 / +0.234 against the bar 0.3, so clause 1 of the informativeness requirement fails on both seeds; clause 2 holds (+0.3995 / +0.3510 against +0.03)",
            "the scale is information but moves no graph number; the same loss and head is not proposed again"],
           ["E319", "heading-guided forward / backward exchange in the maps head", pill("dg", "gate not met, closed"),
            "maps keypoint F1 against E307 +0.0026 / +0.0082 (gate +0.01); entry recall beyond 35 m -0.0035 / -0.0073; the pair stage was not run, so there is no topo F1",
            "the head without the block stays"],
           ["E325", "learned wiring head (545 parameters) replaces the nearest-piece rule", pill("ok", "KEEP"),
            "snapped topo F1 over E312 +0.0023 / +0.0032 (paired intervals exclude 0, stay below +0.005); agreement with the oracle choice on the real choices 0.780 / 0.786 against the rule's 0.757 / 0.751; guards held",
            "trained on stored records of the frozen system, not jointly; the final choice is still an argmax (a named output selection); transfer to E324 s = E350 (planned)"],
           ["E341", "evidence without the 0.5 m paint branch (E312 recipe)", pill("ok", "ADOPT"),
            "unsnapped topo F1 against E312 +0.0000 / -0.0080, snapped +0.0005 / -0.0060 (bound 0.010); saves 1.477 GMACs of the system's 7.955 (-18.6 %) and 0.037 M of 2.632 M parameters",
            "seed 1 unsnapped sits 0.002 inside the bound (interval [-0.0167, -0.0001]): ADOPT says within the band, not no loss; the ms are an estimate until E342 part B"],
           ["E348", "the E341 removal on the E324 s recipe", pill("no", "NOT adopted"),
            "the one failing guard: PRIMARY seed 1 dup_frac +0.0109 against the bound +0.01; topo F1 against E324 s inside the band: snapped -0.0028 / -0.0006, PRIMARY -0.0014 / +0.0022, attached -0.0013 / +0.0014",
            "source: the rule log temp/e348/rule.log (the card has no result section yet); E348.w_s1 crashed at step 5038 (ENOMEM) and was resumed from its checkpoint (named deviation)"],
           ["E334", "one joint training stage instead of two", pill("no", "NOT adopted"),
            "one guard: dup_frac +0.0104 against +0.01 on seed 0, unsnapped; topo F1 inside the band (unsnapped -0.0011 / -0.0034, snapped -0.0024 / -0.0049); pair precision at E312's recall -0.057 / -0.077",
            "the pair head is worse in a way the rule does not see; no re-run on the strength of a near miss"],
           ["E331, E332", "pair stage without the evidence; maps head without paint branch and evidence", pill("no", "NOT adopted (both)"),
            "unsnapped topo F1 against E312: E331 -0.0232 / -0.0124, E332 arm a -0.0189 / -0.0106 (bound 0.010); snapped -0.0029 / -0.0007 and -0.0045 / -0.0015",
            "the loss is the evidence, not the paint branch, which opened E341 (both results date from 2026-11-14, after the last build)"],
           ["E333, E339, E340", "width ladder of the maps head (40, 40 taught by the wide head, 32)", pill("dg", "withdrawn (D92)"), "never run", "the width stays 64; compute is saved by other means"],
           ["E338", "stack check: E324 s heads on the graph of c2 (CPU)", pill("info", "TIE, kept"),
            "sc 0.7066 / 0.7016 against s 0.6981 / 0.7010: +0.0085 / +0.0006 (seed 1 misses the KEEP bound); the additive prediction is a hit on every number; against E312 snapped +0.0242 / +0.0152",
            "c2's own gain on seed 1 is small (+0.0020 over E312); c2 and E324 s stay two separate candidates; adoption is Mirko's"],
           ["E337", "attached ends against the snap's geometry (stored cards)", pill("dg", "answered, no training"),
            "the gap +0.006932 / +0.006155 topo F1 is geometry: the attached end and the keypoint wire the same piece in 96.2 % / 96.6 % of the ends",
            "the sharpening lever stays a stub, not scheduled"],
           ["E342", "where the 17.09 ms of the junction forward go", pill("info", "part A installed, B open"),
            "jgt._groups vectorised: 0.80 ms to 0.095 ms per set, labels equal on 3 322 real sets; about 2.2 ms to 0.75 ms per frame (-1.4 ms of 17.09 ms)",
            "installed by commit 27fee60b; part B (idle-GPU profile of the pair head) is not run"],
           ["E343", "the one TEST of E324 s (D95)", pill("info", "ON HOLD"),
            "no TEST number exists. Dress rehearsal on VAL: GPU branch seed 0 PASS, seed 1 FAIL by one number: vertices per 100 m differs by 4.87e-3 (snapped) against the tolerance 3e-3",
            "decision with Mirko: kill, or a recorded tolerance of 1e-2; no log row, no grant, no TEST until then"],
           ["E345", "pruning by importance, step 0 of D99", pill("dg", "running"), "no table yet (the card folder holds no results.md)", "a measurement on the stored E312 and E324 s models; no training"]]
    hard = ("<ul class='tight'>"
            "<li><b>Wiring of the connector ends</b> (the one-piece, nearest-piece rule: 4 m / 45 degrees, at most 8 candidate places): now REPLACEABLE by the learned head of E325 (KEEP on E312). <b>The final choice is still an argmax</b>, a named output selection; the weights are not edge probabilities; the head is trained on stored records, not jointly. The PRIMARY read-out of E324 s keeps the nearest-piece rule; E350 plans the transfer to the snapped read-out of E324 s.</li>"
            "<li><b>Probability threshold p &gt;= 0.1</b>: a connector is delivered above it; the planner never sees the probability.</li>"
            "<li><b>Top-K keypoint choice</b> (K = 32, 3 x 3 peaks of the end heat): the position inside the window is a soft-argmax, the existence of a token is a hard choice.</li>"
            "<li><b>Close-node step 1.5 m</b> on the read-out of B: greedy removal of the lower-probability node; the soft versions tried so far are E277 (CUT), E279 (KEEP at 1.25 m, Mirko's choice open) and E288 (TIE as a read-out replay); the layer E292 is not covered by this update.</li>"
            "<li><b>Douglas-Peucker 0.3 m and the node threshold</b>: the budget of 14.3 vertices per 100 m is not learned.</li>"
            "<li><b>Snap geometry</b> in the read-out of record (own keypoints snapped): the attached ends of E320 / E324 give up +0.0069 / +0.0062 topo F1 against it, and that gap is geometry (E337).</li>"
            "<li><b>Frozen sensor features</b> (D50): the loss never reaches the camera / radar encoder.</li></ul>")
    plans = [["E342 part B", "idle-GPU profile of the pair head (Sunday night)", "not run"],
             ["E344", "plan from five rule steps to none: the query head on the 1 m grid (Q-Lane); Q0 is E346, Q1 and later wait for Mirko's decision on D84 point 6", "plan only"],
             ["E346", "ceiling of the one-graph code on all 873 VAL frames (B's node + link code against a clean control; 1 m grid; node chains through the junction)", "built in scratch, 31-frame smoke only; open kill item named in the card"],
             ["E347", "the junction ends on the 2 m node grid: can one grid carry the ends, and does a one-to-one target make the peak rule unnecessary?", "plan card, nothing started"],
             ["E349", "the per-frame ground truth of the pair stage computed once (cache + pair-only item path)", "plan card, not built"],
             ["E350", "the E325 wiring head trained on E324 s and read out at its snapped graph", "plan card; phase B waits for the TEST end of E343"]]
    eff = ("<ul class='tight'>"
           "<li><b>GPU utilisation of the trainings is low:</b> about 15 % over 12 s on 2026-11-14 (0 % with peaks to 100 %); the main processes of the two E334 trainings sit at 13 - 17 % CPU, the six data workers (3 per job) at about 95 % each, 12 cores with about 44 % idle. The step (about 2 s) is bound by the loader's CPU work, not by the GPU (E342 section 4).</li>"
           "<li><b>D101:</b> 4 data workers per new training start; against 3 workers every field the pair stage reads is bit-identical (only the simulated map is unseeded, and the pair and maps stages do not read it). Loader per batch of 8 on the loaded box: 2.4 - 2.5 s with 3 workers, 1.8 s with 4.</li>"
           "<li><b>E349</b> plans a cache of the per-frame geometry and a pair-only item path (not built): control 1.83 s per batch, the card predicts 1.5 s less, the rule asks for at least 0.9 s.</li>"
           "<li><b>Saved so far:</b> E341 1.477 GMACs and 0.037 M parameters (ms estimated, not measured); E342 part A about 1.4 ms per frame of the 17.09 ms.</li></ul>")
    dec = [["D92", "no blunt narrowing: the width of the heads stays 64; compute is saved otherwise; E333, E339, E340 withdrawn"],
           ["D93", "dead code may be deleted, only in Spurmitte, in batches after a tag, git history is the way back"],
           ["D94", "system of record = E324 s (with c2; the stack E338 the combined candidate); new arms are compared with E324 s of the same seed; PROBLEM.md and the registry change only after the TEST"],
           ["D95", "ONE TEST use of E324 s (both seeds), control E312's stored TEST cards; the readings R1 - R3 of card E343 confirmed (Confirmed lines of 2026-11-14 22:25 and 23:00); the technical gates of the card stay"],
           ["D96", "the holds on E317, E318 and E319 are lifted; they run on the recipe their cards name"],
           ["D97", "removal of the old reference chain of loops 8 - 18 (option B of D89), in steps: freeze the reference numbers and tag first; waits for the TEST end of E343"],
           ["D98", "weekend rule (GPU jobs by day only one at a time, nice, host gate): void since D100"],
           ["D99", "pruning by importance instead of narrowing: a size-against-loss table (step 0 = E345), then cards for two or three sizes"],
           ["D100", "the weekend rule D98 is void: chains run freely up to the two GPU slots, no host gate; the hard rules and the weekday GPU window stay"],
           ["D101", "4 data workers for every new training start (checked bit-identical against 3)"]]
    return ('<section id="update"><h2>Update 2026-11-15</h2>'
            "<p>State of 2026-11-15; the report was last built on 2026-11-14. <b>The system of record is unchanged:</b> E324 s (the E320 recipe at 6000 pair steps), VAL snapped topo F1 " + pair(head(C_E324S, "topo_F1")) + " against " + pair(head(C_E312S, "topo_F1")) + " for E312; c2 (B fine-tuned) and the stack E338 are candidates next to it (D94). TEST has been used once, for E312 (E321); there is no TEST number for E324 s. Everything below is VAL, two seeds, 873 frames; the tokens are the cards' and the written rule of each card decides.</p>"
            "<h3>What was measured since the last build</h3>"
            + tbl(["experiment", "what", "verdict (written rule)", "key numbers (card, VAL, seed 0 / seed 1)", "reading"], [[f"<b>{a}</b>", E(b), c, E(d), E(e)] for a, b, c, d, e in res])
            + "<h3>The same arms read from the result files</h3>" + readout_table()
            + "<p class='cap'>Read from runs/*/ends_card*.json (topo F1 and dup_frac of the delivered graph, p &gt;= 0.1, VAL 873 frames); the difference is arm minus control of the same seed. E348 and E338 are read against E324 s, the others against the control of their card. dup_frac differences above the guard +0.01 are marked red.</p>"
            "<h3>Hard steps still in the system</h3>" + hard
            + "<div class='note warn'><b>Open item: E343 is on hold.</b> The dress-rehearsal check of the one TEST use of E324 s failed by one guard number (vertices per 100 m, 4.87e-3 against 3e-3, on seed 1 of the GPU branch; the stack arm reproduces exactly). By the letter of the card the check failed and nothing was started; Mirko decides between a kill and a recorded tolerance of 1e-2. <b>No TEST number of E324 s exists.</b></div>"
            "<h3>Efficiency</h3>" + eff
            + "<h3>Planned and running cards</h3>" + tbl(["card", "what", "state"], [[f"<b>{a}</b>", E(b), E(c)] for a, b, c in plans])
            + "<h3>Decisions D92 - D101</h3>" + tbl(["decision", "what (as recorded in experiment_execution/decisions/)"], [[f"<b>{a}</b>", E(b)] for a, b in dec])
            + "</section>")


'''
m = re.search(r"^# -+ page$", s, re.M)
assert m
s = s[:m.start()] + UPDATE + s[m.start():]
open(p, "w", encoding="utf-8").write(s)

# ------------------------------------------------------------------------------------------------------------------ 3. pack_total.sh: the cards and runs of E320 - E350
q = os.path.join(HERE, "pack_total.sh")
t = open(q, encoding="utf-8").read()
for old, new in (("the cards E273 - E319", "the cards E273 - E350"),
                 ("experiments/E31[0-9]_*; do", "experiments/E31[0-9]_* experiments/E3[2-5][0-9]_*; do"),
                 ("runs/e314_p_s? runs/e273/e273_p_s?; do", "runs/e314_p_s? runs/e315_p_s? runs/e316_p_s? runs/e317_g_s? runs/e318_u_s? runs/e320_a_s? runs/e324_s_s? runs/e325_w_s? runs/e331_e_s? runs/e332_pa_s? runs/e334_o_s? runs/e341_p_s? runs/e348_w_s? runs/e273/e273_p_s?; do"),
                 ("-name 'card.json' \\)", "-name 'card.json' -o -name 'wire_agreement.json' \\)"),
                 ('mkdir -p "$OUT/ergebnisdateien/temp/snap";', 'mkdir -p "$OUT/ergebnisdateien/temp/e348"; cp temp/e348/rule.log "$OUT/ergebnisdateien/temp/e348/" 2>/dev/null\nmkdir -p "$OUT/ergebnisdateien/temp/snap";'),
                 ("temp/jr/pack_total.sh \"$OUT/skripte/\"", "temp/jr/pack_total.sh temp/jr/patch_total_2.py temp/jr/patch_total_3.py temp/jr/patch_total_4.py temp/jr/patch_total_5.py \"$OUT/skripte/\"")):
    assert t.count(old) == 1, old
    t = t.replace(old, new, 1)
open(q, "w", encoding="utf-8", newline="\n").write(t)
print("patched", len(T), "rows", len(s), "bytes of builder")
