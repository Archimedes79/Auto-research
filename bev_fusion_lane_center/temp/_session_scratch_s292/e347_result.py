"""E347 result section into the card (section 3 Summary), numbers from runs/e347_<arm>_s<S>/jmaps_val.json and runs/e332_ma_s<S>/jmaps_val_legacy.json (key gate)."""
import io
import json
import time

p = "experiments/E347_ends-on-the-2m-node-grid/experiment.md"
s = io.open(p, encoding="utf-8", newline="").read()
assert s.count("(empty until the runs exist)") == 1


def row(label, S, d, key, sub):
    r = d["gate"][key][sub]
    return (f"| {S} | {label} | {r['thr']['ent']:.2f} / {r['thr']['ext']:.2f} | {r['f1_tol3']:.4f} | {r['f1_tol1_5']:.4f} | {r['crowded_recall']['ent']:.4f} | {r['dup_share']:.4f} | {d['zone_iou']:.4f} | {d['core_iou']:.4f} |")


T = "| seed | run, read-out (at its best threshold on the 0.05 - 0.95 grid) | best thr ent / ext | keypoint F1 at 3 m | F1 at 1.5 m | crowded entry recall | surplus-duplicate share | zone IoU | core IoU |\n|---|---|---|---|---|---|---|---|---|\n"
deltas = {}
for S in (0, 1):
    c = json.load(open(f"runs/e332_ma_s{S}/jmaps_val_legacy.json"))
    T += row("**control** E332 arm a (1 m head, no paint), peak read-out", S, c, "ours", "at_best") + "\n"
    for arm, name in (("g", "arm g (2 m, Gaussian sigma 1 m)"), ("o", "arm o (2 m, one-hot target)")):
        d = json.load(open(f"runs/e347_{arm}_s{S}/jmaps_val.json"))
        T += row(f"{name}, its rule read-out ({'3 x 3 peak + threshold' if arm == 'g' else 'threshold alone'})", S, d, "ours", "at_best") + "\n"
        T += row(f"{name}, the other read-out ({'threshold alone' if arm == 'g' else '3 x 3 peak + threshold'})", S, d, "ours_alt", "at_best") + "\n"
        a, b = d["gate"]["ours"]["at_best"], c["gate"]["ours"]["at_best"]
        deltas[(arm, S)] = (a["f1_tol3"] - b["f1_tol3"], a["f1_tol1_5"] - b["f1_tol1_5"], a["crowded_recall"]["ent"] - b["crowded_recall"]["ent"])
D = "| arm | seed | F1 at 3 m minus control | F1 at 1.5 m minus control | crowded entry recall minus control | FAIL clause (3 m more than 0.03 below, or 1.5 m more than 0.05 below) |\n|---|---|---|---|---|---|\n"
for (arm, S), (f3, f15, cr) in sorted(deltas.items()):
    D += f"| {arm} | {S} | {f3:+.4f} | {f15:+.4f} | {cr:+.4f} | {'FAIL' if (f3 < -0.03 or f15 < -0.05) else 'no'} |\n"
now = time.strftime("%H:%M")
txt = f"""## 3. Summary (2026-11-15 {now}; stage 1 read on both seeds, stage 2 NOT run)
**Result: both arms FAIL the stage-1 gate on both seeds; by the written rule the result is 'the ends need the 1 m grid' for this recipe (question 1 of E344 section 7, for the ends), and E347 closes at stage 1.** No stage 2 was run (the card fixes: stage 2 only when an arm does not FAIL). The code stays staged and is NOT installed (nothing to install); the four maps runs cost about 3.4 slot hours (each about 51 minutes at 1.0 s per step on a loaded host), the control scoring about 0.75 h of wall time (two runs of 22 minutes, CPU), the arm scoring about 0.8 h (four runs of 11 - 14 minutes).
**Numbers (VAL 873 frames, `jmaps_eval.py score` of the staged scorer, every run at its OWN best threshold on the same grid; the control numbers were written into this card BEFORE any arm was read, section 2b):**

{T}
**Stage-1 clauses against the control of the same seed (rule read-out at the run's best threshold):**

{D}
**Reading.** (1) **Arm g (2 m head, Gaussian heat, 3 x 3 peak):** the sub-cell offset works (the F1 at 1.5 m is only 0.018 - 0.019 below the 1 m head), but the F1 at 3 m is 0.049 - 0.051 below and the crowded entry recall 0.17 - 0.19 below: the predicted mechanism (neighbouring ends a lane width apart fall into ADJACENT 2 m cells and the 3 x 3 peak keeps one of them), stronger than predicted. In its threshold-alone read-out g keeps the recall of the crowded ends (0.74 - 0.75) but fires clusters (surplus-duplicate share 0.35, F1 at 3 m 0.58). (2) **Arm o (one-to-one target, focal loss, threshold alone):** the one-hot target did NOT make the output one-to-one: at its best threshold 0.18 - 0.25 of its keypoints are surplus duplicates (control 0.001; the card's bound is 0.01 above arm g) and at p >= 0.1, the threshold of the pair stage, 0.78 - 0.79 of the keypoints are duplicates (about 79 keypoints per frame); its F1 at 3 m is 0.125 - 0.127 below. With the peak step added (the other read-out) o is better (0.689 / 0.691) but still 0.06 - 0.07 below the control and below g. (3) The 2 x 2 of target and read-out is complete: no cell reaches the control (0.757 / 0.751); the best is g with the peak step. (4) The zone and core maps of the 2 m head are worse too (zone IoU 0.747 - 0.759 against 0.773 - 0.776, core IoU 0.496 - 0.539 against 0.605 - 0.607): the 1 m features carry more than the end heat. **Predictions (written before; guesses):** arm g F1 at 3 m -0.02, crowded -0.10: MISSED (-0.049 / -0.051, -0.170 / -0.186); arm o F1 at 3 m -0.015, at 1.5 m -0.03, crowded -0.03, duplicates within the bound: MISSED (-0.127 / -0.125, -0.055 / -0.052, -0.054 / -0.098, duplicates 0.18 / 0.25).
**Compute (not realised, reported):** `jmodel_cost.py` of the staged tree: the 2 m maps head 0.762 GMACs against 1.779 for the 1 m head without the paint branch (E341), the system 5.461 against 6.477 GMACs (about -1.0 GMACs); parameters of the head 0.31 M trainable at the maps stage; ms per frame not measured (the gate failed first). So the saving that the card was after (about 1 GMACs of the junction head, and the end on the grid of B's nodes) costs 0.05 - 0.13 of the end F1 at 3 m with this recipe.
**What this decides / does not decide.** It decides: with the E307 / E332 maps recipe (3000 steps, batch 8, lr 1e-3, no change of the losses or of the training length), a 2 m end head with an offset is not a replacement for the 1 m head, and a one-hot target with a focal loss does not remove the peak rule (arm o: the duplicates stay). It does not decide: a longer training of the 2 m head, a one-to-one assignment inside the loss (a matching of predicted to GT ends, as in set-prediction heads) instead of a one-hot target, or ends on a 1 m grid with a 2 m node (the end IS a node of B only through the snap): each is its own card; none is started today. The arm o result matters for the E2E goal: the peak step stays a named step on the end side (it is not replaced by the target).
**Process notes.** Kill 2 (step 500) passed on all four arms (the first 56 VAL frames: g_s0 479 entry keypoints at 0.3, o_s0 1505; for g_s1 and o_s1 the read shows predictions at the low thresholds of the gate sweep (708 and 15198 at 0.05), the 0.3 row was not read; the focal head of arm o fires broadly at step 500 and sharpens later). The control's extended numbers were computed after the launch and written into the card before any arm was read (named deviation (e)). The scorer reproduces the stored control numbers at 0.3 (3 m F1 0.7409 / 0.7356, 1.5 m F1 0.5631 / 0.5657). The delta critic findings of 11:35 were taken (section 2b). Runs: `E347.g_s0 / g_s1 / o_s0 / o_s1`, `E347.kill2_<arm>_s<S>`, `E347.c_score_s0 / s1`, `E347.score_<arm>_s<S>` (ledger notes); records `runs/e347_<arm>_s<S>/jmaps_val.json`, `runs/e332_ma_s<S>/jmaps_val_legacy.json`.
"""
s = s.replace("## 3. Summary\n(empty until the runs exist)", txt.rstrip("\n"))
assert "## 3. Summary (2026" in s
io.open(p, "w", encoding="utf-8", newline="").write(s)
print(T)
print(D)
