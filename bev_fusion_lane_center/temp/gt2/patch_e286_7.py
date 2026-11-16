import subprocess
tab = subprocess.run(["python3", "temp/gt2/e286_table.py"], capture_output=True, text=True, cwd=".").stdout.split("\n{")[0].strip()
p = "experiments/E286_junction-pixel-maps/experiment.md"
s = open(p, encoding="utf-8").read()
new = f'''

## 7. Result of the three context arms, two seeds each (2026-11-12 01:44, `jmaps_eval.py score` of the final checkpoints with the FIXED scorer of the critic pass, VAL 873, `runs/e286_<arm>_s<S>/jmaps_val.json`; arm jj follows from the night chain)
{tab}
Gain of `kp_f1` over c (B's event heat on the same seed): **j0 +0.180 / +0.177, j1 +0.231 / +0.265, j2 +0.240 / +0.259.** The best-threshold F1 (both maps at their own best threshold, the calibration question of the critic) gives the same order: j0 0.61, j1 0.66, j2 0.66, B 0.40.
**By the rule (written before the run, "by hand: keypoint scorer"): KEEP for all three arms on both seeds** (the KEEP line was +0.05, the prediction +0.15). **Base for the later steps: j1** - j2 has the highest mean (0.638 against 0.636) but the card says a tie inside 0.01 goes to the cheaper arm: the attention layer over 25 x 25 tokens adds nothing measurable (+0.002; zone IoU +0.003, core IoU +0.008) for its cost. Context does matter: j0 (1 x 1 / 3 x 3 convolutions only) loses 0.07 against j1 and 0.1 of zone IoU.
**What the splits add (not part of the rule):** (a) the far range (> 35 m) and Singapore stay the weakest (entry recall 0.55 - 0.56 and 0.56 - 0.59) for every arm; (b) **context costs the crowded ends**: j0 finds entries whose neighbour lane's entry is within 4 m / 30 degrees at 0.660 (isolated 0.608), j1 / j2 at 0.54 - 0.56 (isolated 0.65 - 0.66): the dilated context smooths the heat so that two ends of neighbouring lanes merge into one peak; and j0 is better in the far range (0.574 against 0.551); this is the question of E307 (narrower Gaussian targets) and a reason to read the arms by split; (c) the head no longer depends on paint (painted / unpainted recall differ by 1 - 4 points, B's by 17 - 20); (d) the numbers of section 6 (first scorer version) were within 0.01 of these: the critic's fixes did not change the verdict, only the splits (exits painted by a crossing only, crowded split, threshold sweep).
**Honest limits:** c is a weak control (B's event maps were never trained for this); the targets are the legacy half-cell shifted polygons (E306 asks what the exact ones change); zone / core IoU is measured on those targets too.
'''
s = s.rstrip("\n") + new
open(p, "w", encoding="utf-8").write(s)
print("ok")
