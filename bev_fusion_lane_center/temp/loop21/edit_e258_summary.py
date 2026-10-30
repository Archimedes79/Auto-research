import glob
p = glob.glob("experiments/E258*/experiment.md")[0]
t = open(p, encoding="utf-8").read()
i = t.index("## 3. Summary")
new = """## 3. Summary (2026-10-30, runs E258.p0 (profile), E258.a0 (step 1b), E258.b0 (step 2); VAL 873, cached raw outputs, CPU, load average 3.5 - 4.2 during all timings)
- **Decision (by hand against the written rules, the metric is not a card metric): the profile found two stages above 15 % (prediction hit); no stage is dropped for the product as a whole; the junction anchoring is switched off for the j1 family as a candidate (registry entry `fast_recall_na`).**
- **Step 0, profile** (`profile_step0.txt`, 200 frames): share of the assembly: fork relocation 21 %, junction anchoring 16 %, path-union loop 16 %, simplify (spur / bubble) 11 %, fork pruning 6 %.
- **Step 1b, remove a stage** (rule: Q loss <= 0.003, J loss <= 0.01, vert100 change <= 0.3 on both models, saving >= 0.8 ms):

| model | variant | dQ | dJ | dvert100 | saving ms (relative, loaded) |
|---|---|---|---|---|---|
| fast_recall | relocation off | -0.0111 | -0.0540 | +1.41 | +2.3 |
| fast_recall | anchoring off | -0.0004 | -0.0042 | +0.04 | +2.7 |
| fast_recall | both off | -0.0141 | -0.0698 | +1.41 | +5.8 |
| champion | relocation off | -0.0110 | -0.0524 | +1.07 | +1.7 |
| champion | anchoring off | -0.0076 | -0.0356 | +0.06 | +2.3 |
| champion | both off | -0.0216 | -0.1007 | +1.09 | +4.9 |

  Relocation stays (it carries about 0.05 J on both models, the closed question E215 / E218 concerned its settings, not its removal). Anchoring is needed by the champion (the 288 / 144 chain) and nearly free for fast_recall.
- **Step 2, four seeds of the recall fine-tune (j1rc s0 - s3), anchoring off:** dQ -0.0004 / +0.0005 / -0.0010 / -0.0011; dJ -0.0042 / -0.0003 / -0.0068 / -0.0072; dvert100 +0.04 to +0.05; topo F1 +0.0046 / +0.0048 / +0.0040 / +0.0042; lanes found -0.0021 / -0.0022 / -0.0016 / -0.0023. The rule (all four seeds) holds; prediction (3 of 4) hit. The official card of the new entry (`evaluate.py fast_recall_na`, VAL 873): S 0.771, Q 0.739, J 0.443, topo F1 0.742, lanes found 0.734, R@1 far 0.726, vert100 15.02 against fast_recall 0.772 / 0.739 / 0.447 / 0.737 / 0.736 / 0.727 / 14.98.
- **Speed, honestly:** the saving is relative and was measured on a machine at load 3.5 - 4.2 (base assembly 13.9 - 17.5 ms against 7.5 - 7.8 ms idle in E253). Anchoring is 16 - 25 % of the assembly in every pairing; applied to the idle 7.8 ms that is about 1.3 - 2 ms of the 18.5 ms chain (an estimate, not a measurement). The idle repeat belongs in the next quiet window (E253 driver with the new entry).
- **Weaknesses:** one base model family; the champion needs the stage, so a registry choice per variant (not a global switch) is needed; J is a smaller metric weight than Q but the D4 text quotes J, so the 0.004 - 0.007 loss of J must be stated; the timing carries no idle confirmation.
- **Not done:** an identical-graph optimisation of the stages themselves (step 1a): the profile suggests `relocate_forks` and `anchor_junctions_fast` as targets, both python loops over small arrays.
"""
open(p, "w", encoding="utf-8").write(t[:i] + new)
