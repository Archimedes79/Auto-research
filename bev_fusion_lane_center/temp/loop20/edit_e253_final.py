import glob, re
p = glob.glob("experiments/E253*/experiment.md")[0]
t = open(p, encoding="utf-8").read()
i = t.index("## 3. Summary")
new = """## 3. Summary (2026-10-27, run E253.t, 07:20 - 07:38; `temp/loop20/run_e253.sh`, analysis `temp/loop20/analyse_e253.py`)
- **Decision: TIE, session valid.** j1 minus the champion f1 (both pruning 0.3, no recursion) = -0.23 ms (median of the six per-round differences: -0.21 / -0.68 / -0.83 / +0.21 / -0.07 / -0.58); the rule needs -1.0 ms for KEEP. Prediction -0.3 ms (tolerance 1.0): hit. Assembly guard (a - c <= +1.0 ms): passed. j1 is not a speed variant; at equal ms the cheaper arm wins on quality per ms (D4 evidence, no decision).
- **Session validity (card rule, all met):** 6 of 6 valid replicates per configuration (no replicate voided by the guards; one 'machine busy' wait of 20 s), replicate SD 0.44 / 0.29 / 0.21 / 0.39 / 0.26 / 0.76 ms (at most 0.6 in 5 of 6), stored loop-18 values reproduced within 1.0 ms for a, c, f (new 18.52 / 18.75 / 24.85 against 17.15 - 18.10 / 17.54 - 18.23 / 24.20). The machine was quiet: the BEV chat's t1_r4 training had ended 07:19, GPU 0 %, load average 0.9.
- **Medians (ms per frame, 200 frames, six rounds):** j1 18.52, f1 18.75, j1eq 18.49, j1rc 21.14, rc1 20.81, f1 with time recursion 24.85. Recall weight: j1rc - j1eq +2.65 ms (prediction +2.2); rc1 - f1 +2.06 (prediction +2.5); recursion +6.11. Assembly ms: 7.5 - 7.8 for the rec-0 arms without recall, 8.8 - 8.9 for the recall variants. The invalid E243 session had the same order with absolute values 2 - 3 ms higher.
- **Weaknesses:** one session, one machine state; the recursion configuration has the widest spread (SD 0.76 ms, one outlier at 26.7 ms); the fresh-process replicates measure the chain at batch 1 with cached features, not the full system.
- **Cost:** 18 minutes on the idle machine; it had to wait five hours for another chat's training (N1 / D19).
"""
open(p, "w", encoding="utf-8").write(t[:i] + new)
