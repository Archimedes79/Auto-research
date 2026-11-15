p = "experiments/E293_junction-depth-evidence/experiment.md"
s = open(p, encoding="utf-8").read()
a = s.index("## 3. Result with more frames")
b = s.index("## 4. What follows")
new = '''## 3. Result with 240 frames (2026-11-11, CPU, the first 240 VAL frames = about 6 scenes, 3295 connector runs, seed 0; `runs/e293/depth_probe_s0.json`; the sample is not the whole split)
| depth from the entry | true: heat >= 0.3 | false paths: heat >= 0.3 (mean) | painted within 1 m (true) | Hermite (chord tangent) within 0.5 m / 1 m | error median / p90 |
|---|---|---|---|---|---|
| 0 - 2 m | 0.97 | 0.97 (0.90) | 0.05 | 1.00 / 1.00 | 0.03 / 0.08 m |
| 2 - 5 m | 0.98 | 0.98 (0.92) | 0.06 | 0.94 / 0.995 | 0.06 / 0.39 m |
| 5 - 10 m | 0.99 | 0.98 (0.92) | 0.07 | 0.75 / 0.89 | 0.07 / 1.07 m |
| 10 - 15 m | 0.99 | 0.95 (0.88) | 0.08 | 0.69 / 0.80 | 0.07 / 1.82 m |
| 15 - 25 m | 0.995 | 0.91 (0.85) | 0.06 | 0.68 / 0.77 | 0.09 / 1.95 m |
| 25 m and deeper | 0.995 | 0.87 (0.82) | 0.05 | 0.78 / 0.84 | 0.07 / 2.30 m |
Readings (the 16-frame preview of section 2 was too optimistic about the curve): (a) **the lane heat does not tell the true connector from a false one for the first 10 m** (0.97 - 0.99 against 0.97 - 0.98), it starts to separate at 10 - 15 m (0.99 against 0.95) and at 15 m and deeper (0.995 against 0.91 / 0.87): the pixel stage sees the lanes near the exit side, not in the middle; (b) paint on the true connectors inside the junction is 5 - 8 % (lane dividers and boundaries that run along or through the zone); (c) **the Hermite curve with the CHORD tangent between the true ends is good for the first 5 m (99.5 % within 1 m) and only 77 - 89 % within 1 m beyond 5 m depth** (median error 7 cm, but the p90 error is 1.1 - 2.3 m: long arcs and S-curves): the shape inside the junction is not fixed by the two ends and their directions alone; E273's learned handle scales (two per long link) fixed 98.7 % of the connector points in the oracle (card E273), which is the better representation. So: **points can be put into the junction as far as the pair is known AND the handle scales of the curve are learned (or one midpoint per connector); the chord-tangent curve is accurate for the first 5 m, beyond that it needs a shape parameter.** The first 5 m after the entry are where the pixel evidence for the exit side is still weak and the curve is cheap: points there are safe.

'''
s = s[:a] + new + s[b:]
s = s.replace("(1) No dense node target inside the junction: the representation is entry, exit, pair and the Hermite curve (E287); (2) a MIDPOINT per delivered connector as an extra learned correction (p90 tail) is a cheap refinement of E287 if the pair numbers are good;",
              "(1) No dense node target inside the junction: the representation is entry, exit, pair and a curve with learned handle scales (E273 has them for the long links; E287's delivered connectors use the chord tangent: the curve of E287 should get the two handle scales as an output of the pair head, **E303** below); (2) a MIDPOINT per delivered connector (one learned offset to the curve) is the cheaper alternative;")
open(p, "w", encoding="utf-8").write(s)
print("ok")
