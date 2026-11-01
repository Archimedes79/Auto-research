p = "experiments/E257_dp-output-simplification/experiment.md"
t = open(p, encoding="utf-8").read()
i = t.index("## 3. Summary")
new = """## 3. Summary (2026-10-30, run E257.a0, 09:20 - 09:38; `temp/loop21/e257_simplify.py`, output `dp_sweep.txt`; VAL 873, cached raw outputs, CPU)
- **Decision by the tool (`decide.py --experiment E257 --write`): KEEP** (vert100 a - c = 6.15 fewer vertices per 100 m, mean of the two models; Q -0.0011, F1@0.5 -0.0045, topo F1 -0.0001, J 0.0000; the tool pools the two models, it does not read the guards per model). **Per-model reading of the card (guards "on both models"): at the declared eps 0.3 the champion misses the F1@0.5 guard by 0.0003 (-0.0053 against -0.005)**; the sweep shows eps 0.2 holds every guard on both models. So: KEEP as an output stage, with eps 0.2 as the recommended tolerance (chosen from the sweep after the fact, labelled so; it is not tuned on anything but the guards already declared).
- **Prediction** {-5.0 vertices, tolerance 1.5}: eps 0.3 gave -6.0 (fast_recall) and -6.3 (champion): within the tolerance (errors -1.0 / -1.3), hit. The tool prints "error +11.2" because it compares the card's value -5.0 with an effect that is reported with improvement positive (+6.15): the sign convention for direction "lower" is inconsistent (N17); the card should have said +5.0.
- **Layer 1 passed** for both models (stored Q and vert100 reproduced to four digits); **layer 0 passed**: every end and every shared node kept, maximum deviation of an original vertex from the simplified polyline 0.100 / 0.200 / 0.300 / 0.500 / 0.750 / 1.000 m = eps on all 873 frames (asserted).
- **Numbers (VAL 873, node-id vertex count after simplification):**

| model | eps (m) | vert100 | Q | F1@0.5 | F1@1.0 | topo F1 | J | lanes found | R@1 far |
|---|---|---|---|---|---|---|---|---|---|
| fast_recall | none | 14.98 | 0.7392 | 0.6881 | 0.8658 | 0.7369 | 0.4469 | 0.7364 | 0.7275 |
| fast_recall | 0.1 | 11.41 | 0.7392 | 0.6879 | 0.8660 | 0.7370 | 0.4469 | 0.7361 | 0.7276 |
| fast_recall | 0.2 | 9.86 | 0.7389 | 0.6863 | 0.8664 | 0.7376 | 0.4469 | 0.7369 | 0.7279 |
| fast_recall | 0.3 | 8.99 | 0.7384 | 0.6844 | 0.8661 | 0.7371 | 0.4469 | 0.7368 | 0.7280 |
| fast_recall | 0.5 | 8.01 | 0.7366 | 0.6772 | 0.8659 | 0.7365 | 0.4469 | 0.7346 | 0.7284 |
| fast_recall | 1.0 | 7.20 | 0.7236 | 0.6353 | 0.8573 | 0.7255 | 0.4469 | 0.7199 | 0.7240 |
| champion | none | 14.83 | 0.7350 | 0.6912 | 0.8696 | 0.7413 | 0.4092 | 0.6948 | 0.7087 |
| champion | 0.2 | 9.52 | 0.7345 | 0.6895 | 0.8696 | 0.7414 | 0.4092 | 0.6947 | 0.7084 |
| champion | 0.3 | 8.52 | 0.7336 | 0.6859 | 0.8693 | 0.7408 | 0.4092 | 0.6941 | 0.7082 |
| champion | 0.5 | 7.49 | 0.7305 | 0.6741 | 0.8685 | 0.7400 | 0.4092 | 0.6923 | 0.7099 |
| champion | 1.0 | 6.60 | 0.7173 | 0.6333 | 0.8586 | 0.7282 | 0.4092 | 0.6785 | 0.7075 |

- **What it says:** 33 - 43 % of the vertices of the assembled graphs are not needed to describe the polylines within 0.2 - 0.3 m. The loss shows in F1@0.5 first (it needs the geometry within 0.5 m), then in Q beyond eps 0.5; F1@1.0, J, topo F1, lanes found and R@1 far do not move up to eps 0.3 (J and topo F1 are exactly unchanged by construction: ends and shared nodes are kept). Paired scene bootstrap of the Q change at eps 0.3: [-0.0014, -0.0001] (fast_recall), [-0.0017, -0.0011] (champion): the loss is real but about 1 / 10 of the 0.010 band.
- **Secondary (matched vertex counts, vtx_r sweep, nothing tuned):** the radius merge reaches fewer vertices only at a high price: fast_recall without simplification has Q 0.616 at 12.0 and 0.584 at 11.0 vertices (champion 0.594 at 10.95, 0.564 at 9.9) while the simplified graph has Q 0.738 at 9.0 (fast_recall) and 0.734 at 8.5 (champion). The unsimplified curve does not reach below 10 vertices in this grid, so no interpolation at 8 / 10 / 12 is claimed; the direct reading is Q 0.738 at 9.0 against 0.58 - 0.62 at 11 - 12.
- **What it does not say:** the model, its training and the Q at 14.3 node-id vertices are unchanged: this is an output-size result, not a quality result. The budget 14.3 of PROBLEM.md does not say how vertices are counted; by the oracle's own count (Douglas-Peucker 0.3 m on the ground truth, 14.3) the graphs were already at 12.2 / 11.3 before the stage. Whether the budget is quoted after output simplification is Mirko's decision (protected file): filed as D20. A downstream consumer that needs the node spacing (the route head along the edges, the temporal memory) must be checked before the stage becomes a default: in this experiment it acts only on the output graph.
- **Weakness:** two models, one set of cached outputs, no new training; the stage is deterministic, so no seed noise enters; the eps recommendation 0.2 rests on a six-point sweep with the guards fixed beforehand.
- **Next:** promote `simplify_dp` as a default-off option of `assemble()` with a spec test, reproduce through `evaluate.py`, `ar-reviewer` pass (code); D20 to Mirko; check the route head and the map assignment on simplified graphs (they read vertices) before any default.
"""
open(p, "w", encoding="utf-8").write(t[:i] + new)
