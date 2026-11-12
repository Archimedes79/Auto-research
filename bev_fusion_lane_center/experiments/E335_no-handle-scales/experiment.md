# E335 - the E312 pair head without the learned handle scales (simplification arm S2): closed by data, not run

Card of 2026-11-13 00:10, written by the lane chat (fork D) under D84 item 4 (simplify). The pair head of E303 has a small head that predicts two handle scales per pair (the shape of the delivered Hermite curve); its target is a fit of the GT connector (`qjunction.fit_pair_scales`, a grid search in the loader that costs about 0.15 s per frame, +0.31 s per step). Removing it would drop the scale head and the fit. The evidence to decide that is already in the files: the read-out of the SAME trained E312 pairs with the chord tangent instead of the learned scales (`jends_eval.py score --snap --chord --thr 0.1`, E312 / E314 chord control).

## 0. The card (what the tools run)
```json card
{
 "id": "E335",
 "card": "",
 "title": "chord-tangent connectors (no learned handle scales) on the same E312 pairs, snapped at p >= 0.1, against the learned scales: the read-out evidence for the removal of the scale head",
 "kind": "variant",
 "deterministic": true,
 "metric": "topo_F1",
 "direction": "higher",
 "band": 0.01,
 "prediction": {"value": -0.012, "vs": "c"},
 "arms": {
  "c": {"recipe": "E312 with the learned handle scales, snapped p >= 0.1: topo F1 0.6825 / 0.6864 (`runs/e312_p_s<S>/ends_card_snap_thr0.1.json`)"},
  "k": {"recipe": "the same pairs, connectors with the chord tangent: topo F1 0.6707 / 0.6738 (`runs/e312_p_s<S>/ends_card_snap_chord_thr0.1.json`)"}
 },
 "control": "c",
 "variant": "k",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [{"metric": "topo_F1", "vs": "c", "max_loss": 0.01}],
 "budget_gpu_h": 0.0,
 "kill": "the chord card and the learned-scale card are not scored on the same dump (the pair counts differ)",
 "evaluate": {"split": "val"}
}
```
By hand (a measurement, label "by hand: analysis"): chord minus learned scales: topo F1 -0.0118 (seed 0) / -0.0126 (seed 1), lanes found -0.0101 / -0.0171 (0.7376 -> 0.7275, 0.7423 -> 0.7252). More than the band of 0.01 on both seeds, so the scales are worth keeping at the read-out level (E303 / E312: scales worth +0.012). **Verdict: the removal is NOT adopted and no GPU run is made** (a retrained head without scales would have the same curves as the chord read-out; the pairs themselves are trained identically). What is simplified instead is the cost: the fit is a pure function of the lane GT, so it can be cached and the 0.31 s per step go away (E330 section 3, patch proposed, not applied). Closed question: the learned handle scales stay (re-open only with a new mechanism for the shape of the connector, e.g. the geometry loss of E317 or the attachment of E320 changing what the scale head has to learn).
