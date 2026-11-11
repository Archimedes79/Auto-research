# E303 The shape of the connector: two learned handle scales per delivered pair instead of the chord tangent (follows E293, D82)

Card of 2026-11-11 (plan card). E293: the cubic Hermite curve with the CHORD tangent between the true entry and the true exit is within 1 m of 99.5 % of the connector points in the first 5 m but only 77 - 89 % deeper in (p90 error 1.1 - 2.3 m); E273's long links have two learned handle scales (`qtargets.fit_long_scales`, the oracle fits 98.7 % of the connector points within 1 m). The delivered connectors of the E284 end head and of E287 use `qends.curve` with tangent = chord. The scale is a shape output the pair head can learn (the "spline" of the brief): the pair head gets two regressed scales per candidate pair, trained against the fitted scales of the true pairs.

## 0. The card (what the tools run)
```json card
{
 "id": "E303",
 "card": "",
 "title": "the pair head of E287 with two handle scales per pair (regressed, loss = L1 against the scales fitted to the GT connector), the delivered connector curve = Hermite with those scales, against the chord-tangent curve of E287",
 "kind": "block",
 "deterministic": false,
 "metric": "F1@1.0",
 "direction": "higher",
 "band": 0.01,
 "prediction": {"value": 0.01, "vs": "c"},
 "arms": {
  "c": {"recipe": "control = E287 p2 (the best arm of the night) with the chord-tangent connectors; the delivered graph = B + hard step + the head's connectors"},
  "s": {"recipe": "`jends.py` + a scale head (2 outputs, softplus, on the pair features, trained on the true pairs with the scales of `qtargets.fit_long_scales` run on the true ends), the delivered connectors use the predicted scales; same keypoints and pairs as c (a refinement of the curve only)"}
 },
 "control": "c",
 "variant": "s",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "F1@1.0", "vs": "c", "min_effect": 0.01},
  {"metric": "topo_F1", "vs": "c", "max_loss": 0.01},
  {"metric": "dup_frac", "vs": "c", "max_loss": 0.02},
  {"metric": "vert100_dp", "vs": "c", "max_loss": 1.0}
 ],
 "budget_gpu_h": 1.0,
 "kill": "the fitted scales of the true pairs do not reproduce the GT connector (the oracle with the fitted scales and true ends is below 0.95 within 1 m: the fit is the problem, not the head)",
 "evaluate": {"split": "val"}
}
```
By hand (the lane suite numbers through `qends_eval.score`; label "by hand: lane suite"): KEEP when F1@1.0 (the 1 m geometry measure that sees the connector shape) rises by more than 0.01 on both seeds with the guards held; TIE kept and refined (D82; the refinement is the midpoint offset); CUT below. **CPU bound first (cheap):** replay the E284 / E287 connectors with the GT-fitted scales for the true pairs (an oracle of the curve shape): if F1@1.0 does not move by 0.005 even then, the shape is not the limit and the card stays a TIE without GPU time.
