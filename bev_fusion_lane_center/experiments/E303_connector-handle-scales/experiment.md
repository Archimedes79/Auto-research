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

## 2. Result (2026-11-12 11:27; E303 trained 09:25 - 10:57 on the maps of E286 j1, pair head p2 + the scale head; VAL 873; `runs/e303_s_s<S>/ends_card.json`, the chord control `runs/e303_chord_s<S>/ends_card.json` = the same dump scored with `--chord`, same pairs; the snapped read-outs `ends_card_snap_thr<t>.json`)
Unsnapped read-out (the card's own, p >= 0.5, connectors from the keypoints), differences to the plain graph of the seed:
| seed | arm | topo F1 | F1@1.0 | lanes found | dup_frac | vertices / 100 m |
|---|---|---|---|---|---|---|
| 0 | p2 (E287, control c) | +0.0061 | +0.0070 | +0.0199 | -0.0237 | +0.13 |
| 0 | E303 learned scales | +0.0077 | +0.0080 | +0.0237 | -0.0159 | +0.18 |
| 0 | chord control (same dump) | +0.0068 | +0.0072 | +0.0225 | -0.0162 | +0.12 |
| 1 | p2 (E287, control c) | +0.0037 | +0.0107 | +0.0265 | -0.0218 | +0.10 |
| 1 | E303 learned scales | +0.0085 | +0.0139 | +0.0297 | -0.0204 | +0.16 |
| 1 | chord control (same dump) | +0.0050 | +0.0117 | +0.0263 | -0.0208 | +0.09 |
**Verdict by the rule of this card (F1@1.0 +0.01 over c on both seeds): TIE** (F1@1.0 over p2 +0.0010 / +0.0032; over the chord curves of the same pairs +0.0008 / +0.0023; topo F1 +0.0009 / +0.0035 over the chord control; guards held; label "by hand: pair scorer"). Kept and refined (D82). The pair decision is unchanged by the scale head (precision at the recall of p2 +0.003).
**The same pairs with the snapped connector ends of E311** (second reading, not part of the rule above; both seeds, the chord control as the reference of the same pairs):
| seed | p >= | learned scales: topo F1 / F1@1.0 / lanes found / junction F1 | chord control: topo F1 / F1@1.0 / lanes found / junction F1 | scales over chord (topo F1 / F1@1.0) |
|---|---|---|---|---|
| 0 | 0.1 | +0.0320 / +0.0240 / +0.0751 / +0.0322 | +0.0265 / +0.0204 / +0.0705 / +0.0321 | +0.0055 / +0.0036 |
| 1 | 0.1 | +0.0356 / +0.0363 / +0.0910 / +0.0602 | +0.0251 / +0.0298 / +0.0777 / +0.0599 | +0.0105 / +0.0065 |
| 0 | 0.5 | +0.0164 / +0.0094 / +0.0260 / +0.0089 | pending | |
| 1 | 0.5 | +0.0185 / +0.0150 / +0.0299 / +0.0205 | +0.0142 / +0.0125 / +0.0259 / +0.0205 | +0.0043 / +0.0025 |
Reading: on the connectors drawn from the keypoints the scales are invisible (the curve ends beside the pieces whatever its shape); on the snapped read-out they are worth +0.006 / +0.011 topo F1 and +0.004 / +0.007 F1@1.0 at p >= 0.1, and the E303 pairs reach the best numbers of the thread so far (+0.0320 / +0.0356 over the plain graph, E305: +0.0304 / +0.0285). The scale head costs training time (the fit of the GT scales, 1.9 s / step instead of 1.2). **Used in E312** (the pair head there has `--scales`); the rule of E311 applies to the read-out (snapped, p >= 0.1) and the scales stay the E303 default of the read-out when the dump carries them.
