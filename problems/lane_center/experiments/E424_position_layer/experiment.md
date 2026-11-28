# E424 what the soft position layer does against the zig-zag

The card is the json block below: `research check <problem> E424` checks it, `research start` freezes it in the snapshot. What the runs do,
how the verdict is made and what counts is the card as the snapshot holds it; a changed experiment is a new experiment.

```json card
{
 "id": "E424",
 "title": "what the soft position layer does against the zig-zag",
 "question": "on VAL, behind the three wiring layers: does the soft position layer lower the heading error of the delivered lines and raise F1 at 0.5 m without a loss of lanes found?",
 "hypothesis": "a node lies at its cell centre plus an offset that is shrunk toward the centre, and neighbouring nodes miss the lane to different sides; scaling the offset by 1.2 and half a step of every short link's ends onto a common arc lowers the heading error by about 1.5 degrees, raises F1 at 0.5 m by about 0.012 and topo F1 by about 0.004, at unchanged lanes found",
 "prediction": 0.004,
 "kind": "compare",
 "control": "fork",
 "seeds": [0, 1],
 "arms": {
  "fork": {"from": "E423.fork", "note": "E423's runs with the three wiring layers"},
  "layers": {"from": "E422.layers", "note": "E422's runs with all four layers"}
 },
 "rule": {},
 "kill": [],
 "note": "companion of E422 and E423, no run of its own; written before their results were read; E422 and E423 are started on the same commit; no adoption from this card"
}
```

## Why
Scratch diagnostic of the champion on VAL (`zigzag.py`, written into E398's follow-up, no record): 20 % of the delivered nodes lie in a cell
the true lane does not cross (median 1.19 m off); the predicted offset against the required one has a slope of 0.55; 16.6 % of the links step
more than 0.5 m sideways against their neighbours. Earlier, E395 (tag `v1`) trained a position head on the tokens: F1@0.5 +0.017 to +0.024 on
VAL, below its bar of 0.033, topo F1 -0.006 against its twin, not adopted; the same line of work measured that position errors of neighbouring
nodes are correlated (0.52 at 2 m), which neighbours cannot remove. New here: no training and no features, a calibration of the offset and one
geometric step along the delivered links, after the wiring.

TRAIN (291 frames, seeds 0 / 1; settings applied to the network's own outputs with the three wiring layers):
| variant | heading error (deg) | F1@0.5 | topo F1 | lanes found | vertices / 100 m |
|---|---|---|---|---|---|
| without (fork) | 11.39 / 11.32 | 0.737 / 0.733 | 0.7563 / 0.7561 | 0.803 / 0.795 | 22.4 / 22.4 |
| gain 1.2 only | 10.78 / 10.79 | 0.749 / 0.745 | 0.7594 / 0.7584 | 0.804 / 0.798 | 21.7 / 21.8 |
| gain 1.0, arc step 0.5 | 10.14 / 10.09 | 0.744 / 0.740 | 0.7594 / 0.7589 | 0.801 / 0.793 | 21.0 / 21.0 |
| gain 1.2, arc step 0.5 (taken) | 9.75 / 9.73 | 0.753 / 0.749 | 0.7608 / 0.7601 | 0.802 / 0.795 | 20.2 / 20.3 |
| gain 1.3, arc step 0.5 | 9.67 / 9.68 | 0.753 / 0.750 | 0.7609 / 0.7595 | 0.801 / 0.795 | 20.0 / 20.1 |
| gain 1.4, arc step 0.5 | 9.66 / 9.69 | 0.752 / 0.748 | 0.7612 / 0.7595 | 0.801 / 0.793 | 19.9 / 20.1 |
| gain 1.2, arc step 1.0 | 9.09 / 9.05 | 0.753 / 0.749 | 0.7604 / 0.7591 | 0.796 / 0.790 | 19.2 / 19.2 |
| oracle: every node on its nearest true lane | 8.14 / 8.15 | 0.851 / 0.851 | 0.7799 / 0.7784 | 0.839 / 0.833 | 18.5 / 18.4 |

Selection: the six settings of this table and, in an earlier probe without the fork gate (`probe.py`, another form of the heading measure),
offset gains 1.2 / 1.3, arc steps 0.5 / 1.0 repeated 1, 3 or 8 times and a step toward the neighbours' mean were tried on one TRAIN sample.
Near the taken setting the differences are below 0.001 topo F1; more steps lose lanes. Even with every node on its lane the heading error stays
at 8.1 degrees: most of it comes from which nodes are linked and how, not from their positions.

## Prediction and bar
VAL, `layers` against `fork`: topo F1 +0.004 (inside the band: the standard verdict TIE is expected), heading error 1.5 degrees lower, F1@0.5
+0.012, lanes found within 0.003. The layer works when, on each seed, the heading error falls by at least 1.0 degree and F1@0.5 gains at
least 0.006.

## How the verdict is read
First the check that the two arms differ in positions only: the delivered nodes and links of `E423.fork` and `E422.layers` equal on every VAL
frame (compared by script, reported), and both experiments started on one commit. Then the standard rule on topo F1 with the problem's guards,
then the bar, per seed.

## Design delta
R1: one fixed soft layer more (`jtoken.refine_positions`); the delivered node positions are now an output of the head (before: B''s own).
R7: the link and keep thresholds inside the layer.

## Critic (cards mode, 2026-11-28): revise
Taken in: E395 and its source named, the oracle from this table's measure, all settings tried disclosed, the scratch source of the VAL numbers
named, a bar per seed, the check that only positions differ.

## Conclusion
**Verdict** (`research judge`, 2026-11-28, `verdict.json`): `layers` **TIE** against `fork`: topo F1 +0.0027 [-0.0016, +0.0054], no guard
broken. Checks: the delivered nodes and links of `E423.fork` and `E422.layers` are equal on all 873 VAL frames, both seeds (script), so only
positions differ; both experiments ran the code of one commit.

**The quality bar is met on both seeds** (heading error down by >= 1.0 degree, F1@0.5 up by >= 0.006):
| seed | heading error (deg) | F1@0.5 | lanes found | vertices / 100 m | time per frame (ms) |
|---|---|---|---|---|---|
| 0 | 11.90 -> 10.59 (-1.31) | 0.6384 -> 0.6486 (+0.010) | 0.7114 -> 0.7095 | 20.96 -> 19.14 | 21.3 -> 28.5 |
| 1 | 11.92 -> 10.59 (-1.33) | 0.6429 -> 0.6541 (+0.011) | 0.7136 -> 0.7101 | 21.27 -> 19.50 | 22.1 -> 27.7 |

- The zig-zag is smaller, not gone: with every node on its lane the heading error on TRAIN is still 8.1 degrees.
- The cost as built is too high: about 6 ms per frame (both seeds, quiet GPU).
