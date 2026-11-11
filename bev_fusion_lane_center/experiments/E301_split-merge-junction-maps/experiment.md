# E301 Splits and merges outside junctions on the 1 m maps (brief: "Wo Spuren ohne Kreuzung splitten oder mergen") (D82)

Card of 2026-11-11 (plan card). E286 has entry / exit heat, zone, core and paint; the splits and merges of the lane GT (events `split`, `merge`, with the directions of the branches) are only in B's event heat maps (centre1 channels 6 - 11, F1 0.37 - 0.57) and in the node flags. The brief wants them on the 1 m grid with direction like the entries and exits.

## 0. The card (what the tools run)
```json card
{
 "id": "E301",
 "card": "",
 "title": "split and merge heat with heading on the junction maps head (two more 1 m channels pairs: split point and merge point, heading = the travel direction into the branching), against B's own split / merge event heat read the same way",
 "kind": "block",
 "deterministic": false,
 "metric": "kp_f1_sm",
 "direction": "higher",
 "band": 0.05,
 "prediction": {"value": 0.12, "vs": "c"},
 "arms": {
  "c": {"recipe": "control = the split / merge event heat maps of B's pyramid (centre1 channels 6 / 7), 3 x 3 peaks above 0.3, distance only, scored against the `split` / `merge` events of the lane GT (the events of junctions are excluded: they are the entries / exits; the events outside the zones are the target)"},
  "s": {"recipe": "the E286 head with 6 more channels (split heat, merge heat, two headings) trained on the events of the lane GT outside the junction zones (`qjunction.junction_targets` gets `jsm`; `JMaps(NCH1 = 14)`), best context arm of E286, 2 seeds"}
 },
 "control": "c",
 "variant": "s",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "kp_f1_sm", "vs": "c", "min_effect": 0.05},
  {"metric": "kp_f1", "vs": "c", "max_loss": 0.02}
 ],
 "budget_gpu_h": 1.5,
 "kill": "the split / merge targets of the layer disagree with the events of the lane GT (count check on 100 frames)",
 "evaluate": {"split": "val"}
}
```
By hand (the scorer is `jmaps_eval.py` extended by the split / merge keypoints; label "by hand: keypoint scorer"): `kp_f1_sm` = the mean of the split F1 and the merge F1 at 3 m, 30 degrees; `kp_f1` is the E286 entry / exit number (the guard: the new channels must not hurt the old ones). TIE kept and refined (D82). Code after the night chains (it changes `jmaps.py` / `qjunction.py`, which the running E286 / E287 runs import).

## 1. Closed by the data before any code (2026-11-11, CPU count on the VAL lane GT, every second keyframe, `temp/gt2/ev_peek.py`, scratch)
**The target of this card hardly exists.** Of 2506 split and merge events of the lane GT 2501 (99.8 %) lie inside a junction zone (the connector zone grown by 2 m) and 5 outside; one is partial. In the nuScenes map lane graph splits and merges are made at junctions (a lane that feeds two connectors, two connectors that join); a lane that appears or ends mid-block (a turn-lane pocket, a lane drop) is not modelled as a split. So the brief's "Wo Spuren ohne Kreuzung splitten oder mergen" has about five examples in the whole VAL split: a head cannot be trained or scored on it. Inside the junction the split point is the stop / entry marker (the GT says a stop is also a split) and the merge point the exit marker, and the conflict core (E285) is where connectors diverge and join: **E286 / E287 already carry them** (entry / exit heat with heading, zone, core); the node head of B has split / merge flags for the nodes. **Decision: this card is not run (CUT by evidence, "by hand: count");** the plan table of `experiment_execution/plans/junction_thread_2026-11-11.md` is updated. If Mirko wants mid-block lane drops and adds, the GT has to come from another source (OSM lane counts, a map with lane-level changes): nuScenes does not carry them (a GT-side request for l7, not decided).
