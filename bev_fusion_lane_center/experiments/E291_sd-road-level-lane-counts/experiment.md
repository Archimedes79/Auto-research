# E291 A road-level map with lane counts as the only map input of the junction pair head (D82 bonus question: "nehmen wir an, wir haben nur einen Road Level Graph ... Anzahl der Spuren pro Road in eine Richtung")

Card of 2026-11-11 (a plan card, behind E287), written by the lane chat on Mirko's question and on the SD-map scan of the day (`literature/scan_2026-11-11_sd_map_prior.md`). Mirko's correction of 2026-11-11 stands: the lane-level map of E273 (`map_sim.py`: pieces with succ / pred / left / right ids and a connector kind) carries exactly the connections that a real SD map does not have, so it must not feed the pair head. The scan: no published BEV lane method puts the lane count to use as a constraint (a survey lists it as future work); SMERF and P-MapNet do not simulate an SD map from the HD map (they download OSM); `sd_lib.py::build_sd` of this project (E77) is the one road-plus-count simulator the scan found, and it has no visible connector filter: check before use.

## 0. The card (what the tools run)
```json card
{
 "id": "E291",
 "card": "",
 "title": "pair head v2 with a road-level map input (arms of the junction with the number of lanes per direction, no lane connections, noisy) against the same head without it",
 "kind": "block",
 "deterministic": false,
 "metric": "pair_recall",
 "direction": "higher",
 "band": 0.03,
 "prediction": {"value": 0.08, "vs": "c"},
 "arms": {
  "c": {"recipe": "control = the pair head of E287 that goes on (p1 or p2), no map"},
  "s": {"recipe": "+ the road-level arm table: per arm the axis point, heading, lanes in, lanes out (from the lane GT arms of `jgt.py`, so the information content is what an SD map has: the roads meeting at the junction and the lane count per direction; no connectors, no lane ids), degraded as a real map would be: shift 0 - 2 m (scene-constant), yaw 0 - 5 degrees, count +-1 with p 0.1, unknown count with p 0.1, a minor arm dropped with p 0.1; used as (a) a count prior: a soft capacity of entries / exits per arm and a count loss, (b) the arm graph as a block prior on the pair logits (a pair only between arms that are connected on the road graph)"}
 },
 "control": "c",
 "variant": "s",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "pair_recall", "vs": "c", "min_effect": 0.03},
  {"metric": "pair_precision", "vs": "c", "max_loss": 0.02},
  {"metric": "topo_F1", "vs": "c", "max_loss": 0.01}
 ],
 "budget_gpu_h": 1.5,
 "kill": "the arm table of the SD input contains a connector or lane id (a leak check on the dict keys and on a shuffled-pairs test: with the true pairs shuffled the head must not gain); or the noisy map is worse than no map by more than 0.03 on both seeds",
 "evaluate": {"split": "val"}
}
```
By hand (the keypoint / pair counts come from `jends_eval.py dump`; the delivered graph from `score`; label "by hand: pair scorer"): KEEP when pair recall rises by more than 0.03 on both seeds with the guards held; TIE inside: kept and refined (D82); CUT when the pair precision or topo F1 breaks a guard on both seeds. Reported: the same with an exact map (an upper bound for the input, not a result), with a 2 m shift, with the arm table of a junction that has a dropped arm, the share of arms whose count is right in the entries found by the keypoints (the count loss needs found ends).

## 1. The SD input (what is in it, what is not)
Built on the lane side from the lane GT, by location (`jgt.junction_gt`): the arms with centre, heading, n_in, n_out (the lane count per direction) and the zone centre. Not in it: connectors, lane ids, successor / predecessor / left / right relations, lane index, dividers, crossings, stop lines. The degraded map is fixed per frame by the token (as `map_sim` does), so that every evaluation sees the same map. The existing `sd_lib.build_sd` (road raster with counts) is checked for connector leaks first (a CPU test: the true pairs shuffled -> no gain); if it leaks, the arm table of `jgt.py` is the input.

## 2. Design
Two seeds on the maps of E286 and the pair head of E287; the new input enters `jends.JEnds` as (a) per-keypoint features (distance to the nearest arm axis, the arm's count, the keypoint's rank over the count) and (b) the capacity penalty; the count loss and the block prior are soft (differentiable). The literature caveat (SMART: SD only gives TOP_ll 12.4, SD plus satellite 31.9): the pairs have to come from the sensors, the map only constrains them.
**Prediction (a number):** pair recall +0.08 over c on the noisy map (the entries missing in an arm are found by the count; the block prior removes impossible pairs: precision should hold or rise).

## 3. Command
After E287: `temp/e291/start_sd.sh <seed>` through `run.py --require-card`; the dump / score as E287. Result `runs/e291_s<S>/ends_head.json`, `ends_card.json`.
