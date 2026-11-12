# E304 The far range: a heat loss that counts the cells beyond 35 m more (D82: "what might be overlooked"; E300, E286 section 6)

Card of 2026-11-11. E300 (E284 head) and E286 (j1) agree: the ends beyond 35 m from the ego are the largest hole: 53 % of the GT entries lie there, and their recall is 0.55 - 0.58 (E286 j1 seed 0) against 0.71 - 0.72 within 20 m; occlusion and paint matter 2 - 4 points, the range 15 points. The frozen GaussianCaR features are coarser and noisier at range; the loss treats every cell alike, and the heat BCE is dominated by the many near cells.

## 0. The card (what the tools run)
```json card
{
 "id": "E304",
 "card": "",
 "title": "junction maps head j1 with the heat BCE of the cells beyond 35 m counted 1 + 2 times (`jmaps_train.py --far-w 2`), against the same head with equal weights",
 "kind": "variant",
 "deterministic": false,
 "metric": "recall_far",
 "direction": "higher",
 "band": 0.03,
 "prediction": {"value": 0.04, "vs": "c"},
 "arms": {
  "c": {"recipe": "control = E286 j1 (the best context arm of the night, same seeds, same maps head, 3000 steps)"},
  "f": {"recipe": "`jmaps_train.py --ctx dil --far-w 2` (the same recipe, the heat BCE of the cells farther than 35 m from the ego with the weight 3)"}
 },
 "control": "c",
 "variant": "f",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "recall_far", "vs": "c", "min_effect": 0.03},
  {"metric": "kp_f1", "vs": "c", "max_loss": 0.01},
  {"metric": "recall_near", "vs": "c", "max_loss": 0.02}
 ],
 "budget_gpu_h": 1.5,
 "kill": "the weighted loss does not change the far recall at all after 1000 steps (the weight is not reaching the cells): then the card stops and the lever is closed",
 "evaluate": {"split": "val"}
}
```
By hand (label "by hand: keypoint scorer"; `recall_far` = the mean of entry and exit recall beyond 35 m at thr 0.3, 3 m; `recall_near` within 20 m; both from the range split of `jmaps_eval.py`): KEEP when the far recall rises by more than 0.03 on both seeds with `kp_f1` not lower by more than 0.01 and the near recall not lower by more than 0.02; TIE kept and refined (D82; the refinement is a larger weight or a far-range context level); CUT below. Reported: precision beyond 35 m (a higher weight may buy recall with false ends), city split.

## 2. Stopped (2026-11-12 11:57, by hand)
Started 11:49 (both lanes) behind E307 in day chain 4; stopped at step 200 of 3000 (SIGTERM, exit 143) when E307 (heat sigma 1.0 m, no far-range weight) turned out to lift the entry recall beyond 35 m by +0.10 / +0.09 on its own (E307 card section 2): the single-factor test of the far-range weight on the old sigma 1.5 m is no longer the question; the GPU slots went to E312 (pair head on the E307 maps). The arm combined with sigma 1.0 (`--sigma 1.0 --far-w 2`) is the planned follow-up (E313). No verdict.
