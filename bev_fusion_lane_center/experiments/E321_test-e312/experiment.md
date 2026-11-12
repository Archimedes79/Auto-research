# E321 - the one TEST use of the final system E312 (D84)

Card of 2026-11-12 23:30, written by the lane chat (fork C) before any TEST run. Mirko 2026-11-12 23:00: "Teste E312." (`experiment_execution/decisions/D84.md`). The design is finished and fixed on VAL: E307 maps (end heat sigma 1.0 m) + the pair head `--e2e --scales --maps-w 1.0` (E305, E303) + connectors at p >= 0.1 whose ends are snapped to the delivered pieces (E311), cards E312 (VAL, two seeds: snapped topo F1 +0.0566 / +0.0615 over the plain graph, not snapped +0.0179 / +0.0233). This card is a REPORT: nothing is selected, tuned or re-read on TEST, no champion decision (D4 stays Mirko's), the numbers are labelled TEST.

## 0. The card (what the tools run)
```json card
{
 "id": "E321",
 "card": "",
 "title": "E312 (maps e307_n + pair head e312_p, connectors at p >= 0.1, snapped and not snapped) against the plain delivered graph of B (E273 + close:1.5@0.5) on the TEST split, both seeds, report only",
 "kind": "block",
 "deterministic": false,
 "metric": "topo_F1",
 "direction": "higher",
 "band": 0.01,
 "prediction": {"value": 0.045, "vs": "c"},
 "arms": {
  "c": {"recipe": "control = the plain delivered graph of B of the same seed (runs/e273_p_s<S>, exact map, hard close-node step close:1.5@0.5, read-out e >= 0.5 / link p >= 0.5), no connectors"},
  "w": {"recipe": "winner = E312 (runs/e312_p_s<S>: maps runs/e307_n_s<S> + pair head trained 3000 steps with --e2e --scales --maps-w 1.0), connectors at p >= 0.1 whose ends are snapped to the delivered pieces (E311, the card of record of E312)"},
  "d": {"recipe": "the same checkpoints, connectors at p >= 0.1 drawn from the keypoints, NOT snapped: the differentiable read-out (the same TEST use, labelled; fixed on VAL as ends_card_thr0.1.json)"}
 },
 "control": "c",
 "variant": "w",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "topo_F1", "vs": "c", "min_effect": 0.0}
 ],
 "budget_gpu_h": 0.3,
 "kill": "the VAL check of the pipeline (jtest_eval.py verify) does not reproduce the stored cards of E312, or a dump does not hold all 833 frames in the order of data/splits.json['test']: no TEST number is used",
 "evaluate": {"split": "test"}
}
```
By hand (label "by hand: report only"): no verdict is derived from TEST. The report lists, per seed and read-out (snapped / not snapped), topo F1, lanes found, F1@1.0, junction F1, dup_frac, vertices per 100 m, the maneuver coverage (`jmaneuver` definition: 90 % of a GT maneuver within 1.5 m and 40 degrees), and winner minus control with the paired scene bootstrap (21 scenes, 1000 draws, `autoresearch/Evaluation/stats.py`). The D79 guards (F1@1.0 -0.01, dup_frac +0.02, vertices +1.0, junction F1 -0.03) are printed as on VAL. Whatever it shows, E312 is not re-tuned, no threshold changes, no read-out is added; a result below the VAL gain is a finding about the design, not a reason for a second look.

## 1. Design
- **Data.** TEST = `data/splits.json['test']`, 833 frames, 21 scenes (BuilderData `test_clean`); the old graph GT of the cache (`graph_gt_val.pkl[row]`, the file holds the VAL and the TEST rows), the lane GT l6 store (all 34,149 keyframes). The same loaders, the same exact-map pass for B, the same `close:1.5@0.5` step and the same scorer as the VAL cards of E312; the only thing that changes is the split (`evaluation/jtest_eval.py`, which REFUSES `--split test` outside the launcher and without a TEST-log row for the id).
- **What runs** (one launcher run, both seeds, id `E321.t`): per seed B's full prediction dump on the split, the pair head's dump (maps head + pair head of `runs/e312_p_s<S>`), the delivered graphs plain / not snapped / snapped, the numbers, the maneuver coverage, the paired bootstrap; results `runs/eval/e312_test/card_s<S>.json` and `results.md`; dumps in `temp/e321/test/` (scratch).
- **Check before the run (layer 1).** `jtest_eval.py verify --seed <S>` reproduces on VAL the stored cards `runs/e312_p_s<S>/ends_card_snap_thr0.1.json`, `ends_card_thr0.1.json` and `maneuver_val.json`; the result is in section 2.
- **Device.** The stored VAL dumps were made on the GPU (bf16); on the CPU the top-K selections of nodes / keypoints differ at the margin (section 2), so the TEST run uses the GPU like the VAL dumps. The control and the winner always share the same B dump, so a device effect cancels in winner minus control.
- **Prediction (a number):** snapped topo F1 gain over the plain graph +0.045 on TEST (VAL +0.057 / +0.062; a design that was fixed on VAL usually gives back part of it); not snapped +0.015.

## 2. Check on VAL (filled before the run)
`evaluation/jtest_eval.py verify --seed <S>` (VAL, the stored dumps `temp/e276/predfull_e273_p_s<S>.pkl` and `runs/e312_p_s<S>/ends_val.pkl`, CPU, 12 minutes per seed) reproduces, for BOTH seeds, EXACTLY (max absolute difference 0.00e+00 over all 11 metrics): the snapped card `ends_card_snap_thr0.1.json` (numbers, differences, curves 11378 / 11849, wired both ends 5992 / 6407), the not snapped card `ends_card_thr0.1.json`, and the three variants of `maneuver_val.json` (plain 5773 / 5394, p0.1 7833 / 7760, p0.1s 7770 / 7703 of 10991 maneuvers). The paired scene bootstrap of the module on VAL (22 scenes, 100 draws): seed 0 snapped - plain topo F1 +0.0566 [+0.0446, +0.0690], not snapped +0.0179 [+0.0103, +0.0284]; seed 1 snapped +0.0615 [+0.0456, +0.0781], not snapped +0.0233 [+0.0087, +0.0380].
Fresh dumps (CPU against the stored GPU dumps): pending, see the result section.

## 3. Launch (after the log row and the grant of D84; see the report of fork C)
`E321.t`, from the project folder (GPU: needs a free slot, about 10 minutes of GPU and about 40 minutes of CPU for the scoring of the two seeds):

```
python3 ../autoresearch/tools/run.py --id E321.t --gpu --retries 0 --require-card --split test \
  --hyp "E312 keeps its VAL gain over the plain graph on TEST (one logged use, D84)" \
  --pred "snapped topo F1 +0.045 over the plain graph on TEST, not snapped +0.015" \
  --rule "E321 card: report only, winner minus control with the paired scene bootstrap, nothing selected or tuned on TEST" \
  -- bash tools/env/_pyr.sh evaluation/jtest_eval.py all --split test --test-id E321.t --seeds 0 1 --device cuda
```
CPU fallback (same command without `--gpu` and with `--device cpu`): the numbers then differ from a GPU dump at the margin (section 2), say so in the report.
