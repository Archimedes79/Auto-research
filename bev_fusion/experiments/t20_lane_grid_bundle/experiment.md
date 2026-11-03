# t20 lane / grid bundle - L4 lane branch, a 1 m centre-line head in the grid head, the correctness fixes, one baseline (card 2026-11-03, before any run; Mirko's order D48)

Context: `../../plan/plan_2026-11-03.md` (the order, the answers, the sequence), `../../records/lane_diagnostic_2026-11-03.md` (why L4), `../../plan/head_design.md` (the heads), `../../records/open_findings_2026-11-02.md` (the fixes), the base `../t16_full_network/experiment.md` (`t16.full_s0`, the delivered system).

## 0. The card (what the tools run)

```json card
{
 "id": "t20",
 "card": "D48",
 "title": "t20 lane / grid bundle: L4 on the dual camera features, centre-line head at 1 m in the grid head, correctness fixes",
 "kind": "block",
 "deterministic": false,
 "metric": "seg_fov/seg/dt+0.00/lane_divider/all/all/iou@1",
 "direction": "higher",
 "band": 0.010,
 "prediction": {"value": 0.60, "vs": "c"},
 "arms": {
  "a": {"recipe": "t20.base_s0: t16.full + lane_branch res4 (camera_features dual) + centreline_1m + the hygiene fixes of the open findings, split large, GT v8, 6000 steps, seed 0"},
  "c": {"recipe": "t16.full_s0 (stored: runs/t16_full_s0, tag bf/t16.full_s0), same split, GT, steps, seed"}
 },
 "control": "c",
 "rule": [{"metric": "seg_fov/seg/dt+0.00/lane_divider/all/all/iou@1", "min_effect": 0.03, "note": "by hand: D48, one seed per arm; see section 1"}],
 "budget_gpu_h": 12,
 "kill": "a non-finite loss twice after the fixes, or a module whose gradient is 0 after the short training and two fix rounds, or the short training does not finish in 2 x its estimate",
 "evaluate": {"split": "val"}
}
```

## 1. Card

| | |
|---|---|
| lens | system review (Mirko's order D48) |
| group | lanes / grid head; hygiene |
| sources | `lane_diagnostic_2026-11-03.md`, `t7_lane_branch`, `open_findings_2026-11-02.md`, `head_design.md` |
| status | proposed |

- **Question.** Does a stride-4 image branch in the lift (L4), together with a 1 m centre-line head in the grid-based map head and the correctness fixes, give a better lane layer on the large split, without losing the other outputs, and does the corrected system train to a baseline that replaces `t16.full_s0`?
- **Hypothesis.** The lane divider is weak because the camera evidence that is splatted into the 0.5 m grid is coarse (image stride 16 / 8: 0.57 m at 20 m, 1.7 m at 60 m; the diagnostic shows the near field is poor too and the grid is not the cap). L4 (stride 4, 0.14 m at 20 m) gave +0.065 / +0.042 divider IoU@0.5 in t7 on a weaker model. The 1 m centre-line head adds the centre-lines as an output (none exists in `t16.full`), trained from the existing 0.5 m GT layers pooled to 1 m. The fixes remove known silent errors (strict dt, `vel_known`, `w_obj` as weight, a dead radar branch, float32 splat positions, resume details, the radar cache signature).
- **Prediction** (VAL, 1000 frames, 82 scenes, GT v8, front field of view, dt 0, against `t16.full_s0`: 0.549 at tolerance 1 m, 0.429 at 0.5 m): lane divider IoU@1 0.60 (+0.05), IoU@0.5 0.47 (+0.04); the centre-line head at tolerance 1 m: IoU 0.35 or more (a floor: no reference in this system, the first number of this head); ms per frame without the cached image side <= 105 (t16: 96.4; L4 was predicted +3 to +6 ms in t7); maps, front mAP2 and velocity within their bands of `t16.full_s0`.
- **Decision rule** (written before the run; one seed per arm, so every reading is labelled "by hand: D48, one seed"): (1) the system is accepted as the new reference baseline when the gates are green, the training ends without a stop, and no guard is worse than `t16.full_s0` by more than its band (maps mean IoU 0.010, front mAP2 0.020, vehicle velocity EPE 0.10 m/s); (2) the lane lever is reported as confirmed when the divider IoU@1 is above the control by more than 0.03 (3 x the band), as not confirmed when inside +-0.01, as negative when below -0.01; it is never read as proven by one seed; (3) the centre-line head is reported with its IoU@1 and its failure modes (crossings: one averaged direction); (4) a guard lost by more than its band is named in the report and the baseline label stays with `t16.full_s0` for that output.
- **Test design.** Arms: `t20.base_s0` against the stored `t16.full_s0`. No ablation per item (L4 / head / fixes are bundled by Mirko's order): the attribution of a change to one item is NOT possible, the report says so. The short bug-finding training (small split, 400-1000 steps, GPU slot 1) comes first and is not an experiment: it feeds no verdict.
- **Cost.** About 9 GPU hours for the baseline plus 3 for the short training; coding about 4 h; risk: L4 with `camera_features` 'dual' is refused by `model.py:228` today (to be extended and tested); the 1 m head reads the trunk at 1 m.
- **Kill criterion.** see the card block.
- **Inputs.** To build: L4 for 'dual' (+ test), `layers.py` + `centreline_1m` (unit 4 of the redesign, `tests/test_layers.py`), the fixes #4-#8, the config `t20_base.json` (`"split": "large"`). Ready: GT v8 with `centreline` and `centreline_dir`, the large split, the trunk with the detail path.
- **Starts from** `t16.full_s0` (the recipe of `training/configs/t16_full.json`).
- **Referee** (one line, to be filled by the critic pass of step 3).

## 2. Detail plan

| step | what exactly | where | done when | budget |
|---|---|---|---|---|
| code | L4 for `camera_features` 'dual'; `centreline_1m` option (done by the layer-table unit); fixes #4-#8 | `training/model.py`, `lane_branch.py`, `layers.py`, `losses.py` | gates green; gradient reaches L4 and the 1 m branch; option off = old state_dict keys | 4 h |
| config | `training/configs/t20_base.json` from `t16_full.json` + the three changes, `"split": "large"` | trainsuite compose | `startable.py` check passes | 0.5 h |
| short training | small split, 400-1000 steps, probe + per-module gradient report + timing | `trainsuite start ... --preset small`, GPU slot 1 | every module has gradient > 0, no non-finite loss, resume works | 3 h |
| train | 6000 steps, split large, through `run.py`, first in the queue | `trainsuite start` / `run.py` | done flag, exit 0 | 8-9 h |
| evaluate | the door: `evaluate.py runs/t20_base_s0 --all`; the same for the control from its stored eval | `evaluation/evaluate.py` | metric card, tables, videos, timing on an idle GPU | 1.5 h |
| decide | the rule above, by hand | ledger note | rows written | 0.5 h |

## 3. Summary

(after the run)
