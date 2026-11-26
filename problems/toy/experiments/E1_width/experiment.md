# E1 does a wider network segment the disc better

The first experiment of the toy problem (P0.4): a real question with a real answer, small enough to run in a minute on the CPU, through the same tools as a large one.

## 0. The card (what the tools run)
```json card
{
 "id": "E1",
 "title": "does a wider network segment the disc better",
 "kind": "variant",
 "metric": "iou",
 "direction": "higher",
 "band": 0.010,
 "prediction": {"value": 0.020, "vs": "c"},
 "arms": {
  "c": {"recipe": "width=4", "names": ["E1_c_s0", "E1_c_s1"], "run_ids": ["E1.c_s0", "E1.c_s1"], "seeds": [0, 1]},
  "a": {"recipe": "width=12", "names": ["E1_a_s0", "E1_a_s1"], "run_ids": ["E1.a_s0", "E1.a_s1"], "seeds": [0, 1]}
 },
 "control": "c",
 "rule": [{"metric": "iou", "min_effect": 0.010}, {"metric": "recall", "max_loss": 0.020}],
 "budget_gpu_h": 0.0,
 "kill": "total@200 > 0.25",
 "evaluate": {"split": "val", "compare": "c"}
}
```

## 1. Card text (written before the runs)
- **Statement:** three times the channels raise the pixel IoU on the 160 val frames by at least 0.010.
- **Mechanism:** a pixel is decided from a 7 x 7 neighbourhood; more channels can hold several smoothing filters at once.
- **Inputs:** ready (raw shards, truth, trainer, evaluator).
- **Starts from:** `training/configs/base.json`.
- **Kill:** a training loss above 0.25 at step 200 is no better than calling every pixel background.

## 3. Summary
- **Runs** E1.c_s0, E1.c_s1, E1.a_s0, E1.a_s1 (tags `toy/E1.*`). **Decision: KEEP** by the rule as written (`decision.json`).
- **Numbers** (val, 40 scenes, 160 frames): IoU with width 4: 0.0045 and 0.5348; with width 12: 0.8639 and 0.8628. Effect +0.5937; paired scene intervals per seed [+0.8501, +0.8671] and [+0.3100, +0.3489]. The prediction (+0.020) missed by far.
- **What it says:** not that three times the channels hold more. The control did not finish training in 400 steps: seed 0 predicts almost nothing, seed 1 about half. Why was not examined.
- **Weaknesses:** the kill rule watched the training loss, which falls while the prediction stays empty (`Training status` reads 0.17 and 0.19 at step 200 for the two control runs, under the bar of 0.25), and no watcher enforced it; two seeds; the number of steps decides the answer.
- **Next:** a kill rule on the val IoU, and the same question with enough steps for the control, before any statement about width.
