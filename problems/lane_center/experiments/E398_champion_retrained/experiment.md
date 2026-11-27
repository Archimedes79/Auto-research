# E398 the champion trained again through the research system

The card is the json block below: `research check <problem> E398` checks it, `research start` freezes it in the snapshot. What the runs do,
how the verdict is made and what counts is the card as the snapshot holds it; a changed experiment is a new experiment.

```json card
{
 "id": "E398",
 "title": "the champion trained again through the research system",
 "question": "does the champion's two-stage recipe, run again by the research system from the kept lane head B' and its token cache, give a network as good as the registered champion on VAL, and does it train the way the original runs did?",
 "hypothesis": "code, recipe, data and seeds are those of the champion's runs (E373 cb stage 1, E372 S4 stage 2) and CUDA training is not bit-exact, so the retrained network differs from the champion only by training noise: a topo F1 difference inside the band of 0.010, with the logs of the original runs matched where they are exact",
 "prediction": 0.0,
 "kind": "compare",
 "control": "champ",
 "seeds": [0, 1],
 "steps": ["stage1", "stage2", "predict", "evaluate"],
 "arms": {
  "champ": {"from": "E396.champ", "note": "the registered champion: the stored runs of E396 (the weights of E372 S4)"},
  "retrain": {"set": {}, "note": "the base config unchanged: stage 1 (12000 steps) and stage 2 (3000 steps) trained now, no weights imported"}
 },
 "rule": {},
 "kill": [],
 "note": "a reproduction of the training, asked for by the owner on 2026-11-27 to test the system; no adoption follows from it; the card was revised after a critic pass before the start"
}
```

## How the verdict is read
- **As good as the champion** = TIE with no broken guard AND each seed's topo F1 difference below 0.010 in size. A TIE with a broken guard (the
  judge also calls a better score with a broken guard TIE, and a guard never changes a result inside the band), or with one seed outside the
  band (the pooled gain can hide opposite seeds), is reported as not as good.
- CUT: worse beyond the band. KEEP: better than the champion, which needs an explanation before it counts.
- INCONCLUSIVE: stays open; a third seed is not possible (B' and its token cache exist for seeds 0 and 1 only). INCOMPLETE: a run failed or
  was stopped (a trainer's own kill rule ends its run as failed): no statement about the recipe.
- A TIE alone does not show that stage 2 did its work: E372 read S4 minus its stage-1 head at +0.0002 / +0.0051 topo F1 (card in tag `v1`).
  The training itself is therefore checked on the logs.

## Reproduction checks (diagnostics, against `<runs root>/lane_center/e373_cb_s<S>` and `e372_s4_s<S>`)
- Stage 1: the first loss within 0.001 of 4.5516 / 4.3306 (seed 0 / 1); `gt_links`, `pos_pairs`, `gt_outside` equal to the original's
  `log.jsonl` at every logged step (step 100: 2472 / 2448 / 24 and 2052 / 2027 / 25).
- Stage 2: the frames of steps 1 - 5 identical to the original's log; reported beside the originals, not expected exact: `b_start`,
  the last `b_mean`, the final loss (1.774 / 1.813).
- Both stages: no skipped step, no roll-back, no resume.

## Stop rule
A differing first loss or step-100 line of stage 1 means the run is not this recipe: `research stop`, then find the cause. The trainers' own
kill rules stay on (stage 2 has two newer ones, GT-node coverage and active cells, which can stop a run and change no number).

## Design delta
None: no rule of `DESIGN.md` changes, no code changes.

## Plan
- No code change. The base config `training/configs/base.json` is used as it is; `import_weights` stays empty, so the step `predict` reads
  the run's own `model.pt` (the head and B' as stage 2 left them).
- Tried before: E381 / E383 reproduced the start of stage 1 and E385 20 steps of stage 2 (cards in tag `v1`). New here: both full stages
  through the research worker, with the data moved to the data root.
- Before the start: the self-tests of the system, `research tests lane_center`, the anchor of `E396.champ_s0`; the GPU free; the token cache
  `e365_cache` of B' seeds 0 and 1 present; `lane_live_tok.py fixture --split train --gpu --n 40` bit for bit (no run of this system has
  read the TRAIN features on the data root yet).
- Start: `research start lane_center E398 --seeds 0`, then `--seeds 1` once seed 0 is in stage 2, so that two stage-2 trainings never
  share the GPU (one reserves 4.6 GB in torch alone, the desktop holds about 2 GB of 12; a CUDA crash was seen at 11.6 GB). About 2 h:
  stage 1 about 42 min, stage 2 about 25 min per seed (the original logs), predict and evaluate about 5 min. Watched: seconds per step (the
  originals read the features from the WSL disk, now they come from the data disk).
- Expected warnings of the judge: ground truth compared by name only (E396's records hold no signature), 22 scenes.
- Not measured by the runs: the time per frame of the network (`cost` in `result.json` is the read-out of the stored dump). It is read by
  `jtoken_eval.py timing` as a diagnostic. TEST is not touched.

## Conclusion
TIE (`verdict.json`, research judge, the standard rule of the card, band 0.010): retrained against the stored champion, VAL 873 frames in
22 scenes, seeds pooled: topo F1 0.6743 against 0.6748, gain -0.0004, paired scene bootstrap [-0.0011, +0.0003]; seed 0 0.0000, seed 1
-0.0009 [-0.0022, +0.0005]; no guard broken. By the card's reading: **as good as the champion**. Warnings: 22 scenes; ground truth
compared by name only. Prediction 0.0: hit. No adoption.

**Reproduction.** Total loss at step 100 of stage 1: 3.1413 / 3.0942 (seed 0 / 1), as in the originals; learning rate 0.001 in stage 1 and 0.0001 in stage 2 .
Seed 0 repeats the original bit for bit: first loss 4.5516, link counts equal at 120 of 120 logged steps of stage 1,
81 of 81 tensors of the stage-1 head, 81 of 81 of the stage-2 head and 192 of 192 of B' identical to `e373_cb_s0` / `e372_s4_s0`, and 46
of 46 VAL numbers equal to `E396.champ_s0`. The hypothesis was wrong in one point: training on this machine repeated exactly. The model
files differ in their hash (they record the run's name). Seed 1: stage 1 exact (first loss 4.3306, 120 of 120 steps, 81 of 81 tensors);
stage 2 equal to the original at every logged step up to 1600, then it failed at step 1684 with `CUDA error: unknown error` while a GPU
training chain of another chat, running since 22:39, shared the GPU. `research resume --failed` continued it from the checkpoint of step
1500; from there the batch order is new, so the card's check "no resume" fails for seed 1: 1 of 81 head tensors and 154 of 192 tensors of
B' identical (largest difference 0.0030), final loss 1.8095 against 1.8132, `b_mean` 1.4260 against 1.4305, 1 of 46 VAL numbers equal.

**Time** (seconds; seed 0 / seed 1): stage 1 1268 / 1964, stage 2 1455 / 607 to the crash and 476 after the resume, predict 123 / 102,
evaluate 129 / 121; seed 0 took 50 minutes from start to result, seed 1 (started 25 minutes later) 59. Original training loops: stage 1
2506 / 2493, stage 2 1472 / 1456. **Time per frame: not measured**: `jtoken_eval.py timing` refused (2023 MiB of GPU memory in use against
its limit of 2000 MiB) and another chat's training was running; on record 16.2 ms (v1).

**System tests before the start:** research self-test (133 s), Evaluation 37 of 37 (18 s), trainkit, reporter and drawing self-tests
passed; lane tests 9 of 9 in 2032 s, of which `test_jlive_train` 1442.5 s; anchor of `E396.champ_s0` without a differing number (79 s);
TRAIN fixture of the live tokens bit for bit (17 s); `test_net_layering` passed (run by hand).

**Geometry** (a diagnostic asked for by the owner after the first pictures; the stored VAL dump of seed 0 on the CPU, no record):
delivered graph `vert100_dp` 23.81 (topo F1 0.6682, F1@0.5 0.6448, F1@1.0 0.8459); the same nodes joined by straight chords 20.66
(0.6440, 0.6345, 0.8194); chords after one smoothing pass of the node positions 19.18 (0.6436, 0.6344, 0.8168). Nodes of the delivered
chains: median 0.304 m from the nearest ground-truth line, 34.3 % over 0.5 m, 14.7 % over 1 m; at 9.0 % of the links whose ends are both
within 2 m the line steps more than 0.5 m to the other side. The rough lines come from the node positions, which the token head does not
change; the curves add vertices but carry score, and smoothing alone gains no accuracy.
