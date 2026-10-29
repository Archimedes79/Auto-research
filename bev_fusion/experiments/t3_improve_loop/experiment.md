# t3_improve_loop - eight documented tries to improve the camera + radar BEV system (card + living log, started 2026-10-29)

Request (Mirko, 2026-10-29): improve performance step by step; start by using what already performs (port weak blocks into our system, or our
blocks into a standard network); test components and training on the way to SOA level; look into the components that do NOT work and say why;
document every step; after 8 tries a report with output images, performance, weak spots and compatibility with SOA.
Earlier guidance (2026-10-29): start from a good vision BEV network, one step at a time, calibration first, fall back to fast standard blocks.

## 0 Where we start (from `t2_block_swaps`, cold VAL-1000, 82 scenes, 3000 steps, seed 0, numbers from `runs/t2_*/eval/metrics.json`)

| system | mAP | veh AP@2 | ped AP@2 | drivable | walkway | lane div@0.5 |
|---|---|---|---|---|---|---|
| `ref_s0` (champion: vision net + radar + window memory + concat fusion) | 0.277 | 0.606 | 0.204 | 0.666 | 0.191 | 0.181 |
| `radar_off` (rung 0, camera only) | 0.147 | 0.448 | 0.104 | 0.637 | 0.107 | 0.164 |
| `cam_off` (radar only) | 0.247 | 0.610 | 0.195 | 0.653 | 0.174 | 0.171 |

Calibration is checked and right (`t2_block_swaps/results/calibration.md`). Open puzzles: (a) radar-only keeps almost all detection AP,
including pedestrians (0.195): is that real or an artefact? (b) the camera adds little on top of radar; (c) maps are blurry, detection overfits.

## 1 The eight tries (one change each against the champion `ref_s0`, same data / seed / schedule unless the try says so; ~1.3 h GPU each)

| try | question | what is done | prediction (written before the run) | decision rule |
|---|---|---|---|---|
| 1 | does the model really use each input? is radar-only real? | sanity controls on trained checkpoints: shuffled / zeroed radar, shuffled / zeroed camera features (`evaluate_run.py --control`), no training | `cam_off` + shuffled radar: mAP < 0.03; `ref_s0` + zero camera: mAP 0.20-0.26; `ref_s0` + shuffled camera: below zero camera | shuffled radar mAP > 0.10 for `cam_off` = artefact (prior / leak) -> stop and find it first |
| 2 | standard fusion 1: rectify gates (GaussianCaR style) | `fusion_gate` | tie (+-0.01 mAP) | band = max(0.010, 2 x seed spread), paired bootstrap |
| 3 | tracking / time vs a standard element: ego warp + ConvGRU (FIERY style); null = current frame only | `camtemp_gru`, `camtemp_none` | GRU -0.01 .. tie; none -0.01 | same |
| 4 | standard fusion 2: CMX cross-modal fuser | `fusion_cmx` | -0.02 | same |
| 5 | standard radar encoder: point MLP + scatter-max pillars | `radar_pillars` | tie | same |
| 6 | are the frozen detection-trained image features the bottleneck (blurry maps, little camera gain)? | trainable zero-initialised residual adapter on the cached context features before the lift (`cam_adapter`) | maps +0.02 IoU, mAP +0.01 | BETTER on a headline metric and no WORSE -> adopt |
| 7 | is 3000 steps too short? | champion config with 6000 steps (`ref_long`) | mAP +0.03 | gain > band -> schedule matters, use the long schedule from here |
| 8 | does more data help? | best adopted blocks, 6000 steps, all 700 train scenes (parts 1-10), evaluated on the same 82 val scenes and on all 150 | +0.02 mAP over try 7 | gain > band -> data-limited |

Controls of every try: `ref_s0` and `ref_s1` (noise floor); a try is WORSE / BETTER only outside the band AND with a paired-bootstrap interval that
excludes 0; TIE -> the cheaper variant. Order in the GPU queue: 1, 2, 3, 4, 5 (queue `temp/study/queue.txt`, lane `temp/study/lane2.sh`), then 6, 7, 8
(they need new code; lane 3). Not in the eight (kept in the queue behind them): `trunk_second`, `lift_s`, `trunk_small` of the swap study.

## 2 Step log (one entry per step: what was done, the numbers, what it means)

| time | step | result |
|---|---|---|
| 2026-10-29 20:43 | queue lane started (detached Windows process); `camtemp_none` resumes from step 2000 | running |
