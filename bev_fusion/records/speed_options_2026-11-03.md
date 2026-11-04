# Training speed: where the host-bound step spends its launches, and the levers (CPU analysis, 2026-11-03)

Asked by Mirko ("find options to improve training speed"). Nothing was run on the GPU and no code was changed: the box trained `t20.base_s0` (4.09 s per step at step 4480 of 6000) and the frame is frozen except what D48 lifted. Every number below is labelled: **measured on CPU** (torch.profiler / dynamo on the t20 chain, B = 1, 15 synthetic boxes per sample, a warm keyframe, the small grid; scratch scripts in `bev_fusion/temp/speed_cpu/`, not kept), **read from code**, or **guessed**. The earlier GPU facts (33 % kernel busy, 691 `.item()`, 1,283 syncs, 1,779 copies per step at B = 3 on the old chain; `speed.a3` 2.66 s per step, `speed.b`, `speed.c2`) are in `autoresearch/trainsuite/README.md` section 8 and the ledger rows `speed.*`.

## 1. Operators per keyframe-batch (measured on CPU)
Leaf aten operators with view-only operators dropped, as a proxy for kernel launches. One training step is 6 keyframe-batches (accumulation 2 x clip length 3): about 43-50 thousand operators, about 85 microseconds each at 4.09 s per step. The same method on the old 3-keyframe `t10` step gives about 20 thousand, which matches the measured launch count.

| section (calls) | forward | backward | share |
|---|---|---|---|
| `splat.splat` (9 = 6 mover + 3 window) | 966 | 565 | 21.6 % |
| `loss.map_terms` (8) | 577 | 580 | 16.3 % |
| `sem_head` + `obj_head` (3 + 3) | 234 | 572 | 11.4 % |
| `head_loss` (3) | 315 | 283 | 8.4 % |
| `motion_block` | 298 | 177 | 6.7 % |
| trunk + `model.forward` glue | 179 | 211 | 5.5 % |
| `cam_temporal` (K = 3) | 280 | 86 | 5.2 % |
| mover without the splat (6) | 198 | 109 | 4.3 % |
| `build_layer_targets` (3) | 236 | 0 | 3.3 % |
| `chain_positions`, host target build, rest | | | about 17 % |

Multipliers (read from code): the `for dt` loop at `model.py:451` runs 3 times; both heads call the shared mover (`timed_head.py:104-110`), so there are 6 splats per keyframe and 3 of them repeat work; the 4-corner loop at `splat.py:167`; the window loop at `splat.py:210`; the per-sample lift loop in `feature_cache.py` (+310 operators per extra sample, measured B = 2 against B = 1: +4 %); the per-box target loops at `object_head.py:76-124` (about 38 operators per box, on the host). Optimiser and EMA: 209 parameter tensors, 2 groups, 376 EMA entries, about 150-250 launches per step (estimated), under 1 %.

## 2. Sync-like operations in the step (CPU dispatch trace, per keyframe-batch at B = 1, times 6 per step)
`losses.py:383` (`motion_term`): 12 boolean-mask indexes plus 3 or more backward `index_put` (one `nonzero()` per keyframe gives the same result bit for bit). `radar_encoder.py:84/90/121/124`: 10. `motion_block.py:48-60`: 6. `layers.py:108` (1 m head targets): 3 masks and 3 `unique` (GT only: can be precomputed on the host). `temporal_fusion.py:78`: 2 `.item()` per sample. `lift_pd.py:165-166` and `205-208`: 5 or more `.tolist()` / `.item()` (K can be read from the host batch). `losses.py:348`: 3 device-to-host copies of `w_obj`. Read from code, not traced: `model.py:418/430` `bool(flip.any())` twice (`train.py:261` moves `flip` to the GPU; frozen), `train.py:486` isfinite, `loop.py:166` isfinite once per step. About 44 per keyframe-batch, about 260 per step at B = 1 (the GPU count was 690 `.item()` and 1,283 syncs at B = 3: the B-scaling loops were not measured). Pageable host-to-device copies: `losses.py:128` copies once per sample, GT key and dt: about 207 per keyframe, about 1,300 per step (counted from the code).

## 3. Can it be compiled or captured? (`dynamo.explain`, eager backend, static shapes: measured on CPU)
The whole forward gives 35 graphs and 34 breaks (`lift_pd` `.tolist()` x 9, radar and `motion_block` dynamic shapes x 10, `recentre` `.item()` x 3); the loss gives 40 graphs and 39 breaks (per-box `float()` at `losses.py:312-313`, `layers.py:108`, `losses.py:383`). Zero breaks: trunk + fusion (76 operators), `sem_head` with mover and splat (146), `splat_bev` (127), the motion-block net (180), `map_terms` (94). One CUDA graph of the whole step is therefore impossible; the parts above compile. Triton 3.4 and gcc are in `gcar`. Compiling the trunk alone touches about 5 % of the operators.

## 4. Levers, ranked (operator counts measured, time gains guessed)
| # | lever | size | bit for bit? | where (frozen?) | the GPU measurement that confirms it |
|---|---|---|---|---|---|
| 1 | one mover call per dt, shared by both heads | -660 operators per keyframe (-9 %) | same math (CPU bitwise; CUDA atomics) | `model.py:455`, `timed_head.py` (open) | `trainsuite speed`, 70 steps, quiet GPU |
| 2 | `splat()` with one `index_add` over 4N entries | about -1,000 per keyframe (-13 %) | equal to rounding | `splat.py` (open) | same run |
| 3 | the bit-for-bit sync removals of section 2 | about -35 of 44 per keyframe | yes | open files (`flip` and `isfinite` are frozen) | syncs per step in `profile.txt` |
| 4 | stack the GT on the host, one copy per key | -140 copies per keyframe | yes | `losses.py` (open) | same run |
| 5 | `torch.compile` of the zero-break parts | guess -15 to -20 % of the operators | no (fusion) | model builders | compile time + 70 steps |
| 6 | the 3 dts as one batch of 3B | about -40 % | no: BatchNorm statistics over dt (open finding #5) | `model.py`, loss | needs Mirko's word; a new baseline |

Not worth it while the step is host-bound: fused AdamW (below 1 %), `cudnn_benchmark`, TF32 (risky for the coordinates); a loader lever is capped by the data wait (6-7 %; targets in the workers: at most 4 %, blocked because `w_obj` depends on the splat mass); a larger micro-batch is blocked by memory (8.8-9.3 GB of 11 GB) and by BatchNorm.

## 5. Not done
No GPU timing. The full-grid B = 1 run peaked at 3.44 GB, so B = 3 at full size was not run: the per-step table is scaled from B = 1 and B = 2 on the small grid. The box and radar sections depend on real data (here 15 boxes, 40 radar points). The profiler returned no stack information, so the sync sites come from a dispatch trace and the code, not from stacks.

## 6. Measured on the GPU: where the 4.1 s of a t20 step go (2026-11-04, diagnostic, no verdict)
Command: `temp/speed_tools/phase_sampler.py` (scratch; a wall-clock stack sampler, every 3 ms, on the pinned code of t20.base_s0, config `temp/compose/t20_base_s0.json`, fresh init, 44 steps, the first 10 not counted), run as `timing.phase_a` through run.py (`--no-card`, alone on the GPU, machine lock). Window: 34 steps, median 4.10 s, mean 4.10 s, min 3.64, max 4.66; no trend inside it (first half 4.12, second half 4.09). Box load average 2.3 before and 3.2-3.8 after (other chats' CPU work). `nvidia-smi dmon`: an SM kernel was running 64 % of the window (138 one-second samples). The sampler's wall time per section adds up to the step time by construction; time at a blocking call is counted where the main thread waits.

| what | s per step | share | where |
|---|---|---|---|
| `backward()` (one call, the autograd engine) | 1.24 | 30 % | `train.py:492` |
| blocking reads GPU -> host (wait for the queue to drain) | 0.70 | 17 % | `lift_pd.py:165` `.tolist()` of K and T_cam_car 0.50, `_unpack` 0.11, `model.py:418` `flip.any()` 0.09 |
| small copies host -> GPU (pageable) | 0.64 | 15 % | `splat.py:57` 0.20, `losses.py:127` 0.20, `train.py:405` (ego tensors) 0.20, `to_device` + `head_loss` 0.04 |
| wait for the data (6 workers) | 0.32 | 8 % | `loop.py:132` |
| forward operators (model + loss, without the lines above) | 1.11 | 27 % | `lift_pd` 0.03, `splat` ~0.15, `build_targets` 0.10, the rest spread over 30 files |
| optimiser, EMA, glue, flip augment, radar aux loss | 0.10 | 2 % | |

Reading: the operator floor measured with tiny tensors (26k forward + 20k backward operators, 21 / 32 us) is 0.56 s forward and 0.64 s backward; the forward and the backward each take about twice that. Copies and syncs are 1.34 s (33 %), not the 0.5 s of the first estimate. The GPU is not idle-bound: it ran a kernel 64 % of the time, so the step alternates between host-limited stretches (copies, tiny operators) and GPU-limited ones (the waits at the syncs, the backward). A sync that waits 0.5 s per step over 6 to 18 calls waits for queued GPU work, so removing it moves the wait to the next consumer; the most the host stalls can give is the GPU idle share, 36 % of 4.1 s (4.1 -> about 2.6 s), the real gain is less.
Drift: a fresh model in a fresh process takes 4.10 s today, as long as the late steps of the training (4.1-4.2 s), against 3.2 s in its first 1,000 steps. So the growth of the step time is not (only) a property of the weights or of a long-running process; the box (CPU load of other chats, clocks) or the sampler's own cost remain. Not settled: `timing.phase_c` (warm start from the trained checkpoint) was run next to t21.rec_s0 and is void (see the memory note training-speedup-queue); the rerun adds `phase_b` (fresh, no sampler) to price the instrument.

## 7. The launch- and sync-saving variants measured (D53, 2026-11-04, diagnostic, no verdict)
What: `ModelConfig.share_mover`, `splat_fused`, `sync_free` + `LossConfig.sync_free` (training/launch_saving.py, commit 53e5a22; levers 1 - 3 above, lever 4 was D52). Command: six `queue add` items of `training/configs/t20_base.json` with `--preset timing --set split=small --no-card`
(70 steps, batch 3 x accumulation 2, the 10-scene small set, probe on), one after the other, alone on the GPU (machine lock `tools-timing-d53`), `python -m trainsuite --root ../bev_fusion/runs speed ... --skip 3`; box load 1.6 - 2.3 during the batch. Run ids `speed2.*`.

| run | flags | s per step | vs plain | peak GB | loss at step 70 |
|---|---|---|---|---|---|
| speed2.plain | none | 3.86 | x1.00 | 9.5 | 31.95 |
| speed2.plain2 | none (the same again, last of the batch) | 3.69 | x1.05 | 9.5 | 31.92 |
| speed2.share | model.share_mover | 3.65 | x1.06 | 8.7 | 31.85 |
| speed2.fused | model.splat_fused | 3.61 | x1.07 | 8.9 | 32.03 |
| speed2.sync | model.sync_free + loss.sync_free | 3.68 | x1.05 | 9.5 | 31.96 |
| speed2.fast | all four (preset `fast`) | 3.51 | x1.10 | 8.3 | 31.78 |

Reading: the two identical runs differ by 4.4 % (3.86 against 3.69 s), so a single lever is inside the noise; all four together are 9 % under the first plain run and 5 % under the second, and use 1.2 GB less peak memory (share_mover -0.8, splat_fused -0.6: the fewer
copies of the (C, B X Y) sum). The loss at step 70 agrees within 0.5 % in every run (the identical pair differs by 0.1 %): on the GPU the scatter sums are not bit-reproducible, the CPU test proves the arithmetic. This is what section 6 predicts: the splat is about 0.15 s of the 4.1 s, the
measured stalls are elsewhere. Not done, and the larger lever: `lift_pd.py` calibration key `.tolist()` (0.50 s), the ego tensors `train.py:405` (0.20 s), `flip.any()` `model.py:418` (0.09 s), `_unpack` (0.11 s): a host-side calibration key from the data path and the frozen files. The memory the variants free
is the room for a larger micro-batch (lever 4 of the first ranking), not measured. (Section 8: the host-side key and the ego tensors were done under D54; they save nothing measurable.)


## 8. The host copies of the batch measured (D54, 2026-11-04, diagnostic, no verdict)
What: the waits section 6 named, removed without changing a value (D54: `train.host_side` / `batch["host"]`, `batch_ego` and `clip_step` in one pinned copy, the projection-table cache key and the flip test read from the host copies, pinned copies and a cached index in the loss, the dt embedding filled on the GPU; all behind `ModelConfig.sync_free` / `LossConfig.sync_free` except the ego tensors, which are always the one-copy form; `tests/test_host_batch.py`, FROZEN.md changelog 2026-11-04).

**Census** (how many calls wait for the GPU, per step): `torch.cuda.set_sync_debug_mode("warn")` over 10 steps of the preset-`fast` t20 chain, counted per Python line (`speed3.census0` before, `speed3.census1` after, with `pin_memory`; the probe, checkpoint and model-load lines are left out, they are artefacts of the 10-step run).

| group | before, per step | after, per step |
|---|---|---|
| the batch's own tensors, `to_device` (pageable copies) | 252 | 0 (`pin_memory`) |
| lift and visibility cache keys (`.tolist()` of K, T_cam_car, ida, the height and depth buffers) | 288 | 0 (7 in the whole run: the first tables) |
| loss targets of the object head, GT stacking, weights (`.to(dev)`) | 173 + 93 | 0 |
| channel index lists in the map loss (`logits[:, [..]]`) | 131 | 0 |
| ego tensors (`batch_ego`, `clip_step`) | 48 | 0 |
| `flip.any()` (model, precompute) | 18 | 0 |
| left: radar raster slots (boolean masks), the motion term's one `nonzero`, the object weights read, the memory shift, `isfinite` of the loss | | about 108 in all (radar encoder 54, losses 32) |
| **all, without probe / checkpoint** | **1,122** | **108** |

**Timing** (the section 7 protocol: `t20_base`, preset `timing`, split small, 70 steps, alone on the GPU under the machine lock, preset `fast` = the D53 flags): 

| run | what | s per step | data wait | peak GB | loss at step 70 |
|---|---|---|---|---|---|
| speed2.fast | D53 flags, code of 53e5a22 (section 7) | 3.51 | 4 % | 8.3 | 31.78 |
| speed3.fastb | + D54 host copies (the working tree of this record) | 3.51 | 0 % | 8.3 | 31.89 |
| speed3.pinb | + `pin_memory: true` | 3.28 | 0 % | 8.3 | 31.83 |
| speed3.pin | the same, first run | 3.33 | 0 % | 8.3 | 31.72 |
| speed3.pind | the same, started directly with run.py like the control below | 3.32 | 0 % | 8.3 | 31.84 |
| **speed3.pinold** | **control: `pin_memory: true` on the code of HEAD (D53 only, no D54 copies), started directly** | **4.93** | 0 % | 8.3 | 31.86 |
| (speed3.fast | the D54 run of the first pair: 3.82 s, data wait 8 %; the `ar-critic` ran its tests on the same box at the time, so it is not used) | | | | |

Reading: **the 1,014 waits that went away save nothing by themselves (3.51 s against 3.51 s without `pin_memory`). They are what makes `pin_memory` usable.** The step is bound by the host issuing its ~45 thousand operators (section 1), not by waiting: a wait costs time only when the GPU is far behind, and it is not (kernel busy 64 %); the 4.1 s of section 6 were not a sum of stalls to be recovered. `pin_memory: true` on the D53 code is **40 % slower** (4.93 s, the control; every step of the run is slow, the launch path is not the cause: the D54 code started the same way takes 3.32 s), on the D54 code it is **6 % faster** than without it: 3.28, 3.32 and 3.33 s in three runs against 3.51 s (data wait 4 % -> 0 %). Why is not measured; the likely mechanism is that the loader's pinning thread (a pinned allocation synchronises the device) and the ~1,100 copies and reads from pageable memory of the main thread get in each other's way, and only the second one is removed by D54. Against the plain run of section 7 (3.86 s; its twin 3.69 s) the best configuration, preset `fast` + D54 + `pin_memory`, is -14 % (-11 %). Memory is unchanged, the loss at step 70 agrees with all earlier runs within 0.5 % (scatter order on the GPU). To get further the number of operators has to fall (CUDA graphs per phase, `torch.compile` of the blocks, section 3), not the waits.
