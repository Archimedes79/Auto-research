# E393 - the pyramid of B' trained end to end by the token loss, its pixel loss removed

**CLOSED 2026-11-23 (D138): the 50-step gate read (all checks pass, 3000 steps per arm by the budget rule), the main runs NOT started; where to resume: section 5.** **DRAFT, NOT RUN. Version 1.1 (2026-11-22: first critic pass REVISE, 4 blocking + 6 should-fix, all taken in; second look ACCEPT, G2 met, its two small points taken in; section 4).** Card of 2026-11-22 (system clock), Fable. **The owner's question in chat (2026-11-22):** "Has it been trained E2E including the pyramid removing the pyramid loss?" - answer: no, never (the champion's stage 2, E372 S4, trains only B''s node head; the pyramid ran frozen in every token run; the old chain's joint phase 7 trained the pyramid WITH its pixel loss (`qgraph.py` qgraph_loss, phases 2 / 7), its phase 9 dropped the pixel loss but froze the pyramid; E352's trainable-pyramid stage never ran). The owner: "Yes, draft the card". **Runs only after a second look of the critic at this version, the build with its code critic, and the owner's go recorded as a decision (the owner asked for the draft; a GPU night of about 7 - 8 GPU h is his call).**

**Scope in one line:** today 0.226 M of the champion's 2.323 M parameters train (the token head 0.127 M + B''s node head 0.099 M); the pyramid (2.097 M, counted from `PyramidHead(width=48)`) is frozen. This card makes the whole of B' (pyramid + node head) learn from the loss on the delivered graph, with the pyramid's own pixel loss switched off, and asks whether the graph gets better, stays equal, or breaks.

## 0. The card (what the tools run)
```json card
{
 "id": "E393",
 "card": "",
 "title": "E393 (DRAFT v1.2, builds done): the pyramid of B' trainable in the champion's stage 2 (S4), its pixel loss OFF, from the champion e372_s4 (head + its b_state): arm e = the token head's loss ALONE trains head, node head and pyramid; arm p = the same with B''s node loss kept as an anchor; arm n = node loss off, pyramid frozen (the control of e's second change); control c = S4 continued with the pyramid frozen; 3000 steps, two seeds, read live at D115",
 "kind": "variant",
 "deterministic": false,
 "metric": "topo_F1",
 "direction": "higher",
 "band": 0.01,
 "prediction": {"value": 0.003, "vs": "c", "note": "e - c topo F1, a guess (window -0.02 to +0.02); verdict by hand by section 1.4"},
 "design_delta": [
  {"row": "R1", "change": "'Today GaussianCaR and the pyramid are frozen' -> 'Today GaussianCaR is frozen' (only when the adopted arm trains the pyramid AND is not labelled 'in structure only', section 1.4); otherwise the row keeps the pyramid frozen and records E393 as the attack and its result"},
  {"row": "R6", "change": "stage 2 trains the head and the WHOLE of B' (pyramid + node head) through the token loss, without B''s pixel and node losses (arm e adopted) / with B''s node loss as an anchor (arm p adopted)"}
 ],
 "arms": {
  "ref": {"recipe": "the champion e372_s4_s<S> read LIVE again with its exported b_state (`jtoken_eval.py dump --live`, bs 4) at the D115 cell: the parity target (VAL topo F1 0.6682 / 0.6813, junction F1 0.2538 / 0.2481)", "run_ids": ["E393.ref_s0", "E393.ref_s1"]},
  "c": {"recipe": "the matched control: `jlive_train.py --head-run e372_s4_s<S> --b-state-from e372_s4_s<S> --train-nodes 1 --live-inputs 1 --train-pyr 0 --b-weight 1.0`, 3000 steps, lr 1e-4 (head, node head): the S4 recipe continued from the champion's head AND node head; separates 'more fine-tuning' from the treatments; never adopted", "run_ids": ["E393.c_s0", "E393.c_s1"]},
  "n": {"recipe": "as c with `--b-weight 0.0`: the node anchor dropped, the pyramid frozen: the control of e's second change (n - c = 'no node anchor', e - n = 'the pyramid learns without the anchor'); never adopted", "run_ids": ["E393.n_s0", "E393.n_s1"]},
  "p": {"recipe": "as c with `--train-pyr 1 --pyr-lr-mult 0.1`: the pyramid TRAINABLE (all its 2.097 M parameters, lr 1e-5 through the per-group lr multiplier, its own clip group 3.0), its pixel loss OFF (never computed); loss = the head's loss + 1.0 x B''s node loss", "run_ids": ["E393.p_s0", "E393.p_s1"]},
  "e": {"recipe": "as p with `--b-weight 0.0`: the token head's loss ALONE trains the head, the node head and the pyramid (the owner's question)", "run_ids": ["E393.e_s0", "E393.e_s1"]}
 },
 "control": "c",
 "variant": "e",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "topo_F1", "vs": "c", "min_effect": 0.0, "note": "verdict BY HAND by section 1.4 (LOSES > INCONCLUSIVE > GAINS > HOLDS per arm; the candidate choice p / e; the 'in structure only' label)"},
  {"metric": "junction_F1", "vs": "c", "max_loss": 0.01},
  {"metric": "topo_F1", "vs": "ref", "max_loss": 0.01},
  {"metric": "F1@1.0", "vs": "ref", "max_loss": 0.01},
  {"metric": "lanes_found", "vs": "ref", "max_loss": 0.01}
 ],
 "budget_gpu_h": 8.0,
 "kill": "a loss is not finite; the head is destroyed (mean HEAD loss over steps 1500 - 2000 > 1.10 x its mean over steps 1 - 200); the node coverage collapses (every arm, per-step `cov` in the run record: the share of GT nodes that get a token; its 200-step mean below 0.95 x its mean over steps 1 - 200) or the token set blows up (v1.2, code critic B1: the mean number of ACTIVE cells per frame - existence above P_ACTIVE 0.05 on the whole 2 m map, BEFORE the top-512 cut, uncapped; the valid tokens are this count capped at K = 512 and start near 445, so 1.25 x of them could never fire - over any 200 steps above 1.25 x its mean over steps 1 - 200); the pyramid does not receive the gradient (p, e: its gradient norm of the head's loss 0 or not finite at step 1) or receives it in c / n; more than 1 % of the steps hit LabelOverflow; the B-rise kill of E372 applies ONLY where B''s node loss is in the loss (c, p); in n and e it is reported, not a kill (nothing holds that loss down: its rise is the treatment); peak GPU memory is checked by hand from the gate (reserved, nvidia-smi), not in code; --timeout from the launch gate x 1.3; a resume or roll-back breaks the pairing of that seed (reported)",
 "evaluate": {"split": "val"}
}
```

## 1. Card text
### 1.1 The question and why
Does the delivered lane graph get better when the whole lane head B' (the pyramid included) learns from the loss on the graph, with the pyramid's own pixel loss removed? DESIGN.md R1 (D55: "a loss on the output reaches every module") names the frozen pyramid as an open exception; R6 says stage 2 (S4) did not gain over a frozen B' (E372: s4 - z topo F1 -0.0003 / +0.0049) but trained only the node head (0.099 M parameters). The pyramid is 95 % of B'; its maps were shaped by pixel losses towards GT lines, centres and road areas, not towards what the token head needs. This card measures what the token loss does to it.

### 1.2 Mechanism (E2E, D55)
- **Path of the gradient:** the token head's loss -> the token inputs (tok, pos, dirv, e, e_raw, s, uout, uin, flags; `live_inputs=True` as S4) -> B''s node head -> the pyramid outputs the node head reads (`qnodes.py` NodeHead.forward): `feat2`; `feat1` and `feat05` (pooled, and sampled bilinearly at the pulled node position: differentiable); both `road2` channels; `centre1` channels 0 and 6 - 11 -> the whole pyramid trunk. NOT read by the node head: `lines05` and `centre1` channels 1 - 5 (without the pixel loss they get no gradient: `lines05` has no gradient at all, so AdamW skips it and it stays exactly as it was; the rows 1 - 5 of `centre1` get a zero gradient inside a tensor that trains, so AdamW's weight decay 1e-4 shrinks them: both are off the delivered path, reported, and may be dropped from the network later, not here).
- **Still NOT end to end, named:** GaussianCaR (frozen, its features cached: unfreezing it needs the raw camera / radar inputs, out of scope); the top-512 index choice (argsort on the detached existence: the named budget); the labels (from the detached live tokens); the allowed-pair gate, the 8 m short / long split; the four read-out thresholds.
- **Normalisation:** GroupNorm only in the pyramid and the node head (`head.py`, `pyramid.py`, `qnodes.py`), no BatchNorm, no dropout: the forced eval mode of `LiveSource` is harmless.
- **The risk the arms separate:** without the pixel loss and (arms n, e) without the node loss, nothing but the token loss holds the node existence and positions; the top-512 choice reads the existence, so the token set itself can drift (the moving-label risk of E372, larger here). p keeps B''s node loss as an anchor; n measures dropping the anchor without the pyramid; the coverage and token-count kills guard all arms.
- **Out of scope, named:** a trainable pyramid WITH its pixel loss on (it would separate 'pyramid trainable' from 'pixel loss removed', but `LiveData` does not load the dense pixel targets: a large build, not needed for the owner's question).

### 1.3 Controls and read-outs (named before any run; VAL 873, TEST never touched)
- **c** (the matched control) continues S4 for the same 3000 steps from the champion's head AND node head (`b_state`), pyramid frozen: the same start, data order (head seed of the champion run), schedule, clip, read-out. **n** drops the node anchor with the pyramid frozen. **ref** (the champion read live) proves the parity of the live path.
- Every arm: a LIVE VAL dump at bs 4 with its exported `b_state` (`jtoken_eval.py dump --live`), scored at the fixed D115 cell (short 0.3 / long 0.2 / keep 0.6 / none gate off), the paired scene bootstrap against c and against ref of the same seed (frames paired). **Which B' ran (proven, the base = e372_s4_s<S>'s `b_state`):** p / e: the pyramid AND the node head differ from that start and equal the run's export; c / n: the node head differs from that start, the pyramid equals it.
- **Reported, not judged:** the relative change of the pyramid's output features (feat2 / feat1 / feat05, 100 VAL frames: the 'in structure only' measure of 1.4), the pyramid's relative weight change ||W_end - W_start|| / ||W_start|| (all pyramid tensors), the change of its maps on 100 VAL frames (centre1 channel 0 and road2 IoU against the GT maps, before / after), the gradient-norm share of the pyramid, the update size per group, B''s node loss in n / e, s / step and reserved memory, the coverage and token-count curves, ms per frame (the network is unchanged in shape: inference should equal the champion's 16.2 ms; one quiet check of an adopted arm only).

### 1.4 The verdict (by hand, written before any run; per arm p, e, n against c, at D115, per seed)
Per seed and arm, D = arm - c:
- **LOSES** (on a seed): topo F1 D or junction F1 D below -0.01, or a guard against ref fails (topo F1, F1@1.0, lanes found each below -0.01), or the arm's kill fired.
- **GAINS** (on a seed): not LOSES (so no kill fired), topo F1 D >= +0.010 with its paired interval above 0, junction F1 D >= -0.01, the guards against ref hold.
- **HOLDS** (on a seed): neither LOSES nor GAINS.
**The arm's verdict, in this order:** **E2E-LOSES** = LOSES on both seeds; **INCONCLUSIVE** = LOSES on one seed and HOLDS-or-GAINS on the other (a third seed needs its own B' and cache: named cost; otherwise reported as split); **E2E-GAINS** = GAINS on both seeds; **E2E-HOLDS** = the rest (both seeds HOLDS-or-GAINS, not both GAINS).
**The candidate (only p or e; c and n never):** among p and e with E2E-GAINS or E2E-HOLDS: **p** if p - e topo F1 >= +0.010 with the paired interval above 0 on both seeds, else **e** (both satisfy R1 - an auxiliary loss is allowed - so the anchor is kept only when it pays). The candidate goes to the owner, who decides the champion change (DECISIONS.md); nothing is adopted by this card alone.
**'In structure only':** if the candidate's pyramid OUTPUT barely moved - the relative change of the features the node head reads, ||F_end - F_start|| / ||F_start|| over `feat2`, `feat1`, `feat05` on the same 100 VAL frames, below 1 % (the mean of the two seeds) - its result is labelled 'in structure only' (the pyramid is trainable but its output barely changed) and R1 is not reworded on it. (Second look: a weight-change floor would not fire, since Adam's step noise alone moves the weights more than 0.5 % over 3000 steps; the weight change stays reported.)
**Reading of the decomposition (reported):** e - n = what the pyramid learns without the anchor; n - c = dropping the anchor alone; p - c = the anchored pyramid.

### 1.5 Prediction (a guess, before any run)
e - c topo F1 **+0.003** (window -0.02 to +0.02), p - c +0.004, n - c -0.002; junction F1 about equal. Outcome of arm e: E2E-HOLDS 50 %, E2E-GAINS 15 %, E2E-LOSES 25 % (drift of the token set without an anchor), a voided / killed run 10 %; 'in structure only' about 30 % (lr 1e-5, 3000 steps). Reasons: S4 on the node head gained nothing measurable; the pyramid was trained on dense GT maps that already encode the lanes; the token loss is far sparser than the pixel loss; the frozen foundation features cap what any head can see.

### 1.6 Cost and launch gate
- S4 today: 0.29 - 0.37 s / step, peak allocated 4.1 GB (E372 gate). A trainable pyramid at 200 x 200 x bs 8 adds its float32 activations and backward: unmeasured here (E352 quotes 11 s / step and 11.3 GB in E286's other code; the critic's estimate for this path is 0.5 - 0.8 s / step).
- **The 50-step gate (diagnostic, `run.py --no-card`, before the night):** c, n, p, e on seed 0, ONE slot, nothing else on the GPU, 50 steps each: finite losses, the gradient check (pyramid gradient > 0 in p / e, = 0 in c / n), the per-step `cov` and the active-cell count `nact` recorded (written into this card before the night: its start level and its 50-step spread in c and e; expected near the champion's 445 valid tokens, since those are the active cells cut at 512), s / step, and the RESERVED peak memory (`torch.cuda.max_memory_reserved` and nvidia-smi), not the allocated one.
- **The budget rule (written now):** if p and e take <= 1.5 s / step: 3000 steps for every arm (p / e about 1.3 GPU h per run, c / n about 0.3 h: about 6.4 GPU h for the eight runs, plus the dumps). If 1.5 - 4 s / step: all four arms 1500 steps (equal budget; said in the result). If > 4 s / step or a single run's reserved peak > 10.5 GB: STOP, report (gradient checkpointing of the pyramid is then a code step of its own; the trainer has no gradient accumulation, so no accumulation fallback). **Slots:** two runs in parallel only if the gate's summed reserved peaks of the pair stay below 11 GB; otherwise serially (6.4 h from 22:00: the last start well before 05:00; checked against the window before the launch).

## 2. Detail plan (block experiment; DRAFT)
| step | what | who / where | output |
|---|---|---|---|
| build 1 | `jlive_train.py --b-state-from <run>`: the start B' = `b_state` of that run (the head from `--head-run` as today); the cache sha check stays on the ORIGINAL e292_a file (`batches()` / `lane_live_tok` check `run_sha`); the option in RECIPE | Fable | code + test |
| build 2 | `training/trainer.py`: a per-group `lr_mult` key (default 1.0: every old run bit for bit) that the schedule multiplies in every step, AdamW from groups; `jlive_train.py --pyr-lr-mult` uses it. Shared trainer code: its own test (old runs unchanged, the multiplier applied), the code critic | Fable | code + test |
| build 3 | `lane_live_tok.py LiveSource(train_pyr=)`: the pyramid outside `no_grad` when trainable, `trainable()` includes it; `jlive_train.py`: the pyramid its own clip group in the StepHook (not the node group), the export allows moved pyramid tensors when `train_pyr`, checkpoint / resume of the whole `b_state` | Fable | code + test |
| build 4 | `jlive_train.py`: a per-step `cov` (GT-node coverage) and valid-token count in the run record (`Joint.rec`, survives resume), the coverage and token-count kills in `verdict()`, the B-rise kill only when `b_weight > 0` (reported otherwise) | Fable | code + test |
| build 5 | `jtoken_eval.py check_b` / `score --live`: the proof base = the start run's `b_state` (section 1.3) instead of e292_a for runs started with `--b-state-from` | Fable | code + test |
| tests | `grad` check with mutants: the head's loss reaches every pyramid tensor the node head reads in p / e (zero rows allowed in `centre1.weight` for channels 1 - 5; `lines05` none), none in c / n; a mutant keeping the pyramid under `no_grad` must be caught; the lr multiplier applied (a mutant without it caught); the export round trip; the start state = e372_s4's `b_state` bit for bit | Fable | `test_lane_live.py`, `test_jlive_train.py`, trainer test green |
| read-out tools | before the read, not before the runs (code critic S6): the relative change of the pyramid's output features (feat2 / feat1 / feat05, start against end, the same 100 VAL frames: the 'in structure only' measure) and the map read-outs of 1.3 (centre1 channel 0 and road2 IoU against the GT maps, before / after), one small script on CPU or a free GPU | Fable | numbers in the result section |
| critic | the second look at this card; a fresh ar-critic on the diff (code) | - | ACCEPT |
| gate | the 50-step gate of section 1.6 | run.py --no-card, GPU window | timing, reserved memory, budget and slot choice |
| runs | ref, c, n, p, e x seeds 0 / 1 through `run.py --require-card` in the night window (start <= 05:00), each followed by its live VAL dump and its CPU read-out | GPU | cards, ledger rows |
| read | section 1.4 by hand, the result section, ledger notes, the DESIGN.md delta proposal to the lead | Fable | result |

## 3. Read-out log
(empty: not run)

## 4. Referee
**First critic pass (2026-11-22, fresh ar-critic, cards): REVISE**, not G2; the question confirmed new and the cited numbers checked (pyramid 2 096 708 and node head 99 304 parameters counted; E372 numbers; E352's 11 s / 11.3 GB). Taken in:
- B1 the start state could not be built (`jlive_train.py` loads B' through `QP.load_model` from `--b-run`, which reads `ck["state"]`; e372_s4 keeps B' under `b_state`; the cache sha check passes only e292_a): build 1 `--b-state-from`.
- B2 `--pyr-lr-mult` could not work (`trainer.fit` sets every group's lr to the one schedule value; AdamW from a flat list): build 2, per-group `lr_mult`.
- B3 the B-rise kill would have killed arm e by its treatment: it applies only where the node loss is in the loss; reported in n / e.
- B4 no clean control for e's second change: arm n (node anchor off, pyramid frozen) added. A pixel-loss-ON trainable-pyramid arm named out of scope (no dense pixel targets in `LiveData`).
- S1 overlapping verdict branches: per-seed classes and the order LOSES > INCONCLUSIVE > GAINS > HOLDS.
- S2 the candidate choice: p only if p - e >= +0.010 with the interval above 0 on both seeds, else e.
- S3 the coverage kill was not measurable (`gt_cov` only as a 100-step log window): build 4 per-step `cov` in the record, the kill at 0.95 x, and a token-count blow-up kill.
- S4 memory and slots: reserved memory from the gate, parallel only if the pair stays below 11 GB, no accumulation fallback (the trainer has none), memory checked by hand.
- S5 the 'in structure only' label: relative pyramid weight change below 0.5 %.
- S6 the B' proof needs the start run as its base: build 5.
- Mechanism claims corrected: GroupNorm confirmed (no BatchNorm); the node head reads centre1 channels 0 and 6 - 11 only; the live path cannot train the pyramid today (`no_grad`, frozen parameters, the export refusal, the clip groups): builds 3 - 4.

**Second look (2026-11-22, the same critic): ACCEPT, G2 met for the card** (B1 - B4 and S1 - S6 resolved; the verdict order mechanical without overlaps; arm n set up right). Its two small points taken in (v1.1): the 'in structure only' floor moved from the weights (Adam's noise alone would pass 0.5 %) to the pyramid's output features (below 1 % relative change); GAINS on a seed now says 'not LOSES' (no kill fired). The runs still wait for builds 1 - 5, their code critic, the 50-step gate and the owner's recorded go.

**Code critic (2026-11-23, a fresh ar-critic, mode code, on the diff of builds 1 - 5): REVISE**, one blocking point, in the card's rule, not in the code. Taken in (v1.2, before any run):
- B1 the token-count kill could never fire: the valid tokens are capped at K = 512 and the champion's S4 starts near 445 (e372_s4 n_valid 435 - 451), so 1.25 x is 556. The kill now reads the ACTIVE cells per frame before the top-512 cut (existence above P_ACTIVE, uncapped; `jlive_train.py` NACT_RISE, record `nact`), test values at the real scale.
- S1 with b_weight 0 (n, e) B''s node loss is no longer in the backward (a non-finite gradient there would have turned into NaN through 0 x inf); runs with b_weight > 0 unchanged. S2 a mutant for the pyramid-gradient check that counts B''s loss as the head's, with a test where only B''s loss reaches the pyramid. S3 `jthresh.py` passes `runs` to `check_b`. S4 the read-out test's start run also differs outside the node head and pyramid. S5 the `lines05` wording of 1.2. S6 the read-out tools named in section 2 (needed for the read, not for the runs). S7 a resume from a checkpoint of before `cov` / `nact` restarts the kills' start window: named in the run note if it happens.
- Confirmed by the critic: the older runs (train_pyr 0, no b_state_from) keep their training math bit for bit (the E392 chat checked it on e392_s4_c_s0, which ran this code: steps 100 / 500 / 1000 equal e372_s4_s0 to 4 digits).
- Mutants (2026-11-23): `test_jlive_train.py` 70 of 70 killed (a first pass found 2 stale anchors and 3 weak checks: an old config without the E393 keys, the length of cov / nact per step, NaN in cov / nact on a skipped step; tests added), `test_lane_live.py` 67 of 67 killed; the selftests of `trainer_selftest.py`, `lane_live_tok.py` and `test_jtoken_eval.py live` pass. Not yet done: the 50-step gate on the GPU (`run.py --no-card`) and the owner's go (D<n>).

## 5. The 50-step gate (2026-11-23) and the closure (D138)
**Gate (section 1.6; diagnostic, `run.py --no-card`, no verdict), seed 0, 50 steps each, one job at a time, nothing else on the GPU:** run ids `E393.gate_<arm>_s0`, run folders `runs/e393_gate_<arm>_s0/`; read by `temp/e393_gate/e393_gate_read.py` (scratch: per-step `cov` / `nact` from `model.pt`'s record, s / step and reserved memory from `log.txt`, the nvidia-smi peak from the 0.5 s samples). Arm c started 08:05, before the 08:30 weekday window (the lead: small, leave it); the chain was then paused until the lane chat's r2 timing was done (D138) and ran n / p / e 09:24 - 09:33.

| arm | s / step (last 40) | reserved peak (torch) | nvidia-smi peak (whole GPU) | cov: steps 1 - 10 / all 50 [min, max] | nact: steps 1 - 10 / all 50, sd [min, max] | head loss steps 1 - 10 -> 41 - 50 | gradient at step 1 |
|---|---|---|---|---|---|---|---|
| c | 0.266 | 4.63 GB | 5538 MiB | 0.981 / 0.983 [0.951, 0.996] | 595 / 562, sd 63 [417, 687] | 0.403 -> 0.418 | node head 1.139 |
| n | 0.246 | 4.63 GB | 6273 MiB | 0.980 / 0.982 [0.950, 0.996] | 598 / 560, sd 64 [420, 689] | 0.401 -> 0.418 | node head 1.139 |
| p | 0.288 | 5.26 GB | 6876 MiB | 0.981 / 0.984 [0.951, 0.996] | 596 / 563, sd 63 [418, 690] | 0.403 -> 0.418 | node head 1.139, pyramid 0.641 |
| e | 0.365 | 5.26 GB | 7058 MiB | 0.980 / 0.982 [0.951, 0.996] | 598 / 560, sd 64 [421, 691] | 0.400 -> 0.417 | node head 1.139, pyramid 0.641 |

- **Every check of 1.6 passes:** finite losses in every arm; the pyramid gets a gradient from the head's loss in p / e (0.641 at step 1; only `lines05.weight` / `.bias` without one, as section 1.2 says) and none in c / n (the frozen-gradient kill did not fire); no LabelOverflow. The nvidia-smi peak includes about 1 - 2 GB of the desktop and other processes.
- **`nact` (the active cells per frame before the top-512 cut, uncapped):** starts near 596 (sd 63 per step, range 417 - 691), so above the 445 valid tokens the card expected (those are the active cells cut at K 512); the kill at 1.25 x the start is about 745 on a 200-step mean (whose spread is about 63 / sqrt(200) = 4.5): far from the noise.
- **The budget rule (written before the gate):** p / e take 0.29 - 0.37 s / step (<= 1.5) -> **3000 steps for every arm**; the reserved peaks of a pair (5.26 + 5.26 = 10.5 GB) stay below 11 GB -> two runs in parallel would be allowed. The trainable pyramid costs +0.6 GB reserved and +0.02 - 0.1 s / step, far below the critic's estimate (0.5 - 0.8 s / step).
- **Seen, not judged (50 steps):** the head's loss rises slightly in every arm (0.40 -> 0.42) and B''s node loss too (1.46 -> 1.55), c and n included, so it is not the pyramid's doing; the 200-step windows of the kills would read it.

**CLOSED (2026-11-23, D138): gate read; the main runs were NOT started** (D138: no new experiment is started; the owner leaves the computer). Nothing of E393 is adopted; the champion is unchanged; no verdict exists (the gate is a diagnostic).

**Where to resume:** the code is ready and tested (builds 1 - 5, commit 9a2ec95c; mutants 70 / 70 and 67 / 67); the card's rule, prediction and verdict order (sections 1.4, 1.5) stand as written. To start: a fresh `ar-critic` look at this card (the gate numbers above are new; the D<n> go of the owner), then the ten runs of section 2 through `run.py --require-card` in a GPU window (ref, c, n, p, e x seeds 0 / 1; 3000 steps each; by the gate's clock c / n about 0.2 h and p / e about 0.25 - 0.3 h per training run, so about 2 GPU h for the eight trainings, about 1 h of wall time with two in parallel, plus the dumps), each followed by its live VAL dump; then the read-out tools of section 2 (feature change of the pyramid, map IoU) and the read by section 1.4. The gate's run folders `runs/e393_gate_*` are diagnostics, never a start point.
