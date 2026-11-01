# t15 motion block, joint form - does the motion block trained INSIDE the chain revive the velocity without costing the maps, and is it needed? (DRAFT card 2026-11-01, revised twice the same evening (BEV system chat's points; the IoU misreading; the t13 critic items that hit this card), before any run; Mirko's go for building the joint form: "Yes, go" in the Temporal Fusion chat, 2026-11-01 evening (answer to "Build the joint form of the motion block now?"), code and CPU tests only, no run before the conditions of section 1; critic pass (cards) by the BEV system chat before any run)

Context: `../t12_motion_block/experiment.md` (KEEP on both seeds, block trained alone on the frozen `t10_dual_s0`: moving EPE
6.67 -> 3.32 m/s, box-centre field 2.26 / 2.29 against 2.43 for the hurdle object head; negative: static EPE 0.055 -> 0.08; 17 ms per
frame; the "vehicle IoU at +-0.5 s -0.03" first reported there was a misreading of a loss term, `sem.obj_iou` = 1 - soft IoU, which
the block LOWERED, see the correction in that card). That KEEP means "write the joint-training card": this card. t13 (complete
network) reuses the M runs as its B0+M row and does not run them again (agreed with the BEV system chat, 2026-11-01).

## 1 Card

- **Question.** Trained from the start with the chain (the `t10_dual` recipe), does the motion block give a better velocity than
  (a) the incumbent `t10_dual_s0` / `s1`, whose velocity is dead, and (b) the cheaper fixes that revive the old 1 x 1 head (the
  object-head hurdle; gradient accumulation), with the maps, the boxes and the moved maps at +-0.5 s held?
- **Hypothesis.** Yes for (a) and (b): the old head reads one frame and cannot see motion; the block reads two frames and the
  Doppler. Known risk for (b): the recipe alone revives the old head. Verified in the logs: the validation `group.motion` of
  `t10_dual_s0` / `s1` is 1.0945 flat from step 1000 to 3000, while the stopped long run (`t10.dual_long_s0`: accumulation 2, hurdle,
  from scratch) reached 0.537 at step 2000 and 0.402 at step 4000, the order of magnitude of t12's frozen block (0.67, on another
  subset). So "M beats a dead head" is NOT "M is needed"; arm C2 below answers the second.
- **Design.** Two flags in `ModelConfig`, both default off: `motion_block` (the block of `training/motion_block.py` computes the
  field once per forward from the current camera BEV, the oldest window frame ego-aligned by `splat_window` and the radar raster
  (`block_inputs`); it replaces `TimeMover.motion`, which is then not built; the coarser levels take the same motion term) and
  `box_vel_from_field` (the object head's `vel` = the field; at +-0.5 s the field is moved with the state and turned with the ego
  yaw; the head's own vel branch is frozen). Arms, all `t10_dual` config, batch 3, 3000 steps, GT pinned to `0.5_4_15efbde2d1`:
  - **M**: + both flags, accumulation 1 (the t10 recipe), seeds 0 and 1;
  - **C2** (the recipe control): accumulation 2, no M, no hurdle, seeds 0 and 1: does accumulation alone revive the old head? Its
    seed 0 runs first; seed 1 is run whenever C2's seed 0 is not dead at step 1000 (its `motion@+0.0` below 1.040), because then C2
    enters the verdict and decide.py needs equal seed counts;
  - not combined with the hurdle (`object_moving`): both revive velocity, a run with both cannot attribute it;
  - **order, a sequential screen** (by hand: it decides only what runs next, never a verdict; the verdict needs both seeds): M s0,
    then C2 s0, both evaluated; seed 1 of M and of C2 only if M s0 beats C2 s0 on the headline by more than the band, or C2 s0 is dead
    at step 1000 (then C2 = control and only M s1 runs). Otherwise the screen stops: the record says "M s0 not better than C2 s0 (screen,
    no verdict)", and RECIPE PREFERRED needs C2 s1 and M s1 before it is a verdict (or stays labelled as the screen's outcome).
- **Controls.** `t10_dual_s0` / `s1` (matched recipe, same GT; the incumbent and t13's B0); v = 0 (null, the `centre/zero` rows); C2
  (recipe control); `t11_vel_hurdle_acc2` (the strong simple baseline: the object-head hurdle, ONE seed, a 1000-step warm start from
  `t10_dual_s0` with accumulation 2, so not a matched control; its seed twin is part of step 0 so that decide.py can pair it,
  launched from the pinned snapshot of the tag `bf/t11.vel_hurdle_acc2` (`run.py --no-git`, the same code as seed 0: a twin, not a
  re-implementation); the evaluation of both stays the live committed evaluator); the
  t12 block on the frozen base (lower bound, labelled: other definition). All measured by ONE command on the same frames:
  `bev_suite.py RUN --velocity eval --n 1000` (metric `vel` of `evaluation/bev_velocity.py`, built 2026-11-01 with response rows,
  mutants and a code critic pass; the frames spread deterministically over the val scenes, the frame hash on the card, `compare
  --velocity` refuses other frames or another GT signature); maps / front mAP2 / thresholded IoU are the `seg` / `det` numbers of
  the same evaluation. Both controls are RE-SCORED with the final committed evaluator (step 0), never read from older stores.
- **Recipe differences of M that are part of the arm (named, not hidden).** (i) Supervision: M's motion group is the MEAN of its level
  terms (finest + the coarser ones, `losses.py`), so the group keeps the weight of the controls' one head under the Kendall weights;
  the deep supervision itself (the coarser levels see the same velocity target) is the difference. (A sum would have given M about 3x
  the controls' velocity supervision on the shared camera BEV while Kendall s moves slowly: the control's s.motion is -0.07 at step 580,
  -0.17 at the end.) (ii) The field is not bounded (the controls' VelocityHead caps it at 40 tanh(v / 40)); the trainer's non-finite
  guard and the kill (b) cover a blow-up. (iii) The 1 x 1 velocity head of the time mover is built and then removed (it draws from the
  RNG: the flags-off path is untouched, the M arm's initial weights differ from the control's beyond the new block). (iv) M starts from
  scratch: `warm_start` from an existing checkpoint refuses an M model (the `mover.motion.*` keys), by design for these arms.
- **Definitions (the suite's, for every arm; t12's own definition differs and its numbers are not compared with these).** VAL,
  `eval --n 1000`, dt 0. Moving = GT speed > 0.5 m/s (`MOVING_SPEED`); cells and boxes of unknown velocity left out. Headline: field
  EPE on the moving vehicle cells of the camera view, `vel/dt+0.00/cells/vehicle_moving/epe` (vehicles: 89 % of the moving cells in
  t12, 33564 of 37901; the all-class `curve/dt+0.00/velocity/moving/epe` reported next to it). Second: EPE at the GT centres of moving
  boxes, `vel/dt+0.00/centre/{field,objhead}/moving/epe` (with the box read on, objhead = field; `objhead/all` is not comparable
  across hurdle and non-hurdle models). Guards: the thresholded vehicle IoU at -0.5 / +0.5 s, `seg_obj/seg/dt-0.50/vehicle/all/all/iou@0`
  and `dt+0.50` (p > 0.5; NOT the loss term `sem.obj_iou`), static EPE `vel/dt+0.00/cells/static/epe`, maps (mean IoU of `seg_fov`)
  and front mAP2 (`det`). Cost: block ms per frame (idle GPU, batch 1), parameters, peak GPU memory.
- **Label of every velocity number.** The GT is autogt v5: its ego velocity is halved at the first and last keyframe of a sequence
  (fixed in v6, `autogt/versions.csv`). All arms use the same v5 GT, so the comparisons hold; the absolute EPE carries that error at
  sequence ends.
- **Step 0 (before any M run; evaluation and checks only).**
  1. `--velocity eval --n 1000` of `t10_dual_s0`, `t10_dual_s1`, `t11_vel_hurdle_acc2` and its seed twin, with the final committed
     evaluator; their numbers go into a table in section 2 (from the stores) BEFORE any M run.
  2. Bands, from the paired scene bootstrap of `bev_suite.py compare --velocity` (and `compare` for maps / mAP2), never from the
     formula max(floor, 2 |s0 - s1|) alone (it gives walkway 0.029 and front mAP2 0.003, too narrow; the t2-era 0.054 was called too
     wide). Per metric: band = max(floor, the 95 % half-width of the paired scene bootstrap of the control pair `t10_dual_s0` vs `s1`,
     |s0 - s1| of the controls on that metric) - the bootstrap measures scene sampling, |s0 - s1| the training noise; floors: headline
     and centre 0.10 m/s, static EPE 0.02 m/s, maps 0.010, mAP2 0.010. Written into section 2 with the bootstrap command.
  3. Control parity: the M config with both flags OFF, seed 0, 40 steps on the GPU, against `runs/t10_dual_s0/log.jsonl` (it logs
     every 20 steps: steps 1, 20, 40): step 1 within 3e-3 relative (same seed and first batch; the GPU scatter noise is up to 3e-3,
     `training/architecture.md`), steps 20 and 40 within 3e-2 as a drift check; plus the exact CPU equivalence of the flags-off model
     against the `t10_dual_s0` checkpoint (`tests/equivalence.py`, max abs diff 0). Else the code or the data path drifted and nothing
     is run.
  4. GT pinned: `runs/<arm>/gt.json` names `0.5_4_15efbde2d1` and the evaluation store's source names the same signature (compare
     refuses otherwise); the run folder holds the copy of autogt v5 (the BEV system chat's side at launch).
  5. Smoke run (20 steps through run.py, M config): every tensor of `motion_block` has a non-zero gradient and a non-zero update at
     step 20 (the Training chat's `probe.py` if it has landed, else a one-off check on the smoke checkpoint), not only plumbing
     (smoke.d32 checked plumbing only).
- **Prediction** (relative to step 0, written before it). Headline <= 0.5 x the control mean on both M seeds (t12 halved it on frozen
  features: 6.91 -> 3.40 in its definition); centre EPE (field) <= the hurdle object head minus 0.1 m/s; M better than C2 on the
  headline by more than the band; thresholded vehicle IoU at +-0.5 s >= control - 0.02; static EPE <= control mean + 0.03; maps and
  front mAP2 inside their bands; block <= 5 ms.
- **Decision rule** (bands from step 0.2; the typed calls below):
  - KEEP, label "M needed" (M joins the incumbent chain; t13 takes its B0+M row): the headline is below the control by more than
    the band on BOTH seeds AND below C2 by more than the band (or C2 is dead at step 1000 and therefore = control) AND the centre EPE
    of the field is below the hurdle object head by more than the band AND every guard holds on both seeds (thresholded vehicle IoU
    at +-0.5 s >= control - 0.02, static EPE <= control mean + max(0.03, its band), maps and front mAP2 >= control mean - band) AND the
    block costs <= 10 ms per frame;
  - M beats the control but not C2 (inside the band or worse): RECIPE PREFERRED, label "M beats a dead head, the recipe revives the old
    head as well": M is not adopted at this recipe. C2 costs twice the training time per step but nothing at inference, and the
    product goal is speed at inference, so at an equal effect C2 wins; if C2 beats M by more than the band, M is CUT. M at
    accumulation 2 against C2 is a new card;
  - M beats the control and C2 but not the hurdle: HURDLE PREFERRED (cheaper); M on top of the hurdle is a new card;
  - the box read (`box_vel_from_field`) is kept only if the centre EPE of the field is below M's own object head by more than the
    band (M without the read, if needed, is a variant row);
  - velocity better but a guard fails: PARTIAL, no adoption; the failing guard names the next arm (e.g. IoU: a sharper mover, not a
    bigger block);
  - inside the band against the control: TIE (the control is cheaper); worse: CUT.
- **Typed calls** (decide.py; `--lower-better` on every EPE row; equal seed counts, otherwise decide.py treats them as unpaired and as
  one seed, so "add seed 2" means seed 2 of the control too). Run id t15.M; numbers from the stores, the band from step 0.2:
  - `AR_PROJECT=../bev_fusion python3 ../autoresearch/tools/decide.py --id t15.M --variant <M_s0>,<M_s1> --control <t10_s0>,<t10_s1> --band <b_head> --lower-better --cost-variant <ms_M> --cost-control <ms_t10> --write` (headline vs control);
  - the same with `--control <C2_s0>,<C2_s1>` and `--id t15.M_vs_C2` (only when C2 is alive; else the record says "C2 dead at step 1000");
  - the same on `centre/field/moving/epe` with `--control <hurdle_s0>,<hurdle_s1>` and `--id t15.M_vs_hurdle`;
  - one call per guard (vehicle IoU at -0.5 and +0.5 without `--lower-better`, static EPE with it, maps, mAP2), `--id t15.M_guard_<name>`;
  - the composition: KEEP "M needed" only if the first three say KEEP (or C2 is recorded dead) and no guard says CUT; the ledger note
    names every call and its verdict.
- **Kill.** (a) Liveness at step 1000 (the velocity branches were dead in 28 of 30 batch-3 runs): the in-training validation's
  `motion@+0.0` (the finest level's motion term; `group.motion` of M also sums the coarser levels, so it is not comparable) at step
  1000 above 1.040 -> stop the run. Basis: `t10_dual_s0` / `s1` had 1.0945 / 1.0945 at step 1000 (their `log.jsonl`, val
  `group.motion` = their only motion term; the dead level, flat to 3000); 1.040 = 5 % below it; t12's block cut the motion term by
  45 % on frozen features. Stop by trainsuite (exit 75), write the cause; the Training chat's `probe.py` (gradient norm and update ratio
  per module) when it lands. (b) Non-finite loss in more than 1 % of the steps. (c) The flags-off equivalence or the 20-step parity
  (step 0.3) fails.
- **Budget.** Code + CPU tests about 3 - 4 h (new files and `timed_head.py` first; the `model.py` / `losses.py` / `interface.py` hunks
  as HEAD + own hunks through a temporary index; CPU tests only with 4 GB RAM free). GPU: step 0 (four evaluations of about 1542 s each,
  the `t10.dual_s1.eval` row; the hurdle seed twin about 85 min; the 20-step parity and the smoke run about 15 min); idle timing about
  15 min; M: 2 seeds of about 2 - 2.5 h (t10_dual took 1.9 / 2.2 h; the block adds roughly 10 - 20 % per step); C2: 2 seeds of about
  4 h (accumulation 2 doubles the samples per step); evaluation 1542 s per arm and seed. Totals: the screen branch (M s0 + C2 s0 + their
  2 evaluations, after step 0) about 7 h of training + 2 evaluations; the full design about 18 - 20 GPU hours (one training at a time
  on this box: about a day).
- **Runtime cut** (before the runs, each lever checked on CPU for equal outputs or on one VAL subset for the centre EPE): reuse the
  radar encoder's raster; drop the stride-1 level (`motion_strides (4, 2)`, the stride-2 field upsampled); fp16 / channels_last; no
  per-sample Python loops.
- **What it can and cannot show.** A KEEP adopts M in the `t10_dual` recipe at 3000 steps; the long recipe and the bundle with other
  blocks are t13's. Not tested: the memory gate and the window velocity of the camera temporal fusion (unchanged), warm sequences.

## 2 Step-0 numbers (to be filled from the stores before any M run)

-
