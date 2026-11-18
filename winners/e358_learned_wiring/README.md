# E358 learned wiring head: a learned choice of WHICH piece an end joins (source + selftests, no weights)

A winner folder: the source of the system, small enough to hand over; **no weights**. Written by `winner.py` on 2026-11-19 from commit `a881051638` (the sources had no uncommitted changes; the patch file is committed in `experiments/E358_wiring-head-on-attached-ends/`).

## What it is

A 545-parameter head (15 features per candidate place, 0.12 MMACs per frame) that chooses **which piece an end joins**, in the end-to-end read-out of E324 s (attached ends, snap without geometry). It replaces the nearest-piece distance rule (rule step R1 of the gap table). It is trained by **imitation of a rule-derived label** on frozen records (TRAIN records of 8414 frames from a live pass of the lane head B `e273_p_s<S>` and the pair run `e324_s_s<S>`; label = the rule applied at the matched GT end; heavy-tailed jitter calibrated on the attached-end error; heads 0 - 2 and a distance-only ablation, plus the no-jitter head `wn`).

What stays hard and named: the argmax per end, the candidate gate (4 m, 45 degrees, at most 8 places) and the cover drop. **No gradient of the output reaches the head**: it is a learned selection, not an end-to-end trained one.

Status (card E358, `experiments/E358_wiring-head-on-attached-ends/experiment.md`): verdict **KEEP+** by the written rule (label "by hand: D87 replay"). **Not adopted as the system of record**, and **not in the token champion** (E372 / D120), where the wiring is the softmax of the last attention layer. The price is about +2.0 ms per frame on the host (quiet, torch with one thread; the GPU part is unchanged).

## Numbers to reproduce (VAL 873 frames, seed 0 / seed 1)

| end-to-end read-out | rule (E324 s PRIMARY) | learned head | difference | 95 % interval (scene bootstrap) |
|---|---|---|---|---|
| topo F1 | 0.6912 / 0.6948 | **0.6931 / 0.6964** | **+0.0019 / +0.0016** | [+0.0012, +0.0027] / [+0.0009, +0.0023] |

Guards held on both seeds (F1@1.0, lanes found, vertices per 100 m identical to the rule; duplicate fraction +0.0008 / +0.0013, junction F1 +0.0009 / +0.0013). Agreement with the oracle label on the real choices (7388 / 7504 ends): head 0.787 / 0.782 against the rule 0.764 / 0.748. Head seeds 1 and 2: within 0.0003 of head 0. Curves 11729 / 11880 and wired-at-both-ends 6210 / 6412 equal the rule's: only WHICH piece an end joins changed. Cost: 545 parameters, +2.02 / +2.06 ms per frame on the host (rule 4.27 / 4.29 ms, head 6.29 / 6.36 ms). Not claimed: TEST, the transfer to the snapped read-out (E350), the stack with the E360 layer.

## What is here

`code/` (paths relative to `bev_fusion_lane_center/`): `training/grid/jwire.py` (the replaceable choice of the wired piece and its zero-parameter twin of the rule), `training/grid/jwire_head.py` (the head `Wire`, its instances, training loss, read-out `wire_choice_head`), `evaluation/jwire_data.py` (the records; `--ends att` for attached ends), `training/train/jwire_train.py` (training: `run` / `cv`), `evaluation/jwire_check.py` (twin and live checks), `evaluation/jwire_verdict.py` (the verdict of the card, `--read-out primary`), `evaluation/jends_eval.py` and `evaluation/qends_eval.py` (the scorer, `score --wire twin|head`), and `experiments/E358_wiring-head-on-attached-ends/staged_diff.patch`.

**The staged scorer.** The scoring of E358 ran on staged copies of `qends_eval.py`, `jends_eval.py` and `jwire_head.py` (the originals were frozen by the digest of E343): the patch holds the 6 hunks (`--wire` together with `--attach --snap --snap-nogeom`; the provenance check of the head: its `ends`, seed and pair run must match the read-out). The staged copies were scratch and are gone; **the patch is the record**. The files in `code/` are the committed originals: apply the patch to them to get what ran (its headers name `b/stage/<file>`; point the `+++` lines at `evaluation/qends_eval.py`, `evaluation/jends_eval.py`, `training/grid/jwire_head.py`, then `git apply`; checked on 2026-11-19: applies cleanly to the files of this folder). The promotion of the hunks into `evaluation/` was left to after E343 (recorded deviation in the ledger note of E358).

## How to run

```
# run from bev_fusion_lane_center/; S = 0 or 1
bash tools/env/_py14.sh jwire_train.py run --seed S --pair-run e324_s_sS --ends att --name e358_w_sS --heads 3 --epochs 60 --workers 4   # about 27 min per seed (GPU live pass + head fits)
bash tools/env/_py14.sh jends_eval.py score --run e324_s_sS --seed S --thr 0.1 --attach --snap --snap-nogeom --wire head --wire-run e358_w_sS --tag e358w   # staged scorer (patch applied), CPU, about 16 min
bash tools/env/_py14.sh jwire_verdict.py --card E358 --control-prefix e324_s_s --head-prefix e358_w_s --tag e358w --read-out primary --tag-wn e358wn --prediction 0.001 --window 0.003
# selftests (CPU): jwire.py, jwire_head.py, jwire_data.py, jwire_check.py, jwire_verdict.py, jwire_train.py `selftest`
#   all six ran ok on 2026-11-19 on the committed files (545 parameters; twin = rule on 1200 comparisons)
```

## Weights

`runs/e358_w_s0` and `runs/e358_w_s1` (`model.pt`, `model_h1.pt`, `model_h2.pt`, `model_wn.pt`, `wire_agreement.json`), the pair runs `runs/e324_s_s0` / `s1` (the ends cards `ends_card_e358w{,1,2,n}_snap_nogeom_attach_thr0.1.json`), the lane head B `runs/e273_p_s0` / `s1`: in the repository runs, not in git.

`MANIFEST.json` lists every file with its sha256; `python reporter/winner.py verify winners/e358_learned_wiring` re-checks them and reports drift against the sources.
