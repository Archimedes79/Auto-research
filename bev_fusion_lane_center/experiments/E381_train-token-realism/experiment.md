# E381 - train the head on tokens that look like the ones it will see: the TRAIN-to-VAL quality gap of B''s tokens as training-time noise

**Status: DRAFT, not run, no critic yet** (written 2026-11-17 20:15 by the lane chat on the lead's order of ~19:25; ranking in `E379_next_candidates.md`). The cheapest of the three drafts: a training-time input perturbation, nothing at inference.

## 0. The card (what the tools run)
```json card
{
 "id": "E381",
 "card": "",
 "title": "E381 token realism: the champion recipe (E365 head, none weight 0.25, 12000 steps) trained on the cached B' tokens with training-time input noise that closes the measured TRAIN-to-VAL gap of the token set (position jitter sigma_pos, existence-logit noise sigma_e, token drop 0.03; labels unchanged), fresh head seeds 2 / 3, read at its own TRAIN-chosen per-class cell against the champion (E373 heads at D115)",
 "kind": "variant",
 "deterministic": false,
 "metric": "topo_F1",
 "direction": "higher",
 "band": 0.01,
 "prediction": {"value": 0.003, "vs": "ref"},
 "arms": {
  "ref": {"recipe": "the champion on the same seeds: the E373 fresh heads `runs/e373_cb_s0 / s1` (cache seed 0 / 1, head seed 2 / 3) at the D115 cell: VAL topo F1 0.6679 / 0.6762, F1@1.0 0.8459 / 0.8467, lanes found 0.7191 / 0.7205, junction F1 0.2517 / 0.2452, dup_frac 0.0607 / 0.0637"},
  "n": {"recipe": "`jtoken_train.py --name e381_n_s<S> --seed S --head-seed H --ident e292_a --steps 12000 --bs 8 --lr 1e-3 --workers 4 --none-weight 0.25 --noise-pos SP --noise-e SE --token-drop 0.03` (H = 2 / 3 as E373: the same initialisation and batch order as the ref head of the seed). The two new flags (default 0 = the champion's loss and 20 smoke steps bit for bit, a CPU test) perturb the head's INPUTS per step: `pos += N(0, SP^2)` per axis on the valid tokens, `e` and `e_raw += N(0, SE^2)` (the same draw for both), the noise drawn from a generator seeded by (seed, step) so that a resumed run repeats it; `--token-drop 0.03` is the existing per-step drop of the trainer (`step_keep` / `drop_tokens`); the labels (type, pair, ylong, yscale, gt_id) are those of the clean tokens, the geometry the head computes (`pair_geometry`) is that of the perturbed positions; the values SP and SE are fixed in section 1 BEFORE the run by the CPU calibration (`temp/e379/calibrate_noise.py`, to be written: a bisection on the TRAIN every-3 sample)", "run_ids": ["E381.n_s0", "E381.n_s1"]}
 },
 "control": "ref",
 "variant": "n",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "topo_F1", "vs": "ref", "min_effect": 0.005},
  {"metric": "junction_F1", "vs": "ref", "max_loss": 0.01},
  {"metric": "F1@1.0", "vs": "ref", "max_loss": 0.005},
  {"metric": "lanes_found", "vs": "ref", "max_loss": 0.01},
  {"metric": "dup_frac", "vs": "ref", "max_loss": 0.01}
 ],
 "budget_gpu_h": 1.4,
 "kill": "a loss is not finite; or no learning (the mean loss over steps 1801 - 2000 is not below 0.8 x the mean over steps 101 - 200); or the noise is inactive or wrong at run time (the trainer logs, at step 1 and every 100 steps, the mean displacement of the valid tokens (m) and the mean |noise| on e: both within 0.8 - 1.2 x the calibration's values); or the option changes the default (CPU test: the two flags at 0 and `--token-drop` as the champion gives the champion's loss and 20 smoke steps bit for bit); or the noise reaches the labels or the read-out (a test: the label tensors of a batch are equal with and without the noise, and `jtoken_eval.py dump` takes no noise flag)",
 "evaluate": {"split": "val"}
}
```
Verdict by hand (written before any run; the JSON clauses mirror it): **ADOPT-NOISE** = on BOTH seeds at the (b) cell (the pinned (b) rule of E377: E375's per-class grid on the arm's own TRAIN every-3 table, dup_frac guard 0.05, ties to D115): n - ref topo F1 >= +0.005 with the lower end of the paired 95 % interval above 0; junction F1 not lower by more than 0.010; F1@1.0 -0.005; lanes found -0.010; dup_frac +0.010; the D113 floors and E373's tie conditions; the kill did not fire. **NO** = n - ref <= -0.005 on both seeds or a guard broken on both. **TIE** otherwise: the champion stays. The noise floor (head against head) is +-0.003 (E378's noise reference): a +0.005 bar is above it. Read-outs as E379 (a) D115, (b) the pinned rule, (c) argmax, `anatomy`; additionally the existence AP of the head on VAL, because the mechanism is a calibration claim.

## 1. Card text
- **Statement:** a head trained on tokens perturbed to the quality of B''s VAL tokens generalises better to VAL than the head trained on B''s cleaner TRAIN tokens: +0.005 topo F1.
- **Mechanism, and the measured failure it attacks:** B' (the lane head whose node head makes the tokens) was trained on the TRAIN frames, so its tokens on TRAIN are better than on any new frame, and the head (the second stage) learns to trust positions and existence logits as precise as they are on TRAIN: the classic shift of a stacked model trained on its first stage's in-sample outputs. MEASURED today (CPU, scratch `temp/e379/gap_tokens.py`, the predicted-node caches of B' seed 0 / seed 1: TRAIN every-10th frame = 842 frames against VAL 873): the existence AP of sigmoid(e) for 'assigned to a GT node' 0.9607 / 0.9592 on TRAIN against 0.9269 / 0.9253 on VAL (-0.034); the share of assigned tokens with p < 0.5 0.137 / 0.143 against 0.160 / 0.161; the share of unassigned tokens with p >= 0.5 0.059 / 0.061 against 0.083 / 0.087; the GT-node coverage of the token set 0.981 / 0.982 against 0.952 / 0.954 (-0.029); the mean distance of an assigned token to its GT node 0.380 / 0.384 m against 0.448 / 0.450 m; the mean existence probability of assigned and of unassigned tokens is the same (0.71 / 0.22 against 0.70 / 0.23): the VAL tokens are not mis-calibrated on average, they are noisier. The head's own TRAIN-to-VAL drop at the chosen cell is 0.05 - 0.06 topo F1 (E366: 0.7222 against 0.67): how much of it is B''s (the tokens the head is given) and how much is the head's own over-fit is not known; the plain B read-out has the same kind of gap. The card tests the part the head can still use.
- **What is perturbed and how much (fixed BEFORE the run; the calibration is label-based on B''s token set only, no head score is read):** (i) position jitter: sigma_pos such that the mean distance of an assigned token to its GT node on the TRAIN every-3 sample reaches the VAL value (0.448 / 0.450 m from 0.380 / 0.384 m: a first estimate sqrt(0.448^2 - 0.380^2) = 0.24 m per axis-free radius, to be bisected on the real distance statistic); (ii) existence-logit noise: sigma_e such that the TRAIN-sample existence AP of the noisy logits equals the VAL AP (0.927 from 0.961; bisection); (iii) token drop 0.03 = the coverage gap (0.981 - 0.952 = 0.029 of the GT nodes have no token on VAL). The numbers SP / SE are written into this card by the calibration script before the first run and not changed afterwards. **Disclosure (a limit of the card):** the targets of the calibration (the VAL AP, the VAL mean distance, the VAL coverage) are VAL statistics of B''s token set: the two noise levels are fitted on VAL properties of B', not of the head, and the verdict is read on VAL: it is in this sense 'tuned on VAL' in two scalars; the confirmation on TEST (Mirko's recorded OK, E343) is the clean read; the head's own VAL scores play no part in the choice.
- **E2E:** yes: training-time input noise; nothing is selected by a hard step; inference is the champion's.
- **Cost:** training as the champion (41 min per head, the noise is a few elementwise ops per step), 2 heads = 1.4 GPU h at the champion's measured 2451 s per head on the shared GPU (0.7 h at the 0.108 s per step of its last 100 steps on an idle GPU; the launch gate measures it), dumps 5 min each, the CPU sweeps 25 min each; budget 1.4 GPU h. Inference: unchanged.
- **Prediction (+0.003 on topo F1, window -0.005 to +0.010; junction F1 +0.005):** the shift is real (AP -0.034, coverage -0.029) but the head's TRAIN-to-VAL gap of 0.055 is mostly B''s (the tokens are worse), which noise on the inputs cannot repair; what noise can repair is over-trust; the effect is bounded by the share of the gap that is over-trust: a guess of a few thousandths. **Outcome odds: ADOPT about 15 %, TIE about 65 %, NO about 20 %** (a noisier input may also cost: the head loses the exact positions the pair geometry needs).
- **What would make it fail:** the gap is B''s, not over-trust (then the cure is S4: B' trained on a loss that sees the second stage, or tokens from a B' that did not train on the frame: an out-of-fold B', a different card); jitter blurs the pair geometry the long links depend on (the junction guard catches it); the noise levels are right on average but the VAL noise is structured (missed whole lanes, not Gaussian jitter): the token drop covers a part.
- **Inputs:** ready: the caches, the trainer (`--token-drop` exists), `gap_tokens.py`, the E375 grid. To build: the two trainer flags and their tests (Fable's `jtoken_train.py`), the calibration script (a bisection on two statistics of the TRAIN every-3 cache: scratch to be promoted at its second use).
- **Starts from:** the E373 champion recipe, B' frozen.
- **Lens / group / sources / status:** stacking shift (a second stage trained on first-stage outputs the first stage has seen); today's `gap_tokens.py` measurement; status draft.
- **Referee:** none yet.
