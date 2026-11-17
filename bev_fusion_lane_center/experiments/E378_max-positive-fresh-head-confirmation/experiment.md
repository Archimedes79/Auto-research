# E378 - confirmation of E376's side finding on two FRESH head seeds: does the max-positive likelihood (`--pos-loss max`) raise the champion's topo F1 at its own decode cell, with a rule written before the fresh heads exist?

Card of 2026-11-17 (system clock), **version 1 at 17:05, before the critic pass**; written by the Spurmitte chat that Mirko addresses as "Fable" on the lane lead's order of 16:55 (D86): "do NOT adopt it from this table (it was seen on the selection heads after the verdict: a forking path), CONFIRM it with a rule written BEFORE the fresh seeds exist: E378 = the champion recipe (b_bal25) with --pos-loss max on TWO FRESH head seeds (4 and 5, caches 0 / 1 as E373), the cell chosen by the E375 rule on each head's own TRAIN sample (or the D115 cell if the rule gives it), the control = the E373 heads at D115 with the cross-run paired interval ... ADOPT-LOSS = on BOTH fresh heads topo F1 >= +0.005 with the lower interval end above 0, junction F1 not lower than the champion, F1@1.0 not lower by > 0.005, lanes found not lower by > 0.01; NO otherwise; a prediction with a number (topo F1 +0.006, window 0 to +0.012). Cost is the same (loss only), so a tie in quality would take the plain recipe (D106)". D113 / D109 / D55 (E2E differentiable only, no rule step), the six hard rules. TEST never touched.

**Where the finding comes from (E376, result f21e39931f, VAL 873, the D115 cell, mx minus the E373 head of the SAME cache and head seed):** topo F1 +0.0055 [+0.001, +0.011] / +0.0110 [+0.006, +0.017], lanes found +0.010 / +0.011, F1@1.0 +0.005 / +0.004, junction F1 +0.016 / +0.008, dup_frac +0.009 / +0.006; found AFTER E376's verdict on its own heads (E376 asked about the junction F1): a forking path, hence this confirmation on heads that do not exist yet. **Noise reference** (the champion recipe, another head seed, the same cell: E367 b25_s0 minus E373 cb_s0): topo F1 +0.003 [-0.002, +0.008], junction F1 +0.006 [-0.004, +0.018], lanes found +0.003: the fresh heads of this card are NOT seed twins of the E373 heads (head seeds 4 / 5 against 2 / 3), so the comparison against them carries a head-seed spread of about 0.003 in topo F1; clause (ii) below separates the loss from the head seed.

## 0. The card (what the tools run)
```json card
{
 "id": "E378",
 "card": "",
 "title": "E378 max-positive confirmation: the champion recipe (E365 head, none weight 0.25, 12000 steps) with --pos-loss max on two fresh head seeds (cache 0 / head 4, cache 1 / head 5), each read at its own TRAIN-chosen per-class cell against the E373 champion heads at the D115 cell; the seed-twin control c (the same fresh head seeds, --pos-loss sum) only if the arm passes clause (i)",
 "kind": "variant",
 "deterministic": false,
 "metric": "topo_F1",
 "direction": "higher",
 "band": 0.005,
 "prediction": {"value": 0.006, "vs": "ref"},
 "arms": {
  "ref": {"recipe": "the champion: the E373 fresh heads `runs/e373_cb_s0 / s1` (cache seed 0 / 1, head seed 2 / 3, --pos-loss sum) at the D115 cell (short 0.3 / long 0.2 / keep 0.6 / none gate off): VAL topo F1 0.6679 / 0.6762, F1@1.0 0.8459 / 0.8467, lanes found 0.7191 / 0.7205, junction F1 0.252 / 0.245, dup_frac 0.061 / 0.064"},
  "x": {"recipe": "`jtoken_train.py --name e378_x_s<S> --seed S --head-seed H --ident e292_a --steps 12000 --bs 8 --lr 1e-3 --workers 4 --none-weight 0.25 --pos-loss max` (S = 0 / 1, H = 4 / 5: fresh heads, never used before)", "run_ids": ["E378.x_s0", "E378.x_s1"]},
  "c": {"recipe": "the seed-twin control, trained ONLY if x passes clause (i) on both heads: `jtoken_train.py --name e378_c_s<S> --seed S --head-seed H ... --none-weight 0.25` (the champion recipe, --pos-loss sum, the same S / H as x: the same initialisation and batch order)", "run_ids": ["E378.c_s0", "E378.c_s1"]}
 },
 "control": "ref",
 "variant": "x",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "topo_F1", "vs": "ref", "min_effect": 0.005},
  {"metric": "junction_F1", "vs": "ref", "max_loss": 0.0},
  {"metric": "F1@1.0", "vs": "ref", "max_loss": 0.005},
  {"metric": "lanes_found", "vs": "ref", "max_loss": 0.01}
 ],
 "budget_gpu_h": 1.2,
 "kill": "a loss is not finite; or no learning (the mean of the logged total loss at steps 1800 - 2000 is not below 0.8 x the mean at steps 100 - 200); the option is pinned by test_jtoken.py (83fb010c: --pos-loss sum = the champion bit for bit, max acts only on non-empty G(i)); `--timeout` 55 min (E376's mx runs took 20 / 42 min); one job family on slot 2 with a supervisor and resume; x first (0.9 GPU h incl. dumps), c (0.6 GPU h incl. its VAL dumps) only on the ADOPT path",
 "evaluate": {"split": "val"}
}
```
**The read-outs (named before any run; the standard scorer as E365, all sections 0, no rule step, VAL 873, TEST never touched):** per fresh head of x: the VAL dump and the TRAIN-sample dump (291 frames, every 3), the per-class TRAIN sweep (36 cells), **its cell by the pinned rule** (written into this card before its VAL score), the VAL score at that cell with `score --ref-run e373_cb_s<S> --ref-run-cell 0.3,0.2,0.6,off` (the cross-run paired scene bootstrap, x minus the champion head of the same cache), and the D115 cell with `--ref-run` (reported); `anatomy` (reported). For c (only on the ADOPT path): the VAL dump, the paired score x minus c at x's cell of the same seed (`score --run e378_x_s<S> <x's cell> --ref-run e378_c_s<S>`). **The pinned cell rule (E375's, per head, as E376 applied it):** eligible: dup_frac <= 0.05 and lanes found >= (the head's own TRAIN table at the D115 cell) - 0.010; the highest topo F1 T; the window: topo F1 >= T - 0.005; in the window the highest junction F1; ties by |short - 0.3|, |long - 0.2|, |keep - 0.6| in that order, then the higher thresholds; no eligible cell: the D115 cell.
**The verdict (by hand: written before any run; the JSON clauses mirror (i)):** **ADOPT-LOSS** = (i) on BOTH fresh heads at their own cells against the champion at D115: topo F1 higher by at least 0.005 with the lower end of the paired 95 % interval above 0; junction F1 not lower (difference >= 0); F1@1.0 not lower by more than 0.005; lanes found not lower by more than 0.010; dup_frac not above by more than 0.010; the D113 gate floors and the E373 tie conditions hold; the kill did not fire; AND (ii) on both seeds x minus its seed twin c at x's cell: topo F1 difference > 0 (the gain is the loss, not the head seed). **NO** otherwise: the champion keeps its recipe (as delivered); when (i) fails, c is not trained and the verdict reads 'NO at (i), c not run'. **TIE** (reported, the plain recipe kept by D106: the same cost, a tie takes the plain one): (i) fails only on the topo F1 margin with both differences inside +-0.005. A replacement under ADOPT-LOSS changes the champion's recipe to `--pos-loss max` (same head, same decode cell unless the rule moved it, same cost) and goes into D114's champion line through the lead.
**Prediction (guesses):** topo F1 x - ref at the chosen cells **+0.006** (window 0 to +0.012), junction F1 +0.010, lanes found +0.010; the chosen cell = the D115 cell on both heads (as on both E376 heads); **outcome odds: ADOPT-LOSS about 35 %, NO about 50 %, TIE about 15 %.** Reasons: E376 showed +0.0055 / +0.011 on seed twins of the champion heads; on fresh heads the head-seed spread (about 0.003) adds noise to a +0.005 bar with a lower end above 0 on both heads, and the E376 seed-0 lower end was only +0.001.

## 1. Card text
- **Statement:** the max-positive likelihood on the champion recipe raises the topo F1 at the champion's decode cell by at least 0.005 on two fresh heads, without a loss in junction F1, F1@1.0 or lanes found.
- **Mechanism (as measured in E376, not assumed):** the max over the GT successor group concentrates the successor mass on one member (short sources with 2 or more links >= 0.3: 3.8 % -> 1.5 %); fewer duplicate parallel links per source make the delivered lane chains cleaner (topo F1, lanes found), while the junction false points (the kept near-node tokens: E377) stay.
- **E2E / D55:** a loss-only change: the same head, the same decode, the same named output thresholds; differentiable as before.
- **Not claimed:** the junction F1 lever (E376: NO; E377 is that question), TEST, anything about S2 / S3 / S4.
- **Inputs:** all ready: `--pos-loss` (83fb010c), the read-out tools (b280a87a, 699e9182), the E373 heads and dumps, the scratch chooser `temp/e376/choose376.py` (the same rule; to be promoted if this card adopts).
- **Starts from:** the champion recipe (E367 b_bal25 / E373), frozen B' tokens, K = 512, lane GT signature 9ad1d4ce30.
- **Scheduling:** slot 2 after E377.n_s1's training (about 17:15; the lead's D86 plan of 16:55: x_s1 may start after 17:30, a default, not a hard rule); c, if it runs, may wait for the night window (22:00) and shares the GPU with the lane chat's S4.
- **Lens / group / sources / status:** D113 / D116 (the wiring experiments, the lead's order of 16:55); E376's side finding; status proposed, awaiting the referee.
- **Referee:** pending (fresh `ar-critic`, cards, before any run).

## 2. Detail plan
| step | what exactly | where | done when | budget |
|---|---|---|---|---|
| train x | x_s0, x_s1 (cache 0 / 1, head 4 / 5), one family on slot 2, a supervisor and resume | `run.py --id E378.x_s<S> --gpu --require-card --timeout 55` | done flags | 0.7 h |
| read-out x | dumps (VAL, TRAIN every 3), the TRAIN sweep, the cell written here, the VAL score with `--ref-run`, D115, `anatomy` | `jtoken_eval.py` | cards | 0.2 GPU h + CPU |
| c (ADOPT path only) | c_s0, c_s1 with the same seeds, VAL dumps, x minus c at x's cell | `run.py --id E378.c_s<S>` | cards | 0.6 GPU h |
| decide | the verdict above | by hand: section 0 | result section, ledger notes, commit, one line to the lead | |
