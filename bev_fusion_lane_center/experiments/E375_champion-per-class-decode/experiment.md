# E375 - a per-class decode for the champion (separate link thresholds for short lane links and long connector links), chosen on the TRAIN sample with a junction tie-break, confirmed without training on the fresh heads

Card of 2026-11-17 (version 1, 08:15 system clock, before any sweep; written by the Spurmitte chat that Mirko addresses as "Fable" on the lane lead's order of 08:05: "the champion's honest weakness is junction F1 0.20 vs plain B's 0.27; attack it where the diagnosis points, the operating point, per class"). **Status: draft; the per-class decode option is being added by a coder; no critic yet; nothing run. CPU only, no GPU, no training.**

**The champion (E373, experiments/E373_b-bal5-confirmation-fresh-head-seeds section 3):** the E365 head trained with none weight 0.25, decode link p >= 0.2 (both classes), node p_none < 0.6, none gate off; heads: runs/e367_b25_s0 / s1 (selection) and runs/e373_cb_s0 / s1 (fresh, head seeds 2 / 3). VAL at its cell: topo F1 0.6698 / 0.6769 (fresh heads), F1@1.0 0.846 / 0.846, junction F1 0.203 / 0.197, dup_frac 0.058 / 0.061 (plain B: junction F1 0.272). **The question:** does a decode with ONE more threshold, a separate link threshold for the long (connector, chord > 8 m) class, raise the junction F1 by at least 0.03 without losing topo F1 or F1@1.0?

## 0. The card (what the tools run)
```json card
{
 "id": "E375",
 "card": "",
 "title": "The champion heads read with a per-class decode (link threshold short x link threshold long x keep threshold, none gate off) chosen on the TRAIN sample of the E367 b_bal25 heads by the E366 rule with a junction tie-break, applied once to VAL on the E367 heads and confirmed on the E373 fresh heads; no training",
 "kind": "variant",
 "deterministic": true,
 "metric": "junction_F1",
 "direction": "higher",
 "band": 0.01,
 "prediction": {"value": 0.03, "vs": "ref"},
 "arms": {
  "ref": {"recipe": "the champion's cell (link 0.2 for both classes / keep 0.6 / none gate off) on the same heads: E373 fresh heads VAL junction F1 0.203 / 0.197, topo F1 0.6698 / 0.6769, F1@1.0 0.8456 / 0.8461; the verdict is by hand (decide.py not used)"},
  "pc": {"recipe": "`jtoken_eval.py sweep` with the per-class grid on the TRAIN samples (291 frames, every third of the 873) of runs/e367_b25_s0 / s1 (new-format dumps exist), the cell chosen by the rule below, then `jtoken_eval.py score --link-thr <short> --link-thr-long <long> --keep-thr <keep> --no-none-gate` once on VAL for the E367 heads and once for the E373 fresh heads", "run_ids": ["E375.sweep_s0", "E375.sweep_s1", "E375.val_s0", "E375.val_s1", "E375.cf_s0", "E375.cf_s1"]}
 },
 "control": "ref",
 "variant": "pc",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "junction_F1", "vs": "ref", "min_effect": 0.03}
 ],
 "budget_gpu_h": 0.0,
 "kill": "a read-out is not finite; CPU only (niced, 2 threads), no GPU job, no training; a crashed read-out is re-run unchanged, never with another cell",
 "evaluate": {"split": "val"}
}
```
**The grid (fixed now):** link threshold short in {0.3, 0.2, 0.1}, link threshold long in {0.3, 0.2, 0.1, 0.05}, keep threshold in {0.5, 0.6, 0.7}, none gate off (the champion's) = 36 cells; the champion's cell (0.2, 0.2, 0.6) is one of them.
**The selection rule (written before the sweep; the E366 rule extended with a junction tie-break):** on the two TRAIN-sample tables of the E367 b_bal25 heads (cache 0 / 1), among the cells whose mean TRAIN dup_frac <= 0.05, take the cells whose mean TRAIN topo F1 is within 0.005 of the best such cell, and among those the cell with the highest mean TRAIN junction F1; ties to the cell nearest the champion's cell (short threshold, then long, then keep). One cell for the recipe; written into this card before any VAL score of it.
**The verdict (by hand, written before any run; the JSON rule is NOT the verdict):** **ADOPT-DECODE** = on BOTH E373 fresh heads (never used to choose anything), at the chosen cell against the champion's cell on the same head: topo F1 not lower by more than 0.005, F1@1.0 not lower by more than 0.005, AND junction F1 higher by at least 0.03; then the champion keeps its head and takes the per-class decode (one more tuned output threshold: a named output step, said plainly); **NO** otherwise (the champion's cell stays). The E367 heads' VAL read-out at the chosen cell is reported next to it (not the verdict: their TRAIN tables chose the cell). If the chosen cell IS the champion's cell, the verdict is NO by construction (reported: 'the per-class grid prefers the current cell').
**Reported:** the full 36-cell TRAIN tables (topo F1, junction F1, dup_frac, F1@1.0, lanes found per cell); the paired scene-bootstrap interval of the chosen cell against the champion's cell per fresh head (topo F1 and junction F1); precision / recall; dup_frac; vertices per 100 m; the long-class delivered share.
**Prediction (guesses, with windows):** the chosen cell has a long threshold BELOW the short one (70 %); junction F1 on the fresh heads +0.02 (window -0.01 to +0.06); ADOPT-DECODE about 30 % (the 0.03 bar is high: the junction F1 of the record decode, the most conservative cell, was 0.24 - 0.25, so the room above the champion's 0.20 inside this grid is about +0.05 at most, and topo F1 must hold at the same time).

## 1. Card text
- **Mechanism:** the champion's single link threshold treats short lane links and long connector links alike; long links are rare (about 2.5 % of the source-class rows) and their probabilities are lower (E365 diagnosis: 51 % delivered at the record decode); a class-specific threshold can trade lane-link recall and connector-link precision separately. **Not claimed:** that a decode can fix what the head does not know (the E365 diagnosis says the long-class probability mass on the GT group is low); anything about TEST, S2 / S3 / S4. **Cost:** CPU only: two sweeps of 36 cells x 291 frames (about 30 minutes each, niced), four VAL scores (about 8 minutes each).
- **Referee:** a short fresh ar-critic (cards) before the sweep; the code critic of the per-class option.

## 2. Detail plan
| step | what exactly | where | done when |
|---|---|---|---|
| option | `--link-thr-long` in `jtoken_eval.py` score / sweep / graphs (default = `--link-thr`: the current behaviour bit for bit) and a per-class grid for `sweep` (`--grid perclass`), tests | coder | tests green, committed |
| sweeps | the per-class grid on the TRAIN samples of runs/e367_b25_s0 / s1 (existing new-format dumps `token_train_every3.pkl`) | `run.py --id E375.sweep_s<S>` (CPU) | tables |
| choose | the rule above, by hand | Fable | the cell written here |
| VAL | the chosen cell once on VAL: the E367 heads (`E375.val_s<S>`), the E373 fresh heads (`E375.cf_s<S>`) | run.py (CPU) | cards |
| decide | the verdict above | by hand | result section, ledger notes, commit, one line to the lead |

## 3. Summary
(empty until the runs exist)
