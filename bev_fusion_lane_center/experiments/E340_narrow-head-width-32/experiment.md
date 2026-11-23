# E340 - a still narrower junction maps head, width 32 instead of 40 (simplification arm S4c, the follow-up that E333 names)

**Status 2026-11-14: WITHDRAWN, never run (D92: The owner does not want the width narrowed).** No code was written for it beyond the existing `--d` flag.

Card of 2026-11-14 12:40, written by the lane chat (D84 point 4: simplify; D86 weights: simple, little compute). E333 names this arm in its own card: "if it passes, width 32 (1.629 GMACs, -50 % of the head) is the follow-up card". **It runs only when E333 (width 40) is adopted** (rule executed by `temp/day17/day17.sh`, below); when E333 is not adopted the fallback E339 (width 40 taught by the wide head) runs instead. No new code: `jmaps_train.py --d 32` exists (E333's CPU smoke ran the flag at 40; 32 is a multiple of 8 for the group norms). The context blocks at 2 m and the 1 m branch shrink with the width; the 0.5 m paint branch has its own fixed width of 32 and is 1.108 of the 1.629 GMACs at d = 32: the width cut cannot go below about half of the head's MACs (E332 removes the branch, and failed through the evidence it feeds).

## 0. The card (what the tools run)
```json card
{
 "id": "E340",
 "card": "",
 "title": "junction maps head of width 32 (`jmaps_train.py --d 32`, 1.629 GMACs instead of 3.256), then the E312 pair stage on it, read out unsnapped and snapped at p >= 0.1, against E312 of the same seed; run only when E333 (width 40) is adopted",
 "kind": "block",
 "deterministic": false,
 "metric": "topo_F1",
 "direction": "higher",
 "band": 0.01,
 "prediction": {"value": -0.008, "vs": "c"},
 "arms": {
  "c": {"recipe": "control = E312 of the same seed (width 64, 3.256 GMACs): unsnapped p >= 0.1 topo F1 0.6438 / 0.6481, snapped 0.6825 / 0.6864"},
  "w": {"recipe": "maps: `jmaps_train.py --ctx dil --sigma 1.0 --steps 3000 --d 32` (run e340_m_s<S>), then `jends_train.py --e2e --scales --maps-w 1.0 --steps 3000` on it (run e340_p_s<S>, evidence on as E312), BFLC_SIG=1.0"}
 },
 "control": "c",
 "variant": "w",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "topo_F1", "vs": "c", "max_loss": 0.01},
  {"metric": "F1@1.0", "vs": "c", "max_loss": 0.01},
  {"metric": "dup_frac", "vs": "c", "max_loss": 0.01},
  {"metric": "vert100_dp", "vs": "c", "max_loss": 1.0},
  {"metric": "junction_F1", "vs": "c", "max_loss": 0.03}
 ],
 "budget_gpu_h": 4.8,
 "kill": "stage 1 only: the maps keypoint F1 more than 0.05 below E307 of the same seed on either seed (then the pair stage is not run); or the loss is not finite",
 "evaluate": {"split": "val"}
}
```
By hand (label "by hand: replay", as E333): `jmaps_eval.py score --run e340_m_s<S>`, the dump and the two scores after stage 2, then `evaluation/jrule_removal.py --arm e340_p_s --control e312_p_s`. **Adopted** when the arm loses at most 0.01 topo F1 against E312 of the same seed on BOTH seeds and BOTH read-outs, unsnapped (differentiable, counts) and snapped (rule step, must hold), with the guards held; a TIE inside the band keeps the cheaper variant. Compared with E333 only as a report (the saving 1.967 -> 1.629 GMACs is 0.338, 5 % of the system's 6.665).
**When it runs:** `temp/day17/day17.sh` waits for `temp/e333/E333_DONE` and reads the LAST line starting with `verdict:` of `temp/e333/rule.log`; `verdict: ADOPT ...` -> E340; anything else -> E339.

## 1. Prediction and cost
**Prediction (a number):** topo F1 -0.008 against E312 on both read-outs (E333's -0.004 at width 40, a larger loss at a head that has half of the channels in the context blocks; maps keypoint F1 -0.02 against E307). Honest risk: the narrow context loses the separation of crowded ends that E286 / E307 found in the dilated blocks. Cost: maps stage 0.8 h + pair stage 1.15 - 1.6 h + post 0.25 h: about 2.3 h wall, 4.6 slot hours. Start (not started): `temp/e340/chain.sh`, called by `temp/day17/day17.sh`; `temp/e340/READY` carries the chain.

## 2. Referee
A variant of E333 in one number (the width, written in E333's own card as its follow-up); the fresh critic of E339 is asked to look at this card and at the scheduler together with the second version of E339.
