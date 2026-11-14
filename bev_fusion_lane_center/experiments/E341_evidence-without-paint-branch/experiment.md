# E341 - the evidence along the curve without the 0.5 m paint maps: the maps head loses its 0.5 m branch (-1.48 GMACs, -18.6 % of the system) at the SAME width (compute arm, D92 / D86; the saving in MACs is large, the saving in ms per frame is an estimate of about 1.3 of 17.1 ms and is measured in part B of E342)

Card of 2026-11-14 16:00 (revised 16:50 after the critic), written by the lane chat on Mirko's order of 2026-11-14 ("Schraenke nicht die Breite ein aber gucke, wie man sonst Rechenzeit sparen kann", D92). The 0.5 m paint branch of the junction maps head costs 1.477 of its 3.256 GMACs (45 % of the head, 18.6 % of the system; E330 section 1) and has exactly one reader: the evidence along the curve (two of its six inputs, the maxima of the stop-line and crossing maps along the Hermite curve of a pair). E332 arm a removed the branch AND the whole evidence (the code needed `--no-evidence` with `--no-paint`) and lost -0.0189 / -0.0106 topo F1 on the differentiable read-out; its own reading: **the loss is the evidence, not the paint branch** (arm a against E331, evidence off on the full maps: +0.0043 / +0.0019, within noise). So the question that was never asked: **how much of the evidence loss comes from the two paint inputs, and how much from the four others** (lane heat of the pyramid mean / min, zone mean, core mean, all read from the 1 m layer and the pyramid, none needs the 0.5 m branch)? This arm keeps the four, drops the two, and trains the pair stage of E312 on the E332 arm-a maps (`--no-paint`, zone / core still supervised), so the head runs at 1.779 GMACs with the width unchanged. The delivered path stays the same differentiable network; only an input of the pair MLP and a branch go.

## 0. The card (what the tools run)
```json card
{
 "id": "E341",
 "card": "",
 "title": "pair stage of the E312 recipe (`jends_train.py --e2e --scales --maps-w 1.0 --steps 3000`, evidence ON) on the maps head of E332 arm a (`runs/e332_ma_s<S>`: no 0.5 m branch, width 64) with the evidence along the curve reduced to its four non-paint inputs, read out unsnapped and snapped at p >= 0.1, against E312 of the same seed",
 "kind": "block",
 "deterministic": false,
 "metric": "topo_F1",
 "direction": "higher",
 "band": 0.01,
 "prediction": {"value": -0.008, "vs": "c"},
 "arms": {
  "c": {"recipe": "control = E312 of the same seed (maps head 3.256 GMACs, six evidence inputs): unsnapped p >= 0.1 topo F1 0.6438 / 0.6481, snapped 0.6825 / 0.6864; also reported against E332 arm a (the same maps head, no evidence: unsnapped 0.6249 / 0.6376) and E331 (full maps, no evidence: 0.6206 / 0.6357)"},
  "w": {"recipe": "pair stage: `jends_train.py --name e341_p_s<S> --maps runs/e332_ma_s<S>/model.pt --e2e --scales --maps-w 1.0 --steps 3000 --workers 3` (run E341.p_s<S>, evidence ON: the maps checkpoint has ctx `dil+nopaint`, so JEndModel builds the pair head with 4 evidence inputs), BFLC_SIG=1.0"}
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
 "budget_gpu_h": 3.2,
 "kill": "the pair loss is not finite, or `runs/e332_ma_s<S>/model.pt` is missing, or the head the run builds has not 150 pair inputs (`pair.in_features`: 152 minus the two paint columns; the strict load of the dump stage and `grid/test_jmaps_simplify.py` check it); stage 1 itself passed its kill check in E332 (maps keypoint F1 0.7409 / 0.7355 against E307 0.7233 / 0.7051)",
 "evaluate": {"split": "val"}
}
```
By hand (label "by hand: replay", as E331 / E332): after the pair stage the chain runs the dump and the scores (`jends_eval.py score --thr 0.1` unsnapped, `--snap --thr 0.1` snapped), then `evaluation/jrule_removal.py --arm e341_p_s --control e312_p_s`. **Adopted** when the arm loses at most 0.01 topo F1 against E312 of the same seed on BOTH seeds and BOTH read-outs, unsnapped (the differentiable read-out, D85: counts) and snapped (the rule step: reported, must hold), with the guards held; a TIE inside the band keeps the cheaper variant, which is this one (-1.48 GMACs). **Control and D94:** the control stays E312 (the rule was written for it before the run); D94 makes E324 s the system of record, which is about 0.015 higher snapped, so ADOPT here means "adopted on the E312 recipe"; a transfer to the E324 s recipe needs its own run (a card after an ADOPT here), and this result is also read against E324 s afterwards.
**Intervals (the rule stays the point estimate, D85):** against E332 arm a (same maps, no evidence) on both read-outs: `jdiag.py boot e341_p_s<S> ends_card_thr0.1 e332_pa_s<S> ends_card_thr0.1` (unsnapped) and `... ends_card_snap_thr0.1 ... ends_card_snap_thr0.1` (snapped); the chain runs them. Against E312 the snapped read-out has a counts file (`runs/e312_p_s<S>/ends_card_snap_thr0.1.counts.pkl`); for the unsnapped one, which counts under D85, E312 has none, so the chain first re-scores E312 unsnapped with a TAGGED card (`jends_eval.py score --run e312_p_s<S> --seed <S> --thr 0.1 --tag cnt`: a new file `ends_card_cnt_thr0.1.json` with its counts, a stored card is never overwritten) and then runs `jdiag.py boot e341_p_s<S> ends_card_thr0.1 e312_p_s<S> ends_card_cnt_thr0.1`.
The code (committed with this card): `JEnds(paint_ev=False)` drops the two paint maxima (`evidence` columns 4 and 5, `NEV - 2` inputs); `JEndModel` sets it from the maps ctx (`nopaint` in it); the default is unchanged, so every stored checkpoint and the E312 command line build the same network (selftest: the input of the pair MLP without the paint maps is the one with them minus the two paint columns, column by column, same tokens).

## 1. Prediction and cost
**Prediction (a number):** topo F1 -0.008 against E312 of the same seed on the unsnapped read-out (the middle between E312 = 0 and E332 arm a -0.0189 / -0.0106 unsnapped; the four inputs that stay are the lane heat of the pyramid and the junction zone / core of the 1 m layer, i.e. whether the curve runs along a lane and inside a junction; an assumption, not a measurement), snapped -0.003. Honest risk: the stop-line and crossing maxima are the only inputs that say "a painted junction mouth lies on this curve", and E331 / E332 found the whole evidence worth 0.012 - 0.023 unsnapped; the arm may lose most of it and fail the bound of 0.01 on seed 0 (a loss near -0.012 would be NOT adopted). **Power:** the prediction is 0.002 inside the bound; with a paired spread of 0.004 - 0.006 per seed (E312's two seeds differ by 0.0044; the E331 / E332 seed effects differ by about 2 times; the critic's estimate, not measured) a result equal to the prediction passes both seeds on both read-outs only about 40 % of the time; a TIE or NOT adopted at -0.011 is inside the noise, and the paired difference to E332 arm a is the number that is worth reading. E331 / E332 also showed that the loss sits at the fixed operating point p >= 0.1 (pair precision at E312's recall was higher), so a calibration shift of the pair logits can decide the verdict, not the ranking. Cost: no maps stage (E332's `e332_ma_s<S>` are the maps; their stage-1 numbers are in the E332 card); the pair stage 1.15 - 1.6 h per seed, post-processing 0.25 h: about 3.0 h wall for both seeds one after the other (D98: by day one GPU job; E332's pair runs took 70 - 74 minutes each, the post-processing 0.25 h per seed), 3.2 slot hours. Start (not started): `temp/e341/chain.sh`, called by `temp/day18/day18.sh` once `temp/e341/READY` and `ACCEPTED` exist.
**What it saves when adopted:** the 0.5 m branch (`m05`, `head05`: 1.477 GMACs, 0.037 M parameters), its loss and the two paint inputs of the pair MLP; width, depth and resolution of everything else unchanged. In ms per frame that is an estimate of about 1.3 ms of the 17.09 ms forward (E342: the maps head takes 2.82 ms for 3.256 GMACs; E332 did not measure ms): the E341 checkpoint (maps head and whole forward) is timed in part B of E342 (section 2 there names it). NOT saved yet: the loader still builds the paint layer (`batch["jpaint"]`, read unconditionally by `jmaps.py` and `jmaps_eval.py`) and the sidecars of `data/prep/junction_paint.py` stay needed; skipping them is a follow-up after an adoption (the E332 card says the same). D86: simple (one branch less), little compute (-18.6 % of the system's MACs), E2E (every module still gets the gradient of the pair loss; no new step).

## 2. Referee
Fresh `ar-critic` before the GPU run (card and the diff of `jends.py` / `jends_train.py`).
