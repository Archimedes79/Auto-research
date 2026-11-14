# E348 - the E341 removal on the system of record: the E324 s recipe on the maps head without its 0.5 m paint branch (D94: the transfer needs its own run)

Card of 2026-11-14 21:24, written by the lane experiments chat on the lane lead's order after E341 (ADOPT on the E312 recipe: `jrule_removal.py` exit 0; the card E341 says that a transfer to the E324 s recipe needs its own card; D94 made E324 s the system of record, about 0.015 snapped above E312). Queue position: after E317 - E319 (day20). Nothing is started; the arms start through `run.py` with this card. D92 is not touched (width 64 everywhere); D86: simple (one branch less), little compute (-1.477 GMACs, -18.6 % of the system's MACs, -0.037 M parameters; the ms are an estimate until E342 part B times the E341 checkpoint), E2E (every module keeps the gradient of the pair loss; no new step).

## 0. The card (what the tools run)
```json card
{
 "id": "E348",
 "card": "",
 "title": "pair stage of the E324 s recipe (`jends_train.py --e2e --scales --maps-w 1.0 --steps 6000 --geom-w 0.5 --attach`, evidence ON with its four non-paint inputs) on the maps head of E332 arm a (`runs/e332_ma_s<S>`: no 0.5 m branch, width 64), read out snapped and with the PRIMARY end-to-end read-out at p >= 0.1, against E324 s of the same seed",
 "kind": "block",
 "deterministic": false,
 "metric": "topo_F1",
 "direction": "higher",
 "band": 0.01,
 "prediction": {"value": -0.003, "vs": "c"},
 "arms": {
  "c": {"recipe": "control = E324 s of the same seed (maps head 3.256 GMACs, six evidence inputs; stored cards): snapped (own keypoints) topo F1 0.6981 / 0.7010, PRIMARY (attached ends + snap-nogeom wiring, the differentiable read-out) 0.6912 / 0.6948, dup_frac snapped 0.1781 / 0.1522, PRIMARY 0.1897 / 0.1592"},
  "w": {"recipe": "pair stage: `jends_train.py --name e348_w_s<S> --seed <S> --maps runs/e332_ma_s<S>/model.pt --e2e --scales --maps-w 1.0 --steps 6000 --geom-w 0.5 --attach --workers 3` with BFLC_SIG=1.0 (run E348.w_s<S>; the maps checkpoint has ctx `dil+nopaint`, so JEndModel builds the pair head with 4 evidence inputs as in E341), i.e. the command line of E324 arm s with the maps path of E341", "run_ids": ["E348.w_s0", "E348.w_s1"]}
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
 "budget_gpu_h": 5.5,
 "kill": "the pair loss is not finite, or `runs/e332_ma_s<S>/model.pt` is missing, or the head the run builds has not 150 pair inputs (checked as in E341), or the chain's dump stage does not hold all 873 VAL frames",
 "evaluate": {"split": "val"}
}
```
By hand (label "by hand: replay", as E341): after the pair stage the chain runs the dump and the scores (`jends_eval.py score --snap --thr 0.1` = snapped, `--attach --snap --snap-nogeom --thr 0.1` = PRIMARY, `--snap --snap-nogeom --thr 0.1` as context) and then the removal rule, `evaluation/jrule_removal.py --arm e348_w_s --control e324_s_s` with its read-out list extended by `primary` = `ends_card_snap_nogeom_attach_thr0.1.json` (a one-line change of `READOUTS` with its selftest line; a critic pass before the first run; E324 s has no stored unsnapped card, so the unsnapped read-out is not part of this rule: the differentiable read-out here is PRIMARY, D85). **Adopted (on the E324 s recipe)** when the arm loses at most 0.01 topo F1 against E324 s of the same seed on BOTH seeds and on BOTH read-outs (snapped and PRIMARY) with the guards held (F1@1.0 <= -0.01, dup_frac <= +0.01, vertices per 100 m <= +1.0, junction F1 <= -0.03); a TIE inside the band keeps the cheaper variant, which is this one; **not adopted** when a bound or a guard is missed on a seed. The control is E324 s of the same seed (D94); E312 and E341 are context only.

## 1. Design, prediction, power
- **Why a card of its own.** E341 removed the 0.5 m branch on the E312 recipe (3000 pair steps, snap and unsnapped read-outs: seed 0 +0.0005 / 0.0000, seed 1 -0.0060 / -0.0080; seed 1 sits 0.002 inside the bound). The E324 s recipe differs in three ways that interact with the removal: 6000 steps, the geometry loss (`--geom-w 0.5`) and the attachment (`--attach`, the attached ends read B's nodes and the keypoints). A removal that costs 0.004 on one recipe can cost more or less on another; D94 makes E324 s the control of every new arm.
- **Prediction (a number):** topo F1 -0.003 against E324 s of the same seed on the snapped read-out and -0.004 on PRIMARY (E341 on E312: mean over the seeds -0.0028 snapped, -0.0040 unsnapped); a hit is within +-0.004 per seed. Honest risk: the attachment and the geometry loss lean on end quality, and the paint evidence helped the ends of crowded junctions; the arm may lose more than on E312.
- **Power:** the two seeds of E341 differed by 0.0065 in the snapped loss (+0.0005 against -0.0060): the point estimate of one seed cannot resolve +-0.003; the rule stays the point estimate (D85) and the paired scene bootstrap (`jdiag.py boot e348_w_s<S> ends_card_snap_thr0.1 e324_s_s<S> ends_card_snap_thr0.1`, and for PRIMARY `ends_card_snap_nogeom_attach_thr0.1`) says how firm it is; E324 s has the counts files. An inside-the-band result with an interval that contains the bound is reported as such, not as a clean ADOPT.
- **What it saves when adopted:** as E341: the 0.5 m branch (`m05`, `head05`: 1.477 GMACs, 0.037 M parameters), its loss and the two paint inputs of the pair MLP, nothing else; the ms are an estimate (about 1.3 ms of the 17.09 ms forward) until E342 part B times it.
- **Controls and confounds:** E324 s of the same seed is the matched reference (same recipe, same steps, same sample order for the same seed: the only differences are the maps checkpoint and the four-input evidence); E341 (3000 steps, E312 recipe) shows what the removal costs without the attachment; the pair-stage noise has never been measured (E318 revision item 1: one run per arm and seed): the interval is the scene sample only.
- **Cost and start:** the E324 s step is about 1.5 s on a free GPU: 6000 steps is about 2.5 h per seed, plus the dump and scores (about 25 minutes per seed on the CPU); `budget_gpu_h` 5.5 for both seeds. Start script `temp/e348/start_pairs.sh <seed>` (the command above through `run.py` with this card, written when the card is accepted), the chain as `temp/day20/day20.sh`: the post stage is the `post_e324` of `temp/day14/day14.sh` (dump, `--attach --snap --snap-nogeom`, `--snap`, `--snap-nogeom`, pair PR against E324 s, kperr, maps score, bootstraps against E324 s); the GPU evaluation steps wait for GPU room (`evaluation/gpu_gate.py`, D100). Queue: after E317 - E319; the GPU is first come first served.
- **Not claimed:** anything on TEST (E343 uses E324 s only); that the removal helps the ms (E342 part B); that the maps head width should change (D92).

## 2. Referee
A fresh `ar-critic` (cards) on this card before the first start; a `code` pass on the `jrule_removal.py` extension (`primary` read-out).
