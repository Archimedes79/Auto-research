# E338 - stack check (D87 / D88, CPU, no training): the pair heads of E324 s read out on the delivered graph of c2 (B fine-tuned 8250 steps, pyramid frozen) instead of B's: is the gain of c2 additive to the gain of E324 s?

Card of 2026-11-14 04:00, version 2 (04:35) after the cards critic (REVISE: statement against prediction, no 'worse' branch, mechanism half wrong, controls and file names missing, three numbers off). Written BEFORE any number of the stack is looked at. Lane lead's order: two end-to-end candidates that each stay at or below the adoption bound alone; the stack is only a re-read-out of stored dumps, no retraining. Only differentiable training is involved (both parts are trained end to end); the output selections of the read-out stay named (D88). **Status: card v2; the scoring (CPU, niced, about 6 minutes per read-out and seed) runs next; the verdict script `temp/e338/verdict_e338.py` is written before it.**

## 0. The card (what the tools run)
```json card
{
 "id": "E338",
 "card": "",
 "title": "E324 s (own keypoints, snapped) on the delivered graph of c2 against E324 s on B's graph, same pair heads, both seeds, VAL",
 "kind": "variant",
 "deterministic": false,
 "metric": "topo_F1",
 "direction": "higher",
 "band": 0.01,
 "prediction": {"value": 0.0043, "vs": "s"},
 "better": {"dup_frac": "lower", "vert100_dp": "lower"},
 "arms": {
  "s": {"recipe": "stored: E324 s own keypoints snapped on B's delivered graph (`runs/e324_s_s<S>/ends_card_snap_thr0.1.json`): topo F1 0.6981 / 0.7010, dup_frac 0.1781 / 0.1522, F1@1.0 0.8762 / 0.8763, junction F1 0.3238 / 0.3232, vertices 18.97 / 19.63, lanes found 0.7604 / 0.7594, curves 11729 / 11880"},
  "sc": {"recipe": "the SAME pair run `e324_s_s<S>` (maps E307 + pair head; `ends_val.pkl` reused, nothing retrained, nothing re-dumped) with its own keypoints, snapped, on the delivered graph of c2 read out with the hard step: `jends_eval.py score --run e324_s_s<S> --seed <S> --snap --thr 0.1 --full <root>/temp/e292/predfull_e292_c2_s<S>.pkl --base <root>/runs/e292_c2_s<S>/model.pt --spec close:1.5@0.5 --tag e338` (card `runs/e324_s_s<S>/ends_card_e338_snap_thr0.1.json` + `.counts.pkl`; the scorer asserts that the pyramid of the maps run equals c2's: it holds, checked by the critic on all four checkpoints); by hand: replay scoring, no run.py record and no ledger row (the result goes into section 5 and a ledger NOTE on the rows E324.s_s0 / s1)"}
 },
 "control": "s",
 "variant": "sc",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "topo_F1", "vs": "s", "min_effect": 0.005},
  {"metric": "F1@1.0", "vs": "s", "max_loss": 0.01},
  {"metric": "dup_frac", "vs": "s", "max_loss": 0.01},
  {"metric": "vert100_dp", "vs": "s", "max_loss": 1.0},
  {"metric": "junction_F1", "vs": "s", "max_loss": 0.03}
 ],
 "budget_gpu_h": 0.0,
 "kill": "invalid, not a verdict (plumbing only): the scorer's pyramid assert fails; or a CONTROL fails: (a) the same tagged scoring of e324_s_s<S> on B's own dump (`--snap --thr 0.1 --full <root>/temp/e276/predfull_e273_p_s<S>.pkl --base <root>/runs/e273_p_s<S>/model.pt --spec close:1.5@0.5 --tag e338b`) must reproduce the stored card of s to 1e-6 in every number (the plumbing was already verified for the E312 pair run: `runs/e312_p_s<S>/ends_card_twin_snap_thr0.1.json`, a tagged scoring on B's dump, differs from the card of record by 0.0 in every number); (b) `curves` of sc must equal those of s (11729 / 11880: the pair set does not depend on the graph); (c) the `numbers.base` block of the sc card (the graph without connectors) must equal the `base` block of `runs/e312_p_s<S>/ends_card_c2_snap_thr0.1.json`, and that of e338b the `base` block of s's card",
 "evaluate": {"split": "val"}
}
```
**Verdict rule (written before the numbers; by hand: replay of the scored cards, `decide.py` is not used; the arms are one pair run on two graphs, each graph one fine-tune draw per seed):** let D = sc - s on topo F1, per seed.
- **KEEP as a stack candidate:** D >= +0.005 on BOTH seeds, the paired scene-bootstrap 95 % interval of D (`jdiag.py boot`) excludes 0 on both seeds, and every guard holds on both seeds (F1@1.0 and junction F1 as losses against s, dup_frac and vertices as rises; the bounds in the block).
- **TIE, kept (c2 and E324 s stay two separate candidates, the stack is not claimed):** any seed with -0.003 <= D < +0.005, or a guard missed, or an interval that contains 0.
- **NEGATIVE INTERACTION (reported, the stack is not formed):** D < -0.003 on either seed (the fine-tuned graph hurts the pair heads of E324 s that were trained on B's graph).
- There is no CUT (nothing is built or trained here). The additive claim of section 1 is tested separately: the interaction is read from D against the additive value, window +-0.003.

## 1. Card text
- **Statement (falsifiable, the additive claim):** the gain of the longer fine-tune of B (c2 over E312, snapped: +0.0065 / +0.0020; graph-only +0.0087 / +0.0122) and the gain of the E324 s recipe add: D = sc - s equals c2 - c within +-0.003 per seed (+0.0065 / +0.0020). This is a claim about the SIZE of D; the KEEP rule above is stricter (D >= +0.005 with an interval above 0) and is expected to be met on seed 0 only, so **the expected token is TIE**; the card does not claim KEEP.
- **Mechanism, and what is and is not independent of B (critic-corrected):** the keypoints and the pair logits of a pair run are computed by the maps head and the pair head from the frozen pyramid; B's node head enters the forward only through `Attach` (`jends.py:97 - 136`), and the node tokens only with `ntok`, which is off for E324 s. So the OWN-KEYPOINT read-out (`make_pairs(attach=False)` reads `kp_pos`, `kp_dir`, `scale`, `logit`, `allowed`) does not depend on B's nodes at INFERENCE, and its connectors reach the graph only through the delivered pieces they are wired to (nearest piece end within 4 m and 45 degrees, the long-link cover set, the snap geometry). TRAINING is not independent: during the E324 s run `Attach` receives the keypoint tokens and positions with gradient and the geometry loss acts on the final curve, so B's frozen node outputs shaped the trunk and the maps head (`jends_train.py:34 - 45`); a possible reason for a result BELOW the additive value. The `att_*` fields of `ends_val.pkl` come from B's raw nodes (a convex mix of the own keypoint and raw node-head positions, mid-section nodes included).
- **Pre-registered numbers (additive guess sc = s + (c2 - c) per seed; hit windows: topo F1 and dup_frac +-0.003, F1@1.0 +-0.003, junction F1 +-0.01, vertices +-0.3):** topo F1 **0.7046 / 0.7030** (D +0.0065 / +0.0020); dup_frac 0.1691 / 0.1588 (the seed-1 value already uses two thirds of the 0.01 guard); F1@1.0 0.8742 / 0.8792; junction F1 0.3414 / 0.3332; vertices 18.85 / 19.34; against E312 snapped (c) topo F1 +0.0222 / +0.0165. **Expected direction of a miss: BELOW the additive value** (both fine-tunes raise recall; the connectors of s differ from E312's: 11729 against 11378 curves, 6259 against 5992 wired on seed 0). Either direction beyond +-0.003 is reported as an interaction.
- **Reported, outside the rule (caveat):** the PRIMARY read-out (attached ends, one-piece wiring): `--attach --snap --snap-nogeom --thr 0.1 --tag e338a` on the same dump and base; the attached ends in `ends_val.pkl` were computed on B's nodes, so on c2's graph they point at positions c2's nodes no longer have exactly: labelled 'attached ends from B's nodes, wired on c2's pieces', NOT part of the rule. A faithful test needs a re-dump with c2's node head (GPU, about 5 minutes per run): not done here. Also reported: own keypoints snap-nogeom (`--snap --snap-nogeom --tag e338n`) for the split read-out / geometry.
- **Controls and confounds:** s, c2 and c are stored; the same pairs and keypoints, only the delivered graph differs (c2's fine-tuned graph against B's, both with the hard step); each graph is one fine-tune draw per seed, so 'c2 adds in general' stays unanswered (the scene bootstrap is the interval, the retrain noise is unknown); the c2 retrain noise cancels in D - (c2 - c) because the same c2 dump enters both. VAL only, TEST untouched.
- **Price and named hard steps:** CPU only, 6 - 8 read-outs of about 6 minutes (niced; the GPU is busy); no new code in a shared file. Named steps unchanged: the close-node step (`close:1.5@0.5`) on the graph, the nearest-piece wiring and the snap geometry, p >= 0.1, the keypoint peak choice, Douglas-Peucker in the scorer.
- **Not claimed:** anything on TEST; that the stack is the system of record (the adoption of c2 and of E324 s are decisions of Mirko, together with any TEST use); that the attachment read-out benefits.

## 5. Result
(not yet: written after the scoring by the rule above)

## 6. Referee
Cards critic 2026-11-14 04:25: REVISE (statement contradicted the prediction, no 'worse' branch, `deterministic` and the band, mechanism half wrong, controls (curves, base blocks), the plumbing evidence not cited, file names and the caveat command missing, three numbers off: dup_frac seed 1 is 0.1588, against E312 +0.0222 / +0.0165). Version 2 answers each; the call itself was checked valid (arguments, tags, pyramid assert on all four checkpoints).
