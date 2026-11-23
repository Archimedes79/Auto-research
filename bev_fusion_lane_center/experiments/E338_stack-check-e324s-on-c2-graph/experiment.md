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
- **Not claimed:** anything on TEST; that the stack is the system of record (the adoption of c2 and of E324 s are decisions of the owner, together with any TEST use); that the attachment read-out benefits.

## 5. Result (2026-11-14 04:45, VAL 873 frames, p >= 0.1, both seeds; numbers from `runs/e324_s_s<S>/ends_card_e338*_*.json` against the stored cards; verdict by `temp/e338/verdict_e338.py`, bootstraps `temp/e338/boot_sc_vs_{s,c}_s<S>.log`, scoring log `temp/e338/score_all.log`)
**Controls (the stop rules) all passed on both seeds:** the tagged scoring on B's own dump (e338b) reproduces the stored card of s in every number and its base block; `curves` of sc equal those of s (11729 / 11880); the base block of sc equals the base block of the stored c2 card.
**Token by the rule written above: TIE, kept (c2 and E324 s stay two separate candidates; the stack is not claimed as KEEP).** Seed 0 meets the KEEP bound (D = +0.0085, interval [+0.0036, +0.0144], guards held), **seed 1 does not (D = +0.0006, interval [-0.0043, +0.0058])**; no seed is below -0.003, so no negative interaction.

| own keypoints, snapped | seed 0 | seed 1 |
|---|---|---|
| s on B's graph (stored) | 0.6981 | 0.7010 |
| **sc on c2's graph (E338)** | **0.7066** | **0.7016** |
| D = sc - s | **+0.0085** | **+0.0006** |
| additive prediction (s + c2 - c) | 0.7046 | 0.7030 |
| sc - additive | +0.0020 (HIT) | -0.0014 (HIT) |
| sc - E312 snapped (c); bootstrap | **+0.0242** [+0.0147, +0.0333] | **+0.0152** [+0.0081, +0.0218] |

**The additive claim holds: every pre-registered number is a hit on both seeds** (hit windows +-0.003 topo F1, dup_frac and F1@1.0, +-0.01 junction F1, +-0.3 vertices): dup_frac 0.1668 / 0.1584 (additive 0.1691 / 0.1588), F1@1.0 0.8737 / 0.8787 (0.8742 / 0.8792), junction F1 0.3418 / 0.3312 (0.3414 / 0.3332), vertices 18.90 / 19.33 (18.85 / 19.34). The expected direction of a miss (below the additive value) did not occur. So the gain of c2 adds to the gain of E324 s exactly as much as c2 has on its own, and c2 has little on seed 1 (+0.0020 over E312): the stack rule fails on seed 1 because c2's own gain there is below the rule's bound, not because of an interaction.
Guards of the rule against s (all held on both seeds): F1@1.0 -0.0025 / +0.0024, dup_frac -0.0113 / +0.0063 (the seed-1 rise uses two thirds of the 0.01 guard), vertices -0.07 / -0.29, junction F1 +0.0180 / +0.0080; lanes found +0.0022 / +0.0086. **Against E312 snapped (c), reporting, not a verdict:** topo F1 +0.0242 / +0.0152, junction F1 +0.0224 / +0.0169 (seed 1 interval [-0.0012, +0.0366]), F1@1.0 +0.0068 / +0.0102, lanes found +0.0251 / +0.0258, dup_frac -0.0079 / +0.0027 (intervals contain 0): the stack of two end-to-end changes (the E320 recipe at 6000 pair steps and the 8250-step fine-tune of B with a frozen pyramid) passes the E312 adoption rule (> +0.01 on both seeds, guards held) with more margin than E324 s alone (+0.0156 / +0.0145). Adoption of c2 as B and of E324 s, and any TEST use, are the owner's decisions (HANDOVER questions); nothing adopted, nothing on TEST.
**Caveat read-outs (outside the rule; the attached ends come from B's nodes, wired on c2's pieces):** PRIMARY (attached ends, snap-nogeom wiring) on c2's graph 0.6989 / 0.6949 against 0.6912 / 0.6948 on B's graph (+0.0077 / +0.0001), dup_frac 0.1777 / 0.1655 (B: 0.1897 / 0.1592); own keypoints snap-nogeom on c2's graph 0.6960 / 0.6925 (B: 0.6888 / 0.6916). The geometry share of the snap on the new graph stays: own snapped minus own nogeom +0.0106 / +0.0091, minus PRIMARY +0.0077 / +0.0067. The attached read-out gains from c2 on seed 0 only; a faithful test needs a re-dump with c2's node head (GPU, not done).
**Named hard steps (D88), unchanged:** the close-node step on c2's graph, the nearest-piece wiring and the snap geometry, p >= 0.1, the keypoint peak choice, Douglas-Peucker. Price: none at inference beyond the two parts (c2 costs no ms, E324 s's attachment head is not needed by the own-keypoint read-out).

## 6. Referee
Cards critic 2026-11-14 04:25: REVISE (statement contradicted the prediction, no 'worse' branch, `deterministic` and the band, mechanism half wrong, controls (curves, base blocks), the plumbing evidence not cited, file names and the caveat command missing, three numbers off: dup_frac seed 1 is 0.1588, against E312 +0.0222 / +0.0165). Version 2 answers each; the call itself was checked valid (arguments, tags, pyramid assert on all four checkpoints).
