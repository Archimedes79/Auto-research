# E337 - the attached ends against the snap's geometry (D87 / D88): where the end-to-end read-out of E324 loses +0.0069 / +0.0062 topo F1, and a lever only if step 0 says so

Card of 2026-11-14 (lane chat, on the lane lead's order after E324; version 1, critic first). Follow-up of E324 ('E2E-required, kept and refined'); only differentiable solutions (D88). **Status: card only. Step 0 is CPU, no training, no new code in a shared file (one scratch script, named below). No GPU before this card is critic-accepted and step 0 has a result.**

## 0. The card (what the tools run)
```json card
{
 "id": "E337",
 "card": "",
 "title": "the E324 s attached ends read out with the snap's geometry on the wired piece (replay, by hand), against the own-keypoint snap and the attached read-out without it: is the gap geometry or wiring?",
 "kind": "variant",
 "deterministic": true,
 "metric": "topo_F1",
 "direction": "higher",
 "band": 0.003,
 "prediction": {"value": 0.0, "vs": "c"},
 "better": {"dup_frac": "lower"},
 "arms": {
  "c": {"recipe": "stored: E324 s own keypoints, snapped (`runs/e324_s_s<S>/ends_card_snap_thr0.1.json`): topo F1 0.6981 / 0.7010, dup_frac 0.1781 / 0.1522"},
  "a": {"recipe": "stored: E324 s PRIMARY (attached ends, snap-nogeom wiring): topo F1 0.6912 / 0.6948, dup_frac 0.1897 / 0.1592"},
  "r": {"recipe": "replay, by hand (CPU, the same dump): `jends_eval.py score --run e324_s_s<S> --seed <S> --attach --snap --thr 0.1 --tag r` (the connector starts at the ATTACHED end, the wired piece end replaces it as in the snap, geometry on); card name carries the tag; no training", "run_ids": ["E337.r_s0", "E337.r_s1"]}
 },
 "control": "c",
 "variant": "r",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "topo_F1", "vs": "c", "max_loss": 0.003},
  {"metric": "dup_frac", "vs": "c", "max_loss": 0.004}
 ],
 "budget_gpu_h": 0.0,
 "kill": "none for step 0 (CPU); a training arm needs its own card text and critic pass",
 "evaluate": {"split": "val"}
}
```
**Reading rule (by hand; written before the run):** let G = topo F1 of c minus topo F1 of a (+0.0069 / +0.0062) and R = r minus c. **GEOMETRY** when |R| <= 0.003 on both seeds (the snap's geometry on the wired piece is the whole gap: the attached end adds nothing once the curve starts at the piece end). **WIRING** when r beats c by more than 0.003 on both seeds (the attachment picks better pieces than the keypoints do). **BOTH-WRONG** when r is more than 0.003 below c on both seeds (the attachment moves the wiring away from the nearest piece: it hurts under any geometry). Mixed seeds: reported as such, no step 1. The second reading is the distance statistic below.

## 1. Card text
- **Statement (falsifiable):** the topo F1 the PRIMARY (end-to-end) read-out of E324 gives up against the snap (+0.0069 / +0.0062) is the snap's geometry, i.e. the attached end does not coincide with the delivered piece end it is wired to, and replacing it by that piece end (snap on the attached end) recovers the whole gap.
- **Why it can be this:** `Attach` (`jends.py:92 - 136`) returns the weighted MEAN of the keypoint and up to ATT_M raw nodes of B's head (a convex combination, w0 the 'no attachment' weight): the attached end is only a delivered piece end when one weight dominates, and the training log shows `att_acc` about 0.32 - 0.36. The wiring at the read-out then picks the NEAREST delivered piece to that mean (the rule), so a mean between two candidates gives a curve that starts beside both pieces. The snap's geometry (+0.0093 / +0.0094 over the same wiring without it) is the one rule step of the delivered graph that the attachment was meant to replace, and it recovers only 26 % / 34 % of it.
- **Step 0 (CPU, no training):** (1) the replay r on both seeds, about 6 minutes per seed (niced; existing options, no code change). (2) The distance statistic: for every connector end of the PRIMARY read-out at p >= 0.1 the distance from the attached end to the piece end the snap would wire (4 m / 45 degrees, `jwire.candidates`), the same for the own keypoint, the share within 0.5 / 1 / 2 m, and topo F1 of c computed on the frames split by that distance is NOT claimed (no causal split); the statistic is one scratch script (`temp/e337/dist.py`, reading `ends_val.pkl` and B's full dump); promoted into `evaluation/` with a test only when it is used a second time. (3) The mean attachment weight of the dominant candidate (`att_p`, stored in the dump): the share of ends with a dominant weight above 0.8.
- **What the result decides (rule, not prediction):** GEOMETRY and a median attached-to-piece distance above 0.5 m: write the follow-up card for a lever that makes the attached end COINCIDE with a candidate while staying differentiable (a sharpening term on the attachment weights, i.e. an entropy / margin loss on the existing softmax; or a straight-through argmax in the forward pass; the second is a named hard step, the first is not), one arm, two seeds, 3.2 h per seed, critic first. WIRING: the attachment is better than the rule at choosing the piece; the follow-up is to read it out as a wiring choice (an input of the E325 head). BOTH-WRONG or mixed: close, state the price in the card of E324. Nothing is trained in this card.
- **Prediction (numbers, written before the run; a hit is within +-0.003):** r - c = 0.000 topo F1 (the GEOMETRY reading), dup_frac of r within 0.004 of c's; median distance of the attached end to the wired piece end 0.8 m (a guess: no number exists), own keypoint 1.3 m (kperr: the keypoint is 1.3 - 1.4 m from the GT end on average).
- **Controls and confounds:** c and a are stored; r is the same dump and the same pairs, only the geometry rule differs; the replay uses a rule step (the snap) as a DIAGNOSTIC (D88): it is not a candidate. One training per seed: the scene bootstrap is the interval (`jdiag.py boot`); nothing under 0.003 is claimed.
- **Price:** step 0 costs about 15 CPU minutes, no GPU, no new code in a shared file. **Named hard steps (unchanged from E324):** the nearest-piece wiring at the read-out, the candidate set of the attachment, the keypoint peak choice, p >= 0.1, the close-node step (E292).
- **Not claimed:** that the attached end is the cause of the dup_frac excess on seed 0 (E324 section 5 puts +0.0071 there; this card does not test it); anything on TEST.
