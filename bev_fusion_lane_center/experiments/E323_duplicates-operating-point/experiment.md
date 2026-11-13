# E323 - duplicates and the curve count: E312 read out at E320's number of curves, E320 read out at E312's (read-out only, CPU)

Card of 2026-11-13, written by the lane chat on the lane lead's order (D86) after the E320 verdict (CUT by the dup_frac guard on seed 0, `experiments/E320_soft-attach-replaces-snap/experiment.md` section 3); replaces the stub of 06:45 and the first version of 08:00, which the fresh critic sent back (REVISE: the count hypothesis of that version contradicted E312's own threshold cards). No training, no GPU: re-reads of stored dumps at other pair thresholds. The operating points are fixed below by a COUNT before any of these read-outs was scored.

## 0. The card (what the tools run)
```json card
{
 "id": "E323",
 "card": "",
 "title": "E320 PRIMARY read-out (attached ends, one nearest piece per end, p >= 0.1) against E312 read out at the same number of connector curves (snapped = c, snap without end geometry = u'), and the mirror reading of E320 at E312's number of curves",
 "kind": "variant",
 "deterministic": false,
 "metric": "topo_F1",
 "direction": "higher",
 "band": 0.01,
 "prediction": {"value": -0.0016, "vs": "c"},
 "better": {"dup_frac": "lower", "vert100_dp": "lower"},
 "arms": {
  "c": {"recipe": "control = E312 snapped at the pair threshold t1 that delivers as many curves as E320 does at p >= 0.1 (12018 / 12208): `bash tools/env/_py14.sh jends_eval.py score --run e312_p_s<S> --seed <S> --snap --thr <t1>` with t1 = 0.083 (seed 0, 12006 curves) / 0.090 (seed 1, 12209 curves); card `ends_card_snap_thr<t1>.json`"},
  "u": {"recipe": "floor u'(t1) = E312 snap-nogeom at the same threshold: `... jends_eval.py score --run e312_p_s<S> --seed <S> --snap --snap-nogeom --thr <t1>`; card `ends_card_snap_nogeom_thr<t1>.json`"},
  "a": {"recipe": "the E320 PRIMARY read-out at p >= 0.1, as stored: `runs/e320_a_s<S>/ends_card_snap_nogeom_attach_thr0.1.json` (topo F1 0.6803 / 0.6889, 12018 / 12208 curves)"}
 },
 "control": "c",
 "variant": "a",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "topo_F1", "vs": "u", "min_effect": 0.005},
  {"metric": "F1@1.0", "vs": "c", "max_loss": 0.01},
  {"metric": "dup_frac", "vs": "c", "max_loss": 0.01},
  {"metric": "vert100_dp", "vs": "c", "max_loss": 1.0},
  {"metric": "junction_F1", "vs": "c", "max_loss": 0.03}
 ],
 "budget_gpu_h": 0.0,
 "kill": "none (CPU read-out of stored dumps, no training)",
 "evaluate": {"split": "val"}
}
```
Two directions, both by hand (label **"by hand: replay"**: the arms are stored dumps of separately trained networks, no run of the card feeds the ledger rule; there are no `run_ids`, the E320 runs belong to the card of E320):
- **D1 (the block above): the control gets E320's count.** E320 stays as it was measured (p >= 0.1, stored cards); E312 is read out at the threshold at which it delivers the same number of curves (0.083 / 0.090). The rule of E320 (`E320 section 0`, topo F1 at least u' + 0.005 on both seeds, guards against c) is applied unchanged with c and u' at that count.
- **D2 (the mirror): E320 gets E312's count.** `bash tools/env/_py14.sh jends_eval.py score --run e320_a_s<S> --seed <S> --attach --snap --snap-nogeom --thr <t2>` with t2 = 0.120 (seed 0, 11383 curves) / 0.110 (seed 1, 11845); card `ends_card_snap_nogeom_attach_thr<t2>.json`; against the stored c and u' of E312 at p >= 0.1 (11378 / 11849 curves), the same rule.
The verdict of the card is KEEP at an equal count only if BOTH directions are KEEP (both seeds, all guards); TIE or CUT otherwise as in E320. **This card does not change the verdict of E320** (its rule was fixed at p >= 0.1).

**The operating points are counts** (from the pair logits of the dumps, `sigmoid(logit) * allowed >= t`, `temp/e323/count_thr.py` and `count_thr2.py`, scratch, no metric involved; the grid is 0.001): E312 at E320's count 0.083 / 0.090 (12006 / 12209 curves against 12018 / 12208, -0.10 % / +0.01 %); E320 at E312's count 0.120 / 0.110 (11383 / 11845 against 11378 / 11849, +0.04 % / -0.03 %). The operating point is in-sample on VAL for that one number per run and seed.

Intervals: the paired scene bootstrap next to every point estimate, including the duplicates (`jdiag.py boot e320_a_s<S> ends_card_snap_nogeom_attach_thr0.1 e312_p_s<S> ends_card_snap_thr<t1>`; `dup_frac` is added to the keys of `jdiag.boot` with this card, one line and its selftest).

## 1. Card text
- **Statement (falsifiable):** at an equal number of delivered curves the E320 PRIMARY read-out passes the rule of E320 on both seeds, i.e. the duplicate excess of E320 (+0.0184 / +0.0094 over E312 at p >= 0.1) is caused by the larger number of curves.
- **Two hypotheses, with the numbers that distinguish them.**
  H_count: more curves cause more duplicates (the between-recipe reading of the snapped p >= 0.1 cards: seed 0 E316 10887 curves 0.1678, E312 11378 0.1747, E315 12123 0.1807, E320 12018 0.1834; seed 1 10694 0.1532, 11849 0.1557, 12208 0.1589, 12503 0.1629; the OLS slope is +0.0013 duplicates per 1 % of curves on seed 0 and +0.0006 on seed 1, with an inversion on seed 0: E320 has fewer curves than E315 and more duplicates). H_count predicts that E312 read out at E320's count has dup_frac about +0.007 (s0) / +0.002 (s1) ABOVE its p >= 0.1 value.
  H_within: within one network the relation has the opposite sign, because dup_frac is a share of the delivered length and fewer connector curves leave more of the (more duplicated) plain graph: E312 snapped p >= 0.5 has 4941 / 5550 curves and dup_frac 0.1956 / 0.1645 against 11378 / 11849 curves and 0.1747 / 0.1557 at p >= 0.1 (the critic's check; E305_e and E287_p2 show the same sign), i.e. about -0.00037 (s0) / -0.00017 (s1) per 1 % of curves. H_within predicts that E312 read out at E320's count has dup_frac about 0.002 / 0.0005 BELOW its p >= 0.1 value, and E320 at E312's count about 0.002 / 0.0005 ABOVE its own.
  The stored within-run evidence is the stronger prior (it is the same network), so the prediction below follows H_within.
- **Prediction (numbers, written before the scores; a hit is within +-0.004 for dup_frac and +-0.003 for topo F1):**
  D1, E312 at E320's count (t1): seed 0 dup_frac 0.1727 (E312 at p >= 0.1: 0.1747), topo F1 0.6846 (0.6825; the within-run loss of topo F1 is about 0.0004 per 1 % of curves, from the p >= 0.5 cards 0.6612 / 0.6639); seed 1 dup_frac 0.1552 (0.1557), topo F1 0.6877 (0.6864). Then a - c = -0.0043 (s0) / +0.0012 (s1), a - u'(t1) = +0.0076 / +0.0116 (u'(t1) about 0.6727 / 0.6773), and the dup guard against c(t1) is +0.0204 (s0, broken) / +0.0099 (s1, holds by 0.0001): the verdict stays CUT by the guard on seed 0. The block above carries the mean of the two a - c as its prediction number (-0.0016).
  D2, E320 at E312's count (t2): seed 0 dup_frac 0.1951 (+0.0204 over c), topo F1 0.6783; seed 1 dup_frac 0.1656 (+0.0099), topo F1 0.6877. The same CUT.
  If H_count were right the guard would pass at the matched count (E312 at E320's count: 0.182 / 0.158, a - c(t1) about +0.011 / +0.007): then the verdict would be marginal KEEP-or-CUT on seed 0, so the two hypotheses are separated by the single number dup_frac of E312 at t1 (seed 0: below 0.1747 = H_within, above 0.1787 = H_count).
- **What a result would change:** E312 at t1 shows dup_frac below its p >= 0.1 value and a - c(t1) still above +0.01 on seed 0: the duplicates are not the count, they come with the E320 recipe or the attached ends; the E320 CUT stands and the next lever is not the threshold. E312 at t1 shows dup_frac above +0.004: the count explains part of the guard break, and a KEEP in both directions would make the E320 recipe a candidate for the system of record at a calibrated threshold. **What a KEEP would and would not license:** it would not show that the attachment works (the recipe also carries the geometry loss, E320 section 3: the attached ends are +0.0022 / +0.0040 of a - u'); it is a post hoc operating point chosen after the CUT (the count rule is fixed here before the scores; a threshold for a TEST read-out would have to be fixed on VAL); and the TEST split stays closed: D84 covers only E312 and says a later design needs its own OK for its own TEST use.
- **Mechanism check without scoring (done before the card was written, `temp/e323/neardup.py`, dumps only):** near-duplicate connectors (both ends within R m of the ends of another delivered connector of the same frame, p >= 0.1) are rare and do not rise with the attachment: R = 1 m 0.0 % (E312 keypoints), 0.0 % (E320 keypoints), 0.0 % (E320 attached) on seed 0 and 0.0 / 0.0 / 0.1 % on seed 1; R = 2 m 0.4 / 0.6 / 0.4 % (s0) and 0.6 / 0.3 / 0.5 % (s1). The suspicion of E320 section 3 that the attached ends of neighbouring keypoints land on the same node and draw the same curve is not supported; the duplicates of dup_frac are not identical connectors.
- **Inputs:** the dumps `runs/<run>/ends_val.pkl` of E312 and E320 and B's full dump (`temp/e276/predfull_e273_p_s<S>.pkl`); all exist. Reported, not in the rule: E316 and E315 at their own matched counts are NOT read here (the E316 rule at p >= 0.1 stays a TIE); they would be a free third test of the sign and can be added by a later card.
- **Cost:** CPU only, nice 19, one thread: three scores per seed (E312 snapped and snap-nogeom at t1, E320 primary at t2), about 4 minutes each when idle, longer while a training round runs; every threshold has its own card file, no card of record is touched.
- **Starts from:** E312 (c, u'), E320 as measured. **Lens / group / status:** the guard of E320 and the operating point of the system; proposed 2026-11-13.
- **Referee:** first critic pass 2026-11-13 08:05: REVISE (the points above: the within-run contradiction, the prediction without tolerance, the missing within-run control, `deterministic`, the KEEP wording, commands); this version answers it, the second pass is written below when it has run.

## 2. Weaknesses stated in advance
(1) The count is a proxy: two read-outs with the same count can differ in how many curves are near-parallel to delivered pieces; matching it removes the first-order effect, not every difference. (2) One number per run and seed, chosen by the count on the same VAL frames the metric is read on (in-sample for the count, not for the topo F1). (3) The threshold of a deployed system is a hard step; a calibrated one per recipe is more of the same step. (4) The within-run slopes behind the prediction come from two points per seed (p >= 0.1 and p >= 0.5, far apart), so they are rough; the reading at the matched count is the measurement. (5) E320 differs from E312 in training, not only in the read-out; the card separates the count, not the recipe.

## 3. Second referee pass
Pending.
