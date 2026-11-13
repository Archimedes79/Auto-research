# E323 - duplicates and the curve count: E312 read out at E320's number of curves, E320 read out at E312's (read-out only, CPU)

Card of 2026-11-13, written by the lane chat on the lane lead's order (D86) after the E320 verdict (CUT by the dup_frac guard on seed 0, `experiments/E320_soft-attach-replaces-snap/experiment.md` section 3). Version 3: the first version was sent back by the fresh critic (REVISE: its count hypothesis contradicted E312's own threshold cards), the second (version 2) got REVISE with small edits (card file names, the combined verdict, guards without a prediction); this version answers both. No training, no GPU: re-reads of stored dumps at other pair thresholds. The operating points are fixed below by a COUNT (model outputs only, no label) before any of these read-outs was scored.

**The family it belongs to:** the number of connectors that enter the delivered graph (the pair threshold) and the number of pieces a connector end is wired to (k = 1 in the snap, every piece in reach in the proximity wiring) are one family of COUNT effects at the output: the snap's wiring gain is the count 'one piece instead of every piece' (E322), the duplicate excess of E320 is asked here as a count of curves. Both are read-out only and CPU; a variant with k = 2 weighted pieces is the next member (not planned here).

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
  "c": {"recipe": "control = E312 snapped at the pair threshold t1 that delivers as many curves as E320 does at p >= 0.1 (12018 / 12208): `bash tools/env/_py14.sh jends_eval.py score --run e312_p_s<S> --seed <S> --snap --thr <t1>`; t1 = 0.083 (seed 0, 12006 curves), 0.09 (seed 1, 12209 curves); card files `runs/e312_p_s0/ends_card_snap_thr0.083.json`, `runs/e312_p_s1/ends_card_snap_thr0.09.json`"},
  "u": {"recipe": "floor u'(t1) = E312 snap-nogeom at the same threshold: `... jends_eval.py score --run e312_p_s<S> --seed <S> --snap --snap-nogeom --thr <t1>`; card files `ends_card_snap_nogeom_thr0.083.json` (seed 0), `ends_card_snap_nogeom_thr0.09.json` (seed 1)"},
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
Two directions, both by hand (label **"by hand: replay"**: the arms are stored dumps of separately trained networks, no run of the card feeds the ledger rule; there are no `run_ids`, the E320 runs belong to the card of E320, as in the other read-out cards of this project):
- **D1 (the block above): the control gets E320's count.** E320 stays as it was measured (p >= 0.1, stored cards); E312 is read out at the threshold at which it delivers the same number of curves. The rule of E320 (`E320 section 0`) is applied unchanged with c and u' at that count.
- **D2 (the mirror): E320 gets E312's count.** `bash tools/env/_py14.sh jends_eval.py score --run e320_a_s<S> --seed <S> --attach --snap --snap-nogeom --thr <t2>` with t2 = 0.12 (seed 0, 11383 curves) / 0.111 (seed 1, 11845); card files `runs/e320_a_s0/ends_card_snap_nogeom_attach_thr0.12.json`, `runs/e320_a_s1/ends_card_snap_nogeom_attach_thr0.111.json`; against the stored c and u' of E312 at p >= 0.1 (11378 / 11849 curves), the same rule. D2 moves E320 off its pre-registered threshold, so it is a mirror and a cross-check, not a rescue; the two directions are strongly correlated and add little independent evidence.
(The file stems carry the threshold as the scorer prints a float: 0.083, 0.09, 0.12, 0.111. The counts files are `<stem>.counts.pkl`.)

**Void criterion:** the scored curves must equal the counted ones within 0.1 % (12006 / 12209 for E312 at t1, 11383 / 11845 for E320 at t2: key `curves` of the card files); otherwise the read-out is not used and the thresholds are recounted.

**Combined verdict, defined (the per-direction verdict V is the one of the E320 rule: KEEP = both seeds topo F1 of a at least u' + 0.005 and no guard broken; TIE = no guard broken, both seeds at least u', at least one seed inside [u', u' + 0.005); CUT = a seed below u' or any guard broken):**

| V(D1) \ V(D2) | KEEP | TIE | CUT |
|---|---|---|---|
| KEEP | **KEEP at an equal count** | TIE | CUT |
| TIE | TIE | TIE | CUT |
| CUT | CUT | CUT | **CUT** (the E320 CUT stands) |

**The operating points are counts** (model outputs and the `allowed` mask only: `sigmoid(logit) * allowed >= t` over `runs/<run>/ends_val.pkl`, `temp/e323/count_thr.py` and `count_thr2.py`, scratch; the grid is 0.001): E312 at E320's count 0.083 / 0.090 (12006 / 12209 curves against 12018 / 12208, -0.10 % / +0.01 %); E320 at E312's count 0.120 / 0.111 (11383 / 11845 against 11378 / 11849, +0.04 % / -0.03 %; the first version of this card printed 0.110 for seed 1, a display rounding of the count script: 0.110 delivers 11885 curves). No label and no metric enters the choice.

**Intervals** (paired scene bootstrap, `jdiag.py boot`, the duplicates carry an interval since 2026-11-13; the call is checked on stored counts, there is no selftest of `boot`): D1 `jdiag.py boot e320_a_s<S> ends_card_snap_nogeom_attach_thr0.1 e312_p_s<S> ends_card_snap_thr<t1>` (a - c), `... e312_p_s<S> ends_card_snap_nogeom_thr<t1>` (a - u'), and the key number `jdiag.py boot e312_p_s<S> ends_card_snap_thr<t1> e312_p_s<S> ends_card_snap_thr0.1` (E312 at E320's count against E312 at p >= 0.1: the duplicates and topo F1 of the count itself); D2 `jdiag.py boot e320_a_s<S> ends_card_snap_nogeom_attach_thr<t2> e312_p_s<S> ends_card_snap_thr0.1`.

## 1. Card text
- **Statement (falsifiable):** at an equal number of delivered curves the E320 PRIMARY read-out passes the rule of E320 on both seeds, i.e. the duplicate excess of E320 (+0.0184 / +0.0093 over E312 at p >= 0.1, bootstrap intervals [+0.0128, +0.0233] / [+0.0033, +0.0154]) is caused by the larger number of curves.
- **Two hypotheses, with the numbers that distinguish them.**
  H_count: more curves cause more duplicates (the between-recipe reading of the snapped p >= 0.1 cards: seed 0 E316 10887 curves 0.1678, E312 11378 0.1747, E315 12123 0.1807, E320 12018 0.1834; seed 1 10694 0.1532, 11849 0.1557, 12208 0.1589, 12503 0.1629; the OLS slope is +0.0013 duplicates per 1 % of curves on seed 0 and +0.0006 on seed 1, with an inversion on seed 0: E320 has fewer curves than E315 and more duplicates). H_count predicts that E312 read out at E320's count has dup_frac +0.0072 (s0) / +0.0018 (s1) above its p >= 0.1 value.
  H_within: within one network the relation has the opposite sign. `dup_frac` (`evaluation/sparse_lanes/topo_metrics.py`) is the share of the delivered length that runs parallel within 0.75 m to ANY other delivered piece (B's own lines included; graph neighbours and the first and last 4 m excluded), so connectors dilute the share (more length that is not a duplicate) and also remove B's duplicated long links that they cover (the plain graph has dup_frac 0.2345 on seed 0, the snapped read-out 0.1747; `long_links_dropped` 9023 for E312 and 8986 for E320). The stored within-run evidence has one sign: E312 snapped p >= 0.5 has 4941 / 5550 curves and dup_frac 0.1956 / 0.1645 against 11378 / 11849 curves and 0.1747 / 0.1557 at p >= 0.1, about -0.00037 (s0) / -0.00017 (s1) per 1 % of curves; E305_e and E287_p2 (`ends_card_snap_thr0.2.json`, both seeds, 0.5 and 0.2 against 0.1) show the same sign in all eight cells, with a local slope near 0.1 of -0.0001 to -0.0003. H_within predicts that E312 read out at E320's count has dup_frac 0.0020 / 0.0005 BELOW its p >= 0.1 value.
  The stored within-run evidence is the stronger prior (same network, one sign everywhere), so the predictions below follow H_within.
- **Which single number separates them (seed 0):** dup_frac of E312 at t1 = 0.083. At or below 0.1747 (E312 at p >= 0.1): H_within; at or above 0.1787 (+0.004): H_count; in between: inconclusive. A KEEP of D1 on seed 0 would need dup_frac(c(t1)) at or above 0.1831 (+0.0084), above the H_count centre (0.1819). **Seed 1 is a replication of the sign, not a discriminator:** the two hypotheses differ there by 0.0025, below the tolerance (H_within -0.0005, H_count +0.0018 around 0.1557).
- **Predictions (numbers, written before the scores; a hit is within +-0.004 for dup_frac, +-0.003 for topo F1 and junction F1; the slopes are the within-run ones of E312 between p >= 0.5 and p >= 0.1: dup_frac -0.00037 / -0.00017, topo F1 +0.00038 / +0.00042, junction F1 +0.00051 / +0.00055 per 1 % of curves, seed 0 / 1):**
  D1, E312 at E320's count (t1, +5.5 % / +3.0 % curves): seed 0 dup_frac 0.1727, topo F1 0.6846, junction F1 0.3222; seed 1 dup_frac 0.1552, topo F1 0.6877, junction F1 0.3160. Then a - c = -0.0043 (s0) / +0.0012 (s1) topo F1, a - u'(t1) = +0.0076 / +0.0117 (u'(t1) about 0.6727 / 0.6773), dup guard a - c(t1) = +0.0204 (s0, broken) / +0.0099 (s1: **a coin flip by construction**, the margin 0.0001 is far below the tolerance, the outcome of that cell cannot be predicted), junction guard a - c(t1) = -0.0284 (s0: holds by 0.0016, **a second near-binding guard**, tolerance 0.003 covers the guard) / -0.0155 (s1, holds). The expected verdict of D1 is CUT by the dup guard on seed 0 (and possibly a junction CUT).
  D2, E320 at E312's count (t2, -5.3 % / -3.0 % curves): seed 0 dup_frac 0.1951 (+0.0204 over c at p >= 0.1), topo F1 0.6783, junction F1 0.2911 (a - c -0.0283: again near the guard); seed 1 dup_frac 0.1656 (+0.0099, again a coin flip), topo F1 0.6876, junction F1 0.2989 (-0.0154). The same expected CUT.
  **The slope behind the topo F1 numbers is rough:** the local slope between p >= 0.2 and p >= 0.1 (E305_e, E287_p2) is 3 to 5 times smaller (0.00007 to 0.00014 per 1 %), which would make E312's topo gain at t1 about +0.0006 instead of +0.0021 and a - c about -0.003, a - u' about +0.009; both readings are inside the tolerance and change no verdict. The block above carries the mean of the two a - c (-0.0016).
- **What a result would change:** E312 at t1 shows dup_frac at or below its p >= 0.1 value and a - c(t1) still above +0.01 on seed 0: the duplicates are not the count, they come with the E320 recipe or the attached ends; the E320 CUT stands and the next lever is not the threshold. E312 at t1 shows dup_frac at or above +0.004: the count explains part of the guard break, and a KEEP in both directions would make the E320 recipe a candidate for the system of record at a calibrated threshold. **What a KEEP would and would not license:** it would not show that the attachment works (the recipe also carries the geometry loss, E320 section 3: the attached ends are +0.0022 / +0.0040 of a - u'); it is a post hoc operating point chosen after the CUT (the count rule is fixed here before the scores; a threshold for a TEST read-out would have to be fixed on VAL); and the TEST split stays closed: D84 covers only E312 and says a later design needs its own OK for its own TEST use.
- **Mechanism check without scoring (done before the card was written, `temp/e323/neardup.py`, dumps only):** near-duplicate connectors (both ends within R m of the ends of another delivered connector of the same frame, p >= 0.1, on the raw keypoint ends or the attached ends of the dump) are rare and do not rise with the attachment: R = 1 m 0.0 % (E312 keypoints), 0.0 % (E320 keypoints), 0.0 % (E320 attached) on seed 0 and 0.0 / 0.0 / 0.1 % on seed 1; R = 2 m 0.4 / 0.6 / 0.4 % (s0) and 0.6 / 0.3 / 0.5 % (s1). The suspicion of E320 section 3 that the attached ends of neighbouring keypoints land on the same node and draw the same curve is not supported: the duplicates of dup_frac are not identical connectors. Not tested: connector-to-B-piece parallels, and the snapped ends of c.
- **Inputs:** the dumps `runs/<run>/ends_val.pkl` of E312 and E320 and B's full dump (`temp/e276/predfull_e273_p_s<S>.pkl`); all exist. E316 and E315 at their own matched counts are NOT read here (the E316 rule at p >= 0.1 stays a TIE); they would be a free third test of the sign and can be added by a later card.
- **Cost:** CPU only, nice 19, one thread: three scores per seed (E312 snapped and snap-nogeom at t1, E320 primary at t2), about 4 minutes each when idle, longer while a training round runs; every threshold has its own card file, no card of record is touched. **Record when it has run:** the exact card files in section 3 of this card and a note on the ledger rows of E320 (the E320 runs; there is no run of this card).
- **Starts from:** E312 (c, u'), E320 as measured. **Lens / group / status:** the guard of E320 and the operating point of the system; proposed 2026-11-13, second critic pass answered.
- **Referee:** first pass 08:05 REVISE (within-run contradiction, prediction without tolerance, missing within-run control, `deterministic`, KEEP wording, commands); second pass 08:25 REVISE, small edits, 'no new critic pass needed once the three must-fix points are done' (card file names, the combined verdict table, guard predictions): done in this version; the other points (one-sided cut-offs, seed 1 as replication, the local topo slope, the dup_frac definition, bootstrap commands, void criterion, +0.0093) are done too.

## 2. Weaknesses stated in advance
(1) The count is a proxy: two read-outs with the same count can differ in how many curves are near-parallel to delivered pieces; matching it removes the first-order effect, not every difference. (2) The thresholds come from model outputs on the VAL frames (no labels), which is not a tuning on the metric. (3) The threshold of a deployed system is a hard step; a calibrated one per recipe is more of the same step. (4) The within-run slopes behind the prediction come from two far apart points per seed (p >= 0.5 and p >= 0.1), the local slope for topo F1 is 3 to 5 times smaller (`runs/e305_e_s*/ends_card_snap_thr0.2.json`); the reading at the matched count is the measurement. (5) E320 differs from E312 in training, not only in the read-out; the card separates the count, not the recipe. (6) Seed 1 cannot discriminate and two guards sit within 0.002 of their limits on seed 0, so a verdict can flip on a hair; the bootstrap intervals are given next to every point.

## 3. Result (VAL 873 frames, two seeds, CPU; scored 08:25 - 08:35 by `temp/e323/run_e323.sh`, nice 19, one stream per seed; read by `temp/e323/read_result.py`, bootstraps by `temp/e323/boots.sh`)
Card files: `runs/e312_p_s0/ends_card_snap_thr0.083.json`, `ends_card_snap_nogeom_thr0.083.json`, `runs/e312_p_s1/ends_card_snap_thr0.09.json`, `ends_card_snap_nogeom_thr0.09.json`, `runs/e320_a_s0/ends_card_snap_nogeom_attach_thr0.12.json`, `runs/e320_a_s1/ends_card_snap_nogeom_attach_thr0.111.json` (each with its `.counts.pkl`). **Void criterion:** met after one correction. The scored curves are 12006 / 12209 (E312 at t1) and 11383 / 11845 (E320 at t2) against the counted 12006 / 12209 and 11383 / 11845. The first D2 read of seed 1 used the threshold 0.11 that the card printed, which was a display rounding of the count script (`%.2f` of the true grid point 0.111): it delivered 11885 curves (+0.30 % from the E312 target 11849, outside the 0.1 %), so it is void and was redone at 0.111 (11845 curves, `ends_card_snap_nogeom_attach_thr0.111.json`); the void read stays on disk as `ends_card_snap_nogeom_attach_thr0.11.json` (topo F1 0.6888, dup_frac 0.1641, junction F1 0.3011: the same within 0.0001 - 0.0006) and is not used. The card text above was corrected to 0.111.

| VAL | seed 0 c(0.1) | c(t1 = 0.083) | u'(t1) | **a(0.1)** | a(t2 = 0.12) | seed 1 c(0.1) | c(t1 = 0.09) | u'(t1) | **a(0.1)** | a(t2 = 0.111) |
|---|---|---|---|---|---|---|---|---|---|---|
| curves | 11378 | 12006 | 12006 | 12018 | 11383 | 11849 | 12209 | 12209 | 12208 | 11845 |
| topo F1 | 0.6825 | 0.6828 | 0.6708 | 0.6803 | 0.6801 | 0.6864 | 0.6868 | 0.6762 | 0.6889 | 0.6889 |
| dup_frac | 0.1747 | 0.1758 | 0.1824 | 0.1931 | 0.1919 | 0.1557 | 0.1562 | 0.1617 | 0.1651 | 0.1640 |
| junction F1 | 0.3194 | 0.3205 | 0.3045 | 0.2938 | 0.2940 | 0.3143 | 0.3153 | 0.2972 | 0.3005 | 0.3008 |
| F1@1.0 | 0.8669 | 0.8681 | 0.8673 | 0.8724 | 0.8715 | 0.8685 | 0.8692 | 0.8691 | 0.8741 | 0.8735 |
| vertices / 100 m | 19.55 | 19.51 | 19.71 | 19.31 | 19.36 | 20.10 | 20.08 | 20.25 | 19.84 | 19.88 |

**The key number (E312 at E320's count against E312 at p >= 0.1):** dup_frac +0.0011 [-0.0005, +0.0026] (s0) / +0.0005 [-0.0005, +0.0015] (s1); topo F1 +0.0003 [-0.0008, +0.0015] / +0.0004 [-0.0002, +0.0008]; junction F1 +0.0011 / +0.0010; lanes found +0.0059 [+0.0042, +0.0079] / +0.0032. **The count does not move the duplicates near p = 0.1:** neither H_count (+0.0072 / +0.0018 predicted; seed 0 outside the +-0.004 window) nor H_within (-0.0020 / -0.0005 predicted, hit within the window, but the sign is not confirmed: by the pre-registered cut-offs 0.1758 is 'inconclusive', between 0.1747 and 0.1787). The mirror agrees: E320 at E312's count has dup_frac 0.1919 (-0.0012 against its own 0.1931) and 0.1640 (-0.0011). The local slope is about +0.0002 duplicates per 1 % of curves for both networks, not the -0.00037 of the far points (p >= 0.5 and p >= 0.1) and not the +0.0013 between recipes.

**Rule (E320's, both directions, both seeds):**

| | seed 0 | seed 1 |
|---|---|---|
| D1 a(0.1) - u'(t1) (need +0.005) | +0.0095 [+0.0014, +0.0185] | +0.0127 [+0.0011, +0.0209] |
| D1 a - c(t1) topo F1 | -0.0025 [-0.0082, +0.0034] | +0.0022 [-0.0083, +0.0104] |
| D1 dup_frac a - c(t1) (guard +0.01) | **+0.0172 [+0.0120, +0.0216] BROKEN** | +0.0089 [+0.0031, +0.0149] held |
| D1 junction F1 a - c(t1) (guard -0.03) | -0.0267 [-0.0349, -0.0167] held by 0.0033 | -0.0148 [-0.0243, -0.0028] held |
| D1 F1@1.0 / vertices | +0.0043 / -0.20 held | +0.0049 / -0.24 held |
| D2 a(t2) - c(0.1) topo F1; a(t2) dup_frac - c(0.1) | -0.0023 [-0.0079, +0.0035]; **+0.0172 [+0.0115, +0.0217] BROKEN** | +0.0025 [-0.0080, +0.0107]; +0.0083 [+0.0025, +0.0141] held |
| D2 junction F1 a(t2) - c(0.1) | -0.0254 held | -0.0135 held |

**Verdict (by hand: replay): CUT in both directions, the E320 CUT stands.** V(D1) = CUT (dup_frac guard on seed 0: +0.0172 against 0.01; topo F1 passes the bound on both seeds, the other guards hold), V(D2) = CUT (the same guard, +0.0172). The combined verdict is CUT. The statement of the card (the duplicate excess is caused by the larger number of curves) is falsified: at an equal number of curves E320 has +0.0172 / +0.0089 more duplicates than E312 (intervals above 0), the same excess as at p >= 0.1 (+0.0184 / +0.0093).

**Predictions against the readings (hit = within +-0.004 for dup_frac, +-0.003 for topo F1 and junction F1):** dup_frac of E312 at t1: 0.1727 / 0.1552 predicted, 0.1758 / 0.1562 read (hit, both seeds); topo F1 0.6846 / 0.6877 predicted, 0.6828 / 0.6868 read (hit); junction F1 0.3222 / 0.3160 predicted, 0.3205 / 0.3153 read (hit); D1 dup guard a - c(t1) +0.0204 / +0.0099 predicted, +0.0172 / +0.0089 read (hit); D1 junction guard -0.0284 / -0.0155 predicted, -0.0267 / -0.0148 read (hit; seed 0 held by 0.0033, the guard did not break); D2 dup_frac 0.1951 / 0.1656 predicted, 0.1919 / 0.1640 read (hit). All predictions of the card held; the one the card did not make (the sign of the count effect at p = 0.1) is the finding: it is about zero. The seed-1 dup cell that the card called a coin flip fell on the side of 'held' (+0.0089).

**What it says.** The duplicate excess of E320 is not the number of curves; it belongs to the E320 network: its own keypoints read out with the same snap already carry +0.0087 over c (E320 section 3, 0.1834 against 0.1747), the attached ends with the snap's wiring add about +0.006 more (0.1931 against the run's own snap-nogeom 0.1873), and neither is near-duplicate connectors (the no-scoring check of section 1). So E320 stays CUT, the threshold is not the lever, and the open question about E320 is where its extra parallel length comes from (connector-to-B-piece parallels were not tested: candidates for a diagnostic, not planned). Together with E322 (the choice of the wired piece has no headroom) the wiring side of the delivered graph is closed for accuracy: the next levers are the pieces of B (28.7 / 26.5 % of the connector ends have no delivered piece in reach) and B's own read-out (E292). **Cost of the card:** six CPU scores of about 6 - 8 minutes under a training round; nothing on the GPU.

## 4. Second referee pass
Not needed after the answers above (the critic's own statement); a third pass only if the design changes.
