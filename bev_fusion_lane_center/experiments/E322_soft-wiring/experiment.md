# E322 - soft wiring: the delivered piece a connector end is joined to, chosen by the network instead of the nearest-piece rule (gate G0 on the headroom first)

Card of 2026-11-13, written by the lane chat on the lane lead's order (D86), replaces the stub of 00:10. **Order of events, stated first:** the gate G0 below was measured (06:35 - 07:50, CPU) BEFORE this card text was written; the gate level is the project's noise band (0.010), not a level chosen after the result, and the verdict is labelled "by hand: replay". The learned head is designed here and NOT built: the gate says it has nothing to gain on accuracy.

## 0. The card (what the tools run)
```json card
{
 "id": "E322",
 "card": "",
 "title": "headroom of the wiring (gate G0 of a learned wiring head): the snapped read-out with the delivered piece chosen by the nearest-piece rule at the matched GT end (oracle selection) against the rule at the keypoint end, E312 checkpoints, connectors p >= 0.1",
 "kind": "variant",
 "deterministic": true,
 "metric": "topo_F1",
 "direction": "higher",
 "band": 0.01,
 "prediction": {"value": 0.0, "vs": "c"},
 "arms": {
  "c": {"recipe": "control = E312 snapped, the piece chosen by the rule at the KEYPOINT end (`jwiring.py score`, variant c; reproduces `runs/e312_p_s<S>/ends_card_snap_thr0.1.json`: topo F1 0.6825 / 0.6864)"},
  "o": {"recipe": "`bash tools/env/_py14.sh jwiring.py score --run e312_p_s<S> --seed <S>`, variant o: the piece chosen by the same rule applied at the matched GT END (strict one-to-one, 4 m / 30 degrees); an end that matches no GT end or finds no piece at its GT end keeps the keypoint rule; the connector is drawn from the chosen piece ends as in the snap"}
 },
 "control": "c",
 "variant": "o",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "topo_F1", "vs": "c", "min_effect": 0.01},
  {"metric": "dup_frac", "vs": "c", "max_loss": 0.01}
 ],
 "budget_gpu_h": 0.0,
 "kill": "none (CPU read-out of stored dumps, no training)",
 "evaluate": {"split": "val"}
}
```
The rule is the gate G0: the oracle selection has to beat the rule by more than the band on BOTH seeds, with the duplicate guard held, before a learned selection is built for accuracy. Both seeds share the E312 checkpoints of their seed (no training seed in the comparison), so the paired scene bootstrap next to the point estimate is the interval (`jdiag.py boot <run> wiring_g0_o <run> wiring_g0_c`).

## 1. Card text
- **Statement:** choosing the delivered piece a connector end is wired to with a learned softmax over the pieces in reach (plus a dustbin) beats the nearest-piece rule of the snap by more than the noise band in topo F1. **Falsified at the gate:** an oracle that applies the rule at the matched GT end gains -0.0008 / +0.0014 over the rule (section 2); a head that is trained against that oracle cannot do better.
- **Mechanism (why it was asked):** on E312 the snap's gain over the unsnapped read-out (0.0387 / 0.0383 topo F1) is 69 % / 73 % WIRING (one nearest piece per end instead of every piece in reach: `--snap --snap-nogeom` 0.6706 / 0.6760 against 0.6438 / 0.6481 unsnapped) and 27 - 31 % end geometry (E320 replaced that part inside the network). The wiring is the last selection RULE of the delivered graph that is not learned (D55: hard steps only on the output and named). The keypoints are 1.2 - 1.4 m off the GT ends (`jdiag.py kperr`) and the median gap between the nearest and the second nearest piece end is 1.15 m, so a wrong choice by the rule looked plausible.
- **What the gate measured instead:** the choice is not the problem. Where there is a choice (40 % / 48 % of the entry / exit ends have two or more piece ends in reach, `jwiring.py reach`), the rule at the keypoint picks the same piece end as the rule at the GT end in 84 % of the cases and the other 16 % do not change the score; the reach of 4 m is the optimum of the rule (2 / 3 / 6 m are all lower); dropping the pairs with an end that matches no GT end lowers topo F1. The snap's wiring gain is a COUNT effect (one piece instead of every piece in reach), which a learned selection reproduces at best.
- **Inputs:** `evaluation/jwiring.py` (committed 673ed625, selftest in `run_gates.sh`), the dumps of E312 (`runs/e312_p_s<S>/ends_val.pkl`), B's full dump (`temp/e276/predfull_e273_p_s<S>.pkl`), the GT ends of the VAL frames (`jwiring.py gt-ends`).
- **Starts from:** E312 (the system of record), E320 for the attached-end reading.
- **Lens / group / status:** E2E differentiability (D55) of the delivered graph; closed at the gate, 2026-11-13.
- **Referee:** see section 4.

### The design of the learned wiring (recorded, NOT built: what a later card would have to say)
Only to be taken up if the end-to-end principle is weighted above simplicity (D86 weights: detection up, simple, little compute, end-to-end, planner-ready: the head would be one more module that ties a rule).
- **Base and confound:** the frozen E312 checkpoints, not the E320 recipe. E320's geometry gain is real (the recipe is +0.0075 / +0.0089 over u' through the geometry loss, the attached ends add +0.0022 / +0.0040), but its CUT came from the duplicate guard (+5.6 % / +3.0 % curves at the same threshold; E323 asks how much of it is the count), so a head on top of it would test wiring and curve count at once. E320's PRIMARY read-out (rule-wired) stays in the table as the geometry-recipe reference.
- **What is trained:** a softmax over the delivered piece ends in reach (the same gate: 4 m / 45 degrees, at most 8, an index choice) plus a dustbin ('no piece'), scored by a small MLP (about 10 k parameters like `Attach`) from the distance, the along / across offset and the heading difference of the piece end, the length and kind of the piece (lane / link), B's existence and link probabilities of it, the keypoint's kind and probability and the partner end's geometry. Target: the piece end the rule picks at the matched GT end, dustbin for an end that matches no GT end or has no piece at its GT end; cross entropy. The delivered pieces are a frozen function of B, so they can be cached per TRAIN frame once; the keypoints of TRAIN frames are the pair head's own (a mismatch to VAL noise, to be handled by a keypoint jitter of the measured error). Joint training with the pair head is the follow-up; the first step trains the head on frozen inputs.
- **Read-out:** the softmax weights are the edge probabilities that go to the output (planner-ready); the hard step that stays, named, is the argmax per end (the dustbin = the end is not wired); the p >= 0.1 threshold, the keypoint peak choice, the candidate gate, B's close-node step and the section solve stay as named hard steps.
- **Controls:** c (E312 snapped, rule), u' (snap without the end geometry, the floor), the oracle o (the ceiling), E320 PRIMARY (rule-wired, attached ends) as the reference of the geometry recipe.
- **Rule it would carry:** equal accuracy, not a gain: topo F1 of the learned selection at least c - 0.005 on both seeds with the guards F1@1.0, dup_frac (+0.01), vertices, junction F1 held, and an agreement with the oracle choice above the rule's own 84 %; two seeds, VAL, fresh critic before any build. Cost to state: about +0.01 M parameters and about 4 MMACs (the size of `Attach`: 9.6 k parameters, 3.9 MMACs), and the host time of the candidate lists (not measured; the pair head is already host-bound, 9.7 ms for 0.02 GMACs).
- **The cheaper form first:** a parameter-free soft assignment (softmax of the negative squared distance with a dustbin prior) whose argmax is the rule exactly (accuracy equal to c by construction; selftest) and whose weights are the edge probabilities; a head is only justified if those weights are mis-calibrated against the oracle choice.

## 2. Result (gate G0, VAL 873 frames, E312 checkpoints, p >= 0.1; `jwiring.py score`, CPU; `runs/e312_p_s<S>/wiring_g0.json`)
Variant c reproduces the card of record exactly (0.6825 / 0.6864).

| snapped read-out, piece chosen by | topo F1 s0 | s1 | dup_frac s0 | s1 | junction F1 s0 | s1 | lanes found s0 | s1 |
|---|---|---|---|---|---|---|---|---|
| **c: the rule at the keypoint end** | 0.6825 | 0.6864 | 0.1747 | 0.1557 | 0.3194 | 0.3143 | 0.7376 | 0.7423 |
| **o: the rule at the matched GT end (oracle)** | 0.6817 | 0.6878 | 0.1758 | 0.1559 | 0.3261 | 0.3205 | 0.7365 | 0.7439 |
| o - c | **-0.0008** | **+0.0014** | +0.0011 | +0.0002 | +0.0067 | +0.0062 | -0.0011 | +0.0016 |
| o_drop: and no pair with an unmatched end | 0.6795 | 0.6830 | 0.1734 | 0.1482 | 0.3212 | 0.3157 | 0.7088 | 0.7081 |
| r2: the rule with reach 2 m | 0.6662 | 0.6729 | 0.2477 | 0.2200 | 0.3123 | 0.3037 | 0.7593 | 0.7600 |
| r3: reach 3 m | 0.6752 | 0.6826 | 0.2101 | 0.1892 | 0.3183 | 0.3109 | 0.7519 | 0.7536 |
| r6: reach 6 m | 0.6798 | 0.6828 | 0.1656 | 0.1464 | 0.3174 | 0.3113 | 0.7350 | 0.7372 |

Counts (seed 0 / 1): 22756 / 23698 connector ends, of them 16917 / 17368 match a GT end (74 % / 73 %), 12683 / 13203 find a piece end at their GT end (56 %), 12453 / 12949 of those also find one with the rule at the keypoint and 10482 / 10904 (84.2 % / 84.2 %) are the same piece end. Candidates in reach (4 m / 45 degrees) of the entry / exit ends of E312 seed 0: none 28.7 % / 26.5 %, one 31.7 % / 25.6 %, two 20.5 % / 19.6 %, three or more 19.1 % / 28.3 %; second minus nearest distance (ends with at least two): median 1.15 m, 43 % under 1 m.

**Paired scene bootstrap o - c** (22 scenes, 1000 resamples, `jdiag.py boot <run> wiring_g0_o <run> wiring_g0_c`): topo F1 -0.0007 [-0.0030, +0.0011] (s0) / +0.0013 [-0.0003, +0.0032] (s1); dup_frac +0.0011 [-0.0026, +0.0043] / +0.0001 [-0.0034, +0.0035]; junction F1 +0.0067 [+0.0029, +0.0119] / +0.0063 [+0.0035, +0.0090], the one metric where the oracle selection is distinguishable from zero and still small (0.006 against the junction guard of 0.03).
**E320 attached ends** (`jwiring.py score --run e320_a_s<S> --seed <S> --attach`: the connector ends are the attached ends of the dump, `c` = the rule at the attached end, `o` = the rule at the matched GT end): topo F1 c 0.6880 / 0.6948, o 0.6888 / 0.6967 (o - c +0.0009 [-0.0009, +0.0027] / +0.0019 [+0.0002, +0.0037]), dup_frac c 0.1870 / 0.1605, o 0.1870 / 0.1633, junction F1 o - c +0.0083 [+0.0049, +0.0116] / +0.0040 [+0.0012, +0.0068]; the rule picks the oracle's piece end in 84.6 % / 84.9 % of the ends where both find one (10855 / 12834, 11062 / 13034); reach 2 / 3 / 6 m again lower than 4 m (s0 0.6716 / 0.6824 / 0.6864 against 0.6880; s1 0.6800 / 0.6899 / 0.6925 against 0.6948). The same ceiling with the attached ends.

**Verdict by the gate (by hand: replay): CUT by the gate, the learned selection is not built for accuracy.** o - c is -0.0008 / +0.0014 against the +0.010 the rule demands; the duplicate guard holds. The numbers would need to be an order of magnitude larger to matter, so the late fixing of the level (the band) does not decide the verdict.

## 3. What it says and what it does not
The nearest-piece rule is at the ceiling of what choosing among the pieces in reach can give. The 69 % 'wiring' share of the snap is the effect of wiring ONE piece instead of all of them, which the rule already does. What limits the topology of the delivered graph is elsewhere: 28.7 % / 26.5 % of the entry / exit ends have no delivered piece end in reach (a lane that B does not deliver cannot be wired), 26 % of the ends match no GT end (false keypoints), and 25 % of the ends that match a GT end find no piece at it. The levers that the numbers point to are the pieces themselves (lanes found 0.74: the pixel / node / link stages of B), the number of curves at the threshold (E323), and the hard steps of B's own read-out (the close-node step: E292).
**Weaknesses of the gate:** (1) 'correct' is defined as the rule at the GT end; a wiring that is correct in another sense (the piece whose lane continues the GT lane, several pieces where a junction has two) is not covered, o_drop and the reach variants are weak proxies for a subset choice; (2) the oracle cannot act on the 26 % unmatched and 25 % no-piece ends, where a learned dustbin could in principle help, which o_drop (dropping the pairs with an unmatched end) does not show; (3) two networks (E312 with its keypoint ends, E320 with its attached ends), two seeds each, and one threshold (0.1); (4) the pieces are B's hard read-out, so 'the piece' is itself a product of the close-node step; (5) the gate was measured before the card was written.

## 4. Consequences and referee
Consequences (D86): E322 closes at the gate; the learned head stays designed above, not built; HANDOVER roadmap: E323 (duplicates against the curve count) next, E292 (Matrix-NMS in place of the close-node step) after it; the E316 pair stage (6000 steps) stays a stack candidate. Referee: the fresh `ar-critic` pass (cards mode) on this card is written below when it has run.
