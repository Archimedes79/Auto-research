# 10. Details of the results (the stage-1 chapters, kept for the numbers)

Every number comes from a stored card section or a ledger row. VAL = 873 frames, 22 scenes; seeds are written "s0 / s1"; TEST is untouched. Cells marked PENDING are not measured yet (S4, the split by crowded junctions).

**Metrics** (as in the earlier report): **topo F1** = directed reachability from seeds every 5 m on the GT (30 m reach, match within 1 m); **lanes found** = GT lanes with at least 80 % of their points within 1 m; **F1@1.0** = point F1 within 1 m; **junction F1** = fork / merge points within 3 m; **dup_frac** = delivered length running parallel within 0.75 m of a non-neighbour piece.

**In five lines.** (1) The ladder of end-to-end systems of this project: plain lane head B 0.626 / 0.625, E2E with rule wiring 0.6912 / 0.6948, learned wiring 0.6931 / 0.6964, stack with the soft layer 0.6820 / 0.6854 (topo F1, VAL). (2) **The champion**, the token head on the predicted nodes of the frozen lane head, trained with the none target and the none class at weight 0.25 and read at a decode chosen on its TRAIN sample (link 0.2, keep 0.6, none gate off): on two fresh head seeds topo F1 0.6698 / 0.6769, F1@1.0 0.846 / 0.846, lanes found 0.727 / 0.727, +0.044 / +0.052 over the plain head with intervals excluding 0 (E373). **Its final decode is per class** (short lane links 0.3, long connector links 0.2, keep 0.6, none gate off; E375, D115): topo F1 **0.6679 / 0.6762**, F1@1.0 0.846 / 0.847, lanes found 0.719 / 0.721, junction F1 0.252 / 0.245. The first recipe (none weight 1.0, link 0.1) is the confirmed runner-up: 0.6617 / 0.6767 and 0.6618 / 0.6733 (E366, E371), a tie inside the band with fewer lanes found (0.68). (3) It is cheaper than the previous system: the whole path takes 16.6 - 16.8 ms against 19.7 ms and 2.323 M parameters against 2.741 M, with one network stage and none of the five hard steps. (4) Plainly: precision 0.68, 23.8 - 24.2 vertices per 100 m at the final decode (27.4 - 27.8 at the single threshold; plain head 19), duplicates 0.061 / 0.064 (plain head 0.235) and a **junction F1 of 0.25 (0.20 before the per-class decode), still below the plain head's 0.27**: topology up, junction detection not yet at the plain head's level; four tuned output thresholds. (5) It is stage S1 with the lane head frozen; PENDING: S4 and TEST (none exists); S2 (past) and S3 (map) are measured, both a TIE.

## A. The ladder: what each end-to-end step gave (topo F1, VAL, s0 / s1)

![The ladder of end-to-end systems and the token head, topo F1 on VAL for both seeds. The dashed line is the written floor of the gate (plain lane head + 0.01). Look at the token head at the decode of record (left of the line) against the same head at a decode chosen on the TRAIN sample and against the champion, and at the earlier systems, which keep five hard steps. The axis starts at 0.5.](parts/figures/ladder.png)

Table: The ladder of end-to-end systems of this project (topo F1, VAL 873 frames, seed 0 / seed 1). Each step removes or replaces a hard step; rule-step systems are not in the table.

| step | system | topo F1 | what is hard in it | source |
|---|---|---|---|---|
| 0 | plain lane head B (own sections, close-node step) | 0.6259 / 0.6249 | thresholds, top-K nodes, close-node step | E275 / stored base card |
| 1 | E324 s, end-to-end read-out, rule wiring | 0.6912 / 0.6948 | nearest-piece wiring, close-node step, keypoint peaks + top-K 32, thresholds, candidate gates | E324 card |
| 2 | E324 s with the learned wiring head (E358, +545 parameters, about +2 ms on the host) | 0.6931 / 0.6964 | the wiring is a learned argmax; the rest as step 1 | E358 card, section 6 |
| 3 | the stack: learned wiring + soft Matrix-NMS layer instead of the close-node step (E363) | 0.6820 / 0.6854 | two hard steps replaced; the layer breaks the duplicate guard on both seeds | E363 card, section 6 |
| 4 | **token head S1 on the predicted nodes of B' (E365)** | **0.5486 / 0.5743** | top-K 512 (budget), allowed-pair gate, output thresholds | E365 card, section 3 |
| 5 | E366: the first recipe (none weight 1.0), the decode chosen on the TRAIN sample, applied to VAL once | 0.6617 / 0.6767 | top-K 512 (budget), allowed-pair gate, output thresholds | E366 card, section 3 |
| 6 | E371: the first recipe retrained with fresh head seeds 2 / 3, the decode frozen | 0.6618 / 0.6733 | as step 5 | E371 card, section 3 |
| 7 | the same head with the none weight 0.25 on fresh head seeds 2 / 3, its own TRAIN-chosen decode (E373) | 0.6698 / 0.6769 | as step 5, other threshold values | E373 card, section 3 |
| 8 | **the champion at its final decode**: the same head, per-class link thresholds (short 0.3, long 0.2), no retraining (E375, D115) | **0.6679 / 0.6762** | as step 5, four output thresholds (short link, long link, keep, none gate off) | E375 card, section 3 |

Rule-step systems (the snapped record, 0.6981 / 0.7010) are not in the solution set and are no bar (D113 point 2); they are not in this table.

## B. Stage S1 (E365) in detail: the written gate and the controls
The gate "okayish" was written before the run (D113 point 4; floors per seed from the stored base card of E324 s):

Table: The written gate for an "okayish" champion (D113), per seed, and what stage S1 reached.

| condition | floor (s0 / s1) | S1 | holds? |
|---|---|---|---|
| G-topo: topo F1 at least plain B + 0.01 | 0.6359 / 0.6349 | 0.5486 / 0.5743 | no, on both seeds |
| G-found: F1@1.0 not below B by more than 0.01 | 0.8182 / 0.8060 | 0.7524 / 0.7631 | no, on both seeds |
| G-found: lanes found not below B by more than 0.02 | 0.6066 / 0.5937 | 0.5001 / 0.5252 | no, on both seeds |
| G-cost: whole path at most E324 s (2.741 M parameters, 19.7 ms) | 2.741 M | 2.32 M parameters (B' 2.196 M + head 0.127 M); ms measured on a host that was not quiet (load 6.4: head 7.6, B' 12.7, whole 19.0 ms), not usable, to be repeated | parameters yes, ms open |
| G-simple: at most two stages, none of peaks + top-K 32 / snap / attachment / wiring rule / close-node step | | one stage; none of them | yes |

**Verdict by the written rule at the decode of record: MISS** (G-topo and G-found fail on both seeds). The paired scene-bootstrap difference to plain B is -0.077 [-0.113, -0.044] (s0) and -0.051 [-0.085, -0.019] (s1).

Table: Stage S1 against its controls: plain B, the same nodes and links read with sections 0, a one-line rule on the same tokens, the ceiling of the token set (GT links over the assigned tokens), and an untrained head.

| VAL 873, delivered graph, every section 0 | topo F1 (s0 / s1) | topo P / R (s0, s1) | F1@1.0 | lanes found |
|---|---|---|---|---|
| plain B, stored (own sections, close-node step) | 0.6259 / 0.6249 | | 0.8282 / 0.8160 | 0.6266 / 0.6137 |
| control c-B-sec0 (B' nodes and links, sections 0, close-node step) | 0.6161 / 0.6198 | | 0.8313 / 0.8302 | 0.6534 / 0.6476 |
| control c-rule (rule links on the B' tokens with e >= 0.5) | 0.6265 / 0.6281 | | 0.773 / 0.773 | 0.586 / 0.586 |
| control c-hi (GT links over the assigned tokens: the ceiling of the token set) | 0.8191 / 0.8202 | | 0.908 / 0.909 | 0.816 / 0.818 |
| control c-floor (untrained head) | 0.0125 / 0.0069 | | 0.101 / 0.059 | 0.012 / 0.007 |
| **S1, decode of record** | **0.5486 / 0.5743** | 0.858 / 0.403 and 0.853 / 0.433 | 0.7524 / 0.7631 | 0.5001 / 0.5252 |
| S1 on a TRAIN sample (873 evenly spaced frames) | 0.6099 / 0.6245 | 0.863 / 0.472 and 0.866 / 0.488 | 0.828 / 0.835 | 0.614 / 0.628 |
| S1, degraded read-out (17 % of the tokens removed) | 0.4183 / 0.4296 | | 0.693 / 0.704 | 0.368 / 0.385 |
| E366 / E367 / the champion | see section B2 | | | |

**Why it missed (diagnosis in-sample on VAL, next to the verdict, never the verdict; E365 section 3).** The delivered graph is precise (0.86) and its recall is low (0.40 - 0.43; 0.47 - 0.49 on the TRAIN sample): the head is conservative and does not even fit its own training frames (TRAIN topo F1 0.610 / 0.625, below the floor). B' is cleaner on TRAIN (node recall 0.926 / 0.923 against 0.862 / 0.864 on VAL), which adds 0.05 - 0.06 of train-validation gap. Nodes: of the tokens assigned to a GT node 86.8 % are kept; the dropped ones have a weak existence in B' itself, so the pruning is not worse than B' but it adds no recall. Links: of the sources with a GT successor among the candidates 86 % get a link in the short class, **only 51 % in the long (junction) class**; of the missed links half are missed because the source is not kept and 40 % because no token of the GT group is kept. The decode of record and the argmax alternative differ by only 0.01 - 0.02, so the decode is not the main lever; the head sits at a loss plateau (0.42 over the last 2000 steps).

**Checks.** The scorer path reproduces the stored plain-B base of E324 s to 1e-6 on both seeds (differences exactly 0). The parity check (cache path against a live CPU path of B', 20 VAL frames) fails by its tolerance on 0.1 % of the kept nodes and 0.2 % of the links, caused by the cumulative, order-sensitive Matrix-NMS layer; the numbers belong to the cache path; a GPU live check is open. The prediction missed on topo F1 (about 0.665 predicted), F1@1.0, lanes found and the gap share (the head is worse than the rule, g1 = -0.40 / -0.28); the ceiling prediction hit.

## B2. The champion and its runner-up: the token head at a decode chosen on the TRAIN sample (E366, E371, E373)
For each head a decode (link threshold, keep threshold, none gate) was chosen by a written rule on the 291-frame TRAIN sample of that head, **before any VAL score of a variant**, and applied once to VAL: the first recipe at link 0.1 / keep 0.6 / gate off, the champion at link 0.2 / keep 0.6 / gate off. A decode is a change of thresholds on the output only; the forward pass is that of the trained head.

Table: Stage S1 at the decode of record and at the chosen decode, against plain B and its controls (VAL 873 frames, seed 0 / seed 1).

| VAL 873, delivered graph | topo F1 | topo P / R | F1@1.0 | lanes found | junction F1 | dup_frac | vertices / 100 m |
|---|---|---|---|---|---|---|---|
| floors of the gate | >= 0.6359 / 0.6349 | | >= 0.8182 / 0.8060 | >= 0.6066 / 0.5937 | | | |
| plain B (stored) | 0.6259 / 0.6249 | 0.670 / 0.587 (s0) | 0.8282 / 0.8160 | 0.6266 / 0.6137 | 0.272 (s0) | 0.235 (s0) | 19.29 (s0) |
| one-line rule on the same tokens | 0.6265 / 0.6281 | | 0.773 / 0.773 | 0.586 / 0.586 | 0.163 | 0.005 / 0.004 | |
| S1, decode of record | 0.5486 / 0.5743 | 0.858 / 0.403 and 0.853 / 0.433 | 0.752 / 0.763 | 0.500 / 0.525 | 0.177 / 0.197 | 0.004 | 18.5 / 18.6 |
| **S1, chosen decode (E366), seed 0** | **0.6617** | 0.694 / 0.633 | **0.8359** | **0.6747** | 0.209 | 0.055 | 25.2 |
| **S1 first recipe, chosen decode (E366), seed 1** | **0.6767** | 0.687 / 0.666 | **0.8422** | **0.6905** | 0.204 | 0.054 | 25.7 |
| S1 first recipe, fresh heads (E371), seed 0 / 1 | 0.6618 / 0.6733 | 0.684 / 0.641 and 0.683 / 0.664 | 0.8402 / 0.8398 | 0.6849 / 0.6846 | 0.205 / 0.213 | 0.059 / 0.063 | 25.3 / 25.2 |
| none weight 0.25 (E367 b_bal25, selection heads), seed 0 / 1 | 0.6731 / 0.6728 | | 0.849 / 0.845 | 0.741 / 0.725 | 0.194 / 0.197 | 0.056 / 0.061 | 29.1 / 27.9 |
| **CHAMPION: none weight 0.25, fresh heads (E373), cell 0.2 / 0.6 / off, seed 0** | **0.6698** | 0.680 / 0.660 | **0.8456** | **0.7274** | 0.203 | 0.058 | 27.4 |
| **CHAMPION (E373), seed 1** | **0.6769** | 0.681 / 0.673 | **0.8461** | **0.7267** | 0.197 | 0.061 | 27.8 |
| **CHAMPION at the final per-class decode (E375, D115), cell 0.3 / 0.2 / 0.6 / off, seed 0** | **0.6679** | | **0.8459** | **0.7191** | **0.252** (P 0.157 / R 0.636) | 0.061 | 23.8 |
| **CHAMPION at the final per-class decode (E375), seed 1** | **0.6762** | | **0.8467** | **0.7205** | **0.245** (P 0.152 / R 0.632) | 0.064 | 24.2 |
| champion at the record decode (0.3 / 0.5 / gate on), seed 0 / 1 | 0.6424 / 0.6581 | 0.776 / 0.548 and 0.780 / 0.569 | 0.8162 / 0.8203 | 0.641 / 0.648 | 0.248 / 0.241 | 0.017 / 0.017 | 22.4 / 22.9 |

**Against the gate:** topo F1 0.6617 / 0.6767 against the floors 0.6359 / 0.6349 holds on both seeds (+0.026 / +0.042 over the floor); F1@1.0 0.836 / 0.842 against 0.818 / 0.806 and lanes found 0.675 / 0.691 against 0.607 / 0.594 hold; the parameters of the whole path (2.32 M against 2.741 M) hold; the milliseconds are open (the only measurement, 19.0 ms against 19.7 ms, was taken on a loaded host and is not usable); one training stage and no rule step in the delivered graph: the simplicity condition holds. **Paired scene bootstrap (topo F1, 1000 draws):** against plain B +0.036 [+0.018, +0.052] and +0.052 [+0.032, +0.070]; against the decode of record +0.113 [+0.090, +0.140] and +0.102 [+0.081, +0.129]; every interval excludes 0.

**What it costs, honestly.** Precision falls from 0.86 to 0.69; the duplicate share rises from 0.004 to 0.055 (still far below the plain head's 0.235); the vertex density rises (25 against 19 per 100 m); the junction F1 (0.209 / 0.204) stays below the plain head's 0.272, so the junction links remain the weak part; on the TRAIN sample the same decode scores 0.720 / 0.725, i.e. the TRAIN-to-VAL drop is 0.06 / 0.05.

**Confirmation on fresh heads (E371).** Two heads trained with fresh seeds (2 and 3, on the token caches 0 and 1) and read at the same frozen decode give topo F1 0.6618 / 0.6733 (F1@1.0 0.8402 / 0.8398, lanes found 0.685 / 0.685, junction F1 0.205 / 0.213, duplicates 0.059 / 0.063, 25.3 / 25.2 vertices per 100 m); against plain B +0.036 [+0.020, +0.049] and +0.048 [+0.030, +0.066], both excluding 0. The head seed alone moves topo F1 by +0.0001 and -0.0034, less than the band; the tightest condition, F1@1.0 on cache 0, holds with +0.022. On the TRAIN sample the same decode scores 0.7128 / 0.7255, so the TRAIN-to-VAL drop is 0.051 / 0.052.

**The champion (E373).** The same head trained with the none target and the none class at weight 0.25 (one loss weight, nothing added), on two fresh head seeds, read at the cell its recipe chooses on the TRAIN sample (link 0.2, keep 0.6, none gate off): topo F1 0.6698 / 0.6769, F1@1.0 0.8456 / 0.8461, lanes found 0.7274 / 0.7267; against plain B +0.044 [+0.028, +0.059] and +0.052 [+0.034, +0.068]. At the record decode (no tuned thresholds) it misses the gate on cache 0 by 0.002 in F1@1.0 (0.8162 against 0.8182) and is the cleanest operating point of the night (junction F1 0.24 - 0.25, duplicates 0.017, 22.4 - 22.9 vertices). Against the first recipe the champion has the same topo F1 inside the band (+0.008 / +0.002), a slightly higher F1@1.0 (+0.008 / +0.005) and clearly more lanes found (+0.048 / +0.039), at slightly more vertices (27.4 - 27.8 against 25.2 - 25.5), the same duplicates and junction F1; the rule written before the data (a quality tie with the first champion on both caches) therefore **replaces** the first recipe. Said plainly: b_bal25 was named among the arms read on VAL; VAL has been read by every arm of the night; the confirmation removes the selection of the head instance, not the reuse of VAL; the seed-twin interval and the frame-order check of the card are not done yet.

**Caveats, as the card states them.** The decode was selected on the TRAIN sample of the same heads (the heads were trained on those frames) and is one of 24 cells; the paired intervals use plain B's stored per-frame records; the cache-against-live parity check of E365 failed its tolerance on 0.1 % of the nodes; the champion is declared by the written rule after the confirmation on two fresh head seeds (E373) and the timing (section D, measured on the first recipe's head, identical architecture); it is **staged, lane head B' frozen**.

## B3. Two follow-up experiments of the last hours: the rule-distilled start (E370) and the per-class decode (E375)

**E370, MISS: a head that starts from the rule does not stay there.** Stage A taught the token head the one-line rule as labels, with no ground truth at all (3000 steps); on TRAIN and on VAL it reproduces the rule (VAL topo F1 0.6221 / 0.6244 against the rule's 0.6265 / 0.6281). Stage B then trained it on the ground-truth loss alone (12000 steps, the same loss as the champion with none weight 1.0). Within 500 steps the head fell from the rule's operating point (precision 0.74, recall 0.57 on the TRAIN sample) to the conservative point of the first S1 head (precision 0.85, recall 0.40) and recovered only slowly.

Table: E370 on VAL (873 frames, seed 0 / seed 1); the written gate is the one of chapter 10, section B.

| VAL 873 | topo F1 | F1@1.0 | lanes found | verdict of the card |
|---|---|---|---|---|
| floors of the gate | >= 0.6359 / 0.6349 | >= 0.8182 / 0.8060 | >= 0.6066 / 0.5937 | |
| stage A (the rule distilled), record decode | 0.6221 / 0.6244 | 0.784 / 0.788 | 0.572 / 0.574 | reproduces the rule |
| stage B, record decode | 0.5409 / 0.5535 | 0.744 / 0.753 | 0.494 / 0.509 | **MISS** (topo F1 and lanes found fail on both seeds) |
| stage B at its TRAIN-chosen cell (link 0.1 / keep 0.6 / gate off), diagnosis | 0.6585 / 0.6627 | 0.835 / 0.838 | 0.681 / 0.682 | at the level of the first recipe (0.6617 / 0.6767 at the same cell) |

Table: The drift of stage B on the 291-frame TRAIN sample (topo F1, record decode).

| stage-B step | 0 | 500 | 1000 | 3000 | 6000 | 12000 |
|---|---|---|---|---|---|---|
| topo F1, seed 0 | 0.6441 | 0.5450 | 0.5469 | 0.5731 | 0.5918 | 0.6023 |
| topo F1, seed 1 | 0.6349 | 0.5292 | 0.4948 | 0.5866 | 0.6004 | 0.6122 |

What it says: the ground-truth loss with its none slot pulls the head to a conservative operating point whatever it starts from; the operating point, not the initialisation, is the lever, and the champion moves it with one loss weight (none weight 0.25) or with the decode. The rule-distilled start gives no advantage over the champion (0.6698 / 0.6769).

**E375, ADOPT-DECODE: the final per-class decode of the champion.** The weak number of the champion is the junction F1 (0.20 against the plain head's 0.27). The junction F1 of the scorer counts fork and merge points (degree >= 3), so false forks cost precision; duplicate lane forks come from short lane links that are too permissive. E375 therefore sets the threshold of short (lane) links and of long (connector) links separately. The cell was chosen among 36 on the TRAIN tables of the E367 heads (a rule written before: topo F1 within 0.005 of the best eligible cell, duplicates and lanes found guarded, the highest junction F1 wins) and scored once on VAL, with no training: on the two E373 fresh heads, which played no part in the choice, and on the E367 heads that chose it.

Table: The champion's decode, one threshold against per class (VAL 873 frames; the fresh heads of E373 and the selection heads of E367).

| head | decode (short / long / keep) | topo F1 | F1@1.0 | lanes found | junction F1 (P / R) | dup_frac | vertices / 100 m |
|---|---|---|---|---|---|---|---|
| fresh head, cache 0 | 0.2 / 0.2 / 0.6 | 0.6698 | 0.8456 | 0.7274 | 0.203 (0.119 / 0.689) | 0.058 | 27.4 |
| fresh head, cache 0 | **0.3 / 0.2 / 0.6** | **0.6679** | **0.8459** | **0.7191** | **0.252** (0.157 / 0.636) | 0.061 | 23.8 |
| fresh head, cache 1 | 0.2 / 0.2 / 0.6 | 0.6769 | 0.8461 | 0.7267 | 0.197 (0.115 / 0.680) | 0.061 | 27.8 |
| fresh head, cache 1 | **0.3 / 0.2 / 0.6** | **0.6762** | **0.8467** | **0.7205** | **0.245** (0.152 / 0.632) | 0.064 | 24.2 |
| selection head, seed 0 (chose the cell) | 0.2 / 0.2 / 0.6 and 0.3 / 0.2 / 0.6 | 0.6731 and 0.6709 | 0.847 and 0.848 | 0.729 and 0.722 | 0.204 and 0.258 | 0.053 and 0.056 | 27.6 and 24.1 |
| selection head, seed 1 (chose the cell) | 0.2 / 0.2 / 0.6 and 0.3 / 0.2 / 0.6 | 0.6728 and 0.6730 | 0.845 and 0.846 | 0.725 and 0.718 | 0.197 and 0.247 | 0.061 and 0.064 | 27.9 and 24.2 |

Paired scene bootstrap on the fresh heads (new decode minus the single threshold, cache 0 / cache 1): junction F1 **+0.049 [+0.042, +0.055]** / **+0.048 [+0.041, +0.054]**; topo F1 -0.0019 [-0.0038, 0.0000] / -0.0007 [-0.0028, +0.0014]; F1@1.0 +0.0003 / +0.0006; lanes found -0.0083 [-0.0112, -0.0058] / -0.0062 [-0.0081, -0.0043]. All written conditions hold on both fresh heads (topo F1 and F1@1.0 not lower by more than 0.005, lanes found not lower by more than 0.01, junction F1 at least +0.03 with the paired interval above 0, the gate floors and the tie conditions of E373 at the new cell). The prediction of the card (a threshold above 0.2, junction F1 +0.02 with a window of -0.01 to +0.05, adoption about 20 %) was met: the lever is the short threshold, the gain +0.049 / +0.048 lies at the top of the window.

What it costs, plainly: **four tuned output thresholds** (short link, long link, keep, none gate off; all on the output and named); the junction recall falls from 0.68 to 0.63 while its precision rises from 0.12 to 0.15 - 0.16; the duplicate share rises slightly (0.058 / 0.061 to 0.061 / 0.064). The gap to the plain head's junction F1 (0.272) shrinks from about 0.07 to about 0.02 - 0.03 and is **not closed**. VAL has been read by every experiment of this series; the cell was chosen on TRAIN and read once per head.

## B4. Stage S2 (E368): the past does not pay

**What the S2 runs ask.** The champion reads one frame and one radar scan; S2 adds the one thing D91 allows from the past: the head's own token states of the previous keyframe, warped into the current ego frame and read by one cross-attention block after self-attention layer 1 (+17.4 k parameters, no rule step). The question: does the head that may read its own past wire the lanes better than the same head trained on the same sequences without it? Three arms, five runs, on the two caches: `z` (the control, block disabled), `r` (the block) and `z2` (a second initialisation of the control that sets the noise reference). Training sees sequences of three keyframes; the read-out streams the state through a scene and resets it at the 22 VAL scene starts; each arm is read at the cell chosen on its own TRAIN sample, once, on VAL (873 frames); TEST is untouched. The rule was written before the runs: KEEP needs r - z >= +0.010 topo F1 on both seeds with the paired interval above 0 and no guard broken; a TIE keeps the cheaper variant. The prediction was +0.004 (window -0.006 to +0.020; odds KEEP 15 %, TIE 65 %, CUT 20 %). The arms are the S1 recipe with none weight 1.0, not the b_bal25 champion; a S2 on the champion recipe would be a new run.

Table: S2 on VAL (873 frames, stream read-out, each arm at its own TRAIN-chosen cell: link 0.1 / keep 0.6 / gate off for z and r on both seeds).

| arm | topo F1 | F1@1.0 | lanes found | junction F1 | dup_frac | AP of the existence |
|---|---|---|---|---|---|---|
| z seed 0 | 0.6536 | 0.8368 | 0.6845 | 0.2100 | 0.0614 | 0.9416 |
| r seed 0 | 0.6579 | 0.8383 | 0.6849 | 0.2080 | 0.0552 | 0.9439 |
| **r - z seed 0 (paired interval)** | **+0.0042 [-0.0010, +0.0090]** | +0.0016 [-0.0014, +0.0050] | +0.0004 [-0.0065, +0.0080] | -0.0019 [-0.0098, +0.0058] | -0.0062 | +0.0023 |
| z seed 1 | 0.6621 | 0.8390 | 0.6889 | 0.2028 | 0.0594 | 0.9417 |
| r seed 1 | 0.6644 | 0.8419 | 0.7020 | 0.2025 | 0.0612 | 0.9450 |
| **r - z seed 1 (paired interval)** | **+0.0023 [-0.0028, +0.0069]** | +0.0029 [+0.0010, +0.0051] | +0.0131 [+0.0077, +0.0185] | -0.0003 [-0.0056, +0.0055] | +0.0018 | +0.0033 |
| z2 (seed twin) - z seed 0 | -0.0033 [-0.0093, +0.0026] | -0.0056 | -0.0279 | +0.0139 | | |

**Verdict: TIE, the champion takes no past block.** Both differences are below the bar and their intervals include 0; the noise reference 0.0033 is below the bar, so the bar stays 0.010; AP, F1@1.0 and the duplicate share hold; the kill (the head ignores the past) did not fire (u 0.21 / 0.33). The prediction hit (+0.0042 / +0.0023 inside the window). The lanes-found gain of seed 1 is inside the head-seed noise (the twin differs by -0.028); only topo F1 and F1@1.0 are tight.

**What the checkpoint controls say (seed 0, the same r checkpoint).** Cold (every frame without a past) 0.6561, so the right past gains +0.0018 [-0.0021, +0.0056]; the state of another scene at the same step index 0.6320 (stream minus shuffled +0.0259 [+0.0167, +0.0353]); the identity pose instead of the ego pose 0.6540 (+0.0038); truncated windows of three 0.6568 (stream minus truncated +0.0010: no drift). **The head does read the past** (a wrong past costs 0.026), **but the right past adds only about +0.002.** Against its own cold mode the past changes the kept nodes of 91 % (93 %) of the frames by 8.1 (11.5) flips per frame, 4.6 / 5.8 gained and 3.6 / 5.7 lost: a large change with a small net effect. The step-0 ceiling (+3.2 points of node recall) was an upper bound; the measured net is about a tenth of it. The TRAIN - VAL gap is 0.05 in both arms, the past does not widen it. Not measured: the cost in ms per frame (a quiet-host timing is open); the controls of seed 1; the capacity control `c` (not triggered: r - z stayed below +0.005).

**What it means for the champion.** At the token level the previous keyframe carries mostly what the current frame carries; E299 and E354 said the same at the output and the feature level. The design step "cross-attention to the past" is built, simple, end-to-end and works, but it is not worth +22 % per training step for +0.002 to +0.004 topo F1; it stays an optional block (`jpast.py`) for a later stage that needs the past for something the one-frame head cannot see. The evidence the long wiring lacks did not come from the map (S3, section B7: also a TIE) either; chapter 13 lists what is left.

## B5. A wiring loss against the junction weakness (E376, NO)

**The question.** The champion's weak number is the junction F1 (0.25 at the final decode against 0.27 for plain B). Its own argmax decode (one link per source, no link threshold) gave a junction F1 of 0.299 / 0.295 on the same heads, so the idea was that the multi-positive likelihood tolerates fan-out (0.9 on one token and 0.45 on each of two duplicates cost the same) and that a loss which asks for ONE winner among the duplicates, -log max over the group, would make the head's own probabilities decode cleanly. One loss-only arm (`--pos-loss max`), the champion recipe otherwise, two fresh head seeds, VAL 873 frames, read at the final per-class cell; the rule written before the run: ADOPT-LOSS needs a junction F1 gain of +0.030 on both seeds with the paired interval above 0 and topo F1, F1@1.0, lanes found and the duplicate share not worse than their limits; ADOPT-SIMPLER needs a single link threshold to hold the junction F1.

Table: E376 on VAL (873 frames), the max-positive arm minus the champion head of the same seed (paired scene bootstrap).

| seed, cell | topo F1 | F1@1.0 | lanes found | junction F1 | dup_frac |
|---|---|---|---|---|---|
| seed 0, final per-class cell | +0.0055 [+0.001, +0.011] | +0.0047 [-0.001, +0.011] | +0.0097 [-0.001, +0.021] | **+0.016 [+0.002, +0.033]** | +0.009 |
| seed 1, final per-class cell | +0.0110 [+0.006, +0.017] | +0.0040 [+0.001, +0.007] | +0.0107 [+0.006, +0.016] | **+0.008 [+0.001, +0.015]** | +0.006 |
| seed 0, single threshold 0.2 / 0.2 | +0.0060 | +0.0041 | +0.0178 | **-0.026 [-0.039, -0.011]** | +0.007 |
| seed 1, single threshold 0.2 / 0.2 | +0.0088 | +0.0031 | +0.0168 | **-0.032 [-0.039, -0.024]** | +0.003 |

Absolute at the final cell: topo F1 0.6734 / 0.6872, F1@1.0 0.851 / 0.851, lanes found 0.729 / 0.731, junction F1 0.268 / 0.253 (the champion head 0.252 / 0.245).

**Verdict: NO, the champion keeps its recipe.** The junction gain at the final cell is +0.016 / +0.008 against the bar +0.030; with a single link threshold the junction F1 falls by 0.026 / 0.032, so the loss does not even remove the second threshold. The other guards hold and the kill did not fire. The prediction (+0.010, window -0.030 to +0.040) hit.

**What it rules out.** The max-positive likelihood is not the junction lever: it concentrates the mass further where it was already concentrated (short sources with two or more links at or above 0.3: 3.8 % to 1.5 % on seed 0 and 3.0 % to 1.9 % on seed 1, so fan-out of links was never large), leaves the kept near-node none tokens unchanged (21.8 % to 21.5 % on seed 1), and delivers more long links (13 610 / 13 692 against 12 509 / 12 356) with a lower true share (0.158 / 0.168 against 0.171 / 0.181) at an unchanged long-link AP. **A side finding, not a verdict:** at the champion's own cell the max arm is not worse anywhere and better in topo F1 (+0.006 / +0.011, intervals above 0), lanes found and F1@1.0 at the same cost; a champion update on that basis needs its own written rule and a confirmation on fresh seeds (planned, not run), so the champion of this report is unchanged.

## B6. Where the none push goes (E377, TARGETING-ONLY)

**The question.** E376 left the junction F1 at 0.25 (plain B 0.27). A scalar none weight of 0.5 (the E367 control m) raised it at the champion's cell by +0.038 / +0.034 but cost lanes found (-0.057 / -0.056). The mechanism: false junction points come from kept none tokens next to real nodes (the duplicates). E377 asks whether the same push, placed only on those near-node none tokens (weight 0.73 on them, 0.25 elsewhere, `--none-weight-near`), keeps the junction gain without the recall loss. Two seeds, the champion recipe otherwise, VAL 873 frames. The rule written before the runs: ADOPT needs (i) against the champion at D115 a junction F1 gain of +0.03 with the interval above 0 and topo F1 / F1@1.0 within -0.005 and lanes found within -0.010, and (ii) against the scalar control m a lanes-found gain of +0.010 without a junction loss of more than 0.010; TARGETING-ONLY is (ii) without (i).

Table: E377 on VAL (873 frames), the near-node arm n at its TRAIN-chosen cell (0.2 / 0.2 / 0.6 / gate off, the same cell as the control m) minus the reference (paired scene bootstrap across runs).

| comparison | topo F1 | F1@1.0 | lanes found | junction F1 | dup_frac |
|---|---|---|---|---|---|
| n seed 0 - champion (i) | -0.0095 [-0.017, -0.002] | -0.0103 | **-0.038 [-0.049, -0.030]** | **-0.009 [-0.022, +0.004]** | -0.018 |
| n seed 1 - champion (i) | -0.0059 [-0.011, -0.001] | -0.0073 | **-0.031 [-0.040, -0.021]** | **+0.007 [-0.006, +0.021]** | -0.012 |
| n seed 0 - m (ii) | -0.0067 | +0.0043 | **+0.0115 [+0.000, +0.021]** | +0.0067 [-0.005, +0.018] | +0.012 |
| n seed 1 - m (ii) | +0.0020 | +0.0113 | **+0.0192 [+0.009, +0.032]** | **+0.031 [+0.015, +0.052]** | +0.018 |
| n s0 / s1 - champion at D115 (reported) | -0.020 / -0.017 | -0.011 / -0.008 | -0.047 / -0.037 | **+0.046 / +0.061** | -0.015 / -0.010 |
| m s0 / s1 - champion at D115 (reported) | -0.0125 / -0.0177 | -0.015 / -0.018 | -0.057 / -0.056 | +0.038 / +0.034 | |

Absolute at the (b) cell: topo F1 0.6585 / 0.6703, lanes found 0.681 / 0.690, junction F1 0.243 / 0.252 (the champion 0.252 / 0.245).

**Verdict: TARGETING-ONLY, the champion keeps its recipe.** (i) fails on every clause on both seeds (junction F1 -0.009 / +0.007 against the bar +0.030); (ii) holds on both seeds (lanes found +0.012 / +0.019, junction F1 +0.007 / +0.031). The kill did not fire (loss 2.62 -> 0.54 / 2.60 -> 0.54 between steps 100 - 200 and 1800 - 2000).

**What it says.** The near-node weight does what the mechanism said on the tokens: kept near-node none tokens 21.7 % -> 13.4 % / 13.8 % (predicted about 13 %), tokens with a degree of 3 or more 22.4 -> 13.2 / 13.8 points per frame, duplicates down. At the champion's own cell that is a junction F1 gain of +0.046 / +0.061, but it costs recall (lanes found -0.047 / -0.037, topo F1 -0.02), and the cell chosen on the TRAIN sample (a lower short threshold, 0.2) gives the recall back in part and the junction gain away. Against the scalar weight of the same push the placement is better on both seeds: **where the none push goes matters, but no none weight tried (0.5 scalar, the near placement) beats the champion's recall-junction balance.** The prediction hit on lanes found and on the share of near-node tokens, and missed on the junction F1 (predicted +0.035 at (b), measured -0.009 / +0.007). A junction-precision lever that does not cost recall has to change what the head sees, not only how the none tokens are weighted.

## B7. Stage S3 (E374): the map does not pay either

**What the S3 runs ask.** One cross-attention block after the second self-attention layer reads road-level SD-map tokens (a sample every 2 m of the carriageway centre lines with position, signed direction of travel and the lane count; +20.3 k parameters, no rule step). The map is a simulation derived from the lane ground truth with an SD-style error (a rigid shift of 1 - 2 m, a yaw error, dropped side roads, count errors), and the head is trained on random errors including absent, count-unknown and grossly wrong maps. **The map is an optimistic stand-in: a gain here is an upper bound for a real SD map.** Arms on the two caches: `z` (control, block and map disabled, same trainer), `m` (the block), `z2` (a second initialisation of the control, the noise reference); five runs. Each arm is read at the decode chosen on its own TRAIN sample (all five chose link 0.1 / keep 0.6 / gate off), VAL 873 frames, TEST untouched. The rule was written before the runs: KEEP needs m - z >= +0.010 topo F1 on both seeds with the interval above 0, the guards, and that the gain belongs to this scene's map (the map of another scene at least 0.005 lower); a TIE keeps the cheaper control. The prediction was +0.006 (window 0.000 to +0.012; odds KEEP 25 %, TIE 60 %, CUT 15 %).

Table: S3 on VAL (873 frames), cell link 0.1 / keep 0.6 / gate off.

| arm | topo F1 | lanes found | F1@1.0 | junction F1 (P / R) | dup_frac | AP of the existence |
|---|---|---|---|---|---|---|
| z seed 0 | 0.6617 | 0.6747 | 0.8359 | 0.2089 (0.126 / 0.607) | 0.0548 | 0.9442 |
| m seed 0 | 0.6649 | 0.6920 | 0.8429 | 0.2050 (0.122 / 0.634) | 0.0564 | 0.9539 |
| z2 seed 0 (seed twin) | 0.6575 | 0.6790 | | 0.2050 | | |
| z seed 1 | 0.6767 | 0.6905 | 0.8422 | 0.2036 (0.121 / 0.635) | 0.0536 | 0.9424 |
| m seed 1 | 0.6793 | 0.7027 | 0.8467 | 0.2041 (0.121 / 0.663) | 0.0552 | 0.9563 |

**m - z (paired scene bootstrap):** topo F1 **+0.0031 [-0.0006, +0.0068]** (seed 0) and **+0.0026 [-0.0017, +0.0066]** (seed 1) against the bar +0.010; F1@1.0 +0.0070 / +0.0045; lanes found +0.0173 / +0.0122 (intervals above 0); junction F1 -0.0039 / +0.0005 with junction recall +0.028 / +0.029 and precision flat; AP +0.0097 / +0.0139. The seed twin differs by -0.0042 [-0.0102, +0.0020]: the gain of m is inside the twin spread. The kill measure (the output change when the map of another sample is given) is 0.20 / 0.35, far above the line 0.01: the block reads the map content.

Table: the map controls on the same m checkpoint (topo F1; free read-outs, never a verdict; difference to the record `err` with its paired interval).

| map given | m seed 0 | m seed 1 |
|---|---|---|
| err (the record, simulated SD error) | 0.6649 | 0.6793 |
| clean (the exact road map, an oracle here) | 0.6684 (+0.0036 [+0.0016, +0.0057]) | 0.6845 (+0.0052 [+0.0024, +0.0084]) |
| gross (the error map shifted by 5 m) | 0.6614 (-0.0035 [-0.0062, -0.0010]) | 0.6765 (-0.0028 [-0.0055, -0.0000]) |
| none (no map) | 0.6643 (-0.0006) | 0.6790 (-0.0003) |
| S (the map of another scene) | 0.6617 (-0.0031 [-0.0076, +0.0018]) | 0.6766 (-0.0027 [-0.0064, +0.0005]) |
| U (lane counts unknown) | 0.6657 (+0.0009) | 0.6797 (+0.0003) |
| C2 (every count set to 2) | 0.6649 (+0.0001) | 0.6784 (-0.0009) |

**Verdict: TIE, the champion takes no map block.** KEEP is not met on three clauses: m - z +0.0031 / +0.0026 against +0.010, the interval includes 0, and the other-scene map is only 0.003 below the record (the bar 0.005); the noise band, AP, F1@1.0, lanes found, the duplicate share and the kill hold. The capacity control `c` was not run (its trigger, a gain of +0.005, was not reached). The prediction hit (inside the window on both seeds). The cost of the block is 0.155 against 0.099 s per training step, +20 264 parameters, +0.17 GB and the map build.

**Reading, with its limits.** The head does use the map: the exact road map is +0.004 / +0.005 above the simulated-error map, a map shifted by 5 m costs about 0.003, and a missing or wrong map does no harm (the controls none, S, gross stay within 0.004 of the control). Lane counts carry nothing (U and C2 equal the record within 0.001). What the map improves is the recall of weak tokens (lanes found +0.012 / +0.017, junction recall +0.03), not the junction precision (0.12), so the junction F1 stays flat. A real SD map is of lower quality than the simulation, so the real gain is not larger than this one. **Same pattern as S2: the head extracts what the cached tokens hold; the missing information is not where the junction wiring errors are (wrong pairings of two real nodes).**

## C. The oracle-node stage of the same design (given ground-truth nodes; a different read-out, not comparable with the ladder)
Before predicted nodes, the design was tested on the ground-truth nodes of a 2.5 m lattice (E357 / E359 / E361 / E362): topo F1 0.4255 / 0.3928 (one pair head), 0.7721 / 0.7626 (node-type output + per-source softmax, 3000 steps), 0.8316 / 0.8297 (10000 steps), 0.8427 / 0.8367 (16000 steps, INCONCLUSIVE); the best rule on those nodes is 0.7011, the GT links 0.9666. The wiring is learnable with GT nodes; the weak output was the node-type head (recall 0.82 / 0.81).

## D. Compute (little compute is part of the goal)
Table: Compute of the reference and of the champion and its parts (24 frames, batch 1, bf16, CUDA events, 5 repeats, K = 512; the champion's numbers are two runs in the quiet window, host load 2.2 - 2.9).

| system | parameters (M) | ms per frame | note |
|---|---|---|---|
| E324 s (reference, E2E read-out) | 2.741 | 19.7 (24-frame quiet-host protocol, junction forward) | the G-cost reference |
| B' alone (pyramid + node head + top-K gather) | 2.196 | 11.41 / 11.37 | the frozen lane head, cached tokens are derived from it |
| the token head alone | 0.127 | 6.70 / 6.64 (0.86 GMACs at K = 512) | the three self-attention layers and the wiring layer |
| **champion, whole path** | **2.323** | **16.83 / 16.57** (p90 19.09 / 20.39), 16.6 - 16.8 ms in the quiet window | **2.9 - 3.1 ms and 0.42 M parameters below E324 s**; measured on the head of the first recipe, the champion has the identical architecture (a loss weight differs); the tool flagged the window as not quiet (GPU up to 10 % before the run): contention can only slow the path, so the comparison is conservative; the tail (p90) is not better than E324 s's median |
| the decode thresholds | 0 | 0 | the same forward, a threshold on the output |

Timing protocols are never mixed: only paired differences inside one protocol are compared.

## E. Still difficult situations
The junction links are the weak part: the junction F1 of the champion (0.25 after the per-class decode of E375, 0.20 before) is below the plain head's 0.27 because the written gate has no junction condition; the duplicates are low (0.055 - 0.063) but above the record decode's 0.004; precision is 0.68. PENDING: the split by crowded junctions, far range and own lane, as in the earlier report (not computed for the token head yet). 

## Sources
Cards (`experiments/E<id>_*/experiment.md`): E365 (sections 1 - 3), E368 (section 3), E374 (section 3), E376 (section 3), E377 (section 3), E357, E359, E361, E362 (oracle-node stage), E324, E358, E363 (the ladder), E275 (plain B); ledger rows `E365.*`; decisions D113.
