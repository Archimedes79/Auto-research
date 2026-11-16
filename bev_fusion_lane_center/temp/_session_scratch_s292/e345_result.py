import io
import time

p = "experiments/E345_pruning-sensitivity-step0/experiment.md"
s = io.open(p, encoding="utf-8", newline="").read()
assert "## 7. Result" not in s
now = time.strftime("%H:%M")
txt = f"""
## 7. Result (2026-11-15 {now}; the card's own result: `jprune_table.py table` on the four sweeps, chain day21 finished; tables `results.md` and `table.csv` next to this card; by hand: analysis, selected on half of the VAL scenes and judged on the other half)
**Setup as written:** E312 and E324 s, both seeds each, 45 units (84 and 168 table rows of 3 fractions), the null against the stored cards 0.0 (units of the tolerance) on both read-outs of every model-seed, the two nulls against each other 0.0, plumbing checks passed (identity 0.0, zeroed copy 0.0, zeroing effect 98.6 - 100). **Compute is counted as MACs of the narrowed network (GMACs per frame), no ms** (the timing column waits for a quiet-host timing; E351 / E342 show that the forward is host-bound: MAC shares are not ms).
**Predictions (pre-registered guesses; hit windows narrower than the rule):** (1) v loses 0.005 on the judgement half, mean of the four model-seeds (window 0.002 - 0.008): **MISSED: v loses +0.0096 / +0.0354 (E312 seeds 0 / 1) and +0.0118 / +0.0597 (E324 s), mean +0.029**: the units that are FREE one at a time do not stay free together (interactions are in no single-unit sweep), and seed 1 of both models loses 3 - 5 times more than seed 0. (2) v saves 0.8 GMACs (window 0.4 - 1.2): **HIT: 1.117 GMACs (E312) and 0.744 (E324 s), mean 0.93**. (3) at least 15 of the 45 units have a sweep-FREE point at f >= 0.25 on all four model-seeds: **MISSED by one: 14 units in both models** (17 in E312, 14 in E324 s; classes by the worse seed of a model): the FREE units are `jm.m05.hid` (0.5 m branch, f 0.75: 1.106 GMACs), `pyr.s05+m05` (1.172 GMACs, its only reader is the paint evidence) and 12 units that save nothing (the pair head's hidden units and attention heads 0.000 - 0.004 GMACs, `jm.head05.*`, `jm.head1.c1` 0.001), plus `pyr.e2` (0.138 GMACs at f 0.5, B's pyramid); E312 has two more (`pyr.d2`, `pyr.e3`: 0.107 and 0.053 GMACs at 0.25, B's pyramid). **Falsifiable statement:** the ranked removal at f = 0.50 loses less than the random removal in more than two thirds of the non-tied units on every model-seed (0.737 / 0.789 / 0.789 / 0.684: **holds**), and the reverse ranking loses more than the ranked one in more than 90 % (0.632 / 0.842 / 0.789 / 0.737: **does NOT hold**): the importance ranking carries information, but less than 'more than 90 %'. The sweep as a proxy for the full score: Spearman 0.22 / 0.44 (E312) and 0.52 / 0.56 (E324 s) on the 9 - 10 stratified points: weak to moderate.
**The sizes for step 1 (judgement half, worse seed and read-out; heads-only prefixes of the greedy curve; D99 point 4):**

| model | set | saved GMACs of the heads' 3.278 | loss s0 / s1 (judgement) | 95 % upper s0 / s1 |
|---|---|---|---|---|
| E312 | S1 (10 %) | 0.369 | +0.0007 / +0.0002 | +0.0017 / +0.0002 |
| E312 | S2 (20 %) | 0.737 | +0.0021 / +0.0015 | +0.0039 / +0.0015 |
| E312 | **S3 (35 %)** | **1.152** | +0.0014 / +0.0034 | +0.0035 / +0.0058 |
| E312 | S4 (50 %) | 1.636 | +0.0293 / +0.0386 | +0.0345 / +0.0465 |
| E324 s | S1 (10 %) | 0.369 | +0.0006 / +0.0001 | +0.0009 / +0.0004 |
| E324 s | **S3 (35 %)** | **1.106** | +0.0018 / +0.0019 | +0.0029 / +0.0038 |
| E324 s | S4 (50 %) | 1.728 | +0.0295 / +0.0373 | +0.0350 / +0.0490 |
| E312 | P1 / P2 (pyramid view 10 / 25 %; B's pyramid, not deliverable by step 1) | 0.592 / 1.189 | +0.0083 / +0.0070 ; +0.0083 / +0.0070 | +0.0119 / +0.0089 |
| E324 s | P1 / P2 | 0.436 / 1.225 | +0.0073 / +0.0007 ; +0.0125 / +0.0061 | +0.0071 / +0.0020 ; +0.0161 / +0.0102 |

**S_a = S_b = S_c = S3 on both models** (the largest prefix with a loss of at most 0.01 on both read-outs without retraining; S4 loses 0.029 - 0.039 and breaks every bar; S3 is also the knee). **S3 is the 0.5 m paint branch** (`jm.m05.hid` at 75 % plus one context block at 25 % on E312): the part E341 (E312 recipe) already removed with a retraining. **For the system of record, E324 s, the paint branch is still in (E348, the retrained variant without it, was not adopted: one guard by 0.0009): its hidden units at 75 % can be taken away WITHOUT retraining for +0.0018 / +0.0019 topo F1 (95 % upper 0.0029 / 0.0038, below the FREE bar 0.01) and 1.106 GMACs (13.5 % of the system's 8.202), read on the judgement half of the scenes; v of E324 s (all FREE head units together) is not free (+0.0118 / +0.0597): take S3, not v.** Not claimed: that the narrowed network is as fast (the ms are host bound, a quiet-host timing is needed: E351.q_eager compares E341 / E348 with E312 / E324 s), anything on TEST, that a CRITICAL unit cannot be pruned after a sparsity term (step 1).
**Answer to D99's question, in numbers:** (a) without retraining about 1.1 GMACs of the 3.28 GMACs of the heads are free, all of them in the 0.5 m paint branch (of 8.2 GMACs of E324 s: 13.5 %); (b) beyond the paint branch the heads hold about 0.14 GMACs at a loss of up to 0.009 (three context blocks at 25 %: `jm.blocks.0 / 1 / 3.hid`; the other units are COSTLY or CRITICAL), the pair head has 0.022 GMACs in all; (c) the larger FREE / CHEAP items are in B's pyramid (59 % of the system's MACs: `pyr.d2` at 50 % 0.214 GMACs for +0.004, `pyr.d0` at 25 % 0.263 GMACs for +0.007, `pyr.e2` at 50 % 0.138 GMACs for -0.005 / 0.000; a pruned pyramid changes B's input features: a B card, not a card of the junction layer); (d) with E341 adopted for E312 (and a decision on E348 for E324 s) a step-1 card for the heads alone would chase about 0.14 GMACs (under 2 % of the system). **No step-1 card is written today** (GPU and day budget; the retrain-free S3 for E324 s is the only item with a size: its ms would be the next measurement).
**Process:** chain day21 (one record per job, `E345.*` rows); the E324 s sweep of `v` was restarted once after a WSL out-of-memory error in a data loader (`E345.v_e324s_s0`, first attempt failed at 79 s, same inputs, ok on the second).
"""
s = s.rstrip("\n") + "\n" + txt
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("ok")
