# E267 Stack of spurs kept (E265) and the seed ring (E266) on the recall chain: a 2 x 2 with recall_s1 as seed twin

One file per experiment; it grows from card (P2) to detail plan (P3 start, block experiments only) to summary (P3 end). The machine-readable part lives next to it in `experiment.json`. Written by ar-planner (loop 22, 2026-11-01 ~01:30) on the referee's stacking advice (E265 / E266 verdict paragraphs); the rule below was written before any number of the ring (E266) existed. Known at writing time: only the in-sample step-0 numbers of spurs kept on `recall` (`../E265_far-coverage-loss-budget/step0_cleanup.txt`: lanes found 0.7807, R@1 far 0.7796, DP 13.70).

## 0. What the tools run (the fields of `experiment.json`)
| field | value |
|---|---|
| id / card / title | E267 / H22.E267 / spurs kept + seed ring on rc1, 2 x 2 |
| kind | variant (configuration of two settings built by E265 (existing option) and E266 (code S2)) |
| deterministic | true (each arm is a setting of one checkpoint; the seed twin is the second checkpoint) |
| metric, direction, band | lanes_found, higher, 0.010 |
| prediction | +0.017 lanes found, sr against c |
| arms | c = registry (`recall`, `recall_s1`, re-scored as E266.c, tag l22r); s = `spur_len 0` (E267.s, tag sp0r); r = ring (`recall_ring`, `recall_s1_ring`, E266.b0 / b1); sr = ring + `spur_len 0` (E267.sr, tag sp0 of the ring names) |
| control | c |
| rule | sr - c: lanes found >= +0.015, R@1 far >= +0.015; sr - s: R@1 far >= +0.005; sr - r: lanes found >= +0.010 (a clause on the headline metric is held to the band by decide.py); guards Q 0.005, topo F1 0.005, FL 0.005, dup 0.010, P@1 far 0.010, inner far vs s 0.005 |
| budget_gpu_h, kill | 0 (the ring raw caches come from E266), see below |
| evaluate | val, compare c |

## 1. Card (written in P2, checked by ar-referee: the stacking advice of the E265 / E266 referee paragraphs; no separate referee pass)
| | |
|---|---|
| lens | system review (stack of two accepted level-3 changes) |
| group | A3 sparse lane graph; gap 1 of PROBLEM.md (coverage at range) |
| sources | E265 (spur removal loses true far stubs: lanes found +0.013, R@1 far +0.008 on recall in-sample), E266 (the 45 - 48 m ring holds 70 % of the R@1-far gap; prediction +0.012 R@1 far), both cards' "relation" paragraphs ("if both KEEP, one combined arm before any default changes; the effects need not add"), referee stacking advice (2026-11-01) |
| status | scheduled (gated: E266 G1 passed and E265 not CUT) |

- **Statement** (falsifiable, one sentence): on `recall` and its seed twin `recall_s1`, the combination of spurs kept and the seed ring raises lanes found by >= +0.015 and R@1 far by >= +0.015 over the registry settings, and each half adds to the other on its own metric (ring over spurs-kept: R@1 far >= +0.005; spurs over ring: lanes found >= +0.010, the band, because lanes found is the headline), with Q, topo F1, false length, duplicates, P@1 far and the per-piece DP budget held.
- **Mechanism**: the two changes act on different stages of the far-coverage budget (E265 step 0): the ring adds seed candidates where the inference lattice offers none (the tree lacks the ring before admission: valid-tree ring R@1 0.800 against inner 0.894), the spur rule deletes short true stubs after the union (clean-up loses 1.2 % of far points). Overlap is expected where ring seeds grow short edge pieces that the spur rule would cut (side lanes cut by the 48 m box): spurs kept may make part of the ring's gain visible, or the ring may already cover lanes that only survived as stubs. The 2 x 2 measures the interaction instead of assuming additivity.
- **Prediction** (VAL 873, cached raw outputs, two seeds of rc1): sr - c lanes found +0.017 (each seed +0.013 .. +0.021), R@1 far +0.017, Q +0.002 (+-0.004), topo F1 -0.002, false length +0.004, dup +0.004, per-piece DP 12.82 -> about 13.9 on recall (<= 14.3); interaction (sr - s - r + c) between -0.004 and 0 on both metrics.
- **Decision rule** (written before any ring number; band 0.010 on lanes found; the two rc1 seeds are the paired units; `decide.py --experiment E267 --write` plus the lead checks): **KEEP** (the prototype setting of the recall family is ring + spurs kept) if sr - c >= +0.015 in lanes found (both seeds positive, interval of the recall card excluding 0) and >= +0.015 in R@1 far, sr - s >= +0.005 in R@1 far and sr - r >= +0.010 in lanes found (both seeds positive: each half adds; the second bar is the band because `decide.py` holds every clause on the headline metric to it, written so before any ring number), Q loss <= 0.005, topo F1 loss <= 0.005, false length rise <= 0.005, dup rise <= 0.010, P@1 far loss <= 0.010 against c, inner-far R@1 of sr within +-0.005 of s; absolute on both seeds (lead): Q >= 0.735, per-piece DP <= 14.3, false length <= 0.062. **CUT of the stack** (an "adds" clause fails, or a guard): the recall-family prototype takes the single setting its own experiment adopted; if both single settings are adopted, the one with the higher mean of lanes found and R@1 far over the two seeds; inside 0.005 of each other spurs kept (no network-side change, one rule step less). **TIE** (sr - c inside the band with the guards holding): the settings each experiment adopted on its own stay as decided there; no stack. **INCONCLUSIVE** (the seeds disagree): no stack tonight; the single settings as decided.
- **Test design**: arms c, s, r, sr on `recall` (rc1 seed 0) and `recall_s1` (rc1 seed 1), the same 873 frames; s and sr through `evaluate.py --asm spur_len=0`; r and sr read the ring raw caches written by E266.b0 / b1 (cache key S1); all four arms' cards written with tonight's code (one protocol). Controls: null = c; matched reference = c on the same frames (paired scene bootstrap of every `--compare` card); seed twin = recall_s1; equal budget: same checkpoints, same K0 / M; if sr's per-piece DP exceeds c's by more than 1.0, the E265 equal-budget frontier (E265.eb, rc1 tau sweep) is read at sr's DP and sr must lie above it (+0.006 lanes found or +0.005 Q). Rung: VAL 873, layer 3 - 4; no TEST (the prototype's TEST is E268).
- **Cost** and **kill criterion**: 0 GPU; CPU about 25 min (4 cards with `--compare`, the DP printout of `cleanup_split.py` on the two ring names gives r and sr); no code beyond S1 / S2. Kill: E266 G1 fails or E265 is CUT (the stack is not run); the l22r control cards do not reproduce the stored numbers.
- **Inputs**: ready after S1 / S2 (05:30) and E266.b0 / b1 (~06:15): registry entries `recall_ring`, `recall_s1_ring`; `evaluate.py --asm`; `cleanup_split.py`, `ring_split.py` (E265 folder).
- **Differentiable path**: the training path is untouched; the delivered graph loses one rule step (the spur rule) and gains none (the ring is a fixed candidate set before the existing top-K0 selection).

## 2. Detail plan (written with the card)
What would make it fail: the two gains are the same lanes (ring seeds grow exactly the stubs spurs-kept restores), so the "adds" clauses fail; or the stack's added vertices push recall over 14.3 per-piece DP; or the l22r / sp0r cards are written by different code than the ring cards (protocol mismatch in `decide.py`: then re-score all four arms in one run).

| step | what exactly | where (file / function) | done when | budget |
|---|---|---|---|---|
| code | none beyond S1 / S2 (E266) | - | E266.gold passes | - |
| train | none | - | - | - |
| evaluate | E267.s: `evaluate.py recall --asm spur_len=0 --tag sp0r --compare recall`, same for `recall_s1`; E267.sr: `evaluate.py recall_ring --asm spur_len=0 --tag sp0 --compare recall`, `evaluate.py recall_s1_ring --asm spur_len=0 --tag sp0 --compare recall_s1`; c = E266.c (l22r), r = E266.b0 / b1 cards; DP of r and sr from E266.v (`cleanup_split.py` rows "registry" and "spurs kept") | `evaluation/evaluate.py`, `experiments/E265_far-coverage-loss-budget/cleanup_split.py` | 8 cards exist, `experiment.py check E267` ready | CPU ~25 min |
| decide | the rule above | `decide.py --experiment E267 --write` + lead checks (absolute guards, inner-far two-sided, equal-budget frontier if DP + 1.0) | verdict in the ledger rows E267.s / E267.sr | - |

## 3. Summary (written after the decision)
- **Runs** and **decision**:
- **Setup**:
- **Numbers**:
- **Images**:
- **What it says**:
- **Weaknesses** / **next steps**:

## 3. Summary (lead, loop 22, 2026-11-01 06:10)
- **Runs**: E267.s (spurs kept, ring metric), E267.sr (ring + spurs kept), with E266.b0 / b1 (ring) and E266.c (control) as the other cells of the 2 x 2; E266.v for DP; code tags `lcr/E267.*`. **Decision: CUT** (`decide.py --experiment E267 --write`: false length +0.0060 against the 0.005 guard, P@1 far -0.0147 against 0.010); every coverage clause holds.
- **Numbers** (VAL 873, sr - c, paired scene bootstrap):

| seed | lanes found | R@1 far | Q | false length | P@1 far | DP / 100 m |
|---|---|---|---|---|---|---|
| recall | +0.022 [+0.018, +0.026] (0.767 -> 0.789) | +0.026 (0.771 -> 0.798) | +0.000 | 0.051 -> 0.057 | 0.768 -> 0.753 | 13.95 |
| recall_s1 | +0.022 [+0.018, +0.027] | +0.028 | +0.002 | 0.052 -> 0.058 | | 14.04 |

  Additivity: the ring adds R@1 far +0.019 over spurs kept, the spurs add lanes found +0.015 over the ring: the two act on different losses and add.
- **What it says**: the stack is the strongest system of the loop at range (R@1 far 0.798, lanes found 0.789 on `recall`), but each kept stub and ring seed also adds some false geometry at range (P@1 far -0.015, false length +0.006): two precision guards missed. Against the prototype `recall_sp0_dp` it would add only +0.009 lanes found (under the +0.010 replacement bar), so S stays by both rules.
- **Next**: a precision step at range (e.g. a range-dependent tau for ring seeds, or the union with a wider vertex merge) before the stack can be adopted.
