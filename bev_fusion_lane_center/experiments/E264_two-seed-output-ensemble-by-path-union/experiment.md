# E264 Two-seed output ensemble through the path union (multi-hypothesis trees of two j1rc seeds, one assembly)

The card (read by `experiment.py` and `decide.py`; schema `autoresearch/schemas/experiment.schema.json`):

```json card
{
 "id": "E264",
 "card": "H22.E264",
 "title": "two-seed output ensemble: the raw hypothesis trees of two j1rc seeds assembled once by the existing path union (held-out pair s2 + s3)",
 "kind": "block",
 "deterministic": true,
 "metric": "lanes_found",
 "direction": "higher",
 "band": 0.01,
 "prediction": {
  "value": 0.045,
  "vs": "p"
 },
 "variant": "u",
 "control": "p",
 "arms": {
  "u": {
   "recipe": "registry kind union, members e24_j1rc_s2 + e24_j1rc_s3 at the fast_recall settings (tau 0.7, fork pruning 0.4, vtx_r 1.5): evaluate.py ens_j1rc_23 --compare e24_j1rc_s3",
   "run_ids": [
    "E264.u23"
   ],
   "names": [
    "ens_j1rc_23"
   ]
  },
  "p": {
   "recipe": "better parent of the held-out pair by lanes found: e24_j1rc_s3, re-scored tonight (E265.c)",
   "cards": [
    "runs/eval/e24_j1rc_s3/val_l22.json"
   ]
  },
  "m": {
   "recipe": "matched-false-length single control: e24_j1rc_s3 at the tau whose val250 false_len_frac equals u's on val250 (chosen on val250, judged on 873: 250 frames overlap, labelled): evaluate.py e24_j1rc_s3 --tau <t*> --tag mfl --compare e24_j1rc_s3",
   "run_ids": [
    "E264.mfl",
    "E264.mfl2"
   ],
   "cards": [
    "runs/eval/e24_j1rc_s3/val_mfl.json"
   ]
  },
  "x": {
   "recipe": "replication / exploration pair s0 + s1 (step 0 of the card): evaluate.py ens_j1rc_01 --compare e24_j1rc_s1",
   "run_ids": [
    "E264.u"
   ],
   "names": [
    "ens_j1rc_01"
   ]
  },
  "q": {
   "recipe": "better parent of the exploration pair: e24_j1rc_s1 (E265.c)",
   "cards": [
    "runs/eval/e24_j1rc_s1/val_l22.json"
   ]
  },
  "w": {
   "recipe": "ceiling, not deployable: union of all four j1rc seeds (ens_j1rc_0123 --compare e24_j1rc_s1)",
   "run_ids": [
    "E264.u"
   ],
   "note": "report only, card runs/eval/ens_j1rc_0123/val.json (no card ref here, so a missing report card never blocks decide.py)"
  },
  "k": {
   "recipe": "mechanism control without diversity: union of recall + recall_s1 at the recall settings (ens_rc1 --compare recall); predicted lanes found gain < +0.010",
   "run_ids": [
    "E264.u"
   ],
   "note": "report only, card runs/eval/ens_rc1/val.json"
  },
  "v": {
   "recipe": "matched-vertex union: vtx_r chosen on val250 so that node-id vertices match s3 within +-0.15 (E236 procedure), judged on 873: evaluate.py ens_j1rc_23 --asm vtx_r=<v*> --tag mv --compare e24_j1rc_s3",
   "run_ids": [
    "E264.mv"
   ],
   "note": "report only, card runs/eval/ens_j1rc_23/val_mv.json"
  }
 },
 "rule": [
  {
   "metric": "lanes_found",
   "min_effect": 0.03,
   "note": "u23 against s3 with the paired-bootstrap interval excluding 0; lead check: x (u01) - q > 0 (same sign, exploration pair)"
  },
  {
   "metric": "R@1_far",
   "min_effect": 0.025,
   "note": "lead check: x - q > 0"
  },
  {
   "metric": "Q",
   "max_loss": 0.005
  },
  {
   "metric": "topo_F1",
   "max_loss": 0.005
  },
  {
   "metric": "false_len_frac",
   "max_loss": 0.01,
   "note": "and false length of u <= 0.062 ABSOLUTE (referee fix 1, the control's level)"
  },
  {
   "metric": "dup_frac",
   "max_loss": 0.02,
   "note": "referee fix 1: rise <= 0.020 against the better parent. Lead checks outside decide.py: per-piece DP of u (cleanup_split.py, E264.v) <= 14.3 measured before any verdict (fix 2); lanes found u - m >= +0.010 (else TIE: the union is a lower threshold, keep the single model); ms per frame <= 30 on the idle GPU (E264.t, fix 3; > 32 = CUT; no idle session = no prototype); identity union reproduces its member (spec test of S3)"
  }
 ],
 "budget_gpu_h": 0.3,
 "kill": "identity union does not reproduce its member's graph hash (stop, fix before any number); u23 gains < +0.015 lanes found over s3; per-piece DP of u23 > 14.3; the matched-false-length control within 0.005 lanes found of u23; idle timing > 32 ms",
 "evaluate": {
  "split": "val",
  "compare": "p"
 }
}
```


One file per experiment; it grows from card (P2) to detail plan (P3 start, block experiments only) to summary (P3 end). Section 0 (`experiment.json`) is written by the loop lead when the card is accepted.

## 1. Card (written in P2, checked by ar-referee)
| | |
|---|---|
| lens | state of the art |
| group | A4 multi-hypothesis / ensembles (gap 1 coverage at range); A3 (the assembly's path union as the fusion of two hypothesis trees) |
| sources | deep ensembles and multi-hypothesis output (A4 families: rMCL MH01 "a hypothesis that wins at least one target", relaxed WTA MH04; union of hypotheses as the recall ceiling); K71 model soups 2203.05482 (the weight-space alternative, card E263); E217 (valid nodes cover 85 % of the far points, admitted 74 %: the far gap is not a beam or seed budget), E236 (Q per vertex, matched-vertex procedure by `vtx_r`), E240 (seed SD), E253 (ms: j1rc 21.1, rc1 20.8, recursion +6.1), E257 (per-piece DP vertex basis); step 0 of this card and of E263 (`step0/step0_results.txt`, read-only CPU) |
| status | scheduled, blocked on code S1 / S3 until ~05:30 (loop 22 plan; referee fixes applied in section 2) |

- **Statement** (falsifiable, one sentence): Concatenating the raw hypothesis trees of two j1rc seeds (192 / 96 each, parents offset, junction heat as the element-wise maximum) and assembling them once with the existing path union at the unchanged `fast_recall` registry settings (tau 0.7, fork pruning 0.4, vtx_r 1.5) raises lanes found by at least +0.030 and R@1 far by at least +0.025 over the better of the two parents on VAL 873 for the held-out pair s2 + s3, beats the single parent at its matched false length by at least +0.010 in lanes found, keeps Q (loss <= 0.005), topo F1 (loss <= 0.005), false_len_frac (<= +0.010) and per-piece DP vertices (<= 14.3 per 100 m), and stays at or below about 30 ms per frame on the idle-GPU timing.
- **Mechanism** (why it should work): the far verdict of the j1rc fine-tune is seed-specific: 26.0 % of the far ground-truth points are found by one to three of four seeds and the union of four reaches 0.802 against 0.681 for a single seed (E263 step 0, approximate metric), so two seeds carry different, mostly true far lanes. The path union already fuses hypotheses inside one tree (nodes in descending path validity, snapped onto existing edges within `cov_r`, merged at vertices within `vtx_r`), so two trees can be fused by the same rule without a new fusion step; a lane found by both seeds is merged, a lane found by one is kept if its own validity passes tau. This is the multi-hypothesis principle of the A4 literature (a hypothesis set is scored by the best member per target, so diversity buys recall) applied at the output, where the soup of E263 applies it in weight space. Step 0 on the pair s0 + s1 at the registry operating point (no tuning): lanes found 0.7905 against 0.7433 / 0.7364 for the parents (+0.047 over the better one), R@1 far 0.7760 against 0.7368 / 0.7275 (+0.039), Q 0.7423 against 0.7427 / 0.7392, topo F1 0.7415 against 0.7389 / 0.7369, F1@1.0 +0.0055, false_len_frac 0.0619 against 0.0542 / 0.0565, dup_frac 0.090 against 0.074 / 0.070, node-id vertices 16.65 against 15.04 / 14.98; the identity union (s0 with itself) reproduces s0 to 4 decimals. Against `recall` (rc1 with time recursion, about 27 ms) this pair is at +0.023 lanes found, +0.005 R@1 far, +0.002 Q. What it costs and where it can fail: a second builder forward (single-frame builder 6 ms in K186, less if both members run as one batch-2 CUDA graph) and a larger assembly (CPU time x 1.3 - 1.7 on the loaded machine, step 0), so the chain moves from 21 ms to an estimated 28 - 32 ms; more duplicates (dup_frac +0.017) and +1.6 node-id vertices (the DP-basis count is estimated at 13.6, inside 14.3, and must be measured); the gain must not be a disguised lower threshold, hence the matched-false-length control. Why not temporal aggregation (considered first under this lens): step 0 shows that the warped previous frame adds no far ranking information beyond the current validity (AUC of far nodes 0.8653 single vs 0.8657 with a streaming log-odds memory; +0.6 % true far nodes at equal false count; a confirmation bonus gives at most +1.3 % in-sample), and only 14 % of the far misses were within 40 m of the ego in the last three frames.
- **Prediction** (VAL 873, `evaluate.py`, cards of the members from `runs/eval/e24_j1rc_s{0..3}/val.json`): pair s2 + s3 against its better parent s3 (lanes found 0.7393, R@1 far 0.7363, Q 0.7391): lanes found +0.045 (to about 0.784), R@1 far +0.035 (to about 0.771), Q +0.000 (+-0.004), topo F1 +0.003, false_len_frac +0.007, dup_frac +0.015, node-id vertices +1.6 (DP basis about 13.6); against s3 at its matched false length: lanes found +0.025, R@1 far +0.020; the union of rc1 + rc1s1 (no diversity, 0.59 % weight distance) gains less than +0.010 in lanes found (the mechanism control); the union of all four j1rc seeds (ceiling, not deployable): lanes found about 0.82; matched-vertex union (vtx_r raised until node-id vertices equal the better parent): Q +0.000, lanes found +0.040; idle-GPU ms per frame 28 - 32.
- **Decision rule** (band 0.010; seed SD lanes found 0.004, R@1 far 0.003, Q 0.0025 from E240; arms are deterministic given their members, the replication is the second pair): KEEP (register `fast_recall_ens2` = union of s2 + s3, or of s0 + s1 if both pairs pass, as the prototype) if on pair s2 + s3: lanes found >= +0.030 and R@1 far >= +0.025 over the better parent, the same signs on pair s0 + s1 (exploration pair, labelled), lanes found >= +0.010 over the matched-false-length single control, Q loss <= 0.005 and topo F1 loss <= 0.005 against the better parent, false_len_frac <= +0.010, per-piece DP vertices <= 14.3 per 100 m, and ms per frame <= 30 on an idle GPU (if the idle window does not come tonight: "quality KEEP, speed pending", the D4 option list gets the ms estimate with its range). TIE if the gain over the matched-false-length control is below +0.010 (the union is only a lower threshold: keep the single model, cheaper). CUT if a guard fails or ms > 32 (out of the speed class). decide.py: variant = union cards of both pairs, control = better-parent cards, clauses as above (`deterministic: false`, pairs by identity: (union01, s1), (union23, s3)).
- **Test design** (arms; controls; budget; rung): arms (CPU, cached raw outputs; all caches exist): u23 = union(s2, s3) (the test), u01 = union(s0, s1) (the exploration, step 0), u4 = union(s0 .. s3) (ceiling), u23v = u23 with `vtx_r` chosen on val250 so that node-id vertices match s3 within +-0.15 (E236 procedure, judged on the other 623 frames and on 873), urc = union(recall, recall_s1) at the `recall` settings (mechanism control: no diversity, no gain). Controls: layer 0 identity (union(x, x) reproduces x: graph hashes identical on 873 frames, `temp/loop21/graph_hash.py`); layer 1 (the member cards re-scored to 4 decimals); matched reference = the parent cards on the same 873 frames, paired scene bootstrap (`evaluate.py <union> --compare <parent>`); strong simple baseline = the better parent at the tau that gives the union's false_len_frac (tau chosen on val250, judged on 873 and on the 623 held-out frames); equal budget: no arm trains, all use the same frames and settings; seed twin: the second disjoint pair; cross-reference: E263's soup (the zero-cost alternative). Timing: the E253 driver (`evaluation/drivers/p18_latency.py`) with the two members as two CUDA graphs (and as one batch-2 graph if it is built), idle GPU, batch 1, cached features, 5 replicates; the assembly ms of the union from the same session. Rung: VAL 873 (layer 3 - 4), no TEST.
- **Cost** and **kill criterion**: GPU 0 for the quality arms (all raw caches exist), about 15 min of an idle GPU for the timing; CPU about 30 min (6 assemblies of 873 frames with 768 nodes, metrics, bootstrap); coding 1 - 2 h by an `ar-coder`: `p16_infer.union_raw(raws)` (15 lines: concatenation with parent / link_p offsets, junction max; spec test = identity union reproduces the graph hash) and a registry kind `"union"` with `members` in `evaluate.graphs_of` (default path untouched, `--golden check` unchanged), plus the two-member timing in the latency driver; shared code, so an `ar-reviewer` pass (mode code). Kill: the identity union does not reproduce its member (the merge is not neutral: stop and fix before any number); u23 gains less than +0.015 in lanes found over s3 (the s0 + s1 effect does not replicate); the per-piece DP count of u23 exceeds 14.3; the matched-false-length control reaches the union within 0.005 (threshold effect); timing above 32 ms on an idle GPU.
- **Inputs** (ready / to build): ready: raw caches `~/exp_gcar_b123/p18/raw_val873_e18_D7_j1rc_192.pkl`, `..._e18_D7_j1rcs1_192.pkl` (s0, s1), `~/exp_gcar_b123/eval_cache/raw_val_e24_D7_j1rcs2_192_96.pkl`, `..._j1rcs3_...` (s2, s3), `raw_val873_e123_D7_r3_rc1_288.pkl`, `raw_val873_e18_D7_rc1s1_288.pkl`; registry entries `e24_j1rc_s0` .. `s3`, `recall`, `recall_s1`; the step-0 scripts in `step0/` (the union function is `union()` in `diag_union.py`). To build: `union_raw`, registry kind, latency driver option. Fully differentiable training path: untouched (no training); the delivered graph keeps the same assembly rule steps as `fast_recall` (path union, clean-up, fork pruning; named as today's rule gap), with two trees as input instead of one; a learned merge of the two trees (or a distilled single network that reaches the union, the next step if this keeps) would remove the second forward.
- **Referee (2026-11-01, ar-referee)**: REVISE (strongest coverage candidate; G2-ready after fixes 1 - 3). Falsifiable: yes. The mechanism is measured (s0 + s1 at untuned registry settings: lanes found +0.047, identity union reproduces s0). s2 + s3 is a genuine held-out replication: nothing was tuned on s0 + s1. The matched-FL and matched-vertex controls are chosen on val250 and judged on the other 623 frames, so the held-out design is real. No ensemble was tried before. Fix 1 (duplicate blind spot): the rule has no dup_frac guard, although the union raises dup 0.070 -> 0.090 and Q cannot see parallel duplicates. Add dup_frac a - c <= +0.020 against the better parent, and make false length an absolute guard <= 0.062 (the control's level; the s0 + s1 union already sits at 0.0619, and the +0.010 guard would allow 0.064). Fix 2: the per-piece DP count (13.6) is an extrapolation: measure it before any verdict. Fix 3 (speed): the 28 - 32 ms estimate is optimistic. E253 (idle) gives j1rc 21.1 ms, of which the assembly is 8.8 ms. A second forward adds 6 - 12 ms and the union assembly adds 3 - 6 ms (x 1.3 - 1.7), so 30 - 39 ms is the honest range and the 32 ms kill is a real risk. Run the timing first in the lane window (05:45, nothing else on the GPU, CPU arms paused) before the quality verdict feeds the prototype; "quality KEEP, speed pending" is not a deliverable prototype. Code: `union_raw` plus a registry kind `union` in shared `evaluate.py` need an ar-coder spec test and an ar-reviewer pass before 05:45. The union entry must not resolve through `builder_raw`'s run-keyed cache, or it silently returns the first member's raw output.

## 2. Detail plan (ar-planner, loop 22, 2026-11-01 ~01:30; machine-readable part `experiment.json`)
Restated: the raw hypothesis trees of the held-out j1rc pair s2 + s3, concatenated and assembled once by the existing path union at the unchanged `fast_recall` settings, give >= +0.030 lanes found and >= +0.025 R@1 far over the better parent s3, >= +0.010 lanes found over s3 at the union's false length, with Q / topo F1 loss <= 0.005, false length <= 0.062 absolute, dup rise <= 0.020, per-piece DP <= 14.3, the same signs on the exploration pair s0 + s1, and <= 30 ms per frame on the idle GPU.

What would make it fail: the identity union is not neutral (merge bug); the gain does not replicate on s2 + s3; the union is a disguised lower threshold (matched-false-length control within 0.010); duplicates and false length grow past the guards (Q is blind to parallel duplicates); the second forward plus the larger assembly put the chain at 30 - 39 ms (the referee's honest range; above 32 ms = CUT). The speed clause is the likeliest failure: P(<= 30 ms) about 0.35.

Referee fixes applied: (1) `dup_frac` max_loss 0.020 against the better parent and false length <= 0.062 absolute (lead check; the relative clause stays at +0.010); (2) the per-piece DP of every union is measured (E264.v) before any verdict; (3) the idle timing runs first in the lane window (05:45, GPU exclusive, CPU arms paused) and is part of the verdict: without it E264 cannot be the prototype; the union entry resolves its members' raw outputs through each member's own cache entry, never through a run-keyed cache of its own (S1 / S3 spec). The matched-false-length and matched-vertex choices are made on val250 and judged on the other 623 frames (lead script below) and on 873 (the decide card, labelled "250 of 873 frames used for the choice").

Code (ar-coder, S3, ready ~05:30, reviewer pass): `p16_infer.union_raw(raws)`: concatenation of the member raw dicts with parent / link offsets, junction heat as the element-wise maximum; registry kind `union` with `members` (registry names) in `evaluate.graphs_of`: each member's raw from its own `builder_raw` entry, then `union_raw`, then `PI.assemble` with the union entry's tau and asm (`--tau` and `--asm` overrides honoured); `p18_latency.py` option for two members (two builder forwards as two CUDA graphs, then the union assembly). Spec tests: union(x, x) reproduces x's graph hashes on 873 frames (`temp/loop21/graph_hash.py`); the default path is untouched (`evaluate.py --golden check --cached-only` 6 of 6). Registry entries (lead, after the code): `ens_j1rc_23` (members e24_j1rc_s2, e24_j1rc_s3; tau 0.7, asm of fast_recall), `ens_j1rc_01` (s0, s1), `ens_j1rc_0123` (all four), `ens_rc1` (recall, recall_s1; tau 0.7, asm of recall).

| step | what exactly | where (file / function) | done when | budget |
|---|---|---|---|---|
| code | S3 above + S1 (cache key) | `training/pipeline/p16_infer.py`, `evaluation/evaluate.py`, `evaluation/drivers/p18_latency.py` | spec tests pass; reviewer pass; E266.gold | coder, by 05:30 |
| timing | E264.t, 05:45, GPU and CPU idle: `p18_latency.py` alternating 3 rounds of (a) the union s2 + s3 (two-member option, M 192 / K0 96, rec 0, fork pruning 0.4), (b) single s3 (`--name e24_D7_j1rcs3 --M 192 --K0 96 --rec 0 --fork_prune 0.4`), (c) `recall` (`--name e123_D7_r3_rc1 --rec 1 --fork_prune 0.4`, the reference of the recall family and the prototype's ms), n 60 each; ms = median of the replicate medians of `total` | `evaluation/drivers/p18_latency.py` | json files in `runs/e18/latency/` with tag e264_* | GPU 15 - 20 min |
| quality | E264.u23: `evaluate.py ens_j1rc_23 --compare e24_j1rc_s3`; E264.u: `ens_j1rc_01 --compare e24_j1rc_s1`, `ens_j1rc_0123 --compare e24_j1rc_s1` (ceiling), `ens_rc1 --compare recall` (no-diversity control, predicted < +0.010) | evaluate.py | 4 cards | CPU 25 min |
| DP | E264.v: `cleanup_split.py ens_j1rc_23 ens_j1rc_01` (row "registry": node-id and per-piece DP) | `../E265_*/cleanup_split.py` | DP printed | CPU 10 min |
| matched FL | E264.mfl: on val250, `evaluate.py ens_j1rc_23 --subset val250` and `evaluate.py e24_j1rc_s3 --subset val250 --tau T --tag tT` for T in 0.40 .. 0.65 (step 0.05); lead picks t* = the T whose val250 false length is closest to the union's; E264.mfl2: `evaluate.py e24_j1rc_s3 --tau t* --tag mfl --compare e24_j1rc_s3` (873, arm m) and the 623-frame read: temp/loop22/e264_heldout.py (not written: the 623-frame read was dropped, section 3) (lead, throw-away: frame records of u and m, `bootstrap.pooled` over the VAL frames not in val250, lanes found / R@1 far / false length of both) | evaluate.py, temp/loop22 | card val_mfl.json; 623-frame line | CPU 25 min |
| matched vertex | E264.mv (report): `evaluate.py ens_j1rc_23 --subset val250 --asm vtx_r=V --tag vxV` for V in 1.5 .. 2.5 (step 0.2), V* = node-id vert100 within +-0.15 of s3's val250 value, then `--asm vtx_r=V* --tag mv --compare e24_j1rc_s3` | evaluate.py | card val_mv.json | CPU 20 min |
| decide | `decide.py --experiment E264 --write` (u23 vs s3) + lead checks: u01 - s1 > 0 on lanes found and R@1 far, FL <= 0.062, DP <= 14.3, u - m >= +0.010 on 623 and on 873, ms <= 30 (CUT > 32); KEEP: candidate U of PLAN.md (`fast_recall_ens2` = ens_j1rc_23, or ens_j1rc_01 if both pairs pass with it better) | decide.py | verdict in the ledger row E264.u23 | ~06:45 |

Order tonight: E264.u23 (the u23 card only) 05:30 - 05:40, before the CPU must be quiet; the idle timing E264.t 05:45 - 06:00; E264.u (u01, u4, urc), E264.v, E264.mfl / mfl2 and E264.mv 06:00 - 06:45.

## 3. Summary (lead, loop 22, 2026-11-01 05:20)
- **Runs**: E264.u23 (held-out pair), E264.u (exploration pair u01, ceiling u4, no-diversity control urc), E264.v (per-piece DP), E264.mfl / E264.mfl2 (matched false length), E264.mv (matched vertices, val250, report only); code tags `lcr/E264.*`. E264.t (idle timing) **not run** (no idle GPU before the TEST window). **Decision: not adopted tonight**: `decide.py` gives KEEP on its clauses (`decision.json`), but the card's absolute false-length guard (<= 0.062) fails by 0.0015 and the speed check is missing, so there is no candidate U.
- **Setup**: the raw hypothesis trees of two j1rc seeds joined before the one existing path union (registry kind `union`, `evaluate.raw_of`); nothing trained. Controls: better parent (s3 / s1), matched false length (parent at tau 0.10), matched vertices (union with a wider vertex merge), no-diversity union (two rc1 seeds), ceiling (four seeds).
- **Numbers** (VAL 873 unless named; cards `runs/eval/ens_*/val.json`, `runs/eval/e24_j1rc_s*/val_mfl.json`):

| | lanes found | R@1 far | Q | false length | dup | DP / 100 m |
|---|---|---|---|---|---|---|
| s3 (parent) | 0.739 | 0.736 | 0.739 | 0.056 | 0.073 | 12.15 |
| u23 = s2 + s3 | 0.786 (+0.047 [+0.034, +0.060]) | 0.776 (+0.040) | 0.736 | **0.0635** | 0.090 | 14.13 |
| s3 at matched false length (tau 0.10) | 0.748 | 0.744 | 0.734 | 0.061 | 0.079 | |
| u01 = s0 + s1 vs s1 | +0.047 [+0.034, +0.061] | +0.039 | -0.000 | 0.062 | +0.017 | 14.30 |
| urc = recall + recall_s1 vs recall | +0.007 | +0.004 | -0.002 | | +0.004 | |
| ceiling u4 vs s1 | +0.080 | +0.068 | -0.012 | | +0.027 | |
| u23 with spurs kept | 0.798 | 0.783 | 0.737 | 0.066 | 0.090 | 14.83 (over) |

  val250 (report only): u23 with vtx_r 1.9 keeps lanes found 0.783 at false length 0.060; with vtx_r 2.5 it matches s3's node-id vertices (14.86 vs 14.93) at +0.026 lanes found.
- **What it says**: the union of two differently trained seeds is the largest coverage lever of the loop (+0.047 lanes found, +0.040 R@1 far, replicated on both pairs, prediction +0.045), and it is not a lower threshold (+0.038 over the parent at equal false length). It needs seed diversity: two rc1 seeds give only +0.007. It costs a second network forward and the false-length guard sits at its edge (0.0635 vs 0.062). Against tonight's prototype `recall_sp0_dp` (lanes found 0.781) it would add only +0.005, under the +0.010 needed to replace a simpler candidate.
- **Weaknesses / next**: one held-out pair; timing not measured (two forwards; the two-member latency option `--name2` exists, untested on the GPU); the val250 choices are on frames that overlap the 873. Next loop: the union with vtx_r 1.9 - 2.1 (and spurs kept) on 873 frames with its idle timing; a union of the two rc1-family seeds that already carry spurs-kept assembly is the natural prototype step if it fits 14.3 DP and 0.062 false length.
