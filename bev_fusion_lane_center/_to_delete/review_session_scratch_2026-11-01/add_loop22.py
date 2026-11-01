"""throw-away (ar-planner, loop 22): planned ledger rows + schedule rows. Run from the problem folder: wsl python3 <this> [--dry]"""
import subprocess
import sys

DRY = "--dry" in sys.argv
AR = "../autoresearch/tools/"
TH = "env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMBA_NUM_THREADS=2"
PY = "bash tools/env/_py14.sh"
EV = PY + " evaluate.py"
FS = "experiments/E261_recall-weight-6-operating-point/frontier_scratch.py"
CS = "experiments/E265_far-coverage-loss-budget/cleanup_split.py"
RS = "experiments/E265_far-coverage-loss-budget/ring_split.py"
STORED = {"recall": (0.7404, 0.7673, 0.7714), "recall_s1": (0.7376, 0.7695, 0.7707), "fast_recall": (0.7392, 0.7364, 0.7275),
          "e24_j1rc_s1": (0.7427, 0.7433, 0.7368), "e24_j1rc_s2": (0.7354, 0.7360, 0.7341), "e24_j1rc_s3": (0.7391, 0.7393, 0.7363),
          "champion": (0.7350, 0.6948, 0.7087), "fast_recall_na": (0.7388, 0.7343, 0.7264)}


def chain(cmds, threads=True):
    return (TH + " " if threads else "") + "bash -c '" + " && ".join(cmds) + "'"


def expect(n, tag):
    q, l, r = STORED[n]
    return f"{EV} {n} --tag {tag} --expect Q={q:.4f} lanes_found={l:.4f} R@1_far={r:.4f}"


REP = ["recall_s1", "e24_j1rc_s1", "e24_j1rc_s2", "e24_j1rc_s3"]
P18_SET = ("reach=null spf2=false junc=true temporal=true stage_b=false stage_c=true b=3 mcl=true ver_mode=\\\"survival\\\" link_mode=\\\"learned\\\" "
           "pi_mode=\\\"min\\\" detach_pos=false dedup=\\\"geo\\\" ctx=\\\"global\\\" pattern=\\\"teardrop\\\" seeds=\\\"lattice\\\" warm=true revisit=false "
           "constr=\\\"L\\\" D=1 seed_gscale=1.0 tolerant=false local_r=12.0 M=288 K0=144")
TAUS = "0.40 0.45 0.50 0.55 0.60 0.65"

# item, node, kind, resource, budget, priority, owner, depends, command, note, (hyp, pred, rule) or None
ITEMS = [
    # ---- system / code (agent)
    ("S5.commit", "S5", "agent", "agent", "5m", 1, "lead", "", "", "commit experiments/E261-E267 (cards, scripts, experiment.json) by path before the first CPU run: run.py snapshots code_dirs only", None),
    ("S1.code", "S1", "code", "agent", "60m", 1, "coder", "", "", "evaluate.builder_raw cache key = sorted extra set items; spec test two entries -> two files; golden 6/6; ar-reviewer (code). Ready 05:30", None),
    ("S2.code", "S2", "code", "agent", "60m", 2, "coder", "", "", "p16_builder lattice_ring / lattice_ext (default bit-identical) + topo_metrics R@1_far_inner / R@1_far_ring (reproduces step0_ring.txt); spec tests; reviewer. Ready 05:30", None),
    ("S3.code", "S3", "code", "agent", "90m", 3, "coder", "", "", "p16_infer.union_raw + registry kind union (members; tau / asm overrides) + p18_latency two-member option; identity union = member graph hash; reviewer. Ready 05:30", None),
    # ---- E265 (CPU now)
    ("E265.c", "N1", "evaluate", "cpu", "15m", 1, "lead", "", chain([expect(n, "l22") for n in ["recall", "recall_s1", "fast_recall", "e24_j1rc_s1", "e24_j1rc_s2", "e24_j1rc_s3", "champion", "fast_recall_na"]]),
     "layer 1: control cards of E261 / E262 / E264 / E265 re-scored with tonight's code (tag l22); recall_s1 reference = E261 scratch line (no stored card)",
     ("the registry entries reproduce their stored numbers with tonight's code (layer 1); these cards are the matched controls of E261 / E262 / E264 / E265",
      "all 8 reproduce Q / lanes found / R@1 far within 5e-4", "any --expect miss -> stop, the harness is wrong (E265 kill)")),
    ("E265.a", "N1", "evaluate", "cpu", "30m", 2, "lead", "E265.c", chain([f"{EV} {n} --asm spur_len=0 --tag sp0 --compare {n}" for n in REP + ["recall", "fast_recall", "champion", "fast_recall_na"]]),
     "arm a (spurs kept) on the 4 replication checkpoints, then selection (recall, fast_recall) and scope (champion, fast_recall_na), report only",
     ("switching off the spur half of the clean-up (spur_len 0) raises lanes found at range without guard losses (E265, re-decision of E259 on the per-piece DP basis)",
      "lanes found +0.012 (+0.008 .. +0.016), R@1 far +0.007, Q +0.003 on the 4 replication checkpoints",
      "E265 experiment.json: KEEP if mean lanes found >= +0.010 and R@1 far >= +0.004, guards on every checkpoint (Q -0.003, topo -0.005, FL / dup +0.005, P@1 far -0.010, DP <= 14.3), above the equal-budget tau frontier; TIE with guards = spurs kept (cheaper); CUT otherwise")),
    ("E265.v", "N1", "evaluate", "cpu", "25m", 2, "lead", "", f"{TH} {PY} {CS} " + " ".join(REP + ["champion", "fast_recall_na"]),
     "per-piece DP (budget basis) and node-id vertices of c / a / placebo (bubble_r 0) per checkpoint",
     ("the DP guard and the placebo of E265: spurs kept stay <= 14.3 per-piece DP; bubbles kept change nothing",
      "DP +0.6 .. +0.9 with spurs kept, all <= 14.3; placebo lanes found within +-0.003", "a checkpoint above 14.3 with spurs kept -> E265 CUT")),
    ("E265.b", "N1", "evaluate", "cpu", "20m", 3, "lead", "E265.a", chain([f"{EV} {n} --asm spur_len=2.5 --tag sp25 --compare {n}" for n in REP + ["recall"]]),
     "arm b (spur_len 2.5)",
     ("a shorter spur limit keeps most of arm a's coverage with fewer vertices", "lanes found within 0.004 of arm a, DP -0.3",
      "b replaces a only if lanes found within 0.004 of a with >= 0.3 fewer per-piece DP vertices")),
    ("E265.eb", "N1", "evaluate", "cpu", "30m", 3, "lead", "", f"{TH} {PY} {FS} e123_D7_r3_rc1:288:e123_D7_r3_rc1,e18_D7_j1rcs1:192:e18_D7_j1rcs1 0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7",
     "equal-budget control: tau frontier of rc1 and j1rc s1 down to 0.1 with per-piece DP (referee fix 1); 0.7 = layer 1",
     ("spending the same vertices on a lower tau buys less than spurs kept (equal-budget control of E265)",
      "at arm a's DP the frontier is about equal in lanes found and lower in Q (about -0.004)",
      "E265 section 2 equal-budget rule: arm a must be >= +0.006 lanes found, or >= +0.005 Q at lanes found >= -0.002, above the frontier at its DP; else CUT (budget spend)")),
    ("E265.dec", "N1", "decide", "agent", "15m", 2, "lead", "E265.a;E265.b;E265.v;E265.eb", "",
     "decide.py --experiment E265 --write + lead checks; KEEP / TIE-with-guards -> registry recall_sp0 / recall_sp0_dp, lane_prototype.py recall_sp0_dp (candidate S)", None),
    # ---- E261 (CPU now: rung)
    ("E261.g", "N2", "evaluate", "cpu", "15m", 1, "lead", "", chain([f"{PY} {FS} e123_D7_r3_rc1:288:e123_D7_r3_rc1 0.7,0.2,0.1,0.05 fork_prune={p}" for p in ("0.4", "0.5", "0.6")]),
     "cheap rung (kill test) of E261: rc1's threshold frontier down to tau 0.05 at pruning 0.4 / 0.5 / 0.6 (referee fix 1); 0.7 / 0.4 and 0.7 / 0.5 = layer 1",
     ("rc1's own threshold cannot reach rc2's coverage: the best admissible rc1 point stays >= 0.010 lanes found below rc2's 0.7902 (E261 rung)",
      "best admissible rc1 point lanes found about 0.777 - 0.779 (tau 0.2, pruning 0.4); tau 0.1 / 0.05 fail Q >= 0.735",
      "admissible = Q >= 0.735, DP <= 14.3, FL <= 0.062, dup <= 0.1063; best admissible rc1 >= 0.7802 -> E261 CUT (no CV, E262 cancelled); else E261.cv")),
    ("E261.code", "N2", "code", "agent", "30m", 2, "lead", "E261.g", "", "temp/loop22/e261_cv.py (spec in E261 section 2: frame records of the 60-point grid, layer-1 asserts, 20 fixed splits); only if the rung did not kill E261", None),
    ("E261.cv", "N2", "evaluate", "cpu", "70m", 2, "lead", "E261.g;E261.code", f"{TH} {PY} temp/loop22/e261_cv.py",
     "20-split out-of-fold choice of the rc2 and rc1 operating points (VAL, selection-CV, layer 3)",
     ("rc2 at an out-of-fold operating point beats recall and rc1's own best admissible point on held-out scene halves (E261)",
      "held-out mean: lanes found a - c +0.022, R@1 far +0.015, Q -0.004, a - t +0.013; DP(a) > 14.3 in <= 5 of 20 halves",
      "E261 experiment.json: PASS if a - c lanes found >= +0.015, R@1 far >= +0.010, Q(a) >= 0.735 and loss <= 0.005, topo loss <= 0.010, FL <= 0.062, dup <= +0.030, DP rule, a - t >= +0.010; TIE / CUT as the card")),
    ("E261.reg", "N2", "agent", "agent", "5m", 2, "lead", "E261.cv", "", "registry entries recall6 (rc2 at the all-22 choice) and recall_thr (rc1 at its all-22 choice)", None),
    ("E261.ev", "N2", "evaluate", "cpu", "10m", 2, "lead", "E261.reg;E265.c", chain([f"{EV} recall6 --compare recall", f"{EV} recall_thr --compare recall"]),
     "the 22-scene cards of a and t against recall (paired scene bootstrap)",
     ("rc2 at its point (recall6) gives the coverage gain over recall on the 22-scene bootstrap (E261, VAL, selection-CV)",
      "recall6 - recall: lanes found +0.023, R@1 far +0.015, Q -0.004; recall_thr - recall lanes found +0.009",
      "as E261.cv (decide.py --experiment E261 + held-out means + lead checks)")),
    ("E261.dec", "N2", "decide", "agent", "15m", 2, "lead", "E261.ev", "", "decide.py --experiment E261 --write + held-out means + lead checks; PASS -> E262 at 06:00, candidate W", None),
    # ---- code landed: checks and registry
    ("E266.gold", "S1", "evaluate", "cpu", "10m", 1, "lead", "S1.code;S2.code;S3.code", chain([f"{PY} evaluate.py --golden check --cached-only", "bash tools/maintenance/run_all_tests.sh"], threads=False),
     "layer 0 / 1 after the code: golden numbers and the test suite",
     ("the cache-key, ring and union code leaves every default graph and golden number unchanged", "golden 6 of 6, all tests pass", "any miss -> no ring / union run until fixed")),
    ("L22.reg", "S1", "agent", "agent", "10m", 1, "lead", "E266.gold", "", "registry entries recall_ring, recall_s1_ring, fast_recall_ring, e24_j1rc_s1_ring (set + lattice_ring 47.5); ens_j1rc_23, ens_j1rc_01, ens_j1rc_0123, ens_rc1 (kind union)", None),
    # ---- E264
    ("E264.u23", "N5", "evaluate", "cpu", "10m", 1, "lead", "L22.reg", f"{TH} {EV} ens_j1rc_23 --compare e24_j1rc_s3",
     "the held-out union card, 05:30 - 05:40 (before the quiet window)",
     ("the path union of the raw trees of j1rc s2 + s3 finds more lanes than the better parent s3 (E264, held-out pair)",
      "lanes found +0.045, R@1 far +0.035, Q +0.000, FL +0.007, dup +0.015 against s3",
      "E264 experiment.json: KEEP if lanes found >= +0.030, R@1 far >= +0.025, Q / topo loss <= 0.005, FL <= 0.062 abs, dup <= +0.020, DP <= 14.3, u - m >= +0.010, u01 same signs, ms <= 30 idle (> 32 CUT)")),
    ("E264.t", "N5", "timing", "gpu", "20m", 1, "lead", "S3.code;L22.reg",
     "bash -c 'for r in 1 2 3; do bash tools/env/_py14.sh p18_latency.py --name e24_D7_j1rcs2 --members e24_D7_j1rcs2,e24_D7_j1rcs3 --M 192 --K0 96 --rec 0 --fork_prune 0.4 --tag e264_u23_r$r && bash tools/env/_py14.sh p18_latency.py --name e24_D7_j1rcs3 --M 192 --K0 96 --rec 0 --fork_prune 0.4 --tag e264_s3_r$r && bash tools/env/_py14.sh p18_latency.py --name e123_D7_r3_rc1 --rec 1 --fork_prune 0.4 --tag e264_rc1_r$r || exit 1; done'",
     "05:45, GPU AND CPU idle (no other job of any chat, CPU arms paused); the two-member option name is the one S3 builds (adjust --members before the start)",
     ("the union of two j1rc members (two builder forwards + union assembly) stays in the speed class (E264 fix 3)",
      "union 30 - 34 ms, single j1rc 21 ms, recall (rec 1) 27 ms (median of the replicate medians of total)",
      "union <= 30 ms: E264 can be the prototype; 30 - 32: quality only; > 32: E264 CUT")),
    ("E264.u", "N5", "evaluate", "cpu", "20m", 2, "lead", "E264.t", chain([f"{EV} ens_j1rc_01 --compare e24_j1rc_s1", f"{EV} ens_j1rc_0123 --compare e24_j1rc_s1", f"{EV} ens_rc1 --compare recall"]),
     "exploration pair u01 (replication signs), ceiling u4, no-diversity control urc",
     ("the union gain replicates on s0 + s1, grows with four seeds and vanishes without diversity (rc1 + rc1s1)",
      "u01 lanes found +0.047 over s1; u4 about 0.82; urc < +0.010 over recall", "u01 must have the same signs as u23 (lead check of E264)")),
    ("E264.v", "N5", "evaluate", "cpu", "10m", 2, "lead", "E264.t", f"{TH} {PY} {CS} ens_j1rc_23 ens_j1rc_01",
     "per-piece DP of the unions (referee fix 2: measured before any verdict)",
     ("the union stays inside the per-piece DP budget", "u23 per-piece DP about 13.6", "> 14.3 -> E264 CUT")),
    ("E264.mfl", "N5", "evaluate", "cpu", "15m", 3, "lead", "E264.t",
     chain([f"{EV} ens_j1rc_23 --subset val250 --tag u250", f"for t in {TAUS}; do {EV} e24_j1rc_s3 --subset val250 --tau $t --tag t$t || exit 1; done",
            f"{EV} ens_j1rc_01 --subset val250 --tag u250", f"for t in {TAUS}; do {EV} e24_j1rc_s1 --subset val250 --tau $t --tag t$t || exit 1; done"]),
     "matched-false-length choice on val250 (lead reads t* = tau whose val250 FL is closest to the union's)",
     ("a single parent at a lower tau reaching the union's false length does not reach its lanes found (matched-FL control, choice on val250)",
      "t* about 0.45 - 0.55", "input of E264.mfl2")),
    ("E264.code", "N5", "code", "agent", "20m", 3, "lead", "E264.mfl", "", "temp/loop22/e264_heldout.py: frame records of u and m, bootstrap.pooled over the VAL frames not in val250 (the 623-frame read)", None),
    ("E264.mfl2", "N5", "evaluate", "cpu", "15m", 3, "lead", "E264.mfl;E264.code", "",
     "command filled by the lead with t*: evaluate.py e24_j1rc_s3 --tau <t*> --tag mfl --compare e24_j1rc_s3 && evaluate.py e24_j1rc_s1 --tau <t*01> --tag mfl --compare e24_j1rc_s1 && _py14.sh temp/loop22/e264_heldout.py",
     ("the union is not a disguised lower threshold: u - m >= +0.010 lanes found on 623 held-out frames and on 873",
      "u - m lanes found +0.025, R@1 far +0.020", "u - m < +0.010 -> E264 TIE (keep the single model); within 0.005 -> kill")),
    ("E264.mv", "N5", "evaluate", "cpu", "15m", 5, "lead", "E264.mfl", chain([f"for v in 1.5 1.7 1.9 2.1 2.3 2.5; do {EV} ens_j1rc_23 --subset val250 --asm vtx_r=$v --tag vx$v || exit 1; done"]),
     "matched-vertex union, report only (then --asm vtx_r=<V*> --tag mv --compare e24_j1rc_s3 by the lead)",
     ("at the parent's node-id vertex count the union keeps most of its gain (report)", "Q +0.000, lanes found +0.040 at matched vertices", "report only")),
    ("E264.dec", "N5", "decide", "agent", "15m", 3, "lead", "E264.u23;E264.t;E264.u;E264.v;E264.mfl2", "", "decide.py --experiment E264 --write + lead checks (FL abs, DP, u - m, u01 signs, ms); KEEP -> candidate U", None),
    # ---- E266
    ("E266.c", "N3", "evaluate", "cpu", "10m", 2, "lead", "E266.gold;E264.t", chain([expect(n, "l22r") for n in ["recall", "recall_s1", "fast_recall", "e24_j1rc_s1"]]),
     "control cards with the ring metric (tag l22r): second layer-1 check after the code change",
     ("the code change leaves the registry numbers unchanged and the cards carry R@1_far_inner / R@1_far_ring", "4 of 4 reproduce", "a miss -> stop the ring / union runs")),
    ("E266.b0", "N3", "evaluate", "gpu", "8m", 1, "lead", "L22.reg;E264.t", f"{EV} recall_ring --compare recall",
     "ring raw outputs of recall computed once on the GPU and cached under the S1 key (stop-gap without S1: --fresh, card val_fresh.json)",
     ("seed candidates on the +-47.5 m ring let the builder reach the 45 - 48 m ring (E266)",
      "recall R@1 far 0.771 -> 0.783 (+0.012), ring 0.702 -> 0.74, inner within +-0.003, lanes found +0.006, Q 0.740 +-0.004, DP <= 13.1",
      "E266 experiment.json: KEEP if mean R@1 far >= +0.010 (recall alone >= +0.010, interval > 0, 3 of 4 signs), inner-far +-0.005, Q / topo / FL loss <= 0.005, dup <= +0.010, Q >= 0.735, DP <= 14.3; G0 / G1 kill")),
    ("E266.g01", "N3", "evaluate", "cpu", "10m", 1, "lead", "E266.b0", f"{TH} {PY} {RS} recall_ring recall",
     "G0 (ring cache is its own file; kept seed share beyond 45 m >= 0.110; displaced inner share) and G1 (all-valid tree ring R@1 >= 0.82)",
     ("the ring flag changes the output (G0) and the tree grows ring lanes from ring seeds (G1)", "kept-seed share beyond 45 m about 0.12; tree ring R@1 0.800 -> 0.83",
      "G0 fails after one debug round or G1 < 0.82 -> E266 killed, b1 - b3 and E267 not run")),
    ("E266.b1", "N3", "evaluate", "gpu", "8m", 2, "lead", "E266.g01", f"{EV} recall_s1_ring --compare recall_s1", "seed twin, also needed by E267",
     ("the ring gain replicates on rc1 seed 1", "R@1 far +0.012", "as E266.b0")),
    ("E266.b2", "N3", "evaluate", "gpu", "6m", 4, "lead", "E266.b1", f"{EV} fast_recall_ring --compare fast_recall", "fast chain checkpoint",
     ("the ring gain holds on the fast chain (j1rc s0)", "R@1 far 0.7275 -> 0.742", "as E266.b0")),
    ("E266.b3", "N3", "evaluate", "gpu", "6m", 4, "lead", "E266.b2", f"{EV} e24_j1rc_s1_ring --compare e24_j1rc_s1", "fast chain seed twin",
     ("the ring gain holds on j1rc s1", "R@1 far +0.014", "as E266.b0")),
    ("E266.v", "N3", "evaluate", "cpu", "15m", 2, "lead", "E266.b3", f"{TH} {PY} {CS} recall_ring recall_s1_ring fast_recall_ring e24_j1rc_s1_ring",
     "per-piece DP of arm b (row registry) and of E267's sr (row spurs kept)",
     ("ring (and ring + spurs kept) stay inside the per-piece DP budget", "recall_ring DP about 13.0; sr about 13.9", "> 14.3 on a checkpoint -> guard fails")),
    ("E266.dec", "N3", "decide", "agent", "15m", 2, "lead", "E266.b3;E266.c;E266.v", "", "decide.py --experiment E266 --write + lead checks; KEEP -> candidate R", None),
    # ---- E267
    ("E267.s", "N4", "evaluate", "cpu", "10m", 2, "lead", "E266.c", chain([f"{EV} recall --asm spur_len=0 --tag sp0r --compare recall", f"{EV} recall_s1 --asm spur_len=0 --tag sp0r --compare recall_s1"]),
     "arm s of the 2 x 2 with the ring metric",
     ("spurs kept on rc1 seeds 0 / 1 (E267 arm s)", "lanes found +0.013, R@1 far +0.008", "E267 experiment.json")),
    ("E267.sr", "N4", "evaluate", "cpu", "10m", 2, "lead", "E266.b1;E266.c", chain([f"{EV} recall_ring --asm spur_len=0 --tag sp0 --compare recall", f"{EV} recall_s1_ring --asm spur_len=0 --tag sp0 --compare recall_s1"]),
     "arm sr of the 2 x 2 (ring + spurs kept)",
     ("spurs kept and the seed ring act on different losses and add (E267)",
      "sr - c lanes found +0.017, R@1 far +0.017, Q +0.002; interaction -0.004 .. 0",
      "E267 experiment.json: KEEP if sr - c lanes found and R@1 far >= +0.015 (both seeds), sr - s R@1 far >= +0.005, sr - r lanes found >= +0.005, guards; else the single settings as decided")),
    ("E267.dec", "N4", "decide", "agent", "10m", 2, "lead", "E267.s;E267.sr;E266.v;E265.dec;E266.g01", "", "decide.py --experiment E267 --write + lead checks; KEEP -> candidate SR", None),
    # ---- E262 (gated)
    ("E262.a1", "N2a", "train", "gpu", "60m", 2, "lead", "E261.dec;E264.t",
     "bash -c 'export LCR_DIST_DIR=$HOME/exp_crn_b123/feats; bash tools/env/_py14.sh p16_train.py --design D7 --name e26_D7_rc2s1 --u 4 --seed 1 --init e123_D7_r3_f1 --lr 0.0005 --stem_lr 0.3 --workers 4 --set " + P18_SET + " --w rec=6.0 --warm_frac 0.4 --eval val250'",
     "ONLY if E261 PASS; run.py --gpu --seed 1; first item to drop if the GPU window shrinks",
     ("a second seed of the rc2 recipe reproduces rc2 (E262 seed twin)", "rc2s1 at the E261 point: lanes found 0.789 +- 0.008, R@1 far 0.786, Q 0.736, DP 14.2",
      "kill: val250 Q < 0.725 at the end, or total loss > 1.5 x rc2's at the same iteration; E262 experiment.json after E262.ev")),
    ("E262.ev", "N2a", "evaluate", "gpu", "8m", 2, "lead", "E262.a1", f"{EV} recall6_s1 --compare recall_s1",
     "lead adds registry recall6_s1 first (run e26_D7_rc2s1, point of recall6)",
     ("rc2 beats rc1 on two paired seeds at the E261 point (E262)", "two-seed means: lanes found +0.021, R@1 far +0.015, Q -0.003, dup +0.011",
      "E262 experiment.json: KEEP if lanes found >= +0.015, R@1 far >= +0.010, Q loss <= 0.005, topo <= 0.010, dup <= +0.030, both seeds; Q >= 0.735, DP <= 14.3, FL <= 0.062 each")),
    ("E262.dec", "N2a", "decide", "agent", "10m", 2, "lead", "E262.ev", "", "decide.py --experiment E262 --write + lead checks; KEEP -> candidate W with two seeds", None),
    # ---- prototype, TEST, P4
    ("P.proto", "N7", "agent", "agent", "30m", 1, "lead", "E265.dec;E261.dec", "",
     "06:30: the prototype of record by the PLAN.md rule from the verdicts decided by then (fallback recall_dp); registry <name>_dp, lane_prototype.py reproduces its card, video; updated until ~07:15 by E266 / E267 / E264 / E262", None),
    ("E268.card", "N7", "agent", "agent", "15m", 1, "lead", "P.proto", "",
     "card E268 (TEST check of the prototype vs the control, report only); grant under D28 for evaluation/TEST_USAGE_LOG.md; log row naming E268.p BEFORE the run", None),
    ("E268.p", "N7", "test", "gpu", "15m", 1, "lead", "E268.card", "",
     "command filled when the prototype is fixed: run.py --id E268.p --split test --gpu -- bash tools/env/_py14.sh evaluate.py <name>_dp --split test --test-id E268.p --compare control (TEST raw outputs need the GPU, ~4 min per network)", None),
    ("P4", "N7", "agent", "agent", "60m", 1, "lead", "E268.p", "",
     "summaries (section 3 of E261-E267), morning page, check_numbers.py, ar-reviewer (numbers), check_structure.py, run_all_tests.sh, HANDOVER, shutdown when nothing of ours runs", None),
]


def run(args):
    print(" ".join(args) if DRY else "", flush=True)
    if not DRY:
        r = subprocess.run(args, capture_output=True, text=True)
        print((r.stdout + r.stderr).strip()[:300])
        if r.returncode:
            sys.exit(f"failed: {args[:4]}")


for item, node, kind, res, budget, prio, owner, deps, cmd, note, card in ITEMS:
    is_run = card is not None
    if is_run:
        hyp, pred, rule = card
        run(["python3", AR + "ledger.py", "add", item, "--hyp", hyp, "--pred", pred, "--rule", rule, "--status", "planned"])
    a = ["python3", AR + "schedule.py", "add", item, "--node", node, "--kind", kind, "--resource", res, "--budget", budget,
         "--priority", str(prio), "--owner", owner, "--note", note]
    if deps:
        a += ["--depends", deps]
    if is_run or item == "E268.p":
        a += ["--runs", item]
    if cmd:
        a += ["--command", cmd]
    run(a)
