"""Loop 18 (one-off): experiment folders E209 (rerun) and E213 - E222, E230 with README, small results and figures (the scheme of experiments/: README.md, results/, figures/).
Usage (any python): python3 make_exp_folders.py   (run after the numbers are final; overwrites the READMEs of these folders)"""
import bflc_paths as BP  # project paths from the location of this checkout
import glob
import os
import re
import shutil

R = BP.ROOT
EX = f"{R}/experiments"
E17, E18, FIG = f"{R}/runs/e17", f"{R}/runs/e18", f"{R}/reports/source/figures"
T = "T11 - Fork pruning, map localisation and failure-trained navigation (loop 18)"
TAG = {"E": "E - E2E-usable: learned or parameter-free, differentiable (the trace network has discrete bookkeeping inside)",
       "P": "P - with a classical step (a rule or a geometric stage next to it)", "D": "D - dropped: tried, no gain or worse", "C": "C - classical / reference / analysis"}

# id, folder slug, title, tag, K ids, result text, files to copy {dest: source glob}, figures
EXPS = [
    ("E209", "unseen-failure-types-builder-trained-without", "Unseen failure types: map builder and its no-map control trained WITHOUT night and blur rows, evaluated on night and blur (rerun with checkpoints)", "E", "",
     "Failure half, exact map, paired scene bootstrap of F1@1: single frame night S 0.504 / F1 0.465 (no-map control 0.454 / 0.373, champion 0.419 / 0.312, builder that saw night 0.568 / 0.619), map +0.093 [+0.065, +0.122]; blur single 0.577 / 0.617 (control 0.530 / 0.531), +0.086 [+0.053, +0.125]; "
     "with the spatial hold night 0.655 / 0.749 (control 0.635 / 0.711, seen 0.662 / 0.780), +0.038 [+0.024, +0.057]; blur 0.661 / 0.773 (control 0.631 / 0.723, seen 0.660 / 0.782), +0.050 [+0.031, +0.070]. The map benefit transfers to unseen failure types.",
     {"e209.md": f"{E17}/tables/e209.md"}, ["p18_e209_unseen.png"]),
    ("E213", "rough-localisation-and-map-trust", "Rough localisation of the SD map (pose-grid soft-argmax + causal world-frame accumulation) and a map trust gate in front of the map-conditioned builder", "E", "K188, K189",
     "Pose error of the map 2.55 m -> 0.82 m (night failure half, sigma 2 m); failure-half S of builder + gate: none 0.651, unregistered 2 m 0.643, registered 0.664, exact 0.676 (night); paired F1@1 registered - none +0.035 [+0.018, +0.059]; "
     "wrong map 0.605 -> 0.647 with the trust gate 0.75 (none 0.651); exact map keeps 72 - 87 % of its benefit at 0.75, 88 - 99 % at 0.62 (best chain). S gate (none + 0.015) missed by a hair, F1@1 gate met.",
     {"e213.md": f"{E17}/tables/e213.md", "e213b.md": f"{E17}/tables/e213b.md", "e213c.md": f"{E17}/tables/e213c.md", "regis_train_status.json": f"{E18}/regis_a/status.json"}, ["p18_registration.png", "p18_arch.png"]),
    ("E214", "joint-fine-tune-with-failure-frames", "Longer joint fine-tune of builder + route head with real failure frames (arm a: route loss reaches the builder; arm c: only the route head learns), two route-head seeds", "D", "",
     "Joint training: no gain (arm a - arm c between -0.073 and +0.007 failure-half route success). Route head trained on failure frames (arm c): night 0.372 -> 0.591 / 0.562, blur 0.584 / 0.547 -> 0.781 / 0.803, three cameras off 0.423 / 0.401 -> 0.555 / 0.613; clean half 0.846 / 0.816 -> 0.875 / 0.868.",
     {"j2_table.txt": f"{E18}/j2_table.txt"}, ["p18_route_failure_training.png"]),
    ("E215", "fork-pruning", "Fork pruning: forks without junction-head support are un-linked (geometry unchanged)", "E", "K190",
     "VAL 873 frames, threshold 0.3: S 0.746 -> 0.769, Q 0.701 -> 0.735, J 0.265 -> 0.409 (control 0.408), split P 0.18 -> 0.38, F1@1 0.870 and lanes found 0.695 unchanged, vertices per 100 m 13.5 -> 14.8; paired scene bootstrap S +0.024 [+0.017, +0.030], J +0.144 [+0.104, +0.182]; "
     "route success with the route head along the edges 0.894 / 0.891 -> 0.899 / 0.900; same effect on j1 192 / 96 (J 0.283 -> 0.394) and rc1 (J 0.182 -> 0.428).",
     {"table.md": f"{E18}/fork_prune_val/table.md", "results.json": f"{E18}/fork_prune_val/results.json", "fork_features.npz": f"{E18}/fork_prune_val/fork_features.npz", "asm_grid.log": f"{E18}/asm_grid.log"}, ["p18_forkprune.png", "p18_frames_forkprune.png"]),
    ("E216", "numba-union-kernels", "Numba kernels for the path union of the assembly (identical graphs)", "E", "K191",
     "873 / 873 validation graphs bit-identical to the loop-16b fast path; about 2x faster under load; idle timing in the loop-18 report.", {}, []),
    ("E217", "far-range-recall-negatives", "Far-range recall gap to the control: beam budget, per-level threshold, validity sweep, candidate coverage", "D", "",
     "58 % of the recall gap lies beyond 40 m (recall 0.709 vs 0.811); beams 400 / 200 and 576 / 288: lanes found 0.700 / 0.701; the tree has two levels (per-level threshold: 0.707); tau 0.2: R@1 0.858; all valid nodes cover 85 % of the far points, the admitted 74 %. Not a budget or threshold effect.", {}, []),
    ("E218", "fork-pruning-under-failure-and-best-chain", "Fork pruning under sensor failure and in the best chain (map builder + gate + registration + trust)", "E", "K190",
     "Clean half S 0.752 -> 0.780 (hold), J 0.227 -> 0.384; failure half of the best chain S 0.664 -> 0.682 (night, registered, trust 0.75), J 0.158 -> 0.247; single-frame failure S does not improve (fragments add vertices).", {"e218.md": f"{E17}/tables/e218.md"}, []),
    ("E219", "recall-weight-finetune-under-pruning", "Recall-weighted fine-tune of the champion re-opened under fork pruning (weights 3 / 6, range-weighted)", "E", "K184, K190",
     "Fast j1rc (recall weight 3 on j1 192 / 96): lanes found 0.671 -> 0.736, J 0.398 -> 0.447, Q 0.729 -> 0.739 for +2 ms (VAL, 2 seeds; equal-budget control: only lanes found and J are the weight's). Slow variants: rc1 (weight 3) + pruning 0.4: Q 0.740, lanes found 0.767, J 0.428; rc2 (weight 6): Q 0.735, lanes found 0.793 (a tie with the control 0.800), precision 0.858; rc4 (extra weight beyond 25 m): Q 0.736, lanes found 0.770. The rules written beforehand (Q >= 0.745 with lanes found >= 0.78) were met by none; rc1 stays the recall variant. Extension: seed 1 of rc1 reproduces it (Q 0.738, lanes found 0.770); the same weight-3 fine-tune on the fast j1 192 / 96 (j1rc) lifts lanes found 0.671 -> 0.736, J 0.398 -> 0.447, Q 0.729 -> 0.739 (paired, all intervals above zero) for +2 ms (19.9 vs 17.3 ms): second seed: Q 0.743, lanes found 0.743, J 0.451, 21.8 ms. An equal-budget control (same fine-tune at recall weight 1, j1eq: Q 0.735, J 0.417, lanes found 0.674) shows that the recall weight buys lanes found (+0.062) and J (+0.030) while the Q gain over j1 is mostly extra fine-tuning. Adopted as the fast recall variant on VAL (two seeds, no TEST). The control of rc1 (f1eq) has lanes found 0.707: rc1 buys +0.060 lanes found and nothing measurable in Q or J. Weight 6 on j1 (j1rc6): lanes found 0.758 but Q 0.736 (j1rc 0.739): j1rc stays. The plain fine-tune at twice the budget (j1eq8, 8 u): Q 0.731, J 0.402, lanes found 0.678 - no further gain, the 4 u control j1eq stays.", {"rc1.json": f"{E18}/rc_eval/rc1.json", "rc2.json": f"{E18}/rc_eval/rc2.json", "rc4.json": f"{E18}/rc_eval/rc4.json", "rc1s1.json": f"{E18}/rc_eval/rc1s1.json", "j1rc.json": f"{E18}/rc_eval/j1rc.json", "j1rcs1.json": f"{E18}/rc_eval/j1rcs1.json", "j1eq.json": f"{E18}/rc_eval/j1eq.json", "j1eq8.json": f"{E18}/rc_eval/j1eq8.json", "j1eqs1.json": f"{E18}/rc_eval/j1eqs1.json", "f1eq.json": f"{E18}/rc_eval/f1eq.json", "j1rc6.json": f"{E18}/rc_eval/j1rc6.json", "j1p4.json": f"{E18}/rc_eval/j1p4.json", "final_boot.json": f"{E18}/final_boot.json"}, []),
    ("E220", "failure-trained-route-head-in-the-best-chain", "Route heads trained on failure frames in the best chain (builder B + gate, exact map, fork pruning)", "E", "K190",
     "Failure half of the best chain, route success original heads s0 / s1 -> failure-trained c0 / c1: night 0.832 / 0.818 -> 0.847 / 0.847, blur 0.883 / 0.854 -> 0.905 / 0.891, three cameras off 0.774 / 0.825 -> 0.832 / 0.847. Rule (>= +0.10) not met: the map builder already carries the route.", {"e220_table.txt": f"{E18}/e220_table.txt"}, []),
    ("E221", "calibration-of-fork-heat-and-node-validity", "Calibration of the junction-head heat at predicted forks and of the node validity (literature gap 3)", "C", "",
     "Fork heat is calibrated (ECE 0.034 split / 0.037 merge); node validity is bimodal and not calibrated in the middle (ECE 0.121 raw, 0.021 after an isotonic fit).", {}, []),
    ("E222", "planner-path-error-in-the-new-chain", "Planner consumer path error at 20 m in the best chain with and without fork pruning and with the failure-trained route head", "E", "",
     "Failure half, exact map: path error at 20 m 0.28 / 0.27 / 0.28 m (night / blur / three cameras off) without pruning, 0.28 / 0.27 / 0.27 with pruning, 0.28 / 0.27 / 0.26 with pruning and the failure-trained head: no measurable planner-level effect (floor 0.27 m).", {}, []),
    ("E231", "recall-variant-under-sensor-failure", "The j1 family (j1, equal-budget controls, recall variants j1rc) on the loop-17 failure series", "E", "K190",
     "Failure half, j1rc - equal-budget control j1eq with the spatial hold: lanes found +0.052 / +0.058 / +0.071 (night / blur / three cameras off), Q +0.002 / -0.001 / -0.038 [-0.058, -0.020]. Rule not met (Q below the control for three cameras off): the recall weight is not uniformly the robust choice; the plain longer fine-tune is the safer fast variant where cameras can drop out.", {"e231_table.json": f"{E18}/e231_table.json"}, []),
    ("E233", "range-binned-recall-of-the-recall-variants", "Recall@1 m per range bin of the recall variants against their equal-budget controls and the control E94 (WP4 analysis)", "C", "",
     "Beyond 40 m: j1 0.677, j1eq 0.688, j1rc 0.727, j1rc6 0.746; f1 0.709, f1eq 0.724, rc1 0.771, rc2 0.790; control 0.811. The recall weight lifts every bin, most beyond 40 m (rc1 closes 54 % and rc2 76 % of the far gap of f1eq, j1rc 32 % of j1eq's): the far gap is partly a precision-recall trade of the verifier, not only a feature limit.", {"range_recall.json": f"{E18}/range_recall.json"}, []),
    ("E232", "long-joint-fine-tune-full-builder-loss", "Longer joint fine-tune (4800 steps) of builder + route head with the builder loss at full weight, route loss through the builder (arm a) against a detached control (arm c)", "E", "",
     "Failure-half route success a - c: night -0.036, blur -0.036, three cameras off 0.000 (rule +0.03 in two types not met), graph S level: the third negative for the route loss through the builder (E202, E214, E232); the failure-trained route head itself lifts route success +0.15 ... +0.37 over the base head, but its fine-tuned builder loses graph S with three cameras off (-0.029).", {"e232_table.json": f"{E18}/e232_table.json"}, []),
    ("E230", "test-loop-18", "TEST (once): pruned champion f1, j1, control with the route head along the edges (original and failure-trained heads)", "C", "",
     "TEST (833 frames): f1 + pruning S 0.748, Q 0.718, J 0.422, lanes found 0.653, route 0.841 / 0.844 (failure-trained 0.859 / 0.865); j1 + pruning S 0.746, J 0.419; rc1 + pruning 0.4 S 0.747, Q 0.727, J 0.463, lanes found 0.724; control route 0.894 / 0.868 (loop-16 TEST S 0.702, J 0.404, lanes found 0.762). Details: evaluation/TEST_USAGE_LOG.md.", {"test18_table.json": f"{E18}/test18_table.json"}, []),
]

for eid, slug, title, tag, ks, result, files, figs in EXPS:
    existing = glob.glob(f"{EX}/{eid}_*")
    d = existing[0] if existing else f"{EX}/{eid}_{slug}"
    os.makedirs(d, exist_ok=True)
    os.makedirs(f"{d}/results", exist_ok=True)
    os.makedirs(f"{d}/figures", exist_ok=True)
    got = []
    for dest, src in files.items():
        if src and os.path.exists(src):
            shutil.copy(src, f"{d}/results/{dest}")
            got.append(dest)
    for f in figs:
        if os.path.exists(f"{FIG}/{f}"):
            shutil.copy(f"{FIG}/{f}", f"{d}/figures/{f}")
    kline = f"\n## Techniques used (register K ids)\n\n- {ks} (see `../../literature/register.md`)\n" if ks else ""
    open(f"{d}/README.md", "w", encoding="utf-8").write(
        f"# {os.path.basename(d)}\n\n**{title}**\n\n| | |\n|---|---|\n| experiment id | {eid} |\n| theme | {T} |\n| tag | {TAG[tag]} |\n| loop | loop18 - state: [../../experiment_execution/indexes/LOOP18_STATE.md](../../experiment_execution/indexes/LOOP18_STATE.md) |\n\n"
        f"## Result\n\n{result}\n{kline}\n## Data\n\n"
        + ("".join(f"- `results/{g}`\n" for g in got) if got else "- no separate data files (numbers in the loop-18 report and the state file)\n")
        + ("".join(f"- `figures/{f}`\n" for f in figs))
        + "- code: `../../training/models/p18_*.py`, `../../training/pipeline/p16_infer.py`, `../../evaluation/` (p18_*), launchers `../../temp/launchers/loop18/`\n"
        + "- raw outputs (arrays, checkpoints): `../../runs/`\n")
    print("wrote", d)

# ---- index rows in experiments/README.md (after the E212 row; the E209 row gets its result; idempotent)
idx = f"{EX}/README.md"
lines = open(idx, encoding="utf-8").read().split("\n")
new_rows = []
for eid, slug, title, tag, ks, result, files, figs in EXPS:
    d = glob.glob(f"{EX}/{eid}_*")[0]
    nm = os.path.basename(d)
    n_data = len(glob.glob(f"{d}/results/*"))
    short = result.split(". ")[0][:230]
    row = f"| [{nm}]({nm}/README.md) | {'T10' if eid == 'E209' else 'T11'} | {tag} | {short} | {n_data} data, 0 code |"
    if eid == "E209":
        lines = [row if l.startswith("| [E209_") else l for l in lines]
    else:
        new_rows.append((nm, row))
lines = [l for l in lines if not any(l.startswith(f"| [{nm}]") for nm, _ in new_rows)]
pos = max(i for i, l in enumerate(lines) if l.startswith("| [E212_"))
lines[pos + 1:pos + 1] = [r for _, r in new_rows]
text = "\n".join(lines)
text = re.sub(r"^\d+ experiments \(catalogue ids E15 - E\d+\)", f"{sum(1 for l in lines if l.startswith('| [E') or l.startswith('| [VLA'))} experiments (catalogue ids E15 - E230)", text, count=1, flags=re.M)
open(idx, "w", encoding="utf-8").write(text)
print("index updated:", len(new_rows), "rows added")
