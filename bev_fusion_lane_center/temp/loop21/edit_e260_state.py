import glob
def sub(p, pairs):
    t = open(p, encoding="utf-8").read()
    for a, b in pairs:
        assert a in t, (p, a[:60])
        t = t.replace(a, b)
    open(p, "w", encoding="utf-8").write(t)
p = glob.glob("experiments/E260*/experiment.md")[0]
t = open(p, encoding="utf-8").read()
i = t.index("## 3. Summary")
new = """## 3. Summary (2026-10-31; no run id: a code change with an identity rule; baseline hashes `temp/loop21/graph_hash.json`)
- **Decision: KEEP (rule met).** The graph hashes (sha256 over the points, node ids and edges of all 873 VAL frames) of fast_recall, champion, recall, fast_recall_na, fast_recall_dp and champion_dp are IDENTICAL before and after the change; `evaluate.py --golden check --cached-only` reproduces the 6 golden entries; `test_dp_simplify` 9 / 9 and `test_p16_spec` 18 / 18 pass; every `asm` key of `models/registry.json` is still accepted by `assemble()`. Prediction (all hashes identical): hit.
- **What went:** the options seg_r, seg_frac, glue_r, glue_lo, glue_hi, contract, fork_tau, fork_peak, fork_r, fork_vtx, fork_cov (31 -> 24 parameters) and the code behind them: `Union.covered_fraction`, `glue_ends`, `contract_junctions`, the fork-conditioned admission and the fork-preserving union (`nearp`): 149 lines of `training/pipeline/p16_infer.py` removed, 4 changed.
- **Review (`ar-reviewer`, mode code):** no defect. Beyond our hashes it compared old and new code on 40 frames for 46 configurations (fast False / True / 2, two taus, aux on / off, tau_low, tau_decay, fork_prune_m, clean off, dp_eps): 0 differences. 58 call sites parsed with ast: no unknown keyword. Remaining notes: a removed option now raises TypeError (it used to be ignored at zero; no caller in the repository passes one), and the technique register K183 pointed at fork_tau / fork_vtx / fork_cov (annotated as removed).
- **Not covered:** non-default combinations of junc_edge / junc_free / junc_ang / bubble_r / spur_len / pair and the fast=False heat-map branches; the K183 experiments of loop 16 cannot be re-run with current code (the commit before this one holds it).
- **Speed:** none claimed (the removed code was off); the point is fewer parameters and 150 fewer lines to read and test.
"""
open(p, "w", encoding="utf-8").write(t[:i] + new)
sub("experiment_execution/PLAN.md", [("| I3 | results video of the best system (Mirko's request)", "| I4 | E259 clean-up ablation, CPU time | the spur / bubble clean-up and the stage costs | drop only if guards hold on seven models | CPU | done (E259): clean-up stays (vert100 +0.4 .. +0.7 without it); timing unusable at load 4 - 15, CPU time does not help; anchoring off and relocation off show about 25 % of the assembly each, the dp stage costs ~1 ms loaded |\n| I5 | E260 dead options of assemble() | 11 options and ~150 lines are unused | graph hashes identical | CPU | done (E260): identical on 6 names, golden, tests, review; 31 -> 24 parameters |\n| I3 | results video of the best system (Mirko's request)")])
