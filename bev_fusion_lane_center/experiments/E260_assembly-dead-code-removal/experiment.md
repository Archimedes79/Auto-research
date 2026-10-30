# E260 Remove the dead options of assemble()

## 0. What the tools run (the fields of `experiment.json`)
| field | value |
|---|---|
| id / card / title | E260 / H21.S3 / dead-code removal in `assemble()` (simpler system) |
| kind | code change, identity rule |
| rule | graph hashes identical on six names, golden check, spec tests, review |
| budget | CPU about 20 minutes |

## 1. Card
| | |
|---|---|
| lens | simplification |
| group | A3 sparse lane graph (assembly) |
| sources | `temp/loop21/asm_knob_audit.py` (2026-10-30): assemble() has 31 keyword arguments, 38 registry entries and TM.ASM use 8 of them; the options seg_r, seg_frac, glue_r, glue_lo, glue_hi, contract, fork_tau, fork_peak, fork_r, fork_vtx, fork_cov are named by no other code file (only by old plan tables and the register); each is default-off |
| status | proposed |

- **Statement**: the eleven options and the helper code behind them (`Union.covered_fraction`, `glue_ends`, `contract_junctions`, the fork-conditioned admission and the fork-preserving union with `nearp`) are dead: removing them changes no assembled graph.
- **Mechanism**: they are experiment hooks of loops 16 - 18 that never became part of a registered system; every path is skipped at the default.
- **Prediction** {value 0, class "removal of default-off code"}: all six graph hashes identical.
- **Decision rule**: see json. Kept on purpose: `tau_low` (the hysteresis driver p16_hyst), `tau_decay` (an archived E217 script), `fork_prune_m`, `pair`, `aux`, `clean` and the anchoring constants `junc_edge / free / ang` (real parameters of used stages).
- **Test design**: layer 1 the hash baseline was taken before the change (`graph_hash.py write`: fast_recall, champion, recall, fast_recall_na, fast_recall_dp, champion_dp); after the change `graph_hash.py check`; `evaluate.py --golden check --cached-only`; `test_dp_simplify.py`; `test_p16_spec.py`; an `ar-reviewer` pass (mode code) with the instruction to search all callers.
- **Cost / kill**: CPU 20 minutes; revert with git if any hash differs.
- **Referee**: not run (identity rule, no hypothesis about a number).

## 3. Summary (2026-10-31; no run id: a code change with an identity rule; baseline hashes `temp/loop21/graph_hash.json`)
- **Decision: KEEP (rule met).** The graph hashes (sha256 over the points, node ids and edges of all 873 VAL frames) of fast_recall, champion, recall, fast_recall_na, fast_recall_dp and champion_dp are IDENTICAL before and after the change; `evaluate.py --golden check --cached-only` reproduces the 6 golden entries; `test_dp_simplify` 9 / 9 and `test_p16_spec` 18 / 18 pass; every `asm` key of `models/registry.json` is still accepted by `assemble()`. Prediction (all hashes identical): hit.
- **What went:** the options seg_r, seg_frac, glue_r, glue_lo, glue_hi, contract, fork_tau, fork_peak, fork_r, fork_vtx, fork_cov (31 -> 24 parameters) and the code behind them: `Union.covered_fraction`, `glue_ends`, `contract_junctions`, the fork-conditioned admission and the fork-preserving union (`nearp`): 149 lines of `training/pipeline/p16_infer.py` removed, 4 changed.
- **Review (`ar-reviewer`, mode code):** no defect. Beyond our hashes it compared old and new code on 40 frames for 46 configurations (fast False / True / 2, two taus, aux on / off, tau_low, tau_decay, fork_prune_m, clean off, dp_eps): 0 differences. 58 call sites parsed with ast: no unknown keyword. Remaining notes: a removed option now raises TypeError (it used to be ignored at zero; no caller in the repository passes one), and the technique register K183 pointed at fork_tau / fork_vtx / fork_cov (annotated as removed).
- **Not covered:** non-default combinations of junc_edge / junc_free / junc_ang / bubble_r / spur_len / pair and the fast=False heat-map branches; the K183 experiments of loop 16 cannot be re-run with current code (the commit before this one holds it).
- **Speed:** none claimed (the removed code was off); the point is fewer parameters and 150 fewer lines to read and test.
