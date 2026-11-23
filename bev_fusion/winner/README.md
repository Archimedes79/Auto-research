# winner - the winning system of the BEV Fusion project and its code

One folder per winner (D139 / D140, 2026-11-23: every project keeps its winner with the code that belongs to it). A winner folder is the **source code** of the system as it ran, a README with the commands and the numbers, and a `MANIFEST.json`; **no weights, no data, no logs** (the weights stay in `../runs/`, named in the winner's README).

| folder | what | numbers (VAL 1000 frames of 82 scenes, GT v8, seed 0 / seed 1) |
|---|---|---|
| `t21_ctl/` | the t20 network (dual camera features DualBEV + BEVFusion cached, L4 lane branch, delta-slot radar, camera memory, concat fusion, DualBEV trunk, map / object / velocity heads, motion block, 1 m centre-line head) continued 1500 steps from `runs/t20_base_s0` (the equal-budget control arm of t21); code at tag `bf/t21.ctl_s0` (commit `388a36cdf0`) | board7 0.45489 / 0.45482 (`records/keep3.json`, slot 1); 102.5 ms per frame |

**Read before use (named by the BEV chat on 2026-11-23, from the primary files):**
- **A tie, chosen by value.** t23_C (0.44969, two seeds) and t20_base_s0 (0.44654, one seed) lie inside the band 0.02; t21_ctl is the best by value, its slot was chosen by hand on 2026-11-10; no champion is promoted (`records/champion.json` is stale). The alternative by role is `t20_base_s0` (tag `bf/t20.base_s0`, the accepted reference baseline of D48): the owner's choice.
- **It does not run as is.** The code predates D76 and imports `autogt` (renamed `GtGenerator`, the alias removed by D78; D81: old runs are not run again); the folder is the record of what ran. Only the weights load under the live code (a warm start through `init_from`).
- **GT v8.** The numbers are on GT v8 (signature `0.5_4_6a7aa7864a`); the live GT is v13 / v14 and t21_ctl was not re-scored (the base t20_base_s0 moved by at most 4e-5 on v13, D102).
- **Not SOA.** On the t26 G0 tables (`records/t26_g0/`, 43 common scenes, GT v13) this family is behind GaussianCaR on all six map rows and behind CRN on objects (mAP2 0.489 against 0.598).

Made with `python reporter/winner.py export t21_ctl --out bev_fusion/winner ...` from the code at the tag (`git archive`); `python reporter/winner.py verify bev_fusion/winner/t21_ctl` re-checks the files against the manifest (the sources are named as `git:<tag>:<path>`).
