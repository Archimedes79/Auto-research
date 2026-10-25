# Rung 2 summary (generated)

| run | design | F1@1 | F1@0.5 | J | Sp | FL | v/100 m | S | tau | F1@Sp17 | FL@R85 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E123.D7.r2.b1 | D7 | 0.864 | 0.677 | 0.141 | 1.000 | 0.060 | 10.9 | 0.714 | — | 0.7 | 0.864 | 0.058 |
| E123.D7.r2.b2 | D7 | 0.861 | 0.687 | 0.180 | 0.983 | 0.035 | 14.5 | 0.724 | — | 0.5 | 0.862 | — |
| E123.D7.r2.b3 | D7 | 0.857 | 0.670 | 0.131 | 1.000 | 0.058 | 10.4 | 0.709 | — | 0.8 | 0.857 | 0.063 |
| E123.D7.r2.b4 | D7 | 0.830 | 0.670 | 0.169 | 0.980 | 0.033 | 14.6 | 0.709 | — | 0.5 | 0.831 | — |
| E123.D6.r2.b1 | D6 | 0.802 | 0.637 | 0.086 | 0.986 | 0.034 | 14.5 | 0.678 | — | 0.1 | 0.802 | — |
| E123.D6.r2.b2 | D6 | 0.792 | 0.623 | 0.084 | 0.973 | 0.037 | 14.7 | 0.669 | — | 0.1 | 0.792 | — |
| E123.D6.r2.b3 | D6 | 0.793 | 0.629 | 0.082 | 1.000 | 0.031 | 13.2 | 0.676 | — | 0.1 | 0.793 | — |
| E123.D6.r2.b4 | D6 | 0.795 | 0.629 | 0.090 | 1.000 | 0.033 | 13.2 | 0.677 | — | 0.1 | 0.795 | — |

Cut: removed ; ranking D7 0.724; noise band 0.010; rules none.

![scores](rung_2_scores.png) ![frontier](rung_2_frontier.png) ![effects](rung_2_effects.png)


## Why designs lost and what the factors did

Rung 2 (2 survivors x L4 over HF1 stage B refine, HF2 stage C rescore, HF3 b = 3 children with multiple-choice learning, 4 u each, VAL subset of 250 frames, noise band 0.010). D7 (parents e123_D7_r1_b1): b1 continuation 0.714, b2 stage C + b = 3 / MCL 0.724, b3 stage B + b = 3 / MCL 0.709, b4 stage B + stage C 0.709. D6 (parent e123_D6_r1_b2): b1 0.678, b2 0.669, b3 0.676, b4 0.677. Pooled effects on S over both designs: HF1 stage B -0.004, HF2 stage C +0.001, HF3 b = 3 / MCL -0.000, none supported (SE 0.004). Inside D7 alone: B -0.010, C +0.005, b = 3 +0.005; all inside the noise band. What stage C does change is the junction score: J 0.180 and 0.169 with stage C against 0.141 and 0.131 without (+0.04), at the price of about 3.5 more vertices per 100 m (14.5 against 10.9). That is the only consistent factor effect of the rung and it points at the right weakness; it is a J effect, not an S effect. Cut: D7 stays, D6 is removed. Why D6 lost: all four branches fell to 0.669 - 0.678, below its own rung-1 value 0.708 (the plain continuation b1 fell too, 0.678): the rung-1 value was a favourable point of a noisy curve, not a level. D6's F1@1 is 0.79 - 0.80 against 0.83 - 0.86 for D7, its J 0.08 - 0.09 against 0.13 - 0.18, and its best threshold sits at the floor of the sweep (0.1): the keep gate it carries (SPF2, an effect of -0.015 in rung 1) closes recall and the validity is not calibrated for a threshold. The learned pair head (factor D) was the cost of the design: factor D showed -0.040 in rung 0. Winner of rung 2: D7 b2 (L, teardrop, lattice, geometry, global, warm, revisit off; beam 192, seeds 96; stage C and b = 3 with MCL), S 0.724 = control level (0.724) at 14.5 vertices per 100 m against 21.8, F1@1 0.861 against 0.883, J 0.180 against 0.39.


## Fork diagnosis of the winner

Fork diagnosis of the rung-2 winner D7 b2 on 250 validation frames (649 true forks, 2.6 per frame; p16_junc_diag.py, figure p16_forks.png). Classes of the true forks: geometry incomplete (stem or a branch not covered within 1 m) 41 %, both branches in one predicted piece 35 %, fork predicted within 6 m 22 %, detached or offset branches under 2 %. The true forks are shallow: median fork angle 5.5 deg, branch gap 0.6 m six metres after the fork; 38 % have an angle under 2 deg (gap 0.07 m, two overlapping ground-truth lanes), where the hit rate is 0.15; forks of 10 - 20 deg are hit 0.32, of 20 deg and more 0.23 (geometry complete only 0.24: crossing connectors in intersections). Range: the share of forks with complete geometry falls from 0.69 (0 - 15 m) to 0.41 (30 - 50 m), and 46 % of all true forks lie beyond 30 m. A predicted fork exists within 15 m of about half of all true forks, so the position convention (the fork sits where the map pieces start) is the issue as much as the detection. Assembly grid without retraining (cov_r 0.4 - 1.0 m, vertex reuse 0.6 - 1.5 m): best S 0.730 (cov_r 0.6, vtx_r 1.5, tau 0.5), J 0.197 against 0.180 at the default (1.0 / 1.5): smaller snap radii move the fork vertex up the stem but cannot create the second branch. Consequence for rung 3: the junction head (j1: heat maps of the fork landmarks, anchoring at the peaks) is the test of the learned position convention; a branch that was never predicted needs a second successor hypothesis, not a different assembly radius.

![forks](../../../report_final/figures/p16_forks.png)
