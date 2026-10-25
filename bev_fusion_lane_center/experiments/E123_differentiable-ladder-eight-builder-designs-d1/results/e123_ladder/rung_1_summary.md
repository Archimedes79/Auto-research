# Rung 1 summary (generated)

| run | design | F1@1 | F1@0.5 | J | Sp | FL | v/100 m | S | tau | F1@Sp17 | FL@R85 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E123.D7.r1.b1 | D7 | 0.862 | 0.679 | 0.149 | 1.000 | 0.043 | 11.8 | 0.718 | — | 0.7 | 0.865 | 0.049 |
| E123.D7.r1.b2 | D7 | 0.756 | 0.585 | 0.114 | 0.933 | 0.032 | 15.3 | 0.652 | — | 0.1 | 0.756 | — |
| E123.D7.r1.b3 | D7 | 0.861 | 0.678 | 0.147 | 0.942 | 0.039 | 15.2 | 0.709 | — | 0.6 | 0.862 | — |
| E123.D7.r1.b4 | D7 | 0.725 | 0.543 | 0.108 | 1.000 | 0.040 | 10.2 | 0.642 | — | 0.1 | 0.725 | — |
| E123.D6.r1.b1 | D6 | 0.814 | 0.657 | 0.106 | 0.723 | 0.031 | 19.8 | 0.650 | — | 0.1 | — | — |
| E123.D6.r1.b2 | D6 | 0.839 | 0.671 | 0.142 | 1.000 | 0.039 | 14.3 | 0.708 | — | 0.1 | 0.839 | — |
| E123.D6.r1.b3 | D6 | 0.834 | 0.670 | 0.132 | 1.000 | 0.034 | 14.3 | 0.705 | — | 0.4 | 0.835 | — |
| E123.D6.r1.b4 | D6 | 0.811 | 0.656 | 0.111 | 0.715 | 0.031 | 20.0 | 0.649 | — | 0.1 | — | — |
| E123.D8.r1.b1 | D8 | 0.839 | 0.631 | 0.109 | 1.000 | 0.065 | 9.3 | 0.690 | — | 0.9 | 0.842 | 0.070 |
| E123.D8.r1.b2 | D8 | 0.835 | 0.648 | 0.120 | 1.000 | 0.053 | 12.9 | 0.696 | — | 0.1 | 0.835 | — |
| E123.D8.r1.b3 | D8 | 0.841 | 0.644 | 0.088 | 1.000 | 0.059 | 12.1 | 0.690 | — | 0.9 | 0.842 | 0.067 |
| E123.D8.r1.b4 | D8 | 0.822 | 0.630 | 0.120 | 1.000 | 0.047 | 10.4 | 0.690 | — | 0.8 | 0.825 | — |
| E123.D5.r1.b1 | D5 | 0.844 | 0.663 | 0.097 | 0.922 | 0.041 | 15.5 | 0.687 | — | 0.9 | 0.852 | — |
| E123.D5.r1.b2 | D5 | 0.853 | 0.688 | 0.106 | 1.000 | 0.029 | 13.7 | 0.710 | — | 0.9 | 0.856 | — |
| E123.D5.r1.b3 | D5 | 0.857 | 0.673 | 0.100 | 1.000 | 0.048 | 13.3 | 0.704 | — | 0.7 | 0.859 | — |
| E123.D5.r1.b4 | D5 | 0.848 | 0.684 | 0.106 | 0.880 | 0.029 | 16.2 | 0.690 | — | 0.7 | 0.849 | — |
| E123.D7.r1.xj | D7 | 0.861 | 0.680 | 0.140 | 1.000 | 0.033 | 11.8 | 0.717 | — | 0.8 | 0.866 | 0.048 |

Cut: removed D8, D5; ranking D7 0.718 > D6 0.708 > D5 0.710 > D8 0.696; noise band 0.010; rules none.

![scores](../../figures/e123_ladder__rung_1_scores.png) ![frontier](../../figures/e123_ladder__rung_1_frontier.png) ![effects](../../figures/e123_ladder__rung_1_effects.png)


## Hybrid challenge H

Hybrid challenge H (D7 recombined with the growth part: L pieces + G growth from open ends, 4 u from the shared stem, beam 192 / seeds 96, 35 min): S 0.704, F1@1 0.856, F1@0.5 0.652, J 0.121 (split 0.121 / merge 0.121), 12.4 vertices per 100 m, false length 0.052. Threshold of the plan: S(H) must exceed the lower survivor (D6 0.708) by more than the noise band 0.010, i.e. reach 0.718; it stays 0.004 below D6 and 0.014 below D7. H is dropped; the survivors stay D7 and D6. Reason: the growth part added cost (35 min against 12 - 24 min) and no junction gain (J 0.121 against 0.149 for D7); the fork problem is not a matter of how the pieces are grown but of the topology step after them.
