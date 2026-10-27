# E256 Deployment candidates of D4 by city

## 0. What the tools run (the fields of `experiment.json`)
| field | value |
|---|---|
| id / card / title | E256 / H20.G2 / candidates of D4 by city (follow-up of E252 and of the recommendation to report by city) |
| kind | variant (analysis of existing checkpoints, no training) |
| metric, direction, band | Q, higher, 0.010 (also F1@0.5 / 1.0, lanes found, vert100) per city |
| prediction | the order of the five candidates by Q is the same in Boston and in Singapore One-North up to the band (prediction value 0 = no city-dependent flip) |
| rule | see section 1 |
| budget | CPU about 10 minutes |

## 1. Card
| | |
|---|---|
| lens | system review (evaluation) |
| group | A9 evaluation |
| sources | E252 (Boston scores higher than Singapore for champion, j1eq, j1rc; city gaps +0.049 ... +0.070), D4 (which fast variant to deploy), E240 / E236 (seed floor, matched vertex budget) |
| status | proposed |

- **Statement**: D4 is decided on pooled VAL numbers; with two cities in VAL and a city gap of +0.05, the candidates keep their order in each city.
- **Mechanism**: the candidates differ in recall and sparsity, not in a city-specific property; the city gap is common to all models (E252), so the order should not change.
- **Prediction** {value: no flip of any pair by more than the band 0.010, tolerance: one pair near the band, class "model ranking across two data halves"}.
- **Decision rule**: per city, rank the five candidates by Q (each at its registered operating point, vertices per 100 m printed); a pair that is ordered one way in Boston and the other way in Singapore by more than 0.010 in both cities is a flip and is written into D4 as city-dependent evidence; otherwise the ranking is confirmed. Vertex budgets differ per model (14.8 - 15.5): the table prints them, no matching is done.
- **Test design**: one run of `temp/loop20/e252_gt_offset.py` with the five names, 1 thread; layer 0 identity check inside the script.
- **Cost / kill**: CPU 10 minutes; none.
- **Inputs**: ready.
- **Referee**: not run (the question is a follow-up of E252 whose card was refereed; the analysis cannot change any model); the numbers reviewer reads the summary.

## 3. Summary (written after the decision)
- **Result: no pair flips, the order is not established** (the rule can hardly fire: the whole Singapore spread is 0.008, below the 0.010 flip threshold, so "prediction hit" is weak evidence) (rule: a flip needs opposite order by more than 0.010 in both cities). The prediction (no flip beyond the band, one pair near it) HIT. Source: `by_city.txt` (`e252_gt_offset.py`, five names, VAL 873, run id E256.a0; the city rows are Q at each model's registered operating point).

| candidate | Q Boston (10 scenes) | Q Singapore (12 scenes) | gap | lanes found B / S | vert100 B / S |
|---|---|---|---|---|---|
| fast_recall | 0.7757 | 0.7061 | +0.070 | 0.774 / 0.702 | 15.00 / 14.96 |
| recall | 0.7700 | 0.7137 | +0.056 | 0.803 / 0.736 | 15.62 / 15.45 |
| control (21.7 vertices) | 0.7661 | 0.7119 | +0.054 | 0.830 / 0.773 | 21.66 / 21.91 |
| champion | 0.7625 | 0.7101 | +0.052 | 0.734 / 0.660 | 14.87 / 14.79 |
| fast | 0.7515 | 0.7064 | +0.045 | 0.709 / 0.636 | 15.03 / 14.95 |

- **What the city split adds:** the pooled advantage of fast_recall over fast (+0.0111 pooled, 0.7392 vs 0.7281) is entirely a Boston effect (+0.024 in Boston, -0.0003 in Singapore), and fast_recall is not better than fast, champion or recall in Singapore (all within 0.008). The lanes found gain (fast_recall - fast, +0.065 in Boston, +0.066 in Singapore) is the same in both cities: that is the robust part of the recall variants. The Q ranking of the candidates is thus carried by 10 Boston scenes: the scene intervals of the five Q values overlap everywhere (Boston half widths about 0.04), so the order inside the 0.012 pooled spread (0.7281 - 0.7404) is not established. D4 may use "lanes found +0.06 at equal Q" as the evidence, not the pooled Q order.
- **Signed lateral offsets** (same script as E252): all five candidates have a median offset of the correct predictions below 0.04 m in each city; no candidate shows a city-specific bias.
- **Limits:** two cities, 22 scenes; vertex budgets differ per model (the control sits at 21.7) and are not matched here; this is a follow-up of E252, not a new model comparison.
