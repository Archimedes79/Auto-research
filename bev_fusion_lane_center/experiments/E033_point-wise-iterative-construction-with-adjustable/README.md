# E33: Point-wise iterative construction with adjustable point spacing (line tracer step size)

Status: completed

**Question:** The line tracer is a point-by-point iterative construction: how coarse may the construction step (point spacing) be before accuracy suffers, and what does a coarser spacing buy?

## Setup
- Same dense map (E27 GaussianCaR head, VAL) and tracer parameters as E31; only the construction step is varied: 0.5, 1, 2, 4 m per point (tracer 'step' parameter; the on-line check, fan search and re-centring all work at that step).
- Metrics: dense point F1 at 0.5 / 1 m, precision / recall, 1:1 F1, and the number of support points per sample of the resulting polylines (before resampling).

![architecture](architecture.png)

## Numbers

**Tracer step sweep (VAL)**

| step (point spacing) | F1@0.5m | F1@1m | P@1m | R@1m | 1:1 F1@1m | false length | ms / sample (1 core) | support points / sample |
|---|---|---|---|---|---|---|---|---|
| 0.5 m | 0.698 | 0.876 | 0.899 | 0.854 | 0.796 | 0.040 | 15 | 934 |
| 1.0 m | 0.703 | 0.879 | 0.901 | 0.858 | 0.802 | 0.040 | 14 | 942 |
| 2.0 m | 0.705 | 0.879 | 0.904 | 0.855 | 0.799 | 0.039 | 14 | 945 |
| 4.0 m | 0.694 | 0.868 | 0.887 | 0.849 | 0.768 | 0.046 | 15 | 1048 |

## Result images

![Accuracy vs step size.](e33_spacing.png)
*Accuracy vs step size.*

## Findings
- Accuracy is flat between 0.5 and 2 m spacing (F1@1 m 0.876 / 0.879 / 0.879); at 4 m the 1:1 F1 drops from 0.80 to 0.77 and recall to 0.85 because curved connectors and the first/last step at lane ends are cut.
- The number of support points per sample hardly changes (about 940 for 0.5-2 m) because the output is re-sampled at 1 m for the metrics; the point spacing therefore mainly controls how coarsely a lane can be represented - 2 m spacing (about 11 points per 20 m chunk) is the same resolution used for the sparse extractor in E32.
- Runtime per sample is dominated by Python/numba overhead (about 15 ms) and does not depend on the step; the benefit of a larger step would only appear with a learned (neural) per-step predictor, which was not built (a RoadTracer-style learned tracer was left out for time; the rule-based tracer is the iterative construction used throughout).

## Limits
- Only the rule-based tracer was varied; a learned point-wise predictor with a stop / branch head was not implemented.
