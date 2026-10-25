# Experiment card - E<id> <short title>

*Written BEFORE the run. Copy to the experiment folder (`experiments/E<id>_<slug>/card.md`) when the run is filed.*

| | |
|---|---|
| ledger id(s) | E<id>.a, E<id>.b |
| loop / date | loop NN, 2026-MM-DD |
| basis (earlier experiments, K ids) | E.., K.. |
| change (exactly one) | |
| fidelity / budget | e.g. 2 u, 4,800 iterations, GPU hours |

## Hypothesis
One sentence that can be false.

## Gate and decision rule
- measured with: `evaluation/sparse_lanes` (card S, F1@1, J, vert100) / `evaluation/lane_grid` / route success / planner L2, split VAL, frames N
- seeds: 2 (s0, s1); noise band 0.010 in S
- keep if ...; cut if ...; tie if the difference is inside the noise band

## Architecture
Drawing (`tools/viz` or `scripts/fig_p<loop>_*.py`) and the changed block in two sentences.

## Result (fill after the run)
Numbers table (VAL, both seeds), ms per frame (idle GPU, batch 1), images.

## Conclusion
What it says about the hypothesis, why, what it changes in the next step. Negative results stay.
