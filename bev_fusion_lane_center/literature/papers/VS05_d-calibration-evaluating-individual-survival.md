# VS05 - D-calibration: evaluating individual survival distributions

*Pass: loop 16 (2026-10-23), cluster `verify_sampling` (verification along rays, sampling, survival and hazard read-outs). Reading depth: **A** (A abstract, M method, C code page). Verified: partly. Pipeline stage: E.*

## Papers

Effective ways to build and evaluate individual survival distributions (Haider, Hoehn, Davis, Greiner, JMLR 2020) https://arxiv.org/abs/1811.11347

## Idea

Argues for models that output a whole survival distribution per individual and for "D-calibration" to test whether those probabilities are meaningful; compares concordance, 1-calibration, Brier score and L1-loss variants and says they measure different things.

## Formulation

not read (from memory, unverified: predicted survival probability at each individual's event time should be uniform on (0,1), with a rule for censored cases). - input / output / training: not read. differentiable: not applicable.

## Sensors and data

medical.

## Reported results

none seen.

## Compute

not stated. code: not checked.

## Fit to our project

reuse_idea - S - a probability-integral-transform style check of the endpoint law P(k) on the labelled rays (own suggestion; the discrete ring grid needs randomised ties).

## Needs

censoring handling in the evaluation. compat_conflicts: none.

## Tried in our experiments

none.
