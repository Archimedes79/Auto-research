# MH56 - Modified multiple hypothesis tracking for tree extraction (ranked local hypotheses, search-depth tree)

*Pass: loop 16 (2026-10-23), cluster `multihyp` (multiple hypotheses, set prediction, matching-free losses). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S4, S5.*

## Papers

Extracting Tree-structures in CT data by Tracking Multiple Statistically Ranked Hypotheses (Selvan et al.) https://arxiv.org/abs/1806.08981 (read via ar5iv)

## Idea

Classical hypothesis-tree tracking: at regularly spaced steps tubular templates are fitted in a few directions, local hypotheses are ranked, and a tree of depth d of hypotheses is built; decisions are deferred until a branch scores well across several steps.

## Formulation

step length = scaled local radius; search angle 70 degrees with 2 angles for airways; template profile steepness 8, weighted least squares; score = 1 / rank among the N hypotheses at that step (best = 1, removes scale dependence); global hypothesis = d local hypotheses scored by the average rank score; bifurcations: spectral clustering of hypotheses; tree state saved and tracking resumed per new branch; global threshold 0.7 (airways); search depth 6 (airways), 4 (coronary); after a decision, hypotheses that are not children of the best node are discarded. - input / output: CT volume intensities (or probability images) -> centreline tree.

## Training

none (classical); parameter tuning at depth 6 took about 88 hours.

## Differentiable?

no.

## Sensors and data

airway CT (16 test scans), coronary CT.

## Reported results

mean centreline error 1.93 mm vs 3.96 mm original MHT, 3.37 mm region growing on intensity, 2.06 mm region growing on probability images (p = 0.1 vs the last); coronary overlap 91.8 % vs 67.6 % original MHT; about 15 s per airway scan, 3 s per vessel.

## Compute

depth 4 - 6 feasible, trade-off between under-segmentation and run time.

## Code

not checked

## Fit to our project

baseline - effort M - the non-learned counterpart of our tree: local template sampling, rank-based scores, deferred decision by depth; rank scores 1/R are a calibration-free alternative to raw validity.

## Needs

local scores per hypothesis.

## Conflicts

not trainable.

## Tried in our experiments

none
