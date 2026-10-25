# MH01 - Resilient Multiple Choice Learning (rMCL): multi-target WTA + learned score head

*Pass: loop 16 (2026-10-23), cluster `multihyp` (multiple hypotheses, set prediction, matching-free losses). Reading depth: **M (rMCL), A (ICML 2024 theory)** (A abstract, M method, C code page). Verified: checked. Pipeline stage: L, S4.*

## Papers

Resilient Multiple Choice Learning: A learned scoring scheme with application to audio scene analysis (Letzelter, Fontaine, Chen, Perez, Essid, Richard, NeurIPS 2023) https://arxiv.org/abs/2311.01052 (full text read via https://ar5iv.labs.arxiv.org/html/2311.01052); theory only (A): Winner-takes-all learners are geometry-aware conditional density estimators (Letzelter et al., ICML 2024) https://arxiv.org/abs/2406.04706

## Idea

K hypotheses are trained with a multi-target winner-takes-all loss (each ground-truth target pulls only its nearest hypothesis, i.e. the Voronoi cell it falls in). A separate score per hypothesis is trained with binary cross-entropy to say "this hypothesis wins at least one target of this input". At inference the normalised scores are the probability mass of each cell, so the output is a small weighted set of points; hypotheses sitting in rare regions get low scores instead of looking confident.

## Formulation

hypotheses f^k(x), scores gamma^k(x) in [0,1] from separate last layers on a shared representation. Cell Y^k(x) = {y : l(f^k(x), y) < l(f^r(x), y) for all r != k}. L_WTA = sum over targets y in Y_s of sum_k 1[y in Y^k(x_s)] * l(f^k(x_s), y) (targets assigned independently; two targets may share one hypothesis). K+ = hypotheses winning >= 1 target, K- = rest. L_scoring = -( sum_{k in K+} log gamma^k + sum_{k in K-} log(1 - gamma^k) ). Total = L_WTA + beta * L_scoring, beta = 1. rMCL* draws ONE loser uniformly from K- per step (memory saving). Inference: gamma^k <- gamma^k / sum_r gamma^r; p(y|x) = sum_k gamma^k delta(y - f^k(x)) (or uniform inside the cell). The relaxed weights (1 - eps) / eps/(K-1) are NOT part of rMCL (eps = 0.5 only for the eps-WTA baseline and the eps-rMCL combination).

## Input

shared trunk + K regression heads + K score heads (toy: 3-layer MLP, 256 hidden, 20 hypotheses; audio: SELDnet-style CRNN with 3 / 5 / 10 / 20 heads).

## Output

K points + K scores (sum 1 after normalisation).

## Training

loss above; labels = the GT set only (no matching, no order); regime `wta`. Toy: Adam, 100 k samples, batch 1024, 20 epochs. Audio: AdamW, batch 128, lr 0.05 with the Vaswani schedule, epochs not stated.

## Differentiable?

partial - the argmin / cell assignment is a constant (no gradient through the choice); regression gradient reaches the winner only; whether score gradients are detached from the trunk is not stated.

## Sensors and data

synthetic toy data; sound-source localisation (ANSYN anechoic, RESYN reverberant); no vision.

## Reported results

ANSYN, 5 hypotheses, EMD columns D1 / D2 / D3 (as in the paper table): rMCL 7.04 / 13.87 / 20.76; WTA 48.22 / 44.41 / 41.83; top-n WTA 5.14 / 18.09 / 25.12; eps-WTA 5.51 / 19.20 / 28.39. Oracle error: WTA 3.56 / 6.53 / 10.44, rMCL 3.85 / 7.16 / 11.29. Number of hypotheses (rMCL, EMD D3): 3 -> 20.96, 5 -> 20.76, 10 -> 21.28, 20 -> 22.55. Collapse: described as a risk (one or few heads win almost always); not observed in their audio runs (winner histogram in the appendix).

## Compute

K heads on one trunk; rMCL* avoids the per-loser cost. No latency numbers.

## Code

not found (an author repository exists, from memory; URL not seen)

## Fit to our project

reimplement - effort S - the score means "wins a target" (cell mass), not "lies on a lane", so it cannot replace our validity.

## Needs

a distance between a hypothesis and each GT element that is comparable across hypotheses (child endpoint to GT continuation point or polyline); a GT set of size 1 or 2 per tip; hypotheses in one metric output space (ego metres).

## Conflicts

(1) with one GT continuation per tip (the common case) b - 1 children get no regression gradient and a score target 0 even when they lie on the lane (cell mass versus validity: a separate on-lane validity head is needed); (2) normalising scores to sum 1 removes independent per-node validity (fork = two true children); (3) no remedy inside the method for starved heads (see MH02 - MH05).

## Tried in our experiments

none (E31 / E32 used Hungarian matching for a sparse decoder, dropped)
