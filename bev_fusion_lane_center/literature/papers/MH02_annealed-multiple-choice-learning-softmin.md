# MH02 - Annealed Multiple Choice Learning (aMCL): softmin assignment with a cooling temperature

*Pass: loop 16 (2026-10-23), cluster `multihyp` (multiple hypotheses, set prediction, matching-free losses). Reading depth: **M+C** (A abstract, M method, C code page). Verified: checked. Pipeline stage: L, T.*

## Papers

Annealed Multiple Choice Learning: Overcoming limitations of Winner-takes-all with annealing (Perera, Letzelter, Mariotte, Cortes, Chen, Essid, Richard, NeurIPS 2024) https://arxiv.org/abs/2407.15580 (text via https://arxiv.org/html/2407.15580, last 78 k characters not read); code https://github.com/Victorletzelter/annealed_mcl

## Idea

Replace the hard min of WTA by a Boltzmann softmin over hypotheses and lower its temperature T during training. At high T all hypotheses are pulled to the conditional mean, they split into subgroups at critical temperatures (phase transitions) and end as a WTA solution at T -> 0. This avoids bad local minima of greedy WTA (hypotheses stuck far from the data, unused hypotheses) without the permanent centroid bias of relaxed WTA.

## Formulation

q_T(k | x, y) = exp(-l(f_k(x), y) / T) / sum_s exp(-l(f_s(x), y) / T), l = squared Euclidean distance; L_aWTA = sum_k q_T(k | x, y) * l(f_k(x), y) with q DETACHED. T -> 0: one-hot (MCL); T -> inf: uniform, all hypotheses go to E[Y | x]. First critical temperature T_c = 2 * lambda_max(conditional covariance of Y given x). Schedule: T constant within an epoch; exponential T(t) = T0 * rho^t (UCI: T0 = 0.5, rho = 0.95, 1000 epochs), linear (synthetic: T0 = 1.0, 1000 epochs, 49 hypotheses), several T0 tried for speech; no rule for choosing the schedule (stated as open). Scores: same BCE-on-cell-membership scoring loss as rMCL (MH01).

## Input

any regression network with n heads (toy MLP; UCI n = 5; DPRNN on WSJ0-mix). Repo: PyTorch (+ Lightning for separation); `source_separation/losses.py` has PIT, MCL, aMCL, EMD; `models.py` has an extra scoring head; a temperature scheduler component is not described in the README.

## Output

n hypotheses (+ n scores).

## Training

loss above, GT set only; regime `wta` (annealed).

## Differentiable?

partial - q detached; gradient reaches all hypotheses weighted by q (more than WTA); final solution is WTA.

## Sensors and data

synthetic (three Gaussians), UCI regression, WSJ0-mix speech (2 - 5 speakers; WSJ0 needs LDC access).

## Reported results

UCI distortion (aMCL / MCL / relaxed WTA eps = 0.1): Protein 0.77 / 0.80 / 1.67; Energy 0.28 / 1.25 / 0.30; Yacht 1.15 / 3.09 / 1.34; Boston and Concrete: MCL lower (2.14 vs 2.69; 5.13 vs 5.71). Speech SI-SDR, 2 speakers: PIT 16.88, MCL 16.30 +- 0.59, relaxed WTA 16.70, aMCL 16.85 +- 0.13; 3 speakers 10.01 / 10.06 / 9.43 / 10.00. MCL collapsed for some seeds, aMCL had lower seed variance.

## Compute

assignment O(m n) instead of O(m^3) for PIT; slow cooling lengthens training; no wall-clock.

## Code

https://github.com/Victorletzelter/annealed_mcl

## Fit to our project

reimplement - effort S - the early high-temperature phase keeps both children of a tip alive; the schedule needs tuning.

## Needs

smooth loss l (squared distance); detached assignment weights; an epoch-wise schedule hook.

## Conflicts

at high T both children converge to the mean of the GT set at that tip (for a fork: the midpoint between two branches) - harmless early, wrong if cooling stops early; no remedy for the single-GT starvation after cooling (the second child splits off only where data show two modes).

## Tried in our experiments

none
