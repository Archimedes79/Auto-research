# VS02 - DRSA: deep recurrent survival analysis (hazard per slot + survival-status cross-entropy)

*Pass: loop 16 (2026-10-23), cluster `verify_sampling` (verification along rays, sampling, survival and hazard read-outs). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S3, L.*

## Papers

Deep Recurrent Survival Analysis (Ren et al., AAAI 2019) https://arxiv.org/abs/1809.02403

## Idea

A recurrent cell emits a conditional hazard per time slot, so the hazard at slot l depends on earlier slots. Besides the event-time likelihood, a cross-entropy on the survival STATUS at the observed time supervises the whole range before the event; the authors say this gives denser gradients than a pure event-time likelihood (DeepHit touches mainly the true event time).

## Formulation

h_l = Pr(z in V_l | z > t_{l-1}, x) from an LSTM cell fed (x, t_l, previous hidden state); S(t) = prod_{l<=l^i}(1 - h_l); event probability p_l = h_l prod_{j<l}(1 - h_j). L_z = -log p at the observed slot (uncensored samples only). L_c = survival-status cross-entropy: uncensored -log(1 - S(t^i)), censored -log S(t^i). Objective alpha L_z + (1 - alpha) L_c (Eq. 17); alpha value not stated; whether the hazard output is a sigmoid is not stated on the page seen.

## Input

point_set / features per slot. output: scores (hazard per slot).

## Training

likelihood + status CE; matching_free.

## Differentiable?

yes.

## Sensors and data

CLINIC (6,036 samples), MUSIC (3,296,328, last.fm return times), BIDDING (19,495,974, auction win times); nothing spatial.

## Reported results

C-index CLINIC / MUSIC / BIDDING: DRSA 0.774 / 0.892 / 0.911, DeepHit 0.733 / 0.878 / 0.858, DeepSurv 0.753 / 0.862 / 0.840; ANLP 3.337 / 5.132 / 4.774 (DRSA) vs 5.027 / 5.523 / 5.544 (DeepHit) (paper Table 2; Lasso-Cox, not plain Cox, is the Cox baseline).

## Compute

not stated.

## Code

URL from memory, not opened: https://github.com/rk2900/DRSA

## Fit to our project

reuse_idea - S - our per-ring on-lane labels already act as the status loss; an optional recurrence along the ring index (GRU over 4 / 8 / 12 / 18 / 24 m) makes the hazard depend on earlier samples.

## Needs

ordered samples; a recurrent or attention link over the ring index if sequential dependence is wanted.

## Conflicts

none for K = 5 rings (sequential compute is tiny).

## Tried in our experiments

none.
