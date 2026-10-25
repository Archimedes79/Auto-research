# PL09 - UncAD (per-vertex Laplace uncertainty into planning and trajectory selection)

*Pass: loop 16 (2026-10-23), cluster `planner_nav` (planner interface, navigation, map matching, end-to-end driving). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S10, L.*

## Papers

UncAD: Towards Safe End-to-end Autonomous Driving via Online Map Uncertainty (Yang et al., ICRA 2025) https://arxiv.org/abs/2504.12826

## Idea

Every predicted map point carries a Laplace scale; the scales are encoded into "uncertainty queries", concatenated with the map queries and used as keys / values for agent and ego attention; at inference candidate trajectories are scored against the Laplace ellipses of boundary points and risky ones are set to zero.

## Formulation

map head output V_m of shape N_v x N_p x 4 (N_v = 100, N_p = 20; mu x, y and scale b x, y); loss = Laplace NLL, sum over points of log(2b) + |p - mu| / b; Q_unc = MLP(b); Q_m' = MLP([Q_m, Q_unc]) is the key / value of the agent and ego cross-attention; planner output 3 commands x N modes with confidences; UCAS: filter by command, score each trajectory point by NLL against the boundary ellipses, set the score to 0 if risk is high (threshold not stated), plus the base collision check and L2 distance to boundaries; the best remaining trajectory is output. Total loss = lambda_1 L_map (NLL) + lambda_2 L_mot + lambda_3 L_plan (weights not given).

## Input

VAD- or SparseDrive-style map instances with scales.

## Output

ego trajectory + selection.

## Training

end-to-end, no extra uncertainty label; 8 x RTX 3090; epochs not given.

## Differentiable?

partial - uncertainty enters as features (differentiable); trajectory selection is a hard filter.

## Sensors and data

nuScenes.

## Reported results

nuScenes L2 / collision / drivable-area conflict rate DACR (a frame counts if any of the 4 ego corners lies outside the drivable area): VAD 0.72 / 0.23 % / 1.39 %; VAD + UncAD 0.62 / 0.17 % / 0.80 %; SparseDrive 0.61 / 0.10 / 1.06; SparseDrive + UncAD 0.60 / 0.07 / 0.88. Abstract: up to 26 % fewer collisions, up to 42 % fewer drivable-area conflicts, +1.9 % parameters.

## Compute

+1.9 % parameters; latency not stated.

## Code

https://github.com/pengxuanyang/UncAD

## Fit to our project

reuse_idea - effort S - our validity / keep probability is the analogue of 1 / b; there is no per-vertex position scale yet.

## Needs

per-vertex uncertainty in the same tensor as the positions; a candidate-trajectory scorer.

## Conflicts

thresholds unspecified; selection is non-differentiable.

## Tried in our experiments

none (E120 refinement and the keep gate exist, but not as planner input).
