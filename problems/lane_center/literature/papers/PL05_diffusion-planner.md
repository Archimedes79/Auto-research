# PL05 - Diffusion Planner (lane tensor 70 x 20 x 12, route lanes through adaLN)

*Pass: loop 16 (2026-10-23), cluster `planner_nav` (planner interface, navigation, map matching, end-to-end driving). Reading depth: **M+C** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S10, S5.*

## Papers

Diffusion-based Planning for Autonomous Driving with Flexible Guidance (Zheng et al., ICLR 2025) https://arxiv.org/abs/2501.15564

## Idea

A DiT-style transformer denoises the joint future of the ego and 10 neighbours conditioned on scene tokens; route lanes condition every block via adaptive layer norm; inference-time classifier guidance with energy functions (collision, speed range, comfort, drivable area).

## Formulation

lanes (code map_process.py): 70 x 20 points x 12 channels [centre xy (2), vector to next centre point (2), left-boundary offset (2), right-boundary offset (2), traffic-light one-hot (4)]; speed limit and has-speed-limit stored per lane (70 x 1). Lanes = LANE + LANE_CONNECTOR within a radius, ranked by minimum distance of any centre point to the ego, nearest kept; successors NOT traversed; linear resampling to the fixed point count; empty slots zero-filled with an `avails` mask. Route lanes: 25 (Table 5) with the shared max_points (20) built by the same routine (paper: coordinates only), from route roadblock ids pruned by connectivity. Ego-centric, then z-score x (mu 10, sigma 20), y scaled by sigma 20. Neighbours 32 x 21 steps x 11 features (predicts 10). MLP-Mixer encoders; hidden 192, 3 encoder + 3 decoder blocks, 6 heads. Route embedding is added to the timestep embedding and applied via adaLN to all tokens. VP linear schedule (beta 0.1 ... 20), x0-prediction MSE, DPM-Solver++ 10 steps, temperature 0.5 (0.1 with refinement). Drivable-area energy = signed-distance cost map of the lanes; route not used in guidance. Encoder code (encoder.py): validity is read from all-zero rows, not from `avails` (a lane whose points all have zero geometry is masked through key_padding_mask; padding points inside a valid lane still enter the mean pooling); lane token = MLP-Mixer over the 20 points (token dim 64, channels 128) mean-pooled to 128-d, plus speed-limit Linear(1 -> 128) or a learned unknown embedding, plus traffic-light Linear(4 -> 128) taken from the first point of the lane; positional embedding Linear(7 -> hidden) of (x, y, cos, sin) + 3-way token-type one-hot; tokens concatenated [neighbours, static objects, lanes] and processed by plain self-attention blocks (no cross-attention); encoder.py has no separate route input (route handling sits in the decoder; code not read).

## Input

ego state, neighbours, static objects, lanes, route lanes.

## Output

joint ego + neighbour trajectories, 8 s at 10 Hz.

## Training

x0-prediction MSE; about 11 M sampled nuPlan scenarios, 500 epochs, batch 2048, lr 5e-4; ego-state perturbation + quintic interpolation to the GT future; regime: denoising.

## Differentiable?

yes (network); guidance differentiates energies w.r.t. the sample, not the map.

## Sensors and data

nuPlan GT map; delivery-vehicle data (about 200 h) as a second source.

## Reported results

closed-loop non-reactive / reactive: Val14 89.87 / 82.80, Test14-hard 75.99 / 69.22, Test14 89.19 / 82.93; with refinement Val14 94.26 / 92.90, Test14-hard 78.87 / 82.00; delivery data 92.08.

## Compute

about 20 Hz on one A6000 (appendix); training 8 x A100 80 GB; parameters not stated.

## Code

https://github.com/ZhengYinan-AIR/Diffusion-Planner

## Fit to our project

adapt_code - effort M - the lane tensor contract is useful, but boundary offsets and traffic-light channels do not exist in our graph.

## Needs

fixed-size lane tensor in which invalid lanes are all-zero rows (the encoder derives its mask from zero geometry), ordered route lanes, ego-frame coordinates.

## Conflicts

no edges used; boundaries and traffic lights must be zero / unknown; 100 m vs 50 m range; trained only on the nuPlan distribution.

## Tried in our experiments

none.
