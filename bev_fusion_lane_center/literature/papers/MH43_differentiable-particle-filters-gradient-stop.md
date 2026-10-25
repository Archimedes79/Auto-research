# MH43 - Differentiable particle filters: gradient stop at resampling (Jonschkowski) and soft resampling (Karkus)

*Pass: loop 16 (2026-10-23), cluster `multihyp` (multiple hypotheses, set prediction, matching-free losses). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S4, S8.*

## Papers

Differentiable Particle Filters: End-to-End Learning with Algorithmic Priors (Jonschkowski, Rastogi, Brock, RSS 2018) https://arxiv.org/abs/1805.11122; Particle Filter Networks with Application to Visual Localization (Karkus, Hsu, Lee, CoRL 2018) https://arxiv.org/abs/1805.08975 (both read via ar5iv)

## Idea

Resampling is the beam cut of a particle filter. Jonschkowski: resampling by stochastic universal sampling is not differentiable and simply stops the gradient after one loop iteration, so effects of earlier steps on the current belief are neglected. Karkus: soft resampling draws ancestors from a mixture of the weight distribution and a uniform one and corrects with importance weights, keeping a gradient path.

## Formulation

Jonschkowski loss = -log of the Gaussian-mixture density of the true state under the weighted particles; components pre-trained, then end to end on overlapping subsequences (length 20 global localisation, 50 KITTI); 100 particles train, 1000 test. Karkus: q(k) = alpha * w_k + (1 - alpha)/K, new weight w' = w / (alpha w + (1 - alpha)/K) (alpha value not extracted); K = 30 particles in training with no resampling there; squared-error loss on the weighted mean pose; BPTT limited to 4 steps (1 / 2 / 4 steps: 73 / 79 / 79 % success). - input / output: weighted particle set.

## Training

end to end through the filter.

## Differentiable?

partial - ancestor choice has no gradient (Jonschkowski); soft resampling keeps the weight dependency but stays biased.

## Sensors and data

simulated mazes, KITTI visual odometry (Jonschkowski); House3D (Karkus).

## Reported results

KITTI translational error (m/m), test 100: BKF 0.2062, DPF independent 0.1901, end to end 0.1467 +- 0.0149; House3D tracking RMSE PF-net 40.5 cm vs histogram filter 92.0, LSTM 66.9, odometry 109.4. Resampling helped under high uncertainty (39 -> 42 %) but hurt tracking in training (79 -> 75 %). Neither paper discusses particle collapse or bias explicitly.

## Compute

about 0.6 ms per particle per step (RGB PF-net).

## Code

not checked

## Fit to our project

reuse_idea - effort S - cheap but biased; at least shows that training with and without the cut differ by a few percent in a long-horizon task.

## Needs

weights over a particle set.

## Conflicts

bias; in the Karkus experiments resampling during training hurt tracking success (79 -> 75 %) while helping under high uncertainty (39 -> 42 %).

## Tried in our experiments

none
