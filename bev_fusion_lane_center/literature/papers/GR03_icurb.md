# GR03 - iCurb (DAgger with restricted and free exploration, dynamic expert labels)

*Pass: loop 16 (2026-10-23), cluster `growth` (iterative growth of road / lane graphs (agents, walkers, trackers)). Reading depth: **M+C** (A abstract, M method, C code page). Verified: checked (paper HTML, README, config, environment source).. Pipeline stage: T, S4, L.*

## Papers

iCurb: Imitation Learning-based Detection of Road Curbs using Aerial Images for Autonomous Driving (Xu, Sun, Liu, RA-L 2021) https://arxiv.org/abs/2103.17118

## Idea

A curb graph is grown vertex by vertex; each step predicts a displacement inside a small crop and a stop flag. Its contribution is the training: DAgger with a restricted run (follow the expert whenever the prediction strays more than 15 px) followed by free runs (always follow the prediction), with the expert label computed *after* the prediction as the nearest ground-truth point that lies ahead of the learner. This is a matching-free label that works for any off-track state.

## Formulation

FPN (frozen, pre-trained) feature map F (8 channels) + binary history map g_t; the agent crops d x d around v_t, concatenates v_t and v_{t-1}; coordinate head -> displacement (scaled to [-1, 1]); stop head -> binary. Seeds Q: end vertices of the skeletonised curb mask + local maxima of an initial-vertex heat map (training: GT starts + Gaussian noise, scale not stated). Alg. 1: per image one restricted exploration, then N = 5 free explorations (paper); after each, train on the aggregated dataset D. Code (`config.yml`, `main_env.py`): crop 63 x 63 px (about 9.6 m at 15.2 cm / px), the expert candidate set = GT points inside the crop, with index >= the progress index ii and farther than 15 px from v_now, nearest one by KD-tree; restricted switch thr = 15 px; stop label = 1 when no candidate is left (end of instance) or no progress; circular DAgger buffer 2,048, batch 512, 3 epochs per round, lr 1e-4, README recommends 1 restricted + 1 free round (more free rounds cost time without clear gain), max 120 steps per instance. Loss: L1 between predicted and expert displacement (Eq. 2); stop loss and weights not stated; stop labels are rare (acknowledged).

## Input

aerial RGB + infrared (4 ch), 1000 x 1000 px patches; `image_feat` crop + history raster.

## Output

curb polyline graph; per step one displacement and a stop flag (no fork handling).

## Training

`dagger` (restricted + free, aggregated D) with dynamic labels; no matching; BC with static labels shown to fail on recovery.

## Differentiable?

no - argmax over the crop, stop thresholding, DAgger loop (but each step is a plain regression).

## Sensors and data

NYC planimetric aerial, 29,000 patches (18,000 segmentation pre-training, 10,000 agent training, 1,000 test).

## Reported results

pixel F1 at 1 / 2 / 5 / 10 px: iCurb 0.175 / 0.556 / 0.883 / 0.930, connectivity CC 0.866; RoadTracer 0.101 / 0.334 / 0.640 / 0.736 (CC 0.767); DAGMapper 0.082 / 0.283 / 0.648 / 0.821 (CC 0.832) (Table II, authors' re-implementations). Ablation F1@5 px (Table I; ticks inferred from row order and text, the authors call free exploration critical): no history 0.818, no restricted exploration 0.865, no free exploration 0.750 (CC 0.739), no aggregation (SEARN-like) 0.865 (CC 0.877, higher than the full model's 0.866, not discussed). Does not win at 1 - 2 px.

## Compute

0.89 s / image inference; 22.38 s / image training (one GTX 1080 Ti).

## Code

https://github.com/TonyXuQAQ/iCurb (data and the maintained code in https://github.com/TonyXuQAQ/Topo-boundary)

## Fit to our project

reuse_idea - effort S / M. The expert rule and the restricted / free switch can be applied to each tip of the tree during a free unroll; the removal of free exploration is the largest drop seen in this cluster.

## Needs

GT polyline with a progress index per instance (our piece graph in travel direction provides it); an expert that works on any state.

## Conflicts

single-branch agent (no fork handling, forks are separate instances); label rule assumes one GT chain per instance, forks need one progress index per branch.

## Tried in our experiments

none (E94 is teacher-forced; no DAgger / free-rollout variant in the verdict table).
