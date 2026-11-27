# MH12 - MTR (64 static intention queries from k-means, nearest-to-GT positive, NMS to 6)

*Pass: loop 16 (2026-10-23), cluster `multihyp` (multiple hypotheses, set prediction, matching-free losses). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S4, L.*

## Papers

Motion Transformer with Global Intention Localization and Local Movement Refinement (Shi, Jiang, Dai, Schiele, NeurIPS 2022; authors from memory) https://arxiv.org/abs/2209.13508 (arXiv HTML)

## Idea

One query per motion mode - K = 64 intention points from k-means on GT endpoints (per category) - plus one dynamic searching query per mode that moves to the endpoint predicted by the previous decoder layer. The mode whose intention point is nearest to the GT endpoint is the positive: only it gets the regression loss, its mixture probability gets cross-entropy. A rule-based NMS picks 6 of 64.

## Formulation

static query = MLP(PE(intention point)); dynamic query for layer j+1 = MLP(PE(predicted endpoint of layer j)), gathers features from L = 128 nearest map polylines. Per layer a Gaussian mixture; positive = nearest intention point to the GT endpoint (hard); loss = NLL of the GT under the positive component + cross-entropy on its mixture probability + auxiliary L1 on a dense future head; equal weights. No separate classification head. Inference NMS: top 6 of 64 by endpoint distance, threshold 2.5 m (the authors note it may hurt minADE / minFDE). - input / output: scene tokens (agent + 768 map polylines, D = 256, 6 + 6 layers) -> 64 trajectories + probabilities.

## Training

AdamW, lr 1e-4, batch 80, 30 epochs on 8 GPUs, lr halved every 2 epochs from epoch 20, weight decay 0.01; regime `wta` with a fixed (anchor) assignment.

## Differentiable?

yes for heads; assignment and NMS constant.

## Sensors and data

Waymo Open Motion; vector map and agents (no sensors).

## Reported results

Waymo marginal test mAP 0.4129, minADE 0.6050, minFDE 1.2207, MR 0.1351. Ablation (validation subset): learnable latent queries mAP 0.2633 vs static intention queries 0.3059; uniform grid vs k-means intention points 0.3178 vs 0.3437; full model 0.3437. The paper does not say "mode collapse"; it says direct regression converges slowly and predicts the most frequent modes.

## Compute

about 59 ms (batch 1) / 33 ms per scene (batch 6) on an RTX 8000; parameters not given.

## Code

not checked

## Fit to our project

reuse_idea - effort S - "intention points" = a fixed candidate set in the tip frame (our teardrop ring samples are one); nearest-to-GT positive + classification over candidates trains the responsible child matching-free.

## Needs

a candidate set with fixed geometry in the tip frame; many more candidates than children and a final suppression.

## Conflicts

NMS is a non-differentiable inference step; 64 -> 6 contradicts a tiny fixed b unless the pool is the verified ray endpoints and b is taken after suppression.

## Tried in our experiments

E94 (partial, see MH10)
