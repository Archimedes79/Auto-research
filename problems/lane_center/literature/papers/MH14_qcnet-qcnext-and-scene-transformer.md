# MH14 - QCNet / QCNeXt and Scene Transformer (scene-level WTA, mixture of Laplace, detached proposals)

*Pass: loop 16 (2026-10-23), cluster `multihyp` (multiple hypotheses, set prediction, matching-free losses). Reading depth: **M+C (QCNeXt report, Scene Transformer, QCNet README); A for the QCNet paper itself** (A abstract, M method, C code page). Verified: partly (QCNet itself A). Pipeline stage: L, S4, S5.*

## Papers

Query-Centric Trajectory Prediction (Zhou et al., CVPR 2023) https://openaccess.thecvf.com/content/CVPR2023/html/Zhou_Query-Centric_Trajectory_Prediction_CVPR_2023_paper.html (A only); QCNeXt: A Next-Generation Framework For Joint Multi-Agent Trajectory Prediction (technical report) https://arxiv.org/abs/2306.10508 (read; describes QCNet briefly); Scene Transformer: A unified architecture for predicting multiple agent trajectories (Ngiam et al.) https://arxiv.org/abs/2106.08417 (read)

## Idea

Proposal module with anchor-free mode queries repeated recurrently, then an anchor-based refinement that takes the (detached) proposals as anchors; mixture of Laplace; winner = best mode at scene level (min displacement). Scene Transformer: F = 6 joint futures, each gets an ID embedding, a softmax over futures, loss = min over futures of the summed per-agent loss.

## Formulation

L = L_propose + L_refine + L_cls; L_cls = NLL of the mixing coefficients; winner by scene-level WTA (QCNeXt); proposals have gradients detached before refinement; 3 recurrent steps of 2 s each. Scene Transformer: Laplace-KL loss on position (Huber only on heading), minimum over futures (joint) or per agent (marginal), classifier on the best future weighted 0.1 vs regression 1.0; gradients flow only through the winning future; detaching of the winner choice not stated in either paper. - input / output: query-centric scene tokens -> K modes with Laplace parameters and mixing weights.

## Training

QCNeXt AdamW, 50 epochs, batch 32, cosine lr 5e-4 -> 0, dropout / weight decay 0.1, hidden 128; regime `wta`.

## Differentiable?

partial - winner constant; proposal -> refinement gradient cut by design.

## Sensors and data

Argoverse 1 / 2, Waymo; vectors only.

## Reported results

Argoverse 2 validation single-agent: QCNet minFDE6 1.28, MR6 0.16; QCNeXt 1.24 / 0.15; multi-agent leaderboard QCNeXt avgMinFDE6 1.13 (1.02 with 8-seed ensemble). Scene Transformer Argoverse test (marginal) minADE 0.80, minFDE 1.23, MR 0.13; WOMD marginal vehicle minADE 1.17, minFDE 2.48, MR 0.19. Neither page discusses mode collapse.

## Compute

QCNet README: example training on 8 x RTX 3090 with about 160 GB GPU memory in total (batch 4 per device); latency not reported.

## Code

https://github.com/ZikangZhou/QCNet (README: PyTorch + PyG + Lightning, Apache 2.0 stated; the README does not describe the loss or the number of modes)

## Fit to our project

reuse_idea - effort S - the "propose then refine, detach the proposal as the anchor of the next stage" pattern is the training form of our generate -> refine cascade.

## Needs

refinement that takes proposals as anchors; scene-level winner.

## Conflicts

scene-level (joint) WTA couples all agents; for a tree it would couple all tips (not wanted: each tip is independent).

## Tried in our experiments

E39 (cascade refiner, +0.001 F1)
