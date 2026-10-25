# MH48 - Differentiable token pruning by a learned keep mask and attention masking (DynamicViT)

*Pass: loop 16 (2026-10-23), cluster `multihyp` (multiple hypotheses, set prediction, matching-free losses). Reading depth: **A** (A abstract, M method, C code page). Verified: partly. Pipeline stage: S4, T.*

## Papers

DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification (Rao et al., NeurIPS 2021) https://arxiv.org/abs/2106.02034 (abstract and search summaries only; ar5iv conversion failed); extension: Dynamic Spatial Sparsification for Efficient Vision Transformers and Convolutional Neural Networks https://arxiv.org/abs/2207.01580 (search summary only)

## Idea

A lightweight prediction module scores token importance at several layers and prunes hierarchically; training stays end to end through an attention masking strategy that blocks interactions of pruned tokens differentiably (shapes stay fixed in training; tokens are removed physically at inference).

## Formulation

not read. From the search summaries: losses cross-entropy + distillation toward the unpruned backbone; extension: ratio loss weight 2, KL and distillation weights 0.5, keep ratios geometric [rho, rho^2, rho^3]. A Gumbel-softmax sampled binary mask (from memory, not confirmed on the pages seen). - input / output: token sequence -> pruned token sequence.

## Training

cross-entropy + distillation + ratio loss; regime `none`.

## Differentiable?

partial (binary mask with a differentiable surrogate; from memory).

## Sensors and data

ImageNet.

## Reported results

about 31 - 37 % FLOPs reduction and over 40 % throughput gain with an accuracy drop within 0.5 % (poster page summary).

## Compute

as above.

## Code

https://github.com/raoyongming/DynamicViT (named in a search summary)

## Fit to our project

reuse_idea - effort M - the masking trick is the standard way to prune with fixed tensor shapes in training: dropped tree nodes stay in the tensor but are masked out of attention among siblings and from the loss.

## Needs

attention among nodes (our cascade stage C).

## Conflicts

needs a target keep ratio; details unread.

## Tried in our experiments

none
