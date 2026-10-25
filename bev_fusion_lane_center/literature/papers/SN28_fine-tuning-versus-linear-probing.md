# SN28 - Fine-tuning versus linear probing under distribution shift (LP-FT)

*Pass: loop 16 (2026-10-23), cluster `sensors_bev` (camera-radar BEV foundations, sensors, data sets, robustness). Reading depth: **A** (A abstract, M method, C code page). Verified: partly (abstract only). Pipeline stage: T.*

## Papers

Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution (Kumar, Raghunathan, Jones, Ma, Liang; ICLR 2022) https://arxiv.org/abs/2202.10054

## Idea

Full fine-tuning gains in distribution but can lose out of distribution because the lower layers move while the head is still random; training the head first (linear probe) and then fine-tuning (LP-FT) keeps the pretrained features.

## Formulation

not read.

## Input

pretrained features.

## Output

n/a.

## Training

LP then FT; regime none (supervised).

## Differentiable?

yes.

## Sensors and data

image classification benchmarks (10 shift datasets).

## Reported results

abstract: fine-tuning about +2 % in-distribution and about -7 % out-of-distribution versus linear probing; LP-FT about +1 % ID and about +10 % OOD over full fine-tuning.

## Compute

n/a.

## Code

not checked.

## Fit to our project

reuse_idea - effort S - applies if we ever unfreeze the foundation's decoder: warm up our heads on the frozen features first, then unfreeze with a small LR; classification evidence only.

## Needs

a trained head before unfreezing.

## Conflicts

none.

## Tried in our experiments

none.
