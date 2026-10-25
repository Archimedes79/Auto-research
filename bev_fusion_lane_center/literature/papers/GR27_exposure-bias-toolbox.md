# GR27 - Exposure-bias toolbox (DAgger, scheduled sampling, DART)

*Pass: loop 16 (2026-10-23), cluster `growth` (iterative growth of road / lane graphs (agents, walkers, trackers)). Reading depth: **A** (A abstract, M method, C code page). Verified: . Pipeline stage: T.*

## Papers

A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning (Ross, Gordon, Bagnell, AISTATS 2011) https://arxiv.org/abs/1011.0686 ; Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks (Bengio, Vinyals, Jaitly, Shazeer, arXiv 2015) https://arxiv.org/abs/1506.03099 ; DART: Noise Injection for Robust Imitation Learning (Laskey et al., CoRL 2017) https://arxiv.org/abs/1703.09327

## Idea

Three generic remedies for the gap between teacher-forced training and free-running inference: DAgger trains a single policy on states visited by the learner (labels from an expert), scheduled sampling gradually replaces ground-truth previous inputs by the model's own outputs, DART adds noise to the executed demonstration to widen the state distribution without extra on-policy rollouts. Growers in this cluster use exactly these: RNGDet / CenterLineDet = DART-like noise, iCurb / RoadTracer = DAgger-like dynamic labels, VectorMapNet = second-stage retraining on predicted inputs.

## Formulation

not read beyond abstracts. results: DART reports up to 3 x faster than DAgger on Humanoid with only 5 % loss of supervisor reward during training (abstract).

## Fit to our project

reuse_idea - effort S; compat: all three need an expert / label function defined on arbitrary states (available in our case from the GT lane graph by nearest-point rules; see GR03, GR04).

## Tried in our experiments

none (E94 teacher-forced). verified: partly (abstract pages).
