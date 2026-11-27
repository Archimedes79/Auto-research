# RB02 - ModalPatch: learned history-based feature prediction + uncertainty-gated fusion

*Pass: loop 17 (2026-10-24), file `B_failure_robustness.md`, cluster `failure_robustness`. Depth M here means: method sections queried through a page digest (see sources/lit17/B_failure_robustness.md, section 1).*

- papers: ModalPatch (plug-and-play module for LiDAR / camera drops in multi-modal 3D detectors; full paper title not read; arXiv 2603.02481, 3 March 2026) https://arxiv.org/abs/2603.02481 (opened as HTML https://arxiv.org/html/2603.02481)
- depth: M
- stage: S8, B
- idea: When a modality drops for a frame, a memory of the last 6 frames' features is attended by a learnable BEV query (deformable attention) to predict the missing feature; an MLP then predicts per-location variance of the predicted (compensated) feature, and that uncertainty down-weights it in cross-modal enhancement. Detector weights stay frozen.
- formulation: memory bank tau = 6 frames; stage 1 (12 epochs) trains the history predictor with feature-prediction loss + detection loss; stage 2 (12 epochs) freezes it and trains the uncertainty branch with a Gaussian NLL + fusion + detection loss; AdamW lr 2e-4. Ego-motion alignment of the bank is not stated in what I read.
- input: BEV features of the present modality, memory of past BEV features.
- output: compensated BEV feature + per-cell variance.
- training: matching-free (feature regression + task loss); drop rates 10 / 30 / 50 % simulated.
- differentiable: yes.
- sensors_data: LiDAR + camera, nuScenes val; BEVFusion, CMT, UniBEV, MEFormer as frozen hosts.
- results: average gain over hosts +4.70 mAP / +2.29 NDS at 10 % drop, +11.14 / +5.28 at 30 %, +11.93 / +5.05 at 50 % (paper text); CMT 27.21 -> 44.21 mAP at 50 %. Detection only.
- compute: average throughput 5.33 -> 4.90 FPS at 50 % drop (about 8 % slower); ms not stated.
- code: announced at github.com/Castiel-Lee/MM3Det_MD, "will be available" (not checked live).
- fit_to_us: reimplement - effort M - this is the learned version of our training-free spatial hold: replace "copy the warped previous input" by "predict the current feature from 6 warped past features, with a variance".
- compat_needs: ego-warped feature memory (we already have the input memory); a variance / NLL head.
- compat_conflicts: needs a per-frame "this input is missing" signal or gating (our label-free heat monitor could provide it).
- tried_before: training-free spatial hold of the previous input (verdict in loop 17 prior: partial); the learned variant not tried.
- verified: partly
