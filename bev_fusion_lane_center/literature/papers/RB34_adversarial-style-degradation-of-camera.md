# RB34 - Adversarial-style degradation of camera-radar fusion (blur lens vs radar ghosts)

*Pass: loop 17 (2026-10-24), file `B_failure_robustness.md`, cluster `failure_robustness`. Depth M here means: method sections queried through a page digest (see sources/lit17/B_failure_robustness.md, section 1).*

- papers: Adversarial Robustness of Camera-Radar Sensor Fusion for Autonomous Driving (Zhou et al., arXiv 2610.16012, 12 Oct 2026) https://arxiv.org/abs/2610.16012 (opened HTML, first 100 k of 117 k characters)
- depth: A (experiments table-level)
- stage: E
- idea: Eight camera perturbations (three blurs, four motion blurs, lens attack) and three radar perturbations (depth error, ghost targets, target disappearance) on RADIANT and CRN (detection) and radar-camera depth estimators.
- results: vertical motion blur kernel 20: CRN mAP / NDS 0.473 / 0.562 -> 0.4066 / 0.519; RADIANT 0.363 / 0.340 -> 0.290 / 0.301 (about 20 - 25 % drops); radar depth error 200 % barely changes RADIANT. Camera damage is much larger than radar damage; suggests reliability estimation, cross-modal consistency and failover but does not evaluate them.
- fit_to_us: baseline - effort S - supports prioritising blur / night over radar corruption in our failure mix.
- verified: partly
