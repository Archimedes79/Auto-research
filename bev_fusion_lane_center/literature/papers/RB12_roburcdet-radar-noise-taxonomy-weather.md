# RB12 - RobuRCDet: radar noise taxonomy + weather-adaptive radar-camera fusion

*Pass: loop 17 (2026-10-24), file `B_failure_robustness.md`, cluster `failure_robustness`. Depth M here means: method sections queried through a page digest (see sources/lit17/B_failure_robustness.md, section 1).*

- papers: RobuRCDet: Enhancing Robustness of Radar-Camera Fusion in Bird's Eye View for 3D Object Detection (Yue et al., ICLR 2025) https://arxiv.org/abs/2502.13071 (abstract page opened); ICLR listing https://proceedings.iclr.cc/paper_files/paper/2025/hash/21dabaacda3edba8bb281da45d7cbc17-Abstract-Conference.html (seen in search)
- depth: A
- stage: B
- idea: Systematic noise analysis (five noise kinds: radar position, RCS, velocity and others not named on the page), a 3D Gaussian Expansion module that spreads each radar return with a deformable kernel / variance controlled by RCS and velocity priors, and a weather-adaptive fusion that weights radar vs camera by an estimated camera confidence map.
- formulation, results, compute: not read; abstract says "competitive in regular and noisy conditions".
- sensors_data: camera + radar, nuScenes.
- code: announced, not confirmed.
- fit_to_us: reuse_idea - effort M - radar position / RCS noise is a failure type we did not test (we only switch radar off); the corruption recipe could extend our failure set.
- tried_before: radar-off only.
- verified: partly
