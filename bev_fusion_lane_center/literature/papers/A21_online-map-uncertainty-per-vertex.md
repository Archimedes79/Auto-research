# A21 - Online map uncertainty: per-vertex Laplace scale as a confidence output

*Pass: loop 17 (2026-10-24), file `A_streaming_memory.md`, cluster `streaming_memory`. Depth M here means: method sections queried through a page digest (see sources/lit17/A_streaming_memory.md, section 1).*

- papers: Producing and Leveraging Online Map Uncertainty in Trajectory Prediction (Gu, Song, Gilitschenski, Pavone, Ivanovic, CVPR 2024) https://arxiv.org/abs/2403.16439
- depth: M
- stage: S10, L
- idea: Each vertex is modelled by two univariate Laplace distributions; the regression head outputs (mu, b) per coordinate; training swaps L1 for the Laplace NLL sum[log(2b) + |v - mu| / b]. Vertex features for the predictor = [mu; b; class probs]. Uncertainty correlates with range, occlusion (StreamMapNet's memory lowers it in occluded regions), night for crossings, rain.
- formulation: Gaussian training was unstable; Laplace matches the existing L1 loss.
- input: online map head; output: map + per-vertex scale.
- training: NLL replaces L1; no explicit confidence weighting is applied downstream.
- differentiable: yes.
- sensors_data: camera, nuScenes.
- results: MR -15 % for HiVT + MapTR and HiVT + MapTRv2-Centerline; DenseTNT + StreamMapNet minADE 0.9492 -> 0.9036, MR 0.2569 -> 0.2359; mixed for DenseTNT + MapTRv2 (minADE 1.214 -> 1.262). Table inconsistencies noted in the digest.
- compute: not stated.
- code: not checked
- fit_to_us: reimplement - effort S - a log-scale output per vertex (cheap) gives the memory and the planner consumer a calibrated confidence; this is the planner interface of our E117 (edges + route probability).
- compat_needs: regression head with scale output.
- compat_conflicts: none.
- tried_before: keep gate 0.872 (E120) is a validity probability, not a position scale.
- verified: checked
