# C33 - HMM lane-level map matching with lane markings (non-differentiable reference)

*Pass: loop 17 (2026-10-24), file `C_map_matching_planner.md`, cluster `map_matching_planner`. Depth M here means: method sections queried through a page digest (see sources/lit17/C_map_matching_planner.md, section 1).*

- papers: Driving with Context: Online Map Matching for Complex Roads Using Lane Markings and Scenario Recognition (Bi et al., 2025) https://arxiv.org/abs/2505.05007 ; code https://github.com/TRV-Lab/LMSR-OMM
- depth: M
- stage: S9 (classical baseline)
- idea: Hidden state = road segment; transitions follow road connectivity (direct successors vs via intermediate roads); observation factors: GNSS + lane markings associated with OSM roads (association probability max over sampled points, Gaussian distance and heading terms) + a scenario factor (elevated road / tunnel via detections and fuzzy inference); Viterbi online decoding; ICP of lane-marking points with a penalty for solid vs dashed mismatch.
- results: Zenseact Open Dataset F1 98.04 % vs 96.13 %; Shanghai multilevel roads 94.60 % vs 88.75 %; ablation baseline 66.32 -> scenario only 92.74, lane markings only 79.22. Runtime not reported.
- differentiable: no.
- fit_to_us: baseline - effort S - the temporal HMM prior over "which road am I on" (transition by road connectivity) is the classical form of our temporal accumulation; its log-odds form could be fused with the learned route score.
- tried_before: temporal consistency in route head: partial. verified: partly
