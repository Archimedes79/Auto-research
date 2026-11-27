# C23 - NavigScene (and Geo-VLA): navigation text / map knowledge fused to a driving model

*Pass: loop 17 (2026-10-24), file `C_map_matching_planner.md`, cluster `map_matching_planner`. Depth M here means: method sections queried through a page digest (see sources/lit17/C_map_matching_planner.md, section 1).*

- papers: NavigScene: Bridging Local Perception and Global Navigation for Beyond-Visual-Range Autonomous Driving (Peng et al., 2025) https://arxiv.org/abs/2507.05227 ; Geo-VLA (arXiv Aug 2026) https://arxiv.org/abs/2608.21440
- depth: A/M (fusion path only)
- stage: S9, S10
- idea: NavigScene generates textual navigation guidance (Google Maps static images -> GPT-4o descriptions) and fuses the frozen VLM output (compressed by an MLP, concatenated with BEV features) before the task heads. Geo-VLA does the opposite: map semantics are internalised into the VLM backbone by training on Geo-QA (3,000 QA pairs) and NO map is used at inference; inputs there are front image, ego history, state and a high-level command (no route).
- results: NavigScene nuScenes open loop, SparseDrive average L2 1.01 -> 0.76 m, collision 32.48 -> 20.71 (units as in the paper); VAD 1.22 -> 0.99 m. Navigation errors not analysed; unclear how the guidance is obtained at inference. Geo-VLA NAVSIM PDMS ReCogDrive 90.8 -> 91.2, DynVLA 91.0 -> 92.1; no latency.
- differentiable: partial (VLM frozen). code: none found.
- fit_to_us: not_suitable - text / VLM interface and heavy compute; only the observation that planners gain a little from route context is reusable.
- tried_before: none. verified: partly
