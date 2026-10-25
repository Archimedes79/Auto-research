# C11 - SDTagNet: point-level SD-element tokens with text tags and element identifiers

*Pass: loop 17 (2026-10-24), file `C_map_matching_planner.md`, cluster `map_matching_planner`. Depth M here means: method sections queried through a page digest (see sources/lit17/C_map_matching_planner.md, section 1).*

- papers: SDTagNet: Leveraging Text-Annotated Navigation Maps for Online HD Map Construction (Immel et al., NeurIPS 2025) https://arxiv.org/abs/2506.08997 ; code https://github.com/immel-f/SDTagNet
- depth: M
- stage: S9
- idea: Every SD point is a query = sin/cos position + OSM-tag embedding (compact BERT, dim 144, contrastively pretrained) + orthogonal random-feature element identifiers that tie points to their polyline; resample polylines to 10 points; fuse with an extra cross-attention in the MapTRv2 decoder.
- results (mAP): Argoverse 2 far range (120 x 60 m) 13.0 -> 18.9 (+5.9); P-MapNet best SD baseline 15.7; near range 46.5 -> 48.1; nuScenes far 3.9 -> 8.0, near 22.9 -> 23.3. 42.12 M -> 50.54 M params, 16.0 -> 14.5 FPS (H100), < 7 ms extra.
- robustness: authors say position-noise and element-drop augmentation WORSENED performance in their setting; they note SD/HD discrepancies (missing tunnels / service roads) and wrong one-way directions (attributed to missing point order).
- differentiable: yes. fit_to_us: reuse_idea - effort S - element identifiers (all points of one road share a code) are a cheap way to let attention know which vertices belong to the same SD road; that is exactly the pooling unit that C01 uses. Caution: a counter-result to the "always augment with noise" rule.
- tried_before: none. verified: partly
