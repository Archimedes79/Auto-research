# A18 - TopoFG, UniTopo, LaneSegNet: the non-temporal topology models that the streaming ones build on

*Pass: loop 17 (2026-10-24), file `A_streaming_memory.md`, cluster `streaming_memory`. Depth M here means: method sections queried through a page digest (see sources/lit17/A_streaming_memory.md, section 1).*

- papers: Fine-Grained Representation for Lane Topology Reasoning (TopoFG, 2025) https://arxiv.org/abs/2511.12590 ; Unified Modeling of Lane and Lane Topology for Driving Scene Reasoning (UniTopo, Li, Gao, Liu, Wang, Liu, Mu, 2026) https://arxiv.org/abs/2605.08911 ; LaneSegNet: Map Learning with Lane Segment Perception (ICLR 2024) https://arxiv.org/abs/2312.16108
- depth: A (TopoFG abstract page; UniTopo PDF text not extractable, only title / authors / code link https://github.com/homothetic/UniTopo seen; LaneSegNet from search listing + README)
- stage: S6, S4
- idea: TopoFG: hierarchical prior extractor (BEV mask -> global spatial prior, in-lane keypoint sequences -> local sequential prior), region-focused decoder (fine-grained queries from priors), boundary-point topology reasoning with topological denoising. UniTopo: models lane and lane topology jointly (predecessor / successor), built on TopoNet per the search summary. LaneSegNet: lane segment = centreline + boundaries in one query; base for TopoStreamer / FASTopoWM / Score.
- formulation: not read.
- input / output: BEV features -> lanes + topology.
- training: not read; Hungarian-matched DETR family.
- differentiable: partial (Hungarian).
- sensors_data: camera, OpenLane-V2.
- results: TopoFG OLS 48.0 (subset A), 45.4 (subset B) per the abstract page. UniTopo: search summary says +6.0 % TOP_lsls over LaneSegNet (not verified). LaneSegNet: +5.6 mAP lane segment per search listing (not verified). No temporal component was found for TopoFG; temporal use in UniTopo not determined.
- compute: LaneSegNet README: OpenMMLab stack mmcv-full 1.5.2, mmdet 2.26.0, mmdet3d 1.0.0rc6, Apache-2.0.
- code: https://github.com/OpenDriveLab/LaneSegNet ; UniTopo link above (not opened)
- fit_to_us: baseline - effort - - noted so the lane-topology family is covered; none of the three has a memory, so the "TopoLogic family temporal fusion" in the brief is carried by TopoStreamer, FASTopoWM and Score (A15, A01, A16).
- compat_needs: Hungarian set.
- compat_conflicts: matching (our losses forbid it).
- tried_before: E31 / E32 sparse matched decoder dropped; E42 ... E56 link heads dropped.
- verified: partly
