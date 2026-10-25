# A03 - MapBench: temporal fusion under camera corruption (the one robustness measurement of temporal memory)

*Pass: loop 17 (2026-10-24), file `A_streaming_memory.md`, cluster `streaming_memory`. Depth M here means: method sections queried through a page digest (see sources/lit17/A_streaming_memory.md, section 1).*

- papers: Is Your HD Map Constructor Reliable under Sensor Corruptions? (Hao, Zhang et al., NeurIPS 2024 D&B) https://arxiv.org/abs/2406.12214
- depth: M (only the temporal ablation tables were queried; 4.7 k characters of the page were not read)
- stage: E, S8
- idea: Benchmark of 29 camera / LiDAR corruptions on nuScenes val (camera crash, frame lost, dark, motion blur, snow ...), 31 constructors. One ablation compares StreamMapNet with and without temporal fusion under the corruptions.
- formulation: metrics mCE / mRR (corruption error / resilience ratio relative to a baseline). Temporal fusion is the StreamMapNet BEV + query memory.
- input / output: images; vector maps.
- training: models retrained on a new split for this ablation (absolute numbers differ from Table 1).
- differentiable: n/a (benchmark).
- sensors_data: camera (also LiDAR and fusion models); nuScenes val.
- results: StreamMapNet clean mAP 23.8 -> 28.0 with temporal; mRR 47.1 -> 55.5, mCE 100 -> 85.9. Per-corruption mAP (no temporal -> temporal): camera crash 5.0 -> 10.3, frame lost 5.7 -> 12.1, dark 11.2 -> 14.4, motion blur 12.1 -> 17.6, snow 3.2 -> 2.0 (only corruption where temporal is worse). The text separately reports large drops for missing cameras in other models (e.g. about 40 % mAP loss for one model, about 69 % for another, from a search snippet; not verified in the paper).
- compute: n/a
- code: project page mapbench.github.io (named by search result, not opened)
- fit_to_us: reuse_idea (evaluation protocol: crash, frame lost, dark, blur map directly onto our night / blur / cameras-off set) - effort S.
- compat_needs: corruption generators; our synthetic failures already cover dark, blur, crash.
- compat_conflicts: none.
- tried_before: our own failure set (night, blur, three front cameras off).
- verified: partly
