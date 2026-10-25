# C01 - Online Navigation Refinement, MAT (SD road route -> perceived lane assignment)

*Pass: loop 17 (2026-10-24), file `C_map_matching_planner.md`, cluster `map_matching_planner`. Depth M here means: method sections queried through a page digest (see sources/lit17/C_map_matching_planner.md, section 1).*

- papers: Online Navigation Refinement: Achieving Lane-Level Guidance by Associating SD and Online Perception Maps (Wan et al., ICLR 2026) https://arxiv.org/abs/2507.07487 ; code https://github.com/WallelWan/OMA-MAT
- depth: M+C (README only)
- stage: S9 (also S6, E)
- idea: Turn a road-level route on an SD map into lane-level guidance by assigning every perceived centre-line to exactly one SD road (many lanes per road, one road per lane). The assignment is a plain softmax over roads of a dot product between a centre-line token and the mean-pooled road token; a topology-aware post-processing then picks a start token and runs a bidirectional beam search along road connectivity.
- formulation: vector = [p1x,p1y,p2x,p2y,theta], 2-layer MLP embedding for road / centre-line / boundary vectors. MAT block = Path-Aware attention (tokens re-ordered by path index, grouped attention, group size k, cost O(N) instead of O(N^2)) + Spatial attention (tokens quantised to (x,y,orientation), Hilbert-curve serialisation, grouped attention) + FFN. Prob_ij = softmax_j(F_cl_i . mean(F_road_j) / sqrt(d)). Loss = 1.0 CE + 0.01 CTC (order / sequence consistency). 50 epochs AdamW, lr 1e-4, wd 0.05, batch 128, cosine, 2 warm-up epochs. Variants MAT-T (tiny) / MAT-L. Parameter counts: not stated.
- input: SD map +-75 m (road vectors), perceived centre-line graph and road boundaries +-15 x +-30 m (training on GT, test on MapTR / MapTRv2 / SeqGrowGraph outputs).
- output: per centre-line vector a road id (scores); sequence of SD links per path.
- training: dense CE on manually annotated association labels (OMA dataset, ~30 k scenarios, 480 k road paths, 2.6 M lane vectors, nuScenes + OSM), trained from scratch; no Hungarian matching.
- differentiable: partial - the network and softmax are differentiable; the beam-search post-processing is a rule. It is trained stand-alone, not inside a perception network.
- sensors_data: any perceived map (vector); nuScenes Boston / Singapore, OSM roads; non-overlapping split.
- results: NR-F1(50:95) val: MAT-T 78.2 at 34 ms, MAT-L 78.7 at 70 ms; best classical map-matching baseline 72.9 at 345 ms (EAM3), FastMAC 70.3 at 62 ms (A6000). Test on perceived graphs: MAT-L 41.9 / 45.0 / 54.9 for MapTR / MapTRv2 / SeqGrowGraph (latency 74 ms). Robustness (val / test): global shift 10 / 20 / 50 % 77.5 / 76.0 / 74.6 (test 43.3 / 42.9 / 41.4); missing roads 10 / 12 / 30 % 77.5 / 76.4 / 75.5; element jitter 5 / 10 / 20 % 78.3 / 77.5 / 77.5. nuScenes-C zero-shot rain / fog / dark hard 40.6 / 42.1 / 41.6 (test-style numbers). 5 % of the training data still gives 77.1 val. Ablation: PTv3 baseline 61.8, +PA 74.1, +PA+SA 77.8, +boundary 78.5, +post-processing 78.7.
- compute: 34 ms (MAT-T) on an A6000 including all tokens of a scene; params not stated. This alone is as long as our whole 31 ms builder, so only the idea (not the module) fits.
- code: https://github.com/WallelWan/OMA-MAT (MIT, Pointcept-based per README; checkpoint and dataset released on HuggingFace per README news - not downloaded)
- fit_to_us: reuse_idea - effort S/M - our route head already does cross-attention graph -> SD tokens; the new parts are (a) a softmax over ROAD pools instead of per-vertex route flags, (b) path-ordered attention along edges, (c) a sequence / order loss.
- compat_needs: a centre-line graph with a path index (we have edges + successor chain); road tokens pooled per road segment.
- compat_conflicts: their ground truth is a manual association label; we derive route labels from the driven lane sequence (no road-id labels per lane yet). No wrong-road test, no abstain mechanism (the paper states no "wrong road" category; unmatched not handled).
- tried_before: route head by cross-attention (E104b, 0.853 - 0.820) kept; correlation volume (E103) dropped. Road-pool softmax and path-aware attention: none.
- verified: partly (extraction via summarising fetch; params and appendix tables 22 not seen)
