# GR16 - TopoGPT (map-only pre-trained autoregressive lane prior plus flow-matching perception adapter)

*Pass: loop 16 (2026-10-23), cluster `growth` (iterative growth of road / lane graphs (agents, walkers, trackers)). Reading depth: **M+C** (A abstract, M method, C code page). Verified: checked (paper HTML, README).. Pipeline stage: S4, T, B.*

## Papers

Generative Lane Topology Reasoning via Autoregressive Model with Geometry Prior (Fu et al., ECCV 2026; arXiv 2606.31814) https://arxiv.org/abs/2606.31814

## Idea

Two phases. (1) Pre-train a decoder-only transformer on 3.3 M map scenes (no sensors) with next-token prediction over tokenised lane graphs, conditioned on a rasterised copy of the graph (scene tokens), augmented by flips / translations / rotations and by *condition degradation* (random lanes removed from the raster). (2) A perception adapter turns multi-view BEV features into scene-token-like features by deterministic flow matching (Euler steps; gradients flow through the unrolled steps), then the decoder is fine-tuned with LoRA.

## Formulation

each centerline = K sampled points fit by least squares with a cubic Bezier (4 control points, ends fixed); coordinates quantised to 64 bins per axis over +-52 m (x) and +-32 m (y); token = Q(y) * 64 + Q(x), 4 tokens per lane; lanes sorted by bounding box [x_min, y_min, x_max, y_max]; <SOS>, <SEP>, <EOS>; no connectivity tokens - topology by endpoint distance < 0.5 m at inference; lanes split only at merges and forks. Decoder: LlamaGen-like with M-RoPE; sizes 24.7 M / 45.9 M / 91.3 M. Loss: CE (teacher forcing) + flow MSE + alignment MSE (weights not stated); greedy decoding; 16 epochs pre-training, 24 epochs fine-tuning, 8 H20 GPUs.

## Input

BEV features from 6 - 7 cameras (ResNet-50 + Simple-BEV style lifting); `image_feat` / `map_raster` scene condition.

## Output

`token_sequence` of Bezier lanes.

## Training

`next_token` + `teacher_forcing`; pre-train / fine-tune; flow-matching alignment.

## Differentiable?

partial - adapter path yes (unrolled ODE steps); token generation no.

## Sensors and data

cameras; OpenLane-V2 subsets A and B; pre-training maps from large motion datasets (2.07 M + 1.01 M + 0.25 M scenes).

## Reported results

OpenLane-V2 subset A M-P / M-R / M-T 45.6 / 42.6 / 24.4, APLS 28.7 vs TopoPoint 42.7 / 34.0 / 16.6, 17.4; subset B 43.1 / 39.2 / 23.4, APLS 29.5 (Table 1). From scratch (no pre-training): 7.4 / 7.0 / 0.8; frozen / LoRA / full fine-tune 44.3 / 45.6 / 45.3 M-P (Tables 2 - 3). Flow steps 0 / 3 / 6 / 9: M-P 43.1 / 44.3 / 45.6 / 45.4.

## Compute

24.7 - 91.3 M parameters; latency not stated (sequential decoding acknowledged as a limit); early errors propagate.

## Code

https://github.com/buaa-colalab/TopoGPT

## Fit to our project

reuse_idea - effort L: a map-only geometric prior (condition degradation = train on partial graphs) as a rescoring prior for children; not the decoder.

## Needs

large map corpus (not in our environment; no downloads without asking); BEV features for the adapter.

## Conflicts

tokenisation; sequential decoding; no gradient to geometry.

## Tried in our experiments

none.
